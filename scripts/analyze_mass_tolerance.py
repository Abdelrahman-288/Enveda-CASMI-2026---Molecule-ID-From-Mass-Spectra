from pathlib import Path
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

from rdkit import Chem
from rdkit.Chem import Descriptors

from casmi.chemistry.adducts import (
    is_trusted_for_mass_filter,
    precursor_to_neutral_mass,
    mass_error_ppm,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = DATASET_DIR / "train.parquet"

MAX_ROWS = 250_000

TOLERANCES = [
    2,
    5,
    10,
    20,
    50,
    100,
]


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 4 MASS TOLERANCE ANALYSIS"
    )

    columns = [
        "normalized_smiles",
        "adduct",
        "precursor_mz",
        "precursor_error_ppm",
        "instrument_type",
        "ingest_lib",
    ]

    parquet_file = pq.ParquetFile(
        TRAIN_PATH
    )

    errors_all = []
    errors_by_adduct = defaultdict(list)

    total_rows = 0
    trusted_rows = 0
    valid_rows = 0
    invalid_smiles = 0
    skipped_untrusted = 0

    for batch in parquet_file.iter_batches(
        columns=columns,
        batch_size=2048,
    ):
        df = batch.to_pandas()

        for row in df.itertuples(index=False):

            total_rows += 1

            if not is_trusted_for_mass_filter(
                row.adduct
            ):
                skipped_untrusted += 1

                if total_rows >= MAX_ROWS:
                    break

                continue

            trusted_rows += 1

            mol = Chem.MolFromSmiles(
                row.normalized_smiles
            )

            if mol is None:
                invalid_smiles += 1

                if total_rows >= MAX_ROWS:
                    break

                continue

            exact_mass = float(
                Descriptors.ExactMolWt(mol)
            )

            try:
                reconstructed_mass = (
                    precursor_to_neutral_mass(
                        row.precursor_mz,
                        row.adduct,
                        require_trusted=True,
                    )
                )
            except (KeyError, ValueError):

                if total_rows >= MAX_ROWS:
                    break

                continue

            error = mass_error_ppm(
                reconstructed_mass,
                exact_mass,
            )

            errors_all.append(
                abs(error)
            )

            errors_by_adduct[
                row.adduct
            ].append(
                abs(error)
            )

            valid_rows += 1

            if total_rows >= MAX_ROWS:
                break

        if total_rows >= MAX_ROWS:
            break

    print_section("SUMMARY")

    print(
        f"Rows inspected:       {total_rows:,}"
    )

    print(
        f"Trusted-adduct rows:  {trusted_rows:,}"
    )

    print(
        f"Valid comparisons:    {valid_rows:,}"
    )

    print(
        f"Invalid SMILES:       {invalid_smiles:,}"
    )

    print(
        f"Skipped untrusted:    {skipped_untrusted:,}"
    )

    errors = np.asarray(
        errors_all,
        dtype=float,
    )

    print_section(
        "OVERALL ABSOLUTE MASS ERROR"
    )

    print(
        f"Median |ppm|: "
        f"{np.median(errors):.4f}"
    )

    print(
        f"90th percentile |ppm|: "
        f"{np.percentile(errors, 90):.4f}"
    )

    print(
        f"95th percentile |ppm|: "
        f"{np.percentile(errors, 95):.4f}"
    )

    print(
        f"99th percentile |ppm|: "
        f"{np.percentile(errors, 99):.4f}"
    )

    print(
        f"Maximum |ppm|: "
        f"{np.max(errors):.4f}"
    )

    print_section(
        "CORRECT-STRUCTURE SURVIVAL BY TOLERANCE"
    )

    print(
        f"{'Tolerance':>12s}"
        f"{'Survive':>14s}"
        f"{'Rejected':>14s}"
        f"{'Survival %':>15s}"
    )

    print("-" * 60)

    for tolerance in TOLERANCES:

        survive = int(
            np.sum(
                errors <= tolerance
            )
        )

        rejected = (
            len(errors)
            - survive
        )

        survival_pct = (
            survive
            / len(errors)
            * 100
        )

        print(
            f"{tolerance:10.0f} ppm"
            f"{survive:14,d}"
            f"{rejected:14,d}"
            f"{survival_pct:14.4f}%"
        )

    print_section(
        "ERROR DISTRIBUTION BY ADDUCT"
    )

    print(
        f"{'Adduct':20s}"
        f"{'Rows':>10s}"
        f"{'Median':>12s}"
        f"{'95%':>12s}"
        f"{'99%':>12s}"
    )

    print("-" * 70)

    for adduct in sorted(
        errors_by_adduct
    ):

        values = np.asarray(
            errors_by_adduct[adduct],
            dtype=float,
        )

        print(
            f"{adduct:20s}"
            f"{len(values):10,d}"
            f"{np.median(values):12.3f}"
            f"{np.percentile(values, 95):12.3f}"
            f"{np.percentile(values, 99):12.3f}"
        )

    print_section(
        "SURVIVAL BY ADDUCT AT 10 PPM"
    )

    for adduct in sorted(
        errors_by_adduct
    ):

        values = np.asarray(
            errors_by_adduct[adduct],
            dtype=float,
        )

        survive = np.sum(
            values <= 10.0
        )

        percentage = (
            survive
            / len(values)
            * 100
        )

        print(
            f"{adduct:20s} "
            f"{int(survive):8,d}"
            f" / {len(values):8,d} "
            f"({percentage:7.3f}%)"
        )

    print_section(
        "MASS TOLERANCE ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()