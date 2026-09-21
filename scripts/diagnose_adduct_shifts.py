from pathlib import Path
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

from rdkit import Chem
from rdkit.Chem import Descriptors

from casmi.chemistry.adducts import ADDUCTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = DATASET_DIR / "train.parquet"

MAX_ROWS = 250_000


def print_section(title: str) -> None:
    print("\n" + "=" * 105)
    print(title)
    print("=" * 105)


def main() -> None:

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 4 ADDUCT SHIFT DIAGNOSTIC"
    )

    columns = [
        "normalized_smiles",
        "adduct",
        "precursor_mz",
        "precursor_error_ppm",
    ]

    parquet_file = pq.ParquetFile(TRAIN_PATH)

    results = defaultdict(list)
    examples = defaultdict(list)

    rows_seen = 0

    for batch in parquet_file.iter_batches(
        columns=columns,
        batch_size=2048,
    ):
        df = batch.to_pandas()

        for row in df.itertuples(index=False):

            rows_seen += 1

            if row.adduct not in ADDUCTS:
                if rows_seen >= MAX_ROWS:
                    break
                continue

            mol = Chem.MolFromSmiles(
                row.normalized_smiles
            )

            if mol is None:
                if rows_seen >= MAX_ROWS:
                    break
                continue

            info = ADDUCTS[row.adduct]

            neutral_mass = Descriptors.ExactMolWt(mol)

            # Rearranged from:
            #
            # precursor_mz =
            # (multimer * neutral_mass + mass_shift)
            # / abs(charge)
            #
            empirical_shift = (
                float(row.precursor_mz)
                * abs(info.charge)
                - info.multimer * neutral_mass
            )

            result = {
                "empirical_shift": empirical_shift,
                "reported_ppm": row.precursor_error_ppm,
                "precursor_mz": row.precursor_mz,
                "neutral_mass": neutral_mass,
                "smiles": row.normalized_smiles,
            }

            results[row.adduct].append(result)

            if len(examples[row.adduct]) < 5:
                examples[row.adduct].append(result)

            if rows_seen >= MAX_ROWS:
                break

        if rows_seen >= MAX_ROWS:
            break

    print_section("EMPIRICAL ADDUCT MASS SHIFTS")

    print(
        f"{'Adduct':20s}"
        f"{'Rows':>10s}"
        f"{'Configured':>15s}"
        f"{'Median shift':>15s}"
        f"{'Difference':>15s}"
        f"{'Clean median':>15s}"
    )

    print("-" * 105)

    for adduct in sorted(results):

        info = ADDUCTS[adduct]

        entries = results[adduct]

        shifts = np.asarray(
            [
                x["empirical_shift"]
                for x in entries
            ],
            dtype=float,
        )

        clean_shifts = np.asarray(
            [
                x["empirical_shift"]
                for x in entries
                if (
                    x["reported_ppm"] is not None
                    and np.isfinite(x["reported_ppm"])
                    and abs(x["reported_ppm"]) <= 10
                )
            ],
            dtype=float,
        )

        median_shift = np.median(shifts)

        clean_median = (
            np.median(clean_shifts)
            if len(clean_shifts) > 0
            else np.nan
        )

        difference = (
            median_shift
            - info.mass_shift
        )

        print(
            f"{adduct:20s}"
            f"{len(entries):10,d}"
            f"{info.mass_shift:15.9f}"
            f"{median_shift:15.9f}"
            f"{difference:15.9f}"
            f"{clean_median:15.9f}"
        )

    print_section(
        "SPECIAL INSPECTION: [M+2H]2+"
    )

    special = results.get(
        "[M+2H]2+",
        [],
    )

    if not special:
        print("No [M+2H]2+ rows found.")
    else:
        for index, row in enumerate(
            special[:20],
            start=1,
        ):
            print(f"\nRow {index}")

            print(
                f"SMILES: {row['smiles']}"
            )

            print(
                f"Neutral RDKit mass: "
                f"{row['neutral_mass']:.6f}"
            )

            print(
                f"Precursor m/z: "
                f"{row['precursor_mz']:.6f}"
            )

            print(
                f"Empirical shift: "
                f"{row['empirical_shift']:.6f}"
            )

            print(
                f"Dataset precursor_error_ppm: "
                f"{row['reported_ppm']}"
            )

    print_section(
        "SPECIAL INSPECTION: [M+NH4]+ LARGE ERRORS"
    )

    nh4 = results.get(
        "[M+NH4]+",
        [],
    )

    if nh4:

        nh4_sorted = sorted(
            nh4,
            key=lambda x: (
                abs(x["reported_ppm"])
                if (
                    x["reported_ppm"] is not None
                    and np.isfinite(x["reported_ppm"])
                )
                else -1
            ),
            reverse=True,
        )

        for index, row in enumerate(
            nh4_sorted[:10],
            start=1,
        ):

            print(f"\nRow {index}")

            print(
                f"Neutral RDKit mass: "
                f"{row['neutral_mass']:.6f}"
            )

            print(
                f"Precursor m/z: "
                f"{row['precursor_mz']:.6f}"
            )

            print(
                f"Empirical shift: "
                f"{row['empirical_shift']:.6f}"
            )

            print(
                f"Dataset precursor_error_ppm: "
                f"{row['reported_ppm']}"
            )

            print(
                f"SMILES: "
                f"{row['smiles']}"
            )

    print_section(
        "DIAGNOSTIC COMPLETE"
    )


if __name__ == "__main__":
    main()