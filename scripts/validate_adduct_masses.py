from pathlib import Path
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

from rdkit import Chem
from rdkit.Chem import Descriptors

from casmi.chemistry.adducts import (
    ADDUCTS,
    precursor_to_neutral_mass,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = DATASET_DIR / "train.parquet"


MAX_ROWS = 100_000


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def ppm_error(
    observed: float,
    expected: float,
) -> float:
    return (
        (observed - expected)
        / expected
        * 1_000_000
    )


def main() -> None:

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 4 STEP 3 "
        "ADDUCT MASS VALIDATION"
    )

    columns = [
        "normalized_smiles",
        "adduct",
        "precursor_mz",
        "precursor_error_ppm",
        "molecular_formula",
    ]

    parquet_file = pq.ParquetFile(
        TRAIN_PATH
    )

    results = defaultdict(list)

    total_rows = 0
    supported_rows = 0
    valid_smiles = 0
    invalid_smiles = 0

    examples = []

    for batch in parquet_file.iter_batches(
        columns=columns,
        batch_size=2048,
    ):

        df = batch.to_pandas()

        for row in df.itertuples(index=False):

            total_rows += 1

            if row.adduct not in ADDUCTS:
                if total_rows >= MAX_ROWS:
                    break
                continue

            supported_rows += 1

            mol = Chem.MolFromSmiles(
                row.normalized_smiles
            )

            if mol is None:
                invalid_smiles += 1

                if total_rows >= MAX_ROWS:
                    break

                continue

            valid_smiles += 1

            rdkit_mass = (
                Descriptors.ExactMolWt(mol)
            )

            reconstructed_mass = (
                precursor_to_neutral_mass(
                    row.precursor_mz,
                    row.adduct,
                )
            )

            error_da = (
                reconstructed_mass
                - rdkit_mass
            )

            error_ppm = ppm_error(
                reconstructed_mass,
                rdkit_mass,
            )

            results[row.adduct].append(
                {
                    "error_da": error_da,
                    "error_ppm": error_ppm,
                    "reported_ppm": (
                        row.precursor_error_ppm
                    ),
                }
            )

            if len(examples) < 10:
                examples.append(
                    {
                        "smiles": (
                            row.normalized_smiles
                        ),
                        "formula": (
                            row.molecular_formula
                        ),
                        "adduct": row.adduct,
                        "precursor_mz": (
                            row.precursor_mz
                        ),
                        "rdkit_mass": rdkit_mass,
                        "reconstructed_mass": (
                            reconstructed_mass
                        ),
                        "error_da": error_da,
                        "error_ppm": error_ppm,
                        "reported_ppm": (
                            row.precursor_error_ppm
                        ),
                    }
                )

            if total_rows >= MAX_ROWS:
                break

        if total_rows >= MAX_ROWS:
            break

    print_section(
        "SUMMARY"
    )

    print(
        f"Rows inspected:          "
        f"{total_rows:,}"
    )

    print(
        f"Supported-adduct rows:   "
        f"{supported_rows:,}"
    )

    print(
        f"Valid RDKit structures:  "
        f"{valid_smiles:,}"
    )

    print(
        f"Invalid RDKit structures:"
        f" {invalid_smiles:,}"
    )

    print_section(
        "ERROR BY ADDUCT"
    )

    print(
        f"{'Adduct':20s}"
        f"{'Rows':>10s}"
        f"{'Median ppm':>15s}"
        f"{'Median |ppm|':>15s}"
        f"{'95% |ppm|':>15s}"
        f"{'Median Da':>15s}"
    )

    print("-" * 90)

    for adduct in sorted(
        results.keys()
    ):

        entries = results[adduct]

        ppm_values = np.array(
            [
                item["error_ppm"]
                for item in entries
            ],
            dtype=float,
        )

        da_values = np.array(
            [
                item["error_da"]
                for item in entries
            ],
            dtype=float,
        )

        median_ppm = np.median(
            ppm_values
        )

        median_abs_ppm = np.median(
            np.abs(ppm_values)
        )

        p95_abs_ppm = np.percentile(
            np.abs(ppm_values),
            95,
        )

        median_da = np.median(
            da_values
        )

        print(
            f"{adduct:20s}"
            f"{len(entries):10,d}"
            f"{median_ppm:15.4f}"
            f"{median_abs_ppm:15.4f}"
            f"{p95_abs_ppm:15.4f}"
            f"{median_da:15.6f}"
        )

    print_section(
        "REPRESENTATIVE EXAMPLES"
    )

    for index, example in enumerate(
        examples,
        start=1,
    ):

        print(
            f"\nExample {index}"
        )

        print(
            f"SMILES: "
            f"{example['smiles']}"
        )

        print(
            f"Formula: "
            f"{example['formula']}"
        )

        print(
            f"Adduct: "
            f"{example['adduct']}"
        )

        print(
            f"Precursor m/z: "
            f"{example['precursor_mz']:.6f}"
        )

        print(
            f"RDKit exact mass: "
            f"{example['rdkit_mass']:.6f}"
        )

        print(
            f"Reconstructed mass: "
            f"{example['reconstructed_mass']:.6f}"
        )

        print(
            f"Error Da: "
            f"{example['error_da']:.6f}"
        )

        print(
            f"Calculated error ppm: "
            f"{example['error_ppm']:.4f}"
        )

        print(
            f"Dataset precursor_error_ppm: "
            f"{example['reported_ppm']}"
        )

    print_section(
        "STAGE 4 STEP 3 COMPLETE"
    )


if __name__ == "__main__":
    main()