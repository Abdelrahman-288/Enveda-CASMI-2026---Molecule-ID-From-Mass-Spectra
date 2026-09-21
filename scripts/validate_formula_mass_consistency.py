from pathlib import Path
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from casmi.chemistry.formula import (
    exact_mass_from_formula,
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
        "— STAGE 4 STEP 5 "
        "FORMULA / MASS CONSISTENCY"
    )

    columns = [
        "normalized_smiles",
        "molecular_formula",
        "inchikey14",
    ]

    parquet_file = pq.ParquetFile(
        TRAIN_PATH
    )

    total = 0
    valid_smiles = 0
    invalid_smiles = 0

    formula_match = 0
    formula_mismatch = 0

    formula_mass_errors = []

    mismatch_examples = []

    unsupported_elements = Counter()

    for batch in parquet_file.iter_batches(
        columns=columns,
        batch_size=2048,
    ):

        df = batch.to_pandas()

        for row in df.itertuples(index=False):

            total += 1

            mol = Chem.MolFromSmiles(
                row.normalized_smiles
            )

            if mol is None:
                invalid_smiles += 1

                if total >= MAX_ROWS:
                    break

                continue

            valid_smiles += 1

            rdkit_formula = CalcMolFormula(
                mol
            )

            if rdkit_formula == row.molecular_formula:
                formula_match += 1
            else:
                formula_mismatch += 1

                if len(mismatch_examples) < 15:
                    mismatch_examples.append(
                        {
                            "smiles": row.normalized_smiles,
                            "dataset_formula": row.molecular_formula,
                            "rdkit_formula": rdkit_formula,
                            "inchikey14": row.inchikey14,
                        }
                    )

            rdkit_mass = float(
                Descriptors.ExactMolWt(mol)
            )

            try:
                formula_mass = (
                    exact_mass_from_formula(
                        row.molecular_formula
                    )
                )

                error_ppm = ppm_error(
                    formula_mass,
                    rdkit_mass,
                )

                formula_mass_errors.append(
                    error_ppm
                )

            except KeyError as exc:

                text = str(exc)

                unsupported_elements[text] += 1

            if total >= MAX_ROWS:
                break

        if total >= MAX_ROWS:
            break

    print_section("SUMMARY")

    print(
        f"Rows inspected:       {total:,}"
    )

    print(
        f"Valid SMILES:         {valid_smiles:,}"
    )

    print(
        f"Invalid SMILES:       {invalid_smiles:,}"
    )

    print(
        f"Formula matches:      {formula_match:,}"
    )

    print(
        f"Formula mismatches:   {formula_mismatch:,}"
    )

    if valid_smiles > 0:

        match_pct = (
            formula_match
            / valid_smiles
            * 100
        )

        print(
            f"Formula match rate:   "
            f"{match_pct:.4f}%"
        )

    print_section(
        "FORMULA MASS VS RDKIT MASS"
    )

    if formula_mass_errors:

        errors = np.asarray(
            formula_mass_errors,
            dtype=float,
        )

        print(
            f"Comparisons:            "
            f"{len(errors):,}"
        )

        print(
            f"Median ppm:             "
            f"{np.median(errors):.6f}"
        )

        print(
            f"Median |ppm|:           "
            f"{np.median(np.abs(errors)):.6f}"
        )

        print(
            f"95th percentile |ppm|: "
            f"{np.percentile(np.abs(errors), 95):.6f}"
        )

        print(
            f"Maximum |ppm|:          "
            f"{np.max(np.abs(errors)):.6f}"
        )

    print_section(
        "FORMULA MISMATCH EXAMPLES"
    )

    if not mismatch_examples:
        print(
            "No formula mismatches found "
            "in inspected rows."
        )

    else:
        for i, example in enumerate(
            mismatch_examples,
            start=1,
        ):

            print(f"\nExample {i}")

            print(
                f"InChIKey14: "
                f"{example['inchikey14']}"
            )

            print(
                f"Dataset formula: "
                f"{example['dataset_formula']}"
            )

            print(
                f"RDKit formula:   "
                f"{example['rdkit_formula']}"
            )

            print(
                f"SMILES: "
                f"{example['smiles']}"
            )

    print_section(
        "UNSUPPORTED FORMULA ELEMENTS"
    )

    if not unsupported_elements:
        print(
            "No unsupported elements found."
        )
    else:
        for item, count in (
            unsupported_elements.most_common()
        ):
            print(
                f"{item}: {count:,}"
            )

    print_section(
        "STAGE 4 STEP 5 COMPLETE"
    )


if __name__ == "__main__":
    main()