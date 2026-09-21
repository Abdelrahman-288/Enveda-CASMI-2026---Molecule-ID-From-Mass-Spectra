from pathlib import Path
import time

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = (
    DATASET_DIR
    / "train.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "stage7_candidate_catalog.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage7_candidate_catalog_summary.txt"
)


# =========================================================
# Helpers
# =========================================================

def rdkit_exact_mass(
    smiles: str,
) -> float:
    """
    Compute RDKit molecular exact mass.

    Returns NaN if the SMILES cannot be parsed.
    """

    if not isinstance(
        smiles,
        str,
    ):
        return np.nan

    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        return np.nan

    return float(
        Descriptors.ExactMolWt(
            molecule
        )
    )


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 7 STEP 1 "
        "CANDIDATE CATALOG"
    )

    start_time = time.time()

    # =====================================================
    # Load only required columns
    # =====================================================

    columns = [
        "inchikey14",
        "normalized_smiles",
        "molecular_formula",
        "adduct",
        "ionization_mode",
    ]

    print(
        "\nLoading training metadata..."
    )

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=columns,
    )

    print(
        f"Training spectra: "
        f"{len(train):,}"
    )

    # =====================================================
    # Basic validation
    # =====================================================

    required_columns = set(
        columns
    )

    missing_columns = (
        required_columns
        - set(
            train.columns
        )
    )

    if missing_columns:

        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if train[
        "inchikey14"
    ].isna().any():

        raise RuntimeError(
            "Training data contains missing inchikey14."
        )

    # =====================================================
    # Aggregate one row per structure
    # =====================================================

    print(
        "Aggregating structures..."
    )

    grouped = (
        train
        .groupby(
            "inchikey14",
            sort=False,
            observed=True,
        )
    )

    catalog = (
        grouped
        .agg(
            normalized_smiles=(
                "normalized_smiles",
                "first",
            ),
            molecular_formula=(
                "molecular_formula",
                "first",
            ),
            spectrum_count=(
                "inchikey14",
                "size",
            ),
            adduct_count=(
                "adduct",
                "nunique",
            ),
            ionization_mode_count=(
                "ionization_mode",
                "nunique",
            ),
        )
        .reset_index()
    )

    print(
        f"Unique structures: "
        f"{len(catalog):,}"
    )

    # =====================================================
    # Internal consistency
    # =====================================================

    smiles_counts = (
        grouped[
            "normalized_smiles"
        ]
        .nunique(
            dropna=True
        )
    )

    formula_counts = (
        grouped[
            "molecular_formula"
        ]
        .nunique(
            dropna=True
        )
    )

    multiple_smiles = int(
        (
            smiles_counts
            > 1
        ).sum()
    )

    multiple_formulas = int(
        (
            formula_counts
            > 1
        ).sum()
    )

    print(
        f"InChIKey14 with >1 normalized SMILES: "
        f"{multiple_smiles:,}"
    )

    print(
        f"InChIKey14 with >1 molecular formula: "
        f"{multiple_formulas:,}"
    )

    # =====================================================
    # Exact molecular mass
    # =====================================================

    print(
        "\nComputing RDKit exact masses..."
    )

    catalog[
        "exact_mass"
    ] = (
        catalog[
            "normalized_smiles"
        ]
        .map(
            rdkit_exact_mass
        )
    )

    invalid_mass_count = int(
        catalog[
            "exact_mass"
        ]
        .isna()
        .sum()
    )

    print(
        f"Invalid/unparsed structures: "
        f"{invalid_mass_count:,}"
    )

    valid_mass = (
        catalog[
            "exact_mass"
        ]
        .dropna()
    )

    if len(
        valid_mass
    ) == 0:

        raise RuntimeError(
            "No valid exact masses were generated."
        )

    print(
        f"Exact mass minimum: "
        f"{valid_mass.min():.6f}"
    )

    print(
        f"Exact mass median:  "
        f"{valid_mass.median():.6f}"
    )

    print(
        f"Exact mass maximum: "
        f"{valid_mass.max():.6f}"
    )

    # =====================================================
    # Duplicate structure sanity
    # =====================================================

    duplicate_keys = int(
        catalog[
            "inchikey14"
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate catalog keys: "
        f"{duplicate_keys:,}"
    )

    if duplicate_keys != 0:

        raise RuntimeError(
            "Candidate catalog contains duplicate inchikey14."
        )

    # =====================================================
    # Ordering
    # =====================================================

    catalog = catalog[
        [
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
            "exact_mass",
            "spectrum_count",
            "adduct_count",
            "ionization_mode_count",
        ]
    ]

    catalog = catalog.sort_values(
        [
            "exact_mass",
            "inchikey14",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    # =====================================================
    # Save catalog
    # =====================================================

    print(
        "\nSaving candidate catalog..."
    )

    catalog.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # =====================================================
    # Summary
    # =====================================================

    elapsed = (
        time.time()
        - start_time
    )

    summary_lines = [
        "ENVEDA CASMI 2026 — STAGE 7 CANDIDATE CATALOG",
        "",
        f"Training spectra: {len(train):,}",
        f"Unique structures: {len(catalog):,}",
        f"Invalid exact masses: {invalid_mass_count:,}",
        f"Duplicate candidate keys: {duplicate_keys:,}",
        (
            "InChIKey14 with >1 normalized SMILES: "
            f"{multiple_smiles:,}"
        ),
        (
            "InChIKey14 with >1 molecular formula: "
            f"{multiple_formulas:,}"
        ),
        "",
        f"Exact mass min: {valid_mass.min():.6f}",
        f"Exact mass median: {valid_mass.median():.6f}",
        f"Exact mass max: {valid_mass.max():.6f}",
        "",
        f"Elapsed seconds: {elapsed:.2f}",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    # =====================================================
    # Final output
    # =====================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "STAGE 7 CANDIDATE CATALOG SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        f"Structures: "
        f"{len(catalog):,}"
    )

    print(
        f"Valid exact masses: "
        f"{len(valid_mass):,}"
    )

    print(
        f"Invalid exact masses: "
        f"{invalid_mass_count:,}"
    )

    print(
        f"Elapsed: "
        f"{elapsed:.2f} s"
    )

    print(
        "\nCatalog:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "STAGE 7 STEP 1 COMPLETE"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()