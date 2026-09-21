from pathlib import Path
import time

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from casmi.retrieval.candidate_mass_index import (
    CandidateMassIndex,
)


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
    / "indexes"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VARIANT_PATH = (
    OUTPUT_DIR
    / "stage7_candidate_mass_variants.csv"
)

NPZ_PATH = (
    OUTPUT_DIR
    / "stage7_candidate_mass_index.npz"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage7_mass_index_summary.txt"
)


# =========================================================
# Helpers
# =========================================================

def exact_mass_from_smiles(
    smiles: str,
) -> float:

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
        "— STAGE 7 STEP 3 "
        "VARIANT-AWARE MASS INDEX"
    )

    start_time = time.time()

    # =====================================================
    # Read required metadata
    # =====================================================

    print(
        "\nLoading training structure metadata..."
    )

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
        ],
    )

    print(
        f"Training spectra: "
        f"{len(train):,}"
    )

    # =====================================================
    # Unique candidate representations
    # =====================================================

    variants = (
        train[
            [
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    print(
        f"Unique candidate representations: "
        f"{len(variants):,}"
    )

    # =====================================================
    # Exact masses
    # =====================================================

    print(
        "\nComputing variant exact masses..."
    )

    variants[
        "exact_mass"
    ] = (
        variants[
            "normalized_smiles"
        ]
        .map(
            exact_mass_from_smiles
        )
    )

    invalid_count = int(
        variants[
            "exact_mass"
        ]
        .isna()
        .sum()
    )

    print(
        f"Invalid exact masses: "
        f"{invalid_count:,}"
    )

    if invalid_count:

        raise RuntimeError(
            "Some mass variants could not be "
            "parsed by RDKit."
        )

    # =====================================================
    # Remove exact duplicate mass representations
    #
    # A structure may have distinct SMILES records that give
    # exactly the same mass. Keeping every representation
    # would not improve mass retrieval.
    # =====================================================

    before_mass_dedup = len(
        variants
    )

    variants = (
        variants
        .sort_values(
            [
                "inchikey14",
                "exact_mass",
                "normalized_smiles",
            ]
        )
        .drop_duplicates(
            subset=[
                "inchikey14",
                "exact_mass",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    removed_same_mass = (
        before_mass_dedup
        - len(
            variants
        )
    )

    print(
        f"Same-key duplicate masses removed: "
        f"{removed_same_mass:,}"
    )

    # =====================================================
    # Build in-memory index
    # =====================================================

    index = CandidateMassIndex(
        variants
    )

    print(
        f"Unique structures indexed: "
        f"{index.unique_structure_count:,}"
    )

    print(
        f"Mass variants indexed: "
        f"{index.variant_count:,}"
    )

    multi_variant_counts = (
        variants[
            "inchikey14"
        ]
        .value_counts()
    )

    multi_variant_structures = int(
        (
            multi_variant_counts
            > 1
        ).sum()
    )

    max_variants = int(
        multi_variant_counts.max()
    )

    print(
        f"Structures with >1 mass variant: "
        f"{multi_variant_structures:,}"
    )

    print(
        f"Maximum mass variants per structure: "
        f"{max_variants}"
    )

    # =====================================================
    # Sort by mass before persistence
    # =====================================================

    variants = (
        variants
        .sort_values(
            [
                "exact_mass",
                "inchikey14",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # =====================================================
    # Save human-readable variant table
    # =====================================================

    print(
        "\nSaving variant table..."
    )

    variants.to_csv(
        VARIANT_PATH,
        index=False,
    )

    # =====================================================
    # Save compact NumPy index
    # =====================================================

    print(
        "Saving NumPy mass index..."
    )

    np.savez_compressed(
        NPZ_PATH,
        exact_mass=(
            variants[
                "exact_mass"
            ]
            .to_numpy(
                dtype=np.float64,
            )
        ),
        inchikey14=(
            variants[
                "inchikey14"
            ]
            .astype(str)
            .to_numpy()
        ),
        normalized_smiles=(
            variants[
                "normalized_smiles"
            ]
            .astype(str)
            .to_numpy()
        ),
        molecular_formula=(
            variants[
                "molecular_formula"
            ]
            .fillna("")
            .astype(str)
            .to_numpy()
        ),
    )

    # =====================================================
    # Query sanity checks
    # =====================================================

    print(
        "\nRunning mass-query sanity checks..."
    )

    test_masses = [
        250.0,
        300.0,
        350.0,
        400.0,
        450.0,
    ]

    candidate_counts = []

    for mass in test_masses:

        hits = index.query(
            neutral_mass=mass,
            tolerance_ppm=10.0,
        )

        count = len(
            hits
        )

        candidate_counts.append(
            count
        )

        print(
            f"Mass {mass:7.2f} Da "
            f"| 10 ppm candidates: "
            f"{count:,}"
        )

    # =====================================================
    # Exact self-retrieval validation
    # =====================================================

    print(
        "\nChecking exact self retrieval..."
    )

    rng = np.random.default_rng(
        42
    )

    sample_size = min(
        10_000,
        len(
            variants
        ),
    )

    sample_indices = rng.choice(
        len(
            variants
        ),
        size=sample_size,
        replace=False,
    )

    self_retrieved = 0

    for row_index in sample_indices:

        row = variants.iloc[
            int(
                row_index
            )
        ]

        hits = index.query(
            neutral_mass=float(
                row[
                    "exact_mass"
                ]
            ),
            tolerance_ppm=10.0,
        )

        returned_keys = {
            hit.inchikey14
            for hit in hits
        }

        if (
            row[
                "inchikey14"
            ]
            in returned_keys
        ):

            self_retrieved += 1

    self_retrieval_rate = (
        self_retrieved
        / sample_size
    )

    print(
        f"Self retrieval: "
        f"{self_retrieved:,}"
        f"/"
        f"{sample_size:,}"
    )

    print(
        f"Self retrieval rate: "
        f"{self_retrieval_rate:.6%}"
    )

    if self_retrieval_rate < 1.0:

        raise RuntimeError(
            "Mass index failed exact self retrieval."
        )

    # =====================================================
    # Summary
    # =====================================================

    elapsed = (
        time.time()
        - start_time
    )

    summary_lines = [
        (
            "ENVEDA CASMI 2026 — "
            "STAGE 7 VARIANT-AWARE MASS INDEX"
        ),
        "",
        (
            f"Training spectra: "
            f"{len(train):,}"
        ),
        (
            "Unique candidate representations before "
            f"mass dedup: {before_mass_dedup:,}"
        ),
        (
            "Same-key duplicate masses removed: "
            f"{removed_same_mass:,}"
        ),
        (
            "Unique structures indexed: "
            f"{index.unique_structure_count:,}"
        ),
        (
            "Mass variants indexed: "
            f"{index.variant_count:,}"
        ),
        (
            "Structures with >1 mass variant: "
            f"{multi_variant_structures:,}"
        ),
        (
            "Maximum variants per structure: "
            f"{max_variants}"
        ),
        "",
        (
            f"Self retrieval: "
            f"{self_retrieved:,}/{sample_size:,}"
        ),
        (
            "Self retrieval rate: "
            f"{self_retrieval_rate:.6%}"
        ),
        "",
        "10 ppm example candidate counts:",
    ]

    for mass, count in zip(
        test_masses,
        candidate_counts,
    ):

        summary_lines.append(
            f"{mass:.2f} Da: "
            f"{count:,}"
        )

    summary_lines.extend(
        [
            "",
            (
                f"Elapsed seconds: "
                f"{elapsed:.2f}"
            ),
        ]
    )

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    # =====================================================
    # Final report
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 7 MASS INDEX SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Unique structures: "
        f"{index.unique_structure_count:,}"
    )

    print(
        f"Mass variants: "
        f"{index.variant_count:,}"
    )

    print(
        f"Multi-variant structures: "
        f"{multi_variant_structures:,}"
    )

    print(
        f"Self retrieval rate: "
        f"{self_retrieval_rate:.6%}"
    )

    print(
        f"Elapsed: "
        f"{elapsed:.2f} s"
    )

    print(
        "\nVariant table:"
    )

    print(
        VARIANT_PATH
    )

    print(
        "\nNumPy index:"
    )

    print(
        NPZ_PATH
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 7 STEP 3 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()