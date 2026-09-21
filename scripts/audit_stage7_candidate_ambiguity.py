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

DETAIL_OUTPUT_PATH = (
    OUTPUT_DIR
    / "stage7_candidate_ambiguity_audit.csv"
)

SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIR
    / "stage7_candidate_ambiguity_summary.txt"
)


# =========================================================
# Helpers
# =========================================================

def exact_mass_from_smiles(
    smiles,
):
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


def canonical_smiles_from_smiles(
    smiles,
):
    if not isinstance(
        smiles,
        str,
    ):
        return None

    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        return None

    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 7 STEP 2 "
        "CANDIDATE AMBIGUITY AUDIT"
    )

    start_time = time.time()

    # =====================================================
    # Load metadata
    # =====================================================

    print(
        "\nLoading training metadata..."
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
        f"Training rows: "
        f"{len(train):,}"
    )

    # =====================================================
    # Determine ambiguous keys
    # =====================================================

    grouped = train.groupby(
        "inchikey14",
        sort=False,
        observed=True,
    )

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

    ambiguous_smiles_keys = set(
        smiles_counts[
            smiles_counts > 1
        ].index
    )

    ambiguous_formula_keys = set(
        formula_counts[
            formula_counts > 1
        ].index
    )

    ambiguous_keys = (
        ambiguous_smiles_keys
        | ambiguous_formula_keys
    )

    print(
        f"Keys with >1 SMILES: "
        f"{len(ambiguous_smiles_keys):,}"
    )

    print(
        f"Keys with >1 formula: "
        f"{len(ambiguous_formula_keys):,}"
    )

    print(
        f"Total ambiguous keys: "
        f"{len(ambiguous_keys):,}"
    )

    # =====================================================
    # Work only on ambiguous rows
    # =====================================================

    subset = train[
        train[
            "inchikey14"
        ].isin(
            ambiguous_keys
        )
    ].copy()

    subset = (
        subset[
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
        f"Unique ambiguous representations: "
        f"{len(subset):,}"
    )

    # =====================================================
    # RDKit analysis
    # =====================================================

    print(
        "\nComputing canonical SMILES "
        "and exact masses..."
    )

    subset[
        "rdkit_canonical_smiles"
    ] = (
        subset[
            "normalized_smiles"
        ]
        .map(
            canonical_smiles_from_smiles
        )
    )

    subset[
        "rdkit_exact_mass"
    ] = (
        subset[
            "normalized_smiles"
        ]
        .map(
            exact_mass_from_smiles
        )
    )

    # =====================================================
    # Aggregate ambiguity statistics per key
    # =====================================================

    audit = (
        subset
        .groupby(
            "inchikey14",
            sort=False,
            observed=True,
        )
        .agg(
            normalized_smiles_count=(
                "normalized_smiles",
                "nunique",
            ),
            canonical_smiles_count=(
                "rdkit_canonical_smiles",
                "nunique",
            ),
            formula_count=(
                "molecular_formula",
                "nunique",
            ),
            exact_mass_count=(
                "rdkit_exact_mass",
                "nunique",
            ),
            exact_mass_min=(
                "rdkit_exact_mass",
                "min",
            ),
            exact_mass_max=(
                "rdkit_exact_mass",
                "max",
            ),
        )
        .reset_index()
    )

    audit[
        "exact_mass_range_da"
    ] = (
        audit[
            "exact_mass_max"
        ]
        - audit[
            "exact_mass_min"
        ]
    )

    audit[
        "same_exact_mass"
    ] = (
        audit[
            "exact_mass_range_da"
        ]
        <= 1e-8
    )

    audit[
        "has_formula_conflict"
    ] = (
        audit[
            "formula_count"
        ]
        > 1
    )

    audit[
        "has_mass_conflict"
    ] = (
        audit[
            "exact_mass_count"
        ]
        > 1
    )

    # =====================================================
    # Summary statistics
    # =====================================================

    same_mass_count = int(
        audit[
            "same_exact_mass"
        ].sum()
    )

    mass_conflict_count = int(
        audit[
            "has_mass_conflict"
        ].sum()
    )

    formula_conflict_count = int(
        audit[
            "has_formula_conflict"
        ].sum()
    )

    canonical_collapse_count = int(
        (
            audit[
                "canonical_smiles_count"
            ]
            < audit[
                "normalized_smiles_count"
            ]
        ).sum()
    )

    max_mass_range = float(
        audit[
            "exact_mass_range_da"
        ].max()
    )

    median_mass_range = float(
        audit.loc[
            audit[
                "has_mass_conflict"
            ],
            "exact_mass_range_da",
        ].median()
    ) if mass_conflict_count else 0.0

    # =====================================================
    # Save detailed audit
    # =====================================================

    audit = audit.sort_values(
        [
            "has_formula_conflict",
            "exact_mass_range_da",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    audit.to_csv(
        DETAIL_OUTPUT_PATH,
        index=False,
    )

    # =====================================================
    # Print worst conflicts
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "AMBIGUITY SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Ambiguous inchikey14 values: "
        f"{len(audit):,}"
    )

    print(
        f"Formula conflicts: "
        f"{formula_conflict_count:,}"
    )

    print(
        f"Exact-mass conflicts: "
        f"{mass_conflict_count:,}"
    )

    print(
        f"Same exact mass despite ambiguity: "
        f"{same_mass_count:,}"
    )

    print(
        f"Canonical-SMILES collapses: "
        f"{canonical_collapse_count:,}"
    )

    print(
        f"Median conflicting mass range: "
        f"{median_mass_range:.8f} Da"
    )

    print(
        f"Maximum mass range: "
        f"{max_mass_range:.8f} Da"
    )

    print(
        "\nTop 20 largest exact-mass conflicts:"
    )

    top_conflicts = (
        audit[
            audit[
                "has_mass_conflict"
            ]
        ]
        .head(
            20
        )
    )

    if len(
        top_conflicts
    ) == 0:

        print(
            "None"
        )

    else:

        print(
            top_conflicts[
                [
                    "inchikey14",
                    "normalized_smiles_count",
                    "canonical_smiles_count",
                    "formula_count",
                    "exact_mass_count",
                    "exact_mass_min",
                    "exact_mass_max",
                    "exact_mass_range_da",
                ]
            ].to_string(
                index=False
            )
        )

    # =====================================================
    # Save text summary
    # =====================================================

    elapsed = (
        time.time()
        - start_time
    )

    summary_lines = [
        (
            "ENVEDA CASMI 2026 — "
            "STAGE 7 CANDIDATE AMBIGUITY AUDIT"
        ),
        "",
        (
            f"Training rows: "
            f"{len(train):,}"
        ),
        (
            f"Keys with >1 normalized SMILES: "
            f"{len(ambiguous_smiles_keys):,}"
        ),
        (
            f"Keys with >1 molecular formula: "
            f"{len(ambiguous_formula_keys):,}"
        ),
        (
            f"Total ambiguous keys: "
            f"{len(ambiguous_keys):,}"
        ),
        "",
        (
            f"Formula conflicts: "
            f"{formula_conflict_count:,}"
        ),
        (
            f"Exact-mass conflicts: "
            f"{mass_conflict_count:,}"
        ),
        (
            "Same exact mass despite ambiguity: "
            f"{same_mass_count:,}"
        ),
        (
            "Canonical-SMILES collapses: "
            f"{canonical_collapse_count:,}"
        ),
        "",
        (
            "Median conflicting mass range (Da): "
            f"{median_mass_range:.8f}"
        ),
        (
            "Maximum conflicting mass range (Da): "
            f"{max_mass_range:.8f}"
        ),
        "",
        (
            f"Elapsed seconds: "
            f"{elapsed:.2f}"
        ),
    ]

    SUMMARY_OUTPUT_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print(
        "\nDetailed audit:"
    )

    print(
        DETAIL_OUTPUT_PATH
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        "\nElapsed: "
        f"{elapsed:.2f} s"
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 7 STEP 2 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()