from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = (
    DATASET_DIR
    / "train.parquet"
)

STRUCTURE_SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "structure_validation_split.csv"
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
    / "stage6_training_manifest.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage6_training_manifest_summary.txt"
)


# =========================================================
# Test-domain definition
# =========================================================

TEST_MIN_MZ = 245.0931
TEST_MAX_MZ = 460.1554

TEST_ADDUCTS = {
    "[M+H]+",
    "[M-H]-",
    "[M+CH2O2-H]-",
    "[M+Na]+",
    "[M+NH4]+",
    "[M+K]+",
    "[M+Cl]-",
}


# =========================================================
# Stage 6 configuration
# =========================================================

RANDOM_SEED = 42

# Avoid molecules with hundreds/thousands of spectra
# dominating training.
MAX_SPECTRA_PER_STRUCTURE = 12

# Keep validation evaluation manageable initially.
MAX_VALIDATION_SPECTRA_PER_STRUCTURE = 6


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 6 STEP 1 "
        "NEURAL TRAINING MANIFEST"
    )

    # =====================================================
    # Load structure-disjoint assignment
    # =====================================================

    print(
        "\nLoading structure-disjoint split..."
    )

    structure_split = pd.read_csv(
        STRUCTURE_SPLIT_PATH
    )

    required_columns = {
        "inchikey14",
        "split",
    }

    missing_columns = (
        required_columns
        - set(
            structure_split.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Structure split is missing columns: "
            f"{sorted(missing_columns)}"
        )

    split_map = dict(
        zip(
            structure_split[
                "inchikey14"
            ],
            structure_split[
                "split"
            ],
        )
    )

    print(
        f"Structure assignments loaded: "
        f"{len(split_map):,}"
    )

    # =====================================================
    # Load lightweight spectrum metadata
    # =====================================================

    print(
        "Loading training metadata..."
    )

    columns = [
        "inchikey14",
        "normalized_smiles",
        "molecular_formula",
        "ingest_lib",
        "instrument_type",
        "adduct",
        "precursor_mz",
        "ionization_mode",
        "collision_energy_ev",
        "num_peaks",
    ]

    df = pq.read_table(
        TRAIN_PATH,
        columns=columns,
    ).to_pandas()

    df["row_index"] = np.arange(
        len(df),
        dtype=np.int64,
    )

    print(
        f"Spectra loaded: "
        f"{len(df):,}"
    )

    # =====================================================
    # Attach structure split
    # =====================================================

    df["split"] = (
        df[
            "inchikey14"
        ].map(
            split_map
        )
    )

    missing_split = (
        df["split"]
        .isna()
        .sum()
    )

    if missing_split:
        raise RuntimeError(
            f"{missing_split:,} spectra have no "
            "structure split assignment."
        )

    # =====================================================
    # Test-like flag
    # =====================================================

    df["is_test_like"] = (
        df[
            "instrument_type"
        ].eq(
            "timsTOF"
        )
        &
        df[
            "precursor_mz"
        ].between(
            TEST_MIN_MZ,
            TEST_MAX_MZ,
        )
        &
        df[
            "adduct"
        ].isin(
            TEST_ADDUCTS
        )
    )

    # =====================================================
    # Deterministic random key
    # =====================================================

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    df["_random"] = rng.random(
        len(df)
    )

    # =====================================================
    # Training sampling
    # =====================================================

    print(
        "Sampling training spectra..."
    )

    train_df = df[
        df["split"]
        == "train"
    ].copy()

    # Test-like spectra are deliberately preferred
    # because the hidden test domain is timsTOF and
    # occupies a narrow precursor/adduct distribution.
    train_df = train_df.sort_values(
        [
            "inchikey14",
            "is_test_like",
            "_random",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="mergesort",
    )

    train_df[
        "_structure_rank"
    ] = (
        train_df.groupby(
            "inchikey14"
        )
        .cumcount()
        + 1
    )

    train_selected = train_df[
        train_df[
            "_structure_rank"
        ]
        <= MAX_SPECTRA_PER_STRUCTURE
    ].copy()

    # =====================================================
    # Validation sampling
    # =====================================================

    print(
        "Sampling validation spectra..."
    )

    validation_df = df[
        df["split"]
        == "validation"
    ].copy()

    validation_df = validation_df.sort_values(
        [
            "inchikey14",
            "is_test_like",
            "_random",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="mergesort",
    )

    validation_df[
        "_structure_rank"
    ] = (
        validation_df.groupby(
            "inchikey14"
        )
        .cumcount()
        + 1
    )

    validation_selected = (
        validation_df[
            validation_df[
                "_structure_rank"
            ]
            <= MAX_VALIDATION_SPECTRA_PER_STRUCTURE
        ]
        .copy()
    )

    # =====================================================
    # Combine
    # =====================================================

    manifest = pd.concat(
        [
            train_selected,
            validation_selected,
        ],
        ignore_index=True,
    )

    # =====================================================
    # Leakage checks
    # =====================================================

    train_structures = set(
        manifest.loc[
            manifest["split"]
            == "train",
            "inchikey14",
        ]
    )

    validation_structures = set(
        manifest.loc[
            manifest["split"]
            == "validation",
            "inchikey14",
        ]
    )

    overlap = (
        train_structures
        & validation_structures
    )

    if overlap:
        raise RuntimeError(
            "Structure leakage detected between "
            "training and validation."
        )

    # =====================================================
    # Structure spectrum counts
    # =====================================================

    train_counts = (
        train_selected.groupby(
            "inchikey14"
        )
        .size()
    )

    validation_counts = (
        validation_selected.groupby(
            "inchikey14"
        )
        .size()
    )

    # =====================================================
    # Summary
    # =====================================================

    summary = f"""
ENVEDA CASMI 2026
Stage 6 Neural Training Manifest

Random seed:
{RANDOM_SEED}

Maximum training spectra per structure:
{MAX_SPECTRA_PER_STRUCTURE}

Maximum validation spectra per structure:
{MAX_VALIDATION_SPECTRA_PER_STRUCTURE}

Training structures:
{len(train_structures):,}

Training spectra:
{len(train_selected):,}

Training test-like spectra:
{train_selected['is_test_like'].sum():,}

Validation structures:
{len(validation_structures):,}

Validation spectra:
{len(validation_selected):,}

Validation test-like spectra:
{validation_selected['is_test_like'].sum():,}

Structure overlap:
{len(overlap):,}

Mean training spectra per structure:
{train_counts.mean():.4f}

Median training spectra per structure:
{train_counts.median():.4f}

Maximum training spectra per structure:
{train_counts.max():,}

Mean validation spectra per structure:
{validation_counts.mean():.4f}

Median validation spectra per structure:
{validation_counts.median():.4f}

Maximum validation spectra per structure:
{validation_counts.max():,}
""".strip()

    print_section(
        "STAGE 6 MANIFEST SUMMARY"
    )

    print(
        summary
    )

    # =====================================================
    # Save
    # =====================================================

    output_columns = [
        "row_index",
        "inchikey14",
        "normalized_smiles",
        "molecular_formula",
        "ingest_lib",
        "instrument_type",
        "adduct",
        "precursor_mz",
        "ionization_mode",
        "collision_energy_ev",
        "num_peaks",
        "is_test_like",
        "split",
    ]

    manifest[
        output_columns
    ].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        SUMMARY_PATH
    )

    print_section(
        "STAGE 6 STEP 1 COMPLETE"
    )


if __name__ == "__main__":
    main()