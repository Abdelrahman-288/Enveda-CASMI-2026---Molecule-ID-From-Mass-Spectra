from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = (
    DATASET_DIR
    / "train.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "multispectrum_validation_split.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "multispectrum_validation_split_summary.txt"
)


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

MAX_QUERY_SPECTRA_PER_STRUCTURE = 3

HASH_KEY = "casmi2026_multispectrum_v1"


def print_section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 5 STEP 9B "
        "MULTI-SPECTRUM VALIDATION SPLIT"
    )

    columns = [
        "inchikey14",
        "normalized_smiles",
        "instrument_type",
        "adduct",
        "precursor_mz",
        "ionization_mode",
        "ingest_lib",
    ]

    print(
        "\nLoading metadata..."
    )

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

    print(
        f"Structures: "
        f"{df['inchikey14'].nunique():,}"
    )

    # =====================================================
    # Test-like flag
    # =====================================================

    df["is_test_like"] = (
        df["instrument_type"].eq(
            "timsTOF"
        )
        &
        df["precursor_mz"].between(
            TEST_MIN_MZ,
            TEST_MAX_MZ,
        )
        &
        df["adduct"].isin(
            TEST_ADDUCTS
        )
    )

    # =====================================================
    # Counts
    # =====================================================

    structure_count = (
        df.groupby(
            "inchikey14"
        )["row_index"]
        .transform("size")
    )

    test_like_count = (
        df["is_test_like"]
        .groupby(
            df["inchikey14"]
        )
        .transform("sum")
    )

    df["structure_count"] = (
        structure_count
    )

    df["test_like_count"] = (
        test_like_count
    )

    # Need:
    #
    # >= 3 spectra total
    # >= 2 test-like spectra
    #
    # so we can hold out multiple query spectra while
    # leaving at least one reference spectrum.
    eligible = (
        (df["structure_count"] >= 3)
        &
        (df["test_like_count"] >= 2)
    )

    eligible_structures = set(
        df.loc[
            eligible,
            "inchikey14",
        ].unique()
    )

    print(
        "Eligible structures: "
        f"{len(eligible_structures):,}"
    )

    # =====================================================
    # Deterministic ordering
    # =====================================================

    eligible_df = df[
        df["inchikey14"].isin(
            eligible_structures
        )
    ].copy()

    hash_frame = eligible_df[
        [
            "inchikey14",
            "row_index",
        ]
    ].copy()

    hash_frame["seed"] = (
        HASH_KEY
    )

    eligible_df["_hash"] = (
        pd.util.hash_pandas_object(
            hash_frame,
            index=False,
        )
        .astype("uint64")
    )

    # =====================================================
    # Role assignment
    # =====================================================

    eligible_df["multispectrum_role"] = (
        "reference"
    )

    test_like = eligible_df[
        eligible_df["is_test_like"]
    ].copy()

    test_like = test_like.sort_values(
        [
            "inchikey14",
            "_hash",
        ],
        kind="mergesort",
    )

    # Rank test-like spectra within each molecule.
    test_like["_query_rank"] = (
        test_like
        .groupby(
            "inchikey14"
        )
        .cumcount()
        + 1
    )

    # Never consume every spectrum of a structure.
    #
    # allowed query count =
    # min(
    #   max requested,
    #   test-like count,
    #   total spectra - 1
    # )
    total_map = (
        eligible_df.groupby(
            "inchikey14"
        )["row_index"]
        .size()
    )

    test_like_map = (
        test_like.groupby(
            "inchikey14"
        )["row_index"]
        .size()
    )

    allowed_query_count = {}

    for structure in (
        eligible_structures
    ):

        total = int(
            total_map[
                structure
            ]
        )

        n_test_like = int(
            test_like_map.get(
                structure,
                0,
            )
        )

        allowed_query_count[
            structure
        ] = min(
            MAX_QUERY_SPECTRA_PER_STRUCTURE,
            n_test_like,
            total - 1,
        )

    test_like[
        "_allowed_query_count"
    ] = (
        test_like[
            "inchikey14"
        ].map(
            allowed_query_count
        )
    )

    selected_queries = test_like[
        test_like["_query_rank"]
        <= test_like[
            "_allowed_query_count"
        ]
    ]

    query_row_ids = set(
        selected_queries[
            "row_index"
        ].astype(int)
    )

    eligible_df.loc[
        eligible_df[
            "row_index"
        ].isin(
            query_row_ids
        ),
        "multispectrum_role",
    ] = "query"

    # =====================================================
    # Validation
    # =====================================================

    query_df = eligible_df[
        eligible_df[
            "multispectrum_role"
        ] == "query"
    ]

    reference_df = eligible_df[
        eligible_df[
            "multispectrum_role"
        ] == "reference"
    ]

    query_ids = set(
        query_df[
            "row_index"
        ].astype(int)
    )

    reference_ids = set(
        reference_df[
            "row_index"
        ].astype(int)
    )

    row_overlap = (
        query_ids
        & reference_ids
    )

    reference_structure_set = set(
        reference_df[
            "inchikey14"
        ]
    )

    missing_reference = (
        set(
            query_df[
                "inchikey14"
            ]
        )
        - reference_structure_set
    )

    query_counts = (
        query_df.groupby(
            "inchikey14"
        )
        .size()
    )

    if row_overlap:
        raise RuntimeError(
            "Leakage detected: query/reference "
            "row overlap exists."
        )

    if missing_reference:
        raise RuntimeError(
            "Some query molecules have no "
            "reference spectrum."
        )

    if (
        query_counts.min()
        < 2
    ):
        raise RuntimeError(
            "Some multi-spectrum queries contain "
            "fewer than 2 spectra."
        )

    # =====================================================
    # Save
    # =====================================================

    output_columns = [
        "row_index",
        "inchikey14",
        "normalized_smiles",
        "instrument_type",
        "adduct",
        "precursor_mz",
        "ionization_mode",
        "ingest_lib",
        "is_test_like",
        "multispectrum_role",
    ]

    eligible_df[
        output_columns
    ].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # =====================================================
    # Summary
    # =====================================================

    summary = f"""
ENVEDA CASMI 2026
Stage 5 Multi-Spectrum Validation Split

Eligible structures:
{len(eligible_structures):,}

Query molecules:
{query_df['inchikey14'].nunique():,}

Query spectra:
{len(query_df):,}

Reference spectra:
{len(reference_df):,}

Minimum query spectra per molecule:
{query_counts.min():,}

Maximum query spectra per molecule:
{query_counts.max():,}

Mean query spectra per molecule:
{query_counts.mean():.4f}

Query/reference row overlap:
{len(row_overlap):,}

Query molecules missing reference:
{len(missing_reference):,}
""".strip()

    print_section(
        "MULTI-SPECTRUM SPLIT SUMMARY"
    )

    print(
        summary
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
        "STAGE 5 STEP 9B COMPLETE"
    )


if __name__ == "__main__":
    main()