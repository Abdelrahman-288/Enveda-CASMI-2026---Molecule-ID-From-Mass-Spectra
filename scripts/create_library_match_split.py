from pathlib import Path
from collections import defaultdict
import hashlib

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

TRAIN_PATH = DATASET_DIR / "train.parquet"

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
    / "library_match_validation.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "library_match_validation_summary.txt"
)


# =========================================================
# Test-domain metadata
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

HASH_SEED = "casmi2026_library_match_v1"


def stable_hash(
    structure: str,
    row_index: int,
) -> str:
    """
    Deterministic ordering for selecting query spectra.
    """

    value = (
        f"{HASH_SEED}|{structure}|{row_index}"
    )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 5 STEP 5 "
        "LIBRARY-MATCH VALIDATION SPLIT"
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

    # Stable global row identifier.
    df["row_index"] = range(
        len(df)
    )

    print(
        f"Spectra loaded: "
        f"{len(df):,}"
    )

    print(
        f"Structures: "
        f"{df['inchikey14'].nunique():,}"
    )

    # -----------------------------------------------------
    # Identify test-like spectra
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Count spectra per structure
    # -----------------------------------------------------

    structure_counts = (
        df["inchikey14"]
        .value_counts()
    )

    eligible_structures = set(
        structure_counts[
            structure_counts >= 2
        ].index
    )

    print(
        "Structures with >=2 spectra: "
        f"{len(eligible_structures):,}"
    )

    # -----------------------------------------------------
    # Select one deterministic query spectrum per structure
    # -----------------------------------------------------

    query_rows = []

    structures_with_test_like_query = 0

    grouped = df[
        df["inchikey14"].isin(
            eligible_structures
        )
    ].groupby(
        "inchikey14",
        sort=False,
    )

    print(
        "Selecting held-out query spectra..."
    )

    for structure, group in grouped:

        test_like = group[
            group["is_test_like"]
        ]

        if len(test_like) > 0:
            candidates = test_like
            structures_with_test_like_query += 1
        else:
            candidates = group

        candidates = candidates.copy()

        candidates["_hash"] = [
            stable_hash(
                structure,
                int(row_index),
            )
            for row_index
            in candidates["row_index"]
        ]

        query_row = candidates.sort_values(
            "_hash"
        ).iloc[0]

        query_rows.append(
            int(
                query_row[
                    "row_index"
                ]
            )
        )

    query_rows = set(
        query_rows
    )

    # -----------------------------------------------------
    # Assign role
    # -----------------------------------------------------

    df["library_role"] = "reference"

    df.loc[
        df["row_index"].isin(
            query_rows
        ),
        "library_role",
    ] = "query"

    # Structures with only one spectrum cannot be used for
    # known-compound library-match evaluation.
    singleton_mask = ~df[
        "inchikey14"
    ].isin(
        eligible_structures
    )

    df.loc[
        singleton_mask,
        "library_role",
    ] = "unused"

    # -----------------------------------------------------
    # Verify every query has a same-structure reference
    # -----------------------------------------------------

    query_df = df[
        df["library_role"] == "query"
    ]

    reference_df = df[
        df["library_role"] == "reference"
    ]

    reference_structures = set(
        reference_df[
            "inchikey14"
        ]
    )

    missing_reference = query_df[
        ~query_df["inchikey14"].isin(
            reference_structures
        )
    ]

    if len(missing_reference) > 0:
        raise RuntimeError(
            "Some query structures have no "
            "remaining reference spectrum."
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    query_count = (
        df["library_role"]
        .eq("query")
        .sum()
    )

    reference_count = (
        df["library_role"]
        .eq("reference")
        .sum()
    )

    unused_count = (
        df["library_role"]
        .eq("unused")
        .sum()
    )

    test_like_queries = (
        query_df[
            "is_test_like"
        ].sum()
    )

    summary = f"""
ENVEDA CASMI 2026
Stage 5 Library-Match Validation

Seed:
{HASH_SEED}

Total spectra:
{len(df):,}

Eligible structures:
{len(eligible_structures):,}

Query spectra:
{query_count:,}

Reference spectra:
{reference_count:,}

Unused singleton spectra:
{unused_count:,}

Query structures:
{query_df['inchikey14'].nunique():,}

Queries with test-like spectra:
{test_like_queries:,}

Structures with test-like query available:
{structures_with_test_like_query:,}

Queries missing same-structure reference:
{len(missing_reference):,}
""".strip()

    print(
        "\n"
        + "=" * 90
    )

    print(
        "LIBRARY-MATCH SPLIT SUMMARY"
    )

    print(
        "=" * 90
    )

    print(summary)

    # -----------------------------------------------------
    # Save compact metadata
    # -----------------------------------------------------

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
        "library_role",
    ]

    df[
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

    print(
        "\n"
        + "=" * 90
    )

    print(
        "STAGE 5 STEP 5 COMPLETE"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()