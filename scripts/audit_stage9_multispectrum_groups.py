from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
    / "train.parquet"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage8_candidate_aware_chemistry_results.csv"
)


def describe_group_sizes(
    name: str,
    dataframe: pd.DataFrame,
    group_columns: list[str],
) -> None:

    if len(dataframe) == 0:
        print(
            f"\n{name}: no rows"
        )
        return

    sizes = (
        dataframe
        .groupby(
            group_columns,
            dropna=False,
        )
        .size()
        .to_numpy()
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        name
    )

    print(
        "=" * 90
    )

    print(
        f"Rows:          {len(dataframe):,}"
    )

    print(
        f"Groups:        {len(sizes):,}"
    )

    print(
        f"Mean spectra:  {np.mean(sizes):.2f}"
    )

    print(
        f"Median:        {np.median(sizes):.1f}"
    )

    print(
        f"Min:           {np.min(sizes)}"
    )

    print(
        f"P75:           {np.percentile(sizes, 75):.1f}"
    )

    print(
        f"P90:           {np.percentile(sizes, 90):.1f}"
    )

    print(
        f"P95:           {np.percentile(sizes, 95):.1f}"
    )

    print(
        f"P99:           {np.percentile(sizes, 99):.1f}"
    )

    print(
        f"Max:           {np.max(sizes)}"
    )

    print(
        "\nGroup-size distribution:"
    )

    counts = (
        pd.Series(sizes)
        .value_counts()
        .sort_index()
    )

    for size, count in counts.items():

        if size <= 15:

            print(
                f"{int(size):>2} spectra: "
                f"{int(count):,} groups"
            )

    above_15 = int(
        np.sum(
            sizes > 15
        )
    )

    if above_15 > 0:

        print(
            f">15 spectra: "
            f"{above_15:,} groups"
        )


def main():

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 9 MULTI-SPECTRUM GROUP AUDIT"
    )

    # =====================================================
    # Load Stage 8 validation rows
    # =====================================================

    print(
        "\nLoading Stage 8 results..."
    )

    results = pd.read_csv(
        RESULTS_PATH
    )

    print(
        f"Stage 8 rows: "
        f"{len(results):,}"
    )

    # =====================================================
    # Load only metadata required from original train set
    # =====================================================

    print(
        "\nLoading train metadata..."
    )

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "inchikey14",
            "ingest_lib",
            "instrument_type",
            "adduct",
            "precursor_mz",
            "collision_energy_ev",
            "collision_energy_orig",
            "collision_energy_orig_units",
        ],
    )

    # =====================================================
    # Recover metadata using original parquet row number
    # =====================================================

    original_rows = (
        results[
            "original_row_index"
        ]
        .astype(np.int64)
        .to_numpy()
    )

    metadata = (
        train
        .iloc[
            original_rows
        ]
        .reset_index(
            drop=True
        )
    )

    df = (
        results
        .reset_index(
            drop=True
        )
        .copy()
    )

    # Avoid duplicate names from Stage 8 result columns.

    df[
        "collision_energy_ev"
    ] = metadata[
        "collision_energy_ev"
    ].to_numpy()

    df[
        "collision_energy_orig"
    ] = metadata[
        "collision_energy_orig"
    ].to_numpy()

    df[
        "collision_energy_orig_units"
    ] = metadata[
        "collision_energy_orig_units"
    ].to_numpy()

    # =====================================================
    # Only chemistry-eligible rows
    # =====================================================

    eligible = df[
        (
            df[
                "hard_filter"
            ]
            == True
        )
        &
        (
            df[
                "true_survived"
            ]
            == True
        )
    ].copy()

    print(
        f"\nHard-filter surviving rows: "
        f"{len(eligible):,}"
    )

    # =====================================================
    # All eligible spectra grouped only by structure
    # =====================================================

    describe_group_sizes(
        name=(
            "A) All eligible rows — "
            "grouped only by inchikey14"
        ),
        dataframe=eligible,
        group_columns=[
            "inchikey14",
        ],
    )

    # =====================================================
    # timsTOF only
    # =====================================================

    timstof = eligible[
        eligible[
            "instrument_type"
        ]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq(
            "timstof"
        )
    ].copy()

    describe_group_sizes(
        name=(
            "B) timsTOF — "
            "grouped only by inchikey14"
        ),
        dataframe=timstof,
        group_columns=[
            "inchikey14",
        ],
    )

    # =====================================================
    # timsTOF + source
    # =====================================================

    describe_group_sizes(
        name=(
            "C) timsTOF — "
            "grouped by inchikey14 + ingest_lib"
        ),
        dataframe=timstof,
        group_columns=[
            "inchikey14",
            "ingest_lib",
        ],
    )

    # =====================================================
    # timsTOF + source + adduct
    # =====================================================

    describe_group_sizes(
        name=(
            "D) timsTOF — "
            "grouped by inchikey14 + ingest_lib + adduct"
        ),
        dataframe=timstof,
        group_columns=[
            "inchikey14",
            "ingest_lib",
            "adduct",
        ],
    )

    # =====================================================
    # Collision-energy analysis
    # =====================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "TIMSTOF COLLISION ENERGY AUDIT"
    )

    print(
        "=" * 90
    )

    ce_numeric = pd.to_numeric(
        timstof[
            "collision_energy_ev"
        ],
        errors="coerce",
    )

    print(
        f"Rows with numeric collision energy: "
        f"{ce_numeric.notna().sum():,}/"
        f"{len(timstof):,}"
    )

    if ce_numeric.notna().any():

        unique_ce = (
            ce_numeric
            .dropna()
            .value_counts()
            .sort_index()
        )

        print(
            "\nMost common collision energies:"
        )

        print(
            unique_ce
            .sort_values(
                ascending=False
            )
            .head(20)
            .to_string()
        )

    # =====================================================
    # Number of unique CE values per pseudo molecule
    # =====================================================

    ce_groups = (
        timstof
        .assign(
            collision_energy_numeric=(
                ce_numeric
            )
        )
        .groupby(
            [
                "inchikey14",
                "ingest_lib",
                "adduct",
            ],
            dropna=False,
        )[
            "collision_energy_numeric"
        ]
        .nunique(
            dropna=True
        )
    )

    print(
        "\nUnique collision-energy values "
        "per timsTOF pseudo-group:"
    )

    print(
        ce_groups
        .value_counts()
        .sort_index()
        .head(20)
        .to_string()
    )

    # =====================================================
    # Test-like groups
    #
    # Test group size:
    # median ≈ 3 spectra.
    #
    # Count groups that naturally have 2–9 spectra.
    # =====================================================

    pseudo_sizes = (
        timstof
        .groupby(
            [
                "inchikey14",
                "ingest_lib",
                "adduct",
            ],
            dropna=False,
        )
        .size()
    )

    test_like = pseudo_sizes[
        (
            pseudo_sizes >= 2
        )
        &
        (
            pseudo_sizes <= 9
        )
    ]

    print(
        "\n"
        + "=" * 90
    )

    print(
        "TEST-LIKE TIMSTOF PSEUDO-GROUPS"
    )

    print(
        "=" * 90
    )

    print(
        f"Groups with 2–9 spectra: "
        f"{len(test_like):,}"
    )

    if len(
        test_like
    ) > 0:

        print(
            f"Mean group size: "
            f"{test_like.mean():.2f}"
        )

        print(
            f"Median group size: "
            f"{test_like.median():.1f}"
        )

        print(
            "\nDistribution:"
        )

        print(
            test_like
            .value_counts()
            .sort_index()
            .to_string()
        )

    # =====================================================
    # Sources
    # =====================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "TIMSTOF SOURCES"
    )

    print(
        "=" * 90
    )

    print(
        timstof[
            "ingest_lib"
        ]
        .fillna(
            "<missing>"
        )
        .value_counts()
        .head(20)
        .to_string()
    )


if __name__ == "__main__":
    main()