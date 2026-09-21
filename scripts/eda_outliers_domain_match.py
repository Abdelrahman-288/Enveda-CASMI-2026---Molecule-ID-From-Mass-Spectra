from pathlib import Path

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
TEST_PATH = DATASET_DIR / "test.parquet"


# =========================================================
# Columns
# =========================================================

TRAIN_COLUMNS = [
    "ingest_lib",
    "inchikey14",
    "ionization_mode",
    "instrument_type",
    "adduct",
    "precursor_mz",
    "precursor_error_ppm",
    "num_peaks",
    "base_peak_intensity",
]

TEST_COLUMNS = [
    "molecule_id",
    "ionization_mode",
    "instrument_type",
    "adduct",
    "precursor_mz",
]


def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def pct(n, total):
    return (n / total * 100) if total else 0.0


def main():
    print("\nENVEDA CASMI 2026 — STAGE 2 STEP 2")

    print("\nLoading lightweight train/test metadata...")

    train = pq.read_table(
        TRAIN_PATH,
        columns=TRAIN_COLUMNS,
    ).to_pandas()

    test = pq.read_table(
        TEST_PATH,
        columns=TEST_COLUMNS,
    ).to_pandas()

    total_train = len(train)

    # =====================================================
    # Test-domain definitions
    # =====================================================

    test_min_mz = test["precursor_mz"].min()
    test_max_mz = test["precursor_mz"].max()

    test_adducts = set(test["adduct"].dropna().unique())
    test_modes = set(test["ionization_mode"].dropna().unique())

    print_section("TEST DOMAIN")

    print(f"Test precursor range: {test_min_mz:.4f} -> {test_max_mz:.4f}")

    print("\nTest adducts:")
    for value in sorted(test_adducts):
        print(f"  {value}")

    print("\nTest ionization modes:")
    for value in sorted(test_modes):
        print(f"  {value}")

    # =====================================================
    # Domain filters
    # =====================================================

    timstof_mask = train["instrument_type"].eq("timsTOF")

    precursor_match_mask = train["precursor_mz"].between(
        test_min_mz,
        test_max_mz
    )

    adduct_match_mask = train["adduct"].isin(test_adducts)

    mode_match_mask = train["ionization_mode"].isin(test_modes)

    print_section("TRAINING SPECTRA MATCHING TEST DOMAIN")

    n_timstof = timstof_mask.sum()
    n_precursor = precursor_match_mask.sum()
    n_adduct = adduct_match_mask.sum()

    print(
        f"timsTOF spectra: "
        f"{n_timstof:,} "
        f"({pct(n_timstof, total_train):.2f}%)"
    )

    print(
        f"Inside test precursor range: "
        f"{n_precursor:,} "
        f"({pct(n_precursor, total_train):.2f}%)"
    )

    print(
        f"Test-compatible adducts: "
        f"{n_adduct:,} "
        f"({pct(n_adduct, total_train):.2f}%)"
    )

    # =====================================================
    # Combined matching subsets
    # =====================================================

    timstof_precursor = timstof_mask & precursor_match_mask

    timstof_precursor_adduct = (
        timstof_mask
        & precursor_match_mask
        & adduct_match_mask
    )

    strict_test_like = (
        timstof_mask
        & precursor_match_mask
        & adduct_match_mask
        & mode_match_mask
    )

    print_section("COMBINED TEST-LIKE SUBSETS")

    for name, mask in [
        ("timsTOF + precursor range", timstof_precursor),
        (
            "timsTOF + precursor range + adduct",
            timstof_precursor_adduct,
        ),
        (
            "Strict test-like",
            strict_test_like,
        ),
    ]:
        n = mask.sum()

        print(
            f"{name:45s} "
            f"{n:12,d} "
            f"({pct(n, total_train):7.3f}%)"
        )

    # =====================================================
    # Unique structures in subsets
    # =====================================================

    print_section("UNIQUE STRUCTURES IN TEST-LIKE SUBSETS")

    full_structures = train["inchikey14"].nunique()

    timstof_structures = (
        train.loc[timstof_mask, "inchikey14"]
        .nunique()
    )

    strict_structures = (
        train.loc[strict_test_like, "inchikey14"]
        .nunique()
    )

    print(f"All train structures:       {full_structures:,}")
    print(f"timsTOF structures:         {timstof_structures:,}")
    print(f"Strict test-like structures:{strict_structures:,}")

    # =====================================================
    # Library contribution to strict subset
    # =====================================================

    print_section("LIBRARY CONTRIBUTION TO STRICT TEST-LIKE SUBSET")

    strict = train.loc[strict_test_like]

    print(
        strict["ingest_lib"]
        .value_counts()
        .to_string()
    )

    # =====================================================
    # Outlier investigation
    # =====================================================

    print_section("PRECURSOR M/Z OUTLIERS")

    thresholds = [
        1000,
        1200,
        2000,
        5000,
        10000,
    ]

    for threshold in thresholds:
        n = (train["precursor_mz"] > threshold).sum()

        print(
            f"precursor_mz > {threshold:5d}: "
            f"{n:10,d} "
            f"({pct(n, total_train):7.4f}%)"
        )

    print("\nSmall precursor values")

    for threshold in [10, 50, 100, 130]:
        n = (train["precursor_mz"] < threshold).sum()

        print(
            f"precursor_mz < {threshold:4d}: "
            f"{n:10,d} "
            f"({pct(n, total_train):7.4f}%)"
        )

    # =====================================================
    # Peak-count outliers
    # =====================================================

    print_section("PEAK COUNT OUTLIERS")

    for threshold in [
        500,
        1000,
        2000,
        5000,
        10000,
    ]:
        n = (train["num_peaks"] > threshold).sum()

        print(
            f"num_peaks > {threshold:5d}: "
            f"{n:10,d} "
            f"({pct(n, total_train):7.4f}%)"
        )

    print("\nVery small spectra")

    for threshold in [2, 3, 5, 6, 10]:
        n = (train["num_peaks"] < threshold).sum()

        print(
            f"num_peaks < {threshold:2d}: "
            f"{n:10,d} "
            f"({pct(n, total_train):7.4f}%)"
        )

    # =====================================================
    # Precursor error investigation
    # =====================================================

    print_section("PRECURSOR ERROR OUTLIERS")

    valid_error = train["precursor_error_ppm"].dropna()

    for threshold in [
        5,
        10,
        20,
        50,
        100,
        1000,
    ]:
        n = (valid_error.abs() > threshold).sum()

        print(
            f"|precursor_error_ppm| > {threshold:4d}: "
            f"{n:10,d} "
            f"({pct(n, len(valid_error)):7.4f}%)"
        )

    # =====================================================
    # Candidate training regimes
    # =====================================================

    print_section("POTENTIAL TRAINING REGIMES")

    regimes = {
        "ALL": pd.Series(True, index=train.index),

        "TIMSTOF": timstof_mask,

        "TIMSTOF_TEST_MASS": (
            timstof_mask
            & precursor_match_mask
        ),

        "TIMSTOF_TEST_MASS_ADDUCT": (
            timstof_mask
            & precursor_match_mask
            & adduct_match_mask
        ),
    }

    for name, mask in regimes.items():
        subset = train.loc[mask]

        n_spectra = len(subset)
        n_structures = subset["inchikey14"].nunique()

        print(
            f"{name:30s} "
            f"spectra={n_spectra:10,d} "
            f"structures={n_structures:8,d}"
        )

    print_section("STAGE 2 STEP 2 COMPLETE")


if __name__ == "__main__":
    main()