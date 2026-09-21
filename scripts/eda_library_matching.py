from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = DATASET_DIR / "train.parquet"
TEST_PATH = DATASET_DIR / "test.parquet"


TRAIN_COLUMNS = [
    "ingest_lib",
    "inchikey14",
    "instrument_type",
    "adduct",
    "ionization_mode",
    "precursor_mz",
    "collision_energy_ev",
]

TEST_COLUMNS = [
    "molecule_id",
    "instrument_type",
    "adduct",
    "ionization_mode",
    "precursor_mz",
    "collision_energy_ev",
]


def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def main():

    print("\nENVEDA CASMI 2026 — STAGE 2 STEP 4")

    print("\nLoading metadata...")

    train = pq.read_table(
        TRAIN_PATH,
        columns=TRAIN_COLUMNS,
    ).to_pandas()

    test = pq.read_table(
        TEST_PATH,
        columns=TEST_COLUMNS,
    ).to_pandas()

    test_min_mz = test["precursor_mz"].min()
    test_max_mz = test["precursor_mz"].max()

    test_adducts = set(test["adduct"].unique())

    libraries = sorted(train["ingest_lib"].dropna().unique())

    results = []

    for lib in libraries:

        subset = train[train["ingest_lib"] == lib]

        total = len(subset)

        timstof = subset["instrument_type"].eq("timsTOF").sum()

        mass_match = subset[
            "precursor_mz"
        ].between(
            test_min_mz,
            test_max_mz
        ).sum()

        adduct_match = subset[
            "adduct"
        ].isin(
            test_adducts
        ).sum()

        strict_mask = (
            subset["instrument_type"].eq("timsTOF")
            &
            subset["precursor_mz"].between(
                test_min_mz,
                test_max_mz
            )
            &
            subset["adduct"].isin(test_adducts)
        )

        strict_count = strict_mask.sum()

        structures = subset[
            "inchikey14"
        ].nunique()

        strict_structures = subset.loc[
            strict_mask,
            "inchikey14"
        ].nunique()

        results.append({
            "library": lib,
            "spectra": total,
            "structures": structures,
            "timstof_count": timstof,
            "timstof_pct": timstof / total * 100,
            "mass_match_count": mass_match,
            "mass_match_pct": mass_match / total * 100,
            "adduct_match_count": adduct_match,
            "adduct_match_pct": adduct_match / total * 100,
            "strict_test_like": strict_count,
            "strict_pct": strict_count / total * 100,
            "strict_structures": strict_structures,
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "strict_test_like",
        ascending=False,
    )

    print_section("LIBRARY TEST-DOMAIN MATCHING")

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        200,
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print_section("TOP LIBRARIES BY TEST-LIKE SPECTRA")

    print(
        results_df[
            [
                "library",
                "strict_test_like",
                "strict_pct",
                "strict_structures",
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print_section("TEST-LIKE TRAINING PRIORITY")

    priority = results_df[
        results_df["strict_test_like"] > 0
    ].copy()

    for _, row in priority.iterrows():

        print(
            f"{row['library']:25s} "
            f"spectra={int(row['strict_test_like']):10,d} "
            f"structures={int(row['strict_structures']):8,d} "
            f"coverage={row['strict_pct']:7.2f}%"
        )

    print_section("STAGE 2 STEP 4 COMPLETE")


if __name__ == "__main__":
    main()