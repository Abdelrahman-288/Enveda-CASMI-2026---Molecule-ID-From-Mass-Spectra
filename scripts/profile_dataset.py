from pathlib import Path
from collections import Counter

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
SUBMISSION_PATH = DATASET_DIR / "sample_submission.csv"


# =========================================================
# Helpers
# =========================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_counter(counter: Counter, limit: int = 30) -> None:
    for key, value in counter.most_common(limit):
        print(f"{str(key):30s} {value:,}")


# =========================================================
# Train profiling
# =========================================================

def profile_train() -> None:
    print_section("TRAIN DATASET PROFILE")

    # Only load metadata/lightweight columns.
    columns = [
        "ingest_lib",
        "normalized_smiles",
        "inchikey",
        "inchikey14",
        "molecular_formula",
        "ionization_mode",
        "instrument_type",
        "adduct",
        "adduct_orig",
        "precursor_mz",
        "precursor_error_ppm",
        "num_peaks",
        "base_peak_intensity",
        "collision_energy_ev",
        "collision_energy_orig",
        "collision_energy_orig_units",
    ]

    table = pq.read_table(
        TRAIN_PATH,
        columns=columns,
    )

    df = table.to_pandas()

    print(f"Rows: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    print("\nUnique values")
    print(f"Unique normalized_smiles: {df['normalized_smiles'].nunique(dropna=True):,}")
    print(f"Unique inchikey:          {df['inchikey'].nunique(dropna=True):,}")
    print(f"Unique inchikey14:        {df['inchikey14'].nunique(dropna=True):,}")
    print(f"Unique formulas:          {df['molecular_formula'].nunique(dropna=True):,}")

    print("\nMissing values")
    missing = df.isna().sum().sort_values(ascending=False)

    for column, count in missing.items():
        pct = count / len(df) * 100
        print(f"{column:30s} {count:12,d} ({pct:7.3f}%)")

    print("\nTraining library distribution")
    print_counter(Counter(df["ingest_lib"].dropna()))

    print("\nIonization mode distribution")
    print_counter(Counter(df["ionization_mode"].dropna()))

    print("\nInstrument type distribution")
    print_counter(Counter(df["instrument_type"].dropna()))

    print("\nAdduct distribution")
    print_counter(Counter(df["adduct"].dropna()), limit=50)

    print("\nCollision energy units")
    print_counter(
        Counter(df["collision_energy_orig_units"].dropna()),
        limit=30,
    )

    print("\nPrecursor m/z statistics")
    print(df["precursor_mz"].describe())

    print("\nPrecursor error ppm statistics")
    print(df["precursor_error_ppm"].describe())

    print("\nNumber of peaks statistics")
    print(df["num_peaks"].describe())

    print("\nBase peak intensity statistics")
    print(df["base_peak_intensity"].describe())

    spectra_per_structure = (
        df.groupby("inchikey14", dropna=True)
        .size()
    )

    print("\nSpectra per unique 2D structure")
    print(spectra_per_structure.describe())

    print("\nStructures with most spectra")
    print(
        spectra_per_structure
        .sort_values(ascending=False)
        .head(20)
    )


# =========================================================
# Test profiling
# =========================================================

def profile_test() -> None:
    print_section("TEST DATASET PROFILE")

    test = pd.read_parquet(TEST_PATH)

    print(f"Rows: {len(test):,}")
    print(f"Unique molecule_id: {test['molecule_id'].nunique():,}")
    print(f"Unique spectrum_id: {test['spectrum_id'].nunique():,}")

    print("\nMissing values")
    missing = test.isna().sum().sort_values(ascending=False)

    for column, count in missing.items():
        pct = count / len(test) * 100
        print(f"{column:35s} {count:8,d} ({pct:7.3f}%)")

    print("\nIonization mode distribution")
    print_counter(Counter(test["ionization_mode"].dropna()))

    print("\nInstrument type distribution")
    print_counter(Counter(test["instrument_type"].dropna()))

    print("\nAdduct distribution")
    print_counter(Counter(test["adduct"].dropna()), limit=30)

    print("\nCollision energy units")
    print_counter(
        Counter(test["collision_energy_orig_units"].dropna())
    )

    print("\nPrecursor m/z statistics")
    print(test["precursor_mz"].describe())

    spectra_per_molecule = (
        test.groupby("molecule_id")
        .size()
    )

    print("\nSpectra per test molecule")
    print(spectra_per_molecule.describe())

    print("\nSpectrum-count distribution")
    counts = spectra_per_molecule.value_counts().sort_index()

    for n_spectra, n_molecules in counts.items():
        print(
            f"{n_spectra:2d} spectra -> "
            f"{n_molecules:4d} molecules"
        )

    print("\nMolecules with most spectra")
    print(
        spectra_per_molecule
        .sort_values(ascending=False)
        .head(20)
    )


# =========================================================
# Submission integrity
# =========================================================

def validate_submission_ids() -> None:
    print_section("SAMPLE SUBMISSION INTEGRITY")

    test = pd.read_parquet(
        TEST_PATH,
        columns=["molecule_id"],
    )

    submission = pd.read_csv(SUBMISSION_PATH)

    test_ids = set(test["molecule_id"].unique())
    submission_ids = set(submission["molecule_id"].unique())

    print(f"Unique test molecule IDs:       {len(test_ids):,}")
    print(f"Unique submission molecule IDs: {len(submission_ids):,}")

    missing_from_submission = test_ids - submission_ids
    extra_in_submission = submission_ids - test_ids

    print(f"Missing from submission: {len(missing_from_submission)}")
    print(f"Extra in submission:     {len(extra_in_submission)}")

    if not missing_from_submission and not extra_in_submission:
        print("[OK] Test and sample submission molecule IDs match.")
    else:
        if missing_from_submission:
            print("\nMissing IDs:")
            print(sorted(missing_from_submission)[:20])

        if extra_in_submission:
            print("\nExtra IDs:")
            print(sorted(extra_in_submission)[:20])


# =========================================================
# Main
# =========================================================

def main() -> None:
    print("\nENVEDA CASMI 2026 — FULL DATASET PROFILE")

    profile_train()
    profile_test()
    validate_submission_ids()

    print_section("STAGE 1 DATASET PROFILING COMPLETE")


if __name__ == "__main__":
    main()