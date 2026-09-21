from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "stage2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Load lightweight columns only
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
    "collision_energy_ev",
    "collision_energy_orig_units",
]

TEST_COLUMNS = [
    "molecule_id",
    "spectrum_id",
    "ionization_mode",
    "instrument_type",
    "adduct",
    "precursor_mz",
    "base_peak_intensity",
    "collision_energy_ev",
    "collision_energy_orig_units",
]


def load_data():
    print("Loading lightweight train columns...")
    train = pq.read_table(
        TRAIN_PATH,
        columns=TRAIN_COLUMNS,
    ).to_pandas()

    print("Loading test data...")
    test = pq.read_table(
        TEST_PATH,
        columns=TEST_COLUMNS,
    ).to_pandas()

    return train, test


# =========================================================
# Text summaries
# =========================================================

def print_basic_comparison(train, test):
    print("\n" + "=" * 90)
    print("TRAIN VS TEST DOMAIN COMPARISON")
    print("=" * 90)

    print("\nRows")
    print(f"Train spectra: {len(train):,}")
    print(f"Test spectra:  {len(test):,}")

    print("\nIonization mode")
    print("\nTrain:")
    print(train["ionization_mode"].value_counts(dropna=False))

    print("\nTest:")
    print(test["ionization_mode"].value_counts(dropna=False))

    print("\nTop train instrument types")
    print(train["instrument_type"].value_counts(dropna=False).head(20))

    print("\nTest instrument types")
    print(test["instrument_type"].value_counts(dropna=False))

    print("\nTop train adducts")
    print(train["adduct"].value_counts(dropna=False).head(20))

    print("\nTest adducts")
    print(test["adduct"].value_counts(dropna=False))

    print("\nPrecursor m/z")
    print("\nTrain:")
    print(train["precursor_mz"].describe())

    print("\nTest:")
    print(test["precursor_mz"].describe())

    print("\nTraining num_peaks")
    print(train["num_peaks"].describe())

    print("\nTraining precursor_error_ppm")
    print(train["precursor_error_ppm"].describe())


# =========================================================
# Plots
# =========================================================

def plot_precursor_histogram(train, test):
    train_clean = train[
        train["precursor_mz"].between(0, 1200)
    ]["precursor_mz"]

    test_clean = test[
        test["precursor_mz"].between(0, 1200)
    ]["precursor_mz"]

    plt.figure(figsize=(10, 6))

    plt.hist(
        train_clean,
        bins=100,
        density=True,
        alpha=0.5,
        label="Train",
    )

    plt.hist(
        test_clean,
        bins=100,
        density=True,
        alpha=0.5,
        label="Test",
    )

    plt.xlabel("Precursor m/z")
    plt.ylabel("Density")
    plt.title("Train vs Test Precursor m/z Distribution")
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "precursor_mz_train_vs_test.png"
    plt.savefig(path, dpi=160)
    plt.close()

    print(f"[SAVED] {path}")


def plot_test_adducts(test):
    counts = test["adduct"].value_counts().sort_values()

    plt.figure(figsize=(9, 6))
    counts.plot(kind="barh")

    plt.xlabel("Number of spectra")
    plt.ylabel("Adduct")
    plt.title("Test Adduct Distribution")
    plt.tight_layout()

    path = OUTPUT_DIR / "test_adduct_distribution.png"
    plt.savefig(path, dpi=160)
    plt.close()

    print(f"[SAVED] {path}")


def plot_train_libraries(train):
    counts = (
        train["ingest_lib"]
        .value_counts()
        .sort_values()
    )

    plt.figure(figsize=(10, 7))
    counts.plot(kind="barh")

    plt.xlabel("Number of spectra")
    plt.ylabel("Training library")
    plt.title("Training Library Distribution")
    plt.tight_layout()

    path = OUTPUT_DIR / "train_library_distribution.png"
    plt.savefig(path, dpi=160)
    plt.close()

    print(f"[SAVED] {path}")


def plot_num_peaks(train):
    values = train[
        train["num_peaks"].between(1, 1000)
    ]["num_peaks"]

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=100)

    plt.xlabel("Number of peaks")
    plt.ylabel("Number of spectra")
    plt.title("Training Spectrum Peak Count Distribution (<=1000 peaks)")
    plt.tight_layout()

    path = OUTPUT_DIR / "train_num_peaks_distribution.png"
    plt.savefig(path, dpi=160)
    plt.close()

    print(f"[SAVED] {path}")


# =========================================================
# timsTOF-specific analysis
# =========================================================

def analyze_timstof(train, test):
    print("\n" + "=" * 90)
    print("TIMSTOF DOMAIN ANALYSIS")
    print("=" * 90)

    train_timstof = train[
        train["instrument_type"] == "timsTOF"
    ].copy()

    print(f"\nTraining timsTOF spectra: {len(train_timstof):,}")
    print(f"Test timsTOF spectra:     {len(test):,}")

    print("\nTraining timsTOF precursor m/z")
    print(train_timstof["precursor_mz"].describe())

    print("\nTest precursor m/z")
    print(test["precursor_mz"].describe())

    print("\nTraining timsTOF adducts")
    print(
        train_timstof["adduct"]
        .value_counts()
        .head(20)
    )

    return train_timstof


# =========================================================
# Main
# =========================================================

def main():
    print("\nENVEDA CASMI 2026 — STAGE 2 EDA")

    train, test = load_data()

    print_basic_comparison(train, test)

    plot_precursor_histogram(train, test)
    plot_test_adducts(test)
    plot_train_libraries(train)
    plot_num_peaks(train)

    analyze_timstof(train, test)

    print("\n" + "=" * 90)
    print("STAGE 2 STEP 1 COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()