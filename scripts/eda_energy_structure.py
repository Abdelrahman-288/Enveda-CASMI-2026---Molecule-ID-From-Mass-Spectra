from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


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
# Helpers
# =========================================================

def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def flatten_energy_column(series):
    """
    collision_energy_ev contains lists such as:
    [20.0]
    [20.0, 40.0, 60.0]
    None
    """

    values = []

    for item in series.dropna():
        if isinstance(item, (list, tuple, np.ndarray)):
            for value in item:
                if value is not None and np.isfinite(value):
                    values.append(float(value))

    return np.asarray(values, dtype=float)


# =========================================================
# Main
# =========================================================

def main():

    print("\nENVEDA CASMI 2026 — STAGE 2 STEP 3")

    train_columns = [
        "ingest_lib",
        "inchikey14",
        "instrument_type",
        "adduct",
        "collision_energy_ev",
        "collision_energy_orig_units",
    ]

    test_columns = [
        "molecule_id",
        "collision_energy_ev",
        "collision_energy_orig_units",
    ]

    print("\nLoading metadata...")

    train = pq.read_table(
        TRAIN_PATH,
        columns=train_columns,
    ).to_pandas()

    test = pq.read_table(
        TEST_PATH,
        columns=test_columns,
    ).to_pandas()

    # =====================================================
    # Structure frequency
    # =====================================================

    print_section("STRUCTURE FREQUENCY")

    structure_counts = (
        train.groupby("inchikey14")
        .size()
        .sort_values(ascending=False)
    )

    print(structure_counts.describe())

    thresholds = [
        1,
        2,
        3,
        5,
        10,
        20,
        50,
        100,
        500,
        1000,
    ]

    print("\nStructures with at least N spectra")

    total_structures = len(structure_counts)

    for threshold in thresholds:

        count = (structure_counts >= threshold).sum()

        percentage = count / total_structures * 100

        print(
            f">= {threshold:4d} spectra : "
            f"{count:8,d} structures "
            f"({percentage:7.3f}%)"
        )

    print("\nStructures represented by exactly N spectra")

    exact_distribution = (
        structure_counts
        .value_counts()
        .sort_index()
    )

    for n in range(1, 11):
        count = exact_distribution.get(n, 0)

        print(
            f"{n:2d} spectra : "
            f"{count:8,d} structures"
        )

    # =====================================================
    # Library structure diversity
    # =====================================================

    print_section("STRUCTURE DIVERSITY BY LIBRARY")

    library_stats = (
        train.groupby("ingest_lib")
        .agg(
            spectra=("inchikey14", "size"),
            structures=("inchikey14", "nunique"),
        )
    )

    library_stats["spectra_per_structure"] = (
        library_stats["spectra"]
        / library_stats["structures"]
    )

    library_stats = library_stats.sort_values(
        "spectra",
        ascending=False,
    )

    print(library_stats.to_string())

    # =====================================================
    # Collision energy
    # =====================================================

    print_section("COLLISION ENERGY")

    train_energy = flatten_energy_column(
        train["collision_energy_ev"]
    )

    test_energy = flatten_energy_column(
        test["collision_energy_ev"]
    )

    print("\nTrain collision energy values")

    print(
        pd.Series(train_energy)
        .describe()
    )

    print("\nTest collision energy values")

    print(
        pd.Series(test_energy)
        .describe()
    )

    # =====================================================
    # Collision-energy list length
    # =====================================================

    def energy_list_length(value):

        if value is None:
            return 0

        if isinstance(
            value,
            (list, tuple, np.ndarray),
        ):
            return len(value)

        return 0

    train_lengths = train[
        "collision_energy_ev"
    ].apply(energy_list_length)

    test_lengths = test[
        "collision_energy_ev"
    ].apply(energy_list_length)

    print("\nTrain collision-energy list lengths")

    print(
        train_lengths
        .value_counts()
        .sort_index()
        .head(20)
    )

    print("\nTest collision-energy list lengths")

    print(
        test_lengths
        .value_counts()
        .sort_index()
    )

    # =====================================================
    # Most common test energies
    # =====================================================

    print("\nMost common test collision energies")

    test_energy_counter = Counter(test_energy)

    for energy, count in test_energy_counter.most_common(20):

        print(
            f"{energy:8.2f} eV -> "
            f"{count:5,d}"
        )

    # =====================================================
    # Plot structure frequency
    # =====================================================

    plt.figure(figsize=(10, 6))

    values = structure_counts.values

    plt.hist(
        values,
        bins=np.logspace(
            np.log10(1),
            np.log10(values.max()),
            60,
        ),
    )

    plt.xscale("log")

    plt.xlabel("Spectra per structure")
    plt.ylabel("Number of structures")
    plt.title("Training Structure Frequency Distribution")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "structure_frequency_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()

    print(f"\n[SAVED] {output_path}")

    # =====================================================
    # Plot collision-energy distributions
    # =====================================================

    train_energy_plot = train_energy[
        (train_energy >= 0)
        & (train_energy <= 200)
    ]

    test_energy_plot = test_energy[
        (test_energy >= 0)
        & (test_energy <= 200)
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        train_energy_plot,
        bins=80,
        density=True,
        alpha=0.5,
        label="Train",
    )

    plt.hist(
        test_energy_plot,
        bins=80,
        density=True,
        alpha=0.5,
        label="Test",
    )

    plt.xlabel("Collision energy (eV)")
    plt.ylabel("Density")
    plt.title("Train vs Test Collision Energy")

    plt.legend()
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "collision_energy_train_vs_test.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()

    print(f"[SAVED] {output_path}")

    print_section("STAGE 2 STEP 3 COMPLETE")


if __name__ == "__main__":
    main()