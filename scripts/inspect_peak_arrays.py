from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = DATASET_DIR / "train.parquet"
TEST_PATH = DATASET_DIR / "test.parquet"


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def inspect_dataset(path: Path, name: str, max_rows: int = 100_000) -> None:
    print_section(name)

    columns = [
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_mz",
    ]

    parquet_file = pq.ParquetFile(path)

    checked = 0

    mismatched_lengths = 0
    unsorted_mz = 0
    nonfinite_mz = 0
    nonfinite_intensity = 0
    negative_mz = 0
    negative_intensity = 0
    intensity_above_one = 0
    duplicate_mz = 0
    peaks_above_precursor = 0
    spectra_with_peaks_above_precursor = 0

    total_peaks = 0

    for batch in parquet_file.iter_batches(
        columns=columns,
        batch_size=2048,
    ):
        df = batch.to_pandas()

        for row in df.itertuples(index=False):
            mzs = np.asarray(row.ms2_mzs, dtype=float)
            intensities = np.asarray(
                row.ms2_normalized_intensities,
                dtype=float,
            )

            precursor_mz = float(row.precursor_mz)

            checked += 1

            if len(mzs) != len(intensities):
                mismatched_lengths += 1
                continue

            total_peaks += len(mzs)

            if len(mzs) > 1:
                if np.any(np.diff(mzs) < 0):
                    unsorted_mz += 1

            if not np.all(np.isfinite(mzs)):
                nonfinite_mz += 1

            if not np.all(np.isfinite(intensities)):
                nonfinite_intensity += 1

            if np.any(mzs < 0):
                negative_mz += 1

            if np.any(intensities < 0):
                negative_intensity += 1

            if np.any(intensities > 1.0 + 1e-9):
                intensity_above_one += 1

            if len(mzs) != len(np.unique(mzs)):
                duplicate_mz += 1

            above = mzs > (precursor_mz + 2.0)

            n_above = int(np.sum(above))

            peaks_above_precursor += n_above

            if n_above > 0:
                spectra_with_peaks_above_precursor += 1

            if checked >= max_rows:
                break

        if checked >= max_rows:
            break

    print(f"Rows inspected: {checked:,}")
    print(f"Total peaks inspected: {total_peaks:,}")

    print("\nArray integrity")
    print(f"Mismatched m/z-intensity lengths: {mismatched_lengths:,}")
    print(f"Spectra with unsorted m/z values:  {unsorted_mz:,}")

    print("\nNumerical validity")
    print(f"Spectra with non-finite m/z:        {nonfinite_mz:,}")
    print(f"Spectra with non-finite intensity: {nonfinite_intensity:,}")
    print(f"Spectra with negative m/z:          {negative_mz:,}")
    print(f"Spectra with negative intensity:   {negative_intensity:,}")
    print(f"Spectra with intensity > 1:        {intensity_above_one:,}")

    print("\nPeak structure")
    print(f"Spectra with duplicate m/z values: {duplicate_mz:,}")
    print(
        f"Spectra with peaks > precursor+2: "
        f"{spectra_with_peaks_above_precursor:,}"
    )
    print(
        f"Total peaks > precursor+2: "
        f"{peaks_above_precursor:,}"
    )


def main() -> None:
    print("\nENVEDA CASMI 2026 — STAGE 3 STEP 1")

    inspect_dataset(
        TRAIN_PATH,
        "TRAIN PEAK ARRAY INSPECTION",
        max_rows=100_000,
    )

    inspect_dataset(
        TEST_PATH,
        "TEST PEAK ARRAY INSPECTION",
        max_rows=10_000,
    )

    print_section("STAGE 3 STEP 1 COMPLETE")


if __name__ == "__main__":
    main()