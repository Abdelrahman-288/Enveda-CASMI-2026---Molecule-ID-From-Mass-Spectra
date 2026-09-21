from pathlib import Path
from statistics import mean, median

import numpy as np
import pyarrow.parquet as pq

from casmi.spectra.preprocessing import (
    SpectrumPreprocessingConfig,
    preprocess_spectrum,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = DATASET_DIR / "train.parquet"
TEST_PATH = DATASET_DIR / "test.parquet"


CONFIGS = {
    "raw_normalized": SpectrumPreprocessingConfig(
        min_relative_intensity=0.0,
        max_peaks=None,
        precursor_tolerance_da=None,
        intensity_transform="none",
    ),

    "intensity_1pct": SpectrumPreprocessingConfig(
        min_relative_intensity=0.01,
        max_peaks=None,
        precursor_tolerance_da=None,
        intensity_transform="none",
    ),

    "intensity_2pct": SpectrumPreprocessingConfig(
        min_relative_intensity=0.02,
        max_peaks=None,
        precursor_tolerance_da=None,
        intensity_transform="none",
    ),

    "top64": SpectrumPreprocessingConfig(
        max_peaks=64,
    ),

    "top128": SpectrumPreprocessingConfig(
        max_peaks=128,
    ),

    "top256": SpectrumPreprocessingConfig(
        max_peaks=256,
    ),

    "precursor_plus_2": SpectrumPreprocessingConfig(
        precursor_tolerance_da=2.0,
    ),

    "sqrt": SpectrumPreprocessingConfig(
        intensity_transform="sqrt",
    ),

    "top128_sqrt": SpectrumPreprocessingConfig(
        max_peaks=128,
        intensity_transform="sqrt",
    ),

    "top128_1pct_sqrt": SpectrumPreprocessingConfig(
        min_relative_intensity=0.01,
        max_peaks=128,
        intensity_transform="sqrt",
    ),
}


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def evaluate_dataset(
    path: Path,
    name: str,
    max_rows: int,
) -> None:

    print_section(name)

    parquet_file = pq.ParquetFile(path)

    columns = [
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_mz",
    ]

    results = {
        config_name: {
            "before": [],
            "after": [],
        }
        for config_name in CONFIGS
    }

    processed_rows = 0

    for batch in parquet_file.iter_batches(
        columns=columns,
        batch_size=1024,
    ):
        df = batch.to_pandas()

        for row in df.itertuples(index=False):

            original_count = len(row.ms2_mzs)

            for config_name, config in CONFIGS.items():

                mzs, intensities = preprocess_spectrum(
                    row.ms2_mzs,
                    row.ms2_normalized_intensities,
                    precursor_mz=row.precursor_mz,
                    config=config,
                )

                results[config_name]["before"].append(
                    original_count
                )

                results[config_name]["after"].append(
                    len(mzs)
                )

            processed_rows += 1

            if processed_rows >= max_rows:
                break

        if processed_rows >= max_rows:
            break

    print(f"Spectra evaluated: {processed_rows:,}")

    print(
        "\n"
        f"{'Configuration':25s}"
        f"{'Mean before':>14s}"
        f"{'Mean after':>14s}"
        f"{'Median after':>15s}"
        f"{'Reduction %':>14s}"
    )

    print("-" * 82)

    for config_name, values in results.items():

        before = values["before"]
        after = values["after"]

        mean_before = mean(before)
        mean_after = mean(after)
        median_after = median(after)

        reduction = (
            1.0
            - (
                sum(after)
                / sum(before)
            )
        ) * 100

        print(
            f"{config_name:25s}"
            f"{mean_before:14.2f}"
            f"{mean_after:14.2f}"
            f"{median_after:15.2f}"
            f"{reduction:14.2f}"
        )

    print("\nEmpty spectra after preprocessing")

    for config_name, values in results.items():

        empty = sum(
            count == 0
            for count in values["after"]
        )

        print(
            f"{config_name:25s}: "
            f"{empty:8,d}"
        )


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 3 STEP 3 "
        "PREPROCESSING BENCHMARK"
    )

    evaluate_dataset(
        TRAIN_PATH,
        "TRAIN PREPROCESSING BENCHMARK",
        max_rows=50_000,
    )

    evaluate_dataset(
        TEST_PATH,
        "TEST PREPROCESSING BENCHMARK",
        max_rows=10_000,
    )

    print_section(
        "STAGE 3 STEP 3 COMPLETE"
    )


if __name__ == "__main__":
    main()