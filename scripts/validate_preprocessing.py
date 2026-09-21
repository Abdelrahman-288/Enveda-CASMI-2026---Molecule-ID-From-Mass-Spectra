from pathlib import Path
from statistics import mean, median

import numpy as np
import pyarrow.parquet as pq

from casmi.spectra.config import load_preprocessing_config
from casmi.spectra.preprocessing import preprocess_spectrum


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = DATASET_DIR / "train.parquet"
TEST_PATH = DATASET_DIR / "test.parquet"

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "data"
    / "spectrum_preprocessing.yaml"
)


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def validate_dataset(
    path: Path,
    name: str,
    max_rows: int,
) -> None:

    config = load_preprocessing_config(
        CONFIG_PATH,
        profile="baseline",
    )

    parquet_file = pq.ParquetFile(path)

    columns = [
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_mz",
    ]

    before_counts = []
    after_counts = []

    zero_after = 0
    max_after = 0
    intensity_max_errors = 0
    unsorted_after = 0

    example_rows = []

    processed = 0

    for batch in parquet_file.iter_batches(
        columns=columns,
        batch_size=1024,
    ):
        df = batch.to_pandas()

        for row in df.itertuples(index=False):

            original_mzs = np.asarray(
                row.ms2_mzs,
                dtype=float,
            )

            original_intensities = np.asarray(
                row.ms2_normalized_intensities,
                dtype=float,
            )

            processed_mzs, processed_intensities = (
                preprocess_spectrum(
                    original_mzs,
                    original_intensities,
                    precursor_mz=row.precursor_mz,
                    config=config,
                )
            )

            before_counts.append(
                len(original_mzs)
            )

            after_counts.append(
                len(processed_mzs)
            )

            if len(processed_mzs) == 0:
                zero_after += 1

            max_after = max(
                max_after,
                len(processed_mzs),
            )

            if len(processed_intensities) > 0:
                if not np.isclose(
                    processed_intensities.max(),
                    1.0,
                ):
                    intensity_max_errors += 1

            if len(processed_mzs) > 1:
                if np.any(
                    np.diff(processed_mzs) < 0
                ):
                    unsorted_after += 1

            if len(example_rows) < 5:
                example_rows.append(
                    {
                        "precursor_mz": row.precursor_mz,
                        "before": len(original_mzs),
                        "after": len(processed_mzs),
                        "first_mz_before": (
                            original_mzs[:5].tolist()
                        ),
                        "first_intensity_before": (
                            original_intensities[:5].tolist()
                        ),
                        "first_mz_after": (
                            processed_mzs[:5].tolist()
                        ),
                        "first_intensity_after": (
                            processed_intensities[:5].tolist()
                        ),
                    }
                )

            processed += 1

            if processed >= max_rows:
                break

        if processed >= max_rows:
            break

    total_before = sum(before_counts)
    total_after = sum(after_counts)

    reduction = (
        1 - total_after / total_before
    ) * 100

    print_section(name)

    print(f"Spectra checked: {processed:,}")

    print("\nPeak count statistics")

    print(
        f"Mean before:   "
        f"{mean(before_counts):.2f}"
    )

    print(
        f"Median before: "
        f"{median(before_counts):.2f}"
    )

    print(
        f"Mean after:    "
        f"{mean(after_counts):.2f}"
    )

    print(
        f"Median after:  "
        f"{median(after_counts):.2f}"
    )

    print(
        f"Peak reduction: "
        f"{reduction:.2f}%"
    )

    print(
        f"Maximum processed peaks: "
        f"{max_after}"
    )

    print("\nIntegrity checks")

    print(
        f"Empty spectra after preprocessing: "
        f"{zero_after:,}"
    )

    print(
        f"Spectra with non-normalized max intensity: "
        f"{intensity_max_errors:,}"
    )

    print(
        f"Spectra with unsorted processed m/z: "
        f"{unsorted_after:,}"
    )

    print("\nRepresentative examples")

    for index, example in enumerate(
        example_rows,
        start=1,
    ):

        print(
            f"\nExample {index}"
        )

        print(
            f"Precursor m/z: "
            f"{example['precursor_mz']}"
        )

        print(
            f"Peaks: "
            f"{example['before']} "
            f"-> "
            f"{example['after']}"
        )

        print(
            "First m/z before:"
        )

        print(
            example["first_mz_before"]
        )

        print(
            "First intensities before:"
        )

        print(
            example[
                "first_intensity_before"
            ]
        )

        print(
            "First m/z after:"
        )

        print(
            example["first_mz_after"]
        )

        print(
            "First intensities after:"
        )

        print(
            example[
                "first_intensity_after"
            ]
        )


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 3 STEP 5"
    )

    print("\nBaseline preprocessing profile:")

    config = load_preprocessing_config(
        CONFIG_PATH,
        profile="baseline",
    )

    print(config)

    validate_dataset(
        TRAIN_PATH,
        "TRAIN BASELINE VALIDATION",
        max_rows=50_000,
    )

    validate_dataset(
        TEST_PATH,
        "TEST BASELINE VALIDATION",
        max_rows=10_000,
    )

    print_section(
        "STAGE 3 STEP 5 COMPLETE"
    )


if __name__ == "__main__":
    main()