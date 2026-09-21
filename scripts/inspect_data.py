from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = DATASET_DIR / "train.parquet"
TEST_PATH = DATASET_DIR / "test.parquet"
SUBMISSION_PATH = DATASET_DIR / "sample_submission.csv"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def file_size_gb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 3)


def inspect_parquet(path: Path, name: str) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    parquet_file = pq.ParquetFile(path)

    metadata = parquet_file.metadata

    print(f"Path: {path}")
    print(f"Size: {file_size_gb(path):.3f} GB")
    print(f"Rows: {metadata.num_rows:,}")
    print(f"Columns: {metadata.num_columns}")
    print(f"Row groups: {metadata.num_row_groups:,}")

    print("\nSchema:")
    print(parquet_file.schema_arrow)


def inspect_small_sample(path: Path, name: str, rows: int = 5) -> None:
    print("\n" + "-" * 80)
    print(f"{name} SAMPLE")
    print("-" * 80)

    parquet_file = pq.ParquetFile(path)

    first_batch = next(
        parquet_file.iter_batches(batch_size=rows)
    )

    df = first_batch.to_pandas()

    print(df.head(rows).to_string())


def inspect_submission() -> None:
    print("\n" + "=" * 80)
    print("SAMPLE SUBMISSION")
    print("=" * 80)

    df = pd.read_csv(SUBMISSION_PATH)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    print("\nFirst rows:")
    print(df.head().to_string(index=False))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print("\nENVEDA CASMI 2026 — DATASET INSPECTION")

    required_files = [
        TRAIN_PATH,
        TEST_PATH,
        SUBMISSION_PATH,
    ]

    print("\nChecking required files...")

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

        print(f"[OK] {path.name}")

    inspect_parquet(
        TRAIN_PATH,
        "TRAIN DATASET",
    )

    inspect_parquet(
        TEST_PATH,
        "TEST DATASET",
    )

    inspect_small_sample(
        TRAIN_PATH,
        "TRAIN",
    )

    inspect_small_sample(
        TEST_PATH,
        "TEST",
    )

    inspect_submission()

    print("\n" + "=" * 80)
    print("STAGE 1 INITIAL INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()