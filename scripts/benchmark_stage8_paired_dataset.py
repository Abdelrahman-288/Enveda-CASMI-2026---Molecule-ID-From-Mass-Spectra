from __future__ import annotations

from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from casmi.training.stage8_dataset import (
    SpectrumMoleculeDataset,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

STAGE6_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage6"
)

STAGE8_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage8"
)


BATCH_SIZE = 512

NUM_WORKERS = 0

BENCHMARK_BATCHES = 200


def main():

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 8 PAIRED DATASET BENCHMARK"
    )

    dataset = SpectrumMoleculeDataset(
        stage6_dir=STAGE6_DIR,
        stage8_dir=STAGE8_DIR,
        split_value=0,
    )

    print(
        f"\nTraining samples: "
        f"{len(dataset):,}"
    )

    print(
        f"Max peaks: "
        f"{dataset.max_peaks}"
    )

    print(
        f"Fingerprint dim: "
        f"{dataset.fingerprint_dim}"
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    start = time.perf_counter()

    samples = 0
    batches = 0

    last_batch = None

    for batch in loader:

        last_batch = batch

        samples += int(
            batch[
                "mzs"
            ].shape[0]
        )

        batches += 1

        if (
            batches
            >= BENCHMARK_BATCHES
        ):

            break

    elapsed = (
        time.perf_counter()
        - start
    )

    throughput = (
        samples
        / elapsed
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "PAIRED DATASET BENCHMARK"
    )

    print(
        "=" * 90
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE:,}"
    )

    print(
        f"Workers: "
        f"{NUM_WORKERS}"
    )

    print(
        f"Batches: "
        f"{batches:,}"
    )

    print(
        f"Samples loaded: "
        f"{samples:,}"
    )

    print(
        f"Elapsed: "
        f"{elapsed:.3f} s"
    )

    print(
        f"Throughput: "
        f"{throughput:,.0f} samples/s"
    )

    print(
        "\nBatch shapes:"
    )

    print(
        "mzs:",
        tuple(
            last_batch[
                "mzs"
            ].shape
        ),
    )

    print(
        "intensities:",
        tuple(
            last_batch[
                "intensities"
            ].shape
        ),
    )

    print(
        "mask:",
        tuple(
            last_batch[
                "mask"
            ].shape
        ),
    )

    print(
        "precursor_mz:",
        tuple(
            last_batch[
                "precursor_mz"
            ].shape
        ),
    )

    print(
        "fingerprint:",
        tuple(
            last_batch[
                "fingerprint"
            ].shape
        ),
    )

    print(
        "molecule_index:",
        tuple(
            last_batch[
                "molecule_index"
            ].shape
        ),
    )

    print(
        "\n"
        + "=" * 90
    )


if __name__ == "__main__":
    main()