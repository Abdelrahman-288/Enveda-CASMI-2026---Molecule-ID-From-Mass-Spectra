from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from casmi.models.spectrum_dataset import (
    CachedSpectrumDataset,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage6"
)


BATCH_SIZE = 512
NUM_WORKERS = 4
NUM_BATCHES = 200


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 6 STEP 3 "
        "DATALOADER BENCHMARK"
    )

    dataset = CachedSpectrumDataset(
        CACHE_DIR,
        split="train",
    )

    print(
        f"\nTraining spectra: "
        f"{len(dataset):,}"
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        drop_last=True,
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE:,}"
    )

    print(
        f"Workers: "
        f"{NUM_WORKERS}"
    )

    start = time.time()

    spectra_seen = 0

    for batch_number, batch in enumerate(
        loader,
        start=1,
    ):

        spectra_seen += (
            batch[
                "peaks"
            ].shape[0]
        )

        if batch_number == 1:

            print(
                "\nFirst batch:"
            )

            print(
                "peaks:",
                batch[
                    "peaks"
                ].shape,
                batch[
                    "peaks"
                ].dtype,
            )

            print(
                "mask:",
                batch[
                    "mask"
                ].shape,
                batch[
                    "mask"
                ].dtype,
            )

            print(
                "labels:",
                batch[
                    "label"
                ].shape,
            )

            print(
                "precursor:",
                batch[
                    "precursor_mz"
                ].shape,
            )

        if (
            batch_number
            >= NUM_BATCHES
        ):
            break

    elapsed = (
        time.time()
        - start
    )

    rate = (
        spectra_seen
        / elapsed
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DATALOADER RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"Batches: "
        f"{NUM_BATCHES:,}"
    )

    print(
        f"Spectra loaded: "
        f"{spectra_seen:,}"
    )

    print(
        f"Elapsed: "
        f"{elapsed:.2f} s"
    )

    print(
        f"Throughput: "
        f"{rate:,.0f} spectra/s"
    )

    print(
        "\nCUDA available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STAGE 6 STEP 3 COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()