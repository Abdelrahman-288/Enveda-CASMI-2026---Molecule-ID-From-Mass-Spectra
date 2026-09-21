from pathlib import Path
from collections import Counter

import numpy as np
from torch.utils.data import DataLoader

from casmi.models.spectrum_dataset import (
    CachedSpectrumDataset,
)
from casmi.training.metric_learning import (
    StructureBatchSampler,
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


STRUCTURES_PER_BATCH = 128
SPECTRA_PER_STRUCTURE = 4

NUM_TEST_BATCHES = 10


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 6 STEP 5 "
        "STRUCTURE-AWARE BATCH VALIDATION"
    )

    dataset = CachedSpectrumDataset(
        CACHE_DIR,
        split="train",
    )

    # Labels corresponding specifically to dataset-local
    # positions.
    labels = np.asarray(
        dataset.labels[
            dataset.indices
        ],
        dtype=np.int64,
    )

    print(
        f"\nTraining spectra: "
        f"{len(dataset):,}"
    )

    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    eligible = np.sum(
        counts >= 2
    )

    print(
        f"Training structures: "
        f"{len(unique_labels):,}"
    )

    print(
        "Structures with >=2 spectra: "
        f"{eligible:,}"
    )

    sampler = StructureBatchSampler(
        labels=labels,
        structures_per_batch=(
            STRUCTURES_PER_BATCH
        ),
        spectra_per_structure=(
            SPECTRA_PER_STRUCTURE
        ),
        batches_per_epoch=(
            NUM_TEST_BATCHES
        ),
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    expected_batch_size = (
        STRUCTURES_PER_BATCH
        * SPECTRA_PER_STRUCTURE
    )

    print(
        f"\nStructures/batch: "
        f"{STRUCTURES_PER_BATCH}"
    )

    print(
        f"Spectra/structure: "
        f"{SPECTRA_PER_STRUCTURE}"
    )

    print(
        f"Expected batch size: "
        f"{expected_batch_size}"
    )

    print(
        "\nChecking batches..."
    )

    all_valid = True

    for batch_number, batch in enumerate(
        loader,
        start=1,
    ):

        batch_labels = (
            batch[
                "label"
            ]
            .numpy()
        )

        counts = Counter(
            batch_labels.tolist()
        )

        unique_structure_count = (
            len(counts)
        )

        min_count = min(
            counts.values()
        )

        max_count = max(
            counts.values()
        )

        valid = (
            len(batch_labels)
            == expected_batch_size
            and unique_structure_count
            == STRUCTURES_PER_BATCH
            and min_count
            == SPECTRA_PER_STRUCTURE
            and max_count
            == SPECTRA_PER_STRUCTURE
        )

        all_valid = (
            all_valid
            and valid
        )

        print(
            f"Batch {batch_number:02d} | "
            f"size={len(batch_labels)} | "
            f"structures={unique_structure_count} | "
            f"min/group={min_count} | "
            f"max/group={max_count} | "
            f"valid={valid}"
        )

        if (
            batch_number
            >= NUM_TEST_BATCHES
        ):
            break

    print(
        "\n"
        + "=" * 90
    )

    print(
        "STRUCTURE-AWARE BATCH RESULT"
    )

    print(
        "=" * 90
    )

    print(
        f"All batches valid: "
        f"{all_valid}"
    )

    if not all_valid:
        raise RuntimeError(
            "Structure-aware batch validation failed."
        )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "STAGE 6 STEP 5 COMPLETE"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()