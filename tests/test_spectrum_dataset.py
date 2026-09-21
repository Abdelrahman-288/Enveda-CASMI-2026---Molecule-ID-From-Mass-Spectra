from pathlib import Path

import numpy as np
import torch

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


def test_train_dataset_length():

    dataset = CachedSpectrumDataset(
        CACHE_DIR,
        split="train",
    )

    assert len(dataset) == 1_589_761


def test_validation_dataset_length():

    dataset = CachedSpectrumDataset(
        CACHE_DIR,
        split="validation",
    )

    assert len(dataset) == 129_408


def test_item_shapes():

    dataset = CachedSpectrumDataset(
        CACHE_DIR,
        split="train",
    )

    item = dataset[0]

    assert item["peaks"].shape == (
        128,
        2,
    )

    assert item["mask"].shape == (
        128,
    )

    assert item["peaks"].dtype == (
        torch.float32
    )

    assert item["mask"].dtype == (
        torch.bool
    )

    assert item["label"].dtype == (
        torch.long
    )


def test_mask_has_valid_peaks():

    dataset = CachedSpectrumDataset(
        CACHE_DIR,
        split="train",
    )

    item = dataset[0]

    assert (
        item["mask"]
        .sum()
        .item()
        > 0
    )


def test_padded_values_are_zero():

    dataset = CachedSpectrumDataset(
        CACHE_DIR,
        split="train",
    )

    item = dataset[0]

    peaks = item[
        "peaks"
    ].numpy()

    mask = item[
        "mask"
    ].numpy()

    padded = peaks[
        ~mask
    ]

    if len(padded) > 0:
        assert np.allclose(
            padded,
            0.0,
        )


def test_train_validation_structure_disjoint():

    train_dataset = (
        CachedSpectrumDataset(
            CACHE_DIR,
            split="train",
        )
    )

    validation_dataset = (
        CachedSpectrumDataset(
            CACHE_DIR,
            split="validation",
        )
    )

    train_labels = set(
        np.asarray(
            train_dataset.labels[
                train_dataset.indices
            ]
        ).tolist()
    )

    validation_labels = set(
        np.asarray(
            validation_dataset.labels[
                validation_dataset.indices
            ]
        ).tolist()
    )

    overlap = (
        train_labels
        & validation_labels
    )

    assert len(overlap) == 0