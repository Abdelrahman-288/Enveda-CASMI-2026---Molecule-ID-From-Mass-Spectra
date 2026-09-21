import numpy as np
import pytest
import torch

from casmi.training.metric_learning import (
    StructureBatchSampler,
    supervised_contrastive_loss,
)


def test_structure_batch_sampler_size():

    labels = np.repeat(
        np.arange(20),
        5,
    )

    sampler = StructureBatchSampler(
        labels=labels,
        structures_per_batch=4,
        spectra_per_structure=3,
        batches_per_epoch=2,
        seed=42,
    )

    batch = next(
        iter(
            sampler
        )
    )

    assert len(batch) == 12


def test_structure_batch_sampler_positive_counts():

    labels = np.repeat(
        np.arange(20),
        5,
    )

    sampler = StructureBatchSampler(
        labels=labels,
        structures_per_batch=4,
        spectra_per_structure=3,
        batches_per_epoch=1,
        seed=42,
    )

    batch = next(
        iter(
            sampler
        )
    )

    batch_labels = labels[
        batch
    ]

    unique_labels, counts = np.unique(
        batch_labels,
        return_counts=True,
    )

    assert len(unique_labels) == 4

    assert np.all(
        counts == 3
    )


def test_sampler_is_deterministic():

    labels = np.repeat(
        np.arange(20),
        5,
    )

    sampler_a = StructureBatchSampler(
        labels=labels,
        structures_per_batch=4,
        spectra_per_structure=3,
        batches_per_epoch=1,
        seed=42,
    )

    sampler_b = StructureBatchSampler(
        labels=labels,
        structures_per_batch=4,
        spectra_per_structure=3,
        batches_per_epoch=1,
        seed=42,
    )

    batch_a = next(
        iter(
            sampler_a
        )
    )

    batch_b = next(
        iter(
            sampler_b
        )
    )

    assert batch_a == batch_b


def test_sampler_epoch_changes():

    labels = np.repeat(
        np.arange(20),
        5,
    )

    sampler = StructureBatchSampler(
        labels=labels,
        structures_per_batch=4,
        spectra_per_structure=3,
        batches_per_epoch=1,
        seed=42,
    )

    sampler.set_epoch(
        0
    )

    batch_epoch_0 = next(
        iter(
            sampler
        )
    )

    sampler.set_epoch(
        1
    )

    batch_epoch_1 = next(
        iter(
            sampler
        )
    )

    assert batch_epoch_0 != batch_epoch_1


def test_contrastive_loss_is_finite():

    torch.manual_seed(42)

    embeddings = torch.randn(
        12,
        32,
    )

    embeddings = torch.nn.functional.normalize(
        embeddings,
        dim=1,
    )

    labels = torch.tensor(
        [
            0, 0, 0,
            1, 1, 1,
            2, 2, 2,
            3, 3, 3,
        ]
    )

    loss = supervised_contrastive_loss(
        embeddings,
        labels,
    )

    assert torch.isfinite(
        loss
    )

    assert loss.item() > 0


def test_contrastive_loss_backward():

    torch.manual_seed(42)

    raw_embeddings = torch.randn(
        12,
        32,
        requires_grad=True,
    )

    embeddings = torch.nn.functional.normalize(
        raw_embeddings,
        dim=1,
    )

    labels = torch.tensor(
        [
            0, 0, 0,
            1, 1, 1,
            2, 2, 2,
            3, 3, 3,
        ]
    )

    loss = supervised_contrastive_loss(
        embeddings,
        labels,
    )

    loss.backward()

    assert (
        raw_embeddings.grad
        is not None
    )

    assert torch.isfinite(
        raw_embeddings.grad
    ).all()


def test_loss_rejects_batch_without_positives():

    embeddings = torch.randn(
        4,
        16,
    )

    labels = torch.tensor(
        [
            0,
            1,
            2,
            3,
        ]
    )

    with pytest.raises(
        ValueError
    ):
        supervised_contrastive_loss(
            embeddings,
            labels,
        )