import pytest
import torch

from casmi.training.cross_modal import (
    multi_positive_cross_modal_loss,
)


def test_loss_is_finite():

    spectrum = torch.randn(
        8,
        16,
    )

    molecule = torch.randn(
        8,
        16,
    )

    spectrum = torch.nn.functional.normalize(
        spectrum,
        dim=1,
    )

    molecule = torch.nn.functional.normalize(
        molecule,
        dim=1,
    )

    ids = torch.arange(
        8
    )

    loss, metrics = (
        multi_positive_cross_modal_loss(
            spectrum,
            molecule,
            ids,
        )
    )

    assert torch.isfinite(
        loss
    )

    assert torch.isfinite(
        metrics[
            "loss_s2m"
        ]
    )


def test_perfect_alignment():

    embeddings = torch.eye(
        8
    )

    ids = torch.arange(
        8
    )

    loss, metrics = (
        multi_positive_cross_modal_loss(
            embeddings,
            embeddings,
            ids,
            temperature=0.07,
        )
    )

    assert (
        metrics[
            "s2m_top1"
        ].item()
        == 1.0
    )

    assert (
        metrics[
            "m2s_top1"
        ].item()
        == 1.0
    )

    assert loss.item() < 0.01


def test_duplicate_molecule_ids():

    spectrum = torch.randn(
        6,
        32,
    )

    molecule = torch.randn(
        6,
        32,
    )

    spectrum = torch.nn.functional.normalize(
        spectrum,
        dim=1,
    )

    molecule = torch.nn.functional.normalize(
        molecule,
        dim=1,
    )

    ids = torch.tensor(
        [
            0,
            0,
            1,
            1,
            2,
            2,
        ]
    )

    loss, metrics = (
        multi_positive_cross_modal_loss(
            spectrum,
            molecule,
            ids,
        )
    )

    assert torch.isfinite(
        loss
    )

    assert (
        metrics[
            "unique_molecules"
        ].item()
        == 3
    )


def test_gradient_flow():

    spectrum = torch.randn(
        8,
        32,
        requires_grad=True,
    )

    molecule = torch.randn(
        8,
        32,
        requires_grad=True,
    )

    spectrum_norm = (
        torch.nn.functional.normalize(
            spectrum,
            dim=1,
        )
    )

    molecule_norm = (
        torch.nn.functional.normalize(
            molecule,
            dim=1,
        )
    )

    ids = torch.arange(
        8
    )

    loss, _ = (
        multi_positive_cross_modal_loss(
            spectrum_norm,
            molecule_norm,
            ids,
        )
    )

    loss.backward()

    assert spectrum.grad is not None
    assert molecule.grad is not None

    assert torch.isfinite(
        spectrum.grad
    ).all()

    assert torch.isfinite(
        molecule.grad
    ).all()


def test_invalid_temperature():

    spectrum = torch.randn(
        4,
        8,
    )

    molecule = torch.randn(
        4,
        8,
    )

    ids = torch.arange(
        4
    )

    with pytest.raises(
        ValueError
    ):

        multi_positive_cross_modal_loss(
            spectrum,
            molecule,
            ids,
            temperature=0.0,
        )


def test_shape_mismatch():

    spectrum = torch.randn(
        4,
        8,
    )

    molecule = torch.randn(
        4,
        16,
    )

    ids = torch.arange(
        4
    )

    with pytest.raises(
        ValueError
    ):

        multi_positive_cross_modal_loss(
            spectrum,
            molecule,
            ids,
        )