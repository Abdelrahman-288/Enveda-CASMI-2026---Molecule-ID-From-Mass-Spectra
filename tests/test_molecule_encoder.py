import torch
import pytest

from casmi.models.molecule_encoder import (
    MoleculeEncoder,
)


def test_output_shape():

    model = MoleculeEncoder()

    fingerprints = torch.randint(
        low=0,
        high=2,
        size=(
            16,
            2048,
        ),
        dtype=torch.uint8,
    )

    embeddings = model(
        fingerprints
    )

    assert embeddings.shape == (
        16,
        128,
    )


def test_output_is_normalized():

    model = MoleculeEncoder()

    fingerprints = torch.randint(
        low=0,
        high=2,
        size=(
            32,
            2048,
        ),
        dtype=torch.uint8,
    )

    embeddings = model(
        fingerprints
    )

    norms = torch.linalg.vector_norm(
        embeddings,
        dim=1,
    )

    assert torch.allclose(
        norms,
        torch.ones_like(
            norms
        ),
        atol=1e-5,
    )


def test_zero_input_is_finite():

    model = MoleculeEncoder()

    fingerprints = torch.zeros(
        (
            8,
            2048,
        ),
        dtype=torch.float32,
    )

    embeddings = model(
        fingerprints
    )

    assert torch.isfinite(
        embeddings
    ).all()


def test_gradient_flow():

    model = MoleculeEncoder()

    fingerprints = torch.rand(
        (
            4,
            2048,
        ),
        requires_grad=True,
    )

    embeddings = model(
        fingerprints
    )

    loss = embeddings.pow(
        2
    ).mean()

    loss.backward()

    assert (
        fingerprints.grad
        is not None
    )

    assert torch.isfinite(
        fingerprints.grad
    ).all()


def test_invalid_rank():

    model = MoleculeEncoder()

    fingerprints = torch.zeros(
        2048
    )

    with pytest.raises(
        ValueError
    ):
        model(
            fingerprints
        )


def test_invalid_dimension():

    model = MoleculeEncoder()

    fingerprints = torch.zeros(
        (
            4,
            1024,
        )
    )

    with pytest.raises(
        ValueError
    ):
        model(
            fingerprints
        )


def test_custom_dimensions():

    model = MoleculeEncoder(
        fingerprint_dim=512,
        hidden_dim_1=256,
        hidden_dim_2=128,
        embedding_dim=64,
    )

    fingerprints = torch.randint(
        low=0,
        high=2,
        size=(
            10,
            512,
        ),
        dtype=torch.uint8,
    )

    embeddings = model(
        fingerprints
    )

    assert embeddings.shape == (
        10,
        64,
    )