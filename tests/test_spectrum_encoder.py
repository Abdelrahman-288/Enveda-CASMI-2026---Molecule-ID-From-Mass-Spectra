import torch

from casmi.models.spectrum_encoder import (
    SpectrumEncoder,
)


def make_batch(
    batch_size=4,
    max_peaks=128,
):
    torch.manual_seed(42)

    peaks = torch.rand(
        batch_size,
        max_peaks,
        2,
    )

    mask = torch.ones(
        batch_size,
        max_peaks,
        dtype=torch.bool,
    )

    precursor = torch.rand(
        batch_size
    )

    return (
        peaks,
        mask,
        precursor,
    )


def test_embedding_shape():

    model = SpectrumEncoder(
        embedding_dim=128
    )

    peaks, mask, precursor = (
        make_batch()
    )

    embeddings = model(
        peaks,
        mask,
        precursor,
    )

    assert embeddings.shape == (
        4,
        128,
    )


def test_embeddings_are_normalized():

    model = SpectrumEncoder()

    peaks, mask, precursor = (
        make_batch()
    )

    embeddings = model(
        peaks,
        mask,
        precursor,
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


def test_padding_does_not_change_embedding():

    model = SpectrumEncoder(
        dropout=0.0
    )

    model.eval()

    torch.manual_seed(42)

    valid_peaks = torch.rand(
        1,
        10,
        2,
    )

    padded_peaks = torch.zeros(
        1,
        128,
        2,
    )

    padded_peaks[
        :,
        :10,
    ] = valid_peaks

    padded_mask = torch.zeros(
        1,
        128,
        dtype=torch.bool,
    )

    padded_mask[
        :,
        :10,
    ] = True

    compact_peaks = (
        valid_peaks
    )

    compact_mask = torch.ones(
        1,
        10,
        dtype=torch.bool,
    )

    precursor = torch.tensor(
        [0.35],
        dtype=torch.float32,
    )

    with torch.no_grad():

        padded_embedding = model(
            padded_peaks,
            padded_mask,
            precursor,
        )

        compact_embedding = model(
            compact_peaks,
            compact_mask,
            precursor,
        )

    assert torch.allclose(
        padded_embedding,
        compact_embedding,
        atol=1e-5,
    )


def test_different_spectra_produce_different_embeddings():

    model = SpectrumEncoder(
        dropout=0.0
    )

    model.eval()

    peaks_a = torch.zeros(
        1,
        128,
        2,
    )

    peaks_b = torch.zeros(
        1,
        128,
        2,
    )

    mask = torch.zeros(
        1,
        128,
        dtype=torch.bool,
    )

    mask[
        :,
        :2,
    ] = True

    peaks_a[
        0,
        0,
    ] = torch.tensor(
        [0.10, 1.00]
    )

    peaks_a[
        0,
        1,
    ] = torch.tensor(
        [0.20, 0.50]
    )

    peaks_b[
        0,
        0,
    ] = torch.tensor(
        [0.60, 0.20]
    )

    peaks_b[
        0,
        1,
    ] = torch.tensor(
        [0.80, 1.00]
    )

    precursor = torch.tensor(
        [0.35]
    )

    with torch.no_grad():

        embedding_a = model(
            peaks_a,
            mask,
            precursor,
        )

        embedding_b = model(
            peaks_b,
            mask,
            precursor,
        )

    assert not torch.allclose(
        embedding_a,
        embedding_b,
    )


def test_backward_pass():

    model = SpectrumEncoder()

    peaks, mask, precursor = (
        make_batch()
    )

    embeddings = model(
        peaks,
        mask,
        precursor,
    )

    loss = (
        embeddings ** 2
    ).sum()

    loss.backward()

    gradients_exist = any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    assert gradients_exist


def test_all_padding_does_not_produce_nan():

    model = SpectrumEncoder()

    peaks = torch.zeros(
        2,
        128,
        2,
    )

    mask = torch.zeros(
        2,
        128,
        dtype=torch.bool,
    )

    precursor = torch.tensor(
        [
            0.3,
            0.4,
        ]
    )

    embeddings = model(
        peaks,
        mask,
        precursor,
    )

    assert torch.isfinite(
        embeddings
    ).all()