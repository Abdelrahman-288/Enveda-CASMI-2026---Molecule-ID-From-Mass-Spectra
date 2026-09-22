from __future__ import annotations

import torch


def multi_positive_cross_modal_loss(
    spectrum_embeddings: torch.Tensor,
    molecule_embeddings: torch.Tensor,
    molecule_ids: torch.Tensor,
    temperature: float = 0.07,
):
    """
    Symmetric multi-positive contrastive loss.

    Multiple rows in a batch may correspond to the same molecule.
    All rows with the same molecule_id are treated as positives.

    Parameters
    ----------
    spectrum_embeddings:
        [B, D]

    molecule_embeddings:
        [B, D]

    molecule_ids:
        [B]

    temperature:
        Contrastive temperature.

    Returns
    -------
    loss:
        Scalar tensor.

    metrics:
        Dictionary containing detached diagnostic metrics.
    """

    if spectrum_embeddings.ndim != 2:
        raise ValueError(
            "spectrum_embeddings must have shape [B, D]."
        )

    if molecule_embeddings.ndim != 2:
        raise ValueError(
            "molecule_embeddings must have shape [B, D]."
        )

    if molecule_ids.ndim != 1:
        raise ValueError(
            "molecule_ids must have shape [B]."
        )

    if (
        spectrum_embeddings.shape
        != molecule_embeddings.shape
    ):
        raise ValueError(
            "Spectrum and molecule embeddings "
            "must have identical shape."
        )

    batch_size = (
        spectrum_embeddings.shape[0]
    )

    if molecule_ids.shape[0] != batch_size:
        raise ValueError(
            "molecule_ids length must match batch size."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be > 0."
        )

    # Use FP32 for numerical stability even when
    # encoders run under BF16 autocast.
    spectrum_embeddings = (
        spectrum_embeddings.float()
    )

    molecule_embeddings = (
        molecule_embeddings.float()
    )

    logits = (
        spectrum_embeddings
        @ molecule_embeddings.T
    ) / float(
        temperature
    )

    positive_mask = (
        molecule_ids[:, None]
        == molecule_ids[None, :]
    )

    # -----------------------------------------------------
    # Spectrum -> Molecule
    # -----------------------------------------------------

    positive_logits_s2m = logits.masked_fill(
        ~positive_mask,
        float("-inf"),
    )

    numerator_s2m = torch.logsumexp(
        positive_logits_s2m,
        dim=1,
    )

    denominator_s2m = torch.logsumexp(
        logits,
        dim=1,
    )

    loss_s2m = -(
        numerator_s2m
        - denominator_s2m
    ).mean()

    # -----------------------------------------------------
    # Molecule -> Spectrum
    # -----------------------------------------------------

    logits_m2s = logits.T

    positive_mask_m2s = (
        positive_mask.T
    )

    positive_logits_m2s = (
        logits_m2s.masked_fill(
            ~positive_mask_m2s,
            float("-inf"),
        )
    )

    numerator_m2s = torch.logsumexp(
        positive_logits_m2s,
        dim=1,
    )

    denominator_m2s = torch.logsumexp(
        logits_m2s,
        dim=1,
    )

    loss_m2s = -(
        numerator_m2s
        - denominator_m2s
    ).mean()

    loss = (
        loss_s2m
        + loss_m2s
    ) / 2.0

    # -----------------------------------------------------
    # Top-1 diagnostics
    # -----------------------------------------------------

    top1_molecule = logits.argmax(
        dim=1
    )

    rows = torch.arange(
        batch_size,
        device=logits.device,
    )

    s2m_correct = positive_mask[
        rows,
        top1_molecule,
    ]

    top1_spectrum = logits_m2s.argmax(
        dim=1
    )

    m2s_correct = positive_mask_m2s[
        rows,
        top1_spectrum,
    ]

    metrics = {
        "loss_s2m": (
            loss_s2m.detach()
        ),
        "loss_m2s": (
            loss_m2s.detach()
        ),
        "s2m_top1": (
            s2m_correct
            .float()
            .mean()
            .detach()
        ),
        "m2s_top1": (
            m2s_correct
            .float()
            .mean()
            .detach()
        ),
        "unique_molecules": torch.tensor(
            molecule_ids.unique().numel(),
            device=logits.device,
        ),
    }

    return loss, metrics