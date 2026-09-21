from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


class StructureBatchSampler(
    Sampler[list[int]]
):
    """
    Build metric-learning batches containing multiple spectra
    from each molecular structure.

    Example
    -------
    structures_per_batch = 128
    spectra_per_structure = 4

    produces:

        128 structures
        x
        4 spectra
        =
        512 spectra per batch

    Each anchor therefore has positive examples belonging to
    the same molecular structure.
    """

    def __init__(
        self,
        labels,
        structures_per_batch: int = 128,
        spectra_per_structure: int = 4,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ):
        if structures_per_batch < 2:
            raise ValueError(
                "structures_per_batch must be >= 2."
            )

        if spectra_per_structure < 2:
            raise ValueError(
                "spectra_per_structure must be >= 2 "
                "for contrastive learning."
            )

        self.labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        self.structures_per_batch = int(
            structures_per_batch
        )

        self.spectra_per_structure = int(
            spectra_per_structure
        )

        self.batch_size = (
            self.structures_per_batch
            * self.spectra_per_structure
        )

        self.seed = int(
            seed
        )

        self.epoch = 0

        # =================================================
        # Build structure -> dataset positions mapping
        # =================================================

        structure_to_indices = defaultdict(
            list
        )

        for dataset_index, label in enumerate(
            self.labels
        ):
            structure_to_indices[
                int(label)
            ].append(
                dataset_index
            )

        # =================================================
        # Retain structures with at least two spectra
        #
        # Structures with fewer than spectra_per_structure
        # can still be sampled with replacement.
        # =================================================

        self.structure_to_indices = {
            structure: np.asarray(
                indices,
                dtype=np.int64,
            )
            for structure, indices
            in structure_to_indices.items()
            if len(indices) >= 2
        }

        self.structures = np.asarray(
            list(
                self.structure_to_indices.keys()
            ),
            dtype=np.int64,
        )

        if (
            len(self.structures)
            < self.structures_per_batch
        ):
            raise ValueError(
                "Not enough eligible structures "
                "to construct one batch."
            )

        # =================================================
        # Default epoch length
        # =================================================

        if batches_per_epoch is None:

            batches_per_epoch = math.ceil(
                len(self.labels)
                / self.batch_size
            )

        self.batches_per_epoch = int(
            batches_per_epoch
        )

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        """
        Change deterministic random state between epochs.
        """

        self.epoch = int(
            epoch
        )

    def __len__(
        self,
    ) -> int:
        return self.batches_per_epoch

    def __iter__(
        self,
    ) -> Iterator[list[int]]:

        rng = np.random.default_rng(
            self.seed
            + self.epoch
        )

        for _ in range(
            self.batches_per_epoch
        ):

            # ---------------------------------------------
            # Choose distinct molecular structures.
            # ---------------------------------------------

            chosen_structures = rng.choice(
                self.structures,
                size=self.structures_per_batch,
                replace=False,
            )

            batch_indices = []

            for structure in chosen_structures:

                candidate_indices = (
                    self.structure_to_indices[
                        int(structure)
                    ]
                )

                # -----------------------------------------
                # Use replacement only when a structure
                # contains fewer spectra than requested.
                # -----------------------------------------

                replace = (
                    len(candidate_indices)
                    < self.spectra_per_structure
                )

                selected = rng.choice(
                    candidate_indices,
                    size=self.spectra_per_structure,
                    replace=replace,
                )

                batch_indices.extend(
                    selected.tolist()
                )

            # Shuffle order inside the batch so examples
            # from one structure do not remain adjacent.
            rng.shuffle(
                batch_indices
            )

            yield batch_indices


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive loss for L2-normalized embeddings.

    Positive pairs:
        spectra sharing the same structure label.

    Negative pairs:
        spectra belonging to different structures.

    Parameters
    ----------
    embeddings:
        Tensor [batch, embedding_dim]

    labels:
        Tensor [batch]

    temperature:
        Softmax temperature.

    Returns
    -------
    Scalar loss.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must have shape "
            "[batch, embedding_dim]."
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels must have shape [batch]."
        )

    if (
        embeddings.shape[0]
        != labels.shape[0]
    ):
        raise ValueError(
            "embeddings and labels must "
            "have matching batch sizes."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be > 0."
        )

    # =====================================================
    # Perform the similarity calculation in FP32.
    #
    # The encoder can still run under FP16 autocast.
    # =====================================================

    embeddings = embeddings.float()

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1,
        eps=1e-8,
    )

    labels = labels.view(
        -1
    )

    batch_size = (
        embeddings.shape[0]
    )

    # =====================================================
    # Pairwise cosine similarities
    # =====================================================

    logits = (
        embeddings
        @ embeddings.T
    )

    logits = (
        logits
        / temperature
    )

    # Numerical stability.
    logits = (
        logits
        - logits.max(
            dim=1,
            keepdim=True,
        ).values.detach()
    )

    # =====================================================
    # Masks
    # =====================================================

    device = (
        embeddings.device
    )

    identity_mask = torch.eye(
        batch_size,
        dtype=torch.bool,
        device=device,
    )

    positive_mask = (
        labels.unsqueeze(0)
        == labels.unsqueeze(1)
    )

    # Remove self-comparisons.
    positive_mask = (
        positive_mask
        & ~identity_mask
    )

    comparison_mask = (
        ~identity_mask
    )

    # =====================================================
    # Log probability
    # =====================================================

    exp_logits = torch.exp(
        logits
    )

    exp_logits = (
        exp_logits
        * comparison_mask.to(
            exp_logits.dtype
        )
    )

    denominator = (
        exp_logits.sum(
            dim=1,
            keepdim=True,
        )
        .clamp_min(
            1e-12
        )
    )

    log_probability = (
        logits
        - torch.log(
            denominator
        )
    )

    # =====================================================
    # Average across positive pairs for each anchor
    # =====================================================

    positive_count = (
        positive_mask.sum(
            dim=1
        )
    )

    valid_anchor = (
        positive_count > 0
    )

    if not valid_anchor.any():
        raise ValueError(
            "No positive pairs exist in this batch. "
            "Use a structure-aware batch sampler."
        )

    positive_log_probability = (
        (
            log_probability
            * positive_mask.to(
                log_probability.dtype
            )
        )
        .sum(
            dim=1
        )
    )

    mean_positive_log_probability = (
        positive_log_probability[
            valid_anchor
        ]
        / positive_count[
            valid_anchor
        ].to(
            log_probability.dtype
        )
    )

    loss = (
        -mean_positive_log_probability.mean()
    )

    return loss