from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class CandidatePoolSelection:
    """
    Result of combining per-spectrum candidate sets.

    mode:
        "intersection"
        "union_fallback"

    candidates:
        Sorted unique candidate structure indices.
    """

    candidates: np.ndarray
    mode: str


def _normalize_candidate_set(
    candidates: Iterable[int],
) -> np.ndarray:
    """
    Convert a candidate iterable into a sorted unique int64 array.
    """

    array = np.asarray(
        list(candidates),
        dtype=np.int64,
    )

    if array.ndim != 1:
        raise ValueError(
            "Candidate set must be one-dimensional."
        )

    if len(array) == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    return np.unique(
        array
    )


def candidate_union(
    candidate_sets: Sequence[Iterable[int]],
) -> np.ndarray:
    """
    Return the sorted union of all candidate sets.
    """

    if len(candidate_sets) == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    normalized = [
        _normalize_candidate_set(
            candidates
        )
        for candidates
        in candidate_sets
    ]

    non_empty = [
        candidates
        for candidates
        in normalized
        if len(candidates) > 0
    ]

    if len(non_empty) == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    return np.unique(
        np.concatenate(
            non_empty
        )
    )


def candidate_intersection(
    candidate_sets: Sequence[Iterable[int]],
) -> np.ndarray:
    """
    Return candidates present in every spectrum-level
    candidate set.
    """

    if len(candidate_sets) == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    normalized = [
        _normalize_candidate_set(
            candidates
        )
        for candidates
        in candidate_sets
    ]

    # If any spectrum has no candidates, the strict
    # intersection is empty.
    if any(
        len(candidates) == 0
        for candidates
        in normalized
    ):
        return np.empty(
            0,
            dtype=np.int64,
        )

    intersection = set(
        normalized[0].tolist()
    )

    for candidates in normalized[1:]:

        intersection.intersection_update(
            candidates.tolist()
        )

        if len(intersection) == 0:
            break

    if len(intersection) == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    return np.asarray(
        sorted(intersection),
        dtype=np.int64,
    )


def select_candidate_pool(
    candidate_sets: Sequence[Iterable[int]],
) -> CandidatePoolSelection:
    """
    Stage 9 candidate strategy.

    1. Use the strict intersection if it is non-empty.
    2. Otherwise fall back to the union.

    This reflects the best validation strategy while
    retaining safety for unseen test cases.
    """

    intersection = candidate_intersection(
        candidate_sets
    )

    if len(intersection) > 0:

        return CandidatePoolSelection(
            candidates=intersection,
            mode="intersection",
        )

    union = candidate_union(
        candidate_sets
    )

    return CandidatePoolSelection(
        candidates=union,
        mode="union_fallback",
    )


def build_availability_mask(
    candidate_sets: Sequence[Iterable[int]],
    candidate_pool: Sequence[int],
) -> np.ndarray:
    """
    Build a boolean matrix indicating which candidate is valid
    for which spectrum.

    Shape:
        [num_spectra, num_candidates]

    This is primarily needed for union fallback mode.

    Example:

        spectrum 1 candidates: [1, 2]
        spectrum 2 candidates: [2, 3]

        candidate pool: [1, 2, 3]

        mask:
            [[True,  True,  False],
             [False, True,  True ]]
    """

    pool = np.asarray(
        candidate_pool,
        dtype=np.int64,
    )

    if pool.ndim != 1:
        raise ValueError(
            "candidate_pool must be one-dimensional."
        )

    mask = np.zeros(
        (
            len(candidate_sets),
            len(pool),
        ),
        dtype=np.bool_,
    )

    if len(pool) == 0:
        return mask

    position_lookup = {
        int(candidate): position
        for position, candidate
        in enumerate(pool.tolist())
    }

    for spectrum_index, candidates in enumerate(
        candidate_sets
    ):

        normalized = _normalize_candidate_set(
            candidates
        )

        for candidate in normalized:

            position = position_lookup.get(
                int(candidate)
            )

            if position is not None:

                mask[
                    spectrum_index,
                    position,
                ] = True

    return mask


def masked_mean_scores(
    score_matrix: torch.Tensor,
    availability_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Average each candidate only across spectra where that
    candidate is valid.

    score_matrix:
        [num_spectra, num_candidates]

    availability_mask:
        same shape, boolean

    Candidates available in zero spectra receive -inf.
    """

    if score_matrix.ndim != 2:
        raise ValueError(
            "score_matrix must be 2D."
        )

    if availability_mask.shape != score_matrix.shape:
        raise ValueError(
            "availability_mask must have the same shape "
            "as score_matrix."
        )

    if availability_mask.dtype != torch.bool:
        raise TypeError(
            "availability_mask must be torch.bool."
        )

    mask_float = availability_mask.to(
        dtype=score_matrix.dtype
    )

    counts = mask_float.sum(
        dim=0
    )

    summed_scores = (
        score_matrix
        * mask_float
    ).sum(
        dim=0
    )

    safe_counts = counts.clamp_min(
        1
    )

    means = (
        summed_scores
        / safe_counts
    )

    means = torch.where(
        counts > 0,
        means,
        torch.full_like(
            means,
            float("-inf"),
        ),
    )

    return means


def aggregate_multispectrum_scores(
    score_matrix: torch.Tensor,
    availability_mask: torch.Tensor | None = None,
    method: str = "mean",
) -> torch.Tensor:
    """
    Aggregate spectrum-level scores into one score per candidate.

    Supported methods:
        mean
        max
        top2_mean
        logsumexp

    For Stage 9 final use, method="mean" is the selected default.

    If availability_mask is supplied, unavailable candidate/spectrum
    pairs are excluded from aggregation.
    """

    if score_matrix.ndim != 2:
        raise ValueError(
            "score_matrix must be 2D."
        )

    if score_matrix.shape[0] == 0:
        raise ValueError(
            "score_matrix must contain at least one spectrum."
        )

    if score_matrix.shape[1] == 0:
        return torch.empty(
            0,
            dtype=score_matrix.dtype,
            device=score_matrix.device,
        )

    if availability_mask is None:

        availability_mask = torch.ones(
            score_matrix.shape,
            dtype=torch.bool,
            device=score_matrix.device,
        )

    else:

        availability_mask = availability_mask.to(
            device=score_matrix.device
        )

        if availability_mask.shape != score_matrix.shape:
            raise ValueError(
                "availability_mask must match score_matrix shape."
            )

        if availability_mask.dtype != torch.bool:
            raise TypeError(
                "availability_mask must be torch.bool."
            )

    if method == "mean":

        return masked_mean_scores(
            score_matrix,
            availability_mask,
        )

    if method == "max":

        masked_scores = score_matrix.masked_fill(
            ~availability_mask,
            float("-inf"),
        )

        return masked_scores.max(
            dim=0
        ).values

    if method == "top2_mean":

        candidate_scores = []

        for candidate_index in range(
            score_matrix.shape[1]
        ):

            valid_scores = score_matrix[
                availability_mask[
                    :,
                    candidate_index,
                ],
                candidate_index,
            ]

            if valid_scores.numel() == 0:

                candidate_scores.append(
                    torch.tensor(
                        float("-inf"),
                        dtype=score_matrix.dtype,
                        device=score_matrix.device,
                    )
                )

                continue

            k = min(
                2,
                valid_scores.numel(),
            )

            top_scores = torch.topk(
                valid_scores,
                k=k,
            ).values

            candidate_scores.append(
                top_scores.mean()
            )

        return torch.stack(
            candidate_scores
        )

    if method == "logsumexp":

        temperature = 0.10

        candidate_scores = []

        for candidate_index in range(
            score_matrix.shape[1]
        ):

            valid_scores = score_matrix[
                availability_mask[
                    :,
                    candidate_index,
                ],
                candidate_index,
            ]

            if valid_scores.numel() == 0:

                candidate_scores.append(
                    torch.tensor(
                        float("-inf"),
                        dtype=score_matrix.dtype,
                        device=score_matrix.device,
                    )
                )

                continue

            aggregated = (
                torch.logsumexp(
                    valid_scores
                    / temperature,
                    dim=0,
                )
                * temperature
            )

            candidate_scores.append(
                aggregated
            )

        return torch.stack(
            candidate_scores
        )

    raise ValueError(
        f"Unknown aggregation method: {method}"
    )


def rank_candidates(
    candidate_indices: Sequence[int],
    scores: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rank candidates from highest score to lowest.

    Returns:
        ranked_candidate_indices
        ranked_scores
    """

    candidates = np.asarray(
        candidate_indices,
        dtype=np.int64,
    )

    if candidates.ndim != 1:
        raise ValueError(
            "candidate_indices must be one-dimensional."
        )

    if scores.ndim != 1:
        raise ValueError(
            "scores must be one-dimensional."
        )

    if len(candidates) != scores.numel():
        raise ValueError(
            "candidate_indices and scores must have "
            "the same length."
        )

    if len(candidates) == 0:

        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                0,
                dtype=np.float32,
            ),
        )

    score_numpy = (
        scores
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    # Stable sort so ties retain deterministic candidate order.
    order = np.argsort(
        -score_numpy,
        kind="stable",
    )

    return (
        candidates[
            order
        ],
        score_numpy[
            order
        ],
    )


def top_k_candidates(
    candidate_indices: Sequence[int],
    scores: torch.Tensor,
    k: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rank and keep the top K candidates.
    """

    if k <= 0:
        raise ValueError(
            "k must be > 0."
        )

    ranked_candidates, ranked_scores = (
        rank_candidates(
            candidate_indices,
            scores,
        )
    )

    limit = min(
        k,
        len(
            ranked_candidates
        ),
    )

    return (
        ranked_candidates[
            :limit
        ],
        ranked_scores[
            :limit
        ],
    )