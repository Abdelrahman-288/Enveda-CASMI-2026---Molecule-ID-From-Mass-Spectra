from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateFeatures:
    """
    Candidate-level features used by the Stage 10 learned reranker.

    One instance represents one candidate structure for one query molecule.
    """

    aggregated_score: float = 0.0
    mean_score: float = 0.0
    max_score: float = 0.0
    top2_mean_score: float = 0.0

    spectral_similarity_score: float = 0.0
    embedding_similarity_score: float = 0.0
    neural_compatibility_score: float = 0.0

    mass_error_ppm: float = 0.0
    abs_mass_error_ppm: float = 0.0
    formula_match: float = 0.0

    num_supporting_spectra: float = 0.0
    candidate_availability_fraction: float = 0.0

    retrieval_rank: float = 0.0
    similarity_rank: float = 0.0
    neural_rank: float = 0.0

    score_std: float = 0.0
    score_range: float = 0.0


FEATURE_COLUMNS: List[str] = [
    "aggregated_score",
    "mean_score",
    "max_score",
    "top2_mean_score",
    "spectral_similarity_score",
    "embedding_similarity_score",
    "neural_compatibility_score",
    "mass_error_ppm",
    "abs_mass_error_ppm",
    "formula_match",
    "num_supporting_spectra",
    "candidate_availability_fraction",
    "retrieval_rank",
    "similarity_rank",
    "neural_rank",
    "score_std",
    "score_range",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    """
    Convert a value to a finite float.

    Missing, invalid, NaN, or infinite values are replaced by ``default``.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not np.isfinite(result):
        return float(default)

    return result


def _valid_scores(scores: Optional[Iterable[float]]) -> np.ndarray:
    """
    Convert an iterable of scores into a one-dimensional finite float array.
    """
    if scores is None:
        return np.empty(0, dtype=np.float64)

    values = []

    for score in scores:
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue

        if np.isfinite(value):
            values.append(value)

    return np.asarray(values, dtype=np.float64)


def compute_score_statistics(
    scores: Optional[Iterable[float]],
) -> Dict[str, float]:
    """
    Calculate summary statistics from per-spectrum candidate scores.

    Returns
    -------
    dict
        Dictionary containing:

        - mean_score
        - max_score
        - top2_mean_score
        - score_std
        - score_range

    Empty score sequences produce zeros.
    """
    values = _valid_scores(scores)

    if values.size == 0:
        return {
            "mean_score": 0.0,
            "max_score": 0.0,
            "top2_mean_score": 0.0,
            "score_std": 0.0,
            "score_range": 0.0,
        }

    mean_score = float(values.mean())
    max_score = float(values.max())

    if values.size == 1:
        top2_mean_score = float(values[0])
    else:
        top_two = np.partition(values, -2)[-2:]
        top2_mean_score = float(top_two.mean())

    return {
        "mean_score": mean_score,
        "max_score": max_score,
        "top2_mean_score": top2_mean_score,
        "score_std": float(values.std(ddof=0)),
        "score_range": float(values.max() - values.min()),
    }


def compute_availability_fraction(
    num_supporting_spectra: int,
    total_spectra: int,
) -> float:
    """
    Return the fraction of spectra for which a candidate was available.
    """
    if total_spectra <= 0:
        return 0.0

    fraction = float(num_supporting_spectra) / float(total_spectra)

    return float(np.clip(fraction, 0.0, 1.0))


def reciprocal_rank_feature(rank: object) -> float:
    """
    Convert a one-based rank to reciprocal rank.

    Examples
    --------
    rank 1 -> 1.0
    rank 2 -> 0.5
    rank 5 -> 0.2

    Invalid or non-positive ranks return 0.
    """
    rank_value = _safe_float(rank, default=0.0)

    if rank_value <= 0:
        return 0.0

    return 1.0 / rank_value


def build_candidate_features(
    *,
    per_spectrum_scores: Optional[Sequence[float]] = None,
    aggregated_score: object = 0.0,
    spectral_similarity_score: object = 0.0,
    embedding_similarity_score: object = 0.0,
    neural_compatibility_score: object = 0.0,
    mass_error_ppm: object = 0.0,
    formula_match: object = False,
    num_supporting_spectra: Optional[int] = None,
    total_spectra: Optional[int] = None,
    retrieval_rank: object = 0.0,
    similarity_rank: object = 0.0,
    neural_rank: object = 0.0,
) -> CandidateFeatures:
    """
    Build a normalized CandidateFeatures object.

    This function is deliberately independent of pandas and model code so it
    can be tested and reused by dataset construction and inference.
    """
    score_stats = compute_score_statistics(per_spectrum_scores)

    if num_supporting_spectra is None:
        num_supporting_spectra = len(_valid_scores(per_spectrum_scores))

    if total_spectra is None:
        total_spectra = num_supporting_spectra

    mass_error = _safe_float(mass_error_ppm)

    formula_value = 1.0 if bool(formula_match) else 0.0

    return CandidateFeatures(
        aggregated_score=_safe_float(aggregated_score),
        mean_score=score_stats["mean_score"],
        max_score=score_stats["max_score"],
        top2_mean_score=score_stats["top2_mean_score"],
        spectral_similarity_score=_safe_float(
            spectral_similarity_score
        ),
        embedding_similarity_score=_safe_float(
            embedding_similarity_score
        ),
        neural_compatibility_score=_safe_float(
            neural_compatibility_score
        ),
        mass_error_ppm=mass_error,
        abs_mass_error_ppm=abs(mass_error),
        formula_match=formula_value,
        num_supporting_spectra=float(
            max(int(num_supporting_spectra), 0)
        ),
        candidate_availability_fraction=compute_availability_fraction(
            int(num_supporting_spectra),
            int(total_spectra),
        ),
        retrieval_rank=_safe_float(retrieval_rank),
        similarity_rank=_safe_float(similarity_rank),
        neural_rank=_safe_float(neural_rank),
        score_std=score_stats["score_std"],
        score_range=score_stats["score_range"],
    )


def features_to_dict(
    features: CandidateFeatures,
) -> Dict[str, float]:
    """
    Convert CandidateFeatures to an ordered feature dictionary.
    """
    values = asdict(features)

    return {
        column: float(values[column])
        for column in FEATURE_COLUMNS
    }


def features_to_vector(
    features: CandidateFeatures,
) -> np.ndarray:
    """
    Convert CandidateFeatures to a numeric feature vector.
    """
    values = features_to_dict(features)

    return np.asarray(
        [values[column] for column in FEATURE_COLUMNS],
        dtype=np.float32,
    )


def feature_mapping_to_vector(
    features: Mapping[str, object],
) -> np.ndarray:
    """
    Convert an arbitrary feature mapping into the canonical feature order.

    Missing or non-finite values are replaced with zero.
    """
    return np.asarray(
        [
            _safe_float(features.get(column, 0.0))
            for column in FEATURE_COLUMNS
        ],
        dtype=np.float32,
    )


def feature_matrix(
    rows: Sequence[Mapping[str, object]],
) -> np.ndarray:
    """
    Convert candidate feature mappings into a 2-D model input matrix.
    """
    if len(rows) == 0:
        return np.empty(
            (0, len(FEATURE_COLUMNS)),
            dtype=np.float32,
        )

    return np.vstack(
        [
            feature_mapping_to_vector(row)
            for row in rows
        ]
    )