import numpy as np
import pytest

from casmi.reranking.features import (
    FEATURE_COLUMNS,
    build_candidate_features,
    compute_availability_fraction,
    compute_score_statistics,
    feature_mapping_to_vector,
    feature_matrix,
    features_to_dict,
    features_to_vector,
    reciprocal_rank_feature,
)


def test_score_statistics():
    scores = [0.2, 0.8, 0.6]

    result = compute_score_statistics(scores)

    assert result["mean_score"] == pytest.approx(
        (0.2 + 0.8 + 0.6) / 3
    )
    assert result["max_score"] == pytest.approx(0.8)
    assert result["top2_mean_score"] == pytest.approx(0.7)
    assert result["score_range"] == pytest.approx(0.6)


def test_single_score_statistics():
    result = compute_score_statistics([0.75])

    assert result["mean_score"] == pytest.approx(0.75)
    assert result["max_score"] == pytest.approx(0.75)
    assert result["top2_mean_score"] == pytest.approx(0.75)
    assert result["score_std"] == pytest.approx(0.0)
    assert result["score_range"] == pytest.approx(0.0)


def test_empty_score_statistics():
    result = compute_score_statistics([])

    assert result["mean_score"] == 0.0
    assert result["max_score"] == 0.0
    assert result["top2_mean_score"] == 0.0
    assert result["score_std"] == 0.0
    assert result["score_range"] == 0.0


def test_nonfinite_scores_are_ignored():
    result = compute_score_statistics(
        [0.2, np.nan, np.inf, 0.8]
    )

    assert result["mean_score"] == pytest.approx(0.5)
    assert result["max_score"] == pytest.approx(0.8)
    assert result["top2_mean_score"] == pytest.approx(0.5)


def test_availability_fraction():
    assert compute_availability_fraction(3, 4) == pytest.approx(
        0.75
    )


def test_availability_fraction_zero_total():
    assert compute_availability_fraction(0, 0) == 0.0


def test_availability_fraction_clipped():
    assert compute_availability_fraction(5, 4) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank_feature(1) == pytest.approx(1.0)
    assert reciprocal_rank_feature(2) == pytest.approx(0.5)
    assert reciprocal_rank_feature(5) == pytest.approx(0.2)


def test_invalid_reciprocal_rank():
    assert reciprocal_rank_feature(0) == 0.0
    assert reciprocal_rank_feature(-1) == 0.0
    assert reciprocal_rank_feature(None) == 0.0


def test_build_candidate_features():
    features = build_candidate_features(
        per_spectrum_scores=[0.2, 0.6, 0.8],
        aggregated_score=0.55,
        spectral_similarity_score=0.61,
        embedding_similarity_score=0.72,
        neural_compatibility_score=0.83,
        mass_error_ppm=-2.5,
        formula_match=True,
        num_supporting_spectra=3,
        total_spectra=4,
        retrieval_rank=2,
        similarity_rank=3,
        neural_rank=1,
    )

    assert features.aggregated_score == pytest.approx(0.55)
    assert features.mean_score == pytest.approx(
        (0.2 + 0.6 + 0.8) / 3
    )
    assert features.max_score == pytest.approx(0.8)
    assert features.top2_mean_score == pytest.approx(0.7)

    assert features.mass_error_ppm == pytest.approx(-2.5)
    assert features.abs_mass_error_ppm == pytest.approx(2.5)

    assert features.formula_match == 1.0

    assert (
        features.candidate_availability_fraction
        == pytest.approx(0.75)
    )


def test_features_to_dict_has_canonical_columns():
    features = build_candidate_features(
        per_spectrum_scores=[0.5]
    )

    result = features_to_dict(features)

    assert list(result.keys()) == FEATURE_COLUMNS


def test_features_to_vector_shape():
    features = build_candidate_features(
        per_spectrum_scores=[0.5, 0.7]
    )

    vector = features_to_vector(features)

    assert vector.shape == (len(FEATURE_COLUMNS),)
    assert vector.dtype == np.float32


def test_mapping_missing_values_default_to_zero():
    vector = feature_mapping_to_vector(
        {
            "aggregated_score": 0.5,
        }
    )

    assert vector.shape == (len(FEATURE_COLUMNS),)
    assert vector[0] == pytest.approx(0.5)

    assert np.all(vector[1:] == 0.0)


def test_feature_matrix():
    rows = [
        {
            "aggregated_score": 0.1,
            "mean_score": 0.2,
        },
        {
            "aggregated_score": 0.8,
            "mean_score": 0.7,
        },
    ]

    matrix = feature_matrix(rows)

    assert matrix.shape == (
        2,
        len(FEATURE_COLUMNS),
    )

    assert matrix.dtype == np.float32


def test_empty_feature_matrix():
    matrix = feature_matrix([])

    assert matrix.shape == (
        0,
        len(FEATURE_COLUMNS),
    )