from pathlib import Path

import numpy as np
import pytest

from casmi.reranking.dataset import RankingDataset
from casmi.reranking.model import (
    CandidateRanker,
    RankerConfig,
    hits_at_k,
    mean_reciprocal_rank,
    rank_within_queries,
    ranking_metrics,
    reciprocal_ranks,
)


# ============================================================
# Ranking metric tests
# ============================================================


def test_rank_within_queries():
    scores = [
        0.2,
        0.9,
        0.4,
        0.8,
        0.1,
    ]

    query_ids = [
        "q1",
        "q1",
        "q2",
        "q2",
        "q2",
    ]

    ranks = rank_within_queries(
        scores,
        query_ids,
    )

    assert ranks.tolist() == [
        2,
        1,
        2,
        1,
        3,
    ]


def test_rank_ties_are_deterministic():
    scores = [
        0.5,
        0.5,
        0.2,
    ]

    query_ids = [
        "q1",
        "q1",
        "q1",
    ]

    ranks = rank_within_queries(
        scores,
        query_ids,
    )

    assert ranks.tolist() == [
        1,
        2,
        3,
    ]


def test_reciprocal_ranks():
    ranks = [
        2,
        1,
        3,
        1,
        2,
    ]

    targets = [
        1,
        0,
        0,
        1,
        0,
    ]

    query_ids = [
        "q1",
        "q1",
        "q1",
        "q2",
        "q2",
    ]

    values = reciprocal_ranks(
        ranks,
        targets,
        query_ids,
    )

    assert values["q1"] == pytest.approx(0.5)
    assert values["q2"] == pytest.approx(1.0)


def test_reciprocal_rank_cutoff():
    ranks = [
        30,
        1,
    ]

    targets = [
        1,
        1,
    ]

    query_ids = [
        "q1",
        "q2",
    ]

    result = reciprocal_ranks(
        ranks,
        targets,
        query_ids,
        cutoff=25,
    )

    assert result["q1"] == 0.0
    assert result["q2"] == 1.0


def test_mean_reciprocal_rank():
    ranks = [
        2,
        1,
        3,
        1,
        2,
    ]

    targets = [
        1,
        0,
        0,
        1,
        0,
    ]

    query_ids = [
        "q1",
        "q1",
        "q1",
        "q2",
        "q2",
    ]

    result = mean_reciprocal_rank(
        ranks,
        targets,
        query_ids,
    )

    assert result == pytest.approx(0.75)


def test_hits_at_k():
    ranks = [
        2,
        1,
        4,
        1,
    ]

    targets = [
        1,
        0,
        1,
        0,
    ]

    query_ids = [
        "q1",
        "q1",
        "q2",
        "q2",
    ]

    assert hits_at_k(
        ranks,
        targets,
        query_ids,
        1,
    ) == 0.0

    assert hits_at_k(
        ranks,
        targets,
        query_ids,
        2,
    ) == pytest.approx(0.5)

    assert hits_at_k(
        ranks,
        targets,
        query_ids,
        5,
    ) == 1.0


def test_invalid_k():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        hits_at_k(
            [1],
            [1],
            ["q1"],
            0,
        )


def test_ranking_metrics():
    scores = [
        0.7,
        0.9,
        0.8,
        0.2,
        0.5,
    ]

    targets = [
        0,
        1,
        0,
        0,
        1,
    ]

    query_ids = [
        "q1",
        "q1",
        "q1",
        "q2",
        "q2",
    ]

    metrics = ranking_metrics(
        scores,
        targets,
        query_ids,
    )

    assert metrics["mrr"] == 1.0
    assert metrics["mrr@25"] == 1.0
    assert metrics["hits@1"] == 1.0
    assert metrics["hits@5"] == 1.0
    assert metrics["mean_true_rank"] == 1.0
    assert metrics["median_true_rank"] == 1.0


def test_length_mismatch():
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        rank_within_queries(
            [0.1, 0.2],
            ["q1"],
        )


def test_empty_metrics():
    metrics = ranking_metrics(
        [],
        [],
        [],
    )

    assert metrics["mrr"] == 0.0
    assert metrics["hits@1"] == 0.0
    assert metrics["mean_true_rank"] == 0.0


# ============================================================
# XGBoost integration tests
# ============================================================


def make_toy_ranking_dataset():
    X = np.asarray(
        [
            # q1
            [0.9, 0.8],
            [0.2, 0.3],
            [0.1, 0.2],

            # q2
            [0.3, 0.2],
            [0.8, 0.9],
            [0.2, 0.1],

            # q3
            [0.1, 0.3],
            [0.2, 0.2],
            [0.9, 0.8],
        ],
        dtype=np.float32,
    )

    y = np.asarray(
        [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        ],
        dtype=np.float32,
    )

    query_ids = np.asarray(
        [
            "q1", "q1", "q1",
            "q2", "q2", "q2",
            "q3", "q3", "q3",
        ]
    )

    candidate_ids = np.asarray(
        [
            "q1_true",
            "q1_a",
            "q1_b",
            "q2_a",
            "q2_true",
            "q2_b",
            "q3_a",
            "q3_b",
            "q3_true",
        ]
    )

    return RankingDataset(
        X=X,
        y=y,
        group_sizes=np.asarray(
            [3, 3, 3],
            dtype=np.int32,
        ),
        query_ids=query_ids,
        candidate_ids=candidate_ids,
    )


def test_candidate_ranker_fit_predict():
    dataset = make_toy_ranking_dataset()

    config = RankerConfig(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
    )

    ranker = CandidateRanker(config)

    ranker.fit(dataset)

    scores = ranker.predict(
        dataset.X
    )

    assert scores.shape == (9,)
    assert np.isfinite(scores).all()


def test_candidate_ranker_feature_importance():
    dataset = make_toy_ranking_dataset()

    ranker = CandidateRanker(
        RankerConfig(
            n_estimators=20,
            max_depth=3,
        )
    )

    ranker.fit(dataset)

    importance = ranker.feature_importance()

    assert importance.shape == (2,)
    assert np.isfinite(importance).all()


def test_candidate_ranker_save_load(
    tmp_path: Path,
):
    dataset = make_toy_ranking_dataset()

    ranker = CandidateRanker(
        RankerConfig(
            n_estimators=20,
            max_depth=3,
        )
    )

    ranker.fit(dataset)

    original_scores = ranker.predict(
        dataset.X
    )

    model_path = (
        tmp_path
        / "toy_ranker.json"
    )

    ranker.save(model_path)

    assert model_path.exists()

    loaded = CandidateRanker(
        RankerConfig(
            n_estimators=20,
            max_depth=3,
        )
    )

    loaded.load(model_path)

    loaded_scores = loaded.predict(
        dataset.X
    )

    assert np.allclose(
        original_scores,
        loaded_scores,
    )