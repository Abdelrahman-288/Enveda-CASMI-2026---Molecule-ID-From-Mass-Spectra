import numpy as np
import pytest
import torch

from casmi.ranking.multispectrum import (
    aggregate_multispectrum_scores,
    build_availability_mask,
    candidate_intersection,
    candidate_union,
    masked_mean_scores,
    rank_candidates,
    select_candidate_pool,
    top_k_candidates,
)


def test_candidate_union():

    candidate_sets = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
    ]

    result = candidate_union(
        candidate_sets
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                1,
                2,
                3,
                4,
                5,
            ],
            dtype=np.int64,
        ),
    )


def test_candidate_intersection():

    candidate_sets = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
    ]

    result = candidate_intersection(
        candidate_sets
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [3],
            dtype=np.int64,
        ),
    )


def test_intersection_first_selection():

    candidate_sets = [
        [1, 2, 3],
        [2, 3],
        [2, 4],
    ]

    selection = select_candidate_pool(
        candidate_sets
    )

    assert (
        selection.mode
        == "intersection"
    )

    np.testing.assert_array_equal(
        selection.candidates,
        np.array(
            [2],
            dtype=np.int64,
        ),
    )


def test_union_fallback_when_intersection_empty():

    candidate_sets = [
        [1, 2],
        [2, 3],
        [3, 4],
    ]

    selection = select_candidate_pool(
        candidate_sets
    )

    assert (
        selection.mode
        == "union_fallback"
    )

    np.testing.assert_array_equal(
        selection.candidates,
        np.array(
            [
                1,
                2,
                3,
                4,
            ],
            dtype=np.int64,
        ),
    )


def test_union_fallback_with_empty_spectrum():

    candidate_sets = [
        [1, 2],
        [],
        [2, 3],
    ]

    selection = select_candidate_pool(
        candidate_sets
    )

    assert (
        selection.mode
        == "union_fallback"
    )

    np.testing.assert_array_equal(
        selection.candidates,
        np.array(
            [
                1,
                2,
                3,
            ],
            dtype=np.int64,
        ),
    )


def test_availability_mask():

    candidate_sets = [
        [1, 2],
        [2, 3],
    ]

    pool = np.array(
        [
            1,
            2,
            3,
        ],
        dtype=np.int64,
    )

    mask = build_availability_mask(
        candidate_sets,
        pool,
    )

    expected = np.array(
        [
            [
                True,
                True,
                False,
            ],
            [
                False,
                True,
                True,
            ],
        ],
        dtype=np.bool_,
    )

    np.testing.assert_array_equal(
        mask,
        expected,
    )


def test_masked_mean_scores():

    scores = torch.tensor(
        [
            [
                0.8,
                0.6,
                0.1,
            ],
            [
                0.2,
                0.4,
                0.9,
            ],
        ],
        dtype=torch.float32,
    )

    mask = torch.tensor(
        [
            [
                True,
                True,
                False,
            ],
            [
                False,
                True,
                True,
            ],
        ],
        dtype=torch.bool,
    )

    result = masked_mean_scores(
        scores,
        mask,
    )

    expected = torch.tensor(
        [
            0.8,
            0.5,
            0.9,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(
        result,
        expected,
        atol=1e-6,
    )


def test_mean_aggregation_without_mask():

    scores = torch.tensor(
        [
            [
                0.8,
                0.4,
            ],
            [
                0.6,
                0.8,
            ],
            [
                0.7,
                0.6,
            ],
        ],
        dtype=torch.float32,
    )

    result = aggregate_multispectrum_scores(
        scores,
        method="mean",
    )

    expected = torch.tensor(
        [
            0.7,
            0.6,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(
        result,
        expected,
        atol=1e-6,
    )


def test_mean_aggregation_with_union_mask():

    scores = torch.tensor(
        [
            [
                0.8,
                0.5,
                999.0,
            ],
            [
                999.0,
                0.7,
                0.9,
            ],
        ],
        dtype=torch.float32,
    )

    mask = torch.tensor(
        [
            [
                True,
                True,
                False,
            ],
            [
                False,
                True,
                True,
            ],
        ],
        dtype=torch.bool,
    )

    result = aggregate_multispectrum_scores(
        scores,
        availability_mask=mask,
        method="mean",
    )

    expected = torch.tensor(
        [
            0.8,
            0.6,
            0.9,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(
        result,
        expected,
        atol=1e-6,
    )


def test_max_aggregation():

    scores = torch.tensor(
        [
            [
                0.3,
                0.9,
            ],
            [
                0.8,
                0.4,
            ],
        ],
        dtype=torch.float32,
    )

    result = aggregate_multispectrum_scores(
        scores,
        method="max",
    )

    expected = torch.tensor(
        [
            0.8,
            0.9,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(
        result,
        expected,
        atol=1e-6,
    )


def test_top2_mean_aggregation():

    scores = torch.tensor(
        [
            [
                0.1,
                0.9,
            ],
            [
                0.7,
                0.2,
            ],
            [
                0.5,
                0.6,
            ],
        ],
        dtype=torch.float32,
    )

    result = aggregate_multispectrum_scores(
        scores,
        method="top2_mean",
    )

    expected = torch.tensor(
        [
            0.6,
            0.75,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(
        result,
        expected,
        atol=1e-6,
    )


def test_rank_candidates():

    candidates = np.array(
        [
            10,
            20,
            30,
        ],
        dtype=np.int64,
    )

    scores = torch.tensor(
        [
            0.2,
            0.9,
            0.5,
        ]
    )

    ranked_candidates, ranked_scores = (
        rank_candidates(
            candidates,
            scores,
        )
    )

    np.testing.assert_array_equal(
        ranked_candidates,
        np.array(
            [
                20,
                30,
                10,
            ],
            dtype=np.int64,
        ),
    )

    np.testing.assert_allclose(
        ranked_scores,
        np.array(
            [
                0.9,
                0.5,
                0.2,
            ],
            dtype=np.float32,
        ),
    )


def test_top_k_candidates():

    candidates = [
        10,
        20,
        30,
        40,
    ]

    scores = torch.tensor(
        [
            0.1,
            0.9,
            0.5,
            0.7,
        ]
    )

    ranked_candidates, ranked_scores = (
        top_k_candidates(
            candidates,
            scores,
            k=2,
        )
    )

    np.testing.assert_array_equal(
        ranked_candidates,
        np.array(
            [
                20,
                40,
            ],
            dtype=np.int64,
        ),
    )

    np.testing.assert_allclose(
        ranked_scores,
        np.array(
            [
                0.9,
                0.7,
            ],
            dtype=np.float32,
        ),
    )


def test_empty_candidate_sets():

    selection = select_candidate_pool(
        [
            [],
            [],
        ]
    )

    assert (
        selection.mode
        == "union_fallback"
    )

    assert (
        len(
            selection.candidates
        )
        == 0
    )


def test_invalid_aggregation_method():

    scores = torch.ones(
        2,
        3,
    )

    with pytest.raises(
        ValueError
    ):

        aggregate_multispectrum_scores(
            scores,
            method="invalid",
        )


def test_mask_shape_mismatch():

    scores = torch.ones(
        2,
        3,
    )

    mask = torch.ones(
        3,
        2,
        dtype=torch.bool,
    )

    with pytest.raises(
        ValueError
    ):

        aggregate_multispectrum_scores(
            scores,
            availability_mask=mask,
            method="mean",
        )