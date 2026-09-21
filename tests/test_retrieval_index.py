import numpy as np

from casmi.retrieval.index import (
    build_spectral_index,
    collapse_scores_by_structure,
    rank_structures,
    retrieve_scores,
)


def test_build_sparse_index():

    spectra = [
        {1: 1.0},
        {2: 1.0},
    ]

    index = build_spectral_index(
        spectra=spectra,
        structure_labels=[
            "A",
            "B",
        ],
        row_ids=[
            10,
            20,
        ],
        n_bins=10,
    )

    assert index.matrix.shape == (
        2,
        10,
    )

    assert index.matrix.nnz == 2


def test_retrieve_exact_match():

    spectra = [
        {1: 1.0},
        {2: 1.0},
    ]

    index = build_spectral_index(
        spectra=spectra,
        structure_labels=[
            "A",
            "B",
        ],
        row_ids=[
            1,
            2,
        ],
        n_bins=10,
    )

    scores = retrieve_scores(
        index,
        {1: 1.0},
    )

    assert np.isclose(
        scores[0],
        1.0,
    )

    assert np.isclose(
        scores[1],
        0.0,
    )


def test_structure_max_pooling():

    scores = np.array(
        [
            0.4,
            0.9,
            0.5,
        ]
    )

    labels = [
        "A",
        "A",
        "B",
    ]

    result = (
        collapse_scores_by_structure(
            scores,
            labels,
        )
    )

    assert np.isclose(
        result["A"],
        0.9,
    )

    assert np.isclose(
        result["B"],
        0.5,
    )


def test_rank_structures():

    scores = np.array(
        [
            0.2,
            0.8,
            0.5,
        ]
    )

    labels = [
        "A",
        "B",
        "C",
    ]

    ranked = rank_structures(
        scores,
        labels,
    )

    assert ranked[0][0] == "B"
    assert ranked[1][0] == "C"
    assert ranked[2][0] == "A"


def test_rank_top_k():

    scores = np.array(
        [
            0.2,
            0.8,
            0.5,
        ]
    )

    labels = [
        "A",
        "B",
        "C",
    ]

    ranked = rank_structures(
        scores,
        labels,
        top_k=2,
    )

    assert len(ranked) == 2
    assert ranked[0][0] == "B"