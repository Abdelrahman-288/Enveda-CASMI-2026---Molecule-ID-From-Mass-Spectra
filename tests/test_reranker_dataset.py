import numpy as np
import pandas as pd
import pytest

from casmi.reranking.dataset import (
    build_ranking_dataset,
    compute_group_sizes,
    query_positive_counts,
    ranking_dataframe_summary,
    sort_ranking_dataframe,
    split_by_query_ids,
    validate_reranker_dataframe,
    validate_single_positive_per_query,
)
from casmi.reranking.features import FEATURE_COLUMNS


def make_row(
    query_id,
    candidate_id,
    target,
    value=0.5,
):
    row = {
        "query_id": query_id,
        "candidate_id": candidate_id,
        "target": target,
    }

    for column in FEATURE_COLUMNS:
        row[column] = value

    return row


def make_dataframe():
    return pd.DataFrame(
        [
            make_row("q2", "c4", 0, 0.4),
            make_row("q1", "c2", 0, 0.2),
            make_row("q1", "c1", 1, 0.9),
            make_row("q2", "c3", 1, 0.8),
            make_row("q2", "c5", 0, 0.1),
        ]
    )


def test_validate_dataframe():
    df = make_dataframe()

    validate_reranker_dataframe(df)


def test_missing_required_column():
    df = make_dataframe().drop(
        columns=["aggregated_score"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required",
    ):
        validate_reranker_dataframe(df)


def test_invalid_target():
    df = make_dataframe()

    df.loc[0, "target"] = 2

    with pytest.raises(
        ValueError,
        match="binary",
    ):
        validate_reranker_dataframe(df)


def test_duplicate_candidate_within_query():
    df = make_dataframe()

    duplicate = df.iloc[[0]].copy()

    df = pd.concat(
        [df, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        validate_reranker_dataframe(df)


def test_nonfinite_feature_rejected():
    df = make_dataframe()

    df.loc[
        0,
        "aggregated_score",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="aggregated_score",
    ):
        validate_reranker_dataframe(df)


def test_sort_ranking_dataframe():
    df = make_dataframe()

    result = sort_ranking_dataframe(df)

    assert list(result["query_id"]) == [
        "q1",
        "q1",
        "q2",
        "q2",
        "q2",
    ]


def test_compute_group_sizes():
    query_ids = [
        "q1",
        "q1",
        "q2",
        "q2",
        "q2",
        "q3",
    ]

    result = compute_group_sizes(query_ids)

    assert result.tolist() == [
        2,
        3,
        1,
    ]


def test_noncontiguous_group_ids_rejected():
    with pytest.raises(
        ValueError,
        match="not contiguous",
    ):
        compute_group_sizes(
            ["q1", "q2", "q1"]
        )


def test_build_ranking_dataset():
    df = make_dataframe()

    dataset = build_ranking_dataset(df)

    assert dataset.X.shape == (
        5,
        len(FEATURE_COLUMNS),
    )

    assert dataset.y.shape == (5,)

    assert dataset.group_sizes.tolist() == [
        2,
        3,
    ]

    assert dataset.X.dtype == np.float32
    assert dataset.y.dtype == np.float32


def test_query_positive_counts():
    df = make_dataframe()

    counts = query_positive_counts(df)

    assert counts == {
        "q2": 1,
        "q1": 1,
    }


def test_single_positive_validation():
    df = make_dataframe()

    validate_single_positive_per_query(df)


def test_missing_positive_rejected():
    df = make_dataframe()

    df.loc[
        df["query_id"] == "q1",
        "target",
    ] = 0

    with pytest.raises(
        ValueError,
        match="exactly one positive",
    ):
        validate_single_positive_per_query(df)


def test_multiple_positives_rejected():
    df = make_dataframe()

    q2_indices = df.index[
        df["query_id"] == "q2"
    ]

    df.loc[
        q2_indices[:2],
        "target",
    ] = 1

    with pytest.raises(
        ValueError,
        match="exactly one positive",
    ):
        validate_single_positive_per_query(df)


def test_query_level_split():
    df = make_dataframe()

    train_df, val_df = split_by_query_ids(
        df,
        train_query_ids=["q1"],
        validation_query_ids=["q2"],
    )

    assert set(
        train_df["query_id"]
    ) == {"q1"}

    assert set(
        val_df["query_id"]
    ) == {"q2"}


def test_query_split_overlap_rejected():
    df = make_dataframe()

    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        split_by_query_ids(
            df,
            train_query_ids=["q1"],
            validation_query_ids=[
                "q1",
                "q2",
            ],
        )


def test_dataset_summary():
    df = make_dataframe()

    summary = ranking_dataframe_summary(df)

    assert summary["num_rows"] == 5
    assert summary["num_queries"] == 2

    assert (
        summary[
            "mean_candidates_per_query"
        ]
        == pytest.approx(2.5)
    )

    assert (
        summary[
            "queries_with_exactly_one_positive"
        ]
        == 2
    )