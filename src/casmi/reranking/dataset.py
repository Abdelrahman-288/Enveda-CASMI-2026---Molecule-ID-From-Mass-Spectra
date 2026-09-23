from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from casmi.reranking.features import FEATURE_COLUMNS


REQUIRED_COLUMNS = [
    "query_id",
    "candidate_id",
    "target",
    *FEATURE_COLUMNS,
]


@dataclass(frozen=True)
class RankingDataset:
    """
    Container for learning-to-rank data.

    Attributes
    ----------
    X
        Candidate feature matrix with shape [num_candidates, num_features].

    y
        Binary relevance labels for candidate rows.

    group_sizes
        Number of candidates belonging to each query.

    query_ids
        Query identifier for every row in X/y.

    candidate_ids
        Candidate identifier for every row.
    """

    X: np.ndarray
    y: np.ndarray
    group_sizes: np.ndarray
    query_ids: np.ndarray
    candidate_ids: np.ndarray


def validate_reranker_dataframe(df: pd.DataFrame) -> None:
    """
    Validate the schema and ranking-group structure.

    Raises
    ------
    ValueError
        If required columns are missing or the ranking data is invalid.
    """
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required reranker columns: {missing}"
        )

    if df.empty:
        raise ValueError("Reranker dataframe is empty.")

    if df["query_id"].isna().any():
        raise ValueError("query_id contains missing values.")

    if df["candidate_id"].isna().any():
        raise ValueError("candidate_id contains missing values.")

    if df["target"].isna().any():
        raise ValueError("target contains missing values.")

    invalid_targets = set(df["target"].unique()) - {0, 1}

    if invalid_targets:
        raise ValueError(
            "target must contain only binary values 0/1. "
            f"Found: {sorted(invalid_targets)}"
        )

    duplicate_mask = df.duplicated(
        subset=["query_id", "candidate_id"]
    )

    if duplicate_mask.any():
        duplicates = df.loc[
            duplicate_mask,
            ["query_id", "candidate_id"],
        ]

        raise ValueError(
            "Duplicate candidate rows found within query groups: "
            f"{duplicates.to_dict(orient='records')}"
        )

    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"Feature column '{column}' contains "
                "missing or non-numeric values."
            )

        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(
                f"Feature column '{column}' contains "
                "non-finite values."
            )


def sort_ranking_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort rows so candidate groups for the same query are contiguous.

    Sorting is deterministic and stable.
    """
    return (
        df.sort_values(
            by=["query_id", "candidate_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def compute_group_sizes(
    query_ids: Sequence[object],
) -> np.ndarray:
    """
    Compute contiguous ranking group sizes.

    Parameters
    ----------
    query_ids
        Query ID sequence where rows belonging to a query are expected to
        appear contiguously.

    Returns
    -------
    np.ndarray
        Number of candidate rows for each query group.
    """
    query_ids = np.asarray(query_ids)

    if query_ids.size == 0:
        return np.empty(0, dtype=np.int32)

    sizes: List[int] = []

    current = query_ids[0]
    count = 1

    seen = {current}

    for query_id in query_ids[1:]:
        if query_id == current:
            count += 1
            continue

        sizes.append(count)

        if query_id in seen:
            raise ValueError(
                "Query IDs are not contiguous. "
                f"Query '{query_id}' appears in multiple blocks."
            )

        seen.add(query_id)
        current = query_id
        count = 1

    sizes.append(count)

    return np.asarray(
        sizes,
        dtype=np.int32,
    )


def build_ranking_dataset(
    df: pd.DataFrame,
    *,
    sort_groups: bool = True,
) -> RankingDataset:
    """
    Convert a candidate dataframe into model-ready ranking arrays.

    The dataframe must contain one row per query/candidate pair.

    Parameters
    ----------
    df
        Candidate-level dataframe.

    sort_groups
        If True, rows are sorted to make query groups contiguous.

    Returns
    -------
    RankingDataset
        Model-ready feature matrix, labels, group sizes, and identifiers.
    """
    validate_reranker_dataframe(df)

    working = df.copy()

    if sort_groups:
        working = sort_ranking_dataframe(working)

    X = (
        working[FEATURE_COLUMNS]
        .to_numpy(dtype=np.float32)
    )

    y = (
        working["target"]
        .to_numpy(dtype=np.float32)
    )

    query_ids = working["query_id"].to_numpy()

    candidate_ids = working[
        "candidate_id"
    ].to_numpy()

    group_sizes = compute_group_sizes(
        query_ids
    )

    if int(group_sizes.sum()) != len(working):
        raise RuntimeError(
            "Group-size total does not match number of rows."
        )

    return RankingDataset(
        X=X,
        y=y,
        group_sizes=group_sizes,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
    )


def query_positive_counts(
    df: pd.DataFrame,
) -> Dict[object, int]:
    """
    Count positive candidates within each query.
    """
    validate_reranker_dataframe(df)

    counts = (
        df.groupby("query_id", sort=False)["target"]
        .sum()
        .astype(int)
    )

    return counts.to_dict()


def validate_single_positive_per_query(
    df: pd.DataFrame,
) -> None:
    """
    Ensure every ranking query contains exactly one true candidate.

    This is the expected configuration for CASMI structure identification.
    """
    counts = query_positive_counts(df)

    invalid = {
        query_id: count
        for query_id, count in counts.items()
        if count != 1
    }

    if invalid:
        raise ValueError(
            "Every query must contain exactly one positive candidate. "
            f"Invalid groups: {invalid}"
        )


def split_by_query_ids(
    df: pd.DataFrame,
    train_query_ids: Iterable[object],
    validation_query_ids: Iterable[object],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ranking rows using query IDs rather than individual rows.

    This prevents candidate rows from the same molecule from leaking between
    training and validation datasets.
    """
    train_ids = set(train_query_ids)
    validation_ids = set(validation_query_ids)

    overlap = train_ids & validation_ids

    if overlap:
        raise ValueError(
            "Train and validation query IDs overlap: "
            f"{sorted(overlap)}"
        )

    train_df = df[
        df["query_id"].isin(train_ids)
    ].copy()

    validation_df = df[
        df["query_id"].isin(validation_ids)
    ].copy()

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
    )


def ranking_dataframe_summary(
    df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Return basic diagnostics for a reranker dataset.
    """
    validate_reranker_dataframe(df)

    grouped = df.groupby(
        "query_id",
        sort=False,
    )

    candidate_counts = grouped.size()

    positive_counts = grouped["target"].sum()

    return {
        "num_rows": float(len(df)),
        "num_queries": float(
            df["query_id"].nunique()
        ),
        "num_candidates": float(
            df["candidate_id"].nunique()
        ),
        "mean_candidates_per_query": float(
            candidate_counts.mean()
        ),
        "median_candidates_per_query": float(
            candidate_counts.median()
        ),
        "min_candidates_per_query": float(
            candidate_counts.min()
        ),
        "max_candidates_per_query": float(
            candidate_counts.max()
        ),
        "queries_with_positive": float(
            (positive_counts > 0).sum()
        ),
        "queries_with_exactly_one_positive": float(
            (positive_counts == 1).sum()
        ),
    }