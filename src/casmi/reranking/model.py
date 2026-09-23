from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from casmi.reranking.dataset import RankingDataset


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class RankerConfig:
    """
    Configuration for the Stage 10 XGBoost learning-to-rank model.
    """

    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6

    min_child_weight: float = 1.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9

    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    random_state: int = 42

    tree_method: str = "hist"

    objective: str = "rank:pairwise"
    eval_metric: str = "ndcg@25"

    early_stopping_rounds: Optional[int] = None

    # --------------------------------------------------------
    # LambdaRank configuration
    # --------------------------------------------------------

    lambdarank_pair_method: Optional[str] = None

    lambdarank_num_pair_per_sample: Optional[
        int
    ] = None

    lambdarank_normalization: Optional[
        bool
    ] = None

    lambdarank_score_normalization: Optional[
        bool
    ] = None


# ============================================================
# Candidate ranker
# ============================================================


class CandidateRanker:
    """
    Wrapper around XGBoost learning-to-rank models.

    Supports:
        - rank:pairwise
        - rank:ndcg
        - rank:map

    and optional LambdaRank pair construction.
    """

    def __init__(
        self,
        config: Optional[
            RankerConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or RankerConfig()
        )

        self.model = None

    # ========================================================
    # Configuration validation
    # ========================================================

    def _validate_config(
        self,
    ) -> None:

        supported_objectives = {
            "rank:pairwise",
            "rank:ndcg",
            "rank:map",
        }

        if (
            self.config.objective
            not in supported_objectives
        ):
            raise ValueError(
                "Unsupported ranking objective: "
                f"{self.config.objective}"
            )

        if (
            self.config.n_estimators
            <= 0
        ):
            raise ValueError(
                "n_estimators must be positive."
            )

        if (
            self.config.learning_rate
            <= 0
        ):
            raise ValueError(
                "learning_rate must be positive."
            )

        if (
            self.config.max_depth
            < 0
        ):
            raise ValueError(
                "max_depth must be >= 0."
            )

        if (
            self.config.min_child_weight
            < 0
        ):
            raise ValueError(
                "min_child_weight must be >= 0."
            )

        if not (
            0
            < self.config.subsample
            <= 1
        ):
            raise ValueError(
                "subsample must be in (0, 1]."
            )

        if not (
            0
            < self.config.colsample_bytree
            <= 1
        ):
            raise ValueError(
                "colsample_bytree must be "
                "in (0, 1]."
            )

        if (
            self.config.reg_alpha
            < 0
        ):
            raise ValueError(
                "reg_alpha must be >= 0."
            )

        if (
            self.config.reg_lambda
            < 0
        ):
            raise ValueError(
                "reg_lambda must be >= 0."
            )

        if (
            self.config
            .early_stopping_rounds
            is not None
            and self.config
            .early_stopping_rounds
            <= 0
        ):
            raise ValueError(
                "early_stopping_rounds "
                "must be positive."
            )

        pair_method = (
            self.config
            .lambdarank_pair_method
        )

        if (
            pair_method
            is not None
            and pair_method
            not in {
                "mean",
                "topk",
            }
        ):
            raise ValueError(
                "lambdarank_pair_method "
                "must be 'mean' or 'topk'."
            )

        num_pairs = (
            self.config
            .lambdarank_num_pair_per_sample
        )

        if (
            num_pairs
            is not None
            and num_pairs <= 0
        ):
            raise ValueError(
                "lambdarank_num_pair_per_sample "
                "must be positive."
            )

    # ========================================================
    # Build XGBoost model
    # ========================================================

    def _build_model(
        self,
    ):

        self._validate_config()

        try:
            from xgboost import (
                XGBRanker,
            )

        except ImportError as exc:
            raise ImportError(
                "xgboost is required for "
                "Stage 10 reranking. "
                "Install it with: "
                "pip install xgboost"
            ) from exc

        params = {
            "objective": (
                self.config.objective
            ),

            "eval_metric": (
                self.config.eval_metric
            ),

            "n_estimators": (
                self.config.n_estimators
            ),

            "learning_rate": (
                self.config.learning_rate
            ),

            "max_depth": (
                self.config.max_depth
            ),

            "min_child_weight": (
                self.config
                .min_child_weight
            ),

            "subsample": (
                self.config.subsample
            ),

            "colsample_bytree": (
                self.config
                .colsample_bytree
            ),

            "reg_alpha": (
                self.config.reg_alpha
            ),

            "reg_lambda": (
                self.config.reg_lambda
            ),

            "random_state": (
                self.config.random_state
            ),

            "tree_method": (
                self.config.tree_method
            ),
        }

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            self.config
            .early_stopping_rounds
            is not None
        ):
            params[
                "early_stopping_rounds"
            ] = (
                self.config
                .early_stopping_rounds
            )

        # ----------------------------------------------------
        # LambdaRank pair strategy
        # ----------------------------------------------------

        if (
            self.config
            .lambdarank_pair_method
            is not None
        ):
            params[
                "lambdarank_pair_method"
            ] = (
                self.config
                .lambdarank_pair_method
            )

        if (
            self.config
            .lambdarank_num_pair_per_sample
            is not None
        ):
            params[
                "lambdarank_num_pair_per_sample"
            ] = (
                self.config
                .lambdarank_num_pair_per_sample
            )

        if (
            self.config
            .lambdarank_normalization
            is not None
        ):
            params[
                "lambdarank_normalization"
            ] = (
                self.config
                .lambdarank_normalization
            )

        if (
            self.config
            .lambdarank_score_normalization
            is not None
        ):
            params[
                "lambdarank_score_normalization"
            ] = (
                self.config
                .lambdarank_score_normalization
            )

        return XGBRanker(
            **params
        )

    # ========================================================
    # Dataset validation
    # ========================================================

    @staticmethod
    def _validate_dataset(
        dataset: RankingDataset,
        *,
        name: str,
    ) -> None:

        if dataset.X.ndim != 2:
            raise ValueError(
                f"{name}.X must be "
                "a 2-D feature matrix."
            )

        if dataset.y.ndim != 1:
            raise ValueError(
                f"{name}.y must be "
                "a 1-D label vector."
            )

        if (
            len(dataset.X)
            != len(dataset.y)
        ):
            raise ValueError(
                f"{name} feature and "
                "label counts do not match."
            )

        if len(
            dataset.X
        ) == 0:
            raise ValueError(
                f"Cannot use an empty {name}."
            )

        group_sizes = np.asarray(
            dataset.group_sizes,
            dtype=np.int64,
        )

        if (
            group_sizes.ndim
            != 1
        ):
            raise ValueError(
                f"{name}.group_sizes "
                "must be one-dimensional."
            )

        if (
            len(group_sizes)
            == 0
        ):
            raise ValueError(
                f"{name} contains no "
                "ranking groups."
            )

        if np.any(
            group_sizes <= 0
        ):
            raise ValueError(
                f"{name}.group_sizes "
                "must contain only "
                "positive values."
            )

        if int(
            group_sizes.sum()
        ) != len(
            dataset.X
        ):
            raise ValueError(
                f"{name} ranking group sizes "
                "do not sum to the number "
                "of candidate rows."
            )

        X = np.asarray(
            dataset.X,
            dtype=np.float32,
        )

        y = np.asarray(
            dataset.y,
            dtype=np.float32,
        )

        if not np.isfinite(
            X
        ).all():
            raise ValueError(
                f"{name}.X contains "
                "non-finite values."
            )

        if not np.isfinite(
            y
        ).all():
            raise ValueError(
                f"{name}.y contains "
                "non-finite values."
            )

    # ========================================================
    # Fit
    # ========================================================

    def fit(
        self,
        dataset: RankingDataset,
        validation_dataset: Optional[
            RankingDataset
        ] = None,
        *,
        verbose: bool = False,
    ) -> "CandidateRanker":

        self._validate_dataset(
            dataset,
            name="dataset",
        )

        if (
            self.config
            .early_stopping_rounds
            is not None
            and validation_dataset
            is None
        ):
            raise ValueError(
                "early_stopping_rounds "
                "requires validation_dataset."
            )

        if (
            validation_dataset
            is not None
        ):
            self._validate_dataset(
                validation_dataset,
                name=(
                    "validation_dataset"
                ),
            )

            if (
                dataset.X.shape[1]
                != validation_dataset
                .X.shape[1]
            ):
                raise ValueError(
                    "Training and validation "
                    "datasets must have the "
                    "same number of features."
                )

        self.model = (
            self._build_model()
        )

        fit_kwargs = {
            "X": np.asarray(
                dataset.X,
                dtype=np.float32,
            ),

            "y": np.asarray(
                dataset.y,
                dtype=np.float32,
            ),

            "group": np.asarray(
                dataset.group_sizes,
                dtype=np.int32,
            ),

            "verbose": verbose,
        }

        # ----------------------------------------------------
        # Validation / early stopping
        # ----------------------------------------------------

        if (
            validation_dataset
            is not None
        ):
            fit_kwargs[
                "eval_set"
            ] = [
                (
                    np.asarray(
                        validation_dataset.X,
                        dtype=np.float32,
                    ),

                    np.asarray(
                        validation_dataset.y,
                        dtype=np.float32,
                    ),
                )
            ]

            fit_kwargs[
                "eval_group"
            ] = [
                np.asarray(
                    validation_dataset
                    .group_sizes,
                    dtype=np.int32,
                )
            ]

        self.model.fit(
            **fit_kwargs
        )

        return self

    # ========================================================
    # Predict
    # ========================================================

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        if self.model is None:
            raise RuntimeError(
                "Ranker has not been trained."
            )

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a "
                "2-D feature matrix."
            )

        if len(
            X
        ) == 0:
            return np.empty(
                0,
                dtype=np.float32,
            )

        if not np.isfinite(
            X
        ).all():
            raise ValueError(
                "X contains "
                "non-finite feature values."
            )

        scores = (
            self.model.predict(
                X
            )
        )

        return np.asarray(
            scores,
            dtype=np.float32,
        )

    # ========================================================
    # Feature importance
    # ========================================================

    def feature_importance(
        self,
    ) -> np.ndarray:

        if self.model is None:
            raise RuntimeError(
                "Ranker has not been trained."
            )

        return np.asarray(
            self.model
            .feature_importances_,
            dtype=np.float32,
        )

    # ========================================================
    # Best iteration
    # ========================================================

    def best_iteration(
        self,
    ) -> Optional[int]:

        if self.model is None:
            raise RuntimeError(
                "Ranker has not been trained."
            )

        try:
            value = (
                self.model.best_iteration
            )

        except (
            AttributeError,
            ValueError,
        ):
            return None

        if value is None:
            return None

        return int(
            value
        )

    # ========================================================
    # Best score
    # ========================================================

    def best_score(
        self,
    ) -> Optional[float]:

        if self.model is None:
            raise RuntimeError(
                "Ranker has not been trained."
            )

        try:
            value = (
                self.model.best_score
            )

        except (
            AttributeError,
            ValueError,
        ):
            return None

        if value is None:
            return None

        return float(
            value
        )

    # ========================================================
    # Evaluation history
    # ========================================================

    def evaluation_results(
        self,
    ) -> Dict:

        if self.model is None:
            raise RuntimeError(
                "Ranker has not been trained."
            )

        try:
            results = (
                self.model
                .evals_result()
            )

        except (
            AttributeError,
            ValueError,
        ):
            return {}

        if results is None:
            return {}

        return dict(
            results
        )

    # ========================================================
    # Save
    # ========================================================

    def save(
        self,
        path: str | Path,
    ) -> None:

        if self.model is None:
            raise RuntimeError(
                "Cannot save an "
                "untrained ranker."
            )

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save_model(
            str(path)
        )

    # ========================================================
    # Load
    # ========================================================

    def load(
        self,
        path: str | Path,
    ) -> "CandidateRanker":

        path = Path(
            path
        )

        if not path.exists():
            raise FileNotFoundError(
                "Ranker model "
                f"not found: {path}"
            )

        self.model = (
            self._build_model()
        )

        self.model.load_model(
            str(path)
        )

        return self


# ============================================================
# Ranking helpers
# ============================================================


def rank_within_queries(
    scores: Sequence[float],
    query_ids: Sequence[object],
) -> np.ndarray:

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    query_ids = np.asarray(
        query_ids
    )

    if (
        len(scores)
        != len(query_ids)
    ):
        raise ValueError(
            "scores and query_ids "
            "must have equal length."
        )

    if (
        len(scores)
        == 0
    ):
        return np.empty(
            0,
            dtype=np.int32,
        )

    ranks = np.zeros(
        len(scores),
        dtype=np.int32,
    )

    seen = []

    for query_id in query_ids:

        if (
            query_id
            not in seen
        ):
            seen.append(
                query_id
            )

    for query_id in seen:

        indices = np.flatnonzero(
            query_ids
            == query_id
        )

        query_scores = (
            scores[
                indices
            ]
        )

        order = np.argsort(
            -query_scores,
            kind="mergesort",
        )

        query_ranks = (
            np.empty(
                len(indices),
                dtype=np.int32,
            )
        )

        query_ranks[
            order
        ] = (
            np.arange(
                len(indices),
                dtype=np.int32,
            )
            + 1
        )

        ranks[
            indices
        ] = (
            query_ranks
        )

    return ranks


# ============================================================
# Reciprocal ranks
# ============================================================


def reciprocal_ranks(
    ranks: Sequence[int],
    targets: Sequence[int],
    query_ids: Sequence[object],
    *,
    cutoff: Optional[
        int
    ] = None,
) -> Dict[
    object,
    float,
]:

    ranks = np.asarray(
        ranks
    )

    targets = np.asarray(
        targets
    )

    query_ids = np.asarray(
        query_ids
    )

    if not (
        len(ranks)
        == len(targets)
        == len(query_ids)
    ):
        raise ValueError(
            "ranks, targets, and query_ids "
            "must have equal length."
        )

    result: Dict[
        object,
        float,
    ] = {}

    for query_id in (
        dict.fromkeys(
            query_ids.tolist()
        )
    ):

        mask = (
            query_ids
            == query_id
        )

        positive_ranks = (
            ranks[
                mask
                & (
                    targets
                    == 1
                )
            ]
        )

        if (
            len(
                positive_ranks
            )
            == 0
        ):
            result[
                query_id
            ] = 0.0

            continue

        best_rank = int(
            positive_ranks.min()
        )

        if (
            cutoff
            is not None
            and best_rank
            > cutoff
        ):
            result[
                query_id
            ] = 0.0

        else:
            result[
                query_id
            ] = (
                1.0
                / best_rank
            )

    return result


# ============================================================
# Mean reciprocal rank
# ============================================================


def mean_reciprocal_rank(
    ranks: Sequence[int],
    targets: Sequence[int],
    query_ids: Sequence[object],
    *,
    cutoff: Optional[
        int
    ] = None,
) -> float:

    values = (
        reciprocal_ranks(
            ranks,
            targets,
            query_ids,
            cutoff=cutoff,
        )
    )

    if not values:
        return 0.0

    return float(
        np.mean(
            list(
                values.values()
            )
        )
    )


# ============================================================
# Hits@K
# ============================================================


def hits_at_k(
    ranks: Sequence[int],
    targets: Sequence[int],
    query_ids: Sequence[object],
    k: int,
) -> float:

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    ranks = np.asarray(
        ranks
    )

    targets = np.asarray(
        targets
    )

    query_ids = np.asarray(
        query_ids
    )

    if not (
        len(ranks)
        == len(targets)
        == len(query_ids)
    ):
        raise ValueError(
            "ranks, targets, and query_ids "
            "must have equal length."
        )

    hits = []

    for query_id in (
        dict.fromkeys(
            query_ids.tolist()
        )
    ):

        mask = (
            (
                query_ids
                == query_id
            )
            & (
                targets
                == 1
            )
        )

        positive_ranks = (
            ranks[
                mask
            ]
        )

        if (
            len(
                positive_ranks
            )
            == 0
        ):
            hits.append(
                0.0
            )

        else:
            hits.append(
                float(
                    positive_ranks.min()
                    <= k
                )
            )

    if not hits:
        return 0.0

    return float(
        np.mean(
            hits
        )
    )


# ============================================================
# Ranking metrics
# ============================================================


def ranking_metrics(
    scores: Sequence[float],
    targets: Sequence[int],
    query_ids: Sequence[object],
) -> Dict[
    str,
    float,
]:

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    targets = np.asarray(
        targets
    )

    query_ids = np.asarray(
        query_ids
    )

    if not (
        len(scores)
        == len(targets)
        == len(query_ids)
    ):
        raise ValueError(
            "scores, targets, and query_ids "
            "must have equal length."
        )

    if (
        len(scores)
        == 0
    ):
        return {
            "mrr": 0.0,
            "mrr@5": 0.0,
            "mrr@10": 0.0,
            "mrr@25": 0.0,
            "hits@1": 0.0,
            "hits@5": 0.0,
            "hits@10": 0.0,
            "hits@25": 0.0,
            "mean_true_rank": 0.0,
            "median_true_rank": 0.0,
        }

    ranks = rank_within_queries(
        scores,
        query_ids,
    )

    positive_ranks = (
        ranks[
            targets
            == 1
        ]
    )

    metrics = {
        "mrr": (
            mean_reciprocal_rank(
                ranks,
                targets,
                query_ids,
            )
        ),

        "mrr@5": (
            mean_reciprocal_rank(
                ranks,
                targets,
                query_ids,
                cutoff=5,
            )
        ),

        "mrr@10": (
            mean_reciprocal_rank(
                ranks,
                targets,
                query_ids,
                cutoff=10,
            )
        ),

        "mrr@25": (
            mean_reciprocal_rank(
                ranks,
                targets,
                query_ids,
                cutoff=25,
            )
        ),

        "hits@1": (
            hits_at_k(
                ranks,
                targets,
                query_ids,
                1,
            )
        ),

        "hits@5": (
            hits_at_k(
                ranks,
                targets,
                query_ids,
                5,
            )
        ),

        "hits@10": (
            hits_at_k(
                ranks,
                targets,
                query_ids,
                10,
            )
        ),

        "hits@25": (
            hits_at_k(
                ranks,
                targets,
                query_ids,
                25,
            )
        ),
    }

    if len(
        positive_ranks
    ):

        metrics[
            "mean_true_rank"
        ] = float(
            np.mean(
                positive_ranks
            )
        )

        metrics[
            "median_true_rank"
        ] = float(
            np.median(
                positive_ranks
            )
        )

    else:

        metrics[
            "mean_true_rank"
        ] = 0.0

        metrics[
            "median_true_rank"
        ] = 0.0

    return metrics