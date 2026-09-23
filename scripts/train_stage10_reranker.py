from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from casmi.reranking.dataset import (
    RankingDataset,
    compute_group_sizes,
    sort_ranking_dataframe,
    split_by_query_ids,
)

from casmi.reranking.model import (
    CandidateRanker,
    RankerConfig,
    ranking_metrics,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage10_reranker_dataset.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "stage10"
)

BEST_MODEL_PATH = (
    MODEL_DIR
    / "best_reranker.json"
)

EXPERIMENT_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage10_experiment_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage10_feature_importance.csv"
)

VALIDATION_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage10_validation_predictions.csv"
)

TRAIN_SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage10_train_queries.csv"
)

VALIDATION_SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage10_validation_queries.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage10_training_summary.txt"
)


# ============================================================
# Clean model features
# ============================================================

MODEL_FEATURES = [
    "mean_score",
    "max_score",
    "top2_mean_score",
    "mass_error_ppm",
    "abs_mass_error_ppm",
    "num_supporting_spectra",
    "retrieval_rank",
    "similarity_rank",
    "neural_rank",
    "score_std",
    "score_range",
]


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20


# ============================================================
# Experiments
# ============================================================

EXPERIMENTS = [
    # --------------------------------------------------------
    # 1. LambdaMART NDCG focused directly on top 25
    # --------------------------------------------------------
    {
        "name": "ndcg_topk25_depth6",
        "config": RankerConfig(
            n_estimators=3000,
            learning_rate=0.025,
            max_depth=6,
            min_child_weight=5.0,
            subsample=0.90,
            colsample_bytree=0.85,
            reg_alpha=0.10,
            reg_lambda=5.0,
            random_state=RANDOM_SEED,
            tree_method="hist",
            objective="rank:ndcg",
            eval_metric="ndcg@25",
            early_stopping_rounds=150,
            lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=25,
        ),
    },

    # --------------------------------------------------------
    # 2. LambdaMART with slightly wider top-k focus
    # --------------------------------------------------------
    {
        "name": "ndcg_topk32_depth6",
        "config": RankerConfig(
            n_estimators=3000,
            learning_rate=0.025,
            max_depth=6,
            min_child_weight=5.0,
            subsample=0.90,
            colsample_bytree=0.85,
            reg_alpha=0.10,
            reg_lambda=5.0,
            random_state=RANDOM_SEED,
            tree_method="hist",
            objective="rank:ndcg",
            eval_metric="ndcg@25",
            early_stopping_rounds=150,
            lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=32,
        ),
    },

    # --------------------------------------------------------
    # 3. LambdaMART with mean pair sampling
    # --------------------------------------------------------
    {
        "name": "ndcg_mean16_depth6",
        "config": RankerConfig(
            n_estimators=3000,
            learning_rate=0.025,
            max_depth=6,
            min_child_weight=5.0,
            subsample=0.90,
            colsample_bytree=0.85,
            reg_alpha=0.10,
            reg_lambda=5.0,
            random_state=RANDOM_SEED,
            tree_method="hist",
            objective="rank:ndcg",
            eval_metric="ndcg@25",
            early_stopping_rounds=150,
            lambdarank_pair_method="mean",
            lambdarank_num_pair_per_sample=16,
        ),
    },

    # --------------------------------------------------------
    # 4. Pairwise reference model
    # --------------------------------------------------------
    {
        "name": "pairwise_depth6",
        "config": RankerConfig(
            n_estimators=3000,
            learning_rate=0.025,
            max_depth=6,
            min_child_weight=5.0,
            subsample=0.90,
            colsample_bytree=0.85,
            reg_alpha=0.10,
            reg_lambda=5.0,
            random_state=RANDOM_SEED,
            tree_method="hist",
            objective="rank:pairwise",
            eval_metric="ndcg@25",
            early_stopping_rounds=150,
        ),
    },
]


# ============================================================
# Helpers
# ============================================================

def molecule_from_query(
    query_id: str,
) -> str:
    """
    Extract molecule identifier from:

        inchikey14|ingest_lib
    """

    return str(query_id).split("|", 1)[0]


def make_molecule_disjoint_split(
    dataframe: pd.DataFrame,
):
    """
    Deterministic molecule-level train/validation split.

    This prevents the same molecule from appearing
    on both sides through different library queries.
    """

    query_table = (
        dataframe[
            [
                "query_id",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    query_table[
        "molecule_id"
    ] = (
        query_table[
            "query_id"
        ]
        .astype(str)
        .map(
            molecule_from_query
        )
    )

    molecules = (
        query_table[
            "molecule_id"
        ]
        .drop_duplicates()
        .to_numpy(
            copy=True
        )
    )

    if len(molecules) < 2:
        raise RuntimeError(
            "Not enough molecules for train/validation split."
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rng.shuffle(
        molecules
    )

    validation_count = int(
        round(
            len(molecules)
            * VALIDATION_FRACTION
        )
    )

    validation_count = max(
        1,
        validation_count,
    )

    validation_count = min(
        validation_count,
        len(molecules) - 1,
    )

    validation_molecules = set(
        molecules[
            :validation_count
        ]
    )

    train_molecules = set(
        molecules[
            validation_count:
        ]
    )

    train_query_ids = (
        query_table[
            query_table[
                "molecule_id"
            ].isin(
                train_molecules
            )
        ][
            "query_id"
        ]
        .astype(str)
        .tolist()
    )

    validation_query_ids = (
        query_table[
            query_table[
                "molecule_id"
            ].isin(
                validation_molecules
            )
        ][
            "query_id"
        ]
        .astype(str)
        .tolist()
    )

    return (
        train_query_ids,
        validation_query_ids,
        query_table,
    )


def build_model_ranking_dataset(
    dataframe: pd.DataFrame,
) -> RankingDataset:
    """
    Build an XGBoost ranking dataset using only
    the 11 selected Stage 10 features.
    """

    required_columns = [
        "query_id",
        "candidate_id",
        "target",
        *MODEL_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    df = (
        dataframe[
            required_columns
        ]
        .copy()
    )

    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------

    if df[
        "query_id"
    ].isna().any():
        raise ValueError(
            "query_id contains missing values."
        )

    if df[
        "candidate_id"
    ].isna().any():
        raise ValueError(
            "candidate_id contains missing values."
        )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    if df[
        "target"
    ].isna().any():
        raise ValueError(
            "target contains missing values."
        )

    if not df[
        "target"
    ].isin(
        [0, 1]
    ).all():
        raise ValueError(
            "target must contain only 0 or 1."
        )

    # --------------------------------------------------------
    # Duplicate query/candidate pairs
    # --------------------------------------------------------

    if df.duplicated(
        subset=[
            "query_id",
            "candidate_id",
        ]
    ).any():
        raise ValueError(
            "Duplicate query/candidate pairs detected."
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_values = (
        df[
            MODEL_FEATURES
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    if not np.isfinite(
        feature_values
    ).all():
        raise ValueError(
            "MODEL_FEATURES contain non-finite values."
        )

    # --------------------------------------------------------
    # Sort queries contiguously
    # --------------------------------------------------------

    df = sort_ranking_dataframe(
        df
    )

    # --------------------------------------------------------
    # Build ranking arrays
    # --------------------------------------------------------

    X = (
        df[
            MODEL_FEATURES
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y = (
        df[
            "target"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    query_ids = (
        df[
            "query_id"
        ]
        .astype(str)
        .to_numpy()
    )

    candidate_ids = (
        df[
            "candidate_id"
        ]
        .astype(str)
        .to_numpy()
    )

    group_sizes = (
        compute_group_sizes(
            query_ids
        )
    )

    if int(
        group_sizes.sum()
    ) != len(df):
        raise RuntimeError(
            "Ranking group sizes do not sum "
            "to candidate row count."
        )

    return RankingDataset(
        X=X,
        y=y,
        group_sizes=group_sizes,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
    )


def compute_metrics_from_dataframe(
    dataframe: pd.DataFrame,
    score_column: str,
):
    """
    Calculate Stage 10 metrics using a dataframe score column.
    """

    return ranking_metrics(
        scores=(
            dataframe[
                score_column
            ]
            .to_numpy(
                dtype=np.float64
            )
        ),
        targets=(
            dataframe[
                "target"
            ]
            .to_numpy(
                dtype=np.int32
            )
        ),
        query_ids=(
            dataframe[
                "query_id"
            ]
            .astype(str)
            .to_numpy()
        ),
    )


def print_metrics(
    title: str,
    metrics,
):
    print(
        f"\n{title}"
    )

    print(
        "-" * 65
    )

    for (
        key,
        value,
    ) in metrics.items():
        print(
            f"{key:20s}: "
            f"{value:.6f}"
        )


def metric_improvement(
    new_metrics,
    baseline_metrics,
    key: str,
) -> float:
    return float(
        new_metrics[
            key
        ]
        - baseline_metrics[
            key
        ]
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n"
        + "=" * 90
    )

    print(
        "ENVEDA CASMI 2026 — "
        "STAGE 10 LEARNED CANDIDATE RERANKING"
    )

    print(
        "=" * 90
    )

    # ========================================================
    # Load dataset
    # ========================================================

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print(
        "\nLoading Stage 10 dataset..."
    )

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"Rows: "
        f"{len(dataframe):,}"
    )

    print(
        f"Queries: "
        f"{dataframe['query_id'].nunique():,}"
    )

    print(
        f"Positive rows: "
        f"{int(dataframe['target'].sum()):,}"
    )

    # ========================================================
    # Required feature validation
    # ========================================================

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Missing model features: "
            f"{missing_features}"
        )

    feature_matrix = (
        dataframe[
            MODEL_FEATURES
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    if not np.isfinite(
        feature_matrix
    ).all():
        raise RuntimeError(
            "Non-finite values found in model features."
        )

    # ========================================================
    # Positive-label validation
    # ========================================================

    positive_counts = (
        dataframe
        .groupby(
            "query_id",
            sort=False,
        )[
            "target"
        ]
        .sum()
    )

    invalid_queries = int(
        (
            positive_counts != 1
        )
        .sum()
    )

    print(
        f"Invalid positive-count queries: "
        f"{invalid_queries:,}"
    )

    if invalid_queries:
        raise RuntimeError(
            "Each query must contain exactly one positive."
        )

    # ========================================================
    # Molecule-disjoint split
    # ========================================================

    print(
        "\nCreating molecule-disjoint split..."
    )

    (
        train_query_ids,
        validation_query_ids,
        query_table,
    ) = (
        make_molecule_disjoint_split(
            dataframe
        )
    )

    train_df, validation_df = (
        split_by_query_ids(
            dataframe,
            train_query_ids=(
                train_query_ids
            ),
            validation_query_ids=(
                validation_query_ids
            ),
        )
    )

    print(
        f"Train queries: "
        f"{train_df['query_id'].nunique():,}"
    )

    print(
        f"Validation queries: "
        f"{validation_df['query_id'].nunique():,}"
    )

    print(
        f"Train rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    # ========================================================
    # Leakage validation
    # ========================================================

    train_molecules = set(
        train_df[
            "query_id"
        ]
        .astype(str)
        .map(
            molecule_from_query
        )
    )

    validation_molecules = set(
        validation_df[
            "query_id"
        ]
        .astype(str)
        .map(
            molecule_from_query
        )
    )

    molecule_overlap = (
        train_molecules
        & validation_molecules
    )

    print(
        f"Train molecules: "
        f"{len(train_molecules):,}"
    )

    print(
        f"Validation molecules: "
        f"{len(validation_molecules):,}"
    )

    print(
        f"Molecule overlap: "
        f"{len(molecule_overlap):,}"
    )

    if molecule_overlap:
        raise RuntimeError(
            "Molecule leakage detected."
        )

    # ========================================================
    # Output directories
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPERIMENT_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Save split
    # ========================================================

    train_query_table = (
        query_table[
            query_table[
                "query_id"
            ].isin(
                train_query_ids
            )
        ]
        .copy()
    )

    validation_query_table = (
        query_table[
            query_table[
                "query_id"
            ].isin(
                validation_query_ids
            )
        ]
        .copy()
    )

    train_query_table.to_csv(
        TRAIN_SPLIT_PATH,
        index=False,
    )

    validation_query_table.to_csv(
        VALIDATION_SPLIT_PATH,
        index=False,
    )

    # ========================================================
    # Stage 9 validation baseline
    # ========================================================

    baseline_metrics = (
        compute_metrics_from_dataframe(
            validation_df,
            "mean_score",
        )
    )

    print_metrics(
        "Stage 9 validation baseline",
        baseline_metrics,
    )

    # ========================================================
    # Select model columns
    # ========================================================

    model_columns = [
        "query_id",
        "candidate_id",
        "target",
        *MODEL_FEATURES,
    ]

    train_model_df = (
        train_df[
            model_columns
        ]
        .copy()
    )

    validation_model_df = (
        validation_df[
            model_columns
        ]
        .copy()
    )

    # ========================================================
    # Ranking datasets
    # ========================================================

    print(
        "\nBuilding ranking datasets..."
    )

    train_dataset = (
        build_model_ranking_dataset(
            train_model_df
        )
    )

    validation_dataset = (
        build_model_ranking_dataset(
            validation_model_df
        )
    )

    print(
        f"Train ranking matrix: "
        f"{train_dataset.X.shape}"
    )

    print(
        f"Validation ranking matrix: "
        f"{validation_dataset.X.shape}"
    )

    print(
        f"Features: "
        f"{train_dataset.X.shape[1]}"
    )

    print(
        f"Train groups: "
        f"{len(train_dataset.group_sizes):,}"
    )

    print(
        f"Validation groups: "
        f"{len(validation_dataset.group_sizes):,}"
    )

    # ========================================================
    # Training experiments
    # ========================================================

    experiment_rows = []

    trained_models = {}

    best_model = None
    best_model_name = None
    best_metrics = None

    best_selection_score = -np.inf

    print(
        f"\nExperiments to run: "
        f"{len(EXPERIMENTS)}"
    )

    for (
        experiment_index,
        experiment,
    ) in enumerate(
        EXPERIMENTS,
        start=1,
    ):

        name = (
            experiment[
                "name"
            ]
        )

        config = (
            experiment[
                "config"
            ]
        )

        print(
            "\n"
            + "=" * 90
        )

        print(
            f"EXPERIMENT "
            f"{experiment_index}/"
            f"{len(EXPERIMENTS)}"
        )

        print(
            name
        )

        print(
            "=" * 90
        )

        print(
            f"Objective: "
            f"{config.objective}"
        )

        print(
            f"Pair method: "
            f"{config.lambdarank_pair_method}"
        )

        print(
            f"Pairs/top-k: "
            f"{config.lambdarank_num_pair_per_sample}"
        )

        print(
            f"Max trees: "
            f"{config.n_estimators}"
        )

        print(
            f"Learning rate: "
            f"{config.learning_rate}"
        )

        print(
            f"Depth: "
            f"{config.max_depth}"
        )

        print(
            f"Early stopping: "
            f"{config.early_stopping_rounds}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model = CandidateRanker(
            config=config
        )

        model.fit(
            train_dataset,
            validation_dataset=(
                validation_dataset
            ),
            verbose=False,
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        validation_scores = (
            model.predict(
                validation_dataset.X
            )
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = ranking_metrics(
            scores=(
                validation_scores
            ),
            targets=(
                validation_dataset.y
            ),
            query_ids=(
                validation_dataset.query_ids
            ),
        )

        print_metrics(
            "Validation metrics",
            metrics,
        )

        best_iteration = (
            model.best_iteration()
        )

        xgb_best_score = (
            model.best_score()
        )

        print(
            "\nXGBoost internal validation:"
        )

        print(
            f"Best iteration: "
            f"{best_iteration}"
        )

        print(
            f"Best NDCG@25: "
            f"{xgb_best_score}"
        )

        # ----------------------------------------------------
        # Improvement vs baseline
        # ----------------------------------------------------

        improvement_mrr = (
            metric_improvement(
                metrics,
                baseline_metrics,
                "mrr",
            )
        )

        improvement_mrr25 = (
            metric_improvement(
                metrics,
                baseline_metrics,
                "mrr@25",
            )
        )

        improvement_hits1 = (
            metric_improvement(
                metrics,
                baseline_metrics,
                "hits@1",
            )
        )

        improvement_hits25 = (
            metric_improvement(
                metrics,
                baseline_metrics,
                "hits@25",
            )
        )

        print(
            "\nImprovement over Stage 9:"
        )

        print(
            f"MRR: "
            f"{improvement_mrr:+.6f}"
        )

        print(
            f"MRR@25: "
            f"{improvement_mrr25:+.6f}"
        )

        print(
            f"Hits@1: "
            f"{improvement_hits1:+.6f}"
        )

        print(
            f"Hits@25: "
            f"{improvement_hits25:+.6f}"
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        experiment_rows.append(
            {
                "experiment": (
                    name
                ),
                "objective": (
                    config.objective
                ),
                "pair_method": (
                    config
                    .lambdarank_pair_method
                ),
                "num_pair_per_sample": (
                    config
                    .lambdarank_num_pair_per_sample
                ),
                "n_estimators": (
                    config.n_estimators
                ),
                "learning_rate": (
                    config.learning_rate
                ),
                "max_depth": (
                    config.max_depth
                ),
                "min_child_weight": (
                    config.min_child_weight
                ),
                "subsample": (
                    config.subsample
                ),
                "colsample_bytree": (
                    config.colsample_bytree
                ),
                "reg_alpha": (
                    config.reg_alpha
                ),
                "reg_lambda": (
                    config.reg_lambda
                ),
                "early_stopping_rounds": (
                    config
                    .early_stopping_rounds
                ),
                "best_iteration": (
                    best_iteration
                ),
                "xgb_best_ndcg25": (
                    xgb_best_score
                ),
                "mrr_improvement": (
                    improvement_mrr
                ),
                "mrr25_improvement": (
                    improvement_mrr25
                ),
                "hits1_improvement": (
                    improvement_hits1
                ),
                "hits25_improvement": (
                    improvement_hits25
                ),
                **metrics,
            }
        )

        trained_models[
            name
        ] = (
            model,
            validation_scores,
            metrics,
        )

        # ----------------------------------------------------
        # Select best model using MRR@25
        # ----------------------------------------------------

        selection_score = float(
            metrics[
                "mrr@25"
            ]
        )

        if (
            selection_score
            > best_selection_score
        ):
            best_selection_score = (
                selection_score
            )

            best_model = (
                model
            )

            best_model_name = (
                name
            )

            best_metrics = (
                metrics
            )

    # ========================================================
    # Validate experiments
    # ========================================================

    if (
        best_model is None
        or best_model_name is None
        or best_metrics is None
    ):
        raise RuntimeError(
            "No Stage 10 experiment completed."
        )

    # ========================================================
    # Save experiment table
    # ========================================================

    experiment_df = pd.DataFrame(
        experiment_rows
    )

    experiment_df = (
        experiment_df
        .sort_values(
            by=[
                "mrr@25",
                "hits@1",
                "mrr",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    experiment_df.to_csv(
        EXPERIMENT_RESULTS_PATH,
        index=False,
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "EXPERIMENT RANKING"
    )

    print(
        "=" * 90
    )

    print(
        experiment_df[
            [
                "experiment",
                "objective",
                "pair_method",
                "num_pair_per_sample",
                "mrr",
                "mrr@25",
                "hits@1",
                "hits@5",
                "hits@10",
                "hits@25",
                "mean_true_rank",
                "median_true_rank",
                "best_iteration",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Save best model
    # ========================================================

    best_model.save(
        BEST_MODEL_PATH
    )

    # ========================================================
    # Validation predictions
    # ========================================================

    (
        _,
        best_validation_scores,
        _,
    ) = trained_models[
        best_model_name
    ]

    prediction_df = (
        sort_ranking_dataframe(
            validation_model_df
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        prediction_df
    ) != len(
        best_validation_scores
    ):
        raise RuntimeError(
            "Prediction row count mismatch."
        )

    prediction_df[
        "stage9_score"
    ] = (
        prediction_df[
            "mean_score"
        ]
    )

    prediction_df[
        "stage10_score"
    ] = (
        best_validation_scores
    )

    prediction_df.to_csv(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
    )

    # ========================================================
    # Feature importance
    # ========================================================

    importance = (
        best_model
        .feature_importance()
    )

    if len(
        importance
    ) != len(
        MODEL_FEATURES
    ):
        raise RuntimeError(
            "Feature importance length mismatch."
        )

    importance_df = (
        pd.DataFrame(
            {
                "feature": (
                    MODEL_FEATURES
                ),
                "importance": (
                    importance
                ),
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    # ========================================================
    # Summary file
    # ========================================================

    summary_lines = [
        (
            "ENVEDA CASMI 2026 - "
            "STAGE 10 LEARNED CANDIDATE RERANKING"
        ),
        "=" * 78,
        "",
        (
            f"Dataset rows: "
            f"{len(dataframe):,}"
        ),
        (
            f"Total queries: "
            f"{dataframe['query_id'].nunique():,}"
        ),
        (
            f"Train queries: "
            f"{train_df['query_id'].nunique():,}"
        ),
        (
            f"Validation queries: "
            f"{validation_df['query_id'].nunique():,}"
        ),
        (
            f"Train molecules: "
            f"{len(train_molecules):,}"
        ),
        (
            f"Validation molecules: "
            f"{len(validation_molecules):,}"
        ),
        (
            f"Molecule overlap: "
            f"{len(molecule_overlap):,}"
        ),
        "",
        (
            f"Feature count: "
            f"{len(MODEL_FEATURES)}"
        ),
        (
            f"Experiments: "
            f"{len(EXPERIMENTS)}"
        ),
        "",
        (
            f"Best experiment: "
            f"{best_model_name}"
        ),
        (
            f"Best iteration: "
            f"{best_model.best_iteration()}"
        ),
        (
            f"Best internal NDCG@25: "
            f"{best_model.best_score()}"
        ),
        "",
        "Stage 9 validation baseline:",
    ]

    for (
        key,
        value,
    ) in baseline_metrics.items():
        summary_lines.append(
            f"  {key}: "
            f"{value:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "Stage 10 best model:",
        ]
    )

    for (
        key,
        value,
    ) in best_metrics.items():
        summary_lines.append(
            f"  {key}: "
            f"{value:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "Improvement over Stage 9:",
            (
                f"  MRR: "
                f"{metric_improvement(best_metrics, baseline_metrics, 'mrr'):+.6f}"
            ),
            (
                f"  MRR@5: "
                f"{metric_improvement(best_metrics, baseline_metrics, 'mrr@5'):+.6f}"
            ),
            (
                f"  MRR@10: "
                f"{metric_improvement(best_metrics, baseline_metrics, 'mrr@10'):+.6f}"
            ),
            (
                f"  MRR@25: "
                f"{metric_improvement(best_metrics, baseline_metrics, 'mrr@25'):+.6f}"
            ),
            (
                f"  Hits@1: "
                f"{metric_improvement(best_metrics, baseline_metrics, 'hits@1'):+.6f}"
            ),
            (
                f"  Hits@5: "
                f"{metric_improvement(best_metrics, baseline_metrics, 'hits@5'):+.6f}"
            ),
            (
                f"  Hits@10: "
                f"{metric_improvement(best_metrics, baseline_metrics, 'hits@10'):+.6f}"
            ),
            (
                f"  Hits@25: "
                f"{metric_improvement(best_metrics, baseline_metrics, 'hits@25'):+.6f}"
            ),
            "",
            "Model features:",
        ]
    )

    for feature in MODEL_FEATURES:
        summary_lines.append(
            f"  - {feature}"
        )

    summary_lines.extend(
        [
            "",
            "Feature importance:",
        ]
    )

    for row in (
        importance_df
        .itertuples(
            index=False
        )
    ):
        summary_lines.append(
            f"  {row.feature}: "
            f"{row.importance:.8f}"
        )

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Final report
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "STAGE 10 TRAINING COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        f"\nBest experiment: "
        f"{best_model_name}"
    )

    print(
        f"Best iteration: "
        f"{best_model.best_iteration()}"
    )

    print_metrics(
        "Stage 9 validation baseline",
        baseline_metrics,
    )

    print_metrics(
        "Stage 10 best model",
        best_metrics,
    )

    print(
        "\nImprovement over Stage 9"
    )

    print(
        "-" * 65
    )

    print(
        f"MRR: "
        f"{metric_improvement(best_metrics, baseline_metrics, 'mrr'):+.6f}"
    )

    print(
        f"MRR@25: "
        f"{metric_improvement(best_metrics, baseline_metrics, 'mrr@25'):+.6f}"
    )

    print(
        f"Hits@1: "
        f"{metric_improvement(best_metrics, baseline_metrics, 'hits@1'):+.6f}"
    )

    print(
        f"Hits@5: "
        f"{metric_improvement(best_metrics, baseline_metrics, 'hits@5'):+.6f}"
    )

    print(
        f"Hits@10: "
        f"{metric_improvement(best_metrics, baseline_metrics, 'hits@10'):+.6f}"
    )

    print(
        f"Hits@25: "
        f"{metric_improvement(best_metrics, baseline_metrics, 'hits@25'):+.6f}"
    )

    print(
        "\nFeature importance:"
    )

    print(
        importance_df.to_string(
            index=False
        )
    )

    print(
        f"\nBest model:\n"
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"\nExperiment results:\n"
        f"{EXPERIMENT_RESULTS_PATH}"
    )

    print(
        f"\nFeature importance:\n"
        f"{FEATURE_IMPORTANCE_PATH}"
    )

    print(
        f"\nValidation predictions:\n"
        f"{VALIDATION_PREDICTIONS_PATH}"
    )

    print(
        f"\nTraining split:\n"
        f"{TRAIN_SPLIT_PATH}"
    )

    print(
        f"\nValidation split:\n"
        f"{VALIDATION_SPLIT_PATH}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()