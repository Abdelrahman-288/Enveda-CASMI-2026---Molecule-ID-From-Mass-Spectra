from pathlib import Path
import sys

import pandas as pd


# ============================================================
# Make project root importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.train_stage10_reranker import (
    DATASET_PATH,
    MODEL_FEATURES,
    build_model_ranking_dataset,
)

from casmi.reranking.model import (
    CandidateRanker,
    RankerConfig,
    ranking_metrics,
)


# ============================================================
# Load a small sample
# ============================================================

print(
    "Loading smoke-test sample..."
)

df = pd.read_csv(
    DATASET_PATH,
    nrows=50000,
)

query_counts = (
    df.groupby(
        "query_id"
    )
    .size()
)

complete_queries = (
    query_counts.index.tolist()
)

# nrows=50000 may cut the final query in the middle.
# Remove it to make sure every retained query is complete.
if complete_queries:
    complete_queries = (
        complete_queries[:-1]
    )

df = (
    df[
        df[
            "query_id"
        ].isin(
            complete_queries
        )
    ]
    .copy()
)

query_ids = (
    df[
        "query_id"
    ]
    .drop_duplicates()
    .tolist()
)

print(
    f"Smoke queries: "
    f"{len(query_ids):,}"
)

print(
    f"Smoke rows: "
    f"{len(df):,}"
)

if len(query_ids) < 10:
    raise RuntimeError(
        "Too few complete queries for the smoke test."
    )


# ============================================================
# Query-level split
# ============================================================

split_point = int(
    len(query_ids)
    * 0.80
)

train_queries = (
    query_ids[
        :split_point
    ]
)

validation_queries = (
    query_ids[
        split_point:
    ]
)

train_df = (
    df[
        df[
            "query_id"
        ].isin(
            train_queries
        )
    ]
    .copy()
)

validation_df = (
    df[
        df[
            "query_id"
        ].isin(
            validation_queries
        )
    ]
    .copy()
)

columns = [
    "query_id",
    "candidate_id",
    "target",
    *MODEL_FEATURES,
]

train_dataset = (
    build_model_ranking_dataset(
        train_df[
            columns
        ]
    )
)

validation_dataset = (
    build_model_ranking_dataset(
        validation_df[
            columns
        ]
    )
)

print(
    "Train matrix:",
    train_dataset.X.shape,
)

print(
    "Validation matrix:",
    validation_dataset.X.shape,
)

print(
    "Train groups:",
    len(
        train_dataset.group_sizes
    ),
)

print(
    "Validation groups:",
    len(
        validation_dataset.group_sizes
    ),
)


# ============================================================
# NDCG + top-k smoke model
# ============================================================

config = RankerConfig(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5.0,
    subsample=0.90,
    colsample_bytree=0.85,
    reg_alpha=0.10,
    reg_lambda=5.0,
    random_state=42,
    tree_method="hist",
    objective="rank:ndcg",
    eval_metric="ndcg@25",
    early_stopping_rounds=25,
    lambdarank_pair_method="topk",
    lambdarank_num_pair_per_sample=25,
)

model = CandidateRanker(
    config=config
)

print(
    "\nTraining NDCG top-k smoke model..."
)

model.fit(
    train_dataset,
    validation_dataset=(
        validation_dataset
    ),
    verbose=False,
)

print(
    "Training finished."
)


# ============================================================
# Evaluate
# ============================================================

scores = model.predict(
    validation_dataset.X
)

metrics = ranking_metrics(
    scores=scores,
    targets=(
        validation_dataset.y
    ),
    query_ids=(
        validation_dataset.query_ids
    ),
)

print(
    "\nSmoke validation metrics:"
)

for (
    key,
    value,
) in metrics.items():
    print(
        f"{key}: "
        f"{value:.6f}"
    )

print(
    "\nBest iteration:",
    model.best_iteration(),
)

print(
    "Best XGBoost score:",
    model.best_score(),
)

print(
    "\nSMOKE TEST PASSED"
)