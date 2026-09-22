from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage9_testlike_group_results.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage9_hybrid_strategy_summary.txt"
)

THRESHOLDS = [
    1,
    2,
    5,
    10,
    20,
    30,
    50,
    75,
    100,
]


def calculate_metrics(ranks):

    ranks = np.asarray(
        ranks,
        dtype=np.float64,
    )

    total = len(ranks)

    finite = np.isfinite(ranks)

    reciprocal = np.zeros(
        total,
        dtype=np.float64,
    )

    reciprocal[finite] = (
        1.0 / ranks[finite]
    )

    return {
        "count": int(total),
        "coverage": float(
            np.mean(finite)
        ),
        "recall_1": float(
            np.mean(
                finite
                & (ranks <= 1)
            )
        ),
        "recall_5": float(
            np.mean(
                finite
                & (ranks <= 5)
            )
        ),
        "recall_10": float(
            np.mean(
                finite
                & (ranks <= 10)
            )
        ),
        "recall_25": float(
            np.mean(
                finite
                & (ranks <= 25)
            )
        ),
        "mrr": float(
            np.mean(reciprocal)
        ),
    }


def main():

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 9.5 HYBRID CANDIDATE STRATEGY"
    )

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"\nInput rows: {len(df):,}"
    )

    # -----------------------------------------------------
    # Separate union / intersection rows
    # -----------------------------------------------------

    id_columns = [
        "repeat",
        "seed",
        "inchikey14",
        "ingest_lib",
    ]

    union = (
        df[
            df["candidate_mode"]
            == "union"
        ]
        .copy()
        .rename(
            columns={
                "candidate_count":
                    "union_candidate_count",
                "rank":
                    "union_rank",
            }
        )
    )

    intersection = (
        df[
            df["candidate_mode"]
            == "intersection"
        ]
        .copy()
        .rename(
            columns={
                "candidate_count":
                    "intersection_candidate_count",
                "rank":
                    "intersection_rank",
            }
        )
    )

    merged = union.merge(
        intersection[
            id_columns
            + [
                "intersection_candidate_count",
                "intersection_rank",
            ]
        ],
        on=id_columns,
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Matched group evaluations: "
        f"{len(merged):,}"
    )

    summary_lines = [
        "ENVEDA CASMI 2026 — "
        "STAGE 9.5 HYBRID STRATEGY",
        "",
    ]

    # -----------------------------------------------------
    # Pure baselines
    # -----------------------------------------------------

    print(
        "\nPURE BASELINES"
    )

    print(
        "=" * 80
    )

    for label, column in [
        (
            "UNION",
            "union_rank",
        ),
        (
            "INTERSECTION",
            "intersection_rank",
        ),
    ]:

        per_repeat = []

        for repeat in sorted(
            merged["repeat"].unique()
        ):

            subset = merged[
                merged["repeat"]
                == repeat
            ]

            metrics = calculate_metrics(
                subset[column]
                .to_numpy()
            )

            per_repeat.append(
                metrics["mrr"]
            )

        mean_mrr = float(
            np.mean(per_repeat)
        )

        std_mrr = float(
            np.std(
                per_repeat,
                ddof=1,
            )
        )

        print(
            f"{label:<15} "
            f"MRR = "
            f"{mean_mrr:.6f} "
            f"± {std_mrr:.6f}"
        )

        summary_lines.append(
            f"{label}: "
            f"MRR={mean_mrr:.8f} "
            f"+/- {std_mrr:.8f}"
        )

    # -----------------------------------------------------
    # Threshold sweep
    # -----------------------------------------------------

    print(
        "\nHYBRID THRESHOLD SWEEP"
    )

    print(
        "=" * 80
    )

    print(
        "Use INTERSECTION when "
        "intersection_candidate_count >= threshold"
    )

    print(
        "Otherwise use UNION.\n"
    )

    header = (
        f"{'Threshold':>10} "
        f"{'Intersection%':>15} "
        f"{'MRR':>12} "
        f"{'Std':>12} "
        f"{'R@1':>10} "
        f"{'R@25':>10}"
    )

    print(header)
    print("-" * len(header))

    best_threshold = None
    best_mrr = -1.0
    best_result = None

    for threshold in THRESHOLDS:

        per_repeat_metrics = []

        intersection_usage = []

        for repeat in sorted(
            merged["repeat"].unique()
        ):

            subset = (
                merged[
                    merged["repeat"]
                    == repeat
                ]
                .copy()
            )

            use_intersection = (
                subset[
                    "intersection_candidate_count"
                ]
                >= threshold
            )

            hybrid_ranks = np.where(
                use_intersection,
                subset[
                    "intersection_rank"
                ],
                subset[
                    "union_rank"
                ],
            )

            metrics = (
                calculate_metrics(
                    hybrid_ranks
                )
            )

            per_repeat_metrics.append(
                metrics
            )

            intersection_usage.append(
                float(
                    use_intersection.mean()
                )
            )

        mean_mrr = float(
            np.mean(
                [
                    x["mrr"]
                    for x in
                    per_repeat_metrics
                ]
            )
        )

        std_mrr = float(
            np.std(
                [
                    x["mrr"]
                    for x in
                    per_repeat_metrics
                ],
                ddof=1,
            )
        )

        mean_r1 = float(
            np.mean(
                [
                    x["recall_1"]
                    for x in
                    per_repeat_metrics
                ]
            )
        )

        mean_r25 = float(
            np.mean(
                [
                    x["recall_25"]
                    for x in
                    per_repeat_metrics
                ]
            )
        )

        usage = float(
            np.mean(
                intersection_usage
            )
        )

        print(
            f"{threshold:>10} "
            f"{usage * 100:>14.2f}% "
            f"{mean_mrr:>12.6f} "
            f"{std_mrr:>12.6f} "
            f"{mean_r1 * 100:>9.3f}% "
            f"{mean_r25 * 100:>9.3f}%"
        )

        summary_lines.append(
            (
                f"threshold={threshold}, "
                f"intersection_usage="
                f"{usage:.8f}, "
                f"mrr={mean_mrr:.8f}, "
                f"std={std_mrr:.8f}, "
                f"recall1={mean_r1:.8f}, "
                f"recall25={mean_r25:.8f}"
            )
        )

        if mean_mrr > best_mrr:

            best_mrr = mean_mrr
            best_threshold = (
                threshold
            )

            best_result = {
                "usage": usage,
                "mrr": mean_mrr,
                "std": std_mrr,
                "r1": mean_r1,
                "r25": mean_r25,
            }

    # -----------------------------------------------------
    # Additional diagnostic:
    # intersection candidate-count distribution
    # -----------------------------------------------------

    counts = (
        merged[
            "intersection_candidate_count"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    print(
        "\nINTERSECTION CANDIDATE COUNTS"
    )

    print(
        "=" * 80
    )

    print(
        f"Min:    {counts.min()}"
    )

    print(
        f"Median: "
        f"{np.median(counts):.1f}"
    )

    print(
        f"Mean:   "
        f"{counts.mean():.2f}"
    )

    print(
        f"P10:    "
        f"{np.percentile(counts, 10):.1f}"
    )

    print(
        f"P25:    "
        f"{np.percentile(counts, 25):.1f}"
    )

    print(
        f"P75:    "
        f"{np.percentile(counts, 75):.1f}"
    )

    print(
        f"P90:    "
        f"{np.percentile(counts, 90):.1f}"
    )

    print(
        f"Max:    {counts.max()}"
    )

    # -----------------------------------------------------
    # Best result
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BEST HYBRID STRATEGY"
    )

    print(
        "=" * 80
    )

    print(
        f"Minimum intersection "
        f"candidates: "
        f"{best_threshold}"
    )

    print(
        f"Intersection usage: "
        f"{best_result['usage'] * 100:.2f}%"
    )

    print(
        f"Mean MRR: "
        f"{best_result['mrr']:.6f}"
    )

    print(
        f"MRR std: "
        f"{best_result['std']:.6f}"
    )

    print(
        f"Recall@1: "
        f"{best_result['r1'] * 100:.3f}%"
    )

    print(
        f"Recall@25: "
        f"{best_result['r25'] * 100:.3f}%"
    )

    summary_lines.extend(
        [
            "",
            (
                f"Best threshold: "
                f"{best_threshold}"
            ),
            (
                f"Best mean MRR: "
                f"{best_result['mrr']:.8f}"
            ),
            (
                f"Best MRR std: "
                f"{best_result['std']:.8f}"
            ),
            (
                f"Intersection usage: "
                f"{best_result['usage']:.8f}"
            ),
            (
                f"Recall@1: "
                f"{best_result['r1']:.8f}"
            ),
            (
                f"Recall@25: "
                f"{best_result['r25']:.8f}"
            ),
        ]
    )

    OUTPUT_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print(
        "\nSummary written to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()