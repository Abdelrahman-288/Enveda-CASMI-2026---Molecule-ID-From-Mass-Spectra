from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from casmi.models.spectrum_dataset import (
    CachedSpectrumDataset,
)

from casmi.models.spectrum_encoder import (
    SpectrumEncoder,
)


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage6"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "stage6"
    / "contrastive_full_best.pt"
)


# =========================================================
# Configuration
# =========================================================

SEED = 42

MAX_STRUCTURES = 5_000

BATCH_SIZE = 1024

NUM_WORKERS = 0

TOP_K = 25

EMBEDDING_DIM = 128

RETRIEVAL_CHUNK_SIZE = 256


def print_section(
    title: str,
) -> None:

    print(
        "\n"
        + "=" * 100
    )

    print(
        title
    )

    print(
        "=" * 100
    )


def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def build_model() -> SpectrumEncoder:
    """
    Build the same encoder architecture used during
    Stage 6 contrastive training.
    """

    return SpectrumEncoder(
        peak_input_dim=2,
        peak_hidden_dim=128,
        precursor_hidden_dim=32,
        pooled_hidden_dim=256,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.10,
    )


def encode_indices(
    model: SpectrumEncoder,
    dataset: CachedSpectrumDataset,
    indices,
    device: torch.device,
):
    """
    Encode selected dataset-local indices.

    Returns
    -------
    embeddings:
        NumPy array [N, embedding_dim]

    labels:
        NumPy array [N]
    """

    subset = Subset(
        dataset,
        list(
            indices
        ),
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    embeddings_list = []
    labels_list = []

    model.eval()

    with torch.inference_mode():

        for batch in loader:

            peaks = batch[
                "peaks"
            ].to(
                device,
                non_blocking=True,
            )

            mask = batch[
                "mask"
            ].to(
                device,
                non_blocking=True,
            )

            precursor = batch[
                "precursor_mz"
            ].to(
                device,
                non_blocking=True,
            )

            labels = batch[
                "label"
            ]

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                embeddings = model(
                    peaks,
                    mask,
                    precursor,
                )

            embeddings_list.append(
                embeddings
                .float()
                .cpu()
            )

            labels_list.append(
                labels.cpu()
            )

    embeddings = torch.cat(
        embeddings_list,
        dim=0,
    ).numpy()

    labels = torch.cat(
        labels_list,
        dim=0,
    ).numpy()

    return (
        embeddings,
        labels,
    )


def build_retrieval_split(
    dataset: CachedSpectrumDataset,
):
    """
    Construct leakage-free unseen-structure retrieval.

    For each validation molecule with >=2 spectra:

        1 spectrum       -> query
        remaining       -> reference gallery

    Validation structures were not used during Stage 6
    training.
    """

    random.seed(
        SEED
    )

    validation_labels = np.asarray(
        dataset.labels[
            dataset.indices
        ],
        dtype=np.int64,
    )

    structure_to_indices = defaultdict(
        list
    )

    for dataset_index, label in enumerate(
        validation_labels
    ):

        structure_to_indices[
            int(
                label
            )
        ].append(
            dataset_index
        )

    eligible_structures = [
        structure
        for structure, indices
        in structure_to_indices.items()
        if len(
            indices
        ) >= 2
    ]

    print(
        f"Eligible validation structures "
        f"with >=2 spectra: "
        f"{len(eligible_structures):,}"
    )

    random.shuffle(
        eligible_structures
    )

    selected_structures = (
        eligible_structures[
            :MAX_STRUCTURES
        ]
    )

    query_indices = []
    reference_indices = []

    for structure in selected_structures:

        indices = (
            structure_to_indices[
                structure
            ].copy()
        )

        random.shuffle(
            indices
        )

        query_indices.append(
            indices[
                0
            ]
        )

        reference_indices.extend(
            indices[
                1:
            ]
        )

    return (
        selected_structures,
        query_indices,
        reference_indices,
    )


def compute_metrics(
    query_embeddings,
    query_labels,
    reference_embeddings,
    reference_labels,
):
    """
    Exact cosine-similarity retrieval.

    Since SpectrumEncoder outputs L2-normalized vectors,
    cosine similarity is equivalent to dot product.

    Rank is calculated using the highest-scoring correct
    reference:

        rank =
            1 + number of references scoring higher
            than the best correct reference
    """

    ranks = []

    for start in range(
        0,
        len(
            query_embeddings
        ),
        RETRIEVAL_CHUNK_SIZE,
    ):

        end = min(
            start
            + RETRIEVAL_CHUNK_SIZE,
            len(
                query_embeddings
            ),
        )

        query_chunk = (
            query_embeddings[
                start:end
            ]
        )

        similarities = (
            query_chunk
            @ reference_embeddings.T
        )

        for local_index in range(
            similarities.shape[
                0
            ]
        ):

            global_query_index = (
                start
                + local_index
            )

            true_label = (
                query_labels[
                    global_query_index
                ]
            )

            scores = (
                similarities[
                    local_index
                ]
            )

            true_mask = (
                reference_labels
                == true_label
            )

            if not np.any(
                true_mask
            ):

                ranks.append(
                    np.inf
                )

                continue

            best_true_score = float(
                np.max(
                    scores[
                        true_mask
                    ]
                )
            )

            rank = (
                int(
                    np.sum(
                        scores
                        > best_true_score
                    )
                )
                + 1
            )

            ranks.append(
                rank
            )

    ranks = np.asarray(
        ranks,
        dtype=np.float64,
    )

    valid = np.isfinite(
        ranks
    )

    result = {}

    for k in [
        1,
        5,
        10,
        25,
    ]:

        result[
            f"Recall@{k}"
        ] = float(
            np.mean(
                valid
                & (
                    ranks
                    <= k
                )
            )
        )

    reciprocal_rank = np.where(
        valid
        & (
            ranks
            <= TOP_K
        ),
        1.0
        / ranks,
        0.0,
    )

    result[
        "MRR@25"
    ] = float(
        np.mean(
            reciprocal_rank
        )
    )

    valid_ranks = (
        ranks[
            valid
        ]
    )

    if len(
        valid_ranks
    ) > 0:

        result[
            "median_rank"
        ] = float(
            np.median(
                valid_ranks
            )
        )

        result[
            "mean_rank"
        ] = float(
            np.mean(
                valid_ranks
            )
        )

    else:

        result[
            "median_rank"
        ] = float(
            "inf"
        )

        result[
            "mean_rank"
        ] = float(
            "inf"
        )

    return (
        result,
        ranks,
    )


def print_metrics(
    name,
    result,
):

    print_section(
        name
    )

    print(
        f"Recall@1:   "
        f"{result['Recall@1']:.4%}"
    )

    print(
        f"Recall@5:   "
        f"{result['Recall@5']:.4%}"
    )

    print(
        f"Recall@10:  "
        f"{result['Recall@10']:.4%}"
    )

    print(
        f"Recall@25:  "
        f"{result['Recall@25']:.4%}"
    )

    print(
        f"MRR@25:     "
        f"{result['MRR@25']:.6f}"
    )

    print(
        f"Median rank: "
        f"{result['median_rank']:.1f}"
    )

    print(
        f"Mean rank:   "
        f"{result['mean_rank']:.2f}"
    )


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 6 STEP 10 "
        "FULL-MODEL UNSEEN-STRUCTURE RETRIEVAL"
    )

    # =====================================================
    # Reproducibility
    # =====================================================

    set_seed(
        SEED
    )

    # =====================================================
    # CUDA
    # =====================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    if not torch.cuda.is_bf16_supported():

        raise RuntimeError(
            "BF16 support is required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

    print(
        "CUDA:",
        torch.version.cuda,
    )

    print(
        "BF16 supported:",
        torch.cuda.is_bf16_supported(),
    )

    print(
        "\nCheckpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{CHECKPOINT_PATH}"
        )

    # =====================================================
    # Validation dataset
    # =====================================================

    dataset = CachedSpectrumDataset(
        CACHE_DIR,
        split="validation",
    )

    print(
        f"\nValidation spectra available: "
        f"{len(dataset):,}"
    )

    (
        structures,
        query_indices,
        reference_indices,
    ) = build_retrieval_split(
        dataset
    )

    print(
        f"\nUnseen structures evaluated: "
        f"{len(structures):,}"
    )

    print(
        f"Query spectra: "
        f"{len(query_indices):,}"
    )

    print(
        f"Reference spectra: "
        f"{len(reference_indices):,}"
    )

    # =====================================================
    # Leakage checks
    # =====================================================

    query_set = set(
        query_indices
    )

    reference_set = set(
        reference_indices
    )

    overlap = (
        query_set
        & reference_set
    )

    print(
        f"Query/reference overlap: "
        f"{len(overlap)}"
    )

    if overlap:

        raise RuntimeError(
            "Retrieval leakage detected."
        )

    # =====================================================
    # Random encoder baseline
    # =====================================================

    torch.manual_seed(
        SEED
    )

    random_model = (
        build_model()
        .to(
            device
        )
    )

    print(
        "\nEncoding random-model references..."
    )

    (
        random_reference_embeddings,
        reference_labels,
    ) = encode_indices(
        random_model,
        dataset,
        reference_indices,
        device,
    )

    print(
        "Encoding random-model queries..."
    )

    (
        random_query_embeddings,
        query_labels,
    ) = encode_indices(
        random_model,
        dataset,
        query_indices,
        device,
    )

    # =====================================================
    # Full trained model
    # =====================================================

    print(
        "\nLoading full trained checkpoint..."
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if "epoch" in checkpoint:

        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if "global_step" in checkpoint:

        print(
            f"Checkpoint global step: "
            f"{checkpoint['global_step']:,}"
        )

    if "epoch_loss" in checkpoint:

        print(
            f"Checkpoint epoch loss: "
            f"{checkpoint['epoch_loss']:.6f}"
        )

    trained_model = (
        build_model()
    )

    trained_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    trained_model = (
        trained_model.to(
            device
        )
    )

    print(
        "\nEncoding full-model references..."
    )

    (
        trained_reference_embeddings,
        trained_reference_labels,
    ) = encode_indices(
        trained_model,
        dataset,
        reference_indices,
        device,
    )

    print(
        "Encoding full-model queries..."
    )

    (
        trained_query_embeddings,
        trained_query_labels,
    ) = encode_indices(
        trained_model,
        dataset,
        query_indices,
        device,
    )

    # =====================================================
    # Label-order sanity checks
    # =====================================================

    if not np.array_equal(
        reference_labels,
        trained_reference_labels,
    ):

        raise RuntimeError(
            "Reference label mismatch between "
            "random and trained models."
        )

    if not np.array_equal(
        query_labels,
        trained_query_labels,
    ):

        raise RuntimeError(
            "Query label mismatch between "
            "random and trained models."
        )

    # =====================================================
    # Embedding sanity
    # =====================================================

    if not np.isfinite(
        random_reference_embeddings
    ).all():

        raise RuntimeError(
            "Non-finite random reference "
            "embeddings detected."
        )

    if not np.isfinite(
        random_query_embeddings
    ).all():

        raise RuntimeError(
            "Non-finite random query "
            "embeddings detected."
        )

    if not np.isfinite(
        trained_reference_embeddings
    ).all():

        raise RuntimeError(
            "Non-finite trained reference "
            "embeddings detected."
        )

    if not np.isfinite(
        trained_query_embeddings
    ).all():

        raise RuntimeError(
            "Non-finite trained query "
            "embeddings detected."
        )

    random_reference_norms = (
        np.linalg.norm(
            random_reference_embeddings,
            axis=1,
        )
    )

    trained_reference_norms = (
        np.linalg.norm(
            trained_reference_embeddings,
            axis=1,
        )
    )

    print_section(
        "EMBEDDING SANITY"
    )

    print(
        f"Random reference norm mean:  "
        f"{random_reference_norms.mean():.6f}"
    )

    print(
        f"Trained reference norm mean: "
        f"{trained_reference_norms.mean():.6f}"
    )

    # =====================================================
    # Random retrieval
    # =====================================================

    print(
        "\nRunning random-encoder retrieval..."
    )

    (
        random_result,
        random_ranks,
    ) = compute_metrics(
        random_query_embeddings,
        query_labels,
        random_reference_embeddings,
        reference_labels,
    )

    # =====================================================
    # Full-model retrieval
    # =====================================================

    print(
        "Running full-model retrieval..."
    )

    (
        trained_result,
        trained_ranks,
    ) = compute_metrics(
        trained_query_embeddings,
        query_labels,
        trained_reference_embeddings,
        reference_labels,
    )

    # =====================================================
    # Results
    # =====================================================

    print_metrics(
        "RANDOM ENCODER RETRIEVAL",
        random_result,
    )

    print_metrics(
        "FULL TRAINED ENCODER RETRIEVAL",
        trained_result,
    )

    # =====================================================
    # Random -> full model improvement
    # =====================================================

    recall_1_change = (
        trained_result[
            "Recall@1"
        ]
        - random_result[
            "Recall@1"
        ]
    )

    recall_5_change = (
        trained_result[
            "Recall@5"
        ]
        - random_result[
            "Recall@5"
        ]
    )

    recall_10_change = (
        trained_result[
            "Recall@10"
        ]
        - random_result[
            "Recall@10"
        ]
    )

    recall_25_change = (
        trained_result[
            "Recall@25"
        ]
        - random_result[
            "Recall@25"
        ]
    )

    mrr_change = (
        trained_result[
            "MRR@25"
        ]
        - random_result[
            "MRR@25"
        ]
    )

    median_rank_change = (
        trained_result[
            "median_rank"
        ]
        - random_result[
            "median_rank"
        ]
    )

    better = int(
        np.sum(
            trained_ranks
            < random_ranks
        )
    )

    worse = int(
        np.sum(
            trained_ranks
            > random_ranks
        )
    )

    same = int(
        np.sum(
            trained_ranks
            == random_ranks
        )
    )

    print_section(
        "FULL MODEL VS RANDOM BASELINE"
    )

    print(
        f"Recall@1 change:   "
        f"{recall_1_change:+.4%}"
    )

    print(
        f"Recall@5 change:   "
        f"{recall_5_change:+.4%}"
    )

    print(
        f"Recall@10 change:  "
        f"{recall_10_change:+.4%}"
    )

    print(
        f"Recall@25 change:  "
        f"{recall_25_change:+.4%}"
    )

    print(
        f"MRR@25 change:     "
        f"{mrr_change:+.6f}"
    )

    print(
        f"Median rank change: "
        f"{median_rank_change:+.1f}"
    )

    print()

    print(
        f"Queries improved:  "
        f"{better:,}"
    )

    print(
        f"Queries worsened:   "
        f"{worse:,}"
    )

    print(
        f"Queries unchanged:  "
        f"{same:,}"
    )

    # =====================================================
    # Smoke-model benchmark
    # =====================================================

    smoke_recall_1 = 0.1538
    smoke_recall_5 = 0.2520
    smoke_recall_10 = 0.3110
    smoke_recall_25 = 0.4094
    smoke_mrr_25 = 0.203004
    smoke_median_rank = 53.0
    smoke_mean_rank = 330.62

    full_vs_smoke_recall_1 = (
        trained_result[
            "Recall@1"
        ]
        - smoke_recall_1
    )

    full_vs_smoke_recall_5 = (
        trained_result[
            "Recall@5"
        ]
        - smoke_recall_5
    )

    full_vs_smoke_recall_10 = (
        trained_result[
            "Recall@10"
        ]
        - smoke_recall_10
    )

    full_vs_smoke_recall_25 = (
        trained_result[
            "Recall@25"
        ]
        - smoke_recall_25
    )

    full_vs_smoke_mrr = (
        trained_result[
            "MRR@25"
        ]
        - smoke_mrr_25
    )

    full_vs_smoke_median = (
        trained_result[
            "median_rank"
        ]
        - smoke_median_rank
    )

    full_vs_smoke_mean = (
        trained_result[
            "mean_rank"
        ]
        - smoke_mean_rank
    )

    print_section(
        "FULL MODEL VS 500-STEP SMOKE MODEL"
    )

    print(
        "500-step smoke baseline:"
    )

    print(
        f"Recall@1:   "
        f"{smoke_recall_1:.4%}"
    )

    print(
        f"Recall@5:   "
        f"{smoke_recall_5:.4%}"
    )

    print(
        f"Recall@10:  "
        f"{smoke_recall_10:.4%}"
    )

    print(
        f"Recall@25:  "
        f"{smoke_recall_25:.4%}"
    )

    print(
        f"MRR@25:     "
        f"{smoke_mrr_25:.6f}"
    )

    print(
        f"Median rank: "
        f"{smoke_median_rank:.1f}"
    )

    print(
        f"Mean rank:   "
        f"{smoke_mean_rank:.2f}"
    )

    print()

    print(
        "Full-training improvement over smoke:"
    )

    print(
        f"Recall@1 change:   "
        f"{full_vs_smoke_recall_1:+.4%}"
    )

    print(
        f"Recall@5 change:   "
        f"{full_vs_smoke_recall_5:+.4%}"
    )

    print(
        f"Recall@10 change:  "
        f"{full_vs_smoke_recall_10:+.4%}"
    )

    print(
        f"Recall@25 change:  "
        f"{full_vs_smoke_recall_25:+.4%}"
    )

    print(
        f"MRR@25 change:     "
        f"{full_vs_smoke_mrr:+.6f}"
    )

    print(
        f"Median rank change: "
        f"{full_vs_smoke_median:+.1f}"
    )

    print(
        f"Mean rank change:   "
        f"{full_vs_smoke_mean:+.2f}"
    )

    # =====================================================
    # Final summary
    # =====================================================

    print_section(
        "STAGE 6 STEP 10 COMPLETE"
    )


if __name__ == "__main__":
    main()