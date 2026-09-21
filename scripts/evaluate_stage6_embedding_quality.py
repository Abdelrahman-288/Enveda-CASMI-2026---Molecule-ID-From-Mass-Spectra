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
    / "contrastive_smoke.pt"
)


# =========================================================
# Evaluation configuration
# =========================================================

SEED = 42

MAX_VALIDATION_SPECTRA = 20_000

BATCH_SIZE = 1024

NUM_WORKERS = 0

NUM_POSITIVE_PAIRS = 5_000
NUM_NEGATIVE_PAIRS = 5_000

COMPARISON_SAMPLES = 100_000

EMBEDDING_DIM = 128


def print_section(title: str) -> None:

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


def build_model() -> SpectrumEncoder:

    return SpectrumEncoder(
        peak_input_dim=2,
        peak_hidden_dim=128,
        precursor_hidden_dim=32,
        pooled_hidden_dim=256,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.10,
    )


def encode_subset(
    model,
    loader,
    device,
):
    """
    Encode one validation subset and return:

        embeddings [N, D]
        labels     [N]
    """

    model.eval()

    embeddings_list = []
    labels_list = []

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
                embeddings.float().cpu()
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


def create_pairs(
    labels,
):
    """
    Construct deterministic positive and negative pairs.
    """

    random.seed(
        SEED
    )

    label_to_indices = defaultdict(
        list
    )

    for index, label in enumerate(
        labels
    ):

        label_to_indices[
            int(label)
        ].append(
            index
        )

    positive_labels = [
        label
        for label, indices
        in label_to_indices.items()
        if len(indices) >= 2
    ]

    if not positive_labels:

        raise RuntimeError(
            "No validation structures have "
            "multiple spectra."
        )

    # =====================================================
    # Positive pairs
    # =====================================================

    positive_pairs = []

    while (
        len(positive_pairs)
        < NUM_POSITIVE_PAIRS
    ):

        label = random.choice(
            positive_labels
        )

        first, second = random.sample(
            label_to_indices[
                label
            ],
            2,
        )

        positive_pairs.append(
            (
                first,
                second,
            )
        )

    # =====================================================
    # Negative pairs
    # =====================================================

    all_labels = list(
        label_to_indices.keys()
    )

    negative_pairs = []

    while (
        len(negative_pairs)
        < NUM_NEGATIVE_PAIRS
    ):

        first_label, second_label = (
            random.sample(
                all_labels,
                2,
            )
        )

        first = random.choice(
            label_to_indices[
                first_label
            ]
        )

        second = random.choice(
            label_to_indices[
                second_label
            ]
        )

        negative_pairs.append(
            (
                first,
                second,
            )
        )

    return (
        positive_pairs,
        negative_pairs,
    )


def pair_similarities(
    embeddings,
    pairs,
):
    """
    Embeddings are already L2-normalized, so cosine
    similarity is simply the dot product.
    """

    first_indices = np.asarray(
        [
            first
            for first, _
            in pairs
        ],
        dtype=np.int64,
    )

    second_indices = np.asarray(
        [
            second
            for _, second
            in pairs
        ],
        dtype=np.int64,
    )

    scores = np.sum(
        embeddings[
            first_indices
        ]
        * embeddings[
            second_indices
        ],
        axis=1,
    )

    return scores


def summarize(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(
            np.mean(values)
        ),
        "median": float(
            np.median(values)
        ),
        "p10": float(
            np.percentile(
                values,
                10,
            )
        ),
        "p25": float(
            np.percentile(
                values,
                25,
            )
        ),
        "p75": float(
            np.percentile(
                values,
                75,
            )
        ),
        "p90": float(
            np.percentile(
                values,
                90,
            )
        ),
    }


def discrimination_probability(
    positive,
    negative,
):
    rng = np.random.default_rng(
        SEED
    )

    positive_sample = rng.choice(
        positive,
        size=COMPARISON_SAMPLES,
        replace=True,
    )

    negative_sample = rng.choice(
        negative,
        size=COMPARISON_SAMPLES,
        replace=True,
    )

    return float(
        np.mean(
            positive_sample
            > negative_sample
        )
    )


def evaluate_embedding_set(
    name,
    embeddings,
    labels,
    positive_pairs,
    negative_pairs,
):
    positive_scores = (
        pair_similarities(
            embeddings,
            positive_pairs,
        )
    )

    negative_scores = (
        pair_similarities(
            embeddings,
            negative_pairs,
        )
    )

    positive_summary = summarize(
        positive_scores
    )

    negative_summary = summarize(
        negative_scores
    )

    probability = (
        discrimination_probability(
            positive_scores,
            negative_scores,
        )
    )

    print_section(
        name
    )

    print(
        f"{'Metric':15s}"
        f"{'Positive':>15s}"
        f"{'Negative':>15s}"
    )

    print(
        "-" * 45
    )

    for metric in [
        "mean",
        "median",
        "p10",
        "p25",
        "p75",
        "p90",
    ]:

        print(
            f"{metric:15s}"
            f"{positive_summary[metric]:15.4f}"
            f"{negative_summary[metric]:15.4f}"
        )

    print()

    print(
        "P(same-structure similarity "
        "> different-structure similarity):"
    )

    print(
        f"{probability:.4%}"
    )

    return {
        "positive_scores": (
            positive_scores
        ),
        "negative_scores": (
            negative_scores
        ),
        "probability": (
            probability
        ),
        "positive_summary": (
            positive_summary
        ),
        "negative_summary": (
            negative_summary
        ),
    }


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 6 STEP 7 "
        "LEARNED EMBEDDING QUALITY"
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
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

    # =====================================================
    # Dataset
    # =====================================================

    validation_dataset = (
        CachedSpectrumDataset(
            CACHE_DIR,
            split="validation",
        )
    )

    subset_size = min(
        MAX_VALIDATION_SPECTRA,
        len(
            validation_dataset
        ),
    )

    # Deterministic subset.
    rng = np.random.default_rng(
        SEED
    )

    selected_indices = rng.choice(
        len(validation_dataset),
        size=subset_size,
        replace=False,
    )

    selected_indices = np.sort(
        selected_indices
    )

    subset = Subset(
        validation_dataset,
        selected_indices.tolist(),
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"Validation spectra evaluated: "
        f"{subset_size:,}"
    )

    # =====================================================
    # Random baseline
    # =====================================================

    torch.manual_seed(
        SEED
    )

    random_model = build_model().to(
        device
    )

    print(
        "\nEncoding with random model..."
    )

    random_embeddings, labels = (
        encode_subset(
            random_model,
            loader,
            device,
        )
    )

    # =====================================================
    # Trained model
    # =====================================================

    print(
        "Loading trained checkpoint..."
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    trained_model = build_model()

    trained_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    trained_model = trained_model.to(
        device
    )

    print(
        "Encoding with trained model..."
    )

    trained_embeddings, trained_labels = (
        encode_subset(
            trained_model,
            loader,
            device,
        )
    )

    if not np.array_equal(
        labels,
        trained_labels,
    ):

        raise RuntimeError(
            "Label order mismatch between "
            "random and trained encoding."
        )

    # =====================================================
    # Same pair set for both models
    # =====================================================

    positive_pairs, negative_pairs = (
        create_pairs(
            labels
        )
    )

    print(
        f"\nPositive pairs: "
        f"{len(positive_pairs):,}"
    )

    print(
        f"Negative pairs: "
        f"{len(negative_pairs):,}"
    )

    # =====================================================
    # Evaluate
    # =====================================================

    random_result = (
        evaluate_embedding_set(
            "RANDOM ENCODER",
            random_embeddings,
            labels,
            positive_pairs,
            negative_pairs,
        )
    )

    trained_result = (
        evaluate_embedding_set(
            "TRAINED ENCODER",
            trained_embeddings,
            labels,
            positive_pairs,
            negative_pairs,
        )
    )

    # =====================================================
    # Direct comparison
    # =====================================================

    print_section(
        "RANDOM VS TRAINED"
    )

    random_probability = (
        random_result[
            "probability"
        ]
    )

    trained_probability = (
        trained_result[
            "probability"
        ]
    )

    probability_change = (
        trained_probability
        - random_probability
    )

    random_gap = (
        random_result[
            "positive_summary"
        ][
            "mean"
        ]
        -
        random_result[
            "negative_summary"
        ][
            "mean"
        ]
    )

    trained_gap = (
        trained_result[
            "positive_summary"
        ][
            "mean"
        ]
        -
        trained_result[
            "negative_summary"
        ][
            "mean"
        ]
    )

    print(
        f"Random discrimination:  "
        f"{random_probability:.4%}"
    )

    print(
        f"Trained discrimination: "
        f"{trained_probability:.4%}"
    )

    print(
        f"Change:                 "
        f"{probability_change:+.4%}"
    )

    print()

    print(
        f"Random mean similarity gap:  "
        f"{random_gap:.6f}"
    )

    print(
        f"Trained mean similarity gap: "
        f"{trained_gap:.6f}"
    )

    print(
        f"Gap improvement:              "
        f"{trained_gap - random_gap:+.6f}"
    )

    print_section(
        "STAGE 6 STEP 7 COMPLETE"
    )


if __name__ == "__main__":
    main()