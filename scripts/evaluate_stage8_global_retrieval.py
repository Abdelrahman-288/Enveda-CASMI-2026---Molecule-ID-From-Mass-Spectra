from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from casmi.models.spectrum_encoder import SpectrumEncoder
from casmi.models.molecule_encoder import MoleculeEncoder

from casmi.training.stage8_dataset import (
    SpectrumMoleculeDataset,
)


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

STAGE6_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage6"
)

STAGE8_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage8"
)

STAGE6_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "stage6"
    / "contrastive_full_best.pt"
)

STAGE8_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "stage8"
    / "cross_modal_frozen_best.pt"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage8_global_retrieval_summary.txt"
)


# =========================================================
# Configuration
# =========================================================

BATCH_SIZE = 512

NUM_WORKERS = 0

EMBEDDING_DIM = 128


# =========================================================
# Builders
# =========================================================

def build_spectrum_encoder():

    return SpectrumEncoder(
        peak_input_dim=2,
        peak_hidden_dim=128,
        precursor_hidden_dim=32,
        pooled_hidden_dim=256,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.10,
    )


def build_molecule_encoder():

    return MoleculeEncoder(
        fingerprint_dim=2048,
        hidden_dim_1=1024,
        hidden_dim_2=512,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.10,
    )


# =========================================================
# Spectrum preparation
# =========================================================

def prepare_peaks(
    batch,
    device,
):

    mzs = batch[
        "mzs"
    ].to(
        device,
        non_blocking=True,
    )

    intensities = batch[
        "intensities"
    ].to(
        device,
        non_blocking=True,
    )

    return torch.stack(
        [
            mzs,
            intensities,
        ],
        dim=-1,
    )


# =========================================================
# Main
# =========================================================

def main():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 8 GLOBAL STRUCTURE-DISJOINT RETRIEVAL"
    )

    print(
        f"\nGPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    start_time = time.perf_counter()

    # =====================================================
    # Dataset
    # =====================================================

    print(
        "\nLoading validation dataset..."
    )

    dataset = SpectrumMoleculeDataset(
        stage6_dir=STAGE6_DIR,
        stage8_dir=STAGE8_DIR,
        split_value=1,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"Validation spectra: "
        f"{len(dataset):,}"
    )

    # =====================================================
    # Validation candidate molecule universe
    # =====================================================

    validation_spectrum_indices = (
        dataset.indices
    )

    validation_molecule_indices = (
        dataset.spectrum_molecule_indices[
            validation_spectrum_indices
        ]
    )

    unique_validation_molecules = (
        np.unique(
            validation_molecule_indices
        )
    )

    print(
        f"Validation molecules: "
        f"{len(unique_validation_molecules):,}"
    )

    # Map global Stage 8 molecule index -> local candidate index

    global_to_local = {
        int(global_index): local_index
        for local_index, global_index
        in enumerate(
            unique_validation_molecules
        )
    }

    # =====================================================
    # Models
    # =====================================================

    print(
        "\nLoading Stage 6 spectrum encoder..."
    )

    spectrum_encoder = (
        build_spectrum_encoder()
    )

    stage6_checkpoint = torch.load(
        STAGE6_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    spectrum_encoder.load_state_dict(
        stage6_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    spectrum_encoder = (
        spectrum_encoder
        .to(
            device
        )
        .eval()
    )

    print(
        "Loading Stage 8 molecule encoder..."
    )

    molecule_encoder = (
        build_molecule_encoder()
    )

    stage8_checkpoint = torch.load(
        STAGE8_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    molecule_encoder.load_state_dict(
        stage8_checkpoint[
            "molecule_encoder_state_dict"
        ],
        strict=True,
    )

    molecule_encoder = (
        molecule_encoder
        .to(
            device
        )
        .eval()
    )

    # =====================================================
    # Encode all validation molecules
    # =====================================================

    print(
        "\nEncoding validation candidate molecules..."
    )

    fingerprints = dataset.fingerprints

    molecule_embedding_batches = []

    molecule_batch_size = 2048

    with torch.inference_mode():

        for start in range(
            0,
            len(
                unique_validation_molecules
            ),
            molecule_batch_size,
        ):

            end = min(
                start
                + molecule_batch_size,
                len(
                    unique_validation_molecules
                ),
            )

            molecule_indices = (
                unique_validation_molecules[
                    start:end
                ]
            )

            fp_numpy = np.asarray(
                fingerprints[
                    molecule_indices
                ],
                dtype=np.uint8,
            )

            fp_tensor = torch.tensor(
                fp_numpy,
                dtype=torch.uint8,
                device=device,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                embeddings = (
                    molecule_encoder(
                        fp_tensor
                    )
                )

            molecule_embedding_batches.append(
                embeddings.float()
            )

    molecule_embeddings = torch.cat(
        molecule_embedding_batches,
        dim=0,
    )

    print(
        f"Molecule embedding matrix: "
        f"{tuple(molecule_embeddings.shape)}"
    )

    # =====================================================
    # Global spectrum -> molecule ranking
    # =====================================================

    print(
        "\nEvaluating global retrieval..."
    )

    all_ranks = []

    processed = 0

    eval_start = time.perf_counter()

    with torch.inference_mode():

        for batch_number, batch in enumerate(
            loader,
            start=1,
        ):

            peaks = prepare_peaks(
                batch,
                device,
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

            global_true_indices = (
                batch[
                    "molecule_index"
                ]
                .cpu()
                .numpy()
            )

            local_true_indices = np.array(
                [
                    global_to_local[
                        int(index)
                    ]
                    for index
                    in global_true_indices
                ],
                dtype=np.int64,
            )

            local_true_tensor = torch.tensor(
                local_true_indices,
                dtype=torch.long,
                device=device,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                spectrum_embeddings = (
                    spectrum_encoder(
                        peaks,
                        mask,
                        precursor,
                    )
                )

            spectrum_embeddings = (
                spectrum_embeddings.float()
            )

            # [batch, 27485]
            similarities = (
                spectrum_embeddings
                @ molecule_embeddings.T
            )

            rows = torch.arange(
                similarities.shape[0],
                device=device,
            )

            true_scores = similarities[
                rows,
                local_true_tensor,
            ]

            # Rank is 1 + number of candidates
            # with strictly greater score.
            ranks = (
                (
                    similarities
                    > true_scores[:, None]
                )
                .sum(
                    dim=1
                )
                + 1
            )

            all_ranks.append(
                ranks.cpu()
            )

            processed += int(
                similarities.shape[0]
            )

            if (
                batch_number % 25
                == 0
            ):

                elapsed = (
                    time.perf_counter()
                    - eval_start
                )

                rate = (
                    processed
                    / elapsed
                )

                print(
                    f"Processed "
                    f"{processed:,}/"
                    f"{len(dataset):,} "
                    f"| "
                    f"{rate:,.0f} spectra/s"
                )

    ranks = torch.cat(
        all_ranks
    ).numpy()

    # =====================================================
    # Metrics
    # =====================================================

    recall_1 = float(
        np.mean(
            ranks <= 1
        )
    )

    recall_5 = float(
        np.mean(
            ranks <= 5
        )
    )

    recall_10 = float(
        np.mean(
            ranks <= 10
        )
    )

    recall_25 = float(
        np.mean(
            ranks <= 25
        )
    )

    reciprocal_ranks = (
        1.0
        / ranks.astype(
            np.float64
        )
    )

    mrr = float(
        reciprocal_ranks.mean()
    )

    median_rank = float(
        np.median(
            ranks
        )
    )

    mean_rank = float(
        np.mean(
            ranks
        )
    )

    p90_rank = float(
        np.percentile(
            ranks,
            90,
        )
    )

    p95_rank = float(
        np.percentile(
            ranks,
            95,
        )
    )

    total_elapsed = (
        time.perf_counter()
        - start_time
    )

    # =====================================================
    # Summary
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 8 GLOBAL RETRIEVAL RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        f"Validation spectra: "
        f"{len(dataset):,}"
    )

    print(
        f"Candidate molecules: "
        f"{len(unique_validation_molecules):,}"
    )

    print(
        f"\nRecall@1:  "
        f"{recall_1 * 100:.3f}%"
    )

    print(
        f"Recall@5:  "
        f"{recall_5 * 100:.3f}%"
    )

    print(
        f"Recall@10: "
        f"{recall_10 * 100:.3f}%"
    )

    print(
        f"Recall@25: "
        f"{recall_25 * 100:.3f}%"
    )

    print(
        f"MRR:       "
        f"{mrr:.6f}"
    )

    print(
        f"\nMedian rank: "
        f"{median_rank:.1f}"
    )

    print(
        f"Mean rank:   "
        f"{mean_rank:.2f}"
    )

    print(
        f"P90 rank:    "
        f"{p90_rank:.1f}"
    )

    print(
        f"P95 rank:    "
        f"{p95_rank:.1f}"
    )

    print(
        f"\nElapsed: "
        f"{total_elapsed:.2f} s"
    )

    print(
        "=" * 100
    )

    summary_lines = [
        (
            "ENVEDA CASMI 2026 — "
            "STAGE 8 GLOBAL RETRIEVAL"
        ),
        "",
        (
            f"Validation spectra: "
            f"{len(dataset):,}"
        ),
        (
            f"Validation candidate molecules: "
            f"{len(unique_validation_molecules):,}"
        ),
        "",
        (
            f"Recall@1: "
            f"{recall_1:.8f}"
        ),
        (
            f"Recall@5: "
            f"{recall_5:.8f}"
        ),
        (
            f"Recall@10: "
            f"{recall_10:.8f}"
        ),
        (
            f"Recall@25: "
            f"{recall_25:.8f}"
        ),
        (
            f"MRR: "
            f"{mrr:.8f}"
        ),
        "",
        (
            f"Median rank: "
            f"{median_rank:.2f}"
        ),
        (
            f"Mean rank: "
            f"{mean_rank:.2f}"
        ),
        (
            f"P90 rank: "
            f"{p90_rank:.2f}"
        ),
        (
            f"P95 rank: "
            f"{p95_rank:.2f}"
        ),
        "",
        (
            f"Elapsed seconds: "
            f"{total_elapsed:.2f}"
        ),
    ]

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print(
        "\nSummary saved:"
    )

    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()