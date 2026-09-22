from __future__ import annotations

import argparse
import json
import math
import time

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from casmi.models.molecule_encoder import (
    MoleculeEncoder,
)

from casmi.models.spectrum_encoder import (
    SpectrumEncoder,
)

from casmi.retrieval.candidate_mass_index import (
    CandidateMassIndex,
)

from casmi.training.candidate_aware_dataset import (
    CandidateAwareBatchSampler,
    CandidateAwareSpectrumDataset,
)

from casmi.training.cross_modal import (
    multi_positive_cross_modal_loss,
)


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = (
    DATASET_DIR
    / "train.parquet"
)

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

STAGE8_STRUCTURE_MAP = (
    STAGE8_DIR
    / "structure_map.csv"
)

STAGE7_VARIANTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "stage7_candidate_mass_variants.csv"
)

STAGE6_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "stage6"
    / "contrastive_full_best.pt"
)

STAGE8_BASE_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "stage8"
    / "cross_modal_frozen_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "stage8"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_CHECKPOINT = (
    OUTPUT_DIR
    / "candidate_aware_best.pt"
)

LATEST_CHECKPOINT = (
    OUTPUT_DIR
    / "candidate_aware_latest.pt"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage8_candidate_aware_summary.txt"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage8_candidate_aware_history.json"
)


# =========================================================
# Configuration
# =========================================================

SEED = 42

EPOCHS = 2

BATCH_SIZE = 32

MIN_BATCH_SIZE = 8

BATCHES_PER_EPOCH = 3000

VALIDATION_BATCHES = 250

INITIAL_LR = 1e-4

MIN_LR = 1e-5

WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.07

GRADIENT_CLIP_NORM = 5.0

WARMUP_STEPS = 200

LOG_EVERY = 100

EMBEDDING_DIM = 128

TOLERANCE_PPM = 10.0


# =========================================================
# Models
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
# Batch preparation
# =========================================================

def prepare_batch(
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

    peaks = torch.stack(
        [
            mzs,
            intensities,
        ],
        dim=-1,
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

    fingerprints = batch[
        "fingerprint"
    ].to(
        device,
        non_blocking=True,
    )

    molecule_ids = batch[
        "molecule_index"
    ].to(
        device,
        non_blocking=True,
    )

    return (
        peaks,
        mask,
        precursor,
        fingerprints,
        molecule_ids,
    )


# =========================================================
# Learning rate
# =========================================================

def get_learning_rate(
    step: int,
    total_steps: int,
):

    if step < WARMUP_STEPS:

        return (
            INITIAL_LR
            * (
                step + 1
            )
            / WARMUP_STEPS
        )

    progress = (
        step
        - WARMUP_STEPS
    ) / max(
        1,
        total_steps
        - WARMUP_STEPS,
    )

    progress = float(
        np.clip(
            progress,
            0.0,
            1.0,
        )
    )

    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )

    return (
        MIN_LR
        + (
            INITIAL_LR
            - MIN_LR
        )
        * cosine
    )


def set_learning_rate(
    optimizer,
    learning_rate,
):

    for group in optimizer.param_groups:

        group[
            "lr"
        ] = learning_rate


# =========================================================
# Validation
# =========================================================

@torch.inference_mode()
def evaluate(
    spectrum_encoder,
    molecule_encoder,
    loader,
    device,
):

    spectrum_encoder.eval()

    molecule_encoder.eval()

    total_loss = 0.0

    total_s2m = 0.0

    total_m2s = 0.0

    total_samples = 0

    batch_sizes = []

    for batch in loader:

        (
            peaks,
            mask,
            precursor,
            fingerprints,
            molecule_ids,
        ) = prepare_batch(
            batch,
            device,
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

            molecule_embeddings = (
                molecule_encoder(
                    fingerprints
                )
            )

        loss, metrics = (
            multi_positive_cross_modal_loss(
                spectrum_embeddings,
                molecule_embeddings,
                molecule_ids,
                temperature=TEMPERATURE,
            )
        )

        batch_size = int(
            peaks.shape[0]
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_s2m += (
            metrics[
                "s2m_top1"
            ].item()
            * batch_size
        )

        total_m2s += (
            metrics[
                "m2s_top1"
            ].item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        batch_sizes.append(
            batch_size
        )

    if total_samples == 0:

        raise RuntimeError(
            "Validation produced zero samples."
        )

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "s2m_top1": (
            total_s2m
            / total_samples
        ),
        "m2s_top1": (
            total_m2s
            / total_samples
        ),
        "samples": int(
            total_samples
        ),
        "mean_batch_size": float(
            np.mean(
                batch_sizes
            )
        ),
        "min_batch_size": int(
            np.min(
                batch_sizes
            )
        ),
        "max_batch_size": int(
            np.max(
                batch_sizes
            )
        ),
    }


# =========================================================
# Checkpoint
# =========================================================

def save_checkpoint(
    path,
    molecule_encoder,
    optimizer,
    epoch,
    global_step,
    best_validation_loss,
    validation_metrics,
    config,
):

    torch.save(
        {
            "training_type": (
                "candidate_aware_frozen_spectrum"
            ),
            "spectrum_checkpoint": str(
                STAGE6_CHECKPOINT
            ),
            "base_molecule_checkpoint": str(
                STAGE8_BASE_CHECKPOINT
            ),
            "spectrum_encoder_frozen": True,
            "molecule_encoder_state_dict": (
                molecule_encoder.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "epoch": int(
                epoch
            ),
            "global_step": int(
                global_step
            ),
            "best_validation_loss": float(
                best_validation_loss
            ),
            "validation_metrics": (
                validation_metrics
            ),
            "config": (
                config
            ),
        },
        path,
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a short candidate-aware "
            "training diagnostic."
        ),
    )

    args = parser.parse_args()

    torch.manual_seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    # =====================================================
    # Smoke/full configuration
    # =====================================================

    if args.smoke:

        epochs = 1

        batches_per_epoch = 300

        validation_batches = 50

        print(
            "\nSMOKE MODE ENABLED"
        )

    else:

        epochs = EPOCHS

        batches_per_epoch = (
            BATCHES_PER_EPOCH
        )

        validation_batches = (
            VALIDATION_BATCHES
        )

    total_steps = (
        epochs
        * batches_per_epoch
    )

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 8 CANDIDATE-AWARE TRAINING"
    )

    print(
        f"\nGPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Epochs: "
        f"{epochs}"
    )

    print(
        f"Batches/epoch: "
        f"{batches_per_epoch:,}"
    )

    print(
        f"Target batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Total steps: "
        f"{total_steps:,}"
    )

    # =====================================================
    # Shared training metadata
    # =====================================================

    print(
        "\nLoading training metadata..."
    )

    train_metadata = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "precursor_mz",
            "adduct",
            "instrument_type",
            "ingest_lib",
            "inchikey14",
        ],
    )

    print(
        f"Original train rows: "
        f"{len(train_metadata):,}"
    )

    # =====================================================
    # Stage 8 structure map
    # =====================================================

    print(
        "\nLoading structure map..."
    )

    structure_map = pd.read_csv(
        STAGE8_STRUCTURE_MAP
    )

    print(
        f"Structures: "
        f"{len(structure_map):,}"
    )

    # =====================================================
    # Stage 7 mass index
    # =====================================================

    print(
        "\nLoading Stage 7 mass index..."
    )

    variants = pd.read_csv(
        STAGE7_VARIANTS_PATH
    )

    mass_index = CandidateMassIndex(
        variants[
            [
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
                "exact_mass",
            ]
        ]
    )

    print(
        f"Indexed structures: "
        f"{mass_index.unique_structure_count:,}"
    )

    print(
        f"Mass variants: "
        f"{mass_index.variant_count:,}"
    )

    # =====================================================
    # Candidate-aware datasets
    # =====================================================

    print(
        "\nBuilding candidate-aware "
        "training dataset..."
    )

    train_dataset = (
        CandidateAwareSpectrumDataset(
            stage6_dir=STAGE6_DIR,
            stage8_dir=STAGE8_DIR,
            train_metadata=train_metadata,
            mass_index=mass_index,
            structure_map=structure_map,
            split_value=0,
            tolerance_ppm=TOLERANCE_PPM,
        )
    )

    print(
        "\nBuilding candidate-aware "
        "validation dataset..."
    )

    validation_dataset = (
        CandidateAwareSpectrumDataset(
            stage6_dir=STAGE6_DIR,
            stage8_dir=STAGE8_DIR,
            train_metadata=train_metadata,
            mass_index=mass_index,
            structure_map=structure_map,
            split_value=1,
            tolerance_ppm=TOLERANCE_PPM,
        )
    )

    # =====================================================
    # Samplers
    # =====================================================

    train_sampler = (
        CandidateAwareBatchSampler(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            batches_per_epoch=(
                batches_per_epoch
            ),
            min_batch_size=(
                MIN_BATCH_SIZE
            ),
            seed=SEED,
        )
    )

    validation_sampler = (
        CandidateAwareBatchSampler(
            dataset=validation_dataset,
            batch_size=BATCH_SIZE,
            batches_per_epoch=(
                validation_batches
            ),
            min_batch_size=(
                MIN_BATCH_SIZE
            ),
            seed=SEED + 1000,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        num_workers=0,
        pin_memory=True,
    )

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

    result = (
        spectrum_encoder.load_state_dict(
            stage6_checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )
    )

    print(
        f"Missing keys: "
        f"{len(result.missing_keys)}"
    )

    print(
        f"Unexpected keys: "
        f"{len(result.unexpected_keys)}"
    )

    spectrum_encoder = (
        spectrum_encoder
        .to(
            device
        )
        .eval()
    )

    for parameter in (
        spectrum_encoder.parameters()
    ):
        parameter.requires_grad = False

    print(
        "\nLoading Stage 8 pretrained "
        "molecule encoder..."
    )

    molecule_encoder = (
        build_molecule_encoder()
    )

    stage8_checkpoint = torch.load(
        STAGE8_BASE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    result = (
        molecule_encoder.load_state_dict(
            stage8_checkpoint[
                "molecule_encoder_state_dict"
            ],
            strict=True,
        )
    )

    print(
        f"Missing keys: "
        f"{len(result.missing_keys)}"
    )

    print(
        f"Unexpected keys: "
        f"{len(result.unexpected_keys)}"
    )

    molecule_encoder = (
        molecule_encoder.to(
            device
        )
    )

    print(
        f"\nSpectrum parameters: "
        f"{sum(p.numel() for p in spectrum_encoder.parameters()):,}"
    )

    print(
        f"Molecule parameters: "
        f"{sum(p.numel() for p in molecule_encoder.parameters()):,}"
    )

    print(
        f"Trainable parameters: "
        f"{sum(p.numel() for p in molecule_encoder.parameters() if p.requires_grad):,}"
    )

    # =====================================================
    # Optimizer
    # =====================================================

    optimizer = torch.optim.AdamW(
        molecule_encoder.parameters(),
        lr=INITIAL_LR,
        weight_decay=WEIGHT_DECAY,
    )

    config = {
        "seed": SEED,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "min_batch_size": MIN_BATCH_SIZE,
        "batches_per_epoch": (
            batches_per_epoch
        ),
        "validation_batches": (
            validation_batches
        ),
        "total_steps": (
            total_steps
        ),
        "initial_lr": (
            INITIAL_LR
        ),
        "min_lr": (
            MIN_LR
        ),
        "weight_decay": (
            WEIGHT_DECAY
        ),
        "temperature": (
            TEMPERATURE
        ),
        "gradient_clip_norm": (
            GRADIENT_CLIP_NORM
        ),
        "warmup_steps": (
            WARMUP_STEPS
        ),
        "tolerance_ppm": (
            TOLERANCE_PPM
        ),
        "spectrum_encoder_frozen": (
            True
        ),
        "smoke": bool(
            args.smoke
        ),
    }

    # =====================================================
    # Initial candidate-aware validation
    # =====================================================

    print(
        "\nRunning initial candidate-aware "
        "validation..."
    )

    validation_sampler.set_epoch(
        0
    )

    initial_validation = evaluate(
        spectrum_encoder,
        molecule_encoder,
        validation_loader,
        device,
    )

    print(
        f"Initial validation loss: "
        f"{initial_validation['loss']:.4f}"
    )

    print(
        f"Initial S→M Top-1: "
        f"{initial_validation['s2m_top1'] * 100:.3f}%"
    )

    print(
        f"Initial M→S Top-1: "
        f"{initial_validation['m2s_top1'] * 100:.3f}%"
    )

    print(
        f"Validation batch size: "
        f"mean "
        f"{initial_validation['mean_batch_size']:.2f}, "
        f"range "
        f"{initial_validation['min_batch_size']}–"
        f"{initial_validation['max_batch_size']}"
    )

    # =====================================================
    # Training
    # =====================================================

    history = []

    best_validation_loss = float(
        "inf"
    )

    global_step = 0

    training_start = (
        time.perf_counter()
    )

    for epoch in range(
        epochs
    ):

        train_sampler.set_epoch(
            epoch
        )

        molecule_encoder.train()

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"EPOCH "
            f"{epoch + 1}/"
            f"{epochs}"
        )

        print(
            "=" * 100
        )

        epoch_loss_sum = 0.0

        epoch_s2m_sum = 0.0

        epoch_m2s_sum = 0.0

        epoch_samples = 0

        batch_sizes = []

        epoch_start = (
            time.perf_counter()
        )

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):

            learning_rate = (
                get_learning_rate(
                    global_step,
                    total_steps,
                )
            )

            set_learning_rate(
                optimizer,
                learning_rate,
            )

            (
                peaks,
                mask,
                precursor,
                fingerprints,
                molecule_ids,
            ) = prepare_batch(
                batch,
                device,
            )

            # ---------------------------------------------
            # Candidate-aware sampler should create one
            # sample per molecular structure.
            # ---------------------------------------------

            if (
                molecule_ids.unique().numel()
                != molecule_ids.numel()
            ):

                raise RuntimeError(
                    "Candidate-aware batch contains "
                    "duplicate molecule IDs."
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ---------------------------------------------
            # Frozen spectrum encoder
            # ---------------------------------------------

            with torch.no_grad():

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

            # ---------------------------------------------
            # Fine-tune molecule encoder against
            # mass-compatible hard negatives.
            # ---------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                molecule_embeddings = (
                    molecule_encoder(
                        fingerprints
                    )
                )

            loss, metrics = (
                multi_positive_cross_modal_loss(
                    spectrum_embeddings,
                    molecule_embeddings,
                    molecule_ids,
                    temperature=TEMPERATURE,
                )
            )

            loss.backward()

            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    molecule_encoder.parameters(),
                    GRADIENT_CLIP_NORM,
                )
            )

            optimizer.step()

            batch_size = int(
                peaks.shape[0]
            )

            epoch_loss_sum += (
                loss.item()
                * batch_size
            )

            epoch_s2m_sum += (
                metrics[
                    "s2m_top1"
                ].item()
                * batch_size
            )

            epoch_m2s_sum += (
                metrics[
                    "m2s_top1"
                ].item()
                * batch_size
            )

            epoch_samples += (
                batch_size
            )

            batch_sizes.append(
                batch_size
            )

            global_step += 1

            if (
                batch_index
                % LOG_EVERY
                == 0
            ):

                elapsed = (
                    time.perf_counter()
                    - epoch_start
                )

                throughput = (
                    epoch_samples
                    / elapsed
                )

                print(
                    f"Epoch "
                    f"{epoch + 1} "
                    f"| batch "
                    f"{batch_index:,}/"
                    f"{batches_per_epoch:,} "
                    f"| step "
                    f"{global_step:,}/"
                    f"{total_steps:,} "
                    f"| loss "
                    f"{loss.item():.4f} "
                    f"| S→M "
                    f"{metrics['s2m_top1'].item() * 100:.2f}% "
                    f"| M→S "
                    f"{metrics['m2s_top1'].item() * 100:.2f}% "
                    f"| batch "
                    f"{batch_size} "
                    f"| lr "
                    f"{learning_rate:.2e} "
                    f"| grad "
                    f"{float(gradient_norm):.3f} "
                    f"| "
                    f"{throughput:,.0f} samples/s"
                )

        torch.cuda.synchronize()

        epoch_elapsed = (
            time.perf_counter()
            - epoch_start
        )

        train_loss = (
            epoch_loss_sum
            / epoch_samples
        )

        train_s2m = (
            epoch_s2m_sum
            / epoch_samples
        )

        train_m2s = (
            epoch_m2s_sum
            / epoch_samples
        )

        print(
            f"\nEpoch {epoch + 1} training:"
        )

        print(
            f"Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"S→M Top-1: "
            f"{train_s2m * 100:.3f}%"
        )

        print(
            f"M→S Top-1: "
            f"{train_m2s * 100:.3f}%"
        )

        print(
            f"Mean batch size: "
            f"{np.mean(batch_sizes):.2f}"
        )

        print(
            f"Samples seen: "
            f"{epoch_samples:,}"
        )

        print(
            f"Elapsed: "
            f"{epoch_elapsed:.2f} s"
        )

        # =================================================
        # Validation uses a fixed deterministic epoch
        # =================================================

        molecule_encoder.eval()

        validation_sampler.set_epoch(
            0
        )

        print(
            "\nRunning candidate-aware validation..."
        )

        validation_metrics = evaluate(
            spectrum_encoder,
            molecule_encoder,
            validation_loader,
            device,
        )

        print(
            f"Validation loss: "
            f"{validation_metrics['loss']:.4f}"
        )

        print(
            f"Validation S→M Top-1: "
            f"{validation_metrics['s2m_top1'] * 100:.3f}%"
        )

        print(
            f"Validation M→S Top-1: "
            f"{validation_metrics['m2s_top1'] * 100:.3f}%"
        )

        epoch_record = {
            "epoch": (
                epoch + 1
            ),
            "global_step": (
                global_step
            ),
            "train_loss": (
                train_loss
            ),
            "train_s2m_top1": (
                train_s2m
            ),
            "train_m2s_top1": (
                train_m2s
            ),
            "train_mean_batch_size": float(
                np.mean(
                    batch_sizes
                )
            ),
            "validation_loss": (
                validation_metrics[
                    "loss"
                ]
            ),
            "validation_s2m_top1": (
                validation_metrics[
                    "s2m_top1"
                ]
            ),
            "validation_m2s_top1": (
                validation_metrics[
                    "m2s_top1"
                ]
            ),
        }

        history.append(
            epoch_record
        )

        # =================================================
        # Latest checkpoint
        # =================================================

        latest_best = min(
            best_validation_loss,
            validation_metrics[
                "loss"
            ],
        )

        save_checkpoint(
            path=LATEST_CHECKPOINT,
            molecule_encoder=(
                molecule_encoder
            ),
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            best_validation_loss=(
                latest_best
            ),
            validation_metrics=(
                validation_metrics
            ),
            config=config,
        )

        # =================================================
        # Best checkpoint
        # =================================================

        if (
            validation_metrics[
                "loss"
            ]
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_metrics[
                    "loss"
                ]
            )

            save_checkpoint(
                path=BEST_CHECKPOINT,
                molecule_encoder=(
                    molecule_encoder
                ),
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                best_validation_loss=(
                    best_validation_loss
                ),
                validation_metrics=(
                    validation_metrics
                ),
                config=config,
            )

            print(
                "\nNew best candidate-aware "
                "checkpoint saved."
            )

        HISTORY_PATH.write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )

    # =====================================================
    # Summary
    # =====================================================

    total_elapsed = (
        time.perf_counter()
        - training_start
    )

    best_record = min(
        history,
        key=lambda row: (
            row[
                "validation_loss"
            ]
        ),
    )

    summary = [
        (
            "ENVEDA CASMI 2026 — "
            "STAGE 8 CANDIDATE-AWARE TRAINING"
        ),
        "",
        (
            f"Smoke mode: "
            f"{args.smoke}"
        ),
        (
            f"Epochs: "
            f"{epochs}"
        ),
        (
            f"Total steps: "
            f"{total_steps}"
        ),
        (
            f"Target batch size: "
            f"{BATCH_SIZE}"
        ),
        (
            f"Tolerance ppm: "
            f"{TOLERANCE_PPM}"
        ),
        "",
        (
            f"Train eligible spectra: "
            f"{len(train_dataset.eligible_positions)}"
        ),
        (
            f"Validation eligible spectra: "
            f"{len(validation_dataset.eligible_positions)}"
        ),
        "",
        (
            f"Initial validation loss: "
            f"{initial_validation['loss']:.8f}"
        ),
        (
            f"Initial validation S->M Top-1: "
            f"{initial_validation['s2m_top1']:.8f}"
        ),
        (
            f"Initial validation M->S Top-1: "
            f"{initial_validation['m2s_top1']:.8f}"
        ),
        "",
        (
            f"Best epoch: "
            f"{best_record['epoch']}"
        ),
        (
            f"Best validation loss: "
            f"{best_record['validation_loss']:.8f}"
        ),
        (
            f"Best validation S->M Top-1: "
            f"{best_record['validation_s2m_top1']:.8f}"
        ),
        (
            f"Best validation M->S Top-1: "
            f"{best_record['validation_m2s_top1']:.8f}"
        ),
        "",
        (
            f"Elapsed seconds: "
            f"{total_elapsed:.2f}"
        ),
        "",
        (
            f"Best checkpoint: "
            f"{BEST_CHECKPOINT}"
        ),
        (
            f"Latest checkpoint: "
            f"{LATEST_CHECKPOINT}"
        ),
    ]

    SUMMARY_PATH.write_text(
        "\n".join(
            summary
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 8 CANDIDATE-AWARE "
        "TRAINING SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Initial validation loss: "
        f"{initial_validation['loss']:.4f}"
    )

    print(
        f"Initial S→M Top-1: "
        f"{initial_validation['s2m_top1'] * 100:.3f}%"
    )

    print(
        f"Initial M→S Top-1: "
        f"{initial_validation['m2s_top1'] * 100:.3f}%"
    )

    print(
        f"\nBest epoch: "
        f"{best_record['epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{best_record['validation_loss']:.4f}"
    )

    print(
        f"Best S→M Top-1: "
        f"{best_record['validation_s2m_top1'] * 100:.3f}%"
    )

    print(
        f"Best M→S Top-1: "
        f"{best_record['validation_m2s_top1'] * 100:.3f}%"
    )

    print(
        f"\nTotal steps: "
        f"{total_steps:,}"
    )

    print(
        f"Elapsed: "
        f"{total_elapsed:.2f} s "
        f"({total_elapsed / 60:.2f} min)"
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print(
        "\n"
        + "=" * 100
    )


if __name__ == "__main__":
    main()