from __future__ import annotations

from pathlib import Path
import json
import math
import time

import torch
from torch.utils.data import DataLoader

from casmi.models.spectrum_encoder import (
    SpectrumEncoder,
)

from casmi.models.molecule_encoder import (
    MoleculeEncoder,
)

from casmi.training.stage8_dataset import (
    SpectrumMoleculeDataset,
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "stage8"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LATEST_CHECKPOINT = (
    OUTPUT_DIR
    / "cross_modal_frozen_latest.pt"
)

BEST_CHECKPOINT = (
    OUTPUT_DIR
    / "cross_modal_frozen_best.pt"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage8_cross_modal_frozen_summary.txt"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage8_cross_modal_frozen_history.json"
)


# =========================================================
# Configuration
# =========================================================

SEED = 42

EPOCHS = 3

BATCH_SIZE = 512

NUM_WORKERS = 0

INITIAL_LR = 3e-4

MIN_LR = 3e-5

WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.07

GRADIENT_CLIP_NORM = 5.0

WARMUP_STEPS = 250

LOG_EVERY = 100

EMBEDDING_DIM = 128

PRECISION = "BF16"


# =========================================================
# Model builders
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
# Utility
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


def learning_rate_for_step(
    global_step: int,
    total_steps: int,
):

    if (
        global_step
        < WARMUP_STEPS
    ):

        warmup_fraction = (
            global_step
            + 1
        ) / WARMUP_STEPS

        return (
            INITIAL_LR
            * warmup_fraction
        )

    progress = (
        global_step
        - WARMUP_STEPS
    ) / max(
        1,
        total_steps
        - WARMUP_STEPS,
    )

    progress = min(
        max(
            progress,
            0.0,
        ),
        1.0,
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

    for group in (
        optimizer.param_groups
    ):

        group[
            "lr"
        ] = learning_rate


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
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

    total_batches = 0

    total_samples = 0

    start = time.perf_counter()

    for batch in loader:

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

        total_batches += 1

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
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
        "samples": (
            total_samples
        ),
        "batches": (
            total_batches
        ),
        "elapsed_seconds": (
            elapsed
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

    checkpoint = {
        "spectrum_checkpoint": (
            str(
                STAGE6_CHECKPOINT
            )
        ),
        "spectrum_encoder_frozen": (
            True
        ),
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
    }

    torch.save(
        checkpoint,
        path,
    )


# =========================================================
# Main
# =========================================================

def main():

    torch.manual_seed(
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

    gpu_name = (
        torch.cuda.get_device_name(
            0
        )
    )

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 8 FULL FROZEN "
        "CROSS-MODAL TRAINING"
    )

    print(
        f"\nGPU: "
        f"{gpu_name}"
    )

    # =====================================================
    # Datasets
    # =====================================================

    print(
        "\nLoading datasets..."
    )

    train_dataset = (
        SpectrumMoleculeDataset(
            stage6_dir=STAGE6_DIR,
            stage8_dir=STAGE8_DIR,
            split_value=0,
        )
    )

    validation_dataset = (
        SpectrumMoleculeDataset(
            stage6_dir=STAGE6_DIR,
            stage8_dir=STAGE8_DIR,
            split_value=1,
        )
    )

    print(
        f"Training spectra: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation spectra: "
        f"{len(validation_dataset):,}"
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    batches_per_epoch = len(
        train_loader
    )

    total_steps = (
        EPOCHS
        * batches_per_epoch
    )

    print(
        f"Batches/epoch: "
        f"{batches_per_epoch:,}"
    )

    print(
        f"Total steps: "
        f"{total_steps:,}"
    )

    # =====================================================
    # Load Stage 6 encoder
    # =====================================================

    print(
        "\nLoading pretrained Stage 6 "
        "spectrum encoder..."
    )

    spectrum_encoder = (
        build_spectrum_encoder()
    )

    stage6_checkpoint = (
        torch.load(
            STAGE6_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )
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
        spectrum_encoder.to(
            device
        )
    )

    for parameter in (
        spectrum_encoder.parameters()
    ):

        parameter.requires_grad = False

    spectrum_encoder.eval()

    # =====================================================
    # Molecule encoder
    # =====================================================

    molecule_encoder = (
        build_molecule_encoder()
        .to(
            device
        )
    )

    spectrum_parameter_count = sum(
        p.numel()
        for p in spectrum_encoder.parameters()
    )

    molecule_parameter_count = sum(
        p.numel()
        for p in molecule_encoder.parameters()
    )

    print(
        f"\nSpectrum parameters: "
        f"{spectrum_parameter_count:,}"
    )

    print(
        f"Molecule parameters: "
        f"{molecule_parameter_count:,}"
    )

    print(
        f"Trainable parameters: "
        f"{molecule_parameter_count:,}"
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
        "seed": (
            SEED
        ),
        "epochs": (
            EPOCHS
        ),
        "batch_size": (
            BATCH_SIZE
        ),
        "batches_per_epoch": (
            batches_per_epoch
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
        "warmup_steps": (
            WARMUP_STEPS
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
        "embedding_dim": (
            EMBEDDING_DIM
        ),
        "precision": (
            PRECISION
        ),
        "spectrum_encoder_frozen": (
            True
        ),
    }

    # =====================================================
    # Initial validation
    # =====================================================

    print(
        "\nRunning initial validation..."
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

    # =====================================================
    # Training
    # =====================================================

    history = []

    best_validation_loss = (
        float(
            "inf"
        )
    )

    global_step = 0

    full_training_start = (
        time.perf_counter()
    )

    for epoch in range(
        EPOCHS
    ):

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"EPOCH "
            f"{epoch + 1}/"
            f"{EPOCHS}"
        )

        print(
            "=" * 100
        )

        molecule_encoder.train()

        epoch_loss_sum = 0.0

        epoch_s2m_sum = 0.0

        epoch_m2s_sum = 0.0

        epoch_samples = 0

        epoch_start = (
            time.perf_counter()
        )

        interval_loss = 0.0

        interval_samples = 0

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):

            learning_rate = (
                learning_rate_for_step(
                    global_step,
                    total_steps,
                )
            )

            set_learning_rate(
                optimizer,
                learning_rate,
            )

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

            optimizer.zero_grad(
                set_to_none=True
            )

            # -------------------------------------------------
            # Frozen spectrum encoder
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Train molecule encoder
            # -------------------------------------------------

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

            interval_loss += (
                loss.item()
                * batch_size
            )

            interval_samples += (
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

                average_interval_loss = (
                    interval_loss
                    / interval_samples
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
                    f"{average_interval_loss:.4f} "
                    f"| S→M "
                    f"{metrics['s2m_top1'].item() * 100:.2f}% "
                    f"| M→S "
                    f"{metrics['m2s_top1'].item() * 100:.2f}% "
                    f"| lr "
                    f"{learning_rate:.2e} "
                    f"| grad "
                    f"{float(gradient_norm):.3f} "
                    f"| "
                    f"{throughput:,.0f} samples/s"
                )

                interval_loss = 0.0

                interval_samples = 0

        torch.cuda.synchronize()

        epoch_elapsed = (
            time.perf_counter()
            - epoch_start
        )

        epoch_loss = (
            epoch_loss_sum
            / epoch_samples
        )

        epoch_s2m = (
            epoch_s2m_sum
            / epoch_samples
        )

        epoch_m2s = (
            epoch_m2s_sum
            / epoch_samples
        )

        print(
            f"\nEpoch {epoch + 1} "
            f"training complete:"
        )

        print(
            f"Loss: "
            f"{epoch_loss:.4f}"
        )

        print(
            f"S→M Top-1: "
            f"{epoch_s2m * 100:.3f}%"
        )

        print(
            f"M→S Top-1: "
            f"{epoch_m2s * 100:.3f}%"
        )

        print(
            f"Elapsed: "
            f"{epoch_elapsed:.2f} s"
        )

        # =================================================
        # Validation
        # =================================================

        print(
            "\nRunning validation..."
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

        # =================================================
        # History
        # =================================================

        epoch_record = {
            "epoch": (
                epoch + 1
            ),
            "global_step": (
                global_step
            ),
            "train_loss": (
                epoch_loss
            ),
            "train_s2m_top1": (
                epoch_s2m
            ),
            "train_m2s_top1": (
                epoch_m2s
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
            "epoch_seconds": (
                epoch_elapsed
            ),
        }

        history.append(
            epoch_record
        )

        # =================================================
        # Save latest
        # =================================================

        save_checkpoint(
            path=LATEST_CHECKPOINT,
            molecule_encoder=(
                molecule_encoder
            ),
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            best_validation_loss=(
                min(
                    best_validation_loss,
                    validation_metrics[
                        "loss"
                    ],
                )
            ),
            validation_metrics=(
                validation_metrics
            ),
            config=config,
        )

        # =================================================
        # Save best
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
                "\nNew best checkpoint saved."
            )

        # Save history after every epoch.

        HISTORY_PATH.write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )

    # =====================================================
    # Final
    # =====================================================

    torch.cuda.synchronize()

    total_elapsed = (
        time.perf_counter()
        - full_training_start
    )

    best_epoch_record = min(
        history,
        key=lambda x: (
            x[
                "validation_loss"
            ]
        ),
    )

    summary_lines = [
        (
            "ENVEDA CASMI 2026 — "
            "STAGE 8 FULL FROZEN "
            "CROSS-MODAL TRAINING"
        ),
        "",
        (
            f"GPU: "
            f"{gpu_name}"
        ),
        (
            f"Epochs: "
            f"{EPOCHS}"
        ),
        (
            f"Batch size: "
            f"{BATCH_SIZE}"
        ),
        (
            f"Total steps: "
            f"{total_steps:,}"
        ),
        (
            f"Training spectra: "
            f"{len(train_dataset):,}"
        ),
        (
            f"Validation spectra: "
            f"{len(validation_dataset):,}"
        ),
        "",
        (
            f"Initial validation loss: "
            f"{initial_validation['loss']:.6f}"
        ),
        (
            f"Initial S->M Top-1: "
            f"{initial_validation['s2m_top1']:.6f}"
        ),
        (
            f"Initial M->S Top-1: "
            f"{initial_validation['m2s_top1']:.6f}"
        ),
        "",
        (
            f"Best epoch: "
            f"{best_epoch_record['epoch']}"
        ),
        (
            f"Best validation loss: "
            f"{best_epoch_record['validation_loss']:.6f}"
        ),
        (
            f"Best validation S->M Top-1: "
            f"{best_epoch_record['validation_s2m_top1']:.6f}"
        ),
        (
            f"Best validation M->S Top-1: "
            f"{best_epoch_record['validation_m2s_top1']:.6f}"
        ),
        "",
        (
            f"Total elapsed seconds: "
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
            summary_lines
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 8 FULL FROZEN TRAINING SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Total steps: "
        f"{total_steps:,}"
    )

    print(
        f"Total elapsed: "
        f"{total_elapsed:.2f} s "
        f"("
        f"{total_elapsed / 60:.2f} min"
        f")"
    )

    print(
        f"\nInitial validation loss: "
        f"{initial_validation['loss']:.4f}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch_record['epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{best_epoch_record['validation_loss']:.4f}"
    )

    print(
        f"Best S→M Top-1: "
        f"{best_epoch_record['validation_s2m_top1'] * 100:.3f}%"
    )

    print(
        f"Best M→S Top-1: "
        f"{best_epoch_record['validation_m2s_top1'] * 100:.3f}%"
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print(
        "\nLatest checkpoint:"
    )

    print(
        LATEST_CHECKPOINT
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 8 STEP 6A COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()