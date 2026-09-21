from pathlib import Path
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from casmi.models.spectrum_dataset import (
    CachedSpectrumDataset,
)

from casmi.models.spectrum_encoder import (
    SpectrumEncoder,
)

from casmi.training.metric_learning import (
    StructureBatchSampler,
    supervised_contrastive_loss,
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "stage6"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LATEST_CHECKPOINT = (
    OUTPUT_DIR
    / "contrastive_full_latest.pt"
)

BEST_CHECKPOINT = (
    OUTPUT_DIR
    / "contrastive_full_best.pt"
)


# =========================================================
# Configuration
# =========================================================

SEED = 42

EPOCHS = 3

STRUCTURES_PER_BATCH = 128
SPECTRA_PER_STRUCTURE = 4

BATCH_SIZE = (
    STRUCTURES_PER_BATCH
    * SPECTRA_PER_STRUCTURE
)

NUM_WORKERS = 0

INITIAL_LR = 3e-4
MIN_LR = 3e-5

WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.07

GRADIENT_CLIP_NORM = 5.0

EMBEDDING_DIM = 128

LOG_EVERY = 100

WARMUP_STEPS = 250


def set_seed(
    seed: int,
) -> None:

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

    return SpectrumEncoder(
        peak_input_dim=2,
        peak_hidden_dim=128,
        precursor_hidden_dim=32,
        pooled_hidden_dim=256,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.10,
    )


def learning_rate_for_step(
    step: int,
    total_steps: int,
) -> float:
    """
    Linear warmup followed by cosine decay.
    """

    if step < WARMUP_STEPS:

        warmup_fraction = (
            step + 1
        ) / WARMUP_STEPS

        return (
            INITIAL_LR
            * warmup_fraction
        )

    decay_steps = (
        total_steps
        - WARMUP_STEPS
    )

    decay_position = (
        step
        - WARMUP_STEPS
    )

    decay_fraction = (
        decay_position
        / max(
            decay_steps,
            1,
        )
    )

    decay_fraction = min(
        max(
            decay_fraction,
            0.0,
        ),
        1.0,
    )

    cosine_factor = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * decay_fraction
            )
        )
    )

    return (
        MIN_LR
        + (
            INITIAL_LR
            - MIN_LR
        )
        * cosine_factor
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    global_step,
    epoch_loss,
    best_epoch_loss,
    config,
):

    checkpoint = {
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "epoch": (
            epoch
        ),
        "global_step": (
            global_step
        ),
        "epoch_loss": (
            epoch_loss
        ),
        "best_epoch_loss": (
            best_epoch_loss
        ),
        "config": (
            config
        ),
    }

    torch.save(
        checkpoint,
        path,
    )


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 6 STEP 9 "
        "FULL CONTRASTIVE TRAINING"
    )

    set_seed(
        SEED
    )

    # =====================================================
    # Device
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
        "Precision: BF16"
    )

    # =====================================================
    # Dataset
    # =====================================================

    dataset = CachedSpectrumDataset(
        CACHE_DIR,
        split="train",
    )

    labels = np.asarray(
        dataset.labels[
            dataset.indices
        ],
        dtype=np.int64,
    )

    unique_labels, label_counts = (
        np.unique(
            labels,
            return_counts=True,
        )
    )

    eligible_structures = int(
        np.sum(
            label_counts >= 2
        )
    )

    batches_per_epoch = math.ceil(
        len(dataset)
        / BATCH_SIZE
    )

    total_steps = (
        batches_per_epoch
        * EPOCHS
    )

    print(
        f"\nTraining spectra: "
        f"{len(dataset):,}"
    )

    print(
        f"Training structures: "
        f"{len(unique_labels):,}"
    )

    print(
        f"Structures with >=2 spectra: "
        f"{eligible_structures:,}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Batches per epoch: "
        f"{batches_per_epoch:,}"
    )

    print(
        f"Epochs: "
        f"{EPOCHS}"
    )

    print(
        f"Total optimization steps: "
        f"{total_steps:,}"
    )

    print(
        f"DataLoader workers: "
        f"{NUM_WORKERS}"
    )

    # =====================================================
    # Model
    # =====================================================

    model = build_model().to(
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nModel parameters: "
        f"{parameter_count:,}"
    )

    # =====================================================
    # Optimizer
    # =====================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=INITIAL_LR,
        weight_decay=WEIGHT_DECAY,
    )

    # =====================================================
    # Config stored in checkpoints
    # =====================================================

    config = {
        "seed": (
            SEED
        ),
        "epochs": (
            EPOCHS
        ),
        "structures_per_batch": (
            STRUCTURES_PER_BATCH
        ),
        "spectra_per_structure": (
            SPECTRA_PER_STRUCTURE
        ),
        "batch_size": (
            BATCH_SIZE
        ),
        "batches_per_epoch": (
            batches_per_epoch
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
            "BF16"
        ),
    }

    # =====================================================
    # Training
    # =====================================================

    global_step = 0

    best_epoch_loss = float(
        "inf"
    )

    total_start_time = time.time()

    torch.cuda.reset_peak_memory_stats()

    for epoch in range(
        EPOCHS
    ):

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"EPOCH "
            f"{epoch + 1}/{EPOCHS}"
        )

        print(
            "=" * 100
        )

        # -------------------------------------------------
        # New deterministic sampling sequence every epoch
        # -------------------------------------------------

        sampler = StructureBatchSampler(
            labels=labels,
            structures_per_batch=(
                STRUCTURES_PER_BATCH
            ),
            spectra_per_structure=(
                SPECTRA_PER_STRUCTURE
            ),
            batches_per_epoch=(
                batches_per_epoch
            ),
            seed=SEED,
        )

        sampler.set_epoch(
            epoch
        )

        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=False,
        )

        model.train()

        epoch_start_time = (
            time.time()
        )

        epoch_loss_sum = 0.0

        logging_loss_sum = 0.0

        logging_batches = 0

        spectra_seen_epoch = 0

        # =================================================
        # Epoch loop
        # =================================================

        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):

            # ---------------------------------------------
            # Learning rate schedule
            # ---------------------------------------------

            current_lr = (
                learning_rate_for_step(
                    global_step,
                    total_steps,
                )
            )

            for parameter_group in (
                optimizer.param_groups
            ):

                parameter_group[
                    "lr"
                ] = current_lr

            # ---------------------------------------------
            # Batch -> GPU
            # ---------------------------------------------

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

            batch_labels = batch[
                "label"
            ].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ---------------------------------------------
            # Forward
            # ---------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                embeddings = model(
                    peaks,
                    mask,
                    precursor,
                )

            # ---------------------------------------------
            # FP32 contrastive loss
            # ---------------------------------------------

            loss = (
                supervised_contrastive_loss(
                    embeddings.float(),
                    batch_labels,
                    temperature=TEMPERATURE,
                )
            )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite loss at "
                    f"epoch {epoch + 1}, "
                    f"batch {batch_index}."
                )

            # ---------------------------------------------
            # Backward
            # ---------------------------------------------

            loss.backward()

            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=(
                        GRADIENT_CLIP_NORM
                    ),
                    error_if_nonfinite=True,
                )
            )

            optimizer.step()

            # ---------------------------------------------
            # Statistics
            # ---------------------------------------------

            loss_value = float(
                loss.item()
            )

            epoch_loss_sum += (
                loss_value
            )

            logging_loss_sum += (
                loss_value
            )

            logging_batches += 1

            spectra_seen_epoch += int(
                peaks.shape[0]
            )

            global_step += 1

            # ---------------------------------------------
            # Logging
            # ---------------------------------------------

            if (
                batch_index == 1
                or batch_index
                % LOG_EVERY
                == 0
            ):

                epoch_elapsed = (
                    time.time()
                    - epoch_start_time
                )

                throughput = (
                    spectra_seen_epoch
                    / epoch_elapsed
                )

                average_recent_loss = (
                    logging_loss_sum
                    / logging_batches
                )

                print(
                    f"Epoch "
                    f"{epoch + 1}/{EPOCHS} "
                    f"| batch "
                    f"{batch_index:4d}/"
                    f"{batches_per_epoch} "
                    f"| loss="
                    f"{loss_value:.5f} "
                    f"| avg="
                    f"{average_recent_loss:.5f} "
                    f"| grad="
                    f"{float(gradient_norm):.4f} "
                    f"| lr="
                    f"{current_lr:.2e} "
                    f"| "
                    f"{throughput:,.0f} spectra/s"
                )

                logging_loss_sum = (
                    0.0
                )

                logging_batches = (
                    0
                )

        # =================================================
        # End epoch
        # =================================================

        torch.cuda.synchronize()

        epoch_elapsed = (
            time.time()
            - epoch_start_time
        )

        epoch_loss = (
            epoch_loss_sum
            / batches_per_epoch
        )

        epoch_throughput = (
            spectra_seen_epoch
            / epoch_elapsed
        )

        print(
            "\n"
            f"Epoch "
            f"{epoch + 1} complete"
        )

        print(
            f"Epoch average loss: "
            f"{epoch_loss:.6f}"
        )

        print(
            f"Epoch elapsed: "
            f"{epoch_elapsed:.2f} s"
        )

        print(
            f"Epoch throughput: "
            f"{epoch_throughput:,.0f} "
            f"spectra/s"
        )

        # =================================================
        # Save latest
        # =================================================

        save_checkpoint(
            path=LATEST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            epoch_loss=epoch_loss,
            best_epoch_loss=(
                min(
                    best_epoch_loss,
                    epoch_loss,
                )
            ),
            config=config,
        )

        # =================================================
        # Save best training-loss checkpoint
        # =================================================

        if epoch_loss < best_epoch_loss:

            best_epoch_loss = (
                epoch_loss
            )

            save_checkpoint(
                path=BEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                epoch_loss=epoch_loss,
                best_epoch_loss=(
                    best_epoch_loss
                ),
                config=config,
            )

            print(
                "New best checkpoint saved."
            )

    # =====================================================
    # Final summary
    # =====================================================

    torch.cuda.synchronize()

    total_elapsed = (
        time.time()
        - total_start_time
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FULL CONTRASTIVE TRAINING COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"Epochs completed: "
        f"{EPOCHS}"
    )

    print(
        f"Optimization steps: "
        f"{global_step:,}"
    )

    print(
        f"Best epoch loss: "
        f"{best_epoch_loss:.6f}"
    )

    print(
        f"Total elapsed: "
        f"{total_elapsed:.2f} s"
    )

    print(
        f"Peak GPU memory allocated: "
        f"{peak_allocated:.2f} GB"
    )

    print(
        f"Peak GPU memory reserved: "
        f"{peak_reserved:.2f} GB"
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
        "STAGE 6 STEP 9 TRAINING COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()