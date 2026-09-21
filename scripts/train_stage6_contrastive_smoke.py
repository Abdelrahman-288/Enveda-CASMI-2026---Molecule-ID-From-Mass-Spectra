from pathlib import Path
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

CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "contrastive_smoke.pt"
)


# =========================================================
# Configuration
# =========================================================

SEED = 42

STRUCTURES_PER_BATCH = 128
SPECTRA_PER_STRUCTURE = 4

BATCH_SIZE = (
    STRUCTURES_PER_BATCH
    * SPECTRA_PER_STRUCTURE
)

# Windows-safe for the current memmap + custom sampler.
NUM_WORKERS = 0

NUM_STEPS = 500

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.07

LOG_EVERY = 25

EMBEDDING_DIM = 128

GRADIENT_CLIP_NORM = 5.0


def set_seed(seed: int) -> None:

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 6 STEP 6 "
        "BF16 CONTRASTIVE TRAINING SMOKE RUN"
    )

    set_seed(SEED)

    # =====================================================
    # CUDA
    # =====================================================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    bf16_supported = (
        torch.cuda.is_bf16_supported()
    )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "CUDA:",
        torch.version.cuda,
    )

    print(
        "BF16 supported:",
        bf16_supported,
    )

    if bf16_supported:
        amp_dtype = torch.bfloat16
        precision_name = "BF16"
    else:
        amp_dtype = None
        precision_name = "FP32 fallback"

    print(
        "Training precision:",
        precision_name,
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

    print(
        f"\nTraining spectra: "
        f"{len(dataset):,}"
    )

    print(
        f"Structures per batch: "
        f"{STRUCTURES_PER_BATCH}"
    )

    print(
        f"Spectra per structure: "
        f"{SPECTRA_PER_STRUCTURE}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Training steps: "
        f"{NUM_STEPS}"
    )

    print(
        f"DataLoader workers: "
        f"{NUM_WORKERS}"
    )

    # =====================================================
    # Structure-aware sampler
    # =====================================================

    sampler = StructureBatchSampler(
        labels=labels,
        structures_per_batch=(
            STRUCTURES_PER_BATCH
        ),
        spectra_per_structure=(
            SPECTRA_PER_STRUCTURE
        ),
        batches_per_epoch=(
            NUM_STEPS
        ),
        seed=SEED,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
    )

    # =====================================================
    # Model
    # =====================================================

    model = SpectrumEncoder(
        peak_input_dim=2,
        peak_hidden_dim=128,
        precursor_hidden_dim=32,
        pooled_hidden_dim=256,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.10,
    ).to(device)

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
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # =====================================================
    # Training state
    # =====================================================

    model.train()

    first_loss = None
    last_loss = None

    running_loss = 0.0
    total_spectra_seen = 0

    start_time = time.time()

    torch.cuda.reset_peak_memory_stats()

    print(
        "\nStarting contrastive training..."
    )

    # =====================================================
    # Training
    # =====================================================

    for step, batch in enumerate(
        loader,
        start=1,
    ):

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

        labels_batch = batch[
            "label"
        ].to(
            device,
            non_blocking=True,
        )

        batch_size = int(
            peaks.shape[0]
        )

        total_spectra_seen += batch_size

        optimizer.zero_grad(
            set_to_none=True
        )

        # =================================================
        # Forward
        # =================================================

        if bf16_supported:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                embeddings = model(
                    peaks,
                    mask,
                    precursor,
                )

        else:

            embeddings = model(
                peaks,
                mask,
                precursor,
            )

        # =================================================
        # Contrastive loss in FP32
        # =================================================

        loss = supervised_contrastive_loss(
            embeddings.float(),
            labels_batch,
            temperature=TEMPERATURE,
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss at "
                f"step {step}: "
                f"{loss.item()}"
            )

        # =================================================
        # Backward
        # =================================================

        loss.backward()

        # Check gradients before optimizer update.
        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP_NORM,
                error_if_nonfinite=True,
            )
        )

        optimizer.step()

        # =================================================
        # Stats
        # =================================================

        loss_value = float(
            loss.item()
        )

        if first_loss is None:
            first_loss = loss_value

        last_loss = loss_value

        running_loss += loss_value

        # =================================================
        # Logging
        # =================================================

        if (
            step == 1
            or step % LOG_EVERY == 0
        ):

            elapsed = (
                time.time()
                - start_time
            )

            if step == 1:
                average_loss = (
                    running_loss
                )
            else:
                average_loss = (
                    running_loss
                    / LOG_EVERY
                )

            throughput = (
                total_spectra_seen
                / elapsed
            )

            print(
                f"Step "
                f"{step:4d}/{NUM_STEPS} "
                f"| loss="
                f"{loss_value:.5f} "
                f"| avg="
                f"{average_loss:.5f} "
                f"| grad="
                f"{float(gradient_norm):.4f} "
                f"| lr="
                f"{LEARNING_RATE:.2e} "
                f"| "
                f"{throughput:,.0f} spectra/s"
            )

            running_loss = 0.0

        if step >= NUM_STEPS:
            break

    # =====================================================
    # Final timing
    # =====================================================

    torch.cuda.synchronize()

    elapsed = (
        time.time()
        - start_time
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    if first_loss is None:
        raise RuntimeError(
            "No training loss was produced."
        )

    if last_loss is None:
        raise RuntimeError(
            "No final training loss was produced."
        )

    loss_change = (
        last_loss
        - first_loss
    )

    average_throughput = (
        total_spectra_seen
        / elapsed
    )

    # =====================================================
    # Save checkpoint
    # =====================================================

    checkpoint = {
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "config": {
            "embedding_dim": (
                EMBEDDING_DIM
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
            "num_workers": (
                NUM_WORKERS
            ),
            "temperature": (
                TEMPERATURE
            ),
            "learning_rate": (
                LEARNING_RATE
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "gradient_clip_norm": (
                GRADIENT_CLIP_NORM
            ),
            "precision": (
                precision_name
            ),
            "steps": (
                NUM_STEPS
            ),
            "seed": (
                SEED
            ),
        },
        "first_loss": (
            first_loss
        ),
        "last_loss": (
            last_loss
        ),
        "loss_change": (
            loss_change
        ),
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH,
    )

    # =====================================================
    # Summary
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CONTRASTIVE SMOKE TRAINING SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Precision: "
        f"{precision_name}"
    )

    print(
        f"Steps completed: "
        f"{NUM_STEPS}"
    )

    print(
        f"Spectra processed: "
        f"{total_spectra_seen:,}"
    )

    print(
        f"First loss: "
        f"{first_loss:.6f}"
    )

    print(
        f"Last loss: "
        f"{last_loss:.6f}"
    )

    print(
        f"Loss change: "
        f"{loss_change:+.6f}"
    )

    print(
        f"Elapsed: "
        f"{elapsed:.2f} s"
    )

    print(
        f"Average throughput: "
        f"{average_throughput:,.0f} spectra/s"
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
        "\nCheckpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 6 STEP 6 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()