from __future__ import annotations

from pathlib import Path
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

CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "cross_modal_smoke.pt"
)


# =========================================================
# Configuration
# =========================================================

SEED = 42

BATCH_SIZE = 512

NUM_WORKERS = 0

TRAIN_STEPS = 500

VALIDATION_BATCHES = 50

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.07

GRADIENT_CLIP_NORM = 5.0

LOG_EVERY = 25


# =========================================================
# Utilities
# =========================================================

def build_spectrum_encoder():

    return SpectrumEncoder(
        peak_input_dim=2,
        peak_hidden_dim=128,
        precursor_hidden_dim=32,
        pooled_hidden_dim=256,
        embedding_dim=128,
        dropout=0.10,
    )


def build_molecule_encoder():

    return MoleculeEncoder(
        fingerprint_dim=2048,
        hidden_dim_1=1024,
        hidden_dim_2=512,
        embedding_dim=128,
        dropout=0.10,
    )


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

    peaks = torch.stack(
        [
            mzs,
            intensities,
        ],
        dim=-1,
    )

    return peaks


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

    for batch_index, batch in enumerate(
        loader
    ):

        if (
            batch_index
            >= VALIDATION_BATCHES
        ):
            break

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

        total_loss += (
            loss.item()
        )

        total_s2m += (
            metrics[
                "s2m_top1"
            ].item()
        )

        total_m2s += (
            metrics[
                "m2s_top1"
            ].item()
        )

        total_batches += 1

    return {
        "loss": (
            total_loss
            / total_batches
        ),
        "s2m_top1": (
            total_s2m
            / total_batches
        ),
        "m2s_top1": (
            total_m2s
            / total_batches
        ),
    }


# =========================================================
# Main
# =========================================================

def main():

    torch.manual_seed(
        SEED
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required for Stage 8 training."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 8 CROSS-MODAL SMOKE TRAINING"
    )

    print(
        f"\nGPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    # =====================================================
    # Dataset
    # =====================================================

    print(
        "\nLoading paired datasets..."
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
        f"Train spectra: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation spectra: "
        f"{len(validation_dataset):,}"
    )

    train_generator = torch.Generator()

    train_generator.manual_seed(
        SEED
    )

    validation_generator = (
        torch.Generator()
    )

    validation_generator.manual_seed(
        SEED + 1
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
        shuffle=True,
        generator=validation_generator,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # =====================================================
    # Spectrum encoder
    # =====================================================

    print(
        "\nLoading Stage 6 spectrum encoder..."
    )

    spectrum_encoder = (
        build_spectrum_encoder()
    )

    checkpoint = torch.load(
        STAGE6_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    load_result = (
        spectrum_encoder.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )
    )

    print(
        f"Missing keys: "
        f"{len(load_result.missing_keys)}"
    )

    print(
        f"Unexpected keys: "
        f"{len(load_result.unexpected_keys)}"
    )

    spectrum_encoder = (
        spectrum_encoder.to(
            device
        )
    )

    # Freeze Stage 6 for this first experiment.
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

    spectrum_parameters = sum(
        parameter.numel()
        for parameter
        in spectrum_encoder.parameters()
    )

    molecule_parameters = sum(
        parameter.numel()
        for parameter
        in molecule_encoder.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in molecule_encoder.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nSpectrum parameters: "
        f"{spectrum_parameters:,}"
    )

    print(
        f"Molecule parameters: "
        f"{molecule_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # =====================================================
    # Optimizer
    # =====================================================

    optimizer = torch.optim.AdamW(
        molecule_encoder.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # =====================================================
    # Initial validation
    # =====================================================

    print(
        "\nEvaluating before training..."
    )

    initial_metrics = evaluate(
        spectrum_encoder,
        molecule_encoder,
        validation_loader,
        device,
    )

    print(
        f"Initial validation loss: "
        f"{initial_metrics['loss']:.4f}"
    )

    print(
        f"Initial S→M Top-1: "
        f"{initial_metrics['s2m_top1'] * 100:.3f}%"
    )

    print(
        f"Initial M→S Top-1: "
        f"{initial_metrics['m2s_top1'] * 100:.3f}%"
    )

    # =====================================================
    # Training
    # =====================================================

    print(
        "\nStarting smoke training..."
    )

    molecule_encoder.train()

    start_time = time.perf_counter()

    running_loss = 0.0

    train_iterator = iter(
        train_loader
    )

    for step in range(
        1,
        TRAIN_STEPS + 1,
    ):

        try:

            batch = next(
                train_iterator
            )

        except StopIteration:

            train_iterator = iter(
                train_loader
            )

            batch = next(
                train_iterator
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

        # Spectrum encoder is frozen.
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

        torch.nn.utils.clip_grad_norm_(
            molecule_encoder.parameters(),
            GRADIENT_CLIP_NORM,
        )

        optimizer.step()

        running_loss += (
            loss.item()
        )

        if (
            step % LOG_EVERY
            == 0
        ):

            average_loss = (
                running_loss
                / LOG_EVERY
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            samples_seen = (
                step
                * BATCH_SIZE
            )

            throughput = (
                samples_seen
                / elapsed
            )

            print(
                f"Step "
                f"{step:4d}/"
                f"{TRAIN_STEPS} "
                f"| loss "
                f"{average_loss:.4f} "
                f"| S→M "
                f"{metrics['s2m_top1'].item() * 100:.2f}% "
                f"| M→S "
                f"{metrics['m2s_top1'].item() * 100:.2f}% "
                f"| "
                f"{throughput:,.0f} samples/s"
            )

            running_loss = 0.0

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # =====================================================
    # Final validation
    # =====================================================

    print(
        "\nEvaluating after training..."
    )

    final_metrics = evaluate(
        spectrum_encoder,
        molecule_encoder,
        validation_loader,
        device,
    )

    print(
        f"Final validation loss: "
        f"{final_metrics['loss']:.4f}"
    )

    print(
        f"Final S→M Top-1: "
        f"{final_metrics['s2m_top1'] * 100:.3f}%"
    )

    print(
        f"Final M→S Top-1: "
        f"{final_metrics['m2s_top1'] * 100:.3f}%"
    )

    # =====================================================
    # Save
    # =====================================================

    output_checkpoint = {
        "spectrum_checkpoint": str(
            STAGE6_CHECKPOINT
        ),
        "spectrum_encoder_frozen": True,
        "molecule_encoder_state_dict": (
            molecule_encoder.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "steps": TRAIN_STEPS,
        "config": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "temperature": TEMPERATURE,
            "embedding_dim": 128,
            "precision": "BF16",
        },
        "initial_validation": (
            initial_metrics
        ),
        "final_validation": (
            final_metrics
        ),
    }

    torch.save(
        output_checkpoint,
        CHECKPOINT_PATH,
    )

    # =====================================================
    # Final summary
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 8 CROSS-MODAL SMOKE SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Training steps: "
        f"{TRAIN_STEPS:,}"
    )

    print(
        f"Training spectra seen: "
        f"{TRAIN_STEPS * BATCH_SIZE:,}"
    )

    print(
        f"Elapsed: "
        f"{elapsed:.2f} s"
    )

    print(
        f"Throughput: "
        f"{TRAIN_STEPS * BATCH_SIZE / elapsed:,.0f} "
        f"samples/s"
    )

    print(
        "\nValidation:"
    )

    print(
        f"Loss: "
        f"{initial_metrics['loss']:.4f}"
        f" -> "
        f"{final_metrics['loss']:.4f}"
    )

    print(
        f"S→M Top-1: "
        f"{initial_metrics['s2m_top1'] * 100:.3f}%"
        f" -> "
        f"{final_metrics['s2m_top1'] * 100:.3f}%"
    )

    print(
        f"M→S Top-1: "
        f"{initial_metrics['m2s_top1'] * 100:.3f}%"
        f" -> "
        f"{final_metrics['m2s_top1'] * 100:.3f}%"
    )

    print(
        f"\nCheckpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print(
        "\n"
        + "=" * 100
    )


if __name__ == "__main__":
    main()