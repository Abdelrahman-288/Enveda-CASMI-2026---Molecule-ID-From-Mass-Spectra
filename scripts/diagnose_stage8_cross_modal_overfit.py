from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from casmi.models.spectrum_encoder import SpectrumEncoder
from casmi.models.molecule_encoder import MoleculeEncoder

from casmi.training.stage8_dataset import (
    SpectrumMoleculeDataset,
)

from casmi.training.cross_modal import (
    multi_positive_cross_modal_loss,
)


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


SEED = 42

BATCH_SIZE = 256

TRAIN_STEPS = 500

LEARNING_RATE = 1e-3

TEMPERATURE = 0.07

LOG_EVERY = 25


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
        dropout=0.0,
    )


def prepare_batch(
    batch,
    device,
):

    mzs = batch[
        "mzs"
    ].to(
        device
    )

    intensities = batch[
        "intensities"
    ].to(
        device
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
        device
    )

    precursor = batch[
        "precursor_mz"
    ].to(
        device
    )

    fingerprints = batch[
        "fingerprint"
    ].to(
        device
    )

    molecule_ids = batch[
        "molecule_index"
    ].to(
        device
    )

    return (
        peaks,
        mask,
        precursor,
        fingerprints,
        molecule_ids,
    )


def main():

    torch.manual_seed(
        SEED
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 8 TINY-BATCH OVERFIT DIAGNOSTIC"
    )

    print(
        f"\nGPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    # =====================================================
    # Dataset
    # =====================================================

    dataset = SpectrumMoleculeDataset(
        stage6_dir=STAGE6_DIR,
        stage8_dir=STAGE8_DIR,
        split_value=0,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(
            SEED
        ),
    )

    # We intentionally take ONE batch and repeatedly train
    # on exactly that batch.
    batch = next(
        iter(
            loader
        )
    )

    molecule_ids_cpu = batch[
        "molecule_index"
    ]

    unique_count = int(
        molecule_ids_cpu
        .unique()
        .numel()
    )

    print(
        f"\nFixed batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Unique molecules: "
        f"{unique_count}"
    )

    # =====================================================
    # Spectrum encoder
    # =====================================================

    spectrum_encoder = (
        build_spectrum_encoder()
    )

    checkpoint = torch.load(
        STAGE6_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    spectrum_encoder.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
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

    optimizer = torch.optim.AdamW(
        molecule_encoder.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
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

    # =====================================================
    # Compute frozen spectrum targets once
    # =====================================================

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

    spectrum_embeddings = (
        spectrum_embeddings.detach()
    )

    # =====================================================
    # Initial state
    # =====================================================

    molecule_encoder.eval()

    with torch.no_grad():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            molecule_embeddings = (
                molecule_encoder(
                    fingerprints
                )
            )

        initial_loss, initial_metrics = (
            multi_positive_cross_modal_loss(
                spectrum_embeddings,
                molecule_embeddings,
                molecule_ids,
                temperature=TEMPERATURE,
            )
        )

    print(
        "\nInitial:"
    )

    print(
        f"Loss: "
        f"{initial_loss.item():.4f}"
    )

    print(
        f"S→M Top-1: "
        f"{initial_metrics['s2m_top1'].item() * 100:.2f}%"
    )

    print(
        f"M→S Top-1: "
        f"{initial_metrics['m2s_top1'].item() * 100:.2f}%"
    )

    # =====================================================
    # Repeatedly optimize this same batch
    # =====================================================

    molecule_encoder.train()

    for step in range(
        1,
        TRAIN_STEPS + 1,
    ):

        optimizer.zero_grad(
            set_to_none=True
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

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                molecule_encoder.parameters(),
                max_norm=10.0,
            )
        )

        optimizer.step()

        if (
            step == 1
            or step % LOG_EVERY == 0
        ):

            print(
                f"Step "
                f"{step:4d}/"
                f"{TRAIN_STEPS} "
                f"| loss "
                f"{loss.item():.4f} "
                f"| S→M "
                f"{metrics['s2m_top1'].item() * 100:.2f}% "
                f"| M→S "
                f"{metrics['m2s_top1'].item() * 100:.2f}% "
                f"| grad "
                f"{float(grad_norm):.4f}"
            )

    # =====================================================
    # Final deterministic evaluation
    # =====================================================

    molecule_encoder.eval()

    with torch.no_grad():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            molecule_embeddings = (
                molecule_encoder(
                    fingerprints
                )
            )

        final_loss, final_metrics = (
            multi_positive_cross_modal_loss(
                spectrum_embeddings,
                molecule_embeddings,
                molecule_ids,
                temperature=TEMPERATURE,
            )
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TINY-BATCH OVERFIT RESULT"
    )

    print(
        "=" * 100
    )

    print(
        f"Loss: "
        f"{initial_loss.item():.4f}"
        f" -> "
        f"{final_loss.item():.4f}"
    )

    print(
        f"S→M Top-1: "
        f"{initial_metrics['s2m_top1'].item() * 100:.2f}%"
        f" -> "
        f"{final_metrics['s2m_top1'].item() * 100:.2f}%"
    )

    print(
        f"M→S Top-1: "
        f"{initial_metrics['m2s_top1'].item() * 100:.2f}%"
        f" -> "
        f"{final_metrics['m2s_top1'].item() * 100:.2f}%"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()