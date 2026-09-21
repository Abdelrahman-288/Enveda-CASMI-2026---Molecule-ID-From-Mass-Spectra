from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from casmi.models.spectrum_dataset import (
    CachedSpectrumDataset,
)

from casmi.models.spectrum_encoder import (
    SpectrumEncoder,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage6"
)


BATCH_SIZE = 512
NUM_WORKERS = 4
NUM_BATCHES = 200

EMBEDDING_DIM = 128


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 6 STEP 4 "
        "GPU SPECTRUM ENCODER BENCHMARK"
    )

    # =====================================================
    # Device
    # =====================================================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
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
        "CUDA version:",
        torch.version.cuda,
    )

    # =====================================================
    # Dataset
    # =====================================================

    dataset = CachedSpectrumDataset(
        CACHE_DIR,
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        drop_last=True,
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
    )

    model = model.to(
        device
    )

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nParameters: "
        f"{parameter_count:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    # =====================================================
    # Warmup
    # =====================================================

    print(
        "\nRunning GPU warmup..."
    )

    iterator = iter(
        loader
    )

    with torch.inference_mode():

        for _ in range(10):

            batch = next(
                iterator
            )

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

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                embeddings = model(
                    peaks,
                    mask,
                    precursor,
                )

    torch.cuda.synchronize()

    # =====================================================
    # Benchmark
    # =====================================================

    print(
        "Running encoder benchmark..."
    )

    start = time.time()

    spectra_seen = 0

    first_shape = None

    with torch.inference_mode():

        for batch_number, batch in enumerate(
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

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                embeddings = model(
                    peaks,
                    mask,
                    precursor,
                )

            if first_shape is None:
                first_shape = (
                    embeddings.shape
                )

            spectra_seen += (
                peaks.shape[0]
            )

            if (
                batch_number
                >= NUM_BATCHES
            ):
                break

    torch.cuda.synchronize()

    elapsed = (
        time.time()
        - start
    )

    throughput = (
        spectra_seen
        / elapsed
    )

    # =====================================================
    # GPU memory
    # =====================================================

    allocated_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    reserved_gb = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    # =====================================================
    # Results
    # =====================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "GPU ENCODER RESULTS"
    )

    print(
        "=" * 90
    )

    print(
        f"Embedding shape: "
        f"{first_shape}"
    )

    print(
        f"Batches: "
        f"{NUM_BATCHES:,}"
    )

    print(
        f"Spectra encoded: "
        f"{spectra_seen:,}"
    )

    print(
        f"Elapsed: "
        f"{elapsed:.2f} s"
    )

    print(
        f"End-to-end throughput: "
        f"{throughput:,.0f} spectra/s"
    )

    print(
        f"Maximum GPU memory allocated: "
        f"{allocated_gb:.2f} GB"
    )

    print(
        f"Maximum GPU memory reserved: "
        f"{reserved_gb:.2f} GB"
    )

    # =====================================================
    # Sanity check
    # =====================================================

    embedding_norms = (
        torch.linalg.vector_norm(
            embeddings.float(),
            dim=1,
        )
    )

    print(
        f"Embedding norm mean: "
        f"{embedding_norms.mean().item():.6f}"
    )

    print(
        f"Embedding norm min: "
        f"{embedding_norms.min().item():.6f}"
    )

    print(
        f"Embedding norm max: "
        f"{embedding_norms.max().item():.6f}"
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "STAGE 6 STEP 4 COMPLETE"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()