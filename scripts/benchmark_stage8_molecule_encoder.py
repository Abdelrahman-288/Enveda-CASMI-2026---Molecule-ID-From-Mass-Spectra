from __future__ import annotations

import time

import torch

from casmi.models.molecule_encoder import (
    MoleculeEncoder,
)


BATCH_SIZE = 1024

WARMUP_STEPS = 20

BENCHMARK_STEPS = 200


def main():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 8 MOLECULE ENCODER BENCHMARK"
    )

    print(
        f"\nGPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    model = MoleculeEncoder().to(
        device
    )

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    fingerprints = torch.randint(
        low=0,
        high=2,
        size=(
            BATCH_SIZE,
            2048,
        ),
        dtype=torch.float32,
        device=device,
    )

    # =====================================================
    # Warmup
    # =====================================================

    print(
        "\nWarming up..."
    )

    with torch.inference_mode():

        for _ in range(
            WARMUP_STEPS
        ):

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                _ = model(
                    fingerprints
                )

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    # =====================================================
    # Benchmark
    # =====================================================

    print(
        "Benchmarking..."
    )

    start = time.perf_counter()

    with torch.inference_mode():

        for _ in range(
            BENCHMARK_STEPS
        ):

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                embeddings = model(
                    fingerprints
                )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    total_molecules = (
        BATCH_SIZE
        * BENCHMARK_STEPS
    )

    throughput = (
        total_molecules
        / elapsed
    )

    allocated_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    reserved_gb = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    norms = (
        torch.linalg.vector_norm(
            embeddings.float(),
            dim=1,
        )
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "MOLECULE ENCODER BENCHMARK"
    )

    print(
        "=" * 90
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE:,}"
    )

    print(
        f"Benchmark steps: "
        f"{BENCHMARK_STEPS:,}"
    )

    print(
        f"Molecules encoded: "
        f"{total_molecules:,}"
    )

    print(
        f"Elapsed: "
        f"{elapsed:.3f} s"
    )

    print(
        f"Throughput: "
        f"{throughput:,.0f} molecules/s"
    )

    print(
        f"Peak allocated GPU memory: "
        f"{allocated_gb:.3f} GB"
    )

    print(
        f"Peak reserved GPU memory: "
        f"{reserved_gb:.3f} GB"
    )

    print(
        f"Embedding norm mean: "
        f"{norms.mean().item():.6f}"
    )

    print(
        f"Embedding norm min: "
        f"{norms.min().item():.6f}"
    )

    print(
        f"Embedding norm max: "
        f"{norms.max().item():.6f}"
    )

    print(
        "\n"
        + "=" * 90
    )


if __name__ == "__main__":
    main()