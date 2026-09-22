from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
    / "train.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage8"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FINGERPRINT_PATH = (
    OUTPUT_DIR
    / "morgan_fingerprints.npy"
)

STRUCTURE_MAP_PATH = (
    OUTPUT_DIR
    / "structure_map.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "fingerprint_metadata.json"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage8_fingerprint_cache_summary.txt"
)


# =========================================================
# Configuration
# =========================================================

RADIUS = 2

FP_SIZE = 2048


# =========================================================
# Main
# =========================================================

def main():

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 8 STEP 1 "
        "MOLECULAR FINGERPRINT CACHE"
    )

    start_time = time.time()

    # =====================================================
    # Load unique structures
    # =====================================================

    print(
        "\nLoading Enveda structures..."
    )

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
        ],
    )

    structures = (
        train[
            [
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
            ]
        ]
        .drop_duplicates(
            subset=[
                "inchikey14",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Unique structures: "
        f"{len(structures):,}"
    )

    # =====================================================
    # Fingerprint generator
    # =====================================================

    generator = (
        rdFingerprintGenerator.GetMorganGenerator(
            radius=RADIUS,
            fpSize=FP_SIZE,
        )
    )

    fingerprints = np.zeros(
        (
            len(structures),
            FP_SIZE,
        ),
        dtype=np.uint8,
    )

    valid = np.zeros(
        len(structures),
        dtype=bool,
    )

    invalid_count = 0

    # =====================================================
    # Generate fingerprints
    # =====================================================

    print(
        "\nGenerating Morgan fingerprints..."
    )

    for index, row in enumerate(
        structures.itertuples(
            index=False
        )
    ):

        mol = Chem.MolFromSmiles(
            row.normalized_smiles
        )

        if mol is None:

            invalid_count += 1
            continue

        fp = generator.GetFingerprintAsNumPy(
            mol
        )

        fingerprints[
            index
        ] = fp.astype(
            np.uint8,
            copy=False,
        )

        valid[
            index
        ] = True

        if (
            index + 1
        ) % 25_000 == 0:

            elapsed = (
                time.time()
                - start_time
            )

            rate = (
                (
                    index + 1
                )
                / elapsed
            )

            print(
                f"Processed "
                f"{index + 1:,}/"
                f"{len(structures):,} "
                f"| "
                f"{rate:,.0f} mol/s"
            )

    # =====================================================
    # Remove invalid rows if any
    # =====================================================

    fingerprints = fingerprints[
        valid
    ]

    structures = (
        structures[
            valid
        ]
        .reset_index(
            drop=True
        )
    )

    # =====================================================
    # Fingerprint statistics
    # =====================================================

    bit_counts = (
        fingerprints.sum(
            axis=1,
            dtype=np.int32,
        )
    )

    zero_fp_count = int(
        np.sum(
            bit_counts == 0
        )
    )

    mean_on_bits = float(
        bit_counts.mean()
    )

    median_on_bits = float(
        np.median(
            bit_counts
        )
    )

    max_on_bits = int(
        bit_counts.max()
    )

    min_on_bits = int(
        bit_counts.min()
    )

    # =====================================================
    # Save cache
    # =====================================================

    print(
        "\nSaving fingerprint cache..."
    )

    np.save(
        FINGERPRINT_PATH,
        fingerprints,
    )

    structures[
        "structure_index"
    ] = np.arange(
        len(structures),
        dtype=np.int32,
    )

    structures = structures[
        [
            "structure_index",
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
        ]
    ]

    structures.to_csv(
        STRUCTURE_MAP_PATH,
        index=False,
    )

    metadata = {
        "fingerprint_type": (
            "Morgan"
        ),
        "radius": RADIUS,
        "fp_size": FP_SIZE,
        "dtype": (
            str(
                fingerprints.dtype
            )
        ),
        "shape": list(
            fingerprints.shape
        ),
        "unique_structures": int(
            len(structures)
        ),
        "invalid_structures": int(
            invalid_count
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =====================================================
    # Summary
    # =====================================================

    elapsed = (
        time.time()
        - start_time
    )

    cache_mb = (
        FINGERPRINT_PATH.stat().st_size
        / 1024**2
    )

    summary_lines = [
        (
            "ENVEDA CASMI 2026 — "
            "STAGE 8 MOLECULAR "
            "FINGERPRINT CACHE"
        ),
        "",
        (
            f"Unique structures: "
            f"{len(structures):,}"
        ),
        (
            f"Invalid structures: "
            f"{invalid_count:,}"
        ),
        (
            f"Fingerprint type: Morgan"
        ),
        (
            f"Radius: {RADIUS}"
        ),
        (
            f"Fingerprint size: "
            f"{FP_SIZE}"
        ),
        (
            f"Array dtype: "
            f"{fingerprints.dtype}"
        ),
        "",
        (
            f"Minimum active bits: "
            f"{min_on_bits}"
        ),
        (
            f"Median active bits: "
            f"{median_on_bits:.1f}"
        ),
        (
            f"Mean active bits: "
            f"{mean_on_bits:.2f}"
        ),
        (
            f"Maximum active bits: "
            f"{max_on_bits}"
        ),
        (
            f"Zero fingerprints: "
            f"{zero_fp_count:,}"
        ),
        "",
        (
            f"Cache size MB: "
            f"{cache_mb:.2f}"
        ),
        (
            f"Elapsed seconds: "
            f"{elapsed:.2f}"
        ),
    ]

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    # =====================================================
    # Console
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 8 FINGERPRINT CACHE SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Structures: "
        f"{len(structures):,}"
    )

    print(
        f"Invalid: "
        f"{invalid_count:,}"
    )

    print(
        f"Fingerprint dimension: "
        f"{FP_SIZE}"
    )

    print(
        f"Active bits "
        f"min/median/mean/max: "
        f"{min_on_bits}/"
        f"{median_on_bits:.1f}/"
        f"{mean_on_bits:.2f}/"
        f"{max_on_bits}"
    )

    print(
        f"Zero fingerprints: "
        f"{zero_fp_count:,}"
    )

    print(
        f"Cache size: "
        f"{cache_mb:.2f} MB"
    )

    print(
        f"Elapsed: "
        f"{elapsed:.2f} s"
    )

    print(
        "\nFingerprint cache:"
    )

    print(
        FINGERPRINT_PATH
    )

    print(
        "\nStructure map:"
    )

    print(
        STRUCTURE_MAP_PATH
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 8 STEP 1 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()