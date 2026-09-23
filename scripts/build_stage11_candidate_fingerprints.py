from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGE11_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage11"
)

STRUCTURES_PATH = (
    STAGE11_DIR
    / "test_candidate_structures.parquet"
)

FINGERPRINTS_PATH = (
    STAGE11_DIR
    / "test_candidate_morgan_fingerprints.npy"
)

FINGERPRINT_MAP_PATH = (
    STAGE11_DIR
    / "test_candidate_fingerprint_map.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage11_candidate_fingerprint_summary.txt"
)


# ============================================================
# Configuration
# ============================================================

MORGAN_RADIUS = 2
FINGERPRINT_BITS = 2048


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 100)
    print(
        "ENVEDA CASMI 2026 - STAGE 11.2 "
        "TEST CANDIDATE MORGAN FINGERPRINTS"
    )
    print("=" * 100)

    if not STRUCTURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing Stage 11 structure manifest:\n"
            f"{STRUCTURES_PATH}"
        )

    STAGE11_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load structures
    # --------------------------------------------------------

    print(
        "\nLoading Stage 11 candidate structures..."
    )

    structures = pd.read_parquet(
        STRUCTURES_PATH
    )

    required_columns = [
        "candidate_structure_index",
        "candidate_inchikey14",
        "candidate_smiles",
    ]

    missing = [
        column
        for column in required_columns
        if column not in structures.columns
    ]

    if missing:
        raise ValueError(
            "Structure manifest is missing columns: "
            f"{missing}"
        )

    structures = (
        structures
        .sort_values(
            "candidate_structure_index",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    expected_indices = np.arange(
        len(structures),
        dtype=np.int64,
    )

    observed_indices = (
        structures[
            "candidate_structure_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    if not np.array_equal(
        observed_indices,
        expected_indices,
    ):
        raise RuntimeError(
            "candidate_structure_index must be "
            "contiguous from 0..N-1."
        )

    print(
        f"Candidate structures: "
        f"{len(structures):,}"
    )

    # --------------------------------------------------------
    # Morgan generator
    # --------------------------------------------------------

    generator = (
        rdFingerprintGenerator
        .GetMorganGenerator(
            radius=MORGAN_RADIUS,
            fpSize=FINGERPRINT_BITS,
        )
    )

    # --------------------------------------------------------
    # Allocate fingerprint matrix
    # --------------------------------------------------------

    fingerprints = np.zeros(
        (
            len(structures),
            FINGERPRINT_BITS,
        ),
        dtype=np.uint8,
    )

    invalid_rows = []

    active_bits = np.zeros(
        len(structures),
        dtype=np.int32,
    )

    # --------------------------------------------------------
    # Generate fingerprints
    # --------------------------------------------------------

    print(
        "\nGenerating Morgan fingerprints..."
    )

    for row_index, row in structures.iterrows():

        smiles = str(
            row["candidate_smiles"]
        )

        molecule = Chem.MolFromSmiles(
            smiles
        )

        if molecule is None:

            invalid_rows.append(
                {
                    "candidate_structure_index": (
                        int(
                            row[
                                "candidate_structure_index"
                            ]
                        )
                    ),
                    "candidate_inchikey14": (
                        str(
                            row[
                                "candidate_inchikey14"
                            ]
                        )
                    ),
                    "candidate_smiles": (
                        smiles
                    ),
                }
            )

            continue

        fingerprint = (
            generator
            .GetFingerprint(
                molecule
            )
        )

        array = np.zeros(
            FINGERPRINT_BITS,
            dtype=np.uint8,
        )

        DataStructs.ConvertToNumpyArray(
            fingerprint,
            array,
        )

        fingerprints[
            row_index
        ] = array

        active_bits[
            row_index
        ] = int(
            array.sum()
        )

        if (
            (row_index + 1) % 5000 == 0
            or row_index + 1 == len(structures)
        ):
            print(
                f"  Processed "
                f"{row_index + 1:,}"
                f"/{len(structures):,}"
            )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    invalid_count = len(
        invalid_rows
    )

    if invalid_count > 0:

        invalid_preview = (
            pd.DataFrame(
                invalid_rows
            )
            .head(20)
        )

        print(
            "\nInvalid candidate SMILES:"
        )

        print(
            invalid_preview.to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"{invalid_count:,} candidate structures "
            "could not be parsed by RDKit."
        )

    if fingerprints.shape != (
        len(structures),
        FINGERPRINT_BITS,
    ):
        raise RuntimeError(
            "Unexpected fingerprint matrix shape."
        )

    if fingerprints.dtype != np.uint8:
        raise RuntimeError(
            "Fingerprint dtype must be uint8."
        )

    if not np.isin(
        fingerprints,
        [0, 1],
    ).all():
        raise RuntimeError(
            "Morgan fingerprint matrix must contain "
            "only binary values 0/1."
        )

    zero_fingerprints = int(
        np.sum(
            fingerprints.sum(
                axis=1
            )
            == 0
        )
    )

    if zero_fingerprints > 0:
        raise RuntimeError(
            f"{zero_fingerprints:,} valid structures "
            "produced zero-bit fingerprints."
        )

    # --------------------------------------------------------
    # Save fingerprint matrix
    # --------------------------------------------------------

    print(
        "\nSaving candidate fingerprint matrix..."
    )

    np.save(
        FINGERPRINTS_PATH,
        fingerprints,
        allow_pickle=False,
    )

    fingerprint_map = structures[
        [
            "candidate_structure_index",
            "candidate_inchikey14",
            "candidate_smiles",
        ]
    ].copy()

    fingerprint_map[
        "active_bits"
    ] = active_bits

    fingerprint_map.to_parquet(
        FINGERPRINT_MAP_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    file_size_mb = (
        FINGERPRINTS_PATH
        .stat()
        .st_size
        / (1024 ** 2)
    )

    active_float = (
        active_bits
        .astype(
            np.float64
        )
    )

    summary_lines = [
        (
            "ENVEDA CASMI 2026 - "
            "STAGE 11.2 CANDIDATE FINGERPRINT CACHE"
        ),
        "",
        (
            "Candidate structures: "
            f"{len(structures):,}"
        ),
        (
            "Morgan radius: "
            f"{MORGAN_RADIUS}"
        ),
        (
            "Fingerprint bits: "
            f"{FINGERPRINT_BITS:,}"
        ),
        (
            "Fingerprint dtype: "
            f"{fingerprints.dtype}"
        ),
        (
            "Fingerprint shape: "
            f"{fingerprints.shape}"
        ),
        (
            "Invalid SMILES: "
            f"{invalid_count:,}"
        ),
        (
            "Zero fingerprints: "
            f"{zero_fingerprints:,}"
        ),
        "",
        (
            "Active bits minimum: "
            f"{active_float.min():.0f}"
        ),
        (
            "Active bits median: "
            f"{np.median(active_float):.2f}"
        ),
        (
            "Active bits mean: "
            f"{active_float.mean():.2f}"
        ),
        (
            "Active bits P95: "
            f"{np.percentile(active_float, 95):.2f}"
        ),
        (
            "Active bits maximum: "
            f"{active_float.max():.0f}"
        ),
        "",
        (
            "Fingerprint cache size: "
            f"{file_size_mb:.2f} MB"
        ),
        "",
        (
            "Fingerprint cache: "
            f"{FINGERPRINTS_PATH}"
        ),
        (
            "Fingerprint map: "
            f"{FINGERPRINT_MAP_PATH}"
        ),
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("STAGE 11.2 SUMMARY")
    print("=" * 100)

    for line in summary_lines:
        print(line)

    print()
    print("=" * 100)
    print("STAGE 11.2 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()