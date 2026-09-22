from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


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

STAGE8_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STAGE6_STRUCTURE_MAP = (
    STAGE6_DIR
    / "structure_map.csv"
)

STAGE8_STRUCTURE_MAP = (
    STAGE8_DIR
    / "structure_map.csv"
)

STAGE6_LABELS = (
    STAGE6_DIR
    / "labels.npy"
)

STAGE6_SPLIT = (
    STAGE6_DIR
    / "split.npy"
)

STAGE6_TO_STAGE8_PATH = (
    STAGE8_DIR
    / "stage6_to_stage8.npy"
)

SPECTRUM_MOLECULE_INDEX_PATH = (
    STAGE8_DIR
    / "spectrum_molecule_indices.npy"
)

METADATA_PATH = (
    STAGE8_DIR
    / "alignment_metadata.json"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage8_alignment_summary.txt"
)


# =========================================================
# Main
# =========================================================

def main():

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 8 STEP 3 "
        "SPECTRUM-MOLECULE ALIGNMENT"
    )

    start_time = time.time()

    # =====================================================
    # Load structure maps
    # =====================================================

    print(
        "\nLoading Stage 6 structure map..."
    )

    stage6_map = pd.read_csv(
        STAGE6_STRUCTURE_MAP
    )

    print(
        f"Stage 6 structures: "
        f"{len(stage6_map):,}"
    )

    print(
        "\nLoading Stage 8 structure map..."
    )

    stage8_map = pd.read_csv(
        STAGE8_STRUCTURE_MAP
    )

    print(
        f"Stage 8 structures: "
        f"{len(stage8_map):,}"
    )

    # =====================================================
    # Validate required columns
    # =====================================================

    required_stage6 = {
        "structure_id",
        "inchikey14",
    }

    required_stage8 = {
        "structure_index",
        "inchikey14",
    }

    missing_stage6 = (
        required_stage6
        - set(
            stage6_map.columns
        )
    )

    missing_stage8 = (
        required_stage8
        - set(
            stage8_map.columns
        )
    )

    if missing_stage6:

        raise RuntimeError(
            "Stage 6 structure map missing columns: "
            f"{sorted(missing_stage6)}"
        )

    if missing_stage8:

        raise RuntimeError(
            "Stage 8 structure map missing columns: "
            f"{sorted(missing_stage8)}"
        )

    # =====================================================
    # Validate uniqueness
    # =====================================================

    stage6_duplicate_ids = int(
        stage6_map[
            "structure_id"
        ]
        .duplicated()
        .sum()
    )

    stage6_duplicate_keys = int(
        stage6_map[
            "inchikey14"
        ]
        .duplicated()
        .sum()
    )

    stage8_duplicate_indices = int(
        stage8_map[
            "structure_index"
        ]
        .duplicated()
        .sum()
    )

    stage8_duplicate_keys = int(
        stage8_map[
            "inchikey14"
        ]
        .duplicated()
        .sum()
    )

    print(
        "\nUniqueness validation:"
    )

    print(
        f"Stage 6 duplicate structure IDs: "
        f"{stage6_duplicate_ids:,}"
    )

    print(
        f"Stage 6 duplicate inchikey14: "
        f"{stage6_duplicate_keys:,}"
    )

    print(
        f"Stage 8 duplicate structure indices: "
        f"{stage8_duplicate_indices:,}"
    )

    print(
        f"Stage 8 duplicate inchikey14: "
        f"{stage8_duplicate_keys:,}"
    )

    if (
        stage6_duplicate_ids
        or stage6_duplicate_keys
        or stage8_duplicate_indices
        or stage8_duplicate_keys
    ):

        raise RuntimeError(
            "Structure maps must be one-to-one."
        )

    # =====================================================
    # Validate continuous Stage 6 IDs
    # =====================================================

    stage6_ids = (
        stage6_map[
            "structure_id"
        ]
        .to_numpy(
            dtype=np.int64,
        )
    )

    expected_stage6_ids = np.arange(
        len(stage6_map),
        dtype=np.int64,
    )

    if not np.array_equal(
        np.sort(
            stage6_ids
        ),
        expected_stage6_ids,
    ):

        raise RuntimeError(
            "Stage 6 structure IDs are not contiguous "
            "from 0 to N-1."
        )

    # =====================================================
    # Validate continuous Stage 8 indices
    # =====================================================

    stage8_indices = (
        stage8_map[
            "structure_index"
        ]
        .to_numpy(
            dtype=np.int64,
        )
    )

    expected_stage8_indices = np.arange(
        len(stage8_map),
        dtype=np.int64,
    )

    if not np.array_equal(
        np.sort(
            stage8_indices
        ),
        expected_stage8_indices,
    ):

        raise RuntimeError(
            "Stage 8 structure indices are not contiguous "
            "from 0 to N-1."
        )

    # =====================================================
    # Compare structure universes
    # =====================================================

    stage6_keys = set(
        stage6_map[
            "inchikey14"
        ].astype(str)
    )

    stage8_keys = set(
        stage8_map[
            "inchikey14"
        ].astype(str)
    )

    stage6_only = (
        stage6_keys
        - stage8_keys
    )

    stage8_only = (
        stage8_keys
        - stage6_keys
    )

    print(
        "\nStructure universe comparison:"
    )

    print(
        f"Stage 6 only: "
        f"{len(stage6_only):,}"
    )

    print(
        f"Stage 8 only: "
        f"{len(stage8_only):,}"
    )

    if stage6_only or stage8_only:

        raise RuntimeError(
            "Stage 6 and Stage 8 structure universes "
            "do not match exactly."
        )

    # =====================================================
    # Build Stage 6 ID -> Stage 8 index mapping
    # =====================================================

    print(
        "\nBuilding Stage 6 -> Stage 8 mapping..."
    )

    stage8_lookup = dict(
        zip(
            stage8_map[
                "inchikey14"
            ].astype(str),
            stage8_map[
                "structure_index"
            ].astype(
                np.int32
            ),
        )
    )

    stage6_to_stage8 = np.empty(
        len(stage6_map),
        dtype=np.int32,
    )

    for row in stage6_map.itertuples(
        index=False
    ):

        stage6_to_stage8[
            int(
                row.structure_id
            )
        ] = int(
            stage8_lookup[
                str(
                    row.inchikey14
                )
            ]
        )

    # =====================================================
    # Validate mapping is a permutation
    # =====================================================

    unique_mapped = np.unique(
        stage6_to_stage8
    )

    if len(
        unique_mapped
    ) != len(
        stage8_map
    ):

        raise RuntimeError(
            "Stage 6 -> Stage 8 mapping is not one-to-one."
        )

    if (
        int(
            unique_mapped.min()
        )
        != 0
        or int(
            unique_mapped.max()
        )
        != len(stage8_map) - 1
    ):

        raise RuntimeError(
            "Stage 6 -> Stage 8 mapping does not cover "
            "the complete Stage 8 index range."
        )

    identical_positions = int(
        np.sum(
            stage6_to_stage8
            == np.arange(
                len(
                    stage6_to_stage8
                ),
                dtype=np.int32,
            )
        )
    )

    reordered_positions = (
        len(
            stage6_to_stage8
        )
        - identical_positions
    )

    print(
        f"Same-position structures: "
        f"{identical_positions:,}"
    )

    print(
        f"Reordered structures: "
        f"{reordered_positions:,}"
    )

    # =====================================================
    # Save structure mapping
    # =====================================================

    np.save(
        STAGE6_TO_STAGE8_PATH,
        stage6_to_stage8,
    )

    # =====================================================
    # Load spectrum labels
    # =====================================================

    print(
        "\nLoading Stage 6 spectrum labels..."
    )

    labels = np.load(
        STAGE6_LABELS,
        mmap_mode="r",
    )

    print(
        f"Spectra: "
        f"{len(labels):,}"
    )

    print(
        f"Label dtype: "
        f"{labels.dtype}"
    )

    print(
        f"Label range: "
        f"{int(labels.min()):,} "
        f"to "
        f"{int(labels.max()):,}"
    )

    if int(
        labels.min()
    ) < 0:

        raise RuntimeError(
            "Negative structure labels detected."
        )

    if int(
        labels.max()
    ) >= len(
        stage6_to_stage8
    ):

        raise RuntimeError(
            "Spectrum labels exceed Stage 6 structure map."
        )

    # =====================================================
    # Convert every spectrum label to Stage 8 molecule index
    # =====================================================

    print(
        "\nBuilding per-spectrum molecule indices..."
    )

    spectrum_molecule_indices = (
        stage6_to_stage8[
            labels
        ]
    )

    spectrum_molecule_indices = (
        np.asarray(
            spectrum_molecule_indices,
            dtype=np.int32,
        )
    )

    if len(
        spectrum_molecule_indices
    ) != len(
        labels
    ):

        raise RuntimeError(
            "Spectrum alignment length mismatch."
        )

    # =====================================================
    # Save spectrum mapping
    # =====================================================

    np.save(
        SPECTRUM_MOLECULE_INDEX_PATH,
        spectrum_molecule_indices,
    )

    # =====================================================
    # Validate split information
    # =====================================================

    split = np.load(
        STAGE6_SPLIT,
        mmap_mode="r",
    )

    if len(
        split
    ) != len(
        labels
    ):

        raise RuntimeError(
            "Stage 6 split array length mismatch."
        )

    train_mask = (
        split == 0
    )

    validation_mask = (
        split == 1
    )

    train_spectra = int(
        train_mask.sum()
    )

    validation_spectra = int(
        validation_mask.sum()
    )

    train_molecules = int(
        np.unique(
            spectrum_molecule_indices[
                train_mask
            ]
        ).size
    )

    validation_molecules = int(
        np.unique(
            spectrum_molecule_indices[
                validation_mask
            ]
        ).size
    )

    train_structure_set = set(
        np.unique(
            spectrum_molecule_indices[
                train_mask
            ]
        ).tolist()
    )

    validation_structure_set = set(
        np.unique(
            spectrum_molecule_indices[
                validation_mask
            ]
        ).tolist()
    )

    split_overlap = len(
        train_structure_set
        & validation_structure_set
    )

    # =====================================================
    # Fingerprint cache validation
    # =====================================================

    fingerprint_path = (
        STAGE8_DIR
        / "morgan_fingerprints.npy"
    )

    fingerprints = np.load(
        fingerprint_path,
        mmap_mode="r",
    )

    if (
        fingerprints.shape[0]
        != len(
            stage8_map
        )
    ):

        raise RuntimeError(
            "Fingerprint row count does not match "
            "Stage 8 structure map."
        )

    # =====================================================
    # Random alignment verification
    # =====================================================

    print(
        "\nRunning random alignment verification..."
    )

    rng = np.random.default_rng(
        42
    )

    sample_size = min(
        10_000,
        len(
            labels
        ),
    )

    sampled_spectra = rng.choice(
        len(
            labels
        ),
        size=sample_size,
        replace=False,
    )

    stage6_key_lookup = (
        stage6_map
        .set_index(
            "structure_id"
        )[
            "inchikey14"
        ]
        .astype(str)
        .to_dict()
    )

    stage8_key_array = (
        stage8_map
        .sort_values(
            "structure_index"
        )[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    verified = 0

    for spectrum_index in sampled_spectra:

        stage6_label = int(
            labels[
                spectrum_index
            ]
        )

        stage8_index = int(
            spectrum_molecule_indices[
                spectrum_index
            ]
        )

        stage6_key = (
            stage6_key_lookup[
                stage6_label
            ]
        )

        stage8_key = (
            stage8_key_array[
                stage8_index
            ]
        )

        if (
            stage6_key
            == stage8_key
        ):

            verified += 1

    verification_rate = (
        verified
        / sample_size
    )

    print(
        f"Verified alignments: "
        f"{verified:,}/"
        f"{sample_size:,}"
    )

    print(
        f"Verification rate: "
        f"{verification_rate:.6%}"
    )

    if verification_rate < 1.0:

        raise RuntimeError(
            "Spectrum-to-molecule alignment failed."
        )

    # =====================================================
    # Metadata
    # =====================================================

    elapsed = (
        time.time()
        - start_time
    )

    stage6_to_stage8_mb = (
        STAGE6_TO_STAGE8_PATH.stat().st_size
        / 1024**2
    )

    spectrum_mapping_mb = (
        SPECTRUM_MOLECULE_INDEX_PATH.stat().st_size
        / 1024**2
    )

    metadata = {
        "num_structures": int(
            len(
                stage6_to_stage8
            )
        ),
        "num_spectra": int(
            len(
                labels
            )
        ),
        "mapping_dtype": (
            str(
                stage6_to_stage8.dtype
            )
        ),
        "train_spectra": (
            train_spectra
        ),
        "validation_spectra": (
            validation_spectra
        ),
        "train_molecules": (
            train_molecules
        ),
        "validation_molecules": (
            validation_molecules
        ),
        "structure_split_overlap": int(
            split_overlap
        ),
        "verification_sample_size": int(
            sample_size
        ),
        "verification_rate": float(
            verification_rate
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

    summary_lines = [
        (
            "ENVEDA CASMI 2026 — "
            "STAGE 8 SPECTRUM-MOLECULE ALIGNMENT"
        ),
        "",
        (
            f"Structures: "
            f"{len(stage6_to_stage8):,}"
        ),
        (
            f"Spectra: "
            f"{len(labels):,}"
        ),
        "",
        (
            f"Same-position structures: "
            f"{identical_positions:,}"
        ),
        (
            f"Reordered structures: "
            f"{reordered_positions:,}"
        ),
        "",
        (
            f"Train spectra: "
            f"{train_spectra:,}"
        ),
        (
            f"Validation spectra: "
            f"{validation_spectra:,}"
        ),
        (
            f"Train molecules: "
            f"{train_molecules:,}"
        ),
        (
            f"Validation molecules: "
            f"{validation_molecules:,}"
        ),
        (
            f"Structure split overlap: "
            f"{split_overlap:,}"
        ),
        "",
        (
            f"Verified alignments: "
            f"{verified:,}/{sample_size:,}"
        ),
        (
            f"Verification rate: "
            f"{verification_rate:.6%}"
        ),
        "",
        (
            f"Structure mapping size MB: "
            f"{stage6_to_stage8_mb:.2f}"
        ),
        (
            f"Spectrum mapping size MB: "
            f"{spectrum_mapping_mb:.2f}"
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
    # Console summary
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 8 ALIGNMENT SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Structures: "
        f"{len(stage6_to_stage8):,}"
    )

    print(
        f"Spectra: "
        f"{len(labels):,}"
    )

    print(
        f"Same-position structures: "
        f"{identical_positions:,}"
    )

    print(
        f"Reordered structures: "
        f"{reordered_positions:,}"
    )

    print(
        f"\nTrain spectra: "
        f"{train_spectra:,}"
    )

    print(
        f"Validation spectra: "
        f"{validation_spectra:,}"
    )

    print(
        f"Train molecules: "
        f"{train_molecules:,}"
    )

    print(
        f"Validation molecules: "
        f"{validation_molecules:,}"
    )

    print(
        f"Structure overlap: "
        f"{split_overlap:,}"
    )

    print(
        f"\nAlignment verification: "
        f"{verification_rate:.6%}"
    )

    print(
        f"Stage6→Stage8 map: "
        f"{stage6_to_stage8_mb:.2f} MB"
    )

    print(
        f"Spectrum→molecule map: "
        f"{spectrum_mapping_mb:.2f} MB"
    )

    print(
        f"Elapsed: "
        f"{elapsed:.2f} s"
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 8 STEP 3 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()