from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi.spectra.config import load_preprocessing_config
from casmi.spectra.preprocessing import preprocess_spectrum


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = (
    DATASET_DIR
    / "train.parquet"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage6_training_manifest.csv"
)

PREPROCESS_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "data"
    / "spectrum_preprocessing.yaml"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage6"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Cache configuration
# =========================================================

MAX_PEAKS = 128

MZ_SCALE = 1000.0

PROGRESS_EVERY = 50_000


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 6 STEP 2 "
        "NEURAL SPECTRUM CACHE"
    )

    # =====================================================
    # Load manifest
    # =====================================================

    print(
        "\nLoading Stage 6 manifest..."
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    manifest = manifest.sort_values(
        "row_index"
    ).reset_index(
        drop=True
    )

    n_spectra = len(
        manifest
    )

    print(
        f"Manifest spectra: "
        f"{n_spectra:,}"
    )

    print(
        f"Training spectra: "
        f"{(manifest['split'] == 'train').sum():,}"
    )

    print(
        f"Validation spectra: "
        f"{(manifest['split'] == 'validation').sum():,}"
    )

    # =====================================================
    # Structure label mapping
    # =====================================================

    structures = (
        manifest[
            "inchikey14"
        ]
        .drop_duplicates()
        .tolist()
    )

    structure_to_id = {
        structure: index
        for index, structure
        in enumerate(
            structures
        )
    }

    labels = (
        manifest[
            "inchikey14"
        ]
        .map(
            structure_to_id
        )
        .to_numpy(
            dtype=np.int32
        )
    )

    print(
        f"Unique structures: "
        f"{len(structures):,}"
    )

    # =====================================================
    # Output files
    # =====================================================

    mz_path = (
        CACHE_DIR
        / "mzs.npy"
    )

    intensity_path = (
        CACHE_DIR
        / "intensities.npy"
    )

    mask_path = (
        CACHE_DIR
        / "mask.npy"
    )

    label_path = (
        CACHE_DIR
        / "labels.npy"
    )

    row_index_path = (
        CACHE_DIR
        / "row_indices.npy"
    )

    precursor_path = (
        CACHE_DIR
        / "precursor_mz.npy"
    )

    split_path = (
        CACHE_DIR
        / "split.npy"
    )

    metadata_path = (
        CACHE_DIR
        / "cache_metadata.json"
    )

    structure_map_path = (
        CACHE_DIR
        / "structure_map.csv"
    )

    # =====================================================
    # Create memory-mapped arrays
    # =====================================================

    print(
        "\nCreating memory-mapped cache..."
    )

    mz_cache = np.lib.format.open_memmap(
        mz_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            n_spectra,
            MAX_PEAKS,
        ),
    )

    intensity_cache = np.lib.format.open_memmap(
        intensity_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            n_spectra,
            MAX_PEAKS,
        ),
    )

    mask_cache = np.lib.format.open_memmap(
        mask_path,
        mode="w+",
        dtype=np.bool_,
        shape=(
            n_spectra,
            MAX_PEAKS,
        ),
    )

    label_cache = np.lib.format.open_memmap(
        label_path,
        mode="w+",
        dtype=np.int32,
        shape=(
            n_spectra,
        ),
    )

    row_index_cache = np.lib.format.open_memmap(
        row_index_path,
        mode="w+",
        dtype=np.int64,
        shape=(
            n_spectra,
        ),
    )

    precursor_cache = np.lib.format.open_memmap(
        precursor_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            n_spectra,
        ),
    )

    split_cache = np.lib.format.open_memmap(
        split_path,
        mode="w+",
        dtype=np.uint8,
        shape=(
            n_spectra,
        ),
    )

    # Initialize.
    mz_cache[:] = 0.0
    intensity_cache[:] = 0.0
    mask_cache[:] = False

    label_cache[:] = labels

    row_index_cache[:] = (
        manifest[
            "row_index"
        ].to_numpy(
            dtype=np.int64
        )
    )

    precursor_cache[:] = (
        manifest[
            "precursor_mz"
        ].to_numpy(
            dtype=np.float32
        )
    )

    split_cache[:] = np.where(
        manifest["split"].eq(
            "train"
        ),
        0,
        1,
    ).astype(
        np.uint8
    )

    # =====================================================
    # Stage 3 preprocessing
    # =====================================================

    preprocessing_config = (
        load_preprocessing_config(
            PREPROCESS_CONFIG_PATH,
            profile="baseline",
        )
    )

    # =====================================================
    # Fast row lookup
    # =====================================================

    row_to_cache_position = {
        int(row_index): cache_position
        for cache_position, row_index
        in enumerate(
            manifest[
                "row_index"
            ].to_numpy()
        )
    }

    needed_rows = set(
        row_to_cache_position
    )

    print(
        f"\nRows required from parquet: "
        f"{len(needed_rows):,}"
    )

    # =====================================================
    # Stream source parquet once
    # =====================================================

    parquet_file = pq.ParquetFile(
        TRAIN_PATH
    )

    columns = [
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_mz",
    ]

    global_row_index = 0
    processed = 0

    start_time = time.time()

    print(
        "\nStreaming and preprocessing spectra..."
    )

    for batch in parquet_file.iter_batches(
        columns=columns,
        batch_size=4096,
    ):

        batch_df = batch.to_pandas()

        for row in batch_df.itertuples(
            index=False
        ):

            cache_position = (
                row_to_cache_position.get(
                    global_row_index
                )
            )

            if cache_position is not None:

                mzs, intensities = (
                    preprocess_spectrum(
                        row.ms2_mzs,
                        row.ms2_normalized_intensities,
                        precursor_mz=(
                            row.precursor_mz
                        ),
                        config=(
                            preprocessing_config
                        ),
                    )
                )

                n_peaks = min(
                    len(mzs),
                    MAX_PEAKS,
                )

                if n_peaks > 0:

                    mz_values = np.asarray(
                        mzs[:n_peaks],
                        dtype=np.float32,
                    )

                    intensity_values = np.asarray(
                        intensities[:n_peaks],
                        dtype=np.float32,
                    )

                    # Neural-network-friendly m/z scale.
                    mz_values = (
                        mz_values
                        / MZ_SCALE
                    )

                    mz_cache[
                        cache_position,
                        :n_peaks,
                    ] = mz_values

                    intensity_cache[
                        cache_position,
                        :n_peaks,
                    ] = intensity_values

                    mask_cache[
                        cache_position,
                        :n_peaks,
                    ] = True

                processed += 1

                if (
                    processed
                    % PROGRESS_EVERY
                    == 0
                ):

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    rate = (
                        processed
                        / elapsed
                    )

                    remaining = (
                        n_spectra
                        - processed
                    )

                    eta_seconds = (
                        remaining / rate
                        if rate > 0
                        else 0
                    )

                    print(
                        f"Processed "
                        f"{processed:,}"
                        f"/{n_spectra:,} "
                        f"({processed / n_spectra:.1%}) "
                        f"| "
                        f"{rate:,.0f} spectra/s "
                        f"| ETA "
                        f"{eta_seconds / 60:.1f} min"
                    )

            global_row_index += 1

        if processed >= n_spectra:
            break

    # =====================================================
    # Flush cache
    # =====================================================

    mz_cache.flush()
    intensity_cache.flush()
    mask_cache.flush()
    label_cache.flush()
    row_index_cache.flush()
    precursor_cache.flush()
    split_cache.flush()

    # =====================================================
    # Validation
    # =====================================================

    print(
        "\nValidating cache..."
    )

    if processed != n_spectra:
        raise RuntimeError(
            f"Expected {n_spectra:,} spectra "
            f"but processed {processed:,}."
        )

    peak_counts = (
        mask_cache.sum(
            axis=1
        )
    )

    empty_spectra = int(
        np.sum(
            peak_counts == 0
        )
    )

    max_peaks_found = int(
        peak_counts.max()
    )

    mean_peaks = float(
        peak_counts.mean()
    )

    median_peaks = float(
        np.median(
            peak_counts
        )
    )

    max_mz_value = float(
        mz_cache.max()
    )

    min_valid_intensity = float(
        intensity_cache[
            mask_cache
        ].min()
    )

    max_valid_intensity = float(
        intensity_cache[
            mask_cache
        ].max()
    )

    # =====================================================
    # Structure mapping
    # =====================================================

    structure_map = pd.DataFrame(
        {
            "structure_id": np.arange(
                len(structures),
                dtype=np.int32,
            ),
            "inchikey14": structures,
        }
    )

    structure_map.to_csv(
        structure_map_path,
        index=False,
    )

    # =====================================================
    # Metadata
    # =====================================================

    metadata = {
        "num_spectra": int(
            n_spectra
        ),
        "num_structures": int(
            len(structures)
        ),
        "max_peaks": int(
            MAX_PEAKS
        ),
        "mz_scale": float(
            MZ_SCALE
        ),
        "train_split_value": 0,
        "validation_split_value": 1,
        "mean_peak_count": mean_peaks,
        "median_peak_count": median_peaks,
        "max_peak_count": max_peaks_found,
        "empty_spectra": empty_spectra,
        "max_normalized_mz": max_mz_value,
        "min_intensity": min_valid_intensity,
        "max_intensity": max_valid_intensity,
    }

    metadata_path.write_text(
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

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 6 CACHE SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Spectra cached: "
        f"{n_spectra:,}"
    )

    print(
        f"Structures: "
        f"{len(structures):,}"
    )

    print(
        f"Maximum peaks: "
        f"{MAX_PEAKS}"
    )

    print(
        f"Mean peaks: "
        f"{mean_peaks:.2f}"
    )

    print(
        f"Median peaks: "
        f"{median_peaks:.2f}"
    )

    print(
        f"Maximum observed peaks: "
        f"{max_peaks_found}"
    )

    print(
        f"Empty spectra: "
        f"{empty_spectra:,}"
    )

    print(
        f"Maximum normalized m/z: "
        f"{max_mz_value:.4f}"
    )

    print(
        f"Intensity range: "
        f"{min_valid_intensity:.6f} "
        f"to "
        f"{max_valid_intensity:.6f}"
    )

    print(
        f"Elapsed time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        "\nCache directory:"
    )

    print(
        CACHE_DIR
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 6 STEP 2 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()