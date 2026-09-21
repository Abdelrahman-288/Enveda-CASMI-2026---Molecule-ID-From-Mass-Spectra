from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import pyarrow.parquet as pq

from casmi.spectra.config import load_preprocessing_config
from casmi.spectra.preprocessing import preprocess_spectrum

from casmi.retrieval.spectral_similarity import (
    BinningConfig,
    spectral_cosine_similarity,
)


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = DATASET_DIR / "train.parquet"

PREPROCESS_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "data"
    / "spectrum_preprocessing.yaml"
)


# =========================================================
# Configuration
# =========================================================

MAX_ROWS = 50_000

MAX_POSITIVE_PAIRS = 5_000
MAX_NEGATIVE_PAIRS = 5_000

RANDOM_SEED = 42

BINNING_CONFIG = BinningConfig(
    bin_width=0.1,
    min_mz=0.0,
    max_mz=1000.0,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def percentile_summary(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean": float(
            np.mean(values)
        ),
        "median": float(
            np.median(values)
        ),
        "p10": float(
            np.percentile(
                values,
                10,
            )
        ),
        "p25": float(
            np.percentile(
                values,
                25,
            )
        ),
        "p75": float(
            np.percentile(
                values,
                75,
            )
        ),
        "p90": float(
            np.percentile(
                values,
                90,
            )
        ),
    }


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 5 STEP 3 "
        "REAL SPECTRAL SIMILARITY VALIDATION"
    )

    random.seed(
        RANDOM_SEED
    )

    preprocessing_config = (
        load_preprocessing_config(
            PREPROCESS_CONFIG_PATH,
            profile="baseline",
        )
    )

    columns = [
        "inchikey14",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_mz",
        "adduct",
        "instrument_type",
    ]

    parquet_file = pq.ParquetFile(
        TRAIN_PATH
    )

    spectra = []
    groups = defaultdict(list)

    rows_loaded = 0

    print(
        "\nLoading and preprocessing spectra..."
    )

    for batch in parquet_file.iter_batches(
        columns=columns,
        batch_size=1024,
    ):
        df = batch.to_pandas()

        for row in df.itertuples(
            index=False
        ):

            mzs, intensities = (
                preprocess_spectrum(
                    row.ms2_mzs,
                    row.ms2_normalized_intensities,
                    precursor_mz=row.precursor_mz,
                    config=preprocessing_config,
                )
            )

            index = len(
                spectra
            )

            spectra.append(
                {
                    "inchikey14": row.inchikey14,
                    "mzs": mzs,
                    "intensities": intensities,
                    "adduct": row.adduct,
                    "instrument_type": row.instrument_type,
                }
            )

            groups[
                row.inchikey14
            ].append(
                index
            )

            rows_loaded += 1

            if rows_loaded >= MAX_ROWS:
                break

        if rows_loaded >= MAX_ROWS:
            break

    print(
        f"Spectra loaded: "
        f"{len(spectra):,}"
    )

    print(
        f"Unique structures: "
        f"{len(groups):,}"
    )

    structures_with_multiple = [
        key
        for key, indices
        in groups.items()
        if len(indices) >= 2
    ]

    print(
        "Structures with >=2 spectra: "
        f"{len(structures_with_multiple):,}"
    )

    # =====================================================
    # Positive pairs
    # =====================================================

    print_section(
        "BUILDING POSITIVE PAIRS"
    )

    positive_pairs = []

    shuffled_structures = (
        structures_with_multiple.copy()
    )

    random.shuffle(
        shuffled_structures
    )

    while (
        len(positive_pairs)
        < MAX_POSITIVE_PAIRS
    ):

        added_this_round = 0

        for structure in (
            shuffled_structures
        ):

            indices = groups[
                structure
            ]

            first_index, second_index = (
                random.sample(
                    indices,
                    2,
                )
            )

            positive_pairs.append(
                (
                    first_index,
                    second_index,
                )
            )

            added_this_round += 1

            if (
                len(positive_pairs)
                >= MAX_POSITIVE_PAIRS
            ):
                break

        if added_this_round == 0:
            break

    # =====================================================
    # Negative pairs
    # =====================================================

    print_section(
        "BUILDING NEGATIVE PAIRS"
    )

    structure_keys = list(
        groups.keys()
    )

    negative_pairs = []

    while (
        len(negative_pairs)
        < MAX_NEGATIVE_PAIRS
    ):

        first_structure, second_structure = (
            random.sample(
                structure_keys,
                2,
            )
        )

        first_index = random.choice(
            groups[
                first_structure
            ]
        )

        second_index = random.choice(
            groups[
                second_structure
            ]
        )

        negative_pairs.append(
            (
                first_index,
                second_index,
            )
        )

    # =====================================================
    # Calculate similarities
    # =====================================================

    print_section(
        "CALCULATING SIMILARITIES"
    )

    positive_scores = []

    for first_index, second_index in (
        positive_pairs
    ):

        first = spectra[
            first_index
        ]

        second = spectra[
            second_index
        ]

        score = (
            spectral_cosine_similarity(
                first["mzs"],
                first["intensities"],
                second["mzs"],
                second["intensities"],
                config=BINNING_CONFIG,
            )
        )

        positive_scores.append(
            score
        )

    negative_scores = []

    for first_index, second_index in (
        negative_pairs
    ):

        first = spectra[
            first_index
        ]

        second = spectra[
            second_index
        ]

        score = (
            spectral_cosine_similarity(
                first["mzs"],
                first["intensities"],
                second["mzs"],
                second["intensities"],
                config=BINNING_CONFIG,
            )
        )

        negative_scores.append(
            score
        )

    # =====================================================
    # Summaries
    # =====================================================

    positive_summary = (
        percentile_summary(
            positive_scores
        )
    )

    negative_summary = (
        percentile_summary(
            negative_scores
        )
    )

    print_section(
        "SIMILARITY DISTRIBUTIONS"
    )

    print(
        f"{'Metric':15s}"
        f"{'Positive':>15s}"
        f"{'Negative':>15s}"
    )

    print("-" * 45)

    for metric in [
        "mean",
        "median",
        "p10",
        "p25",
        "p75",
        "p90",
    ]:

        print(
            f"{metric:15s}"
            f"{positive_summary[metric]:15.4f}"
            f"{negative_summary[metric]:15.4f}"
        )

    # =====================================================
    # Pairwise discrimination probability
    # =====================================================

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    comparisons = 100_000

    positive_array = np.asarray(
        positive_scores,
        dtype=float,
    )

    negative_array = np.asarray(
        negative_scores,
        dtype=float,
    )

    sampled_positive = (
        rng.choice(
            positive_array,
            size=comparisons,
            replace=True,
        )
    )

    sampled_negative = (
        rng.choice(
            negative_array,
            size=comparisons,
            replace=True,
        )
    )

    probability = np.mean(
        sampled_positive
        > sampled_negative
    )

    print_section(
        "DISCRIMINATION CHECK"
    )

    print(
        "Probability that a random "
        "same-structure pair scores higher "
        "than a random different-structure pair:"
    )

    print(
        f"{probability:.4%}"
    )

    # =====================================================
    # Threshold sanity checks
    # =====================================================

    print_section(
        "THRESHOLD SANITY CHECK"
    )

    for threshold in [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
    ]:

        positive_above = np.mean(
            positive_array
            >= threshold
        )

        negative_above = np.mean(
            negative_array
            >= threshold
        )

        print(
            f"score >= {threshold:.1f} | "
            f"positive="
            f"{positive_above:7.2%} | "
            f"negative="
            f"{negative_above:7.2%}"
        )

    print_section(
        "STAGE 5 STEP 3 COMPLETE"
    )


if __name__ == "__main__":
    main()