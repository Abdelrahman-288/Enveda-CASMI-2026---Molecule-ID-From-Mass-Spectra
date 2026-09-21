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


MAX_ROWS = 50_000

MAX_POSITIVE_PAIRS = 3_000
MAX_NEGATIVE_PAIRS = 3_000

RANDOM_SEED = 42

BIN_WIDTHS = [
    0.02,
    0.05,
    0.10,
    0.20,
]


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def build_pairs(
    spectra,
    groups,
):
    random.seed(RANDOM_SEED)

    structures_with_multiple = [
        structure
        for structure, indices in groups.items()
        if len(indices) >= 2
    ]

    positive_pairs = []

    shuffled = structures_with_multiple.copy()

    random.shuffle(shuffled)

    while len(positive_pairs) < MAX_POSITIVE_PAIRS:

        for structure in shuffled:

            indices = groups[structure]

            first, second = random.sample(
                indices,
                2,
            )

            positive_pairs.append(
                (first, second)
            )

            if (
                len(positive_pairs)
                >= MAX_POSITIVE_PAIRS
            ):
                break

    structure_keys = list(
        groups.keys()
    )

    negative_pairs = []

    while len(negative_pairs) < MAX_NEGATIVE_PAIRS:

        first_structure, second_structure = (
            random.sample(
                structure_keys,
                2,
            )
        )

        first = random.choice(
            groups[first_structure]
        )

        second = random.choice(
            groups[second_structure]
        )

        negative_pairs.append(
            (first, second)
        )

    return (
        positive_pairs,
        negative_pairs,
    )


def calculate_scores(
    spectra,
    pairs,
    config,
):
    scores = []

    for first_index, second_index in pairs:

        first = spectra[first_index]
        second = spectra[second_index]

        score = spectral_cosine_similarity(
            first["mzs"],
            first["intensities"],
            second["mzs"],
            second["intensities"],
            config=config,
        )

        scores.append(score)

    return np.asarray(
        scores,
        dtype=float,
    )


def discrimination_probability(
    positive,
    negative,
):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    n = 100_000

    positive_sample = rng.choice(
        positive,
        n,
        replace=True,
    )

    negative_sample = rng.choice(
        negative,
        n,
        replace=True,
    )

    return float(
        np.mean(
            positive_sample
            > negative_sample
        )
    )


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 5 STEP 4 "
        "BIN WIDTH COMPARISON"
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

            index = len(spectra)

            spectra.append(
                {
                    "inchikey14": row.inchikey14,
                    "mzs": mzs,
                    "intensities": intensities,
                }
            )

            groups[
                row.inchikey14
            ].append(index)

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

    positive_pairs, negative_pairs = (
        build_pairs(
            spectra,
            groups,
        )
    )

    print(
        f"Positive pairs: "
        f"{len(positive_pairs):,}"
    )

    print(
        f"Negative pairs: "
        f"{len(negative_pairs):,}"
    )

    print_section(
        "BIN WIDTH RESULTS"
    )

    print(
        f"{'Bin width':>12s}"
        f"{'Pos mean':>12s}"
        f"{'Pos median':>14s}"
        f"{'Neg mean':>12s}"
        f"{'Neg median':>14s}"
        f"{'P(pos>neg)':>14s}"
    )

    print("-" * 80)

    for width in BIN_WIDTHS:

        config = BinningConfig(
            bin_width=width,
            min_mz=0.0,
            max_mz=1000.0,
        )

        positive_scores = calculate_scores(
            spectra,
            positive_pairs,
            config,
        )

        negative_scores = calculate_scores(
            spectra,
            negative_pairs,
            config,
        )

        probability = (
            discrimination_probability(
                positive_scores,
                negative_scores,
            )
        )

        print(
            f"{width:12.2f}"
            f"{np.mean(positive_scores):12.4f}"
            f"{np.median(positive_scores):14.4f}"
            f"{np.mean(negative_scores):12.4f}"
            f"{np.median(negative_scores):14.4f}"
            f"{probability:14.4%}"
        )

    print_section(
        "STAGE 5 STEP 4 COMPLETE"
    )


if __name__ == "__main__":
    main()