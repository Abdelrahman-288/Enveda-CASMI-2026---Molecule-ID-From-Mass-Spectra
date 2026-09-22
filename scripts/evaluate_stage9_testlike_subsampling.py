from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from casmi.chemistry.adducts import (
    is_supported_adduct,
    is_trusted_for_mass_filter,
)

from casmi.chemistry.mass_filter_policy import (
    get_mass_filter_decision,
)

from casmi.models.molecule_encoder import MoleculeEncoder
from casmi.models.spectrum_encoder import SpectrumEncoder

from casmi.retrieval.candidate_generation import (
    AdductAwareCandidateGenerator,
)

from casmi.retrieval.candidate_mass_index import (
    CandidateMassIndex,
)

from casmi.training.stage8_dataset import (
    SpectrumMoleculeDataset,
)


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = (
    DATASET_DIR
    / "train.parquet"
)

TEST_PATH = (
    DATASET_DIR
    / "test.parquet"
)

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

STAGE8_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "stage8"
    / "candidate_aware_best.pt"
)

STAGE7_VARIANTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "stage7_candidate_mass_variants.csv"
)

STAGE8_STRUCTURE_MAP = (
    STAGE8_DIR
    / "structure_map.csv"
)

STAGE6_ROW_INDICES_PATH = (
    STAGE6_DIR
    / "row_indices.npy"
)

OUTPUT_REPEAT_RESULTS = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage9_testlike_repeat_results.csv"
)

OUTPUT_GROUP_RESULTS = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage9_testlike_group_results.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage9_testlike_subsampling_summary.txt"
)


# =========================================================
# Configuration
# =========================================================

SPECTRUM_BATCH_SIZE = 512
MOLECULE_BATCH_SIZE = 4096

NUM_WORKERS = 0

EMBEDDING_DIM = 128

REPEAT_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

AGGREGATION_METHOD = "mean"


# =========================================================
# Models
# =========================================================

def build_spectrum_encoder():

    return SpectrumEncoder(
        peak_input_dim=2,
        peak_hidden_dim=128,
        precursor_hidden_dim=32,
        pooled_hidden_dim=256,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.10,
    )


def build_molecule_encoder():

    return MoleculeEncoder(
        fingerprint_dim=2048,
        hidden_dim_1=1024,
        hidden_dim_2=512,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.10,
    )


# =========================================================
# Metrics
# =========================================================

def calculate_metrics(
    ranks,
):

    ranks = np.asarray(
        ranks,
        dtype=np.float64,
    )

    total = len(
        ranks
    )

    if total == 0:

        return None

    finite = np.isfinite(
        ranks
    )

    reciprocal = np.zeros(
        total,
        dtype=np.float64,
    )

    reciprocal[
        finite
    ] = (
        1.0
        / ranks[
            finite
        ]
    )

    def recall_at(
        k,
    ):

        return float(
            np.mean(
                finite
                & (
                    ranks
                    <= k
                )
            )
        )

    if finite.any():

        successful = ranks[
            finite
        ]

        median_rank = float(
            np.median(
                successful
            )
        )

        mean_rank = float(
            np.mean(
                successful
            )
        )

    else:

        median_rank = np.nan
        mean_rank = np.nan

    return {
        "count": int(
            total
        ),
        "coverage": float(
            np.mean(
                finite
            )
        ),
        "recall_1": recall_at(
            1
        ),
        "recall_5": recall_at(
            5
        ),
        "recall_10": recall_at(
            10
        ),
        "recall_25": recall_at(
            25
        ),
        "mrr": float(
            np.mean(
                reciprocal
            )
        ),
        "median_rank": (
            median_rank
        ),
        "mean_rank": (
            mean_rank
        ),
    }


def print_metrics(
    title,
    metrics,
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        title
    )

    print(
        "=" * 90
    )

    if metrics is None:

        print(
            "No groups."
        )

        return

    print(
        f"Groups:       "
        f"{metrics['count']:,}"
    )

    print(
        f"Coverage:     "
        f"{metrics['coverage'] * 100:.3f}%"
    )

    print(
        f"Recall@1:     "
        f"{metrics['recall_1'] * 100:.3f}%"
    )

    print(
        f"Recall@5:     "
        f"{metrics['recall_5'] * 100:.3f}%"
    )

    print(
        f"Recall@10:    "
        f"{metrics['recall_10'] * 100:.3f}%"
    )

    print(
        f"Recall@25:    "
        f"{metrics['recall_25'] * 100:.3f}%"
    )

    print(
        f"MRR:          "
        f"{metrics['mrr']:.6f}"
    )

    print(
        f"Median rank:  "
        f"{metrics['median_rank']:.1f}"
    )

    print(
        f"Mean rank:    "
        f"{metrics['mean_rank']:.2f}"
    )


# =========================================================
# Ranking
# =========================================================

def rank_true_candidate(
    scores: torch.Tensor,
    candidate_indices: np.ndarray,
    true_structure_index: int,
):

    candidate_indices = np.asarray(
        candidate_indices,
        dtype=np.int64,
    )

    locations = np.flatnonzero(
        candidate_indices
        == int(
            true_structure_index
        )
    )

    if len(
        locations
    ) == 0:

        return np.nan

    true_position = int(
        locations[
            0
        ]
    )

    true_score = scores[
        true_position
    ]

    rank = int(
        (
            scores
            > true_score
        )
        .sum()
        .item()
        + 1
    )

    return rank


# =========================================================
# Test distribution
# =========================================================

def load_test_group_size_distribution():

    test_metadata = pd.read_parquet(
        TEST_PATH,
        columns=[
            "molecule_id",
        ],
    )

    group_sizes = (
        test_metadata
        .groupby(
            "molecule_id"
        )
        .size()
    )

    counts = (
        group_sizes
        .value_counts()
        .sort_index()
    )

    values = (
        counts
        .index
        .to_numpy(
            dtype=np.int64
        )
    )

    probabilities = (
        counts
        .to_numpy(
            dtype=np.float64
        )
    )

    probabilities = (
        probabilities
        / probabilities.sum()
    )

    return (
        values,
        probabilities,
        group_sizes,
    )


# =========================================================
# Main
# =========================================================

def main():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    total_start = (
        time.perf_counter()
    )

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 9 TEST-LIKE MULTI-SPECTRUM SUBSAMPLING"
    )

    print(
        f"\nGPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    # =====================================================
    # File checks
    # =====================================================

    required_paths = [
        TRAIN_PATH,
        TEST_PATH,
        STAGE6_CHECKPOINT,
        STAGE8_CHECKPOINT,
        STAGE7_VARIANTS_PATH,
        STAGE8_STRUCTURE_MAP,
        STAGE6_ROW_INDICES_PATH,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

    # =====================================================
    # Read actual test spectrum-count distribution
    # =====================================================

    print(
        "\nLoading empirical test-set "
        "spectrum-count distribution..."
    )

    (
        test_count_values,
        test_count_probabilities,
        test_group_sizes,
    ) = (
        load_test_group_size_distribution()
    )

    print(
        f"Test molecules: "
        f"{len(test_group_sizes):,}"
    )

    print(
        f"Test mean spectra/molecule: "
        f"{test_group_sizes.mean():.4f}"
    )

    print(
        f"Test median: "
        f"{test_group_sizes.median():.1f}"
    )

    print(
        f"Test range: "
        f"{test_group_sizes.min()}–"
        f"{test_group_sizes.max()}"
    )

    print(
        "\nEmpirical test distribution:"
    )

    for value, probability in zip(
        test_count_values,
        test_count_probabilities,
    ):

        count = int(
            (
                test_group_sizes
                == value
            )
            .sum()
        )

        print(
            f"{int(value)} spectra: "
            f"{count:>3} molecules "
            f"({probability * 100:6.2f}%)"
        )

    # =====================================================
    # Validation dataset
    # =====================================================

    print(
        "\nLoading Stage 8 validation dataset..."
    )

    dataset = (
        SpectrumMoleculeDataset(
            stage6_dir=STAGE6_DIR,
            stage8_dir=STAGE8_DIR,
            split_value=1,
        )
    )

    print(
        f"Validation spectra: "
        f"{len(dataset):,}"
    )

    loader = DataLoader(
        dataset,
        batch_size=SPECTRUM_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # =====================================================
    # Structure map
    # =====================================================

    print(
        "\nLoading structure map..."
    )

    structure_map = pd.read_csv(
        STAGE8_STRUCTURE_MAP
    )

    structure_map[
        "structure_index"
    ] = (
        structure_map[
            "structure_index"
        ]
        .astype(
            np.int64
        )
    )

    structure_map[
        "inchikey14"
    ] = (
        structure_map[
            "inchikey14"
        ]
        .astype(str)
    )

    inchikey_to_structure_index = dict(
        zip(
            structure_map[
                "inchikey14"
            ],
            structure_map[
                "structure_index"
            ],
        )
    )

    print(
        f"Structures: "
        f"{len(structure_map):,}"
    )

    # =====================================================
    # Stage 7 mass index
    # =====================================================

    print(
        "\nLoading Stage 7 candidate mass index..."
    )

    variants = pd.read_csv(
        STAGE7_VARIANTS_PATH
    )

    mass_index = CandidateMassIndex(
        variants[
            [
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
                "exact_mass",
            ]
        ]
    )

    print(
        f"Indexed structures: "
        f"{mass_index.unique_structure_count:,}"
    )

    print(
        f"Mass variants: "
        f"{mass_index.variant_count:,}"
    )

    generator_cache = {}

    def get_generator(
        tolerance_ppm,
    ):

        tolerance_ppm = float(
            tolerance_ppm
        )

        if tolerance_ppm not in (
            generator_cache
        ):

            generator_cache[
                tolerance_ppm
            ] = (
                AdductAwareCandidateGenerator(
                    mass_index=mass_index,
                    tolerance_ppm=(
                        tolerance_ppm
                    ),
                )
            )

        return generator_cache[
            tolerance_ppm
        ]

    # =====================================================
    # Metadata
    # =====================================================

    print(
        "\nLoading validation metadata..."
    )

    train_metadata = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "inchikey14",
            "ingest_lib",
            "instrument_type",
            "adduct",
            "precursor_mz",
        ],
    )

    row_indices = np.load(
        STAGE6_ROW_INDICES_PATH,
        mmap_mode="r",
    )

    original_row_indices = (
        row_indices[
            dataset.indices
        ]
    )

    metadata = (
        train_metadata
        .iloc[
            original_row_indices
        ]
        .reset_index(
            drop=True
        )
    )

    metadata[
        "validation_index"
    ] = np.arange(
        len(
            metadata
        ),
        dtype=np.int64,
    )

    # =====================================================
    # timsTOF domain only
    # =====================================================

    timstof_metadata = (
        metadata[
            metadata[
                "instrument_type"
            ]
            .fillna("")
            .astype(str)
            .str.lower()
            .eq(
                "timstof"
            )
        ]
        .copy()
    )

    print(
        f"timsTOF validation spectra: "
        f"{len(timstof_metadata):,}"
    )

    # =====================================================
    # Load encoders
    # =====================================================

    print(
        "\nLoading Stage 6 spectrum encoder..."
    )

    spectrum_encoder = (
        build_spectrum_encoder()
    )

    stage6_checkpoint = torch.load(
        STAGE6_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    spectrum_encoder.load_state_dict(
        stage6_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    spectrum_encoder = (
        spectrum_encoder
        .to(
            device
        )
        .eval()
    )

    print(
        "Spectrum encoder loaded."
    )

    print(
        "\nLoading candidate-aware molecule encoder..."
    )

    molecule_encoder = (
        build_molecule_encoder()
    )

    stage8_checkpoint = torch.load(
        STAGE8_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    molecule_encoder.load_state_dict(
        stage8_checkpoint[
            "molecule_encoder_state_dict"
        ],
        strict=True,
    )

    molecule_encoder = (
        molecule_encoder
        .to(
            device
        )
        .eval()
    )

    print(
        "Molecule encoder loaded."
    )

    # =====================================================
    # Molecule embeddings
    # =====================================================

    print(
        "\nEncoding molecular fingerprints..."
    )

    fingerprints = (
        dataset.fingerprints
    )

    molecule_parts = []

    encode_start = (
        time.perf_counter()
    )

    with torch.inference_mode():

        for start in range(
            0,
            len(
                fingerprints
            ),
            MOLECULE_BATCH_SIZE,
        ):

            end = min(
                start
                + MOLECULE_BATCH_SIZE,
                len(
                    fingerprints
                ),
            )

            fp_numpy = np.array(
                fingerprints[
                    start:end
                ],
                dtype=np.uint8,
                copy=True,
            )

            fp_tensor = (
                torch.from_numpy(
                    fp_numpy
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                embedding = (
                    molecule_encoder(
                        fp_tensor
                    )
                )

            molecule_parts.append(
                embedding.float()
            )

    molecule_embeddings = (
        torch.cat(
            molecule_parts,
            dim=0,
        )
    )

    torch.cuda.synchronize()

    print(
        f"Molecule embeddings: "
        f"{tuple(molecule_embeddings.shape)}"
    )

    print(
        f"Elapsed: "
        f"{time.perf_counter() - encode_start:.2f}s"
    )

    # =====================================================
    # Spectrum embeddings
    # =====================================================

    print(
        "\nEncoding validation spectra..."
    )

    spectrum_parts = []

    spectrum_start = (
        time.perf_counter()
    )

    with torch.inference_mode():

        for batch_number, batch in enumerate(
            loader,
            start=1,
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
                dtype=torch.bfloat16,
            ):

                embedding = (
                    spectrum_encoder(
                        peaks,
                        mask,
                        precursor,
                    )
                )

            spectrum_parts.append(
                embedding
                .float()
                .cpu()
                .clone()
            )

            if (
                batch_number
                % 50
                == 0
            ):

                processed = min(
                    batch_number
                    * SPECTRUM_BATCH_SIZE,
                    len(
                        dataset
                    ),
                )

                print(
                    f"Encoded "
                    f"{processed:,}/"
                    f"{len(dataset):,}"
                )

    spectrum_embeddings = (
        torch.cat(
            spectrum_parts,
            dim=0,
        )
    )

    print(
        f"Spectrum embeddings: "
        f"{tuple(spectrum_embeddings.shape)}"
    )

    print(
        f"Elapsed: "
        f"{time.perf_counter() - spectrum_start:.2f}s"
    )

    # =====================================================
    # Candidate sets
    # =====================================================

    print(
        "\nGenerating per-spectrum candidate sets..."
    )

    spectrum_candidate_sets = {}

    eligible_indices = []

    survival_count = 0

    generation_start = (
        time.perf_counter()
    )

    for row_counter, row in enumerate(
        timstof_metadata.itertuples(
            index=False
        ),
        start=1,
    ):

        validation_index = int(
            row.validation_index
        )

        adduct = (
            ""
            if pd.isna(
                row.adduct
            )
            else str(
                row.adduct
            )
        )

        supported = (
            is_supported_adduct(
                adduct
            )
        )

        trusted = (
            is_trusted_for_mass_filter(
                adduct
            )
            if supported
            else False
        )

        decision = (
            get_mass_filter_decision(
                adduct_supported=(
                    supported
                ),
                adduct_trusted=(
                    trusted
                ),
                instrument_type=str(
                    row.instrument_type
                ),
                ingest_lib=str(
                    row.ingest_lib
                ),
                inference_mode=False,
            )
        )

        if not decision.hard_filter:

            continue

        generator = get_generator(
            decision.tolerance_ppm
        )

        generation = (
            generator.generate(
                precursor_mz=float(
                    row.precursor_mz
                ),
                adduct=adduct,
                require_trusted=True,
            )
        )

        candidate_indices = []

        for hit in (
            generation.candidates
        ):

            structure_index = (
                inchikey_to_structure_index
                .get(
                    str(
                        hit.inchikey14
                    )
                )
            )

            if structure_index is not None:

                candidate_indices.append(
                    int(
                        structure_index
                    )
                )

        if not candidate_indices:

            continue

        candidate_indices = (
            np.unique(
                np.asarray(
                    candidate_indices,
                    dtype=np.int64,
                )
            )
        )

        spectrum_candidate_sets[
            validation_index
        ] = candidate_indices

        eligible_indices.append(
            validation_index
        )

        true_index = (
            inchikey_to_structure_index
            .get(
                str(
                    row.inchikey14
                )
            )
        )

        if (
            true_index is not None
            and true_index
            in candidate_indices
        ):

            survival_count += 1

        if (
            row_counter
            % 10_000
            == 0
        ):

            print(
                f"Processed "
                f"{row_counter:,}/"
                f"{len(timstof_metadata):,}"
            )

    print(
        f"\nEligible timsTOF spectra: "
        f"{len(eligible_indices):,}"
    )

    print(
        f"True candidate survived: "
        f"{survival_count:,}/"
        f"{len(eligible_indices):,}"
    )

    print(
        f"Candidate generation elapsed: "
        f"{time.perf_counter() - generation_start:.2f}s"
    )

    # =====================================================
    # Eligible groups
    # =====================================================

    eligible_set = set(
        eligible_indices
    )

    eligible_metadata = (
        timstof_metadata[
            timstof_metadata[
                "validation_index"
            ]
            .isin(
                eligible_set
            )
        ]
        .copy()
    )

    groups = list(
        eligible_metadata.groupby(
            [
                "inchikey14",
                "ingest_lib",
            ],
            sort=False,
            dropna=False,
        )
    )

    print(
        f"\nValidation molecular groups: "
        f"{len(groups):,}"
    )

    validation_group_sizes = np.asarray(
        [
            len(
                group_df
            )
            for _, group_df
            in groups
        ],
        dtype=np.int64,
    )

    print(
        f"Validation original mean: "
        f"{validation_group_sizes.mean():.2f}"
    )

    print(
        f"Validation original median: "
        f"{np.median(validation_group_sizes):.1f}"
    )

    # =====================================================
    # Repeated test-like evaluation
    # =====================================================

    print(
        "\n"
        + "#" * 90
    )

    print(
        "TEST-LIKE REPEATED SUBSAMPLING"
    )

    print(
        "#" * 90
    )

    repeat_rows = []

    group_result_rows = []

    for repeat_number, seed in enumerate(
        REPEAT_SEEDS,
        start=1,
    ):

        print(
            "\n"
            + "=" * 90
        )

        print(
            f"REPEAT {repeat_number}/"
            f"{len(REPEAT_SEEDS)} "
            f"| seed={seed}"
        )

        print(
            "=" * 90
        )

        rng = np.random.default_rng(
            seed
        )

        ranks_by_mode = {
            "union": [],
            "intersection": [],
        }

        actual_sample_sizes = []

        target_sample_sizes = []

        capped_count = 0

        repeat_start = (
            time.perf_counter()
        )

        for group_number, (
            group_key,
            group_df,
        ) in enumerate(
            groups,
            start=1,
        ):

            true_inchikey = str(
                group_key[
                    0
                ]
            )

            true_structure_index = (
                inchikey_to_structure_index
                .get(
                    true_inchikey
                )
            )

            if true_structure_index is None:

                continue

            available_indices = (
                group_df[
                    "validation_index"
                ]
                .astype(
                    np.int64
                )
                .to_numpy()
            )

            available_count = len(
                available_indices
            )

            # ---------------------------------------------
            # Draw target size directly from empirical
            # Kaggle test distribution.
            # ---------------------------------------------

            target_count = int(
                rng.choice(
                    test_count_values,
                    p=(
                        test_count_probabilities
                    ),
                )
            )

            target_sample_sizes.append(
                target_count
            )

            actual_count = min(
                target_count,
                available_count,
            )

            if (
                target_count
                > available_count
            ):

                capped_count += 1

            actual_sample_sizes.append(
                actual_count
            )

            # ---------------------------------------------
            # Randomly choose spectra without replacement.
            # ---------------------------------------------

            if (
                actual_count
                == available_count
            ):

                selected_indices = (
                    available_indices.copy()
                )

            else:

                selected_indices = (
                    rng.choice(
                        available_indices,
                        size=actual_count,
                        replace=False,
                    )
                )

            selected_indices = (
                np.asarray(
                    selected_indices,
                    dtype=np.int64,
                )
            )

            candidate_sets = [
                spectrum_candidate_sets[
                    int(
                        index
                    )
                ]
                for index
                in selected_indices
            ]

            # ---------------------------------------------
            # UNION
            # ---------------------------------------------

            union_candidates = (
                np.unique(
                    np.concatenate(
                        candidate_sets
                    )
                )
            )

            # ---------------------------------------------
            # INTERSECTION
            # ---------------------------------------------

            intersection_set = set(
                candidate_sets[
                    0
                ].tolist()
            )

            for candidate_set in (
                candidate_sets[
                    1:
                ]
            ):

                intersection_set.intersection_update(
                    candidate_set.tolist()
                )

            intersection_candidates = (
                np.asarray(
                    sorted(
                        intersection_set
                    ),
                    dtype=np.int64,
                )
            )

            # ---------------------------------------------
            # Spectrum embeddings
            #
            # clone() makes the tensor writable and avoids
            # the previous warning.
            # ---------------------------------------------

            spectrum_indices_tensor = (
                torch.tensor(
                    selected_indices,
                    dtype=torch.long,
                )
            )

            group_spectrum_embeddings = (
                spectrum_embeddings[
                    spectrum_indices_tensor
                ]
                .clone()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            candidate_modes = {
                "union": (
                    union_candidates
                ),
                "intersection": (
                    intersection_candidates
                ),
            }

            num_adducts = int(
                group_df[
                    "adduct"
                ]
                .fillna(
                    "<missing>"
                )
                .astype(str)
                .nunique()
            )

            for mode_name, candidate_indices in (
                candidate_modes.items()
            ):

                if len(
                    candidate_indices
                ) == 0:

                    rank = np.nan

                else:

                    candidate_tensor = (
                        torch.tensor(
                            candidate_indices,
                            dtype=torch.long,
                            device=device,
                        )
                    )

                    candidate_embeddings = (
                        molecule_embeddings[
                            candidate_tensor
                        ]
                    )

                    # [spectra, candidates]
                    score_matrix = (
                        group_spectrum_embeddings
                        @ candidate_embeddings.T
                    )

                    # Current Stage 9 winner:
                    # MEAN aggregation
                    aggregated_scores = (
                        score_matrix.mean(
                            dim=0
                        )
                    )

                    rank = (
                        rank_true_candidate(
                            scores=(
                                aggregated_scores
                            ),
                            candidate_indices=(
                                candidate_indices
                            ),
                            true_structure_index=(
                                true_structure_index
                            ),
                        )
                    )

                ranks_by_mode[
                    mode_name
                ].append(
                    rank
                )

                group_result_rows.append(
                    {
                        "repeat": (
                            repeat_number
                        ),
                        "seed": (
                            seed
                        ),
                        "inchikey14": (
                            true_inchikey
                        ),
                        "ingest_lib": (
                            str(
                                group_key[
                                    1
                                ]
                            )
                        ),
                        "candidate_mode": (
                            mode_name
                        ),
                        "available_spectra": (
                            available_count
                        ),
                        "target_spectra": (
                            target_count
                        ),
                        "sampled_spectra": (
                            actual_count
                        ),
                        "num_adducts": (
                            num_adducts
                        ),
                        "candidate_count": (
                            len(
                                candidate_indices
                            )
                        ),
                        "rank": (
                            rank
                        ),
                    }
                )

            if (
                group_number
                % 5_000
                == 0
            ):

                print(
                    f"Processed "
                    f"{group_number:,}/"
                    f"{len(groups):,}"
                )

        # =================================================
        # Repeat summary
        # =================================================

        actual_sample_sizes = np.asarray(
            actual_sample_sizes,
            dtype=np.float64,
        )

        target_sample_sizes = np.asarray(
            target_sample_sizes,
            dtype=np.float64,
        )

        print(
            f"\nRequested mean spectra/group: "
            f"{target_sample_sizes.mean():.3f}"
        )

        print(
            f"Actual mean spectra/group: "
            f"{actual_sample_sizes.mean():.3f}"
        )

        print(
            f"Actual median: "
            f"{np.median(actual_sample_sizes):.1f}"
        )

        print(
            f"Capped groups: "
            f"{capped_count:,}/"
            f"{len(groups):,} "
            f"({capped_count / len(groups) * 100:.2f}%)"
        )

        for mode_name in [
            "union",
            "intersection",
        ]:

            metrics = (
                calculate_metrics(
                    ranks_by_mode[
                        mode_name
                    ]
                )
            )

            print_metrics(
                (
                    f"REPEAT {repeat_number} "
                    f"| {mode_name.upper()} "
                    f"+ MEAN"
                ),
                metrics,
            )

            repeat_rows.append(
                {
                    "repeat": (
                        repeat_number
                    ),
                    "seed": (
                        seed
                    ),
                    "candidate_mode": (
                        mode_name
                    ),
                    "groups": (
                        metrics[
                            "count"
                        ]
                    ),
                    "coverage": (
                        metrics[
                            "coverage"
                        ]
                    ),
                    "recall_1": (
                        metrics[
                            "recall_1"
                        ]
                    ),
                    "recall_5": (
                        metrics[
                            "recall_5"
                        ]
                    ),
                    "recall_10": (
                        metrics[
                            "recall_10"
                        ]
                    ),
                    "recall_25": (
                        metrics[
                            "recall_25"
                        ]
                    ),
                    "mrr": (
                        metrics[
                            "mrr"
                        ]
                    ),
                    "median_rank": (
                        metrics[
                            "median_rank"
                        ]
                    ),
                    "mean_rank": (
                        metrics[
                            "mean_rank"
                        ]
                    ),
                    "requested_mean_spectra": (
                        target_sample_sizes.mean()
                    ),
                    "actual_mean_spectra": (
                        actual_sample_sizes.mean()
                    ),
                    "capped_groups": (
                        capped_count
                    ),
                }
            )

        print(
            f"\nRepeat elapsed: "
            f"{time.perf_counter() - repeat_start:.2f}s"
        )

    # =====================================================
    # Save detailed results
    # =====================================================

    repeat_df = pd.DataFrame(
        repeat_rows
    )

    repeat_df.to_csv(
        OUTPUT_REPEAT_RESULTS,
        index=False,
    )

    group_df = pd.DataFrame(
        group_result_rows
    )

    group_df.to_csv(
        OUTPUT_GROUP_RESULTS,
        index=False,
    )

    # =====================================================
    # Aggregate across repeated seeds
    # =====================================================

    print(
        "\n"
        + "#" * 90
    )

    print(
        "FINAL TEST-LIKE SUBSAMPLING RESULTS"
    )

    print(
        "#" * 90
    )

    summary_lines = [
        (
            "ENVEDA CASMI 2026 — "
            "STAGE 9 TEST-LIKE SUBSAMPLING"
        ),
        "",
        (
            f"Repeats: "
            f"{len(REPEAT_SEEDS)}"
        ),
        (
            f"Seeds: "
            f"{REPEAT_SEEDS}"
        ),
        (
            f"Aggregation: "
            f"{AGGREGATION_METHOD}"
        ),
        "",
    ]

    comparison = {}

    for mode_name in [
        "union",
        "intersection",
    ]:

        subset = repeat_df[
            repeat_df[
                "candidate_mode"
            ]
            == mode_name
        ]

        print(
            "\n"
            + "=" * 90
        )

        print(
            f"{mode_name.upper()} + MEAN"
        )

        print(
            "=" * 90
        )

        metrics_to_report = [
            "coverage",
            "recall_1",
            "recall_5",
            "recall_10",
            "recall_25",
            "mrr",
        ]

        summary_lines.append(
            f"{mode_name.upper()} + MEAN"
        )

        for metric_name in (
            metrics_to_report
        ):

            values = (
                subset[
                    metric_name
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            mean_value = float(
                np.mean(
                    values
                )
            )

            std_value = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

            min_value = float(
                np.min(
                    values
                )
            )

            max_value = float(
                np.max(
                    values
                )
            )

            if metric_name in {
                "coverage",
                "recall_1",
                "recall_5",
                "recall_10",
                "recall_25",
            }:

                print(
                    f"{metric_name:>10}: "
                    f"{mean_value * 100:8.3f}% "
                    f"± "
                    f"{std_value * 100:.3f}% "
                    f"["
                    f"{min_value * 100:.3f}, "
                    f"{max_value * 100:.3f}"
                    f"]"
                )

            else:

                print(
                    f"{metric_name:>10}: "
                    f"{mean_value:.6f} "
                    f"± "
                    f"{std_value:.6f} "
                    f"["
                    f"{min_value:.6f}, "
                    f"{max_value:.6f}"
                    f"]"
                )

            summary_lines.append(
                (
                    f"{metric_name}: "
                    f"mean={mean_value:.8f}, "
                    f"std={std_value:.8f}, "
                    f"min={min_value:.8f}, "
                    f"max={max_value:.8f}"
                )
            )

        comparison[
            mode_name
        ] = {
            "mrr_mean": float(
                subset[
                    "mrr"
                ].mean()
            ),
            "mrr_std": float(
                subset[
                    "mrr"
                ].std(
                    ddof=1
                )
            ),
        }

        actual_mean = float(
            subset[
                "actual_mean_spectra"
            ].mean()
        )

        requested_mean = float(
            subset[
                "requested_mean_spectra"
            ].mean()
        )

        print(
            f"\nRequested spectra/group: "
            f"{requested_mean:.3f}"
        )

        print(
            f"Actual spectra/group: "
            f"{actual_mean:.3f}"
        )

        summary_lines.append(
            f"Requested mean spectra/group: "
            f"{requested_mean:.6f}"
        )

        summary_lines.append(
            f"Actual mean spectra/group: "
            f"{actual_mean:.6f}"
        )

        summary_lines.append("")

    # =====================================================
    # Direct comparison
    # =====================================================

    union_mrr = comparison[
        "union"
    ][
        "mrr_mean"
    ]

    intersection_mrr = comparison[
        "intersection"
    ][
        "mrr_mean"
    ]

    difference = (
        intersection_mrr
        - union_mrr
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "UNION VS INTERSECTION"
    )

    print(
        "=" * 90
    )

    print(
        f"UNION mean MRR:        "
        f"{union_mrr:.6f}"
    )

    print(
        f"INTERSECTION mean MRR: "
        f"{intersection_mrr:.6f}"
    )

    print(
        f"Difference:            "
        f"{difference:+.6f}"
    )

    if intersection_mrr > union_mrr:

        winner = (
            "intersection"
        )

    elif union_mrr > intersection_mrr:

        winner = (
            "union"
        )

    else:

        winner = (
            "tie"
        )

    print(
        f"Winner:                "
        f"{winner.upper()}"
    )

    summary_lines.extend(
        [
            (
                f"Union mean MRR: "
                f"{union_mrr:.8f}"
            ),
            (
                f"Intersection mean MRR: "
                f"{intersection_mrr:.8f}"
            ),
            (
                f"Intersection - Union: "
                f"{difference:+.8f}"
            ),
            (
                f"Winner: "
                f"{winner}"
            ),
        ]
    )

    # =====================================================
    # Adduct-count analysis under test-like sampling
    # =====================================================

    print(
        "\n"
        + "#" * 90
    )

    print(
        "TEST-LIKE BREAKDOWN BY ADDUCT COUNT"
    )

    print(
        "#" * 90
    )

    for mode_name in [
        "union",
        "intersection",
    ]:

        print(
            f"\n{mode_name.upper()} + MEAN"
        )

        mode_groups = (
            group_df[
                group_df[
                    "candidate_mode"
                ]
                == mode_name
            ]
        )

        for num_adducts in sorted(
            mode_groups[
                "num_adducts"
            ]
            .unique()
        ):

            subset = (
                mode_groups[
                    mode_groups[
                        "num_adducts"
                    ]
                    == num_adducts
                ]
            )

            per_repeat_mrr = []

            for repeat in sorted(
                subset[
                    "repeat"
                ]
                .unique()
            ):

                repeat_subset = (
                    subset[
                        subset[
                            "repeat"
                        ]
                        == repeat
                    ]
                )

                metrics = (
                    calculate_metrics(
                        repeat_subset[
                            "rank"
                        ]
                        .to_numpy()
                    )
                )

                per_repeat_mrr.append(
                    metrics[
                        "mrr"
                    ]
                )

            print(
                f"{int(num_adducts)} adduct(s): "
                f"groups/repeat≈"
                f"{len(subset) / len(REPEAT_SEEDS):.0f}, "
                f"MRR="
                f"{np.mean(per_repeat_mrr):.6f} "
                f"± "
                f"{np.std(per_repeat_mrr, ddof=1):.6f}"
            )

    # =====================================================
    # Final timing and outputs
    # =====================================================

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "STAGE 9.4 COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        f"Total elapsed: "
        f"{total_elapsed:.2f}s "
        f"({total_elapsed / 60:.2f} min)"
    )

    print(
        "\nRepeat summary CSV:"
    )

    print(
        OUTPUT_REPEAT_RESULTS
    )

    print(
        "\nDetailed group CSV:"
    )

    print(
        OUTPUT_GROUP_RESULTS
    )

    print(
        "\nText summary:"
    )

    print(
        OUTPUT_SUMMARY
    )

    summary_lines.extend(
        [
            "",
            (
                f"Total elapsed seconds: "
                f"{total_elapsed:.2f}"
            ),
        ]
    )

    OUTPUT_SUMMARY.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()