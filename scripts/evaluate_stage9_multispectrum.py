from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = DATASET_DIR / "train.parquet"

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

OUTPUT_RESULTS = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage9_multispectrum_results.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage9_multispectrum_summary.txt"
)


# =========================================================
# Configuration
# =========================================================

SPECTRUM_BATCH_SIZE = 512
MOLECULE_BATCH_SIZE = 4096

NUM_WORKERS = 0

EMBEDDING_DIM = 128

LOGSUMEXP_TEMPERATURE = 0.10

TOP_K = 25


# =========================================================
# Model builders
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

def calculate_metrics(ranks):
    """
    ranks can contain NaN for candidate-generation failures.

    Missing true structures receive zero reciprocal-rank credit,
    which is closer to end-to-end competition evaluation.
    """

    ranks = np.asarray(
        ranks,
        dtype=np.float64,
    )

    total = len(ranks)

    if total == 0:
        return None

    finite = np.isfinite(ranks)

    successful_ranks = ranks[finite]

    reciprocal = np.zeros(
        total,
        dtype=np.float64,
    )

    reciprocal[finite] = (
        1.0 / successful_ranks
    )

    def recall_at(k):
        return float(
            np.mean(
                finite
                & (ranks <= k)
            )
        )

    if len(successful_ranks) > 0:
        median_rank = float(
            np.median(successful_ranks)
        )

        mean_rank = float(
            np.mean(successful_ranks)
        )
    else:
        median_rank = np.nan
        mean_rank = np.nan

    return {
        "count": int(total),
        "coverage": float(
            np.mean(finite)
        ),
        "recall_1": recall_at(1),
        "recall_5": recall_at(5),
        "recall_10": recall_at(10),
        "recall_25": recall_at(25),
        "mrr": float(
            np.mean(reciprocal)
        ),
        "median_rank_conditional": (
            median_rank
        ),
        "mean_rank_conditional": (
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

    print(title)

    print(
        "=" * 90
    )

    if metrics is None:
        print("No groups.")
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
        f"Median rank*: "
        f"{metrics['median_rank_conditional']:.1f}"
    )

    print(
        f"Mean rank*:   "
        f"{metrics['mean_rank_conditional']:.2f}"
    )

    print(
        "* rank statistics conditional on "
        "true candidate being present"
    )


# =========================================================
# Aggregation
# =========================================================

def aggregate_scores(
    score_matrix: torch.Tensor,
    method: str,
):
    """
    score_matrix:
        [num_spectra, num_candidates]
    """

    if score_matrix.ndim != 2:
        raise ValueError(
            "score_matrix must be 2D."
        )

    if method == "mean":
        return score_matrix.mean(
            dim=0
        )

    if method == "max":
        return score_matrix.max(
            dim=0
        ).values

    if method == "top2_mean":
        k = min(
            2,
            score_matrix.shape[0],
        )

        values = torch.topk(
            score_matrix,
            k=k,
            dim=0,
        ).values

        return values.mean(
            dim=0
        )

    if method == "logsumexp":
        temperature = (
            LOGSUMEXP_TEMPERATURE
        )

        return (
            torch.logsumexp(
                score_matrix
                / temperature,
                dim=0,
            )
            * temperature
        )

    raise ValueError(
        f"Unknown aggregation method: {method}"
    )


AGGREGATION_METHODS = [
    "mean",
    "max",
    "top2_mean",
    "logsumexp",
]


# =========================================================
# Ranking helper
# =========================================================

def rank_true_candidate(
    scores,
    candidate_indices,
    true_structure_index,
):
    candidate_indices = np.asarray(
        candidate_indices,
        dtype=np.int64,
    )

    locations = np.flatnonzero(
        candidate_indices
        == true_structure_index
    )

    if len(locations) == 0:
        return np.nan

    true_position = int(
        locations[0]
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

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 9 MULTI-SPECTRUM AGGREGATION"
    )

    print(
        f"\nGPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        "\nStage 8 checkpoint:"
    )

    print(
        STAGE8_CHECKPOINT
    )

    total_start = time.perf_counter()

    # =====================================================
    # Validate files
    # =====================================================

    required_paths = [
        TRAIN_PATH,
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
    # Dataset
    # =====================================================

    print(
        "\nLoading Stage 8 validation dataset..."
    )

    dataset = SpectrumMoleculeDataset(
        stage6_dir=STAGE6_DIR,
        stage8_dir=STAGE8_DIR,
        split_value=1,
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
        "\nLoading Stage 8 structure map..."
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
        .astype(np.int64)
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
    # Candidate mass index
    # =====================================================

    print(
        "\nLoading Stage 7 candidate index..."
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
    # Original metadata
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
            "collision_energy_ev",
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
        len(metadata),
        dtype=np.int64,
    )

    print(
        f"Mapped metadata rows: "
        f"{len(metadata):,}"
    )

    # =====================================================
    # Filter to timsTOF
    # =====================================================

    timstof_mask = (
        metadata[
            "instrument_type"
        ]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("timstof")
    )

    timstof_metadata = (
        metadata[
            timstof_mask
        ]
        .copy()
    )

    print(
        f"timsTOF validation spectra: "
        f"{len(timstof_metadata):,}"
    )

    # =====================================================
    # Load models
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
        .to(device)
        .eval()
    )

    print(
        "Stage 6 spectrum encoder loaded."
    )

    print(
        "\nLoading candidate-aware "
        "molecule encoder..."
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
        .to(device)
        .eval()
    )

    print(
        "Candidate-aware molecule encoder loaded."
    )

    # =====================================================
    # Encode all molecule fingerprints
    # =====================================================

    print(
        "\nEncoding molecular fingerprints..."
    )

    fingerprint_array = (
        dataset.fingerprints
    )

    molecule_embedding_parts = []

    encode_start = time.perf_counter()

    with torch.inference_mode():

        for start in range(
            0,
            len(fingerprint_array),
            MOLECULE_BATCH_SIZE,
        ):

            end = min(
                start
                + MOLECULE_BATCH_SIZE,
                len(fingerprint_array),
            )

            # copy=True avoids the non-writable NumPy warning
            fingerprint_numpy = np.array(
                fingerprint_array[
                    start:end
                ],
                dtype=np.uint8,
                copy=True,
            )

            fingerprint_tensor = (
                torch.from_numpy(
                    fingerprint_numpy
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

                embeddings = (
                    molecule_encoder(
                        fingerprint_tensor
                    )
                )

            molecule_embedding_parts.append(
                embeddings.float()
            )

    molecule_embeddings = torch.cat(
        molecule_embedding_parts,
        dim=0,
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
    # Encode all validation spectra
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

                embeddings = (
                    spectrum_encoder(
                        peaks,
                        mask,
                        precursor,
                    )
                )

            spectrum_parts.append(
                embeddings.float().cpu()
            )

            if batch_number % 50 == 0:

                processed = min(
                    batch_number
                    * SPECTRUM_BATCH_SIZE,
                    len(dataset),
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

    torch.cuda.synchronize()

    print(
        f"Spectrum embeddings: "
        f"{tuple(spectrum_embeddings.shape)}"
    )

    print(
        f"Elapsed: "
        f"{time.perf_counter() - spectrum_start:.2f}s"
    )

    # =====================================================
    # Precompute per-spectrum candidate sets
    # =====================================================

    print(
        "\nGenerating per-spectrum "
        "candidate sets..."
    )

    spectrum_candidate_sets = {}

    eligible_indices = []

    true_survival_spectra = 0

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
            if pd.isna(row.adduct)
            else str(row.adduct)
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
                instrument_type=(
                    str(
                        row.instrument_type
                    )
                ),
                ingest_lib=(
                    str(
                        row.ingest_lib
                    )
                ),
                inference_mode=False,
            )
        )

        if not decision.hard_filter:
            continue

        precursor_mz = float(
            row.precursor_mz
        )

        generator = get_generator(
            decision.tolerance_ppm
        )

        generation = generator.generate(
            precursor_mz=precursor_mz,
            adduct=adduct,
            require_trusted=True,
        )

        candidate_indices = []

        for hit in generation.candidates:

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
                    int(structure_index)
                )

        if not candidate_indices:
            continue

        candidate_indices = np.unique(
            np.asarray(
                candidate_indices,
                dtype=np.int64,
            )
        )

        spectrum_candidate_sets[
            validation_index
        ] = candidate_indices

        eligible_indices.append(
            validation_index
        )

        true_structure_index = (
            inchikey_to_structure_index
            .get(
                str(
                    row.inchikey14
                )
            )
        )

        if (
            true_structure_index
            is not None
            and true_structure_index
            in candidate_indices
        ):
            true_survival_spectra += 1

        if row_counter % 10_000 == 0:

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
        f"{true_survival_spectra:,}/"
        f"{len(eligible_indices):,}"
    )

    print(
        f"Candidate generation elapsed: "
        f"{time.perf_counter() - generation_start:.2f}s"
    )

    # =====================================================
    # Restrict metadata to eligible spectra
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

    # =====================================================
    # Group definition
    #
    # Same structure + same ingest library.
    #
    # DO NOT split by adduct because ~27% of test molecules
    # contain multiple adducts.
    # =====================================================

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
        f"\nMulti-spectrum groups: "
        f"{len(groups):,}"
    )

    group_sizes = np.asarray(
        [
            len(group)
            for _, group
            in groups
        ],
        dtype=np.int64,
    )

    print(
        f"Mean spectra/group: "
        f"{group_sizes.mean():.2f}"
    )

    print(
        f"Median spectra/group: "
        f"{np.median(group_sizes):.1f}"
    )

    # =====================================================
    # Evaluate multi-spectrum aggregation
    # =====================================================

    print(
        "\nEvaluating multi-spectrum aggregation..."
    )

    result_rows = []

    eval_start = (
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
            group_key[0]
        )

        ingest_lib = str(
            group_key[1]
        )

        true_structure_index = (
            inchikey_to_structure_index
            .get(
                true_inchikey
            )
        )

        if true_structure_index is None:
            continue

        spectrum_indices = (
            group_df[
                "validation_index"
            ]
            .astype(np.int64)
            .to_numpy()
        )

        candidate_sets = [
            spectrum_candidate_sets[
                int(index)
            ]
            for index
            in spectrum_indices
        ]

        # -------------------------------------------------
        # Candidate UNION
        #
        # Used as the main Stage 9 candidate universe.
        # -------------------------------------------------

        union_candidates = np.unique(
            np.concatenate(
                candidate_sets
            )
        )

        # -------------------------------------------------
        # Candidate INTERSECTION
        #
        # Diagnostic chemistry-consensus universe.
        # -------------------------------------------------

        intersection_set = set(
            candidate_sets[0].tolist()
        )

        for candidate_set in (
            candidate_sets[1:]
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

        # -------------------------------------------------
        # Group metadata
        # -------------------------------------------------

        adduct_values = sorted(
            group_df[
                "adduct"
            ]
            .fillna(
                "<missing>"
            )
            .astype(str)
            .unique()
            .tolist()
        )

        num_adducts = len(
            adduct_values
        )

        num_spectra = len(
            spectrum_indices
        )

        # -------------------------------------------------
        # Spectrum embeddings
        # -------------------------------------------------

        group_spectrum_embeddings = (
            spectrum_embeddings[
                spectrum_indices
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        # -------------------------------------------------
        # Evaluate function
        # -------------------------------------------------

        candidate_modes = {
            "union": (
                union_candidates
            ),
            "intersection": (
                intersection_candidates
            ),
        }

        for candidate_mode, candidate_indices in (
            candidate_modes.items()
        ):

            if len(candidate_indices) == 0:

                for method in (
                    AGGREGATION_METHODS
                ):

                    result_rows.append(
                        {
                            "inchikey14": (
                                true_inchikey
                            ),
                            "ingest_lib": (
                                ingest_lib
                            ),
                            "num_spectra": (
                                num_spectra
                            ),
                            "num_adducts": (
                                num_adducts
                            ),
                            "adducts": (
                                "|".join(
                                    adduct_values
                                )
                            ),
                            "candidate_mode": (
                                candidate_mode
                            ),
                            "aggregation": (
                                method
                            ),
                            "candidate_count": 0,
                            "true_survived": False,
                            "rank": np.nan,
                        }
                    )

                continue

            candidate_tensor = torch.tensor(
                candidate_indices,
                dtype=torch.long,
                device=device,
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

            true_survived = bool(
                np.any(
                    candidate_indices
                    == true_structure_index
                )
            )

            for method in (
                AGGREGATION_METHODS
            ):

                aggregated = (
                    aggregate_scores(
                        score_matrix,
                        method,
                    )
                )

                rank = (
                    rank_true_candidate(
                        scores=aggregated,
                        candidate_indices=(
                            candidate_indices
                        ),
                        true_structure_index=(
                            true_structure_index
                        ),
                    )
                )

                result_rows.append(
                    {
                        "inchikey14": (
                            true_inchikey
                        ),
                        "ingest_lib": (
                            ingest_lib
                        ),
                        "num_spectra": (
                            num_spectra
                        ),
                        "num_adducts": (
                            num_adducts
                        ),
                        "adducts": (
                            "|".join(
                                adduct_values
                            )
                        ),
                        "candidate_mode": (
                            candidate_mode
                        ),
                        "aggregation": (
                            method
                        ),
                        "candidate_count": (
                            len(
                                candidate_indices
                            )
                        ),
                        "true_survived": (
                            true_survived
                        ),
                        "rank": (
                            rank
                        ),
                    }
                )

        if group_number % 2_000 == 0:

            print(
                f"Processed "
                f"{group_number:,}/"
                f"{len(groups):,} groups"
            )

    results = pd.DataFrame(
        result_rows
    )

    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    evaluation_elapsed = (
        time.perf_counter()
        - eval_start
    )

    # =====================================================
    # Overall results
    # =====================================================

    summary_lines = []

    print(
        "\n"
        + "#" * 90
    )

    print(
        "STAGE 9 MOLECULE-LEVEL RESULTS"
    )

    print(
        "#" * 90
    )

    best_method = None
    best_mrr = -1.0

    for candidate_mode in [
        "union",
        "intersection",
    ]:

        print(
            "\n"
            + "#" * 90
        )

        print(
            f"CANDIDATE MODE: "
            f"{candidate_mode.upper()}"
        )

        print(
            "#" * 90
        )

        summary_lines.append(
            f"Candidate mode: "
            f"{candidate_mode}"
        )

        for method in (
            AGGREGATION_METHODS
        ):

            subset = results[
                (
                    results[
                        "candidate_mode"
                    ]
                    == candidate_mode
                )
                &
                (
                    results[
                        "aggregation"
                    ]
                    == method
                )
            ]

            metrics = (
                calculate_metrics(
                    subset[
                        "rank"
                    ]
                    .to_numpy()
                )
            )

            print_metrics(
                f"{candidate_mode.upper()} "
                f"+ {method.upper()}",
                metrics,
            )

            if metrics is not None:

                summary_lines.extend(
                    [
                        (
                            f"{method}:"
                        ),
                        (
                            f"  groups="
                            f"{metrics['count']}"
                        ),
                        (
                            f"  coverage="
                            f"{metrics['coverage']:.8f}"
                        ),
                        (
                            f"  recall@1="
                            f"{metrics['recall_1']:.8f}"
                        ),
                        (
                            f"  recall@5="
                            f"{metrics['recall_5']:.8f}"
                        ),
                        (
                            f"  recall@10="
                            f"{metrics['recall_10']:.8f}"
                        ),
                        (
                            f"  recall@25="
                            f"{metrics['recall_25']:.8f}"
                        ),
                        (
                            f"  mrr="
                            f"{metrics['mrr']:.8f}"
                        ),
                    ]
                )

                if (
                    candidate_mode
                    == "union"
                    and metrics[
                        "mrr"
                    ]
                    > best_mrr
                ):
                    best_mrr = (
                        metrics[
                            "mrr"
                        ]
                    )

                    best_method = (
                        method
                    )

        summary_lines.append("")

    # =====================================================
    # Adduct-count breakdown
    # =====================================================

    print(
        "\n"
        + "#" * 90
    )

    print(
        "BREAKDOWN BY NUMBER OF ADDUCTS"
    )

    print(
        "#" * 90
    )

    print(
        f"\nUsing main candidate mode = UNION "
        f"and best aggregation = "
        f"{best_method}"
    )

    best_subset = results[
        (
            results[
                "candidate_mode"
            ]
            == "union"
        )
        &
        (
            results[
                "aggregation"
            ]
            == best_method
        )
    ].copy()

    summary_lines.append(
        f"Best union aggregation: "
        f"{best_method}"
    )

    summary_lines.append(
        f"Best union MRR: "
        f"{best_mrr:.8f}"
    )

    summary_lines.append("")

    for num_adducts in sorted(
        best_subset[
            "num_adducts"
        ].unique()
    ):

        subset = best_subset[
            best_subset[
                "num_adducts"
            ]
            == num_adducts
        ]

        metrics = calculate_metrics(
            subset[
                "rank"
            ].to_numpy()
        )

        print_metrics(
            f"{int(num_adducts)} ADDUCT(S)",
            metrics,
        )

        if metrics is not None:

            summary_lines.extend(
                [
                    (
                        f"{int(num_adducts)} "
                        f"adduct(s):"
                    ),
                    (
                        f"  groups="
                        f"{metrics['count']}"
                    ),
                    (
                        f"  recall@1="
                        f"{metrics['recall_1']:.8f}"
                    ),
                    (
                        f"  recall@25="
                        f"{metrics['recall_25']:.8f}"
                    ),
                    (
                        f"  mrr="
                        f"{metrics['mrr']:.8f}"
                    ),
                ]
            )

    # =====================================================
    # Per-adduct membership breakdown
    # =====================================================

    print(
        "\n"
        + "#" * 90
    )

    print(
        "GROUPS CONTAINING EACH ADDUCT"
    )

    print(
        "#" * 90
    )

    all_adducts = defaultdict(
        int
    )

    for adduct_string in (
        best_subset[
            "adducts"
        ]
    ):

        for adduct in (
            str(adduct_string)
            .split("|")
        ):

            all_adducts[
                adduct
            ] += 1

    top_adducts = [
        item[0]
        for item
        in sorted(
            all_adducts.items(),
            key=lambda item: (
                -item[1]
            ),
        )[:10]
    ]

    for adduct in top_adducts:

        mask = (
            best_subset[
                "adducts"
            ]
            .astype(str)
            .str.split("|")
            .apply(
                lambda values: (
                    adduct
                    in values
                )
            )
        )

        subset = best_subset[
            mask
        ]

        metrics = calculate_metrics(
            subset[
                "rank"
            ].to_numpy()
        )

        print_metrics(
            f"Contains adduct: "
            f"{adduct}",
            metrics,
        )

    # =====================================================
    # Single-spectrum vs multi-spectrum
    # =====================================================

    print(
        "\n"
        + "#" * 90
    )

    print(
        "SINGLE VS MULTI-SPECTRUM"
    )

    print(
        "#" * 90
    )

    single = best_subset[
        best_subset[
            "num_spectra"
        ]
        == 1
    ]

    multi = best_subset[
        best_subset[
            "num_spectra"
        ]
        >= 2
    ]

    print_metrics(
        "Single-spectrum groups",
        calculate_metrics(
            single[
                "rank"
            ].to_numpy()
        ),
    )

    print_metrics(
        "Multi-spectrum groups",
        calculate_metrics(
            multi[
                "rank"
            ].to_numpy()
        ),
    )

    # =====================================================
    # By number of spectra
    # =====================================================

    print(
        "\n"
        + "#" * 90
    )

    print(
        "BREAKDOWN BY NUMBER OF SPECTRA"
    )

    print(
        "#" * 90
    )

    for count in sorted(
        best_subset[
            "num_spectra"
        ].unique()
    ):

        subset = best_subset[
            best_subset[
                "num_spectra"
            ]
            == count
        ]

        metrics = calculate_metrics(
            subset[
                "rank"
            ].to_numpy()
        )

        print_metrics(
            f"{int(count)} spectrum/spectra",
            metrics,
        )

    # =====================================================
    # Timing
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
        "FINAL STAGE 9 SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        f"Best UNION aggregation: "
        f"{best_method}"
    )

    print(
        f"Best UNION molecule MRR: "
        f"{best_mrr:.6f}"
    )

    print(
        f"Group evaluation elapsed: "
        f"{evaluation_elapsed:.2f}s"
    )

    print(
        f"Total elapsed: "
        f"{total_elapsed:.2f}s"
    )

    print(
        f"\nResults CSV:\n"
        f"{OUTPUT_RESULTS}"
    )

    print(
        f"\nSummary:\n"
        f"{OUTPUT_SUMMARY}"
    )

    summary_lines.extend(
        [
            "",
            (
                f"Evaluation elapsed seconds: "
                f"{evaluation_elapsed:.2f}"
            ),
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