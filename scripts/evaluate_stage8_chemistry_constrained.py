from __future__ import annotations

from collections import defaultdict
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

from casmi.models.molecule_encoder import (
    MoleculeEncoder,
)

from casmi.models.spectrum_encoder import (
    SpectrumEncoder,
)

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

# =========================================================
# IMPORTANT:
# Candidate-aware Stage 8 checkpoint
# =========================================================

STAGE8_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "stage8"
    / "candidate_aware_best.pt"
)

ENVEDA_VARIANTS_PATH = (
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

# =========================================================
# Keep candidate-aware outputs separate from baseline
# =========================================================

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage8_candidate_aware_chemistry_results.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage8_candidate_aware_chemistry_summary.txt"
)


# =========================================================
# Configuration
# =========================================================

BATCH_SIZE = 512

NUM_WORKERS = 0

MOLECULE_BATCH_SIZE = 2048

EMBEDDING_DIM = 128


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
# Spectrum preparation
# =========================================================

def prepare_peaks(
    batch,
    device,
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

    return peaks


# =========================================================
# Statistics helpers
# =========================================================

def describe(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:

        return {
            "min": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

    return {
        "min": float(
            np.min(values)
        ),
        "median": float(
            np.median(values)
        ),
        "mean": float(
            np.mean(values)
        ),
        "p90": float(
            np.percentile(
                values,
                90,
            )
        ),
        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),
        "p99": float(
            np.percentile(
                values,
                99,
            )
        ),
        "max": float(
            np.max(values)
        ),
    }


def calculate_rank_metrics(
    ranks,
):

    ranks = np.asarray(
        ranks,
        dtype=np.float64,
    )

    ranks = ranks[
        np.isfinite(
            ranks
        )
    ]

    if len(ranks) == 0:

        return None

    return {
        "count": int(
            len(ranks)
        ),
        "recall_1": float(
            np.mean(
                ranks <= 1
            )
        ),
        "recall_5": float(
            np.mean(
                ranks <= 5
            )
        ),
        "recall_10": float(
            np.mean(
                ranks <= 10
            )
        ),
        "recall_25": float(
            np.mean(
                ranks <= 25
            )
        ),
        "mrr": float(
            np.mean(
                1.0
                / ranks
            )
        ),
        "median_rank": float(
            np.median(
                ranks
            )
        ),
        "mean_rank": float(
            np.mean(
                ranks
            )
        ),
    }


def print_rank_metrics(
    title,
    metrics,
):

    print(
        "\n"
        + title
    )

    if metrics is None:

        print(
            "No eligible rows."
        )

        return

    print(
        f"Rows:       "
        f"{metrics['count']:,}"
    )

    print(
        f"Recall@1:   "
        f"{metrics['recall_1'] * 100:.3f}%"
    )

    print(
        f"Recall@5:   "
        f"{metrics['recall_5'] * 100:.3f}%"
    )

    print(
        f"Recall@10:  "
        f"{metrics['recall_10'] * 100:.3f}%"
    )

    print(
        f"Recall@25:  "
        f"{metrics['recall_25'] * 100:.3f}%"
    )

    print(
        f"MRR:        "
        f"{metrics['mrr']:.6f}"
    )

    print(
        f"Median rank: "
        f"{metrics['median_rank']:.1f}"
    )

    print(
        f"Mean rank:   "
        f"{metrics['mean_rank']:.2f}"
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

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 8 CANDIDATE-AWARE "
        "CHEMISTRY-CONSTRAINED RETRIEVAL"
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
    # File validation
    # =====================================================

    required_paths = [
        TRAIN_PATH,
        STAGE6_CHECKPOINT,
        STAGE8_CHECKPOINT,
        ENVEDA_VARIANTS_PATH,
        STAGE8_STRUCTURE_MAP,
        STAGE6_ROW_INDICES_PATH,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # =====================================================
    # Validation dataset
    # =====================================================

    print(
        "\nLoading Stage 8 validation dataset..."
    )

    dataset = SpectrumMoleculeDataset(
        stage6_dir=STAGE6_DIR,
        stage8_dir=STAGE8_DIR,
        split_value=1,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"Validation spectra: "
        f"{len(dataset):,}"
    )

    # =====================================================
    # Stage 8 structure map
    # =====================================================

    print(
        "\nLoading Stage 8 structure map..."
    )

    structure_map = pd.read_csv(
        STAGE8_STRUCTURE_MAP
    )

    required_structure_columns = {
        "structure_index",
        "inchikey14",
    }

    missing_structure_columns = (
        required_structure_columns
        - set(
            structure_map.columns
        )
    )

    if missing_structure_columns:

        raise RuntimeError(
            "Stage 8 structure map is missing columns: "
            f"{sorted(missing_structure_columns)}"
        )

    structure_map[
        "inchikey14"
    ] = (
        structure_map[
            "inchikey14"
        ]
        .astype(str)
    )

    structure_map[
        "structure_index"
    ] = (
        structure_map[
            "structure_index"
        ]
        .astype(int)
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
        f"Stage 8 structures: "
        f"{len(structure_map):,}"
    )

    # =====================================================
    # Stage 7 exact-mass index
    # =====================================================

    print(
        "\nLoading Stage 7 Enveda mass variants..."
    )

    variants = pd.read_csv(
        ENVEDA_VARIANTS_PATH
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

    # =====================================================
    # Generator cache
    # =====================================================

    generators = {}

    def get_generator(
        tolerance_ppm,
    ):

        tolerance_ppm = float(
            tolerance_ppm
        )

        if tolerance_ppm not in generators:

            generators[
                tolerance_ppm
            ] = AdductAwareCandidateGenerator(
                mass_index=mass_index,
                tolerance_ppm=tolerance_ppm,
            )

        return generators[
            tolerance_ppm
        ]

    # =====================================================
    # Original train metadata
    # =====================================================

    print(
        "\nLoading original training metadata..."
    )

    train_metadata = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "precursor_mz",
            "adduct",
            "instrument_type",
            "ingest_lib",
            "inchikey14",
        ],
    )

    print(
        f"Original train rows: "
        f"{len(train_metadata):,}"
    )

    # =====================================================
    # Map Stage 6 cache -> train.parquet rows
    # =====================================================

    stage6_row_indices = np.load(
        STAGE6_ROW_INDICES_PATH,
        mmap_mode="r",
    )

    validation_cache_indices = (
        dataset.indices
    )

    original_row_indices = (
        stage6_row_indices[
            validation_cache_indices
        ]
    )

    validation_metadata = (
        train_metadata
        .iloc[
            original_row_indices
        ]
        .reset_index(
            drop=True
        )
    )

    if len(
        validation_metadata
    ) != len(
        dataset
    ):

        raise RuntimeError(
            "Validation metadata length mismatch."
        )

    print(
        f"Mapped validation metadata rows: "
        f"{len(validation_metadata):,}"
    )

    # =====================================================
    # Load Stage 6 spectrum encoder
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

    spectrum_result = (
        spectrum_encoder.load_state_dict(
            stage6_checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )
    )

    print(
        f"Stage 6 missing keys: "
        f"{len(spectrum_result.missing_keys)}"
    )

    print(
        f"Stage 6 unexpected keys: "
        f"{len(spectrum_result.unexpected_keys)}"
    )

    spectrum_encoder = (
        spectrum_encoder
        .to(
            device
        )
        .eval()
    )

    # =====================================================
    # Load candidate-aware Stage 8 molecule encoder
    # =====================================================

    print(
        "\nLoading candidate-aware "
        "Stage 8 molecule encoder..."
    )

    molecule_encoder = (
        build_molecule_encoder()
    )

    stage8_checkpoint = torch.load(
        STAGE8_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if (
        "molecule_encoder_state_dict"
        not in stage8_checkpoint
    ):

        raise KeyError(
            "Checkpoint does not contain "
            "'molecule_encoder_state_dict'."
        )

    molecule_result = (
        molecule_encoder.load_state_dict(
            stage8_checkpoint[
                "molecule_encoder_state_dict"
            ],
            strict=True,
        )
    )

    print(
        f"Stage 8 missing keys: "
        f"{len(molecule_result.missing_keys)}"
    )

    print(
        f"Stage 8 unexpected keys: "
        f"{len(molecule_result.unexpected_keys)}"
    )

    print(
        f"Checkpoint training type: "
        f"{stage8_checkpoint.get('training_type', 'unknown')}"
    )

    print(
        f"Checkpoint epoch: "
        f"{stage8_checkpoint.get('epoch', 'unknown')}"
    )

    molecule_encoder = (
        molecule_encoder
        .to(
            device
        )
        .eval()
    )

    # =====================================================
    # Encode all Stage 8 molecules
    # =====================================================

    print(
        "\nEncoding all Stage 8 molecules..."
    )

    fingerprints = (
        dataset.fingerprints
    )

    all_molecule_embeddings = []

    molecule_start = (
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

            fp_numpy = np.asarray(
                fingerprints[
                    start:end
                ],
                dtype=np.uint8,
            )

            fp_tensor = torch.from_numpy(
                fp_numpy
            ).to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                embeddings = (
                    molecule_encoder(
                        fp_tensor
                    )
                )

            all_molecule_embeddings.append(
                embeddings.float()
            )

    molecule_embeddings = torch.cat(
        all_molecule_embeddings,
        dim=0,
    )

    torch.cuda.synchronize()

    molecule_elapsed = (
        time.perf_counter()
        - molecule_start
    )

    print(
        f"Molecule embedding matrix: "
        f"{tuple(molecule_embeddings.shape)}"
    )

    print(
        f"Molecule encoding elapsed: "
        f"{molecule_elapsed:.2f} s"
    )

    # =====================================================
    # Encode validation spectra
    # =====================================================

    print(
        "\nEncoding validation spectra..."
    )

    spectrum_embedding_batches = []

    spectrum_start = (
        time.perf_counter()
    )

    with torch.inference_mode():

        for batch_number, batch in enumerate(
            loader,
            start=1,
        ):

            peaks = prepare_peaks(
                batch,
                device,
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

            spectrum_embedding_batches.append(
                embeddings.float().cpu()
            )

            if (
                batch_number
                % 50
                == 0
            ):

                processed = min(
                    batch_number
                    * BATCH_SIZE,
                    len(dataset),
                )

                print(
                    f"Encoded "
                    f"{processed:,}/"
                    f"{len(dataset):,}"
                )

    spectrum_embeddings = torch.cat(
        spectrum_embedding_batches,
        dim=0,
    )

    torch.cuda.synchronize()

    spectrum_elapsed = (
        time.perf_counter()
        - spectrum_start
    )

    print(
        f"Spectrum embedding matrix: "
        f"{tuple(spectrum_embeddings.shape)}"
    )

    print(
        f"Spectrum encoding elapsed: "
        f"{spectrum_elapsed:.2f} s"
    )

    # =====================================================
    # Chemistry-constrained ranking
    # =====================================================

    print(
        "\nGenerating and ranking "
        "chemistry-constrained candidates..."
    )

    results = []

    candidate_counts = []

    surviving_ranks = []

    timstof_surviving_ranks = []

    high_confidence_surviving_ranks = []

    rows_by_source = defaultdict(
        list
    )

    supported_count = 0

    trusted_count = 0

    hard_filter_count = 0

    survival_count = 0

    zero_candidate_count = 0

    ranking_start = (
        time.perf_counter()
    )

    for i in range(
        len(dataset)
    ):

        row = validation_metadata.iloc[
            i
        ]

        precursor_mz = float(
            row[
                "precursor_mz"
            ]
        )

        adduct = (
            ""
            if pd.isna(
                row[
                    "adduct"
                ]
            )
            else str(
                row[
                    "adduct"
                ]
            )
        )

        instrument_type = (
            ""
            if pd.isna(
                row[
                    "instrument_type"
                ]
            )
            else str(
                row[
                    "instrument_type"
                ]
            )
        )

        ingest_lib = (
            ""
            if pd.isna(
                row[
                    "ingest_lib"
                ]
            )
            else str(
                row[
                    "ingest_lib"
                ]
            )
        )

        true_inchikey = (
            ""
            if pd.isna(
                row[
                    "inchikey14"
                ]
            )
            else str(
                row[
                    "inchikey14"
                ]
            )
        )

        # -------------------------------------------------
        # Adduct eligibility
        # -------------------------------------------------

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

        if supported:
            supported_count += 1

        if trusted:
            trusted_count += 1

        # -------------------------------------------------
        # Existing Stage 7 mass-filter policy
        # -------------------------------------------------

        decision = (
            get_mass_filter_decision(
                adduct_supported=(
                    supported
                ),
                adduct_trusted=(
                    trusted
                ),
                instrument_type=(
                    instrument_type
                ),
                ingest_lib=(
                    ingest_lib
                ),
                inference_mode=False,
            )
        )

        result_row = {
            "validation_index": int(
                i
            ),
            "original_row_index": int(
                original_row_indices[
                    i
                ]
            ),
            "inchikey14": (
                true_inchikey
            ),
            "precursor_mz": (
                precursor_mz
            ),
            "adduct": (
                adduct
            ),
            "instrument_type": (
                instrument_type
            ),
            "ingest_lib": (
                ingest_lib
            ),
            "adduct_supported": (
                bool(
                    supported
                )
            ),
            "adduct_trusted": (
                bool(
                    trusted
                )
            ),
            "hard_filter": (
                bool(
                    decision.hard_filter
                )
            ),
            "mass_filter_confidence": (
                decision.confidence
            ),
            "tolerance_ppm": (
                decision.tolerance_ppm
            ),
            "candidate_count": np.nan,
            "true_survived": False,
            "rank": np.nan,
        }

        # -------------------------------------------------
        # Skip rows where policy refuses hard filtering
        # -------------------------------------------------

        if not decision.hard_filter:

            results.append(
                result_row
            )

            continue

        hard_filter_count += 1

        # -------------------------------------------------
        # Generate candidate set
        # -------------------------------------------------

        generator = get_generator(
            decision.tolerance_ppm
        )

        generation = generator.generate(
            precursor_mz=precursor_mz,
            adduct=adduct,
            require_trusted=True,
        )

        candidate_indices = []

        candidate_inchikeys = []

        for hit in generation.candidates:

            structure_index = (
                inchikey_to_structure_index
                .get(
                    hit.inchikey14
                )
            )

            if structure_index is None:
                continue

            candidate_indices.append(
                int(
                    structure_index
                )
            )

            candidate_inchikeys.append(
                str(
                    hit.inchikey14
                )
            )

        candidate_count = len(
            candidate_indices
        )

        candidate_counts.append(
            candidate_count
        )

        result_row[
            "candidate_count"
        ] = int(
            candidate_count
        )

        if candidate_count == 0:

            zero_candidate_count += 1

            results.append(
                result_row
            )

            rows_by_source[
                ingest_lib
            ].append(
                np.nan
            )

            continue

        # -------------------------------------------------
        # Check true structure survival
        # -------------------------------------------------

        true_survived = (
            true_inchikey
            in candidate_inchikeys
        )

        result_row[
            "true_survived"
        ] = bool(
            true_survived
        )

        if not true_survived:

            results.append(
                result_row
            )

            rows_by_source[
                ingest_lib
            ].append(
                np.nan
            )

            continue

        survival_count += 1

        # -------------------------------------------------
        # Candidate-aware Stage 8 scoring
        # -------------------------------------------------

        candidate_index_tensor = torch.tensor(
            candidate_indices,
            dtype=torch.long,
            device=device,
        )

        spectrum_embedding = (
            spectrum_embeddings[
                i
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        candidate_embedding_matrix = (
            molecule_embeddings[
                candidate_index_tensor
            ]
        )

        similarities = (
            candidate_embedding_matrix
            @ spectrum_embedding
        )

        true_position = (
            candidate_inchikeys
            .index(
                true_inchikey
            )
        )

        true_score = similarities[
            true_position
        ]

        rank = int(
            (
                similarities
                > true_score
            )
            .sum()
            .item()
            + 1
        )

        result_row[
            "rank"
        ] = int(
            rank
        )

        surviving_ranks.append(
            rank
        )

        rows_by_source[
            ingest_lib
        ].append(
            rank
        )

        if (
            instrument_type
            .strip()
            .lower()
            == "timstof"
        ):

            timstof_surviving_ranks.append(
                rank
            )

        if (
            decision.confidence
            == "high"
        ):

            high_confidence_surviving_ranks.append(
                rank
            )

        results.append(
            result_row
        )

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if (
            (i + 1)
            % 10_000
            == 0
        ):

            elapsed = (
                time.perf_counter()
                - ranking_start
            )

            print(
                f"Processed "
                f"{i + 1:,}/"
                f"{len(dataset):,} "
                f"| hard-filtered "
                f"{hard_filter_count:,} "
                f"| survived "
                f"{survival_count:,} "
                f"| {elapsed:.1f}s"
            )

    ranking_elapsed = (
        time.perf_counter()
        - ranking_start
    )

    # =====================================================
    # Row-level output
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # =====================================================
    # Metrics
    # =====================================================

    candidate_stats = describe(
        candidate_counts
    )

    surviving_metrics = (
        calculate_rank_metrics(
            surviving_ranks
        )
    )

    timstof_metrics = (
        calculate_rank_metrics(
            timstof_surviving_ranks
        )
    )

    high_confidence_metrics = (
        calculate_rank_metrics(
            high_confidence_surviving_ranks
        )
    )

    hard_filter_rate = (
        hard_filter_count
        / len(dataset)
        if len(dataset) > 0
        else 0.0
    )

    survival_rate = (
        survival_count
        / hard_filter_count
        if hard_filter_count > 0
        else 0.0
    )

    zero_candidate_rate = (
        zero_candidate_count
        / hard_filter_count
        if hard_filter_count > 0
        else 0.0
    )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    # =====================================================
    # Console summary
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 8 CANDIDATE-AWARE "
        "CHEMISTRY-CONSTRAINED RETRIEVAL RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        f"Validation spectra:       "
        f"{len(dataset):,}"
    )

    print(
        f"Supported adduct:         "
        f"{supported_count:,}"
    )

    print(
        f"Trusted adduct:           "
        f"{trusted_count:,}"
    )

    print(
        f"Hard-filter eligible:     "
        f"{hard_filter_count:,} "
        f"({hard_filter_rate * 100:.2f}%)"
    )

    print(
        f"True structure survived:  "
        f"{survival_count:,}/"
        f"{hard_filter_count:,} "
        f"({survival_rate * 100:.3f}%)"
    )

    print(
        f"Zero-candidate rows:      "
        f"{zero_candidate_count:,} "
        f"({zero_candidate_rate * 100:.3f}%)"
    )

    print(
        "\nCandidate counts"
    )

    print(
        f"Min:     "
        f"{candidate_stats['min']:.0f}"
    )

    print(
        f"Median:  "
        f"{candidate_stats['median']:.1f}"
    )

    print(
        f"Mean:    "
        f"{candidate_stats['mean']:.2f}"
    )

    print(
        f"P90:     "
        f"{candidate_stats['p90']:.1f}"
    )

    print(
        f"P95:     "
        f"{candidate_stats['p95']:.1f}"
    )

    print(
        f"P99:     "
        f"{candidate_stats['p99']:.1f}"
    )

    print(
        f"Max:     "
        f"{candidate_stats['max']:.0f}"
    )

    print_rank_metrics(
        "Ranking — all surviving hard-filter rows",
        surviving_metrics,
    )

    print_rank_metrics(
        "Ranking — timsTOF surviving rows",
        timstof_metrics,
    )

    print_rank_metrics(
        "Ranking — high-confidence surviving rows",
        high_confidence_metrics,
    )

    print(
        f"\nRanking elapsed: "
        f"{ranking_elapsed:.2f} s"
    )

    print(
        f"Total elapsed: "
        f"{total_elapsed:.2f} s"
    )

    print(
        "=" * 100
    )

    # =====================================================
    # Summary file
    # =====================================================

    summary_lines = [
        (
            "ENVEDA CASMI 2026 — "
            "STAGE 8 CANDIDATE-AWARE "
            "CHEMISTRY-CONSTRAINED RETRIEVAL"
        ),
        "",
        (
            f"Checkpoint: "
            f"{STAGE8_CHECKPOINT}"
        ),
        "",
        (
            f"Validation spectra: "
            f"{len(dataset):,}"
        ),
        (
            f"Supported adduct: "
            f"{supported_count:,}"
        ),
        (
            f"Trusted adduct: "
            f"{trusted_count:,}"
        ),
        (
            f"Hard-filter eligible: "
            f"{hard_filter_count:,}"
        ),
        (
            f"Hard-filter rate: "
            f"{hard_filter_rate:.8f}"
        ),
        (
            f"True structure survived: "
            f"{survival_count:,}"
        ),
        (
            f"Survival rate: "
            f"{survival_rate:.8f}"
        ),
        (
            f"Zero-candidate rows: "
            f"{zero_candidate_count:,}"
        ),
        (
            f"Zero-candidate rate: "
            f"{zero_candidate_rate:.8f}"
        ),
        "",
        "Candidate counts:",
        (
            f"min: "
            f"{candidate_stats['min']}"
        ),
        (
            f"median: "
            f"{candidate_stats['median']}"
        ),
        (
            f"mean: "
            f"{candidate_stats['mean']}"
        ),
        (
            f"p90: "
            f"{candidate_stats['p90']}"
        ),
        (
            f"p95: "
            f"{candidate_stats['p95']}"
        ),
        (
            f"p99: "
            f"{candidate_stats['p99']}"
        ),
        (
            f"max: "
            f"{candidate_stats['max']}"
        ),
        "",
    ]

    def append_metrics(
        title,
        metrics,
    ):

        summary_lines.append(
            title
        )

        if metrics is None:

            summary_lines.append(
                "No eligible rows."
            )

        else:

            summary_lines.extend(
                [
                    (
                        f"count: "
                        f"{metrics['count']}"
                    ),
                    (
                        f"recall@1: "
                        f"{metrics['recall_1']:.8f}"
                    ),
                    (
                        f"recall@5: "
                        f"{metrics['recall_5']:.8f}"
                    ),
                    (
                        f"recall@10: "
                        f"{metrics['recall_10']:.8f}"
                    ),
                    (
                        f"recall@25: "
                        f"{metrics['recall_25']:.8f}"
                    ),
                    (
                        f"mrr: "
                        f"{metrics['mrr']:.8f}"
                    ),
                    (
                        f"median_rank: "
                        f"{metrics['median_rank']:.4f}"
                    ),
                    (
                        f"mean_rank: "
                        f"{metrics['mean_rank']:.4f}"
                    ),
                ]
            )

        summary_lines.append(
            ""
        )

    append_metrics(
        "All surviving hard-filter rows:",
        surviving_metrics,
    )

    append_metrics(
        "timsTOF surviving rows:",
        timstof_metrics,
    )

    append_metrics(
        "High-confidence surviving rows:",
        high_confidence_metrics,
    )

    summary_lines.extend(
        [
            (
                f"Ranking elapsed seconds: "
                f"{ranking_elapsed:.2f}"
            ),
            (
                f"Total elapsed seconds: "
                f"{total_elapsed:.2f}"
            ),
        ]
    )

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    # =====================================================
    # Final file locations
    # =====================================================

    print(
        "\nRow-level candidate-aware results:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nCandidate-aware summary:"
    )

    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()