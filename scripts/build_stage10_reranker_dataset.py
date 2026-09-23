from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from casmi.chemistry.adducts import (
    is_supported_adduct,
    is_trusted_for_mass_filter,
    mass_error_ppm,
    precursor_to_neutral_mass,
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

from casmi.ranking.multispectrum import (
    aggregate_multispectrum_scores,
    build_availability_mask,
    select_candidate_pool,
)

from casmi.reranking.dataset import (
    ranking_dataframe_summary,
    validate_reranker_dataframe,
    validate_single_positive_per_query,
)

from casmi.reranking.features import (
    FEATURE_COLUMNS,
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


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

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

STAGE8_FINGERPRINTS = (
    STAGE8_DIR
    / "morgan_fingerprints.npy"
)

STAGE6_ROW_INDICES_PATH = (
    STAGE6_DIR
    / "row_indices.npy"
)

OUTPUT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage10_reranker_dataset.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage10_reranker_dataset_summary.txt"
)


# ============================================================
# Configuration
# ============================================================

SPECTRUM_BATCH_SIZE = 512
MOLECULE_BATCH_SIZE = 4096
NUM_WORKERS = 0
EMBEDDING_DIM = 128


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build Stage 10 candidate-level "
            "learning-to-rank dataset."
        )
    )

    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help=(
            "Maximum number of multispectrum groups "
            "to process. Useful for smoke testing."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DATASET,
        help="Output candidate-level CSV path.",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=OUTPUT_SUMMARY,
        help="Output summary text path.",
    )

    parser.add_argument(
        "--all-instruments",
        action="store_true",
        help=(
            "Use all validation instruments instead "
            "of only the Stage 9 timsTOF domain."
        ),
    )

    return parser.parse_args()


# ============================================================
# Model builders
# ============================================================

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


# ============================================================
# Checkpoint helpers
# ============================================================

def _is_state_dict(value):
    if not isinstance(
        value,
        dict,
    ):
        return False

    if not value:
        return False

    return all(
        isinstance(
            tensor,
            torch.Tensor,
        )
        for tensor in value.values()
    )


def _candidate_state_dicts(
    checkpoint,
    preferred_keys,
):
    if _is_state_dict(
        checkpoint
    ):
        yield checkpoint

    if not isinstance(
        checkpoint,
        dict,
    ):
        return

    for key in preferred_keys:
        value = checkpoint.get(
            key
        )

        if _is_state_dict(
            value
        ):
            yield value

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
    ):
        value = checkpoint.get(
            key
        )

        if _is_state_dict(
            value
        ):
            yield value


def _strip_prefix(
    state_dict,
    prefix,
):
    if not any(
        key.startswith(prefix)
        for key in state_dict
    ):
        return state_dict

    result = {}

    for key, value in (
        state_dict.items()
    ):
        if key.startswith(
            prefix
        ):
            result[
                key[len(prefix):]
            ] = value

    return result


def load_model_checkpoint(
    model,
    checkpoint_path,
    *,
    preferred_keys,
    prefixes=(),
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    candidates = list(
        _candidate_state_dicts(
            checkpoint,
            preferred_keys,
        )
    )

    expanded = []

    for state_dict in candidates:
        expanded.append(
            state_dict
        )

        for prefix in prefixes:
            stripped = (
                _strip_prefix(
                    state_dict,
                    prefix,
                )
            )

            if stripped is not state_dict:
                expanded.append(
                    stripped
                )

    errors = []

    for state_dict in expanded:
        try:
            missing, unexpected = (
                model.load_state_dict(
                    state_dict,
                    strict=False,
                )
            )

            if (
                len(missing) == 0
                and len(unexpected) == 0
            ):
                return

            errors.append(
                {
                    "missing": len(
                        missing
                    ),
                    "unexpected": len(
                        unexpected
                    ),
                }
            )

        except RuntimeError as exc:
            errors.append(
                str(exc)
            )

    raise RuntimeError(
        "Could not load checkpoint into "
        f"{type(model).__name__}.\n"
        f"Checkpoint: {checkpoint_path}\n"
        f"Attempts: {errors}"
    )


# ============================================================
# Ranking helpers
# ============================================================

def scores_to_ranks(scores):
    """
    Convert higher-is-better scores to deterministic,
    one-based ranks.
    """

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if scores.ndim != 1:
        raise ValueError(
            "scores must be one-dimensional."
        )

    if len(scores) == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    order = np.argsort(
        -scores,
        kind="stable",
    )

    ranks = np.empty(
        len(scores),
        dtype=np.int64,
    )

    ranks[order] = (
        np.arange(
            len(scores),
            dtype=np.int64,
        )
        + 1
    )

    return ranks


def tensor_to_numpy(
    tensor,
):
    return np.asarray(
        tensor
        .detach()
        .float()
        .cpu()
        .numpy(),
        dtype=np.float64,
    )


# ============================================================
# Candidate metadata helpers
# ============================================================

def build_exact_mass_lookup(
    variants,
):
    grouped = (
        variants
        .groupby(
            "inchikey14",
            sort=False,
        )["exact_mass"]
        .median()
    )

    return {
        str(key): float(value)
        for key, value
        in grouped.items()
        if np.isfinite(value)
    }


def calculate_candidate_mass_error(
    *,
    candidate_position,
    candidate_exact_mass,
    group_df,
    availability_numpy,
):
    """
    Mean signed ppm error across spectra where this
    candidate is available.

    Uses only test-time-safe information:
    candidate exact mass + precursor m/z + adduct.
    """

    if candidate_exact_mass is None:
        return 0.0

    ppm_values = []

    for (
        spectrum_position,
        row,
    ) in enumerate(
        group_df.itertuples(
            index=False
        )
    ):
        if not availability_numpy[
            spectrum_position,
            candidate_position,
        ]:
            continue

        adduct = (
            ""
            if pd.isna(
                row.adduct
            )
            else str(
                row.adduct
            )
        )

        if not is_supported_adduct(
            adduct
        ):
            continue

        if not is_trusted_for_mass_filter(
            adduct
        ):
            continue

        try:
            observed_neutral_mass = (
                precursor_to_neutral_mass(
                    precursor_mz=float(
                        row.precursor_mz
                    ),
                    adduct=adduct,
                    require_trusted=True,
                )
            )

            ppm = mass_error_ppm(
                observed_mass=(
                    observed_neutral_mass
                ),
                expected_mass=(
                    candidate_exact_mass
                ),
            )

        except (
            ValueError,
            KeyError,
            ZeroDivisionError,
        ):
            continue

        if np.isfinite(
            ppm
        ):
            ppm_values.append(
                float(ppm)
            )

    if not ppm_values:
        return 0.0

    return float(
        np.mean(
            ppm_values
        )
    )


# ============================================================
# Score statistics
# ============================================================

def candidate_score_statistics(
    score_matrix,
    availability_mask,
):
    num_candidates = (
        score_matrix.shape[1]
    )

    std_values = np.zeros(
        num_candidates,
        dtype=np.float64,
    )

    range_values = np.zeros(
        num_candidates,
        dtype=np.float64,
    )

    for candidate_position in range(
        num_candidates
    ):
        valid_scores = score_matrix[
            availability_mask[
                :,
                candidate_position,
            ],
            candidate_position,
        ]

        if (
            valid_scores.numel()
            == 0
        ):
            continue

        values = (
            valid_scores
            .detach()
            .float()
        )

        std_values[
            candidate_position
        ] = float(
            values.std(
                unbiased=False
            ).item()
        )

        range_values[
            candidate_position
        ] = float(
            (
                values.max()
                - values.min()
            ).item()
        )

    return (
        std_values,
        range_values,
    )


# ============================================================
# File validation
# ============================================================

def validate_required_files():
    required_paths = [
        TRAIN_PATH,
        STAGE6_DIR,
        STAGE8_DIR,
        STAGE6_CHECKPOINT,
        STAGE8_CHECKPOINT,
        STAGE7_VARIANTS_PATH,
        STAGE8_STRUCTURE_MAP,
        STAGE8_FINGERPRINTS,
        STAGE6_ROW_INDICES_PATH,
    ]

    missing = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required Stage 10 files are missing:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )


# ============================================================
# Validation metadata
# ============================================================

def load_validation_metadata(
    dataset,
):
    print(
        "\nLoading validation metadata..."
    )

    train_metadata = (
        pd.read_parquet(
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
    )

    row_indices = np.load(
        STAGE6_ROW_INDICES_PATH,
        mmap_mode="r",
    )

    original_row_indices = np.array(
        row_indices[
            dataset.indices
        ],
        dtype=np.int64,
        copy=True,
    )

    metadata = (
        train_metadata
        .iloc[
            original_row_indices
        ]
        .copy()
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

    metadata[
        "original_row_index"
    ] = original_row_indices

    return metadata


# ============================================================
# Candidate generation
# ============================================================

def generate_candidate_sets(
    metadata,
    *,
    mass_index,
    inchikey_to_structure_index,
):
    print(
        "\nGenerating per-spectrum candidate sets..."
    )

    generator_cache = {}

    def get_generator(
        tolerance_ppm,
    ):
        tolerance_ppm = float(
            tolerance_ppm
        )

        if (
            tolerance_ppm
            not in generator_cache
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

    spectrum_candidate_sets = {}

    eligible_indices = []

    true_survival_spectra = 0

    skipped_unsupported = 0

    skipped_policy = 0

    skipped_empty = 0

    start = (
        time.perf_counter()
    )

    for (
        row_counter,
        row,
    ) in enumerate(
        metadata.itertuples(
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

        if not supported:
            skipped_unsupported += 1
            continue

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

        if (
            not decision.hard_filter
            or decision.tolerance_ppm
            is None
        ):
            skipped_policy += 1
            continue

        generator = get_generator(
            decision.tolerance_ppm
        )

        try:
            generation = (
                generator.generate(
                    precursor_mz=float(
                        row.precursor_mz
                    ),
                    adduct=adduct,
                    require_trusted=True,
                )
            )

        except (
            ValueError,
            KeyError,
        ):
            skipped_policy += 1
            continue

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

            if (
                structure_index
                is not None
            ):
                candidate_indices.append(
                    int(
                        structure_index
                    )
                )

        if not candidate_indices:
            skipped_empty += 1
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
            and int(
                true_structure_index
            )
            in candidate_indices
        ):
            true_survival_spectra += 1

        if (
            row_counter % 10_000
            == 0
        ):
            print(
                f"Processed "
                f"{row_counter:,}/"
                f"{len(metadata):,} spectra"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"\nEligible spectra: "
        f"{len(eligible_indices):,}"
    )

    print(
        f"True candidate survived: "
        f"{true_survival_spectra:,}"
    )

    print(
        f"Unsupported skipped: "
        f"{skipped_unsupported:,}"
    )

    print(
        f"Policy skipped: "
        f"{skipped_policy:,}"
    )

    print(
        f"Empty candidate sets: "
        f"{skipped_empty:,}"
    )

    print(
        f"Candidate generation elapsed: "
        f"{elapsed:.2f}s"
    )

    return (
        spectrum_candidate_sets,
        np.asarray(
            eligible_indices,
            dtype=np.int64,
        ),
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    validate_required_files()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the Stage 10 "
            "dataset builder."
        )

    if (
        args.max_groups
        is not None
        and args.max_groups <= 0
    ):
        raise ValueError(
            "--max-groups must be > 0."
        )

    device = torch.device(
        "cuda"
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "ENVEDA CASMI 2026 — "
        "STAGE 10 RERANKER DATASET BUILDER"
    )

    print(
        "=" * 90
    )

    print(
        f"\nGPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    if args.max_groups is not None:
        print(
            f"Smoke limit: "
            f"{args.max_groups:,} groups"
        )

    total_start = (
        time.perf_counter()
    )

    # ========================================================
    # Stage 8 validation dataset
    # ========================================================

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

    loader = DataLoader(
        dataset,
        batch_size=(
            SPECTRUM_BATCH_SIZE
        ),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"Validation spectra: "
        f"{len(dataset):,}"
    )

    # ========================================================
    # Stage 8 structure map
    # ========================================================

    print(
        "\nLoading Stage 8 structure map..."
    )

    structure_map = pd.read_csv(
        STAGE8_STRUCTURE_MAP
    )

    required_structure_columns = {
        "structure_index",
        "inchikey14",
        "normalized_smiles",
        "molecular_formula",
    }

    missing = (
        required_structure_columns
        - set(
            structure_map.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Structure map missing columns: "
            f"{sorted(missing)}"
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

    structure_lookup = (
        structure_map
        .set_index(
            "structure_index",
            drop=False,
        )
    )

    print(
        f"Structures: "
        f"{len(structure_map):,}"
    )

    # ========================================================
    # Stage 7 mass index
    # ========================================================

    print(
        "\nLoading Stage 7 candidate mass index..."
    )

    variants = pd.read_csv(
        STAGE7_VARIANTS_PATH
    )

    required_variant_columns = {
        "inchikey14",
        "normalized_smiles",
        "molecular_formula",
        "exact_mass",
    }

    missing = (
        required_variant_columns
        - set(
            variants.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Stage 7 variant table missing columns: "
            f"{sorted(missing)}"
        )

    variants[
        "inchikey14"
    ] = (
        variants[
            "inchikey14"
        ]
        .astype(str)
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

    exact_mass_lookup = (
        build_exact_mass_lookup(
            variants
        )
    )

    print(
        f"Indexed structures: "
        f"{mass_index.unique_structure_count:,}"
    )

    print(
        f"Mass variants: "
        f"{mass_index.variant_count:,}"
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = (
        load_validation_metadata(
            dataset
        )
    )

    if not args.all_instruments:
        instrument_normalized = (
            metadata[
                "instrument_type"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        metadata = (
            metadata[
                instrument_normalized
                == "timstof"
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        print(
            f"timsTOF spectra: "
            f"{len(metadata):,}"
        )

    else:
        print(
            "Using all validation instruments."
        )

    if metadata.empty:
        raise RuntimeError(
            "No spectra remain after "
            "metadata filtering."
        )

    # ========================================================
    # Spectrum encoder
    # ========================================================

    print(
        "\nLoading Stage 6 spectrum encoder..."
    )

    spectrum_encoder = (
        build_spectrum_encoder()
    )

    load_model_checkpoint(
        spectrum_encoder,
        STAGE6_CHECKPOINT,
        preferred_keys=(
            "model_state_dict",
            "spectrum_encoder_state_dict",
        ),
        prefixes=(
            "spectrum_encoder.",
            "module.spectrum_encoder.",
        ),
    )

    spectrum_encoder = (
        spectrum_encoder
        .to(device)
        .eval()
    )

    # ========================================================
    # Candidate-aware molecule encoder
    # ========================================================

    print(
        "Loading Stage 8 candidate-aware "
        "molecule encoder..."
    )

    molecule_encoder = (
        build_molecule_encoder()
    )

    load_model_checkpoint(
        molecule_encoder,
        STAGE8_CHECKPOINT,
        preferred_keys=(
            "model_state_dict",
            "molecule_encoder_state_dict",
        ),
        prefixes=(
            "molecule_encoder.",
            "module.molecule_encoder.",
        ),
    )

    molecule_encoder = (
        molecule_encoder
        .to(device)
        .eval()
    )

    # ========================================================
    # Fingerprints
    # ========================================================

    print(
        "\nLoading molecule fingerprint cache..."
    )

    fingerprint_array = np.load(
        STAGE8_FINGERPRINTS,
        mmap_mode="r",
    )

    if (
        fingerprint_array.shape[0]
        != len(structure_map)
    ):
        raise RuntimeError(
            "Fingerprint count does not match "
            "Stage 8 structure map."
        )

    print(
        f"Fingerprint matrix: "
        f"{fingerprint_array.shape}"
    )

    # ========================================================
    # Encode molecules
    # ========================================================

    print(
        "\nEncoding candidate molecules..."
    )

    molecule_embedding_parts = []

    encode_start = (
        time.perf_counter()
    )

    with torch.inference_mode():
        for start in range(
            0,
            len(fingerprint_array),
            MOLECULE_BATCH_SIZE,
        ):
            end = min(
                start
                + MOLECULE_BATCH_SIZE,
                len(
                    fingerprint_array
                ),
            )

            # Copy is intentional because the memmap
            # may be read-only.
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

            batch_number = (
                start
                // MOLECULE_BATCH_SIZE
            )

            if (
                batch_number % 25
                == 0
            ):
                print(
                    f"Encoded molecules "
                    f"{end:,}/"
                    f"{len(fingerprint_array):,}"
                )

    molecule_embeddings = (
        torch.cat(
            molecule_embedding_parts,
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

    # ========================================================
    # Encode spectra
    # ========================================================

    print(
        "\nEncoding validation spectra..."
    )

    spectrum_parts = []

    spectrum_start = (
        time.perf_counter()
    )

    with torch.inference_mode():
        for (
            batch_number,
            batch,
        ) in enumerate(
            loader,
            start=1,
        ):
            mzs = (
                batch[
                    "mzs"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            intensities = (
                batch[
                    "intensities"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            peaks = torch.stack(
                [
                    mzs,
                    intensities,
                ],
                dim=-1,
            )

            mask = (
                batch[
                    "mask"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            precursor = (
                batch[
                    "precursor_mz"
                ]
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
                    spectrum_encoder(
                        peaks,
                        mask,
                        precursor,
                    )
                )

            spectrum_parts.append(
                embeddings
                .float()
                .cpu()
            )

            if (
                batch_number % 50
                == 0
            ):
                processed = min(
                    batch_number
                    * SPECTRUM_BATCH_SIZE,
                    len(dataset),
                )

                print(
                    f"Encoded spectra "
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

    # ========================================================
    # Candidate generation
    # ========================================================

    (
        spectrum_candidate_sets,
        eligible_indices,
    ) = generate_candidate_sets(
        metadata,
        mass_index=mass_index,
        inchikey_to_structure_index=(
            inchikey_to_structure_index
        ),
    )

    if len(
        eligible_indices
    ) == 0:
        raise RuntimeError(
            "No chemistry-eligible spectra found."
        )

    eligible_metadata = (
        metadata[
            metadata[
                "validation_index"
            ].isin(
                eligible_indices
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Stage 9 grouping
    # ========================================================

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
        f"\nEligible multispectrum groups: "
        f"{len(groups):,}"
    )

    if args.max_groups is not None:
        groups = groups[
            :args.max_groups
        ]

        print(
            f"Groups after smoke limit: "
            f"{len(groups):,}"
        )

    # ========================================================
    # Build Stage 10 rows
    # ========================================================

    dataset_rows = []

    groups_processed = 0
    groups_saved = 0

    groups_missing_true_structure = 0
    groups_true_not_in_pool = 0
    groups_empty_pool = 0

    candidate_mode_counts: Dict[
        str,
        int,
    ] = {}

    build_start = (
        time.perf_counter()
    )

    for (
        group_number,
        (
            group_key,
            group_df,
        ),
    ) in enumerate(
        groups,
        start=1,
    ):
        groups_processed += 1

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

        if (
            true_structure_index
            is None
        ):
            groups_missing_true_structure += 1
            continue

        # IMPORTANT FIX:
        # Force a writable NumPy copy before indexing
        # PyTorch tensors.
        spectrum_indices = np.array(
            group_df[
                "validation_index"
            ].astype(np.int64),
            dtype=np.int64,
            copy=True,
        )

        candidate_sets = [
            spectrum_candidate_sets[
                int(index)
            ]
            for index in (
                spectrum_indices
            )
        ]

        selection = (
            select_candidate_pool(
                candidate_sets
            )
        )

        candidate_indices = np.array(
            selection.candidates,
            dtype=np.int64,
            copy=True,
        )

        candidate_mode = (
            selection.mode
        )

        candidate_mode_counts[
            candidate_mode
        ] = (
            candidate_mode_counts
            .get(
                candidate_mode,
                0,
            )
            + 1
        )

        if len(
            candidate_indices
        ) == 0:
            groups_empty_pool += 1
            continue

        true_survived = bool(
            np.any(
                candidate_indices
                == int(
                    true_structure_index
                )
            )
        )

        if not true_survived:
            groups_true_not_in_pool += 1
            continue

        num_spectra = int(
            len(
                spectrum_indices
            )
        )

        # ====================================================
        # Spectrum embeddings
        # ====================================================

        group_spectrum_embeddings = (
            spectrum_embeddings[
                spectrum_indices
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        # ====================================================
        # Candidate embeddings
        # ====================================================

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

        # ====================================================
        # Learned spectrum-candidate compatibility
        #
        # [num_spectra, num_candidates]
        # ====================================================

        score_matrix = (
            group_spectrum_embeddings
            @ candidate_embeddings.T
        )

        # ====================================================
        # Availability mask
        # ====================================================

        availability_numpy = (
            build_availability_mask(
                candidate_sets,
                candidate_indices,
            )
        )

        availability_numpy = np.array(
            availability_numpy,
            dtype=np.bool_,
            copy=True,
        )

        availability_mask = (
            torch.from_numpy(
                availability_numpy
            )
            .to(
                device=device
            )
        )

        # ====================================================
        # Stage 9 aggregates
        # ====================================================

        mean_scores = (
            aggregate_multispectrum_scores(
                score_matrix,
                availability_mask,
                method="mean",
            )
        )

        max_scores = (
            aggregate_multispectrum_scores(
                score_matrix,
                availability_mask,
                method="max",
            )
        )

        top2_scores = (
            aggregate_multispectrum_scores(
                score_matrix,
                availability_mask,
                method="top2_mean",
            )
        )

        mean_numpy = (
            tensor_to_numpy(
                mean_scores
            )
        )

        max_numpy = (
            tensor_to_numpy(
                max_scores
            )
        )

        top2_numpy = (
            tensor_to_numpy(
                top2_scores
            )
        )

        if (
            not np.isfinite(
                mean_numpy
            ).all()
            or not np.isfinite(
                max_numpy
            ).all()
            or not np.isfinite(
                top2_numpy
            ).all()
        ):
            raise RuntimeError(
                "Non-finite Stage 10 aggregate "
                f"score in group {group_key}."
            )

        # ====================================================
        # Score statistics
        # ====================================================

        (
            score_std,
            score_range,
        ) = (
            candidate_score_statistics(
                score_matrix,
                availability_mask,
            )
        )

        support_counts = (
            availability_mask
            .sum(
                dim=0
            )
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int64
            )
        )

        availability_fractions = (
            support_counts.astype(
                np.float64
            )
            / float(
                max(
                    num_spectra,
                    1,
                )
            )
        )

        # ====================================================
        # Baseline ranks
        # ====================================================

        mean_ranks = (
            scores_to_ranks(
                mean_numpy
            )
        )

        max_ranks = (
            scores_to_ranks(
                max_numpy
            )
        )

        top2_ranks = (
            scores_to_ranks(
                top2_numpy
            )
        )

        query_id = (
            f"{true_inchikey}"
            f"|{ingest_lib}"
        )

        group_positive_count = 0

        # ====================================================
        # Candidate rows
        # ====================================================

        for (
            candidate_position,
            candidate_index,
        ) in enumerate(
            candidate_indices
        ):
            candidate_index = int(
                candidate_index
            )

            if (
                candidate_index
                not in structure_lookup.index
            ):
                continue

            candidate_info = (
                structure_lookup.loc[
                    candidate_index
                ]
            )

            if isinstance(
                candidate_info,
                pd.DataFrame,
            ):
                candidate_info = (
                    candidate_info.iloc[0]
                )

            candidate_inchikey = str(
                candidate_info[
                    "inchikey14"
                ]
            )

            target = int(
                candidate_index
                == int(
                    true_structure_index
                )
            )

            group_positive_count += (
                target
            )

            candidate_exact_mass = (
                exact_mass_lookup.get(
                    candidate_inchikey
                )
            )

            candidate_ppm_error = (
                calculate_candidate_mass_error(
                    candidate_position=(
                        candidate_position
                    ),
                    candidate_exact_mass=(
                        candidate_exact_mass
                    ),
                    group_df=group_df,
                    availability_numpy=(
                        availability_numpy
                    ),
                )
            )

            row = {
                # --------------------------------------------
                # IDs / labels
                # --------------------------------------------

                "query_id": (
                    query_id
                ),

                "candidate_id": (
                    candidate_inchikey
                ),

                "candidate_structure_index": (
                    candidate_index
                ),

                "target": (
                    target
                ),

                # --------------------------------------------
                # Diagnostics
                # --------------------------------------------

                "candidate_mode": (
                    candidate_mode
                ),

                "num_spectra": (
                    num_spectra
                ),

                "ingest_lib": (
                    ingest_lib
                ),

                "true_inchikey14": (
                    true_inchikey
                ),

                # --------------------------------------------
                # Reranker features
                # --------------------------------------------

                "aggregated_score": float(
                    mean_numpy[
                        candidate_position
                    ]
                ),

                "mean_score": float(
                    mean_numpy[
                        candidate_position
                    ]
                ),

                "max_score": float(
                    max_numpy[
                        candidate_position
                    ]
                ),

                "top2_mean_score": float(
                    top2_numpy[
                        candidate_position
                    ]
                ),

                "spectral_similarity_score": float(
                    mean_numpy[
                        candidate_position
                    ]
                ),

                "embedding_similarity_score": float(
                    max_numpy[
                        candidate_position
                    ]
                ),

                "neural_compatibility_score": float(
                    top2_numpy[
                        candidate_position
                    ]
                ),

                "mass_error_ppm": float(
                    candidate_ppm_error
                ),

                "abs_mass_error_ppm": float(
                    abs(
                        candidate_ppm_error
                    )
                ),

                # Kept disabled to avoid leaking the
                # true molecule formula.
                "formula_match": 0.0,

                "num_supporting_spectra": float(
                    support_counts[
                        candidate_position
                    ]
                ),

                "candidate_availability_fraction": float(
                    availability_fractions[
                        candidate_position
                    ]
                ),

                "retrieval_rank": float(
                    mean_ranks[
                        candidate_position
                    ]
                ),

                "similarity_rank": float(
                    max_ranks[
                        candidate_position
                    ]
                ),

                "neural_rank": float(
                    top2_ranks[
                        candidate_position
                    ]
                ),

                "score_std": float(
                    score_std[
                        candidate_position
                    ]
                ),

                "score_range": float(
                    score_range[
                        candidate_position
                    ]
                ),
            }

            dataset_rows.append(
                row
            )

        if (
            group_positive_count
            != 1
        ):
            raise RuntimeError(
                "Internal Stage 10 error: "
                f"query {query_id} produced "
                f"{group_positive_count} positives."
            )

        groups_saved += 1

        if (
            group_number % 100
            == 0
        ):
            elapsed = (
                time.perf_counter()
                - build_start
            )

            print(
                f"Groups "
                f"{group_number:,}/"
                f"{len(groups):,} | "
                f"saved {groups_saved:,} | "
                f"candidate rows "
                f"{len(dataset_rows):,} | "
                f"{elapsed:.1f}s"
            )

    # ========================================================
    # Final dataframe
    # ========================================================

    if not dataset_rows:
        raise RuntimeError(
            "Stage 10 produced no candidate rows."
        )

    reranker_df = pd.DataFrame(
        dataset_rows
    )

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in (
            reranker_df.columns
        )
    ]

    if missing_features:
        raise RuntimeError(
            "Stage 10 builder is missing "
            f"feature columns: "
            f"{missing_features}"
        )

    output_columns = [
        "query_id",
        "candidate_id",
        "candidate_structure_index",
        "target",
        "candidate_mode",
        "num_spectra",
        "ingest_lib",
        "true_inchikey14",
        *FEATURE_COLUMNS,
    ]

    reranker_df = (
        reranker_df[
            output_columns
        ]
    )

    # ========================================================
    # Validation
    # ========================================================

    print(
        "\nValidating Stage 10 dataset..."
    )

    validate_reranker_dataframe(
        reranker_df
    )

    validate_single_positive_per_query(
        reranker_df
    )

    duplicate_mask = (
        reranker_df.duplicated(
            subset=[
                "query_id",
                "candidate_id",
            ]
        )
    )

    if duplicate_mask.any():
        raise RuntimeError(
            "Duplicate query/candidate pairs "
            "detected in Stage 10 output."
        )

    positive_counts = (
        reranker_df
        .groupby(
            "query_id"
        )["target"]
        .sum()
    )

    invalid_query_count = int(
        (
            positive_counts
            != 1
        ).sum()
    )

    if invalid_query_count != 0:
        raise RuntimeError(
            "Stage 10 positive-label validation "
            f"failed for {invalid_query_count} queries."
        )

    # ========================================================
    # Dataset summary
    # ========================================================

    summary = (
        ranking_dataframe_summary(
            reranker_df
        )
    )

    true_rows = (
        reranker_df[
            reranker_df[
                "target"
            ]
            == 1
        ]
    )

    baseline_mean_rank = (
        true_rows[
            "retrieval_rank"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    baseline_top1 = float(
        np.mean(
            baseline_mean_rank
            <= 1
        )
    )

    baseline_top5 = float(
        np.mean(
            baseline_mean_rank
            <= 5
        )
    )

    baseline_top10 = float(
        np.mean(
            baseline_mean_rank
            <= 10
        )
    )

    baseline_top25 = float(
        np.mean(
            baseline_mean_rank
            <= 25
        )
    )

    baseline_mrr = float(
        np.mean(
            1.0
            / baseline_mean_rank
        )
    )

    # ========================================================
    # Save
    # ========================================================

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reranker_df.to_csv(
        args.output,
        index=False,
    )

    elapsed_total = (
        time.perf_counter()
        - total_start
    )

    summary_lines = [
        (
            "ENVEDA CASMI 2026 - "
            "STAGE 10 RERANKER DATASET SUMMARY"
        ),
        "=" * 72,
        "",
        (
            f"Output: "
            f"{args.output}"
        ),
        "",
        (
            f"Groups considered: "
            f"{groups_processed:,}"
        ),
        (
            f"Groups saved: "
            f"{groups_saved:,}"
        ),
        (
            "Groups missing true structure: "
            f"{groups_missing_true_structure:,}"
        ),
        (
            "Groups true candidate absent from pool: "
            f"{groups_true_not_in_pool:,}"
        ),
        (
            f"Groups with empty pool: "
            f"{groups_empty_pool:,}"
        ),
        "",
        (
            f"Candidate rows: "
            f"{len(reranker_df):,}"
        ),
        (
            f"Queries: "
            f"{reranker_df['query_id'].nunique():,}"
        ),
        (
            f"Positive rows: "
            f"{int(reranker_df['target'].sum()):,}"
        ),
        (
            "Unique candidate structures: "
            f"{reranker_df['candidate_id'].nunique():,}"
        ),
        "",
        (
            "Mean candidates/query: "
            f"{summary['mean_candidates_per_query']:.2f}"
        ),
        (
            "Median candidates/query: "
            f"{summary['median_candidates_per_query']:.1f}"
        ),
        (
            "Min candidates/query: "
            f"{summary['min_candidates_per_query']:.0f}"
        ),
        (
            "Max candidates/query: "
            f"{summary['max_candidates_per_query']:.0f}"
        ),
        "",
        (
            f"Invalid positive-count queries: "
            f"{invalid_query_count:,}"
        ),
        "",
        "Candidate modes:",
    ]

    for (
        mode,
        count,
    ) in sorted(
        candidate_mode_counts.items()
    ):
        summary_lines.append(
            f"  {mode}: {count:,}"
        )

    summary_lines.extend(
        [
            "",
            (
                "Baseline Stage 9 mean-score metrics "
                "on retained queries:"
            ),
            (
                f"  Recall@1:  "
                f"{baseline_top1 * 100:.3f}%"
            ),
            (
                f"  Recall@5:  "
                f"{baseline_top5 * 100:.3f}%"
            ),
            (
                f"  Recall@10: "
                f"{baseline_top10 * 100:.3f}%"
            ),
            (
                f"  Recall@25: "
                f"{baseline_top25 * 100:.3f}%"
            ),
            (
                f"  MRR:       "
                f"{baseline_mrr:.6f}"
            ),
            "",
            (
                f"Elapsed: "
                f"{elapsed_total:.2f}s "
                f"({elapsed_total / 60:.2f} min)"
            ),
        ]
    )

    summary_text = "\n".join(
        summary_lines
    )

    args.summary.write_text(
        summary_text,
        encoding="utf-8",
    )

    # ========================================================
    # Final console output
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "STAGE 10 DATASET BUILD COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        f"\nCandidate rows: "
        f"{len(reranker_df):,}"
    )

    print(
        f"Queries: "
        f"{reranker_df['query_id'].nunique():,}"
    )

    print(
        f"Positive rows: "
        f"{int(reranker_df['target'].sum()):,}"
    )

    print(
        f"Invalid queries: "
        f"{invalid_query_count:,}"
    )

    print(
        f"\nBaseline MRR: "
        f"{baseline_mrr:.6f}"
    )

    print(
        f"Baseline Recall@1: "
        f"{baseline_top1 * 100:.3f}%"
    )

    print(
        f"Baseline Recall@5: "
        f"{baseline_top5 * 100:.3f}%"
    )

    print(
        f"Baseline Recall@10: "
        f"{baseline_top10 * 100:.3f}%"
    )

    print(
        f"Baseline Recall@25: "
        f"{baseline_top25 * 100:.3f}%"
    )

    print(
        f"\nDataset:\n"
        f"{args.output}"
    )

    print(
        f"\nSummary:\n"
        f"{args.summary}"
    )

    print(
        f"\nTotal elapsed: "
        f"{elapsed_total:.2f}s "
        f"({elapsed_total / 60:.2f} min)"
    )


if __name__ == "__main__":
    main()