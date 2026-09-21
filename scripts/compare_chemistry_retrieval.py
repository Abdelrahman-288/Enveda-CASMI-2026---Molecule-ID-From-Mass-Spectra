from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi.chemistry.adducts import (
    mass_error_ppm,
    precursor_to_neutral_mass,
)
from casmi.chemistry.config import load_chemistry_config
from casmi.chemistry.formula import exact_mass_from_smiles

from casmi.retrieval.index import (
    build_spectral_index,
    rank_structures,
    retrieve_scores,
)
from casmi.retrieval.spectral_similarity import (
    BinningConfig,
    normalize_sparse_spectrum,
    spectrum_to_sparse_bins,
)

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

TRAIN_PATH = DATASET_DIR / "train.parquet"

SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "library_match_validation.csv"
)

PREPROCESS_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "data"
    / "spectrum_preprocessing.yaml"
)

CHEMISTRY_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "data"
    / "chemistry.yaml"
)


# =========================================================
# Prototype configuration
# =========================================================

MAX_REFERENCE_SPECTRA = 50_000
MAX_QUERY_SPECTRA = 500

TOP_K = 25

BINNING_CONFIG = BinningConfig(
    bin_width=0.02,
    min_mz=0.0,
    max_mz=1000.0,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def compute_metrics(ranks):
    ranks = np.asarray(
        ranks,
        dtype=float,
    )

    valid = np.isfinite(
        ranks
    )

    recall_1 = np.mean(
        valid & (ranks <= 1)
    )

    recall_5 = np.mean(
        valid & (ranks <= 5)
    )

    recall_10 = np.mean(
        valid & (ranks <= 10)
    )

    recall_25 = np.mean(
        valid & (ranks <= 25)
    )

    reciprocal = np.where(
        valid & (ranks <= 25),
        1.0 / ranks,
        0.0,
    )

    return {
        "Recall@1": recall_1,
        "Recall@5": recall_5,
        "Recall@10": recall_10,
        "Recall@25": recall_25,
        "MRR@25": np.mean(
            reciprocal
        ),
    }


def find_rank(
    ranked_structures,
    true_structure,
):
    for rank, structure in enumerate(
        ranked_structures,
        start=1,
    ):
        if structure == true_structure:
            return rank

    return np.inf


def chemistry_filter_ranked(
    ranked,
    query_precursor_mz,
    query_adduct,
    structure_masses,
    chemistry_config,
):
    """
    Apply Stage 4 mass-filter policy to an already ranked
    list of candidate structures.
    """

    # -----------------------------------------------------
    # Disabled / unknown adduct
    # -----------------------------------------------------

    if (
        query_adduct
        in chemistry_config.disabled_hard_filter_adducts
    ):
        return ranked

    if (
        query_adduct
        not in chemistry_config.hard_filter_adducts
        and query_adduct
        not in chemistry_config.soft_fallback_adducts
    ):
        return ranked

    # -----------------------------------------------------
    # Reconstruct neutral query mass
    # -----------------------------------------------------

    try:
        neutral_mass = (
            precursor_to_neutral_mass(
                query_precursor_mz,
                query_adduct,
                require_trusted=False,
            )
        )

    except (KeyError, ValueError):
        return ranked

    tolerance = (
        chemistry_config.default_tolerance_ppm
    )

    filtered = []

    for structure, score in ranked:

        candidate_mass = structure_masses.get(
            structure
        )

        if candidate_mass is None:
            continue

        error = abs(
            mass_error_ppm(
                neutral_mass,
                candidate_mass,
            )
        )

        if error <= tolerance:
            filtered.append(
                (
                    structure,
                    score,
                )
            )

    # -----------------------------------------------------
    # Hard-filter adducts
    # -----------------------------------------------------

    if (
        query_adduct
        in chemistry_config.hard_filter_adducts
    ):
        return filtered

    # -----------------------------------------------------
    # Soft-filter adducts
    #
    # Use filtered results when available.
    # If filtering removes everything, fall back to the
    # unfiltered spectral ranking.
    # -----------------------------------------------------

    if (
        query_adduct
        in chemistry_config.soft_fallback_adducts
    ):
        if filtered:
            return filtered

        if chemistry_config.fallback_enabled:
            return ranked

    return ranked


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 5 STEP 8 "
        "CHEMISTRY-AWARE RETRIEVAL"
    )

    preprocessing_config = (
        load_preprocessing_config(
            PREPROCESS_CONFIG_PATH,
            profile="baseline",
        )
    )

    chemistry_config = (
        load_chemistry_config(
            CHEMISTRY_CONFIG_PATH
        )
    )

    split = pd.read_csv(
        SPLIT_PATH
    )

    # =====================================================
    # Select prototype reference library
    # =====================================================

    reference_meta = (
        split[
            split["library_role"]
            == "reference"
        ]
        .head(
            MAX_REFERENCE_SPECTRA
        )
        .copy()
    )

    reference_structures = set(
        reference_meta[
            "inchikey14"
        ]
    )

    query_meta = (
        split[
            split["library_role"]
            == "query"
        ]
        .copy()
    )

    # Only evaluate queries whose true molecule exists
    # inside the prototype reference subset.
    query_meta = query_meta[
        query_meta[
            "inchikey14"
        ].isin(
            reference_structures
        )
    ].head(
        MAX_QUERY_SPECTRA
    )

    print_section(
        "PROTOTYPE DATASET"
    )

    print(
        f"Reference spectra: "
        f"{len(reference_meta):,}"
    )

    print(
        f"Reference structures: "
        f"{reference_meta['inchikey14'].nunique():,}"
    )

    print(
        f"Queries: "
        f"{len(query_meta):,}"
    )

    # =====================================================
    # Candidate structure masses
    # =====================================================

    print(
        "\nCalculating candidate exact masses..."
    )

    structure_masses = {}

    structure_smiles = (
        reference_meta[
            [
                "inchikey14",
                "normalized_smiles",
            ]
        ]
        .drop_duplicates(
            "inchikey14"
        )
    )

    for row in structure_smiles.itertuples(
        index=False
    ):
        try:
            structure_masses[
                row.inchikey14
            ] = exact_mass_from_smiles(
                row.normalized_smiles
            )
        except ValueError:
            pass

    print(
        f"Candidate masses calculated: "
        f"{len(structure_masses):,}"
    )

    # =====================================================
    # Load required spectral rows
    # =====================================================

    needed_rows = set(
        reference_meta[
            "row_index"
        ].astype(int)
    )

    needed_rows.update(
        query_meta[
            "row_index"
        ].astype(int)
    )

    print(
        "\nLoading required spectra..."
    )

    columns = [
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_mz",
    ]

    parquet_file = pq.ParquetFile(
        TRAIN_PATH
    )

    spectra_by_row = {}

    global_index = 0

    for batch in parquet_file.iter_batches(
        columns=columns,
        batch_size=2048,
    ):
        df = batch.to_pandas()

        for row in df.itertuples(
            index=False
        ):

            if global_index in needed_rows:

                mzs, intensities = (
                    preprocess_spectrum(
                        row.ms2_mzs,
                        row.ms2_normalized_intensities,
                        precursor_mz=row.precursor_mz,
                        config=preprocessing_config,
                    )
                )

                sparse = (
                    spectrum_to_sparse_bins(
                        mzs,
                        intensities,
                        BINNING_CONFIG,
                    )
                )

                sparse = (
                    normalize_sparse_spectrum(
                        sparse
                    )
                )

                spectra_by_row[
                    global_index
                ] = sparse

            global_index += 1

            if (
                len(spectra_by_row)
                >= len(needed_rows)
            ):
                break

        if (
            len(spectra_by_row)
            >= len(needed_rows)
        ):
            break

    print(
        f"Spectra loaded: "
        f"{len(spectra_by_row):,}"
    )

    # =====================================================
    # Build sparse index
    # =====================================================

    print(
        "\nBuilding sparse spectral index..."
    )

    n_bins = int(
        (
            BINNING_CONFIG.max_mz
            - BINNING_CONFIG.min_mz
        )
        / BINNING_CONFIG.bin_width
    )

    reference_spectra = []
    reference_labels = []
    reference_row_ids = []

    for row in reference_meta.itertuples(
        index=False
    ):

        reference_spectra.append(
            spectra_by_row[
                int(row.row_index)
            ]
        )

        reference_labels.append(
            row.inchikey14
        )

        reference_row_ids.append(
            int(row.row_index)
        )

    index = build_spectral_index(
        spectra=reference_spectra,
        structure_labels=reference_labels,
        row_ids=reference_row_ids,
        n_bins=n_bins,
    )

    print(
        f"Index shape: "
        f"{index.matrix.shape}"
    )

    print(
        f"Index non-zero values: "
        f"{index.matrix.nnz:,}"
    )

    # =====================================================
    # Retrieval comparison
    # =====================================================

    spectral_ranks = []
    chemistry_ranks = []

    adduct_counts = {}

    chemistry_removed_true = 0

    print(
        "\nRunning retrieval comparison..."
    )

    for query_number, row in enumerate(
        query_meta.itertuples(index=False),
        start=1,
    ):

        query = spectra_by_row[
            int(row.row_index)
        ]

        scores = retrieve_scores(
            index,
            query,
        )

        # Full molecule-level spectral ranking.
        ranked = rank_structures(
            scores,
            index.structure_labels,
            top_k=None,
        )

        spectral_structures = [
            structure
            for structure, _
            in ranked
        ]

        spectral_rank = find_rank(
            spectral_structures,
            row.inchikey14,
        )

        spectral_ranks.append(
            spectral_rank
        )

        chemistry_ranked = (
            chemistry_filter_ranked(
                ranked=ranked,
                query_precursor_mz=row.precursor_mz,
                query_adduct=row.adduct,
                structure_masses=structure_masses,
                chemistry_config=chemistry_config,
            )
        )

        chemistry_structures = [
            structure
            for structure, _
            in chemistry_ranked
        ]

        chemistry_rank = find_rank(
            chemistry_structures,
            row.inchikey14,
        )

        chemistry_ranks.append(
            chemistry_rank
        )

        if (
            np.isfinite(spectral_rank)
            and not np.isfinite(
                chemistry_rank
            )
        ):
            chemistry_removed_true += 1

        adduct_counts[
            row.adduct
        ] = (
            adduct_counts.get(
                row.adduct,
                0,
            )
            + 1
        )

        if query_number % 50 == 0:
            print(
                f"Processed "
                f"{query_number:,}"
                f"/{len(query_meta):,}"
            )

    # =====================================================
    # Metrics
    # =====================================================

    spectral_metrics = compute_metrics(
        spectral_ranks
    )

    chemistry_metrics = compute_metrics(
        chemistry_ranks
    )

    print_section(
        "RETRIEVAL COMPARISON"
    )

    print(
        f"{'Metric':15s}"
        f"{'Spectral':>15s}"
        f"{'Chemistry':>15s}"
        f"{'Delta':>15s}"
    )

    print("-" * 60)

    for metric in [
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Recall@25",
        "MRR@25",
    ]:

        spectral_value = (
            spectral_metrics[
                metric
            ]
        )

        chemistry_value = (
            chemistry_metrics[
                metric
            ]
        )

        delta = (
            chemistry_value
            - spectral_value
        )

        if metric == "MRR@25":

            print(
                f"{metric:15s}"
                f"{spectral_value:15.6f}"
                f"{chemistry_value:15.6f}"
                f"{delta:15.6f}"
            )

        else:

            print(
                f"{metric:15s}"
                f"{spectral_value:14.4%}"
                f"{chemistry_value:14.4%}"
                f"{delta:14.4%}"
            )

    print_section(
        "CHEMISTRY SAFETY"
    )

    print(
        "Queries where spectral retrieval contained "
        "the true molecule but chemistry removed it:"
    )

    print(
        f"{chemistry_removed_true:,}"
    )

    print_section(
        "QUERY ADDUCT DISTRIBUTION"
    )

    for adduct, count in sorted(
        adduct_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        print(
            f"{adduct:20s} "
            f"{count:6,d}"
        )

    print_section(
        "STAGE 5 STEP 8 COMPLETE"
    )


if __name__ == "__main__":
    main()