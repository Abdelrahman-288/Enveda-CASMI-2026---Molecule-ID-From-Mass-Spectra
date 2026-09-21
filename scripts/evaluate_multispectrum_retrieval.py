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

TRAIN_PATH = (
    DATASET_DIR
    / "train.parquet"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "multispectrum_validation_split.csv"
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
# Prototype settings
# =========================================================

MAX_REFERENCE_SPECTRA = 50_000
MAX_QUERY_MOLECULES = 300

BINNING_CONFIG = BinningConfig(
    bin_width=0.02,
    min_mz=0.0,
    max_mz=1000.0,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def collapse_reference_scores(
    scores,
    structure_labels,
):
    """
    Convert spectrum-level similarity scores into
    molecule-level scores using max pooling over reference
    spectra belonging to the same structure.
    """

    molecule_scores = {}

    for structure, score in zip(
        structure_labels,
        scores,
    ):
        score = float(score)

        previous = molecule_scores.get(
            structure
        )

        if (
            previous is None
            or score > previous
        ):
            molecule_scores[
                structure
            ] = score

    return molecule_scores


def rank_true_structure(
    molecule_scores,
    true_structure,
):
    """
    Return 1-based rank of the true structure.

    Returns np.inf if the structure does not appear.
    """

    ranked = sorted(
        molecule_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for rank, (
        structure,
        _,
    ) in enumerate(
        ranked,
        start=1,
    ):
        if structure == true_structure:
            return rank

    return np.inf


def compute_metrics(ranks):
    """
    Compute Recall@K and MRR@25.
    """

    ranks = np.asarray(
        ranks,
        dtype=float,
    )

    valid = np.isfinite(
        ranks
    )

    result = {}

    for k in [
        1,
        5,
        10,
        25,
    ]:
        result[
            f"Recall@{k}"
        ] = float(
            np.mean(
                valid
                & (ranks <= k)
            )
        )

    result["MRR@25"] = float(
        np.mean(
            np.where(
                valid
                & (ranks <= 25),
                1.0 / ranks,
                0.0,
            )
        )
    )

    return result


def chemistry_filter(
    molecule_scores,
    query_group,
    structure_masses,
    chemistry_config,
):
    """
    Apply Stage 4 mass-filter policy.

    Policy:
    - Prefer trusted hard-filter adduct observations.
    - If no hard observations exist, use soft-fallback adducts.
    - A candidate survives if it matches at least one selected
      precursor/adduct observation within the configured ppm tolerance.
    - If soft filtering removes everything, fall back to the
      unfiltered spectral ranking.
    """

    hard_observations = []
    soft_observations = []

    for row in query_group.itertuples(
        index=False
    ):
        adduct = row.adduct

        if (
            adduct
            in chemistry_config.disabled_hard_filter_adducts
        ):
            continue

        try:
            neutral_mass = (
                precursor_to_neutral_mass(
                    row.precursor_mz,
                    adduct,
                    require_trusted=False,
                )
            )

        except (
            KeyError,
            ValueError,
        ):
            continue

        observation = (
            neutral_mass,
            adduct,
        )

        if (
            adduct
            in chemistry_config.hard_filter_adducts
        ):
            hard_observations.append(
                observation
            )

        elif (
            adduct
            in chemistry_config.soft_fallback_adducts
        ):
            soft_observations.append(
                observation
            )

    # Prefer hard constraints.
    if hard_observations:
        observations = hard_observations
        use_fallback = False

    elif soft_observations:
        observations = soft_observations
        use_fallback = True

    else:
        return molecule_scores

    tolerance = (
        chemistry_config
        .default_tolerance_ppm
    )

    filtered = {}

    for structure, score in (
        molecule_scores.items()
    ):
        candidate_mass = (
            structure_masses.get(
                structure
            )
        )

        if candidate_mass is None:
            continue

        compatible = False

        for neutral_mass, _ in observations:

            error = abs(
                mass_error_ppm(
                    neutral_mass,
                    candidate_mass,
                )
            )

            if error <= tolerance:
                compatible = True
                break

        if compatible:
            filtered[
                structure
            ] = score

    if filtered:
        return filtered

    if (
        use_fallback
        and chemistry_config.fallback_enabled
    ):
        return molecule_scores

    return filtered


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 5 STEP 9D "
        "LEAKAGE-FREE MULTI-SPECTRUM RETRIEVAL "
        "WITH MAX/MEAN CHEMISTRY COMPARISON"
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

    # =====================================================
    # Load split
    # =====================================================

    print(
        "\nLoading validation split..."
    )

    split = pd.read_csv(
        SPLIT_PATH
    )

    # =====================================================
    # Reference subset
    # =====================================================

    reference_meta = (
        split[
            split[
                "multispectrum_role"
            ] == "reference"
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

    # =====================================================
    # Query subset
    # =====================================================

    query_meta = split[
        (
            split[
                "multispectrum_role"
            ] == "query"
        )
        &
        (
            split[
                "inchikey14"
            ].isin(
                reference_structures
            )
        )
    ].copy()

    # Preserve complete molecule groups.
    available_query_structures = (
        query_meta[
            "inchikey14"
        ]
        .drop_duplicates()
        .head(
            MAX_QUERY_MOLECULES
        )
        .tolist()
    )

    query_meta = query_meta[
        query_meta[
            "inchikey14"
        ].isin(
            available_query_structures
        )
    ].copy()

    query_groups = list(
        query_meta.groupby(
            "inchikey14",
            sort=False,
        )
    )

    # =====================================================
    # Leakage checks
    # =====================================================

    print_section(
        "LEAKAGE-FREE PROTOTYPE"
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
        f"Query molecules: "
        f"{len(query_groups):,}"
    )

    print(
        f"Query spectra: "
        f"{len(query_meta):,}"
    )

    query_ids = set(
        query_meta[
            "row_index"
        ].astype(int)
    )

    reference_ids = set(
        reference_meta[
            "row_index"
        ].astype(int)
    )

    overlap = (
        query_ids
        & reference_ids
    )

    print(
        "Query/reference row overlap: "
        f"{len(overlap):,}"
    )

    if overlap:
        raise RuntimeError(
            "Evaluation leakage detected."
        )

    # =====================================================
    # Load required spectra
    # =====================================================

    needed_rows = set(
        reference_ids
    )

    needed_rows.update(
        query_ids
    )

    print(
        "\nLoading required spectra..."
    )

    parquet_file = (
        pq.ParquetFile(
            TRAIN_PATH
        )
    )

    columns = [
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_mz",
    ]

    spectra_by_row = {}

    global_index = 0

    for batch in (
        parquet_file.iter_batches(
            columns=columns,
            batch_size=2048,
        )
    ):
        df = batch.to_pandas()

        for row in df.itertuples(
            index=False
        ):

            if (
                global_index
                in needed_rows
            ):
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

    if (
        len(spectra_by_row)
        != len(needed_rows)
    ):
        raise RuntimeError(
            "Not all requested spectra were loaded."
        )

    # =====================================================
    # Build reference index
    # =====================================================

    n_bins = int(
        (
            BINNING_CONFIG.max_mz
            - BINNING_CONFIG.min_mz
        )
        /
        BINNING_CONFIG.bin_width
    )

    reference_spectra = []
    reference_labels = []
    reference_row_ids = []

    for row in (
        reference_meta.itertuples(
            index=False
        )
    ):
        row_id = int(
            row.row_index
        )

        reference_spectra.append(
            spectra_by_row[
                row_id
            ]
        )

        reference_labels.append(
            row.inchikey14
        )

        reference_row_ids.append(
            row_id
        )

    print(
        "\nBuilding sparse index..."
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
        f"Index nnz: "
        f"{index.matrix.nnz:,}"
    )

    # =====================================================
    # Candidate exact masses
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

    for row in (
        structure_smiles.itertuples(
            index=False
        )
    ):
        try:
            structure_masses[
                row.inchikey14
            ] = (
                exact_mass_from_smiles(
                    row.normalized_smiles
                )
            )

        except ValueError:
            pass

    print(
        "Candidate masses calculated: "
        f"{len(structure_masses):,}"
    )

    # =====================================================
    # Evaluation
    # =====================================================

    single_ranks = []
    multi_max_ranks = []
    multi_mean_ranks = []

    multi_max_chemistry_ranks = []
    multi_mean_chemistry_ranks = []

    query_spectrum_counts = []

    print(
        "\nRunning leakage-free retrieval..."
    )

    for group_number, (
        true_structure,
        group,
    ) in enumerate(
        query_groups,
        start=1,
    ):
        per_spectrum_scores = []

        for row in group.itertuples(
            index=False
        ):
            query_spectrum = (
                spectra_by_row[
                    int(
                        row.row_index
                    )
                ]
            )

            scores = retrieve_scores(
                index,
                query_spectrum,
            )

            per_spectrum_scores.append(
                scores
            )

        score_matrix = np.vstack(
            per_spectrum_scores
        )

        query_spectrum_counts.append(
            score_matrix.shape[0]
        )

        # =================================================
        # 1. Single-spectrum baseline
        # =================================================

        single_spectrum_scores = (
            score_matrix[0]
        )

        single_molecules = (
            collapse_reference_scores(
                single_spectrum_scores,
                index.structure_labels,
            )
        )

        single_ranks.append(
            rank_true_structure(
                single_molecules,
                true_structure,
            )
        )

        # =================================================
        # 2. Multi-spectrum MAX
        # =================================================

        spectrum_max_scores = np.max(
            score_matrix,
            axis=0,
        )

        max_molecules = (
            collapse_reference_scores(
                spectrum_max_scores,
                index.structure_labels,
            )
        )

        multi_max_ranks.append(
            rank_true_structure(
                max_molecules,
                true_structure,
            )
        )

        # =================================================
        # 3. Multi-spectrum MEAN
        # =================================================

        spectrum_mean_scores = np.mean(
            score_matrix,
            axis=0,
        )

        mean_molecules = (
            collapse_reference_scores(
                spectrum_mean_scores,
                index.structure_labels,
            )
        )

        multi_mean_ranks.append(
            rank_true_structure(
                mean_molecules,
                true_structure,
            )
        )

        # =================================================
        # 4. Multi-Max + Chemistry
        # =================================================

        max_chemistry_molecules = (
            chemistry_filter(
                molecule_scores=max_molecules,
                query_group=group,
                structure_masses=structure_masses,
                chemistry_config=chemistry_config,
            )
        )

        multi_max_chemistry_ranks.append(
            rank_true_structure(
                max_chemistry_molecules,
                true_structure,
            )
        )

        # =================================================
        # 5. Multi-Mean + Chemistry
        # =================================================

        mean_chemistry_molecules = (
            chemistry_filter(
                molecule_scores=mean_molecules,
                query_group=group,
                structure_masses=structure_masses,
                chemistry_config=chemistry_config,
            )
        )

        multi_mean_chemistry_ranks.append(
            rank_true_structure(
                mean_chemistry_molecules,
                true_structure,
            )
        )

        if (
            group_number % 50
            == 0
        ):
            print(
                f"Processed "
                f"{group_number:,}"
                f"/{len(query_groups):,}"
            )

    # =====================================================
    # Metrics
    # =====================================================

    results = {
        "Single": compute_metrics(
            single_ranks
        ),
        "Multi-Max": compute_metrics(
            multi_max_ranks
        ),
        "Multi-Mean": compute_metrics(
            multi_mean_ranks
        ),
        "Multi-Max+Chem": compute_metrics(
            multi_max_chemistry_ranks
        ),
        "Multi-Mean+Chem": compute_metrics(
            multi_mean_chemistry_ranks
        ),
    }

    print_section(
        "LEAKAGE-FREE MULTI-SPECTRUM RESULTS"
    )

    print(
        f"{'Method':20s}"
        f"{'R@1':>10s}"
        f"{'R@5':>10s}"
        f"{'R@10':>10s}"
        f"{'R@25':>10s}"
        f"{'MRR@25':>12s}"
    )

    print(
        "-" * 72
    )

    for method, result in (
        results.items()
    ):
        print(
            f"{method:20s}"
            f"{result['Recall@1']:10.2%}"
            f"{result['Recall@5']:10.2%}"
            f"{result['Recall@10']:10.2%}"
            f"{result['Recall@25']:10.2%}"
            f"{result['MRR@25']:12.6f}"
        )

    print(
        "\nAverage query spectra per molecule: "
        f"{np.mean(query_spectrum_counts):.2f}"
    )

    # =====================================================
    # Pairwise improvement counts
    # =====================================================

    single_array = np.asarray(
        single_ranks,
        dtype=float,
    )

    max_array = np.asarray(
        multi_max_ranks,
        dtype=float,
    )

    mean_array = np.asarray(
        multi_mean_ranks,
        dtype=float,
    )

    max_chemistry_array = np.asarray(
        multi_max_chemistry_ranks,
        dtype=float,
    )

    mean_chemistry_array = np.asarray(
        multi_mean_chemistry_ranks,
        dtype=float,
    )

    print_section(
        "PAIRWISE IMPROVEMENT COUNTS"
    )

    print(
        "Multi-Max better than Single: "
        f"{np.sum(max_array < single_array):,}"
    )

    print(
        "Multi-Max worse than Single:  "
        f"{np.sum(max_array > single_array):,}"
    )

    print()

    print(
        "Multi-Mean better than Single: "
        f"{np.sum(mean_array < single_array):,}"
    )

    print(
        "Multi-Mean worse than Single:  "
        f"{np.sum(mean_array > single_array):,}"
    )

    print()

    print(
        "Chemistry better than Multi-Max:  "
        f"{np.sum(max_chemistry_array < max_array):,}"
    )

    print(
        "Chemistry worse than Multi-Max:   "
        f"{np.sum(max_chemistry_array > max_array):,}"
    )

    print()

    print(
        "Chemistry better than Multi-Mean: "
        f"{np.sum(mean_chemistry_array < mean_array):,}"
    )

    print(
        "Chemistry worse than Multi-Mean:  "
        f"{np.sum(mean_chemistry_array > mean_array):,}"
    )

    # =====================================================
    # Direct chemistry comparison
    # =====================================================

    print_section(
        "CHEMISTRY AGGREGATION COMPARISON"
    )

    max_chem_metrics = results[
        "Multi-Max+Chem"
    ]

    mean_chem_metrics = results[
        "Multi-Mean+Chem"
    ]

    print(
        f"Multi-Max+Chem MRR@25:  "
        f"{max_chem_metrics['MRR@25']:.6f}"
    )

    print(
        f"Multi-Mean+Chem MRR@25: "
        f"{mean_chem_metrics['MRR@25']:.6f}"
    )

    mrr_difference = (
        max_chem_metrics["MRR@25"]
        - mean_chem_metrics["MRR@25"]
    )

    print(
        "MRR difference "
        "(Max+Chem - Mean+Chem): "
        f"{mrr_difference:+.6f}"
    )

    print_section(
        "STAGE 5 STEP 9D COMPLETE"
    )


if __name__ == "__main__":
    main()