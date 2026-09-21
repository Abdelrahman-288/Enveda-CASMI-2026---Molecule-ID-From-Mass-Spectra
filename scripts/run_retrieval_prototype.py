from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

from casmi.spectra.config import load_preprocessing_config
from casmi.spectra.preprocessing import preprocess_spectrum
from casmi.retrieval.spectral_similarity import (
    BinningConfig,
    spectrum_to_sparse_bins,
    normalize_sparse_spectrum,
)


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


BINNING_CONFIG = BinningConfig(
    bin_width=0.02,
    min_mz=0.0,
    max_mz=1000.0,
)

MAX_REFERENCE_SPECTRA = 50_000
MAX_QUERY_SPECTRA = 500

TOP_K = 25


def print_section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def sparse_dict_to_csr_row(
    sparse_spectrum,
    n_bins,
):
    if not sparse_spectrum:
        return csr_matrix(
            (1, n_bins),
            dtype=np.float32,
        )

    indices = np.fromiter(
        sparse_spectrum.keys(),
        dtype=np.int32,
    )

    data = np.fromiter(
        sparse_spectrum.values(),
        dtype=np.float32,
    )

    indptr = np.array(
        [0, len(indices)],
        dtype=np.int32,
    )

    return csr_matrix(
        (
            data,
            indices,
            indptr,
        ),
        shape=(1, n_bins),
        dtype=np.float32,
    )


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 5 STEP 6 "
        "RETRIEVAL PROTOTYPE"
    )

    preprocessing_config = (
        load_preprocessing_config(
            PREPROCESS_CONFIG_PATH,
            profile="baseline",
        )
    )

    split = pd.read_csv(
        SPLIT_PATH
    )

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

    query_meta = (
        split[
            split["library_role"]
            == "query"
        ]
        .copy()
    )

    # Keep only queries whose true structure exists in
    # our prototype reference subset.
    reference_structures = set(
        reference_meta[
            "inchikey14"
        ]
    )

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
        f"Query spectra: "
        f"{len(query_meta):,}"
    )

    print(
        f"Reference structures: "
        f"{reference_meta['inchikey14'].nunique():,}"
    )

    if len(query_meta) == 0:
        raise RuntimeError(
            "No prototype queries have a matching "
            "reference structure."
        )

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
        "\nLoading required spectra from parquet..."
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

            if len(
                spectra_by_row
            ) >= len(
                needed_rows
            ):
                break

        if len(
            spectra_by_row
        ) >= len(
            needed_rows
        ):
            break

    print(
        f"Spectra loaded: "
        f"{len(spectra_by_row):,}"
    )

    n_bins = int(
        (
            BINNING_CONFIG.max_mz
            - BINNING_CONFIG.min_mz
        )
        / BINNING_CONFIG.bin_width
    )

    print(
        f"Vector bins: "
        f"{n_bins:,}"
    )

    # =====================================================
    # Build reference matrix
    # =====================================================

    print(
        "\nBuilding sparse reference matrix..."
    )

    rows = []

    ref_structure_labels = []

    for row in reference_meta.itertuples(
        index=False
    ):

        sparse = spectra_by_row[
            int(row.row_index)
        ]

        rows.append(
            sparse_dict_to_csr_row(
                sparse,
                n_bins,
            )
        )

        ref_structure_labels.append(
            row.inchikey14
        )

    reference_matrix = csr_matrix(
        np.vstack(
            [
                row.toarray()
                for row in rows
            ]
        ),
        dtype=np.float32,
    )

    print(
        f"Reference matrix shape: "
        f"{reference_matrix.shape}"
    )

    # =====================================================
    # Retrieval
    # =====================================================

    reciprocal_ranks = []
    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    recall_at_25 = []

    print(
        "\nRunning retrieval..."
    )

    for query_index, row in enumerate(
        query_meta.itertuples(index=False),
        start=1,
    ):

        query_sparse = spectra_by_row[
            int(row.row_index)
        ]

        query_vector = (
            sparse_dict_to_csr_row(
                query_sparse,
                n_bins,
            )
        )

        scores = (
            reference_matrix
            @ query_vector.T
        ).toarray().ravel()

        # Collapse spectrum-level results to
        # molecule-level scores using max pooling.
        molecule_scores = {}

        for ref_idx, score in enumerate(
            scores
        ):

            structure = (
                ref_structure_labels[
                    ref_idx
                ]
            )

            previous = (
                molecule_scores.get(
                    structure,
                    -np.inf,
                )
            )

            if score > previous:
                molecule_scores[
                    structure
                ] = float(score)

        ranked = sorted(
            molecule_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        ranked_structures = [
            structure
            for structure, _
            in ranked
        ]

        true_structure = (
            row.inchikey14
        )

        if true_structure in ranked_structures:

            rank = (
                ranked_structures.index(
                    true_structure
                )
                + 1
            )

            reciprocal_rank = (
                1.0 / rank
                if rank <= TOP_K
                else 0.0
            )

        else:
            rank = None
            reciprocal_rank = 0.0

        reciprocal_ranks.append(
            reciprocal_rank
        )

        recall_at_1.append(
            rank is not None
            and rank <= 1
        )

        recall_at_5.append(
            rank is not None
            and rank <= 5
        )

        recall_at_10.append(
            rank is not None
            and rank <= 10
        )

        recall_at_25.append(
            rank is not None
            and rank <= 25
        )

        if (
            query_index % 50
            == 0
        ):
            print(
                f"Processed "
                f"{query_index:,}"
                f"/{len(query_meta):,}"
            )

    print_section(
        "RETRIEVAL METRICS"
    )

    print(
        f"Queries evaluated: "
        f"{len(query_meta):,}"
    )

    print(
        f"Recall@1:  "
        f"{np.mean(recall_at_1):.4%}"
    )

    print(
        f"Recall@5:  "
        f"{np.mean(recall_at_5):.4%}"
    )

    print(
        f"Recall@10: "
        f"{np.mean(recall_at_10):.4%}"
    )

    print(
        f"Recall@25: "
        f"{np.mean(recall_at_25):.4%}"
    )

    print(
        f"MRR@25:    "
        f"{np.mean(reciprocal_ranks):.6f}"
    )

    print_section(
        "STAGE 5 STEP 6 COMPLETE"
    )


if __name__ == "__main__":
    main()