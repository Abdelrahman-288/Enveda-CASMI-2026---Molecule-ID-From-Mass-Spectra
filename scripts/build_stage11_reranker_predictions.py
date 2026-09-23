from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRanker

from casmi.chemistry.adducts import (
    is_supported_adduct,
    is_trusted_for_mass_filter,
    precursor_to_neutral_mass,
    mass_error_ppm,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TEST_PATH = (
    DATASET_DIR
    / "test.parquet"
)

STAGE11_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage11"
)

MOLECULE_CANDIDATES_PATH = (
    STAGE11_DIR
    / "test_molecule_candidates.parquet"
)

SPECTRUM_CANDIDATES_PATH = (
    STAGE11_DIR
    / "test_spectrum_candidates.parquet"
)

STRUCTURE_MAP_PATH = (
    STAGE11_DIR
    / "test_candidate_structures.parquet"
)

SPECTRUM_MAP_PATH = (
    STAGE11_DIR
    / "test_spectrum_embedding_map.parquet"
)

MOLECULE_EMBEDDINGS_PATH = (
    STAGE11_DIR
    / "test_candidate_embeddings.npy"
)

SPECTRUM_EMBEDDINGS_PATH = (
    STAGE11_DIR
    / "test_spectrum_embeddings.npy"
)

COMBINED_VARIANTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "stage7_combined_candidate_mass_variants.parquet"
)

RERANKER_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "stage10"
    / "best_reranker.json"
)

FEATURES_OUTPUT_PATH = (
    STAGE11_DIR
    / "test_reranker_features.parquet"
)

PREDICTIONS_OUTPUT_PATH = (
    STAGE11_DIR
    / "test_ranked_candidates.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage11_reranker_predictions_summary.txt"
)


# ============================================================
# Exact Stage 10 model features
# ============================================================

MODEL_FEATURES = [
    "mean_score",
    "max_score",
    "top2_mean_score",
    "mass_error_ppm",
    "abs_mass_error_ppm",
    "num_supporting_spectra",
    "retrieval_rank",
    "similarity_rank",
    "neural_rank",
    "score_std",
    "score_range",
]


# ============================================================
# Ranking helpers
# ============================================================

def scores_to_ranks(
    scores,
):
    """
    Exact Stage 10 rank conversion.

    Higher scores are better.

    Stable sorting is important because Stage 10 used:

        np.argsort(-scores, kind="stable")
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


# ============================================================
# Exact mass lookup
# ============================================================

def build_exact_mass_lookup(
    variants,
):
    """
    Exact Stage 10 behavior.

    Map each InChIKey14 to all finite positive exact-mass
    variants.
    """

    lookup = {}

    for (
        inchikey14,
        group,
    ) in variants.groupby(
        "inchikey14",
        sort=False,
    ):

        masses = (
            pd.to_numeric(
                group["exact_mass"],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64,
            )
        )

        masses = masses[
            np.isfinite(masses)
            & (masses > 0)
        ]

        if len(masses) == 0:
            continue

        lookup[
            str(inchikey14)
        ] = np.unique(
            masses
        )

    return lookup


def calculate_candidate_mass_error(
    *,
    candidate_position,
    candidate_exact_masses,
    group_df,
    availability_numpy,
):
    """
    Exact Stage 10 mass-error feature.

    Mean signed ppm error across spectra where this candidate
    is available.

    For every spectrum, choose the candidate exact-mass
    variant with minimum absolute ppm error.
    """

    if candidate_exact_masses is None:
        return 0.0

    candidate_exact_masses = np.asarray(
        candidate_exact_masses,
        dtype=np.float64,
    )

    candidate_exact_masses = (
        candidate_exact_masses[
            np.isfinite(
                candidate_exact_masses
            )
            & (
                candidate_exact_masses > 0
            )
        ]
    )

    if len(
        candidate_exact_masses
    ) == 0:
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

        except (
            ValueError,
            KeyError,
            ZeroDivisionError,
        ):
            continue

        variant_ppm_values = []

        for exact_mass in (
            candidate_exact_masses
        ):

            try:

                ppm = mass_error_ppm(
                    observed_mass=(
                        observed_neutral_mass
                    ),
                    expected_mass=float(
                        exact_mass
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
                variant_ppm_values.append(
                    float(ppm)
                )

        if not variant_ppm_values:
            continue

        ppm_values.append(
            min(
                variant_ppm_values,
                key=abs,
            )
        )

    if not ppm_values:
        return 0.0

    return float(
        np.mean(
            ppm_values
        )
    )


# ============================================================
# Score aggregation
# ============================================================

def aggregate_candidate_scores(
    score_matrix,
    availability_mask,
):
    """
    Reproduce the score statistics used during Stage 10.

    score_matrix
        [num_spectra, num_candidates]

    availability_mask
        True only when the candidate survives the individual
        spectrum's chemistry filter.
    """

    num_candidates = (
        score_matrix.shape[1]
    )

    mean_scores = np.zeros(
        num_candidates,
        dtype=np.float64,
    )

    max_scores = np.zeros(
        num_candidates,
        dtype=np.float64,
    )

    top2_mean_scores = np.zeros(
        num_candidates,
        dtype=np.float64,
    )

    score_std = np.zeros(
        num_candidates,
        dtype=np.float64,
    )

    score_range = np.zeros(
        num_candidates,
        dtype=np.float64,
    )

    supporting_spectra = np.zeros(
        num_candidates,
        dtype=np.int64,
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

        valid_scores = np.asarray(
            valid_scores,
            dtype=np.float64,
        )

        supporting_spectra[
            candidate_position
        ] = len(
            valid_scores
        )

        if len(
            valid_scores
        ) == 0:
            continue

        mean_scores[
            candidate_position
        ] = float(
            valid_scores.mean()
        )

        max_scores[
            candidate_position
        ] = float(
            valid_scores.max()
        )

        if len(
            valid_scores
        ) == 1:

            top2_mean_scores[
                candidate_position
            ] = float(
                valid_scores[0]
            )

        else:

            # Mean of the two highest scores.
            top2 = np.partition(
                valid_scores,
                -2,
            )[-2:]

            top2_mean_scores[
                candidate_position
            ] = float(
                top2.mean()
            )

        # Stage 10 used population std:
        # torch.std(unbiased=False)
        score_std[
            candidate_position
        ] = float(
            valid_scores.std(
                ddof=0
            )
        )

        score_range[
            candidate_position
        ] = float(
            valid_scores.max()
            - valid_scores.min()
        )

    return (
        mean_scores,
        max_scores,
        top2_mean_scores,
        supporting_spectra,
        score_std,
        score_range,
    )


# ============================================================
# Model loading
# ============================================================

def load_reranker(
    model_path: Path,
):

    model = XGBRanker()

    model.load_model(
        model_path
    )

    return model


def predict_reranker(
    model,
    X,
):
    """
    Predict using the saved Stage 10 best iteration.

    The saved model contains all trees up through the
    early-stopping window, but validation predictions used the
    best iteration.

    Explicit iteration_range guarantees test inference uses the
    same cutoff.
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    try:
        best_iteration = int(
            model.best_iteration
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
    ):
        best_iteration = None

    if best_iteration is not None:

        scores = model.predict(
            X,
            iteration_range=(
                0,
                best_iteration + 1,
            ),
        )

    else:

        scores = model.predict(
            X
        )

    return (
        np.asarray(
            scores,
            dtype=np.float32,
        ),
        best_iteration,
    )


# ============================================================
# Validation
# ============================================================

def validate_required_files():

    required = [
        TEST_PATH,
        MOLECULE_CANDIDATES_PATH,
        SPECTRUM_CANDIDATES_PATH,
        STRUCTURE_MAP_PATH,
        SPECTRUM_MAP_PATH,
        MOLECULE_EMBEDDINGS_PATH,
        SPECTRUM_EMBEDDINGS_PATH,
        COMBINED_VARIANTS_PATH,
        RERANKER_MODEL_PATH,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing required Stage 11.4 files:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 100)
    print(
        "ENVEDA CASMI 2026 - "
        "STAGE 11.4 TEST RERANKER INFERENCE"
    )
    print("=" * 100)

    validate_required_files()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        "\nLoading Stage 11 manifests..."
    )

    test = (
        pd.read_parquet(
            TEST_PATH
        )
        .reset_index(
            drop=True
        )
    )

    test[
        "spectrum_index"
    ] = np.arange(
        len(test),
        dtype=np.int64,
    )

    molecule_candidates = (
        pd.read_parquet(
            MOLECULE_CANDIDATES_PATH
        )
    )

    spectrum_candidates = (
        pd.read_parquet(
            SPECTRUM_CANDIDATES_PATH
        )
    )

    structure_map = (
        pd.read_parquet(
            STRUCTURE_MAP_PATH
        )
    )

    spectrum_map = (
        pd.read_parquet(
            SPECTRUM_MAP_PATH
        )
    )

    print(
        "Molecule-candidate rows: "
        f"{len(molecule_candidates):,}"
    )

    print(
        "Spectrum-candidate rows: "
        f"{len(spectrum_candidates):,}"
    )

    print(
        "Unique candidate structures: "
        f"{len(structure_map):,}"
    )

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    print(
        "\nLoading neural embeddings..."
    )

    molecule_embeddings = np.load(
        MOLECULE_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    spectrum_embeddings = np.load(
        SPECTRUM_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    if molecule_embeddings.shape != (
        len(structure_map),
        128,
    ):
        raise RuntimeError(
            "Candidate embedding shape mismatch."
        )

    if spectrum_embeddings.shape != (
        len(test),
        128,
    ):
        raise RuntimeError(
            "Spectrum embedding shape mismatch."
        )

    # --------------------------------------------------------
    # Load exact-mass variants
    # --------------------------------------------------------

    print(
        "\nLoading exact-mass variants..."
    )

    variants = pd.read_parquet(
        COMBINED_VARIANTS_PATH,
        columns=[
            "inchikey14",
            "exact_mass",
        ],
    )

    exact_mass_lookup = (
        build_exact_mass_lookup(
            variants
        )
    )

    print(
        "Exact-mass lookup structures: "
        f"{len(exact_mass_lookup):,}"
    )

    # --------------------------------------------------------
    # Availability lookup
    # --------------------------------------------------------

    print(
        "\nBuilding spectrum candidate availability..."
    )

    availability_lookup = defaultdict(
        set
    )

    for row in (
        spectrum_candidates[
            [
                "spectrum_index",
                "candidate_structure_index",
            ]
        ]
        .itertuples(
            index=False
        )
    ):

        availability_lookup[
            int(
                row.spectrum_index
            )
        ].add(
            int(
                row.candidate_structure_index
            )
        )

    # --------------------------------------------------------
    # Build reranker rows molecule by molecule
    # --------------------------------------------------------

    print(
        "\nComputing compatibility scores "
        "and Stage 10 features..."
    )

    feature_rows = []

    molecule_ids = (
        molecule_candidates[
            "molecule_id"
        ]
        .drop_duplicates()
        .tolist()
    )

    total_molecules = len(
        molecule_ids
    )

    for molecule_number, molecule_id in enumerate(
        molecule_ids,
        start=1,
    ):

        # ----------------------------------------------------
        # Spectra belonging to this molecule
        # ----------------------------------------------------

        group_df = (
            test[
                test[
                    "molecule_id"
                ]
                == molecule_id
            ]
            .sort_values(
                "spectrum_index",
                kind="stable",
            )
            .reset_index(
                drop=True
            )
        )

        spectrum_indices = (
            group_df[
                "spectrum_index"
            ]
            .to_numpy(
                dtype=np.int64,
            )
        )

        # ----------------------------------------------------
        # Candidate pool belonging to this molecule
        # ----------------------------------------------------

        candidate_df = (
            molecule_candidates[
                molecule_candidates[
                    "molecule_id"
                ]
                == molecule_id
            ]
            .sort_values(
                "candidate_structure_index",
                kind="stable",
            )
            .reset_index(
                drop=True
            )
        )

        candidate_structure_indices = (
            candidate_df[
                "candidate_structure_index"
            ]
            .to_numpy(
                dtype=np.int64,
            )
        )

        num_spectra = len(
            spectrum_indices
        )

        num_candidates = len(
            candidate_structure_indices
        )

        if (
            num_spectra <= 0
            or num_candidates <= 0
        ):
            raise RuntimeError(
                "Empty molecule query encountered: "
                f"{molecule_id}"
            )

        # ----------------------------------------------------
        # Compatibility matrix
        #
        # embeddings are normalized, so dot product = cosine.
        # ----------------------------------------------------

        spectrum_vectors = np.asarray(
            spectrum_embeddings[
                spectrum_indices
            ],
            dtype=np.float32,
        )

        molecule_vectors = np.asarray(
            molecule_embeddings[
                candidate_structure_indices
            ],
            dtype=np.float32,
        )

        score_matrix = (
            spectrum_vectors
            @ molecule_vectors.T
        )

        if score_matrix.shape != (
            num_spectra,
            num_candidates,
        ):
            raise RuntimeError(
                "Unexpected compatibility matrix shape."
            )

        if not np.isfinite(
            score_matrix
        ).all():
            raise RuntimeError(
                "Non-finite compatibility scores."
            )

        # ----------------------------------------------------
        # Exact per-spectrum candidate availability mask
        # ----------------------------------------------------

        availability_mask = np.zeros(
            (
                num_spectra,
                num_candidates,
            ),
            dtype=bool,
        )

        candidate_position_lookup = {
            int(structure_index): position
            for position, structure_index
            in enumerate(
                candidate_structure_indices
            )
        }

        for spectrum_position, spectrum_index in enumerate(
            spectrum_indices
        ):

            available_structures = (
                availability_lookup[
                    int(
                        spectrum_index
                    )
                ]
            )

            for structure_index in (
                available_structures
            ):

                candidate_position = (
                    candidate_position_lookup.get(
                        int(
                            structure_index
                        )
                    )
                )

                if candidate_position is None:
                    continue

                availability_mask[
                    spectrum_position,
                    candidate_position,
                ] = True

        # Every selected molecule candidate must be available
        # in at least one spectrum.
        support_counts = (
            availability_mask.sum(
                axis=0
            )
        )

        if np.any(
            support_counts <= 0
        ):
            raise RuntimeError(
                "Candidate with zero spectrum support "
                f"in molecule {molecule_id}."
            )

        # ----------------------------------------------------
        # Aggregate scores exactly as Stage 10
        # ----------------------------------------------------

        (
            mean_scores,
            max_scores,
            top2_mean_scores,
            supporting_spectra,
            score_std,
            score_range,
        ) = aggregate_candidate_scores(
            score_matrix,
            availability_mask,
        )

        # ----------------------------------------------------
        # Stable ranks
        #
        # Stage 10 mappings:
        #
        # retrieval_rank  <- mean aggregation
        # similarity_rank <- max aggregation
        # neural_rank     <- top2 mean aggregation
        # ----------------------------------------------------

        mean_ranks = scores_to_ranks(
            mean_scores
        )

        max_ranks = scores_to_ranks(
            max_scores
        )

        top2_ranks = scores_to_ranks(
            top2_mean_scores
        )

        # ----------------------------------------------------
        # Candidate feature rows
        # ----------------------------------------------------

        for candidate_position, candidate_row in enumerate(
            candidate_df.itertuples(
                index=False
            )
        ):

            candidate_id = str(
                candidate_row.candidate_inchikey14
            )

            exact_masses = (
                exact_mass_lookup.get(
                    candidate_id
                )
            )

            signed_mass_error = (
                calculate_candidate_mass_error(
                    candidate_position=(
                        candidate_position
                    ),
                    candidate_exact_masses=(
                        exact_masses
                    ),
                    group_df=group_df,
                    availability_numpy=(
                        availability_mask
                    ),
                )
            )

            row = {
                "molecule_id": (
                    str(
                        molecule_id
                    )
                ),
                "candidate_inchikey14": (
                    candidate_id
                ),
                "candidate_smiles": (
                    str(
                        candidate_row
                        .candidate_smiles
                    )
                ),
                "candidate_structure_index": (
                    int(
                        candidate_row
                        .candidate_structure_index
                    )
                ),
                "num_spectra": (
                    int(
                        num_spectra
                    )
                ),
                "mean_score": (
                    float(
                        mean_scores[
                            candidate_position
                        ]
                    )
                ),
                "max_score": (
                    float(
                        max_scores[
                            candidate_position
                        ]
                    )
                ),
                "top2_mean_score": (
                    float(
                        top2_mean_scores[
                            candidate_position
                        ]
                    )
                ),
                "mass_error_ppm": (
                    float(
                        signed_mass_error
                    )
                ),
                "abs_mass_error_ppm": (
                    float(
                        abs(
                            signed_mass_error
                        )
                    )
                ),
                "num_supporting_spectra": (
                    float(
                        supporting_spectra[
                            candidate_position
                        ]
                    )
                ),
                "retrieval_rank": (
                    float(
                        mean_ranks[
                            candidate_position
                        ]
                    )
                ),
                "similarity_rank": (
                    float(
                        max_ranks[
                            candidate_position
                        ]
                    )
                ),
                "neural_rank": (
                    float(
                        top2_ranks[
                            candidate_position
                        ]
                    )
                ),
                "score_std": (
                    float(
                        score_std[
                            candidate_position
                        ]
                    )
                ),
                "score_range": (
                    float(
                        score_range[
                            candidate_position
                        ]
                    )
                ),
            }

            feature_rows.append(
                row
            )

        if (
            molecule_number % 25 == 0
            or molecule_number
            == total_molecules
        ):

            print(
                "  Processed "
                f"{molecule_number:,}"
                "/"
                f"{total_molecules:,} molecules"
            )

    # --------------------------------------------------------
    # Feature dataframe
    # --------------------------------------------------------

    features_df = pd.DataFrame(
        feature_rows
    )

    if len(
        features_df
    ) != len(
        molecule_candidates
    ):

        raise RuntimeError(
            "Stage 11 feature row count mismatch.\n"
            f"Features: {len(features_df):,}\n"
            f"Candidates: {len(molecule_candidates):,}"
        )

    # --------------------------------------------------------
    # Validate exact 11 model features
    # --------------------------------------------------------

    missing_features = [
        column
        for column in MODEL_FEATURES
        if column
        not in features_df.columns
    ]

    if missing_features:

        raise RuntimeError(
            "Missing reranker features: "
            f"{missing_features}"
        )

    X = (
        features_df[
            MODEL_FEATURES
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    if X.shape != (
        len(features_df),
        11,
    ):
        raise RuntimeError(
            "Unexpected reranker feature matrix shape: "
            f"{X.shape}"
        )

    if not np.isfinite(
        X
    ).all():
        raise RuntimeError(
            "Reranker feature matrix contains "
            "non-finite values."
        )

    print(
        "\nReranker feature matrix: "
        f"{X.shape}"
    )

    print(
        "Feature order:"
    )

    for index, feature in enumerate(
        MODEL_FEATURES,
        start=1,
    ):
        print(
            f"  {index:2d}. "
            f"{feature}"
        )

    # --------------------------------------------------------
    # Save features
    # --------------------------------------------------------

    features_df.to_parquet(
        FEATURES_OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Load and run Stage 10 model
    # --------------------------------------------------------

    print(
        "\nLoading Stage 10 XGBoost reranker..."
    )

    reranker = load_reranker(
        RERANKER_MODEL_PATH
    )

    predictions, best_iteration = (
        predict_reranker(
            reranker,
            X,
        )
    )

    if len(
        predictions
    ) != len(
        features_df
    ):
        raise RuntimeError(
            "Prediction row count mismatch."
        )

    if not np.isfinite(
        predictions
    ).all():
        raise RuntimeError(
            "Stage 10 predictions contain "
            "non-finite values."
        )

    print(
        "Best iteration: "
        f"{best_iteration}"
    )

    print(
        "Predictions: "
        f"{len(predictions):,}"
    )

    # --------------------------------------------------------
    # Add Stage 10 scores
    # --------------------------------------------------------

    predictions_df = (
        features_df.copy()
    )

    predictions_df[
        "stage9_score"
    ] = (
        predictions_df[
            "mean_score"
        ]
    )

    predictions_df[
        "stage10_score"
    ] = predictions

    # --------------------------------------------------------
    # Final rank within each molecule
    # --------------------------------------------------------

    ranked_groups = []

    for molecule_id, group in (
        predictions_df.groupby(
            "molecule_id",
            sort=False,
        )
    ):

        group = (
            group.copy()
            .sort_values(
                [
                    "stage10_score",
                    "mean_score",
                    "candidate_structure_index",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
                kind="stable",
            )
            .reset_index(
                drop=True
            )
        )

        group[
            "final_rank"
        ] = np.arange(
            1,
            len(group) + 1,
            dtype=np.int64,
        )

        ranked_groups.append(
            group
        )

    ranked_df = pd.concat(
        ranked_groups,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if (
        ranked_df[
            "molecule_id"
        ].nunique()
        != 400
    ):
        raise RuntimeError(
            "Expected 400 test molecule IDs."
        )

    duplicate_pairs = (
        ranked_df.duplicated(
            [
                "molecule_id",
                "candidate_inchikey14",
            ]
        )
    )

    if duplicate_pairs.any():
        raise RuntimeError(
            "Duplicate evaluation identities found "
            "within molecule candidate rankings."
        )

    rank_min = (
        ranked_df.groupby(
            "molecule_id"
        )[
            "final_rank"
        ]
        .min()
    )

    if not (
        rank_min == 1
    ).all():
        raise RuntimeError(
            "Every molecule must begin at rank 1."
        )

    ranked_df.to_parquet(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    group_sizes = (
        ranked_df.groupby(
            "molecule_id"
        )
        .size()
        .to_numpy(
            dtype=np.int64
        )
    )

    top1_scores = (
        ranked_df[
            ranked_df[
                "final_rank"
            ]
            == 1
        ][
            "stage10_score"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    top25_rows = int(
        np.minimum(
            group_sizes,
            25,
        ).sum()
    )

    support_distribution = (
        ranked_df[
            "num_supporting_spectra"
        ]
        .describe()
    )

    # --------------------------------------------------------
    # Print top examples
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("TOP-1 PREDICTION PREVIEW")
    print("=" * 100)

    preview = (
        ranked_df[
            ranked_df[
                "final_rank"
            ]
            == 1
        ][
            [
                "molecule_id",
                "candidate_inchikey14",
                "candidate_smiles",
                "stage10_score",
                "mean_score",
                "mass_error_ppm",
                "num_supporting_spectra",
            ]
        ]
        .head(15)
    )

    print(
        preview.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_lines = [
        (
            "ENVEDA CASMI 2026 - "
            "STAGE 11.4 TEST RERANKER INFERENCE"
        ),
        "",
        (
            "Test molecules: "
            f"{ranked_df['molecule_id'].nunique():,}"
        ),
        (
            "Candidate rows: "
            f"{len(ranked_df):,}"
        ),
        (
            "Unique candidate structures: "
            f"{ranked_df['candidate_inchikey14'].nunique():,}"
        ),
        (
            "Feature count: "
            f"{len(MODEL_FEATURES)}"
        ),
        (
            "Feature matrix shape: "
            f"{X.shape}"
        ),
        (
            "XGBoost best iteration: "
            f"{best_iteration}"
        ),
        "",
        "CANDIDATES PER MOLECULE",
        (
            "Minimum: "
            f"{group_sizes.min()}"
        ),
        (
            "Median: "
            f"{np.median(group_sizes):.2f}"
        ),
        (
            "Mean: "
            f"{group_sizes.mean():.2f}"
        ),
        (
            "Maximum: "
            f"{group_sizes.max()}"
        ),
        "",
        "TOP-1 STAGE 10 SCORE",
        (
            "Minimum: "
            f"{top1_scores.min():.6f}"
        ),
        (
            "Median: "
            f"{np.median(top1_scores):.6f}"
        ),
        (
            "Mean: "
            f"{top1_scores.mean():.6f}"
        ),
        (
            "Maximum: "
            f"{top1_scores.max():.6f}"
        ),
        "",
        (
            "Rows that will enter top-25 submission: "
            f"{top25_rows:,}"
        ),
        "",
        (
            "Supporting spectra minimum: "
            f"{support_distribution['min']:.0f}"
        ),
        (
            "Supporting spectra median: "
            f"{support_distribution['50%']:.2f}"
        ),
        (
            "Supporting spectra mean: "
            f"{support_distribution['mean']:.2f}"
        ),
        (
            "Supporting spectra maximum: "
            f"{support_distribution['max']:.0f}"
        ),
        "",
        "MODEL FEATURES",
        *[
            f"{index}. {feature}"
            for index, feature
            in enumerate(
                MODEL_FEATURES,
                start=1,
            )
        ],
        "",
        (
            "Feature output: "
            f"{FEATURES_OUTPUT_PATH}"
        ),
        (
            "Ranked predictions: "
            f"{PREDICTIONS_OUTPUT_PATH}"
        ),
    ]

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("STAGE 11.4 SUMMARY")
    print("=" * 100)

    for line in summary_lines:
        print(
            line
        )

    print()
    print("=" * 100)
    print("STAGE 11.4 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()