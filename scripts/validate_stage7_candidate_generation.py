from pathlib import Path
import time

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from casmi.chemistry.adducts import (
    is_supported_adduct,
    is_trusted_for_mass_filter,
)

from casmi.retrieval.candidate_generation import (
    AdductAwareCandidateGenerator,
)

from casmi.retrieval.candidate_mass_index import (
    CandidateMassIndex,
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

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage7_candidate_generation_summary.txt"
)


# =========================================================
# Configuration
# =========================================================

SEED = 42

MAX_ROWS = 50_000

TOLERANCES = [
    5.0,
    10.0,
    20.0,
]


def exact_mass_from_smiles(
    smiles: str,
) -> float:

    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        return np.nan

    return float(
        Descriptors.ExactMolWt(
            molecule
        )
    )


def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 7 STEP 4 "
        "REAL CANDIDATE-GENERATION VALIDATION"
    )

    start_time = time.time()

    # =====================================================
    # Load training data
    # =====================================================

    print(
        "\nLoading training metadata..."
    )

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
            "precursor_mz",
            "adduct",
        ],
    )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    # =====================================================
    # Keep supported + trusted adducts
    # =====================================================

    supported_mask = (
        train[
            "adduct"
        ]
        .map(
            is_supported_adduct
        )
    )

    trusted_mask = (
        train[
            "adduct"
        ]
        .map(
            is_trusted_for_mass_filter
        )
    )

    valid_precursor_mask = (
        train[
            "precursor_mz"
        ]
        .notna()
        &
        (
            train[
                "precursor_mz"
            ]
            > 0
        )
    )

    eligible = train[
        supported_mask
        & trusted_mask
        & valid_precursor_mask
    ].copy()

    print(
        f"Trusted eligible rows: "
        f"{len(eligible):,}"
    )

    # =====================================================
    # Deterministic sample
    # =====================================================

    if len(
        eligible
    ) > MAX_ROWS:

        eligible = (
            eligible.sample(
                n=MAX_ROWS,
                random_state=SEED,
            )
            .reset_index(
                drop=True
            )
        )

    print(
        f"Rows evaluated: "
        f"{len(eligible):,}"
    )

    # =====================================================
    # Build variant table
    # =====================================================

    print(
        "\nBuilding candidate variant index..."
    )

    variants = (
        train[
            [
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    variants[
        "exact_mass"
    ] = (
        variants[
            "normalized_smiles"
        ]
        .map(
            exact_mass_from_smiles
        )
    )

    variants = variants.dropna(
        subset=[
            "exact_mass",
        ]
    )

    variants = (
        variants
        .sort_values(
            [
                "inchikey14",
                "exact_mass",
                "normalized_smiles",
            ]
        )
        .drop_duplicates(
            subset=[
                "inchikey14",
                "exact_mass",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    mass_index = CandidateMassIndex(
        variants
    )

    print(
        f"Indexed structures: "
        f"{mass_index.unique_structure_count:,}"
    )

    print(
        f"Indexed mass variants: "
        f"{mass_index.variant_count:,}"
    )

    # =====================================================
    # Evaluate tolerances
    # =====================================================

    summary_lines = [
        (
            "ENVEDA CASMI 2026 — "
            "STAGE 7 CANDIDATE GENERATION VALIDATION"
        ),
        "",
        f"Rows evaluated: {len(eligible):,}",
        "",
    ]

    for tolerance_ppm in TOLERANCES:

        print(
            "\n"
            + "=" * 90
        )

        print(
            f"TOLERANCE: "
            f"{tolerance_ppm:.1f} ppm"
        )

        print(
            "=" * 90
        )

        generator = (
            AdductAwareCandidateGenerator(
                mass_index=mass_index,
                tolerance_ppm=(
                    tolerance_ppm
                ),
            )
        )

        survived = 0

        candidate_counts = []

        zero_candidate_rows = 0

        for row in eligible.itertuples(
            index=False
        ):

            result = generator.generate(
                precursor_mz=(
                    float(
                        row.precursor_mz
                    )
                ),
                adduct=(
                    str(
                        row.adduct
                    )
                ),
                require_trusted=True,
            )

            candidate_keys = {
                hit.inchikey14
                for hit in result.candidates
            }

            count = len(
                candidate_keys
            )

            candidate_counts.append(
                count
            )

            if count == 0:
                zero_candidate_rows += 1

            if (
                str(
                    row.inchikey14
                )
                in candidate_keys
            ):
                survived += 1

        candidate_counts = np.asarray(
            candidate_counts,
            dtype=np.int64,
        )

        survival_rate = (
            survived
            / len(
                eligible
            )
        )

        zero_rate = (
            zero_candidate_rows
            / len(
                eligible
            )
        )

        mean_candidates = float(
            candidate_counts.mean()
        )

        median_candidates = float(
            np.median(
                candidate_counts
            )
        )

        p90_candidates = float(
            np.percentile(
                candidate_counts,
                90,
            )
        )

        p99_candidates = float(
            np.percentile(
                candidate_counts,
                99,
            )
        )

        max_candidates = int(
            candidate_counts.max()
        )

        print(
            f"True candidate survived: "
            f"{survived:,}/"
            f"{len(eligible):,}"
        )

        print(
            f"Survival rate: "
            f"{survival_rate:.6%}"
        )

        print(
            f"Zero-candidate rows: "
            f"{zero_candidate_rows:,}"
        )

        print(
            f"Zero-candidate rate: "
            f"{zero_rate:.6%}"
        )

        print(
            f"Candidate count mean: "
            f"{mean_candidates:.2f}"
        )

        print(
            f"Candidate count median: "
            f"{median_candidates:.1f}"
        )

        print(
            f"Candidate count p90: "
            f"{p90_candidates:.1f}"
        )

        print(
            f"Candidate count p99: "
            f"{p99_candidates:.1f}"
        )

        print(
            f"Candidate count max: "
            f"{max_candidates:,}"
        )

        summary_lines.extend(
            [
                (
                    f"Tolerance: "
                    f"{tolerance_ppm:.1f} ppm"
                ),
                (
                    f"Survived: "
                    f"{survived:,}/"
                    f"{len(eligible):,}"
                ),
                (
                    f"Survival rate: "
                    f"{survival_rate:.6%}"
                ),
                (
                    f"Zero-candidate rate: "
                    f"{zero_rate:.6%}"
                ),
                (
                    f"Candidate mean: "
                    f"{mean_candidates:.2f}"
                ),
                (
                    f"Candidate median: "
                    f"{median_candidates:.1f}"
                ),
                (
                    f"Candidate p90: "
                    f"{p90_candidates:.1f}"
                ),
                (
                    f"Candidate p99: "
                    f"{p99_candidates:.1f}"
                ),
                (
                    f"Candidate max: "
                    f"{max_candidates:,}"
                ),
                "",
            ]
        )

    # =====================================================
    # Save summary
    # =====================================================

    elapsed = (
        time.time()
        - start_time
    )

    summary_lines.append(
        f"Elapsed seconds: "
        f"{elapsed:.2f}"
    )

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_PATH
    )

    print(
        f"\nElapsed: "
        f"{elapsed:.2f} s"
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "STAGE 7 STEP 4 COMPLETE"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()