from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

SAMPLE_SUBMISSION_PATH = (
    DATASET_DIR
    / "sample_submission.csv"
)

RANKED_CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage11"
    / "test_ranked_candidates.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "stage11"
)

SUBMISSION_PATH = (
    OUTPUT_DIR
    / "submission.csv"
)

TOP25_DEBUG_PATH = (
    OUTPUT_DIR
    / "submission_top25_debug.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage11_submission_summary.txt"
)


# ============================================================
# Configuration
# ============================================================

TOP_K = 25


# ============================================================
# RDKit validation
# ============================================================

tautomer_enumerator = (
    rdMolStandardize
    .TautomerEnumerator()
)


def evaluation_inchikey14(
    smiles: str,
) -> str | None:
    """
    Approximate the competition identity procedure:

    SMILES
        ->
    RDKit molecule
        ->
    canonical tautomer
        ->
    InChIKey
        ->
    first 14 characters

    Used only as a final consistency check.
    """

    molecule = Chem.MolFromSmiles(
        str(smiles)
    )

    if molecule is None:
        return None

    try:
        molecule = (
            tautomer_enumerator
            .Canonicalize(
                molecule
            )
        )
    except Exception:
        return None

    try:
        inchikey = (
            Chem.MolToInchiKey(
                molecule
            )
        )
    except Exception:
        return None

    if not inchikey:
        return None

    return (
        inchikey
        .split("-")[0]
        [:14]
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 100)
    print(
        "ENVEDA CASMI 2026 - "
        "STAGE 11.5 FINAL SUBMISSION"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    required = [
        SAMPLE_SUBMISSION_PATH,
        RANKED_CANDIDATES_PATH,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    print(
        "\nLoading sample submission..."
    )

    sample = pd.read_csv(
        SAMPLE_SUBMISSION_PATH
    )

    print(
        "Sample submission shape: "
        f"{sample.shape}"
    )

    print(
        "Sample submission columns: "
        f"{sample.columns.tolist()}"
    )

    required_sample_columns = {
        "molecule_id",
        "smiles",
    }

    missing_sample_columns = (
        required_sample_columns
        - set(
            sample.columns
        )
    )

    if missing_sample_columns:
        raise RuntimeError(
            "Unexpected sample_submission.csv schema. "
            f"Missing: "
            f"{sorted(missing_sample_columns)}"
        )

    ranked = pd.read_parquet(
        RANKED_CANDIDATES_PATH
    )

    print(
        "\nRanked candidate rows: "
        f"{len(ranked):,}"
    )

    print(
        "Ranked molecules: "
        f"{ranked['molecule_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # Basic ranked-data validation
    # --------------------------------------------------------

    required_ranked_columns = [
        "molecule_id",
        "candidate_inchikey14",
        "candidate_smiles",
        "stage10_score",
        "final_rank",
    ]

    missing_ranked_columns = [
        column
        for column
        in required_ranked_columns
        if column not in ranked.columns
    ]

    if missing_ranked_columns:
        raise RuntimeError(
            "Ranked candidate file is missing columns: "
            f"{missing_ranked_columns}"
        )

    if ranked[
        "stage10_score"
    ].isna().any():
        raise RuntimeError(
            "Stage 10 scores contain NaN."
        )

    if not np.isfinite(
        ranked[
            "stage10_score"
        ].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "Stage 10 scores contain "
            "non-finite values."
        )

    duplicate_candidates = (
        ranked.duplicated(
            [
                "molecule_id",
                "candidate_inchikey14",
            ]
        )
    )

    if duplicate_candidates.any():
        raise RuntimeError(
            "Duplicate candidate identities exist "
            "inside ranked queries."
        )

    # --------------------------------------------------------
    # Select top 25
    # --------------------------------------------------------

    top25 = (
        ranked[
            ranked[
                "final_rank"
            ]
            <= TOP_K
        ]
        .copy()
    )

    top25 = (
        top25.sort_values(
            [
                "molecule_id",
                "final_rank",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nTop-25 candidate rows: "
        f"{len(top25):,}"
    )

    # --------------------------------------------------------
    # RDKit validation
    # --------------------------------------------------------

    print(
        "\nValidating top-25 structures with RDKit..."
    )

    parsed_ok = []

    computed_inchikey14 = []

    identity_match = []

    for index, row in enumerate(
        top25.itertuples(
            index=False
        )
    ):

        smiles = str(
            row.candidate_smiles
        )

        candidate_id = str(
            row.candidate_inchikey14
        )

        molecule = Chem.MolFromSmiles(
            smiles
        )

        valid = (
            molecule is not None
        )

        parsed_ok.append(
            valid
        )

        eval_id = (
            evaluation_inchikey14(
                smiles
            )
            if valid
            else None
        )

        computed_inchikey14.append(
            eval_id
        )

        identity_match.append(
            eval_id
            == candidate_id
        )

        if (
            (index + 1) % 2000 == 0
            or index + 1 == len(top25)
        ):
            print(
                "  Validated "
                f"{index + 1:,}"
                "/"
                f"{len(top25):,}"
            )

    top25[
        "rdkit_valid"
    ] = parsed_ok

    top25[
        "evaluation_inchikey14"
    ] = computed_inchikey14

    top25[
        "identity_match"
    ] = identity_match

    invalid_count = int(
        (~top25["rdkit_valid"])
        .sum()
    )

    if invalid_count > 0:

        invalid_preview = (
            top25[
                ~top25[
                    "rdkit_valid"
                ]
            ]
            .head(20)
        )

        print(
            invalid_preview[
                [
                    "molecule_id",
                    "candidate_smiles",
                ]
            ]
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"{invalid_count:,} top-25 candidate SMILES "
            "failed RDKit parsing."
        )

    # --------------------------------------------------------
    # Evaluation-identity deduplication
    # --------------------------------------------------------

    print(
        "\nChecking evaluation-level identities..."
    )

    evaluation_duplicates = (
        top25.duplicated(
            [
                "molecule_id",
                "evaluation_inchikey14",
            ]
        )
    )

    duplicate_evaluation_count = int(
        evaluation_duplicates.sum()
    )

    if duplicate_evaluation_count > 0:

        print(
            "WARNING: "
            f"{duplicate_evaluation_count:,} top-25 rows "
            "collapse to duplicate evaluation identities."
        )

        # Keep the highest-ranked representation.
        top25 = (
            top25[
                ~evaluation_duplicates
            ]
            .copy()
        )

        # Reassign ranks after deduplication.
        top25[
            "submission_rank"
        ] = (
            top25.groupby(
                "molecule_id",
                sort=False,
            )
            .cumcount()
            + 1
        )

    else:

        top25[
            "submission_rank"
        ] = (
            top25[
                "final_rank"
            ]
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # After evaluation-level dedupe, some queries could have
    # fewer than 25 predictions.
    #
    # The competition allows up to 25 predictions, so this is
    # valid. We do not insert duplicates just to reach 25.
    # --------------------------------------------------------

    final_counts = (
        top25.groupby(
            "molecule_id"
        )
        .size()
    )

    # --------------------------------------------------------
    # Build submission strings
    # --------------------------------------------------------

    prediction_lookup = {}

    for molecule_id, group in (
        top25.groupby(
            "molecule_id",
            sort=False,
        )
    ):

        group = (
            group.sort_values(
                "submission_rank",
                kind="stable",
            )
        )

        smiles_values = (
            group[
                "candidate_smiles"
            ]
            .astype(str)
            .tolist()
        )

        if len(
            smiles_values
        ) > TOP_K:
            raise RuntimeError(
                f"{molecule_id} has more than "
                f"{TOP_K} predictions."
            )

        prediction_lookup[
            str(molecule_id)
        ] = ";".join(
            smiles_values
        )

    # --------------------------------------------------------
    # Preserve sample submission row order
    # --------------------------------------------------------

    submission = (
        sample.copy()
    )

    submission[
        "molecule_id"
    ] = (
        submission[
            "molecule_id"
        ]
        .astype(str)
    )

    submission[
        "smiles"
    ] = (
        submission[
            "molecule_id"
        ]
        .map(
            prediction_lookup
        )
    )

    # --------------------------------------------------------
    # Strong final validation
    # --------------------------------------------------------

    missing_predictions = (
        submission[
            "smiles"
        ]
        .isna()
    )

    if missing_predictions.any():

        missing_ids = (
            submission.loc[
                missing_predictions,
                "molecule_id",
            ]
            .tolist()
        )

        raise RuntimeError(
            "Missing predictions for molecule IDs: "
            f"{missing_ids[:20]}"
        )

    if len(
        submission
    ) != len(
        sample
    ):
        raise RuntimeError(
            "Submission row count differs from "
            "sample submission."
        )

    if (
        submission[
            "molecule_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate molecule_id rows in submission."
        )

    if (
        submission[
            "smiles"
        ]
        .astype(str)
        .str.len()
        .eq(0)
        .any()
    ):
        raise RuntimeError(
            "Empty prediction string found."
        )

    # --------------------------------------------------------
    # Verify semicolon counts
    # --------------------------------------------------------

    submission_candidate_counts = (
        submission[
            "smiles"
        ]
        .astype(str)
        .map(
            lambda value:
            len(
                value.split(";")
            )
        )
    )

    if (
        submission_candidate_counts
        > TOP_K
    ).any():
        raise RuntimeError(
            "Submission contains more than "
            f"{TOP_K} predictions for a molecule."
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    submission.to_csv(
        SUBMISSION_PATH,
        index=False,
    )

    top25.to_parquet(
        TOP25_DEBUG_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    identity_match_count = int(
        top25[
            "identity_match"
        ].sum()
    )

    identity_mismatch_count = int(
        (
            ~top25[
                "identity_match"
            ]
        ).sum()
    )

    summary_lines = [
        (
            "ENVEDA CASMI 2026 - "
            "STAGE 11.5 FINAL SUBMISSION"
        ),
        "",
        (
            "Submission rows: "
            f"{len(submission):,}"
        ),
        (
            "Unique molecule IDs: "
            f"{submission['molecule_id'].nunique():,}"
        ),
        (
            "Top-K limit: "
            f"{TOP_K}"
        ),
        "",
        "PREDICTIONS PER MOLECULE",
        (
            "Minimum: "
            f"{submission_candidate_counts.min()}"
        ),
        (
            "Median: "
            f"{np.median(submission_candidate_counts):.2f}"
        ),
        (
            "Mean: "
            f"{submission_candidate_counts.mean():.2f}"
        ),
        (
            "Maximum: "
            f"{submission_candidate_counts.max()}"
        ),
        "",
        (
            "Top candidate rows validated by RDKit: "
            f"{len(top25):,}"
        ),
        (
            "Invalid RDKit SMILES: "
            f"{invalid_count:,}"
        ),
        (
            "Duplicate evaluation identities removed: "
            f"{duplicate_evaluation_count:,}"
        ),
        (
            "Stored identity matches after tautomer "
            "canonicalization: "
            f"{identity_match_count:,}"
        ),
        (
            "Stored identity mismatches after tautomer "
            "canonicalization: "
            f"{identity_mismatch_count:,}"
        ),
        "",
        (
            "Submission path: "
            f"{SUBMISSION_PATH}"
        ),
        (
            "Debug top-25 path: "
            f"{TOP25_DEBUG_PATH}"
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
    print(
        "FINAL SUBMISSION PREVIEW"
    )
    print("=" * 100)

    print(
        submission
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print(
        "STAGE 11.5 SUMMARY"
    )
    print("=" * 100)

    for line in summary_lines:
        print(
            line
        )

    print()
    print("=" * 100)
    print(
        "STAGE 11.5 COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()