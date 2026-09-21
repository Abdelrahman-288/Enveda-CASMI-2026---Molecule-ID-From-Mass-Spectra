from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from casmi.retrieval.candidate_mass_index import (
    CandidateMassIndex,
)


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

ENVEDA_VARIANTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "stage7_candidate_mass_variants.csv"
)

COCONUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage7"
    / "coconut_candidates.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "indexes"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COMBINED_PARQUET_PATH = (
    OUTPUT_DIR
    / "stage7_combined_candidate_mass_variants.parquet"
)

COMBINED_CSV_PATH = (
    OUTPUT_DIR
    / "stage7_combined_candidate_mass_variants.csv"
)

COMBINED_NPZ_PATH = (
    OUTPUT_DIR
    / "stage7_combined_candidate_mass_index.npz"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage7_combined_candidate_index_summary.txt"
)


# =========================================================
# Configuration
# =========================================================

MASS_ROUND_DECIMALS = 8

SELF_RETRIEVAL_SAMPLE_SIZE = 10_000

SEED = 42


# =========================================================
# Helpers
# =========================================================

def normalize_enveda(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "inchikey14",
        "normalized_smiles",
        "molecular_formula",
        "exact_mass",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Enveda variants missing columns: "
            f"{sorted(missing)}"
        )

    result = dataframe[
        [
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
            "exact_mass",
        ]
    ].copy()

    result[
        "source"
    ] = "enveda"

    result[
        "source_id"
    ] = ""

    result[
        "formal_charge"
    ] = 0

    result[
        "fragment_count"
    ] = 1

    return result


def normalize_coconut(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "inchikey14",
        "canonical_smiles",
        "molecular_formula",
        "exact_mass",
        "source_id",
        "formal_charge",
        "fragment_count",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            "COCONUT candidates missing columns: "
            f"{sorted(missing)}"
        )

    result = dataframe[
        [
            "inchikey14",
            "canonical_smiles",
            "molecular_formula",
            "exact_mass",
            "source_id",
            "formal_charge",
            "fragment_count",
        ]
    ].copy()

    result = result.rename(
        columns={
            "canonical_smiles":
            "normalized_smiles",
        }
    )

    result[
        "source"
    ] = "coconut"

    return result


def source_priority(
    source: str,
) -> int:
    """
    Prefer Enveda representation when an exact candidate
    variant exists in both databases.
    """

    if source == "enveda":
        return 0

    return 1


def main():

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 7 STEP 9 "
        "COMBINED ENVEDA + COCONUT INDEX"
    )

    start_time = time.time()

    # =====================================================
    # Validate inputs
    # =====================================================

    if not ENVEDA_VARIANTS_PATH.exists():

        raise FileNotFoundError(
            f"Missing Enveda variants:\n"
            f"{ENVEDA_VARIANTS_PATH}"
        )

    if not COCONUT_PATH.exists():

        raise FileNotFoundError(
            f"Missing COCONUT catalog:\n"
            f"{COCONUT_PATH}"
        )

    # =====================================================
    # Load inputs
    # =====================================================

    print(
        "\nLoading Enveda variants..."
    )

    enveda_raw = pd.read_csv(
        ENVEDA_VARIANTS_PATH
    )

    enveda = normalize_enveda(
        enveda_raw
    )

    print(
        f"Enveda mass variants: "
        f"{len(enveda):,}"
    )

    print(
        f"Enveda unique structures: "
        f"{enveda['inchikey14'].nunique():,}"
    )

    print(
        "\nLoading COCONUT candidates..."
    )

    coconut_raw = pd.read_parquet(
        COCONUT_PATH
    )

    coconut = normalize_coconut(
        coconut_raw
    )

    print(
        f"COCONUT mass variants: "
        f"{len(coconut):,}"
    )

    print(
        f"COCONUT unique structures: "
        f"{coconut['inchikey14'].nunique():,}"
    )

    # =====================================================
    # Structure overlap
    # =====================================================

    enveda_keys = set(
        enveda[
            "inchikey14"
        ].astype(str)
    )

    coconut_keys = set(
        coconut[
            "inchikey14"
        ].astype(str)
    )

    structure_overlap = (
        enveda_keys
        & coconut_keys
    )

    enveda_only_keys = (
        enveda_keys
        - coconut_keys
    )

    coconut_only_keys = (
        coconut_keys
        - enveda_keys
    )

    combined_structure_keys = (
        enveda_keys
        | coconut_keys
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STRUCTURE OVERLAP"
    )

    print(
        "=" * 100
    )

    print(
        f"Enveda structures: "
        f"{len(enveda_keys):,}"
    )

    print(
        f"COCONUT structures: "
        f"{len(coconut_keys):,}"
    )

    print(
        f"Shared structures: "
        f"{len(structure_overlap):,}"
    )

    print(
        f"Enveda-only structures: "
        f"{len(enveda_only_keys):,}"
    )

    print(
        f"COCONUT-only structures: "
        f"{len(coconut_only_keys):,}"
    )

    print(
        f"Combined unique structures: "
        f"{len(combined_structure_keys):,}"
    )

    enveda_overlap_pct = (
        len(
            structure_overlap
        )
        / len(
            enveda_keys
        )
        * 100.0
    )

    coconut_overlap_pct = (
        len(
            structure_overlap
        )
        / len(
            coconut_keys
        )
        * 100.0
    )

    print(
        f"Enveda covered by COCONUT: "
        f"{enveda_overlap_pct:.2f}%"
    )

    print(
        f"COCONUT overlapping Enveda: "
        f"{coconut_overlap_pct:.2f}%"
    )

    # =====================================================
    # Build common candidate table
    # =====================================================

    columns = [
        "inchikey14",
        "normalized_smiles",
        "molecular_formula",
        "exact_mass",
        "source",
        "source_id",
        "formal_charge",
        "fragment_count",
    ]

    combined = pd.concat(
        [
            enveda[
                columns
            ],
            coconut[
                columns
            ],
        ],
        ignore_index=True,
    )

    combined[
        "exact_mass"
    ] = pd.to_numeric(
        combined[
            "exact_mass"
        ],
        errors="coerce",
    )

    combined = combined.dropna(
        subset=[
            "inchikey14",
            "normalized_smiles",
            "exact_mass",
        ]
    )

    combined = combined[
        np.isfinite(
            combined[
                "exact_mass"
            ]
        )
    ].copy()

    combined = combined[
        combined[
            "exact_mass"
        ]
        > 0
    ].copy()

    # =====================================================
    # Variant identity
    # =====================================================

    combined[
        "mass_key"
    ] = (
        combined[
            "exact_mass"
        ]
        .round(
            MASS_ROUND_DECIMALS
        )
    )

    combined[
        "source_priority"
    ] = (
        combined[
            "source"
        ]
        .map(
            source_priority
        )
    )

    # =====================================================
    # Determine source provenance for each mass variant
    # =====================================================

    provenance = (
        combined
        .groupby(
            [
                "inchikey14",
                "mass_key",
            ],
            sort=False,
            observed=True,
        )[
            "source"
        ]
        .agg(
            lambda values:
            "+".join(
                sorted(
                    set(
                        values
                    )
                )
            )
        )
        .rename(
            "sources"
        )
        .reset_index()
    )

    # =====================================================
    # Deduplicate exact connectivity + mass variants
    #
    # Prefer Enveda representation where available.
    # =====================================================

    before_variant_dedup = len(
        combined
    )

    combined = (
        combined
        .sort_values(
            [
                "inchikey14",
                "mass_key",
                "source_priority",
                "normalized_smiles",
            ]
        )
        .drop_duplicates(
            subset=[
                "inchikey14",
                "mass_key",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    combined = combined.merge(
        provenance,
        on=[
            "inchikey14",
            "mass_key",
        ],
        how="left",
        validate="one_to_one",
    )

    duplicates_removed = (
        before_variant_dedup
        - len(
            combined
        )
    )

    # =====================================================
    # Structure-level source information
    # =====================================================

    structure_provenance = (
        pd.concat(
            [
                enveda[
                    [
                        "inchikey14",
                        "source",
                    ]
                ],
                coconut[
                    [
                        "inchikey14",
                        "source",
                    ]
                ],
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .groupby(
            "inchikey14",
            sort=False,
            observed=True,
        )[
            "source"
        ]
        .agg(
            lambda values:
            "+".join(
                sorted(
                    set(
                        values
                    )
                )
            )
        )
        .rename(
            "structure_sources"
        )
    )

    combined[
        "structure_sources"
    ] = (
        combined[
            "inchikey14"
        ]
        .map(
            structure_provenance
        )
    )

    # =====================================================
    # Final ordering
    # =====================================================

    combined = (
        combined
        .sort_values(
            [
                "exact_mass",
                "inchikey14",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    combined_unique_structures = int(
        combined[
            "inchikey14"
        ].nunique()
    )

    combined_variants = len(
        combined
    )

    structure_variant_counts = (
        combined[
            "inchikey14"
        ]
        .value_counts()
    )

    multi_variant_structures = int(
        (
            structure_variant_counts
            > 1
        ).sum()
    )

    max_variants_per_structure = int(
        structure_variant_counts.max()
    )

    # =====================================================
    # Provenance counts
    # =====================================================

    structure_source_counts = (
        structure_provenance
        .value_counts()
    )

    both_structure_count = int(
        structure_source_counts.get(
            "coconut+enveda",
            0,
        )
    )

    enveda_only_structure_count = int(
        structure_source_counts.get(
            "enveda",
            0,
        )
    )

    coconut_only_structure_count = int(
        structure_source_counts.get(
            "coconut",
            0,
        )
    )

    # =====================================================
    # CandidateMassIndex validation
    # =====================================================

    index_input = combined[
        [
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
            "exact_mass",
        ]
    ].copy()

    mass_index = CandidateMassIndex(
        index_input
    )

    if (
        mass_index.unique_structure_count
        != combined_unique_structures
    ):

        raise RuntimeError(
            "Combined CandidateMassIndex structure count "
            "does not match combined catalog."
        )

    # =====================================================
    # Self retrieval
    # =====================================================

    print(
        "\nChecking combined-index self retrieval..."
    )

    rng = np.random.default_rng(
        SEED
    )

    sample_size = min(
        SELF_RETRIEVAL_SAMPLE_SIZE,
        len(
            combined
        ),
    )

    sampled_indices = rng.choice(
        len(
            combined
        ),
        size=sample_size,
        replace=False,
    )

    self_retrieved = 0

    for row_index in sampled_indices:

        row = combined.iloc[
            int(
                row_index
            )
        ]

        hits = mass_index.query(
            neutral_mass=float(
                row[
                    "exact_mass"
                ]
            ),
            tolerance_ppm=10.0,
        )

        returned_keys = {
            hit.inchikey14
            for hit in hits
        }

        if (
            str(
                row[
                    "inchikey14"
                ]
            )
            in returned_keys
        ):
            self_retrieved += 1

    self_retrieval_rate = (
        self_retrieved
        / sample_size
    )

    print(
        f"Self retrieval: "
        f"{self_retrieved:,}/"
        f"{sample_size:,}"
    )

    print(
        f"Self retrieval rate: "
        f"{self_retrieval_rate:.6%}"
    )

    if self_retrieval_rate < 1.0:

        raise RuntimeError(
            "Combined index failed exact self retrieval."
        )

    # =====================================================
    # Save combined Parquet
    # =====================================================

    output_columns = [
        "inchikey14",
        "normalized_smiles",
        "molecular_formula",
        "exact_mass",
        "sources",
        "structure_sources",
        "source_id",
        "formal_charge",
        "fragment_count",
    ]

    output = combined[
        output_columns
    ].copy()

    print(
        "\nSaving combined Parquet..."
    )

    output.to_parquet(
        COMBINED_PARQUET_PATH,
        index=False,
        compression="zstd",
    )

    # =====================================================
    # Save CSV
    # =====================================================

    print(
        "Saving combined CSV..."
    )

    output.to_csv(
        COMBINED_CSV_PATH,
        index=False,
    )

    # =====================================================
    # Save compact NumPy index
    # =====================================================

    print(
        "Saving combined NumPy index..."
    )

    np.savez_compressed(
        COMBINED_NPZ_PATH,
        exact_mass=(
            output[
                "exact_mass"
            ]
            .to_numpy(
                dtype=np.float64,
            )
        ),
        inchikey14=(
            output[
                "inchikey14"
            ]
            .astype(str)
            .to_numpy()
        ),
        normalized_smiles=(
            output[
                "normalized_smiles"
            ]
            .astype(str)
            .to_numpy()
        ),
        molecular_formula=(
            output[
                "molecular_formula"
            ]
            .fillna("")
            .astype(str)
            .to_numpy()
        ),
    )

    # =====================================================
    # File sizes
    # =====================================================

    parquet_mb = (
        COMBINED_PARQUET_PATH.stat().st_size
        / 1024**2
    )

    csv_mb = (
        COMBINED_CSV_PATH.stat().st_size
        / 1024**2
    )

    npz_mb = (
        COMBINED_NPZ_PATH.stat().st_size
        / 1024**2
    )

    elapsed = (
        time.time()
        - start_time
    )

    # =====================================================
    # Summary
    # =====================================================

    summary_lines = [
        (
            "ENVEDA CASMI 2026 — "
            "STAGE 7 COMBINED CANDIDATE INDEX"
        ),
        "",
        "STRUCTURE COUNTS",
        (
            f"Enveda unique structures: "
            f"{len(enveda_keys):,}"
        ),
        (
            f"COCONUT unique structures: "
            f"{len(coconut_keys):,}"
        ),
        (
            f"Shared structures: "
            f"{len(structure_overlap):,}"
        ),
        (
            f"Enveda-only structures: "
            f"{len(enveda_only_keys):,}"
        ),
        (
            f"COCONUT-only structures: "
            f"{len(coconut_only_keys):,}"
        ),
        (
            f"Combined unique structures: "
            f"{combined_unique_structures:,}"
        ),
        "",
        "OVERLAP",
        (
            f"Enveda covered by COCONUT: "
            f"{enveda_overlap_pct:.4f}%"
        ),
        (
            f"COCONUT overlapping Enveda: "
            f"{coconut_overlap_pct:.4f}%"
        ),
        "",
        "VARIANTS",
        (
            f"Combined mass variants: "
            f"{combined_variants:,}"
        ),
        (
            f"Duplicate source variants removed: "
            f"{duplicates_removed:,}"
        ),
        (
            f"Structures with >1 mass variant: "
            f"{multi_variant_structures:,}"
        ),
        (
            f"Maximum variants per structure: "
            f"{max_variants_per_structure}"
        ),
        "",
        "STRUCTURE PROVENANCE",
        (
            f"Enveda only: "
            f"{enveda_only_structure_count:,}"
        ),
        (
            f"COCONUT only: "
            f"{coconut_only_structure_count:,}"
        ),
        (
            f"Both sources: "
            f"{both_structure_count:,}"
        ),
        "",
        "VALIDATION",
        (
            f"Self retrieval: "
            f"{self_retrieved:,}/"
            f"{sample_size:,}"
        ),
        (
            f"Self retrieval rate: "
            f"{self_retrieval_rate:.6%}"
        ),
        "",
        "OUTPUT FILES",
        (
            f"Parquet MB: "
            f"{parquet_mb:.2f}"
        ),
        (
            f"CSV MB: "
            f"{csv_mb:.2f}"
        ),
        (
            f"NPZ MB: "
            f"{npz_mb:.2f}"
        ),
        "",
        (
            f"Elapsed seconds: "
            f"{elapsed:.2f}"
        ),
    ]

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    # =====================================================
    # Console summary
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 7 COMBINED CANDIDATE INDEX SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Enveda structures: "
        f"{len(enveda_keys):,}"
    )

    print(
        f"COCONUT structures: "
        f"{len(coconut_keys):,}"
    )

    print(
        f"Shared structures: "
        f"{len(structure_overlap):,}"
    )

    print(
        f"COCONUT-only structures: "
        f"{len(coconut_only_keys):,}"
    )

    print(
        f"Combined unique structures: "
        f"{combined_unique_structures:,}"
    )

    print(
        f"Combined mass variants: "
        f"{combined_variants:,}"
    )

    print(
        f"Structures with >1 variant: "
        f"{multi_variant_structures:,}"
    )

    print(
        f"Maximum variants/structure: "
        f"{max_variants_per_structure}"
    )

    print(
        f"Self retrieval: "
        f"{self_retrieval_rate:.6%}"
    )

    print(
        f"\nParquet size: "
        f"{parquet_mb:.2f} MB"
    )

    print(
        f"CSV size: "
        f"{csv_mb:.2f} MB"
    )

    print(
        f"NPZ size: "
        f"{npz_mb:.2f} MB"
    )

    print(
        f"Elapsed: "
        f"{elapsed:.2f} s"
    )

    print(
        "\nCombined Parquet:"
    )

    print(
        COMBINED_PARQUET_PATH
    )

    print(
        "\nCombined NumPy index:"
    )

    print(
        COMBINED_NPZ_PATH
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 7 STEP 9 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()