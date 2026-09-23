from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.chemistry.adducts import (
    is_supported_adduct,
    is_trusted_for_mass_filter,
)
from casmi.chemistry.mass_filter_policy import (
    get_mass_filter_decision,
)
from casmi.retrieval.candidate_generation import (
    AdductAwareCandidateGenerator,
)
from casmi.retrieval.candidate_mass_index import (
    CandidateMassIndex,
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

COMBINED_VARIANTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "stage7_combined_candidate_mass_variants.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage11"
)

SPECTRUM_CANDIDATES_PATH = (
    OUTPUT_DIR
    / "test_spectrum_candidates.parquet"
)

MOLECULE_CANDIDATES_PATH = (
    OUTPUT_DIR
    / "test_molecule_candidates.parquet"
)

CANDIDATE_STRUCTURES_PATH = (
    OUTPUT_DIR
    / "test_candidate_structures.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage11_test_candidates_summary.txt"
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_TOLERANCE_PPM = 10.0


# ============================================================
# Helpers
# ============================================================

def validate_required_files() -> None:
    required = [
        TEST_PATH,
        COMBINED_VARIANTS_PATH,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required Stage 11 files:\n"
            + "\n".join(str(path) for path in missing)
        )


def build_mass_index(
    variants: pd.DataFrame,
) -> CandidateMassIndex:

    required_columns = [
        "inchikey14",
        "normalized_smiles",
        "molecular_formula",
        "exact_mass",
    ]

    missing = [
        column
        for column in required_columns
        if column not in variants.columns
    ]

    if missing:
        raise ValueError(
            "Combined Stage 7 variants are missing columns: "
            f"{missing}"
        )

    return CandidateMassIndex(
        variants[
            required_columns
        ].copy()
    )


def get_row_value(
    row,
    name: str,
    default=None,
):
    if name not in row.index:
        return default

    value = row[name]

    if pd.isna(value):
        return default

    return value


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 100)
    print(
        "ENVEDA CASMI 2026 - STAGE 11.1 "
        "TEST CANDIDATE MANIFEST"
    )
    print("=" * 100)

    validate_required_files()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Test spectra
    # --------------------------------------------------------

    print("\nLoading test spectra...")

    test = pd.read_parquet(
        TEST_PATH
    ).reset_index(
        drop=True
    )

    required_test_columns = [
        "molecule_id",
        "precursor_mz",
        "adduct",
    ]

    missing_test_columns = [
        column
        for column in required_test_columns
        if column not in test.columns
    ]

    if missing_test_columns:
        raise ValueError(
            "Test dataframe is missing columns: "
            f"{missing_test_columns}"
        )

    test["spectrum_index"] = np.arange(
        len(test),
        dtype=np.int64,
    )

    print(
        f"Test spectra: {len(test):,}"
    )

    print(
        "Test molecules: "
        f"{test['molecule_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # Combined candidate index
    # --------------------------------------------------------

    print(
        "\nLoading combined "
        "Enveda + COCONUT candidate index..."
    )

    variants = pd.read_parquet(
        COMBINED_VARIANTS_PATH
    )

    mass_index = build_mass_index(
        variants
    )

    print(
        "Indexed structures: "
        f"{mass_index.unique_structure_count:,}"
    )

    print(
        "Mass variants: "
        f"{mass_index.variant_count:,}"
    )

    # --------------------------------------------------------
    # Per-spectrum candidates
    # --------------------------------------------------------

    print(
        "\nGenerating per-spectrum candidates..."
    )

    spectrum_candidate_rows = []

    candidate_ids_by_spectrum = {}

    generation_metadata = {}

    for row_number, row in test.iterrows():

        spectrum_index = int(
            row["spectrum_index"]
        )

        molecule_id = str(
            row["molecule_id"]
        )

        precursor_mz = float(
            row["precursor_mz"]
        )

        adduct = str(
            row["adduct"]
        )

        instrument_type = str(
            get_row_value(
                row,
                "instrument_type",
                "",
            )
        )

        ingest_lib = str(
            get_row_value(
                row,
                "ingest_lib",
                "",
            )
        )

        supported = is_supported_adduct(
            adduct
        )

        trusted = is_trusted_for_mass_filter(
            adduct
        )

        decision = get_mass_filter_decision(
            adduct_supported=supported,
            adduct_trusted=trusted,
            instrument_type=instrument_type,
            ingest_lib=ingest_lib,
            inference_mode=True,
        )

        if (
            not decision.hard_filter
            or decision.tolerance_ppm is None
        ):
            candidate_ids_by_spectrum[
                spectrum_index
            ] = set()

            generation_metadata[
                spectrum_index
            ] = {
                "hard_filter_applied": False,
                "candidate_count": 0,
                "message": decision.reason,
            }

            continue

        generator = (
            AdductAwareCandidateGenerator(
                mass_index=mass_index,
                tolerance_ppm=float(
                    decision.tolerance_ppm
                ),
            )
        )

        result = generator.generate(
            precursor_mz=precursor_mz,
            adduct=adduct,
            require_trusted=True,
        )

        candidate_ids = {
            hit.inchikey14
            for hit in result.candidates
        }

        candidate_ids_by_spectrum[
            spectrum_index
        ] = candidate_ids

        generation_metadata[
            spectrum_index
        ] = {
            "hard_filter_applied": (
                result.hard_filter_applied
            ),
            "candidate_count": len(
                candidate_ids
            ),
            "message": result.message,
        }

        for hit in result.candidates:

            spectrum_candidate_rows.append(
                {
                    "molecule_id": molecule_id,
                    "spectrum_index": (
                        spectrum_index
                    ),
                    "precursor_mz": (
                        precursor_mz
                    ),
                    "adduct": adduct,
                    "instrument_type": (
                        instrument_type
                    ),
                    "candidate_inchikey14": (
                        hit.inchikey14
                    ),
                    "candidate_smiles": (
                        hit.normalized_smiles
                    ),
                    "molecular_formula": (
                        hit.molecular_formula
                    ),
                    "candidate_exact_mass": (
                        float(hit.exact_mass)
                    ),
                    "mass_error_da": (
                        float(hit.mass_error_da)
                    ),
                    "mass_error_ppm": (
                        float(hit.mass_error_ppm)
                    ),
                    "variant_count": (
                        int(hit.variant_count)
                    ),
                    "tolerance_ppm": (
                        float(
                            decision.tolerance_ppm
                        )
                    ),
                }
            )

        if (
            (row_number + 1) % 100 == 0
            or row_number + 1 == len(test)
        ):
            print(
                f"  Processed "
                f"{row_number + 1:,}"
                f"/{len(test):,} spectra"
            )

    spectrum_candidates = pd.DataFrame(
        spectrum_candidate_rows
    )

    if spectrum_candidates.empty:
        raise RuntimeError(
            "Stage 11 produced no "
            "per-spectrum candidates."
        )

    # --------------------------------------------------------
    # Molecule-level intersection pools
    # --------------------------------------------------------

    print(
        "\nBuilding molecule-level "
        "intersection candidate pools..."
    )

    molecule_candidate_rows = []

    pool_sizes = []

    fallback_count = 0

    for molecule_id, group in test.groupby(
        "molecule_id",
        sort=True,
    ):

        spectrum_indices = (
            group["spectrum_index"]
            .astype(int)
            .tolist()
        )

        usable_sets = [
            candidate_ids_by_spectrum[
                spectrum_index
            ]
            for spectrum_index
            in spectrum_indices
            if len(
                candidate_ids_by_spectrum[
                    spectrum_index
                ]
            ) > 0
        ]

        if not usable_sets:
            raise RuntimeError(
                f"No usable candidate sets for "
                f"molecule_id={molecule_id}"
            )

        intersection = set.intersection(
            *usable_sets
        )

        if intersection:
            candidate_pool = intersection
            candidate_mode = "intersection"
        else:
            candidate_pool = set.union(
                *usable_sets
            )
            candidate_mode = "union_fallback"
            fallback_count += 1

        pool_sizes.append(
            len(candidate_pool)
        )

        molecule_spectrum_candidates = (
            spectrum_candidates[
                spectrum_candidates[
                    "molecule_id"
                ]
                == molecule_id
            ]
        )

        for candidate_id in sorted(
            candidate_pool
        ):

            candidate_rows = (
                molecule_spectrum_candidates[
                    molecule_spectrum_candidates[
                        "candidate_inchikey14"
                    ]
                    == candidate_id
                ]
            )

            if candidate_rows.empty:
                raise RuntimeError(
                    "Candidate pool contains a "
                    "candidate with no spectrum rows."
                )

            # CandidateMassIndex already chooses the
            # closest matching mass variant per spectrum.
            # Keep a deterministic representative SMILES.
            candidate_rows = (
                candidate_rows.sort_values(
                    [
                        "spectrum_index",
                        "candidate_smiles",
                    ],
                    kind="mergesort",
                )
            )

            first = candidate_rows.iloc[0]

            supporting_spectra = int(
                candidate_rows[
                    "spectrum_index"
                ].nunique()
            )

            molecule_candidate_rows.append(
                {
                    "molecule_id": (
                        str(molecule_id)
                    ),
                    "candidate_inchikey14": (
                        candidate_id
                    ),
                    "candidate_smiles": (
                        str(
                            first[
                                "candidate_smiles"
                            ]
                        )
                    ),
                    "molecular_formula": (
                        str(
                            first[
                                "molecular_formula"
                            ]
                        )
                    ),
                    "candidate_mode": (
                        candidate_mode
                    ),
                    "num_spectra": (
                        len(spectrum_indices)
                    ),
                    "num_filtered_spectra": (
                        len(usable_sets)
                    ),
                    "num_supporting_spectra": (
                        supporting_spectra
                    ),
                    "mean_mass_error_ppm": (
                        float(
                            candidate_rows[
                                "mass_error_ppm"
                            ].mean()
                        )
                    ),
                    "mean_abs_mass_error_ppm": (
                        float(
                            candidate_rows[
                                "mass_error_ppm"
                            ]
                            .abs()
                            .mean()
                        )
                    ),
                }
            )

    molecule_candidates = pd.DataFrame(
        molecule_candidate_rows
    )

    # --------------------------------------------------------
    # Candidate structure universe
    # --------------------------------------------------------

    print(
        "\nBuilding unique candidate "
        "structure universe..."
    )

    candidate_structures = (
        molecule_candidates[
            [
                "candidate_inchikey14",
                "candidate_smiles",
                "molecular_formula",
            ]
        ]
        .drop_duplicates(
            subset=[
                "candidate_inchikey14",
            ]
        )
        .sort_values(
            "candidate_inchikey14",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    candidate_structures.insert(
        0,
        "candidate_structure_index",
        np.arange(
            len(candidate_structures),
            dtype=np.int64,
        ),
    )

    structure_index_lookup = dict(
        zip(
            candidate_structures[
                "candidate_inchikey14"
            ],
            candidate_structures[
                "candidate_structure_index"
            ],
        )
    )

    molecule_candidates[
        "candidate_structure_index"
    ] = (
        molecule_candidates[
            "candidate_inchikey14"
        ]
        .map(
            structure_index_lookup
        )
        .astype(
            np.int64
        )
    )

    spectrum_candidates[
        "candidate_structure_index"
    ] = (
        spectrum_candidates[
            "candidate_inchikey14"
        ]
        .map(
            structure_index_lookup
        )
    )

    # Spectrum table currently includes every raw
    # per-spectrum hit. Keep only structures that survived
    # molecule-level candidate-pool selection.
    spectrum_candidates = (
        spectrum_candidates[
            spectrum_candidates[
                "candidate_structure_index"
            ].notna()
        ]
        .copy()
    )

    spectrum_candidates[
        "candidate_structure_index"
    ] = (
        spectrum_candidates[
            "candidate_structure_index"
        ]
        .astype(
            np.int64
        )
    )

    # --------------------------------------------------------
    # Deterministic output ordering
    # --------------------------------------------------------

    spectrum_candidates = (
        spectrum_candidates.sort_values(
            [
                "molecule_id",
                "spectrum_index",
                "candidate_structure_index",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    molecule_candidates = (
        molecule_candidates.sort_values(
            [
                "molecule_id",
                "candidate_structure_index",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    molecule_count = int(
        molecule_candidates[
            "molecule_id"
        ].nunique()
    )

    if molecule_count != int(
        test["molecule_id"].nunique()
    ):
        raise RuntimeError(
            "Not all test molecule IDs received "
            "candidate pools."
        )

    duplicate_pairs = (
        molecule_candidates.duplicated(
            subset=[
                "molecule_id",
                "candidate_inchikey14",
            ]
        )
    )

    if duplicate_pairs.any():
        raise RuntimeError(
            "Duplicate molecule/candidate pairs "
            "found."
        )

    if (
        molecule_candidates[
            "candidate_structure_index"
        ].isna().any()
    ):
        raise RuntimeError(
            "Missing candidate structure indices."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    spectrum_candidates.to_parquet(
        SPECTRUM_CANDIDATES_PATH,
        index=False,
    )

    molecule_candidates.to_parquet(
        MOLECULE_CANDIDATES_PATH,
        index=False,
    )

    candidate_structures.to_parquet(
        CANDIDATE_STRUCTURES_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    pool_sizes_array = np.asarray(
        pool_sizes,
        dtype=np.float64,
    )

    candidate_modes = (
        molecule_candidates[
            [
                "molecule_id",
                "candidate_mode",
            ]
        ]
        .drop_duplicates()
        ["candidate_mode"]
        .value_counts()
        .to_dict()
    )

    summary_lines = [
        (
            "ENVEDA CASMI 2026 - "
            "STAGE 11.1 TEST CANDIDATE MANIFEST"
        ),
        "",
        f"Test spectra: {len(test):,}",
        (
            "Test molecules: "
            f"{test['molecule_id'].nunique():,}"
        ),
        (
            "Combined indexed structures: "
            f"{mass_index.unique_structure_count:,}"
        ),
        (
            "Combined mass variants: "
            f"{mass_index.variant_count:,}"
        ),
        "",
        "MOLECULE CANDIDATE POOLS",
        (
            f"Minimum: "
            f"{pool_sizes_array.min():.2f}"
        ),
        (
            f"Median: "
            f"{np.median(pool_sizes_array):.2f}"
        ),
        (
            f"Mean: "
            f"{pool_sizes_array.mean():.2f}"
        ),
        (
            f"P90: "
            f"{np.percentile(pool_sizes_array, 90):.2f}"
        ),
        (
            f"P95: "
            f"{np.percentile(pool_sizes_array, 95):.2f}"
        ),
        (
            f"P99: "
            f"{np.percentile(pool_sizes_array, 99):.2f}"
        ),
        (
            f"Maximum: "
            f"{pool_sizes_array.max():.2f}"
        ),
        "",
        (
            "Intersection pools: "
            f"{candidate_modes.get('intersection', 0):,}"
        ),
        (
            "Union fallbacks: "
            f"{candidate_modes.get('union_fallback', 0):,}"
        ),
        "",
        (
            "Molecule-candidate rows: "
            f"{len(molecule_candidates):,}"
        ),
        (
            "Spectrum-candidate rows: "
            f"{len(spectrum_candidates):,}"
        ),
        (
            "Unique candidate structures: "
            f"{len(candidate_structures):,}"
        ),
        "",
        f"Spectrum manifest: {SPECTRUM_CANDIDATES_PATH}",
        f"Molecule manifest: {MOLECULE_CANDIDATES_PATH}",
        f"Structure manifest: {CANDIDATE_STRUCTURES_PATH}",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("STAGE 11.1 SUMMARY")
    print("=" * 100)

    for line in summary_lines:
        print(line)

    # --------------------------------------------------------
    # Strong regression checks against Stage 7 profile
    # --------------------------------------------------------

    expected_median = 107.0
    expected_mean = 142.48
    expected_max = 604.0

    observed_median = float(
        np.median(pool_sizes_array)
    )

    observed_mean = float(
        pool_sizes_array.mean()
    )

    observed_max = float(
        pool_sizes_array.max()
    )

    print()
    print("Regression check vs Stage 7 profile:")

    print(
        f"  Median: "
        f"{observed_median:.2f} "
        f"(expected ~{expected_median:.2f})"
    )

    print(
        f"  Mean: "
        f"{observed_mean:.2f} "
        f"(expected ~{expected_mean:.2f})"
    )

    print(
        f"  Max: "
        f"{observed_max:.2f} "
        f"(expected {expected_max:.0f})"
    )

    if fallback_count != 0:
        print(
            "\nWARNING: one or more molecule groups "
            "required union fallback."
        )

    print()
    print("=" * 100)
    print("STAGE 11.1 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()