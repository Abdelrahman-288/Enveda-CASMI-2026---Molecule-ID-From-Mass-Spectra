from pathlib import Path
from collections import defaultdict

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

TEST_PATH = (
    DATASET_DIR
    / "test.parquet"
)

VARIANT_PATH = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "stage7_candidate_mass_variants.csv"
)

DETAIL_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage7_test_candidate_profile.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage7_test_candidate_profile_summary.txt"
)


# =========================================================
# Helpers
# =========================================================

def describe_counts(
    values,
):
    values = np.asarray(
        values,
        dtype=np.int64,
    )

    if len(values) == 0:
        return {
            "min": 0,
            "median": 0.0,
            "mean": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }

    return {
        "min": int(
            values.min()
        ),
        "median": float(
            np.median(values)
        ),
        "mean": float(
            values.mean()
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
        "max": int(
            values.max()
        ),
    }


def print_stats(
    title,
    values,
):

    stats = describe_counts(
        values
    )

    print(
        "\n"
        + title
    )

    print(
        "-" * len(title)
    )

    print(
        f"Min:    "
        f"{stats['min']:,}"
    )

    print(
        f"Median: "
        f"{stats['median']:.1f}"
    )

    print(
        f"Mean:   "
        f"{stats['mean']:.2f}"
    )

    print(
        f"P90:    "
        f"{stats['p90']:.1f}"
    )

    print(
        f"P95:    "
        f"{stats['p95']:.1f}"
    )

    print(
        f"P99:    "
        f"{stats['p99']:.1f}"
    )

    print(
        f"Max:    "
        f"{stats['max']:,}"
    )

    return stats


# =========================================================
# Main
# =========================================================

def main():

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 7 STEP 6 "
        "TEST CANDIDATE PROFILING"
    )

    # =====================================================
    # Load test set
    # =====================================================

    print(
        "\nLoading competition test set..."
    )

    test = pd.read_parquet(
        TEST_PATH
    )

    required_columns = {
        "molecule_id",
        "precursor_mz",
        "adduct",
        "instrument_type",
    }

    missing = (
        required_columns
        - set(
            test.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Missing required test columns: "
            f"{sorted(missing)}"
        )

    print(
        f"Test spectra: "
        f"{len(test):,}"
    )

    print(
        f"Test molecules: "
        f"{test['molecule_id'].nunique():,}"
    )

    # =====================================================
    # Load mass variants
    # =====================================================

    print(
        "\nLoading Stage 7 mass variants..."
    )

    variants = pd.read_csv(
        VARIANT_PATH
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
    # Generator cache by tolerance
    # =====================================================

    generators = {}

    spectrum_records = []

    candidate_sets_by_molecule = (
        defaultdict(list)
    )

    hard_filter_rows_by_molecule = (
        defaultdict(int)
    )

    # =====================================================
    # Per-spectrum candidate generation
    # =====================================================

    print(
        "\nProfiling test spectra..."
    )

    for row_index, row in enumerate(
        test.itertuples(
            index=False
        )
    ):

        molecule_id = str(
            row.molecule_id
        )

        adduct = str(
            row.adduct
        )

        instrument = str(
            row.instrument_type
        )

        precursor_mz = float(
            row.precursor_mz
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
        )

        decision = (
            get_mass_filter_decision(
                adduct_supported=supported,
                adduct_trusted=trusted,
                instrument_type=instrument,
                ingest_lib=None,
                inference_mode=True,
            )
        )

        candidate_set = set()

        neutral_mass = np.nan

        if (
            decision.hard_filter
            and decision.tolerance_ppm
            is not None
        ):

            tolerance = float(
                decision.tolerance_ppm
            )

            if tolerance not in generators:

                generators[
                    tolerance
                ] = (
                    AdductAwareCandidateGenerator(
                        mass_index=mass_index,
                        tolerance_ppm=tolerance,
                    )
                )

            generator = (
                generators[
                    tolerance
                ]
            )

            result = generator.generate(
                precursor_mz=precursor_mz,
                adduct=adduct,
                require_trusted=True,
            )

            if (
                result.neutral_mass
                is not None
            ):

                neutral_mass = float(
                    result.neutral_mass
                )

            candidate_set = {
                hit.inchikey14
                for hit in result.candidates
            }

            candidate_sets_by_molecule[
                molecule_id
            ].append(
                candidate_set
            )

            hard_filter_rows_by_molecule[
                molecule_id
            ] += 1

        spectrum_records.append(
            {
                "row_index": row_index,
                "molecule_id": molecule_id,
                "precursor_mz": precursor_mz,
                "adduct": adduct,
                "instrument_type": instrument,
                "adduct_supported": supported,
                "adduct_trusted": trusted,
                "hard_filter": (
                    decision.hard_filter
                ),
                "tolerance_ppm": (
                    decision.tolerance_ppm
                ),
                "confidence": (
                    decision.confidence
                ),
                "neutral_mass": neutral_mass,
                "candidate_count": (
                    len(
                        candidate_set
                    )
                ),
            }
        )

    spectrum_df = pd.DataFrame(
        spectrum_records
    )

    # =====================================================
    # Per-molecule aggregation
    # =====================================================

    molecule_records = []

    grouped_test = test.groupby(
        "molecule_id",
        sort=False,
    )

    for molecule_id, group in grouped_test:

        molecule_id = str(
            molecule_id
        )

        candidate_sets = (
            candidate_sets_by_molecule.get(
                molecule_id,
                [],
            )
        )

        if candidate_sets:

            union_set = set().union(
                *candidate_sets
            )

            intersection_set = set(
                candidate_sets[
                    0
                ]
            )

            for candidate_set in (
                candidate_sets[
                    1:
                ]
            ):

                intersection_set.intersection_update(
                    candidate_set
                )

        else:

            union_set = set()

            intersection_set = set()

        individual_counts = [
            len(
                candidate_set
            )
            for candidate_set
            in candidate_sets
        ]

        molecule_records.append(
            {
                "molecule_id": (
                    molecule_id
                ),
                "spectrum_count": int(
                    len(
                        group
                    )
                ),
                "hard_filter_spectrum_count": int(
                    hard_filter_rows_by_molecule.get(
                        molecule_id,
                        0,
                    )
                ),
                "min_spectrum_candidates": (
                    min(
                        individual_counts
                    )
                    if individual_counts
                    else 0
                ),
                "max_spectrum_candidates": (
                    max(
                        individual_counts
                    )
                    if individual_counts
                    else 0
                ),
                "union_candidate_count": int(
                    len(
                        union_set
                    )
                ),
                "intersection_candidate_count": int(
                    len(
                        intersection_set
                    )
                ),
                "intersection_empty": (
                    len(
                        intersection_set
                    )
                    == 0
                ),
                "union_empty": (
                    len(
                        union_set
                    )
                    == 0
                ),
            }
        )

    molecule_df = pd.DataFrame(
        molecule_records
    )

    # =====================================================
    # Summary
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TEST CANDIDATE PROFILE"
    )

    print(
        "=" * 100
    )

    hard_rows = int(
        spectrum_df[
            "hard_filter"
        ].sum()
    )

    unsupported_rows = int(
        (
            ~spectrum_df[
                "adduct_supported"
            ]
        ).sum()
    )

    untrusted_rows = int(
        (
            spectrum_df[
                "adduct_supported"
            ]
            &
            ~spectrum_df[
                "adduct_trusted"
            ]
        ).sum()
    )

    zero_spectrum_candidates = int(
        (
            spectrum_df[
                "candidate_count"
            ]
            == 0
        ).sum()
    )

    print(
        f"Hard-filter spectra: "
        f"{hard_rows:,}/"
        f"{len(spectrum_df):,}"
    )

    print(
        f"Unsupported-adduct spectra: "
        f"{unsupported_rows:,}"
    )

    print(
        f"Untrusted-adduct spectra: "
        f"{untrusted_rows:,}"
    )

    print(
        f"Zero-candidate spectra: "
        f"{zero_spectrum_candidates:,}"
    )

    spectrum_stats = print_stats(
        "PER-SPECTRUM CANDIDATE COUNTS",
        spectrum_df[
            "candidate_count"
        ].to_numpy(),
    )

    union_stats = print_stats(
        "PER-MOLECULE UNION CANDIDATE COUNTS",
        molecule_df[
            "union_candidate_count"
        ].to_numpy(),
    )

    intersection_stats = print_stats(
        "PER-MOLECULE INTERSECTION CANDIDATE COUNTS",
        molecule_df[
            "intersection_candidate_count"
        ].to_numpy(),
    )

    empty_intersections = int(
        molecule_df[
            "intersection_empty"
        ].sum()
    )

    empty_unions = int(
        molecule_df[
            "union_empty"
        ].sum()
    )

    print(
        "\nMulti-spectrum aggregation:"
    )

    print(
        f"Empty intersections: "
        f"{empty_intersections:,}/"
        f"{len(molecule_df):,}"
    )

    print(
        f"Empty unions: "
        f"{empty_unions:,}/"
        f"{len(molecule_df):,}"
    )

    # =====================================================
    # Largest candidate sets
    # =====================================================

    print(
        "\nTop 20 molecules by "
        "intersection candidate count:"
    )

    top_intersections = (
        molecule_df.sort_values(
            "intersection_candidate_count",
            ascending=False,
        )
        .head(
            20
        )
    )

    print(
        top_intersections[
            [
                "molecule_id",
                "spectrum_count",
                "union_candidate_count",
                "intersection_candidate_count",
            ]
        ].to_string(
            index=False
        )
    )

    # =====================================================
    # Save detailed output
    # =====================================================

    DETAIL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    molecule_df.to_csv(
        DETAIL_OUTPUT_PATH,
        index=False,
    )

    # =====================================================
    # Save summary
    # =====================================================

    summary_lines = [
        (
            "ENVEDA CASMI 2026 — "
            "STAGE 7 TEST CANDIDATE PROFILE"
        ),
        "",
        (
            f"Test spectra: "
            f"{len(spectrum_df):,}"
        ),
        (
            f"Test molecules: "
            f"{len(molecule_df):,}"
        ),
        (
            f"Hard-filter spectra: "
            f"{hard_rows:,}"
        ),
        (
            f"Unsupported-adduct spectra: "
            f"{unsupported_rows:,}"
        ),
        (
            f"Untrusted-adduct spectra: "
            f"{untrusted_rows:,}"
        ),
        (
            f"Zero-candidate spectra: "
            f"{zero_spectrum_candidates:,}"
        ),
        "",
        "Per-spectrum candidates:",
        (
            f"Median: "
            f"{spectrum_stats['median']:.1f}"
        ),
        (
            f"Mean: "
            f"{spectrum_stats['mean']:.2f}"
        ),
        (
            f"P90: "
            f"{spectrum_stats['p90']:.1f}"
        ),
        (
            f"P99: "
            f"{spectrum_stats['p99']:.1f}"
        ),
        (
            f"Max: "
            f"{spectrum_stats['max']:,}"
        ),
        "",
        "Per-molecule union:",
        (
            f"Median: "
            f"{union_stats['median']:.1f}"
        ),
        (
            f"Mean: "
            f"{union_stats['mean']:.2f}"
        ),
        (
            f"P90: "
            f"{union_stats['p90']:.1f}"
        ),
        (
            f"Max: "
            f"{union_stats['max']:,}"
        ),
        "",
        "Per-molecule intersection:",
        (
            f"Median: "
            f"{intersection_stats['median']:.1f}"
        ),
        (
            f"Mean: "
            f"{intersection_stats['mean']:.2f}"
        ),
        (
            f"P90: "
            f"{intersection_stats['p90']:.1f}"
        ),
        (
            f"P99: "
            f"{intersection_stats['p99']:.1f}"
        ),
        (
            f"Max: "
            f"{intersection_stats['max']:,}"
        ),
        "",
        (
            f"Empty intersections: "
            f"{empty_intersections:,}"
        ),
        (
            f"Empty unions: "
            f"{empty_unions:,}"
        ),
    ]

    SUMMARY_OUTPUT_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print(
        "\nDetailed molecule profile:"
    )

    print(
        DETAIL_OUTPUT_PATH
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 7 STEP 6 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()