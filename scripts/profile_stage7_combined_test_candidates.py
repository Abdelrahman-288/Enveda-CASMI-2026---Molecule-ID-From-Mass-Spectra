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

ENVEDA_VARIANTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "stage7_candidate_mass_variants.csv"
)

COMBINED_VARIANTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "stage7_combined_candidate_mass_variants.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage7_combined_test_candidate_profile.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage7_combined_test_candidate_profile_summary.txt"
)


def describe(values):

    values = np.asarray(
        values,
        dtype=np.int64,
    )

    return {
        "min": int(values.min()),
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
        "max": int(values.max()),
    }


def build_index(
    dataframe,
):

    return CandidateMassIndex(
        dataframe[
            [
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
                "exact_mass",
            ]
        ]
    )


def generate_candidate_set(
    generator,
    precursor_mz,
    adduct,
):

    result = generator.generate(
        precursor_mz=float(
            precursor_mz
        ),
        adduct=str(
            adduct
        ),
        require_trusted=True,
    )

    return {
        hit.inchikey14
        for hit in result.candidates
    }


def main():

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 7 STEP 10 "
        "COMBINED TEST CANDIDATE PROFILE"
    )

    # =====================================================
    # Load test
    # =====================================================

    test = pd.read_parquet(
        TEST_PATH
    )

    print(
        f"\nTest spectra: "
        f"{len(test):,}"
    )

    print(
        f"Test molecules: "
        f"{test['molecule_id'].nunique():,}"
    )

    # =====================================================
    # Load Enveda-only index
    # =====================================================

    print(
        "\nLoading Enveda-only index..."
    )

    enveda_variants = (
        pd.read_csv(
            ENVEDA_VARIANTS_PATH
        )
    )

    enveda_index = build_index(
        enveda_variants
    )

    print(
        f"Enveda structures: "
        f"{enveda_index.unique_structure_count:,}"
    )

    # =====================================================
    # Load combined index
    # =====================================================

    print(
        "\nLoading combined index..."
    )

    combined_variants = (
        pd.read_parquet(
            COMBINED_VARIANTS_PATH
        )
    )

    combined_index = build_index(
        combined_variants
    )

    print(
        f"Combined structures: "
        f"{combined_index.unique_structure_count:,}"
    )

    # =====================================================
    # Generators
    # =====================================================

    enveda_generator = (
        AdductAwareCandidateGenerator(
            mass_index=enveda_index,
            tolerance_ppm=10.0,
        )
    )

    combined_generator = (
        AdductAwareCandidateGenerator(
            mass_index=combined_index,
            tolerance_ppm=10.0,
        )
    )

    # =====================================================
    # Spectrum candidate sets
    # =====================================================

    enveda_sets = defaultdict(
        list
    )

    combined_sets = defaultdict(
        list
    )

    spectrum_records = []

    for row in test.itertuples(
        index=False
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

        if not decision.hard_filter:

            continue

        enveda_candidates = (
            generate_candidate_set(
                generator=enveda_generator,
                precursor_mz=(
                    row.precursor_mz
                ),
                adduct=adduct,
            )
        )

        combined_candidates = (
            generate_candidate_set(
                generator=combined_generator,
                precursor_mz=(
                    row.precursor_mz
                ),
                adduct=adduct,
            )
        )

        enveda_sets[
            molecule_id
        ].append(
            enveda_candidates
        )

        combined_sets[
            molecule_id
        ].append(
            combined_candidates
        )

        coconut_added = (
            combined_candidates
            - enveda_candidates
        )

        spectrum_records.append(
            {
                "molecule_id": (
                    molecule_id
                ),
                "adduct": adduct,
                "precursor_mz": float(
                    row.precursor_mz
                ),
                "enveda_candidate_count": (
                    len(
                        enveda_candidates
                    )
                ),
                "combined_candidate_count": (
                    len(
                        combined_candidates
                    )
                ),
                "coconut_added_count": (
                    len(
                        coconut_added
                    )
                ),
            }
        )

    spectrum_df = pd.DataFrame(
        spectrum_records
    )

    # =====================================================
    # Molecule aggregation
    # =====================================================

    molecule_records = []

    for molecule_id, group in test.groupby(
        "molecule_id",
        sort=False,
    ):

        molecule_id = str(
            molecule_id
        )

        env_sets = enveda_sets[
            molecule_id
        ]

        comb_sets = combined_sets[
            molecule_id
        ]

        env_union = set().union(
            *env_sets
        )

        comb_union = set().union(
            *comb_sets
        )

        env_intersection = set(
            env_sets[
                0
            ]
        )

        for candidate_set in (
            env_sets[
                1:
            ]
        ):

            env_intersection.intersection_update(
                candidate_set
            )

        comb_intersection = set(
            comb_sets[
                0
            ]
        )

        for candidate_set in (
            comb_sets[
                1:
            ]
        ):

            comb_intersection.intersection_update(
                candidate_set
            )

        coconut_only_intersection = (
            comb_intersection
            - env_intersection
        )

        molecule_records.append(
            {
                "molecule_id": (
                    molecule_id
                ),
                "spectrum_count": int(
                    len(group)
                ),
                "enveda_union_count": (
                    len(
                        env_union
                    )
                ),
                "combined_union_count": (
                    len(
                        comb_union
                    )
                ),
                "enveda_intersection_count": (
                    len(
                        env_intersection
                    )
                ),
                "combined_intersection_count": (
                    len(
                        comb_intersection
                    )
                ),
                "coconut_added_intersection": (
                    len(
                        coconut_only_intersection
                    )
                ),
                "intersection_growth_factor": (
                    (
                        len(
                            comb_intersection
                        )
                        /
                        len(
                            env_intersection
                        )
                    )
                    if len(
                        env_intersection
                    ) > 0
                    else np.nan
                ),
            }
        )

    molecule_df = pd.DataFrame(
        molecule_records
    )

    # =====================================================
    # Statistics
    # =====================================================

    env_stats = describe(
        molecule_df[
            "enveda_intersection_count"
        ]
    )

    combined_stats = describe(
        molecule_df[
            "combined_intersection_count"
        ]
    )

    added_stats = describe(
        molecule_df[
            "coconut_added_intersection"
        ]
    )

    growth = (
        molecule_df[
            "combined_intersection_count"
        ]
        /
        molecule_df[
            "enveda_intersection_count"
        ]
    )

    # =====================================================
    # Console output
    # =====================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "ENVEDA-ONLY VS COMBINED "
        "MOLECULE INTERSECTION CANDIDATES"
    )

    print(
        "=" * 110
    )

    print(
        f"{'Metric':20s}"
        f"{'Enveda':>16s}"
        f"{'Combined':>16s}"
        f"{'Difference':>16s}"
    )

    print(
        "-" * 110
    )

    for key in [
        "min",
        "median",
        "mean",
        "p90",
        "p95",
        "p99",
        "max",
    ]:

        env_value = (
            env_stats[
                key
            ]
        )

        comb_value = (
            combined_stats[
                key
            ]
        )

        difference = (
            comb_value
            - env_value
        )

        print(
            f"{key.upper():20s}"
            f"{env_value:16.2f}"
            f"{comb_value:16.2f}"
            f"{difference:16.2f}"
        )

    print(
        "\nCOCONUT-only candidates added "
        "to molecule intersections:"
    )

    print(
        f"Median added: "
        f"{added_stats['median']:.1f}"
    )

    print(
        f"Mean added: "
        f"{added_stats['mean']:.2f}"
    )

    print(
        f"P90 added: "
        f"{added_stats['p90']:.1f}"
    )

    print(
        f"P99 added: "
        f"{added_stats['p99']:.1f}"
    )

    print(
        f"Max added: "
        f"{added_stats['max']:,}"
    )

    print(
        f"\nMedian candidate growth factor: "
        f"{np.median(growth):.3f}x"
    )

    print(
        f"Mean candidate growth factor: "
        f"{growth.mean():.3f}x"
    )

    print(
        f"Maximum candidate growth factor: "
        f"{growth.max():.3f}x"
    )

    # =====================================================
    # Most expanded molecules
    # =====================================================

    print(
        "\nTop 20 molecules by COCONUT "
        "candidate additions:"
    )

    top = (
        molecule_df.sort_values(
            "coconut_added_intersection",
            ascending=False,
        )
        .head(
            20
        )
    )

    print(
        top[
            [
                "molecule_id",
                "spectrum_count",
                "enveda_intersection_count",
                "combined_intersection_count",
                "coconut_added_intersection",
                "intersection_growth_factor",
            ]
        ].to_string(
            index=False
        )
    )

    # =====================================================
    # Save output
    # =====================================================

    molecule_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary_lines = [
        (
            "ENVEDA CASMI 2026 — "
            "STAGE 7 COMBINED TEST CANDIDATE PROFILE"
        ),
        "",
        (
            f"Enveda structures: "
            f"{enveda_index.unique_structure_count:,}"
        ),
        (
            f"Combined structures: "
            f"{combined_index.unique_structure_count:,}"
        ),
        "",
        "ENVEDA INTERSECTION",
        (
            f"Median: "
            f"{env_stats['median']:.2f}"
        ),
        (
            f"Mean: "
            f"{env_stats['mean']:.2f}"
        ),
        (
            f"P90: "
            f"{env_stats['p90']:.2f}"
        ),
        (
            f"P99: "
            f"{env_stats['p99']:.2f}"
        ),
        (
            f"Max: "
            f"{env_stats['max']:,}"
        ),
        "",
        "COMBINED INTERSECTION",
        (
            f"Median: "
            f"{combined_stats['median']:.2f}"
        ),
        (
            f"Mean: "
            f"{combined_stats['mean']:.2f}"
        ),
        (
            f"P90: "
            f"{combined_stats['p90']:.2f}"
        ),
        (
            f"P99: "
            f"{combined_stats['p99']:.2f}"
        ),
        (
            f"Max: "
            f"{combined_stats['max']:,}"
        ),
        "",
        "COCONUT ADDITIONS",
        (
            f"Median: "
            f"{added_stats['median']:.2f}"
        ),
        (
            f"Mean: "
            f"{added_stats['mean']:.2f}"
        ),
        (
            f"P90: "
            f"{added_stats['p90']:.2f}"
        ),
        (
            f"P99: "
            f"{added_stats['p99']:.2f}"
        ),
        (
            f"Max: "
            f"{added_stats['max']:,}"
        ),
        "",
        (
            "Median growth factor: "
            f"{np.median(growth):.4f}x"
        ),
        (
            "Mean growth factor: "
            f"{growth.mean():.4f}x"
        ),
        (
            "Maximum growth factor: "
            f"{growth.max():.4f}x"
        ),
    ]

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print(
        "\nDetailed profile:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "STAGE 7 STEP 10 COMPLETE"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()