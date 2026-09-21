from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from casmi.chemistry.adducts import (
    is_supported_adduct,
    precursor_to_neutral_mass,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
    / "train.parquet"
)

SEED = 42
MAX_ROWS = 100_000

TOLERANCES = [
    5.0,
    10.0,
    20.0,
    50.0,
]


def exact_mass_from_smiles(
    smiles: str,
) -> float:

    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:
        return np.nan

    return float(
        Descriptors.ExactMolWt(
            mol
        )
    )


def ppm_error(
    observed_mass: float,
    expected_mass: float,
) -> float:

    return (
        (
            observed_mass
            - expected_mass
        )
        / expected_mass
        * 1e6
    )


def main():

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 7 TRUE-STRUCTURE MASS ERROR DIAGNOSTIC"
    )

    # =====================================================
    # Load data
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
    # Build all true structure mass variants
    # =====================================================

    print(
        "\nBuilding true-structure mass variants..."
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

    # Keep every genuinely distinct mass for the same
    # connectivity identifier.
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

    structure_masses = {}

    for key, group in variants.groupby(
        "inchikey14",
        sort=False,
    ):

        structure_masses[
            str(key)
        ] = (
            group[
                "exact_mass"
            ]
            .to_numpy(
                dtype=np.float64,
            )
        )

    print(
        f"Structures with mass records: "
        f"{len(structure_masses):,}"
    )

    # =====================================================
    # Eligible observed spectra
    # =====================================================

    supported_mask = (
        train[
            "adduct"
        ]
        .map(
            is_supported_adduct
        )
    )

    precursor_mask = (
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
        & precursor_mask
    ].copy()

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
    # Diagnostics
    # =====================================================

    stats = defaultdict(
        lambda: {
            "rows": 0,
            "errors": [],
            "neutral_masses": [],
        }
    )

    global_errors = []

    conversion_failures = 0
    missing_structure_mass = 0

    for row in eligible.itertuples(
        index=False
    ):

        key = str(
            row.inchikey14
        )

        adduct = str(
            row.adduct
        )

        masses = structure_masses.get(
            key
        )

        if masses is None:

            missing_structure_mass += 1
            continue

        try:

            neutral_mass = (
                precursor_to_neutral_mass(
                    precursor_mz=float(
                        row.precursor_mz
                    ),
                    adduct=adduct,
                    require_trusted=False,
                )
            )

        except Exception:

            conversion_failures += 1
            continue

        if (
            not np.isfinite(
                neutral_mass
            )
            or neutral_mass <= 0
        ):
            conversion_failures += 1
            continue

        # ---------------------------------------------
        # Use the closest legitimate mass variant for
        # the true inchikey14.
        # ---------------------------------------------

        errors = np.asarray(
            [
                ppm_error(
                    neutral_mass,
                    exact_mass,
                )
                for exact_mass
                in masses
            ],
            dtype=np.float64,
        )

        best_index = int(
            np.argmin(
                np.abs(
                    errors
                )
            )
        )

        best_error = float(
            errors[
                best_index
            ]
        )

        stats[
            adduct
        ][
            "rows"
        ] += 1

        stats[
            adduct
        ][
            "errors"
        ].append(
            best_error
        )

        stats[
            adduct
        ][
            "neutral_masses"
        ].append(
            neutral_mass
        )

        global_errors.append(
            best_error
        )

    # =====================================================
    # Per-adduct report
    # =====================================================

    print(
        "\n"
        + "=" * 145
    )

    header = (
        f"{'Adduct':20s}"
        f"{'Rows':>10s}"
        f"{'Median |ppm|':>15s}"
        f"{'P90 |ppm|':>13s}"
        f"{'P99 |ppm|':>13s}"
        f"{'<=5ppm':>11s}"
        f"{'<=10ppm':>11s}"
        f"{'<=20ppm':>11s}"
        f"{'<=50ppm':>11s}"
    )

    print(
        header
    )

    print(
        "-" * 145
    )

    ordered = sorted(
        stats.items(),
        key=lambda item: (
            -item[
                1
            ][
                "rows"
            ]
        ),
    )

    for adduct, record in ordered:

        errors = np.asarray(
            record[
                "errors"
            ],
            dtype=np.float64,
        )

        absolute_errors = np.abs(
            errors
        )

        row_count = len(
            absolute_errors
        )

        if row_count == 0:
            continue

        median_error = float(
            np.median(
                absolute_errors
            )
        )

        p90_error = float(
            np.percentile(
                absolute_errors,
                90,
            )
        )

        p99_error = float(
            np.percentile(
                absolute_errors,
                99,
            )
        )

        rates = {}

        for tolerance in TOLERANCES:

            rates[
                tolerance
            ] = float(
                np.mean(
                    absolute_errors
                    <= tolerance
                )
            )

        print(
            f"{adduct:20s}"
            f"{row_count:10,d}"
            f"{median_error:15.3f}"
            f"{p90_error:13.3f}"
            f"{p99_error:13.3f}"
            f"{rates[5.0]:11.4%}"
            f"{rates[10.0]:11.4%}"
            f"{rates[20.0]:11.4%}"
            f"{rates[50.0]:11.4%}"
        )

    # =====================================================
    # Overall report
    # =====================================================

    global_errors = np.asarray(
        global_errors,
        dtype=np.float64,
    )

    absolute_global = np.abs(
        global_errors
    )

    print(
        "\n"
        + "=" * 145
    )

    print(
        "OVERALL TRUE-STRUCTURE MASS ERROR"
    )

    print(
        "=" * 145
    )

    print(
        f"Valid rows: "
        f"{len(absolute_global):,}"
    )

    print(
        f"Conversion failures: "
        f"{conversion_failures:,}"
    )

    print(
        f"Missing structure masses: "
        f"{missing_structure_mass:,}"
    )

    print(
        f"Median |ppm|: "
        f"{np.median(absolute_global):.4f}"
    )

    print(
        f"P90 |ppm|: "
        f"{np.percentile(absolute_global, 90):.4f}"
    )

    print(
        f"P95 |ppm|: "
        f"{np.percentile(absolute_global, 95):.4f}"
    )

    print(
        f"P99 |ppm|: "
        f"{np.percentile(absolute_global, 99):.4f}"
    )

    print()

    for tolerance in TOLERANCES:

        rate = float(
            np.mean(
                absolute_global
                <= tolerance
            )
        )

        print(
            f"Within "
            f"{tolerance:5.1f} ppm: "
            f"{rate:.6%}"
        )

    # =====================================================
    # Extreme error diagnostics
    # =====================================================

    print(
        "\n"
        + "=" * 145
    )

    print(
        "OUTLIER COUNTS"
    )

    print(
        "=" * 145
    )

    thresholds = [
        10,
        20,
        50,
        100,
        500,
        1000,
    ]

    for threshold in thresholds:

        count = int(
            np.sum(
                absolute_global
                > threshold
            )
        )

        rate = (
            count
            / len(
                absolute_global
            )
        )

        print(
            f"|ppm| > "
            f"{threshold:4d}: "
            f"{count:7,d} "
            f"({rate:.4%})"
        )

    print(
        "\n"
        + "=" * 145
    )

    print(
        "STAGE 7 TRUE-MASS DIAGNOSTIC COMPLETE"
    )

    print(
        "=" * 145
    )


if __name__ == "__main__":
    main()