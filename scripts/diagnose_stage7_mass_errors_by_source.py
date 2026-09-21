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

MAX_ROWS = 250_000

TOLERANCE_PPM = 10.0


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


def main():

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 7 MASS ERROR BY SOURCE"
    )

    # =====================================================
    # Load data
    # =====================================================

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "ingest_lib",
            "instrument_type",
            "inchikey14",
            "normalized_smiles",
            "precursor_mz",
            "precursor_error_ppm",
            "adduct",
        ],
    )

    print(
        f"\nTraining rows: "
        f"{len(train):,}"
    )

    # =====================================================
    # Valid supported rows
    # =====================================================

    mask = (
        train[
            "adduct"
        ].map(
            is_supported_adduct
        )
        &
        train[
            "precursor_mz"
        ].notna()
        &
        (
            train[
                "precursor_mz"
            ]
            > 0
        )
    )

    eligible = (
        train[
            mask
        ]
        .copy()
    )

    print(
        f"Eligible rows: "
        f"{len(eligible):,}"
    )

    # =====================================================
    # Random sample over whole dataset
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
        f"Rows sampled: "
        f"{len(eligible):,}"
    )

    # =====================================================
    # Cache structure masses
    # =====================================================

    unique_smiles = (
        eligible[
            "normalized_smiles"
        ]
        .drop_duplicates()
    )

    mass_lookup = {}

    print(
        "\nComputing RDKit masses..."
    )

    for smiles in unique_smiles:

        mass_lookup[
            smiles
        ] = (
            exact_mass_from_smiles(
                smiles
            )
        )

    # =====================================================
    # Statistics
    # =====================================================

    source_stats = defaultdict(
        lambda: {
            "errors": [],
            "reported_errors": [],
        }
    )

    source_adduct_stats = defaultdict(
        list
    )

    instrument_stats = defaultdict(
        list
    )

    valid_rows = 0

    for row in eligible.itertuples(
        index=False
    ):

        exact_mass = (
            mass_lookup.get(
                row.normalized_smiles
            )
        )

        if (
            exact_mass is None
            or not np.isfinite(
                exact_mass
            )
        ):
            continue

        neutral_mass = (
            precursor_to_neutral_mass(
                precursor_mz=float(
                    row.precursor_mz
                ),
                adduct=str(
                    row.adduct
                ),
                require_trusted=False,
            )
        )

        error_ppm = (
            (
                neutral_mass
                - exact_mass
            )
            / exact_mass
            * 1e6
        )

        abs_error = abs(
            float(
                error_ppm
            )
        )

        source = str(
            row.ingest_lib
        )

        instrument = str(
            row.instrument_type
        )

        adduct = str(
            row.adduct
        )

        source_stats[
            source
        ][
            "errors"
        ].append(
            abs_error
        )

        reported_error = (
            row.precursor_error_ppm
        )

        if (
            reported_error is not None
            and np.isfinite(
                reported_error
            )
        ):

            source_stats[
                source
            ][
                "reported_errors"
            ].append(
                abs(
                    float(
                        reported_error
                    )
                )
            )

        source_adduct_stats[
            (
                source,
                adduct,
            )
        ].append(
            abs_error
        )

        instrument_stats[
            instrument
        ].append(
            abs_error
        )

        valid_rows += 1

    print(
        f"\nValid comparisons: "
        f"{valid_rows:,}"
    )

    # =====================================================
    # Source table
    # =====================================================

    print(
        "\n"
        + "=" * 130
    )

    print(
        "SURVIVAL BY INGEST SOURCE"
    )

    print(
        "=" * 130
    )

    print(
        f"{'Source':35s}"
        f"{'Rows':>10s}"
        f"{'Median ppm':>14s}"
        f"{'P95 ppm':>12s}"
        f"{'P99 ppm':>12s}"
        f"{'<=10 ppm':>13s}"
        f"{'>100 ppm':>13s}"
    )

    print(
        "-" * 130
    )

    ordered_sources = sorted(
        source_stats.items(),
        key=lambda item: (
            -len(
                item[
                    1
                ][
                    "errors"
                ]
            )
        ),
    )

    for source, record in ordered_sources:

        errors = np.asarray(
            record[
                "errors"
            ],
            dtype=np.float64,
        )

        survival = float(
            np.mean(
                errors
                <= TOLERANCE_PPM
            )
        )

        huge_error = float(
            np.mean(
                errors
                > 100.0
            )
        )

        print(
            f"{source[:35]:35s}"
            f"{len(errors):10,d}"
            f"{np.median(errors):14.3f}"
            f"{np.percentile(errors, 95):12.3f}"
            f"{np.percentile(errors, 99):12.3f}"
            f"{survival:13.4%}"
            f"{huge_error:13.4%}"
        )

    # =====================================================
    # Instrument table
    # =====================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "SURVIVAL BY INSTRUMENT"
    )

    print(
        "=" * 110
    )

    print(
        f"{'Instrument':30s}"
        f"{'Rows':>10s}"
        f"{'Median ppm':>14s}"
        f"{'P95 ppm':>12s}"
        f"{'<=10 ppm':>13s}"
    )

    print(
        "-" * 110
    )

    ordered_instruments = sorted(
        instrument_stats.items(),
        key=lambda item: (
            -len(
                item[
                    1
                ]
            )
        ),
    )

    for instrument, values in ordered_instruments:

        errors = np.asarray(
            values,
            dtype=np.float64,
        )

        survival = float(
            np.mean(
                errors
                <= TOLERANCE_PPM
            )
        )

        print(
            f"{instrument[:30]:30s}"
            f"{len(errors):10,d}"
            f"{np.median(errors):14.3f}"
            f"{np.percentile(errors, 95):12.3f}"
            f"{survival:13.4%}"
        )

    # =====================================================
    # Source + adduct
    # =====================================================

    print(
        "\n"
        + "=" * 145
    )

    print(
        "SOURCE + ADDUCT COMBINATIONS"
    )

    print(
        "=" * 145
    )

    print(
        f"{'Source':32s}"
        f"{'Adduct':18s}"
        f"{'Rows':>9s}"
        f"{'Median ppm':>14s}"
        f"{'P95 ppm':>12s}"
        f"{'<=10 ppm':>13s}"
    )

    print(
        "-" * 145
    )

    combinations = sorted(
        source_adduct_stats.items(),
        key=lambda item: (
            -len(
                item[
                    1
                ]
            )
        ),
    )

    for (
        source,
        adduct,
    ), values in combinations:

        if len(
            values
        ) < 100:
            continue

        errors = np.asarray(
            values,
            dtype=np.float64,
        )

        survival = float(
            np.mean(
                errors
                <= TOLERANCE_PPM
            )
        )

        print(
            f"{source[:32]:32s}"
            f"{adduct[:18]:18s}"
            f"{len(errors):9,d}"
            f"{np.median(errors):14.3f}"
            f"{np.percentile(errors, 95):12.3f}"
            f"{survival:13.4%}"
        )

    print(
        "\n"
        + "=" * 145
    )

    print(
        "STAGE 7 SOURCE DIAGNOSTIC COMPLETE"
    )

    print(
        "=" * 145
    )


if __name__ == "__main__":
    main()