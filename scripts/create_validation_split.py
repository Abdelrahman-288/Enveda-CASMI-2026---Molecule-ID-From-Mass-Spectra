from pathlib import Path
import hashlib

import pandas as pd
import pyarrow.parquet as pq


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TRAIN_PATH = DATASET_DIR / "train.parquet"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STRUCTURE_SPLIT_PATH = (
    OUTPUT_DIR
    / "structure_validation_split.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "structure_validation_split_summary.txt"
)


# =========================================================
# Configuration
# =========================================================

VALIDATION_FRACTION = 0.10

HASH_SEED = "casmi2026_stage5_v1"

TEST_MIN_MZ = 245.0931
TEST_MAX_MZ = 460.1554

TEST_ADDUCTS = {
    "[M+H]+",
    "[M-H]-",
    "[M+CH2O2-H]-",
    "[M+Na]+",
    "[M+NH4]+",
    "[M+K]+",
    "[M+Cl]-",
}


# =========================================================
# Stable split
# =========================================================

def stable_fraction(
    value: str,
) -> float:
    """
    Convert a string into a deterministic number in [0, 1).

    This makes the structure split reproducible across machines
    without relying on random-number-generator state.
    """

    text = (
        f"{HASH_SEED}|{value}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        text
    ).hexdigest()

    integer = int(
        digest[:16],
        16,
    )

    maximum = float(
        16**16
    )

    return integer / maximum


def assign_split(
    inchikey14: str,
) -> str:
    """
    Assign an entire molecular structure to train or validation.
    """

    fraction = stable_fraction(
        inchikey14
    )

    if fraction < VALIDATION_FRACTION:
        return "validation"

    return "train"


# =========================================================
# Main
# =========================================================

def main():

    print(
        "\nENVEDA CASMI 2026 "
        "— STAGE 5 STEP 1 "
        "STRUCTURE-DISJOINT VALIDATION"
    )

    columns = [
        "inchikey14",
        "normalized_smiles",
        "molecular_formula",
        "ingest_lib",
        "instrument_type",
        "adduct",
        "precursor_mz",
        "ionization_mode",
    ]

    print(
        "\nLoading lightweight metadata..."
    )

    train = pq.read_table(
        TRAIN_PATH,
        columns=columns,
    ).to_pandas()

    print(
        f"Training spectra loaded: "
        f"{len(train):,}"
    )

    # -----------------------------------------------------
    # Spectrum-level test-like flag
    # -----------------------------------------------------

    train["is_test_like_spectrum"] = (
        train["instrument_type"].eq(
            "timsTOF"
        )
        &
        train["precursor_mz"].between(
            TEST_MIN_MZ,
            TEST_MAX_MZ,
        )
        &
        train["adduct"].isin(
            TEST_ADDUCTS
        )
    )

    # -----------------------------------------------------
    # Collapse to one row per molecular structure
    # -----------------------------------------------------

    print(
        "Building structure-level metadata..."
    )

    structure_df = (
        train
        .groupby(
            "inchikey14",
            as_index=False,
        )
        .agg(
            normalized_smiles=(
                "normalized_smiles",
                "first",
            ),

            molecular_formula=(
                "molecular_formula",
                "first",
            ),

            spectrum_count=(
                "inchikey14",
                "size",
            ),

            timstof_spectra=(
                "instrument_type",
                lambda x: (
                    x == "timsTOF"
                ).sum(),
            ),

            test_like_spectra=(
                "is_test_like_spectrum",
                "sum",
            ),

            library_count=(
                "ingest_lib",
                "nunique",
            ),

            adduct_count=(
                "adduct",
                "nunique",
            ),

            ionization_mode_count=(
                "ionization_mode",
                "nunique",
            ),
        )
    )

    # -----------------------------------------------------
    # Stable structure-disjoint split
    # -----------------------------------------------------

    structure_df["split"] = (
        structure_df[
            "inchikey14"
        ].apply(
            assign_split
        )
    )

    structure_df[
        "has_timstof"
    ] = (
        structure_df[
            "timstof_spectra"
        ] > 0
    )

    structure_df[
        "has_test_like_spectrum"
    ] = (
        structure_df[
            "test_like_spectra"
        ] > 0
    )

    # -----------------------------------------------------
    # Verify no overlap
    # -----------------------------------------------------

    train_keys = set(
        structure_df.loc[
            structure_df["split"]
            == "train",
            "inchikey14",
        ]
    )

    val_keys = set(
        structure_df.loc[
            structure_df["split"]
            == "validation",
            "inchikey14",
        ]
    )

    overlap = (
        train_keys
        & val_keys
    )

    if overlap:
        raise RuntimeError(
            "Structure leakage detected "
            "between train and validation."
        )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    total_structures = len(
        structure_df
    )

    train_structures = (
        structure_df["split"]
        .eq("train")
        .sum()
    )

    val_structures = (
        structure_df["split"]
        .eq("validation")
        .sum()
    )

    train_spectrum_count = (
        structure_df.loc[
            structure_df["split"]
            == "train",
            "spectrum_count",
        ].sum()
    )

    val_spectrum_count = (
        structure_df.loc[
            structure_df["split"]
            == "validation",
            "spectrum_count",
        ].sum()
    )

    val_timstof_structures = (
        structure_df.loc[
            structure_df["split"]
            == "validation",
            "has_timstof",
        ].sum()
    )

    val_test_like_structures = (
        structure_df.loc[
            structure_df["split"]
            == "validation",
            "has_test_like_spectrum",
        ].sum()
    )

    val_test_like_spectra = (
        structure_df.loc[
            structure_df["split"]
            == "validation",
            "test_like_spectra",
        ].sum()
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    structure_df.to_csv(
        STRUCTURE_SPLIT_PATH,
        index=False,
    )

    summary = f"""
ENVEDA CASMI 2026
Stage 5 Structure-Disjoint Validation Split

Seed:
{HASH_SEED}

Validation fraction:
{VALIDATION_FRACTION}

Total structures:
{total_structures:,}

Train structures:
{train_structures:,}

Validation structures:
{val_structures:,}

Train spectra:
{train_spectrum_count:,}

Validation spectra:
{val_spectrum_count:,}

Structure overlap:
{len(overlap)}

Validation structures with timsTOF:
{val_timstof_structures:,}

Validation structures with test-like spectra:
{val_test_like_structures:,}

Validation test-like spectra:
{val_test_like_spectra:,}
""".strip()

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "STRUCTURE SPLIT SUMMARY"
    )

    print(
        "=" * 90
    )

    print(summary)

    print(
        "\nSaved:"
    )

    print(
        STRUCTURE_SPLIT_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "STAGE 5 STEP 1 COMPLETE"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()