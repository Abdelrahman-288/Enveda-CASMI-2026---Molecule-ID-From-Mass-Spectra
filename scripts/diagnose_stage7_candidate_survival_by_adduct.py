from pathlib import Path
from collections import defaultdict

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
TOLERANCE_PPM = 10.0


def exact_mass_from_smiles(smiles: str) -> float:

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.nan

    return float(
        Descriptors.ExactMolWt(mol)
    )


def main():

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 7 CANDIDATE SURVIVAL BY ADDUCT"
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

    variants = (
        train[
            [
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    variants["exact_mass"] = (
        variants["normalized_smiles"]
        .map(exact_mass_from_smiles)
    )

    variants = (
        variants
        .dropna(subset=["exact_mass"])
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
        .reset_index(drop=True)
    )

    index = CandidateMassIndex(
        variants
    )

    generator = (
        AdductAwareCandidateGenerator(
            mass_index=index,
            tolerance_ppm=TOLERANCE_PPM,
        )
    )

    supported = train[
        train["adduct"].map(
            is_supported_adduct
        )
        & train["adduct"].map(
            is_trusted_for_mass_filter
        )
        & train["precursor_mz"].notna()
        & (train["precursor_mz"] > 0)
    ].copy()

    if len(supported) > MAX_ROWS:

        supported = (
            supported.sample(
                n=MAX_ROWS,
                random_state=SEED,
            )
            .reset_index(drop=True)
        )

    stats = defaultdict(
        lambda: {
            "rows": 0,
            "survived": 0,
            "zero": 0,
            "candidate_counts": [],
        }
    )

    for row in supported.itertuples(
        index=False
    ):

        adduct = str(
            row.adduct
        )

        result = generator.generate(
            precursor_mz=float(
                row.precursor_mz
            ),
            adduct=adduct,
            require_trusted=True,
        )

        keys = {
            hit.inchikey14
            for hit in result.candidates
        }

        count = len(keys)

        record = stats[adduct]

        record["rows"] += 1

        record[
            "candidate_counts"
        ].append(
            count
        )

        if count == 0:
            record["zero"] += 1

        if (
            str(row.inchikey14)
            in keys
        ):
            record["survived"] += 1

    print(
        "\n"
        + "=" * 110
    )

    print(
        f"{'Adduct':20s}"
        f"{'Rows':>10s}"
        f"{'Survival':>14s}"
        f"{'Zero':>12s}"
        f"{'Median Cands':>16s}"
        f"{'Mean Cands':>14s}"
    )

    print(
        "-" * 110
    )

    ordered = sorted(
        stats.items(),
        key=lambda item: (
            -item[1]["rows"]
        ),
    )

    for adduct, record in ordered:

        rows = record["rows"]

        survived = record[
            "survived"
        ]

        zero = record[
            "zero"
        ]

        counts = np.asarray(
            record[
                "candidate_counts"
            ],
            dtype=np.int64,
        )

        survival_rate = (
            survived
            / rows
        )

        zero_rate = (
            zero
            / rows
        )

        median_candidates = float(
            np.median(counts)
        )

        mean_candidates = float(
            counts.mean()
        )

        print(
            f"{adduct:20s}"
            f"{rows:10,d}"
            f"{survival_rate:13.4%}"
            f"{zero_rate:11.4%}"
            f"{median_candidates:16.1f}"
            f"{mean_candidates:14.2f}"
        )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "STAGE 7 ADDUCT DIAGNOSTIC COMPLETE"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()