from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateMassHit:
    """
    One unique molecular candidate returned by an exact-mass
    search.

    Multiple mass variants belonging to the same inchikey14
    are collapsed to the closest matching variant.
    """

    inchikey14: str

    normalized_smiles: str

    molecular_formula: str

    exact_mass: float

    mass_error_da: float

    mass_error_ppm: float

    variant_count: int


class CandidateMassIndex:
    """
    Sorted exact-mass index over molecular candidate variants.

    The index may contain more than one mass variant for the
    same inchikey14.

    Queries are performed against every variant, then results
    are deduplicated back to one candidate per inchikey14.

    The closest mass variant is retained.
    """

    REQUIRED_COLUMNS = {
        "inchikey14",
        "normalized_smiles",
        "molecular_formula",
        "exact_mass",
    }

    def __init__(
        self,
        variants: pd.DataFrame,
    ):
        missing = (
            self.REQUIRED_COLUMNS
            - set(
                variants.columns
            )
        )

        if missing:
            raise ValueError(
                "Missing required variant columns: "
                f"{sorted(missing)}"
            )

        if len(
            variants
        ) == 0:
            raise ValueError(
                "Candidate mass index cannot be empty."
            )

        table = variants[
            [
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
                "exact_mass",
            ]
        ].copy()

        table = table.dropna(
            subset=[
                "inchikey14",
                "normalized_smiles",
                "exact_mass",
            ]
        )

        table[
            "exact_mass"
        ] = pd.to_numeric(
            table[
                "exact_mass"
            ],
            errors="coerce",
        )

        table = table.dropna(
            subset=[
                "exact_mass",
            ]
        )

        table = table[
            table[
                "exact_mass"
            ] > 0
        ]

        table = table.drop_duplicates(
            subset=[
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
                "exact_mass",
            ]
        )

        table = table.sort_values(
            [
                "exact_mass",
                "inchikey14",
            ]
        ).reset_index(
            drop=True
        )

        if len(
            table
        ) == 0:
            raise ValueError(
                "No valid candidate mass variants remain "
                "after validation."
            )

        self.table = table

        self.masses = (
            table[
                "exact_mass"
            ]
            .to_numpy(
                dtype=np.float64,
            )
        )

        self.inchikey14 = (
            table[
                "inchikey14"
            ]
            .astype(str)
            .to_numpy()
        )

        self.smiles = (
            table[
                "normalized_smiles"
            ]
            .astype(str)
            .to_numpy()
        )

        self.formulas = (
            table[
                "molecular_formula"
            ]
            .fillna("")
            .astype(str)
            .to_numpy()
        )

        structure_counts = (
            table[
                "inchikey14"
            ]
            .value_counts()
        )

        self.variant_counts = (
            structure_counts
            .to_dict()
        )

        self.unique_structure_count = int(
            table[
                "inchikey14"
            ].nunique()
        )

        self.variant_count = int(
            len(
                table
            )
        )

    @staticmethod
    def ppm_to_da(
        mass: float,
        tolerance_ppm: float,
    ) -> float:
        """
        Convert ppm tolerance to absolute mass tolerance.
        """

        if mass <= 0:
            raise ValueError(
                "mass must be > 0."
            )

        if tolerance_ppm < 0:
            raise ValueError(
                "tolerance_ppm must be >= 0."
            )

        return (
            mass
            * tolerance_ppm
            * 1e-6
        )

    def query(
        self,
        neutral_mass: float,
        tolerance_ppm: float = 10.0,
        max_candidates: int | None = None,
    ) -> list[CandidateMassHit]:
        """
        Retrieve unique structures compatible with the given
        neutral exact mass.

        Candidate variants are first retrieved using binary
        search over sorted exact masses.

        Multiple variants of one inchikey14 are then
        deduplicated by retaining the closest exact-mass
        match.

        Results are sorted by absolute ppm error.
        """

        neutral_mass = float(
            neutral_mass
        )

        tolerance_ppm = float(
            tolerance_ppm
        )

        if not np.isfinite(
            neutral_mass
        ):
            raise ValueError(
                "neutral_mass must be finite."
            )

        if neutral_mass <= 0:
            raise ValueError(
                "neutral_mass must be > 0."
            )

        if not np.isfinite(
            tolerance_ppm
        ):
            raise ValueError(
                "tolerance_ppm must be finite."
            )

        if tolerance_ppm < 0:
            raise ValueError(
                "tolerance_ppm must be >= 0."
            )

        if (
            max_candidates is not None
            and max_candidates <= 0
        ):
            raise ValueError(
                "max_candidates must be > 0 "
                "when provided."
            )

        tolerance_da = self.ppm_to_da(
            neutral_mass,
            tolerance_ppm,
        )

        lower_mass = (
            neutral_mass
            - tolerance_da
        )

        upper_mass = (
            neutral_mass
            + tolerance_da
        )

        left = int(
            np.searchsorted(
                self.masses,
                lower_mass,
                side="left",
            )
        )

        right = int(
            np.searchsorted(
                self.masses,
                upper_mass,
                side="right",
            )
        )

        if left >= right:
            return []

        best_by_structure = {}

        for index in range(
            left,
            right,
        ):
            candidate_mass = float(
                self.masses[
                    index
                ]
            )

            mass_error_da = (
                candidate_mass
                - neutral_mass
            )

            mass_error_ppm = (
                mass_error_da
                / neutral_mass
                * 1e6
            )

            candidate = CandidateMassHit(
                inchikey14=(
                    str(
                        self.inchikey14[
                            index
                        ]
                    )
                ),
                normalized_smiles=(
                    str(
                        self.smiles[
                            index
                        ]
                    )
                ),
                molecular_formula=(
                    str(
                        self.formulas[
                            index
                        ]
                    )
                ),
                exact_mass=(
                    candidate_mass
                ),
                mass_error_da=(
                    float(
                        mass_error_da
                    )
                ),
                mass_error_ppm=(
                    float(
                        mass_error_ppm
                    )
                ),
                variant_count=int(
                    self.variant_counts[
                        str(
                            self.inchikey14[
                                index
                            ]
                        )
                    ]
                ),
            )

            key = (
                candidate.inchikey14
            )

            existing = (
                best_by_structure.get(
                    key
                )
            )

            if existing is None:
                best_by_structure[
                    key
                ] = candidate

            elif (
                abs(
                    candidate.mass_error_ppm
                )
                <
                abs(
                    existing.mass_error_ppm
                )
            ):
                best_by_structure[
                    key
                ] = candidate

        hits = list(
            best_by_structure.values()
        )

        hits.sort(
            key=lambda hit: (
                abs(
                    hit.mass_error_ppm
                ),
                hit.inchikey14,
            )
        )

        if max_candidates is not None:
            hits = hits[
                :max_candidates
            ]

        return hits

    def query_dataframe(
        self,
        neutral_mass: float,
        tolerance_ppm: float = 10.0,
        max_candidates: int | None = None,
    ) -> pd.DataFrame:
        """
        Convenience wrapper returning query results as a
        DataFrame.
        """

        hits = self.query(
            neutral_mass=neutral_mass,
            tolerance_ppm=tolerance_ppm,
            max_candidates=max_candidates,
        )

        columns = [
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
            "exact_mass",
            "mass_error_da",
            "mass_error_ppm",
            "variant_count",
        ]

        if not hits:
            return pd.DataFrame(
                columns=columns
            )

        return pd.DataFrame(
            [
                {
                    "inchikey14": hit.inchikey14,
                    "normalized_smiles": (
                        hit.normalized_smiles
                    ),
                    "molecular_formula": (
                        hit.molecular_formula
                    ),
                    "exact_mass": (
                        hit.exact_mass
                    ),
                    "mass_error_da": (
                        hit.mass_error_da
                    ),
                    "mass_error_ppm": (
                        hit.mass_error_ppm
                    ),
                    "variant_count": (
                        hit.variant_count
                    ),
                }
                for hit in hits
            ],
            columns=columns,
        )

    def count_candidates(
        self,
        neutral_mass: float,
        tolerance_ppm: float = 10.0,
    ) -> int:
        """
        Count unique inchikey14 candidates within the mass
        window.
        """

        return len(
            self.query(
                neutral_mass=neutral_mass,
                tolerance_ppm=tolerance_ppm,
            )
        )