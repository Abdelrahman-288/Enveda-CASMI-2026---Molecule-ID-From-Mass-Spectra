from __future__ import annotations

from dataclasses import dataclass

from casmi.chemistry.adducts import (
    is_supported_adduct,
    is_trusted_for_mass_filter,
    precursor_to_neutral_mass,
)
from casmi.retrieval.candidate_mass_index import (
    CandidateMassHit,
    CandidateMassIndex,
)


@dataclass(frozen=True)
class CandidateGenerationResult:
    """
    Result of converting one observed precursor into
    neutral-mass-compatible molecular candidates.
    """

    precursor_mz: float
    adduct: str

    neutral_mass: float | None

    tolerance_ppm: float

    supported_adduct: bool
    trusted_adduct: bool

    hard_filter_applied: bool

    candidates: tuple[
        CandidateMassHit,
        ...
    ]

    message: str


class AdductAwareCandidateGenerator:
    """
    Connect Stage 4 adduct chemistry with the Stage 7
    variant-aware molecular exact-mass index.

    Pipeline:

        precursor m/z
            +
        adduct annotation
            |
            v
        neutral mass
            |
            v
        exact-mass index
            |
            v
        unique inchikey14 candidates
    """

    def __init__(
        self,
        mass_index: CandidateMassIndex,
        tolerance_ppm: float = 10.0,
    ):
        if tolerance_ppm < 0:
            raise ValueError(
                "tolerance_ppm must be >= 0."
            )

        self.mass_index = (
            mass_index
        )

        self.tolerance_ppm = float(
            tolerance_ppm
        )

    def generate(
        self,
        precursor_mz: float,
        adduct: str,
        require_trusted: bool = True,
        max_candidates: int | None = None,
    ) -> CandidateGenerationResult:
        """
        Generate candidates for one observed precursor.

        When require_trusted=True:

        - unsupported adducts do not receive a hard filter
        - known but untrusted adducts do not receive a hard filter

        This avoids accidentally eliminating the true
        structure because of unreliable adduct annotations.
        """

        precursor_mz = float(
            precursor_mz
        )

        supported = (
            is_supported_adduct(
                adduct
            )
        )

        if not supported:

            return CandidateGenerationResult(
                precursor_mz=precursor_mz,
                adduct=adduct,
                neutral_mass=None,
                tolerance_ppm=(
                    self.tolerance_ppm
                ),
                supported_adduct=False,
                trusted_adduct=False,
                hard_filter_applied=False,
                candidates=tuple(),
                message=(
                    "Unsupported adduct; "
                    "hard mass filtering skipped."
                ),
            )

        trusted = (
            is_trusted_for_mass_filter(
                adduct
            )
        )

        if (
            require_trusted
            and not trusted
        ):

            return CandidateGenerationResult(
                precursor_mz=precursor_mz,
                adduct=adduct,
                neutral_mass=None,
                tolerance_ppm=(
                    self.tolerance_ppm
                ),
                supported_adduct=True,
                trusted_adduct=False,
                hard_filter_applied=False,
                candidates=tuple(),
                message=(
                    "Adduct is recognized but "
                    "not trusted for hard mass filtering."
                ),
            )

        neutral_mass = (
            precursor_to_neutral_mass(
                precursor_mz=precursor_mz,
                adduct=adduct,
                require_trusted=(
                    require_trusted
                ),
            )
        )

        if neutral_mass <= 0:

            return CandidateGenerationResult(
                precursor_mz=precursor_mz,
                adduct=adduct,
                neutral_mass=neutral_mass,
                tolerance_ppm=(
                    self.tolerance_ppm
                ),
                supported_adduct=True,
                trusted_adduct=trusted,
                hard_filter_applied=False,
                candidates=tuple(),
                message=(
                    "Computed neutral mass is "
                    "not physically valid."
                ),
            )

        hits = self.mass_index.query(
            neutral_mass=neutral_mass,
            tolerance_ppm=(
                self.tolerance_ppm
            ),
            max_candidates=(
                max_candidates
            ),
        )

        return CandidateGenerationResult(
            precursor_mz=precursor_mz,
            adduct=adduct,
            neutral_mass=neutral_mass,
            tolerance_ppm=(
                self.tolerance_ppm
            ),
            supported_adduct=True,
            trusted_adduct=trusted,
            hard_filter_applied=True,
            candidates=tuple(
                hits
            ),
            message=(
                f"{len(hits)} unique candidates "
                "retrieved."
            ),
        )

    def count_candidates(
        self,
        precursor_mz: float,
        adduct: str,
        require_trusted: bool = True,
    ) -> int:
        """
        Return the number of unique hard-filter candidates.
        """

        result = self.generate(
            precursor_mz=precursor_mz,
            adduct=adduct,
            require_trusted=(
                require_trusted
            ),
        )

        return len(
            result.candidates
        )