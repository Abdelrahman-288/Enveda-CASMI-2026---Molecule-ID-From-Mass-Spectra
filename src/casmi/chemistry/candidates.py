from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from casmi.chemistry.adducts import (
    is_trusted_for_mass_filter,
    mass_error_ppm,
    precursor_to_neutral_mass,
)
from casmi.chemistry.formula import (
    exact_mass_from_formula,
    exact_mass_from_smiles,
    formula_from_smiles,
)


@dataclass(frozen=True)
class CandidateChemistryResult:
    smiles: str
    candidate_mass: float
    neutral_mass_from_precursor: float
    mass_error_ppm: float
    candidate_formula: str
    formula_match: Optional[bool]
    passes_mass_filter: bool
    adduct_trusted: bool


def evaluate_candidate(
    smiles: str,
    precursor_mz: float,
    adduct: str,
    expected_formula: Optional[str] = None,
    tolerance_ppm: float = 10.0,
) -> CandidateChemistryResult:
    """
    Evaluate whether a candidate molecular structure is chemically
    compatible with an observed precursor.

    Parameters
    ----------
    smiles
        Candidate molecular structure.

    precursor_mz
        Observed precursor m/z.

    adduct
        Observed precursor adduct.

    expected_formula
        Optional known or predicted molecular formula.

    tolerance_ppm
        Maximum absolute neutral-mass error allowed.

    Returns
    -------
    CandidateChemistryResult
    """

    trusted = is_trusted_for_mass_filter(
        adduct
    )

    candidate_mass = exact_mass_from_smiles(
        smiles
    )

    candidate_formula = formula_from_smiles(
        smiles
    )

    neutral_mass = precursor_to_neutral_mass(
        precursor_mz,
        adduct,
        require_trusted=False,
    )

    error_ppm = mass_error_ppm(
        neutral_mass,
        candidate_mass,
    )

    if trusted:
        passes_mass = (
            abs(error_ppm)
            <= float(tolerance_ppm)
        )
    else:
        # Do not perform a hard rejection using an adduct that
        # was found to be unreliable in the training data.
        passes_mass = True

    if expected_formula is None:
        formula_match = None
    else:
        formula_match = (
            candidate_formula
            == expected_formula
        )

    return CandidateChemistryResult(
        smiles=smiles,
        candidate_mass=candidate_mass,
        neutral_mass_from_precursor=neutral_mass,
        mass_error_ppm=error_ppm,
        candidate_formula=candidate_formula,
        formula_match=formula_match,
        passes_mass_filter=passes_mass,
        adduct_trusted=trusted,
    )


def candidate_passes_constraints(
    smiles: str,
    precursor_mz: float,
    adduct: str,
    expected_formula: Optional[str] = None,
    tolerance_ppm: float = 10.0,
    require_formula_match: bool = False,
) -> bool:
    """
    Convenience function for hard candidate filtering.
    """

    result = evaluate_candidate(
        smiles=smiles,
        precursor_mz=precursor_mz,
        adduct=adduct,
        expected_formula=expected_formula,
        tolerance_ppm=tolerance_ppm,
    )

    if not result.passes_mass_filter:
        return False

    if require_formula_match:
        if expected_formula is None:
            raise ValueError(
                "expected_formula is required when "
                "require_formula_match=True."
            )

        if result.formula_match is not True:
            return False

    return True


def mass_tolerance_da(
    neutral_mass: float,
    tolerance_ppm: float,
) -> float:
    """
    Convert a ppm tolerance into an absolute Dalton window.
    """

    return (
        float(neutral_mass)
        * float(tolerance_ppm)
        / 1_000_000.0
    )


def formula_mass(
    formula: str,
) -> float:
    """
    Convenience wrapper for molecular-formula exact mass.
    """

    return exact_mass_from_formula(
        formula
    )