import math

import pytest

from casmi.chemistry.adducts import (
    neutral_mass_to_precursor,
)
from casmi.chemistry.candidates import (
    candidate_passes_constraints,
    evaluate_candidate,
    mass_tolerance_da,
)


ETHANOL_SMILES = "CCO"
ETHANOL_FORMULA = "C2H6O"
ETHANOL_MASS = 46.041864812


def test_correct_candidate_passes_mass():
    precursor = neutral_mass_to_precursor(
        ETHANOL_MASS,
        "[M+H]+",
    )

    result = evaluate_candidate(
        smiles=ETHANOL_SMILES,
        precursor_mz=precursor,
        adduct="[M+H]+",
        tolerance_ppm=10.0,
    )

    assert result.passes_mass_filter

    assert math.isclose(
        result.mass_error_ppm,
        0.0,
        abs_tol=1e-6,
    )


def test_wrong_candidate_fails_mass():
    precursor = neutral_mass_to_precursor(
        ETHANOL_MASS,
        "[M+H]+",
    )

    result = evaluate_candidate(
        smiles="CCCC",
        precursor_mz=precursor,
        adduct="[M+H]+",
        tolerance_ppm=10.0,
    )

    assert not result.passes_mass_filter


def test_formula_match():
    precursor = neutral_mass_to_precursor(
        ETHANOL_MASS,
        "[M+H]+",
    )

    result = evaluate_candidate(
        smiles=ETHANOL_SMILES,
        precursor_mz=precursor,
        adduct="[M+H]+",
        expected_formula=ETHANOL_FORMULA,
    )

    assert result.formula_match is True


def test_formula_mismatch():
    precursor = neutral_mass_to_precursor(
        ETHANOL_MASS,
        "[M+H]+",
    )

    result = evaluate_candidate(
        smiles=ETHANOL_SMILES,
        precursor_mz=precursor,
        adduct="[M+H]+",
        expected_formula="C3H8O",
    )

    assert result.formula_match is False


def test_hard_formula_filter():
    precursor = neutral_mass_to_precursor(
        ETHANOL_MASS,
        "[M+H]+",
    )

    assert candidate_passes_constraints(
        smiles=ETHANOL_SMILES,
        precursor_mz=precursor,
        adduct="[M+H]+",
        expected_formula=ETHANOL_FORMULA,
        require_formula_match=True,
    )


def test_wrong_formula_rejected():
    precursor = neutral_mass_to_precursor(
        ETHANOL_MASS,
        "[M+H]+",
    )

    assert not candidate_passes_constraints(
        smiles=ETHANOL_SMILES,
        precursor_mz=precursor,
        adduct="[M+H]+",
        expected_formula="C3H8O",
        require_formula_match=True,
    )


def test_formula_required_for_hard_formula_filter():
    precursor = neutral_mass_to_precursor(
        ETHANOL_MASS,
        "[M+H]+",
    )

    with pytest.raises(ValueError):
        candidate_passes_constraints(
            smiles=ETHANOL_SMILES,
            precursor_mz=precursor,
            adduct="[M+H]+",
            require_formula_match=True,
        )


def test_untrusted_adduct_not_hard_rejected():
    result = evaluate_candidate(
        smiles=ETHANOL_SMILES,
        precursor_mz=500.0,
        adduct="[M+2H]2+",
        tolerance_ppm=1.0,
    )

    assert result.adduct_trusted is False
    assert result.passes_mass_filter is True


def test_mass_tolerance_da():
    tolerance = mass_tolerance_da(
        neutral_mass=500.0,
        tolerance_ppm=10.0,
    )

    assert math.isclose(
        tolerance,
        0.005,
        abs_tol=1e-12,
    )