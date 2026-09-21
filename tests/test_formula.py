import math

import pytest

from casmi.chemistry.formula import (
    canonicalize_smiles,
    exact_mass_from_formula,
    exact_mass_from_smiles,
    formula_from_smiles,
    formula_matches_smiles,
    parse_formula,
)


def test_parse_formula():
    result = parse_formula(
        "C6H12O6"
    )

    assert result == {
        "C": 6,
        "H": 12,
        "O": 6,
    }


def test_parse_formula_with_halogen():
    result = parse_formula(
        "C10H12ClNO"
    )

    assert result["C"] == 10
    assert result["H"] == 12
    assert result["Cl"] == 1
    assert result["N"] == 1
    assert result["O"] == 1


def test_ethanol_formula():
    formula = formula_from_smiles(
        "CCO"
    )

    assert formula == "C2H6O"


def test_ethanol_exact_mass():
    mass = exact_mass_from_smiles(
        "CCO"
    )

    assert math.isclose(
        mass,
        46.041864812,
        abs_tol=1e-9,
    )


def test_formula_exact_mass_matches_rdkit():
    formula_mass = exact_mass_from_formula(
        "C2H6O"
    )

    smiles_mass = exact_mass_from_smiles(
        "CCO"
    )

    assert math.isclose(
        formula_mass,
        smiles_mass,
        abs_tol=1e-6,
    )


def test_formula_matches_smiles():
    assert formula_matches_smiles(
        "CCO",
        "C2H6O",
    )

    assert not formula_matches_smiles(
        "CCO",
        "C3H8O",
    )


def test_canonicalize_smiles():
    canonical = canonicalize_smiles(
        "OCC"
    )

    assert canonical == "CCO"


def test_invalid_smiles():
    with pytest.raises(ValueError):
        exact_mass_from_smiles(
            "not_a_smiles"
        )


def test_unsupported_element():
    with pytest.raises(KeyError):
        exact_mass_from_formula(
            "C2H6Xe"
        )


def test_parse_charged_formula():
    result = parse_formula(
        "C12H19N2O2+"
    )

    assert result == {
        "C": 12,
        "H": 19,
        "N": 2,
        "O": 2,
    }


def test_parse_negative_formula():
    result = parse_formula(
        "C8H10N4O2-"
    )

    assert result == {
        "C": 8,
        "H": 10,
        "N": 4,
        "O": 2,
    }