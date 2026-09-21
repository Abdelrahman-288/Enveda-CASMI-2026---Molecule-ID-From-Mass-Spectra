import math

import pytest

from casmi.chemistry.adducts import (
    precursor_to_neutral_mass,
    neutral_mass_to_precursor,
    is_trusted_for_mass_filter,
    mass_error_ppm,
    within_mass_tolerance,
)

from casmi.chemistry.adducts import (
    precursor_to_neutral_mass,
    neutral_mass_to_precursor,
)


ETHANOL_EXACT_MASS = 46.041864812


def test_protonated_round_trip():
    mz = neutral_mass_to_precursor(
        ETHANOL_EXACT_MASS,
        "[M+H]+",
    )

    recovered = precursor_to_neutral_mass(
        mz,
        "[M+H]+",
    )

    assert math.isclose(
        recovered,
        ETHANOL_EXACT_MASS,
        rel_tol=0,
        abs_tol=1e-9,
    )


def test_deprotonated_round_trip():
    mz = neutral_mass_to_precursor(
        ETHANOL_EXACT_MASS,
        "[M-H]-",
    )

    recovered = precursor_to_neutral_mass(
        mz,
        "[M-H]-",
    )

    assert math.isclose(
        recovered,
        ETHANOL_EXACT_MASS,
        rel_tol=0,
        abs_tol=1e-9,
    )


def test_sodium_round_trip():
    mz = neutral_mass_to_precursor(
        ETHANOL_EXACT_MASS,
        "[M+Na]+",
    )

    recovered = precursor_to_neutral_mass(
        mz,
        "[M+Na]+",
    )

    assert math.isclose(
        recovered,
        ETHANOL_EXACT_MASS,
        rel_tol=0,
        abs_tol=1e-9,
    )


def test_dimer_round_trip():
    mz = neutral_mass_to_precursor(
        ETHANOL_EXACT_MASS,
        "[2M+H]+",
    )

    recovered = precursor_to_neutral_mass(
        mz,
        "[2M+H]+",
    )

    assert math.isclose(
        recovered,
        ETHANOL_EXACT_MASS,
        rel_tol=0,
        abs_tol=1e-9,
    )


def test_double_charge_round_trip():
    mz = neutral_mass_to_precursor(
        ETHANOL_EXACT_MASS,
        "[M+2H]2+",
    )

    recovered = precursor_to_neutral_mass(
        mz,
        "[M+2H]2+",
    )

    assert math.isclose(
        recovered,
        ETHANOL_EXACT_MASS,
        rel_tol=0,
        abs_tol=1e-9,
    )


def test_untrusted_double_charge_annotation():
    assert not is_trusted_for_mass_filter(
        "[M+2H]2+"
    )


def test_untrusted_adduct_can_be_rejected():
    with pytest.raises(ValueError):
        precursor_to_neutral_mass(
            500.0,
            "[M+2H]2+",
            require_trusted=True,
        )


def test_test_adducts_are_trusted():
    test_adducts = [
        "[M+H]+",
        "[M-H]-",
        "[M+Na]+",
        "[M+K]+",
        "[M+NH4]+",
        "[M+Cl]-",
        "[M+CH2O2-H]-",
    ]

    for adduct in test_adducts:
        assert is_trusted_for_mass_filter(
            adduct
        )


def test_mass_error_ppm():
    error = mass_error_ppm(
        100.001,
        100.0,
    )

    assert math.isclose(
        error,
        10.0,
        abs_tol=1e-9,
    )


def test_within_mass_tolerance():
    assert within_mass_tolerance(
        100.0005,
        100.0,
        tolerance_ppm=10.0,
    )

    assert not within_mass_tolerance(
        100.002,
        100.0,
        tolerance_ppm=10.0,
    )