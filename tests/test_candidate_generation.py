import pandas as pd
import pytest

from casmi.chemistry.adducts import (
    neutral_mass_to_precursor,
)

from casmi.retrieval.candidate_generation import (
    AdductAwareCandidateGenerator,
)

from casmi.retrieval.candidate_mass_index import (
    CandidateMassIndex,
)


@pytest.fixture
def generator():

    variants = pd.DataFrame(
        {
            "inchikey14": [
                "AAAAAAAAAAAAAA",
                "BBBBBBBBBBBBBB",
                "CCCCCCCCCCCCCC",
            ],
            "normalized_smiles": [
                "CCO",
                "CCN",
                "CCC",
            ],
            "molecular_formula": [
                "C2H6O",
                "C2H7N",
                "C3H8",
            ],
            "exact_mass": [
                100.000000,
                100.000500,
                150.000000,
            ],
        }
    )

    index = CandidateMassIndex(
        variants
    )

    return AdductAwareCandidateGenerator(
        mass_index=index,
        tolerance_ppm=10.0,
    )


def test_protonated_candidate_generation(
    generator,
):

    precursor = (
        neutral_mass_to_precursor(
            neutral_mass=100.0,
            adduct="[M+H]+",
        )
    )

    result = generator.generate(
        precursor_mz=precursor,
        adduct="[M+H]+",
    )

    keys = {
        hit.inchikey14
        for hit in result.candidates
    }

    assert (
        "AAAAAAAAAAAAAA"
        in keys
    )

    assert (
        result.hard_filter_applied
        is True
    )

    assert (
        result.trusted_adduct
        is True
    )


def test_deprotonated_candidate_generation(
    generator,
):

    precursor = (
        neutral_mass_to_precursor(
            neutral_mass=150.0,
            adduct="[M-H]-",
        )
    )

    result = generator.generate(
        precursor_mz=precursor,
        adduct="[M-H]-",
    )

    keys = {
        hit.inchikey14
        for hit in result.candidates
    }

    assert (
        "CCCCCCCCCCCCCC"
        in keys
    )


def test_multimer_candidate_generation(
    generator,
):

    precursor = (
        neutral_mass_to_precursor(
            neutral_mass=100.0,
            adduct="[2M+H]+",
        )
    )

    result = generator.generate(
        precursor_mz=precursor,
        adduct="[2M+H]+",
    )

    keys = {
        hit.inchikey14
        for hit in result.candidates
    }

    assert (
        "AAAAAAAAAAAAAA"
        in keys
    )


def test_sodium_candidate_generation(
    generator,
):

    precursor = (
        neutral_mass_to_precursor(
            neutral_mass=100.0,
            adduct="[M+Na]+",
        )
    )

    result = generator.generate(
        precursor_mz=precursor,
        adduct="[M+Na]+",
    )

    keys = {
        hit.inchikey14
        for hit in result.candidates
    }

    assert (
        "AAAAAAAAAAAAAA"
        in keys
    )


def test_unsupported_adduct_skips_filter(
    generator,
):

    result = generator.generate(
        precursor_mz=101.0,
        adduct="[M+XYZ]+",
    )

    assert (
        result.supported_adduct
        is False
    )

    assert (
        result.hard_filter_applied
        is False
    )

    assert (
        len(
            result.candidates
        )
        == 0
    )


def test_untrusted_adduct_skips_filter(
    generator,
):

    result = generator.generate(
        precursor_mz=100.0,
        adduct="[M+2H]2+",
        require_trusted=True,
    )

    assert (
        result.supported_adduct
        is True
    )

    assert (
        result.trusted_adduct
        is False
    )

    assert (
        result.hard_filter_applied
        is False
    )


def test_untrusted_can_be_forced(
    generator,
):

    precursor = (
        neutral_mass_to_precursor(
            neutral_mass=100.0,
            adduct="[M+2H]2+",
            require_trusted=False,
        )
    )

    result = generator.generate(
        precursor_mz=precursor,
        adduct="[M+2H]2+",
        require_trusted=False,
    )

    keys = {
        hit.inchikey14
        for hit in result.candidates
    }

    assert (
        "AAAAAAAAAAAAAA"
        in keys
    )


def test_neutral_mass_is_recovered(
    generator,
):

    precursor = (
        neutral_mass_to_precursor(
            neutral_mass=100.0,
            adduct="[M+H]+",
        )
    )

    result = generator.generate(
        precursor_mz=precursor,
        adduct="[M+H]+",
    )

    assert result.neutral_mass == pytest.approx(
        100.0,
        abs=1e-9,
    )


def test_count_candidates(
    generator,
):

    precursor = (
        neutral_mass_to_precursor(
            neutral_mass=100.0,
            adduct="[M+H]+",
        )
    )

    count = generator.count_candidates(
        precursor_mz=precursor,
        adduct="[M+H]+",
    )

    assert count == 2