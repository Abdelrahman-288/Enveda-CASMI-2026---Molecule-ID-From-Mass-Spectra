import pandas as pd
import pytest

from casmi.retrieval.candidate_mass_index import (
    CandidateMassIndex,
)


@pytest.fixture
def sample_variants():

    return pd.DataFrame(
        {
            "inchikey14": [
                "AAAAAAAAAAAAAA",
                "BBBBBBBBBBBBBB",
                "BBBBBBBBBBBBBB",
                "CCCCCCCCCCCCCC",
            ],
            "normalized_smiles": [
                "CCO",
                "CCN",
                "C[NH3+]",
                "CCC",
            ],
            "molecular_formula": [
                "C2H6O",
                "C2H7N",
                "C2H8N",
                "C3H8",
            ],
            "exact_mass": [
                100.0000,
                100.0005,
                101.007776,
                150.0000,
            ],
        }
    )


def test_index_counts(
    sample_variants,
):

    index = CandidateMassIndex(
        sample_variants
    )

    assert (
        index.unique_structure_count
        == 3
    )

    assert (
        index.variant_count
        == 4
    )


def test_exact_match(
    sample_variants,
):

    index = CandidateMassIndex(
        sample_variants
    )

    hits = index.query(
        neutral_mass=100.0000,
        tolerance_ppm=10.0,
    )

    assert len(
        hits
    ) == 2

    assert (
        hits[
            0
        ].inchikey14
        == "AAAAAAAAAAAAAA"
    )

    assert abs(
        hits[
            0
        ].mass_error_ppm
    ) < 1e-10


def test_variant_deduplication(
    sample_variants,
):

    index = CandidateMassIndex(
        sample_variants
    )

    hits = index.query(
        neutral_mass=101.007776,
        tolerance_ppm=10.0,
    )

    matching = [
        hit
        for hit in hits
        if hit.inchikey14
        == "BBBBBBBBBBBBBB"
    ]

    assert len(
        matching
    ) == 1

    assert (
        matching[
            0
        ].variant_count
        == 2
    )

    assert abs(
        matching[
            0
        ].mass_error_ppm
    ) < 1e-10


def test_returns_closest_variant(
    sample_variants,
):

    index = CandidateMassIndex(
        sample_variants
    )

    hits = index.query(
        neutral_mass=100.0004,
        tolerance_ppm=20.0,
    )

    b_hits = [
        hit
        for hit in hits
        if hit.inchikey14
        == "BBBBBBBBBBBBBB"
    ]

    assert len(
        b_hits
    ) == 1

    assert (
        b_hits[
            0
        ].exact_mass
        == pytest.approx(
            100.0005
        )
    )


def test_results_sorted_by_absolute_ppm(
    sample_variants,
):

    index = CandidateMassIndex(
        sample_variants
    )

    hits = index.query(
        neutral_mass=100.0002,
        tolerance_ppm=10.0,
    )

    errors = [
        abs(
            hit.mass_error_ppm
        )
        for hit in hits
    ]

    assert errors == sorted(
        errors
    )


def test_empty_query(
    sample_variants,
):

    index = CandidateMassIndex(
        sample_variants
    )

    hits = index.query(
        neutral_mass=200.0,
        tolerance_ppm=10.0,
    )

    assert hits == []


def test_invalid_mass_rejected(
    sample_variants,
):

    index = CandidateMassIndex(
        sample_variants
    )

    with pytest.raises(
        ValueError
    ):
        index.query(
            neutral_mass=-1.0
        )


def test_invalid_tolerance_rejected(
    sample_variants,
):

    index = CandidateMassIndex(
        sample_variants
    )

    with pytest.raises(
        ValueError
    ):
        index.query(
            neutral_mass=100.0,
            tolerance_ppm=-1.0,
        )


def test_max_candidates(
    sample_variants,
):

    index = CandidateMassIndex(
        sample_variants
    )

    hits = index.query(
        neutral_mass=100.0002,
        tolerance_ppm=20.0,
        max_candidates=1,
    )

    assert len(
        hits
    ) == 1


def test_dataframe_output(
    sample_variants,
):

    index = CandidateMassIndex(
        sample_variants
    )

    result = index.query_dataframe(
        neutral_mass=100.0,
        tolerance_ppm=10.0,
    )

    assert "inchikey14" in (
        result.columns
    )

    assert "mass_error_ppm" in (
        result.columns
    )

    assert "variant_count" in (
        result.columns
    )


def test_ppm_conversion():

    tolerance = (
        CandidateMassIndex.ppm_to_da(
            mass=500.0,
            tolerance_ppm=10.0,
        )
    )

    assert tolerance == pytest.approx(
        0.005
    )