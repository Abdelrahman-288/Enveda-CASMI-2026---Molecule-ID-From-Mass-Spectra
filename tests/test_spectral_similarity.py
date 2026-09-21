import math

from casmi.retrieval.spectral_similarity import (
    BinningConfig,
    cosine_similarity,
    mz_to_bin,
    normalize_sparse_spectrum,
    spectral_cosine_similarity,
    spectrum_to_sparse_bins,
)


def test_mz_to_bin():
    config = BinningConfig(
        bin_width=0.1,
        min_mz=0.0,
        max_mz=1000.0,
    )

    assert mz_to_bin(
        100.05,
        config,
    ) == 1000


def test_identical_spectra_similarity_one():
    mzs = [
        100.0,
        200.0,
        300.0,
    ]

    intensities = [
        0.2,
        1.0,
        0.5,
    ]

    similarity = spectral_cosine_similarity(
        mzs,
        intensities,
        mzs,
        intensities,
    )

    assert math.isclose(
        similarity,
        1.0,
        abs_tol=1e-12,
    )


def test_non_overlapping_spectra_zero():
    similarity = spectral_cosine_similarity(
        [100.0, 200.0],
        [1.0, 0.5],
        [400.0, 500.0],
        [1.0, 0.5],
    )

    assert math.isclose(
        similarity,
        0.0,
        abs_tol=1e-12,
    )


def test_partial_overlap():
    similarity = spectral_cosine_similarity(
        [100.0, 200.0],
        [1.0, 1.0],
        [100.0, 300.0],
        [1.0, 1.0],
    )

    assert math.isclose(
        similarity,
        0.5,
        abs_tol=1e-12,
    )


def test_peaks_in_same_bin_are_merged():
    config = BinningConfig(
        bin_width=0.1,
    )

    spectrum = spectrum_to_sparse_bins(
        [100.01, 100.04],
        [0.3, 0.7],
        config,
    )

    assert len(spectrum) == 1

    value = next(
        iter(
            spectrum.values()
        )
    )

    assert math.isclose(
        value,
        1.0,
        abs_tol=1e-12,
    )


def test_out_of_range_peaks_removed():
    config = BinningConfig(
        bin_width=0.1,
        min_mz=50.0,
        max_mz=500.0,
    )

    spectrum = spectrum_to_sparse_bins(
        [20.0, 100.0, 600.0],
        [1.0, 1.0, 1.0],
        config,
    )

    assert len(spectrum) == 1


def test_sparse_normalization():
    spectrum = {
        1: 3.0,
        2: 4.0,
    }

    normalized = normalize_sparse_spectrum(
        spectrum
    )

    assert math.isclose(
        normalized[1],
        0.6,
        abs_tol=1e-12,
    )

    assert math.isclose(
        normalized[2],
        0.8,
        abs_tol=1e-12,
    )


def test_cosine_symmetry():
    first = {
        1: 1.0,
        2: 2.0,
    }

    second = {
        2: 3.0,
        3: 4.0,
    }

    assert math.isclose(
        cosine_similarity(
            first,
            second,
        ),
        cosine_similarity(
            second,
            first,
        ),
        abs_tol=1e-12,
    )