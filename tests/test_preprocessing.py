import numpy as np

from casmi.spectra.preprocessing import (
    SpectrumPreprocessingConfig,
    preprocess_spectrum,
)


def test_basic_preprocessing():
    mzs = np.array(
        [100.0, 200.0, 300.0]
    )

    intensities = np.array(
        [0.2, 1.0, 0.5]
    )

    processed_mz, processed_intensity = (
        preprocess_spectrum(
            mzs,
            intensities,
        )
    )

    assert len(processed_mz) == 3
    assert len(processed_intensity) == 3

    assert np.isclose(
        processed_intensity.max(),
        1.0,
    )


def test_relative_intensity_filter():
    mzs = np.array(
        [100.0, 200.0, 300.0]
    )

    intensities = np.array(
        [0.01, 1.0, 0.2]
    )

    config = SpectrumPreprocessingConfig(
        min_relative_intensity=0.05
    )

    processed_mz, _ = preprocess_spectrum(
        mzs,
        intensities,
        config=config,
    )

    assert 100.0 not in processed_mz
    assert 200.0 in processed_mz
    assert 300.0 in processed_mz


def test_precursor_filter():
    mzs = np.array(
        [100.0, 200.0, 303.0]
    )

    intensities = np.array(
        [0.2, 1.0, 0.5]
    )

    config = SpectrumPreprocessingConfig(
        precursor_tolerance_da=2.0
    )

    processed_mz, _ = preprocess_spectrum(
        mzs,
        intensities,
        precursor_mz=300.0,
        config=config,
    )

    assert 303.0 not in processed_mz


def test_top_k():
    mzs = np.array(
        [
            100.0,
            200.0,
            300.0,
            400.0,
        ]
    )

    intensities = np.array(
        [
            0.1,
            0.9,
            0.5,
            1.0,
        ]
    )

    config = SpectrumPreprocessingConfig(
        max_peaks=2
    )

    processed_mz, _ = preprocess_spectrum(
        mzs,
        intensities,
        config=config,
    )

    assert len(processed_mz) == 2

    assert 200.0 in processed_mz
    assert 400.0 in processed_mz


def test_duplicate_mz_merge():
    mzs = np.array(
        [
            100.0,
            100.0,
            200.0,
        ]
    )

    intensities = np.array(
        [
            0.2,
            0.3,
            1.0,
        ]
    )

    processed_mz, processed_intensity = (
        preprocess_spectrum(
            mzs,
            intensities,
        )
    )

    assert len(processed_mz) == 2

    idx = np.where(
        processed_mz == 100.0
    )[0][0]

    assert np.isclose(
        processed_intensity[idx],
        0.5,
    )


def test_sqrt_transform():
    mzs = np.array(
        [100.0, 200.0]
    )

    intensities = np.array(
        [0.25, 1.0]
    )

    config = SpectrumPreprocessingConfig(
        intensity_transform="sqrt"
    )

    _, processed_intensity = preprocess_spectrum(
        mzs,
        intensities,
        config=config,
    )

    assert np.allclose(
        processed_intensity,
        [0.5, 1.0],
    )