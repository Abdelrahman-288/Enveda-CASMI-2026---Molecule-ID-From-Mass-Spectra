from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class BinningConfig:
    """
    Configuration for classical binned MS/MS representations.
    """

    bin_width: float = 0.1
    min_mz: float = 0.0
    max_mz: float = 1000.0


SparseSpectrum = Dict[int, float]


def mz_to_bin(
    mz: float,
    config: BinningConfig,
) -> int:
    """
    Convert an m/z value to an integer bin index.
    """

    return int(
        np.floor(
            (float(mz) - config.min_mz)
            / config.bin_width
        )
    )


def spectrum_to_sparse_bins(
    mzs,
    intensities,
    config: BinningConfig | None = None,
) -> SparseSpectrum:
    """
    Convert an MS/MS spectrum into a sparse binned representation.

    Multiple peaks falling into the same bin have their intensities
    summed.

    Peaks outside the configured m/z range are ignored.
    """

    if config is None:
        config = BinningConfig()

    mzs = np.asarray(
        mzs,
        dtype=np.float64,
    )

    intensities = np.asarray(
        intensities,
        dtype=np.float64,
    )

    if len(mzs) != len(intensities):
        raise ValueError(
            "mzs and intensities must have identical lengths."
        )

    bins: SparseSpectrum = {}

    for mz, intensity in zip(
        mzs,
        intensities,
    ):
        if not np.isfinite(mz):
            continue

        if not np.isfinite(intensity):
            continue

        if intensity <= 0:
            continue

        if mz < config.min_mz:
            continue

        if mz >= config.max_mz:
            continue

        bin_index = mz_to_bin(
            mz,
            config,
        )

        bins[bin_index] = (
            bins.get(bin_index, 0.0)
            + float(intensity)
        )

    return bins


def sparse_l2_norm(
    spectrum: SparseSpectrum,
) -> float:
    """
    L2 norm of a sparse spectrum.
    """

    if not spectrum:
        return 0.0

    return float(
        np.sqrt(
            sum(
                intensity * intensity
                for intensity
                in spectrum.values()
            )
        )
    )


def normalize_sparse_spectrum(
    spectrum: SparseSpectrum,
) -> SparseSpectrum:
    """
    L2-normalize a sparse spectral vector.
    """

    norm = sparse_l2_norm(
        spectrum
    )

    if norm == 0:
        return {}

    return {
        bin_index: intensity / norm
        for bin_index, intensity
        in spectrum.items()
    }


def sparse_dot(
    first: SparseSpectrum,
    second: SparseSpectrum,
) -> float:
    """
    Dot product between two sparse spectral vectors.
    """

    if len(first) > len(second):
        first, second = (
            second,
            first,
        )

    return float(
        sum(
            intensity
            * second.get(
                bin_index,
                0.0,
            )
            for bin_index, intensity
            in first.items()
        )
    )


def cosine_similarity(
    first: SparseSpectrum,
    second: SparseSpectrum,
) -> float:
    """
    Standard cosine similarity between two sparse spectra.
    """

    norm_first = sparse_l2_norm(
        first
    )

    norm_second = sparse_l2_norm(
        second
    )

    if (
        norm_first == 0
        or norm_second == 0
    ):
        return 0.0

    return (
        sparse_dot(
            first,
            second,
        )
        / (
            norm_first
            * norm_second
        )
    )


def spectral_cosine_similarity(
    mzs_a,
    intensities_a,
    mzs_b,
    intensities_b,
    config: BinningConfig | None = None,
) -> float:
    """
    Convenience function:

    raw peak arrays
        -> binned sparse vectors
        -> cosine similarity
    """

    first = spectrum_to_sparse_bins(
        mzs_a,
        intensities_a,
        config=config,
    )

    second = spectrum_to_sparse_bins(
        mzs_b,
        intensities_b,
        config=config,
    )

    return cosine_similarity(
        first,
        second,
    )