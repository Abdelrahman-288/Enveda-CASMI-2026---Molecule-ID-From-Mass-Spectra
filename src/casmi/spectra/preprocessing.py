from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SpectrumPreprocessingConfig:
    """
    Configuration for MS/MS spectrum preprocessing.
    """

    min_relative_intensity: float = 0.0

    max_peaks: Optional[int] = None

    precursor_tolerance_da: Optional[float] = None

    intensity_transform: str = "none"

    normalize_intensity: bool = True

    remove_duplicate_mz: bool = True


def _validate_arrays(
    mzs: np.ndarray,
    intensities: np.ndarray,
) -> None:
    if mzs.ndim != 1:
        raise ValueError("mzs must be one-dimensional.")

    if intensities.ndim != 1:
        raise ValueError("intensities must be one-dimensional.")

    if len(mzs) != len(intensities):
        raise ValueError(
            "mzs and intensities must have the same length."
        )

    if not np.all(np.isfinite(mzs)):
        raise ValueError("mzs contains non-finite values.")

    if not np.all(np.isfinite(intensities)):
        raise ValueError(
            "intensities contains non-finite values."
        )


def _merge_duplicate_mz(
    mzs: np.ndarray,
    intensities: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge exactly duplicated m/z values.

    Duplicate intensities are summed.
    """

    if len(mzs) == 0:
        return mzs, intensities

    unique_mz, inverse = np.unique(
        mzs,
        return_inverse=True,
    )

    if len(unique_mz) == len(mzs):
        return mzs, intensities

    merged_intensity = np.zeros(
        len(unique_mz),
        dtype=float,
    )

    np.add.at(
        merged_intensity,
        inverse,
        intensities,
    )

    return unique_mz, merged_intensity


def _apply_intensity_transform(
    intensities: np.ndarray,
    transform: str,
) -> np.ndarray:
    if transform == "none":
        return intensities

    if transform == "sqrt":
        return np.sqrt(
            np.clip(
                intensities,
                a_min=0.0,
                a_max=None,
            )
        )

    if transform == "log1p":
        return np.log1p(
            np.clip(
                intensities,
                a_min=0.0,
                a_max=None,
            )
        )

    raise ValueError(
        f"Unsupported intensity transform: {transform}"
    )


def _normalize(
    intensities: np.ndarray,
) -> np.ndarray:
    if len(intensities) == 0:
        return intensities

    max_intensity = np.max(intensities)

    if max_intensity <= 0:
        return intensities

    return intensities / max_intensity


def preprocess_spectrum(
    mzs,
    intensities,
    precursor_mz: Optional[float] = None,
    config: Optional[SpectrumPreprocessingConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess one MS/MS spectrum.

    Parameters
    ----------
    mzs
        Fragment m/z array.

    intensities
        Corresponding normalized intensity array.

    precursor_mz
        Precursor m/z. Required only when precursor filtering
        is enabled.

    config
        Spectrum preprocessing configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Processed m/z and intensity arrays.
    """

    if config is None:
        config = SpectrumPreprocessingConfig()

    mzs = np.asarray(
        mzs,
        dtype=np.float64,
    )

    intensities = np.asarray(
        intensities,
        dtype=np.float64,
    )

    _validate_arrays(
        mzs,
        intensities,
    )

    if len(mzs) == 0:
        return mzs, intensities

    # Remove physically invalid values if encountered.
    valid_mask = (
        (mzs >= 0)
        & (intensities >= 0)
    )

    mzs = mzs[valid_mask]
    intensities = intensities[valid_mask]

    # Ensure sorted m/z values.
    if len(mzs) > 1:
        order = np.argsort(mzs)

        mzs = mzs[order]
        intensities = intensities[order]

    # Merge exact duplicate m/z values.
    if config.remove_duplicate_mz:
        mzs, intensities = _merge_duplicate_mz(
            mzs,
            intensities,
        )

    # Optional precursor cutoff.
    if config.precursor_tolerance_da is not None:
        if precursor_mz is None:
            raise ValueError(
                "precursor_mz is required when "
                "precursor filtering is enabled."
            )

        upper_limit = (
            float(precursor_mz)
            + config.precursor_tolerance_da
        )

        mask = mzs <= upper_limit

        mzs = mzs[mask]
        intensities = intensities[mask]

    # Relative intensity filtering.
    if (
        config.min_relative_intensity > 0
        and len(intensities) > 0
    ):
        max_intensity = np.max(intensities)

        if max_intensity > 0:
            threshold = (
                max_intensity
                * config.min_relative_intensity
            )

            mask = intensities >= threshold

            mzs = mzs[mask]
            intensities = intensities[mask]

    # Keep only the strongest peaks.
    if (
        config.max_peaks is not None
        and len(mzs) > config.max_peaks
    ):
        strongest = np.argpartition(
            intensities,
            -config.max_peaks,
        )[-config.max_peaks:]

        mzs = mzs[strongest]
        intensities = intensities[strongest]

        # Restore m/z ordering.
        order = np.argsort(mzs)

        mzs = mzs[order]
        intensities = intensities[order]

    intensities = _apply_intensity_transform(
        intensities,
        config.intensity_transform,
    )

    if config.normalize_intensity:
        intensities = _normalize(
            intensities,
        )

    return mzs, intensities