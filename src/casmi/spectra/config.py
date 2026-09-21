from pathlib import Path

import yaml

from casmi.spectra.preprocessing import (
    SpectrumPreprocessingConfig,
)


def load_preprocessing_config(
    config_path: str | Path,
    profile: str = "baseline",
) -> SpectrumPreprocessingConfig:
    """
    Load a named spectrum-preprocessing profile from YAML.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if profile not in data:
        available = ", ".join(data.keys())

        raise KeyError(
            f"Unknown preprocessing profile '{profile}'. "
            f"Available profiles: {available}"
        )

    profile_data = data[profile]

    return SpectrumPreprocessingConfig(
        min_relative_intensity=profile_data.get(
            "min_relative_intensity",
            0.0,
        ),
        max_peaks=profile_data.get(
            "max_peaks",
        ),
        precursor_tolerance_da=profile_data.get(
            "precursor_tolerance_da",
        ),
        intensity_transform=profile_data.get(
            "intensity_transform",
            "none",
        ),
        normalize_intensity=profile_data.get(
            "normalize_intensity",
            True,
        ),
        remove_duplicate_mz=profile_data.get(
            "remove_duplicate_mz",
            True,
        ),
    )