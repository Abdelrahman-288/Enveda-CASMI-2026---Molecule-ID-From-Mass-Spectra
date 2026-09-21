from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet

import yaml


@dataclass(frozen=True)
class ChemistryConfig:
    default_tolerance_ppm: float

    hard_filter_adducts: FrozenSet[str]
    soft_fallback_adducts: FrozenSet[str]
    disabled_hard_filter_adducts: FrozenSet[str]

    use_formula_constraint: bool
    require_exact_formula_match_when_formula_known: bool

    fallback_enabled: bool
    keep_best_mass_candidates_if_empty: int


def load_chemistry_config(
    config_path: str | Path,
) -> ChemistryConfig:
    """
    Load chemistry and candidate-filtering configuration.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Chemistry config not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    mass_filter = data["mass_filter"]
    formula = data["formula"]
    fallback = data["fallback"]

    return ChemistryConfig(
        default_tolerance_ppm=float(
            mass_filter["default_tolerance_ppm"]
        ),

        hard_filter_adducts=frozenset(
            mass_filter["hard_filter_adducts"]
        ),

        soft_fallback_adducts=frozenset(
            mass_filter["soft_fallback_adducts"]
        ),

        disabled_hard_filter_adducts=frozenset(
            mass_filter[
                "disabled_hard_filter_adducts"
            ]
        ),

        use_formula_constraint=bool(
            formula["use_formula_constraint"]
        ),

        require_exact_formula_match_when_formula_known=bool(
            formula[
                "require_exact_formula_match_when_formula_known"
            ]
        ),

        fallback_enabled=bool(
            fallback["enabled"]
        ),

        keep_best_mass_candidates_if_empty=int(
            fallback[
                "keep_best_mass_candidates_if_empty"
            ]
        ),
    )