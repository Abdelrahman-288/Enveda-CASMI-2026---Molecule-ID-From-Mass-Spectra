from dataclasses import dataclass
from typing import Dict


# =========================================================
# Monoisotopic ionic mass shifts
# =========================================================

PROTON_MASS = 1.007276466621
SODIUM_MASS = 22.989218
POTASSIUM_MASS = 38.963158
AMMONIUM_MASS = 18.033823
CHLORIDE_MASS = 34.969402
FORMATE_MASS = 44.998201


@dataclass(frozen=True)
class AdductInfo:
    """
    Relationship between an observed precursor ion and the
    corresponding neutral molecule.

    observed_mz =
        (multimer * neutral_mass + mass_shift) / abs(charge)

    Therefore:

    neutral_mass =
        (
            observed_mz * abs(charge)
            - mass_shift
        ) / multimer

    trusted_for_mass_filter indicates whether this adduct should
    currently be used for hard candidate mass filtering.

    Some training annotations are known to be inconsistent even
    though the adduct notation itself is chemically meaningful.
    """

    name: str
    charge: int
    multimer: int
    mass_shift: float
    trusted_for_mass_filter: bool = True


# =========================================================
# Supported adduct definitions
# =========================================================

ADDUCTS: Dict[str, AdductInfo] = {

    "[M+H]+": AdductInfo(
        name="[M+H]+",
        charge=1,
        multimer=1,
        mass_shift=PROTON_MASS,
    ),

    "[M-H]-": AdductInfo(
        name="[M-H]-",
        charge=-1,
        multimer=1,
        mass_shift=-PROTON_MASS,
    ),

    "[M+Na]+": AdductInfo(
        name="[M+Na]+",
        charge=1,
        multimer=1,
        mass_shift=SODIUM_MASS,
    ),

    "[M+K]+": AdductInfo(
        name="[M+K]+",
        charge=1,
        multimer=1,
        mass_shift=POTASSIUM_MASS,
    ),

    "[M+NH4]+": AdductInfo(
        name="[M+NH4]+",
        charge=1,
        multimer=1,
        mass_shift=AMMONIUM_MASS,
    ),

    "[M+Cl]-": AdductInfo(
        name="[M+Cl]-",
        charge=-1,
        multimer=1,
        mass_shift=CHLORIDE_MASS,
    ),

    "[M+CH2O2-H]-": AdductInfo(
        name="[M+CH2O2-H]-",
        charge=-1,
        multimer=1,
        mass_shift=FORMATE_MASS,
    ),

    "[2M+H]+": AdductInfo(
        name="[2M+H]+",
        charge=1,
        multimer=2,
        mass_shift=PROTON_MASS,
    ),

    "[2M+Na]+": AdductInfo(
        name="[2M+Na]+",
        charge=1,
        multimer=2,
        mass_shift=SODIUM_MASS,
    ),

    "[2M-H]-": AdductInfo(
        name="[2M-H]-",
        charge=-1,
        multimer=2,
        mass_shift=-PROTON_MASS,
    ),

    # Training-data diagnostic showed that these rows do not
    # behave as genuine doubly charged ions. The notation is kept
    # so the rows can be recognized, but it must not be used for
    # hard neutral-mass filtering.
    "[M+2H]2+": AdductInfo(
        name="[M+2H]2+",
        charge=2,
        multimer=1,
        mass_shift=2 * PROTON_MASS,
        trusted_for_mass_filter=False,
    ),

    "[M-2H]2-": AdductInfo(
        name="[M-2H]2-",
        charge=-2,
        multimer=1,
        mass_shift=-2 * PROTON_MASS,
    ),
}


# =========================================================
# Lookup helpers
# =========================================================

def get_adduct_info(
    adduct: str,
) -> AdductInfo:
    """
    Return metadata for a supported adduct.
    """

    if adduct not in ADDUCTS:
        raise KeyError(
            f"Unsupported adduct: {adduct}. "
            f"Supported adducts: {sorted(ADDUCTS)}"
        )

    return ADDUCTS[adduct]


def is_supported_adduct(
    adduct: str,
) -> bool:
    """
    Check whether an adduct has a known definition.
    """

    return adduct in ADDUCTS


def is_trusted_for_mass_filter(
    adduct: str,
) -> bool:
    """
    Check whether an adduct is currently trusted for hard
    neutral-mass candidate filtering.
    """

    if adduct not in ADDUCTS:
        return False

    return ADDUCTS[
        adduct
    ].trusted_for_mass_filter


# =========================================================
# Mass conversions
# =========================================================

def precursor_to_neutral_mass(
    precursor_mz: float,
    adduct: str,
    require_trusted: bool = False,
) -> float:
    """
    Convert observed precursor m/z into estimated neutral
    monoisotopic mass.

    If require_trusted=True, adducts known to have unreliable
    annotations are rejected.
    """

    info = get_adduct_info(adduct)

    if (
        require_trusted
        and not info.trusted_for_mass_filter
    ):
        raise ValueError(
            f"Adduct {adduct} is currently not trusted "
            "for hard mass filtering."
        )

    observed_ion_mass = (
        float(precursor_mz)
        * abs(info.charge)
    )

    neutral_mass = (
        observed_ion_mass
        - info.mass_shift
    ) / info.multimer

    return neutral_mass


def neutral_mass_to_precursor(
    neutral_mass: float,
    adduct: str,
    require_trusted: bool = False,
) -> float:
    """
    Convert neutral monoisotopic mass into expected precursor m/z.
    """

    info = get_adduct_info(adduct)

    if (
        require_trusted
        and not info.trusted_for_mass_filter
    ):
        raise ValueError(
            f"Adduct {adduct} is currently not trusted "
            "for hard mass filtering."
        )

    precursor_mz = (
        info.multimer * float(neutral_mass)
        + info.mass_shift
    ) / abs(info.charge)

    return precursor_mz


# =========================================================
# Mass-error utilities
# =========================================================

def mass_error_da(
    observed_mass: float,
    expected_mass: float,
) -> float:
    """
    Signed mass error in Daltons.
    """

    return (
        float(observed_mass)
        - float(expected_mass)
    )


def mass_error_ppm(
    observed_mass: float,
    expected_mass: float,
) -> float:
    """
    Signed mass error in parts per million.
    """

    expected_mass = float(expected_mass)

    if expected_mass == 0:
        raise ValueError(
            "expected_mass cannot be zero."
        )

    return (
        (
            float(observed_mass)
            - expected_mass
        )
        / expected_mass
        * 1_000_000.0
    )


def within_mass_tolerance(
    observed_mass: float,
    expected_mass: float,
    tolerance_ppm: float,
) -> bool:
    """
    Return True when two masses agree within the specified
    absolute ppm tolerance.
    """

    error = mass_error_ppm(
        observed_mass,
        expected_mass,
    )

    return (
        abs(error)
        <= float(tolerance_ppm)
    )