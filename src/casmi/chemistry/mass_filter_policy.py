from __future__ import annotations

from dataclasses import dataclass


DEFAULT_HARD_TOLERANCE_PPM = 10.0

FALLBACK_TOLERANCE_PPM = 20.0


HIGH_CONFIDENCE_SOURCES = {
    "enveda-180",
    "pluskal_ms2",
    "spectraverse",
    "msdial",
    "enveda-np-examples",
}


NOISY_SOURCES = {
    "riken",
    "gnps",
    "massbank",
    "mona",
}


@dataclass(frozen=True)
class MassFilterDecision:
    hard_filter: bool

    tolerance_ppm: float | None

    confidence: str

    reason: str


def get_mass_filter_decision(
    adduct_supported: bool,
    adduct_trusted: bool,
    instrument_type: str | None = None,
    ingest_lib: str | None = None,
    inference_mode: bool = False,
) -> MassFilterDecision:
    """
    Decide whether precursor-derived neutral mass should be
    used as a hard molecular candidate filter.

    inference_mode=True is intended for competition/test-time
    use, where the observed domain is timsTOF and closely
    matches the high-confidence Enveda source.
    """

    if not adduct_supported:

        return MassFilterDecision(
            hard_filter=False,
            tolerance_ppm=None,
            confidence="unsupported",
            reason=(
                "Adduct is not supported by the chemistry "
                "conversion model."
            ),
        )

    if not adduct_trusted:

        return MassFilterDecision(
            hard_filter=False,
            tolerance_ppm=None,
            confidence="untrusted_adduct",
            reason=(
                "Adduct is recognized but marked unsafe "
                "for hard mass filtering."
            ),
        )

    instrument = (
        ""
        if instrument_type is None
        else str(instrument_type).strip()
    )

    source = (
        ""
        if ingest_lib is None
        else str(ingest_lib).strip()
    )

    if inference_mode:

        if instrument.lower() == "timstof":

            return MassFilterDecision(
                hard_filter=True,
                tolerance_ppm=(
                    DEFAULT_HARD_TOLERANCE_PPM
                ),
                confidence="high",
                reason=(
                    "Competition-domain timsTOF spectra "
                    "show near-perfect 10 ppm survival."
                ),
            )

        return MassFilterDecision(
            hard_filter=True,
            tolerance_ppm=(
                FALLBACK_TOLERANCE_PPM
            ),
            confidence="medium",
            reason=(
                "Inference sample is outside the confirmed "
                "timsTOF domain; use wider fallback mass "
                "window."
            ),
        )

    if source in HIGH_CONFIDENCE_SOURCES:

        return MassFilterDecision(
            hard_filter=True,
            tolerance_ppm=(
                DEFAULT_HARD_TOLERANCE_PPM
            ),
            confidence="high",
            reason=(
                "Training source has high observed mass "
                "consistency."
            ),
        )

    if instrument.lower() == "timstof":

        return MassFilterDecision(
            hard_filter=True,
            tolerance_ppm=(
                DEFAULT_HARD_TOLERANCE_PPM
            ),
            confidence="high",
            reason=(
                "timsTOF rows show near-perfect 10 ppm "
                "mass consistency."
            ),
        )

    if source in NOISY_SOURCES:

        return MassFilterDecision(
            hard_filter=False,
            tolerance_ppm=None,
            confidence="low",
            reason=(
                "Training source contains substantial "
                "precursor/adduct mass annotation noise."
            ),
        )

    return MassFilterDecision(
        hard_filter=False,
        tolerance_ppm=None,
        confidence="unknown",
        reason=(
            "Mass reliability has not been established "
            "for this source/domain."
        ),
    )