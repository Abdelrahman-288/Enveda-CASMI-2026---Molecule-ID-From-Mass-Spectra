from casmi.chemistry.mass_filter_policy import (
    get_mass_filter_decision,
)


def test_enveda_source_is_hard_filtered():

    decision = get_mass_filter_decision(
        adduct_supported=True,
        adduct_trusted=True,
        instrument_type="timsTOF",
        ingest_lib="enveda-180",
        inference_mode=False,
    )

    assert decision.hard_filter is True

    assert decision.tolerance_ppm == 10.0

    assert decision.confidence == "high"


def test_noisy_riken_source_is_not_hard_filtered():

    decision = get_mass_filter_decision(
        adduct_supported=True,
        adduct_trusted=True,
        instrument_type="QTOF",
        ingest_lib="riken",
        inference_mode=False,
    )

    assert decision.hard_filter is False

    assert decision.tolerance_ppm is None


def test_timsTOF_is_high_confidence():

    decision = get_mass_filter_decision(
        adduct_supported=True,
        adduct_trusted=True,
        instrument_type="timsTOF",
        ingest_lib="unknown",
        inference_mode=False,
    )

    assert decision.hard_filter is True

    assert decision.tolerance_ppm == 10.0


def test_unsupported_adduct_skips_filter():

    decision = get_mass_filter_decision(
        adduct_supported=False,
        adduct_trusted=False,
        instrument_type="timsTOF",
        ingest_lib="enveda-180",
        inference_mode=True,
    )

    assert decision.hard_filter is False


def test_untrusted_adduct_skips_filter():

    decision = get_mass_filter_decision(
        adduct_supported=True,
        adduct_trusted=False,
        instrument_type="timsTOF",
        ingest_lib="enveda-180",
        inference_mode=True,
    )

    assert decision.hard_filter is False


def test_test_time_timstof_uses_10ppm():

    decision = get_mass_filter_decision(
        adduct_supported=True,
        adduct_trusted=True,
        instrument_type="timsTOF",
        ingest_lib=None,
        inference_mode=True,
    )

    assert decision.hard_filter is True

    assert decision.tolerance_ppm == 10.0


def test_non_timstof_inference_uses_wider_fallback():

    decision = get_mass_filter_decision(
        adduct_supported=True,
        adduct_trusted=True,
        instrument_type="Orbitrap",
        ingest_lib=None,
        inference_mode=True,
    )

    assert decision.hard_filter is True

    assert decision.tolerance_ppm == 20.0