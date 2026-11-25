import math
from dataclasses import replace

import pytest

from manta_mapper.requirements import MissionRequirements
from manta_mapper.survey import (
    altitude_for_gsd,
    footprint,
    gsd_at_altitude,
    max_airspeed_for_trigger,
    max_exposure_for_blur,
    plan_survey,
    turn_radius,
)

REQ = MissionRequirements()  # baseline: portrait camera, 75 % / 65 % overlap
CAM = REQ.camera
# Landscape (long side across track) keeps the hand calcs below simple.
LAND = replace(REQ, camera=replace(CAM, long_side_across_track=True))


def test_altitude_for_3cm_gsd_matches_hand_calc():
    # h = GSD * f / p = 0.03 * 4.74 mm / 1.4 um = 101.57 m
    assert altitude_for_gsd(CAM, 0.03) == pytest.approx(101.571, rel=1e-4)


def test_gsd_and_altitude_are_inverse():
    for h in (50.0, 100.0, 120.0):
        assert altitude_for_gsd(CAM, gsd_at_altitude(CAM, h)) == pytest.approx(h)


def test_footprint_orientation():
    h = altitude_for_gsd(CAM, 0.03)
    across, along = footprint(LAND.camera, h)
    assert (across, along) == (pytest.approx(4608 * 0.03), pytest.approx(2592 * 0.03))
    across, along = footprint(CAM, h)  # portrait baseline swaps them
    assert (across, along) == (pytest.approx(2592 * 0.03), pytest.approx(4608 * 0.03))


def test_turn_radius_30deg():
    # R = V^2 / (g tan 30) = 196 / (9.80665 * 0.57735)
    assert turn_radius(14, 30) == pytest.approx(34.618, rel=1e-4)


def test_blur_limit():
    # 0.5 px of 3 cm at 20 m/s -> 0.75 ms
    assert max_exposure_for_blur(20, 0.03, 0.5) == pytest.approx(7.5e-4)


def test_plan_25ha_square_landscape():
    p = plan_survey(LAND, area_ha=25, airspeed_m_s=14)
    assert not p.altitude_limited
    assert p.photo_spacing_m == pytest.approx(2592 * 0.03 * 0.25)
    assert p.line_spacing_m == pytest.approx(4608 * 0.03 * 0.35)
    # 500 m square: ceil(500 / 48.384) + 1 = 12 lines of 500 m
    assert p.n_lines == 12
    assert p.line_length_m == pytest.approx(500)
    assert p.total_line_length_m == pytest.approx(6000)
    # Flown line length exceeds the area/spacing ideal because of edge coverage
    assert p.total_line_length_m >= p.ideal_line_length_m
    # Downwind leg: 19 m/s over 19.44 m -> 1.023 s between photos
    assert p.trigger_interval_min_s == pytest.approx(19.44 / 19.0)
    # 0.98 Hz needed but only 70 % of the 1 Hz camera may be used
    assert not p.trigger_ok


def test_portrait_baseline_halves_trigger_rate():
    land = plan_survey(LAND, area_ha=25)
    port = plan_survey(REQ, area_ha=25)
    assert port.trigger_rate_max_hz == pytest.approx(land.trigger_rate_max_hz * 2592 / 4608)
    assert port.trigger_ok
    assert port.total_path_m > land.total_path_m  # the price: more lines


def test_survey_time_includes_wind_penalty():
    calm = plan_survey(replace(REQ, survey=replace(REQ.survey, design_wind_m_s=0.0)), area_ha=25)
    windy = plan_survey(REQ, area_ha=25)
    # Upwind legs cost more time than downwind legs save
    assert windy.survey_time_s > calm.survey_time_s


def test_fallback_4cm_is_altitude_limited():
    # 4 cm/px needs ~135 m with the standard lens, above the 121.9 m (400 ft) ceiling
    p = plan_survey(REQ, gsd_m=0.04)
    assert p.altitude_limited
    assert p.altitude_m == pytest.approx(121.92)
    assert p.gsd_achieved_m == pytest.approx(0.0360, abs=2e-4)


def test_trigger_rate_flagged_when_too_fast():
    slow_cam = replace(REQ, camera=replace(CAM, max_trigger_rate_hz=0.7))
    assert not plan_survey(slow_cam).trigger_ok


def test_max_airspeed_for_trigger():
    # landscape: 19.44 m photo spacing x (0.7 x 1 Hz) - 5 m/s wind
    assert max_airspeed_for_trigger(LAND) == pytest.approx(19.44 * 0.7 - 5.0)
    assert max_airspeed_for_trigger(REQ) == pytest.approx(4608 * 0.03 * 0.25 * 0.7 - 5.0)


def test_airspeed_must_beat_wind():
    with pytest.raises(ValueError):
        plan_survey(REQ, airspeed_m_s=4.0)


def test_area_scales_path_roughly_linearly():
    a = plan_survey(REQ, area_ha=25).total_line_length_m
    b = plan_survey(REQ, area_ha=50).total_line_length_m
    assert 1.6 < b / a < 2.1
    assert not math.isnan(b)
