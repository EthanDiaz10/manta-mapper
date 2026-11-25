from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from manta_mapper.atmosphere import density, density_altitude, isa
from manta_mapper.constraints import (
    constraint_analysis,
    induced_drag_factor,
    size_aircraft,
    stall_speed,
)
from manta_mapper.requirements import G0, MissionRequirements

REQ = MissionRequirements()
ROOT = Path(__file__).resolve().parents[1]


# --- atmosphere ---------------------------------------------------------------

def test_isa_sea_level():
    t, p, rho = isa(0.0)
    assert t == pytest.approx(288.15)
    assert p == pytest.approx(101325)
    assert rho == pytest.approx(1.2250, abs=1e-4)


def test_isa_1500m_and_hot_day():
    assert density(1500) == pytest.approx(1.0581, abs=2e-4)
    assert density(0, 15) < density(0)


def test_density_altitude_roundtrip():
    assert density_altitude(density(1000)) == pytest.approx(1000, abs=0.5)


# --- constraint analysis ----------------------------------------------------

def test_stall_limit_matches_hand_calc():
    res = constraint_analysis(REQ)
    rho = res.densities.field
    expected = 0.5 * rho * 9.0**2 * REQ.assumptions.cl_max
    assert res.limits["stall"] == pytest.approx(expected)
    assert float(stall_speed(res.limits["stall"], rho, REQ.assumptions.cl_max)) == pytest.approx(9.0)


def test_stall_drives_design_below_hand_launch_cap():
    # With CL_max = 1.0, a 9 m/s stall at 1220 m on an ISA+25 day allows only
    # ~4.1 kg/m^2, so stall (not the 6 kg/m^2 cap) is the active limit.
    req = MissionRequirements.from_toml(ROOT / "config" / "mission.toml")
    res = constraint_analysis(req)
    assert res.driving_limit == "stall"
    assert res.design_wing_loading_kg_m2 < req.limits.max_wing_loading_kg_m2
    assert res.design_wing_loading_kg_m2 == pytest.approx(4.12, abs=0.02)


def test_cap_drives_when_clmax_is_high():
    req = replace(REQ, assumptions=replace(REQ.assumptions, cl_max=1.6))
    assert constraint_analysis(req).driving_limit == "hand-launch cap"


def test_cruise_curve_minimum_at_min_drag_wing_loading():
    # Minimum P/W at fixed speed happens at CL = sqrt(CD0 / k), i.e. max L/D.
    res = constraint_analysis(REQ, ws_range_n_m2=(10, 200), n_points=4000)
    a = REQ.assumptions
    k = induced_drag_factor(a.aspect_ratio, a.oswald_e)
    q = 0.5 * res.densities.cruise * a.cruise_speed_m_s**2
    ws_star = q * np.sqrt(a.cd0 / k)
    ws_min = res.wing_loading[np.argmin(res.curves["cruise"])]
    assert ws_min == pytest.approx(ws_star, rel=0.01)


def test_turn_needs_more_power_than_cruise():
    res = constraint_analysis(REQ)
    assert np.all(res.curves["turn"] >= res.curves["cruise"])


def test_design_power_is_envelope_at_design_point():
    res = constraint_analysis(REQ)
    at = {n: np.interp(res.design_wing_loading, res.wing_loading, c) for n, c in res.curves.items()}
    assert res.design_power_loading == pytest.approx(max(at.values()))
    assert res.driving_curve == max(at, key=at.get)


# --- sizing -----------------------------------------------------------------

def test_size_aircraft_consistency():
    sz = size_aircraft(REQ, mass_kg=1.5)
    assert sz.wing_area_m2 * constraint_analysis(REQ).design_wing_loading == pytest.approx(1.5 * G0)
    assert sz.span_m**2 / sz.wing_area_m2 == pytest.approx(REQ.assumptions.aspect_ratio)
    assert sz.stall_speed_m_s == pytest.approx(9.0)
    assert 8 < sz.cruise_l_over_d < 16
    assert sz.airspeed_window_ok


def test_reserve_is_respected():
    sz = size_aircraft(REQ, mass_kg=1.5)
    used = (sz.cruise_power_elec_w + REQ.assumptions.avionics_power_w) * sz.mission_time_s / 3600
    assert used == pytest.approx(sz.energy_used_wh)
    assert used / sz.energy_required_wh == pytest.approx(1 - REQ.limits.battery_reserve)


def test_battery_sized_for_stretch_with_margin():
    lim = REQ.limits
    small = size_aircraft(REQ, 1.5, area_ha=25)
    big = size_aircraft(REQ, 1.5, area_ha=50)
    assert small.battery_energy_wh == pytest.approx(big.battery_energy_wh)
    # Stretch mission lands with more than the reserve because of the margin:
    # 1 - 0.7 / 1.25 = 0.44
    assert big.reserve_on_landing == pytest.approx(1 - (1 - lim.battery_reserve) / lim.energy_design_margin)
    assert small.reserve_on_landing > big.reserve_on_landing


def test_airspeed_window_needs_minimum_width():
    sz = size_aircraft(REQ, 1.5)
    assert sz.airspeed_window_width_m_s >= REQ.limits.min_airspeed_window_m_s
    assert sz.airspeed_window_ok
    land = replace(REQ, camera=replace(REQ.camera, long_side_across_track=True))
    assert not size_aircraft(land, 1.5).airspeed_window_ok  # landscape at 1 Hz is too tight


def test_stretch_mission_needs_more_energy():
    a = size_aircraft(REQ, 1.5, area_ha=25)
    b = size_aircraft(REQ, 1.5, area_ha=50)
    assert b.energy_required_wh > a.energy_required_wh
    assert b.mission_time_s > a.mission_time_s
