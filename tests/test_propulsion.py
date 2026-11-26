"""M1.5 propulsion: method checks on known answers, then the baseline choice and missions."""

import math
from pathlib import Path

import pytest

pytest.importorskip("aerosandbox")

from manta_mapper import propulsion as P  # noqa: E402
from manta_mapper import weights as W  # noqa: E402
from manta_mapper.geometry import baseline  # noqa: E402
from manta_mapper.requirements import MissionRequirements  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REQ = MissionRequirements.from_toml(ROOT / "config" / "mission.toml")
PROP_9X6 = next(p for p in P.PROPS if p.diameter_in == 9 and p.pitch_in == 6)


# --- method checks ---------------------------------------------------------------

def test_prop_matches_published_thin_electric_ranges():
    """Static CT ~0.09-0.12, CP ~0.035-0.06, peak efficiency 0.55-0.75, zero thrust near J = 0.8."""
    ct0, cp0, _ = P.prop_coefficients(PROP_9X6, 0.0)
    assert 0.08 <= ct0 <= 0.12
    assert 0.035 <= cp0 <= 0.06
    etas = [P.prop_coefficients(PROP_9X6, j)[2] for j in (0.3, 0.4, 0.5, 0.6, 0.7)]
    assert 0.55 <= max(etas) <= 0.75
    assert P.prop_coefficients(PROP_9X6, 0.85)[0] < 0.01


def test_more_pitch_more_thrust_at_speed():
    lo = next(p for p in P.PROPS if (p.diameter_in, p.pitch_in) == (10, 6))
    hi = next(p for p in P.PROPS if (p.diameter_in, p.pitch_in) == (10, 7))
    assert P.prop_thrust_torque(hi, 12, 7000, 1.1, 1.8e-5)[0] > P.prop_thrust_torque(lo, 12, 7000, 1.1, 1.8e-5)[0]


def test_prop_map_agrees_with_direct_bemt():
    t1, q1 = P.prop_thrust_torque(PROP_9X6, 10.0, 8000, 1.05, 1.85e-5)
    t2, q2 = P.prop_map_thrust_torque(PROP_9X6, 10.0, 8000, 1.05, 1.85e-5)
    assert t2 == pytest.approx(t1, rel=0.08)
    assert q2 == pytest.approx(q1, rel=0.08)


def test_motor_spins_near_kv_times_volts_when_unloaded():
    """A tiny prop barely loads the motor: rpm ~ Kv x duty x pack voltage."""
    tiny = P.Prop("tiny", 2.0, 1.0, 0.001, 0)
    ps = P.PropulsionSet(P.MOTORS[0], tiny, P.PACKS[2])
    op = P.full_throttle(ps, 0.0, P.Air(1.2, 1.8e-5), 1.0)
    v = P.MAX_DUTY * ps.pack.open_circuit_v(1.0)
    assert op.rpm == pytest.approx(ps.motor.kv_rpm_per_v * (v - ps.motor.no_load_current_a * ps.motor.resistance_ohm),
                                   rel=0.05)


def test_cell_voltage_curve_is_monotonic():
    v = [P.cell_ocv(s / 20) for s in range(21)]
    assert all(b > a for a, b in zip(v, v[1:]))
    assert v[0] == 3.0 and v[-1] == pytest.approx(4.18)


def test_polar_scaling_hits_the_target_ld():
    for ld in (14, 12, 10):
        assert P.M14_POLAR.scaled_to(ld).best_ld == pytest.approx(ld, rel=1e-6)
    assert P.M14_POLAR.best_ld == pytest.approx(16.4, abs=0.3)  # docs/stability_report.txt


# --- baseline --------------------------------------------------------------------

@pytest.fixture(scope="module")
def sel():
    geom = baseline(REQ, 1.5)
    cond = P.conditions(REQ, 1.5, geom.reference_area_m2, 8.9, 5.2)
    return P.select_propulsion(REQ, cond, P.M14_POLAR), cond


def test_selection_is_the_pinned_default(sel):
    """weights.default_propulsion (used for the inertia in stability.py) must stay what the search picks."""
    s, _ = sel
    d = W.default_propulsion()
    c = s.chosen.ps
    assert (c.motor, c.prop, c.pack, c.esc) == (d.motor, d.prop, d.pack, d.esc)


def test_chosen_set_meets_every_check(sel):
    s, _ = sel
    assert s.chosen.feasible, {k: v for k, v in s.chosen.checks.items() if not v[0]}
    assert s.chosen.static_thrust_to_weight >= 0.6
    assert s.chosen.climb_rate_m_s >= REQ.limits.min_climb_rate_m_s


def test_missions_land_with_the_reserve(sel):
    s, cond = sel
    ps = s.chosen.ps
    r = REQ.limits.battery_reserve
    planning, pess = P.M14_POLAR.scaled_to(P.LD_PLANNING), P.M14_POLAR.scaled_to(P.LD_PESSIMISTIC)
    assert P.mission_energy(ps, cond, REQ, pess, REQ.survey.area_min_ha).reserve >= r
    assert P.mission_energy(ps, cond, REQ, planning, REQ.survey.area_stretch_ha).reserve >= r


def test_cruise_power_is_sane(sel):
    """At L/D 12 the drag power is ~W V / 12 = 17 W; with prop x motor x ESC 0.4-0.6 that is 30-45 W."""
    s, _ = sel
    p = s.chosen.cruise.electrical_power_w
    assert 28 <= p <= 45
    assert math.isfinite(s.chosen.cruise.efficiency)
