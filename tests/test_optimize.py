"""M1.6 optimizer: variables round-trip, constraints behave, one coarse evaluation of the baseline."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerosandbox")
pytest.importorskip("scipy")

from manta_mapper import aero as A  # noqa: E402
from manta_mapper import optimize as O  # noqa: E402
from manta_mapper.geometry import BWBGeometry, baseline  # noqa: E402
from manta_mapper.requirements import MissionRequirements  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REQ = MissionRequirements.from_toml(ROOT / "config" / "mission.toml")


@pytest.fixture(scope="module")
def x0():
    return O.design_point(baseline(REQ, 1.5))


@pytest.fixture(scope="module")
def ev0(x0):
    return O.evaluate(x0, REQ)  # coarse, uncorrected: ~5 s


def test_design_point_round_trips(x0):
    g = baseline(REQ, 1.5)
    g2 = O.make_geometry(x0)
    for n in BWBGeometry.DESIGN_VARIABLES:
        assert getattr(g2, n) == pytest.approx(getattr(g, n))
    u = O.to_unit(x0)
    assert np.all((u >= 0) & (u <= 1)), "baseline must sit inside the bounds"
    back = O.from_unit(u)
    for n in O.NAMES:
        assert back[n] == pytest.approx(x0[n])


def test_latin_hypercube_covers_each_strata():
    u = O.latin_hypercube(20, 4, seed=1)
    assert u.shape == (20, 4)
    for j in range(4):
        assert sorted(np.floor(u[:, j] * 20).astype(int)) == list(range(20))


def test_vlm_resolution_context_restores():
    before = (A.SPANWISE_RESOLUTION, A.CHORDWISE_RESOLUTION)
    with A.vlm_resolution(2, 6):
        assert (A.SPANWISE_RESOLUTION, A.CHORDWISE_RESOLUTION) == (2, 6)
    assert (A.SPANWISE_RESOLUTION, A.CHORDWISE_RESOLUTION) == before


def test_baseline_evaluates_close_to_m15(ev0):
    assert not ev0.error, ev0.error
    m = ev0.metrics
    # M1.5: 1195 g as built, ~$470, X2216 set; M2.2 spar (telescoping tubes, sleeves, carry-through)
    # adds ~30 g and ~$28. M1.4: stall 8.88 m/s at SM 10 %, NP 33.6 % MAC
    assert m["mass_built_kg"] == pytest.approx(1.225, abs=0.01)
    assert m["cost_usd"] == pytest.approx(497, abs=3)
    assert "X2216" in ev0.propulsion
    assert m["stall_speed_m_s"] == pytest.approx(8.88, abs=0.15)
    assert m["np_pct_mac"] == pytest.approx(33.6, abs=0.5)
    assert m["propulsion_fails"] == 0


def test_baseline_only_misses_dutch_roll(ev0):
    """The M1.3 baseline passes everything except the Dutch roll yardstick (0.073-0.078 vs 0.08)."""
    failing = {n for n, c in ev0.constraints.items() if c[3] > 0}
    assert failing <= {"Dutch roll damping"}


def test_constraint_signs():
    s = O.Settings()
    m = dict(stall_speed_m_s=9.5, tip_stall_margin=0.2, stall_first_strip_rel=-0.1, stall_elevon_deg=-10,
             cruise_elevon_sm5_deg=1, cruise_elevon_sm10_deg=-1, pod_pitch_cruise_deg=6, dutch_roll_damping=0.1,
             short_period_damping=0.6, spiral_rate_per_s=0.02, cn_beta=0.01, wing_loading_kg_m2=4,
             window_low_m_s=11, propulsion_fails=0, payload_floor_ratio=2, bay_height_margin_m=0.02,
             fit_problems=0, spar_depth_tip_m=0.015, spar_needed_m=0.012, fin_clearance_m=0.03, tip_chord_m=0.15,
             battery_fwd_margin_m=0.05, battery_aft_margin_m=0.02, mass_built_kg=1.2)
    c = O.constraints_from(m, REQ, s)
    bad = {n for n, v in c.items() if v[3] > 0}
    assert bad == {"stall speed at forward CG, m/s (AC-1)"}
    c2 = O.constraints_from(dict(m, stall_speed_m_s=8.5), REQ, s)
    assert all(v[3] <= 0 for v in c2.values())


def test_bad_geometry_is_infeasible_not_a_crash(x0):
    ev = O.evaluate(dict(x0, span_m=0.5), REQ)  # no room for an outer panel
    assert ev.error and not ev.feasible
    assert ev.objective() > 1e3


def test_correction_is_added(ev0, x0):
    corr = {k: 0.0 for k in O.AERO_METRICS}
    corr["dutch_roll_damping"] = 0.05
    ev = O.evaluate(x0, REQ, correction=corr)
    assert ev.metrics["dutch_roll_damping"] == pytest.approx(ev0.metrics["dutch_roll_damping"] + 0.05, abs=1e-6)
    assert ev.raw["dutch_roll_damping"] == pytest.approx(ev0.raw["dutch_roll_damping"], abs=1e-6)
    assert "Dutch roll damping" not in {n for n, c in ev.constraints.items() if c[3] > 0}


def test_rank_prefers_feasible(ev0):
    good = replace(ev0, constraints={k: (0, 0, "<=", -1.0) for k in ev0.constraints})
    bad = replace(ev0, constraints={k: (0, 0, "<=", 0.5) for k in ev0.constraints})
    assert O.rank_key(good, 100) < O.rank_key(bad, 100)
