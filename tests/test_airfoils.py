import math
from pathlib import Path

import numpy as np
import pytest

from manta_mapper import airfoils as A
from manta_mapper.atmosphere import dynamic_viscosity, speed_of_sound
from manta_mapper.constraints import size_aircraft
from manta_mapper.requirements import MissionRequirements

ROOT = Path(__file__).resolve().parents[1]
REQ = MissionRequirements.from_toml(ROOT / "config" / "mission.toml")


# --- atmosphere additions ---------------------------------------------------

def test_viscosity_and_speed_of_sound_sea_level():
    assert dynamic_viscosity(288.15) == pytest.approx(1.789e-5, rel=2e-3)
    assert speed_of_sound(288.15) == pytest.approx(340.3, abs=0.2)
    assert dynamic_viscosity(305) > dynamic_viscosity(288)  # gases get more viscous when hot


# --- flight conditions --------------------------------------------------------

def test_conditions_match_sizing():
    cond = A.conditions_from_requirements(REQ, 1.5)
    sz = size_aircraft(REQ, 1.5)
    assert cond.chord_mean_m == pytest.approx(sz.mean_chord_m)
    assert cond.chord_tip_m < cond.chord_mean_m < cond.chord_center_m
    assert cond.cl_required(cond.v_cruise) == pytest.approx(sz.cruise_cl, rel=1e-6)
    assert (cond.v_min, cond.v_max) == pytest.approx(sz.airspeed_window_m_s)


def test_reynolds_hand_calc():
    cond = A.conditions_from_requirements(REQ, 1.5)
    v, c = 14.0, 0.25
    expected = cond.rho_cruise * v * c / cond.mu_cruise
    assert cond.reynolds(v, c) == pytest.approx(expected)
    # Hot, high field: well below sea-level Re (~240k for 14 m/s, 0.25 m).
    assert 150e3 < expected < 220e3
    # Tip at stall is the lowest Reynolds number the wing sees.
    assert cond.reynolds(cond.v_stall, cond.chord_tip_m, at_field=True) < 100e3


# --- polar metrics on a synthetic polar ---------------------------------------

def synthetic_polar(drop: float = 0.3, cm0: float = 0.01) -> A.Polar:
    """CL = 0.1 (alpha + 2) up to 10 deg (CL 1.2), then lose ``drop`` of it."""
    a = np.arange(-6.0, 20.01, 0.25)
    cl = 0.1 * (a + 2)
    post = a > 10
    cl[post] = 1.2 * (1 - drop * np.minimum((a[post] - 10) / 2, 1))
    cd = 0.01 + 0.01 * (cl - 0.3) ** 2
    cm = cm0 - 0.02 * cl
    return A.Polar(alpha=a, cl=cl, cd=cd, cm=cm, re=2e5, confidence=np.ones_like(a))


def test_cl_max_and_alpha():
    clmax, a_s = synthetic_polar().cl_max()
    assert clmax == pytest.approx(1.2)
    assert a_s == pytest.approx(10.0)


def test_stall_retention_gentle_vs_sharp():
    assert synthetic_polar(drop=0.0).stall_retention() == pytest.approx(1.0)
    assert synthetic_polar(drop=0.3).stall_retention() == pytest.approx(0.7)


def test_cm0_interpolates_at_zero_lift():
    assert synthetic_polar(cm0=0.015).cm0() == pytest.approx(0.015, abs=1e-9)


def test_values_at_cl_and_outside_range():
    p = synthetic_polar()
    assert p.cd_at_cl(0.3) == pytest.approx(0.01)
    assert p.cm_at_cl(0.5) == pytest.approx(0.01 - 0.01)
    assert math.isnan(p.cd_at_cl(1.5))  # above CL_max: not a flyable point


# --- scoring ------------------------------------------------------------------

def test_scale_clips_and_handles_reversed_scales():
    assert A.scale(0.0, -0.03, 0.03) == pytest.approx(0.5)
    assert A.scale(1.0, -0.03, 0.03) == 1.0
    assert A.scale(0.012, 0.016, 0.008) == pytest.approx(0.5)  # lower drag is better
    assert A.scale(math.nan, 0, 1) == 0.0


def test_weights_sum_to_one():
    assert sum(A.WEIGHTS.values()) == pytest.approx(1.0)
    for w in A.ALTERNATE_WEIGHTS.values():
        assert sum(w.values()) == pytest.approx(1.0)
        assert set(w) == set(A.WEIGHTS)


def fake_result(role="outer", cm_trim=0.0, depth=20.0, **kw) -> A.AirfoilResult:
    p = synthetic_polar()
    base = dict(
        candidate=A.Candidate("X", "x", role, ""), thickness=0.10, depth_mm=depth * 4, spar_depth_mm=depth,
        cm0=0.0, cm_trim=cm_trim, cl_max=1.2, alpha_stall=10.0, retention=0.9, cl_max_built=1.15,
        cd_window=0.011, cd_built_increase=0.05, ld_cruise=40.0, confidence=0.95, stall_confidence=0.9,
        stall_polar=p, cruise_polar=p, built_cruise_polar=p,
    )
    base.update(kw)
    return A.score(A.AirfoilResult(**base))


def test_gates():
    assert fake_result().passed
    assert not fake_result(cm_trim=-0.03).passed  # too nose-down for the outer panel
    assert fake_result(cm_trim=-0.01).passed  # washout can cover this on a swept panel
    assert not fake_result(role="center", cm_trim=-0.01).passed  # centre body must self-trim
    assert not fake_result(depth=10.0).passed  # spar does not fit
    assert not fake_result(confidence=0.5).passed


def test_rank_puts_passing_first_and_scores_monotonic():
    good = fake_result(cm_trim=0.02)
    ok = fake_result(cm_trim=0.0)
    failing = fake_result(cm_trim=-0.05, retention=1.0, cd_window=0.008)
    assert good.total > ok.total
    ranked = A.rank([failing, ok, good])
    assert ranked[0] is good and ranked[-1] is failing


def test_weight_sensitivity_ignores_failing():
    good = fake_result(cm_trim=0.02)
    failing = fake_result(cm_trim=-0.05, cl_max_built=1.5)
    assert set(A.weight_sensitivity([good, failing]).values()) == {"X"}


def test_wing_cl_max_estimate():
    assert A.wing_cl_max_estimate(1.0) == pytest.approx(0.81)


# --- NeuralFoil (needs the aero extra) ----------------------------------------

def _neuralfoil():
    pytest.importorskip("aerosandbox")
    pytest.importorskip("neuralfoil")


def test_symmetric_airfoil_has_no_camber_effects():
    _neuralfoil()
    p = A.run_polar(A.load_airfoil("naca0012"), 3e5, 0.04)
    assert np.interp(0.0, p.alpha, p.cl) == pytest.approx(0.0, abs=0.02)
    assert p.cm0() == pytest.approx(0.0, abs=0.01)


def test_reflex_beats_ordinary_camber_on_pitch():
    """The whole point of a reflexed section: Cm0 near zero or positive."""
    _neuralfoil()
    reflexed = A.run_polar(A.load_airfoil("mh78"), 3e5, 0.04).cm0()
    ordinary = A.run_polar(A.load_airfoil("naca4412"), 3e5, 0.04).cm0()
    assert reflexed > 0.02
    assert ordinary < -0.06


def test_blunt_trailing_edge_is_applied():
    _neuralfoil()
    af = A.load_airfoil("s5010")
    built = A.as_built(af, 0.15)
    assert built.TE_thickness() == pytest.approx(1e-3 / 0.15, rel=0.05)


@pytest.fixture(scope="module")
def ranked():
    _neuralfoil()
    return A.compare(A.conditions_from_requirements(REQ, 1.5))


def test_compare_finds_a_passing_airfoil_for_each_role(ranked):
    for role in ("outer", "center"):
        assert ranked[role][0].passed
        assert ranked[role][0].confidence >= A.MIN_CONFIDENCE


def test_documented_choice_matches_pipeline(ranked):
    """docs/airfoils.md names the chosen sections. If a NeuralFoil update or a
    mission change flips the answer, this fails so the doc gets revisited."""
    doc = (ROOT / "docs" / "airfoils.md").read_text()
    outer = ranked["outer"][0].candidate.name
    center = ranked["center"][0].candidate.name
    assert f"Outer panels: **{outer}**" in doc
    assert f"Centre body: **{center}**" in doc
