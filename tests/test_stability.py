"""M1.4: aerodynamics, trim, stall pattern and modes.

The first group checks the method on cases with known answers; the second
pins the baseline's key results to the requirements they serve.
"""

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerosandbox")

from manta_mapper import aero as A  # noqa: E402
from manta_mapper import geometry as G  # noqa: E402
from manta_mapper import stability as S  # noqa: E402
from manta_mapper.requirements import MissionRequirements  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REQ = MissionRequirements.from_toml(ROOT / "config" / "mission.toml")
FLIGHT = A.Flight(14.0, 1320.0, 25.0)


def plain_wing(**kw):
    """Straight-tapered swept wing, symmetric section, no twist, no real fins."""
    d = dict(span_m=1.5, pod_chord_m=0.3, pod_half_width_m=0.01, blend_width_m=0.0, root_chord_m=0.3,
             tip_chord_m=0.17, sweep_le_deg=25.0, dihedral_deg=0.0, washout_deg=0.0, fin_length_m=0.02,
             pod_airfoil="naca0009", outer_airfoil="naca0009")
    d.update(kw)
    return G.BWBGeometry(**d)


# --- method checks -------------------------------------------------------------

def test_plain_swept_wing_neutral_point_is_textbook():
    """A swept tapered wing of AR ~6 has its aerodynamic centre near 25 % MAC."""
    g = plain_wing()
    m = A.longitudinal_model(g, FLIGHT)
    assert 22.0 < m.neutral_point_pct_mac < 28.0
    # symmetric section, no twist: no lift and no moment at zero alpha
    assert abs(m.cl0) < 1e-3 and abs(m.cm0) < 1e-3
    # finite-wing lift slope: below 2 pi, above the low-AR limit
    assert 3.5 < m.cl_alpha * 180 / math.pi < 5.5


def test_elevon_signs_and_size():
    g = plain_wing(elevon_start=0.1, elevon_end=0.9)
    r0 = A.run_vlm(g, FLIGHT, 3.0)
    down = A.run_vlm(g, FLIGHT, 3.0, elevator_deg=5.0, xyz_ref=(0.25, 0, 0))
    ref = A.run_vlm(g, FLIGHT, 3.0, xyz_ref=(0.25, 0, 0))
    assert down["CL"] > r0["CL"]  # trailing edge down adds lift
    assert down["Cm"] < ref["Cm"]  # ...behind the CG, so it pitches the nose down
    roll = A.run_vlm(g, FLIGHT, 3.0, aileron_deg=5.0)
    assert roll["Cl"] > 0  # right trailing edge up rolls right
    assert roll["CL"] == pytest.approx(r0["CL"], abs=2e-3)
    # About 0.6 x CL_alpha x flapped area fraction per radian (thin-airfoil tau for 25 % chord),
    # times the 0.75 effectiveness factor. Loose band: this guards gross modelling errors.
    dcl = (down["CL"] - r0["CL"]) / math.radians(5.0)
    assert 0.5 < dcl < 1.5


def test_split_halves_match_symmetric_wing():
    """The two-half model with no deflection must equal AeroSandbox's own symmetric wing."""
    import aerosandbox as asb

    g = plain_wing()
    mine = A.run_vlm(g, FLIGHT, 4.0)
    ap = g.to_airplane(xyz_ref=(0, 0, 0))
    op = asb.OperatingPoint(atmosphere=FLIGHT.atmosphere(), velocity=FLIGHT.velocity, alpha=4.0)
    ref = asb.VortexLatticeMethod(ap, op, spanwise_resolution=12, chordwise_resolution=10).run()
    assert mine["CL"] == pytest.approx(float(ref["CL"]), rel=0.02)


def test_trefftz_efficiency_of_elliptic_loading_is_one():
    g = plain_wing()
    y = np.linspace(0.005, g.semispan_m - 0.005, 80)
    width = np.full_like(y, g.semispan_m / 80)
    cl_loc = np.sqrt(1 - (y / g.semispan_m) ** 2) / A.chord_at(g, y)  # c*cl elliptic
    _, e = A.induced_drag_trefftz(g, y, cl_loc, width)
    assert e == pytest.approx(1.0, abs=0.01)


def test_linear_model_trim_is_consistent():
    g = plain_wing()
    m = A.LongitudinalModel(g, cl0=0.0, cl_alpha=0.08, cl_de=0.015, cm0=0.05, cm_alpha=-0.03, cm_de=-0.02)
    x = 0.2
    a, de = m.trim(0.4, x)
    assert float(m.cl(a, de)) == pytest.approx(0.4)
    assert float(m.cm_about(x, a, de)) == pytest.approx(0.0, abs=1e-12)
    assert m.static_margin(m.cg_for_static_margin(0.075)) == pytest.approx(0.075)


def test_short_period_matches_two_state_approximation():
    """The 4-state modes must reproduce the classic short-period approximation."""
    g = plain_wing()
    d = S.Derivatives(alpha_deg=4.0, cl=0.4, cd=0.03, cl_a=4.5, cd_a=0.2, cm_a=-0.4, cl_q=3.0, cm_q=-2.5,
                      cy_b=-0.02, cl_b=-0.05, cn_b=0.01, cy_p=0.0, cl_p=-0.45, cn_p=-0.03, cy_r=0.01, cl_r=0.1,
                      cn_r=-0.01, cl_da=0.2, cn_da=0.0, cm_de=-0.4, cl_de_lift=0.9)
    inertia = S.Inertia(1.5, 0.2, 0.1, 0.02, 0.12)
    md = S.flight_modes(d, inertia, g, FLIGHT)
    zw, mw, mq, V = md.a_long[1, 1], md.a_long[2, 1], md.a_long[2, 2], FLIGHT.velocity
    wn_approx = math.sqrt(zw * mq - V * mw)
    assert md.short_period.omega_n == pytest.approx(wn_approx, rel=0.1)
    assert md.short_period.damping > 0 and md.phugoid.period_s > 3 * md.short_period.period_s


def test_inertia_estimate_is_plausible():
    g = G.baseline(REQ, 1.5)
    i = S.estimate_inertia(g, 1.5, 0.27)
    kx, ky, kz = i.radii_of_gyration_m
    assert 0.2 * g.semispan_m < kx < 0.6 * g.semispan_m
    assert 0.2 * g.mac_m < ky < 0.8 * g.pod_chord_m
    assert i.izz == pytest.approx(i.ixx + i.iyy)


# --- baseline results ------------------------------------------------------------

@pytest.fixture(scope="module")
def base():
    g = G.baseline(REQ, 1.5)
    fl = A.cruise_flight(REQ)
    m = A.longitudinal_model(g, fl)
    return g, fl, m


def test_baseline_neutral_point_and_self_trim(base):
    g, fl, m = base
    assert 30.0 < m.neutral_point_pct_mac < 37.0
    for sm in S.STATIC_MARGINS:  # AC-8: trims at cruise within 3 deg over the whole AC-7 band
        a, de = m.trim(A.cl_level(1.5, g, fl), m.cg_for_static_margin(sm))
        assert abs(de) <= 3.0
        assert m.cm0_about(m.cg_for_static_margin(sm)) > 0  # reflex + washout: nose-up at zero lift


def test_baseline_stalls_inboard_first_and_meets_stall_speed(base):
    g, fl, m = base
    st = A.stall(g, m, m.cg_for_static_margin(S.DESIGN_SM), REQ, 1.5)
    assert st.stall_speed_m_s <= REQ.limits.max_stall_speed_m_s  # AC-1
    # the stall starts on the inner third of the outer panel, not near the tips
    assert st.first_strip_y < g.root_y_m + g.outer_semispan_m / 3
    assert st.tip_margin > 0.08  # outboard elevon region keeps real margin


def test_baseline_cruise_drag_is_reasonable(base):
    g, fl, m = base
    pt, _ = A.trimmed_point(g, m, m.cg_for_static_margin(S.DESIGN_SM), fl, 1.5)
    assert 0.010 < pt.cd - pt.cd_induced < 0.030
    assert 0.85 < pt.span_efficiency <= 1.0
    assert 10 < pt.ld < 20
