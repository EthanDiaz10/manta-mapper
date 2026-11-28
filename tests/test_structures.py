"""M2.2 spar sizing and load cases."""

import math

import numpy as np
import pytest

from manta_mapper import structures as S
from manta_mapper import weights as W
from manta_mapper.geometry import BWBGeometry
from manta_mapper.requirements import MissionRequirements


@pytest.fixture(scope="module")
def req():
    return MissionRequirements()


@pytest.fixture(scope="module")
def geom():
    return S.load_geometry()


@pytest.fixture(scope="module")
def result(req, geom):
    return S.run(req, geom, use_vlm=False, progress=lambda *a: None)


def test_tube_section_properties():
    t = S.Tube(0.010, 0.008, S.PULTRUDED, 6.0)
    assert t.I == pytest.approx(math.pi / 64 * (1e-8 - 0.4096e-8))
    assert t.Z == pytest.approx(t.I / 0.005)
    assert t.kg_per_m == pytest.approx(0.0438, rel=0.01)  # ~44 g/m
    # every catalogue tube has a 1 mm wall so each nests inside the next size up
    for tube in S.CATALOGUE:
        assert tube.wall == pytest.approx(0.001)


def test_point_load_at_tip_gives_textbook_root_moment(req, geom):
    pm = S.panel_model(geom, req, use_vlm=False)
    tip = pm.line.point(pm.line.length)
    case = S.LoadCase("unit", "test", np.array([[*tip, 0, 0, 10.0]]))
    s = np.array([-0.05, 0.0, pm.line.length / 2])
    it = S.internal_loads(case, pm, s, socket_m=0.10)
    d = pm.line.d  # vertical load at the tip of a swept, dihedralled spar: arm = length x horizontal part of d
    assert it.bending[1] == pytest.approx(10.0 * pm.line.length * math.hypot(d[0], d[1]), rel=1e-6)
    assert it.bending[2] == pytest.approx(it.bending[1] / 2, rel=0.02)
    assert abs(it.torsion[1]) < 1e-6  # load on the spar axis: no twist
    # inside the sleeve the moment falls linearly to the inner rib
    assert it.bending[0] == pytest.approx(it.bending[1] / 2, rel=1e-6)


def test_lift_shape_sums_to_half_the_aircraft(req, geom):
    pm = S.panel_model(geom, req, use_vlm=False)
    assert 0.3 < pm.panel_lift_frac < 0.5  # outer panel carries most of its half; the pod/blend the rest


def test_lighter_wing_rides_gusts_harder(req, geom):
    pm = S.panel_model(geom, req, use_vlm=False)
    env = S.flight_envelope(pm, req, S.LoadSettings(), 1.18)
    n_light = max(g["n"] for g in env["as built"]["gusts"])
    n_heavy = max(g["n"] for g in env["design"]["gusts"])
    assert n_light > n_heavy > 4.0


def test_chosen_spar_passes_and_nests(result, geom):
    ch = result.chosen
    assert ch.ok
    segs = ch.layout.segments
    for a, b in zip(segs[:-1], segs[1:]):
        assert b.tube.od == pytest.approx(a.tube.id)  # telescopes
        assert b.s0 < a.s1  # bonded overlap
    assert segs[0].s0 == pytest.approx(-result.settings.socket_length_m)
    assert segs[-1].s1 == pytest.approx(result.panel.line.length)
    assert all(c.ok for c in ch.checks)


def test_m15_single_tube_is_no_longer_enough(result):
    """10x8 full length: too thick for 2 mm of EPP at the tip of the 1.84 m wing."""
    assert not result.current.ok


def test_landing_sweep_is_monotonic_and_far_from_failure(result):
    u = [x[2] for x in result.landing_sweep]
    assert all(b > a for a, b in zip(u[:-1], u[1:]))
    assert u[-1] < 1.0  # even a 4 m/s arrival does not break the spar


def test_roll_wrapped_resists_reversal_better(req, geom):
    pm = S.panel_model(geom, req, use_vlm=False)
    L = pm.line.length
    pul = S.SparLayout((S.Segment(S.Tube(0.010, 0.008, S.PULTRUDED, 6), -0.1, L),))
    rw = S.SparLayout((S.Segment(S.Tube(0.010, 0.008, S.ROLL_WRAPPED, 11), -0.1, L),))
    assert S.reversal_speed(rw, pm, req) > S.reversal_speed(pul, pm, req) * 1.1


def test_joint_and_retention(result):
    j = result.joint
    assert j["sleeve"].id == pytest.approx(result.chosen.layout.segments[0].tube.od)
    assert j["rib"][0] < result.settings.max_utilisation
    assert j["bond"][0] < result.settings.max_utilisation
    assert j["retention_design_n"] > j["retention_n"] > 0  # launch pulls the swept panel outward


def test_weights_use_the_m22_spar(geom):
    old = W.build_up(geom, W.default_propulsion(), W.BuildChoices(spar_layout=None))
    new = W.build_up(geom, W.default_propulsion())
    names = [i.name for i in new.items]
    assert any(n.startswith("Spar carry-through") for n in names)
    assert 0.0 < new.mass_kg() - old.mass_kg() < 0.06
    assert not new.fit_problems
    # legacy path keeps the M1.5 numbers for the baseline
    b = W.build_up(BWBGeometry(), W.default_propulsion(), W.BuildChoices(spar_layout=None))
    assert b.mass_kg() == pytest.approx(1.195, abs=0.005)
