import math
from pathlib import Path

import numpy as np
import pytest

from manta_mapper import geometry as G
from manta_mapper.requirements import MissionRequirements

ROOT = Path(__file__).resolve().parents[1]
REQ = MissionRequirements.from_toml(ROOT / "config" / "mission.toml")


@pytest.fixture(scope="module")
def base():
    return G.baseline(REQ, 1.5)


# --- pure geometry -------------------------------------------------------------

def simple_wing(**kw):
    """A plain trapezoid: pod and outer panel with the same chord, no sweep, no blend."""
    d = dict(span_m=1.0, pod_chord_m=0.2, pod_half_width_m=0.1, blend_width_m=0.0,
             root_chord_m=0.2, tip_chord_m=0.2, sweep_le_deg=0.0, dihedral_deg=0.0, washout_deg=0.0)
    d.update(kw)
    return G.BWBGeometry(**d)


def test_rectangular_wing_hand_calc():
    g = simple_wing()
    assert g.reference_area_m2 == pytest.approx(0.2)
    assert g.mac_m == pytest.approx(0.2)
    assert g.aspect_ratio == pytest.approx(5.0)
    assert g.mac_le_x_m == pytest.approx(0.0)
    assert g.mac_y_m == pytest.approx(0.25)  # centroid of a half-span rectangle


def test_tapered_wing_mac_matches_textbook():
    # Straight taper from the centreline: MAC = 2/3 cr (1 + l + l^2)/(1 + l)
    cr, ct = 0.3, 0.12
    g = simple_wing(pod_half_width_m=1e-6, root_chord_m=cr, pod_chord_m=cr, tip_chord_m=ct, span_m=1.2)
    lam = ct / cr
    assert g.mac_m == pytest.approx(2 / 3 * cr * (1 + lam + lam**2) / (1 + lam), rel=1e-4)
    assert g.reference_area_m2 == pytest.approx(1.2 * (cr + ct) / 2, rel=1e-4)
    # MAC station: b/6 (1 + 2 l)/(1 + l)
    assert g.mac_y_m == pytest.approx(1.2 / 6 * (1 + 2 * lam) / (1 + lam), rel=1e-3)


def test_sweep_moves_mac_aft_and_quarter_chord_sweep():
    g = simple_wing(sweep_le_deg=30.0)
    assert g.mac_le_x_m > 0
    # Constant chord: every chord-fraction line has the LE sweep
    assert g.outer_sweep_deg(0.25) == pytest.approx(30.0)
    assert g.spar_sweep_deg == pytest.approx(30.0)


def test_trailing_edge_is_straight_from_pod_to_panel_root(base):
    st = base.stations()
    te = [s["le"][0] + s["chord"] for s in st[:3]]
    assert te == pytest.approx([base.pod_chord_m] * 3)


def test_washout_is_linear_and_nose_down(base):
    assert base.twist_at(0.0) == pytest.approx(base.root_twist_deg)
    assert base.twist_at(1.0) == pytest.approx(base.root_twist_deg - base.washout_deg)
    assert base.twist_at(0.5) == pytest.approx(base.root_twist_deg - base.washout_deg / 2)
    assert base.stations()[-1]["twist"] < base.stations()[2]["twist"]


def test_outer_panel_stations_are_collinear(base):
    """Hot-wire friendly: LE and TE of the outer panel are straight lines through all stations."""
    outer = [s for s in base.stations() if s["blend"] == 1.0]
    le = np.array([s["le"] for s in outer])
    te = le + np.array([[s["chord"], 0, 0] for s in outer])
    for line in (le, te):
        d = line - line[0]
        cross = np.cross(d[1:], d[-1])
        assert np.allclose(cross, 0, atol=1e-12)


def test_elevon_lines_up_with_stations(base):
    st = base.stations()
    flags = [s["elevon"] for s in st]
    assert sum(flags) >= 1
    names = [s["name"] for s in st]
    assert "elevon start" in names and "elevon end" in names


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError, match="reverse taper"):
        G.BWBGeometry(tip_chord_m=0.3, root_chord_m=0.25)
    with pytest.raises(ValueError, match="no outer panel"):
        G.BWBGeometry(span_m=0.3)
    with pytest.raises(ValueError, match="spar"):
        G.BWBGeometry(spar_chord_fraction=0.8)


def test_design_vector_round_trip(base):
    v = base.design_vector()
    assert set(v) == set(G.BWBGeometry.DESIGN_VARIABLES)
    assert base.with_changes(**v) == base


# --- baseline against the requirements ----------------------------------------

def test_baseline_hits_target_area(base):
    assert base.reference_area_m2 == pytest.approx(G.target_area_m2(REQ, 1.5), rel=1e-9)


def test_baseline_meets_stall_with_margin(base):
    vs = G.stall_speed_m_s(base, REQ, 1.5)
    assert vs < REQ.limits.max_stall_speed_m_s  # AC-1
    assert vs == pytest.approx(9.0 / math.sqrt(G.AREA_MARGIN), rel=1e-6)
    assert 1.5 / base.reference_area_m2 <= REQ.limits.max_wing_loading_kg_m2  # AC-2


def test_baseline_proportions_are_sensible(base):
    assert 1.4 < base.span_m < 1.7  # in line with the ~1.48 m first-cut sizing
    assert 5.0 < base.aspect_ratio < 7.0
    assert 15 < base.outer_sweep_deg(0.25) < 30
    assert 0.4 < base.taper_ratio < 0.7
    assert base.washout_deg > 0  # tips nose-down so they stall last


def test_spar_fits_root_to_tip(base):
    assert base.spar_fits
    assert base.spar_depth_m(0.0) > base.spar_depth_m(1.0)
    need = base.spar_diameter_m + 2 * base.spar_min_cover_m
    assert base.spar_depth_m(1.0) >= need


def test_payload_fits_in_pod(base):  # AC-12
    fit = base.payload_fit()
    assert fit["fits"]
    bay = base.pod_bay()
    assert bay["max_internal_height_m"] > max(i.height_m for i in base.payload) + 0.01
    # A pod that is too thin fails the check
    thin = base.with_changes(pod_chord_m=0.22, root_chord_m=0.2, tip_chord_m=0.15)
    assert not thin.payload_fit()["fits"]


def test_level_belly_landing_hits_skid_first(base):  # AC-13
    assert base.fin_ground_clearance_m > 0.015
    steep = base.with_changes(fin_droop_deg=90.0, fin_length_m=0.10)
    assert steep.fin_ground_clearance_m < 0


def test_volumes_are_consistent(base):
    v = base.volumes_m3()
    parts = v["pod"] + v["blend (both sides)"] + 2 * v["outer panel (each)"] + 2 * v["tip fin (each)"]
    assert v["total"] == pytest.approx(parts)
    assert all(x > 0 for x in v.values())


# --- AeroSandbox model and exports ---------------------------------------------

def test_airplane_matches_own_reference_values(base):
    ap = base.to_airplane()
    wing = ap.wings[0]
    assert wing.area("xy") == pytest.approx(base.reference_area_m2, rel=0.01)
    assert wing.mean_aerodynamic_chord() == pytest.approx(base.mac_m, rel=0.01)
    assert wing.span("y") == pytest.approx(base.span_m, rel=1e-3)  # ASB measures a twisted c/4 line
    assert ap.s_ref == pytest.approx(base.reference_area_m2)
    assert "elevon" in wing.get_control_surface_names()


def test_openvsp_and_avl_export(base, tmp_path):
    vsp = Path(base.export_openvsp(str(tmp_path / "manta.vspscript")))
    text = vsp.read_text()
    assert "AddGeom" in text and "WING" in text
    avl = Path(base.export_avl(str(tmp_path / "manta.avl")))
    assert "SURFACE" in avl.read_text()


def test_vlm_preview_baseline(base):
    pv = G.vlm_preview(base, REQ, 1.5)
    # Swept flying wing: neutral point aft of the quarter chord, positive lift slope
    assert 0.05 < pv.cl_alpha_per_deg < 0.09
    assert 25 < pv.neutral_point_pct_mac < 45
    # Reflex + washout give a nose-up zero-lift moment (self-trimming, AC-8)
    assert pv.cm0_c4 > 0
    # Tips stall last: at the wing CL_max estimate the tip strip has far more margin than the peak
    loc = pv.local_cl[G.WING_CL_MAX]
    assert loc[-1] < 0.8 * np.nanmax(loc)
    assert np.nanargmax(loc) < len(loc) // 2


def test_more_washout_more_nose_up_moment(base):
    lo = G.vlm_preview(base.with_changes(washout_deg=1.0), REQ, wing_cls=(0.95,))
    hi = G.vlm_preview(base.with_changes(washout_deg=6.0), REQ, wing_cls=(0.95,))
    assert hi.cm0_c4 > lo.cm0_c4
    assert hi.local_cl[0.95][-1] < lo.local_cl[0.95][-1]


def test_report_mentions_key_numbers(base):
    text = G.report(base, REQ, 1.5)
    assert "Span" in text and "fits" in text and "MAC" in text
