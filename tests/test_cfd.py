"""M2.1: CFD surface, case writing and force post-processing.

OpenFOAM itself is not needed: the surface is checked for closure and against
the geometry module, and the force reader on a synthetic forces file.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerosandbox")

from manta_mapper import cfd as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "vsp_runs/v2/manifest.json"


@pytest.fixture(scope="module")
def cond():
    return C.condition_from_manifest(MANIFEST, "F1")


@pytest.fixture(scope="module")
def surf(cond):
    return C.surface(cond.geometry())


def test_surface_is_closed_and_outward(surf):
    trimesh = pytest.importorskip("trimesh")
    v, f = surf
    m = trimesh.Trimesh(v, f, process=True)
    assert m.is_watertight and m.is_winding_consistent
    assert m.volume > 0  # outward normals


def test_surface_matches_geometry(cond, surf):
    g = cond.geometry()
    v, _ = surf
    # span to the fin tips, symmetric
    assert v[:, 1].max() == pytest.approx(g.overall_span_m / 2, abs=5e-3)  # fin thickness pokes out
    assert v[:, 1].min() == pytest.approx(-v[:, 1].max())
    # pod nose at the origin, trailing edge at the pod chord
    assert v[:, 0].min() == pytest.approx(0.0, abs=2e-3)
    assert v[:, 0].max() >= g.pod_chord_m - 1e-3


def test_sections_follow_vlm_stations(cond):
    g = cond.geometry()
    secs = C.sections(g)
    st = g.stations()
    for s, ref in zip(secs, st):
        assert np.allclose(s.le, ref["le"]) and s.chord == pytest.approx(ref["chord"])
        assert s.twist == pytest.approx(ref["twist"])
    assert secs[-1].twist == 0.0  # fin tip untwisted like the VLM fin


def test_reynolds_matches_vspaero_case(cond):
    man = json.loads(MANIFEST.read_text())
    assert cond.reynolds == pytest.approx(man["cases"][0]["re_cref"], rel=1e-9)


def test_write_case(tmp_path, cond):
    case = C.write_case(cond, tmp_path, "coarse", iterations=10, stl=None)
    for f in ("system/blockMeshDict", "system/snappyHexMeshDict", "system/controlDict", "0/U", "0/p",
              "constant/triSurface/aircraft.stl", "Allrun", "condition.json"):
        assert (case / f).exists(), f
    u = (case / "0/U").read_text()
    a = math.radians(cond.alpha_deg)
    assert f"{cond.velocity*math.cos(a):.6f} 0 {cond.velocity*math.sin(a):.6f}" in u
    assert f"CofR            ({cond.x_cg} 0 0)" in (case / "system/controlDict").read_text()


def test_read_forces_synthetic(tmp_path, cond):
    """Known lift, drag and moment on the half model come back as the whole-aircraft coefficients."""
    case = C.write_case(cond, tmp_path, "coarse", iterations=10)
    a = math.radians(cond.alpha_deg)
    q = 0.5 * cond.rho * cond.velocity**2
    CL, CDp, CDv, Cm = 0.40, 0.012, 0.009, -0.02
    lift = np.array([-math.sin(a), 0, math.cos(a)])
    drag = np.array([math.cos(a), 0, math.sin(a)])
    half = q * cond.s_ref / 2
    fp = half * (CL * lift + CDp * drag)
    fv = half * CDv * drag
    my = q * cond.s_ref * cond.c_ref / 2 * Cm
    d = case / "postProcessing/forces/0"
    d.mkdir(parents=True)
    rows_f = ["# Time total_x total_y total_z pressure_x pressure_y pressure_z viscous_x viscous_y viscous_z"]
    rows_m = ["# Time total_x total_y total_z pressure_x pressure_y pressure_z viscous_x viscous_y viscous_z"]
    for it in range(1, 11):
        t = fp + fv
        rows_f.append(f"{it} {t[0]} {t[1]} {t[2]} {fp[0]} {fp[1]} {fp[2]} {fv[0]} {fv[1]} {fv[2]}")
        rows_m.append(f"{it} 0 {my} 0 0 {my} 0 0 0 0")
    (d / "force.dat").write_text("\n".join(rows_f) + "\n")
    (d / "moment.dat").write_text("\n".join(rows_m) + "\n")
    r = C.read_forces(case, average_last=5)
    assert r["CL"] == pytest.approx(CL)
    assert r["CD"] == pytest.approx(CDp + CDv)
    assert r["CD_friction"] == pytest.approx(CDv)
    assert r["Cm"] == pytest.approx(Cm)


def test_neutral_point():
    """Cm = -0.1 (CL - 0.2) about x_ref: NP is 0.1 c behind x_ref."""
    pts = [dict(alpha_deg=a, CL=0.08 * a, Cm=-0.1 * (0.08 * a - 0.2)) for a in (0, 2, 4, 6, 8)]
    r = C.neutral_point(pts, c_ref=0.3, x_ref=0.25, mac_le_x=0.2)
    assert r["x_np"] == pytest.approx(0.28)
    assert r["np_pct_mac"] == pytest.approx(100 * 0.08 / 0.3)
