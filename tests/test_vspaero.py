"""M1.7: VSPAERO script generation, output parsing and sign conventions.

OpenVSP itself is not needed: the parsers and the convention logic are checked
on synthetic VSPAERO files built from known derivatives.
"""

import json
import math
from pathlib import Path

import pytest

from manta_mapper import vspaero as V

ROOT = Path(__file__).resolve().parents[1]

BODY = dict(cy_b=-0.08, cl_b=-0.09, cn_b=0.006, cy_p=0.01, cl_p=-0.42, cn_p=-0.03,
            cy_r=0.02, cl_r=0.12, cn_r=-0.012, cl_a=4.3, cm_a=-0.3, cl_q=4.0, cm_q=-1.5)


def write_stab(path, body, moment_sign=-1, rate_sign=-1, beta_sign=1):
    """A .stab file as VSPAERO would write it for ``body`` (standard-frame body derivatives)."""
    s_m, s_r, s_b = moment_sign, rate_sign, beta_sign
    rows = {
        "CFx": [0.0] * 6,
        "CFy": [0.0, 0.0, s_b * body["cy_b"], s_r * body["cy_p"], 0.0, s_r * body["cy_r"]],
        "CFz": [0.0] * 6,
        "CMx": [0.0, 0.0, s_m * s_b * body["cl_b"], s_m * s_r * body["cl_p"], 0.0, s_m * s_r * body["cl_r"]],
        "CMy": [0.0, body["cm_a"], 0.0, 0.0, s_r * body["cm_q"], 0.0],
        "CMz": [0.0, 0.0, s_m * s_b * body["cn_b"], s_m * s_r * body["cn_p"], 0.0, s_m * s_r * body["cn_r"]],
        "CL": [0.5, body["cl_a"], 0.0, 0.0, s_r * body["cl_q"], 0.0],
        "CD": [0.01] + [0.0] * 5,
        "CS": [0.0] * 6, "CMl": [0.0] * 6, "CMm": [0.0] * 6, "CMn": [0.0] * 6,
    }
    lines = ["# Name   Value   Units", "Sref_    0.4  Lunit^2", "AoA_     5.0  deg", "#",
             "#   Coef   Total   Alpha   Beta   p   q   r   Mach   U",
             "#   -      -       per     per    per per per per    per",
             "#   -      -       rad     rad    rad rad rad M      u", "#"]
    for k, v in rows.items():
        lines.append(f"{k:6s} " + " ".join(f"{x: .8f}" for x in v + [0.0, 0.0]))
    lines += ["#", "# Result  Value  Units", "SM  0.07  no_unit", "X_np  0.29  Lunit"]
    Path(path).write_text("\n".join(lines) + "\n")


def test_parse_stab_reads_rows_and_scalars(tmp_path):
    p = tmp_path / "a.stab"
    write_stab(p, BODY)
    st = V.parse_stab(p)
    assert st.d("CL", "alpha") == pytest.approx(4.3)
    assert st.d("CMx", "p") == pytest.approx(-0.42)  # two sign flips cancel
    assert st.scalars["X_np"] == pytest.approx(0.29)


@pytest.mark.parametrize("ms,rs,bs", [(-1, -1, 1), (1, 1, 1), (-1, 1, -1), (1, -1, 1)])
def test_conventions_recovered_whatever_vspaero_uses(tmp_path, ms, rs, bs):
    p = tmp_path / "a.stab"
    write_stab(p, BODY, ms, rs, bs)
    st = V.parse_stab(p)
    c = V.conventions(st, BODY, 5.0)
    assert (c.moment_sign, c.rate_sign, c.beta_sign) == (ms, rs, bs)
    assert c.rates_axes == "body"
    d = V.vsp_derivs(st, c, 5.0)
    ref = V.asb_derivs(BODY, 5.0)
    for k in V.DerivSet.LATERAL + ("cm_q", "cl_a"):
        assert getattr(d, k) == pytest.approx(getattr(ref, k), abs=1e-9)


def test_stability_axes_rotation_is_identity_at_zero_alpha():
    s = V.to_stability_axes(BODY, 0.0)
    for k in V.DerivSet.LATERAL:
        assert s[k] == pytest.approx(BODY[k])


def test_stability_axes_rotation_matches_textbook():
    """Etkin: Cn_r(stab) = Cn_r c^2 - (Cl_r + Cn_p) s c + Cl_p s^2 (body values)."""
    a = math.radians(6.0)
    c, s = math.cos(a), math.sin(a)
    st = V.to_stability_axes(BODY, 6.0)
    expect = BODY["cn_r"] * c * c - (BODY["cl_r"] + BODY["cn_p"]) * s * c + BODY["cl_p"] * s * s
    assert st["cn_r"] == pytest.approx(expect)
    assert st["cl_b"] == pytest.approx(BODY["cl_b"] * c + BODY["cn_b"] * s)


def test_parse_polar_handles_new_and_old_headers(tmp_path):
    new = tmp_path / "n.polar"
    new.write_text("Beta Mach AoA Re/1e6 CLo CLi CLtot CDo CDi CDtot CMytot\n"
                   "0 0.04 0.0 0.2 0 0.1 0.10 0 0.002 0.002 0.01\n"
                   "0 0.04 2.0 0.2 0 0.25 0.25 0 0.004 0.004 -0.01\n")
    d = V.parse_polar(new)
    assert list(d["alpha"]) == [0.0, 2.0] and d["CL"][1] == pytest.approx(0.25)
    old = tmp_path / "o.polar"
    old.write_text("Beta Mach AoA Re/1e6 CL CDo CDi CDtot CS L/D E CFx CFy CFz CMx CMy CMz\n"
                   "0 0.04 4.0 0.2 0.4 0 0.006 0.006 0 66 0.9 0 0 0 0 -0.02 0\n")
    d = V.parse_polar(old)
    assert d["CMy"][0] == pytest.approx(-0.02)


def test_scripts_cover_every_run(tmp_path):
    pytest.importorskip("aerosandbox")
    from manta_mapper.requirements import MissionRequirements

    req = MissionRequirements.from_toml(ROOT / "config" / "mission.toml")
    res = json.loads((ROOT / "docs" / "optimizer_result.json").read_text())
    cases = V.build_cases(req, res, n=1)
    man = V.prepare(cases, tmp_path)
    assert len(man["runs"]) == 6  # vlm/panel x sweep/stab/beta
    text = (tmp_path / "F1_vlm_stab.vspscript").read_text()
    assert "STABILITY_DEFAULT" in text and "Sref" in text and "void main()" in text
    assert text.count("AddGeom") == 2  # wing and tip fins
    assert 0 < cases[0].alpha_cruise_deg < 8


def test_rates_turn_about_the_cg_not_the_nose():
    """Regression for the bug M1.7 found: a pitch rate must not change CL by turning about the pod nose."""
    pytest.importorskip("aerosandbox")
    from manta_mapper import aero as A
    from manta_mapper import geometry as G

    g = G.baseline()
    fl = A.Flight(14.0, 1320.0)
    V_, c = 14.0, g.mac_m
    q = 0.01 * 2 * V_ / c
    out = []
    for x_ref in (0.20, 0.30):
        b = A.run_vlm(g, fl, 4.0, xyz_ref=(x_ref, 0, 0))
        r = A.run_vlm(g, fl, 4.0, q=q, xyz_ref=(x_ref, 0, 0))
        out.append((r["CL"] - b["CL"]) / 0.01)
    # CL_q about the CG is a few per unit q-hat and moves only modestly with the reference point;
    # about the nose it was ~13
    assert all(2.0 < v < 7.5 for v in out)


def test_vspaero_vlm_results_parse_and_pass_sanity(tmp_path):
    runs = ROOT / "vsp_runs"
    if not (runs / "F1_vlm_sweep.polar").exists():
        pytest.skip("no VSPAERO results checked in")
    sw = V.parse_polar(runs / "F1_vlm_sweep.polar")
    st = V.parse_stab(runs / "F1_vlm_stab.stab")
    assert V.solution_problems(sw, st) == ""
    bad = V.parse_polar(runs / "F1_panel_sweep.polar")
    assert V.solution_problems(bad, None) != ""  # the broken thick mesh is caught
