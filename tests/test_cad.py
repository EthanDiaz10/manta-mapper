"""M2.3 scripted CAD: layout, recipe, templates and (if manifold3d is installed) the built parts."""

import json
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from manta_mapper import cad
from manta_mapper import cad_templates as T
from manta_mapper.structures import load_geometry

FUSION_SCRIPT = Path(__file__).resolve().parents[1] / "cad" / "fusion" / "MantaMapperCAD" / "MantaMapperCAD.py"


@pytest.fixture(scope="module")
def geom():
    return load_geometry()


@pytest.fixture(scope="module")
def lay(geom):
    return cad.layout(geom)


@pytest.fixture(scope="module")
def recipe(lay):
    return cad.build_recipe(lay)


# --- sections and offsets ---------------------------------------------------------


def test_offset_loop_shrinks_a_circle_by_d():
    t = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    P = np.column_stack([10 * np.cos(t), 10 * np.sin(t)])  # counter-clockwise
    r = np.linalg.norm(cad.offset_loop(P, 1.5), axis=1)
    assert r == pytest.approx(8.5, abs=0.01)
    assert np.linalg.norm(cad.offset_loop(P, -1.0), axis=1) == pytest.approx(11.0, abs=0.01)


def test_pod_stations_match_the_geometry(lay, geom):
    pod = lay.pod
    s0, s1 = pod.station(0.0), pod.station(geom.root_y_m * 1000)
    assert (s0.le_x, s0.chord, s0.blend) == (0.0, pytest.approx(geom.pod_chord_m * 1000), 0.0)
    assert s1.le_x == pytest.approx(geom.root_le_x_m * 1000)
    assert s1.chord == pytest.approx(geom.root_chord_m * 1000)
    assert s1.blend == 1.0
    # trailing edges line up (straight TE for the pusher and hinge line)
    for y in np.linspace(0, geom.root_y_m * 1000, 7):
        s = pod.station(y)
        assert s.le_x + s.chord == pytest.approx(geom.pod_chord_m * 1000)


def test_inside_surface_sits_inside_the_skin(lay):
    pod = lay.pod
    for y in (0.0, pod.w + 30, pod.root_y - 5):
        out = pod.loop2d(y)
        ins = pod.loop2d(y, lay.st.wall_mm)
        assert len(ins) == len(out)
        for x, z in ins[5:-5]:
            zo = pod.z_out(x, y)
            assert zo is not None and zo[0] < z < zo[1]


def test_pod_loft_sections_are_compatible(recipe):
    for name, d in recipe.solids.items():
        if d["op"] == "loft":
            n = {len(s) for s in d["sections"]}
            assert len(n) == 1, name


# --- layout ---------------------------------------------------------------------


def test_every_check_passes(lay):
    bad = [f"{c.name}: {c.value:.2f} vs {c.sense} {c.limit}" for c in lay.checks if not c.ok]
    assert not bad, bad


def test_pod_is_split_into_four_printable_halves(lay):
    use = cad._usable(lay.st)
    assert len(lay.cuts_x) == 1
    assert len(lay.sections) == 4
    for s in lay.sections:
        assert np.all(np.sort(s["bbox"]) <= use)


def test_cuts_avoid_the_keepouts(lay):
    for c in lay.cuts_x:
        for a, b, name in cad.keepouts_x(lay):
            assert not a < c < b, name


def test_printer_size_drives_the_cuts(geom):
    huge = cad.layout(geom, replace(cad.CadSettings(), printer_mm=(600, 600, 600)))
    assert huge.cuts_x == []
    # the ribs, sleeves, strips and nacelle leave no legal cut aft of x ~ 170 mm, so a much smaller
    # printer can't print the rear section: the layout says so instead of cutting through a rib
    with pytest.raises(ValueError, match="no set of chordwise cuts"):
        cad.layout(geom, replace(cad.CadSettings(), printer_mm=(220, 220, 220)))


def test_spar_line_and_joint_follow_m22(lay):
    # the sleeve's inner end (spar end) lands where structures.py put the carry-through
    c = lay.inner_rib["centre"]
    assert c[0] == pytest.approx(205, abs=3)
    assert c[1] == pytest.approx(74, abs=2)
    assert lay.strip["x"] == pytest.approx(c[0])
    assert lay.strip["half"] - c[1] >= lay.st.strip_bond_mm - 1e-6 or lay.strip["half"] == pytest.approx(
        lay.pod.root_y - lay.st.end_plate_mm - 1)
    sweep = math.degrees(math.atan2(lay.spar_dir[0], lay.spar_dir[1]))
    assert sweep == pytest.approx(13.8, abs=0.3)
    top, bot, _ = cad.spar_cover(lay)
    assert min(top, bot) >= 2.0
    # spar sits halfway between the skins at the root (not on the chord line)
    zo = lay.pod.z_out(lay.spar_root[0], lay.pod.root_y)
    assert lay.spar_root[2] == pytest.approx(0.5 * (zo[0] + zo[1]), abs=0.6)


def test_hatches_miss_the_roof_strip_and_the_cut(lay):
    sx, w = lay.strip["x"], lay.st.strip_w_mm
    for h in lay.hatches:
        assert h["x1"] < sx - w / 2 or h["x0"] > sx + w / 2
        for c in lay.cuts_x:
            assert not h["x0"] < c < h["x1"]


def test_payload_packing_is_clear_and_holds_the_cg(lay, geom):
    p = {b.name: b for b in lay.payload}
    assert set(p) >= {"battery", "camera + mount", "flight controller", "Pi Zero 2 W", "ESC"}
    cam = p["camera + mount"]
    assert 0.5 * (cam.lo[1] + cam.hi[1]) == pytest.approx(0.0)
    names = list(p)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not p[a].overlaps(p[b]), (a, b)
    # moving items and the battery keeps the CG where weights.py put it
    moment = sum(s.mass_g * (0.5 * (p[s.name].lo[0] + p[s.name].hi[0]) - s.pref[0] * lay.pod.c0) for s in cad.PAYLOAD)
    bat = lay.battery
    assert moment + bat["mass_g"] * (bat["x"] - bat["x_weights"]) == pytest.approx(0, abs=0.6 * bat["mass_g"])


def test_battery_tray_spans_the_travel_with_stop_slots(lay):
    tr, b, st = lay.tray, lay.battery, lay.st
    assert tr["x0"] <= b["x_min"] - b["L"] / 2 - st.stop_len_mm + 1e-6
    assert tr["x1"] >= b["x_max"] + b["L"] / 2 + st.stop_len_mm - 1e-6
    assert np.allclose(np.diff(tr["slots"]), st.stop_pitch_mm)


def test_camera_window_and_skid(lay):
    cam = lay.camera
    assert cam["axis"][0] < 0 and cam["axis"][2] < 0  # looks down and forward
    assert math.degrees(math.atan2(-cam["axis"][0], -cam["axis"][2])) == pytest.approx(6.0)
    w = cam["window"]
    assert w["y"] < lay.skid["y0"]  # window between the rails
    assert w["x0"] < cam["lens_tip"][0] < w["x1"]


def test_fin_root_template_is_the_tip_section_seen_square_to_the_fin(lay, geom):
    root2d, tip2d, fr = cad.fin_template_loops(geom, 49)
    tip_sec = cad.outer_section3d(geom, 1.0, 49)
    t_tip = tip_sec[:, 2].max() - tip_sec[:, 2].min()
    t_tpl = root2d[:, 1].max() - root2d[:, 1].min()
    # vertical thickness seen in a plane tilted by the droop (washout makes it slightly inexact)
    assert t_tpl == pytest.approx(t_tip * math.cos(math.radians(geom.fin_droop_deg)), rel=0.12)
    assert tip2d[:, 0].max() == pytest.approx(geom.tip_chord_m * geom.fin_taper * 1000, rel=0.01)


# --- recipe -----------------------------------------------------------------------


def test_recipe_only_uses_ops_the_fusion_script_knows(recipe):
    src = FUSION_SCRIPT.read_text()
    known = set(re.findall(r'op == "(\w+)"', src)) | set(re.findall(r'op in \(([^)]*)\)', src)[0].replace('"', "").replace(" ", "").split(","))
    used = {d["op"] for d in recipe.solids.values()}
    assert used <= known, used - known
    assert used <= set(cad.Recipe.OPS)


def test_fusion_script_compiles():
    compile(FUSION_SCRIPT.read_text(), str(FUSION_SCRIPT), "exec")


def test_recipe_is_a_valid_dag_and_serialises(lay, recipe):
    order = recipe.order()
    pos = {n: i for i, n in enumerate(order)}
    for n in order:
        d = recipe.solids[n]
        for r in ([d["of"]] if "of" in d else []) + (d.get("items") or []):
            assert pos[r] < pos[n]
    data = json.loads(json.dumps(cad.to_json(lay, recipe)))
    assert data["units"] == "mm"
    printed = [p["name"] for p in data["recipe"]["parts"] if p["printed"]]
    for need in ("pod_front_L", "pod_front_R", "pod_rear_L", "pod_rear_R", "nose_cap", "nacelle",
                 "hatch_front_lid", "hatch_rear_lid", "camera_mount", "battery_stop", "template_panel_root"):
        assert need in printed


def test_recipe_rejects_bad_input():
    R = cad.Recipe()
    R.box("a", (0, 0, 0), (1, 1, 1))
    with pytest.raises(KeyError):
        R.box("a", (0, 0, 0), (1, 1, 1))
    with pytest.raises(KeyError):
        R.union("u", ["a", "missing"])
    with pytest.raises(ValueError):
        R.box("b", (0, 0, 0), (0, 1, 1))
    with pytest.raises(ValueError):
        R.loft("l", [np.zeros((3, 3)), np.zeros((4, 3))])


# --- templates ------------------------------------------------------------------------


def test_dxf_templates(lay, tmp_path):
    files = T.write_all(lay, tmp_path)
    names = {f.name for f in files}
    assert names == {"panel_root_template.dxf", "panel_tip_template.dxf", "fin_root_template.dxf",
                     "fin_tip_template.dxf", "panel_plan_R.dxf", "skid_rail.dxf"}
    root = T.read_dxf_entities((tmp_path / "panel_root_template.dxf").read_text())
    assert ("POLYLINE", "CUT") in root and ("POLYLINE", "PROFILE") in root
    assert root.count(("CIRCLE", "CUT")) == 2  # spar + pin
    assert (tmp_path / "panel_root_template.dxf").read_text().rstrip().endswith("EOF")


def test_template_holes_fit_the_stepped_spar(lay):
    loops = {n: (loop, d, ex) for n, loop, d, ex in cad.template_loops(lay)}
    assert loops["panel_root"][1] == pytest.approx(12 + lay.st.spar_hole_clear_mm)
    assert loops["panel_tip"][1] == pytest.approx(8 + lay.st.spar_hole_clear_mm)
    # the spar hole is inside the section with room to spare at both ends
    for n in ("panel_root", "panel_tip"):
        loop, d, _ = loops[n]
        m = (len(loop) + 1) // 2
        up, lo = loop[:m][::-1], loop[m - 1:]
        assert np.interp(0, up[:, 0], up[:, 1]) - d / 2 > 1.5
        assert -np.interp(0, lo[:, 0], lo[:, 1]) - d / 2 > 1.5


# --- built parts (manifold3d) -------------------------------------------------------------


@pytest.fixture(scope="module")
def built(recipe):
    pytest.importorskip("manifold3d")
    from manta_mapper import cad_mesh

    return cad_mesh.build(recipe)


def test_printed_parts_are_single_closed_solids_that_fit(recipe, built, lay):
    from manta_mapper import cad_mesh

    stats = cad_mesh.part_stats(recipe, built, lay.st)
    assert len(stats) == sum(1 for p in recipe.parts if p["printed"])
    for name, s in stats.items():
        assert s["watertight"], name
        assert s["pieces"] == 1, name
        assert s["fits"], name
        assert s["volume_cm3"] > 0.5, name


def test_pod_halves_are_near_mirror_images(recipe, built, lay):
    from manta_mapper import cad_mesh

    s = cad_mesh.part_stats(recipe, built, lay.st)
    for sec in ("front", "rear"):
        L, R = s[f"pod_{sec}_L"]["volume_cm3"], s[f"pod_{sec}_R"]["volume_cm3"]
        assert R == pytest.approx(L, rel=0.08)  # the right half carries the centreline spigot


def test_printed_mass_close_to_the_weights_estimate(recipe, built, lay):
    from manta_mapper import cad_mesh

    s = cad_mesh.part_stats(recipe, built, lay.st)
    fb = cad.mass_feedback(lay, s)
    # the CAD should land near the M1.5 estimate, and the battery must still be able to hold the CG
    assert fb["delta_g"] == pytest.approx(0, abs=100)
    assert fb["within_travel"]


def test_stl_export_roundtrip(recipe, built, tmp_path):
    from manta_mapper import cad_mesh

    files = cad_mesh.export_stl(recipe, built, tmp_path)
    f = next(p for p in files if p.name == "nose_cap.stl")
    data = f.read_bytes()
    n = int(np.frombuffer(data[80:84], dtype=np.uint32)[0])
    assert len(data) == 84 + 50 * n
