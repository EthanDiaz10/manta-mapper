"""M2.4 test prints: coupons, fit gauges, check sheet and weigh-in log.

Before ~15 h of LW-ASA goes into the four pod sections, a few short prints
prove that the printer, the filament and the CAD's clearances agree:

* **calibration**: a 40 mm skin box (0.8 mm walls, like the pod skin) that
  gives the real wall thickness and the real foamed density;
* **gauges**: a fit gauge (tube bore, magnet pockets, pin and M3 holes, five
  sizes each, the CAD's size in the middle) and a joint-gap ladder (one
  socket, five spigots), so a misfit says *how much* to change, not just that
  it is wrong;
* **coupons**: short slices cut from the real pod parts by the same recipe
  (both joints, the root face with its sleeve bore, magnets and pin, the
  inner rib with its strip slots, the nose ring for the TPU cap).

The check sheet (``fit_checks.csv``) and weigh-in log (``weigh_in.csv``) are
plain CSV files to fill in by hand. ``check`` reads them back and prints what
passes, what to change in ``CadSettings``, the foamed density, a forecast of
the printed parts at that density, the aircraft mass against the budget and
where the battery has to sit for the CG.

    python -m manta_mapper.testprints generate --out cad/testprints --render docs/img/testprints_coupons.png
    python -m manta_mapper.testprints check --log cad/testprints --report docs/testprints_report.txt

``generate`` needs manifold3d; ``check`` needs only the two CSV files.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cad import CadSettings

# --------------------------------------------------------------------------
# Numbers the checks are judged against
# --------------------------------------------------------------------------

LW_ASA_DENSITY_ASSUMED = 0.70  # g/cm^3, weights.py and cad_mesh
LW_ASA_DENSITY_OK = 0.80  # above this the pod gets heavy: lower the flow or raise the temperature
LW_ASA_DENSITY_MAX = 0.95  # barely foaming
OPTIMIZER_CAP_G = 1275.0  # 85 % of the 1.5 kg design mass (M1.6)
DESIGN_MASS_G = 1500.0  # every aero and stability check was run at this mass
AMBER_LIMIT_G = 0.95 * DESIGN_MASS_G
CG_TARGET_MM = 261.0
CG_BAND_MM = (261.0, 275.0)  # M1.4 / M2.1 (static margin 5-10 %)
PART_HEAVY = 0.10  # a printed part more than 10 % over its CAD mass gets flagged
GROWTH_PRINTED = 0.20  # weights.GROWTH["structure"]: what an unmeasured printed part is budgeted at

# Fit gauge ladders: five sizes, tightest on the left, the CAD's size in the middle.
# (name, nominal size, step, what goes in, CadSettings field and how a size maps to it)
LADDER_STEPS = (-2, -1, 0, 1, 2)


def ladders(st: CadSettings) -> list[dict]:
    """Hole diameters on the fit gauge, from the CAD's own clearances."""
    out = [
        dict(key="tube", what="14 mm sleeve tube (real one)", d0=st.sleeve_od_mm + 2 * st.sleeve_bond_gap_mm, step=0.10,
             base=st.sleeve_od_mm, setting="sleeve_bond_gap_mm", depth=None),
        dict(key="magnet", what="10x3 mm panel magnet", d0=st.magnet_d_mm + 2 * st.pocket_clear_mm, step=0.10,
             base=st.magnet_d_mm, setting="pocket_clear_mm", depth=st.magnet_t_mm + 0.2),
        dict(key="hmagnet", what="6x2 mm hatch magnet", d0=st.hatch_magnet_d_mm + 2 * st.pocket_clear_mm, step=0.10,
             base=st.hatch_magnet_d_mm, setting="pocket_clear_mm", depth=st.hatch_magnet_t_mm + 0.2),
        dict(key="pin", what="3 mm carbon pin", d0=st.pin_d_mm + 2 * st.pin_clear_mm, step=0.10, base=st.pin_d_mm,
             setting="pin_clear_mm", depth=None),
        dict(key="m3", what="M3 bolt", d0=st.bolt_hole_d_mm, step=0.20, base=None, setting="bolt_hole_d_mm", depth=None),
    ]
    for L in out:
        L["sizes"] = [round(L["d0"] + k * L["step"], 3) for k in LADDER_STEPS]
    return out


def joint_ladder(st: CadSettings) -> list[float]:
    """Glue gaps per side on the five spigots of the joint ladder (the CAD's in the middle)."""
    return [round(max(st.joint_clear_mm + k * 0.05, 0.0), 3) for k in LADDER_STEPS]


# --------------------------------------------------------------------------
# Geometry (manifold3d)
# --------------------------------------------------------------------------


def _m3():
    import manifold3d as m3

    return m3


def _box(lo, hi):
    m3 = _m3()
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return m3.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))


def _cyl_y(x, z, y0, y1, d, seg=64):
    """Cylinder along +y (a horizontal hole when the gauge stands on its edge)."""
    m3 = _m3()
    c = m3.Manifold.cylinder(y1 - y0, d / 2, d / 2, seg).rotate((-90.0, 0.0, 0.0))  # +z -> +y
    return c.translate((x, y0, z))


def _cyl_z(x, y, z0, z1, d, seg=32):
    m3 = _m3()
    return m3.Manifold.cylinder(z1 - z0, d / 2, d / 2, seg).translate((x, y, z0))


def fit_gauge(st: CadSettings):
    """Plate 10 mm thick that prints standing on its long edge, so every hole is horizontal like
    the holes in the pod (which prints cut face down). Tightest hole on the left of each ladder;
    the CAD's size (middle) has a dimple under it. The top-left corner is chamfered."""
    m3 = _m3()
    L = {d["key"]: d for d in ladders(st)}
    T = 10.0  # thickness along the hole axis (y)
    W, H = 124.0, 58.0
    plate = _box((0, 0, 0), (W, T, H))
    plate = plate - _box((-1, -1, H - 5), (4, T + 1, H + 1))  # orientation notch, top left
    holes, marks = [], []

    def row(z, specs):
        x = 6.0
        for key, pitch in specs:
            d = L[key]
            for i, size in enumerate(d["sizes"]):
                cx = x + size / 2
                if d["depth"]:  # blind pocket from the front face (y = 0)
                    holes.append(_cyl_y(cx, z, -1.0, d["depth"], size))
                else:
                    holes.append(_cyl_y(cx, z, -1.0, T + 1.0, size))
                if i == 2:  # dimple under the CAD's size
                    marks.append(_cyl_y(cx, z - size / 2 - 2.5, -1.0, 0.6, 1.4, 16))
                x += size + pitch
            x += 4.0

    row(44.0, [("tube", 7.0)])
    row(25.0, [("magnet", 3.5), ("hmagnet", 3.0)])
    row(9.0, [("pin", 3.5), ("m3", 3.5)])
    return plate - m3.Manifold.batch_boolean(holes + marks, m3.OpType.Add)


def joint_ladder_parts(st: CadSettings) -> dict:
    """One socket (a 40 x 20 mm rectangle in pod skin, 0.8 mm) and five spigot bands (1.6 mm),
    each a glue gap per side from ``joint_ladder``. Spigot k has k dots on its cap (1 = tightest).
    Both print open end up, walls vertical, like the pod's joints."""
    m3 = _m3()
    w, h, skin, band = 40.0, 20.0, st.wall_mm, st.joint_band_mm
    lap, floor = st.joint_lap_mm, 1.2
    sock = _box((-skin, -skin, 0), (w + skin, h + skin, lap + 2 + floor)) - _box((0, 0, floor), (w, h, 99))
    out = {"joint_socket": sock}
    for k, g in enumerate(joint_ladder(st), start=1):
        ow, oh = w - 2 * g, h - 2 * g
        cup = _box((0, 0, 0), (ow, oh, lap + 4 + floor)) - _box((band, band, floor), (ow - band, oh - band, 99))
        dots = [_cyl_z(6 + 4 * j, oh / 2, -1, 0.6, 2.0) for j in range(k)]
        cup = cup - m3.Manifold.batch_boolean(dots, m3.OpType.Add)
        out[f"joint_spigot_{k}"] = cup.translate((g, g, 0))
    return out


def skin_cal(st: CadSettings):
    """40 mm open box, walls and floor one skin thick (0.8 mm = two 0.4 mm perimeters)."""
    s = st.wall_mm
    return _box((0, 0, 0), (40, 40, 40)) - _box((s, s, s), (40 - s, 40 - s, 99))


@dataclass
class Coupon:
    name: str
    material: str
    stage: str
    checks: str
    orient: str
    solids: dict = field(default_factory=dict)  # file name -> manifold


def coupons(recipe, built: dict, st: CadSettings) -> list[Coupon]:
    """Short slices of the real pod parts, cut by boxes from the same recipe."""
    def clip(part, lo, hi):
        return built[part] ^ _box(lo, hi)

    B = 999.0
    out = [
        Coupon("C0_skin_cal", "LW-ASA", "1 calibrate", "wall thickness, outside size, foamed density",
               "floor down, 2 perimeters, 0 % infill, no top", {"C0_skin_cal": skin_cal(st)}),
        Coupon("G1_fit_gauge", "LW-ASA", "2 gauges", "tube bore, magnet pockets, pin and M3 holes (5 sizes each)",
               "standing on its long bottom edge (holes horizontal), brim", {"G1_fit_gauge": fit_gauge(st)}),
        Coupon("G2_joint_ladder", "LW-ASA", "2 gauges", "spigot glue gap 0.00-0.20 mm per side",
               "open end up", joint_ladder_parts(st)),
        Coupon("C1_joint_cross", "LW-ASA", "3 coupons",
               "chordwise spigot joint all round the section (both halves), centreline butt faces, the hatch's aft rim",
               "x = 173 mm cut face down (the rear pieces stand on their spigot band), like the pod sections",
               {f"C1_{p}": clip(p, (155, -B, -B), (180, B, B)) for p in
                ("pod_front_L", "pod_front_R", "pod_rear_L", "pod_rear_R")}),
        Coupon("C2_root_face_R", "LW-ASA", "3 coupons",
               "root rib boss and 14 mm sleeve bore, both panel magnet pockets, pin hole, root face flatness, boss bond push-out",
               "x = 173 mm face down, root face vertical (brim: it is tall and thin)",
               {"C2_root_face_R": clip("pod_rear_R", (173, 148, -B), (330, B, B))}),
        Coupon("C3_inner_rib_R", "LW-ASA", "3 coupons", "inner rib boss bore, 10x1 strip slots in floor and roof",
               "x = 178 mm face down", {"C3_inner_rib_R": clip("pod_rear_R", (178, 62, -B), (232, 88, B))}),
        Coupon("C4_nose_ring", "LW-ASA", "3 coupons", "TPU nose cap push fit over the nose spigot",
               "x = 42 mm face down (spigot up)",
               {f"C4_{p}": clip(p, (17, -B, -B), (42, B, B)) for p in ("pod_front_L", "pod_front_R")}),
        Coupon("C5_centre_floor", "LW-ASA", "3 coupons", "centreline spigot band (it runs along the front floor)",
               "flat, outside down (this is the only coupon not printed like its parent: it checks the band's width)",
               {f"C5_{p}": clip(p, (40, -25, -B), (100, 25, 0)) for p in ("pod_front_L", "pod_front_R")}),
    ]
    return out


def coupon_stats(cps: list[Coupon], density: float = LW_ASA_DENSITY_ASSUMED) -> dict:
    out = {}
    for c in cps:
        for n, m in c.solids.items():
            bb = m.bounding_box()
            out[n] = dict(coupon=c.name, volume_cm3=m.volume() / 1000.0, mass_g=m.volume() / 1000.0 * density,
                          bbox=(np.array(bb[3:]) - np.array(bb[:3])).tolist(), pieces=len(m.decompose()),
                          closed=m.status().name == "NoError" and m.num_tri() > 0)
    return out


# --------------------------------------------------------------------------
# Check sheet
# --------------------------------------------------------------------------

# kind: "range" (number between lo and hi), "yn" (y = pass), "ladder:<key>" (the size that fits, from the gauge),
# "joint" (the spigot number that slides home), "min" (number >= lo)
CHECK_FIELDS = ("id", "stage", "part", "check", "how", "kind", "nominal", "lo", "hi", "unit", "measured", "notes")


def check_rows(st: CadSettings) -> list[dict]:
    L = {d["key"]: d for d in ladders(st)}
    jl = joint_ladder(st)

    def r(id, stage, part, check, how, kind="yn", nominal="", lo="", hi="", unit=""):
        return dict(id=id, stage=stage, part=part, check=check, how=how, kind=kind, nominal=nominal, lo=lo, hi=hi,
                    unit=unit, measured="", notes="")

    def lad(key):
        s = L[key]["sizes"]
        return " / ".join(f"{v:.2f}" for v in s)

    return [
        r("CAL-1", "1 calibrate", "C0_skin_cal", "wall thickness, average of 4 sides at mid-height",
          "calipers", "range", st.wall_mm, 0.70, 0.95, "mm"),
        r("CAL-2", "1 calibrate", "C0_skin_cal", "outside size (foam swell)", "calipers, both directions, average",
          "range", 40.0, 39.6, 40.5, "mm"),
        r("CAL-3", "1 calibrate", "C0_skin_cal", "walls fused: no gaps between the two perimeters, squeeze doesn't split it",
          "look and squeeze"),
        r("CAL-4", "1 calibrate", "C0_skin_cal", "no warping: floor flat on the glass", "0.2 mm feeler / paper under a corner"),
        r("G1-OD", "2 gauges", "14 mm sleeve tube", "real tube outside diameter", "calipers, 3 places",
          "range", st.sleeve_od_mm, st.sleeve_od_mm - 0.10, st.sleeve_od_mm + 0.10, "mm"),
        r("G1-TUBE", "2 gauges", "G1_fit_gauge", f"smallest bore the real tube slides through by hand ({lad('tube')})",
          "enter the hole size", "ladder:tube", L["tube"]["d0"], unit="mm"),
        r("G1-MAG", "2 gauges", "G1_fit_gauge", f"smallest pocket the 10x3 magnet presses into with a thumb ({lad('magnet')})",
          "enter the pocket size", "ladder:magnet", L["magnet"]["d0"], unit="mm"),
        r("G1-HMAG", "2 gauges", "G1_fit_gauge", f"smallest pocket the 6x2 magnet presses into ({lad('hmagnet')})",
          "enter the pocket size", "ladder:hmagnet", L["hmagnet"]["d0"], unit="mm"),
        r("G1-PIN", "2 gauges", "G1_fit_gauge", f"smallest hole the 3 mm rod slides through ({lad('pin')})",
          "enter the hole size", "ladder:pin", L["pin"]["d0"], unit="mm"),
        r("G1-M3", "2 gauges", "G1_fit_gauge", f"smallest hole an M3 bolt drops through ({lad('m3')})",
          "enter the hole size", "ladder:m3", L["m3"]["d0"], unit="mm"),
        r("G2-JOINT", "2 gauges", "G2_joint_ladder",
          "tightest spigot that slides fully home by hand with no rocking (dots 1-5 = " +
          " / ".join(f"{g:.2f}" for g in jl) + " mm per side)", "enter the dot count", "joint", 3, unit="spigot"),
        r("C1-A", "3 coupons", "C1_joint_cross", "chordwise spigot slides fully home (8 mm) by hand, dry", "dry fit"),
        r("C1-B", "3 coupons", "C5_centre_floor", "centreline spigot band slides under the left floor by hand, dry", "dry fit"),
        r("C1-C", "3 coupons", "C1_joint_cross", "step between outer skins at the seams", "calipers depth / feel",
          "range", 0.0, 0.0, 0.3, "mm"),
        r("C1-D", "3 coupons", "C1_joint_cross", "all four rings assemble at once without forcing (the centreline faces meet)",
          "dry fit"),
        r("C2-A", "3 coupons", "C2_root_face_R", "real 14 mm tube slides through the root rib bore by hand", "dry fit"),
        r("C2-B", "3 coupons", "C2_root_face_R", "magnet sits below the root face (never proud)",
          "calipers depth, both pockets, worst one", "range", 0.2, 0.0, 0.4, "mm"),
        r("C2-C", "3 coupons", "C2_root_face_R", "3 mm rod goes the full 30 mm into the pin hole", "dry fit"),
        r("C2-D", "3 coupons", "C2_root_face_R", "root face flat: gap under a straightedge", "feeler / paper",
          "range", 0.0, 0.0, 0.3, "mm"),
        r("C3-A", "3 coupons", "C3_inner_rib_R", "10x1 carbon strip slides through floor and roof slots", "dry fit"),
        r("C3-B", "3 coupons", "C3_inner_rib_R", "real 14 mm tube slides through the inner rib bore", "dry fit"),
        r("C4-A", "3 coupons", "C4_nose_ring + nose_cap", "TPU nose cap pushes on fully and stays on when shaken", "dry fit"),
        r("P-NAC-A", "3 coupons", "nacelle (LW-ASA, solid)", "motor's M3 bolts go into the 16/19 mm pattern by hand", "test fit the X2216"),
        r("P-NAC-B", "3 coupons", "nacelle (LW-ASA, solid)", "prop shaft / motor back clears the 10 mm centre hole", "test fit"),
        r("P-CAM-A", "3 coupons", "camera_mount", "Camera Module 3 M2 holes line up (21 x 12.5 mm)", "test fit the board"),
        r("P-STOP-A", "3 coupons", "battery_stop (PETG)", "drops into a C1 tray slot without force, no rattle", "test fit"),
        r("P-KERF", "3 coupons", "EPP test cut", "hot-wire kerf (template assumes 1.0 mm)",
          "cut a 50 mm block between templates; kerf = template width - foam width", "range", st.hotwire_kerf_mm, 0.6, 1.4, "mm"),
        r("T-FIN", "4 tests", "fin on scrap EPP", "fin tape release force (2 pegs + tape as planned)",
          "luggage scale at the fin tip, pull sideways, peak", "range", 30, 10, 40, "N"),
        r("T-MAG", "4 tests", "C2 + EPP root scrap", "panel magnet pull-off along the spar", "luggage scale, peak",
          "range", 30, 19, 50, "N"),
        r("C2-E", "4 tests", "C2_root_face_R", "sleeve offcut bonded in the root rib boss (24 h cure) holds a push-out load",
          "dowel on the sleeve, coupon on blocks over a bathroom scale; press until 40 kg reads", "min", 390, 390, "", "N"),
        r("P-POD-A", "5 real pod", "pod_rear_L/R", "straight 14 mm tube passes through both bosses each side (alignment)",
          "dry fit, before any glue"),
        r("P-POD-B", "5 real pod", "pod + hatch lids", "lids drop in, even gap, magnets hold with the pod upside down", "test fit"),
        r("P-POD-C", "5 real pod", "pod + battery", "pack goes in through the front hatch over the camera mount", "test fit"),
        r("P-POD-D", "5 real pod", "pod + nacelle", "nacelle tongue slides in, 3 bolt holes line up", "test fit"),
        r("T-CG", "5 real pod", "assembled aircraft", "balance point behind the nose cap tip, battery fitted",
          "balance on fingertips / CG stand", "range", CG_TARGET_MM, CG_BAND_MM[0], CG_BAND_MM[1], "mm"),
    ]


# --------------------------------------------------------------------------
# Weigh-in log
# --------------------------------------------------------------------------

WEIGH_FIELDS = ("id", "kind", "item", "material", "count", "target_g", "budget_g", "x_mm", "volume_cm3", "measured_g",
                "notes")

# weights.py lines whose x the CAD's automatic packing changed (keyword in the line -> CAD payload name)
PAYLOAD_X = {"Camera Module 3": "camera + mount", "Raspberry Pi": "Pi Zero 2 W", "microSD": "Pi Zero 2 W",
             "Flight controller": "flight controller", "ESC:": "ESC", "Remote ID": "Remote ID",
             "Airspeed": "airspeed sensor", "BEC": "BEC", "ELRS": "receiver"}


def weigh_rows(lay, part_stats: dict, cstats: dict) -> list[dict]:
    """Every line of the build: printed parts from the CAD volumes, the rest from weights.py.
    ``target_g`` is the CAD / paper mass, ``budget_g`` what the 1295 g as-built estimate allows."""
    from . import weights as W
    from .cad import REPLACED_WEIGHT_LINES

    rows = []
    for name, s in part_stats.items():
        if not s.get("airframe"):
            continue
        n = s["count"]
        rows.append(dict(id=name, kind="printed", item=name, material=s["material"], count=n,
                         target_g=round(s["mass_g"] * n, 1), budget_g=round(s["mass_g"] * n * (1 + GROWTH_PRINTED), 1),
                         x_mm=round(s["centroid"][0], 1), volume_cm3=round(s["volume_cm3"] * n, 2), measured_g="", notes=""))
    pay = {p.name: (p.lo[0] + p.hi[0]) / 2 for p in lay.payload}
    bu = W.build_up(lay.geom, W.default_propulsion())
    for k, it in enumerate(bu.items):
        if it.name.startswith(REPLACED_WEIGHT_LINES):
            continue
        x = it.cg[0] * 1000
        for key, pname in PAYLOAD_X.items():
            if key in it.name and pname in pay:
                x = pay[pname]
        kind = "battery" if it.name.startswith("Battery") else ("epp/structure" if it.group == "structure" else it.group)
        if kind == "battery":
            x = lay.battery["x"]
        rows.append(dict(id=f"W{k:02d}", kind=kind, item=it.name, material="", count=1,
                         target_g=round(it.mass_kg * 1000, 1), budget_g=round(it.mass_as_built_kg * 1000, 1),
                         x_mm=round(x, 1), volume_cm3="", measured_g="", notes=""))
    rows.append(dict(id="TOTAL", kind="total", item="assembled aircraft, ready to fly (overrides the sum)", material="",
                     count=1, target_g="", budget_g="", x_mm="", volume_cm3="", measured_g="", notes=""))
    for name, s in cstats.items():
        rows.append(dict(id=name, kind="coupon", item=f"{s['coupon']} piece", material="LW-ASA", count=1,
                         target_g=round(s["mass_g"], 1), budget_g="", x_mm="", volume_cm3=round(s["volume_cm3"], 2),
                         measured_g="", notes="weigh to 0.1 g; gives the foamed density"))
    return rows


def write_csv(rows: list[dict], fields, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


# --------------------------------------------------------------------------
# Evaluate what was measured
# --------------------------------------------------------------------------


def _num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


@dataclass
class Verdict:
    id: str
    text: str
    status: str  # PASS, FAIL, ---
    advice: str = ""


def judge_checks(rows: list[dict], st: CadSettings | None = None) -> tuple[list[Verdict], dict]:
    """Pass / fail for each filled-in row, and CadSettings changes the gauges call for."""
    st = st or CadSettings()
    L = {d["key"]: d for d in ladders(st)}
    jl = joint_ladder(st)
    out, changes = [], {}
    for r in rows:
        m = str(r.get("measured", "")).strip()
        kind = r["kind"]
        text = f"{r['part']}: {r['check']}"
        if not m:
            out.append(Verdict(r["id"], text, "---"))
            continue
        if kind == "yn":
            ok = m.lower() in ("y", "yes", "pass", "ok", "1", "true")
            out.append(Verdict(r["id"], text, "PASS" if ok else "FAIL", "" if ok else _yn_advice(r["id"])))
        elif kind in ("range", "min"):
            v = _num(m)
            lo, hi = _num(r["lo"]), _num(r["hi"])
            ok = v is not None and (lo is None or v >= lo - 1e-9) and (hi is None or v <= hi + 1e-9)
            out.append(Verdict(r["id"], f"{text} = {m} {r['unit']}", "PASS" if ok else "FAIL",
                               "" if ok else _range_advice(r["id"], v, lo, hi)))
        elif kind.startswith("ladder:"):
            d = L[kind.split(":")[1]]
            v = _num(m)
            if m.lower() in ("none", "no", "n"):
                out.append(Verdict(r["id"], f"{text}: none fit", "FAIL",
                                   f"even {d['sizes'][-1]:.2f} mm is tight: check the part is real size and the filament dry, "
                                   "then reprint the gauge with larger steps"))
                continue
            ok = v is not None and v <= d["d0"] + 1e-6
            adv = ""
            if v is not None and not ok:
                if d["base"] is None:
                    new = round(v, 2)
                else:
                    new = round((v - d["base"]) / 2 + (0.05 if d["key"] == "tube" else 0.0), 3)
                changes[d["setting"]] = max(new, changes.get(d["setting"], 0.0))
                adv = f"prints tight: set {d['setting']} = {new} and regenerate the CAD"
            out.append(Verdict(r["id"], f"{text}: {m} mm (CAD {d['d0']:.2f})", "PASS" if ok else "FAIL", adv))
        elif kind == "joint":
            k = _num(m)
            if k is None or not 1 <= k <= 5:
                out.append(Verdict(r["id"], f"{text}: '{m}'?", "FAIL", "enter 1-5, or reprint the ladder looser"))
                continue
            g = jl[int(k) - 1]
            ok = g <= st.joint_clear_mm + 1e-9
            adv = "" if ok else f"set joint_clear_mm = {g} and regenerate the CAD"
            if not ok:
                changes["joint_clear_mm"] = g
            out.append(Verdict(r["id"], f"{text}: spigot {int(k)} ({g:.2f} mm)", "PASS" if ok else "FAIL", adv))
    return out, changes


def _yn_advice(i: str) -> str:
    return {
        "CAL-3": "under-extruded: raise the flow 5 % (or drop the temperature 5 C) and reprint",
        "CAL-4": "close the enclosure, preheat the chamber, larger brim",
        "C1-A": "sand the spigot band; if more than a few passes, the G2 ladder sets joint_clear_mm",
        "C1-B": "sand the spigot band; see G2-JOINT",
        "C1-D": "check each joint alone first; a warped piece shows as a corner that won't close",
        "C2-A": "ream with a 14.3 mm drill or sandpaper on a rod; G1-TUBE sets sleeve_bond_gap_mm",
        "C2-C": "drill out 3.3 mm; G1-PIN sets pin_clear_mm",
        "C3-A": "file the slot; if the strip is thicker than 1.0 mm, measure it",
        "C3-B": "ream; see G1-TUBE",
        "C4-A": "TPU too tight: scale the cap 100.5 % in the slicer; too loose: a wrap of tape on the spigot",
        "P-NAC-A": "measure the motor's pattern; change motor_pattern_mm if it is not 16/19",
        "P-CAM-A": "measure the board; change camera_holes_mm",
        "P-STOP-A": "sand the key; PETG shrinks less than LW-ASA swells",
        "P-POD-A": "do not force it: find the tight boss with a short tube and ream that one",
        "P-POD-B": "sand the lid lip; check the lid isn't warped",
        "P-POD-C": "check the camera mount height and the strap; tilt the pack more steeply",
        "P-POD-D": "ream the floor holes; don't bend the tongue",
    }.get(i, "see the notes in docs/test_prints.md")


def _range_advice(i: str, v, lo, hi) -> str:
    if v is None:
        return "not a number"
    low = lo is not None and v < lo
    return {
        "CAL-1": "walls thin: raise the flow" if low else "walls thick (over-foaming): drop the flow or the temperature 5 C",
        "CAL-2": "undersize: raise the flow" if low else "oversize: drop the flow; foam swell closes holes and joints",
        "G1-OD": "tube out of tolerance: measure the real tube and use that OD in sleeve_od_mm",
        "C1-C": "",
        "C2-B": "magnet proud: deepen the pocket 0.3 mm with a drill bit, never leave it standing out" if low else
                "deep pocket: a drop of epoxy under the magnet",
        "C2-D": "warped root face: chamber hotter, brim, slower cooling",
        "C2-E": "bond slipped: sand and degrease both faces, thicker epoxy, more boss length",
        "P-KERF": f"set hotwire_kerf_mm = {v} and regenerate the templates",
        "T-FIN": "fin falls off too easily: more tape" if low else "tape too strong for the spar (sized for 30 N): less tape",
        "T-MAG": "catch weak: magnets closer to flush" if low else "catch strong: a layer of tape over the magnets, or smaller magnets",
        "T-CG": "nose heavy: slide the battery aft" if low else "tail heavy: slide the battery forward",
    }.get(i, "out of range")


def mass_rollup(rows: list[dict], density_assumed: float = LW_ASA_DENSITY_ASSUMED) -> dict:
    """Foamed density from the coupons, forecast of unmeasured LW-ASA parts at that density,
    aircraft mass (measured where weighed, budget elsewhere) and the battery position for the CG."""
    coupons = [r for r in rows if r["kind"] == "coupon" and _num(r["measured_g"]) is not None]
    vol = sum(_num(r["volume_cm3"]) for r in coupons)
    rho = sum(_num(r["measured_g"]) for r in coupons) / vol if vol > 0 else None
    scale = rho / density_assumed if rho else 1.0

    lines = [r for r in rows if r["kind"] not in ("coupon", "total")]
    total_m = total_mx = 0.0
    bat = None
    printed_target = printed_now = 0.0
    measured_n = 0
    for r in lines:
        meas = _num(r["measured_g"])
        tgt, bud, x = _num(r["target_g"]), _num(r["budget_g"]), _num(r["x_mm"])
        if r["kind"] == "printed":
            fc = tgt * (scale if "LW-ASA" in r["material"] else 1.0)
            r["forecast_g"] = fc
            printed_target += tgt
            printed_now += meas if meas is not None else fc
            m = meas if meas is not None else fc * (1 + GROWTH_PRINTED)
        else:
            m = meas if meas is not None else bud
        measured_n += meas is not None
        r["used_g"] = m
        if r["kind"] == "battery":
            bat = (m, x)
            continue
        total_m += m
        total_mx += m * x
    m_bat = bat[0] if bat else 0.0
    aircraft = total_m + m_bat
    tot = next((r for r in rows if r["kind"] == "total"), None)
    weighed = _num(tot["measured_g"]) if tot else None
    x_bat = (CG_TARGET_MM * aircraft - total_mx) / m_bat if m_bat else float("nan")
    final = weighed if weighed is not None else aircraft
    status = "GREEN" if final <= OPTIMIZER_CAP_G else ("AMBER" if final <= AMBER_LIMIT_G else "RED")
    return dict(density=rho, scale=scale, aircraft_g=aircraft, weighed_g=weighed, final_g=final, status=status,
                battery_x_mm=x_bat, printed_target_g=printed_target, printed_now_g=printed_now,
                measured_lines=measured_n, lines=lines, n_coupons=len(coupons))


def report(checks: list[dict], weigh: list[dict], tray=(191.4, 231.4), st: CadSettings | None = None) -> str:
    v, changes = judge_checks(checks, st)
    mr = mass_rollup(weigh)
    o = []
    out = o.append
    n = {s: sum(x.status == s for x in v) for s in ("PASS", "FAIL", "---")}
    out("== M2.4 test prints: checks ==")
    out(f"{n['PASS']} pass, {n['FAIL']} fail, {n['---']} not measured yet")
    stage = None
    rows = {r["id"]: r for r in checks}
    for x in v:
        s = rows[x.id]["stage"]
        if s != stage:
            out(f"-- stage {s} --")
            stage = s
        out(f"  [{x.status:4s}] {x.id:9s} {x.text}")
        if x.advice:
            out(f"           -> {x.advice}")
    if changes:
        out("CAD changes the gauges ask for (CadSettings, then rerun python -m manta_mapper.cad ... and testprints generate):")
        for k, val in changes.items():
            out(f"  {k} = {val}")
    out("")
    out("== Mass ==")
    if mr["density"]:
        d = mr["density"]
        flag = "OK" if d <= LW_ASA_DENSITY_OK else ("HEAVY" if d <= LW_ASA_DENSITY_MAX else "NOT FOAMING")
        out(f"LW-ASA foamed density from {mr['n_coupons']} coupon(s): {d:.2f} g/cm3 (CAD assumes {LW_ASA_DENSITY_ASSUMED:.2f}; "
            f"aim <= {LW_ASA_DENSITY_OK:.2f}) [{flag}]")
    else:
        out(f"LW-ASA density: no coupon weighed yet, using {LW_ASA_DENSITY_ASSUMED:.2f} g/cm3")
    out(f"Printed airframe parts: {mr['printed_now_g']:.0f} g measured/forecast vs {mr['printed_target_g']:.0f} g from the CAD")
    out(f"  {'part':18s} {'CAD g':>7s} {'forecast':>8s} {'measured':>8s} {'diff':>6s}")
    for r in mr["lines"]:
        if r["kind"] != "printed":
            continue
        meas = _num(r["measured_g"])
        diff = flag = ""
        if meas is not None:
            d = meas / _num(r["target_g"]) - 1
            diff, flag = f"{d * 100:+5.0f}%", ("" if d <= PART_HEAVY else "  HEAVY")
        out(f"  {r['item']:18s} {_num(r['target_g']):7.1f} {r['forecast_g']:8.1f} "
            f"{('%.1f' % meas) if meas is not None else '':>8s} {diff:>6s}{flag}")
    out(f"Lines weighed: {mr['measured_lines']} of {len(mr['lines'])} (the rest count at their as-built budget)")
    if mr["weighed_g"] is not None:
        out(f"Assembled aircraft WEIGHED: {mr['weighed_g']:.0f} g (sum of lines {mr['aircraft_g']:.0f} g)")
    else:
        out(f"Aircraft estimate: {mr['aircraft_g']:.0f} g")
    out(f"  green <= {OPTIMIZER_CAP_G:.0f} g (optimizer cap), amber <= {AMBER_LIMIT_G:.0f} g (95 % of the "
        f"{DESIGN_MASS_G:.0f} g design mass), red above: [{mr['status']}]")
    xb = mr["battery_x_mm"]
    inside = tray[0] <= xb <= tray[1]
    out(f"Battery centre for CG {CG_TARGET_MM:.0f} mm: x = {xb:.0f} mm; tray stops reach {tray[0]:.0f}-{tray[1]:.0f} mm "
        f"[{'OK' if inside else 'OUTSIDE: ballast or move an item'}]")
    if mr["status"] != "GREEN":
        out("  Levers (docs/cad.md): 0.6 mm skin on the front sections, lightening holes in the root ribs, "
            "LW-ASA nacelle cone with a solid firewall, a lower print flow")
    return "\n".join(o)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def generate(out: str | Path, render_path: str | Path | None = None) -> dict:
    from . import cad, cad_mesh
    from .structures import load_geometry

    out = Path(out)
    (out / "stl").mkdir(parents=True, exist_ok=True)
    lay = cad.layout(load_geometry())
    rec = cad.build_recipe(lay)
    built = cad_mesh.build(rec)
    pstats = cad_mesh.part_stats(rec, built, lay.st)
    cps = coupons(rec, built, lay.st)
    cstats = coupon_stats(cps)
    for c in cps:
        for n, m in c.solids.items():
            cad_mesh.write_stl(m, out / "stl" / f"{n}.stl")
    write_csv(check_rows(lay.st), CHECK_FIELDS, out / "fit_checks.csv")
    write_csv(weigh_rows(lay, pstats, cstats), WEIGH_FIELDS, out / "weigh_in.csv")
    if render_path:
        render(cps, render_path)
    return dict(coupons=cps, stats=cstats, layout=lay)


def render(cps: list[Coupon], path) -> None:  # pragma: no cover - visual
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    light = np.array([0.4, -0.5, 0.8])
    light /= np.linalg.norm(light)
    fig = plt.figure(figsize=(16, 8))
    for k, c in enumerate(cps):
        ax = fig.add_subplot(2, 4, k + 1, projection="3d")
        allv = []
        cols = plt.cm.tab10(np.arange(len(c.solids)) % 10)
        for j, (n, m) in enumerate(c.solids.items()):
            mesh = m.to_mesh()
            V = np.asarray(mesh.vert_properties)[:, :3].copy()
            if c.name == "G2_joint_ladder" and j:
                V[:, 0] += 0
                V[:, 1] += 28 * j
            T = np.asarray(mesh.tri_verts)
            tri = V[T]
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            shade = 0.35 + 0.65 * np.clip(np.abs(nrm @ light), 0, 1)
            fc = np.clip(cols[j][None, :3] * shade[:, None], 0, 1)
            ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolors="none"))
            allv.append(V)
        allv = np.vstack(allv)
        lo, hi = allv.min(0), allv.max(0)
        mid, r = (lo + hi) / 2, (hi - lo).max() / 2
        for setlim, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            setlim(mid[i] - r, mid[i] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=28, azim=-60)
        ax.set_axis_off()
        ax.set_title(f"{c.name}\n{c.checks[:60]}", fontsize=8)
    fig.suptitle("M2.4 test prints: calibration box, fit gauge, joint ladder and slices of the real pod parts", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(prog="manta_mapper.testprints", description="M2.4 test prints.")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="coupon STLs + blank check sheet and weigh-in log (needs manifold3d)")
    g.add_argument("--out", default="cad/testprints")
    g.add_argument("--render", help="PNG of the coupons")
    c = sub.add_parser("check", help="judge a filled-in check sheet and weigh-in log")
    c.add_argument("--log", default="cad/testprints", help="folder with fit_checks.csv and weigh_in.csv")
    c.add_argument("--report", help="save the report here")
    a = p.parse_args(argv)
    if a.cmd == "generate":
        res = generate(a.out, a.render)
        for n, s in res["stats"].items():
            b = s["bbox"]
            print(f"{n:24s} {s['volume_cm3']:6.1f} cm3 {s['mass_g']:6.1f} g  {b[0]:.0f} x {b[1]:.0f} x {b[2]:.0f} mm"
                  f"  pieces {s['pieces']}")
        print(f"wrote {a.out}/stl, fit_checks.csv, weigh_in.csv")
    else:
        log = Path(a.log)
        txt = report(read_csv(log / "fit_checks.csv"), read_csv(log / "weigh_in.csv"))
        print(txt)
        if a.report:
            Path(a.report).write_text(txt + "\n")


if __name__ == "__main__":  # pragma: no cover
    main()
