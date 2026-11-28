"""Scripted CAD: printable parts from the pipeline geometry (milestone M2.3).

This module does every bit of the CAD *thinking* in plain Python, so it can be
tested without Fusion 360:

* where the pod is split into print sections that fit a Creality K2 Pro
  (300 x 300 x 300 mm, less a margin), and where the joints go so they miss
  the spar ribs, the carbon strips and the hatches;
* the spar sleeve mounts for the M2.2 joint (14x12 sleeve, two printed ribs
  with bosses, carbon strip caps in the floor and roof, anti-rotation pin,
  magnetic panel catch);
* hatches (front hook + rear magnets), battery tray with printed end stops,
  camera mount tilted 6 deg, pusher motor nacelle, replaceable nose cap,
  twin sacrificial skid rails, taped drooped tip fins;
* EPP hot-wire templates for the outer panels and fins.

The output is a **recipe**: a list of named solids built from a handful of
primitive operations (ruled loft through airfoil sections, box, cylinder or
cone, extruded polygon, translate, rotate, union, difference, intersection)
plus a list of parts that point at those solids. Two thin interpreters turn
the recipe into real geometry:

* ``cad/fusion/MantaMapperCAD/MantaMapperCAD.py``, a Fusion 360 script, which
  builds editable bodies and exports STLs;
* :mod:`manta_mapper.cad_mesh` (needs ``manifold3d``), which builds the same
  parts as meshes in plain Python. The tests use it to prove the parts are
  watertight and fit the printer, and it doubles as a fallback STL source.

Conventions: millimetres in the recipe (Fusion works in cm internally; the
script converts). Same axes as the rest of the pipeline: x aft from the pod
nose, y to the right wingtip, z up.

    python -m manta_mapper.cad config/mission.toml --json cad/manta_cad.json --dxf cad/templates \
        --report docs/cad_report.txt --plots docs/img
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .geometry import BWBGeometry, airfoil_surfaces

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CadSettings:
    """Build choices for the CAD. Defaults favour robust, forgiving parts."""

    # --- printer -----------------------------------------------------------
    printer_mm: tuple[float, float, float] = (300.0, 300.0, 300.0)  # Creality K2 Pro
    printer_margin_mm: float = 10.0  # per side: brim, skirt, warping, and not printing to the very edge
    # --- sections and skin -------------------------------------------------
    n_airfoil: int = 49  # points per surface in each section (cosine spaced)
    te_thickness_mm: float = 1.0  # printable trailing edge
    wall_mm: float = 0.8  # LW-ASA skin, two 0.4 mm perimeters (weights.BuildChoices.pod_wall_m)
    end_plate_mm: float = 3.0  # solid wall at the panel root face
    min_cavity_gap_mm: float = 1.0  # inside surfaces stop where the cavity is thinner than this
    # --- joints between print sections -------------------------------------
    joint_lap_mm: float = 8.0  # spigot band reaches this far into the mating piece
    joint_band_mm: float = 1.6  # spigot band thickness, inside the skin
    joint_clear_mm: float = 0.1  # glue gap, per side
    cut_keepout_mm: float = 12.0  # a cut stays this far from ribs, strip lands, sleeves and the camera
    centre_band_gap_mm: float = 10.0  # centreline band stops this far short of other joints and parts
    # --- nose cap (TPU) ----------------------------------------------------
    nose_len_mm: float = 25.0
    nose_wall_mm: float = 1.2
    # --- spar joint (M2.2, structures.py) ------------------------------------
    sleeve_od_mm: float = 14.0
    sleeve_len_mm: float = 120.0
    socket_mm: float = 100.0  # spar tube inside the sleeve
    sleeve_bond_gap_mm: float = 0.15  # radial, epoxy
    rib_web_mm: float = 8.0
    rib_boss_len_mm: float = 15.0
    boss_wall_mm: float = 4.0
    inner_rib_half_len_mm: float = 20.0  # inner rib is local to the sleeve so it doesn't split the payload bay
    root_rib_fwd_mm: float = 16.0  # root rib starts this far ahead of the spar
    strip_w_mm: float = 10.0  # 10 x 1 mm carbon strip caps
    strip_t_mm: float = 1.0
    strip_bond_mm: float = 80.0  # strip bonded at least this far past each inner rib
    land_mm: float = 1.2  # local skin thickening under each strip
    pin_d_mm: float = 3.0  # anti-rotation carbon pin
    pin_chord_frac: float = 0.60
    pin_depth_mm: float = 30.0
    magnet_d_mm: float = 10.0  # 10 x 3 mm N52 discs, 2 per side: catch holds > 19 N, lets go well before damage
    magnet_t_mm: float = 3.0
    magnet_chord_fracs: tuple[float, float] = (0.12, 0.45)
    # --- printed fits (radial clearance unless noted; the M2.4 fit gauge checks them) ---
    pocket_clear_mm: float = 0.1  # magnet pockets, panel catch and hatch
    pin_clear_mm: float = 0.15  # anti-rotation pin hole
    bolt_hole_d_mm: float = 3.4  # M3 clearance (diameter): motor, nacelle tongue
    servo_lead_hole_mm: float = 9.0
    servo_lead_chord_frac: float = 0.70
    # --- hatches -----------------------------------------------------------
    hatch_width_mm: float = 90.0
    hatch_rim_mm: float = 6.0
    hatch_lip_mm: float = 4.0  # width of the lid's locating lip
    hatch_clear_mm: float = 0.3
    hatch_magnet_d_mm: float = 6.0  # 6 x 2 mm magnets at the aft edge
    hatch_magnet_t_mm: float = 2.0
    min_hatch_inside_mm: float = 18.0  # hand room under a hatch
    # --- battery tray --------------------------------------------------------
    tray_travel_mm: float = 20.0  # the end stops move the pack this far either way of nominal
    stop_pitch_mm: float = 5.0
    rail_w_mm: float = 4.0
    rail_rise_mm: float = 5.0  # flat rail top this far above the highest floor point under the pack
    slot_w_mm: float = 3.0
    slot_depth_mm: float = 2.5
    stop_len_mm: float = 10.0
    strap_w_mm: float = 20.0
    # --- camera --------------------------------------------------------------
    camera_tilt_deg: float = 6.0  # M1.4: lens tilted forward so it looks straight down at cruise
    camera_fov_deg: tuple[float, float] = (66.0, 41.0)  # Camera Module 3 standard: along-track (rotated 90 deg), across
    fov_margin_deg: float = 3.0
    lens_r_mm: float = 4.5
    lens_drop_mm: float = 5.0  # lens below the board
    camera_mount_rise_mm: float = 10.0  # board plane above the inside floor at the lens
    camera_board_mm: tuple[float, float] = (25.0, 24.0)
    camera_holes_mm: tuple[float, float] = (21.0, 12.5)
    # --- motor nacelle (separate ASA part, bolted through a floor tongue) ----
    nacelle_len_mm: float = 105.0
    nacelle_wall_mm: float = 2.5
    nacelle_front_r_mm: float = 12.0
    firewall_mm: float = 4.0
    motor_overhang_mm: float = 4.0  # firewall aft face this far behind the pod TE (weights.py)
    motor_pattern_mm: tuple[float, float] = (16.0, 19.0)  # X2216 cross pattern, M3
    tongue_len_mm: float = 60.0
    tongue_w_mm: float = 40.0
    tongue_t_mm: float = 3.0
    folded_prop_drop_mm: float = 20.0  # folded blades lie this far below the motor axis
    min_prop_pitch_deg: float = 6.0  # nose-up attitude on the ground before folded blades touch
    # --- skid (EPP, velcro) -----------------------------------------------------
    skid_rail_w_mm: float = 30.0
    skid_inner_y_mm: float = 30.0  # rails clear of the camera window
    skid_x0_frac: float = 0.14
    skid_aft_gap_mm: float = 12.0  # rails stop this far ahead of the TE
    # --- fins and templates ------------------------------------------------------
    fin_peg_d_mm: float = 3.0
    fin_peg_fracs: tuple[float, float] = (0.30, 0.60)
    fin_peg_len_mm: float = 20.0
    hotwire_kerf_mm: float = 1.0  # EPP; templates grow by half of it. Check on scrap.
    template_t_mm: float = 3.0
    spar_hole_clear_mm: float = 0.3
    payload_clear_mm: float = 2.0
    item_floor_gap_mm: float = 2.5  # velcro under each item


# --------------------------------------------------------------------------
# Airfoil sections
# --------------------------------------------------------------------------


def xc_grid(n: int) -> np.ndarray:
    return 0.5 * (1 - np.cos(np.linspace(0.0, math.pi, n)))


def _surf(key: str, xc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    (xu, zu), (xl, zl) = airfoil_surfaces(key)
    return np.interp(xc, xu, zu), np.interp(xc, xl, zl)


def _loop(xu, zu, xl, zl) -> np.ndarray:
    """Closed 2-D loop: TE upper -> LE -> TE lower (counter-clockwise in x-z)."""
    up = np.column_stack([xu, zu])[::-1]
    lo = np.column_stack([xl, zl])[1:]
    return np.vstack([up, lo])


def offset_loop(P: np.ndarray, d: float) -> np.ndarray:
    """Offset a closed counter-clockwise loop inward by ``d`` (negative grows it)."""
    e = np.roll(P, -1, 0) - P
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    n_edge = np.column_stack([-e[:, 1], e[:, 0]])  # left normal = inward for CCW
    nv = np.roll(n_edge, 1, 0) + n_edge
    nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-12)
    cos = np.clip(np.sum(nv * n_edge, axis=1), 0.5, 1.0)
    return P + d * nv / cos[:, None]


def _resample_halves(P: np.ndarray, n: int, gap_min: float) -> np.ndarray:
    """Split an (offset) airfoil loop into its halves, trim the trailing edge where
    the halves come closer than ``gap_min``, and resample both on a cosine grid."""
    m = (len(P) + 1) // 2
    up = P[:m][::-1]  # LE -> TE
    lo = P[m - 1:]
    x_le = float(min(up[0, 0], lo[0, 0]))

    def mono(c):
        c = c[c[:, 0] >= x_le - 1e-9]
        keep = np.concatenate([[True], np.diff(np.maximum.accumulate(c[:, 0])) > 1e-9])
        return c[keep]

    up, lo = mono(up), mono(lo)
    x_te = float(min(up[-1, 0], lo[-1, 0]))
    xs = np.linspace(x_le, x_te, 400)
    gap = np.interp(xs, up[:, 0], up[:, 1]) - np.interp(xs, lo[:, 0], lo[:, 1])
    ok = np.flatnonzero(gap >= gap_min)
    if len(ok) == 0:
        raise ValueError("section too thin for this offset")
    # first run of good points from the LE (allow the very LE itself to be thin)
    start = ok[0]
    run_end = start
    while run_end + 1 < len(xs) and gap[run_end + 1] >= gap_min:
        run_end += 1
    x_te = float(xs[run_end])
    x0 = float(xs[0])
    g = x0 + (x_te - x0) * xc_grid(n)
    zu = np.interp(g, up[:, 0], up[:, 1])
    zl = np.interp(g, lo[:, 0], lo[:, 1])
    zle = 0.5 * (zu[0] + zl[0])
    zu[0] = zl[0] = zle
    return _loop(g, zu, g, zl)


def _rotate_xz(P: np.ndarray, twist_deg: float, about=(0.0, 0.0)) -> np.ndarray:
    """Rotate 2-D (x, z) points nose-up by ``twist_deg`` about ``about``."""
    t = math.radians(twist_deg)
    c, s = math.cos(t), math.sin(t)
    q = P - np.asarray(about)
    return np.column_stack([q[:, 0] * c + q[:, 1] * s, -q[:, 0] * s + q[:, 1] * c]) + np.asarray(about)


@dataclass(frozen=True)
class PodStation:
    y_mm: float
    le_x: float
    chord: float
    blend: float  # 0 pod section, 1 outer section


class PodShape:
    """The printed centre body (pod + blend strakes), in mm, from the geometry."""

    def __init__(self, geom: BWBGeometry, st: CadSettings):
        self.g, self.st = geom, st
        self.c0 = geom.pod_chord_m * 1000
        self.w = geom.pod_half_width_m * 1000
        self.root_y = geom.root_y_m * 1000
        self.root_le = geom.root_le_x_m * 1000
        self.root_c = geom.root_chord_m * 1000
        self.xc = xc_grid(st.n_airfoil)
        self._pod = _surf(geom.pod_airfoil, self.xc)
        self._out = _surf(geom.outer_airfoil, self.xc)
        self._fine_xc = np.linspace(0, 1, 801)
        self._pod_f = _surf(geom.pod_airfoil, self._fine_xc)
        self._out_f = _surf(geom.outer_airfoil, self._fine_xc)

    def station(self, y_mm: float) -> PodStation:
        y = abs(y_mm)
        if y <= self.w or self.g.blend_width_m == 0:
            return PodStation(y_mm, 0.0, self.c0, 0.0)
        f = min((y - self.w) / (self.root_y - self.w), 1.0)
        return PodStation(y_mm, f * self.root_le, self.c0 + f * (self.root_c - self.c0), f)

    def _normalised(self, f: float, fine: bool = False):
        a, b = (self._pod_f, self._out_f) if fine else (self._pod, self._out)
        return (1 - f) * a[0] + f * b[0], (1 - f) * a[1] + f * b[1]

    def loop2d(self, y_mm: float, inset_mm: float = 0.0) -> np.ndarray:
        """Section at y as a closed (x, z) loop in mm; ``inset_mm`` > 0 gives the inside surface."""
        s = self.station(y_mm)
        zu, zl = self._normalised(s.blend)
        x = s.le_x + self.xc * s.chord
        t = 0.5 * self.st.te_thickness_mm * self.xc
        P = _loop(x, zu * s.chord + t, x, zl * s.chord - t)
        if inset_mm > 0:
            P = _resample_halves(offset_loop(P, inset_mm), self.st.n_airfoil, self.st.min_cavity_gap_mm)
        return P

    def loop3d(self, y_mm: float, inset_mm: float = 0.0) -> np.ndarray:
        P = self.loop2d(y_mm, inset_mm)
        return np.column_stack([P[:, 0], np.full(len(P), y_mm), P[:, 1]])

    def z_out(self, x_mm: float, y_mm: float) -> tuple[float, float] | None:
        """(lower, upper) outer surface z at (x, y), or None off the planform."""
        s = self.station(y_mm)
        u = (x_mm - s.le_x) / s.chord
        if not 0 <= u <= 1 or abs(y_mm) > self.root_y + 1e-9:
            return None
        zu, zl = self._normalised(s.blend, fine=True)
        t = 0.5 * self.st.te_thickness_mm * u
        return float(np.interp(u, self._fine_xc, zl) * s.chord - t), float(np.interp(u, self._fine_xc, zu) * s.chord + t)

    def z_in(self, x_mm: float, y_mm: float) -> tuple[float, float] | None:
        """(floor, roof) of the inside skin at (x, y) (vertical offset; fine away from the LE)."""
        z = self.z_out(x_mm, y_mm)
        if z is None:
            return None
        return z[0] + self.st.wall_mm, z[1] - self.st.wall_mm

    def inside_height(self, x_mm: float, y_mm: float = 0.0) -> float:
        z = self.z_in(x_mm, y_mm)
        return 0.0 if z is None else z[1] - z[0]

    def lofts_y(self, inset_mm: float = 0.0) -> list[float]:
        y_end = self.root_y if inset_mm <= 0 else self.root_y - self.st.end_plate_mm
        if self.g.blend_width_m == 0:
            return [-y_end, y_end]
        return [-y_end, -self.w, self.w, y_end]


# --------------------------------------------------------------------------
# Recipe: named solids from primitive operations
# --------------------------------------------------------------------------


def _r(a) -> list:
    return np.round(np.asarray(a, dtype=float), 4).tolist()


class Recipe:
    """Named solids (a DAG of primitive operations) and the parts that use them.

    Every op is a dict with ``op`` and its arguments; the interpreters
    (Fusion 360, manifold3d) implement exactly these and nothing else:

    * ``loft``: ruled loft through ``sections`` (each a closed 3-D polyline,
      all with the same point count; consecutive sections joined by straight
      rulings, ends capped);
    * ``box``: axis-aligned, ``lo`` / ``hi`` corners;
    * ``cyl``: cylinder or cone from ``p0`` (radius ``r0``) to ``p1`` (``r1``);
    * ``prism``: polygon ``poly`` in the XY plane extruded from ``z0`` to ``z1``;
    * ``move``: translate ``of`` by ``v``;
    * ``rotate``: rotate ``of`` by ``deg`` about ``axis`` through ``origin``;
    * ``union`` / ``inter``: of ``items``; ``diff``: ``of`` minus ``items``.
    """

    OPS = ("loft", "box", "cyl", "prism", "move", "rotate", "union", "inter", "diff")

    def __init__(self):
        self.solids: dict[str, dict] = {}
        self.parts: list[dict] = []

    def _add(self, name: str, d: dict) -> str:
        if name in self.solids:
            raise KeyError(f"duplicate solid {name}")
        for k in ("of", "items"):
            refs = d.get(k)
            for ref in [refs] if isinstance(refs, str) else (refs or []):
                if ref not in self.solids:
                    raise KeyError(f"{name} refers to unknown solid {ref}")
        self.solids[name] = d
        return name

    def loft(self, name, sections):
        n = {len(s) for s in sections}
        if len(n) != 1 or len(sections) < 2:
            raise ValueError("loft needs >= 2 sections with the same point count")
        return self._add(name, dict(op="loft", sections=[_r(s) for s in sections]))

    def box(self, name, lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if np.any(hi <= lo):
            raise ValueError(f"empty box {name}")
        return self._add(name, dict(op="box", lo=_r(lo), hi=_r(hi)))

    def cyl(self, name, p0, p1, r0, r1=None):
        return self._add(name, dict(op="cyl", p0=_r(p0), p1=_r(p1), r0=round(float(r0), 4),
                                    r1=round(float(r0 if r1 is None else r1), 4)))

    def prism(self, name, poly, z0, z1):
        return self._add(name, dict(op="prism", poly=_r(poly), z0=float(z0), z1=float(z1)))

    def move(self, name, of, v):
        return self._add(name, dict(op="move", of=of, v=_r(v)))

    def rotate(self, name, of, axis, deg, origin=(0, 0, 0)):
        return self._add(name, dict(op="rotate", of=of, axis=_r(axis), deg=float(deg), origin=_r(origin)))

    def union(self, name, items):
        return self._add(name, dict(op="union", items=list(items)))

    def inter(self, name, items):
        return self._add(name, dict(op="inter", items=list(items)))

    def diff(self, name, of, items):
        return self._add(name, dict(op="diff", of=of, items=list(items)))

    def part(self, name, solid, material, printed=True, count=1, orient="", note="", mirror_of=None):
        if solid not in self.solids:
            raise KeyError(solid)
        self.parts.append(dict(name=name, solid=solid, material=material, printed=printed, count=count,
                               orient=orient, note=note, mirror_of=mirror_of))

    def order(self, roots=None) -> list[str]:
        """Solids in dependency order (only those needed by ``roots``)."""
        roots = roots or [p["solid"] for p in self.parts]
        seen, out = set(), []

        def visit(n):
            if n in seen:
                return
            seen.add(n)
            d = self.solids[n]
            refs = d.get("items") or []
            if "of" in d:
                refs = [d["of"], *refs]
            for r in refs:
                visit(r)
            out.append(n)

        for r in roots:
            visit(r)
        return out

    def to_dict(self) -> dict:
        return dict(solids=self.solids, parts=self.parts)


# --------------------------------------------------------------------------
# Layout: everything the parts are placed by
# --------------------------------------------------------------------------


@dataclass
class Box3:
    name: str
    lo: np.ndarray
    hi: np.ndarray

    def overlaps(self, o: "Box3", clear: float = 0.0) -> bool:
        return bool(np.all(self.lo - clear < o.hi) and np.all(o.lo - clear < self.hi))


@dataclass
class Check:
    name: str
    value: float
    limit: float
    sense: str  # "<=" or ">="
    unit: str = "mm"
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.value <= self.limit + 1e-9 if self.sense == "<=" else self.value >= self.limit - 1e-9


@dataclass
class Layout:
    """All positions and sizes, in mm. Built by :func:`layout`."""

    geom: BWBGeometry
    st: CadSettings
    pod: PodShape
    spar_root: np.ndarray  # spar centre at the panel root face (right side)
    spar_tip: np.ndarray
    spar_dir: np.ndarray  # unit, root -> tip
    spar_layout: list[dict]  # telescoping tubes from structures (from_mm/to_mm along the spar)
    inner_rib: dict  # centre, x/y extents
    root_rib: dict
    strip: dict  # x, half length, z of floor/roof
    pin: dict
    magnets: list[np.ndarray]
    servo_hole: np.ndarray
    cuts_x: list[float]  # chordwise cuts between print sections (plus the centreline split)
    sections: list[dict]  # print sections: name, x range, side, bbox
    hatches: list[dict]
    tray: dict
    battery: dict
    camera: dict
    nacelle: dict
    skid: dict
    fin: dict
    payload: list[Box3]
    structure_boxes: list[Box3]
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def point_on_spar(self, s_mm: float) -> np.ndarray:
        return self.spar_root + s_mm * self.spar_dir


def _structures_spar(path: Path | None) -> tuple[list[dict], dict]:
    p = path or Path(__file__).resolve().parents[2] / "docs" / "structures_result.json"
    if p.exists():
        d = json.loads(p.read_text())
        return d.get("spar", []), d.get("settings", {})
    return [dict(tube="12x10", od_mm=12.0, id_mm=10.0, from_mm=-100.0, to_mm=1e9)], {}


def _section_mid(geom: BWBGeometry, eta: float, frac: float) -> np.ndarray:
    """Point on the outer panel at chord fraction ``frac``, halfway between the skins (mm)."""
    le, c, tw = geom._outer_station(eta)
    zu, zl = _surf(geom.outer_airfoil, np.array([frac]))
    p = _rotate_xz(np.array([[frac * c, 0.5 * (zu[0] + zl[0]) * c]]), tw)[0]
    return np.array([le[0] + p[0], le[1], le[2] + p[1]]) * 1000


def outer_section3d(geom: BWBGeometry, eta: float, n: int, te_mm: float = 0.0) -> np.ndarray:
    """Outer-panel section (S5010, twisted about its LE) as a closed 3-D loop in mm."""
    le, c, tw = geom._outer_station(eta)
    xc = xc_grid(n)
    zu, zl = _surf(geom.outer_airfoil, xc)
    c_mm = c * 1000
    t = 0.5 * te_mm * xc
    P = _rotate_xz(_loop(xc * c_mm, zu * c_mm + t, xc * c_mm, zl * c_mm - t), tw)
    le = le * 1000
    return np.column_stack([le[0] + P[:, 0], np.full(len(P), le[1]), le[2] + P[:, 1]])


def fin_sections(geom: BWBGeometry, n: int, te_mm: float = 0.6) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fin root (= the panel tip section, in its vertical plane) and fin tip (NACA 0009 in the plane
    square to the fin's span), both closed 3-D loops in mm, plus the template frame."""
    root = outer_section3d(geom, 1.0, n, te_mm)
    fr, ft = geom.fin_stations()
    d = math.radians(geom.fin_droop_deg)
    u = np.array([0.0, math.cos(d), -math.sin(d)])  # fin span direction (no sweep)
    nrm = np.array([0.0, math.sin(d), math.cos(d)])  # "up" in the fin's section plane
    xc = xc_grid(n)
    zu, zl = _surf(geom.fin_airfoil, xc)
    c = ft["chord"] * 1000
    t = 0.5 * te_mm * xc
    P = _loop(xc * c, zu * c + t, xc * c, zl * c - t)
    tip_le = np.asarray(ft["le"]) * 1000
    tip = tip_le + P[:, :1] * np.array([1.0, 0, 0]) + P[:, 1:2] * nrm
    return root, tip, dict(u=u, n=nrm, root_le=np.asarray(fr["le"]) * 1000, tip_le=tip_le, droop_deg=geom.fin_droop_deg)


def fin_template_loops(geom: BWBGeometry, n: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """2-D hot-wire templates for the fin, both in planes square to the fin span.

    The root template is where the ruled fin crosses the plane through the root
    LE; the fin block is cut a little long and its root end trimmed to the
    panel's (vertical) tip face, a rotation of ``droop`` about the chord line.
    """
    root, tip, fr = fin_sections(geom, n)
    u = fr["u"]
    a, b = root, tip
    t = -((a - fr["root_le"]) @ u) / ((b - a) @ u)
    P = a + t[:, None] * (b - a)
    rel = P - fr["root_le"]
    root2d = np.column_stack([rel[:, 0], rel @ fr["n"]])
    relt = tip - fr["tip_le"]
    tip2d = np.column_stack([relt[:, 0], relt @ fr["n"]])
    return root2d, tip2d, fr


@dataclass(frozen=True)
class PayloadSpec:
    name: str
    weights_key: str  # name prefix of the item in weights.py
    size_mm: tuple[float, float, float]  # length (x), width (y), height (z)
    mass_g: float
    pref: tuple[float, float]  # weights.py position: (fraction of pod chord, y mm)
    centre: bool = False  # must sit on the centreline (camera)
    rotatable: bool = True


#: What has to fit in the pod besides the battery, with the M1.5 positions as the preference.
PAYLOAD = (
    PayloadSpec("camera + mount", "Camera Module 3", (36, 32, 22), 14, (0.55, 0), centre=True, rotatable=False),
    PayloadSpec("Pi Zero 2 W", "Raspberry Pi", (65, 30, 12), 14, (0.63, 55)),
    PayloadSpec("flight controller", "Flight controller", (54, 36, 15), 25, (0.60, -50)),
    PayloadSpec("ESC", "ESC", (45, 25, 10), 35, (0.74, -45)),
    PayloadSpec("Remote ID", "Remote ID", (35, 25, 12), 12, (0.72, 30)),
    PayloadSpec("airspeed sensor", "Airspeed", (32, 22, 12), 12, (0.11, 0)),
    PayloadSpec("BEC", "5 V 3 A buck", (25, 15, 8), 6, (0.70, 60)),
    PayloadSpec("receiver", "ELRS", (18, 11, 5), 2, (0.73, -40)),
)


def _height_map(pod: PodShape, step: float):
    xs = np.arange(0, pod.c0 + 1e-9, step)
    ys = np.arange(-pod.root_y, pod.root_y + 1e-9, step)
    floor = np.full((len(ys), len(xs)), np.inf)
    roof = np.full((len(ys), len(xs)), -np.inf)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            z = pod.z_in(x, y)
            if z is not None:
                floor[j, i], roof[j, i] = z
    return xs, ys, floor, roof


def pack_payload(pod: PodShape, st: CadSettings, obstacles: list[Box3], y_limit: float, step: float = 2.5,
                 hmap=None) -> list[Box3]:
    """Greedy packing: each item (biggest first) goes to the free spot nearest its weights.py position
    where it sits on the floor, clears the roof by ``payload_clear_mm`` and clears every obstacle and
    every item already placed by the same margin. Items may turn 90 deg."""
    from numpy.lib.stride_tricks import sliding_window_view as win

    xs, ys, floor, roof = hmap or _height_map(pod, step)
    placed: list[Box3] = []
    clear = st.payload_clear_mm
    order = sorted(PAYLOAD, key=lambda p: (not p.centre, -p.size_mm[0] * p.size_mm[1]))
    for spec in order:
        best = None
        L0, W0, H = spec.size_mm
        for L, W in ([(L0, W0), (W0, L0)] if spec.rotatable else [(L0, W0)]):
            nx, ny = int(math.ceil(L / step)) + 1, int(math.ceil(W / step)) + 1
            if nx > len(xs) or ny > len(ys):
                continue
            fmax = win(floor, (ny, nx)).max(axis=(2, 3))
            rmin = win(roof, (ny, nx)).min(axis=(2, 3))
            cx = xs[: fmax.shape[1]] + (nx - 1) * step / 2
            cy = ys[: fmax.shape[0]] + (ny - 1) * step / 2
            CX, CY = np.meshgrid(cx, cy)
            z0 = fmax + st.item_floor_gap_mm
            ok = np.isfinite(fmax) & np.isfinite(rmin) & (rmin - (z0 + H) >= clear)
            ok &= (np.abs(CY) + W / 2 <= y_limit)
            if spec.centre:
                ok &= np.abs(CY) < step / 2
            for o in [*obstacles, *placed]:
                ok &= ~((np.abs(CX - 0.5 * (o.lo[0] + o.hi[0])) < L / 2 + 0.5 * (o.hi[0] - o.lo[0]) + clear)
                        & (np.abs(CY - 0.5 * (o.lo[1] + o.hi[1])) < W / 2 + 0.5 * (o.hi[1] - o.lo[1]) + clear))
            if not ok.any():
                continue
            cost = np.abs(CX - spec.pref[0] * pod.c0) + 0.5 * np.abs(CY - spec.pref[1])
            cost = np.where(ok, cost, np.inf)
            j, i = np.unravel_index(np.argmin(cost), cost.shape)
            if best is None or cost[j, i] < best[0]:
                best = (cost[j, i], CX[j, i], CY[j, i], L, W, z0[j, i])
        if best is None:
            raise ValueError(f"no room in the pod for the {spec.name}")
        _, x, y, L, W, z0 = best
        if spec.centre:
            y = 0.0
        placed.append(Box3(spec.name, np.array([x - L / 2, y - W / 2, z0]), np.array([x + L / 2, y + W / 2, z0 + H])))
    return placed


def _sleeve_boxes(on, st: CadSettings, n: int = 6) -> list[Box3]:
    """The sleeve and its bosses as a chain of boxes along the spar (an oblique tube's single bounding
    box would block half the bay)."""
    r = st.sleeve_od_mm / 2 + st.boss_wall_mm
    out = []
    ss = np.linspace(-st.sleeve_len_mm, 0, n + 1)
    for k in range(n):
        a, b = on(ss[k]), on(ss[k + 1])
        out.append(Box3("sleeve + bosses R", np.minimum(a, b) - r, np.maximum(a, b) + r))
    return out


def _mirror_box(b: Box3, name: str) -> Box3:
    return Box3(name, np.array([b.lo[0], -b.hi[1], b.lo[2]]), np.array([b.hi[0], -b.lo[1], b.hi[2]]))


def layout(geom: BWBGeometry, st: CadSettings | None = None, battery_x_mm: float | None = None,
           structures_json: Path | None = None) -> Layout:
    """Place every feature and run the fit checks. Pure Python (numpy + the airfoil tables)."""
    st = st or CadSettings()
    pod = PodShape(geom, st)
    c0 = pod.c0
    spar, _ = _structures_spar(structures_json)

    # --- spar line through the middle of the section at root and tip --------
    p_root = _section_mid(geom, 0.0, geom.spar_chord_fraction)
    p_tip = _section_mid(geom, 1.0, geom.spar_chord_fraction)
    d = (p_tip - p_root) / np.linalg.norm(p_tip - p_root)

    def on(s):
        return p_root + s * d

    r_sl = st.sleeve_od_mm / 2
    r_boss = r_sl + st.boss_wall_mm
    inner_c = on(-st.socket_mm)
    inner_rib = dict(centre=inner_c, x0=inner_c[0] - st.inner_rib_half_len_mm, x1=inner_c[0] + st.inner_rib_half_len_mm,
                     y0=inner_c[1] - st.rib_web_mm / 2, y1=inner_c[1] + st.rib_web_mm / 2,
                     boss=(-st.socket_mm - st.rib_boss_len_mm / 2, -st.socket_mm + st.rib_boss_len_mm / 2))
    x_pin = pod.root_le + st.pin_chord_frac * pod.root_c
    pin_p = _section_mid(geom, 0.0, st.pin_chord_frac)
    root_rib = dict(x0=p_root[0] - st.root_rib_fwd_mm, x1=x_pin + 6, y0=pod.root_y - st.end_plate_mm - st.rib_web_mm,
                    y1=pod.root_y, boss=(-(st.end_plate_mm + st.rib_web_mm + 4) / d[1], 2.0))
    wx0, wx1 = p_root[0] + r_boss + 2, x_pin - 5
    zw = [pod.z_in(x, root_rib["y0"]) for x in np.linspace(wx0, wx1, 15)]
    root_rib["window"] = dict(x0=wx0, x1=wx1, z0=max(z[0] for z in zw) + 3.5, z1=min(z[1] for z in zw) - 3.5)
    strip_half = inner_c[1] + st.strip_bond_mm
    strip = dict(x=float(inner_c[0]), half=float(min(strip_half, pod.root_y - st.end_plate_mm - 1)))
    pin = dict(p=pin_p, d=st.pin_d_mm, depth=st.pin_depth_mm)
    magnets = [_section_mid(geom, 0.0, f) for f in st.magnet_chord_fracs]
    servo_hole = _section_mid(geom, 0.0, st.servo_lead_chord_frac)

    # --- nacelle (pusher motor), skid --------------------------------------------
    x_fw = c0 + st.motor_overhang_mm
    x_a = x_fw - st.nacelle_len_mm
    r_b = 28.0 / 2 + 3.0
    nacelle = dict(x_a=x_a, x_fw=x_fw, x_te=c0, r_a=st.nacelle_front_r_mm, r_b=r_b, z=0.0,
                   tongue=dict(x0=x_a - st.tongue_len_mm, x1=x_a + 12, w=st.tongue_w_mm, t=st.tongue_t_mm),
                   bolts=[(x_a - st.tongue_len_mm + 10, 0.0), (x_a - 12, -12.0), (x_a - 12, 12.0)])
    skid_x0, skid_x1 = st.skid_x0_frac * c0, c0 - st.skid_aft_gap_mm
    z_skid = min(pod.z_out(x, 0)[0] for x in np.linspace(0, c0, 200)) - geom.skid_depth_m * 1000
    skid = dict(x0=skid_x0, x1=skid_x1, y0=st.skid_inner_y_mm, y1=st.skid_inner_y_mm + st.skid_rail_w_mm, z_bottom=z_skid)

    # --- fixed structure the payload must clear ------------------------------------
    sb: list[Box3] = []
    ir = inner_rib
    rib_box = Box3("inner rib R", np.array([ir["x0"], ir["y0"], -99]), np.array([ir["x1"], ir["y1"], 99]))
    rr_box = Box3("root rib R", np.array([root_rib["x0"], root_rib["y0"], -99]), np.array([root_rib["x1"], root_rib["y1"], 99]))
    for b_ in [rib_box, rr_box, *_sleeve_boxes(on, st)]:
        sb += [b_, _mirror_box(b_, b_.name[:-1] + "L")]
    sb.append(Box3("floor strip land", np.array([strip["x"] - st.strip_w_mm / 2 - 3, -strip["half"], -99]),
                   np.array([strip["x"] + st.strip_w_mm / 2 + 3, strip["half"], -99 + 1e-3])))  # low: only the camera cares
    tg = nacelle["tongue"]
    sb.append(Box3("nacelle tongue", np.array([tg["x0"], -tg["w"] / 2, -99]),
                   np.array([tg["x1"], tg["w"] / 2, pod.z_in(tg["x0"], 0)[0] + tg["t"] + 6])))  # + nuts
    sb.append(Box3("nacelle", np.array([x_a, -r_b, -r_b]), np.array([x_fw + 40, r_b, r_b])))

    # --- battery + payload: pack, move the battery to hold the CG, repeat ------------
    from . import weights as W

    bu = W.build_up(geom, W.default_propulsion())
    m_bat = next(it.mass_kg for it in bu.items if it.movable) * 1000
    base_x = bu.battery_x_m * 1000 if battery_x_mm is None else battery_x_mm
    bat_L, bat_W, bat_H = 75.0, 66.0, 23.0
    rail_y = bat_W / 2 + 1.0 + st.rail_w_mm / 2
    hmap = _height_map(pod, 2.5)
    y_limit = pod.root_y - st.end_plate_mm - st.rib_web_mm - 2
    battery_x = base_x
    for _ in range(8):
        tray_x0 = battery_x - st.tray_travel_mm - bat_L / 2 - st.stop_len_mm
        tray_x1 = battery_x + st.tray_travel_mm + bat_L / 2 + st.stop_len_mm
        tray_box = Box3("battery tray (pack travel + stops)", np.array([tray_x0, -rail_y - st.rail_w_mm / 2, -99]),
                        np.array([tray_x1, rail_y + st.rail_w_mm / 2, 99]))
        payload = pack_payload(pod, st, [*sb, tray_box], y_limit, hmap=hmap)
        moment = 0.0
        for p in payload:
            spec = next(s for s in PAYLOAD if s.name == p.name)
            moment += spec.mass_g * (0.5 * (p.lo[0] + p.hi[0]) - spec.pref[0] * c0)
        new_x = base_x - moment / m_bat
        if abs(new_x - battery_x) < 0.5:
            battery_x = new_x
            break
        battery_x = new_x
    tray_x0 = battery_x - st.tray_travel_mm - bat_L / 2 - st.stop_len_mm
    tray_x1 = battery_x + st.tray_travel_mm + bat_L / 2 + st.stop_len_mm
    floor_max = max(pod.z_in(x, y)[0] for x in np.linspace(tray_x0, tray_x1, 30) for y in (0, bat_W / 2 + 4))
    rail_top = floor_max + st.rail_rise_mm
    slots = np.arange(tray_x0 + st.stop_len_mm / 2, tray_x1 - st.stop_len_mm / 2 + 1e-6, st.stop_pitch_mm)
    tray = dict(x0=tray_x0, x1=tray_x1, rail_y=rail_y, rail_top=rail_top, slots=slots.tolist(),
                straps=[battery_x - 20, battery_x + 20])
    battery = dict(x=battery_x, x_weights=base_x, L=bat_L, W=bat_W, H=bat_H, z0=rail_top,
                   x_min=battery_x - st.tray_travel_mm, x_max=battery_x + st.tray_travel_mm, mass_g=m_bat)
    sb.append(Box3("battery tray rails", np.array([tray_x0, -rail_y - st.rail_w_mm / 2, -99]),
                   np.array([tray_x1, rail_y + st.rail_w_mm / 2, rail_top])))
    payload.append(Box3("battery", np.array([battery_x - bat_L / 2, -bat_W / 2, rail_top]),
                        np.array([battery_x + bat_L / 2, bat_W / 2, rail_top + bat_H])))

    # --- camera, from where the packer put it ----------------------------------------
    cb = next(p for p in payload if p.name == "camera + mount")
    x_cam = 0.5 * (cb.lo[0] + cb.hi[0])
    zf = max(pod.z_in(x, y)[0] for x in (cb.lo[0], x_cam, cb.hi[0]) for y in (-16, 0, 16))
    tilt = math.radians(st.camera_tilt_deg)
    axis = np.array([-math.sin(tilt), 0.0, -math.cos(tilt)])  # down and forward
    board = np.array([x_cam, 0.0, zf + st.camera_mount_rise_mm])
    lens_tip = board + st.lens_drop_mm * axis
    h = lens_tip[2] - pod.z_out(x_cam, 0)[0]
    ax, ay = (math.radians(a / 2 + st.fov_margin_deg) for a in st.camera_fov_deg)
    win = dict(x0=lens_tip[0] - h * math.tan(ax + tilt) - st.lens_r_mm, x1=lens_tip[0] + h * math.tan(ax - tilt) + st.lens_r_mm,
               y=h * math.tan(ay) + st.lens_r_mm)
    camera = dict(x=x_cam, board=board, axis=axis, lens_tip=lens_tip, recess=h, window=win, floor_in=zf,
                  block=dict(x0=float(cb.lo[0]), x1=float(cb.hi[0]), y=16.0))

    # --- fins -------------------------------------------------------------------------
    _, _, fr = fin_sections(geom, 9)
    fin = dict(droop_deg=geom.fin_droop_deg, root_le=fr["root_le"], tip_le=fr["tip_le"],
               pegs=[_section_mid(geom, 1.0, f) for f in st.fin_peg_fracs])

    lay = Layout(geom, st, pod, p_root, p_tip, d, spar, inner_rib, root_rib, strip, pin, magnets, servo_hole,
                 [], [], [], tray, battery, camera, nacelle, skid, fin, payload, sb)
    lay.cuts_x = choose_cuts(lay)
    lay.sections = print_sections(lay)
    lay.hatches = place_hatches(lay)
    lay.checks = run_checks(lay)
    for p in payload:
        spec = next((s for s in PAYLOAD if s.name == p.name), None)
        if spec is None:
            continue
        c = 0.5 * (p.lo + p.hi)
        dx, dy = c[0] - spec.pref[0] * c0, c[1] - spec.pref[1]
        turned = spec.rotatable and abs((p.hi[0] - p.lo[0]) - spec.size_mm[0]) > 1e-6
        lay.notes.append(f"{p.name:<18s} at ({c[0]:5.0f}, {c[1] + 0.0:+4.0f}) mm: {dx + 0.0:+5.0f} mm along, {dy + 0.0:+4.0f} mm across"
                         + (", turned 90 deg" if turned else ""))
    lay.notes.append(f"battery centre {battery_x:.0f} mm (weights.py {base_x:.0f} mm) keeps the CG on "
                     f"{bu.choices.cg_target_x_m*1000:.0f} mm")
    return lay


# --------------------------------------------------------------------------
# Print sections
# --------------------------------------------------------------------------


def _usable(st: CadSettings) -> np.ndarray:
    return np.sort(np.asarray(st.printer_mm) - 2 * st.printer_margin_mm)


def _z_extent(pod: PodShape, x0: float, x1: float) -> tuple[float, float]:
    zs = [pod.z_out(x, y) for x in np.linspace(x0, x1, 40) for y in np.linspace(0, pod.root_y, 12)]
    zs = [z for z in zs if z is not None]
    return min(z[0] for z in zs), max(z[1] for z in zs)


def keepouts_x(lay: Layout) -> list[tuple[float, float, str]]:
    """x ranges a chordwise cut must not fall in (with the keep-out margin added)."""
    st, k = lay.st, lay.st.cut_keepout_mm
    r_boss = st.sleeve_od_mm / 2 + st.boss_wall_mm
    a, b = lay.point_on_spar(-st.sleeve_len_mm), lay.point_on_spar(0)
    out = [
        (lay.inner_rib["x0"], lay.inner_rib["x1"], "inner spar rib"),
        (min(a[0], b[0]) - r_boss, max(a[0], b[0]) + r_boss, "sleeve and bosses"),
        (lay.root_rib["x0"], lay.root_rib["x1"], "root rib"),
        (lay.strip["x"] - st.strip_w_mm, lay.strip["x"] + st.strip_w_mm, "carbon strip lands"),
        (lay.camera["block"]["x0"], lay.camera["block"]["x1"], "camera mount"),
        (lay.nacelle["tongue"]["x0"], lay.pod.c0 + 50, "nacelle and its tongue"),
        (-50, lay.st.nose_len_mm + lay.st.joint_lap_mm, "nose cap joint"),
    ]
    return [(x0 - k, x1 + k, n) for x0, x1, n in out]


def _piece_dims(lay: Layout, x0: float, x1: float, fwd_spigot: bool) -> np.ndarray:
    st, pod = lay.st, lay.pod
    zlo, zhi = _z_extent(pod, x0, x1)
    lx = (x1 - x0) + (st.joint_lap_mm if fwd_spigot else 0)
    ly = pod.root_y + st.joint_lap_mm  # the right half carries the centreline spigot
    return np.sort([lx, ly, zhi - zlo])


def choose_cuts(lay: Layout, max_pieces: int = 4) -> list[float]:
    """Fewest chordwise cuts so every half-section fits the printer, each cut clear of the keep-outs,
    then the cut set with the most room to spare (printer margin and distance to keep-outs)."""
    st, pod = lay.st, lay.pod
    use = _usable(st)
    x_start, x_end = st.nose_len_mm, pod.c0
    ko = keepouts_x(lay)
    grid = np.arange(x_start + 30, x_end - 30, 1.0)
    free = [x for x in grid if not any(a <= x <= b for a, b, _ in ko)]

    def score(cuts):
        edges = [x_start, *cuts, x_end]
        spare = []
        for i in range(len(edges) - 1):
            dims = _piece_dims(lay, edges[i], edges[i + 1], fwd_spigot=True)  # into the nose cap or the piece ahead
            spare.append(float(np.min(use - dims)))
        if min(spare) < 0:
            return None
        clear = min(min(abs(c - a), abs(c - b)) for c in cuts for a, b, _ in ko) if cuts else 99
        return min(spare) + 0.5 * min(clear, 20)

    if score([]) is not None:
        return []
    free = np.array(free)
    for k in range(1, max_pieces):
        best, best_s = None, -1e9
        cand = free[::2] if k > 1 else free
        if k == 1:
            combos = ([c] for c in cand)
        elif k == 2:
            combos = ([a, b] for i, a in enumerate(cand) for b in cand[i + 1:] if b - a > 40)
        else:
            combos = ([a, b, c] for i, a in enumerate(cand[::3]) for b in cand[cand > a + 40][::3]
                      for c in cand[cand > b + 40][::3])
        for cuts in combos:
            s = score(list(cuts))
            if s is not None and s > best_s:
                best, best_s = list(map(float, cuts)), s
        if best is not None:
            return best
    raise ValueError("no set of chordwise cuts fits the printer: every legal cut leaves a piece over "
                     f"{use[0]:.0f} mm (keep-outs: " + "; ".join(f"{n} {a:.0f}-{b:.0f}" for a, b, n in ko) + ")")


def print_sections(lay: Layout) -> list[dict]:
    st, pod = lay.st, lay.pod
    edges = [st.nose_len_mm, *lay.cuts_x, pod.c0]
    names = ["front", "rear"] if len(edges) == 3 else [f"s{i+1}" for i in range(len(edges) - 1)]
    out = []
    for i in range(len(edges) - 1):
        for side in ("L", "R"):
            x0, x1 = edges[i], edges[i + 1]
            zlo, zhi = _z_extent(pod, x0, x1)
            lx = (x1 - x0) + st.joint_lap_mm  # spigot into the nose cap or the piece ahead
            ly = pod.root_y + (st.joint_lap_mm if side == "R" else 0)
            out.append(dict(name=f"pod_{names[i]}_{side}", x0=x0, x1=x1, side=side,
                            bbox=[round(lx, 1), round(ly, 1), round(zhi - zlo, 1)]))
    return out


def place_hatches(lay: Layout) -> list[dict]:
    """Front hatch: battery bay, ahead of the roof strip, inside the first section.
    Rear hatch: electronics, behind the roof strip, ahead of the nacelle."""
    st, pod = lay.st, lay.pod
    k = st.cut_keepout_mm
    strip0 = lay.strip["x"] - st.strip_w_mm / 2 - st.land_mm - 8
    strip1 = lay.strip["x"] + st.strip_w_mm / 2 + 8
    hw = st.hatch_width_mm

    def roomy(x):
        return min(pod.inside_height(x, y) for y in (-hw / 2, 0, hw / 2)) >= st.min_hatch_inside_mm

    # front: from the nose joint to the strip, skipping a cut
    f0 = st.nose_len_mm + st.joint_lap_mm + k
    f1 = strip0
    for c in lay.cuts_x:
        if f0 < c < f1:
            f1 = c - k
    while not roomy(f0) and f0 < f1:
        f0 += 1
    # rear: from the strip to the nacelle tongue's front bolt or where the pod gets too thin
    r0 = strip1
    for c in lay.cuts_x:
        if r0 - k < c < r0 + 60:
            r0 = c + k
    r1 = lay.nacelle["x_a"] - 8
    while not roomy(r1) and r1 > r0:
        r1 -= 1
    for c in lay.cuts_x:
        if r0 < c < r1:
            r1 = c - k
    out = []
    for name, x0, x1 in (("front", f0, f1), ("rear", r0, r1)):
        out.append(dict(name=name, x0=float(x0), x1=float(x1), half_w=hw / 2, magnets=_hatch_magnets(lay, x0, x1, hw / 2),
                        hook=dict(x=float(x0), half_w=15.0)))
    return out


def _magnet_box(lay: Layout, mx: float, my: float, name: str) -> Box3:
    zr = lay.pod.z_out(mx, my)[1]
    r = lay.st.hatch_magnet_d_mm / 2 + 2
    return Box3(name, np.array([mx - r, my - r, zr - lay.st.wall_mm - 7]), np.array([mx + r, my + r, zr]))


def _hatch_magnets(lay: Layout, x0: float, x1: float, hw: float) -> list[tuple[float, float]]:
    """Two magnets, mirrored, at the back of the hatch if the payload below allows, else further along the sides."""
    st = lay.st
    xa = x1 - st.hatch_rim_mm / 2 - 1
    cands = [(xa, hw - 12), (xa, hw - 24), (xa, 12.0)]
    cands += [(x, hw - 4.0) for x in np.linspace(xa - 10, x0 + 0.4 * (x1 - x0), 8)]
    # keep the tongue bolts reachable from above
    access = [Box3("bolt access", np.array([bx - 7, by - 7, -99]), np.array([bx + 7, by + 7, 99]))
              for bx, by in lay.nacelle["bolts"]]
    for mx, my in cands:
        boxes = [_magnet_box(lay, mx, s * my, "m") for s in (-1, 1)]
        if not any(b.overlaps(p, st.payload_clear_mm) for b in boxes for p in [*lay.payload, *access]):
            return [(float(mx), -float(my)), (float(mx), float(my))]
    return [(float(xa), -(hw - 12)), (float(xa), hw - 12)]


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def spar_cover(lay: Layout, n: int = 60) -> tuple[float, float, float]:
    """Smallest EPP cover over the stepped tube along the panel (top, bottom, s at the worst point), mm."""
    g = lay.geom
    L = np.linalg.norm(lay.spar_tip - lay.spar_root)
    worst = (1e9, 1e9, 0.0)
    for s in np.linspace(0, L, n):
        p = lay.point_on_spar(s)
        eta = (p[1] / 1000 - g.root_y_m) / g.outer_semispan_m
        le, c, tw = g._outer_station(min(max(eta, 0), 1))
        sec = outer_section3d(g, min(max(eta, 0), 1), 121)
        # local section z at the spar's x (upper / lower), section is twisted so search the loop
        m = (len(sec) + 1) // 2
        up, lo = sec[:m][::-1], sec[m - 1:]
        zu = np.interp(p[0], up[:, 0], up[:, 2])
        zl = np.interp(p[0], lo[:, 0], lo[:, 2])
        od = next((t["od_mm"] for t in lay.spar_layout if t["from_mm"] - 1e-6 <= s <= t["to_mm"] + 1e-6),
                  lay.spar_layout[-1]["od_mm"])
        top, bot = zu - p[2] - od / 2, p[2] - zl - od / 2
        if min(top, bot) < min(worst[0], worst[1]):
            worst = (float(top), float(bot), float(s))
    return worst


def folded_prop_pitch_deg(lay: Layout) -> float:
    """Nose-up attitude, sitting on the skid's aft end, at which the folded prop touches the ground."""
    st, sk, nc = lay.st, lay.skid, lay.nacelle
    # pivot: aft bottom corner of the skid rails
    px, pz = sk["x1"], sk["z_bottom"]
    blade = 0.5 * 10 * 25.4 - 20  # folded blade length behind the hub
    hub_x = nc["x_fw"] + 34 + 10
    pts = [(nc["x_fw"], nc["z"] - nc["r_b"]), (hub_x + blade, nc["z"] - st.folded_prop_drop_mm)]
    return min(math.degrees(math.atan2(z - pz, x - px)) for x, z in pts)


def run_checks(lay: Layout) -> list[Check]:
    st, pod, g = lay.st, lay.pod, lay.geom
    ch: list[Check] = []
    use = _usable(st)
    for s in lay.sections:
        spare = float(np.min(use - np.sort(s["bbox"])))
        ch.append(Check(f"{s['name']} fits the printer ({' x '.join(f'{v:.0f}' for v in s['bbox'])} mm)", spare, 0, ">=",
                        note=f"spare inside {use[0]:.0f} mm (printer less {st.printer_margin_mm:.0f} mm a side)"))
    ko = keepouts_x(lay)
    for c in lay.cuts_x:
        clear = min(min(abs(c - a), abs(c - b)) for a, b, _ in ko)
        ch.append(Check(f"cut at x = {c:.0f} mm clear of ribs, strips, camera, nacelle", clear, 0, ">=",
                        note="beyond the keep-out margin"))
    top, bot, s = spar_cover(lay)
    ch.append(Check("EPP over the stepped spar, tightest point (top / bottom)", min(top, bot), 2.0, ">=",
                    note=f"top {top:.1f}, bottom {bot:.1f} mm at {s:.0f} mm out (M2.2 asks for 2 mm)"))
    # sleeve buried in the pod: boss inside the outer skin at the inner rib
    c = lay.inner_rib["centre"]
    z = pod.z_in(c[0], c[1])
    r_boss = st.sleeve_od_mm / 2 + st.boss_wall_mm
    ch.append(Check("inner rib boss inside the pod (room above / below)", min(z[1] - c[2] - r_boss, c[2] - r_boss - z[0]), 0.5, ">="))
    z = pod.z_in(lay.spar_root[0], pod.root_y - st.end_plate_mm)
    ch.append(Check("sleeve wall inside the root face (room above / below)",
                    min(z[1] - lay.spar_root[2], lay.spar_root[2] - z[0]) - st.sleeve_od_mm / 2, 1.0, ">="))
    # magnets inside the root section
    for i, m in enumerate(lay.magnets):
        z = pod.z_in(m[0], pod.root_y - 1)
        ch.append(Check(f"panel catch magnet {i+1} inside the root face", min(z[1] - m[2], m[2] - z[0]) - (st.magnet_d_mm / 2 + 2), 0, ">="))
    for h in lay.hatches:
        L = h["x1"] - h["x0"]
        need = lay.battery["L"] + 25 if h["name"] == "front" else 90
        ch.append(Check(f"{h['name']} hatch opening length", L, need, ">=",
                        note="pack length + 25 mm to tilt it in" if h["name"] == "front" else "room to reach the electronics"))
        clear = min(abs(h["x0"] - lay.strip["x"]), abs(h["x1"] - lay.strip["x"])) - st.strip_w_mm / 2
        ch.append(Check(f"{h['name']} hatch clear of the roof strip (M2.2: hatch must not cut it)", clear, 5, ">="))
    # battery: under the roof everywhere it can slide, and inside the tray
    b = lay.battery
    worst = min(pod.z_in(x, y)[1] - (b["z0"] + b["H"]) for x in np.linspace(b["x_min"] - b["L"] / 2, b["x_max"] + b["L"] / 2, 30)
                for y in (-b["W"] / 2, 0, b["W"] / 2))
    ch.append(Check("battery clears the roof over its whole tray travel", worst, 3.0, ">="))
    under_strip = pod.z_in(lay.strip["x"], 0)[1] - st.land_mm - st.strip_t_mm - (b["z0"] + b["H"])
    ch.append(Check("battery slides under the roof strip", under_strip, 3.0, ">="))
    cam = lay.camera
    fh = next(h for h in lay.hatches if h["name"] == "front")
    blk = cam["block"]
    if blk["x0"] < fh["x1"] and blk["x1"] > fh["x0"]:
        cam_top = cam["board"][2] + 12
        room = min(pod.z_in(x, 0)[1] for x in np.linspace(blk["x0"], blk["x1"], 8)) - cam_top - b["H"]
        ch.append(Check("pack passes over the camera mount on its way in through the front hatch", room, 5.0, ">="))
    ch.append(Check("camera lens recessed above the outer skin", cam["recess"], 3.0, ">=",
                    note="belly landings: the lens never touches the dirt"))
    # skid rails out of the camera view (across-track half angle + margin)
    ay = math.radians(st.camera_fov_deg[1] / 2 + st.fov_margin_deg)
    dz = cam["lens_tip"][2] - lay.skid["z_bottom"]
    ch.append(Check("skid rails outside the camera view", lay.skid["y0"] - dz * math.tan(ay) - st.lens_r_mm, 2.0, ">="))
    ch.append(Check("fins clear of the ground sitting level on the skid", g.fin_ground_clearance_m * 1000, 20.0, ">="))
    ch.append(Check("nose-up attitude before the folded prop touches", folded_prop_pitch_deg(lay), st.min_prop_pitch_deg, ">=",
                    unit="deg", note="sitting on the aft end of the skid rails"))
    # payload: inside the pod and clear of structure and each other
    worst_roof, worst_name = 1e9, ""
    for p in lay.payload:
        for x in (p.lo[0], p.hi[0]):
            for y in (p.lo[1], p.hi[1]):
                z = pod.z_in(x, y)
                r = -99 if z is None else z[1] - p.hi[2]
                if r < worst_roof:
                    worst_roof, worst_name = r, p.name
    ch.append(Check("payload under the roof (tightest item)", worst_roof, st.payload_clear_mm, ">=", note=worst_name))
    clashes = []
    mag_boxes = [_magnet_box(lay, mx, my, f"{h['name']} hatch magnet boss") for h in lay.hatches for mx, my in h["magnets"]]
    for i, a in enumerate(lay.payload):
        for mb in mag_boxes:
            if a.overlaps(mb):
                clashes.append(f"{a.name} / {mb.name}")
        for b2 in lay.payload[i + 1:]:
            if a.overlaps(b2):
                clashes.append(f"{a.name} / {b2.name}")
        for s2 in lay.structure_boxes:
            if a.name == "battery" and s2.name == "battery tray rails":
                continue
            if a.overlaps(s2, st.payload_clear_mm):
                clashes.append(f"{a.name} / {s2.name}")
    ch.append(Check("payload clashes with structure or other items", len(clashes), 0, "<=", unit="count",
                    note="; ".join(clashes) if clashes else "none"))
    return ch


# --------------------------------------------------------------------------
# The recipe
# --------------------------------------------------------------------------


def _mirror_y(p) -> np.ndarray:
    return np.asarray(p, float) * np.array([1, -1, 1])


def build_recipe(lay: Layout) -> Recipe:
    st, pod, g = lay.st, lay.pod, lay.geom
    R = Recipe()
    BIG = 1000.0
    w = st.wall_mm

    # ---- lofts: outside, inside (cavity) and the offsets the joints and nose need
    def pod_loft(name, inset):
        return R.loft(name, [pod.loop3d(y, inset) for y in pod.lofts_y(inset)])

    outer = pod_loft("pod_outer", 0.0)
    cav = pod_loft("pod_cavity", w)
    ins_clear = pod_loft("pod_inset_clear", w + st.joint_clear_mm)
    ins_band = pod_loft("pod_inset_band", w + st.joint_band_mm)
    nose_in = pod_loft("nose_inset", st.nose_wall_mm)
    nose_clear = pod_loft("nose_inset_clear", st.nose_wall_mm + st.joint_clear_mm)
    nose_band = pod_loft("nose_inset_band", st.nose_wall_mm + st.joint_band_mm)
    shell = R.diff("pod_shell", outer, [cav])

    def shifted(t):  # the cavity moved down (t > 0: roof band) or up (t < 0: floor band)
        key = f"cav_dz_{t:+.1f}".replace(".", "p").replace("+", "m").replace("-", "p")
        if key not in R.solids:
            R.move(key, cav, (0, 0, -t))
        return key

    core = [shell]
    cuts = []

    # ---- spar joint, both sides --------------------------------------------------
    r_sl = st.sleeve_od_mm / 2
    r_boss = r_sl + st.boss_wall_mm
    for sgn, side in ((1, "R"), (-1, "L")):
        def P(p):
            return np.asarray(p) if sgn > 0 else _mirror_y(p)

        on = lay.point_on_spar
        ir = lay.inner_rib
        y0, y1 = sorted([sgn * ir["y0"], sgn * ir["y1"]])
        rib_i = R.box(f"inner_rib_box_{side}", (ir["x0"], y0, -BIG), (ir["x1"], y1, BIG))
        core.append(R.inter(f"inner_rib_{side}", [rib_i, outer]))
        b0, b1 = ir["boss"]
        core.append(R.cyl(f"inner_boss_{side}", P(on(b0)), P(on(b1)), r_boss))
        rr = lay.root_rib
        y0, y1 = sorted([sgn * rr["y0"], sgn * rr["y1"]])
        rib_r = R.box(f"root_rib_box_{side}", (rr["x0"], y0, -BIG), (rr["x1"], y1, BIG))
        wd = rr["window"]
        win = R.box(f"root_rib_window_{side}", (wd["x0"], y0 - 1, wd["z0"]), (wd["x1"], y1 + 1, wd["z1"]))
        rib_r2 = R.inter(f"root_rib_raw_{side}", [rib_r, outer])
        core.append(R.diff(f"root_rib_{side}", rib_r2, [win]))
        b0, b1 = rr["boss"]
        boss = R.cyl(f"root_boss_raw_{side}", P(on(b0)), P(on(b1)), r_boss)
        core.append(R.inter(f"root_boss_{side}", [boss, outer]))
        for i, m in enumerate(lay.magnets):
            mb = R.cyl(f"magnet_boss_raw_{side}{i}", P(m - [0, 8.0, 0]), P(m + [0, 1.0, 0]), st.magnet_d_mm / 2 + 3)
            core.append(R.inter(f"magnet_boss_{side}{i}", [mb, outer]))
            cuts.append(R.cyl(f"magnet_pocket_{side}{i}", P(m - [0, st.magnet_t_mm + 0.2, 0]), P(m + [0, 2.0, 0]),
                              st.magnet_d_mm / 2 + st.pocket_clear_mm))
        cuts.append(R.cyl(f"sleeve_bore_{side}", P(on(-st.sleeve_len_mm - 5)), P(on(8)), r_sl + st.sleeve_bond_gap_mm))
        pp = lay.pin["p"]
        cuts.append(R.cyl(f"pin_hole_{side}", P(pp - lay.spar_dir * st.pin_depth_mm), P(pp + lay.spar_dir * 5),
                          st.pin_d_mm / 2 + st.pin_clear_mm))
        sh = lay.servo_hole
        cuts.append(R.cyl(f"servo_lead_hole_{side}", P(sh - [0, 12, 0]), P(sh + [0, 3, 0]), st.servo_lead_hole_mm / 2))

    # ---- carbon strip lands (floor and roof) and their slots through the inner ribs ----
    sx, sh_ = lay.strip["x"], lay.strip["half"]
    land_slab = R.box("strip_land_slab", (sx - st.strip_w_mm / 2 - 3, -sh_ - 3, -BIG), (sx + st.strip_w_mm / 2 + 3, sh_ + 3, BIG))
    roof_band = R.diff("roof_land_band", outer, [shifted(st.land_mm)])
    floor_band = R.diff("floor_land_band", outer, [shifted(-st.land_mm)])
    core.append(R.inter("roof_land", [land_slab, roof_band]))
    core.append(R.inter("floor_land", [land_slab, floor_band]))
    slot_slab = R.box("strip_slot_slab", (sx - st.strip_w_mm / 2 - 0.2, -sh_ - 1, -BIG), (sx + st.strip_w_mm / 2 + 0.2, sh_ + 1, BIG))
    t1, t2 = st.land_mm, st.land_mm + st.strip_t_mm + 0.1
    roof_slot_band = R.diff("roof_slot_band", shifted(t1), [shifted(t2)])
    floor_slot_band = R.diff("floor_slot_band", shifted(-t1), [shifted(-t2)])
    cuts.append(R.inter("roof_strip_slot", [slot_slab, roof_slot_band]))
    cuts.append(R.inter("floor_strip_slot", [slot_slab, floor_slot_band]))

    # ---- battery tray: rails, stop slots, strap tunnels -------------------------
    tr = lay.tray
    rails = []
    for s in (-1, 1):
        y = s * tr["rail_y"]
        rb = R.box(f"tray_rail_box_{'L' if s < 0 else 'R'}", (tr["x0"], y - st.rail_w_mm / 2, -BIG),
                   (tr["x1"], y + st.rail_w_mm / 2, tr["rail_top"]))
        rails.append(R.inter(f"tray_rail_{'L' if s < 0 else 'R'}", [rb, outer]))
    # keep the rails out of the joint spigots
    gaps = [R.box(f"tray_joint_gap_{i}", (c - st.joint_lap_mm - 1.5, -BIG, -BIG), (c + st.joint_lap_mm + 1.5, BIG, BIG))
            for i, c in enumerate(lay.cuts_x) if tr["x0"] - 10 < c < tr["x1"] + 10]
    rail_u = R.union("tray_rails_raw", rails)
    core.append(R.diff("tray_rails", rail_u, gaps) if gaps else rail_u)
    slot_boxes = [R.box(f"tray_slot_{i}", (x - st.slot_w_mm / 2, -tr["rail_y"] - st.rail_w_mm, tr["rail_top"] - st.slot_depth_mm),
                        (x + st.slot_w_mm / 2, tr["rail_y"] + st.rail_w_mm, tr["rail_top"] + 1)) for i, x in enumerate(tr["slots"])]
    cuts.append(R.union("tray_slots", slot_boxes))
    for i, x in enumerate(tr["straps"]):
        fl = max(pod.z_in(xx, tr["rail_y"])[0] for xx in (x - st.strap_w_mm / 2, x + st.strap_w_mm / 2))
        cuts.append(R.box(f"strap_tunnel_{i}", (x - st.strap_w_mm / 2, -tr["rail_y"] - st.rail_w_mm, fl + 0.6),
                          (x + st.strap_w_mm / 2, tr["rail_y"] + st.rail_w_mm, fl + 2.6)))

    # ---- hatches: rim + ledge on the pod, lid with a lip, front hook, aft magnets ----
    lids = []
    for h in lay.hatches:
        n = h["name"]
        x0, x1, hw = h["x0"], h["x1"], h["half_w"]

        def hb(tag, grow, z0=-BIG, z1=BIG):
            return R.box(f"hatch_{n}_{tag}", (x0 - grow, -hw - grow, z0), (x1 + grow, hw + grow, z1))

        opening = hb("open", 0)
        rim_out = hb("rim_out", st.hatch_rim_mm)
        rim = R.inter(f"hatch_{n}_rim", [R.diff(f"hatch_{n}_rim_ring", rim_out, [opening]),
                                         R.diff(f"hatch_{n}_rim_band", outer, [shifted(3.7)])])
        ledge_in = hb("ledge_in", -(st.hatch_lip_mm + st.hatch_clear_mm + 3))
        hook_notch = R.box(f"hatch_{n}_hook_notch", (x0 - 1, -h["hook"]["half_w"] - 1, -BIG), (x0 + 3.5, h["hook"]["half_w"] + 1, BIG))
        ledge = R.inter(f"hatch_{n}_ledge", [R.diff(f"hatch_{n}_ledge_ring", hb("ledge_out", 1.0), [ledge_in, hook_notch]),
                                             R.diff(f"hatch_{n}_ledge_band", shifted(2.2), [shifted(3.7)])])
        core += [rim, ledge]
        cuts.append(R.inter(f"hatch_{n}_cut", [opening, R.diff(f"hatch_{n}_cut_band", outer, [shifted(2.2)])]))
        # lid
        z_roof_min = min(pod.z_out(x, y)[1] for x in np.linspace(x0, x1, 12) for y in (-hw, 0, hw))
        lid_box = hb("lidbox", -st.hatch_clear_mm, z0=z_roof_min - 8.0)  # top skin only, never the floor
        lid_skin = R.inter(f"hatch_{n}_lid_skin", [lid_box, shell])
        lip_ring = R.diff(f"hatch_{n}_lip_ring", lid_box, [hb("lip_in", -(st.hatch_clear_mm + st.hatch_lip_mm))])
        lip = R.inter(f"hatch_{n}_lip", [lip_ring, R.diff(f"hatch_{n}_lip_band", outer, [shifted(2.0)])])
        hk = h["hook"]
        hook_v = R.inter(f"hatch_{n}_hook_v", [R.box(f"hatch_{n}_hook_v_box", (x0 + st.hatch_clear_mm, -hk["half_w"], -BIG),
                                                     (x0 + 3.0, hk["half_w"], BIG)),
                                               R.diff(f"hatch_{n}_hook_v_band", outer, [shifted(5.4)])])
        hook_t = R.inter(f"hatch_{n}_hook_t", [R.box(f"hatch_{n}_hook_t_box", (x0 - 5.0, -hk["half_w"], -BIG),
                                                     (x0 + 3.0, hk["half_w"], BIG)),
                                               R.diff(f"hatch_{n}_hook_t_band", shifted(3.9), [shifted(5.4)])])
        lid_parts = [lid_skin, lip, hook_v, hook_t]
        lid_cuts = []
        for j, (mx, my) in enumerate(h["magnets"]):
            zr = pod.z_out(mx, my)[1]
            # pod side: boss under the ledge, pocket from the top
            core.append(R.inter(f"hatch_{n}_mboss_{j}", [R.cyl(f"hatch_{n}_mboss_raw_{j}", (mx, my, zr - w - 3.0 - 4.0),
                                                               (mx, my, zr - w - 3.0), st.hatch_magnet_d_mm / 2 + 2.0), cav]))
            cuts.append(R.cyl(f"hatch_{n}_mpocket_{j}", (mx, my, zr - 3.0 - st.hatch_magnet_t_mm - 0.2), (mx, my, zr - 2.9),
                              st.hatch_magnet_d_mm / 2 + st.pocket_clear_mm))
            # lid side: boss down to the ledge, pocket from below
            lid_parts.append(R.inter(f"hatch_{n}_lboss_{j}", [R.cyl(f"hatch_{n}_lboss_raw_{j}", (mx, my, zr - 2.8),
                                                                    (mx, my, zr), st.hatch_magnet_d_mm / 2 + 1.6), lid_box, outer]))
            lid_cuts.append(R.cyl(f"hatch_{n}_lpocket_{j}", (mx, my, zr - 2.9), (mx, my, zr - 2.8 + st.hatch_magnet_t_mm + 0.2),
                                  st.hatch_magnet_d_mm / 2 + st.pocket_clear_mm))
        lid = R.union(f"hatch_{n}_lid_raw", lid_parts)
        lids.append((n, R.diff(f"hatch_{n}_lid", lid, lid_cuts)))

    # ---- camera window ------------------------------------------------------------
    cam = lay.camera
    wnd = cam["window"]
    cuts.append(R.box("camera_window", (wnd["x0"], -wnd["y"], -BIG), (wnd["x1"], wnd["y"], cam["floor_in"] + 0.5)))

    # ---- nacelle: hole through the pod TE, bolts through the floor --------------------
    nc = lay.nacelle
    cuts.append(R.cyl("nacelle_hole", (nc["x_a"], 0, nc["z"]), (nc["x_fw"] + 1, 0, nc["z"]), nc["r_a"] + 0.3, nc["r_b"] + 0.3))
    for i, (bx, by) in enumerate(nc["bolts"]):
        cuts.append(R.cyl(f"tongue_bolt_{i}", (bx, by, -60), (bx, by, 30), st.bolt_hole_d_mm / 2))

    # ---- joints between pieces ---------------------------------------------------------
    edges = [st.nose_len_mm, *lay.cuts_x, pod.c0 + 50]
    gap = st.centre_band_gap_mm
    excl = sorted([(st.nose_len_mm - gap, st.nose_len_mm + st.joint_lap_mm + gap),
                   *[(c - st.joint_lap_mm - gap, c + st.joint_lap_mm + gap) for c in lay.cuts_x],
                   (sx - st.strip_w_mm / 2 - 3 - gap, sx + st.strip_w_mm / 2 + 3 + gap),
                   (cam["block"]["x0"] - gap, cam["block"]["x1"] + gap),
                   (nc["tongue"]["x0"] - gap, BIG)])
    centre_runs, x = [], st.nose_len_mm
    for a, b in excl:
        if a > x + 10:
            centre_runs.append((x, a))
        x = max(x, b)

    def band(name, lo, hi, own: bool, mate_clear, mate_band):
        slab = R.box(f"{name}_slab", lo, hi)
        if own:
            return R.inter(name, [slab, R.diff(f"{name}_ring", outer, [mate_band])])
        return R.inter(name, [slab, R.diff(f"{name}_ring", mate_clear, [mate_band])])

    piece_bands: dict[str, list[str]] = {}

    def add_band(piece, s):
        piece_bands.setdefault(piece, []).append(s)

    names = ["front", "rear"] if len(edges) == 3 else [f"s{i+1}" for i in range(len(edges) - 1)]
    L = st.joint_lap_mm
    for side, (ya, yb) in (("L", (-BIG, 0.0)), ("R", (0.0, BIG))):
        # nose spigot: front piece reaches into the TPU cap
        add_band(f"{names[0]}_{side}", band(f"nose_own_{side}", (st.nose_len_mm, ya, -BIG), (st.nose_len_mm + L, yb, BIG), True,
                                            nose_clear, nose_band))
        add_band(f"{names[0]}_{side}", band(f"nose_mate_{side}", (st.nose_len_mm - L, ya, -BIG), (st.nose_len_mm + 1, yb, BIG), False,
                                            nose_clear, nose_band))
        for i, c in enumerate(lay.cuts_x):  # chordwise: the aft piece carries the spigot
            add_band(f"{names[i+1]}_{side}", band(f"cut{i}_own_{side}", (c, ya, -BIG), (c + L, yb, BIG), True, ins_clear, ins_band))
            add_band(f"{names[i+1]}_{side}", band(f"cut{i}_mate_{side}", (c - L, ya, -BIG), (c + 1, yb, BIG), False, ins_clear, ins_band))
    for j, (a, b) in enumerate(centre_runs):  # centreline: the right half carries the spigot
        for i in range(len(edges) - 1):
            lo, hi = max(a, edges[i]), min(b, edges[i + 1])
            if hi - lo > 5:
                add_band(f"{names[i]}_R", band(f"ctr{j}_{i}_own", (lo, 0, -BIG), (hi, L, BIG), True, ins_clear, ins_band))
                add_band(f"{names[i]}_R", band(f"ctr{j}_{i}_mate", (lo, -L, -BIG), (hi, 1, BIG), False, ins_clear, ins_band))

    core_u = R.union("pod_core", core)
    cut_u = R.union("pod_cuts", cuts)
    for i in range(len(edges) - 1):
        for side, (ya, yb) in (("L", (-BIG, 0.0)), ("R", (0.0, BIG))):
            key = f"{names[i]}_{side}"
            region = R.box(f"region_{key}", (edges[i], ya, -BIG), (edges[i + 1], yb, BIG))
            piece = R.inter(f"piece_{key}_core", [core_u, region])
            piece = R.union(f"piece_{key}_banded", [piece, *piece_bands.get(key, [])])
            solid = R.diff(f"pod_{key}", piece, [cut_u])
            R.part(f"pod_{key}", solid, "LW-ASA", orient="cut face down on the bed; tree supports under the roof",
                   note="bond with epoxy; spigot band slides inside the neighbour")

    # ---- nose cap (TPU) -------------------------------------------------------------
    nose_region = R.box("nose_region", (-BIG, -BIG, -BIG), (st.nose_len_mm, BIG, BIG))
    cap = R.diff("nose_cap", R.inter("nose_cap_raw", [outer, nose_region]), [nose_in])
    R.part("nose_cap", cap, "TPU 95A", orient="open end down", note="push on over the spigot; a strip of tape holds it")

    # ---- lids ----------------------------------------------------------------------
    for n, lid in lids:
        R.part(f"hatch_{n}_lid", lid, "LW-ASA", orient="outside up, supports under the lip",
               note="hook under the front rim, 2 magnets at the back" + ("; GPS on the underside" if n == "front" else ""))

    # ---- nacelle (separate ASA part) ------------------------------------------------
    t = nc["tongue"]
    cone = R.cyl("nacelle_cone", (nc["x_a"], 0, nc["z"]), (nc["x_fw"], 0, nc["z"]), nc["r_a"], nc["r_b"])
    bore = R.cyl("nacelle_bore", (nc["x_a"] - 1, 0, nc["z"]), (nc["x_te"], 0, nc["z"]),
                 nc["r_a"] - st.nacelle_wall_mm, nc["r_b"] - st.nacelle_wall_mm - 0.5)
    tongue_box = R.box("tongue_box", (t["x0"], -t["w"] / 2, -BIG), (t["x1"], t["w"] / 2, BIG))
    tongue = R.inter("tongue", [tongue_box, R.diff("tongue_band", cav, [shifted(-t["t"])])])
    tongue = R.diff("tongue_trim", tongue, [bore])
    holes = [R.cyl(f"motor_hole_{i}", (nc["x_te"] - 1, py, nc["z"] + pz), (nc["x_fw"] + 1, py, nc["z"] + pz), st.bolt_hole_d_mm / 2)
             for i, (py, pz) in enumerate([(st.motor_pattern_mm[0] / 2, 0), (-st.motor_pattern_mm[0] / 2, 0),
                                            (0, st.motor_pattern_mm[1] / 2), (0, -st.motor_pattern_mm[1] / 2)])]
    holes.append(R.cyl("motor_shaft_hole", (nc["x_te"] - 1, 0, nc["z"]), (nc["x_fw"] + 1, 0, nc["z"]), 5.0))
    holes += [R.cyl(f"nacelle_bolt_{i}", (bx, by, -60), (bx, by, 30), st.bolt_hole_d_mm / 2) for i, (bx, by) in enumerate(nc["bolts"])]
    nac = R.diff("nacelle", R.union("nacelle_raw", [R.diff("nacelle_shell", cone, [bore]), tongue]), holes)
    R.part("nacelle", nac, "ASA (solid, not foamed)", orient="firewall down",
           note="slides in from behind; 3 M3 bolts through the tongue and the pod floor, nyloc nuts on top")

    # ---- camera mount -------------------------------------------------------------
    blk = cam["block"]
    mb = R.box("cam_block_box", (blk["x0"], -blk["y"], cam["floor_in"] - 10), (blk["x1"], blk["y"], cam["board"][2] + 20))
    mount = R.inter("cam_block", [mb, cav])
    big = R.box("cam_top_cut_raw", (-100, -100, 0), (100, 100, 100))
    top_cut = R.move("cam_top_cut", R.rotate("cam_top_cut_rot", big, (0, 1, 0), st.camera_tilt_deg), cam["board"])
    ax = cam["axis"]
    lens = R.cyl("cam_lens_hole", cam["board"] - 30 * ax, cam["board"] + 30 * ax, 6.5)
    u = np.array([math.cos(math.radians(st.camera_tilt_deg)), 0, -math.sin(math.radians(st.camera_tilt_deg))])
    v = np.array([0.0, 1.0, 0.0])
    hx, hy = st.camera_holes_mm
    ch = [R.cyl(f"cam_screw_{i}", cam["board"] + a * hx / 2 * u + b * hy / 2 * v + 6 * ax,
                cam["board"] + a * hx / 2 * u + b * hy / 2 * v - 1 * ax, 0.9)
          for i, (a, b) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)])]
    R.part("camera_mount", R.diff("camera_mount", mount, [top_cut, lens, *ch]), "LW-ASA",
           orient="tilted face up", note=f"{st.camera_tilt_deg:g} deg wedge; epoxy to the floor over the window; "
                                         "check the Camera Module 3 hole pattern before printing")

    # ---- battery end stops (x2) -----------------------------------------------------
    tw = 2 * tr["rail_y"] + st.rail_w_mm + 4
    stop = R.box("stop_body", (0, -tw / 2, 0), (st.stop_len_mm, tw / 2, 10))
    keys = [R.box(f"stop_key_{s}", ((st.stop_len_mm - st.slot_w_mm + 0.3) / 2, s * tr["rail_y"] - st.rail_w_mm / 2 - 0.2, -st.slot_depth_mm + 0.3),
                  ((st.stop_len_mm + st.slot_w_mm - 0.3) / 2, s * tr["rail_y"] + st.rail_w_mm / 2 + 0.2, 0.01))
            for s in (-1, 1)]
    R.part("battery_stop", R.union("battery_stop", [stop, *keys]), "PETG", count=2, orient="flat",
           note="keys drop into the rail slots at 5 mm steps; the battery strap holds them down")

    # ---- reference parts (not printed): panels, fins, sleeves, skid rails ----------------
    etas = sorted({0.0, g.elevon_start, g.elevon_end, 1.0})
    n = st.n_airfoil
    panel = R.loft("panel_R_ref", [outer_section3d(g, e, n, st.te_thickness_mm) for e in etas])
    R.part("panel_R", panel, "EPP 1.9 lb (hot-wired)", printed=False, note="reference only: cut from the templates")
    root, tip, _ = fin_sections(g, n)
    R.part("fin_R", R.loft("fin_R_ref", [root, tip]), "EPP 1.9 lb (hot-wired)", printed=False,
           note="reference only; pegged and taped on")
    a, b = lay.point_on_spar(-st.sleeve_len_mm), lay.point_on_spar(0)
    sl = R.diff("sleeve_R_ref", R.cyl("sleeve_R_o", a, b, r_sl), [R.cyl("sleeve_R_i", a - lay.spar_dir, b + lay.spar_dir, r_sl - 1)])
    R.part("sleeve_R", sl, "14x12 roll-wrapped carbon tube", printed=False, note="bought; epoxy into the bosses")
    sk = lay.skid
    zt = max(pod.z_out(x, 0)[0] for x in np.linspace(sk["x0"], sk["x1"], 50))
    rail = R.diff("skid_rail_R_ref", R.box("skid_rail_box", (sk["x0"], sk["y0"], sk["z_bottom"]), (sk["x1"], sk["y1"], zt)), [outer])
    R.part("skid_rail_R", rail, "EPP 1.9 lb + 1 mm HDPE wear strip", printed=False, count=2,
           note="velcro to the floor; cut from the skid_rail template")

    # ---- printed hot-wire templates ------------------------------------------------------
    for tname, loop2d, hole_d, extra in template_loops(lay):
        poly = offset_loop(loop2d, -st.hotwire_kerf_mm / 2)
        body = R.prism(f"tpl_{tname}_body", poly, 0, st.template_t_mm)
        hs = [R.cyl(f"tpl_{tname}_hole", (0, 0, -1), (0, 0, st.template_t_mm + 1), hole_d / 2)]
        for k, (ex, ez, ed) in enumerate(extra):
            hs.append(R.cyl(f"tpl_{tname}_x{k}", (ex, ez, -1), (ex, ez, st.template_t_mm + 1), ed / 2))
        # engraved datum: horizontal through the reference hole
        x0, x1 = float(poly[:, 0].min()) - 1, float(poly[:, 0].max()) + 1
        hs.append(R.box(f"tpl_{tname}_datum", (x0, -0.3, st.template_t_mm - 0.6), (x1, 0.3, st.template_t_mm + 1)))
        R.part(f"template_{tname}", R.diff(f"template_{tname}", body, hs), "PETG", orient="flat",
               note="hot-wire template (kerf allowance included); the hole goes on a straight rod to line both ends up")
    return R


def template_loops(lay: Layout) -> list[tuple[str, np.ndarray, float, list]]:
    """(name, 2-D loop in template coordinates, reference hole diameter, [(x, z, d) extra holes]).

    Panel templates are drawn about the spar centre (the hole fits the tube at that end);
    fin templates about the ruling through 30 % chord, mid-thickness."""
    g, st = lay.geom, lay.st
    out = []
    for name, eta, p in (("panel_root", 0.0, lay.spar_root), ("panel_tip", 1.0, lay.spar_tip)):
        sec = outer_section3d(g, eta, st.n_airfoil, st.te_thickness_mm)
        loop = np.column_stack([sec[:, 0] - p[0], sec[:, 2] - p[2]])
        s = 0.0 if eta == 0 else float(np.linalg.norm(lay.spar_tip - lay.spar_root))
        od = next((t["od_mm"] for t in lay.spar_layout if t["from_mm"] - 1e-6 <= s <= t["to_mm"] + 1e-6), lay.spar_layout[-1]["od_mm"])
        extra = []
        if eta == 0:
            extra.append((lay.pin["p"][0] - p[0], lay.pin["p"][2] - p[2], st.pin_d_mm + 0.2))
        out.append((name, loop, od + st.spar_hole_clear_mm, extra))
    r2, t2, _ = fin_template_loops(g, st.n_airfoil)
    for name, loop in (("fin_root", r2), ("fin_tip", t2)):
        m = (len(loop) + 1) // 2
        up, lo = loop[:m][::-1], loop[m - 1:]
        xr = 0.3 * (up[-1, 0] - up[0, 0]) + up[0, 0]
        zr = 0.5 * (np.interp(xr, up[:, 0], up[:, 1]) + np.interp(xr, lo[:, 0], lo[:, 1]))
        out.append((name, loop - [xr, zr], 2.0, []))
    return out


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def to_json(lay: Layout, recipe: Recipe) -> dict:
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return _r(o)
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if isinstance(o, (np.floating, float)):
            return round(float(o), 3)
        if isinstance(o, np.integer):
            return int(o)
        return o

    return dict(
        milestone="M2.3",
        units="mm",
        geometry={k: getattr(lay.geom, k) for k in lay.geom.DESIGN_VARIABLES},
        settings=clean(asdict(lay.st)),
        layout=clean(dict(spar_root=lay.spar_root, spar_tip=lay.spar_tip, spar_dir=lay.spar_dir, spar=lay.spar_layout,
                          inner_rib=lay.inner_rib, root_rib=lay.root_rib, strip=lay.strip, pin=lay.pin,
                          magnets=lay.magnets, cuts_x=lay.cuts_x, sections=lay.sections, hatches=lay.hatches,
                          tray=lay.tray, battery=lay.battery, camera=lay.camera, nacelle=lay.nacelle, skid=lay.skid,
                          fin=lay.fin, payload=[dict(name=p.name, lo=p.lo, hi=p.hi) for p in lay.payload])),
        checks=[dict(name=c.name, value=round(float(c.value), 2), limit=c.limit, sense=c.sense, unit=c.unit, ok=bool(c.ok),
                     note=c.note) for c in lay.checks],
        recipe=recipe.to_dict(),
    )


def report(lay: Layout, recipe: Recipe, mesh_stats: dict | None = None) -> str:
    st, pod = lay.st, lay.pod
    L = []
    out = L.append
    out("== M2.3 scripted CAD: layout ==")
    out(f"Pod + blend: {pod.c0:.0f} mm chord, {2*pod.root_y:.0f} mm across the panel root faces, "
        f"{max(pod.inside_height(x) for x in np.linspace(0, pod.c0, 100)):.0f} mm deepest inside")
    use = _usable(st)
    out(f"Printer: {' x '.join(f'{v:.0f}' for v in st.printer_mm)} mm, usable {use[0]:.0f} mm a side "
        f"({st.printer_margin_mm:.0f} mm margin)")
    out(f"Split: centreline + {len(lay.cuts_x)} chordwise cut(s) at x = "
        + ", ".join(f"{c:.0f}" for c in lay.cuts_x) + f" mm, plus a {st.nose_len_mm:.0f} mm TPU nose cap")
    for s in lay.sections:
        out(f"  {s['name']:<14s} x {s['x0']:.0f}-{s['x1']:.0f} mm, bounding box {' x '.join(f'{v:.0f}' for v in s['bbox'])} mm")
    out("Keep-outs for cuts (x, mm, incl. margin): " + "; ".join(f"{n} {a:.0f}-{b:.0f}" for a, b, n in keepouts_x(lay)))
    out("")
    out("-- Spar joint --")
    out(f"Spar line through mid-depth: root ({lay.spar_root[0]:.1f}, {lay.spar_root[1]:.1f}, {lay.spar_root[2]:.1f}) mm, "
        f"tip ({lay.spar_tip[0]:.1f}, {lay.spar_tip[1]:.1f}, {lay.spar_tip[2]:.1f}) mm, "
        f"sweep {math.degrees(math.atan2(lay.spar_dir[0], lay.spar_dir[1])):.1f} deg, "
        f"dihedral {math.degrees(math.atan2(lay.spar_dir[2], lay.spar_dir[1])):.2f} deg")
    c = lay.inner_rib["centre"]
    out(f"Sleeve {st.sleeve_od_mm:.0f} mm OD x {st.sleeve_len_mm:.0f} mm, bore +{st.sleeve_bond_gap_mm:.2f} mm for epoxy; "
        f"inner rib at the spar's end, ({c[0]:.0f}, +-{c[1]:.0f}) mm, {2*st.inner_rib_half_len_mm:.0f} mm long; root rib "
        f"{lay.root_rib['x0']:.0f}-{lay.root_rib['x1']:.0f} mm")
    out(f"Carbon strips 10x1 mm at x = {lay.strip['x']:.0f} mm, floor and roof, {2*lay.strip['half']:.0f} mm long "
        f"(bonded {st.strip_bond_mm:.0f} mm past each inner rib), on {st.land_mm:.1f} mm skin lands, slotted through the ribs")
    out(f"Anti-rotation pin {st.pin_d_mm:.0f} mm at {st.pin_chord_frac:.0%} chord; panel catch: {len(lay.magnets)} x "
        f"{st.magnet_d_mm:.0f}x{st.magnet_t_mm:.0f} mm magnets per side at "
        + ", ".join(f"{f:.0%}" for f in st.magnet_chord_fracs) + " chord")
    out("")
    out("-- Hatches, tray, camera, motor, skid --")
    for h in lay.hatches:
        out(f"{h['name'].capitalize()} hatch x {h['x0']:.0f}-{h['x1']:.0f} mm, {2*h['half_w']:.0f} mm wide: "
            "hook under the front rim, 2 x 6x2 mm magnets at the back")
    b = lay.battery
    out(f"Battery tray x {lay.tray['x0']:.0f}-{lay.tray['x1']:.0f} mm, stops every {st.stop_pitch_mm:.0f} mm: pack centre "
        f"{b['x_min']:.0f}-{b['x_max']:.0f} mm (nominal {b['x']:.0f} for the CG target)")
    cam = lay.camera
    out(f"Camera at x = {cam['x']:.0f} mm, lens tilted {st.camera_tilt_deg:g} deg forward, recessed {cam['recess']:.1f} mm; "
        f"window {cam['window']['x1']-cam['window']['x0']:.0f} x {2*cam['window']['y']:.0f} mm")
    nc = lay.nacelle
    out(f"Nacelle x {nc['x_a']:.0f}-{nc['x_fw']:.0f} mm, {2*nc['r_a']:.0f} -> {2*nc['r_b']:.0f} mm, firewall "
        f"{st.firewall_mm:.0f} mm with the X2216 16/19 mm M3 pattern; tongue bolted through the floor")
    sk = lay.skid
    out(f"Skid: twin EPP rails x {sk['x0']:.0f}-{sk['x1']:.0f} mm at y = +-{sk['y0']:.0f}-{sk['y1']:.0f} mm, flat bottom "
        f"at z = {sk['z_bottom']:.0f} mm (one centre skid would cover the camera window)")
    out(f"Tip fins: pegged ({len(st.fin_peg_fracs)} x {st.fin_peg_d_mm:.0f} mm) and taped; fin root trimmed "
        f"{lay.fin['droop_deg']:.1f} deg to meet the panel's vertical tip face")
    out("")
    out("-- Checks --")
    for c in lay.checks:
        u = "" if c.unit == "count" else f" {c.unit}"
        out(f"  [{'OK ' if c.ok else 'BAD'}] {c.name}: {c.value:.1f}{u} (limit {c.sense} {c.limit:g}{u})"
            + (f"  {c.note}" if c.note else ""))
    if lay.notes:
        out("")
        out("-- Payload, packed automatically (nearest free spot to its weights.py position) --")
        for n in lay.notes:
            out(f"  {n}")
    if mesh_stats:
        out("")
        out("-- Parts (built by cad_mesh from the same recipe) --")
        tot = 0.0
        for name, s in mesh_stats.items():
            out(f"  {name:<22s} {s['material']:<26s} x{s['count']}  {s['volume_cm3']:6.1f} cm3  {s['mass_g']:5.1f} g  "
                f"bbox {' x '.join(f'{v:.0f}' for v in s['bbox'])} mm  {'fits' if s['fits'] else 'DOES NOT FIT'}"
                f"{'' if s['watertight'] else '  NOT WATERTIGHT'}")
            if s.get("airframe"):
                tot += s["mass_g"] * s["count"]
        out(f"  printed airframe parts (pod, nose, lids, nacelle, camera mount, stops): {tot:.0f} g")
        fb = mass_feedback(lay, mesh_stats)
        out(f"  weights.py had {fb['old_g']:.0f} g for these (pod shell + frames, nose cap, motor plate, camera mount): "
            f"{fb['delta_g']:+.0f} g. Aircraft as built ~{fb['aircraft_g']:.0f} g with the 20 % structure allowance.")
        out(f"  CG: the battery slides {fb['battery_shift_mm']:+.1f} mm to stay on target "
            f"({'inside' if fb['within_travel'] else 'OUTSIDE'} the +-{st.tray_travel_mm:.0f} mm tray travel)")
    out("")
    out(f"Recipe: {len(recipe.solids)} solids, {len(recipe.parts)} parts "
        f"({sum(1 for p in recipe.parts if p['printed'])} printed)")
    return "\n".join(L)


#: weights.py lines the printed parts replace: (name prefix, paper mass g, x mm) are read from weights.build_up
REPLACED_WEIGHT_LINES = ("Centre pod", "Replaceable nose cap", "Motor mount plate", "Printed camera mount")


def mass_feedback(lay: Layout, stats: dict) -> dict:
    """Printed-part masses from the CAD against the weights.py lines they replace, and how far the
    battery must slide to keep the CG on target (the M2.4 weigh-in will settle the real numbers)."""
    from . import weights as W

    bu = W.build_up(lay.geom, W.default_propulsion())
    old = [(it.mass_kg * 1000, it.cg[0] * 1000) for it in bu.items if it.name.startswith(REPLACED_WEIGHT_LINES)]
    new = [(s["mass_g"] * s["count"], s["centroid"][0]) for s in stats.values() if s.get("airframe")]
    cg = bu.choices.cg_target_x_m * 1000
    m_old, m_new = sum(m for m, _ in old), sum(m for m, _ in new)
    dmom = sum(m * (x - cg) for m, x in new) - sum(m * (x - cg) for m, x in old)
    shift = -dmom / lay.battery["mass_g"]
    return dict(old_g=m_old, new_g=m_new, delta_g=m_new - m_old, battery_shift_mm=shift,
                aircraft_g=bu.mass_kg() * 1000 + (m_new - m_old) * (1 + W.GROWTH["structure"]),
                within_travel=abs(shift) <= lay.st.tray_travel_mm)


def all_ok(lay: Layout) -> bool:
    return all(c.ok for c in lay.checks)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_layout(lay: Layout, path: str):  # pragma: no cover - visual
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    pod, st = lay.pod, lay.st
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 10), gridspec_kw=dict(height_ratios=[1.5, 1]))
    # top view (x across, y up)
    ys = np.linspace(-pod.root_y, pod.root_y, 81)
    le = [pod.station(y).le_x for y in ys]
    te = [pod.station(y).le_x + pod.station(y).chord for y in ys]
    ax.fill(np.r_[le, te[::-1]], np.r_[ys, ys[::-1]], color="#e8eef6", ec="#456", lw=1.2)
    for c in lay.cuts_x:
        ax.axvline(c, color="#c33", lw=1.5, ls="--")
    ax.axhline(0, color="#c33", lw=1.5, ls="--")
    ax.axvline(st.nose_len_mm, color="#a5a", lw=1.2, ls=":")
    for h in lay.hatches:
        ax.add_patch(Rectangle((h["x0"], -h["half_w"]), h["x1"] - h["x0"], 2 * h["half_w"], fill=False, ec="#2a7", lw=1.5))
        ax.text(h["x0"] + 3, h["half_w"] - 10, f"{h['name']} hatch", color="#2a7", fontsize=8)
    for sgn in (1, -1):
        a, b = lay.point_on_spar(-st.sleeve_len_mm), lay.point_on_spar(0)
        ax.plot([a[0], b[0]], [sgn * a[1], sgn * b[1]], color="#222", lw=5, solid_capstyle="butt")
        ir = lay.inner_rib
        ax.add_patch(Rectangle((ir["x0"], min(sgn * ir["y0"], sgn * ir["y1"])), ir["x1"] - ir["x0"], st.rib_web_mm, color="#777"))
        rr = lay.root_rib
        ax.add_patch(Rectangle((rr["x0"], min(sgn * rr["y0"], sgn * rr["y1"])), rr["x1"] - rr["x0"], rr["y1"] - rr["y0"], color="#999"))
        for m in lay.magnets:
            ax.plot(m[0], sgn * (pod.root_y - 2), "o", color="#d80", ms=5)
        sk = lay.skid
        ax.add_patch(Rectangle((sk["x0"], sgn * sk["y0"] if sgn > 0 else -sk["y1"]), sk["x1"] - sk["x0"], sk["y1"] - sk["y0"],
                               fill=False, ec="#a62", lw=1, ls="-."))
    ax.add_patch(Rectangle((lay.strip["x"] - 5, -lay.strip["half"]), 10, 2 * lay.strip["half"], color="#333", alpha=0.5))
    for p in lay.payload:
        ax.add_patch(Rectangle(p.lo[:2], *(p.hi[:2] - p.lo[:2]), fill=True, fc="#fdd", ec="#a44", lw=0.8, alpha=0.8))
        ax.text(*(0.5 * (p.lo[:2] + p.hi[:2])), p.name.replace(" ", "\n", 1), fontsize=6, ha="center", va="center")
    tr = lay.tray
    for s in (-1, 1):
        ax.plot([tr["x0"], tr["x1"]], [s * tr["rail_y"]] * 2, color="#46c", lw=2)
    nc = lay.nacelle
    ax.add_patch(Polygon([[nc["x_a"], -nc["r_a"]], [nc["x_fw"], -nc["r_b"]], [nc["x_fw"], nc["r_b"]], [nc["x_a"], nc["r_a"]]],
                         fc="#ccc", ec="#333"))
    w = lay.camera["window"]
    ax.add_patch(Rectangle((w["x0"], -w["y"]), w["x1"] - w["x0"], 2 * w["y"], fill=False, ec="#06c", lw=1.2))
    ax.set_aspect("equal")
    ax.set_xlabel("x aft of the nose, mm")
    ax.set_ylabel("y, mm")
    ax.set_title("Top view: print joints (red dashed), hatches (green), spar sleeves and ribs (grey), payload (pink), "
                 "skid rails (brown), camera window (blue)", fontsize=9)
    # side view at the centreline
    xs = np.linspace(0, pod.c0, 300)
    zz = np.array([pod.z_out(x, 0) for x in xs])
    bx.fill_between(xs, zz[:, 0], zz[:, 1], color="#e8eef6", ec="#456")
    for c in lay.cuts_x:
        bx.axvline(c, color="#c33", lw=1.5, ls="--")
    bx.axvline(st.nose_len_mm, color="#a5a", lw=1.2, ls=":")
    for p in lay.payload:
        if p.lo[1] <= 0 <= p.hi[1] or p.name in ("Pi Zero 2 W", "flight controller", "ESC"):
            bx.add_patch(Rectangle((p.lo[0], p.lo[2]), p.hi[0] - p.lo[0], p.hi[2] - p.lo[2], fc="#fdd", ec="#a44", alpha=0.7))
    b = lay.battery
    for xb, ls in ((b["x_min"], ":"), (b["x_max"], ":")):
        bx.add_patch(Rectangle((xb - b["L"] / 2, b["z0"]), b["L"], b["H"], fill=False, ec="#a44", ls=ls))
    sk = lay.skid
    zt = np.array([pod.z_out(x, 0)[0] for x in np.linspace(sk["x0"], sk["x1"], 60)])
    xs2 = np.linspace(sk["x0"], sk["x1"], 60)
    bx.fill_between(xs2, sk["z_bottom"], zt, color="#e6c9a0", alpha=0.8)
    bx.add_patch(Polygon([[nc["x_a"], -nc["r_a"]], [nc["x_fw"], -nc["r_b"]], [nc["x_fw"], nc["r_b"]], [nc["x_a"], nc["r_a"]]],
                         fc="#ccc", ec="#333"))
    cam = lay.camera
    bx.plot(*cam["lens_tip"][[0, 2]], "o", color="#06c")
    for sgn in (-1, 1):
        a = math.radians(st.camera_fov_deg[0] / 2) * sgn + math.radians(st.camera_tilt_deg)
        bx.plot([cam["lens_tip"][0], cam["lens_tip"][0] - 60 * math.sin(a)], [cam["lens_tip"][2], cam["lens_tip"][2] - 60 * math.cos(a)],
                color="#06c", lw=0.8)
    for h in lay.hatches:
        bx.plot([h["x0"], h["x1"]], [pod.z_out(h["x0"], 0)[1] + 3, pod.z_out(h["x1"], 0)[1] + 3], color="#2a7", lw=3)
    bx.axhline(sk["z_bottom"], color="#865", lw=0.8)
    bx.set_aspect("equal")
    bx.set_xlim(-10, nc["x_fw"] + 50)
    bx.set_xlabel("x, mm")
    bx.set_ylabel("z, mm")
    bx.set_title("Side view at the centreline: battery tray travel (dotted), skid rail, camera view, nacelle", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    from .structures import load_geometry

    p = argparse.ArgumentParser(prog="manta_mapper.cad", description="M2.3 scripted CAD layout and recipe.")
    p.add_argument("mission", nargs="?", help="mission TOML (not needed: the geometry comes from the optimizer result)")
    p.add_argument("--design", help="optimizer result JSON (default docs/optimizer_result.json)")
    p.add_argument("--json", help="write the recipe + layout here (what the Fusion 360 script reads)")
    p.add_argument("--dxf", help="directory for the 2-D templates (DXF)")
    p.add_argument("--stl", help="directory for STLs built in Python (needs manifold3d)")
    p.add_argument("--render", help="PNG of the parts built in Python (needs manifold3d)")
    p.add_argument("--report", help="save the text report here")
    p.add_argument("--plots", help="directory for the layout plot")
    a = p.parse_args(argv)
    geom = load_geometry(a.design)
    lay = layout(geom)
    rec = build_recipe(lay)
    stats = None
    if a.stl or a.render:
        from . import cad_mesh

        built = cad_mesh.build(rec)
        stats = cad_mesh.part_stats(rec, built, lay.st)
        if a.stl:
            cad_mesh.export_stl(rec, built, a.stl)
        if a.render:
            cad_mesh.render(rec, built, a.render)
    txt = report(lay, rec, stats)
    print(txt)
    if a.report:
        Path(a.report).write_text(txt + "\n")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(to_json(lay, rec), separators=(",", ":")))
    if a.dxf:
        from . import cad_templates

        for f in cad_templates.write_all(lay, a.dxf):
            print("wrote", f)
    if a.plots:
        Path(a.plots).mkdir(parents=True, exist_ok=True)
        plot_layout(lay, str(Path(a.plots) / "cad_layout.png"))


if __name__ == "__main__":  # pragma: no cover
    main()
