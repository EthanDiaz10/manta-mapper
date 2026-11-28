"""Spar sizing, load cases and the panel-to-pod joint (milestone M2.2).

Sizes the carbon spar of each removable outer panel, the sleeve it plugs
into, and the carry-through that joins the two sleeves across the pod, for
every load the airframe sees (AC-10, AC-13, AC-14):

* **flight**: +4 g / -1.5 g manoeuvre at the 1.5 kg design mass, and the
  gust envelope (Pratt formula, 7.6 m/s gust at a fast cruise, 3.8 m/s at
  the dive speed) at both 1.5 kg and the as-built mass, capped where the
  wing stalls first. A light wing (~28 N/m^2) rides gusts hard, so gusts set
  the flight case, not the 4 g of AC-10;
* **hand launch** (baseline) and **bungee catapult** (stretch goal): the
  panels' own inertia while the pod is accelerated forward, plus gravity;
* **belly landing** (AC-13, repeated 20 times): the panels' inertia when the
  skid stops the sink rate. The deceleration depends on how far the skid and
  the dirt give, so the case is run as a sweep and the result read as "how
  hard a landing the spar takes", not one uncertain number;
* **tip strike**: the drooped fins sit 26 mm above the ground and touch at
  ~1.5 deg of bank, so a fin catching the dirt is a normal landing, not an
  accident. The fins are taped on; the tape lets go at a known force, which
  caps the load the spar ever sees from the tip.

Every load is a 3-D force at a point; moments are taken about each spar
station and split into bending (the tube is round, so only the magnitude
counts) and torsion. That handles sweep, dihedral and chordwise loads in one
place.

**The spar** is a telescoping set of off-the-shelf carbon tubes, one size
nesting inside the next (12x10 -> 10x8 -> 8x6), each run as far out as the
foam is deep enough to bury it with 2 mm of cover. The search tries every
combination of tube sizes, materials (pultruded or roll-wrapped) and step
stations, and keeps the cheapest + lightest (same 100 $/kg as M1.6) that
passes every check:

* strength at ultimate (1.5 x limit) in bending and torsion, every station;
* tip deflection in level flight (the stability model assumes a rigid wing);
* **control reversal**: the elevons twist the panel nose-down as they
  push the trailing edge down. EPP is soft in torsion and a pultruded
  tube is too (its fibres all run lengthwise), so above some speed the
  elevons lose roll authority altogether. Reversal must be at least 1.2 x
  the dive speed;
* **crushing at the sleeve mouth**: where the spar enters the sleeve the
  load is a line contact across the tube. Pultruded tubes are weak across
  their fibres and split there; roll-wrapped tubes have hoop fibres.

Material properties are typical datasheet values, knocked down for hobby
quality, and every one is a placeholder for the M2.4 test prints and the
AC-10 wingtip load test (sandbag test of the assembled wing).

Run: ``python -m manta_mapper.structures config/mission.toml --report docs/structures_report.txt --plots docs/img``
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .geometry import BWBGeometry, airfoil_local_thickness
from .requirements import G0, MissionRequirements

# --------------------------------------------------------------------------
# Materials and the tube catalogue
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TubeMaterial:
    name: str
    E: float  # Pa, axial modulus
    G: float  # Pa, in-plane shear modulus (torsion)
    sigma_allow: float  # Pa, ultimate bending stress (already knocked down)
    tau_allow: float  # Pa, ultimate torsional shear stress
    bearing_allow: float  # Pa, ultimate transverse bearing on the wall: crushing at a socket lip
    density: float  # kg/m^3
    note: str = ""


#: Pultruded: all fibres lengthwise. Stiff and strong in bending, cheap, but
#: soft in torsion and weak across the fibres (splits when crushed).
PULTRUDED = TubeMaterial("pultruded", 115e9, 4.5e9, 700e6, 35e6, 50e6, 1550,
                         "all-0 deg fibres: stiff, cheap; soft in torsion, splits when crushed")
#: Roll-wrapped: unidirectional plies with a woven or +-45 deg wrap. A little
#: less stiff in bending, much better in torsion and crushing.
ROLL_WRAPPED = TubeMaterial("roll-wrapped", 85e9, 12e9, 600e6, 90e6, 120e6, 1500,
                            "UD + woven/45 deg wrap: 3x torsion stiffness, resists crushing")
MATERIALS = {m.name: m for m in (PULTRUDED, ROLL_WRAPPED)}


@dataclass(frozen=True)
class Tube:
    od: float  # m
    id: float  # m
    material: TubeMaterial
    usd_per_m: float  # sold in 1 m lengths; rough 2026 hobby-shop prices

    @property
    def label(self) -> str:
        return f"{self.od*1000:.0f}x{self.id*1000:.0f} {self.material.name}"

    @property
    def area(self) -> float:
        return math.pi / 4 * (self.od**2 - self.id**2)

    @property
    def I(self) -> float:
        return math.pi / 64 * (self.od**4 - self.id**4)

    @property
    def J(self) -> float:
        return 2 * self.I

    @property
    def Z(self) -> float:
        return self.I / (self.od / 2)

    @property
    def wall(self) -> float:
        return (self.od - self.id) / 2

    @property
    def kg_per_m(self) -> float:
        return self.area * self.material.density

    @property
    def EI(self) -> float:
        return self.material.E * self.I

    @property
    def GJ(self) -> float:
        return self.material.G * self.J


def _cat():
    p = PULTRUDED
    r = ROLL_WRAPPED
    sizes = [(6, 4, 4.0, None), (8, 6, 5.0, 9.0), (10, 8, 6.0, 11.0), (12, 10, 8.0, 13.0),
             (14, 12, 10.0, 16.0), (16, 14, 12.0, 19.0)]
    out = []
    for od, idd, usd_p, usd_r in sizes:
        out.append(Tube(od / 1000, idd / 1000, p, usd_p))
        if usd_r is not None:
            out.append(Tube(od / 1000, idd / 1000, r, usd_r))
    return tuple(out)


#: Off-the-shelf 1 m carbon tubes (mm OD x ID, 1 mm wall so each nests in the next size up).
CATALOGUE: tuple[Tube, ...] = _cat()

# Printed parts and bonds (LW-ASA foamed at ~0.7 g/cm^3, epoxy on sanded ASA)
PRINT_BEARING_ALLOW = 10e6  # Pa, ultimate bearing on a printed boss
PRINT_SHEAR_ALLOW = 4e6  # Pa, ultimate in-plane shear, printed rib
BOND_SHEAR_ALLOW = 2.0e6  # Pa, ultimate lap shear, epoxy on sanded ASA / carbon (conservative)
EPP_SHEAR_MODULUS = 3e6  # Pa, EPP 1.9 lb (30 kg/m^3)
FILM_SHEAR_MODULUS = 1.0e9  # Pa, BOPP/polyester laminating film
FILM_THICKNESS = 38e-6  # m, 1.5 mil
SKIN_TORSION_KNOCKDOWN = 0.5  # film seams, elevon cut, imperfect bond to the fuzzy EPP
CAP_STRIP = (0.010, 0.001)  # 10 x 1 mm pultruded carbon strip (carry-through caps)
CAP_STRIP_ALLOW = 500e6  # Pa, compression with the strip bonded to a printed wall
CAP_STRIP_USD_PER_M = 3.0


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadSettings:
    design_mass_kg: float = 1.5  # same sizing mass as the aero/propulsion checks
    safety_factor: float = 1.5  # AC-10
    n_pos: float = 4.0  # AC-10 manoeuvre limit
    n_neg: float = -1.5
    v_cruise_fast_m_s: float = 20.0  # fast transit/upwind leg (V_C for gusts)
    v_dive_m_s: float = 30.0  # dive speed V_D (level top speed ~24 m/s x 1.25)
    gust_c_m_s: float = 7.6  # 25 ft/s at V_C: summer thermals over the desert
    gust_d_m_s: float = 3.8  # 12.5 ft/s at V_D
    cl_max: float = 1.0  # caps gust/manoeuvre lift where the wing stalls (aircraft CL, as built ~0.95)
    # hand launch: arm swing from rest to release speed
    hand_release_m_s: float = 11.0
    hand_stroke_m: float = 0.7
    hand_peak_factor: float = 1.5
    # bungee catapult (stretch goal)
    cat_release_m_s: float = 13.0
    cat_stroke_m: float = 1.2
    cat_peak_factor: float = 2.0  # bungee pull is highest at the start of the stroke
    # belly landing (AC-13): sink rate stopped over the skid crush + dirt give
    land_sink_m_s: float = 1.5  # 1.3 Vs glide (~0.9 m/s) + a sloppy flare
    land_stroke_m: float = 0.012
    land_hard_sink_m_s: float = 2.5  # once-off arrival, checked at ultimate without the 1.5
    # tip strike: fin tape releases at this force (any direction)
    tip_release_n: float = 30.0
    # torsion: full elevon at the dive speed
    elevon_max_deg: float = 15.0
    reversal_margin: float = 1.2  # reversal speed >= this x V_D
    # stiffness: level-flight tip deflection (stability models assume a rigid wing)
    max_tip_deflection_1g_m: float = 0.012
    # spar fit and joint
    min_cover_m: float = 0.002  # EPP over the tube, top and bottom
    socket_length_m: float = 0.10  # tube inside the pod sleeve
    telescope_overlap_m: float = 0.05  # bonded overlap where one tube nests in the next
    rib_thickness_m: float = 0.008  # printed ribs at each end of the sleeve (web)
    rib_boss_m: float = 0.015  # length of the printed boss around the sleeve where it passes through a rib
    pin_offset_frac: float = 0.35  # anti-rotation pin aft of the spar, fraction of root chord
    pin_d_m: float = 0.003
    lip_contact_d: float = 0.5  # bearing length at the sleeve mouth, in tube diameters (edge contact)
    #: robust margin on top of the 1.5 factor: spar and joint utilisation at ultimate must stay under this,
    #: because the tip-strike force (the fin tape letting go) is a guess until it is pull-tested
    max_utilisation: float = 0.8
    usd_per_kg: float = 100.0  # same price on mass as the M1.6 optimizer
    usd_per_joint: float = 4.0  # build effort and risk of each bonded telescoping joint
    min_piece_m: float = 0.15  # shortest exposed length of a telescoping piece worth cutting


# --------------------------------------------------------------------------
# Spar layout
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    tube: Tube
    s0: float  # m along the spar from the panel root (negative: inside the pod sleeve)
    s1: float


@dataclass(frozen=True)
class SparLayout:
    segments: tuple[Segment, ...]  # root first; each nests inside the previous one

    def tube_at(self, s: float) -> Tube:
        """The outermost (largest) tube at station s (the one whose surface carries bending)."""
        for seg in self.segments:
            if seg.s0 - 1e-9 <= s <= seg.s1 + 1e-9:
                return seg.tube
        return self.segments[-1].tube

    def section(self, s: float) -> tuple[float, float]:
        """(EI, GJ) at s, adding a nested tube where two overlap."""
        ei = gj = 0.0
        for seg in self.segments:
            if seg.s0 - 1e-9 <= s <= seg.s1 + 1e-9:
                ei += seg.tube.EI
                gj += seg.tube.GJ
        return ei, gj

    @property
    def mass_kg(self) -> float:  # one panel
        return sum(seg.tube.kg_per_m * (seg.s1 - seg.s0) for seg in self.segments)

    @property
    def cost_usd(self) -> float:  # one panel; 1 m lengths, cut (both panels share a length when it fits)
        c = 0.0
        for seg in self.segments:
            L = seg.s1 - seg.s0
            lengths = math.ceil(2 * L - 1e-6)  # both panels' pieces from 1 m sticks
            c += seg.tube.usd_per_m * lengths / 2
        return c

    @property
    def label(self) -> str:
        return " -> ".join(f"{s.tube.label} [{s.s0*1000:+.0f}..{s.s1*1000:.0f} mm]" for s in self.segments)


# --------------------------------------------------------------------------
# Geometry of the spar line and the panel loads
# --------------------------------------------------------------------------


@dataclass
class SparLine:
    p0: np.ndarray  # panel root, on the spar line
    d: np.ndarray  # unit vector root -> tip
    length: float  # root to tip

    def point(self, s: float) -> np.ndarray:
        return self.p0 + s * self.d

    def s_of_y(self, y):
        return (np.asarray(y) - self.p0[1]) / self.d[1]


def spar_line(geom: BWBGeometry) -> SparLine:
    f = geom.spar_chord_fraction
    r_le, r_c, _ = geom._outer_station(0.0)
    t_le, t_c, _ = geom._outer_station(1.0)
    p0 = r_le + np.array([f * r_c, 0, 0])
    p1 = t_le + np.array([f * t_c, 0, 0])
    L = float(np.linalg.norm(p1 - p0))
    return SparLine(p0, (p1 - p0) / L, L)


def depth_at(geom: BWBGeometry, line: SparLine, s: float) -> float:
    """Section depth at the spar at s (inside the pod/blend: the outer-root depth, the section only gets deeper)."""
    eta = min(max(s / line.length, 0.0), 1.0)
    return geom.spar_depth_m(eta)


@dataclass
class PanelModel:
    """Distributed lift shape and masses of one (right) outer panel."""

    geom: BWBGeometry
    line: SparLine
    lift_pts: np.ndarray  # (n, 3) application points (quarter chord) of outer-panel lift strips
    lift_frac: np.ndarray  # (n,) fraction of the aircraft lift on each strip
    fin_frac: float  # fraction of aircraft lift on one fin (at the spar tip)
    mass_pts: np.ndarray  # (m, 4) [kg, x, y, z] non-spar panel masses (as built)
    cla_per_rad: float  # aircraft lift-curve slope
    strips_y: np.ndarray
    strips_c: np.ndarray
    strips_w: np.ndarray
    elevon: np.ndarray  # bool per strip
    source: str = "VLM"

    @property
    def panel_lift_frac(self) -> float:
        return float(self.lift_frac.sum() + self.fin_frac)


def _lift_shape_vlm(geom: BWBGeometry, req: MissionRequirements, cl_target: float = 0.9):
    """Spanwise lift shape from the AeroSandbox VLM at a high lift coefficient.

    With washout the loading moves outboard as alpha rises (the twist is a smaller
    share of the local angle), so a near-stall shape is the conservative one for
    root bending.
    """
    from . import aero as A

    fl = A.field_flight(req, 12.0)
    r0 = A.run_vlm(geom, fl, 0.0)
    r1 = A.run_vlm(geom, fl, 6.0)
    cla = (r1["CL"] - r0["CL"]) / math.radians(6.0)
    alpha = math.degrees((cl_target - r0["CL"]) / cla)
    res = A.run_vlm(geom, fl, alpha)
    a = math.radians(res.alpha)
    lift_dir = np.array([-math.sin(a), 0.0, math.cos(a)])
    lift = res.forces @ lift_dir
    c = res.centers
    total = float(lift.sum())
    ys, cl, w = A.spanwise_cl(geom, res)
    chord = A.chord_at(geom, ys)
    strip_lift = cl * fl.q * chord * w  # N, right half
    fin = float(lift[c[:, 1] > geom.semispan_m + 1e-6].sum())
    return ys, strip_lift / total, fin / total, cla


def _lift_shape_schrenk(geom: BWBGeometry, n: int = 60):
    """Schrenk's approximation (average of planform and elliptic loading): no AeroSandbox needed."""
    from . import aero as A

    y = (np.arange(n) + 0.5) / n * geom.semispan_m
    w = np.full(n, geom.semispan_m / n)
    c = A.chord_at(geom, y)
    S_half = geom.reference_area_m2 / 2
    ell = 4 * S_half / (math.pi * geom.semispan_m) * np.sqrt(1 - (y / geom.semispan_m) ** 2)
    load = 0.5 * (c + ell) * w
    return y, load / (2 * load.sum()), 0.0, 2 * math.pi * geom.aspect_ratio / (geom.aspect_ratio + 2)


def panel_model(geom: BWBGeometry, req: MissionRequirements, build=None, use_vlm: bool = True) -> PanelModel:
    """Lift shape plus the as-built masses that ride on one outer panel (spar excluded)."""
    from . import weights as W

    line = spar_line(geom)
    if use_vlm:
        try:
            ys, frac, fin, cla = _lift_shape_vlm(geom, req)
            src = "AeroSandbox VLM at CL 0.9"
        except ImportError:  # pragma: no cover - aero extras not installed
            ys, frac, fin, cla = _lift_shape_schrenk(geom)
            src = "Schrenk"
    else:
        ys, frac, fin, cla = _lift_shape_schrenk(geom)
        src = "Schrenk"
    on = ys > geom.root_y_m
    ys_o, frac_o = ys[on], frac[on]
    s = line.s_of_y(ys_o)
    pts = line.p0 + s[:, None] * line.d  # quarter chord ~ spar line (spar at 25 % chord)
    from . import aero as A

    c = A.chord_at(geom, ys_o)
    edges = np.r_[geom.root_y_m, 0.5 * (ys_o[1:] + ys_o[:-1]), geom.semispan_m]
    w = np.diff(edges)
    eta = (ys_o - geom.root_y_m) / geom.outer_semispan_m
    elev = (eta >= geom.elevon_start) & (eta <= geom.elevon_end)

    if build is None:
        build = W.build_up(geom, W.default_propulsion())
    lumps = []
    for it in build.items:
        if it.name.startswith("Spar"):
            continue
        g = 1 + W.GROWTH[it.known]
        for m, x, y, z in it.points:
            if y > geom.root_y_m + 1e-6:
                lumps.append([m * g, x, y, z])
    return PanelModel(geom, line, pts, frac_o, fin, np.array(lumps), cla, ys_o, c, w, elev, src)


# --------------------------------------------------------------------------
# Load cases
# --------------------------------------------------------------------------


@dataclass
class LoadCase:
    name: str
    kind: str  # flight / launch / landing / tip
    forces: np.ndarray  # (k, 6): x, y, z, Fx, Fy, Fz (N) on the right panel, geometry axes (x aft, z up)
    torque_per_m: np.ndarray | None = None  # (n,) N m/m about the spar axis along the lift strips (elevon)
    factor: float = 1.5  # ultimate / limit
    repeated: bool = False  # happens every flight: printed joints held to half their strength
    note: str = ""


def _gust_kg(mu: float) -> float:
    return 0.88 * mu / (5.3 + mu)


def flight_envelope(pm: PanelModel, req: MissionRequirements, ls: LoadSettings, mass_built_kg: float) -> dict:
    """Limit lift (N, whole aircraft) for manoeuvre and gusts, with the stall cap."""
    from . import aero as A

    geom = pm.geom
    S = geom.reference_area_m2
    out = {}
    for label, m in (("design", ls.design_mass_kg), ("as built", mass_built_kg)):
        W = m * G0
        rows = []
        for v, u, vname in ((ls.v_cruise_fast_m_s, ls.gust_c_m_s, "V_C"), (ls.v_dive_m_s, ls.gust_d_m_s, "V_D")):
            fl = A.cruise_flight(req, v)
            mu = 2 * (W / S) / (fl.rho * geom.mac_m * pm.cla_per_rad * G0)
            dL = 0.5 * fl.rho * v * u * pm.cla_per_rad * S * _gust_kg(mu)
            cap = fl.q * S * ls.cl_max
            rows.append(dict(speed=vname, v=v, gust=u, n=(W + dL) / W, n_neg=(W - dL) / W,
                             n_cap=cap / W, lift_pos=min(W + dL, cap), lift_neg=max(W - dL, -0.6 * cap)))
        out[label] = dict(mass=m, W=W, gusts=rows, man_pos=ls.n_pos * W, man_neg=ls.n_neg * W)
    return out


def _flight_case(pm: PanelModel, total_lift: float, n_inertia: float, name: str, ls: LoadSettings,
                 torque_q: float = 0.0, elevon_deg: float = 0.0) -> LoadCase:
    lift = total_lift * pm.lift_frac
    f = [np.column_stack([pm.lift_pts, np.zeros((len(lift), 2)), lift])]
    tip = pm.line.point(pm.line.length)
    f.append(np.array([[*tip, 0, 0, total_lift * pm.fin_frac]]))
    mp = pm.mass_pts
    f.append(np.column_stack([mp[:, 1:], np.zeros((len(mp), 2)), -n_inertia * mp[:, 0] * G0]))
    torque = None
    if torque_q:
        # thin-aerofoil flap moment about c/4 (~ the spar): Cm_delta = -0.5 sin(th)(1 - cos(th)), hinge at 1 - E
        E = pm.geom.elevon_chord_fraction
        th = math.acos(1 - 2 * (1 - E))
        cmd = -0.5 * math.sin(th) * (1 - math.cos(th))
        torque = np.where(pm.elevon, torque_q * pm.strips_c**2 * cmd * math.radians(elevon_deg), 0.0)
    return LoadCase(name, "flight", np.vstack(f), torque, ls.safety_factor,
                    note=f"aircraft lift {total_lift:.0f} N, panel inertia {n_inertia:.1f} g")


def load_cases(pm: PanelModel, req: MissionRequirements, ls: LoadSettings, mass_built_kg: float,
               spar_kg_per_m: float = 0.0) -> tuple[list[LoadCase], dict]:
    """All limit load cases on the right panel. ``spar_kg_per_m`` adds the spar's own mass as a line load."""
    from . import aero as A

    env = flight_envelope(pm, req, ls, mass_built_kg)
    cases = []
    # add the spar mass to the panel lumps (it moves with the sizing, so it is passed in)
    line = pm.line
    if spar_kg_per_m:
        ss = (np.arange(10) + 0.5) / 10 * line.length
        sp = line.p0 + ss[:, None] * line.d
        pm = replace(pm, mass_pts=np.vstack([pm.mass_pts,
                                             np.column_stack([np.full(10, spar_kg_per_m * line.length / 10), sp])]))
    q_d = A.cruise_flight(req, ls.v_dive_m_s).q

    best_pos = max(((env[k]["man_pos"], k, "manoeuvre +%.1f g" % ls.n_pos) for k in ("design",)) , key=lambda t: t[0])
    cases.append(_flight_case(pm, best_pos[0], best_pos[0] / env["design"]["W"], f"Manoeuvre +{ls.n_pos:g} g, 1.5 kg", ls,
                              q_d, -ls.elevon_max_deg))
    cases.append(_flight_case(pm, env["design"]["man_neg"], ls.n_neg, f"Manoeuvre {ls.n_neg:g} g, 1.5 kg", ls,
                              q_d, ls.elevon_max_deg))
    for k in ("design", "as built"):
        e = env[k]
        g = max(e["gusts"], key=lambda r: r["lift_pos"])
        cases.append(_flight_case(pm, g["lift_pos"], g["lift_pos"] / e["W"],
                                  f"Gust +{g['gust']:.1f} m/s at {g['speed']} {g['v']:.0f} m/s, {e['mass']:.2f} kg", ls,
                                  q_d, -ls.elevon_max_deg))
        gn = min(e["gusts"], key=lambda r: r["lift_neg"])
        cases.append(_flight_case(pm, gn["lift_neg"], gn["lift_neg"] / e["W"],
                                  f"Gust -{gn['gust']:.1f} m/s at {gn['speed']} {gn['v']:.0f} m/s, {e['mass']:.2f} kg", ls,
                                  q_d, ls.elevon_max_deg))

    mp = pm.mass_pts
    for name, v, s, pk, rep in (("Hand launch", ls.hand_release_m_s, ls.hand_stroke_m, ls.hand_peak_factor, True),
                                ("Bungee catapult (stretch)", ls.cat_release_m_s, ls.cat_stroke_m, ls.cat_peak_factor, True)):
        ax = v**2 / (2 * s) * pk  # forward acceleration of the pod
        f = np.column_stack([mp[:, 1:], mp[:, 0] * ax, np.zeros(len(mp)), -mp[:, 0] * G0])  # inertia aft (+x) + weight
        cases.append(LoadCase(f"{name}: {ax/G0:.1f} g forward", "launch", f, None, ls.safety_factor, rep,
                              f"{v:.0f} m/s over {s:.1f} m, peak {pk:g}x average"))
    for name, sink, fac, rep in (("Belly landing", ls.land_sink_m_s, ls.safety_factor, True),
                                 ("Hard landing (once)", ls.land_hard_sink_m_s, 1.0, False)):
        n = sink**2 / (2 * G0 * ls.land_stroke_m)
        f = np.column_stack([mp[:, 1:], np.zeros((len(mp), 2)), -n * mp[:, 0] * G0])  # no lift relief
        cases.append(LoadCase(f"{name}: {sink:.1f} m/s sink, {n:.0f} g", "landing", f, None, fac, rep,
                              f"stopped over {ls.land_stroke_m*1000:.0f} mm of skid + dirt, no lift relief"))
    tip = line.point(line.length)
    F = ls.tip_release_n / math.sqrt(2)
    for name, vec in (("Tip strike: fin drags back and up", (F, 0, F)), ("Tip strike: fin drags back and down", (F, 0, -F))):
        f = np.array([[*tip, *vec]])
        cases.append(LoadCase(f"{name} ({ls.tip_release_n:.0f} N)", "tip", f, None, ls.safety_factor, True,
                              "capped by the fin tape letting go"))
    return cases, env


# --------------------------------------------------------------------------
# Internal loads along the spar
# --------------------------------------------------------------------------


@dataclass
class Internal:
    s: np.ndarray
    bending: np.ndarray  # |M_bending| N m
    m_flap: np.ndarray  # bending about the chordwise axis (signed, + = tips up)
    torsion: np.ndarray  # about the spar axis
    shear: np.ndarray  # |V| perpendicular to the spar
    shear_vec: np.ndarray  # (n, 3)
    moment_vec: np.ndarray  # (n, 3)


def internal_loads(case: LoadCase, pm: PanelModel, s: np.ndarray, socket_m: float = 0.10) -> Internal:
    """Bending, torsion and shear along the spar. Inside the sleeve (s < 0) the tube is a beam on
    two ribs: the root moment falls linearly to zero at the inner rib, the torque has gone into
    the anti-rotation pin, and the shear is the inner rib's reaction."""
    line = pm.line
    F = case.forces
    pts, fv = F[:, :3], F[:, 3:]
    y_load = pts[:, 1]
    d = line.d
    # flap axis: perpendicular to the spar, in the panel plane, pointing aft-ish (chordwise)
    up = np.array([0, 0, 1.0])
    chord_axis = np.cross(up, d)
    chord_axis /= np.linalg.norm(chord_axis)
    out = {k: [] for k in ("b", "mf", "t", "v", "vv", "mv")}
    tq = case.torque_per_m
    for si in s:
        p = line.point(si)
        mask = y_load > p[1] + 1e-9
        r = pts[mask] - p
        M = np.cross(r, fv[mask]).sum(0) if mask.any() else np.zeros(3)
        V = fv[mask].sum(0) if mask.any() else np.zeros(3)
        T = float(M @ d)
        if tq is not None:
            sel = pm.strips_y > p[1]
            T += float(np.sum(tq[sel] * pm.strips_w[sel]))
        Mb = M - (M @ d) * d
        Vp = V - (V @ d) * d
        out["b"].append(float(np.linalg.norm(Mb)))
        out["mf"].append(float(-(Mb @ chord_axis)))
        out["t"].append(T)
        out["v"].append(float(np.linalg.norm(Vp)))
        out["vv"].append(V)
        out["mv"].append(M)
    b, mf, t, v = (np.array(out[k]) for k in ("b", "mf", "t", "v"))
    i0 = int(np.argmin(np.abs(s)))
    inside = s < -1e-9
    frac = np.clip(1 + s[inside] / socket_m, 0.0, 1.0)
    b[inside] = b[i0] * frac
    mf[inside] = mf[i0] * frac
    t[inside] = 0.0
    v[inside] = b[i0] / socket_m
    return Internal(s, b, mf, t, v, np.array(out["vv"]), np.array(out["mv"]))


# --------------------------------------------------------------------------
# Checks on one spar layout
# --------------------------------------------------------------------------


def skin_gj(geom: BWBGeometry, chord: float) -> float:
    """Torsional stiffness of the laminated EPP section ahead of the elevon hinge (N m^2), knocked down."""
    key = geom.outer_airfoil
    xh = 1 - geom.elevon_chord_fraction
    xs = np.linspace(0.0, xh, 60)
    t = np.array([airfoil_local_thickness(key, x) for x in xs]) * chord
    A = np.trapezoid(t, xs * chord) if hasattr(np, "trapezoid") else np.trapz(t, xs * chord)
    per = 2 * xh * chord * 1.02 + t[-1]
    gj_film = 4 * A**2 * FILM_SHEAR_MODULUS * FILM_THICKNESS / per
    j_foam = (np.trapezoid(t**3, xs * chord) if hasattr(np, "trapezoid") else np.trapz(t**3, xs * chord)) / 3
    return SKIN_TORSION_KNOCKDOWN * (gj_film + EPP_SHEAR_MODULUS * j_foam)


@dataclass
class Check:
    name: str
    value: float
    limit: float
    unit: str
    ok: bool
    where: str = ""

    @property
    def rf(self) -> float:
        return self.limit / self.value if self.value else float("inf")


@dataclass
class LayoutResult:
    layout: SparLayout
    checks: list[Check]
    ok: bool
    mass_kg: float  # both panels
    cost_usd: float  # both panels
    objective: float
    reversal_m_s: float
    tip_deflection_1g_m: float
    worst_case: str


def lip_bearing_stress(tube: Tube, load_n: float, contact_len_m: float) -> float:
    """Bearing pressure where the spar presses on the sleeve mouth: edge contact over half the
    projected width (a snug sleeve supports the wall; a loose one would make this much worse)."""
    return load_n / (0.5 * tube.od * contact_len_m)


def evaluate_layout(lay: SparLayout, pm: PanelModel, cases: list[LoadCase], ls: LoadSettings,
                    req: MissionRequirements, s_grid: np.ndarray, cases_1g: LoadCase,
                    internals: dict | None = None) -> LayoutResult:

    geom, line = pm.geom, pm.line
    checks = []
    # fit: every tube buried with cover where it runs inside the panel
    worst_fit = min(depth_at(geom, line, s) - lay.tube_at(s).od - 2 * ls.min_cover_m
                    for s in s_grid if s >= 0)
    checks.append(Check(f"tube buried in the foam with {ls.min_cover_m*1000:.0f} mm cover (spare depth)", worst_fit, 0.0,
                        "m+", worst_fit >= -1e-6))
    # strength
    internals = internals or {c.name: internal_loads(c, pm, s_grid, ls.socket_length_m) for c in cases}
    worst = (0.0, "", 0.0)
    for c in cases:
        it = internals[c.name]
        for i, s in enumerate(s_grid):
            tube = lay.tube_at(s)
            ei_tot, gj_tot = lay.section(s)
            # nested tubes share the moment in proportion to stiffness; the outer one is the critical surface
            share_b = tube.EI / ei_tot
            share_t = tube.GJ / gj_tot
            sig = c.factor * it.bending[i] * share_b / tube.Z
            tau = c.factor * abs(it.torsion[i]) * share_t * (tube.od / 2) / tube.J
            u = math.hypot(sig / tube.material.sigma_allow, tau / tube.material.tau_allow)
            if u > worst[0]:
                worst = (u, c.name, s)
    checks.append(Check("spar strength (ultimate, bending + torsion)", worst[0], ls.max_utilisation, "utilisation",
                        worst[0] <= ls.max_utilisation, f"{worst[1]} at {worst[2]*1000:.0f} mm"))
    # sleeve mouth crushing (s = 0): ultimate root reaction
    root_tube = lay.tube_at(0.0)
    worst_hoop = (0.0, "")
    for c in cases:
        it = internals[c.name]
        i0 = int(np.argmin(abs(s_grid)))
        R = c.factor * (it.shear[i0] + it.bending[i0] / ls.socket_length_m)
        u = lip_bearing_stress(root_tube, R, ls.lip_contact_d * root_tube.od) / root_tube.material.bearing_allow
        if u > worst_hoop[0]:
            worst_hoop = (u, c.name)
    checks.append(Check("crushing at the sleeve mouth (ultimate)", worst_hoop[0], ls.max_utilisation, "utilisation",
                        worst_hoop[0] <= ls.max_utilisation, worst_hoop[1]))
    # deflection in level flight at 1.5 kg (tube only, foam ignored)
    it1 = internal_loads(cases_1g, pm, s_grid, ls.socket_length_m)
    ei = np.array([lay.section(s)[0] for s in s_grid])
    kappa = it1.m_flap / ei
    pos = s_grid >= 0
    sp, kp = s_grid[pos], kappa[pos]
    slope = np.concatenate([[0], np.cumsum(0.5 * (kp[1:] + kp[:-1]) * np.diff(sp))])
    defl = np.concatenate([[0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * np.diff(sp))])
    tip_defl = float(defl[-1])
    checks.append(Check("tip deflection, level flight 1.5 kg", tip_defl, ls.max_tip_deflection_1g_m, "m",
                        tip_defl <= ls.max_tip_deflection_1g_m))
    # control reversal: elevon torque twists the panel nose-down (spar ~ at the aerodynamic centre)
    v_rev = reversal_speed(lay, pm, req)
    need = ls.reversal_margin * ls.v_dive_m_s
    checks.append(Check("elevon reversal speed", need, v_rev, "m/s", v_rev >= need))
    ok = all(ch.ok for ch in checks)
    mass = 2 * lay.mass_kg
    cost = 2 * lay.cost_usd
    obj = cost + ls.usd_per_kg * mass + ls.usd_per_joint * (len(lay.segments) - 1)
    return LayoutResult(lay, checks, ok, mass, cost, obj, v_rev, tip_defl, worst[1])


def reversal_speed(lay: SparLayout, pm: PanelModel, req: MissionRequirements) -> float:
    """Speed at which elevon roll effectiveness reaches zero (strip theory, root clamped in torsion)."""
    from . import aero as A

    geom, line = pm.geom, pm.line
    E = geom.elevon_chord_fraction
    th = math.acos(1 - 2 * (1 - E))
    cld = 2 * (math.pi - th + math.sin(th))  # thin aerofoil flap lift
    cmd = 0.5 * math.sin(th) * (1 - math.cos(th))  # magnitude, about c/4
    a = 2 * math.pi * geom.aspect_ratio / (geom.aspect_ratio + 2) / 1.0  # section slope with 3-D relief
    y, c, w, el = pm.strips_y, pm.strips_c, pm.strips_w, pm.elevon
    s = line.s_of_y(y)
    gj = np.array([lay.section(si)[1] + skin_gj(geom, ci) for si, ci in zip(s, c)])
    # unit-q twist: torque per span t(y) = c^2 cmd (per rad of elevon, per Pa)
    t = np.where(el, c**2 * cmd, 0.0)
    T = np.array([np.sum(t[y > yi] * w[y > yi]) + 0.5 * t[i] * w[i] for i, yi in enumerate(y)])  # torque inboard of strip
    ds = w / line.d[1]
    phi = np.cumsum(T / gj * ds)  # twist per (q * delta), nose-down
    lift_flap = np.sum(np.where(el, c * cld, 0.0) * w * y)  # rolling moment from the flap per (q delta)
    lift_twist = np.sum(c * a * phi * w * y)  # lost per q^2 delta
    q_rev = lift_flap / lift_twist
    rho = A.cruise_flight(req).rho
    return math.sqrt(2 * q_rev / rho)


# --------------------------------------------------------------------------
# Search over telescoping layouts
# --------------------------------------------------------------------------


def _furthest_fit(geom, line, od, ls, s_grid) -> float:
    ok = [s for s in s_grid if s >= 0 and depth_at(geom, line, s) >= od + 2 * ls.min_cover_m - 1e-9]
    # stations must be contiguous from the root
    last = -ls.socket_length_m
    for s in s_grid[s_grid >= 0]:
        if s in ok:
            last = s
        else:
            break
    return float(last)


def candidate_layouts(geom: BWBGeometry, line: SparLine, ls: LoadSettings, s_grid: np.ndarray,
                      step_fracs=(1.0, 0.85, 0.7, 0.55)) -> list[SparLayout]:
    """Telescoping 1-3 tube layouts. Each step sits at the furthest station the bigger tube fits,
    or earlier (fractions of that), and the next tube overlaps back into it."""
    L = line.length
    out = []
    by_od = {}
    for t in CATALOGUE:
        by_od.setdefault(round(t.od * 1000), []).append(t)
    for od_root in sorted(by_od):
        for chain_len in (1, 2, 3):
            ods = [od_root - 2 * k for k in range(chain_len)]
            if any(o not in by_od for o in ods):
                continue
            for tubes in itertools.product(*(by_od[o] for o in ods)):
                fits = [_furthest_fit(geom, line, t.od, ls, s_grid) for t in tubes]
                if fits[0] < 0:
                    continue
                if chain_len == 1:
                    if fits[0] >= L - 1e-6:
                        out.append(SparLayout((Segment(tubes[0], -ls.socket_length_m, L),)))
                    continue
                if fits[-1] < L - 1e-6:
                    continue
                for fr in itertools.product(step_fracs, repeat=chain_len - 1):
                    steps = []
                    ok = True
                    prev = 0.0
                    for k, f_ in enumerate(fr):
                        st = min(fits[k], L) * f_
                        if st < prev + ls.min_piece_m:
                            ok = False
                            break
                        if st > fits[k] + 1e-9 or st >= L - ls.min_piece_m:
                            ok = False
                            break
                        steps.append(st)
                        prev = st
                    if not ok:
                        continue
                    segs = [Segment(tubes[0], -ls.socket_length_m, steps[0])]
                    for k in range(1, chain_len):
                        s0 = steps[k - 1] - ls.telescope_overlap_m
                        s1 = steps[k] if k < chain_len - 1 else L
                        segs.append(Segment(tubes[k], s0, s1))
                    out.append(SparLayout(tuple(segs)))
    # de-duplicate
    seen, uniq = set(), []
    for lay in out:
        if lay.label not in seen:
            seen.add(lay.label)
            uniq.append(lay)
    return uniq


# --------------------------------------------------------------------------
# Joint: sleeve, ribs, anti-rotation pin, carry-through, retention
# --------------------------------------------------------------------------


def joint_checks(lay: SparLayout, pm: PanelModel, cases: list[LoadCase], ls: LoadSettings,
                 internals: dict) -> dict:
    from . import weights as W

    geom, line = pm.geom, pm.line
    root = lay.tube_at(0.0)
    sleeve = next((t for t in CATALOGUE if abs(t.id - root.od) < 1e-6 and t.material is ROLL_WRAPPED), None)
    s_grid = next(iter(internals.values())).s
    i0 = int(np.argmin(abs(s_grid)))
    rows = []
    # carry-through: moment about the roll axis at the sleeve's inner end, carried across the
    # bay by carbon strip caps top and bottom (couple over the pod's inside height there)
    x_in = line.point(-ls.socket_length_m)
    lo, h_in = W.pod_inside_height(geom, float(x_in[0]))
    h_cap = h_in - 0.004
    pin_arm = ls.pin_offset_frac * geom.root_chord_m
    for c in cases:
        it = internals[c.name]
        Mb, V, T = it.bending[i0], it.shear[i0], abs(it.torsion[i0])
        R_out = V + Mb / ls.socket_length_m
        R_in = Mb / ls.socket_length_m
        p_rib = max(R_out, R_in) / (sleeve.od * ls.rib_boss_m) if sleeve else float("nan")
        tau_bond = max(R_out, R_in) / (math.pi * sleeve.od * ls.rib_boss_m) if sleeve else float("nan")
        F_pin = T / pin_arm
        p_pin = F_pin / (ls.pin_d_m * ls.rib_thickness_m)
        F = c.forces
        Mx = abs(float(np.sum(F[:, 1] * F[:, 5] - F[:, 2] * F[:, 4])))  # roll-axis moment at the centreline
        F_cap = Mx / h_cap
        axial = float(it.shear_vec[i0] @ line.d)  # + = pulls the panel out of the pod
        rows.append(dict(case=c.name, factor=c.factor, repeated=c.repeated, M_root=Mb, V_root=V, T_root=T,
                         R_out=R_out, R_in=R_in, p_rib=p_rib, tau_bond=tau_bond, F_pin=F_pin, p_pin=p_pin, M_carry=Mx, F_cap=F_cap,
                         pull_out=axial))

    def util(key, allow, use_rep=True):
        best = (0.0, "")
        for r in rows:
            lim = allow * (0.5 if (use_rep and r["repeated"]) else 1.0)
            u = r[key] * r["factor"] / lim
            if u > best[0]:
                best = (u, r["case"])
        return best

    cap_area = CAP_STRIP[0] * CAP_STRIP[1]
    caps_needed = 1
    while max(r["F_cap"] * r["factor"] for r in rows) / (caps_needed * cap_area) > CAP_STRIP_ALLOW:
        caps_needed += 1
    F_cap_ult = max(r["F_cap"] * r["factor"] for r in rows)
    bond_len = max(r["F_cap"] * r["factor"] / (caps_needed * CAP_STRIP[0] * BOND_SHEAR_ALLOW * (0.5 if r["repeated"] else 1.0))
                   for r in rows)
    pull = max(r["pull_out"] for r in rows)
    caps_needed = max(caps_needed, 1)
    out = dict(
        rows=rows, sleeve=sleeve, socket_length_m=ls.socket_length_m,
        rib=util("p_rib", PRINT_BEARING_ALLOW), bond=util("tau_bond", BOND_SHEAR_ALLOW), pin=util("p_pin", PRINT_BEARING_ALLOW),
        carry_height_m=h_cap, carry_x_m=float(x_in[0]), carry_y_m=float(x_in[1]), caps_per_side=caps_needed,
        cap_force_ult_n=F_cap_ult, cap_bond_length_m=max(bond_len, 0.03),
        retention_n=max(pull, 0.0), retention_design_n=max(2.0 * pull * ls.safety_factor, 15.0),
        sleeve_length_m=ls.socket_length_m + 0.02,
    )
    return out


def restraint_loads(build, ls: LoadSettings) -> list[tuple[str, float, float, float]]:
    """(part, mass g, fore-aft N at the catapult, vertical N in a belly landing) for parts >= 10 g in the pod."""

    n_land = ls.land_sink_m_s**2 / (2 * G0 * ls.land_stroke_m)
    a_cat = ls.cat_release_m_s**2 / (2 * ls.cat_stroke_m) * ls.cat_peak_factor
    a_hand = ls.hand_release_m_s**2 / (2 * ls.hand_stroke_m) * ls.hand_peak_factor
    out = []
    for it in build.items:
        if it.group in ("structure", "wiring") or abs(it.cg[1]) > build.geom.pod_half_width_m:
            continue
        m = it.mass_as_built_kg
        if m < 0.010:
            continue
        out.append((it.name.split(":")[0].split("(")[0].strip(), m * 1000, ls.safety_factor * m * max(a_cat, a_hand),
                    ls.safety_factor * m * (n_land + 1) * G0))
    return sorted(out, key=lambda r: -r[1])


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------


@dataclass
class M22:
    geom: BWBGeometry
    settings: LoadSettings
    panel: PanelModel
    env: dict
    cases: list[LoadCase]
    internals: dict
    s_grid: np.ndarray
    chosen: LayoutResult
    ranked: list[LayoutResult]
    n_candidates: int
    current: LayoutResult  # the M1.5 assumption (10x8 pultruded, full length)
    joint: dict
    restraints: list
    landing_sweep: list[tuple[float, float, float]]  # (sink, n, utilisation)
    mass_built_kg: float
    spar_mass_old_g: float
    spar_mass_new_g: float
    extra_items_g: float  # sleeves + caps + overlaps beyond the M1.5 assumption
    mass_new_kg: float = 0.0
    cost_delta_usd: float = 0.0
    battery_x_old_m: float = 0.0
    battery_x_new_m: float = 0.0


def load_geometry(result_json: str | Path | None = None) -> BWBGeometry:
    """The merged optimum (docs/optimizer_result.json), or the M1.3 baseline if there is none."""
    from .optimize import make_geometry

    p = Path(result_json) if result_json else Path(__file__).resolve().parents[2] / "docs" / "optimizer_result.json"
    if p.exists():
        return make_geometry(json.loads(p.read_text())["best"]["x"])
    from .geometry import BWBGeometry

    return BWBGeometry()


def run(req: MissionRequirements, geom: BWBGeometry | None = None, ls: LoadSettings | None = None,
        use_vlm: bool = True, progress=print) -> M22:
    from . import weights as W

    ls = ls or LoadSettings()
    geom = geom or load_geometry()
    # Weights with the M1.5 spar (10x8 full length) to get the as-built mass and the non-spar panel masses
    build_old = W.build_up(geom, W.default_propulsion(), W.BuildChoices(spar_layout=None))
    progress("lift shape ...")
    pm = panel_model(geom, req, build_old, use_vlm=use_vlm)
    line = pm.line
    s_grid = np.r_[np.linspace(-ls.socket_length_m, 0, 6)[:-1], np.linspace(0, line.length, 61)]
    mass_built = build_old.mass_kg(True)
    # spar line mass ~ 10x8 for the loads (its effect on the loads is small; re-checked below)
    cases, env = load_cases(pm, req, ls, mass_built, spar_kg_per_m=0.045)
    internals = {c.name: internal_loads(c, pm, s_grid, ls.socket_length_m) for c in cases}
    cruise_1g = _flight_case(pm, ls.design_mass_kg * G0, 1.0, "Level flight 1.5 kg", ls)
    cands = candidate_layouts(geom, line, ls, s_grid)
    progress(f"{len(cands)} candidate spar layouts ...")
    res = [evaluate_layout(lay, pm, cases, ls, req, s_grid, cruise_1g, internals) for lay in cands]
    ok = sorted((r for r in res if r.ok), key=lambda r: r.objective)
    if not ok:
        raise RuntimeError("no spar layout passes every check")
    chosen = ok[0]
    cur = SparLayout((Segment(Tube(0.010, 0.008, PULTRUDED, 6.0), -ls.socket_length_m, line.length),))
    current = evaluate_layout(cur, pm, cases, ls, req, s_grid, cruise_1g, internals)
    joint = joint_checks(chosen.layout, pm, cases, ls, internals)
    # landing sweep: utilisation of the chosen spar vs sink rate
    sweep = []
    for sink in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        n = sink**2 / (2 * G0 * ls.land_stroke_m)
        mp = pm.mass_pts
        f = np.column_stack([mp[:, 1:], np.zeros((len(mp), 2)), -n * mp[:, 0] * G0])
        c = LoadCase("sweep", "landing", f, None, 1.0)
        it = internal_loads(c, pm, s_grid, ls.socket_length_m)
        u = max(it.bending[i] * chosen.layout.tube_at(s).EI / chosen.layout.section(s)[0]
                / chosen.layout.tube_at(s).Z / chosen.layout.tube_at(s).material.sigma_allow
                for i, s in enumerate(s_grid))
        sweep.append((sink, n, float(u)))
    old_spar = sum(it.mass_as_built_kg for it in build_old.items if it.name.startswith("Spar")) * 1000
    build_new = W.build_up(geom, W.default_propulsion(), W.BuildChoices(spar_layout=to_weights_layout(chosen.layout, line)))
    new_spar = sum(it.mass_as_built_kg for it in build_new.items if it.name.startswith("Spar")) * 1000
    extra = sum(it.mass_as_built_kg for it in build_new.items if it.name.startswith("Spar carry")) * 1000
    restraints = restraint_loads(build_new, ls)
    return M22(geom, ls, pm, env, cases, internals, s_grid, chosen, ok, len(cands), current, joint, restraints,
               sweep, mass_built, old_spar, new_spar, extra, build_new.mass_kg(True),
               build_new.cost_usd - build_old.cost_usd, build_old.battery_x_m, build_new.battery_x_m)


def to_weights_layout(lay: SparLayout, line: SparLine) -> tuple:
    """The layout as weights.SparPiece tuples (stations as fractions of the panel spar length)."""
    from .weights import SparPiece

    L = line.length
    return tuple(SparPiece(seg.tube.od, seg.tube.id, seg.s0 / L, seg.s1 / L, seg.tube.material.density,
                           seg.tube.usd_per_m, seg.tube.material.name) for seg in lay.segments)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def report(r: M22) -> str:
    L = []
    out = L.append
    g, ls, ch = r.geom, r.settings, r.chosen
    out("== Manta Mapper: spar, load cases and the panel joint (M2.2) ==")
    out(f"Design: span {g.span_m:.3f} m, {g.reference_area_m2:.3f} m^2, outer panel spar {r.panel.line.length*1000:.0f} mm "
        f"swept {g.spar_sweep_deg:.1f} deg; {r.mass_built_kg*1000:.0f} g as built, sized at {ls.design_mass_kg:g} kg")
    out(f"Lift shape: {r.panel.source}; each outer panel + fin carries {r.panel.panel_lift_frac*100:.0f} % of the lift "
        f"(fin {r.panel.fin_frac*100:.1f} %); CL_alpha {r.panel.cla_per_rad:.2f}/rad")
    out(f"Factor of safety {ls.safety_factor:g} on every limit load (AC-10); repeated cases (launch, landing, tip "
        "strike) hold printed parts to half their strength")
    out("")
    out("-- Flight envelope (limit) --")
    for k in ("design", "as built"):
        e = r.env[k]
        out(f"  {e['mass']:.2f} kg: manoeuvre {ls.n_pos:+.1f} / {ls.n_neg:+.1f} g")
        for gst in e["gusts"]:
            out(f"     gust {gst['gust']:.1f} m/s at {gst['speed']} {gst['v']:.0f} m/s: {gst['n']:+.1f} / {gst['n_neg']:+.1f} g "
                f"(stall caps at {gst['n_cap']:.1f} g) -> lift {gst['lift_pos']:.0f} / {gst['lift_neg']:.0f} N")
    out("  A wing this lightly loaded is thrown around by gusts harder than by the 4 g of AC-10, so the gust lines set "
        "the flight case. The spar is sized for both.")
    out("")
    out("-- Limit loads at the panel root (the joint) and where they come from --")
    out(f"{'case':58s} {'M_bend':>7s} {'shear':>6s} {'torque':>7s} {'pull':>6s}")
    out(f"{'':58s} {'N m':>7s} {'N':>6s} {'N m':>7s} {'N':>6s}")
    for row in r.joint["rows"]:
        out(f"{row['case'][:58]:58s} {row['M_root']:7.2f} {row['V_root']:6.1f} {row['T_root']:7.2f} {row['pull_out']:6.1f}")
    out("")
    out(f"-- Spar: {r.n_candidates} telescoping layouts searched; {len(r.ranked)} pass every check --")
    out(f"CHOSEN: {ch.layout.label}")
    out(f"  both panels: {ch.mass_kg*1000:.0f} g, ${ch.cost_usd:.0f}")
    for c in ch.checks:
        tag = "OK " if c.ok else "FAIL"
        if c.unit == "m+":
            out(f"  [{tag}] {c.name}: {c.value*1000:+.1f} mm at the tightest station")
        elif c.unit == "m":
            out(f"  [{tag}] {c.name}: {c.value*1000:.1f} mm (limit {c.limit*1000:.1f} mm) {c.where}")
        elif c.unit == "m/s":
            out(f"  [{tag}] {c.name}: {c.limit:.0f} m/s (need {c.value:.0f} m/s = {ls.reversal_margin:g} x V_D)")
        else:
            out(f"  [{tag}] {c.name}: {c.value:.2f} of allowable, limit {c.limit:.2f} ({c.where})")
    out("")
    out("Runner-up layouts (objective = cost + $100/kg x mass + $4 per bonded joint, both panels):")
    for lr in r.ranked[1:6]:
        out(f"  ${lr.objective:5.1f}  {lr.mass_kg*1000:4.0f} g  ${lr.cost_usd:3.0f}  reversal {lr.reversal_m_s:3.0f} m/s  "
            f"defl {lr.tip_deflection_1g_m*1000:4.1f} mm  {lr.layout.label}")
    out("")
    cur = r.current
    out(f"M1.5 assumption, 10x8 pultruded full length: {'passes' if cur.ok else 'FAILS'}")
    for c in cur.checks:
        if not c.ok:
            if c.unit == "m/s":
                out(f"  fails {c.name}: {c.limit:.0f} m/s, need {c.value:.0f}")
            elif c.unit == "m+":
                out(f"  fails {c.name}: {c.value*1000:+.1f} mm")
            elif c.unit == "m":
                out(f"  fails {c.name}: {c.value*1000:.1f} mm vs {c.limit*1000:.1f} mm")
            else:
                out(f"  fails {c.name}: {c.value:.2f} of allowable ({c.where})")
    out("")
    j = r.joint
    out("-- Joint: panel spar -> sleeve -> printed ribs -> carry-through --")
    sl = j["sleeve"]
    out(f"Sleeve: {sl.label if sl else 'n/a'} tube, {j['sleeve_length_m']*1000:.0f} mm, bonded through two printed ribs "
        f"{ls.rib_thickness_m*1000:.0f} mm thick at its ends, each with a {ls.rib_boss_m*1000:.0f} mm boss round the sleeve; "
        f"spar goes {ls.socket_length_m*1000:.0f} mm in")
    out(f"  rib boss bearing: {j['rib'][0]:.2f} of allowable ({j['rib'][1]})")
    out(f"  sleeve-to-boss epoxy bond: {j['bond'][0]:.2f} of allowable ({j['bond'][1]})")
    out("  (repeated cases count against half the printed/bond strength)")
    out(f"  anti-rotation pin {ls.pin_d_m*1000:.0f} mm at {ls.pin_offset_frac:.0%} root chord behind the spar: bearing "
        f"{j['pin'][0]:.2f} of allowable ({j['pin'][1]})")
    out(f"Carry-through at x = {j['carry_x_m']*1000:.0f} mm (sleeve inner ends at y = +-{j['carry_y_m']*1000:.0f} mm): "
        f"{j['caps_per_side']} x {CAP_STRIP[0]*1000:.0f}x{CAP_STRIP[1]*1000:.0f} mm carbon strip in the floor and "
        f"{j['caps_per_side']} in the roof, {j['carry_height_m']*1000:.0f} mm apart")
    out(f"  ultimate cap force {j['cap_force_ult_n']:.0f} N; bond each strip end >= {j['cap_bond_length_m']*1000:.0f} mm "
        f"to the rib")
    out(f"Panel retention: launch pulls each panel out along the swept spar with {j['retention_n']:.1f} N (limit); "
        f"hold it with >= {j['retention_design_n']:.0f} N and let go above ~3x that so a cartwheel pops the panel off "
        "instead of breaking the sleeve")
    out("")
    out("-- Belly-landing sweep (AC-13): spar utilisation at limit, no factor --")
    for sink, n, u in r.landing_sweep:
        out(f"  sink {sink:.1f} m/s -> {n:5.1f} g over {ls.land_stroke_m*1000:.0f} mm: spar at {u*100:4.1f} % of strength")
    out("  The panels are light, so even a 4 m/s arrival is far from breaking the spar: landings are a skid,")
    out("  fin and pod-floor problem, not a spar problem.")
    out("")
    out("-- Restraint loads for parts in the pod (ultimate) --")
    out(f"{'part':44s} {'g':>5s} {'fore-aft N':>11s} {'down N':>8s}")
    for name, m, fx, fz in r.restraints:
        out(f"{name[:44]:44s} {m:5.0f} {fx:11.1f} {fz:8.1f}")
    out("")
    out("-- Weights feedback --")
    out(f"Spar as built: {r.spar_mass_old_g:.0f} g (M1.5, 10x8 full length) -> {r.spar_mass_new_g:.0f} g "
        f"(includes sleeves and carry-through caps {r.extra_items_g:.0f} g); change {r.spar_mass_new_g - r.spar_mass_old_g:+.0f} g")
    out(f"Aircraft as built: {r.mass_built_kg*1000:.0f} g -> {r.mass_new_kg*1000:.0f} g "
        f"({r.mass_new_kg/ls.design_mass_kg:.0%} of the {ls.design_mass_kg:g} kg sizing mass); parts {r.cost_delta_usd:+.0f} USD; "
        f"battery moves {1000*(r.battery_x_new_m - r.battery_x_old_m):+.0f} mm to hold the CG target")
    out("weights.py now uses this spar by default (BuildChoices.spar_layout = M22_SPAR).")
    return "\n".join(L)


def to_json(r: M22) -> dict:
    ch = r.chosen
    return dict(
        milestone="M2.2",
        settings=asdict(r.settings),
        span_m=r.geom.span_m,
        spar=[dict(tube=s.tube.label, od_mm=s.tube.od * 1000, id_mm=s.tube.id * 1000, from_mm=s.s0 * 1000,
                   to_mm=s.s1 * 1000) for s in ch.layout.segments],
        spar_mass_g_both=ch.mass_kg * 1000,
        spar_cost_usd_both=ch.cost_usd,
        checks=[dict(name=c.name, value=c.value, limit=c.limit, unit=c.unit, ok=c.ok, where=c.where) for c in ch.checks],
        reversal_m_s=ch.reversal_m_s,
        tip_deflection_1g_mm=ch.tip_deflection_1g_m * 1000,
        joint={k: (v.label if isinstance(v, Tube) else v) for k, v in r.joint.items() if k != "rows"},
        root_loads=r.joint["rows"],
        weights=dict(spar_old_g=r.spar_mass_old_g, spar_new_g=r.spar_mass_new_g, aircraft_old_g=r.mass_built_kg * 1000,
                     aircraft_new_g=r.mass_new_kg * 1000, cost_delta_usd=r.cost_delta_usd),
    )


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_loads(r: M22, path: str):  # pragma: no cover - visual
    import matplotlib.pyplot as plt

    s = r.s_grid * 1000
    lay = r.chosen.layout
    cap = np.array([lay.tube_at(x).material.sigma_allow * lay.tube_at(x).Z * lay.section(x)[0] / lay.tube_at(x).EI
                    for x in r.s_grid])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    kinds = {"flight": "#2a6fdb", "launch": "#d1802e", "landing": "#2a9d55", "tip": "#8e44ad"}
    seen = set()
    for c in r.cases:
        it = r.internals[c.name]
        lab = c.kind if c.kind not in seen else None
        seen.add(c.kind)
        ax.plot(s, it.bending * c.factor, color=kinds[c.kind], lw=1.4, alpha=0.85, label=lab and f"{lab} (ultimate)")
    ax.plot(s, cap, color="#222222", lw=2, label="chosen spar: bending strength")
    ax.axvline(0, color="#999999", ls=":", lw=1)
    ax.text(2, ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] else 0, "panel root", color="#777777", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("station along the spar from the panel root, mm (negative: inside the pod sleeve)")
    ax.set_ylabel("bending moment, N m (log)")
    ax.set_title("ultimate bending moment by case vs spar strength")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax = axes[1]
    xs = np.linspace(0, r.panel.line.length, 100)
    depth = [depth_at(r.geom, r.panel.line, x) * 1000 for x in xs]
    ax.plot(xs * 1000, depth, color="#444444", lw=2, label="foam depth at the spar")
    ax.plot(xs * 1000, np.array(depth) - 2 * r.settings.min_cover_m * 1000, color="#444444", ls="--", lw=1,
            label=f"max tube OD ({r.settings.min_cover_m*1000:.0f} mm cover)")
    for seg in lay.segments:
        ax.fill_between([seg.s0 * 1000, seg.s1 * 1000], [0, 0], [seg.tube.od * 1000] * 2, alpha=0.35,
                        label=seg.tube.label)
    ax.axvline(0, color="#999999", ls=":", lw=1)
    ax.set_xlabel("station along the spar, mm")
    ax.set_ylabel("mm")
    ax.set_title("telescoping spar inside the panel")
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vn(r: M22, path: str):  # pragma: no cover - visual
    import matplotlib.pyplot as plt
    from . import aero as A

    ls = r.settings
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for k, col in (("design", "#2a6fdb"), ("as built", "#d1802e")):
        e = r.env[k]
        W = e["W"]
        v = np.linspace(0.1, ls.v_dive_m_s, 200)
        rho = A.cruise_flight(MissionRequirements()).rho
        cap = 0.5 * rho * v**2 * r.geom.reference_area_m2 * ls.cl_max / W
        ax.plot(v, np.minimum(cap, ls.n_pos), color=col, lw=2, label=f"{e['mass']:.2f} kg manoeuvre (stall-limited)")
        ax.plot(v, np.maximum(-0.6 * cap, ls.n_neg), color=col, lw=2)
        g = e["gusts"]
        vv = [0] + [x["v"] for x in g]
        ax.plot(vv, [1] + [min(x["n"], x["n_cap"]) for x in g], color=col, ls="--", lw=1.4, label=f"{e['mass']:.2f} kg gust lines")
        ax.plot(vv, [1] + [x["n_neg"] for x in g], color=col, ls="--", lw=1.4)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("airspeed, m/s")
    ax.set_ylabel("load factor, g")
    ax.set_title("V-n diagram: manoeuvre (AC-10) and gusts")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(prog="manta_mapper.structures", description="M2.2 spar sizing and load cases.")
    p.add_argument("mission", nargs="?", help="mission TOML (defaults built in)")
    p.add_argument("--design", help="optimizer result JSON (default docs/optimizer_result.json)")
    p.add_argument("--schrenk", action="store_true", help="Schrenk lift shape instead of the VLM (no AeroSandbox)")
    p.add_argument("--report", help="save the text report here")
    p.add_argument("--json", help="save the result here")
    p.add_argument("--plots", help="directory for the plots")
    a = p.parse_args(argv)
    req = MissionRequirements.from_toml(a.mission) if a.mission else MissionRequirements()
    r = run(req, load_geometry(a.design), use_vlm=not a.schrenk)
    txt = report(r)
    print(txt)
    if a.report:
        Path(a.report).write_text(txt + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(to_json(r), indent=1, default=float))
    if a.plots:
        Path(a.plots).mkdir(parents=True, exist_ok=True)
        plot_loads(r, str(Path(a.plots) / "structures_loads.png"))
        plot_vn(r, str(Path(a.plots) / "structures_vn.png"))


if __name__ == "__main__":  # pragma: no cover
    main()
