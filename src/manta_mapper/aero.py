"""Whole-aircraft aerodynamics (milestone M1.4), lowest fidelity first.

Two cheap tools, each doing the part it is good at:

* **AeroSandbox vortex lattice (VLM)** for everything that depends on the
  planform: lift slope, pitching moment, neutral point, spanwise loading,
  induced drag, elevon power and the rate derivatives used for the flight
  modes. It is inviscid and treats the wing as a thin cambered sheet.
* **NeuralFoil** (from M1.2) for what the VLM can't see: profile drag and the
  stall of each spanwise strip, at that strip's own Reynolds number, using the
  as-built sections (1 mm blunt trailing edge, early transition, AC-15).

Gluing them together is classic strip theory: the VLM says what lift
coefficient each strip carries, NeuralFoil says what that strip's section drag
is at that lift coefficient, and where it stalls.

What the VLM can't do and how it is handled here:

* **Elevons.** AeroSandbox's VLM ignores ``ControlSurface`` objects, so an
  elevon deflection is modelled by deflecting the camber line of the sections
  it covers (the flap bends the camber line about the hinge). Thin-airfoil
  theory overstates real flap effectiveness, especially at low Reynolds number
  with a blunt trailing edge, so every deflection is multiplied by
  :data:`ELEVON_EFFECTIVENESS` (0.75). A lower number is the cautious choice:
  it makes every trim deflection larger, never smaller.
* **Stall.** The VLM lift curve never bends over. The wing's maximum lift is
  found with the "critical section" method: increase angle of attack until the
  first strip reaches its own section CL_max.
* **Odds and ends.** Skid, motor and prop hub, pitot, antennas, seams and
  control gaps are added as an explicit, itemised drag allowance
  (:data:`MISC_DRAG_AREA_M2`, :data:`INTERFERENCE_FACTOR`).

AeroSandbox's AeroBuildup was tried as a cross-check and rejected for the
pitching moment: on a plain swept, tapered wing with a symmetric section it
puts the neutral point at 16-20 % MAC, where the VLM and textbook values are
about 24-26 %. The VSPAERO run in M1.7 is the independent check (PIPE-4).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .airfoils import BUILT_N_CRIT, BUILT_TE_THICKNESS_M, NOMINAL_N_CRIT, Polar, run_polar
from .atmosphere import dynamic_viscosity, isa, speed_of_sound
from .geometry import BWBGeometry, _blended_airfoil, _load_airfoil
from .requirements import G0, MissionRequirements

# --------------------------------------------------------------------------
# Modelling settings (each one is a judgement call; the reason is next to it)
# --------------------------------------------------------------------------

#: Real elevon lift per degree / thin-airfoil value. Raymer and DATCOM give
#: about 0.8-0.9 for small deflections of a sealed plain flap at high Reynolds
#: number; 0.75 allows for the low Reynolds number, the 1 mm blunt trailing
#: edge and an unsealed hinge gap. Lower is the cautious direction.
ELEVON_EFFECTIVENESS = 0.75

#: Spanwise width over which the camber changes at each end of an elevon
#: (a zero-width step breaks the vortex lattice). 10 mm is converged: 5 mm and
#: 20 mm give elevon power within 5 % of each other.
ELEVON_TRANSITION_M = 0.010

#: VLM sections are split so none is wider than this; each gets
#: :data:`SPANWISE_RESOLUTION` panels, so spanwise panels are ~12 mm wide.
MAX_SECTION_WIDTH_M = 0.05
SPANWISE_RESOLUTION = 4
CHORDWISE_RESOLUTION = 10

#: Drag of items that are not wing or fin, as drag area D/q (m^2).
MISC_DRAG_AREA_M2 = {
    # 20 mm deep x 60 mm wide EPP skid, rounded nose: Cd ~0.4 on frontal area
    "landing skid": 0.4 * 0.020 * 0.060,
    # 28 mm motor + folding-prop hub at the pod trailing edge, half in the wake
    "motor and prop hub": 0.3 * math.pi * 0.014**2,
    # pitot tube, GPS / Remote ID / receiver antennas, servo horns and linkages
    "pitot, antennas, servo horns": 1.5e-4,
    # camera window recess under the pod
    "camera window": 0.2e-4,
}

#: Extra fraction on wing + fin profile drag for hatch seams, panel joints,
#: elevon hinge gaps, tape edges and the wing-fin junction.
INTERFERENCE_FACTOR = 0.15


# --------------------------------------------------------------------------
# Flight conditions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Flight:
    """Airspeed and altitude (MSL) in the design atmosphere."""

    velocity: float  # true airspeed, m/s
    altitude_m: float
    temperature_offset_K: float = 0.0

    @property
    def _isa(self):
        return isa(self.altitude_m, self.temperature_offset_K)

    @property
    def rho(self) -> float:
        return self._isa[2]

    @property
    def mu(self) -> float:
        return dynamic_viscosity(self._isa[0])

    @property
    def mach(self) -> float:
        return self.velocity / speed_of_sound(self._isa[0])

    @property
    def q(self) -> float:
        return 0.5 * self.rho * self.velocity**2

    def reynolds(self, chord_m: float) -> float:
        return self.rho * self.velocity * chord_m / self.mu

    def at(self, velocity: float) -> "Flight":
        return Flight(velocity, self.altitude_m, self.temperature_offset_K)

    def atmosphere(self):
        import aerosandbox as asb

        return asb.Atmosphere(altitude=self.altitude_m, temperature_deviation=self.temperature_offset_K)


def survey_altitude_m(req: MissionRequirements) -> float:
    from .survey import plan_survey

    return req.environment.field_elevation_m + plan_survey(req).altitude_m


def cruise_flight(req: MissionRequirements, velocity: float | None = None) -> Flight:
    """Survey altitude above the field, design-day temperature."""
    v = req.assumptions.cruise_speed_m_s if velocity is None else velocity
    return Flight(v, survey_altitude_m(req), req.environment.temperature_offset_K)


def field_flight(req: MissionRequirements, velocity: float) -> Flight:
    """At the field elevation (hand launch, stall, landing), design-day temperature."""
    return Flight(velocity, req.environment.field_elevation_m, req.environment.temperature_offset_K)


def cl_level(mass_kg: float, geom: BWBGeometry, flight: Flight, load_factor: float = 1.0) -> float:
    return load_factor * mass_kg * G0 / (flight.q * geom.reference_area_m2)


# --------------------------------------------------------------------------
# VLM model with elevons
# --------------------------------------------------------------------------


def blend_at(geom: BWBGeometry, y: float) -> float:
    """Section blend fraction at span station y: 0 = pod airfoil, 1 = outer airfoil."""
    y = abs(y)
    if y <= geom.pod_half_width_m:
        return 0.0
    if y >= geom.root_y_m:
        return 1.0
    return (y - geom.pod_half_width_m) / geom.blend_width_m


def chord_at(geom: BWBGeometry, y: np.ndarray | float) -> np.ndarray:
    st = geom.stations()
    return np.interp(np.abs(y), [s["le"][1] for s in st], [s["chord"] for s in st])


def _station_list(geom: BWBGeometry, deflection_deg: float) -> list[dict]:
    """Right-half stations root to tip, split into narrow sections, with the elevon deflection inserted.

    An elevon segment [a, b] becomes: a (clean), a + gap (deflected), ...,
    b - gap (deflected), b (clean), so the deflection ramps in over
    :data:`ELEVON_TRANSITION_M` at each end instead of stepping.
    """
    st = geom.stations()

    def lerp(a, b, t, defl):
        return dict(le=np.asarray(a["le"], float) + t * (np.asarray(b["le"], float) - np.asarray(a["le"], float)),
                    chord=a["chord"] + t * (b["chord"] - a["chord"]), twist=a["twist"] + t * (b["twist"] - a["twist"]),
                    blend=a["blend"] + t * (b["blend"] - a["blend"]), deflection=defl)

    out: list[dict] = []
    for a, b in zip(st[:-1], st[1:]):
        width = b["le"][1] - a["le"][1]
        if a["elevon"] and deflection_deg != 0.0:
            g = min(0.25, ELEVON_TRANSITION_M / max(width, 1e-9))
            ts = [0.0] + list(np.linspace(g, 1 - g, max(2, math.ceil((1 - 2 * g) * width / MAX_SECTION_WIDTH_M) + 1)))
            defl = [0.0] + [deflection_deg] * (len(ts) - 1)
        else:
            ts = list(np.linspace(0.0, 1.0, max(1, math.ceil(width / MAX_SECTION_WIDTH_M)) + 1))[:-1]
            defl = [0.0] * len(ts)
        out += [lerp(a, b, t, d) for t, d in zip(ts, defl)]
    last = st[-1]
    out.append(dict(le=np.asarray(last["le"], float), chord=last["chord"], twist=last["twist"],
                    blend=last["blend"], deflection=0.0))
    return out


@lru_cache(maxsize=256)
def _deflected_airfoil(pod: str, outer: str, blend: float, deflection_deg: float, hinge_x: float):
    af = _blended_airfoil(pod, outer, blend)
    if deflection_deg == 0.0:
        return af
    return af.add_control_surface(deflection=deflection_deg * ELEVON_EFFECTIVENESS, hinge_point_x=hinge_x)


def _xsecs(geom: BWBGeometry, stations: list[dict], side: int):
    import aerosandbox as asb

    hinge = 1 - geom.elevon_chord_fraction
    xs = []
    for s in stations:
        le = s["le"].copy()
        le[1] *= side
        af = _deflected_airfoil(geom.pod_airfoil, geom.outer_airfoil, round(float(s["blend"]), 3),
                                round(float(s["deflection"]), 4), hinge)
        xs.append(asb.WingXSec(xyz_le=list(le), chord=s["chord"], twist=s["twist"], airfoil=af))
    # VLM panels must run in +y. The left half is listed tip to root.
    return xs if side > 0 else xs[::-1]


def build_airplane(geom: BWBGeometry, xyz_ref=(0.0, 0.0, 0.0), elevator_deg: float = 0.0,
                   aileron_deg: float = 0.0):
    """AeroSandbox ``Airplane`` with the elevons deflected.

    ``elevator_deg``: both elevons together, trailing edge down positive
    (nose-down). ``aileron_deg``: differential, right trailing edge up and
    left down positive (rolls right). Each half is a separate wing so the
    two sides can deflect differently.
    """
    import aerosandbox as asb

    right = _station_list(geom, elevator_deg - aileron_deg)
    left = _station_list(geom, elevator_deg + aileron_deg)
    wr = asb.Wing(name="wing right", symmetric=False, xsecs=_xsecs(geom, right, +1))
    wl = asb.Wing(name="wing left", symmetric=False, xsecs=_xsecs(geom, left, -1))
    fin = asb.Wing(
        name="tip fins", symmetric=True,
        xsecs=[asb.WingXSec(xyz_le=list(f["le"]), chord=f["chord"], twist=f["twist"],
                            airfoil=_load_airfoil(f["airfoil"])) for f in geom.fin_stations()],
    )
    return asb.Airplane(name=geom.name, xyz_ref=list(xyz_ref), wings=[wr, wl, fin],
                        s_ref=geom.reference_area_m2, c_ref=geom.mac_m, b_ref=geom.span_m)


@dataclass
class VLMResult:
    """One VLM solution: coefficients about the reference point plus the panel data."""

    alpha: float
    elevator: float
    coeffs: dict[str, float]
    centers: np.ndarray
    forces: np.ndarray  # geometry axes, N
    areas: np.ndarray
    flight: Flight

    def __getitem__(self, k: str) -> float:
        return self.coeffs[k]


def _operating_point(flight: Flight, alpha, beta, p, q, r, center):
    """An AeroSandbox operating point whose body rates turn about ``center`` (the CG).

    AeroSandbox's VLM applies p, q, r about the geometry origin (here the pod
    nose), not about the moment reference point. With the CG ~0.26 m aft of
    the nose, a pitch rate then also raised the angle of attack everywhere
    (Cm_q and CL_q ~60 % and ~3x too large) and a yaw rate added sideslip
    (Cn_r, CY_r, Cl_r wrong). Found by the M1.7 VSPAERO check, which rotates
    about Xcg. The rotation velocity is evaluated relative to ``center``.
    """
    import aerosandbox as asb

    op = asb.OperatingPoint(atmosphere=flight.atmosphere(), velocity=flight.velocity, alpha=float(alpha),
                            beta=float(beta), p=float(p), q=float(q), r=float(r))
    c = np.asarray(center, dtype=float)
    if np.any(c) and (p or q or r):
        base = op.compute_rotation_velocity_geometry_axes
        op.compute_rotation_velocity_geometry_axes = lambda points: base(np.asarray(points) - c)
    return op


def run_vlm(geom: BWBGeometry, flight: Flight, alpha: float, elevator_deg: float = 0.0, aileron_deg: float = 0.0,
            beta: float = 0.0, p: float = 0.0, q: float = 0.0, r: float = 0.0, xyz_ref=(0.0, 0.0, 0.0),
            spanwise_resolution: int | None = None,
            chordwise_resolution: int | None = None) -> VLMResult:
    """Run the vortex lattice. Rates p, q, r in rad/s, body axes (x fwd, y right, z down), about ``xyz_ref``.

    Panel counts default to the module settings, which :func:`vlm_resolution`
    can change for a block of code (the optimizer's coarse search tier).
    """
    import aerosandbox as asb

    spanwise_resolution = spanwise_resolution or SPANWISE_RESOLUTION
    chordwise_resolution = chordwise_resolution or CHORDWISE_RESOLUTION

    ap = build_airplane(geom, xyz_ref, elevator_deg, aileron_deg)
    op = _operating_point(flight, alpha, beta, p, q, r, xyz_ref)
    vlm = asb.VortexLatticeMethod(ap, op, spanwise_resolution=spanwise_resolution,
                                  chordwise_resolution=chordwise_resolution,
                                  spanwise_spacing_function=np.linspace)
    res = vlm.run()
    coeffs = {k: float(np.ravel(res[k])[0]) for k in ("CL", "CD", "CY", "Cl", "Cm", "Cn")}
    return VLMResult(alpha=float(alpha), elevator=float(elevator_deg), coeffs=coeffs,
                     centers=np.asarray(vlm.vortex_centers), forces=np.asarray(vlm.forces_geometry),
                     areas=np.asarray(vlm.areas), flight=flight)


@contextmanager
def vlm_resolution(spanwise: int, chordwise: int):
    """Temporarily change the default VLM panel counts (``with vlm_resolution(2, 6): ...``)."""
    global SPANWISE_RESOLUTION, CHORDWISE_RESOLUTION
    old = SPANWISE_RESOLUTION, CHORDWISE_RESOLUTION
    SPANWISE_RESOLUTION, CHORDWISE_RESOLUTION = spanwise, chordwise
    try:
        yield
    finally:
        SPANWISE_RESOLUTION, CHORDWISE_RESOLUTION = old


def spanwise_cl(geom: BWBGeometry, res: VLMResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local section CL across the right half of the wing (fins excluded).

    Returns (y of each spanwise panel column, local CL, column width).
    """
    a = math.radians(res.alpha)
    lift_dir = np.array([-math.sin(a), 0.0, math.cos(a)])  # geometry axes, x aft, z up
    lift = res.forces @ lift_dir
    c = res.centers
    on_wing = (c[:, 1] > 0) & (c[:, 1] <= geom.semispan_m + 1e-9)
    yw, l_w, a_w = c[on_wing, 1], lift[on_wing], res.areas[on_wing]
    # Group panels into spanwise columns. Twist on a dihedralled panel shifts a
    # column's panel centres by under 1 mm; columns are at least 2.5 mm apart.
    order = np.argsort(yw)
    col = np.zeros(len(yw), int)
    col[order] = np.cumsum(np.r_[0, np.diff(yw[order]) > 0.4e-3])
    n = col.max() + 1
    lift_c = np.bincount(col, l_w, n)
    area_c = np.bincount(col, a_w, n)
    ys = np.bincount(col, a_w * yw, n) / area_c
    return ys, lift_c / (res.flight.q * area_c), area_c / chord_at(geom, ys)


# --------------------------------------------------------------------------
# Linear longitudinal model (the VLM is linear in alpha and elevator)
# --------------------------------------------------------------------------


@dataclass
class LongitudinalModel:
    """CL and Cm (about the pod nose, x = 0) as linear functions of alpha and elevator, in degrees.

    Built from three VLM runs. Valid below the stall; the VLM itself is linear,
    so this is exact for the VLM to within its small-angle approximations.
    """

    geom: BWBGeometry
    cl0: float
    cl_alpha: float  # per deg
    cl_de: float  # per deg of elevator
    cm0: float  # about x = 0
    cm_alpha: float
    cm_de: float

    @property
    def neutral_point_x_m(self) -> float:
        """Stick-fixed neutral point: the x where Cm does not change with alpha."""
        return -self.cm_alpha / self.cl_alpha * self.geom.mac_m

    @property
    def neutral_point_pct_mac(self) -> float:
        return (self.neutral_point_x_m - self.geom.mac_le_x_m) / self.geom.mac_m * 100

    @property
    def alpha_zero_lift_deg(self) -> float:
        return -self.cl0 / self.cl_alpha

    def cg_for_static_margin(self, sm: float) -> float:
        return self.neutral_point_x_m - sm * self.geom.mac_m

    def static_margin(self, x_cg: float) -> float:
        return (self.neutral_point_x_m - x_cg) / self.geom.mac_m

    def cm_about(self, x_cg: float, alpha, elevator=0.0):
        cl = self.cl(alpha, elevator)
        cm = self.cm0 + self.cm_alpha * np.asarray(alpha) + self.cm_de * np.asarray(elevator)
        return cm + cl * x_cg / self.geom.mac_m  # moment of the lift about the CG (x aft)

    def cl(self, alpha, elevator=0.0):
        return self.cl0 + self.cl_alpha * np.asarray(alpha) + self.cl_de * np.asarray(elevator)

    def cm0_about(self, x_cg: float) -> float:
        """Zero-lift pitching moment about the CG (positive = self-trimming, nose-up)."""
        a0 = self.alpha_zero_lift_deg
        return float(self.cm_about(x_cg, a0))

    def cm_de_about(self, x_cg: float) -> float:
        return self.cm_de + self.cl_de * x_cg / self.geom.mac_m

    def trim(self, cl_target: float, x_cg: float) -> tuple[float, float]:
        """(alpha, elevator) in degrees for CL = ``cl_target`` and Cm = 0 about ``x_cg``."""
        k = x_cg / self.geom.mac_m
        a = np.array([[self.cl_alpha, self.cl_de],
                      [self.cm_alpha + k * self.cl_alpha, self.cm_de + k * self.cl_de]])
        b = np.array([cl_target - self.cl0, -(self.cm0 + k * self.cl0)])
        alpha, de = np.linalg.solve(a, b)
        return float(alpha), float(de)

    def trim_at_alpha(self, alpha: float, x_cg: float) -> tuple[float, float]:
        """(elevator, CL) that trims at a given alpha."""
        k = x_cg / self.geom.mac_m
        de = -(self.cm0 + self.cm_alpha * alpha + k * (self.cl0 + self.cl_alpha * alpha)) / (self.cm_de + k * self.cl_de)
        return float(de), float(self.cl(alpha, de))


def longitudinal_model(geom: BWBGeometry, flight: Flight, **vlm_kw) -> LongitudinalModel:
    r0 = run_vlm(geom, flight, 0.0, **vlm_kw)
    r1 = run_vlm(geom, flight, 6.0, **vlm_kw)
    r2 = run_vlm(geom, flight, 0.0, elevator_deg=5.0, **vlm_kw)
    return LongitudinalModel(
        geom=geom, cl0=r0["CL"], cl_alpha=(r1["CL"] - r0["CL"]) / 6.0, cl_de=(r2["CL"] - r0["CL"]) / 5.0,
        cm0=r0["Cm"], cm_alpha=(r1["Cm"] - r0["Cm"]) / 6.0, cm_de=(r2["Cm"] - r0["Cm"]) / 5.0,
    )


# --------------------------------------------------------------------------
# Section data from NeuralFoil (as built)
# --------------------------------------------------------------------------


def _re_bin(re: float) -> float:
    """Round Reynolds number to ~2.5 % steps so polars can be cached."""
    return float(round(math.exp(round(math.log(re) / 0.025) * 0.025), -2))


@lru_cache(maxsize=2048)
def section_polar(pod: str, outer: str, blend: float, chord_m: float, re: float, mach: float,
                  n_crit: float = BUILT_N_CRIT) -> Polar:
    """NeuralFoil polar of the (blended) section, with the as-built blunt trailing edge."""
    af = _blended_airfoil(pod, outer, blend).set_TE_thickness(BUILT_TE_THICKNESS_M / chord_m)
    return run_polar(af, re, mach, n_crit)


def _strip_polar(geom: BWBGeometry, y: float, flight: Flight, n_crit: float) -> Polar:
    c = round(float(chord_at(geom, y)) / 0.005) * 0.005  # 5 mm chord bins keep the cache small
    return section_polar(geom.pod_airfoil, geom.outer_airfoil, round(blend_at(geom, y), 1), round(c, 3),
                         _re_bin(flight.reynolds(c)), round(flight.mach, 3), n_crit)


def strip_cl_max(geom: BWBGeometry, y: np.ndarray, flight: Flight, n_crit: float = BUILT_N_CRIT) -> np.ndarray:
    """Section CL_max (as built) of each strip at its own Reynolds number."""
    return np.array([_strip_polar(geom, float(yy), flight, n_crit).cl_max()[0] for yy in y])


def strip_cd(geom: BWBGeometry, y: np.ndarray, local_cl: np.ndarray, flight: Flight,
             n_crit: float = BUILT_N_CRIT) -> np.ndarray:
    """Section profile drag of each strip at its local CL (clamped to the attached range)."""
    out = []
    for yy, cl in zip(y, local_cl):
        p = _strip_polar(geom, float(yy), flight, n_crit)
        s = p.attached
        cls, cds = p.cl[s], p.cd[s]
        order = np.argsort(cls)
        out.append(float(np.interp(cl, cls[order], cds[order])))
    return np.array(out)


@lru_cache(maxsize=256)
def _fin_cd0(chord_m: float, re: float, mach: float, n_crit: float) -> float:
    af = _load_airfoil("naca0009").set_TE_thickness(BUILT_TE_THICKNESS_M / chord_m)
    r = af.get_aero_from_neuralfoil(alpha=0.0, Re=re, mach=mach, n_crit=n_crit, model_size="xlarge")
    return float(np.ravel(r["CD"])[0])


def fin_cd(geom: BWBGeometry, flight: Flight, n_crit: float = BUILT_N_CRIT) -> float:
    """Fin profile drag referred to the wing area (fins at zero lift in straight flight)."""
    r, t = geom.fin_stations()
    c = (r["chord"] + t["chord"]) / 2
    return _fin_cd0(round(c, 3), _re_bin(flight.reynolds(c)), round(flight.mach, 3), n_crit) * \
        geom.fin_area_m2 / geom.reference_area_m2


def misc_cd(geom: BWBGeometry) -> float:
    return sum(MISC_DRAG_AREA_M2.values()) / geom.reference_area_m2


def induced_drag_trefftz(geom: BWBGeometry, y: np.ndarray, local_cl: np.ndarray, width: np.ndarray) -> tuple[float, float]:
    """(CD_induced, span efficiency e) of the wing's own spanwise loading, planar Trefftz plane.

    The loading c*cl(y) is fitted with a symmetric Fourier sine series
    (odd terms up to 15) and e = 1 / (1 + sum n (A_n / A_1)^2). This ignores
    any help from the drooped fins, so it is the cautious number: the VLM's
    own near-field drag reads about 5 % low even on a plain rectangular wing
    (e = 1.00 where lifting-line theory gives about 0.95), and lower still
    here because it credits the fins as winglets.
    """
    s = geom.semispan_m
    theta = np.arccos(np.clip(y / s, -1, 1))
    load = chord_at(geom, y) * local_cl  # proportional to circulation
    ns = np.arange(1, 16, 2)
    basis = np.sin(np.outer(theta, ns))
    w = np.sqrt(width)
    coef, *_ = np.linalg.lstsq(basis * w[:, None], load * w, rcond=None)
    a = coef / coef[0]
    e = 1.0 / float(np.sum(ns * a**2))
    cl = 2 * float(np.sum(load * width)) / geom.reference_area_m2
    return cl**2 / (math.pi * geom.aspect_ratio * e), e


# --------------------------------------------------------------------------
# Trimmed drag polar
# --------------------------------------------------------------------------


@dataclass
class PolarPoint:
    velocity: float
    cl: float
    alpha: float
    elevator: float
    cd_induced: float  # planar Trefftz from the VLM loading (cautious; see induced_drag_trefftz)
    span_efficiency: float
    cd_induced_vlm: float  # the VLM's own near-field value, for reference
    cd_wing: float  # wing profile drag (strip theory, NeuralFoil)
    cd_fin: float
    cd_misc: float  # interference + itemised extras
    n_crit: float
    strips: dict = field(default_factory=dict, repr=False)  # y, chord, width, local cl, section cd (right half)

    @property
    def cd(self) -> float:
        return self.cd_induced + self.cd_wing + self.cd_fin + self.cd_misc

    @property
    def ld(self) -> float:
        return self.cl / self.cd

    def sink_rate(self) -> float:
        """Motor-off sink rate (m/s), small glide angle."""
        return self.velocity * self.cd / self.cl

    def power_required_w(self, mass_kg: float) -> float:
        """Aerodynamic power D V in level flight (before prop and motor losses)."""
        return mass_kg * G0 * self.velocity / self.ld


def trimmed_point(geom: BWBGeometry, model: LongitudinalModel, x_cg: float, flight: Flight, mass_kg: float,
                  n_crit: float = BUILT_N_CRIT, res: VLMResult | None = None) -> tuple[PolarPoint, VLMResult]:
    """Level, trimmed flight at ``flight``: VLM at the trim, then profile drag strip by strip."""
    cl = cl_level(mass_kg, geom, flight)
    alpha, de = model.trim(cl, x_cg)
    if res is None:
        res = run_vlm(geom, flight, alpha, elevator_deg=de, xyz_ref=(x_cg, 0, 0))
    y, loc, width = spanwise_cl(geom, res)
    chord = chord_at(geom, y)
    cd_sec = strip_cd(geom, y, loc, flight, n_crit)
    cd_wing = 2 * float(np.sum(cd_sec * chord * width)) / geom.reference_area_m2
    cd_fin = fin_cd(geom, flight, n_crit)
    cdi, e = induced_drag_trefftz(geom, y, loc, width)
    pt = PolarPoint(velocity=flight.velocity, cl=res["CL"], alpha=alpha, elevator=de, cd_induced=cdi,
                    span_efficiency=e, cd_induced_vlm=res["CD"], cd_wing=cd_wing, cd_fin=cd_fin,
                    cd_misc=INTERFERENCE_FACTOR * (cd_wing + cd_fin) + misc_cd(geom), n_crit=n_crit,
                    strips=dict(y=y, chord=chord, width=width, cl=loc, cd=cd_sec))
    return pt, res


def trimmed_polar(geom: BWBGeometry, model: LongitudinalModel, x_cg: float, base: Flight, mass_kg: float,
                  speeds, n_crit: float = BUILT_N_CRIT) -> list[PolarPoint]:
    return [trimmed_point(geom, model, x_cg, base.at(float(v)), mass_kg, n_crit)[0] for v in speeds]


# --------------------------------------------------------------------------
# Maximum lift and where the stall starts
# --------------------------------------------------------------------------


@dataclass
class StallResult:
    cl_max: float  # trimmed wing CL when the first strip reaches its section CL_max
    alpha: float
    elevator: float
    stall_speed_m_s: float  # at the field on the design day
    y: np.ndarray  # strip centres
    local_cl: np.ndarray  # at the stall
    local_cl_max: np.ndarray
    first_strip_y: float  # where the stall starts
    tip_margin: float  # 1 - (local CL / CL_max) on the outboard elevon region at the stall
    sweep: list[tuple[float, float, float]] = field(default_factory=list)  # (alpha, CL, worst ratio)

    @property
    def ratio(self) -> np.ndarray:
        return self.local_cl / self.local_cl_max


def _stall_ratio(geom, model, x_cg, flight, alpha, n_crit):
    de, _ = model.trim_at_alpha(alpha, x_cg)
    res = run_vlm(geom, flight, alpha, elevator_deg=de, xyz_ref=(x_cg, 0, 0))
    y, loc, _ = spanwise_cl(geom, res)
    clmax = strip_cl_max(geom, y, flight, n_crit)
    return de, res, y, loc, clmax


def stall(geom: BWBGeometry, model: LongitudinalModel, x_cg: float, req: MissionRequirements, mass_kg: float,
          n_crit: float = BUILT_N_CRIT, step_deg: float = 1.0) -> StallResult:
    """Critical-section stall estimate, trimmed, at the field on the design day.

    Angle of attack rises (elevon re-trimmed at each step) until the first
    spanwise strip reaches its own as-built section CL_max; the crossing is
    then found by interpolation and re-run. Section CL_max depends on Reynolds
    number and so on the stall speed, which depends on the answer; two passes
    converge it.
    """
    vs = req.limits.max_stall_speed_m_s
    result = None
    for _ in range(2):
        flight = field_flight(req, vs)
        a = model.trim(0.6, x_cg)[0]  # start well below the stall
        sweep, prev = [], None
        while a < 25.0:
            de, res, y, loc, clmax = _stall_ratio(geom, model, x_cg, flight, a, n_crit)
            worst = float(np.nanmax(loc / clmax))
            sweep.append((a, res["CL"], worst))
            if worst >= 1.0:
                break
            prev = (a, worst)
            a += step_deg
        else:
            raise RuntimeError("no strip reached CL_max below 25 deg")
        if prev is not None:  # linear interpolation to the crossing, then re-run there
            pa, pw = prev
            a = pa + (1.0 - pw) / (worst - pw) * (a - pa)
            de, res, y, loc, clmax = _stall_ratio(geom, model, x_cg, flight, a, n_crit)
        cl0 = res["CL"]
        vs = math.sqrt(2 * mass_kg * G0 / (flight.rho * geom.reference_area_m2 * cl0))
        ratio = loc / clmax
        y_e0 = geom.root_y_m + geom.elevon_start * geom.outer_semispan_m
        y_e1 = geom.root_y_m + geom.elevon_end * geom.outer_semispan_m
        outboard = (y >= y_e0 + 0.5 * (y_e1 - y_e0)) & (y <= y_e1)  # outer half of the elevon
        result = StallResult(
            cl_max=float(cl0), alpha=float(a), elevator=float(de), stall_speed_m_s=vs, y=y, local_cl=loc,
            local_cl_max=clmax, first_strip_y=float(y[int(np.nanargmax(ratio))]),
            tip_margin=float(1 - np.nanmax(ratio[outboard])), sweep=sweep,
        )
    return result
