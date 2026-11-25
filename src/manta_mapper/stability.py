"""Tailless stability, trim, flight modes and camera tilt (milestone M1.4).

Run everything with one command (writes plots and a text report)::

    python -m manta_mapper.stability config/mission.toml --plots docs/img

What it answers, in order:

1. **Where is the neutral point, and where may the CG go?** Static margin
   5-10 % MAC (AC-7) turns into a CG range in millimetres from the pod nose.
2. **Does it trim with the elevons near neutral?** (AC-8, within 3 deg at
   cruise) and how much elevon is left over at the slow end.
3. **Does the inner wing stall before the tips?** Spanwise local CL against
   each strip's own CL_max at the moment the stall starts.
4. **How much do the elevons do?** Pitch and roll power, roll rate.
5. **Lateral-directional stability** with the small drooped tip fins.
6. **Flight modes** (short period, phugoid, Dutch roll, roll, spiral), and the
   predicted response to a pitch doublet for the flight-test correlation (VAL-6).
7. **Camera tilt** (CAM-4): tilted camera mount vs more outer-panel incidence.

Masses and inertias are placeholders until the weight build-up (M1.5); see
:func:`estimate_inertia`. Everything else comes from :mod:`manta_mapper.aero`.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np

from . import aero as A
from .geometry import BWBGeometry, baseline
from .requirements import G0, MissionRequirements

STATIC_MARGINS = (0.05, 0.075, 0.10)  # AC-7 band and its middle
DESIGN_SM = 0.075
MAX_ELEVON_DEG = 20.0  # mechanical throw assumed for the servo linkage, each way
CAMERA_TOLERANCE_DEG = 3.0  # CAM-4

# Handling-quality yardsticks (MIL-F-8785C, class I aircraft, cruise flight
# phase, level 1). A small UAV flown by an autopilot does not have to meet a
# piloted-aircraft spec; they are used as "is this sensible" marks.
SHORT_PERIOD_DAMPING = (0.30, 2.0)
DUTCH_ROLL_MIN_DAMPING = 0.08
SPIRAL_MIN_DOUBLE_TIME_S = 12.0  # level 2 (level 1 is 20 s); the autopilot holds the wings level
ROLL_MIN_PB2V = 0.05  # steady roll rate p b / 2V at full aileron (typical floor 0.05-0.07)


# --------------------------------------------------------------------------
# Mass properties (placeholder until M1.5)
# --------------------------------------------------------------------------


@dataclass
class Inertia:
    mass_kg: float
    x_cg: float
    ixx: float
    iyy: float
    izz: float
    note: str = ""

    @property
    def radii_of_gyration_m(self) -> tuple[float, float, float]:
        return tuple(math.sqrt(i / self.mass_kg) for i in (self.ixx, self.iyy, self.izz))


SPREAD_FRACTION = 0.45  # airframe, spar, servos, fins: spread over the planform
MOTOR_FRACTION = 0.07  # motor, prop, mount at the pod trailing edge
# The remaining 48 % (battery, avionics, camera) is a block in the payload bay.


def estimate_inertia(geom: BWBGeometry, mass_kg: float, x_cg: float, n: int = 400) -> Inertia:
    """Moments of inertia from a simple mass model, about the CG, body axes.

    * 45 % of the mass spread over the planform in proportion to area, at
      45 % of the local chord (shell, spar, panels, servos, fins).
    * 7 % at the pod trailing edge (motor, prop).
    * 48 % as a uniform block in the payload bay (battery, avionics, camera),
      positioned so the overall CG lands at ``x_cg``.

    Rough on purpose: the short-period frequency goes as 1/sqrt(Iyy), so a
    30 % error in Iyy moves it about 15 %, inside the VAL-6 band of 25 %.
    M1.5 replaces this with the real weight build-up.
    """
    y = (np.arange(n) + 0.5) / n * geom.semispan_m
    st = geom.stations()
    sy = np.array([s["le"][1] for s in st])
    le = np.interp(y, sy, [s["le"][0] for s in st])
    c = np.interp(y, sy, [s["chord"] for s in st])
    m_spread = SPREAD_FRACTION * mass_kg
    w = c / c.sum()
    xs = le + 0.45 * c
    m_motor = MOTOR_FRACTION * mass_kg
    x_motor = geom.pod_chord_m + 0.02
    m_block = mass_kg - m_spread - m_motor
    # block CG so that the total CG is x_cg
    x_block = (mass_kg * x_cg - m_spread * float(np.sum(w * xs)) - m_motor * x_motor) / m_block
    bay = geom.pod_bay()
    L, W = bay["length_m"], bay["width_m"] * 0.8
    # half span each side: spread mass is mirrored, so sums over y use both halves
    ixx = m_spread * float(np.sum(w * y**2)) + m_block * W**2 / 12
    iyy = (m_spread * float(np.sum(w * ((xs - x_cg) ** 2 + (0.1 * c) ** 2)))
           + m_motor * (x_motor - x_cg) ** 2 + m_block * ((x_block - x_cg) ** 2 + L**2 / 12))
    izz = ixx + iyy  # a flat aircraft: perpendicular-axis theorem
    return Inertia(mass_kg, x_cg, ixx, iyy, izz,
                   note="placeholder mass model (45 % spread over planform, 7 % motor, 48 % payload block)")


# --------------------------------------------------------------------------
# Stability derivatives (VLM, about the CG, stability axes)
# --------------------------------------------------------------------------


@dataclass
class Derivatives:
    """Non-dimensional derivatives per radian (rates as p b/2V, q c/2V, r b/2V)."""

    alpha_deg: float
    cl: float
    cd: float
    cl_a: float
    cd_a: float
    cm_a: float
    cl_q: float
    cm_q: float
    cy_b: float
    cl_b: float  # rolling moment (lower-case l)
    cn_b: float
    cy_p: float
    cl_p: float
    cn_p: float
    cy_r: float
    cl_r: float
    cn_r: float
    cl_da: float  # per rad of aileron (right TE up)
    cn_da: float
    cm_de: float
    cl_de_lift: float


def _stab_moments(res: A.VLMResult, alpha_deg: float) -> tuple[float, float]:
    """Body-axis rolling / yawing moment coefficients rotated into stability axes."""
    a = math.radians(alpha_deg)
    cl, cn = res["Cl"], res["Cn"]
    return cl * math.cos(a) + cn * math.sin(a), cn * math.cos(a) - cl * math.sin(a)


def profile_yaw_damping(geom: BWBGeometry, point: A.PolarPoint) -> float:
    """Cn_r from section profile drag, which the inviscid VLM can't see.

    Yawing nose-right slows the right wing and speeds up the left; the drag
    difference opposes the yaw: Cn_r = -(4 / S b^2) * integral of cd c y^2 dy
    over the span (both halves), with the interference factor on top.
    """
    st = point.strips
    integral = 2 * float(np.sum(st["cd"] * st["chord"] * st["y"] ** 2 * st["width"]))
    return -4 * integral * (1 + A.INTERFERENCE_FACTOR) / (geom.reference_area_m2 * geom.span_m**2)


def derivatives(geom: BWBGeometry, model: A.LongitudinalModel, x_cg: float, flight: A.Flight, mass_kg: float,
                point: A.PolarPoint) -> Derivatives:
    """Finite differences of the VLM about the trimmed condition at ``flight``.

    ``point`` is the trimmed polar point at the same condition (total drag,
    and the strip drag for the profile part of the yaw damping).
    """
    cd_total = point.cd
    cl0 = A.cl_level(mass_kg, geom, flight)
    alpha, de = model.trim(cl0, x_cg)
    V, c, b = flight.velocity, geom.mac_m, geom.span_m
    ref = (x_cg, 0.0, 0.0)
    kw = dict(elevator_deg=de, xyz_ref=ref)
    a_r = math.radians(alpha)

    base = A.run_vlm(geom, flight, alpha, **kw)
    d_hat = 0.01
    # pitch rate (stability and body y axes coincide)
    rq = A.run_vlm(geom, flight, alpha, q=d_hat * 2 * V / c, **kw)
    # sideslip
    db = 2.0
    rb = A.run_vlm(geom, flight, alpha, beta=db, **kw)
    # roll and yaw rate about stability axes, converted to body rates
    w = d_hat * 2 * V / b
    rp = A.run_vlm(geom, flight, alpha, p=w * math.cos(a_r), r=w * math.sin(a_r), **kw)
    rr = A.run_vlm(geom, flight, alpha, p=-w * math.sin(a_r), r=w * math.cos(a_r), **kw)
    da = 5.0
    ra = A.run_vlm(geom, flight, alpha, aileron_deg=da, **kw)

    l0, n0 = _stab_moments(base, alpha)
    lb, nb = _stab_moments(rb, alpha)
    lp, np_ = _stab_moments(rp, alpha)
    lr, nr = _stab_moments(rr, alpha)
    la, na = _stab_moments(ra, alpha)
    dbr, dar = math.radians(db), math.radians(da)

    # drag slope from the trimmed polar around this point
    p_lo = A.trimmed_point(geom, model, x_cg, flight.at(V * 1.05), mass_kg)[0]
    p_hi = A.trimmed_point(geom, model, x_cg, flight.at(V / 1.05), mass_kg)[0]
    cd_a = (p_hi.cd - p_lo.cd) / (p_hi.cl - p_lo.cl) * model.cl_alpha * 180 / math.pi

    return Derivatives(
        alpha_deg=alpha, cl=cl0, cd=cd_total,
        cl_a=model.cl_alpha * 180 / math.pi, cd_a=cd_a,
        cm_a=(model.cm_alpha + model.cl_alpha * x_cg / c) * 180 / math.pi,
        cl_q=(rq["CL"] - base["CL"]) / d_hat, cm_q=(rq["Cm"] - base["Cm"]) / d_hat,
        cy_b=(rb["CY"] - base["CY"]) / dbr, cl_b=(lb - l0) / dbr, cn_b=(nb - n0) / dbr,
        cy_p=(rp["CY"] - base["CY"]) / d_hat, cl_p=(lp - l0) / d_hat, cn_p=(np_ - n0) / d_hat,
        cy_r=(rr["CY"] - base["CY"]) / d_hat, cl_r=(lr - l0) / d_hat,
        cn_r=(nr - n0) / d_hat + profile_yaw_damping(geom, point),
        cl_da=(la - l0) / dar, cn_da=(na - n0) / dar,
        cm_de=model.cm_de_about(x_cg) * 180 / math.pi, cl_de_lift=model.cl_de * 180 / math.pi,
    )


# --------------------------------------------------------------------------
# Linear flight modes
# --------------------------------------------------------------------------


@dataclass
class Mode:
    name: str
    eigenvalue: complex

    @property
    def omega_n(self) -> float:
        return abs(self.eigenvalue)

    @property
    def damping(self) -> float:
        return -self.eigenvalue.real / abs(self.eigenvalue) if abs(self.eigenvalue) > 0 else 0.0

    @property
    def frequency_hz(self) -> float:
        return abs(self.eigenvalue.imag) / (2 * math.pi)

    @property
    def period_s(self) -> float:
        return 2 * math.pi / abs(self.eigenvalue.imag) if self.eigenvalue.imag else math.inf

    @property
    def time_constant_s(self) -> float:
        """For a real mode: time to halve (stable, > 0) or double (unstable, reported < 0)."""
        s = self.eigenvalue.real
        return -math.log(2) / s if s != 0 else math.inf


@dataclass
class Modes:
    short_period: Mode
    phugoid: Mode
    dutch_roll: Mode
    roll: Mode
    spiral: Mode
    a_long: np.ndarray
    b_long: np.ndarray  # input: elevator (rad)
    a_lat: np.ndarray


def flight_modes(d: Derivatives, inertia: Inertia, geom: BWBGeometry, flight: A.Flight) -> Modes:
    """Small-perturbation equations about level, trimmed flight, stability axes.

    Longitudinal states (u, w, q, theta); lateral (v, p, r, phi). Textbook form
    (Etkin and Reid; Nelson). Cm_alpha-dot is taken as zero: there is no tail
    for downwash lag to act on.
    """
    m, V = inertia.mass_kg, flight.velocity
    S, c, b = geom.reference_area_m2, geom.mac_m, geom.span_m
    qS = flight.q * S
    a = math.radians(d.alpha_deg)

    # inertia tensor rotated from body to stability axes
    J = np.diag([inertia.ixx, inertia.iyy, inertia.izz])
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Js = R @ J @ R.T
    ixx, iyy, izz, ixz = Js[0, 0], Js[1, 1], Js[2, 2], -Js[0, 2]

    xu = -2 * d.cd * qS / (m * V)
    xw = -(d.cd_a - d.cl) * qS / (m * V)
    zu = -2 * d.cl * qS / (m * V)
    zw = -(d.cl_a + d.cd) * qS / (m * V)
    zq = -d.cl_q * qS * c / (2 * m * V)
    mw = d.cm_a * qS * c / (iyy * V)
    mq = d.cm_q * qS * c * c / (2 * iyy * V)
    a_long = np.array([
        [xu, xw, 0.0, -G0],
        [zu, zw, V + zq, 0.0],
        [0.0, mw, mq, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    b_long = np.array([0.0, -d.cl_de_lift * qS / m, d.cm_de * qS * c / iyy, 0.0])

    yv = d.cy_b * qS / (m * V)
    yp = d.cy_p * qS * b / (2 * m * V)
    yr = d.cy_r * qS * b / (2 * m * V)
    L = {k: v * qS * b for k, v in dict(v=d.cl_b / V, p=d.cl_p * b / (2 * V), r=d.cl_r * b / (2 * V)).items()}
    N = {k: v * qS * b for k, v in dict(v=d.cn_b / V, p=d.cn_p * b / (2 * V), r=d.cn_r * b / (2 * V)).items()}
    g_ = 1 - ixz**2 / (ixx * izz)
    Lp = {k: (L[k] / ixx + ixz / ixx * N[k] / izz) / g_ for k in L}
    Np = {k: (N[k] / izz + ixz / izz * L[k] / ixx) / g_ for k in N}
    a_lat = np.array([
        [yv, yp, yr - V, G0],
        [Lp["v"], Lp["p"], Lp["r"], 0.0],
        [Np["v"], Np["p"], Np["r"], 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])

    ev_lon = np.linalg.eigvals(a_long)
    cplx = sorted([e for e in ev_lon if e.imag > 0], key=abs)
    if len(cplx) == 2:
        ph, sp = cplx
    else:  # short period split into two real roots (heavily damped)
        reals = sorted(ev_lon, key=abs)
        ph = cplx[0] if cplx else reals[0]
        sp = reals[-1]
    ev_lat = np.linalg.eigvals(a_lat)
    dr = [e for e in ev_lat if e.imag > 1e-9]
    real = sorted([e.real for e in ev_lat if abs(e.imag) <= 1e-9], key=abs)
    dutch = dr[0] if dr else complex(real[1] if len(real) > 2 else 0)
    return Modes(
        short_period=Mode("short period", complex(sp)), phugoid=Mode("phugoid", complex(ph)),
        dutch_roll=Mode("Dutch roll", complex(dutch)),
        roll=Mode("roll subsidence", complex(real[-1])), spiral=Mode("spiral", complex(real[0])),
        a_long=a_long, b_long=b_long, a_lat=a_lat,
    )


def doublet_response(modes: Modes, amplitude_deg: float = 3.0, half_period_s: float | None = None,
                     t_end: float = 4.0, dt: float = 0.002) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pitch-rate response (deg/s) to an elevator doublet, from the linear model.

    The default half-period is about half the short-period period, the
    usual choice for exciting the mode. Returns (t, elevator deg, q deg/s).
    """
    if half_period_s is None:
        half_period_s = max(0.1, 0.5 * 2 * math.pi / modes.short_period.omega_n)
    t = np.arange(0, t_end, dt)
    de = np.where((t >= 0.2) & (t < 0.2 + half_period_s), -amplitude_deg,
                  np.where((t >= 0.2 + half_period_s) & (t < 0.2 + 2 * half_period_s), amplitude_deg, 0.0))
    x = np.zeros(4)
    q = np.zeros_like(t)
    A_, B_ = modes.a_long, modes.b_long
    for i in range(len(t)):  # RK4
        u = math.radians(de[i])
        f = lambda s: A_ @ s + B_ * u  # noqa: E731
        k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)  # noqa: E702
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q[i] = math.degrees(x[2])
    return t, de, q


# --------------------------------------------------------------------------
# Camera tilt (CAM-4)
# --------------------------------------------------------------------------


@dataclass
class CameraOption:
    name: str
    geom: BWBGeometry
    model: A.LongitudinalModel
    pod_pitch_cruise_deg: float  # pod (camera reference) nose-up at cruise
    pod_pitch_window: tuple[float, float]  # over the survey airspeed window
    elevator_cruise_deg: float
    cm0_cg: float
    cd_cruise: float
    stall: A.StallResult | None = None


def camera_options(geom: BWBGeometry, req: MissionRequirements, mass_kg: float, model: A.LongitudinalModel,
                   window: tuple[float, float], run_stall: bool = True) -> list[CameraOption]:
    """Option A: keep the geometry, tilt the camera. Option B: add outer-panel incidence so the pod flies level."""
    out = []
    fl = A.cruise_flight(req)
    for name, g in (("A: tilted camera mount", geom), ("B: more outer-panel incidence", None)):
        if g is None:
            # incidence that puts the pod at ~0 deg at cruise: iterate (incidence lowers trim alpha almost 1:1
            # once the pod's share of lift is accounted for)
            inc = out[0].pod_pitch_cruise_deg
            for _ in range(3):
                g = geom.with_changes(root_twist_deg=geom.root_twist_deg + inc)
                m = A.longitudinal_model(g, fl)
                a = m.trim(A.cl_level(mass_kg, g, fl), m.cg_for_static_margin(DESIGN_SM))[0]
                inc += a * 1.15
            g = geom.with_changes(root_twist_deg=geom.root_twist_deg + round(inc * 2) / 2)
        m = model if g is geom else A.longitudinal_model(g, fl)
        x = m.cg_for_static_margin(DESIGN_SM)
        a_c, de_c = m.trim(A.cl_level(mass_kg, g, fl), x)
        a_lo = m.trim(A.cl_level(mass_kg, g, fl.at(window[0])), x)[0]
        a_hi = m.trim(A.cl_level(mass_kg, g, fl.at(window[1])), x)[0]
        pt = A.trimmed_point(g, m, x, fl, mass_kg)[0]
        st = A.stall(g, m, x, req, mass_kg) if run_stall else None
        out.append(CameraOption(name, g, m, a_c + g.pod_twist_deg, (a_hi, a_lo), de_c, m.cm0_about(x), pt.cd, st))
    return out


# --------------------------------------------------------------------------
# Full analysis
# --------------------------------------------------------------------------


@dataclass
class Analysis:
    geom: BWBGeometry
    req: MissionRequirements
    mass_kg: float
    model: A.LongitudinalModel
    x_cg: float
    trims: list[dict]
    polar: list[A.PolarPoint]
    polar_clean: list[A.PolarPoint]
    cruise: A.PolarPoint
    cruise_res: A.VLMResult
    stall: A.StallResult
    stall_by_sm: dict[float, A.StallResult]
    deriv: Derivatives
    inertia: Inertia
    modes: Modes
    window: tuple[float, float]
    camera: list[CameraOption]
    extras: dict = field(default_factory=dict)


def survey_window(req: MissionRequirements, mass_kg: float, geom: BWBGeometry, v_stall: float) -> tuple[float, float]:
    from .survey import max_airspeed_for_trigger

    return 1.3 * v_stall, max_airspeed_for_trigger(req)


def analyse(geom: BWBGeometry, req: MissionRequirements, mass_kg: float = 1.5, camera: bool = True,
            progress=print) -> Analysis:
    fl = A.cruise_flight(req)
    progress("Vortex lattice: lift, moment and elevon slopes ...")
    model = A.longitudinal_model(geom, fl)
    x_cg = model.cg_for_static_margin(DESIGN_SM)

    progress("Stall (critical section, trimmed) at static margins 5 / 7.5 / 10 % ...")
    stall_by_sm = {sm: A.stall(geom, model, model.cg_for_static_margin(sm), req, mass_kg) for sm in STATIC_MARGINS}
    st = stall_by_sm[DESIGN_SM]
    window = survey_window(req, mass_kg, geom, st.stall_speed_m_s)

    progress("Trim table ...")
    trims = []
    for sm in STATIC_MARGINS:
        x = model.cg_for_static_margin(sm)
        vs = stall_by_sm[sm].stall_speed_m_s
        for label, flight in (("stall (trimmed CL_max)", A.field_flight(req, vs)),
                              ("hand launch, 1.2 Vs", A.field_flight(req, 1.2 * vs)),
                              ("slowest survey, 1.3 Vs", A.cruise_flight(req, window[0])),
                              ("cruise", fl),
                              ("fastest survey", A.cruise_flight(req, window[1]))):
            cl = A.cl_level(mass_kg, geom, flight)
            a, de = model.trim(cl, x)
            trims.append(dict(sm=sm, x_cg=x, label=label, v=flight.velocity, cl=cl, alpha=a, elevator=de))

    progress("Trimmed drag polar (NeuralFoil strip drag, as built and clean) ...")
    speeds = np.arange(math.ceil(window[0] * 2) / 2 - 2.0, 22.01, 1.0)
    polar = A.trimmed_polar(geom, model, x_cg, fl, mass_kg, speeds)
    polar_clean = A.trimmed_polar(geom, model, x_cg, fl, mass_kg, speeds, n_crit=A.NOMINAL_N_CRIT)
    cruise, cruise_res = A.trimmed_point(geom, model, x_cg, fl, mass_kg)

    progress("Stability derivatives and flight modes ...")
    d = derivatives(geom, model, x_cg, fl, mass_kg, cruise)
    inertia = estimate_inertia(geom, mass_kg, x_cg)
    modes = flight_modes(d, inertia, geom, fl)
    # sensitivity of the short period to the inertia guess
    sens = {}
    for f in (0.7, 1.3):
        i2 = Inertia(inertia.mass_kg, inertia.x_cg, inertia.ixx, inertia.iyy * f, inertia.izz)
        sens[f] = flight_modes(d, i2, geom, fl).short_period
    modes_sm = {}
    for sm in (0.05, 0.10):
        x2 = model.cg_for_static_margin(sm)
        d2 = derivatives(geom, model, x2, fl, mass_kg, A.trimmed_point(geom, model, x2, fl, mass_kg)[0])
        modes_sm[sm] = flight_modes(d2, estimate_inertia(geom, mass_kg, x2), geom, fl).short_period

    cams = []
    if camera:
        progress("Camera tilt options ...")
        cams = camera_options(geom, req, mass_kg, model, window)
    return Analysis(geom, req, mass_kg, model, x_cg, trims, polar, polar_clean, cruise, cruise_res, st,
                    stall_by_sm, d, inertia, modes, window, cams,
                    extras=dict(sp_inertia_sens=sens, sp_by_sm=modes_sm))


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _mm(x: float) -> str:
    return f"{x*1000:.0f} mm"


def report(an: Analysis) -> str:
    g, m, d, md = an.geom, an.model, an.deriv, an.modes
    lines: list[str] = []
    out = lines.append
    out(f"== {g.name}: aerodynamics and stability (M1.4), {an.mass_kg:g} kg ==")
    out(f"Cruise {an.cruise.velocity:g} m/s at {A.cruise_flight(an.req).altitude_m:.0f} m MSL, design day; "
        f"survey window {an.window[0]:.1f}-{an.window[1]:.1f} m/s")
    out("")
    out("-- Longitudinal static stability --")
    out(f"Lift slope {m.cl_alpha:.4f}/deg ({m.cl_alpha*180/math.pi:.2f}/rad), zero-lift alpha {m.alpha_zero_lift_deg:.2f} deg")
    out(f"Neutral point x = {_mm(m.neutral_point_x_m)} from the pod nose = {m.neutral_point_pct_mac:.1f} % MAC "
        f"(MAC {_mm(g.mac_m)}, its LE at x = {_mm(g.mac_le_x_m)})")
    for sm in STATIC_MARGINS:
        x = m.cg_for_static_margin(sm)
        out(f"  static margin {sm:5.1%}: CG at x = {_mm(x)} ({(x-g.mac_le_x_m)/g.mac_m*100:.1f} % MAC), "
            f"Cm0 about CG {m.cm0_about(x):+.4f}")
    out(f"CG range for AC-7 (5-10 %): x = {_mm(m.cg_for_static_margin(0.10))} to {_mm(m.cg_for_static_margin(0.05))}"
        f" ({(m.cg_for_static_margin(0.05)-m.cg_for_static_margin(0.10))*1000:.0f} mm wide)")
    out("")
    out("-- Trim (elevon: + = trailing edge down) --")
    out(f"{'SM':>5} {'condition':<24} {'V m/s':>6} {'CL':>5} {'alpha':>6} {'elevon':>7}")
    for t in an.trims:
        out(f"{t['sm']:5.1%} {t['label']:<24} {t['v']:6.1f} {t['cl']:5.2f} {t['alpha']:6.1f} {t['elevator']:+7.1f}")
    cr = [t for t in an.trims if t["label"] == "cruise"]
    ok = all(abs(t["elevator"]) <= 3.0 for t in cr)
    out(f"AC-8 (cruise trim within 3 deg across the 5-10 % band): {'PASS' if ok else 'FAIL'}")
    out(f"Elevon pitch power about the CG {d.cm_de*math.pi/180:+.4f}/deg; lift {m.cl_de:+.4f}/deg "
        f"(effectiveness factor {A.ELEVON_EFFECTIVENESS} on thin-airfoil theory)")
    out("")
    out("-- Drag (trimmed, as built: n_crit 5, 1 mm trailing edge) --")
    c = an.cruise
    out(f"At cruise: CL {c.cl:.3f}, CD {c.cd:.4f} = induced {c.cd_induced:.4f} (e = {c.span_efficiency:.2f}) + wing profile "
        f"{c.cd_wing:.4f} + fins {c.cd_fin:.4f} + extras {c.cd_misc:.4f}; L/D {c.ld:.1f}; "
        f"aero power {c.power_required_w(an.mass_kg):.1f} W")
    out(f"Zero-lift-ish drag (profile + fins + extras) {c.cd - c.cd_induced:.4f} vs constraint-analysis guess "
        f"{an.req.assumptions.cd0}")
    best = max(an.polar, key=lambda p: p.ld)
    minsink = min(an.polar, key=lambda p: p.sink_rate())
    out(f"Best L/D {best.ld:.1f} at {best.velocity:g} m/s; minimum sink {minsink.sink_rate():.2f} m/s at {minsink.velocity:g} m/s")
    out(f"{'V':>5} {'CL':>6} {'CD':>7} {'L/D':>5} {'sink':>5} {'P_aero W':>8}  (clean CD)")
    for p, pc in zip(an.polar, an.polar_clean):
        out(f"{p.velocity:5.1f} {p.cl:6.3f} {p.cd:7.4f} {p.ld:5.1f} {p.sink_rate():5.2f} {p.power_required_w(an.mass_kg):8.1f}  ({pc.cd:.4f})")
    out("")
    out("-- Stall: where does it start? (critical-section method, trimmed, field, design day) --")
    for sm, s in an.stall_by_sm.items():
        out(f"  SM {sm:5.1%}: wing CL_max {s.cl_max:.3f} -> stall {s.stall_speed_m_s:.2f} m/s at {an.mass_kg:g} kg; "
            f"alpha {s.alpha:.1f}, elevon {s.elevator:+.1f}; first strip at y = {_mm(s.first_strip_y)}; "
            f"outer elevon half at {1 - s.tip_margin:.0%} of its CL_max")
    s = an.stall
    i_pod = s.y <= g.pod_half_width_m
    out(f"Pod at {np.nanmax(s.ratio[i_pod]):.0%} of its CL_max at the stall; panel joint at y = {_mm(g.root_y_m)}")
    out(f"AC-1 (stall <= {an.req.limits.max_stall_speed_m_s:g} m/s at the design SM): "
        f"{'PASS' if s.stall_speed_m_s <= an.req.limits.max_stall_speed_m_s else 'FAIL'}")
    out("")
    out("-- Lateral-directional (cruise, stability axes, per rad) --")
    out(f"Cn_beta {d.cn_b:+.4f} (want > 0), Cl_beta {d.cl_b:+.4f} (want < 0), CY_beta {d.cy_b:+.4f}")
    inr = an.inertia
    cnb_dyn = d.cn_b * math.cos(math.radians(d.alpha_deg)) - inr.izz / inr.ixx * d.cl_b * math.sin(math.radians(d.alpha_deg))
    out(f"Cn_beta,dynamic {cnb_dyn:+.4f} (departure resistance, want > 0)")
    out(f"Cl_p {d.cl_p:+.3f}, Cn_p {d.cn_p:+.4f}, Cl_r {d.cl_r:+.4f}, Cn_r {d.cn_r:+.4f}")
    out(f"Aileron: Cl_da {d.cl_da*math.pi/180:+.5f}/deg, adverse yaw Cn_da {d.cn_da*math.pi/180:+.5f}/deg")
    pb2v = -d.cl_da * math.radians(MAX_ELEVON_DEG / 2) / d.cl_p
    v_launch = 1.2 * s.stall_speed_m_s
    out(f"Cn_r includes {profile_yaw_damping(g, an.cruise):+.4f} from section profile drag (not in the VLM)")
    out(f"Steady roll at {MAX_ELEVON_DEG/2:g} deg differential: pb/2V {pb2v:.3f} -> {math.degrees(pb2v*2*v_launch/g.span_m):.0f} deg/s "
        f"at launch speed, {math.degrees(pb2v*2*an.cruise.velocity/g.span_m):.0f} deg/s at cruise "
        f"({'OK' if pb2v >= ROLL_MIN_PB2V else 'WATCH'}: yardstick pb/2V >= {ROLL_MIN_PB2V} at half throw)")
    out("")
    out("-- Flight modes (cruise, design SM, placeholder inertia) --")
    kx, ky, kz = inr.radii_of_gyration_m
    out(f"Inertia: Ixx {inr.ixx:.4f}, Iyy {inr.iyy:.4f}, Izz {inr.izz:.4f} kg m^2 (k = {kx*1000:.0f} / {ky*1000:.0f} / "
        f"{kz*1000:.0f} mm); {inr.note}")
    out(f"Cm_alpha {d.cm_a:+.3f}/rad, Cm_q {d.cm_q:+.3f}, CL_q {d.cl_q:+.3f}")
    sp, ph = md.short_period, md.phugoid
    lo_, hi_ = SHORT_PERIOD_DAMPING
    out(f"Short period: {sp.frequency_hz:.2f} Hz damped ({sp.omega_n:.1f} rad/s natural), damping ratio {sp.damping:.2f} "
        f"({'OK' if lo_ <= sp.damping <= hi_ else 'WATCH'}: yardstick {lo_}-{hi_})")
    for f, mo in an.extras["sp_inertia_sens"].items():
        out(f"   with Iyy x{f}: {mo.frequency_hz:.2f} Hz, damping {mo.damping:.2f}")
    for sm, mo in an.extras["sp_by_sm"].items():
        out(f"   at SM {sm:.0%}: {mo.frequency_hz:.2f} Hz, damping {mo.damping:.2f}")
    out(f"Phugoid: period {ph.period_s:.1f} s, damping ratio {ph.damping:.3f}")
    dr, rl, spi = md.dutch_roll, md.roll, md.spiral
    out(f"Dutch roll: period {dr.period_s:.2f} s, damping ratio {dr.damping:.3f} "
        f"({'OK' if dr.damping >= DUTCH_ROLL_MIN_DAMPING else 'WATCH'}: yardstick >= {DUTCH_ROLL_MIN_DAMPING})")
    out(f"Roll subsidence: time constant {-1/rl.eigenvalue.real:.2f} s")
    tc = spi.time_constant_s
    sp_ok = tc > 0 or abs(tc) >= SPIRAL_MIN_DOUBLE_TIME_S
    out(f"Spiral: {'stable, halves' if tc > 0 else 'unstable, doubles'} in {abs(tc):.1f} s "
        f"({'OK' if sp_ok else 'WATCH'}: yardstick doubling >= {SPIRAL_MIN_DOUBLE_TIME_S:g} s)")
    out("")
    if an.camera:
        out("-- Camera tilt (CAM-4: within 3 deg of vertical) --")
        for o in an.camera:
            lo, hi = o.pod_pitch_window
            stall_txt = (f"; stall {o.stall.stall_speed_m_s:.2f} m/s, outer elevon half at {1 - o.stall.tip_margin:.0%}"
                         if o.stall else "")
            out(f"{o.name}: panel root incidence {o.geom.root_twist_deg:+.1f} deg; pod pitch at cruise "
                f"{o.pod_pitch_cruise_deg:+.1f} deg ({lo:+.1f} to {hi:+.1f} over the survey window); "
                f"cruise elevon {o.elevator_cruise_deg:+.1f}; Cm0 about CG {o.cm0_cg:+.4f}; CD {o.cd_cruise:.4f}{stall_txt}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac",
                                                      "#008a17", "#6a4ce2")
INK, INK2, GRID = "#191919", "#666666", "#e4e4e4"


def _style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def plot_polar(an: Analysis, path: str | None = None):  # pragma: no cover - visual
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    v = np.array([p.velocity for p in an.polar])
    ax = axs[0, 0]
    al = np.linspace(-4, 16, 50)
    ax.plot(al, an.model.cl(al), color=BLUE, lw=2, label="VLM (linear, no stall)")
    ax.plot([p.alpha for p in an.polar], [p.cl for p in an.polar], "o", color=BLUE, ms=6, mfc="white",
            label="trimmed level flight (up-elevon at low speed costs some lift)")
    ax.axhline(an.stall.cl_max, color=ORANGE, lw=2, ls="--", label=f"trimmed CL_max {an.stall.cl_max:.2f} (stall starts)")
    ax.set_xlabel("angle of attack (deg, pod chord line)")
    ax.set_ylabel("CL")
    ax.set_title("Lift curve", loc="left")
    ax.legend(fontsize=8, frameon=False)

    ax = axs[0, 1]
    cl = np.array([p.cl for p in an.polar])
    parts = [("induced", [p.cd_induced for p in an.polar], BLUE), ("wing profile", [p.cd_wing for p in an.polar], AQUA),
             ("fins", [p.cd_fin for p in an.polar], YELLOW), ("extras + interference", [p.cd_misc for p in an.polar], ORANGE)]
    base = np.zeros_like(cl)
    for name, vals, col in parts:
        ax.fill_betweenx(cl, base, base + np.array(vals), color=col, alpha=0.85, label=name, lw=0)
        base = base + np.array(vals)
    ax.plot([p.cd for p in an.polar_clean], [p.cl for p in an.polar_clean], color=INK2, lw=1.5, ls=":",
            label="total, clean model (n_crit 9)")
    ax.set_xlabel("CD")
    ax.set_ylabel("CL")
    ax.set_title("Trimmed drag polar, as built, with breakdown", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = axs[1, 0]
    ax.plot(v, [p.ld for p in an.polar], color=BLUE, lw=2, label="as built")
    ax.plot(v, [p.ld for p in an.polar_clean], color=BLUE, lw=1.5, ls=":", label="clean")
    ax.axvspan(*an.window, color=AQUA, alpha=0.12, lw=0, label="survey airspeed window")
    ax.axvline(an.cruise.velocity, color=INK2, lw=1, ls="--")
    ax.set_xlabel("airspeed (m/s)")
    ax.set_ylabel("L/D")
    ax.set_title("Lift-to-drag ratio", loc="left")
    ax.legend(fontsize=8, frameon=False)

    ax = axs[1, 1]
    ax.plot(v, [p.sink_rate() for p in an.polar], color=ORANGE, lw=2, label="as built")
    ax.plot(v, [p.sink_rate() for p in an.polar_clean], color=ORANGE, lw=1.5, ls=":", label="clean")
    s = np.array([p.sink_rate() for p in an.polar])
    ax.fill_between(v, s * 0.85, s * 1.15, color=ORANGE, alpha=0.15, lw=0, label="VAL-1 pass band (±15 %)")
    ax.invert_yaxis()
    ax.set_xlabel("airspeed (m/s)")
    ax.set_ylabel("sink rate (m/s, motor off)")
    ax.set_title("Glide prediction for the flight test (VAL-1)", loc="left")
    ax.legend(fontsize=8, frameon=False)
    for a in axs.flat:
        _style(a)
    fig.suptitle(f"{an.geom.name}: trimmed aerodynamics at {an.mass_kg:g} kg, survey altitude, design day", x=0.01,
                 ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_trim(an: Analysis, path: str | None = None):  # pragma: no cover - visual
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    m = an.model
    cl = np.linspace(-0.1, 1.0, 50)
    cols = {0.05: ORANGE, 0.075: BLUE, 0.10: AQUA}
    for sm in STATIC_MARGINS:
        x = m.cg_for_static_margin(sm)
        al = (cl - m.cl0) / m.cl_alpha
        ax.plot(cl, m.cm_about(x, al), color=cols[sm], lw=2, label=f"SM {sm:.1%} (CG {x*1000:.0f} mm), elevon neutral")
    ax.axhline(0, color=INK2, lw=1)
    ax.axvline(an.cruise.cl, color=INK2, lw=1, ls="--")
    ax.text(an.cruise.cl, ax.get_ylim()[1], " cruise CL", color=INK2, fontsize=8, va="top")
    ax.set_xlabel("CL")
    ax.set_ylabel("Cm about the CG (+ nose-up)")
    ax.set_title("Pitching moment: negative slope = stable; crossing zero = trim", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    v = np.linspace(an.window[0] - 2.5, 22, 60)
    fl = A.cruise_flight(an.req)
    for sm in STATIC_MARGINS:
        x = m.cg_for_static_margin(sm)
        de = [m.trim(A.cl_level(an.mass_kg, an.geom, fl.at(vv)), x)[1] for vv in v]
        ax2.plot(v, de, color=cols[sm], lw=2, label=f"SM {sm:.1%}")
    ax2.axhspan(-3, 3, color=AQUA, alpha=0.12, lw=0, label="AC-8: ±3° at cruise")
    ax2.axvspan(*an.window, color=GRID, alpha=0.5, lw=0, label="survey window")
    ax2.axvline(an.cruise.velocity, color=INK2, lw=1, ls="--")
    ax2.set_xlabel("airspeed (m/s), survey altitude")
    ax2.set_ylabel("trim elevon (deg, + = trailing edge down)")
    ax2.set_title("Elevon needed to trim", loc="left", fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    for a in (ax, ax2):
        _style(a)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_spanwise(an: Analysis, path: str | None = None):  # pragma: no cover - visual
    import matplotlib.pyplot as plt

    g, s = an.geom, an.stall
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(s.y, s.local_cl_max, color=INK, lw=1.5, ls="--", label="section CL_max, as built, at its own Reynolds number")
    ax.plot(s.y, s.local_cl, color=ORANGE, lw=2, label=f"local CL at the stall (wing CL {s.cl_max:.2f}, trimmed)")
    yc, clc, _ = A.spanwise_cl(g, an.cruise_res)
    ax.plot(yc, clc, color=BLUE, lw=2, label=f"local CL at cruise (wing CL {an.cruise.cl:.2f})")
    i = int(np.nanargmax(s.ratio))
    ax.annotate("stall starts here", (s.y[i], s.local_cl[i]), xytext=(s.y[i] + 0.08, s.local_cl[i] + 0.22),
                arrowprops=dict(arrowstyle="->", color=INK2), fontsize=9, color=INK)
    y_e0 = g.root_y_m + g.elevon_start * g.outer_semispan_m
    y_e1 = g.root_y_m + g.elevon_end * g.outer_semispan_m
    ax.axvspan(y_e0, y_e1, color=AQUA, alpha=0.10, lw=0, label="elevon")
    for yy, lab in ((g.pod_half_width_m, "pod edge"), (g.root_y_m, "panel joint")):
        ax.axvline(yy, color=INK2, lw=0.8, ls=":")
        ax.text(yy, 1.55, lab, rotation=90, fontsize=8, va="top", ha="right", color=INK2)
    ax.set_ylim(0, 1.6)
    ax.set_xlim(0, g.semispan_m * 1.01)
    ax.set_xlabel("spanwise station y (m), centreline to tip")
    ax.set_ylabel("section lift coefficient")
    ax.set_title(f"Stall pattern at SM {DESIGN_SM:.1%}: the inner panel reaches its limit first, the elevons keep "
                 f"{s.tip_margin:.0%} in hand", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    _style(ax)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_modes(an: Analysis, path: str | None = None):  # pragma: no cover - visual
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    md = an.modes
    t, de, q = doublet_response(md)
    ax.plot(t, de, color=INK2, lw=1.5, label="elevon doublet (deg)")
    ax.plot(t, q, color=BLUE, lw=2, label="pitch rate q (deg/s), linear model")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_title(f"Predicted doublet response (VAL-6): short period {md.short_period.frequency_hz:.1f} Hz, "
                 f"ζ = {md.short_period.damping:.2f}", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ms = [(md.short_period, BLUE), (md.phugoid, AQUA), (md.dutch_roll, ORANGE), (md.roll, VIOLET), (md.spiral, MAGENTA)]
    for mo, col in ms:
        e = mo.eigenvalue
        ax2.plot([e.real, e.real] if e.imag else [e.real], [e.imag, -e.imag] if e.imag else [0], "o", color=col, ms=9,
                 mec="white", mew=1.5, label=mo.name)
    ax2.axvline(0, color=INK2, lw=1)
    ax2.axhline(0, color=INK2, lw=0.8)
    ax2.set_xscale("symlog", linthresh=0.5)
    ax2.set_xlabel("real part (1/s): left of zero = stable (symlog)")
    ax2.set_ylabel("imaginary part (rad/s)")
    ax2.set_title("Flight-mode roots at cruise", loc="left", fontsize=10)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    for a in (ax, ax2):
        _style(a)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_camera(an: Analysis, path: str | None = None):  # pragma: no cover - visual
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.3))
    fl = A.cruise_flight(an.req)
    v = np.linspace(an.window[0], an.window[1], 40)
    mount = an.camera[0].pod_pitch_cruise_deg
    cols = [BLUE, ORANGE]
    for o, col in zip(an.camera, cols):
        x = o.model.cg_for_static_margin(DESIGN_SM)
        pitch = np.array([o.model.trim(A.cl_level(an.mass_kg, o.geom, fl.at(vv)), x)[0] for vv in v]) + o.geom.pod_twist_deg
        tilt = mount if o is an.camera[0] else 0.0
        ax.plot(v, pitch - tilt, color=col, lw=2, label=f"{o.name}" + (f" (camera tilted {tilt:.1f}° forward)" if tilt else ""))
    cold = A.Flight(fl.velocity, fl.altitude_m, an.req.environment.temperature_offset_K - 32.0)
    o = an.camera[0]
    x = o.model.cg_for_static_margin(DESIGN_SM)
    pitch_cold = np.array([o.model.trim(A.cl_level(an.mass_kg, o.geom, cold.at(vv)), x)[0] for vv in v]) - mount
    ax.plot(v, pitch_cold, color=BLUE, lw=1.5, ls=":", label="A on a 0 °C winter day")
    ax.axhspan(-CAMERA_TOLERANCE_DEG, CAMERA_TOLERANCE_DEG, color=AQUA, alpha=0.12, lw=0, label="CAM-4: within ±3°")
    ax.axvline(an.cruise.velocity, color=INK2, lw=1, ls="--")
    ax.set_xlabel("survey airspeed (m/s)")
    ax.set_ylabel("camera tilt from vertical (deg, + = looking aft)")
    ax.set_title("Camera pointing over the survey airspeed window", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(prog="manta_mapper.stability",
                                description="M1.4: aerodynamics, trim, stall pattern, stability and camera tilt.")
    p.add_argument("mission", nargs="?", help="mission TOML (defaults built in)")
    p.add_argument("--mass", type=float, default=1.5, help="take-off mass, kg")
    p.add_argument("--plots", help="directory to save plots into")
    p.add_argument("--report", help="also write the text report to this file")
    p.add_argument("--no-camera", action="store_true", help="skip the camera-tilt comparison (faster)")
    a = p.parse_args(argv)
    req = MissionRequirements.from_toml(a.mission) if a.mission else MissionRequirements()
    geom = baseline(req, a.mass)
    an = analyse(geom, req, a.mass, camera=not a.no_camera)
    text = report(an)
    print()
    print(text)
    if a.report:
        with open(a.report, "w") as f:
            f.write(text + "\n")
    if a.plots:
        import os

        import matplotlib

        matplotlib.use("Agg")
        os.makedirs(a.plots, exist_ok=True)
        jobs = [("aero_polar.png", plot_polar), ("aero_trim.png", plot_trim), ("aero_spanwise_stall.png", plot_spanwise),
                ("stability_modes.png", plot_modes)]
        if an.camera:
            jobs.append(("camera_tilt.png", plot_camera))
        for name, fn in jobs:
            fn(an, os.path.join(a.plots, name))
        print("Saved " + ", ".join(os.path.join(a.plots, n) for n, _ in jobs))


if __name__ == "__main__":
    main()
