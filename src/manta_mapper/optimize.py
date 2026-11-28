"""Design-to-cost optimizer (milestone M1.6).

Finds the cheapest, lightest airframe that meets every requirement, by moving
ten geometry variables and running the whole M1.3-M1.5 chain on each shape:

    geometry -> vortex lattice (lift, moment, elevon) -> trim -> stall
    (critical section, trimmed) -> trimmed drag polar -> propulsion checks
    -> weight build-up, CG and battery position -> inertia -> flight modes

**Objective.** Parts cost plus a price on weight: ``cost + 100 $/kg x
as-built mass``. 100 g is worth $10. That price is the "lightest" half of
"lowest-cost, lightest"; the ranking is checked at $0 and $300/kg as well.

**Constraints.** Every check the earlier milestones used, plus the margins
Ethan asked for (forgiving rather than tight). Each is written as
``g <= 0`` so the optimizer can see how far it is from the edge. See
:func:`constraints_from` for the list.

**Sizing mass.** The aerodynamic, stall and propulsion checks all run at the
1.5 kg design mass even though the build-up says ~1.2 kg (M1.5
recommendation). The weight saved by the optimizer shows up as margin, not as
a smaller wing.

**Search, in three steps** (each evaluation is the expensive part):

1. *Calibrate.* Run the baseline at full and at coarse VLM resolution. The
   difference in each aerodynamic output is kept as an additive correction
   (a standard multi-fidelity trick) so the coarse runs aim at the
   full-resolution answer. Coarse runs are ~15x faster.
2. *Screen.* A Latin-hypercube sample of the design space, run in parallel.
3. *Polish.* COBYLA (derivative-free, handles inequality constraints) from
   the best screened designs and from the baseline.

Then every finalist is re-run at full resolution with the full propulsion
catalogue. If a finalist fails at full resolution the correction is
recalibrated at that design and the polish repeated.

Run: ``python -m manta_mapper.optimize config/mission.toml`` (see ``--help``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, replace
from functools import partial

import numpy as np

from .geometry import BWBGeometry, baseline
from .requirements import G0, MissionRequirements

# --------------------------------------------------------------------------
# Design variables
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Variable:
    name: str
    lo: float
    hi: float
    unit: str
    note: str

    def scale(self, v: float) -> float:
        return (v - self.lo) / (self.hi - self.lo)

    def unscale(self, u: float) -> float:
        return self.lo + float(np.clip(u, 0.0, 1.0)) * (self.hi - self.lo)


#: The ten variables the optimizer moves. ``taper`` replaces the tip chord
#: so a tip chord bigger than the root can't be generated. The other M1.3
#: variables stay at their baseline values: blend width (sets the strake the
#: CAD blends), root twist (M1.4 chose a tilted camera mount instead), elevon
#: chord (25 %, generous for low-speed authority) and spar position (25 %).
VARIABLES: tuple[Variable, ...] = (
    Variable("span_m", 1.30, 2.40, "m", "tip to tip, wing only (no hard limit, Ethan 2026-11-27)"),
    Variable("root_chord_m", 0.22, 0.34, "m", "outer-panel root chord"),
    Variable("taper", 0.40, 0.90, "", "outer-panel tip chord / root chord"),
    Variable("sweep_le_deg", 12.0, 35.0, "deg", "outer-panel leading-edge sweep"),
    Variable("washout_deg", 2.0, 8.0, "deg", "tip nose-down twist"),
    Variable("dihedral_deg", 0.0, 6.0, "deg", "outer-panel dihedral"),
    Variable("fin_length_m", 0.05, 0.11, "m", "drooped tip fin length"),
    Variable("fin_droop_deg", 20.0, 60.0, "deg", "fin angle below horizontal"),
    Variable("pod_chord_m", 0.38, 0.50, "m", "centre pod chord"),
    Variable("pod_half_width_m", 0.085, 0.12, "m", "centreline to pod edge"),
)
NAMES = tuple(v.name for v in VARIABLES)


def design_point(geom: BWBGeometry) -> dict[str, float]:
    """The optimizer's variables for an existing geometry."""
    d = {n: float(getattr(geom, n)) for n in NAMES if n != "taper"}
    d["taper"] = geom.tip_chord_m / geom.root_chord_m
    return {n: d[n] for n in NAMES}


def make_geometry(x: dict[str, float], template: BWBGeometry | None = None) -> BWBGeometry:
    template = template or BWBGeometry()
    ch = {n: float(x[n]) for n in NAMES if n != "taper"}
    ch["tip_chord_m"] = float(x["root_chord_m"] * x["taper"])
    return replace(template, **ch)


def to_unit(x: dict[str, float]) -> np.ndarray:
    return np.array([v.scale(x[v.name]) for v in VARIABLES])


def from_unit(u) -> dict[str, float]:
    return {v.name: v.unscale(ui) for v, ui in zip(VARIABLES, u)}


# --------------------------------------------------------------------------
# Settings: margins and fidelity
# --------------------------------------------------------------------------

COARSE_VLM = (2, 6)  # spanwise, chordwise panels per section (full: 4, 10)


@dataclass(frozen=True)
class Settings:
    design_mass_kg: float = 1.5  # all aero/stall/propulsion checks run here (M1.5 recommendation)
    usd_per_kg: float = 100.0  # price on as-built mass in the objective
    # --- margins (forgiving, per Ethan's preference) ----------------------
    stall_margin_m_s: float = 0.0  # stall must come in this far under the AC-1 limit
    stall_sm: float = 0.10  # stall checked at the FORWARD CG limit (first flights), the worst case
    tip_stall_margin: float = 0.10  # outer half of the elevon at <= 90 % of its CL_max when the wing stalls
    max_stall_elevon_deg: float = 15.0  # of the 20 deg throw: keeps a quarter for control at the stall
    max_cruise_elevon_deg: float = 3.0  # AC-8, at both ends of the 5-10 % band
    camera_mount_deg: float = 6.0  # M1.4 wedge; pod pitch at cruise must stay near it
    camera_tolerance_deg: float = 2.0  # of the 3 deg CAM-4 budget; leaves 1 deg for gusts and build
    min_dutch_roll_damping: float = 0.08
    short_period_damping: tuple[float, float] = (0.30, 2.0)
    min_spiral_double_s: float = 12.0
    min_fin_clearance_m: float = 0.020  # skid touches first in a level landing (AC-13)
    min_tip_chord_m: float = 0.12  # EPP panel tip stiffness and a fin joint worth gluing (AC-13)
    payload_floor_margin: float = 0.0  # bay floor >= needed / 0.6 packing (already forgiving)
    max_built_fraction: float = 0.85  # as-built mass <= 85 % of the design mass: keeps >= 0.22 kg unspent
    fidelity: str = "coarse"  # "coarse" (search) or "full" (verification)
    #: the polish aims this far inside every limit (in g units: 0.2 = 0.1 m/s of stall, 0.004 of
    #: Dutch roll damping) so coarse-vs-full scatter doesn't push the verified design over the edge
    search_buffer: float = 0.2


#: Extra margin on the constraints the optimizer pushes against (see docs/optimizer.md):
#: 0.2 m/s on the stall (model error on CL_max is about that size), Dutch roll 0.09
#: instead of 0.08, and 25 mm of fin clearance instead of 20.
ROBUST = dict(stall_margin_m_s=0.2, min_dutch_roll_damping=0.09, min_fin_clearance_m=0.025)


# --------------------------------------------------------------------------
# One evaluation
# --------------------------------------------------------------------------

#: Metrics that come from the VLM and so get the coarse-to-full correction.
AERO_METRICS = (
    "np_pct_mac", "stall_speed_m_s", "tip_stall_margin", "stall_first_strip_rel", "stall_elevon_deg",
    "cruise_elevon_sm5_deg", "cruise_elevon_sm10_deg", "pod_pitch_cruise_deg", "cruise_ld",
    "dutch_roll_damping", "short_period_damping", "spiral_rate_per_s", "cn_beta",
)


@dataclass
class Evaluation:
    x: dict[str, float]
    geom: BWBGeometry | None
    metrics: dict[str, float] = field(default_factory=dict)
    raw: dict[str, float] = field(default_factory=dict)  # aero metrics before the correction
    constraints: dict[str, tuple[float, float, str, float]] = field(default_factory=dict)  # name -> (value, limit, sense, g)
    propulsion: str = ""
    error: str = ""
    seconds: float = 0.0
    fidelity: str = "coarse"
    source: str = ""

    @property
    def violation(self) -> float:
        """Sum of the normalised constraint violations (0 = feasible)."""
        if self.error:
            return 1e3
        return float(sum(max(0.0, c[3]) for c in self.constraints.values()))

    @property
    def feasible(self) -> bool:
        return not self.error and self.violation <= 1e-9

    def objective(self, usd_per_kg: float = 100.0) -> float:
        if self.error:
            return 1e4
        return self.metrics["cost_usd"] + usd_per_kg * self.metrics["mass_built_kg"]

    def worst(self) -> tuple[str, float]:
        if not self.constraints:
            return ("error", 1e3)
        k = max(self.constraints, key=lambda n: self.constraints[n][3])
        return k, self.constraints[k][3]


def _shortlist(req: MissionRequirements, mass_kg: float):
    """Motor/prop/pack sets that pass the checks that don't depend on the airframe shape.

    Static thrust, and motor and cell current at full static throttle, depend
    only on the mass and the field air. Sets that fail them can't be rescued
    by any geometry, so the search skips them (the final check runs the whole
    catalogue anyway).
    """
    from . import propulsion as P

    air = P.Air.at(req.environment.field_elevation_m, req.environment.temperature_offset_K)
    out = []
    for m in P.MOTORS:
        for p in P.PROPS:
            for pk in P.PACKS:
                op = P.full_throttle(P.PropulsionSet(m, p, pk, P.ESCS[-1]), 0.0, air, P.SOC_LAUNCH)
                if (op.thrust_n >= P.STATIC_THRUST_TO_WEIGHT * mass_kg * G0
                        and op.motor_current_a <= P.MOTOR_CURRENT_USE * m.max_current_a
                        and op.battery_current_a <= P.CELL_CURRENT_USE * pk.max_current_a):
                    out.append((m, p, pk))
    return tuple(out)


_SHORTLIST_CACHE: dict = {}


def _select(req, cond, polar, full_catalogue: bool):
    """Cheapest feasible set (same rule as M1.5), or the least-bad one if none passes."""
    from . import propulsion as P

    if full_catalogue:
        try:
            return P.select_propulsion(req, cond, polar).chosen, 0
        except RuntimeError:
            triples = [(m, p, pk) for m in P.MOTORS for p in P.PROPS for pk in P.PACKS]
    else:
        key = (req, round(cond.mass_kg, 4))
        if key not in _SHORTLIST_CACHE:
            _SHORTLIST_CACHE[key] = _shortlist(req, cond.mass_kg)
        triples = _SHORTLIST_CACHE[key]
    results = []
    for m, p, pk in triples:
        perf = None
        for esc in P.ESCS:
            perf = P.evaluate(P.PropulsionSet(m, p, pk, esc), cond, req, polar)
            if perf.checks["ESC current"][0]:
                break
        perf.checks.update(P.pack_energy_checks(perf.ps, cond, req, polar))
        results.append(perf)
    if not results:
        return None, 99
    feasible = [r for r in results if r.feasible]
    if feasible:
        best_p = min(r.cruise.electrical_power_w for r in feasible)
        short = [r for r in feasible if r.cruise.electrical_power_w <= (1 + P.CRUISE_POWER_TOLERANCE) * best_p]
        return min(short, key=lambda r: (r.ps.cost_usd, r.cruise.electrical_power_w)), 0
    best = min(results, key=lambda r: (sum(not ok for ok, _ in r.checks.values()), r.ps.cost_usd))
    return best, sum(not ok for ok, _ in best.checks.values())


def _metrics(geom: BWBGeometry, req: MissionRequirements, s: Settings) -> tuple[dict, str]:
    """Run the analysis chain on one shape. Returns (metrics, propulsion set name)."""
    from . import aero as A
    from . import propulsion as P
    from . import stability as S
    from . import weights as W

    m: dict[str, float] = {}
    mass = s.design_mass_kg
    full = s.fidelity == "full"

    # --- geometry-only checks -----------------------------------------------
    fit = geom.payload_fit()
    m["payload_floor_ratio"] = fit["floor_available_m2"] / fit["floor_needed_m2"]
    bay = geom.pod_bay()
    m["bay_height_margin_m"] = bay["max_internal_height_m"] - bay["needed_height_m"]
    m["spar_depth_tip_m"] = geom.spar_depth_m(1.0)
    m["spar_needed_m"] = geom.spar_diameter_m + 2 * geom.spar_min_cover_m
    m["fin_clearance_m"] = geom.fin_ground_clearance_m
    m["tip_chord_m"] = geom.tip_chord_m
    m["area_m2"] = geom.reference_area_m2
    m["span_m"] = geom.span_m
    m["aspect_ratio"] = geom.aspect_ratio
    m["wing_loading_kg_m2"] = mass / geom.reference_area_m2

    # --- longitudinal model, CG band, trim ----------------------------------
    fl = A.cruise_flight(req)
    model = A.longitudinal_model(geom, fl)
    x5, x75, x10 = (model.cg_for_static_margin(sm) for sm in (0.05, 0.075, 0.10))
    m["np_pct_mac"] = model.neutral_point_pct_mac
    m["x_np_m"] = model.neutral_point_x_m
    m["mac_m"] = geom.mac_m
    cl_c = A.cl_level(mass, geom, fl)
    m["cruise_elevon_sm5_deg"] = model.trim(cl_c, x5)[1]
    m["cruise_elevon_sm10_deg"] = model.trim(cl_c, x10)[1]
    a75, _ = model.trim(cl_c, x75)
    m["pod_pitch_cruise_deg"] = a75 + geom.pod_twist_deg

    # --- stall at the forward CG (worst case) -------------------------------
    st = A.stall(geom, model, model.cg_for_static_margin(s.stall_sm), req, mass, step_deg=1.0 if full else 2.0)
    m["stall_speed_m_s"] = st.stall_speed_m_s
    m["tip_stall_margin"] = st.tip_margin
    y_e0 = geom.root_y_m + geom.elevon_start * geom.outer_semispan_m
    m["stall_first_strip_rel"] = (st.first_strip_y - y_e0) / geom.semispan_m  # < 0: starts inboard of the elevon
    m["stall_elevon_deg"] = st.elevator

    # --- trimmed polar at the design CG -------------------------------------
    vs = st.stall_speed_m_s
    speeds = np.linspace(max(1.1 * vs, 9.0), 22.0, 13 if full else 5)
    pts = A.trimmed_polar(geom, model, x75, fl, mass, speeds)
    polar = P.QuadraticPolar.fit([p.cl for p in pts], [p.cd for p in pts])
    cruise = A.trimmed_point(geom, model, x75, fl, mass)[0]
    m["cruise_ld"] = cruise.ld
    m["cruise_cd"] = cruise.cd
    m["best_ld_model"] = polar.best_ld

    # --- propulsion at the design mass --------------------------------------
    avionics_w = W.build_up(geom, W.default_propulsion()).avionics_power_w
    cond = P.conditions(req, mass, geom.reference_area_m2, vs, avionics_w)
    perf, n_fail = _select(req, cond, polar, full_catalogue=full)
    m["propulsion_fails"] = float(n_fail)
    if perf is None:
        raise RuntimeError("no propulsion set to evaluate")
    ps = perf.ps
    m["static_tw"] = perf.static_thrust_to_weight
    m["cruise_power_w"] = perf.cruise.electrical_power_w

    # --- weights, CG, battery position ---------------------------------------
    bu = W.build_up(geom, ps, W.BuildChoices(cg_target_x_m=x10))  # first flights at the forward limit
    m["mass_built_kg"] = bu.mass_kg(True)
    m["mass_paper_kg"] = bu.mass_kg(False)
    m["cost_usd"] = bu.cost_usd
    m["ballast_kg"] = bu.ballast_kg
    lo, hi = bu.battery_range_m
    # where the pack has to go for each end of the CG band (unclamped), vs where it fits
    others = [i for i in bu.items if not i.movable and not i.name.startswith("Nose ballast")]
    bat = next(i for i in bu.items if i.movable)
    m_o = sum(i.mass_as_built_kg for i in others)
    mx_o = sum(i.mass_as_built_kg * i.cg[0] for i in others)
    m_b = bat.mass_as_built_kg
    x_bat10 = (x10 * (m_o + m_b) - mx_o) / m_b
    x_bat5 = (x5 * (m_o + m_b) - mx_o) / m_b
    m["battery_fwd_margin_m"] = x_bat10 - lo  # >= 0: no nose ballast at the forward CG
    m["battery_aft_margin_m"] = hi - x_bat5  # >= 0: the pack can reach the aft CG
    m["fit_problems"] = float(len(bu.fit_problems))
    m["structure_g"] = bu.by_group(True).get("structure", 0.0) * 1000

    # --- flight modes at the design CG, real inertia scaled to design mass --
    bu75 = W.build_up(geom, ps, W.BuildChoices(cg_target_x_m=x75))
    inertia = bu75.inertia(mass)
    d = S.derivatives(geom, model, x75, fl, mass, cruise)
    modes = S.flight_modes(d, inertia, geom, fl)
    m["dutch_roll_damping"] = modes.dutch_roll.damping
    m["short_period_damping"] = modes.short_period.damping
    m["short_period_hz"] = modes.short_period.frequency_hz
    tc = modes.spiral.time_constant_s
    # spiral divergence as 1 / doubling time (0 if it is stable): this, not the time itself, is what the
    # coarse-to-full correction tracks well (the time blows up as the root nears zero)
    m["spiral_rate_per_s"] = 1.0 / abs(tc) if tc < 0 else 0.0
    m["cn_beta"] = d.cn_b

    # --- missions (reported; the 25 ha and 50 ha rules are in the propulsion checks)
    m["window_low_m_s"] = 1.3 * vs
    return m, ps.name


def constraints_from(m: dict[str, float], req: MissionRequirements, s: Settings) -> dict:
    """``name -> (value, limit, sense, g)`` with g <= 0 feasible, normalised by a typical size."""
    lim = req.limits
    c = {}

    def le(name, value, limit, scale):
        c[name] = (value, limit, "<=", (value - limit) / scale)

    def ge(name, value, limit, scale):
        c[name] = (value, limit, ">=", (limit - value) / scale)

    def count(name, value):  # integer "number of failures": g = -0.5 when none, so a search buffer still passes
        c[name] = (value, 0.0, "<=", value - 0.5)

    from .survey import max_airspeed_for_trigger

    le("stall speed at forward CG, m/s (AC-1)", m["stall_speed_m_s"], lim.max_stall_speed_m_s - s.stall_margin_m_s, 0.5)
    ge("tip stall margin (centre stalls first)", m["tip_stall_margin"], s.tip_stall_margin, 0.05)
    le("stall starts inboard of elevon (rel. span)", m["stall_first_strip_rel"], 0.0, 0.05)
    le("elevon used at stall, deg", -m["stall_elevon_deg"], s.max_stall_elevon_deg, 3.0)
    le("cruise elevon at SM 5 %, |deg| (AC-8)", abs(m["cruise_elevon_sm5_deg"]), s.max_cruise_elevon_deg, 1.0)
    le("cruise elevon at SM 10 %, |deg| (AC-8)", abs(m["cruise_elevon_sm10_deg"]), s.max_cruise_elevon_deg, 1.0)
    le("pod pitch at cruise vs camera wedge, |deg| (CAM-4)",
       abs(m["pod_pitch_cruise_deg"] - s.camera_mount_deg), s.camera_tolerance_deg, 1.0)
    ge("Dutch roll damping", m["dutch_roll_damping"], s.min_dutch_roll_damping, 0.02)
    ge("short-period damping", m["short_period_damping"], s.short_period_damping[0], 0.1)
    le("short-period damping (upper)", m["short_period_damping"], s.short_period_damping[1], 0.1)
    le("spiral divergence, 1/doubling time, 1/s", m["spiral_rate_per_s"], 1.0 / s.min_spiral_double_s, 0.02)
    ge("Cn_beta (weathercock), /rad", m["cn_beta"], 0.0, 0.005)
    le("wing loading, kg/m^2 (AC-2)", m["wing_loading_kg_m2"], lim.max_wing_loading_kg_m2, 1.0)
    ge("airspeed window, m/s (AC-9)", max_airspeed_for_trigger(req) - m["window_low_m_s"],
       lim.min_airspeed_window_m_s, 1.0)
    count("propulsion checks failed (AC-5, AC-6, MIS-4)", m["propulsion_fails"])
    ge("payload floor / needed (AC-12)", m["payload_floor_ratio"], 1.0 + s.payload_floor_margin, 0.1)
    ge("payload bay height margin, m (AC-12)", m["bay_height_margin_m"], 0.0, 0.005)
    count("parts too tall for the pod", m["fit_problems"])
    ge("spar depth at tip, m", m["spar_depth_tip_m"], m["spar_needed_m"], 0.002)
    ge("fin ground clearance, m (AC-13)", m["fin_clearance_m"], s.min_fin_clearance_m, 0.005)
    ge("tip chord, m (AC-13)", m["tip_chord_m"], s.min_tip_chord_m, 0.01)
    ge("battery fwd room at SM 10 %, m (no ballast)", m["battery_fwd_margin_m"], 0.0, 0.01)
    ge("battery aft room at SM 5 %, m", m["battery_aft_margin_m"], 0.0, 0.01)
    le("as-built mass, kg (margin to design mass)", m["mass_built_kg"], s.max_built_fraction * s.design_mass_kg, 0.05)
    return c


def evaluate(x: dict[str, float], req: MissionRequirements, settings: Settings | None = None,
             correction: dict[str, float] | None = None, source: str = "") -> Evaluation:
    """Evaluate one design. Never raises: a broken design comes back with ``error`` set."""
    from . import aero as A

    s = settings or Settings()
    t0 = time.time()
    ev = Evaluation(x=dict(x), geom=None, fidelity=s.fidelity, source=source)
    try:
        geom = make_geometry(x)
        ev.geom = geom
        if s.fidelity == "full":
            m, ps = _metrics(geom, req, s)
        else:
            with A.vlm_resolution(*COARSE_VLM):
                m, ps = _metrics(geom, req, s)
        ev.raw = {k: m[k] for k in AERO_METRICS}
        if correction:
            for k, dv in correction.items():
                m[k] = m[k] + dv
        ev.metrics, ev.propulsion = m, ps
        ev.constraints = constraints_from(m, req, s)
    except Exception as e:  # noqa: BLE001 - any failure is an infeasible design to the optimizer
        ev.error = f"{type(e).__name__}: {e}"
    ev.seconds = time.time() - t0
    return ev


def calibrate(x: dict[str, float], req: MissionRequirements, s: Settings) -> tuple[dict, Evaluation, Evaluation]:
    """Full minus coarse for each aerodynamic metric at ``x``."""
    fine = evaluate(x, req, replace(s, fidelity="full"), source="calibration (full)")
    coarse = evaluate(x, req, replace(s, fidelity="coarse"), source="calibration (coarse)")
    if fine.error or coarse.error:
        raise RuntimeError(f"calibration failed: {fine.error or coarse.error}")
    return {k: fine.raw[k] - coarse.raw[k] for k in AERO_METRICS}, fine, coarse


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def latin_hypercube(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d))) / n
    return u


def _eval_unit(u, req, s, corr, source):
    return evaluate(from_unit(u), req, s, corr, source)


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _map(fn, items, workers: int):
    if workers <= 1:
        return [fn(i) for i in items]
    import multiprocessing as mp
    import os

    # one BLAS thread per worker: otherwise 4 workers x 4 threads thrash the cores
    old = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update({k: "1" for k in _THREAD_VARS})
    try:
        with mp.get_context("spawn").Pool(workers) as pool:
            return pool.map(fn, items, chunksize=1)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def rank_key(ev: Evaluation, usd_per_kg: float):
    """Feasible first, cheapest objective; then least violation."""
    return (0, ev.objective(usd_per_kg)) if ev.feasible else (1, ev.violation)


def polish(start: dict[str, float], req: MissionRequirements, s: Settings, corr: dict,
           maxiter: int = 60, rhobeg: float = 0.08, label: str = "polish") -> list[Evaluation]:
    """COBYLA from ``start`` in the unit cube. Returns every evaluation it made."""
    from scipy.optimize import minimize

    history: list[Evaluation] = []
    cache: dict[tuple, Evaluation] = {}
    names = None
    obj0 = None

    def ev_at(u):
        nonlocal names, obj0
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        k = tuple(np.round(u, 6))
        if k not in cache:
            e = evaluate(from_unit(u), req, s, corr, label)
            cache[k] = e
            history.append(e)
            if names is None and not e.error:
                names = list(e.constraints)
                obj0 = e.objective(s.usd_per_kg)
        return cache[k]

    first = ev_at(to_unit(start))
    if first.error:
        return history
    scale = max(abs(obj0), 1.0)

    def f(u):
        e = ev_at(u)
        return 1e3 if e.error else e.objective(s.usd_per_kg) / scale * 100

    buf = s.search_buffer
    cons = [{"type": "ineq",
             "fun": (lambda u, n=n: -(ev_at(u).constraints[n][3] + buf if not ev_at(u).error else 10.0))}
            for n in names]
    cons += [{"type": "ineq", "fun": (lambda u, i=i: u[i])} for i in range(len(VARIABLES))]
    cons += [{"type": "ineq", "fun": (lambda u, i=i: 1 - u[i])} for i in range(len(VARIABLES))]
    minimize(f, to_unit(start), method="COBYLA", constraints=cons,
             options=dict(maxiter=maxiter, rhobeg=rhobeg, tol=1e-3))
    return history


def _polish_job(args):
    start, req, s, corr, maxiter, label = args
    return polish(start, req, s, corr, maxiter=maxiter, label=label)


@dataclass
class Result:
    req: MissionRequirements
    settings: Settings
    baseline: Evaluation  # full fidelity
    correction: dict[str, float]
    archive: list[Evaluation]  # every coarse evaluation
    finalists: list[Evaluation]  # full fidelity, best first
    best: Evaluation
    selection: object = None  # full-catalogue propulsion selection at the best design
    timings: dict[str, float] = field(default_factory=dict)
    rounds: int = 1


def optimize(req: MissionRequirements, settings: Settings | None = None, n_screen: int = 120, n_polish: int = 3,
             maxiter: int = 60, workers: int = 4, seed: int = 16, progress=print) -> Result:
    s = settings or Settings()
    t = {}
    x0 = design_point(baseline(req, s.design_mass_kg))

    t0 = time.time()
    progress("Calibrating coarse vs full VLM on the M1.3 baseline ...")
    corr, base_full, _ = calibrate(x0, req, s)
    t["calibrate"] = time.time() - t0

    t0 = time.time()
    progress(f"Screening {n_screen} designs (Latin hypercube, {workers} workers) ...")
    U = latin_hypercube(n_screen, len(VARIABLES), seed)
    archive = _map(partial(_eval_unit, req=req, s=s, corr=corr, source="screen"), list(U), workers)
    t["screen"] = time.time() - t0
    n_ok = sum(e.feasible for e in archive)
    progress(f"  {n_ok} of {n_screen} feasible")

    finalists: list[Evaluation] = []
    rounds = 0
    while rounds < 3:
        rounds += 1
        t0 = time.time()
        ranked = sorted([e for e in archive if not e.error], key=lambda e: rank_key(e, s.usd_per_kg))
        starts, seen = [], []
        for e in ranked:  # distinct starts: at least 0.15 apart in the unit cube
            u = to_unit(e.x)
            if all(np.linalg.norm(u - v) > 0.15 for v in seen):
                starts.append(e.x)
                seen.append(u)
            if len(starts) >= n_polish - 1:
                break
        starts.append(x0 if rounds == 1 else finalists[0].x)
        progress(f"Polishing from {len(starts)} starts (COBYLA, up to {maxiter} evaluations each) ...")
        jobs = [(st, req, s, corr, maxiter, f"polish {rounds}.{i + 1}") for i, st in enumerate(starts)]
        for h in _map(_polish_job, jobs, min(workers, len(jobs))):
            archive.extend(h)
        t[f"polish {rounds}"] = time.time() - t0

        t0 = time.time()
        progress("Verifying the best designs at full resolution, full propulsion catalogue ...")
        ranked = sorted([e for e in archive if not e.error], key=lambda e: rank_key(e, s.usd_per_kg))
        picks, seen = [], []
        for e in ranked:  # top distinct designs
            u = to_unit(e.x)
            if all(np.linalg.norm(u - v) > 0.05 for v in seen):
                picks.append(e.x)
                seen.append(u)
            if len(picks) >= 3:
                break
        full = replace(s, fidelity="full")
        finalists = _map(partial(evaluate, req=req, settings=full, source="finalist"), picks, workers)
        finalists.sort(key=lambda e: rank_key(e, s.usd_per_kg))
        t[f"verify {rounds}"] = time.time() - t0
        if finalists and finalists[0].feasible:
            break
        if not finalists:
            break
        progress("  best finalist fails at full resolution: recalibrating there and polishing again")
        corr = {k: finalists[0].raw[k] - evaluate(finalists[0].x, req, s).raw[k] for k in AERO_METRICS}

    best = finalists[0] if finalists else base_full
    from . import propulsion as P
    from . import aero as A

    sel = None
    if best.geom is not None and not best.error:
        # the full selection report at the winner (same inputs as the finalist run)
        g = best.geom
        fl = A.cruise_flight(req)
        model = A.longitudinal_model(g, fl)
        x75 = model.cg_for_static_margin(0.075)
        pts = A.trimmed_polar(g, model, x75, fl, s.design_mass_kg,
                              np.linspace(max(1.1 * best.metrics["stall_speed_m_s"], 9.0), 22.0, 13))
        polar = P.QuadraticPolar.fit([p.cl for p in pts], [p.cd for p in pts])
        from . import weights as W

        cond = P.conditions(req, s.design_mass_kg, g.reference_area_m2, best.metrics["stall_speed_m_s"],
                            W.build_up(g, W.default_propulsion()).avionics_power_w)
        try:
            sel = P.select_propulsion(req, cond, polar)
        except RuntimeError:
            sel = None
    return Result(req, s, base_full, corr, archive, finalists, best, sel, t, rounds)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

_METRIC_ROWS = (
    ("span_m", "span", "m", 1, 3),
    ("area_m2", "wing area", "m^2", 1, 3),
    ("aspect_ratio", "aspect ratio", "", 1, 2),
    ("stall_speed_m_s", "stall at forward CG, design mass", "m/s", 1, 2),
    ("tip_stall_margin", "outer elevon margin at stall", "%", 100, 0),
    ("np_pct_mac", "neutral point", "% MAC", 1, 1),
    ("cruise_ld", "cruise L/D (model)", "", 1, 1),
    ("cruise_power_w", "cruise power at L/D 12, design mass", "W", 1, 1),
    ("dutch_roll_damping", "Dutch roll damping", "", 1, 3),
    ("short_period_hz", "short period", "Hz", 1, 2),
    ("spiral_rate_per_s", "spiral divergence (1/doubling time)", "1/s", 1, 3),
    ("pod_pitch_cruise_deg", "pod pitch at cruise", "deg", 1, 1),
    ("fin_clearance_m", "fin ground clearance", "mm", 1000, 0),
    ("structure_g", "structure, as built", "g", 1, 0),
    ("mass_built_kg", "mass as built", "kg", 1, 3),
    ("cost_usd", "parts cost", "$", 1, 0),
)


def _fmt(v, mult, nd):
    return f"{v * mult:.{nd}f}"


def report(r: Result) -> str:
    s = r.settings
    lines: list[str] = []
    out = lines.append
    b, w = r.baseline, r.best
    out("== Manta Mapper: design-to-cost optimizer (M1.6) ==")
    out(f"Objective: parts cost + ${s.usd_per_kg:g}/kg x as-built mass. Checks at {s.design_mass_kg:g} kg, "
        f"design day ({r.req.environment.field_elevation_m:g} m, ISA+{r.req.environment.temperature_offset_K:g}).")
    n_ok = sum(e.feasible for e in r.archive)
    out(f"Evaluations: {len(r.archive)} coarse ({n_ok} feasible), {len(r.finalists)} finalists at full resolution; "
        f"{r.rounds} round(s). Time: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in r.timings.items()))
    out("")
    out("-- Design variables --")
    out(f"{'variable':<18} {'baseline':>9} {'optimum':>9} {'bounds':>15}  note")
    for v in VARIABLES:
        out(f"{v.name:<18} {b.x[v.name]:9.3f} {w.x[v.name]:9.3f} {v.lo:7.3f}-{v.hi:<7.3f}  {v.note}")
    if w.geom is not None:
        out(f"{'tip_chord_m':<18} {b.geom.tip_chord_m:9.3f} {w.geom.tip_chord_m:9.3f}")
    out("")
    out("-- Results (full resolution) --")
    out(f"{'':<40} {'baseline':>9} {'optimum':>9}")
    for k, label, unit, mult, nd in _METRIC_ROWS:
        if k in b.metrics and k in w.metrics:
            out(f"{label + (', ' + unit if unit else ''):<40} {_fmt(b.metrics[k], mult, nd):>9} "
                f"{_fmt(w.metrics[k], mult, nd):>9}")
    out(f"{'objective, $':<40} {b.objective(s.usd_per_kg):9.1f} {w.objective(s.usd_per_kg):9.1f}")
    out(f"{'feasible':<40} {str(b.feasible):>9} {str(w.feasible):>9}")
    out(f"Propulsion: baseline {b.propulsion}; optimum {w.propulsion}")
    out("")
    out("-- Constraints at the optimum (g <= 0 passes; g = -1 is one 'typical size' of margin) --")
    out(f"{'constraint':<52} {'value':>9} {'limit':>9}  {'g':>6}  baseline g")
    for n, (val, lim, sense, g) in sorted(w.constraints.items(), key=lambda kv: -kv[1][3]):
        bg = b.constraints.get(n, (0, 0, "", float("nan")))[3]
        flag = "" if g <= 0 else "  FAIL"
        out(f"{n:<52} {val:9.4g} {sense}{lim:<8.4g} {g:6.2f}  {bg:6.2f}{flag}")
    out("")
    out("-- Finalists (full resolution) --")
    for i, e in enumerate(r.finalists):
        k, g = e.worst()
        out(f"{i + 1}. objective ${e.objective(s.usd_per_kg):.1f}, {e.metrics.get('mass_built_kg', float('nan')) * 1000:.0f} g, "
            f"${e.metrics.get('cost_usd', float('nan')):.0f}, span {e.x['span_m']:.3f} m, sweep {e.x['sweep_le_deg']:.1f} deg, "
            f"{'feasible' if e.feasible else 'INFEASIBLE'}; tightest: {k} (g {g:+.2f})")
    out("")
    out("-- Does the price on weight change the answer? (best feasible coarse design) --")
    feas = [e for e in r.archive if e.feasible]
    for p in (0.0, s.usd_per_kg, 300.0):
        if feas:
            e = min(feas, key=lambda e: e.objective(p))
            out(f"  ${p:>5.0f}/kg: {e.metrics['mass_built_kg'] * 1000:.0f} g, ${e.metrics['cost_usd']:.0f}, "
                f"span {e.x['span_m']:.3f} m, root chord {e.x['root_chord_m']:.3f} m, taper {e.x['taper']:.2f}, "
                f"pod {e.x['pod_chord_m']:.3f} x {2 * e.x['pod_half_width_m']:.3f} m")
    out("")
    out("-- Coarse-to-full correction (added to coarse results; calibrated on the baseline) --")
    out("  " + ", ".join(f"{k} {v:+.3g}" for k, v in r.correction.items()))
    if r.selection is not None:
        out("")
        out("-- Propulsion catalogue at the optimum (feasible sets) --")
        for perf in [c for c in r.selection.candidates if c.feasible][:5]:
            out(f"  {perf.ps.name} + {perf.ps.esc.name}: ${perf.ps.cost_usd:.0f}, cruise "
                f"{perf.cruise.electrical_power_w:.0f} W, static T/W {perf.static_thrust_to_weight:.2f}")
    return "\n".join(lines)


def to_json(r: Result) -> dict:
    def ev(e: Evaluation):
        return dict(x=e.x, tip_chord_m=e.geom.tip_chord_m if e.geom else None, feasible=e.feasible,
                    objective=e.objective(r.settings.usd_per_kg), propulsion=e.propulsion,
                    metrics={k: float(v) for k, v in e.metrics.items()},
                    constraints={k: dict(value=float(c[0]), limit=float(c[1]), sense=c[2], g=float(c[3]))
                                 for k, c in e.constraints.items()})

    return dict(milestone="M1.6", settings={k: v for k, v in r.settings.__dict__.items()},
                baseline=ev(r.baseline), best=ev(r.best), finalists=[ev(e) for e in r.finalists],
                correction=r.correction)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_screen(r: Result, path: str | None = None):  # pragma: no cover - visual
    import matplotlib.pyplot as plt

    s = r.settings
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ok = [e for e in r.archive if not e.error]
    inf = [e for e in ok if not e.feasible]
    feas = [e for e in ok if e.feasible]
    ax.scatter([e.metrics["mass_built_kg"] * 1000 for e in inf], [e.metrics["cost_usd"] for e in inf],
               s=12, color="#c9c9c9", label=f"fails a check ({len(inf)})")
    ax.scatter([e.metrics["mass_built_kg"] * 1000 for e in feas], [e.metrics["cost_usd"] for e in feas],
               s=16, color="#3a78b5", label=f"passes every check ({len(feas)})")
    b, w = r.baseline, r.best
    ax.scatter([b.metrics["mass_built_kg"] * 1000], [b.metrics["cost_usd"]], marker="s", s=70, color="#d1802e",
               label="M1.3 baseline", zorder=5)
    ax.scatter([w.metrics["mass_built_kg"] * 1000], [w.metrics["cost_usd"]], marker="*", s=180, color="#2a9d55",
               label="optimum", zorder=6)
    ax.set_xlabel("mass as built (g)")
    ax.set_ylabel("parts cost ($)")
    ax.set_title("Every design the optimizer tried (coarse VLM, corrected)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_planform(r: Result, path: str | None = None):  # pragma: no cover - visual
    import matplotlib.pyplot as plt

    from .geometry import _outline

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for e, col, lab in ((r.baseline, "#d1802e", "M1.3 baseline"), (r.best, "#2a9d55", "optimum")):
        y, le, te = _outline(e.geom)
        yy = np.r_[y, y[::-1]]
        xx = -np.r_[le, te[::-1]]
        ax.plot(yy, xx, color=col, lw=2, label=f"{lab}: span {e.geom.span_m:.2f} m, "
                f"{e.geom.reference_area_m2:.3f} m², {e.metrics['mass_built_kg'] * 1000:.0f} g")
        ax.plot(-yy, xx, color=col, lw=2)
    ax.set_aspect("equal")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m, nose up)")
    ax.set_title("Planform: baseline vs optimum")
    ax.legend(frameon=False, fontsize=9, loc="lower center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_margins(r: Result, path: str | None = None):  # pragma: no cover - visual
    import matplotlib.pyplot as plt

    names = list(r.best.constraints)
    gb = [r.baseline.constraints[n][3] for n in names]
    gw = [r.best.constraints[n][3] for n in names]
    order = np.argsort(gw)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    y = np.arange(len(names))
    clip = lambda v: float(np.clip(v, -4, 2))  # noqa: E731
    ax.barh(y + 0.2, [clip(gb[i]) for i in order], height=0.4, color="#d1802e", label="baseline")
    ax.barh(y - 0.2, [clip(gw[i]) for i in order], height=0.4, color="#2a9d55", label="optimum")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y, [names[i] for i in order], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("g (left of 0 passes; clipped at -4)")
    ax.set_title("Constraint margins")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(prog="manta_mapper.optimize", description="M1.6 design-to-cost optimizer.")
    p.add_argument("mission", nargs="?", help="mission TOML (defaults built in)")
    p.add_argument("--screen", type=int, default=120, help="Latin-hypercube designs to screen")
    p.add_argument("--polish", type=int, default=3, help="COBYLA starts")
    p.add_argument("--maxiter", type=int, default=60, help="evaluations per COBYLA start")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=16)
    p.add_argument("--usd-per-kg", type=float, default=100.0, help="price on as-built mass")
    p.add_argument("--margins", choices=("robust", "standard"), default="robust",
                   help="robust (default): extra margin on stall, Dutch roll and fin clearance; standard: requirement limits")
    p.add_argument("--quick", action="store_true", help="tiny run to check the plumbing (24 screened, 1 short polish)")
    p.add_argument("--report", help="save the text report here")
    p.add_argument("--json", help="save the optimum and finalists here")
    p.add_argument("--plots", help="directory for the plots")
    a = p.parse_args(argv)
    req = MissionRequirements.from_toml(a.mission) if a.mission else MissionRequirements()
    s = Settings(usd_per_kg=a.usd_per_kg, **(ROBUST if a.margins == "robust" else {}))
    if a.quick:
        a.screen, a.polish, a.maxiter = 24, 1, 12
    r = optimize(req, s, n_screen=a.screen, n_polish=a.polish, maxiter=a.maxiter, workers=a.workers, seed=a.seed,
                 progress=partial(print, flush=True))
    txt = report(r)
    print(txt)
    if a.report:
        with open(a.report, "w") as f:
            f.write(txt + "\n")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(to_json(r), f, indent=1, default=float)
    if a.plots:
        import os

        os.makedirs(a.plots, exist_ok=True)
        plot_screen(r, os.path.join(a.plots, "optimizer_screen.png"))
        plot_planform(r, os.path.join(a.plots, "optimizer_planform.png"))
        plot_margins(r, os.path.join(a.plots, "optimizer_margins.png"))


if __name__ == "__main__":  # pragma: no cover
    main()
