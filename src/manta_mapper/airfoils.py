"""Airfoil selection for the tailless BWB (milestone M1.2).

Why this matters for a flying wing
----------------------------------
A normal airplane uses its tail to balance the nose-down pitching moment of a
cambered wing. A tailless wing has no tail, so the airfoil itself (helped by
sweep and washout) must be *self-trimming*: its zero-lift pitching moment
``Cm0`` should be about zero or slightly nose-up. Airfoils that do this have
**reflex**: the rear of the camber line curls back up, like a small, fixed,
up-deflected elevon. Reflex costs some maximum lift, so picking a flying-wing
airfoil is a trade between pitch trim, low-speed lift, drag and buildability.

What this module does
---------------------
1. Works out the real flight conditions from the mission: air density and
   viscosity at the field on the design day, the stall speed, the airspeed
   window, and the chord at the centre body, mid-span and tip. From those it
   gets the Reynolds numbers each section actually flies at (70k-500k here,
   where airfoils behave very differently from the textbook 3-million data).
2. Runs NeuralFoil (a neural-network surrogate trained on XFOIL) for each
   candidate at those Reynolds numbers, on the clean airfoil and on an
   "as-built" version: a 1 mm blunt trailing edge (foam and prints cannot be
   knife-sharp) and early boundary-layer transition (tape seams, bugs, dust).
3. Measures what matters (``Cm0``, ``CL_max`` and how gently lift falls off
   past it, drag across the airspeed window, thickness for the spar) and how
   much each one degrades when built imperfectly.
4. Applies pass/fail gates, scores the survivors on fixed scales and
   recommends one airfoil for the outer panels and one for the centre body.

Fidelity: NeuralFoil is fast enough to sit inside the optimizer and agrees with
XFOIL to within its own scatter, but it is still 2-D, steady and attached-flow
biased. Post-stall numbers in particular are indicative only. The next rungs of
the ladder (VLM, VSPAERO, CFD; PIPE-3) check the whole wing.

Run it: ``python -m manta_mapper.airfoils config/mission.toml --plots docs/img``.
Needs the ``aero`` extra (AeroSandbox + NeuralFoil); the pure-numpy helpers
(``Polar`` metrics, scoring) work without it.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np

from .atmosphere import dynamic_viscosity, isa, speed_of_sound
from .constraints import size_aircraft
from .requirements import G0, MissionRequirements

# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    name: str  # display name
    key: str  # name in the AeroSandbox / UIUC coordinate database
    role: str  # "outer" (swept outer panels) or "center" (centre body)
    note: str  # why it is on the list


OUTER_CANDIDATES: tuple[Candidate, ...] = (
    Candidate("MH 45", "mh45", "outer", "Hepperle's classic swept-wing section, 9.8 %"),
    Candidate("MH 60", "mh60", "outer", "MH 45's slightly thicker, more cambered sibling, 10.1 %"),
    Candidate("E 186", "e186", "outer", "Eppler reflexed section, 10.3 %, stronger reflex"),
    Candidate("S5010", "s5010", "outer", "Selig flying-wing section, 9.8 %, low-Re design"),
    Candidate("EH 2.0/12", "eh2012", "outer", "Eppler-Hepperle 12 % reflexed, thicker for a spar"),
)

CENTER_CANDIDATES: tuple[Candidate, ...] = (
    Candidate("EH 2.0/12", "eh2012", "center", "same family as the outer panel option, 12 %"),
    Candidate("MH 80", "mh80", "center", "Hepperle flying-wing section, 12.7 %"),
    Candidate("MH 81", "mh81", "center", "MH 80 with more camber, 13.0 %"),
    Candidate("MH 78", "mh78", "center", "thick strongly reflexed section, 14.4 %"),
)

# --------------------------------------------------------------------------
# Flight conditions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Conditions:
    """Where on the airfoil's map the wing actually flies."""

    rho_field: float  # kg/m^3, stall and launch
    mu_field: float  # Pa s
    rho_cruise: float  # kg/m^3, survey altitude
    mu_cruise: float
    a_cruise: float  # speed of sound, m/s
    wing_loading_n_m2: float
    v_stall: float  # m/s, design stall speed (AC-1)
    v_min: float  # m/s, bottom of the airspeed window (1.3 Vs)
    v_cruise: float  # m/s
    v_max: float  # m/s, top of the window (camera trigger limit)
    chord_center_m: float
    chord_mean_m: float
    chord_tip_m: float

    def reynolds(self, v: float, chord: float, at_field: bool = False) -> float:
        rho, mu = (self.rho_field, self.mu_field) if at_field else (self.rho_cruise, self.mu_cruise)
        return rho * v * chord / mu

    def mach(self, v: float) -> float:
        return v / self.a_cruise

    def cl_required(self, v: float) -> float:
        """Lift coefficient for 1 g level flight at ``v`` (cruise altitude)."""
        return self.wing_loading_n_m2 / (0.5 * self.rho_cruise * v**2)

    def window_speeds(self, n: int = 6) -> np.ndarray:
        return np.linspace(self.v_min, self.v_max, n)

    def chord(self, role: str) -> float:
        """Chord whose Reynolds number represents the section's cruise behaviour."""
        return self.chord_center_m if role == "center" else self.chord_mean_m

    def smallest_chord(self, role: str) -> float:
        """Chord that sees the lowest Reynolds number (worst case for stall)."""
        return self.chord_center_m if role == "center" else self.chord_tip_m


def conditions_from_requirements(
    req: MissionRequirements,
    mass_kg: float = 1.5,
    center_chord_ratio: float = 1.7,
    tip_chord_ratio: float = 0.6,
) -> Conditions:
    """Flight conditions from the first-cut sizing.

    The planform does not exist yet (``geometry.py``, M1.3), so the centre-body
    and tip chords are placeholders: 1.7x and 0.6x the mean chord, typical for
    a small BWB with a deep centre pod and tapered outer panels.
    """
    sz = size_aircraft(req, mass_kg)
    env = req.environment
    t_field = isa(env.field_elevation_m, env.temperature_offset_K)[0]
    t_cruise = isa(env.field_elevation_m + sz.survey.altitude_m, env.temperature_offset_K)[0]
    rho_field = isa(env.field_elevation_m, env.temperature_offset_K)[2]
    rho_cruise = isa(env.field_elevation_m + sz.survey.altitude_m, env.temperature_offset_K)[2]
    v_min, v_max = sz.airspeed_window_m_s
    return Conditions(
        rho_field=rho_field,
        mu_field=dynamic_viscosity(t_field),
        rho_cruise=rho_cruise,
        mu_cruise=dynamic_viscosity(t_cruise),
        a_cruise=speed_of_sound(t_cruise),
        wing_loading_n_m2=mass_kg * G0 / sz.wing_area_m2,
        v_stall=sz.stall_speed_m_s,
        v_min=v_min,
        v_cruise=req.assumptions.cruise_speed_m_s,
        v_max=v_max,
        chord_center_m=center_chord_ratio * sz.mean_chord_m,
        chord_mean_m=sz.mean_chord_m,
        chord_tip_m=tip_chord_ratio * sz.mean_chord_m,
    )


# --------------------------------------------------------------------------
# Polar metrics (pure numpy, testable without NeuralFoil)
# --------------------------------------------------------------------------

ALPHAS_DEG = np.arange(-6.0, 20.01, 0.25)


@dataclass(frozen=True)
class Polar:
    """One alpha sweep at one Reynolds number."""

    alpha: np.ndarray  # deg
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray  # about the quarter chord, nose-up positive
    re: float
    confidence: np.ndarray | None = None

    def _stall_index(self) -> int:
        """Index of the first local maximum of CL (the stall), searching upward."""
        cl = self.cl
        for i in range(1, len(cl) - 1):
            if cl[i] >= cl[i - 1] and cl[i] > cl[i + 1] and self.alpha[i] > 0:
                # Ignore wiggles: a real stall stays below this peak for a while.
                ahead = cl[i + 1 : i + 9]
                if np.all(ahead <= cl[i] + 1e-3):
                    return i
        return int(np.argmax(cl))

    @property
    def attached(self) -> slice:
        """Pre-stall part of the sweep, where CL rises with alpha."""
        return slice(0, self._stall_index() + 1)

    def cl_max(self) -> tuple[float, float]:
        """(CL_max, alpha at CL_max in deg)."""
        i = self._stall_index()
        return float(self.cl[i]), float(self.alpha[i])

    def stall_retention(self, delta_deg: float = 3.0) -> float:
        """CL ``delta_deg`` past the stall divided by CL_max.

        1.0 is a perfectly flat-topped lift curve (gentle, mushy stall that the
        pilot can feel coming); 0.7 means 30 % of the lift vanishes almost at
        once (a sharp break, likely to drop a wing).
        """
        clmax, a_s = self.cl_max()
        after = float(np.interp(a_s + delta_deg, self.alpha, self.cl))
        return after / clmax

    def cm0(self) -> float:
        """Pitching moment at zero lift (the self-trim number)."""
        s = self.attached
        cl, cm = self.cl[s], self.cm[s]
        if cl.min() <= 0 <= cl.max():
            order = np.argsort(cl)
            return float(np.interp(0.0, cl[order], cm[order]))
        # CL never crosses zero in the sweep: extrapolate from the lowest points.
        k = min(8, len(cl))
        p_cm = np.polyfit(cl[:k], cm[:k], 1)
        return float(np.polyval(p_cm, 0.0))

    def at_cl(self, cl_target: float, values: np.ndarray) -> float:
        """Interpolate ``values`` (cd or cm) at a lift coefficient in the attached range."""
        s = self.attached
        cl = self.cl[s]
        if not cl.min() <= cl_target <= cl.max():
            return math.nan
        order = np.argsort(cl)
        return float(np.interp(cl_target, cl[order], values[s][order]))

    def cd_at_cl(self, cl_target: float) -> float:
        return self.at_cl(cl_target, self.cd)

    def cm_at_cl(self, cl_target: float) -> float:
        return self.at_cl(cl_target, self.cm)


# --------------------------------------------------------------------------
# NeuralFoil runs
# --------------------------------------------------------------------------

NOMINAL_N_CRIT = 9.0  # XFOIL default: smooth model in calm air
BUILT_N_CRIT = 5.0  # earlier transition: tape seams, film wrinkles, dust, turbulent air
BUILT_TE_THICKNESS_M = 1.0e-3  # hot-wire foam / printed trailing edge
MODEL_SIZE = "xlarge"


def load_airfoil(key: str):
    import aerosandbox as asb

    return asb.Airfoil(key)


def run_polar(airfoil, re: float, mach: float, n_crit: float = NOMINAL_N_CRIT) -> Polar:
    """NeuralFoil alpha sweep, returned as a :class:`Polar`."""
    r = airfoil.get_aero_from_neuralfoil(
        alpha=ALPHAS_DEG, Re=re, mach=mach, n_crit=n_crit, model_size=MODEL_SIZE
    )
    return Polar(
        alpha=ALPHAS_DEG.copy(),
        cl=np.asarray(r["CL"], float),
        cd=np.asarray(r["CD"], float),
        cm=np.asarray(r["CM"], float),
        re=float(re),
        confidence=np.asarray(r["analysis_confidence"], float),
    )


def as_built(airfoil, chord_m: float):
    """The airfoil with a blunt trailing edge of :data:`BUILT_TE_THICKNESS_M`."""
    return airfoil.set_TE_thickness(BUILT_TE_THICKNESS_M / chord_m)


@dataclass
class AirfoilResult:
    candidate: Candidate
    thickness: float  # max t/c
    depth_mm: float  # deepest point of the section on the role's smallest chord
    spar_depth_mm: float  # section depth at 25 % chord on the smallest chord
    cm0: float  # clean zero-lift moment at cruise (textbook number, for reference)
    cm_trim: float  # worst Cm at the CL actually flown, over the window, clean or as built
    cl_max: float  # clean, at the lowest stall Reynolds number
    alpha_stall: float
    retention: float  # stall gentleness (CL 3 deg past stall / CL_max)
    cl_max_built: float
    cd_window: float  # mean clean CD across the airspeed window
    cd_built_increase: float  # fractional CD change as built, mean over the window
    ld_cruise: float  # clean section L/D at cruise
    confidence: float  # lowest NeuralFoil confidence over the flown CL range
    stall_confidence: float  # lowest confidence within 3 deg of the stall (post-stall is the weak spot)
    stall_polar: Polar
    cruise_polar: Polar
    built_cruise_polar: Polar
    window: list[dict] = field(default_factory=list)  # per speed: V, CL, CD, Cm clean/built
    gates: dict[str, bool] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.gates.values())

    @property
    def total(self) -> float:
        return sum(WEIGHTS[k] * v for k, v in self.scores.items())


def _confidence_in_range(p: Polar, cl_lo: float, cl_hi: float) -> float:
    s = p.attached
    mask = (p.cl[s] >= cl_lo) & (p.cl[s] <= cl_hi)
    return float(np.min(p.confidence[s][mask])) if mask.any() else 0.0


def evaluate(cand: Candidate, cond: Conditions, airfoil=None) -> AirfoilResult:
    """Run every polar a candidate needs and reduce them to the selection metrics."""
    af = airfoil if airfoil is not None else load_airfoil(cand.key)
    chord = cond.chord(cand.role)
    small = cond.smallest_chord(cand.role)
    # As built: blunt TE sized on the smallest chord (worst case), early transition.
    built = as_built(af, small)

    # Stall happens at the field, slowest speed, smallest chord: lowest Re.
    re_stall = cond.reynolds(cond.v_stall, small, at_field=True)
    stall = run_polar(af, re_stall, cond.mach(cond.v_stall))
    built_stall = run_polar(built, re_stall, cond.mach(cond.v_stall), BUILT_N_CRIT)
    clmax, a_s = stall.cl_max()

    # Across the airspeed window: at each speed the wing flies one CL at one Re.
    window, confs = [], []
    for v in cond.window_speeds():
        re, m, cl = cond.reynolds(v, chord), cond.mach(v), cond.cl_required(v)
        p = run_polar(af, re, m)
        b = run_polar(built, re, m, BUILT_N_CRIT)
        window.append(dict(v=float(v), re=re, cl=cl, cd=p.cd_at_cl(cl), cd_built=b.cd_at_cl(cl),
                           cm=p.cm_at_cl(cl), cm_built=b.cm_at_cl(cl)))
        confs.append(_confidence_in_range(p, 0.0, cond.cl_required(cond.v_min)))
    cruise = run_polar(af, cond.reynolds(cond.v_cruise, chord), cond.mach(cond.v_cruise))
    built_cruise = run_polar(built, cond.reynolds(cond.v_cruise, chord), cond.mach(cond.v_cruise), BUILT_N_CRIT)
    cl_c = cond.cl_required(cond.v_cruise)

    cms = [w["cm"] for w in window] + [w["cm_built"] for w in window]
    return AirfoilResult(
        candidate=cand,
        thickness=float(af.max_thickness()),
        depth_mm=float(af.max_thickness()) * small * 1000,
        spar_depth_mm=float(af.local_thickness(0.25)) * small * 1000,
        cm0=cruise.cm0(),
        cm_trim=float(np.nanmin(cms)) if not np.all(np.isnan(cms)) else math.nan,
        cl_max=clmax,
        alpha_stall=a_s,
        retention=stall.stall_retention(),
        cl_max_built=built_stall.cl_max()[0],
        cd_window=float(np.nanmean([w["cd"] for w in window])),
        cd_built_increase=float(np.nanmean([w["cd_built"] / w["cd"] - 1 for w in window])),
        ld_cruise=cl_c / cruise.cd_at_cl(cl_c),
        confidence=min(confs),
        stall_confidence=float(np.min(stall.confidence[np.abs(stall.alpha - a_s) <= 3.0])),
        stall_polar=stall,
        cruise_polar=cruise,
        built_cruise_polar=built_cruise,
        window=window,
    )


# --------------------------------------------------------------------------
# Gates and scores
# --------------------------------------------------------------------------

# Pass/fail. Every candidate must clear these regardless of its score.
#  * Outer panels: a 10 mm carbon tube with 1 mm of material each side must fit
#    at 25 % chord of the tip chord.
#  * Centre body: ~40 mm of payload (21700 cells, Pi Zero + camera, flight
#    controller) plus 3 mm walls top and bottom must fit at the deepest point.
MIN_DEPTH_MM = {"outer": ("spar_depth_mm", 12.0), "center": ("depth_mm", 46.0)}
# Worst pitching moment at the CL actually flown, anywhere in the airspeed
# window, clean or as built. To trim with the elevons neutral the whole wing
# needs Cm about its aerodynamic centre of about +CL x static margin (+0.02 to
# +0.04 here). The centre body is big-chord and barely swept, so its section
# must not fight that. The swept outer panels can be helped by washout (tips
# twisted nose-down), so a small nose-down section moment is acceptable; M1.4
# (VLM) checks the whole wing trims within AC-8 (+-3 deg elevon).
MIN_CM_TRIM = {"outer": -0.02, "center": 0.0}
MIN_CONFIDENCE = 0.7  # below this NeuralFoil is extrapolating in the flown CL range

# Scores: each metric mapped linearly onto 0..1 between a "poor" and a "great"
# value, then clipped. Fixed scales (not "best of this list = 1") so adding a
# candidate never changes another candidate's score.
SCALES = {
    "pitch": ("cm_trim", -0.03, 0.03),  # nose-down .. comfortably self-trimming
    "low_speed": ("cl_max_built", 0.8, 1.4),  # as built, at the stall Reynolds number
    "gentle_stall": ("retention", 0.70, 1.00),
    "cruise_drag": ("cd_window", 0.016, 0.008),  # lower is better
    "forgiving": ("cd_built_increase", 0.30, 0.0),  # drag rise when built imperfectly
    "thickness": ("thickness", 0.09, 0.14),
}
# Ethan's priorities: robust and forgiving over squeezing out drag.
WEIGHTS = {
    "pitch": 0.25,
    "low_speed": 0.20,
    "gentle_stall": 0.15,
    "cruise_drag": 0.15,
    "forgiving": 0.15,
    "thickness": 0.10,
}


def scale(value: float, poor: float, great: float) -> float:
    if math.isnan(value):
        return 0.0
    return float(np.clip((value - poor) / (great - poor), 0.0, 1.0))


def score(res: AirfoilResult) -> AirfoilResult:
    role = res.candidate.role
    attr, depth = MIN_DEPTH_MM[role]
    res.gates = {
        f"depth >= {depth:g} mm": getattr(res, attr) >= depth,
        f"Cm in window >= {MIN_CM_TRIM[role]:+.2f}": res.cm_trim >= MIN_CM_TRIM[role],
        "NeuralFoil confident": res.confidence >= MIN_CONFIDENCE,
    }
    res.scores = {k: scale(getattr(res, attr), lo, hi) for k, (attr, lo, hi) in SCALES.items()}
    return res


def rank(results: list[AirfoilResult]) -> list[AirfoilResult]:
    """Passing candidates first, then by weighted score."""
    return sorted(results, key=lambda r: (not r.passed, -r.total))


def wing_cl_max_estimate(section_cl_max: float) -> float:
    """Rough trimmed wing CL_max from a section CL_max.

    A swept, tapered wing reaches about 90 % of its section CL_max, and a
    tailless wing loses roughly another 10 % because it trims with the elevons
    up. Replaced by the VLM result in M1.4.
    """
    return 0.9 * 0.9 * section_cl_max


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def compare(cond: Conditions, candidates=OUTER_CANDIDATES + CENTER_CANDIDATES) -> dict[str, list[AirfoilResult]]:
    out: dict[str, list[AirfoilResult]] = {"outer": [], "center": []}
    cache = {}
    for c in candidates:
        af = cache.setdefault(c.key, load_airfoil(c.key))
        out[c.role].append(score(evaluate(c, cond, af)))
    return {k: rank(v) for k, v in out.items()}


def conditions_table(cond: Conditions) -> str:
    rows = [
        ("Stall, tip chord, at the field", cond.v_stall, cond.chord_tip_m, True),
        ("Stall, centre body", cond.v_stall, cond.chord_center_m, True),
        ("Slowest survey speed, mean chord", cond.v_min, cond.chord_mean_m, False),
        ("Cruise, mean chord", cond.v_cruise, cond.chord_mean_m, False),
        ("Cruise, centre body", cond.v_cruise, cond.chord_center_m, False),
        ("Fastest survey speed, centre body", cond.v_max, cond.chord_center_m, False),
    ]
    lines = ["| Condition | Airspeed (m/s) | Chord (mm) | CL needed | Reynolds number |", "|---|---|---|---|---|"]
    for name, v, c, f in rows:
        cl = "CL_max" if f else f"{cond.cl_required(v):.2f}"
        lines.append(f"| {name} | {v:.1f} | {c*1000:.0f} | {cl} | {cond.reynolds(v, c, f):,.0f} |")
    return "\n".join(lines)


def _pct(x: float) -> str:
    return "0%" if abs(x) < 0.005 else f"{x:+.0%}"


def results_table(results: list[AirfoilResult]) -> str:
    head = (
        "| Rank | Airfoil | t/c | Depth (mm) | Cm0 | Worst Cm in window | CL_max clean / built "
        "| Stall retention (confidence) | Mean CD in window | Cruise L/D | CD change as built | Gates | Score |"
    )
    lines = [head, "|" + "---|" * 13]
    for i, r in enumerate(results, 1):
        gates = "pass" if r.passed else "FAIL: " + ", ".join(k for k, ok in r.gates.items() if not ok)
        depth = r.spar_depth_mm if r.candidate.role == "outer" else r.depth_mm
        lines.append(
            f"| {i} | {r.candidate.name} | {r.thickness:.1%} | {depth:.0f} | {r.cm0:+.3f} "
            f"| {r.cm_trim:+.3f} | {r.cl_max:.2f} / {r.cl_max_built:.2f} | {r.retention:.2f} ({r.stall_confidence:.2f}) "
            f"| {r.cd_window:.4f} | {r.ld_cruise:.0f} | {_pct(r.cd_built_increase)} | {gates} | {r.total:.2f} |"
        )
    return "\n".join(lines)


ALTERNATE_WEIGHTS = {
    "equal": {k: 1 / 6 for k in WEIGHTS},
    "pitch-heavy": dict(pitch=0.40, low_speed=0.15, gentle_stall=0.10, cruise_drag=0.10, forgiving=0.15, thickness=0.10),
    "lift-heavy": dict(pitch=0.20, low_speed=0.35, gentle_stall=0.15, cruise_drag=0.10, forgiving=0.10, thickness=0.10),
    "drag-heavy": dict(pitch=0.20, low_speed=0.15, gentle_stall=0.10, cruise_drag=0.35, forgiving=0.10, thickness=0.10),
}


def weight_sensitivity(results: list[AirfoilResult], weight_sets=None) -> dict[str, str | None]:
    """Winner among passing candidates under other reasonable weightings.

    If the same airfoil wins under most of them, the choice does not hinge on
    the exact weights.
    """
    weight_sets = weight_sets or {"baseline": WEIGHTS, **ALTERNATE_WEIGHTS}
    passing = [r for r in results if r.passed]
    out = {}
    for name, w in weight_sets.items():
        best = max(passing, key=lambda r: sum(w[k] * v for k, v in r.scores.items()), default=None)
        out[name] = best.candidate.name if best else None
    return out


def recommendation(ranked: dict[str, list[AirfoilResult]]) -> str:
    lines = []
    for role, label in (("outer", "Outer panels"), ("center", "Centre body")):
        best = next((r for r in ranked[role] if r.passed), None)
        if best is None:
            lines.append(f"{label}: no candidate passes the gates.")
            continue
        lines.append(
            f"{label}: {best.candidate.name} (score {best.total:.2f}; worst Cm in window {best.cm_trim:+.3f}, "
            f"CL_max {best.cl_max_built:.2f} as built, {best.thickness:.1%} thick)"
        )
    outer = next((r for r in ranked["outer"] if r.passed), None)
    if outer:
        lines.append(
            f"Wing CL_max estimate from the outer section: {wing_cl_max_estimate(outer.cl_max_built):.2f} "
            "(the constraint analysis assumes 1.0; see docs/airfoils.md)"
        )
    return "\n".join(lines)


def plot_all(ranked: dict[str, list[AirfoilResult]], cond: Conditions, outdir: str) -> list[str]:  # pragma: no cover
    """Save the comparison figures. Needs matplotlib."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    # Categorical slots in fixed order (validated default palette), plus a
    # line style per candidate so identity never relies on colour alone.
    colours = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
    styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
    plt.rcParams.update({"axes.edgecolor": "#999999", "axes.labelcolor": "#191919", "xtick.color": "#666666",
                         "ytick.color": "#666666", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    # Each airfoil keeps one colour across every figure.
    names = []
    for role in ("outer", "center"):
        for r in sorted(ranked[role], key=lambda r: r.candidate.name):
            if r.candidate.name not in names:
                names.append(r.candidate.name)
    colour = {n: colours[i % len(colours)] for i, n in enumerate(names)}

    for role, title in (("outer", "Outer-panel candidates"), ("center", "Centre-body candidates")):
        res = sorted(ranked[role], key=lambda r: r.candidate.name)
        fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
        (a1, a2), (a3, a4) = axs
        for i, r in enumerate(res):
            kw = dict(color=colour[r.candidate.name], ls=styles[i % len(styles)], lw=2, label=r.candidate.name)
            s = r.stall_polar
            a1.plot(s.alpha, s.cl, **kw)
            c = r.cruise_polar
            att = c.attached
            a2.plot(c.cd[att], c.cl[att], **kw)
            a3.plot(c.cl[att], c.cm[att], **kw)
            v = [w["v"] for w in r.window]
            a4.plot(v, [min(w["cm"], w["cm_built"]) for w in r.window], marker="o", ms=4, **kw)
        re_s = res[0].stall_polar.re
        re_c = res[0].cruise_polar.re
        a1.set(xlabel="Angle of attack (deg)", ylabel="Lift coefficient CL",
               title=f"Lift curve at stall speed (Re {re_s:,.0f})")
        a2.set(xlabel="Drag coefficient CD", ylabel="Lift coefficient CL",
               title=f"Drag polar at cruise (Re {re_c:,.0f})")
        a2.axhline(cond.cl_required(cond.v_cruise), color="#666666", lw=1, ls=":")
        a2.text(a2.get_xlim()[1], cond.cl_required(cond.v_cruise), " cruise CL", va="bottom", ha="right",
                color="#666666", fontsize=8)
        a3.set(xlabel="Lift coefficient CL", ylabel="Pitching moment Cm (c/4)",
               title="Pitching moment: above 0 at CL = 0 means self-trimming")
        a3.axhline(0, color="#191919", lw=0.8)
        a3.axvline(0, color="#191919", lw=0.8)
        a3.axvspan(cond.cl_required(cond.v_max), cond.cl_required(cond.v_min), color="#666666", alpha=0.08, lw=0)
        a3.text(cond.cl_required(cond.v_cruise), a3.get_ylim()[1], "CL range flown", ha="center", va="top",
                fontsize=8, color="#666666")
        a4.axhline(0, color="#191919", lw=0.8)
        a4.axhline(MIN_CM_TRIM[role], color="#666666", lw=1, ls=":")
        a4.text(cond.v_max, MIN_CM_TRIM[role], "gate ", ha="right", va="bottom", fontsize=8, color="#666666")
        a4.set(xlabel="Airspeed (m/s)", ylabel="Cm at the CL flown (worst of clean / as built)",
               title="Pitching moment across the airspeed window")
        a1.legend(frameon=False)
        fig.suptitle(f"{title} (NeuralFoil, design day at the field)", fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        p = out / f"airfoil_polars_{role}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))

    # Shapes
    fig, ax = plt.subplots(figsize=(6.5, 6))
    seen = []
    for role in ("outer", "center"):
        for r in ranked[role]:
            if r.candidate.key in seen:
                continue
            seen.append(r.candidate.key)
            af = load_airfoil(r.candidate.key)
            off = -0.16 * (len(seen) - 1)
            ax.plot(af.x(), af.y() + off, color=colour[r.candidate.name], lw=1.5)
            ax.text(1.03, off, f"{r.candidate.name}  ({r.thickness:.1%})", va="center", fontsize=9, color="#191919")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.02, 1.35)
    ax.set_title("Candidate shapes (reflex = the trailing edge curling up)", fontsize=11, loc="left")
    fig.tight_layout()
    p = out / "airfoil_shapes.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # Score breakdown
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    keys = list(WEIGHTS)
    seg_colours = colours[:len(keys)]
    for ax, role, title in ((axs[0], "outer", "Outer panels"), (axs[1], "center", "Centre body")):
        res = ranked[role][::-1]
        for j, r in enumerate(res):
            left = 0.0
            for k, col in zip(keys, seg_colours):
                w = WEIGHTS[k] * r.scores[k]
                ax.barh(j, w, left=left, color=col, height=0.6, edgecolor="white", linewidth=2,
                        label=k.replace("_", " ") if j == 0 else None)
                left += w
            ax.text(left + 0.01, j, f"{r.total:.2f}" + ("" if r.passed else "  (fails a gate)"),
                    va="center", fontsize=8, color="#191919")
        ax.set_yticks(range(len(res)), [r.candidate.name for r in res])
        ax.set_xlim(0, 1.15)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Weighted score (max 1.0)")
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = out / "airfoil_scores.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))
    return written


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(prog="python -m manta_mapper.airfoils", description=__doc__.splitlines()[0])
    p.add_argument("mission", nargs="?", help="mission TOML (defaults built in)")
    p.add_argument("--mass", type=float, default=1.5, help="take-off mass guess, kg")
    p.add_argument("--plots", help="directory to save figures into")
    args = p.parse_args(argv)
    req = MissionRequirements.from_toml(args.mission) if args.mission else MissionRequirements()
    cond = conditions_from_requirements(req, args.mass)
    ranked = compare(cond)
    print("== Flight conditions ==")
    print(conditions_table(cond))
    for role in ("outer", "center"):
        print(f"\n== {role} candidates ==")
        print(results_table(ranked[role]))
    print("\n== Winner under other weightings ==")
    for role in ("outer", "center"):
        print(role, weight_sensitivity(ranked[role]))
    print("\n== Recommendation ==")
    print(recommendation(ranked))
    if args.plots:
        for f in plot_all(ranked, cond, args.plots):
            print(f"Saved {f}")


if __name__ == "__main__":  # pragma: no cover
    main()
