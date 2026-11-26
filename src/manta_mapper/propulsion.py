"""Motor, propeller and battery models (milestone M1.5).

Turns "how much thrust does the aircraft need" into "which motor, prop and
pack, how many amps, and how many minutes". Four small models, chained:

1. **Drag** comes from the M1.4 trimmed polar (:mod:`manta_mapper.aero`),
   fitted as ``CD = a + b CL + c CL^2`` so it can be evaluated anywhere, then
   multiplied by a *realism factor* so the best L/D is 14, 12 or 10 instead
   of the model's ~16. Small printed flying wings usually measure 10-14.
2. **Propeller**: blade-element momentum theory (BEMT) on a generic
   thin-electric / folding blade (chord and pitch distributions below), with a
   Prandtl tip loss and a simple low-Reynolds-number section. On a 9x6 it
   gives static CT 0.11, CP 0.04-0.05, peak efficiency 0.73 at J = 0.6 and
   zero thrust at J = 0.8, which is where published wind-tunnel data for
   thin electric props sit. A pusher behind a thick pod loses some thrust to
   the pod wake and blockage: :data:`PUSHER_THRUST_FACTOR`.
3. **Motor**: the standard three-constant brushed-DC model (Kv, winding
   resistance, no-load current), which is what hobby outrunner datasheets
   give and what tools like eCalc use.
4. **Battery**: Li-ion 21700 cells, open-circuit voltage from a typical NMC
   discharge curve, internal resistance per cell plus wiring. The ESC duty
   cycle tops out at :data:`MAX_DUTY`.

Then :func:`select_propulsion` tries every motor x prop x pack in the small
catalogue below, throws out any that miss a requirement in the hot/high
design-day air with pessimistic drag, and picks the cheapest one whose cruise
power is within :data:`CRUISE_POWER_TOLERANCE` of the most efficient.
:func:`mission_energy` then checks the 25 ha and 50 ha flights.

Part numbers are common, cheap hobby parts; masses and electrical constants
are typical datasheet values and prices are rough US street prices in late
2026. Both drift, so treat the catalogue as a starting list, not a quote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .atmosphere import dynamic_viscosity, isa
from .requirements import G0, MissionRequirements

INCH = 0.0254

# --------------------------------------------------------------------------
# Modelling settings (each one a judgement call, with the reason)
# --------------------------------------------------------------------------

#: Thrust actually delivered / isolated-prop thrust for a pusher sitting in
#: the wake of a 63 mm deep pod. Published pusher installations lose about
#: 5-10 %; 0.90 is the cautious end. It also soaks up swirl and hub losses
#: that the strip model leaves out.
PUSHER_THRUST_FACTOR = 0.90

#: Shaft torque multiplier on the strip model. The BEMT leaves out swirl, the
#: hub and folding hinge, and the rough nylon blades; on its own it gives a
#: static figure of merit near 0.7 where small props measure 0.5-0.6. With
#: 1.15 a 10x6 comes out at static FM ~0.6 and peak efficiency ~0.64, which is
#: the cautious side of published thin-electric and folding-prop data.
TORQUE_FACTOR = 1.15

#: Highest ESC duty cycle (full throttle never quite reaches battery voltage).
MAX_DUTY = 0.97
ESC_EFFICIENCY = 0.95
#: Wires, XT60 and ESC FETs, as series resistance (ohm).
WIRING_RESISTANCE_OHM = 0.012
#: BEC efficiency for the avionics (switching regulators).
BEC_EFFICIENCY = 0.85

#: Pack mass on top of the cells: nickel strip, wrap, balance lead, XT60, 14 AWG.
PACK_OVERHEAD_KG = 0.016

#: Fraction of nominal energy that can be used: the voltage sag near empty,
#: a 3.3 V/cell loaded cut-off, ageing and a warm-but-not-hot pack.
#: (Same value as ``Assumptions.battery_usable_fraction``.)
USABLE_FRACTION = 0.85

#: Selection requirements (on top of AC-5 and AC-6 from the mission file).
STATIC_THRUST_TO_WEIGHT = 0.60  # forgiving hand launches: ~0.5 is the usual floor
MOTOR_CURRENT_USE = 0.90  # peak motor current <= 90 % of the rated max
CELL_CURRENT_USE = 0.80  # peak pack current <= 80 % of the cells' continuous rating (hot cells sag)
ESC_CURRENT_USE = 0.75  # peak current <= 75 % of the ESC rating (hot day, enclosed pod)
TOP_SPEED_OVER_SURVEY = 1.15  # level top speed >= 1.15 x the fastest survey speed
CRUISE_POWER_TOLERANCE = 0.08  # pick the cheapest within 8 % of the best cruise power

#: Battery states used for the checks. Launch happens on a fresh pack (80 % is
#: a conservative "it sat in the sun and you ran the motor on the bench"), the
#: worst climb is a go-around with only the reserve left.
SOC_LAUNCH = 0.80
SOC_RESERVE = 0.30
SOC_CRUISE = 0.55

#: L/D cases: the model's own value, then "planning" (sizing case) and
#: "pessimistic". Each scales all drag by (model best L/D) / (target).
LD_PLANNING = 12.0
LD_PESSIMISTIC = 10.0
LD_CASES = (None, 14.0, LD_PLANNING, LD_PESSIMISTIC)

# --------------------------------------------------------------------------
# Parts catalogue
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Motor:
    name: str
    kv_rpm_per_v: float
    resistance_ohm: float
    no_load_current_a: float
    mass_kg: float
    max_current_a: float  # datasheet maximum (usually a 30-60 s rating)
    cost_usd: float
    length_m: float = 0.030
    diameter_m: float = 0.028

    @property
    def kv_rad(self) -> float:
        """rad/s per volt (and 1/Kv is the torque constant in N m/A)."""
        return self.kv_rpm_per_v * 2 * math.pi / 60


@dataclass(frozen=True)
class Prop:
    name: str
    diameter_in: float
    pitch_in: float
    mass_kg: float  # blades + yoke + spinner/adapter
    cost_usd: float
    blades: int = 2
    chord_scale: float = 0.85  # folding blades are a bit narrower than a fixed APC-E

    @property
    def diameter_m(self) -> float:
        return self.diameter_in * INCH

    @property
    def pitch_m(self) -> float:
        return self.pitch_in * INCH


@dataclass(frozen=True)
class Cell:
    name: str
    capacity_ah: float
    max_current_a: float  # continuous
    resistance_ohm: float  # DC internal resistance
    mass_kg: float
    cost_usd: float
    nominal_v: float = 3.6


@dataclass(frozen=True)
class Pack:
    cell: Cell
    series: int = 3
    parallel: int = 1

    @property
    def name(self) -> str:
        return f"{self.series}S{self.parallel}P {self.cell.name}"

    @property
    def n_cells(self) -> int:
        return self.series * self.parallel

    @property
    def capacity_ah(self) -> float:
        return self.cell.capacity_ah * self.parallel

    @property
    def nominal_energy_wh(self) -> float:
        return self.cell.nominal_v * self.series * self.capacity_ah

    @property
    def usable_energy_wh(self) -> float:
        return USABLE_FRACTION * self.nominal_energy_wh

    @property
    def resistance_ohm(self) -> float:
        return self.series * self.cell.resistance_ohm / self.parallel + WIRING_RESISTANCE_OHM

    @property
    def max_current_a(self) -> float:
        return self.cell.max_current_a * self.parallel

    @property
    def mass_kg(self) -> float:
        return self.n_cells * self.cell.mass_kg + PACK_OVERHEAD_KG * self.parallel ** 0.5

    @property
    def cost_usd(self) -> float:
        return self.n_cells * self.cell.cost_usd + 8.0  # strip, wrap, XT60, balance lead

    def open_circuit_v(self, soc: float) -> float:
        return self.series * cell_ocv(soc)


@dataclass(frozen=True)
class ESC:
    name: str
    max_current_a: float
    mass_kg: float
    cost_usd: float


MOTORS = (
    Motor("A2212/13T 1000KV (generic)", 1000, 0.090, 0.50, 0.052, 15, 9),
    Motor("SunnySky X2212 980KV", 980, 0.133, 0.30, 0.057, 15, 24),
    Motor("SunnySky X2212 1250KV", 1250, 0.075, 0.50, 0.057, 20, 24),
    Motor("SunnySky X2216 880KV", 880, 0.105, 0.40, 0.068, 20, 27, length_m=0.034),
    Motor("SunnySky X2216 1100KV", 1100, 0.075, 0.55, 0.068, 26, 27, length_m=0.034),
    Motor("T-Motor AT2312 1150KV", 1150, 0.087, 0.60, 0.061, 20, 30),
)

PROPS = tuple(
    Prop(f"folding {d:g}x{p:g} (nylon blades, 40 mm yoke, spinner)", d, p, m, c)
    for d, p, m, c in ((8, 6, 0.014, 9), (9, 5, 0.016, 9), (9, 6, 0.017, 9), (10, 6, 0.019, 10), (10, 7, 0.020, 10))
)

CELLS = (
    Cell("Samsung 50E", 5.0, 9.8, 0.025, 0.069, 5.0),
    Cell("Samsung 50S", 5.0, 25.0, 0.015, 0.072, 6.5),
    Cell("Molicel P45B", 4.5, 45.0, 0.009, 0.070, 7.0),
)

PACKS = tuple(Pack(c, 3, 1) for c in CELLS) + (Pack(CELLS[0], 3, 2), Pack(CELLS[2], 3, 2))

ESCS = (
    ESC("30 A BLHeli_S ESC (no BEC needed: FC has one)", 30, 0.025, 14),
    ESC("40 A BLHeli_S ESC", 40, 0.035, 18),
)

# --------------------------------------------------------------------------
# Li-ion cell voltage
# --------------------------------------------------------------------------

# Typical NMC 21700 open-circuit voltage vs state of charge (rest voltage).
_SOC = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
_OCV = np.array([3.00, 3.25, 3.35, 3.48, 3.56, 3.63, 3.70, 3.78, 3.86, 3.95, 4.05, 4.18])


def cell_ocv(soc: float) -> float:
    return float(np.interp(soc, _SOC, _OCV))


# --------------------------------------------------------------------------
# Drag polar
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QuadraticPolar:
    """CD = a + b CL + c CL^2, times ``factor`` (the realism factor)."""

    a: float
    b: float
    c: float
    factor: float = 1.0
    label: str = "model"

    @classmethod
    def fit(cls, cl, cd, label: str = "model") -> "QuadraticPolar":
        c, b, a = np.polyfit(np.asarray(cl), np.asarray(cd), 2)
        return cls(float(a), float(b), float(c), 1.0, label)

    def cd(self, cl):
        cl = np.asarray(cl)
        return self.factor * (self.a + self.b * cl + self.c * cl**2)

    @property
    def cl_best_ld(self) -> float:
        return math.sqrt(self.a / self.c)

    @property
    def best_ld(self) -> float:
        cl = self.cl_best_ld
        return float(cl / self.cd(cl))

    def scaled_to(self, best_ld: float | None) -> "QuadraticPolar":
        """Same shape, every drag source scaled so the best L/D is ``best_ld``."""
        if best_ld is None:
            return self
        base = QuadraticPolar(self.a, self.b, self.c)
        return QuadraticPolar(self.a, self.b, self.c, base.best_ld / best_ld, f"L/D {best_ld:g}")


#: Fit of the M1.4 trimmed polar (docs/stability_report.txt, 1.5 kg, design CG,
#: as built). Used by the tests and by ``--quick``; the CLI recomputes it.
M14_POLAR = QuadraticPolar.fit(
    [0.807, 0.665, 0.557, 0.472, 0.406, 0.352, 0.309, 0.273, 0.243, 0.217, 0.195, 0.177, 0.161],
    [0.0589, 0.0441, 0.0349, 0.0289, 0.0248, 0.0219, 0.0198, 0.0183, 0.0170, 0.0161, 0.0154, 0.0147, 0.0142],
)


# --------------------------------------------------------------------------
# Propeller: blade-element momentum theory
# --------------------------------------------------------------------------

# Generic thin-electric blade: chord/R at r/R (APC "E" style planform).
_X_TAB = np.array([0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00])
_C_TAB = np.array([0.11, 0.16, 0.18, 0.17, 0.14, 0.10, 0.05])
_N_STATIONS = 30
# Section: thin cambered plate-like airfoil at Re 30k-150k
_A0 = math.radians(-4.0)  # zero-lift angle
_CLA = 5.5  # per rad (0.88 x 2 pi: low Reynolds number, thin section)
_CLMAX, _CLMIN = 1.15, -0.5


def _section(alpha, re):
    cl = np.clip(_CLA * (alpha - _A0), _CLMIN, _CLMAX)
    a_stall = _A0 + _CLMAX / _CLA
    cd0 = 0.02 * (np.maximum(re, 2e4) / 1e5) ** -0.4
    cd = cd0 + 0.02 * (cl - 0.45) ** 2 + 1.8 * np.sin(np.clip(alpha - a_stall, 0, None)) ** 2
    return cl, cd


def prop_thrust_torque(prop: Prop, airspeed: float, rpm: float, rho: float, mu: float) -> tuple[float, float]:
    """Isolated-prop thrust (N) and shaft torque (N m) by BEMT (no swirl, Prandtl tip loss)."""
    R = prop.diameter_m / 2
    edges = np.linspace(0.15, 1.0, _N_STATIONS + 1)
    x = 0.5 * (edges[1:] + edges[:-1])
    dr = np.diff(edges) * R
    r = x * R
    c = np.interp(x, _X_TAB, _C_TAB) * R * prop.chord_scale
    beta = np.arctan(prop.pitch_m / (2 * np.pi * r))
    om = rpm * 2 * np.pi / 60
    B = prop.blades

    def terms(v):
        ua, ut = airspeed + v, om * r
        phi = np.arctan2(ua, ut)
        w2 = ua**2 + ut**2
        cl, cd = _section(beta - phi, rho * np.sqrt(w2) * c / mu)
        cn = cl * np.cos(phi) - cd * np.sin(phi)
        ct = cl * np.sin(phi) + cd * np.cos(phi)
        f = np.exp(-B / 2 * (1 - x) / (x * np.maximum(np.sin(phi), 1e-3)))
        F = np.maximum(2 / np.pi * np.arccos(np.clip(f, 0, 1)), 1e-3)
        residual = 4 * np.pi * r * rho * F * ua * v - 0.5 * rho * w2 * B * c * cn
        return residual, w2, cn, ct

    # induced velocity per annulus by bisection (momentum = blade element)
    lo, hi = np.zeros_like(x), np.full_like(x, 100.0)
    active = terms(lo)[0] < 0  # annuli making thrust
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        pos = terms(mid)[0] > 0
        hi, lo = np.where(pos, mid, hi), np.where(pos, lo, mid)
    v = np.where(active, 0.5 * (lo + hi), 0.0)
    _, w2, cn, ct = terms(v)
    q = 0.5 * rho * w2 * B * c
    return float(np.sum(q * cn * dr)), TORQUE_FACTOR * float(np.sum(q * ct * r * dr))


#: The operating-point solvers use a CT(J), CP(J) map from the BEMT at this
#: rpm instead of re-running it every time (Reynolds number then follows
#: airspeed but not rpm: a few percent, far below the other uncertainties).
MAP_RPM = 7000.0
_J_GRID = np.linspace(0.0, 1.2, 61)


@lru_cache(maxsize=None)
def _prop_map(prop: Prop, rho: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    n = MAP_RPM / 60
    D = prop.diameter_m
    ct, cq = [], []
    for J in _J_GRID:
        T, Q = prop_thrust_torque(prop, J * n * D, MAP_RPM, rho, mu)
        ct.append(T / (rho * n**2 * D**4))
        cq.append(Q / (rho * n**2 * D**5))
    return np.array(ct), np.array(cq)


def prop_map_thrust_torque(prop: Prop, airspeed: float, rpm: float, rho: float, mu: float) -> tuple[float, float]:
    """Thrust and torque from the cached CT/CQ map (fast; used by the solvers)."""
    ct, cq = _prop_map(prop, round(rho, 4), round(mu, 9))
    n = rpm / 60
    D = prop.diameter_m
    J = airspeed / (n * D)
    return (float(np.interp(J, _J_GRID, ct, right=ct[-1])) * rho * n**2 * D**4,
            float(np.interp(J, _J_GRID, cq, right=cq[-1])) * rho * n**2 * D**5)


def prop_coefficients(prop: Prop, J: float, rpm: float = 6000.0, rho: float = 1.225,
                      mu: float = 1.81e-5) -> tuple[float, float, float]:
    """(CT, CP, efficiency) at advance ratio J, for checking against published data."""
    n = rpm / 60
    D = prop.diameter_m
    T, Q = prop_thrust_torque(prop, J * n * D, rpm, rho, mu)
    ct = T / (rho * n**2 * D**4)
    cp = 2 * math.pi * Q / (rho * n**2 * D**5)
    return ct, cp, (J * ct / cp if cp > 0 else 0.0)


# --------------------------------------------------------------------------
# Motor + ESC + battery operating points
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Air:
    rho: float
    mu: float

    @classmethod
    def at(cls, altitude_m: float, temperature_offset_K: float) -> "Air":
        t, _, rho = isa(altitude_m, temperature_offset_K)
        return cls(rho, dynamic_viscosity(t))


@dataclass(frozen=True)
class PropulsionSet:
    motor: Motor
    prop: Prop
    pack: Pack
    esc: ESC = ESCS[0]

    @property
    def name(self) -> str:
        return f"{self.motor.name} + {self.prop.name.split(' (')[0]} + {self.pack.name}"

    @property
    def mass_kg(self) -> float:
        return self.motor.mass_kg + self.prop.mass_kg + self.pack.mass_kg + self.esc.mass_kg

    @property
    def cost_usd(self) -> float:
        return self.motor.cost_usd + self.prop.cost_usd + self.pack.cost_usd + self.esc.cost_usd


@dataclass(frozen=True)
class OperatingPoint:
    airspeed: float
    rpm: float
    thrust_n: float  # installed (after the pusher factor)
    shaft_power_w: float
    motor_current_a: float
    motor_voltage_v: float
    battery_current_a: float
    battery_voltage_v: float
    throttle: float  # ESC duty
    feasible: bool

    @property
    def electrical_power_w(self) -> float:
        """Power leaving the pack terminals."""
        return self.battery_current_a * self.battery_voltage_v

    @property
    def efficiency(self) -> float:
        """Thrust power / electrical power (prop x motor x ESC x installation)."""
        return self.thrust_n * self.airspeed / self.electrical_power_w if self.electrical_power_w > 0 else 0.0


def _point(ps: PropulsionSet, v: float, rpm: float, air: Air, soc: float) -> OperatingPoint:
    m, pack = ps.motor, ps.pack
    T, Q = prop_map_thrust_torque(ps.prop, v, rpm, air.rho, air.mu)
    om = rpm * 2 * math.pi / 60
    i_m = Q * m.kv_rad + m.no_load_current_a
    v_m = i_m * m.resistance_ohm + om / m.kv_rad
    p_batt = v_m * i_m / ESC_EFFICIENCY
    voc, R = pack.open_circuit_v(soc), pack.resistance_ohm
    disc = voc**2 - 4 * R * p_batt
    if disc < 0:  # the pack can't deliver this power at all
        return OperatingPoint(v, rpm, T * PUSHER_THRUST_FACTOR, Q * om, i_m, v_m, float("inf"), 0.0, float("inf"), False)
    i_b = (voc - math.sqrt(disc)) / (2 * R)
    v_b = voc - i_b * R
    duty = v_m / v_b
    return OperatingPoint(v, rpm, T * PUSHER_THRUST_FACTOR, Q * om, i_m, v_m, i_b, v_b, duty, duty <= MAX_DUTY)


def full_throttle(ps: PropulsionSet, airspeed: float, air: Air, soc: float) -> OperatingPoint:
    """Highest rpm the motor reaches at full throttle (bisection on rpm)."""
    lo, hi = 500.0, ps.motor.kv_rpm_per_v * ps.pack.open_circuit_v(1.0)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _point(ps, airspeed, mid, air, soc).feasible:
            lo = mid
        else:
            hi = mid
    return _point(ps, airspeed, lo, air, soc)


def for_thrust(ps: PropulsionSet, thrust_n: float, airspeed: float, air: Air, soc: float) -> OperatingPoint:
    """Part-throttle point that delivers ``thrust_n`` (installed). ``feasible`` False if it can't."""
    top = full_throttle(ps, airspeed, air, soc)
    if top.thrust_n < thrust_n:
        return OperatingPoint(**{**top.__dict__, "feasible": False})
    lo, hi = 300.0, top.rpm
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _point(ps, airspeed, mid, air, soc).thrust_n < thrust_n:
            lo = mid
        else:
            hi = mid
    return _point(ps, airspeed, hi, air, soc)


# --------------------------------------------------------------------------
# Aircraft performance with a given set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Conditions:
    """Air, speeds and loads for the checks, from the mission file."""

    mass_kg: float
    wing_area_m2: float
    stall_speed_m_s: float
    cruise_speed_m_s: float
    fastest_survey_m_s: float
    field: Air
    survey: Air
    survey_height_m: float
    avionics_w: float

    @property
    def weight_n(self) -> float:
        return self.mass_kg * G0

    def cl(self, air: Air, v: float, n: float = 1.0) -> float:
        return n * self.weight_n / (0.5 * air.rho * v**2 * self.wing_area_m2)

    def drag(self, polar: QuadraticPolar, air: Air, v: float, n: float = 1.0) -> float:
        return float(0.5 * air.rho * v**2 * self.wing_area_m2 * polar.cd(self.cl(air, v, n)))


def conditions(req: MissionRequirements, mass_kg: float, wing_area_m2: float, stall_speed_m_s: float,
               avionics_w: float) -> Conditions:
    from .survey import max_airspeed_for_trigger, plan_survey

    env = req.environment
    h = plan_survey(req).altitude_m
    return Conditions(
        mass_kg=mass_kg, wing_area_m2=wing_area_m2, stall_speed_m_s=stall_speed_m_s,
        cruise_speed_m_s=req.assumptions.cruise_speed_m_s, fastest_survey_m_s=max_airspeed_for_trigger(req),
        field=Air.at(env.field_elevation_m, env.temperature_offset_K),
        survey=Air.at(env.field_elevation_m + h, env.temperature_offset_K),
        survey_height_m=h, avionics_w=avionics_w,
    )


@dataclass
class Performance:
    ps: PropulsionSet
    static: OperatingPoint  # field, launch SoC
    launch: OperatingPoint  # full throttle at 1.2 Vs, field, launch SoC
    launch_gradient: float  # pessimistic drag
    climb: OperatingPoint  # full throttle, best climb speed, reserve SoC
    climb_rate_m_s: float  # pessimistic drag
    climb_speed_m_s: float
    top_speed_m_s: float  # level, survey altitude, reserve SoC, pessimistic drag
    cruise: OperatingPoint  # planning drag, survey altitude, mid SoC
    cruise_fast: OperatingPoint  # fastest survey speed, planning drag
    static_thrust_to_weight: float
    checks: dict[str, tuple[bool, str]] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return all(ok for ok, _ in self.checks.values())


def _top_speed(ps: PropulsionSet, cond: Conditions, polar: QuadraticPolar, soc: float) -> float:
    lo, hi = cond.cruise_speed_m_s, 45.0
    if full_throttle(ps, lo, cond.survey, soc).thrust_n < cond.drag(polar, cond.survey, lo):
        return 0.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if full_throttle(ps, mid, cond.survey, soc).thrust_n >= cond.drag(polar, cond.survey, mid):
            lo = mid
        else:
            hi = mid
    return lo


def evaluate(ps: PropulsionSet, cond: Conditions, req: MissionRequirements, model_polar: QuadraticPolar) -> Performance:
    """All the propulsion checks for one set at one mass."""
    planning = model_polar.scaled_to(LD_PLANNING)
    pess = model_polar.scaled_to(LD_PESSIMISTIC)
    W = cond.weight_n

    static = full_throttle(ps, 0.0, cond.field, SOC_LAUNCH)
    v_launch = 1.2 * cond.stall_speed_m_s
    launch = full_throttle(ps, v_launch, cond.field, SOC_LAUNCH)
    gradient = (launch.thrust_n - cond.drag(pess, cond.field, v_launch)) / W

    # best climb rate over a sweep of speeds, reserve SoC, pessimistic drag
    best = (-1e9, None, 0.0)
    for v in np.arange(max(1.25 * cond.stall_speed_m_s, 10.0), 18.01, 0.5):
        op = full_throttle(ps, float(v), cond.field, SOC_RESERVE)
        rc = (op.thrust_n - cond.drag(pess, cond.field, float(v))) * v / W
        if rc > best[0]:
            best = (rc, op, float(v))
    climb_rate, climb, v_climb = best

    top = _top_speed(ps, cond, pess, SOC_RESERVE)
    cruise = for_thrust(ps, cond.drag(planning, cond.survey, cond.cruise_speed_m_s), cond.cruise_speed_m_s,
                        cond.survey, SOC_CRUISE)
    fast = for_thrust(ps, cond.drag(planning, cond.survey, cond.fastest_survey_m_s), cond.fastest_survey_m_s,
                      cond.survey, SOC_CRUISE)

    tw = static.thrust_n / W
    lim = req.limits
    peak_motor = max(static.motor_current_a, launch.motor_current_a)
    peak_batt = max(static.battery_current_a, launch.battery_current_a, climb.battery_current_a)
    checks = {
        "static thrust/weight": (tw >= STATIC_THRUST_TO_WEIGHT,
                                 f"{tw:.2f} (want >= {STATIC_THRUST_TO_WEIGHT}, hot day, {SOC_LAUNCH:.0%} charge)"),
        "launch climb gradient (AC-6)": (gradient >= lim.launch_climb_gradient,
                                         f"{gradient:.2f} at {v_launch:.1f} m/s (want >= {lim.launch_climb_gradient}, L/D {LD_PESSIMISTIC:g})"),
        "climb rate (AC-5)": (climb_rate >= lim.min_climb_rate_m_s,
                              f"{climb_rate:.1f} m/s at {v_climb:.1f} m/s (want >= {lim.min_climb_rate_m_s}, {SOC_RESERVE:.0%} charge, L/D {LD_PESSIMISTIC:g})"),
        "top speed": (top >= TOP_SPEED_OVER_SURVEY * cond.fastest_survey_m_s,
                      f"{top:.1f} m/s (want >= {TOP_SPEED_OVER_SURVEY * cond.fastest_survey_m_s:.1f})"),
        "motor current": (peak_motor <= MOTOR_CURRENT_USE * ps.motor.max_current_a,
                          f"{peak_motor:.1f} A peak (rated {ps.motor.max_current_a:g} A)"),
        "cell current": (peak_batt <= CELL_CURRENT_USE * ps.pack.max_current_a,
                         f"{peak_batt:.1f} A peak (cells rated {ps.pack.max_current_a:g} A)"),
        "ESC current": (peak_batt <= ESC_CURRENT_USE * ps.esc.max_current_a,
                        f"{peak_batt:.1f} A peak ({ps.esc.max_current_a:g} A ESC)"),
        "cruise within throttle": (cruise.feasible and fast.feasible, f"throttle {cruise.throttle:.0%} at cruise"),
    }
    return Performance(ps, static, launch, gradient, climb, climb_rate, v_climb, top, cruise, fast, tw, checks)


# --------------------------------------------------------------------------
# Mission energy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MissionEnergy:
    area_ha: float
    polar_label: str
    mass_kg: float
    time_s: float
    energy_wh: float  # drawn from the cells (incl. resistive loss)
    usable_wh: float
    cruise_power_w: float  # propulsion, electrical
    turn_power_w: float
    climb_time_s: float

    @property
    def reserve(self) -> float:
        return 1 - self.energy_wh / self.usable_wh

    def ok(self, req: MissionRequirements) -> bool:
        return self.reserve >= req.limits.battery_reserve


def mission_energy(ps: PropulsionSet, cond: Conditions, req: MissionRequirements, polar: QuadraticPolar,
                   area_ha: float) -> MissionEnergy:
    """Energy for one survey: climb at full power, survey lines and turns, the rest at cruise."""
    from .survey import plan_survey

    plan = plan_survey(req, area_ha=area_ha)
    v = cond.cruise_speed_m_s
    R = ps.pack.resistance_ohm

    def drawn(op: OperatingPoint) -> float:
        # power out of the cells: terminal power + loss in the pack resistance + avionics via the BEC
        return op.electrical_power_w + op.battery_current_a**2 * R + cond.avionics_w / BEC_EFFICIENCY

    cruise = for_thrust(ps, cond.drag(polar, cond.survey, v), v, cond.survey, SOC_CRUISE)
    n_turn = 1 / math.cos(math.radians(req.survey.turn_bank_deg))
    turn = for_thrust(ps, cond.drag(polar, cond.survey, v, n_turn), v, cond.survey, SOC_CRUISE)

    # climb from the field to survey height at full throttle and the best climb speed
    best = (0.0, None, v)
    for vc in np.arange(max(1.25 * cond.stall_speed_m_s, 10.0), 16.01, 0.5):
        op = full_throttle(ps, float(vc), cond.field, 0.9)
        rc = (op.thrust_n - cond.drag(polar, cond.field, float(vc))) * vc / cond.weight_n
        if rc > best[0]:
            best = (rc, op, float(vc))
    rc, climb, _ = best
    t_climb = min(cond.survey_height_m / rc, req.survey.overhead_time_s) if rc > 0 else req.survey.overhead_time_s

    t_turn = plan.total_turn_length_m / v
    t_lines = plan.survey_time_s - t_turn
    t_rest = req.survey.overhead_time_s - t_climb
    e = (drawn(climb) * t_climb + drawn(cruise) * (t_lines + t_rest) + drawn(turn) * t_turn) / 3600
    return MissionEnergy(area_ha, polar.label, cond.mass_kg, plan.survey_time_s + req.survey.overhead_time_s, e,
                         ps.pack.usable_energy_wh, cruise.electrical_power_w, turn.electrical_power_w, t_climb)


def endurance_min(ps: PropulsionSet, cond: Conditions, polar: QuadraticPolar, reserve: float,
                  airspeed: float | None = None) -> float:
    """Minutes of level flight at ``airspeed`` (cruise by default) down to ``reserve``."""
    v = airspeed or cond.cruise_speed_m_s
    op = for_thrust(ps, cond.drag(polar, cond.survey, v), v, cond.survey, SOC_CRUISE)
    p = op.electrical_power_w + op.battery_current_a**2 * ps.pack.resistance_ohm + cond.avionics_w / BEC_EFFICIENCY
    return ps.pack.usable_energy_wh * (1 - reserve) / p * 60


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


@dataclass
class Selection:
    chosen: Performance
    candidates: list[Performance]  # every combination evaluated, feasible first
    energy_ok: dict[str, bool]


def pack_energy_checks(ps: PropulsionSet, cond: Conditions, req: MissionRequirements,
                       model_polar: QuadraticPolar) -> dict[str, tuple[bool, str]]:
    """The battery-size rules: MIS-4a on the model, 50 ha at planning drag, 25 ha at pessimistic."""
    s = req.survey
    e_model = mission_energy(ps, cond, req, model_polar, s.area_stretch_ha)
    need = req.limits.energy_design_margin * e_model.energy_wh / (1 - req.limits.battery_reserve)
    e_plan = mission_energy(ps, cond, req, model_polar.scaled_to(LD_PLANNING), s.area_stretch_ha)
    e_pess = mission_energy(ps, cond, req, model_polar.scaled_to(LD_PESSIMISTIC), s.area_min_ha)
    r = req.limits.battery_reserve
    return {
        "MIS-4a pack size": (ps.pack.usable_energy_wh >= need,
                             f"{ps.pack.usable_energy_wh:.1f} Wh usable vs {need:.1f} Wh "
                             f"({req.limits.energy_design_margin:g} x 50 ha on the model polar, with reserve)"),
        f"{s.area_stretch_ha:g} ha at L/D {LD_PLANNING:g}": (e_plan.reserve >= r, f"lands with {e_plan.reserve:.0%}"),
        f"{s.area_min_ha:g} ha at L/D {LD_PESSIMISTIC:g}": (e_pess.reserve >= r, f"lands with {e_pess.reserve:.0%}"),
    }


def select_propulsion(req: MissionRequirements, cond: Conditions, model_polar: QuadraticPolar,
                      motors=MOTORS, props=PROPS, packs=PACKS, escs=ESCS) -> Selection:
    """Cheapest set that passes every check and is within the cruise-power tolerance of the best."""
    results = []
    for m in motors:
        for p in props:
            for pk in packs:
                # smallest ESC that the current check allows
                perf = None
                for esc in escs:
                    perf = evaluate(PropulsionSet(m, p, pk, esc), cond, req, model_polar)
                    if perf.checks["ESC current"][0]:
                        break
                perf.checks.update(pack_energy_checks(perf.ps, cond, req, model_polar))
                results.append(perf)
    feasible = [r for r in results if r.feasible]
    if not feasible:
        raise RuntimeError("no motor/prop/pack combination in the catalogue passes every check")
    best_p = min(r.cruise.electrical_power_w for r in feasible)
    short = [r for r in feasible if r.cruise.electrical_power_w <= (1 + CRUISE_POWER_TOLERANCE) * best_p]
    chosen = min(short, key=lambda r: (r.ps.cost_usd, r.cruise.electrical_power_w))
    results.sort(key=lambda r: (not r.feasible, r.ps.cost_usd, r.cruise.electrical_power_w))
    return Selection(chosen, results, {k: v[0] for k, v in chosen.checks.items()})
