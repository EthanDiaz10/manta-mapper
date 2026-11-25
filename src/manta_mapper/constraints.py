"""Constraint analysis: which wing loadings and power loadings can fly the mission.

This is the first sizing step in almost every aircraft design textbook
(Raymer ch. 5, Gudmundsson ch. 3, Sadraey ch. 4). Each requirement becomes a
curve on a chart of

* **wing loading** W/S (N/m^2): weight per unit wing area, and
* **power loading** P/W (W/N): electrical power per unit weight.

Two kinds of requirement appear:

* *Wing-loading limits* (vertical lines). Stall speed and the hand-launch cap
  only care about how much wing carries each newton, so they set a maximum W/S.
* *Power requirements* (curves). Cruise, climb, launch climb-out and turning
  each need a minimum P/W that depends on W/S. Too little wing (high W/S) means
  flying at high lift coefficient with lots of induced drag; too much wing
  (low W/S) means a lot of skin-friction drag.

Every point below the limits and above all power curves is a feasible design.
For a mission like ours the best point is usually *at* the wing-loading limit:
the smallest wing that still stalls slowly enough, which is also the lightest
and cheapest. The power curves then say how much motor and battery it needs.

Drag model: parabolic polar ``CD = CD0 + k CL^2`` with ``k = 1 / (pi AR e)``.
Its numbers come from :class:`~manta_mapper.requirements.Assumptions` until the
aero modules replace them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .atmosphere import density
from .requirements import G0, MissionRequirements
from .survey import SurveyPlan, max_airspeed_for_trigger, plan_survey


def induced_drag_factor(aspect_ratio: float, oswald_e: float) -> float:
    return 1.0 / (math.pi * aspect_ratio * oswald_e)


def stall_speed(wing_loading_n_m2, rho: float, cl_max: float):
    """V_s = sqrt(2 (W/S) / (rho CL_max)). Works on scalars or arrays."""
    return np.sqrt(2.0 * np.asarray(wing_loading_n_m2) / (rho * cl_max))


def _power_level(ws, v, rho, cd0, k, n=1.0):
    """Shaft P/W (W/N) for level flight at speed v and load factor n."""
    q = 0.5 * rho * v**2
    return v * (q * cd0 / ws + k * n**2 * ws / q)


@dataclass(frozen=True)
class Densities:
    field: float  # stall and launch happen here
    cruise: float  # survey altitude above the field


@dataclass(frozen=True)
class ConstraintResult:
    """Constraint curves on a common W/S grid plus the selected design point."""

    wing_loading: np.ndarray  # N/m^2
    curves: dict[str, np.ndarray]  # name -> required electrical P/W (W/N)
    limits: dict[str, float]  # name -> maximum W/S (N/m^2)
    design_wing_loading: float  # N/m^2
    design_power_loading: float  # W/N, electrical
    driving_limit: str
    driving_curve: str
    densities: Densities

    @property
    def design_wing_loading_kg_m2(self) -> float:
        return self.design_wing_loading / G0

    def required_power_loading(self, ws: float) -> tuple[float, str]:
        """Max over all power curves at ``ws`` and the name of the one that drives it."""
        values = {name: float(np.interp(ws, self.wing_loading, c)) for name, c in self.curves.items()}
        name = max(values, key=values.get)
        return values[name], name


def densities(req: MissionRequirements, survey_altitude_m: float) -> Densities:
    env = req.environment
    return Densities(
        field=density(env.field_elevation_m, env.temperature_offset_K),
        cruise=density(env.field_elevation_m + survey_altitude_m, env.temperature_offset_K),
    )


def constraint_analysis(
    req: MissionRequirements,
    survey_altitude_m: float | None = None,
    ws_range_n_m2: tuple[float, float] = (10.0, 120.0),
    n_points: int = 400,
) -> ConstraintResult:
    a, lim = req.assumptions, req.limits
    if survey_altitude_m is None:
        survey_altitude_m = plan_survey(req).altitude_m
    rho = densities(req, survey_altitude_m)
    k = induced_drag_factor(a.aspect_ratio, a.oswald_e)
    eta_cruise = a.eta_prop_cruise * a.eta_motor_esc
    eta_launch = a.eta_prop_launch * a.eta_motor_esc
    ws = np.linspace(*ws_range_n_m2, n_points)

    # --- wing-loading limits (max W/S) -----------------------------------
    limits = {
        "stall": 0.5 * rho.field * lim.max_stall_speed_m_s**2 * a.cl_max,
        "hand-launch cap": lim.max_wing_loading_kg_m2 * G0,
        # Keep cruise at least 1.3 x stall speed for gust and turn margin.
        "cruise >= 1.3 Vs": 0.5 * rho.cruise * (a.cruise_speed_m_s / 1.3) ** 2 * a.cl_max,
    }

    # --- power requirements (min P/W) --------------------------------------
    v = a.cruise_speed_m_s
    cruise = _power_level(ws, v, rho.cruise, a.cd0, k) / eta_cruise

    n_turn = 1.0 / math.cos(math.radians(req.survey.turn_bank_deg))
    turn = _power_level(ws, v, rho.cruise, a.cd0, k, n=n_turn) / eta_cruise

    # Climb at the minimum-power speed, but never slower than 1.2 Vs.
    v_mp = np.sqrt(2 * ws / rho.field) * (k / (3 * a.cd0)) ** 0.25
    v_climb = np.maximum(v_mp, 1.2 * stall_speed(ws, rho.field, a.cl_max))
    climb = (lim.min_climb_rate_m_s + _power_level(ws, v_climb, rho.field, a.cd0, k)) / eta_cruise

    # Hand-launch climb-out: just after release at 1.2 Vs (CL = CL_max / 1.44),
    # the aircraft must climb at the required gradient. P = V (D + W sin(gamma)) / eta.
    v_launch = 1.2 * stall_speed(ws, rho.field, a.cl_max)
    cl_launch = a.cl_max / 1.44
    cd_launch = a.cd0 + k * cl_launch**2
    launch = v_launch * (lim.launch_climb_gradient + cd_launch / cl_launch) / eta_launch

    curves = {"cruise": cruise, "turn": turn, "climb": climb, "launch climb-out": launch}

    driving_limit = min(limits, key=limits.get)
    ws_design = limits[driving_limit]
    at_design = {name: float(np.interp(ws_design, ws, c)) for name, c in curves.items()}
    driving_curve = max(at_design, key=at_design.get)
    return ConstraintResult(
        wing_loading=ws,
        curves=curves,
        limits=limits,
        design_wing_loading=ws_design,
        design_power_loading=at_design[driving_curve],
        driving_limit=driving_limit,
        driving_curve=driving_curve,
        densities=rho,
    )


@dataclass(frozen=True)
class SizingSummary:
    """First-cut sizing for a given take-off mass at the design point."""

    mass_kg: float
    wing_area_m2: float
    span_m: float
    mean_chord_m: float
    stall_speed_m_s: float
    cruise_cl: float
    cruise_l_over_d: float
    cruise_power_elec_w: float  # propulsion only
    peak_power_elec_w: float  # max over all power curves (motor sizing)
    airspeed_window_m_s: tuple[float, float]  # (1.3 Vs, camera trigger limit with margin)
    min_window_m_s: float  # AC-9
    survey: SurveyPlan
    mission_time_s: float
    energy_used_wh: float  # energy THIS mission draws from the pack
    energy_required_wh: float  # usable energy THIS mission needs incl. reserve
    battery_energy_wh: float  # usable energy the pack must hold (stretch area, reserve, margin)
    battery_mass_kg: float

    @property
    def airspeed_window_width_m_s(self) -> float:
        lo, hi = self.airspeed_window_m_s
        return hi - lo

    @property
    def airspeed_window_ok(self) -> bool:
        lo, hi = self.airspeed_window_m_s
        in_window = lo <= self.survey.airspeed_m_s <= hi
        return in_window and self.airspeed_window_width_m_s >= self.min_window_m_s

    @property
    def reserve_on_landing(self) -> float:
        """Fraction of the pack's usable energy left after THIS mission."""
        return 1 - self.energy_used_wh / self.battery_energy_wh


def size_aircraft(
    req: MissionRequirements,
    mass_kg: float,
    area_ha: float | None = None,
    result: ConstraintResult | None = None,
) -> SizingSummary:
    """Turn the constraint-chart design point into wing area, power and battery.

    Mission energy covers propulsion plus avionics for the survey and the fixed
    overhead (launch, climb, transit, landing). The pack is sized for the
    stretch area with the reserve still in it on landing, times a design margin.
    """
    a, lim = req.assumptions, req.limits
    plan = plan_survey(req, area_ha=area_ha)
    result = result or constraint_analysis(req, survey_altitude_m=plan.altitude_m)
    rho = result.densities

    w = mass_kg * G0
    ws = result.design_wing_loading
    s = w / ws
    span = math.sqrt(a.aspect_ratio * s)
    vs = float(stall_speed(ws, rho.field, a.cl_max))

    k = induced_drag_factor(a.aspect_ratio, a.oswald_e)
    v = a.cruise_speed_m_s
    q = 0.5 * rho.cruise * v**2
    cl = ws / q
    cd = a.cd0 + k * cl**2
    p_cruise = w * v * cd / cl / (a.eta_prop_cruise * a.eta_motor_esc)
    p_peak = w * result.design_power_loading

    def mission_energy(p: SurveyPlan) -> tuple[float, float]:
        t = p.survey_time_s + req.survey.overhead_time_s
        return t, (p_cruise + a.avionics_power_w) * t / 3600

    t_mission, energy_used_wh = mission_energy(plan)
    energy_required_wh = energy_used_wh / (1 - lim.battery_reserve)

    # The pack is sized for the stretch area, then given a design margin, so
    # the minimum mission lands with far more than the required reserve.
    stretch = plan_survey(req, area_ha=max(plan.area_ha, req.survey.area_stretch_ha))
    _, stretch_wh = mission_energy(stretch)
    battery_energy_wh = stretch_wh / (1 - lim.battery_reserve) * lim.energy_design_margin
    battery_mass = battery_energy_wh / (a.battery_specific_energy_wh_kg * a.battery_usable_fraction)

    return SizingSummary(
        mass_kg=mass_kg,
        wing_area_m2=s,
        span_m=span,
        mean_chord_m=s / span,
        stall_speed_m_s=vs,
        cruise_cl=cl,
        cruise_l_over_d=cl / cd,
        cruise_power_elec_w=p_cruise,
        peak_power_elec_w=p_peak,
        airspeed_window_m_s=(1.3 * vs, max_airspeed_for_trigger(req, plan.gsd_achieved_m)),
        min_window_m_s=lim.min_airspeed_window_m_s,
        survey=plan,
        mission_time_s=t_mission,
        energy_used_wh=energy_used_wh,
        energy_required_wh=energy_required_wh,
        battery_energy_wh=battery_energy_wh,
        battery_mass_kg=battery_mass,
    )


def plot(result: ConstraintResult, ax=None):  # pragma: no cover - visual
    """Draw the constraint diagram. Needs the optional ``plot`` extra (matplotlib)."""
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(7, 5))[1]
    ws_kg = result.wing_loading / G0
    for name, c in result.curves.items():
        ax.plot(ws_kg, c, label=f"{name} (min P/W)")
    for (name, ws_max), ls in zip(result.limits.items(), ["--", "-.", ":"]):
        ax.axvline(ws_max / G0, ls=ls, lw=1.2, label=f"{name} (max W/S)", color="0.3")
    envelope = np.max(np.vstack(list(result.curves.values())), axis=0)
    feasible = result.wing_loading <= min(result.limits.values())
    ax.fill_between(ws_kg, envelope, envelope.max() * 1.5, where=feasible, alpha=0.15, label="feasible")
    ax.plot(result.design_wing_loading_kg_m2, result.design_power_loading, "k*", ms=14, label="design point")
    ax.set_xlabel("Wing loading W/S (kg/m$^2$)")
    ax.set_ylabel("Electrical power loading P/W (W/N)")
    ax.set_ylim(0, float(np.percentile(envelope, 90)) * 1.3)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax
