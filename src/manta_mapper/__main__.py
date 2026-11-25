"""Print a first-cut sizing report: ``python -m manta_mapper [mission.toml] [--mass KG] [--plot FILE]``."""

from __future__ import annotations

import argparse

from .constraints import constraint_analysis, plot, size_aircraft
from .requirements import G0, MissionRequirements


def report(req: MissionRequirements, mass_kg: float) -> str:
    lines = []
    out = lines.append
    for label, area in (("minimum", req.survey.area_min_ha), ("stretch", req.survey.area_stretch_ha)):
        sz = size_aircraft(req, mass_kg, area_ha=area)
        sv = sz.survey
        if label == "minimum":
            res = constraint_analysis(req, survey_altitude_m=sv.altitude_m)
            out("== Survey geometry ==")
            out(f"GSD {sv.gsd_achieved_m*100:.2f} cm/px at {sv.altitude_m:.1f} m AGL"
                + ("  (ALTITUDE-LIMITED)" if sv.altitude_limited else ""))
            out(f"Footprint {sv.footprint_across_m:.1f} m across x {sv.footprint_along_m:.1f} m along")
            out(f"Photo every {sv.photo_spacing_m:.1f} m, lines every {sv.line_spacing_m:.1f} m")
            out(f"Downwind ground speed {sv.max_groundspeed_m_s:.1f} m/s -> trigger every "
                f"{sv.trigger_interval_min_s:.2f} s ({sv.trigger_rate_max_hz:.2f} Hz, "
                f"{'OK' if sv.trigger_ok else 'TOO FAST'}: limit is {req.camera.trigger_rate_margin:.0%} "
                f"of the {req.camera.max_trigger_rate_hz:g} Hz camera)")
            out(f"Shutter must be <= 1/{1/sv.max_exposure_s:.0f} s for {req.survey.max_blur_px} px blur")
            out("")
            out("== Constraint analysis ==")
            for name, ws in res.limits.items():
                out(f"  max W/S from {name:<18} {ws/G0:5.2f} kg/m^2")
            out(f"Design point W/S = {res.design_wing_loading_kg_m2:.2f} kg/m^2 (set by {res.driving_limit}), "
                f"P/W = {res.design_power_loading:.2f} W/N (set by {res.driving_curve})")
            out("")
            out(f"== Sizing at {mass_kg:.2f} kg take-off mass ==")
            out(f"Wing area {sz.wing_area_m2:.3f} m^2, span {sz.span_m:.2f} m (AR {req.assumptions.aspect_ratio:g}), "
                f"mean chord {sz.mean_chord_m*1000:.0f} mm")
            out(f"Stall {sz.stall_speed_m_s:.1f} m/s, cruise CL {sz.cruise_cl:.2f}, L/D {sz.cruise_l_over_d:.1f}")
            lo, hi = sz.airspeed_window_m_s
            out(f"Airspeed window {lo:.1f}-{hi:.1f} m/s ({hi - lo:.1f} m/s wide, need >= {sz.min_window_m_s:g}), "
                f"cruise {sv.airspeed_m_s:.1f} m/s {'OK' if sz.airspeed_window_ok else 'NOT OK'}")
            out(f"Battery: >= {sz.battery_energy_wh:.0f} Wh usable (~{sz.battery_mass_kg*1000:.0f} g of cells), "
                f"sized for {req.survey.area_stretch_ha:g} ha x {req.limits.energy_design_margin:g} margin")
            out(f"Cruise power {sz.cruise_power_elec_w:.0f} W, peak (motor sizing) {sz.peak_power_elec_w:.0f} W")
            out("")
            out("== Missions ==")
        out(f"{label:>8} {area:4.0f} ha: {sv.n_lines} lines, {sv.total_path_m/1000:.1f} km, {sv.n_photos} photos, "
            f"{sz.mission_time_s/60:.1f} min, uses {sz.energy_used_wh:.1f} Wh, "
            f"lands with {sz.reserve_on_landing:.0%} left (need {req.limits.battery_reserve:.0%})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="manta_mapper", description=__doc__)
    p.add_argument("mission", nargs="?", help="mission TOML (defaults built in)")
    p.add_argument("--mass", type=float, default=1.5, help="take-off mass guess, kg")
    p.add_argument("--plot", help="save the constraint diagram to this image file")
    args = p.parse_args(argv)
    req = MissionRequirements.from_toml(args.mission) if args.mission else MissionRequirements()
    print(report(req, args.mass))
    if args.plot:
        import matplotlib.pyplot as plt

        plot(constraint_analysis(req))
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"\nSaved {args.plot}")


if __name__ == "__main__":
    main()
