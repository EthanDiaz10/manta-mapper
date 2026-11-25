"""Mission requirements as data.

Every number the design pipeline is sized against lives here, so a change to
the mission is a change to one file (``config/mission.toml``) rather than a
hunt through the code. The IDs in the comments match ``docs/requirements.md``.

Units are SI throughout (m, s, kg, N, W, Pa) unless a field name says otherwise
(``_ha``, ``_kg_m2``, ``_deg``).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

G0 = 9.80665  # m/s^2
FT_TO_M = 0.3048


@dataclass(frozen=True)
class Camera:
    """Raspberry Pi Camera Module 3 (Sony IMX708), standard (non-wide) lens.

    Defaults from the Raspberry Pi camera datasheet: 4608 x 2592 px,
    1.4 um pixels, 4.74 mm focal length.
    """

    name: str = "Pi Camera Module 3 (standard)"
    width_px: int = 4608
    height_px: int = 2592
    pixel_pitch_m: float = 1.4e-6
    focal_length_m: float = 4.74e-3
    # Long side of the sensor across the flight track: widest swath, so the
    # fewest flight lines. Along-track spacing is then set by the short side.
    long_side_across_track: bool = True
    max_trigger_rate_hz: float = 1.0  # CAM-1: what the Pi Zero 2 W must sustain
    min_exposure_s: float = 1 / 4000


@dataclass(frozen=True)
class Survey:
    """Photogrammetry survey settings (MIS-*, CAM-*)."""

    area_min_ha: float = 25.0  # MIS-1 minimum
    area_stretch_ha: float = 50.0  # MIS-1 stretch
    gsd_target_m: float = 0.03  # MIS-2
    gsd_fallback_m: float = 0.04  # MIS-2 fallback
    forward_overlap: float = 0.75  # MIS-3 (WebODM guidance: 70-80 %)
    side_overlap: float = 0.65  # MIS-3 (WebODM guidance: 60-70 %)
    max_altitude_agl_m: float = 400 * FT_TO_M  # REG-2: 121.9 m
    max_blur_px: float = 0.5  # CAM-2
    design_wind_m_s: float = 5.0  # MIS-5: survey must work in this steady wind
    turn_bank_deg: float = 30.0  # bank angle used for line-to-line turns
    # Non-survey flight time: launch, climb, transit, loiter, approach (s)
    overhead_time_s: float = 240.0  # MIS-4


@dataclass(frozen=True)
class Environment:
    """Atmosphere at the flying field. Set ``field_elevation_m`` to your site."""

    field_elevation_m: float = 0.0
    temperature_offset_K: float = 15.0  # ISA + 15 K: a hot summer day is the sizing case


@dataclass(frozen=True)
class AircraftLimits:
    """Hard limits from launch, landing and regulations (AC-*)."""

    max_stall_speed_m_s: float = 9.0  # AC-1 (hand launch)
    max_wing_loading_kg_m2: float = 6.0  # AC-2 (hand launch)
    max_takeoff_mass_kg: float = 2.0  # AC-3
    battery_reserve: float = 0.30  # MIS-4: fraction of usable energy left on landing
    min_climb_rate_m_s: float = 2.5  # AC-5
    launch_climb_gradient: float = 0.15  # AC-6: ~8.5 deg climb straight off the hand


@dataclass(frozen=True)
class Assumptions:
    """First-guess aerodynamics, propulsion and battery, used only until the pipeline
    replaces them with computed values (NeuralFoil / AeroSandbox / VSPAERO).

    Values are typical for a small printed tailless flying wing with a reflexed
    airfoil; each one is a deliberate, documented guess.
    """

    cl_max: float = 1.0  # reflex airfoils give up CL_max for pitch stability
    cd0: float = 0.025  # printed surface, pod, skid, exposed hardware
    aspect_ratio: float = 6.0
    oswald_e: float = 0.85
    eta_prop_cruise: float = 0.65
    eta_prop_launch: float = 0.45  # prop is far off-design at low speed / high thrust
    eta_motor_esc: float = 0.80
    cruise_speed_m_s: float = 14.0  # first guess; checked against the speed window
    avionics_power_w: float = 5.0  # FC, GPS, receiver, Remote ID, Pi Zero 2 W, servos
    battery_specific_energy_wh_kg: float = 180.0  # Li-ion 21700 pack incl. wiring/wrap
    battery_usable_fraction: float = 0.85  # voltage sag and cut-off leave some unusable


@dataclass(frozen=True)
class MissionRequirements:
    survey: Survey = field(default_factory=Survey)
    camera: Camera = field(default_factory=Camera)
    environment: Environment = field(default_factory=Environment)
    limits: AircraftLimits = field(default_factory=AircraftLimits)
    assumptions: Assumptions = field(default_factory=Assumptions)

    @classmethod
    def from_toml(cls, path: str | Path) -> "MissionRequirements":
        """Load requirements, overriding defaults with any keys in the file.

        Unknown sections or keys raise ``KeyError`` so typos don't silently
        fall back to defaults.
        """
        data = tomllib.loads(Path(path).read_text())
        base = cls()
        updates = {}
        for section, values in data.items():
            if section not in {f.name for f in fields(cls)}:
                raise KeyError(f"unknown section [{section}] in {path}")
            current = getattr(base, section)
            allowed = {f.name for f in fields(current)}
            unknown = set(values) - allowed
            if unknown:
                raise KeyError(f"unknown keys in [{section}]: {sorted(unknown)}")
            updates[section] = replace(current, **values)
        return replace(base, **updates)
