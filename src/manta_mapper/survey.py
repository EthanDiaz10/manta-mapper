"""Photogrammetry geometry: from a ground resolution to a flight plan.

The chain is short and every step is plain geometry:

1. GSD (m/px) and the camera's pixel pitch and focal length fix the altitude:
   ``h = GSD * f / pixel_pitch`` (similar triangles through the lens).
2. Altitude fixes the photo footprint on the ground: ``pixels * GSD`` per side.
3. Overlap turns the footprint into spacing: photos every
   ``along * (1 - forward_overlap)`` metres, flight lines every
   ``across * (1 - side_overlap)`` metres.
4. Area / line spacing gives the total flight-line length; ground speed / photo
   spacing gives the trigger rate the camera must sustain.

Wind matters because the camera sees *ground* speed: on the downwind leg the
aircraft covers ground faster, so photos must come faster. That leg sets the
trigger-rate limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .requirements import G0, Camera, MissionRequirements


def altitude_for_gsd(camera: Camera, gsd_m: float) -> float:
    """Height above ground (m) that gives ``gsd_m`` at the image centre."""
    return gsd_m * camera.focal_length_m / camera.pixel_pitch_m


def gsd_at_altitude(camera: Camera, altitude_m: float) -> float:
    return altitude_m * camera.pixel_pitch_m / camera.focal_length_m


def footprint(camera: Camera, altitude_m: float) -> tuple[float, float]:
    """Ground footprint (across_track_m, along_track_m) of one photo."""
    gsd = gsd_at_altitude(camera, altitude_m)
    long_side, short_side = (
        max(camera.width_px, camera.height_px) * gsd,
        min(camera.width_px, camera.height_px) * gsd,
    )
    if camera.long_side_across_track:
        return long_side, short_side
    return short_side, long_side


def turn_radius(airspeed_m_s: float, bank_deg: float) -> float:
    """Radius of a level coordinated turn: R = V^2 / (g tan(bank))."""
    return airspeed_m_s**2 / (G0 * math.tan(math.radians(bank_deg)))


def max_exposure_for_blur(groundspeed_m_s: float, gsd_m: float, max_blur_px: float) -> float:
    """Longest shutter time (s) that keeps forward-motion blur under ``max_blur_px``."""
    return max_blur_px * gsd_m / groundspeed_m_s


@dataclass(frozen=True)
class SurveyPlan:
    """Result of :func:`plan_survey`. Lengths in m, times in s."""

    area_ha: float
    gsd_requested_m: float
    gsd_achieved_m: float
    altitude_m: float
    altitude_limited: bool  # True if the GSD needed more height than allowed
    footprint_across_m: float
    footprint_along_m: float
    photo_spacing_m: float
    line_spacing_m: float
    n_lines: int
    line_length_m: float  # length of each line
    total_line_length_m: float
    turn_radius_m: float
    total_turn_length_m: float
    total_path_m: float
    n_photos: int
    airspeed_m_s: float
    wind_m_s: float
    max_groundspeed_m_s: float  # downwind leg
    trigger_interval_min_s: float  # shortest interval, on the downwind leg
    trigger_rate_max_hz: float
    trigger_ok: bool
    max_exposure_s: float  # blur limit on the downwind leg
    survey_time_s: float

    @property
    def ideal_line_length_m(self) -> float:
        """Area / line spacing: the lower bound on line length, ignoring edges."""
        return self.area_ha * 1e4 / self.line_spacing_m


def plan_survey(
    req: MissionRequirements,
    area_ha: float | None = None,
    gsd_m: float | None = None,
    airspeed_m_s: float | None = None,
    aspect_ratio: float = 1.0,
) -> SurveyPlan:
    """Lawn-mower survey of a rectangular block.

    The block is ``aspect_ratio`` times longer than it is wide, and lines run
    along its long side (fewer turns). One extra line is flown so the swath
    covers both edges. Turns use a U-turn when the line spacing is at least the
    turn diameter, otherwise a wider loop; both are approximated as
    ``pi * R + |spacing - 2R|``, which is close enough at the concept stage.
    """
    s, cam = req.survey, req.camera
    area_ha = s.area_min_ha if area_ha is None else area_ha
    gsd_req = s.gsd_target_m if gsd_m is None else gsd_m
    v = req.assumptions.cruise_speed_m_s if airspeed_m_s is None else airspeed_m_s
    wind = s.design_wind_m_s
    if v <= wind:
        raise ValueError(f"airspeed {v} m/s cannot penetrate the {wind} m/s design wind")

    h = altitude_for_gsd(cam, gsd_req)
    limited = h > s.max_altitude_agl_m
    if limited:
        h = s.max_altitude_agl_m
    gsd = gsd_at_altitude(cam, h)

    across, along = footprint(cam, h)
    photo_spacing = along * (1 - s.forward_overlap)
    line_spacing = across * (1 - s.side_overlap)

    area_m2 = area_ha * 1e4
    width = math.sqrt(area_m2 / aspect_ratio)
    length = width * aspect_ratio
    n_lines = math.ceil(width / line_spacing) + 1
    total_lines = n_lines * length

    r = turn_radius(v, s.turn_bank_deg)
    turn_len = math.pi * r + abs(line_spacing - 2 * r)
    total_turns = (n_lines - 1) * turn_len

    v_down, v_up = v + wind, v - wind
    trigger_interval = photo_spacing / v_down
    trigger_rate = 1 / trigger_interval
    photos_per_line = math.ceil(length / photo_spacing) + 1

    # Lines alternate upwind/downwind (worst case: wind along the lines).
    # Turns are flown at roughly airspeed on average.
    n_down = math.ceil(n_lines / 2)
    n_up = n_lines - n_down
    survey_time = length * (n_down / v_down + n_up / v_up) + total_turns / v

    return SurveyPlan(
        area_ha=area_ha,
        gsd_requested_m=gsd_req,
        gsd_achieved_m=gsd,
        altitude_m=h,
        altitude_limited=limited,
        footprint_across_m=across,
        footprint_along_m=along,
        photo_spacing_m=photo_spacing,
        line_spacing_m=line_spacing,
        n_lines=n_lines,
        line_length_m=length,
        total_line_length_m=total_lines,
        turn_radius_m=r,
        total_turn_length_m=total_turns,
        total_path_m=total_lines + total_turns,
        n_photos=n_lines * photos_per_line,
        airspeed_m_s=v,
        wind_m_s=wind,
        max_groundspeed_m_s=v_down,
        trigger_interval_min_s=trigger_interval,
        trigger_rate_max_hz=trigger_rate,
        trigger_ok=trigger_rate <= cam.max_trigger_rate_hz,
        max_exposure_s=max_exposure_for_blur(v_down, gsd, s.max_blur_px),
        survey_time_s=survey_time,
    )


def max_airspeed_for_trigger(req: MissionRequirements, gsd_m: float | None = None) -> float:
    """Highest airspeed at which the camera keeps up on the downwind leg."""
    s, cam = req.survey, req.camera
    gsd = s.gsd_target_m if gsd_m is None else gsd_m
    h = min(altitude_for_gsd(cam, gsd), s.max_altitude_agl_m)
    _, along = footprint(cam, h)
    return along * (1 - s.forward_overlap) * cam.max_trigger_rate_hz - s.design_wind_m_s
