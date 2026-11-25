"""International Standard Atmosphere (troposphere only, which is all we need)."""

from __future__ import annotations

T0 = 288.15  # K
P0 = 101_325.0  # Pa
LAPSE = 0.0065  # K/m
R_AIR = 287.053  # J/(kg K)
G0 = 9.80665


def isa(altitude_m: float, temperature_offset_K: float = 0.0) -> tuple[float, float, float]:
    """Return (temperature K, pressure Pa, density kg/m^3) at a geometric altitude.

    ``temperature_offset_K`` shifts temperature at constant pressure ("ISA + dT"),
    which is how hot-day performance is normally quoted.
    """
    if not -500 <= altitude_m <= 11_000:
        raise ValueError("isa() covers the troposphere only (-500 to 11000 m)")
    t_std = T0 - LAPSE * altitude_m
    p = P0 * (t_std / T0) ** (G0 / (LAPSE * R_AIR))
    t = t_std + temperature_offset_K
    rho = p / (R_AIR * t)
    return t, p, rho


def density(altitude_m: float, temperature_offset_K: float = 0.0) -> float:
    return isa(altitude_m, temperature_offset_K)[2]


def dynamic_viscosity(temperature_K: float) -> float:
    """Air dynamic viscosity (Pa s) from Sutherland's law."""
    mu_ref, t_ref, s = 1.716e-5, 273.15, 110.4
    return mu_ref * (temperature_K / t_ref) ** 1.5 * (t_ref + s) / (temperature_K + s)


def speed_of_sound(temperature_K: float) -> float:
    return (1.4 * R_AIR * temperature_K) ** 0.5


def density_altitude(rho: float) -> float:
    """Altitude in the standard atmosphere that has density ``rho``."""
    exponent = G0 / (LAPSE * R_AIR) - 1
    rho0 = P0 / (R_AIR * T0)
    return (T0 / LAPSE) * (1 - (rho / rho0) ** (1 / exponent))


__all__ = ["isa", "density", "density_altitude", "dynamic_viscosity", "speed_of_sound"]
