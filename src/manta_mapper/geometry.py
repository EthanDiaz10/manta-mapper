"""Parametric BWB geometry (milestone M1.3).

One dataclass, :class:`BWBGeometry`, holds every number that defines the
shape. Everything downstream (VLM, VSPAERO, weights, CAD) reads the shape from
here, so there is one source of truth.

The shape, from the middle out (right half shown, the left is a mirror)::

        x (aft)
        ^
        |   pod  | blend |        outer panel (removable)        | tip fin
        |--------+-------+---------------------------------------+---------
     LE 0 ______                                                   drooped
        |      |\\___                                              down and
        | MH78 |    \\___ S5010 ____________________________       swept back
        |      |         |    spar (straight carbon tube)    \\___
        |      |         |                        [ elevon ]     |
     TE c0_____|_________|___________________________________/
             w_pod     y_root                               b/2   -> y (span)

* **Centre pod**: constant chord, unswept, MH 78 section. It is a box-like
  bay for the battery, Pi + camera and flight controller, easy to print.
* **Blend**: a short strake that tapers the deep pod down to the outer-panel
  root. The trailing edge stays straight (pod TE and outer-root TE line up),
  so the leading edge sweeps sharply here, which gives the BWB look. Airfoil
  morphs from MH 78 to S5010.
* **Outer panels**: straight-tapered, constant S5010 section, linear twist.
  Every line on them is straight, so they hot-wire from two templates and
  slide onto a straight carbon spar. They plug into the pod (AC-11).
* **Tip fins**: the tips droop down and sweep back to act as winglets and
  vertical fins (a tailless wing needs them for yaw stability). They are
  separate, replaceable parts (AC-11, AC-13).

Conventions: SI units, aircraft at the origin with x aft, y right, z up (the
AeroSandbox convention). The pod leading edge at the centreline is x = 0.
Twist is positive nose-up; ``washout_deg`` is positive *nose-down* at the tip.
Reference area is the projected planform area of the wing including the pod,
excluding the tip fins (the usual convention for a BWB).

What this module does NOT do yet: aerodynamics and stability (M1.4), masses
and CG (M1.5). :func:`vlm_preview` runs a quick vortex-lattice pass only to
show the geometry is analysable and to explain the baseline washout choice.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache

import numpy as np

from .atmosphere import isa
from .requirements import G0, MissionRequirements

# --------------------------------------------------------------------------
# Payload that must fit in the centre pod (AC-12). Rough boxes, mm.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PayloadItem:
    name: str
    length_m: float  # along x
    width_m: float  # along y
    height_m: float  # along z


DEFAULT_PAYLOAD: tuple[PayloadItem, ...] = (
    # 3S1P 21700 Li-ion, cells laid flat side by side, with wrap and leads (~54 Wh nominal)
    PayloadItem("battery 3S1P 21700", 0.075, 0.066, 0.023),
    PayloadItem("flight controller (wing FC)", 0.054, 0.036, 0.015),
    PayloadItem("Pi Zero 2 W", 0.065, 0.030, 0.012),
    # Camera Module 3 rotated 90 deg (portrait, MIS-3a) looking down through a window
    PayloadItem("Camera Module 3 + mount", 0.030, 0.030, 0.020),
    PayloadItem("ESC", 0.045, 0.025, 0.010),
    PayloadItem("receiver + Remote ID", 0.035, 0.025, 0.012),
    PayloadItem("GPS/compass (under top hatch)", 0.030, 0.030, 0.010),
)


# --------------------------------------------------------------------------
# Design variables
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BWBGeometry:
    """All the numbers that define the airframe shape.

    The first group are the design variables the optimizer (M1.6) will move.
    The rest are settings chosen once for buildability and kept fixed.
    """

    # --- overall ----------------------------------------------------------
    span_m: float = 1.50  # tip to tip, projected, wing only (fins droop outside this)

    # --- centre pod (MH 78) -----------------------------------------------
    pod_chord_m: float = 0.44  # chord of the pod, constant across its width
    pod_half_width_m: float = 0.10  # centreline to pod edge (pod is twice this wide)
    blend_width_m: float = 0.08  # spanwise length of the pod-to-panel strake

    # --- outer panels (S5010) ---------------------------------------------
    root_chord_m: float = 0.28  # outer-panel chord at the joint with the blend
    tip_chord_m: float = 0.16  # outer-panel chord at the tip (before the fin)
    sweep_le_deg: float = 25.0  # leading-edge sweep of the outer panel
    dihedral_deg: float = 3.0  # outer-panel dihedral (pod and blend are flat)

    # --- twist (positive = nose-up) ---------------------------------------
    pod_twist_deg: float = 0.0  # incidence of the pod relative to the body axis
    root_twist_deg: float = 0.0  # outer-panel root, relative to the body axis
    washout_deg: float = 5.0  # tip twisted this much nose-DOWN from the panel root

    # --- tip fins (drooped, swept winglets) -------------------------------
    fin_length_m: float = 0.065  # length along the fin, root to tip
    fin_droop_deg: float = 45.0  # angle below horizontal: 0 = flat, 90 = straight down
    fin_sweep_le_deg: float = 35.0  # leading-edge sweep of the fin
    fin_taper: float = 0.6  # fin tip chord / wing tip chord

    # --- control surfaces --------------------------------------------------
    elevon_chord_fraction: float = 0.25  # of local chord; generous for authority at low speed
    elevon_start: float = 0.15  # spanwise start, as a fraction of the outer panel span
    elevon_end: float = 0.90  # spanwise end, same measure (leaves a stiff tip for the fin joint)

    # --- structure and landing --------------------------------------------
    spar_chord_fraction: float = 0.25  # spar centre, fraction of local chord (outer panel)
    spar_diameter_m: float = 0.010  # carbon tube OD
    spar_min_cover_m: float = 0.001  # material needed above and below the tube
    skid_depth_m: float = 0.020  # sacrificial EPP skid under the pod (deep = soft landings)
    wall_m: float = 0.003  # pod wall thickness (for internal volume)

    # --- sections ----------------------------------------------------------
    pod_airfoil: str = "mh78"
    outer_airfoil: str = "s5010"
    fin_airfoil: str = "naca0009"

    name: str = "Manta Mapper"
    payload: tuple[PayloadItem, ...] = field(default=DEFAULT_PAYLOAD, repr=False)

    # Design variables the optimizer is expected to move (M1.6).
    DESIGN_VARIABLES = (
        "span_m", "pod_chord_m", "pod_half_width_m", "blend_width_m", "root_chord_m",
        "tip_chord_m", "sweep_le_deg", "dihedral_deg", "root_twist_deg", "washout_deg",
        "fin_length_m", "fin_droop_deg", "elevon_chord_fraction", "spar_chord_fraction",
    )

    def __post_init__(self):
        problems = self.validate()
        if problems:
            raise ValueError("invalid BWB geometry: " + "; ".join(problems))

    def validate(self) -> list[str]:
        """Return a list of things that make the shape impossible (empty = fine)."""
        p = []
        if self.outer_semispan_m <= 0.05:
            p.append(f"span {self.span_m:.3f} m leaves no outer panel past the blend")
        for n in ("pod_chord_m", "pod_half_width_m", "root_chord_m", "tip_chord_m",
                  "fin_length_m", "spar_diameter_m"):
            if getattr(self, n) <= 0:
                p.append(f"{n} must be positive")
        if self.blend_width_m < 0:
            p.append("blend_width_m must be >= 0")
        if self.root_chord_m > self.pod_chord_m:
            p.append("outer root chord must not exceed the pod chord")
        if self.tip_chord_m > self.root_chord_m:
            p.append("tip chord must not exceed root chord (reverse taper)")
        if not 0 <= self.fin_droop_deg <= 90:
            p.append("fin_droop_deg must be 0-90")
        if not 0 < self.elevon_chord_fraction < 0.5:
            p.append("elevon_chord_fraction must be in (0, 0.5)")
        if not 0 <= self.elevon_start < self.elevon_end <= 1:
            p.append("need 0 <= elevon_start < elevon_end <= 1")
        if not 0.1 <= self.spar_chord_fraction <= 1 - self.elevon_chord_fraction - 0.05:
            p.append("spar must sit between 10 % chord and the elevon hinge")
        if not -30 < self.sweep_le_deg < 50:
            p.append("sweep_le_deg out of range")
        return p

    # ------------------------------------------------------------------
    # Key stations (right half)
    # ------------------------------------------------------------------

    @property
    def semispan_m(self) -> float:
        return self.span_m / 2

    @property
    def root_y_m(self) -> float:
        """Spanwise station of the outer-panel joint."""
        return self.pod_half_width_m + self.blend_width_m

    @property
    def outer_semispan_m(self) -> float:
        return self.semispan_m - self.root_y_m

    @property
    def root_le_x_m(self) -> float:
        """Outer-root leading edge. Trailing edge lines up with the pod's."""
        return self.pod_chord_m - self.root_chord_m

    @property
    def tip_le_x_m(self) -> float:
        return self.root_le_x_m + self.outer_semispan_m * math.tan(math.radians(self.sweep_le_deg))

    @property
    def tip_z_m(self) -> float:
        return self.outer_semispan_m * math.tan(math.radians(self.dihedral_deg))

    @property
    def taper_ratio(self) -> float:
        """Outer-panel taper (tip chord / root chord)."""
        return self.tip_chord_m / self.root_chord_m

    @property
    def blend_sweep_le_deg(self) -> float:
        if self.blend_width_m == 0:
            return 90.0
        return math.degrees(math.atan2(self.root_le_x_m, self.blend_width_m))

    def outer_sweep_deg(self, chord_fraction: float) -> float:
        """Sweep of the line at ``chord_fraction`` on the outer panel (0.25 = quarter chord)."""
        dx = (self.tip_le_x_m + chord_fraction * self.tip_chord_m) - (
            self.root_le_x_m + chord_fraction * self.root_chord_m
        )
        return math.degrees(math.atan2(dx, self.outer_semispan_m))

    def twist_at(self, eta: float) -> float:
        """Twist on the outer panel at fraction ``eta`` (0 root, 1 tip) of its span.

        Linear, which is exactly what a hot-wire cut between two rotated templates gives.
        """
        return self.root_twist_deg - self.washout_deg * eta

    def _outer_station(self, eta: float) -> tuple[np.ndarray, float, float]:
        """(LE xyz, chord, twist) at fraction ``eta`` of the outer-panel span."""
        le = np.array([
            self.root_le_x_m + eta * (self.tip_le_x_m - self.root_le_x_m),
            self.root_y_m + eta * self.outer_semispan_m,
            eta * self.tip_z_m,
        ])
        chord = self.root_chord_m + eta * (self.tip_chord_m - self.root_chord_m)
        return le, chord, self.twist_at(eta)

    def stations(self) -> list[dict]:
        """Wing sections from the centreline to the tip (right half).

        Includes the elevon ends so the control surfaces line up with sections.
        Each dict has ``le`` (xyz), ``chord``, ``twist``, ``airfoil`` (name),
        ``blend`` (0 = pod section, 1 = outer section) and ``elevon`` (whether
        the elevon runs from this station to the next).
        """
        st = [
            dict(name="centreline", le=np.zeros(3), chord=self.pod_chord_m, twist=self.pod_twist_deg,
                 airfoil=self.pod_airfoil, blend=0.0, elevon=False),
            dict(name="pod edge", le=np.array([0.0, self.pod_half_width_m, 0.0]), chord=self.pod_chord_m,
                 twist=self.pod_twist_deg, airfoil=self.pod_airfoil, blend=0.0, elevon=False),
        ]
        etas = sorted({0.0, self.elevon_start, self.elevon_end, 1.0})
        for eta in etas:
            le, c, tw = self._outer_station(eta)
            name = {0.0: "outer root", 1.0: "tip"}.get(eta, f"elevon {'start' if eta == self.elevon_start else 'end'}")
            st.append(dict(name=name, le=le, chord=c, twist=tw, airfoil=self.outer_airfoil, blend=1.0,
                           elevon=self.elevon_start <= eta < self.elevon_end))
        if self.blend_width_m == 0:  # pod edge and outer root coincide in y: drop the pod edge
            st.pop(1)
        return st

    def fin_stations(self) -> list[dict]:
        tip_le, tip_c, _ = self._outer_station(1.0)
        d, s = math.radians(self.fin_droop_deg), math.radians(self.fin_sweep_le_deg)
        L = self.fin_length_m
        fin_tip = tip_le + np.array([L * math.tan(s), L * math.cos(d), -L * math.sin(d)])
        return [
            dict(name="fin root", le=tip_le, chord=tip_c, twist=0.0, airfoil=self.fin_airfoil),
            dict(name="fin tip", le=fin_tip, chord=tip_c * self.fin_taper, twist=0.0, airfoil=self.fin_airfoil),
        ]

    # ------------------------------------------------------------------
    # Reference quantities (pure numpy; no AeroSandbox needed)
    # ------------------------------------------------------------------

    def _segments(self):
        st = self.stations()
        for a, b in zip(st[:-1], st[1:]):
            yield a, b

    @property
    def reference_area_m2(self) -> float:
        """Projected planform area of the wing, both halves, pod included, fins excluded."""
        return 2 * sum((b["le"][1] - a["le"][1]) * (a["chord"] + b["chord"]) / 2 for a, b in self._segments())

    def _mac(self) -> tuple[float, float, float]:
        """(MAC, x of its leading edge, y of its station), exact for piecewise-linear chord and LE."""
        s_half = self.reference_area_m2 / 2
        int_c2 = int_cx = int_cy = 0.0
        for a, b in self._segments():
            h = b["le"][1] - a["le"][1]
            c0, c1 = a["chord"], b["chord"]
            x0, x1 = a["le"][0], b["le"][0]
            y0, y1 = a["le"][1], b["le"][1]
            # Simpson's rule is exact for these quadratics
            cm, xm, ym = (c0 + c1) / 2, (x0 + x1) / 2, (y0 + y1) / 2
            int_c2 += h / 6 * (c0**2 + 4 * cm**2 + c1**2)
            int_cx += h / 6 * (c0 * x0 + 4 * cm * xm + c1 * x1)
            int_cy += h / 6 * (c0 * y0 + 4 * cm * ym + c1 * y1)
        return int_c2 / s_half, int_cx / s_half, int_cy / s_half

    @property
    def mac_m(self) -> float:
        """Mean aerodynamic chord (the length static margin is quoted in)."""
        return self._mac()[0]

    @property
    def mac_le_x_m(self) -> float:
        return self._mac()[1]

    @property
    def mac_y_m(self) -> float:
        return self._mac()[2]

    @property
    def mac_quarter_chord_x_m(self) -> float:
        """Rough neutral-point location for a first CG guess (the VLM replaces it in M1.4)."""
        return self.mac_le_x_m + 0.25 * self.mac_m

    @property
    def aspect_ratio(self) -> float:
        return self.span_m**2 / self.reference_area_m2

    @property
    def fin_area_m2(self) -> float:
        """Both fins, true (not projected) area."""
        r, t = self.fin_stations()
        return 2 * self.fin_length_m * (r["chord"] + t["chord"]) / 2

    @property
    def fin_tip_xyz(self) -> np.ndarray:
        return self.fin_stations()[1]["le"]

    @property
    def overall_span_m(self) -> float:
        """Tip to tip including the fins' outward reach (for hangar and car)."""
        return 2 * self.fin_tip_xyz[1]

    @property
    def elevon_area_m2(self) -> float:
        """Both elevons."""
        _, c0, _ = self._outer_station(self.elevon_start)
        _, c1, _ = self._outer_station(self.elevon_end)
        h = (self.elevon_end - self.elevon_start) * self.outer_semispan_m
        return 2 * h * self.elevon_chord_fraction * (c0 + c1) / 2

    # ------------------------------------------------------------------
    # Section-dependent quantities (need airfoil coordinates)
    # ------------------------------------------------------------------

    def _section_area_coeff(self, station: dict) -> float:
        """Cross-section area / chord^2 for a station (blend interpolated)."""
        kp = airfoil_area_coeff(self.pod_airfoil)
        ko = airfoil_area_coeff(self.outer_airfoil)
        return kp + station["blend"] * (ko - kp)

    def _volume(self, a: dict, b: dict) -> float:
        """Enclosed volume between two stations, both halves not included."""
        h = b["le"][1] - a["le"][1]
        ka, kb = self._section_area_coeff(a), self._section_area_coeff(b)
        ca, cb = a["chord"], b["chord"]
        km, cm = (ka + kb) / 2, (ca + cb) / 2
        return h / 6 * (ka * ca**2 + 4 * km * cm**2 + kb * cb**2)

    def volumes_m3(self) -> dict[str, float]:
        """Enclosed volume of each part. Pod and blend: both sides; panel and fin: one each.

        The outer-panel volume times foam density is the panel's core mass (M1.5).
        """
        st = self.stations()
        pod = 2 * self._volume(st[0], st[1])
        if self.blend_width_m > 0:
            blend = 2 * self._volume(st[1], st[2])
            outer = st[2:]
        else:
            blend, outer = 0.0, st[1:]
        panel = sum(self._volume(a, b) for a, b in zip(outer[:-1], outer[1:]))
        r, t = self.fin_stations()
        kf = airfoil_area_coeff(self.fin_airfoil)
        fin = self.fin_length_m / 6 * kf * (r["chord"] ** 2 + 4 * ((r["chord"] + t["chord"]) / 2) ** 2 + t["chord"] ** 2)
        return {"pod": pod, "blend (both sides)": blend, "outer panel (each)": panel, "tip fin (each)": fin,
                "total": pod + blend + 2 * panel + 2 * fin}

    def spar_depth_m(self, eta: float) -> float:
        """Section depth at the spar line, fraction ``eta`` along the outer panel."""
        _, c, _ = self._outer_station(eta)
        return float(airfoil_local_thickness(self.outer_airfoil, self.spar_chord_fraction)) * c

    @property
    def spar_fits(self) -> bool:
        """The tube plus cover fits at the tip, the thinnest point (root is deeper)."""
        return self.spar_depth_m(1.0) >= self.spar_diameter_m + 2 * self.spar_min_cover_m

    @property
    def spar_sweep_deg(self) -> float:
        return self.outer_sweep_deg(self.spar_chord_fraction)

    @property
    def spar_length_m(self) -> float:
        """Length of one panel's spar inside the panel (add the socket depth into the pod)."""
        f = self.spar_chord_fraction
        r = self._outer_station(0.0)
        t = self._outer_station(1.0)
        p0 = r[0] + np.array([f * r[1], 0, 0])
        p1 = t[0] + np.array([f * t[1], 0, 0])
        return float(np.linalg.norm(p1 - p0))

    # --- pod bay and landing ---------------------------------------------

    def pod_depth_profile(self, n: int = 201) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, z_upper, z_lower) of the pod section at the centreline."""
        up, lo = airfoil_surfaces(self.pod_airfoil)
        xs = np.linspace(0, 1, n)
        zu, zl = np.interp(xs, *up), np.interp(xs, *lo)
        c = self.pod_chord_m
        return xs * c, zu * c, zl * c

    def pod_bay(self, min_internal_height_m: float | None = None) -> dict[str, float]:
        """Usable payload bay in the pod: the stretch of chord with enough inside height.

        ``min_internal_height_m`` defaults to the tallest payload item plus 10 mm.
        """
        tallest = max(i.height_m for i in self.payload)
        need = min_internal_height_m if min_internal_height_m is not None else tallest + 0.010
        x, zu, zl = self.pod_depth_profile()
        inside = (zu - zl) - 2 * self.wall_m
        ok = inside >= need
        if not ok.any():
            return dict(length_m=0.0, width_m=0.0, x_start_m=float("nan"), x_end_m=float("nan"),
                        max_internal_height_m=float(inside.max()), needed_height_m=need)
        idx = np.flatnonzero(ok)
        return dict(
            length_m=float(x[idx[-1]] - x[idx[0]]),
            width_m=2 * self.pod_half_width_m - 2 * self.wall_m,
            x_start_m=float(x[idx[0]]),
            x_end_m=float(x[idx[-1]]),
            max_internal_height_m=float(inside.max()),
            needed_height_m=need,
        )

    def payload_fit(self, packing_efficiency: float = 0.6) -> dict[str, float | bool]:
        """First check that the payload fits (AC-12); the CAD does the real packing.

        Floor area of all items / ``packing_efficiency`` must fit in the bay, every
        item must fit the bay height and width. 0.6 leaves room for wiring,
        mounts and access, so it is a forgiving check rather than a tight one.
        """
        bay = self.pod_bay()
        floor_needed = sum(i.length_m * i.width_m for i in self.payload) / packing_efficiency
        floor_have = bay["length_m"] * bay["width_m"]
        widest = max(min(i.width_m, i.length_m) for i in self.payload)
        return dict(
            floor_needed_m2=floor_needed,
            floor_available_m2=floor_have,
            fits=bool(floor_have >= floor_needed and widest <= bay["width_m"] and bay["length_m"] > 0),
        )

    @property
    def skid_bottom_z_m(self) -> float:
        _, _, zl = self.pod_depth_profile()
        return float(zl.min()) - self.skid_depth_m

    @property
    def fin_ground_clearance_m(self) -> float:
        """Height of the lowest fin point above the ground when sitting level on the skid.

        Positive means a level belly landing touches the skid first (AC-13).
        """
        r, t = self.fin_stations()
        half_t = airfoil_max_thickness(self.fin_airfoil) * t["chord"] / 2
        lowest = min(r["le"][2], t["le"][2] - half_t * math.cos(math.radians(self.fin_droop_deg)))
        return lowest - self.skid_bottom_z_m

    @property
    def roll_to_fin_strike_deg(self) -> float:
        """Bank angle on the ground (resting on the skid) at which a fin tip touches."""
        r, t = self.fin_stations()
        return math.degrees(math.atan2(self.fin_ground_clearance_m, t["le"][1]))

    # ------------------------------------------------------------------
    # AeroSandbox model
    # ------------------------------------------------------------------

    def to_airplane(self, xyz_ref: tuple[float, float, float] | None = None):
        """Build an :class:`aerosandbox.Airplane` (needs the ``aero`` extra).

        The wing and the tip fins are separate ``Wing`` objects so reference
        values come from the wing alone. Elevons are ``ControlSurface`` named
        "elevon" (symmetric=False, i.e. they deflect opposite for roll by
        default; set symmetric deflection for pitch in the analysis).
        """
        import aerosandbox as asb

        st = self.stations()
        xsecs = []
        for s in st:
            af = _blended_airfoil(self.pod_airfoil, self.outer_airfoil, s["blend"])
            cs = []
            if s["elevon"]:
                cs = [asb.ControlSurface(name="elevon", symmetric=False,
                                         hinge_point=1 - self.elevon_chord_fraction, trailing_edge=True)]
            xsecs.append(asb.WingXSec(xyz_le=list(s["le"]), chord=s["chord"], twist=s["twist"],
                                      airfoil=af, control_surfaces=cs))
        wing = asb.Wing(name="Main wing", symmetric=True, xsecs=xsecs)
        fin = asb.Wing(
            name="Tip fins", symmetric=True,
            xsecs=[asb.WingXSec(xyz_le=list(f["le"]), chord=f["chord"], twist=f["twist"],
                                airfoil=_load_airfoil(f["airfoil"])) for f in self.fin_stations()],
        )
        if xyz_ref is None:
            xyz_ref = (self.mac_quarter_chord_x_m, 0.0, 0.0)
        return asb.Airplane(
            name=self.name, xyz_ref=list(xyz_ref), wings=[wing, fin],
            s_ref=self.reference_area_m2, c_ref=self.mac_m, b_ref=self.span_m,
        )

    def export_openvsp(self, path: str) -> str:
        """Write an OpenVSP ``.vspscript`` that rebuilds this geometry (for M1.7 VSPAERO).

        Run it in OpenVSP via File > Run Script, or ``vspscript -script <file>``.
        """
        self.to_airplane().export_OpenVSP_vspscript(path)
        return path

    def export_avl(self, path: str) -> str:
        """Write an AVL input file (handy cross-check for the VLM)."""
        self.to_airplane().export_AVL(path)
        return path

    # ------------------------------------------------------------------

    def with_changes(self, **changes) -> "BWBGeometry":
        return replace(self, **changes)

    def design_vector(self) -> dict[str, float]:
        d = asdict(self)
        return {k: d[k] for k in self.DESIGN_VARIABLES}

    def summary(self) -> dict[str, float]:
        vol = self.volumes_m3()
        bay = self.pod_bay()
        fit = self.payload_fit()
        return {
            "span_m": self.span_m,
            "overall_span_m": self.overall_span_m,
            "reference_area_m2": self.reference_area_m2,
            "aspect_ratio": self.aspect_ratio,
            "mac_m": self.mac_m,
            "mac_le_x_m": self.mac_le_x_m,
            "mac_y_m": self.mac_y_m,
            "outer_taper": self.taper_ratio,
            "outer_sweep_c4_deg": self.outer_sweep_deg(0.25),
            "blend_sweep_le_deg": self.blend_sweep_le_deg,
            "fin_area_m2": self.fin_area_m2,
            "elevon_area_m2": self.elevon_area_m2,
            "volume_total_m3": vol["total"],
            "volume_pod_m3": vol["pod"],
            "volume_outer_panel_m3": vol["outer panel (each)"],
            "spar_depth_root_mm": self.spar_depth_m(0.0) * 1000,
            "spar_depth_tip_mm": self.spar_depth_m(1.0) * 1000,
            "spar_sweep_deg": self.spar_sweep_deg,
            "spar_length_m": self.spar_length_m,
            "bay_length_m": bay["length_m"],
            "bay_internal_height_m": bay["max_internal_height_m"],
            "payload_fits": float(fit["fits"]),
            "fin_ground_clearance_mm": self.fin_ground_clearance_m * 1000,
            "roll_to_fin_strike_deg": self.roll_to_fin_strike_deg,
        }


# --------------------------------------------------------------------------
# Airfoil helpers (cached; coordinates from the AeroSandbox / UIUC database)
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _load_airfoil(key: str):
    import aerosandbox as asb

    return asb.Airfoil(key).repanel(n_points_per_side=80)


@lru_cache(maxsize=None)
def _blended_airfoil(a: str, b: str, fraction: float):
    if fraction <= 0:
        return _load_airfoil(a)
    if fraction >= 1:
        return _load_airfoil(b)
    return _load_airfoil(a).blend_with_another_airfoil(_load_airfoil(b), blend_fraction=fraction)


@lru_cache(maxsize=None)
def airfoil_area_coeff(key: str) -> float:
    return float(_load_airfoil(key).area())


@lru_cache(maxsize=None)
def airfoil_max_thickness(key: str) -> float:
    return float(_load_airfoil(key).max_thickness())


def airfoil_local_thickness(key: str, x_over_c: float) -> float:
    return float(_load_airfoil(key).local_thickness(x_over_c))


@lru_cache(maxsize=None)
def airfoil_surfaces(key: str):
    """((x, z) upper, (x, z) lower), each sorted by x, chord-normalised."""
    af = _load_airfoil(key)
    up = af.upper_coordinates()[::-1]
    lo = af.lower_coordinates()
    return (up[:, 0], up[:, 1]), (lo[:, 0], lo[:, 1])


# --------------------------------------------------------------------------
# Baseline sized from the mission
# --------------------------------------------------------------------------

WING_CL_MAX = 0.95  # docs/airfoils.md section 7: S5010 1.17 x 0.9 x 0.9
AREA_MARGIN = 1.05  # 5 % more wing than the 9 m/s stall needs (forgiving: ~8.8 m/s)


def target_area_m2(req: MissionRequirements, mass_kg: float, wing_cl_max: float = WING_CL_MAX,
                   area_margin: float = AREA_MARGIN) -> float:
    """Wing area for the AC-1 stall speed at the field on the design day, plus a margin."""
    rho = isa(req.environment.field_elevation_m, req.environment.temperature_offset_K)[2]
    vs = req.limits.max_stall_speed_m_s
    return area_margin * mass_kg * G0 / (0.5 * rho * vs**2 * wing_cl_max)


def stall_speed_m_s(geom: BWBGeometry, req: MissionRequirements, mass_kg: float,
                    wing_cl_max: float = WING_CL_MAX) -> float:
    rho = isa(req.environment.field_elevation_m, req.environment.temperature_offset_K)[2]
    return math.sqrt(2 * mass_kg * G0 / (rho * geom.reference_area_m2 * wing_cl_max))


def baseline(req: MissionRequirements | None = None, mass_kg: float = 1.5,
             template: BWBGeometry | None = None, **kw) -> BWBGeometry:
    """Baseline design: the template shape with the span solved to hit the target area.

    The pod is sized by the payload, not by the wing, so only the span moves:
    area grows linearly with span at fixed chords, so this is one linear solve.
    """
    req = req or MissionRequirements()
    template = template or BWBGeometry()
    s_target = target_area_m2(req, mass_kg, **kw)
    g = template
    # Area = fixed (pod + blend) + outer semispan * (root + tip chord), both sides
    fixed = 2 * (g.pod_half_width_m * g.pod_chord_m + g.blend_width_m * (g.pod_chord_m + g.root_chord_m) / 2)
    outer_semispan = (s_target - fixed) / (g.root_chord_m + g.tip_chord_m)
    return replace(g, span_m=2 * (g.root_y_m + outer_semispan))


# --------------------------------------------------------------------------
# Quick-look vortex-lattice preview (the real aero and stability work is M1.4)
# --------------------------------------------------------------------------


@dataclass
class VLMPreview:
    """Result of :func:`vlm_preview`. Inviscid, thin-surface: compare shapes, don't quote numbers."""

    cl_alpha_per_deg: float
    alpha_zero_lift_deg: float
    cm0_c4: float  # zero-lift pitching moment about the MAC quarter chord
    neutral_point_x_m: float
    neutral_point_pct_mac: float
    cruise_cl: float
    cruise_alpha_deg: float
    cm_cg_cruise: float  # about a CG giving ``static_margin``; > 0 means it trims nose-up without elevon
    static_margin: float
    span_y_m: np.ndarray  # spanwise strip centres (right half)
    local_cl: dict[float, np.ndarray]  # wing CL -> local section CL at each strip


def vlm_preview(geom: BWBGeometry, req: MissionRequirements, mass_kg: float = 1.5,
                static_margin: float = 0.075, wing_cls=(0.42, WING_CL_MAX), n_strips: int = 15) -> VLMPreview:
    """Run AeroSandbox's VLM a few times to preview trim and spanwise loading.

    * Neutral point from the slope of Cm vs CL between two angles of attack.
    * Pitching moment at cruise about a CG placed ``static_margin`` ahead of it
      (AC-7 asks for 5-10 %; 7.5 % is the middle).
    * Local section CL across the span at each wing CL in ``wing_cls``. Compare
      it with each section's CL_max (S5010 ~1.17, MH 78 ~1.37 as built) to see
      where the stall starts. The tips must not be first.
    """
    import aerosandbox as asb

    env = req.environment
    alt = env.field_elevation_m + 100.0  # survey altitude, ~100 m AGL (MIS-2)
    rho = isa(alt, env.temperature_offset_K)[2]
    v = req.assumptions.cruise_speed_m_s
    atm = asb.Atmosphere(altitude=alt, temperature_deviation=env.temperature_offset_K)
    ap = geom.to_airplane()

    def solve(alpha):
        op = asb.OperatingPoint(atmosphere=atm, velocity=v, alpha=float(alpha))
        vlm = asb.VortexLatticeMethod(ap, op, spanwise_resolution=10, chordwise_resolution=8,
                                      spanwise_spacing_function=np.linspace)
        return vlm, vlm.run()

    _, r0 = solve(0.0)
    _, r1 = solve(4.0)
    cla = float(r1["CL"] - r0["CL"]) / 4.0
    cma = float(r1["Cm"] - r0["Cm"]) / 4.0
    a0 = -float(r0["CL"]) / cla
    cm0 = float(r0["Cm"]) + cma * a0
    x_np = geom.mac_quarter_chord_x_m - cma / cla * geom.mac_m
    x_cg = x_np - static_margin * geom.mac_m

    cruise_cl = mass_kg * G0 / (0.5 * rho * v**2 * geom.reference_area_m2)
    q = 0.5 * float(atm.density()) * v**2
    edges = np.linspace(0.0, geom.semispan_m, n_strips + 1)
    local: dict[float, np.ndarray] = {}
    cm_cg = float("nan")
    cruise_alpha = float("nan")
    for cl in sorted(set(wing_cls) | {cruise_cl}):
        alpha = a0 + cl / cla
        vlm, res = solve(alpha)
        f = np.asarray(vlm.forces_geometry)
        c = np.asarray(vlm.vortex_centers)
        area = np.asarray(vlm.areas)
        nz = np.abs(np.asarray(vlm.normal_directions)[:, 2])
        on_wing = (c[:, 1] > 0) & (nz > 0.5)  # right half, near-horizontal panels (excludes fins)
        strips = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = on_wing & (c[:, 1] >= lo) & (c[:, 1] < hi)
            strips.append(f[m, 2].sum() / (q * area[m].sum()) if m.any() else np.nan)
        if cl in wing_cls:
            local[cl] = np.array(strips)
        if cl == cruise_cl:
            cruise_alpha = alpha
            cm_cg = float(res["Cm"]) + float(res["CL"]) * (x_cg - geom.mac_quarter_chord_x_m) / geom.mac_m
    return VLMPreview(
        cl_alpha_per_deg=cla, alpha_zero_lift_deg=a0, cm0_c4=cm0, neutral_point_x_m=x_np,
        neutral_point_pct_mac=(x_np - geom.mac_le_x_m) / geom.mac_m * 100,
        cruise_cl=cruise_cl, cruise_alpha_deg=cruise_alpha, cm_cg_cruise=cm_cg, static_margin=static_margin,
        span_y_m=(edges[:-1] + edges[1:]) / 2, local_cl=local,
    )


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _outline(geom: BWBGeometry):
    st = geom.stations()
    y = np.array([s["le"][1] for s in st])
    le = np.array([s["le"][0] for s in st])
    te = le + np.array([s["chord"] for s in st])
    return y, le, te


def plot_views(geom: BWBGeometry, path: str | None = None):  # pragma: no cover - visual
    """Top view (planform with spar, elevons, bay, MAC) and front view (droop, skid, ground)."""
    import matplotlib.pyplot as plt

    fig, (ax, axf) = plt.subplots(2, 1, figsize=(10, 6.6), gridspec_kw=dict(height_ratios=[3.2, 1]))
    y, le, te = _outline(geom)
    fins = geom.fin_stations()
    for sgn in (1, -1):
        ax.fill(np.r_[sgn * y, sgn * y[::-1]], np.r_[le, te[::-1]], color="#cfd8e3", ec="#33475b", lw=1.2)
        # fin (top view projection)
        fy = [sgn * f["le"][1] for f in fins]
        fl = [f["le"][0] for f in fins]
        ft = [f["le"][0] + f["chord"] for f in fins]
        ax.fill(np.r_[fy, fy[::-1]], np.r_[fl, ft[::-1]], color="#e8c9a0", ec="#8a5a20", lw=1)
        # joints
        for yy in (geom.pod_half_width_m, geom.root_y_m):
            ax.plot([sgn * yy] * 2, [np.interp(yy, y, le), np.interp(yy, y, te)], color="#33475b", lw=0.8, ls=":")
        # spar
        f = geom.spar_chord_fraction
        r, t = geom._outer_station(0.0), geom._outer_station(1.0)
        ax.plot([sgn * r[0][1], sgn * t[0][1]], [r[0][0] + f * r[1], t[0][0] + f * t[1]], color="k", lw=2.2,
                label="carbon spar" if sgn == 1 else None)
        # elevon
        e0, e1 = geom._outer_station(geom.elevon_start), geom._outer_station(geom.elevon_end)
        h = 1 - geom.elevon_chord_fraction
        ax.fill([sgn * e0[0][1], sgn * e1[0][1], sgn * e1[0][1], sgn * e0[0][1]],
                [e0[0][0] + h * e0[1], e1[0][0] + h * e1[1], e1[0][0] + e1[1], e0[0][0] + e0[1]],
                color="#7fb3d5", ec="#1f618d", lw=1, label="elevon" if sgn == 1 else None)
    bay = geom.pod_bay()
    w = bay["width_m"] / 2
    ax.add_patch(plt.Rectangle((-w, bay["x_start_m"]), 2 * w, bay["length_m"], fill=False, ec="#c0392b",
                               ls="--", lw=1.2, label=f"payload bay ({bay['length_m']*1000:.0f} mm long)"))
    ax.plot([geom.mac_y_m] * 2, [geom.mac_le_x_m, geom.mac_le_x_m + geom.mac_m], color="#27ae60", lw=3,
            label=f"MAC {geom.mac_m*1000:.0f} mm")
    ax.plot(0, geom.mac_quarter_chord_x_m, "o", color="#27ae60", ms=6, label="MAC quarter chord")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("y, span (m)")
    ax.set_ylabel("x, aft (m)")
    ax.set_title(f"{geom.name}: span {geom.span_m:.2f} m, area {geom.reference_area_m2:.3f} m², "
                 f"AR {geom.aspect_ratio:.1f}, outer sweep {geom.sweep_le_deg:g}° LE, washout {geom.washout_deg:g}°")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # Front view (looking from behind, z up)
    st = geom.stations()
    sy = np.array([s["le"][1] for s in st])
    sz = np.array([s["le"][2] for s in st])
    tmax = [airfoil_max_thickness(geom.pod_airfoil) if s["blend"] == 0 else airfoil_max_thickness(geom.outer_airfoil)
            for s in st]
    half = np.array(tmax) * np.array([s["chord"] for s in st]) / 2
    x, zu, zl = geom.pod_depth_profile()
    for sgn in (1, -1):
        axf.fill(np.r_[sgn * sy, sgn * sy[::-1]], np.r_[sz + half, (sz - half)[::-1]],
                 color="#cfd8e3", ec="#33475b")
        axf.plot([sgn * f["le"][1] for f in fins], [f["le"][2] for f in fins], color="#8a5a20", lw=3)
    axf.fill_between([-geom.pod_half_width_m, geom.pod_half_width_m], zl.min(), zu.max(), color="#cfd8e3",
                     ec="#33475b")
    axf.fill_between([-0.04, 0.04], geom.skid_bottom_z_m, zl.min(), color="#f5b041", label="EPP skid")
    axf.axhline(geom.skid_bottom_z_m, color="#7f8c8d", lw=1, ls="--", label="ground (level on skid)")
    axf.set_aspect("equal")
    axf.set_xlabel("y (m)")
    axf.set_ylabel("z (m)")
    axf.set_title(f"Front view: dihedral {geom.dihedral_deg:g}°, fins droop {geom.fin_droop_deg:g}°, "
                  f"fin tips {geom.fin_ground_clearance_m*1000:.0f} mm above ground when level", fontsize=10)
    axf.legend(loc="upper center", fontsize=8, ncol=2)
    axf.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_3d(geom: BWBGeometry, path: str | None = None):  # pragma: no cover - visual
    """Shaded three-view plus isometric from AeroSandbox."""
    import matplotlib.pyplot as plt

    geom.to_airplane().draw_three_view(show=False)
    fig = plt.gcf()
    fig.suptitle(geom.name)
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_spanwise(geom: BWBGeometry, pv: VLMPreview, path: str | None = None):  # pragma: no cover - visual
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for cl, loc in pv.local_cl.items():
        ax.plot(pv.span_y_m, loc, "o-", ms=3, label=f"local CL at wing CL {cl:.2f}")
    yy = np.linspace(0, geom.semispan_m, 200)
    ax.plot(yy, np.where(yy <= geom.pod_half_width_m, 1.37,
                         np.where(yy >= geom.root_y_m, 1.17,
                                  1.37 + (yy - geom.pod_half_width_m) / max(geom.blend_width_m, 1e-9) * (1.17 - 1.37))),
            "k--", label="section CL_max as built (MH 78 / S5010)")
    for yy0, lab in ((geom.pod_half_width_m, "pod edge"), (geom.root_y_m, "panel joint")):
        ax.axvline(yy0, color="grey", lw=0.8, ls=":")
        ax.text(yy0, 1.48, lab, rotation=90, fontsize=8, va="top", ha="right", color="grey")
    ax.set_xlabel("spanwise station y (m)")
    ax.set_ylabel("section lift coefficient")
    ax.set_ylim(0, 1.5)
    ax.set_xlim(0, geom.semispan_m * 1.02)
    ax.set_title(f"VLM preview, washout {geom.washout_deg:g}°: where would the stall start?")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


# --------------------------------------------------------------------------
# Report / CLI
# --------------------------------------------------------------------------


def report(geom: BWBGeometry, req: MissionRequirements, mass_kg: float, pv: VLMPreview | None = None) -> str:
    lines: list[str] = []
    out = lines.append
    out(f"== {geom.name} baseline geometry ({mass_kg:g} kg) ==")
    out(f"Span {geom.span_m:.3f} m (overall with fins {geom.overall_span_m:.3f} m), "
        f"area {geom.reference_area_m2:.4f} m^2 (target {target_area_m2(req, mass_kg):.4f}), AR {geom.aspect_ratio:.2f}")
    out(f"MAC {geom.mac_m*1000:.0f} mm, its LE at x = {geom.mac_le_x_m*1000:.0f} mm, y = {geom.mac_y_m*1000:.0f} mm")
    out(f"Pod {geom.pod_chord_m*1000:.0f} mm chord x {2*geom.pod_half_width_m*1000:.0f} mm wide; "
        f"blend LE sweep {geom.blend_sweep_le_deg:.0f} deg")
    out(f"Outer panel {geom.outer_semispan_m*1000:.0f} mm long, chords {geom.root_chord_m*1000:.0f} -> "
        f"{geom.tip_chord_m*1000:.0f} mm (taper {geom.taper_ratio:.2f}), sweep LE {geom.sweep_le_deg:g} / "
        f"c/4 {geom.outer_sweep_deg(0.25):.1f} deg, washout {geom.washout_deg:g} deg")
    out(f"Stall at the field on the design day, wing CL_max {WING_CL_MAX}: "
        f"{stall_speed_m_s(geom, req, mass_kg):.2f} m/s (AC-1 limit {req.limits.max_stall_speed_m_s:g})")
    sp = geom.spar_depth_m
    out(f"Spar {geom.spar_diameter_m*1000:.0f} mm at {geom.spar_chord_fraction:.0%} chord: depth "
        f"{sp(0)*1000:.1f} mm root, {sp(1)*1000:.1f} mm tip ({'fits' if geom.spar_fits else 'DOES NOT FIT'}); "
        f"length {geom.spar_length_m*1000:.0f} mm per panel, swept {geom.spar_sweep_deg:.1f} deg")
    bay, fit = geom.pod_bay(), geom.payload_fit()
    out(f"Payload bay {bay['length_m']*1000:.0f} x {bay['width_m']*1000:.0f} mm, up to "
        f"{bay['max_internal_height_m']*1000:.0f} mm inside height; floor needed "
        f"{fit['floor_needed_m2']*1e4:.0f} cm^2 of {fit['floor_available_m2']*1e4:.0f} -> "
        f"{'fits' if fit['fits'] else 'DOES NOT FIT'}")
    vol = geom.volumes_m3()
    out("Volumes: " + ", ".join(f"{k} {v*1e6:.0f} cm^3" for k, v in vol.items()))
    out(f"Fins {geom.fin_area_m2*1e4:.0f} cm^2 (both), elevons {geom.elevon_area_m2*1e4:.0f} cm^2 (both); "
        f"fin tips {geom.fin_ground_clearance_m*1000:.0f} mm above ground when level on the skid "
        f"(touch at {geom.roll_to_fin_strike_deg:.1f} deg bank)")
    if pv is not None:
        out("")
        out("== VLM preview (inviscid; the real check is M1.4) ==")
        out(f"CL_alpha {pv.cl_alpha_per_deg:.4f}/deg, zero-lift alpha {pv.alpha_zero_lift_deg:.2f} deg, "
            f"Cm0 about MAC c/4 {pv.cm0_c4:+.4f}")
        out(f"Neutral point {pv.neutral_point_pct_mac:.1f} % MAC (x = {pv.neutral_point_x_m*1000:.0f} mm)")
        out(f"Cruise CL {pv.cruise_cl:.2f} at alpha {pv.cruise_alpha_deg:.1f} deg; Cm about a CG at "
            f"{pv.static_margin:.1%} static margin {pv.cm_cg_cruise:+.4f} "
            f"({'self-trims' if abs(pv.cm_cg_cruise) < 0.01 else 'needs elevon trim'})")
        for cl, loc in pv.local_cl.items():
            i = int(np.nanargmax(loc))
            out(f"At wing CL {cl:.2f}: peak local CL {loc[i]:.2f} at y = {pv.span_y_m[i]*1000:.0f} mm; "
                f"tip strip {loc[-1]:.2f}, centre {loc[0]:.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(prog="manta_mapper.geometry", description="Build and report the baseline BWB geometry.")
    p.add_argument("mission", nargs="?", help="mission TOML (defaults built in)")
    p.add_argument("--mass", type=float, default=1.5, help="take-off mass, kg")
    p.add_argument("--plots", help="directory to save geometry plots into")
    p.add_argument("--vsp", help="write an OpenVSP .vspscript to this path")
    p.add_argument("--avl", help="write an AVL file to this path")
    p.add_argument("--no-vlm", action="store_true", help="skip the VLM preview")
    a = p.parse_args(argv)
    req = MissionRequirements.from_toml(a.mission) if a.mission else MissionRequirements()
    geom = baseline(req, a.mass)
    pv = None if a.no_vlm else vlm_preview(geom, req, a.mass)
    print(report(geom, req, a.mass, pv))
    if a.vsp:
        print(f"Wrote {geom.export_openvsp(a.vsp)}")
    if a.avl:
        print(f"Wrote {geom.export_avl(a.avl)}")
    if a.plots:
        import os

        import matplotlib

        matplotlib.use("Agg")
        os.makedirs(a.plots, exist_ok=True)
        files = [plot_views(geom, os.path.join(a.plots, "geometry_views.png")) and "geometry_views.png",
                 plot_3d(geom, os.path.join(a.plots, "geometry_3d.png")) and "geometry_3d.png"]
        if pv is not None:
            files.append(plot_spanwise(geom, pv, os.path.join(a.plots, "geometry_spanwise_cl.png"))
                         and "geometry_spanwise_cl.png")
        print("Saved " + ", ".join(os.path.join(a.plots, f) for f in files))


if __name__ == "__main__":
    main()
