"""Component weight build-up, CG placement and inertia (milestone M1.5).

Every part of the aircraft is an :class:`Item` with a mass, a position and a
price. Structure masses come from the M1.3 geometry (foam volume, printed
shell area, spar length) times a material; bought parts come from a parts
list with datasheet masses. Then:

* the battery, the heaviest part that can slide, is placed so the CG lands
  on the target (261 mm from the pod nose, the 10 % static-margin end of the
  M1.4 band, for the first flights);
* every item is checked to fit the pod's inside height where it sits;
* the moments of inertia are summed from the real parts (distributed parts
  such as panels, shell and spar are split into strips), replacing the
  placeholder used in M1.4.

**Growth allowance.** Hobby builds almost always come out heavier than the
spreadsheet. Each item carries an allowance by how well its mass is known:
5 % for bought parts with datasheet masses, 20 % for structure estimated from
geometry, 30 % for wiring, glue and hardware. "As built" below means with the
allowance; it is the number the scale should be compared with (VAL-4).

**Build choices** (:class:`BuildChoices`, defaults are the recommendation):

* outer panels: hot-wired **EPP** foam (1.9 lb/ft^3, 30 kg/m^3) covered in
  laminating film, sliding onto a carbon tube. EPP bends and springs back in a
  cartwheel where EPS or XPS would crack, which is what AC-13 asks for.
* centre pod: printed in **LW-ASA** (foamed ASA). ASA keeps its shape up to
  ~95 C; PLA and LW-PLA soften around 55-60 C, which a car in a Las Cruces
  summer passes easily. Replaceable printed TPU nose cap.
* tip fins and skid: EPP, held on with tape/velcro so they are throwaway parts.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace

import numpy as np

from .geometry import BWBGeometry, airfoil_area_coeff, airfoil_surfaces
from .propulsion import PropulsionSet

# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Material:
    name: str
    density_kg_m3: float  # as used (foamed filaments: as printed)
    cost_usd_per_kg: float
    softens_c: float  # rough heat-deflection / softening temperature
    note: str = ""


MATERIALS = {
    # foams (hot-wire cut)
    "EPP 1.9 lb": Material("EPP foam 1.9 lb/ft3", 30, 25, 100, "springs back after impacts; fuzzy hot-wire cut"),
    "EPP 1.3 lb": Material("EPP foam 1.3 lb/ft3", 21, 25, 100, "lighter, floppier"),
    "EPS 1.5 lb": Material("EPS bead foam 1.5 lb/ft3", 24, 8, 80, "cheap, clean cut; dents and cracks"),
    "XPS": Material("XPS (Foamular 150)", 28, 10, 75, "stiff, clean cut; snaps in a cartwheel"),
    # filaments
    "LW-PLA": Material("LW-PLA (foamed)", 600, 60, 55, "light, brittle, softens in a hot car"),
    "LW-ASA": Material("LW-ASA (foamed)", 700, 70, 95, "light, tougher, heat- and UV-proof; needs an enclosure"),
    "ASA": Material("ASA", 1070, 22, 95, "tough, heat- and UV-proof; needs an enclosure"),
    "PETG": Material("PETG", 1270, 18, 75, "tough, cheap; heavier, creeps when hot"),
    "TPU": Material("TPU 95A", 1210, 28, 80, "rubbery; nose cap and bumpers"),
}

CARBON_DENSITY = 1550  # pultruded carbon tube / rod, kg/m^3
LAMINATE_KG_M2 = 0.045  # 1.5 mil laminating film incl. adhesive
CARBON_TUBE_USD_PER_M = 8.0
CARBON_ROD_USD_PER_M = 3.0

# Growth allowances by how well the mass is known
GROWTH = {"bought": 0.05, "structure": 0.20, "wiring": 0.30}


@dataclass(frozen=True)
class SparPiece:
    """One tube of a telescoping spar (M2.2). Stations are fractions of the panel's
    spar length from the panel root; negative runs into the pod sleeve."""

    od_m: float
    id_m: float
    eta0: float
    eta1: float
    density: float = CARBON_DENSITY
    usd_per_m: float = CARBON_TUBE_USD_PER_M
    kind: str = "pultruded"


#: M2.2 recommendation (structures.py, sized on the 1.84 m optimum, spar 770 mm): telescoping pultruded
#: tubes 12x10 (from 100 mm inside the pod sleeve to 303 mm) > 10x8 (253-467 mm) > 8x6 (417 mm to the tip),
#: 50 mm bonded overlaps. Stations are fractions of the panel spar so the optimizer can scale them.
M22_SPAR = (
    SparPiece(0.012, 0.010, -0.1299, 0.3936, CARBON_DENSITY, 8.0, "pultruded"),
    SparPiece(0.010, 0.008, 0.3287, 0.6066, CARBON_DENSITY, 6.0, "pultruded"),
    SparPiece(0.008, 0.006, 0.5417, 1.0, CARBON_DENSITY, 5.0, "pultruded"),
)


@dataclass(frozen=True)
class BuildChoices:
    panel_foam: str = "EPP 1.9 lb"
    pod_material: str = "LW-ASA"
    pod_wall_m: float = 0.0008  # two perimeters with a 0.4 mm nozzle
    #: bulkheads, spar sockets, hatch frame, battery tray, camera window frame,
    #: motor boss, panel-root plates, as a fraction of the pod shell mass
    pod_internal_fraction: float = 0.6
    spar_od_m: float | None = None  # None: use the geometry's spar diameter
    spar_wall_m: float = 0.001  # 10 x 8 mm tube
    spar_socket_m: float = 0.10  # tube length inside the pod socket
    #: M2.2 telescoping spar (structures.py). None: one tube of spar_od_m x spar_wall_m, full length (M1.5).
    spar_layout: tuple[SparPiece, ...] | None = M22_SPAR
    rear_pin_d_m: float = 0.003  # anti-rotation carbon pin at ~60 % chord
    rear_pin_len_m: float = 0.15
    cg_target_x_m: float = 0.261  # M1.4: 10 % static margin end of the band (first flights)


# --------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------


@dataclass
class Item:
    """One line of the weight statement.

    ``points`` is an (n, 4) array of [mass, x, y, z] lumps (both sides included
    for mirrored parts); a single bought part is one lump. ``box`` is the part's
    own size, for its self-inertia (lumps of distributed parts use ``spread``).
    """

    name: str
    group: str  # structure / propulsion / avionics / camera / wiring
    points: np.ndarray
    cost_usd: float
    known: str = "bought"  # key into GROWTH
    box: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spread: np.ndarray | None = None  # per-lump (sx, sy, sz) extents for strips
    power_w: float = 0.0  # average electrical draw (avionics)
    note: str = ""
    movable: bool = False

    @property
    def mass_kg(self) -> float:
        return float(self.points[:, 0].sum())

    @property
    def mass_as_built_kg(self) -> float:
        return self.mass_kg * (1 + GROWTH[self.known])

    @property
    def cg(self) -> np.ndarray:
        m = self.points[:, 0]
        return (m[:, None] * self.points[:, 1:]).sum(0) / m.sum()

    def moved_to(self, x: float) -> "Item":
        p = self.points.copy()
        p[:, 1] += x - self.cg[0]
        return replace(self, points=p)


def _lump(mass: float, x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.array([[mass, x, y, z]])


def _mirror(mass_each: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([[mass_each, x, y, z], [mass_each, x, -y, z]])


@dataclass(frozen=True)
class Part:
    """A bought part: datasheet mass, rough price, size and average power."""

    name: str
    mass_kg: float
    cost_usd: float
    size_m: tuple[float, float, float]  # length (x), width (y), height (z)
    power_w: float = 0.0
    group: str = "avionics"


AVIONICS = {
    "fc": Part("Flight controller: SpeedyBee F405 WING or Matek F405-WTE (ArduPilot; PDB, current sensor, BECs)",
               0.025, 50, (0.054, 0.036, 0.015), 0.8),
    "gps": Part("GPS + compass: Matek M10Q-5883", 0.009, 30, (0.022, 0.022, 0.009), 0.4),
    "airspeed": Part("Airspeed: Matek ASPD-4525 + pitot tube, silicone line", 0.012, 45, (0.032, 0.022, 0.012), 0.05),
    "rx": Part("ELRS 2.4 GHz receiver with telemetry: RadioMaster RP2", 0.002, 18, (0.018, 0.011, 0.005), 0.4),
    "rid": Part("Remote ID broadcast module (must be on the FAA DoC list)", 0.012, 45, (0.035, 0.025, 0.012), 0.3),
    "bec": Part("5 V 3 A buck BEC for the Pi (keeps camera noise off the FC rail)", 0.006, 6, (0.025, 0.015, 0.008)),
    "servo": Part("Elevon servo: Emax ES08MD II, 12 g digital metal-gear (each)", 0.012, 10, (0.023, 0.012, 0.024), 0.6),
    "linkage": Part("Servo horn, pushrod, clevis (each side)", 0.004, 2.5, (0.04, 0.01, 0.01)),
}
CAMERA = {
    "pi": Part("Raspberry Pi Zero 2 W + heatsink", 0.013, 17, (0.065, 0.030, 0.012), 2.0, "camera"),
    "sd": Part("32 GB high-endurance microSD", 0.001, 9, (0.015, 0.011, 0.001), 0.0, "camera"),
    "cam": Part("Camera Module 3 (standard lens) + 15 cm Zero ribbon", 0.006, 28, (0.025, 0.024, 0.012), 0.0, "camera"),
    "mount": Part("Printed camera mount on a 6 deg wedge + window", 0.008, 0, (0.035, 0.035, 0.012), 0.0, "camera"),
}


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _perimeter_coeff(key: str) -> float:
    (xu, zu), (xl, zl) = airfoil_surfaces(key)
    return float(np.sum(np.hypot(np.diff(xu), np.diff(zu))) + np.sum(np.hypot(np.diff(xl), np.diff(zl))))


def _strips(geom: BWBGeometry, y0: float, y1: float, n: int):
    """Strip centres (y), widths, LE x, chord, z and blend fraction between y0 and y1 (right half)."""
    st = geom.stations()
    sy = np.array([s["le"][1] for s in st])
    edges = np.linspace(y0, y1, n + 1)
    y = 0.5 * (edges[1:] + edges[:-1])
    w = np.diff(edges)
    le = np.interp(y, sy, [s["le"][0] for s in st])
    c = np.interp(y, sy, [s["chord"] for s in st])
    z = np.interp(y, sy, [s["le"][2] for s in st])
    blend = np.interp(y, sy, [s["blend"] for s in st])
    return y, w, le, c, z, blend


def pod_inside_height(geom: BWBGeometry, x: float) -> tuple[float, float]:
    """(floor z, inside height) of the pod at station x (centreline, inside the walls)."""
    xs, zu, zl = geom.pod_depth_profile()
    lo = float(np.interp(x, xs, zl)) + geom.wall_m
    hi = float(np.interp(x, xs, zu)) - geom.wall_m
    return lo, hi - lo


# --------------------------------------------------------------------------
# The build-up
# --------------------------------------------------------------------------


def structure_items(geom: BWBGeometry, ch: BuildChoices, n: int = 40) -> list[Item]:
    items = []
    foam = MATERIALS[ch.panel_foam]
    shell_mat = MATERIALS[ch.pod_material]
    k_outer = airfoil_area_coeff(geom.outer_airfoil)
    p_outer = _perimeter_coeff(geom.outer_airfoil)
    p_pod = _perimeter_coeff(geom.pod_airfoil)
    c0 = geom.pod_chord_m

    # --- outer panels: foam core + laminating film, both sides ---------------
    y, w, le, c, z, _ = _strips(geom, geom.root_y_m, geom.semispan_m, n)
    m_core = foam.density_kg_m3 * k_outer * c**2 * w
    m_skin = LAMINATE_KG_M2 * p_outer * c * w
    for label, m, xc, known, cost in (
        (f"Outer panels: {foam.name} cores (hot-wired, elevons cut free)", m_core, le + 0.42 * c, "structure",
         2 * float(m_core.sum()) * foam.cost_usd_per_kg + 20),
        ("Outer panels: laminating film, hinge and leading-edge tape", m_skin, le + 0.5 * c, "structure", 20),
    ):
        pts = np.vstack([np.column_stack([m, xc, y, z]), np.column_stack([m, xc, -y, z])])
        spread = np.tile(np.column_stack([c, w, 0.03 * c]), (2, 1))
        items.append(Item(label, "structure", pts, cost, known, spread=spread))

    # elevon stiffeners: 3 x 1 mm carbon strip along each elevon
    eta0, eta1 = geom.elevon_start, geom.elevon_end
    (le0, c_e0, _), (le1, c_e1, _) = geom._outer_station(eta0), geom._outer_station(eta1)
    span_e = (eta1 - eta0) * geom.outer_semispan_m
    xm = 0.5 * (le0[0] + le1[0]) + (1 - geom.elevon_chord_fraction / 2) * 0.5 * (c_e0 + c_e1)
    items.append(Item("Elevon stiffeners: 3 x 1 mm carbon strip", "structure",
                      _mirror(3e-6 * span_e * CARBON_DENSITY, xm, 0.5 * (le0[1] + le1[1]), 0.5 * (le0[2] + le1[2])),
                      2 * span_e * CARBON_ROD_USD_PER_M, "structure"))

    # --- spar tubes (one per panel, plugging into pod sockets) ---------------
    f = geom.spar_chord_fraction
    r_le, r_c, _ = geom._outer_station(0.0)
    t_le, t_c, _ = geom._outer_station(1.0)
    p0 = r_le + np.array([f * r_c, 0, 0])
    p1 = t_le + np.array([f * t_c, 0, 0])
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    Lp = geom.spar_length_m
    if ch.spar_layout is None:
        od = ch.spar_od_m or geom.spar_diameter_m
        pieces = (SparPiece(od, od - 2 * ch.spar_wall_m, -ch.spar_socket_m / Lp, 1.0),)
    else:
        pieces = ch.spar_layout
    sp_pts, cost, names = [], 0.0, []
    for pc in pieces:
        lin = math.pi / 4 * (pc.od_m**2 - pc.id_m**2) * pc.density  # kg/m
        a, b = pc.eta0 * Lp, pc.eta1 * Lp
        s = a + (np.arange(10) + 0.5) / 10 * (b - a)
        pts = p0 + s[:, None] * d
        m_seg = lin * (b - a) / 10
        sp_pts += [np.column_stack([np.full(10, m_seg), pts]), np.column_stack([np.full(10, m_seg), pts * [1, -1, 1]])]
        sticks = 2 * (b - a) if ch.spar_layout is None else math.ceil(2 * (b - a) - 1e-6)  # M2.2: whole 1 m sticks
        cost += pc.usd_per_m * sticks
        names.append(f"{pc.od_m*1000:.0f}x{pc.id_m*1000:.0f}" + ("" if ch.spar_layout is None else f" {pc.kind}"))
    L = (pieces[0].eta1 - min(pc.eta0 for pc in pieces)) * Lp
    label = (f"Spar: {names[0]} mm carbon tube, 2 x {L*1000:.0f} mm" if len(pieces) == 1 and ch.spar_layout is None
             else "Spar: telescoping carbon tubes " + " > ".join(names) + " mm, both panels")
    items.append(Item(label, "structure", np.vstack(sp_pts), cost, "bought"))
    if ch.spar_layout is not None:
        # M2.2 joint: roll-wrapped sleeve per side (ID = root tube OD) and carbon strip caps across the pod
        root = pieces[0]
        sl_len = ch.spar_socket_m + 0.02
        m_sleeve = math.pi / 4 * ((root.od_m + 0.002) ** 2 - root.od_m**2) * 1500 * sl_len
        x_in = p0 - d * ch.spar_socket_m
        cap_len = 2 * x_in[1] + 0.06
        m_caps = 2 * 0.010 * 0.001 * cap_len * CARBON_DENSITY  # one in the floor, one in the roof
        pts = np.vstack([_mirror(m_sleeve, *(p0 - d * sl_len / 2)), _lump(m_caps, x_in[0], 0, 0)])
        items.append(Item("Spar carry-through: 2 roll-wrapped sleeves + 2 carbon strip caps (10x1 mm)", "structure",
                          pts, 16.0 + 2 * cap_len * CARBON_ROD_USD_PER_M + 5, "bought"))
    pin = math.pi / 4 * ch.rear_pin_d_m**2 * CARBON_DENSITY * ch.rear_pin_len_m
    items.append(Item("Anti-rotation pins: 3 mm carbon rod at 60 % chord", "structure",
                      _mirror(pin, geom.root_le_x_m + 0.6 * geom.root_chord_m, geom.root_y_m, 0.0),
                      2 * ch.rear_pin_len_m * CARBON_ROD_USD_PER_M, "bought"))

    # --- centre pod + blend: printed shell + internal structure --------------
    yp, wp, lep, cp, zp, bl = _strips(geom, 0.0, geom.root_y_m, 12)
    per = p_pod + bl * (p_outer - p_pod)
    m_shell = shell_mat.density_kg_m3 * ch.pod_wall_m * per * cp * wp
    m_shell = m_shell * (1 + ch.pod_internal_fraction)
    pts = np.vstack([np.column_stack([m_shell, lep + 0.45 * cp, yp, zp + 0.005]),
                     np.column_stack([m_shell, lep + 0.45 * cp, -yp, zp + 0.005])])
    spread = np.tile(np.column_stack([cp, wp, 0.04 * np.ones_like(cp)]), (2, 1))
    m_pod = 2 * float(m_shell.sum())
    items.append(Item(f"Centre pod: {shell_mat.name} shell {ch.pod_wall_m*1000:.1f} mm + internal frames "
                      f"(+{ch.pod_internal_fraction:.0%}), hatch, spar sockets", "structure", pts,
                      m_pod * shell_mat.cost_usd_per_kg, "structure", spread=spread))
    tpu = MATERIALS["TPU"]
    items.append(Item("Replaceable nose cap: TPU 95A, 15 % gyroid", "structure", _lump(0.016, 0.012, 0, 0.005),
                      0.05 * tpu.cost_usd_per_kg, "structure", box=(0.025, 0.12, 0.03)))
    items.append(Item("Motor mount plate + screws (printed ASA, in the pod trailing edge)", "structure",
                      _lump(0.010, c0 - 0.008, 0, 0.0), 1, "structure", box=(0.016, 0.04, 0.03)))

    # --- tip fins and skid: EPP, throwaway parts -------------------------------
    fr, ft = geom.fin_stations()
    kf = airfoil_area_coeff(geom.fin_airfoil)
    v_fin = geom.volumes_m3()["tip fin (each)"]
    epp = MATERIALS["EPP 1.9 lb"]
    m_fin = v_fin * epp.density_kg_m3 + LAMINATE_KG_M2 * 2.05 * geom.fin_area_m2 / 2 + 0.002  # + pegs/glue
    fc = 0.5 * (fr["le"] + ft["le"]) + np.array([0.45 * 0.5 * (fr["chord"] + ft["chord"]), 0, 0])
    items.append(Item("Tip fins: EPP, laminated, pegged and taped on (replaceable)", "structure",
                      _mirror(m_fin, fc[0], fc[1], fc[2]), 2, "structure"))
    del kf
    skid_l, skid_w = 0.6 * c0, 0.06
    m_skid = skid_l * skid_w * geom.skid_depth_m * epp.density_kg_m3 + 0.005 + 0.004  # + HDPE wear strip + velcro
    _, _, zl = geom.pod_depth_profile()
    items.append(Item("Belly skid: EPP block + HDPE wear strip, velcro (replaceable)", "structure",
                      _lump(m_skid, 0.45 * c0, 0, geom.skid_bottom_z_m + geom.skid_depth_m / 2), 4, "structure",
                      box=(skid_l, skid_w, geom.skid_depth_m)))
    return items


def _part_item(p: Part, x: float, y: float, z: float, count: int = 1, mirrored: bool = False, **kw) -> Item:
    pts = _mirror(p.mass_kg, x, y, z) if mirrored else _lump(p.mass_kg * count, x, y, z)
    n = 2 if mirrored else count
    name = p.name if n == 1 else f"{p.name} x{n}"
    return Item(name, p.group, pts, p.cost_usd * n, "bought", box=p.size_m, power_w=p.power_w * n, **kw)


def equipment_items(geom: BWBGeometry, ps: PropulsionSet) -> list[Item]:
    """Bought parts, placed in the pod (fractions of the pod chord, so they follow the geometry)."""
    c0 = geom.pod_chord_m

    def floor(x, h):
        lo, _ = pod_inside_height(geom, x)
        return lo + h / 2

    def roof(x, h):
        lo, hgt = pod_inside_height(geom, x)
        return lo + hgt - h / 2

    A, C = AVIONICS, CAMERA
    m, pr, pk, esc = ps.motor, ps.prop, ps.pack, ps.esc
    items = [
        _part_item(A["airspeed"], 0.11 * c0, 0.0, floor(0.11 * c0, 0.012)),
        _part_item(A["gps"], 0.22 * c0, 0.0, roof(0.22 * c0, 0.009)),
        _part_item(C["cam"], 0.55 * c0, 0.0, floor(0.55 * c0, 0.012)),
        _part_item(C["mount"], 0.55 * c0, 0.0, floor(0.55 * c0, 0.012)),
        _part_item(C["pi"], 0.63 * c0, 0.055, floor(0.63 * c0, 0.012)),
        _part_item(C["sd"], 0.63 * c0, 0.055, floor(0.63 * c0, 0.012)),
        _part_item(A["fc"], 0.60 * c0, -0.050, floor(0.60 * c0, 0.015)),
        _part_item(A["bec"], 0.70 * c0, 0.060, floor(0.70 * c0, 0.008)),
        _part_item(A["rx"], 0.73 * c0, -0.040, floor(0.73 * c0, 0.005)),
        _part_item(A["rid"], 0.72 * c0, 0.030, floor(0.72 * c0, 0.012)),
        Item(f"ESC: {esc.name}", "propulsion", _lump(esc.mass_kg, 0.74 * c0, -0.045, floor(0.74 * c0, 0.010)),
             esc.cost_usd, "bought", box=(0.045, 0.025, 0.010)),
        Item(f"Motor: {m.name}", "propulsion", _lump(m.mass_kg, c0 + 0.004 + m.length_m / 2, 0, 0.0),
             m.cost_usd, "bought", box=(m.length_m, m.diameter_m, m.diameter_m)),
        Item(f"Prop: {pr.name}", "propulsion", _lump(pr.mass_kg, c0 + 0.010 + m.length_m, 0, 0.0),
             pr.cost_usd, "bought", box=(0.02, pr.diameter_m * 0.3, 0.02)),
        Item(f"Battery: {pk.name} ({pk.nominal_energy_wh:.0f} Wh nominal)", "propulsion",
             _lump(pk.mass_kg, 0.30 * c0, 0, floor(0.30 * c0, 0.023)), pk.cost_usd, "bought",
             box=(0.075, 0.066 * pk.parallel, 0.023), movable=True),
        Item("Wiring: XT60 + 14 AWG power, motor leads, servo extensions with panel plugs, signal leads", "wiring",
             np.vstack([_lump(0.025, 0.55 * c0, 0, -0.005), _mirror(0.006, 0.40 * c0, 0.12, 0.0)]), 12, "wiring"),
        Item("Hardware: hatch magnets, nylon panel bolts, velcro straps, glue", "wiring",
             _lump(0.025, 0.50 * c0, 0, 0.0), 15, "wiring"),
    ]
    # elevon servos in the panels, ~30 % along, ahead of the hinge line
    le, c, _ = geom._outer_station(0.30)
    x_hinge = le[0] + (1 - geom.elevon_chord_fraction) * c
    items.append(_part_item(A["servo"], x_hinge - 0.035, le[1], le[2], mirrored=True))
    items.append(_part_item(A["linkage"], x_hinge, le[1], le[2] - 0.005, mirrored=True))
    return items


@dataclass
class BuildUp:
    geom: BWBGeometry
    choices: BuildChoices
    propulsion: PropulsionSet
    items: list[Item]
    battery_x_m: float
    battery_range_m: tuple[float, float]  # where the pack can physically go
    ballast_kg: float  # nose ballast needed if the pack can't reach far enough forward
    fit_problems: list[str] = field(default_factory=list)

    def _all_points(self, as_built: bool) -> np.ndarray:
        pts = []
        for it in self.items:
            p = it.points.copy()
            if as_built:
                p[:, 0] *= 1 + GROWTH[it.known]
            pts.append(p)
        return np.vstack(pts)

    def mass_kg(self, as_built: bool = True) -> float:
        return float(self._all_points(as_built)[:, 0].sum())

    def cg(self, as_built: bool = True) -> np.ndarray:
        p = self._all_points(as_built)
        return (p[:, :1] * p[:, 1:]).sum(0) / p[:, 0].sum()

    @property
    def cost_usd(self) -> float:
        return sum(it.cost_usd for it in self.items)

    @property
    def avionics_power_w(self) -> float:
        return sum(it.power_w for it in self.items)

    def by_group(self, as_built: bool = True) -> dict[str, float]:
        out: dict[str, float] = {}
        for it in self.items:
            out[it.group] = out.get(it.group, 0.0) + (it.mass_as_built_kg if as_built else it.mass_kg)
        return out

    def inertia(self, mass_kg: float | None = None, as_built: bool = True):
        """Inertia about the CG, body axes. ``mass_kg`` scales every part uniformly
        (e.g. to the 1.5 kg design mass) keeping the CG and the radii of gyration."""
        from .stability import Inertia

        cg = self.cg(as_built)
        ixx = iyy = izz = 0.0
        for it in self.items:
            g = 1 + GROWTH[it.known] if as_built else 1.0
            m = it.points[:, 0] * g
            dx, dy, dz = (it.points[:, 1:] - cg).T
            ixx += float(np.sum(m * (dy**2 + dz**2)))
            iyy += float(np.sum(m * (dx**2 + dz**2)))
            izz += float(np.sum(m * (dx**2 + dy**2)))
            if it.spread is not None:
                sx, sy, sz = it.spread.T
            else:
                sx, sy, sz = (np.full(len(m), v) for v in it.box)
            ixx += float(np.sum(m * (sy**2 + sz**2) / 12))
            iyy += float(np.sum(m * (sx**2 + sz**2) / 12))
            izz += float(np.sum(m * (sx**2 + sy**2) / 12))
        m_tot = self.mass_kg(as_built)
        k = 1.0 if mass_kg is None else mass_kg / m_tot
        return Inertia(m_tot * k, float(cg[0]), ixx * k, iyy * k, izz * k,
                       note=f"M1.5 weight build-up ({len(self.items)} items"
                            + (f", scaled from {m_tot:.2f} kg" if mass_kg is not None else "") + ")")


def build_up(geom: BWBGeometry, ps: PropulsionSet, choices: BuildChoices | None = None) -> BuildUp:
    """Weight statement with the battery placed to put the as-built CG on the target."""
    ch = choices or BuildChoices()
    items = structure_items(geom, ch) + equipment_items(geom, ps)
    bat = next(i for i in items if i.movable)
    others = [i for i in items if not i.movable]

    def mx(seq):
        m = sum(i.mass_as_built_kg for i in seq)
        return m, sum(i.mass_as_built_kg * i.cg[0] for i in seq)

    m_o, mx_o = mx(others)
    m_b = bat.mass_as_built_kg
    x_bat = (ch.cg_target_x_m * (m_o + m_b) - mx_o) / m_b

    # Where can the pack go? Its whole length must sit where the pod is tall enough.
    L, H = bat.box[0], bat.box[2]
    xs = np.linspace(L / 2, geom.pod_chord_m - L / 2, 400)
    ok = np.array([min(pod_inside_height(geom, x - L / 2)[1], pod_inside_height(geom, x + L / 2)[1]) >= H + 0.003
                   for x in xs])
    lo, hi = (float(xs[ok][0]), float(xs[ok][-1])) if ok.any() else (float("nan"), float("nan"))
    ballast = 0.0
    if x_bat < lo:  # pack as far forward as it goes, then nose ballast at x = 0.03
        x_bat = lo
        x_bal = 0.03
        ballast = (mx_o + m_b * x_bat - ch.cg_target_x_m * (m_o + m_b)) / (ch.cg_target_x_m - x_bal) / (1 + GROWTH["bought"])
        others.append(Item("Nose ballast (lead)", "structure", _lump(ballast, x_bal), 1, "bought"))
    elif x_bat > hi:
        x_bat = hi
    lo_z, _ = pod_inside_height(geom, x_bat)
    bat = bat.moved_to(x_bat)
    bat.points[:, 3] = lo_z + H / 2
    items = others + [bat]

    problems = []
    for it in items:
        if it.box == (0.0, 0.0, 0.0) or it.points.shape[0] != 1 or it.group not in ("avionics", "camera", "propulsion"):
            continue
        x, _, z = it.cg
        if not 0 < x < geom.pod_chord_m:
            continue
        h = it.box[2]
        for xe in (x - it.box[0] / 2, x + it.box[0] / 2):
            if 0 < xe < geom.pod_chord_m and pod_inside_height(geom, xe)[1] < h + 0.002:
                problems.append(f"{it.name.split(':')[0]} too tall for the pod at x = {xe*1000:.0f} mm")
                break
    return BuildUp(geom, ch, ps, items, x_bat, (lo, hi), ballast, problems)


def pod_material_options(geom: BWBGeometry, ps: PropulsionSet) -> list[tuple[str, float, float, float]]:
    """(material, pod mass as built g, filament cost $, softening C) for the pod shell options."""
    out = []
    for key, wall in (("LW-PLA", 0.0006), ("LW-ASA", 0.0008), ("ASA", 0.0008), ("PETG", 0.0008)):
        ch = BuildChoices(pod_material=key, pod_wall_m=wall)
        it = next(i for i in structure_items(geom, ch) if i.name.startswith("Centre pod"))
        out.append((f"{key} {wall*1000:.1f} mm", it.mass_as_built_kg * 1000, it.cost_usd, MATERIALS[key].softens_c))
    return out


def panel_foam_options(geom: BWBGeometry) -> list[tuple[str, float, float, str]]:
    out = []
    for key in ("EPP 1.9 lb", "EPP 1.3 lb", "EPS 1.5 lb", "XPS"):
        it = structure_items(geom, BuildChoices(panel_foam=key))[0]
        out.append((MATERIALS[key].name, it.mass_as_built_kg * 1000, it.cost_usd, MATERIALS[key].note))
    return out


# --------------------------------------------------------------------------
# Default build (used by stability.py for the inertia)
# --------------------------------------------------------------------------


def default_propulsion() -> PropulsionSet:
    """The set :func:`manta_mapper.propulsion.select_propulsion` picks for the baseline (pinned by a test)."""
    from . import propulsion as P

    motor = next(m for m in P.MOTORS if m.name.startswith("SunnySky X2216 1100KV"))
    prop = next(p for p in P.PROPS if p.diameter_in == 10 and p.pitch_in == 6)
    pack = next(p for p in P.PACKS if p.cell.name == "Molicel P45B" and p.parallel == 1)
    return P.PropulsionSet(motor, prop, pack, P.ESCS[1])


def default_inertia(geom: BWBGeometry, mass_kg: float, x_cg: float):
    """Inertia from the default build-up, CG moved to ``x_cg``, scaled to ``mass_kg``."""
    bu = build_up(geom, default_propulsion(), BuildChoices(cg_target_x_m=x_cg))
    return bu.inertia(mass_kg)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def weight_table(bu: BuildUp) -> str:
    lines = []
    out = lines.append
    order = ["structure", "propulsion", "avionics", "camera", "wiring"]
    out(f"{'item':<92} {'g':>6} {'+grow':>6} {'x mm':>6} {'$':>5}")
    for grp in order:
        its = [i for i in bu.items if i.group == grp]
        if not its:
            continue
        out(f"[{grp}]")
        for it in its:
            out(f"  {it.name[:90]:<90} {it.mass_kg*1000:6.0f} {it.mass_as_built_kg*1000:6.0f} "
                f"{it.cg[0]*1000:6.0f} {it.cost_usd:5.0f}")
        sub = sum(i.mass_kg for i in its)
        sub_b = sum(i.mass_as_built_kg for i in its)
        out(f"  {'subtotal':<90} {sub*1000:6.0f} {sub_b*1000:6.0f} {'':>6} {sum(i.cost_usd for i in its):5.0f}")
    out(f"{'TOTAL':<92} {bu.mass_kg(False)*1000:6.0f} {bu.mass_kg(True)*1000:6.0f} "
        f"{bu.cg(True)[0]*1000:6.0f} {bu.cost_usd:5.0f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# M1.5 run: propulsion selection, weight build-up, missions, modes
# --------------------------------------------------------------------------


@dataclass
class M15:
    req: object
    geom: BWBGeometry
    polar: object  # QuadraticPolar fitted to the M1.4 trimmed polar
    polar_source: str
    stall_design: float  # stall speed at the design mass and the CG target
    selection: object
    buildup: BuildUp
    perf_design: object  # Performance at the design mass
    perf_built: object  # Performance at the as-built mass
    missions: list  # MissionEnergy
    endurance: dict
    modes: dict = field(default_factory=dict)  # label -> (Inertia, Modes)
    design_mass_kg: float = 1.5


def run(req, design_mass_kg: float = 1.5, quick: bool = False, progress=print) -> M15:
    from . import propulsion as P
    from .geometry import baseline, stall_speed_m_s

    geom = baseline(req, design_mass_kg)
    ch = BuildChoices()
    if quick:
        polar, src = P.M14_POLAR, "fit of the M1.4 polar (docs/stability_report.txt)"
        stall = stall_speed_m_s(geom, req, design_mass_kg)
        model = None
    else:
        from . import aero as A

        progress("Vortex lattice model and trimmed polar at the CG target ...")
        fl = A.cruise_flight(req)
        model = A.longitudinal_model(geom, fl)
        pts = A.trimmed_polar(geom, model, ch.cg_target_x_m, fl, design_mass_kg, np.arange(9.5, 22.01, 1.0))
        polar = P.QuadraticPolar.fit([p.cl for p in pts], [p.cd for p in pts])
        src = "M1.4 trimmed polar recomputed at the CG target, quadratic fit"
        progress("Stall at the CG target ...")
        stall = A.stall(geom, model, ch.cg_target_x_m, req, design_mass_kg).stall_speed_m_s

    progress("Propulsion: every motor x prop x pack in the catalogue ...")
    avionics_w = build_up(geom, default_propulsion(), ch).avionics_power_w
    cond_d = P.conditions(req, design_mass_kg, geom.reference_area_m2, stall, avionics_w)
    sel = P.select_propulsion(req, cond_d, polar)
    bu = build_up(geom, sel.chosen.ps, ch)
    m_b = bu.mass_kg(True)
    cond_b = P.conditions(req, m_b, geom.reference_area_m2, stall * math.sqrt(m_b / design_mass_kg), avionics_w)
    perf_b = P.evaluate(sel.chosen.ps, cond_b, req, polar)

    missions, endurance = [], {}
    for cond in (cond_b, cond_d):
        for ld in P.LD_CASES:
            pol = polar.scaled_to(ld)
            for area in (req.survey.area_min_ha, req.survey.area_stretch_ha):
                missions.append(P.mission_energy(sel.chosen.ps, cond, req, pol, area))
            endurance[(round(cond.mass_kg, 3), pol.label)] = P.endurance_min(
                sel.chosen.ps, cond, pol, req.limits.battery_reserve)

    out = M15(req, geom, polar, src, stall, sel, bu, sel.chosen, perf_b, missions, endurance,
              design_mass_kg=design_mass_kg)
    if model is not None:
        from . import aero as A
        from . import stability as S

        progress("Flight modes with the real inertia ...")
        fl = A.cruise_flight(req)
        x = ch.cg_target_x_m
        for label, mass in (("as built", m_b), ("design mass", design_mass_kg)):
            cruise = A.trimmed_point(geom, model, x, fl, mass)[0]
            d = S.derivatives(geom, model, x, fl, mass, cruise)
            real = bu.inertia(mass)
            old = S.estimate_inertia(geom, mass, x)
            out.modes[label] = (real, S.flight_modes(d, real, geom, fl), old, S.flight_modes(d, old, geom, fl))
    return out


N_FAILS_SHOWN = 15


def report(r: M15) -> str:
    from . import propulsion as P

    lines: list[str] = []
    out = lines.append
    bu, g = r.buildup, r.geom
    c = r.selection.chosen
    out(f"== {g.name}: weights and propulsion (M1.5) ==")
    out(f"Wing {g.span_m:.3f} m, {g.reference_area_m2:.3f} m^2, sized at {r.design_mass_kg:g} kg; "
        f"field {r.req.environment.field_elevation_m:g} m, ISA+{r.req.environment.temperature_offset_K:g} K")
    out(f"Drag: {r.polar_source}; model best L/D {r.polar.best_ld:.1f} at CL {r.polar.cl_best_ld:.2f}")
    out("")
    out("-- Weight statement (as built = with growth allowance: bought 5 %, structure 20 %, wiring/hardware 30 %) --")
    out(weight_table(bu))
    m_b = bu.mass_kg(True)
    out("")
    out(f"Mass: {bu.mass_kg(False)*1000:.0f} g on paper, {m_b*1000:.0f} g as built; "
        f"{(r.design_mass_kg - m_b)*1000:.0f} g under the {r.design_mass_kg:g} kg design mass "
        f"(wing loading {m_b/g.reference_area_m2:.2f} kg/m^2, AC-2 limit {r.req.limits.max_wing_loading_kg_m2:g})")
    out(f"Stall: {r.stall_design:.2f} m/s at {r.design_mass_kg:g} kg -> {r.stall_design*math.sqrt(m_b/r.design_mass_kg):.2f} m/s as built")
    cg = bu.cg(True)
    out(f"CG: x = {cg[0]*1000:.0f} mm (target {bu.choices.cg_target_x_m*1000:.0f}), y = {cg[1]*1000:+.1f} mm, "
        f"z = {cg[2]*1000:+.0f} mm; battery centre at x = {bu.battery_x_m*1000:.0f} mm "
        f"(it fits anywhere from {bu.battery_range_m[0]*1000:.0f} to {bu.battery_range_m[1]*1000:.0f} mm)")
    bat = next(i for i in bu.items if i.movable)
    shift = bat.mass_as_built_kg / m_b
    out(f"  sliding the pack 10 mm moves the CG {shift*10:.1f} mm, so the whole 261-275 mm band is "
        f"{14/shift:.0f} mm of pack travel; nose ballast needed: {bu.ballast_kg*1000:.0f} g")
    out(f"Fit check (inside height where each part sits): {'all fit' if not bu.fit_problems else '; '.join(bu.fit_problems)}")
    out(f"Avionics power: {bu.avionics_power_w:.1f} W at the parts, {bu.avionics_power_w/P.BEC_EFFICIENCY:.1f} W from the pack")
    out(f"Parts cost: ${bu.cost_usd:.0f} (aircraft only; radio, charger, tools, spares not included)")
    out("")
    out("Pod shell options (as built, incl. internal frames):")
    for name, mass, cost, soft in pod_material_options(g, c.ps):
        out(f"  {name:<16} {mass:5.0f} g  ${cost:4.0f} filament  softens ~{soft:.0f} C")
    out("Outer panel core options (both panels, cores only, as built):")
    for name, mass, cost, note in panel_foam_options(g):
        out(f"  {name:<26} {mass:5.0f} g  ${cost:4.0f}  {note}")
    out("")

    out(f"-- Propulsion selection ({len(r.selection.candidates)} combinations, checked at {r.design_mass_kg:g} kg, "
        f"field/survey air on the design day) --")
    out(f"Rule: pass every check, then the cheapest within {P.CRUISE_POWER_TOLERANCE:.0%} of the lowest cruise power")
    out(f"Showing the sets that pass, then the {N_FAILS_SHOWN} cheapest that don't")
    out(f"{'set':<72} {'$':>4} {'g':>5} {'T/W':>5} {'climb':>5} {'Vmax':>5} {'P_cr W':>7}  result")
    passing = [pf for pf in r.selection.candidates if pf.feasible]
    failing = [pf for pf in r.selection.candidates if not pf.feasible][:N_FAILS_SHOWN]
    for pf in passing + failing:
        fails = [k for k, (ok, _) in pf.checks.items() if not ok]
        tag = "CHOSEN" if pf is c else ("ok" if not fails else "fails: " + ", ".join(fails))
        out(f"{pf.ps.name[:72]:<72} {pf.ps.cost_usd:4.0f} {pf.ps.mass_kg*1000:5.0f} {pf.static_thrust_to_weight:5.2f} "
            f"{pf.climb_rate_m_s:5.1f} {pf.top_speed_m_s:5.1f} {pf.cruise.electrical_power_w:7.1f}  {tag}")
    out("")
    out(f"Chosen: {c.ps.name}, {c.ps.esc.name}")
    for k, (ok, txt) in c.checks.items():
        out(f"  {'PASS' if ok else 'FAIL'}  {k}: {txt}")
    out("")
    out("Operating points (chosen set):")
    out(f"{'case':<46} {'mass':>5} {'V':>5} {'rpm':>6} {'T N':>5} {'I_bat A':>7} {'P_el W':>7} {'thr':>4} {'eff':>5}")
    for label, pf in (("design mass", r.perf_design), ("as built", r.perf_built)):
        m = pf.ps and (r.design_mass_kg if pf is r.perf_design else m_b)
        for case, op in (("static, field, 80 % charge", pf.static), ("hand launch 1.2 Vs, field", pf.launch),
                         ("full-power climb, 30 % charge", pf.climb), ("cruise, survey height, L/D 12", pf.cruise),
                         ("fastest survey speed, L/D 12", pf.cruise_fast)):
            out(f"{label + ': ' + case:<46} {m:5.2f} {op.airspeed:5.1f} {op.rpm:6.0f} {op.thrust_n:5.2f} "
                f"{op.battery_current_a:7.1f} {op.electrical_power_w:7.1f} {op.throttle:4.0%} {op.efficiency:5.2f}")
    pb = r.perf_built
    out(f"As built: static thrust/weight {pb.static_thrust_to_weight:.2f}, launch climb gradient {pb.launch_gradient:.2f}, "
        f"climb {pb.climb_rate_m_s:.1f} m/s, top speed {pb.top_speed_m_s:.1f} m/s (pessimistic drag)")
    out("")
    out(f"-- Missions (pack {c.ps.pack.usable_energy_wh:.1f} Wh usable of {c.ps.pack.nominal_energy_wh:.1f} nominal; "
        f"need {r.req.limits.battery_reserve:.0%} left on landing) --")
    out(f"{'mass':>5} {'drag case':<12} {'area':>5} {'time':>6} {'P_cruise':>8} {'used Wh':>8} {'left':>5}  verdict")
    for me in r.missions:
        out(f"{me.mass_kg:5.2f} {me.polar_label:<12} {me.area_ha:5.0f} {me.time_s/60:5.1f}m {me.cruise_power_w:7.1f}W "
            f"{me.energy_wh:8.1f} {me.reserve:5.0%}  {'OK' if me.ok(r.req) else 'SHORT'}")
    out("Endurance at 14 m/s down to the 30 % reserve:")
    for (mass, lab), mins in r.endurance.items():
        out(f"  {mass:.2f} kg, {lab:<10} {mins:5.1f} min")
    if r.modes:
        out("")
        out("-- Flight modes with the real inertia (cruise, CG target) --")
        for label, (real, md, old, md_old) in r.modes.items():
            kx, ky, kz = real.radii_of_gyration_m
            out(f"{label} ({real.mass_kg:.2f} kg): Ixx {real.ixx:.4f}, Iyy {real.iyy:.4f}, Izz {real.izz:.4f} kg m^2 "
                f"(k = {kx*1000:.0f} / {ky*1000:.0f} / {kz*1000:.0f} mm); placeholder was "
                f"{old.ixx:.4f} / {old.iyy:.4f} / {old.izz:.4f}")
            for name, a, b in (("short period", md.short_period, md_old.short_period),
                               ("Dutch roll", md.dutch_roll, md_old.dutch_roll)):
                out(f"  {name:<13} {a.frequency_hz:.2f} Hz, damping {a.damping:.3f} "
                    f"(placeholder: {b.frequency_hz:.2f} Hz, {b.damping:.3f})")
            sp = md.spiral.time_constant_s
            out(f"  {'spiral':<13} {'halves' if sp > 0 else 'doubles'} in {abs(sp):.1f} s; "
                f"phugoid {md.phugoid.period_s:.1f} s, damping {md.phugoid.damping:.3f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_layout(r: M15, path: str | None = None):  # pragma: no cover - visual
    """Side view and top view of the pod with every part's mass and the CG band."""
    import matplotlib.pyplot as plt

    from .stability import BLUE, GRID, INK, INK2, ORANGE, AQUA, YELLOW, MAGENTA, _style

    g, bu = r.geom, r.buildup
    cols = {"structure": INK2, "propulsion": ORANGE, "avionics": BLUE, "camera": AQUA, "wiring": YELLOW}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw=dict(height_ratios=[1, 1.5]))
    xs, zu, zl = g.pod_depth_profile()
    ax.fill_between(xs * 1000, zl * 1000, zu * 1000, color=GRID, lw=0)
    ax.plot(xs * 1000, zu * 1000, color=INK2, lw=1)
    ax.plot(xs * 1000, zl * 1000, color=INK2, lw=1)
    for it in bu.items:
        if it.points.shape[0] != 1 or it.box == (0.0, 0.0, 0.0):
            continue
        x, _, z = it.cg
        lx, _, lz = it.box
        ax.add_patch(plt.Rectangle(((x - lx / 2) * 1000, (z - lz / 2) * 1000), lx * 1000, lz * 1000,
                                   fc=cols[it.group], ec="white", lw=1, alpha=0.9))
    ax.axvspan(261, 275, color=MAGENTA, alpha=0.15, lw=0, label="CG band 261-275 mm (5-10 % SM)")
    ax.axvline(bu.cg()[0] * 1000, color=MAGENTA, lw=1.5, label=f"CG as built {bu.cg()[0]*1000:.0f} mm")
    ax.set_aspect("equal")
    ax.set_ylabel("z (mm)")
    ax.set_title("Side view of the centre pod (centreline): parts placed and CG", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    # top view (span across the page, nose up): outline + every lump as a dot sized by mass
    st = g.stations()
    le = np.array([s["le"][:2] for s in st]) * 1000
    te = np.array([[s["le"][0] + s["chord"], s["le"][1]] for s in st]) * 1000
    for sgn in (1, -1):
        ax2.plot(sgn * np.r_[le[:, 1], te[::-1, 1], le[0, 1]], np.r_[le[:, 0], te[::-1, 0], le[0, 0]], color=INK2, lw=1)
    for grp, col in cols.items():
        pts = np.vstack([it.points for it in bu.items if it.group == grp])
        ax2.scatter(pts[:, 2] * 1000, pts[:, 1] * 1000, s=np.maximum(pts[:, 0] * 1000 * 4, 8), color=col,
                    alpha=0.8, edgecolor="white", lw=0.5, label=f"{grp} {bu.by_group()[grp]*1000:.0f} g")
    ax2.axhspan(261, 275, color=MAGENTA, alpha=0.15, lw=0)
    ax2.axhline(bu.cg()[0] * 1000, color=MAGENTA, lw=1.5)
    ax2.set_aspect("equal")
    ax2.invert_yaxis()
    ax2.set_ylabel("x from the pod nose (mm)")
    ax2.set_xlabel("y (mm)")
    ax2.set_title(f"Top view: mass lumps (dot area ~ mass), {bu.mass_kg()*1000:.0f} g as built", loc="left", fontsize=10)
    leg = ax2.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), markerscale=0.5)
    del leg
    for a in (ax, ax2):
        _style(a)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_propulsion(r: M15, path: str | None = None):  # pragma: no cover - visual
    """Thrust available vs drag, and mission reserve per drag case."""
    import matplotlib.pyplot as plt

    from . import propulsion as P
    from .stability import BLUE, INK2, ORANGE, AQUA, VIOLET, MAGENTA, _style

    ps = r.selection.chosen.ps
    m = r.design_mass_kg
    cond = P.conditions(r.req, m, r.geom.reference_area_m2, r.stall_design, r.buildup.avionics_power_w)
    v = np.linspace(0, 28, 57)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for soc, col, ls in ((P.SOC_LAUNCH, BLUE, "-"), (P.SOC_RESERVE, BLUE, "--")):
        ax.plot(v, [P.full_throttle(ps, vv, cond.field, soc).thrust_n for vv in v], color=col, ls=ls, lw=2,
                label=f"full-throttle thrust, field, {soc:.0%} charge")
    vd = np.linspace(r.stall_design, 28, 40)
    for ld, col in ((None, AQUA), (P.LD_PLANNING, ORANGE), (P.LD_PESSIMISTIC, MAGENTA)):
        pol = r.polar.scaled_to(ld)
        ax.plot(vd, [cond.drag(pol, cond.field, vv) for vv in vd], color=col, lw=2,
                label=f"drag at {m:g} kg, {pol.label if ld else f'model (L/D {r.polar.best_ld:.0f})'}")
    ax.axvline(1.2 * r.stall_design, color=INK2, lw=1, ls=":")
    ax.text(1.2 * r.stall_design - 0.3, 0.2, "hand launch", fontsize=8, color=INK2, ha="right")
    ax.axvline(cond.cruise_speed_m_s, color=INK2, lw=1, ls=":")
    ax.text(cond.cruise_speed_m_s + 0.3, 0.2, "survey", fontsize=8, color=INK2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("airspeed (m/s)")
    ax.set_ylabel("force (N)")
    ax.set_title(f"{ps.motor.name} + {ps.prop.diameter_in:g}x{ps.prop.pitch_in:g} folding, 32 °C at 1220 m",
                 loc="left", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    # reserve bars
    labels, vals25, vals50 = [], [], []
    for me in [x for x in r.missions if abs(x.mass_kg - m) < 1e-6]:
        lab = me.polar_label if me.polar_label != "model" else f"model {r.polar.best_ld:.0f}"
        if lab not in labels:
            labels.append(lab)
        (vals25 if me.area_ha == r.req.survey.area_min_ha else vals50).append(me.reserve * 100)
    xi = np.arange(len(labels))
    ax2.bar(xi - 0.2, vals25, 0.38, color=BLUE, label=f"{r.req.survey.area_min_ha:g} ha")
    ax2.bar(xi + 0.2, vals50, 0.38, color=VIOLET, label=f"{r.req.survey.area_stretch_ha:g} ha")
    for x0, a, b in zip(xi, vals25, vals50):
        ax2.text(x0 - 0.2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        ax2.text(x0 + 0.2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    ax2.axhline(r.req.limits.battery_reserve * 100, color=ORANGE, lw=1.5, ls="--", label="30 % reserve (MIS-4)")
    ax2.set_xticks(xi, [f"L/D {l}" if not l.startswith("L/D") else l for l in labels])
    ax2.set_ylabel("battery left on landing (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title(f"Battery left on landing at {m:g} kg, {ps.pack.name}", loc="left", fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    for a in (ax, ax2):
        _style(a)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    from .requirements import MissionRequirements

    p = argparse.ArgumentParser(prog="manta_mapper.weights",
                                description="M1.5: weight build-up, CG, inertia, propulsion choice and mission energy.")
    p.add_argument("mission", nargs="?", help="mission TOML (defaults built in)")
    p.add_argument("--mass", type=float, default=1.5, help="design (sizing) mass, kg")
    p.add_argument("--plots", help="directory to save plots into")
    p.add_argument("--report", help="also write the text report to this file")
    p.add_argument("--quick", action="store_true", help="use the stored M1.4 polar fit; skip the VLM and the modes")
    a = p.parse_args(argv)
    req = MissionRequirements.from_toml(a.mission) if a.mission else MissionRequirements()
    r = run(req, a.mass, quick=a.quick)
    text = report(r)
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
        jobs = [("weights_layout.png", plot_layout), ("propulsion.png", plot_propulsion)]
        for name, fn in jobs:
            fn(r, os.path.join(a.plots, name))
        print("Saved " + ", ".join(os.path.join(a.plots, n) for n, _ in jobs))


if __name__ == "__main__":
    main()
