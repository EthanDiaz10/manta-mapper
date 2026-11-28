"""2-D templates for the EPP parts (milestone M2.3), as DXF.

Plain DXF R12 (lines, closed polylines, circles, text), millimetres, written
without any library so it opens in Fusion 360, Inkscape, LibreCAD or a laser
cutter's software. Every drawing is 1:1.

* ``panel_root_template.dxf`` / ``panel_tip_template.dxf``: hot-wire templates
  for the outer panels, drawn about the spar centre. Layer ``CUT`` is the
  template edge (grown by half the wire's kerf), ``PROFILE`` the true airfoil.
  The spar hole fits the tube at that end, so a straight rod through both
  holes lines the templates up (the spar is straight, sweep and dihedral come
  out by themselves). Level both ``DATUM`` lines. The root template also marks
  the anti-rotation pin.
* ``fin_root_template.dxf`` / ``fin_tip_template.dxf``: the same for the tip
  fins, in planes square to the fin span; cut the block a little long and trim
  the root end at the droop angle (noted on the drawing).
* ``panel_plan_R.dxf``: top view of the right panel from its root LE: outline,
  spar steps, elevon cut-out, servo pocket and lead channel, pin, magnets, fin
  pegs. Flip it for the left panel.
* ``skid_rail.dxf``: side profile of a skid rail (cut two from EPP).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .cad import Layout, offset_loop, template_loops


class DXF:
    def __init__(self):
        self.ents: list[str] = []
        self.layers: set[str] = {"0"}

    def _e(self, *pairs):
        self.ents.append("\n".join(f"{c}\n{v}" for c, v in pairs))

    def line(self, a, b, layer="0"):
        self.layers.add(layer)
        self._e((0, "LINE"), (8, layer), (10, f"{a[0]:.4f}"), (20, f"{a[1]:.4f}"), (30, "0.0"),
                (11, f"{b[0]:.4f}"), (21, f"{b[1]:.4f}"), (31, "0.0"))

    def poly(self, pts, layer="0", closed=True):
        self.layers.add(layer)
        out = [(0, "POLYLINE"), (8, layer), (66, 1), (10, "0.0"), (20, "0.0"), (30, "0.0"), (70, 1 if closed else 0)]
        self._e(*out)
        for p in pts:
            self._e((0, "VERTEX"), (8, layer), (10, f"{p[0]:.4f}"), (20, f"{p[1]:.4f}"), (30, "0.0"))
        self._e((0, "SEQEND"), (8, layer))

    def circle(self, c, r, layer="0"):
        self.layers.add(layer)
        self._e((0, "CIRCLE"), (8, layer), (10, f"{c[0]:.4f}"), (20, f"{c[1]:.4f}"), (30, "0.0"), (40, f"{r:.4f}"))

    def text(self, p, s, h=3.0, layer="TEXT"):
        self.layers.add(layer)
        self._e((0, "TEXT"), (8, layer), (10, f"{p[0]:.4f}"), (20, f"{p[1]:.4f}"), (30, "0.0"), (40, f"{h:.2f}"), (1, s))

    def dumps(self) -> str:
        head = "0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1009\n9\n$INSBASE\n10\n0.0\n20\n0.0\n30\n0.0\n0\nENDSEC"
        layers = "\n".join(f"0\nLAYER\n2\n{n}\n70\n0\n62\n{i % 7 + 1}\n6\nCONTINUOUS" for i, n in enumerate(sorted(self.layers)))
        tables = f"0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n70\n{len(self.layers)}\n{layers}\n0\nENDTAB\n0\nENDSEC"
        ents = "0\nSECTION\n2\nENTITIES\n" + "\n".join(self.ents) + "\n0\nENDSEC"
        return f"{head}\n{tables}\n{ents}\n0\nEOF\n"

    def save(self, path) -> Path:
        Path(path).write_text(self.dumps())
        return Path(path)


def read_dxf_entities(text: str) -> list[tuple[str, str]]:
    """(entity type, layer) for every entity in a DXF written here (for tests)."""
    lines = [s.strip() for s in text.splitlines()]
    pairs = list(zip(lines[0::2], lines[1::2]))
    out, in_ents = [], False
    for i, (c, v) in enumerate(pairs):
        if c == "2" and v == "ENTITIES":
            in_ents = True
        elif c == "0" and v == "ENDSEC":
            in_ents = False
        elif in_ents and c == "0":
            layer = next((vv for cc, vv in pairs[i + 1:i + 4] if cc == "8"), "0")
            out.append((v, layer))
    return out


def _template(name: str, loop: np.ndarray, hole_d: float, extra: list, lay: Layout, note: list[str]) -> DXF:
    st = lay.st
    d = DXF()
    cut = offset_loop(loop, -st.hotwire_kerf_mm / 2)
    d.poly(cut, "CUT")
    d.poly(loop, "PROFILE")
    d.circle((0, 0), hole_d / 2, "CUT")
    for ex, ez, ed in extra:
        d.circle((ex, ez), ed / 2, "CUT")
    x0, x1 = float(cut[:, 0].min()), float(cut[:, 0].max())
    z0, z1 = float(cut[:, 1].min()), float(cut[:, 1].max())
    d.line((x0 - 15, 0), (x1 + 15, 0), "DATUM")
    d.line((0, z0 - 10), (0, z1 + 10), "DATUM")
    y = z0 - 16
    d.text((x0, y), f"Manta Mapper {name} (1:1, mm)", 3.5)
    for i, s in enumerate(note):
        d.text((x0, y - 6 * (i + 1)), s, 2.5)
    return d


def write_all(lay: Layout, folder: str | Path) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    g, st = lay.geom, lay.st
    out = []
    kerf = f"CUT = template edge, grown {st.hotwire_kerf_mm/2:.1f} mm for a {st.hotwire_kerf_mm:.1f} mm kerf (check on scrap)"
    for name, loop, hole_d, extra in template_loops(lay):
        if name.startswith("panel"):
            eta = 0.0 if name.endswith("root") else 1.0
            le, c, tw = g._outer_station(eta)
            hinge = (1 - g.elevon_chord_fraction) * c * 1000
            # hinge marks: where the 75 % chord line meets the profile
            p = lay.spar_root if eta == 0 else lay.spar_tip
            sec_x = le[0] * 1000 - p[0] + hinge * math.cos(math.radians(tw))
            note = [kerf, f"Hole {hole_d:.1f} mm on the spar line (tube {hole_d - st.spar_hole_clear_mm:.0f} mm here); "
                    "a straight rod through both templates lines them up",
                    f"Chord {c*1000:.1f} mm, twist {tw:+.2f} deg (included), S5010",
                    "Level the DATUM line on both ends"]
            if eta == 0:
                note.append(f"Small hole: {st.pin_d_mm:.0f} mm anti-rotation pin at {st.pin_chord_frac:.0%} chord")
            d = _template(name, loop, hole_d, extra, lay, note)
            zz = loop[:, 1]
            d.line((sec_x, zz.min() - 6), (sec_x, zz.min() - 1), "MARK")
            d.line((sec_x, zz.max() + 1), (sec_x, zz.max() + 6), "MARK")
            d.text((sec_x + 1, zz.max() + 7), "elevon hinge", 2.0)
        else:
            note = [kerf, "Hole: reference rod through 30 % chord, mid-thickness, on both fin templates",
                    f"Fin droops {g.fin_droop_deg:.1f} deg: cut the block 10 mm long, then trim the root end "
                    f"{g.fin_droop_deg:.1f} deg about the chord line so it sits flat on the panel tip",
                    f"Pegs: {len(st.fin_peg_fracs)} x {st.fin_peg_d_mm:.0f} mm carbon, {st.fin_peg_len_mm:.0f} mm, at "
                    + ", ".join(f"{f:.0%}" for f in st.fin_peg_fracs) + " chord; tape over the joint (~30 N to pull off)"]
            d = _template(name, loop, hole_d, extra, lay, note)
        out.append(d.save(folder / f"{name}_template.dxf"))
    out.append(panel_plan(lay).save(folder / "panel_plan_R.dxf"))
    out.append(skid_rail(lay).save(folder / "skid_rail.dxf"))
    return out


def panel_plan(lay: Layout) -> DXF:
    """Top view of the right panel, origin at its root leading edge, x aft, y outboard (mm)."""
    g, st = lay.geom, lay.st
    d = DXF()
    r_le, r_c, _ = g._outer_station(0.0)
    t_le, t_c, _ = g._outer_station(1.0)
    o = r_le * 1000

    def P(le, dx=0.0):
        return (le[0] * 1000 - o[0] + dx, le[1] * 1000 - o[1])

    span = (t_le[1] - r_le[1]) * 1000
    out = [P(r_le), P(t_le), P(t_le, t_c * 1000), P(r_le, r_c * 1000)]
    d.poly(out, "OUTLINE")
    # elevon cut-out
    e0, e1 = g._outer_station(g.elevon_start), g._outer_station(g.elevon_end)
    f = 1 - g.elevon_chord_fraction
    ev = [P(e0[0], f * e0[1] * 1000), P(e1[0], f * e1[1] * 1000), P(e1[0], e1[1] * 1000), P(e0[0], e0[1] * 1000)]
    d.poly(ev, "ELEVON")
    d.text((ev[0][0] + 5, ev[0][1] + 10), f"elevon {g.elevon_chord_fraction:.0%} chord, hinge on top, 45 deg bevel underneath", 3)
    # spar line and steps
    a = lay.spar_root - o
    b = lay.spar_tip - o
    d.line((a[0], a[1]), (b[0], b[1]), "SPAR")
    L = float(np.linalg.norm(lay.spar_tip - lay.spar_root))
    for t in lay.spar_layout:
        for s in (t["from_mm"], t["to_mm"]):
            if 0 < s < L - 1:
                p = lay.point_on_spar(s) - o
                d.line((p[0] - 8, p[1]), (p[0] + 8, p[1]), "SPAR")
                d.text((p[0] + 10, p[1] - 1.5), f"{s:.0f} mm", 2.5)
    for t in lay.spar_layout:
        s = 0.5 * (max(t["from_mm"], 0) + min(t["to_mm"], L))
        p = lay.point_on_spar(s) - o
        d.text((p[0] - 40, p[1]), f"{t['od_mm']:.0f}x{t['id_mm']:.0f}", 3)
    # servo pocket ahead of the hinge at 30 % span, lead channel to the root
    le30, c30, _ = g._outer_station(0.30)
    xh = le30[0] * 1000 + f * c30 * 1000 - o[0]
    yc = le30[1] * 1000 - o[1]
    sx0, sx1 = xh - 35 - 12, xh - 35 + 12
    d.poly([(sx0, yc - 6), (sx1, yc - 6), (sx1, yc + 6), (sx0, yc + 6)], "SERVO")
    d.text((sx1 + 3, yc - 1), "servo pocket 24 x 12 mm (Emax ES08MD II), cut from below", 2.5)
    sh = lay.servo_hole - o
    d.line(((sx0 + sx1) / 2, yc - 6), (sh[0], 0.0), "SERVO")
    d.text((sh[0] + 3, 4), "lead channel to the root (servo plug through the root plate)", 2.5)
    # root features
    pp = lay.pin["p"] - o
    d.circle((pp[0], 3), st.pin_d_mm / 2, "ROOT")
    for m in lay.magnets:
        mm = m - o
        d.circle((mm[0], 2), st.magnet_d_mm / 2, "ROOT")
    d.text((0, -8), f"Root: {len(lay.magnets)} magnets {st.magnet_d_mm:.0f}x{st.magnet_t_mm:.0f} mm, pin {st.pin_d_mm:.0f} mm, "
           f"spar {lay.spar_layout[0]['od_mm']:.0f} mm", 3)
    for pg in lay.fin["pegs"]:
        q = pg - o
        d.circle((q[0], span - 2), st.fin_peg_d_mm / 2, "TIP")
    d.text((0, span + 10), f"Manta Mapper right outer panel, top view, 1:1 mm (flip for the left). Span {span:.0f} mm, "
           f"root chord {r_c*1000:.0f}, tip chord {t_c*1000:.0f}", 4)
    return d


def skid_rail(lay: Layout) -> DXF:
    pod, sk = lay.pod, lay.skid
    d = DXF()
    xs = np.linspace(sk["x0"], sk["x1"], 80)
    zt = np.array([pod.z_out(x, 0.5 * (sk["y0"] + sk["y1"]))[0] for x in xs])
    pts = [(x - sk["x0"], z - sk["z_bottom"]) for x, z in zip(xs, zt)]
    pts = pts + [(xs[-1] - sk["x0"], 0.0), (0.0, 0.0)]
    d.poly(pts, "CUT")
    h = zt - sk["z_bottom"]
    d.text((0, -8), f"Skid rail (cut 2), EPP from {math.ceil(h.max() / 5) * 5:.0f} mm sheet, {sk['y1'] - sk['y0']:.0f} mm wide; "
           "top follows the pod floor, 1 mm HDPE strip on the bottom, velcro on top", 3)
    d.text((0, -14), f"Front of the rail at x = {sk['x0']:.0f} mm from the pod nose; inner edge {sk['y0']:.0f} mm off the centreline", 3)
    return d


