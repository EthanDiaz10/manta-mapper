"""Build the M2.3 CAD recipe as meshes in plain Python (needs ``manifold3d``).

The Fusion 360 script is the CAD of record (editable bodies). This module
evaluates the *same* recipe with the manifold3d mesh kernel so that, without
Fusion:

* the tests can prove every part is a closed solid and fits the printer;
* the report can give each part's volume and mass;
* there are STLs to slice if the Fusion run hits a snag.

    pip install manifold3d
    python -m manta_mapper.cad --stl cad/stl --render docs/img/cad_parts.png
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .cad import CadSettings, Recipe, _usable

DENSITY_G_CM3 = {  # as printed
    "LW-ASA": 0.70,  # foamed (weights.py)
    "ASA (solid, not foamed)": 1.07,
    "TPU 95A": 1.21,
    "PETG": 1.27,
}
AIRFRAME_PARTS = ("pod_", "nose_cap", "hatch_", "nacelle", "camera_mount", "battery_stop")


def _m():
    import manifold3d as m3

    return m3


def _triangulate_cap(loop: np.ndarray) -> np.ndarray:
    """Triangles (indices into ``loop``) closing a planar polygon, all wound the same way."""
    m3 = _m()
    c = loop.mean(axis=0)
    _, _, vt = np.linalg.svd(loop - c)
    p2 = (loop - c) @ vt[:2].T
    area = 0.5 * np.sum(p2[:, 0] * np.roll(p2[:, 1], -1) - np.roll(p2[:, 0], -1) * p2[:, 1])
    if area < 0:
        p2[:, 0] *= -1  # triangulate wants counter-clockwise
    return np.asarray(m3.triangulate([p2.astype(np.float64)]), dtype=np.int64)


def loft_mesh(sections: list) -> "object":
    """Ruled loft through closed 3-D polylines with equal point counts, capped at both ends."""
    m3 = _m()
    S = [np.asarray(s, dtype=np.float64) for s in sections]
    n, k = len(S[0]), len(S)
    V = np.vstack(S)
    i = np.arange(n)
    i2 = (i + 1) % n
    side = []
    for j in range(k - 1):
        a, b = j * n, (j + 1) * n
        side.append(np.column_stack([a + i, a + i2, b + i2]))
        side.append(np.column_stack([a + i, b + i2, b + i]))
    side = np.vstack(side)

    def facing(T, away_from):
        a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        nrm = np.cross(b - a, c - a).sum(axis=0)
        return np.dot(nrm, V[T].reshape(-1, 3).mean(0) - away_from) > 0

    cap0 = _triangulate_cap(S[0])
    cap1 = _triangulate_cap(S[-1]) + (k - 1) * n
    if not facing(cap0, S[1].mean(0)):
        cap0 = cap0[:, ::-1]
    if not facing(cap1, S[-2].mean(0)):
        cap1 = cap1[:, ::-1]
    # walls: outward if they agree with the caps (signed volume positive)
    T = np.vstack([side, cap0, cap1])
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    if vol < 0:
        side = side[:, ::-1]
        T = np.vstack([side, cap0, cap1])
    mesh = m3.Mesh(vert_properties=V.astype(np.float32), tri_verts=T.astype(np.uint32))
    out = m3.Manifold(mesh)
    if out.status() != m3.Error.NoError:  # pragma: no cover - diagnostic
        raise ValueError(f"loft did not make a closed solid: {out.status()}")
    return out


def _rot_to(v: np.ndarray) -> np.ndarray:
    """3x3 rotation taking +z onto unit vector v."""
    z = np.array([0.0, 0.0, 1.0])
    v = v / np.linalg.norm(v)
    c = float(np.dot(z, v))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    ax = np.cross(z, v)
    s = np.linalg.norm(ax)
    ax /= s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def _affine(R: np.ndarray, t) -> list:
    M = np.hstack([R, np.asarray(t, float).reshape(3, 1)])
    return M.tolist()


def build(recipe: Recipe, segments: int = 48, only: list[str] | None = None) -> dict:
    """Evaluate the recipe; returns {solid name: manifold3d.Manifold}."""
    m3 = _m()
    out: dict = {}
    roots = only or [p["solid"] for p in recipe.parts]
    for name in recipe.order(roots):
        d = recipe.solids[name]
        op = d["op"]
        if op == "loft":
            out[name] = loft_mesh(d["sections"])
        elif op == "box":
            lo, hi = np.array(d["lo"]), np.array(d["hi"])
            out[name] = m3.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))
        elif op == "cyl":
            p0, p1 = np.array(d["p0"]), np.array(d["p1"])
            L = float(np.linalg.norm(p1 - p0))
            c = m3.Manifold.cylinder(L, d["r0"], d["r1"], segments)
            out[name] = c.transform(_affine(_rot_to(p1 - p0), p0))
        elif op == "prism":
            poly = np.asarray(d["poly"], dtype=np.float64)
            area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
            if area < 0:
                poly = poly[::-1]
            cs = m3.CrossSection([poly])
            out[name] = cs.extrude(d["z1"] - d["z0"]).translate((0, 0, d["z0"]))
        elif op == "move":
            out[name] = out[d["of"]].translate(tuple(d["v"]))
        elif op == "rotate":
            ax = np.array(d["axis"], float)
            ax /= np.linalg.norm(ax)
            t = math.radians(d["deg"])
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R = np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K
            o = np.array(d["origin"], float)
            out[name] = out[d["of"]].transform(_affine(R, o - R @ o))
        elif op == "union":
            out[name] = m3.Manifold.batch_boolean([out[i] for i in d["items"]], m3.OpType.Add)
        elif op == "inter":
            r = out[d["items"][0]]
            for i in d["items"][1:]:
                r = r ^ out[i]
            out[name] = r
        elif op == "diff":
            tools = [out[i] for i in d["items"]]
            out[name] = out[d["of"]] - m3.Manifold.batch_boolean(tools, m3.OpType.Add) if tools else out[d["of"]]
        else:  # pragma: no cover
            raise ValueError(op)
    return out


def centroid(manifold) -> np.ndarray:
    mesh = manifold.to_mesh()
    V = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    T = np.asarray(mesh.tri_verts)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    v = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    return ((a + b + c) / 4 * v[:, None]).sum(0) / v.sum()


def part_stats(recipe: Recipe, built: dict, st: CadSettings) -> dict:
    use = _usable(st)
    stats = {}
    for p in recipe.parts:
        if not p["printed"]:
            continue
        m = built[p["solid"]]
        bb = m.bounding_box()
        dims = np.array(bb[3:]) - np.array(bb[:3])
        vol = m.volume() / 1000.0
        stats[p["name"]] = dict(material=p["material"], count=p["count"], volume_cm3=vol,
                                mass_g=vol * DENSITY_G_CM3.get(p["material"], 1.0), bbox=dims.tolist(),
                                fits=bool(np.all(np.sort(dims) <= use + 1e-6)),
                                watertight=bool(m.status().name == "NoError" and m.num_tri() > 0),
                                pieces=len(m.decompose()),
                                genus=int(m.genus()), airframe=p["name"].startswith(AIRFRAME_PARTS),
                                centroid=centroid(m).tolist())
    return stats


def write_stl(manifold, path: str | Path) -> None:
    mesh = manifold.to_mesh()
    V = np.asarray(mesh.vert_properties)[:, :3]
    T = np.asarray(mesh.tri_verts)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    nrm = np.cross(b - a, c - a)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(T), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, np.stack([a, b, c], axis=1)
    with open(path, "wb") as f:
        f.write(b"Manta Mapper M2.3 (cad_mesh)".ljust(80, b" "))
        f.write(np.uint32(len(T)).tobytes())
        f.write(rec.tobytes())


def export_stl(recipe: Recipe, built: dict, folder: str | Path) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for p in recipe.parts:
        if not p["printed"]:
            continue
        f = folder / f"{p['name']}.stl"
        write_stl(built[p["solid"]], f)
        out.append(f)
    return out


def render(recipe: Recipe, built: dict, path: str | Path, exploded: float = 1.0) -> None:  # pragma: no cover - visual
    """Exploded views of the printed centre body from above and from below (sleeves and skid rails for reference)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    colors = {"pod_front": "#8fb4dd", "pod_rear": "#5b88c0", "nose_cap": "#333333", "hatch": "#e3a33b",
              "nacelle": "#666666", "camera_mount": "#cc4444", "battery_stop": "#2a9d5c", "sleeve": "#222222",
              "skid": "#c49a62"}
    offs = {"hatch_front_lid": (0, 0, 45), "hatch_rear_lid": (0, 0, 45), "nose_cap": (-35, 0, 0), "nacelle": (45, 0, 0),
            "pod_front_L": (-12, -14, 0), "pod_front_R": (-12, 14, 0), "pod_rear_L": (12, -14, 0), "pod_rear_R": (12, 14, 0),
            "skid_rail_R": (0, 14, -30), "sleeve_R": (12, 14, 0), "camera_mount": (0, 0, -60),
            "battery_stop": (0, 0, 60)}
    show = [p for p in recipe.parts if p["name"].split("_")[0] in ("pod", "nose", "hatch", "nacelle", "camera", "sleeve", "skid")]
    light = np.array([0.35, -0.45, 0.82])
    light /= np.linalg.norm(light)
    fig = plt.figure(figsize=(14, 6.5))
    for k, (elev, azim, title) in enumerate(((30, -130, "from above, front left"), (-28, -50, "from below, front right"))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        allv = []
        for p in show:
            mesh = built[p["solid"]].to_mesh()
            V = np.asarray(mesh.vert_properties)[:, :3] + np.array(offs.get(p["name"], (0, 0, 0))) * exploded
            T = np.asarray(mesh.tri_verts)
            for flip in ((1, -1) if p["name"] in ("skid_rail_R", "sleeve_R") else (1,)):
                VV = V * np.array([1, flip, 1])
                tri = VV[T]
                nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                shade = 0.4 + 0.6 * np.clip(np.abs(nrm @ light), 0, 1)
                key = next((c for c in colors if p["name"].startswith(c)), "pod_rear")
                fc = np.clip(np.array(matplotlib.colors.to_rgb(colors[key]))[None, :] * shade[:, None], 0, 1)
                ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolors="none"))
                allv.append(VV)
        allv = np.vstack(allv)
        lo, hi = allv.min(0), allv.max(0)
        mid, r = (lo + hi) / 2, (hi - lo).max() / 2 * 0.8
        ax.set_xlim(mid[0] - r, mid[0] + r)
        ax.set_ylim(mid[1] - r, mid[1] + r)
        ax.set_zlim(mid[2] - r, mid[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
    fig.suptitle("Printed centre body, exploded: four pod sections (blue), TPU nose cap (black), hatch lids (orange), "
                 "nacelle (grey), camera mount (red); sleeves and EPP skid rails for reference", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
