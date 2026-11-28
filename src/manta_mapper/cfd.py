"""CFD check of the final design in OpenFOAM (milestone M2.1).

The top rung of the fidelity ladder (NeuralFoil -> AeroSandbox VLM -> VSPAERO
-> CFD). VSPAERO's panel mode could not mesh the thick aircraft (the fin
joint broke it), so this is the first analysis that sees the real thickness,
the real pod and the fin junction, with viscous flow.

What it does:

1. **Surface.** Lofts a watertight, full-span surface straight from
   :class:`~manta_mapper.geometry.BWBGeometry`: the same stations, airfoils,
   chords, twist, dihedral and fins the VLM uses, with ruled surfaces between
   stations (exactly what a hot-wire cut gives). The fin is part of the same
   loft, so the junction is closed by construction. Trailing edges get a
   minimum thickness (0.8 mm, about what the build will have) so the mesher
   has something to snap to.
2. **Case.** Writes an OpenFOAM case (half model, symmetry plane at y = 0):
   blockMesh background, snappyHexMesh with prism layers, steady
   incompressible RANS (simpleFoam, k-omega SST, Spalding wall function so
   any y+ is handled), forces about the planned CG.
3. **Post.** Reads the forces, turns them into CL, CD, Cm about the CG and
   finds the neutral point from dCm/dCL; compares with the AeroSandbox and
   VSPAERO results in ``docs/vspaero_v2_result.json``.

The flow is fully turbulent (no transition model). At Re ~ 1.9e5 the real
wing has laminar runs, so friction drag is an upper bound; lift slope and
neutral point, which are pressure-dominated, are what this rung is for.

Run (each step can be repeated on its own)::

    python -m manta_mapper.cfd stl  config/mission.toml --out cfd/F1.stl
    python -m manta_mapper.cfd case config/mission.toml --out cfd/runs --alpha 5.2 --mesh medium
    cfd/run_case.sh cfd/runs/<case>             # OpenFOAM v2406 in Docker (opencfd/openfoam-run:2406)
    python -m manta_mapper.cfd analyse config/mission.toml --runs cfd/runs --report docs/cfd_report.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .geometry import BWBGeometry

MIN_TE_THICKNESS_M = 0.0008
N_PER_SIDE = 81  # points per airfoil surface (cosine spaced, LE shared)
FIN_BLEND = 0.25  # fraction of fin length over which the section morphs to the fin airfoil

# Mesh presets: background cell (m) and surface refinement levels (min, max)
MESHES = {
    "coarse": dict(cell=0.32, surf=(5, 6), feature=6, box=3, wake=3, layers=3),
    "medium": dict(cell=0.32, surf=(6, 7), feature=7, box=3, wake=4, layers=4),
    "fine": dict(cell=0.24, surf=(6, 7), feature=8, box=4, wake=4, layers=5),
}


# --------------------------------------------------------------------------
# Surface
# --------------------------------------------------------------------------


def _airfoil_xz(key_a: str, key_b: str | None = None, frac: float = 0.0, n: int = N_PER_SIDE):
    """(upper z, lower z) at shared cosine-spaced x, optionally blended a -> b."""
    from .geometry import airfoil_surfaces

    x = 0.5 * (1 - np.cos(np.linspace(0, math.pi, n)))

    def zs(key):
        (xu, zu), (xl, zl) = airfoil_surfaces(key)
        return np.interp(x, xu, zu), np.interp(x, xl, zl)

    zu, zl = zs(key_a)
    if key_b is not None and frac > 0:
        zu2, zl2 = zs(key_b)
        zu, zl = (1 - frac) * zu + frac * zu2, (1 - frac) * zl + frac * zl2
    # make the LE a single point
    zle = 0.5 * (zu[0] + zl[0])
    zu[0] = zl[0] = zle
    return x, zu, zl


@dataclass
class Section:
    le: np.ndarray
    chord: float
    twist: float  # deg, nose-up positive, about the LE
    x: np.ndarray
    zu: np.ndarray
    zl: np.ndarray
    name: str = ""


def sections(geom: BWBGeometry) -> list[Section]:
    """Right-half sections from the centreline to the fin tip."""
    out = []
    for s in geom.stations():
        b = s["blend"]
        x, zu, zl = _airfoil_xz(geom.pod_airfoil, geom.outer_airfoil, b)
        out.append(Section(np.asarray(s["le"], float), s["chord"], s["twist"], x, zu, zl, s["name"]))
    root, tip = geom.fin_stations()
    tw_tip = out[-1].twist
    for frac in (FIN_BLEND, 1.0):
        le = root["le"] + frac * (tip["le"] - root["le"])
        c = root["chord"] + frac * (tip["chord"] - root["chord"])
        mix = min(frac / FIN_BLEND, 1.0)
        x, zu, zl = _airfoil_xz(geom.outer_airfoil, geom.fin_airfoil, mix)
        out.append(Section(np.asarray(le, float), c, (1 - frac) * tw_tip, x, zu, zl,
                           "fin tip" if frac == 1 else "fin blend"))
    return out


def _ring(sec: Section, up: np.ndarray, up_scale: float) -> np.ndarray:
    """3-D points of one section: upper TE -> LE -> lower TE (2n-1 points)."""
    c = sec.chord
    te_now = sec.zu[-1] - sec.zl[-1]
    add = max(0.0, MIN_TE_THICKNESS_M / c - te_now)
    zu = sec.zu + 0.5 * add * sec.x
    zl = sec.zl - 0.5 * add * sec.x
    xs = np.concatenate([sec.x[::-1], sec.x[1:]])
    zs = np.concatenate([zu[::-1], zl[1:]]) * up_scale
    t = math.radians(sec.twist)
    ex = np.array([1.0, 0.0, 0.0])
    chord_dir = math.cos(t) * ex - math.sin(t) * up
    up_dir = math.sin(t) * ex + math.cos(t) * up
    return sec.le + c * (xs[:, None] * chord_dir + zs[:, None] * up_dir)


def surface(geom: BWBGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Closed full-span triangle surface (vertices, faces), metres, geometry axes.

    x aft from the pod nose, y to the right, z up.
    """
    right = sections(geom)
    # mirror for the left half (skip the centreline section)
    left = []
    for s in right[:0:-1]:
        le = s.le * np.array([1, -1, 1])
        left.append(Section(le, s.chord, s.twist, s.x, s.zu, s.zl, s.name + " (L)"))
    secs = left + right
    les = np.array([s.le for s in secs])
    n = len(secs)
    rings = []
    for i, s in enumerate(secs):
        def tang(a, b):
            d = les[b] - les[a]
            d[0] = 0.0
            return d / np.linalg.norm(d)
        if i == 0:
            t = tang(0, 1)
            scale = 1.0
        elif i == n - 1:
            t = tang(n - 2, n - 1)
            scale = 1.0
        else:
            t1, t2 = tang(i - 1, i), tang(i, i + 1)
            t = t1 + t2
            t /= np.linalg.norm(t)
            half = 0.5 * math.acos(float(np.clip(t1 @ t2, -1, 1)))
            scale = 1.0 / math.cos(half)
        up = np.cross([1.0, 0.0, 0.0], t)
        rings.append(_ring(s, up, scale))
    m = rings[0].shape[0]
    verts = np.vstack(rings)
    faces = []
    for i in range(n - 1):
        a0, b0 = i * m, (i + 1) * m
        for j in range(m):
            j2 = (j + 1) % m  # j = m-1 closes the blunt trailing edge
            faces.append([a0 + j, b0 + j, b0 + j2])
            faces.append([a0 + j, b0 + j2, a0 + j2])
    # end caps: ladder between upper and lower points at the same x
    k = N_PER_SIDE
    for i, flip in ((0, True), (n - 1, False)):
        base = i * m
        up_idx = [base + (k - 1 - q) for q in range(k)]  # q = 0 at LE
        lo_idx = [base + k - 1 + q for q in range(k)]  # q = 0 at LE (shared)
        for q in range(1, k):
            if q == 1:
                tri = [[up_idx[0], up_idx[1], lo_idx[1]]]
            else:
                tri = [[up_idx[q - 1], up_idx[q], lo_idx[q]], [up_idx[q - 1], lo_idx[q], lo_idx[q - 1]]]
            for f in tri:
                faces.append(f[::-1] if flip else f)
    faces = np.array(faces, dtype=np.int64)
    # outward normals: flip everything if the signed volume is negative
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    if np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() < 0:
        faces = faces[:, ::-1]
    return verts, faces


def surface_stats(verts: np.ndarray, faces: np.ndarray) -> dict:
    import trimesh

    m = trimesh.Trimesh(verts, faces, process=True)
    return dict(watertight=bool(m.is_watertight), winding=bool(m.is_winding_consistent),
                volume_m3=float(m.volume), area_m2=float(m.area), n_faces=int(len(m.faces)),
                bounds=m.bounds.tolist())


def write_stl(verts: np.ndarray, faces: np.ndarray, path: str | Path, solid: str = "aircraft") -> str:
    """ASCII STL (OpenFOAM's surface readers like ASCII best)."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    lines = [f"solid {solid}"]
    for nn, a, b, c in zip(nrm, v0, v1, v2):
        lines.append(f" facet normal {nn[0]:.6e} {nn[1]:.6e} {nn[2]:.6e}\n  outer loop")
        for p in (a, b, c):
            lines.append(f"   vertex {p[0]:.7e} {p[1]:.7e} {p[2]:.7e}")
        lines.append("  endloop\n endfacet")
    lines.append(f"endsolid {solid}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
    return str(path)


# --------------------------------------------------------------------------
# Flight condition and OpenFOAM case
# --------------------------------------------------------------------------


@dataclass
class Condition:
    label: str
    alpha_deg: float
    velocity: float
    rho: float
    nu: float
    x_cg: float
    s_ref: float
    c_ref: float
    b_ref: float
    x: dict = field(default_factory=dict)

    @property
    def reynolds(self) -> float:
        return self.velocity * self.c_ref / self.nu

    def geometry(self) -> BWBGeometry:
        from .optimize import make_geometry

        return make_geometry(self.x)


def condition_from_manifest(manifest: str | Path, label: str = "F1", alpha_deg: float | None = None) -> Condition:
    """The VSPAERO case (same CG, speed, density, Re) so the rungs compare like for like."""
    man = json.loads(Path(manifest).read_text())
    c = next(c for c in man["cases"] if c["label"] == label)
    nu = c["velocity"] * c["c_ref"] / c["re_cref"]
    return Condition(label, c["alpha_cruise_deg"] if alpha_deg is None else alpha_deg, c["velocity"],
                     c["rho"], nu, c["x_cg"], c["s_ref"], c["c_ref"], c["b_ref"], c["x"])


def _foam_header(cls: str, obj: str, location: str | None = None) -> str:
    loc = f'    location    "{location}";\n' if location else ""
    return ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
            f"    class       {cls};\n{loc}    object      {obj};\n}}\n\n")


def case_name(cond: Condition, mesh: str) -> str:
    return f"{cond.label}_{mesh}_a{cond.alpha_deg:+05.1f}".replace("+", "p").replace("-", "m").replace(".", "_")


def write_case(cond: Condition, out_dir: str | Path, mesh: str = "medium", iterations: int = 800,
               n_procs: int = 4, stl: str | Path | None = None, mesh_from: str | Path | None = None) -> Path:
    """Write a complete OpenFOAM case directory. Returns its path.

    ``mesh_from``: another case of the same mesh preset whose finished mesh is
    reused (the mesh does not depend on alpha), so only the solver runs.
    """
    g = cond.geometry()
    mp = MESHES[mesh]
    case = Path(out_dir) / case_name(cond, mesh)
    for sub in ("0", "constant/triSurface", "system"):
        (case / sub).mkdir(parents=True, exist_ok=True)
    stl_path = case / "constant/triSurface/aircraft.stl"
    if stl:
        shutil.copy(stl, stl_path)
    else:
        write_stl(*surface(g), stl_path)
    if mesh_from:
        src = Path(mesh_from) / "constant/polyMesh"
        if not (src / "owner").exists():
            raise FileNotFoundError(f"no finished mesh in {mesh_from}")
        shutil.copytree(src, case / "constant/polyMesh", dirs_exist_ok=True)
        for log in ("log.checkMesh", "log.snappyHexMesh"):
            if (Path(mesh_from) / log).exists():
                shutil.copy(Path(mesh_from) / log, case / log)

    a = math.radians(cond.alpha_deg)
    U = cond.velocity
    ux, uz = U * math.cos(a), U * math.sin(a)
    lift_dir = (-math.sin(a), 0.0, math.cos(a))
    drag_dir = (math.cos(a), 0.0, math.sin(a))
    tu, visc_ratio = 0.002, 5.0
    k = 1.5 * (tu * U) ** 2
    omega = k / (cond.nu * visc_ratio)
    nut = cond.nu * visc_ratio

    # domain: ~20 chords upstream/around, 40 downstream (pod chord ~0.42 m)
    cell = mp["cell"]
    x0, x1, y1, z0, z1 = -6.4, 12.8, 6.4, -6.4, 6.4
    nx, ny, nz = (round((x1 - x0) / cell), round(y1 / cell), round((z1 - z0) / cell))
    sys_ = case / "system"
    (sys_ / "blockMeshDict").write_text(_foam_header("dictionary", "blockMeshDict") + f"""
convertToMeters 1;
vertices
(
    ({x0} 0 {z0}) ({x1} 0 {z0}) ({x1} {y1} {z0}) ({x0} {y1} {z0})
    ({x0} 0 {z1}) ({x1} 0 {z1}) ({x1} {y1} {z1}) ({x0} {y1} {z1})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
boundary
(
    inlet    {{ type patch; faces ((0 4 7 3)); }}
    outlet   {{ type patch; faces ((1 2 6 5) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
    symmetry {{ type symmetryPlane; faces ((0 1 5 4)); }}
);
""")
    # inlet = upstream face; outlet = downstream, far side, top and bottom (freestream patches use
    # inletOutlet / freestream-type BCs so the direction of the flow there does not matter)

    span_half = g.overall_span_m / 2 + 0.05
    lo, hi = surf = mp["surf"]
    (sys_ / "surfaceFeatureExtractDict").write_text(_foam_header("dictionary", "surfaceFeatureExtractDict") + """
aircraft.stl
{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs { includedAngle 150; }
    subsetFeatures { nonManifoldEdges no; openEdges yes; }
    writeObj            no;
}
""")
    first_layer = 0.0005
    (sys_ / "snappyHexMeshDict").write_text(_foam_header("dictionary", "snappyHexMeshDict") + f"""
castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    aircraft.stl {{ type triSurfaceMesh; name aircraft; }}
    nearBox  {{ type searchableBox; min (-0.25 0 -0.25); max (0.95 {span_half + 0.15:.3f} 0.25); }}
    wakeBox  {{ type searchableBox; min (-0.6 0 -0.6); max (3.5 {span_half + 0.5:.3f} 0.6); }}
}}

castellatedMeshControls
{{
    maxLocalCells 4000000;
    maxGlobalCells 12000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    features ( {{ file "aircraft.eMesh"; level {mp['feature']}; }} );
    refinementSurfaces
    {{
        aircraft {{ level ({lo} {hi}); patchInfo {{ type wall; }} }}
    }}
    resolveFeatureAngle 25;
    refinementRegions
    {{
        nearBox {{ mode inside; levels ((1e15 {mp['box'] + 1})); }}
        wakeBox {{ mode inside; levels ((1e15 {mp['wake']})); }}
    }}
    locationInMesh (-3.0 3.0 2.0);
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 100;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes false;
    layers {{ aircraft {{ nSurfaceLayers {mp['layers']}; }} }}
    expansionRatio 1.25;
    firstLayerThickness {first_layer};
    minThickness 0.0001;
    nGrow 0;
    featureAngle 120;
    slipFeatureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-15;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
    relaxed {{ maxNonOrtho 75; }}
}}

mergeTolerance 1e-6;
""")
    (sys_ / "decomposeParDict").write_text(_foam_header("dictionary", "decomposeParDict") + f"""
numberOfSubdomains {n_procs};
method scotch;
""")
    (sys_ / "controlDict").write_text(_foam_header("dictionary", "controlDict") + f"""
application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {iterations};
deltaT          1;
writeControl    timeStep;
writeInterval   {iterations};
purgeWrite      1;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    forces
    {{
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (aircraft);
        rho             rhoInf;
        rhoInf          {cond.rho};
        CofR            ({cond.x_cg} 0 0);
        log             false;
    }}
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (aircraft);
        rho             rhoInf;
        rhoInf          {cond.rho};
        CofR            ({cond.x_cg} 0 0);
        liftDir         ({lift_dir[0]:.8f} 0 {lift_dir[2]:.8f});
        dragDir         ({drag_dir[0]:.8f} 0 {drag_dir[2]:.8f});
        pitchAxis       (0 1 0);
        magUInf         {U};
        lRef            {cond.c_ref};
        Aref            {cond.s_ref / 2};
        log             true;
    }}
    yPlus
    {{
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
    }}
    wallShearStress
    {{
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        patches         (aircraft);
    }}
}}
""")
    (sys_ / "fvSchemes").write_text(_foam_header("dictionary", "fvSchemes") + """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; grad(U) cellLimited Gauss linear 1; grad(k) cellLimited Gauss linear 1; grad(omega) cellLimited Gauss linear 1; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwindV grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited 0.5; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited 0.5; }
wallDist        { method meshWave; }
""")
    (sys_ / "fvSolution").write_text(_foam_header("dictionary", "fvSolution") + """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.05; }
    Phi { $p; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent yes;  // SIMPLEC
    // no residualControl: force coefficients are what has to settle, checked afterwards
}
potentialFlow { nNonOrthogonalCorrectors 5; }
relaxationFactors
{
    fields { p 1; }
    equations { U 0.7; k 0.6; omega 0.6; }
}
""")
    const = case / "constant"
    (const / "transportProperties").write_text(_foam_header("dictionary", "transportProperties") +
                                               f"transportModel Newtonian;\nnu {cond.nu:.6e};\n")
    (const / "turbulenceProperties").write_text(_foam_header("dictionary", "turbulenceProperties") +
                                                "simulationType RAS;\nRAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }\n")

    far = "inlet|outlet"
    zero = case / "0"
    (zero / "U").write_text(_foam_header("volVectorField", "U", "0") + f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({ux:.6f} 0 {uz:.6f});
boundaryField
{{
    inlet    {{ type freestreamVelocity; freestreamValue $internalField; }}
    outlet   {{ type freestreamVelocity; freestreamValue $internalField; }}
    symmetry {{ type symmetryPlane; }}
    aircraft {{ type noSlip; }}
}}
""")
    (zero / "p").write_text(_foam_header("volScalarField", "p", "0") + """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet    { type freestreamPressure; freestreamValue $internalField; }
    outlet   { type freestreamPressure; freestreamValue $internalField; }
    symmetry { type symmetryPlane; }
    aircraft { type zeroGradient; }
}
""")
    for name, dim, val, wall in (
        ("k", "[0 2 -2 0 0 0 0]", k, "kqRWallFunction"),
        ("omega", "[0 0 -1 0 0 0 0]", omega, "omegaWallFunction"),
        ("nut", "[0 2 -1 0 0 0 0]", nut, "nutUSpaldingWallFunction"),
    ):
        far_bc = ("type calculated; value $internalField;" if name == "nut"
                  else "type inletOutlet; inletValue $internalField; value $internalField;")
        (zero / name).write_text(_foam_header("volScalarField", name, "0") + f"""
dimensions {dim};
internalField uniform {val:.6e};
boundaryField
{{
    "({far})" {{ {far_bc} }}
    symmetry {{ type symmetryPlane; }}
    aircraft {{ type {wall}; value $internalField; }}
}}
""")
    (case / "Allrun").write_text(f"""#!/bin/bash
# Mesh and solve. Logs go to log.*; forces to postProcessing/.
cd "${{0%/*}}" || exit 1
set -e
NP={n_procs}
[ -d constant/polyMesh ] || {{
  blockMesh > log.blockMesh 2>&1
  surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1
  decomposePar -force > log.decomposeMesh 2>&1
  mpirun --allow-run-as-root -np $NP snappyHexMesh -parallel -overwrite > log.snappyHexMesh 2>&1
  reconstructParMesh -constant > log.reconstructParMesh 2>&1
  rm -rf processor*
  checkMesh -constant > log.checkMesh 2>&1 || true
  rm -rf 0.orig; cp -r 0 0.orig
}}
decomposePar -force -latestTime > log.decomposePar 2>&1
if [ "$(foamListTimes -latestTime 2>/dev/null)" = "" ]; then  # fresh start: potential-flow initial field
  mpirun --allow-run-as-root -np $NP potentialFoam -parallel -writephi > log.potentialFoam 2>&1 || true
fi
log=log.simpleFoam; [ -f $log ] && log=log.simpleFoam.$(foamListTimes -latestTime)
mpirun --allow-run-as-root -np $NP simpleFoam -parallel > $log 2>&1
reconstructPar -latestTime > log.reconstructPar 2>&1
rm -rf processor*
foamToVTK -latestTime -fields '(p U wallShearStress yPlus)' > log.foamToVTK 2>&1
""")
    (case / "Allrun").chmod(0o755)
    (case / "condition.json").write_text(json.dumps(dict(asdict(cond), mesh=mesh, iterations=iterations),
                                                    indent=1, default=float))
    return case


# --------------------------------------------------------------------------
# 2-D section calibration (same solver settings on one airfoil)
# --------------------------------------------------------------------------

SECTION_DEPTH = 0.02  # spanwise thickness of the quasi-2-D slab


def section_surface(airfoil: str, chord: float, span: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """A straight, untwisted wing of one airfoil spanning y = -span/2 .. span/2 (closed)."""
    x, zu, zl = _airfoil_xz(airfoil)
    sec = lambda y: Section(np.array([0.0, y, 0.0]), chord, 0.0, x, zu, zl)
    up = np.array([0.0, 0.0, 1.0])
    r0, r1 = _ring(sec(-span / 2), up, 1.0), _ring(sec(span / 2), up, 1.0)
    m = r0.shape[0]
    verts = np.vstack([r0, r1])
    faces = []
    for j in range(m):
        j2 = (j + 1) % m
        faces += [[j, m + j, m + j2], [j, m + j2, j2]]
    k = N_PER_SIDE
    for base, flip in ((0, True), (m, False)):
        up_idx = [base + (k - 1 - q) for q in range(k)]
        lo_idx = [base + k - 1 + q for q in range(k)]
        for q in range(1, k):
            tri = ([[up_idx[0], up_idx[1], lo_idx[1]]] if q == 1 else
                   [[up_idx[q - 1], up_idx[q], lo_idx[q]], [up_idx[q - 1], lo_idx[q], lo_idx[q - 1]]])
            faces += [f[::-1] if flip else f for f in tri]
    faces = np.array(faces)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    if np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() < 0:
        faces = faces[:, ::-1]
    return verts, faces


def write_section_case(airfoil: str, cond: Condition, out_dir: str | Path, alpha_deg: float,
                       iterations: int = 1500, n_procs: int = 4) -> Path:
    """Quasi-2-D case: one airfoil at the MAC chord and Reynolds number, symmetry planes both sides.

    Same mesh controls, turbulence model and wall treatment as the aircraft
    cases, so comparing its CL_max with NeuralFoil shows how far the RANS
    stall prediction can be trusted at this Reynolds number.
    """
    c = cond.c_ref
    sc = Condition(f"{airfoil}2d", alpha_deg, cond.velocity, cond.rho, cond.nu, 0.25 * c,
                   2 * c * SECTION_DEPTH, c, SECTION_DEPTH, cond.x)  # s_ref x2: read_forces doubles
    tmp = Path(out_dir) / "_stl"
    tmp.mkdir(parents=True, exist_ok=True)
    stl = write_stl(*section_surface(airfoil, c), tmp / f"{airfoil}.stl")
    case = write_case(sc, out_dir, "medium", iterations, n_procs, stl=stl)
    cell = 0.12
    x0, x1, z0, z1 = -3.0, 6.0, -3.0, 3.0
    nx, nz = round((x1 - x0) / cell), round((z1 - z0) / cell)
    d = SECTION_DEPTH
    (case / "system/blockMeshDict").write_text(_foam_header("dictionary", "blockMeshDict") + f"""
convertToMeters 1;
vertices
(
    ({x0} 0 {z0}) ({x1} 0 {z0}) ({x1} {d} {z0}) ({x0} {d} {z0})
    ({x0} 0 {z1}) ({x1} 0 {z1}) ({x1} {d} {z1}) ({x0} {d} {z1})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} 1 {nz}) simpleGrading (1 1 1) );
boundary
(
    inlet    {{ type patch; faces ((0 4 7 3)); }}
    outlet   {{ type patch; faces ((1 2 6 5) (0 3 2 1) (4 5 6 7)); }}
    symmetry {{ type symmetry; faces ((0 1 5 4) (3 7 6 2)); }}
);
""")
    for f in (case / "0").iterdir():  # two parallel planes: the general 'symmetry' type
        f.write_text(f.read_text().replace("symmetry {{ type symmetryPlane; }}".replace("{{", "{").replace("}}", "}"),
                                           "symmetry { type symmetry; }"))
    sh = (case / "system/snappyHexMeshDict").read_text()
    sh = re.sub(r"nearBox  \{[^}]*\}", f"nearBox  {{ type searchableBox; min (-0.1 -1 -0.12); max ({1.3*c:.3f} 1 0.12); }}", sh)
    sh = re.sub(r"wakeBox  \{[^}]*\}", f"wakeBox  {{ type searchableBox; min (-0.4 -1 -0.4); max ({c + 1.5:.3f} 1 0.4); }}", sh)
    sh = sh.replace("locationInMesh (-3.0 3.0 2.0);", f"locationInMesh (-2.0 {d/2:.4f} 2.0);")
    sh = re.sub(r"aircraft \{ level \(\d+ \d+\);", "aircraft { level (6 7);", sh)
    (case / "system/snappyHexMeshDict").write_text(sh)
    return case


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------


def _read_dat(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        nums = [float(t) for t in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]
        rows.append(nums)
    w = min(len(r) for r in rows)
    return np.array([r[:w] for r in rows])


def _find(case: Path, fo: str, name: str) -> list[Path]:
    return sorted((case / "postProcessing" / fo).glob(f"*/{name}"), key=lambda p: float(p.parent.name))


def read_forces(case: str | Path, average_last: int = 200) -> dict:
    """Coefficients about the CG, whole aircraft, averaged over the last iterations.

    Uses the raw ``forces`` output (total force and moment about the CG) and
    does the axis algebra here, so OpenFOAM's coefficient conventions don't
    need to be trusted.
    """
    case = Path(case)
    cond = json.loads((case / "condition.json").read_text())
    f_files = _find(case, "forces", "force.dat")
    m_files = _find(case, "forces", "moment.dat")
    if f_files and m_files:
        F = np.vstack([_read_dat(p) for p in f_files])
        M = np.vstack([_read_dat(p) for p in m_files])
        it = F[:, 0]
        ftot, fp, fv = F[:, 1:4], F[:, 4:7], F[:, 7:10]
        mtot = M[:, 1:4]
    else:  # older layout: forces.dat with ((p) (v)) ((p) (v))
        D = np.vstack([_read_dat(p) for p in _find(case, "forces", "forces.dat")])
        it = D[:, 0]
        fp, fv = D[:, 1:4], D[:, 4:7]
        ftot = fp + fv
        mtot = D[:, 7:10] + D[:, 10:13]
    a = math.radians(cond["alpha_deg"])
    q = 0.5 * cond["rho"] * cond["velocity"] ** 2
    S, c = cond["s_ref"], cond["c_ref"]
    half = 2.0  # half model
    lift = np.array([-math.sin(a), 0, math.cos(a)])
    drag = np.array([math.cos(a), 0, math.sin(a)])
    sl = slice(-min(average_last, len(it)), None)

    def coef(vec, d, ref):
        return half * (vec[sl] @ d) / (q * ref)

    CL, CD = coef(ftot, lift, S), coef(ftot, drag, S)
    CDp, CDv = coef(fp, drag, S), coef(fv, drag, S)
    Cm = half * mtot[sl, 1] / (q * S * c)
    return dict(
        case=case.name, alpha_deg=cond["alpha_deg"], mesh=cond["mesh"], iterations=int(it[-1]),
        CL=float(CL.mean()), CD=float(CD.mean()), CD_pressure=float(CDp.mean()), CD_friction=float(CDv.mean()),
        Cm=float(Cm.mean()), x_ref=cond["x_cg"],
        CL_osc=float(np.ptp(CL)), CD_osc=float(np.ptp(CD)), Cm_osc=float(np.ptp(Cm)),
        history=dict(it=it.tolist(), CL=(half * (ftot @ lift) / (q * S)).tolist(),
                     CD=(half * (ftot @ drag) / (q * S)).tolist(),
                     Cm=(half * mtot[:, 1] / (q * S * c)).tolist()),
    )


def mesh_cells(case: str | Path) -> int | None:
    log = Path(case) / "log.checkMesh"
    if not log.exists():
        return None
    m = re.search(r"cells:\s+(\d+)", log.read_text())
    return int(m.group(1)) if m else None


def neutral_point(points: list[dict], c_ref: float, x_ref: float, mac_le_x: float,
                  alpha_range=(-0.1, 8.5)) -> dict:
    """Least-squares CL(alpha) and Cm(CL) over the linear range -> NP."""
    pts = [p for p in points if alpha_range[0] <= p["alpha_deg"] <= alpha_range[1]]
    al = np.radians([p["alpha_deg"] for p in pts])
    CL = np.array([p["CL"] for p in pts])
    Cm = np.array([p["Cm"] for p in pts])
    cla, cl0 = np.polyfit(al, CL, 1)
    dcm, cm0 = np.polyfit(CL, Cm, 1)
    x_np = x_ref - dcm * c_ref
    return dict(cl_alpha=float(cla), cl0=float(cl0), dcm_dcl=float(dcm), cm0=float(cm0), x_np=float(x_np),
                np_pct_mac=float(100 * (x_np - mac_le_x) / c_ref), n_points=len(pts))


def _drag_at_cl(points: list[dict], cl: float, key: str = "CD") -> float:
    pts = sorted(points, key=lambda p: p["CL"])
    CL = np.array([p["CL"] for p in pts])
    v = np.array([p[key] for p in pts])
    k = np.polyfit(CL, v, 2) if len(pts) >= 3 else np.polyfit(CL, v, 1)
    return float(np.polyval(k, cl))


# --------------------------------------------------------------------------
# Surface data (pressure, skin friction) from foamToVTK output
# --------------------------------------------------------------------------


def surface_vtk(case: str | Path):
    """The aircraft patch of the latest time as a pyvista PolyData (cell data p, wallShearStress)."""
    import pyvista as pv

    files = sorted(Path(case).glob("VTK/*/boundary/aircraft.vtp"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no VTK/*/boundary/aircraft.vtp in {case} (run foamToVTK)")
    m = pv.read(files[-1])
    if "p" not in m.cell_data:
        m = m.point_data_to_cell_data()
    return m


def internal_vtk(case: str | Path):
    import pyvista as pv

    files = sorted(Path(case).glob("VTK/*/internal.vtu"), key=lambda p: p.stat().st_mtime)
    if not files:
        files = sorted(Path(case).glob("VTK/*_*.vtm"), key=lambda p: p.stat().st_mtime)
        return pv.read(files[-1])["internal"]
    return pv.read(files[-1])


def face_forces(case: str | Path) -> dict:
    """Per-face force on the half aircraft (N), its centres, normals and Cp, Cf."""
    cond = json.loads((Path(case) / "condition.json").read_text())
    m = surface_vtk(case)
    m = m.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=False,
                          consistent_normals=False, split_vertices=False, flip_normals=False)
    area = m.compute_cell_sizes(length=False, volume=False)["Area"]
    n = np.asarray(m.cell_data["Normals"])  # OpenFOAM boundary faces point out of the fluid (into the body)
    ctr = np.asarray(m.cell_centers().points)
    p = np.asarray(m.cell_data["p"])  # kinematic
    tau = np.asarray(m.cell_data.get("wallShearStress", np.zeros((len(p), 3))))
    rho, V = cond["rho"], cond["velocity"]
    # orientation check: normals must point into the body, i.e. away from the outside
    if np.mean(np.einsum("ij,ij->i", n, ctr - ctr.mean(0))) > 0:
        n = -n
    f_p = rho * (p[:, None] * n) * area[:, None]
    f_v = -rho * tau * area[:, None]  # OpenFOAM's wallShearStress is the stress on the fluid
    q = 0.5 * V**2
    return dict(center=ctr, normal_out=-n, area=area, f_p=f_p, f_v=f_v, cp=p / q,
                cf_vec=-tau / q, cond=cond, mesh=m)


def _face_y_extent(mesh) -> tuple[np.ndarray, np.ndarray]:
    """(ymin, ymax) of every face of a pyvista PolyData."""
    pts = np.asarray(mesh.points)
    faces = mesh.faces
    lo, hi, i = [], [], 0
    while i < len(faces):
        n = faces[i]
        y = pts[faces[i + 1:i + 1 + n], 1]
        lo.append(y.min())
        hi.append(y.max())
        i += n + 1
    return np.array(lo), np.array(hi)


def _bin_spread(values: np.ndarray, ylo: np.ndarray, yhi: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Sum face values into spanwise bins, spreading each face evenly over its own y extent."""
    out = np.zeros(len(edges) - 1)
    width = np.maximum(yhi - ylo, 1e-9)
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        ov = np.clip(np.minimum(yhi, hi) - np.maximum(ylo, lo), 0, None) / width
        point = (yhi - ylo) < 1e-9
        ov[point] = ((ylo[point] >= lo) & (ylo[point] < hi)).astype(float)
        out[j] = (values * ov).sum()
    return out


def span_loading(case: str | Path, dy: float = 0.015) -> dict:
    """Section lift coefficient along the span of the main wing (fins excluded), from the surface forces."""
    ff = face_forces(case)
    cond = ff["cond"]
    g = Condition(**{k: cond[k] for k in Condition.__dataclass_fields__}).geometry()
    a = math.radians(cond["alpha_deg"])
    lift = np.array([-math.sin(a), 0, math.cos(a)])
    F = (ff["f_p"] + ff["f_v"]) @ lift
    ylo, yhi = _face_y_extent(ff["mesh"])
    tip_y = g.semispan_m
    on_wing = ff["center"][:, 1] < tip_y - 1e-4
    nb = max(1, int(round(tip_y / dy)))
    edges = np.linspace(0.0, tip_y, nb + 1)
    Fb = _bin_spread(np.where(on_wing, F, 0.0), ylo, np.minimum(yhi, tip_y), edges)
    st = g.stations()
    ys = np.array([s["le"][1] for s in st])
    cs = np.array([s["chord"] for s in st])
    q = 0.5 * cond["rho"] * cond["velocity"] ** 2
    yc = 0.5 * (edges[:-1] + edges[1:])
    c_loc = np.interp(yc, ys, cs)
    cl = Fb / (q * np.diff(edges) * c_loc)
    return dict(y=yc, eta=yc / tip_y, cl=cl, chord=c_loc, load=cl * c_loc / cond["c_ref"],
                fin_lift_fraction=float(F[~on_wing].sum() / F.sum()))


def reversed_flow(case: str | Path, dy: float = 0.015) -> dict:
    """Fraction of the upper-surface area with reversed skin friction, station by station.

    Reversed flow (the wall shear pointing forward, against the free stream)
    is where the boundary layer has separated. Where it first appears as
    alpha rises is where the stall starts. The blunt trailing-edge base is not
    "upper surface" (its normal points aft) so it is left out.
    """
    ff = face_forces(case)
    cond = ff["cond"]
    g = Condition(**{k: cond[k] for k in Condition.__dataclass_fields__}).geometry()
    upper = ff["normal_out"][:, 2] > 0.2
    rev = ff["cf_vec"][:, 0] < 0
    A = ff["area"]
    ylo, yhi = _face_y_extent(ff["mesh"])
    tip_y = g.semispan_m
    nb = max(1, int(round(tip_y / dy)))
    edges = np.linspace(0.0, tip_y, nb + 1)
    a_up = _bin_spread(np.where(upper, A, 0.0), ylo, yhi, edges)
    a_rev = _bin_spread(np.where(upper & rev, A, 0.0), ylo, yhi, edges)
    yc = 0.5 * (edges[:-1] + edges[1:])
    k = upper & (ff["center"][:, 1] < tip_y)
    return dict(y=yc, eta=yc / tip_y, fraction=a_rev / np.maximum(a_up, 1e-12),
                total=float(A[k & rev].sum() / A[k].sum()))


# --------------------------------------------------------------------------
# Comparison with the lower rungs and report
# --------------------------------------------------------------------------


def collect(runs_dir: str | Path) -> list[dict]:
    out = []
    for c in sorted(Path(runs_dir).glob("*/condition.json")):
        case = c.parent
        if not _find(case, "forces", "force.dat") and not _find(case, "forces", "forces.dat"):
            continue
        r = read_forces(case)
        r["cells"] = mesh_cells(case)
        r["velocity"] = json.loads(c.read_text())["velocity"]
        r["path"] = str(case)
        out.append(r)
    return out


def archive(runs_dir: str | Path, dest: str | Path) -> list[str]:
    """Copy the small results of every finished case (condition, mesh stats, force histories).

    The meshes and fields stay out of git; ``analyse --runs <dest> --no-renders``
    re-makes the report and plots from the archive alone.
    """
    out = []
    for c in sorted(Path(runs_dir).glob("*/condition.json")):
        case = c.parent
        forces = case / "postProcessing/forces"
        if not forces.exists():
            continue
        d = Path(dest) / case.name
        (d / "postProcessing").mkdir(parents=True, exist_ok=True)
        shutil.copy(c, d / "condition.json")
        if (case / "log.checkMesh").exists():
            shutil.copy(case / "log.checkMesh", d / "log.checkMesh")
        shutil.copytree(forces, d / "postProcessing/forces", dirs_exist_ok=True)
        out.append(str(d))
    return out


def lower_rungs(manifest: str | Path, vsp_runs: str | Path, label: str = "F1") -> dict:
    """AeroSandbox and VSPAERO sweeps for the same case (CG, speed, Re), elevons neutral."""
    from .vspaero import Case, asb_sweep, find_output, parse_polar

    man = json.loads(Path(manifest).read_text())
    case = Case(**next(c for c in man["cases"] if c["label"] == label))
    out = {"AeroSandbox VLM": asb_sweep(case)}
    pol = find_output(vsp_runs, f"{label}_vlm_sweep", ".polar")
    if pol is not None:
        out["VSPAERO VLM"] = parse_polar(pol)
    return out


def _np_from_sweep(sw: dict, x_ref: float, c_ref: float, mac_le: float, lo=-0.1, hi=8.1) -> dict:
    al = np.asarray(sw["alpha"], float)
    m = (al >= lo) & (al <= hi)
    cla = np.polyfit(np.radians(al[m]), np.asarray(sw["CL"])[m], 1)[0]
    cma = np.polyfit(np.radians(al[m]), np.asarray(sw["CMy"])[m], 1)[0]
    x_np = x_ref - cma / cla * c_ref
    return dict(cl_alpha=float(cla), x_np=float(x_np), np_pct_mac=float(100 * (x_np - mac_le) / c_ref))


def summarise(results: list[dict], cond: Condition, rungs: dict, sweep_mesh: str = "medium",
              cl_cruise: float | None = None, profile_cd: dict | None = None) -> dict:
    g = cond.geometry()
    sweep = sorted([r for r in results if r["mesh"] == sweep_mesh and abs(r["velocity"] - cond.velocity) < 1e-6],
                   key=lambda r: r["alpha_deg"])
    npt = neutral_point(sweep, cond.c_ref, cond.x_cg, g.mac_le_x_m)
    out = dict(cfd=npt, rungs={k: _np_from_sweep(v, cond.x_cg, cond.c_ref, g.mac_le_x_m) for k, v in rungs.items()},
               sweep=[{k: v for k, v in r.items() if k != "history"} for r in sweep])
    lin = [r for r in sweep if r["alpha_deg"] <= 8.1]
    if cl_cruise is not None and len(lin) >= 2:
        al = np.array([r["alpha_deg"] for r in lin])
        CL = np.array([r["CL"] for r in lin])
        a_cr = float(np.interp(cl_cruise, CL, al))
        Cm = np.array([r["Cm"] for r in lin])
        CD = np.array([r["CD"] for r in lin])
        CDf = np.array([r["CD_friction"] for r in lin])
        out["cruise"] = dict(cl=cl_cruise, alpha_deg=a_cr, cm_cg=float(np.interp(a_cr, al, Cm)),
                             cd=float(np.interp(a_cr, al, CD)), cd_friction=float(np.interp(a_cr, al, CDf)))
        out["cruise"]["ld"] = cl_cruise / out["cruise"]["cd"]
        for k, sw in rungs.items():
            o = np.argsort(sw["CL"])
            out["rungs"][k]["alpha_cruise_deg"] = float(np.interp(cl_cruise, np.asarray(sw["CL"])[o], np.asarray(sw["alpha"])[o]))
    # stall: highest CL in the sweep (any mesh preset at this speed)
    allv = [r for r in results if abs(r["velocity"] - cond.velocity) < 1e-6 and r["mesh"] == sweep_mesh]
    if allv:
        top = max(allv, key=lambda r: r["CL"])
        out["cl_max_seen"] = dict(CL=top["CL"], alpha_deg=top["alpha_deg"])
    study = sorted([r for r in results if r["mesh"] != "x" and abs(r["alpha_deg"] - cond.alpha_deg) < 0.05
                    and abs(r["velocity"] - cond.velocity) < 1e-6], key=lambda r: r["cells"] or 0)
    out["mesh_study"] = [{k: r[k] for k in ("mesh", "cells", "CL", "CD", "CD_friction", "Cm")} for r in study]
    return out


def report(summary: dict, cond: Condition) -> str:
    g = cond.geometry()
    c = summary["cfd"]
    L = [f"M2.1 CFD check: {cond.label}, span {g.span_m:.3f} m, S {cond.s_ref:.4f} m^2, MAC {cond.c_ref*1000:.0f} mm",
         f"OpenFOAM v2406 simpleFoam, k-omega SST (fully turbulent), half model; {cond.velocity:.1f} m/s, "
         f"rho {cond.rho:.4f}, Re_MAC {cond.reynolds:.3g}; moments about x = {cond.x_cg*1000:.1f} mm "
         f"(the pipeline's SM 7.5 % CG); elevons neutral, no prop, no skid",
         "", "Mesh study at alpha %.2f deg:" % cond.alpha_deg,
         f"  {'mesh':8s} {'cells':>9s} {'CL':>7s} {'CD':>8s} {'CDf':>8s} {'Cm':>8s}"]
    for r in summary["mesh_study"]:
        L.append(f"  {r['mesh']:8s} {r['cells'] or 0:9d} {r['CL']:7.4f} {r['CD']:8.5f} {r['CD_friction']:8.5f} {r['Cm']:8.4f}")
    L += ["", "Alpha sweep (%s mesh):" % (summary["sweep"][0]["mesh"] if summary["sweep"] else "-"),
          f"  {'alpha':>6s} {'CL':>7s} {'CD':>8s} {'CDp':>8s} {'CDf':>8s} {'Cm_cg':>8s} {'L/D':>6s}"]
    for r in summary["sweep"]:
        L.append(f"  {r['alpha_deg']:6.2f} {r['CL']:7.4f} {r['CD']:8.5f} {r['CD_pressure']:8.5f} "
                 f"{r['CD_friction']:8.5f} {r['Cm']:8.4f} {r['CL']/r['CD']:6.1f}")
    L += ["", "Lift slope and neutral point (alpha 0-8 deg):",
          f"  {'':18s} {'CL_alpha /rad':>13s} {'x_np mm':>8s} {'NP % MAC':>9s} {'SM at CG %':>11s}"]
    sm = lambda x: 100 * (x - cond.x_cg) / cond.c_ref
    for k, v in list(summary["rungs"].items()) + [("CFD (this)", c)]:
        L.append(f"  {k:18s} {v['cl_alpha']:13.3f} {v['x_np']*1000:8.1f} {v['np_pct_mac']:9.1f} {sm(v['x_np']):11.1f}")
    if "cruise" in summary:
        cr = summary["cruise"]
        L += ["", f"Cruise (CL {cr['cl']:.3f} for 1.5 kg at {cond.velocity:.0f} m/s): alpha {cr['alpha_deg']:.2f} deg "
              f"(AeroSandbox {summary['rungs'].get('AeroSandbox VLM', {}).get('alpha_cruise_deg', float('nan')):.2f}, "
              f"VSPAERO {summary['rungs'].get('VSPAERO VLM', {}).get('alpha_cruise_deg', float('nan')):.2f}), "
              f"Cm at CG {cr['cm_cg']:+.4f}, CD {cr['cd']:.4f} (friction {cr['cd_friction']:.4f}), L/D {cr['ld']:.1f}"]
    if "cl_max_seen" in summary:
        L.append(f"Highest CL in the sweep: {summary['cl_max_seen']['CL']:.3f} at alpha {summary['cl_max_seen']['alpha_deg']:.1f} deg")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Plots and renders
# --------------------------------------------------------------------------

COLORS = {"AeroSandbox VLM": "#444444", "VSPAERO VLM": "#2a6fdb", "CFD": "#c0392b"}


def _style(ax):
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_comparison(summary: dict, rungs: dict, path: str, cl_cruise: float | None = None):  # pragma: no cover
    import matplotlib.pyplot as plt

    sw = summary["sweep"]
    al = np.array([r["alpha_deg"] for r in sw])
    CL = np.array([r["CL"] for r in sw])
    CD = np.array([r["CD"] for r in sw])
    Cm = np.array([r["Cm"] for r in sw])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    for k, v in rungs.items():
        ax[0].plot(v["alpha"], v["CL"], "o--", ms=4, color=COLORS[k], label=k)
        ax[1].plot(v["CL"], v["CMy"], "o--", ms=4, color=COLORS[k], label=k)
    ax[0].plot(al, CL, "s-", color=COLORS["CFD"], lw=2, label="CFD (OpenFOAM, SST)")
    ax[1].plot(CL, Cm, "s-", color=COLORS["CFD"], lw=2, label="CFD")
    ax[2].plot(CD, CL, "s-", color=COLORS["CFD"], lw=2, label="CFD, fully turbulent")
    if "VSPAERO VLM" in rungs and "cruise" in summary:
        pass
    ax[0].set(xlabel="alpha, deg", ylabel="CL", title="lift (elevons neutral)")
    ax[1].set(xlabel="CL", ylabel="Cm about the CG", title="pitching moment: slope sets the neutral point")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[2].set(xlabel="CD", ylabel="CL", title="drag polar")
    if cl_cruise:
        for a in (ax[1],):
            a.axvline(cl_cruise, color="#888", ls=":", lw=1)
        ax[2].axhline(cl_cruise, color="#888", ls=":", lw=1)
        ax[2].text(ax[2].get_xlim()[0], cl_cruise, " cruise CL", va="bottom", color="#666", fontsize=8)
    if "cruise_ref_cd" in summary:
        for k, (cd, lab) in summary["cruise_ref_cd"].items():
            ax[2].plot([cd], [cl_cruise], "D", color=COLORS.get(k, "#444"), label=lab)
    for a in ax:
        _style(a)
    ax[0].legend(frameon=False, fontsize=8)
    ax[2].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spanload(loads: dict, lod_path: str | None, path: str, cond: Condition):  # pragma: no cover
    """Section cl along the span from CFD (several alphas) and VSPAERO at cruise."""
    import matplotlib.pyplot as plt
    from .vspaero import parse_lod

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    cmap = plt.get_cmap("Reds")
    keys = sorted(loads)
    for i, a in enumerate(keys):
        d = loads[a]
        col = cmap(0.35 + 0.6 * i / max(1, len(keys) - 1))
        ax[1].plot(d["eta"], d["cl"], "-", color=col, lw=1.8, label=f"CFD alpha {a:g}")
    if lod_path and Path(lod_path).exists():
        d = parse_lod(lod_path)
        st = d["strips"]
        w = (st["VortexSheet"] == 1) & (st["Yavg"] > 0)
        y, c, cl = st["Yavg"][w], st["Chord"][w], st["Cl"][w]
        o = np.argsort(y)
        ax[0].plot(y[o] / (cond.b_ref / 2), cl[o] * c[o] / cond.c_ref, "--", color=COLORS["VSPAERO VLM"], lw=2,
                   label=f"VSPAERO VLM, alpha {cond.alpha_deg:.1f}")
        ax[1].plot(y[o] / (cond.b_ref / 2), cl[o], "--", color=COLORS["VSPAERO VLM"], lw=2,
                   label=f"VSPAERO alpha {cond.alpha_deg:.1f}")
    a0 = min(keys, key=lambda a: abs(a - cond.alpha_deg))
    ax[0].plot(loads[a0]["eta"], loads[a0]["load"], "-", color=COLORS["CFD"], lw=2, label=f"CFD, alpha {a0:g}")
    g = cond.geometry()
    for a in ax:
        for x, lab in ((g.pod_half_width_m, "pod"), (g.root_y_m, "joint")):
            a.axvline(x / g.semispan_m, color="#999", lw=0.8, ls=":")
            a.text(x / g.semispan_m, a.get_ylim()[1] if False else 0.0, lab, rotation=90, va="bottom", ha="right",
                   fontsize=7, color="#777")
        e0, e1 = (g.root_y_m + g.elevon_start * g.outer_semispan_m) / g.semispan_m, (g.root_y_m + g.elevon_end * g.outer_semispan_m) / g.semispan_m
        a.axvspan(e0, e1, color="#2a6fdb", alpha=0.05)
        a.set_xlim(0, 1.0)
        a.set_xlabel("spanwise station 2y/b  (shaded: elevon)")
        _style(a)
    ax[0].set(ylabel="load  cl c / c_ref", title="spanwise lift at cruise")
    ax[1].set(ylabel="section cl", title="section lift coefficient vs alpha (peak = where stall starts)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_separation(seps: dict, path: str, cond: Condition):  # pragma: no cover
    import matplotlib.pyplot as plt

    g = cond.geometry()
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    cmap = plt.get_cmap("Reds")
    keys = sorted(seps)
    for i, a in enumerate(keys):
        d = seps[a]
        ax.plot(d["eta"], 100 * d["fraction"], "-", color=cmap(0.35 + 0.6 * i / max(1, len(keys) - 1)), lw=1.8,
                label=f"alpha {a:g} ({100*d['total']:.0f} % of upper surface)")
    e0, e1 = (g.root_y_m + g.elevon_start * g.outer_semispan_m) / g.semispan_m, (g.root_y_m + g.elevon_end * g.outer_semispan_m) / g.semispan_m
    ax.axvspan(e0, e1, color="#2a6fdb", alpha=0.06, label="elevon span")
    ax.set(xlabel="spanwise station 2y/b", ylabel="upper surface with reversed flow, %", xlim=(0, 1),
           title="where the flow separates first")
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(results: list[dict], path: str):  # pragma: no cover
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for r in sorted(results, key=lambda r: (r["mesh"], r["alpha_deg"])):
        h = r["history"]
        for a, k in zip(ax, ("CL", "CD", "Cm")):
            a.plot(h["it"], h[k], lw=1, label=f"{r['mesh']} a{r['alpha_deg']:g}")
    for a, k, lim in zip(ax, ("CL", "CD", "Cm"), ((-0.2, 1.3), (0, 0.15), (-0.1, 0.1))):
        a.set(xlabel="iteration", ylabel=k, ylim=lim)
        _style(a)
    ax[0].legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle("force convergence history (all runs)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _mirror(m):
    """Half model + its mirror image (faces re-wound so lighting works on both halves)."""
    r = m.reflect((0, 1, 0), point=(0, 0, 0))
    try:
        r = r.flip_faces()
    except Exception:  # older pyvista
        r.flip_normals()
    return m.merge(r)


def _wing_le(g: BWBGeometry, y: float) -> tuple[float, float]:
    st = g.stations()
    ys = [s["le"][1] for s in st]
    return float(np.interp(y, ys, [s["le"][0] for s in st])), float(np.interp(y, ys, [s["le"][2] for s in st]))


def render_cp(case: str | Path, path: str, view: str = "iso"):  # pragma: no cover
    """Whole aircraft (mirrored half model) coloured by surface Cp."""
    import pyvista as pv

    ff = face_forces(case)
    cond = ff["cond"]
    m = ff["mesh"].copy()
    m.cell_data["Cp"] = ff["cp"]
    full = _mirror(m).cell_data_to_point_data()
    pl = pv.Plotter(off_screen=True, window_size=(2200, 1300))
    pl.set_background("white")
    pl.add_mesh(full, scalars="Cp", cmap="RdBu", clim=(-1.6, 1.0), smooth_shading=True, specular=0.3,
                scalar_bar_args=dict(title="Cp", vertical=True, position_x=0.9, position_y=0.25, height=0.5,
                                     title_font_size=30, label_font_size=24, fmt="%.1f", color="black"))
    if view == "top":
        pl.view_xy()
        pl.camera.up = (-1, 0, 0)
        pl.camera.zoom(1.45)
    elif view == "bottom":
        pl.view_xy(negative=True)
        pl.camera.up = (-1, 0, 0)
        pl.camera.zoom(1.45)
    else:
        pl.camera_position = [(-1.3, -2.1, 1.4), (0.3, 0.0, 0.0), (0, 0, 1)]
        pl.camera.zoom(1.05)
    pl.add_text(f"Manta Mapper {cond['label']}: surface pressure, alpha {cond['alpha_deg']:g} deg, "
                f"{cond['velocity']:g} m/s ({view} view)", font_size=13, color="black")
    pl.screenshot(path)
    pl.close()


def render_flow(case: str | Path, path: str):  # pragma: no cover
    """Streamlines over the wing, coloured by speed (the wake picture is :func:`plot_wake`)."""
    import pyvista as pv

    ff = face_forces(case)
    cond = ff["cond"]
    g = Condition(**{k: cond[k] for k in Condition.__dataclass_fields__}).geometry()
    m = ff["mesh"].copy()
    m.cell_data["Cp"] = ff["cp"]
    vol = internal_vtk(case)
    if "U" not in vol.point_data:
        vol = vol.cell_data_to_point_data()
    a = math.radians(cond["alpha_deg"])
    seeds = []
    for y in np.linspace(0.03, g.semispan_m - 0.03, 18):
        x, z = _wing_le(g, y)
        seeds.append([x - 0.06, y, z + 0.012 - 0.06 * math.tan(a)])
    sl = vol.streamlines_from_source(pv.PolyData(np.array(seeds)), vectors="U", max_length=1.6,
                                     integration_direction="forward", initial_step_length=0.05, max_steps=8000)
    pl = pv.Plotter(off_screen=True, window_size=(2200, 1300))
    pl.set_background("white")
    pl.add_mesh(_mirror(m).cell_data_to_point_data(), scalars="Cp", cmap="RdBu", clim=(-1.6, 1.0),
                smooth_shading=True, show_scalar_bar=False)
    if sl.n_points:
        sl["speed / V"] = np.linalg.norm(sl["U"], axis=1) / cond["velocity"]
        pl.add_mesh(sl.tube(radius=0.0018), scalars="speed / V", cmap="viridis", clim=(0.7, 1.3),
                    scalar_bar_args=dict(title="speed / V", vertical=True, position_x=0.9, position_y=0.25,
                                         height=0.5, title_font_size=28, label_font_size=22, fmt="%.1f",
                                         color="black"))
    # from behind, outboard and above the right wing, looking forward along the wake
    pl.camera_position = [(-1.0, -1.2, 1.0), (0.6, 0.35, 0.0), (0, 0, 1)]
    pl.camera.zoom(1.15)
    pl.add_text(f"streamlines over the wing, alpha {cond['alpha_deg']:g} deg, {cond['velocity']:g} m/s",
                font_size=13, color="black")
    pl.screenshot(path)
    pl.close()


def plot_section(case: str | Path, y: float, path: str):  # pragma: no cover
    """Flow field round one wing section: speed contours and the section's Cp."""
    import matplotlib.pyplot as plt
    import pyvista as pv

    ff = face_forces(case)
    cond = ff["cond"]

    vol = internal_vtk(case)
    g = Condition(**{k: cond[k] for k in Condition.__dataclass_fields__}).geometry()
    x_le, z_le = _wing_le(g, y)
    chord = float(np.interp(y, [s["le"][1] for s in g.stations()], [s["chord"] for s in g.stations()]))
    nx, nz = 500, 230
    xs = np.linspace(x_le - 0.4 * chord, x_le + 1.6 * chord, nx)
    zs = np.linspace(z_le - 0.45 * chord, z_le + 0.45 * chord, nz)
    X, Z = np.meshgrid(xs, zs)
    probe = pv.PolyData(np.c_[X.ravel(), np.full(X.size, y), Z.ravel()]).sample(vol)
    U = np.asarray(probe["U"]).reshape(nz, nx, 3)
    valid = np.asarray(probe["vtkValidPointMask"]).reshape(nz, nx).astype(bool)
    spd = np.ma.masked_where(~valid, np.linalg.norm(U, axis=2) / cond["velocity"])
    surf = ff["mesh"].slice(normal="y", origin=(0, y, 0))
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    Xc, Zc = (X - x_le) / chord, (Z - z_le) / chord
    cs = ax[0].contourf(Xc, Zc, spd, levels=np.linspace(0, 1.5, 31), cmap="viridis", extend="both")
    fig.colorbar(cs, ax=ax[0], label="speed / V", shrink=0.85)
    Ux, Uz = U[..., 0], U[..., 2]
    ax[0].streamplot(Xc, Zc, np.where(valid, Ux, np.nan), np.where(valid, Uz, np.nan), color="white",
                     linewidth=0.5, density=1.4, arrowsize=0.6)
    sp = np.asarray(surf.points)
    ax[0].plot((sp[:, 0] - x_le) / chord, (sp[:, 2] - z_le) / chord, ".", ms=1, color="#222")
    ax[0].set(xlim=(-0.4, 1.6), ylim=(-0.45, 0.45), aspect="equal", xlabel="x / c", ylabel="z / c",
              title=f"speed round the section at y = {y*1000:.0f} mm (alpha {cond['alpha_deg']:g} deg)")
    # Cp on the section
    sc = surf.point_data_to_cell_data() if "p" not in surf.cell_data else surf
    cc = np.asarray(sc.cell_centers().points)
    cp = np.asarray(sc.cell_data["p"]) / (0.5 * cond["velocity"] ** 2)
    xc = (cc[:, 0] - x_le) / chord
    zc = (cc[:, 2] - z_le) / chord
    # upper/lower split by the local mean line
    order = np.argsort(xc)
    mid = np.interp(xc, xc[order], np.convolve(zc[order], np.ones(15) / 15, mode="same"))
    up = zc >= mid
    ax[1].plot(xc[up], cp[up], ".", ms=3, color="#c0392b", label="upper")
    ax[1].plot(xc[~up], cp[~up], ".", ms=3, color="#2a6fdb", label="lower")
    ax[1].invert_yaxis()
    ax[1].set(xlabel="x / c", ylabel="Cp", title="section pressure (suction up)")
    _style(ax[1])
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_wake(case: str | Path, path: str, dx: float = 0.15):  # pragma: no cover
    """Cross-flow in a plane just behind the trailing edge: the tip vortex and the fin's effect on it."""
    import matplotlib.pyplot as plt
    import pyvista as pv

    ff = face_forces(case)
    cond = ff["cond"]
    g = Condition(**{k: cond[k] for k in Condition.__dataclass_fields__}).geometry()
    vol = internal_vtk(case)
    fr, ft = g.fin_stations()
    x = float(ft["le"][0] + ft["chord"]) + dx
    ny, nz = 420, 200
    ys = np.linspace(0.0, g.overall_span_m / 2 + 0.15, ny)
    zs = np.linspace(-0.2, 0.15, nz)
    Y, Z = np.meshgrid(ys, zs)
    probe = pv.PolyData(np.c_[np.full(Y.size, x), Y.ravel(), Z.ravel()]).sample(vol)
    U = np.asarray(probe["U"]).reshape(nz, ny, 3) / cond["velocity"]
    a = math.radians(cond["alpha_deg"])
    # cross-flow relative to the free stream
    v = U[..., 1]
    w = U[..., 2] - U[..., 0] * math.tan(a)  # vertical velocity relative to the free-stream direction
    cross = np.hypot(v, w)
    fig, ax = plt.subplots(figsize=(13, 3.8))
    cs = ax.contourf(Y / g.semispan_m, Z * 1000, cross, levels=np.linspace(0, 0.12, 25), cmap="magma_r",
                     extend="max")
    fig.colorbar(cs, ax=ax, label="cross-flow speed / V", shrink=0.9)
    k = (slice(None, None, 9), slice(None, None, 9))
    ax.quiver(Y[k] / g.semispan_m, Z[k] * 1000, v[k], w[k], scale=2.5, width=0.0015, color="#222")
    tip = np.asarray(ft["le"])
    ax.plot([g.semispan_m / g.semispan_m, tip[1] / g.semispan_m], [g.tip_z_m * 1000, tip[2] * 1000], "-",
            color="#888", lw=3, alpha=0.6, label="fin (projected)")
    ax.set(xlabel="spanwise station 2y/b", ylabel="z, mm", xlim=(0, ys[-1] / g.semispan_m),
           title=f"wake {dx*1000:.0f} mm behind the fin trailing edge: cross-flow and tip vortex "
                 f"(alpha {cond['alpha_deg']:g} deg)")
    ax.set_aspect("auto")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_separation(case: str | Path, path: str):  # pragma: no cover
    """Top view, upper surface: streamwise skin friction, reversed flow in red."""
    import pyvista as pv
    from matplotlib.colors import LinearSegmentedColormap

    ff = face_forces(case)
    cond = ff["cond"]
    m = ff["mesh"].copy()
    m.cell_data["Cf_x"] = ff["cf_vec"][:, 0]
    full = _mirror(m)
    cmap = LinearSegmentedColormap.from_list("sep", [(0, "#c0392b"), (0.499, "#f5b7b1"), (0.5, "#e8eef5"),
                                                     (1, "#1f4e79")])
    pl = pv.Plotter(off_screen=True, window_size=(2200, 1200))
    pl.set_background("white")
    pl.add_mesh(full, scalars="Cf_x", cmap=cmap, clim=(-0.008, 0.008), lighting=False,
                scalar_bar_args=dict(title="streamwise skin friction Cf_x (red: reversed, separated)",
                                     vertical=False, position_x=0.25, position_y=0.05, width=0.5,
                                     title_font_size=24, label_font_size=20, fmt="%.3f", color="black"))
    pl.view_xy()
    pl.camera.up = (-1, 0, 0)
    pl.camera.zoom(1.45)
    pl.add_text(f"upper surface, alpha {cond['alpha_deg']:g} deg, {cond['velocity']:g} m/s", font_size=13,
                color="black")
    pl.screenshot(path)
    pl.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def section_calibration(sec_dir: str | Path, airfoil: str = "s5010", path: str | None = None) -> dict:  # pragma: no cover
    """2-D RANS (same settings) vs NeuralFoil for one airfoil at the MAC Reynolds number."""
    import neuralfoil as nf

    from .geometry import _load_airfoil

    res = sorted(collect(sec_dir), key=lambda r: r["alpha_deg"])
    if not res:
        return {}
    cond = json.loads((Path(res[0]["path"]) / "condition.json").read_text())
    re_ = cond["velocity"] * cond["c_ref"] / cond["nu"]
    al = np.arange(-2, 16.01, 0.5)
    af = _load_airfoil(airfoil)
    nfp = {"NeuralFoil, natural transition (n_crit 9)": nf.get_aero_from_airfoil(af, al, re_, n_crit=9),
           "NeuralFoil, tripped at the leading edge": nf.get_aero_from_airfoil(af, al, re_, n_crit=9,
                                                                               xtr_upper=0.01, xtr_lower=0.01)}
    out = dict(reynolds=re_, cfd=[{k: r[k] for k in ("alpha_deg", "CL", "CD", "CL_osc")} for r in res],
               neuralfoil={k: dict(cl_max=float(np.max(v["CL"])), alpha_cl_max=float(al[np.argmax(v["CL"])]))
                           for k, v in nfp.items()})
    if path:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for (k, v), col in zip(nfp.items(), ("#444444", "#888888")):
            ax[0].plot(al, v["CL"], "--", color=col, label=k)
            ax[1].plot(v["CD"], v["CL"], "--", color=col)
        a = [r["alpha_deg"] for r in res]
        ax[0].errorbar(a, [r["CL"] for r in res], yerr=[r["CL_osc"] / 2 for r in res], fmt="s-",
                       color=COLORS["CFD"], lw=2, capsize=3, label="2-D CFD, same mesh and SST settings")
        ax[1].plot([r["CD"] for r in res], [r["CL"] for r in res], "s-", color=COLORS["CFD"], lw=2)
        ax[0].set(xlabel="alpha, deg", ylabel="cl", title=f"{airfoil.upper()} at Re {re_:.2g}: section lift")
        ax[1].set(xlabel="cd", ylabel="cl", title="section drag polar", xlim=(0, 0.12))
        for x in ax:
            _style(x)
        ax[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return out


def analyse(mission: str, runs: str, manifest: str, vsp_runs: str, label: str = "F1", sweep_mesh: str = "medium",
            vsp_result: str = "docs/vspaero_v2_result.json", report_path: str | None = None, sec_dir: str | None = None,
            json_path: str | None = None, plots: str | None = None, renders: bool = True) -> dict:  # pragma: no cover
    from . import aero as A
    from .requirements import MissionRequirements

    req = MissionRequirements.from_toml(mission)
    cond = condition_from_manifest(manifest, label)
    g = cond.geometry()
    cl_cruise = A.cl_level(1.5, g, A.cruise_flight(req))
    results = collect(runs)
    rungs = lower_rungs(manifest, vsp_runs, label)
    summ = summarise(results, cond, rungs, sweep_mesh, cl_cruise)
    if Path(vsp_result).exists():
        vr = json.loads(Path(vsp_result).read_text())
        mt = next(c for c in vr["cases"] if c["label"] == label)["metrics"]
        summ["cruise_ref_cd"] = {
            "AeroSandbox VLM": (mt["AeroSandbox"]["cd_total_cruise"], "pipeline (VLM + NeuralFoil, transitional)"),
            "VSPAERO VLM": (mt["VSPAERO vlm"]["cd_total_cruise"], "VSPAERO induced + NeuralFoil profile")}
    text = report(summ, cond)
    if "cruise_ref_cd" in summ:
        text += "\nPipeline cruise CD for comparison: " + ", ".join(
            f"{lab} {cd:.4f}" for cd, lab in summ["cruise_ref_cd"].values())
    if sec_dir and Path(sec_dir).exists():
        cal = section_calibration(sec_dir, path=str(Path(plots) / "cfd_section_calibration.png") if plots else None)
        summ["section_calibration"] = cal
        if cal:
            text += "\n\nTurbulence-model check, S5010 alone (2-D, same settings, Re %.3g):" % cal["reynolds"]
            text += "\n  " + ", ".join(f"alpha {r['alpha_deg']:g}: cl {r['CL']:.3f}" for r in cal["cfd"])
            for k, v in cal["neuralfoil"].items():
                text += f"\n  {k}: cl_max {v['cl_max']:.2f} at {v['alpha_cl_max']:g} deg"
    if report_path:
        Path(report_path).write_text(text + "\n")
    if json_path:
        Path(json_path).write_text(json.dumps(summ, indent=1, default=float))
    if plots:
        out = Path(plots)
        out.mkdir(parents=True, exist_ok=True)
        plot_comparison(summ, rungs, str(out / "cfd_comparison.png"), cl_cruise)
        plot_convergence(results, str(out / "cfd_convergence.png"))
        sweep = {r["alpha_deg"]: r["path"] for r in results if r["mesh"] == sweep_mesh
                 and abs(r["velocity"] - cond.velocity) < 1e-6}
        has_vtk = {a: c for a, c in sweep.items() if list(Path(c).glob("VTK/*/boundary/aircraft.vtp"))}
        loads = {a: span_loading(c) for a, c in has_vtk.items()}
        if loads:
            plot_spanload(loads, str(Path(vsp_runs) / f"{label}_vlm_cruise.lod"), str(out / "cfd_spanload.png"), cond)
            summ["spanload"] = {a: dict(eta_peak=float(d["eta"][np.argmax(d["cl"])]), cl_peak=float(d["cl"].max()),
                                        fin_lift_fraction=d["fin_lift_fraction"]) for a, d in loads.items()}
            seps = {a: reversed_flow(c) for a, c in has_vtk.items()}
            plot_separation(seps, str(out / "cfd_separation.png"), cond)
            summ["separation"] = {a: d["total"] for a, d in seps.items()}
        if renders and has_vtk:
            a_cr = min(has_vtk, key=lambda a: abs(a - cond.alpha_deg))
            render_cp(has_vtk[a_cr], str(out / "cfd_cp_iso.png"), "iso")
            render_cp(has_vtk[a_cr], str(out / "cfd_cp_top.png"), "top")
            plot_wake(has_vtk[a_cr], str(out / "cfd_wake.png"))
            plot_section(has_vtk[a_cr], g.mac_y_m, str(out / "cfd_section_mac.png"))
            a_hi = max(has_vtk)
            render_separation(has_vtk[a_hi], str(out / f"cfd_skin_friction_a{a_hi:g}.png"))
            render_separation(has_vtk[a_cr], str(out / "cfd_skin_friction_cruise.png"))
            a8 = min(has_vtk, key=lambda a: abs(a - 8.0))
            render_separation(has_vtk[a8], str(out / f"cfd_skin_friction_a{a8:g}.png"))
        if json_path:
            Path(json_path).write_text(json.dumps(summ, indent=1, default=float))
    print(text)
    return summ


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(prog="python -m manta_mapper.cfd", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("stl", "case"):
        q = sub.add_parser(name)
        q.add_argument("mission")
        q.add_argument("--manifest", default="vsp_runs/v2/manifest.json")
        q.add_argument("--label", default="F1")
        q.add_argument("--out", required=True)
    sub.choices["case"].add_argument("--alpha", type=float, nargs="+", required=True)
    sub.choices["case"].add_argument("--mesh", default="medium", choices=list(MESHES))
    sub.choices["case"].add_argument("--iterations", type=int, default=800)
    sub.choices["case"].add_argument("--np", type=int, default=4)
    sub.choices["case"].add_argument("--mesh-from", help="reuse the mesh of this finished case")
    q = sub.add_parser("analyse")
    q.add_argument("mission")
    q.add_argument("--manifest", default="vsp_runs/v2/manifest.json")
    q.add_argument("--vsp-runs", default="vsp_runs/v2")
    q.add_argument("--label", default="F1")
    q.add_argument("--runs", default="cfd/runs")
    q.add_argument("--sweep-mesh", default="medium")
    q.add_argument("--report")
    q.add_argument("--json")
    q.add_argument("--plots")
    q.add_argument("--no-renders", action="store_true")
    q.add_argument("--archive", help="also copy the small per-case results here (for git)")
    q.add_argument("--sec", help="folder of 2-D section calibration cases")
    a = p.parse_args(argv)
    if a.cmd == "analyse":
        if a.archive:
            archive(a.runs, a.archive)
        if a.archive and a.sec:
            archive(a.sec, Path(a.archive) / "section")
        analyse(a.mission, a.runs, a.manifest, a.vsp_runs, a.label, a.sweep_mesh, report_path=a.report, sec_dir=a.sec,
                json_path=a.json, plots=a.plots, renders=not a.no_renders)
        return
    cond = condition_from_manifest(a.manifest, a.label)
    if a.cmd == "stl":
        v, f = surface(cond.geometry())
        write_stl(v, f, a.out)
        print(json.dumps(surface_stats(v, f), indent=1))
    elif a.cmd == "case":
        for al in a.alpha:
            c = condition_from_manifest(a.manifest, a.label, al)
            print(write_case(c, a.out, a.mesh, a.iterations, a.np, mesh_from=a.mesh_from))


if __name__ == "__main__":  # pragma: no cover
    main()
