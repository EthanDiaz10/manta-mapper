"""VSPAERO check of the optimizer finalists (milestone M1.7).

The fidelity ladder (PIPE-4) is NeuralFoil -> AeroSandbox VLM -> VSPAERO ->
CFD. This module is the third rung. It takes the top designs from the M1.6
optimizer, rebuilds each one in OpenVSP from the same :class:`BWBGeometry`,
runs VSPAERO two ways and compares the answers with the AeroSandbox VLM that
the whole pipeline (trim, stall, modes, optimizer) is built on:

* **VSPAERO VLM** (thin camber surfaces). Same physics as AeroSandbox, a
  different code and a different mesh. Disagreement here means the geometry
  translation or a convention is wrong, not the physics.
* **VSPAERO panel** (thick surfaces). Adds thickness, which matters most on
  the thick centre pod. This is the real check on the VLM.

What is compared (PIPE-4 limits in brackets): lift slope CL_alpha [10 %],
neutral point [3 % MAC], induced drag at cruise CL [20 % on total CD], and the
lateral derivatives that set Dutch roll damping, which the optimizer only
just met (>= 0.09). Dutch roll is then recomputed with each VSPAERO
derivative set, using the same inertia and the same profile-drag yaw damping
(which neither inviscid method can see).

Workflow (OpenVSP runs wherever it is installed; the rest runs anywhere)::

    python -m manta_mapper.vspaero prepare config/mission.toml --out vsp_runs
    # copy vsp_runs/ to a machine with OpenVSP, then in that folder:
    run_all.bat  (or ./run_all.sh)    # calls vspscript on each case
    # copy the folder back, then:
    python -m manta_mapper.vspaero analyse config/mission.toml --runs vsp_runs \\
        --report docs/vspaero_report.txt --json docs/vspaero_result.json --plots docs/img

Conventions. VSPAERO works in the OpenVSP frame (x aft, y right, z up), which
is the standard body frame (x forward, y right, z down) turned 180 deg about
y. Body moments CMx and CMz therefore flip sign against Cl and Cn, and so do
roll and yaw rates. Rather than trust a remembered convention, :func:`conventions`
works the signs out from derivatives whose sign is never in doubt (side
force with sideslip, roll damping) and says what it found in the report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .geometry import BWBGeometry
from .requirements import MissionRequirements

DESIGN_MASS_KG = 1.5
DESIGN_SM = 0.075  # same CG as the optimizer's mode checks
ALPHA_SWEEP_DEG = (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
METHODS = ("vlm", "panel")
KINDS = ("sweep", "stab", "beta")
BETA_STEP_DEG = 4.0  # explicit sideslip runs: VSPAERO's own 0.01 deg perturbation is lost in solver noise

# PIPE-4: each rung agrees with the next within these
PIPE4 = dict(cl_alpha_rel=0.10, cd_cruise_rel=0.20, np_pct_mac=3.0)
DUTCH_ROLL_MIN = 0.09  # optimizer's robust margin


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


@dataclass
class Case:
    """One design at its cruise condition, with everything VSPAERO needs."""

    label: str
    x: dict
    alpha_cruise_deg: float
    elevon_cruise_deg: float
    velocity: float
    mach: float
    rho: float
    re_cref: float
    x_cg: float
    s_ref: float
    b_ref: float
    c_ref: float
    optimizer_metrics: dict = field(default_factory=dict)

    def geometry(self) -> BWBGeometry:
        from .optimize import make_geometry

        return make_geometry(self.x)


def build_cases(req: MissionRequirements, result: dict, n: int = 3) -> list[Case]:
    """The top ``n`` optimizer finalists (they arrive ranked) at cruise, CG at SM 7.5 %."""
    from . import aero as A
    from .optimize import make_geometry

    fl = A.cruise_flight(req)
    cases = []
    for i, f in enumerate(result["finalists"][:n]):
        g = make_geometry(f["x"])
        model = A.longitudinal_model(g, fl)
        x_cg = model.cg_for_static_margin(DESIGN_SM)
        alpha, de = model.trim(A.cl_level(DESIGN_MASS_KG, g, fl), x_cg)
        cases.append(Case(
            label=f"F{i + 1}", x=f["x"], alpha_cruise_deg=alpha, elevon_cruise_deg=de,
            velocity=fl.velocity, mach=fl.mach, rho=fl.rho, re_cref=fl.reynolds(g.mac_m), x_cg=x_cg,
            s_ref=g.reference_area_m2, b_ref=g.span_m, c_ref=g.mac_m,
            optimizer_metrics={k: f["metrics"][k] for k in (
                "dutch_roll_damping", "np_pct_mac", "x_np_m", "stall_speed_m_s", "mass_built_kg", "cruise_ld")},
        ))
    return cases


# --------------------------------------------------------------------------
# OpenVSP script generation
# --------------------------------------------------------------------------


def geometry_block(geom: BWBGeometry) -> str:
    """The body of an OpenVSP script that builds the wing and tip fins.

    The section-by-section build (airfoil coordinates, sweep, dihedral, twist)
    comes from AeroSandbox's exporter, which is fed the same airplane the VLM
    analyses, so both codes see one geometry.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "g.vspscript"
        geom.to_airplane().export_OpenVSP_vspscript(str(p))
        text = p.read_text()
    start = text.index("{", text.index("void main()")) + 1
    end = text.rindex("}")
    return text[start:end]


_ANALYSIS = """
    //==== Mesh: about 15 mm spanwise panels, 41 chordwise points ====//
    array<string> geoms = FindGeoms();
    for ( int g = 0; g < int(geoms.size()); g++ )
    {
        SetParmVal( geoms[g], "Tess_W", "Shape", %(tess_w)d );
        int nx = GetNumXSec( GetXSecSurf( geoms[g], 0 ) );
        for ( int i = 1; i < nx; i++ )
        {
            string grp = "XSec_" + i;
            double span = GetParmVal( geoms[g], "Span", grp );
            int n = int( span / %(panel_m)f ) + 3;
            SetParmVal( geoms[g], "SectTess_U", grp, n );
        }
        Update();
        Print( "GEOM " + GetGeomName( geoms[g] ) + " TotalArea " + GetParmVal( geoms[g], "TotalArea", "WingGeom" )
               + " TotalProjectedSpan " + GetParmVal( geoms[g], "TotalProjectedSpan", "WingGeom" ) );
    }
    Update();

    SetVSP3FileName( "%(base)s.vsp3" );
    WriteVSPFile( "%(base)s.vsp3", SET_ALL );

    //==== Geometry for VSPAERO (%(method)s) ====//
    // OpenVSP 3.57: VLM vs panel is chosen only by the sets (thin = VLM, thick = panel);
    // stability derivatives are switched on with UnsteadyType = STABILITY_DEFAULT.
    array<int> thick = { %(thick)s };
    array<int> thin = { %(thin)s };
    string cg = "VSPAEROComputeGeometry";
    SetAnalysisInputDefaults( cg );
    SetIntAnalysisInput( cg, "GeomSet", thick );
    SetIntAnalysisInput( cg, "ThinGeomSet", thin );
    string rcg = ExecAnalysis( cg );

    //==== Solver ====//
    string sw = "VSPAEROSweep";
    SetAnalysisInputDefaults( sw );
    SetIntAnalysisInput( sw, "GeomSet", thick );
    SetIntAnalysisInput( sw, "ThinGeomSet", thin );
    array<int> manual = { MANUAL_REF };
    SetIntAnalysisInput( sw, "RefFlag", manual );
    array<double> sref = { %(s_ref).6f };
    array<double> bref = { %(b_ref).6f };
    array<double> cref = { %(c_ref).6f };
    array<double> xcg = { %(x_cg).6f };
    array<double> zero = { 0.0 };
    SetDoubleAnalysisInput( sw, "Sref", sref );
    SetDoubleAnalysisInput( sw, "bref", bref );
    SetDoubleAnalysisInput( sw, "cref", cref );
    SetDoubleAnalysisInput( sw, "Xcg", xcg );
    SetDoubleAnalysisInput( sw, "Ycg", zero );
    SetDoubleAnalysisInput( sw, "Zcg", zero );
    array<double> a0 = { %(a0).4f };
    array<double> a1 = { %(a1).4f };
    array<int> na = { %(na)d };
    array<double> mach = { %(mach).5f };
    array<int> one = { 1 };
    array<double> b0 = { %(b0).4f };
    array<double> b1 = { %(b1).4f };
    array<int> nb = { %(nb)d };
    array<double> re = { %(re).1f };
    array<double> vinf = { %(vinf).4f };
    array<double> rho = { %(rho).5f };
    SetDoubleAnalysisInput( sw, "AlphaStart", a0 );
    SetDoubleAnalysisInput( sw, "AlphaEnd", a1 );
    SetIntAnalysisInput( sw, "AlphaNpts", na );
    SetDoubleAnalysisInput( sw, "BetaStart", b0 );
    SetDoubleAnalysisInput( sw, "BetaEnd", b1 );
    SetIntAnalysisInput( sw, "BetaNpts", nb );
    SetDoubleAnalysisInput( sw, "MachStart", mach );
    SetDoubleAnalysisInput( sw, "MachEnd", mach );
    SetIntAnalysisInput( sw, "MachNpts", one );
    SetDoubleAnalysisInput( sw, "ReCref", re );
    SetDoubleAnalysisInput( sw, "Vinf", vinf );
    SetDoubleAnalysisInput( sw, "Rho", rho );
    array<int> wake = { 5 };
    SetIntAnalysisInput( sw, "WakeNumIter", wake );
    array<int> ncpu = { 4 };
    SetIntAnalysisInput( sw, "NCPU", ncpu );
    array<int> stab = { %(stab)s };
    SetIntAnalysisInput( sw, "UnsteadyType", stab );
    PrintAnalysisInputs( sw );
    string rid = ExecAnalysis( sw );
    WriteResultsCSVFile( rid, "%(base)s_results.csv" );
    Print( "DONE %(base)s" );
"""


def case_script(case: Case, method: str, kind: str) -> tuple[str, str]:
    """(base name, script text) for one VSPAERO run.

    ``method`` is "vlm" or "panel"; ``kind`` is "sweep" (alpha sweep at zero
    sideslip) or "stab" (stability derivatives at the trimmed cruise alpha).
    """
    if method not in METHODS or kind not in KINDS:
        raise ValueError(f"bad run {method}/{kind}")
    base = f"{case.label}_{method}_{kind}"
    geom = case.geometry()
    b0 = b1 = 0.0
    nb = 1
    if kind == "sweep":
        a0, a1, na = ALPHA_SWEEP_DEG[0], ALPHA_SWEEP_DEG[-1], len(ALPHA_SWEEP_DEG)
    else:
        a0 = a1 = round(case.alpha_cruise_deg, 4)
        na = 1
    if kind == "beta":
        b0, b1, nb = -BETA_STEP_DEG, BETA_STEP_DEG, 3
    body = _ANALYSIS % dict(
        base=base, method=method, tess_w=41, panel_m=0.015,
        thick="SET_ALL" if method == "panel" else "SET_NONE",
        thin="SET_NONE" if method == "panel" else "SET_ALL",
        s_ref=case.s_ref, b_ref=case.b_ref, c_ref=case.c_ref, x_cg=case.x_cg,
        a0=a0, a1=a1, na=na, b0=b0, b1=b1, nb=nb, mach=case.mach, re=case.re_cref, vinf=case.velocity, rho=case.rho,
        stab="STABILITY_DEFAULT" if kind == "stab" else "STABILITY_OFF",
    )
    head = (f"// Manta Mapper M1.7: {base}\n// generated by manta_mapper.vspaero; design {json.dumps(case.x)}\n\n"
            "void main()\n{\n    Print( \"VERSION \" + GetVSPVersion() );\n")
    return base, head + geometry_block(geom) + body + "}\n"


def prepare(cases: list[Case], out_dir: str | Path, methods=METHODS) -> dict:
    """Write one script per run, run_all.bat / run_all.sh and a manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for c in cases:
        for m in methods:
            for k in KINDS:
                base, text = case_script(c, m, k)
                (out / f"{base}.vspscript").write_text(text, newline="\n")
                runs.append(base)
    bat = ["@echo off", "rem Runs every case with OpenVSP's vspscript. Set VSP to the OpenVSP folder if needed.",
           'if "%VSP%"=="" set VSP=%USERPROFILE%\\OpenVSP', "cd /d %~dp0"]
    bat += [f'"%VSP%\\vspscript.exe" -script {r}.vspscript > {r}.log 2>&1' for r in runs]
    (out / "run_all.bat").write_text("\r\n".join(bat) + "\r\n", newline="")
    sh = ["#!/bin/sh", 'VSP="${VSP:-$HOME/OpenVSP}"', 'cd "$(dirname "$0")"']
    sh += [f'"$VSP/vspscript" -script {r}.vspscript > {r}.log 2>&1' for r in runs]
    (out / "run_all.sh").write_text("\n".join(sh) + "\n", newline="\n")
    manifest = dict(milestone="M1.7", runs=runs, cases=[asdict(c) for c in cases])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), newline="\n")
    return manifest


# --------------------------------------------------------------------------
# Output parsers
# --------------------------------------------------------------------------


def _floats(tokens) -> list[float] | None:
    try:
        return [float(t) for t in tokens]
    except ValueError:
        return None


def parse_polar(path: str | Path) -> dict[str, np.ndarray]:
    """VSPAERO ``.polar``: one header line of names, one row per condition."""
    header, rows = None, []
    for line in Path(path).read_text().splitlines():
        tok = line.split()
        if not tok:
            continue
        vals = _floats(tok)
        if vals is None:
            if "AoA" in tok or "Alpha" in tok:
                header = tok
            continue
        if header and len(vals) == len(header):
            rows.append(vals)
    if header is None or not rows:
        raise ValueError(f"no polar table in {path}")
    arr = np.array(rows)
    out = {h: arr[:, i] for i, h in enumerate(header)}
    return _alias(out)


def _alias(d: dict) -> dict:
    """Common names across VSPAERO versions (CL vs CLtot, CMy vs CMytot, AoA vs Alpha)."""
    pick = {"alpha": ("AoA", "Alpha"), "CL": ("CLtot", "CL"), "CDi": ("CDi", "CDind", "CDtot"),
            "CMy": ("CMytot", "CMy", "CMm"), "CS": ("CStot", "CS")}
    for k, names in pick.items():
        for n in names:
            if n in d:
                d.setdefault(k, d[n])
                break
    return d


STAB_COEFS = ("CFx", "CFy", "CFz", "CMx", "CMy", "CMz", "CL", "CD", "CS", "CMl", "CMm", "CMn")
STAB_COLUMNS = ("base", "alpha", "beta", "p", "q", "r")


@dataclass
class StabTable:
    """One block of a ``.stab`` file: base values and derivatives (per rad; rates non-dimensional)."""

    values: dict  # coef -> {column -> value}
    header: list
    units: list
    scalars: dict  # Sref_, Mach_, AoA_, SM, X_np, ...

    def d(self, coef: str, col: str) -> float:
        return self.values[coef][col]


def parse_stab(path: str | Path) -> StabTable:
    """VSPAERO ``.stab``. Coefficient rows are matched by name; columns by position.

    Column order is fixed by VSPAERO: total, then d/d alpha, beta, p, q, r
    (then Mach, U and control groups, unused here).
    """
    values, header, units, scalars = {}, [], [], {}
    lines = Path(path).read_text().splitlines()
    for line in lines:
        tok = line.replace("#", " ").split()
        if not tok:
            continue
        if tok[0] == "Coef" and not header:
            header = tok
            continue
        if header and not units and tok[0] in ("-",) or (header and len(units) < 2 and tok[0] == "-"):
            units.append(" ".join(tok))
            continue
        if line.lstrip().startswith("#"):
            continue
        if tok[0] in STAB_COEFS and tok[0] not in values:
            vals = _floats(tok[1:])
            if vals and len(vals) >= len(STAB_COLUMNS):
                values[tok[0]] = dict(zip(STAB_COLUMNS, vals))
            continue
        if len(tok) >= 2:
            v = _floats(tok[1:2])
            if v is not None and tok[0] not in values:
                scalars[tok[0]] = v[0]
    missing = [c for c in ("CFy", "CMx", "CMy", "CMz", "CL") if c not in values]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    return StabTable(values, header, units, scalars)


def find_output(runs_dir: str | Path, base: str, ext: str) -> Path | None:
    hits = sorted(Path(runs_dir).glob(f"{base}*{ext}"))
    return hits[0] if hits else None


# --------------------------------------------------------------------------
# AeroSandbox reference at exactly the VSPAERO conditions
# --------------------------------------------------------------------------


@dataclass
class DerivSet:
    """Stability-axis derivatives, standard signs (x fwd, z down), per rad; rates p b/2V etc."""

    cl_a: float
    cm_a: float
    cl_q: float
    cm_q: float
    cy_b: float
    cl_b: float
    cn_b: float
    cy_p: float
    cl_p: float
    cn_p: float
    cy_r: float
    cl_r: float
    cn_r: float

    LATERAL = ("cy_b", "cl_b", "cn_b", "cy_p", "cl_p", "cn_p", "cy_r", "cl_r", "cn_r")


def to_stability_axes(body: dict, alpha_deg: float) -> dict:
    """Body-axis lateral derivatives (moments Cl, Cn; rates p, r about body axes) -> stability axes.

    ``body`` holds cy_b, cl_b, cn_b and the 2x2 rate block {cl_p, cl_r, cn_p, cn_r} plus cy_p, cy_r,
    all in the standard body frame. Moments and rates both rotate by alpha about y.
    """
    a = math.radians(alpha_deg)
    ca, sa = math.cos(a), math.sin(a)
    R = np.array([[ca, sa], [-sa, ca]])  # body (l, n) -> stability (l, n)
    M = np.array([[body["cl_p"], body["cl_r"]], [body["cn_p"], body["cn_r"]]])
    # stability rates (ps, rs) -> body rates: p = ps ca - rs sa, r = ps sa + rs ca
    T = np.array([[ca, -sa], [sa, ca]])
    Ms = R @ M @ T
    lb, nb = R @ np.array([body["cl_b"], body["cn_b"]])
    cy_ps = body["cy_p"] * ca + body["cy_r"] * sa
    cy_rs = -body["cy_p"] * sa + body["cy_r"] * ca
    return dict(cy_b=body["cy_b"], cl_b=float(lb), cn_b=float(nb), cy_p=float(cy_ps), cy_r=float(cy_rs),
                cl_p=float(Ms[0, 0]), cl_r=float(Ms[0, 1]), cn_p=float(Ms[1, 0]), cn_r=float(Ms[1, 1]))


def asb_body_derivatives(case: Case) -> dict:
    """AeroSandbox VLM derivatives at the VSPAERO stability-run condition (elevons neutral), body axes."""
    from . import aero as A

    g = case.geometry()
    fl = A.Flight(case.velocity, 0.0)
    fl = _flight_like(case)
    V, b, c = case.velocity, g.span_m, g.mac_m
    al = case.alpha_cruise_deg
    kw = dict(xyz_ref=(case.x_cg, 0.0, 0.0))
    base = A.run_vlm(g, fl, al, **kw)
    da, db, dh = 1.0, 2.0, 0.01
    ra = A.run_vlm(g, fl, al + da, **kw)
    rb = A.run_vlm(g, fl, al, beta=db, **kw)
    rp = A.run_vlm(g, fl, al, p=dh * 2 * V / b, **kw)
    rq = A.run_vlm(g, fl, al, q=dh * 2 * V / c, **kw)
    rr = A.run_vlm(g, fl, al, r=dh * 2 * V / b, **kw)
    dar, dbr = math.radians(da), math.radians(db)

    def dd(r, k, h):
        return (r[k] - base[k]) / h

    return dict(
        cl=base["CL"], cdi=base["CD"], cm=base["Cm"],
        cl_a=dd(ra, "CL", dar), cm_a=dd(ra, "Cm", dar), cl_q=dd(rq, "CL", dh), cm_q=dd(rq, "Cm", dh),
        cy_b=dd(rb, "CY", dbr), cl_b=dd(rb, "Cl", dbr), cn_b=dd(rb, "Cn", dbr),
        cy_p=dd(rp, "CY", dh), cl_p=dd(rp, "Cl", dh), cn_p=dd(rp, "Cn", dh),
        cy_r=dd(rr, "CY", dh), cl_r=dd(rr, "Cl", dh), cn_r=dd(rr, "Cn", dh),
    )


def _flight_like(case: Case):
    """A Flight with the case's airspeed and density (altitude found from density)."""
    from . import aero as A
    from .atmosphere import isa

    lo, hi = -500.0, 6000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if isa(mid)[2] > case.rho:
            lo = mid
        else:
            hi = mid
    return A.Flight(case.velocity, (lo + hi) / 2)


def asb_sweep(case: Case) -> dict[str, np.ndarray]:
    """AeroSandbox CL, CDi, Cm (about the CG) over the same alpha sweep, elevons neutral."""
    from . import aero as A

    g, fl = case.geometry(), _flight_like(case)
    rs = [A.run_vlm(g, fl, a, xyz_ref=(case.x_cg, 0.0, 0.0)) for a in ALPHA_SWEEP_DEG]
    return dict(alpha=np.array(ALPHA_SWEEP_DEG), CL=np.array([r["CL"] for r in rs]),
                CDi=np.array([r["CD"] for r in rs]), CMy=np.array([r["Cm"] for r in rs]))


# --------------------------------------------------------------------------
# VSPAERO derivatives in the standard frame
# --------------------------------------------------------------------------


@dataclass
class Conventions:
    beta_sign: float  # +1: VSPAERO beta is the standard sideslip
    moment_sign: float  # multiplies CMx -> Cl and CMz -> Cn
    rate_sign: float  # multiplies d/dp and d/dr
    rates_axes: str  # "body" or "stability"
    notes: list = field(default_factory=list)


def conventions(stab: StabTable, ref_body: dict, alpha_deg: float) -> Conventions:
    """Work out VSPAERO's signs from derivatives whose sign is not in doubt.

    * side force with sideslip, CY_beta < 0 (the fins and the body all push that way) -> beta sign
    * dihedral effect Cl_beta, same sign as AeroSandbox's (sweep + dihedral, |Cl_beta| ~ 0.1) -> moment sign
    * roll damping Cl_p < 0 -> rate sign
    Whether p and r are body or stability rates is chosen by which reading agrees
    better with AeroSandbox on the cross terms (Cl_r, Cn_p), and reported.
    """
    v = stab.values
    notes = []
    cy_b = v["CFy"]["beta"]
    beta_sign = -1.0 if cy_b > 0 else 1.0
    cl_b = v["CMx"]["beta"] * beta_sign
    moment_sign = 1.0 if np.sign(cl_b) == np.sign(ref_body["cl_b"]) else -1.0
    cl_p = moment_sign * v["CMx"]["p"]
    rate_sign = 1.0 if cl_p < 0 else -1.0
    notes.append(f"CY_beta raw {cy_b:+.4f} -> beta sign {beta_sign:+.0f}; "
                 f"Cl_beta raw {v['CMx']['beta']:+.4f} -> moment sign {moment_sign:+.0f}; "
                 f"Cl_p raw {v['CMx']['p']:+.4f} -> rate sign {rate_sign:+.0f}")
    best, err = None, {}
    for axes in ("body", "stability"):
        c = Conventions(beta_sign, moment_sign, rate_sign, axes)
        dv = _vsp_stability_derivs(stab, c, alpha_deg)
        ref = to_stability_axes(ref_body, alpha_deg)
        err[axes] = sum(abs(dv[k] - ref[k]) / max(abs(ref[k]), 0.02) for k in ("cl_r", "cn_p", "cn_r"))
    best = min(err, key=err.get)
    notes.append("rates read as " + best + " axes (mismatch vs AeroSandbox on Cl_r, Cn_p, Cn_r: "
                 + ", ".join(f"{k} {e:.2f}" for k, e in err.items()) + ")")
    return Conventions(beta_sign, moment_sign, rate_sign, best, notes)


def _vsp_stability_derivs(stab: StabTable, c: Conventions, alpha_deg: float) -> dict:
    v = stab.values
    s_m, s_r, s_b = c.moment_sign, c.rate_sign, c.beta_sign
    raw = dict(
        cy_b=s_b * v["CFy"]["beta"], cl_b=s_m * s_b * v["CMx"]["beta"], cn_b=s_m * s_b * v["CMz"]["beta"],
        cy_p=s_r * v["CFy"]["p"], cl_p=s_m * s_r * v["CMx"]["p"], cn_p=s_m * s_r * v["CMz"]["p"],
        cy_r=s_r * v["CFy"]["r"], cl_r=s_m * s_r * v["CMx"]["r"], cn_r=s_m * s_r * v["CMz"]["r"],
    )
    if c.rates_axes == "body":
        return to_stability_axes(raw, alpha_deg)
    # rates already about stability axes: only the moments (body CMx, CMz) need rotating
    a = math.radians(alpha_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = dict(cy_b=raw["cy_b"], cy_p=raw["cy_p"], cy_r=raw["cy_r"])
    for k in ("b", "p", "r"):
        lk, nk = raw[f"cl_{k}"], raw[f"cn_{k}"]
        out[f"cl_{k}"] = lk * ca + nk * sa
        out[f"cn_{k}"] = nk * ca - lk * sa
    return out


def vsp_derivs(stab: StabTable, c: Conventions, alpha_deg: float) -> DerivSet:
    v = stab.values
    lat = _vsp_stability_derivs(stab, c, alpha_deg)
    return DerivSet(cl_a=v["CL"]["alpha"], cm_a=v["CMy"]["alpha"], cl_q=c.rate_sign * v["CL"]["q"],
                    cm_q=c.rate_sign * v["CMy"]["q"], **lat)


def asb_derivs(ref_body: dict, alpha_deg: float) -> DerivSet:
    lat = to_stability_axes(ref_body, alpha_deg)
    return DerivSet(cl_a=ref_body["cl_a"], cm_a=ref_body["cm_a"], cl_q=ref_body["cl_q"], cm_q=ref_body["cm_q"],
                    **lat)


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


@dataclass
class CaseResult:
    case: Case
    asb: dict  # sweep + derivatives
    vsp: dict  # method -> {"sweep": polar dict, "derivs": DerivSet, "conv": Conventions, "stab": StabTable}
    metrics: dict  # method -> comparison numbers
    dutch_roll: dict  # label -> damping
    modes: dict  # label -> Modes


def _slope(alpha_deg, y, lo=-2.0, hi=6.0):
    m = (alpha_deg >= lo - 1e-9) & (alpha_deg <= hi + 1e-9)
    return float(np.polyfit(np.radians(alpha_deg[m]), y[m], 1)[0])


def _at_cl(sweep: dict, cl: float, key: str) -> float:
    o = np.argsort(sweep["CL"])
    return float(np.interp(cl, sweep["CL"][o], sweep[key][o]))


def analyse_case(case: Case, runs_dir: str | Path, req: MissionRequirements, progress=print) -> CaseResult:
    from . import aero as A
    from . import stability as S
    from .weights import default_inertia

    g = case.geometry()
    fl = A.cruise_flight(req)
    progress(f"{case.label}: AeroSandbox reference ...")
    ref_sweep = asb_sweep(case)
    ref_body = asb_body_derivatives(case)
    ref = asb_derivs(ref_body, case.alpha_cruise_deg)

    # the optimizer's own (trimmed, with profile yaw damping) derivatives and modes
    model = A.longitudinal_model(g, fl)
    cruise = A.trimmed_point(g, model, case.x_cg, fl, DESIGN_MASS_KG)[0]
    d0 = S.derivatives(g, model, case.x_cg, fl, DESIGN_MASS_KG, cruise)
    inertia = default_inertia(g, DESIGN_MASS_KG, case.x_cg)
    modes = {"AeroSandbox (pipeline)": S.flight_modes(d0, inertia, g, fl)}

    cl_cruise = A.cl_level(DESIGN_MASS_KG, g, fl)
    metrics, vsp = {}, {}
    ref_cla = _slope(ref_sweep["alpha"], ref_sweep["CL"])
    ref_cma = _slope(ref_sweep["alpha"], ref_sweep["CMy"])
    ref_np = case.x_cg - ref_cma / ref_cla * g.mac_m
    metrics["AeroSandbox"] = dict(
        cl_alpha=ref_cla, x_np=ref_np, np_pct_mac=(ref_np - g.mac_le_x_m) / g.mac_m * 100,
        cdi_cruise=_at_cl(ref_sweep, cl_cruise, "CDi"), cl0=float(np.interp(0.0, ref_sweep["alpha"], ref_sweep["CL"])),
        **{k: getattr(ref, k) for k in DerivSet.LATERAL + ("cm_q",)})
    for m in METHODS:
        pol_p = find_output(runs_dir, f"{case.label}_{m}_sweep", ".polar")
        stab_p = find_output(runs_dir, f"{case.label}_{m}_stab", ".stab")
        if pol_p is None or stab_p is None:
            progress(f"{case.label}: no VSPAERO {m} output, skipped")
            continue
        sw = parse_polar(pol_p)
        st = parse_stab(stab_p)
        conv = conventions(st, ref_body, case.alpha_cruise_deg)
        dv = vsp_derivs(st, conv, case.alpha_cruise_deg)
        cla = _slope(sw["alpha"], sw["CL"])
        cma = _slope(sw["alpha"], sw["CMy"])
        x_np = case.x_cg - cma / cla * g.mac_m
        cdi = _at_cl(sw, cl_cruise, "CDi")
        metrics[f"VSPAERO {m}"] = dict(
            cl_alpha=cla, x_np=x_np, np_pct_mac=(x_np - g.mac_le_x_m) / g.mac_m * 100, cdi_cruise=cdi,
            cl0=float(np.interp(0.0, sw["alpha"], sw["CL"])),
            cl_alpha_stab=st.d("CL", "alpha"), x_np_stab=st.scalars.get("X_np"),
            **{k: getattr(dv, k) for k in DerivSet.LATERAL + ("cm_q",)})
        vsp[m] = dict(sweep=sw, derivs=dv, conv=conv, stab=st)
        # Dutch roll: pipeline derivatives + (VSPAERO - AeroSandbox) at the same untrimmed condition
        d1 = S.Derivatives(**asdict(d0))
        for k in DerivSet.LATERAL + ("cl_a", "cm_a", "cl_q", "cm_q"):
            setattr(d1, k, getattr(d1, k) + getattr(dv, k) - getattr(ref, k))
        modes[f"VSPAERO {m}"] = S.flight_modes(d1, inertia, g, fl)

    # PIPE-4 total-CD check: profile + misc drag from the pipeline, induced from each code
    cd_total_ref = cruise.cd
    for k, mt in metrics.items():
        mt["cd_total_cruise"] = cd_total_ref - metrics["AeroSandbox"]["cdi_cruise"] + mt["cdi_cruise"]
    dr = {k: v.dutch_roll.damping for k, v in modes.items()}
    return CaseResult(case, dict(sweep=ref_sweep, body=ref_body, derivs=ref), vsp, metrics, dr, modes)


def pipe4(res: CaseResult) -> dict:
    """PIPE-4 verdicts of each VSPAERO method against AeroSandbox."""
    a = res.metrics["AeroSandbox"]
    out = {}
    for k, m in res.metrics.items():
        if k == "AeroSandbox":
            continue
        out[k] = dict(
            cl_alpha=(m["cl_alpha"] / a["cl_alpha"] - 1, abs(m["cl_alpha"] / a["cl_alpha"] - 1) <= PIPE4["cl_alpha_rel"]),
            np=(m["np_pct_mac"] - a["np_pct_mac"], abs(m["np_pct_mac"] - a["np_pct_mac"]) <= PIPE4["np_pct_mac"]),
            cd=(m["cd_total_cruise"] / a["cd_total_cruise"] - 1,
                abs(m["cd_total_cruise"] / a["cd_total_cruise"] - 1) <= PIPE4["cd_cruise_rel"]),
        )
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

_LAT_NAMES = dict(cy_b="CY_beta", cl_b="Cl_beta", cn_b="Cn_beta", cy_p="CY_p", cl_p="Cl_p", cn_p="Cn_p",
                  cy_r="CY_r", cl_r="Cl_r", cn_r="Cn_r", cm_q="Cm_q")


def report(results: list[CaseResult]) -> str:
    L = ["Manta Mapper M1.7: VSPAERO check of the optimizer finalists", "=" * 62, ""]
    for r in results:
        c = r.case
        g = c.geometry()
        L.append(f"{c.label}: span {g.span_m:.3f} m, area {g.reference_area_m2:.4f} m^2, MAC {g.mac_m*1000:.0f} mm, "
                 f"CG {c.x_cg*1000:.0f} mm (SM 7.5 %), cruise {c.velocity:.1f} m/s, alpha {c.alpha_cruise_deg:.2f} deg")
        L.append("")
        cols = list(r.metrics)
        L.append(f"  {'':24s}" + "".join(f"{k:>18s}" for k in cols))
        rows = [("CL_alpha, /rad", "cl_alpha", 1, 3), ("CL at alpha 0", "cl0", 1, 4),
                ("neutral point, % MAC", "np_pct_mac", 1, 2), ("neutral point, mm", "x_np", 1000, 1),
                ("CDi at cruise CL", "cdi_cruise", 1, 5), ("CD total at cruise", "cd_total_cruise", 1, 5)]
        rows += [(f"{_LAT_NAMES[k]}", k, 1, 4) for k in DerivSet.LATERAL + ("cm_q",)]
        for name, key, mult, nd in rows:
            L.append(f"  {name:24s}" + "".join(f"{r.metrics[k][key]*mult:18.{nd}f}" for k in cols))
        L.append("")
        L.append("  PIPE-4 (VSPAERO vs AeroSandbox):")
        for k, v in pipe4(r).items():
            L.append(f"    {k:14s} CL_alpha {v['cl_alpha'][0]*100:+5.1f} % {'ok' if v['cl_alpha'][1] else 'FAIL'}   "
                     f"NP {v['np'][0]:+5.2f} % MAC {'ok' if v['np'][1] else 'FAIL'}   "
                     f"CD {v['cd'][0]*100:+5.1f} % {'ok' if v['cd'][1] else 'FAIL'}")
        L.append("")
        L.append("  Dutch roll damping (same inertia, profile yaw damping and trim; VSPAERO rows swap in its derivatives):")
        for k, m in r.modes.items():
            dr = m.dutch_roll
            flag = "ok" if dr.damping >= DUTCH_ROLL_MIN else "BELOW 0.09"
            L.append(f"    {k:24s} zeta {dr.damping:6.3f}  {dr.frequency_hz:5.2f} Hz   {flag}   "
                     f"spiral {m.spiral.time_constant_s:+7.1f} s   roll tau {-1/m.roll.eigenvalue.real:5.3f} s")
        for k, v in r.vsp.items():
            for n in v["conv"].notes:
                L.append(f"    [{k}] {n}")
        L.append("")
    return "\n".join(L)


def to_json(results: list[CaseResult]) -> dict:
    out = dict(milestone="M1.7", pipe4_limits=PIPE4, cases=[])
    for r in results:
        out["cases"].append(dict(
            label=r.case.label, x=r.case.x, metrics=r.metrics, pipe4={k: {kk: list(vv) for kk, vv in v.items()}
                                                                       for k, v in pipe4(r).items()},
            dutch_roll={k: dict(damping=m.dutch_roll.damping, hz=m.dutch_roll.frequency_hz,
                                spiral_s=m.spiral.time_constant_s) for k, m in r.modes.items()},
            conventions={k: dict(asdict(v["conv"])) for k, v in r.vsp.items()},
        ))
    return out


def plot_comparison(results: list[CaseResult], path: str | None = None):  # pragma: no cover - visual
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = {"AeroSandbox": "#444444", "VSPAERO vlm": "#2a6fdb", "VSPAERO panel": "#d9822b"}
    r = results[0]
    axes[0].plot(r.asb["sweep"]["alpha"], r.asb["sweep"]["CL"], "o-", color=colors["AeroSandbox"], label="AeroSandbox VLM")
    axes[1].plot(r.asb["sweep"]["alpha"], r.asb["sweep"]["CMy"], "o-", color=colors["AeroSandbox"])
    for m, v in r.vsp.items():
        axes[0].plot(v["sweep"]["alpha"], v["sweep"]["CL"], "s--", color=colors[f"VSPAERO {m}"], label=f"VSPAERO {m}")
        axes[1].plot(v["sweep"]["alpha"], v["sweep"]["CMy"], "s--", color=colors[f"VSPAERO {m}"])
    axes[0].set(xlabel="alpha, deg", ylabel="CL", title=f"{r.case.label}: lift")
    axes[1].set(xlabel="alpha, deg", ylabel="Cm about CG (SM 7.5 %)", title="pitching moment, elevons neutral")
    axes[0].legend(frameon=False)
    labels = [x.case.label for x in results]
    keys = list(results[0].dutch_roll)
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        axes[2].bar(np.arange(len(labels)) + i * w, [x.dutch_roll.get(k, np.nan) for x in results], w,
                    label=k, color=list(colors.values())[i % 3])
    axes[2].axhline(DUTCH_ROLL_MIN, color="#b00020", lw=1, ls="--", label="robust limit 0.09")
    axes[2].set(xticks=np.arange(len(labels)) + w * (len(keys) - 1) / 2, xticklabels=labels,
                ylabel="Dutch roll damping ratio", title="Dutch roll")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(prog="python -m manta_mapper.vspaero", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare", help="write OpenVSP scripts for the optimizer finalists")
    pa = sub.add_parser("analyse", help="compare VSPAERO output with the AeroSandbox VLM")
    for q in (pp, pa):
        q.add_argument("mission", help="mission TOML")
        q.add_argument("--optimizer", default="docs/optimizer_result.json")
        q.add_argument("-n", type=int, default=3, help="number of finalists")
    pp.add_argument("--out", default="vsp_runs")
    pa.add_argument("--runs", default="vsp_runs")
    pa.add_argument("--report")
    pa.add_argument("--json")
    pa.add_argument("--plots")
    a = p.parse_args(argv)
    req = MissionRequirements.from_toml(a.mission)
    result = json.loads(Path(a.optimizer).read_text())
    if a.cmd == "prepare":
        cases = build_cases(req, result, a.n)
        man = prepare(cases, a.out)
        print(f"wrote {len(man['runs'])} scripts to {a.out}")
        return
    man = json.loads((Path(a.runs) / "manifest.json").read_text())
    cases = [Case(**c) for c in man["cases"]][: a.n]
    results = [analyse_case(c, a.runs, req) for c in cases]
    text = report(results)
    print(text)
    if a.report:
        Path(a.report).write_text(text + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(to_json(results), indent=1, default=float))
    if a.plots:
        plot_comparison(results, str(Path(a.plots) / "vspaero_comparison.png"))


if __name__ == "__main__":  # pragma: no cover
    main()
