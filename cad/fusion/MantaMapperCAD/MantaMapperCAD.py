"""Manta Mapper M2.3: build the printable parts in Fusion 360 from the pipeline's recipe.

This script is deliberately thin. All the design logic (where the pod is
split, where the spar sleeves, hatches, tray, camera and motor go, how big
everything is) lives in ``src/manta_mapper/cad.py`` and is tested in plain
Python. The pipeline writes it out as ``cad/manta_cad.json``: a list of named
solids made from nine primitive operations, and a list of parts. This script
only interprets those operations:

* ``loft``: a Fusion loft through closed sketch sections (each a fitted spline
  from the trailing edge round the nose and back, closed by a short line);
* ``box`` / ``cyl``: TemporaryBRepManager primitives;
* ``prism``: a sketch polygon extruded;
* ``move`` / ``rotate`` / ``union`` / ``inter`` / ``diff``: TemporaryBRepManager
  transforms and booleans.

Each part becomes its own component (one base feature per part), and every
printed part is exported as an STL.

Run: Utilities > Add-Ins > Scripts and Add-Ins > + (My Scripts) > pick this
folder, then Run. Regenerate the JSON first if the design changed:

    python -m manta_mapper.cad --json cad/manta_cad.json
"""

import json
import math
import os
import time
import traceback

import adsk.core
import adsk.fusion

CM = 0.1  # the recipe is in mm; Fusion's internal length unit is the cm
HERE = os.path.dirname(os.path.abspath(__file__))
# Standalone download: manta_cad.json sits next to this script and the STLs
# go beside it. Inside the repo: cad/manta_cad.json and cad/stl_fusion.
_LOCAL_JSON = os.path.join(HERE, "manta_cad.json")
if os.path.exists(_LOCAL_JSON):
    DEFAULT_JSON = _LOCAL_JSON
    DEFAULT_STL = os.path.join(HERE, "stl_fusion")
else:
    DEFAULT_JSON = os.path.normpath(os.path.join(HERE, "..", "..", "manta_cad.json"))
    DEFAULT_STL = os.path.normpath(os.path.join(HERE, "..", "..", "stl_fusion"))


def P(p):
    return adsk.core.Point3D.create(p[0] * CM, p[1] * CM, p[2] * CM)


class Builder:
    """Evaluates recipe solids into TemporaryBRep bodies (cached; callers get copies)."""

    def __init__(self, root, solids, log):
        self.root = root
        self.solids = solids
        self.tbm = adsk.fusion.TemporaryBRepManager.get()
        self.cache = {}
        self.log = log

    def get(self, name):
        if name not in self.cache:
            self.cache[name] = self._make(name)
        return self.tbm.copy(self.cache[name])

    # --- operations -------------------------------------------------------
    def _make(self, name):
        d = self.solids[name]
        op = d["op"]
        tbm = self.tbm
        if op == "loft":
            return self._loft(d["sections"])
        if op == "box":
            lo, hi = d["lo"], d["hi"]
            c = [(a + b) / 2 for a, b in zip(lo, hi)]
            obb = adsk.core.OrientedBoundingBox3D.create(
                P(c), adsk.core.Vector3D.create(1, 0, 0), adsk.core.Vector3D.create(0, 1, 0),
                (hi[0] - lo[0]) * CM, (hi[1] - lo[1]) * CM, (hi[2] - lo[2]) * CM)
            return tbm.createBox(obb)
        if op == "cyl":
            return tbm.createCylinderOrCone(P(d["p0"]), d["r0"] * CM, P(d["p1"]), d["r1"] * CM)
        if op == "prism":
            return self._prism(d["poly"], d["z0"], d["z1"])
        if op == "move":
            b = self.get(d["of"])
            m = adsk.core.Matrix3D.create()
            m.translation = adsk.core.Vector3D.create(*[v * CM for v in d["v"]])
            tbm.transform(b, m)
            return b
        if op == "rotate":
            b = self.get(d["of"])
            m = adsk.core.Matrix3D.create()
            m.setToRotation(math.radians(d["deg"]), adsk.core.Vector3D.create(*d["axis"]), P(d["origin"]))
            tbm.transform(b, m)
            return b
        if op in ("union", "inter", "diff"):
            if op == "diff":
                target, tools = self.get(d["of"]), d["items"]
                kind = adsk.fusion.BooleanTypes.DifferenceBooleanType
            else:
                target, tools = self.get(d["items"][0]), d["items"][1:]
                kind = (adsk.fusion.BooleanTypes.UnionBooleanType if op == "union"
                        else adsk.fusion.BooleanTypes.IntersectionBooleanType)
            for t in tools:
                if not tbm.booleanOperation(target, self.get(t), kind):
                    self.log.append(f"{name}: boolean {op} with {t} failed")
            return target
        raise ValueError(f"unknown op {op}")

    def _loft(self, sections):
        root = self.root
        sketches = []
        loft_in = root.features.loftFeatures.createInput(adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        for sec in sections:
            sk = root.sketches.add(root.xYConstructionPlane)  # XY sketch: sketch space == model space
            sk.isComputeDeferred = True
            pts = adsk.core.ObjectCollection.create()
            for p in sec:
                pts.add(P(p))
            spline = sk.sketchCurves.sketchFittedSplines.add(pts)
            sk.sketchCurves.sketchLines.addByTwoPoints(spline.endSketchPoint, spline.startSketchPoint)
            sk.isComputeDeferred = False
            path = root.features.createPath(spline, True)
            loft_in.loftSections.add(path)
            sketches.append(sk)
        loft_in.isSolid = True
        feat = root.features.loftFeatures.add(loft_in)
        body = self.tbm.copy(feat.bodies.item(0))
        feat.deleteMe()
        for sk in sketches:
            sk.deleteMe()
        return body

    def _prism(self, poly, z0, z1):
        root = self.root
        sk = root.sketches.add(root.xYConstructionPlane)
        sk.isComputeDeferred = True
        lines = sk.sketchCurves.sketchLines
        pts = [adsk.core.Point3D.create(x * CM, y * CM, 0) for x, y in poly]
        first = prev = None
        for i in range(len(pts)):
            a = prev.endSketchPoint if prev else pts[0]
            b = pts[(i + 1) % len(pts)] if i < len(pts) - 1 else first.startSketchPoint
            prev = lines.addByTwoPoints(a, b)
            first = first or prev
        sk.isComputeDeferred = False
        prof = sk.profiles.item(0)
        ext = root.features.extrudeFeatures
        inp = ext.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        inp.startExtent = adsk.fusion.OffsetStartDefinition.create(adsk.core.ValueInput.createByReal(z0 * CM))
        inp.setDistanceExtent(False, adsk.core.ValueInput.createByReal((z1 - z0) * CM))
        feat = ext.add(inp)
        body = self.tbm.copy(feat.bodies.item(0))
        feat.deleteMe()
        sk.deleteMe()
        return body


def add_part(root, name, temp_body):
    occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    comp = occ.component
    comp.name = name
    base = comp.features.baseFeatures.add()
    base.startEdit()
    body = comp.bRepBodies.add(temp_body, base)
    base.finishEdit()
    body = comp.bRepBodies.item(comp.bRepBodies.count - 1)
    body.name = name
    return comp, body


def mirror_part(comp, body, name):
    """Left-hand copy of a right-hand reference part (mirrored about the XZ plane)."""
    objs = adsk.core.ObjectCollection.create()
    objs.add(body)
    inp = comp.features.mirrorFeatures.createInput(objs, comp.xZConstructionPlane)
    feat = comp.features.mirrorFeatures.add(inp)
    feat.bodies.item(0).name = name


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        path = DEFAULT_JSON
        if not os.path.exists(path):
            dlg = ui.createFileDialog()
            dlg.title = "Manta Mapper: pick cad/manta_cad.json"
            dlg.filter = "Recipe (*.json)"
            if dlg.showOpen() != adsk.core.DialogResults.DialogOK:
                return
            path = dlg.filename
        with open(path) as f:
            data = json.load(f)
        recipe = data["recipe"]
        parts = recipe["parts"]

        doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
        doc.name = "Manta Mapper M2.3"
        design = adsk.fusion.Design.cast(app.activeProduct)
        design.designType = adsk.fusion.DesignTypes.ParametricDesignType
        design.fusionUnitsManager.distanceDisplayUnits = adsk.fusion.DistanceUnits.MillimeterDistanceUnits
        root = design.rootComponent

        log = []
        builder = Builder(root, recipe["solids"], log)
        prog = ui.createProgressDialog()
        prog.isCancelButtonShown = True
        prog.show("Manta Mapper", "Building %v of %m: %p %", 0, len(parts), 1)
        t0 = time.time()
        made = []
        for i, p in enumerate(parts):
            if prog.wasCancelled:
                break
            prog.message = f"{p['name']} ({i + 1} of {len(parts)})"
            prog.progressValue = i
            adsk.doEvents()
            try:
                comp, body = add_part(root, p["name"], builder.get(p["solid"]))
                if not p["printed"]:
                    comp.isLightBulbOn = True
                    if p["name"].endswith("_R"):
                        mirror_part(comp, body, p["name"][:-2] + "_L")
                made.append((p, body))
            except Exception:
                log.append(f"{p['name']}: {traceback.format_exc().splitlines()[-1]}")
        prog.hide()

        # STLs of the printed parts
        out = DEFAULT_STL
        if ui.messageBox(f"Built {len(made)} of {len(parts)} parts in {time.time() - t0:.0f} s.\n\n"
                         f"Export the printed parts as STL to\n{out} ?", "Manta Mapper",
                         adsk.core.MessageBoxButtonTypes.YesNoButtonType) == adsk.core.DialogResults.DialogYes:
            os.makedirs(out, exist_ok=True)
            em = design.exportManager
            for p, body in made:
                if not p["printed"]:
                    continue
                opts = em.createSTLExportOptions(body, os.path.join(out, p["name"] + ".stl"))
                opts.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementHigh
                em.execute(opts)
        if log:
            ui.messageBox("Finished with problems:\n\n" + "\n".join(log[:30]), "Manta Mapper")
    except Exception:
        if ui:
            ui.messageBox("Failed:\n{}".format(traceback.format_exc()))
