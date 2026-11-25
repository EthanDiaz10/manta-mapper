# Manta Mapper plan

Four phases. Each ends with something that runs or flies, so progress is
visible and mistakes surface early.

## Phase 1: concept-design pipeline (through VSPAERO)

| Milestone | Done when |
|---|---|
| M1.1 Requirements + constraint analysis | `python -m manta_mapper` prints survey geometry, the constraint-chart design point and first-cut sizing; tests pass in CI (done) |
| M1.2 Airfoil selection | NeuralFoil polars for reflexed candidates at the real Reynolds numbers, clean and as built; S5010 outer, MH 78 centre ([airfoils.md](airfoils.md)) |
| M1.3 Parametric geometry | `geometry.py` builds the BWB (centre pod, swept outer panels, drooped tip fins) from 14 design variables; baseline 1.53 m, 0.40 m², 5° washout ([geometry.md](geometry.md)) (done) |
| M1.4 Aero + stability | AeroSandbox VLM gives CL, CD, Cm, neutral point; static margin and trim checked; spanwise CL vs local CL_max so the centre stalls before the tips (washout) |
| M1.5 Weights + propulsion | weight build-up with CG; motor/prop/Li-ion pack matched to the power loadings |
| M1.6 Optimizer | design-to-cost optimizer returns a feasible lowest-cost/lightest design |
| M1.7 VSPAERO check | top 2–3 designs exported to OpenVSP; VSPAERO agrees with VLM within PIPE-4 |

## Phase 2: CFD and generated CAD

| Milestone | Done when |
|---|---|
| M2.1 CFD | final design run in OpenFOAM (or ANSYS Student) at cruise and near stall; compared with VSPAERO |
| M2.2 Structure | spar sized for AC-10; FEA of the spar/centre-body joint |
| M2.3 Scripted CAD | Fusion 360 Python script builds the printable sections from the pipeline output |
| M2.4 Test prints | joints, spar fit and hatch printed and checked; print weights fed back into `weights.py` |

## Phase 3: build, fly, validate

| Milestone | Done when |
|---|---|
| M3.1 Build | airframe printed and assembled; mass and CG measured (VAL-4) |
| M3.2 Bench tests | camera trigger at ≥ 1 Hz for 40 min (CAM-1) and distance triggering (CAM-1b); failsafes (REG-5); Remote ID |
| M3.3 Glide test | manual hand-launched glide, trim and CG confirmed |
| M3.4 Manual flights | stall, glides for the drag polar, cruise power, pitch doublets (VAL-1 to VAL-7) |
| M3.5 Autonomous survey | 25 ha survey in AUTO, WebODM products (MIS-1, MIS-6) |
| M3.6 Correlation report | predicted vs measured for every VAL target, with what was learned |

## Phase 4: stretch

- Bungee catapult (design, build, launch-load check)
- 50 ha in one flight
- Export the pipeline model to ArduPilot SITL
- Autonomous landing

## Approach notes

- **Lowest fidelity that answers the question.** Cheap tools inside the
  optimizer loop; expensive ones (VSPAERO, CFD) only to check finalists.
- **Requirements are data.** One mission file drives everything; requirement IDs
  appear in the code next to the numbers they set.
- **Every assumption is a placeholder.** The constraint analysis starts from
  textbook guesses (CL_max, CD0, efficiencies). Each later module replaces one
  with a computed value, and flight tests check the result.
