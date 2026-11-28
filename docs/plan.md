# Manta Mapper plan

Four phases. Each ends with something that runs or flies, so progress is
visible and mistakes surface early.

## Phase 1: concept-design pipeline (through VSPAERO)

| Milestone | Done when |
|---|---|
| M1.1 Requirements + constraint analysis | `python -m manta_mapper` prints survey geometry, the constraint-chart design point and first-cut sizing; tests pass in CI (done) |
| M1.2 Airfoil selection | NeuralFoil polars for reflexed candidates at the real Reynolds numbers, clean and as built; S5010 outer, MH 78 centre ([airfoils.md](airfoils.md)) |
| M1.3 Parametric geometry | `geometry.py` builds the BWB (centre pod, swept outer panels, drooped tip fins) from 14 design variables; baseline 1.53 m, 0.40 m², 5° washout ([geometry.md](geometry.md)) (done) |
| M1.4 Aero + stability | VLM + NeuralFoil strip drag: neutral point 33.6 % MAC, CG band 261–275 mm, trims within ±3° at cruise, stall 8.7 m/s starting inboard, short period 1.8 Hz; camera mount tilted 6° ([aero_stability.md](aero_stability.md)) (done) |
| M1.5 Weights + propulsion | 1.20 kg as built (EPP panels, LW-ASA pod), CG on 261 mm with no ballast, real inertia; X2216 1100KV + 10x6 folding + 3S1P P45B; 25 ha lands with ≥ 51 %, 50 ha with ≥ 33 % except at 1.5 kg and L/D 10; ~$470 ([weights_propulsion.md](weights_propulsion.md)) (done) |
| M1.6 Optimizer | design-to-cost optimizer returns a feasible lowest-cost/lightest design. Re-run after the M1.7 rate fix, span bound raised to 2.4 m: 1.84 m span, AR 8.0, 1182 g as built, $470, Dutch roll 0.093, stall 8.71 m/s ([optimizer.md](optimizer.md)) (done) |
| M1.7 VSPAERO check | top 3 designs run in VSPAERO 3.57 (VLM; panel mesh broken). Lift and drag agree, neutral point 3 % MAC forward (borderline PIPE-4). Found and fixed a rate-derivative bug: Dutch roll is really ~0.07, below the 0.09 margin. Re-checked on the 1.84 m re-run optimum: Dutch roll 0.11 with VSPAERO derivatives (0.07 pessimistic), NP still 3 % MAC forward, so CG band moves 8 mm forward ([vspaero.md](vspaero.md)) (done) |

## Phase 2: CFD and generated CAD

| Milestone | Done when |
|---|---|
| M2.1 CFD | final design run in OpenFOAM (or ANSYS Student) at cruise and near stall; compared with VSPAERO. OpenFOAM v2406, k-ω SST, 0° to 14°: NP 279 mm (34.8 % MAC), between AeroSandbox and VSPAERO, so the M1.4 CG band holds; cruise CD 0.028 fully turbulent (upper bound); fin junction clean; stall starts inboard as predicted, CL_max not resolvable by SST at this Re ([cfd.md](cfd.md)) (done) |
| M2.2 Structure | spar sized for AC-10 and AC-13: telescoping pultruded 12x10 > 10x8 > 8x6 tubes in a 14x12 roll-wrapped sleeve, carbon strip carry-through; set by fin tip strikes and gusts (+5.7 g), not the 4 g manoeuvre; +32 g, +$27 ([structures.md](structures.md)) (done, analytic; joint FEA replaced by the M2.4 pull test and the AC-10 wingtip load test) |
| M2.3 Scripted CAD | Fusion 360 Python script builds the printable sections from the pipeline output. The layout, cut search, payload packing and checks are in `cad.py`; one recipe drives the Fusion script and a manifold3d mesh build. Four LW-ASA half-sections + TPU nose fit the K2 Pro with 22 mm spare; 23 of 23 checks pass; printed parts +67 g on M1.5 ([cad.md](cad.md)) (Fusion script ran first time on the laptop, 2026-11-28) |
| M2.4 Test prints | joints, spar fit and hatch printed and checked; a pod section and an EPP panel weighed and fed back into `weights.py` |

## Phase 3: build, fly, validate

| Milestone | Done when |
|---|---|
| M3.1 Build | pod printed, panels hot-wired and laminated, assembled; mass and CG measured (VAL-4) |
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
