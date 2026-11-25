# Manta Mapper requirements

Every requirement has an ID, a value (minimum and, where it applies, stretch),
and a way to check it. "Shall" means pass/fail; "should" is a goal.
The numbers the code uses live in [`config/mission.toml`](../config/mission.toml)
and `src/manta_mapper/requirements.py`, tagged with the same IDs.

Verification methods: **A** analysis (pipeline), **T** test (bench or flight),
**I** inspection, **D** demonstration.

## 1. Mission (MIS)

| ID | Requirement | Minimum | Stretch | Verify |
|---|---|---|---|---|
| MIS-1 | Area mapped in one flight | 25 ha | 50 ha | A, then D (flight) |
| MIS-2 | Ground sample distance (GSD) at image centre | ≤ 3.0 cm/px | — | A, T (VAL-9) |
| MIS-2a | GSD is never relaxed to save the camera. If CAM-1 falls short, in order: (1) forward overlap 75 → 70 % (MIS-3 minimum), (2) Camera Module 3 Wide. A coarser GSD is not a fallback: with the standard lens, 4 cm/px needs 135 m, above REG-2 | 3.0 cm/px kept | — | A |
| MIS-3 | Photo overlap, forward / side | plan 75 % / 65 %; never below 70 % / 60 % | 80 % / 70 % | A, T (WebODM report) |
| MIS-3a | Camera mounted with the sensor's long side along the flight track ("portrait"): photos ~1.8× further apart, so about half the trigger rate, for ~1.7× more flight-line length. Switch to landscape only if CAM-1 proves ≥ 1.4 Hz | yes | — | I |
| MIS-4 | Battery left on landing after the full mission incl. 4 min of launch, climb, transit and landing | ≥ 30 % of usable energy | — | A, T (battery log) |
| MIS-4a | Battery sized for the **stretch** area (50 ha) with the MIS-4 reserve, × 1.25 for optimistic drag/efficiency guesses. Predicted: 25 ha lands with ~66 %, 50 ha with ~44 % | ≥ 1.25 × predicted | — | A, T |
| MIS-4b | Design atmosphere: field elevation 1220 m (NMSU area, ~4000 ft MSL), ISA + 25 °C (~32 °C) | — | — | A |
| MIS-5 | Survey still meets MIS-2/3 in steady wind | 5 m/s | 7 m/s | A, T |
| MIS-6 | Products | orthomosaic, DSM, 3D model from WebODM | — | D |
| MIS-7 | Fully autonomous survey from launch climb-out to approach (ArduPilot AUTO) | yes | autonomous landing | D |

## 2. Regulations and operations (REG)

| ID | Requirement | Value | Verify |
|---|---|---|---|
| REG-1 | Fly under FAA recreational rules (49 USC 44809): FAA registration, TRUST certificate, CBO safety guidelines | yes | I |
| REG-2 | Max height | ≤ 400 ft (121.9 m) AGL; geofence ceiling set below it | A, I (fence params), T (log) |
| REG-3 | Visual line of sight for the whole flight | survey block within ~500 m of the pilot | A (plan), D |
| REG-4 | Broadcast Remote ID (standard RID or RID module) | yes | I |
| REG-5 | Failsafes: RC loss → RTL, low battery → RTL, geofence breach → RTL | configured and bench-tested | T (SITL + bench) |

## 3. Aircraft (AC)

| ID | Requirement | Value | Verify |
|---|---|---|---|
| AC-1 | Stall speed at max take-off mass in the MIS-4b design atmosphere | ≤ 9 m/s (hand launch) | A, T (VAL-3) |
| AC-2 | Wing loading | ≤ 6 kg/m² (hand launch) | A, I (weigh) |
| AC-3 | Max take-off mass | ≤ 2.0 kg | A, I |
| AC-4 | Configuration: tailless blended-wing-body, swept outer panels, drooped tip fins, pusher prop, outer-panel elevons, fixed-wing only | yes | I |
| AC-5 | Climb rate at full power | ≥ 2.5 m/s | A, T |
| AC-6 | Climb gradient right after hand launch at 1.2 Vs | ≥ 0.15 (8.5°) | A, T |
| AC-7 | Static margin at cruise CG | 5–10 % MAC | A, T (VAL-6) |
| AC-8 | Self-trimming: trims at cruise with elevon deflection | ≤ ±3° | A, T (log) |
| AC-9 | Airspeed window from 1.3 Vs up to the camera limit (CAM-1a rate × photo spacing − wind), with cruise inside it | ≥ 3 m/s wide | A |
| AC-10 | Structure: spar + skin carry launch, gust and landing loads | 4 g limit, 1.5 safety factor | A (FEA), T (wingtip load test) |
| AC-11 | Removable outer panels; replaceable nose, tips and skid | yes | I |
| AC-12 | Payload bay fits camera, Pi, battery with top hatch access | yes | I (CAD) |
| AC-13 | Belly landing on a sacrificial skid without damage beyond the skid | 5 landings | T |
| AC-14 | Launch method | hand launch; bungee catapult is a stretch goal | D |

## 4. Camera (CAM)

| ID | Requirement | Value | Verify |
|---|---|---|---|
| CAM-1 | Pi Zero 2 W + Camera Module 3 sustains full-resolution stills | ≥ 1 Hz for 40 min without dropped frames | T (bench test, before first flight) |
| CAM-1a | The flight plan uses at most 70 % of the CAM-1 rate on the downwind leg (baseline needs 0.55 Hz at 14 m/s in 5 m/s wind) | ≤ 0.7 × CAM-1 | A |
| CAM-1b | Trigger by distance (ArduPilot `CAM_TRIGG_DIST`), not by time, so photo spacing holds on upwind and downwind legs | yes | I (params), T (log) |
| CAM-2 | Forward-motion blur at the downwind ground speed | ≤ 0.5 px (shutter ≤ 1/1300 s at 3 cm, 19 m/s) | A, T (inspect images) |
| CAM-3 | Every photo is geotagged from ArduPilot CAM messages | ≥ 99 % of photos | T |
| CAM-4 | Camera points straight down in cruise | within 3° | I, T (log pitch) |

## 5. Design pipeline (PIPE)

| ID | Requirement | Verify |
|---|---|---|
| PIPE-1 | All code original (openvsp-scripts by prahaas123 is approach-only inspiration) | I |
| PIPE-2 | One mission file drives the whole pipeline; changing it re-sizes the aircraft | D |
| PIPE-3 | Fidelity ladder: NeuralFoil/XFOIL → AeroSandbox VLM → VSPAERO (top 2–3) → CFD (final) | D |
| PIPE-4 | Each rung agrees with the next within: CL_α 10 %, CD at cruise 20 %, neutral point 3 % MAC | A |
| PIPE-5 | Design-to-cost optimizer returns the lowest-cost/lightest design that meets all "shall" requirements | A |
| PIPE-6 | Fusion 360 CAD is generated by script from the pipeline output, no hand-edited geometry | D |
| PIPE-7 | Tests pass in CI on every PR | I |

## 6. Validation targets (VAL): predicted vs measured

A target passes when the measured value falls inside the band around the
pipeline prediction for the as-built aircraft (as-built mass and CG).

| ID | Quantity | How it is measured | Pass if |
|---|---|---|---|
| VAL-1 | Drag polar (sink rate vs airspeed) | motor-off glides at 4+ speeds, airspeed sensor + baro | sink rate within 15 %; max L/D within 15 % |
| VAL-2 | Cruise power | battery current × voltage in steady level cruise | within 15 % |
| VAL-3 | Stall speed | slow-down at altitude, airspeed at break | within 1 m/s |
| VAL-4 | Mass and CG | scale, balance | mass within 10 % of weight build-up; CG within 5 mm |
| VAL-5 | Trim elevon deflection at cruise | log | within 2° |
| VAL-6 | Short-period mode (pitch doublets) | IMU pitch rate | frequency within 25 %, damping ratio within 0.1 |
| VAL-7 | Hand-launch speed and climb-out | airspeed + GPS after release | speed ≥ 1.1 Vs; climb gradient within 25 % |
| VAL-8 | Coverage rate and endurance | log, flight plan | ha/min within 15 %; endurance within 15 % |
| VAL-9 | Achieved GSD | WebODM report | within 10 % of planned |
| VAL-10 | Map accuracy | tape-measured distances between ≥ 5 ground markers vs map | every distance within 2 × GSD |
| VAL-11 | Trigger rate | photo count and timestamps | no gaps > 1.5 × planned interval |

## 7. Budget (BUD)

| ID | Requirement | Value |
|---|---|---|
| BUD-1 | Aircraft hardware (excl. radio, charger, tools) | target $450–600 |
| BUD-2 | Software | free / student licences only |

## Open items

- CAM-1 bench test result: if below 1 Hz, apply MIS-2a; if ≥ 1.4 Hz, consider landscape mounting (MIS-3a) to shorten flights.
- Hybrid printed centre body + foam outer panels vs fully printed (affects weights and cost).
- Possible extra: export the pipeline model to ArduPilot SITL.
