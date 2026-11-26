# Manta Mapper

A tailless blended-wing-body (BWB) flying-wing UAV for autonomous aerial
mapping, designed by an automated concept-design pipeline, checked with CFD,
drawn by script in Fusion 360, 3D printed, flown, and then compared against
its own predictions.

**Mission:** map 25 ha (stretch: 50 ha) per flight at 3 cm/px with a Raspberry
Pi Zero 2 W + Camera Module 3, land with 30 % battery left, hand launch, belly
landing, under 400 ft AGL in visual line of sight. Photos go to WebODM for an
orthomosaic, elevation model and 3D model.

## How the design is made

```
mission requirements ─▶ constraint analysis ─▶ parametric BWB geometry
                                                    │
     ┌──────────────── design-to-cost optimizer ◀───┤
     │   aero (NeuralFoil ▶ AeroSandbox VLM), weights, propulsion, stability
     ▼
 VSPAERO on top 2–3 ─▶ CFD on the final design ─▶ Fusion 360 CAD by script
                                                    │
                     print pod, hot-wire panels  ◀──┘ ─▶ fly ─▶ predicted vs measured
```

See [docs/requirements.md](docs/requirements.md) for the numbered requirements
and validation targets, and [docs/plan.md](docs/plan.md) for phases and
milestones.

## Status

Phase 1: requirements and constraint analysis (M1.1), airfoil selection
(M1.2), the parametric geometry (M1.3), the aero and stability analysis
(M1.4) and the weights and propulsion (M1.5) are working. The airfoils are S5010
on the outer panels and MH 78 on the centre body; [docs/airfoils.md](docs/airfoils.md)
explains why. [docs/geometry.md](docs/geometry.md) explains every shape
parameter and the baseline: 1.53 m span, 0.40 m² wing, 5° washout.
[docs/aero_stability.md](docs/aero_stability.md) covers aerodynamics and
stability (M1.4): neutral point, CG range, trim, drag, where the stall starts,
flight modes and the camera tilt.
[docs/weights_propulsion.md](docs/weights_propulsion.md) covers weights and
propulsion (M1.5): what every part weighs and where it goes, the motor, prop
and battery, and whether the 25 ha and 50 ha flights make it home with 30 %.

```bash
pip install -e ".[dev]"
python -m manta_mapper config/mission.toml --plot constraint_diagram.png
python -m manta_mapper.airfoils config/mission.toml --plots docs/img
python -m manta_mapper.geometry config/mission.toml --plots docs/img --vsp manta.vspscript
python -m manta_mapper.stability config/mission.toml --plots docs/img --report docs/stability_report.txt
python -m manta_mapper.weights config/mission.toml --plots docs/img --report docs/weights_report.txt
pytest
```

The report gives the survey geometry (altitude for 3 cm/px, photo footprint,
trigger interval, flight-line length), the constraint-chart design point
(wing loading and power loading), and a first-cut size for a guessed take-off
mass.

![Constraint diagram](docs/img/constraint_diagram.png)

![Baseline geometry](docs/img/geometry_views.png)

## Repository layout

```
config/mission.toml     the mission, as data
docs/                   requirements, plan, figures
src/manta_mapper/
  requirements.py       mission requirements and first-guess assumptions
  atmosphere.py         ISA atmosphere
  survey.py             GSD, footprint, overlap, flight lines, trigger rate
  constraints.py        wing-loading / power-loading constraint analysis, sizing
  airfoils.py           reflex airfoil comparison (NeuralFoil), gates, scores, plots
  geometry.py           parametric BWB: AeroSandbox model, reference values, checks, OpenVSP export
  aero.py               VLM + NeuralFoil strip drag, trim, stall (M1.4)
  stability.py          neutral point, CG band, flight modes, camera tilt (M1.4)
  propulsion.py         prop (BEMT), motor, Li-ion pack, parts search, mission energy (M1.5)
  weights.py            weight build-up, CG placement, inertia, M1.5 report (M1.5)
  optimize.py, cad.py, validation.py     (placeholders for later milestones)
tests/
```

## Credits

The idea of scripting a UAV concept design end to end was inspired by
openvsp-scripts by prahaas123. All code here is original.

## License

MIT
