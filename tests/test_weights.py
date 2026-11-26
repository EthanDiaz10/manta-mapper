"""M1.5 weight build-up: CG on target, parts fit, sensible mass, cost and inertia."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerosandbox")

from manta_mapper import weights as W  # noqa: E402
from manta_mapper.geometry import baseline  # noqa: E402
from manta_mapper.requirements import MissionRequirements  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REQ = MissionRequirements.from_toml(ROOT / "config" / "mission.toml")


@pytest.fixture(scope="module")
def bu():
    return W.build_up(baseline(REQ, 1.5), W.default_propulsion())


def test_cg_lands_on_target_without_ballast(bu):
    cg = bu.cg(True)
    assert cg[0] == pytest.approx(bu.choices.cg_target_x_m, abs=1e-4)
    assert abs(cg[1]) < 0.002  # left-right balanced
    assert bu.ballast_kg == 0.0
    lo, hi = bu.battery_range_m
    assert lo <= bu.battery_x_m <= hi


def test_whole_cg_band_reachable_by_sliding_the_pack(bu):
    """261-275 mm (M1.4 band) must be reachable without ballast."""
    g = bu.geom
    for x in (0.261, 0.275):
        b = W.build_up(g, bu.propulsion, W.BuildChoices(cg_target_x_m=x))
        assert b.ballast_kg == 0.0 and b.cg(True)[0] == pytest.approx(x, abs=1e-4)


def test_every_part_fits_the_pod(bu):
    assert bu.fit_problems == []


def test_mass_is_under_the_design_mass_with_growth(bu):
    m = bu.mass_kg(True)
    assert 1.0 <= m <= 1.5  # wing was sized at 1.5 kg
    assert bu.mass_kg(False) < m  # growth allowance applied
    assert m / bu.geom.reference_area_m2 <= REQ.limits.max_wing_loading_kg_m2


def test_cost_within_budget(bu):
    assert bu.cost_usd <= 600  # BUD-1


def test_mirrored_parts_are_symmetric(bu):
    for it in bu.items:
        if it.points.shape[0] > 1 and np.any(it.points[:, 2] != 0):
            assert it.points[:, 2].sum() == pytest.approx(0.0, abs=1e-9), it.name


def test_inertia_is_plausible(bu):
    """Radii of gyration: roll/yaw a fraction of the semispan, pitch a fraction of the pod chord."""
    i = bu.inertia()
    kx, ky, kz = i.radii_of_gyration_m
    s = bu.geom.semispan_m
    assert 0.2 * s < kx < 0.45 * s
    assert 0.08 < ky < 0.2
    assert i.izz >= max(i.ixx, i.iyy)  # flat aircraft
    scaled = bu.inertia(1.5)
    assert scaled.mass_kg == pytest.approx(1.5)
    assert scaled.radii_of_gyration_m == pytest.approx(i.radii_of_gyration_m)


def test_heat_proof_pod_is_the_default():
    """LW-PLA softens in a hot car; the default pod material must not."""
    assert W.MATERIALS[W.BuildChoices().pod_material].softens_c >= 90
