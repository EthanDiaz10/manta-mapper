from pathlib import Path

import pytest

from manta_mapper.__main__ import report
from manta_mapper.requirements import MissionRequirements

ROOT = Path(__file__).resolve().parents[1]


def test_repo_mission_file_loads():
    req = MissionRequirements.from_toml(ROOT / "config" / "mission.toml")
    assert req.survey.area_min_ha == 25
    assert req.limits.max_stall_speed_m_s == 9.0
    assert req.survey.max_altitude_agl_m == pytest.approx(121.92)


def test_toml_overrides_and_rejects_typos(tmp_path):
    f = tmp_path / "m.toml"
    f.write_text("[survey]\narea_min_ha = 30\n")
    assert MissionRequirements.from_toml(f).survey.area_min_ha == 30
    f.write_text("[survey]\narea_min = 30\n")
    with pytest.raises(KeyError):
        MissionRequirements.from_toml(f)
    f.write_text("[surveys]\narea_min_ha = 30\n")
    with pytest.raises(KeyError):
        MissionRequirements.from_toml(f)


def test_report_runs():
    text = report(MissionRequirements(), mass_kg=1.5)
    assert "Design point" in text and "stretch" in text
