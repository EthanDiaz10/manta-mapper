"""M2.4 test prints: gauges follow the CAD's clearances, the check sheet judges entries, the mass roll-up."""

import csv
from pathlib import Path

import pytest

from manta_mapper import testprints as tp
from manta_mapper.cad import CadSettings

LOG = Path(__file__).resolve().parents[1] / "cad" / "testprints"


def test_ladders_put_the_cad_size_in_the_middle():
    st = CadSettings()
    L = {d["key"]: d for d in tp.ladders(st)}
    assert L["tube"]["sizes"][2] == pytest.approx(14.30)
    assert L["magnet"]["sizes"][2] == pytest.approx(10.20)
    assert L["pin"]["sizes"][2] == pytest.approx(3.30)
    assert L["m3"]["sizes"][2] == pytest.approx(3.40)
    assert tp.joint_ladder(st)[2] == pytest.approx(st.joint_clear_mm)
    for d in L.values():
        assert d["sizes"] == sorted(d["sizes"])


def _rows(**measured):
    rows = tp.check_rows(CadSettings())
    for r in rows:
        r["measured"] = measured.get(r["id"].replace("-", "_"), "")
    return rows


def test_checks_pass_fail_and_settings_advice():
    v, changes = tp.judge_checks(_rows(CAL_1="0.82", CAL_3="y", C2_A="n", G1_TUBE="14.40", G1_MAG="10.10",
                                       G2_JOINT="4", T_FIN="45", C2_E="420"))
    s = {x.id: x.status for x in v}
    assert s["CAL-1"] == "PASS" and s["CAL-3"] == "PASS" and s["G1-MAG"] == "PASS" and s["C2-E"] == "PASS"
    assert s["C2-A"] == "FAIL" and s["G1-TUBE"] == "FAIL" and s["G2-JOINT"] == "FAIL" and s["T-FIN"] == "FAIL"
    assert s["CAL-2"] == "---"
    assert changes["sleeve_bond_gap_mm"] == pytest.approx(0.25)  # (14.40 - 14) / 2 + 0.05 for the epoxy
    assert changes["joint_clear_mm"] == pytest.approx(0.15)


def _weigh():
    return tp.read_csv(LOG / "weigh_in.csv")


def test_blank_log_reproduces_the_cad_estimate():
    mr = tp.mass_rollup(_weigh())
    assert mr["density"] is None
    assert mr["aircraft_g"] == pytest.approx(1295, abs=3)
    assert 191 <= mr["battery_x_mm"] <= 231


def test_density_forecast_and_measured_parts():
    rows = _weigh()
    for r in rows:
        if r["id"] == "C0_skin_cal":
            r["measured_g"] = str(float(r["volume_cm3"]) * 0.56)  # foams better than assumed
        if r["id"] == "Battery" or r["item"].startswith("Battery"):
            r["measured_g"] = "230"
    mr = tp.mass_rollup(rows)
    assert mr["density"] == pytest.approx(0.56, rel=1e-3)
    assert mr["printed_now_g"] < mr["printed_target_g"]  # LW-ASA parts forecast lighter
    assert mr["status"] in ("GREEN", "AMBER")
    rows2 = _weigh()
    next(r for r in rows2 if r["kind"] == "total")["measured_g"] = "1460"
    assert tp.mass_rollup(rows2)["status"] == "RED"


def test_committed_logs_have_every_check_and_part():
    ids = {r["id"] for r in tp.read_csv(LOG / "fit_checks.csv")}
    assert ids == {r["id"] for r in tp.check_rows(CadSettings())}
    w = _weigh()
    assert {"pod_rear_R", "nacelle", "nose_cap", "TOTAL", "C0_skin_cal"} <= {r["id"] for r in w}
    for f in ("C0_skin_cal", "G1_fit_gauge", "joint_socket", "C1_pod_rear_R", "C2_root_face_R", "C4_pod_front_L"):
        assert (LOG / "stl" / f"{f}.stl").exists()


def test_report_runs_on_blank_logs():
    txt = tp.report(tp.read_csv(LOG / "fit_checks.csv"), _weigh())
    assert "not measured yet" in txt and "Battery centre" in txt


def test_coupons_are_closed_single_pieces_that_fit():
    pytest.importorskip("manifold3d")
    from manta_mapper import cad, cad_mesh
    from manta_mapper.structures import load_geometry

    lay = cad.layout(load_geometry())
    rec = cad.build_recipe(lay)
    built = cad_mesh.build(rec)
    stats = tp.coupon_stats(tp.coupons(rec, built, lay.st))
    for n, s in stats.items():
        assert s["closed"] and s["pieces"] == 1, n
        assert max(s["bbox"]) <= 280, n
    # coupons are short: all of them together are a small fraction of the pod
    assert sum(s["mass_g"] for s in stats.values()) < 150
