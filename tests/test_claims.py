"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über feste Wolken (SWEEP_SEEDS), nie über einen einzelnen Seed als Befund; die Standardwolke der Primäransicht (Seed 35) ist ausdrücklich eine einzelne.
Bänder statt Punktwerte, wo ein Zufallsstrom oder eine Bibliotheksversion die letzte Stelle bewegt (CI installiert ungepinnt, Linux rundet anders als Windows)."""

from statistics import median

import numpy as np
import pytest

import kdu_constants as C
import kdu_evaluation as ev
import kdu_forms as F
from kdu_scenario import HAND_POINTS, make_cloud


@pytest.fixture(scope="module")
def std():
    return ev.analyse(ev.Settings())


@pytest.fixture(scope="module")
def runs():
    """Alle sechs Formen, Primal und Dual, auf 15 festen Wolken (5 Seeds, 3 Zuschnitte)."""
    out = {f: [] for f in F.FORMS}
    for seed in C.SWEEP_SEEDS:
        for n, stretch, angle in ((20, 4.0, 45), (12, 2.0, 10), (30, 5.0, 80)):
            pts = make_cloud(n, stretch, angle, seed).points
            for f in F.FORMS:
                out[f].append((n, ev.solve_form_pair(f, pts)))
    return out


def test_hand_instance_primal_and_dual_agree_in_all_six_forms():
    for form, value in (("cube", 1.0), ("diamond", 2.0), ("ball_soc", 2 ** 0.5), ("ball_psd", 2 ** 0.5), ("ellipse_axis", 4.0), ("ellipse_psd", 2.5)):
        r = ev.solve_form_pair(form, HAND_POINTS)
        assert r.pair.primal_value == pytest.approx(value, abs=1e-6) and r.pair.dual_value == pytest.approx(value, abs=1e-6) and r.ok, form


def test_standard_cloud_numbers(std):
    rows = {r["form"]: r for r in ev.pair_rows(std)}
    assert {f: round(r["primal"], 3) for f, r in rows.items()} == pytest.approx({"cube": 5.784, "diamond": 10.090, "ball_soc": 7.167, "ball_psd": 7.167, "ellipse_axis": 134.265, "ellipse_psd": 78.517}, abs=0.002)
    assert all(abs(r["primal"] - r["dual"]) < 1e-7 and r["gap"] < 1e-7 and r["residual"] < 1e-10 and r["ok"] for r in rows.values())     # Lücke < 1e-7 wie bei den 90 Läufen (README: bis 3,4e-8); die allgemeine Ellipse (PSD) liegt mit Randabstand ~1e-7 so nah am Rand, dass die letzte Stelle von der CPU abhängt
    assert {f: r["dual_variables"] for f, r in rows.items()} == {"cube": 77, "diamond": 57, "ball_soc": 57, "ball_psd": 117, "ellipse_axis": 22, "ellipse_psd": 24}
    assert {f: r["rows"] for f, r in rows.items()} == {"cube": 80, "diamond": 100, "ball_soc": 60, "ball_psd": 120, "ellipse_axis": 26, "ellipse_psd": 30}
    assert {f: r["level"] for f, r in rows.items()}["ball_soc"] == 1e-3 and rows["ellipse_axis"]["level"] > 1e-3


def test_dual_variables_equal_rows_minus_variables(runs):
    for f, items in runs.items():
        for n, fp in items:
            assert fp.pair.start.null.shape[1] == fp.problem.A.shape[0] - np.linalg.matrix_rank(fp.problem.A), (f, n)


def test_all_90_pairs_close_the_gap_exactly_satisfy_the_equation_and_pass_the_checks(runs):
    """README: 90 Läufe (6 Formen, 15 Wolken): Lücke höchstens 1e-7, Residuum der Dualitätsgleichung höchstens 1e-10, beide Prüfungen bestehen."""
    items = [fp for rs in runs.values() for _, fp in rs]
    assert len(items) == 90 and all(fp.ok for fp in items)
    assert not any(fp.pair.primal.far_from_path or fp.pair.dual.far_from_path for fp in items)     # README: der Kern meldet in diesen 90 Läufen nie „weit vom Pfad“
    assert max(fp.pair.gap for fp in items) < 1e-7 and min(fp.pair.gap for fp in items) > -1e-8
    assert max(fp.pair.residual for fp in items) < 1e-10


def test_start_levels_by_form(runs):
    """README: Würfel, Raute und beide Kugeln starten immer bei ν/t ≈ 1e-3; die Ellipse achsenparallel braucht in 9 von 15 Wolken ein gröberes Niveau (1e-2 oder 1e-1), die allgemeine in 1 von 15."""
    levels = {f: [fp.pair.start.level for _, fp in rs] for f, rs in runs.items()}
    for f in ("cube", "diamond", "ball_soc", "ball_psd"):
        assert set(levels[f]) == {1e-3}, f
    coarse_axis = sum(l > 1e-3 for l in levels["ellipse_axis"])
    coarse_psd = sum(l > 1e-3 for l in levels["ellipse_psd"])
    assert 4 <= coarse_axis <= 13 and coarse_psd <= 4 and coarse_axis > coarse_psd


def test_newton_steps_of_the_dual_program(runs):
    """README: das Dual braucht 8 bis 22 Newton-Schritte (Würfel, Raute, Kugeln) und bis 67 (Ellipsen)."""
    steps = {f: [fp.pair.dual.newton_steps for _, fp in rs] for f, rs in runs.items()}
    for f in ("cube", "diamond", "ball_soc"):
        assert 5 <= min(steps[f]) and max(steps[f]) <= 20, f
    assert 8 <= min(steps["ball_psd"]) and max(steps["ball_psd"]) <= 30
    assert max(steps["ellipse_axis"]) <= 100 and max(steps["ellipse_psd"]) <= 60 and median(steps["ellipse_axis"]) > median(steps["ball_soc"])


def test_the_dual_read_from_the_path_end_is_not_feasible(std):
    """README: z = -grad phi / t erfüllt A^T z = -c nur bei mäßigem t: bei ν/t ≈ 4e-4 liegt das Residuum bei feinem Zentrieren bei 9e-9 (unter 1e-6) und bei grobem bei 1e-4 (über 1e-5), am Pfadende (ν/t ≈ 1e-11) über 1e-6."""
    rows = ev.path_quality("ball_soc", std.cloud.points)
    mid = ev.quality_level_row(rows)
    assert 1e-4 <= mid["gap"] <= 1e-2 and mid["residual_fine"] < 1e-6 and mid["residual_coarse"] > 1e-5 and mid["residual_coarse"] > 100 * mid["residual_fine"]
    assert rows[-1]["gap"] < 1e-9 and rows[-1]["residual_fine"] > 1e-6 and rows[-1]["residual_coarse"] > 1e-6
    assert all(b["gap"] < a["gap"] for a, b in zip(rows, rows[1:]))


def test_ball_dual_weights_are_the_weights_of_stueck_1_on_the_standard_cloud(std):
    data = ev.complementarity_data(std.cloud.points)
    assert data["ok"] and len(data["support"]) == 2 and data["exact"] == pytest.approx([0.5, 0.5])
    assert data["max_dev_support"] < 1e-8 and data["weight_outside"] < 1e-8 and abs(data["sum_weights"] - 1.0) < 1e-9 and data["centre_residual"] < 1e-9


def test_ball_dual_weights_on_the_fixed_round_clouds_with_two_or_three_supports():
    seen = set()
    for seed in C.SWEEP_SEEDS:
        data = ev.complementarity_data(make_cloud(20, 1.2, 0, seed).points)
        seen.add(len(data["support"]))
        assert data["ok"] and data["max_dev_support"] < 1e-6 and data["weight_outside"] < 1e-6, seed
    assert seen <= {2, 3} and 3 in seen or seen == {2}


def test_gap_example_numbers():
    res = ev.gapex_experiment()
    assert res["primal_value"] == 0.0 and res["dual_value"] == -1.0 and res["dual_min"] == pytest.approx(-1.0) and res["dual_max"] == pytest.approx(-1.0)
    assert res["feasible"] == 1000 and res["primal_x1_max"] == 0.0 and res["dual_samples"] == 200
    assert all(abs(r["value"] + 1.0) < 2e-3 for r in res["sweep"]) and all(r["x2"] > 1e8 for r in res["sweep"]) and max(r["gap"] for r in res["sweep"]) == pytest.approx(7.3e-4, rel=0.02)
    assert (res["starts_found"], res["starts_total"]) == (0, 5)


def test_nonattained_numbers():
    res = ev.nonattained_experiment()
    first, last = res["rows"][0], res["rows"][-1]
    assert first["x1"] == pytest.approx(1.0, abs=1e-3) and 50 < first["x2"] < 120 and last["x1"] < 1e-6 and 1e7 < last["x2"] < 1e9 and len(res["rows"]) == 10
    assert res["bounded_converged"] and res["bounded_value"] == pytest.approx(2.0, abs=1e-7) and res["bounded_x"] == pytest.approx([1.0, 1.0], abs=1e-6) and 15 <= res["bounded_steps"] <= 40
    assert not res["start_original"]["found"] and res["start_bounded"]["found"]


def test_weak_infeasibility_numbers():
    res = ev.weak_experiment()
    assert [r["x"] for r in res["rows"]] == list(C.WEAK_XS) and res["rows"][-1]["eig"] == pytest.approx(-1e-8, rel=1e-5)
    assert res["cert_best"] == 0.0 and res["cert_samples"] == 500 and res["farkas"]["valid"] and res["farkas"]["bTy"] == 1.0


def test_the_dual_start_succeeds_at_every_slider_corner():
    """README: an den Reglerecken (n = 8, 20, 30; Streckung 1,2 und 6; Winkel 0, 45, 90; 108 Läufe) gelingt der Dualstart immer und beide Prüfungen bestehen (Regression: Streckung 6, 45 Grad, n = 20, Ellipse achsenparallel)."""
    count = 0
    for n in (8, 20, 30):
        for stretch in (1.2, 6.0):
            for angle in (0, 45, 90):
                pts = make_cloud(n, stretch, angle, 35).points
                for f in F.FORMS:
                    r = ev.solve_form_pair(f, pts)
                    assert r.ok and r.pair.gap < 1e-6, (n, stretch, angle, f)
                    count += 1
    assert count == 108
