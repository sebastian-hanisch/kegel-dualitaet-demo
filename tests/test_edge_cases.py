"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import kdu_dual as D
import kdu_evaluation as ev
import kdu_forms as F
import kegel_kern as K
from kdu_scenario import make_cloud


def test_two_points_by_hand_for_the_forms_that_attain_their_optimum():
    """Punkte (1,2) und (-1,-2): Würfel R = 2, Raute R = 3 (l1-Abstand 6), Kugel R = sqrt 5; achsenparallele Ellipse: p1 + 4 p2 <= 1, min 1/p1 + 1/p2 bei p = (1/3, 1/6): 3 + 6 = 9."""
    pts = np.array([[1.0, 2.0], [-1.0, -2.0]])
    for form, value in (("cube", 2.0), ("diamond", 3.0), ("ball_soc", 5 ** 0.5), ("ball_psd", 5 ** 0.5), ("ellipse_axis", 9.0)):
        r = ev.solve_form_pair(form, pts)
        assert r.pair.primal_value == pytest.approx(value, abs=1e-5) and r.pair.dual_value == pytest.approx(value, abs=1e-5) and r.ok, form


def test_two_points_in_the_general_ellipse_have_an_infimum_that_is_not_attained():
    """Zwei Punkte gegenüber: nur eine Bedingung x^T P x <= 1, die andere Halbachse darf gegen null gehen: Infimum 5 nicht angenommen, das Primal läuft davon (P-Einträge ~ 1e9).
    Das Paar darf nicht als in Ordnung durchgehen: entweder findet die Suche keinen Dualstart, oder die Prüfung schlägt an und die Lücke ist deutlich größer null."""
    pts = np.array([[1.0, 2.0], [-1.0, -2.0]])
    try:
        r = ev.solve_form_pair("ellipse_psd", pts)
    except D.DualStartError as e:
        assert "kein streng zulässiger Dualstart" in str(e)
    else:
        assert not r.ok and r.pair.gap > 0.5 and abs(r.pair.primal_value - 5.0) < 1e-3


def test_identical_points_give_value_zero_in_primal_and_dual():
    pts = np.array([[1.0, 1.0]] * 4)
    for form in ("cube", "ball_soc", "ball_psd"):
        r = ev.solve_form_pair(form, pts - pts.mean(axis=0))
        assert abs(r.pair.primal_value) < 1e-7 and abs(r.pair.dual_value) < 1e-7, form


def test_collinear_points_use_the_outer_ones():
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]) - 1.5
    assert ev.solve_form_pair("ball_soc", pts).pair.dual_value == pytest.approx(3 * 2 ** 0.5 / 2, abs=1e-6)
    z = ev.solve_form_pair("ball_soc", pts).pair.z.reshape(4, 3)
    assert z[[0, 3], 0] == pytest.approx([0.5, 0.5], abs=1e-6) and np.abs(z[[1, 2], 0]).max() < 1e-6


def test_one_and_three_dimensions_work_in_every_form():
    p1 = np.array([[-2.0], [1.0], [3.0]])
    p3 = np.random.default_rng(5).normal(0, 1, (9, 3))
    for pts in (p1 - p1.mean(), p3 - p3.mean(axis=0)):
        for form in F.FORMS:
            r = ev.solve_form_pair(form, pts)
            assert r.ok and r.pair.gap < 1e-6, (form, pts.shape)
    assert ev.solve_form_pair("cube", p1 - p1.mean()).pair.dual_value == pytest.approx(2.5, abs=1e-6)


def test_inputs_are_validated():
    with pytest.raises(ValueError, match="unbekannte Form"):
        F.build_form("dreieck", np.zeros((3, 2)))
    with pytest.raises(ValueError, match="mindestens zwei Punkte"):
        make_cloud(1, 2.0, 0, 1)
    with pytest.raises(ValueError, match="eps muss positiv sein"):
        __import__("kdu_examples").gapex_problem(-1.0)


def test_central_path_stops_gracefully_and_short_paths_are_reported():
    prob = F.build_form("ball_soc", make_cloud(8, 3.0, 20, 1).points)
    assert len(D.central_path(prob, tol=1e3)) == 1                                  # ein Punkt genügt schon
    assert D.central_path(prob, max_rounds=3).__len__() == 3


def test_a_dual_that_leaves_the_cone_is_flagged_by_the_checker():
    prob = F.build_form("ball_soc", make_cloud(8, 3.0, 20, 1).points)
    r = D.solve_pair(prob)
    bad = r.z.copy()
    bad[0] = -1.0
    problems = D.check_dual(prob, bad)
    assert any("(dual) verletzt" in p for p in problems) and any("A^T z + c ist nicht null" in p for p in problems)
    assert D.check_dual(prob, r.z) == []


def test_start_status_reports_success_and_failure_for_known_cases():
    from kdu_examples import gapex_problem, nonattained_problem

    assert ev.start_status(F.build_form("ball_soc", make_cloud(8, 3.0, 20, 1).points))["found"]
    bad = ev.start_status(gapex_problem(0.1))
    assert not bad["found"] and "kein streng zulässiger Dualstart" in bad["message"] and bad["level"] is None
    assert ev.start_status(nonattained_problem((1.0, 1.0)))["found"] and not ev.start_status(nonattained_problem())["found"]
