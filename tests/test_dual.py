"""Das Kegel-Dual (`kdu_dual.py`) je Einheit gegen Handrechnung und Mini-Programme (DEMO-PLAYBOOK §1b)."""

import numpy as np
import pytest

import kdu_dual as D
import kdu_examples as E
import kdu_forms as F
import kegel_kern as K
from kdu_scenario import HAND_POINTS


def test_dual_point_by_hand_for_the_orthant():
    """min x mit x >= 1 (s = x - 1): bei x = 3 ist s = 2, z = -grad phi / t = (1/s) / t = 1/(2t); A^T z = -z = -c ... gilt nur im zentrierten Punkt."""
    prob = K.ConeProblem(np.array([1.0]), np.array([[-1.0]]), np.array([-1.0]), (("l", 1),), np.array([2.0]))
    z = D.dual_point(prob, np.array([3.0]), 4.0)
    assert z.tolist() == [pytest.approx(1.0 / (2.0 * 4.0))] and z @ (prob.b - prob.A @ np.array([3.0])) == pytest.approx(1.0 / 4.0)      # z^T s = nu / t mit nu = 1
    # zentriert bei t: t c - A^T grad phi(s) = 0  <=>  t + (-1)*(-1/s)... -> s = 1/t, x = 1 + 1/t
    t = 4.0
    z_c = D.dual_point(prob, np.array([1.0 + 1.0 / t]), t)
    assert D.dual_residual(prob, z_c) == pytest.approx(0.0, abs=1e-12) and z_c[0] == pytest.approx(1.0)                       # A^T z = -c: z = 1


def test_dual_point_in_the_second_order_and_psd_cone_is_interior_with_the_right_complementarity(tiny_cones):
    for name in ("soc", "sdp"):
        prob, _ = tiny_cones[name]
        x, t = np.array([prob.x0[0]]), 5.0
        z = D.dual_point(prob, x, t)
        assert D.cone_margin_all(prob.cones, z) > 0
        assert z @ (prob.b - prob.A @ x) == pytest.approx(K.barrier_parameter(prob.cones) / t)


def test_dual_residual_and_cone_margin_by_hand():
    prob = K.ConeProblem(np.array([1.0, 2.0]), np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.zeros(3), (("l", 3),), np.zeros(2))
    z = np.array([-1.0, -2.0, 0.0])                      # A^T z = (-1, -2) = -c
    assert D.dual_residual(prob, z) == 0.0
    assert D.dual_residual(prob, np.array([0.0, 0.0, 0.0])) == 2.0
    assert D.cone_margin_all((("l", 2), ("q", 3)), np.array([1.0, 0.5, 5.0, 3.0, 4.0])) == pytest.approx(0.0)       # Lorentz-Kegel am Rand, Orthant 0,5
    assert D.cone_margin_all((("l", 2), ("q", 3)), np.array([1.0, -0.5, 6.0, 3.0, 4.0])) == pytest.approx(-0.5)


def test_check_dual_names_each_violation():
    prob = K.ConeProblem(np.array([1.0]), np.array([[-1.0]]), np.array([-1.0]), (("l", 1),), np.array([2.0]))
    assert D.check_dual(prob, np.array([1.0])) == []                                          # A^T z = -1 = -c, z >= 0
    bad_eq = D.check_dual(prob, np.array([2.0]))
    assert len(bad_eq) == 1 and "A^T z + c ist nicht null" in bad_eq[0]
    bad_cone = D.check_dual(prob, np.array([-1.0]))
    assert any("(dual) verletzt" in p for p in bad_cone)


def test_nullspace_and_correction_by_hand():
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    N = D.nullspace_of_transpose(A)
    assert N.shape == (3, 1) and np.abs(A.T @ N).max() < 1e-12 and np.linalg.norm(N) == pytest.approx(1.0)
    prob = K.ConeProblem(np.array([1.0, 2.0]), A, np.zeros(3), (("l", 3),), np.zeros(2))
    z0 = D.correct_to_equation(prob, np.array([0.3, 0.2, 0.1]))
    assert A.T @ z0 == pytest.approx(-prob.c, abs=1e-12)
    assert D.correct_to_equation(prob, np.array([-1.0, -2.0, 0.0])) == pytest.approx([-1.0, -2.0, 0.0])         # schon zulässig: keine Änderung


def test_block_complementarity_sums_to_the_gap():
    pts = HAND_POINTS
    prob = F.build_form("ball_soc", pts)
    r = D.solve_pair(prob)
    blocks = D.block_complementarity(prob, r.primal.x, r.z)
    assert len(blocks) == 4 and sum(blocks) == pytest.approx(r.complementarity, abs=1e-14)
    assert r.complementarity == pytest.approx(r.gap, abs=1e-8) and max(abs(b) for b in blocks) < 1e-8


@pytest.mark.parametrize("form,value", [("cube", 1.0), ("diamond", 2.0), ("ball_soc", 2 ** 0.5), ("ball_psd", 2 ** 0.5), ("ellipse_axis", 4.0), ("ellipse_psd", 2.5)])
def test_pair_on_the_hand_instance(form, value):
    prob = F.build_form(form, HAND_POINTS)
    r = D.solve_pair(prob)
    assert r.primal_value == pytest.approx(value, abs=1e-6) and r.dual_value == pytest.approx(value, abs=1e-6)
    assert 0 <= r.gap < 1e-7 and r.residual < 1e-9 and D.check_dual(prob, r.z) == []


def test_ball_dual_by_hand_has_weights_one_half_on_the_diameter():
    """Handinstanz: die Punkte (1,1) und (-1,-1) liegen auf dem Rand (R = sqrt 2); Dual: z_i0 = 1/2 dort, null sonst, Summe z_i = 0 (Mittelpunkt im Ursprung), Wert sqrt 2."""
    prob = F.build_form("ball_soc", HAND_POINTS)
    r = D.solve_pair(prob)
    z = r.z.reshape(4, 3)
    assert z[:2, 0] == pytest.approx([0.5, 0.5], abs=1e-6) and np.abs(z[2:, 0]).max() < 1e-6
    assert np.abs(z[:, 1:].sum(axis=0)).max() < 1e-9
    assert -np.sum(HAND_POINTS * z[:, 1:]) == pytest.approx(2 ** 0.5, abs=1e-6)                     # Dualwert -sum x_i^T zbar_i


def test_dual_program_has_the_documented_structure():
    prob = F.build_form("ball_soc", HAND_POINTS)
    start = D.find_dual_start(prob)
    dual = D.dual_program(prob, start)
    assert dual.cones == prob.cones and dual.A.shape == (12, 9) and dual.x0.tolist() == [0.0] * 9     # 12 Zeilen, Nullraum 12 - 3 = 9
    assert dual.b == pytest.approx(start.z0) and dual.c == pytest.approx(start.null.T @ prob.b)
    assert np.isfinite(K.barrier_value(dual.cones, dual.b - dual.A @ dual.x0))                       # y = 0 ist streng zulässig
    assert prob.A.T @ start.z0 == pytest.approx(-prob.c, abs=1e-12)


def test_every_dual_point_of_the_program_satisfies_the_equation_exactly():
    prob = F.build_form("ball_psd", HAND_POINTS)
    start = D.find_dual_start(prob)
    dual = D.dual_program(prob, start)
    rng = np.random.default_rng(3)
    for _ in range(20):
        y = rng.normal(size=dual.A.shape[1]) * 1e-3
        z = dual.b - dual.A @ y
        assert D.dual_residual(prob, z) < 1e-12


def test_weak_duality_on_pairs_from_different_path_levels():
    """Primalpunkte vom Pfad bei verschiedenen t, Dualpunkte z0 + N y mit kleinem y: c^T x + b^T z >= 0 (schwache Dualität), am Rand der Rundung."""
    prob = F.build_form("ball_soc", HAND_POINTS)
    path = D.central_path(prob)
    start = D.find_dual_start(prob, path)
    dual = D.dual_program(prob, start)
    rng = np.random.default_rng(1)
    for p in path[:8]:
        for _ in range(5):
            z = dual.b - dual.A @ (rng.normal(size=dual.A.shape[1]) * 1e-4)
            if D.cone_margin_all(prob.cones, z) <= 0:
                continue
            assert prob.c @ p.x + prob.b @ z >= -1e-9


def test_central_path_gap_falls_by_the_factor_eight_and_levels_are_searched_in_order(monkeypatch):
    prob = F.build_form("ball_soc", HAND_POINTS)
    path = D.central_path(prob)
    gaps = [p.gap for p in path]
    assert all(a / b == pytest.approx(8.0) for a, b in zip(gaps, gaps[1:])) and gaps[-1] <= 1e-6
    start = D.find_dual_start(prob, path)
    assert start.level == 1e-3 and start.margin > 0 and start.residual_before >= 0
    only_coarse = D.find_dual_start(prob, path, levels=(1e-1,))
    assert only_coarse.level == 1e-1


def test_no_dual_start_without_a_strictly_feasible_dual():
    for prob in (E.nonattained_problem(), E.gapex_problem(0.1)):
        with pytest.raises(D.DualStartError, match="kein streng zulässiger Dualstart"):
            D.find_dual_start(prob)
    assert D.find_dual_start(E.nonattained_problem((1.0, 1.0))).level > 0


def test_a_too_short_path_is_reported_per_level():
    prob = F.build_form("ball_soc", HAND_POINTS)
    short = D.central_path(prob, tol=10.0)
    with pytest.raises(D.DualStartError, match="Pfad zu kurz"):
        D.find_dual_start(prob, short, levels=(1e-9,))
