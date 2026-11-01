"""Die drei Beispiele ohne Slater-Punkt und das LP-Gegenstück gegen Handrechnung (jede Zahl aus der Herleitung, nicht aus eingefrorenen eigenen Werten)."""

import numpy as np
import pytest

import kdu_examples as E
import kegel_kern as K


def test_hand_values_of_the_gap_example():
    assert (E.GAPEX_PRIMAL, E.GAPEX_DUAL) == (0.0, -1.0)


def test_gap_example_at_eps_zero_only_x1_zero_is_primal_feasible():
    """(1,1)-Eintrag null: [[0, x1],[x1, x2]] ist nur für x1 = 0 positiv semidefinit; x3-Zeile verlangt x1 >= -1."""
    assert E.gapex_primal_feasible(0.0, 3.0) and E.gapex_primal_feasible(0.0, 0.0)
    for x1 in (1e-3, -1e-3, 0.5, -0.5, 2.0, -0.9):
        for x2 in (0.0, 1.0, 100.0, 1e6):
            assert not E.gapex_primal_feasible(x1, x2) or abs(x1) * abs(x1) / max(x2, 1e-300) < 1e-12, (x1, x2)       # nur bei numerisch verschwindender Verletzung
    assert not E.gapex_primal_feasible(-1.5, 100.0)                                                                   # x1 + 1 < 0


def test_gap_example_dual_value_is_minus_one_for_every_feasible_matrix():
    rng = np.random.default_rng(0)
    for _ in range(100):
        e, a = rng.uniform(-3, 3), rng.uniform(0, 4)
        Z = np.array([[a + e * e, 0.0, e], [0.0, 0.0, 0.0], [e, 0.0, 1.0]])
        assert E.gapex_dual_feasible(Z) and E.gapex_dual_value(Z) == pytest.approx(-1.0, abs=1e-12)
    assert not E.gapex_dual_feasible(np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]))      # Z22 != 0
    assert not E.gapex_dual_feasible(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]))      # <Z, M1> = 2 != 1
    assert not E.gapex_dual_feasible(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]))      # nicht positiv semidefinit


def test_gap_example_family_has_value_minus_one_for_every_positive_eps():
    for eps in (1.0, 0.1, 0.01):
        res = K.solve_barrier(E.gapex_problem(eps), tol=1e-4, max_newton=400)
        assert res.value == pytest.approx(-1.0, abs=2e-3) and res.value >= -1.0 - 1e-9
        assert res.value - (-1.0) <= res.gap_bound + 1e-9                                   # die Lückenschranke deckt den Abstand zu -1 (dem wahren Wert für jedes eps > 0)
        assert res.x[1] > 1e3                                                              # x2 wächst: der Pfad läuft entlang der unbeschränkten optimalen Menge davon


def test_gap_problem_needs_a_positive_eps():
    with pytest.raises(ValueError, match="eps muss positiv sein"):
        E.gapex_problem(0.0)


def test_sweep_rows_have_the_documented_fields():
    rows = E.gapex_sweep((0.1, 0.01), tol=1e-3, max_newton=300)
    assert [r["eps"] for r in rows] == [0.1, 0.01] and all(set(r) == {"eps", "value", "x1", "x2", "gap", "converged", "steps"} for r in rows)


def test_nonattained_problem_by_hand():
    """x1 x2 >= 1: der Punkt (2, 2) ist streng zulässig (Determinante 3), (1, 1) liegt auf dem Rand (Determinante 0)."""
    p = E.nonattained_problem()
    assert np.linalg.det(K.smat(p.b - p.A @ np.array([2.0, 2.0]), 2)) == pytest.approx(3.0)
    assert np.linalg.det(K.smat(p.b - p.A @ np.array([1.0, 1.0]), 2)) == pytest.approx(0.0, abs=1e-12)
    assert p.c.tolist() == [1.0, 0.0] and E.nonattained_problem((1.0, 1.0)).c.tolist() == [1.0, 1.0]


def test_nonattained_path_diverges_and_f_t_has_no_minimum():
    rows = E.nonattained_path(8)
    x1, x2 = [r["x1"] for r in rows], [r["x2"] for r in rows]
    assert all(b < a for a, b in zip(x1, x1[1:])) and all(b > a for a, b in zip(x2, x2[1:]))          # x1 fällt, x2 wächst in jeder Runde
    assert x2[-1] > 1e6 and x1[-1] < 1e-5 and rows[-1]["round"] == 7
    # Handrechnung: df_t/dx2 = -x1/(x1 x2 - 1) < 0 für zulässige Punkte mit x1 > 0
    prob = E.nonattained_problem()
    for xv in ([1.0, 2.0], [0.1, 50.0], [0.01, 1e4]):
        g, _ = K.newton_system(prob, np.array(xv), 7.0)
        assert g[1] == pytest.approx(-xv[0] / (xv[0] * xv[1] - 1.0))


def test_nonattained_with_a_strictly_feasible_dual_has_the_optimum_two_at_one_one():
    res = K.solve_barrier(E.nonattained_problem((1.0, 1.0)), tol=1e-9)
    assert res.converged and res.value == pytest.approx(2.0, abs=1e-7) and res.x == pytest.approx([1.0, 1.0], abs=1e-6)       # x1 + x2 >= 2 sqrt(x1 x2) >= 2


def test_weak_minimum_eigenvalue_matches_numpy_and_goes_to_zero():
    for x in (-3.0, 0.0, 0.5, 10.0, 1e3, 1e6):
        assert E.weak_min_eigenvalue(x) == pytest.approx(np.linalg.eigvalsh(np.array([[x, 1.0], [1.0, 0.0]])).min(), rel=1e-6, abs=1e-12)
    assert E.weak_min_eigenvalue(1e8) == pytest.approx(-1e-8, rel=1e-6)
    assert all(E.weak_min_eigenvalue(x) < 0 for x in (-5.0, 0.0, 1.0, 1e12))


def test_weak_infeasible_system_has_no_certificate_and_the_value_is_zero_for_every_candidate():
    """Z mit Z11 = 0 ist nur PSD, wenn Z12 = 0; dann ist <Z, M0> = 0 (nie negativ)."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        Z = np.array([[0.0, 0.0], [0.0, rng.uniform(0, 5)]])
        assert E.weak_certificate_value(Z) == 0.0 and np.linalg.eigvalsh(Z).min() >= 0
    assert not np.linalg.eigvalsh(np.array([[0.0, -0.5], [-0.5, 1.0]])).min() >= 0           # ein Z12 != 0 mit Z11 = 0 ist nicht PSD


def test_lp_farkas_certificate_by_hand():
    f = E.lp_farkas_example()
    assert f["valid"] and f["ATy"] == 0.0 and f["bTy"] == 1.0
    assert f["A"].T @ f["y"] == pytest.approx(0.0) and f["b"] @ f["y"] > 0
    # kein x erfüllt A x >= b: x >= 1 und -x >= 0
    for x in np.linspace(-5, 5, 41):
        assert not np.all(f["A"][:, 0] * x >= f["b"])


def test_sdp_problem_builder_maps_matrices_to_the_cone_form():
    p = E.sdp_problem(np.array([[1.0, 0.0], [0.0, 1.0]]), [np.array([[1.0, 0.0], [0.0, 0.0]])], [1.0], [0.0])
    assert p.cones == (("s", 2),) and p.b.tolist() == pytest.approx(K.svec(np.eye(2)).tolist())
    s = p.b - p.A @ np.array([0.5])                   # M(x) = M0 + x M1 = I + 0,5 E11 = diag(1,5; 1)
    assert K.smat(s, 2) == pytest.approx(np.diag([1.5, 1.0]))
