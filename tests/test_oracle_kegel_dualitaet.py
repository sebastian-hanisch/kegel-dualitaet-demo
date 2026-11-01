"""Orakel (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): Primal und Dual aller sechs Formen und die Beispiele ohne Slater gegen einen anderen Rechenweg.

- cvxpy mit Clarabel: der Primalwert jeder Form; der Dualwert der Demo muss ihn treffen (starke Dualität, anderer Löser)
- eine unabhängige Kugel in der Ebene (Kreise durch zwei oder drei Punkte) für Radius, Stützpunkte und deren Gewichte (Stück 1: Σ λ x = c, Σ λ = 1)
- das Dual von Hand für den Würfel (LP): Summe der Betragsgewichte
- Eigenschaften, die auch ein Fehler in beiden Rechenwegen nicht überlebt (Komplementarität, Dual des Duals, Skalierung, Drehinvarianz)

Fehlt cvxpy im Test-venv, überspringen sich diese Tests (pytest.importorskip), sie raten nie."""

import itertools
import math

import numpy as np
import pytest

import kdu_dual as D
import kdu_evaluation as ev
import kdu_forms as F
import kegel_kern as K
from kdu_scenario import make_cloud, rotation


def _cloud(seed, n, d=2):
    pts = np.random.default_rng(seed).normal(0.0, 1.0, (n, d))
    return pts - pts.mean(axis=0)


def _circle(pts):
    """Kleinster Kreis um alle Punkte (Mittelpunkt, Radius) über alle Kreise durch zwei (Durchmesser) und drei Punkte (Umkreis)."""
    best = (math.inf, None)
    for i, j in itertools.combinations(range(len(pts)), 2):
        c, r = (pts[i] + pts[j]) / 2, np.linalg.norm(pts[i] - pts[j]) / 2
        if r < best[0] and np.all(np.linalg.norm(pts - c, axis=1) <= r + 1e-9):
            best = (r, c)
    for i, j, k in itertools.combinations(range(len(pts)), 3):
        a, b, c3 = pts[i], pts[j], pts[k]
        d = 2 * (a[0] * (b[1] - c3[1]) + b[0] * (c3[1] - a[1]) + c3[0] * (a[1] - b[1]))
        if abs(d) < 1e-12:
            continue
        ux = ((a @ a) * (b[1] - c3[1]) + (b @ b) * (c3[1] - a[1]) + (c3 @ c3) * (a[1] - b[1])) / d
        uy = ((a @ a) * (c3[0] - b[0]) + (b @ b) * (a[0] - c3[0]) + (c3 @ c3) * (b[0] - a[0])) / d
        c = np.array([ux, uy])
        r = np.linalg.norm(a - c)
        if r < best[0] and np.all(np.linalg.norm(pts - c, axis=1) <= r + 1e-9):
            best = (r, c)
    return best[1], best[0]


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_dual_value_hits_the_cvxpy_primal_value_for_all_six_forms():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel fehlt")
    kw = dict(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    for seed in range(8):
        d = 2 + seed % 2
        pts = _cloud(seed, 8 + seed % 4, d)
        c, R = cp.Variable(d), cp.Variable()
        refs = {}
        for form, norm in (("cube", "inf"), ("diamond", 1), ("ball_soc", 2), ("ball_psd", 2)):
            cp.Problem(cp.Minimize(R), [cp.norm(p - c, norm) <= R for p in pts]).solve(**kw)
            refs[form] = float(R.value)
        pv = cp.Variable(d, nonneg=True)
        cp.Problem(cp.Minimize(cp.sum(cp.inv_pos(pv))), [(pts ** 2) @ pv <= 1]).solve(**kw)
        refs["ellipse_axis"] = float(np.sum(1 / pv.value))
        M = cp.Variable((d, d), PSD=True)
        cp.Problem(cp.Minimize(cp.tr_inv(M)), [cp.sum(cp.multiply(np.outer(p, p), M)) <= 1 for p in pts]).solve(**kw)
        refs["ellipse_psd"] = float(np.trace(np.linalg.inv(M.value)))
        for form in F.FORMS:
            r = ev.solve_form_pair(form, pts).pair
            tol = 1e-5 if form.startswith("ellipse") else 1e-6
            assert r.dual_value == pytest.approx(refs[form], rel=tol), (form, seed)
            assert r.primal_value == pytest.approx(refs[form], rel=tol), (form, seed)


def test_ball_dual_weights_and_support_match_an_independent_circle():
    for seed in range(20):
        pts = _cloud(seed, 6 + seed % 10)
        c_ref, r_ref = _circle(pts)
        fp = ev.solve_form_pair("ball_soc", pts)
        z = fp.pair.z.reshape(len(pts), 3)
        assert fp.pair.dual_value == pytest.approx(r_ref, rel=1e-6)
        dist = np.linalg.norm(pts - c_ref, axis=1)
        support = np.where(dist >= r_ref * (1 - 1e-7))[0]
        lam = np.linalg.lstsq(np.vstack([pts[support].T, np.ones(len(support))]), np.append(c_ref, 1.0), rcond=None)[0]
        assert z[support, 0] == pytest.approx(lam, abs=1e-6), seed                                 # Dualgewichte = Gewichte aus Stück 1
        assert np.delete(z[:, 0], support).max(initial=0.0) < 1e-6                                 # komplementärer Schlupf
        assert abs(z[:, 0].sum() - 1.0) < 1e-9 and np.abs(z[:, 1:].sum(axis=0)).max() < 1e-9      # Simplex und Schwerpunkt


def test_cube_dual_by_hand_is_the_sum_of_absolute_weights():
    """LP: min R mit |x_ij - c_j| <= R. Das Dual wählt je Koordinate die beiden extremen Punkte; Dualwert = halbe größte Spannweite; Gewicht nur auf diesen Zeilen."""
    for seed in range(10):
        pts = _cloud(seed, 8, 3)
        r = ev.solve_form_pair("cube", pts).pair
        assert r.dual_value == pytest.approx(((pts.max(axis=0) - pts.min(axis=0)) / 2).max(), rel=1e-7)
        assert r.z.min() >= -1e-12 and r.z.sum() == pytest.approx(1.0, abs=1e-8)                    # Orthant: z >= 0, die R-Gleichung verlangt Summe 1


def test_complementarity_is_zero_at_the_optimum_and_nu_over_t_on_the_path():
    prob = F.build_form("ball_psd", _cloud(1, 9))
    path = D.central_path(prob)
    for p in path:
        z = D.dual_point(prob, p.x, p.t)
        assert z @ (prob.b - prob.A @ p.x) == pytest.approx(K.barrier_parameter(prob.cones) / p.t, rel=1e-6)
    r = D.solve_pair(prob)
    assert r.complementarity < 1e-8 and abs(r.gap - r.complementarity) < 1e-8


def test_scaling_laws_for_primal_and_dual():
    pts = _cloud(2, 10)
    for form in F.FORMS:
        a, b = ev.solve_form_pair(form, pts).pair, ev.solve_form_pair(form, 3.0 * pts).pair
        exponent = 2 if form in F.ELLIPSES else 1
        assert b.dual_value == pytest.approx(3.0 ** exponent * a.dual_value, rel=1e-6), form
        assert b.gap < 1e-7 and a.gap < 1e-7


def test_rotation_leaves_the_ball_dual_unchanged():
    pts = make_cloud(14, 4.0, 0, 7).points
    turned = pts @ rotation(45).T
    assert ev.solve_form_pair("ball_soc", turned).pair.dual_value == pytest.approx(ev.solve_form_pair("ball_soc", pts).pair.dual_value, rel=1e-6)


def test_the_dual_of_the_dual_program_returns_the_primal_value():
    """Das Dual als Kegelprogramm hat selbst ein Dual; dessen Wert ist der Wert des Programms (ohne die Konstante b^T z0 des Nullraum-Ansatzes)."""
    prob = F.build_form("ball_soc", _cloud(3, 8))
    start = D.find_dual_start(prob)
    dprob = D.dual_program(prob, start)
    pair = D.solve_pair(dprob)
    assert pair.gap < 1e-7 and pair.primal_value == pytest.approx(pair.dual_value, abs=1e-7)
    primal = K.solve_barrier(prob, tol=1e-10).value
    assert primal == pytest.approx(-(prob.b @ start.z0 + pair.primal_value), abs=1e-6)                 # p* = -min b^T z = -(b^T z0 + min c'^T y)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_the_gap_example_at_eps_zero_has_dual_value_minus_one_in_cvxpy_too():
    """Dual von Hand (Z22 = 0 => Z12 = 0 => Z33 = 1) gegen cvxpy: max -<Z, M0> unter den zwei Gleichungen und Z PSD gibt -1 (das Dual selbst ist lösbar, es hat nur keinen streng zulässigen Punkt)."""
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel fehlt")
    Z = cp.Variable((3, 3), PSD=True)
    M0 = np.diag([0.0, 0.0, 1.0]); M1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]]); M2 = np.diag([0.0, 1.0, 0.0])
    prob = cp.Problem(cp.Maximize(-cp.sum(cp.multiply(Z, M0))), [cp.sum(cp.multiply(Z, M1)) == 1, cp.sum(cp.multiply(Z, M2)) == 0])
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        pytest.skip("Clarabel scheitert am Dual ohne Slater-Punkt (versionsabhängig)")
    assert prob.value == pytest.approx(-1.0, abs=1e-5)
