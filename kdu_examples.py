"""Drei kleine, von Hand gerechnete SDPs ohne Slater-Punkt und ihr LP-Gegenstück (nur numpy, Kern unverändert).

Allgemein: min c^T x unter M(x) = M0 + sum_j x_j M_j im PSD-Kegel; Kegelform s = svec(M0) - A x mit A = -[svec(M_j)].
Dual (Matrixform): max -<Z, M0> mit <Z, M_j> = c_j, Z im PSD-Kegel (Herleitung: c^T x >= c^T x - <Z, M(x)> = sum_j x_j (c_j - <Z, M_j>) - <Z, M0>).

1. Lückenbeispiel:      min x1  mit  [[eps, x1, 0],[x1, x2, 0],[0, 0, x1 + 1]] >= 0.   eps = 0: p* = 0, d* = -1 (Lücke 1); eps > 0: p = -1.
2. Nicht angenommen:     min x1  mit  [[x1, 1],[1, x2]] >= 0.                            inf = 0, nie erreicht; der zentrale Pfad existiert nicht (f_t fällt mit x2 ins Unendliche).
3. Schwach unzulässig:   [[x, 1],[1, 0]] >= 0.                                           nie erfüllt, aber der Abstand zum Kegel ist null (kleinster Eigenwert -1/x), kein Zertifikat."""

import math

import numpy as np

import kegel_kern as K

GAPEX_PRIMAL, GAPEX_DUAL = 0.0, -1.0      # von Hand, eps = 0


def sdp_problem(M0: np.ndarray, Ms: list, c, x0) -> K.ConeProblem:
    """Matrixform in Kegelform: s = svec(M0) - A x mit A = -[svec(M_j)] (Spalten)."""
    A = -np.stack([K.svec(M) for M in Ms], axis=1)
    return K.ConeProblem(np.array(c, float), A, K.svec(M0), (("s", M0.shape[0]),), np.array(x0, float))


# ----------------------------------------------------------------------------- 1. Beispiel mit Lücke

def gapex_matrices(eps: float) -> tuple:
    M0 = np.diag([eps, 0.0, 1.0])
    M1 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    M2 = np.diag([0.0, 1.0, 0.0])
    return M0, [M1, M2]


def gapex_problem(eps: float) -> K.ConeProblem:
    """Für eps > 0 streng zulässig (Start x = (0, 1)); für eps = 0 gibt es keinen Startpunkt, und der Aufruf ist ein Fehler."""
    if eps <= 0:
        raise ValueError("eps muss positiv sein: bei eps = 0 gibt es keinen streng zulässigen Punkt")
    M0, Ms = gapex_matrices(eps)
    return sdp_problem(M0, Ms, [1.0, 0.0], [0.0, 1.0])


def gapex_primal_feasible(x1: float, x2: float, tol: float = 1e-12) -> bool:
    """Zulässigkeit bei eps = 0: nur x1 = 0 (und x2 >= 0), denn der (1,1)-Eintrag ist null."""
    M0, (M1, M2) = gapex_matrices(0.0)
    return bool(np.linalg.eigvalsh(M0 + x1 * M1 + x2 * M2).min() >= -tol)


def gapex_dual_feasible(Z: np.ndarray, tol: float = 1e-12) -> bool:
    """Dual bei eps = 0: Z PSD, <Z, M1> = 1 (c_1) und <Z, M2> = 0 (c_2)."""
    _, (M1, M2) = gapex_matrices(0.0)
    return bool(np.linalg.eigvalsh(Z).min() >= -tol and abs(np.sum(Z * M1) - 1.0) <= tol and abs(np.sum(Z * M2)) <= tol)


def gapex_dual_value(Z: np.ndarray) -> float:
    """-<Z, M0> bei eps = 0: für jedes zulässige Z gleich -Z33 = -1."""
    return float(-np.sum(Z * gapex_matrices(0.0)[0]))


def gapex_sweep(eps_list, tol: float = 1e-4, max_newton: int = 400) -> list[dict]:
    """Primalwert der streng zulässigen Familie für jedes eps > 0: der Wert ist -1 (nicht p(0) = 0). Der Pfad läuft entlang der unbeschränkten optimalen Menge davon (x2 wächst)."""
    rows = []
    for eps in eps_list:
        res = K.solve_barrier(gapex_problem(eps), tol=tol, max_newton=max_newton)
        rows.append({"eps": eps, "value": res.value, "x1": float(res.x[0]), "x2": float(res.x[1]), "gap": res.gap_bound, "converged": res.converged, "steps": res.newton_steps})
    return rows


# ----------------------------------------------------------------------------- 2. nicht angenommenes Optimum, Pfad existiert nicht

def nonattained_problem(c=(1.0, 0.0)) -> K.ConeProblem:
    """min c1 x1 + c2 x2 mit [[x1, 1],[1, x2]] >= 0, Start (2, 2). Mit c = (1, 0): inf 0 nie erreicht; mit c = (1, 1): Optimum 2 bei (1, 1)."""
    return sdp_problem(np.array([[0.0, 1.0], [1.0, 0.0]]), [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])], list(c), [2.0, 2.0])


def nonattained_path(rounds: int = 10, newton_per_round: int = 6, mu: float = 8.0) -> list[dict]:
    """Je Runde (t mal mu) feste Zahl Newton-Schritte auf f_t = t x1 + phi: x1 -> 0, aber x2 wächst unbeschränkt (f_t hat kein Minimum)."""
    prob = nonattained_problem()
    x, t, rows = prob.x0.copy(), 1.0, []
    for r in range(rounds):
        dec2 = 0.0
        for _ in range(newton_per_round):
            g, h = K.newton_system(prob, x, t)
            dx, dec2 = K.newton_step(g, h)
            x, _ = K.backtrack(prob, x, dx, t, dec2)
        f = float(t * prob.c @ x + K.barrier_value(prob.cones, prob.b - prob.A @ x))
        rows.append({"round": r, "t": t, "x1": float(x[0]), "x2": float(x[1]), "f": f, "decrement2": float(dec2)})
        t *= mu
    return rows


# ----------------------------------------------------------------------------- 3. schwach unzulässig

def weak_min_eigenvalue(x: float) -> float:
    """Kleinster Eigenwert von [[x, 1],[1, 0]]: (x - sqrt(x^2 + 4)) / 2, nahe -1/x für große x; der Abstand zum PSD-Kegel geht gegen null, wird aber nie null."""
    return 0.5 * (x - math.sqrt(x * x + 4.0)) if x < 0 else -2.0 / (x + math.sqrt(x * x + 4.0))


def weak_certificate_value(Z: np.ndarray) -> float:
    """Zertifikatssuche für [[x,1],[1,0]] >= 0: Z PSD mit <Z, M1> = 0 (der Koeffizient von x, also Z11 = 0) und <Z, M0> < 0 wäre ein Zertifikat. Gibt <Z, M0> = 2 Z12 zurück."""
    return float(np.sum(Z * np.array([[0.0, 1.0], [1.0, 0.0]])))


def lp_farkas_example() -> dict:
    """LP x >= 1 und x <= 0 (A x >= b mit A = [[1], [-1]], b = [1, 0]): Zertifikat y >= 0 mit A^T y = 0 und b^T y > 0."""
    A, b, y = np.array([[1.0], [-1.0]]), np.array([1.0, 0.0]), np.array([1.0, 1.0])
    return {"A": A, "b": b, "y": y, "ATy": float((A.T @ y)[0]), "bTy": float(b @ y), "valid": bool(np.all(y >= 0) and abs((A.T @ y)[0]) < 1e-15 and b @ y > 0)}
