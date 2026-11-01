"""Kegel-Dual im selben Format (nur numpy, benutzt den gemeinsamen Kern `kegel_kern.py` unverändert).

Primal:  min c^T x  mit  s = b - A x  im Kegel K.      Dual:  max -b^T z  mit  A^T z = -c,  z im Kegel K  (Orthant, Lorentz- und PSD-Kegel sind selbstdual).
Schwache Dualität: für zulässige x und z gilt c^T x + b^T z = z^T s >= 0; die Lücke ist die Komplementarität z^T s.

Das Dual wird als eigenes Kegelprogramm gelöst: z = z0 + N y mit N als Basis des Nullraums von A^T (Gleichungen exakt), der Start z0 stammt aus dem zentralen Pfad des Primals
(z = -grad phi(s) / t am Pfadpunkt, danach auf A^T z0 = -c korrigiert). Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b)."""

import contextlib
from dataclasses import dataclass

import numpy as np

import kegel_kern as K

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - threadpoolctl ist optional
    threadpool_limits = None

LEVELS = (1e-3, 1e-2, 1e-1, 1e-4, 1e-5, 1.0, 10.0)    # Lückenniveaus nu/t, bei denen ein Dualstart versucht wird (Reihenfolge der Versuche; die groben am Ende sind die Rückfälle)


def single_thread():
    """Kontext, der die BLAS-Threads auf eins setzt: die Matrizen sind klein (bis etwa 120 x 120), und mehrere Threads kosten auf einem ausgelasteten Rechner das Zwanzigfache (gemessen: 2,8 s statt 0,12 s)."""
    return threadpool_limits(limits=1) if threadpool_limits is not None else contextlib.nullcontext()


class DualStartError(ValueError):
    """Kein streng zulässiger Dualstart gefunden (z.B. weil das Dual selbst keinen streng zulässigen Punkt hat)."""


@dataclass(frozen=True, eq=False)
class PathPoint:
    t: float
    x: np.ndarray
    gap: float          # nu / t


@dataclass(frozen=True, eq=False)
class DualStart:
    z0: np.ndarray           # streng im Kegel, A^T z0 = -c bis auf Rundung
    null: np.ndarray         # Basis des Nullraums von A^T (Spalten)
    level: float             # gewünschtes Lückenniveau nu/t
    t: float
    residual_before: float   # ||A^T z + c|| vor der Korrektur
    margin: float            # Randabstand von z0 im Kegel (> 0)


@dataclass(frozen=True, eq=False)
class PairResult:
    primal: K.BarrierResult
    dual: K.BarrierResult
    z: np.ndarray
    primal_value: float      # c^T x
    dual_value: float        # -b^T z
    gap: float               # primal_value - dual_value = c^T x + b^T z
    complementarity: float   # z^T s
    residual: float          # ||A^T z + c||_inf nach dem Lösen (exakt über den Nullraum)
    start: DualStart


# ----------------------------------------------------------------------------- Pfad und Dualpunkt

def central_path(problem: K.ConeProblem, tol: float = 1e-6, mu: float = 8.0, center_tol: float = 1e-10, max_rounds: int = 40) -> list[PathPoint]:
    """Zentraler Pfad des Primals: je Runde ein (t, x) mit fest zentriertem x, bis nu / t <= tol oder die Rechengenauigkeit endet."""
    nu = K.barrier_parameter(problem.cones)
    x, t, out = problem.x0.copy(), K.initial_t(problem), []
    for _ in range(max_rounds):
        x_new, _, ok = K.center(problem, x, t, tol=center_tol)
        if not ok:
            break
        x = x_new
        out.append(PathPoint(t, x.copy(), nu / t))
        if nu / t <= tol:
            break
        t *= mu
    return out


def dual_point(problem: K.ConeProblem, x: np.ndarray, t: float) -> np.ndarray:
    """z = -grad phi(b - A x) / t: liegt im Inneren des Kegels, und z^T s = nu / t (Gleichung A^T z = -c gilt nur, wenn x genau zentriert ist)."""
    s = problem.b - problem.A @ x
    z = np.zeros_like(s)
    for kind, m, nb, off in K.groups(problem.cones):
        d = K.cone_dim(kind, m)
        g, _ = K.group_grad_hess(kind, s[off:off + nb * d].reshape(nb, d), m, nb)
        z[off:off + nb * d] = (-g / t).reshape(-1)
    return z


def dual_residual(problem: K.ConeProblem, z: np.ndarray) -> float:
    """||A^T z + c||_inf: wie weit z von der Dualitätsgleichung entfernt ist."""
    return float(np.linalg.norm(problem.A.T @ z + problem.c, np.inf))


def cone_margin_all(cones, v: np.ndarray) -> float:
    """Kleinster Randabstand von v über alle Kegel (>= 0 heißt im Kegel), unabhängig von der Barriere."""
    off, margins = 0, []
    for kind, m in cones:
        d = K.cone_dim(kind, m)
        margins.append(K.cone_margin(kind, v[off:off + d], m))
        off += d
    return min(margins)


def check_dual(problem: K.ConeProblem, z: np.ndarray, tol: float = 1e-7) -> list[str]:
    """Unabhängiger Prüfer für das Dual: nennt jede verletzte Regel (Gleichung, jeder Kegel)."""
    problems = []
    res = dual_residual(problem, z)
    if res > tol * (1.0 + float(np.linalg.norm(problem.c, np.inf))):
        problems.append(f"A^T z + c ist nicht null (Residuum {res:.2e})")
    problems += [p.replace("verletzt", "(dual) verletzt") for p in K.check_solution(K.ConeProblem(np.zeros(0), np.zeros((len(z), 0)), z, problem.cones, np.zeros(0)), np.zeros(0), tol)]
    return problems


def block_complementarity(problem: K.ConeProblem, x: np.ndarray, z: np.ndarray) -> list[float]:
    """z_i^T s_i je Kegelblock: im Optimum null (komplementärer Schlupf), und ihre Summe ist die Lücke z^T s."""
    s = problem.b - problem.A @ x
    out, off = [], 0
    for kind, m in problem.cones:
        d = K.cone_dim(kind, m)
        out.append(float(z[off:off + d] @ s[off:off + d]))
        off += d
    return out


# ----------------------------------------------------------------------------- Dualstart und Dual als Kegelprogramm

def nullspace_of_transpose(A: np.ndarray) -> np.ndarray:
    """Basis (Spalten) des Nullraums von A^T, also aller z mit A^T z = 0."""
    u, sv, _ = np.linalg.svd(A, full_matrices=True)
    rank = int((sv > 1e-12 * sv[0]).sum()) if len(sv) and sv[0] > 0 else 0
    return u[:, rank:]


def correct_to_equation(problem: K.ConeProblem, z: np.ndarray) -> np.ndarray:
    """Kleinste Korrektur (in der 2-Norm) von z auf A^T z0 = -c."""
    r = problem.A.T @ z + problem.c
    return z - problem.A @ np.linalg.solve(problem.A.T @ problem.A, r)


def find_dual_start(problem: K.ConeProblem, path: list[PathPoint] | None = None, levels=LEVELS) -> DualStart:
    """Je Niveau den ersten Pfadpunkt mit nu / t <= Niveau nehmen, fest nachzentrieren, z bilden, auf die Gleichung korrigieren und prüfen, ob z0 streng im Kegel liegt.

    Der erste gültige Start gewinnt. Wirft DualStartError, wenn keines der Niveaus einen gültigen Start liefert (das Dual hat dann keinen streng zulässigen Punkt in Reichweite)."""
    path = central_path(problem) if path is None else path
    null = nullspace_of_transpose(problem.A)
    tried = []
    for level in levels:
        point = next((p for p in path if p.gap <= level), None)
        if point is None:
            tried.append(f"{level:.0e}: Pfad zu kurz")
            continue
        x, _, ok = K.center(problem, point.x, point.t, tol=1e-12)
        if not ok:
            tried.append(f"{level:.0e}: Zentrieren scheitert")
            continue
        z = dual_point(problem, x, point.t)
        before = dual_residual(problem, z)
        z0 = correct_to_equation(problem, z)
        margin = cone_margin_all(problem.cones, z0)
        if margin > 0 and np.isfinite(K.barrier_value(problem.cones, z0)):
            return DualStart(z0, null, level, point.t, before, margin)
        tried.append(f"{level:.0e}: z0 liegt nicht streng im Kegel (Randabstand {margin:.1e})")
    raise DualStartError("kein streng zulässiger Dualstart: " + "; ".join(tried))


def dual_program(problem: K.ConeProblem, start: DualStart) -> K.ConeProblem:
    """Das Dual als Kegelprogramm im selben Format: z = z0 + N y, min b^T z = b^T z0 + (N^T b)^T y, s' = z0 - (-N) y im Kegel, Start y = 0."""
    N = start.null
    return K.ConeProblem(N.T @ problem.b, -N, start.z0, problem.cones, np.zeros(N.shape[1]))


def solve_pair(problem: K.ConeProblem, tol: float = 1e-9) -> PairResult:
    """Primal und Dual mit demselben Kern lösen: Primal, Dualstart aus dem Pfad, Dual als Kegelprogramm."""
    primal = K.solve_barrier(problem, tol=tol)
    start = find_dual_start(problem)
    dprob = dual_program(problem, start)
    dual = K.solve_barrier(dprob, tol=tol)
    z = dprob.b - dprob.A @ dual.x
    s = problem.b - problem.A @ primal.x
    p, d = float(problem.c @ primal.x), float(-(problem.b @ z))
    return PairResult(primal, dual, z, p, d, p - d, float(z @ s), dual_residual(problem, z), start)
