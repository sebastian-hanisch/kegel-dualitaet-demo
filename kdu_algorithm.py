"""Kern der Demo (nur numpy/Standardbibliothek): wählt die k billigsten Kandidaten. Gerüst: durch das echte Verfahren ersetzen.

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b): `select_cheapest` ist die Auswahlregel,
`check_feasible` der unabhängige Zulässigkeitsprüfer, `solve` ruft nur der Reihe nach auf."""

import itertools
import math
from dataclasses import dataclass

from kdu_scenario import Instance


@dataclass(frozen=True)
class Result:
    chosen: tuple[int, ...]   # Indizes der gewählten Kandidaten, aufsteigend
    cost: float


def check_feasible(inst: Instance, k: int, chosen) -> bool:
    """Genau k verschiedene, existierende Kandidaten."""
    chosen = tuple(chosen)
    return len(chosen) == k and len(set(chosen)) == k and all(0 <= i < inst.n for i in chosen)


def select_cheapest(inst: Instance, k: int) -> Result:
    """Auswahlregel: die k billigsten Kandidaten; bei gleichen Kosten gewinnt der kleinere Index."""
    if not 0 <= k <= inst.n:
        raise ValueError(f"k = {k} liegt außerhalb von 0 bis {inst.n}")
    order = sorted(range(inst.n), key=lambda i: (inst.costs[i], i))
    chosen = tuple(sorted(order[:k]))
    return Result(chosen, sum(inst.costs[i] for i in chosen))


def solve(inst: Instance, k: int) -> Result:
    return select_cheapest(inst, k)


def count_selections(n: int, k: int) -> int:
    """Zahl aller möglichen Auswahlen (Binomialkoeffizient)."""
    return math.comb(n, k)


def brute_force(inst: Instance, k: int) -> Result:
    """Zählt alle Auswahlen auf (nur für kleine Instanzen): die exakte Gegenprobe für den Methodenvergleich."""
    if not 0 <= k <= inst.n:
        raise ValueError(f"k = {k} liegt außerhalb von 0 bis {inst.n}")
    best = min(itertools.combinations(range(inst.n), k), key=lambda c: (sum(inst.costs[i] for i in c), c))
    return Result(tuple(best), sum(inst.costs[i] for i in best))
