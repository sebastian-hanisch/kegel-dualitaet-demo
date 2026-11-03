"""Kennzahlen, Experimente und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)."""

import math
from dataclasses import dataclass

import numpy as np

import kdu_constants as C
import kdu_dual as D
import kdu_examples as E
import kdu_forms as A
import kegel_kern as K
from kdu_scenario import Cloud, make_cloud


@dataclass(frozen=True)
class Settings:
    form: str = C.DEFAULT_FORM
    n: int = C.DEFAULT_N
    stretch: float = C.DEFAULT_STRETCH
    angle: int = C.DEFAULT_ANGLE
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True, eq=False)
class FormPair:
    form: str
    problem: K.ConeProblem
    pair: D.PairResult
    primal_problems: list     # Rückumformung des Primals (Kegelprogramm und Ausgangsaufgabe)
    dual_problems: list       # Prüfer des Duals (Gleichung, Kegel)

    @property
    def ok(self) -> bool:
        return not self.primal_problems and not self.dual_problems


@dataclass(frozen=True, eq=False)
class Analysis:
    settings: Settings
    cloud: Cloud
    pairs: dict               # Form -> FormPair (nur die Formen, für die ein Dualstart gefunden wurde)
    failed: dict              # Form -> Meldung, wenn die Suche über Lückenniveaus keinen streng zulässigen Dualstart fand (die App meldet das, statt abzustürzen)

    @property
    def selected(self) -> FormPair:
        return self.pairs[self.settings.form]

    @property
    def selected_failed(self) -> str | None:
        return self.failed.get(self.settings.form)


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,0·10⁻¹⁰."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def level_text(level: float) -> str:
    return sci(level, 0)


# ----------------------------------------------------------------------------- Analyse der eingestellten Wolke

def solve_form_pair(form: str, points: np.ndarray, tol: float = 1e-9) -> FormPair:
    problem = A.build_form(form, points)
    pair = D.solve_pair(problem, tol=tol)
    primal_problems = K.check_solution(problem, pair.primal.x) + A.original_violations(form, np.asarray(points, dtype=float), pair.primal.x, pair.primal.value)
    return FormPair(form, problem, pair, primal_problems, D.check_dual(problem, pair.z))


def analyse(settings: Settings) -> Analysis:
    cloud = make_cloud(settings.n, settings.stretch, settings.angle, settings.seed)
    pairs, failed = {}, {}
    with D.single_thread():
        for f in A.FORMS:
            try:
                pairs[f] = solve_form_pair(f, cloud.points)
            except D.DualStartError as e:
                failed[f] = str(e)
    return Analysis(settings, cloud, pairs, failed)


def pair_rows(a: Analysis) -> list[dict]:
    """Eine Zeile je Form: Primal- und Dualwert, Lücke, Komplementarität, Residuum, Dualstart-Niveau, Größe des Dualprogramms."""
    rows = []
    for f in A.FORMS:
        if f not in a.pairs:
            continue
        fp = a.pairs[f]
        p = fp.pair
        rows.append({"form": f, "name": A.FORM_NAMES[f], "primal": p.primal_value, "dual": p.dual_value, "gap": p.gap, "complementarity": p.complementarity, "residual": p.residual,
                     "level": p.start.level, "dual_variables": p.start.null.shape[1], "rows": fp.problem.A.shape[0], "newton_primal": p.primal.newton_steps,
                     "newton_dual": p.dual.newton_steps, "ok": fp.ok})
    return rows


# ----------------------------------------------------------------------------- Experiment 1b: Wie genau ist das Dual aus dem Pfadende?

def path_quality(form: str, points: np.ndarray) -> list[dict]:
    """Je Runde des zentralen Pfades: Lücke nu/t, Residuum von A^T z + c für z = -grad phi / t bei grobem (1e-6) und feinem (1e-12) Zentrieren, Randabstand von z."""
    problem = A.build_form(form, points)
    nu = K.barrier_parameter(problem.cones)
    x, t, rows = problem.x0.copy(), K.initial_t(problem), []
    with D.single_thread():
        for _ in range(16):
            x_c, _, ok = K.center(problem, x, t, tol=1e-6)
            if not ok:
                break
            x = x_c
            z = D.dual_point(problem, x, t)
            x_f, _, ok_f = K.center(problem, x, t, tol=1e-12)
            z_f = D.dual_point(problem, x_f, t) if ok_f else z
            rows.append({"t": t, "gap": nu / t, "residual_coarse": D.dual_residual(problem, z), "residual_fine": D.dual_residual(problem, z_f), "margin": D.cone_margin_all(problem.cones, z_f)})
            if nu / t < 1e-10:
                break
            t *= 8.0
    return rows


# ----------------------------------------------------------------------------- Experiment 2: Komplementarität der Kugel

def support_weights(points: np.ndarray, centre: np.ndarray, radius: float, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Stützpunkte (Abstand = Radius) und ihre Gewichte aus `sum lam x = c`, `sum lam = 1` (kleinste Quadrate): die Gewichte aus Stück 1."""
    dist = np.linalg.norm(points - centre, axis=1)
    support = np.where(dist >= radius * (1.0 - tol))[0]
    M = np.vstack([points[support].T, np.ones(len(support))])
    lam = np.linalg.lstsq(M, np.append(centre, 1.0), rcond=None)[0]
    return support, lam


def complementarity_data(points: np.ndarray) -> dict:
    """Kugel (Lorentz-Form) auf der Wolke: Dualgewichte z_i0, Schlupf R - |x_i - c| und die Gewichte aus Stück 1 auf den Stützpunkten."""
    with D.single_thread():
        fp = solve_form_pair("ball_soc", points)
    n = len(points)
    z = fp.pair.z.reshape(n, 3)
    centre, radius = fp.pair.primal.x[:2], float(fp.pair.primal.x[2])
    dist = np.linalg.norm(points - centre, axis=1)
    support, lam = support_weights(points, centre, radius)
    return {"weights": z[:, 0], "slack": radius - dist, "support": support, "exact": lam, "sum_weights": float(z[:, 0].sum()),
            "max_dev_support": float(np.abs(z[support, 0] - lam).max()), "weight_outside": float(np.delete(z[:, 0], support).max()) if len(support) < n else 0.0,
            "centre_residual": float(np.abs(z[:, 1:].sum(axis=0)).max()), "radius": radius, "ok": fp.ok}


def start_status(problem: K.ConeProblem) -> dict:
    """Findet die Suche über Lückenniveaus einen streng zulässigen Dualstart? Ohne streng zulässiges Dual nicht: das ist die Probe auf die Slater-Bedingung für das Dual."""
    try:
        with D.single_thread():
            st = D.find_dual_start(problem)
        return {"found": True, "level": st.level, "margin": st.margin, "message": ""}
    except (D.DualStartError, np.linalg.LinAlgError) as e:
        return {"found": False, "level": None, "margin": None, "message": str(e)}


# ----------------------------------------------------------------------------- Experiment 3: Beispiel mit Lücke (Lücke 1)

def gapex_experiment(seed: int = 7) -> dict:
    """Bei eps = 0: Zufallskandidaten für Primal (nur x1 = 0 zulässig) und Dual (Wert immer -1); dazu die streng zulässige Familie eps > 0 (Wert -1)."""
    rng = np.random.default_rng(seed)
    half = C.GAPEX_SAMPLES // 2
    x1 = np.concatenate([np.zeros(half), rng.choice([-1.0, 1.0], size=C.GAPEX_SAMPLES - half) * rng.uniform(1e-3, 2.0, size=C.GAPEX_SAMPLES - half)])
    cand = np.stack([x1, rng.uniform(0.0, 50.0, size=C.GAPEX_SAMPLES)], axis=1)       # die Hälfte mit x1 = 0, die andere mit |x1| zwischen 1e-3 und 2
    feasible = [x for x in cand if E.gapex_primal_feasible(*x)]
    duals = []
    for _ in range(200):
        a_, e = rng.uniform(0.0, 5.0), rng.uniform(-1.0, 1.0)
        Z = np.array([[a_ + e * e, 0.0, e], [0.0, 0.0, 0.0], [e, 0.0, 1.0]])
        if E.gapex_dual_feasible(Z):
            duals.append(E.gapex_dual_value(Z))
    sweep = E.gapex_sweep(C.GAPEX_EPS, tol=C.GAPEX_TOL, max_newton=C.GAPEX_MAX_NEWTON)
    starts = [start_status(E.gapex_problem(eps)) for eps in C.GAPEX_EPS]
    return {"starts_found": sum(1 for x in starts if x["found"]), "starts_total": len(starts), "candidates": len(cand), "feasible": len(feasible), "primal_x1_max": max((abs(x[0]) for x in feasible), default=0.0), "primal_value": E.GAPEX_PRIMAL,
            "dual_samples": len(duals), "dual_min": min(duals), "dual_max": max(duals), "dual_value": E.GAPEX_DUAL, "sweep": sweep}


# ----------------------------------------------------------------------------- Experiment 4: der Pfad läuft davon

def nonattained_experiment() -> dict:
    """`[[x1,1],[1,x2]]`: x1 -> 0, x2 wächst, f_t fällt; dagegen mit streng zulässigem Dual (min x1 + x2) Optimum 2 bei (1, 1)."""
    rows = E.nonattained_path(C.NONATTAINED_ROUNDS)
    bounded = K.solve_barrier(E.nonattained_problem((1.0, 1.0)), tol=1e-9)
    return {"start_original": start_status(E.nonattained_problem()), "start_bounded": start_status(E.nonattained_problem((1.0, 1.0))), "rows": rows, "bounded_value": bounded.value, "bounded_x": bounded.x.tolist(), "bounded_steps": bounded.newton_steps, "bounded_converged": bounded.converged}


# ----------------------------------------------------------------------------- Experiment 5: schwach unzulässig

def weak_experiment(seed: int = 7) -> dict:
    """`[[x,1],[1,0]]`: kleinster Eigenwert gegen x; Zertifikatssuche mit Z11 = 0 (dann Z12 = 0); LP-Farkas-Zertifikat im Kontrast."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(C.CERT_SAMPLES):
        a_, c_ = rng.uniform(0.0, 5.0), rng.uniform(0.0, 5.0)
        Z = np.array([[0.0, 0.0], [0.0, a_ + c_]])                   # Z11 = 0 erzwingt Z12 = 0 (sonst nicht PSD); Z22 beliebig
        values.append(E.weak_certificate_value(Z))
    rows = [{"x": x, "eig": E.weak_min_eigenvalue(x)} for x in C.WEAK_XS]
    return {"rows": rows, "cert_best": min(values), "cert_samples": len(values), "farkas": E.lp_farkas_example()}


# ----------------------------------------------------------------------------- Verdikte (rein)

def verdict_pairs(rows: list[dict]) -> tuple[str, str]:
    """Experiment 1: Primal und Dual aller sechs Formen."""
    worst = max(r["gap"] for r in rows)
    levels = sorted({r["level"] for r in rows}, reverse=True)
    base = (f"Für alle sechs Formen löst derselbe Kern das Dual als eigenes Kegelprogramm: **größte Lücke zwischen Primal- und Dualwert {sci(worst)}**, "
            f"größtes Residuum der Dualitätsgleichung {sci(max(r['residual'] for r in rows))} (die Gleichungen gelten exakt über den Nullraum). "
            f"Der Dualstart aus dem zentralen Pfad lag bei den Lückenniveaus {', '.join(level_text(v) for v in levels)}.")
    if not all(r["ok"] for r in rows):
        return "error", base + " Eine Prüfung (Rückumformung oder Dualzulässigkeit) besteht nicht: Das darf nicht passieren."
    return "success", base


def verdict_form(row: dict) -> tuple[str, str]:
    """Kernfrage der Primäransicht: Primal- und Dualwert der gewählten Form."""
    base = (f"Primal ({row['name']}) **{de(row['primal'], 6)}**, Dual **{de(row['dual'], 6)}**: die Lücke ist **{sci(row['gap'])}**, die Komplementarität `zᵀs` {sci(row['complementarity'])}. "
            f"Das Dual hat {row['dual_variables']} freie Variablen (Nullraum von Aᵀ bei {row['rows']} Zeilen); der Dualstart kam bei ν/t ≈ {level_text(row['level'])} aus dem zentralen Pfad.")
    if not row["ok"]:
        return "error", base + " Die Prüfung besteht nicht."
    return "success", base + " Beide Prüfungen bestehen: die Dualitätsgleichung gilt, und z liegt im Kegel."


def quality_level_row(rows: list[dict], level: float = 1e-3) -> dict:
    """Die Pfadrunde, deren Lücke nu/t dem Niveau am nächsten liegt (im Logarithmus): dort setzt die Demo ihren Dualstart an."""
    return min(rows, key=lambda r: abs(math.log10(r["gap"]) - math.log10(level)))


def verdict_quality(rows: list[dict]) -> tuple[str, str]:
    """Experiment 1b: Dualpunkt am Pfad bei grobem gegen feines Zentrieren, auf dem Niveau des Dualstarts und am Pfadende."""
    mid, last = quality_level_row(rows), rows[-1]
    return "warning", (f"Das Dual direkt aus dem Pfad (`z = −∇φ/t`) erfüllt `Aᵀz = −c` nur, wenn der Punkt genau zentriert ist: bei ν/t = {sci(mid['gap'])} (dort setzt die Demo an) sind es {sci(mid['residual_coarse'])} bei grobem "
                       f"und {sci(mid['residual_fine'])} bei feinem Zentrieren, am Pfadende bei ν/t = {sci(last['gap'])} aber {sci(last['residual_fine'])} (fein) bzw. {sci(last['residual_coarse'])} (grob), "
                       "weil die Hesse-Matrix mit wachsendem t schlecht konditioniert wird. Deshalb nimmt die Demo aus dem Pfad nur den Start und löst das Dual als eigenes Programm.")


def verdict_complementarity(data: dict) -> tuple[str, str]:
    """Experiment 2: Gewichte auf den Stützpunkten, Gleichheit mit Stück 1."""
    k = len(data["support"])
    base = (f"Die Dualgewichte `z_i0` summieren sich zu {de(data['sum_weights'], 9)}; **{k} Stützpunkte** tragen alles, außerhalb höchstens {sci(data['weight_outside'])} (komplementärer Schlupf). "
            f"Auf den Stützpunkten weichen sie um höchstens {sci(data['max_dev_support'])} von den Gewichten aus Stück 1 ab (`Σ λᵢ xᵢ = c`, `Σ λᵢ = 1`); der Schwerpunkt-Fehler `Σ z̄ᵢ` ist {sci(data['centre_residual'])}. Diese Werte sind Löser- und Rundungsrauschen und fallen auf anderer CPU anders aus.")
    return ("success" if data["ok"] else "error"), base


def verdict_gapex(res: dict) -> tuple[str, str]:
    """Experiment 3: Beispiel mit Lücke."""
    sweep = res["sweep"]
    values = [r["value"] for r in sweep]
    spread = res["dual_max"] - res["dual_min"]
    return "warning", (f"Bei ε = 0 sind von {res['candidates']} Kandidaten {res['feasible']} zulässig, **alle mit x₁ = 0** (größtes |x₁| unter den zulässigen: {sci(res['primal_x1_max'])}): **Primalwert {de(res['primal_value'], 0)}**. "
                       f"Jede der {res['dual_samples']} zulässigen Dualmatrizen hat denselben Wert **{de(res['dual_min'], 0)}** (Streuung {sci(spread)}): **Lücke {de(res['primal_value'] - res['dual_value'], 0)}**. "
                       f"Macht man den (1,1)-Eintrag ε > 0, ist das Problem streng zulässig, und für ε von {sci(sweep[0]['eps'], 0)} bis {sci(sweep[-1]['eps'], 0)} findet der Löser Werte von {de(min(values), 4)} bis {de(max(values), 4)} "
                       f"(bewiesene Lücke höchstens {sci(max(r['gap'] for r in sweep))}, also {de(res['dual_value'], 0)} im Rahmen der Lücke): der Wert springt bei ε = 0 von {de(res['dual_value'], 0)} auf {de(res['primal_value'], 0)}. "
                       "Der Löser läuft dabei entlang der unbeschränkten optimalen Menge davon (x₂ wächst). "
                       f"Die Suche nach einem streng zulässigen Dualstart gelingt bei {res['starts_found']} von {res['starts_total']} Werten von ε: das Dual hat keinen streng zulässigen Punkt. "
                       "Die Zählungen illustrieren die Handrechnung (die Kandidaten sind so gezogen, dass die Hälfte x₁ = 0 hat, die Dualmatrizen schon in zulässiger Form), sie sind keine unabhängige Stichprobe.")


def verdict_nonattained(res: dict) -> tuple[str, str]:
    """Experiment 4: der Pfad existiert nicht."""
    first, last = res["rows"][0], res["rows"][-1]
    base = (f"Mit jeder Runde fällt x₁ (von {sci(first['x1'])} auf {sci(last['x1'])}), aber x₂ wächst (von {sci(first['x2'])} auf {sci(last['x2'])}) und `f_t` hat kein Minimum: "
            "es gibt keinen zentralen Pfad, denn das Dual ist nicht streng zulässig und die optimale Menge liegt im Unendlichen. ")
    base += (f"Die Suche nach einem streng zulässigen Dualstart {'gelingt' if res['start_original']['found'] else 'scheitert'} hier und "
             f"{'gelingt' if res['start_bounded']['found'] else 'scheitert'} bei der Variante mit streng zulässigem Dual. ")
    if res["bounded_converged"]:
        base += f"Dieselbe Nebenbedingung mit streng zulässigem Dual (`min x₁ + x₂`) hat Optimum {de(res['bounded_value'], 6)} bei ({de(res['bounded_x'][0], 3)}; {de(res['bounded_x'][1], 3)}), nach {res['bounded_steps']} Newton-Schritten."
    return "warning", base


def verdict_weak(res: dict) -> tuple[str, str]:
    """Experiment 5: keine Zertifikate bei schwacher Unzulässigkeit, im LP schon."""
    last = res["rows"][-1]
    f = res["farkas"]
    return "warning", (f"Der kleinste Eigenwert von `[[x,1],[1,0]]` ist {sci(last['eig'])} bei x = {sci(last['x'], 0)}: er geht gegen null, wird aber nie null, das System ist unzulässig und kommt dem Kegel beliebig nahe. "
                       f"Ein Zertifikat der Unzulässigkeit (Z ⪰ 0 mit Z₁₁ = 0 und `⟨Z, M₀⟩ < 0`) gibt es nicht: bei {res['cert_samples']} Zufallsmatrizen ist der beste Wert {de(res['cert_best'], 2)} (Z₁₁ = 0 erzwingt Z₁₂ = 0; die Zufallsmatrizen sind mit Z₁₂ = 0 gezogen und nur Beispiele, der Beweis ist die Handrechnung). "
                       f"Im LP `x ≥ 1, x ≤ 0` dagegen zeigt `y = (1, 1)` mit `Aᵀy = {de(f['ATy'], 0)}` und `bᵀy = {de(f['bTy'], 0)} > 0` die Unzulässigkeit.")
