"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import pytest

import kdu_evaluation as ev


def test_formatting_helpers():
    assert ev.de(2.3456, 3) == "2,346" and ev.de(-1.0, 0) == "−1" and ev.de(0.5, 1) == "0,5"
    assert ev.sci(1.7e-10) == "1,7·10⁻¹⁰" and ev.sci(2.5e5) == "2,5·10⁵" and ev.sci(0.0) == "0,0" and ev.sci(9.96e-3) == "1,0·10⁻²" and ev.sci(float("inf")) == "unendlich"
    assert ev.level_text(1e-3) == "1·10⁻³" and ev.level_text(0.1) == "1·10⁻¹"


def _row(form="ball_soc", ok=True, name="Kugel (Lorentz-Kegel)", gap=1.7e-10, level=1e-3, rows=60, dual_variables=57):
    return {"form": form, "name": name, "primal": 7.167474, "dual": 7.167474, "gap": gap, "complementarity": 1.7e-10, "residual": 5.5e-17, "level": level,
            "dual_variables": dual_variables, "rows": rows, "newton_primal": 34, "newton_dual": 11, "ok": ok}


def test_form_verdict_names_values_gap_size_and_start():
    kind, text = ev.verdict_form(_row())
    assert kind == "success"
    for part in ("Primal (Kugel (Lorentz-Kegel)) **7,167474**", "Dual **7,167474**", "die Lücke ist **1,7·10⁻¹⁰**", "`zᵀs` 1,7·10⁻¹⁰", "Das Dual hat 57 freie Variablen (Nullraum von Aᵀ bei 60 Zeilen)",
                 "ν/t ≈ 1·10⁻³", "Beide Prüfungen bestehen"):
        assert part in text, part
    kind, text = ev.verdict_form(_row(ok=False))
    assert kind == "error" and text.endswith("Die Prüfung besteht nicht.")


def test_pairs_verdict_names_worst_gap_residual_and_levels():
    rows = [_row(gap=1e-10, level=1e-3), _row("ellipse_axis", gap=7.7e-10, level=0.1), _row("cube", gap=3e-10, level=1e-3)]
    rows[1]["residual"] = 9.1e-13
    kind, text = ev.verdict_pairs(rows)
    assert kind == "success"
    assert "größte Lücke zwischen Primal- und Dualwert 7,7·10⁻¹⁰" in text and "größtes Residuum der Dualitätsgleichung 9,1·10⁻¹³" in text and "bei den Lückenniveaus 1·10⁻¹, 1·10⁻³" in text
    rows[0]["ok"] = False
    assert ev.verdict_pairs(rows)[0] == "error"


def test_quality_verdict_reports_the_demo_level_and_the_path_end():
    rows = [{"gap": 12.0, "residual_coarse": 1e-7, "residual_fine": 1e-13, "margin": 0.02}, {"gap": 0.18, "residual_coarse": 1e-5, "residual_fine": 1e-10, "margin": 1e-3},
            {"gap": 2.9e-3, "residual_coarse": 3.6e-5, "residual_fine": 1.2e-12, "margin": 1e-6}, {"gap": 2.2e-11, "residual_coarse": 4.2e-4, "residual_fine": 5.0e-4, "margin": 1e-13}]
    assert ev.quality_level_row(rows)["gap"] == 2.9e-3 and ev.quality_level_row(rows, 1.0)["gap"] == 0.18
    kind, text = ev.verdict_quality(rows)
    assert kind == "warning"
    for part in ("bei ν/t = 2,9·10⁻³ (dort setzt die Demo an) sind es 3,6·10⁻⁵ bei grobem und 1,2·10⁻¹² bei feinem Zentrieren", "am Pfadende bei ν/t = 2,2·10⁻¹¹ aber 5,0·10⁻⁴ (fein) bzw. 4,2·10⁻⁴ (grob)"):
        assert part in text, part


def test_complementarity_verdict_names_support_deviation_and_residuals():
    data = {"sum_weights": 1.0, "support": [3, 13], "weight_outside": 2.3e-11, "max_dev_support": 2.0e-11, "centre_residual": 5.5e-17, "ok": True}
    kind, text = ev.verdict_complementarity(data)
    assert kind == "success"
    for part in ("summieren sich zu 1,000000000", "**2 Stützpunkte** tragen alles", "außerhalb höchstens 2,3·10⁻¹¹", "um höchstens 2,0·10⁻¹¹ von den Gewichten aus Stück 1", "ist 5,5·10⁻¹⁷"):
        assert part in text, part
    assert ev.verdict_complementarity(dict(data, ok=False))[0] == "error"


def _gapex():
    return {"candidates": 2000, "feasible": 1000, "primal_x1_max": 0.0, "primal_value": 0.0, "dual_samples": 200, "dual_min": -1.0, "dual_max": -1.0, "dual_value": -1.0,
            "starts_found": 0, "starts_total": 5, "sweep": [{"eps": 0.1, "value": -0.9998, "gap": 7.3e-4}, {"eps": 1e-5, "value": -0.9997, "gap": 7.3e-4}]}


def test_gap_example_verdict_names_candidates_gap_family_and_start():
    kind, text = ev.verdict_gapex(_gapex())
    assert kind == "warning"
    for part in ("von 2000 Kandidaten 1000 zulässig, **alle mit x₁ = 0**", "**Primalwert 0**", "Jede der 200 zulässigen Dualmatrizen hat denselben Wert **−1** (Streuung 0,0)", "**Lücke 1**",
                 "für ε von 1·10⁻¹ bis 1·10⁻⁵", "Werte von −0,9998 bis −0,9997", "höchstens 7,3·10⁻⁴", "springt bei ε = 0 von −1 auf 0", "gelingt bei 0 von 5 Werten von ε"):
        assert part in text, part


def _nonattained(found_bounded=True, found_original=False, converged=True):
    return {"start_original": {"found": found_original}, "start_bounded": {"found": found_bounded}, "bounded_value": 2.0, "bounded_x": [1.0, 1.0], "bounded_steps": 24, "bounded_converged": converged,
            "rows": [{"round": 0, "t": 1.0, "x1": 1.0, "x2": 82.0, "f": -3.4, "decrement2": 1.0}, {"round": 9, "t": 1e8, "x1": 1.9e-8, "x2": 8.7e7, "f": 3.0, "decrement2": 0.09}]}


def test_nonattained_verdict_describes_the_run_away_and_the_start_probe():
    kind, text = ev.verdict_nonattained(_nonattained())
    assert kind == "warning"
    for part in ("x₁ (von 1,0·10⁰ auf 1,9·10⁻⁸)", "x₂ wächst (von 8,2·10¹ auf 8,7·10⁷)", "kein Minimum", "scheitert hier und gelingt bei der Variante", "Optimum 2,000000 bei (1,000; 1,000), nach 24 Newton-Schritten"):
        assert part in text, part
    assert "gelingt hier und scheitert bei der Variante" in ev.verdict_nonattained(_nonattained(found_bounded=False, found_original=True))[1]
    assert "Optimum 2,000000" not in ev.verdict_nonattained(_nonattained(converged=False))[1]


def test_weak_verdict_names_eigenvalue_certificate_search_and_the_lp_contrast():
    res = {"rows": [{"x": 10.0, "eig": -0.099}, {"x": 1e8, "eig": -1e-8}], "cert_best": 0.0, "cert_samples": 500, "farkas": {"ATy": 0.0, "bTy": 1.0}}
    kind, text = ev.verdict_weak(res)
    assert kind == "warning"
    for part in ("ist −1,0·10⁻⁸ bei x = 1·10⁸", "bei 500 Zufallsmatrizen ist der beste Wert 0,00", "`Aᵀy = 0` und `bᵀy = 1 > 0`"):
        assert part in text, part
