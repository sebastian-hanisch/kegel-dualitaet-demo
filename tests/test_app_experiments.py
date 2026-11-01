"""Pfade der App, die der Rauchtest nicht erreicht: jede Form, die fünf Experimente im Seiteninhalt, PDF, Eckfälle aller Regler zugleich (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kdu_constants as C
import kdu_forms as F

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        (at.sidebar.selectbox if key == "form_select" else at.sidebar.number_input if key == "seed_input" else at.sidebar.slider)(key=key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_primal_dual_and_all_five_experiment_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert values["Primalwert"] == values["Dualwert"] == "7,167474" and values["Lücke Primal − Dual"].endswith("⁻¹⁰") and values["Residuum Aᵀz + c"].endswith("⁻¹⁷") or values["Residuum Aᵀz + c"].endswith("⁻¹⁶")
    text = _texts(at)
    for part in ("Primal (Kugel (Lorentz-Kegel)) **7,167474**", "Für alle sechs Formen löst derselbe Kern das Dual", "Das Dual direkt aus dem Pfad", "**2 Stützpunkte** tragen alles",
                 "**alle mit x₁ = 0**", "es gibt keinen zentralen Pfad", "kleinste Eigenwert von"):
        assert part in text, part


def test_pdf_button_produces_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert at.session_state["pdf_made"][1][:4] == b"%PDF"


@pytest.mark.parametrize("form", F.FORMS)
def test_every_form_renders_without_error_and_both_checks_pass(form):
    at = _app(form_select=form)
    assert not at.error
    assert any("Beide Prüfungen bestehen" in s.value for s in at.success)


@pytest.mark.parametrize("edge", ["min", "max"])
def test_all_sliders_at_their_minimum_or_maximum_together(edge):
    pick = (lambda lo, hi: lo) if edge == "min" else (lambda lo, hi: hi)
    state = {"n_slider": pick(C.N_MIN, C.N_MAX), "stretch_slider": pick(C.STRETCH_MIN, C.STRETCH_MAX), "angle_slider": pick(C.ANGLE_MIN, C.ANGLE_MAX)}
    at = _app(**state)
    assert not at.exception and not at.error


def test_the_smallest_cloud_works_in_every_form():
    for form in F.FORMS:
        at = _app(form_select=form, n_slider=C.N_MIN)
        assert not at.error, form


def test_a_form_without_a_dual_start_is_reported_instead_of_crashing(monkeypatch):
    """Findet die Suche keinen Dualstart (hier erzwungen), meldet die App das für die gewählte Form und rechnet nicht weiter; die übrigen Formen bleiben in der Analyse."""
    import kdu_dual as D
    import kdu_evaluation as ev

    real = D.find_dual_start

    def refuse_for_axis_ellipse(problem, *args, **kwargs):
        if len(problem.cones) == 3 and problem.cones[0][0] == "l" and problem.cones[1] == ("q", 3):      # ellipse_axis: Orthant + zwei Lorentz-Kegel (3)
            raise D.DualStartError("kein streng zulässiger Dualstart: erzwungen")
        return real(problem, *args, **kwargs)

    monkeypatch.setattr(D, "find_dual_start", refuse_for_axis_ellipse)
    a = ev.analyse(ev.Settings(seed=424242))
    assert list(a.failed) == ["ellipse_axis"] and "erzwungen" in a.failed["ellipse_axis"] and len(a.pairs) == 5 and len(ev.pair_rows(a)) == 5
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["form"] = "ellipse_axis"
    at.query_params["seed"] = "424243"
    at.run()
    assert not at.exception
    assert any("findet die Suche keinen streng zulässigen Dualstart" in e.value and "erzwungen" in e.value for e in at.error)
    assert not at.metric                                              # nach dem Fehler stoppt die App vor dem Ergebnis
