"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink verwirft ungültige Werte."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import kdu_constants as C
import kdu_evaluation as ev
import kdu_forms as A
import kdu_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert preset["form"] in A.FORMS
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.STRETCH_MIN <= preset["stretch"] <= C.STRETCH_MAX
        assert C.ANGLE_MIN <= preset["angle"] <= C.ANGLE_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall: Kugel (Lorentz-Kegel)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_each_preset_uses_the_form_its_name_announces():
    forms = {name: p["form"] for name, p in C.PRESETS.items()}
    assert forms["Dieselbe Kugel als SDP"] == "ball_psd" and forms["Raute: das Dual ist ein LP"] == "diamond"
    assert forms["Achsenparallele Ellipse: der Dualstart ist heikel"] == "ellipse_axis" and forms["Allgemeine Ellipse: das Dual ist ein SDP"] == "ellipse_psd"


def test_the_help_texts_name_numbers_that_the_presets_produce():
    """Dual hat 117 freie Variablen statt 57 (PSD-Kugel gegen Lorentz-Kegel, n = 20); die achsenparallele Ellipse braucht ein gröberes Niveau als die anderen."""
    a = ev.analyse(ev.Settings(**C.PRESETS["Dieselbe Kugel als SDP"]))
    rows = {r["form"]: r for r in ev.pair_rows(a)}
    assert (rows["ball_psd"]["dual_variables"], rows["ball_soc"]["dual_variables"]) == (117, 57)
    assert rows["ellipse_axis"]["level"] > rows["ball_soc"]["level"]
    assert "117" in C.PRESET_HELP["Dieselbe Kugel als SDP"] and "57" in C.PRESET_HELP["Dieselbe Kugel als SDP"]


def test_permalink_takes_valid_values_and_drops_invalid_ones():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["form"] = "ellipse_psd"
    at.query_params["n"] = "999"            # über der Grenze: wird begrenzt
    at.query_params["stretch"] = "nan"      # keine endliche Zahl: bleibt Standard
    at.query_params["angle"] = "abc"        # keine Zahl: bleibt Standard
    at.run()
    assert not at.exception
    assert at.sidebar.selectbox(key="form_select").value == "ellipse_psd"
    assert at.sidebar.slider(key="n_slider").value == C.N_MAX
    assert at.sidebar.slider(key="stretch_slider").value == C.DEFAULT_STRETCH
    assert at.sidebar.slider(key="angle_slider").value == C.DEFAULT_ANGLE
    bad = AppTest.from_file(APP, default_timeout=180)
    bad.query_params["form"] = "dreieck"
    bad.run()
    assert not bad.exception and bad.sidebar.selectbox(key="form_select").value == C.DEFAULT_FORM
