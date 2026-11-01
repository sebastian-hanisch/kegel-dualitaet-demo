"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import kdu_forms as A

DEFAULT_FORM = "ball_soc"
N_MIN, N_MAX, DEFAULT_N = 8, 30, 20           # die Dualprogramme haben bis 6 n Zeilen und Nullraum-Dimension bis 6 n - 3: O((6 n)^3) je Newton-Schritt
STRETCH_MIN, STRETCH_MAX, DEFAULT_STRETCH = 1.2, 6.0, 4.0
ANGLE_MIN, ANGLE_MAX, DEFAULT_ANGLE = 0, 90, 45
SEED_MAX, DEFAULT_SEED = 999999, 35
SWEEP_SEEDS = tuple(range(100000, 100005))   # feste Wolken für alle Sweeps: nie ein einzelner Seed als Befund

GAPEX_EPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)    # Familie der streng zulässigen Beispiel mit Lückee
GAPEX_TOL = 1e-4                              # Lückenziel für die Familie (der Pfad läuft entlang der unbeschränkten optimalen Menge davon)
GAPEX_MAX_NEWTON = 400
GAPEX_SAMPLES = 2000                          # Zufallskandidaten für Primal und Dual bei eps = 0
NONATTAINED_ROUNDS = 10
WEAK_XS = (1e1, 1e2, 1e4, 1e6, 1e8)
CERT_SAMPLES = 500                           # Zufallsmatrizen Z für die Zertifikatssuche bei schwacher Unzulässigkeit

COLORS = {"cube": "#4c78a8", "diamond": "#f58518", "ball_soc": "#2F6B65", "ball_psd": "#7a9e9a", "ellipse_axis": "#c0504d", "ellipse_psd": "#8a2b27"}
COLOR_PRIMAL = "#2F6B65"
COLOR_DUAL = "#c0504d"
COLOR_POINT = "rgba(150,150,150,0.85)"
COLOR_REF = "#444444"

PRESETS = {
    "Standardfall: Kugel (Lorentz-Kegel)": {"form": DEFAULT_FORM, "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED},
    "Dieselbe Kugel als SDP": {"form": "ball_psd", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED},
    "Raute: das Dual ist ein LP": {"form": "diamond", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED},
    "Achsenparallele Ellipse: der Dualstart ist heikel": {"form": "ellipse_axis", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED},
    "Allgemeine Ellipse: das Dual ist ein SDP": {"form": "ellipse_psd", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall: Kugel (Lorentz-Kegel)": "20 Punkte, in die Länge gezogen und um 45 Grad gedreht; das Dual der kleinsten Kugel sind Gewichte auf zwei Stützpunkten (wie in Stück 1).",
    "Dieselbe Kugel als SDP": "Dieselbe Kugel als 3 x 3-Matrizen im PSD-Kegel: das Dual hat 117 freie Variablen statt 57, aber denselben Wert.",
    "Raute: das Dual ist ein LP": "Die l1-Kugel ist ein reines LP; ihr Dual ist wieder ein LP (Orthant), das derselbe Löser löst.",
    "Achsenparallele Ellipse: der Dualstart ist heikel": "Hier liegt der korrigierte Dualstart nur bei einem gröberen Lückenniveau streng im Kegel (die Suche über Niveaus zeigt, welches).",
    "Allgemeine Ellipse: das Dual ist ein SDP": "Die Ellipse mit PSD-Kegel hat ein Dual mit PSD-Kegel; Primal und Dual haben dieselbe Struktur.",
}
assert set(PRESETS) == set(PRESET_HELP) and all(p["form"] in A.FORMS for p in PRESETS.values())
