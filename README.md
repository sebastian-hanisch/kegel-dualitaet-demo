# Kegel-Dualität – die Lücke im Kegel und wo Slater fehlt – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kegel-dualitaet-demo.streamlit.app/)** -->

Stück 4 von 16 der **Konzepte-Reihe „Konvexe und konische Optimierung“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wie jede Konzepte-Demo dreht sich alles um **ein** Verfahren mit wachsendem Beispiel: ein Löser für Kegelprogramme, dessen Kegel von Stück zu Stück wächst. Vorgänger sind [konvexitaet-zertifikat-demo](https://github.com/sebastian-hanisch/konvexitaet-zertifikat-demo) (Stück 1: die kleinste Kugel mit Zertifikat) und [kegelform-demo](https://github.com/sebastian-hanisch/kegelform-demo) (Stück 2: die sechs Kegelformen und der gemeinsame Löser); Stück 3 (DCP, optional) ist übersprungen. Nachfolger ist die Barriere im Kegel (Stück 5). Die Linien-Spezifikation und die Einbindung in die Website folgen, wenn die Reihe vollständig ist.

## Kernfrage

**Stimmt das Dual mit dem Primal überein, und wann nicht?** Zu `min cᵀx` mit `s = b − A x ∈ K` gehört das Dual `max −bᵀz` mit `Aᵀz = −c` und `z ∈ K`; Orthant, Lorentz- und PSD-Kegel sind selbstdual. Für zulässige `x` und `z` gilt `cᵀx + bᵀz = zᵀs ≥ 0`: **die Lücke ist das Zertifikat**, wie die untere Schranke aus Stück 1. Im LP gilt starke Dualität fast immer; **im Kegel trägt sie nur mit einem streng zulässigen Punkt (Slater)**. Drei kleine SDPs ohne Slater-Punkt zeigen, was dann anders ist als im LP.

Das Beispiel bleibt die kleinste umschließende Form einer Punktwolke in den sechs Formen aus Stück 2 (Würfel, Raute, Kugel als Lorentz- und PSD-Kegel, Ellipse achsenparallel und allgemein, die Ellipsen um den Ursprung mit Ziel kleinste Summe der quadrierten Halbachsen).

## Modell

- **Das Dual als eigenes Kegelprogramm im selben Format** (`kdu_dual.py`): `z = z₀ + N y` mit einer Basis `N` des Nullraums von `Aᵀ` erfüllt `Aᵀz = −c` exakt; `min bᵀz` über `y` ist ein Kegelprogramm mit `A' = −N`, `b' = z₀`, `c' = Nᵀb`, gelöst mit demselben Kern (`kegel_kern.py`, wortgleich mit Stück 2 und unverändert).
- **Der Startpunkt `z₀`** kommt aus dem zentralen Pfad des Primals: `z = −∇φ(s)/t` (liegt im Inneren des Kegels, `zᵀs = ν/t`), danach auf `Aᵀz = −c` korrigiert (kleinste Änderung). Gesucht wird über Lückenniveaus `ν/t` = 10⁻³, 10⁻², 10⁻¹, 10⁻⁴, 10⁻⁵, 1, 10: der erste Start, der streng im Kegel liegt, gewinnt.
- **Prüfungen:** Rückumformung des Primals (Kegel und Ausgangsaufgabe, wie in Stück 2); für das Dual ein unabhängiger Prüfer (Gleichung `Aᵀz + c`, jeder Kegel über Eigenwerte, `s₀ − ‖s̄‖`, Komponenten). Die App meldet jede verletzte Regel als Fehler und findet die Suche keinen Dualstart, meldet sie das, statt abzustürzen.
- **Drei Beispiele ohne Slater-Punkt** (`kdu_examples.py`), von Hand gerechnet, im 📐-Expander hergeleitet.

## Methodik

Fünf Experimente mit live berechnetem Verdikt: (1) das Dual aus dem Löser (Tabelle der sechs Formen mit Primal, Dual, Lücke, Größe des Dualprogramms; dazu die Güte des Dualpunkts aus dem Pfad); (2) Komplementarität: wer trägt das Gewicht (die Kugel gegen die Gewichte aus Stück 1); (3) ohne Slater: eine Lücke von 1; (4) der zentrale Pfad läuft davon; (5) unzulässig, ohne dass es ein Zertifikat gibt.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (wo dort nur ein Band geprüft wird, wegen Rundung auf Linux, steht die gemessene Zahl trotzdem hier); Experimente laufen über feste Wolken `SWEEP_SEEDS` (5 Seeds, 3 Zuschnitte: 15 Wolken je Form), nicht über einen einzelnen Seed. Die Standardwolke (20 Punkte, Streckung 4, 45 Grad, Seed 35) ist ausdrücklich eine einzelne.

| Frage | Befund |
|---|---|
| Handinstanz (Stück 2) | Primal- und Dualwert aller sechs Formen stimmen mit der Handrechnung überein: Würfel 1, Raute 2, Kugel √2 (beide Formen), Ellipse achsenparallel 4, Ellipse allgemein 2,5 |
| Primal gegen Dual, 90 Läufe (6 Formen, 15 Wolken) | Lücke zwischen Primal- und Dualwert höchstens 3,4·10⁻⁸ (Test: unter 10⁻⁷), Residuum der Dualitätsgleichung höchstens 1,5·10⁻¹¹ (die Gleichungen gelten exakt über den Nullraum), alle Prüfungen bestehen |
| Standardwolke | Würfel 5,784; Raute 10,090; Kugel 7,167 (beide Formen); Ellipse achsenparallel 134,27; Ellipse allgemein 78,52. Das Dual hat 77 / 57 / 57 / 117 / 22 / 24 freie Variablen bei 80 / 100 / 60 / 120 / 26 / 30 Zeilen; die PSD-Kugel hat also 117 gegen 57 der Lorentz-Form |
| Dualstart aus dem Pfad | Würfel, Raute und beide Kugeln starten in allen 15 Wolken bei `ν/t ≈ 10⁻³`; die Ellipse achsenparallel braucht in 9 von 15 Wolken ein gröberes Niveau (10⁻² oder 10⁻¹; der Test prüft das Band 4 bis 13), die allgemeine in 1 von 15. An den Reglerecken (n = 8, 20, 30; Streckung 1,2 und 6; Winkel 0, 45, 90 Grad; 108 Läufe) gelingt der Start immer (Test). Ein größeres Raster (n = 8 bis 30, vier Streckungen, vier Winkel, zwei Seeds: 768 Läufe) lief einmal ohne Fehler; die Ellipse achsenparallel startete dort in 66 von 128 Fällen bei 10⁻³, in 62 gröber, in 5 davon erst bei 1 (einmalig gemessen, nicht getestet) |
| Dual direkt aus dem Pfadende (`z = −∇φ/t`) | Erfüllt `Aᵀz = −c` nur bei genauer Zentrierung und mäßigem t: bei `ν/t ≈ 4·10⁻⁴` liegt das Residuum bei 9·10⁻⁹ (fein zentriert) bzw. 1·10⁻⁴ (grob), am Pfadende (`ν/t ≈ 2·10⁻¹¹`) bei 5·10⁻⁴ bzw. 4·10⁻⁴: die Hesse-Matrix wird schlecht konditioniert. Deshalb nimmt die Demo aus dem Pfad nur den Start |
| Newton-Schritte des Dual-Programms | 8 bis 12 (Würfel), 8 bis 11 (Raute), 8 bis 13 (Kugel), 12 bis 22 (Kugel als SDP), 10 bis 67 (Ellipse achsenparallel), 16 bis 38 (Ellipse allgemein). Keine Eigenschaft der Kegel: Implementierung mit festem Faktor 8 für `t` |
| Komplementarität der Kugel (Standardwolke) | Die Dualgewichte `z_i0` summieren sich zu 1; **zwei Stützpunkte** tragen je ½, außerhalb höchstens 2,3·10⁻¹¹; auf den Stützpunkten weichen sie um höchstens 2,0·10⁻¹¹ von den Gewichten aus Stück 1 ab (`Σ λᵢ xᵢ = c`, `Σ λᵢ = 1`); `Σ z̄ᵢ` (Mittelpunkt-Gleichung) 5,5·10⁻¹⁷. Dasselbe auf runden Wolken mit zwei oder drei Stützpunkten (Test bis 10⁻⁶) |
| Beispiel mit Lücke: `min x₁` mit `[[ε, x₁, 0],[x₁, x₂, 0],[0, 0, x₁ + 1]] ⪰ 0` | Bei ε = 0 von 2000 Kandidaten 1000 zulässig, **alle mit x₁ = 0**: Primalwert 0. Jede der 200 zulässigen Dualmatrizen hat den Wert **−1**: **Lücke 1**. Für ε von 0,1 bis 10⁻⁵ (streng zulässig) findet der Löser −0,99976 mit bewiesener Lücke 7,3·10⁻⁴, also −1; x₂ wächst über 10⁸ (der Pfad läuft entlang der unbeschränkten optimalen Menge davon). Die Suche nach einem Dualstart scheitert bei allen 5 Werten von ε |
| Pfad ohne streng zulässiges Dual: `min x₁` mit `[[x₁,1],[1,x₂]] ⪰ 0` | Infimum 0 nie erreicht. Nach 10 Runden x₁ = 1,9·10⁻⁸, x₂ = 8,7·10⁷, und `f_t` hat kein Minimum. Mit streng zulässigem Dual (`min x₁ + x₂`): Optimum 2 bei (1, 1) nach 24 Newton-Schritten; dort gelingt der Dualstart, im ersten Fall scheitert er |
| Schwach unzulässig: `[[x,1],[1,0]] ⪰ 0` | Kleinster Eigenwert −1,0·10⁻⁸ bei x = 10⁸: er geht gegen null, wird nie null. Von 500 Zufallsmatrizen `Z ⪰ 0` mit `Z₁₁ = 0` (nur dann nicht negativ gegen `M₁`) ist der beste Wert für `⟨Z, M₀⟩` 0: kein Zertifikat. Im LP `x ≥ 1, x ≤ 0` dagegen `y = (1, 1)`, `Aᵀy = 0`, `bᵀy = 1 > 0` |
| Fremdlöser (cvxpy 1.9.3, Clarabel 0,11,1, SCS 3,3,1; **versionsabhängig, nicht getestet**) | Beispiel mit Lücke bei ε = 0: Clarabel `SolverError`, SCS „optimal“ mit Wert −0,935 (weder 0 noch −1). Schwach unzulässig: Clarabel `SolverError`, SCS „optimal“ mit Wert 0, obwohl unzulässig. Das LP `x ≥ 1, x ≤ 0` melden beide als „infeasible“. Stand 2026-11-01 |

## Befunde und Korrekturen gegenüber dem Plan

- **Der Dualstart braucht gröbere Niveaus als geplant.** Der Plan sah 10⁻³ bis 10⁻⁵ (und 10⁻² für die Ellipse achsenparallel) vor. Ein Eckfall-Test (Streckung 6, 45 Grad, 20 Punkte) fand eine Wolke, bei der kein Niveau bis 10⁻⁵ einen streng zulässigen Start lieferte und die App abstürzte. Die Niveaus 1 und 10 sind jetzt Rückfälle; zusätzlich meldet die App, wenn auch das scheitert. Im ganzen Raster (768 Läufe) trat der Fall danach nicht mehr auf.
- **Zwei Punkte, allgemeine Ellipse: das Infimum wird nicht angenommen** (nur eine Bedingung, die andere Halbachse darf gegen null gehen). Der Plan erwartete, dass die Suche hier scheitert; sie findet bei einem groben Niveau einen Start, aber das Paar besteht die Prüfung nicht (Dualwert 3,975, Primalwert 5,0, Lücke 1,02, P-Einträge ~10⁹, einmal gemessen). Der Test hält beides fest: entweder Fehler der Suche oder Prüfung schlägt an.
- **BLAS-Threads.** Bei einem ausgelasteten Rechner kostete die Lösung der PSD-Kugel 2,8 bis 4,5 s statt 0,12 s, weil mehrere BLAS-Threads auf kleinen Matrizen (bis 120 × 120) gegeneinander arbeiten. Die Rechnung läuft jetzt mit einem Thread (`threadpoolctl`); gemessen, einmalig, nicht getestet.
- **Beispiel mit Lücke:** Der Wert −1 für die Familie ε > 0 wurde im Plan nur „gemessen“ genannt; die Herleitung steht im 📐-Expander (`x₂ ≥ x₁²/ε` und `x₁ ≥ −1` ist alles, was verlangt wird).
- **Keine Zuschreibung.** Der Plan nannte es „Boyd-Beispiel“. Die Herkunft des Beispiels habe ich nicht an der Quelle geprüft, es ist hier nur ein kleines, von Hand gerechnetes SDP ohne Slater-Punkt.
- **Port:** 9301 (TCP-Prüfung und `launch.json` gelesen, frei).

## Ehrliche Grenzen

- **Der Dualstart ist ein Verfahren, das scheitern kann.** Die Suche über Niveaus lieferte in 768 Läufen immer einen Start, aber eine allgemeine Regel gibt es nicht. Phase I und die homogene Einbettung folgen in Stück 6.
- **Start und Primal durch Konstruktion:** Jede Form liefert einen streng zulässigen Primalstart mit; für allgemeine Programme gibt es das nicht.
- **Die drei Beispiele ohne Slater-Punkt sind klein (2 × 2 und 3 × 3) und handgemacht;** sie zeigen die Phänomene, nicht ihre Häufigkeit in der Praxis.
- **Fremdlöser-Aussagen** gelten für die genannten Versionen und werden nicht getestet.
- **Selbstduale Kegel nur:** der Exponential- und der Potenzkegel (Stück 7) sind es nicht.
- **Zeiten** hängen vom Rechner ab und werden nicht getestet.

## Verifikation

- **Kern** (`tests/test_kern.py`, wortgleich aus Stück 2): Barrieren gegen finite Differenzen, Mini-Programme mit bekanntem Optimum, singuläre Hesse-Matrix.
- **Dual** (`tests/test_dual.py`): `z = −∇φ/t` von Hand; Residuum und Kegelrand von Hand; Nullraum und Korrektur; der Prüfer nennt jede verletzte Regel; Dual der Kugel auf der Handinstanz (Gewichte ½ und ½, Wert √2); jeder Dualpunkt des Programms erfüllt die Gleichung exakt; schwache Dualität auf Paaren aus verschiedenen Pfadniveaus; kein Dualstart ohne streng zulässiges Dual.
- **Beispiele** (`tests/test_examples.py`): jede Zahl aus der Herleitung (nur x₁ = 0 primal zulässig; Dualwert −1 für jedes zulässige Z; Familie ε > 0; Pfad läuft davon mit `∂f_t/∂x₂ = −x₁/(x₁x₂ − 1)`; Eigenwertformel gegen numpy; LP-Zertifikat).
- **Orakel** (`tests/test_oracle_kegel_dualitaet.py`): der Dualwert trifft den cvxpy-Primalwert aller sechs Formen (8 Wolken); Stützpunkte und Gewichte gegen einen unabhängigen Kreis durch zwei oder drei Punkte; das Dual des Würfels von Hand; Komplementarität; Skalierung; Drehinvarianz; Dual des Duals; das Dual des Beispiels mit Lücke in cvxpy (überspringt sich, wenn Clarabel scheitert).
- **Randfälle** (`tests/test_edge_cases.py`): zwei Punkte von Hand, gleiche und kollineare Punkte, d = 1 und 3, ungültige Eingaben, kurzer Pfad, Prüfer erkennt ein verletztes Dual, Zustand der Dualstart-Suche.
- **Verdikte** (`tests/test_verdicts.py`), **Claims** (`tests/test_claims.py`: jede Zahl dieser README), **Oberfläche** (AppTest: jeder Button, jeder Regler an Minimum und Maximum, jede Form, PDF, ein Dualstart-Fehler wird gemeldet statt abzustürzen), Presets, Permalink, Layout, Footer, Laufzeit-Importe.
- **Umfang:** 133 Tests (2 davon, die cvxpy-Orakel, überspringen sich ohne cvxpy), rund 2,5 Minuten (2026-11-01, Windows).
- **Nicht gelaufen:** kein Mutationstest (seit 2026-10-28 nicht mehr Teil des Ablaufs). cvxpy und scipy sind Gegenproben im dev-Satz; ohne sie überspringen sich die jeweiligen Tests.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kegel_kern.py` | Gemeinsamer Kern der Reihe (Kegelform, Barrieren, Barrierenmethode, Prüfer), wortgleich mit Stück 2 |
| `kdu_forms.py` | Die sechs Formen als Kegelprogramm samt Rückumformung (aus Stück 2 übernommen, ohne LP-Nachbildung) |
| `kdu_dual.py` | Das Dual: Pfad, Dualpunkt, Nullraum, Dualstart, Dual als Kegelprogramm, Prüfer, Komplementarität |
| `kdu_examples.py` | Die drei Beispiele ohne Slater-Punkt und das LP-Gegenstück |
| `kdu_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `kdu_evaluation.py` | Analyse, Experimente, Verdikte, Formatierung |
| `kdu_layout.py` | Fester Kopf (Stylesheet) |
| `kdu_pdf_export.py` | PDF-Export (fpdf2) |
| `kdu_presets.py` | Permalink, Presets (Schnellstart, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kdu_scenario.py` | Punktwolken (gestreckt, gedreht), Handinstanz |
| `kdu_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und cvxpy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Kern, Dual, Beispiele, Orakel, Randfälle, Verdikte, Claims, Presets, Layout, Abbildungen, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Zentraler Pfad im Detail, Barriere-Parameter, Iterationsschranken:** Stück 5. **Primal-Dual-Schritt, allgemeiner Start (Phase I) und Einbettung:** Stück 6. **Exponential- und Potenzkegel (nicht selbstdual):** Stück 7.
- **Fremdlöser in der App:** keine Laufzeit-Abhängigkeit von cvxpy, Clarabel oder SCS; ihre Statusmeldungen stehen nur hier.
- **Zufallsfamilien ohne Slater-Punkt:** nur die drei handgerechneten Beispiele.
- **Sensitivität und Schattenpreise im Kegel:** das LP-Gegenstück liegt in `lp-dualitaet-demo`.
- **Mutationstest:** nicht Teil des Ablaufs.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, fpdf2 und threadpoolctl.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
