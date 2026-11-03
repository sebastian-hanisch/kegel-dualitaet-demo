"""Kegel-Dualität – die Lücke im Kegel und wo Slater fehlt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 4 der Reihe „Konvexe und konische Optimierung“: Zu jedem Kegelprogramm `min cᵀx, s = b − A x ∈ K` gehört ein Dual im selben Format (`max −bᵀz, Aᵀz = −c, z ∈ K`); der gemeinsame
Barrierlöser (kegel_kern.py, unverändert aus Stück 2) löst beide, die Lücke ist das Zertifikat. Drei kleine SDPs ohne Slater-Punkt zeigen, was im Kegel anders ist als im LP.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import kdu_constants as C
import kdu_dual as D
import kdu_forms as A
from kdu_evaluation import (
    Settings,
    analyse,
    gapex_experiment,
    complementarity_data,
    de,
    level_text,
    nonattained_experiment,
    pair_rows,
    path_quality,
    sci,
    verdict_gapex,
    verdict_complementarity,
    verdict_form,
    verdict_nonattained,
    verdict_pairs,
    verdict_quality,
    verdict_weak,
    weak_experiment,
)
from kdu_layout import dunkles_design, kopf_css
from kdu_pdf_export import build_pdf
from kdu_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kdu_visualization import (
    build_gapex,
    build_complementarity,
    build_gaps,
    build_nonattained,
    build_quality,
    build_shape,
    build_weak,
)

st.set_page_config(page_title="Kegel-Dualität – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _quality(form, n, stretch, angle, seed):
    return path_quality(form, analyse_cloud(n, stretch, angle, seed))


def analyse_cloud(n, stretch, angle, seed):
    from kdu_scenario import make_cloud

    return make_cloud(n, stretch, angle, seed).points


@st.cache_data(show_spinner=False)
def _compl(n, stretch, angle, seed):
    return complementarity_data(analyse_cloud(n, stretch, angle, seed))


@st.cache_data(show_spinner=False)
def _gapex():
    return gapex_experiment()


@st.cache_data(show_spinner=False)
def _nonattained():
    return nonattained_experiment()


@st.cache_data(show_spinner=False)
def _weak():
    return weak_experiment()


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kdu_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Kegel-Dualität – die Lücke im Kegel und wo Slater fehlt")
st.markdown(
    """
**Zu jedem Kegelprogramm gehört ein Dual im selben Format.** Aus `min cᵀx` mit `s = b − A x ∈ K` wird `max −bᵀz` mit `Aᵀz = −c` und `z ∈ K` (Orthant, Lorentz- und PSD-Kegel sind selbstdual).
Für beliebige zulässige `x` und `z` gilt `cᵀx + bᵀz = zᵀs ≥ 0`: **die Lücke ist das Zertifikat**, wie die untere Schranke aus Stück 1. Hier löst derselbe Kern das Primal und das Dual der sechs Formen aus Stück 2.
Im LP gilt starke Dualität, sobald eines von beiden zulässig ist; **im Kegel garantiert ein streng zulässiger Punkt (Slater) starke Dualität, ohne ihn kann eine Lücke bleiben**. Drei kleine SDPs ohne Slater-Punkt zeigen eine Lücke von 1, einen zentralen Pfad, der davonläuft, und ein System, das unzulässig ist, ohne dass ein Zertifikat existiert.
Stück 4 der Reihe „Konvexe und konische Optimierung“.

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Kegel-Dual", expanded=True):
    st.markdown(
        """
1. **Das Dual hinschreiben:** Jede Zeile von `s = b − A x` bekommt eine Dualvariable `zᵢ` im selben Kegel. Dann gilt `cᵀx + bᵀz = zᵀs`, und `zᵀs ≥ 0`, weil `z` und `s` im selbstdualen Kegel liegen.
2. **Einen Startpunkt aus dem Primal holen:** Die Barrierenmethode des Primals gibt bei jedem `t` den Punkt `z = −∇φ(s)/t`; er liegt im Inneren des Kegels, erfüllt `Aᵀz = −c` aber nur, wenn `x` genau zentriert ist. Die Demo korrigiert ihn auf die Gleichung (kleinste Änderung).
3. **Das Dual als Programm lösen:** `z = z₀ + N y` mit einer Basis `N` des Nullraums von `Aᵀ` erfüllt die Gleichung exakt. `min bᵀz` über `y` ist wieder ein Kegelprogramm im selben Format und wird mit demselben Kern gelöst.
4. **Vergleichen:** `p* − d* = cᵀx + bᵀz`. Bei allen sechs Formen ist sie null bis auf die geforderte Genauigkeit; wo kein streng zulässiger Punkt existiert, ist sie es nicht.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Form**")
    form = st.selectbox("Gesuchte Form", A.FORMS, key="form_select", format_func=A.FORM_NAMES.get,
                        help="Für welche der sechs Formen das Hauptdiagramm, die Kernfrage und die Experimente 1b und 2 gelten; die Tabelle vergleicht immer alle sechs.")
    st.markdown("**Punktwolke**")
    n = st.slider("Punkte n", *bounds("n_slider"), key="n_slider", help="Wie viele Punkte die Wolke hat.")
    stretch = st.slider("Streckung", *bounds("stretch_slider"), step=0.1, key="stretch_slider", help="Wie viel breiter die Wolke entlang ihrer Längsachse ist als quer dazu.")
    angle = st.slider("Drehung (Grad)", *bounds("angle_slider"), key="angle_slider", help="Winkel der Längsachse gegen die x-Achse.")
    seed = st.number_input("Zufalls-Seed der Wolke", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Wolke generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"form_select": form, "n_slider": int(n), "stretch_slider": float(stretch), "angle_slider": int(angle), "seed_input": int(seed)})

settings = Settings(form, int(n), float(stretch), int(angle), int(seed))
a = _analysis(settings)
rows = pair_rows(a)
if a.selected_failed:   # die Suche über Lückenniveaus fand keinen streng zulässigen Dualstart: melden, nicht abstürzen
    st.error(f"Für {A.FORM_NAMES[form]} findet die Suche keinen streng zulässigen Dualstart ({a.selected_failed}). Eine andere Form oder Wolke wählen.")
    st.stop()
row = next(r for r in rows if r["form"] == form)
sel = a.selected

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if not sel.ok:
    st.error("Eine Prüfung besteht nicht: " + "; ".join(sel.primal_problems + sel.dual_problems))
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Primalwert", de(row["primal"], 6), help=f"Zielwert des Primals ({A.objective_name(form)}).")
m2.metric("Dualwert", de(row["dual"], 6), help="−bᵀz des Duals, gelöst als eigenes Kegelprogramm.")
m3.metric("Lücke Primal − Dual", sci(row["gap"]), help="cᵀx + bᵀz: null im Optimum, wenn starke Dualität gilt.")
m4.metric("Komplementarität zᵀs", sci(row["complementarity"]), help="Summe der z_iᵀs_i über alle Kegel; im Optimum null (komplementärer Schlupf).")
m5.metric("Residuum Aᵀz + c", sci(row["residual"]), help="Wie genau das Dual die Gleichung erfüllt; exakt bis auf Rundung, weil z im Nullraum-Ansatz gerechnet wird.")
weights = None
if form == "ball_soc":
    weights = sel.pair.z.reshape(-1, 3)[:, 0]
st.plotly_chart(build_shape(a.cloud.points, form, A.decode(form, a.cloud.points, sel.pair.primal.x), weights), width="stretch", key="main_chart")
st.caption("Die Form ist die Lösung des Primals" + ("; große Punkte tragen Gewicht im Dual (Stützpunkte)." if weights is not None else "; bei der Kugel im Lorentz-Kegel zeigt die Punktgröße die Dualgewichte.") +
           " Die Wolke ist um ihren Mittelwert verschoben; Würfel, Raute und Kugel haben einen frei gewählten Mittelpunkt, die Ellipsen liegen im Ursprung.")

if st.button("📄 PDF der aktuellen Wolke erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="kegel-dualitaet.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Stimmt das Dual mit dem Primal überein?")
verdict(verdict_form(row))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Das Dual aus dem Löser")
st.caption("Alle sechs Formen derselben Wolke: Primal, Dual, Lücke, Größe des Dualprogramms. Das Dual wird nicht aus dem Pfadende gelesen, sondern als eigenes Programm gelöst; die zweite Abbildung zeigt, warum.")
t1, t2 = st.columns(2)
t1.plotly_chart(build_gaps(rows), width="stretch", key="gap_chart")
quality = _quality(form, settings.n, settings.stretch, settings.angle, settings.seed)
t2.plotly_chart(build_quality(quality), width="stretch", key="quality_chart")
st.dataframe(pd.DataFrame([{"Form": r["name"], "Primal": round(r["primal"], 6), "Dual": round(r["dual"], 6), "Lücke": sci(r["gap"]), "zᵀs": sci(r["complementarity"]),
                            "Residuum": sci(r["residual"]), "Dualstart bei ν/t": level_text(r["level"]), "Dualvariablen": r["dual_variables"], "Zeilen": r["rows"],
                            "Newton Primal": r["newton_primal"], "Newton Dual": r["newton_dual"]} for r in rows]), hide_index=True, width="stretch")
verdict(verdict_pairs(rows))
verdict(verdict_quality(quality))

st.markdown("---")

st.subheader("🔬 Experiment 2: Komplementarität – wer trägt das Gewicht?")
st.caption("Die kleinste Kugel der eingestellten Wolke in der Lorentz-Form: Gewicht `z_i0` und Schlupf `R − Abstand` je Punkt. Wo Gewicht liegt, ist der Schlupf null; die Gewichte sind die aus Stück 1.")
compl = _compl(settings.n, settings.stretch, settings.angle, settings.seed)
st.plotly_chart(build_complementarity(compl), width="stretch", key="compl_chart")
verdict(verdict_complementarity(compl))

st.markdown("---")

st.subheader("🔬 Experiment 3: Ohne Slater – eine Lücke von 1")
st.caption("`min x₁` mit `[[ε, x₁, 0],[x₁, x₂, 0],[0, 0, x₁ + 1]] ⪰ 0`. Bei ε = 0 gibt es keinen streng zulässigen Punkt (und im Dual auch nicht): Primal und Dual haben verschiedene Werte, beide werden angenommen.")
gapex = _gapex()
st.plotly_chart(build_gapex(gapex), width="stretch", key="gapex_chart")
verdict(verdict_gapex(gapex))

st.markdown("---")

st.subheader("🔬 Experiment 4: Der zentrale Pfad läuft davon")
st.caption("`min x₁` mit `[[x₁, 1],[1, x₂]] ⪰ 0`: das Infimum 0 wird nie erreicht (`x₁x₂ ≥ 1`). Je Runde t mal 8 und sechs Newton-Schritte; zum Vergleich dieselbe Nebenbedingung mit streng zulässigem Dual.")
nonatt = _nonattained()
st.plotly_chart(build_nonattained(nonatt), width="stretch", key="nonatt_chart")
verdict(verdict_nonattained(nonatt))

st.markdown("---")

st.subheader("🔬 Experiment 5: Unzulässig, ohne dass es ein Zertifikat gibt")
st.caption("`[[x, 1],[1, 0]] ⪰ 0` ist für kein x erfüllt, kommt dem Kegel aber beliebig nahe. Im LP hat jede Unzulässigkeit ein Farkas-Zertifikat.")
weak = _weak()
st.plotly_chart(build_weak(weak), width="stretch", key="weak_chart")
verdict(verdict_weak(weak))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Primal ist streng zulässig, und der Start ist bekannt** | Jede Form liefert einen Start durch Konstruktion mit. Für allgemeine Programme gibt es das nicht; Phase I und Einbettung lösen es. | Stück 6 |
| **Der Dualstart gelingt bei einem der Niveaus 10⁻¹ bis 10⁻⁵** | Die Suche über Niveaus ist ein Verfahren, das scheitern kann (z₀ liegt dann nicht streng im Kegel); sie meldet das statt einer stillen Zahl. | Stück 6 |
| **Beide Seiten sind streng zulässig** | Sonst gibt es keinen zentralen Pfad (Experiment 4), die Optimalwerte können auseinanderliegen (Experiment 3) oder nicht angenommen werden. | – |
| **Der Löser meldet ehrlich** | Fremdlöser können bei Problemen ohne Slater-Punkt „optimal“ melden, obwohl weder Primal- noch Dualwert stimmen (siehe README, versionsabhängig). | – |
| **Selbstduale Kegel** | Orthant, Lorentz- und PSD-Kegel sind selbstdual; der Exponential- und der Potenzkegel sind es nicht (ihr Dual ist ein anderer Kegel). | Stück 7 |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Primal und Dual.** $\min\; c^\top x$ unter $s = b - Ax \in K$, mit $K$ aus Orthant, Lorentz- und PSD-Kegeln (alle selbstdual: $K^* = K$). Lagrange-Funktion mit $z \in K^*$:
$L(x, z) = c^\top x + z^\top (Ax - b + s)$ ist für $A^\top z = -c$ gleich $-b^\top z + z^\top s$. Das Dual ist $\max\; -b^\top z$ unter $A^\top z = -c$, $z \in K$.

**Schwache Dualität.** Für zulässige $x$ und $z$ gilt $c^\top x + b^\top z = z^\top s \ge 0$, denn $z, s \in K = K^*$. Die Lücke $z^\top s$ misst den Abstand zum Optimum von beiden Seiten.

**Starke Dualität.** Gibt es ein $x$ mit $s \in \operatorname{int} K$ (Slater für das Primal) und ist der Optimalwert endlich, so ist der Dualwert gleich dem Primalwert und wird angenommen. Gilt das Entsprechende für das Dual, wird der Primalwert angenommen.

**Zentraler Pfad.** $x(t)$ minimiert $t\,c^\top x + \phi(b - Ax)$; dort ist $z(t) = -\nabla\phi(s)/t$ dual zulässig und $z(t)^\top s = \nu/t$. Der Pfad existiert genau dann, wenn Primal und Dual streng zulässig sind (sonst ist $f_t$ nicht nach unten beschränkt oder hat kein Minimum).

**Das Dual als Kegelprogramm.** Sei $z_0$ eine Lösung von $A^\top z_0 = -c$ im Inneren von $K$ und $N$ eine Basis des Nullraums von $A^\top$. Dann ist $z = z_0 + N y$ jede dual zulässige Gleichung, und $\min\, b^\top z = b^\top z_0 + (N^\top b)^\top y$ unter $z_0 + N y \in K$ ist ein Kegelprogramm mit $A' = -N$, $b' = z_0$, $c' = N^\top b$.

**Die Kugel.** Mit $z_i = (z_{i0}, \bar z_i) \in \mathcal{Q}$: $\sum_i z_{i0} = 1$ (die Koordinate $R$), $\sum_i \bar z_i = 0$ (die Mittelpunkt-Koordinaten), Wert $\max\, -\sum_i x_i^\top \bar z_i$. Im Optimum sind die Gewichte $\lambda_i = z_{i0}$ auf den Stützpunkten die aus Stück 1 ($\sum \lambda_i x_i = c$, $\sum \lambda_i = 1$). Das Dual aus Stück 1 gehört zur quadrierten Aufgabe (Wert $R^{*2}$), das Kegel-Dual zum Wert $R^*$.

**Beispiel mit Lücke.** $\min x_1$ unter $M(x) = \begin{psmallmatrix} 0 & x_1 & 0 \\ x_1 & x_2 & 0 \\ 0 & 0 & x_1 + 1 \end{psmallmatrix} \succeq 0$. Der (1,1)-Eintrag ist null, also muss $x_1 = 0$ sein: $p^* = 0$. Dual: $Z \succeq 0$ mit $\langle Z, M_2 \rangle = Z_{22} = 0$ (also $Z_{12} = 0$) und $\langle Z, M_1 \rangle = 2 Z_{12} + Z_{33} = 1$: $Z_{33} = 1$, Wert $-\langle Z, M_0\rangle = -Z_{33} = -1$: $d^* = -1$, **Lücke 1**.
Mit $\varepsilon > 0$ im (1,1)-Eintrag ist $x_2 \ge x_1^2/\varepsilon$ und $x_1 \ge -1$ alles, was verlangt wird: der Wert ist $-1$ für jedes $\varepsilon > 0$.

**Nicht angenommen.** $\min x_1$ unter $\begin{psmallmatrix} x_1 & 1 \\ 1 & x_2 \end{psmallmatrix} \succeq 0$ heißt $x_1 x_2 \ge 1$, $x_1 > 0$: Infimum 0, nicht angenommen. $f_t = t x_1 - \log(x_1 x_2 - 1)$ hat $\partial f_t / \partial x_2 = -x_1/(x_1 x_2 - 1) < 0$: es fällt mit $x_2$ ins Unendliche, es gibt kein Minimum.

**Schwach unzulässig.** $\begin{psmallmatrix} x & 1 \\ 1 & 0 \end{psmallmatrix}$ hat den kleinsten Eigenwert $(x - \sqrt{x^2 + 4})/2 \approx -1/x$: nie $\ge 0$, aber beliebig nahe. Ein Zertifikat wäre $Z \succeq 0$ mit $\langle Z, M_1\rangle = Z_{11} = 0$ und $\langle Z, M_0\rangle = 2Z_{12} < 0$; $Z_{11} = 0$ erzwingt $Z_{12} = 0$ (Positivsemidefinitheit), also gibt es keins.
Im LP $x \ge 1$, $x \le 0$ ist $y = (1, 1)$ mit $A^\top y = 0$ und $b^\top y = 1 > 0$ ein Zertifikat (Farkas).

**Literatur.** S. Boyd und L. Vandenberghe: *Convex Optimization*, Cambridge University Press 2004, Kapitel 5 (Dualität) und Kapitel 11 (Innere-Punkte-Verfahren, zentraler Pfad). L. Vandenberghe und S. Boyd: *Semidefinite programming*, SIAM Review 38(1), 1996, S. 49–95 (Dualität und Slater-Bedingungen bei SDP). Dort steht auch, dass die Dualitätsaussagen für SDP schwächer sind als für LP. Die drei Beispiele ohne Slater-Punkt sind klein und von Hand durchgerechnet.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
