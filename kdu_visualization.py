"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kdu_constants as C
import kdu_forms as A


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def outline(shape: A.Shape) -> tuple[np.ndarray, np.ndarray]:
    """Umriss der Form als Linienzug (x, y): Würfel = Quadrat, Raute = Quadrat auf der Spitze, Kugel = Kreis, Ellipse aus x^T P x = 1."""
    t = np.linspace(0.0, 2.0 * math.pi, 241)
    cx, cy = shape.center
    if shape.kind == "ball":
        return cx + shape.radius * np.cos(t), cy + shape.radius * np.sin(t)
    if shape.kind == "box":
        r = shape.radius
        return np.array([cx - r, cx + r, cx + r, cx - r, cx - r]), np.array([cy - r, cy - r, cy + r, cy + r, cy - r])
    if shape.kind == "diamond":
        r = shape.radius
        return np.array([cx + r, cx, cx - r, cx, cx + r]), np.array([cy, cy + r, cy, cy - r, cy])
    eig, vec = np.linalg.eigh(shape.matrix)
    pts = vec @ (np.stack([np.cos(t), np.sin(t)]) / np.sqrt(eig)[:, None])
    return cx + pts[0], cy + pts[1]


def build_shape(points: np.ndarray, form: str, shape: A.Shape, weights: np.ndarray | None = None):
    """Punktwolke und die gefundene Form; Punktgröße nach Dualgewicht (soweit gegeben), gleicher Maßstab auf beiden Achsen."""
    fig = go.Figure()
    xs, ys = outline(shape)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=A.FORM_NAMES[form], line=dict(color=C.COLORS[form], width=3), hoverinfo="skip"))
    sizes = 8.0 if weights is None else 8.0 + 40.0 * np.clip(weights, 0.0, 1.0)
    fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers", name="Punkte" if weights is None else "Punkte (Größe = Dualgewicht)",
                             marker=dict(size=sizes, color=C.COLOR_POINT, line=dict(width=1, color="#444")), hovertemplate="(%{x:.2f}; %{y:.2f})<extra></extra>"))
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y", scaleanchor="x", scaleratio=1)
    return _base(fig, 440)


def build_gaps(rows: list[dict]):
    """Lücke zwischen Primal- und Dualwert je Form (logarithmisch): alle liegen bei der geforderten Genauigkeit."""
    fig = go.Figure(go.Bar(x=[r["name"] for r in rows], y=[max(r["gap"], 1e-16) for r in rows], marker_color=[C.COLORS[r["form"]] for r in rows],
                           hovertemplate="%{x}: %{y:.2e}<extra></extra>"))
    fig.update_yaxes(title_text="Primal minus Dual (logarithmisch)", type="log")
    fig.update_layout(showlegend=False)
    return _base(fig)


def build_quality(rows: list[dict]):
    """Residuum der Dualitätsgleichung für z = -grad phi / t über die Lücke nu/t, bei grobem und feinem Zentrieren; die Lücke selbst als Referenz."""
    gaps = [r["gap"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gaps, y=[max(r["residual_coarse"], 1e-17) for r in rows], mode="lines+markers", name="Residuum, grob zentriert (1e-6)", line=dict(color=C.COLOR_DUAL, width=2)))
    fig.add_trace(go.Scatter(x=gaps, y=[max(r["residual_fine"], 1e-17) for r in rows], mode="lines+markers", name="Residuum, fein zentriert (1e-12)", line=dict(color=C.COLOR_PRIMAL, width=2)))
    fig.update_xaxes(title_text="Lücke ν/t (logarithmisch, nach links wächst t)", type="log", autorange="reversed")
    fig.update_yaxes(title_text="‖Aᵀz + c‖ (logarithmisch)", type="log")
    return _base(fig)


def build_complementarity(data: dict):
    """Je Punkt Dualgewicht z_i0 (oben) und Schlupf R - Abstand (unten): wo Gewicht liegt, ist der Schlupf null."""
    n = len(data["weights"])
    idx = np.arange(1, n + 1)
    support = set(int(i) for i in data["support"])
    colors = [C.COLOR_DUAL if i in support else C.COLOR_POINT for i in range(n)]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Dualgewicht z_i0", "Schlupf R − Abstand zum Mittelpunkt"))
    fig.add_trace(go.Bar(x=idx, y=data["weights"], marker_color=colors, hovertemplate="Punkt %{x}: %{y:.4f}<extra></extra>", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=idx, y=data["slack"], marker_color=colors, hovertemplate="Punkt %{x}: %{y:.4f}<extra></extra>", showlegend=False), row=2, col=1)
    fig.update_xaxes(title_text="Punkt", row=2, col=1)
    return _base(fig, 440)


def build_gapex(res: dict):
    """Wert der streng zulässigen Familie über eps (logarithmisch) mit den Werten bei eps = 0: Primal 0 und Dual -1."""
    sweep = res["sweep"]
    eps = [r["eps"] for r in sweep]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eps, y=[r["value"] for r in sweep], mode="lines+markers", name="Wert für ε > 0", line=dict(color=C.COLOR_DUAL, width=3)))
    fig.add_trace(go.Scatter(x=[min(eps) / 3.0], y=[res["primal_value"]], mode="markers", name="Primal bei ε = 0", marker=dict(symbol="diamond", size=14, color=C.COLOR_PRIMAL)))
    fig.add_trace(go.Scatter(x=[min(eps) / 3.0], y=[res["dual_value"]], mode="markers", name="Dual bei ε = 0", marker=dict(symbol="x", size=14, color=C.COLOR_REF)))
    fig.update_xaxes(title_text="ε (logarithmisch; die Marken stehen links davon bei ε = 0)", type="log", autorange="reversed")
    fig.update_yaxes(title_text="Optimalwert")
    return _base(fig)


def build_nonattained(res: dict):
    """x1 und x2 über die Runden (logarithmisch): x1 fällt gegen null, x2 wächst ohne Grenze."""
    rows = res["rows"]
    r = [row["round"] for row in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=[row["x1"] for row in rows], mode="lines+markers", name="x₁", line=dict(color=C.COLOR_PRIMAL, width=3)))
    fig.add_trace(go.Scatter(x=r, y=[row["x2"] for row in rows], mode="lines+markers", name="x₂", line=dict(color=C.COLOR_DUAL, width=3)))
    fig.update_xaxes(title_text="Runde (t mal 8 je Runde)")
    fig.update_yaxes(title_text="Wert (logarithmisch)", type="log")
    return _base(fig)


def build_weak(res: dict):
    """Betrag des kleinsten Eigenwerts über x (doppelt logarithmisch) mit der Referenz 1/x: er fällt, ohne je null zu werden."""
    rows = res["rows"]
    xs = [r["x"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[abs(r["eig"]) for r in rows], mode="lines+markers", name="|kleinster Eigenwert|", line=dict(color=C.COLOR_DUAL, width=3)))
    fig.add_trace(go.Scatter(x=xs, y=[1.0 / x for x in xs], mode="lines", name="1/x", line=dict(color=C.COLOR_REF, dash="dash", width=1.5)))
    fig.update_xaxes(title_text="x (logarithmisch)", type="log")
    fig.update_yaxes(title_text="Abstand zum PSD-Kegel (logarithmisch)", type="log")
    return _base(fig)
