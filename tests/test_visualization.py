"""Abbildungen: Achsen gesperrt, Umrisse stimmen mit den Formen überein, Diagramme bauen an den Rändern."""

import numpy as np
import pytest

import kdu_evaluation as ev
import kdu_forms as F
import kdu_visualization as V


def _locked(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


@pytest.fixture(scope="module")
def a():
    return ev.analyse(ev.Settings(n=12))


def test_outlines_by_hand():
    x, y = V.outline(F.Shape("box", np.array([1.0, 2.0]), 3.0))
    assert (x.min(), x.max(), y.min(), y.max()) == (-2.0, 4.0, -1.0, 5.0)
    x, y = V.outline(F.Shape("diamond", np.array([0.0, 0.0]), 2.0))
    assert sorted(set(zip(x.round(6), y.round(6)))) == [(-2.0, 0.0), (0.0, -2.0), (0.0, 2.0), (2.0, 0.0)]
    x, y = V.outline(F.Shape("ball", np.array([1.0, 1.0]), 2.0))
    assert np.hypot(x - 1.0, y - 1.0) == pytest.approx(2.0)
    x, y = V.outline(F.Shape("ellipse", np.zeros(2), matrix=np.diag([1.0, 4.0])))
    assert (abs(x).max(), abs(y).max()) == pytest.approx((1.0, 0.5), abs=1e-3) and np.allclose(x ** 2 + 4 * y ** 2, 1.0)


def test_shape_chart_names_weights_only_when_given(a):
    for form in F.FORMS:
        shape = F.decode(form, a.cloud.points, a.pairs[form].pair.primal.x)
        plain = V.build_shape(a.cloud.points, form, shape)
        assert _locked(plain) and plain.layout.yaxis.scaleanchor == "x" and plain.data[1].name == "Punkte"
    weights = a.pairs["ball_soc"].pair.z.reshape(-1, 3)[:, 0]
    fig = V.build_shape(a.cloud.points, "ball_soc", F.decode("ball_soc", a.cloud.points, a.pairs["ball_soc"].pair.primal.x), weights)
    assert fig.data[1].name == "Punkte (Größe = Dualgewicht)" and max(fig.data[1].marker.size) == pytest.approx(28.0, abs=1e-3) and sum(np.array(fig.data[1].marker.size) > 8.5) == 2 and min(fig.data[1].marker.size) == pytest.approx(8.0, abs=1e-3)


def test_experiment_charts_build_and_are_locked(a):
    rows = ev.pair_rows(a)
    gaps = V.build_gaps(rows)
    assert _locked(gaps) and gaps.layout.yaxis.type == "log" and len(gaps.data[0].y) == 6
    quality = V.build_quality(ev.path_quality("ball_soc", a.cloud.points))
    assert _locked(quality) and quality.layout.xaxis.autorange == "reversed" and [t.name for t in quality.data][0].startswith("Residuum, grob")
    compl = V.build_complementarity(ev.complementarity_data(a.cloud.points))
    assert _locked(compl) and len(compl.data) == 2 and len(compl.data[0].y) == 12
    gap = V.build_gapex(ev.gapex_experiment())
    assert _locked(gap) and [t.name for t in gap.data] == ["Wert für ε > 0", "Primal bei ε = 0", "Dual bei ε = 0"]
    non = V.build_nonattained(ev.nonattained_experiment())
    assert _locked(non) and non.layout.yaxis.type == "log" and [t.name for t in non.data] == ["x₁", "x₂"]
    weak = V.build_weak(ev.weak_experiment())
    assert _locked(weak) and weak.layout.xaxis.type == "log" and weak.data[-1].name == "1/x"
