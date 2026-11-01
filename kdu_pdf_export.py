"""PDF-Export der aktuellen Wolke mit Primal und Dual aller sechs Formen (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from kdu_evaluation import Analysis, de, level_text, pair_rows, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "€": "EUR", "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3",
            "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Kegel-Dualität: Primal und Dual der sechs Formen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Wolke: {s.n} Punkte in der Ebene, Streckung {de(s.stretch, 1)}, um {s.angle} Grad gedreht, Seed {s.seed}; um den Mittelwert verschoben. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Form", "Primal", "Dual", "Lücke", "z^T s", "Residuum", "Start bei nu/t", "Dualvariablen", "Newton P/D"], [50, 28, 28, 24, 24, 24, 28, 28, 24]
    pdf.set_font("Helvetica", "B", 8)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for r in pair_rows(a):
        for v, w in zip([r["name"], de(r["primal"], 6), de(r["dual"], 6), sci(r["gap"]), sci(r["complementarity"]), sci(r["residual"]), level_text(r["level"]),
                         r["dual_variables"], f"{r['newton_primal']}/{r['newton_dual']}"], widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Primal: min c^T x mit s = b - A x im Kegel; Dual: max -b^T z mit A^T z = -c, z im Kegel; Lücke = c^T x + b^T z = z^T s. "
                            "Quelle: kegel-dualitaet-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
