from __future__ import annotations

import textwrap
from typing import List

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from analysis.quality import VERDICT_ADVICE, assess_quality, criteria_lines
from stats.heatmap import infer_cm_per_px

A4 = (8.27, 11.69)  # pulgadas; la misma página que el resto del reporte


def calibration_lines(result) -> List[str]:
    """Escala, origen XY, ROI y tamaño del video usados en la corrida (la primera línea es siempre la escala)."""
    s = getattr(result, "settings", None) or {}
    records = getattr(result, "records", None) or []

    cm = s.get("escala_cm_por_px")
    deducida = False
    if cm is None and "escala_cm_por_px" not in s:  # resultado de una versión que no guardaba la calibración
        cm = infer_cm_per_px(records)
        deducida = cm is not None

    if cm:
        escala = f"Escala: 1 píxel = {cm:.4g} cm ({1.0 / cm:.4g} píxeles por cm)"
        if deducida:
            escala += " [deducida de los datos]"
    else:
        escala = "Escala: no calibrada (distancias y velocidades en píxeles)"
    lines = [escala]

    origen = s.get("origen_xy_px")
    if origen:
        lines.append(f"Origen del sistema XY: ({origen[0]:g}, {origen[1]:g}) px de imagen; "
                     "x_rel = x − ox, y_rel = oy − y (y hacia arriba)")
    elif "origen_xy_px" in s:
        lines.append("Origen del sistema XY: no definido")

    roi = s.get("roi_box")
    if roi:
        lines.append(f"ROI: esquina superior izquierda ({roi[0]}, {roi[1]}) px; tamaño {roi[2]} × {roi[3]} px")
    tam = s.get("video_tamano_px")
    if tam:
        lines.append(f"Video: {tam[0]} × {tam[1]} px")
    return lines


def quality_lines(result) -> List[str]:
    """Veredicto, recomendación, valores medidos, motivos y las reglas de clasificación."""
    records = getattr(result, "records", None)
    if not records:
        return ["No se calculó (no se guardaron los registros por frame)."]
    rep = assess_quality(records, result.timing.dt)
    lines = [
        f"Veredicto: {rep.verdict.upper()}",
        f"Recomendación: {VERDICT_ADVICE.get(rep.verdict, '')}",
        "",
        f"Frames detectados: {rep.frames_detected} de {rep.frames_total} ({rep.lost_pct:.1f} % perdidos)",
        f"Hueco más largo: {rep.longest_gap_s:.2f} s; interrupciones: {rep.n_interruptions}",
        f"Saltos de posición sospechosos: {rep.n_jumps}",
    ]
    if rep.frames_rejected or rep.frames_recovered:
        lines.append(f"Descartados por el filtro de saltos: {rep.frames_rejected} ({rep.rejected_pct:.1f} %); "
                     f"recuperados: {rep.frames_recovered}")
    if rep.reasons:
        lines += ["", "Motivos de este veredicto:"] + [f"  - {r}" for r in rep.reasons]
    lines += ["", "Cómo se clasifica:"] + [f"  - {c}" for c in criteria_lines(rep.thresholds)]
    return lines


def _wrap(lines: List[str], width: int) -> List[str]:
    out: List[str] = []
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue
        sangria = " " * (len(ln) - len(ln.lstrip()))
        out += textwrap.wrap(ln, width=width, subsequent_indent=sangria + "    ") or [""]
    return out


def page_text(result) -> List[str]:
    """Todas las líneas de la página, ya ordenadas (sin ajustar al ancho)."""
    return (["CALIBRACIÓN"] + calibration_lines(result) + [""]
            + ["CALIDAD DEL SEGUIMIENTO"] + quality_lines(result))


def report_page(result) -> Figure:
    """Una página A4 con la calibración y la calidad. Reduce la letra si el texto es largo."""
    fig = Figure(figsize=A4)
    FigureCanvasAgg(fig)
    fig.suptitle("TRACKERLAB - CALIBRACIÓN Y CALIDAD", fontsize=18, fontweight="bold", y=0.965)
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.84])
    ax.axis("off")
    base = page_text(result)
    texto, size = _wrap(base, 82), 10.5
    if len(texto) > 44:  # una página A4 admite ~49 líneas a 10.5 pt; se deja margen
        texto, size = _wrap(base, 94), 9.0
    ax.text(0.0, 1.0, "\n".join(texto), va="top", ha="left", fontsize=size, linespacing=1.35, transform=ax.transAxes)
    return fig


def report_pages(result) -> List[Figure]:
    """Páginas extra para el PDF del experimento (hoy una): ``for p in report_pages(result): pdf.savefig(p)``."""
    return [report_page(result)]