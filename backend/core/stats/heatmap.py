from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from analysis.occupancy import detected_points, occupancy_map

__all__ = ["PlotContext", "infer_cm_per_px", "create_heatmap_plot", "create_trajectory_plot"]


@dataclass
class PlotContext:
    """Información opcional del experimento para dibujar mejor. Todo es opcional."""

    cm_per_px: Optional[float] = None
    frame_size: Optional[Tuple[int, int]] = None  # (ancho, alto) del video
    roi_box: Optional[Tuple[int, int, int, int]] = None  # (x0, y0, w0, h0) en px de imagen
    xy_origin: Optional[Tuple[float, float]] = None  # (ox, oy) en px de imagen
    background_bgr: Optional[np.ndarray] = None  # frame sobre el que dibujar
    zones: Sequence = ()  # objetos con .name y .mask (máscara del FRAME COMPLETO)


def infer_cm_per_px(records) -> Optional[float]:
    """Escala deducida de los registros (mediana de dist_cm / dist_px), o None si no hay."""
    ratios = [r.dist_cm / r.dist_px for r in records
              if r.dist_px is not None and r.dist_cm is not None and r.dist_px > 1e-9]
    return float(np.median(ratios)) if ratios else None



# Vista: de píxeles de imagen (Y hacia arriba) a las unidades que se muestran
@dataclass(frozen=True)
class _View:
    k: float  # unidades mostradas por píxel (cm por px, o 1)
    ox: float  # origen en x (px)
    oy: float  # origen en y (px, con Y hacia arriba)
    units: str  # "cm" o "px"
    relative: bool  # ¿relativa al origen XY calibrado?

    def x(self, px):
        return (px - self.ox) * self.k

    def y(self, py_up):
        return (py_up - self.oy) * self.k


def _make_view(ctx: PlotContext, records) -> _View:
    cm = ctx.cm_per_px if ctx.cm_per_px else infer_cm_per_px(records)
    k, units = (float(cm), "cm") if cm else (1.0, "px")
    if ctx.xy_origin is not None and ctx.frame_size is not None:
        ox, oy_img = ctx.xy_origin
        return _View(k, float(ox), float(ctx.frame_size[1]) - float(oy_img), units, True)
    return _View(k, 0.0, 0.0, units, False)


def _extent_px(ctx: PlotContext, xs: np.ndarray, ys: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Cuadrícula (en px, Y hacia arriba): la del ROI si se conoce, y si no la de los datos con margen."""
    if ctx.roi_box is not None and ctx.frame_size is not None:
        x0, y0, w0, h0 = ctx.roi_box
        h = ctx.frame_size[1]
        return (float(x0), float(x0 + w0)), (float(h - (y0 + h0)), float(h - y0))
    pad_x = max(0.05 * float(np.ptp(xs)), 5.0)
    pad_y = max(0.05 * float(np.ptp(ys)), 5.0)
    return (float(xs.min() - pad_x), float(xs.max() + pad_x)), (float(ys.min() - pad_y), float(ys.max() + pad_y))


def _segments(records, view: _View):
    """Segmentos entre registros consecutivos detectados (un hueco corta la línea)."""
    segs, times = [], []
    prev = None
    for r in records:
        if not r.found:
            prev = None
            continue
        cur = (view.x(r.cx_abs), view.y(r.cy_abs_inv))
        if prev is not None:
            segs.append([prev, cur])
            times.append(r.time_s)
        prev = cur
    return segs, times


# Elementos de dibujo
def _draw_background(ax, ctx: PlotContext, view: _View) -> None:
    img = ctx.background_bgr
    if img is None:
        return
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # origin="upper" con extent (izq, der, abajo, arriba): la fila 0 queda arriba, igual que en la imagen
    ax.imshow(rgb, extent=(view.x(0), view.x(w), view.y(0), view.y(h)), origin="upper", zorder=0)


def _draw_zones(ax, ctx: PlotContext, view: _View) -> None:
    for z in ctx.zones:
        mask = np.asarray(z.mask)
        h = mask.shape[0]
        cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            pts = c.reshape(-1, 2).astype(float)
            xs = view.x(np.append(pts[:, 0], pts[0, 0]))
            ys = view.y(h - np.append(pts[:, 1], pts[0, 1]))  # Y de imagen -> Y hacia arriba
            ax.plot(xs, ys, color="white", lw=1.6, ls="--", zorder=4)
            ax.plot(xs, ys, color="black", lw=0.5, ls="--", zorder=5)
        M = cv2.moments((mask > 0).astype(np.uint8))
        if M["m00"] > 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            ax.text(view.x(cx), view.y(h - cy), str(z.name), color="white", fontsize=9, ha="center", va="center",
                    zorder=6, bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55))


def _axis_labels(ax, view: _View) -> None:
    rel = "relativo al origen, " if view.relative else ""
    ax.set_xlabel(f"x ({rel}{view.units})")
    ax.set_ylabel(f"y ({rel}{view.units})")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)


def _set_limits(ax, view: _View, x_range, y_range) -> None:
    ax.set_xlim(view.x(x_range[0]), view.x(x_range[1]))
    ax.set_ylim(view.y(y_range[0]), view.y(y_range[1]))


def _new_figure(size=(8.5, 7.0)):
    fig = Figure(figsize=size)
    FigureCanvasAgg(fig)
    return fig, fig.add_subplot(111)


# Figuras (devuelven el Figure para poder inspeccionarlo; las funciones públicas lo guardan)
def _render_heatmap(records, dt, ctx: PlotContext, bin_cm: float, sigma_cm: float, exp_name: str = ""):
    xs, ys = detected_points(records)
    if len(xs) < 2:
        return None
    view = _make_view(ctx, records)
    x_range, y_range = _extent_px(ctx, xs, ys)

    if view.units == "cm":
        bin_px, sigma_px = bin_cm / view.k, sigma_cm / view.k
    else:
        bin_px = max(x_range[1] - x_range[0], y_range[1] - y_range[0]) / 60.0
        sigma_px = 1.5 * bin_px
    m = occupancy_map(records, dt, x_range, y_range, bin_px, sigma_px)
    if m.seconds.max() <= 0:
        return None

    fig, ax = _new_figure()
    _draw_background(ax, ctx, view)
    # Transparencia que crece con el tiempo: las regiones casi vacías se funden con el fondo sin bordes escalonados
    alpha = np.clip(m.seconds / m.seconds.max() / 0.12, 0.0, 1.0) * 0.88
    x0, x1, y0, y1 = m.extent
    im = ax.imshow(m.seconds, extent=(view.x(x0), view.x(x1), view.y(y0), view.y(y1)), origin="lower", cmap="inferno",
                   vmin=0.0, alpha=alpha, interpolation="bilinear", zorder=2)
    fig.colorbar(im, ax=ax, label="Tiempo en la región (s)", fraction=0.046, pad=0.03)

    segs, _ = _segments(records, view)
    if segs:
        ax.add_collection(LineCollection(segs, colors="white", linewidths=0.6, alpha=0.45, zorder=3))
    _draw_zones(ax, ctx, view)
    _set_limits(ax, view, x_range, y_range)
    _axis_labels(ax, view)
    ax.set_title(f"Mapa de calor de ocupación{' - ' + exp_name if exp_name else ''}")
    celda = f"celda {bin_cm:g} cm, suavizado {sigma_cm:g} cm" if view.units == "cm" else "celda y suavizado en px"
    nota = f"{m.total_time_s:.1f} s con el sujeto detectado | {celda}"
    if m.n_outside:
        nota += f" | {m.n_outside} puntos fuera de la cuadrícula"
    fig.text(0.5, 0.012, nota, ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def _render_trajectory(records, ctx: PlotContext, exp_name: str = ""):
    xs, ys = detected_points(records)
    if len(xs) < 2:
        return None
    view = _make_view(ctx, records)
    x_range, y_range = _extent_px(ctx, xs, ys)
    segs, times = _segments(records, view)

    fig, ax = _new_figure()
    _draw_background(ax, ctx, view)
    if segs:
        lc = LineCollection(segs, cmap="viridis", linewidths=1.4, zorder=3)
        lc.set_array(np.asarray(times, dtype=float))
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label="Tiempo (s)", fraction=0.046, pad=0.03)
    first = next(r for r in records if r.found)
    last = next(r for r in reversed(list(records)) if r.found)
    ax.plot(view.x(first.cx_abs), view.y(first.cy_abs_inv), marker="^", color="lime", mec="black", ms=10, ls="", zorder=6, label="Inicio")
    ax.plot(view.x(last.cx_abs), view.y(last.cy_abs_inv), marker="s", color="red", mec="black", ms=9, ls="", zorder=6, label="Final")
    _draw_zones(ax, ctx, view)
    _set_limits(ax, view, x_range, y_range)
    _axis_labels(ax, view)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Trayectoria{' - ' + exp_name if exp_name else ''}")
    lost = sum(1 for r in records if not r.found)
    if lost:
        fig.text(0.5, 0.012, f"{lost} frames con el sujeto perdido (la línea se interrumpe en esos huecos)",
                 ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def _save(fig, output_folder: str, filename: str) -> str:
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, filename)
    fig.savefig(path, dpi=180)
    return path


def create_heatmap_plot(result, output_folder, exp_name, context: Optional[PlotContext] = None,
                        bin_cm: float = 1.0, sigma_cm: float = 1.5) -> Optional[str]:
    """Guarda ``eda_mapa_calor_<exp>.png``. Devuelve la ruta, o None si no hay datos suficientes."""
    fig = _render_heatmap(result.records, result.timing.dt, context or PlotContext(), bin_cm, sigma_cm, exp_name)
    return None if fig is None else _save(fig, output_folder, f"eda_mapa_calor_{exp_name}.png")


def create_trajectory_plot(result, output_folder, exp_name, context: Optional[PlotContext] = None) -> Optional[str]:
    """Guarda ``eda_trayectoria_<exp>.png``. Devuelve la ruta, o None si no hay datos suficientes."""
    fig = _render_trajectory(result.records, context or PlotContext(), exp_name)
    return None if fig is None else _save(fig, output_folder, f"eda_trayectoria_{exp_name}.png")