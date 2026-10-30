from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np

__all__ = ["OccupancyMap", "detected_points", "occupancy_map", "average_maps"]


def detected_points(
    records: Iterable,
    x_field: str = "cx_abs",
    y_field: str = "cy_abs_inv",
) -> Tuple[np.ndarray, np.ndarray]:
    """Coordenadas (x, y) de los registros con el sujeto detectado (sin None ni NaN)."""
    xs, ys = [], []
    for r in records:
        x, y = getattr(r, x_field, None), getattr(r, y_field, None)
        if x is None or y is None:
            continue
        x, y = float(x), float(y)
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


@dataclass(frozen=True, eq=False)
class OccupancyMap:
    """Resultado: tiempo (s) por celda sobre una cuadrícula regular."""

    seconds: np.ndarray  # forma (ny, nx); fila 0 = y más baja (Y hacia arriba)
    x_edges: np.ndarray  # (nx + 1,)
    y_edges: np.ndarray  # (ny + 1,)
    total_time_s: float  # tiempo total con sujeto dentro de la cuadrícula (antes de suavizar)
    n_points: int  # registros con sujeto detectado
    n_outside: int  # de ellos, los que cayeron fuera de la cuadrícula (descartados)
    bin_size: float
    sigma: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.seconds.shape

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) de la cuadrícula."""
        return (float(self.x_edges[0]), float(self.x_edges[-1]), float(self.y_edges[0]), float(self.y_edges[-1]))

    def percent(self) -> np.ndarray:
        """Porcentaje del tiempo en cada celda (suma 100). Comparable entre animales."""
        total = float(self.seconds.sum())
        if total <= 0:
            return np.zeros_like(self.seconds)
        return self.seconds / total * 100.0


def occupancy_map(
    records: Iterable,
    dt: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    bin_size: float,
    sigma: float = 0.0,
    x_field: str = "cx_abs",
    y_field: str = "cy_abs_inv",
) -> OccupancyMap:
    """Mapa de ocupación de una corrida. """
    if not dt > 0:
        raise ValueError(f"dt debe ser > 0, recibido {dt}")
    if not bin_size > 0:
        raise ValueError(f"bin_size debe ser > 0, recibido {bin_size}")
    if sigma < 0:
        raise ValueError("sigma no puede ser negativo")
    (x0, x1), (y0, y1) = x_range, y_range
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"Rangos inválidos: x={x_range}, y={y_range}")

    xs, ys = detected_points(records, x_field, y_field)
    nx = max(1, int(math.ceil((x1 - x0) / bin_size - 1e-9)))
    ny = max(1, int(math.ceil((y1 - y0) / bin_size - 1e-9)))
    x_edges = x0 + bin_size * np.arange(nx + 1)
    y_edges = y0 + bin_size * np.arange(ny + 1)

    counts, _, _ = np.histogram2d(ys, xs, bins=[y_edges, x_edges])
    n_inside = int(counts.sum())
    seconds = counts * float(dt)

    if sigma > 0:
        s_bins = sigma / bin_size
        k = int(2 * math.ceil(3 * s_bins) + 1)
        k = min(k, min(nx, ny) - (1 - min(nx, ny) % 2))  # impar y no mayor que la cuadrícula
        if k >= 3:
            # BORDER_REFLECT (espejo en el borde) conserva exactamente la suma
            seconds = cv2.GaussianBlur(seconds, (k, k), s_bins, borderType=cv2.BORDER_REFLECT)

    return OccupancyMap(
        seconds=seconds,
        x_edges=x_edges,
        y_edges=y_edges,
        total_time_s=n_inside * float(dt),
        n_points=int(len(xs)),
        n_outside=int(len(xs)) - n_inside,
        bin_size=float(bin_size),
        sigma=float(sigma),
    )


def average_maps(maps: Sequence[OccupancyMap], normalize: bool = True) -> np.ndarray:
    """Promedio de varios mapas (p. ej. los animales de un grupo)."""
    if not maps:
        raise ValueError("No hay mapas que promediar.")
    ref = maps[0]
    for m in maps[1:]:
        if m.shape != ref.shape or not (np.allclose(m.x_edges, ref.x_edges) and np.allclose(m.y_edges, ref.y_edges)):
            raise ValueError("Los mapas no comparten la misma cuadrícula: usa el mismo ROI y el mismo bin_size.")
    stack = [m.percent() if normalize else m.seconds for m in maps]
    return np.mean(stack, axis=0)