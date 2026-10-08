from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from analysis import ZoneRecord
from analysis.kinematic_metrics import euclidean

__all__ = ["Point", "ZoneGeometry", "ZoneSummary", "ZoneTracker", "point_in_mask"]

Point = Tuple[float, float]


# Geometría de una zona

def point_in_mask(mask: np.ndarray, x: float, y: float) -> int:
    """1 si el punto cae en un píxel activo de la máscara, 0 si no (o si está fuera).

    Igual que el original: las coordenadas se redondean con ``round`` de
    Python (que lleva .5 al entero PAR más cercano: 2.5 -> 2).
    """
    h, w = mask.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return 0
    return 1 if mask[yi, xi] > 0 else 0


class ZoneGeometry:
    """Una zona: nombre, máscara del frame completo, centroide y contornos.

    Los contornos se calculan UNA vez al crear el objeto.
    """

    def __init__(self, name: str, mask: np.ndarray, centroid: Optional[Point] = None):
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError("La máscara de la zona debe ser 2D (alto x ancho).")
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        self.name = str(name)
        self.mask = mask
        self.centroid: Optional[Point] = None if centroid is None else (float(centroid[0]), float(centroid[1]))
        # Solo contornos EXTERNOS (como el original): los huecos no cuentan.
        self.contours = tuple(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])

    def contains(self, point: Point) -> int:
        """1 si el punto está dentro de la zona (según los píxeles), 0 si no."""
        return point_in_mask(self.mask, point[0], point[1])

    def perimeter_distance(self, point: Point) -> Optional[float]:
        if not self.contours:
            return None
        x, y = float(point[0]), float(point[1])
        min_abs_dist = None
        for c in self.contours:
            d = cv2.pointPolygonTest(c, (x, y), True)
            if d >= 0:
                return 0.0  # dentro de algún contorno: no hace falta seguir
            ad = abs(float(d))
            if min_abs_dist is None or ad < min_abs_dist:
                min_abs_dist = ad
        return min_abs_dist


# Resultado de resumen

@dataclass(frozen=True)
class ZoneSummary:
    """Fila de resumen de una zona (los ``<zona>_...`` del CSV de resumen).

    ``None`` = sin dato (en el CSV se escribe vacío).
    """

    name: str
    tiempo_total_s: float
    frames_total: int
    proporcion_tiempo: Optional[float]  # tiempo en zona / tiempo total válido
    entradas: int
    lat_primera_entrada_s: Optional[float]
    frame_primera_entrada: Optional[int]
    dist_media_centro_px: Optional[float]
    dist_media_centro_cm: Optional[float]
    dist_media_perimetro_px: Optional[float]
    dist_media_perimetro_cm: Optional[float]


class _Stats:
    """Acumuladores de una zona (equivale a un ``zone_stats[nombre]`` original)."""

    __slots__ = (
        "time_s", "frames_in_zone", "entries", "lat_first_s", "frame_first",
        "prev_inside", "sum_center_px", "sum_center_cm", "sum_perim_px",
        "sum_perim_cm", "n_dist_valid",
    )

    def __init__(self) -> None:
        self.time_s = 0.0
        self.frames_in_zone = 0
        self.entries = 0
        self.lat_first_s: Optional[float] = None
        self.frame_first: Optional[int] = None
        # 0 = fuera, 1 = dentro, None = desconocido (tras pérdida o reinit).
        # Empieza en 0: si el sujeto arranca dentro, cuenta como una entrada.
        self.prev_inside: Optional[int] = 0
        self.sum_center_px = 0.0
        self.sum_center_cm = 0.0
        self.sum_perim_px = 0.0
        self.sum_perim_cm = 0.0
        self.n_dist_valid = 0


# Calculador con estado

class ZoneTracker:
    """Calcula las métricas de todas las zonas frame a frame."""

    def __init__(
        self,
        zones: Sequence[ZoneGeometry],
        dt: float,
        cm_per_px: Optional[float] = None,
    ):
        """
        zones: zonas del experimento (puede ser una lista vacía).
        dt: segundos entre frames procesados (``TrackingTiming.dt``).
        cm_per_px: escala de calibración, o ``None`` si no hay.
        """
        if not dt > 0:
            raise ValueError(f"dt debe ser > 0, recibido {dt}")
        if cm_per_px is not None and not cm_per_px > 0:
            raise ValueError(f"cm_per_px debe ser > 0, recibido {cm_per_px}")
        names = [z.name for z in zones]
        if len(set(names)) != len(names):
            raise ValueError(f"Hay nombres de zona repetidos: {names}")

        self.zones: List[ZoneGeometry] = list(zones)
        self.dt = float(dt)
        self.cm_per_px = None if cm_per_px is None else float(cm_per_px)
        self._stats: Dict[str, _Stats] = {z.name: _Stats() for z in self.zones}
        self._valid_frames = 0

    def reset_history(self) -> None:
        """Marca el estado dentro/fuera como desconocido SIN borrar acumulados.

        Evita contar una entrada falsa si el sujeto reaparece dentro de una
        zona. Se llama solo dentro de ``update(None, ...)`` y, a mano, tras un
        reinit ('r').
        """
        for st in self._stats.values():
            st.prev_inside = None

    def update(self, point: Optional[Point], frame_idx: int, time_s: float) -> Tuple[ZoneRecord, ...]:
        """Procesa un frame.

        point: (x, y) absolutos, o ``None`` si el sujeto se perdió.
        frame_idx / time_s: se guardan como "primera entrada" si procede
                            (``time_s`` es el tiempo relativo al inicio).
        Devuelve un ``ZoneRecord`` por zona, o ``()`` si el sujeto se perdió.
        """
        if point is None:
            self.reset_history()
            return ()

        self._valid_frames += 1
        cm = self.cm_per_px
        dt = self.dt
        out: List[ZoneRecord] = []

        for z in self.zones:
            st = self._stats[z.name]
            inside = z.contains(point)

            d_center_px = d_center_cm = d_perim_px = d_perim_cm = None
            if z.centroid is not None:
                d_center_px = euclidean(point, z.centroid)
                if cm is not None:
                    d_center_cm = d_center_px * cm

            d_perim_px = z.perimeter_distance(point)
            if d_perim_px is not None and cm is not None:
                d_perim_cm = d_perim_px * cm

            if inside == 1:
                st.time_s += dt
                st.frames_in_zone += 1

            # Entrada = transición fuera -> dentro. Si prev_inside es None
            # (tras pérdida/reinit) no se cuenta una entrada falsa.
            if st.prev_inside == 0 and inside == 1:
                st.entries += 1
                if st.lat_first_s is None:
                    st.lat_first_s = time_s
                    st.frame_first = frame_idx
            st.prev_inside = inside

            if d_center_px is not None:
                st.sum_center_px += d_center_px
            if d_center_cm is not None:
                st.sum_center_cm += d_center_cm
            if d_perim_px is not None:
                st.sum_perim_px += d_perim_px
            if d_perim_cm is not None:
                st.sum_perim_cm += d_perim_cm
            # Igual que el original: cuenta SIEMPRE, aunque alguna distancia
            # sea None (zona sin centroide o sin contornos) -> su promedio
            # sale 0.0 en lugar de vacío. Ver ZoneTrackerTests.
            st.n_dist_valid += 1

            out.append(
                ZoneRecord(
                    name=z.name,
                    inside=inside,
                    dist_center_px=d_center_px,
                    dist_center_cm=d_center_cm,
                    dist_perim_px=d_perim_px,
                    dist_perim_cm=d_perim_cm,
                    time_acum_s=st.time_s,
                    frames_acum=st.frames_in_zone,
                    entries_acum=st.entries,
                    lat_first_s=st.lat_first_s,
                    frame_first=st.frame_first,
                )
            )
        return tuple(out)

    def summary(self) -> Tuple[ZoneSummary, ...]:
        """Resumen de todas las zonas hasta este momento (una entrada por zona)."""
        has_cm = self.cm_per_px is not None
        total_valid_s = self._valid_frames * self.dt
        res: List[ZoneSummary] = []
        for z in self.zones:
            st = self._stats[z.name]
            n = st.n_dist_valid
            res.append(
                ZoneSummary(
                    name=z.name,
                    tiempo_total_s=st.time_s,
                    frames_total=st.frames_in_zone,
                    proporcion_tiempo=st.time_s / total_valid_s if total_valid_s > 0 else None,
                    entradas=st.entries,
                    lat_primera_entrada_s=st.lat_first_s,
                    frame_primera_entrada=st.frame_first,
                    dist_media_centro_px=st.sum_center_px / n if n > 0 else None,
                    dist_media_centro_cm=st.sum_center_cm / n if has_cm and n > 0 else None,
                    dist_media_perimetro_px=st.sum_perim_px / n if n > 0 else None,
                    dist_media_perimetro_cm=st.sum_perim_cm / n if has_cm and n > 0 else None,
                )
            )
        return tuple(res)