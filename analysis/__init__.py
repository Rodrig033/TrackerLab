from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = ["ZoneRecord", "FrameRecord"]


@dataclass(frozen=True)
class ZoneRecord:
    name: str
    inside: int  # 1 = el punto está dentro de la zona, 0 = fuera

    dist_center_px: Optional[float] = None
    dist_center_cm: Optional[float] = None  # None si no hay escala
    dist_perim_px: Optional[float] = None
    dist_perim_cm: Optional[float] = None  # None si no hay escala

    # Acumulados hasta este frame (inclusive)
    time_acum_s: float = 0.0
    frames_acum: int = 0
    entries_acum: int = 0

    # Primera entrada a la zona; None mientras no haya ocurrido
    lat_first_s: Optional[float] = None
    frame_first: Optional[int] = None


@dataclass(frozen=True)
class FrameRecord:
    """Resultados de UN frame del tracking.

    Un frame con el sujeto perdido (``found == False``) también genera
    registro, igual que en el script original: se conserva ``frame``,
    ``time_s`` y ``dist_acum_*``; todo lo demás queda en ``None`` y
    ``zones`` queda vacío (el exportador escribe celdas en blanco).
    """

    # --- Identificación del frame ---
    frame: int
    time_s: float  # relativo al frame de inicio del segmento exportado

    # --- Distancia acumulada: se escribe SIEMPRE, incluso con sujeto perdido ---
    dist_acum_px: float
    dist_acum_cm: Optional[float] = None  # None si no hay escala

    # --- Posición absoluta en la imagen completa ---
    cx_abs: Optional[float] = None
    # Y invertida respecto a la imagen: (alto_frame - cy). Es la columna
    # ``cy_abs_inv`` del CSV (Y positiva hacia arriba).
    cy_abs_inv: Optional[float] = None

    # --- Posición relativa al origen XY (None si no se calibró origen/escala) ---
    x_rel_px: Optional[float] = None
    y_rel_px: Optional[float] = None
    x_rel_cm: Optional[float] = None
    y_rel_cm: Optional[float] = None

    # --- Cinemática instantánea ---
    dist_px: Optional[float] = None  # desplazamiento respecto al frame previo
    dist_cm: Optional[float] = None
    vel_px_s: Optional[float] = None
    vel_cm_s: Optional[float] = None
    acel_px_s2: Optional[float] = None
    acel_cm_s2: Optional[float] = None
    ang_deg: Optional[float] = None  # cambio de dirección entre desplazamientos
    vel_angular_deg_s: Optional[float] = None

    # --- Métricas por zona (vacío si el sujeto no fue encontrado) ---
    zones: Tuple[ZoneRecord, ...] = ()

    @property
    def found(self) -> bool:
        """True si el sujeto fue detectado en este frame."""
        return self.cx_abs is not None