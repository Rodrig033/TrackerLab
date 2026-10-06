from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = ["ZoneRecord", "FrameRecord"]

@dataclass(frozen=True)

class ZoneRecord:
    name: str
    inside: int # 1 = el punto está en la zona, 0 = fuera

    dist_center_px: Optional[float] = None
    dist_center_cm: Optional[float] = None # None si no hay escala
    dist_perim_px: Optional[float] = None
    dist_center_cm: Optional[float] = None # None si no hay escala

    # Acumulados hasta este frame (inclusive)
    time_acum_s: float = 0.0
    frames_acum: int = 0
    entries_acum: int = 0

    # Primera entrada a la zona; None mientras no haya ocurrido
    last_first_s: Optional[float] = None
    first_frame: Optional[int] = None


@dataclass(frozen=True)

class FrameRecord:

    #   Identificación del frame
    frame_int: int
    time_s: float

    # Distancia acumulada: se escribe SIEMPRE, incluso con un sujeto perdido
    dist_acum_px: float
    dist_acum_cm: Optional[float] = None # None si no hay escala

    # Posición absoluta en la imagen completa
    cx_abs: Optional[float] = None
    """Y invertida respecto a la imagen: (alto_frame - cy). Es la columna
        cy_abs_inv del CSV (Y positiva hacia arriba).
    """
    cy_abs_inv: Optional[float] = None

    # Posición relativa al origen XY (None si no se calibro el origen/escala)
    x_rel_px: Optional[float] = None
    y_rel_px: Optional[float] = None
    x_rel_cm: Optional[float] = None
    y_rel_cm: Optional[float] = None

    # Cinemática instantanea
    dist_px: Optional[float] = None
    dist_cm: Optional[float] = None
    vel_px_s: Optional[float] = None
    vel_cm_s: Optional[float] = None
    acel_px_s2: Optional[float] = None
    acel_cm_s2: Optional[float] = None
    ang_deg: Optional[float] = None
    vel_angular_deg_s: Optional[float] = None # Cambion de dirección entre desplazamientos



    # Métricas por zona de vacío (vacío si el sujeto no fue encontrado)
    zones: Tuple["ZoneRecord", ...] = ()

    @property
    def found(self) -> bool:
        """True si el sujeto fue detectado en este frame"""
        return self.cx_abs is not None