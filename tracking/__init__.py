from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "Point",
    "Box",
    "MASK_FOREGROUND",
    "MASK_SILHOUETTE",
    "MODE_BACKGROUND",
    "MODE_SILHOUETTE",
    "MODE_COLOR",
    "Detection",
    "BaseTracker",
    "tracking_mode_label",
]

Point = Tuple[float, float]  # (cx, cy)
Box = Tuple[int, int, int, int]  # (x, y, w, h)

# Cómo debe dibujarse ``Detection.mask`` (igual que en el script original):
#   foreground -> se mezcla sobre el frame con transparencia (modos 1 y 3)
#   silhouette -> se dibuja solo el contorno más grande (modo 2)
MASK_FOREGROUND = "foreground"
MASK_SILHOUETTE = "silhouette"

# Los identificadores son texto ("1", "2", "3") para seguir siendo
# compatibles con los JSON de experimentos ya guardados.
MODE_BACKGROUND = "1"
MODE_SILHOUETTE = "2"
MODE_COLOR = "3"

_MODE_LABELS = {
    MODE_BACKGROUND: "diferencia_fondo",
    MODE_SILHOUETTE: "silueta_ventana",
    MODE_COLOR: "marcador_color",
}


def tracking_mode_label(mode: Any) -> str:
    """Nombre legible del modo; se usa en nombres de archivo y mensajes."""
    mode = str(mode)
    return _MODE_LABELS.get(mode, f"modo_{mode}")


@dataclass(frozen=True, eq=False)  # eq=False: comparar arrays de numpy es ambiguo
class Detection:
    """
    Si el sujeto fue encontrado, ``center`` y ``bbox`` tienen valor. Si no,
    ambos son ``None``; aun así ``mask`` puede traer la máscara para dibujarla
    """

    center: Optional[Point] = None  # (cx, cy) en coordenadas del recorte
    bbox: Optional[Box] = None  # (x, y, w, h) en coordenadas del recorte
    mask: Optional[np.ndarray] = None  # uint8 0/255, mismo tamaño que el recorte
    mask_kind: str = MASK_FOREGROUND

    def __post_init__(self) -> None:
        if (self.center is None) != (self.bbox is None):
            raise ValueError("center y bbox deben venir juntos o ambos ser None")

    @property
    def found(self) -> bool:
        return self.center is not None


class BaseTracker(ABC):
    #: Identificador del modo ("1", "2" o "3"); lo fija cada subclase.
    mode_id: str = ""

    def __init__(self, include_mask: np.ndarray, params: Optional[Dict[str, Any]] = None):
        self.include_mask = include_mask
        self.params: Dict[str, Any] = dict(params) if params else {}
        self.last_center: Optional[Point] = None
        self.last_bbox: Optional[Box] = None

    def reset(self, center: Point, bbox: Box) -> None:
        """Fija la posición de referencia: inicio del tracking o reinit con 'r'."""
        self.last_center = (float(center[0]), float(center[1]))
        self.last_bbox = tuple(int(round(v)) for v in bbox)  # type: ignore[assignment]

    def update(self, roi_bgr: np.ndarray) -> Detection:
        """Procesa un frame y actualiza la memoria del último punto válido."""
        if self.last_center is None or self.last_bbox is None:
            raise RuntimeError("Llama a reset(center, bbox) antes de update().")
        det = self._detect(roi_bgr)
        if det.found:
            self.last_center = det.center
            self.last_bbox = det.bbox
        return det

    @abstractmethod
    def _detect(self, roi_bgr: np.ndarray) -> Detection:
        """Detecta al sujeto en el recorte; usa self.last_center / last_bbox."""

    def preview(
        self,
        roi_bgr: np.ndarray,
        params: Dict[str, Any],
        ref_center: Point,
    ) -> Detection:

        raise NotImplementedError(f"El modo {self.mode_id!r} no ofrece preview().")