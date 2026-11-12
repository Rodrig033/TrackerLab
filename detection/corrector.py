"""Tecolote como corrector del seguimiento: cada cierto tiempo mide dónde está el ratón y corrige al seguimiento.

El seguimiento clásico sigue corriendo en cada frame (es rápido); Tecolote, que tarda ~200-300 ms por imagen, solo mide cada
``correct_every_s`` segundos y en el primer frame en que el seguimiento se pierde. Reglas:

  * Solo cuenta una medición con confianza alta (``min_score``, por omisión 0.8). Si Tecolote no ve al ratón o duda, manda el
    seguimiento (bajo el tubo, por ejemplo, Tecolote calla o da una caja parcial con confianza baja).
  * Si el seguimiento está CERCA de lo que mide Tecolote (a menos de ``max_offset_ratio`` veces el lado mayor de su caja), está
    sobre el ratón pero SESGADO (la cola, un reflejo pegado): se corrige ese frame y se guarda el desvío, que se aplica a los
    frames siguientes hasta la próxima medición (o hasta que caduque).
  * Si está LEJOS, o perdido: RESCATE. Se reubica al seguimiento sobre el ratón.

La decisión está en ``classify`` (pura); el bucle que la aplica está en pipeline.run_tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

Point = Tuple[float, float]
BoxXYWH = Tuple[float, float, float, float]

SESGO, RESCATE = "sesgo", "rescate"
DEFAULT_MIN_SCORE = 0.8


@dataclass(frozen=True)
class Measure:
    """Lo que midió Tecolote, en píxeles del FRAME completo."""
    center: Point
    bbox: BoxXYWH
    score: float


def classify(tracker_pos: Optional[Point], meas: Measure, max_offset_ratio: float) -> Tuple[str, Optional[Point]]:
    """(SESGO, desvío) si el seguimiento está sobre el ratón pero corrido; (RESCATE, None) si está lejos o perdido.

    El desvío es lo que hay que SUMAR a la posición del seguimiento para llegar a la de Tecolote.
    """
    if tracker_pos is None:
        return RESCATE, None
    dx, dy = meas.center[0] - tracker_pos[0], meas.center[1] - tracker_pos[1]
    if math.hypot(dx, dy) <= max_offset_ratio * max(meas.bbox[2], meas.bbox[3]):
        return SESGO, (dx, dy)
    return RESCATE, None


class TecoloteCorrector:
    """Envuelve a OnnxDetector: una medición por llamada, o None si no ve al ratón con confianza suficiente."""

    def __init__(self, detector: Any, min_score: float = DEFAULT_MIN_SCORE) -> None:
        if not (0.0 < min_score <= 1.0):
            raise ValueError("min_score debe estar entre 0 (sin incluirlo) y 1.")
        self.detector = detector
        self.min_score = float(min_score)

    def measure(self, frame_bgr: Any, roi_box: Any) -> Optional[Measure]:
        d = self.detector.best(frame_bgr, roi_box)
        if d is None or d.score < self.min_score:
            return None
        return Measure(d.center, d.box, d.score)

    def describe(self) -> Dict[str, Any]:
        base = self.detector.describe() if hasattr(self.detector, "describe") else {}
        return {"modelo_ruta": getattr(self.detector, "model_path", None), "confianza_minima": self.min_score, **{k: base[k] for k in ("modelo", "umbral", "margen_roi_px") if k in base}}


def make_corrector(model_path: str, min_score: float = DEFAULT_MIN_SCORE, **detector_kw: Any) -> TecoloteCorrector:
    """Tecolote listo para corregir, a partir de la ruta del modelo. Lanza DetectorError si el modelo no sirve."""
    from detection.onnx_detector import OnnxDetector
    return TecoloteCorrector(OnnxDetector(model_path, **detector_kw), min_score)