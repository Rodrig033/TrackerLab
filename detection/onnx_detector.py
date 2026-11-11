"""Detector de ratones (modelo ONNX exportado por el cuaderno de Colab) para usar dentro de TrackerLab.

    detector = OnnxDetector("/ruta/rfdetr-small.onnx")
    det = detector.best(frame_bgr, roi_box)      # None si no ve al ratón
    if det: x, y, w, h = det.box                 # en píxeles del FRAME completo

El modelo se usa igual que se entrenó: sobre el recorte del ROI, con el margen que el cuaderno guardó en las notas del ONNX.
Tarda del orden de 200 a 300 ms por imagen en CPU: sirve para proponer al sujeto, verificar o recuperar, no para cada frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2

from detection.model import crop_rect, make_session, read_notes, resolve_margin
from learning.onnx_decode import detect

Box = Tuple[float, float, float, float]  # x, y, ancho, alto
DEFAULT_THRESHOLD = 0.5


class DetectorError(RuntimeError):
    """El modelo no existe, no se pudo abrir o no sirve. El mensaje está pensado para mostrarse al usuario."""


@dataclass(frozen=True)
class Detection:
    box: Box  # en píxeles del frame completo
    score: float

    @property
    def center(self) -> Tuple[float, float]:
        return self.box[0] + self.box[2] / 2.0, self.box[1] + self.box[3] / 2.0


class OnnxDetector:
    def __init__(self, model_path: str, *, threshold: Optional[float] = None, margin: Optional[int] = None, threads: Optional[int] = None,
                 session_factory: Callable[[str, Optional[int]], Any] = make_session) -> None:
        if not model_path or not os.path.isfile(model_path):
            raise DetectorError(f"No existe el modelo {model_path!r}.")
        if threshold is not None and not (0.0 < threshold < 1.0):
            raise ValueError("El umbral de confianza debe estar entre 0 y 1 (sin incluirlos).")
        if margin is not None and margin < 0:
            raise ValueError("El margen del ROI no puede ser negativo.")
        try:
            self.session = session_factory(model_path, threads)
        except Exception as exc:  # onnxruntime lanza tipos propios; cualquier fallo al abrir se explica igual
            raise DetectorError(f"No se pudo abrir el modelo {os.path.basename(model_path)}: {exc}") from exc
        forma = list(self.session.get_inputs()[0].shape)
        if len(forma) != 4 or not all(isinstance(v, int) and v > 0 for v in forma[2:4]):
            raise DetectorError(f"El modelo {os.path.basename(model_path)} no tiene una entrada de tamaño fijo (1, 3, alto, ancho); la suya es {forma}.")
        self.model_path = model_path
        self.notes: Dict[str, Any] = read_notes(self.session)
        self.crops_roi: bool = self.notes.get("recorte_roi") is not False  # False solo si las notas dicen que se entrenó con frames completos
        self.margin, self.margin_source = resolve_margin(self.notes, margin, not self.crops_roi)
        sugerido = self.notes.get("umbral_sugerido")
        if threshold is not None:
            self.threshold = float(threshold)
        elif isinstance(sugerido, (int, float)) and not isinstance(sugerido, bool) and 0.0 < sugerido < 1.0:
            self.threshold = float(sugerido)
        else:
            self.threshold = DEFAULT_THRESHOLD

    def detect(self, frame_bgr: Any, roi_box: Optional[Sequence[int]] = None, threshold: Optional[float] = None) -> List[Detection]:
        """Detecciones de mayor a menor confianza, en píxeles del frame completo.

        Con ``roi_box`` (x, y, ancho, alto) el modelo ve el recorte del ROI (con su margen) y solo se devuelven las detecciones cuyo
        centro cae dentro del ROI. Lanza ValueError si el ROI queda fuera del frame.
        """
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = crop_rect(roi_box, self.margin, w, h) if (roi_box is not None and self.crops_roi) else (0, 0, w, h)
        crudo = detect(self.session, cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), self.threshold if threshold is None else threshold)
        out: List[Detection] = []
        for (bx1, by1, bx2, by2), sc in zip(crudo.xyxy, crudo.scores):
            if bx2 - bx1 <= 0 or by2 - by1 <= 0:
                continue
            d = Detection((float(bx1) + x0, float(by1) + y0, float(bx2 - bx1), float(by2 - by1)), float(sc))
            if roi_box is not None:
                rx, ry, rw, rh = (int(v) for v in roi_box)
                cx, cy = d.center
                if not (rx <= cx < rx + rw and ry <= cy < ry + rh):
                    continue
            out.append(d)
        return out

    def best(self, frame_bgr: Any, roi_box: Optional[Sequence[int]] = None, threshold: Optional[float] = None) -> Optional[Detection]:
        """La detección de mayor confianza, o None si no hay ninguna."""
        dets = self.detect(frame_bgr, roi_box, threshold)
        return dets[0] if dets else None

    def describe(self) -> Dict[str, Any]:
        """Datos para dejar constancia de con qué modelo se trabajó."""
        return {"modelo": os.path.basename(self.model_path), "umbral": self.threshold, "margen_roi_px": self.margin,
                "origen_del_margen": self.margin_source, "recorte_roi": self.crops_roi, "notas": self.notes}