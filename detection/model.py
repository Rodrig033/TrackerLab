"""Núcleo para usar el detector ONNX: abrir el modelo, leer sus notas y decidir el recorte.

Es la FUENTE ÚNICA de estas reglas. La usan la aplicación (detection/onnx_detector.py) y las herramientas
(tools/eval_detector.py, tools/scan_video.py, tools/build_training_set.py), para que el modelo se use exactamente como se
entrenó y se evaluó: mismo recorte al ROI y mismo margen.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence, Tuple

Rect = Tuple[int, int, int, int]  # x0, y0, x1, y1 (x1, y1 exclusivos)


def make_session(ruta: str, hilos: Optional[int] = None) -> Any:
    """Sesión de onnxruntime en CPU. ``hilos`` limita los hilos de cálculo (None = los que decida onnxruntime)."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("No está instalado onnxruntime. Instálalo con: pip install onnxruntime") from exc
    opciones = ort.SessionOptions()
    if hilos:
        opciones.intra_op_num_threads = int(hilos)
    return ort.InferenceSession(ruta, opciones, providers=["CPUExecutionProvider"])


def read_notes(sesion: Any) -> Dict[str, Any]:
    """Notas que el cuaderno guardó dentro del ONNX (modelo, resolución, margen del ROI...). {} si no hay o no se pueden leer."""
    try:
        meta = sesion.get_modelmeta().custom_metadata_map
        crudo = meta.get("rfdetr_notes") if meta else None
        datos = json.loads(crudo) if crudo else {}
        return datos if isinstance(datos, dict) else {}
    except Exception:  # un ONNX sin metadatos, o con metadatos raros, no impide usar el modelo
        return {}


def resolve_margin(notas: Dict[str, Any], margen_roi: Optional[int], sin_recorte: bool) -> Tuple[int, str]:
    """Margen del ROI con que se usa el modelo, y de dónde sale. Lo que se pase manda; si no, el de las notas del modelo; si no, 0.

    Con frame completo no hay margen. Es la ÚNICA regla: la usan la aplicación, eval_detector.py y scan_video.py.
    """
    if sin_recorte:
        return 0, "no aplica (frame completo)"
    if margen_roi is not None:
        return margen_roi, "indicado con --margen-roi"
    if isinstance(notas.get("margen_roi_px"), int):
        return notas["margen_roi_px"], "leído de las notas del modelo"
    return 0, "por omisión (el modelo no trae notas)"


def crop_rect(roi_box: Sequence[int], margin: int, frame_w: int, frame_h: int) -> Rect:
    """Recorte alrededor del ROI con ``margin`` px por lado, limitado al frame."""
    x, y, w, h = (int(v) for v in roi_box)
    m = max(0, int(margin))
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(frame_w, x + w + m), min(frame_h, y + h + m)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError(f"El ROI {tuple(roi_box)} queda fuera del frame de {frame_w}x{frame_h}.")
    return x0, y0, x1, y1