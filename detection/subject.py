"""Elección del sujeto: el detector propone la caja del ratón en el frame de muestra y la persona la confirma.

Sustituye a «dibuja la caja del sujeto» cuando hay un modelo configurado, y NUNCA bloquea: si no hay modelo, no abre, no ve al
ratón o la persona prefiere marcarlo ella, se usa la selección manual de siempre.

    bbox, info = choose_subject(sample_frame, roi_box, model_path=DETECTOR_MODEL, manual_select=lambda: select_object_bbox(sample_roi, "..."))

``bbox`` es (x, y, ancho, alto) en coordenadas del RECORTE del ROI, igual que la selección manual; None si se canceló.
``info`` deja constancia de cómo se eligió (se puede guardar con las entradas de la corrida).
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from detection.onnx_detector import Detection, OnnxDetector

ACEPTAR, MANUAL, CANCELAR = "aceptar", "manual", "cancelar"
BANDA = 34  # alto, en píxeles, de la franja de instrucciones que se añade ARRIBA de la imagen (no tapa nada)
CONFIANZA_ALTA = 0.8
IntBox = Tuple[int, int, int, int]


def box_in_roi(det: Detection, roi_box: Sequence[int]) -> Optional[IntBox]:
    """Caja del detector en coordenadas del recorte del ROI, en enteros y limitada al recorte. None si queda sin tamaño."""
    rx, ry, rw, rh = (int(v) for v in roi_box)
    x, y, w, h = det.box
    x0, y0 = max(0, int(math.floor(x - rx))), max(0, int(math.floor(y - ry)))
    x1, y1 = min(rw, int(math.ceil(x + w - rx))), min(rh, int(math.ceil(y + h - ry)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1 - x0, y1 - y0


def draw_proposal(sample_roi: np.ndarray, bbox: IntBox, score: float) -> np.ndarray:
    """Imagen para confirmar: el recorte con la caja propuesta y, ARRIBA, una franja con las instrucciones."""
    h, w = sample_roi.shape[:2]
    img = np.zeros((h + BANDA, w, 3), np.uint8)
    img[BANDA:] = sample_roi
    x, y, bw, bh = bbox
    color = (0, 220, 0) if score >= CONFIANZA_ALTA else (0, 220, 255)  # verde, o amarillo si la confianza es baja
    cv2.rectangle(img, (x, y + BANDA), (x + bw, y + bh + BANDA), color, 2)
    texto = f"Confianza {score:.2f}   ENTER/ESPACIO = aceptar   M = marcar a mano   ESC = cancelar"
    escala = min(0.6, max(0.35, w / 1300.0))
    cv2.putText(img, texto, (8, BANDA - 11), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def key_to_choice(key: int) -> Optional[str]:
    """Decisión que corresponde a una tecla (None si la tecla no significa nada)."""
    key &= 0xFF
    if key in (13, 10, 32, ord("a"), ord("A")):
        return ACEPTAR
    if key in (ord("m"), ord("M")):
        return MANUAL
    if key in (27, ord("q"), ord("Q")):
        return CANCELAR
    return None


def confirm_subject(sample_roi: np.ndarray, bbox: IntBox, score: float, title: str = "SUJETO PROPUESTO POR EL DETECTOR") -> str:
    """Muestra la propuesta y espera la decisión. Cerrar la ventana equivale a «marcar a mano»."""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1200, 800)
    cv2.imshow(title, draw_proposal(sample_roi, bbox, score))
    try:
        while True:
            key = cv2.waitKey(50)
            if key != -1:
                decision = key_to_choice(key)
                if decision:
                    return decision
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return MANUAL
    finally:
        cv2.destroyWindow(title)


def assistance_record(model_path: Optional[str], info: Dict[str, Any]) -> Dict[str, Any]:
    """Constancia de la asistencia del detector en una corrida: qué hizo (``modo``) y cómo se eligió al sujeto.

    Va dentro del TrackingJob y de ahí a inputs.json, a result.settings y al resumen. Es independiente del modo de seguimiento.
    """
    return {"modo": "proponer_sujeto" if model_path else "ninguna", "sujeto": dict(info)}


def describe_assistance(record: Optional[Dict[str, Any]], mode: str = "") -> str:
    """Una línea para la terminal: el modo de seguimiento y, aparte, la asistencia del detector."""
    pre = f"Modo: {mode} · " if mode else ""
    if not record or record.get("modo") in (None, "ninguna"):
        return pre + "asistencia del detector: ninguna (sujeto marcado a mano)"
    s = record.get("sujeto") or {}
    conf = f", confianza {s['confianza']:.2f}" if isinstance(s.get("confianza"), (int, float)) else ""
    quien = {"detector": "sujeto propuesto por el detector y aceptado", "manual": "sujeto marcado a mano", "cancelado": "cancelado"}.get(s.get("metodo"), str(s.get("metodo")))
    motivo = f" ({s['motivo']})" if s.get("metodo") != "detector" and s.get("motivo") else ""
    return pre + f"asistencia del detector: proponer sujeto · {quien}{motivo} · modelo {s.get('modelo') or '?'}{conf}"


def choose_subject(
    sample_frame: np.ndarray,
    roi_box: Sequence[int],
    *,
    model_path: Optional[str],
    manual_select: Callable[[], Optional[IntBox]],
    confirm: Callable[[np.ndarray, IntBox, float], str] = confirm_subject,
    detector_factory: Callable[[str], Any] = OnnxDetector,
    out: Callable[[str], None] = print,
) -> Tuple[Optional[IntBox], Dict[str, Any]]:
    """Devuelve (caja del sujeto en coordenadas del recorte del ROI, constancia de cómo se eligió)."""
    info: Dict[str, Any] = {"metodo": "manual", "motivo": "", "modelo": "", "confianza": None}

    def a_mano(motivo: str) -> Tuple[Optional[IntBox], Dict[str, Any]]:
        info.update(metodo="manual", motivo=motivo)
        return manual_select(), info

    if not model_path:
        return a_mano("sin modelo configurado")
    info["modelo"] = os.path.basename(model_path)
    try:
        detector = detector_factory(model_path)
        det = detector.best(sample_frame, roi_box)
        info.update(umbral=detector.threshold, margen_roi_px=detector.margin)
        notas = getattr(detector, "notes", {}) or {}
        info["notas_modelo"] = {k: notas[k] for k in ("corrida", "modelo", "version_rfdetr", "resolucion") if k in notas}
    except Exception as exc:  # frontera con la interfaz: pase lo que pase con el modelo, la persona puede seguir a mano
        out(f"Detector no disponible ({exc}). Selecciona el sujeto a mano.")
        return a_mano(f"detector no disponible: {exc}")
    bbox = box_in_roi(det, roi_box) if det is not None else None
    if det is None or bbox is None:
        out(f"El detector no encontró al ratón en el frame de muestra (confianza mínima {detector.threshold:g}). Selecciónalo a mano.")
        return a_mano("el detector no encontró al sujeto")
    info["confianza"] = round(det.score, 4)
    info["caja_propuesta"] = list(bbox)  # en coordenadas del recorte del ROI, aunque luego se marque a mano
    out(f"El detector propone al sujeto con confianza {det.score:.2f}" + ("." if det.score >= CONFIANZA_ALTA else " (baja: revisa bien la caja)."))
    x, y, w, h = (int(v) for v in roi_box)
    decision = confirm(sample_frame[y:y + h, x:x + w], bbox, det.score)
    if decision == ACEPTAR:
        info.update(metodo="detector", motivo="propuesta aceptada")
        return bbox, info
    if decision == CANCELAR:
        info.update(metodo="cancelado", motivo="cancelado al confirmar la propuesta")
        return None, info
    return a_mano("propuesta rechazada: se marcó a mano")