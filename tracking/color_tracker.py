from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from tracking import (
    MASK_FOREGROUND,
    MODE_COLOR,
    BaseTracker,
    Box,
    Detection,
    Point,
)
from tracking.background_tracker import pick_component

__all__ = [
    "DEFAULT_PARAMS",
    "COLOR_MODE_HSV",
    "default_start",
    "initial_params_from_bbox",
    "finalize_params",
    "hsv_mask",
    "ColorTracker",
]

COLOR_MODE_HSV = "HSV"

#: Valores que se asumen cuando falta una clave (los ``params.get(k, valor)``
#: del original). No son los que propone la calibración inicial.
DEFAULT_PARAMS: Dict[str, Any] = {
    "H_MIN": 0,
    "H_MAX": 179,
    "S_MIN": 40,
    "S_MAX": 255,
    "V_MIN": 40,
    "V_MAX": 255,
    "OPEN": 3,
    "CLOSE": 5,
    "MINA": 20,
    "MAXA": 120000,
}
_KEYS = tuple(DEFAULT_PARAMS)


# Utilidades

@lru_cache(maxsize=32)
def _ellipse_kernel(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _clamp_box(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(0, min(int(w), W - x))
    h = max(0, min(int(h), H - y))
    return x, y, w, h


def _odd_kernel(value: Any) -> int:
    """Tamaño de kernel forzado a impar y >= 1 (como el original)."""
    return max(1, int(value) | 1)


class _Spec(NamedTuple):
    """Parámetros ya convertidos y listos para usar en cada frame."""

    circular: bool
    lower: np.ndarray
    upper: np.ndarray
    lower2: Optional[np.ndarray]
    upper2: Optional[np.ndarray]
    open_k: int
    close_k: int
    mina: Any
    maxa: Any


def _u8(*vals: int) -> np.ndarray:
    # Se acotan a 0..255: el original fallaba (o daba la vuelta) con valores
    # fuera de rango escritos a mano en un JSON.
    return np.array([min(255, max(0, v)) for v in vals], dtype=np.uint8)


def _make_spec(params: Dict[str, Any]) -> _Spec:
    p = {k: params.get(k, d) for k, d in DEFAULT_PARAMS.items()}
    hmin, hmax = int(p["H_MIN"]), int(p["H_MAX"])
    smin, smax = int(p["S_MIN"]), int(p["S_MAX"])
    vmin, vmax = int(p["V_MIN"]), int(p["V_MAX"])
    if hmin <= hmax:
        circular = False
        lower, upper = _u8(hmin, smin, vmin), _u8(hmax, smax, vmax)
        lower2 = upper2 = None
    else:  # rango circular: [0, hmax] U [hmin, 179]
        circular = True
        lower, upper = _u8(0, smin, vmin), _u8(hmax, smax, vmax)
        lower2, upper2 = _u8(hmin, smin, vmin), _u8(179, smax, vmax)
    return _Spec(
        circular, lower, upper, lower2, upper2,
        _odd_kernel(p["OPEN"]), _odd_kernel(p["CLOSE"]), p["MINA"], p["MAXA"],
    )


def _mask_from_spec(roi_bgr: np.ndarray, spec: _Spec, include_mask: Optional[np.ndarray]) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    if spec.circular:
        mask = cv2.bitwise_or(cv2.inRange(hsv, spec.lower, spec.upper),
                              cv2.inRange(hsv, spec.lower2, spec.upper2))
    else:
        mask = cv2.inRange(hsv, spec.lower, spec.upper)

    if spec.open_k > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse_kernel(spec.open_k), iterations=1)
    if spec.close_k > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse_kernel(spec.close_k), iterations=1)

    if include_mask is not None:
        mask = cv2.copyTo(mask, include_mask)
    return mask


# Funciones públicas
def hsv_mask(
    roi_bgr: np.ndarray,
    params: Dict[str, Any],
    include_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Máscara binaria (0/255) de los píxeles dentro del rango HSV.

    Equivale a ``_hsv_mask_from_params`` del original: NO enmascara el BGR de
    entrada (quien llama decide); ``include_mask`` solo se aplica al final.
    """
    return _mask_from_spec(roi_bgr, _make_spec(params), include_mask)


def default_start(w0: int, h0: int) -> Tuple[Point, Box]:
    """Posición de arranque cuando no hay una previa: centro del ROI y una
    caja del 10 % de su tamaño (lo que ``main()`` usaba en el modo 3)."""
    center = (w0 / 2.0, h0 / 2.0)
    bbox = (0, 0, max(1, int(w0 * 0.10)), max(1, int(h0 * 0.10)))
    return center, bbox


def initial_params_from_bbox(
    roi_bgr: np.ndarray,
    bbox: Tuple[float, float, float, float],
    include_mask: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Propone un rango HSV inicial a partir del color dentro de ``bbox``.

    Es el punto de partida de la calibración: el rango queda amplio alrededor
    del color elegido (mediana de H ± 12; percentiles 10-90 de S y V ± 35).
    Devuelve ``None`` si la caja queda vacía.
    """
    x, y, w, h = [int(round(v)) for v in bbox]
    x, y, w, h = _clamp_box(x, y, w, h, roi_bgr.shape[1], roi_bgr.shape[0])
    crop = roi_bgr[y:y + h, x:x + w].copy()
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    valid = np.ones(hsv.shape[:2], dtype=bool)
    if include_mask is not None:
        valid_crop = include_mask[y:y + h, x:x + w]
        if valid_crop.shape[:2] == hsv.shape[:2] and cv2.countNonZero(valid_crop) > 0:
            valid = valid_crop > 0

    pix = hsv[valid]
    if pix.size == 0:
        pix = hsv.reshape(-1, 3)

    hh = pix[:, 0].astype(np.int32)
    ss = pix[:, 1].astype(np.int32)
    vv = pix[:, 2].astype(np.int32)

    h_med = int(np.median(hh))
    h_margin = 12
    s_lo = max(0, int(np.percentile(ss, 10)) - 35)
    s_hi = min(255, int(np.percentile(ss, 90)) + 35)
    v_lo = max(0, int(np.percentile(vv, 10)) - 35)
    v_hi = min(255, int(np.percentile(vv, 90)) + 35)

    return {
        "H_MIN": max(0, h_med - h_margin),
        "H_MAX": min(179, h_med + h_margin),
        "S_MIN": s_lo,
        "S_MAX": s_hi,
        "V_MIN": v_lo,
        "V_MAX": v_hi,
        "OPEN": 3,
        "CLOSE": 5,
        "MINA": 20,
        "MAXA": max(500, int(0.95 * roi_bgr.shape[0] * roi_bgr.shape[1])),
    }


def finalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parámetros tal como el original los guardaba al aceptar la calibración.

    Fuerza OPEN y CLOSE a impares >= 1 y agrega ``COLOR_MODE: "HSV"``. La
    ventana de calibración debe llamarla antes de entregar los parámetros, para
    que el JSON del experimento sea igual al de la versión anterior.
    """
    out = dict(params)
    out["OPEN"] = _odd_kernel(out.get("OPEN", DEFAULT_PARAMS["OPEN"]))
    out["CLOSE"] = _odd_kernel(out.get("CLOSE", DEFAULT_PARAMS["CLOSE"]))
    out["COLOR_MODE"] = COLOR_MODE_HSV
    return out


# Tracker
class ColorTracker(BaseTracker):
    """Tracker por color HSV (modo "3")."""

    mode_id = MODE_COLOR

    def __init__(self, include_mask: np.ndarray, params: Optional[Dict[str, Any]] = None):
        """
        include_mask: máscara uint8 (0/255) del ROI, en coordenadas del recorte.
        params: ver el docstring del módulo (todo opcional).

        ``self.params`` se puede reemplazar o modificar en cualquier momento
        (p. ej. tras recalibrar con 'r'): el tracker detecta el cambio.
        """
        super().__init__(include_mask, params)
        self._spec_key: Optional[tuple] = None
        self._spec: Optional[_Spec] = None

    def _current_spec(self) -> _Spec:
        key = tuple(self.params.get(k) for k in _KEYS)
        if self._spec is None or key != self._spec_key:
            self._spec = _make_spec(self.params)
            self._spec_key = key
        return self._spec

    def _run(self, roi_bgr: np.ndarray, spec: _Spec, ref_center: Point) -> Detection:
        # El BGR se enmascara ANTES de pasar a HSV, como en el original: con
        # rangos permisivos lo enmascarado (negro) entra en el rango y afectaría
        # a la morfología cerca del borde del ROI.
        masked = cv2.copyTo(roi_bgr, self.include_mask)
        mask = _mask_from_spec(masked, spec, self.include_mask)
        pick = pick_component(mask, ref_center, spec.mina, spec.maxa, sol_min=0.0)
        if pick is None:
            return Detection(mask=mask, mask_kind=MASK_FOREGROUND)
        cx, cy, bbox = pick
        return Detection(center=(cx, cy), bbox=bbox, mask=mask, mask_kind=MASK_FOREGROUND)

    def _detect(self, roi_bgr: np.ndarray) -> Detection:
        return self._run(roi_bgr, self._current_spec(), self.last_center)

    def preview(self, roi_bgr: np.ndarray, params: Dict[str, Any], ref_center: Point) -> Detection:
        """Detección de PRUEBA con parámetros candidatos (sliders de calibración).

        No modifica el estado ni los parámetros del tracker.
        """
        return self._run(roi_bgr, _make_spec(params), ref_center)