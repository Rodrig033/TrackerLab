from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from tracking import (
    MASK_FOREGROUND,
    MODE_BACKGROUND,
    BaseTracker,
    Box,
    Detection,
    Point,
)

__all__ = [
    "DEFAULT_PARAMS",
    "DIST_W",
    "POLARITY_WHITE_ON_BLACK",
    "shrink_mask",
    "preprocess_gray",
    "compute_background",
    "foreground_binary",
    "pick_component",
    "BackgroundTracker",
]

#: Valores iniciales de la calibración (``DEF_*`` del original). Son los que se
#: ofrecen en la ventana de calibración; NO son los que se asumen al cargar un
#: JSON antiguo sin ``SOL`` (ver ``normalize_params``).
DEFAULT_PARAMS: Dict[str, Any] = {
    "DIFF": 18,
    "OPEN": 5,
    "CLOSE": 9,
    "MINA": 200,
    "MAXA": 80000,
    "SHRINK": 0,
    "SOL": 0.55,
}

#: Peso de la distancia en la elección del componente (igual que el original).
DIST_W = 0.003

POLARITY_WHITE_ON_BLACK = "white_on_black"

_BW_THRESHOLD = 160  # umbral fijo del filtro opcional "apply_bw"
_REQUIRED_PARAMS = ("DIFF", "OPEN", "CLOSE", "MINA", "MAXA")

# Utilidades
def _apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Deja ``img`` donde ``mask`` es distinta de cero y pone 0 en el resto.

    Es lo mismo que ``cv2.bitwise_and(img, img, mask=mask)`` del original, con
    el mismo resultado bit a bit para cualquier valor de la máscara, pero
    ``cv2.copyTo`` es ~80 veces más rápido: la variante con ``mask=`` de
    bitwise_and tardaba ~25 ms en una imagen de 2 MP y, al llamarse dos veces
    por frame, era el 90 % del tiempo del modo 1.
    """
    return cv2.copyTo(img, mask)


@lru_cache(maxsize=64)
def _ellipse_kernel(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def shrink_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Erosiona la máscara ``px`` píxeles (``ROI_SHRINK`` de la calibración)."""
    px = int(px)
    if px <= 0:
        return mask
    k = 2 * px + 1
    return cv2.erode(mask, _ellipse_kernel(k), iterations=1)


def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Valida los parámetros y completa los opcionales como lo hacía el original.

    Obligatorios: DIFF, OPEN, CLOSE, MINA, MAXA. Opcionales: SHRINK (0) y SOL
    (0.0, es decir SIN filtro de solidez). Ojo: ese 0.0 no es el 0.55 de
    ``DEFAULT_PARAMS``; es lo que el original asumía al leer un JSON antiguo,
    y cambiarlo alteraría los resultados de experimentos ya guardados.
    """
    missing = [k for k in _REQUIRED_PARAMS if k not in params]
    if missing:
        raise ValueError(f"Faltan parámetros del modo fondo: {missing}")
    out = dict(params)
    out.setdefault("SHRINK", 0)
    out.setdefault("SOL", 0.0)
    return out


# Procesamiento de imagen (mismas operaciones y orden que el original)
def preprocess_gray(
    roi_bgr: np.ndarray,
    include_mask: np.ndarray,
    use_clahe: bool,
    clahe: Optional["cv2.CLAHE"] = None,
) -> np.ndarray:
    """Gris + máscara del ROI (+ CLAHE opcional).

    ``clahe``: objeto ya creado para reutilizarlo; si es ``None`` y
    ``use_clahe`` es True se crea uno nuevo (comportamiento del original).
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = _apply_mask(gray, include_mask)
    if use_clahe:
        if clahe is None:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray


def compute_background(
    frames: Iterable[np.ndarray],
    roi_box: Tuple[int, int, int, int],
    include_mask: np.ndarray,
    use_clahe: bool,
) -> Optional[np.ndarray]:
    """Fondo = mediana (por píxel) de los frames dados, recortados al ROI.

    frames: frames COMPLETOS en BGR, p. ej. ``read_background_frames(...)``.
    roi_box: (x0, y0, w0, h0) del ROI dentro del frame.
    Devuelve el fondo en gris (uint8) o ``None`` si no llegó ningún frame.
    """
    x0, y0, w0, h0 = roi_box
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
    grays = [
        preprocess_gray(fr[y0:y0 + h0, x0:x0 + w0], include_mask, use_clahe, clahe)
        for fr in frames
    ]
    if not grays:
        return None
    return np.median(np.stack(grays, axis=0), axis=0).astype(np.uint8)


def foreground_binary(
    roi_bgr: np.ndarray,
    bg_gray: np.ndarray,
    include_mask: np.ndarray,
    diff_thr: float,
    open_k: int,
    close_k: int,
    use_clahe: bool,
    apply_bw: bool,
    polarity: str,
    clahe: Optional["cv2.CLAHE"] = None,
) -> np.ndarray:
    """Máscara binaria (0/255) de lo que cambió respecto al fondo."""
    gray = preprocess_gray(roi_bgr, include_mask, use_clahe, clahe)

    diff = cv2.absdiff(gray, bg_gray)
    _, fg = cv2.threshold(diff, int(diff_thr), 255, cv2.THRESH_BINARY)

    if apply_bw:
        if polarity == POLARITY_WHITE_ON_BLACK:
            _, bw = cv2.threshold(gray, _BW_THRESHOLD, 255, cv2.THRESH_BINARY)
        else:
            _, bw = cv2.threshold(gray, _BW_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        fg = cv2.bitwise_and(fg, bw)

    # Los tamaños de kernel se fuerzan a impar (como el original).
    open_k = int(open_k) | 1
    close_k = int(close_k) | 1
    if open_k > 1:
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _ellipse_kernel(open_k), iterations=1)
    if close_k > 1:
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _ellipse_kernel(close_k), iterations=1)

    return _apply_mask(fg, include_mask)


def pick_component(
    mask: np.ndarray,
    last_center: Point,
    min_area: float,
    max_area: float,
    sol_min: float = 0.0,
) -> Optional[Tuple[float, float, Box]]:
    """Elige el componente conectado válido más cercano a ``last_center``.

    Filtra por área (y por solidez si ``sol_min > 0``). Devuelve
    ``(cx, cy, (x, y, w, h))`` o ``None`` si no hay ninguno válido.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    lx, ly = last_center
    best = None
    best_score = 1e18

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        if sol_min > 0:
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 1e-6:
                continue
            solidity = area / hull_area
            if solidity < sol_min:
                continue

        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
        score = DIST_W * dist

        if score < best_score:
            x, y, w, h = cv2.boundingRect(c)
            best_score = score
            best = (cx, cy, (x, y, w, h))

    return best


# Tracker
class BackgroundTracker(BaseTracker):
    """Tracker por diferencia con el fondo (modo "1")."""

    mode_id = MODE_BACKGROUND

    def __init__(
        self,
        include_mask: np.ndarray,
        bg_gray: np.ndarray,
        params: Dict[str, Any],
        polarity: str = POLARITY_WHITE_ON_BLACK,
        apply_bw: bool = False,
        use_clahe: bool = False,
    ):
        """
        include_mask: máscara uint8 (0/255) del ROI, en coordenadas del recorte.
        bg_gray: fondo en gris (``compute_background``), mismo tamaño que el recorte.
        params: DIFF, OPEN, CLOSE, MINA, MAXA y opcionalmente SHRINK y SOL.
        polarity / apply_bw / use_clahe: opciones de la sección ``mode`` del JSON.
        """
        super().__init__(include_mask, normalize_params(params))
        self.polarity = polarity
        self.apply_bw = bool(apply_bw)
        self.use_clahe = bool(use_clahe)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if self.use_clahe else None

        self.bg_gray: np.ndarray
        self.set_background(bg_gray)

        # Caché de la máscara erosionada: (SHRINK, máscara)
        self._shrunk: Optional[Tuple[int, np.ndarray]] = None

    def set_background(self, bg_gray: np.ndarray) -> None:
        """Cambia el fondo (tecla 'b': recapturar alrededor del frame actual)."""
        if bg_gray is None or bg_gray.shape != self.include_mask.shape or bg_gray.dtype != np.uint8:
            raise ValueError("El fondo debe ser uint8 y del mismo tamaño que la máscara del ROI.")
        self.bg_gray = bg_gray

    def _mask_for(self, shrink: int) -> np.ndarray:
        """Máscara del ROI erosionada ``shrink`` px, calculada una sola vez."""
        shrink = int(shrink)
        if self._shrunk is None or self._shrunk[0] != shrink:
            self._shrunk = (shrink, shrink_mask(self.include_mask, shrink))
        return self._shrunk[1]

    def _run(self, roi_bgr: np.ndarray, p: Dict[str, Any], ref_center: Point) -> Detection:
        fg = foreground_binary(
            roi_bgr, self.bg_gray, self._mask_for(p["SHRINK"]),
            p["DIFF"], p["OPEN"], p["CLOSE"],
            use_clahe=self.use_clahe, apply_bw=self.apply_bw,
            polarity=self.polarity, clahe=self._clahe,
        )
        pick = pick_component(fg, ref_center, p["MINA"], p["MAXA"], sol_min=p["SOL"])
        if pick is None:
            return Detection(mask=fg, mask_kind=MASK_FOREGROUND)
        cx, cy, bbox = pick
        return Detection(center=(cx, cy), bbox=bbox, mask=fg, mask_kind=MASK_FOREGROUND)

    def _detect(self, roi_bgr: np.ndarray) -> Detection:
        # normalize_params por si alguien reasignó self.params (copia barata)
        return self._run(roi_bgr, normalize_params(self.params), self.last_center)

    def preview(self, roi_bgr: np.ndarray, params: Dict[str, Any], ref_center: Point) -> Detection:
        """Detección de prueba con parámetros candidatos (sliders de calibración).

        No modifica el estado del tracker ni sus parámetros.
        """
        return self._run(roi_bgr, normalize_params(params), ref_center)