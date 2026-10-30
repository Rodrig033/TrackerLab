from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from tracking import (
    MASK_SILHOUETTE,
    MODE_SILHOUETTE,
    BaseTracker,
    Box,
    Detection,
    Point,
)

__all__ = [
    "POLARITY_LIGHT",
    "POLARITY_DARK",
    "SilhouetteModel",
    "create_model_from_bbox",
    "SilhouetteTracker",
]

POLARITY_LIGHT = "light"
POLARITY_DARK = "dark"


# Utilidades
@lru_cache(maxsize=8)
def _ellipse_kernel(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _clamp_box(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(0, min(int(w), W - x))
    h = max(0, min(int(h), H - y))
    return x, y, w, h


def _otsu_light(gray: np.ndarray) -> np.ndarray:
    """Blur + Otsu: máscara de lo CLARO (se invierte para la polaridad oscura)."""
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, m_light = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m_light


def _binary_from_light(m_light: np.ndarray, polarity: str, valid_mask: Optional[np.ndarray]) -> np.ndarray:
    """Aplica polaridad, máscara válida y morfología (apertura 3x3, cierre 5x5)."""
    m = m_light if polarity == POLARITY_LIGHT else cv2.bitwise_not(m_light)
    if valid_mask is not None:
        m = cv2.copyTo(m, valid_mask)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse_kernel(3), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse_kernel(5), iterations=1)
    return m


def _make_binary_for_subject(gray: np.ndarray, polarity: str = POLARITY_LIGHT,
                             valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    return _binary_from_light(_otsu_light(gray), polarity, valid_mask)


class _Component(NamedTuple):
    area: float
    bbox: Box
    center: Point
    label: int


def _components(mask: np.ndarray, min_area: float = 10,
                max_area: Optional[float] = None) -> Tuple[np.ndarray, List[_Component]]:
    """Componentes conectados (8-vecinos) con área en [min_area, max_area].

    Devuelve también la imagen de etiquetas: la máscara de un candidato se
    obtiene después con ``labels == comp.label`` (solo para el elegido).
    """
    _, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    keep = areas >= min_area
    if max_area is not None:
        keep &= areas <= max_area
    out: List[_Component] = []
    for i in np.nonzero(keep)[0]:
        lab = int(i) + 1
        out.append(_Component(
            area=float(areas[i]),
            bbox=(int(stats[lab, cv2.CC_STAT_LEFT]), int(stats[lab, cv2.CC_STAT_TOP]),
                  int(stats[lab, cv2.CC_STAT_WIDTH]), int(stats[lab, cv2.CC_STAT_HEIGHT])),
            center=(float(cents[lab][0]), float(cents[lab][1])),
            label=lab,
        ))
    return labels, out


def _touches_border(bbox: Box, box_w: int, box_h: int) -> bool:
    x, y, w, h = bbox
    return x <= 0 or y <= 0 or x + w >= box_w or y + h >= box_h


def _choose_best_initial_component(cands: List[_Component], box_w: int, box_h: int,
                                   border_aware: bool = False) -> Optional[_Component]:
    """Elige el componente que mejor representa al sujeto dentro de la ventana dibujada.

    Con ``border_aware`` los componentes que NO tocan el borde de la ventana tienen prioridad: el sujeto
    dibujado queda dentro de la ventana, mientras que el fondo siempre la toca. Si todos la tocan (ventana
    muy ajustada), se aplica la regla de siempre.
    """
    if not cands:
        return None
    cx_box, cy_box = box_w / 2.0, box_h / 2.0
    box_area = float(box_w * box_h)
    best, best_key = None, None
    for c in cands:
        if c.area < max(20.0, 0.01 * box_area) or c.area > 0.92 * box_area:
            continue
        cx, cy = c.center
        dist = ((cx - cx_box) ** 2 + (cy - cy_box) ** 2) ** 0.5
        score = dist - 0.008 * c.area
        key = (border_aware and _touches_border(c.bbox, box_w, box_h), score)
        if best is None or key < best_key:
            best, best_key = c, key
    return best


def _mask_of(labels: np.ndarray, label: int) -> np.ndarray:
    return (labels == label).astype(np.uint8) * 255


# Modelo del sujeto
@dataclass
class SilhouetteModel:
    """Lo aprendido del sujeto; se actualiza mientras se hace el seguimiento.

    Equivale al diccionario ``window_model`` del original.
    """

    polarity: str  # "light" (sujeto claro sobre fondo oscuro) o "dark"
    area_ref: float  # área de referencia; se suaviza 95 % / 5 % en cada frame
    bbox: Box  # (x, y, w, h) en coordenadas del recorte del ROI
    center: Point
    last_silhouette: Optional[np.ndarray] = None  # uint8 0/255, tamaño del ROI


def create_model_from_bbox(
    roi_bgr: np.ndarray,
    bbox: Tuple[float, float, float, float],
    include_mask: Optional[np.ndarray] = None,
    border_aware: bool = False,
) -> Optional[SilhouetteModel]:
    x, y, w, h = [int(round(v)) for v in bbox]
    x, y, w, h = _clamp_box(x, y, w, h, roi_bgr.shape[1], roi_bgr.shape[0])
    if w <= 5 or h <= 5:
        return None

    crop = roi_bgr[y:y + h, x:x + w]
    valid_crop = None
    if include_mask is not None:
        inc = include_mask[y:y + h, x:x + w]
        crop = cv2.copyTo(crop, inc)  # enmascarar igual que main() antes de pasar a gris
        if inc.shape[:2] == crop.shape[:2] and cv2.countNonZero(inc) > 0:
            valid_crop = inc
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    m_light = _otsu_light(gray)
    best = None  # (clave, componente, etiquetas, polaridad); clave = (toca el borde, puntaje)
    for pol in (POLARITY_LIGHT, POLARITY_DARK):
        m = _binary_from_light(m_light, pol, valid_crop)
        labels, cand = _components(m, min_area=10, max_area=0.95 * w * h)
        chosen = _choose_best_initial_component(cand, w, h, border_aware)
        if chosen is None:
            continue
        cx, cy = chosen.center
        dist = ((cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2) ** 0.5
        score = dist - 0.008 * chosen.area
        key = (border_aware and _touches_border(chosen.bbox, w, h), score)
        if best is None or key < best[0]:
            best = (key, chosen, labels, pol)

    if best is None:
        return None

    _, chosen, labels, pol = best
    bx, by, bw, bh = chosen.bbox
    sil_full = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    sil_full[y:y + h, x:x + w] = _mask_of(labels, chosen.label)
    return SilhouetteModel(
        polarity=pol,
        area_ref=float(chosen.area),
        bbox=(x + bx, y + by, bw, bh),
        center=(float(x + chosen.center[0]), float(y + chosen.center[1])),
        last_silhouette=sil_full,
    )



# Tracker
class SilhouetteTracker(BaseTracker):
    """Tracker por silueta (modo "2")."""

    mode_id = MODE_SILHOUETTE

    def __init__(self, include_mask: np.ndarray, model: SilhouetteModel, border_aware: bool = False):
        super().__init__(include_mask, None)
        self.model = model
        self.border_aware = bool(border_aware)

    @classmethod
    def from_bbox(
        cls,
        include_mask: np.ndarray,
        roi_bgr: np.ndarray,
        bbox: Tuple[float, float, float, float],
        border_aware: bool = False,
    ) -> Optional["SilhouetteTracker"]:
        """Crea el tracker aprendiendo la silueta de ``bbox`` (o ``None`` si no se pudo).

        border_aware=True hace la selección independiente del margen de la ventana (ver
        ``create_model_from_bbox``). Se recuerda para los reinicios con ``reinit``.
        """
        model = create_model_from_bbox(roi_bgr, bbox, include_mask, border_aware)
        if model is None:
            return None
        tr = cls(include_mask, model, border_aware)
        tr.reset(model.center, model.bbox)
        return tr

    def _check(self, roi_bgr: np.ndarray) -> None:
        if roi_bgr.shape[:2] != self.include_mask.shape[:2]:
            raise ValueError(
                f"El recorte {roi_bgr.shape[:2]} no coincide con la máscara del ROI {self.include_mask.shape[:2]}."
            )

    def locate(self, roi_bgr: np.ndarray) -> Detection:
        self._check(roi_bgr)
        model = self.model
        H, W = roi_bgr.shape[:2]
        if H <= 2 or W <= 2:
            return Detection(mask_kind=MASK_SILHOUETTE)

        gray = cv2.cvtColor(cv2.copyTo(roi_bgr, self.include_mask), cv2.COLOR_BGR2GRAY)
        valid = self.include_mask if cv2.countNonZero(self.include_mask) > 0 else None

        area_ref = max(float(model.area_ref), 20.0)
        min_area = max(12.0, 0.25 * area_ref)
        max_area = min(float(H * W) * 0.85, 4.0 * area_ref)

        preferred = model.polarity
        pols = [preferred, POLARITY_DARK if preferred == POLARITY_LIGHT else POLARITY_LIGHT]
        m_light = _otsu_light(gray)

        best = None  # (score, componente, etiquetas, polaridad)
        for pol_i, pol in enumerate(pols):
            m = _binary_from_light(m_light, pol, valid)
            labels, cand = _components(m, min_area=min_area, max_area=max_area)
            for c in cand:
                area_penalty = abs(c.area - area_ref) / area_ref
                _, _, bw, bh = c.bbox
                aspect = max(bw / max(bh, 1), bh / max(bw, 1))
                aspect_penalty = max(0.0, aspect - 5.0)
                pol_penalty = 0.0 if pol_i == 0 else 15.0
                score = 45.0 * area_penalty + 8.0 * aspect_penalty + pol_penalty
                if best is None or score < best[0]:
                    best = (score, c, labels, pol)

        if best is None:
            return Detection(mask_kind=MASK_SILHOUETTE)

        _, chosen, labels, pol = best
        sil_full = _mask_of(labels, chosen.label)
        model.polarity = pol
        model.area_ref = 0.95 * area_ref + 0.05 * float(chosen.area)
        model.bbox = chosen.bbox
        model.center = (float(chosen.center[0]), float(chosen.center[1]))
        model.last_silhouette = sil_full
        return Detection(center=model.center, bbox=chosen.bbox, mask=sil_full, mask_kind=MASK_SILHOUETTE)

    def start(self, roi_start_bgr: np.ndarray) -> Detection:
        det = self.locate(roi_start_bgr)
        if det.found:
            self.reset(det.center, det.bbox)
            return det
        self.reset(self.model.center, self.model.bbox)
        return Detection(mask=self.model.last_silhouette, mask_kind=MASK_SILHOUETTE)

    def reinit(self, roi_bgr: np.ndarray, bbox: Tuple[float, float, float, float]) -> bool:
        model = create_model_from_bbox(roi_bgr, bbox, self.include_mask, self.border_aware)
        if model is None:
            return False
        self.model = model
        self.reset(model.center, model.bbox)
        return True

    def _detect(self, roi_bgr: np.ndarray) -> Detection:
        self._check(roi_bgr)
        model = self.model
        H, W = roi_bgr.shape[:2]
        bx, by, bw, bh = [int(round(v)) for v in self.last_bbox]
        lx, ly = float(self.last_center[0]), float(self.last_center[1])

        margin = int(max(50, 1.4 * max(bw, bh)))
        sx1 = max(0, min(int(lx - margin), W - 1))
        sy1 = max(0, min(int(ly - margin), H - 1))
        sx2 = min(W, max(int(lx + margin), sx1 + 2))
        sy2 = min(H, max(int(ly + margin), sy1 + 2))

        search = roi_bgr[sy1:sy2, sx1:sx2]
        if search.size == 0:
            return Detection(mask_kind=MASK_SILHOUETTE)

        # Enmascarar SOLO la ventana (el original enmascaraba el ROI completo).
        inc = self.include_mask[sy1:sy2, sx1:sx2]
        gray = cv2.cvtColor(cv2.copyTo(search, inc), cv2.COLOR_BGR2GRAY)
        valid_search = inc if (inc.shape[:2] == gray.shape[:2] and cv2.countNonZero(inc) > 0) else None

        area_ref = max(float(model.area_ref), 20.0)
        min_area = max(12.0, 0.25 * area_ref)
        max_area = min(float(gray.shape[0] * gray.shape[1]) * 0.85, 4.0 * area_ref)

        pref = model.polarity
        m_light = _otsu_light(gray)
        candidates: List[_Component] = []
        labels = None
        for pol in (pref, POLARITY_DARK if pref == POLARITY_LIGHT else POLARITY_LIGHT):
            m = _binary_from_light(m_light, pol, valid_search)
            labels_i, cand = _components(m, min_area=min_area, max_area=max_area)
            if cand:
                candidates, labels = cand, labels_i
                break

        if not candidates:
            return Detection(mask_kind=MASK_SILHOUETTE)

        best, best_score, best_cx, best_cy = None, None, 0.0, 0.0
        for c in candidates:
            cx = sx1 + c.center[0]
            cy = sy1 + c.center[1]
            dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
            area_penalty = abs(c.area - area_ref) / area_ref
            score = dist + 35.0 * area_penalty
            if best is None or score < best_score:
                best, best_score, best_cx, best_cy = c, score, cx, cy

        cbx, cby, cbw, cbh = best.bbox
        bbox_body = (sx1 + cbx, sy1 + cby, cbw, cbh)
        sil_full = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
        sil_full[sy1:sy2, sx1:sx2] = _mask_of(labels, best.label)

        model.area_ref = 0.95 * area_ref + 0.05 * float(best.area)
        model.bbox = bbox_body
        model.center = (float(best_cx), float(best_cy))
        model.last_silhouette = sil_full
        return Detection(center=(float(best_cx), float(best_cy)), bbox=bbox_body,
                         mask=sil_full, mask_kind=MASK_SILHOUETTE)