from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.dataset import parse_yolo_label_line  # noqa: E402
from learning.review import (ACCEPTED, CORRECTED, DISCARDED, DOUBTFUL, PENDING, box_iou, format_stats, load_revision,  # noqa: E402
                             new_revision, review_stats, save_revision)
from tools.build_training_set import crop_rect, find_roi, list_items, read_info  # noqa: E402

Box = Tuple[float, float, float, float]
MIN_SIDE = 3  # px del frame: una caja más pequeña se considera un clic accidental
BAR_H = 24
WINDOW = "TrackerLab - revisar cajas (h = ayuda)"
COL_AUTO, COL_OK, COL_NEW, COL_OLD = (0, 220, 255), (0, 200, 0), (255, 160, 0), (150, 150, 150)  # BGR

HELP = [
    "COMO REVISAR (caja ajustada al CUERPO, SIN la cola)",
    "espacio o Enter: la caja esta bien, siguiente",
    "arrastrar: dibuja la caja correcta; luego espacio",
    "x: descartar (no se ve el raton / caja absurda)",
    "k: dudosa (no se usara)      u: deshacer esta imagen",
    "d / a: siguiente / anterior sin decidir    n: pendiente",
    "q o ESC: salir  (todo se guarda al instante)",
]


@dataclass
class Item:
    key: str  # "train/nombre": así se identifica en revision.json
    image_path: str
    original: Optional[Box]  # caja automática, en píxeles del frame completo


def load_items(dsdir: str) -> Tuple[List[Item], Tuple[int, int]]:
    """Imágenes del conjunto en orden de nombre (= orden de frame) y el tamaño (ancho, alto) del frame."""
    crudos = sorted(list_items(dsdir, ("train", "val")), key=lambda t: (t[1], t[0]))
    if not crudos:
        raise ValueError(f"{dsdir} no tiene imágenes en images/train ni images/val.")
    primera = cv2.imread(crudos[0][2])
    if primera is None:
        raise ValueError(f"No se pudo leer {crudos[0][2]}.")
    fh, fw = primera.shape[:2]
    items: List[Item] = []
    for sp, nombre, img, lab in crudos:
        caja: Optional[Box] = None
        if os.path.exists(lab):
            try:
                with open(lab, encoding="utf-8-sig") as fh_:
                    lineas = [ln for ln in fh_.read().splitlines() if ln.strip()]
                if len(lineas) == 1:
                    _, caja = parse_yolo_label_line(lineas[0], fw, fh)
            except ValueError:
                caja = None
        items.append(Item(f"{sp}/{nombre}", img, caja))
    return items, (fw, fh)


class Reviewer:
    """Estado y lógica de la revisión. La ventana solo le pasa teclas y eventos del ratón y pide ``render()``."""

    def __init__(self, dsdir: str, items: List[Item], frame_size: Tuple[int, int], roi_box: Optional[Tuple[int, int, int, int]] = None,
                 margin: int = 20, max_w: int = 1100, max_h: int = 760, max_zoom: float = 3.0, start_at_pending: bool = True):
        self.dir, self.items = dsdir, items
        self.fw, self.fh = frame_size
        self.rect = crop_rect(roi_box, margin, self.fw, self.fh) if roi_box else (0, 0, self.fw, self.fh)
        cw, ch = self.rect[2] - self.rect[0], self.rect[3] - self.rect[1]
        self.scale = min(max_zoom, max_w / cw, max_h / ch)
        self.rev = load_revision(dsdir) or new_revision()  # ValueError si está dañado: no se sobrescribe
        self.i = 0
        self.msg = ""
        self.show_help = False
        self._drag: Optional[Tuple[float, float]] = None
        self._cursor: Optional[Tuple[float, float]] = None
        self._cache: Optional[Tuple[int, np.ndarray]] = None
        if start_at_pending:
            k = self._next_pending(-1)
            self.i = k if k is not None else 0

    # ------------------------------------------------------------- estado
    @property
    def item(self) -> Item:
        return self.items[self.i]

    def entry(self, k: Optional[int] = None) -> Optional[dict]:
        return self.rev["imagenes"].get(self.items[self.i if k is None else k].key)

    @property
    def state(self) -> Tuple[str, Optional[Box]]:
        """(estado, caja que se muestra): la final si ya se decidió; si no, la automática."""
        e = self.entry()
        if e is None:
            return PENDING, self.item.original
        return e["estado"], (tuple(e["final"]) if e.get("final") else None)  # type: ignore[return-value]

    def counts(self) -> Dict[str, int]:
        c = {PENDING: 0, ACCEPTED: 0, CORRECTED: 0, DISCARDED: 0, DOUBTFUL: 0}
        for it in self.items:
            e = self.rev["imagenes"].get(it.key)
            c[e["estado"] if e else PENDING] += 1
        c["total"] = len(self.items)
        return c

    def stats(self) -> Dict:
        return review_stats(self.rev, total=len(self.items))

    def _next_pending(self, after: int) -> Optional[int]:
        n = len(self.items)
        for step in range(1, n + 1):
            k = (after + step) % n
            if self.items[k].key not in self.rev["imagenes"]:
                return k
        return None

    def _decide(self, estado: str, final: Optional[Box]) -> None:
        o = self.item.original
        self.rev["imagenes"][self.item.key] = {"estado": estado, "original": list(o) if o else None, "final": [float(v) for v in final] if final else None}
        save_revision(self.dir, self.rev)

    def goto(self, k: int) -> None:
        self.i = max(0, min(len(self.items) - 1, k))
        self._drag = None

    def _advance(self) -> None:
        if self.i < len(self.items) - 1:
            self.goto(self.i + 1)
        else:
            self.msg = "es la ultima imagen" + ("" if self._next_pending(self.i) is None else " (quedan pendientes: n)")

    # ------------------------------------------------------------ geometría
    def _to_frame(self, x: float, y: float) -> Tuple[float, float]:
        """De coordenadas de la ventana (con la barra de arriba) a coordenadas del frame completo."""
        return self.rect[0] + x / self.scale, self.rect[1] + (y - BAR_H) / self.scale

    def _to_view(self, x: float, y: float) -> Tuple[int, int]:
        return int(round((x - self.rect[0]) * self.scale)), int(round((y - self.rect[1]) * self.scale))

    def _normalize(self, x0: float, y0: float, x1: float, y1: float) -> Optional[Box]:
        """Caja (x, y, ancho, alto) en píxeles enteros del frame, recortada a lo que se ve. None si es demasiado pequeña."""
        L, T, R, B = self.rect
        ax, bx = sorted((min(max(x0, L), R), min(max(x1, L), R)))
        ay, by = sorted((min(max(y0, T), B), min(max(y1, T), B)))
        x, y, x2, y2 = int(round(ax)), int(round(ay)), int(round(bx)), int(round(by))
        if x2 - x < MIN_SIDE or y2 - y < MIN_SIDE:
            return None
        return (float(x), float(y), float(x2 - x), float(y2 - y))

    # --------------------------------------------------------------- eventos
    def on_mouse(self, event: int, x: float, y: float, flags: int = 0) -> None:
        px, py = self._to_frame(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag = (px, py)
            self._cursor = (px, py)
        elif event == cv2.EVENT_MOUSEMOVE:
            self._cursor = (px, py)
        elif event == cv2.EVENT_LBUTTONUP and self._drag is not None:
            x0, y0 = self._drag
            self._drag = None
            box = self._normalize(x0, y0, px, py)
            if box is None:
                self.msg = "caja demasiado pequena: se ignora"
                return
            self._decide(CORRECTED, box)
            self.msg = "caja corregida guardada (espacio = siguiente)"

    def on_key(self, key: int) -> Optional[str]:
        """Procesa una tecla. Devuelve ``"quit"`` para salir."""
        if key == 27:
            return "quit"
        if key in (32, 13, 10):
            ch = " "
        else:
            try:
                ch = chr(key).lower()
            except (ValueError, OverflowError):
                return None
        self.msg = ""
        if ch == "q":
            return "quit"
        if ch == " ":
            estado, _ = self.state
            if estado == PENDING:
                if self.item.original is None:
                    self.msg = "no hay caja automatica: dibuja una o pulsa x"
                    return None
                self._decide(ACCEPTED, self.item.original)
                self.msg = "aceptada"
            self._advance()
        elif ch == "d":
            self._advance()
        elif ch == "a":
            if self.i > 0:
                self.goto(self.i - 1)
            else:
                self.msg = "es la primera imagen"
        elif ch == "x":
            self._decide(DISCARDED, None)
            self.msg = "descartada"
            self._advance()
        elif ch == "k":
            self._decide(DOUBTFUL, None)
            self.msg = "marcada como dudosa"
            self._advance()
        elif ch == "u":
            if self.rev["imagenes"].pop(self.item.key, None) is not None:
                save_revision(self.dir, self.rev)
                self.msg = "decision deshecha (vuelve a la caja automatica)"
            else:
                self.msg = "esta imagen ya estaba pendiente"
        elif ch == "n":
            k = self._next_pending(self.i)
            if k is None:
                self.msg = "no quedan imagenes pendientes"
            else:
                self.goto(k)
        elif ch == "h":
            self.show_help = not self.show_help
        return None

    # ---------------------------------------------------------------- dibujo
    def _base(self) -> np.ndarray:
        if self._cache is not None and self._cache[0] == self.i:
            return self._cache[1]
        img = cv2.imread(self.item.image_path)
        if img is None:
            raise OSError(f"No se pudo leer {self.item.image_path}")
        if img.shape[1] != self.fw or img.shape[0] != self.fh:
            raise OSError(f"{self.item.image_path} mide {img.shape[1]}x{img.shape[0]} y el conjunto es de {self.fw}x{self.fh}")
        L, T, R, B = self.rect
        crop = img[T:B, L:R]
        interp = cv2.INTER_CUBIC if self.scale > 1 else cv2.INTER_AREA
        crop = cv2.resize(crop, (int(round(crop.shape[1] * self.scale)), int(round(crop.shape[0] * self.scale))), interpolation=interp)
        self._cache = (self.i, crop)
        return crop

    def _rectangle(self, img: np.ndarray, box: Box, color: Tuple[int, int, int], thick: int = 2) -> None:
        x, y, w, h = box
        cv2.rectangle(img, self._to_view(x, y), self._to_view(x + w, y + h), color, thick)

    def render(self) -> np.ndarray:
        """Ventana completa: barra de estado, imagen recortada con zoom y barra de ayuda. Las barras NO tapan la imagen."""
        disp = self._base().copy()
        estado, box = self.state
        o = self.item.original
        if self._drag is not None and self._cursor is not None:
            live = self._normalize(self._drag[0], self._drag[1], self._cursor[0], self._cursor[1])
            if o is not None:
                self._rectangle(disp, o, COL_OLD, 1)
            if live is not None:
                self._rectangle(disp, live, COL_NEW)
        elif estado == PENDING and box is not None:
            self._rectangle(disp, box, COL_AUTO)  # amarilla: la automática, pendiente de decidir
        elif estado == ACCEPTED and box is not None:
            self._rectangle(disp, box, COL_OK)
        elif estado == CORRECTED and box is not None:
            if o is not None:
                self._rectangle(disp, o, COL_OLD, 1)  # la automática original, fina y gris
            self._rectangle(disp, box, COL_OK)
        elif estado in (DISCARDED, DOUBTFUL):
            cv2.putText(disp, "DESCARTADA" if estado == DISCARDED else "DUDOSA", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 255) if estado == DISCARDED else (0, 160, 255), 2, cv2.LINE_AA)
        if self.show_help:
            cv2.rectangle(disp, (0, 0), (min(disp.shape[1], 600), 22 * len(HELP) + 8), (20, 20, 20), -1)
            for j, ln in enumerate(HELP):
                cv2.putText(disp, ln, (8, 20 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        c = self.counts()
        hechas = c[ACCEPTED] + c[CORRECTED] + c[DISCARDED] + c[DOUBTFUL]
        etiqueta = {PENDING: "PENDIENTE", ACCEPTED: "ACEPTADA", CORRECTED: "CORREGIDA", DISCARDED: "DESCARTADA", DOUBTFUL: "DUDOSA"}[estado]
        extra = f"  IoU con la automatica {box_iou(o, box):.2f}" if estado == CORRECTED and o is not None and box is not None else ""
        w = disp.shape[1]
        top = np.full((BAR_H, w, 3), 30, np.uint8)
        cv2.putText(top, f"{self.i + 1}/{c['total']}  [{etiqueta}]{extra}  hechas {hechas}/{c['total']}  corregidas {c[CORRECTED]}", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        bottom = np.full((BAR_H, w, 3), 30, np.uint8)
        cv2.putText(bottom, self.msg or "espacio=aceptar  arrastrar=corregir  x=descartar  k=dudosa  u=deshacer  d/a=mover  n=pendiente  h=ayuda  q=salir",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200) if self.msg else (210, 210, 210), 1, cv2.LINE_AA)
        return np.vstack([top, disp, bottom])


# ----------------------------------------------------------------------
# Ventana
# ----------------------------------------------------------------------
def run_gui(rv: Reviewer, window: str = WINDOW) -> None:
    """Bucle de la ventana: dibuja, lee teclas y manda los eventos a ``rv``. Todo se guarda al instante."""
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, lambda event, x, y, flags, param: rv.on_mouse(event, x, y, flags))
    try:
        while True:
            cv2.imshow(window, rv.render())
            key = cv2.waitKey(20) & 0xFF
            if key != 255 and rv.on_key(key) == "quit":
                break
    finally:
        cv2.destroyAllWindows()


def main(argv: Optional[List[str]] = None, gui: Callable[[Reviewer], None] = run_gui, out: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(description="Revisa y corrige las cajas automáticas de un conjunto dataset_<fecha>.")
    ap.add_argument("dataset", help="carpeta dataset_<fecha> (la que deja una corrida con COLLECT_DATASET = True)")
    ap.add_argument("--margen-roi", type=int, default=20, help="píxeles de contexto alrededor del ROI al mostrar la imagen (por omisión 20)")
    ap.add_argument("--solo-resumen", action="store_true", help="no abrir la ventana: solo cuántas van y cuánto se equivocó la caja automática")
    ap.add_argument("--desde-inicio", action="store_true", help="empezar en la primera imagen, no en la primera pendiente")
    a = ap.parse_args(argv)
    try:
        info = read_info(a.dataset)
        items, size = load_items(a.dataset)
        roi, origen = find_roi(a.dataset, info)
        rv = Reviewer(a.dataset, items, size, roi, a.margen_roi, start_at_pending=not a.desde_inicio)
    except (FileNotFoundError, ValueError, OSError) as exc:
        out(str(exc))
        return 2
    if roi is None:
        out("AVISO: no se encontró el ROI de esta corrida; se muestra el frame completo (el ratón se verá pequeño).")
    sin_caja = sum(1 for it in items if it.original is None)
    if sin_caja:
        out(f"AVISO: {sin_caja} imágenes no tienen una caja automática válida: dibuja una o descártalas con x.")
    if not a.solo_resumen:
        out(f"{len(items)} imágenes. Espacio = aceptar la caja, arrastrar = corregirla, x = descartar, h = ayuda.")
        gui(rv)
    for ln in format_stats(rv.stats()):
        out(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())