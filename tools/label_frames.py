from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.dataset import parse_yolo_label_line, yolo_label_line  # noqa: E402

Box = Tuple[float, float, float, float]

PENDING, BOX, EMPTY, DOUBTFUL = "pendiente", "caja", "sin raton", "dudosa"
MIN_SIDE = 3  # px de imagen: una caja más pequeña se considera un clic accidental
BAR_H = 24  # alto de las barras de estado (arriba) y de ayuda (abajo): van FUERA de la imagen, no encima
WINDOW = "TrackerLab - etiquetar (h = ayuda)"

HELP = [
    "COMO ETIQUETAR",
    "arrastra: caja ajustada al CUERPO del raton, SIN la cola",
    "x: no se ve el raton (etiqueta vacia) y pasa a la siguiente",
    "d o espacio: siguiente    a: anterior    n: proxima pendiente",
    "u: borra la etiqueta de esta imagen",
    "k: dudosa (queda sin etiquetar y se anota en dudosas.txt)",
    "q o ESC: salir   (todo se guarda al instante)",
]


def stem_of(video_id: str, frame: int) -> str:
    """Nombre (sin extensión) de la imagen y de su etiqueta. Debe coincidir con tools/sample_frames.py y eval_tracking.py."""
    return f"{video_id}_f{frame:06d}"


@dataclass
class Item:
    stem: str
    image_path: str
    label_path: str
    video: str
    frame: int
    width: int
    height: int


def find_items(items: List[Item], spec: str) -> List[int]:
    """Posiciones de las imágenes que corresponden a ``spec``: un número de frame («2757», «f2757») o «video:frame» («Grupo1R1S2:2757»).

    Si el frame existe en varios videos del conjunto y no se dice cuál, devuelve todas las coincidencias.
    """
    texto = spec.strip()
    video: Optional[str] = None
    if ":" in texto:
        video, texto = texto.rsplit(":", 1)
        video = video.strip()
    texto = texto.strip().lstrip("fF")
    if not texto.isdigit():
        return []
    frame = int(texto)
    return [k for k, it in enumerate(items) if it.frame == frame and (video is None or it.video == video)]


def load_items(valdir: str) -> List[Item]:
    """Imágenes del conjunto, en el orden del manifiesto (video por video, frame por frame). Solo las que existen."""
    path = os.path.join(valdir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe {path}: genera el conjunto con tools/sample_frames.py.")
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    items: List[Item] = []
    for v in manifest["videos"]:
        ext = v.get("formato", "jpg")
        revisado = False
        for frame in v["frames"]:
            stem = stem_of(v["id"], int(frame))
            img = os.path.join(valdir, "images", f"{stem}.{ext}")
            if not os.path.exists(img):
                continue
            if not revisado:  # una imagen por video: su tamaño debe ser el del manifiesto (con él se normalizan las cajas)
                probe = cv2.imread(img)
                if probe is None or (probe.shape[1], probe.shape[0]) != (int(v["ancho"]), int(v["alto"])):
                    got = None if probe is None else f"{probe.shape[1]}x{probe.shape[0]}"
                    raise ValueError(f"La imagen {os.path.basename(img)} mide {got}, pero el manifiesto dice {v['ancho']}x{v['alto']}.")
                revisado = True
            items.append(Item(stem, img, os.path.join(valdir, "labels", stem + ".txt"), v["id"], int(frame), int(v["ancho"]), int(v["alto"])))
    return items


def _atomic_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


class Labeler:
    """Estado y lógica del etiquetador. No abre ventanas: recibe eventos (``on_mouse``, ``on_key``) y dibuja con ``render``."""

    def __init__(self, valdir: str, items: List[Item], max_w: int = 1280, max_h: int = 800, start_at_pending: bool = True):
        if not items:
            raise ValueError("No hay imágenes que etiquetar.")
        self.valdir, self.items = valdir, items
        self.max_w, self.max_h = max_w, max_h
        self.doubtful_path = os.path.join(valdir, "dudosas.txt")
        self.states: List[Tuple[str, Optional[Box]]] = []
        self.notes: Dict[int, str] = {}  # etiquetas existentes que no se pudieron leer
        self.msg = ""
        self.show_help = False
        self._drag: Optional[Tuple[float, float]] = None
        self._cursor: Optional[Tuple[float, float]] = None
        self._cache: Optional[Tuple[int, np.ndarray, float]] = None  # (índice, imagen base ya escalada, escala)
        self._doubtful = self._read_doubtful()
        self._load_states()
        self.i = 0
        if start_at_pending:
            pend = self._next_pending(-1)
            self.i = pend if pend is not None else 0

    # ---------------------------------------------------------------- carga
    def _read_doubtful(self) -> set:
        if not os.path.exists(self.doubtful_path):
            return set()
        with open(self.doubtful_path, encoding="utf-8") as fh:
            return {ln.strip() for ln in fh if ln.strip()}

    def _load_states(self) -> None:
        for k, it in enumerate(self.items):
            state: Tuple[str, Optional[Box]] = (PENDING, None)
            if os.path.exists(it.label_path):
                with open(it.label_path, encoding="utf-8-sig") as fh:
                    lines = [ln for ln in fh.read().splitlines() if ln.strip()]
                if not lines:
                    state = (EMPTY, None)
                elif len(lines) == 1:
                    try:
                        box = parse_yolo_label_line(lines[0], it.width, it.height)[1]
                        state = (BOX, tuple(round(v, 3) for v in box))
                    except ValueError:
                        self.notes[k] = "la etiqueta anterior es invalida: se reemplaza al dibujar la caja"
                else:
                    self.notes[k] = "la etiqueta anterior tiene varias cajas: se reemplaza al dibujar la caja"
            elif it.stem in self._doubtful:
                state = (DOUBTFUL, None)
            self.states.append(state)

    # ----------------------------------------------------------- consultas
    @property
    def item(self) -> Item:
        return self.items[self.i]

    @property
    def state(self) -> Tuple[str, Optional[Box]]:
        return self.states[self.i]

    def counts(self) -> Dict[str, int]:
        out = {PENDING: 0, BOX: 0, EMPTY: 0, DOUBTFUL: 0}
        for s, _ in self.states:
            out[s] += 1
        out["total"] = len(self.states)
        return out

    def _next_pending(self, after: int) -> Optional[int]:
        n = len(self.states)
        for off in range(1, n + 1):
            k = (after + off) % n
            if self.states[k][0] == PENDING:
                return k
        return None

    # --------------------------------------------------------------- guardar
    def _write_doubtful(self) -> None:
        _atomic_write(self.doubtful_path, "".join(f"{s}\n" for s in sorted(self._doubtful)))

    def _set_state(self, status: str, box: Optional[Box] = None) -> None:
        """Cambia el estado de la imagen actual y lo escribe a disco de inmediato."""
        it, k = self.item, self.i
        if status == BOX:
            _atomic_write(it.label_path, yolo_label_line(box, it.width, it.height) + "\n")
        elif status == EMPTY:
            _atomic_write(it.label_path, "")
        else:  # PENDING o DOUBTFUL: no debe quedar archivo de etiqueta
            if os.path.exists(it.label_path):
                os.remove(it.label_path)
        if status == DOUBTFUL:
            self._doubtful.add(it.stem)
            self._write_doubtful()
        elif it.stem in self._doubtful:
            self._doubtful.discard(it.stem)
            self._write_doubtful()
        self.states[k] = (status, box if status == BOX else None)
        self.notes.pop(k, None)

    # ------------------------------------------------------------- navegar
    def goto(self, k: int) -> None:
        self.i = max(0, min(len(self.items) - 1, k))
        self._drag = None

    def _advance(self) -> None:
        if self.i + 1 < len(self.items):
            self.goto(self.i + 1)
        else:
            self.msg = "es la ultima imagen"

    # ------------------------------------------------------------- escala
    def _scale_for(self, w: int, h: int) -> float:
        return min(1.0, self.max_w / w, self.max_h / h)

    def _to_image(self, x: float, y: float) -> Tuple[float, float]:
        """De coordenadas de la ventana (con la barra de arriba) a coordenadas de la imagen original."""
        s = self._scale_for(self.item.width, self.item.height)
        return x / s, (y - BAR_H) / s

    # -------------------------------------------------------------- eventos
    @staticmethod
    def _normalize(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> Optional[Box]:
        """Caja (x, y, ancho, alto) en píxeles enteros, recortada a la imagen. None si es demasiado pequeña."""
        ax, bx = sorted((min(max(x0, 0.0), w), min(max(x1, 0.0), w)))
        ay, by = sorted((min(max(y0, 0.0), h), min(max(y1, 0.0), h)))
        x, y, x2, y2 = int(round(ax)), int(round(ay)), int(round(bx)), int(round(by))
        if x2 - x < MIN_SIDE or y2 - y < MIN_SIDE:
            return None
        return (float(x), float(y), float(x2 - x), float(y2 - y))

    def on_mouse(self, event: int, x: float, y: float, flags: int = 0) -> None:
        px, py = self._to_image(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag = (px, py)
            self._cursor = (px, py)
        elif event == cv2.EVENT_MOUSEMOVE:
            self._cursor = (px, py)
        elif event == cv2.EVENT_LBUTTONUP and self._drag is not None:
            x0, y0 = self._drag
            self._drag = None
            box = self._normalize(x0, y0, px, py, self.item.width, self.item.height)
            if box is None:
                self.msg = "caja demasiado pequena: se ignora"
                return
            self._set_state(BOX, box)
            self.msg = "caja guardada"

    def on_key(self, key: int) -> Optional[str]:
        """Procesa una tecla. Devuelve ``"quit"`` para salir."""
        if key == 27:
            return "quit"
        if key == 32:
            ch = "d"
        else:
            try:
                ch = chr(key).lower()
            except (ValueError, OverflowError):
                return None
        self.msg = ""
        if ch == "q":
            return "quit"
        if ch == "d":
            self._advance()
        elif ch == "a":
            if self.i > 0:
                self.goto(self.i - 1)
            else:
                self.msg = "es la primera imagen"
        elif ch == "x":
            self._set_state(EMPTY)
            self.msg = "sin raton guardado"
            self._advance()
        elif ch == "u":
            self._set_state(PENDING)
            self.msg = "etiqueta borrada"
        elif ch == "k":
            self._set_state(DOUBTFUL)
            self.msg = "marcada como dudosa"
            self._advance()
        elif ch == "n":
            k = self._next_pending(self.i)
            if k is None:
                self.msg = "no quedan imagenes pendientes"
            else:
                self.goto(k)
        elif ch == "h":
            self.show_help = not self.show_help
        return None

    # --------------------------------------------------------------- dibujo
    def _base(self) -> Tuple[np.ndarray, float]:
        if self._cache is not None and self._cache[0] == self.i:
            return self._cache[1], self._cache[2]
        img = cv2.imread(self.item.image_path)
        if img is None:
            raise OSError(f"No se pudo leer {self.item.image_path}")
        s = self._scale_for(img.shape[1], img.shape[0])
        if s != 1.0:
            img = cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        self._cache = (self.i, img, s)
        return img, s

    def render(self) -> np.ndarray:
        """Ventana completa: barra de estado, imagen (con la caja) y barra de ayuda. Las barras NO tapan la imagen."""
        base, s = self._base()
        disp = base.copy()
        status, box = self.state
        if self._drag is not None and self._cursor is not None:  # caja en construcción
            live = self._normalize(self._drag[0], self._drag[1], self._cursor[0], self._cursor[1], self.item.width, self.item.height)
            if live is not None:
                self._rect(disp, live, s, (0, 220, 255))
        elif status == BOX and box is not None:
            self._rect(disp, box, s, (0, 200, 0))
        if self.show_help:
            cv2.rectangle(disp, (0, 0), (min(disp.shape[1], 560), 22 * len(HELP) + 8), (20, 20, 20), -1)
            for j, ln in enumerate(HELP):
                cv2.putText(disp, ln, (8, 20 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        c = self.counts()
        done = c[BOX] + c[EMPTY]
        etiqueta = {PENDING: "PENDIENTE", BOX: "CAJA", EMPTY: "SIN RATON", DOUBTFUL: "DUDOSA"}[status]
        w = disp.shape[1]
        top = np.full((BAR_H, w, 3), 30, np.uint8)
        cv2.putText(top, f"{self.i + 1}/{c['total']}  {self.item.stem}  [{etiqueta}]  hechas {done}/{c['total']}",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        aviso = self.msg or self.notes.get(self.i, "")
        bottom = np.full((BAR_H, w, 3), 30, np.uint8)
        cv2.putText(bottom, aviso or "arrastra=caja x=sin raton d=sig a=ant u=borrar k=dudosa n=pend h=ayuda q=salir",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200) if aviso else (210, 210, 210), 1, cv2.LINE_AA)
        return np.vstack([top, disp, bottom])

    @staticmethod
    def _rect(img: np.ndarray, box: Box, scale: float, color: Tuple[int, int, int]) -> None:
        x, y, w, h = box
        cv2.rectangle(img, (int(round(x * scale)), int(round(y * scale))), (int(round((x + w) * scale)), int(round((y + h) * scale))), color, 2)


# ----------------------------------------------------------------------
# Ventana
# ----------------------------------------------------------------------
def run_gui(lab: Labeler, window: str = WINDOW) -> None:
    """Bucle de la ventana: dibuja, lee teclas y manda los eventos a ``lab``. Todo se guarda al instante."""
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, lambda event, x, y, flags, param: lab.on_mouse(event, x, y, flags))
    try:
        while True:
            cv2.imshow(window, lab.render())
            key = cv2.waitKey(20) & 0xFF
            if key != 255 and lab.on_key(key) == "quit":
                break
    finally:
        cv2.destroyAllWindows()


def format_summary(c: Dict[str, int]) -> str:
    return (f"{c[BOX]} con caja, {c[EMPTY]} sin ratón, {c[DOUBTFUL]} dudosas, {c[PENDING]} pendientes "
            f"(de {c['total']} imágenes)")


def main(argv: Optional[List[str]] = None, gui: Callable[[Labeler], None] = run_gui, out: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(description="Etiquetador sencillo para el conjunto de validación del ratón.")
    ap.add_argument("validacion", help="carpeta creada por tools/sample_frames.py")
    ap.add_argument("--solo-resumen", action="store_true", help="muestra cuántas van, sin abrir ventana")
    ap.add_argument("--desde-inicio", action="store_true", help="empezar en la primera imagen (por omisión, en la primera pendiente)")
    ap.add_argument("--ir-a", metavar="FRAME", help="empezar en un frame concreto, p. ej. 2757 o Grupo1R1S2:2757 (sirve para corregir lo que marcó audit_labels.py)")
    ap.add_argument("--ancho-max", type=int, default=1280, help="ancho máximo de la ventana en píxeles (por omisión 1280)")
    ap.add_argument("--alto-max", type=int, default=800, help="alto máximo de la ventana en píxeles (por omisión 800)")
    a = ap.parse_args(argv)
    try:
        items = load_items(a.validacion)
        if not items:
            out(f"No hay imágenes en {os.path.join(a.validacion, 'images')}.")
            return 2
        lab = Labeler(a.validacion, items, a.ancho_max, a.alto_max, start_at_pending=not a.desde_inicio)
    except (FileNotFoundError, ValueError, OSError) as exc:
        out(str(exc))
        return 2
    if a.ir_a:
        hits = find_items(items, a.ir_a)
        if not hits:
            muestra = ", ".join(f"{it.video}:{it.frame}" for it in items[:12]) + (" ..." if len(items) > 12 else "")
            out(f"No hay ninguna imagen «{a.ir_a}» en el conjunto. Algunas que sí hay: {muestra}")
            return 2
        if len(hits) > 1:
            out(f"«{a.ir_a}» es ambiguo: está en varios videos ({', '.join(items[k].video for k in hits)}). Usa VIDEO:FRAME, por ejemplo {items[hits[0]].video}:{items[hits[0]].frame}.")
            return 2
        lab.goto(hits[0])
    out("Estado: " + format_summary(lab.counts()))
    if a.solo_resumen:
        return 0
    out("Abriendo la ventana. Pulsa h para ver la ayuda y q para salir.")
    gui(lab)
    out("Estado final: " + format_summary(lab.counts()))
    if lab.counts()[PENDING] == 0:
        out("Listo: todas las imágenes tienen una decisión. Lo que sigue depende de para qué sea este conjunto:")
        out(f'  · en cualquier caso, comprueba el alineamiento: python tools/check_frame_alignment.py "{a.validacion}"')
        out(f'  · para ENTRENAR el detector: audita la consistencia con python tools/audit_labels.py "{a.validacion}" y luego arma el conjunto con tools/build_training_set.py (sección 11 de la guía)')
        out(f'  · para VALIDAR el seguimiento clásico: python tools/eval_tracking.py "{a.validacion}" --entradas RUTA_DE_LA_CARPETA_entradas_DE_ESE_VIDEO (sustitúyela por la real)')
    else:
        out(f'Faltan {lab.counts()[PENDING]}. Vuelve a correr: python tools/label_frames.py "{a.validacion}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())