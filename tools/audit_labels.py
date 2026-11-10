from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

COL_CAJA, COL_ALERTA, COL_TEXTO = (0, 255, 0), (0, 0, 255), (255, 255, 255)  # BGR
ORDENES = ("area", "frame", "aspecto")


@dataclass
class Item:
    video: str
    id: str
    frame: int
    imagen: str
    caja: Tuple[float, float, float, float]  # x, y, ancho, alto en píxeles del frame completo
    ancho: int
    alto: int

    @property
    def area(self) -> float:
        return self.caja[2] * self.caja[3]

    @property
    def aspecto(self) -> float:
        return self.caja[2] / self.caja[3]


# ----------------------------------------------------------------------
# Lectura
# ----------------------------------------------------------------------
def collect(directorio: str) -> Tuple[List[Item], Dict[str, int]]:
    """Cajas de un conjunto hecho con tools/sample_frames.py y etiquetado con tools/label_frames.py, y el conteo por clase de etiqueta."""
    p = os.path.join(directorio, "manifest.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{directorio} no es un conjunto de frames etiquetados a mano: falta manifest.json.")
    with open(p, encoding="utf-8") as fh:
        man = json.load(fh)
    items: List[Item] = []
    cuenta = {"con_caja": 0, "sin_raton": 0, "varias_cajas": 0, "invalidas": 0, "sin_etiqueta": 0}
    for v in man.get("videos", []):
        ext = v.get("formato", "jpg")
        for f in sorted(int(x) for x in v.get("frames", [])):
            stem = f"{v['id']}_f{f:06d}"
            lab = os.path.join(directorio, "labels", stem + ".txt")
            if not os.path.exists(lab):
                cuenta["sin_etiqueta"] += 1
                continue
            with open(lab, encoding="utf-8-sig") as fh:
                lineas = [ln for ln in fh.read().splitlines() if ln.strip()]
            if not lineas:
                cuenta["sin_raton"] += 1
                continue
            if len(lineas) > 1:
                cuenta["varias_cajas"] += 1
                continue
            try:
                c, cx, cy, w, h = lineas[0].split()
                cx, cy, w, h = float(cx), float(cy), float(w), float(h)
                if c != "0" or not (w > 0 and h > 0):
                    raise ValueError
            except ValueError:
                cuenta["invalidas"] += 1
                continue
            W, H = int(v["ancho"]), int(v["alto"])
            bw, bh = w * W, h * H
            items.append(Item(v.get("nombre", v["id"]), v["id"], f, os.path.join(directorio, "images", f"{stem}.{ext}"),
                              (cx * W - bw / 2, cy * H - bh / 2, bw, bh), W, H))
            cuenta["con_caja"] += 1
    return items, cuenta


# ----------------------------------------------------------------------
# Estadística y orden
# ----------------------------------------------------------------------
def _cuartiles(vals: Sequence[float]) -> Tuple[float, float]:
    s = sorted(vals)
    n = len(s)

    def q(p: float) -> float:
        pos = p * (n - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)

    return q(0.25), q(0.75)


def tukey_bounds(vals: Sequence[float], k: float = 1.5) -> Tuple[float, float]:
    """Límites (inferior, superior) de lo habitual: Q1 - k*IQR y Q3 + k*IQR."""
    q1, q3 = _cuartiles(vals)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def sort_items(items: Sequence[Item], orden: str) -> List[Item]:
    if orden == "area":
        return sorted(items, key=lambda i: (i.area, i.frame))
    if orden == "frame":
        return sorted(items, key=lambda i: (i.video, i.frame))
    if orden == "aspecto":
        return sorted(items, key=lambda i: (i.aspecto, i.frame))
    raise ValueError(f"--orden debe ser uno de {ORDENES}.")


def flag_outliers(items: Sequence[Item], k: float = 1.5) -> List[Item]:
    """Cajas con un área fuera de lo habitual. Con menos de 8 cajas no se marca nada (no hay base para decir qué es habitual)."""
    if len(items) < 8:
        return []
    lo, hi = tukey_bounds([i.area for i in items], k)
    return [i for i in items if i.area < lo or i.area > hi]


# ----------------------------------------------------------------------
# Dibujo
# ----------------------------------------------------------------------
def extract(img: np.ndarray, cx: float, cy: float, lado: int) -> Tuple[np.ndarray, int, int]:
    """Recorte cuadrado de ``lado`` px centrado en (cx, cy) y la esquina (x0, y0) que corresponde en la imagen original.

    La ventana se DESPLAZA para no salirse de la imagen (así todos los recortes conservan la misma escala); si la imagen es
    más chica que la ventana, el sobrante se rellena de negro.
    """
    h, w = img.shape[:2]
    x0 = int(round(cx - lado / 2))
    y0 = int(round(cy - lado / 2))
    x0 = max(0, min(x0, w - lado)) if w >= lado else 0
    y0 = max(0, min(y0, h - lado)) if h >= lado else 0
    out = np.zeros((lado, lado, 3), np.uint8)
    sub = img[y0:y0 + lado, x0:x0 + lado]
    out[:sub.shape[0], :sub.shape[1]] = sub
    return out, x0, y0


def render(items: Sequence[Item], lado: int, columnas: int, celda: int, alertas: Sequence[Item] = ()) -> np.ndarray:
    """Hoja de contactos: una celda de ``celda`` px por caja, todas con la misma ventana de ``lado`` px del frame."""
    if not items:
        raise ValueError("No hay cajas que dibujar.")
    marcadas = {id(a) for a in alertas}
    filas = (len(items) + columnas - 1) // columnas
    hoja = np.full((filas * celda, columnas * celda, 3), 25, np.uint8)
    esc = celda / lado
    for k, it in enumerate(items):
        r, c = divmod(k, columnas)
        img = cv2.imread(it.imagen)
        if img is None:
            continue
        x, y, w, h = it.caja
        crop, x0, y0 = extract(img, x + w / 2, y + h / 2, lado)
        crop = cv2.resize(crop, (celda, celda), interpolation=cv2.INTER_AREA)
        color = COL_ALERTA if id(it) in marcadas else COL_CAJA
        cv2.rectangle(crop, (int(round((x - x0) * esc)), int(round((y - y0) * esc))),
                      (int(round((x + w - x0) * esc)), int(round((y + h - y0) * esc))), color, 2)
        cv2.putText(crop, f"{it.id[:9]} f{it.frame}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_TEXTO, 1, cv2.LINE_AA)
        cv2.putText(crop, f"{w:.0f}x{h:.0f}" + ("  !" if id(it) in marcadas else ""), (4, celda - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    COL_ALERTA if id(it) in marcadas else COL_TEXTO, 1, cv2.LINE_AA)
        hoja[r * celda:(r + 1) * celda, c * celda:(c + 1) * celda] = crop
        cv2.rectangle(hoja, (c * celda, r * celda), ((c + 1) * celda - 1, (r + 1) * celda - 1), (60, 60, 60), 1)
    return hoja


# ----------------------------------------------------------------------
def audit(directorio: str, salida: Optional[str] = None, *, orden: str = "area", columnas: int = 8, celda: int = 200,
          contexto: float = 1.5, out: Callable[[str], None] = print) -> int:
    if orden not in ORDENES or columnas < 1 or celda < 60 or contexto < 1.0:
        out(f"Argumentos inválidos: --orden {ORDENES}, --columnas >= 1, --celda >= 60, --contexto >= 1.")
        return 2
    try:
        items, cuenta = collect(directorio)
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        out(str(exc))
        return 2
    total = sum(cuenta.values())
    out(f"{total} imágenes: {cuenta['con_caja']} con caja, {cuenta['sin_raton']} sin ratón, {cuenta['varias_cajas']} con varias cajas, "
        f"{cuenta['invalidas']} inválidas, {cuenta['sin_etiqueta']} sin etiqueta (dudosas o pendientes).")
    if not items:
        out("No hay ninguna caja que auditar.")
        return 2
    areas = [i.area for i in items]
    out(f"Ancho mediano {statistics.median(i.caja[2] for i in items):.0f} px, alto mediano {statistics.median(i.caja[3] for i in items):.0f} px.")
    out(f"Área: mínima {min(areas):.0f}, mediana {statistics.median(areas):.0f}, máxima {max(areas):.0f} px² (máx/mín = {max(areas) / min(areas):.1f}).")
    alertas = flag_outliers(items)
    if len(items) < 8:
        out("AVISO: hay menos de 8 cajas: no se marcan posibles inconsistentes (no hay base para decir qué es habitual).")
    elif alertas:
        lo, hi = tukey_bounds(areas)
        out(f"Posibles inconsistentes (área fuera de {max(lo, 0):.0f} a {hi:.0f} px²): "
            + ", ".join(f"{a.video} f{a.frame} ({a.caja[2]:.0f}x{a.caja[3]:.0f})" for a in sort_items(alertas, 'area')))
    else:
        out("Ninguna caja sale de lo habitual por su área.")
    lado = int(math.ceil(max(max(i.caja[2], i.caja[3]) for i in items) * contexto))
    hoja = render(sort_items(items, orden), lado, columnas, celda, alertas)
    ruta = salida or os.path.join(directorio, "auditoria.png")
    if not cv2.imwrite(ruta, hoja):
        out(f"No se pudo escribir {ruta}.")
        return 2
    out(f"\nHoja ({orden}, ventana de {lado} px por celda; borde rojo = posible inconsistente): {ruta}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Hoja de auditoría de las cajas etiquetadas a mano (misma escala, ordenadas).")
    ap.add_argument("directorio", help="carpeta hecha con sample_frames.py y etiquetada con label_frames.py")
    ap.add_argument("--orden", default="area", choices=ORDENES, help="area (por omisión), frame o aspecto (ancho/alto)")
    ap.add_argument("--columnas", type=int, default=8)
    ap.add_argument("--celda", type=int, default=200, help="píxeles de cada celda de la hoja")
    ap.add_argument("--contexto", type=float, default=1.5, help="la ventana mide esto por el lado mayor de la caja más grande (por omisión 1.5)")
    ap.add_argument("--salida", help="ruta del PNG (por omisión, auditoria.png dentro de la carpeta)")
    a = ap.parse_args(argv)
    return audit(a.directorio, a.salida, orden=a.orden, columnas=a.columnas, celda=a.celda, contexto=a.contexto)


if __name__ == "__main__":
    sys.exit(main())