#!/usr/bin/env python3
"""Comprueba si las imágenes de ``validacion/`` son EXACTAMENTE los frames que procesa el seguimiento.

Uso (desde la raíz del proyecto):
    python tools/check_frame_alignment.py validacion
    python tools/check_frame_alignment.py validacion --ventana 12
    python tools/check_frame_alignment.py validacion --entradas "<carpeta entradas_...>"   # como lo leyó ESA corrida

Por qué existe: la evaluación compara lo que dibujaste en una imagen con lo que el seguimiento dijo en el frame
con el MISMO número. Si la imagen es de otro instante (por ejemplo, porque al sacarla se hizo un salto en un mp4 y
cayó unos frames fuera de lugar), el animal se movió entre los dos y la evaluación mediría ese desfase, no el
seguimiento: cajas del mismo tamaño pero corridas a lo largo de la trayectoria.

Cómo funciona: el video se lee de corrido DESDE EL INICIO DEL RANGO de la corrida (como el pipeline, para que los
números de frame sean los mismos) y, para cada imagen, se mira cuál de los frames cercanos al esperado se parece más. Si es el esperado, el desfase es 0. Si no, el desfase ``k`` indica que la imagen se parece
al frame ``N + k``. Si hay etiqueta, la comparación se hace alrededor del ratón, donde está el movimiento.

Un desfase positivo significa que la imagen es POSTERIOR al frame que vio el seguimiento (el seguimiento va «detrás»
del ratón que tú dibujaste).

Con ``--entradas`` se lee el video EXACTAMENTE como lo leyó esa corrida (desde el inicio de SU rango, con el mismo lector) y
solo se revisan las imágenes dentro de su rango. Hace falta cuando el conjunto se muestreó por su cuenta (por ejemplo, directo
del video) y la corrida empezó en otro frame: si el lector, al saltar a ese inicio, cayó unos frames corrido, todo lo que la
corrida llama «frame N» es otro instante, y la evaluación de esa corrida contra estas etiquetas mediría el desfase.

Escribe ``alineacion.json`` en la carpeta del conjunto (``alineacion_<corrida>.json`` con ``--entradas``). Código de salida: 0 = se pudo revisar, 2 = no se pudo.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.dataset import parse_yolo_label_line  # noqa: E402
from video.reader import VideoOpenError, VideoReader  # noqa: E402

Box = Tuple[float, float, float, float]
CLEAR_GAP = 1.0  # niveles de gris: cuánto debe destacar el mejor candidato sobre la mediana para fiarse de él


def stem_of(video_id: str, frame: int) -> str:
    return f"{video_id}_f{frame:06d}"


def read_label_box(path: str, w: int, h: int) -> Optional[Box]:
    """Caja etiquetada (px) o None si no hay archivo, está vacío o no se puede leer."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8-sig") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    if len(lines) != 1:
        return None
    try:
        return parse_yolo_label_line(lines[0], w, h)[1]
    except ValueError:
        return None


def crop_region(box: Optional[Box], w: int, h: int, factor: float = 2.0, min_margin: int = 60) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) alrededor de la caja con un margen generoso; la imagen entera si no hay caja."""
    if box is None:
        return 0, 0, w, h
    x, y, bw, bh = box
    m = max(float(min_margin), factor * max(bw, bh))
    return int(max(0, x - m)), int(max(0, y - m)), int(min(w, x + bw + m)), int(min(h, y + bh + m))


def best_offset(maes: Dict[int, float]) -> Dict[str, Any]:
    """Del diccionario {desfase: error medio} elige el mejor. ``clara`` es False si nada destaca (escena sin movimiento)."""
    if not maes:
        return {"desfase": None, "mae_mejor": None, "mae_cero": None, "clara": False}
    mejor = min(maes, key=lambda d: (maes[d], abs(d)))  # a igualdad, el más cercano a 0
    valores = list(maes.values())
    clara = len(valores) >= 3 and (statistics.median(valores) - maes[mejor]) >= CLEAR_GAP
    return {"desfase": int(mejor), "mae_mejor": float(maes[mejor]), "mae_cero": float(maes[0]) if 0 in maes else None, "clara": bool(clara)}


def summarize(rows: List[Dict[str, Any]], tolerance: float = 0.9) -> Dict[str, Any]:
    claras = [r for r in rows if r["clara"]]
    cuenta: Dict[int, int] = {}
    for r in claras:
        cuenta[r["desfase"]] = cuenta.get(r["desfase"], 0) + 1
    alineadas = cuenta.get(0, 0)
    out: Dict[str, Any] = {"imagenes": len(rows), "determinadas": len(claras), "ambiguas": len(rows) - len(claras),
                           "alineadas": alineadas, "desfases": dict(sorted(cuenta.items()))}
    if not claras:
        out["veredicto"] = "no_determinado"
    elif alineadas / len(claras) >= tolerance:
        out["veredicto"] = "alineado"
    else:
        out["veredicto"] = "desalineado"
        out["desfase_mediano"] = float(statistics.median(r["desfase"] for r in claras))
    return out


def read_run(entradas: str) -> Dict[str, Any]:
    """Rango y video de una corrida, de su ``inputs.json``. Lanza OSError/ValueError/KeyError si no se puede leer."""
    with open(os.path.join(entradas, "inputs.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    v = d.get("video") or {}
    return {"inicio": int(d["frame_range"]["start"]), "fin": int(d["frame_range"]["end"]), "huella": v.get("sha256_extremos"),
            "nombre": os.path.basename(str(v.get("path", ""))), "carpeta": os.path.basename(os.path.normpath(entradas))}


def same_video(entry: Dict[str, Any], corrida: Dict[str, Any]) -> bool:
    """La corrida es de este video del conjunto: por huella si las dos la tienen; si no, por nombre de archivo."""
    if entry.get("sha256_extremos") and corrida.get("huella"):
        return entry["sha256_extremos"] == corrida["huella"]
    return bool(corrida.get("nombre")) and entry.get("nombre") == corrida["nombre"]


def analyze_video(valdir: str, entry: Dict[str, Any], path: str, window: int,
                  run_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
    """Desfase de cada imagen de un video. Lee de corrido, una sola vez, de la primera a la última imagen."""
    ext = entry.get("formato", "jpg")
    w, h = int(entry["ancho"]), int(entry["alto"])
    refs: Dict[int, Tuple[np.ndarray, Tuple[int, int, int, int]]] = {}
    for f in sorted(int(x) for x in entry["frames"]):
        stem = stem_of(entry["id"], f)
        img = cv2.imread(os.path.join(valdir, "images", f"{stem}.{ext}"), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (h, w):
            continue
        region = crop_region(read_label_box(os.path.join(valdir, "labels", stem + ".txt"), w, h), w, h)
        x0, y0, x1, y1 = region
        refs[f] = (img[y0:y1, x0:x1].astype(np.float32), region)
    if not refs:
        return []
    frames = sorted(refs)
    if run_range is not None:  # solo lo que esa corrida procesó
        frames = [f for f in frames if run_range[0] <= f <= run_range[1]]
        if not frames:
            return []
    maes: Dict[int, Dict[int, float]] = {f: {} for f in frames}
    # Los números de frame deben ser LOS DEL PIPELINE: se lee desde el mismo punto de partida que la corrida (el inicio de su
    # rango, que el manifiesto guarda). Si el video se posiciona con un salto inicial impreciso, todos los índices de esa
    # lectura quedan corridos por igual; empezar en otro sitio mediría contra una numeración distinta a la del seguimiento.
    if run_range is not None:
        start = run_range[0]  # EXACTAMENTE donde empezó la corrida: el mismo salto inicial, con su posible imprecisión
    else:
        start = int(entry["rango"][0]) if entry.get("rango") else frames[0] - window
        start = max(0, min(start, frames[0]))
    with VideoReader(path) as rd:
        for idx, fr in rd.iter_frames(start, frames[-1] + window, 1):
            gray: Optional[np.ndarray] = None
            for f in frames:
                d = idx - f
                if abs(d) > window:
                    continue
                if gray is None:
                    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                ref, (x0, y0, x1, y1) = refs[f]
                maes[f][d] = float(np.mean(np.abs(gray[y0:y1, x0:x1].astype(np.float32) - ref)))
    return [{"video": entry["id"], "frame": f, **best_offset(maes[f])} for f in frames]


def run(valdir: str, window: int = 8, video: Optional[str] = None, out: Callable[[str], None] = print, entradas: Optional[str] = None) -> int:
    path = os.path.join(valdir, "manifest.json")
    if not os.path.exists(path):
        out(f"No existe {path}: genera el conjunto con tools/sample_frames.py.")
        return 2
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    corrida: Optional[Dict[str, Any]] = None
    if entradas:
        try:
            corrida = read_run(entradas)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            out(f"No se pudieron leer las entradas {entradas}: {exc}")
            return 2
        out(f"Se lee el video como la corrida {corrida['carpeta']}: desde su frame {corrida['inicio']} (el mismo salto inicial que el seguimiento).")
    rows: List[Dict[str, Any]] = []
    for v in manifest["videos"]:
        rango: Optional[Tuple[int, int]] = None
        if corrida is not None:
            if not same_video(v, corrida):
                out(f"AVISO: la corrida no es de {v['nombre']}; sus imágenes no se revisan.")
                continue
            rango = (corrida["inicio"], corrida["fin"])
            fuera = [f for f in v["frames"] if not (rango[0] <= int(f) <= rango[1])]
            if fuera:
                out(f"{len(fuera)} imágenes de {v['nombre']} quedan fuera del rango de la corrida ({rango[0]}-{rango[1]}); no se revisan.")
        ruta = video or v["path"]
        if not os.path.exists(ruta):
            out(f"AVISO: no se encuentra el video {v['nombre']}; se omite.")
            continue
        try:
            rows += analyze_video(valdir, v, ruta, window, rango)
        except (VideoOpenError, OSError) as exc:
            out(f"AVISO: no se pudo leer {v['nombre']}: {exc}")
    if not rows:
        out("No se pudo revisar ninguna imagen.")
        return 2

    out("imagen                         desfase  (error con el frame esperado / con el mejor)")
    for r in rows:
        if r["desfase"] is None:
            continue
        marca = "" if r["clara"] else "   (ambigua: la escena casi no cambia)"
        cero = "-" if r["mae_cero"] is None else f"{r['mae_cero']:.1f}"
        out(f"{r['video'][:18]:<18} f{r['frame']:<8} {r['desfase']:+4d}     ({cero} / {r['mae_mejor']:.1f}){marca}")
    s = summarize(rows)
    out("")
    if s["veredicto"] == "alineado":
        out(f"ALINEADO: {s['alineadas']} de {s['determinadas']} imágenes determinadas coinciden con el frame que procesa el seguimiento.")
    elif s["veredicto"] == "desalineado":
        out(f"ATENCIÓN: solo {s['alineadas']} de {s['determinadas']} imágenes coinciden con el frame esperado. Desfases: {s['desfases']}.")
        out(f"  Desfase mediano: {s['desfase_mediano']:+g} frames. Positivo = la imagen es posterior al frame que vio el seguimiento.")
        if corrida is not None:
            out("  Esa corrida vio frames corridos respecto a estas imágenes: su evaluación contra este conjunto mide el desfase, no el seguimiento.")
        else:
            out("  La evaluación mide ese desfase, no el seguimiento. Vuelve a generar las imágenes con tools/sample_frames.py actualizado.")
    else:
        out("NO SE PUDO DETERMINAR: en las imágenes revisadas la escena casi no cambia entre frames vecinos.")
    if s["ambiguas"] and s["veredicto"] != "no_determinado":
        out(f"({s['ambiguas']} imágenes ambiguas no cuentan)")
    nombre = "alineacion.json" if corrida is None else f"alineacion_{corrida['carpeta']}.json"
    with open(os.path.join(valdir, nombre), "w", encoding="utf-8") as fh:
        json.dump({"ventana": window, "entradas": entradas, "resumen": s, "imagenes": rows}, fh, ensure_ascii=False, indent=2)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Comprueba que las imágenes de validación son los frames que procesa el seguimiento.")
    ap.add_argument("validacion", help="carpeta creada por tools/sample_frames.py")
    ap.add_argument("--ventana", type=int, default=8, help="cuántos frames a cada lado se buscan (por omisión 8)")
    ap.add_argument("--video", help="ruta del video, si cambió de lugar (solo para conjuntos de un video)")
    ap.add_argument("--entradas", help="carpeta entradas_... de una corrida: leer el video como lo leyó ella (desde el inicio de su rango)")
    a = ap.parse_args(argv)
    return run(a.validacion, a.ventana, a.video, entradas=a.entradas)


if __name__ == "__main__":
    sys.exit(main())