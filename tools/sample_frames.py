from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.dataset import video_fingerprint  # noqa: E402
from persistence.run_inputs import load_run_inputs  # noqa: E402
from video.reader import VideoOpenError, VideoReader  # noqa: E402

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv")
SCHEMA = 1

LEEME = """CÓMO ETIQUETAR (conjunto de validación del ratón)
====================================================

Clase única: raton (clase 0). Formato: YOLO (un .txt por imagen, con el MISMO nombre que la imagen).

1. Una sola caja por imagen, AJUSTADA AL CUERPO VISIBLE del ratón, SIN la cola. Incluye orejas y patas visibles.
   Se mide con la caja y con el centro; usar siempre la misma convención es lo que importa.

2. Si el ratón NO está visible o está tapado casi por completo: no dibujes nada. La etiqueta debe existir y
   estar VACÍA. Estas imágenes sirven para medir los falsos positivos (que el seguimiento "vea" un ratón donde
   no hay). Si tu herramienta no crea el .txt vacío, créalo a mano.
   Un archivo de etiqueta que NO existe significa "esta imagen no se etiquetó" y se omite en la evaluación.

3. No uses cajas sugeridas por TrackerLab ni por ningún modelo: tienden a aceptarse sin mirar y falsean la
   medición. Etiqueta mirando solo la imagen.

4. Si no estás seguro de si algo es el ratón, no adivines: déjalo vacío y anótalo aparte.

5. Los videos de este conjunto idealmente NO se usan para entrenar: así se mide cómo funciona con videos nuevos.

Herramientas de anotación que exportan YOLO: CVAT, Label Studio (versión de código abierto).
Al terminar: python tools/eval_tracking.py <esta carpeta> --entradas "<carpeta entradas_...>"
"""


# Piezas puras (se prueban sin videos)
def safe_stem(name: str) -> str:
    """Nombre de archivo seguro: letras, números, guion y guion bajo (lo demás pasa a «_»)."""
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r"[^0-9A-Za-z_-]+", "_", stem).strip("_")
    return stem or "video"


def plan_frames(start: int, end: int, step: int, n: int, rng: random.Random) -> List[int]:
    """Frames a muestrear: alineados a ``start + k*step`` y estratificados (uno al azar por tramo).

    Si hay menos candidatos que ``n``, se devuelven todos.
    """
    if step < 1:
        raise ValueError("step debe ser >= 1")
    if n < 1:
        raise ValueError("n debe ser >= 1")
    if end < start:
        return []
    cand = range(start, end + 1, step)
    m = len(cand)
    if n >= m:
        return list(cand)
    picks = []
    for i in range(n):
        a, b = i * m // n, (i + 1) * m // n  # tramo [a, b): con n < m siempre tiene al menos un candidato
        picks.append(cand[rng.randrange(a, b)])
    return picks


def image_name(video_id: str, frame: int, ext: str = "jpg") -> str:
    return f"{video_id}_f{frame:06d}.{ext}"


@dataclass
class Fuente:
    """Un video y el tramo (alineado) del que se puede muestrear."""

    video: str
    start: int = 0
    end: Optional[int] = None  # None = hasta el final
    step: int = 1
    desde: str = "video"  # "video" o "entradas"


def _expand_videos(rutas: Sequence[str]) -> List[str]:
    out: List[str] = []
    for r in rutas:
        if os.path.isdir(r):
            out += sorted(os.path.join(r, f) for f in os.listdir(r) if f.lower().endswith(VIDEO_EXTS))
        else:
            out.append(r)
    return out


def fuentes_desde_args(videos: Sequence[str], entradas: Sequence[str], paso: int, desde_s: float, hasta_s: Optional[float],
                       out: Callable[[str], None] = print) -> List[Fuente]:
    fuentes: List[Fuente] = []
    for carpeta in entradas:
        try:
            inp = load_run_inputs(carpeta)
        except (FileNotFoundError, ValueError, OSError) as exc:
            out(f"AVISO: no se pudieron leer las entradas de {carpeta}: {exc}")
            continue
        fr = inp.data["frame_range"]
        fuentes.append(Fuente(inp.video_path, int(fr["start"]), int(fr["end"]), int(inp.data["timing"]["frame_step"]), "entradas"))
    for v in _expand_videos(videos):
        fuentes.append(Fuente(v, step=max(1, int(paso)), desde=f"video|{desde_s}|{hasta_s}"))
    return fuentes


# Muestreo
def _rango(f: Fuente, reader) -> tuple:
    """(start, end) en frames, ya recortado al video."""
    last = max(0, reader.total_frames - 1)
    if f.desde == "entradas":
        return max(0, f.start), min(last, f.end if f.end is not None else last)
    _, desde_s, hasta_s = f.desde.split("|")
    start = max(0, int(round(float(desde_s) * reader.fps)))
    end = last if hasta_s == "None" else min(last, int(round(float(hasta_s) * reader.fps)))
    return start, end


def _unique_id(base: str, existentes: Dict[str, str], huella: str) -> str:
    """Mismo video -> mismo id; otro video con el mismo nombre -> id con sufijo."""
    cand, n = base, 1
    while cand in existentes and existentes[cand] != huella:
        n += 1
        cand = f"{base}_{n}"
    return cand


def _write_json_atomic(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def muestrear(
    fuentes: Sequence[Fuente],
    salida: str,
    por_video: int = 20,
    semilla: int = 7,
    formato: str = "jpg",
    anadir: bool = False,
    sobrescribir: bool = False,
    out: Callable[[str], None] = print,
) -> int:
    """Escribe imágenes y manifiesto. Devuelve 0 si se generó algo, 2 si no."""
    if por_video < 1:
        out("--por-video debe ser al menos 1.")
        return 2
    if formato not in ("jpg", "png"):
        out("--formato debe ser jpg o png.")
        return 2
    if not fuentes:
        out("No hay videos que muestrear (indica videos o --entradas).")
        return 2

    manifest_path = os.path.join(salida, "manifest.json")
    existente = os.path.exists(manifest_path)
    if existente and not (anadir or sobrescribir):
        out(f"Ya existe {manifest_path}. Usa --anadir para agregar videos sin tocar lo anterior, o --sobrescribir.")
        return 2

    manifest: Dict[str, Any] = {"schema": SCHEMA, "clases": ["raton"], "proposito": "validacion", "ciego": True, "videos": []}
    if existente and anadir:
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, ValueError) as exc:
            out(f"No se pudo leer el manifiesto existente: {exc}")
            return 2

    os.makedirs(os.path.join(salida, "images"), exist_ok=True)
    os.makedirs(os.path.join(salida, "labels"), exist_ok=True)
    ids_huella = {v["id"]: v.get("sha256_extremos", "") for v in manifest["videos"]}
    huellas_hechas = {v.get("sha256_extremos") for v in manifest["videos"] if v.get("sha256_extremos")}

    total_nuevas = 0
    for f in fuentes:
        nombre = os.path.basename(f.video)
        if not os.path.exists(f.video):
            out(f"AVISO: no se encuentra {f.video}; se omite.")
            continue
        huella = video_fingerprint(f.video)
        if huella["sha256_extremos"] in huellas_hechas:  # ya en el manifiesto, o repetido en esta misma llamada
            out(f"  {nombre}: ese video ya está en el conjunto (o lo pasaste dos veces); se omite.")
            continue
        try:
            reader = VideoReader(f.video)
        except (VideoOpenError, OSError) as exc:
            out(f"AVISO: no se pudo abrir {nombre}: {exc}")
            continue
        with reader:
            start, end = _rango(f, reader)
            vid = _unique_id(safe_stem(f.video), ids_huella, huella["sha256_extremos"])
            rng = random.Random(f"{semilla}:{huella['sha256_extremos']}:{start}:{end}:{f.step}")
            plan = plan_frames(start, end, f.step, por_video, rng)
            quiero = set(plan)
            hechos: List[int] = []
            # Se leen DE CORRIDO, con la misma lectura que usa el pipeline (iter_frames). Un salto con read_at puede
            # caer unos frames fuera de lugar en algunos códecs (mp4): la imagen etiquetada sería de otro instante que
            # el frame que procesa el seguimiento, y la evaluación mediría un desfase en vez del seguimiento.
            if plan:
                for idx, frame in reader.iter_frames(start, max(plan), f.step):
                    if idx not in quiero:
                        continue
                    ruta = os.path.join(salida, "images", image_name(vid, idx, formato))
                    ok = cv2.imwrite(ruta, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95] if formato == "jpg" else [])
                    if not ok:
                        out(f"  AVISO: no se pudo guardar {ruta}.")
                        continue
                    hechos.append(idx)
            for idx in plan:
                if idx not in hechos:
                    out(f"  AVISO: no se pudo leer el frame {idx} de {nombre}; se omite.")
            if not hechos:
                out(f"AVISO: {nombre}: no se generó ninguna imagen.")
                continue
            ids_huella[vid] = huella["sha256_extremos"]
            huellas_hechas.add(huella["sha256_extremos"])
            manifest["videos"].append({
                "id": vid, "path": os.path.abspath(f.video), "nombre": nombre, **huella,
                "ancho": int(reader.width), "alto": int(reader.height), "fps": float(reader.fps),
                "total_frames": int(reader.total_frames), "rango": [start, end], "paso": int(f.step),
                "semilla": int(semilla), "formato": formato, "frames": hechos,
            })
            total_nuevas += len(hechos)
            if len(hechos) < por_video:
                out(f"  {nombre}: {len(hechos)} frames (se pidieron {por_video}; el tramo no daba más).")
            else:
                out(f"  {nombre}: {len(hechos)} frames, de {start} a {end}, paso {f.step}")

    if total_nuevas == 0:
        out("No se generó ninguna imagen.")
        return 2

    _write_json_atomic(manifest_path, manifest)
    with open(os.path.join(salida, "classes.txt"), "w", encoding="utf-8") as fh:
        fh.write("raton\n")
    with open(os.path.join(salida, "LEEME_ETIQUETADO.txt"), "w", encoding="utf-8") as fh:
        fh.write(LEEME)
    n_total = sum(len(v["frames"]) for v in manifest["videos"])
    out(f"\nListo: {total_nuevas} imágenes nuevas ({n_total} en total, de {len(manifest['videos'])} videos) en {salida}")
    out("Siguiente: etiqueta las imágenes (lee LEEME_ETIQUETADO.txt), guarda las etiquetas en labels/ y corre")
    out("  python tools/eval_tracking.py " + f'"{salida}" --entradas "<carpeta entradas_...>"')
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Elige frames de tus videos para etiquetar a mano un conjunto de validación.")
    ap.add_argument("videos", nargs="*", help="videos o carpetas con videos")
    ap.add_argument("--entradas", action="append", default=[], help="carpeta entradas_... (usa su rango y frame_step); se puede repetir")
    ap.add_argument("--salida", default="validacion", help="carpeta donde escribir (por omisión: validacion)")
    ap.add_argument("--por-video", type=int, default=20, help="frames por video (por omisión 20)")
    ap.add_argument("--semilla", type=int, default=7)
    ap.add_argument("--paso", type=int, default=1, help="alinear a un frame de cada N (para videos sin --entradas)")
    ap.add_argument("--desde-s", type=float, default=0.0, help="empezar en este segundo (videos sin --entradas)")
    ap.add_argument("--hasta-s", type=float, default=None, help="terminar en este segundo (videos sin --entradas)")
    ap.add_argument("--formato", choices=("jpg", "png"), default="jpg")
    ap.add_argument("--anadir", action="store_true", help="agregar videos a un conjunto que ya existe, sin tocar lo anterior")
    ap.add_argument("--sobrescribir", action="store_true", help="rehacer el manifiesto (las etiquetas de labels/ no se borran)")
    a = ap.parse_args(argv)
    fuentes = fuentes_desde_args(a.videos, a.entradas, a.paso, a.desde_s, a.hasta_s)
    return muestrear(fuentes, a.salida, a.por_video, a.semilla, a.formato, a.anadir, a.sobrescribir)


if __name__ == "__main__":
    sys.exit(main())