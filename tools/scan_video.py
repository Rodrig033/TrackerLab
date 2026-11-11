#!/usr/bin/env python3
"""Escanea un video COMPLETO con el detector (una imagen por segundo, por omisión) y te muestra DÓNDE tiene problemas.

No necesita etiquetas: sirve para descubrir, antes de integrar el detector en TrackerLab, con qué casos difíciles se va a topar
(reflejos, el tubo, la luz, cambios bruscos...) en videos que nadie ha etiquetado.

Uso (desde la raíz del proyecto; hace falta ``pip install onnxruntime``):
    python tools/scan_video.py "<video.mp4>" --modelo "<modelo.onnx>" --entradas "<carpeta entradas_...>"
    python tools/scan_video.py "<video.mp4>" --modelo "<modelo.onnx>" --entradas "<...>" --hasta-s 120      # solo los 2 primeros minutos

``--entradas`` es una corrida de ESE video (de ahí sale el ROI; se comprueba que sea del mismo video por su huella).
El detector se usa sobre el recorte del ROI con el margen con que se entrenó (leído de las notas del modelo).

Qué marca en cada imagen (de más a menos grave):
  SIN_DETECCION    ningún candidato con confianza suficiente
  VARIAS           otro candidato distinto del principal, también con confianza suficiente (¿reflejo? ¿dos animales?)
  CONFIANZA_BAJA   hay detección, pero con menos confianza de la deseable
  CAJA_RARA        el área de la caja se sale de lo habitual en ESTE video (criterio de Tukey)
  SALTO            el centro se movió mucho más de lo habitual entre dos imágenes seguidas
  BORDE            la caja toca el borde del recorte (el ratón puede estar saliéndose del ROI; informativo)

Una marca es una ALERTA para mirar la imagen, no un veredicto: el ratón puede haberse movido de verdad, o estar tapado por el tubo.

Escribe en una carpeta ``escaneo_<fecha>``: ``escaneo.csv`` (una fila por imagen), ``resumen.json`` y ``casos_dificiles.png`` (los peores
casos, repartidos entre los tipos de marca, con la caja principal en verde o amarillo y los otros candidatos en rojo).
Se puede interrumpir con Ctrl+C: se analiza lo que se llevaba. Código de salida: 0 = listo, 2 = no se pudo.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.dataset import video_fingerprint  # noqa: E402
from learning.onnx_decode import detect  # noqa: E402
from persistence.run_inputs import load_run_inputs  # noqa: E402
from tools.audit_labels import tukey_bounds  # noqa: E402
from tools.build_training_set import crop_rect  # noqa: E402
from tools.eval_detector import make_session, read_notes, resolve_margin  # noqa: E402
from video.reader import VideoReader  # noqa: E402

Box = Tuple[float, float, float, float]  # x, y, ancho, alto
Rect = Tuple[int, int, int, int]  # x0, y0, x1, y1 (x1, y1 exclusivos)

SEVERIDAD = ("SIN_DETECCION", "VARIAS", "CONFIANZA_BAJA", "CAJA_RARA", "SALTO", "BORDE")
COL_BUENA, COL_DUDOSA, COL_OTRA = (0, 220, 0), (0, 220, 255), (0, 0, 255)  # BGR
MIN_PARA_ESTADISTICA = 8  # con menos datos no se puede decir qué es «habitual» (Tukey)


@dataclass
class Det:
    score: float
    box: Box


@dataclass
class Sample:
    idx: int
    t_s: float
    ms: float
    dets: List[Det]  # de mayor a menor confianza, en píxeles del frame completo
    crop: Rect  # recorte que se le dio al detector, en píxeles del frame completo
    thumb: Optional[bytes] = field(default=None, repr=False)  # el recorte reducido, en JPEG
    escala: float = 1.0  # píxeles de la miniatura por píxel del frame


# ----------------------------------------------------------------------
# Análisis (puro: no necesita el video)
# ----------------------------------------------------------------------
def _iou(a: Box, b: Box) -> float:
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def best_of(s: Sample, umbral: float) -> Optional[Det]:
    """La detección principal: la de mayor confianza, si supera el umbral."""
    return s.dets[0] if s.dets and s.dets[0].score > umbral else None


def _centro(b: Box) -> Tuple[float, float]:
    return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0


def analyze(samples: Sequence[Sample], umbral: float, confianza: float, borde_px: int = 2) -> List[Tuple[str, ...]]:
    """Marcas de cada imagen, ordenadas de más a menos grave."""
    best = [best_of(s, umbral) for s in samples]
    marcas: List[set] = [set() for _ in samples]

    seguras = [b.box[2] * b.box[3] for b in best if b is not None and b.score >= confianza]
    area_lim = tukey_bounds(seguras) if len(seguras) >= MIN_PARA_ESTADISTICA else None
    pasos = [math.dist(_centro(best[i - 1].box), _centro(best[i].box))  # type: ignore[union-attr]
             for i in range(1, len(samples)) if best[i - 1] is not None and best[i] is not None]
    paso_lim = tukey_bounds(pasos)[1] if len(pasos) >= MIN_PARA_ESTADISTICA else None

    for i, (s, b) in enumerate(zip(samples, best)):
        if b is None:
            marcas[i].add("SIN_DETECCION")
            continue
        if b.score < confianza:
            marcas[i].add("CONFIANZA_BAJA")
        if any(d.score > umbral and _iou(b.box, d.box) < 0.3 for d in s.dets[1:]):
            marcas[i].add("VARIAS")
        if area_lim is not None and not (area_lim[0] <= b.box[2] * b.box[3] <= area_lim[1]):
            marcas[i].add("CAJA_RARA")
        if i > 0 and best[i - 1] is not None and paso_lim is not None:
            d = math.dist(_centro(best[i - 1].box), _centro(b.box))  # type: ignore[union-attr]
            if d > paso_lim and d > b.box[2]:
                marcas[i].add("SALTO")
        x0, y0, x1, y1 = s.crop
        x, y, w, h = b.box
        if x <= x0 + borde_px or y <= y0 + borde_px or x + w >= x1 - borde_px or y + h >= y1 - borde_px:
            marcas[i].add("BORDE")
    return [tuple(m for m in SEVERIDAD if m in conj) for conj in marcas]


def primary(marcas: Sequence[str]) -> Optional[str]:
    """La marca más grave de una imagen (None si no tiene ninguna)."""
    return marcas[0] if marcas else None


def gaps(samples: Sequence[Sample], marcas: Sequence[Sequence[str]], minimo: int = 2) -> List[Dict[str, Any]]:
    """Tramos de ``minimo`` o más imágenes seguidas SIN detección: lo que el seguimiento tendría que salvar por su cuenta."""
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(samples):
        if "SIN_DETECCION" in marcas[i]:
            j = i
            while j + 1 < len(samples) and "SIN_DETECCION" in marcas[j + 1]:
                j += 1
            if j - i + 1 >= minimo:
                out.append({"desde_s": samples[i].t_s, "hasta_s": samples[j].t_s, "imagenes": j - i + 1})
            i = j + 1
        else:
            i += 1
    return out


def select_cases(marcas: Sequence[Sequence[str]], n: int) -> List[int]:
    """Índices de las imágenes con problemas (sin contar las que solo tienen BORDE), repartidos entre los tipos de marca.

    Se toman por turnos de cada tipo (el más grave primero) para que un tipo muy frecuente no ocupe toda la hoja; dentro de cada
    tipo, en orden de tiempo. Devuelve a lo más ``n``, ordenados por gravedad y luego por tiempo.
    """
    grupos: Dict[str, List[int]] = {m: [] for m in SEVERIDAD if m != "BORDE"}
    for i, ms in enumerate(marcas):
        p = primary([m for m in ms if m != "BORDE"])
        if p:
            grupos[p].append(i)
    elegidos: List[int] = []
    punteros = {m: 0 for m in grupos}
    while len(elegidos) < n and any(punteros[m] < len(g) for m, g in grupos.items()):
        for m, g in grupos.items():
            if punteros[m] < len(g) and len(elegidos) < n:
                elegidos.append(g[punteros[m]])
                punteros[m] += 1
    orden = {m: k for k, m in enumerate(SEVERIDAD)}
    return sorted(elegidos, key=lambda i: (orden[primary([m for m in marcas[i] if m != "BORDE"])], i))  # type: ignore[index]


def summarize(samples: Sequence[Sample], marcas: Sequence[Sequence[str]], umbral: float, confianza: float, cada_s: float) -> Dict[str, Any]:
    n = len(samples)
    best = [best_of(s, umbral) for s in samples]
    scores = [b.score for b in best if b is not None]
    tiempos = [s.ms for s in samples[2:]] if n > 4 else [s.ms for s in samples]
    cuenta = {m: sum(1 for ms in marcas if m in ms) for m in SEVERIDAD}
    problemas = sum(1 for ms in marcas if any(m != "BORDE" for m in ms))
    pct = lambda a: (100.0 * a / n) if n else None  # noqa: E731
    return {
        "imagenes": n, "cada_s": cada_s, "umbral": umbral, "confianza": confianza,
        "con_deteccion": len(scores), "con_deteccion_pct": pct(len(scores)),
        "marcas": cuenta, "marcas_pct": {m: pct(c) for m, c in cuenta.items()},
        "imagenes_con_problemas": problemas, "imagenes_con_problemas_pct": pct(problemas),
        "confianza_mediana": float(statistics.median(scores)) if scores else None,
        "confianza_p10": float(np.percentile(scores, 10)) if scores else None,
        "ms_mediana": float(statistics.median(tiempos)) if tiempos else None,
        "tramos_sin_deteccion": gaps(samples, marcas),
    }


# ----------------------------------------------------------------------
# Salidas
# ----------------------------------------------------------------------
def mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60}:{t % 60:02d}"


def write_csv(path: str, samples: Sequence[Sample], marcas: Sequence[Sequence[str]], umbral: float) -> None:
    cab = ["frame", "tiempo_s", "ms", "n_candidatos", "confianza", "confianza_2", "x", "y", "ancho", "alto", "cx", "cy", "area", "marcas"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(cab)
        for s, ms in zip(samples, marcas):
            b = best_of(s, umbral)
            seg = f"{s.dets[1].score:.3f}" if len(s.dets) > 1 else ""
            if b is None:
                wr.writerow([s.idx, f"{s.t_s:.2f}", f"{s.ms:.1f}", len(s.dets), f"{s.dets[0].score:.3f}" if s.dets else "", seg, "", "", "", "", "", "", "", "|".join(ms)])
                continue
            x, y, w, h = b.box
            cx, cy = _centro(b.box)
            wr.writerow([s.idx, f"{s.t_s:.2f}", f"{s.ms:.1f}", len(s.dets), f"{b.score:.3f}", seg, f"{x:.1f}", f"{y:.1f}", f"{w:.1f}", f"{h:.1f}",
                         f"{cx:.1f}", f"{cy:.1f}", f"{w * h:.0f}", "|".join(ms)])


def render_sheet(samples: Sequence[Sample], marcas: Sequence[Sequence[str]], indices: Sequence[int], umbral: float, confianza: float,
                 columnas: int = 5, celda: int = 320) -> Optional[np.ndarray]:
    """Hoja con las imágenes de ``indices``: la caja principal en verde (confianza suficiente) o amarillo (dudosa), los otros candidatos en rojo."""
    if not indices:
        return None
    celdas = []
    for i in indices:
        s = samples[i]
        im = cv2.imdecode(np.frombuffer(s.thumb, np.uint8), cv2.IMREAD_COLOR) if s.thumb else None
        if im is None:
            im = np.full((celda, celda, 3), 40, np.uint8)
        else:
            x0, y0 = s.crop[0], s.crop[1]
            for k, d in enumerate(s.dets):
                x, y, w, h = d.box
                col = (COL_BUENA if d.score >= confianza else COL_DUDOSA) if k == 0 and d.score > umbral else COL_OTRA
                cv2.rectangle(im, (int(round((x - x0) * s.escala)), int(round((y - y0) * s.escala))),
                              (int(round((x + w - x0) * s.escala)), int(round((y + h - y0) * s.escala))), col, 2)
        im = cv2.resize(im, (celda, int(round(im.shape[0] * celda / im.shape[1]))), interpolation=cv2.INTER_AREA)
        cv2.putText(im, f"f{s.idx}  {mmss(s.t_s)}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        etiqueta = ",".join(m for m in marcas[i] if m != "BORDE")
        sc = f" {s.dets[0].score:.2f}" if s.dets else ""
        cv2.putText(im, (etiqueta + sc)[:44], (4, im.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        celdas.append(im)
    alto = max(c.shape[0] for c in celdas)
    filas = (len(celdas) + columnas - 1) // columnas
    hoja = np.full((filas * alto, columnas * celda, 3), 25, np.uint8)
    for k, c in enumerate(celdas):
        r, col = divmod(k, columnas)
        hoja[r * alto:r * alto + c.shape[0], col * celda:(col + 1) * celda] = c
        cv2.rectangle(hoja, (col * celda, r * alto), ((col + 1) * celda - 1, (r + 1) * alto - 1), (60, 60, 60), 1)
    return hoja


# ----------------------------------------------------------------------
# Lectura del video
# ----------------------------------------------------------------------
def roi_from_run(video: str, entradas: str) -> Rect:
    """ROI (x, y, ancho, alto) de una corrida, comprobando que sea DE ESE video (por su huella; sin huella, por nombre de archivo)."""
    inputs = load_run_inputs(entradas)
    vd = inputs.data.get("video") or {}
    huella = vd.get("sha256_extremos")
    if huella:
        if huella != video_fingerprint(video)["sha256_extremos"]:
            raise ValueError(f"La corrida {os.path.basename(os.path.normpath(entradas))} es de otro video (la huella no coincide con {os.path.basename(video)}).")
    elif os.path.basename(str(vd.get("path", ""))) != os.path.basename(video):
        raise ValueError(f"La corrida {os.path.basename(os.path.normpath(entradas))} es de {os.path.basename(str(vd.get('path', '')))}, no de {os.path.basename(video)}.")
    roi = inputs.data.get("roi_box")
    if not (isinstance(roi, list) and len(roi) == 4):
        raise ValueError(f"La corrida {os.path.basename(os.path.normpath(entradas))} no trae roi_box.")
    return tuple(int(x) for x in roi)  # type: ignore[return-value]


def scan(
    video: str,
    modelo: str,
    entradas: Optional[str],
    *,
    cada_s: float = 1.0,
    desde_s: float = 0.0,
    hasta_s: Optional[float] = None,
    umbral: float = 0.5,
    umbral_bajo: float = 0.2,
    confianza: float = 0.8,
    margen_roi: Optional[int] = None,
    sin_recorte: bool = False,
    hilos: Optional[int] = None,
    salida: Optional[str] = None,
    hoja: int = 40,
    columnas: int = 5,
    celda: int = 320,
    out: Callable[[str], None] = print,
    session_factory: Callable[[str, Optional[int]], Any] = make_session,
    now: Optional[Callable[[], datetime]] = None,
) -> int:
    if cada_s <= 0 or desde_s < 0 or (hasta_s is not None and hasta_s <= desde_s):
        out("--cada-s debe ser mayor que 0, y --desde-s menor que --hasta-s.")
        return 2
    if not (0.0 < umbral_bajo <= umbral < 1.0) or not (umbral <= confianza <= 1.0):
        out("Los umbrales deben cumplir 0 < --umbral-bajo <= --umbral < 1 y --umbral <= --confianza <= 1.")
        return 2
    if margen_roi is not None and margen_roi < 0:
        out("--margen-roi no puede ser negativo.")
        return 2
    if columnas < 1 or celda < 60 or hoja < 0:
        out("--columnas >= 1, --celda >= 60 y --hoja >= 0.")
        return 2
    if not os.path.isfile(video):
        out(f"No existe el video {video}.")
        return 2
    if not os.path.isfile(modelo):
        out(f"No existe el modelo {modelo}.")
        return 2
    if not sin_recorte and not entradas:
        out("Indica el ROI con --entradas \"<carpeta entradas_...>\" (una corrida de ese video), o usa --sin-recorte para el frame completo.")
        return 2
    roi: Optional[Tuple[int, int, int, int]] = None
    if not sin_recorte:
        try:
            roi = roi_from_run(video, entradas)  # type: ignore[arg-type]
        except (FileNotFoundError, ValueError, OSError) as exc:
            out(str(exc))
            return 2
    try:
        sesion = session_factory(modelo, hilos)
    except Exception as exc:  # onnxruntime lanza tipos propios
        out(f"No se pudo abrir el modelo {modelo}: {exc}")
        return 2
    notas = read_notes(sesion)
    margen, origen = resolve_margin(notas, margen_roi, sin_recorte)
    if notas.get("recorte_roi") is False and not sin_recorte:
        out("AVISO: este modelo se entrenó con frames completos, no con recorte al ROI; quizá debas usar --sin-recorte.")

    muestras: List[Sample] = []
    interrumpido = False
    t_inicio = time.perf_counter()
    try:
        with VideoReader(video) as r:
            if r.fps_is_fallback:
                out("AVISO: el video no trae su FPS; se supone 30 y los tiempos pueden salir corridos.")
            paso = max(1, int(round(r.fps * cada_s)))
            r.skip_limit = max(r.skip_limit, paso)  # saltar con grab() mantiene exactos los números de frame (buscar en un mp4 no)
            inicio = int(desde_s * r.fps)
            fin = None if hasta_s is None else int(hasta_s * r.fps)
            if r.total_frames:
                fin = r.total_frames - 1 if fin is None else min(fin, r.total_frames - 1)
            esperadas = None if fin is None else (fin - inicio) // paso + 1
            try:
                rect = None if sin_recorte else crop_rect(roi, margen, r.width, r.height)  # type: ignore[arg-type]
            except ValueError as exc:
                out(str(exc))
                return 2
            out(f"Video: {os.path.basename(video)} ({r.fps:.1f} fps, {r.width}x{r.height}). Una imagen cada {cada_s:g} s = cada {paso} frames"
                + (f" ({esperadas} imágenes)." if esperadas else ".") + f" Margen del ROI: {margen} px ({origen}).")
            marca_progreso = max(1, (esperadas or 0) // 10)
            try:
                for idx, frame in r.iter_frames(inicio, fin, paso):
                    x0, y0, x1, y1 = rect if rect else (0, 0, frame.shape[1], frame.shape[0])
                    t = time.perf_counter()
                    try:
                        det = detect(sesion, cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), umbral_bajo)
                    except ValueError as exc:  # p. ej. un ONNX con entrada de tamaño dinámico
                        out(f"El modelo no se puede usar así: {exc}")
                        return 2
                    ms = (time.perf_counter() - t) * 1000.0
                    dets = []
                    for (bx1, by1, bx2, by2), sc in zip(det.xyxy, det.scores):
                        if bx2 - bx1 > 0 and by2 - by1 > 0:
                            dets.append(Det(float(sc), (float(bx1) + x0, float(by1) + y0, float(bx2 - bx1), float(by2 - by1))))
                    esc = celda / (x1 - x0)
                    miniatura = cv2.resize(frame[y0:y1, x0:x1], (celda, max(1, int(round((y1 - y0) * esc)))), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", miniatura, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    muestras.append(Sample(idx, idx / r.fps, ms, dets, (x0, y0, x1, y1), buf.tobytes() if ok else None, esc))
                    n = len(muestras)
                    if n == 5 and esperadas:
                        seg = (time.perf_counter() - t_inicio) / n * (esperadas - n)
                        out(f"  Ritmo: ~{(time.perf_counter() - t_inicio) / n * 1000:.0f} ms por imagen (con la lectura del video); faltan unos {seg / 60:.1f} min. Ctrl+C para parar y analizar lo que haya.")
                    elif esperadas and n % marca_progreso == 0 and n != 5:
                        out(f"  {n}/{esperadas} imágenes ({100 * n // esperadas} %)")
            except KeyboardInterrupt:
                interrumpido = True
                out(f"\nInterrumpido: se analizan las {len(muestras)} imágenes que se llevaban.")
    except (FileNotFoundError, OSError) as exc:
        out(str(exc))
        return 2
    if not muestras:
        out("No se leyó ninguna imagen del video (¿el rango queda fuera?).")
        return 2

    marcas = analyze(muestras, umbral, confianza)
    resumen = summarize(muestras, marcas, umbral, confianza, cada_s)
    resumen.update(video=os.path.basename(video), modelo=os.path.basename(modelo), margen_roi_px=margen, interrumpido=interrumpido,
                   duracion_escaneada_s=muestras[-1].t_s - muestras[0].t_s, notas_del_modelo=notas, hilos=hilos,
                   tiempo_total_s=time.perf_counter() - t_inicio)
    out(f"\n{resumen['imagenes']} imágenes escaneadas (de {mmss(muestras[0].t_s)} a {mmss(muestras[-1].t_s)}).")
    out(f"Con detección (confianza > {umbral:g}): {resumen['con_deteccion']} ({resumen['con_deteccion_pct']:.1f} %). "
        f"Confianza mediana {resumen['confianza_mediana']:.2f}" + (f", el 10 % peor {resumen['confianza_p10']:.2f}." if resumen["confianza_p10"] is not None else ".")
        if resumen["con_deteccion"] else f"Con detección: 0 de {resumen['imagenes']}.")
    out("Marcas (una imagen puede tener varias):")
    for m in SEVERIDAD:
        out(f"  {m:<15} {resumen['marcas'][m]:>5}  ({resumen['marcas_pct'][m]:.1f} %)")
    out(f"Imágenes con alguna marca grave (sin contar BORDE): {resumen['imagenes_con_problemas']} ({resumen['imagenes_con_problemas_pct']:.1f} %).")
    for g in resumen["tramos_sin_deteccion"][:8]:
        out(f"  Sin detección entre {mmss(g['desde_s'])} y {mmss(g['hasta_s'])} ({g['imagenes']} imágenes seguidas)")
    if len(resumen["tramos_sin_deteccion"]) > 8:
        out(f"  ... y {len(resumen['tramos_sin_deteccion']) - 8} tramos más (están en resumen.json)")
    if resumen["ms_mediana"] is not None:
        out(f"Tiempo del detector por imagen: mediana {resumen['ms_mediana']:.0f} ms (sin contar la lectura del video). Total del escaneo: {resumen['tiempo_total_s'] / 60:.1f} min.")

    stamp = (now or datetime.now)().strftime("%Y%m%d_%H%M%S")
    salida = salida or os.path.join(os.path.dirname(os.path.abspath(entradas)) if entradas else os.getcwd(), f"escaneo_{stamp}")  # junto a la carpeta de la corrida
    os.makedirs(salida, exist_ok=True)
    write_csv(os.path.join(salida, "escaneo.csv"), muestras, marcas, umbral)
    with open(os.path.join(salida, "resumen.json"), "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, ensure_ascii=False, indent=2)
    if hoja > 0:
        elegidos = select_cases(marcas, hoja)
        img = render_sheet(muestras, marcas, elegidos, umbral, confianza, columnas, celda)
        if img is not None and cv2.imwrite(os.path.join(salida, "casos_dificiles.png"), img):
            out(f"\nCasos difíciles ({len(elegidos)}, repartidos entre los tipos de marca): {os.path.join(salida, 'casos_dificiles.png')}")
            out("  verde = detección con confianza suficiente, amarillo = dudosa, rojo = otros candidatos")
        elif img is None:
            out("\nNo hay casos difíciles que mostrar.")
    out(f"Resultados: {salida}")
    return 0


def main(argv: Optional[List[str]] = None, **kw: Any) -> int:
    ap = argparse.ArgumentParser(description="Escanea un video completo con el detector y muestra dónde tiene problemas.")
    ap.add_argument("video", help="el video (.mp4, etc.)")
    ap.add_argument("--modelo", required=True, help="el .onnx")
    ap.add_argument("--entradas", help="carpeta entradas_... de una corrida de ESE video (de ahí sale el ROI)")
    ap.add_argument("--sin-recorte", action="store_true", help="usar el frame completo en vez del recorte al ROI")
    ap.add_argument("--margen-roi", type=int, help="margen del ROI en px (por omisión, el de las notas del modelo, o 0)")
    ap.add_argument("--cada-s", type=float, default=1.0, help="una imagen cada tantos segundos de video (1)")
    ap.add_argument("--desde-s", type=float, default=0.0, help="empezar en este segundo (0)")
    ap.add_argument("--hasta-s", type=float, help="terminar en este segundo (por omisión, el final)")
    ap.add_argument("--umbral", type=float, default=0.5, help="confianza mínima para contar una detección (0.5)")
    ap.add_argument("--umbral-bajo", type=float, default=0.2, help="confianza mínima para considerar un candidato (0.2); permite ver los dudosos")
    ap.add_argument("--confianza", type=float, default=0.8, help="por debajo de esta confianza la detección se marca CONFIANZA_BAJA (0.8)")
    ap.add_argument("--hilos", type=int, help="hilos de cálculo de onnxruntime")
    ap.add_argument("--salida", help="dónde escribir los resultados")
    ap.add_argument("--hoja", type=int, default=40, help="cuántos casos difíciles mostrar en casos_dificiles.png (40; 0 = ninguno)")
    ap.add_argument("--columnas", type=int, default=5)
    ap.add_argument("--celda", type=int, default=320, help="ancho en píxeles de cada celda de la hoja (320)")
    a = ap.parse_args(argv)
    return scan(a.video, a.modelo, a.entradas, cada_s=a.cada_s, desde_s=a.desde_s, hasta_s=a.hasta_s, umbral=a.umbral, umbral_bajo=a.umbral_bajo,
                confianza=a.confianza, margen_roi=a.margen_roi, sin_recorte=a.sin_recorte, hilos=a.hilos, salida=a.salida, hoja=a.hoja,
                columnas=a.columnas, celda=a.celda, **kw)


if __name__ == "__main__":
    sys.exit(main())