#!/usr/bin/env python3
"""Evalúa un detector ONNX (el que exporta el cuaderno de Colab) contra frames etiquetados a mano.

Usa LAS MISMAS métricas y la misma tabla que tools/eval_tracking.py, así el resultado del detector se compara directamente con
el del seguimiento clásico sobre los mismos frames. Además mide el TIEMPO POR IMAGEN en esta máquina.

Uso (desde la raíz del proyecto; hace falta ``pip install onnxruntime`` en el entorno):
    python tools/eval_detector.py "<validacion>" --modelo "<modelo.onnx>" --entradas "<carpeta entradas_...>"

  * ``<validacion>``: carpeta hecha con tools/sample_frames.py y etiquetada con tools/label_frames.py (por ejemplo la de tu
    video RESERVADO). No hace falta el video: se leen sus imágenes.
  * ``--modelo``: el ``.onnx``. Se puede repetir para comparar varios (por ejemplo small y nano) en la misma tabla.
  * ``--entradas``: carpeta ``entradas_<fecha>`` de una corrida de ESE video; de ahí sale el ROI. El detector se usa sobre el
    recorte del ROI, que es como se entrenó. Con ``--sin-recorte`` se usa el frame completo.

Detalles que importan:
  * MARGEN DEL ROI. El modelo se entrenó con un recorte al ROI con cierto margen (``--margen-roi`` al armar el conjunto).
    Al usarlo debe ser el MISMO. El cuaderno guarda ese margen dentro del ONNX (notas); si no pasas ``--margen-roi``, se lee de
    ahí. Si lo pasas, manda lo que pases.
  * Se toma la detección de MAYOR confianza (un solo ratón). Un frame «sin ratón» con una detección cuenta como falso positivo.
  * La posición que se evalúa es el centro de la caja del detector.
  * El tiempo por imagen incluye recortar, preparar, correr el modelo y decodificar; se descartan las primeras imágenes
    (la primera inferencia es más lenta). Es el tiempo de ESTA máquina y de este número de hilos.

Escribe ``resultados.json``, ``por_frame.csv`` y ``fallos.png`` en una carpeta ``evaluacion_detector_<fecha>`` dentro de la de
validación (o en ``--salida``). Código de salida: 0 = evaluado, 2 = no se pudo.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.onnx_decode import detect  # noqa: E402
from persistence.run_inputs import load_run_inputs  # noqa: E402
from tools.build_training_set import crop_rect  # noqa: E402
from tools.eval_tracking import (Pred, collect_truths, format_table, load_manifest, match_video, score_frames, write_csv,  # noqa: E402
                                 write_failure_sheet)

WARMUP = 2  # imágenes que no cuentan para el tiempo: la primera inferencia es más lenta
Rect = Tuple[int, int, int, int]


# ----------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------
def make_session(ruta: str, hilos: Optional[int] = None) -> Any:
    """Sesión de onnxruntime en CPU. ``hilos`` limita los hilos de cálculo (None = los que decida onnxruntime)."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("No está instalado onnxruntime. Instálalo con: pip install onnxruntime") from exc
    opciones = ort.SessionOptions()
    if hilos:
        opciones.intra_op_num_threads = int(hilos)
    return ort.InferenceSession(ruta, opciones, providers=["CPUExecutionProvider"])


def read_notes(sesion: Any) -> Dict[str, Any]:
    """Notas que el cuaderno guardó dentro del ONNX (modelo, resolución, margen del ROI...). {} si no hay o no se pueden leer."""
    try:
        meta = sesion.get_modelmeta().custom_metadata_map
        crudo = meta.get("rfdetr_notes") if meta else None
        datos = json.loads(crudo) if crudo else {}
        return datos if isinstance(datos, dict) else {}
    except Exception:  # un ONNX sin metadatos, o con metadatos raros, no impide evaluar
        return {}


def timing_stats(ms: Sequence[float], descartar: int = WARMUP) -> Dict[str, Optional[float]]:
    """Mediana y percentil 90 del tiempo por imagen, sin las primeras ``descartar`` (si quedan menos de 3, se usan todas)."""
    vals = list(ms[descartar:]) if len(ms) - descartar >= 3 else list(ms)
    if not vals:
        return {"n": 0, "mediana_ms": None, "p90_ms": None, "max_ms": None}
    s = sorted(vals)
    pos = 0.9 * (len(s) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return {"n": len(vals), "mediana_ms": float(statistics.median(vals)), "p90_ms": float(s[lo] + (s[hi] - s[lo]) * (pos - lo)), "max_ms": float(s[-1])}


def box_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """IoU (mediano y el 10 % peor) y cuánto más ancha o alta es la caja del detector que la tuya, en los frames con ratón y detección."""
    pares = [(r["caja_etiqueta"], r["caja_prediccion"]) for r in rows if r["tiene_raton"] and r["caja_etiqueta"] and r["caja_prediccion"]]
    ious = [r["iou"] for r in rows if r["tiene_raton"] and r.get("iou") is not None]
    anchos = [p[2] / t[2] for t, p in pares if t[2] > 0]
    altos = [p[3] / t[3] for t, p in pares if t[3] > 0]
    return {
        "con_caja": len(pares),
        "iou_mediano": float(statistics.median(ious)) if ious else None,
        "iou_p10": float(np.percentile(ious, 10)) if ious else None,
        "ancho_razon_mediana": float(statistics.median(anchos)) if anchos else None,
        "alto_razon_mediana": float(statistics.median(altos)) if altos else None,
    }


def predict_image(sesion: Any, img_bgr: Any, rect: Optional[Rect], umbral: float) -> Tuple[Pred, float]:
    """Corre el detector sobre un frame (recortado al ``rect``, o completo si es None). Devuelve (predicción en coordenadas del
    FRAME completo, milisegundos que tardó todo el proceso)."""
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = rect if rect else (0, 0, w, h)
    t = time.perf_counter()
    det = detect(sesion, cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), umbral)
    ms = (time.perf_counter() - t) * 1000.0
    if len(det.scores) == 0:
        return Pred(False), ms
    bx1, by1, bx2, by2 = (float(v) for v in det.xyxy[0])  # la de mayor confianza
    if bx2 - bx1 <= 0 or by2 - by1 <= 0:
        return Pred(False), ms
    caja = (bx1 + x0, by1 + y0, bx2 - bx1, by2 - by1)
    return Pred(True, caja, (caja[0] + caja[2] / 2.0, caja[1] + caja[3] / 2.0)), ms


# ----------------------------------------------------------------------
# Evaluación
# ----------------------------------------------------------------------
def _nombres_de_modelos(rutas: Sequence[str]) -> List[str]:
    base = [os.path.splitext(os.path.basename(r))[0] for r in rutas]
    vistos: Dict[str, int] = {}
    out = []
    for n in base:
        vistos[n] = vistos.get(n, 0) + 1
        out.append(n if vistos[n] == 1 else f"{n} #{vistos[n]}")
    return out


def _imagen(valdir: str, v: Dict[str, Any], frame: int) -> Optional[Any]:
    ext = v.get("formato", "jpg")
    return cv2.imread(os.path.join(valdir, "images", f"{v['id']}_f{frame:06d}.{ext}"))


def evaluate_model(sesion: Any, valdir: str, manifest: Dict[str, Any], truths: Dict[Any, Any], rois: Dict[str, Optional[Rect]],
                   umbral: float, iou_thr: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Optional[float]], List[str]]:
    """Filas por frame, resumen, tiempos y avisos de un modelo sobre todos los frames etiquetados con ROI conocido."""
    por_id = {v["id"]: v for v in manifest["videos"]}
    tr: Dict[Any, Any] = {}
    pr: Dict[Any, Pred] = {}
    ms: List[float] = []
    avisos: List[str] = []
    for (vid, frame), caja in truths.items():
        if vid not in rois:
            continue
        img = _imagen(valdir, por_id[vid], frame)
        if img is None:
            avisos.append(f"no se pudo leer la imagen de {por_id[vid].get('nombre', vid)} f{frame}; se omite.")
            continue
        pred, t = predict_image(sesion, img, rois[vid], umbral)
        tr[(vid, frame)], pr[(vid, frame)] = caja, pred
        ms.append(t)
    rows, resumen = score_frames(tr, pr, iou_thr)
    for r in rows:
        r["_pos"] = pr[r["clave"]].pos
        r.update(modo="detector", video=por_id[r["clave"][0]].get("nombre", r["clave"][0]), frame=r["clave"][1])
    return rows, resumen, timing_stats(ms), avisos


def run(
    valdir: str,
    modelos: Sequence[str],
    entradas: Sequence[str],
    *,
    umbral: float = 0.5,
    iou_thr: float = 0.5,
    margen_roi: Optional[int] = None,
    sin_recorte: bool = False,
    hilos: Optional[int] = None,
    salida: Optional[str] = None,
    hoja_fallos: int = 12,
    faltantes_sin_raton: bool = False,
    out: Callable[[str], None] = print,
    session_factory: Callable[[str, Optional[int]], Any] = make_session,
    now: Optional[Callable[[], datetime]] = None,
) -> int:
    if not modelos:
        out("Indica al menos un modelo con --modelo \"<modelo.onnx>\".")
        return 2
    if not (0.0 < umbral < 1.0):
        out("--umbral debe estar entre 0 y 1 (sin incluirlos).")
        return 2
    if margen_roi is not None and margen_roi < 0:
        out("--margen-roi no puede ser negativo.")
        return 2
    for m in modelos:
        if not os.path.exists(m):
            out(f"No existe el modelo {m}.")
            return 2
    if not sin_recorte and not entradas:
        out("Indica el ROI con --entradas \"<carpeta entradas_...>\" (una corrida de ese video), o usa --sin-recorte para el frame completo.")
        return 2
    try:
        manifest = load_manifest(valdir)
    except (FileNotFoundError, ValueError, OSError) as exc:
        out(str(exc))
        return 2
    truths, info = collect_truths(valdir, manifest, faltantes_sin_raton)
    if not truths:
        out("No hay etiquetas que evaluar en labels/. Etiqueta las imágenes (ver LEEME_ETIQUETADO.txt).")
        return 2
    out(f"Etiquetas: {info['con_caja']} con ratón, {info['vacias']} sin ratón; {len(info['sin_etiqueta'])} imágenes sin etiqueta (se omiten).")

    # ROI de cada video del conjunto (se empareja por huella, o por nombre, igual que en eval_tracking.py)
    rois_base: Dict[str, Optional[Tuple[int, int, int, int]]] = {}
    if sin_recorte:
        rois_base = {v["id"]: None for v in manifest["videos"]}
    for carpeta in entradas:
        try:
            inputs = load_run_inputs(carpeta)
        except (FileNotFoundError, ValueError, OSError) as exc:
            out(f"AVISO: no se pudieron leer las entradas {carpeta}: {exc}")
            continue
        v = match_video(manifest, inputs.data)
        if v is None:
            out(f"AVISO: la corrida {os.path.basename(carpeta)} es de un video que no está en el conjunto; se omite.")
            continue
        roi = inputs.data.get("roi_box")
        if not (isinstance(roi, list) and len(roi) == 4):
            out(f"AVISO: la corrida {os.path.basename(carpeta)} no trae roi_box; se omite.")
            continue
        rois_base.setdefault(v["id"], tuple(int(x) for x in roi))  # type: ignore[arg-type]
    con_etiquetas = {vid for (vid, _f) in truths}
    sin_roi = [v.get("nombre", v["id"]) for v in manifest["videos"] if v["id"] in con_etiquetas and v["id"] not in rois_base]
    for n in sin_roi:
        out(f"AVISO: no hay ROI para {n}; sus frames no se evalúan.")
    if not any(vid in rois_base for vid in con_etiquetas):
        out("No se pudo evaluar ningún video: falta su ROI.")
        return 2

    nombres = _nombres_de_modelos(modelos)
    resultados: List[Dict[str, Any]] = []
    todas: List[Dict[str, Any]] = []
    for ruta, nombre in zip(modelos, nombres):
        try:
            sesion = session_factory(ruta, hilos)
        except Exception as exc:  # onnxruntime lanza tipos propios: cualquier fallo al abrir el modelo se explica igual
            out(f"No se pudo abrir el modelo {ruta}: {exc}")
            continue
        notas = read_notes(sesion)
        if sin_recorte:
            margen, origen = 0, "no aplica (frame completo)"
        elif margen_roi is not None:
            margen, origen = margen_roi, "indicado con --margen-roi"
        elif isinstance(notas.get("margen_roi_px"), int):
            margen, origen = notas["margen_roi_px"], "leído de las notas del modelo"
        else:
            margen, origen = 0, "por omisión (el modelo no trae notas)"
        if notas.get("recorte_roi") is False and not sin_recorte:
            out(f"AVISO ({nombre}): este modelo se entrenó con frames completos, no con recorte al ROI; quizá debas usar --sin-recorte.")
        out(f"\nModelo {nombre}: margen del ROI {margen} px ({origen})" + (f"; notas: {json.dumps(notas, ensure_ascii=False)}" if notas else ""))
        rois: Dict[str, Optional[Rect]] = {}
        for v in manifest["videos"]:
            if v["id"] in rois_base:
                roi = rois_base[v["id"]]
                try:
                    rois[v["id"]] = None if roi is None else crop_rect(roi, margen, int(v["ancho"]), int(v["alto"]))
                except ValueError as exc:
                    out(f"AVISO: {exc} Se omite {v.get('nombre', v['id'])}.")
        try:
            rows, resumen, tiempos, avisos = evaluate_model(sesion, valdir, manifest, truths, rois, umbral, iou_thr)
        except ValueError as exc:  # p. ej. un ONNX con entrada de tamaño dinámico
            out(f"El modelo {nombre} no se puede usar así: {exc}")
            continue
        for a in avisos:
            out(f"  AVISO: {a}")
        for r in rows:
            r["corrida"] = nombre
        resultados.append({"nombre": nombre, "ruta": ruta, "notas": notas, "margen_roi_px": margen, "resumen": resumen, "cajas": box_stats(rows),
                           "tiempos": tiempos, "_rows": rows})
        todas += rows

    if not resultados:
        out("No se pudo evaluar ningún modelo.")
        return 2
    out("\n" + format_table([(f"{r['nombre'][:30]} · detector", r["resumen"]) for r in resultados]))
    out(f"\nUmbral de confianza: {umbral:g}. Umbral de IoU: {iou_thr:g}.")
    for r in resultados:
        b = r["cajas"]
        if b["iou_mediano"] is not None and b["ancho_razon_mediana"] is not None:
            out(f"{r['nombre']}: IoU mediano {b['iou_mediano']:.2f} (el 10 % peor: {b['iou_p10']:.2f}); la caja del detector es, en mediana, "
                f"{b['ancho_razon_mediana']:.2f} veces el ancho y {b['alto_razon_mediana']:.2f} veces el alto de la tuya.")
    for r in resultados:
        t = r["tiempos"]
        if t["mediana_ms"] is not None:
            out(f"Tiempo por imagen de {r['nombre']}: mediana {t['mediana_ms']:.0f} ms, p90 {t['p90_ms']:.0f} ms (sobre {t['n']} imágenes, "
                f"{('con ' + str(hilos) + ' hilos') if hilos else 'con los hilos por omisión'}). Es el de ESTA máquina.")

    stamp = (now or datetime.now)().strftime("%Y%m%d_%H%M%S")
    salida = salida or os.path.join(valdir, f"evaluacion_detector_{stamp}")
    os.makedirs(salida, exist_ok=True)
    limpio = [{k: v for k, v in r.items() if k != "_rows"} for r in resultados]
    with open(os.path.join(salida, "resultados.json"), "w", encoding="utf-8") as fh:
        json.dump({"umbral_confianza": umbral, "umbral_iou": iou_thr, "hilos": hilos, "sin_recorte": sin_recorte,
                   "etiquetas": {k: (v if not isinstance(v, list) else len(v)) for k, v in info.items()}, "modelos": limpio}, fh, ensure_ascii=False, indent=2)
    write_csv(os.path.join(salida, "por_frame.csv"), todas)
    if hoja_fallos > 0 and write_failure_sheet(os.path.join(salida, "fallos.png"), todas, valdir, manifest, hoja_fallos):
        out(f"Peores casos: {os.path.join(salida, 'fallos.png')}  (verde = etiqueta, cruz roja = centro de la caja del detector)")
    out(f"Resultados: {salida}")
    return 0


def main(argv: Optional[List[str]] = None, **kw: Any) -> int:
    ap = argparse.ArgumentParser(description="Evalúa un detector ONNX contra frames etiquetados a mano, con las métricas de eval_tracking.py.")
    ap.add_argument("validacion", help="carpeta creada por tools/sample_frames.py (con labels/ llena)")
    ap.add_argument("--modelo", action="append", default=[], help="el .onnx (se puede repetir para comparar varios)")
    ap.add_argument("--entradas", action="append", default=[], help="carpeta entradas_... de una corrida de ese video (de ahí sale el ROI); se puede repetir")
    ap.add_argument("--sin-recorte", action="store_true", help="usar el frame completo en vez del recorte al ROI")
    ap.add_argument("--margen-roi", type=int, help="margen del ROI en px (por omisión, el que traiga el modelo en sus notas, o 0)")
    ap.add_argument("--umbral", type=float, default=0.5, help="confianza mínima para contar una detección (0.5)")
    ap.add_argument("--iou", type=float, default=0.5, help="umbral de IoU para contar un acierto (0.5)")
    ap.add_argument("--hilos", type=int, help="hilos de cálculo de onnxruntime (por omisión, los que decida él)")
    ap.add_argument("--salida", help="dónde escribir los resultados")
    ap.add_argument("--hoja-fallos", type=int, default=12, help="cuántos fallos mostrar en fallos.png (0 = ninguno)")
    ap.add_argument("--faltantes-sin-raton", action="store_true", help="una etiqueta que no existe cuenta como «sin ratón»")
    a = ap.parse_args(argv)
    return run(a.validacion, a.modelo, a.entradas, umbral=a.umbral, iou_thr=a.iou, margen_roi=a.margen_roi, sin_recorte=a.sin_recorte, hilos=a.hilos,
               salida=a.salida, hoja_fallos=a.hoja_fallos, faltantes_sin_raton=a.faltantes_sin_raton, **kw)


if __name__ == "__main__":
    sys.exit(main())