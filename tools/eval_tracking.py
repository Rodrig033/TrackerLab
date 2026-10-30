#!/usr/bin/env python3
"""Mide qué tan bien sigue TrackerLab al ratón, comparando contra frames ETIQUETADOS A MANO.

Uso (desde la raíz del proyecto):
    python tools/eval_tracking.py validacion --entradas "<carpeta entradas_...>"
    python tools/eval_tracking.py validacion --entradas "<entradas fondo>" --entradas "<entradas silueta>" --entradas "<entradas color>"
    python tools/eval_tracking.py validacion --entradas "<entradas_...>" --sin-filtros

``validacion`` es la carpeta que creó tools/sample_frames.py, con tus etiquetas YOLO dentro de labels/.
Cada ``--entradas`` es la carpeta ``entradas_<fecha>`` de una corrida normal de la aplicación (el modo, el ROI, la
ventana del sujeto... ya están ahí). Se repite esa corrida sin ventanas y se mira dónde dijo que estaba el ratón en
los frames etiquetados. Si das varias corridas del mismo video con distintos modos, la tabla los compara.

Qué se mide (por corrida, por modo y en total):
  * Acierto (IoU): la CAJA del tracker se solapa con la etiquetada al menos ``--iou`` (0.5 por omisión).
    Depende de la convención de la caja (por ejemplo, con o sin cola).
  * Posición dentro de la caja: la posición que queda en el CSV (con suavizado y rechazo de saltos) cae dentro
    de la caja etiquetada. Es la medida que más se parece a lo que usa un investigador y no depende de los bordes.
  * Error del centro: distancia (px, y cm si hay escala) entre esa posición y el centro de la caja etiquetada.
  * Lugar equivocado: dio una posición, pero no era el ratón. Perdido: no dio nada y el ratón estaba.
  * Falso positivo: dio una posición en una imagen que etiquetaste SIN ratón.
Los frames que el pipeline no procesa (fuera del rango o saltados por frame_step) no se evalúan y se cuentan aparte.

Escribe, en ``<validacion>/evaluacion_<fecha>/``: resultados.json, por_frame.csv y fallos.png (los peores casos, con la
caja etiquetada en verde y la posición del tracker en rojo).

Código de salida: 0 = evaluado, 2 = no se pudo evaluar.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.dataset import parse_yolo_label_line  # noqa: E402
from learning.metrics import center_distance, evaluate_single_object, iou  # noqa: E402
from persistence.run_inputs import build_job, load_run_inputs  # noqa: E402
from pipeline import Quit, run_tracking  # noqa: E402
from video.reader import VideoReader  # noqa: E402

Box = Tuple[float, float, float, float]

ST_OK = "acierto"
ST_WRONG = "lugar equivocado"
ST_MISS = "perdido"
ST_FP = "falso positivo"
ST_TN = "sin raton: bien"


# Etiquetas
def parse_label_file(path: str, frame_w: int, frame_h: int) -> Tuple[str, Optional[Box]]:
    """Lee una etiqueta YOLO. Devuelve (estado, caja): ``caja`` | ``vacio`` | ``varias`` | ``invalida``."""
    with open(path, encoding="utf-8-sig") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    if not lines:
        return "vacio", None
    if len(lines) > 1:
        return "varias", None
    try:
        _cls, box = parse_yolo_label_line(lines[0], frame_w, frame_h)
    except ValueError:
        return "invalida", None
    return "caja", box


def image_stem(video_id: str, frame: int) -> str:
    return f"{video_id}_f{frame:06d}"


def load_manifest(valdir: str) -> Dict[str, Any]:
    path = os.path.join(valdir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe {path}: genera el conjunto con tools/sample_frames.py.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def collect_truths(valdir: str, manifest: Dict[str, Any], faltantes_sin_raton: bool = False) -> Tuple[Dict[Tuple[str, int], Optional[Box]], Dict[str, Any]]:
    """Etiquetas por (id_de_video, frame). ``None`` = etiquetada sin ratón. Devuelve también un resumen de problemas."""
    truths: Dict[Tuple[str, int], Optional[Box]] = {}
    info: Dict[str, Any] = {"sin_etiqueta": [], "invalidas": [], "con_varias_cajas": [], "con_caja": 0, "vacias": 0}
    labels_dir = os.path.join(valdir, "labels")
    for v in manifest["videos"]:
        for frame in v["frames"]:
            stem = image_stem(v["id"], frame)
            path = os.path.join(labels_dir, stem + ".txt")
            key = (v["id"], int(frame))
            if not os.path.exists(path):
                if faltantes_sin_raton:
                    truths[key] = None
                    info["vacias"] += 1
                else:
                    info["sin_etiqueta"].append(stem)
                continue
            estado, box = parse_label_file(path, v["ancho"], v["alto"])
            if estado == "caja":
                truths[key] = box
                info["con_caja"] += 1
            elif estado == "vacio":
                truths[key] = None
                info["vacias"] += 1
            elif estado == "varias":
                info["con_varias_cajas"].append(stem)
            else:
                info["invalidas"].append(stem)
    return truths, info

# Evaluación (pura: no abre videos)
@dataclass(frozen=True)
class Pred:
    """Lo que dijo el seguimiento en un frame: caja del tracker y posición que queda en el CSV (px de imagen, y hacia abajo)."""

    found: bool
    box: Optional[Box] = None
    pos: Optional[Tuple[float, float]] = None


def _inside(pt: Tuple[float, float], box: Box) -> bool:
    return box[0] <= pt[0] <= box[0] + box[2] and box[1] <= pt[1] <= box[1] + box[3]


def score_frames(truths: Dict[Any, Optional[Box]], preds: Dict[Any, Pred], iou_thr: float = 0.5,
                 cm_per_px: Optional[float] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compara frame a frame. Solo se evalúan las claves presentes en ``truths`` Y en ``preds``."""
    keys = [k for k in truths if k in preds]
    rows: List[Dict[str, Any]] = []
    pboxes: List[Optional[Box]] = []
    tboxes: List[Optional[Box]] = []
    for k in keys:
        t, p = truths[k], preds[k]
        pbox = p.box if p.found else None
        pboxes.append(pbox)
        tboxes.append(t)
        row: Dict[str, Any] = {"clave": k, "tiene_raton": t is not None, "caja_etiqueta": t, "caja_prediccion": pbox,
                               "iou": None, "error_centro_px": None, "error_centro_rel": None, "posicion_dentro": None}
        if t is None:
            row["estado"] = ST_FP if p.found else ST_TN
        elif not p.found:
            row["estado"] = ST_MISS
        else:
            if pbox is not None:
                row["iou"] = iou(pbox, t)
            if p.pos is not None:
                tc = (t[0] + t[2] / 2.0, t[1] + t[3] / 2.0)
                row["error_centro_px"] = math.dist(p.pos, tc)
                diag = math.hypot(t[2], t[3])
                row["error_centro_rel"] = row["error_centro_px"] / diag if diag > 0 else None
                row["posicion_dentro"] = _inside(p.pos, t)
            ok = (row["iou"] >= iou_thr) if row["iou"] is not None else bool(row["posicion_dentro"])
            row["estado"] = ST_OK if ok else ST_WRONG
        rows.append(row)

    ev = evaluate_single_object(pboxes, tboxes, iou_thr)
    con = [r for r in rows if r["tiene_raton"]]
    sin = [r for r in rows if not r["tiene_raton"]]
    errs = [r["error_centro_px"] for r in con if r["error_centro_px"] is not None]
    rels = [r["error_centro_rel"] for r in con if r["error_centro_rel"] is not None]
    ious = [r["iou"] for r in con if r["iou"] is not None]
    dentro = sum(1 for r in con if r["posicion_dentro"])

    def pct(a: float, b: float) -> Optional[float]:
        return 100.0 * a / b if b else None

    def stat(vals: List[float], q: float) -> Optional[float]:
        return float(np.percentile(vals, q)) if vals else None

    summary: Dict[str, Any] = {
        "evaluados": len(rows), "con_raton": len(con), "sin_raton": len(sin),
        "aciertos": ev.true_positive, "acierto_pct": pct(ev.true_positive, len(con)),
        "posicion_dentro": dentro, "posicion_dentro_pct": pct(dentro, len(con)),
        "lugar_equivocado": ev.wrong_place, "perdidos": ev.missed, "perdidos_pct": pct(ev.missed, len(con)),
        "falsos_positivos": ev.false_positive, "falsos_positivos_pct": pct(ev.false_positive, len(sin)),
        "iou_mediano": stat(ious, 50),
        "error_centro_mediana_px": stat(errs, 50), "error_centro_p90_px": stat(errs, 90),
        "error_centro_max_px": max(errs) if errs else None, "error_centro_rel_mediana": stat(rels, 50),
        "umbral_iou": iou_thr,
    }
    if cm_per_px:
        for k in ("error_centro_mediana_px", "error_centro_p90_px", "error_centro_max_px"):
            summary[k.replace("_px", "_cm")] = summary[k] * cm_per_px if summary[k] is not None else None
    return rows, summary


def merge_scored(rows_lists: Sequence[List[Dict[str, Any]]], iou_thr: float) -> Dict[str, Any]:
    """Resumen de varias corridas juntas (todas las filas, no un promedio de promedios)."""
    rows = [r for lst in rows_lists for r in lst]
    truths = {i: r["caja_etiqueta"] for i, r in enumerate(rows)}
    preds = {}
    for i, r in enumerate(rows):
        preds[i] = Pred(r["estado"] not in (ST_MISS, ST_TN), r["caja_prediccion"],
                        r.get("_pos"))
    _, s = score_frames(truths, preds, iou_thr)
    return s


# Correr el seguimiento sobre los frames etiquetados
def predict_frames(carpeta: str, video_path: str, frames: Sequence[int], overrides: Dict[str, Any]) -> Tuple[Dict[int, Pred], Dict[str, Any]]:
    """Repite la corrida grabada y devuelve lo que dijo el seguimiento en cada frame pedido que se procesó."""
    inputs = load_run_inputs(carpeta)
    pend = set(int(f) for f in frames)
    preds: Dict[int, Pred] = {}
    with VideoReader(video_path) as reader:
        job = build_job(inputs, reader, **overrides)
        x0, y0 = job.roi_box[0], job.roi_box[1]
        alto = reader.height

        def on_frame(ev):
            idx = ev.frame_idx
            if idx not in pend:
                return None
            rec, det = ev.record, ev.detection
            if rec.found:
                box = (det.bbox[0] + x0, det.bbox[1] + y0, det.bbox[2], det.bbox[3]) if det.found else None
                preds[idx] = Pred(True, box, (float(rec.cx_abs), float(alto - rec.cy_abs_inv)))
            else:
                preds[idx] = Pred(False)
            pend.discard(idx)
            return None if pend else Quit()

        run_tracking(job, on_frame=on_frame, keep_records=False, log=None)
    meta = {"modo": inputs.mode, "cm_per_px": inputs.data.get("cm_per_px"), "no_procesados": sorted(pend),
            "rango": [job.frame_range.start, job.frame_range.end], "frame_step": job.timing.frame_step}
    return preds, meta


def match_video(manifest: Dict[str, Any], inputs_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """El video del manifiesto al que corresponde una corrida: por huella; si no hay, por nombre de archivo."""
    vid = inputs_data.get("video", {})
    huella = vid.get("sha256_extremos")
    if huella:
        for v in manifest["videos"]:
            if v.get("sha256_extremos") == huella:
                return v
    nombre = os.path.basename(vid.get("path", ""))
    for v in manifest["videos"]:
        if v["nombre"] == nombre:
            return v
    return None


# Salidas
def _fmt(v: Optional[float], dec: int = 1, suf: str = "") -> str:
    return "-" if v is None else f"{v:.{dec}f}{suf}"


def format_table(rows: List[Tuple[str, Dict[str, Any]]]) -> str:
    cols = ["", "n", "con ratón", "acierto IoU", "pos. dentro", "mal ubicado", "perdido", "falso pos.", "error centro (mediana / p90 px)"]
    data = [cols]
    for nombre, s in rows:
        data.append([nombre, str(s["evaluados"]), str(s["con_raton"]), _fmt(s["acierto_pct"], 0, " %"), _fmt(s["posicion_dentro_pct"], 0, " %"),
                     str(s["lugar_equivocado"]), str(s["perdidos"]), f"{s['falsos_positivos']}/{s['sin_raton']}",
                     f"{_fmt(s['error_centro_mediana_px'])} / {_fmt(s['error_centro_p90_px'])}"])
    w = [max(len(r[i]) for r in data) for i in range(len(cols))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in data)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    cab = ["entradas", "modo", "video", "frame", "estado", "tiene_raton", "iou", "error_centro_px", "error_centro_rel",
           "posicion_dentro", "caja_etiqueta", "caja_prediccion"]

    def c(v):
        if v is None:
            return ""
        if isinstance(v, bool):
            return "si" if v else "no"
        if isinstance(v, float):
            return f"{v:.4f}"
        if isinstance(v, tuple):
            return " ".join(f"{x:.1f}" for x in v)
        return v

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(cab)
        for r in rows:
            wr.writerow([c(r.get(k)) for k in cab])


_SEVERIDAD = {ST_FP: 0, ST_MISS: 1, ST_WRONG: 2}


def write_failure_sheet(path: str, rows: List[Dict[str, Any]], valdir: str, manifest: Dict[str, Any], n: int = 12, ancho: int = 360) -> bool:
    """Hoja con los peores casos: etiqueta en verde, posición del tracker en rojo. Devuelve False si no hubo fallos."""
    fallos = [r for r in rows if r["estado"] in _SEVERIDAD]
    fallos.sort(key=lambda r: (_SEVERIDAD[r["estado"]], -(r.get("error_centro_px") or 0.0)))
    fallos = fallos[:n]
    if not fallos:
        return False
    por_id = {v["id"]: v for v in manifest["videos"]}
    celdas = []
    for r in fallos:
        vid, frame = r["clave"]
        v = por_id[vid]
        img = cv2.imread(os.path.join(valdir, "images", f"{image_stem(vid, frame)}.{v.get('formato', 'jpg')}"))
        if img is None:
            continue
        if r["caja_etiqueta"] is not None:
            x, y, w, h = (int(round(q)) for q in r["caja_etiqueta"])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
        if r.get("_pos") is not None:
            px, py = int(round(r["_pos"][0])), int(round(r["_pos"][1]))
            cv2.drawMarker(img, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 22, 3)
        esc = ancho / img.shape[1]
        img = cv2.resize(img, (ancho, max(1, int(img.shape[0] * esc))))
        ascii_estado = {ST_FP: "FALSO POSITIVO", ST_MISS: "PERDIDO", ST_WRONG: "MAL UBICADO"}[r["estado"]]
        cv2.rectangle(img, (0, 0), (ancho, 22), (30, 30, 30), -1)
        cv2.putText(img, f"{ascii_estado} {vid[:16]} f{frame}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        celdas.append(img)
    if not celdas:
        return False
    alto = max(c.shape[0] for c in celdas)
    cols = min(4, len(celdas))
    filas = []
    for i in range(0, len(celdas), cols):
        fila = [np.pad(c, ((0, alto - c.shape[0]), (0, 0), (0, 0))) for c in celdas[i:i + cols]]
        while len(fila) < cols:
            fila.append(np.zeros((alto, ancho, 3), np.uint8))
        filas.append(np.hstack(fila))
    return bool(cv2.imwrite(path, np.vstack(filas)))


# Programa principal
def run(valdir: str, entradas: Sequence[str], *, iou_thr: float = 0.5, salida: Optional[str] = None, sin_filtros: bool = False,
        sin_busqueda: bool = False, faltantes_sin_raton: bool = False, hoja_fallos: int = 12, out: Callable[[str], None] = print,
        now: Optional[Callable[[], datetime]] = None) -> int:
    if not entradas:
        out("Indica al menos una corrida con --entradas \"<carpeta entradas_...>\".")
        return 2
    try:
        manifest = load_manifest(valdir)
    except (FileNotFoundError, ValueError, OSError) as exc:
        out(str(exc))
        return 2
    truths, info = collect_truths(valdir, manifest, faltantes_sin_raton)
    if not truths:
        out("No hay etiquetas que evaluar en labels/. Etiqueta las imágenes (ver LEEME_ETIQUETADO.txt).")
        if info["sin_etiqueta"]:
            out(f"  {len(info['sin_etiqueta'])} imágenes sin archivo de etiqueta.")
        return 2
    out(f"Etiquetas: {info['con_caja']} con ratón, {info['vacias']} sin ratón; "
        f"{len(info['sin_etiqueta'])} imágenes sin etiqueta (se omiten).")
    for clave, titulo in (("invalidas", "con formato inválido"), ("con_varias_cajas", "con más de una caja (un solo ratón: corrígelas)")):
        if info[clave]:
            out(f"AVISO: {len(info[clave])} etiquetas {titulo}, se omiten: {', '.join(info[clave][:5])}{'...' if len(info[clave]) > 5 else ''}")

    overrides: Dict[str, Any] = {}
    if sin_filtros:
        overrides.update(smoothing_alpha=None, jump_max_px=None, jump_max_speed_px_s=None)
    if sin_busqueda:
        overrides["reacquire"] = False

    corridas: List[Dict[str, Any]] = []
    todas_filas: List[Dict[str, Any]] = []
    for carpeta in entradas:
        try:
            inputs = load_run_inputs(carpeta)
        except (FileNotFoundError, ValueError, OSError) as exc:
            out(f"AVISO: no se pudieron leer las entradas {carpeta}: {exc}")
            continue
        v = match_video(manifest, inputs.data)
        if v is None:
            out(f"AVISO: la corrida {os.path.basename(carpeta)} es de un video que no está en el conjunto de validación; se omite.")
            continue
        ruta = inputs.video_path if os.path.exists(inputs.video_path) else v["path"]
        if not os.path.exists(ruta):
            out(f"AVISO: no se encuentra el video {v['nombre']}; se omite.")
            continue
        frames = [f for (vid, f) in truths if vid == v["id"]]
        if not frames:
            out(f"AVISO: no hay frames etiquetados de {v['nombre']}; se omite.")
            continue
        out(f"Evaluando {os.path.basename(carpeta)} ({inputs.mode}, {v['nombre']}, {len(frames)} frames etiquetados)...")
        try:
            preds, meta = predict_frames(carpeta, ruta, frames, overrides)
        except (ValueError, RuntimeError, OSError) as exc:
            out(f"AVISO: no se pudo repetir la corrida {os.path.basename(carpeta)}: {exc}")
            continue
        tr = {(v["id"], f): truths[(v["id"], f)] for f in frames}
        pr = {(v["id"], f): p for f, p in preds.items()}
        rows, summary = score_frames(tr, pr, iou_thr, meta["cm_per_px"])
        for r in rows:  # la posición se guarda en la fila para la hoja de fallos y el resumen conjunto
            r["_pos"] = pr[r["clave"]].pos
            r.update(entradas=os.path.basename(carpeta.rstrip("/\\")), modo=meta["modo"], video=v["nombre"], frame=r["clave"][1])
        if meta["no_procesados"]:
            out(f"  AVISO: {len(meta['no_procesados'])} frames etiquetados no se procesan en esta corrida "
                f"(rango {meta['rango'][0]}-{meta['rango'][1]}, paso {meta['frame_step']}); no se evalúan. "
                "Para alinear el muestreo, usa tools/sample_frames.py con --entradas.")
        corridas.append({"entradas": carpeta, "modo": meta["modo"], "video": v["nombre"], "no_procesados": len(meta["no_procesados"]),
                         "resumen": summary, "_rows": rows})
        todas_filas += rows

    if not corridas:
        out("No se pudo evaluar ninguna corrida.")
        return 2

    tabla = [(f"{c['modo']} · {c['video'][:22]}", c["resumen"]) for c in corridas]
    por_modo: Dict[str, List[List[Dict[str, Any]]]] = {}
    for c in corridas:
        por_modo.setdefault(c["modo"], []).append(c["_rows"])
    resumen_modos = {m: merge_scored(lst, iou_thr) for m, lst in por_modo.items()}
    if len(corridas) > 1:
        tabla += [(f"TOTAL {m}", s) for m, s in resumen_modos.items()]
    out("\n" + format_table(tabla))
    out(f"\nFiltros: {'apagados (--sin-filtros)' if sin_filtros else 'los de la corrida'}. Umbral de IoU: {iou_thr:g}.")

    stamp = (now or datetime.now)().strftime("%Y%m%d_%H%M%S")
    salida = salida or os.path.join(valdir, f"evaluacion_{stamp}")
    os.makedirs(salida, exist_ok=True)
    limpio = [{k: v for k, v in c.items() if k != "_rows"} for c in corridas]
    with open(os.path.join(salida, "resultados.json"), "w", encoding="utf-8") as fh:
        json.dump({"iou": iou_thr, "sin_filtros": sin_filtros, "etiquetas": {k: (v if not isinstance(v, list) else len(v)) for k, v in info.items()},
                   "corridas": limpio, "por_modo": resumen_modos}, fh, ensure_ascii=False, indent=2)
    write_csv(os.path.join(salida, "por_frame.csv"), todas_filas)
    if hoja_fallos > 0 and write_failure_sheet(os.path.join(salida, "fallos.png"), todas_filas, valdir, manifest, hoja_fallos):
        out(f"Peores casos: {os.path.join(salida, 'fallos.png')}  (verde = etiqueta, cruz roja = posición del tracker)")
    out(f"Resultados: {salida}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evalúa el seguimiento de TrackerLab contra frames etiquetados a mano.")
    ap.add_argument("validacion", help="carpeta creada por tools/sample_frames.py (con labels/ llena)")
    ap.add_argument("--entradas", action="append", default=[], help="carpeta entradas_... de una corrida; se puede repetir")
    ap.add_argument("--iou", type=float, default=0.5, help="umbral de IoU para contar un acierto (0.5)")
    ap.add_argument("--salida", help="dónde escribir los resultados")
    ap.add_argument("--sin-filtros", action="store_true", help="evalúa sin suavizado ni rechazo de saltos")
    ap.add_argument("--sin-busqueda", action="store_true", help="evalúa sin la búsqueda del sujeto (modo silueta)")
    ap.add_argument("--faltantes-sin-raton", action="store_true", help="una etiqueta que no existe cuenta como «sin ratón»")
    ap.add_argument("--hoja-fallos", type=int, default=12, help="cuántos fallos mostrar en fallos.png (0 = ninguno)")
    a = ap.parse_args(argv)
    return run(a.validacion, a.entradas, iou_thr=a.iou, salida=a.salida, sin_filtros=a.sin_filtros, sin_busqueda=a.sin_busqueda,
               faltantes_sin_raton=a.faltantes_sin_raton, hoja_fallos=a.hoja_fallos)


if __name__ == "__main__":
    sys.exit(main())