from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.model import crop_rect  # noqa: E402,F401  (fuente única; se reexporta)
from learning.dataset import parse_yolo_label_line, yolo_label_line  # noqa: E402
from learning.review import ACCEPTED, CORRECTED, EXCLUDED, PENDING, load_revision  # noqa: E402

CLASS_NAME = "raton"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
JPEG_QUALITY = 95
Rect = Tuple[int, int, int, int]  # x0, y0, x1, y1 (x1, y1 exclusivos)

ADV_AUTOMATICAS, ADV_BORDE, ADV_POSITIVOS = 0, 1, 2  # posiciones de ADVERTENCIAS
ADVERTENCIAS = [
    "Las etiquetas son AUTOMÁTICAS (las produjo el seguimiento clásico): heredan sus errores y su forma de caja "
    "(medido en validación: más angostas que las dibujadas a mano y a veces con parte de la cola). Revísalas.",
    "El generador descarta los frames cuya caja toca el borde del ROI: el conjunto casi no contiene ratones pegados a la pared.",
    "Solo hay ejemplos positivos (frames con ratón); no hay imágenes sin ratón.",
]


# ----------------------------------------------------------------------
# Piezas puras
# ----------------------------------------------------------------------
def safe_name(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r"[^0-9A-Za-z_-]+", "_", stem).strip("_")
    return stem or "video"


def convert_label(line: str, frame_w: int, frame_h: int, rect: Optional[Rect]) -> str:
    """Etiqueta YOLO del frame completo -> etiqueta YOLO del recorte (o la misma, si no hay recorte)."""
    cls, (x, y, w, h) = parse_yolo_label_line(line, frame_w, frame_h)
    if cls != 0:
        raise ValueError(f"clase {cls}: solo existe la clase 0 ({CLASS_NAME})")
    if rect is None:
        return yolo_label_line((x, y, w, h), frame_w, frame_h)
    x0, y0, x1, y1 = rect
    nx0, ny0 = max(x, x0) - x0, max(y, y0) - y0
    nx1, ny1 = min(x + w, x1) - x0, min(y + h, y1) - y0
    if nx1 - nx0 < 1 or ny1 - ny0 < 1:
        raise ValueError("la caja queda fuera del recorte")
    return yolo_label_line((nx0, ny0, nx1 - nx0, ny1 - ny0), x1 - x0, y1 - y0)


def evenly(items: Sequence[Any], n: Optional[int]) -> List[Any]:
    """``n`` elementos repartidos de forma pareja (y determinista) a lo largo de la lista; todos si n es None o sobra."""
    if n is None or n >= len(items):
        return list(items)
    if n < 1:
        return []
    return [items[(2 * i + 1) * len(items) // (2 * n)] for i in range(n)]


@dataclass
class Fuente:
    """Una carpeta ``dataset_<fecha>`` (un video) y el papel que juega."""

    ruta: str
    rol: str  # "train" | "valid"
    info: Dict[str, Any] = field(default_factory=dict)
    prefijo: str = ""
    huella: Optional[str] = None
    roi_box: Optional[Tuple[int, int, int, int]] = None
    roi_origen: str = ""
    revision: Optional[Dict[str, Any]] = None
    tipo: str = "auto"  # "auto" (dataset_<fecha>) o "manual" (sample_frames + label_frames)
    items: Optional[List[Tuple[str, str, str, str]]] = None  # solo "manual": (división, nombre, imagen, etiqueta)


def read_info(ruta: str) -> Dict[str, Any]:
    p = os.path.join(ruta, "dataset_info.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{ruta} no es una carpeta dataset_<fecha>: falta dataset_info.json")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def find_roi(ruta: str, info: Dict[str, Any]) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    """ROI de la corrida: primero en la ficha del conjunto; si no, en ``entradas_<mismo sello>/inputs.json``."""
    roi = (info.get("opciones_de_procesamiento") or {}).get("roi_box")
    if roi and len(roi) == 4:
        return tuple(int(v) for v in roi), "dataset_info.json"  # type: ignore[return-value]
    base = os.path.basename(os.path.normpath(ruta))
    m = re.match(r"^dataset_(.+)$", base)
    if m:
        p = os.path.join(os.path.dirname(os.path.normpath(ruta)), f"entradas_{m.group(1)}", "inputs.json")
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    roi = json.load(fh).get("roi_box")
            except (OSError, ValueError):
                roi = None
            if roi and len(roi) == 4:
                return tuple(int(v) for v in roi), f"entradas_{m.group(1)}/inputs.json"  # type: ignore[return-value]
    return None, ""


def read_manual_manifest(ruta: str) -> Dict[str, Any]:
    """Manifiesto de un conjunto hecho con tools/sample_frames.py (frames para etiquetar a mano)."""
    p = os.path.join(ruta, "manifest.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{ruta} no es un conjunto de frames etiquetados a mano: falta manifest.json (lo crea tools/sample_frames.py).")
    try:
        with open(p, encoding="utf-8") as fh:
            man = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f"No se pudo leer {p}: {exc}") from exc
    if not isinstance(man.get("videos"), list) or not man["videos"]:
        raise ValueError(f"{p} no trae ningún video.")
    return man


def manual_items(ruta: str, video: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    """(división, nombre, imagen, etiqueta) de los frames de un video del manifiesto, en orden de frame.

    Los nombres siguen a ``sample_frames.image_name``: ``<id>_f<frame de 6 cifras>.<formato>``.
    """
    ext = video.get("formato", "jpg")
    out = []
    for idx in sorted(int(f) for f in video.get("frames", [])):
        stem = f"{video['id']}_f{idx:06d}"
        out.append(("manual", f"f{idx:06d}", os.path.join(ruta, "images", f"{stem}.{ext}"), os.path.join(ruta, "labels", stem + ".txt")))
    return out


def read_run_roi(carpeta: str) -> Tuple[Dict[str, Any], Tuple[int, int, int, int]]:
    """(datos del video, ROI) de una carpeta ``entradas_<fecha>`` (la que deja una corrida con SAVE_RUN_INPUTS = True)."""
    p = os.path.join(carpeta, "inputs.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{carpeta} no es una carpeta entradas_<fecha>: falta inputs.json.")
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    roi = d.get("roi_box")
    if not (isinstance(roi, list) and len(roi) == 4):
        raise ValueError(f"{p} no trae roi_box.")
    return d.get("video") or {}, tuple(int(v) for v in roi)  # type: ignore[return-value]


def match_roi(video: Dict[str, Any], corridas: Sequence[Tuple[str, Dict[str, Any], Tuple[int, int, int, int]]]) -> List[Tuple[Tuple[int, int, int, int], str]]:
    """ROIs de las corridas que son de ``video`` (un video del manifiesto), por huella; si no hay huella, por nombre de archivo."""
    huella = video.get("sha256_extremos")
    por_huella = [(roi, os.path.basename(os.path.normpath(c)) + "/inputs.json") for c, vd, roi in corridas if huella and vd.get("sha256_extremos") == huella]
    if por_huella:
        return por_huella
    nombre = video.get("nombre")
    return [(roi, os.path.basename(os.path.normpath(c)) + "/inputs.json") for c, vd, roi in corridas
            if nombre and os.path.basename(vd.get("path", "")) == nombre and not (huella and vd.get("sha256_extremos"))]


def list_items(ruta: str, splits: Sequence[str]) -> List[Tuple[str, str, str, str]]:
    """(split, nombre, ruta_imagen, ruta_etiqueta) de las imágenes del conjunto, ordenadas."""
    out: List[Tuple[str, str, str, str]] = []
    for sp in splits:
        d = os.path.join(ruta, "images", sp)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            nombre, ext = os.path.splitext(f)
            if ext.lower() in IMAGE_EXTS:
                out.append((sp, nombre, os.path.join(d, f), os.path.join(ruta, "labels", sp, nombre + ".txt")))
    return out


def reserved_fingerprints(carpetas: Sequence[str]) -> Dict[str, str]:
    """Huellas de los videos de los conjuntos de validación a mano (``manifest.json`` de tools/sample_frames.py)."""
    out: Dict[str, str] = {}
    for c in carpetas:
        p = os.path.join(c, "manifest.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{c} no es un conjunto de validación (falta manifest.json)")
        with open(p, encoding="utf-8") as fh:
            for v in json.load(fh).get("videos", []):
                if v.get("sha256_extremos"):
                    out[v["sha256_extremos"]] = v.get("nombre", "?")
    return out


# ----------------------------------------------------------------------
# Armado
# ----------------------------------------------------------------------
def _write_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def build(
    train: Sequence[str],
    valid: Sequence[str],
    salida: str,
    *,
    recorte_roi: bool = True,
    margen_roi: int = 0,
    excluir_videos_de: Sequence[str] = (),
    max_por_video: Optional[int] = None,
    sobrescribir: bool = False,
    solo_revisadas: bool = False,
    train_manual: Sequence[str] = (),
    valid_manual: Sequence[str] = (),
    entradas: Sequence[str] = (),
    out: Callable[[str], None] = print,
) -> int:
    if not train and not train_manual:
        out("Indica al menos un video de entrenamiento con --train o --train-manual.")
        return 2
    if margen_roi < 0:
        out("--margen-roi no puede ser negativo.")
        return 2
    if max_por_video is not None and max_por_video < 1:
        out("--max-por-video debe ser al menos 1.")
        return 2

    # 1) fuentes
    fuentes: List[Fuente] = []
    vistos: Dict[str, str] = {}
    for rol, rutas in (("train", train), ("valid", valid)):
        for r in rutas:
            real = os.path.realpath(r)
            if real in vistos:
                out(f"La carpeta {r} está repetida (ya figura como {vistos[real]}).")
                return 2
            vistos[real] = rol
            try:
                info = read_info(r)
            except (FileNotFoundError, ValueError, OSError) as exc:
                out(str(exc))
                return 2
            huella = (info.get("video") or {}).get("sha256_extremos")
            nombre = (info.get("video") or {}).get("nombre") or os.path.basename(os.path.dirname(os.path.normpath(r)))
            prefijo = safe_name(nombre) + (f"_{huella[:8]}" if huella else "")
            f = Fuente(r, rol, info, prefijo, huella)
            try:
                f.revision = load_revision(r)
            except ValueError as exc:  # revision.json dañado: no se ignora en silencio
                out(str(exc))
                return 2
            if recorte_roi:
                f.roi_box, f.roi_origen = find_roi(r, info)
                if f.roi_box is None:
                    out(f"No se encontró el ROI de {r} (ni en dataset_info.json ni en entradas_<fecha>/inputs.json). "
                        "Vuelve a correr ese video con la versión actual, o usa --sin-recorte.")
                    return 2
            fuentes.append(f)

    avisos_roi: List[str] = []
    corridas: List[Tuple[str, Dict[str, Any], Tuple[int, int, int, int]]] = []
    for c in entradas:
        try:
            vd, roi = read_run_roi(c)
        except (FileNotFoundError, ValueError, OSError) as exc:
            out(str(exc))
            return 2
        corridas.append((c, vd, roi))
    for rol, rutas in (("train", train_manual), ("valid", valid_manual)):
        for r in rutas:
            real = os.path.realpath(r)
            if real in vistos:
                out(f"La carpeta {r} está repetida (ya figura como {vistos[real]}).")
                return 2
            vistos[real] = rol
            try:
                man = read_manual_manifest(r)
            except (FileNotFoundError, ValueError, OSError) as exc:
                out(str(exc))
                return 2
            for v in man["videos"]:
                huella = v.get("sha256_extremos")
                prefijo = safe_name(v.get("nombre") or v["id"]) + (f"_{huella[:8]}" if huella else "")
                f = Fuente(f"{r}#{v['id']}", rol, {"video": {"nombre": v.get("nombre"), "sha256_extremos": huella}, "modo_de_seguimiento": "a mano"},
                           prefijo, huella, tipo="manual")
                f.items = manual_items(r, v)
                if recorte_roi:
                    hits = match_roi(v, corridas)
                    if not hits:
                        out(f"No se encontró el ROI de «{v.get('nombre')}»: pasa --entradas con la carpeta entradas_<fecha> de una corrida de ESE video, o usa --sin-recorte.")
                        return 2
                    f.roi_box, f.roi_origen = hits[0]
                    if len({h[0] for h in hits}) > 1:
                        avisos_roi.append(f"«{v.get('nombre')}» tiene {len(hits)} corridas con ROI distintos; se usa el de {hits[0][1]}.")
                fuentes.append(f)

    prefs = [f.prefijo for f in fuentes]
    if len(set(prefs)) != len(prefs):  # dos videos con el mismo nombre y sin huella: se numeran
        cuenta: Dict[str, int] = {}
        for f in fuentes:
            cuenta[f.prefijo] = cuenta.get(f.prefijo, 0) + 1
            if cuenta[f.prefijo] > 1:
                f.prefijo = f"{f.prefijo}_{cuenta[f.prefijo]}"

    # 2) el mismo video no puede estar dos veces ni en entrenamiento y validación a la vez
    por_huella: Dict[str, Fuente] = {}
    for f in fuentes:
        if f.huella:
            if f.huella in por_huella:
                a = por_huella[f.huella]
                out(f"Los datasets {a.ruta} y {f.ruta} son del MISMO video"
                    + (" (uno de entrenamiento y otro de validación: la validación sería una fuga)." if a.rol != f.rol else "."))
                return 2
            por_huella[f.huella] = f

    # 3) protección del video reservado
    hay_auto = any(f.tipo == "auto" for f in fuentes)
    advertencias: List[str] = ([ADVERTENCIAS[ADV_AUTOMATICAS], ADVERTENCIAS[ADV_BORDE]] if hay_auto else []) + [ADVERTENCIAS[ADV_POSITIVOS]] + avisos_roi
    if excluir_videos_de:
        try:
            reservadas = reserved_fingerprints(excluir_videos_de)
        except (FileNotFoundError, ValueError, OSError) as exc:
            out(str(exc))
            return 2
        for f in fuentes:
            if f.huella and f.huella in reservadas:
                out(f"FUGA DE DATOS EVITADA: {f.ruta} es del video «{reservadas[f.huella]}», que está en el conjunto de validación "
                    "etiquetado a mano. Ese video se reserva para medir: no puede usarse para entrenar ni para validar.")
                return 2
            if not f.huella:
                advertencias.append(f"No se pudo comprobar si {f.ruta} es un video reservado (la ficha no trae su huella).")

    # 4) selección de imágenes por fuente
    usa_val_interno = not (valid or valid_manual)
    if usa_val_interno and any(f.tipo == "auto" and f.rol == "train" for f in fuentes):
        advertencias.append("No se indicó --valid: la validación sale de bloques de tiempo de los MISMOS videos de entrenamiento. "
                            "Es optimista; para una validación honesta usa un video aparte con --valid.")
    plan: List[Tuple[Fuente, str, str, str, str, Optional[Dict[str, Any]]]] = []  # (fuente, destino, nombre, imagen, etiqueta, decisión de la revisión)
    pre_omit: Dict[str, Dict[str, int]] = {f.ruta: {} for f in fuentes}
    for f in fuentes:
        items = f.items if f.tipo == "manual" else list_items(f.ruta, ("train", "val"))
        if f.tipo == "manual" or f.rol == "valid":
            elegidos = [(f.rol, sp, n, i, l) for sp, n, i, l in (items or [])]
        elif usa_val_interno:
            elegidos = [("train" if sp == "train" else "valid", sp, n, i, l) for sp, n, i, l in items]
        else:
            elegidos = [("train", sp, n, i, l) for sp, n, i, l in items]
        # la revisión manda: descartadas y dudosas no entran (ni las pendientes, con --solo-revisadas). Esto va ANTES
        # del tope por video, para que el tope se reparta entre imágenes que sí se pueden usar.
        vigentes: List[Tuple[str, str, str, str, Optional[Dict[str, Any]]]] = []
        for dest, sp, n, i, l in elegidos:
            entrada = ((f.revision or {}).get("imagenes") or {}).get(f"{sp}/{n}")
            estado = entrada["estado"] if entrada else PENDING
            if estado in EXCLUDED:
                motivo = f"{estado}_en_la_revision"
            elif estado == PENDING and solo_revisadas:
                motivo = "sin_revisar"
            else:
                vigentes.append((dest, n, i, l, entrada))
                continue
            pre_omit[f.ruta][motivo] = pre_omit[f.ruta].get(motivo, 0) + 1
        if max_por_video is not None:
            por_destino: Dict[str, List[Tuple[str, str, str, str, Optional[Dict[str, Any]]]]] = {}
            for e in vigentes:
                por_destino.setdefault(e[0], []).append(e)
            vigentes = [x for dest in por_destino.values() for x in evenly(dest, max_por_video)]
        for dest, n, i, l, entrada in vigentes:
            plan.append((f, dest, n, i, l, entrada))
    if not any(p[1] == "train" for p in plan):
        out("No hay imágenes de entrenamiento en las carpetas indicadas.")
        return 2
    if not any(p[1] == "valid" for p in plan):
        out("No hay imágenes de validación: indica un video con --valid o --valid-manual, o usa conjuntos que traigan su propia separación (images/val).")
        return 2

    # 5) carpeta de salida
    if os.path.exists(salida) and os.listdir(salida):
        if not sobrescribir:
            out(f"{salida} ya existe y no está vacía. Usa otra carpeta o --sobrescribir (solo rehace train/, valid/ y los archivos de este conjunto).")
            return 2
        for sub in ("train", "valid"):
            shutil.rmtree(os.path.join(salida, sub), ignore_errors=True)
        for fn in ("data.yaml", "conjunto_info.json", "LEEME.txt"):
            if os.path.exists(os.path.join(salida, fn)):
                os.remove(os.path.join(salida, fn))
    for dest in ("train", "valid"):
        os.makedirs(os.path.join(salida, dest, "images"), exist_ok=True)
        os.makedirs(os.path.join(salida, dest, "labels"), exist_ok=True)

    # 6) escribir
    conteo: Dict[str, Dict[str, int]] = {f.ruta: {"train": 0, "valid": 0, "omitidas": 0} for f in fuentes}
    motivos: Dict[str, Dict[str, int]] = {f.ruta: {} for f in fuentes}

    usadas: Dict[str, Dict[str, int]] = {f.ruta: {"aceptadas": 0, "corregidas": 0, "sin_revisar": 0} for f in fuentes}
    a_mano: Dict[str, int] = {f.ruta: 0 for f in fuentes}  # imágenes de fuentes etiquetadas a mano que sí se usaron
    for ruta, d in pre_omit.items():  # lo que la revisión dejó fuera también cuenta como omitido
        for motivo, k in d.items():
            conteo[ruta]["omitidas"] += k
            motivos[ruta][motivo] = k

    def omitir(f: Fuente, motivo: str) -> None:
        conteo[f.ruta]["omitidas"] += 1
        motivos[f.ruta][motivo] = motivos[f.ruta].get(motivo, 0) + 1

    for f, dest, nombre, img_path, lab_path, entrada in plan:
        if entrada is None and not os.path.exists(lab_path):  # con una caja corregida a mano no hace falta la etiqueta original
            omitir(f, "sin_etiqueta")
            continue
        img = cv2.imread(img_path)
        if img is None:
            omitir(f, "imagen_ilegible")
            continue
        fh, fw = img.shape[:2]
        try:
            if entrada is not None:
                lineas = [yolo_label_line(tuple(entrada["final"]), fw, fh)]
            else:
                with open(lab_path, encoding="utf-8-sig") as fh_:
                    lineas = [ln for ln in fh_.read().splitlines() if ln.strip()]
            if len(lineas) != 1:
                if f.tipo == "manual":  # a mano: «sin ratón» y «varias cajas» son decisiones distintas
                    omitir(f, "sin_raton" if not lineas else "varias_cajas")
                else:
                    omitir(f, "etiqueta_con_cero_o_varias_cajas")
                continue
            rect = crop_rect(f.roi_box, margen_roi, fw, fh) if (recorte_roi and f.roi_box) else None
            nueva = convert_label(lineas[0], fw, fh, rect)
        except ValueError:
            omitir(f, "etiqueta_invalida")
            continue
        stem = f"{f.prefijo}_{nombre}"
        destino_img = os.path.join(salida, dest, "images", stem + ".jpg")
        if rect is not None:
            x0, y0, x1, y1 = rect
            ok = cv2.imwrite(destino_img, img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                omitir(f, "no_se_pudo_escribir")
                continue
        elif img_path.lower().endswith((".jpg", ".jpeg")):
            shutil.copyfile(img_path, destino_img)  # sin recortar: se copia el archivo tal cual, sin recomprimir
        else:
            cv2.imwrite(destino_img, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        with open(os.path.join(salida, dest, "labels", stem + ".txt"), "w", encoding="utf-8") as fh_:
            fh_.write(nueva + "\n")
        conteo[f.ruta][dest] += 1
        if f.tipo == "manual":
            a_mano[f.ruta] += 1
        elif entrada is None:
            usadas[f.ruta]["sin_revisar"] += 1
        else:
            usadas[f.ruta]["corregidas" if entrada["estado"] == CORRECTED else "aceptadas"] += 1

    n_train = sum(c["train"] for c in conteo.values())
    n_valid = sum(c["valid"] for c in conteo.values())
    if n_train == 0 or n_valid == 0:
        out(f"El conjunto quedó incompleto (entrenamiento: {n_train}, validación: {n_valid}). Revisa los motivos de omisión.")
        for f in fuentes:
            if motivos[f.ruta]:
                out(f"  {f.ruta}: {motivos[f.ruta]}")
        return 2
    tot_sin = sum(u["sin_revisar"] for u in usadas.values())
    tot_cor = sum(u["corregidas"] for u in usadas.values())
    n_man = sum(a_mano.values())
    if not hay_auto:
        advertencias.insert(0, f"Todas las etiquetas están dibujadas a mano ({n_man} imágenes): su calidad depende de aplicar siempre el mismo criterio "
                               "(caja sobre el ratón y no sobre su reflejo, ajustada al cuerpo, sin la cola).")
    elif tot_sin == 0:  # todo lo automático que se usa pasó por una persona: ya no son etiquetas solo automáticas
        advertencias.remove(ADVERTENCIAS[ADV_AUTOMATICAS])
        advertencias.insert(0, f"Todas las imágenes usadas fueron revisadas o dibujadas a mano ({tot_cor} con la caja corregida, {n_man} dibujadas a mano): convención de cuerpo sin cola.")
    else:
        advertencias.append(f"{tot_sin} de {n_train + n_valid} imágenes usadas no se revisaron: llevan la caja automática tal cual.")
    if n_train < 100:
        advertencias.append(f"Solo hay {n_train} imágenes de entrenamiento: probablemente pocas; la variedad de videos pesa más que la cantidad.")

    with open(os.path.join(salida, "data.yaml"), "w", encoding="utf-8") as fh_:
        fh_.write(f"train: train/images\nval: valid/images\nnc: 1\nnames:\n  - {CLASS_NAME}\n")
    ficha = {
        "creado": datetime.now().isoformat(timespec="seconds"), "clase": CLASS_NAME,
        "recorte_roi": bool(recorte_roi), "margen_roi_px": int(margen_roi) if recorte_roi else None,
        "validacion_aparte": bool(valid), "max_por_video": max_por_video, "solo_revisadas": bool(solo_revisadas),
        "imagenes": {"train": n_train, "valid": n_valid},
        "fuentes": [{"ruta": f.ruta, "tipo": f.tipo, "rol": f.rol, "prefijo": f.prefijo, **({"etiquetas_a_mano": a_mano[f.ruta]} if f.tipo == "manual" else {}), "video": (f.info.get("video") or {}).get("nombre"),
                     "huella": f.huella, "roi_box": list(f.roi_box) if f.roi_box else None, "roi_origen": f.roi_origen or None,
                     "modo_de_seguimiento": f.info.get("modo_de_seguimiento"), **conteo[f.ruta], "motivos_de_omision": motivos[f.ruta],
                     "revision": {"tiene_revision": f.revision is not None, **{"usadas_" + k: v for k, v in usadas[f.ruta].items()}}}
                    for f in fuentes],
        "videos_reservados_excluidos_de": list(excluir_videos_de), "advertencias": advertencias,
    }
    _write_json(os.path.join(salida, "conjunto_info.json"), ficha)
    with open(os.path.join(salida, "LEEME.txt"), "w", encoding="utf-8") as fh_:
        fh_.write("Conjunto de entrenamiento para RF-DETR (formato YOLO: data.yaml + train/valid con images y labels).\n"
                  "Para Colab: comprime esta carpeta en un .zip, súbela y descomprímela; pasa su ruta a model.train(dataset_dir=...).\n"
                  "Lee conjunto_info.json: ahí están el origen de cada imagen y las advertencias.\n")

    out(f"\n{'video (rol)':<42} {'train':>6} {'valid':>6} {'omitidas':>9}")
    for f in fuentes:
        c = conteo[f.ruta]
        out(f"{(f.prefijo + ' (' + f.rol + ')')[:42]:<42} {c['train']:>6} {c['valid']:>6} {c['omitidas']:>9}")
    out(f"{'TOTAL':<42} {n_train:>6} {n_valid:>6}")
    out(f"\nRecorte al ROI: {'sí, con margen de ' + str(margen_roi) + ' px' if recorte_roi else 'no (frames completos)'}")
    for a in advertencias:
        out(f"AVISO: {a}")
    out(f"\nConjunto listo en {salida}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Arma el conjunto de entrenamiento de RF-DETR desde las carpetas dataset_<fecha>.")
    ap.add_argument("--train", action="append", default=[], help="carpeta dataset_<fecha> de un video de ENTRENAMIENTO (se puede repetir)")
    ap.add_argument("--train-manual", action="append", default=[], help="carpeta de frames ETIQUETADOS A MANO (sample_frames + label_frames) cuyos videos entrenan (se puede repetir)")
    ap.add_argument("--valid-manual", action="append", default=[], help="idem, para videos de VALIDACIÓN (se puede repetir)")
    ap.add_argument("--entradas", action="append", default=[], help="carpeta entradas_<fecha> de una corrida de cada video etiquetado a mano: de ahí sale el ROI para recortar (se puede repetir)")
    ap.add_argument("--valid", action="append", default=[], help="carpeta dataset_<fecha> de un video de VALIDACIÓN (se puede repetir)")
    ap.add_argument("--salida", default="entrenamiento", help="carpeta donde escribir (por omisión: entrenamiento)")
    ap.add_argument("--excluir-videos-de", action="append", default=[], help="conjunto de validación a mano (carpeta con manifest.json) cuyos videos NO deben usarse")
    ap.add_argument("--sin-recorte", action="store_true", help="no recortar al ROI: dejar el frame completo")
    ap.add_argument("--margen-roi", type=int, default=0, help="píxeles de margen por lado alrededor del ROI al recortar (por omisión 0)")
    ap.add_argument("--max-por-video", type=int, help="tope de imágenes por video y por división (train / valid), repartidas de forma pareja")
    ap.add_argument("--solo-revisadas", action="store_true", help="usar solo las imágenes aceptadas o corregidas en la revisión (tools/review_boxes.py)")
    ap.add_argument("--sobrescribir", action="store_true", help="rehacer una carpeta de salida que ya existe")
    a = ap.parse_args(argv)
    return build(a.train, a.valid, a.salida, recorte_roi=not a.sin_recorte, margen_roi=a.margen_roi,
                 excluir_videos_de=a.excluir_videos_de, max_por_video=a.max_por_video, sobrescribir=a.sobrescribir, solo_revisadas=a.solo_revisadas,
                 train_manual=a.train_manual, valid_manual=a.valid_manual, entradas=a.entradas)


if __name__ == "__main__":
    sys.exit(main())