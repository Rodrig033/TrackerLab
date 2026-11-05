"""Revisión a mano de las cajas automáticas: formato de ``revision.json`` y estadísticas.

Lo usan ``tools/review_boxes.py`` (que lo escribe) y ``tools/build_training_set.py`` (que lo aplica). Las etiquetas
originales de ``labels/`` NUNCA se modifican: las decisiones de la revisión viven solo aquí, en una sola fuente de verdad.

Formato (``dataset_<fecha>/revision.json``)::

    {"schema": 1, "creado": "...", "actualizado": "...", "convencion": "...",
     "imagenes": {"train/rataA_0000123": {"estado": "corregida", "original": [x, y, w, h], "final": [x, y, w, h]},
                  "val/rataA_0000456":   {"estado": "descartada", "original": [x, y, w, h], "final": null}}}

Las cajas son (x, y, ancho, alto) en píxeles del FRAME COMPLETO (no del recorte que se ve en la ventana).
Una imagen que no aparece en ``imagenes`` está pendiente.

Estados: ``aceptada`` (la caja automática estaba bien), ``corregida`` (la dibujaste tú), ``descartada`` (no se ve el
ratón o la caja no sirve), ``dudosa``. Descartadas y dudosas no se usan para entrenar.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

REVISION_FILE = "revision.json"
SCHEMA = 1
CONVENTION = "caja ajustada al cuerpo visible del ratón, sin la cola"

PENDING, ACCEPTED, CORRECTED, DISCARDED, DOUBTFUL = "pendiente", "aceptada", "corregida", "descartada", "dudosa"
STATES = (ACCEPTED, CORRECTED, DISCARDED, DOUBTFUL)  # los que se guardan en el archivo
EXCLUDED = (DISCARDED, DOUBTFUL)  # no se usan para entrenar
USABLE = (ACCEPTED, CORRECTED)  # llevan una caja final válida

Box = Tuple[float, float, float, float]


def revision_path(dsdir: str) -> str:
    return os.path.join(dsdir, REVISION_FILE)


def new_revision() -> Dict[str, Any]:
    ahora = datetime.now().isoformat(timespec="seconds")
    return {"schema": SCHEMA, "creado": ahora, "actualizado": ahora, "convencion": CONVENTION, "imagenes": {}}


def _valid_box(b: Any) -> bool:
    return (isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in b)
            and b[2] > 0 and b[3] > 0)


def load_revision(dsdir: str) -> Optional[Dict[str, Any]]:
    """La revisión guardada, o None si no hay. Lanza ``ValueError`` si el archivo está dañado (nunca lo ignora en silencio)."""
    p = revision_path(dsdir)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{p} está dañado o no se puede leer ({exc}). No se sobrescribe: revísalo o muévelo a otra carpeta.") from exc
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or not isinstance(data.get("imagenes"), dict):
        raise ValueError(f"{p} no tiene el formato esperado (schema {SCHEMA}).")
    for clave, e in data["imagenes"].items():
        if not isinstance(e, dict) or e.get("estado") not in STATES:
            raise ValueError(f"{p}: la imagen {clave!r} tiene un estado inválido.")
        if e.get("original") is not None and not _valid_box(e["original"]):
            raise ValueError(f"{p}: la imagen {clave!r} tiene una caja original inválida.")
        if e["estado"] in USABLE and not _valid_box(e.get("final")):
            raise ValueError(f"{p}: la imagen {clave!r} figura como {e['estado']} pero no tiene una caja final válida.")
        if e.get("final") is not None and not _valid_box(e["final"]):
            raise ValueError(f"{p}: la imagen {clave!r} tiene una caja final inválida.")
    return data


def save_revision(dsdir: str, rev: Dict[str, Any]) -> None:
    """Escribe de forma atómica (archivo temporal y reemplazo): un corte a mitad no deja el archivo a medias."""
    rev["actualizado"] = datetime.now().isoformat(timespec="seconds")
    p = revision_path(dsdir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rev, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Percentil ``q`` (0 a 1) con interpolación lineal, sobre valores ya ordenados."""
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo))


def review_stats(rev: Optional[Dict[str, Any]], total: Optional[int] = None) -> Dict[str, Any]:
    """Cuántas imágenes hay en cada estado y, de las corregidas, cuánto se equivocó la caja automática."""
    imgs = (rev or {}).get("imagenes", {})
    cuenta = {s: 0 for s in STATES}
    for e in imgs.values():
        cuenta[e["estado"]] += 1
    decididas = sum(cuenta.values())
    revisadas = cuenta[ACCEPTED] + cuenta[CORRECTED]
    st: Dict[str, Any] = {
        "total": total, "pendientes": (total - decididas) if total is not None else None, **cuenta,
        "revisadas": revisadas, "pct_corregidas": (100.0 * cuenta[CORRECTED] / revisadas) if revisadas else None,
        "iou_mediana": None, "iou_p10": None, "ancho_razon_mediana": None, "alto_razon_mediana": None,
        "area_razon_mediana": None, "desplazamiento_centro_mediana_px": None,
    }
    ious: List[float] = []
    anchos: List[float] = []
    altos: List[float] = []
    areas: List[float] = []
    mov: List[float] = []
    for e in imgs.values():
        if e["estado"] != CORRECTED or e.get("original") is None:
            continue
        o, f = e["original"], e["final"]
        ious.append(box_iou(o, f))
        anchos.append(f[2] / o[2])
        altos.append(f[3] / o[3])
        areas.append((f[2] * f[3]) / (o[2] * o[3]))
        mov.append(math.dist((o[0] + o[2] / 2, o[1] + o[3] / 2), (f[0] + f[2] / 2, f[1] + f[3] / 2)))
    if ious:
        st.update(iou_mediana=statistics.median(ious), iou_p10=_percentile(sorted(ious), 0.10), ancho_razon_mediana=statistics.median(anchos),
                  alto_razon_mediana=statistics.median(altos), area_razon_mediana=statistics.median(areas),
                  desplazamiento_centro_mediana_px=statistics.median(mov))
    st["n_comparadas"] = len(ious)
    return st


def format_stats(st: Dict[str, Any]) -> List[str]:
    partes = []
    if st["total"] is not None:
        partes.append(f"{st['total']} imágenes")
        partes.append(f"pendientes {st['pendientes']}")
    partes += [f"aceptadas {st[ACCEPTED]}", f"corregidas {st[CORRECTED]}", f"descartadas {st[DISCARDED]}", f"dudosas {st[DOUBTFUL]}"]
    out = ["Revisión: " + " | ".join(partes)]
    if st["revisadas"]:
        out.append(f"De las {st['revisadas']} revisadas, {st[CORRECTED]} ({st['pct_corregidas']:.0f} %) necesitaron corrección.")
    if st["iou_mediana"] is not None:
        out.append(f"En esas {st['n_comparadas']}: IoU mediano entre la caja automática y la corregida {st['iou_mediana']:.2f} (el 10 % peor: {st['iou_p10']:.2f}).")
        out.append(f"  La caja corregida es, en mediana, {st['ancho_razon_mediana']:.2f} veces el ancho y {st['alto_razon_mediana']:.2f} veces el alto de la automática; "
                   f"su centro se movió {st['desplazamiento_centro_mediana_px']:.1f} px.")
    return out