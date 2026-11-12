from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

STEM_FMT = "{vid}_f{frame:06d}"


def stem(vid: str, frame: int) -> str:
    return STEM_FMT.format(vid=vid, frame=int(frame))


def offsets_from_alignment(path: str) -> Dict[str, int]:
    """Desfase de cada video según un resultado de check_frame_alignment. ValueError si en algún video no es constante."""
    with open(path, encoding="utf-8") as fh:
        datos = json.load(fh)
    por_video: Dict[str, set] = {}
    for r in datos.get("imagenes", []):
        if r.get("clara") and r.get("desfase") is not None:
            por_video.setdefault(r["video"], set()).add(int(r["desfase"]))
    if not por_video:
        raise ValueError(f"{os.path.basename(path)} no tiene ninguna imagen con un desfase claro: no hay nada que aplicar.")
    out: Dict[str, int] = {}
    for vid, ds in por_video.items():
        if len(ds) != 1:
            raise ValueError(f"En {vid} el desfase no es constante ({sorted(ds)}): no es un corrimiento que se pueda renumerar.")
        out[vid] = ds.pop()
    return out


def plan(manifest: Dict[str, Any], offsets: Dict[str, int]) -> List[Tuple[str, int, int]]:
    """(video, frame viejo, frame nuevo) de cada imagen a renumerar. ValueError si algún número nuevo no es válido."""
    cambios: List[Tuple[str, int, int]] = []
    for v in manifest["videos"]:
        d = offsets.get(v["id"], 0)
        if d == 0:
            continue
        total = int(v.get("total_frames") or 0)
        for f in sorted(int(x) for x in v["frames"]):
            nuevo = f + d
            if nuevo < 0 or (total and nuevo >= total):
                raise ValueError(f"El frame {f} de {v.get('nombre', v['id'])} quedaría en {nuevo}, fuera del video (0 a {total - 1 if total else '?'}).")
            cambios.append((v["id"], f, nuevo))
    return cambios


def _files_of(valdir: str, vid: str, frame: int) -> List[str]:
    s = stem(vid, frame)
    return sorted(glob.glob(os.path.join(glob.escape(valdir), "images", glob.escape(s) + ".*"))) + \
        sorted(glob.glob(os.path.join(glob.escape(valdir), "labels", glob.escape(s) + ".txt")))


def apply(valdir: str, manifest: Dict[str, Any], offsets: Dict[str, int], cambios: List[Tuple[str, int, int]], motivo: str,
          now: Callable[[], datetime]) -> None:
    """Renombra en dos pasos (primero a nombres temporales) para que dos frames vecinos no se pisen."""
    temporales: List[Tuple[str, str]] = []
    for vid, viejo, nuevo in cambios:
        for ruta in _files_of(valdir, vid, viejo):
            carpeta, nombre = os.path.split(ruta)
            ext = nombre[len(stem(vid, viejo)):]
            tmp = os.path.join(carpeta, f".renumerando_{stem(vid, nuevo)}{ext}")
            os.rename(ruta, tmp)
            temporales.append((tmp, os.path.join(carpeta, stem(vid, nuevo) + ext)))
    for tmp, final in temporales:
        os.rename(tmp, final)

    dud = os.path.join(valdir, "dudosas.txt")
    if os.path.exists(dud):
        mapa = {stem(vid, viejo): stem(vid, nuevo) for vid, viejo, nuevo in cambios}
        with open(dud, encoding="utf-8") as fh:
            lineas = fh.read().splitlines()
        with open(dud, "w", encoding="utf-8") as fh:
            fh.write("".join(mapa.get(ln.strip(), ln.strip()) + "\n" for ln in lineas if ln.strip()))

    for v in manifest["videos"]:
        d = offsets.get(v["id"], 0)
        if d == 0:
            continue
        v["frames"] = sorted(int(f) + d for f in v["frames"])
        if v.get("rango"):
            v["rango"] = [max(0, int(v["rango"][0]) + d), max(0, int(v["rango"][1]) + d)]
        v.setdefault("renumerado", []).append({"desfase": d, "fecha": now().strftime("%Y-%m-%d %H:%M:%S"), "motivo": motivo})
    tmp = os.path.join(valdir, ".manifest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, os.path.join(valdir, "manifest.json"))
    for viejo in glob.glob(os.path.join(glob.escape(valdir), "alineacion*.json")):
        os.remove(viejo)  # medían la numeración vieja; están en la copia de seguridad


def run(valdir: str, *, desfase: Optional[int] = None, desde_alineacion: Optional[str] = None, video: Optional[str] = None,
        aplicar: bool = False, out: Callable[[str], None] = print, now: Callable[[], datetime] = datetime.now) -> int:
    ruta_man = os.path.join(valdir, "manifest.json")
    if not os.path.exists(ruta_man):
        out(f"{valdir} no es un conjunto de frames etiquetados: falta manifest.json.")
        return 2
    if (desfase is None) == (desde_alineacion is None):
        out("Indica el desfase con --desde-alineacion (recomendado) o con --desfase, uno de los dos.")
        return 2
    with open(ruta_man, encoding="utf-8") as fh:
        manifest = json.load(fh)
    ids = [v["id"] for v in manifest["videos"]]
    try:
        if desde_alineacion is not None:
            offsets = offsets_from_alignment(desde_alineacion)
            motivo = f"desfase medido en {os.path.basename(desde_alineacion)}"
        else:
            if video is None and len(ids) > 1:
                out(f"El conjunto tiene varios videos ({', '.join(ids)}): indica a cuál aplicar el desfase con --video.")
                return 2
            offsets = {video or ids[0]: int(desfase)}  # type: ignore[arg-type]
            motivo = "desfase indicado a mano"
        desconocidos = [k for k in offsets if k not in ids]
        if desconocidos:
            out(f"{', '.join(desconocidos)} no está en este conjunto (videos: {', '.join(ids)}).")
            return 2
        cambios = plan(manifest, offsets)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        out(str(exc))
        return 2
    if not cambios:
        out("El desfase es 0: no hay nada que renumerar.")
        return 0
    for vid, d in offsets.items():
        if d:
            n = sum(1 for c in cambios if c[0] == vid)
            out(f"{vid}: {n} imágenes; cada frame N pasa a N {'+' if d > 0 else '-'} {abs(d)} (la imagen rotulada «N» es en realidad el frame N {d:+d}).")
    for vid, viejo, nuevo in cambios[:5]:
        out(f"  {stem(vid, viejo)}  ->  {stem(vid, nuevo)}")
    if len(cambios) > 5:
        out(f"  ... y {len(cambios) - 5} más")
    if not aplicar:
        out("\nNo se cambió nada. Para aplicarlo, repite el comando con --aplicar (antes se hace una copia de seguridad completa).")
        return 0
    respaldo = f"{os.path.normpath(valdir)}_antes_de_renumerar_{now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copytree(valdir, respaldo)
    except OSError as exc:
        out(f"No se pudo hacer la copia de seguridad ({exc}); no se cambió nada.")
        return 2
    try:
        apply(valdir, manifest, offsets, cambios, motivo, now)
    except OSError as exc:
        out(f"Falló a medio camino ({exc}). El conjunto original está intacto en {respaldo}: bórralo y renombra la copia.")
        return 2
    out(f"\nRenumerado. Copia de seguridad: {respaldo}")
    out("Comprueba con check_frame_alignment.py (con las mismas --entradas): debería decir ALINEADO.")
    return 0


def main(argv: Optional[List[str]] = None, **kw: Any) -> int:
    ap = argparse.ArgumentParser(description="Renumera un conjunto de frames etiquetados corrido un desfase constante.")
    ap.add_argument("conjunto", help="carpeta hecha con sample_frames.py")
    ap.add_argument("--desde-alineacion", help="alineacion_<corrida>.json de check_frame_alignment.py (toma de ahí el desfase)")
    ap.add_argument("--desfase", type=int, help="desfase a mano: la imagen rotulada N es el frame N + desfase")
    ap.add_argument("--video", help="con --desfase, a qué video del conjunto aplicarlo (si hay varios)")
    ap.add_argument("--aplicar", action="store_true", help="aplicar los cambios (sin esto solo se muestra el plan)")
    a = ap.parse_args(argv)
    return run(a.conjunto, desfase=a.desfase, desde_alineacion=a.desde_alineacion, video=a.video, aplicar=a.aplicar, **kw)


if __name__ == "__main__":
    sys.exit(main())