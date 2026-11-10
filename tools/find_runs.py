from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.build_training_set import read_manual_manifest  # noqa: E402

DEFAULT_ROOT = os.path.join("~", "TrackerLab_Resultados")


@dataclass
class Run:
    carpeta: str
    experimento: str
    sello: str  # YYYYMMDD_HHMMSS (de la carpeta entradas_<sello>); ordena por fecha
    video_path: str = ""
    huella: Optional[str] = None
    roi: Optional[Tuple[int, int, int, int]] = None
    rango: Optional[Tuple[int, int]] = None
    paso: Optional[int] = None
    modo: str = ""
    error: str = ""

    @property
    def video(self) -> str:
        return os.path.basename(self.video_path)

    @property
    def fecha(self) -> str:
        s = self.sello
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]} {s[9:11]}:{s[11:13]}" if len(s) >= 13 and s[8] == "_" else s


# ----------------------------------------------------------------------
# Lectura
# ----------------------------------------------------------------------
def read_run(carpeta: str) -> Run:
    """Datos de una carpeta ``entradas_<sello>``. Nunca lanza: un archivo ilegible queda anotado en ``error``."""
    nombre = os.path.basename(os.path.normpath(carpeta))
    run = Run(carpeta, os.path.basename(os.path.dirname(os.path.normpath(carpeta))), nombre[len("entradas_"):] if nombre.startswith("entradas_") else nombre)
    p = os.path.join(carpeta, "inputs.json")
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        v = d.get("video") or {}
        run.video_path = str(v.get("path") or "")
        run.huella = v.get("sha256_extremos") or None
        roi = d.get("roi_box")
        if isinstance(roi, list) and len(roi) == 4:
            run.roi = tuple(int(x) for x in roi)  # type: ignore[assignment]
        fr = d.get("frame_range") or {}
        if "start" in fr and "end" in fr:
            run.rango = (int(fr["start"]), int(fr["end"]))
        run.paso = (d.get("timing") or {}).get("frame_step")
        run.modo = str((d.get("tracker") or {}).get("mode") or "")
        if run.roi is None:
            run.error = "no trae roi_box"
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        run.error = f"no se pudo leer inputs.json ({exc.__class__.__name__})"
    return run


def scan_runs(raiz: str) -> List[Run]:
    """Corridas de ``raiz/<experimento>/entradas_<fecha>`` (y de ``raiz/entradas_<fecha>``), de la más reciente a la más antigua."""
    raiz = os.path.expanduser(raiz)
    carpetas = sorted(set(glob.glob(os.path.join(glob.escape(raiz), "*", "entradas_*")) + glob.glob(os.path.join(glob.escape(raiz), "entradas_*"))))
    runs = [read_run(c) for c in carpetas if os.path.isdir(c)]
    return sorted(runs, key=lambda r: (r.sello, r.carpeta), reverse=True)


def match_runs(video: Dict[str, Any], runs: Sequence[Run]) -> List[Run]:
    """Corridas que son del video ``video`` (una entrada del manifiesto de sample_frames), de la más reciente a la más antigua.

    Misma regla que ``build_training_set.match_roi``: por huella; si no hay huella, por nombre de archivo.
    """
    validas = [r for r in runs if r.roi is not None and not r.error]
    huella = video.get("sha256_extremos")
    hits = [r for r in validas if huella and r.huella == huella]
    if not hits:
        nombre = video.get("nombre")
        hits = [r for r in validas if nombre and r.video == nombre and not (huella and r.huella)]
    return sorted(hits, key=lambda r: (r.sello, r.carpeta), reverse=True)


# ----------------------------------------------------------------------
# Salida
# ----------------------------------------------------------------------
def _fmt_roi(roi: Optional[Tuple[int, int, int, int]]) -> str:
    return "?" if roi is None else f"[{roi[0]} {roi[1]} {roi[2]} {roi[3]}]"


def format_runs(runs: Sequence[Run]) -> List[str]:
    out = [f"{'video':<22} {'experimento':<22} {'fecha':<16} {'ROI (x y ancho alto)':<22} {'rango (paso)':<20} modo"]
    for r in runs:
        rango = "?" if r.rango is None else f"{r.rango[0]}-{r.rango[1]} ({r.paso})"
        out.append(f"{r.video[:22] or '?':<22} {r.experimento[:22]:<22} {r.fecha:<16} {_fmt_roi(r.roi):<22} {rango:<20} {r.modo}" + (f"   ! {r.error}" if r.error else ""))
    return out


def build_command(train: Sequence[str], valid: Sequence[str], entradas: Sequence[str], reservados: Sequence[str], salida: str) -> str:
    """Comando de build_training_set.py con las rutas escritas con shlex.quote (sin comillas curvas ni rutas que pegar a mano)."""
    q = shlex.quote
    partes = ["python tools/build_training_set.py"]
    partes += [f"--train-manual {q(t)}" for t in train]
    partes += [f"--valid-manual {q(v)}" for v in valid]
    partes += [f"--entradas {q(e)}" for e in entradas]
    partes += [f"--excluir-videos-de {q(r)}" for r in reservados]
    partes += [f"--salida {q(salida)}"]
    return " \\\n  ".join(partes)


def plan(raiz: str, train: Sequence[str], valid: Sequence[str], reservados: Sequence[str], salida: Optional[str], out: Callable[[str], None]) -> int:
    runs = scan_runs(raiz)
    if not runs:
        out(f"No encontré ninguna corrida (entradas_<fecha>) en {os.path.expanduser(raiz)}. ¿Corriste TrackerLab con SAVE_RUN_INPUTS = True?")
        return 2
    out(f"Corridas encontradas en {os.path.expanduser(raiz)} ({len(runs)}):")
    for ln in format_runs(runs):
        out("  " + ln)
    if not (train or valid):
        return 0

    elegidas: List[str] = []
    faltan: List[str] = []
    vistos: Dict[str, str] = {}
    problemas = 0
    for rol, carpetas in (("entrenamiento", train), ("validación", valid)):
        for c in carpetas:
            try:
                man = read_manual_manifest(c)
            except (FileNotFoundError, ValueError, OSError) as exc:
                out(f"\n{exc}")
                return 2
            out(f"\nConjunto de {rol}: {c}")
            for v in man["videos"]:
                nombre = v.get("nombre") or v["id"]
                huella = v.get("sha256_extremos")
                if huella and huella in vistos:
                    out(f"  ! «{nombre}» ya figura en {vistos[huella]}: el mismo video no puede estar en dos conjuntos (la validación sería una fuga).")
                    problemas += 1
                if huella:
                    vistos[huella] = f"el conjunto de {rol} {c}"
                hits = match_runs(v, runs)
                if not hits:
                    out(f"  ✗ {nombre}: no hay ninguna corrida de este video. Corre TrackerLab sobre él (SAVE_RUN_INPUTS = True, ROI holgado) y vuelve a correr esto.")
                    faltan.append(nombre)
                    continue
                r = hits[0]
                out(f"  ✓ {nombre} ({len(v.get('frames', []))} frames) -> {r.carpeta}  ROI {_fmt_roi(r.roi)}" + (f"  [la más reciente de {len(hits)}]" if len(hits) > 1 else ""))
                if len({h.roi for h in hits}) > 1:
                    out(f"    AVISO: este video tiene {len(hits)} corridas con ROI distintos: " + "; ".join(f"{h.sello} {_fmt_roi(h.roi)}" for h in hits)
                        + ". Se eligió la más reciente; si quieres otra, cambia la ruta de --entradas en el comando.")
                if r.carpeta not in elegidas:
                    elegidas.append(r.carpeta)
    if faltan or problemas:
        out("\nNo se arma el comando todavía: " + (f"falta la corrida de {', '.join(faltan)}." if faltan else "resuelve lo marcado con !."))
        return 2
    destino = salida or os.path.join(os.path.expanduser(raiz), "entrenamiento")
    out("\nComando para armar el conjunto (cópialo entero, es un solo comando):\n")
    out(build_command(train, valid, elegidas, reservados, destino))
    if not reservados:
        out("\nAVISO: no indicaste --excluir-videos-de. Si tienes un video reservado para evaluar (con tus etiquetas a mano), pásalo con esa opción: así el armado se niega a mezclarlo con el entrenamiento.")
    return 0


def main(argv: Optional[List[str]] = None, out: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(description="Localiza las corridas de tus videos y arma el comando de build_training_set.py.")
    ap.add_argument("resultados", nargs="?", default=DEFAULT_ROOT, help=f"carpeta de resultados de TrackerLab (por omisión {DEFAULT_ROOT})")
    ap.add_argument("--train-manual", action="append", default=[], help="carpeta de frames etiquetados a mano para ENTRENAR (se puede repetir)")
    ap.add_argument("--valid-manual", action="append", default=[], help="idem, para VALIDAR (se puede repetir)")
    ap.add_argument("--excluir-videos-de", action="append", default=[], help="conjunto de validación a mano de un video RESERVADO (se puede repetir)")
    ap.add_argument("--salida", help="carpeta del conjunto de entrenamiento (por omisión, <resultados>/entrenamiento)")
    a = ap.parse_args(argv)
    return plan(a.resultados, a.train_manual, a.valid_manual, a.excluir_videos_de, a.salida, out)


if __name__ == "__main__":
    sys.exit(main())