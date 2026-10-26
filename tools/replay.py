from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Any, Callable, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.quality import assess_quality, format_quality  # noqa: E402
from learning.dataset import video_fingerprint  # noqa: E402
from persistence.run_inputs import KEEP, build_job, load_run_inputs  # noqa: E402
from pipeline import run_tracking  # noqa: E402
from video.reader import VideoReader  # noqa: E402

_KINEMATIC_HEADERS = ("dist_px", "dist_cm", "dist_acum_px", "dist_acum_cm", "vel_px_s", "vel_cm_s",
                      "acel_px_s2", "acel_cm_s2", "ang_deg", "vel_angular_deg_s")
_CELLS_PER_ZONE = 10


def _f(value: Any, nd: int) -> Any:
    """Texto con ``nd`` decimales; None -> celda vacía."""
    return "" if value is None else f"{float(value):.{nd}f}"


def _zone_headers(name: str) -> List[str]:
    return [f"en_{name}", f"dist_centro_{name}_px", f"dist_centro_{name}_cm", f"dist_perimetro_{name}_px",
            f"dist_perimetro_{name}_cm", f"tiempo_acum_{name}_s", f"frames_acum_{name}", f"entradas_acum_{name}",
            f"lat_primera_entrada_{name}_s", f"frame_primera_entrada_{name}"]


def frame_csv_headers(zone_names: Sequence[str]) -> List[str]:
    # Como la aplicación: las cuatro columnas relativas SIEMPRE están; quedan vacías si no hay origen o escala.
    out = ["frame", "time_s", "cx_abs", "cy_abs_inv", "x_rel_px", "y_rel_px", "x_rel_cm", "y_rel_cm"]
    for name in zone_names:
        out += _zone_headers(name)
    return out + list(_KINEMATIC_HEADERS)


def frame_csv_row(rec, zone_names: Sequence[str]) -> List[Any]:
    """Fila del CSV para un FrameRecord (también cuando el sujeto se perdió: celdas vacías)."""
    out: List[Any] = [rec.frame, _f(rec.time_s, 6), _f(rec.cx_abs, 3), _f(rec.cy_abs_inv, 3),
                      _f(rec.x_rel_px, 3), _f(rec.y_rel_px, 3), _f(rec.x_rel_cm, 4), _f(rec.y_rel_cm, 4)]
    if zone_names:
        if not rec.found:
            out += [""] * (_CELLS_PER_ZONE * len(zone_names))
        else:
            got = tuple(z.name for z in rec.zones)
            if got != tuple(zone_names):
                raise ValueError(f"Las zonas del registro {got} no coinciden con las del CSV {tuple(zone_names)}.")
            for z in rec.zones:
                out += [z.inside, _f(z.dist_center_px, 4), _f(z.dist_center_cm, 4), _f(z.dist_perim_px, 4), _f(z.dist_perim_cm, 4),
                        _f(z.time_acum_s, 4), z.frames_acum, z.entries_acum, _f(z.lat_first_s, 4), "" if z.frame_first is None else z.frame_first]
    out += [_f(rec.dist_px, 4), _f(rec.dist_cm, 4), _f(rec.dist_acum_px, 4), _f(rec.dist_acum_cm, 4), _f(rec.vel_px_s, 4),
            _f(rec.vel_cm_s, 4), _f(rec.acel_px_s2, 4), _f(rec.acel_cm_s2, 4), _f(rec.ang_deg, 4), _f(rec.vel_angular_deg_s, 4)]
    return out


def write_frames_csv(path: str, records: Sequence[Any], zones: Sequence[Any]) -> None:
    """Escribe el CSV por frame (con BOM, igual que el de la aplicación)."""
    names = tuple(z.name for z in zones)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(frame_csv_headers(names))
        for rec in records:
            w.writerow(frame_csv_row(rec, names))


def run(
    carpeta: str,
    *,
    video: Optional[str] = None,
    salida: Optional[str] = None,
    sin_filtros: bool = False,
    sin_busqueda: bool = False,
    sin_seleccion_margen: bool = False,
    alpha: Optional[float] = None,
    salto_px: Optional[float] = None,
    comparar: Optional[str] = None,
    tol: float = 1e-9,
    out: Callable[[str], None] = print,
    now: Optional[Callable[[], datetime]] = None,
) -> int:
    try:
        inputs = load_run_inputs(carpeta)
    except (FileNotFoundError, ValueError, OSError) as exc:
        out(f"No se pudieron leer las entradas: {exc}")
        return 2

    path = video or inputs.video_path
    if not os.path.exists(path):
        out(f"No se encuentra el video: {path}")
        out("  Si lo moviste de lugar, indica su ruta con --video.")
        return 2
    grabada = inputs.data["video"].get("sha256_extremos")
    if grabada and video_fingerprint(path).get("sha256_extremos") != grabada:
        out("AVISO: este video no es idéntico al de la corrida grabada (otra copia o recodificado): "
            "los resultados podrían no coincidir.")

    overrides = {}
    if sin_filtros:
        overrides.update(smoothing_alpha=None, jump_max_px=None, jump_max_speed_px_s=None)
    if alpha is not None:
        overrides["smoothing_alpha"] = alpha
    if salto_px is not None:
        overrides["jump_max_px"] = salto_px
    if sin_busqueda:
        overrides["reacquire"] = False
    if sin_seleccion_margen:
        overrides["border_aware"] = False

    stamp = (now or datetime.now)().strftime("%Y%m%d_%H%M%S")
    salida = salida or os.path.join(carpeta, f"replay_{stamp}")
    try:
        with VideoReader(path) as reader:
            job = build_job(inputs, reader, **overrides)
            total = max(1, job.frame_range.count(job.timing.frame_step))
            hecho = {"n": 0, "marca": -1}

            def progreso(_rec) -> None:
                hecho["n"] += 1
                decena = int(10 * hecho["n"] / total)
                if decena != hecho["marca"] and decena % 2 == 0:
                    hecho["marca"] = decena
                    out(f"  procesando... {min(100, decena * 10)} %")

            out(f"Repitiendo la corrida ({inputs.mode}, {total} frames) con {os.path.basename(path)}")
            result = run_tracking(job, on_record=progreso, keep_records=True, log=None)
    except (ValueError, RuntimeError, OSError) as exc:
        out(f"No se pudo repetir la corrida: {exc}")
        return 2

    os.makedirs(salida, exist_ok=True)
    csv_path = os.path.join(salida, "replay_FRAMES.csv")
    write_frames_csv(csv_path, result.records, job.zones)
    informe = assess_quality(result.records, result.timing.dt)
    s = result.settings
    resumen: List[str] = [
        f"frames procesados: {result.frames_processed} | con posición válida: {result.frames_found} | "
        f"rechazados: {result.frames_rejected} | recuperados: {result.frames_recovered} (reencontrados: {result.frames_reacquired})",
        f"opciones: suavizado={s['suavizado_alpha']} salto_max_px={s['salto_max_px']} búsqueda={s['buscador_sujeto']} "
        f"selección_independiente_del_margen={s['seleccion_independiente_del_margen']}",
        format_quality(informe),
    ]
    with open(os.path.join(salida, "replay_resumen.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(resumen) + "\n")
    out("")
    for linea in resumen:
        out(linea)
    out(f"\nResultados en: {salida}")

    if comparar:
        if not os.path.exists(comparar):
            out(f"No se encuentra el CSV a comparar: {comparar}")
            return 2
        try:
            from tools.compare_csv import compare_frame_csv, format_report
        except ImportError:
            out("Falta tools/compare_csv.py, que --comparar necesita.")
            return 2
        cmp = compare_frame_csv(comparar, csv_path, tol=tol)
        out("")
        out(format_report(cmp))
        return 0 if cmp.ok else 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Repite una corrida de TrackerLab sin ventanas, desde las entradas grabadas.")
    ap.add_argument("carpeta", help="carpeta entradas_... de la corrida")
    ap.add_argument("--video")
    ap.add_argument("--salida")
    ap.add_argument("--sin-filtros", action="store_true")
    ap.add_argument("--sin-busqueda", action="store_true")
    ap.add_argument("--sin-seleccion-margen", action="store_true")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--salto-px", type=float)
    ap.add_argument("--comparar")
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args(argv)
    return run(a.carpeta, video=a.video, salida=a.salida, sin_filtros=a.sin_filtros, sin_busqueda=a.sin_busqueda,
               sin_seleccion_margen=a.sin_seleccion_margen, alpha=a.alpha, salto_px=a.salto_px, comparar=a.comparar, tol=a.tol)


if __name__ == "__main__":
    sys.exit(main())