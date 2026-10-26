"""Entradas de una corrida: grabarlas y repetir el seguimiento sin ventanas."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from analysis.spatial_metrics import ZoneGeometry
from learning.dataset import video_fingerprint
from pipeline import TrackingJob, _mode_name
from tracking.background_tracker import BackgroundTracker, compute_background
from tracking.color_tracker import ColorTracker
from tracking.reacquisition import SilhouetteReacquirer
from tracking.silhouette_tracker import SilhouetteTracker
from video.frame_selection import FrameRange, compute_timing, read_background_frames

__all__ = ["SCHEMA", "KEEP", "RunInputs", "save_run_inputs", "load_run_inputs", "build_job"]

SCHEMA = 1
#: Opciones del launcher que se graban (el resto, como el ``root`` de Tk, no es serializable ni hace falta).
OPTION_KEYS = ("exp_name", "csv_base_name", "custom_tracking_fps", "tracking_fps", "tracking_mode",
               "polarity", "apply_bw", "use_clahe", "mutually_exclusive_zones")
PIPELINE_KEYS = ("smoothing_alpha", "jump_max_px", "jump_max_speed_px_s", "recovery_after", "recovery_distance_px",
                 "adopt_after_s", "reacquire_after_frames", "reacquire_every")


class _Keep:
    def __repr__(self) -> str:  # pragma: no cover
        return "KEEP"


#: Valor por omisión de las opciones de ``build_job``: "conservar lo grabado" (distinto de None = apagado).
KEEP = _Keep()


def _plain(value: Any) -> Any:
    """Convierte a tipos de Python puros (listas, números, texto) para escribir JSON."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_png(path: str, mask: np.ndarray) -> None:
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError(f"La máscara de {os.path.basename(path)} debe ser uint8 2D (es {mask.dtype}, {mask.shape}).")
    if not cv2.imwrite(path, mask):
        raise OSError(f"OpenCV no pudo escribir {path}")


def _read_png(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"No se pudo leer {path}")
    return img


# Grabar

def save_run_inputs(
    base_dir: str,
    job: TrackingJob,
    *,
    video_path: str,
    options: Dict[str, Any],
    sample_idx: int,
    app_version: str = "",
    log: Optional[Callable[[str], None]] = print,
    now: Optional[Callable[[], datetime]] = None,
) -> Optional[str]:
    say = log if log is not None else (lambda _m: None)
    try:
        tr = job.tracker
        mode = _mode_name(tr)
        spec: Dict[str, Any] = {"mode": mode}
        if mode == "fondo":
            spec.update(params=_plain(tr.params), polarity=tr.polarity, apply_bw=bool(tr.apply_bw), use_clahe=bool(tr.use_clahe))
        elif mode == "silueta":
            spec.update(border_aware=bool(tr.border_aware), model={"polarity": tr.model.polarity, "area_ref": float(tr.model.area_ref)})
        elif mode == "color":
            spec.update(params=_plain(tr.params))
        else:
            raise ValueError(f"tracker no soportado para repetir la corrida: {mode}")
        if job.start is None:
            raise ValueError("el trabajo no tiene posición inicial (job.start)")

        stamp = (now or datetime.now)().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(base_dir, f"entradas_{stamp}")
        os.makedirs(folder, exist_ok=True)
        _write_png(os.path.join(folder, "roi_mask.png"), tr.include_mask)
        zones = []
        for i, z in enumerate(job.zones):
            _write_png(os.path.join(folder, f"zone_{i}.png"), z.mask)
            zones.append({"name": z.name, "centroid": _plain(z.centroid), "mask": f"zone_{i}.png"})

        (center, bbox) = job.start
        data = {
            "schema": SCHEMA,
            "trackerlab_version": app_version,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "video": {"path": video_path, **video_fingerprint(video_path)} if os.path.exists(video_path) else {"path": video_path},
            "options": {k: _plain(options[k]) for k in OPTION_KEYS if k in options},
            "frame_range": {"start": job.frame_range.start, "end": job.frame_range.end},
            "sample_idx": int(sample_idx),
            "timing": {"frame_step": job.timing.frame_step, "dt": job.timing.dt, "video_fps": job.timing.video_fps},
            "roi_box": _plain(job.roi_box),
            "roi_mask": "roi_mask.png",
            "subject": {"center": _plain(center), "bbox": _plain(bbox)},
            "cm_per_px": job.cm_per_px,
            "xy_origin": _plain(job.xy_origin),
            "zones": zones,
            "tracker": spec,
            "pipeline": {**{k: getattr(job, k) for k in PIPELINE_KEYS},
                         "reacquirer": type(job.reacquirer).__name__ if job.reacquirer is not None else None},
        }
        with open(os.path.join(folder, "inputs.json"), "w", encoding="utf-8") as f:
            json.dump(_plain(data), f, indent=2, ensure_ascii=False)
        say(f"Entradas guardadas (para repetir esta corrida sin ventanas): {folder}")
        return folder
    except Exception as exc:
        say(f"No se pudieron guardar las entradas de la corrida ({type(exc).__name__}: {exc}); el seguimiento continúa.")
        return None


# Cargar
@dataclass
class RunInputs:
    folder: str
    data: Dict[str, Any]
    include_mask: np.ndarray
    zone_masks: List[np.ndarray] = field(default_factory=list)

    @property
    def video_path(self) -> str:
        return self.data["video"]["path"]

    @property
    def roi_box(self) -> Tuple[int, int, int, int]:
        return tuple(int(v) for v in self.data["roi_box"])  # type: ignore[return-value]

    @property
    def mode(self) -> str:
        return self.data["tracker"]["mode"]


def load_run_inputs(folder: str) -> RunInputs:
    """Lee una carpeta ``entradas_*``. Lanza ``FileNotFoundError`` / ``ValueError`` / ``OSError`` con un mensaje claro."""
    path = os.path.join(folder, "inputs.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró inputs.json en {folder}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"Versión de formato no soportada: {data.get('schema')} (se esperaba {SCHEMA}).")
    mask = _read_png(os.path.join(folder, data["roi_mask"]))
    zone_masks = [_read_png(os.path.join(folder, z["mask"])) for z in data["zones"]]
    return RunInputs(folder, data, mask, zone_masks)


# Reconstruir

def build_job(
    inputs: RunInputs,
    reader,
    *,
    smoothing_alpha: Any = KEEP,
    jump_max_px: Any = KEEP,
    jump_max_speed_px_s: Any = KEEP,
    recovery_after: Any = KEEP,
    recovery_distance_px: Any = KEEP,
    adopt_after_s: Any = KEEP,
    reacquire: Any = KEEP,
    reacquire_after_frames: Any = KEEP,
    reacquire_every: Any = KEEP,
    border_aware: Any = KEEP,
    strict: bool = True,
) -> TrackingJob:
    """Reconstruye el ``TrackingJob`` de la corrida grabada sobre ``reader`` (un ``VideoReader`` ya abierto).

    Cada opción por omisión conserva lo grabado; ``None`` apaga la etapa (por ejemplo ``smoothing_alpha=None``).
    ``reacquire=True/False`` enciende o apaga la búsqueda del sujeto (solo modo silueta).
    strict: si el modelo de silueta reconstruido no coincide con el grabado, lanza ``ValueError`` (False = solo continúa).
    """
    d = inputs.data
    opts = d["options"]
    rec_t = d["timing"]
    if abs(float(reader.fps) - float(rec_t["video_fps"])) > 1e-6:
        raise ValueError(f"El video tiene {reader.fps} FPS y la corrida grabada usó {rec_t['video_fps']}: no es el mismo video.")
    timing = compute_timing(reader.fps, custom_fps=bool(opts.get("custom_tracking_fps", False)), requested_fps=opts.get("tracking_fps"))
    if timing.frame_step != rec_t["frame_step"] or abs(timing.dt - rec_t["dt"]) > 1e-9:
        raise ValueError("La velocidad de procesamiento reconstruida no coincide con la grabada "
                         f"(frame_step {timing.frame_step} contra {rec_t['frame_step']}).")

    x, y, w, h = inputs.roi_box
    include_mask = inputs.include_mask
    if include_mask.shape != (h, w):
        raise ValueError(f"La máscara del ROI mide {include_mask.shape} y el ROI es {(h, w)}.")
    sample_idx = int(d["sample_idx"])
    bbox = tuple(d["subject"]["bbox"])
    center = tuple(d["subject"]["center"])
    spec = d["tracker"]
    mode = spec["mode"]
    pipe = d["pipeline"]

    use_border = spec.get("border_aware", False) if border_aware is KEEP else bool(border_aware)
    if mode == "fondo":
        bg = compute_background(read_background_frames(reader, sample_idx), (x, y, w, h), include_mask, bool(spec["use_clahe"]))
        if bg is None:
            raise RuntimeError("No se pudo calcular el fondo.")
        tracker = BackgroundTracker(include_mask, bg, dict(spec["params"]), polarity=spec["polarity"],
                                    apply_bw=bool(spec["apply_bw"]), use_clahe=bool(spec["use_clahe"]))
    elif mode == "silueta":
        frame = reader.read_at(sample_idx)
        if frame is None:
            raise RuntimeError(f"No se pudo leer el frame de referencia {sample_idx}.")
        tracker = SilhouetteTracker.from_bbox(include_mask, frame[y:y + h, x:x + w].copy(), bbox, border_aware=use_border)
        if tracker is None:
            raise RuntimeError("No se pudo crear el modelo de silueta.")
        rec_m = spec["model"]
        if use_border == spec.get("border_aware", False) and strict and (
                tracker.model.polarity != rec_m["polarity"] or abs(float(tracker.model.area_ref) - rec_m["area_ref"]) > 1e-6):
            raise ValueError("El modelo de silueta reconstruido no coincide con el grabado "
                             f"({tracker.model.polarity}, {tracker.model.area_ref:.1f} contra {rec_m['polarity']}, {rec_m['area_ref']:.1f}): "
                             "¿otro video o una decodificación distinta del frame de referencia?")
    elif mode == "color":
        tracker = ColorTracker(include_mask, dict(spec["params"]))
    else:
        raise ValueError(f"Modo de seguimiento desconocido: {mode}")

    def pick(name: str, value: Any) -> Any:
        return pipe[name] if value is KEEP else value

    want_reacquire = (pipe.get("reacquirer") is not None) if reacquire is KEEP else bool(reacquire)
    zones = [ZoneGeometry(z["name"], m, tuple(z["centroid"])) for z, m in zip(d["zones"], inputs.zone_masks)]
    return TrackingJob(
        reader=reader,
        frame_range=FrameRange(d["frame_range"]["start"], d["frame_range"]["end"]),
        timing=timing,
        tracker=tracker,
        roi_box=(x, y, w, h),
        zones=zones,
        cm_per_px=d["cm_per_px"],
        xy_origin=None if d["xy_origin"] is None else tuple(d["xy_origin"]),
        start=(center, bbox),
        smoothing_alpha=pick("smoothing_alpha", smoothing_alpha),
        jump_max_px=pick("jump_max_px", jump_max_px),
        jump_max_speed_px_s=pick("jump_max_speed_px_s", jump_max_speed_px_s),
        recovery_after=pick("recovery_after", recovery_after),
        recovery_distance_px=pick("recovery_distance_px", recovery_distance_px),
        adopt_after_s=pick("adopt_after_s", adopt_after_s),
        reacquirer=SilhouetteReacquirer(tracker) if (want_reacquire and mode == "silueta") else None,
        reacquire_after_frames=pick("reacquire_after_frames", reacquire_after_frames),
        reacquire_every=pick("reacquire_every", reacquire_every),
    )