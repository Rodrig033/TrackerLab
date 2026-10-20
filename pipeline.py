from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from analysis import FrameRecord
from analysis.kinematic_metrics import KinematicSample, KinematicSummary, KinematicTracker
from analysis.position_filter import JumpRejector, PositionEMAFilter
from analysis.quality import STATUS_DETECTED, STATUS_LOST, STATUS_RECOVERED, STATUS_REJECTED
from analysis.spatial_metrics import ZoneGeometry, ZoneSummary, ZoneTracker
from tracking import BaseTracker, Box, Detection, Point
from tracking.background_tracker import BackgroundTracker, compute_background
from tracking.color_tracker import ColorTracker, default_start
from tracking.reacquisition import Reacquirer
from tracking.silhouette_tracker import SilhouetteTracker
from video.frame_selection import FrameRange, TrackingTiming, frame_time_s, read_background_frames
from video.reader import VideoReader

__all__ = [
    "RoiBox",
    "TrackingJob",
    "Quit",
    "Reinit",
    "RecaptureBackground",
    "Command",
    "FrameEvent",
    "TrackingResult",
    "run_tracking",
]

RoiBox = Tuple[int, int, int, int]  # (x0, y0, w0, h0) del ROI dentro del frame

_SAMPLE_FIELDS = tuple(f.name for f in dataclasses.fields(KinematicSample))


# Entrada
@dataclass
class TrackingJob:
    reader: VideoReader
    frame_range: FrameRange
    timing: TrackingTiming
    tracker: BaseTracker
    roi_box: RoiBox
    zones: Sequence[ZoneGeometry] = ()
    cm_per_px: Optional[float] = None
    xy_origin: Optional[Tuple[float, float]] = None
    start: Optional[Tuple[Point, Box]] = None
    smoothing_alpha: Optional[float] = 0.35
    jump_max_px: Optional[float] = 120.0
    jump_max_speed_px_s: Optional[float] = None
    recovery_after: int = 5
    recovery_distance_px: float = 250.0
    adopt_after_s: Optional[float] = 2.0
    reacquirer: Optional[Reacquirer] = None
    reacquire_after_frames: int = 3
    reacquire_every: int = 5


# Órdenes que la interfaz puede devolver desde ``on_frame``
@dataclass(frozen=True)
class Quit:
    """Detener el tracking (tecla 'q')."""


@dataclass(frozen=True)
class RecaptureBackground:
    """Recapturar el fondo alrededor del frame actual (solo modo fondo)."""


@dataclass(frozen=True)
class Reinit:
    """Reiniciar el sujeto. Los datos están en coordenadas del ROI.

    ``Reinit()`` sin datos = el usuario canceló: no cambia al tracker, pero igual se olvida el
    movimiento previo (cinemática, zonas y filtros).
    """

    bbox: Optional[Box] = None
    center: Optional[Point] = None
    params: Optional[dict] = None


Command = Union[Quit, Reinit, RecaptureBackground]


# Salida
@dataclass(frozen=True, eq=False)
class FrameEvent:
    """Lo que recibe ``on_frame`` en cada frame. ``frame`` es de SOLO LECTURA: copiarlo para dibujar.

    ``detection`` es la detección CRUDA del tracker (coordenadas del recorte del ROI); la posición
    validada y filtrada está en ``record``.
    """

    frame_idx: int
    frame: np.ndarray
    roi_box: RoiBox
    detection: Detection
    record: FrameRecord
    processed: int
    total: int

    @property
    def roi(self) -> np.ndarray:
        x0, y0, w0, h0 = self.roi_box
        return self.frame[y0:y0 + h0, x0:x0 + w0]


@dataclass
class TrackingResult:
    frames_processed: int
    frames_found: int  # frames con posición VÁLIDA (excluye los rechazados)
    stopped_by_user: bool
    kinematic: KinematicSummary
    zones: Tuple[ZoneSummary, ...]
    timing: TrackingTiming
    frame_range: FrameRange
    warnings: List[str] = field(default_factory=list)
    records: List[FrameRecord] = field(default_factory=list)
    frames_rejected: int = 0  # el tracker vio algo y el filtro de saltos lo descartó
    frames_recovered: int = 0  # posiciones aceptadas por la regla de recuperación (incluye las reencontradas)
    frames_reacquired: int = 0  # de ellas, las encontradas por el buscador (reacquirer)
    # Con qué opciones se procesó (para registrarlo junto a los resultados). Los valores None significan
    # "apagado / no aplica". Ver ``analysis.provenance``.
    settings: Dict[str, Any] = field(default_factory=dict)


def _mode_name(tracker: BaseTracker) -> str:
    if isinstance(tracker, BackgroundTracker):
        return "fondo"
    if isinstance(tracker, SilhouetteTracker):
        return "silueta"
    if isinstance(tracker, ColorTracker):
        return "color"
    return type(tracker).__name__


# Validación e inicio
def _validate(job: TrackingJob) -> None:
    x0, y0, w0, h0 = job.roi_box
    if w0 <= 0 or h0 <= 0 or x0 < 0 or y0 < 0:
        raise ValueError(f"roi_box inválido: {job.roi_box}")
    rd = job.reader
    if rd.width > 0 and rd.height > 0:
        if x0 + w0 > rd.width or y0 + h0 > rd.height:
            raise ValueError(f"El ROI {job.roi_box} se sale del frame ({rd.width}x{rd.height}).")
        for z in job.zones:
            if z.mask.shape[:2] != (rd.height, rd.width):
                raise ValueError(
                    f"La máscara de la zona {z.name!r} mide {z.mask.shape[:2]} y debe ser del frame "
                    f"completo {(rd.height, rd.width)} (no del recorte del ROI)."
                )
    if job.tracker.include_mask.shape[:2] != (h0, w0):
        raise ValueError(
            f"La máscara del tracker mide {job.tracker.include_mask.shape[:2]} y el ROI es {(h0, w0)}."
        )
    if job.smoothing_alpha is not None and not (0.0 < job.smoothing_alpha <= 1.0):
        raise ValueError(f"smoothing_alpha debe estar en (0, 1], recibido {job.smoothing_alpha}")
    if job.jump_max_px is not None and not job.jump_max_px > 0:
        raise ValueError(f"jump_max_px debe ser > 0, recibido {job.jump_max_px}")
    if job.jump_max_speed_px_s is not None and not job.jump_max_speed_px_s > 0:
        raise ValueError(f"jump_max_speed_px_s debe ser > 0, recibido {job.jump_max_speed_px_s}")
    if job.adopt_after_s is not None and not job.adopt_after_s > 0:
        raise ValueError(f"adopt_after_s debe ser > 0, recibido {job.adopt_after_s}")
    if job.reacquire_after_frames < 1 or job.reacquire_every < 1:
        raise ValueError("reacquire_after_frames y reacquire_every deben ser >= 1")


def _start_tracker(job: TrackingJob, roi_start: np.ndarray, warnings: List[str]) -> None:
    """Deja al tracker listo en el frame de inicio, según su modo."""
    tr = job.tracker
    h0, w0 = roi_start.shape[:2]
    if isinstance(tr, SilhouetteTracker):
        if not tr.start(roi_start).found:
            warnings.append("No se encontró la silueta en el frame inicial; se arrancó desde el modelo aprendido.")
    elif isinstance(tr, ColorTracker):
        center, bbox = job.start if job.start is not None else default_start(w0, h0)
        tr.reset(center, bbox)
        if not tr.update(roi_start).found:
            warnings.append("No se encontró el marcador de color en el frame inicial.")
    else:
        if job.start is None:
            raise ValueError("El tracker de fondo necesita una posición inicial (job.start = (centro, caja)).")
        tr.reset(*job.start)


# Órdenes
def _apply_reinit(tr: BaseTracker, cmd: Reinit, roi: np.ndarray, warnings: List[str]) -> None:
    if cmd.center is not None and cmd.bbox is None:
        raise ValueError("Reinit con center necesita también bbox.")
    if isinstance(tr, SilhouetteTracker):
        if cmd.bbox is not None and not tr.reinit(roi, cmd.bbox):
            warnings.append("No se pudo extraer silueta en la nueva ventana.")
        return
    if cmd.params is not None:
        tr.params = dict(cmd.params)
    if cmd.bbox is not None:
        x, y, w, h = cmd.bbox
        tr.reset(cmd.center if cmd.center is not None else (x + w / 2.0, y + h / 2.0), cmd.bbox)


def _recapture_background(job: TrackingJob, frame_idx: int, warnings: List[str]) -> None:
    tr = job.tracker
    if not isinstance(tr, BackgroundTracker):
        warnings.append("RecaptureBackground solo aplica al modo de diferencia con fondo.")
        return
    bg = compute_background(read_background_frames(job.reader, frame_idx), job.roi_box, tr.include_mask, tr.use_clahe)
    if bg is None:
        warnings.append("No se pudo recapturar el fondo.")
    else:
        tr.set_background(bg)


# Bucle principal
def run_tracking(
    job: TrackingJob,
    on_frame: Optional[Callable[[FrameEvent], Optional[Command]]] = None,
    on_record: Optional[Callable[[FrameRecord], None]] = None,
    keep_records: bool = False,
    log: Optional[Callable[[str], None]] = print,
) -> TrackingResult:
    _validate(job)
    x0, y0, w0, h0 = job.roi_box
    rng, timing, tracker = job.frame_range, job.timing, job.tracker
    say = log if log is not None else (lambda _msg: None)

    kin = KinematicTracker(dt=timing.dt, cm_per_px=job.cm_per_px)
    zone_tracker = ZoneTracker(job.zones, dt=timing.dt, cm_per_px=job.cm_per_px)
    ema = PositionEMAFilter(alpha=job.smoothing_alpha) if job.smoothing_alpha is not None else None
    gate = job.jump_max_speed_px_s * timing.dt if job.jump_max_speed_px_s is not None else job.jump_max_px
    rejector = JumpRejector(max_distance_px=gate) if gate is not None else None
    adopt_frames = None if job.adopt_after_s is None else max(1, math.ceil(job.adopt_after_s / timing.dt))

    full_height = job.reader.height
    cm = job.cm_per_px
    total = rng.count(timing.frame_step)

    settings: Dict[str, Any] = {
        "modo_seguimiento": _mode_name(tracker),
        "suavizado_alpha": job.smoothing_alpha,
        "salto_max_px": gate,  # umbral efectivo, en px por frame procesado (None = sin rechazo de saltos)
        "salto_max_vel_px_s": job.jump_max_speed_px_s,
        "recuperacion_tras_frames": job.recovery_after if rejector is not None else None,
        "recuperacion_distancia_px": job.recovery_distance_px if rejector is not None else None,
        "adopcion_tras_s": job.adopt_after_s if rejector is not None else None,
        "buscador_sujeto": type(job.reacquirer).__name__ if job.reacquirer is not None else None,
        "buscador_tras_frames": job.reacquire_after_frames if job.reacquirer is not None else None,
        "buscador_cada_frames": job.reacquire_every if job.reacquirer is not None else None,
        "seleccion_independiente_del_margen": getattr(tracker, "border_aware", None),
    }

    warnings: List[str] = []
    records: List[FrameRecord] = []
    processed = found_count = tracker_found = rejected = recovered = reacquired = 0
    reacquirer = job.reacquirer
    lost_frames = 0  # frames seguidos sin posición válida (perdidos o rechazados)
    streak = 0  # detecciones rechazadas consecutivas y consistentes entre sí
    last_rejected: Optional[Tuple[float, float]] = None
    stopped = False

    frames = job.reader.iter_frames(rng.start, rng.end, timing.frame_step)
    first = next(frames, None)
    if first is None:
        raise RuntimeError(f"No se pudo leer el frame inicial ({rng.start}) del video.")
    first[1].flags.writeable = False
    _start_tracker(job, first[1][y0:y0 + h0, x0:x0 + w0], warnings)

    def forget_motion() -> None:
        """Olvida el movimiento previo (reinicio del sujeto)."""
        nonlocal lost_frames, streak, last_rejected
        kin.reset_history()
        zone_tracker.reset_history()
        if ema is not None:
            ema.reset()
        if rejector is not None:
            rejector.reset()
        lost_frames, streak, last_rejected = 0, 0, None

    for frame_idx, frame in itertools.chain([first], frames):
        frame.flags.writeable = False
        roi = frame[y0:y0 + h0, x0:x0 + w0]
        time_s = frame_time_s(frame_idx, rng.start, timing.video_fps)

        det = tracker.update(roi)

        status = STATUS_LOST
        raw: Optional[Tuple[float, float]] = None
        validated: Optional[Tuple[float, float]] = None

        if det.found:
            tracker_found += 1
            raw = (det.center[0] + x0, det.center[1] + y0)
            if rejector is None:
                validated, status = raw, STATUS_DETECTED
            else:
                dist = rejector.distance_to_last(raw)
                if lost_frames < job.recovery_after:
                    validated = rejector.update(raw)
                    if validated is not None:
                        status = STATUS_DETECTED
                elif dist is None or dist <= job.recovery_distance_px:
                    validated = rejector.accept(raw)
                    status = STATUS_RECOVERED
                    if ema is not None:
                        ema.reset()
                    say(f"Tracking recuperado en frame {frame_idx}")

                if validated is None:
                    status = STATUS_REJECTED
                    # ¿El tracker sigue viendo algo estable en otro lugar? Entonces probablemente es el sujeto.
                    if last_rejected is not None and math.dist(raw, last_rejected) <= gate:
                        streak += 1
                    else:
                        streak = 1
                    last_rejected = raw
                    if adopt_frames is not None and streak >= adopt_frames:
                        validated = rejector.accept(raw)
                        status = STATUS_RECOVERED
                        if ema is not None:
                            ema.reset()
                        msg = (f"Posición adoptada en el frame {frame_idx} tras {streak} detecciones consistentes "
                               f"a {dist:.0f} px de la última válida: confirma que es el mismo sujeto.")
                        warnings.append(msg)
                        say(msg)
                        streak, last_rejected = 0, None
                else:
                    streak, last_rejected = 0, None
        else:
            streak, last_rejected = 0, None

        # ¿Llevamos varios frames sin posición válida? Entonces se busca al sujeto en todo el ROI.
        if validated is None and reacquirer is not None:
            gap = lost_frames + 1  # frames seguidos sin posición válida, contando éste
            if gap >= job.reacquire_after_frames and (gap - job.reacquire_after_frames) % job.reacquire_every == 0:
                try:
                    cand = reacquirer.find(roi, tracker.last_center, tracker.last_bbox)
                except Exception as exc:  # un buscador que falla no debe tirar abajo una corrida larga
                    warnings.append(f"El buscador del sujeto falló en el frame {frame_idx} "
                                    f"({type(exc).__name__}: {exc}) y se desactivó para el resto de la corrida.")
                    reacquirer, cand = None, None
                if cand is not None and cand.found:
                    tracker.reset(cand.center, cand.bbox)
                    raw = (cand.center[0] + x0, cand.center[1] + y0)
                    validated = rejector.accept(raw) if rejector is not None else raw
                    status = STATUS_RECOVERED
                    if ema is not None:
                        ema.reset()
                    streak, last_rejected = 0, None
                    reacquired += 1
                    det = cand
                    say(f"Sujeto reencontrado en el frame {frame_idx}")

        cx_abs = cy_abs = cy_abs_inv = None
        x_rel_px = y_rel_px = x_rel_cm = y_rel_cm = None
        point: Optional[Tuple[float, float]] = None
        if validated is not None:
            lost_frames = 0
            found_count += 1
            point = ema.update(validated) if ema is not None else validated
            cx_abs, cy_abs = point
            cy_abs_inv = full_height - cy_abs
            if job.xy_origin is not None:
                ox, oy = job.xy_origin
                x_rel_px = cx_abs - ox
                y_rel_px = oy - cy_abs  # Y positiva hacia arriba
                if cm is not None:
                    x_rel_cm = x_rel_px * cm
                    y_rel_cm = y_rel_px * cm
        else:
            lost_frames += 1
        rejected += status == STATUS_REJECTED
        recovered += status == STATUS_RECOVERED

        sample = kin.update(point)
        zone_records = zone_tracker.update(point, frame_idx, time_s)

        record = FrameRecord(
            frame=frame_idx,
            time_s=time_s,
            cx_abs=cx_abs,
            cy_abs_inv=cy_abs_inv,
            x_rel_px=x_rel_px,
            y_rel_px=y_rel_px,
            x_rel_cm=x_rel_cm,
            y_rel_cm=y_rel_cm,
            zones=zone_records,
            status=status,
            cx_raw=None if raw is None else raw[0],
            cy_raw_inv=None if raw is None else full_height - raw[1],
            **{name: getattr(sample, name) for name in _SAMPLE_FIELDS},
        )
        processed += 1
        if keep_records:
            records.append(record)
        if on_record is not None:
            on_record(record)

        cmd: Optional[Command] = None
        if on_frame is not None:
            cmd = on_frame(FrameEvent(frame_idx, frame, job.roi_box, det, record, processed, total))
        if cmd is None:
            continue
        if isinstance(cmd, Quit):
            stopped = True
            break
        if isinstance(cmd, Reinit):
            _apply_reinit(tracker, cmd, roi, warnings)
            forget_motion()
        elif isinstance(cmd, RecaptureBackground):
            _recapture_background(job, frame_idx, warnings)
        else:
            raise TypeError(f"Orden desconocida: {cmd!r}")

    if tracker_found >= 10 and rejected / tracker_found >= 0.2:
        warnings.append(
            f"El filtro de saltos descartó {rejected} de {tracker_found} detecciones ({100 * rejected / tracker_found:.0f} %): "
            "el umbral probablemente es demasiado bajo para esta resolución, escala o velocidad de procesamiento "
            "(ajusta jump_max_px o jump_max_speed_px_s)."
        )

    return TrackingResult(
        frames_processed=processed,
        frames_found=found_count,
        stopped_by_user=stopped,
        kinematic=kin.summary(),
        zones=zone_tracker.summary(),
        timing=timing,
        frame_range=rng,
        warnings=warnings,
        records=records,
        frames_rejected=rejected,
        frames_recovered=recovered,
        frames_reacquired=reacquired,
        settings=settings,
    )