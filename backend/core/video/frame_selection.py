from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional

if TYPE_CHECKING:  # solo para anotaciones; evita importar cv2 aquí
    import numpy as np

    from video.reader import VideoReader

__all__ = [
    "BG_N_FRAMES",
    "BG_STRIDE",
    "FrameRange",
    "TrackingTiming",
    "compute_timing",
    "frame_time_s",
    "background_sample_indices",
    "read_background_frames",
]

#: Frames usados para estimar el fondo (mediana) y separación entre ellos.
#: Mismos valores por defecto que el script original.
BG_N_FRAMES = 35
BG_STRIDE = 2


# Rango de frames
@dataclass(frozen=True)
class FrameRange:
    """Segmento del video a procesar; ``start`` y ``end`` son INCLUSIVOS."""

    start: int
    end: int
    #: True si el usuario eligió inicio > fin y se intercambiaron.
    #: La interfaz puede usarlo para avisar (el original imprimía un aviso).
    was_swapped: bool = False

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Rango inválido: start={self.start}, end={self.end}")

    @classmethod
    def ordered(cls, a: int, b: int) -> "FrameRange":
        """Crea el rango a partir de dos frames elegidos en cualquier orden."""
        a, b = int(a), int(b)
        if b < a:
            return cls(start=b, end=a, was_swapped=True)
        return cls(start=a, end=b)

    def indices(self, step: int = 1) -> range:
        """Índices que se procesan, de ``step`` en ``step``, ``end`` incluido."""
        if int(step) < 1:
            raise ValueError("step debe ser >= 1")
        return range(self.start, self.end + 1, int(step))

    def count(self, step: int = 1) -> int:
        """Cuántos frames se procesarán (útil para barras de progreso)."""
        return len(self.indices(step))


# Resolución temporal del tracking

@dataclass(frozen=True)
class TrackingTiming:
    video_fps: float  # FPS del archivo
    requested_fps: float  # FPS pedido por el usuario (ya acotado a [1, video_fps])
    frame_step: int  # se procesa 1 de cada ``frame_step`` frames
    effective_fps: float  # video_fps / frame_step
    dt: float  # segundos entre dos frames procesados = frame_step / video_fps


def compute_timing(
    video_fps: float,
    custom_fps: bool = False,
    requested_fps: Optional[float] = None,
) -> TrackingTiming:

    video_fps = float(video_fps)
    if not video_fps > 0:
        raise ValueError(f"FPS del video inválido: {video_fps}")

    if custom_fps and requested_fps is not None:
        requested = max(1.0, min(float(requested_fps), video_fps))
        step = max(1, int(round(video_fps / requested)))
    else:
        requested = video_fps
        step = 1

    return TrackingTiming(
        video_fps=video_fps,
        requested_fps=requested,
        frame_step=step,
        effective_fps=video_fps / step,
        dt=step / video_fps,
    )


def frame_time_s(frame_idx: int, start_idx: int, video_fps: float) -> float:
    return (int(frame_idx) - int(start_idx)) / float(video_fps)


# Muestreo para el fondo

def background_sample_indices(
    center_idx: int,
    total_frames: int = 0,
    n: int = BG_N_FRAMES,
    stride: int = BG_STRIDE,
) -> List[int]:

    if n < 1 or stride < 1:
        raise ValueError("n y stride deben ser >= 1")

    idx = max(0, int(center_idx) - (n // 2) * stride)
    out: List[int] = []
    for _ in range(n):
        if total_frames > 0 and idx >= total_frames:
            break
        out.append(idx)
        idx += stride
    return out


def read_background_frames(
    reader: "VideoReader",
    center_idx: int,
    n: int = BG_N_FRAMES,
    stride: int = BG_STRIDE,
) -> Iterator["np.ndarray"]:
    for idx in background_sample_indices(center_idx, reader.total_frames, n, stride):
        frame = reader.read_at(idx)
        if frame is None:
            return
        yield frame