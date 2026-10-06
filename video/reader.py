from __future__ import annotations

import math
import os
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np

__all__ = ["VideoReader", "VideoOpenError", "DEFAULT_FPS"]

PathLike = Union[str, "os.PathLike[str]"]

#: FPS que se usa si el archivo no declara uno válido (igual que el original).
DEFAULT_FPS = 30.0
GRAB_SKIP_MAX = 30


class VideoOpenError(OSError):
    """El archivo existe pero OpenCV no pudo abrirlo como video."""


def _prop(cap: "cv2.VideoCapture", prop: int) -> float:
    """Lee una propiedad de OpenCV devolviendo 0.0 si no es un número válido."""
    value = cap.get(prop)
    return float(value) if value is not None and math.isfinite(value) else 0.0


class VideoReader:

    def __init__(self, path: PathLike):
        path = os.fspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No existe el video: {path}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            raise VideoOpenError(f"No se pudo abrir el video (¿formato o códec no soportado?): {path}")

        self._cap: Optional[cv2.VideoCapture] = cap
        self.path: str = path

        fps = _prop(cap, cv2.CAP_PROP_FPS)
        #: True si el archivo no traía FPS válido y se usó DEFAULT_FPS.
        #: Conviene avisar al usuario: velocidades y tiempos dependen del FPS.
        self.fps_is_fallback: bool = not (fps > 0)
        self.fps: float = DEFAULT_FPS if self.fps_is_fallback else fps

        #: Algunos contenedores reportan 0 o un valor estimado.
        self.total_frames: int = max(0, int(_prop(cap, cv2.CAP_PROP_FRAME_COUNT)))
        self.width: int = int(_prop(cap, cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(_prop(cap, cv2.CAP_PROP_FRAME_HEIGHT))

        #: Hasta cuántos frames adelante se usa grab() en vez de saltar.
        self.skip_limit: int = GRAB_SKIP_MAX

        # Índice del frame que devolvería el próximo read(); None = se desconoce.
        self._pos: Optional[int] = 0

    # Ciclo de vida
    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._pos = None

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # Lectura
    def read_at(self, idx: int) -> Optional[np.ndarray]:
        if self._cap is None:
            raise ValueError("El lector está cerrado.")
        idx = int(idx)
        if idx < 0:
            return None

        pos = self._pos
        if pos is not None and 0 <= idx - pos <= self.skip_limit:
            # Ya estamos en idx (0 saltos) o falta poco: avanzar sin buscar.
            for _ in range(idx - pos):
                if not self._cap.grab():
                    self._pos = None
                    return None
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ok, frame = self._cap.read()
        if not ok:
            self._pos = None  # tras un fallo no confiamos en la posición
            return None
        self._pos = idx + 1
        return frame

    def iter_frames(
        self, start: int = 0, end: Optional[int] = None, step: int = 1
    ) -> Iterator[Tuple[int, np.ndarray]]:
        step = int(step)
        if step < 1:
            raise ValueError("step debe ser >= 1")
        return self._iter(int(start), None if end is None else int(end), step)

    def _iter(self, start: int, end: Optional[int], step: int) -> Iterator[Tuple[int, np.ndarray]]:
        idx = start
        while end is None or idx <= end:
            frame = self.read_at(idx)
            if frame is None:
                return
            yield idx, frame
            idx += step