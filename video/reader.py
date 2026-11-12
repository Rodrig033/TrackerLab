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

#: Hasta cuántos frames adelante se avanza con grab() en vez de saltar.
#: Es un punto de partida; conviene ajustarlo midiendo con videos reales.
GRAB_SKIP_MAX = 30


class VideoOpenError(OSError):
    """El archivo existe pero OpenCV no pudo abrirlo como video."""


def _prop(cap: "cv2.VideoCapture", prop: int) -> float:
    """Lee una propiedad de OpenCV devolviendo 0.0 si no es un número válido."""
    value = cap.get(prop)
    return float(value) if value is not None and math.isfinite(value) else 0.0


class VideoReader:
    """Acceso a los frames de un video por índice.

    Uso::

        with VideoReader("video.mp4") as reader:
            print(reader.fps, reader.total_frames)
            first = reader.read_at(0)
            for idx, frame in reader.iter_frames(start=100, end=400, step=2):
                ...

    Los frames se devuelven como arreglos BGR de numpy (formato de OpenCV).
    Los índices empiezan en 0.
    """

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
        # True si ``_pos`` es exacto (solo se avanzó de corrido desde el inicio); False tras un salto con set().
        self._pos_exact: bool = True
        #: Cuántas veces se reabrió el video para llegar a un frame de forma exacta (para diagnóstico y pruebas).
        self.reopens: int = 0

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._pos = None
            self._pos_exact = False

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lectura
    # ------------------------------------------------------------------
    def read_at(self, idx: int) -> Optional[np.ndarray]:
        """Devuelve el frame ``idx`` (BGR) o ``None`` si no se pudo leer.

        ``None`` ocurre al pedir un índice negativo o más allá del final.
        """
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
            self._pos_exact = False  # un salto puede caer corrido: desde aquí la posición ya no es confiable

        ok, frame = self._cap.read()
        if not ok:
            self._pos = None  # tras un fallo no confiamos en la posición
            return None
        self._pos = idx + 1
        return frame

    def _reopen(self) -> None:
        """Vuelve al frame 0 de forma segura: reabre el archivo (un salto a 0 no siempre es exacto en todos los códecs)."""
        if self._cap is not None:
            self._cap.release()
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            self._cap, self._pos = None, None
            raise VideoOpenError(f"No se pudo reabrir el video: {self.path}")
        self._cap, self._pos, self._pos_exact = cap, 0, True
        self.reopens += 1

    def read_exact(self, idx: int) -> Optional[np.ndarray]:
        """El frame ``idx`` con numeración EXACTA: se llega avanzando de corrido, nunca con un salto.

        Si la posición actual es exacta y está antes de ``idx``, solo avanza; si no, reabre el video y avanza desde 0.
        """
        if self._cap is None:
            raise ValueError("El lector está cerrado.")
        idx = int(idx)
        if idx < 0:
            return None
        if not (self._pos is not None and self._pos_exact and self._pos <= idx):
            self._reopen()
        for _ in range(idx - self._pos):  # type: ignore[operator]
            if not self._cap.grab():  # type: ignore[union-attr]
                self._pos, self._pos_exact = None, False
                return None
        ok, frame = self._cap.read()  # type: ignore[union-attr]
        if not ok:
            self._pos, self._pos_exact = None, False
            return None
        self._pos = idx + 1
        return frame

    def iter_frames(
        self, start: int = 0, end: Optional[int] = None, step: int = 1, exact: bool = True
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Itera ``(índice, frame)`` desde ``start`` hasta ``end`` INCLUSIVE.

        Igual que el bucle original, avanza de ``step`` en ``step`` frames y se
        detiene al pasar ``end`` o al fallar la lectura. Con ``end=None`` llega
        hasta el final del video.

        Con ``exact=True`` (por omisión) los números de frame son los del video: se llega a cada frame avanzando de corrido
        (ver ``read_exact``). ``exact=False`` usa ``read_at`` (puede saltar, y en un mp4 caer corrido).
        """
        step = int(step)
        if step < 1:
            raise ValueError("step debe ser >= 1")
        return self._iter(int(start), None if end is None else int(end), step, bool(exact))

    def _iter(self, start: int, end: Optional[int], step: int, exact: bool = True) -> Iterator[Tuple[int, np.ndarray]]:
        idx = start
        leer = self.read_exact if exact else self.read_at
        while end is None or idx <= end:
            frame = leer(idx)
            if frame is None:
                return
            yield idx, frame
            idx += step