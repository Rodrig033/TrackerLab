"""Conexión del generador de datos con la aplicación."""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from learning.dataset import DatasetCollector, make_audit_sheet

__all__ = ["DatasetSession"]


def _safe_name(video_path: Optional[str]) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0] if video_path else ""
    return re.sub(r"[^\w\-]+", "_", stem).strip("_")[:40] or "video"


class DatasetSession:
    """Un conjunto de datos asociado a UNA corrida de seguimiento."""

    def __init__(
        self,
        enabled: bool,
        base_dir: str,
        *,
        video_path: Optional[str] = None,
        duration_s: Optional[float] = None,
        min_interval_s: float = 1.0,
        max_images: Optional[int] = 2000,
        log: Optional[Callable[[str], None]] = print,
        now: Optional[Callable[[], datetime]] = None,
    ):
        """
        enabled:        interruptor. False = no hace nada.
        base_dir:       carpeta donde se crea ``dataset_<fecha>`` (por ejemplo, la carpeta del experimento).
        video_path:     video procesado (para el nombre de las imágenes y la huella del video en la ficha).
        duration_s:     duración del segmento, para repartir bien entrenamiento y validación en videos cortos.
        min_interval_s: a lo sumo una imagen cada tantos segundos.
        max_images:     tope de imágenes (None = sin tope). Si se alcanza, solo se cubre el inicio del video.
        log:            recibe los mensajes (por omisión ``print``); None = en silencio.
        """
        self.enabled = bool(enabled)
        self.log = log if log is not None else (lambda _msg: None)
        self.max_images = max_images
        self.collector: Optional[DatasetCollector] = None
        self.out_dir: Optional[str] = None
        self.error: Optional[str] = None
        self._summary: Optional[Dict[str, Any]] = None
        # Con pocos bloques no habría validación: en videos cortos los bloques se acortan (entre 4 y 20 s).
        self.block_seconds = 20.0 if not duration_s else min(20.0, max(4.0, duration_s / 10.0))
        self.buffer_s = min(1.0, self.block_seconds / 5.0)
        if not self.enabled:
            return
        stamp = (now or datetime.now)().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(base_dir, f"dataset_{stamp}")
        try:
            self.collector = DatasetCollector(
                self.out_dir, video_name=_safe_name(video_path), stride=1, min_interval_s=min_interval_s,
                max_images=max_images, block_seconds=self.block_seconds, buffer_s=self.buffer_s, video_path=video_path)
        except Exception as exc:  # no debe impedir el seguimiento
            self.enabled, self.error = False, f"{type(exc).__name__}: {exc}"
            self.log(f"No se pudo preparar el generador de datos ({self.error}); el seguimiento continúa sin él.")

    def wrap(self, on_frame: Optional[Callable]) -> Optional[Callable]:
        """Devuelve el ``on_frame`` a usar en ``run_tracking``. Apagado, es exactamente el mismo objeto."""
        if self.collector is None:
            return on_frame

        def chained(event):
            cmd = on_frame(event) if on_frame is not None else None  # primero la interfaz; sus órdenes pasan intactas
            if self.error is None:
                try:
                    self.collector(event)
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.log(f"El generador de datos se desactivó durante la corrida ({self.error}); el seguimiento continúa.")
            return cmd

        return chained

    def finish(self, result=None) -> Optional[Dict[str, Any]]:
        """Cierra el conjunto de datos, genera ``revision.png`` y avisa por consola. Devuelve un resumen (o None)."""
        if self.collector is None:
            return None
        if self._summary is not None:
            return self._summary
        extra: Dict[str, Any] = {}
        settings = getattr(result, "settings", None)
        if settings:
            extra["modo_de_seguimiento"] = settings.get("modo_seguimiento", "")
            extra["opciones_de_procesamiento"] = dict(settings)
        try:
            info = self.collector.finalize(extra)
            sheet = make_audit_sheet(self.out_dir, os.path.join(self.out_dir, "revision.png"))
        except Exception as exc:
            self.log(f"No se pudo cerrar el conjunto de datos ({type(exc).__name__}: {exc}).")
            return None

        n = info["imagenes"]
        self.log("")
        self.log("Conjunto de datos (etiquetas AUTOMÁTICAS del seguimiento):")
        self.log(self.out_dir)
        self.log(f"  imágenes: {n['total']}  (entrenamiento {n['train']}, validación {n['val']})")
        if sheet:
            self.log("Hoja de revisión (revísala antes de usar el conjunto):")
            self.log(sheet)
        if n["total"] == 0:
            self.log(f"  AVISO: no se guardó ninguna imagen. Motivos de descarte: {info['frames_descartados_por_motivo']}")
        elif n["val"] == 0:
            self.log("  AVISO: no hay imágenes de validación (el video es muy corto): con este conjunto no se podrá evaluar.")
        if self.max_images is not None and info["frames_descartados_por_motivo"].get("limite"):
            self.log(f"  AVISO: se alcanzó el límite de {self.max_images} imágenes: solo se cubrió el inicio del video.")
        if self.error:
            self.log(f"  AVISO: el generador se desactivó durante la corrida ({self.error}); el conjunto está incompleto.")
        self._summary = {"carpeta": self.out_dir, "info": info, "hoja_de_revision": sheet}
        return self._summary