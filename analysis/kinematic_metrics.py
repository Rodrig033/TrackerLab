from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

__all__ = [
    "Point",
    "euclidean",
    "angle_between_vectors_deg",
    "KinematicSample",
    "KinematicSummary",
    "KinematicTracker",
]

Point = Tuple[float, float]


# Funciones básicas

def euclidean(p1: Point, p2: Point) -> float:
    """Distancia entre dos puntos."""
    return float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)


def angle_between_vectors_deg(v1, v2) -> Optional[float]:
    
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 <= 1e-12 or n2 <= 1e-12:
        return None
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


# Resultados
@dataclass(frozen=True)
class KinematicSample:

    dist_acum_px: float
    dist_acum_cm: Optional[float] = None  # None si no hay escala

    dist_px: Optional[float] = None
    dist_cm: Optional[float] = None
    vel_px_s: Optional[float] = None
    vel_cm_s: Optional[float] = None
    acel_px_s2: Optional[float] = None
    acel_cm_s2: Optional[float] = None
    ang_deg: Optional[float] = None
    vel_angular_deg_s: Optional[float] = None


@dataclass(frozen=True)
class KinematicSummary:

    frames_validos: int
    tiempo_valido_s: float  # frames_validos * dt
    dist_total_px: float
    dist_total_cm: Optional[float] = None
    vel_media_px_s: Optional[float] = None
    vel_media_cm_s: Optional[float] = None
    acel_media_px_s2: Optional[float] = None
    acel_media_cm_s2: Optional[float] = None
    angulo_medio_deg: Optional[float] = None
    vel_angular_media_deg_s: Optional[float] = None


# Calculador con estado
class KinematicTracker:
    """Calcula la cinemática frame a frame y acumula los totales de la sesión."""

    def __init__(self, dt: float, cm_per_px: Optional[float] = None):
        """
        dt: segundos entre frames procesados (``TrackingTiming.dt``).
        cm_per_px: escala de calibración; ``None`` si no se calibró (entonces
                   todas las magnitudes en cm quedan en ``None``).
        """
        if not dt > 0:
            raise ValueError(f"dt debe ser > 0, recibido {dt}")
        if cm_per_px is not None and not cm_per_px > 0:
            raise ValueError(f"cm_per_px debe ser > 0, recibido {cm_per_px}")
        self.dt = float(dt)
        self.cm_per_px = None if cm_per_px is None else float(cm_per_px)

        # Memoria corta (se borra al perder al sujeto o al reiniciar)
        self._prev_point: Optional[Point] = None
        self._prev_prev_point: Optional[Point] = None  # para el ángulo
        self._prev_vel_px: Optional[float] = None
        self._prev_vel_cm: Optional[float] = None

        # Acumulados de la sesión (NO se borran al perder al sujeto)
        self._dist_acum_px = 0.0
        self._dist_acum_cm: Optional[float] = 0.0 if self.cm_per_px is not None else None
        self._valid_frames = 0
        self._sum_vel_px = 0.0
        self._sum_vel_cm = 0.0
        self._n_vel = 0
        self._sum_acc_px = 0.0
        self._sum_acc_cm = 0.0
        self._n_acc = 0
        self._sum_ang = 0.0
        self._n_ang = 0
        self._sum_angvel = 0.0
        self._n_angvel = 0

    def reset_history(self) -> None:
        """Olvida el movimiento previo SIN tocar los acumulados.

        Se llama solo (dentro de ``update(None)``) cuando se pierde al sujeto, y
        a mano tras un reinit ('r'). El siguiente punto válido se trata como
        un nuevo comienzo: sin velocidad, aceleración ni ángulo.
        """
        self._prev_point = None
        self._prev_prev_point = None
        self._prev_vel_px = None
        self._prev_vel_cm = None

    @property
    def dist_acum_px(self) -> float:
        return self._dist_acum_px

    @property
    def dist_acum_cm(self) -> Optional[float]:
        return self._dist_acum_cm

    def update(self, point: Optional[Point]) -> KinematicSample:
        """Procesa un frame. ``point`` = (x, y) absolutos, o ``None`` si se perdió."""
        if point is None:
            self.reset_history()
            return KinematicSample(dist_acum_px=self._dist_acum_px, dist_acum_cm=self._dist_acum_cm)

        self._valid_frames += 1
        cm = self.cm_per_px
        dt = self.dt
        prev = self._prev_point

        dist_px = dist_cm = vel_px = vel_cm = acc_px = acc_cm = None

        if prev is not None:
            dist_px = euclidean(point, prev)
            self._dist_acum_px += dist_px

            vel_px = dist_px / dt
            self._sum_vel_px += vel_px
            self._n_vel += 1

            if self._prev_vel_px is not None:
                acc_px = (vel_px - self._prev_vel_px) / dt
                self._sum_acc_px += acc_px
                self._n_acc += 1

            if cm is not None:
                dist_cm = dist_px * cm
                self._dist_acum_cm += dist_cm

                vel_cm = dist_cm / dt
                self._sum_vel_cm += vel_cm

                if self._prev_vel_cm is not None:
                    acc_cm = (vel_cm - self._prev_vel_cm) / dt
                    self._sum_acc_cm += acc_cm
        else:
            # Primer punto tras un inicio/pérdida: sin desplazamiento ni velocidad.
            dist_px = 0.0
            if cm is not None:
                dist_cm = 0.0

        # Ángulo: giro entre los dos últimos desplazamientos (necesita 3 puntos).
        ang = angvel = None
        if self._prev_prev_point is not None and prev is not None:
            v1 = (prev[0] - self._prev_prev_point[0], prev[1] - self._prev_prev_point[1])
            v2 = (point[0] - prev[0], point[1] - prev[1])
            ang = angle_between_vectors_deg(v1, v2)
            if ang is not None:
                angvel = ang / dt
                self._sum_ang += ang
                self._n_ang += 1
                self._sum_angvel += angvel
                self._n_angvel += 1

        # Actualizar memoria (solo hubo tracking)
        self._prev_prev_point = prev
        self._prev_point = (point[0], point[1])
        self._prev_vel_px = vel_px
        self._prev_vel_cm = vel_cm

        return KinematicSample(
            dist_acum_px=self._dist_acum_px,
            dist_acum_cm=self._dist_acum_cm,
            dist_px=dist_px,
            dist_cm=dist_cm,
            vel_px_s=vel_px,
            vel_cm_s=vel_cm,
            acel_px_s2=acc_px,
            acel_cm_s2=acc_cm,
            ang_deg=ang,
            vel_angular_deg_s=angvel,
        )

    def summary(self) -> KinematicSummary:
        """Totales y promedios de toda la sesión hasta este momento."""
        has_cm = self.cm_per_px is not None
        return KinematicSummary(
            frames_validos=self._valid_frames,
            tiempo_valido_s=self._valid_frames * self.dt,
            dist_total_px=self._dist_acum_px,
            dist_total_cm=self._dist_acum_cm,
            vel_media_px_s=self._sum_vel_px / self._n_vel if self._n_vel > 0 else None,
            vel_media_cm_s=self._sum_vel_cm / self._n_vel if has_cm and self._n_vel > 0 else None,
            acel_media_px_s2=self._sum_acc_px / self._n_acc if self._n_acc > 0 else None,
            acel_media_cm_s2=self._sum_acc_cm / self._n_acc if has_cm and self._n_acc > 0 else None,
            angulo_medio_deg=self._sum_ang / self._n_ang if self._n_ang > 0 else None,
            vel_angular_media_deg_s=self._sum_angvel / self._n_angvel if self._n_angvel > 0 else None,
        )