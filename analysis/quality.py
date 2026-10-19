from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

__all__ = [
    "STATUS_DETECTED",
    "STATUS_LOST",
    "STATUS_RECOVERED",
    "STATUS_REJECTED",
    "frame_status",
    "Gap",
    "Jump",
    "QualityThresholds",
    "QualityReport",
    "QualityMonitor",
    "assess_quality",
    "quality_rows",
    "format_quality",
]

STATUS_DETECTED = "detectado"
STATUS_LOST = "perdido"
#: Reservado: lo usará la recuperación del sujeto cuando exista. Una posición
#: recuperada NO es una detección y debe poder distinguirse en los resultados.
STATUS_RECOVERED = "recuperado"
#: El tracker vio algo, pero el filtro de saltos de la corrida lo descartó (cuenta como frame sin dato).
STATUS_REJECTED = "rechazado"

VERDICT_GOOD = "bueno"
VERDICT_REVIEW = "revisar"
VERDICT_BAD = "deficiente"
VERDICT_NONE = "sin datos"


def frame_status(record) -> str:
    """Estado de un frame. Usa ``record.status`` si el pipeline lo informó; si no, detectado o perdido."""
    status = getattr(record, "status", "")
    if status:
        return status
    return STATUS_DETECTED if record.found else STATUS_LOST


@dataclass(frozen=True)
class Gap:
    """Racha de frames consecutivos con el sujeto perdido."""

    first_frame: int
    last_frame: int
    n_frames: int  # frames PROCESADOS perdidos (no cuenta los que se saltó frame_step)
    duration_s: float
    kind: str  # "inicio" (antes de la 1.ª detección), "final" (después de la última) o "interna"


@dataclass(frozen=True)
class Jump:
    """Desplazamiento sospechosamente grande entre dos frames consecutivos detectados."""

    frame: int  # frame al que llega el sujeto
    step_px: float


@dataclass(frozen=True)
class QualityThresholds:
    lost_pct_warn: float = 5.0  # % de frames perdidos a partir del cual conviene revisar
    lost_pct_bad: float = 15.0  # % a partir del cual el seguimiento es deficiente
    longest_gap_s_warn: float = 1.0
    longest_gap_s_bad: float = 5.0
    #: Un paso es un "salto" si supera esta fracción de la diagonal del recorrido…
    jump_fraction_of_extent: float = 0.30
    #: …y nunca se considera salto algo menor a esto (píxeles).
    jump_min_px: float = 25.0
    #: % de frames que el filtro de saltos descartó a partir del cual conviene revisar.
    rejected_pct_warn: float = 5.0


@dataclass(frozen=True)
class QualityReport:
    frames_total: int
    frames_detected: int
    frames_lost: int
    lost_pct: float
    duration_s: float  # tiempo cubierto por todos los frames procesados
    valid_time_s: float  # tiempo con el sujeto detectado
    gaps: Tuple[Gap, ...]
    n_interruptions: int  # huecos internos (entre dos detecciones)
    interruptions_per_min: float
    longest_gap_frames: int
    longest_gap_s: float
    leading_lost_frames: int
    trailing_lost_frames: int
    jumps: Tuple[Jump, ...]
    jump_threshold_px: Optional[float]
    verdict: str
    reasons: Tuple[str, ...]
    # Informados por el pipeline mediante FrameRecord.status (0 si no se informó).
    frames_rejected: int = 0
    frames_recovered: int = 0
    rejected_pct: float = 0.0

    @property
    def n_jumps(self) -> int:
        return len(self.jumps)

    @property
    def is_reliable(self) -> bool:
        """True solo si el veredicto es 'bueno'."""
        return self.verdict == VERDICT_GOOD


class QualityMonitor:
    """Acumula la calidad registro por registro (apto para ``on_record``)."""

    def __init__(
        self,
        dt: float,
        thresholds: QualityThresholds = QualityThresholds(),
        max_step_px: Optional[float] = None,
    ):
        """
        dt: segundos entre frames procesados (``TrackingTiming.dt``).
        max_step_px: umbral de salto fijo (px entre dos frames). Si es None se
                     deduce del recorrido (``thresholds``).
        """
        if not dt > 0:
            raise ValueError(f"dt debe ser > 0, recibido {dt}")
        self.dt = float(dt)
        self.th = thresholds
        self.max_step_px = max_step_px

        self._total = 0
        self._detected = 0
        self._rejected = 0
        self._recovered = 0
        self._gaps: List[Gap] = []
        self._gap_first: Optional[int] = None
        self._gap_last: Optional[int] = None
        self._gap_n = 0
        self._seen_detection = False
        self._prev: Optional[Tuple[float, float]] = None  # posición previa, solo si el frame anterior fue detectado
        self._steps: List[Tuple[int, float]] = []
        self._min = [math.inf, math.inf]
        self._max = [-math.inf, -math.inf]

    def _close_gap(self, kind: str) -> None:
        if self._gap_n:
            self._gaps.append(Gap(self._gap_first, self._gap_last, self._gap_n, self._gap_n * self.dt, kind))
        self._gap_first = self._gap_last = None
        self._gap_n = 0

    def update(self, rec) -> None:
        self._total += 1
        status = getattr(rec, "status", "")
        if status == STATUS_REJECTED:
            self._rejected += 1
        elif status == STATUS_RECOVERED:
            self._recovered += 1
        if not rec.found:
            if self._gap_n == 0:
                self._gap_first = rec.frame
            self._gap_last = rec.frame
            self._gap_n += 1
            self._prev = None  # tras un hueco, la distancia no se compara con la posición vieja
            return

        self._detected += 1
        self._close_gap("inicio" if not self._seen_detection else "interna")
        self._seen_detection = True

        x, y = float(rec.cx_abs), float(rec.cy_abs_inv)
        if self._prev is not None:
            self._steps.append((rec.frame, math.hypot(x - self._prev[0], y - self._prev[1])))
        self._prev = (x, y)
        self._min = [min(self._min[0], x), min(self._min[1], y)]
        self._max = [max(self._max[0], x), max(self._max[1], y)]

    def report(self) -> QualityReport:
        """Informe con lo acumulado hasta ahora (se puede pedir varias veces)."""
        gaps = list(self._gaps)
        if self._gap_n:  # hueco que sigue abierto al terminar
            kind = "final" if self._seen_detection else "inicio"
            gaps.append(Gap(self._gap_first, self._gap_last, self._gap_n, self._gap_n * self.dt, kind))

        total, detected = self._total, self._detected
        lost = total - detected
        lost_pct = 100.0 * lost / total if total else 0.0
        duration = total * self.dt
        internal = [g for g in gaps if g.kind == "interna"]
        longest = max(gaps, key=lambda g: g.n_frames, default=None)

        thr: Optional[float] = None
        jumps: List[Jump] = []
        if self._steps:
            if self.max_step_px is not None:
                thr = float(self.max_step_px)
            else:
                diag = math.hypot(self._max[0] - self._min[0], self._max[1] - self._min[1])
                thr = max(self.th.jump_fraction_of_extent * diag, self.th.jump_min_px)
            jumps = [Jump(f, s) for f, s in self._steps if s > thr]

        rejected_pct = 100.0 * self._rejected / total if total else 0.0
        verdict, reasons = self._judge(total, lost_pct, longest, jumps, rejected_pct)
        return QualityReport(
            frames_total=total,
            frames_detected=detected,
            frames_lost=lost,
            lost_pct=lost_pct,
            duration_s=duration,
            valid_time_s=detected * self.dt,
            gaps=tuple(gaps),
            n_interruptions=len(internal),
            interruptions_per_min=(len(internal) / (duration / 60.0)) if duration > 0 else 0.0,
            longest_gap_frames=longest.n_frames if longest else 0,
            longest_gap_s=longest.duration_s if longest else 0.0,
            leading_lost_frames=sum(g.n_frames for g in gaps if g.kind == "inicio"),
            trailing_lost_frames=sum(g.n_frames for g in gaps if g.kind == "final"),
            jumps=tuple(jumps),
            jump_threshold_px=thr,
            verdict=verdict,
            reasons=tuple(reasons),
            frames_rejected=self._rejected,
            frames_recovered=self._recovered,
            rejected_pct=rejected_pct,
        )

    def _judge(self, total, lost_pct, longest, jumps, rejected_pct=0.0):
        th = self.th
        if total == 0:
            return VERDICT_NONE, ["No hay registros."]
        if self._detected == 0:
            return VERDICT_BAD, ["El sujeto no se detectó en ningún frame."]
        verdict, reasons = VERDICT_GOOD, []

        def raise_to(level: str, text: str) -> None:
            nonlocal verdict
            order = {VERDICT_GOOD: 0, VERDICT_REVIEW: 1, VERDICT_BAD: 2}
            if order[level] > order[verdict]:
                verdict = level
            reasons.append(text)

        if lost_pct >= th.lost_pct_bad:
            raise_to(VERDICT_BAD, f"Frames perdidos: {lost_pct:.1f} % (límite de calidad: {th.lost_pct_bad:g} %).")
        elif lost_pct >= th.lost_pct_warn:
            raise_to(VERDICT_REVIEW, f"Frames perdidos: {lost_pct:.1f} % (conviene revisar a partir de {th.lost_pct_warn:g} %).")
        if longest is not None:
            if longest.duration_s >= th.longest_gap_s_bad:
                raise_to(VERDICT_BAD, f"Hueco más largo: {longest.duration_s:.1f} s (frames {longest.first_frame} a {longest.last_frame}).")
            elif longest.duration_s >= th.longest_gap_s_warn:
                raise_to(VERDICT_REVIEW, f"Hueco más largo: {longest.duration_s:.1f} s (frames {longest.first_frame} a {longest.last_frame}).")
        if jumps:
            raise_to(VERDICT_REVIEW, f"{len(jumps)} salto(s) de posición sospechosos; primero en el frame {jumps[0].frame}.")
        if self._rejected and rejected_pct >= th.rejected_pct_warn:
            raise_to(VERDICT_REVIEW, f"{self._rejected} frames descartados por el filtro de saltos ({rejected_pct:.1f} %): "
                                     "el umbral puede ser demasiado bajo para esta resolución o velocidad.")
        if self._recovered:
            reasons.append(f"{self._recovered} posición(es) recuperada(s) tras una pérdida: conviene confirmar que es el mismo sujeto.")
        return verdict, reasons


def assess_quality(
    records: Iterable,
    dt: float,
    thresholds: QualityThresholds = QualityThresholds(),
    max_step_px: Optional[float] = None,
) -> QualityReport:
    """Informe de calidad de una corrida ya terminada."""
    mon = QualityMonitor(dt, thresholds, max_step_px)
    for r in records:
        mon.update(r)
    return mon.report()


def quality_rows(report: QualityReport) -> List[List[object]]:
    """Filas ``[metrica, valor]`` para agregar al CSV de resumen (si se decide incluirlas)."""
    thr = "" if report.jump_threshold_px is None else f"{report.jump_threshold_px:.2f}"
    return [
        ["calidad_veredicto", report.verdict],
        ["calidad_frames_totales", report.frames_total],
        ["calidad_frames_perdidos", report.frames_lost],
        ["calidad_frames_perdidos_pct", f"{report.lost_pct:.4f}"],
        ["calidad_interrupciones", report.n_interruptions],
        ["calidad_interrupciones_por_min", f"{report.interruptions_per_min:.4f}"],
        ["calidad_hueco_mas_largo_s", f"{report.longest_gap_s:.4f}"],
        ["calidad_perdidos_al_inicio", report.leading_lost_frames],
        ["calidad_perdidos_al_final", report.trailing_lost_frames],
        ["calidad_saltos", report.n_jumps],
        ["calidad_umbral_salto_px", thr],
        ["calidad_frames_rechazados", report.frames_rejected],
        ["calidad_frames_recuperados", report.frames_recovered],
    ]


def format_quality(report: QualityReport) -> str:
    """Texto legible para mostrar al terminar una corrida."""
    lines = [
        f"Calidad del seguimiento: {report.verdict.upper()}",
        f"  Frames: {report.frames_detected} detectados de {report.frames_total} "
        f"({report.lost_pct:.1f} % perdidos)",
        f"  Interrupciones: {report.n_interruptions} ({report.interruptions_per_min:.1f} por minuto); "
        f"hueco más largo: {report.longest_gap_s:.2f} s",
        f"  Saltos sospechosos: {report.n_jumps}",
    ]
    if report.frames_rejected or report.frames_recovered:
        lines.append(f"  Descartados por el filtro de saltos: {report.frames_rejected}; recuperados: {report.frames_recovered}")
    lines += [f"  - {r}" for r in report.reasons]
    return "\n".join(lines)