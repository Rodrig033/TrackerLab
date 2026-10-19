from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

__all__ = ["Box", "iou", "center_distance", "SingleObjectEval", "evaluate_single_object"]

Box = Tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    """Intersección sobre unión de dos cajas (0 si no se tocan)."""
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: Box, b: Box) -> float:
    return math.dist((a[0] + a[2] / 2.0, a[1] + a[3] / 2.0), (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0))


@dataclass(frozen=True)
class SingleObjectEval:
    frames: int
    with_truth: int  # frames donde la referencia dice que el sujeto está
    true_positive: int  # lo encontró y bien ubicado (IoU >= umbral)
    wrong_place: int  # dio una caja, pero en otro lugar
    missed: int  # no dio caja y el sujeto estaba
    false_positive: int  # dio una caja donde no había sujeto (según la referencia)
    mean_iou: Optional[float]  # sobre los aciertos
    mean_center_error: Optional[float]  # px, sobre los aciertos
    iou_threshold: float

    @property
    def recall(self) -> Optional[float]:
        """Fracción de frames con sujeto que se ubicó bien."""
        return self.true_positive / self.with_truth if self.with_truth else None

    @property
    def precision(self) -> Optional[float]:
        """De las cajas que dio, fracción bien ubicada."""
        given = self.true_positive + self.wrong_place + self.false_positive
        return self.true_positive / given if given else None


def evaluate_single_object(
    predictions: Sequence[Optional[Box]],
    truths: Sequence[Optional[Box]],
    iou_threshold: float = 0.5,
) -> SingleObjectEval:
    """Compara, frame a frame, la caja predicha con la de referencia (``None`` = sin sujeto / sin predicción)."""
    if len(predictions) != len(truths):
        raise ValueError("predictions y truths deben tener la misma longitud (una entrada por frame).")
    tp = wrong = missed = fp = with_truth = 0
    ious, dists = [], []
    for p, t in zip(predictions, truths):
        if t is not None:
            with_truth += 1
            if p is None:
                missed += 1
            elif iou(p, t) >= iou_threshold:
                tp += 1
                ious.append(iou(p, t))
                dists.append(center_distance(p, t))
            else:
                wrong += 1
        elif p is not None:
            fp += 1
    return SingleObjectEval(
        frames=len(truths), with_truth=with_truth, true_positive=tp, wrong_place=wrong, missed=missed, false_positive=fp,
        mean_iou=sum(ious) / len(ious) if ious else None,
        mean_center_error=sum(dists) / len(dists) if dists else None, iou_threshold=iou_threshold,
    )