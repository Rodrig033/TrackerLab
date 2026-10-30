from __future__ import annotations

import numpy as np


def valid_values(records, field):
    values = []

    for record in records:
        value = getattr(record, field, None)

        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if np.isfinite(value):
            values.append(value)

    return np.array(values, dtype=float)


def calculate_statistics(values, total):
    if len(values) == 0:
        return {
            "n": 0,
            "faltantes": total,
            "media": "",
            "desviacion_estandar": "",
            "minimo": "",
            "q1": "",
            "mediana": "",
            "q3": "",
            "maximo": ""
        }

    return {
        "n": len(values),
        "faltantes": total - len(values),
        "media": float(np.mean(values)),
        "desviacion_estandar": float(np.std(values)),
        "minimo": float(np.min(values)),
        "q1": float(np.percentile(values, 25)),
        "mediana": float(np.median(values)),
        "q3": float(np.percentile(values, 75)),
        "maximo": float(np.max(values))
    }