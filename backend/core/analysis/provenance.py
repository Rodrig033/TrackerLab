"""Filas de "cómo se obtuvo este resultado" para el final del resumen CSV."""

from __future__ import annotations
from typing import Any, List, Optional
from analysis.quality import assess_quality, quality_rows

__all__ = ["provenance_rows"]


def _num(value: Optional[float], decimals: int) -> Any:
    return "" if value is None else f"{float(value):.{decimals}f}"

def _plain(value: Any) -> Any:
    return "" if value is None else value

def _yes_no(value: Optional[bool]) -> str:
    return "" if value is None else ("si" if value else "no")

def provenance_rows(result, app_version: str) -> List[List[Any]]:
    """Filas ``[metrica, valor]`` para añadir al final del resumen.
       result:  ``TrackingResult`` (usa ``settings``, ``frames_reacquired``, ``records`` y ``timing``).
    """
    rows: List[List[Any]] = [["trackerlab_version", app_version]]

    s = getattr(result, "settings", None) or {}
    if not s:
        rows.append(["proc_opciones", "no registradas"])
    else:
        rows += [
            ["proc_modo_seguimiento", _plain(s.get("modo_seguimiento"))],
            ["proc_suavizado_alpha", _num(s.get("suavizado_alpha"), 4)],
            ["proc_salto_max_px", _num(s.get("salto_max_px"), 2)],
            ["proc_salto_max_vel_px_s", _num(s.get("salto_max_vel_px_s"), 2)],
            ["proc_recuperacion_tras_frames", _plain(s.get("recuperacion_tras_frames"))],
            ["proc_recuperacion_distancia_px", _num(s.get("recuperacion_distancia_px"), 2)],
            ["proc_adopcion_tras_s", _num(s.get("adopcion_tras_s"), 2)],
            ["proc_busqueda_sujeto", _yes_no(s.get("buscador_sujeto") is not None)],
            ["proc_busqueda_tras_frames", _plain(s.get("buscador_tras_frames"))],
            ["proc_busqueda_cada_frames", _plain(s.get("buscador_cada_frames"))],
            ["proc_seleccion_independiente_del_margen", _yes_no(s.get("seleccion_independiente_del_margen"))],
        ]
    rows.append(["proc_frames_reencontrados", getattr(result, "frames_reacquired", 0)])

    records = getattr(result, "records", None)
    if records:
        rows += quality_rows(assess_quality(records, result.timing.dt))
    else:
        rows.append(["calidad_veredicto", "no calculado (no se guardaron los registros por frame)"])
    return rows