"""Tecolote en TrackerLab: los tres niveles de asistencia y su preparación (sin ventanas, para que se pueda probar).

    plan = prepare(options.get("tecolote_mode", PROPONER), DETECTOR_MODEL, options.get("tecolote_every_s"))
    # plan.propose_model -> a choose_subject(model_path=...);  plan.corrector y plan.every_s -> al TrackingJob

Niveles (independientes del MODO de seguimiento: fondo, silueta o color):
  * apagado:  el seguimiento de siempre; el sujeto se marca a mano.
  * proponer: Tecolote propone al sujeto al inicio y la persona lo confirma.
  * corregir: además, cada ``every_s`` segundos verifica dónde está el ratón; corrige el desvío del seguimiento o lo
              reubica si se perdió (ver detection/corrector.py).
Nunca bloquea una corrida: sin modelo, o si el modelo no abre, avisa y baja de nivel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from detection.corrector import make_corrector

APAGADO, PROPONER, CORREGIR = "apagado", "proponer", "corregir"
MODES = (APAGADO, PROPONER, CORREGIR)
LABELS = {APAGADO: "Apagado", PROPONER: "Proponer sujeto", CORREGIR: "Proponer y corregir"}
DESCRIPTIONS = {
    APAGADO: "El seguimiento de siempre; el sujeto se marca a mano.",
    PROPONER: "Tecolote propone al sujeto al inicio y tú lo confirmas.",
    CORREGIR: "Además, cada cierto tiempo verifica dónde está el ratón: corrige el desvío del seguimiento o lo reubica si se pierde.",
}
DEFAULT_MODE = CORREGIR
DEFAULT_EVERY_S = 0.5
MIN_EVERY_S, MAX_EVERY_S = 0.05, 60.0


def parse_every(text: Any) -> float:
    """Segundos entre mediciones, escritos por una persona («0.5» o «0,5»). ValueError con un mensaje para mostrarle."""
    try:
        v = float(str(text).strip().replace(",", "."))
    except (TypeError, ValueError):
        raise ValueError("Escribe cada cuántos segundos mide Tecolote, por ejemplo 0.5.") from None
    if not (MIN_EVERY_S <= v <= MAX_EVERY_S):
        raise ValueError(f"Tecolote debe medir cada {MIN_EVERY_S:g} a {MAX_EVERY_S:g} segundos.")
    return v


@dataclass
class TecolotePlan:
    mode: str  # el nivel con el que realmente se trabajará (puede ser menor que el pedido)
    propose_model: Optional[str]  # modelo para proponer al sujeto, o None (selección manual)
    corrector: Optional[Any]  # TecoloteCorrector, o None
    every_s: float

    @property
    def corrects(self) -> bool:
        return self.corrector is not None


def prepare(mode: str, model_path: Optional[str], every_s: Any = None, *,
            corrector_factory: Callable[[str], Any] = make_corrector, out: Callable[[str], None] = print) -> TecolotePlan:
    """Prepara lo que hará Tecolote en esta corrida. Lanza ValueError si el nivel o la frecuencia no son válidos."""
    if mode not in MODES:
        raise ValueError(f"Nivel de Tecolote desconocido: {mode!r} (debe ser uno de {', '.join(MODES)}).")
    every = DEFAULT_EVERY_S if every_s is None else parse_every(every_s)
    if mode == APAGADO:
        return TecolotePlan(APAGADO, None, None, every)
    if not model_path:
        out("Tecolote está activado, pero no hay modelo configurado (DETECTOR_MODEL en main.py): se trabaja sin él.")
        return TecolotePlan(APAGADO, None, None, every)
    if mode == PROPONER:
        return TecolotePlan(PROPONER, model_path, None, every)
    try:
        corrector = corrector_factory(model_path)
    except Exception as exc:  # el modelo no existe, no abre o falta onnxruntime: la corrida sigue, sin corregir
        out(f"Tecolote no podrá corregir ({exc}). Seguirá intentando proponer al sujeto.")
        return TecolotePlan(PROPONER, model_path, None, every)
    out(f"Tecolote: proponer y corregir (verifica cada {every:g} s).")
    return TecolotePlan(CORREGIR, model_path, corrector, every)