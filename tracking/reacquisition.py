"""
Reencontrar al sujeto cuando el seguimiento lo pierde.

El seguimiento por silueta solo busca dentro de una ventana alrededor de la última posición: si el
sujeto sale de esa ventana (se movió rápido, hubo una oclusión, el seguimiento se corrió), no vuelve a
encontrarlo nunca. Un ``Reacquirer`` hace entonces una búsqueda en todo el recorte del ROI.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from tracking import Box, Detection, Point
from tracking.silhouette_tracker import SilhouetteTracker

__all__ = ["Reacquirer", "SilhouetteReacquirer"]


class Reacquirer(ABC):

    @abstractmethod
    def find(
        self,
        roi_bgr: np.ndarray,
        last_center: Optional[Point],
        last_bbox: Optional[Box],
    ) -> Optional[Detection]:
        """Devuelve la detección del sujeto (coordenadas del RECORTE del ROI) o ``None`` si no lo encuentra.

        last_center / last_bbox: última posición conocida (puede servir de pista, no es obligatorio usarla).
        """


class SilhouetteReacquirer(Reacquirer):
    """
    Elige el candidato más parecido al modelo (área, forma y polaridad) en todo el ROI.
    """

    def __init__(self, tracker: SilhouetteTracker):
        self.tracker = tracker

    def find(self, roi_bgr, last_center=None, last_bbox=None) -> Optional[Detection]:
        det = self.tracker.locate(roi_bgr)
        return det if det.found else None