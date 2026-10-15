from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

__all__ = ["Point", "Hit", "VERTEX", "MIDPOINT", "PolygonEditor", "polygon_self_intersects", "polygon_area"]

Point = Tuple[int, int]

VERTEX = "vertex"
MIDPOINT = "midpoint"

_MAX_HISTORY = 200


@dataclass(frozen=True)
class Hit:
    """Lo que hay bajo el cursor: un vértice o el centro de un lado."""

    kind: str  # VERTEX o MIDPOINT
    index: int  # índice del vértice; para MIDPOINT, índice i del lado i -> (i+1) % n


# ----------------------------------------------------------------------
# Geometría
# ----------------------------------------------------------------------
def polygon_area(points: Sequence[Point]) -> float:
    """Área (fórmula del cordón); 0.0 con menos de 3 puntos o puntos alineados."""
    n = len(points)
    if n < 3:
        return 0.0
    s = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _orient(a: Point, b: Point, c: Point) -> int:
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (v > 0) - (v < 0)


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    """¿c, alineado con a-b, cae dentro del segmento?"""
    return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])


def _segments_touch(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    o1, o2 = _orient(p1, p2, p3), _orient(p1, p2, p4)
    o3, o4 = _orient(p3, p4, p1), _orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and _on_segment(p1, p2, p3)) or (o2 == 0 and _on_segment(p1, p2, p4))
            or (o3 == 0 and _on_segment(p3, p4, p1)) or (o4 == 0 and _on_segment(p3, p4, p2)))


def polygon_self_intersects(points: Sequence[Point]) -> bool:
    """True si algún par de lados NO contiguos se cruza o se toca (polígono cerrado)."""
    n = len(points)
    if n < 4:
        return False
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue  # lados contiguos: comparten un vértice
            if _segments_touch(a, b, points[j], points[(j + 1) % n]):
                return True
    return False


# ----------------------------------------------------------------------
# Editor
# ----------------------------------------------------------------------
class PolygonEditor:
    """Estado y reglas de edición de un polígono."""

    #: Radio (en píxeles de pantalla) para atrapar un vértice o el centro de un lado.
    VERTEX_RADIUS_PX = 12.0
    MIDPOINT_RADIUS_PX = 9.0

    def __init__(self, bounds: Tuple[int, int], points: Sequence[Point] = ()):
        """bounds = (ancho, alto) de la imagen: los vértices no salen de ella."""
        self.width, self.height = int(bounds[0]), int(bounds[1])
        if self.width <= 0 or self.height <= 0:
            raise ValueError("bounds debe ser (ancho, alto) positivos")
        self._points: List[Point] = [self._clamp(p[0], p[1]) for p in points]
        self._undo: List[Tuple[Point, ...]] = []
        self._redo: List[Tuple[Point, ...]] = []
        self.selected: Optional[int] = None
        self.hover: Optional[Hit] = None
        self._drag_index: Optional[int] = None
        self._drag_origin: Optional[Tuple[Point, ...]] = None  # estado previo, si falta registrar el movimiento

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------
    @property
    def points(self) -> List[Point]:
        return list(self._points)

    @property
    def dragging(self) -> bool:
        return self._drag_index is not None

    @property
    def can_accept(self) -> bool:
        """Mínimo para aceptar el polígono (igual que antes: 3 puntos)."""
        return len(self._points) >= 3

    @property
    def self_intersects(self) -> bool:
        return polygon_self_intersects(self._points)

    @property
    def warning(self) -> Optional[str]:
        """Texto (solo ASCII, para cv2.putText) si el polígono tiene un problema evidente."""
        if len(self._points) >= 3 and polygon_area(self._points) == 0:
            return "ATENCION: los puntos estan alineados (area cero)"
        if self.self_intersects:
            return "ATENCION: el poligono se cruza a si mismo"
        return None

    def midpoints(self) -> List[Tuple[int, Tuple[float, float]]]:
        """Centros de los lados, ``(índice del lado, (x, y))``. Solo con 3 o más vértices."""
        n = len(self._points)
        if n < 3:
            return []
        out = []
        for i in range(n):
            (x1, y1), (x2, y2) = self._points[i], self._points[(i + 1) % n]
            out.append((i, ((x1 + x2) / 2.0, (y1 + y2) / 2.0)))
        return out

    def hit_test(self, x: float, y: float, scale: float) -> Optional[Hit]:
        """Qué hay en (x, y) (coordenadas de imagen). Los vértices tienen prioridad."""
        scale = max(float(scale), 1e-6)
        best, best_d2 = None, None
        tol_v = self.VERTEX_RADIUS_PX / scale
        for i, (px, py) in enumerate(self._points):
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 <= tol_v ** 2 and (best_d2 is None or d2 < best_d2):
                best, best_d2 = Hit(VERTEX, i), d2
        if best is not None:
            return best
        tol_m = self.MIDPOINT_RADIUS_PX / scale
        for i, (mx, my) in self.midpoints():
            d2 = (mx - x) ** 2 + (my - y) ** 2
            if d2 <= tol_m ** 2 and (best_d2 is None or d2 < best_d2):
                best, best_d2 = Hit(MIDPOINT, i), d2
        return best

    # ------------------------------------------------------------------
    # Historial
    # ------------------------------------------------------------------
    def _snapshot(self) -> Tuple[Point, ...]:
        return tuple(self._points)

    def _record(self, before: Tuple[Point, ...]) -> None:
        self._undo.append(before)
        if len(self._undo) > _MAX_HISTORY:
            del self._undo[0]
        self._redo.clear()

    def _clamp(self, x: float, y: float) -> Point:
        return (min(max(int(round(x)), 0), self.width - 1), min(max(int(round(y)), 0), self.height - 1))

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------
    def hover_at(self, x: float, y: float, scale: float) -> bool:
        """Actualiza lo que está bajo el cursor. Devuelve True si cambió (hay que redibujar)."""
        new = self.hit_test(x, y, scale)
        changed = new != self.hover
        self.hover = new
        return changed

    def press(self, x: float, y: float, scale: float) -> str:
        """Botón presionado en (x, y). Devuelve la acción: ``"vertex"``, ``"insert"`` o ``"add"``.

        Siempre deja un arrastre activo sobre el vértice tocado, insertado o agregado, de modo
        que se puede acomodar sin soltar el botón.
        """
        self.release()  # por si un 'soltar' anterior no llegó
        hit = self.hit_test(x, y, scale)

        if hit is not None and hit.kind == VERTEX:
            self.selected = hit.index
            self._drag_index = hit.index
            self._drag_origin = self._snapshot()  # el movimiento se registra al soltar, si hubo
            return "vertex"

        if hit is not None and hit.kind == MIDPOINT:
            before = self._snapshot()
            mx, my = dict(self.midpoints())[hit.index]
            self._points.insert(hit.index + 1, self._clamp(mx, my))
            self._record(before)
            self.selected = self._drag_index = hit.index + 1
            self._drag_origin = None
            return "insert"

        before = self._snapshot()
        self._points.append(self._clamp(x, y))
        self._record(before)
        self.selected = self._drag_index = len(self._points) - 1
        self._drag_origin = None
        return "add"

    def drag_to(self, x: float, y: float) -> bool:
        """Mueve el vértice que se arrastra. Devuelve True si cambió de lugar."""
        if self._drag_index is None:
            return False
        new = self._clamp(x, y)
        if new == self._points[self._drag_index]:
            return False
        self._points[self._drag_index] = new
        return True

    def release(self) -> bool:
        """Botón soltado. Registra el movimiento (si hubo) como UNA acción deshacible."""
        if self._drag_index is None:
            return False
        changed = False
        if self._drag_origin is not None and self._drag_origin != self._snapshot():
            self._record(self._drag_origin)
            changed = True
        self._drag_index = None
        self._drag_origin = None
        return changed

    # ------------------------------------------------------------------
    # Acciones de teclado
    # ------------------------------------------------------------------
    def delete_selected(self) -> bool:
        """Borra el vértice seleccionado. Devuelve False si no hay ninguno."""
        self.release()
        if self.selected is None or not (0 <= self.selected < len(self._points)):
            return False
        before = self._snapshot()
        del self._points[self.selected]
        self._record(before)
        self.selected = None
        self.hover = None
        return True

    def clear(self) -> bool:
        """Quita todos los puntos. Se puede deshacer con ``undo()``."""
        self.release()
        if not self._points:
            return False
        before = self._snapshot()
        self._points = []
        self._record(before)
        self.selected = None
        self.hover = None
        return True

    def undo(self) -> bool:
        self.release()
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._points = list(self._undo.pop())
        self.selected = None
        self.hover = None
        return True

    def redo(self) -> bool:
        self.release()
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._points = list(self._redo.pop())
        self.selected = None
        self.hover = None
        return True