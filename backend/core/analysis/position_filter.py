from __future__ import annotations

from typing import Optional, Tuple


Point = Tuple[float, float]


class PositionEMAFilter:

    def __init__(self, alpha=0.35):
        self.alpha = float(alpha)
        self.last_point: Optional[Point] = None

    def reset(self):
        self.last_point = None

    def update(
        self,
        point: Optional[Point]
    ) -> Optional[Point]:

        if point is None:
            return None

        x, y = point

        if self.last_point is None:
            self.last_point = (
                float(x),
                float(y)
            )

            return self.last_point

        last_x, last_y = self.last_point

        new_x = (
            self.alpha * float(x)
            +
            (1.0 - self.alpha) * last_x
        )

        new_y = (
            self.alpha * float(y)
            +
            (1.0 - self.alpha) * last_y
        )

        self.last_point = (
            new_x,
            new_y
        )

        return self.last_point


class JumpRejector:

    def __init__(self, max_distance_px=120.0):
        self.max_distance_px = float(
            max_distance_px
        )

        self.last_point: Optional[Point] = None

    def reset(self):
        self.last_point = None

    def update(
        self,
        point: Optional[Point]
    ) -> Optional[Point]:

        if point is None:
            return None

        x, y = point

        if self.last_point is None:
            self.last_point = (
                float(x),
                float(y)
            )

            return self.last_point

        last_x, last_y = self.last_point

        distance = (
            (
                float(x) - last_x
            ) ** 2
            +
            (
                float(y) - last_y
            ) ** 2
        ) ** 0.5

        if distance > self.max_distance_px:
            return None

        self.last_point = (
            float(x),
            float(y)
        )

        return self.last_point

    def accept(
        self,
        point: Optional[Point]
    ) -> Optional[Point]:

        if point is None:
            return None

        x, y = point

        self.last_point = (
            float(x),
            float(y)
        )

        return self.last_point

    def distance_to_last(
        self,
        point: Optional[Point]
    ) -> Optional[float]:

        if point is None:
            return None

        if self.last_point is None:
            return 0.0

        x, y = point

        last_x, last_y = self.last_point

        distance = (
            (
                float(x) - last_x
            ) ** 2
            +
            (
                float(y) - last_y
            ) ** 2
        ) ** 0.5

        return distance


    