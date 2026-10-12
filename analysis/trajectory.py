from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


class TrajectoryTracker:

    def __init__(
        self,
        max_points: Optional[int] = None
    ):
        self.max_points = max_points
        self.points: List[Point] = []


    def reset(self):
        self.points.clear()


    def add_point(
        self,
        point: Optional[Point]
    ):
        if point is None:
            return

        x, y = point

        self.points.append(
            (
                float(x),
                float(y)
            )
        )

        if (
            self.max_points is not None
            and len(self.points) > self.max_points
        ):
            self.points = self.points[
                -self.max_points:
            ]


    def get_points(self):
        return tuple(
            self.points
        )


    def draw(
        self,
        image,
        offset=(0, 0),
        thickness=2
    ):
        output = image.copy()

        if len(self.points) < 2:
            return output

        ox, oy = offset

        pts = np.array(
            [
                [
                    int(round(x - ox)),
                    int(round(y - oy))
                ]
                for x, y
                in self.points
            ],
            dtype=np.int32
        )

        cv2.polylines(
            output,
            [pts],
            False,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        last_x, last_y = pts[-1]

        cv2.circle(
            output,
            (
                int(last_x),
                int(last_y)
            ),
            5,
            (0, 255, 255),
            -1,
            cv2.LINE_AA
        )

        return output


def trajectory_from_records(
    records
):
    points = []

    for record in records:

        if (
            record.cx_abs is None
            or record.cy_abs_inv is None
        ):
            continue

        points.append(
            (
                float(record.cx_abs),
                float(record.cy_abs_inv)
            )
        )

    return points