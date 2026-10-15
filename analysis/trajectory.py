from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

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


    def reset(self) -> None:
        self.points.clear()


    def add_point(
        self,
        point: Optional[Point]
    ) -> None:

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
            and self.max_points > 0
            and len(self.points) > self.max_points
        ):
            self.points = self.points[
                -self.max_points:
            ]


    def get_points(
        self
    ) -> Tuple[Point, ...]:

        return tuple(
            self.points
        )


    def __len__(
        self
    ) -> int:

        return len(
            self.points
        )


    def draw(
        self,
        image_bgr: np.ndarray,
        offset: Point = (0.0, 0.0),
        thickness: int = 2,
        draw_last_point: bool = True
    ) -> np.ndarray:

        output = image_bgr.copy()

        if not self.points:
            return output

        ox, oy = offset

        pts = np.array(
            [
                [
                    int(round(x - ox)),
                    int(round(y - oy))
                ]
                for x, y in self.points
            ],
            dtype=np.int32
        )

        if len(pts) >= 2:

            cv2.polylines(
                output,
                [pts],
                False,
                (255, 255, 255),
                max(
                    1,
                    int(thickness)
                ),
                cv2.LINE_AA
            )

        if draw_last_point:

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
    records: Iterable
) -> List[Point]:

    points = []

    for record in records:

        x = getattr(
            record,
            "cx_abs",
            None
        )

        y_inv = getattr(
            record,
            "cy_abs_inv",
            None
        )

        if (
            x is None
            or y_inv is None
        ):
            continue

        points.append(
            (
                float(x),
                float(y_inv)
            )
        )

    return points


def trajectory_from_xy(
    xs: Sequence,
    ys: Sequence
) -> List[Point]:

    points = []

    for x, y in zip(
        xs,
        ys
    ):

        if (
            x is None
            or y is None
        ):
            continue

        try:

            points.append(
                (
                    float(x),
                    float(y)
                )
            )

        except (
            TypeError,
            ValueError
        ):
            continue

    return points


def draw_trajectory(
    image_bgr: np.ndarray,
    points: Sequence[Point],
    offset: Point = (0.0, 0.0),
    thickness: int = 2,
    draw_last_point: bool = True
) -> np.ndarray:

    tracker = TrajectoryTracker()

    for point in points:

        tracker.add_point(
            point
        )

    return tracker.draw(
        image_bgr,
        offset=offset,
        thickness=thickness,
        draw_last_point=draw_last_point
    )


def trajectory_length_px(
    points: Sequence[Point]
) -> float:

    if len(points) < 2:
        return 0.0

    total = 0.0

    for i in range(
        1,
        len(points)
    ):

        x1, y1 = points[i - 1]
        x2, y2 = points[i]

        total += (
            (
                x2 - x1
            ) ** 2
            +
            (
                y2 - y1
            ) ** 2
        ) ** 0.5

    return float(
        total
    )


def trajectory_length_cm(
    points: Sequence[Point],
    cm_per_px: Optional[float]
) -> Optional[float]:

    if cm_per_px is None:
        return None

    if cm_per_px <= 0:
        return None

    return (
        trajectory_length_px(
            points
        )
        * float(
            cm_per_px
        )
<<<<<<< HEAD
    )
=======
    )

def create_clean_background(
    reader,
    start_frame,
    end_frame,
    samples=35
):
    if end_frame <= start_frame:
        frame = reader.read_at(start_frame)

        if frame is None:
            return None

        return frame.copy()

    indices = np.linspace(
        start_frame,
        end_frame,
        num=min(
            samples,
            end_frame - start_frame + 1
        ),
        dtype=int
    )

    frames = []

    for index in indices:
        frame = reader.read_at(
            int(index)
        )

        if frame is not None:
            frames.append(
                frame.astype(
                    np.float32
                )
            )

    if not frames:
        return None

    stack = np.stack(
        frames,
        axis=0
    )

    background = np.median(
        stack,
        axis=0
    )

    return np.clip(
        background,
        0,
        255
    ).astype(
        np.uint8
    )

def save_trajectory_map(
    background,
    points,
    output_path,
    thickness=3
):
    if background is None:
        return False

    result = background.copy()

    valid_points = []

    for point in points:
        if point is None:
            continue

        x, y = point

        valid_points.append(
            (
                int(round(x)),
                int(round(y))
            )
        )

    if len(valid_points) >= 2:
        pts = np.array(
            valid_points,
            dtype=np.int32
        )

        cv2.polylines(
            result,
            [pts],
            False,
            (
                0,
                0,
                255
            ),
            thickness,
            cv2.LINE_AA
        )

    if valid_points:
        cv2.circle(
            result,
            valid_points[0],
            7,
            (
                0,
                255,
                0
            ),
            -1
        )

        cv2.circle(
            result,
            valid_points[-1],
            7,
            (
                255,
                0,
                0
            ),
            -1
        )

    return cv2.imwrite(
        output_path,
        result
    )
>>>>>>> origin/desarrollo-rafa
