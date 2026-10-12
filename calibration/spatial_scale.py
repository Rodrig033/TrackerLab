import cv2

from ui.calibration_views import (
    _make_zoom_view,
    FRAMESEL_WIN_W,
    FRAMESEL_WIN_H
)


def select_scale_line(
    image_bgr,
    title="ESCALA REAL (elige 2 puntos)"
):
    height, width = image_bgr.shape[:2]

    window = title

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window,
        FRAMESEL_WIN_W,
        FRAMESEL_WIN_H
    )

    points = []

    zoom = 1.0
    pan_x = 0
    pan_y = 0

    redraw = True
    done = False
    cancel = False

    view_state = {
        "scale": 1.0,
        "pan_x": 0,
        "pan_y": 0
    }

    def redraw_image():
        nonlocal pan_x
        nonlocal pan_y

        (
            display,
            scale,
            new_pan_x,
            new_pan_y,
            _,
            _
        ) = _make_zoom_view(
            image_bgr,
            title,
            points,
            zoom,
            pan_x,
            pan_y,
            unavailable_mask_full=None
        )

        pan_x = new_pan_x
        pan_y = new_pan_y

        view_state["scale"] = scale
        view_state["pan_x"] = new_pan_x
        view_state["pan_y"] = new_pan_y

        if len(points) == 2:
            p1 = points[0]
            p2 = points[1]

            px_distance = (
                (
                    p2[0] - p1[0]
                ) ** 2
                +
                (
                    p2[1] - p1[1]
                ) ** 2
            ) ** 0.5

            cv2.putText(
                display,
                f"Distancia seleccionada: {px_distance:.2f} px",
                (
                    20,
                    display.shape[0] - 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imshow(
            window,
            display
        )

    def on_mouse(
        event,
        x,
        y,
        flags,
        param
    ):
        nonlocal redraw

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(points) >= 2:
            return

        scale = view_state["scale"]

        px = int(
            round(
                view_state["pan_x"]
                + x / scale
            )
        )

        py = int(
            round(
                view_state["pan_y"]
                + y / scale
            )
        )

        px = max(
            0,
            min(
                px,
                width - 1
            )
        )

        py = max(
            0,
            min(
                py,
                height - 1
            )
        )

        points.append(
            (
                px,
                py
            )
        )

        redraw = True

    cv2.setMouseCallback(
        window,
        on_mouse
    )

    redraw_image()

    while True:
        if redraw:
            redraw_image()
            redraw = False

        key = (
            cv2.waitKey(
                20
            )
            & 0xFF
        )

        step = max(
            10,
            int(
                60 / zoom
            )
        )

        if key in (
            13,
            10
        ):
            if len(points) == 2:
                done = True
                break

        elif key == 27:
            cancel = True
            break

        elif key == ord("z"):
            if points:
                points.pop()
                redraw = True

        elif key == ord("c"):
            points = []
            redraw = True

        elif key in (
            ord("+"),
            ord("="),
            ord("q")
        ):
            zoom = min(
                12.0,
                zoom * 1.35
            )

            redraw = True

        elif key in (
            ord("-"),
            ord("_"),
            ord("e")
        ):
            zoom = max(
                1.0,
                zoom / 1.35
            )

            redraw = True

        elif key in (
            ord("a"),
            81
        ):
            pan_x -= step
            redraw = True

        elif key in (
            ord("d"),
            83
        ):
            pan_x += step
            redraw = True

        elif key in (
            ord("w"),
            82
        ):
            pan_y -= step
            redraw = True

        elif key in (
            ord("s"),
            84
        ):
            pan_y += step
            redraw = True

    cv2.destroyWindow(
        window
    )

    if cancel or not done:
        return None

    return points


def compute_scale_from_points(
    pt1,
    pt2,
    real_cm
):
    px_distance = (
        (
            pt2[0] - pt1[0]
        ) ** 2
        +
        (
            pt2[1] - pt1[1]
        ) ** 2
    ) ** 0.5

    if (
        real_cm is None
        or
        float(real_cm) <= 0
        or
        px_distance <= 0
    ):
        return None

    real_cm = float(
        real_cm
    )

    px_per_cm = (
        px_distance
        / real_cm
    )

    cm_per_px = (
        real_cm
        / px_distance
    )

    return {
        "pt1": [
            int(pt1[0]),
            int(pt1[1])
        ],

        "pt2": [
            int(pt2[0]),
            int(pt2[1])
        ],

        "real_cm":
            real_cm,

        "px_dist":
            float(px_distance),

        "px_per_cm":
            float(px_per_cm),

        "cm_per_px":
            float(cm_per_px)
    }