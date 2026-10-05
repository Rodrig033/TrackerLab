import cv2

from ui.calibration_views import (
    make_zoom_view,
    FRAMESEL_WIN_W,
    FRAMESEL_WIN_H
)


def select_scale_line(
    image_bgr,
    title="ESCALA (2 puntos)"
):

    H, W = image_bgr.shape[:2]

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

    view = {
        "scale": 1,
        "pan_x": 0,
        "pan_y": 0
    }

    def redraw():

        nonlocal pan_x
        nonlocal pan_y

        display, scale, px, py, _, _ = make_zoom_view(
            image_bgr,
            title,
            points,
            zoom,
            pan_x,
            pan_y
        )

        pan_x = px
        pan_y = py

        view["scale"] = scale
        view["pan_x"] = px
        view["pan_y"] = py

        if len(points) == 2:

            p1, p2 = points

            distance = (
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
                f"Distancia: {distance:.2f} px",
                (10, 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        cv2.imshow(
            window,
            display
        )

    def mouse(
        event,
        x,
        y,
        flags,
        param
    ):

        if (
            event
            == cv2.EVENT_LBUTTONDOWN
            and
            len(points) < 2
        ):

            scale = view["scale"]

            px = int(
                round(
                    view["pan_x"]
                    + x / scale
                )
            )

            py = int(
                round(
                    view["pan_y"]
                    + y / scale
                )
            )

            px = max(
                0,
                min(
                    px,
                    W - 1
                )
            )

            py = max(
                0,
                min(
                    py,
                    H - 1
                )
            )

            points.append(
                (
                    px,
                    py
                )
            )

            redraw()

    cv2.setMouseCallback(
        window,
        mouse
    )

    redraw()

    accepted = False

    while True:

        key = cv2.waitKey(
            20
        ) & 0xFF

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

                accepted = True

                break

        elif key == 27:
            break

        elif key == ord("z"):

            if points:
                points.pop()
                redraw()

        elif key == ord("c"):

            points.clear()
            redraw()

        elif key in (
            ord("+"),
            ord("="),
            ord("q")
        ):

            zoom = min(
                12,
                zoom * 1.35
            )

            redraw()

        elif key in (
            ord("-"),
            ord("_"),
            ord("e")
        ):

            zoom = max(
                1,
                zoom / 1.35
            )

            redraw()

        elif key in (
            ord("a"),
            81
        ):

            pan_x -= step
            redraw()

        elif key in (
            ord("d"),
            83
        ):

            pan_x += step
            redraw()

        elif key in (
            ord("w"),
            82
        ):

            pan_y -= step
            redraw()

        elif key in (
            ord("s"),
            84
        ):

            pan_y += step
            redraw()

    cv2.destroyWindow(
        window
    )

    if not accepted:
        return None

    return points


def compute_scale_from_points(
    pt1,
    pt2,
    real_cm
):

    px_dist = (
        (
            pt2[0] - pt1[0]
        ) ** 2
        +
        (
            pt2[1] - pt1[1]
        ) ** 2
    ) ** 0.5

    if (
        real_cm <= 0
        or
        px_dist <= 0
    ):
        return None

    px_per_cm = (
        px_dist
        / float(real_cm)
    )

    cm_per_px = (
        float(real_cm)
        / px_dist
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
            float(real_cm),

        "px_dist":
            float(px_dist),

        "px_per_cm":
            float(px_per_cm),

        "cm_per_px":
            float(cm_per_px)
    }