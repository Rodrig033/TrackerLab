import cv2

from ui.calibration_views import (
    make_zoom_view,
    FRAMESEL_WIN_W,
    FRAMESEL_WIN_H
)


def select_xy_origin(
    image_bgr,
    title="ORIGEN XY (1 punto)"
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

    origin = None

    zoom = 1

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

        points = (
            []
            if origin is None
            else [origin]
        )

        display, scale, px, py, vw, vh = make_zoom_view(
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

        if origin is not None:

            ox, oy = origin

            if (
                pan_x <= ox < pan_x + vw
                and
                pan_y <= oy < pan_y + vh
            ):

                dx = int(
                    (ox - pan_x)
                    * scale
                )

                dy = int(
                    (oy - pan_y)
                    * scale
                )

                length = 90

                cv2.circle(
                    display,
                    (dx, dy),
                    6,
                    (0, 255, 255),
                    -1
                )

                cv2.line(
                    display,
                    (dx, dy),
                    (
                        min(
                            display.shape[1] - 1,
                            dx + length
                        ),
                        dy
                    ),
                    (0, 255, 0),
                    3
                )

                cv2.line(
                    display,
                    (dx, dy),
                    (
                        dx,
                        max(
                            0,
                            dy - length
                        )
                    ),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    display,
                    "X+",
                    (
                        dx + length,
                        dy
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    "Y+",
                    (
                        dx,
                        max(
                            20,
                            dy - length
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
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

        nonlocal origin

        if event == cv2.EVENT_LBUTTONDOWN:

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

            origin = (
                px,
                py
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

            if origin is not None:

                accepted = True
                break

        elif key == 27:

            break

        elif key in (
            ord("z"),
            ord("c")
        ):

            origin = None
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

    if (
        not accepted
        or
        origin is None
    ):

        return None

    return {

        "origin_px": [
            int(origin[0]),
            int(origin[1])
        ],

        "x_positive":
            "right",

        "y_positive":
            "up"
    }


def relative_coordinates(
    x,
    y,
    origin
):

    if origin is None:
        return x, y

    ox, oy = origin[
        "origin_px"
    ]

    relative_x = (
        x - ox
    )

    relative_y = (
        oy - y
    )

    return (
        relative_x,
        relative_y
    )