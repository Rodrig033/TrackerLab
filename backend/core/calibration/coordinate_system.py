import cv2

from ui.calibration_views import (
    _make_zoom_view,
    FRAMESEL_WIN_W,
    FRAMESEL_WIN_H
)


def select_xy_origin(
    image_bgr,
    title="ORIGEN XY (elige 1 punto)"
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

    origin = None

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

        points = (
            []
            if origin is None
            else [origin]
        )

        (
            display,
            scale,
            new_pan_x,
            new_pan_y,
            view_w,
            view_h
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

        if origin is not None:
            ox, oy = origin

            if (
                pan_x <= ox < pan_x + view_w
                and
                pan_y <= oy < pan_y + view_h
            ):
                display_x = int(
                    (ox - pan_x)
                    * scale
                )

                display_y = int(
                    (oy - pan_y)
                    * scale
                )

                axis_length = 100

                cv2.circle(
                    display,
                    (
                        display_x,
                        display_y
                    ),
                    7,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA
                )

                cv2.arrowedLine(
                    display,
                    (
                        display_x,
                        display_y
                    ),
                    (
                        min(
                            display.shape[1] - 10,
                            display_x + axis_length
                        ),
                        display_y
                    ),
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                    tipLength=0.15
                )

                cv2.arrowedLine(
                    display,
                    (
                        display_x,
                        display_y
                    ),
                    (
                        display_x,
                        max(
                            10,
                            display_y - axis_length
                        )
                    ),
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                    tipLength=0.15
                )

                cv2.putText(
                    display,
                    "X+",
                    (
                        min(
                            display.shape[1] - 45,
                            display_x + axis_length + 5
                        ),
                        display_y + 5
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    display,
                    "Y+",
                    (
                        display_x + 8,
                        max(
                            25,
                            display_y - axis_length
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    display,
                    f"Origen seleccionado: ({ox}, {oy}) px",
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
        nonlocal origin
        nonlocal redraw

        if event != cv2.EVENT_LBUTTONDOWN:
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

        origin = (
            px,
            py
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
            if origin is not None:
                done = True
                break

        elif key == 27:
            cancel = True
            break

        elif key in (
            ord("z"),
            ord("c")
        ):
            origin = None
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

    if cancel or not done or origin is None:
        return None

    return {
        "origin_px": [
            int(origin[0]),
            int(origin[1])
        ],
        "x_positive": "right",
        "y_positive": "up"
    }


def relative_coordinates(
    x,
    y,
    xy_origin_info
):
    if xy_origin_info is None:
        return (
            x,
            y
        )

    origin = xy_origin_info.get(
        "origin_px"
    )

    if (
        origin is None
        or
        len(origin) != 2
    ):
        return (
            x,
            y
        )

    ox = float(
        origin[0]
    )

    oy = float(
        origin[1]
    )

    relative_x = (
        float(x) - ox
    )

    relative_y = (
        oy - float(y)
    )

    return (
        relative_x,
        relative_y
    )