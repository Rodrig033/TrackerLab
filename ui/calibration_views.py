import cv2
import numpy as np


MAX_PREVIEW_W = 1280
MAX_PREVIEW_H = 800

FRAMESEL_WIN_W = 1400
FRAMESEL_WIN_H = 900


def contours_from_mask(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = []

    for contour in contours:
        if len(contour) < 3:
            continue

        epsilon = 0.002 * cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        source = (
            approx
            if len(approx) >= 3
            else contour
        )

        points = [
            (
                int(point[0][0]),
                int(point[0][1])
            )
            for point in source
        ]

        result.append(points)

    return result


def _draw_top_bar(
    image,
    title,
    points_count,
    zoom
):
    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (
            image.shape[1],
            95
        ),
        (20, 20, 20),
        -1
    )

    image = cv2.addWeighted(
        overlay,
        0.80,
        image,
        0.20,
        0
    )

    cv2.putText(
        image,
        title,
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    instructions = (
        f"Puntos: {points_count} | "
        f"Zoom: {zoom:.1f}x | "
        "Q/E zoom | WASD o flechas mover | "
        "Z deshacer | C limpiar | ENTER aceptar | ESC cancelar"
    )

    cv2.putText(
        image,
        instructions,
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (210, 210, 210),
        1,
        cv2.LINE_AA
    )

    return image


def _make_zoom_view(
    image_bgr,
    title,
    pts_full,
    zoom,
    pan_x,
    pan_y,
    unavailable_mask_full=None,
    max_w=MAX_PREVIEW_W,
    max_h=MAX_PREVIEW_H
):
    height, width = image_bgr.shape[:2]

    zoom = max(
        1.0,
        float(zoom)
    )

    view_w = int(
        min(
            width,
            max(
                50,
                width / zoom
            )
        )
    )

    view_h = int(
        min(
            height,
            max(
                50,
                height / zoom
            )
        )
    )

    pan_x = int(
        max(
            0,
            min(
                pan_x,
                width - view_w
            )
        )
    )

    pan_y = int(
        max(
            0,
            min(
                pan_y,
                height - view_h
            )
        )
    )

    crop = image_bgr[
        pan_y:pan_y + view_h,
        pan_x:pan_x + view_w
    ].copy()

    if unavailable_mask_full is not None:
        unavailable_crop = unavailable_mask_full[
            pan_y:pan_y + view_h,
            pan_x:pan_x + view_w
        ]

        blocked = (
            unavailable_crop > 0
        )

        if np.any(blocked):
            darkened = crop.copy()

            darkened[blocked] = (
                darkened[blocked] * 0.30
            ).astype(
                np.uint8
            )

            red_overlay = np.zeros_like(
                crop
            )

            red_overlay[:] = (
                40,
                40,
                160
            )

            darkened[blocked] = cv2.addWeighted(
                darkened[blocked],
                0.75,
                red_overlay[blocked],
                0.25,
                0
            )

            crop = darkened

    scale = min(
        max_w / float(view_w),
        max_h / float(view_h)
    )

    scale = max(
        0.1,
        scale
    )

    display = cv2.resize(
        crop,
        (
            int(view_w * scale),
            int(view_h * scale)
        )
    )

    visible_points = []

    for px, py in pts_full:
        if (
            pan_x <= px < pan_x + view_w
            and
            pan_y <= py < pan_y + view_h
        ):
            visible_points.append(
                (
                    int(
                        (px - pan_x) * scale
                    ),
                    int(
                        (py - pan_y) * scale
                    )
                )
            )

    for point in visible_points:
        cv2.circle(
            display,
            point,
            6,
            (0, 220, 255),
            -1,
            cv2.LINE_AA
        )

    if len(visible_points) >= 2:
        for index in range(
            len(visible_points) - 1
        ):
            cv2.line(
                display,
                visible_points[index],
                visible_points[index + 1],
                (0, 220, 0),
                2,
                cv2.LINE_AA
            )

    if len(visible_points) >= 3:
        cv2.line(
            display,
            visible_points[-1],
            visible_points[0],
            (0, 180, 0),
            2,
            cv2.LINE_AA
        )

    display = _draw_top_bar(
        display,
        title,
        len(pts_full),
        zoom
    )

    return (
        display,
        scale,
        pan_x,
        pan_y,
        view_w,
        view_h
    )


def _select_polygon_mask_zoom_core(
    image_bgr,
    title,
    available_mask_full=None,
    mutually_exclusive=False
):
    height, width = image_bgr.shape[:2]

    if available_mask_full is None:
        available_mask_full = (
            np.ones(
                (
                    height,
                    width
                ),
                dtype=np.uint8
            ) * 255
        )

    unavailable_mask_full = (
        cv2.bitwise_not(
            available_mask_full
        )
        if mutually_exclusive
        else None
    )

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
            unavailable_mask_full=unavailable_mask_full
        )

        pan_x = new_pan_x
        pan_y = new_pan_y

        view_state["scale"] = scale
        view_state["pan_x"] = new_pan_x
        view_state["pan_y"] = new_pan_y

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

        scale = view_state[
            "scale"
        ]

        px = int(
            round(
                view_state[
                    "pan_x"
                ]
                + x / scale
            )
        )

        py = int(
            round(
                view_state[
                    "pan_y"
                ]
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
            if len(points) >= 3:
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
        return (
            None,
            None,
            None
        )

    points_np = np.array(
        points,
        dtype=np.int32
    )

    raw_mask = np.zeros(
        (
            height,
            width
        ),
        dtype=np.uint8
    )

    cv2.fillPoly(
        raw_mask,
        [
            points_np
        ],
        255
    )

    if mutually_exclusive:
        final_mask = cv2.bitwise_and(
            raw_mask,
            available_mask_full
        )
    else:
        final_mask = raw_mask

    if (
        cv2.countNonZero(
            final_mask
        )
        <= 0
    ):
        print(
            "La zona dibujada quedó vacía."
        )

        return (
            None,
            None,
            None
        )

    final_contours = contours_from_mask(
        final_mask
    )

    if not final_contours:
        print(
            "No se pudo obtener un contorno válido."
        )

        return (
            None,
            None,
            None
        )

    return (
        final_mask,
        points_np,
        final_contours
    )


def select_polygon_mask(
    image_bgr,
    title
):
    mask, points, _ = (
        _select_polygon_mask_zoom_core(
            image_bgr,
            title,
            available_mask_full=None,
            mutually_exclusive=False
        )
    )

    return (
        mask,
        points
    )


def select_polygon_mask_exclusive(
    image_bgr,
    title,
    available_mask_full=None
):
    return _select_polygon_mask_zoom_core(
        image_bgr,
        title,
        available_mask_full=available_mask_full,
        mutually_exclusive=True
    )


def shrink_mask(
    mask,
    px
):
    px = int(
        px
    )

    if px <= 0:
        return mask

    kernel_size = (
        2 * px
        + 1
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size
        )
    )

    return cv2.erode(
        mask,
        kernel,
        iterations=1
    )