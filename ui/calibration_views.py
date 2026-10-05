import cv2
import numpy as np


MAX_PREVIEW_W = 1280
MAX_PREVIEW_H = 800

FRAMESEL_WIN_W = 1400
FRAMESEL_WIN_H = 900


def contours_from_mask(mask):

    cnts, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = []

    for contour in cnts:

        if len(contour) < 3:
            continue

        eps = 0.002 * cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            eps,
            True
        )

        if len(approx) >= 3:
            points = [
                (
                    int(p[0][0]),
                    int(p[0][1])
                )
                for p in approx
            ]

        else:
            points = [
                (
                    int(p[0][0]),
                    int(p[0][1])
                )
                for p in contour
            ]

        result.append(
            points
        )

    return result


def make_zoom_view(
    image_bgr,
    title,
    points,
    zoom,
    pan_x,
    pan_y,
    unavailable_mask=None
):

    H, W = image_bgr.shape[:2]

    zoom = max(
        1.0,
        float(zoom)
    )

    view_w = int(
        min(
            W,
            max(
                50,
                W / zoom
            )
        )
    )

    view_h = int(
        min(
            H,
            max(
                50,
                H / zoom
            )
        )
    )

    pan_x = int(
        max(
            0,
            min(
                pan_x,
                W - view_w
            )
        )
    )

    pan_y = int(
        max(
            0,
            min(
                pan_y,
                H - view_h
            )
        )
    )

    crop = image_bgr[
        pan_y:pan_y + view_h,
        pan_x:pan_x + view_w
    ].copy()

    if unavailable_mask is not None:

        unavailable = unavailable_mask[
            pan_y:pan_y + view_h,
            pan_x:pan_x + view_w
        ]

        mask = unavailable > 0

        if np.any(mask):

            crop[mask] = (
                crop[mask] * 0.25
            ).astype(
                np.uint8
            )

            red = np.zeros_like(
                crop
            )

            red[:] = (
                0,
                0,
                180
            )

            crop[mask] = cv2.addWeighted(
                crop[mask],
                0.75,
                red[mask],
                0.25,
                0
            )

    scale = min(
        MAX_PREVIEW_W / float(view_w),
        MAX_PREVIEW_H / float(view_h)
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

    display_points = []

    for px, py in points:

        if (
            pan_x <= px < pan_x + view_w
            and
            pan_y <= py < pan_y + view_h
        ):

            display_points.append(
                (
                    int(
                        (px - pan_x)
                        * scale
                    ),
                    int(
                        (py - pan_y)
                        * scale
                    )
                )
            )

    for point in display_points:

        cv2.circle(
            display,
            point,
            5,
            (0, 255, 255),
            -1
        )

    if len(display_points) >= 2:

        for i in range(
            len(display_points) - 1
        ):

            cv2.line(
                display,
                display_points[i],
                display_points[i + 1],
                (0, 255, 0),
                2
            )

    if len(display_points) >= 3:

        cv2.line(
            display,
            display_points[-1],
            display_points[0],
            (0, 200, 0),
            1
        )

    cv2.putText(
        display,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        display,
        "Q/E zoom | WASD mover | ENTER aceptar | z deshacer | c limpiar",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return (
        display,
        scale,
        pan_x,
        pan_y,
        view_w,
        view_h
    )


def select_polygon_mask(
    image_bgr,
    title
):

    mask, points, _ = select_polygon_mask_core(
        image_bgr,
        title
    )

    return mask, points


def select_polygon_mask_exclusive(
    image_bgr,
    title,
    available_mask_full
):

    return select_polygon_mask_core(
        image_bgr,
        title,
        available_mask_full,
        True
    )


def select_polygon_mask_core(
    image_bgr,
    title,
    available_mask_full=None,
    mutually_exclusive=False
):

    H, W = image_bgr.shape[:2]

    if available_mask_full is None:

        available_mask_full = np.ones(
            (H, W),
            dtype=np.uint8
        ) * 255

    unavailable = None

    if mutually_exclusive:

        unavailable = cv2.bitwise_not(
            available_mask_full
        )

    points = []

    zoom = 1.0

    pan_x = 0
    pan_y = 0

    view = {
        "scale": 1.0,
        "pan_x": 0,
        "pan_y": 0
    }

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

    def redraw():

        nonlocal pan_x
        nonlocal pan_y

        display, scale, px, py, _, _ = make_zoom_view(
            image_bgr,
            title,
            points,
            zoom,
            pan_x,
            pan_y,
            unavailable
        )

        pan_x = px
        pan_y = py

        view["scale"] = scale
        view["pan_x"] = px
        view["pan_y"] = py

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

        if event != cv2.EVENT_LBUTTONDOWN:
            return

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

            if len(points) >= 3:

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
        (H, W),
        dtype=np.uint8
    )

    cv2.fillPoly(
        raw_mask,
        [points_np],
        255
    )

    if mutually_exclusive:

        final_mask = cv2.bitwise_and(
            raw_mask,
            available_mask_full
        )

    else:

        final_mask = raw_mask

    if cv2.countNonZero(
        final_mask
    ) <= 0:

        return (
            None,
            None,
            None
        )

    contours = contours_from_mask(
        final_mask
    )

    return (
        final_mask,
        points_np,
        contours
    )