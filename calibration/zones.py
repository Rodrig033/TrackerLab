import cv2
import numpy as np

from ui.dialogs import (
    ask_yes_no,
    gui_ask_string
)

from ui.calibration_views import (
    select_polygon_mask,
    select_polygon_mask_exclusive,
    contours_from_mask
)


def sanitize_name(name):

    text = name.strip().lower()

    result = []

    for char in text:

        if (
            char.isalnum()
            or
            char == "_"
        ):

            result.append(
                char
            )

        elif char in (
            " ",
            "-"
        ):

            result.append(
                "_"
            )

    text = "".join(
        result
    )

    while "__" in text:

        text = text.replace(
            "__",
            "_"
        )

    return text.strip(
        "_"
    )


def mask_from_pts(
    shape_hw,
    points
):

    mask = np.zeros(
        shape_hw,
        dtype=np.uint8
    )

    points_np = np.array(
        points,
        dtype=np.int32
    )

    cv2.fillPoly(
        mask,
        [points_np],
        255
    )

    return mask


def mask_from_contours(
    shape_hw,
    contours
):

    mask = np.zeros(
        shape_hw,
        dtype=np.uint8
    )

    if not contours:
        return mask

    polygons = []

    for contour in contours:

        arr = np.array(
            contour,
            dtype=np.int32
        )

        if len(arr) >= 3:
            polygons.append(
                arr
            )

    if polygons:

        cv2.fillPoly(
            mask,
            polygons,
            255
        )

    return mask


def centroid_from_mask(mask):

    moments = cv2.moments(
        mask
    )

    if moments["m00"] <= 1e-12:
        return None

    cx = (
        moments["m10"]
        / moments["m00"]
    )

    cy = (
        moments["m01"]
        / moments["m00"]
    )

    return (
        float(cx),
        float(cy)
    )


def point_in_mask(
    mask,
    x,
    y
):

    h, w = mask.shape[:2]

    x = int(
        round(x)
    )

    y = int(
        round(y)
    )

    if (
        x < 0
        or
        x >= w
        or
        y < 0
        or
        y >= h
    ):

        return 0

    return (
        1
        if mask[y, x] > 0
        else 0
    )


def select_zones(
    image_bgr,
    roi_mask_full=None,
    mutually_exclusive=True
):

    zones = []

    H, W = image_bgr.shape[:2]

    if roi_mask_full is not None:

        available_mask = (
            roi_mask_full.copy()
        )

    else:

        available_mask = np.ones(
            (H, W),
            dtype=np.uint8
        ) * 255

    while True:

        add_zone = ask_yes_no(
            "¿Agregar una zona interna?",
            default_yes=False
        )

        if not add_zone:
            break

        name = gui_ask_string(
            "Nombre de zona",
            "Nombre de la zona:",
            default=f"zona_{len(zones) + 1}"
        )

        if not name:

            name = (
                f"zona_{len(zones) + 1}"
            )

        name = sanitize_name(
            name
        )

        if mutually_exclusive:

            mask, points, contours = (
                select_polygon_mask_exclusive(
                    image_bgr,
                    f"ZONA EXCLUYENTE: {name}",
                    available_mask
                )
            )

        else:

            mask, points = (
                select_polygon_mask(
                    image_bgr,
                    f"ZONA: {name}"
                )
            )

            if mask is not None:

                if roi_mask_full is not None:

                    mask = cv2.bitwise_and(
                        mask,
                        roi_mask_full
                    )

                contours = contours_from_mask(
                    mask
                )

            else:

                contours = None

        if (
            mask is None
            or
            contours is None
        ):

            print(
                "Zona cancelada o inválida."
            )

            continue

        if roi_mask_full is not None:

            mask = cv2.bitwise_and(
                mask,
                roi_mask_full
            )

        area = int(
            cv2.countNonZero(
                mask
            )
        )

        if area <= 0:

            print(
                "La zona quedó vacía."
            )

            continue

        centroid = centroid_from_mask(
            mask
        )

        zone = {

            "name":
                name,

            "contours_full":
                contours,

            "mask_full":
                mask.copy(),

            "centroid_full":
                centroid,

            "raw_pts_full":
                [
                    (
                        int(x),
                        int(y)
                    )
                    for x, y
                    in points.tolist()
                ]
                if points is not None
                else []
        }

        zones.append(
            zone
        )

        if mutually_exclusive:

            available_mask = cv2.bitwise_and(
                available_mask,
                cv2.bitwise_not(
                    mask
                )
            )

        print(
            f"Zona agregada: {name}"
        )

    return zones


def zones_to_crop(
    zones,
    x0,
    y0,
    w0,
    h0
):

    result = []

    for zone in zones:

        mask_crop = zone[
            "mask_full"
        ][
            y0:y0 + h0,
            x0:x0 + w0
        ].copy()

        contours_crop = contours_from_mask(
            mask_crop
        )

        centroid_crop = None

        if zone.get(
            "centroid_full"
        ) is not None:

            centroid_crop = (

                zone[
                    "centroid_full"
                ][0] - x0,

                zone[
                    "centroid_full"
                ][1] - y0
            )

        result.append(
            {

                "name":
                    zone["name"],

                "mask_crop":
                    mask_crop,

                "contours_crop":
                    contours_crop,

                "centroid_crop":
                    centroid_crop
            }
        )

    return result