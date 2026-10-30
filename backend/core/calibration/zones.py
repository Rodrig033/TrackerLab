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

from persistence.experiment_config import (
    zone_mask_path
)


def sanitize_name(name):
    text = str(
        name
    ).strip().lower()

    result = []

    for char in text:
        if (
            char.isalnum()
            or char == "_"
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

    if points is None:
        return mask

    points_np = np.array(
        points,
        dtype=np.int32
    )

    if len(points_np) >= 3:
        cv2.fillPoly(
            mask,
            [
                points_np
            ],
            255
        )

    return mask


def mask_from_contours(
    shape_hw,
    contours_pts
):
    mask = np.zeros(
        shape_hw,
        dtype=np.uint8
    )

    if contours_pts is None:
        return mask

    polygons = []

    for contour in contours_pts:
        array = np.array(
            contour,
            dtype=np.int32
        )

        if len(array) >= 3:
            polygons.append(
                array
            )

    if polygons:
        cv2.fillPoly(
            mask,
            polygons,
            255
        )

    return mask


def centroid_from_mask(mask):
    if mask is None:
        return None

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
    if mask is None:
        return 0

    height, width = mask.shape[:2]

    xi = int(
        round(x)
    )

    yi = int(
        round(y)
    )

    if (
        xi < 0
        or xi >= width
        or yi < 0
        or yi >= height
    ):
        return 0

    return (
        1
        if mask[
            yi,
            xi
        ] > 0
        else 0
    )


def select_zones(
    image_bgr,
    roi_mask_full=None,
    mutually_exclusive=True,
    exp_name=None
):
    zones = []

    height, width = image_bgr.shape[:2]

    if roi_mask_full is not None:
        available_mask = (
            roi_mask_full.copy()
        )

    else:
        available_mask = (
            np.ones(
                (
                    height,
                    width
                ),
                dtype=np.uint8
            )
            * 255
        )

    while True:
        add_more = ask_yes_no(
            "¿Agregar una zona interna?",
            default_yes=False
        )

        if not add_more:
            break

        default_name = (
            f"zona_{len(zones) + 1}"
        )

        zone_name = gui_ask_string(
            "Nombre de zona",
            "Nombre de la zona:",
            default=default_name
        )

        if zone_name is None:
            zone_name = ""

        if zone_name == "":
            zone_name = default_name

        zone_name = sanitize_name(
            zone_name
        )

        if zone_name == "":
            zone_name = default_name

        if mutually_exclusive:
            (
                zone_mask,
                raw_points,
                zone_contours
            ) = select_polygon_mask_exclusive(
                image_bgr,
                f"ZONA EXCLUYENTE: {zone_name}",
                available_mask_full=available_mask
            )

        else:
            (
                zone_mask,
                raw_points
            ) = select_polygon_mask(
                image_bgr,
                f"ZONA: {zone_name}"
            )

            if zone_mask is not None:
                if roi_mask_full is not None:
                    zone_mask = cv2.bitwise_and(
                        zone_mask,
                        roi_mask_full
                    )

                zone_contours = contours_from_mask(
                    zone_mask
                )

            else:
                zone_contours = None

        if (
            zone_mask is None
            or zone_contours is None
        ):
            print(
                "Zona cancelada o inválida."
            )

            continue

        if roi_mask_full is not None:
            zone_mask = cv2.bitwise_and(
                zone_mask,
                roi_mask_full
            )

        area_final = int(
            cv2.countNonZero(
                zone_mask
            )
        )

        if area_final <= 0:
            print(
                "La zona final quedó vacía."
            )

            continue

        centroid = centroid_from_mask(
            zone_mask
        )

        raw_points_list = []

        if raw_points is not None:
            raw_points_array = np.array(
                raw_points
            ).reshape(
                -1,
                2
            )

            raw_points_list = [
                (
                    int(x),
                    int(y)
                )
                for x, y
                in raw_points_array
            ]

        zone_record = {
            "name":
                zone_name,

            "contours_full":
                zone_contours,

            "mask_full":
                zone_mask.copy(),

            "centroid_full":
                centroid,

            "raw_pts_full":
                raw_points_list
        }

        if exp_name is not None:
            mask_file = zone_mask_path(
                exp_name,
                zone_name
            )

            if cv2.imwrite(
                mask_file,
                zone_mask
            ):
                zone_record[
                    "mask_file"
                ] = mask_file

        zones.append(
            zone_record
        )

        if mutually_exclusive:
            available_mask = cv2.bitwise_and(
                available_mask,
                cv2.bitwise_not(
                    zone_mask
                )
            )

        print(
            f"Zona agregada: {zone_name} | "
            f"área final: {area_final} px"
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
        mask_full = zone.get(
            "mask_full"
        )

        if mask_full is None:
            continue

        mask_crop = mask_full[
            y0:y0 + h0,
            x0:x0 + w0
        ].copy()

        contours_crop = contours_from_mask(
            mask_crop
        )

        centroid_crop = None

        centroid_full = zone.get(
            "centroid_full"
        )

        if centroid_full is not None:
            centroid_crop = (
                centroid_full[0] - x0,
                centroid_full[1] - y0
            )

        result.append(
            {
                "name":
                    zone.get(
                        "name",
                        ""
                    ),

                "mask_crop":
                    mask_crop,

                "contours_crop":
                    contours_crop,

                "centroid_crop":
                    centroid_crop
            }
        )

    return result