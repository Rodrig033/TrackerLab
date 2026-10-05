import json
import os
import sys

from datetime import datetime

import cv2


def app_directory():

    if getattr(
        sys,
        "frozen",
        False
    ):

        return os.path.dirname(
            sys.executable
        )

    return os.path.dirname(
        os.path.dirname(
            os.path.abspath(
                __file__
            )
        )
    )


OUTPUT_DIR = app_directory()

EXPERIMENTS_DIR = os.path.join(
    OUTPUT_DIR,
    "Experimentos"
)


def ensure_dir(path):

    os.makedirs(
        path,
        exist_ok=True
    )


def safe_experiment_name(
    name
):

    return "".join(
        char
        if (
            char.isalnum()
            or
            char in (
                "-",
                "_"
            )
        )
        else "_"
        for char in name.strip()
    )


def experiment_dir(
    exp_name
):

    safe_name = safe_experiment_name(
        exp_name
    )

    return os.path.join(
        EXPERIMENTS_DIR,
        safe_name
    )


def experiment_json_path(
    exp_name
):

    return os.path.join(
        experiment_dir(
            exp_name
        ),
        "params.json"
    )


def zone_mask_path(
    exp_name,
    zone_name
):

    safe_zone = safe_experiment_name(
        zone_name
    )

    return os.path.join(
        experiment_dir(
            exp_name
        ),
        f"zone_mask_{safe_zone}.png"
    )


def persist_zone_masks_for_payload(
    exp_name,
    zones
):

    ensure_dir(
        experiment_dir(
            exp_name
        )
    )

    for zone in zones:

        mask = zone.get(
            "mask_full"
        )

        if mask is None:
            continue

        path = zone_mask_path(
            exp_name,
            zone["name"]
        )

        cv2.imwrite(
            path,
            mask
        )

        zone[
            "mask_file"
        ] = path

    return zones


def save_experiment(
    exp_name,
    payload
):

    path = experiment_json_path(
        exp_name
    )

    ensure_dir(
        os.path.dirname(
            path
        )
    )

    data = dict(
        payload
    )

    data[
        "_saved_at"
    ] = datetime.now().isoformat(
        timespec="seconds"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Parámetros guardados: {path}"
    )


def load_experiment(
    exp_name
):

    path = experiment_json_path(
        exp_name
    )

    old_path = os.path.join(
        OUTPUT_DIR,
        "experiments",
        safe_experiment_name(
            exp_name
        ),
        "params.json"
    )

    if (
        not os.path.exists(path)
        and
        os.path.exists(old_path)
    ):

        path = old_path

    if not os.path.exists(
        path
    ):

        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )