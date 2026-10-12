import json
import os
import platform
from datetime import datetime

import cv2


APP_NAME = "TrackerLab"
APP_VERSION = "1.0"


def get_user_data_directory():
    system = platform.system()

    if system == "Windows":
        base = os.environ.get(
            "APPDATA"
        )

        if not base:
            base = os.path.join(
                os.path.expanduser("~"),
                "AppData",
                "Roaming"
            )

        path = os.path.join(
            base,
            APP_NAME
        )

    elif system == "Darwin":
        path = os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Application Support",
            APP_NAME
        )

    else:
        base = os.environ.get(
            "XDG_DATA_HOME"
        )

        if not base:
            base = os.path.join(
                os.path.expanduser("~"),
                ".local",
                "share"
            )

        path = os.path.join(
            base,
            APP_NAME
        )

    os.makedirs(
        path,
        exist_ok=True
    )

    return path


OUTPUT_DIR = get_user_data_directory()

EXPERIMENTS_DIR = os.path.join(
    OUTPUT_DIR,
    "Experimentos"
)


def ensure_dir(path):
    os.makedirs(
        path,
        exist_ok=True
    )


def safe_experiment_name(name):
    if name is None:
        return "experimento"

    name = str(
        name
    ).strip()

    result = []

    for char in name:
        if (
            char.isalnum()
            or char in (
                "-",
                "_"
            )
        ):
            result.append(
                char
            )

        else:
            result.append(
                "_"
            )

    safe_name = "".join(
        result
    )

    while "__" in safe_name:
        safe_name = safe_name.replace(
            "__",
            "_"
        )

    safe_name = safe_name.strip(
        "_"
    )

    if not safe_name:
        safe_name = "experimento"

    return safe_name


def experiment_dir(exp_name):
    safe_name = safe_experiment_name(
        exp_name
    )

    path = os.path.join(
        EXPERIMENTS_DIR,
        safe_name
    )

    ensure_dir(
        path
    )

    return path


def experiment_json_path(exp_name):
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

        mask_path = zone_mask_path(
            exp_name,
            zone.get(
                "name",
                "zona"
            )
        )

        saved = cv2.imwrite(
            mask_path,
            mask
        )

        if saved:
            zone[
                "mask_file"
            ] = mask_path

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

    data[
        "_application"
    ] = APP_NAME

    data[
        "_version"
    ] = APP_VERSION

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


def load_experiment(exp_name):
    path = experiment_json_path(
        exp_name
    )

    if not os.path.exists(
        path
    ):
        return None

    try:
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(
                file
            )

    except (
        OSError,
        json.JSONDecodeError
    ):
        return None


def get_saved_date(config):
    if not config:
        return None

    saved_at = config.get(
        "_saved_at"
    )

    if not saved_at:
        return None

    try:
        saved_date = datetime.fromisoformat(
            saved_at
        )

        return saved_date.strftime(
            "%d/%m/%Y %H:%M"
        )

    except ValueError:
        return saved_at


def experiment_exists(exp_name):
    path = experiment_json_path(
        exp_name
    )

    return os.path.exists(
        path
    )


def delete_experiment_config(
    exp_name
):
    path = experiment_json_path(
        exp_name
    )

    if not os.path.exists(
        path
    ):
        return False

    try:
        os.remove(
            path
        )

        return True

    except OSError:
        return False