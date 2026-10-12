import csv
import os
from datetime import datetime

from persistence.experiment_config import (
    experiment_dir,
    ensure_dir,
    safe_experiment_name
)


DEFAULT_CSV_NAME = "track_RATA"


def _timestamp():
    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def make_output_csv(
    base_name,
    exp_name
):
    safe_base = (
        safe_experiment_name(base_name)
        if base_name
        else DEFAULT_CSV_NAME
    )

    safe_exp = safe_experiment_name(
        exp_name
    )

    output_dir = experiment_dir(
        exp_name
    )

    ensure_dir(
        output_dir
    )

    return os.path.join(
        output_dir,
        f"{safe_base}_{safe_exp}_{_timestamp()}.csv"
    )


def make_summary_csv(
    base_name,
    exp_name
):
    safe_base = (
        safe_experiment_name(base_name)
        if base_name
        else DEFAULT_CSV_NAME
    )

    safe_exp = safe_experiment_name(
        exp_name
    )

    output_dir = experiment_dir(
        exp_name
    )

    ensure_dir(
        output_dir
    )

    return os.path.join(
        output_dir,
        f"{safe_base}_{safe_exp}_{_timestamp()}_SUMMARY.csv"
    )


def safe_float_str(
    value,
    nd=6
):
    if value is None:
        return ""

    if isinstance(
        value,
        str
    ):
        return value

    try:
        return f"{float(value):.{nd}f}"

    except (
        TypeError,
        ValueError
    ):
        return str(value)


def create_csv_writer(
    path,
    headers
):
    directory = os.path.dirname(
        path
    )

    if directory:
        ensure_dir(
            directory
        )

    file = open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    )

    writer = csv.writer(
        file
    )

    writer.writerow(
        headers
    )

    return (
        file,
        writer
    )


def write_csv(
    path,
    headers,
    rows
):
    directory = os.path.dirname(
        path
    )

    if directory:
        ensure_dir(
            directory
        )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            headers
        )

        writer.writerows(
            rows
        )


def append_csv_row(
    writer,
    row
):
    writer.writerow(
        row
    )


def close_csv_file(
    file
):
    try:
        file.flush()
    except Exception:
        pass

    try:
        file.close()
    except Exception:
        pass