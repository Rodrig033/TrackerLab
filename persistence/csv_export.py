import csv
import os

from datetime import datetime

from persistence.experiment_config import (
    experiment_dir,
    ensure_dir,
    safe_experiment_name
)


DEFAULT_CSV_NAME = "track_RATA"


def make_output_csv(
    base_name,
    exp_name
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    safe_exp = safe_experiment_name(
        exp_name
    )

    safe_base = (
        safe_experiment_name(
            base_name
        )
        or
        DEFAULT_CSV_NAME
    )

    output_dir = experiment_dir(
        exp_name
    )

    ensure_dir(
        output_dir
    )

    return os.path.join(
        output_dir,
        f"{safe_base}_{safe_exp}_{timestamp}.csv"
    )


def make_summary_csv(
    base_name,
    exp_name
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    safe_exp = safe_experiment_name(
        exp_name
    )

    safe_base = (
        safe_experiment_name(
            base_name
        )
        or
        DEFAULT_CSV_NAME
    )

    output_dir = experiment_dir(
        exp_name
    )

    ensure_dir(
        output_dir
    )

    return os.path.join(
        output_dir,
        f"{safe_base}_{safe_exp}_{timestamp}_SUMMARY.csv"
    )


def safe_float_str(
    value,
    decimals=6
):

    if value is None:
        return ""

    if isinstance(
        value,
        str
    ):
        return value

    return f"{value:.{decimals}f}"


def write_csv(
    path,
    headers,
    rows
):

    ensure_dir(
        os.path.dirname(
            path
        )
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


def create_csv_writer(
    path,
    headers
):

    ensure_dir(
        os.path.dirname(
            path
        )
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

    return file, writer