from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from stats.descriptive import valid_values


def create_histogram(
    values,
    title,
    xlabel,
    output_path
):
    if len(values) == 0:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(values, bins=30)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    ax.grid(True, alpha=0.20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def create_velocity_plot(
    result,
    output_folder,
    exp_name
):
    use_cm = result.kinematic.dist_total_cm is not None

    field = (
        "vel_cm_s"
        if use_cm
        else "vel_px_s"
    )

    values = valid_values(
        result.records,
        field
    )

    output_path = os.path.join(
        output_folder,
        f"eda_velocidad_{exp_name}.png"
    )

    return create_histogram(
        values,
        "Distribución de velocidad",
        (
            "Velocidad (cm/s)"
            if use_cm
            else "Velocidad (px/s)"
        ),
        output_path
    )


def create_acceleration_plot(
    result,
    output_folder,
    exp_name
):
    use_cm = result.kinematic.dist_total_cm is not None

    field = (
        "acel_cm_s2"
        if use_cm
        else "acel_px_s2"
    )

    values = valid_values(
        result.records,
        field
    )

    output_path = os.path.join(
        output_folder,
        f"eda_aceleracion_{exp_name}.png"
    )

    return create_histogram(
        values,
        "Distribución de aceleración",
        (
            "Aceleración (cm/s²)"
            if use_cm
            else "Aceleración (px/s²)"
        ),
        output_path
    )


def create_correlation_plot(
    result,
    output_folder,
    exp_name
):
    use_cm = result.kinematic.dist_total_cm is not None

    if use_cm:
        fields = [
            "dist_cm",
            "vel_cm_s",
            "acel_cm_s2",
            "vel_angular_deg_s"
        ]
    else:
        fields = [
            "dist_px",
            "vel_px_s",
            "acel_px_s2",
            "vel_angular_deg_s"
        ]

    labels = [
        "Distancia",
        "Velocidad",
        "Aceleración",
        "Vel. angular"
    ]

    rows = []

    for record in result.records:
        row = []
        valid = True

        for field in fields:
            value = getattr(record, field, None)

            if value is None:
                valid = False
                break

            try:
                value = float(value)
            except (TypeError, ValueError):
                valid = False
                break

            if not np.isfinite(value):
                valid = False
                break

            row.append(value)

        if valid:
            rows.append(row)

    if len(rows) < 2:
        return None

    data = np.array(rows, dtype=float)
    correlation = np.corrcoef(data, rowvar=False)

    output_path = os.path.join(
        output_folder,
        f"eda_correlacion_{exp_name}.png"
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(
        correlation,
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(
        range(len(labels)),
        labels,
        rotation=35,
        ha="right"
    )

    ax.set_yticks(
        range(len(labels)),
        labels
    )

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{correlation[i, j]:.2f}",
                ha="center",
                va="center"
            )

    ax.set_title("Matriz de correlación")

    fig.colorbar(
        image,
        ax=ax,
        label="Correlación"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path