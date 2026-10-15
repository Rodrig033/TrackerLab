from __future__ import annotations

import csv
import os

from stats.descriptive import (
    valid_values,
    calculate_statistics
)

from stats.plots import (
    create_velocity_plot,
    create_acceleration_plot,
    create_correlation_plot
)

from stats.heatmap import (
    create_heatmap_plot,
    create_trajectory_plot
)


def create_summary_csv(
    result,
    output_folder,
    exp_name
):
    use_cm = result.kinematic.dist_total_cm is not None

    if use_cm:
        fields = [
            ("dist_cm", "distancia_cm"),
            ("dist_acum_cm", "distancia_acumulada_cm"),
            ("vel_cm_s", "velocidad_cm_s"),
            ("acel_cm_s2", "aceleracion_cm_s2"),
            ("ang_deg", "angulo_deg"),
            (
                "vel_angular_deg_s",
                "velocidad_angular_deg_s"
            )
        ]
    else:
        fields = [
            ("dist_px", "distancia_px"),
            ("dist_acum_px", "distancia_acumulada_px"),
            ("vel_px_s", "velocidad_px_s"),
            ("acel_px_s2", "aceleracion_px_s2"),
            ("ang_deg", "angulo_deg"),
            (
                "vel_angular_deg_s",
                "velocidad_angular_deg_s"
            )
        ]

    headers = [
        "variable",
        "n",
        "faltantes",
        "media",
        "desviacion_estandar",
        "minimo",
        "q1",
        "mediana",
        "q3",
        "maximo"
    ]

    rows = []
    total = len(result.records)

    for field, name in fields:
        values = valid_values(
            result.records,
            field
        )

        stats = calculate_statistics(
            values,
            total
        )

        rows.append([
            name,
            stats["n"],
            stats["faltantes"],
            stats["media"],
            stats["desviacion_estandar"],
            stats["minimo"],
            stats["q1"],
            stats["mediana"],
            stats["q3"],
            stats["maximo"]
        ])

    output_path = os.path.join(
        output_folder,
        f"eda_resumen_{exp_name}.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    return output_path


def generate_eda(
    result,
    output_folder,
    exp_name,
    context=None
):
    # context: stats.heatmap.PlotContext (opcional). Con escala, ROI, origen XY, un frame
    # de fondo y las zonas, el mapa de calor y la trayectoria salen sobre el video real.
    summary = create_summary_csv(
        result,
        output_folder,
        exp_name
    )

    velocity = create_velocity_plot(
        result,
        output_folder,
        exp_name
    )

    acceleration = create_acceleration_plot(
        result,
        output_folder,
        exp_name
    )

    correlation = create_correlation_plot(
        result,
        output_folder,
        exp_name
    )

    heatmap = create_heatmap_plot(
        result,
        output_folder,
        exp_name,
        context=context
    )

    trajectory = create_trajectory_plot(
        result,
        output_folder,
        exp_name,
        context=context
    )

    return {
        "summary": summary,
        "velocity": velocity,
        "acceleration": acceleration,
        "correlation": correlation,
        "heatmap": heatmap,
        "trajectory": trajectory
    }