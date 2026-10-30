
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import zipfile

from datetime import datetime
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


router = APIRouter(
    prefix="/api/exportaciones",
    tags=["Exportaciones"]
)

_trabajos = None
_resultados = None
_lock = Lock()


def configurar(trabajos: dict, resultados: Path):
    global _trabajos, _resultados

    _trabajos = trabajos
    _resultados = Path(resultados).resolve()
    _resultados.mkdir(parents=True, exist_ok=True)


def _numero(valor):
    if valor is None or str(valor).strip() == "":
        return None

    try:
        numero = float(valor)
        return numero if math.isfinite(numero) else None
    except (TypeError, ValueError):
        return None


def _columna(fila, *nombres):
    for nombre in nombres:
        if nombre in fila:
            numero = _numero(fila[nombre])

            if numero is not None:
                return numero

    return None


def _cargar_csv(ruta):
    with open(
        ruta,
        newline="",
        encoding="utf-8-sig"
    ) as archivo:
        return list(csv.DictReader(archivo))


def _series(filas):
    puntos = []

    for fila in filas:
        puntos.append({
            "x": _columna(
                fila, "x_px", "cx_abs", "x", "x_abs"
            ),
            "y": _columna(
                fila, "y_px", "y", "y_abs"
            ),
            "t": _columna(
                fila, "tiempo_s", "time_s", "tiempo"
            ),
            "v": _columna(
                fila, "velocidad_px_s", "vel_px_s"
            ),
            "a": _columna(
                fila, "aceleracion_px_s2", "acel_px_s2"
            ),
            "d": _columna(
                fila,
                "distancia_acumulada_px",
                "dist_acum_px"
            )
        })

    return puntos


def _grafica_linea(
    puntos,
    clave,
    titulo,
    etiqueta,
    destino
):
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    pares = [
        (p["t"], p[clave])
        for p in puntos
        if p["t"] is not None
        and p[clave] is not None
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    if pares:
        ax.plot(
            [p[0] for p in pares],
            [p[1] for p in pares],
            linewidth=1.4
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Sin datos suficientes",
            ha="center",
            va="center",
            transform=ax.transAxes
        )

    ax.set(
        title=titulo,
        xlabel="Tiempo (s)",
        ylabel=etiqueta
    )

    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def _trayectoria(puntos, destino):
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    validos = [
        (p["x"], p["y"])
        for p in puntos
        if p["x"] is not None
        and p["y"] is not None
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    if validos:
        xs, ys = zip(*validos)

        ax.plot(
            xs,
            ys,
            linewidth=1,
            alpha=0.8
        )

        ax.scatter(
            [validos[0][0]],
            [validos[0][1]],
            color="green",
            label="Inicio",
            s=40
        )

        ax.scatter(
            [validos[-1][0]],
            [validos[-1][1]],
            color="red",
            label="Final",
            s=40
        )

        ax.legend()
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim")

    else:
        ax.text(
            0.5,
            0.5,
            "Sin posiciones válidas",
            ha="center",
            transform=ax.transAxes
        )

    ax.set(
        title="Trayectoria del animal",
        xlabel="X (px)",
        ylabel="Y (px)"
    )

    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def _calor(puntos, destino):
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    validos = [
        (p["x"], p["y"])
        for p in puntos
        if p["x"] is not None
        and p["y"] is not None
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    if len(validos) >= 2:
        xs, ys = zip(*validos)

        if len(set(xs)) > 1 and len(set(ys)) > 1:
            imagen = ax.hist2d(
                xs,
                ys,
                bins=45,
                cmap="inferno"
            )

            fig.colorbar(
                imagen[3],
                ax=ax,
                label="Detecciones por celda"
            )
        else:
            ax.scatter(xs, ys, s=12)

        ax.invert_yaxis()

    else:
        ax.text(
            0.5,
            0.5,
            "Sin datos suficientes",
            ha="center",
            transform=ax.transAxes
        )

    ax.set(
        title="Mapa de calor de posiciones",
        xlabel="X (px)",
        ylabel="Y (px)"
    )

    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def _correlacion(puntos, destino):
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    claves = ["t", "v", "a", "d"]

    nombres = [
        "Tiempo",
        "Velocidad",
        "Aceleración",
        "Distancia"
    ]

    matriz = np.full((4, 4), np.nan)

    for i, clave_a in enumerate(claves):
        for j, clave_b in enumerate(claves):
            pares = [
                (p[clave_a], p[clave_b])
                for p in puntos
                if p[clave_a] is not None
                and p[clave_b] is not None
            ]

            if len(pares) < 2:
                continue

            x, y = np.array(pares).T

            if np.std(x) > 0 and np.std(y) > 0:
                matriz[i, j] = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))

    imagen = ax.imshow(
        np.ma.masked_invalid(matriz),
        vmin=-1,
        vmax=1,
        cmap="coolwarm"
    )

    ax.set_xticks(
        range(4),
        nombres,
        rotation=35,
        ha="right"
    )

    ax.set_yticks(range(4), nombres)
    ax.set_title("Correlaciones de variables numéricas")

    fig.colorbar(imagen, ax=ax)

    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def _excel(filas, metricas, destino):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    libro = Workbook()

    resumen = libro.active
    resumen.title = "Resumen"

    resumen.append(["Métrica", "Valor"])

    for clave, valor in metricas.items():
        if not isinstance(
            valor,
            (str, int, float, bool, type(None))
        ):
            valor = str(valor)

        resumen.append([clave, valor])

    datos = libro.create_sheet("Seguimiento")

    if filas:
        cabeceras = list(filas[0])
        datos.append(cabeceras)

        for fila in filas:
            valores = []

            for clave in cabeceras:
                valor = fila.get(clave)
                numero = _numero(valor)

                valores.append(
                    numero if numero is not None else valor
                )

            datos.append(valores)

    for hoja in libro:
        hoja.freeze_panes = "A2"

        for celda in hoja[1]:
            celda.font = Font(
                color="FFFFFF",
                bold=True
            )

            celda.fill = PatternFill(
                "solid",
                fgColor="17365D"
            )

        hoja.column_dimensions["A"].width = 29
        hoja.column_dimensions["B"].width = 23

    libro.save(destino)


def _pdf(metricas, imagenes, destino, nombre):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet

    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image,
        PageBreak
    )

    estilos = getSampleStyleSheet()

    historia = [
        Paragraph(
            "TrackerLab Web - Reporte del experimento",
            estilos["Title"]
        ),
        Paragraph(nombre, estilos["Heading2"]),
        Spacer(1, 14)
    ]

    filas = [["Métrica", "Valor"]]

    for clave, valor in metricas.items():
        filas.append([str(clave), str(valor)])

    tabla = Table(
        filas,
        colWidths=[220, 230],
        repeatRows=1
    )

    tabla.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#17365D")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.3,
                colors.lightgrey
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    historia.append(tabla)

    for ruta in imagenes:
        historia.append(PageBreak())

        historia.append(
            Paragraph(
                ruta.stem.replace("_", " ").title(),
                estilos["Heading2"]
            )
        )

        historia.append(Spacer(1, 12))

        historia.append(
            Image(
                str(ruta),
                width=480,
                height=310,
                kind="proportional"
            )
        )

    documento = SimpleDocTemplate(
        str(destino),
        pagesize=A4
    )

    documento.build(historia)


def _carpeta_trabajo(trabajo_id):
    if _resultados is None:
        raise HTTPException(
            503,
            "Exportaciones no configuradas."
        )

    identificador = re.sub(
        r"[^a-zA-Z0-9_-]",
        "",
        trabajo_id
    )

    return _resultados / f"experimento_{identificador}"


def generar(trabajo_id):
    if _trabajos is None:
        raise HTTPException(
            503,
            "Exportaciones no configuradas."
        )

    trabajo = _trabajos.get(trabajo_id)

    if not trabajo:
        raise HTTPException(
            404,
            "Seguimiento no encontrado."
        )

    if trabajo.get("estado") not in {
        "finalizado",
        "detenido"
    }:
        raise HTTPException(
            409,
            "El seguimiento todavía no terminó."
        )

    origen = trabajo.get("csv")

    if not origen or not Path(origen).is_file():
        raise HTTPException(
            404,
            "No se encontró el CSV del seguimiento."
        )

    carpeta = _carpeta_trabajo(trabajo_id)
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_csv = carpeta / "seguimiento.csv"

    if Path(origen).resolve() != ruta_csv.resolve():
        shutil.copy2(origen, ruta_csv)

    filas = _cargar_csv(ruta_csv)
    puntos = _series(filas)

    metricas = trabajo.get("metricas") or {}

    configuracion = {
        "trabajo_id": trabajo_id,
        "estado": trabajo.get("estado"),
        "fecha_exportacion": datetime.now().isoformat(
            timespec="seconds"
        ),
        "metricas": metricas,
        "filas_csv": len(filas)
    }

    ruta_configuracion = (
        carpeta / "configuracion_resultados.json"
    )

    ruta_configuracion.write_text(
        json.dumps(
            configuracion,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    with (
        carpeta / "resumen.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as archivo:
        escritor = csv.writer(archivo)

        escritor.writerow(["metrica", "valor"])
        escritor.writerows(metricas.items())

    imagenes = []

    ruta = carpeta / "trayectoria.png"
    _trayectoria(puntos, ruta)
    imagenes.append(ruta)

    ruta = carpeta / "mapa_calor.png"
    _calor(puntos, ruta)
    imagenes.append(ruta)

    ruta = carpeta / "velocidad.png"

    _grafica_linea(
        puntos,
        "v",
        "Velocidad",
        "px/s",
        ruta
    )

    imagenes.append(ruta)

    ruta = carpeta / "distancia.png"

    _grafica_linea(
        puntos,
        "d",
        "Distancia acumulada",
        "px",
        ruta
    )

    imagenes.append(ruta)

    ruta = carpeta / "aceleracion.png"

    _grafica_linea(
        puntos,
        "a",
        "Aceleración",
        "px/s²",
        ruta
    )

    imagenes.append(ruta)

    ruta = carpeta / "correlacion.png"
    _correlacion(puntos, ruta)
    imagenes.append(ruta)

    _excel(
        filas,
        metricas,
        carpeta / "analisis.xlsx"
    )

    _pdf(
        metricas,
        imagenes,
        carpeta / "reporte.pdf",
        carpeta.name
    )

    zip_path = carpeta / "resultados.zip"

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as paquete:
        for archivo in sorted(carpeta.iterdir()):
            if archivo.is_file() and archivo != zip_path:
                paquete.write(
                    archivo,
                    arcname=archivo.name
                )

    archivos = [
        archivo.name
        for archivo in sorted(carpeta.iterdir())
        if archivo.is_file()
    ]

    return {
        "trabajo_id": trabajo_id,
        "archivos": archivos
    }


@router.post("/{trabajo_id}/generar")
def generar_endpoint(trabajo_id: str):
    with _lock:
        try:
            return generar(trabajo_id)

        except HTTPException:
            raise

        except ImportError as error:
            raise HTTPException(
                500,
                f"Falta instalar una dependencia: {error}"
            )

        except Exception as error:
            raise HTTPException(
                500,
                f"No se pudo generar la exportación: {error}"
            )


@router.get("/{trabajo_id}")
def listar(trabajo_id: str):
    carpeta = _carpeta_trabajo(trabajo_id)

    if not carpeta.is_dir():
        return {"archivos": []}

    return {
        "archivos": [
            archivo.name
            for archivo in sorted(carpeta.iterdir())
            if archivo.is_file()
        ]
    }


@router.get("/{trabajo_id}/archivo/{nombre}")
def descargar(trabajo_id: str, nombre: str):
    carpeta = _carpeta_trabajo(trabajo_id).resolve()
    ruta = (carpeta / nombre).resolve()

    if ruta.parent != carpeta or not ruta.is_file():
        raise HTTPException(
            404,
            "Archivo no encontrado."
        )

    return FileResponse(
        str(ruta),
        filename=ruta.name
    )
