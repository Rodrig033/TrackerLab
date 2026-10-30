
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4
import base64
import csv
import json
import shutil
import sys
import traceback

import cv2
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE = Path(__file__).resolve().parent.parent
CORE = BASE / "backend" / "core"
FRONTEND = BASE / "frontend"
UPLOADS = BASE / "uploads"
RESULTADOS = BASE / "resultados"

UPLOADS.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)

if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


from video.reader import VideoReader
from video.frame_selection import FrameRange, TrackingTiming
from tracking.background_tracker import (
    BackgroundTracker,
    compute_background,
    foreground_binary,
    pick_component,
    shrink_mask,
)
from pipeline import TrackingJob, Quit, run_tracking

from backend.exportaciones import (
    router as exportaciones_router,
    configurar as configurar_exportaciones,
    generar as generar_exportaciones,
)


app = FastAPI(title="TrackerLab Web")
app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND)),
    name="static"
)

videos = {}
trabajos = {}

configurar_exportaciones(trabajos, RESULTADOS)
app.include_router(exportaciones_router)


class Punto(BaseModel):
    x: int
    y: int


class DatosROI(BaseModel):
    puntos: list[Punto] = Field(min_length=3)


class DatosCalibracion(BaseModel):
    animal: Punto
    diff: int = Field(18, ge=0, le=255)
    open_k: int = Field(5, ge=1)
    close_k: int = Field(9, ge=1)
    mina: int = Field(200, ge=1)
    maxa: int = Field(80000, ge=1)
    shrink: int = Field(0, ge=0)
    sol: float = Field(0.55, ge=0, le=1)


class DatosSeguimiento(BaseModel):
    inicio: int = Field(0, ge=0)
    fin: int | None = None
    salto: int = Field(1, ge=1, le=100)


def leer_json(ruta):
    if not ruta.exists():
        return None

    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def guardar_json(ruta, contenido):
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(
            contenido,
            archivo,
            ensure_ascii=False,
            indent=4
        )


def ruta_roi(video_id):
    return UPLOADS / f"{video_id}_roi.json"


def ruta_calibracion(video_id):
    return UPLOADS / f"{video_id}_calibracion.json"


def obtener_video(video_id):
    info = videos.get(video_id)

    if info is None:
        raise HTTPException(
            404,
            "Video no registrado. Cárgalo nuevamente."
        )

    ruta = Path(info["ruta"])

    if not ruta.exists():
        raise HTTPException(
            404,
            "No existe el archivo del video."
        )

    return ruta


def codificar(imagen):
    correcto, buffer = cv2.imencode(".jpg", imagen)

    if not correcto:
        raise ValueError(
            "No se pudo codificar la imagen."
        )

    return base64.b64encode(buffer).decode("utf-8")


def preparar_roi(video_id, ancho, alto):
    datos = leer_json(ruta_roi(video_id))

    if not datos:
        raise ValueError("Primero guarda la ROI.")

    puntos = np.asarray(
        datos["roi"],
        dtype=np.int32
    )

    if len(puntos) < 3:
        raise ValueError(
            "La ROI necesita al menos tres puntos."
        )

    mascara_total = np.zeros(
        (alto, ancho),
        dtype=np.uint8
    )

    cv2.fillPoly(
        mascara_total,
        [puntos],
        255
    )

    x, y, w, h = cv2.boundingRect(puntos)

    x = max(0, x)
    y = max(0, y)
    w = min(w, ancho - x)
    h = min(h, alto - y)

    if w <= 0 or h <= 0:
        raise ValueError(
            "La ROI está fuera del video."
        )

    mascara = mascara_total[
        y:y + h,
        x:x + w
    ]

    return (x, y, w, h), mascara, puntos


def obtener_calibracion(video_id):
    datos = leer_json(
        ruta_calibracion(video_id)
    )

    if not datos:
        raise ValueError(
            "Primero guarda la calibración."
        )

    return datos


def crear_fondo(
    reader,
    roi_box,
    mascara,
    inicio,
    fin
):
    cantidad = min(
        15,
        fin - inicio + 1
    )

    indices = np.linspace(
        inicio,
        fin,
        num=cantidad,
        dtype=int
    )

    frames = []

    for indice in sorted(set(indices.tolist())):
        frame = reader.read_at(int(indice))

        if frame is not None:
            frames.append(frame)

    fondo = compute_background(
        frames,
        roi_box,
        mascara,
        False
    )

    if fondo is None:
        raise ValueError(
            "No se pudo calcular el fondo."
        )

    return fondo


def redondear(valor, decimales=2):
    if valor is None:
        return None

    return round(
        float(valor),
        decimales
    )


@app.get("/")
def pagina_principal():
    html = (
        FRONTEND / "index.html"
    ).read_text(encoding="utf-8")

    script = (
        '<script src="/static/js/exports.js" '
        'defer></script>'
    )

    if "/static/js/exports.js" not in html:
        html = html.replace(
            "</head>",
            script + "\n</head>"
        )

    return HTMLResponse(html)


@app.get("/api/estado")
def estado():
    return {
        "estado": "activo",
        "aplicacion": "TrackerLab Web"
    }


@app.post("/api/videos")
def subir_video(
    archivo: UploadFile = File(...)
):
    extension = Path(
        archivo.filename or ""
    ).suffix.lower()

    if extension not in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    }:
        raise HTTPException(
            400,
            "Formato de video no permitido."
        )

    video_id = uuid4().hex
    ruta = UPLOADS / f"{video_id}{extension}"

    with open(ruta, "wb") as destino:
        shutil.copyfileobj(
            archivo.file,
            destino
        )

    try:
        with VideoReader(str(ruta)) as reader:
            informacion = {
                "id": video_id,
                "nombre": archivo.filename,
                "fps": float(reader.fps),
                "frames": int(reader.total_frames),
                "ancho": int(reader.width),
                "alto": int(reader.height)
            }

        if (
            informacion["fps"] <= 0
            or informacion["frames"] <= 0
        ):
            raise ValueError(
                "FPS o cantidad de fotogramas inválidos."
            )

    except Exception as error:
        ruta.unlink(missing_ok=True)

        raise HTTPException(
            400,
            f"No se pudo abrir el video: {error}"
        )

    videos[video_id] = {
        **informacion,
        "ruta": str(ruta)
    }

    return informacion


@app.get("/api/videos/{video_id}/frame/{numero}")
def obtener_frame(video_id: str, numero: int):
    ruta = obtener_video(video_id)

    with VideoReader(str(ruta)) as reader:
        if (
            numero < 0
            or numero >= reader.total_frames
        ):
            raise HTTPException(
                400,
                "Fotograma fuera de rango."
            )

        frame = reader.read_at(numero)

    if frame is None:
        raise HTTPException(
            404,
            "No se pudo leer el fotograma."
        )

    correcto, buffer = cv2.imencode(
        ".jpg",
        frame
    )

    if not correcto:
        raise HTTPException(
            500,
            "Error al generar el fotograma."
        )

    return Response(
        buffer.tobytes(),
        media_type="image/jpeg"
    )


@app.post("/api/videos/{video_id}/roi")
def guardar_roi(
    video_id: str,
    datos: DatosROI
):
    obtener_video(video_id)
    info = videos[video_id]

    puntos = []

    for punto in datos.puntos:
        if not (
            0 <= punto.x < info["ancho"]
            and 0 <= punto.y < info["alto"]
        ):
            raise HTTPException(
                400,
                "Hay puntos fuera del video."
            )

        puntos.append([
            punto.x,
            punto.y
        ])

    guardar_json(
        ruta_roi(video_id),
        {"roi": puntos}
    )

    return {
        "mensaje": "ROI guardada correctamente."
    }


@app.post("/api/videos/{video_id}/calibracion")
def guardar_calibracion(
    video_id: str,
    datos: DatosCalibracion
):
    obtener_video(video_id)

    if not ruta_roi(video_id).exists():
        raise HTTPException(
            400,
            "Primero guarda la ROI."
        )

    if datos.mina > datos.maxa:
        raise HTTPException(
            400,
            "MINA no puede superar MAXA."
        )

    info = videos[video_id]

    if not (
        0 <= datos.animal.x < info["ancho"]
        and 0 <= datos.animal.y < info["alto"]
    ):
        raise HTTPException(
            400,
            "El animal está fuera del video."
        )

    contenido = {
        "animal": datos.animal.model_dump(),
        "parametros": {
            "DIFF": datos.diff,
            "OPEN": datos.open_k,
            "CLOSE": datos.close_k,
            "MINA": datos.mina,
            "MAXA": datos.maxa,
            "SHRINK": datos.shrink,
            "SOL": datos.sol
        }
    }

    guardar_json(
        ruta_calibracion(video_id),
        contenido
    )

    return {
        "mensaje": "Calibración guardada correctamente."
    }


@app.get(
    "/api/videos/{video_id}/deteccion/{numero}"
)
def probar_deteccion(
    video_id: str,
    numero: int
):
    ruta = obtener_video(video_id)

    try:
        configuracion = obtener_calibracion(
            video_id
        )

        with VideoReader(str(ruta)) as reader:
            if (
                numero < 0
                or numero >= reader.total_frames
            ):
                raise ValueError(
                    "Fotograma inválido."
                )

            roi_box, mascara, puntos = preparar_roi(
                video_id,
                reader.width,
                reader.height
            )

            fondo = crear_fondo(
                reader,
                roi_box,
                mascara,
                0,
                reader.total_frames - 1
            )

            frame = reader.read_at(numero)

        if frame is None:
            raise ValueError(
                "No se pudo leer el fotograma."
            )

        x, y, w, h = roi_box
        p = configuracion["parametros"]
        animal = configuracion["animal"]

        mascara_reducida = shrink_mask(
            mascara,
            p["SHRINK"]
        )

        binaria = foreground_binary(
            frame[y:y + h, x:x + w],
            fondo,
            mascara_reducida,
            p["DIFF"],
            p["OPEN"],
            p["CLOSE"],
            use_clahe=False,
            apply_bw=False,
            polarity="white_on_black"
        )

        componente = pick_component(
            binaria,
            (
                animal["x"] - x,
                animal["y"] - y
            ),
            p["MINA"],
            p["MAXA"],
            sol_min=p["SOL"]
        )

        imagen = frame.copy()

        cv2.polylines(
            imagen,
            [puntos],
            True,
            (0, 255, 0),
            3
        )

        centro = None

        if componente is not None:
            cx, cy, caja = componente

            centro = {
                "x": redondear(cx + x),
                "y": redondear(cy + y)
            }

            cv2.circle(
                imagen,
                (
                    round(cx + x),
                    round(cy + y)
                ),
                10,
                (0, 255, 255),
                -1
            )

            bx, by, bw, bh = caja

            cv2.rectangle(
                imagen,
                (bx + x, by + y),
                (
                    bx + x + bw,
                    by + y + bh
                ),
                (255, 0, 0),
                2
            )

        return {
            "encontrado": componente is not None,
            "centro": centro,
            "imagen": codificar(imagen),
            "mascara": codificar(binaria)
        }

    except ValueError as error:
        raise HTTPException(
            400,
            str(error)
        )


def ejecutar_seguimiento(
    trabajo_id,
    video_id,
    datos
):
    trabajo = trabajos[trabajo_id]

    try:
        ruta = obtener_video(video_id)

        with VideoReader(str(ruta)) as reader:
            inicio = datos.inicio

            fin = (
                reader.total_frames - 1
                if datos.fin is None
                else datos.fin
            )

            if inicio >= reader.total_frames:
                raise ValueError(
                    "El inicio supera el total de frames."
                )

            if (
                fin < inicio
                or fin >= reader.total_frames
            ):
                raise ValueError(
                    "El fotograma final es inválido."
                )

            roi_box, mascara, _ = preparar_roi(
                video_id,
                reader.width,
                reader.height
            )

            configuracion = obtener_calibracion(
                video_id
            )

            parametros = configuracion["parametros"]
            animal = configuracion["animal"]

            fondo = crear_fondo(
                reader,
                roi_box,
                mascara,
                inicio,
                fin
            )

            tracker = BackgroundTracker(
                include_mask=mascara,
                bg_gray=fondo,
                params=parametros
            )

            x, y, w, h = roi_box

            centro_inicial = (
                float(animal["x"] - x),
                float(animal["y"] - y)
            )

            caja_inicial = (
                max(
                    0,
                    int(centro_inicial[0] - 15)
                ),
                max(
                    0,
                    int(centro_inicial[1] - 15)
                ),
                30,
                30
            )

            rango = FrameRange(
                start=inicio,
                end=fin
            )

            fps_video = float(reader.fps)
            salto = datos.salto
            fps_efectivos = fps_video / salto
            dt = salto / fps_video

            tiempo = TrackingTiming(
                video_fps=fps_video,
                requested_fps=fps_efectivos,
                frame_step=salto,
                effective_fps=fps_efectivos,
                dt=dt
            )

            job = TrackingJob(
                reader=reader,
                frame_range=rango,
                timing=tiempo,
                tracker=tracker,
                roi_box=roi_box,
                start=(
                    centro_inicial,
                    caja_inicial
                )
            )

            total = rango.count(salto)

            trabajo["total"] = total
            trabajo["estado"] = "procesando"

            ruta_csv = (
                RESULTADOS / f"{trabajo_id}.csv"
            )

            with open(
                ruta_csv,
                "w",
                newline="",
                encoding="utf-8-sig"
            ) as archivo:
                escritor = csv.writer(archivo)

                escritor.writerow([
                    "frame",
                    "tiempo_s",
                    "x_px",
                    "y_px",
                    "estado",
                    "distancia_acumulada_px",
                    "velocidad_px_s",
                    "aceleracion_px_s2",
                    "angulo_deg"
                ])

                anterior = None
                anterior_velocidad = None
                distancia_acumulada = 0.0

                def guardar_registro(registro):
                    nonlocal anterior
                    nonlocal anterior_velocidad
                    nonlocal distancia_acumulada

                    x_px = getattr(
                        registro,
                        "cx_abs",
                        None
                    )

                    y_invertida = getattr(
                        registro,
                        "cy_abs_inv",
                        None
                    )

                    y_px = (
                        None
                        if y_invertida is None
                        else reader.height - y_invertida
                    )

                    tiempo_s = float(
                        registro.time_s
                    )

                    velocidad = None
                    aceleracion = None
                    angulo = None

                    if (
                        x_px is not None
                        and y_px is not None
                    ):
                        punto = (
                            float(x_px),
                            float(y_px)
                        )

                        if anterior is not None:
                            distancia = float(
                                np.hypot(
                                    punto[0] - anterior[0],
                                    punto[1] - anterior[1]
                                )
                            )

                            distancia_acumulada += distancia
                            velocidad = distancia / dt

                            if anterior_velocidad is not None:
                                aceleracion = (
                                    velocidad
                                    - anterior_velocidad
                                ) / dt

                        anterior = punto
                        anterior_velocidad = velocidad

                        trabajo["trayectoria"].append([
                            redondear(x_px),
                            redondear(y_px)
                        ])

                    else:
                        anterior = None
                        anterior_velocidad = None

                    escritor.writerow([
                        registro.frame,
                        tiempo_s,
                        x_px,
                        y_px,
                        str(registro.status),
                        redondear(distancia_acumulada),
                        redondear(velocidad),
                        redondear(aceleracion),
                        angulo
                    ])

                    trabajo["series"].append({
                        "tiempo": redondear(
                            tiempo_s,
                            3
                        ),
                        "velocidad": redondear(
                            velocidad,
                            3
                        ),
                        "distancia": redondear(
                            distancia_acumulada,
                            3
                        )
                    })

                def actualizar_progreso(evento):
                    trabajo["procesados"] = (
                        evento.processed
                    )

                    trabajo["porcentaje"] = redondear(
                        evento.processed
                        / max(1, total)
                        * 100,
                        1
                    )

                    if trabajo["detener"].is_set():
                        return Quit()

                    return None

                resultado = run_tracking(
                    job,
                    on_frame=actualizar_progreso,
                    on_record=guardar_registro,
                    keep_records=False,
                    log=lambda mensaje: None
                )

            trabajo["procesados"] = (
                resultado.frames_processed
            )

            trabajo["detectados"] = (
                resultado.frames_found
            )

            trabajo["rechazados"] = (
                resultado.frames_rejected
            )

            trabajo["recuperados"] = (
                resultado.frames_recovered
            )

            trabajo["advertencias"] = (
                resultado.warnings
            )

            trabajo["porcentaje"] = redondear(
                resultado.frames_processed
                / max(1, total)
                * 100,
                1
            )

            resumen = resultado.kinematic
            procesados = resultado.frames_processed

            trabajo["metricas"] = {
                "tiempo_s": redondear(
                    max(0, procesados - 1) * dt
                ),
                "deteccion_pct": redondear(
                    resultado.frames_found
                    / max(1, procesados)
                    * 100
                ),
                "tiempo_valido_s": redondear(
                    resumen.tiempo_valido_s
                ),
                "distancia_px": redondear(
                    resumen.dist_total_px
                ),
                "velocidad_px_s": redondear(
                    resumen.vel_media_px_s
                ),
                "aceleracion_px_s2": redondear(
                    resumen.acel_media_px_s2
                ),
                "angulo_deg": redondear(
                    resumen.angulo_medio_deg
                ),
                "velocidad_angular_deg_s": redondear(
                    resumen.vel_angular_media_deg_s
                )
            }

            trabajo["csv"] = str(ruta_csv)

            trabajo["estado"] = (
                "detenido"
                if resultado.stopped_by_user
                else "finalizado"
            )

            if trabajo["estado"] == "finalizado":
                try:
                    generar_exportaciones(trabajo_id)
                    trabajo["exportacion_error"] = None

                except Exception as error:
                    trabajo["exportacion_error"] = str(error)
                    traceback.print_exc()

    except Exception as error:
        trabajo["estado"] = "error"
        trabajo["error"] = str(error)
        traceback.print_exc()


@app.post(
    "/api/videos/{video_id}/seguimiento"
)
def iniciar_seguimiento(
    video_id: str,
    datos: DatosSeguimiento
):
    obtener_video(video_id)

    if not ruta_roi(video_id).exists():
        raise HTTPException(
            400,
            "Primero guarda la ROI."
        )

    if not ruta_calibracion(video_id).exists():
        raise HTTPException(
            400,
            "Primero guarda la calibración."
        )

    if any(
        t["video_id"] == video_id
        and t["estado"] in {
            "preparando",
            "procesando"
        }
        for t in trabajos.values()
    ):
        raise HTTPException(
            409,
            "Ya existe un seguimiento activo para este video."
        )

    trabajo_id = uuid4().hex

    trabajos[trabajo_id] = {
        "id": trabajo_id,
        "video_id": video_id,
        "estado": "preparando",
        "procesados": 0,
        "total": 0,
        "porcentaje": 0,
        "detectados": 0,
        "rechazados": 0,
        "recuperados": 0,
        "trayectoria": [],
        "series": [],
        "advertencias": [],
        "metricas": {},
        "error": None,
        "csv": None,
        "exportacion_error": None,
        "detener": Event()
    }

    Thread(
        target=ejecutar_seguimiento,
        args=(
            trabajo_id,
            video_id,
            datos
        ),
        daemon=True
    ).start()

    return {
        "trabajo_id": trabajo_id,
        "mensaje": "Seguimiento iniciado."
    }


@app.get(
    "/api/seguimientos/{trabajo_id}"
)
def consultar_seguimiento(trabajo_id: str):
    trabajo = trabajos.get(trabajo_id)

    if trabajo is None:
        raise HTTPException(
            404,
            "Seguimiento no encontrado."
        )

    puntos = trabajo["trayectoria"]
    series = trabajo["series"]

    paso_puntos = max(
        1,
        (len(puntos) + 1499) // 1500
    )

    paso_series = max(
        1,
        (len(series) + 999) // 1000
    )

    return {
        "id": trabajo["id"],
        "estado": trabajo["estado"],
        "procesados": trabajo["procesados"],
        "total": trabajo["total"],
        "porcentaje": trabajo["porcentaje"],
        "detectados": trabajo["detectados"],
        "rechazados": trabajo["rechazados"],
        "recuperados": trabajo["recuperados"],
        "trayectoria": puntos[::paso_puntos],
        "series": series[::paso_series],
        "advertencias": trabajo["advertencias"],
        "metricas": trabajo["metricas"],
        "error": trabajo["error"],
        "exportacion_error": trabajo.get(
            "exportacion_error"
        )
    }


@app.post(
    "/api/seguimientos/{trabajo_id}/detener"
)
def detener_seguimiento(trabajo_id: str):
    trabajo = trabajos.get(trabajo_id)

    if trabajo is None:
        raise HTTPException(
            404,
            "Seguimiento no encontrado."
        )

    trabajo["detener"].set()

    return {
        "mensaje": "Solicitud de detención enviada."
    }


@app.get(
    "/api/seguimientos/{trabajo_id}/csv"
)
def descargar_csv(trabajo_id: str):
    trabajo = trabajos.get(trabajo_id)

    if trabajo is None:
        raise HTTPException(
            404,
            "Seguimiento no encontrado."
        )

    if trabajo["estado"] not in {
        "finalizado",
        "detenido"
    }:
        raise HTTPException(
            400,
            "El seguimiento todavía no ha terminado."
        )

    ruta = trabajo["csv"]

    if not ruta or not Path(ruta).exists():
        raise HTTPException(
            404,
            "No se encontró el archivo CSV."
        )

    return FileResponse(
        ruta,
        media_type="text/csv",
        filename=f"trackerlab_{trabajo_id}.csv"
    )
