from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from ui.launcher import (
    TrackerLauncher,
    show_splash_screen
)

from ui.dialogs import (
    set_tk_root,
    gui_info,
    gui_error,
    gui_ask_float,
    ask_yes_no
)

from ui.calibration_views import (
    select_polygon_mask
)

from calibration.zones import (
    select_zones,
    mask_from_pts,
    centroid_from_mask
)

from calibration.spatial_scale import (
    select_scale_line,
    compute_scale_from_points
)

from calibration.coordinate_system import (
    select_xy_origin
)

from analysis import FrameRecord

from analysis.kinematic_metrics import (
    KinematicSample,
    KinematicSummary,
    KinematicTracker
)

from analysis.spatial_metrics import (
    ZoneGeometry,
    ZoneSummary,
    ZoneTracker
)

from analysis.trajectory import (
    TrajectoryTracker
)

from tracking import (
    BaseTracker,
    Box,
    Detection,
    Point
)

from tracking.background_tracker import (
    BackgroundTracker,
    compute_background,
    DEFAULT_PARAMS as BACKGROUND_DEFAULT_PARAMS
)

from tracking.color_tracker import (
    ColorTracker,
    default_start,
    initial_params_from_bbox,
    finalize_params
)

from tracking.silhouette_tracker import (
    SilhouetteTracker
)

from video.frame_selection import (
    FrameRange,
    TrackingTiming,
    compute_timing,
    frame_time_s,
    read_background_frames
)

from video.reader import (
    VideoReader
)

from persistence.csv_export import (
    make_output_csv,
    make_summary_csv,
    write_csv,
    safe_float_str
)

RoiBox = Tuple[int, int, int, int]


_SAMPLE_FIELDS = tuple(
    f.name
    for f in dataclasses.fields(KinematicSample)
)


@dataclass
class TrackingJob:
    reader: VideoReader
    frame_range: FrameRange
    timing: TrackingTiming
    tracker: BaseTracker
    roi_box: RoiBox
    zones: Sequence[ZoneGeometry] = ()
    cm_per_px: Optional[float] = None
    xy_origin: Optional[Tuple[float, float]] = None
    start: Optional[Tuple[Point, Box]] = None


@dataclass(frozen=True)
class Quit:
    pass


@dataclass(frozen=True)
class RecaptureBackground:
    pass


@dataclass(frozen=True)
class Reinit:
    bbox: Optional[Box] = None
    center: Optional[Point] = None
    params: Optional[dict] = None


Command = Union[
    Quit,
    Reinit,
    RecaptureBackground
]


@dataclass(frozen=True, eq=False)
class FrameEvent:
    frame_idx: int
    frame: np.ndarray
    roi_box: RoiBox
    detection: Detection
    record: FrameRecord
    processed: int
    total: int

    @property
    def roi(self):
        x0, y0, w0, h0 = self.roi_box

        return self.frame[
            y0:y0 + h0,
            x0:x0 + w0
        ]


@dataclass
class TrackingResult:
    frames_processed: int
    frames_found: int
    stopped_by_user: bool
    kinematic: KinematicSummary
    zones: Tuple[ZoneSummary, ...]
    timing: TrackingTiming
    frame_range: FrameRange
    warnings: List[str] = field(default_factory=list)
    records: List[FrameRecord] = field(default_factory=list)


def clamp_box(
    x,
    y,
    w,
    h,
    width,
    height
):
    x = max(
        0,
        min(
            int(x),
            width - 1
        )
    )

    y = max(
        0,
        min(
            int(y),
            height - 1
        )
    )

    w = max(
        1,
        min(
            int(w),
            width - x
        )
    )

    h = max(
        1,
        min(
            int(h),
            height - y
        )
    )

    return (
        x,
        y,
        w,
        h
    )


def choose_frame(
    reader,
    title,
    initial=0
):
    total = max(
        1,
        reader.total_frames
    )

    current = max(
        0,
        min(
            int(initial),
            total - 1
        )
    )

    window = title

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window,
        1200,
        800
    )

    cv2.createTrackbar(
        "frame",
        window,
        current,
        total - 1,
        lambda value: None
    )

    last_value = -1
    frame = None

    while True:

        value = cv2.getTrackbarPos(
            "frame",
            window
        )

        if value != last_value:

            frame = reader.read_at(
                value
            )

            last_value = value

        if frame is None:
            continue

        show = frame.copy()

        cv2.putText(
            show,
            f"Frame: {value}/{total - 1}",
            (
                20,
                35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                0,
                255,
                0
            ),
            2
        )

        cv2.putText(
            show,
            "ENTER = aceptar | ESC = cancelar",
            (
                20,
                70
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                0,
                255,
                0
            ),
            2
        )

        cv2.imshow(
            window,
            show
        )

        key = cv2.waitKey(
            30
        ) & 0xFF

        if key in (
            13,
            10
        ):
            result = frame.copy()

            cv2.destroyWindow(
                window
            )

            return (
                result,
                value
            )

        if key == 27:

            cv2.destroyWindow(
                window
            )

            return (
                None,
                None
            )


def select_object_bbox(
    image,
    title="Seleccionar sujeto"
):
    cv2.namedWindow(
        title,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        title,
        1200,
        800
    )

    bbox = cv2.selectROI(
        title,
        image,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow(
        title
    )

    x, y, w, h = [
        int(v)
        for v in bbox
    ]

    if (
        w <= 2
        or h <= 2
    ):
        return None

    return (
        x,
        y,
        w,
        h
    )


def _validate(
    job
):
    x0, y0, w0, h0 = (
        job.roi_box
    )

    if (
        w0 <= 0
        or h0 <= 0
    ):
        raise ValueError(
            "ROI inválido."
        )

    if (
        job.tracker.include_mask.shape[:2]
        != (
            h0,
            w0
        )
    ):
        raise ValueError(
            "La máscara del tracker "
            "no coincide con el ROI."
        )


def _start_tracker(
    job,
    roi_start,
    warnings
):
    tr = job.tracker

    h0, w0 = (
        roi_start.shape[:2]
    )

    if isinstance(
        tr,
        SilhouetteTracker
    ):

        det = tr.start(
            roi_start
        )

        if not det.found:

            warnings.append(
                "No se encontró la silueta "
                "en el frame inicial."
            )

    elif isinstance(
        tr,
        ColorTracker
    ):

        center, bbox = (
            job.start
            if job.start is not None
            else default_start(
                w0,
                h0
            )
        )

        tr.reset(
            center,
            bbox
        )

        tr.update(
            roi_start
        )

    else:

        if job.start is None:
            raise ValueError(
                "El tracker de fondo "
                "necesita una posición inicial."
            )

        tr.reset(
            *job.start
        )


def run_tracking(
    job,
    on_frame=None,
    on_record=None,
    keep_records=False
):
    _validate(
        job
    )

    x0, y0, w0, h0 = (
        job.roi_box
    )

    rng = job.frame_range
    timing = job.timing
    tracker = job.tracker

    kin = KinematicTracker(
        dt=timing.dt,
        cm_per_px=job.cm_per_px
    )

    zone_tracker = ZoneTracker(
        job.zones,
        dt=timing.dt,
        cm_per_px=job.cm_per_px
    )

    full_height = (
        job.reader.height
    )

    total = rng.count(
        timing.frame_step
    )

    warnings = []
    records = []

    processed = 0
    found_count = 0
    stopped = False

    frames = job.reader.iter_frames(
        rng.start,
        rng.end,
        timing.frame_step
    )

    first = next(
        frames,
        None
    )

    if first is None:
        raise RuntimeError(
            "No se pudo leer "
            "el frame inicial."
        )

    first[
        1
    ].flags.writeable = False

    roi_start = first[
        1
    ][
        y0:y0 + h0,
        x0:x0 + w0
    ]

    _start_tracker(
        job,
        roi_start,
        warnings
    )

    for frame_idx, frame in itertools.chain(
        [
            first
        ],
        frames
    ):

        frame.flags.writeable = False

        roi = frame[
            y0:y0 + h0,
            x0:x0 + w0
        ]

        time_s = frame_time_s(
            frame_idx,
            rng.start,
            timing.video_fps
        )

        det = tracker.update(
            roi
        )

        cx_abs = None
        cy_abs = None
        cy_abs_inv = None

        x_rel_px = None
        y_rel_px = None

        x_rel_cm = None
        y_rel_cm = None

        point = None

        if det.found:

            found_count += 1

            cx_abs = (
                det.center[0]
                + x0
            )

            cy_abs = (
                det.center[1]
                + y0
            )

            point = (
                cx_abs,
                cy_abs
            )

            cy_abs_inv = (
                full_height
                - cy_abs
            )

            if job.xy_origin is not None:

                ox, oy = (
                    job.xy_origin
                )

                x_rel_px = (
                    cx_abs
                    - ox
                )

                y_rel_px = (
                    oy
                    - cy_abs
                )

                if job.cm_per_px is not None:

                    x_rel_cm = (
                        x_rel_px
                        * job.cm_per_px
                    )

                    y_rel_cm = (
                        y_rel_px
                        * job.cm_per_px
                    )

        sample = kin.update(
            point
        )

        zone_records = (
            zone_tracker.update(
                point,
                frame_idx,
                time_s
            )
        )

        record = FrameRecord(
            frame=frame_idx,
            time_s=time_s,
            cx_abs=cx_abs,
            cy_abs_inv=cy_abs_inv,
            x_rel_px=x_rel_px,
            y_rel_px=y_rel_px,
            x_rel_cm=x_rel_cm,
            y_rel_cm=y_rel_cm,
            zones=zone_records,
            **{
                name: getattr(
                    sample,
                    name
                )
                for name
                in _SAMPLE_FIELDS
            }
        )

        processed += 1

        if keep_records:
            records.append(
                record
            )

        if on_record is not None:
            on_record(
                record
            )

        if on_frame is not None:

            cmd = on_frame(
                FrameEvent(
                    frame_idx,
                    frame,
                    job.roi_box,
                    det,
                    record,
                    processed,
                    total
                )
            )

            if isinstance(
                cmd,
                Quit
            ):
                stopped = True
                break

    return TrackingResult(
        frames_processed=processed,
        frames_found=found_count,
        stopped_by_user=stopped,
        kinematic=kin.summary(),
        zones=zone_tracker.summary(),
        timing=timing,
        frame_range=rng,
        warnings=warnings,
        records=records
    )


def build_zones(
    zones_data
):
    result = []

    for zone in zones_data:

        mask = zone.get(
            "mask_full"
        )

        if mask is None:
            continue

        result.append(
            ZoneGeometry(
                zone.get(
                    "name",
                    "zona"
                ),
                mask,
                zone.get(
                    "centroid_full"
                )
            )
        )

    return result

def export_tracking_csv(
    result,
    exp_name,
    base_name="track_RATA"
):
    output_csv = make_output_csv(
        base_name,
        exp_name
    )

    summary_csv = make_summary_csv(
        base_name,
        exp_name
    )

    zone_names = [
        z.name
        for z in result.zones
    ]

    headers = [
        "frame",
        "time_s",
        "cx_abs",
        "cy_abs_inv",
        "x_rel_px",
        "y_rel_px",
        "x_rel_cm",
        "y_rel_cm",
    ]

    for zone_name in zone_names:
        headers.extend([
            f"en_{zone_name}",
            f"dist_centro_{zone_name}_px",
            f"dist_centro_{zone_name}_cm",
            f"dist_perimetro_{zone_name}_px",
            f"dist_perimetro_{zone_name}_cm",
            f"tiempo_acum_{zone_name}_s",
            f"frames_acum_{zone_name}",
            f"entradas_acum_{zone_name}",
            f"lat_primera_entrada_{zone_name}_s",
            f"frame_primera_entrada_{zone_name}",
        ])

    headers.extend([
        "dist_px",
        "dist_cm",
        "dist_acum_px",
        "dist_acum_cm",
        "vel_px_s",
        "vel_cm_s",
        "acel_px_s2",
        "acel_cm_s2",
        "ang_deg",
        "vel_angular_deg_s",
    ])

    rows = []

    for record in result.records:

        row = [
            record.frame,
            safe_float_str(record.time_s, 6),
            safe_float_str(record.cx_abs, 3),
            safe_float_str(record.cy_abs_inv, 3),
            safe_float_str(record.x_rel_px, 3),
            safe_float_str(record.y_rel_px, 3),
            safe_float_str(record.x_rel_cm, 4),
            safe_float_str(record.y_rel_cm, 4),
        ]

        zone_map = {
            zone.name: zone
            for zone in record.zones
        }

        for zone_name in zone_names:

            zone = zone_map.get(
                zone_name
            )

            if zone is None:

                row.extend([
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ])

                continue

            row.extend([
                zone.inside,
                safe_float_str(
                    zone.dist_center_px,
                    4
                ),
                safe_float_str(
                    zone.dist_center_cm,
                    4
                ),
                safe_float_str(
                    zone.dist_perim_px,
                    4
                ),
                safe_float_str(
                    zone.dist_perim_cm,
                    4
                ),
                safe_float_str(
                    zone.time_acum_s,
                    4
                ),
                zone.frames_acum,
                zone.entries_acum,
                safe_float_str(
                    zone.lat_first_s,
                    4
                ),
                (
                    zone.frame_first
                    if zone.frame_first is not None
                    else ""
                ),
            ])

        row.extend([
            safe_float_str(
                record.dist_px,
                4
            ),
            safe_float_str(
                record.dist_cm,
                4
            ),
            safe_float_str(
                record.dist_acum_px,
                4
            ),
            safe_float_str(
                record.dist_acum_cm,
                4
            ),
            safe_float_str(
                record.vel_px_s,
                4
            ),
            safe_float_str(
                record.vel_cm_s,
                4
            ),
            safe_float_str(
                record.acel_px_s2,
                4
            ),
            safe_float_str(
                record.acel_cm_s2,
                4
            ),
            safe_float_str(
                record.ang_deg,
                4
            ),
            safe_float_str(
                record.vel_angular_deg_s,
                4
            ),
        ])

        rows.append(
            row
        )

    write_csv(
        output_csv,
        headers,
        rows
    )

    summary_headers = [
        "metrica",
        "valor"
    ]

    summary_rows = [
        [
            "experimento",
            exp_name
        ],
        [
            "frames_procesados",
            result.frames_processed
        ],
        [
            "frames_detectados",
            result.frames_found
        ],
        [
            "frame_inicio",
            result.frame_range.start
        ],
        [
            "frame_fin",
            result.frame_range.end
        ],
        [
            "fps_video_original",
            safe_float_str(
                result.timing.video_fps,
                6
            )
        ],
        [
            "fps_tracking_solicitado",
            safe_float_str(
                result.timing.requested_fps,
                6
            )
        ],
        [
            "fps_tracking_efectivo",
            safe_float_str(
                result.timing.effective_fps,
                6
            )
        ],
        [
            "tracking_frame_step",
            result.timing.frame_step
        ],
        [
            "dt_s",
            safe_float_str(
                result.timing.dt,
                6
            )
        ],
        [
            "frames_validos",
            result.kinematic.frames_validos
        ],
        [
            "tiempo_valido_s",
            safe_float_str(
                result.kinematic.tiempo_valido_s,
                4
            )
        ],
        [
            "dist_total_px",
            safe_float_str(
                result.kinematic.dist_total_px,
                4
            )
        ],
        [
            "dist_total_cm",
            safe_float_str(
                result.kinematic.dist_total_cm,
                4
            )
        ],
        [
            "vel_media_px_s",
            safe_float_str(
                result.kinematic.vel_media_px_s,
                4
            )
        ],
        [
            "vel_media_cm_s",
            safe_float_str(
                result.kinematic.vel_media_cm_s,
                4
            )
        ],
        [
            "acel_media_px_s2",
            safe_float_str(
                result.kinematic.acel_media_px_s2,
                4
            )
        ],
        [
            "acel_media_cm_s2",
            safe_float_str(
                result.kinematic.acel_media_cm_s2,
                4
            )
        ],
        [
            "angulo_medio_deg",
            safe_float_str(
                result.kinematic.angulo_medio_deg,
                4
            )
        ],
        [
            "vel_angular_media_deg_s",
            safe_float_str(
                result.kinematic.vel_angular_media_deg_s,
                4
            )
        ],
    ]

    for zone in result.zones:

        summary_rows.extend([
            [
                f"{zone.name}_tiempo_total_s",
                safe_float_str(
                    zone.tiempo_total_s,
                    4
                )
            ],
            [
                f"{zone.name}_frames_total",
                zone.frames_total
            ],
            [
                f"{zone.name}_proporcion_tiempo",
                safe_float_str(
                    zone.proporcion_tiempo,
                    6
                )
            ],
            [
                f"{zone.name}_entradas",
                zone.entradas
            ],
            [
                f"{zone.name}_lat_primera_entrada_s",
                safe_float_str(
                    zone.lat_primera_entrada_s,
                    4
                )
            ],
            [
                f"{zone.name}_frame_primera_entrada",
                (
                    zone.frame_primera_entrada
                    if zone.frame_primera_entrada is not None
                    else ""
                )
            ],
            [
                f"{zone.name}_dist_media_centro_px",
                safe_float_str(
                    zone.dist_media_centro_px,
                    4
                )
            ],
            [
                f"{zone.name}_dist_media_centro_cm",
                safe_float_str(
                    zone.dist_media_centro_cm,
                    4
                )
            ],
            [
                f"{zone.name}_dist_media_perimetro_px",
                safe_float_str(
                    zone.dist_media_perimetro_px,
                    4
                )
            ],
            [
                f"{zone.name}_dist_media_perimetro_cm",
                safe_float_str(
                    zone.dist_media_perimetro_cm,
                    4
                )
            ],
        ])

    write_csv(
        summary_csv,
        summary_headers,
        summary_rows
    )

    return (
        output_csv,
        summary_csv
    )



def main():
    reader = None

    try:
        show_splash_screen()

        launcher = (
            TrackerLauncher()
        )

        options = (
            launcher.run()
        )

        if options is None:
            return

        set_tk_root(
            options[
                "root"
            ]
        )

        video_path = (
            options[
                "video_path"
            ]
        )

        reader = VideoReader(
            video_path
        )

        print(
            "Video cargado correctamente"
        )

        print(
            "FPS:",
            reader.fps
        )

        print(
            "Frames:",
            reader.total_frames
        )

        start_frame, start_idx = (
            choose_frame(
                reader,
                "FRAME INICIO",
                0
            )
        )

        if start_frame is None:
            return

        end_frame, end_idx = (
            choose_frame(
                reader,
                "FRAME FINAL",
                max(
                    0,
                    reader.total_frames - 1
                )
            )
        )

        if end_frame is None:
            return

        frame_range = (
            FrameRange.ordered(
                start_idx,
                end_idx
            )
        )

        sample_frame, sample_idx = (
            choose_frame(
                reader,
                "FRAME MUESTRA",
                frame_range.start
            )
        )

        if sample_frame is None:
            return

        print(
            "Selecciona el ROI general."
        )

        (
            roi_mask,
            roi_points
        ) = select_polygon_mask(
            sample_frame,
            "ROI GENERAL"
        )

        if roi_mask is None:
            return

        roi_points = np.array(
            roi_points,
            dtype=np.int32
        )

        x, y, w, h = cv2.boundingRect(
            roi_points
        )

        x, y, w, h = clamp_box(
            x,
            y,
            w,
            h,
            reader.width,
            reader.height
        )

        roi_box = (
            x,
            y,
            w,
            h
        )

        include_mask = roi_mask[
            y:y + h,
            x:x + w
        ].copy()

        sample_roi = sample_frame[
            y:y + h,
            x:x + w
        ].copy()

        bbox = select_object_bbox(
            sample_roi,
            "SELECCIONA EL SUJETO"
        )

        if bbox is None:
            return

        bx, by, bw, bh = bbox

        start_center = (
            bx + bw / 2.0,
            by + bh / 2.0
        )

        tracking_mode = str(
            options.get(
                "tracking_mode",
                "1"
            )
        )

        tracker = None

        if tracking_mode == "1":

            bg = compute_background(
                read_background_frames(
                    reader,
                    sample_idx
                ),
                roi_box,
                include_mask,
                bool(
                    options.get(
                        "use_clahe",
                        False
                    )
                )
            )

            if bg is None:
                raise RuntimeError(
                    "No se pudo calcular "
                    "el fondo."
                )

            tracker = BackgroundTracker(
                include_mask,
                bg,
                dict(
                    BACKGROUND_DEFAULT_PARAMS
                ),
                polarity=options.get(
                    "polarity",
                    "white_on_black"
                ),
                apply_bw=bool(
                    options.get(
                        "apply_bw",
                        False
                    )
                ),
                use_clahe=bool(
                    options.get(
                        "use_clahe",
                        False
                    )
                )
            )

        elif tracking_mode == "2":

            tracker = (
                SilhouetteTracker.from_bbox(
                    include_mask,
                    sample_roi,
                    bbox
                )
            )

            if tracker is None:
                raise RuntimeError(
                    "No se pudo crear "
                    "el modelo de silueta."
                )

        elif tracking_mode == "3":

            params = initial_params_from_bbox(
                sample_roi,
                bbox,
                include_mask
            )

            if params is None:
                raise RuntimeError(
                    "No se pudo calibrar "
                    "el color."
                )

            params = finalize_params(
                params
            )

            tracker = ColorTracker(
                include_mask,
                params
            )

        else:
            raise ValueError(
                f"Modo desconocido: "
                f"{tracking_mode}"
            )

        zones_data = select_zones(
            sample_frame,
            roi_mask_full=roi_mask,
            mutually_exclusive=bool(
                options.get(
                    "mutually_exclusive_zones",
                    True
                )
            )
        )

        zones = build_zones(
            zones_data
        )

        scale_info = None

        if ask_yes_no(
            "¿Deseas definir una escala real?",
            default_yes=True
        ):

            scale_points = (
                select_scale_line(
                    sample_frame
                )
            )

            if scale_points is not None:

                real_cm = gui_ask_float(
                    "Escala",
                    "¿Cuántos cm existen entre los dos puntos?",
                    default=None
                )

                if real_cm is not None:

                    scale_info = (
                        compute_scale_from_points(
                            scale_points[0],
                            scale_points[1],
                            real_cm
                        )
                    )

        xy_origin_info = None

        if ask_yes_no(
            "¿Deseas definir un origen XY?",
            default_yes=False
        ):

            xy_origin_info = (
                select_xy_origin(
                    sample_frame
                )
            )

        cm_per_px = None

        if scale_info is not None:
            cm_per_px = scale_info.get(
                "cm_per_px"
            )

        xy_origin = None

        if xy_origin_info is not None:
            xy_origin = tuple(
                xy_origin_info[
                    "origin_px"
                ]
            )

        timing = compute_timing(
            reader.fps,
            custom_fps=bool(
                options.get(
                    "custom_tracking_fps",
                    False
                )
            ),
            requested_fps=options.get(
                "tracking_fps"
            )
        )

        trajectory = (
            TrajectoryTracker()
        )

        def on_record(
            record
        ):
            if record.cx_abs is None:
                return

            if record.cy_abs_inv is None:
                return

            real_y = (
                reader.height
                - record.cy_abs_inv
            )

            trajectory.add_point(
                (
                    record.cx_abs,
                    real_y
                )
            )

        def on_frame(
            event
        ):
            display = (
                event.roi.copy()
            )

            x0, y0, _, _ = (
                event.roi_box
            )

            display = trajectory.draw(
                display,
                offset=(
                    x0,
                    y0
                )
            )

            det = (
                event.detection
            )

            if det.found:

                bx, by, bw, bh = (
                    det.bbox
                )

                cv2.rectangle(
                    display,
                    (
                        bx,
                        by
                    ),
                    (
                        bx + bw,
                        by + bh
                    ),
                    (
                        0,
                        255,
                        0
                    ),
                    2
                )

                cx, cy = (
                    det.center
                )

                cv2.circle(
                    display,
                    (
                        int(cx),
                        int(cy)
                    ),
                    5,
                    (
                        0,
                        255,
                        255
                    ),
                    -1
                )

            else:

                cv2.putText(
                    display,
                    "LOST",
                    (
                        15,
                        40
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (
                        0,
                        0,
                        255
                    ),
                    2
                )

            cv2.putText(
                display,
                (
                    f"Frame "
                    f"{event.frame_idx} "
                    f"| "
                    f"{event.processed}/"
                    f"{event.total}"
                ),
                (
                    15,
                    30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    255,
                    255,
                    255
                ),
                2
            )

            cv2.putText(
                display,
                "q = salir",
                (
                    15,
                    display.shape[0] - 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (
                    255,
                    255,
                    255
                ),
                2
            )

            cv2.imshow(
                "TRACKERLAB - TRACKING",
                display
            )

            key = cv2.waitKey(
                1
            ) & 0xFF

            if key == ord(
                "q"
            ):
                return Quit()

            return None

        job = TrackingJob(
            reader=reader,
            frame_range=frame_range,
            timing=timing,
            tracker=tracker,
            roi_box=roi_box,
            zones=zones,
            cm_per_px=cm_per_px,
            xy_origin=xy_origin,
            start=(
                start_center,
                bbox
            )
        )

        result = run_tracking(
            job,
            on_frame=on_frame,
            on_record=on_record,
            keep_records=True
        )

        cv2.destroyAllWindows()

        exp_name = options.get(
            "exp_name",
            "experimento"
        )

        csv_base_name = options.get(
            "csv_base_name",
            "track_RATA"
        )

        output_csv, summary_csv = export_tracking_csv(
            result,
            exp_name,
            csv_base_name
        )

        print()
        print("CSV generado:")
        print(output_csv)

        print()
        print("Resumen generado:")
        print(summary_csv)

        print()
        print(
            "Tracking terminado."
        )

        print(
            "Frames procesados:",
            result.frames_processed
        )

        print(
            "Frames detectados:",
            result.frames_found
        )

        print(
            "Puntos de trayectoria:",
            len(
                trajectory
            )
        )

        print(
            "Distancia total:",
            result.kinematic.dist_total_px,
            "px"
        )

        if (
            result.kinematic.dist_total_cm
            is not None
        ):

            print(
                "Distancia total:",
                result.kinematic.dist_total_cm,
                "cm"
            )

        gui_info(
        "Tracking finalizado",
        (
            "Tracking terminado correctamente.\n\n"
            f"Frames procesados: {result.frames_processed}\n"
            f"Frames detectados: {result.frames_found}\n"
            f"Puntos de trayectoria: {len(trajectory)}\n\n"
            f"CSV:\n{output_csv}\n\n"
            f"Resumen:\n{summary_csv}"
            )
        )

    except Exception as error:

        cv2.destroyAllWindows()

        print(
            "ERROR:",
            error
        )

        try:
            gui_error(
                "Error de TrackerLab",
                str(
                    error
                )
            )

        except Exception:
            pass

    finally:

        if reader is not None:

            try:
                reader.close()

            except Exception:
                pass


if __name__ == "__main__":
    main()