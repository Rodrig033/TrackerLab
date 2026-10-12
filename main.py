from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ui.launcher import (
    TrackerLauncher,
    show_splash_screen
)

from ui.dialogs import (
    set_tk_root,
    gui_info,
    gui_error
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

from tracking import (
    BaseTracker,
    Box,
    Detection,
    Point
)

from tracking.background_tracker import (
    BackgroundTracker,
    compute_background
)

from tracking.color_tracker import (
    ColorTracker,
    default_start
)

from tracking.silhouette_tracker import (
    SilhouetteTracker
)

from video.frame_selection import (
    FrameRange,
    TrackingTiming,
    frame_time_s,
    read_background_frames
)

from video.reader import (
    VideoReader
)


__all__ = [
    "RoiBox",
    "TrackingJob",
    "Quit",
    "Reinit",
    "RecaptureBackground",
    "Command",
    "FrameEvent",
    "TrackingResult",
    "run_tracking",
]


RoiBox = Tuple[int, int, int, int]


_SAMPLE_FIELDS = tuple(
    f.name
    for f in dataclasses.fields(
        KinematicSample
    )
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
    xy_origin: Optional[
        Tuple[
            float,
            float
        ]
    ] = None
    start: Optional[
        Tuple[
            Point,
            Box
        ]
    ] = None


@dataclass(
    frozen=True
)
class Quit:
    """Detener el tracking."""


@dataclass(
    frozen=True
)
class RecaptureBackground:
    """Recapturar el fondo."""


@dataclass(
    frozen=True
)
class Reinit:
    bbox: Optional[Box] = None
    center: Optional[Point] = None
    params: Optional[dict] = None


Command = Union[
    Quit,
    Reinit,
    RecaptureBackground
]


@dataclass(
    frozen=True,
    eq=False
)
class FrameEvent:
    frame_idx: int
    frame: np.ndarray
    roi_box: RoiBox
    detection: Detection
    record: FrameRecord
    processed: int
    total: int

    @property
    def roi(
        self
    ) -> np.ndarray:

        x0, y0, w0, h0 = (
            self.roi_box
        )

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
    zones: Tuple[
        ZoneSummary,
        ...
    ]
    timing: TrackingTiming
    frame_range: FrameRange

    warnings: List[
        str
    ] = field(
        default_factory=list
    )

    records: List[
        FrameRecord
    ] = field(
        default_factory=list
    )


def _validate(
    job: TrackingJob
) -> None:

    x0, y0, w0, h0 = (
        job.roi_box
    )

    if (
        w0 <= 0
        or h0 <= 0
        or x0 < 0
        or y0 < 0
    ):
        raise ValueError(
            f"roi_box inválido: "
            f"{job.roi_box}"
        )

    rd = job.reader

    if (
        rd.width > 0
        and rd.height > 0
    ):

        if (
            x0 + w0 > rd.width
            or y0 + h0 > rd.height
        ):
            raise ValueError(
                f"El ROI "
                f"{job.roi_box} "
                f"se sale del frame "
                f"({rd.width}x{rd.height})."
            )

        for z in job.zones:

            if (
                z.mask.shape[:2]
                != (
                    rd.height,
                    rd.width
                )
            ):
                raise ValueError(
                    f"La máscara de la zona "
                    f"{z.name!r} mide "
                    f"{z.mask.shape[:2]} "
                    f"y debe ser del frame "
                    f"completo "
                    f"{(rd.height, rd.width)}."
                )

    if (
        job.tracker.include_mask.shape[:2]
        != (
            h0,
            w0
        )
    ):
        raise ValueError(
            f"La máscara del tracker "
            f"mide "
            f"{job.tracker.include_mask.shape[:2]} "
            f"y el ROI es "
            f"{(h0, w0)}."
        )


def _start_tracker(
    job: TrackingJob,
    roi_start: np.ndarray,
    warnings: List[str]
) -> None:

    tr = job.tracker

    h0, w0 = (
        roi_start.shape[:2]
    )

    if isinstance(
        tr,
        SilhouetteTracker
    ):

        detection = tr.start(
            roi_start
        )

        if not detection.found:
            warnings.append(
                "No se pudo ubicar "
                "automáticamente la silueta "
                "en el frame inicial; "
                "puedes recalibrar con 'r'."
            )

    elif isinstance(
        tr,
        ColorTracker
    ):

        if job.start is not None:
            center, bbox = (
                job.start
            )

        else:
            center, bbox = (
                default_start(
                    w0,
                    h0
                )
            )

        tr.reset(
            center,
            bbox
        )

        detection = tr.update(
            roi_start
        )

        if not detection.found:
            warnings.append(
                "No se encontró el marcador "
                "en el frame inicial; "
                "puedes recalibrar con 'r'."
            )

    else:

        if job.start is None:
            raise ValueError(
                "Este tracker necesita "
                "job.start = "
                "(centro, caja) "
                "del sujeto."
            )

        tr.reset(
            *job.start
        )


def _apply_reinit(
    tr: BaseTracker,
    cmd: Reinit,
    roi: np.ndarray,
    warnings: List[str]
) -> None:

    if (
        cmd.center is not None
        and cmd.bbox is None
    ):
        raise ValueError(
            "Reinit con center "
            "necesita también bbox."
        )

    if isinstance(
        tr,
        SilhouetteTracker
    ):

        if (
            cmd.bbox is not None
            and not tr.reinit(
                roi,
                cmd.bbox
            )
        ):
            warnings.append(
                "No se pudo extraer "
                "silueta en la nueva ventana."
            )

        return

    if cmd.params is not None:
        tr.params = dict(
            cmd.params
        )

    if cmd.bbox is not None:

        x, y, w, h = (
            cmd.bbox
        )

        if cmd.center is not None:
            center = (
                cmd.center
            )

        else:
            center = (
                x + w / 2.0,
                y + h / 2.0
            )

        tr.reset(
            center,
            cmd.bbox
        )


def _recapture_background(
    job: TrackingJob,
    frame_idx: int,
    warnings: List[str]
) -> None:

    tr = job.tracker

    if not isinstance(
        tr,
        BackgroundTracker
    ):
        warnings.append(
            "RecaptureBackground "
            "solo aplica al modo "
            "de diferencia con fondo."
        )

        return

    frames = read_background_frames(
        job.reader,
        frame_idx
    )

    bg = compute_background(
        frames,
        job.roi_box,
        tr.include_mask,
        tr.use_clahe
    )

    if bg is None:
        warnings.append(
            "No se pudo recapturar "
            "el fondo."
        )

    else:
        tr.set_background(
            bg
        )


def run_tracking(
    job: TrackingJob,
    on_frame: Optional[
        Callable[
            [
                FrameEvent
            ],
            Optional[
                Command
            ]
        ]
    ] = None,
    on_record: Optional[
        Callable[
            [
                FrameRecord
            ],
            None
        ]
    ] = None,
    keep_records: bool = False
) -> TrackingResult:

    _validate(
        job
    )

    x0, y0, w0, h0 = (
        job.roi_box
    )

    rng = (
        job.frame_range
    )

    timing = (
        job.timing
    )

    tracker = (
        job.tracker
    )

    kin = KinematicTracker(
        dt=timing.dt,
        cm_per_px=job.cm_per_px
    )

    zt = ZoneTracker(
        job.zones,
        dt=timing.dt,
        cm_per_px=job.cm_per_px
    )

    full_height = (
        job.reader.height
    )

    cm = (
        job.cm_per_px
    )

    total = rng.count(
        timing.frame_step
    )

    warnings: List[
        str
    ] = []

    records: List[
        FrameRecord
    ] = []

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
            f"No se pudo leer "
            f"el frame inicial "
            f"({rng.start}) "
            f"del video."
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

    for (
        frame_idx,
        frame
    ) in itertools.chain(
        [
            first
        ],
        frames
    ):

        frame.flags.writeable = (
            False
        )

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

        point: Optional[
            Point
        ] = None

        if det.found:

            found_count += 1

            cx_abs = (
                det.center[
                    0
                ]
                + x0
            )

            cy_abs = (
                det.center[
                    1
                ]
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

            if (
                job.xy_origin
                is not None
            ):

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

                if cm is not None:

                    x_rel_cm = (
                        x_rel_px
                        * cm
                    )

                    y_rel_cm = (
                        y_rel_px
                        * cm
                    )

        sample = kin.update(
            point
        )

        zone_records = zt.update(
            point,
            frame_idx,
            time_s
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

        cmd: Optional[
            Command
        ] = None

        if on_frame is not None:

            event = FrameEvent(
                frame_idx,
                frame,
                job.roi_box,
                det,
                record,
                processed,
                total
            )

            cmd = on_frame(
                event
            )

        if cmd is None:
            continue

        if isinstance(
            cmd,
            Quit
        ):
            stopped = True
            break

        if isinstance(
            cmd,
            Reinit
        ):

            _apply_reinit(
                tracker,
                cmd,
                roi,
                warnings
            )

            kin.reset_history()
            zt.reset_history()

        elif isinstance(
            cmd,
            RecaptureBackground
        ):

            _recapture_background(
                job,
                frame_idx,
                warnings
            )

        else:
            raise TypeError(
                f"Orden desconocida: "
                f"{cmd!r}"
            )

    return TrackingResult(
        frames_processed=processed,
        frames_found=found_count,
        stopped_by_user=stopped,
        kinematic=kin.summary(),
        zones=zt.summary(),
        timing=timing,
        frame_range=rng,
        warnings=warnings,
        records=records
    )


def main() -> None:

    try:
        show_splash_screen()

        launcher = TrackerLauncher()

        app_options = (
            launcher.run()
        )

        if app_options is None:
            return

        root = app_options.get(
            "root"
        )

        if root is not None:
            set_tk_root(
                root
            )

        video_path = app_options.get(
            "video_path"
        )

        if not video_path:

            gui_error(
                "TrackerLab",
                "No se seleccionó "
                "ningún video."
            )

            return

        exp_name = app_options.get(
            "exp_name",
            "experimento"
        )

        tracking_mode = str(
            app_options.get(
                "tracking_mode",
                "1"
            )
        )

        print()
        print(
            "TrackerLab"
        )

        print(
            "------------------------------"
        )

        print(
            "Video:",
            video_path
        )

        print(
            "Experimento:",
            exp_name
        )

        print(
            "Modo:",
            tracking_mode
        )

        print(
            "------------------------------"
        )

        print()

        gui_info(
            "TrackerLab",
            (
                "Configuración inicial "
                "cargada correctamente.\n\n"
                f"Experimento: "
                f"{exp_name}\n"
                f"Video: "
                f"{video_path}\n\n"
                "TrackerLab está listo "
                "para continuar con "
                "la calibración."
            )
        )

    except Exception as error:

        print(
            "Error al iniciar TrackerLab:",
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


if __name__ == "__main__":
    main()