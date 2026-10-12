"""Pruebas de main.py (pipeline de tracking).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_main.py -v

Estrategia: no existe un "bucle original" aislado que copiar (en el original
está mezclado con ventanas y teclado), pero sí los oráculos literales de cada
pieza que ya se verificó (tracker, cinemática, zonas). Aquí se COMPONEN en el
mismo orden que el original y se compara registro por registro.
"""

import importlib
import os
import tempfile
import unittest

import cv2
import numpy as np

import main as pipeline
from analysis import FrameRecord
from main import (
    FrameEvent,
    Quit,
    RecaptureBackground,
    Reinit,
    TrackingJob,
    run_tracking,
)
from tracking.background_tracker import DEFAULT_PARAMS, BackgroundTracker, compute_background
from tracking.color_tracker import ColorTracker, default_start, finalize_params, initial_params_from_bbox
from tracking.silhouette_tracker import SilhouetteTracker
from analysis.spatial_metrics import ZoneGeometry
from video.frame_selection import FrameRange, compute_timing, read_background_frames
from video.reader import VideoReader


def _imp(name):
    try:
        return importlib.import_module(f"tests.{name}")
    except ImportError:
        return importlib.import_module(name)


BG = _imp("test_background_tracker")
KIN = _imp("test_kinematic_metrics")
SPA = _imp("test_spatial_metrics")
SIL = _imp("test_silhouette_tracker")
COL = _imp("test_color_tracker")


def write_video(path, frames, size):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, size)
    for f in frames:
        vw.write(f)
    vw.release()


def full_zone(name, drawer, shape):
    m = np.zeros(shape, np.uint8)
    drawer(m)
    M = cv2.moments(m)
    centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] > 1e-12 else None
    return {"name": name, "mask_full": m, "centroid_full": centroid}


def zone_tuple(r):
    return (r.name, r.inside, r.dist_center_px, r.dist_center_cm, r.dist_perim_px, r.dist_perim_cm,
            r.time_acum_s, r.frames_acum, r.entries_acum, r.lat_first_s, r.frame_first)


KIN_FIELDS = ("dist_acum_px", "dist_acum_cm", "dist_px", "dist_cm", "vel_px_s", "vel_cm_s",
              "acel_px_s2", "acel_cm_s2", "ang_deg", "vel_angular_deg_s")


class ModoFondoContraOriginalTests(unittest.TestCase):
    """Modo 1 de punta a punta: video -> tracker -> cinemática -> zonas -> registros."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "v.avi")
        frames, cls.truth = BG.make_frames(90, lost=set(range(30, 36)) | {50}, second_blob=True)
        write_video(cls.path, frames, (BG.FW, BG.FH))
        cls.mask = BG.roi_mask("elipse")
        cls.zones_d = [
            full_zone("rect", lambda m: cv2.rectangle(m, (60, 60), (160, 170), 255, -1), (BG.FH, BG.FW)),
            full_zone("circulo", lambda m: cv2.circle(m, (250, 120), 40, 255, -1), (BG.FH, BG.FW)),
        ]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def make_job(self, reader, **kw):
        bg = compute_background(read_background_frames(reader, 35), BG.ROI, self.mask, False)
        tracker = BackgroundTracker(self.mask, bg, DEFAULT_PARAMS)
        timing = compute_timing(reader.fps, True, reader.fps / 2)  # un frame de cada 2
        job = TrackingJob(
            reader=reader, frame_range=FrameRange(5, 80), timing=timing, tracker=tracker, roi_box=BG.ROI,
            zones=[ZoneGeometry(z["name"], z["mask_full"], z["centroid_full"]) for z in self.zones_d],
            cm_per_px=kw.get("cm", 0.0731), xy_origin=kw.get("origin", (50.0, 200.0)),
            start=((40.0, 90.0), (28, 78, 24, 24)),
        )
        return job, bg

    def test_registro_a_registro_identico(self):
        for cm, origin in [(0.0731, (50.0, 200.0)), (None, (50.0, 200.0)), (0.0731, None), (None, None)]:
            with VideoReader(self.path) as reader, VideoReader(self.path) as oracle_reader:
                job, bg = self.make_job(reader, cm=cm, origin=origin)
                timing, x0, y0 = job.timing, BG.ROI[0], BG.ROI[1]
                H_full = reader.height

                usados = {}  # lo que decide la "interfaz": una reinicialización con caja y una cancelación

                def on_frame(ev):
                    if ev.processed == 11 and ev.detection.found:
                        usados["bbox"] = ev.detection.bbox
                        return Reinit(bbox=ev.detection.bbox)
                    if ev.processed == 26:
                        return Reinit()  # el usuario canceló: solo se olvida el movimiento previo
                    return None

                res = run_tracking(job, on_frame=on_frame, keep_records=True)

                # ---- composición de los oráculos, en el orden del original ----
                idxs = list(range(5, 81, timing.frame_step))
                self.assertEqual(res.frames_processed, len(idxs))
                rois = [oracle_reader.read_at(i)[BG.ROI[1]:BG.ROI[1] + BG.ROI[3], BG.ROI[0]:BG.ROI[0] + BG.ROI[2]]
                        for i in idxs]
                bx, by, bw, bh = usados["bbox"]
                reinit_tracker = {11: ((bx + bw / 2.0, by + bh / 2.0), (bx, by, bw, bh))}  # antes del frame 11
                trk = BG._o_track(rois, bg, self.mask, DEFAULT_PARAMS, "white_on_black", False, False,
                                  (40.0, 90.0), (28, 78, 24, 24), reinit_tracker)
                points = [None if t[0] is None else (t[2][0] + x0, t[2][1] + y0) for t in trk]
                resets = {11, 26}
                scale = None if cm is None else {"cm_per_px": cm}
                kin_rows, kin_sum = KIN.original_kinematics(points, timing.dt, scale, resets)
                zon_rows, zon_sum = SPA.original_zones(points, self.zones_d, timing.dt, scale, 5,
                                                       timing.frame_step, timing.video_fps, resets)

                hallados = 0
                for i, rec in enumerate(res.records):
                    tag = f"cm={cm} origen={origin} i={i}"
                    self.assertEqual((rec.frame, rec.time_s), (idxs[i], (idxs[i] - 5) / timing.video_fps), msg=tag)
                    p = points[i]
                    self.assertEqual(rec.found, p is not None, msg=tag)
                    if p is None:
                        self.assertIsNone(rec.cx_abs, msg=tag)
                        self.assertIsNone(rec.cy_abs_inv, msg=tag)
                    else:
                        hallados += 1
                        self.assertEqual((rec.cx_abs, rec.cy_abs_inv), (p[0], H_full - p[1]), msg=tag)
                    # coordenadas relativas (fórmula del original, con las mismas condiciones)
                    if p is not None and origin is not None:
                        xr, yr = p[0] - origin[0], origin[1] - p[1]
                        self.assertEqual((rec.x_rel_px, rec.y_rel_px), (xr, yr), msg=tag)
                        if cm is not None:
                            self.assertEqual((rec.x_rel_cm, rec.y_rel_cm), (xr * cm, yr * cm), msg=tag)
                        else:
                            self.assertEqual((rec.x_rel_cm, rec.y_rel_cm), (None, None), msg=tag)
                    else:
                        self.assertEqual((rec.x_rel_px, rec.y_rel_px, rec.x_rel_cm, rec.y_rel_cm), (None,) * 4, msg=tag)
                    self.assertEqual(tuple(getattr(rec, f) for f in KIN_FIELDS), kin_rows[i], msg=tag)
                    self.assertEqual(tuple(zone_tuple(z) for z in rec.zones), zon_rows[i], msg=tag)

                m = res.kinematic
                self.assertEqual((m.frames_validos, m.tiempo_valido_s, m.dist_total_px, m.dist_total_cm,
                                  m.vel_media_px_s, m.vel_media_cm_s, m.acel_media_px_s2, m.acel_media_cm_s2,
                                  m.angulo_medio_deg, m.vel_angular_media_deg_s), kin_sum)
                self.assertEqual(tuple((s.name, s.tiempo_total_s, s.frames_total, s.proporcion_tiempo, s.entradas,
                                        s.lat_primera_entrada_s, s.frame_primera_entrada, s.dist_media_centro_px,
                                        s.dist_media_centro_cm, s.dist_media_perimetro_px,
                                        s.dist_media_perimetro_cm) for s in res.zones), zon_sum)
                self.assertEqual(res.frames_found, hallados)
                self.assertGreater(hallados, 25)  # la comparación no pasa en vacío
                self.assertLess(hallados, len(idxs))  # y hay frames perdidos
                self.assertTrue(any(r.entries_acum > 0 for rec in res.records for r in rec.zones))

class OtrosModosTests(unittest.TestCase):
    def test_modo_silueta_coincide_con_el_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.avi")
            frames, truth = SIL.make_frames(60, dark=True, seed=4)
            write_video(path, frames, (SIL.FW, SIL.FH))
            m = SIL.roi_mask("elipse")
            x0, y0, w0, h0 = SIL.ROI
            with VideoReader(path) as reader, VideoReader(path) as rd2:
                sample = rd2.read_at(0)[y0:y0 + h0, x0:x0 + w0]
                bbox0 = SIL.window_around(truth[0])
                tracker = SilhouetteTracker.from_bbox(m, sample, bbox0)
                job = TrackingJob(reader, FrameRange(10, 59), compute_timing(reader.fps), tracker, SIL.ROI)
                res = run_tracking(job, keep_records=True)
                idxs = list(range(10, 60))
                rois = [rd2.read_at(i)[y0:y0 + h0, x0:x0 + w0] for i in idxs]
                _, out = SIL._o_run(rois, sample, bbox0, m)
                self.assertEqual(len(res.records), len(idxs))
                for rec, (pick, *_rest) in zip(res.records, out):
                    self.assertEqual(rec.found, pick is not None)
                    if pick is not None:
                        self.assertEqual((rec.cx_abs, rec.cy_abs_inv), (pick[0] + x0, reader.height - (pick[1] + y0)))
                self.assertGreater(res.frames_found, 40)

    def test_modo_color_coincide_con_el_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.avi")
            frames, truth = COL.make_frames(60, COL.GREEN, distractor=COL.BLUE, twin=True)
            write_video(path, frames, (COL.FW, COL.FH))
            m = COL.roi_mask("elipse")
            x0, y0, w0, h0 = COL.ROI
            with VideoReader(path) as reader, VideoReader(path) as rd2:
                first = rd2.read_at(0)[y0:y0 + h0, x0:x0 + w0]
                cx, cy = truth[0][0] - x0, truth[0][1] - y0
                params = initial_params_from_bbox(first, (cx - 14, cy - 14, 28, 28), m)
                params.update({"S_MIN": max(params["S_MIN"], 120), "V_MIN": max(params["V_MIN"], 100)})
                params = finalize_params(params)
                tracker = ColorTracker(m, params)
                job = TrackingJob(reader, FrameRange(0, 59), compute_timing(reader.fps), tracker, COL.ROI)
                res = run_tracking(job, keep_records=True)
                rois = [rd2.read_at(i)[y0:y0 + h0, x0:x0 + w0] for i in range(60)]
                # arranque del original: centro del ROI, y el primer hallazgo reemplaza la posición inicial
                c0, b0 = default_start(w0, h0)
                pick0, _ = COL._o_track_color_marker(cv2.bitwise_and(rois[0], rois[0], mask=m), params, c0, m)
                if pick0 is not None:
                    c0, b0 = (pick0[0], pick0[1]), pick0[2]
                out = COL._o_run(rois, params, m, c0, b0)
                for rec, (pick, *_rest) in zip(res.records, out):
                    self.assertEqual(rec.found, pick is not None)
                    if pick is not None:
                        self.assertEqual(rec.cx_abs, pick[0] + x0)
                self.assertGreater(res.frames_found, 50)
                self.assertEqual(res.warnings, [])

class ComportamientoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "v.avi")
        frames, cls.truth = BG.make_frames(70, lost={20, 21})
        write_video(cls.path, frames, (BG.FW, BG.FH))
        cls.mask = BG.roi_mask("elipse")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def job(self, reader, **kw):
        bg = compute_background(read_background_frames(reader, 35), BG.ROI, self.mask, False)
        tracker = BackgroundTracker(self.mask, bg, DEFAULT_PARAMS)
        return TrackingJob(reader, kw.pop("rng", FrameRange(0, 60)), compute_timing(reader.fps), tracker, BG.ROI,
                           start=kw.pop("start", ((40.0, 90.0), (28, 78, 24, 24))), **kw)

    def test_sin_callbacks_y_conteos(self):
        with VideoReader(self.path) as r:
            res = run_tracking(self.job(r))
        self.assertEqual((res.frames_processed, res.stopped_by_user, res.records), (61, False, []))
        self.assertEqual(res.frames_found, 59)
        self.assertEqual(res.kinematic.frames_validos, 59)

    def test_on_record_va_antes_que_on_frame_y_recibe_todo(self):
        orden, vistos = [], []
        with VideoReader(self.path) as r:
            res = run_tracking(self.job(r), on_record=lambda rec: (orden.append("r"), vistos.append(rec.frame)),
                               on_frame=lambda ev: orden.append("f"), keep_records=True)
        self.assertEqual(orden[:4], ["r", "f", "r", "f"])
        self.assertEqual(vistos, [rec.frame for rec in res.records])
        self.assertEqual(vistos, list(range(0, 61)))

    def test_evento_del_frame(self):
        eventos = []
        with VideoReader(self.path) as r:
            run_tracking(self.job(r, rng=FrameRange(3, 8)), on_frame=eventos.append)
        ev = eventos[2]
        self.assertIsInstance(ev, FrameEvent)
        self.assertEqual((ev.frame_idx, ev.processed, ev.total), (5, 3, 6))
        x0, y0, w0, h0 = BG.ROI
        self.assertTrue(np.array_equal(ev.roi, ev.frame[y0:y0 + h0, x0:x0 + w0]))
        self.assertIsInstance(ev.record, FrameRecord)
        self.assertEqual(ev.record.frame, ev.frame_idx)
        with self.assertRaises((ValueError, cv2.error)):  # solo lectura: hay que copiarlo para dibujar
            cv2.circle(ev.frame, (10, 10), 5, (0, 0, 255), -1)
        dibujable = ev.frame.copy()
        cv2.circle(dibujable, (10, 10), 5, (0, 0, 255), -1)  # una copia sí

    def test_quit_detiene_el_tracking(self):
        with VideoReader(self.path) as r:
            res = run_tracking(self.job(r), on_frame=lambda ev: Quit() if ev.processed == 6 else None, keep_records=True)
        self.assertEqual((res.frames_processed, res.stopped_by_user, len(res.records)), (6, True, 6))

    def test_reinit_cancelado_reinicia_la_memoria_pero_no_al_tracker(self):
        with VideoReader(self.path) as r:
            job = self.job(r)
            antes = {}

            def on_frame(ev):
                if ev.processed == 10:
                    antes["pos"] = (job.tracker.last_center, job.tracker.last_bbox)
                    return Reinit()
                if ev.processed == 11:
                    antes["despues"] = (job.tracker.last_center, job.tracker.last_bbox)

            res = run_tracking(job, on_frame=on_frame, keep_records=True)
        r10, r11 = res.records[9], res.records[10]
        self.assertIsNotNone(r10.vel_px_s)
        self.assertEqual((r11.dist_px, r11.vel_px_s, r11.acel_px_s2, r11.ang_deg), (0.0, None, None, None))
        self.assertEqual(r11.dist_acum_px, r10.dist_acum_px)  # los acumulados se conservan
        self.assertNotEqual(antes["pos"], antes["despues"])  # el tracker siguió avanzando normalmente

    def test_reinit_con_datos_cambia_posicion_y_parametros(self):
        with VideoReader(self.path) as r:
            job = self.job(r)
            nuevos = {**DEFAULT_PARAMS, "DIFF": 40}
            vistos = {}

            def on_frame(ev):
                if ev.processed == 3:
                    return Reinit(bbox=(100, 90, 20, 20), params=nuevos)
                if ev.processed == 4:
                    vistos["params"] = dict(job.tracker.params)

            run_tracking(job, on_frame=on_frame)
        self.assertEqual(vistos["params"]["DIFF"], 40)

    def test_reinit_con_centro_sin_caja_es_error(self):
        with VideoReader(self.path) as r:
            with self.assertRaises(ValueError):
                run_tracking(self.job(r), on_frame=lambda ev: Reinit(center=(1.0, 1.0)))

    def test_recapturar_fondo(self):
        with VideoReader(self.path) as r, VideoReader(self.path) as r2:
            job = self.job(r)
            original = job.tracker.bg_gray.copy()
            run_tracking(job, on_frame=lambda ev: RecaptureBackground() if ev.processed == 41 else None)
            esperado = compute_background(read_background_frames(r2, 40), BG.ROI, self.mask, False)
        self.assertTrue(np.array_equal(job.tracker.bg_gray, esperado))
        self.assertFalse(np.array_equal(job.tracker.bg_gray, original))

    def test_recapturar_fondo_en_otro_modo_solo_avisa(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.avi")
            write_video(path, COL.make_frames(10, COL.GREEN)[0], (COL.FW, COL.FH))
            with VideoReader(path) as r:
                tracker = ColorTracker(COL.roi_mask(), {"H_MIN": 50, "H_MAX": 70, "S_MIN": 100, "V_MIN": 100})
                job = TrackingJob(r, FrameRange(0, 5), compute_timing(r.fps), tracker, COL.ROI)
                res = run_tracking(job, on_frame=lambda ev: RecaptureBackground() if ev.processed == 2 else None)
        self.assertTrue(any("modo de diferencia con fondo" in w for w in res.warnings))

    def test_silueta_no_localizada_avisa_y_arranca_del_modelo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.avi")
            frames, truth = SIL.make_frames(12, dark=True, seed=4)
            write_video(path, frames, (SIL.FW, SIL.FH))
            m = SIL.roi_mask("elipse")
            x0, y0, w0, h0 = SIL.ROI
            with VideoReader(path) as r:
                sample = r.read_at(0)[y0:y0 + h0, x0:x0 + w0]
                tracker = SilhouetteTracker.from_bbox(m, sample, SIL.window_around(truth[0]))
                tracker.model.area_ref = 1e7  # ningún candidato será compatible
                res = run_tracking(TrackingJob(r, FrameRange(0, 5), compute_timing(r.fps), tracker, SIL.ROI))
        self.assertTrue(any("silueta" in w for w in res.warnings))

    def test_silueta_reinit_invalido_avisa(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.avi")
            frames, truth = SIL.make_frames(12, dark=True, seed=4)
            write_video(path, frames, (SIL.FW, SIL.FH))
            m = SIL.roi_mask("elipse")
            x0, y0, w0, h0 = SIL.ROI
            with VideoReader(path) as r:
                sample = r.read_at(0)[y0:y0 + h0, x0:x0 + w0]
                tracker = SilhouetteTracker.from_bbox(m, sample, SIL.window_around(truth[0]))
                res = run_tracking(TrackingJob(r, FrameRange(0, 5), compute_timing(r.fps), tracker, SIL.ROI),
                                   on_frame=lambda ev: Reinit(bbox=(0, 0, 4, 4)) if ev.processed == 2 else None)
        self.assertTrue(any("No se pudo extraer silueta" in w for w in res.warnings))

    def test_orden_desconocida(self):
        with VideoReader(self.path) as r:
            with self.assertRaises(TypeError):
                run_tracking(self.job(r), on_frame=lambda ev: "salir")

    def test_el_error_del_csv_se_propaga(self):
        def on_record(rec):
            raise PermissionError("CSV abierto en Excel")
        with VideoReader(self.path) as r:
            with self.assertRaises(PermissionError):
                run_tracking(self.job(r), on_record=on_record)


class ValidacionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "v.avi")
        write_video(cls.path, BG.make_frames(20)[0], (BG.FW, BG.FH))
        cls.mask = BG.roi_mask("elipse")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def base(self, r, **over):
        bg = compute_background(read_background_frames(r, 10), BG.ROI, self.mask, False)
        args = dict(reader=r, frame_range=FrameRange(0, 10), timing=compute_timing(r.fps),
                    tracker=BackgroundTracker(self.mask, bg, DEFAULT_PARAMS), roi_box=BG.ROI,
                    start=((40.0, 90.0), (28, 78, 24, 24)))
        args.update(over)
        return TrackingJob(**args)

    def test_roi_fuera_del_frame(self):
        with VideoReader(self.path) as r:
            with self.assertRaisesRegex(ValueError, "se sale del frame"):
                run_tracking(self.base(r, roi_box=(200, 100, 260, 190)))
            with self.assertRaises(ValueError):
                run_tracking(self.base(r, roi_box=(0, 0, 0, 10)))

    def test_mascara_del_tracker_con_otro_tamano(self):
        with VideoReader(self.path) as r:
            with self.assertRaisesRegex(ValueError, "máscara del tracker"):
                run_tracking(self.base(r, roi_box=(10, 10, 100, 100)))

    def test_zona_con_mascara_de_recorte_en_vez_de_frame_completo(self):
        with VideoReader(self.path) as r:
            malas = [ZoneGeometry("z", np.full((BG.ROI[3], BG.ROI[2]), 255, np.uint8), (10.0, 10.0))]
            with self.assertRaisesRegex(ValueError, "frame completo"):
                run_tracking(self.base(r, zones=malas))

    def test_modo_fondo_sin_start(self):
        with VideoReader(self.path) as r:
            with self.assertRaisesRegex(ValueError, "job.start"):
                run_tracking(self.base(r, start=None))

    def test_frame_inicial_fuera_del_video(self):
        with VideoReader(self.path) as r:
            with self.assertRaisesRegex(RuntimeError, "frame inicial"):
                run_tracking(self.base(r, frame_range=FrameRange(500, 600)))


class ModuloTests(unittest.TestCase):
    def test_main_sin_integracion_termina_con_mensaje(self):
        with self.assertRaises(SystemExit) as cm:
            pipeline.main()
        self.assertIn("integración", str(cm.exception))


if __name__ == "__main__":
    unittest.main()