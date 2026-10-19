import importlib
import math
import os
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from analysis import FrameRecord
from analysis.quality import assess_quality
from analysis.spatial_metrics import ZoneGeometry
from pipeline import (
    FrameEvent,
    Quit,
    RecaptureBackground,
    Reinit,
    TrackingJob,
    run_tracking,
)
from tracking import BaseTracker, Detection
from tracking.reacquisition import Reacquirer, SilhouetteReacquirer
from tracking.background_tracker import DEFAULT_PARAMS, BackgroundTracker, compute_background
from tracking.color_tracker import ColorTracker, default_start, finalize_params, initial_params_from_bbox
from tracking.silhouette_tracker import SilhouetteTracker
from video.frame_selection import FrameRange, compute_timing, read_background_frames
from video.reader import VideoReader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

RAW = dict(smoothing_alpha=None, jump_max_px=None)  # sin filtrar: posiciones medidas


def write_video(path, frames, size):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, size)
    for f in frames:
        vw.write(f)
    vw.release()


def full_zone(name, drawer, shape):
    m = np.zeros(shape, np.uint8)
    drawer(m)
    M = cv2.moments(m)
    return {"name": name, "mask_full": m, "centroid_full": (M["m10"] / M["m00"], M["m01"] / M["m00"])}


def zone_tuple(r):
    return (r.name, r.inside, r.dist_center_px, r.dist_center_cm, r.dist_perim_px, r.dist_perim_cm,
            r.time_acum_s, r.frames_acum, r.entries_acum, r.lat_first_s, r.frame_first)


KIN_FIELDS = ("dist_acum_px", "dist_acum_cm", "dist_px", "dist_cm", "vel_px_s", "vel_cm_s",
              "acel_px_s2", "acel_cm_s2", "ang_deg", "vel_angular_deg_s")


# ----------------------------------------------------------------------
# Dobles para probar la lógica de filtrado con posiciones conocidas
# ----------------------------------------------------------------------
class FakeReader:
    fps = 30.0
    total_frames = 100000

    def __init__(self, w=4000, h=200):
        self.width, self.height = w, h

    def iter_frames(self, start, end, step):
        for i in range(start, end + 1, step):
            yield i, np.zeros((self.height, self.width, 3), np.uint8)


class Scripted(BaseTracker):
    """Devuelve las posiciones x indicadas (None = no ve al sujeto); y fija en 100."""

    def __init__(self, xs, w=4000, h=200):
        super().__init__(np.full((h, w), 255, np.uint8))
        self.xs, self.n = list(xs), 0

    def _detect(self, roi):
        x = self.xs[self.n]
        self.n += 1
        return Detection() if x is None else Detection(center=(float(x), 100.0), bbox=(int(x) - 5, 95, 10, 10))


def run_script(xs, w=4000, **kw):
    reader = FakeReader(w=w)
    job = TrackingJob(reader, FrameRange(0, len(xs) - 1), compute_timing(30.0), Scripted(xs, w=w), (0, 0, w, reader.height),
                      start=((100.0, 100.0), (95, 95, 10, 10)), **kw)
    return run_tracking(job, keep_records=True, log=None)


def statuses(res):
    return [r.status for r in res.records]

class ParidadSinFiltroTests(unittest.TestCase):
    """Con el filtrado desactivado, el pipeline reproduce al original registro por registro."""

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

    def make_job(self, reader, cm, origin):
        bg = compute_background(read_background_frames(reader, 35), BG.ROI, self.mask, False)
        job = TrackingJob(
            reader=reader, frame_range=FrameRange(5, 80), timing=compute_timing(reader.fps, True, reader.fps / 2),
            tracker=BackgroundTracker(self.mask, bg, DEFAULT_PARAMS), roi_box=BG.ROI,
            zones=[ZoneGeometry(z["name"], z["mask_full"], z["centroid_full"]) for z in self.zones_d],
            cm_per_px=cm, xy_origin=origin, start=((40.0, 90.0), (28, 78, 24, 24)), **RAW)
        return job, bg

    def test_registro_a_registro_identico(self):
        for cm, origin in [(0.0731, (50.0, 200.0)), (None, (50.0, 200.0)), (0.0731, None), (None, None)]:
            with VideoReader(self.path) as reader, VideoReader(self.path) as oracle_reader:
                job, bg = self.make_job(reader, cm, origin)
                timing, x0, y0 = job.timing, BG.ROI[0], BG.ROI[1]
                usados = {}

                def on_frame(ev):
                    if ev.processed == 11 and ev.detection.found:
                        usados["bbox"] = ev.detection.bbox
                        return Reinit(bbox=ev.detection.bbox)
                    if ev.processed == 26:
                        return Reinit()
                    return None

                res = run_tracking(job, on_frame=on_frame, keep_records=True, log=None)
                idxs = list(range(5, 81, timing.frame_step))
                self.assertEqual(res.frames_processed, len(idxs))
                rois = [oracle_reader.read_at(i)[y0:y0 + BG.ROI[3], x0:x0 + BG.ROI[2]] for i in idxs]
                bx, by, bw, bh = usados["bbox"]
                trk = BG._o_track(rois, bg, self.mask, DEFAULT_PARAMS, "white_on_black", False, False, (40.0, 90.0),
                                  (28, 78, 24, 24), {11: ((bx + bw / 2.0, by + bh / 2.0), (bx, by, bw, bh))})
                points = [None if t[0] is None else (t[2][0] + x0, t[2][1] + y0) for t in trk]
                scale = None if cm is None else {"cm_per_px": cm}
                kin_rows, kin_sum = KIN.original_kinematics(points, timing.dt, scale, {11, 26})
                zon_rows, zon_sum = SPA.original_zones(points, self.zones_d, timing.dt, scale, 5, timing.frame_step,
                                                       timing.video_fps, {11, 26})
                hallados = 0
                for i, rec in enumerate(res.records):
                    tag = f"cm={cm} origen={origin} i={i}"
                    p = points[i]
                    self.assertEqual(rec.found, p is not None, msg=tag)
                    if p is None:
                        self.assertEqual((rec.cx_abs, rec.cy_abs_inv, rec.status), (None, None, "perdido"), msg=tag)
                    else:
                        hallados += 1
                        self.assertEqual((rec.cx_abs, rec.cy_abs_inv), (p[0], reader.height - p[1]), msg=tag)
                        # sin filtro, la posición medida y la registrada son la misma
                        self.assertEqual((rec.cx_raw, rec.cy_raw_inv), (rec.cx_abs, rec.cy_abs_inv), msg=tag)
                        self.assertEqual(rec.status, "detectado", msg=tag)
                        if origin is not None:
                            xr, yr = p[0] - origin[0], origin[1] - p[1]
                            self.assertEqual((rec.x_rel_px, rec.y_rel_px), (xr, yr), msg=tag)
                    self.assertEqual(tuple(getattr(rec, f) for f in KIN_FIELDS), kin_rows[i], msg=tag)
                    self.assertEqual(tuple(zone_tuple(z) for z in rec.zones), zon_rows[i], msg=tag)
                m = res.kinematic
                self.assertEqual((m.frames_validos, m.tiempo_valido_s, m.dist_total_px, m.dist_total_cm, m.vel_media_px_s,
                                  m.vel_media_cm_s, m.acel_media_px_s2, m.acel_media_cm_s2, m.angulo_medio_deg,
                                  m.vel_angular_media_deg_s), kin_sum)
                self.assertEqual(res.frames_found, hallados)
                self.assertGreater(hallados, 25)
                self.assertLess(hallados, len(idxs))
                self.assertEqual((res.frames_rejected, res.frames_recovered), (0, 0))

    def test_modo_silueta(self):
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
                res = run_tracking(TrackingJob(reader, FrameRange(10, 59), compute_timing(reader.fps), tracker, SIL.ROI, **RAW),
                                   keep_records=True, log=None)
                rois = [rd2.read_at(i)[y0:y0 + h0, x0:x0 + w0] for i in range(10, 60)]
                _, out = SIL._o_run(rois, sample, bbox0, m)
                for rec, (pick, *_r) in zip(res.records, out):
                    self.assertEqual(rec.found, pick is not None)
                    if pick is not None:
                        self.assertEqual((rec.cx_abs, rec.cy_abs_inv), (pick[0] + x0, reader.height - (pick[1] + y0)))
                self.assertGreater(res.frames_found, 40)

    def test_modo_color(self):
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
                res = run_tracking(TrackingJob(reader, FrameRange(0, 59), compute_timing(reader.fps), ColorTracker(m, params),
                                               COL.ROI, **RAW), keep_records=True, log=None)
                rois = [rd2.read_at(i)[y0:y0 + h0, x0:x0 + w0] for i in range(60)]
                c0, b0 = default_start(w0, h0)
                pick0, _ = COL._o_track_color_marker(cv2.bitwise_and(rois[0], rois[0], mask=m), params, c0, m)
                if pick0 is not None:
                    c0, b0 = (pick0[0], pick0[1]), pick0[2]
                out = COL._o_run(rois, params, m, c0, b0)
                for rec, (pick, *_r) in zip(res.records, out):
                    self.assertEqual(rec.found, pick is not None)
                    if pick is not None:
                        self.assertEqual(rec.cx_abs, pick[0] + x0)
                self.assertGreater(res.frames_found, 50)


# ======================================================================
class SuavizadoTests(unittest.TestCase):
    def test_formula_exponencial_a_mano(self):
        res = run_script([100, 120, 140], jump_max_px=None, smoothing_alpha=0.5)
        self.assertEqual([r.cx_abs for r in res.records], [100.0, 110.0, 125.0])  # 100; .5*120+.5*100; .5*140+.5*110
        self.assertEqual([r.cx_raw for r in res.records], [100.0, 120.0, 140.0])  # la posición medida se conserva

    def test_por_defecto_alpha_035(self):
        res = run_script([100, 110])
        self.assertAlmostEqual(res.records[1].cx_abs, 0.35 * 110 + 0.65 * 100)

    def test_la_cinematica_usa_la_posicion_suavizada(self):
        res = run_script([100, 120, 140, 160, 180], jump_max_px=None, smoothing_alpha=0.5)
        a, b = res.records[3], res.records[4]
        self.assertAlmostEqual(b.dist_px, math.dist((b.cx_abs, b.cy_abs_inv), (a.cx_abs, a.cy_abs_inv)))
        self.assertLess(b.dist_px, 20.0)  # el suavizado acorta el paso medido (20 px)

    def test_sin_suavizado_las_posiciones_son_las_medidas(self):
        res = run_script([100, 120, 140], jump_max_px=None, smoothing_alpha=None)
        self.assertEqual([r.cx_abs for r in res.records], [100.0, 120.0, 140.0])

    def test_hueco_corto_no_reinicia_el_suavizado(self):
        res = run_script([100, 110, None, None, 130])  # 2 frames perdidos < recovery_after (5)
        prev = 0.35 * 110 + 0.65 * 100
        self.assertAlmostEqual(res.records[4].cx_abs, 0.35 * 130 + 0.65 * prev)
        self.assertEqual(res.records[4].status, "detectado")

    def test_hueco_largo_reinicia_el_suavizado(self):
        res = run_script([100, 110] + [None] * 5 + [130])
        r = res.records[-1]
        self.assertEqual((r.cx_abs, r.status), (130.0, "recuperado"))  # sin arrastre de la posición anterior

    def test_posicion_medida_en_frames_sin_detector_es_none(self):
        res = run_script([100, None, 110])
        self.assertEqual((res.records[1].cx_raw, res.records[1].cy_raw_inv, res.records[1].status), (None, None, "perdido"))

    def test_cy_raw_sigue_la_misma_convencion_que_cy_abs_inv(self):
        res = run_script([100, 110], jump_max_px=None, smoothing_alpha=None)
        self.assertEqual(res.records[0].cy_raw_inv, 200 - 100)  # alto del video menos y


class SaltosYRecuperacionTests(unittest.TestCase):
    def test_un_salto_se_rechaza_y_el_frame_queda_sin_dato(self):
        res = run_script([100, 110, 500, 120])
        self.assertEqual(statuses(res), ["detectado", "detectado", "rechazado", "detectado"])
        r = res.records[2]
        self.assertEqual((r.cx_abs, r.cy_abs_inv, r.cx_raw), (None, None, 500.0))  # se ve dónde lo vio el tracker
        self.assertFalse(r.found)
        self.assertEqual(res.frames_rejected, 1)
        self.assertEqual(res.frames_found, 3)

    def test_sin_rechazo_de_saltos_se_acepta_todo(self):
        res = run_script([100, 110, 500, 120], jump_max_px=None, smoothing_alpha=None)
        self.assertEqual(statuses(res), ["detectado"] * 4)
        self.assertEqual(res.frames_rejected, 0)

    def test_recuperacion_cercana_tras_5_frames_sin_dato(self):
        res = run_script([100, 110] + [None] * 5 + [310])  # a 200 px: más que 120, menos que 250
        self.assertEqual(res.records[-1].status, "recuperado")
        self.assertEqual(res.frames_recovered, 1)
        self.assertTrue(res.records[-1].found)

    def test_mas_alla_de_250_px_no_se_recupera_sin_adopcion(self):
        res = run_script([100, 110] + [None] * 5 + [500], adopt_after_s=None)
        self.assertEqual(res.records[-1].status, "rechazado")

    def test_defecto_original_sin_adopcion_el_sujeto_se_pierde_para_siempre(self):
        """El caso atascado de la versión anterior (adopt_after_s=None lo reproduce)."""
        xs = [100 + 10 * i if i < 30 else None if i < 70 else 100 + 10 * i for i in range(190)]
        res = run_script(xs, adopt_after_s=None)
        self.assertEqual(sum(r.found for r in res.records[70:]), 0)
        self.assertEqual(res.frames_rejected, 120)

    def test_adopcion_tras_detecciones_consistentes(self):
        xs = [100, 110] + [800 + 5 * i for i in range(70)]  # lejos, pero estable
        res = run_script(xs)  # adopt_after_s=2.0 -> 60 frames a 30 fps
        st = statuses(res)
        self.assertEqual(st[:2], ["detectado"] * 2)
        self.assertEqual(st[2:61], ["rechazado"] * 59)
        self.assertEqual(st[61], "recuperado")
        self.assertEqual(st[62:], ["detectado"] * 10)
        adoptada = res.records[61]
        self.assertEqual(adoptada.cx_abs, adoptada.cx_raw)  # el suavizado se reinició
        self.assertTrue(any("adoptada" in w and "mismo sujeto" in w for w in res.warnings))
        self.assertEqual((res.frames_found, res.frames_rejected, res.frames_recovered), (13, 59, 1))

    def test_la_adopcion_depende_del_tiempo_no_de_los_frames(self):
        xs = [100, 110] + [800 + 5 * i for i in range(30)]
        self.assertEqual(sum(r.status == "recuperado" for r in run_script(xs, adopt_after_s=1.0).records), 1)  # 30 frames
        self.assertEqual(sum(r.status == "recuperado" for r in run_script(xs, adopt_after_s=2.0).records), 0)

    def test_detecciones_inconsistentes_no_se_adoptan(self):
        xs = [100, 110] + [800 if i % 2 == 0 else 1600 for i in range(80)]  # saltando entre dos lugares lejanos
        res = run_script(xs)
        self.assertEqual(res.frames_recovered, 0)
        self.assertEqual(sum(r.found for r in res.records[2:]), 0)

    def test_perder_al_sujeto_reinicia_la_cuenta_de_consistencia(self):
        xs = [100, 110] + [800 + 5 * i for i in range(40)] + [None] + [900 + 5 * i for i in range(40)]
        res = run_script(xs)  # 40 + 40 frames consistentes, pero interrumpidos: ninguno llega a 60
        self.assertEqual(res.frames_recovered, 0)

    def test_velocidad_maxima_en_lugar_de_pixeles(self):
        xs = [100 + 130 * i for i in range(25)]  # 130 px por frame procesado, sin pérdidas reales
        por_defecto = run_script(xs)
        self.assertEqual(por_defecto.frames_found, 1)  # umbral de 120 px: todo se rechaza salvo el primero
        self.assertTrue(any("umbral" in w for w in por_defecto.warnings))
        ajustado = run_script(xs, jump_max_speed_px_s=5000.0)  # 5000 px/s * (1/30 s) = 166.7 px por frame
        self.assertEqual((ajustado.frames_found, ajustado.frames_rejected), (25, 0))
        self.assertFalse(any("umbral" in w for w in ajustado.warnings))

    def test_el_umbral_por_velocidad_no_depende_de_frame_step(self):
        """A 10 fps efectivos (frame_step 3) el mismo movimiento real recorre 3 veces más por frame procesado."""
        reader = FakeReader()
        xs = [100 + 300 * i for i in range(12)]  # 300 px por frame procesado = 3000 px/s a 10 fps
        for kw, esperado in ((dict(jump_max_px=120.0), 1), (dict(jump_max_speed_px_s=5000.0), 12)):
            job = TrackingJob(reader, FrameRange(0, 35), compute_timing(30.0, True, 10.0), Scripted(xs), (0, 0, 4000, 200),
                              start=((100.0, 100.0), (95, 95, 10, 10)), **kw)
            self.assertEqual(job.timing.frame_step, 3)
            self.assertEqual(run_tracking(job, log=None).frames_found, esperado, msg=str(kw))

    def test_el_umbral_por_velocidad_rechaza_lo_que_la_excede(self):
        """umbral = velocidad * dt: lo que lo supera se rechaza, y dt cambia con frame_step."""
        # dt = 1/30 s  ->  5000 px/s equivalen a 166.7 px por frame procesado
        res = run_script([100, 230, 630], jump_max_speed_px_s=5000.0)
        self.assertEqual(statuses(res), ["detectado", "detectado", "rechazado"])  # pasos de 130 (sí) y 400 px (no)
        # frame_step = 3 -> dt = 0.1 s  ->  500 px por frame procesado
        reader = FakeReader()
        xs = [100, 230, 630, 1230]  # pasos de 130, 400 y 600 px
        job = TrackingJob(reader, FrameRange(0, 11), compute_timing(30.0, True, 10.0), Scripted(xs), (0, 0, 4000, 200),
                          start=((100.0, 100.0), (95, 95, 10, 10)), jump_max_speed_px_s=5000.0)
        self.assertAlmostEqual(job.timing.dt, 0.1)
        self.assertEqual(statuses(run_tracking(job, keep_records=True, log=None)),
                         ["detectado", "detectado", "detectado", "rechazado"])  # 400 px caben; 600 no

    def test_aviso_solo_si_hay_suficientes_detecciones_y_muchos_rechazos(self):
        pocos = run_script([100, 900, 1700, 2500])  # 3 rechazos de 4, pero solo 4 detecciones
        self.assertFalse(any("umbral" in w for w in pocos.warnings))
        pocos_rechazos = run_script([100 + 5 * i for i in range(30)] + [3000])
        self.assertFalse(any("umbral" in w for w in pocos_rechazos.warnings))

    def test_los_contadores_coinciden_con_los_registros(self):
        xs = [100, 110, 500, 120, None, 130] + [800 + 5 * i for i in range(70)]
        res = run_script(xs)
        st = statuses(res)
        self.assertEqual(res.frames_rejected, st.count("rechazado"))
        self.assertEqual(res.frames_recovered, st.count("recuperado"))
        self.assertEqual(res.frames_found, sum(r.found for r in res.records))
        self.assertEqual(res.frames_found, st.count("detectado") + st.count("recuperado"))

    def test_el_informe_de_calidad_distingue_rechazados_y_recuperados(self):
        xs = [100, 110] + [800 + 5 * i for i in range(70)]
        res = run_script(xs)
        rep = assess_quality(res.records, res.timing.dt)
        self.assertEqual((rep.frames_rejected, rep.frames_recovered), (res.frames_rejected, res.frames_recovered))
        self.assertGreater(rep.rejected_pct, 5.0)
        self.assertEqual(rep.verdict, "deficiente")  # 59 de 72 frames sin dato (82 %)
        self.assertTrue(any("filtro de saltos" in r for r in rep.reasons))
        self.assertTrue(any("recuperada" in r for r in rep.reasons))

    def test_pocos_rechazos_aislados_piden_revisar_no_descartan_la_corrida(self):
        xs = [100 + i for i in range(100)]
        for i in (20, 40, 60, 80, 90, 95):  # seis saltos aislados que el filtro descarta
            xs[i] = 1500
        res = run_script(xs)
        self.assertEqual(res.frames_rejected, 6)
        rep = assess_quality(res.records, res.timing.dt)
        self.assertEqual((rep.frames_rejected, rep.verdict), (6, "revisar"))
        self.assertTrue(any("filtro de saltos" in r for r in rep.reasons))

    def test_parametros_invalidos(self):
        for kw in (dict(smoothing_alpha=0), dict(smoothing_alpha=1.5), dict(jump_max_px=0), dict(jump_max_speed_px_s=-1),
                   dict(adopt_after_s=0)):
            with self.assertRaises(ValueError, msg=str(kw)):
                run_script([100, 110], **kw)


class OrdenesYValidacionTests(unittest.TestCase):
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
        return TrackingJob(reader, kw.pop("rng", FrameRange(0, 60)), compute_timing(reader.fps), BackgroundTracker(self.mask, bg, DEFAULT_PARAMS),
                           BG.ROI, start=kw.pop("start", ((40.0, 90.0), (28, 78, 24, 24))), **kw)

    def test_con_los_valores_por_defecto_el_seguimiento_funciona_de_punta_a_punta(self):
        with VideoReader(self.path) as r:
            res = run_tracking(self.job(r), keep_records=True, log=None)
        self.assertEqual(res.frames_processed, 61)
        self.assertGreater(res.frames_found, 50)
        self.assertEqual(res.frames_rejected, 0)  # un seguimiento normal no pierde datos por el filtro

    def test_on_record_va_antes_que_on_frame(self):
        orden = []
        with VideoReader(self.path) as r:
            run_tracking(self.job(r), on_record=lambda rec: orden.append("r"), on_frame=lambda ev: orden.append("f"), log=None)
        self.assertEqual(orden[:4], ["r", "f", "r", "f"])

    def test_el_frame_del_evento_es_de_solo_lectura(self):
        eventos = []
        with VideoReader(self.path) as r:
            run_tracking(self.job(r, rng=FrameRange(3, 6)), on_frame=eventos.append, log=None)
        ev = eventos[1]
        self.assertIsInstance(ev, FrameEvent)
        with self.assertRaises((ValueError, cv2.error)):
            cv2.circle(ev.frame, (10, 10), 5, (0, 0, 255), -1)

    def test_quit(self):
        with VideoReader(self.path) as r:
            res = run_tracking(self.job(r), on_frame=lambda ev: Quit() if ev.processed == 6 else None, keep_records=True, log=None)
        self.assertEqual((res.frames_processed, res.stopped_by_user), (6, True))

    def test_reinit_cancelado_olvida_el_movimiento_y_reinicia_los_filtros(self):
        with VideoReader(self.path) as r:
            job = self.job(r)
            res = run_tracking(job, on_frame=lambda ev: Reinit() if ev.processed == 10 else None, keep_records=True, log=None)
        r10, r11 = res.records[9], res.records[10]
        self.assertIsNotNone(r10.vel_px_s)
        self.assertEqual((r11.dist_px, r11.vel_px_s, r11.acel_px_s2, r11.ang_deg), (0.0, None, None, None))
        self.assertEqual(r11.dist_acum_px, r10.dist_acum_px)
        # el suavizado también se reinició: la primera posición tras el reinicio es la medida
        self.assertEqual(r11.cx_abs, r11.cx_raw)

    def test_reinit_con_datos_cambia_parametros_y_posicion(self):
        with VideoReader(self.path) as r:
            job = self.job(r)
            vistos = {}

            def on_frame(ev):
                if ev.processed == 3:
                    return Reinit(bbox=(100, 90, 20, 20), params={**DEFAULT_PARAMS, "DIFF": 40})
                if ev.processed == 4:
                    vistos["params"] = dict(job.tracker.params)

            run_tracking(job, on_frame=on_frame, log=None)
        self.assertEqual(vistos["params"]["DIFF"], 40)

    def test_reinit_con_centro_sin_caja_es_error(self):
        with VideoReader(self.path) as r:
            with self.assertRaises(ValueError):
                run_tracking(self.job(r), on_frame=lambda ev: Reinit(center=(1.0, 1.0)), log=None)

    def test_recapturar_fondo(self):
        with VideoReader(self.path) as r, VideoReader(self.path) as r2:
            job = self.job(r)
            run_tracking(job, on_frame=lambda ev: RecaptureBackground() if ev.processed == 41 else None, log=None)
            esperado = compute_background(read_background_frames(r2, 40), BG.ROI, self.mask, False)
        self.assertTrue(np.array_equal(job.tracker.bg_gray, esperado))

    def test_recapturar_fondo_en_otro_modo_solo_avisa(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.avi")
            write_video(path, COL.make_frames(10, COL.GREEN)[0], (COL.FW, COL.FH))
            with VideoReader(path) as r:
                tr = ColorTracker(COL.roi_mask(), {"H_MIN": 50, "H_MAX": 70, "S_MIN": 100, "V_MIN": 100})
                res = run_tracking(TrackingJob(r, FrameRange(0, 5), compute_timing(r.fps), tr, COL.ROI),
                                   on_frame=lambda ev: RecaptureBackground() if ev.processed == 2 else None, log=None)
        self.assertTrue(any("modo de diferencia con fondo" in w for w in res.warnings))

    def test_silueta_reinit_invalido_avisa(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.avi")
            frames, truth = SIL.make_frames(12, dark=True, seed=4)
            write_video(path, frames, (SIL.FW, SIL.FH))
            x0, y0, w0, h0 = SIL.ROI
            with VideoReader(path) as r:
                sample = r.read_at(0)[y0:y0 + h0, x0:x0 + w0]
                tr = SilhouetteTracker.from_bbox(SIL.roi_mask("elipse"), sample, SIL.window_around(truth[0]))
                res = run_tracking(TrackingJob(r, FrameRange(0, 5), compute_timing(r.fps), tr, SIL.ROI),
                                   on_frame=lambda ev: Reinit(bbox=(0, 0, 4, 4)) if ev.processed == 2 else None, log=None)
        self.assertTrue(any("No se pudo extraer silueta" in w for w in res.warnings))

    def test_orden_desconocida(self):
        with VideoReader(self.path) as r:
            with self.assertRaises(TypeError):
                run_tracking(self.job(r), on_frame=lambda ev: "salir", log=None)

    def test_el_error_del_csv_se_propaga(self):
        def on_record(rec):
            raise PermissionError("CSV abierto en Excel")
        with VideoReader(self.path) as r:
            with self.assertRaises(PermissionError):
                run_tracking(self.job(r), on_record=on_record, log=None)

    def test_validaciones(self):
        with VideoReader(self.path) as r:
            j = self.job(r)
            j.roi_box = (200, 100, 260, 190)  # el video mide 320 x 240
            with self.assertRaisesRegex(ValueError, "se sale del frame"):
                run_tracking(j, log=None)
            j = self.job(r)
            j.roi_box = (10, 10, 100, 100)  # la máscara del tracker es de otro tamaño
            with self.assertRaisesRegex(ValueError, "máscara del tracker"):
                run_tracking(j, log=None)
            malas = [ZoneGeometry("z", np.full((BG.ROI[3], BG.ROI[2]), 255, np.uint8), (10.0, 10.0))]
            with self.assertRaisesRegex(ValueError, "frame completo"):
                run_tracking(self.job(r, zones=malas), log=None)
            with self.assertRaisesRegex(ValueError, "posición inicial"):
                run_tracking(self.job(r, start=None), log=None)
            with self.assertRaisesRegex(RuntimeError, "frame inicial"):
                run_tracking(self.job(r, rng=FrameRange(500, 600)), log=None)

    def test_el_mensaje_de_recuperacion_va_a_log(self):
        mensajes = []
        reader = FakeReader()
        job = TrackingJob(reader, FrameRange(0, 7), compute_timing(30.0), Scripted([100, 110] + [None] * 5 + [310]), (0, 0, 4000, 200),
                          start=((100.0, 100.0), (95, 95, 10, 10)))
        run_tracking(job, log=mensajes.append)
        self.assertEqual(mensajes, ["Tracking recuperado en frame 7"])


class FakeReacquirer(Reacquirer):
    """Devuelve lo indicado en cada llamada (una lista de (x, y) o None); registra cómo se le llamó."""

    def __init__(self, respuestas):
        self.respuestas, self.llamadas = list(respuestas), []

    def find(self, roi_bgr, last_center, last_bbox):
        self.llamadas.append((last_center, last_bbox))
        r = self.respuestas[min(len(self.llamadas) - 1, len(self.respuestas) - 1)]
        if isinstance(r, Exception):
            raise r
        return None if r is None else Detection(center=(float(r[0]), float(r[1])), bbox=(int(r[0]) - 5, int(r[1]) - 5, 10, 10))


class BusquedaDelSujetoTests(unittest.TestCase):
    def test_apagada_por_defecto(self):
        res = run_script([100, 110] + [None] * 10)
        self.assertEqual((res.frames_reacquired, res.frames_recovered), (0, 0))

    def test_se_llama_tras_n_frames_sin_posicion_y_se_repite_cada_k(self):
        fr = FakeReacquirer([None])
        res = run_script([100, 110] + [None] * 20, reacquirer=fr)  # huecos de 1 a 20 frames
        self.assertEqual(len(fr.llamadas), 4)  # con 3 y cada 5: huecos de 3, 8, 13 y 18 frames
        self.assertEqual(res.frames_reacquired, 0)
        self.assertEqual(statuses(res)[2:], ["perdido"] * 20)

    def test_parametros_de_frecuencia(self):
        fr = FakeReacquirer([None])
        run_script([100, 110] + [None] * 10, reacquirer=fr, reacquire_after_frames=2, reacquire_every=1)
        self.assertEqual(len(fr.llamadas), 9)  # huecos de 2 a 10
        with self.assertRaises(ValueError):
            run_script([100], reacquirer=fr, reacquire_after_frames=0)
        with self.assertRaises(ValueError):
            run_script([100], reacquirer=fr, reacquire_every=0)

    def test_no_se_llama_mientras_el_seguimiento_funciona(self):
        fr = FakeReacquirer([(500, 100)])
        run_script([100 + i for i in range(30)], reacquirer=fr)
        self.assertEqual(fr.llamadas, [])

    def test_lo_que_encuentra_se_acepta_y_se_marca_como_recuperado(self):
        fr = FakeReacquirer([None, (300, 100)])  # la 1.ª búsqueda falla; la 2.ª encuentra al sujeto en x=300
        xs = [100, 110] + [None] * 10 + [305]
        res = run_script(xs, reacquirer=fr)
        st = statuses(res)
        self.assertEqual(st[:2], ["detectado"] * 2)
        self.assertEqual(st[2:9], ["perdido"] * 7)  # huecos de 1 a 7 frames; la 1.ª búsqueda (hueco de 3) no encontró nada
        self.assertEqual(st[9], "recuperado")  # hueco de 8 frames: 2.ª búsqueda
        self.assertEqual(st[10:12], ["perdido"] * 2)  # el hueco vuelve a contar desde cero tras recuperar
        self.assertEqual(st[12], "detectado")  # 305 está a 5 px de lo recuperado: no se rechaza
        r = res.records[9]
        self.assertEqual((r.cx_raw, r.cx_abs), (300.0, 300.0))  # sin arrastre del suavizado anterior
        self.assertTrue(r.found)
        self.assertEqual(len(fr.llamadas), 2)
        self.assertEqual((res.frames_reacquired, res.frames_recovered), (1, 1))

    def test_si_el_tracker_ya_vio_al_sujeto_no_hace_falta_buscar(self):
        """Si en el frame de la búsqueda el propio tracker encuentra algo plausible, la recuperación normal se adelanta."""
        fr = FakeReacquirer([None, (300, 100)])
        res = run_script([100, 110] + [None] * 7 + [305], reacquirer=fr)  # a los 7 frames sin dato aparece 305
        self.assertEqual(res.records[9].status, "recuperado")
        self.assertEqual(res.records[9].cx_abs, 305.0)  # la posición del tracker, no la del buscador
        self.assertEqual((res.frames_reacquired, len(fr.llamadas)), (0, 1))

    def test_el_tracker_se_recoloca_donde_se_encontro_al_sujeto(self):
        fr = FakeReacquirer([(300, 100)])
        reader = FakeReader()
        tracker = Scripted([100, 110, None, None, None, None, 305])
        job = TrackingJob(reader, FrameRange(0, 6), compute_timing(30.0), tracker, (0, 0, 4000, 200),
                          start=((100.0, 100.0), (95, 95, 10, 10)), reacquirer=fr)
        res = run_tracking(job, keep_records=True, log=None)
        self.assertEqual(statuses(res)[4], "recuperado")  # hueco de 3 frames: índices 2, 3, 4
        self.assertEqual(statuses(res)[5], "perdido")  # el tracker de mentira no ve nada en el frame 5
        self.assertEqual(statuses(res)[6], "detectado")  # y el 6 está a 5 px de lo recuperado: sin rechazo de salto
        self.assertEqual(tracker.last_center, (305.0, 100.0))

    def test_tambien_cuentan_los_frames_rechazados_por_el_filtro_de_saltos(self):
        fr = FakeReacquirer([(905, 100)])
        res = run_script([100, 110, 900, 905, 910], reacquirer=fr)  # tres detecciones lejanas seguidas -> rechazadas
        self.assertEqual(statuses(res), ["detectado", "detectado", "rechazado", "rechazado", "recuperado"])
        self.assertEqual(len(fr.llamadas), 1)

    def test_lo_encontrado_no_pasa_por_el_rechazo_de_saltos(self):
        fr = FakeReacquirer([(3000, 100)])  # a 2900 px de la última posición válida
        res = run_script([100, 110, None, None, None], reacquirer=fr)
        self.assertEqual(statuses(res)[-1], "recuperado")
        self.assertEqual(res.records[-1].cx_abs, 3000.0)

    def test_se_le_entrega_la_ultima_posicion_del_tracker(self):
        fr = FakeReacquirer([None])
        run_script([100, 110, None, None, None], reacquirer=fr)
        self.assertEqual(fr.llamadas[0][0], (110.0, 100.0))  # la última posición que vio el tracker
        self.assertIsNotNone(fr.llamadas[0][1])

    def test_el_evento_trae_la_deteccion_reencontrada(self):
        fr = FakeReacquirer([(300, 100)])
        vistos = {}

        def on_frame(ev):
            if ev.record.status == "recuperado":
                vistos["det"] = ev.detection

        reader = FakeReader()
        job = TrackingJob(reader, FrameRange(0, 4), compute_timing(30.0), Scripted([100, 110, None, None, None]), (0, 0, 4000, 200),
                          start=((100.0, 100.0), (95, 95, 10, 10)), reacquirer=fr)
        run_tracking(job, on_frame=on_frame, log=None)
        self.assertEqual(vistos["det"].center, (300.0, 100.0))  # la interfaz puede dibujar el cuadro en el lugar correcto

    def test_si_el_buscador_falla_se_avisa_y_la_corrida_continua(self):
        fr = FakeReacquirer([RuntimeError("modelo corrupto")])
        res = run_script([100, 110] + [None] * 20, reacquirer=fr)
        self.assertEqual(len(fr.llamadas), 1)  # se desactivó tras el primer fallo
        self.assertEqual(res.frames_processed, 22)
        self.assertTrue(any("modelo corrupto" in w and "desactivó" in w for w in res.warnings))

    def test_el_mensaje_va_a_log(self):
        mensajes = []
        reader = FakeReader()
        job = TrackingJob(reader, FrameRange(0, 4), compute_timing(30.0), Scripted([100, 110, None, None, None]), (0, 0, 4000, 200),
                          start=((100.0, 100.0), (95, 95, 10, 10)), reacquirer=FakeReacquirer([(300, 100)]))
        run_tracking(job, log=mensajes.append)
        self.assertEqual(mensajes, ["Sujeto reencontrado en el frame 4"])

    def test_el_informe_de_calidad_cuenta_lo_recuperado(self):
        res = run_script([100, 110] + [None] * 8, reacquirer=FakeReacquirer([(300, 100)]))
        rep = assess_quality(res.records, res.timing.dt)
        self.assertEqual(rep.frames_recovered, res.frames_recovered)
        self.assertTrue(any("recuperada" in r for r in rep.reasons))


class BusquedaConSilluetaRealTests(unittest.TestCase):
    """El caso que motivó esto: el ratón sale de la ventana de búsqueda de la silueta y no se vuelve a ver."""

    W, H = 400, 300

    @staticmethod
    def escena(n=70, w=400, h=300):
        frames = []
        for i in range(n):
            f = np.full((h, w, 3), 175, np.uint8)
            x, y = (80 + 2 * i, 110) if i < 20 else (300, 220)  # a partir del frame 20 "salta" fuera de la ventana
            cv2.ellipse(f, (x, y), (15, 9), 0, 0, 360, (45, 45, 45), -1)
            frames.append(f)
        return frames

    class FramesReader:
        fps, total_frames = 30.0, 1000

        def __init__(self, frames):
            self.frames, self.height, self.width = frames, frames[0].shape[0], frames[0].shape[1]

        def iter_frames(self, s, e, step):
            for i in range(s, min(e, len(self.frames) - 1) + 1, step):
                yield i, self.frames[i].copy()

    def correr(self, con_buscador, **kw):
        frames = self.escena()
        mask = np.full((self.H, self.W), 255, np.uint8)
        tr = SilhouetteTracker.from_bbox(mask, frames[0], (60, 95, 40, 30))
        job = TrackingJob(self.FramesReader(frames), FrameRange(0, 69), compute_timing(30.0), tr, (0, 0, self.W, self.H),
                          reacquirer=SilhouetteReacquirer(tr) if con_buscador else None, **kw)
        return run_tracking(job, keep_records=True, log=None)

    def test_sin_buscador_el_sujeto_se_pierde_para_siempre(self):
        res = self.correr(False)
        self.assertEqual(sum(r.found for r in res.records[:20]), 20)
        self.assertEqual(sum(r.found for r in res.records[20:]), 0)

    def test_con_buscador_se_reencuentra_y_se_sigue(self):
        res = self.correr(True)
        self.assertEqual(sum(r.found for r in res.records[:20]), 20)
        self.assertEqual(res.frames_reacquired, 1)
        r = next(r for r in res.records if r.status == "recuperado")
        self.assertEqual(r.frame, 22)  # hueco de 3 frames (20, 21, 22) -> primera búsqueda
        self.assertLess(math.dist((r.cx_abs, 300 - r.cy_abs_inv), (300, 220)), 3.0)  # lo encontró donde está
        despues = res.records[22:]
        self.assertEqual(sum(x.found for x in despues), len(despues))  # y lo sigue sin perderlo
        for x in despues:
            self.assertLess(math.dist((x.cx_raw, 300 - x.cy_raw_inv), (300, 220)), 3.0)


class SinInterfazTests(unittest.TestCase):
    def test_pipeline_no_importa_tkinter_ni_matplotlib(self):
        codigo = "import sys, pipeline; print([m for m in ('tkinter', 'matplotlib', 'matplotlib.pyplot') if m in sys.modules])"
        out = subprocess.run([sys.executable, "-c", codigo], cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": ROOT})
        self.assertEqual(out.returncode, 0, msg=out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()