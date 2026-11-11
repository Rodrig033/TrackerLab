"""Pruebas de persistence/run_inputs.py y de tools/replay.py.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_run_inputs.py -v

La propiedad central: GRABAR las entradas de una corrida y REPETIRLA debe dar exactamente los mismos
registros, frame por frame, en los tres modos de seguimiento.
"""

import ast
import csv
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from unittest import mock

import cv2
import numpy as np

from analysis.spatial_metrics import ZoneGeometry
from persistence.run_inputs import KEEP, build_job, load_run_inputs, save_run_inputs
from pipeline import TrackingJob, run_tracking
from tracking.background_tracker import DEFAULT_PARAMS, BackgroundTracker, compute_background
from tracking.color_tracker import ColorTracker, finalize_params, initial_params_from_bbox
from tracking.reacquisition import SilhouetteReacquirer
from tracking.silhouette_tracker import SilhouetteTracker
from video.frame_selection import FrameRange, compute_timing, read_background_frames
from video.reader import VideoReader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.replay as RP  # noqa: E402

try:  # tools/compare_csv.py solo hace falta para las pruebas de --comparar
    import tools.compare_csv  # noqa: F401, E402
    HAY_COMPARE = True
except ImportError:
    HAY_COMPARE = False
NECESITA_COMPARE = unittest.skipUnless(HAY_COMPARE, "falta tools/compare_csv.py")


def _imp(name):
    try:
        return importlib.import_module(f"tests.{name}")
    except ImportError:
        return importlib.import_module(name)


BG = _imp("test_background_tracker")
COL = _imp("test_color_tracker")
TP = _imp("test_pipeline")

AHORA = lambda: datetime(2026, 10, 22, 10, 15, 0)  # noqa: E731


def escribir_video(tmp, nombre, frames, fps=25.0):
    path = os.path.join(tmp, nombre)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return path


class Escenas:
    """Tres videos sintéticos (uno por modo) y la forma de armar el TrackingJob de cada uno, como lo haría main.py."""

    @classmethod
    def preparar(cls, tmp):
        frames, _ = BG.make_frames(90, lost={30, 31, 32})
        cls.p_bg = escribir_video(tmp, "bg.avi", frames)
        cls.mask_bg = BG.roi_mask("elipse")
        cls.zona = np.zeros((BG.FH, BG.FW), np.uint8)
        cv2.rectangle(cls.zona, (60, 60), (160, 170), 255, -1)
        cls.p_sil = escribir_video(tmp, "sil.avi", TP.BusquedaConSilluetaRealTests.escena(), fps=30.0)
        frames, cls.truth_col = COL.make_frames(60, COL.GREEN, distractor=COL.BLUE, twin=True)
        cls.p_col = escribir_video(tmp, "col.avi", frames)

    @classmethod
    def job_fondo(cls, r, **kw):
        bg = compute_background(read_background_frames(r, 35), BG.ROI, cls.mask_bg, False)
        tr = BackgroundTracker(cls.mask_bg, bg, dict(DEFAULT_PARAMS), polarity="white_on_black", apply_bw=False, use_clahe=False)
        return TrackingJob(r, FrameRange(0, 85), compute_timing(r.fps, True, r.fps / 2), tr, BG.ROI,
                           zones=[ZoneGeometry("zona_A", cls.zona, (110.0, 115.0))], cm_per_px=0.07, xy_origin=(50.0, 200.0),
                           start=((40.0, 90.0), (28, 78, 24, 24)), **kw)

    OPT_FONDO = {"tracking_mode": "1", "polarity": "white_on_black", "custom_tracking_fps": True, "tracking_fps": 12.5,
                 "use_clahe": False, "apply_bw": False, "exp_name": "demo", "csv_base_name": "RATA", "root": object()}

    @classmethod
    def job_silueta(cls, r, border_aware=True, buscador=True, **kw):
        tr = SilhouetteTracker.from_bbox(np.full((300, 400), 255, np.uint8), r.read_at(0), (60, 95, 40, 30), border_aware=border_aware)
        return TrackingJob(r, FrameRange(0, 69), compute_timing(r.fps), tr, (0, 0, 400, 300), start=((80.0, 110.0), (60, 95, 40, 30)),
                           reacquirer=SilhouetteReacquirer(tr) if buscador else None, **kw)

    OPT_SIL = {"tracking_mode": "2", "exp_name": "demo"}

    @classmethod
    def job_color(cls, r, **kw):
        m = COL.roi_mask("elipse")
        x0, y0, w0, h0 = COL.ROI
        cx, cy = cls.truth_col[0][0] - x0, cls.truth_col[0][1] - y0
        params = initial_params_from_bbox(r.read_at(0)[y0:y0 + h0, x0:x0 + w0], (cx - 14, cy - 14, 28, 28), m)
        params.update({"S_MIN": max(params["S_MIN"], 120), "V_MIN": max(params["V_MIN"], 100)})
        return TrackingJob(r, FrameRange(0, 59), compute_timing(r.fps), ColorTracker(m, finalize_params(params)), COL.ROI,
                           start=((cx, cy), (cx - 14, cy - 14, 28, 28)), **kw)

    OPT_COL = {"tracking_mode": "3", "exp_name": "demo"}


def ida_y_vuelta(base, path, hacer_job, opciones, sample_idx=0, **overrides):
    """Graba las entradas del trabajo, lo corre, carga lo grabado, reconstruye el trabajo y lo vuelve a correr."""
    with VideoReader(path) as r:
        job = hacer_job(r)
        # como en main.py: las entradas se graban ANTES de correr (el modelo de silueta cambia durante la corrida)
        carpeta = save_run_inputs(base, job, video_path=path, options=opciones, sample_idx=sample_idx, app_version="1.0",
                                  log=None, now=AHORA)
        original = run_tracking(job, keep_records=True, log=None)
    assert carpeta is not None, "no se pudo grabar"
    with VideoReader(path) as r2:
        repetida = run_tracking(build_job(load_run_inputs(carpeta), r2, **overrides), keep_records=True, log=None)
    return original, repetida, carpeta


# ======================================================================
class IdaYVueltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        Escenas.preparar(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def base(self, nombre):
        return os.path.join(self.tmp.name, "exp_" + nombre)

    def comprobar(self, original, repetida):
        self.assertGreater(len(original.records), 20)
        self.assertEqual(original.records, repetida.records)  # idénticos, registro por registro
        self.assertEqual(original.settings, repetida.settings)
        self.assertEqual((original.frames_found, original.frames_rejected, original.frames_recovered, original.frames_reacquired),
                         (repetida.frames_found, repetida.frames_rejected, repetida.frames_recovered, repetida.frames_reacquired))

    def test_modo_fondo_con_zona_escala_origen_y_velocidad_personalizada(self):
        o, r, _ = ida_y_vuelta(self.base("fondo"), Escenas.p_bg, Escenas.job_fondo, Escenas.OPT_FONDO, sample_idx=35)
        self.comprobar(o, r)
        self.assertEqual(o.timing.frame_step, 2)  # la velocidad personalizada se reconstruyó
        self.assertTrue(any(rec.zones for rec in r.records))
        self.assertTrue(any(rec.x_rel_cm is not None for rec in r.records))

    def test_la_asistencia_del_detector_se_graba_y_se_conserva_al_repetir(self):
        constancia = {"modo": "proponer_sujeto", "sujeto": {"metodo": "detector", "motivo": "propuesta aceptada", "modelo": "rfdetr-small.onnx",
                                              "confianza": 0.86, "caja_propuesta": [28, 78, 24, 24], "umbral": 0.5, "margen_roi_px": 0,
                                              "notas_modelo": {"corrida": "rfdetr_small_20261110_214151", "version_rfdetr": "1.12.0"}}}
        import dataclasses

        def hacer(r):
            return dataclasses.replace(Escenas.job_fondo(r), assistance=constancia)
        o, r, carpeta = ida_y_vuelta(self.base("asistencia"), Escenas.p_bg, hacer, Escenas.OPT_FONDO, sample_idx=35)
        self.comprobar(o, r)  # registros idénticos: la constancia no cambia el seguimiento
        self.assertEqual(o.settings["asistencia_detector"], constancia)
        self.assertEqual(r.settings["asistencia_detector"], constancia)  # la corrida repetida dice cómo se eligió al sujeto
        self.assertEqual(load_run_inputs(carpeta).assistance, constancia)
        with open(os.path.join(carpeta, "inputs.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["asistencia_detector"], constancia)

    def test_la_constancia_no_cambia_el_seguimiento(self):
        """La misma corrida con y sin constancia: registros idénticos."""
        import dataclasses
        with VideoReader(Escenas.p_bg) as r:
            sin = run_tracking(Escenas.job_fondo(r), keep_records=True, log=None)
        with VideoReader(Escenas.p_bg) as r:
            con = run_tracking(dataclasses.replace(Escenas.job_fondo(r), assistance={"modo": "proponer_sujeto", "sujeto": {}}), keep_records=True, log=None)
        self.assertEqual(sin.records, con.records)
        self.assertIsNone(sin.settings["asistencia_detector"])

    def test_modo_silueta_con_busqueda_y_seleccion_independiente_del_margen(self):
        o, r, _ = ida_y_vuelta(self.base("sil"), Escenas.p_sil, Escenas.job_silueta, Escenas.OPT_SIL)
        self.comprobar(o, r)
        self.assertEqual(o.frames_reacquired, 1)  # la búsqueda se grabó y se repitió
        self.assertEqual(r.settings["seleccion_independiente_del_margen"], True)

    def test_modo_silueta_sin_busqueda_ni_seleccion_independiente(self):
        o, r, _ = ida_y_vuelta(self.base("sil2"), Escenas.p_sil, lambda rd: Escenas.job_silueta(rd, border_aware=False, buscador=False), Escenas.OPT_SIL)
        self.comprobar(o, r)
        self.assertEqual((r.settings["buscador_sujeto"], r.settings["seleccion_independiente_del_margen"]), (None, False))

    def test_modo_color(self):
        o, r, _ = ida_y_vuelta(self.base("color"), Escenas.p_col, Escenas.job_color, Escenas.OPT_COL)
        self.comprobar(o, r)

    def test_las_opciones_de_filtrado_grabadas_se_conservan(self):
        kw = dict(smoothing_alpha=0.5, jump_max_px=80.0, adopt_after_s=3.0, recovery_after=4, recovery_distance_px=200.0)
        o, r, _ = ida_y_vuelta(self.base("opc"), Escenas.p_bg, lambda rd: Escenas.job_fondo(rd, **kw), Escenas.OPT_FONDO, sample_idx=35)
        self.comprobar(o, r)
        self.assertEqual((r.settings["suavizado_alpha"], r.settings["salto_max_px"], r.settings["adopcion_tras_s"]), (0.5, 80.0, 3.0))

    def test_filtros_apagados_grabados_se_conservan_apagados(self):
        kw = dict(smoothing_alpha=None, jump_max_px=None)
        o, r, _ = ida_y_vuelta(self.base("raw"), Escenas.p_bg, lambda rd: Escenas.job_fondo(rd, **kw), Escenas.OPT_FONDO, sample_idx=35)
        self.comprobar(o, r)
        self.assertIsNone(r.settings["suavizado_alpha"])

    def test_cambiar_opciones_al_repetir_equivale_a_correr_con_esas_opciones(self):
        """Repetir una corrida normal con --sin-filtros = haberla corrido sin filtros desde el principio."""
        _, sin_filtros, _ = ida_y_vuelta(self.base("ov"), Escenas.p_bg, Escenas.job_fondo, Escenas.OPT_FONDO, sample_idx=35,
                                         smoothing_alpha=None, jump_max_px=None)
        with VideoReader(Escenas.p_bg) as r:
            directo = run_tracking(Escenas.job_fondo(r, smoothing_alpha=None, jump_max_px=None), keep_records=True, log=None)
        self.assertEqual(sin_filtros.records, directo.records)
        normal, _, _ = ida_y_vuelta(self.base("ov2"), Escenas.p_bg, Escenas.job_fondo, Escenas.OPT_FONDO, sample_idx=35)
        self.assertNotEqual(normal.records, sin_filtros.records)  # y los filtros SÍ cambian los números

    def test_apagar_la_busqueda_y_la_seleccion_al_repetir(self):
        with VideoReader(Escenas.p_sil) as r:
            job = Escenas.job_silueta(r)
            carpeta = save_run_inputs(self.base("ov3"), job, video_path=Escenas.p_sil, options=Escenas.OPT_SIL, sample_idx=0, log=None, now=AHORA)
        inp = load_run_inputs(carpeta)
        with VideoReader(Escenas.p_sil) as r:
            sin_busqueda = build_job(inp, r, reacquire=False)
            self.assertIsNone(sin_busqueda.reacquirer)
            self.assertEqual(run_tracking(sin_busqueda, log=None).frames_reacquired, 0)
            sin_margen = build_job(inp, r, border_aware=False)
            self.assertFalse(sin_margen.tracker.border_aware)
            self.assertIsNotNone(build_job(inp, r).reacquirer)  # por omisión se conserva lo grabado


class TrabajoReconstruidoTests(unittest.TestCase):
    """El TrabajoReconstruido tiene los MISMOS campos que el original (no solo los mismos resultados:
    un desvío pequeño en un campo puede no alterar los números de una escena y sí los de otra)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        Escenas.preparar(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pares(self, path, hacer_job, opciones, sample_idx=0):
        with VideoReader(path) as r, VideoReader(path) as r2:
            original = hacer_job(r)
            carpeta = save_run_inputs(os.path.join(self.tmp.name, "exp"), original, video_path=path, options=opciones, sample_idx=sample_idx,
                                      log=None, now=AHORA)
            return original, build_job(load_run_inputs(carpeta), r2)

    def comparar_comunes(self, a, b):
        self.assertEqual((a.frame_range.start, a.frame_range.end), (b.frame_range.start, b.frame_range.end))
        self.assertEqual((a.timing.frame_step, a.timing.dt, a.timing.video_fps), (b.timing.frame_step, b.timing.dt, b.timing.video_fps))
        self.assertEqual(tuple(a.roi_box), tuple(b.roi_box))
        self.assertEqual((a.cm_per_px, a.xy_origin), (b.cm_per_px, b.xy_origin))
        self.assertEqual(a.start, b.start)  # centro y caja iniciales, exactos
        for k in ("smoothing_alpha", "jump_max_px", "jump_max_speed_px_s", "recovery_after", "recovery_distance_px", "adopt_after_s",
                  "reacquire_after_frames", "reacquire_every"):
            self.assertEqual(getattr(a, k), getattr(b, k), msg=k)
        self.assertEqual([z.name for z in a.zones], [z.name for z in b.zones])
        self.assertEqual([tuple(z.centroid) for z in a.zones], [tuple(z.centroid) for z in b.zones])
        for za, zb in zip(a.zones, b.zones):
            self.assertTrue(np.array_equal(za.mask, zb.mask))
        self.assertTrue(np.array_equal(a.tracker.include_mask, b.tracker.include_mask))
        self.assertEqual(type(a.tracker), type(b.tracker))
        self.assertEqual(type(a.reacquirer), type(b.reacquirer))

    def test_fondo(self):
        a, b = self.pares(Escenas.p_bg, Escenas.job_fondo, Escenas.OPT_FONDO, 35)
        self.comparar_comunes(a, b)
        self.assertEqual((a.tracker.params, a.tracker.polarity, a.tracker.apply_bw, a.tracker.use_clahe),
                         (b.tracker.params, b.tracker.polarity, b.tracker.apply_bw, b.tracker.use_clahe))
        self.assertTrue(np.array_equal(a.tracker.bg_gray, b.tracker.bg_gray))  # el fondo reconstruido es idéntico

    def test_silueta(self):
        a, b = self.pares(Escenas.p_sil, Escenas.job_silueta, Escenas.OPT_SIL)
        self.comparar_comunes(a, b)
        self.assertEqual((a.tracker.border_aware, a.tracker.model.polarity, a.tracker.model.area_ref),
                         (b.tracker.border_aware, b.tracker.model.polarity, b.tracker.model.area_ref))
        self.assertEqual(a.tracker.model.bbox, b.tracker.model.bbox)

    def test_color(self):
        a, b = self.pares(Escenas.p_col, Escenas.job_color, Escenas.OPT_COL)
        self.comparar_comunes(a, b)
        self.assertEqual(a.tracker.params, b.tracker.params)


class ContenidoGrabadoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        Escenas.preparar(cls.tmp.name)
        with VideoReader(Escenas.p_bg) as r:
            cls.job = Escenas.job_fondo(r)
            cls.carpeta = save_run_inputs(os.path.join(cls.tmp.name, "exp"), cls.job, video_path=Escenas.p_bg,
                                          options=Escenas.OPT_FONDO, sample_idx=35, app_version="1.0", log=None, now=AHORA)
        with open(os.path.join(cls.carpeta, "inputs.json"), encoding="utf-8") as f:
            cls.data = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_sin_detector_la_constancia_es_nula(self):
        self.assertIn("asistencia_detector", self.data)
        self.assertIsNone(self.data["asistencia_detector"])
        self.assertIsNone(load_run_inputs(self.carpeta).assistance)

    def test_unas_entradas_anteriores_a_la_constancia_siguen_cargando(self):
        import shutil
        vieja = os.path.join(self.tmp.name, "entradas_vieja")
        shutil.copytree(self.carpeta, vieja)
        with open(os.path.join(vieja, "inputs.json"), encoding="utf-8") as f:
            d = json.load(f)
        del d["asistencia_detector"]
        with open(os.path.join(vieja, "inputs.json"), "w", encoding="utf-8") as f:
            json.dump(d, f)
        ins = load_run_inputs(vieja)
        self.assertIsNone(ins.assistance)
        with VideoReader(Escenas.p_bg) as r:
            self.assertIsNone(build_job(ins, r).assistance)

    def test_nombre_de_la_carpeta_y_archivos(self):
        self.assertEqual(os.path.basename(self.carpeta), "entradas_20261022_101500")
        self.assertEqual(sorted(os.listdir(self.carpeta)), ["inputs.json", "roi_mask.png", "zone_0.png"])

    def test_el_json_tiene_lo_necesario_y_solo_tipos_simples(self):
        d = self.data
        self.assertEqual(d["schema"], 1)
        self.assertEqual(d["trackerlab_version"], "1.0")
        self.assertEqual(d["frame_range"], {"start": 0, "end": 85})
        self.assertEqual(d["sample_idx"], 35)
        self.assertEqual(d["roi_box"], list(BG.ROI))
        self.assertEqual(d["subject"], {"center": [40.0, 90.0], "bbox": [28, 78, 24, 24]})
        self.assertEqual((d["cm_per_px"], d["xy_origin"]), (0.07, [50.0, 200.0]))
        self.assertEqual(d["timing"]["frame_step"], 2)
        self.assertEqual(d["tracker"]["mode"], "fondo")
        self.assertEqual(d["tracker"]["params"], dict(DEFAULT_PARAMS))
        self.assertEqual([z["name"] for z in d["zones"]], ["zona_A"])
        self.assertEqual(d["pipeline"]["smoothing_alpha"], 0.35)
        self.assertIsNone(d["pipeline"]["reacquirer"])
        self.assertIn("sha256_extremos", d["video"])
        self.assertEqual(d["video"]["path"], Escenas.p_bg)

    def test_solo_se_graban_las_opciones_serializables(self):
        self.assertNotIn("root", self.data["options"])  # el objeto de Tk no se puede (ni hace falta) grabar
        self.assertEqual(self.data["options"]["tracking_mode"], "1")
        self.assertEqual(self.data["options"]["tracking_fps"], 12.5)

    def test_las_mascaras_png_son_exactas(self):
        inp = load_run_inputs(self.carpeta)
        self.assertTrue(np.array_equal(inp.include_mask, Escenas.mask_bg))
        self.assertTrue(np.array_equal(inp.zone_masks[0], Escenas.zona))
        self.assertEqual(inp.include_mask.dtype, np.uint8)

    def test_numpy_se_convierte_a_tipos_simples(self):
        with VideoReader(Escenas.p_bg) as r:
            job = Escenas.job_fondo(r)
            job.cm_per_px = np.float64(0.07)
            job.xy_origin = (np.float32(50.0), np.int64(200))
            job.roi_box = tuple(np.int64(v) for v in BG.ROI)
            carpeta = save_run_inputs(os.path.join(self.tmp.name, "exp_np"), job, video_path=Escenas.p_bg, options={"tracking_mode": np.str_("1")},
                                      sample_idx=np.int64(35), log=None, now=AHORA)
        with open(os.path.join(carpeta, "inputs.json"), encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual((d["cm_per_px"], d["xy_origin"], d["roi_box"]), (0.07, [50.0, 200], list(BG.ROI)))

    def test_cada_grabacion_va_en_su_carpeta(self):
        with VideoReader(Escenas.p_bg) as r:
            c1 = save_run_inputs(os.path.join(self.tmp.name, "exp2"), Escenas.job_fondo(r), video_path=Escenas.p_bg, options={}, sample_idx=35,
                                 log=None, now=lambda: datetime(2026, 10, 22, 10, 0, 0))
            c2 = save_run_inputs(os.path.join(self.tmp.name, "exp2"), Escenas.job_fondo(r), video_path=Escenas.p_bg, options={}, sample_idx=35,
                                 log=None, now=lambda: datetime(2026, 10, 22, 11, 0, 0))
        self.assertNotEqual(c1, c2)
        self.assertTrue(os.path.exists(os.path.join(c1, "inputs.json")) and os.path.exists(os.path.join(c2, "inputs.json")))

    def test_el_mensaje_dice_donde_quedaron(self):
        mensajes = []
        with VideoReader(Escenas.p_bg) as r:
            c = save_run_inputs(os.path.join(self.tmp.name, "exp3"), Escenas.job_fondo(r), video_path=Escenas.p_bg, options={}, sample_idx=35,
                                log=mensajes.append, now=AHORA)
        self.assertEqual(mensajes, [f"Entradas guardadas (para repetir esta corrida sin ventanas): {c}"])


class IntegridadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        Escenas.preparar(cls.tmp.name)
        with VideoReader(Escenas.p_sil) as r:
            cls.carpeta = save_run_inputs(os.path.join(cls.tmp.name, "exp"), Escenas.job_silueta(r), video_path=Escenas.p_sil,
                                          options=Escenas.OPT_SIL, sample_idx=0, log=None, now=AHORA)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def modificar(self, cambio):
        copia = tempfile.mkdtemp(dir=self.tmp.name)
        for f in os.listdir(self.carpeta):
            with open(os.path.join(self.carpeta, f), "rb") as a, open(os.path.join(copia, f), "wb") as b:
                b.write(a.read())
        with open(os.path.join(copia, "inputs.json"), encoding="utf-8") as f:
            d = json.load(f)
        cambio(d)
        with open(os.path.join(copia, "inputs.json"), "w", encoding="utf-8") as f:
            json.dump(d, f)
        return copia

    def test_un_video_con_otros_fps_se_rechaza(self):
        otro = escribir_video(self.tmp.name, "otros_fps.avi", TP.BusquedaConSilluetaRealTests.escena(), fps=15.0)
        with VideoReader(otro) as r:
            with self.assertRaisesRegex(ValueError, "no es el mismo video"):
                build_job(load_run_inputs(self.carpeta), r)

    def test_una_velocidad_de_procesamiento_que_no_coincide_se_rechaza(self):
        carpeta = self.modificar(lambda d: d["timing"].update(frame_step=3))
        with VideoReader(Escenas.p_sil) as r:
            with self.assertRaisesRegex(ValueError, "velocidad de procesamiento"):
                build_job(load_run_inputs(carpeta), r)

    def test_el_modelo_de_silueta_distinto_al_grabado_se_detecta(self):
        carpeta = self.modificar(lambda d: d["tracker"]["model"].update(area_ref=123456.0))
        with VideoReader(Escenas.p_sil) as r:
            with self.assertRaisesRegex(ValueError, "no coincide con el grabado"):
                build_job(load_run_inputs(carpeta), r)
            self.assertIsNotNone(build_job(load_run_inputs(carpeta), r, strict=False))  # strict=False sigue adelante

    def test_cambiar_la_seleccion_a_proposito_no_dispara_la_comprobacion(self):
        carpeta = self.modificar(lambda d: d["tracker"]["model"].update(area_ref=123456.0))
        with VideoReader(Escenas.p_sil) as r:
            self.assertIsNotNone(build_job(load_run_inputs(carpeta), r, border_aware=False))  # el modelo puede diferir legítimamente

    def test_una_mascara_de_roi_de_otro_tamano_se_rechaza(self):
        carpeta = self.modificar(lambda d: d.update(roi_box=[0, 0, 100, 100]))
        with VideoReader(Escenas.p_sil) as r:
            with self.assertRaisesRegex(ValueError, "máscara del ROI"):
                build_job(load_run_inputs(carpeta), r)

    def test_un_modo_desconocido_se_rechaza(self):
        carpeta = self.modificar(lambda d: d["tracker"].update(mode="raro"))
        with VideoReader(Escenas.p_sil) as r:
            with self.assertRaisesRegex(ValueError, "desconocido"):
                build_job(load_run_inputs(carpeta), r)

    def test_errores_al_cargar(self):
        with self.assertRaises(FileNotFoundError):
            load_run_inputs(os.path.join(self.tmp.name, "no_existe"))
        with self.assertRaisesRegex(ValueError, "Versión de formato"):
            load_run_inputs(self.modificar(lambda d: d.update(schema=99)))
        sin_png = self.modificar(lambda d: None)
        os.remove(os.path.join(sin_png, "roi_mask.png"))
        with self.assertRaises(OSError):
            load_run_inputs(sin_png)


class GrabarNuncaRompeLaCorridaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        Escenas.preparar(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def grabar(self, base, job, **kw):
        mensajes = []
        res = save_run_inputs(base, job, video_path=Escenas.p_bg, options={}, sample_idx=35, log=mensajes.append, now=AHORA, **kw)
        return res, mensajes

    def test_carpeta_imposible(self):
        archivo = os.path.join(self.tmp.name, "archivo")
        open(archivo, "w").close()
        with VideoReader(Escenas.p_bg) as r:
            res, m = self.grabar(archivo, Escenas.job_fondo(r))
        self.assertIsNone(res)
        self.assertTrue(m and "el seguimiento continúa" in m[0])

    def test_tracker_no_soportado(self):
        reader = TP.FakeReader()
        job = TrackingJob(reader, FrameRange(0, 3), compute_timing(30.0), TP.Scripted([100, 110]), (0, 0, 4000, 200), start=((100.0, 100.0), (95, 95, 10, 10)))
        res, m = self.grabar(os.path.join(self.tmp.name, "exp"), job)
        self.assertIsNone(res)
        self.assertIn("no soportado", m[0])

    def test_trabajo_sin_posicion_inicial(self):
        with VideoReader(Escenas.p_bg) as r:
            job = Escenas.job_fondo(r)
            job.start = None
            res, m = self.grabar(os.path.join(self.tmp.name, "exp"), job)
        self.assertIsNone(res)
        self.assertIn("posición inicial", m[0])

    def test_no_se_puede_escribir_una_mascara(self):
        with VideoReader(Escenas.p_bg) as r, mock.patch.object(cv2, "imwrite", lambda *a, **k: False):
            res, m = self.grabar(os.path.join(self.tmp.name, "exp"), Escenas.job_fondo(r))
        self.assertIsNone(res)
        self.assertIn("OpenCV no pudo escribir", m[0])

    def test_una_mascara_con_formato_inesperado(self):
        with VideoReader(Escenas.p_bg) as r:
            job = Escenas.job_fondo(r)
            job.zones = [ZoneGeometry("z", np.zeros((BG.FH, BG.FW), np.uint8), (1.0, 1.0))]
            job.zones[0].mask = job.zones[0].mask.astype(np.float32)
            res, m = self.grabar(os.path.join(self.tmp.name, "exp"), job)
        self.assertIsNone(res)
        self.assertIn("uint8", m[0])

    def test_sin_log_no_imprime_ni_lanza(self):
        with VideoReader(Escenas.p_bg) as r:
            self.assertIsNone(save_run_inputs(os.path.join(self.tmp.name, "archivo_no_dir", "x\0y"), Escenas.job_fondo(r), video_path=Escenas.p_bg,
                                              options={}, sample_idx=35, log=None, now=AHORA))


# ======================================================================
class HerramientaReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        Escenas.preparar(cls.tmp.name)
        with VideoReader(Escenas.p_bg) as r:
            job = Escenas.job_fondo(r)
            cls.original = run_tracking(job, keep_records=True, log=None)
            cls.carpeta = save_run_inputs(os.path.join(cls.tmp.name, "exp"), job, video_path=Escenas.p_bg, options=Escenas.OPT_FONDO,
                                          sample_idx=35, log=None, now=AHORA)
            zonas = job.zones
        # CSV por frame de la corrida original (para probar la comparación). Que el formato coincide con el de la aplicación
        # lo comprueba test_el_csv_repetido_es_identico_al_que_escribe_export_tracking_csv_de_main
        cls.csv_app = os.path.join(cls.tmp.name, "app_FRAMES.csv")
        RP.write_frames_csv(cls.csv_app, cls.original.records, zonas)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ejecutar(self, **kw):
        salida = []
        kw.setdefault("salida", os.path.join(self.tmp.name, f"sal_{len(os.listdir(self.tmp.name))}"))
        rc = RP.run(self.carpeta, out=salida.append, **kw)
        return rc, "\n".join(salida), kw["salida"]

    def test_repite_y_escribe_resultados(self):
        rc, texto, salida = self.ejecutar()
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(os.path.join(salida, "replay_FRAMES.csv")))
        self.assertTrue(os.path.exists(os.path.join(salida, "replay_resumen.txt")))
        self.assertIn(f"frames procesados: {self.original.frames_processed}", texto)
        self.assertIn("procesando...", texto)
        self.assertIn("Calidad", texto)

    @NECESITA_COMPARE
    def test_el_csv_repetido_es_identico_al_de_la_corrida_original(self):
        rc, texto, salida = self.ejecutar(comparar=self.csv_app, tol=0.0)
        self.assertEqual(rc, 0, msg=texto)
        with open(self.csv_app, "rb") as a, open(os.path.join(salida, "replay_FRAMES.csv"), "rb") as b:
            self.assertEqual(a.read(), b.read())  # byte a byte

    @NECESITA_COMPARE
    def test_sin_filtros_se_distingue_de_la_corrida_normal(self):
        rc, texto, _ = self.ejecutar(sin_filtros=True, comparar=self.csv_app)
        self.assertEqual(rc, 1)  # los filtros cambian los números, y la comparación lo detecta
        self.assertIn("suavizado=None", texto)

    def test_cambiar_el_suavizado(self):
        rc, texto, _ = self.ejecutar(alpha=0.9)
        self.assertEqual(rc, 0)
        self.assertIn("suavizado=0.9", texto)
        self.assertIn("salto_max_px=120.0", self.ejecutar(salto_px=120.0)[1])

    def test_video_inexistente(self):
        rc, texto, _ = self.ejecutar(video=os.path.join(self.tmp.name, "no.avi"))
        self.assertEqual(rc, 2)
        self.assertIn("--video", texto)

    @NECESITA_COMPARE
    def test_video_movido_se_indica_con_video(self):
        copia = os.path.join(self.tmp.name, "copia.avi")
        with open(Escenas.p_bg, "rb") as a, open(copia, "wb") as b:
            b.write(a.read())
        rc, texto, _ = self.ejecutar(video=copia, comparar=self.csv_app, tol=0.0)
        self.assertEqual(rc, 0, msg=texto)  # copia idéntica: misma huella, mismos resultados
        self.assertNotIn("AVISO", texto)

    def test_un_video_distinto_avisa(self):
        frames, _ = BG.make_frames(90, seed=9, lost={30, 31, 32})
        otro = escribir_video(self.tmp.name, "otro.avi", frames)
        rc, texto, _ = self.ejecutar(video=otro)
        self.assertIn("AVISO: este video no es idéntico", texto)

    def test_carpeta_de_entradas_invalida(self):
        out = []
        self.assertEqual(RP.run(os.path.join(self.tmp.name, "nada"), out=out.append), 2)
        self.assertIn("No se pudieron leer las entradas", out[0])

    def test_csv_a_comparar_inexistente(self):
        rc, texto, _ = self.ejecutar(comparar=os.path.join(self.tmp.name, "no.csv"))
        self.assertEqual(rc, 2)

    def test_la_salida_por_omision_va_dentro_de_la_carpeta_de_entradas(self):
        out = []
        rc = RP.run(self.carpeta, out=out.append, now=lambda: datetime(2026, 10, 23, 8, 0, 0))
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(os.path.join(self.carpeta, "replay_20261023_080000", "replay_FRAMES.csv")))

    @NECESITA_COMPARE
    def test_se_ejecuta_como_script_desde_otra_carpeta(self):
        salida = os.path.join(self.tmp.name, "sal_script")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "replay.py"), self.carpeta, "--salida", salida, "--comparar", self.csv_app, "--tol", "0"],
                           cwd=self.tmp.name, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        self.assertTrue(os.path.exists(os.path.join(salida, "replay_FRAMES.csv")))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "replay.py"), os.path.join(self.tmp.name, "nada")], cwd=self.tmp.name,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)


# ======================================================================
class FormatoDelCsvContraLaAplicacionTests(unittest.TestCase):
    """El CSV de la repetición debe ser idéntico, byte a byte, al que escribe la export_tracking_csv REAL de main.py
    (extraída del archivo), con y sin zonas, origen y escala: las 8 combinaciones."""

    @classmethod
    def setUpClass(cls):
        import builtins
        main_py, csv_py = os.path.join(ROOT, "main.py"), os.path.join(ROOT, "persistence", "csv_export.py")
        if not (os.path.exists(main_py) and os.path.exists(csv_py)):
            raise unittest.SkipTest("no se encontró main.py o persistence/csv_export.py")
        with open(main_py, encoding="utf-8") as f:
            src = f.read()
        fn = next((n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "export_tracking_csv"), None)
        if fn is None:
            raise unittest.SkipTest("main.py no define export_tracking_csv")
        with open(csv_py, encoding="utf-8") as f:
            csrc = f.read()
        aux = {n.name: ast.get_source_segment(csrc, n) for n in ast.parse(csrc).body
               if isinstance(n, ast.FunctionDef) and n.name in ("safe_float_str", "write_csv")}
        cls.tmp = tempfile.TemporaryDirectory()
        cls.carpeta_app = os.path.join(cls.tmp.name, "app")
        ns = {"os": os, "csv": csv, "ensure_dir": lambda d: os.makedirs(d, exist_ok=True)}
        exec("\n\n".join(aux.values()), ns)
        ns["make_output_csv"] = lambda base, exp: os.path.join(cls.carpeta_app, f"{base}_{exp}_FRAMES.csv")
        ns["make_summary_csv"] = lambda base, exp: os.path.join(cls.carpeta_app, f"{base}_{exp}_SUMMARY.csv")
        ns["provenance_rows"] = lambda result, version: []
        ns["APP_VERSION"] = "1.0"
        usados = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        propios = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} | {a.arg for a in ast.walk(fn) if isinstance(a, ast.arg)}
        faltan = sorted(usados - propios - set(dir(builtins)) - set(ns))
        if faltan:
            cls.tmp.cleanup()
            raise unittest.SkipTest(f"export_tracking_csv usa nombres que esta prueba no provee: {faltan}")
        exec(ast.get_source_segment(src, fn), ns)
        cls.export = staticmethod(ns["export_tracking_csv"])
        Escenas.preparar(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def comprobar(self, n_zonas, origen, escala):
        with VideoReader(Escenas.p_bg) as r:
            job = Escenas.job_fondo(r)
            job.xy_origin, job.cm_per_px = origen, escala
            if not n_zonas:
                job.zones = []
            carpeta = save_run_inputs(os.path.join(self.tmp.name, f"exp_{n_zonas}_{origen is not None}_{escala is not None}"), job,
                                      video_path=Escenas.p_bg, options=Escenas.OPT_FONDO, sample_idx=35, log=None, now=AHORA)
            original = run_tracking(job, keep_records=True, log=None)
        frames_app, _ = self.export(original, f"e{n_zonas}{origen is not None}{escala is not None}", "track")
        salida = os.path.join(self.tmp.name, f"sal_{n_zonas}_{origen is not None}_{escala is not None}")
        out = []
        RP.run(carpeta, salida=salida, out=out.append)
        with open(frames_app, "rb") as a, open(os.path.join(salida, "replay_FRAMES.csv"), "rb") as b:
            app, rep = a.read(), b.read()
        self.assertEqual(app, rep, msg=f"zonas={n_zonas} origen={origen} escala={escala}")
        self.assertGreater(app.count(b"\n"), 40)

    def test_sin_zonas_ni_origen_ni_escala(self):  # el caso más simple: lo que tendrías si no calibras nada
        self.comprobar(0, None, None)

    def test_sin_zonas_con_origen_y_escala(self):
        self.comprobar(0, (50.0, 200.0), 0.07)

    def test_sin_zonas_con_origen_sin_escala(self):
        self.comprobar(0, (50.0, 200.0), None)

    def test_sin_zonas_con_escala_sin_origen(self):
        self.comprobar(0, None, 0.07)

    def test_con_zona_sin_origen_ni_escala(self):
        self.comprobar(1, None, None)

    def test_con_zona_con_origen_y_escala(self):
        self.comprobar(1, (50.0, 200.0), 0.07)

    def test_con_zona_con_origen_sin_escala(self):
        self.comprobar(1, (50.0, 200.0), None)

    def test_con_zona_con_escala_sin_origen(self):
        self.comprobar(1, None, 0.07)


# ======================================================================
class EnlaceEnMainTests(unittest.TestCase):
    """Comprueba, leyendo main.py como texto, que la grabación de entradas está bien enchufada."""

    @classmethod
    def setUpClass(cls):
        ruta = os.path.join(ROOT, "main.py")
        if not os.path.exists(ruta):
            raise unittest.SkipTest("no se encontró main.py en la raíz del proyecto")
        with open(ruta, encoding="utf-8") as f:
            cls.tree = ast.parse(f.read())
        cls.main_fn = next((n for n in cls.tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
        if cls.main_fn is None:
            raise unittest.SkipTest("main.py no define main()")

    def llamada(self, nombre):
        return [n for n in ast.walk(self.main_fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == nombre]

    def test_interruptor_y_import(self):
        asig = [n for n in self.tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "SAVE_RUN_INPUTS" for t in n.targets)]
        self.assertEqual(len(asig), 1, "falta SAVE_RUN_INPUTS al inicio de main.py")
        self.assertIs(asig[0].value.value, True)
        imp = [n for n in self.tree.body if isinstance(n, ast.ImportFrom) and n.module == "persistence.run_inputs"]
        self.assertTrue(imp and any(a.name == "save_run_inputs" for a in imp[0].names))

    def test_la_llamada_esta_protegida_por_el_interruptor_y_tiene_los_argumentos(self):
        llamadas = self.llamada("save_run_inputs")
        self.assertEqual(len(llamadas), 1, "falta la llamada a save_run_inputs en main()")
        protegidas = [n for n in ast.walk(self.main_fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "SAVE_RUN_INPUTS"
                      and any(c is llamadas[0] for c in ast.walk(n))]
        self.assertEqual(len(protegidas), 1, "la llamada debe ir dentro de 'if SAVE_RUN_INPUTS:'")
        c = llamadas[0]
        self.assertEqual(ast.unparse(c.args[1]), "job")
        claves = {k.arg: ast.unparse(k.value) for k in c.keywords}
        self.assertEqual(claves["video_path"], "video_path")
        self.assertEqual(claves["options"], "options")
        self.assertEqual(claves["sample_idx"], "sample_idx")
        self.assertEqual(claves["app_version"], "APP_VERSION")

    def test_orden_el_trabajo_existe_antes_de_grabar_y_se_graba_antes_de_correr(self):
        job = [n for n in ast.walk(self.main_fn) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "TrackingJob"]
        run = [n for n in ast.walk(self.main_fn) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "run_tracking"]
        grabar = self.llamada("save_run_inputs")[0]
        self.assertTrue(job[0].lineno < grabar.lineno < run[0].lineno,
                        "debe grabarse DESPUÉS de crear el trabajo y ANTES de correr el seguimiento (si la corrida falla, las entradas quedan)")

    def test_las_variables_usadas_existen_antes_de_grabar(self):
        grabar = self.llamada("save_run_inputs")[0].lineno
        for nombre in ("video_path", "options", "sample_idx", "job"):
            definidas = [n for n in ast.walk(self.main_fn) if isinstance(n, ast.Name) and n.id == nombre and isinstance(n.ctx, ast.Store) and n.lineno < grabar]
            self.assertTrue(definidas, f"{nombre} no está definida antes de grabar")


if __name__ == "__main__":
    unittest.main()