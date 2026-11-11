"""Pruebas de analysis/provenance.py, del registro de opciones en el pipeline y de su integración con
``export_tracking_csv`` de main.py.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_provenance.py -v

La prueba de integración extrae ``export_tracking_csv`` de main.py leyendo el archivo (no lo importa:
importar main.py cargaría Tk y pyplot) y se salta si no encuentra main.py o persistence/csv_export.py.
"""

import ast
import builtins
import csv
import importlib
import os
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

import pipeline
from analysis.provenance import provenance_rows
from analysis.spatial_metrics import ZoneGeometry
from pipeline import TrackingJob, run_tracking
from tracking.background_tracker import DEFAULT_PARAMS, BackgroundTracker, compute_background
from tracking.color_tracker import ColorTracker
from tracking.reacquisition import Reacquirer
from tracking.silhouette_tracker import SilhouetteTracker
from video.frame_selection import FrameRange, compute_timing, read_background_frames
from video.reader import VideoReader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _imp(name):
    try:
        return importlib.import_module(f"tests.{name}")
    except ImportError:
        return importlib.import_module(name)


TP = _imp("test_pipeline")
BG = _imp("test_background_tracker")
SIL = _imp("test_silhouette_tracker")
COL = _imp("test_color_tracker")

# Filas base del resumen de main.py (en su orden): no deben cambiar nunca.
BASE_LABELS = [
    "experimento", "frames_procesados", "frames_detectados", "frame_inicio", "frame_fin", "fps_video_original",
    "fps_tracking_solicitado", "fps_tracking_efectivo", "tracking_frame_step", "dt_s", "frames_validos", "tiempo_valido_s",
    "dist_total_px", "dist_total_cm", "vel_media_px_s", "vel_media_cm_s", "acel_media_px_s2", "acel_media_cm_s2",
    "angulo_medio_deg", "vel_angular_media_deg_s",
]
ZONE_SUFFIXES = ["tiempo_total_s", "frames_total", "proporcion_tiempo", "entradas", "lat_primera_entrada_s",
                 "frame_primera_entrada", "dist_media_centro_px", "dist_media_centro_cm", "dist_media_perimetro_px",
                 "dist_media_perimetro_cm"]


def dict_rows(rows):
    return {r[0]: r[1] for r in rows}


class Nada(Reacquirer):
    def find(self, roi_bgr, last_center, last_bbox):
        return None


# ======================================================================
class OpcionesRegistradasPorElPipelineTests(unittest.TestCase):
    def test_valores_por_defecto(self):
        s = TP.run_script([100, 110, 120]).settings
        self.assertEqual(s["suavizado_alpha"], 0.35)
        self.assertEqual(s["salto_max_px"], 120.0)
        self.assertIsNone(s["salto_max_vel_px_s"])
        self.assertEqual((s["recuperacion_tras_frames"], s["recuperacion_distancia_px"], s["adopcion_tras_s"]), (5, 250.0, 2.0))
        self.assertEqual((s["buscador_sujeto"], s["buscador_tras_frames"], s["buscador_cada_frames"]), (None, None, None))
        self.assertEqual(s["modo_seguimiento"], "Scripted")  # un tracker que no es de los tres modos usa su nombre de clase

    def test_sin_filtros_las_opciones_dependientes_quedan_en_none(self):
        s = TP.run_script([100, 110, 120], smoothing_alpha=None, jump_max_px=None).settings
        self.assertIsNone(s["suavizado_alpha"])
        self.assertIsNone(s["salto_max_px"])
        for k in ("recuperacion_tras_frames", "recuperacion_distancia_px", "adopcion_tras_s"):
            self.assertIsNone(s[k], msg=k)  # sin rechazo de saltos no hay recuperación que describir

    def test_el_umbral_efectivo_por_velocidad(self):
        s = TP.run_script([100, 110], jump_max_speed_px_s=5000.0).settings
        self.assertAlmostEqual(s["salto_max_px"], 5000.0 / 30.0)  # velocidad * dt
        self.assertEqual(s["salto_max_vel_px_s"], 5000.0)

    def test_buscador_configurado(self):
        s = TP.run_script([100, 110], reacquirer=Nada(), reacquire_after_frames=4, reacquire_every=7).settings
        self.assertEqual((s["buscador_sujeto"], s["buscador_tras_frames"], s["buscador_cada_frames"]), ("Nada", 4, 7))

    def test_nombres_de_los_tres_modos(self):
        mask = np.full((60, 80), 255, np.uint8)
        self.assertEqual(pipeline._mode_name(BackgroundTracker(mask, np.zeros((60, 80), np.uint8), DEFAULT_PARAMS)), "fondo")
        self.assertEqual(pipeline._mode_name(ColorTracker(mask, {"H_MIN": 50, "H_MAX": 70, "S_MIN": 100, "V_MIN": 100})), "color")
        tr = SilhouetteTracker.from_bbox(np.full((300, 400), 255, np.uint8), TP.BusquedaConSilluetaRealTests.escena()[0], (60, 95, 40, 30))
        self.assertEqual(pipeline._mode_name(tr), "silueta")

    def test_se_registra_si_la_seleccion_es_independiente_del_margen(self):
        frames = TP.BusquedaConSilluetaRealTests.escena()
        for aware in (True, False):
            tr = SilhouetteTracker.from_bbox(np.full((300, 400), 255, np.uint8), frames[0], (60, 95, 40, 30), border_aware=aware)
            job = TrackingJob(TP.BusquedaConSilluetaRealTests.FramesReader(frames), FrameRange(0, 4), compute_timing(30.0), tr, (0, 0, 400, 300))
            s = run_tracking(job, log=None).settings
            self.assertEqual((s["modo_seguimiento"], s["seleccion_independiente_del_margen"]), ("silueta", aware))



# ======================================================================
class CalibracionEnElResumenTests(unittest.TestCase):
    """El resumen dice con qué escala, origen y ROI se midió: sin eso las unidades no se pueden interpretar."""

    def test_el_pipeline_guarda_la_calibracion_usada(self):
        s = TP.run_script([100, 110, 120], cm_per_px=0.05, xy_origin=(10.0, 290.0)).settings
        self.assertEqual(s["escala_cm_por_px"], 0.05)
        self.assertEqual(s["origen_xy_px"], [10.0, 290.0])  # listas: se guardan en JSON y deben salir idénticas
        self.assertEqual(s["roi_box"][:3], [0, 0, 4000])
        import json
        self.assertEqual(json.loads(json.dumps(s)), s)  # ida y vuelta por JSON sin cambios
        self.assertEqual(s["video_tamano_px"][0], 4000)

    def test_filas_con_escala_y_origen(self):
        d = dict_rows(provenance_rows(TP.run_script([100, 110, 120], cm_per_px=0.05, xy_origin=(10.0, 290.0)), "1.0"))
        self.assertEqual(d["calibracion_escala_cm_por_px"], "0.050000")
        self.assertEqual(d["calibracion_escala_px_por_cm"], "20.0000")  # 1 / 0.05
        self.assertEqual(d["calibracion_origen_xy_px"], "10, 290")
        self.assertTrue(d["calibracion_roi_x0_y0_ancho_alto_px"].startswith("0, 0, 4000, "))
        self.assertTrue(d["calibracion_video_ancho_alto_px"].startswith("4000, "))

    def test_la_escala_inversa_es_consistente_con_la_directa(self):
        for cm in (0.0731, 0.5, 1.0, 0.012):
            d = dict_rows(provenance_rows(TP.run_script([100, 110], cm_per_px=cm), "1.0"))
            self.assertAlmostEqual(float(d["calibracion_escala_cm_por_px"]) * float(d["calibracion_escala_px_por_cm"]), 1.0, places=2)

    def test_sin_escala_ni_origen_se_dice_en_lugar_de_dejar_celdas_ambiguas(self):
        d = dict_rows(provenance_rows(TP.run_script([100, 110, 120]), "1.0"))
        self.assertEqual(d["calibracion_escala_cm_por_px"], "sin escala (resultados en píxeles)")
        self.assertEqual(d["calibracion_escala_px_por_cm"], "")
        self.assertEqual(d["calibracion_origen_xy_px"], "no definido")

    def test_resultado_de_una_version_anterior_dice_que_no_se_registro(self):
        res = SimpleNamespace(frames_reacquired=0, records=[], timing=None, settings={"modo_seguimiento": "fondo"})
        d = dict_rows(provenance_rows(res, "1.0"))
        self.assertEqual(d["calibracion"], "no registrada")
        self.assertNotIn("calibracion_escala_cm_por_px", d)

# ======================================================================
class FilasDeProcedenciaTests(unittest.TestCase):
    def resultado(self, **kw):
        res = TP.run_script([100, 110, None, None, None, 120, 130], **kw)
        return res

    def test_orden_y_etiquetas(self):
        etiquetas = [r[0] for r in provenance_rows(self.resultado(), "9.9")]
        self.assertEqual(etiquetas[:19], [
            "trackerlab_version", "proc_modo_seguimiento", "proc_suavizado_alpha", "proc_salto_max_px", "proc_salto_max_vel_px_s",
            "proc_recuperacion_tras_frames", "proc_recuperacion_distancia_px", "proc_adopcion_tras_s", "proc_busqueda_sujeto",
            "proc_busqueda_tras_frames", "proc_busqueda_cada_frames", "proc_seleccion_independiente_del_margen",
            "calibracion_escala_cm_por_px", "calibracion_escala_px_por_cm", "calibracion_origen_xy_px",
            "calibracion_roi_x0_y0_ancho_alto_px", "calibracion_video_ancho_alto_px",
            "proc_frames_reencontrados", "calidad_veredicto"])
        calidad = [e for e in etiquetas if e.startswith("calidad_")]
        self.assertEqual(etiquetas[19:19 + len(calidad) - 1], calidad[1:])
        # la asistencia del detector va AL FINAL: se añade sin reordenar ninguna fila anterior
        self.assertEqual(etiquetas[19 + len(calidad) - 1:], ["proc_asistencia_detector", "proc_sujeto_elegido_por",
                                                             "proc_sujeto_confianza_detector", "proc_detector_modelo"])

    def test_asistencia_sin_detector(self):
        d = dict_rows(provenance_rows(self.resultado(), "9.9"))
        self.assertEqual((d["proc_asistencia_detector"], d["proc_sujeto_elegido_por"], d["proc_sujeto_confianza_detector"], d["proc_detector_modelo"]),
                         ("ninguna", "a mano", "", ""))

    def test_asistencia_con_el_sujeto_propuesto_por_el_detector(self):
        a = {"modo": "proponer_sujeto", "sujeto": {"metodo": "detector", "confianza": 0.86, "modelo": "rfdetr-small.onnx"}}
        d = dict_rows(provenance_rows(self.resultado(assistance=a), "9.9"))
        self.assertEqual((d["proc_asistencia_detector"], d["proc_sujeto_elegido_por"], d["proc_sujeto_confianza_detector"], d["proc_detector_modelo"]),
                         ("proponer_sujeto", "detector", "0.8600", "rfdetr-small.onnx"))

    def test_asistencia_con_la_propuesta_rechazada_o_sin_deteccion(self):
        a = {"modo": "proponer_sujeto", "sujeto": {"metodo": "manual", "confianza": 0.62, "modelo": "rfdetr-small.onnx"}}
        d = dict_rows(provenance_rows(self.resultado(assistance=a), "9.9"))
        self.assertEqual((d["proc_sujeto_elegido_por"], d["proc_sujeto_confianza_detector"]), ("a mano", "0.6200"))  # consta lo que proponía
        a = {"modo": "proponer_sujeto", "sujeto": {"metodo": "manual", "confianza": None, "modelo": "rfdetr-small.onnx"}}
        self.assertEqual(dict_rows(provenance_rows(self.resultado(assistance=a), "9.9"))["proc_sujeto_confianza_detector"], "")
        a = {"modo": "proponer_sujeto", "sujeto": {"metodo": "detector", "confianza": True}}  # un booleano no es una confianza
        self.assertEqual(dict_rows(provenance_rows(self.resultado(assistance=a), "9.9"))["proc_sujeto_confianza_detector"], "")

    def test_resultados_anteriores_a_la_constancia(self):
        import types
        res = self.resultado()
        viejo = types.SimpleNamespace(settings={k: v for k, v in res.settings.items() if k != "asistencia_detector"}, frames_reacquired=0,
                                      records=res.records, timing=res.timing)
        etiquetas = [r[0] for r in provenance_rows(viejo, "9.9")]
        self.assertEqual(etiquetas[-1], "proc_asistencia_detector")
        self.assertEqual(dict_rows(provenance_rows(viejo, "9.9"))["proc_asistencia_detector"], "no registrada")
        self.assertEqual(sum(1 for e in etiquetas if e.startswith(("proc_sujeto", "proc_detector"))), 0)
        sin_nada = types.SimpleNamespace(settings={}, frames_reacquired=0, records=None, timing=res.timing)
        self.assertFalse(any("asistencia" in r[0] for r in provenance_rows(sin_nada, "9.9")))  # sin opciones registradas, no se inventa nada

    def test_valores_y_formato(self):
        d = dict_rows(provenance_rows(self.resultado(), "9.9"))
        self.assertEqual(d["trackerlab_version"], "9.9")
        self.assertEqual(d["proc_suavizado_alpha"], "0.3500")
        self.assertEqual(d["proc_salto_max_px"], "120.00")
        self.assertEqual(d["proc_salto_max_vel_px_s"], "")  # no aplica: celda vacía, no un número inventado
        self.assertEqual(d["proc_recuperacion_tras_frames"], 5)
        self.assertEqual(d["proc_recuperacion_distancia_px"], "250.00")
        self.assertEqual(d["proc_adopcion_tras_s"], "2.00")
        self.assertEqual(d["proc_busqueda_sujeto"], "no")
        self.assertEqual((d["proc_busqueda_tras_frames"], d["proc_busqueda_cada_frames"]), ("", ""))
        self.assertEqual(d["proc_seleccion_independiente_del_margen"], "")  # un tracker sin esa opción
        self.assertEqual(d["proc_frames_reencontrados"], 0)

    def test_sin_filtros_las_celdas_quedan_vacias(self):
        d = dict_rows(provenance_rows(self.resultado(smoothing_alpha=None, jump_max_px=None), "1.0"))
        for k in ("proc_suavizado_alpha", "proc_salto_max_px", "proc_recuperacion_tras_frames", "proc_recuperacion_distancia_px",
                  "proc_adopcion_tras_s"):
            self.assertEqual(d[k], "", msg=k)

    def test_busqueda_activada_y_conteo_de_reencontrados(self):
        res = TP.run_script([100, 110] + [None] * 3, reacquirer=TP.FakeReacquirer([(300, 100)]))
        d = dict_rows(provenance_rows(res, "1.0"))
        self.assertEqual((d["proc_busqueda_sujeto"], d["proc_busqueda_tras_frames"], d["proc_busqueda_cada_frames"]), ("si", 3, 5))
        self.assertEqual(d["proc_frames_reencontrados"], 1)
        self.assertEqual(d["calidad_frames_recuperados"], 1)

    def test_seleccion_independiente_del_margen_se_escribe_si_o_no(self):
        for valor, esperado in ((True, "si"), (False, "no")):
            res = SimpleNamespace(settings={"seleccion_independiente_del_margen": valor}, frames_reacquired=0, records=[], timing=None)
            self.assertEqual(dict_rows(provenance_rows(res, "1.0"))["proc_seleccion_independiente_del_margen"], esperado)

    def test_incluye_las_filas_de_calidad_del_informe(self):
        res = self.resultado()
        d = dict_rows(provenance_rows(res, "1.0"))
        self.assertEqual(d["calidad_frames_totales"], 7)
        self.assertEqual(d["calidad_frames_perdidos"], 3)
        self.assertEqual(d["calidad_veredicto"], "deficiente")  # 3 de 7 frames perdidos

    def test_sin_registros_no_se_inventa_una_calidad(self):
        res = SimpleNamespace(settings={"modo_seguimiento": "fondo"}, frames_reacquired=0, records=[], timing=None)
        d = dict_rows(provenance_rows(res, "1.0"))
        self.assertIn("no calculado", d["calidad_veredicto"])
        self.assertNotIn("calidad_frames_totales", d)

    def test_resultado_de_una_version_anterior_sin_opciones(self):
        res = SimpleNamespace(frames_reacquired=0, records=[], timing=None)  # sin atributo settings
        d = dict_rows(provenance_rows(res, "1.0"))
        self.assertEqual(d["proc_opciones"], "no registradas")
        self.assertNotIn("proc_suavizado_alpha", d)

    def test_etiquetas_unicas_con_prefijo_y_sin_chocar_con_el_resumen(self):
        etiquetas = [r[0] for r in provenance_rows(self.resultado(), "1.0")]
        self.assertEqual(len(etiquetas), len(set(etiquetas)))
        for e in etiquetas:
            self.assertTrue(e.startswith(("trackerlab_", "proc_", "calibracion_", "calidad_")), msg=e)
        self.assertFalse(set(etiquetas) & set(BASE_LABELS))
        for zona in ("proc", "calidad", "calibracion", "trackerlab", "A", "zona_1"):  # ni siquiera con zonas de nombres raros
            self.assertFalse(set(etiquetas) & {f"{zona}_{suf}" for suf in ZONE_SUFFIXES}, msg=zona)

    def test_solo_agrega_no_modifica(self):
        base = [["experimento", "x"], ["frames_procesados", 7]]
        copia = [list(r) for r in base]
        base.extend(provenance_rows(self.resultado(), "1.0"))
        self.assertEqual(base[:2], copia)
        self.assertGreater(len(base), 2)


# ======================================================================
class IntegracionConExportTrackingCsvTests(unittest.TestCase):
    """Ejecuta la export_tracking_csv REAL de main.py (extraída del archivo) sobre un resultado real."""

    @classmethod
    def setUpClass(cls):
        main_py = os.path.join(ROOT, "main.py")
        csv_py = os.path.join(ROOT, "persistence", "csv_export.py")
        if not (os.path.exists(main_py) and os.path.exists(csv_py)):
            raise unittest.SkipTest("no se encontró main.py o persistence/csv_export.py en la raíz del proyecto")
        cls.tmp = tempfile.TemporaryDirectory()
        src = open(main_py, encoding="utf-8").read()
        tree = ast.parse(src)
        fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "export_tracking_csv"), None)
        if fn is None:
            raise unittest.SkipTest("main.py no define export_tracking_csv")
        csrc = open(csv_py, encoding="utf-8").read()
        aux = {n.name: ast.get_source_segment(csrc, n) for n in ast.parse(csrc).body
               if isinstance(n, ast.FunctionDef) and n.name in ("safe_float_str", "write_csv")}
        ns = {"os": os, "csv": csv, "ensure_dir": lambda d: os.makedirs(d, exist_ok=True)}
        exec("\n\n".join(aux.values()), ns)
        ns["make_output_csv"] = lambda base, exp: os.path.join(cls.tmp.name, f"{base}_{exp}_FRAMES.csv")
        ns["make_summary_csv"] = lambda base, exp: os.path.join(cls.tmp.name, f"{base}_{exp}_SUMMARY.csv")
        ns["provenance_rows"] = provenance_rows
        ns["APP_VERSION"] = "1.0"
        usados = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        propios = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} | {a.arg for a in ast.walk(fn) if isinstance(a, ast.arg)}
        faltan = sorted(usados - propios - set(dir(builtins)) - set(ns))
        if faltan:
            raise unittest.SkipTest(f"export_tracking_csv usa nombres que esta prueba no provee: {faltan}")
        exec(ast.get_source_segment(src, fn), ns)
        cls.export = staticmethod(ns["export_tracking_csv"])
        cls.usa_provenance = "provenance_rows" in usados

        frames, _ = BG.make_frames(90, lost={30, 31, 32})
        path = os.path.join(cls.tmp.name, "v.avi")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (BG.FW, BG.FH))
        for f in frames:
            vw.write(f)
        vw.release()
        mask = BG.roi_mask("elipse")
        zm = np.zeros((BG.FH, BG.FW), np.uint8)
        cv2.rectangle(zm, (60, 60), (160, 170), 255, -1)
        with VideoReader(path) as r:
            bg = compute_background(read_background_frames(r, 35), BG.ROI, mask, False)
            job = TrackingJob(r, FrameRange(0, 85), compute_timing(r.fps), BackgroundTracker(mask, bg, DEFAULT_PARAMS), BG.ROI,
                              zones=[ZoneGeometry("zona_A", zm, (110.0, 115.0))], cm_per_px=0.07, xy_origin=(50.0, 200.0),
                              start=((40.0, 90.0), (28, 78, 24, 24)))
            cls.result = run_tracking(job, keep_records=True, log=None)
        frames_csv, summary_csv = cls.export(cls.result, "exp", "track")
        with open(summary_csv, newline="", encoding="utf-8-sig") as f:
            cls.rows = list(csv.reader(f))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_main_ya_usa_las_filas_de_procedencia(self):
        self.assertTrue(self.usa_provenance, "export_tracking_csv no llama a provenance_rows: falta el cambio en main.py")

    def test_las_filas_base_siguen_intactas_y_en_su_orden(self):
        self.assertEqual(self.rows[0], ["metrica", "valor"])
        self.assertEqual([r[0] for r in self.rows[1:1 + len(BASE_LABELS)]], BASE_LABELS)

    def test_las_filas_de_zona_siguen_a_las_base(self):
        zona = [r[0] for r in self.rows[1 + len(BASE_LABELS):1 + len(BASE_LABELS) + len(ZONE_SUFFIXES)]]
        self.assertEqual(zona, [f"zona_A_{s}" for s in ZONE_SUFFIXES])

    def test_las_filas_nuevas_van_al_final_y_son_las_de_provenance_rows(self):
        esperadas = [[str(a), str(b)] for a, b in provenance_rows(self.result, "1.0")]
        self.assertEqual(self.rows[-len(esperadas):], esperadas)
        self.assertEqual(len(self.rows), 1 + len(BASE_LABELS) + len(ZONE_SUFFIXES) + len(esperadas))  # nada más, nada menos

    def test_ninguna_etiqueta_se_repite(self):
        etiquetas = [r[0] for r in self.rows[1:]]
        self.assertEqual(len(etiquetas), len(set(etiquetas)))


if __name__ == "__main__":
    unittest.main()