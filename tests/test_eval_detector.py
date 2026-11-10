"""Pruebas de tools/eval_detector.py (evalúa un detector ONNX contra frames etiquetados a mano).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_eval_detector.py -v

El «modelo» es un detector de juguete que LEE EL TENSOR de entrada, encuentra la mancha más brillante (el círculo blanco del video
sintético) y devuelve su caja SIN decodificar, igual que el ONNX real. Así se prueba de verdad el recorte, el redimensionado y la
conversión de coordenadas. Lo que depende de RF-DETR real solo se prueba con el modelo de Colab.
"""

import csv
import importlib
import json
import os
import sys
import types
import unittest
from unittest import mock

import cv2
import numpy as np

from tools import eval_detector as ED


def _imp(nombre):
    try:
        return importlib.import_module(f"tests.{nombre}")
    except ImportError:
        return importlib.import_module(nombre)


TM = _imp("test_build_manual")
TE, BG = TM.TE, TM.BG


class _IO:
    def __init__(self, name, shape=None):
        self.name, self.shape = name, shape


class SesionBrillo:
    """Detector de juguete: la caja de la mancha más brillante del tensor de entrada (sin decodificar, como el ONNX real)."""

    def __init__(self, hw=(96, 96), desvio=(0.0, 0.0), logit=4.0, notas=None, forma=None, tiempo=None):
        self.hw, self.desvio, self.logit, self.notas, self.forma = hw, desvio, logit, notas, forma
        self.llamadas = 0

    def get_inputs(self):
        return [_IO("input", self.forma if self.forma is not None else [1, 3, self.hw[0], self.hw[1]])]

    def get_outputs(self):
        return [_IO("dets"), _IO("labels")]

    def get_modelmeta(self):
        return types.SimpleNamespace(custom_metadata_map={} if self.notas is None else {"rfdetr_notes": json.dumps(self.notas)})

    def run(self, nombres, feed):
        self.llamadas += 1
        r = feed["input"][0, 0]  # canal R normalizado: el blanco vale ~2.25 y el fondo oscuro, negativo
        h, w = r.shape
        ys, xs = np.nonzero(r > 1.0)
        if len(xs) == 0:
            return [np.array([[[0.5, 0.5, 0.1, 0.1]]]), np.array([[[-4.0, 4.0]]])]
        cx, cy = (xs.min() + xs.max() + 1) / 2 / w + self.desvio[0], (ys.min() + ys.max() + 1) / 2 / h + self.desvio[1]
        return [np.array([[[cx, cy, (xs.max() - xs.min() + 1) / w, (ys.max() - ys.min() + 1) / h]]]), np.array([[[self.logit, -self.logit]]])]


class Base(TM.Base):
    def setUp(self):
        super().setUp()
        self.modelo = os.path.join(self.w, "small.onnx")
        with open(self.modelo, "w") as fh:
            fh.write("no es un modelo de verdad: la sesión es de juguete")

    def etiquetado(self, claves=("A",), por_video=100):
        """Frames de la escena sintética, etiquetados con la posición REAL del círculo (vacío si el círculo no está)."""
        val, vt = self.conjunto(claves, por_video=por_video, etiquetar=False)
        _, truth = BG.make_frames(90, lost={30, 31, 32})
        for v in vt:
            if v["nombre"] == "bg.avi":
                cajas = {f: (None if truth[f] is None else TE.caja_circulo(*truth[f], 13)) for f in v["frames"]}
            else:
                cajas = {f: TM.caja(f) for f in v["frames"]}
            TE.etiquetar(val, v["id"], v["ancho"], v["alto"], cajas)
        return val, vt, truth

    def correr(self, val, sesion=None, modelos=None, entradas=None, **kw):
        msgs = []
        sesion = sesion or SesionBrillo()
        kw.setdefault("session_factory", lambda ruta, hilos: sesion)
        rc = ED.run(val, modelos if modelos is not None else [self.modelo], [self.entradas["A"]] if entradas is None else entradas, out=msgs.append, **kw)
        return rc, "\n".join(msgs)

    def resultados(self, salida):
        with open(os.path.join(salida, "resultados.json"), encoding="utf-8") as fh:
            return json.load(fh)


# ======================================================================
class EvaluacionDePuntaAPuntaTests(Base):
    def test_el_recorte_el_redimensionado_y_las_coordenadas_son_correctos(self):
        val, vt, truth = self.etiquetado()
        salida = os.path.join(self.w, "sal")
        rc, texto = self.correr(val, salida=salida)
        self.assertEqual(rc, 0, msg=texto)
        r = self.resultados(salida)["modelos"][0]["resumen"]
        sin = sum(1 for f in vt[0]["frames"] if truth[f] is None)
        self.assertGreater(sin, 0)  # hay frames sin círculo: el detector no debe inventarlo
        self.assertEqual((r["evaluados"], r["con_raton"], r["sin_raton"]), (len(vt[0]["frames"]), len(vt[0]["frames"]) - sin, sin))
        self.assertEqual((r["perdidos"], r["lugar_equivocado"], r["falsos_positivos"]), (0, 0, 0))
        self.assertEqual(r["acierto_pct"], 100.0)
        self.assertLess(r["error_centro_mediana_px"], 2.0)  # el centro cae sobre el círculo, en coordenadas del frame completo
        self.assertLess(r["error_centro_max_px"], 4.0)
        self.assertEqual(r["posicion_dentro_pct"], 100.0)

    def test_con_margen_y_sin_recorte_tambien_cae_sobre_el_circulo(self):
        val, _, _ = self.etiquetado()
        for kw in ({"margen_roi": 10}, {"margen_roi": 30}, {"sin_recorte": True}):
            salida = os.path.join(self.w, "s" + "".join(str(v) for v in kw.values()))
            rc, texto = self.correr(val, salida=salida, **kw)
            self.assertEqual(rc, 0, msg=texto)
            r = self.resultados(salida)["modelos"][0]["resumen"]
            self.assertEqual((r["perdidos"], r["falsos_positivos"]), (0, 0), msg=str(kw))
            self.assertGreaterEqual(r["acierto_pct"], 95.0, msg=str(kw))
            self.assertLess(r["error_centro_mediana_px"], 2.5, msg=str(kw))

    def test_una_deteccion_desviada_es_mal_ubicada_y_se_ve_en_la_hoja_de_fallos(self):
        val, _, _ = self.etiquetado()
        salida = os.path.join(self.w, "sal")
        rc, texto = self.correr(val, SesionBrillo(desvio=(0.25, 0.0)), salida=salida)  # corrida de un cuarto del recorte
        self.assertEqual(rc, 0)
        r = self.resultados(salida)["modelos"][0]["resumen"]
        self.assertEqual(r["acierto_pct"], 0.0)
        self.assertGreater(r["lugar_equivocado"], 30)
        # con el desvío, alguna caja sale por completo del recorte y queda recortada a ancho cero: cuenta como perdida, no como mal ubicada
        self.assertLessEqual(r["perdidos"], 3)
        self.assertEqual(r["lugar_equivocado"] + r["perdidos"], r["con_raton"])
        self.assertGreater(r["error_centro_mediana_px"], 40)  # un cuarto de ~260 px
        self.assertTrue(os.path.exists(os.path.join(salida, "fallos.png")))
        self.assertIn("Peores casos", texto)

    def test_una_deteccion_fuera_del_recorte_es_un_frame_perdido(self):
        val, _, _ = self.etiquetado()
        salida = os.path.join(self.w, "sal")
        self.correr(val, SesionBrillo(desvio=(2.0, 0.0)), salida=salida)
        r = self.resultados(salida)["modelos"][0]["resumen"]
        self.assertEqual((r["perdidos"], r["lugar_equivocado"]), (r["con_raton"], 0))

    def test_el_umbral_de_confianza(self):
        val, _, _ = self.etiquetado()
        _, t1 = self.correr(val, SesionBrillo(logit=0.0), salida=os.path.join(self.w, "a"))  # confianza 0.5 exacta: no es > 0.5
        r = self.resultados(os.path.join(self.w, "a"))["modelos"][0]["resumen"]
        self.assertEqual(r["perdidos"], r["con_raton"])
        self.correr(val, SesionBrillo(logit=0.0), salida=os.path.join(self.w, "b"), umbral=0.4)
        self.assertEqual(self.resultados(os.path.join(self.w, "b"))["modelos"][0]["resumen"]["perdidos"], 0)

    def test_sin_fallos_no_hay_hoja(self):
        val, _, _ = self.etiquetado()
        salida = os.path.join(self.w, "sal")
        _, texto = self.correr(val, salida=salida)
        self.assertFalse(os.path.exists(os.path.join(salida, "fallos.png")))
        self.assertNotIn("Peores casos", texto)

    def test_hoja_de_fallos_desactivada(self):
        val, _, _ = self.etiquetado()
        salida = os.path.join(self.w, "sal")
        self.correr(val, SesionBrillo(desvio=(0.25, 0.0)), salida=salida, hoja_fallos=0)
        self.assertFalse(os.path.exists(os.path.join(salida, "fallos.png")))

    def test_un_frame_sin_ratón_con_deteccion_es_falso_positivo(self):
        """Un círculo claro en un frame etiquetado «sin ratón»: el detector lo marca, y eso debe contar como falso positivo."""
        val, vt, truth = self.etiquetado()
        v = vt[0]
        f = next(f for f in v["frames"] if truth[f] is not None)
        TE.etiquetar(val, v["id"], v["ancho"], v["alto"], {f: None})  # se declara que ahí no hay ratón, aunque sí hay círculo
        salida = os.path.join(self.w, "sal")
        self.correr(val, salida=salida)
        r = self.resultados(salida)["modelos"][0]["resumen"]
        self.assertEqual(r["falsos_positivos"], 1)


# ======================================================================
class MargenDelRoiTests(Base):
    def margenes_usados(self, val, sesion, **kw):
        with mock.patch.object(ED, "crop_rect", wraps=ED.crop_rect) as espia:
            rc, texto = self.correr(val, sesion, **kw)
        return rc, texto, {c.args[1] for c in espia.call_args_list}

    def test_se_lee_de_las_notas_del_modelo(self):
        val, _, _ = self.etiquetado()
        rc, texto, margenes = self.margenes_usados(val, SesionBrillo(notas={"margen_roi_px": 10, "recorte_roi": True}))
        self.assertEqual((rc, margenes), (0, {10}))
        self.assertIn("margen del ROI 10 px (leído de las notas del modelo)", texto)
        self.assertIn('"margen_roi_px": 10', texto)  # y se muestran las notas: de dónde viene el modelo

    def test_lo_que_se_pasa_manda_sobre_las_notas(self):
        val, _, _ = self.etiquetado()
        _, texto, margenes = self.margenes_usados(val, SesionBrillo(notas={"margen_roi_px": 10}), margen_roi=0)
        self.assertEqual(margenes, {0})
        self.assertIn("margen del ROI 0 px (indicado con --margen-roi)", texto)

    def test_sin_notas_es_cero(self):
        val, _, _ = self.etiquetado()
        _, texto, margenes = self.margenes_usados(val, SesionBrillo())
        self.assertEqual(margenes, {0})
        self.assertIn("(por omisión (el modelo no trae notas))", texto)

    def test_notas_ilegibles_o_con_un_margen_que_no_es_un_numero_se_ignoran(self):
        val, _, _ = self.etiquetado()
        for notas in ({"margen_roi_px": "diez"}, {"margen_roi_px": None}, {"otra_cosa": 1}):
            _, _, margenes = self.margenes_usados(val, SesionBrillo(notas=notas))
            self.assertEqual(margenes, {0}, msg=str(notas))

    def test_sin_recorte_no_hay_margen(self):
        val, _, _ = self.etiquetado()
        _, texto, margenes = self.margenes_usados(val, SesionBrillo(notas={"margen_roi_px": 10}), sin_recorte=True, entradas=[])
        self.assertEqual(margenes, set())
        self.assertIn("margen del ROI 0 px (no aplica (frame completo))", texto)

    def test_aviso_si_el_modelo_se_entreno_con_frames_completos(self):
        val, _, _ = self.etiquetado()
        _, texto = self.correr(val, SesionBrillo(notas={"recorte_roi": False}))
        self.assertIn("se entrenó con frames completos", texto)
        _, texto = self.correr(val, SesionBrillo(notas={"recorte_roi": False}), sin_recorte=True, entradas=[])
        self.assertNotIn("se entrenó con frames completos", texto)  # si ya pides frame completo, no hay nada que avisar
        _, texto = self.correr(val, SesionBrillo(notas={"recorte_roi": True}))
        self.assertNotIn("se entrenó con frames completos", texto)

    def test_read_notes(self):
        def con(meta):
            s = types.SimpleNamespace(get_modelmeta=lambda: types.SimpleNamespace(custom_metadata_map=meta))
            return ED.read_notes(s)
        self.assertEqual(con({"rfdetr_notes": json.dumps({"a": 1})}), {"a": 1})
        self.assertEqual(con({"rfdetr_notes": "{ no es json"}), {})
        self.assertEqual(con({"rfdetr_notes": json.dumps([1, 2])}), {})  # JSON válido pero no un diccionario
        self.assertEqual(con({}), {})
        self.assertEqual(con(None), {})
        self.assertEqual(con({"otra": "x"}), {})

        def explota():
            raise RuntimeError("sin metadatos")
        self.assertEqual(ED.read_notes(types.SimpleNamespace(get_modelmeta=explota)), {})
        self.assertEqual(ED.read_notes(object()), {})  # ni siquiera tiene el método


# ======================================================================
class TiempoTests(Base):
    def test_estadisticas_descartando_las_primeras(self):
        s = ED.timing_stats([100.0, 90.0, 10.0, 20.0, 30.0])  # sin las dos primeras: 10, 20, 30
        self.assertEqual((s["n"], s["mediana_ms"], s["max_ms"]), (3, 20.0, 30.0))
        self.assertAlmostEqual(s["p90_ms"], 28.0)  # posición 0.9 * 2 = 1.8 -> 20 + (30 - 20) * 0.8
        self.assertEqual(ED.timing_stats([5.0, 7.0, 9.0], descartar=0)["n"], 3)

    def test_si_quedan_menos_de_tres_se_usan_todas(self):
        s = ED.timing_stats([100.0, 90.0, 10.0, 20.0])  # al descartar 2 quedan 2: se usan las 4
        self.assertEqual((s["n"], s["mediana_ms"]), (4, 55.0))
        self.assertEqual(ED.timing_stats([42.0])["mediana_ms"], 42.0)

    def test_sin_datos(self):
        self.assertEqual(ED.timing_stats([]), {"n": 0, "mediana_ms": None, "p90_ms": None, "max_ms": None})

    def test_la_corrida_informa_el_tiempo(self):
        val, vt, _ = self.etiquetado()
        salida = os.path.join(self.w, "sal")
        _, texto = self.correr(val, salida=salida, hilos=3)
        t = self.resultados(salida)["modelos"][0]["tiempos"]
        self.assertEqual(t["n"], len(vt[0]["frames"]) - ED.WARMUP)  # sin los de calentamiento
        self.assertGreater(t["mediana_ms"], 0)
        self.assertGreaterEqual(t["p90_ms"], t["mediana_ms"])
        self.assertIn("Tiempo por imagen de small: mediana", texto)
        self.assertIn("con 3 hilos", texto)
        self.assertIn("ESTA máquina", texto)
        self.assertEqual(self.resultados(salida)["hilos"], 3)

    def test_sin_hilos_dice_los_de_omision(self):
        val, _, _ = self.etiquetado()
        _, texto = self.correr(val)
        self.assertIn("con los hilos por omisión", texto)

    def test_el_cronometro_incluye_recortar_y_convertir_el_color_ademas_de_detectar(self):
        """Un reloj que avanza 3 ms dentro de la conversión de color y 2 ms dentro de la detección: el tiempo medido debe ser 5 ms."""
        reloj = {"t": 0.0}
        real_cvt, real_detect = cv2.cvtColor, ED.detect

        def cvt(*a, **k):
            reloj["t"] += 0.003
            return real_cvt(*a, **k)

        def det(*a, **k):
            reloj["t"] += 0.002
            return real_detect(*a, **k)
        pred = None
        with mock.patch.object(ED.time, "perf_counter", side_effect=lambda: reloj["t"]), mock.patch.object(ED.cv2, "cvtColor", cvt), mock.patch.object(ED, "detect", det):
            pred, ms = ED.predict_image(SesionBrillo(), np.zeros((50, 80, 3), np.uint8), (0, 0, 40, 40), 0.5)
        self.assertAlmostEqual(ms, 5.0, places=6)

    def test_el_tiempo_mide_el_proceso_completo(self):
        """Debe incluir recortar, preparar y decodificar, no solo la inferencia: se comprueba con un reloj controlado."""
        val, vt, _ = self.etiquetado(por_video=6)
        reloj = iter(range(0, 10000, 5))
        with mock.patch.object(ED.time, "perf_counter", side_effect=lambda: next(reloj) / 1000.0):
            salida = os.path.join(self.w, "sal")
            self.correr(val, salida=salida)
        self.assertAlmostEqual(self.resultados(salida)["modelos"][0]["tiempos"]["mediana_ms"], 5.0, places=6)  # entre las dos lecturas del reloj: 5 ms


# ======================================================================
class SesionDos(SesionBrillo):
    """Devuelve DOS detecciones: la primera (por posición) con menos confianza que la segunda."""

    def run(self, nombres, feed):
        return [np.array([[[0.25, 0.25, 0.1, 0.1], [0.75, 0.75, 0.1, 0.1]]]), np.array([[[1.0, -1.0], [4.0, -4.0]]])]


class PrediccionTests(unittest.TestCase):
    def test_el_color_se_pasa_de_BGR_a_RGB(self):
        """OpenCV lee en BGR y el modelo espera RGB. Un cuadro ROJO puro solo se ve si se convierte: el detector de juguete mira el canal R."""
        img = np.zeros((100, 200, 3), np.uint8)
        img[40:60, 100:130] = (0, 0, 255)  # BGR: rojo
        pred, _ = ED.predict_image(SesionBrillo(hw=(100, 200)), img, None, 0.5)
        self.assertTrue(pred.found)
        self.assertLess(abs(pred.box[0] - 100), 3)
        azul = np.zeros((100, 200, 3), np.uint8)
        azul[40:60, 100:130] = (255, 0, 0)  # BGR: azul; en RGB no tiene canal R y el detector no debe verlo
        self.assertFalse(ED.predict_image(SesionBrillo(hw=(100, 200)), azul, None, 0.5)[0].found)

    def test_con_varias_detecciones_se_toma_la_de_mayor_confianza(self):
        pred, _ = ED.predict_image(SesionDos(hw=(100, 200)), np.zeros((100, 200, 3), np.uint8), None, 0.5)
        self.assertTrue(pred.found)
        # la de confianza 4.0 está en (0.75, 0.75): centro en x = 150, y = 75
        self.assertLess(abs(pred.pos[0] - 150), 1.5)
        self.assertLess(abs(pred.pos[1] - 75), 1.5)
        # con un umbral alto solo pasa la segunda; con uno bajo pasan las dos y sigue ganando la misma
        self.assertLess(abs(ED.predict_image(SesionDos(hw=(100, 200)), np.zeros((100, 200, 3), np.uint8), None, 0.1)[0].pos[0] - 150), 1.5)

    def test_una_caja_totalmente_fuera_del_recorte_no_cuenta_como_deteccion(self):
        """Tras recortarla a la imagen queda sin ancho: no es una detección válida."""
        pred, _ = ED.predict_image(SesionBrillo(hw=(100, 200), desvio=(2.0, 0.0)), np.full((100, 200, 3), 255, np.uint8), None, 0.5)
        self.assertFalse(pred.found)
        self.assertIsNone(pred.box)
        pred, _ = ED.predict_image(SesionBrillo(hw=(100, 200), desvio=(0.0, -2.0)), np.full((100, 200, 3), 255, np.uint8), None, 0.5)
        self.assertFalse(pred.found)


    def test_la_caja_se_devuelve_en_coordenadas_del_frame_completo(self):
        img = np.zeros((100, 200, 3), np.uint8)
        img[40:60, 100:130] = 255  # cuadro blanco en el frame: x 100..130, y 40..60
        rect = (50, 20, 150, 80)  # recorte x 50..150, y 20..80 (100 x 60)
        pred, ms = ED.predict_image(SesionBrillo(hw=(60, 100)), img, rect, 0.5)
        self.assertTrue(pred.found)
        self.assertGreater(ms, 0)
        x, y, w, h = pred.box
        self.assertLess(abs(x - 100), 2)
        self.assertLess(abs(y - 40), 2)
        self.assertLess(abs(w - 30), 2.5)
        self.assertLess(abs(h - 20), 2.5)
        self.assertEqual(pred.pos, (x + w / 2, y + h / 2))  # la posición es el centro de la caja

    def test_lo_que_queda_fuera_del_recorte_se_ignora(self):
        """Dos cuadros blancos: uno dentro del recorte y otro fuera. El detector de juguete devolvería la unión de TODO lo brillante que vea:
        si la caja es solo la del cuadro de adentro, es que de verdad se recortó antes de detectar."""
        img = np.zeros((100, 200, 3), np.uint8)
        img[40:60, 100:130] = 255  # dentro del recorte (x 50..150, y 20..80)
        img[5:15, 170:195] = 255  # fuera
        pred, _ = ED.predict_image(SesionBrillo(hw=(60, 100)), img, (50, 20, 150, 80), 0.5)
        self.assertTrue(pred.found)
        self.assertLess(abs(pred.box[0] - 100), 2)
        self.assertLess(abs(pred.box[2] - 30), 2.5)  # 30 px de ancho, no la unión hasta x = 195
        sin_recorte, _ = ED.predict_image(SesionBrillo(hw=(100, 200)), img, None, 0.5)
        self.assertGreater(sin_recorte.box[2], 80)  # y sin recorte sí ve los dos (la unión)

    def test_sin_recorte_y_sin_nada_que_detectar(self):
        pred, _ = ED.predict_image(SesionBrillo(), np.zeros((50, 80, 3), np.uint8), None, 0.5)
        self.assertFalse(pred.found)
        self.assertIsNone(pred.box)


# ======================================================================
class CajasTests(Base):
    def fila(self, etiqueta, pred, iou=None, tiene=True):
        return {"tiene_raton": tiene, "caja_etiqueta": etiqueta, "caja_prediccion": pred, "iou": iou}

    def test_razones_e_iou_a_mano(self):
        filas = [self.fila((100, 100, 50, 40), (95, 100, 60, 40), 0.8), self.fila((200, 200, 60, 60), (200, 200, 60, 66), 0.6)]
        b = ED.box_stats(filas)
        self.assertEqual(b["con_caja"], 2)
        self.assertAlmostEqual(b["ancho_razon_mediana"], 1.1)  # (60/50 = 1.2 y 60/60 = 1.0) -> mediana 1.1
        self.assertAlmostEqual(b["alto_razon_mediana"], 1.05)  # (40/40 = 1.0 y 66/60 = 1.1)
        self.assertAlmostEqual(b["iou_mediano"], 0.7)
        self.assertAlmostEqual(b["iou_p10"], 0.62)  # percentil 10 de [0.6, 0.8] con interpolación lineal

    def test_es_la_mediana_y_no_la_media(self):
        """Con tres valores y uno extremo, la media (1.33) y la mediana (1.0) se separan: un solo frame raro no debe mover el resumen."""
        filas = [self.fila((0, 0, 50, 50), (0, 0, 50, 50), 0.9), self.fila((0, 0, 50, 50), (0, 0, 50, 50), 0.9), self.fila((0, 0, 50, 50), (0, 0, 100, 100), 0.1)]
        b = ED.box_stats(filas)
        self.assertEqual((b["ancho_razon_mediana"], b["alto_razon_mediana"]), (1.0, 1.0))
        self.assertAlmostEqual(b["iou_mediano"], 0.9)

    def test_la_razon_es_del_detector_sobre_la_etiqueta(self):
        b = ED.box_stats([self.fila((0, 0, 100, 50), (0, 0, 50, 100), 0.3)])  # el detector es la mitad de ancho y el doble de alto
        self.assertEqual((b["ancho_razon_mediana"], b["alto_razon_mediana"]), (0.5, 2.0))

    def test_los_frames_sin_ratón_o_sin_deteccion_no_cuentan(self):
        filas = [self.fila(None, (1, 1, 5, 5), 0.99, tiene=False),  # falso positivo: no hay etiqueta con qué comparar (aunque traiga un IoU, no debe contar)
                 self.fila((10, 10, 40, 40), None, None),  # frame perdido
                 self.fila((10, 10, 40, 40), (10, 10, 20, 20), 0.25)]
        b = ED.box_stats(filas)
        self.assertEqual(b["con_caja"], 1)
        self.assertEqual((b["ancho_razon_mediana"], b["iou_mediano"]), (0.5, 0.25))

    def test_sin_datos(self):
        self.assertEqual(ED.box_stats([]), {"con_caja": 0, "iou_mediano": None, "iou_p10": None, "ancho_razon_mediana": None, "alto_razon_mediana": None})

    def test_la_corrida_los_imprime_y_los_guarda(self):
        val, _, _ = self.etiquetado()
        salida = os.path.join(self.w, "sal")
        _, texto = self.correr(val, salida=salida)
        self.assertIn("small: IoU mediano", texto)
        self.assertIn("veces el ancho y", texto)
        c = self.resultados(salida)["modelos"][0]["cajas"]
        self.assertGreater(c["iou_mediano"], 0.5)
        self.assertLessEqual(c["iou_p10"], c["iou_mediano"])
        # el detector de juguete marca solo el interior más brillante del círculo: su caja es algo menor que la del conjunto
        self.assertTrue(0.7 < c["ancho_razon_mediana"] < 1.0 and 0.7 < c["alto_razon_mediana"] < 1.0, msg=str(c))

    def test_sin_detecciones_no_se_imprime_la_linea(self):
        val, _, _ = self.etiquetado()
        _, texto = self.correr(val, SesionBrillo(logit=-4.0), salida=os.path.join(self.w, "sal"))
        self.assertNotIn("IoU mediano", texto)


# ======================================================================
class ErroresYSalidasTests(Base):
    def test_errores_de_uso(self):
        val, _, _ = self.etiquetado()
        casos = [
            ({"modelos": []}, "al menos un modelo"),
            ({"modelos": [os.path.join(self.w, "no_existe.onnx")]}, "No existe el modelo"),
            ({"entradas": []}, "--sin-recorte"),
            ({"umbral": 0.0}, "--umbral"),
            ({"umbral": 1.0}, "--umbral"),
            ({"margen_roi": -1}, "--margen-roi"),
        ]
        for kw, texto in casos:
            rc, out = self.correr(val, **kw)
            self.assertEqual(rc, 2, msg=str(kw))
            self.assertIn(texto, out, msg=str(kw))

    def test_una_carpeta_que_no_es_un_conjunto_o_sin_etiquetas(self):
        rc, texto = self.correr(self.w)
        self.assertEqual(rc, 2)
        self.assertIn("manifest.json", texto)
        vacio, _ = self.conjunto(("A",), por_video=5, etiquetar=False)
        rc, texto = self.correr(vacio)
        self.assertEqual(rc, 2)
        self.assertIn("No hay etiquetas que evaluar", texto)

    def test_las_entradas_de_otro_video_no_dan_roi(self):
        val, _, _ = self.etiquetado()
        rc, texto = self.correr(val, entradas=[self.entradas["B"]])
        self.assertEqual(rc, 2)
        self.assertIn("es de un video que no está en el conjunto", texto)
        self.assertIn("falta su ROI", texto)

    def test_entradas_ilegibles_o_sin_roi(self):
        val, _, _ = self.etiquetado()
        rc, texto = self.correr(val, entradas=[self.w])
        self.assertEqual(rc, 2)
        self.assertIn("no se pudieron leer las entradas", texto)
        malo = os.path.join(self.w, "entradas_sin_roi")
        import shutil
        shutil.copytree(self.entradas["A"], malo)
        with open(os.path.join(malo, "inputs.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        del d["roi_box"]
        with open(os.path.join(malo, "inputs.json"), "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        rc, texto = self.correr(val, entradas=[malo])
        self.assertEqual(rc, 2)
        self.assertIn("no trae roi_box", texto)

    def test_un_video_sin_roi_se_avisa_y_los_demas_se_evaluan(self):
        val, vt, _ = self.etiquetado(claves=("A", "B"), por_video=10)
        salida = os.path.join(self.w, "sal")
        rc, texto = self.correr(val, entradas=[self.entradas["A"]], salida=salida)  # falta la corrida de B
        self.assertEqual(rc, 0, msg=texto)
        self.assertIn("AVISO: no hay ROI para col.avi", texto)
        r = self.resultados(salida)["modelos"][0]["resumen"]
        self.assertEqual(r["evaluados"], len(next(v for v in vt if v["nombre"] == "bg.avi")["frames"]))  # solo los de A

    def test_un_modelo_que_no_abre_no_impide_evaluar_los_otros(self):
        val, _, _ = self.etiquetado()
        otro = os.path.join(self.w, "nano.onnx")
        with open(otro, "w") as fh:
            fh.write("x")

        def fabrica(ruta, hilos):
            if ruta == self.modelo:
                raise RuntimeError("archivo dañado")
            return SesionBrillo()
        salida = os.path.join(self.w, "sal")
        msgs = []
        rc = ED.run(val, [self.modelo, otro], [self.entradas["A"]], salida=salida, out=msgs.append, session_factory=fabrica)
        texto = "\n".join(msgs)
        self.assertEqual(rc, 0)
        self.assertIn("No se pudo abrir el modelo", texto)
        self.assertIn("archivo dañado", texto)
        self.assertEqual([m["nombre"] for m in self.resultados(salida)["modelos"]], ["nano"])
        rc = ED.run(val, [self.modelo], [self.entradas["A"]], out=msgs.append, session_factory=fabrica)
        self.assertEqual(rc, 2)
        self.assertIn("No se pudo evaluar ningún modelo", "\n".join(msgs))

    def test_un_modelo_de_entrada_de_tamano_dinamico_se_explica(self):
        val, _, _ = self.etiquetado()
        rc, texto = self.correr(val, SesionBrillo(forma=[1, 3, "alto", "ancho"]))
        self.assertEqual(rc, 2)
        self.assertIn("El modelo small no se puede usar así", texto)
        self.assertIn("tamaño fijo", texto)

    def test_varios_modelos_en_la_misma_tabla(self):
        val, _, _ = self.etiquetado()
        carpeta = os.path.join(self.w, "otra")
        os.makedirs(carpeta)
        gemelo = os.path.join(carpeta, "small.onnx")  # mismo nombre de archivo, otra carpeta
        with open(gemelo, "w") as fh:
            fh.write("x")
        sesiones = {self.modelo: SesionBrillo(), gemelo: SesionBrillo(desvio=(0.25, 0.0))}
        salida = os.path.join(self.w, "sal")
        msgs = []
        rc = ED.run(val, [self.modelo, gemelo], [self.entradas["A"]], salida=salida, out=msgs.append, session_factory=lambda r, h: sesiones[r])
        texto = "\n".join(msgs)
        self.assertEqual(rc, 0)
        res = self.resultados(salida)["modelos"]
        self.assertEqual([m["nombre"] for m in res], ["small", "small #2"])  # no se confunden
        self.assertEqual(res[0]["resumen"]["acierto_pct"], 100.0)
        self.assertEqual(res[1]["resumen"]["acierto_pct"], 0.0)
        self.assertIn("small · detector", texto)
        self.assertIn("small #2 · detector", texto)
        with open(os.path.join(salida, "por_frame.csv"), newline="", encoding="utf-8-sig") as fh:
            filas = list(csv.DictReader(fh))
        self.assertEqual({f["corrida"] for f in filas}, {"small", "small #2"})
        self.assertEqual({f["modo"] for f in filas}, {"detector"})

    def test_archivos_de_salida(self):
        val, vt, _ = self.etiquetado()
        salida = os.path.join(self.w, "sal")
        self.correr(val, SesionBrillo(notas={"margen_roi_px": 0, "modelo": "small"}), salida=salida, umbral=0.6, iou_thr=0.4)
        j = self.resultados(salida)
        self.assertEqual((j["umbral_confianza"], j["umbral_iou"], j["sin_recorte"]), (0.6, 0.4, False))
        m = j["modelos"][0]
        self.assertEqual((m["nombre"], m["ruta"], m["margen_roi_px"]), ("small", self.modelo, 0))
        self.assertEqual(m["notas"], {"margen_roi_px": 0, "modelo": "small"})
        self.assertIn("mediana_ms", m["tiempos"])
        self.assertEqual(j["etiquetas"]["con_caja"] + j["etiquetas"]["vacias"], len(vt[0]["frames"]))
        with open(os.path.join(salida, "por_frame.csv"), newline="", encoding="utf-8-sig") as fh:
            filas = list(csv.DictReader(fh))
        self.assertEqual(len(filas), len(vt[0]["frames"]))
        self.assertEqual(filas[0]["video"], "bg.avi")

    def test_salida_por_omision_dentro_de_la_validacion(self):
        val, _, _ = self.etiquetado()
        from datetime import datetime
        msgs = []
        rc = ED.run(val, [self.modelo], [self.entradas["A"]], out=msgs.append, session_factory=lambda r, h: SesionBrillo(), now=lambda: datetime(2026, 11, 10, 18, 30, 0))
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(os.path.join(val, "evaluacion_detector_20261110_183000", "resultados.json")))

    def test_una_imagen_ilegible_se_avisa_y_no_rompe(self):
        val, vt, _ = self.etiquetado(por_video=10)
        v = vt[0]
        with open(os.path.join(val, "images", f"{v['id']}_f{v['frames'][0]:06d}.{v['formato']}"), "w") as fh:
            fh.write("no soy una imagen")
        rc, texto = self.correr(val, salida=os.path.join(self.w, "sal"))
        self.assertEqual(rc, 0)
        self.assertIn("no se pudo leer la imagen de bg.avi", texto)

    def test_sin_hilos_ni_opciones_el_modelo_se_abre_en_cpu(self):
        """make_session con un onnxruntime de juguete: solo CPU y el número de hilos pedido."""
        capturado = {}

        class Opciones:
            intra_op_num_threads = 0

        class Sesion:
            def __init__(self, ruta, opciones, providers=None):
                capturado.update(ruta=ruta, hilos=opciones.intra_op_num_threads, providers=providers)

        ort = types.ModuleType("onnxruntime")
        ort.SessionOptions, ort.InferenceSession = Opciones, Sesion
        with mock.patch.dict(sys.modules, {"onnxruntime": ort}):
            ED.make_session("m.onnx", 3)
            self.assertEqual(capturado, {"ruta": "m.onnx", "hilos": 3, "providers": ["CPUExecutionProvider"]})
            ED.make_session("m.onnx")
            self.assertEqual(capturado["hilos"], 0)  # sin pedirlo, no se toca

    def test_si_falta_onnxruntime_el_mensaje_dice_como_instalarlo(self):
        with mock.patch.dict(sys.modules, {"onnxruntime": None}):
            with self.assertRaisesRegex(RuntimeError, "pip install onnxruntime"):
                ED.make_session("m.onnx")

    def test_un_archivo_que_no_es_un_modelo_con_el_onnxruntime_de_verdad(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("no hay onnxruntime instalado")
        val, _, _ = self.etiquetado()
        msgs = []
        rc = ED.run(val, [self.modelo], [self.entradas["A"]], out=msgs.append)  # fábrica real: el archivo es texto
        self.assertEqual(rc, 2)
        self.assertIn("No se pudo abrir el modelo", "\n".join(msgs))

    def test_cli(self):
        val, _, _ = self.etiquetado()
        salida = os.path.join(self.w, "cli")
        rc = ED.main([val, "--modelo", self.modelo, "--entradas", self.entradas["A"], "--umbral", "0.4", "--iou", "0.3", "--hilos", "2",
                      "--margen-roi", "5", "--salida", salida, "--hoja-fallos", "0"], session_factory=lambda r, h: SesionBrillo())
        self.assertEqual(rc, 0)
        j = self.resultados(salida)
        self.assertEqual((j["umbral_confianza"], j["umbral_iou"], j["hilos"], j["modelos"][0]["margen_roi_px"]), (0.4, 0.3, 2, 5))
        self.assertEqual(ED.main([val, "--modelo", self.modelo], session_factory=lambda r, h: SesionBrillo()), 2)  # sin --entradas ni --sin-recorte


if __name__ == "__main__":
    unittest.main()