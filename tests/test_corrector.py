"""Pruebas de Tecolote como corrector del seguimiento (detection/corrector.py y su uso en pipeline.run_tracking).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_corrector.py -v

Se usa la escena sintética (un círculo que se mueve; se sabe su posición real en cada frame) con un seguimiento DE JUGUETE que
reproduce los casos reales: sesgado (como la cola), pegado a un punto falso (como un reflejo o el fantasma) y perdido.
«Tecolote» es un detector de juguete que devuelve la posición real (o calla, o duda), envuelto en el TecoloteCorrector real.
"""

import importlib
import math
import tempfile
import unittest

import numpy as np

from detection import corrector as K
from detection.onnx_detector import Detection as TDet
from detection.onnx_detector import DetectorError
from pipeline import TrackingJob, run_tracking
from tracking import BaseTracker, Detection
from analysis.quality import STATUS_DETECTED, STATUS_RECOVERED
from video.frame_selection import FrameRange, compute_timing
from video.reader import VideoReader


def _imp(nombre):
    try:
        return importlib.import_module(f"tests.{nombre}")
    except ImportError:
        return importlib.import_module(nombre)


RI = _imp("test_run_inputs")
BG = RI.BG
ROI = (20, 20, 260, 190)
X0, Y0 = ROI[0], ROI[1]
_, VERDAD = BG.make_frames(90, lost={30, 31, 32})  # posiciones reales en el frame completo (None donde no está el círculo)
_TMP = None


def setUpModule():
    global _TMP
    _TMP = tempfile.TemporaryDirectory()
    RI.Escenas.preparar(_TMP.name)  # escribe la escena sintética (bg.avi y las demás)


def tearDownModule():
    _TMP.cleanup()


def asegurar_escena():
    """La escena es compartida: otras clases de prueba la recrean en SU carpeta temporal y la borran al terminar."""
    import os
    if not os.path.exists(getattr(RI.Escenas, "p_bg", "")):
        RI.Escenas.preparar(_TMP.name)


class Reloj:
    frame = -1


class Juguete(BaseTracker):
    """Seguimiento de juguete. ``sesgo``: lo que se corre respecto a la verdad. ``pegado``: un punto (del ROI) donde se queda
    hasta que lo reubiquen. ``perdido``: frames en que no encuentra nada."""

    mode_id = "1"

    def __init__(self, reloj, sesgo=(0.0, 0.0), pegado=None, perdido=(), pegado_desde=None, sesgo_tras_reubicar=None):
        super().__init__(np.full((ROI[3], ROI[2]), 255, np.uint8))
        self.reloj, self.sesgo, self.perdido = reloj, sesgo, set(perdido)
        self.pegado = pegado if pegado_desde is None else None
        self.pegar_en = (pegado_desde, pegado) if pegado_desde is not None else None  # se queda pegado A PARTIR de ese frame
        self.sesgo_tras_reubicar = sesgo_tras_reubicar
        self.reubicaciones = []
        self.vistos = 0  # frames procesados: el reset del arranque (antes del primero) no es una reubicación

    def reset(self, center, bbox):
        if self.vistos > 0:  # el reset del arranque no cuenta como reubicación
            self.reubicaciones.append((self.reloj.frame, tuple(center), tuple(bbox)))
            self.pegado, self.pegar_en = None, None
            if self.sesgo_tras_reubicar is not None:
                self.sesgo = self.sesgo_tras_reubicar
        super().reset(center, bbox)

    def _detect(self, roi):
        self.vistos += 1
        self.reloj.frame += 1
        f = self.reloj.frame
        if f in self.perdido:
            return Detection()
        if self.pegar_en is not None and f >= self.pegar_en[0]:
            self.pegado, self.pegar_en = self.pegar_en[1], None
        if self.pegado is not None:
            c = self.pegado
        else:
            v = VERDAD[f]
            if v is None:
                return Detection()
            c = (v[0] - X0 + self.sesgo[0], v[1] - Y0 + self.sesgo[1])
        return Detection(center=c, bbox=(int(c[0] - 13), int(c[1] - 13), 27, 27))


class DetectorJuguete:
    """Tecolote de juguete: la posición real del círculo, con la confianza dada (o None si está «ciego» en ese frame)."""

    def __init__(self, reloj, score=0.95, ciego=(), falla_en=None):
        self.reloj, self.score, self.ciego, self.falla_en = reloj, score, set(ciego), falla_en
        self.llamadas = []
        self.model_path = "/modelos/rfdetr-small.onnx"

    def best(self, frame, roi_box):
        f = self.reloj.frame
        self.llamadas.append(f)
        if self.falla_en is not None and f >= self.falla_en:
            raise RuntimeError("el modelo se cayó")
        v = VERDAD[f]
        if v is None or f in self.ciego:
            return None
        return TDet((v[0] - 13.0, v[1] - 13.0, 26.0, 26.0), self.score)

    def describe(self):
        return {"modelo": "rfdetr-small.onnx", "umbral": 0.5, "margen_roi_px": 0}


def correr(tracker, corrector=None, rango=(0, 85), pasos=None, filtros=False, guardar_eventos=False, **kw):
    eventos, msgs = {}, []
    asegurar_escena()
    tracker.reloj.frame = rango[0] - 1  # el reloj de juguete empieza donde empieza la corrida
    with VideoReader(RI.Escenas.p_bg) as r:
        timing = compute_timing(r.fps) if pasos is None else compute_timing(r.fps, True, r.fps / pasos)
        extra = {} if filtros else {"smoothing_alpha": None, "jump_max_px": None, "adopt_after_s": None}
        v0 = VERDAD[rango[0]]
        job = TrackingJob(r, FrameRange(*rango), timing, tracker, ROI, start=((v0[0] - X0, v0[1] - Y0), (0, 0, 27, 27)),
                          corrector=corrector, **extra, **kw)

        def on_frame(ev):
            if guardar_eventos:
                eventos[ev.frame_idx] = ev.detection
            return None
        res = run_tracking(job, on_frame=on_frame, keep_records=True, log=msgs.append)
    return res, eventos, "\n".join(msgs)


def errores(res):
    """Error (px) de la posición registrada respecto a la real, por frame; None si no hubo posición."""
    out = {}
    for rec in res.records:
        v = VERDAD[rec.frame]
        if rec.found and v is not None:
            out[rec.frame] = math.hypot(rec.cx_abs - v[0], (240 - rec.cy_abs_inv) - v[1])
        else:
            out[rec.frame] = None
    return out


# ======================================================================
class ClasificacionTests(unittest.TestCase):
    M = K.Measure((100.0, 100.0), (80.0, 90.0, 40.0, 20.0), 0.9)  # lado mayor: 40

    def test_cerca_es_sesgo_y_da_el_desvio(self):
        tipo, d = K.classify((90.0, 95.0), self.M, 1.0)
        self.assertEqual((tipo, d), (K.SESGO, (10.0, 5.0)))  # lo que hay que SUMAR al seguimiento

    def test_lejos_o_perdido_es_rescate(self):
        self.assertEqual(K.classify((200.0, 100.0), self.M, 1.0), (K.RESCATE, None))
        self.assertEqual(K.classify(None, self.M, 1.0), (K.RESCATE, None))

    def test_el_limite_es_el_lado_mayor_por_la_proporcion(self):
        self.assertEqual(K.classify((60.0, 100.0), self.M, 1.0)[0], K.SESGO)  # 40 px = 1.0 x 40: justo en el límite, cuenta
        self.assertEqual(K.classify((59.0, 100.0), self.M, 1.0)[0], K.RESCATE)
        self.assertEqual(K.classify((80.0, 100.0), self.M, 0.5)[0], K.SESGO)  # 20 px = 0.5 x 40
        self.assertEqual(K.classify((79.0, 100.0), self.M, 0.5)[0], K.RESCATE)
        alto = K.Measure((100.0, 100.0), (90.0, 70.0, 20.0, 60.0), 0.9)  # el lado mayor es el alto
        self.assertEqual(K.classify((100.0, 41.0), alto, 1.0)[0], K.SESGO)


class CorrectorTests(unittest.TestCase):
    def test_solo_cuenta_la_confianza_suficiente(self):
        reloj = Reloj()
        reloj.frame = 50
        c = K.TecoloteCorrector(DetectorJuguete(reloj, score=0.79))
        self.assertIsNone(c.measure(None, ROI))
        c = K.TecoloteCorrector(DetectorJuguete(reloj, score=0.80))
        m = c.measure(None, ROI)
        self.assertEqual((m.score, m.center), (0.80, (VERDAD[50][0], VERDAD[50][1])))
        self.assertIsNone(K.TecoloteCorrector(DetectorJuguete(reloj, ciego={50})).measure(None, ROI))
        self.assertIsNotNone(K.TecoloteCorrector(DetectorJuguete(reloj, score=0.6), min_score=0.5).measure(None, ROI))

    def test_confianza_minima_valida(self):
        for malo in (0.0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                K.TecoloteCorrector(DetectorJuguete(Reloj()), min_score=malo)

    def test_describe(self):
        d = K.TecoloteCorrector(DetectorJuguete(Reloj()), 0.85).describe()
        self.assertEqual(d, {"modelo_ruta": "/modelos/rfdetr-small.onnx", "confianza_minima": 0.85, "modelo": "rfdetr-small.onnx",
                             "umbral": 0.5, "margen_roi_px": 0})

    def test_un_modelo_que_no_existe(self):
        with self.assertRaises(DetectorError):
            K.make_corrector("/no/existe.onnx")


# ======================================================================
class CorreccionDelSesgoTests(unittest.TestCase):
    def test_sin_tecolote_el_sesgo_queda(self):
        res, _, _ = correr(Juguete(Reloj(), sesgo=(20.0, 0.0)))
        e = [v for v in errores(res).values() if v is not None]
        self.assertTrue(all(abs(v - 20.0) < 1e-6 for v in e))
        self.assertIsNone(res.settings["tecolote_corrector"])

    def test_con_tecolote_el_sesgo_se_corrige_en_todos_los_frames(self):
        reloj = Reloj()
        res, _, texto = correr(Juguete(reloj, sesgo=(20.0, 0.0)), K.TecoloteCorrector(DetectorJuguete(reloj)), correct_every_s=0.2)
        e = errores(res)
        con = [v for v in e.values() if v is not None]
        self.assertGreater(len(con), 70)
        self.assertLess(max(con), 1e-6)  # el desvío medido se aplica también ENTRE mediciones
        self.assertEqual(res.settings["tecolote_cada_frames"], 5)  # 0.2 s a 25 fps
        self.assertEqual(res.settings["tecolote_reubicaciones"], 0)
        self.assertEqual(res.settings["tecolote_correcciones"], res.settings["tecolote_mediciones"])
        self.assertIn("Tecolote: ", texto)
        self.assertIn("correcciones de desvío y 0 reubicaciones", texto)

    def test_mide_cada_tantos_frames_y_en_el_primero(self):
        reloj = Reloj()
        det = DetectorJuguete(reloj)
        correr(Juguete(reloj, sesgo=(20.0, 0.0)), K.TecoloteCorrector(det), correct_every_s=0.4)  # 10 frames
        self.assertEqual(det.llamadas, list(range(0, 86, 10)))

    def test_con_paso_de_frames_la_frecuencia_es_en_segundos(self):
        reloj = Reloj()
        res, _, _ = correr(Juguete(reloj, sesgo=(20.0, 0.0)), K.TecoloteCorrector(DetectorJuguete(reloj)), pasos=2, correct_every_s=0.48)
        self.assertEqual(res.settings["tecolote_cada_frames"], 6)  # 0.48 s / (2/25 s) = 6 frames procesados

    def test_con_confianza_baja_no_se_corrige(self):
        reloj = Reloj()
        res, _, _ = correr(Juguete(reloj, sesgo=(20.0, 0.0)), K.TecoloteCorrector(DetectorJuguete(reloj, score=0.7)), correct_every_s=0.2)
        self.assertTrue(all(abs(v - 20.0) < 1e-6 for v in errores(res).values() if v is not None))
        self.assertEqual(res.settings["tecolote_mediciones"], 0)

    def test_el_desvio_caduca_si_tecolote_deja_de_confirmarlo(self):
        reloj = Reloj()
        res, _, _ = correr(Juguete(reloj, sesgo=(20.0, 0.0)), K.TecoloteCorrector(DetectorJuguete(reloj, ciego=set(range(41, 90)))),
                           correct_every_s=0.2, correct_offset_max_age_s=0.4)  # 10 frames
        e = errores(res)
        self.assertLess(e[45], 1e-6)  # la última medición útil fue en 40; en 45 Tecolote no ve nada, pero el desvío se sigue aplicando
        self.assertLess(e[50], 1e-6)  # 10 frames: todavía
        self.assertAlmostEqual(e[51], 20.0)  # 11 frames sin confirmar: se deja de aplicar
        self.assertAlmostEqual(e[80], 20.0)

    def test_en_los_frames_sin_posicion_no_se_inventa_nada(self):
        reloj = Reloj()
        res, _, _ = correr(Juguete(reloj, sesgo=(20.0, 0.0)), K.TecoloteCorrector(DetectorJuguete(reloj)), correct_every_s=0.2)
        recs = {r.frame: r for r in res.records}
        for f in (30, 31, 32):  # el círculo no está: ni el seguimiento ni Tecolote lo ven
            self.assertFalse(recs[f].found)

    def test_la_posicion_cruda_del_seguimiento_queda_registrada(self):
        reloj = Reloj()
        res, _, _ = correr(Juguete(reloj, sesgo=(20.0, 0.0)), K.TecoloteCorrector(DetectorJuguete(reloj)), correct_every_s=0.2)
        r = next(r for r in res.records if r.frame == 47)
        self.assertAlmostEqual(r.cx_raw - VERDAD[47][0], 20.0)  # lo que dijo el seguimiento, sin corregir
        self.assertAlmostEqual(r.cx_abs, VERDAD[47][0])  # la posición corregida

    def test_la_caja_del_evento_se_corrige_tambien(self):
        reloj = Reloj()
        _, ev, _ = correr(Juguete(reloj, sesgo=(20.0, 0.0)), K.TecoloteCorrector(DetectorJuguete(reloj)), correct_every_s=0.2, guardar_eventos=True)
        v = VERDAD[45]  # frame de medición: la caja es la de Tecolote
        self.assertEqual(ev[45].bbox, (v[0] - 13.0 - X0, v[1] - 13.0 - Y0, 26.0, 26.0))
        v = VERDAD[47]  # entre mediciones: la del seguimiento, corrida por el desvío
        self.assertAlmostEqual(ev[47].center[0], v[0] - X0)
        self.assertAlmostEqual(ev[47].bbox[0], int(v[0] - X0 + 20.0 - 13) - 20.0)


class RescateTests(unittest.TestCase):
    def test_pegado_a_un_punto_falso_se_reubica(self):
        reloj = Reloj()
        tr = Juguete(reloj, pegado=(240.0, 170.0))  # lejos del círculo (que empieza a la izquierda)
        res, _, texto = correr(tr, K.TecoloteCorrector(DetectorJuguete(reloj)), correct_every_s=0.4)
        self.assertEqual(tr.reubicaciones[0][0], 0)  # en la primera medición
        f0, c, b = tr.reubicaciones[0]
        self.assertEqual(c, (VERDAD[0][0] - X0, VERDAD[0][1] - Y0))  # en coordenadas del ROI, como espera el seguimiento
        self.assertEqual(b, (VERDAD[0][0] - 13.0 - X0, VERDAD[0][1] - 13.0 - Y0, 26.0, 26.0))
        self.assertEqual(res.records[0].status, STATUS_RECOVERED)
        self.assertLess(max(v for v in errores(res).values() if v is not None), 1e-6)  # después, sigue al círculo
        self.assertEqual(res.settings["tecolote_reubicaciones"], 1)
        self.assertIn("Tecolote reubicó al sujeto en el frame 0", texto)

    def test_sin_tecolote_se_queda_pegado(self):
        res, _, _ = correr(Juguete(Reloj(), pegado=(240.0, 170.0)))
        self.assertGreater(np.median([v for v in errores(res).values() if v is not None]), 60)

    def test_al_perderse_se_le_pregunta_en_el_momento(self):
        reloj = Reloj()
        det = DetectorJuguete(reloj)
        tr = Juguete(reloj, perdido=set(range(50, 61)))
        res, _, _ = correr(tr, K.TecoloteCorrector(det), correct_every_s=1.0)  # mediciones programadas: 0, 25, 50, 75...
        self.assertIn(50, det.llamadas)
        recs = {r.frame: r for r in res.records}
        self.assertTrue(recs[50].found)
        self.assertEqual(recs[50].status, STATUS_RECOVERED)
        tr2_reloj = Reloj()
        det2 = DetectorJuguete(tr2_reloj)
        correr(Juguete(tr2_reloj, perdido=set(range(53, 61))), K.TecoloteCorrector(det2), correct_every_s=1.0)
        self.assertIn(53, det2.llamadas)  # el primer frame perdido, fuera del calendario
        self.assertNotIn(54, det2.llamadas)  # y no en cada frame perdido: eso costaría ~250 ms por frame
        self.assertEqual([f for f in det2.llamadas if 53 <= f <= 60], [53])

    def test_con_los_filtros_de_siempre_el_rescate_no_lo_rechaza_el_filtro_de_saltos(self):
        """Pegado a 200 px (más que el salto máximo de 120): tras el rescate debe seguir al círculo igual que un seguimiento
        perfecto con los mismos filtros (el suavizado tiene su propio retraso, que no es culpa de la corrección)."""
        reloj = Reloj()
        res, _, _ = correr(Juguete(reloj, pegado=(250.0, 180.0)), K.TecoloteCorrector(DetectorJuguete(reloj)), filtros=True, correct_every_s=0.4)
        perfecto, _, _ = correr(Juguete(Reloj()), filtros=True)
        e, ep = errores(res), errores(perfecto)
        self.assertEqual(res.records[0].status, STATUS_RECOVERED)
        self.assertLess(max(abs(e[f] - ep[f]) for f in range(40, 86) if e[f] is not None and ep[f] is not None), 0.5)

    def test_con_los_filtros_de_siempre_el_sesgo_tambien_se_corrige(self):
        """Un sesgo de 13 px (la mitad del lado de la caja de Tecolote, como 27 px en un ratón de ~110). Corregido, el error
        queda como el de un seguimiento perfecto con los mismos filtros; sin corregir, el sesgo se suma."""
        reloj = Reloj()
        res, _, _ = correr(Juguete(reloj, sesgo=(12.0, -5.0)), K.TecoloteCorrector(DetectorJuguete(reloj)), filtros=True, correct_every_s=0.2)
        sin, _, _ = correr(Juguete(Reloj(), sesgo=(12.0, -5.0)), filtros=True)
        perfecto, _, _ = correr(Juguete(Reloj()), filtros=True)
        med = lambda r: float(np.median([v for v in errores(r).values() if v is not None]))  # noqa: E731
        self.assertLess(abs(med(res) - med(perfecto)), 0.5)
        self.assertGreater(med(sin) - med(res), 3.0)  # sin corregir es claramente peor (el sesgo y el retraso del suavizado en parte se compensan)
        self.assertEqual(res.settings["tecolote_reubicaciones"], 0)  # todas fueron correcciones de sesgo

    def test_un_sesgo_mayor_que_el_animal_se_trata_como_rescate(self):
        """Más lejos que un lado de la caja de Tecolote: no es «el mismo ratón corrido» y se reubica (con la proporción por omisión)."""
        reloj = Reloj()
        res, _, _ = correr(Juguete(reloj, sesgo=(30.0, 0.0)), K.TecoloteCorrector(DetectorJuguete(reloj)), correct_every_s=0.4)
        self.assertEqual(res.settings["tecolote_correcciones"], 0)
        self.assertEqual(res.settings["tecolote_reubicaciones"], res.settings["tecolote_mediciones"])


class RescateAMitadDeCorridaTests(unittest.TestCase):
    """Con los filtros de siempre: va bien (con un sesgo corregido), y en el frame 40 se queda pegado a 200 px."""

    def corrida(self):
        reloj = Reloj()
        tr = Juguete(reloj, sesgo=(10.0, 0.0), pegado=(250.0, 180.0), pegado_desde=40, sesgo_tras_reubicar=(0.0, 0.0))
        res, _, _ = correr(tr, K.TecoloteCorrector(DetectorJuguete(reloj)), filtros=True, correct_every_s=0.2)
        return res, tr

    def test_el_rescate_cae_en_el_primer_frame_perdido_y_es_exacto(self):
        res, tr = self.corrida()
        recs = {r.frame: r for r in res.records}
        self.assertEqual(tr.reubicaciones[0][0], 40)  # el salto lo rechaza el filtro; el seguimiento queda sin posición; se mide ahí mismo
        self.assertTrue(recs[40].found)
        self.assertEqual(recs[40].status, STATUS_RECOVERED)
        self.assertLess(errores(res)[40], 1e-6)  # el suavizado se reinicia: la posición es la de Tecolote, sin arrastre del pasado

    def test_tras_el_rescate_no_se_aplica_el_desvio_de_antes(self):
        res, _ = self.corrida()
        perfecto, _, _ = correr(Juguete(Reloj()), filtros=True, rango=(40, 85))
        e, ep = errores(res), errores(perfecto)
        # el sesgo de antes (10 px) ya no existe tras reubicar: si se siguiera sumando, estos frames tendrían ~10 px de más
        self.assertLess(max(abs(e[f] - ep[f]) for f in range(41, 45)), 0.5)

    def test_un_rescate_al_principio_tambien_da_posicion_en_ese_frame(self):
        reloj = Reloj()
        res, _, _ = correr(Juguete(reloj, pegado=(250.0, 180.0)), K.TecoloteCorrector(DetectorJuguete(reloj)), filtros=True, correct_every_s=0.4)
        self.assertTrue(res.records[0].found)
        self.assertLess(errores(res)[0], 1e-6)  # aunque el filtro de saltos ya «conocía» el punto falso: el rescate no pasa por él


class FallosTests(unittest.TestCase):
    def test_un_tecolote_que_falla_se_desactiva_y_la_corrida_termina(self):
        reloj = Reloj()
        res, _, _ = correr(Juguete(reloj, sesgo=(20.0, 0.0)), K.TecoloteCorrector(DetectorJuguete(reloj, falla_en=40)), correct_every_s=0.2)
        self.assertEqual(res.frames_processed, 86)
        self.assertTrue(any("Tecolote falló en el frame 40" in w for w in res.warnings))
        e = errores(res)
        self.assertLess(e[35], 1e-6)
        self.assertAlmostEqual(e[45], 20.0)  # sin corrector, vuelve el sesgo (el desvío viejo no se sigue aplicando)

    def test_los_ajustes_quedan_registrados(self):
        reloj = Reloj()
        res, _, _ = correr(Juguete(reloj, sesgo=(20.0, 0.0)), K.TecoloteCorrector(DetectorJuguete(reloj), 0.85), correct_every_s=0.2,
                           correct_max_offset_ratio=0.7, correct_offset_max_age_s=1.0)
        s = res.settings
        self.assertEqual(s["tecolote_corrector"]["confianza_minima"], 0.85)
        self.assertEqual((s["tecolote_cada_frames"], s["tecolote_desvio_max_relativo"], s["tecolote_desvio_caduca_frames"]), (5, 0.7, 25))


if __name__ == "__main__":
    unittest.main()


# ======================================================================
# Grabar, repetir y evaluar con Tecolote (persistence/run_inputs.py y tools/eval_tracking.py)
# ======================================================================
import os  # noqa: E402
from unittest import mock  # noqa: E402

from detection import onnx_detector as D  # noqa: E402
from persistence import run_inputs as P  # noqa: E402
from tools import eval_tracking as ET  # noqa: E402

TD = _imp("test_eval_detector")
TM = _imp("test_build_manual")


def tecolote_de_juguete(ruta, min_score=0.8):
    """Tecolote real (OnnxDetector + TecoloteCorrector) con la sesión de juguete que ve el círculo blanco en el tensor."""
    return K.TecoloteCorrector(D.OnnxDetector(ruta, session_factory=lambda r, h: TD.SesionBrillo()), min_score)


class GrabarYRepetirTests(unittest.TestCase):
    def setUp(self):
        asegurar_escena()
        self.tmp = tempfile.TemporaryDirectory()
        self.modelo = os.path.join(self.tmp.name, "rfdetr-small.onnx")
        with open(self.modelo, "w") as fh:
            fh.write("juguete")

    def tearDown(self):
        self.tmp.cleanup()

    def hacer(self, **kw):
        import dataclasses
        return lambda r: dataclasses.replace(RI.Escenas.job_fondo(r), corrector=tecolote_de_juguete(self.modelo, 0.85), correct_every_s=0.4,
                                             correct_max_offset_ratio=0.9, correct_offset_max_age_s=1.5, **kw)

    def test_se_graba_lo_necesario_para_volver_a_crearlo(self):
        with mock.patch("detection.corrector.make_corrector", side_effect=lambda ruta, ms: tecolote_de_juguete(ruta, ms)):
            o, r, carpeta = RI.ida_y_vuelta(os.path.join(self.tmp.name, "a"), RI.Escenas.p_bg, self.hacer(), RI.Escenas.OPT_FONDO, sample_idx=35)
        with open(os.path.join(carpeta, "inputs.json"), encoding="utf-8") as fh:
            spec = json_load(fh)["tecolote_corrector"]
        self.assertEqual(spec, {"modelo_ruta": self.modelo, "confianza_minima": 0.85, "cada_s": 0.4, "desvio_max_relativo": 0.9, "desvio_caduca_s": 1.5})
        self.assertEqual(o.records, r.records)  # repetida con el mismo Tecolote: registros idénticos
        self.assertEqual(r.settings["tecolote_cada_frames"], o.settings["tecolote_cada_frames"])
        self.assertGreater(o.settings["tecolote_mediciones"], 0)

    def test_al_repetir_se_vuelve_a_crear_con_el_mismo_modelo_y_confianza(self):
        visto = []
        with mock.patch("detection.corrector.make_corrector", side_effect=lambda ruta, ms: visto.append((ruta, ms)) or tecolote_de_juguete(ruta, ms)):
            RI.ida_y_vuelta(os.path.join(self.tmp.name, "a"), RI.Escenas.p_bg, self.hacer(), RI.Escenas.OPT_FONDO, sample_idx=35)
        self.assertEqual(visto, [(self.modelo, 0.85)])

    def test_se_puede_quitar_o_cambiar_la_frecuencia_al_repetir(self):
        with mock.patch("detection.corrector.make_corrector", side_effect=AssertionError("no debía crearse")):
            _, r, _ = RI.ida_y_vuelta(os.path.join(self.tmp.name, "b"), RI.Escenas.p_bg, self.hacer(), RI.Escenas.OPT_FONDO, sample_idx=35, corrector=None)
        self.assertIsNone(r.settings["tecolote_corrector"])
        with mock.patch("detection.corrector.make_corrector", side_effect=lambda ruta, ms: tecolote_de_juguete(ruta, ms)):
            _, r, _ = RI.ida_y_vuelta(os.path.join(self.tmp.name, "c"), RI.Escenas.p_bg, self.hacer(), RI.Escenas.OPT_FONDO, sample_idx=35, correct_every_s=0.08)
        self.assertEqual(r.settings["tecolote_cada_frames"], 1)  # 0.08 s a 12.5 fps procesados

    def test_si_el_modelo_ya_no_esta_la_repeticion_lo_dice(self):
        with mock.patch("detection.corrector.make_corrector", side_effect=lambda ruta, ms: tecolote_de_juguete(ruta, ms)):
            with VideoReader(RI.Escenas.p_bg) as r:
                carpeta = P.save_run_inputs(os.path.join(self.tmp.name, "d"), self.hacer()(r), video_path=RI.Escenas.p_bg, options=RI.Escenas.OPT_FONDO,
                                            sample_idx=35, app_version="1.0", log=None, now=RI.AHORA)
        os.remove(self.modelo)
        with VideoReader(RI.Escenas.p_bg) as r:
            with self.assertRaisesRegex(D.DetectorError, "No existe el modelo"):
                P.build_job(P.load_run_inputs(carpeta), r)

    def test_una_corrida_sin_tecolote_se_repite_sin_el(self):
        with mock.patch("detection.corrector.make_corrector", side_effect=AssertionError("no debía crearse")):
            o, r, carpeta = RI.ida_y_vuelta(os.path.join(self.tmp.name, "e"), RI.Escenas.p_bg, RI.Escenas.job_fondo, RI.Escenas.OPT_FONDO, sample_idx=35)
        self.assertEqual(o.records, r.records)
        with open(os.path.join(carpeta, "inputs.json"), encoding="utf-8") as fh:
            self.assertIsNone(json_load(fh)["tecolote_corrector"])

    def test_unas_entradas_anteriores_sin_la_clave_se_repiten_sin_tecolote(self):
        import shutil
        with VideoReader(RI.Escenas.p_bg) as r:
            carpeta = P.save_run_inputs(os.path.join(self.tmp.name, "f"), RI.Escenas.job_fondo(r), video_path=RI.Escenas.p_bg, options=RI.Escenas.OPT_FONDO,
                                        sample_idx=35, app_version="1.0", log=None, now=RI.AHORA)
        with open(os.path.join(carpeta, "inputs.json"), encoding="utf-8") as fh:
            d = json_load(fh)
        del d["tecolote_corrector"]
        with open(os.path.join(carpeta, "inputs.json"), "w", encoding="utf-8") as fh:
            import json
            json.dump(d, fh)
        with VideoReader(RI.Escenas.p_bg) as r:
            self.assertIsNone(P.build_job(P.load_run_inputs(carpeta), r).corrector)


def json_load(fh):
    import json
    return json.load(fh)


class EvaluarConTecoloteTests(TM.Base):
    def setUp(self):
        super().setUp()
        self.modelo = os.path.join(self.w, "rfdetr-small.onnx")
        with open(self.modelo, "w") as fh:
            fh.write("juguete")

    def evaluar(self, val, **kw):
        msgs = []
        kw.setdefault("corrector_factory", lambda ruta: tecolote_de_juguete(ruta))
        rc = ET.run(val, [self.entradas["A"]], out=msgs.append, salida=os.path.join(self.w, f"ev{len(os.listdir(self.w))}"), hoja_fallos=0, **kw)
        return rc, "\n".join(msgs)

    def test_con_tecolote_la_fila_lo_dice_y_se_usa_el_modelo_pedido(self):
        val, _ = self.conjunto(("A",), por_video=6)
        pedidos = []
        rc, texto = self.evaluar(val, tecolote=self.modelo, corrector_factory=lambda ruta: pedidos.append(ruta) or tecolote_de_juguete(ruta))
        self.assertEqual(rc, 0, msg=texto)
        self.assertIn("fondo + Tecolote", texto)
        self.assertIn("Tecolote corrigiendo con rfdetr-small.onnx, cada 0.5 s", texto)
        self.assertEqual(pedidos, [self.modelo])

    def test_la_frecuencia_y_quitar_a_tecolote_llegan_a_la_corrida(self):
        val, _ = self.conjunto(("A",), por_video=6)
        with mock.patch.object(ET, "build_job", wraps=ET.build_job) as espia:
            self.evaluar(val, tecolote=self.modelo, tecolote_cada_s=0.25)
            self.assertEqual(espia.call_args.kwargs["correct_every_s"], 0.25)
            self.assertIsInstance(espia.call_args.kwargs["corrector"], K.TecoloteCorrector)
            rc, texto = self.evaluar(val, sin_tecolote=True)
            self.assertIsNone(espia.call_args.kwargs["corrector"])
        self.assertIn("fondo sin Tecolote", texto)

    def test_sin_opciones_no_se_toca_lo_grabado(self):
        val, _ = self.conjunto(("A",), por_video=6)
        with mock.patch.object(ET, "build_job", wraps=ET.build_job) as espia:
            rc, texto = self.evaluar(val)
        self.assertEqual(rc, 0)
        self.assertNotIn("corrector", espia.call_args.kwargs)
        self.assertNotIn("Tecolote", texto)

    def test_errores(self):
        val, _ = self.conjunto(("A",), por_video=6)
        self.assertEqual(self.evaluar(val, tecolote=self.modelo, sin_tecolote=True)[0], 2)
        self.assertEqual(self.evaluar(val, tecolote=self.modelo, tecolote_cada_s=0)[0], 2)

        def falla(ruta):
            raise D.DetectorError("No existe el modelo 'x'.")
        rc, texto = self.evaluar(val, tecolote="/no/existe.onnx", corrector_factory=falla)
        self.assertEqual(rc, 2)
        self.assertIn("No se pudo preparar a Tecolote con /no/existe.onnx", texto)
        rc, texto = self.evaluar(val, tecolote="/no/existe.onnx", corrector_factory=None)  # la fábrica real
        self.assertEqual(rc, 2)
        self.assertIn("No existe el modelo", texto)

    def test_la_linea_de_comandos(self):
        val, _ = self.conjunto(("A",), por_video=6)
        with mock.patch("detection.corrector.make_corrector", side_effect=lambda ruta: tecolote_de_juguete(ruta)):
            rc = ET.main([val, "--entradas", self.entradas["A"], "--tecolote", self.modelo, "--tecolote-cada-s", "0.3", "--hoja-fallos", "0",
                          "--salida", os.path.join(self.w, "cli")])
        self.assertEqual(rc, 0)
        with open(os.path.join(self.w, "cli", "resultados.json"), encoding="utf-8") as fh:
            j = json_load(fh)
        self.assertEqual((j["tecolote"], j["tecolote_cada_s"], j["sin_tecolote"]), (self.modelo, 0.3, False))