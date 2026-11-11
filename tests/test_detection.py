"""Pruebas del detector dentro de TrackerLab: detection/onnx_detector.py y detection/subject.py.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_detection.py -v

El «modelo» es el detector de juguete de tests/test_eval_detector.py, que lee el tensor de entrada y devuelve la caja de lo más
brillante sin decodificar, como el ONNX real. Lo que depende de RF-DETR real solo se prueba con el modelo de Colab.
"""

import importlib
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from detection import onnx_detector as D
from detection import subject as S


def _imp(nombre):
    try:
        return importlib.import_module(f"tests.{nombre}")
    except ImportError:
        return importlib.import_module(nombre)


TD = _imp("test_eval_detector")
BG = TD.BG
ROI = (50, 20, 100, 60)  # x 50..150, y 20..80


def frame_con_cuadro(x=100, y=40, w=30, h=20, color=(255, 255, 255), alto=100, ancho=200):
    img = np.zeros((alto, ancho, 3), np.uint8)
    img[y:y + h, x:x + w] = color
    return img


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.modelo = os.path.join(self.tmp.name, "raton.onnx")
        with open(self.modelo, "w") as fh:
            fh.write("la sesión es de juguete")

    def tearDown(self):
        self.tmp.cleanup()

    def detector(self, sesion=None, **kw):
        sesion = sesion or TD.SesionBrillo(hw=(60, 100))
        return D.OnnxDetector(self.modelo, session_factory=lambda ruta, hilos: sesion, **kw)


# ======================================================================
class AperturaTests(Base):
    def test_modelo_que_no_existe(self):
        for ruta in ("", None, os.path.join(self.tmp.name, "no.onnx"), self.tmp.name):
            with self.assertRaisesRegex(D.DetectorError, "No existe el modelo"):
                D.OnnxDetector(ruta, session_factory=lambda r, h: TD.SesionBrillo())

    def test_modelo_que_no_abre(self):
        def falla(ruta, hilos):
            raise OSError("archivo dañado")
        with self.assertRaisesRegex(D.DetectorError, "No se pudo abrir el modelo raton.onnx: archivo dañado"):
            D.OnnxDetector(self.modelo, session_factory=falla)

    def test_entrada_de_tamano_dinamico(self):
        with self.assertRaisesRegex(D.DetectorError, "tamaño fijo"):
            self.detector(TD.SesionBrillo(forma=[1, 3, "alto", "ancho"]))
        with self.assertRaises(D.DetectorError):
            self.detector(TD.SesionBrillo(forma=[3, 64, 64]))

    def test_argumentos_invalidos(self):
        for kw in ({"threshold": 0.0}, {"threshold": 1.0}, {"threshold": -0.1}, {"margin": -1}):
            with self.assertRaises(ValueError, msg=str(kw)):
                self.detector(**kw)

    def test_los_hilos_llegan_a_la_sesion(self):
        visto = {}

        def fabrica(ruta, hilos):
            visto.update(ruta=ruta, hilos=hilos)
            return TD.SesionBrillo()
        D.OnnxDetector(self.modelo, threads=3, session_factory=fabrica)
        self.assertEqual(visto, {"ruta": self.modelo, "hilos": 3})

    def test_con_el_onnxruntime_de_verdad_un_archivo_de_texto_se_explica(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("no hay onnxruntime instalado")
        with self.assertRaisesRegex(D.DetectorError, "No se pudo abrir el modelo"):
            D.OnnxDetector(self.modelo)


class NotasTests(Base):
    def test_margen_de_las_notas_y_lo_indicado_manda(self):
        d = self.detector(TD.SesionBrillo(notas={"margen_roi_px": 12, "recorte_roi": True}))
        self.assertEqual((d.margin, d.margin_source, d.crops_roi), (12, "leído de las notas del modelo", True))
        self.assertEqual(self.detector(TD.SesionBrillo(notas={"margen_roi_px": 12}), margin=0).margin, 0)
        self.assertEqual(self.detector().margin, 0)  # sin notas

    def test_umbral_sugerido_por_el_modelo(self):
        self.assertEqual(self.detector(TD.SesionBrillo(notas={"umbral_sugerido": 0.6})).threshold, 0.6)
        self.assertEqual(self.detector(TD.SesionBrillo(notas={"umbral_sugerido": 0.6}), threshold=0.3).threshold, 0.3)
        self.assertEqual(self.detector().threshold, 0.5)
        for malo in (True, "0.6", 0, 1, 1.5, -0.2, None):
            self.assertEqual(self.detector(TD.SesionBrillo(notas={"umbral_sugerido": malo})).threshold, 0.5, msg=repr(malo))

    def test_modelo_entrenado_con_frames_completos(self):
        d = self.detector(TD.SesionBrillo(notas={"recorte_roi": False, "margen_roi_px": 9}))
        self.assertEqual((d.crops_roi, d.margin), (False, 0))
        self.assertTrue(self.detector(TD.SesionBrillo(notas={"recorte_roi": None})).crops_roi)  # solo un False explícito lo desactiva

    def test_describe(self):
        d = self.detector(TD.SesionBrillo(notas={"margen_roi_px": 4, "modelo": "small"}))
        self.assertEqual(d.describe(), {"modelo": "raton.onnx", "umbral": 0.5, "margen_roi_px": 4, "origen_del_margen": "leído de las notas del modelo",
                                        "recorte_roi": True, "notas": {"margen_roi_px": 4, "modelo": "small"}})


# ======================================================================
class DeteccionTests(Base):
    def test_la_caja_sale_en_coordenadas_del_frame_completo(self):
        det = self.detector().best(frame_con_cuadro(), ROI)
        self.assertIsNotNone(det)
        x, y, w, h = det.box
        self.assertLess(abs(x - 100), 2)
        self.assertLess(abs(y - 40), 2)
        self.assertLess(abs(w - 30), 2.5)
        self.assertLess(abs(h - 20), 2.5)
        self.assertEqual(det.center, (x + w / 2, y + h / 2))
        self.assertGreater(det.score, 0.9)

    def test_lo_que_esta_fuera_del_roi_no_se_ve(self):
        img = frame_con_cuadro()
        img[5:15, 170:195] = 255  # otro cuadro, fuera del ROI
        det = self.detector().best(img, ROI)
        self.assertLess(abs(det.box[2] - 30), 2.5)  # solo el de adentro, no la unión de los dos
        solo_fuera = frame_con_cuadro(x=170, y=5, w=25, h=10)
        self.assertIsNone(self.detector().best(solo_fuera, ROI))

    def test_sin_roi_se_usa_el_frame_completo(self):
        det = self.detector(TD.SesionBrillo(hw=(100, 200))).best(frame_con_cuadro(x=170, y=5, w=25, h=10))
        self.assertLess(abs(det.box[0] - 170), 2)

    def test_con_margen_lo_que_cae_en_el_margen_no_es_el_sujeto(self):
        """Con margen el modelo ve un poco más allá del ROI, pero el sujeto debe tener el centro DENTRO del ROI."""
        en_el_margen = frame_con_cuadro(x=152, y=40, w=16, h=16)  # justo a la derecha del ROI (que acaba en x = 150)
        self.assertIsNone(self.detector(TD.SesionBrillo(hw=(80, 140)), margin=20).best(en_el_margen, ROI))
        dentro = frame_con_cuadro(x=120, y=40, w=16, h=16)
        self.assertIsNotNone(self.detector(TD.SesionBrillo(hw=(80, 140)), margin=20).best(dentro, ROI))

    def test_el_margen_cambia_lo_que_ve_el_modelo(self):
        with mock.patch.object(D, "crop_rect", wraps=D.crop_rect) as espia:
            self.detector(margin=7).best(frame_con_cuadro(), ROI)
        self.assertEqual(espia.call_args.args[:2], (ROI, 7))

    def test_un_modelo_de_frames_completos_no_recorta_pero_respeta_el_roi(self):
        d = self.detector(TD.SesionBrillo(hw=(100, 200), notas={"recorte_roi": False}))
        with mock.patch.object(D, "crop_rect", wraps=D.crop_rect) as espia:
            self.assertIsNotNone(d.best(frame_con_cuadro(), ROI))
            self.assertIsNone(d.best(frame_con_cuadro(x=170, y=5, w=25, h=10), ROI))  # lo ve, pero su centro no está en el ROI
        espia.assert_not_called()

    def test_el_color_se_convierte_de_bgr_a_rgb(self):
        self.assertIsNotNone(self.detector().best(frame_con_cuadro(color=(0, 0, 255)), ROI))  # rojo en BGR
        self.assertIsNone(self.detector().best(frame_con_cuadro(color=(255, 0, 0)), ROI))  # azul: el juguete mira el canal R

    def test_varias_detecciones_de_mayor_a_menor_confianza(self):
        d = self.detector(TD.SesionDos(hw=(100, 200)))
        dets = d.detect(np.zeros((100, 200, 3), np.uint8))
        self.assertEqual(len(dets), 2)
        self.assertGreater(dets[0].score, dets[1].score)
        self.assertLess(abs(dets[0].center[0] - 150), 1.5)  # la de mayor confianza está en (0.75, 0.75)
        self.assertEqual(d.best(np.zeros((100, 200, 3), np.uint8)), dets[0])

    def test_el_umbral_y_su_cambio_por_llamada(self):
        d = self.detector(TD.SesionBrillo(hw=(60, 100), logit=0.0))  # confianza 0.5 exacta
        self.assertIsNone(d.best(frame_con_cuadro(), ROI))  # el umbral es estricto
        self.assertIsNotNone(d.best(frame_con_cuadro(), ROI, threshold=0.4))
        self.assertIsNotNone(self.detector(TD.SesionBrillo(hw=(60, 100), logit=0.0), threshold=0.4).best(frame_con_cuadro(), ROI))

    def test_una_caja_fuera_del_recorte_no_cuenta(self):
        self.assertEqual(self.detector(TD.SesionBrillo(hw=(60, 100), desvio=(2.0, 0.0))).detect(frame_con_cuadro(), ROI), [])
        # también sin ROI (ahí no actúa la regla del centro): una caja recortada a ancho cero no es una detección
        self.assertEqual(self.detector(TD.SesionBrillo(hw=(100, 200), desvio=(2.0, 0.0))).detect(frame_con_cuadro()), [])

    def test_sin_nada_que_detectar(self):
        self.assertEqual(self.detector().detect(np.zeros((100, 200, 3), np.uint8), ROI), [])

    def test_un_roi_fuera_del_frame(self):
        with self.assertRaisesRegex(ValueError, "queda fuera del frame"):
            self.detector().best(frame_con_cuadro(), (500, 500, 50, 50))


# ======================================================================
class CajaEnElRoiTests(unittest.TestCase):
    def test_se_resta_el_origen_y_se_redondea_hacia_afuera(self):
        det = D.Detection((100.4, 40.6, 29.2, 19.1), 0.9)  # x 100.4..129.6, y 40.6..59.7
        self.assertEqual(S.box_in_roi(det, ROI), (50, 20, 30, 20))  # x 50.4..79.6 -> 50..80; y 20.6..39.7 -> 20..40

    def test_se_limita_al_recorte(self):
        self.assertEqual(S.box_in_roi(D.Detection((40, 10, 30, 30), 0.9), ROI), (0, 0, 20, 20))  # sobresale por arriba y por la izquierda
        self.assertEqual(S.box_in_roi(D.Detection((130, 60, 40, 40), 0.9), ROI), (80, 40, 20, 20))  # sobresale por abajo y por la derecha

    def test_sin_tamano_es_none(self):
        self.assertIsNone(S.box_in_roi(D.Detection((300, 300, 20, 20), 0.9), ROI))
        self.assertIsNone(S.box_in_roi(D.Detection((149.5, 40, 20, 20), 0.9), ROI))  # solo 1 px dentro
        self.assertIsNotNone(S.box_in_roi(D.Detection((148, 40, 20, 20), 0.9), ROI))  # 2 px: lo mínimo


class PropuestaEnPantallaTests(unittest.TestCase):
    def pix(self, img, color):
        ys, xs = np.nonzero(np.all(img == np.array(color, np.uint8), axis=2))
        return None if len(xs) == 0 else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def test_la_franja_va_arriba_y_no_tapa_la_imagen(self):
        recorte = np.full((60, 400, 3), 90, np.uint8)
        img = S.draw_proposal(recorte, (300, 30, 40, 20), 0.9)
        self.assertEqual(img.shape, (60 + S.BANDA, 400, 3))
        self.assertTrue((img[S.BANDA:, :250] == 90).all())  # la imagen original, intacta, debajo de la franja
        self.assertGreater(int((img[:S.BANDA] > 200).sum()), 50)  # en la franja hay texto

    def test_la_caja_se_dibuja_desplazada_por_la_franja(self):
        img = S.draw_proposal(np.full((120, 400, 3), 90, np.uint8), (100, 30, 60, 40), 0.9)
        x0, y0, x1, y1 = self.pix(img, (0, 220, 0))
        self.assertLessEqual(abs(x0 - 100), 2)
        self.assertLessEqual(abs(x1 - 160), 2)
        self.assertLessEqual(abs(y0 - (30 + S.BANDA)), 2)
        self.assertLessEqual(abs(y1 - (70 + S.BANDA)), 2)

    def test_verde_si_la_confianza_es_alta_y_amarillo_si_es_baja(self):
        recorte = np.full((120, 400, 3), 90, np.uint8)
        self.assertIsNotNone(self.pix(S.draw_proposal(recorte, (100, 30, 60, 40), 0.80), (0, 220, 0)))
        self.assertIsNone(self.pix(S.draw_proposal(recorte, (100, 30, 60, 40), 0.80), (0, 220, 255)))
        self.assertIsNotNone(self.pix(S.draw_proposal(recorte, (100, 30, 60, 40), 0.79), (0, 220, 255)))
        self.assertIsNone(self.pix(S.draw_proposal(recorte, (100, 30, 60, 40), 0.79), (0, 220, 0)))

    def test_no_modifica_la_imagen_que_recibe(self):
        recorte = np.full((120, 400, 3), 90, np.uint8)
        S.draw_proposal(recorte, (100, 30, 60, 40), 0.9)
        self.assertTrue((recorte == 90).all())

    def test_teclas(self):
        for k in (13, 10, 32, ord("a"), ord("A"), 13 + 256):  # algunos sistemas devuelven bits altos
            self.assertEqual(S.key_to_choice(k), S.ACEPTAR, msg=str(k))
        for k in (ord("m"), ord("M")):
            self.assertEqual(S.key_to_choice(k), S.MANUAL)
        for k in (27, ord("q"), ord("Q")):
            self.assertEqual(S.key_to_choice(k), S.CANCELAR)
        for k in (ord("x"), ord("1"), 0):
            self.assertIsNone(S.key_to_choice(k))

    def ventana(self, teclas, visible=1.0):
        teclas = iter(teclas)
        with mock.patch.object(S.cv2, "namedWindow"), mock.patch.object(S.cv2, "resizeWindow"), mock.patch.object(S.cv2, "imshow") as show, \
                mock.patch.object(S.cv2, "waitKey", side_effect=lambda ms: next(teclas)), \
                mock.patch.object(S.cv2, "getWindowProperty", return_value=visible), mock.patch.object(S.cv2, "destroyWindow") as cerrar:
            r = S.confirm_subject(np.full((60, 400, 3), 90, np.uint8), (10, 10, 20, 20), 0.9)
        return r, show, cerrar

    def test_la_ventana_espera_una_tecla_con_significado(self):
        r, show, cerrar = self.ventana([-1, -1, ord("x"), 13])
        self.assertEqual(r, S.ACEPTAR)
        self.assertEqual(show.call_args.args[1].shape, (60 + S.BANDA, 400, 3))  # lo que se muestra es la propuesta dibujada
        cerrar.assert_called_once()
        self.assertEqual(self.ventana([ord("m")])[0], S.MANUAL)
        self.assertEqual(self.ventana([27])[0], S.CANCELAR)

    def test_cerrar_la_ventana_equivale_a_marcar_a_mano(self):
        r, _, cerrar = self.ventana([-1, -1], visible=0.0)
        self.assertEqual(r, S.MANUAL)
        cerrar.assert_called_once()


# ======================================================================
class ElegirSujetoTests(Base):
    MANUAL_BOX = (1, 2, 3, 4)

    def elegir(self, frame=None, decision=S.ACEPTAR, modelo="usar", sesion=None, fabrica=None, roi=ROI):
        self.llamadas_manual, self.confirmaciones, msgs = 0, [], []

        def manual():
            self.llamadas_manual += 1
            return self.MANUAL_BOX

        def confirmar(recorte, bbox, score):
            self.confirmaciones.append((recorte.shape, bbox, score))
            return decision
        sesion = sesion or TD.SesionBrillo(hw=(60, 100))
        fabrica = fabrica or (lambda ruta: D.OnnxDetector(ruta, session_factory=lambda r, h: sesion))
        bbox, info = S.choose_subject(frame_con_cuadro() if frame is None else frame, roi, model_path=self.modelo if modelo == "usar" else modelo,
                                      manual_select=manual, confirm=confirmar, detector_factory=fabrica, out=msgs.append)
        return bbox, info, "\n".join(msgs)

    def test_sin_modelo_configurado_va_directo_a_mano_y_en_silencio(self):
        for modelo in ("", None):
            bbox, info, texto = self.elegir(modelo=modelo)
            self.assertEqual((bbox, self.llamadas_manual, texto), (self.MANUAL_BOX, 1, ""))
            self.assertEqual((info["metodo"], info["motivo"], info["modelo"]), ("manual", "sin modelo configurado", ""))
            self.assertEqual(self.confirmaciones, [])

    def test_propuesta_aceptada(self):
        bbox, info, texto = self.elegir()
        x, y, w, h = bbox  # en coordenadas del recorte: el cuadro está en (100, 40) del frame y el ROI empieza en (50, 20)
        self.assertLess(abs(x - 50), 3)
        self.assertLess(abs(y - 20), 3)
        self.assertLess(abs(w - 30), 4)
        self.assertLess(abs(h - 20), 4)
        self.assertEqual(self.llamadas_manual, 0)
        self.assertEqual((info["metodo"], info["motivo"], info["modelo"], info["umbral"], info["margen_roi_px"]), ("detector", "propuesta aceptada", "raton.onnx", 0.5, 0))
        self.assertGreater(info["confianza"], 0.9)
        self.assertIn("El detector propone al sujeto con confianza 0.98.", texto)
        self.assertEqual(self.confirmaciones[0][0], (60, 100, 3))  # se confirma sobre el recorte del ROI (100 x 60)
        self.assertEqual(self.confirmaciones[0][1], bbox)

    def test_la_persona_prefiere_marcar_a_mano(self):
        bbox, info, _ = self.elegir(decision=S.MANUAL)
        self.assertEqual((bbox, self.llamadas_manual), (self.MANUAL_BOX, 1))
        self.assertEqual((info["metodo"], info["motivo"]), ("manual", "propuesta rechazada: se marcó a mano"))
        self.assertGreater(info["confianza"], 0.9)  # queda constancia de lo que proponía el detector

    def test_cancelar_no_abre_la_seleccion_manual(self):
        bbox, info, _ = self.elegir(decision=S.CANCELAR)
        self.assertEqual((bbox, self.llamadas_manual, info["metodo"]), (None, 0, "cancelado"))

    def test_una_decision_desconocida_se_trata_como_a_mano(self):
        bbox, info, _ = self.elegir(decision="otra cosa")
        self.assertEqual((bbox, info["metodo"]), (self.MANUAL_BOX, "manual"))

    def test_si_el_detector_no_ve_al_ratón_se_marca_a_mano(self):
        bbox, info, texto = self.elegir(frame=np.zeros((100, 200, 3), np.uint8))
        self.assertEqual((bbox, self.llamadas_manual, self.confirmaciones), (self.MANUAL_BOX, 1, []))
        self.assertEqual((info["metodo"], info["motivo"], info["confianza"]), ("manual", "el detector no encontró al sujeto", None))
        self.assertIn("no encontró al ratón en el frame de muestra (confianza mínima 0.5)", texto)

    def test_si_el_modelo_no_abre_se_avisa_y_se_sigue_a_mano(self):
        def falla(ruta):
            raise D.DetectorError("No se pudo abrir el modelo raton.onnx: dañado")
        bbox, info, texto = self.elegir(fabrica=falla)
        self.assertEqual((bbox, self.llamadas_manual), (self.MANUAL_BOX, 1))
        self.assertIn("Detector no disponible (No se pudo abrir el modelo raton.onnx: dañado). Selecciona el sujeto a mano.", texto)
        self.assertIn("detector no disponible", info["motivo"])
        self.assertEqual(info["modelo"], "raton.onnx")

    def test_un_modelo_inexistente_no_bloquea(self):
        bbox, info, texto = self.elegir(modelo=os.path.join(self.tmp.name, "no_existe.onnx"), fabrica=lambda ruta: D.OnnxDetector(ruta))
        self.assertEqual(bbox, self.MANUAL_BOX)
        self.assertIn("No existe el modelo", texto)

    def test_cualquier_fallo_al_detectar_cae_a_la_seleccion_manual(self):
        class Rota(TD.SesionBrillo):
            def run(self, nombres, feed):
                raise RuntimeError("fallo interno del modelo")
        bbox, info, texto = self.elegir(sesion=Rota(hw=(60, 100)))
        self.assertEqual((bbox, self.llamadas_manual), (self.MANUAL_BOX, 1))
        self.assertIn("fallo interno del modelo", texto)
        bbox, info, texto = self.elegir(roi=(500, 500, 50, 50))  # ROI fuera del frame
        self.assertEqual(bbox, self.MANUAL_BOX)
        self.assertIn("queda fuera del frame", texto)

    def test_una_confianza_baja_se_propone_pero_se_advierte(self):
        bbox, info, texto = self.elegir(sesion=TD.SesionBrillo(hw=(60, 100), logit=1.0))  # 0.73
        self.assertIsNotNone(bbox)
        self.assertEqual(info["metodo"], "detector")
        self.assertIn("confianza 0.73 (baja: revisa bien la caja)", texto)

    def test_la_constancia_guarda_la_caja_propuesta_y_de_donde_viene_el_modelo(self):
        notas = {"corrida": "rfdetr_small_20261110_214151", "version_rfdetr": "1.12.0", "modelo": "small", "resolucion": 512, "otra": "x"}
        sesion = TD.SesionBrillo(hw=(60, 100), notas=notas)
        bbox, info, _ = self.elegir(sesion=sesion)
        self.assertEqual(info["caja_propuesta"], list(bbox))
        self.assertEqual(info["notas_modelo"], {k: notas[k] for k in ("corrida", "modelo", "version_rfdetr", "resolucion")})
        _, info, _ = self.elegir(sesion=TD.SesionBrillo(hw=(60, 100)), decision=S.MANUAL)
        self.assertIn("caja_propuesta", info)  # aunque se marque a mano, queda lo que proponía el detector
        self.assertEqual(info["notas_modelo"], {})  # un modelo sin notas
        _, info, _ = self.elegir(frame=np.zeros((100, 200, 3), np.uint8))
        self.assertNotIn("caja_propuesta", info)  # no hubo propuesta

    def test_registro_de_asistencia(self):
        info = {"metodo": "detector", "confianza": 0.86}
        rec = S.assistance_record(self.modelo, info)
        self.assertEqual(rec, {"modo": "proponer_sujeto", "sujeto": {"metodo": "detector", "confianza": 0.86}})
        info["metodo"] = "otro"
        self.assertEqual(rec["sujeto"]["metodo"], "detector")  # es una copia
        self.assertEqual(S.assistance_record(None, {"metodo": "manual"})["modo"], "ninguna")
        self.assertEqual(S.assistance_record("", {})["modo"], "ninguna")

    def test_descripcion_en_una_linea(self):
        d = S.describe_assistance
        acep = S.assistance_record("m.onnx", {"metodo": "detector", "motivo": "propuesta aceptada", "modelo": "rfdetr-small.onnx", "confianza": 0.861})
        self.assertEqual(d(acep, "fondo"), "Modo: fondo · asistencia del detector: proponer sujeto · sujeto propuesto por el detector y aceptado · modelo rfdetr-small.onnx, confianza 0.86")
        man = S.assistance_record("m.onnx", {"metodo": "manual", "motivo": "el detector no encontró al sujeto", "modelo": "rfdetr-small.onnx", "confianza": None})
        self.assertEqual(d(man), "asistencia del detector: proponer sujeto · sujeto marcado a mano (el detector no encontró al sujeto) · modelo rfdetr-small.onnx")
        self.assertEqual(d(S.assistance_record(None, {"metodo": "manual"}), "silueta"), "Modo: silueta · asistencia del detector: ninguna (sujeto marcado a mano)")
        self.assertEqual(d(None), "asistencia del detector: ninguna (sujeto marcado a mano)")
        canc = S.assistance_record("m.onnx", {"metodo": "cancelado", "motivo": "cancelado al confirmar la propuesta", "modelo": "x.onnx"})
        self.assertIn("· cancelado (cancelado al confirmar la propuesta) · modelo x.onnx", d(canc))

    def test_la_constancia_se_puede_guardar_como_json(self):
        import json
        for decision in (S.ACEPTAR, S.MANUAL, S.CANCELAR):
            _, info, _ = self.elegir(decision=decision)
            json.dumps(info)

    def test_de_punta_a_punta_con_la_escena_sintetica(self):
        """Un frame real de la escena de prueba: la caja propuesta, en coordenadas del recorte, cae sobre el círculo."""
        frames, truth = BG.make_frames(90, lost={30, 31, 32})
        roi = (20, 20, 260, 190)
        f = 50
        bbox, info, _ = self.elegir(frame=frames[f], sesion=TD.SesionBrillo(), roi=roi)
        self.assertEqual(info["metodo"], "detector")
        cx, cy = bbox[0] + bbox[2] / 2 + roi[0], bbox[1] + bbox[3] / 2 + roi[1]
        self.assertLess(np.hypot(cx - truth[f][0], cy - truth[f][1]), 3.0)
        bbox, info, _ = self.elegir(frame=frames[31], sesion=TD.SesionBrillo(), roi=roi)  # un frame donde el círculo no está
        self.assertEqual(info["motivo"], "el detector no encontró al sujeto")


if __name__ == "__main__":
    unittest.main()