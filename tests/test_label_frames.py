"""Pruebas de tools/label_frames.py.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_label_frames.py -v

La ventana real no se puede abrir en un entorno sin pantalla: la lógica (cajas, teclas, guardado, escala) se prueba
con eventos simulados, y el bucle de la ventana con funciones de OpenCV simuladas.
"""

import importlib
import json
import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from persistence.run_inputs import save_run_inputs
from tools import eval_tracking as E
from tools import label_frames as L
from tools import sample_frames as S
from video.reader import VideoReader


def _imp(name):
    try:
        return importlib.import_module(f"tests.{name}")
    except ImportError:
        return importlib.import_module(name)


RI = _imp("test_run_inputs")
BG = _imp("test_background_tracker")

DOWN, MOVE, UP = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP
FRAMES = [10, 20, 30, 40, 50]


def leer(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def escribir(path, texto):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(texto)


def crear_conjunto(base, w=100, h=80, frames=FRAMES, vid="v"):
    """Conjunto mínimo como el de tools/sample_frames.py: manifiesto + imágenes (un color distinto por frame)."""
    val = os.path.join(base, "validacion")
    os.makedirs(os.path.join(val, "images"), exist_ok=True)
    os.makedirs(os.path.join(val, "labels"), exist_ok=True)
    for k, f in enumerate(frames):
        cv2.imwrite(os.path.join(val, "images", f"{vid}_f{f:06d}.png"), np.full((h, w, 3), 40 + 30 * k, np.uint8))
    with open(os.path.join(val, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"videos": [{"id": vid, "ancho": w, "alto": h, "formato": "png", "frames": list(frames)}]}, fh)
    return val


def raton(lab, evento, x, y):
    """Evento de ratón en coordenadas de la imagen mostrada; la ventana real añade la barra de estado de arriba."""
    lab.on_mouse(evento, x, y + L.BAR_H)


def arrastrar(lab, x0, y0, x1, y1):
    raton(lab, DOWN, x0, y0)
    raton(lab, MOVE, (x0 + x1) / 2, (y0 + y1) / 2)
    raton(lab, UP, x1, y1)


def tecla(lab, ch):
    return lab.on_key(ord(ch) if isinstance(ch, str) else ch)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.val = crear_conjunto(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def lab(self, **kw):
        return L.Labeler(self.val, L.load_items(self.val), **kw)

    def etiqueta(self, f):
        return os.path.join(self.val, "labels", f"v_f{f:06d}.txt")


# ======================================================================
class CargaTests(Base):
    def test_sin_etiquetas_todo_pendiente_y_empieza_en_la_primera(self):
        lab = self.lab()
        self.assertEqual(lab.counts(), {L.PENDING: 5, L.BOX: 0, L.EMPTY: 0, L.DOUBTFUL: 0, "total": 5})
        self.assertEqual(lab.i, 0)

    def test_retoma_donde_se_quedo(self):
        escribir(self.etiqueta(10), "0 0.3 0.5 0.4 0.5\n")
        escribir(self.etiqueta(20), "")
        lab = self.lab()
        self.assertEqual(lab.i, 2)  # primera pendiente
        self.assertEqual(self.lab(start_at_pending=False).i, 0)

    def test_lee_cajas_con_numeros_calculados_a_mano(self):
        # W=100, H=80: x=(0.3-0.2)*100=10, y=(0.5-0.25)*80=20, w=0.4*100=40, h=0.5*80=40
        escribir(self.etiqueta(10), "0 0.3 0.5 0.4 0.5\n")
        escribir(self.etiqueta(20), "")
        lab = self.lab()
        estado, box = lab.states[0]
        self.assertEqual(estado, L.BOX)
        for got, want in zip(box, (10.0, 20.0, 40.0, 40.0)):
            self.assertAlmostEqual(got, want, places=5)
        self.assertEqual(lab.states[1], (L.EMPTY, None))

    def test_etiquetas_invalidas_quedan_pendientes_con_aviso_y_no_se_tocan_al_navegar(self):
        escribir(self.etiqueta(10), "basura")
        escribir(self.etiqueta(20), "0 .5 .5 .1 .1\n0 .2 .2 .1 .1\n")
        lab = self.lab()
        self.assertEqual([lab.states[0][0], lab.states[1][0]], [L.PENDING, L.PENDING])
        self.assertIn(0, lab.notes)
        self.assertIn(1, lab.notes)
        for ch in "dda":  # navegar no debe reescribirlas ni borrarlas
            tecla(lab, ch)
        self.assertEqual(leer(self.etiqueta(10)), "basura")
        self.assertIn("0 .2 .2", leer(self.etiqueta(20)))

    def test_una_etiqueta_invalida_se_reemplaza_al_dibujar(self):
        escribir(self.etiqueta(10), "basura")
        lab = self.lab()
        arrastrar(lab, 10, 20, 50, 60)
        self.assertEqual(leer(self.etiqueta(10)), "0 0.300000 0.500000 0.400000 0.500000\n")
        self.assertNotIn(0, lab.notes)

    def test_dudosas_se_cargan(self):
        escribir(os.path.join(self.val, "dudosas.txt"), "v_f000020\n")
        self.assertEqual(self.lab().states[1][0], L.DOUBTFUL)

    def test_las_imagenes_que_faltan_se_omiten(self):
        os.remove(os.path.join(self.val, "images", "v_f000030.png"))
        self.assertEqual([it.frame for it in L.load_items(self.val)], [10, 20, 40, 50])

    def test_errores(self):
        with self.assertRaises(FileNotFoundError):
            L.load_items(os.path.join(self.tmp.name, "no_existe"))
        with self.assertRaises(ValueError):
            L.Labeler(self.val, [])
        cv2.imwrite(os.path.join(self.val, "images", "v_f000010.png"), np.zeros((50, 50, 3), np.uint8))  # tamaño distinto al manifiesto
        with self.assertRaisesRegex(ValueError, "manifiesto"):
            L.load_items(self.val)

    def test_el_nombre_de_las_etiquetas_coincide_con_el_muestreador_y_el_evaluador(self):
        for vid, f in (("rata01", 42), ("a_b-c", 1234567), ("v", 0)):
            self.assertEqual(L.stem_of(vid, f), os.path.splitext(S.image_name(vid, f))[0])
            self.assertEqual(L.stem_of(vid, f), E.image_stem(vid, f))


# ======================================================================
class CajasTests(Base):
    def test_arrastrar_guarda_al_instante_el_texto_exacto(self):
        lab = self.lab()
        arrastrar(lab, 10, 20, 50, 60)
        # caja (10,20,40,40) en 100x80: cx=(10+20)/100=0.3, cy=(20+20)/80=0.5, w=0.4, h=0.5
        self.assertEqual(leer(self.etiqueta(10)), "0 0.300000 0.500000 0.400000 0.500000\n")
        self.assertEqual(lab.states[0], (L.BOX, (10.0, 20.0, 40.0, 40.0)))

    def test_se_puede_arrastrar_en_cualquier_direccion(self):
        lab = self.lab()
        arrastrar(lab, 50, 60, 10, 20)  # de abajo a la derecha hacia arriba a la izquierda
        self.assertEqual(lab.states[0][1], (10.0, 20.0, 40.0, 40.0))
        arrastrar(lab, 50, 20, 10, 60)
        self.assertEqual(lab.states[0][1], (10.0, 20.0, 40.0, 40.0))

    def test_se_recorta_a_la_imagen(self):
        lab = self.lab()
        arrastrar(lab, -20, -10, 150, 100)
        self.assertEqual(lab.states[0][1], (0.0, 0.0, 100.0, 80.0))
        self.assertEqual(leer(self.etiqueta(10)), "0 0.500000 0.500000 1.000000 1.000000\n")

    def test_un_clic_accidental_no_crea_etiqueta(self):
        lab = self.lab()
        arrastrar(lab, 30, 30, 32, 32)  # 2x2 px
        self.assertEqual(lab.states[0], (L.PENDING, None))
        self.assertFalse(os.path.exists(self.etiqueta(10)))
        self.assertIn("pequena", lab.msg)
        arrastrar(lab, 30, 30, 33, 33)  # 3x3 px: ya vale
        self.assertEqual(lab.states[0][0], L.BOX)

    def test_dibujar_de_nuevo_reemplaza_y_deja_una_sola_caja(self):
        lab = self.lab()
        arrastrar(lab, 10, 20, 50, 60)
        arrastrar(lab, 5, 5, 25, 45)
        self.assertEqual(lab.states[0][1], (5.0, 5.0, 20.0, 40.0))
        self.assertEqual(len(leer(self.etiqueta(10)).strip().splitlines()), 1)

    def test_soltar_sin_haber_pulsado_no_hace_nada(self):
        lab = self.lab()
        raton(lab, UP, 50, 60)
        self.assertEqual(lab.states[0], (L.PENDING, None))

    def test_no_deja_archivos_temporales(self):
        lab = self.lab()
        arrastrar(lab, 10, 20, 50, 60)
        tecla(lab, "x")
        self.assertEqual([f for f in os.listdir(os.path.join(self.val, "labels")) if f.endswith(".tmp")], [])

    def test_con_escala_las_coordenadas_de_pantalla_se_convierten_a_las_de_la_imagen(self):
        base = os.path.join(self.tmp.name, "grande")
        val = crear_conjunto(base, w=2000, h=1000, frames=[1, 2])
        lab = L.Labeler(val, L.load_items(val), max_w=1000, max_h=800)  # escala 0.5
        arrastrar(lab, 100, 100, 300, 200)  # en pantalla
        self.assertEqual(lab.states[0][1], (200.0, 200.0, 400.0, 200.0))  # en la imagen
        # cx=(200+200)/2000=0.2, cy=(200+100)/1000=0.3, w=0.2, h=0.2
        self.assertEqual(leer(os.path.join(val, "labels", "v_f000001.txt")), "0 0.200000 0.300000 0.200000 0.200000\n")

    def test_escala_con_decimales_redondea_a_pixel(self):
        base = os.path.join(self.tmp.name, "grande2")
        val = crear_conjunto(base, w=2000, h=1000, frames=[1])
        lab = L.Labeler(val, L.load_items(val), max_w=1000, max_h=800)
        arrastrar(lab, 100.5, 99.5, 300.2, 200.4)
        x, y, w, h = lab.states[0][1]
        self.assertTrue(all(float(v).is_integer() for v in (x, y, w, h)))
        self.assertAlmostEqual(x, 201.0, delta=1.0)


# ======================================================================
class TeclasTests(Base):
    def test_x_escribe_etiqueta_vacia_y_pasa_a_la_siguiente(self):
        lab = self.lab()
        tecla(lab, "x")
        self.assertEqual(leer(self.etiqueta(10)), "")
        self.assertTrue(os.path.exists(self.etiqueta(10)))  # vacío pero EXISTE: significa «sin ratón», no «sin etiquetar»
        self.assertEqual(lab.states[0], (L.EMPTY, None))
        self.assertEqual(lab.i, 1)

    def test_x_en_una_imagen_con_caja_la_reemplaza_por_vacia(self):
        lab = self.lab()
        arrastrar(lab, 10, 20, 50, 60)
        tecla(lab, "x")
        self.assertEqual(leer(self.etiqueta(10)), "")

    def test_d_a_espacio_y_mayusculas(self):
        lab = self.lab()
        tecla(lab, "d")
        self.assertEqual(lab.i, 1)
        tecla(lab, 32)
        self.assertEqual(lab.i, 2)
        tecla(lab, "A")
        self.assertEqual(lab.i, 1)
        tecla(lab, "D")
        self.assertEqual(lab.i, 2)

    def test_limites_de_la_lista(self):
        lab = self.lab()
        tecla(lab, "a")
        self.assertEqual(lab.i, 0)
        self.assertIn("primera", lab.msg)
        lab.goto(4)
        tecla(lab, "d")
        self.assertEqual(lab.i, 4)
        self.assertIn("ultima", lab.msg)
        tecla(lab, "x")  # en la última: guarda y no se sale de la lista
        self.assertEqual((lab.i, lab.states[4][0]), (4, L.EMPTY))

    def test_navegar_sin_decidir_no_etiqueta_nada(self):
        lab = self.lab()
        for ch in "dddd":
            tecla(lab, ch)
        self.assertEqual(os.listdir(os.path.join(self.val, "labels")), [])
        self.assertEqual(lab.counts()[L.PENDING], 5)

    def test_u_borra_la_etiqueta(self):
        lab = self.lab()
        arrastrar(lab, 10, 20, 50, 60)
        tecla(lab, "u")
        self.assertFalse(os.path.exists(self.etiqueta(10)))
        self.assertEqual(lab.states[0], (L.PENDING, None))
        tecla(lab, "u")  # borrar lo que no existe no falla
        self.assertEqual(lab.states[0], (L.PENDING, None))

    def test_k_deja_sin_etiquetar_y_anota_en_dudosas_ordenado(self):
        lab = self.lab()
        arrastrar(lab, 10, 20, 50, 60)
        tecla(lab, "k")  # sobre una imagen ya etiquetada: la etiqueta desaparece
        self.assertFalse(os.path.exists(self.etiqueta(10)))
        lab.goto(2)
        tecla(lab, "k")
        lab.goto(1)
        tecla(lab, "k")
        self.assertEqual(leer(os.path.join(self.val, "dudosas.txt")), "v_f000010\nv_f000020\nv_f000030\n")
        self.assertEqual(lab.counts()[L.DOUBTFUL], 3)

    def test_etiquetar_una_dudosa_la_saca_de_la_lista(self):
        lab = self.lab()
        tecla(lab, "k")
        lab.goto(0)
        arrastrar(lab, 10, 20, 50, 60)
        self.assertEqual(leer(os.path.join(self.val, "dudosas.txt")), "")
        self.assertEqual(lab.states[0][0], L.BOX)

    def test_n_salta_a_la_proxima_pendiente_hacia_adelante(self):
        lab = self.lab()
        tecla(lab, "n")  # estando en la 0, que está pendiente, va a la siguiente pendiente (no se queda en la misma)
        self.assertEqual(lab.i, 1)
        lab.goto(1)
        tecla(lab, "x")  # 20 vacía (ahora en la 30)
        lab.goto(0)
        tecla(lab, "n")
        self.assertEqual(lab.i, 2)  # salta la 20, que ya está hecha

    def test_n_da_la_vuelta_a_la_lista(self):
        lab = self.lab()
        lab.goto(1)
        for _ in range(4):
            tecla(lab, "x")  # 20, 30, 40 y 50 vacías: solo queda pendiente la 10 (índice 0)
        lab.goto(4)
        tecla(lab, "n")
        self.assertEqual(lab.i, 0)  # no hay pendientes después de la 4: vuelve al principio

    def test_n_avisa_cuando_no_queda_ninguna(self):
        lab = self.lab()
        for _ in range(5):
            tecla(lab, "x")
        lab.goto(2)
        tecla(lab, "n")
        self.assertIn("no quedan", lab.msg)
        self.assertEqual(lab.i, 2)  # no se mueve

    def test_salir_con_q_y_esc(self):
        lab = self.lab()
        self.assertEqual(tecla(lab, "q"), "quit")
        self.assertEqual(tecla(lab, "Q"), "quit")
        self.assertEqual(lab.on_key(27), "quit")
        self.assertIsNone(tecla(lab, "d"))
        self.assertIsNone(lab.on_key(250))  # una tecla cualquiera no hace nada

    def test_ayuda(self):
        lab = self.lab()
        sin = lab.render()
        tecla(lab, "h")
        self.assertTrue(lab.show_help)
        self.assertFalse(np.array_equal(sin, lab.render()))
        tecla(lab, "h")
        self.assertFalse(lab.show_help)

    def test_se_retoma_con_el_mismo_estado(self):
        lab = self.lab()
        arrastrar(lab, 10, 20, 50, 60)  # 10: caja
        tecla(lab, "x")  # 10: vacía (reemplaza) -> pasa a 20
        arrastrar(lab, 5, 5, 25, 45)  # 20: caja
        lab.goto(3)
        tecla(lab, "k")  # 40: dudosa
        otra = self.lab()
        self.assertEqual([s for s, _ in otra.states], [L.EMPTY, L.BOX, L.PENDING, L.DOUBTFUL, L.PENDING])
        self.assertEqual(otra.i, 2)
        self.assertEqual(otra.states[1][1], (5.0, 5.0, 20.0, 40.0))


# ======================================================================
class DibujoTests(Base):
    VERDE, AMARILLO, BARRA = (0, 200, 0), (0, 220, 255), (30, 30, 30)

    def hay(self, img, color, fila, cols):
        return bool(np.all(img[fila, cols] == color, axis=1).any())

    def test_tamano_y_barras(self):
        img = self.lab().render()
        self.assertEqual(img.shape, (80 + 2 * L.BAR_H, 100, 3))  # imagen + barra arriba + barra abajo
        self.assertTrue(np.all(img[3, 3] == self.BARRA))  # barra de estado arriba
        self.assertTrue(np.all(img[-3, 3] == self.BARRA))  # y barra de ayuda abajo
        self.assertTrue(np.all(img[L.BAR_H + 40, 50] == 40))  # en medio, la imagen sin tocar (el primer gris es 40)

    def test_la_caja_guardada_se_dibuja_en_verde_en_su_sitio(self):
        lab = self.lab()
        arrastrar(lab, 10, 20, 50, 60)
        img = lab.render()
        self.assertTrue(self.hay(img, self.VERDE, L.BAR_H + 40, slice(8, 13)))  # borde izquierdo, a media altura
        self.assertTrue(np.any(np.all(img[L.BAR_H + 18:L.BAR_H + 23, 30] == self.VERDE, axis=1)))  # borde superior

    def test_la_caja_desaparece_al_borrar_y_no_se_ensucia_la_imagen_base(self):
        lab = self.lab()
        arrastrar(lab, 10, 20, 50, 60)
        con = lab.render()
        tecla(lab, "u")
        sin = lab.render()
        self.assertFalse(np.array_equal(con, sin))
        self.assertFalse(self.hay(sin, self.VERDE, L.BAR_H + 40, slice(0, 100)))  # ni rastro de verde

    def test_mientras_se_arrastra_se_ve_la_caja_en_amarillo(self):
        lab = self.lab()
        raton(lab, DOWN, 10, 20)
        raton(lab, MOVE, 50, 60)
        img = lab.render()
        self.assertTrue(self.hay(img, self.AMARILLO, L.BAR_H + 40, slice(8, 13)))
        self.assertEqual(lab.states[0], (L.PENDING, None))  # todavía no se guardó nada
        self.assertFalse(os.path.exists(self.etiqueta(10)))

    def test_una_caja_en_el_borde_superior_de_la_imagen_se_ve_completa(self):
        """Regresión: antes la barra de estado se dibujaba ENCIMA y tapaba los primeros 24 px de la imagen."""
        lab = self.lab()
        arrastrar(lab, 10, 0, 50, 30)
        img = lab.render()
        self.assertTrue(np.any(np.all(img[L.BAR_H:L.BAR_H + 2, 30] == self.VERDE, axis=1)))  # el borde superior (y = 0) está a la vista

    def test_un_clic_sobre_la_barra_se_recorta_al_borde_de_la_imagen(self):
        lab = self.lab()
        lab.on_mouse(DOWN, 10, 5)  # y = 5 cae en la barra de arriba (alto 24): equivale a y < 0 en la imagen
        lab.on_mouse(UP, 50, L.BAR_H + 30)
        self.assertEqual(lab.states[0][1], (10.0, 0.0, 40.0, 30.0))

    def test_cada_imagen_muestra_su_propio_fondo(self):
        lab = self.lab()
        a = lab.render()[L.BAR_H + 40, 50].copy()
        tecla(lab, "d")
        b = lab.render()[L.BAR_H + 40, 50]
        self.assertFalse(np.array_equal(a, b))  # el caché no confunde unas imágenes con otras

    def test_imagen_grande_se_reduce_a_la_ventana(self):
        base = os.path.join(self.tmp.name, "grande3")
        val = crear_conjunto(base, w=2000, h=1000, frames=[1])
        lab = L.Labeler(val, L.load_items(val), max_w=1000, max_h=800)
        self.assertEqual(lab.render().shape, (500 + 2 * L.BAR_H, 1000, 3))
        arrastrar(lab, 100, 100, 300, 200)
        img = lab.render()
        self.assertTrue(self.hay(img, self.VERDE, L.BAR_H + 150, slice(98, 103)))  # borde izquierdo de la caja, en pantalla


# ======================================================================
class VentanaSimuladaTests(Base):
    """El bucle de la ventana, con las funciones de OpenCV sustituidas (aquí no hay pantalla)."""

    def correr(self, teclas, eventos=None, imshow_falla=False):
        lab = self.lab()
        captura = {}
        llamadas = []
        eventos = eventos or {}
        secuencia = iter(teclas)
        paso = {"n": 0}

        def wait(_ms):
            paso["n"] += 1
            for ev in eventos.get(paso["n"], []):
                evento, x, y = ev
                captura["callback"](evento, x, y + L.BAR_H, 0, None)  # (evento, x, y, flags, param) como lo manda OpenCV
            return next(secuencia)

        def imshow(nombre, img):
            if imshow_falla:
                raise RuntimeError("fallo al dibujar")
            llamadas.append((nombre, img.shape))

        with mock.patch.object(cv2, "namedWindow") as nw, \
                mock.patch.object(cv2, "setMouseCallback", side_effect=lambda w, cb: captura.__setitem__("callback", cb)), \
                mock.patch.object(cv2, "imshow", side_effect=imshow), mock.patch.object(cv2, "waitKey", side_effect=wait), \
                mock.patch.object(cv2, "destroyAllWindows") as destroy:
            try:
                L.run_gui(lab)
                error = None
            except RuntimeError as exc:
                error = exc
        return lab, nw, destroy, llamadas, error

    def test_el_bucle_dibuja_atiende_el_raton_y_las_teclas_y_cierra(self):
        eventos = {2: [(DOWN, 10, 20), (MOVE, 30, 40), (UP, 50, 60)]}
        lab, nw, destroy, llamadas, _ = self.correr([255, 255, ord("x"), 255, ord("q")], eventos)
        self.assertEqual(lab.states[0][0], L.EMPTY)  # el ratón dibujó una caja y luego x la reemplazó por «sin ratón»
        self.assertEqual(leer(self.etiqueta(10)), "")
        nw.assert_called_once()
        destroy.assert_called_once()
        self.assertEqual(len(llamadas), 5)
        self.assertEqual(llamadas[0][1], (80 + 2 * L.BAR_H, 100, 3))

    def test_el_raton_conectado_al_bucle_guarda_la_caja(self):
        eventos = {1: [(DOWN, 10, 20), (UP, 50, 60)]}
        lab, *_ = self.correr([ord("q")], eventos)
        self.assertEqual(leer(self.etiqueta(10)), "0 0.300000 0.500000 0.400000 0.500000\n")

    def test_si_algo_falla_la_ventana_se_cierra_igual(self):
        _, _, destroy, _, error = self.correr([ord("q")], imshow_falla=True)
        self.assertIsNotNone(error)
        destroy.assert_called_once()


# ======================================================================
class LineaDeComandosTests(Base):
    def test_solo_resumen_no_abre_ventana(self):
        escribir(self.etiqueta(10), "0 0.3 0.5 0.4 0.5\n")
        escribir(self.etiqueta(20), "")
        msgs = []

        def no_debe_abrirse(_lab):
            raise AssertionError("se abrió la ventana")

        self.assertEqual(L.main([self.val, "--solo-resumen"], gui=no_debe_abrirse, out=msgs.append), 0)
        self.assertIn("1 con caja, 1 sin ratón, 0 dudosas, 3 pendientes (de 5 imágenes)", msgs[0])

    def test_errores(self):
        msgs = []
        self.assertEqual(L.main([os.path.join(self.tmp.name, "no"), "--solo-resumen"], out=msgs.append), 2)
        self.assertIn("sample_frames", msgs[0])
        vacio = crear_conjunto(os.path.join(self.tmp.name, "v2"), frames=[])
        self.assertEqual(L.main([vacio, "--solo-resumen"], out=msgs.append), 2)

    def test_empieza_en_la_primera_pendiente_o_en_el_inicio(self):
        escribir(self.etiqueta(10), "")
        vistos = []
        L.main([self.val], gui=lambda lab: vistos.append(lab.i), out=lambda s: None)
        L.main([self.val, "--desde-inicio"], gui=lambda lab: vistos.append(lab.i), out=lambda s: None)
        self.assertEqual(vistos, [1, 0])

    def test_mensaje_final_segun_lo_que_falte(self):
        msgs = []

        def etiquetar_todo(lab):
            for _ in range(5):
                tecla(lab, "x")

        L.main([self.val], gui=etiquetar_todo, out=msgs.append)
        self.assertTrue(any("Listo. Siguiente: python tools/eval_tracking.py" in m for m in msgs))
        msgs.clear()
        L.main([self.val, "--desde-inicio"], gui=lambda lab: None, out=msgs.append)
        self.assertIn("Estado final: 0 con caja, 5 sin ratón", msgs[-2])  # lo anterior se conservó

    def test_faltan(self):
        msgs = []
        L.main([self.val], gui=lambda lab: tecla(lab, "x"), out=msgs.append)
        self.assertTrue(any(m.startswith("Faltan 4.") for m in msgs))

    def test_argumentos_de_tamano(self):
        vistos = []
        L.main([self.val, "--ancho-max", "50", "--alto-max", "40"], gui=lambda lab: vistos.append(lab.render().shape), out=lambda s: None)
        self.assertEqual(vistos, [(40 + 2 * L.BAR_H, 50, 3)])  # 100x80 -> escala 0.5, más las dos barras


# ======================================================================
class EtiquetarYEvaluarTests(unittest.TestCase):
    """De punta a punta: se etiqueta con el ratón simulado (con la ventana reducida a la mitad) y el evaluador lo da por bueno."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        RI.Escenas.preparar(cls.tmp.name)
        cls.frames, cls.truth = BG.make_frames(90, lost={30, 31, 32})
        with VideoReader(RI.Escenas.p_bg) as r:
            job = RI.Escenas.job_fondo(r)
            cls.carpeta = save_run_inputs(os.path.join(cls.tmp.name, "exp"), job, video_path=RI.Escenas.p_bg, options=RI.Escenas.OPT_FONDO,
                                          sample_idx=0, app_version="1.0", log=None, now=RI.AHORA)
        cls.val = os.path.join(cls.tmp.name, "validacion")
        S.muestrear(S.fuentes_desde_args([], [cls.carpeta], 1, 0.0, None), cls.val, por_video=100, out=lambda s: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_lo_etiquetado_con_la_ventana_lo_evalua_el_evaluador(self):
        lab = L.Labeler(self.val, L.load_items(self.val), max_w=160, max_h=120)  # 320x240 -> escala 0.5
        self.assertAlmostEqual(lab._scale_for(320, 240), 0.5)
        sin_raton = 0
        for k, it in enumerate(lab.items):
            lab.goto(k)
            t = self.truth[it.frame]
            if t is None:
                tecla(lab, "x")
                sin_raton += 1
            else:
                cx, cy = int(t[0]), int(t[1])
                arrastrar(lab, (cx - 13) * 0.5, (cy - 13) * 0.5, (cx + 14) * 0.5, (cy + 14) * 0.5)  # círculo de radio 13, en pantalla
        self.assertGreaterEqual(sin_raton, 1)
        self.assertEqual(lab.counts()[L.PENDING], 0)
        msgs = []
        self.assertEqual(E.run(self.val, [self.carpeta], sin_filtros=True, out=msgs.append), 0)
        carpeta = sorted(d for d in os.listdir(self.val) if d.startswith("evaluacion_"))[-1]
        with open(os.path.join(self.val, carpeta, "resultados.json"), encoding="utf-8") as fh:
            s = json.load(fh)["corridas"][0]["resumen"]
        self.assertEqual(s["evaluados"], len(lab.items))
        self.assertEqual((s["acierto_pct"], s["posicion_dentro_pct"]), (100.0, 100.0))
        self.assertEqual((s["perdidos"], s["lugar_equivocado"], s["falsos_positivos"]), (0, 0, 0))
        self.assertLess(s["error_centro_max_px"], 2.0)
        self.assertFalse(any("sin etiqueta" in m and not m.startswith("Etiquetas: ") for m in msgs))


if __name__ == "__main__":
    unittest.main()