"""Pruebas de la lectura EXACTA de video/reader.py (iter_frames y read_exact).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_reader_exact.py -v

Un VideoCapture de juguete reproduce lo medido con Grupo1R3S1.mp4: saltar con set(POS_FRAMES, n) deja el video en n + 27.
Cada frame lleva su número real pintado, así que se puede comprobar qué frame se leyó de verdad.
"""

import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from video import reader as R

N = 300
CORRIMIENTO = 27  # lo que se midió en el video real


class CapJuguete:
    """Un VideoCapture cuyo salto cae ``k`` frames después (salvo a 0). El píxel (0, 0) dice el número REAL del frame."""

    creados = 0

    def __init__(self, path, n=N, k=CORRIMIENTO, abre=True):
        CapJuguete.creados += 1
        self.n, self.k, self.pos, self.abierto = n, k, 0, abre
        self.saltos = []

    def isOpened(self):
        return self.abierto

    def release(self):
        self.abierto = False

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 14.3, cv2.CAP_PROP_FRAME_COUNT: float(self.n), cv2.CAP_PROP_FRAME_WIDTH: 4.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 2.0}.get(prop, 0.0)

    def set(self, prop, v):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.saltos.append(int(v))
        self.pos = 0 if v <= 0 else min(self.n, int(v) + self.k)
        return True

    def grab(self):
        if self.pos >= self.n:
            return False
        self.pos += 1
        return True

    def read(self):
        if self.pos >= self.n:
            return False, None
        f = np.zeros((2, 4, 3), np.uint16)
        f[0, 0, 0] = self.pos
        self.pos += 1
        return True, f


def real(frame):
    return int(frame[0, 0, 0])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "v.mp4")
        with open(self.path, "w") as fh:
            fh.write("x")
        CapJuguete.creados = 0
        self.parche = mock.patch.object(R.cv2, "VideoCapture", side_effect=lambda p: CapJuguete(p))
        self.parche.start()
        self.r = R.VideoReader(self.path)

    def tearDown(self):
        self.r.close()
        self.parche.stop()
        self.tmp.cleanup()


class ElProblemaTests(Base):
    def test_un_salto_cae_corrido_con_read_at(self):
        """Documenta el problema: read_at salta para ir lejos, y el frame que devuelve no es el pedido."""
        self.assertEqual(real(self.r.read_at(200)), 200 + CORRIMIENTO)

    def test_y_con_exact_false_se_recorre_corrido_como_antes(self):
        got = [(i, real(f)) for i, f in self.r.iter_frames(100, 104, exact=False)]
        self.assertEqual(got, [(i, i + CORRIMIENTO) for i in range(100, 105)])


class LecturaExactaTests(Base):
    def test_recorrer_desde_un_frame_lejano_da_los_frames_reales(self):
        got = [(i, real(f)) for i, f in self.r.iter_frames(100, 110)]
        self.assertEqual(got, [(i, i) for i in range(100, 111)])

    def test_tras_un_salto_impreciso_no_se_confia_en_la_posicion(self):
        """Como en el pipeline: antes de recorrer se leen frames sueltos (el fondo) con saltos; luego el recorrido debe ser exacto."""
        self.r.read_at(250)  # cae en 277, pero el lector «cree» estar en 251
        got = [(i, real(f)) for i, f in self.r.iter_frames(251, 255)]  # justo «a continuación»: sin reabrir sería 278, 279...
        self.assertEqual(got, [(i, i) for i in range(251, 256)])

    def test_un_paso_mayor_que_el_limite_de_avance_sigue_exacto(self):
        got = [(i, real(f)) for i, f in self.r.iter_frames(0, 200, 40)]  # 40 > 30: antes se saltaba
        self.assertEqual(got, [(i, i) for i in range(0, 201, 40)])

    def test_un_frame_anterior_reabre_el_video(self):
        list(self.r.iter_frames(150, 152))
        antes = self.r.reopens
        self.assertEqual(real(self.r.read_exact(20)), 20)
        self.assertEqual(self.r.reopens, antes + 1)

    def test_si_la_posicion_es_exacta_no_se_reabre(self):
        """Empezar desde una posición exacta solo avanza: no paga el costo de reabrir."""
        list(self.r.iter_frames(10, 20))
        self.assertEqual(self.r.reopens, 0)  # recién abierto: posición 0 exacta
        list(self.r.iter_frames(30, 40))  # más adelante: solo avanza
        self.assertEqual(self.r.reopens, 0)
        self.r.read_at(5)  # hacia atrás: read_at salta
        list(self.r.iter_frames(100, 101))
        self.assertEqual(self.r.reopens, 1)

    def test_read_at_cerca_y_hacia_adelante_conserva_la_exactitud(self):
        list(self.r.iter_frames(0, 5))
        self.r.read_at(10)  # 4 frames adelante: avanza con grab, sin saltar
        list(self.r.iter_frames(20, 21))
        self.assertEqual(self.r.reopens, 0)

    def test_al_final_del_video(self):
        got = [i for i, _ in self.r.iter_frames(N - 3, N + 10)]
        self.assertEqual(got, [N - 3, N - 2, N - 1])
        self.assertIsNone(self.r.read_exact(N + 5))
        self.assertEqual(real(self.r.read_exact(7)), 7)  # tras el final, se puede volver a leer (reabre)

    def test_un_frame_que_no_se_puede_decodificar_al_avanzar_no_se_esconde(self):
        """Si un frame intermedio falla, no se puede devolver otro frame con el número pedido: se devuelve None."""
        class ConRoto(CapJuguete):
            def grab(self):
                if self.pos == 120:
                    return False  # no se pudo decodificar; la posición no avanza
                return super().grab()
        with mock.patch.object(R.cv2, "VideoCapture", side_effect=lambda p: ConRoto(p)):
            r = R.VideoReader(self.path)
            try:
                self.assertIsNone(r.read_exact(150))
            finally:
                r.close()

    def test_indices_negativos_y_lector_cerrado(self):
        self.assertIsNone(self.r.read_exact(-1))
        self.r.close()
        with self.assertRaises(ValueError):
            self.r.read_exact(3)

    def test_si_no_se_puede_reabrir(self):
        self.r.read_at(250)
        with mock.patch.object(R.cv2, "VideoCapture", side_effect=lambda p: CapJuguete(p, abre=False)):
            with self.assertRaises(R.VideoOpenError):
                self.r.read_exact(260)


class ConVideoRealTests(unittest.TestCase):
    """Con un video AVI real (donde el salto es exacto), la lectura exacta da lo mismo que la de corrido."""

    def test_coincide_con_la_lectura_de_corrido(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v.avi")
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
            for i in range(80):
                f = np.zeros((48, 64, 3), np.uint8)
                cv2.putText(f, str(i), (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                vw.write(f)
            vw.release()
            cap, todos = cv2.VideoCapture(path), []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                todos.append(f)
            cap.release()
            with R.VideoReader(path) as r:
                r.read_at(70)
                for i, f in r.iter_frames(33, 79, 7):
                    self.assertTrue(np.array_equal(f, todos[i]), msg=f"frame {i}")


if __name__ == "__main__":
    unittest.main()