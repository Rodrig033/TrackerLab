"""Pruebas de tools/check_frame_alignment.py.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_check_frame_alignment.py -v

Se construyen conjuntos cuyas imágenes vienen a propósito de frames DISTINTOS del esperado (desfase conocido) y
se comprueba que la herramienta recupera cada desfase.
"""

import importlib
import json
import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from tools import check_frame_alignment as C
from tools import label_frames as L


def _imp(name):
    try:
        return importlib.import_module(f"tests.{name}")
    except ImportError:
        return importlib.import_module(name)


SF = _imp("test_sample_frames")

W, H = 64, 48
N = 120


def silencio(_):
    pass


def decodificar(path):
    """Todos los frames, leídos de corrido (la referencia exacta de lo que procesa el pipeline)."""
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def video_disco(tmp, nombre, n=N, w=320, h=240):
    """Un círculo que se mueve en cada frame (hay movimiento que medir). Devuelve (ruta, centros)."""
    rng = np.random.default_rng(2)
    base = (np.linspace(50, 100, w)[None, :] + np.zeros((h, 1))).astype(np.float32)
    path = os.path.join(tmp, nombre)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (w, h))
    centros = []
    for i in range(n):
        f = cv2.cvtColor(np.clip(base + rng.normal(0, 2.0, (h, w)), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        c = (int(40 + 2.2 * i), int(120 + 50 * np.sin(i / 9.0)))
        cv2.circle(f, c, 12, (235, 235, 235), -1)
        vw.write(f)
        centros.append(c)
    vw.release()
    return path, centros


def crear_conjunto(tmp, video, frames_decod, desfases, w, h, formato="png", ext_frames=None):
    """Conjunto cuya imagen del frame N es, en realidad, el frame N + desfases[N]."""
    val = os.path.join(tmp, "validacion")
    os.makedirs(os.path.join(val, "images"), exist_ok=True)
    os.makedirs(os.path.join(val, "labels"), exist_ok=True)
    frames = sorted(desfases)
    for f in frames:
        img = frames_decod[f + desfases[f]]
        cv2.imwrite(os.path.join(val, "images", f"v_f{f:06d}.{formato}"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95] if formato == "jpg" else [])
    with open(os.path.join(val, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"videos": [{"id": "v", "path": os.path.abspath(video), "nombre": os.path.basename(video), "ancho": w, "alto": h,
                               "formato": formato, "frames": frames}]}, fh)
    return val


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.bits = SF.video_con_indices(cls.tmp.name, "bits.avi", N)  # el contenido de cada frame es único
        cls.bits_frames = decodificar(cls.bits)
        cls.disco, cls.centros = video_disco(cls.tmp.name, "disco.avi")
        cls.disco_frames = decodificar(cls.disco)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.work.cleanup()

    def resultado(self, val):
        with open(os.path.join(val, "alineacion.json"), encoding="utf-8") as fh:
            return json.load(fh)


# ======================================================================
class PiezasTests(unittest.TestCase):
    def test_crop_region(self):
        self.assertEqual(C.crop_region((100, 100, 40, 30), 640, 480), (20, 20, 220, 210))  # margen = max(60, 2*40) = 80
        self.assertEqual(C.crop_region((100, 100, 20, 20), 640, 480), (40, 40, 180, 180))  # margen mínimo de 60
        self.assertEqual(C.crop_region((5, 5, 40, 30), 640, 480), (0, 0, 125, 115))  # recortado al borde
        self.assertEqual(C.crop_region((600, 450, 40, 30), 640, 480), (520, 370, 640, 480))
        self.assertEqual(C.crop_region(None, 640, 480), (0, 0, 640, 480))  # sin caja: imagen entera

    def test_best_offset(self):
        r = C.best_offset({-1: 10.0, 0: 2.0, 1: 11.0, 2: 12.0})
        self.assertEqual((r["desfase"], r["clara"], r["mae_cero"], r["mae_mejor"]), (0, True, 2.0, 2.0))
        r = C.best_offset({-1: 10.0, 0: 9.0, 1: 11.0, 2: 1.5, 3: 10.0})
        self.assertEqual((r["desfase"], r["clara"], r["mae_cero"]), (2, True, 9.0))

    def test_empates_van_al_mas_cercano_a_cero(self):
        self.assertEqual(C.best_offset({-2: 1.0, 0: 1.0, 2: 1.0})["desfase"], 0)
        self.assertEqual(C.best_offset({-3: 1.0, 2: 1.0, 5: 9.0})["desfase"], 2)

    def test_una_escena_sin_cambios_es_ambigua(self):
        self.assertFalse(C.best_offset({0: 5.0, 1: 5.1, -1: 5.0, 2: 5.2})["clara"])

    def test_con_menos_de_tres_candidatos_no_se_fia(self):
        self.assertFalse(C.best_offset({0: 1.0, 1: 30.0})["clara"])
        r = C.best_offset({})
        self.assertEqual((r["desfase"], r["clara"]), (None, False))

    def test_resumen_umbral_del_90_por_ciento(self):
        fila = lambda d, clara=True: {"desfase": d, "clara": clara}  # noqa: E731
        s = C.summarize([fila(0)] * 9 + [fila(2)])
        self.assertEqual((s["veredicto"], s["alineadas"], s["determinadas"]), ("alineado", 9, 10))  # 9/10 = 90 %
        s = C.summarize([fila(0)] * 8 + [fila(2)] * 2)
        self.assertEqual(s["veredicto"], "desalineado")
        self.assertEqual(s["desfases"], {0: 8, 2: 2})

    def test_resumen_ignora_las_ambiguas_y_calcula_la_mediana(self):
        fila = lambda d, clara=True: {"desfase": d, "clara": clara}  # noqa: E731
        s = C.summarize([fila(3), fila(3), fila(1), fila(0, False), fila(5, False)])
        self.assertEqual((s["determinadas"], s["ambiguas"], s["veredicto"], s["desfase_mediano"]), (3, 2, "desalineado", 3.0))
        self.assertEqual(C.summarize([fila(0, False)])["veredicto"], "no_determinado")
        self.assertEqual(C.summarize([])["veredicto"], "no_determinado")


# ======================================================================
class RecuperacionDeDesfasesTests(Base):
    FRAMES = [10, 25, 40, 55, 70, 85, 100]

    def correr(self, val, **kw):
        msgs = []
        rc = C.run(val, out=msgs.append, **kw)
        return rc, msgs

    def test_todo_alineado(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {f: 0 for f in self.FRAMES}, W, H)
        rc, msgs = self.correr(val)
        self.assertEqual(rc, 0)
        res = self.resultado(val)
        self.assertEqual(res["resumen"]["veredicto"], "alineado")
        self.assertEqual([r["desfase"] for r in res["imagenes"]], [0] * 7)
        self.assertTrue(any(m.startswith("ALINEADO") for m in msgs))

    def test_un_desfase_constante_se_recupera_exacto(self):
        for d in (1, 2, 3, -1, -2, -4, 6):
            with self.subTest(desfase=d):
                with tempfile.TemporaryDirectory() as tmp:
                    val = crear_conjunto(tmp, self.bits, self.bits_frames, {f: d for f in self.FRAMES}, W, H)
                    rc, msgs = self.correr(val)
                    self.assertEqual(rc, 0)
                    res = self.resultado(val)
                    self.assertEqual([r["desfase"] for r in res["imagenes"]], [d] * 7)
                    self.assertEqual(res["resumen"]["veredicto"], "desalineado")
                    self.assertEqual(res["resumen"]["desfase_mediano"], float(d))
                    self.assertTrue(any(m.startswith("ATENCIÓN") for m in msgs))

    def test_desfases_distintos_por_imagen_se_recuperan_uno_a_uno(self):
        desf = {10: 0, 25: 1, 40: -2, 55: 3, 70: 0, 85: -4, 100: 2}  # lo que pasaría con saltos que caen distinto cada vez
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, desf, W, H)
        self.correr(val)
        got = {r["frame"]: r["desfase"] for r in self.resultado(val)["imagenes"]}
        self.assertEqual(got, desf)

    def test_cerca_del_final_del_video(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {116: 2, 118: 1}, W, H)  # 118 + 1 = 119 (último)
        self.correr(val)
        self.assertEqual({r["frame"]: r["desfase"] for r in self.resultado(val)["imagenes"]}, {116: 2, 118: 1})

    def test_cerca_del_principio_no_falla(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {1: 3, 2: 0}, W, H)
        rc, _ = self.correr(val)
        self.assertEqual(rc, 0)
        self.assertEqual({r["frame"]: r["desfase"] for r in self.resultado(val)["imagenes"]}[1], 3)

    def test_el_error_con_el_frame_esperado_refleja_la_distancia(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {40: 0, 55: 2}, W, H)
        self.correr(val)
        r = {x["frame"]: x for x in self.resultado(val)["imagenes"]}
        self.assertEqual(r[40]["mae_cero"], 0.0)  # idéntica: error cero
        self.assertGreater(r[55]["mae_cero"], 0.0)  # desfasada: el frame esperado ya no coincide
        self.assertEqual(r[55]["mae_mejor"], 0.0)

    def test_una_escena_sin_movimiento_se_declara_no_determinada(self):
        quieto = os.path.join(self.work.name, "quieto.avi")
        vw = cv2.VideoWriter(quieto, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (W, H))
        for _ in range(60):
            vw.write(np.full((H, W, 3), 90, np.uint8))
        vw.release()
        val = crear_conjunto(self.work.name, quieto, decodificar(quieto), {20: 0, 30: 0}, W, H)
        rc, msgs = self.correr(val)
        self.assertEqual(rc, 0)
        self.assertEqual(self.resultado(val)["resumen"]["veredicto"], "no_determinado")
        self.assertTrue(any("NO SE PUDO DETERMINAR" in m for m in msgs))
        self.assertTrue(any("ambigua" in m for m in msgs))


class ConMovimientoRealYEtiquetasTests(Base):
    FRAMES = [15, 30, 45, 60, 75, 90]

    def conjunto(self, desf, formato="png"):
        val = crear_conjunto(self.work.name, self.disco, self.disco_frames, desf, 320, 240, formato=formato)
        for f in desf:  # etiqueta: caja alrededor del círculo tal como se ve en la imagen guardada (el frame f + desfase)
            cx, cy = self.centros[f + desf[f]]
            L_ = (cx - 12, cy - 12, 25, 25)
            with open(os.path.join(val, "labels", f"v_f{f:06d}.txt"), "w") as fh:
                from learning.dataset import yolo_label_line
                fh.write(yolo_label_line(L_, 320, 240) + "\n")
        return val

    def test_con_etiquetas_se_recuperan_los_desfases(self):
        desf = {15: 0, 30: 2, 45: -3, 60: 1, 75: 0, 90: 4}
        val = self.conjunto(desf)
        self.assertEqual(C.run(val, out=silencio), 0)
        self.assertEqual({r["frame"]: r["desfase"] for r in self.resultado(val)["imagenes"]}, desf)
        self.assertTrue(all(r["clara"] for r in self.resultado(val)["imagenes"]))

    def test_resiste_la_compresion_jpeg_de_las_imagenes_guardadas(self):
        desf = {f: 2 for f in self.FRAMES}
        val = self.conjunto(desf, formato="jpg")
        C.run(val, out=silencio)
        res = self.resultado(val)
        self.assertEqual([r["desfase"] for r in res["imagenes"]], [2] * 6)
        self.assertEqual(res["resumen"]["veredicto"], "desalineado")

    def test_jpeg_alineado_no_da_falsas_alarmas(self):
        val = self.conjunto({f: 0 for f in self.FRAMES}, formato="jpg")
        C.run(val, out=silencio)
        self.assertEqual(self.resultado(val)["resumen"]["veredicto"], "alineado")

    def test_el_desfase_de_un_frame_a_la_velocidad_del_circulo_son_unos_2_px(self):
        """Orden de magnitud: con este círculo, un frame de desfase son ~2 px; a velocidad de ratón real, decenas."""
        c0, c1 = self.centros[40], self.centros[41]
        self.assertAlmostEqual(abs(c1[0] - c0[0]), 2, delta=1)


class RatonPequenoEnCuadroGrandeTests(unittest.TestCase):
    """El caso real: un ratón diminuto en un cuadro grande con textura y compresión JPEG.

    Con la imagen entera el ratón casi no se nota (el ruido de compresión es igual para todos los candidatos y la
    diferencia es de décimas de nivel); alrededor de la etiqueta sí. Por eso la herramienta compara en esa zona, y
    sin etiquetas dice «ambigua» en vez de afirmar algo que no puede sostener.
    """

    W, H, N = 960, 540, 80
    DESF = {15: 2, 30: -3, 45: 1, 55: 0, 65: 3}

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(5)
        tex = np.clip(100 + rng.normal(0, 14, (cls.H, cls.W)), 0, 255).astype(np.uint8)  # textura ESTÁTICA, igual en todos los frames
        cls.video = os.path.join(cls.tmp.name, "textura.avi")
        vw = cv2.VideoWriter(cls.video, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (cls.W, cls.H))
        cls.centros = []
        for i in range(cls.N):
            f = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
            c = (80 + 6 * i, 270 + int(80 * np.sin(i / 8.0)))
            cv2.circle(f, c, 12, (235, 235, 235), -1)
            vw.write(f)
            cls.centros.append(c)
        vw.release()
        cls.dec = decodificar(cls.video)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def conjunto(self, etiquetas):
        val = tempfile.mkdtemp(dir=self.tmp.name)
        os.makedirs(os.path.join(val, "images"))
        os.makedirs(os.path.join(val, "labels"))
        from learning.dataset import yolo_label_line

        for f, d in self.DESF.items():
            cv2.imwrite(os.path.join(val, "images", f"v_f{f:06d}.jpg"), self.dec[f + d], [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if etiquetas:
                cx, cy = self.centros[f + d]
                with open(os.path.join(val, "labels", f"v_f{f:06d}.txt"), "w") as fh:
                    fh.write(yolo_label_line((cx - 12, cy - 12, 25, 25), self.W, self.H) + "\n")
        with open(os.path.join(val, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump({"videos": [{"id": "v", "path": self.video, "nombre": "textura.avi", "ancho": self.W, "alto": self.H,
                                   "formato": "jpg", "frames": list(self.DESF)}]}, fh)
        return val

    def filas(self, val):
        C.run(val, out=silencio)
        with open(os.path.join(val, "alineacion.json"), encoding="utf-8") as fh:
            return json.load(fh)["imagenes"]

    def test_con_etiquetas_el_desfase_es_claro_y_exacto(self):
        filas = self.filas(self.conjunto(etiquetas=True))
        self.assertEqual({r["frame"]: r["desfase"] for r in filas}, self.DESF)
        self.assertTrue(all(r["clara"] for r in filas))

    def test_sin_etiquetas_la_herramienta_admite_que_no_puede_estar_segura(self):
        filas = self.filas(self.conjunto(etiquetas=False))
        self.assertFalse(any(r["clara"] for r in filas))  # la diferencia en la imagen entera es de décimas de nivel
        # y por eso no afirma «desalineado» (ni «alineado») con una evidencia que no sostiene
        self.assertEqual(C.summarize(filas)["veredicto"], "no_determinado")


class NumeracionDelPipelineTests(Base):
    """Un salto impreciso del lector corre TODOS los índices de una lectura por igual. El diagnóstico debe leer desde
    el mismo punto de partida que la corrida; empezar en otro sitio mediría contra una numeración distinta."""

    @staticmethod
    def lector_con_saltos_variables():
        from video.reader import VideoReader

        class SaltoVariable(VideoReader):
            """Un salto cae (idx % 4) frames más adelante y el lector NO lo sabe: cree estar en idx + 1. Lo que sigue,
            leído de corrido, continúa desde donde cayó de verdad, así que TODA esa lectura queda corrida por igual."""

            def read_at(self, idx):
                pos = self._pos
                es_salto = not (pos is not None and 0 <= idx - pos <= self.skip_limit)
                if not es_salto:
                    return super().read_at(idx)
                frame = super().read_at(idx + (idx % 4))
                self._pos = idx + 1  # lo que cree el lector, no la posición real del decodificador
                return frame

        return SaltoVariable

    def conjunto(self, frames, rango, desfase_stream=0):
        """Imágenes = el contenido que entrega el frame N en la numeración del pipeline (N + desfase_stream)."""
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {f: desfase_stream for f in frames}, W, H)
        with open(os.path.join(val, "manifest.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        man["videos"][0]["rango"] = list(rango)
        with open(os.path.join(val, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(man, fh)
        return val

    def offsets(self, val):
        with mock.patch.object(C, "VideoReader", self.lector_con_saltos_variables()):
            C.run(val, out=silencio)
        return {r["frame"]: r["desfase"] for r in self.resultado(val)["imagenes"]}

    def test_lee_desde_el_inicio_del_rango_y_no_desde_la_primera_imagen(self):
        # El salto a 40 es exacto (40 % 4 = 0); un salto a 42 (50 - 8) caería 2 frames más adelante y daría -2 por error.
        val = self.conjunto([50, 62, 74, 86], rango=(40, 119))
        self.assertEqual(self.offsets(val), {50: 0, 62: 0, 74: 0, 86: 0})

    def test_un_conjunto_sacado_con_la_lectura_corrida_de_antes_ahora_se_ve_desfasado(self):
        # ANTES el recorrido del pipeline podía heredar un salto corrido (a 42 caía en 44: su «frame 50» era el 52), y un conjunto
        # muestreado con esa misma lectura «coincidía» con él. AHORA iter_frames llega de corrido y es exacto aunque read_at salte
        # corrido: esas imágenes viejas (contenido 52 rotulado «50») se ven desfasadas +2, que es la verdad respecto al video.
        val = self.conjunto([50, 62, 74, 86], rango=(42, 119), desfase_stream=2)
        self.assertEqual(self.offsets(val), {50: 2, 62: 2, 74: 2, 86: 2})

    def test_sin_rango_en_el_manifiesto_se_empieza_cerca_de_la_primera_imagen(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {50: 0, 70: 0}, W, H)  # sin «rango»
        C.run(val, out=silencio)
        self.assertEqual({r["frame"]: r["desfase"] for r in self.resultado(val)["imagenes"]}, {50: 0, 70: 0})

    def test_un_rango_que_empieza_despues_de_la_primera_imagen_no_rompe(self):
        val = self.conjunto([50, 62], rango=(60, 119))  # la primera imagen es anterior al inicio del rango
        rc = C.run(val, out=silencio)
        self.assertEqual(rc, 0)


class ErroresYLineaDeComandosTests(Base):
    def test_sin_manifiesto(self):
        msgs = []
        self.assertEqual(C.run(os.path.join(self.work.name, "no_existe"), out=msgs.append), 2)
        self.assertIn("sample_frames", msgs[0])

    def test_video_que_no_esta(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {40: 0}, W, H)
        os.rename(self.bits, self.bits + ".movido")
        try:
            msgs = []
            self.assertEqual(C.run(val, out=msgs.append), 2)
            self.assertTrue(any("no se encuentra el video" in m for m in msgs))
            # con --video apuntando a donde está ahora, funciona
            self.assertEqual(C.run(val, video=self.bits + ".movido", out=silencio), 0)
        finally:
            os.rename(self.bits + ".movido", self.bits)

    def test_imagenes_ilegibles_o_de_otro_tamano_se_omiten(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {40: 0, 55: 0}, W, H)
        cv2.imwrite(os.path.join(val, "images", "v_f000055.png"), np.zeros((10, 10, 3), np.uint8))  # tamaño distinto al del manifiesto
        self.assertEqual(C.run(val, out=silencio), 0)
        self.assertEqual([r["frame"] for r in self.resultado(val)["imagenes"]], [40])

    def test_ventana_pequena_no_encuentra_un_desfase_grande(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {50: 6}, W, H)
        C.run(val, window=3, out=silencio)
        r = self.resultado(val)["imagenes"][0]
        self.assertNotEqual(r["desfase"], 6)  # fuera de la ventana: no puede verlo (y por eso existe --ventana)
        C.run(val, window=8, out=silencio)
        self.assertEqual(self.resultado(val)["imagenes"][0]["desfase"], 6)

    def test_main(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {40: 0, 70: 0}, W, H)
        self.assertEqual(C.main([val, "--ventana", "5"]), 0)
        self.assertEqual(self.resultado(val)["ventana"], 5)


class ConsistenciaConElRestoTests(unittest.TestCase):
    def test_el_nombre_de_las_imagenes_coincide_con_el_etiquetador(self):
        for vid, f in (("rata01", 42), ("a_b-c", 1234567)):
            self.assertEqual(C.stem_of(vid, f), L.stem_of(vid, f))


if __name__ == "__main__":
    unittest.main()


# ======================================================================
class ComoLaLeyoUnaCorridaTests(Base):
    """--entradas: el video se lee como lo leyó esa corrida (desde SU inicio). Detecta un salto inicial impreciso."""

    FRAMES = [40, 55, 70, 85, 100]

    def entradas(self, inicio, fin, nombre="bits.avi", huella=None):
        d = os.path.join(self.work.name, f"entradas_{inicio}_{fin}")
        os.makedirs(d, exist_ok=True)
        video = {"path": os.path.join("/otra/carpeta", nombre)}
        if huella:
            video["sha256_extremos"] = huella
        with open(os.path.join(d, "inputs.json"), "w", encoding="utf-8") as fh:
            json.dump({"frame_range": {"start": inicio, "end": fin}, "video": video}, fh)
        return d

    def lector_que_cae_corrido(self, k):
        """Un lector que, al saltar a un inicio > 0, cae k frames DESPUÉS (lo que hace a veces un mp4)."""
        frames = self.bits_frames

        class Lector:
            def __init__(self, path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def iter_frames(self, start, end, step):
                corr = k if start > 0 else 0
                for idx in range(start, min(end, len(frames) - 1 - corr) + 1, step):
                    yield idx, frames[idx + corr]
        return Lector

    def correr(self, val, **kw):
        msgs = []
        rc = C.run(val, out=msgs.append, **kw)
        return rc, msgs

    def test_un_salto_inicial_corrido_se_detecta(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {f: 0 for f in self.FRAMES}, W, H)  # imágenes exactas
        ent = self.entradas(30, 110)
        with mock.patch.object(C, "VideoReader", self.lector_que_cae_corrido(3)):
            rc, msgs = self.correr(val, entradas=ent)
        self.assertEqual(rc, 0)
        with open(os.path.join(val, f"alineacion_{os.path.basename(ent)}.json"), encoding="utf-8") as fh:
            res = json.load(fh)
        # la corrida llama «frame N» a lo que en realidad es N + 3: cada imagen se parece a su «frame N - 3»
        self.assertEqual([r["desfase"] for r in res["imagenes"]], [-3] * 5)
        self.assertEqual(res["resumen"]["veredicto"], "desalineado")
        self.assertEqual(res["entradas"], ent)
        texto = "\n".join(msgs)
        self.assertIn("desde su frame 30", texto)
        self.assertIn("Esa corrida vio frames corridos", texto)

    def test_sin_salto_corrido_esta_alineado(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {f: 0 for f in self.FRAMES}, W, H)
        with mock.patch.object(C, "VideoReader", self.lector_que_cae_corrido(0)):
            self.correr(val, entradas=self.entradas(30, 110))
        with open(os.path.join(val, "alineacion_entradas_30_110.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["resumen"]["veredicto"], "alineado")

    def test_sin_entradas_se_lee_desde_el_inicio_del_conjunto_y_no_se_ve_el_problema(self):
        """El mismo lector defectuoso, pero empezando en 0 (como el conjunto): no salta, no cae corrido. Por eso hace falta --entradas."""
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {f: 0 for f in self.FRAMES}, W, H)
        with open(os.path.join(val, "manifest.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        man["videos"][0]["rango"] = [0, 119]
        with open(os.path.join(val, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(man, fh)
        with mock.patch.object(C, "VideoReader", self.lector_que_cae_corrido(3)):
            self.correr(val)
        self.assertEqual(self.resultado(val)["resumen"]["veredicto"], "alineado")

    def test_el_lector_real_empieza_donde_empezo_la_corrida(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {f: 0 for f in self.FRAMES}, W, H)
        llamadas = []
        real = C.VideoReader

        class Espia(real):
            def iter_frames(self, start, end, step):
                llamadas.append((start, end, step))
                return super().iter_frames(start, end, step)
        with mock.patch.object(C, "VideoReader", Espia):
            rc, _ = self.correr(val, entradas=self.entradas(37, 110))
        self.assertEqual(rc, 0)
        self.assertEqual(llamadas[0][0], 37)  # no antes (como sin --entradas): exactamente su inicio
        with open(os.path.join(val, "alineacion_entradas_37_110.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["resumen"]["veredicto"], "alineado")  # en AVI el salto es exacto

    def test_solo_se_revisan_las_imagenes_dentro_del_rango(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {f: 0 for f in [10, 20] + self.FRAMES}, W, H)
        rc, msgs = self.correr(val, entradas=self.entradas(30, 90))
        with open(os.path.join(val, "alineacion_entradas_30_90.json"), encoding="utf-8") as fh:
            self.assertEqual([r["frame"] for r in json.load(fh)["imagenes"]], [40, 55, 70, 85])
        self.assertIn("3 imágenes de bits.avi quedan fuera del rango de la corrida (30-90); no se revisan.", "\n".join(msgs))

    def test_no_se_pisa_la_alineacion_del_conjunto(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {f: 0 for f in self.FRAMES}, W, H)
        self.correr(val)
        with open(os.path.join(val, "alineacion.json"), encoding="utf-8") as fh:
            antes = fh.read()
        self.correr(val, entradas=self.entradas(30, 110))
        with open(os.path.join(val, "alineacion.json"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), antes)

    def test_una_corrida_de_otro_video(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {f: 0 for f in self.FRAMES}, W, H)
        rc, msgs = self.correr(val, entradas=self.entradas(30, 110, nombre="otro.avi"))
        self.assertEqual(rc, 2)
        self.assertIn("AVISO: la corrida no es de bits.avi", "\n".join(msgs))

    def test_con_huellas_manda_la_huella(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {f: 0 for f in self.FRAMES}, W, H)
        with open(os.path.join(val, "manifest.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        man["videos"][0]["sha256_extremos"] = "a" * 64
        with open(os.path.join(val, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(man, fh)
        self.assertEqual(self.correr(val, entradas=self.entradas(30, 110, huella="b" * 64))[0], 2)  # mismo nombre, otra huella: otro video
        self.assertEqual(self.correr(val, entradas=self.entradas(31, 110, nombre="renombrado.avi", huella="a" * 64))[0], 0)  # otro nombre, misma huella

    def test_entradas_ilegibles(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {f: 0 for f in self.FRAMES}, W, H)
        rc, msgs = self.correr(val, entradas=os.path.join(self.work.name, "no_existe"))
        self.assertEqual(rc, 2)
        self.assertIn("No se pudieron leer las entradas", "\n".join(msgs))

    def test_cli(self):
        val = crear_conjunto(self.work.name, self.bits, self.bits_frames, {f: 0 for f in self.FRAMES}, W, H)
        ent = self.entradas(30, 110)
        with mock.patch.object(C, "VideoReader", self.lector_que_cae_corrido(2)):
            self.assertEqual(C.main([val, "--entradas", ent]), 0)
        with open(os.path.join(val, "alineacion_entradas_30_110.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["resumen"]["desfase_mediano"], -2.0)