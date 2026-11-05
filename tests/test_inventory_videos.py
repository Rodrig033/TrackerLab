"""Pruebas de tools/inventory_videos.py.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_inventory_videos.py -v
"""

import csv
import math
import os
import shutil
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from tools import inventory_videos as I


def hacer_video(path, fondo="gx", n=100, semilla=1, size=(160, 90), fps=25.0):
    """Video sintético: un fondo fijo (``gx`` degradado horizontal, ``gy`` vertical, ``rayas``) con ruido y un círculo que se mueve."""
    w, h = size
    rng = np.random.default_rng(semilla)
    if fondo == "gx":
        base = np.tile(np.linspace(30, 220, w, dtype=np.float32), (h, 1))
    elif fondo == "gy":
        base = np.tile(np.linspace(30, 220, h, dtype=np.float32)[:, None], (1, w))
    else:
        yy, xx = np.mgrid[0:h, 0:w]
        base = 125 + 90 * np.sin(xx / 5.0) * np.sin(yy / 4.0)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        f = np.clip(base + rng.normal(0, 2, (h, w)), 0, 255).astype(np.uint8)
        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        t = i + 17 * semilla
        cv2.circle(f, (int(w / 2 + (w / 2 - 12) * math.sin(t / 9.0)), int(h / 2 + (h / 2 - 12) * math.sin(t / 5.0))), 7, (255, 255, 255), -1)
        vw.write(f)
    vw.release()
    return path


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = self.tmp.name
        self.salida = os.path.join(self.t, "inv")

    def tearDown(self):
        self.tmp.cleanup()

    def correr(self, carpetas=None, **kw):
        msgs = []
        rc = I.inventory(carpetas or [self.t + "/videos"], self.salida, out=msgs.append, **kw)
        return rc, "\n".join(msgs)

    def filas(self):
        with open(os.path.join(self.salida, "inventario.csv"), newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))

    def carpeta_videos(self):
        d = os.path.join(self.t, "videos")
        os.makedirs(d, exist_ok=True)
        return d

class PiezasPurasTests(unittest.TestCase):
    def test_fmt_duration(self):
        self.assertEqual(I.fmt_duration(None), "?")
        self.assertEqual(I.fmt_duration(float("nan")), "?")
        self.assertEqual(I.fmt_duration(-5), "?")
        self.assertEqual(I.fmt_duration(59), "0:59")
        self.assertEqual(I.fmt_duration(59.6), "1:00")
        self.assertEqual(I.fmt_duration(3725), "1:02:05")
        self.assertEqual(I.fmt_duration(0), "0:00")

    def test_looks_like_drive(self):
        self.assertTrue(I.looks_like_drive("/Users/farid/Library/CloudStorage/GoogleDrive-x/Mi unidad/videos"))
        self.assertTrue(I.looks_like_drive("/Volumes/Google Drive/videos"))
        self.assertFalse(I.looks_like_drive("/Users/farid/Videos"))

    def test_sample_positions(self):
        self.assertEqual(I.sample_positions(0), [])
        self.assertEqual(I.sample_positions(3), [0, 1, 2])
        self.assertEqual(I.sample_positions(5), [0, 1, 2, 3, 4])
        self.assertEqual(I.sample_positions(1000), [100, 300, 500, 700, 900])  # entre el 10 % y el 90 %, sin los extremos
        p = I.sample_positions(6)
        self.assertEqual(p, sorted(set(p)))
        self.assertTrue(all(0 <= x < 6 for x in p))

    def test_la_firma_del_fondo_quita_al_animal(self):
        fondo = np.tile(np.linspace(30, 220, 160, dtype=np.uint8), (90, 1))
        frames = []
        for k in range(5):
            f = cv2.cvtColor(fondo, cv2.COLOR_GRAY2BGR)
            cv2.circle(f, (20 + 30 * k, 45), 8, (255, 255, 255), -1)  # el círculo cambia de lugar en cada frame
            frames.append(f)
        sig = I.background_signature(frames)
        self.assertEqual(sig.shape, (36, 64))
        esperado = cv2.resize(fondo, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        self.assertLess(float(np.abs(sig - esperado).max()), 3.0)  # el círculo (blanco) no aparece
        self.assertIsNone(I.background_signature([]))

    def test_correlation(self):
        rng = np.random.default_rng(0)
        a = rng.random((36, 64)).astype(np.float32)
        self.assertAlmostEqual(I.correlation(a, a), 1.0, places=9)
        self.assertAlmostEqual(I.correlation(a, 1 - a), -1.0, places=9)
        self.assertAlmostEqual(I.correlation(a, 3 * a + 10), 1.0, places=9)  # no depende del brillo ni del contraste globales
        self.assertLess(abs(I.correlation(a, rng.random((36, 64)))), 0.2)
        k = np.full((4, 4), 5.0)
        self.assertEqual(I.correlation(k, k), 1.0)
        self.assertEqual(I.correlation(k, k + 1), 0.0)
        self.assertEqual(I.correlation(k, a[:4, :4]), 0.0)

    def test_agrupacion(self):
        rng = np.random.default_rng(1)
        a = rng.random((36, 64)).astype(np.float32)
        a2 = a + rng.normal(0, 0.01, a.shape).astype(np.float32)
        b = rng.random((36, 64)).astype(np.float32)
        g, m = I.group_by_similarity([a, b, a2, None], 0.9)
        self.assertEqual(g, [1, 2, 1, None])  # numerados por orden de aparición; sin firma = sin grupo
        self.assertTrue(np.isnan(m[3, 0]) and np.isnan(m[0, 3]) and np.isnan(m[3, 3]))
        self.assertAlmostEqual(float(m[0, 2]), float(m[2, 0]))  # simétrica
        self.assertGreater(m[0, 2], 0.99)
        self.assertLess(m[0, 1], 0.5)

    def test_umbral_y_firmas_identicas(self):
        rng = np.random.default_rng(2)
        a = rng.random((36, 64)).astype(np.float32)
        a2 = a + rng.normal(0, 0.01, a.shape).astype(np.float32)
        self.assertEqual(I.group_by_similarity([a, a2], 1.0)[0], [1, 2])  # parecidas pero no idénticas
        self.assertEqual(I.group_by_similarity([a, a.copy()], 1.0)[0], [1, 1])  # idénticas se unen aunque el redondeo dé 0.99999...

    def test_union_por_cadena_se_documenta(self):
        """Dos videos se unen si se parecen entre sí; si A~B y B~C, A y C quedan juntos aunque A y C no se parezcan tanto."""
        rng = np.random.default_rng(3)
        a = rng.random((36, 64)).astype(np.float32)
        b = a + rng.normal(0, 0.45, a.shape).astype(np.float32)
        c = b + rng.normal(0, 0.45, a.shape).astype(np.float32)
        ab, bc, ac = I.correlation(a, b), I.correlation(b, c), I.correlation(a, c)
        self.assertGreater(min(ab, bc), ac)
        umbral = (min(ab, bc) + ac) / 2
        self.assertEqual(I.group_by_similarity([a, b, c], umbral)[0], [1, 1, 1])


class ScanTests(Base):
    def test_recursivo_extensiones_ocultos_y_sin_repetir(self):
        d = self.carpeta_videos()
        os.makedirs(os.path.join(d, "sesion2", "sub"))
        os.makedirs(os.path.join(d, ".oculta"))
        for rel in ("b.mp4", "a.MOV", "sesion2/c.avi", "sesion2/sub/d.mkv", "notas.txt", "foto.jpg", "._a.MOV", ".escondido.mp4", ".oculta/e.mp4"):
            with open(os.path.join(d, rel), "w") as fh:
                fh.write("x")
        encontrados = [os.path.relpath(p, d) for p in I.scan([d])]
        self.assertEqual(encontrados, ["a.MOV", "b.mp4", "sesion2/c.avi", "sesion2/sub/d.mkv"])
        # el mismo archivo por dos caminos (carpeta + archivo suelto) cuenta una vez
        self.assertEqual(len(I.scan([d, os.path.join(d, "b.mp4")])), 4)

    def test_acepta_un_archivo_suelto(self):
        d = self.carpeta_videos()
        p = os.path.join(d, "x.mp4")
        open(p, "w").close()
        self.assertEqual(I.scan([p]), [p])
        self.assertEqual(I.scan([os.path.join(d, "no_es_video.txt")]), [])


class LecturaTests(Base):
    def test_un_video_real_entrega_sus_metadatos_firma_huella_y_miniatura(self):
        p = hacer_video(os.path.join(self.t, "v.avi"), n=100, fps=25.0, size=(160, 90))
        v = I.Video("V1", p, os.path.getsize(p), "2026-11-05 10:00")
        I.read_video(v)
        self.assertEqual(v.error, "")
        self.assertTrue(v.abierto)
        self.assertEqual((v.ancho, v.alto), (160, 90))
        self.assertAlmostEqual(v.fps, 25.0)
        self.assertEqual(v.frames, 100)
        self.assertAlmostEqual(v.duracion_s, 4.0)
        self.assertEqual(v.firma.shape, (36, 64))
        self.assertEqual(v.miniatura.shape, (90, 160, 3))
        self.assertEqual(len(v.huella), 64)

    def test_un_archivo_que_no_es_video_no_rompe(self):
        p = os.path.join(self.t, "malo.mp4")
        with open(p, "w") as fh:
            fh.write("no soy un video")
        v = I.Video("V1", p, 15, "")
        I.read_video(v)
        self.assertFalse(v.abierto)
        self.assertEqual(v.error, "no se pudo abrir")
        self.assertIsNone(v.firma)
        self.assertEqual(len(v.huella), 64)  # la huella se calcula igual: sirve para detectar duplicados aunque el video no abra

    def test_si_no_se_puede_leer_ningun_frame(self):
        falsa = mock.MagicMock()
        falsa.isOpened.return_value = True
        falsa.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS: 25.0, cv2.CAP_PROP_FRAME_COUNT: 100, cv2.CAP_PROP_FRAME_WIDTH: 160, cv2.CAP_PROP_FRAME_HEIGHT: 90}.get(prop, 0)
        falsa.read.return_value = (False, None)
        v = I.Video("V1", "x.mp4", 1, "")
        with mock.patch.object(I.cv2, "VideoCapture", return_value=falsa), mock.patch.object(I, "video_fingerprint", return_value={"sha256_extremos": "a" * 64, "size_bytes": 1}):
            I.read_video(v)
        self.assertEqual(v.error, "no se pudo leer ningún frame")
        falsa.release.assert_called_once()

    def test_sin_cantidad_de_frames_el_mensaje_es_honesto_y_no_hay_division_por_cero(self):
        falsa = mock.MagicMock()
        falsa.isOpened.return_value = True
        falsa.get.return_value = 0  # el contenedor no informa fps ni cantidad de frames
        falsa.read.return_value = (True, np.zeros((90, 160, 3), np.uint8))
        v = I.Video("V1", "x.mp4", 1, "")
        with mock.patch.object(I.cv2, "VideoCapture", return_value=falsa), mock.patch.object(I, "video_fingerprint", return_value={"sha256_extremos": "a" * 64, "size_bytes": 1}):
            I.read_video(v)
        self.assertIsNone(v.fps)
        self.assertIsNone(v.frames)
        self.assertIsNone(v.duracion_s)
        self.assertIsNone(v.ancho)
        self.assertIn("cuántos frames", v.error)  # no se intentó leer: no debe decir que «no se pudo leer ningún frame»
        self.assertNotIn("error al leer", v.error)
        self.assertEqual(v.huella, "a" * 64)  # y la huella se conserva

    def test_si_falta_el_fps_pero_se_conoce_la_cantidad_de_frames_igual_se_muestrea(self):
        falsa = mock.MagicMock()
        falsa.isOpened.return_value = True
        falsa.get.side_effect = lambda prop: {cv2.CAP_PROP_FRAME_COUNT: 100, cv2.CAP_PROP_FRAME_WIDTH: 160, cv2.CAP_PROP_FRAME_HEIGHT: 90}.get(prop, 0)
        falsa.read.return_value = (True, np.full((90, 160, 3), 90, np.uint8))
        v = I.Video("V1", "x.mp4", 1, "")
        with mock.patch.object(I.cv2, "VideoCapture", return_value=falsa), mock.patch.object(I, "video_fingerprint", return_value={"sha256_extremos": "a" * 64, "size_bytes": 1}):
            I.read_video(v)
        self.assertEqual(v.error, "")
        self.assertIsNone(v.fps)
        self.assertIsNone(v.duracion_s)  # sin fps no se puede dar la duración, pero eso no impide leer
        self.assertEqual(v.frames, 100)
        self.assertIsNotNone(v.firma)

    def test_un_error_inesperado_al_leer_queda_anotado(self):
        falsa = mock.MagicMock()
        falsa.isOpened.return_value = True
        falsa.get.side_effect = RuntimeError("boom")
        v = I.Video("V1", "x.mp4", 1, "")
        with mock.patch.object(I.cv2, "VideoCapture", return_value=falsa), mock.patch.object(I, "video_fingerprint", return_value={"sha256_extremos": "a" * 64, "size_bytes": 1}):
            I.read_video(v)
        self.assertIn("boom", v.error)
        falsa.release.assert_called_once()  # el video se cierra aunque falle

    def test_duplicados(self):
        a, b, c, d = I.Video("V1", "a", 100, ""), I.Video("V2", "b", 100, ""), I.Video("V3", "c", 200, ""), I.Video("V4", "d", 100, "")
        a.huella = b.huella = c.huella = "h1"
        d.huella = ""
        e = I.Video("V5", "e", 100, "")  # dos videos sin huella y del mismo tamaño no se pueden declarar duplicados
        I.mark_duplicates([a, b, c, d, e])
        self.assertEqual((d.duplicado_de, e.duplicado_de), ("", ""))
        self.assertEqual((a.duplicado_de, b.duplicado_de), ("", "V1"))  # misma huella y mismo tamaño
        self.assertEqual(c.duplicado_de, "")  # misma huella pero otro tamaño: no es el mismo archivo
        self.assertEqual(d.duplicado_de, "")  # sin huella: no se puede saber

    def test_hoja_de_contactos(self):
        self.assertIsNone(I.contact_sheet([]))
        vs = [I.Video(f"V{i}", f"/x/v{i}.mp4", 1, "", grupo=1 + i % 2, duracion_s=60.0 * i) for i in range(5)]
        vs[0].miniatura = np.full((90, 160, 3), 200, np.uint8)
        hoja = I.contact_sheet(vs, cols=4, thumb_w=320)
        self.assertEqual(hoja.shape, (2 * (180 + 34), 4 * 320, 3))  # 5 videos en 4 columnas = 2 filas
        self.assertGreater(float(hoja[10:170, 10:310].mean()), 150)  # la miniatura del primero está pintada


class InventarioCompletoTests(Base):
    def crear(self):
        d = self.carpeta_videos()
        hacer_video(os.path.join(d, "a1.avi"), "gx", semilla=1)
        hacer_video(os.path.join(d, "a2.avi"), "gx", semilla=2)  # misma arena y cámara, otro recorrido y otro ruido
        hacer_video(os.path.join(d, "b.avi"), "gy", semilla=3)  # otra arena
        shutil.copyfile(os.path.join(d, "a1.avi"), os.path.join(d, "dup.avi"))
        return d

    def test_tabla_csv_duplicados_y_grupos(self):
        d = self.crear()
        with open(os.path.join(d, "._a1.avi"), "w") as fh:
            fh.write("basura de macOS")  # se ignora
        rc, texto = self.correr()
        self.assertEqual(rc, 0)
        f = self.filas()
        self.assertEqual([r["nombre"] for r in f], ["a1.avi", "a2.avi", "b.avi", "dup.avi"])
        self.assertEqual([r["id"] for r in f], ["V01", "V02", "V03", "V04"])
        self.assertEqual([r["grupo_visual"] for r in f], ["G1", "G1", "G2", "G1"])  # a1, a2 y la copia comparten arena; b no
        self.assertEqual([r["duplicado_de"] for r in f], ["", "", "", "V01"])
        self.assertEqual(f[0]["duracion"], "0:04")
        self.assertEqual((f[0]["ancho"], f[0]["alto"], f[0]["fps"]), ("160", "90", "25.00"))
        self.assertEqual(len(f[0]["huella"]), 12)
        self.assertIn("duplicado de V01", texto)
        self.assertIn("2 grupos visuales", texto)
        self.assertIn("G1: 3 videos (V01, V02, V04)", texto)
        self.assertIn("duplicados", texto)
        self.assertIn("por GRUPO", texto)

    def test_hoja_y_matriz_de_similitud(self):
        self.crear()
        self.correr()
        hoja = cv2.imread(os.path.join(self.salida, "hoja.png"))
        self.assertIsNotNone(hoja)
        self.assertEqual(hoja.shape[1], 4 * 320)
        with open(os.path.join(self.salida, "similitud.csv"), newline="", encoding="utf-8-sig") as fh:
            filas = list(csv.reader(fh))
        self.assertEqual(filas[0], ["", "V01", "V02", "V03", "V04"])
        self.assertEqual(filas[1][1], "1.000")
        self.assertGreater(float(filas[1][2]), 0.9)  # a1 ~ a2
        self.assertLess(float(filas[1][3]), 0.5)  # a1 !~ b
        self.assertEqual(filas[1][4], "1.000")  # a1 == dup

    def test_sin_miniaturas(self):
        self.crear()
        self.correr(miniaturas=False)
        self.assertFalse(os.path.exists(os.path.join(self.salida, "hoja.png")))
        self.assertTrue(os.path.exists(os.path.join(self.salida, "similitud.csv")))

    def test_el_umbral_cambia_los_grupos(self):
        self.crear()
        self.correr(umbral=1.0)
        self.assertEqual([r["grupo_visual"] for r in self.filas()], ["G1", "G2", "G3", "G1"])  # solo la copia exacta se une
        self.correr(umbral=0.0001)
        self.assertEqual({r["grupo_visual"] for r in self.filas()}, {"G1"})  # con un umbral laxo todo se junta

    def test_un_video_ilegible_se_anota_y_el_resto_se_agrupa(self):
        d = self.crear()
        with open(os.path.join(d, "roto.avi"), "w") as fh:
            fh.write("no es video")
        rc, texto = self.correr()
        self.assertEqual(rc, 0)
        f = {r["nombre"]: r for r in self.filas()}
        self.assertEqual(f["roto.avi"]["error"], "no se pudo abrir")
        self.assertEqual(f["roto.avi"]["grupo_visual"], "")
        self.assertEqual(f["a1.avi"]["grupo_visual"], "G1")
        self.assertIn("no se pudo abrir", texto)
        with open(os.path.join(self.salida, "similitud.csv"), newline="", encoding="utf-8-sig") as fh:
            sim = {r[0]: r[1:] for r in csv.reader(fh)}
        self.assertEqual(sim["V05"], ["", "", "", "", ""])  # el video roto (V05, por orden alfabético) no tiene similitud con nadie
        self.assertEqual(sim["V01"][4], "")  # y nadie la tiene con él (columna V05)

    def test_resoluciones_distintas_se_avisan(self):
        d = self.crear()
        hacer_video(os.path.join(d, "grande.avi"), "rayas", semilla=4, size=(200, 120), fps=30.0)
        rc, texto = self.correr()
        self.assertIn("combinaciones distintas de resolución y fps", texto)

    def test_modo_sin_abrir_no_toca_ningun_video(self):
        self.crear()
        with mock.patch.object(I.cv2, "VideoCapture", side_effect=AssertionError("se abrió un video")), \
                mock.patch.object(I, "video_fingerprint", side_effect=AssertionError("se leyó un video")):
            rc, texto = self.correr(sin_abrir=True)
        self.assertEqual(rc, 0)
        f = self.filas()
        self.assertEqual(len(f), 4)
        self.assertTrue(all(int(r["tam_bytes"]) > 0 for r in f))  # el tamaño exacto no se pierde aunque el archivo sea chico
        self.assertTrue(all(r["modificado"] and r["grupo_visual"] == "" and r["huella"] == "" and r["duracion"] == "" for r in f))
        self.assertFalse(os.path.exists(os.path.join(self.salida, "hoja.png")))
        self.assertFalse(os.path.exists(os.path.join(self.salida, "similitud.csv")))
        self.assertNotIn("grupos visuales", texto)

    def test_aviso_de_drive_solo_si_se_van_a_abrir(self):
        base = os.path.join(self.t, "Library", "CloudStorage", "GoogleDrive-x")
        os.makedirs(base)
        hacer_video(os.path.join(base, "a.avi"))
        rc, texto = self.correr([base])
        self.assertIn("parece estar en Google Drive", texto)
        self.assertIn("--sin-abrir", texto)
        rc, texto = self.correr([base], sin_abrir=True)
        self.assertNotIn("parece estar en Google Drive", texto)

    def test_errores_de_uso(self):
        rc, texto = self.correr([os.path.join(self.t, "no_existe")])
        self.assertEqual(rc, 2)
        self.assertIn("No existe", texto)
        vacia = os.path.join(self.t, "vacia")
        os.makedirs(vacia)
        rc, texto = self.correr([vacia])
        self.assertEqual(rc, 2)
        self.assertIn("ningún video", texto)
        self.crear()
        for u in (0.0, -1, 1.5):
            self.assertEqual(self.correr(umbral=u)[0], 2)

    def test_cli(self):
        self.crear()
        self.assertEqual(I.main([self.t + "/videos", "--salida", self.salida, "--umbral", "0.95", "--sin-miniaturas"]), 0)
        self.assertEqual(len(self.filas()), 4)
        self.assertFalse(os.path.exists(os.path.join(self.salida, "hoja.png")))
        self.assertEqual(I.main([self.t + "/videos", "--salida", self.salida, "--sin-abrir"]), 0)


if __name__ == "__main__":
    unittest.main()