"""Pruebas de tools/build_training_set.py.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_build_training_set.py -v
"""

import importlib
import json
import math
import os
import tempfile
import unittest
from datetime import datetime

import cv2
import numpy as np

from learning.dataset import parse_yolo_label_line, yolo_label_line
from learning.session import DatasetSession
from pipeline import TrackingJob, run_tracking
from tools import build_training_set as B
from tracking.background_tracker import DEFAULT_PARAMS, BackgroundTracker, compute_background
from video.frame_selection import FrameRange, compute_timing, read_background_frames
from video.reader import VideoReader


def _imp(nombre):
    try:
        return importlib.import_module(f"tests.{nombre}")
    except ImportError:
        return importlib.import_module(nombre)

FW, FH = 400, 300
ROI = (100, 50, 200, 150)  # x, y, ancho, alto


def silencio(_):
    pass


def leer(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def imagen_degradado():
    """Cada píxel codifica su posición: canal B = x, canal G = y (suave, para que JPEG casi no la altere)."""
    img = np.zeros((FH, FW, 3), np.uint8)
    img[:, :, 0] = (np.arange(FW) * 255 // (FW - 1)).astype(np.uint8)[None, :]
    img[:, :, 1] = (np.arange(FH) * 255 // (FH - 1)).astype(np.uint8)[:, None]
    img[:, :, 2] = 128
    return img


def hacer_dataset(base, experimento, sello, video, huella, n_train=6, n_val=2, roi_en_info=True, roi_en_entradas=False, ext=".jpg"):
    """Carpeta dataset_<sello> como la deja el generador (imágenes + etiquetas por división + ficha)."""
    exp = os.path.join(base, experimento)
    ds = os.path.join(exp, f"dataset_{sello}")
    for sp, n in (("train", n_train), ("val", n_val)):
        os.makedirs(os.path.join(ds, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(ds, "labels", sp), exist_ok=True)
        for k in range(n):
            nombre = f"{video}_{sp}_{k:03d}"
            cv2.imwrite(os.path.join(ds, "images", sp, nombre + ext), imagen_degradado())
            with open(os.path.join(ds, "labels", sp, nombre + ".txt"), "w") as fh:
                fh.write(yolo_label_line((150 + k, 80 + k, 40, 30), FW, FH) + "\n")
    info = {"video": {"nombre": video + ".mp4", "size_bytes": 1000, **({"sha256_extremos": huella} if huella else {})},
            "modo_de_seguimiento": "1"}
    if roi_en_info:
        info["opciones_de_procesamiento"] = {"roi_box": list(ROI)}
    with open(os.path.join(ds, "dataset_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh)
    if roi_en_entradas:
        os.makedirs(os.path.join(exp, f"entradas_{sello}"), exist_ok=True)
        with open(os.path.join(exp, f"entradas_{sello}", "inputs.json"), "w") as fh:
            json.dump({"roi_box": list(ROI)}, fh)
    return ds


def hacer_validacion(base, nombre, huellas):
    val = os.path.join(base, nombre)
    os.makedirs(val, exist_ok=True)
    with open(os.path.join(val, "manifest.json"), "w") as fh:
        json.dump({"videos": [{"nombre": f"reservado{i}.mp4", "sha256_extremos": h} for i, h in enumerate(huellas)]}, fh)
    return val


def casi(test, linea, esperado, tol=2e-6):
    """Compara una etiqueta YOLO con la esperada número a número (las etiquetas guardan 6 decimales: el viaje de ida y vuelta deja ~1e-4 px)."""
    a, b = linea.split(), esperado.split()
    test.assertEqual(len(a), len(b), msg=linea)
    test.assertEqual(a[0], b[0])
    for x, y in zip(a[1:], b[1:]):
        test.assertAlmostEqual(float(x), float(y), delta=tol, msg=f"{linea} != {esperado}")


def validar_estructura_rfdetr(test, raiz):
    """Lo que la documentación de RF-DETR pide de un conjunto YOLO: data.yaml + train/valid con images y labels pareadas."""
    yaml = leer(os.path.join(raiz, "data.yaml"))
    test.assertIn("train: train/images", yaml)
    test.assertIn("val: valid/images", yaml)
    test.assertIn("nc: 1", yaml)
    test.assertIn("- raton", yaml)
    total = {}
    for sp in ("train", "valid"):
        imgs = sorted(os.listdir(os.path.join(raiz, sp, "images")))
        labs = sorted(os.listdir(os.path.join(raiz, sp, "labels")))
        test.assertEqual([os.path.splitext(i)[0] for i in imgs], [os.path.splitext(l)[0] for l in labs], msg=f"{sp}: imágenes y etiquetas no coinciden")
        for i in imgs:
            im = cv2.imread(os.path.join(raiz, sp, "images", i))
            test.assertIsNotNone(im)
            h, w = im.shape[:2]
            lineas = [x for x in leer(os.path.join(raiz, sp, "labels", os.path.splitext(i)[0] + ".txt")).splitlines() if x.strip()]
            test.assertEqual(len(lineas), 1)
            c, (x, y, bw, bh) = parse_yolo_label_line(lineas[0], w, h)
            test.assertEqual(c, 0)
            test.assertTrue(0 <= x and 0 <= y and x + bw <= w + 1e-6 and y + bh <= h + 1e-6, msg=f"caja fuera de {i}")
        total[sp] = len(imgs)
    return total


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = self.tmp.name
        self.salida = os.path.join(self.t, "entrenamiento")

    def tearDown(self):
        self.tmp.cleanup()

    def correr(self, train, valid=(), **kw):
        msgs = []
        rc = B.build(train, valid, self.salida, out=msgs.append, **kw)
        return rc, "\n".join(msgs)

    def ficha(self):
        return json.loads(leer(os.path.join(self.salida, "conjunto_info.json")))


class MatematicaDeEtiquetasTests(unittest.TestCase):
    def test_recorte_sin_margen_a_mano(self):
        rect = B.crop_rect(ROI, 0, FW, FH)
        self.assertEqual(rect, (100, 50, 300, 200))
        # caja (150,80,40,30) en el frame -> (50,30,40,30) en el recorte de 200x150: cx=70/200, cy=45/150, w=40/200, h=30/150
        linea = yolo_label_line((150, 80, 40, 30), FW, FH)
        casi(self, B.convert_label(linea, FW, FH, rect), "0 0.350000 0.300000 0.200000 0.200000")

    def test_recorte_con_margen_a_mano(self):
        rect = B.crop_rect(ROI, 10, FW, FH)
        self.assertEqual(rect, (90, 40, 310, 210))  # 220 x 170
        # caja en el recorte: (60,40,40,30): cx=80/220, cy=55/170, w=40/220, h=30/170
        casi(self, B.convert_label(yolo_label_line((150, 80, 40, 30), FW, FH), FW, FH, rect), "0 0.363636 0.323529 0.181818 0.176471")

    def test_el_margen_se_recorta_al_frame(self):
        self.assertEqual(B.crop_rect((0, 0, 100, 100), 20, FW, FH), (0, 0, 120, 120))
        self.assertEqual(B.crop_rect((300, 200, 100, 100), 20, FW, FH), (280, 180, 400, 300))

    def test_margen_negativo_se_trata_como_cero(self):
        self.assertEqual(B.crop_rect(ROI, -5, FW, FH), (100, 50, 300, 200))

    def test_roi_fuera_del_frame_falla(self):
        with self.assertRaises(ValueError):
            B.crop_rect((500, 400, 50, 50), 0, FW, FH)

    def test_sin_recorte_conserva_la_caja(self):
        linea = yolo_label_line((150, 80, 40, 30), FW, FH)
        self.assertEqual(B.convert_label(linea, FW, FH, None), linea)

    def test_caja_que_sobresale_del_recorte_se_recorta(self):
        rect = (100, 50, 300, 200)
        linea = yolo_label_line((90, 80, 40, 30), FW, FH)  # sale 10 px por la izquierda
        # queda x de 100 a 130 -> en el recorte (0,30,30,30) de 200x150: cx=15/200, cy=45/150, w=30/200, h=30/150
        casi(self, B.convert_label(linea, FW, FH, rect), "0 0.075000 0.300000 0.150000 0.200000")

    def test_caja_que_sobresale_por_la_derecha_y_por_abajo_se_recorta(self):
        rect = (100, 50, 300, 200)
        # derecha: x de 290 a 330 -> queda 290..300 -> en el recorte (190,30,10,30) de 200x150: cx=195/200, cy=45/150, w=10/200, h=30/150
        casi(self, B.convert_label(yolo_label_line((290, 80, 40, 30), FW, FH), FW, FH, rect), "0 0.975000 0.300000 0.050000 0.200000")
        # abajo: y de 190 a 220 -> queda 190..200 -> en el recorte (50,140,40,10): cx=70/200, cy=145/150, w=40/200, h=10/150
        casi(self, B.convert_label(yolo_label_line((150, 190, 40, 30), FW, FH), FW, FH, rect), "0 0.350000 0.966667 0.200000 0.066667")

    def test_caja_totalmente_fuera_o_clase_distinta_se_rechaza(self):
        with self.assertRaises(ValueError):
            B.convert_label(yolo_label_line((10, 10, 20, 20), FW, FH), FW, FH, (100, 50, 300, 200))
        with self.assertRaises(ValueError):
            B.convert_label("1 0.5 0.5 0.1 0.1", FW, FH, None)

    def test_evenly(self):
        self.assertEqual(B.evenly(list(range(10)), 3), [1, 5, 8])
        self.assertEqual(B.evenly(list(range(100)), 4), [12, 37, 62, 87])  # (2i+1)*100 // 8
        self.assertEqual(B.evenly([1, 2, 3], None), [1, 2, 3])
        self.assertEqual(B.evenly([1, 2, 3], 10), [1, 2, 3])
        self.assertEqual(B.evenly([1, 2, 3], 0), [])

    def test_safe_name(self):
        self.assertEqual(B.safe_name("/x/Mi video (1).mp4"), "Mi_video_1")
        self.assertEqual(B.safe_name("???.mp4"), "video")


class EstructuraYContenidoTests(Base):
    def test_estructura_que_documenta_rfdetr_y_conteos(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64, n_train=6, n_val=2)
        rc, _ = self.correr([a])
        self.assertEqual(rc, 0)
        self.assertEqual(validar_estructura_rfdetr(self, self.salida), {"train": 6, "valid": 2})

    def test_la_imagen_recortada_es_el_trozo_correcto(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64, n_train=2, n_val=1)
        self.correr([a])
        nombre = sorted(os.listdir(os.path.join(self.salida, "train", "images")))[0]
        recorte = cv2.imread(os.path.join(self.salida, "train", "images", nombre))
        self.assertEqual(recorte.shape[:2], (150, 200))  # alto x ancho del ROI
        original = imagen_degradado()
        for (cx, cy) in ((0, 0), (199, 0), (0, 149), (100, 75), (199, 149)):
            esperado = original[50 + cy, 100 + cx].astype(int)
            self.assertLessEqual(int(np.abs(recorte[cy, cx].astype(int) - esperado).max()), 6, msg=f"píxel ({cx},{cy}) no coincide con el original")

    def test_la_etiqueta_apunta_al_mismo_contenido_tras_recortar(self):
        """El objeto marcado en el recorte debe ser el mismo del frame: se comprueba con el degradado."""
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64, n_train=1, n_val=1)
        self.correr([a])
        nombre = sorted(os.listdir(os.path.join(self.salida, "train", "images")))[0]
        im = cv2.imread(os.path.join(self.salida, "train", "images", nombre))
        _, (x, y, w, h) = parse_yolo_label_line(leer(os.path.join(self.salida, "train", "labels", os.path.splitext(nombre)[0] + ".txt")).strip(), im.shape[1], im.shape[0])
        # la esquina de la caja en el recorte corresponde a (150, 80) del frame: su píxel original vale B=150*255//399, G=80*255//299
        px = im[int(round(y)), int(round(x))].astype(int)
        self.assertLessEqual(abs(px[0] - 150 * 255 // 399), 6)
        self.assertLessEqual(abs(px[1] - 80 * 255 // 299), 6)

    def test_margen_del_roi(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64, n_train=2, n_val=1)
        self.correr([a], margen_roi=10)
        nombre = sorted(os.listdir(os.path.join(self.salida, "train", "images")))[0]
        self.assertEqual(cv2.imread(os.path.join(self.salida, "train", "images", nombre)).shape[:2], (170, 220))
        self.assertEqual(self.ficha()["margen_roi_px"], 10)
        validar_estructura_rfdetr(self, self.salida)

    def test_sin_recorte_copia_los_archivos_tal_cual(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64, n_train=2, n_val=1, roi_en_info=False)  # no necesita el ROI
        rc, _ = self.correr([a], recorte_roi=False)
        self.assertEqual(rc, 0)
        orig = os.path.join(a, "images", "train", "rataA_train_000.jpg")
        copia = os.path.join(self.salida, "train", "images", [f for f in os.listdir(os.path.join(self.salida, "train", "images")) if f.endswith("train_000.jpg")][0])
        with open(orig, "rb") as f1, open(copia, "rb") as f2:
            self.assertEqual(f1.read(), f2.read())  # sin recomprimir
        self.assertIsNone(self.ficha()["margen_roi_px"])
        self.assertFalse(self.ficha()["recorte_roi"])

    def test_imagenes_png_de_entrada_salen_como_jpg(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64, n_train=2, n_val=1, ext=".png")
        self.assertEqual(self.correr([a])[0], 0)
        self.assertTrue(all(f.endswith(".jpg") for f in os.listdir(os.path.join(self.salida, "train", "images"))))
        validar_estructura_rfdetr(self, self.salida)

    def test_la_ficha_lleva_origen_y_advertencias(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64)
        self.correr([a])
        f = self.ficha()
        self.assertEqual(f["clase"], "raton")
        self.assertEqual(f["imagenes"], {"train": 6, "valid": 2})
        self.assertEqual(f["fuentes"][0]["video"], "rataA.mp4")
        self.assertEqual(f["fuentes"][0]["huella"], "a" * 64)
        self.assertEqual(f["fuentes"][0]["roi_box"], list(ROI))
        textos = " ".join(f["advertencias"])
        for clave in ("AUTOMÁTICAS", "borde del ROI", "positivos"):
            self.assertIn(clave, textos)


class RepartoPorVideoTests(Base):
    def setUp(self):
        super().setUp()
        self.a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64, n_train=6, n_val=2)
        self.b = hacer_dataset(self.t, "expB", "20261102_100000", "rataB", "b" * 64, n_train=5, n_val=3)
        self.c = hacer_dataset(self.t, "expC", "20261103_100000", "rataC", "c" * 64, n_train=4, n_val=1)

    def test_con_valid_el_video_de_validacion_va_entero_y_la_validacion_interna_se_suma_al_entrenamiento(self):
        rc, texto = self.correr([self.a, self.b], [self.c])
        self.assertEqual(rc, 0)
        # entrenamiento: A (6+2) + B (5+3) = 16; validación: todo C (4+1) = 5
        self.assertEqual(validar_estructura_rfdetr(self, self.salida), {"train": 16, "valid": 5})
        self.assertTrue(self.ficha()["validacion_aparte"])
        self.assertNotIn("optimista", texto)

    def test_ningun_frame_de_un_video_de_validacion_cae_en_entrenamiento(self):
        self.correr([self.a, self.b], [self.c])
        tr = [f for f in os.listdir(os.path.join(self.salida, "train", "images"))]
        va = [f for f in os.listdir(os.path.join(self.salida, "valid", "images"))]
        self.assertTrue(all("rataC" not in f for f in tr))
        self.assertTrue(all("rataC" in f for f in va))

    def test_sin_valid_se_usa_la_separacion_interna_y_se_avisa(self):
        rc, texto = self.correr([self.a, self.b])
        self.assertEqual(rc, 0)
        self.assertEqual(validar_estructura_rfdetr(self, self.salida), {"train": 11, "valid": 5})  # 6+5 y 2+3
        self.assertIn("optimista", texto)
        self.assertFalse(self.ficha()["validacion_aparte"])

    def test_el_mismo_video_en_entrenamiento_y_validacion_es_una_fuga(self):
        otro = hacer_dataset(self.t, "expA2", "20261104_100000", "rataA", "a" * 64)  # otra corrida del MISMO video
        rc, texto = self.correr([self.a], [otro])
        self.assertEqual(rc, 2)
        self.assertIn("MISMO video", texto)
        self.assertIn("fuga", texto)
        self.assertFalse(os.path.exists(os.path.join(self.salida, "data.yaml")))  # no escribió nada

    def test_dos_corridas_del_mismo_video_como_entrenamiento_tambien_se_rechazan(self):
        otro = hacer_dataset(self.t, "expA2", "20261104_100000", "rataA", "a" * 64)
        self.assertEqual(self.correr([self.a, otro])[0], 2)

    def test_la_misma_carpeta_dos_veces(self):
        rc, texto = self.correr([self.a, self.a])
        self.assertEqual(rc, 2)
        self.assertIn("repetida", texto)

    def test_sin_entrenamiento_o_sin_validacion(self):
        self.assertEqual(self.correr([])[0], 2)
        sin_val = hacer_dataset(self.t, "expD", "20261105_100000", "rataD", "d" * 64, n_train=4, n_val=0)
        rc, texto = self.correr([sin_val])
        self.assertEqual(rc, 2)
        self.assertIn("No hay imágenes de validación", texto)  # el aviso temprano, no el del conjunto incompleto


class ProteccionDelVideoReservadoTests(Base):
    def setUp(self):
        super().setUp()
        self.a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64)
        self.c = hacer_dataset(self.t, "expC", "20261103_100000", "rataC", "c" * 64)

    def test_un_video_reservado_en_entrenamiento_se_rechaza_y_no_se_escribe_nada(self):
        val = hacer_validacion(self.t, "validacion", ["c" * 64])
        rc, texto = self.correr([self.a, self.c], [], excluir_videos_de=[val])
        self.assertEqual(rc, 2)
        self.assertIn("FUGA DE DATOS EVITADA", texto)
        self.assertIn("reservado0.mp4", texto)
        self.assertFalse(os.path.exists(self.salida))

    def test_tambien_se_protege_la_validacion_del_entrenador(self):
        val = hacer_validacion(self.t, "validacion", ["c" * 64])
        rc, texto = self.correr([self.a], [self.c], excluir_videos_de=[val])
        self.assertEqual(rc, 2)
        self.assertIn("ni para validar", texto)

    def test_si_no_hay_coincidencia_se_arma_normalmente(self):
        val = hacer_validacion(self.t, "validacion", ["z" * 64])
        rc, _ = self.correr([self.a], [self.c], excluir_videos_de=[val])
        self.assertEqual(rc, 0)
        self.assertEqual(self.ficha()["videos_reservados_excluidos_de"], [val])

    def test_varios_conjuntos_reservados(self):
        v1, v2 = hacer_validacion(self.t, "val1", ["z" * 64]), hacer_validacion(self.t, "val2", ["y" * 64, "c" * 64])
        self.assertEqual(self.correr([self.a], [self.c], excluir_videos_de=[v1, v2])[0], 2)

    def test_si_la_ficha_no_trae_huella_se_avisa_que_no_se_pudo_comprobar(self):
        sin = hacer_dataset(self.t, "expS", "20261106_100000", "rataS", None)
        val = hacer_validacion(self.t, "validacion", ["c" * 64])
        rc, texto = self.correr([self.a, sin], [self.c], excluir_videos_de=[val])
        self.assertEqual(rc, 2)  # c es reservado
        rc, texto = self.correr([self.a, sin], [], excluir_videos_de=[val])
        self.assertEqual(rc, 0)
        self.assertIn("No se pudo comprobar", texto)

    def test_una_carpeta_que_no_es_validacion_es_un_error_claro(self):
        rc, texto = self.correr([self.a], [self.c], excluir_videos_de=[self.t])
        self.assertEqual(rc, 2)
        self.assertIn("manifest.json", texto)


class RoiYNombresTests(Base):
    def test_el_roi_viene_de_la_ficha(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64)
        self.correr([a])
        self.assertEqual(self.ficha()["fuentes"][0]["roi_origen"], "dataset_info.json")

    def test_si_la_ficha_no_lo_trae_se_busca_en_las_entradas_de_la_misma_corrida(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64, roi_en_info=False, roi_en_entradas=True)
        rc, _ = self.correr([a])
        self.assertEqual(rc, 0)
        self.assertEqual(self.ficha()["fuentes"][0]["roi_origen"], "entradas_20261101_100000/inputs.json")
        self.assertEqual(self.ficha()["fuentes"][0]["roi_box"], list(ROI))

    def test_las_entradas_de_otra_corrida_no_se_usan(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64, roi_en_info=False)
        os.makedirs(os.path.join(self.t, "expA", "entradas_20269999_000000"))
        with open(os.path.join(self.t, "expA", "entradas_20269999_000000", "inputs.json"), "w") as fh:
            json.dump({"roi_box": [1, 1, 5, 5]}, fh)
        rc, texto = self.correr([a])
        self.assertEqual(rc, 2)

    def test_sin_roi_en_ningun_lado_el_error_dice_que_hacer(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64, roi_en_info=False)
        rc, texto = self.correr([a])
        self.assertEqual(rc, 2)
        self.assertIn("--sin-recorte", texto)

    def test_dos_videos_con_el_mismo_nombre_no_se_pisan(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "video", "a" * 64, n_train=3, n_val=1)
        b = hacer_dataset(self.t, "expB", "20261102_100000", "video", "b" * 64, n_train=3, n_val=1)
        rc, _ = self.correr([a, b])
        self.assertEqual(rc, 0)
        self.assertEqual(validar_estructura_rfdetr(self, self.salida), {"train": 6, "valid": 2})  # nada se sobrescribió
        prefijos = {f["prefijo"] for f in self.ficha()["fuentes"]}
        self.assertEqual(prefijos, {"video_aaaaaaaa", "video_bbbbbbbb"})

    def test_dos_videos_con_el_mismo_nombre_y_sin_huella_se_numeran(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "video", None, n_train=3, n_val=1)
        b = hacer_dataset(self.t, "expB", "20261102_100000", "video", None, n_train=3, n_val=1)
        self.assertEqual(self.correr([a, b])[0], 0)
        self.assertEqual(validar_estructura_rfdetr(self, self.salida), {"train": 6, "valid": 2})
        self.assertEqual({f["prefijo"] for f in self.ficha()["fuentes"]}, {"video", "video_2"})


class EntradasProblematicasTests(Base):
    def setUp(self):
        super().setUp()
        self.a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64, n_train=7, n_val=2)

    def motivos(self):
        return self.ficha()["fuentes"][0]["motivos_de_omision"]

    def test_etiquetas_y_imagenes_defectuosas_se_omiten_y_se_cuentan(self):
        d = os.path.join(self.a, "labels", "train")
        os.remove(os.path.join(d, "rataA_train_000.txt"))  # sin etiqueta
        with open(os.path.join(d, "rataA_train_001.txt"), "w") as fh:
            fh.write("0 .5 .5 .1 .1\n0 .2 .2 .1 .1\n")  # dos cajas
        with open(os.path.join(d, "rataA_train_002.txt"), "w") as fh:
            fh.write("basura")  # inválida
        with open(os.path.join(d, "rataA_train_003.txt"), "w") as fh:
            fh.write("1 0.5 0.5 0.1 0.1\n")  # clase que no existe
        with open(os.path.join(self.a, "images", "train", "rataA_train_004.jpg"), "w") as fh:
            fh.write("no soy una imagen")
        rc, _ = self.correr([self.a])
        self.assertEqual(rc, 0)
        self.assertEqual(self.motivos(), {"sin_etiqueta": 1, "etiqueta_con_cero_o_varias_cajas": 1, "etiqueta_invalida": 2, "imagen_ilegible": 1})
        self.assertEqual(self.ficha()["imagenes"]["train"], 2)  # de 7, quedan 000..006 menos 5 defectuosas
        validar_estructura_rfdetr(self, self.salida)

    def test_etiqueta_vacia_se_omite(self):
        with open(os.path.join(self.a, "labels", "train", "rataA_train_000.txt"), "w") as fh:
            fh.write("")
        self.correr([self.a])
        self.assertEqual(self.motivos(), {"etiqueta_con_cero_o_varias_cajas": 1})

    def test_si_todo_se_omite_el_conjunto_queda_incompleto(self):
        for sp in ("train", "val"):
            for f in os.listdir(os.path.join(self.a, "labels", sp)):
                os.remove(os.path.join(self.a, "labels", sp, f))
        rc, texto = self.correr([self.a])
        self.assertEqual(rc, 2)

    def test_carpeta_que_no_es_un_dataset(self):
        rc, texto = self.correr([self.t])
        self.assertEqual(rc, 2)
        self.assertIn("dataset_info.json", texto)

    def test_pocas_imagenes_se_avisa(self):
        rc, texto = self.correr([self.a])
        self.assertIn("imágenes de entrenamiento: probablemente pocas", texto)  # no confundir con «Solo hay ejemplos positivos»

    def test_argumentos_invalidos(self):
        rc, texto = self.correr([self.a], margen_roi=-1)
        self.assertEqual(rc, 2)
        self.assertIn("--margen-roi", texto)
        rc, texto = self.correr([self.a], max_por_video=0)
        self.assertEqual(rc, 2)
        self.assertIn("--max-por-video", texto)


class TopeYSalidaTests(Base):
    def test_max_por_video_es_parejo_y_determinista(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64, n_train=20, n_val=8)
        self.assertEqual(self.correr([a], max_por_video=4)[0], 0)
        tr1 = sorted(os.listdir(os.path.join(self.salida, "train", "images")))
        self.assertEqual(len(tr1), 4)
        self.assertEqual(len(os.listdir(os.path.join(self.salida, "valid", "images"))), 4)
        # repartidas: índices (2i+1)*20//8 = 2, 7, 12, 17
        self.assertEqual([int(f.split("_")[-1].split(".")[0]) for f in tr1], [2, 7, 12, 17])
        rc, _ = self.correr([a], max_por_video=4, sobrescribir=True)
        self.assertEqual(sorted(os.listdir(os.path.join(self.salida, "train", "images"))), tr1)

    def test_no_pisa_una_carpeta_que_ya_existe(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64)
        self.assertEqual(self.correr([a])[0], 0)
        rc, texto = self.correr([a])
        self.assertEqual(rc, 2)
        self.assertIn("--sobrescribir", texto)

    def test_sobrescribir_rehace_el_conjunto_y_no_toca_lo_demas(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64, n_train=6, n_val=2)
        self.correr([a])
        with open(os.path.join(self.salida, "notas_mias.txt"), "w") as fh:
            fh.write("importante")
        with open(os.path.join(self.salida, "train", "images", "residuo.jpg"), "w") as fh:
            fh.write("x")
        rc, _ = self.correr([a], max_por_video=3, sobrescribir=True)
        self.assertEqual(rc, 0)
        self.assertEqual(leer(os.path.join(self.salida, "notas_mias.txt")), "importante")  # un archivo ajeno no se borra
        self.assertNotIn("residuo.jpg", os.listdir(os.path.join(self.salida, "train", "images")))  # lo viejo de train/ sí se rehace
        self.assertEqual(validar_estructura_rfdetr(self, self.salida), {"train": 3, "valid": 2})

    def test_una_carpeta_de_salida_vacia_o_inexistente_no_necesita_permiso(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64)
        os.makedirs(self.salida)
        self.assertEqual(self.correr([a])[0], 0)

    def test_cli(self):
        a = hacer_dataset(self.t, "expA", "20261101_100000", "rataA", "a" * 64)
        c = hacer_dataset(self.t, "expC", "20261103_100000", "rataC", "c" * 64)
        val = hacer_validacion(self.t, "validacion", ["z" * 64])
        rc = B.main(["--train", a, "--valid", c, "--salida", self.salida, "--excluir-videos-de", val, "--margen-roi", "5"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.ficha()["margen_roi_px"], 5)
        self.assertEqual(B.main(["--salida", self.salida]), 2)


class IntegracionConElGeneradorRealTests(unittest.TestCase):
    """Cadena completa con la salida REAL del generador (no una imitación): video sintético -> seguimiento ->
    DatasetSession -> build_training_set. Las posiciones reales del círculo se conocen en cada frame."""

    W, H = 160, 120
    ROI = (20, 10, 120, 100)

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        escena = _imp("test_dataset_session").escena_larga
        cls.datasets, cls.verdad, cls.videos = {}, {}, {}
        for nombre, n in (("a", 600), ("b", 500)):  # 24 s y 20 s a 25 fps: dos videos distintos
            frames, verdad = escena(n)
            path = os.path.join(cls.tmp.name, f"v{nombre}.avi")
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (cls.W, cls.H))
            for f in frames:
                vw.write(f)
            vw.release()
            x0, y0, w0, h0 = cls.ROI
            mask = np.full((h0, w0), 255, np.uint8)
            with VideoReader(path) as r:
                bg = compute_background(read_background_frames(r, 35), cls.ROI, mask, False)
                job = TrackingJob(r, FrameRange(0, n - 1), compute_timing(r.fps), BackgroundTracker(mask, bg, DEFAULT_PARAMS), cls.ROI,
                                  start=((60.0, 50.0), (50, 40, 20, 20)))  # el círculo arranca en (80,60) del frame = (60,50) del ROI
                sesion = DatasetSession(True, os.path.join(cls.tmp.name, f"exp_{nombre}"), video_path=path, duration_s=n / 25.0,
                                        log=None, now=lambda nombre=nombre: datetime(2026, 11, 4, 10, 0, 0 if nombre == "a" else 1))
                res = run_tracking(job, on_frame=sesion.wrap(None), keep_records=True, log=None)
            sesion.finish(res)
            cls.datasets[nombre], cls.verdad[nombre], cls.videos[nombre] = sesion.out_dir, verdad, path

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.salida = tempfile.mkdtemp(dir=self.tmp.name)
        os.rmdir(self.salida)

    def armar(self, train, valid, **kw):
        msgs = []
        rc = B.build(train, valid, self.salida, out=msgs.append, **kw)
        return rc, "\n".join(msgs)

    def test_el_generador_real_deja_los_campos_que_esta_herramienta_lee(self):
        info = B.read_info(self.datasets["a"])
        self.assertEqual(B.find_roi(self.datasets["a"], info), (self.ROI, "dataset_info.json"))
        self.assertIn("sha256_extremos", info["video"])
        self.assertGreaterEqual(info["imagenes"]["train"], 5)

    def test_las_etiquetas_recortadas_caen_sobre_el_circulo_real(self):
        rc, _ = self.armar([self.datasets["a"]], [self.datasets["b"]])
        self.assertEqual(rc, 0)
        total = validar_estructura_rfdetr(self, self.salida)
        self.assertGreaterEqual(total["train"], 5)
        self.assertGreaterEqual(total["valid"], 1)
        x0, y0, w0, h0 = self.ROI
        comprobadas = 0
        for sp, nombre_video in (("train", "a"), ("valid", "b")):
            for f in os.listdir(os.path.join(self.salida, sp, "images")):
                im = cv2.imread(os.path.join(self.salida, sp, "images", f))
                self.assertEqual(im.shape[:2], (h0, w0))  # el recorte tiene el tamaño del ROI
                frame = int(os.path.splitext(f)[0].split("_")[-1])
                linea = leer(os.path.join(self.salida, sp, "labels", os.path.splitext(f)[0] + ".txt")).strip()
                _, (x, y, w, h) = parse_yolo_label_line(linea, w0, h0)
                cx, cy = x + w / 2, y + h / 2
                vx, vy = self.verdad[nombre_video][frame]
                self.assertLess(math.dist((cx, cy), (vx - x0, vy - y0)), 3.0, msg=f"{sp}/{f}")
                comprobadas += 1
        self.assertGreaterEqual(comprobadas, 6)

    def test_el_centro_de_la_caja_coincide_con_el_circulo_visible_en_la_imagen_recortada(self):
        """Comprobación independiente de la anterior: se mira la IMAGEN (píxeles claros del círculo), no la verdad numérica."""
        self.armar([self.datasets["a"]], [self.datasets["b"]])
        f = sorted(os.listdir(os.path.join(self.salida, "train", "images")))[0]
        im = cv2.imread(os.path.join(self.salida, "train", "images", f), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.nonzero(im > 200)
        linea = leer(os.path.join(self.salida, "train", "labels", os.path.splitext(f)[0] + ".txt")).strip()
        _, (x, y, w, h) = parse_yolo_label_line(linea, im.shape[1], im.shape[0])
        self.assertLess(math.dist((x + w / 2, y + h / 2), (xs.mean(), ys.mean())), 3.0)

    def test_la_huella_real_activa_la_proteccion(self):
        info = B.read_info(self.datasets["b"])
        val = hacer_validacion(self.tmp.name, "val_reservada", [info["video"]["sha256_extremos"]])
        rc, texto = self.armar([self.datasets["a"]], [self.datasets["b"]], excluir_videos_de=[val])
        self.assertEqual(rc, 2)
        self.assertIn("FUGA DE DATOS EVITADA", texto)

    def test_un_conjunto_de_validacion_real_hecho_con_sample_frames_protege_de_verdad(self):
        """Se genera el conjunto de validación con la herramienta REAL (tools/sample_frames.py) y se comprueba que la
        protección contra fugas lee su manifest.json: si los nombres de campo no coincidieran, la protección no
        encontraría ninguna huella y daría una falsa sensación de seguridad."""
        from tools import sample_frames as SF
        val = os.path.join(self.tmp.name, "validacion_real")
        rc = SF.main([self.videos["b"], "--salida", val, "--por-video", "4"])
        self.assertEqual(rc, 0)
        reservadas = B.reserved_fingerprints([val])
        info_b = B.read_info(self.datasets["b"])
        self.assertIn(info_b["video"]["sha256_extremos"], reservadas)  # el manifiesto real SÍ trae la huella, y es la misma
        self.assertEqual(reservadas[info_b["video"]["sha256_extremos"]], "vb.avi")
        # el video reservado no puede entrenar ni validar...
        for train, valid in (([self.datasets["b"]], [self.datasets["a"]]), ([self.datasets["a"]], [self.datasets["b"]])):
            rc, texto = self.armar(train, valid, excluir_videos_de=[val])
            self.assertEqual(rc, 2)
            self.assertIn("«vb.avi»", texto)  # y el mensaje dice cuál es
        # ...y un video distinto sí puede
        self.assertNotIn(info_b["video"]["sha256_extremos"], [B.read_info(self.datasets["a"])["video"]["sha256_extremos"]])


if __name__ == "__main__":
    unittest.main()