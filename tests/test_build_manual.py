"""Pruebas de las fuentes ETIQUETADAS A MANO de tools/build_training_set.py (--train-manual, --valid-manual, --entradas).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_build_manual.py -v

Se usan las herramientas REALES sobre videos sintéticos: sample_frames.py crea el conjunto de frames, save_run_inputs crea la
carpeta entradas_<fecha> (de donde sale el ROI) y las etiquetas siguen la convención de label_frames.py: archivo vacío =
«sin ratón»; sin archivo = dudosa o pendiente.
"""

import importlib
import json
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from persistence.run_inputs import save_run_inputs
from tools import build_training_set as B
from tools import sample_frames as S
from video.reader import VideoReader


def _imp(nombre):
    try:
        return importlib.import_module(f"tests.{nombre}")
    except ImportError:
        return importlib.import_module(nombre)


TE = _imp("test_eval_tracking")
TB = _imp("test_build_training_set")
RI, BG = TE.RI, TE.BG
ROI = BG.ROI  # (20, 20, 260, 190) dentro de un frame de 320 x 240: un recorte de verdad


def silencio(_):
    pass


def caja(frame):
    """Una caja conocida dentro del ROI, distinta para cada frame."""
    return (60 + frame % 40, 70 + frame % 30, 30, 20)


def esperada(box, roi=ROI):
    """Etiqueta YOLO esperada para una caja del frame completo al recortar al ROI (sin margen)."""
    x, y, w, h = box
    rx, ry, rw, rh = roi
    return f"0 {(x - rx + w / 2) / rw:.6f} {(y - ry + h / 2) / rh:.6f} {w / rw:.6f} {h / rh:.6f}"


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        RI.Escenas.preparar(cls.tmp.name)
        cls.entradas = {}
        for clave, ruta in (("A", RI.Escenas.p_bg), ("B", RI.Escenas.p_col)):
            with VideoReader(ruta) as r:
                job = RI.Escenas.job_fondo(r)
                cls.entradas[clave] = save_run_inputs(os.path.join(cls.tmp.name, f"exp{clave}"), job, video_path=ruta, options=RI.Escenas.OPT_FONDO,
                                                       sample_idx=0, app_version="1.0", log=None, now=RI.AHORA)
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.w = tempfile.mkdtemp(dir=self.tmp.name)
        self.salida = os.path.join(self.w, "entrenamiento")

    def conjunto(self, claves=("A",), por_video=10, etiquetar=True, cajas=None, formato="jpg"):
        """Conjunto de frames hecho con la herramienta real y, si se pide, etiquetado. Devuelve (carpeta, [videos del manifiesto])."""
        type(self).n += 1
        val = os.path.join(self.w, f"set{self.n}")
        fuentes = S.fuentes_desde_args([], [self.entradas[c] for c in claves], 1, 0.0, None)
        self.assertEqual(S.muestrear(fuentes, val, por_video=por_video, formato=formato, out=silencio), 0)
        with open(os.path.join(val, "manifest.json"), encoding="utf-8") as fh:
            videos = json.load(fh)["videos"]
        if etiquetar:
            for v in videos:
                TE.etiquetar(val, v["id"], v["ancho"], v["alto"], {f: (cajas or caja)(f) for f in v["frames"]})
        return val, videos

    def armar(self, train_manual=(), valid_manual=(), entradas=None, **kw):
        msgs = []
        entradas = [self.entradas["A"], self.entradas["B"]] if entradas is None else entradas
        kw.setdefault("out", msgs.append)
        rc = B.build(kw.pop("train", ()), kw.pop("valid", ()), self.salida, train_manual=train_manual, valid_manual=valid_manual, entradas=entradas, **kw)
        return rc, "\n".join(msgs)

    def ficha(self):
        with open(os.path.join(self.salida, "conjunto_info.json"), encoding="utf-8") as fh:
            return json.load(fh)

    def etiqueta(self, split, stem):
        return TB.leer(os.path.join(self.salida, split, "labels", stem + ".txt")).strip()


# ======================================================================
class ArmadoBasicoTests(Base):
    def test_estructura_conteos_y_recorte_al_roi(self):
        tr, vt = self.conjunto(("A",), por_video=12)
        va, vv = self.conjunto(("B",), por_video=6)
        rc, texto = self.armar([tr], [va])
        self.assertEqual(rc, 0, msg=texto)
        self.assertEqual(TB.validar_estructura_rfdetr(self, self.salida), {"train": 12, "valid": 6})
        for f in os.listdir(os.path.join(self.salida, "train", "images")):
            self.assertEqual(cv2.imread(os.path.join(self.salida, "train", "images", f)).shape[:2], (ROI[3], ROI[2]))  # alto x ancho del ROI

    def test_cada_etiqueta_es_la_caja_dibujada_convertida_al_recorte(self):
        tr, vt = self.conjunto(("A",), por_video=12)
        va, _ = self.conjunto(("B",), por_video=4)
        self.armar([tr], [va])
        prefijo = f"{B.safe_name(vt[0]['nombre'])}_{vt[0]['sha256_extremos'][:8]}"
        for f in vt[0]["frames"]:
            TB.casi(self, self.etiqueta("train", f"{prefijo}_f{f:06d}"), esperada(caja(f)))

    def test_las_etiquetas_siguen_a_la_verdad_del_circulo(self):
        """Con el círculo real del video: la caja recortada cae sobre él (los frames sin círculo quedan como «sin ratón»)."""
        tr, vt = self.conjunto(("A",), por_video=100, etiquetar=False)
        v = vt[0]
        _, truth = BG.make_frames(90, lost={30, 31, 32})
        TE.etiquetar(tr, v["id"], v["ancho"], v["alto"], {f: (None if truth[f] is None else TE.caja_circulo(*truth[f], 13)) for f in v["frames"]})
        va, _ = self.conjunto(("B",), por_video=4)
        rc, texto = self.armar([tr], [va])
        self.assertEqual(rc, 0, msg=texto)
        perdidos = [f for f in v["frames"] if truth[f] is None]
        self.assertGreaterEqual(len(perdidos), 1)
        self.assertEqual(self.ficha()["fuentes"][0]["motivos_de_omision"], {"sin_raton": len(perdidos)})
        prefijo = f"{B.safe_name(v['nombre'])}_{v['sha256_extremos'][:8]}"
        usados = 0
        for f in v["frames"]:
            if truth[f] is None:
                continue
            _, (x, y, w, h) = TB.parse_yolo_label_line(self.etiqueta("train", f"{prefijo}_f{f:06d}"), ROI[2], ROI[3])
            self.assertLess(np.hypot(x + w / 2 - (truth[f][0] - ROI[0]), y + h / 2 - (truth[f][1] - ROI[1])), 1.5, msg=f"frame {f}")
            usados += 1
        self.assertEqual(usados, len(v["frames"]) - len(perdidos))

    def test_un_conjunto_en_png_tambien_se_arma(self):
        tr, vt = self.conjunto(("A",), por_video=5, formato="png")
        va, _ = self.conjunto(("B",), por_video=3, formato="png")
        self.assertTrue(all(f.endswith(".png") for f in os.listdir(os.path.join(tr, "images"))))
        rc, texto = self.armar([tr], [va])
        self.assertEqual(rc, 0, msg=texto)
        self.assertEqual(self.ficha()["imagenes"], {"train": 5, "valid": 3})
        # y sin recortar, un PNG sale como JPG (el entrenador espera los dos formatos, pero así queda uniforme)
        rc, _ = self.armar([tr], [va], recorte_roi=False, entradas=[], sobrescribir=True)
        self.assertEqual(rc, 0)
        self.assertTrue(all(f.endswith(".jpg") for f in os.listdir(os.path.join(self.salida, "train", "images"))))

    def test_margen_y_sin_recorte(self):
        tr, vt = self.conjunto(("A",), por_video=6)
        va, _ = self.conjunto(("B",), por_video=3)
        self.assertEqual(self.armar([tr], [va], margen_roi=10)[0], 0)
        f = sorted(os.listdir(os.path.join(self.salida, "train", "images")))[0]
        self.assertEqual(cv2.imread(os.path.join(self.salida, "train", "images", f)).shape[:2], (ROI[3] + 20, ROI[2] + 20))
        # sin recorte no hace falta ni el ROI ni --entradas: el frame completo, copiado tal cual
        rc, _ = self.armar([tr], [va], entradas=[], recorte_roi=False, sobrescribir=True)
        self.assertEqual(rc, 0)
        f = sorted(os.listdir(os.path.join(self.salida, "train", "images")))[0]
        self.assertEqual(cv2.imread(os.path.join(self.salida, "train", "images", f)).shape[:2], (240, 320))
        frame = int(f.split("_f")[1].split(".")[0])
        TB.casi(self, self.etiqueta("train", os.path.splitext(f)[0]), esperada(caja(frame), (0, 0, 320, 240)))

    def test_el_nombre_de_las_imagenes_coincide_con_el_de_sample_frames(self):
        """La herramienta de armado debe encontrar los archivos con el mismo nombre con que los escribe sample_frames."""
        tr, vt = self.conjunto(("A",), por_video=5)
        v = vt[0]
        items = B.manual_items(tr, v)
        self.assertEqual(len(items), 5)
        for (sp, nombre, img, lab), f in zip(items, sorted(v["frames"])):
            self.assertEqual(sp, "manual")
            self.assertEqual(os.path.basename(img), S.image_name(v["id"], f, v["formato"]))
            self.assertTrue(os.path.exists(img))
            self.assertEqual(os.path.splitext(os.path.basename(lab))[0], os.path.splitext(S.image_name(v["id"], f))[0])  # el de label_frames
            self.assertEqual(nombre, f"f{f:06d}")

    def test_un_conjunto_con_dos_videos_se_expande_en_dos_fuentes(self):
        tr, vt = self.conjunto(("A", "B"), por_video=5)
        self.assertEqual(len(vt), 2)
        # para validar hace falta un video distinto de A y B: se usa un dataset automático de las pruebas
        d = TB.hacer_dataset(self.w, "expZ", "20261101_100000", "ratz", "z" * 64, n_train=3, n_val=1)
        rc, texto = self.armar([tr], [], entradas=[self.entradas["A"], self.entradas["B"]], valid=[d])
        self.assertEqual(rc, 0, msg=texto)
        fuentes = self.ficha()["fuentes"]
        self.assertEqual([(f["tipo"], f["rol"]) for f in fuentes], [("auto", "valid"), ("manual", "train"), ("manual", "train")])
        self.assertEqual(sum(f["train"] for f in fuentes), 10)
        self.assertEqual(len({f["prefijo"] for f in fuentes}), 3)  # prefijos distintos: ninguna imagen se pisa

    def test_max_por_video_con_manuales(self):
        tr, vt = self.conjunto(("A",), por_video=20)
        va, _ = self.conjunto(("B",), por_video=8)
        self.armar([tr], [va], max_por_video=5)
        self.assertEqual(self.ficha()["imagenes"], {"train": 5, "valid": 5})
        TB.validar_estructura_rfdetr(self, self.salida)

    def test_cli(self):
        tr, _ = self.conjunto(("A",), por_video=6)
        va, _ = self.conjunto(("B",), por_video=3)
        rc = B.main(["--train-manual", tr, "--valid-manual", va, "--entradas", self.entradas["A"], "--entradas", self.entradas["B"], "--salida", self.salida])
        self.assertEqual(rc, 0)
        self.assertEqual(self.ficha()["imagenes"], {"train": 6, "valid": 3})
        self.assertEqual(B.main(["--salida", self.salida]), 2)  # sin ninguna fuente de entrenamiento


# ======================================================================
class EtiquetasDeTodoTipoTests(Base):
    def test_cada_clase_de_etiqueta_se_cuenta_por_separado(self):
        tr, vt = self.conjunto(("A",), por_video=10)
        v = vt[0]
        f = sorted(v["frames"])
        lab = lambda k: os.path.join(tr, "labels", S.image_name(v["id"], f[k]).rsplit(".", 1)[0] + ".txt")  # noqa: E731
        os.remove(lab(0))  # dudosa o pendiente: sin archivo
        open(lab(1), "w").close()  # sin ratón: archivo vacío
        with open(lab(2), "w") as fh:
            fh.write("0 .5 .5 .1 .1\n0 .3 .3 .1 .1\n")  # dos cajas
        with open(lab(3), "w") as fh:
            fh.write("basura")  # inválida
        with open(lab(4), "w") as fh:
            fh.write("1 0.5 0.5 0.1 0.1\n")  # una clase que no existe
        with open(os.path.join(tr, "images", S.image_name(v["id"], f[5], v["formato"])), "w") as fh:
            fh.write("no soy una imagen")
        va, _ = self.conjunto(("B",), por_video=4)
        rc, _ = self.armar([tr], [va])
        self.assertEqual(rc, 0)
        fa = self.ficha()["fuentes"][0]
        self.assertEqual(fa["motivos_de_omision"], {"sin_etiqueta": 1, "sin_raton": 1, "varias_cajas": 1, "etiqueta_invalida": 2, "imagen_ilegible": 1})
        self.assertEqual((fa["train"], fa["omitidas"], fa["etiquetas_a_mano"]), (4, 6, 4))

    def test_si_nada_queda_se_explica_por_que(self):
        tr, vt = self.conjunto(("A",), por_video=5, etiquetar=False)  # nada etiquetado todavía
        va, _ = self.conjunto(("B",), por_video=3)
        rc, texto = self.armar([tr], [va])
        self.assertEqual(rc, 2)
        self.assertIn("incompleto", texto)
        self.assertIn("sin_etiqueta", texto)  # y dice que es porque faltan etiquetas


# ======================================================================
class BusquedaDelRoiTests(Base):
    def test_sin_entradas_el_error_dice_que_hacer(self):
        tr, _ = self.conjunto(("A",), por_video=4)
        va, _ = self.conjunto(("B",), por_video=3)
        rc, texto = self.armar([tr], [va], entradas=[])
        self.assertEqual(rc, 2)
        self.assertIn("No se encontró el ROI de", texto)
        self.assertIn("--entradas", texto)
        self.assertIn("--sin-recorte", texto)
        self.assertFalse(os.path.exists(self.salida))

    def test_entradas_de_otro_video_no_sirven(self):
        tr, _ = self.conjunto(("A",), por_video=4)
        va, _ = self.conjunto(("B",), por_video=3)
        rc, texto = self.armar([tr], [va], entradas=[self.entradas["B"]])  # falta la de A
        self.assertEqual(rc, 2)
        self.assertIn("bg.avi", texto)  # nombra el video al que le falta el ROI

    def test_se_empareja_por_huella_en_cualquier_orden(self):
        tr, _ = self.conjunto(("A",), por_video=4)
        va, _ = self.conjunto(("B",), por_video=3)
        for orden in ([self.entradas["A"], self.entradas["B"]], [self.entradas["B"], self.entradas["A"]]):
            rc, _ = self.armar([tr], [va], entradas=orden, sobrescribir=os.path.exists(self.salida))
            self.assertEqual(rc, 0)
        fa = self.ficha()["fuentes"]
        self.assertEqual({f["roi_box"] and tuple(f["roi_box"]) for f in fa}, {ROI})
        self.assertTrue(all(f["roi_origen"].endswith("/inputs.json") and f["roi_origen"].startswith("entradas_") for f in fa))

    def test_se_empareja_por_nombre_si_no_hay_huella(self):
        tr, vt = self.conjunto(("A",), por_video=4)
        va, _ = self.conjunto(("B",), por_video=3)
        sin_huella = {}
        for clave, carpeta in self.entradas.items():
            copia = os.path.join(self.w, f"entradas_{clave}")
            shutil.copytree(carpeta, copia)
            with open(os.path.join(copia, "inputs.json"), encoding="utf-8") as fh:
                d = json.load(fh)
            d["video"].pop("sha256_extremos", None)
            if clave == "B":
                d["roi_box"] = [10, 10, 200, 150]  # un ROI distinto: así se nota si a un video le toca el de otro
            with open(os.path.join(copia, "inputs.json"), "w", encoding="utf-8") as fh:
                json.dump(d, fh)
            sin_huella[clave] = copia
        for man in (tr, va):  # y el manifiesto tampoco la trae
            with open(os.path.join(man, "manifest.json"), encoding="utf-8") as fh:
                m = json.load(fh)
            for v in m["videos"]:
                v.pop("sha256_extremos", None)
            with open(os.path.join(man, "manifest.json"), "w", encoding="utf-8") as fh:
                json.dump(m, fh)
        rc, texto = self.armar([tr], [va], entradas=list(sin_huella.values()))
        self.assertEqual(rc, 0, msg=texto)
        fa = self.ficha()["fuentes"]
        self.assertEqual([f["huella"] for f in fa], [None, None])
        self.assertEqual([f["prefijo"] for f in fa], ["bg", "col"])  # sin huella el prefijo es solo el nombre
        self.assertEqual([tuple(f["roi_box"]) for f in fa], [ROI, (10, 10, 200, 150)])  # a cada video, el ROI de SU corrida
        # sin huella no se puede saber si es un video reservado: con --excluir-videos-de lo dice
        reservado, _ = self.conjunto(("B",), por_video=2)
        rc, texto = self.armar([tr], [va], entradas=list(sin_huella.values()), excluir_videos_de=[reservado], sobrescribir=True)
        self.assertEqual(rc, 0, msg=texto)
        self.assertIn("No se pudo comprobar", texto)

    def test_la_huella_manda_sobre_el_nombre(self):
        """Una corrida con el MISMO nombre de archivo pero otra huella no es del mismo video."""
        tr, _ = self.conjunto(("A",), por_video=4)
        va, _ = self.conjunto(("B",), por_video=3)
        falsa = os.path.join(self.w, "entradas_falsa")
        shutil.copytree(self.entradas["A"], falsa)
        with open(os.path.join(falsa, "inputs.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        d["video"]["sha256_extremos"] = "f" * 64
        d["roi_box"] = [1, 1, 50, 50]
        with open(os.path.join(falsa, "inputs.json"), "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        rc, _ = self.armar([tr], [va], entradas=[falsa, self.entradas["A"], self.entradas["B"]])
        self.assertEqual(rc, 0)
        self.assertEqual(tuple(self.ficha()["fuentes"][0]["roi_box"]), ROI)  # no usó la de la huella distinta

    def test_el_mismo_nombre_con_otra_huella_no_es_el_mismo_video(self):
        tr, _ = self.conjunto(("A",), por_video=4)
        va, _ = self.conjunto(("B",), por_video=3)
        falsa = os.path.join(self.w, "entradas_falsa")
        shutil.copytree(self.entradas["A"], falsa)
        with open(os.path.join(falsa, "inputs.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        d["video"]["sha256_extremos"] = "f" * 64  # mismo nombre de archivo, otro contenido
        with open(os.path.join(falsa, "inputs.json"), "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        rc, texto = self.armar([tr], [va], entradas=[falsa, self.entradas["B"]])  # la única candidata para A es la falsa
        self.assertEqual(rc, 2)
        self.assertIn("No se encontró el ROI de «bg.avi»", texto)

    def test_dos_corridas_del_mismo_video_con_roi_distinto_se_avisa(self):
        tr, _ = self.conjunto(("A",), por_video=4)
        va, _ = self.conjunto(("B",), por_video=3)
        otra = os.path.join(self.w, "entradas_otra")
        shutil.copytree(self.entradas["A"], otra)
        with open(os.path.join(otra, "inputs.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        d["roi_box"] = [30, 30, 200, 150]
        with open(os.path.join(otra, "inputs.json"), "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        rc, texto = self.armar([tr], [va], entradas=[self.entradas["A"], otra, self.entradas["B"]])
        self.assertEqual(rc, 0)
        self.assertIn("2 corridas con ROI distintos", texto)
        self.assertEqual(tuple(self.ficha()["fuentes"][0]["roi_box"]), ROI)  # se usa la primera que se pasó
        rc, texto = self.armar([tr], [va], entradas=[self.entradas["A"], self.entradas["A"], self.entradas["B"]], sobrescribir=True)
        self.assertNotIn("ROI distintos", texto)  # el mismo ROI dos veces no es un problema

    def test_entradas_que_no_lo_son(self):
        tr, _ = self.conjunto(("A",), por_video=4)
        va, _ = self.conjunto(("B",), por_video=3)
        rc, texto = self.armar([tr], [va], entradas=[self.w])
        self.assertEqual(rc, 2)
        self.assertIn("no es una carpeta entradas_<fecha>", texto)
        malo = os.path.join(self.w, "entradas_sin_roi")
        shutil.copytree(self.entradas["A"], malo)
        with open(os.path.join(malo, "inputs.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        del d["roi_box"]
        with open(os.path.join(malo, "inputs.json"), "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        rc, texto = self.armar([tr], [va], entradas=[malo])
        self.assertEqual(rc, 2)
        self.assertIn("no trae roi_box", texto)


# ======================================================================
class FugasYReglasTests(Base):
    def test_el_mismo_video_a_mano_en_entrenamiento_y_validacion_es_una_fuga(self):
        tr, _ = self.conjunto(("A",), por_video=4)
        otra, _ = self.conjunto(("A",), por_video=3)  # otro muestreo del MISMO video
        rc, texto = self.armar([tr], [otra])
        self.assertEqual(rc, 2)
        self.assertIn("MISMO video", texto)
        self.assertIn("fuga", texto)
        self.assertFalse(os.path.exists(self.salida))

    def test_un_video_reservado_no_puede_entrenar_aunque_este_etiquetado_a_mano(self):
        tr, vt = self.conjunto(("A",), por_video=4)
        va, _ = self.conjunto(("B",), por_video=3)
        reservado, _ = self.conjunto(("A",), por_video=2)  # el conjunto de validación a mano que se reserva contiene el video A
        rc, texto = self.armar([tr], [va], excluir_videos_de=[reservado])
        self.assertEqual(rc, 2)
        self.assertIn("FUGA DE DATOS EVITADA", texto)
        self.assertIn(vt[0]["nombre"], texto)
        self.assertFalse(os.path.exists(self.salida))

    def test_la_misma_carpeta_dos_veces(self):
        tr, _ = self.conjunto(("A",), por_video=4)
        rc, texto = self.armar([tr, tr], [])
        self.assertEqual(rc, 2)
        self.assertIn("repetida", texto)

    def test_sin_validacion_el_error_menciona_la_opcion_manual(self):
        tr, _ = self.conjunto(("A",), por_video=4)
        rc, texto = self.armar([tr], [])
        self.assertEqual(rc, 2)
        self.assertIn("No hay imágenes de validación", texto)
        self.assertIn("--valid-manual", texto)

    def test_solo_validacion_sin_entrenamiento(self):
        va, _ = self.conjunto(("B",), por_video=3)
        rc, texto = self.armar([], [va])
        self.assertEqual(rc, 2)
        self.assertIn("--train-manual", texto)

    def test_carpetas_que_no_son_conjuntos_a_mano(self):
        rc, texto = self.armar([self.w], [])
        self.assertEqual(rc, 2)
        self.assertIn("falta manifest.json", texto)  # el mensaje claro, no un error de lectura de archivo
        mal = os.path.join(self.w, "mal")
        os.makedirs(mal)
        with open(os.path.join(mal, "manifest.json"), "w") as fh:
            fh.write("{ roto")
        self.assertEqual(self.armar([mal], [])[0], 2)
        with open(os.path.join(mal, "manifest.json"), "w") as fh:
            json.dump({"videos": []}, fh)
        rc, texto = self.armar([mal], [])
        self.assertEqual(rc, 2)
        self.assertIn("ningún video", texto)


# ======================================================================
class AvisosYMezclaTests(Base):
    def test_solo_a_mano_no_hay_avisos_de_etiquetas_automaticas(self):
        tr, _ = self.conjunto(("A",), por_video=6)
        va, _ = self.conjunto(("B",), por_video=3)
        rc, texto = self.armar([tr], [va])
        self.assertEqual(rc, 0)
        self.assertNotIn("AUTOMÁTICAS", texto)
        self.assertNotIn("borde del ROI", texto)  # eso era del generador, no de lo dibujado a mano
        self.assertNotIn("no se revisaron", texto)
        self.assertNotIn("optimista", texto)
        self.assertIn("Todas las etiquetas están dibujadas a mano (9 imágenes)", texto)
        self.assertIn("sobre el ratón y no sobre su reflejo", texto)
        self.assertIn("positivos", texto)  # lo de que no hay imágenes sin ratón sigue valiendo
        self.assertIn("Solo hay 6 imágenes de entrenamiento", texto)
        f = self.ficha()
        self.assertEqual([x["tipo"] for x in f["fuentes"]], ["manual", "manual"])
        self.assertEqual([x["etiquetas_a_mano"] for x in f["fuentes"]], [6, 3])
        self.assertTrue(all(x["modo_de_seguimiento"] == "a mano" for x in f["fuentes"]))

    def test_mezcla_con_un_dataset_automatico_sin_revisar(self):
        tr, _ = self.conjunto(("A",), por_video=6)
        auto = TB.hacer_dataset(self.w, "expZ", "20261101_100000", "ratz", "z" * 64, n_train=4, n_val=2)
        rc, texto = self.armar([tr], [], valid=[auto])
        self.assertEqual(rc, 0, msg=texto)
        self.assertIn("AUTOMÁTICAS", texto)  # lo automático sigue avisando
        self.assertIn("6 de 12 imágenes usadas no se revisaron", texto)  # y cuenta SOLO las automáticas sin revisar (6 de 6 del auto, no las 6 a mano)
        self.assertEqual([x["tipo"] for x in self.ficha()["fuentes"]], ["auto", "manual"])

    def test_mezcla_con_un_dataset_automatico_ya_revisado(self):
        from learning import review as R
        tr, _ = self.conjunto(("A",), por_video=6)
        auto = TB.hacer_dataset(self.w, "expZ", "20261101_100000", "ratz", "z" * 64, n_train=3, n_val=1)
        rev = R.new_revision()
        for k in range(3):
            o = [150.0 + k, 80.0 + k, 40.0, 30.0]
            rev["imagenes"][f"train/ratz_train_{k:03d}"] = {"estado": "aceptada", "original": o, "final": o}
        o = [150.0, 80.0, 40.0, 30.0]
        rev["imagenes"]["val/ratz_val_000"] = {"estado": "corregida", "original": o, "final": [140.0, 70.0, 60.0, 40.0]}
        R.save_revision(auto, rev)
        rc, texto = self.armar([tr], [], valid=[auto])
        self.assertEqual(rc, 0, msg=texto)
        self.assertIn("Todas las imágenes usadas fueron revisadas o dibujadas a mano (1 con la caja corregida, 6 dibujadas a mano)", texto)
        self.assertNotIn("AUTOMÁTICAS", texto)

    def test_con_validacion_manual_el_dataset_automatico_no_usa_su_validacion_interna(self):
        va, _ = self.conjunto(("B",), por_video=4)
        auto = TB.hacer_dataset(self.w, "expZ", "20261101_100000", "ratz", "z" * 64, n_train=5, n_val=2)
        rc, texto = self.armar([], [va], train=[auto])
        self.assertEqual(rc, 0, msg=texto)
        # hay una validación aparte (manual): TODO el dataset automático (5 + 2) entrena, y la validación son las 4 manuales
        self.assertEqual(self.ficha()["imagenes"], {"train": 7, "valid": 4})
        self.assertNotIn("optimista", texto)

    def test_validacion_interna_solo_si_hay_un_dataset_automatico_de_entrenamiento(self):
        tr, _ = self.conjunto(("A",), por_video=6)
        auto = TB.hacer_dataset(self.w, "expZ", "20261101_100000", "ratz", "z" * 64, n_train=4, n_val=2)
        rc, texto = self.armar([tr], [], train=[auto])  # un auto de entrenamiento y sin --valid: val interna del auto
        self.assertEqual(rc, 0, msg=texto)
        self.assertIn("optimista", texto)
        self.assertEqual(self.ficha()["imagenes"], {"train": 10, "valid": 2})  # los manuales (6) + train del auto (4); val del auto (2)


if __name__ == "__main__":
    unittest.main()