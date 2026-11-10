"""Pruebas del cuaderno de Colab (tools/make_colab_notebook.py).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_colab_notebook.py -v

IMPORTANTE: estas pruebas NO prueban que RF-DETR funcione. Ejecutan el cuaderno completo contra DOBLES de torch, rfdetr,
onnxruntime y google.colab (que imitan la API que describe la documentación oficial), para comprobar que el código de
pegamento del cuaderno no tiene errores de nombres, orden, rutas ni lógica. Lo que depende de RF-DETR real solo se
prueba en Colab.
"""

import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

from learning import onnx_decode as OD  # noqa: E402
from tools import build_training_set as B  # noqa: E402
from tools import make_colab_notebook as MK  # noqa: E402


def _imp(nombre):
    try:
        return importlib.import_module(f"tests.{nombre}")
    except ImportError:
        return importlib.import_module(nombre)


TB = _imp("test_build_training_set")


# ======================================================================
# Dobles
# ======================================================================
class FakeOOM(RuntimeError):
    pass


def fake_torch(gpu=True, vram=16e9):
    m = types.ModuleType("torch")
    m.cuda = types.SimpleNamespace(
        is_available=lambda: gpu, get_device_name=lambda i=0: "Tesla T4 (doble)",
        get_device_properties=lambda i=0: types.SimpleNamespace(total_memory=vram),
        reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 3.2e9, empty_cache=lambda: None, OutOfMemoryError=FakeOOM)
    return m


class FakeModel:
    """Imita lo que el cuaderno usa de RFDETRNano / RFDETRSmall (según la documentación oficial)."""

    llamadas = []
    instancias = 0
    rechazar_skip = False
    rechazar_notes = False
    oom = False
    sin_detecciones = False

    @classmethod
    def reiniciar(cls):
        FakeModel.llamadas, FakeModel.instancias = [], 0
        FakeModel.rechazar_skip = FakeModel.rechazar_notes = FakeModel.oom = FakeModel.sin_detecciones = False

    def __init__(self, **kw):
        FakeModel.instancias += 1
        FakeModel.llamadas.append(("init", type(self).__name__, dict(kw)))
        self.model_config = types.SimpleNamespace(resolution=kw.get("resolution", 384))

    @classmethod
    def from_checkpoint(cls, path, **kw):
        FakeModel.llamadas.append(("from_checkpoint", os.path.basename(str(path))))
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return cls()

    def train(self, **kw):
        FakeModel.llamadas.append(("train", dict(kw)))
        if FakeModel.oom:
            raise FakeOOM("CUDA out of memory")
        if FakeModel.rechazar_skip and "skip_best_epochs" in kw:
            raise ValueError("1 validation error for TrainConfig: skip_best_epochs Extra inputs are not permitted")
        os.makedirs(kw["output_dir"], exist_ok=True)
        with open(os.path.join(kw["output_dir"], "checkpoint_best_total.pth"), "wb") as fh:
            fh.write(b"pesos")
        with open(os.path.join(kw["output_dir"], "training_config.json"), "w") as fh:
            json.dump({"epochs": kw["epochs"]}, fh)

    def evaluate(self, **kw):
        FakeModel.llamadas.append(("evaluate", dict(kw)))
        return {"val/mAP_50_95": 0.61, "val/mAP_50": 0.93}

    def predict(self, imagen, threshold=0.5):
        if FakeModel.sin_detecciones:
            return types.SimpleNamespace(xyxy=np.zeros((0, 4)), confidence=np.zeros(0))
        return types.SimpleNamespace(xyxy=np.array([[40.0, 20.0, 100.0, 60.0]]), confidence=np.array([0.9]))

    def export(self, output_dir="output", notes=None, **kw):
        FakeModel.llamadas.append(("export", {"output_dir": output_dir, "notes": notes}))
        if FakeModel.rechazar_notes and notes is not None:
            raise TypeError("export() got an unexpected keyword argument 'notes'")
        os.makedirs(output_dir, exist_ok=True)
        ruta = Path(output_dir) / "inference_model.onnx"
        ruta.write_bytes(b"onnx falso")
        return ruta


class FakeNano(FakeModel):
    pass


class FakeSmall(FakeModel):
    pass


def fake_rfdetr():
    m = types.ModuleType("rfdetr")
    m.RFDETRNano, m.RFDETRSmall = FakeNano, FakeSmall
    return m


class _IO:
    def __init__(self, name, shape=None):
        self.name, self.shape = name, shape


def fake_onnxruntime(desvio=False):
    """Una sesión que devuelve, sin decodificar, la misma caja que predice FakeModel (en un recorte de 200x150)."""
    m = types.ModuleType("onnxruntime")

    class Sesion:
        def __init__(self, ruta, providers=None):
            self.ruta = ruta

        def get_inputs(self):
            return [_IO("input", [1, 3, 16, 24])]

        def get_outputs(self):
            return [_IO("dets"), _IO("labels")]

        def run(self, nombres, feed):
            cx = 0.85 if desvio else 70 / 200
            return [np.array([[[cx, 40 / 150, 60 / 200, 40 / 150]]]), np.array([[[3.0, -3.0]]])]

    m.InferenceSession = Sesion
    return m


class FakeColab:
    def __init__(self, zip_path):
        self.zip_path, self.subidas, self.descargas, self.montajes = zip_path, 0, [], []
        outer = self

        def upload():
            outer.subidas += 1
            with open(outer.zip_path, "rb") as fh:
                datos = fh.read()
            nombre = os.path.basename(outer.zip_path)
            with open(os.path.join(os.getcwd(), nombre), "wb") as fh:  # Colab guarda lo subido en la carpeta actual
                fh.write(datos)
            return {nombre: datos}

        self.files = types.SimpleNamespace(upload=upload, download=lambda ruta: outer.descargas.append(ruta))
        self.drive = types.SimpleNamespace(mount=lambda ruta: outer.montajes.append(ruta))

    def modulos(self):
        colab = types.ModuleType("google.colab")
        colab.files, colab.drive = self.files, self.drive
        google = types.ModuleType("google")
        google.colab = colab
        return {"google": google, "google.colab": colab}


# ======================================================================
# Ejecutor del cuaderno
# ======================================================================
def transformar(fuente):
    """Las líneas ``!comando`` y ``%magia`` solo existen en Jupyter: se reemplazan por algo que Python entienda."""
    salida = []
    for ln in fuente.split("\n"):
        s = ln.lstrip()
        sangria = ln[: len(ln) - len(s)]
        if s.startswith("!"):
            salida.append(f'{sangria}_shell(f"""{s[1:]}""")')
        elif s.startswith("%"):
            salida.append(f"{sangria}pass")
        else:
            salida.append(ln)
    return "\n".join(salida)


def codigo(nb):
    return [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]


def indice(nb, marca):
    """Posición, entre las celdas de CÓDIGO, de la primera que contiene ``marca``."""
    for k, src in enumerate(codigo(nb)):
        if marca in src:
            return k
    raise KeyError(marca)


class Ejecucion:
    def __init__(self, nb):
        self.nb, self.comandos, self.out = nb, [], io.StringIO()
        self.ns = {"__name__": "__main__", "_shell": self.comandos.append}

    def celda(self, k):
        src = transformar(codigo(self.nb)[k])
        with contextlib.redirect_stdout(self.out):
            exec(compile(src, f"<celda de codigo {k}>", "exec"), self.ns)

    def todo(self, tras=None, hasta=None):
        tras = tras or {}
        n = len(codigo(self.nb)) if hasta is None else hasta + 1
        for k in range(n):
            self.celda(k)
            if k in tras:
                tras[k](self.ns)

    @property
    def texto(self):
        return self.out.getvalue()


# ======================================================================
class EstructuraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = MK.build_notebook()

    def test_formato_de_cuaderno(self):
        nb = self.nb
        self.assertEqual(nb["nbformat"], 4)
        self.assertEqual(nb["nbformat_minor"], 4)  # sin identificadores de celda: lo abre cualquier versión
        self.assertEqual(nb["metadata"]["accelerator"], "GPU")
        self.assertEqual(nb["metadata"]["kernelspec"]["name"], "python3")
        for c in nb["cells"]:
            self.assertIn(c["cell_type"], ("markdown", "code"))
            self.assertIsInstance(c["source"], list)
            self.assertTrue(all(isinstance(s, str) for s in c["source"]))
            self.assertNotIn("id", c)
            if c["cell_type"] == "code":
                self.assertEqual((c["outputs"], c["execution_count"]), ([], None))
            for s in c["source"][:-1]:
                self.assertTrue(s.endswith("\n"))
            self.assertFalse(c["source"][-1].endswith("\n"))
        self.assertEqual(json.loads(json.dumps(nb)), nb)

    def test_cada_celda_de_codigo_es_sintacticamente_valida(self):
        import ast
        for k, src in enumerate(codigo(self.nb)):
            try:
                ast.parse(transformar(src))
            except SyntaxError as exc:
                self.fail(f"celda de código {k}: {exc}")

    def test_orden_de_los_pasos(self):
        marcas = ["CONFIGURACIÓN", "Paso 1 · Instalar", "Paso 2 · Comprobar la GPU", "Paso 3 · Carpetas", "Paso 4 · Traer", "Paso 5 · Revisar",
                  "learning/onnx_decode.py (fuente única", "Paso 6 · Prueba rápida", "Paso 7 · Entrenamiento real", "Paso 8 · El mejor punto",
                  "Paso 9 · Cajas del detector", "Paso 10 · Exportar", "Paso 11 · ¿El ONNX", "Paso 12 · Guardar"]
        pos = [indice(self.nb, m) for m in marcas]
        self.assertEqual(pos, sorted(pos))
        self.assertEqual(len(set(pos)), len(pos))

    def test_la_decodificacion_incrustada_es_la_del_modulo(self):
        celda = codigo(self.nb)[indice(self.nb, "fuente única")]
        cabecera, resto = celda.split("\n", 1)
        self.assertIn("no editar aquí", cabecera)
        with open(os.path.join(MK.ROOT, "learning", "onnx_decode.py"), encoding="utf-8") as fh:
            modulo = fh.read()
        self.assertEqual(resto.strip("\n"), modulo.replace("from __future__ import annotations\n", "").strip("\n"))
        self.assertNotIn("__future__", celda)

    def test_el_codigo_incrustado_se_ejecuta_solo_y_define_lo_que_el_cuaderno_usa(self):
        ns = {}
        exec(compile(codigo(self.nb)[indice(self.nb, "fuente única")], "<decodificador>", "exec"), ns)
        for nombre in ("detect", "decode", "preprocess", "find_outputs", "sigmoid", "Detections"):
            self.assertIn(nombre, ns)
        d = ns["decode"](np.array([[0.5, 0.5, 0.2, 0.4]]), np.array([[2.0, -3.0]]), 200, 100, 0.5)
        np.testing.assert_allclose(d.xyxy, [[80, 30, 120, 70]], atol=1e-4)  # y funciona igual que el módulo
        np.testing.assert_allclose(OD.decode(np.array([[0.5, 0.5, 0.2, 0.4]]), np.array([[2.0, -3.0]]), 200, 100, 0.5).xyxy, d.xyxy)

    def test_el_script_escribe_el_cuaderno(self):
        with tempfile.TemporaryDirectory() as t:
            ruta = os.path.join(t, "sub", "n.ipynb")
            msgs = []
            self.assertEqual(MK.main(["--salida", ruta], out=msgs.append), 0)
            with open(ruta, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), self.nb)
            self.assertFalse(os.path.exists(ruta + ".tmp"))
            self.assertIn("Colab", "\n".join(msgs))
            bloqueada = os.path.join(t, "archivo")
            open(bloqueada, "w").close()
            self.assertEqual(MK.main(["--salida", os.path.join(bloqueada, "n.ipynb")], out=msgs.append), 2)

    def test_el_cuaderno_de_la_carpeta_colab_esta_al_dia(self):
        ruta = MK.DEFAULT_OUT
        if not os.path.exists(ruta):
            self.skipTest("todavía no se generó colab/entrenar_rfdetr.ipynb")
        with open(ruta, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh), self.nb, msg="Vuelve a correr: python tools/make_colab_notebook.py")

    def test_decode_source_exige_un_unico_from_future(self):
        with mock.patch.object(MK, "DECODE_PATH", __file__):
            with self.assertRaises(RuntimeError):
                MK.decode_source()


# ======================================================================
class EjecucionCompletaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = cls.tmp.name
        A = TB.hacer_dataset(t, "expA", "20261101_100000", "rata", "a" * 64, n_train=8, n_val=2)
        Bd = TB.hacer_dataset(t, "expB", "20261102_100000", "ratb", "b" * 64, n_train=3, n_val=3)
        cls.conjunto = os.path.join(t, "entrenamiento")
        assert B.build([A], [Bd], cls.conjunto, out=lambda m: None) == 0
        cls.zip = os.path.join(t, "entrenamiento.zip")
        with zipfile.ZipFile(cls.zip, "w") as z:
            for raiz, _, fs in os.walk(cls.conjunto):
                for f in fs:
                    z.write(os.path.join(raiz, f), os.path.join("entrenamiento", os.path.relpath(os.path.join(raiz, f), cls.conjunto)))
        cls.nb = MK.build_notebook()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        FakeModel.reiniciar()
        self.trabajo = tempfile.mkdtemp(dir=self.tmp.name)
        self.cwd = os.getcwd()
        os.chdir(self.trabajo)  # la subida de Colab deja el zip en la carpeta actual
        self.colab = FakeColab(self.zip)

    def tearDown(self):
        os.chdir(self.cwd)
        import matplotlib.pyplot as plt
        plt.close("all")

    def entorno(self, gpu=True, vram=16e9, desvio=False):
        mods = {"torch": fake_torch(gpu, vram), "rfdetr": fake_rfdetr(), "onnxruntime": fake_onnxruntime(desvio), **self.colab.modulos()}
        pila = contextlib.ExitStack()
        pila.enter_context(mock.patch.dict(sys.modules, mods))
        pila.enter_context(mock.patch("importlib.metadata.version", return_value="9.9.9"))
        return pila

    def ajustar(self, **extra):
        raiz_drive, raiz_local = os.path.join(self.trabajo, "drive"), os.path.join(self.trabajo, "local")
        os.makedirs(raiz_local, exist_ok=True)
        base = dict(RAIZ_DRIVE=raiz_drive, RAIZ_LOCAL=raiz_local, EPOCAS_PRUEBA=1, EPOCAS=5, EPOCAS_A_IGNORAR=2, GUARDAR_EN_DRIVE=False)  # valores TODOS distintos: no se pueden confundir
        base.update(extra)
        return lambda ns: ns.update(base)

    def correr(self, hasta=None, gpu=True, vram=16e9, desvio=False, **cfg):
        ej = Ejecucion(self.nb)
        with self.entorno(gpu, vram, desvio):
            ej.todo(tras={0: self.ajustar(**cfg)}, hasta=hasta)
        return ej

    def trenes(self):
        return [l[1] for l in FakeModel.llamadas if l[0] == "train"]

    # ------------------------------------------------------------------ camino feliz
    def test_el_cuaderno_corre_completo_y_deja_los_resultados(self):
        ej = self.correr()
        ns = ej.ns
        entrega = ns["ENTREGA"]
        for f in ("checkpoint_best_total.pth", "inference_model.onnx", "training_config.json", "conjunto_info.json", "run_info.json", ns["NOMBRE_CORRIDA"] + ".zip"):
            self.assertTrue(os.path.exists(os.path.join(entrega, f)), msg=f)
        with zipfile.ZipFile(os.path.join(entrega, ns["NOMBRE_CORRIDA"] + ".zip")) as z:
            self.assertEqual(sorted(z.namelist()), ["checkpoint_best_total.pth", "conjunto_info.json", "inference_model.onnx", "run_info.json", "training_config.json"])
        self.assertIn("El ONNX coincide con el modelo original.", ej.texto)
        self.assertNotIn("ADVERTENCIA", ej.texto)

    def test_instalacion_version_y_comando(self):
        ej = self.correr(hasta=1)
        self.assertEqual(ej.comandos, ['pip install -q "rfdetr[train,onnx]" onnxruntime'])
        self.assertEqual(ej.ns["VERSION_INSTALADA"], "9.9.9")
        self.assertIn("rfdetr 9.9.9", ej.texto)
        ej = self.correr(hasta=1, VERSION_RFDETR="1.2.3")
        self.assertEqual(ej.comandos, ['pip install -q "rfdetr[train,onnx]==1.2.3" onnxruntime'])  # para reproducir un entrenamiento

    def test_la_version_desconocida_no_rompe(self):
        ej = Ejecucion(self.nb)
        with self.entorno(), mock.patch("importlib.metadata.version", side_effect=Exception("no instalado")):
            ej.todo(tras={0: self.ajustar()}, hasta=1)
        self.assertEqual(ej.ns["VERSION_INSTALADA"], "desconocida")

    def test_argumentos_del_entrenamiento(self):
        self.correr()
        prueba, real = self.trenes()
        for kw in (prueba, real):
            self.assertEqual((kw["batch_size"], kw["grad_accum_steps"]), (4, 4))  # lote efectivo 16, como recomienda la documentación
            self.assertTrue(kw["dataset_dir"].endswith("entrenamiento"))
            self.assertNotIn("lr", kw)  # None = el de fábrica
        self.assertEqual(prueba["epochs"], 1)
        self.assertNotIn("early_stopping", prueba)
        self.assertIn(os.sep + "prueba", prueba["output_dir"])
        self.assertEqual(real["epochs"], 5)
        self.assertTrue(real["early_stopping"])
        self.assertEqual((real["early_stopping_patience"], real["skip_best_epochs"]), (10, 2))
        self.assertTrue(real["output_dir"].endswith("entrenamiento"))
        self.assertNotEqual(prueba["output_dir"], real["output_dir"])

    def test_lr_solo_se_pasa_si_se_pide(self):
        self.correr(LR=5e-5)
        self.assertTrue(all(kw["lr"] == 5e-5 for kw in self.trenes()))

    def test_el_entrenamiento_real_parte_de_un_modelo_nuevo(self):
        self.correr()
        # 1 para la prueba, 1 NUEVO para el entrenamiento real (parte de los pesos preentrenados, no de los de la prueba) y
        # 1 cargado del mejor checkpoint, que sirve para evaluar, comparar y exportar
        self.assertEqual(FakeModel.instancias, 3)
        orden = [l[0] for l in FakeModel.llamadas if l[0] in ("init", "train", "from_checkpoint")]
        self.assertEqual(orden, ["init", "train", "init", "train", "from_checkpoint", "init"])

    def test_modelo_y_resolucion(self):
        self.correr(MODELO="nano")
        self.assertEqual([l[1] for l in FakeModel.llamadas if l[0] == "init"][0], "FakeNano")
        FakeModel.reiniciar()
        self.correr(MODELO="small", RESOLUCION=448)
        primero = [l for l in FakeModel.llamadas if l[0] == "init"][0]
        self.assertEqual((primero[1], primero[2]), ("FakeSmall", {"resolution": 448}))
        FakeModel.reiniciar()
        self.correr()
        self.assertEqual([l for l in FakeModel.llamadas if l[0] == "init"][0][2], {})  # sin resolución: la de fábrica

    def test_se_evalua_el_mejor_punto_en_validacion(self):
        self.correr()
        self.assertIn(("from_checkpoint", "checkpoint_best_total.pth"), FakeModel.llamadas)
        evals = [l[1] for l in FakeModel.llamadas if l[0] == "evaluate"]
        self.assertEqual(len(evals), 1)
        self.assertEqual(evals[0]["split"], "val")
        self.assertTrue(evals[0]["dataset_dir"].endswith("entrenamiento"))

    def test_las_notas_del_onnx_llevan_la_procedencia(self):
        self.correr(MODELO="nano")
        exp = [l[1] for l in FakeModel.llamadas if l[0] == "export"][0]
        n = exp["notes"]
        self.assertEqual((n["clase"], n["modelo"], n["version_rfdetr"], n["umbral_sugerido"]), ("raton", "nano", "9.9.9", 0.5))
        self.assertIs(n["recorte_roi"], True)
        self.assertEqual(n["margen_roi_px"], 0)
        self.assertEqual(n["resolucion"], 384)
        self.assertEqual(n["imagenes"], {"train": 10, "valid": 6})

    def test_run_info(self):
        ej = self.correr()
        with open(os.path.join(ej.ns["ENTREGA"], "run_info.json"), encoding="utf-8") as fh:
            ri = json.load(fh)
        self.assertEqual((ri["modelo"], ri["version_rfdetr"], ri["resolucion"], ri["gpu"]), ("small", "9.9.9", 384, "Tesla T4 (doble)"))
        self.assertEqual(ri["vram_gb"], 16.0)
        self.assertEqual(ri["parametros"], {"epocas": 5, "lote": 4, "acumulacion": 4, "lr": None, "paciencia": 10, "epocas_a_ignorar": 2, "umbral": 0.5})
        self.assertEqual(ri["metricas_validacion"], {"val/mAP_50_95": 0.61, "val/mAP_50": 0.93})
        self.assertEqual(ri["comparacion_en_validacion"]["imagenes"], 6)
        self.assertEqual(ri["verificacion_onnx"]["uno_detecta_y_otro_no"], 0)
        self.assertIn("no es la evaluación final", ri["aviso"])
        self.assertTrue(ri["conjunto"]["advertencias"])
        self.assertEqual(ri["notas_en_el_onnx"]["corrida"], ri["corrida"])

    # ------------------------------------------------------------------ dataset
    def test_trae_y_revisa_el_conjunto(self):
        ej = self.correr(hasta=5)
        self.assertTrue(ej.ns["DATASET_DIR"].endswith("entrenamiento"))  # encontró la carpeta con data.yaml aunque el zip traiga una carpeta arriba
        self.assertEqual(self.colab.subidas, 1)
        self.assertIn("Entrenamiento: 10 imágenes | validación: 6", ej.texto)
        self.assertIn("mide, en mediana, 40 x 30 px", ej.texto)  # la caja de los datos de prueba mide 40 x 30
        self.assertIn("AVISO: solo 10 imágenes de entrenamiento", ej.texto)
        self.assertIn("nc: 1", ej.texto)
        self.assertIn("(train): train", ej.texto)

    def test_zip_en_drive_no_pide_subir_nada(self):
        destino = os.path.join(self.trabajo, "drive", "TrackerLab_Entrenamiento")
        os.makedirs(destino)
        import shutil
        shutil.copyfile(self.zip, os.path.join(destino, "entrenamiento.zip"))
        ej = self.correr(hasta=5, ZIP_EN_DRIVE="TrackerLab_Entrenamiento/entrenamiento.zip")
        self.assertEqual(self.colab.subidas, 0)
        self.assertEqual(len(self.colab.montajes), 1)  # monta Drive para poder leerlo
        self.assertTrue(ej.ns["DATASET_DIR"].endswith("entrenamiento"))

    def test_sin_subir_nada_falla_con_un_mensaje(self):
        self.colab.files.upload = lambda: {}
        with self.assertRaisesRegex(RuntimeError, "No se subió ningún archivo"):
            self.correr(hasta=4)

    def malo(self, nombre, contenido_zip):
        ruta = os.path.join(self.trabajo, nombre)
        with zipfile.ZipFile(ruta, "w") as z:
            for n, datos in contenido_zip.items():
                z.writestr(n, datos)
        self.colab.zip_path = ruta

    def test_zip_sin_o_con_dos_data_yaml(self):
        self.malo("sin.zip", {"a/algo.txt": "x"})
        with self.assertRaisesRegex(RuntimeError, "exactamente una carpeta con data.yaml \\(encontré 0\\)"):
            self.correr(hasta=4)
        self.malo("dos.zip", {"a/data.yaml": "x", "b/data.yaml": "y"})
        with self.assertRaisesRegex(RuntimeError, "encontré 2"):
            self.correr(hasta=4)

    def zip_con_primera_etiqueta(self, nombre, contenido):
        """Copia del zip donde solo cambia la etiqueta de la primera imagen de entrenamiento."""
        ruta = os.path.join(self.trabajo, nombre + ".zip")
        with zipfile.ZipFile(self.zip) as z, zipfile.ZipFile(ruta, "w") as w:
            primera = sorted(n for n in z.namelist() if "/train/labels/" in n and n.endswith(".txt"))[0]
            for n in z.namelist():
                w.writestr(n, contenido if n == primera else z.read(n))
        return ruta

    def test_cada_tipo_de_etiqueta_mala_detiene_antes_de_entrenar(self):
        malas = {"otra clase": b"1 0.5 0.5 0.2 0.2\n", "ancho mayor que 1": b"0 0.5 0.5 1.7 0.2\n", "ancho cero": b"0 0.5 0.5 0 0.2\n",
                 "alto cero": b"0 0.5 0.5 0.2 0\n", "centro fuera": b"0 1.5 0.5 0.2 0.2\n", "centro negativo": b"0 -0.1 0.5 0.2 0.2\n",
                 "dos cajas": b"0 .5 .5 .2 .2\n0 .3 .3 .1 .1\n", "vacia": b"", "pocos datos": b"0 0.5 0.5\n", "texto": b"hola\n"}
        for nombre, contenido in malas.items():
            self.colab.zip_path = self.zip_con_primera_etiqueta("mala", contenido)
            with self.assertRaisesRegex(RuntimeError, "1 imágenes con problemas en sus etiquetas", msg=nombre):
                self.correr(hasta=5)
            self.assertEqual(self.trenes(), [], msg=nombre)  # no se entrenó nada

    def test_los_limites_validos_de_una_etiqueta_no_se_rechazan(self):
        for nombre, contenido in {"caja que ocupa todo": b"0 0.5 0.5 1.0 1.0\n", "centro en la esquina": b"0 0 0 0.2 0.2\n", "con BOM y espacios": b"\xef\xbb\xbf0 0.5 0.5 0.2 0.2  \n\n"}.items():
            self.colab.zip_path = self.zip_con_primera_etiqueta("limite", contenido)
            ej = self.correr(hasta=5)
            self.assertIn("Entrenamiento: 10 imágenes", ej.texto, msg=nombre)

    def test_conjunto_sin_validacion(self):
        with zipfile.ZipFile(self.zip) as z:
            ruta = os.path.join(self.trabajo, "sinval.zip")
            with zipfile.ZipFile(ruta, "w") as w:
                for n in z.namelist():
                    if "/valid/" not in n or n.endswith("/images/") or n.endswith("/labels/"):
                        w.writestr(n, z.read(n))
        self.colab.zip_path = ruta
        with self.assertRaises(RuntimeError):
            self.correr(hasta=5)

    def test_una_carpeta_que_existe_pero_esta_vacia(self):
        """Distinto de que falte la carpeta: aquí las carpetas están, pero sin imágenes."""
        for vaciar, nombre in (("/valid/", "valid"), ("/train/", "train")):
            ruta = os.path.join(self.trabajo, f"vacio_{nombre}.zip")
            with zipfile.ZipFile(self.zip) as z, zipfile.ZipFile(ruta, "w") as w:
                for n in z.namelist():
                    if vaciar not in n:
                        w.writestr(n, z.read(n))
                for sub in ("images", "labels"):
                    w.writestr(f"entrenamiento/{nombre}/{sub}/", "")  # la carpeta existe, vacía
            self.colab.zip_path = ruta
            with self.assertRaisesRegex(RuntimeError, "al menos una imagen de entrenamiento y una de validación", msg=nombre):
                self.correr(hasta=5)

    # ------------------------------------------------------------------ GPU y lote
    def test_sin_gpu_se_detiene_con_instrucciones(self):
        with self.assertRaisesRegex(RuntimeError, "No hay GPU.*Entorno de ejecución"):
            self.correr(hasta=2, gpu=False)

    def test_lote_sugerido_segun_la_memoria(self):
        for vram, esperado in ((8e9, "LOTE=1, ACUMULACION=16"), (12e9, "LOTE=2, ACUMULACION=8"), (16e9, "LOTE=4, ACUMULACION=4"),
                               (24e9, "LOTE=8, ACUMULACION=2"), (40e9, "LOTE=16, ACUMULACION=1")):
            ej = self.correr(hasta=2, vram=vram)
            self.assertIn("Lote sugerido para esta GPU: " + esperado, ej.texto, msg=str(vram))
        for vram, esperado in ((15.0e9, "LOTE=4"), (14.9e9, "LOTE=2"), (22.0e9, "LOTE=8"), (21.9e9, "LOTE=4"), (38.0e9, "LOTE=16"), (37.9e9, "LOTE=8"),
                               (11.0e9, "LOTE=2"), (10.9e9, "LOTE=1"), (16.1e9, "LOTE=4"), (22.5e9, "LOTE=8"), (11.4e9, "LOTE=2")):  # las tres últimas: T4, L4 y K80 reales
            ej = self.correr(hasta=2, vram=vram)
            self.assertIn("Lote sugerido para esta GPU: " + esperado, ej.texto, msg=str(vram))  # los límites exactos
        ej = self.correr(hasta=2, vram=16e9)
        self.assertNotIn("ajusta LOTE", ej.texto)  # con la configuración ya adecuada no molesta
        ej = self.correr(hasta=2, vram=8e9)
        self.assertIn("ajusta LOTE y ACUMULACION", ej.texto)

    # ------------------------------------------------------------------ ramas de fallo
    def test_si_la_version_no_acepta_skip_best_epochs_se_entrena_sin_ella(self):
        FakeModel.rechazar_skip = True
        ej = self.correr()
        trenes = self.trenes()
        self.assertEqual(len(trenes), 3)  # prueba + intento con skip_best_epochs + reintento sin ella
        self.assertIn("skip_best_epochs", trenes[1])
        self.assertNotIn("skip_best_epochs", trenes[2])
        self.assertTrue(trenes[2]["early_stopping"])  # lo demás se conserva
        self.assertIn("no acepta skip_best_epochs", ej.texto)
        self.assertTrue(os.path.exists(os.path.join(ej.ns["ENTREGA"], "run_info.json")))

    def test_un_error_de_valor_ajeno_a_skip_no_se_traga(self):
        def malo(self, **kw):
            raise ValueError("el conjunto no tiene el formato esperado")
        with mock.patch.object(FakeModel, "train", malo):
            with self.assertRaisesRegex(ValueError, "formato esperado"):
                self.correr(hasta=7)

    def test_un_error_ajeno_durante_el_entrenamiento_real_no_se_reintenta(self):
        """Solo un rechazo de skip_best_epochs justifica reintentar; cualquier otro error del entrenamiento real se propaga."""
        original = FakeModel.train

        def falla_en_el_real(self, **kw):
            if "early_stopping" in kw:
                FakeModel.llamadas.append(("train", dict(kw)))
                raise ValueError("el conjunto no tiene el formato esperado")
            return original(self, **kw)
        with mock.patch.object(FakeModel, "train", falla_en_el_real):
            with self.assertRaisesRegex(ValueError, "formato esperado"):
                self.correr(hasta=8)
        self.assertEqual(len(self.trenes()), 2)  # la prueba y UN intento del real: no hubo reintento

    def test_sin_memoria_se_explica_y_se_relanza(self):
        FakeModel.oom = True
        ej = Ejecucion(self.nb)
        with self.assertRaises(FakeOOM):
            with self.entorno():
                ej.todo(tras={0: self.ajustar()})
        self.assertIn("Se acabó la memoria de la GPU", ej.texto)
        self.assertIn("Baja LOTE", ej.texto)

    def test_si_la_version_no_acepta_notes_se_exporta_sin_metadatos(self):
        FakeModel.rechazar_notes = True
        ej = self.correr()
        exports = [l[1] for l in FakeModel.llamadas if l[0] == "export"]
        self.assertEqual(len(exports), 2)
        self.assertIsNone(exports[1]["notes"])
        self.assertIn("no acepta notes", ej.texto)
        self.assertTrue(os.path.exists(ej.ns["RUTA_ONNX"]))

    def test_un_typeerror_ajeno_a_notes_no_se_reintenta(self):
        def export_roto(self, output_dir="output", notes=None, **kw):
            FakeModel.llamadas.append(("export", {"output_dir": output_dir, "notes": notes}))
            raise TypeError("unsupported operand type(s) for +: 'int' and 'str'")
        with mock.patch.object(FakeModel, "export", export_roto):
            with self.assertRaisesRegex(TypeError, "unsupported operand"):
                self.correr(hasta=indice(self.nb, "Paso 10 · Exportar"))
        self.assertEqual(len([l for l in FakeModel.llamadas if l[0] == "export"]), 1)  # un solo intento: no se tapó el error real

    def test_sin_el_mejor_checkpoint_el_error_dice_que_hay(self):
        def sin_ckpt(self, **kw):
            os.makedirs(kw["output_dir"], exist_ok=True)
            open(os.path.join(kw["output_dir"], "otro.txt"), "w").close()
        with mock.patch.object(FakeModel, "train", sin_ckpt):
            with self.assertRaisesRegex(FileNotFoundError, "otro.txt"):
                self.correr(hasta=9)

    def test_un_onnx_que_no_coincide_se_advierte(self):
        ej = self.correr(desvio=True)
        self.assertIn("ADVERTENCIA: el ONNX no coincide", ej.texto)
        self.assertNotIn("El ONNX coincide con el modelo original.", ej.texto)
        with open(os.path.join(ej.ns["ENTREGA"], "run_info.json"), encoding="utf-8") as fh:
            self.assertLess(json.load(fh)["verificacion_onnx"]["iou_mediana_onnx_vs_torch"], 0.9)  # y queda registrado

    def test_si_solo_uno_de_los_dos_detecta_se_advierte(self):
        ej = Ejecucion(self.nb)
        with self.entorno():
            ej.todo(tras={0: self.ajustar(), indice(self.nb, "Paso 10 · Exportar"): lambda ns: setattr(FakeModel, "sin_detecciones", True)}, hasta=indice(self.nb, "Paso 11 · ¿El ONNX"))
        self.assertIn("ADVERTENCIA", ej.texto)
        self.assertGreater(ej.ns["VERIF_ONNX"]["uno_detecta_y_otro_no"], 0)

    def test_sin_detecciones_en_validacion_no_rompe_el_resumen(self):
        FakeModel.sin_detecciones = True
        ej = self.correr(hasta=indice(self.nb, "Paso 9 · Cajas del detector"))
        self.assertIn("Se detectó un ratón en 0 de 6", ej.texto)
        self.assertEqual(ej.ns["RESUMEN_TORCH"], {"imagenes": 6, "con_deteccion": 0})

    # ------------------------------------------------------------------ Drive
    def test_con_drive_el_resultado_va_a_drive_y_no_se_descarga(self):
        ej = self.correr(GUARDAR_EN_DRIVE=True)
        entrega = ej.ns["ENTREGA"]
        self.assertTrue(entrega.startswith(os.path.join(self.trabajo, "drive", "TrackerLab_Entrenamiento")))
        self.assertEqual(self.colab.montajes, ["/content/drive"])
        self.assertEqual(self.colab.descargas, [])
        self.assertTrue(ej.ns["SALIDA"].startswith(os.path.join(self.trabajo, "local")))  # entrena en el disco local: los checkpoints no llenan Drive
        self.assertFalse(os.path.exists(os.path.join(entrega, "last.ckpt")))
        # en Drive solo quedan los entregables, no la carpeta de entrenamiento
        self.assertEqual(sorted(os.listdir(entrega)), sorted(["checkpoint_best_total.pth", "inference_model.onnx", "training_config.json", "conjunto_info.json", "run_info.json", ej.ns["NOMBRE_CORRIDA"] + ".zip"]))

    def test_sin_drive_se_descarga_el_zip(self):
        ej = self.correr(GUARDAR_EN_DRIVE=False)
        self.assertEqual(self.colab.montajes, [])
        self.assertEqual(self.colab.descargas, [os.path.join(ej.ns["ENTREGA"], ej.ns["NOMBRE_CORRIDA"] + ".zip")])
        self.assertTrue(ej.ns["ENTREGA"].startswith(os.path.join(self.trabajo, "local", "entregas")))

    def test_el_nombre_de_la_corrida_lleva_el_modelo_y_la_hora(self):
        import re
        for modelo in ("small", "nano"):
            ej = self.correr(hasta=3, MODELO=modelo)
            self.assertRegex(ej.ns["NOMBRE_CORRIDA"], rf"^rfdetr_{modelo}_\d{{8}}_\d{{6}}$")  # cada corrida tiene su carpeta: no se pisan


if __name__ == "__main__":
    unittest.main()