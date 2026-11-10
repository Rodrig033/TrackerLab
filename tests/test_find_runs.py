"""Pruebas de tools/find_runs.py (localiza las corridas de tus videos y arma el comando de build_training_set.py).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_find_runs.py -v

Se usan corridas REALES (save_run_inputs) y conjuntos REALES de sample_frames sobre videos sintéticos.
"""

import importlib
import json
import os
import shlex
import shutil
import unittest
from unittest import mock

from tools import build_training_set as B
from tools import find_runs as F


def _imp(nombre):
    try:
        return importlib.import_module(f"tests.{nombre}")
    except ImportError:
        return importlib.import_module(nombre)


TM = _imp("test_build_manual")


def leer_json(ruta):
    with open(ruta, encoding="utf-8") as fh:
        return json.load(fh)


def escribir_json(ruta, datos):
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(datos, fh)


class Base(TM.Base):
    def setUp(self):
        super().setUp()
        self.root = os.path.join(self.w, "resultados")
        os.makedirs(self.root)

    def montar(self, clave, exp, sello="20261103_100000", roi=None, huella="igual"):
        """Copia la corrida real de ``clave`` (A o B) a ``root/exp/entradas_<sello>``, opcionalmente con otro ROI o huella."""
        destino = os.path.join(self.root, exp, f"entradas_{sello}")
        shutil.copytree(self.entradas[clave], destino)
        d = leer_json(os.path.join(destino, "inputs.json"))
        if roi is not None:
            d["roi_box"] = list(roi)
        if huella != "igual":
            if huella is None:
                d["video"].pop("sha256_extremos", None)
            else:
                d["video"]["sha256_extremos"] = huella
        escribir_json(os.path.join(destino, "inputs.json"), d)
        return destino

    def correr(self, *args, root=None):
        msgs = []
        rc = F.main([root or self.root, *args], out=msgs.append)
        return rc, "\n".join(msgs)

    def comando(self, texto):
        """El comando impreso, separado en palabras tal como lo vería la terminal."""
        i = texto.index("python tools/build_training_set.py")
        return shlex.split(texto[i:].split("\n\n")[0].replace("\\\n", " "))


# ======================================================================
class LecturaTests(Base):
    def test_lee_los_campos_de_una_corrida_real(self):
        destino = self.montar("A", "expA", "20261103_174940")
        r = F.read_run(destino)
        d = leer_json(os.path.join(destino, "inputs.json"))
        self.assertEqual((r.experimento, r.sello, r.video, r.error), ("expA", "20261103_174940", "bg.avi", ""))
        self.assertEqual(r.huella, d["video"]["sha256_extremos"])
        self.assertEqual(r.roi, tuple(d["roi_box"]))
        self.assertEqual(r.rango, (d["frame_range"]["start"], d["frame_range"]["end"]))
        self.assertEqual(r.paso, d["timing"]["frame_step"])
        self.assertEqual(r.modo, str(d["tracker"]["mode"]))
        self.assertEqual(r.fecha, "2026-11-03 17:49")

    def test_fecha_de_un_nombre_raro_se_deja_tal_cual(self):
        self.assertEqual(F.Run("x", "e", "manual").fecha, "manual")
        self.assertEqual(F.Run("x", "e", "20261103-174940").fecha, "20261103-174940")

    def test_un_archivo_ilegible_o_sin_roi_se_anota_sin_romper(self):
        for nombre, contenido, esperado in (("roto", "{ no es json", "no se pudo leer"), ("sin_roi", json.dumps({"video": {"path": "/x/v.mp4"}}), "no trae roi_box")):
            d = os.path.join(self.root, nombre, "entradas_20261103_100000")
            os.makedirs(d)
            with open(os.path.join(d, "inputs.json"), "w") as fh:
                fh.write(contenido)
            self.assertIn(esperado, F.read_run(d).error, msg=nombre)
        faltante = os.path.join(self.root, "vacia", "entradas_20261103_100000")
        os.makedirs(faltante)
        self.assertIn("no se pudo leer", F.read_run(faltante).error)  # ni siquiera hay inputs.json
        d = os.path.join(self.root, "raro", "entradas_20261103_100000")
        os.makedirs(d)
        escribir_json(os.path.join(d, "inputs.json"), [1, 2, 3])  # un JSON válido pero que no es un diccionario
        self.assertTrue(F.read_run(d).error)

    def test_el_listado_va_de_la_mas_reciente_a_la_mas_antigua(self):
        self.montar("A", "expA", "20261101_090000")
        self.montar("B", "expB", "20261105_120000")
        self.montar("A", "expA2", "20261103_100000")
        runs = F.scan_runs(self.root)
        self.assertEqual([r.sello for r in runs], ["20261105_120000", "20261103_100000", "20261101_090000"])
        self.assertEqual([r.video for r in runs], ["col.avi", "bg.avi", "bg.avi"])

    def test_solo_busca_donde_corresponde(self):
        self.montar("A", "expA")
        os.makedirs(os.path.join(self.root, "expA", "dataset_20261103_100000"))  # otras carpetas no cuentan
        os.makedirs(os.path.join(self.root, "expA", "sub", "entradas_20261103_110000"))  # demasiado profunda
        with open(os.path.join(self.root, "expA", "entradas_suelto.txt"), "w") as fh:  # un archivo no es una carpeta
            fh.write("x")
        self.assertEqual(len(F.scan_runs(self.root)), 1)
        # también acepta una corrida directamente en la carpeta de resultados
        shutil.copytree(self.entradas["B"], os.path.join(self.root, "entradas_20261104_080000"))
        self.assertEqual(len(F.scan_runs(self.root)), 2)

    def test_la_carpeta_con_corchetes_no_se_interpreta_como_patron(self):
        raiz = os.path.join(self.w, "resultados [v1]")
        os.makedirs(os.path.join(raiz, "exp"))
        shutil.copytree(self.entradas["A"], os.path.join(raiz, "exp", "entradas_20261103_100000"))
        self.assertEqual(len(F.scan_runs(raiz)), 1)

    def test_listado_en_pantalla(self):
        self.montar("A", "expA", "20261103_174940")
        rc, texto = self.correr()
        self.assertEqual(rc, 0)
        self.assertIn("Corridas encontradas en", texto)
        self.assertIn("bg.avi", texto)
        self.assertIn("expA", texto)
        self.assertIn("2026-11-03 17:49", texto)
        self.assertIn("[20 20 260 190]", texto)  # el ROI
        self.assertNotIn("Comando", texto)  # sin conjuntos, solo lista

    def test_sin_corridas_se_explica(self):
        rc, texto = self.correr()
        self.assertEqual(rc, 2)
        self.assertIn("SAVE_RUN_INPUTS", texto)
        rc, texto = self.correr(root=os.path.join(self.w, "no_existe"))
        self.assertEqual(rc, 2)


# ======================================================================
class EmparejamientoTests(Base):
    def video(self, huella="x" * 64, nombre="bg.avi"):
        v = {"id": "bg", "nombre": nombre}
        if huella:
            v["sha256_extremos"] = huella
        return v

    def runs(self):
        return F.scan_runs(self.root)

    def test_por_huella(self):
        a = self.montar("A", "expA")
        self.montar("B", "expB")
        v = {"id": "bg", "nombre": "cualquier_nombre.avi", "sha256_extremos": F.read_run(a).huella}  # el nombre no cuenta si hay huella
        self.assertEqual([r.carpeta for r in F.match_runs(v, self.runs())], [a])

    def test_por_nombre_si_no_hay_huella_en_ninguno_de_los_dos(self):
        a = self.montar("A", "expA", huella=None)
        self.montar("B", "expB", huella=None)
        self.assertEqual([r.carpeta for r in F.match_runs(self.video(huella=None), self.runs())], [a])

    def test_el_mismo_nombre_con_otra_huella_no_es_el_mismo_video(self):
        self.montar("A", "expA", huella="f" * 64)
        self.assertEqual(F.match_runs(self.video(huella="a" * 64), self.runs()), [])  # la huella manda sobre el nombre

    def test_sin_huella_en_el_manifiesto_pero_con_huella_en_la_corrida(self):
        a = self.montar("A", "expA")
        self.assertEqual([r.carpeta for r in F.match_runs(self.video(huella=None), self.runs())], [a])  # se cae a nombre

    def test_con_huella_en_el_manifiesto_y_sin_huella_en_la_corrida(self):
        a = self.montar("A", "expA", huella=None)
        self.assertEqual([r.carpeta for r in F.match_runs(self.video(huella="a" * 64), self.runs())], [a])

    def test_corridas_invalidas_no_empatan(self):
        self.montar("A", "expA")
        malo = os.path.join(self.root, "roto", "entradas_20261104_100000")
        os.makedirs(malo)
        with open(os.path.join(malo, "inputs.json"), "w") as fh:
            fh.write("{")
        runs = self.runs()
        self.assertEqual(len(runs), 2)
        self.assertEqual(len(F.match_runs(self.video(huella=None), runs)), 1)

    def test_una_corrida_reciente_sin_roi_no_se_elige(self):
        """La más reciente de ESE video, pero sin roi_box: elegirla haría fallar el armado. Debe quedarse la anterior, que sí lo tiene."""
        buena = self.montar("A", "expA", "20261101_090000")
        reciente = self.montar("A", "expA2", "20261109_090000")
        d = leer_json(os.path.join(reciente, "inputs.json"))
        del d["roi_box"]  # conserva el video y la huella: solo le falta el ROI
        escribir_json(os.path.join(reciente, "inputs.json"), d)
        runs = self.runs()
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0].carpeta, reciente)  # es la más reciente
        self.assertEqual(runs[0].huella, runs[1].huella)  # y es del mismo video
        self.assertEqual([r.carpeta for r in F.match_runs({"id": "bg", "nombre": "bg.avi", "sha256_extremos": runs[0].huella}, runs)], [buena])

    def test_varias_corridas_de_la_mas_reciente_a_la_mas_antigua(self):
        vieja = self.montar("A", "expA", "20261101_090000")
        nueva = self.montar("A", "expA2", "20261107_090000")
        media = self.montar("A", "expA3", "20261103_090000")
        hits = F.match_runs({"id": "bg", "nombre": "bg.avi", "sha256_extremos": F.read_run(vieja).huella}, self.runs())
        self.assertEqual([r.carpeta for r in hits], [nueva, media, vieja])

    def test_es_la_misma_regla_que_la_del_armado(self):
        """Para los mismos casos, find_runs y build_training_set.match_roi dan los mismos ROI."""
        casos = [("con huella en los dos", "igual", "igual"), ("sin huella en los dos", None, None), ("huellas distintas", "f" * 64, "a" * 64),
                 ("solo la corrida la tiene", "igual", None), ("solo el manifiesto la tiene", None, "a" * 64)]
        for nombre, h_corrida, h_manifiesto in casos:
            shutil.rmtree(self.root)
            os.makedirs(self.root)
            self.montar("A", "expA", "20261101_090000", roi=(20, 20, 260, 190), huella=h_corrida)
            self.montar("A", "expA2", "20261103_090000", roi=(30, 30, 200, 150), huella=h_corrida)
            runs = F.scan_runs(self.root)
            huella_real = leer_json(os.path.join(self.entradas["A"], "inputs.json"))["video"]["sha256_extremos"]
            video = {"id": "bg", "nombre": "bg.avi"}
            if h_manifiesto == "igual":
                video["sha256_extremos"] = huella_real
            elif h_manifiesto:
                video["sha256_extremos"] = h_manifiesto
            corridas = [(r.carpeta,) + B.read_run_roi(r.carpeta) for r in runs]
            del_armado = {roi for roi, _ in B.match_roi(video, corridas)}
            de_esta = {r.roi for r in F.match_runs(video, runs)}
            self.assertEqual(de_esta, del_armado, msg=nombre)


# ======================================================================
class ComandoTests(Base):
    def conjuntos(self):
        tr, _ = self.conjunto(("A",), por_video=6)
        va, _ = self.conjunto(("B",), por_video=3)
        return tr, va

    def test_el_comando_impreso_sirve_de_verdad(self):
        """Se ejecuta el comando tal como se imprime: debe armar el conjunto."""
        tr, va = self.conjuntos()
        self.montar("A", "expA")
        self.montar("B", "expB")
        rc, texto = self.correr("--train-manual", tr, "--valid-manual", va)
        self.assertEqual(rc, 0, msg=texto)
        args = self.comando(texto)
        i = texto.index("python tools/build_training_set.py")
        lineas = texto[i:].split("\n\n")[0].split("\n")
        self.assertGreaterEqual(len(lineas), 5)  # una opción por línea: se lee y se pega mejor
        self.assertTrue(all(ln.endswith(" \\") for ln in lineas[:-1]))  # con la barra de continuación al final
        self.assertTrue(lineas[0].startswith("python tools/build_training_set.py") and all(ln.startswith("  --") for ln in lineas[1:]))
        self.assertEqual(args[:2], ["python", "tools/build_training_set.py"])
        self.assertEqual(B.main(args[2:]), 0)
        salida = args[args.index("--salida") + 1]
        self.assertEqual(salida, os.path.join(self.root, "entrenamiento"))  # por omisión, dentro de la carpeta de resultados
        with open(os.path.join(salida, "conjunto_info.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["imagenes"], {"train": 6, "valid": 3})

    def test_lleva_cada_opcion_en_su_lugar(self):
        tr, va = self.conjuntos()
        a, b = self.montar("A", "expA"), self.montar("B", "expB")
        rc, texto = self.correr("--train-manual", tr, "--valid-manual", va, "--excluir-videos-de", os.path.join(self.w, "otro_reservado"), "--salida", os.path.join(self.w, "mi_salida"))
        self.assertEqual(rc, 0)
        args = self.comando(texto)
        self.assertEqual([args[i + 1] for i, x in enumerate(args) if x == "--train-manual"], [tr])
        self.assertEqual([args[i + 1] for i, x in enumerate(args) if x == "--valid-manual"], [va])
        self.assertEqual(sorted(args[i + 1] for i, x in enumerate(args) if x == "--entradas"), sorted([a, b]))
        self.assertEqual([args[i + 1] for i, x in enumerate(args) if x == "--excluir-videos-de"], [os.path.join(self.w, "otro_reservado")])
        self.assertEqual(args[args.index("--salida") + 1], os.path.join(self.w, "mi_salida"))
        self.assertNotIn("AVISO: no indicaste --excluir-videos-de", texto)

    def test_sin_reservado_avisa(self):
        tr, va = self.conjuntos()
        self.montar("A", "expA")
        self.montar("B", "expB")
        _, texto = self.correr("--train-manual", tr, "--valid-manual", va)
        self.assertIn("AVISO: no indicaste --excluir-videos-de", texto)

    def test_elige_la_corrida_mas_reciente_y_avisa_si_el_roi_difiere(self):
        tr, va = self.conjuntos()
        vieja = self.montar("A", "expA", "20261101_090000", roi=(20, 20, 260, 190))
        nueva = self.montar("A", "expA2", "20261107_090000", roi=(30, 30, 200, 150))
        self.montar("B", "expB")
        rc, texto = self.correr("--train-manual", tr, "--valid-manual", va)
        self.assertEqual(rc, 0)
        args = self.comando(texto)
        entradas = [args[i + 1] for i, x in enumerate(args) if x == "--entradas"]
        self.assertIn(nueva, entradas)
        self.assertNotIn(vieja, entradas)
        self.assertIn("[la más reciente de 2]", texto)
        self.assertIn("AVISO: este video tiene 2 corridas con ROI distintos", texto)
        self.assertIn("20261107_090000 [30 30 200 150]", texto)
        self.assertIn("20261101_090000 [20 20 260 190]", texto)

    def test_con_el_mismo_roi_no_avisa(self):
        tr, va = self.conjuntos()
        self.montar("A", "expA", "20261101_090000")
        self.montar("A", "expA2", "20261107_090000")
        self.montar("B", "expB")
        _, texto = self.correr("--train-manual", tr, "--valid-manual", va)
        self.assertIn("[la más reciente de 2]", texto)
        self.assertNotIn("ROI distintos", texto)

    def test_falta_la_corrida_de_un_video(self):
        tr, va = self.conjuntos()
        self.montar("A", "expA")  # falta la de B
        rc, texto = self.correr("--train-manual", tr, "--valid-manual", va)
        self.assertEqual(rc, 2)
        self.assertIn("✗ col.avi: no hay ninguna corrida de este video", texto)
        self.assertIn("✓ bg.avi", texto)
        self.assertIn("falta la corrida de col.avi", texto)
        self.assertNotIn("Comando para armar", texto)  # no se imprime un comando que no serviría

    def test_el_mismo_video_en_dos_conjuntos(self):
        tr, _ = self.conjuntos()
        otro, _ = self.conjunto(("A",), por_video=3)
        self.montar("A", "expA")
        rc, texto = self.correr("--train-manual", tr, "--valid-manual", otro)
        self.assertEqual(rc, 2)
        self.assertIn("! «bg.avi» ya figura en el conjunto de entrenamiento", texto)
        self.assertIn("fuga", texto)
        self.assertNotIn("Comando para armar", texto)

    def test_un_conjunto_con_dos_videos_pide_las_dos_corridas(self):
        dos, _ = self.conjunto(("A", "B"), por_video=4)  # un solo conjunto con los dos videos
        self.montar("A", "expA")
        self.montar("B", "expB")
        rc, texto = self.correr("--train-manual", dos)
        self.assertEqual(rc, 0)
        self.assertEqual(texto.count("--entradas"), 2)

    def test_una_carpeta_que_no_es_un_conjunto(self):
        self.montar("A", "expA")
        rc, texto = self.correr("--train-manual", self.w)
        self.assertEqual(rc, 2)
        self.assertIn("falta manifest.json", texto)

    def test_las_rutas_con_espacios_y_comillas_se_escriben_bien(self):
        tr, va = self.conjuntos()
        raro_tr = os.path.join(self.w, "mi carpeta 'rara'")
        shutil.copytree(tr, raro_tr)
        self.montar("A", "expA")
        self.montar("B", "expB")
        rc, texto = self.correr("--train-manual", raro_tr, "--valid-manual", va)
        self.assertEqual(rc, 0)
        args = self.comando(texto)
        self.assertEqual(args[args.index("--train-manual") + 1], raro_tr)  # tal cual, aunque lleve espacios y comillas
        self.assertNotIn("“", texto)
        self.assertNotIn("”", texto)
        self.assertEqual(B.main(args[2:]), 0)

    def test_se_arma_con_la_ruta_de_resultados_por_omision(self):
        tr, va = self.conjuntos()
        casa = os.path.join(self.w, "casa")
        raiz = os.path.join(casa, "TrackerLab_Resultados")
        shutil.copytree(self.root, raiz)
        os.makedirs(os.path.join(raiz, "expA"))
        shutil.copytree(self.entradas["A"], os.path.join(raiz, "expA", "entradas_20261103_100000"))
        shutil.copytree(self.entradas["B"], os.path.join(raiz, "expB", "entradas_20261103_100000"))
        msgs = []
        with mock.patch.dict(os.environ, {"HOME": casa}):
            rc = F.main(["--train-manual", tr, "--valid-manual", va], out=msgs.append)
        self.assertEqual(rc, 0, msg="\n".join(msgs))
        self.assertIn(raiz, "\n".join(msgs))


if __name__ == "__main__":
    unittest.main()