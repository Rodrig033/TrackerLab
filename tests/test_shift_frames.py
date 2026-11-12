"""Pruebas de tools/shift_frames.py (renumerar un conjunto corrido un desfase constante).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_shift_frames.py -v

La prueba central es de punta a punta: un conjunto cuyas imágenes están REALMENTE corridas respecto al video se mide con
check_frame_alignment.py, se renumera con el desfase medido y se vuelve a medir: debe quedar alineado.
"""

import importlib
import json
import os
import tempfile
import unittest
from datetime import datetime
from unittest import mock

from tools import check_frame_alignment as C
from tools import shift_frames as S


def _imp(nombre):
    try:
        return importlib.import_module(f"tests.{nombre}")
    except ImportError:
        return importlib.import_module(nombre)


CFA = _imp("test_check_frame_alignment")
AHORA = lambda: datetime(2026, 11, 12, 10, 0, 0)  # noqa: E731


def leer(ruta):
    with open(ruta, encoding="utf-8") as fh:
        return fh.read()


def escribir(ruta, texto):
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write(texto)


def leer_json(ruta):
    with open(ruta, encoding="utf-8") as fh:
        return json.load(fh)


class Base(CFA.Base):
    FRAMES = [20, 40, 41, 42, 70, 100]

    def conjunto(self, desfase=0, frames=None, total=120, rango=(10, 110)):
        """Conjunto cuya imagen «N» es el contenido del frame N + desfase, con una etiqueta distinta en cada una y dudosas."""
        frames = frames or self.FRAMES
        val = CFA.crear_conjunto(self.work.name, self.bits, self.bits_frames, {f: desfase for f in frames}, CFA.W, CFA.H)
        man = leer_json(os.path.join(val, "manifest.json"))
        man["videos"][0].update(total_frames=total, rango=list(rango), paso=1)
        escribir(os.path.join(val, "manifest.json"), json.dumps(man))
        for f in frames[:-1]:
            escribir(os.path.join(val, "labels", f"v_f{f:06d}.txt"), f"0 {f / 1000:.6f} 0.5 0.1 0.1\n")  # la etiqueta «sabe» de qué imagen es
        escribir(os.path.join(val, "dudosas.txt"), f"v_f{frames[-1]:06d}\n")
        return val

    def correr(self, val, **kw):
        msgs = []
        kw.setdefault("now", AHORA)
        rc = S.run(val, out=msgs.append, **kw)
        return rc, "\n".join(msgs)

    def foto(self, carpeta):
        """Todo el contenido de una carpeta: {ruta relativa: bytes}."""
        out = {}
        for raiz, _, archivos in os.walk(carpeta):
            for a in archivos:
                p = os.path.join(raiz, a)
                with open(p, "rb") as fh:
                    out[os.path.relpath(p, carpeta)] = fh.read()
        return out


# ======================================================================
class DePuntaAPuntaTests(Base):
    def test_medir_renumerar_y_volver_a_medir_queda_alineado(self):
        for d in (-1, 2, -7):
            with self.subTest(desfase=d):
                self.work.cleanup()
                self.work = tempfile.TemporaryDirectory()
                val = self.conjunto(desfase=d, frames=[20, 40, 55, 70, 85])
                C.run(val, out=lambda m: None)
                self.assertEqual(leer_json(os.path.join(val, "alineacion.json"))["resumen"]["desfases"], {str(d): 5})
                rc, texto = self.correr(val, desde_alineacion=os.path.join(val, "alineacion.json"), aplicar=True)
                self.assertEqual(rc, 0, msg=texto)
                C.run(val, out=lambda m: None)
                r = leer_json(os.path.join(val, "alineacion.json"))["resumen"]
                self.assertEqual(r["veredicto"], "alineado")
                self.assertEqual(r["desfases"], {"0": 5})


class RenumerarTests(Base):
    def test_sin_aplicar_no_cambia_nada(self):
        val = self.conjunto()
        antes = self.foto(val)
        rc, texto = self.correr(val, desfase=-1)
        self.assertEqual(rc, 0)
        self.assertEqual(self.foto(val), antes)
        self.assertIn("No se cambió nada", texto)
        self.assertIn("v_f000020  ->  v_f000019", texto)
        self.assertIn("cada frame N pasa a N - 1", texto)
        self.assertFalse(any("antes_de_renumerar" in n for n in os.listdir(self.work.name)))

    def test_aplicar_renombra_y_las_etiquetas_siguen_a_su_imagen(self):
        val = self.conjunto()
        contenido = {}
        for f in self.FRAMES:
            with open(os.path.join(val, "images", f"v_f{f:06d}.png"), "rb") as fh:
                contenido[f] = fh.read()
        rc, texto = self.correr(val, desfase=-1, aplicar=True)
        self.assertEqual(rc, 0, msg=texto)
        for f in self.FRAMES:
            with open(os.path.join(val, "images", f"v_f{f - 1:06d}.png"), "rb") as fh:
                self.assertEqual(fh.read(), contenido[f], msg=f"imagen {f}")
        for f in self.FRAMES[:-1]:
            self.assertEqual(leer(os.path.join(val, "labels", f"v_f{f - 1:06d}.txt")), f"0 {f / 1000:.6f} 0.5 0.1 0.1\n")
        self.assertEqual(sorted(os.listdir(os.path.join(val, "images"))), [f"v_f{f - 1:06d}.png" for f in self.FRAMES])
        self.assertEqual(sorted(os.listdir(os.path.join(val, "labels"))), [f"v_f{f - 1:06d}.txt" for f in self.FRAMES[:-1]])
        self.assertEqual(leer(os.path.join(val, "dudosas.txt")), "v_f000099\n")

    def test_frames_vecinos_no_se_pisan(self):
        """40, 41 y 42 con desfase -1 y +1: renombrar uno a uno sobrescribiría al vecino."""
        for d in (-1, 1):
            with self.subTest(desfase=d):
                self.work.cleanup()
                self.work = tempfile.TemporaryDirectory()
                val = self.conjunto()
                self.correr(val, desfase=d, aplicar=True)
                for f in (40, 41, 42):
                    self.assertEqual(leer(os.path.join(val, "labels", f"v_f{f + d:06d}.txt")), f"0 {f / 1000:.6f} 0.5 0.1 0.1\n")
                self.assertFalse(any(n.startswith(".renumerando") for n in os.listdir(os.path.join(val, "labels"))))

    def test_el_manifiesto_queda_en_la_numeracion_nueva_y_con_constancia(self):
        val = self.conjunto()
        self.correr(val, desfase=-1, aplicar=True)
        v = leer_json(os.path.join(val, "manifest.json"))["videos"][0]
        self.assertEqual(v["frames"], [f - 1 for f in self.FRAMES])
        self.assertEqual(v["rango"], [9, 109])
        self.assertEqual(v["renumerado"], [{"desfase": -1, "fecha": "2026-11-12 10:00:00", "motivo": "desfase indicado a mano"}])
        self.assertEqual(v["total_frames"], 120)  # lo demás no cambia

    def test_el_rango_no_queda_negativo(self):
        val = self.conjunto(frames=[3, 20], rango=(1, 30))
        self.correr(val, desfase=-2, aplicar=True)
        self.assertEqual(leer_json(os.path.join(val, "manifest.json"))["videos"][0]["rango"], [0, 28])

    def test_copia_de_seguridad_identica_al_original(self):
        val = self.conjunto()
        escribir(os.path.join(val, "alineacion.json"), "{}")
        escribir(os.path.join(val, "alineacion_entradas_x.json"), "{}")
        antes = self.foto(val)
        rc, texto = self.correr(val, desfase=-1, aplicar=True)
        respaldo = os.path.join(self.work.name, "validacion_antes_de_renumerar_20261112_100000")
        self.assertEqual(self.foto(respaldo), antes)
        self.assertIn(respaldo, texto)
        self.assertFalse(os.path.exists(os.path.join(val, "alineacion.json")))  # medían la numeración vieja
        self.assertFalse(os.path.exists(os.path.join(val, "alineacion_entradas_x.json")))
        self.assertTrue(os.path.exists(os.path.join(respaldo, "alineacion.json")))

    def test_un_numero_fuera_del_video_no_cambia_nada(self):
        for d, frames in ((-25, None), (25, None)):  # 20 - 25 < 0 ; 100 + 25 >= 120
            with self.subTest(desfase=d):
                self.work.cleanup()
                self.work = tempfile.TemporaryDirectory()
                val = self.conjunto(frames=frames)
                antes = self.foto(val)
                rc, texto = self.correr(val, desfase=d, aplicar=True)
                self.assertEqual(rc, 2)
                self.assertIn("fuera del video", texto)
                self.assertEqual(self.foto(val), antes)

    def test_sin_total_de_frames_solo_se_exige_no_negativo(self):
        val = self.conjunto(total=0)
        self.assertEqual(self.correr(val, desfase=30, aplicar=True)[0], 0)

    def test_desfase_cero(self):
        val = self.conjunto()
        rc, texto = self.correr(val, desfase=0, aplicar=True)
        self.assertEqual(rc, 0)
        self.assertIn("no hay nada que renumerar", texto)

    def test_sin_dudosas(self):
        val = self.conjunto()
        os.remove(os.path.join(val, "dudosas.txt"))
        self.assertEqual(self.correr(val, desfase=-1, aplicar=True)[0], 0)
        self.assertFalse(os.path.exists(os.path.join(val, "dudosas.txt")))


class DesdeAlineacionTests(Base):
    def alineacion(self, val, filas, nombre="alineacion_entradas_x.json"):
        p = os.path.join(val, nombre)
        escribir(p, json.dumps({"imagenes": filas}))
        return p

    def test_desfase_constante(self):
        val = self.conjunto()
        filas = [{"video": "v", "frame": f, "desfase": -1, "clara": True} for f in self.FRAMES] + [{"video": "v", "frame": 99, "desfase": 3, "clara": False}]
        p = self.alineacion(val, filas)
        self.assertEqual(S.offsets_from_alignment(p), {"v": -1})  # las ambiguas no cuentan
        rc, _ = self.correr(val, desde_alineacion=p, aplicar=True)
        self.assertEqual(rc, 0)
        self.assertEqual(leer_json(os.path.join(val, "manifest.json"))["videos"][0]["renumerado"][0]["motivo"], "desfase medido en alineacion_entradas_x.json")

    def test_un_desfase_que_no_es_constante_no_se_aplica(self):
        val = self.conjunto()
        p = self.alineacion(val, [{"video": "v", "frame": 20, "desfase": -1, "clara": True}, {"video": "v", "frame": 40, "desfase": -2, "clara": True}])
        antes = self.foto(val)
        rc, texto = self.correr(val, desde_alineacion=p, aplicar=True)
        self.assertEqual(rc, 2)
        self.assertIn("no es constante", texto)
        self.assertEqual(self.foto(val), antes)

    def test_sin_desfases_claros(self):
        val = self.conjunto()
        p = self.alineacion(val, [{"video": "v", "frame": 20, "desfase": -1, "clara": False}])
        rc, texto = self.correr(val, desde_alineacion=p)
        self.assertEqual(rc, 2)
        self.assertIn("ninguna imagen con un desfase claro", texto)

    def test_un_video_que_no_esta_en_el_conjunto(self):
        val = self.conjunto()
        p = self.alineacion(val, [{"video": "otro", "frame": 20, "desfase": -1, "clara": True}])
        rc, texto = self.correr(val, desde_alineacion=p)
        self.assertEqual(rc, 2)
        self.assertIn("otro no está en este conjunto", texto)

    def test_archivo_ilegible(self):
        val = self.conjunto()
        rc, _ = self.correr(val, desde_alineacion=os.path.join(val, "no_existe.json"))
        self.assertEqual(rc, 2)


class ErroresTests(Base):
    def test_no_es_un_conjunto(self):
        rc, texto = self.correr(self.work.name, desfase=-1)
        self.assertEqual(rc, 2)
        self.assertIn("falta manifest.json", texto)

    def test_hay_que_elegir_una_sola_forma_de_dar_el_desfase(self):
        val = self.conjunto()
        self.assertEqual(self.correr(val)[0], 2)
        valida = os.path.join(val, "alineacion_valida.json")
        escribir(valida, json.dumps({"imagenes": [{"video": "v", "frame": 20, "desfase": -1, "clara": True}]}))
        rc, texto = self.correr(val, desfase=-1, desde_alineacion=valida, aplicar=True)  # las dos válidas por separado: igual se rechaza
        self.assertEqual(rc, 2)
        self.assertIn("uno de los dos", texto)
        self.assertTrue(os.path.exists(os.path.join(val, "images", "v_f000020.png")))

    def test_varios_videos_exigen_indicar_cual(self):
        val = self.conjunto()
        man = leer_json(os.path.join(val, "manifest.json"))
        otro = dict(man["videos"][0], id="w", frames=[])
        man["videos"].append(otro)
        escribir(os.path.join(val, "manifest.json"), json.dumps(man))
        rc, texto = self.correr(val, desfase=-1)
        self.assertEqual(rc, 2)
        self.assertIn("--video", texto)
        self.assertEqual(self.correr(val, desfase=-1, video="v", aplicar=True)[0], 0)
        self.assertEqual(self.correr(val, desfase=-1, video="z")[0], 2)

    def test_si_falla_la_copia_de_seguridad_no_se_toca_nada(self):
        val = self.conjunto()
        antes = self.foto(val)
        with mock.patch.object(S.shutil, "copytree", side_effect=OSError("disco lleno")):
            rc, texto = self.correr(val, desfase=-1, aplicar=True)
        self.assertEqual(rc, 2)
        self.assertIn("disco lleno", texto)
        self.assertEqual(self.foto(val), antes)

    def test_si_falla_a_medio_camino_se_indica_donde_esta_el_original(self):
        val = self.conjunto()
        antes = self.foto(val)
        real = os.rename
        llamadas = []

        def falla(a, b):
            llamadas.append(a)
            if len(llamadas) == 3:
                raise OSError("permiso denegado")
            return real(a, b)
        with mock.patch.object(S.os, "rename", side_effect=falla):
            rc, texto = self.correr(val, desfase=-1, aplicar=True)
        self.assertEqual(rc, 2)
        self.assertIn("El conjunto original está intacto en", texto)
        self.assertEqual(self.foto(os.path.join(self.work.name, "validacion_antes_de_renumerar_20261112_100000")), antes)

    def test_cli(self):
        val = self.conjunto()
        self.assertEqual(S.main([val, "--desfase", "-1"], now=AHORA, out=lambda m: None), 0)
        self.assertTrue(os.path.exists(os.path.join(val, "images", "v_f000020.png")))  # sin --aplicar
        self.assertEqual(S.main([val, "--desfase", "-1", "--aplicar"], now=AHORA, out=lambda m: None), 0)
        self.assertTrue(os.path.exists(os.path.join(val, "images", "v_f000019.png")))


if __name__ == "__main__":
    unittest.main()