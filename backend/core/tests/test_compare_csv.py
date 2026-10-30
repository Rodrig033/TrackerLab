"""Pruebas de tools/compare_csv.py.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_compare_csv.py -v
"""

import contextlib
import csv
import io
import os
import tempfile
import unittest

from tools.compare_csv import compare_frame_csv, compare_summary_csv, format_report, main

HEAD = ["frame", "time_s", "cx_abs", "cy_abs_inv", "vel_px_s"]
ROWS = [
    ["0", "0.000000", "100.000", "380.000", ""],
    ["1", "0.040000", "101.500", "379.000", "37.5000"],
    ["2", "0.080000", "", "", ""],  # sujeto perdido
    ["3", "0.120000", "104.000", "377.250", "12.0000"],
]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.n = 0

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, head, rows, bom=False):
        self.n += 1
        path = os.path.join(self.tmp.name, f"f{self.n}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig" if bom else "utf-8") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(rows)
        return path

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()


class FrameCsvTests(Base):
    def test_identicos(self):
        a, b = self.write(HEAD, ROWS), self.write(HEAD, ROWS)
        c = compare_frame_csv(a, b)
        self.assertTrue(c.identical and c.ok)
        self.assertIsNone(c.first_divergence)
        self.assertIn("IDÉNTICOS", format_report(c))

    def test_el_bom_no_cuenta(self):
        c = compare_frame_csv(self.write(HEAD, ROWS, bom=False), self.write(HEAD, ROWS, bom=True))
        self.assertTrue(c.identical)

    def test_diferencia_numerica_con_y_sin_tolerancia(self):
        rows2 = [r[:] for r in ROWS]
        rows2[1][4] = "37.5004"
        a, b = self.write(HEAD, ROWS), self.write(HEAD, rows2)
        estricta = compare_frame_csv(a, b, tol=0.0)
        self.assertFalse(estricta.ok)
        self.assertEqual(estricta.first_divergence, (1, "vel_px_s", "37.5000", "37.5004"))
        self.assertAlmostEqual(estricta.columns["vel_px_s"].max_abs, 4e-4, places=9)
        laxa = compare_frame_csv(a, b, tol=1e-3)
        self.assertTrue(laxa.ok and not laxa.identical)
        self.assertEqual(laxa.total_within_tol, 1)
        self.assertIn("EQUIVALENTES", format_report(laxa))

    def test_primera_divergencia_es_la_mas_temprana(self):
        rows2 = [r[:] for r in ROWS]
        rows2[3][2] = "999.000"
        rows2[1][3] = "1.000"
        c = compare_frame_csv(self.write(HEAD, ROWS), self.write(HEAD, rows2))
        self.assertEqual(c.first_divergence[:2], (1, "cy_abs_inv"))

    def test_deteccion_distinta_se_senala(self):
        rows2 = [r[:] for r in ROWS]
        rows2[2][2], rows2[2][3] = "50.000", "300.000"  # el nuevo SÍ encuentra al sujeto en el frame 2
        rows2[0][2], rows2[0][3] = "", ""  # y NO lo encuentra en el frame 0
        c = compare_frame_csv(self.write(HEAD, ROWS), self.write(HEAD, rows2))
        self.assertEqual(c.presence_mismatch_frames, [0, 2])
        self.assertFalse(c.ok)
        self.assertIn("Detección distinta", format_report(c))

    def test_frames_faltantes_y_sobrantes(self):
        c = compare_frame_csv(self.write(HEAD, ROWS), self.write(HEAD, ROWS[1:] + [["4", "0.16", "1", "1", ""]]))
        self.assertEqual((c.frames_only_a, c.frames_only_b), ([0], [4]))
        self.assertFalse(c.ok)

    def test_cabeceras_distintas(self):
        head2 = HEAD[:-1] + ["vel_cm_s", "otra"]
        rows2 = [r + ["x"] for r in ROWS]
        c = compare_frame_csv(self.write(HEAD, ROWS), self.write(head2, rows2))
        self.assertFalse(c.same_headers)
        self.assertEqual((c.headers_only_a, c.headers_only_b), (["vel_px_s"], ["vel_cm_s", "otra"]))
        self.assertIn("CABECERAS NO COINCIDEN", format_report(c))

    def test_mismas_columnas_en_otro_orden(self):
        head2 = [HEAD[1], HEAD[0]] + HEAD[2:]
        rows2 = [[r[1], r[0]] + r[2:] for r in ROWS]
        c = compare_frame_csv(self.write(HEAD, ROWS), self.write(head2, rows2))
        self.assertFalse(c.same_headers)
        self.assertEqual(c.total_different, 0)  # el contenido coincide
        self.assertIn("otro orden", format_report(c))

    def test_texto_distinto_no_numerico(self):
        h = ["frame", "nota"]
        c = compare_frame_csv(self.write(h, [["0", "a"]]), self.write(h, [["0", "b"]]), tol=1.0)
        self.assertFalse(c.ok)

    def test_errores_de_lectura(self):
        sin_frame = self.write(["a", "b"], [["1", "2"]])
        with self.assertRaises(ValueError):
            compare_frame_csv(sin_frame, sin_frame)
        vacio = os.path.join(self.tmp.name, "vacio.csv")
        open(vacio, "w").close()
        with self.assertRaises(ValueError):
            compare_frame_csv(vacio, vacio)


class SummaryTests(Base):
    S = [["experimento", "e1"], ["video_path", "/a/v.mp4"], ["dist_total_px", "100.0000"], ["vel_media_px_s", "3.5000"]]

    def test_identicos(self):
        c = compare_summary_csv(self.write(["metrica", "valor"], self.S), self.write(["metrica", "valor"], self.S))
        self.assertTrue(c.identical)

    def test_las_rutas_y_nombres_son_solo_informativos(self):
        s2 = [["experimento", "e2"], ["video_path", "C:\\v.mp4"]] + self.S[2:]
        c = compare_summary_csv(self.write(["metrica", "valor"], self.S), self.write(["metrica", "valor"], s2))
        self.assertTrue(c.ok and not c.identical)
        self.assertEqual(len(c.informational), 2)

    def test_diferencia_real_y_metricas_faltantes(self):
        s2 = [r[:] for r in self.S[:2]] + [["dist_total_px", "100.0100"], ["extra", "1"]]
        c = compare_summary_csv(self.write(["metrica", "valor"], self.S), self.write(["metrica", "valor"], s2))
        self.assertFalse(c.ok)
        self.assertEqual(c.different, [("dist_total_px", "100.0000", "100.0100")])
        self.assertEqual((c.only_a, c.only_b), (["vel_media_px_s"], ["extra"]))
        self.assertTrue(compare_summary_csv(self.write(["metrica", "valor"], self.S),
                                            self.write(["metrica", "valor"], s2), tol=0.1).different == [])

    def test_vacio_contra_valor_es_diferencia(self):
        s2 = [r[:] for r in self.S]
        s2[3][1] = ""
        c = compare_summary_csv(self.write(["metrica", "valor"], self.S), self.write(["metrica", "valor"], s2), tol=100)
        self.assertEqual(len(c.different), 1)


class LineaDeComandosTests(Base):
    def test_codigos_de_salida(self):
        a = self.write(HEAD, ROWS)
        self.assertEqual(self.run_main(a, self.write(HEAD, ROWS))[0], 0)
        rows2 = [r[:] for r in ROWS]
        rows2[1][4] = "99.0000"
        code, out, _ = self.run_main(a, self.write(HEAD, rows2))
        self.assertEqual(code, 1)
        self.assertIn("PRIMERA DIVERGENCIA: frame 1", out)
        self.assertEqual(self.run_main(a, os.path.join(self.tmp.name, "no_existe.csv"))[0], 2)
        self.assertEqual(self.run_main(a, self.write(["a"], [["1"]]))[0], 2)

    def test_tolerancia_por_linea_de_comandos(self):
        rows2 = [r[:] for r in ROWS]
        rows2[1][4] = "37.5004"
        a, b = self.write(HEAD, ROWS), self.write(HEAD, rows2)
        self.assertEqual(self.run_main(a, b)[0], 1)
        self.assertEqual(self.run_main(a, b, "--tol", "1e-3")[0], 0)

    def test_detecta_resumen_automaticamente(self):
        a = self.write(["metrica", "valor"], SummaryTests.S)
        code, out, _ = self.run_main(a, self.write(["metrica", "valor"], SummaryTests.S))
        self.assertEqual(code, 0)
        self.assertIn("Resumen", out)


if __name__ == "__main__":
    unittest.main()