"""Pruebas de stats/report_info.py (página del reporte con calibración y calidad).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_report_info.py -v

Requieren matplotlib (el reporte ya depende de él); si no está, se omiten.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

try:
    from matplotlib.backends.backend_pdf import PdfPages

    from stats import report_info as R

    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False

from analysis import FrameRecord
from analysis.quality import VERDICT_ADVICE, criteria_lines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = 0.1
CALIBRADO = {"escala_cm_por_px": 0.05, "origen_xy_px": (10.0, 290.0), "roi_box": (0, 0, 400, 300), "video_tamano_px": (400, 300)}


def recs(n, lost=(), rejected=()):
    out = []
    for i in range(n):
        if i in lost or i in rejected:
            out.append(FrameRecord(frame=i, time_s=i * DT, dist_acum_px=0.0, status="rechazado" if i in rejected else ""))
        else:
            out.append(FrameRecord(frame=i, time_s=i * DT, dist_acum_px=0.0, cx_abs=float(i), cy_abs_inv=10.0,
                                   dist_px=3.0, dist_cm=0.21))
    return out


def result(n=100, lost=(), rejected=(), settings=None):
    r = SimpleNamespace(records=recs(n, lost, rejected), timing=SimpleNamespace(dt=DT))
    if settings is not None:
        r.settings = settings
    return r


def aislados(n, k):
    paso = n // k
    return {paso * (i + 1) - 1 for i in range(k)}


@unittest.skipUnless(HAVE_MPL, "requiere matplotlib")
class CalibracionTests(unittest.TestCase):
    def test_escala_origen_roi_y_video(self):
        lines = R.calibration_lines(result(settings=CALIBRADO))
        self.assertEqual(lines[0], "Escala: 1 píxel = 0.05 cm (20 píxeles por cm)")
        texto = "\n".join(lines)
        self.assertIn("(10, 290)", texto)
        self.assertIn("400 × 300", texto)
        self.assertIn("y hacia arriba", texto)

    def test_sin_escala_lo_dice(self):
        lines = R.calibration_lines(result(settings={**CALIBRADO, "escala_cm_por_px": None}))
        self.assertIn("no calibrada", lines[0])
        self.assertNotIn("deducida", lines[0])

    def test_un_resultado_antiguo_deduce_la_escala_de_los_datos_y_lo_avisa(self):
        for settings in (None, {}):
            lines = R.calibration_lines(result(settings=settings))
            self.assertIn("0.07 cm", lines[0])  # 0.21 cm / 3 px
            self.assertIn("deducida de los datos", lines[0])

    def test_origen_no_definido(self):
        lines = R.calibration_lines(result(settings={**CALIBRADO, "origen_xy_px": None}))
        self.assertTrue(any("no definido" in ln for ln in lines))

    def test_sin_nada_no_falla(self):
        lines = R.calibration_lines(SimpleNamespace())
        self.assertEqual(len(lines), 1)
        self.assertIn("no calibrada", lines[0])


@unittest.skipUnless(HAVE_MPL, "requiere matplotlib")
class CalidadTests(unittest.TestCase):
    def test_veredicto_motivos_y_criterios(self):
        lines = R.quality_lines(result(100, lost=aislados(100, 10), settings=CALIBRADO))
        texto = "\n".join(lines)
        self.assertEqual(lines[0], "Veredicto: REVISAR")
        self.assertIn("Motivos de este veredicto", texto)
        self.assertIn("10.0 %", texto)
        for criterio in criteria_lines():  # los criterios de la página son exactamente los del módulo de calidad
            self.assertIn(criterio, texto)

    def test_el_peor_veredicto_recomienda_intentar_de_nuevo(self):
        lines = R.quality_lines(result(100, lost=aislados(100, 25), settings=CALIBRADO))
        self.assertEqual(lines[0], "Veredicto: DEFICIENTE")
        self.assertEqual(lines[1], f"Recomendación: {VERDICT_ADVICE['deficiente']}")
        self.assertIn("intentar de nuevo", lines[1])

    def test_un_bueno_no_trae_motivos(self):
        lines = R.quality_lines(result(100, settings=CALIBRADO))
        self.assertEqual(lines[0], "Veredicto: BUENO")
        self.assertNotIn("Motivos de este veredicto", "\n".join(lines))

    def test_se_informan_los_descartados_por_el_filtro(self):
        lines = R.quality_lines(result(100, rejected={10, 11, 12, 13, 14, 15}, settings=CALIBRADO))
        self.assertTrue(any(ln.startswith("Descartados por el filtro de saltos: 6") for ln in lines))

    def test_sin_registros(self):
        self.assertEqual(R.quality_lines(SimpleNamespace(records=[], timing=SimpleNamespace(dt=DT))),
                         ["No se calculó (no se guardaron los registros por frame)."])


@unittest.skipUnless(HAVE_MPL, "requiere matplotlib")
class PaginaTests(unittest.TestCase):
    def test_titulos_y_contenido(self):
        fig = R.report_page(result(100, lost=aislados(100, 10), settings=CALIBRADO))
        cuerpo = fig.axes[0].texts[0].get_text()
        self.assertIn("CALIBRACIÓN", cuerpo)
        self.assertIn("CALIDAD DEL SEGUIMIENTO", cuerpo)
        self.assertIn("Veredicto: REVISAR", cuerpo)
        self.assertIn("Escala: 1 píxel = 0.05 cm", cuerpo)
        self.assertIn("CALIBRACIÓN Y CALIDAD", fig._suptitle.get_text())

    def test_el_texto_cabe_en_la_pagina_incluso_en_el_peor_caso(self):
        """Muchos motivos a la vez: se compara contra la altura real de la hoja (no contra un número fijo de líneas)."""
        peor = result(200, lost=aislados(200, 60) | set(range(10, 40)), rejected=set(range(100, 130)), settings=CALIBRADO)
        t = R.report_page(peor).axes[0].texts[0]
        altura_pt = 0.84 * R.A4[1] * 72  # alto del área de texto, en puntos
        lineas = t.get_text().count("\n") + 1
        self.assertLessEqual(lineas * t.get_fontsize() * 1.35, altura_pt, msg=f"{lineas} líneas a {t.get_fontsize()} pt")

    def test_ninguna_linea_excede_el_ancho_de_ajuste(self):
        t = R.report_page(result(100, lost=aislados(100, 10), settings=CALIBRADO)).axes[0].texts[0].get_text()
        self.assertLessEqual(max(len(ln) for ln in t.split("\n")), 94)

    def test_se_guarda_en_un_pdf_junto_con_otras_paginas(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = os.path.join(tmp, "reporte.pdf")
            with PdfPages(ruta) as pdf:
                for pagina in R.report_pages(result(100, lost=aislados(100, 10), settings=CALIBRADO)):
                    pdf.savefig(pagina, bbox_inches="tight")
                self.assertEqual(pdf.get_pagecount(), 1)
            with open(ruta, "rb") as f:
                self.assertEqual(f.read(5), b"%PDF-")
            self.assertGreater(os.path.getsize(ruta), 1000)

    def test_sin_efectos_globales(self):
        """No debe cargar pyplot (fija el backend de todo el proceso y abre una interfaz gráfica)."""
        code = ("import sys; sys.path.insert(0, %r); import stats.report_info; "
                "sys.exit(1 if 'matplotlib.pyplot' in sys.modules else 0)" % ROOT)
        self.assertEqual(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()