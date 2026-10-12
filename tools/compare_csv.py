from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "FrameComparison",
    "SummaryComparison",
    "compare_frame_csv",
    "compare_summary_csv",
    "format_report",
    "main",
]

SUMMARY_HEADER = ["metrica", "valor"]
#: Claves del resumen que pueden diferir sin que sea un problema (rutas y nombres).
INFORMATIONAL_KEYS = ("experimento", "nombre_base_csv", "video_path")


def _read(path: str) -> List[List[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def _to_float(text: str) -> Optional[float]:
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _cell_status(a: str, b: str, tol: float) -> Tuple[str, float]:
    """('igual' | 'tolerancia' | 'vacio' | 'distinto', diferencia absoluta)."""
    if a == b:
        return "igual", 0.0
    if (a == "") != (b == ""):
        return "vacio", math.inf
    fa, fb = _to_float(a), _to_float(b)
    if fa is None or fb is None:
        return "distinto", math.inf
    d = abs(fa - fb)
    return ("tolerancia" if d <= tol else "distinto"), d


# ----------------------------------------------------------------------
# CSV frame a frame
# ----------------------------------------------------------------------
@dataclass
class ColumnStats:
    compared: int = 0
    exact: int = 0
    within_tol: int = 0
    different: int = 0
    max_abs: float = 0.0
    max_frame: Optional[int] = None


@dataclass
class FrameComparison:
    n_a: int
    n_b: int
    same_headers: bool
    headers_only_a: List[str]
    headers_only_b: List[str]
    frames_only_a: List[int]
    frames_only_b: List[int]
    columns: Dict[str, ColumnStats]
    first_divergence: Optional[Tuple[int, str, str, str]]  # (frame, columna, valor A, valor B)
    presence_mismatch_frames: List[int]  # frames donde uno encontró al sujeto y el otro no
    examples: List[Tuple[int, str, str, str]] = field(default_factory=list)
    tol: float = 0.0

    @property
    def total_different(self) -> int:
        return sum(c.different for c in self.columns.values())

    @property
    def total_within_tol(self) -> int:
        return sum(c.within_tol for c in self.columns.values())

    @property
    def ok(self) -> bool:
        return (self.same_headers and not self.frames_only_a and not self.frames_only_b
                and self.total_different == 0)

    @property
    def identical(self) -> bool:
        return self.ok and self.total_within_tol == 0


def compare_frame_csv(path_a: str, path_b: str, tol: float = 0.0, max_examples: int = 10) -> FrameComparison:
    """Compara dos CSV frame a frame, alineando las filas por la columna ``frame``."""
    rows_a, rows_b = _read(path_a), _read(path_b)
    if not rows_a or not rows_b:
        raise ValueError("Alguno de los archivos está vacío.")
    head_a, head_b = rows_a[0], rows_b[0]
    for name, head in (("A", head_a), ("B", head_b)):
        if "frame" not in head:
            raise ValueError(f"El archivo {name} no tiene columna 'frame'; ¿es un CSV frame a frame?")

    def index(head, rows):
        i = head.index("frame")
        return {int(r[i]): dict(zip(head, r)) for r in rows[1:] if r}

    by_a, by_b = index(head_a, rows_a), index(head_b, rows_b)
    common_cols = [c for c in head_a if c in head_b]
    columns = {c: ColumnStats() for c in common_cols if c != "frame"}
    first: Optional[Tuple[int, str, str, str]] = None
    examples: List[Tuple[int, str, str, str]] = []
    presence: List[int] = []

    for fr in sorted(set(by_a) & set(by_b)):
        ra, rb = by_a[fr], by_b[fr]
        if "cx_abs" in ra and "cx_abs" in rb and ((ra["cx_abs"] == "") != (rb["cx_abs"] == "")):
            presence.append(fr)
        for col in columns:
            st = columns[col]
            kind, d = _cell_status(ra[col], rb[col], tol)
            st.compared += 1
            if kind == "igual":
                st.exact += 1
                continue
            if kind == "tolerancia":
                st.within_tol += 1
                st.max_abs = max(st.max_abs, d)
                continue
            st.different += 1
            if d != math.inf and d > st.max_abs:
                st.max_abs, st.max_frame = d, fr
            elif st.max_frame is None:
                st.max_frame = fr
            if first is None:
                first = (fr, col, ra[col], rb[col])
            if len(examples) < max_examples:
                examples.append((fr, col, ra[col], rb[col]))

    return FrameComparison(
        n_a=len(by_a), n_b=len(by_b), same_headers=head_a == head_b,
        headers_only_a=[c for c in head_a if c not in head_b],
        headers_only_b=[c for c in head_b if c not in head_a],
        frames_only_a=sorted(set(by_a) - set(by_b)), frames_only_b=sorted(set(by_b) - set(by_a)),
        columns=columns, first_divergence=first, presence_mismatch_frames=presence,
        examples=examples, tol=tol,
    )


# ----------------------------------------------------------------------
# CSV de resumen
# ----------------------------------------------------------------------
@dataclass
class SummaryComparison:
    only_a: List[str]
    only_b: List[str]
    different: List[Tuple[str, str, str]]  # (metrica, valor A, valor B)
    within_tol: List[Tuple[str, str, str]]
    informational: List[Tuple[str, str, str]]  # difieren pero no cuentan
    n_a: int
    n_b: int
    tol: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.only_a and not self.only_b and not self.different

    @property
    def identical(self) -> bool:
        return self.ok and not self.within_tol and not self.informational


def compare_summary_csv(path_a: str, path_b: str, tol: float = 0.0,
                        informational: Sequence[str] = INFORMATIONAL_KEYS) -> SummaryComparison:
    ra, rb = _read(path_a), _read(path_b)
    if not ra or not rb or ra[0] != SUMMARY_HEADER or rb[0] != SUMMARY_HEADER:
        raise ValueError("Alguno de los archivos no es un CSV de resumen (cabecera 'metrica,valor').")
    da = {r[0]: (r[1] if len(r) > 1 else "") for r in ra[1:] if r}
    db = {r[0]: (r[1] if len(r) > 1 else "") for r in rb[1:] if r}
    different, within, info = [], [], []
    for k in da:
        if k not in db:
            continue
        kind, _ = _cell_status(da[k], db[k], tol)
        if kind == "igual":
            continue
        item = (k, da[k], db[k])
        if k in informational:
            info.append(item)
        elif kind == "tolerancia":
            within.append(item)
        else:
            different.append(item)
    return SummaryComparison(
        only_a=[k for k in da if k not in db], only_b=[k for k in db if k not in da],
        different=different, within_tol=within, informational=info, n_a=len(da), n_b=len(db), tol=tol,
    )


# ----------------------------------------------------------------------
# Informe
# ----------------------------------------------------------------------
def _verdict(cmp) -> str:
    if cmp.identical:
        return "RESULTADO: IDÉNTICOS"
    if cmp.ok:
        return f"RESULTADO: EQUIVALENTES (diferencias dentro de la tolerancia {cmp.tol:g})"
    return "RESULTADO: HAY DIFERENCIAS"


def format_report(cmp, max_rows: int = 12) -> str:
    out: List[str] = []
    if isinstance(cmp, SummaryComparison):
        out.append(f"Resumen: {cmp.n_a} métricas en A, {cmp.n_b} en B")
        if cmp.only_a:
            out.append(f"  Solo en A: {cmp.only_a}")
        if cmp.only_b:
            out.append(f"  Solo en B: {cmp.only_b}")
        for title, items in (("Distintas", cmp.different), ("Dentro de la tolerancia", cmp.within_tol),
                             ("Informativas (no cuentan)", cmp.informational)):
            if items:
                out.append(f"  {title}:")
                out += [f"    {k}: {a!r} -> {b!r}" for k, a, b in items[:max_rows]]
        out.append(_verdict(cmp))
        return "\n".join(out)

    out.append(f"Frames: {cmp.n_a} en A, {cmp.n_b} en B")
    if not cmp.same_headers:
        out.append("  ¡LAS CABECERAS NO COINCIDEN!")
        if cmp.headers_only_a:
            out.append(f"    Solo en A: {cmp.headers_only_a}")
        if cmp.headers_only_b:
            out.append(f"    Solo en B: {cmp.headers_only_b}")
        if not cmp.headers_only_a and not cmp.headers_only_b:
            out.append("    (mismas columnas, pero en otro orden)")
    if cmp.frames_only_a:
        out.append(f"  Frames solo en A: {len(cmp.frames_only_a)} (p. ej. {cmp.frames_only_a[:5]})")
    if cmp.frames_only_b:
        out.append(f"  Frames solo en B: {len(cmp.frames_only_b)} (p. ej. {cmp.frames_only_b[:5]})")
    if cmp.presence_mismatch_frames:
        p = cmp.presence_mismatch_frames
        out.append(f"  ¡Detección distinta (uno encontró al sujeto y el otro no) en {len(p)} frames!"
                   f" Primeros: {p[:8]}")
    if cmp.first_divergence:
        fr, col, a, b = cmp.first_divergence
        out.append(f"  PRIMERA DIVERGENCIA: frame {fr}, columna '{col}': A={a!r}  B={b!r}")

    malas = [(c, s) for c, s in cmp.columns.items() if s.different or s.within_tol]
    if malas:
        out.append("  Columnas con diferencias:")
        out.append(f"    {'columna':34s} {'distintas':>9s} {'en tol.':>8s} {'máx |dif|':>11s} {'en frame':>9s}")
        for c, s in sorted(malas, key=lambda x: -x[1].different)[:max_rows * 3]:
            mx = "—" if s.max_abs in (0.0, math.inf) else f"{s.max_abs:.6g}"
            out.append(f"    {c:34s} {s.different:9d} {s.within_tol:8d} {mx:>11s} {str(s.max_frame or ''):>9s}")
    if cmp.examples:
        out.append("  Primeras celdas distintas:")
        out += [f"    frame {fr:>6} {col}: {a!r} -> {b!r}" for fr, col, a, b in cmp.examples[:max_rows]]
    total = sum(s.compared for s in cmp.columns.values())
    exact = sum(s.exact for s in cmp.columns.values())
    out.append(f"  Celdas comparadas: {total} | idénticas: {exact} | en tolerancia: {cmp.total_within_tol}"
               f" | distintas: {cmp.total_different}")
    out.append(_verdict(cmp))
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compara dos CSV de TrackerLab (original vs nuevo).")
    ap.add_argument("original", help="CSV de la versión original (A)")
    ap.add_argument("nuevo", help="CSV de la versión nueva (B)")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="tolerancia absoluta para columnas numéricas (por defecto 0: texto idéntico)")
    ap.add_argument("--max-examples", type=int, default=10, help="cuántas celdas distintas listar")
    args = ap.parse_args(argv)
    try:
        head_a = _read(args.original)[:1]
        if head_a and head_a[0] == SUMMARY_HEADER:
            cmp = compare_summary_csv(args.original, args.nuevo, args.tol)
        else:
            cmp = compare_frame_csv(args.original, args.nuevo, args.tol, args.max_examples)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    print(format_report(cmp, args.max_examples))
    return 0 if cmp.ok else 1


if __name__ == "__main__":
    sys.exit(main())