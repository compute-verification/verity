"""A linear claim on a committed table: g·M·e = v, for g in E^rows, e in E^width, v in E (`PROTOCOL.md` §4).

1. The prover sends w = g·M; the verifier checks ⟨w, e⟩ = v.
2. A coin β gives γ = (1, β, β², ...); the prover sends z = γ·M.
3. A coin gives `queries` columns j; the prover opens each, and the verifier checks Enc(w)[j] = Σ_i g_i U[i, j] and
   Enc(z)[j] = Σ_i γ_i U[i, j], U the committed codewords.

z's check is the proximity test, w's the consistency test; both read the same columns.
"""
from __future__ import annotations

from collections.abc import Sequence

from . import code
from .field import Ext, dot, dot_f, ext_bytes, ext_from_bytes, powers
from .session import ProverSession, Reader, Refused, VerifierSession, ext_coin
from .table import Shape, Table, column, opening_bytes, parse_opening


def _rows_combination(coeffs: Sequence[Ext], rows: Sequence[Sequence[int]]) -> list[Ext]:
    width = len(rows[0])
    return [dot_f(coeffs, [r[k] for r in rows]) for k in range(width)]


def _columns(s: ProverSession | VerifierSession, shape: Shape, queries: int) -> list[int]:
    reader = Reader(s, 8 * queries)
    return [reader.index(shape.n) for _ in range(queries)]


def prove(s: ProverSession, table: Table, g: Sequence[Ext], label: str, queries: int) -> None:
    s.message(f"{label}/w", b"".join(ext_bytes(x) for x in _rows_combination(g, table.rows)))
    gamma = powers(ext_coin(s), table.shape.rows)
    s.message(f"{label}/z", b"".join(ext_bytes(x) for x in _rows_combination(gamma, table.rows)))
    js = _columns(s, table.shape, queries)
    s.message(f"{label}/open", b"".join(opening_bytes(table.open(j)) for j in js))


def verify(s: VerifierSession, shape: Shape, root: bytes, g: Sequence[Ext], e: Sequence[Ext], v: Ext, label: str,
           queries: int) -> None:
    if len(g) != shape.rows or len(e) != shape.width:
        raise Refused("shape", f"{label}: g or e doesn't fit the table")
    w = ext_from_bytes(s.message(f"{label}/w"))
    if w is None or len(w) != shape.width:
        raise Refused("encoding", f"{label}/w")
    if dot(w, e) != v:
        raise Refused("claim", label)
    gamma = powers(ext_coin(s), shape.rows)
    z = ext_from_bytes(s.message(f"{label}/z"))
    if z is None or len(z) != shape.width:
        raise Refused("encoding", f"{label}/z")
    js = _columns(s, shape, queries)
    data = s.message(f"{label}/open")
    enc_w, enc_z = code.encode_ext(w, shape.n), code.encode_ext(z, shape.n)
    for j in js:
        parsed = parse_opening(data)
        if parsed is None:
            raise Refused("encoding", f"{label}/open")
        opening, data = parsed
        if opening.position != j:
            raise Refused("column-position", f"{label}: opened {opening.position}, sampled {j}")
        col = column(shape, root, opening)
        if col is None:
            raise Refused("column-path", f"{label}: column {j}")
        if dot_f(g, col) != enc_w[j]:
            raise Refused("column-claim", f"{label}: column {j}")
        if dot_f(gamma, col) != enc_z[j]:
            raise Refused("column-proximity", f"{label}: column {j}")
    if data:
        raise Refused("encoding", f"{label}/open: trailing bytes")
