"""A linear claim on a committed table: g·M·e = v, for g in E^rows, e in E^width, v in E (`PROTOCOL.md` §4).

1. The prover sends w = g·M; the verifier checks ⟨w, e⟩ = v.
2. A coin gives γ, `rows` independent uniform elements of E; the prover sends z = γ·M.
3. A coin gives `queries` columns j; the prover opens them, and the verifier checks Enc(w)[j] = Σ_i g_i U[i, j] and
   Enc(z)[j] = Σ_i γ_i U[i, j], U the committed codewords.

z's check is the proximity test, w's the consistency test; both read the same columns.
"""
from __future__ import annotations

from collections.abc import Sequence

from . import code
from .field import Ext, dot, dot_f, ext_bytes, ext_from_bytes
from .session import ProverSession, Reader, Refused, VerifierSession, ext_coins, word_bytes
from .table import Shape, Table, columns


def _rows_combination(coeffs: Sequence[Ext], rows: Sequence[Sequence[int]]) -> list[Ext]:
    width = len(rows[0])
    return [dot_f(coeffs, [r[k] for r in rows]) for k in range(width)]


def _columns(s: ProverSession | VerifierSession, shape: Shape, queries: int) -> list[int]:
    reader = Reader(s, word_bytes(shape.n) * queries)
    return [reader.uniform(shape.n) for _ in range(queries)]


def prove(s: ProverSession, table: Table, g: Sequence[Ext], label: str, queries: int) -> None:
    s.message(f"{label}/w", b"".join(ext_bytes(x) for x in _rows_combination(g, table.rows)))
    gamma = ext_coins(s, table.shape.rows)
    s.message(f"{label}/z", b"".join(ext_bytes(x) for x in _rows_combination(gamma, table.rows)))
    s.message(f"{label}/open", table.open(_columns(s, table.shape, queries)))


def verify(s: VerifierSession, shape: Shape, root: bytes, g: Sequence[Ext], e: Sequence[Ext], v: Ext, label: str,
           queries: int) -> None:
    if len(g) != shape.rows or len(e) != shape.width:
        raise Refused("shape", f"{label}: g or e doesn't fit the table")
    w = ext_from_bytes(s.message(f"{label}/w"))
    if w is None or len(w) != shape.width:
        raise Refused("encoding", f"{label}/w")
    if dot(w, e) != v:
        raise Refused("claim", label)
    gamma = ext_coins(s, shape.rows)
    z = ext_from_bytes(s.message(f"{label}/z"))
    if z is None or len(z) != shape.width:
        raise Refused("encoding", f"{label}/z")
    js = _columns(s, shape, queries)
    cols, why = columns(shape, root, s.message(f"{label}/open"))
    if cols is None:
        raise Refused(why, f"{label}/open")
    if set(cols) != set(js):
        raise Refused("column-position", f"{label}: opened other columns than the coin drew")
    enc_w, enc_z = code.encode_ext(w, shape.n), code.encode_ext(z, shape.n)
    for j in sorted(cols):
        if dot_f(g, cols[j]) != enc_w[j]:
            raise Refused("column-claim", f"{label}: column {j}")
        if dot_f(gamma, cols[j]) != enc_z[j]:
            raise Refused("column-proximity", f"{label}: column {j}")
