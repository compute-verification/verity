"""The statement Lin proves: a stratum's sampled matrix–vector proof blocks, each y_u = W[o_u : o_u + N] · a_u over F
(`PROTOCOL.md` §5).

`W` is a committed table of `rows` rows of `width` (the registered weights, committed once), `A` the sampled blocks'
input rows and `Y` their output rows (N padded to a power of two). The offsets o_u are public: which output coordinates a
proof block computes is the partition's structure. One session proves every block:

1. A coin β batches the n·N claims with ρ_{u,j} = β^{u·N + j}: C = Σ ρ_{u,j} y_{u,j}, which the prover sends.
2. A sumcheck reduces C = Σ_{u,k} G(u, k)·A(u, k), G(u, k) = Σ_j ρ_{u,j} W[o_u + j, k], to G̃(r) and Ã(r) at one point.
3. Three linear claims (`linear`) settle them: Y's ⟨(β^{uN})_u ⊗ (β^j)_j⟩ = C, A's eq(r_u) ⊗ eq(r_k) = Ã(r), and W's
   c ⊗ eq(r_k) = G̃(r), c_i = Σ_{o_u + j = i} eq(r_u, u)·ρ_{u,j}. W's is FlashProve's g·M·e: one opening of the weights
   for every sampled block.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from . import linear, sumcheck
from .field import Ext, add, embed, eq_table, ext_bytes, ext_from_bytes, mul, powers
from .session import Challenger, Coin, Message, ProverSession, Refused, VerifierSession, ext_coin
from .table import Shape, Table


def _pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


@dataclass(frozen=True)
class Params:
    blowup: int = 4
    queries: int = 241


@dataclass(frozen=True)
class Statement:
    rows: int
    width: int
    outputs: int
    offsets: tuple[int, ...]
    w_root: bytes
    a_root: bytes
    y_root: bytes
    params: Params = Params()

    def __post_init__(self) -> None:
        if self.width < 1 or self.width & (self.width - 1):
            raise ValueError("width is a power of two")
        if not 1 <= self.outputs <= self.rows:
            raise ValueError("a block has between 1 and `rows` outputs")
        if not self.offsets or any(not 0 <= o <= self.rows - self.outputs for o in self.offsets):
            raise ValueError("every block reads `outputs` rows of W")

    @property
    def blocks(self) -> int:
        return _pow2(len(self.offsets))

    @property
    def n_out(self) -> int:
        return _pow2(self.outputs)

    def shapes(self) -> tuple[Shape, Shape, Shape]:
        b = self.params.blowup
        return (Shape("verity/lin/weights", self.rows, self.width, b),
                Shape("verity/lin/inputs", self.blocks, self.width, b),
                Shape("verity/lin/outputs", self.blocks, self.n_out, b))

    def digest(self) -> bytes:
        doc = {"format": "verity/lin/matvec/v0", "rows": self.rows, "width": self.width, "outputs": self.outputs,
               "offsets": list(self.offsets), "roots": [self.w_root.hex(), self.a_root.hex(), self.y_root.hex()],
               "blowup": self.params.blowup, "queries": self.params.queries}
        return hashlib.sha512(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).digest()


def queries(bits: int, width: int, blowup: int) -> int:
    """The column queries a linear claim on a table of `width` at `blowup` needs for 2^-bits (`PROTOCOL.md` §4): a far
    table is caught per query with probability at least (e + 1)/n and a false claim on a close one with (d - e)/n,
    e = ⌊(d - 1)/3⌋; the weaker of the two sets the count."""
    n, d = width * blowup, width * blowup - width + 1
    e = (d - 1) // 3
    catch = min((e + 1) / n, (d - e) / n)
    return math.ceil(bits / -math.log2(1 - catch))


def input_tables(statement: Statement, a: Sequence[Sequence[int]], y: Sequence[Sequence[int]], salt=None) -> tuple[Table, Table]:
    """The sampled blocks' input and output tables, padded with zero rows and columns."""
    _, sa, sy = statement.shapes()
    pad_a = [list(r) for r in a] + [[0] * statement.width] * (sa.rows - len(a))
    pad_y = [list(r) + [0] * (sy.width - len(r)) for r in y] + [[0] * sy.width] * (sy.rows - len(y))
    kw = {} if salt is None else {"salt": salt}
    return Table(sa, pad_a, **kw), Table(sy, pad_y, **kw)


def _rho(statement: Statement, beta: Ext) -> tuple[list[Ext], list[Ext]]:
    """ρ_{u,j} = β^{uN} · β^j as its two factors, zero past the sampled blocks and outputs."""
    n, N = len(statement.offsets), statement.outputs
    bu = powers(powers(beta, N + 1)[N], n) + [(0, 0, 0)] * (statement.blocks - n)
    bj = powers(beta, N) + [(0, 0, 0)] * (statement.n_out - N)
    return bu, bj


def _w_coeffs(statement: Statement, eq_u: Sequence[Ext], bu: Sequence[Ext], bj: Sequence[Ext]) -> list[Ext]:
    c = [(0, 0, 0)] * statement.rows
    for u, o in enumerate(statement.offsets):
        for j in range(statement.outputs):
            c[o + j] = add(c[o + j], mul(mul(eq_u[u], bu[u]), bj[j]))
    return c


def prove(statement: Statement, w: Table, a: Table, y: Table, challenger: Challenger) -> list[Message]:
    sw, sa, sy = statement.shapes()
    if (w.shape, a.shape, y.shape) != (sw, sa, sy) or (w.root, a.root, y.root) != (statement.w_root, statement.a_root,
                                                                                      statement.y_root):
        raise ValueError("the tables are not the statement's")
    s = ProverSession(statement.digest(), challenger)
    bu, bj = _rho(statement, ext_coin(s))
    rho_y = [[mul(bu[u], bj[j]) for j in range(statement.n_out)] for u in range(statement.blocks)]
    claim = (0, 0, 0)
    for u in range(statement.blocks):
        for j in range(statement.n_out):
            claim = add(claim, mul(rho_y[u][j], embed(y.rows[u][j])))
    s.message("claim", ext_bytes(claim))
    K = statement.width
    gtab = [(0, 0, 0)] * (statement.blocks * K)
    for u, o in enumerate(statement.offsets):
        for j in range(statement.outputs):
            r = mul(bu[u], bj[j])
            row = w.rows[o + j]
            for k in range(K):
                gtab[u * K + k] = add(gtab[u * K + k], mul(r, embed(row[k])))
    atab = [embed(x) for row in a.rows for x in row]
    point, g_at, a_at = sumcheck.prove(s, gtab, atab, "sumcheck")
    s.message("ends", ext_bytes(g_at) + ext_bytes(a_at))
    eq_u = eq_table(point[K.bit_length() - 1:])
    q = statement.params.queries
    linear.prove(s, y, bu, "outputs", q)
    linear.prove(s, a, eq_u, "inputs", q)
    linear.prove(s, w, _w_coeffs(statement, eq_u, bu, bj), "weights", q)
    return s.messages


def verify(statement: Statement, messages: list[Message], log: list[Coin]) -> str | None:
    """None if the session proves the statement, else the first check that failed, by name."""
    try:
        _verify(statement, messages, log)
    except Refused as r:
        return r.code
    return None


def _verify(statement: Statement, messages: list[Message], log: list[Coin]) -> None:
    sw, sa, sy = statement.shapes()
    s = VerifierSession(statement.digest(), messages, log)
    bu, bj = _rho(statement, ext_coin(s))
    claim = ext_from_bytes(s.message("claim"))
    if claim is None or len(claim) != 1:
        raise Refused("encoding", "claim")
    K = statement.width
    lk = K.bit_length() - 1
    m = lk + (statement.blocks.bit_length() - 1)
    point, final = sumcheck.verify(s, m, claim[0], "sumcheck")
    ends = ext_from_bytes(s.message("ends"))
    if ends is None or len(ends) != 2:
        raise Refused("encoding", "ends")
    g_at, a_at = ends
    if mul(g_at, a_at) != final:
        raise Refused("sumcheck-final")
    eq_k, eq_u = eq_table(point[:lk]), eq_table(point[lk:])
    q = statement.params.queries
    linear.verify(s, sy, statement.y_root, bu, bj, claim[0], "outputs", q)
    linear.verify(s, sa, statement.a_root, eq_u, eq_k, a_at, "inputs", q)
    linear.verify(s, sw, statement.w_root, _w_coeffs(statement, eq_u, bu, bj), eq_k, g_at, "weights", q)
    s.finish()

