"""Lookups at private positions by permutation: a Beneš network with prover-set switches.

The table rows `(j, T[j], tag 0)` (j public, T[j] a register), the queries `(i_k, v_k, tag 1)` (index and value both
witnesses) and padding copies of row 0 go through one Beneš network. The circuit checks only that the first output has tag
0 and that every tag-1 output repeats its predecessor's index and value. Every tag-0 element is a genuine table row, since
the network permutes its inputs, so every query's value is `T[i_k]`. No sort order is checked: the prover's routing just has
to put each query after a row with its index. Cost: `(2 log2 n - 1)·n/2·w` ANDs for the network and `n·w` for the
adjacency check, `w` = index bits + value bits + 1.
"""
from __future__ import annotations

import numpy as np

from .program import Program, Reg


def benes_route(dest: np.ndarray) -> np.ndarray:
    """Switch settings (2 log n - 1, n/2) that send input i to output dest[i] (`program.benes_apply`'s layout)."""
    dest = np.asarray(dest, dtype=np.int64)
    n = len(dest)
    L = n.bit_length() - 1
    assert 1 << L == n and n >= 2 and np.array_equal(np.sort(dest), np.arange(n))
    layers = [None] * (2 * L - 1)
    problems = [dest.tolist()]
    for lvl in range(L - 1):
        ins, outs, nxt = [], [], []
        for d in problems:
            m = len(d)
            src = [0] * m
            for i, o in enumerate(d):
                src[o] = i
            sub = [-1] * m
            for start in range(0, m, 2):
                if sub[start] != -1:
                    continue
                i = start
                while True:
                    sub[i], sub[i ^ 1] = 0, 1
                    j = src[d[i ^ 1] ^ 1]
                    if sub[j] != -1:
                        break
                    i = j
            in_sw = [sub[2 * k] for k in range(m // 2)]
            out_sw = [sub[src[2 * k]] for k in range(m // 2)]
            up = [0] * (m // 2)
            lo = [0] * (m // 2)
            for k in range(m // 2):
                i0 = 2 * k + in_sw[k]
                up[k] = d[i0] // 2
                lo[k] = d[i0 ^ 1] // 2
            ins += in_sw
            outs += out_sw
            nxt += [up, lo]
        layers[lvl] = ins
        layers[2 * L - 2 - lvl] = outs
        problems = nxt
    layers[L - 1] = [1 if d[0] == 1 else 0 for d in problems]
    return np.array(layers, dtype=np.uint8)


def _next_pow2(x: int) -> int:
    return 1 << max(1, (x - 1).bit_length())


def lookup(prog: Program, name: str, table: Reg, q_idx: Reg) -> Reg:
    """Values `table[q_idx]` as a witness register, checked by permutation.

    table: bits (N, wv), row j has index j; q_idx: bits (Q, wi), LSB first. Returns the hint register bits (Q, wv).
    Witnesses `<name>.vals` and `<name>.sw` (see `lookup_witness`)."""
    N, wv = table.shape
    Q, wi = q_idx.shape
    assert N <= 1 << wi
    n = _next_pow2(N + Q)
    L = n.bit_length() - 1
    vals = prog.witness(f"{name}.vals", "bit", (Q, wv))
    sw = prog.witness(f"{name}.sw", "bit", (2 * L - 1, n // 2))
    tidx = ((np.arange(N)[:, None] >> np.arange(wi)[None, :]) & 1).astype(np.uint8)
    pad = n - N - Q
    rows_idx = prog.concat([prog.const("bit", tidx), q_idx, prog.const("bit", np.zeros((pad, wi), np.uint8))])
    rows_val = prog.concat([table, vals, table[np.zeros(pad, dtype=np.int64)]])
    tag = prog.const("bit", np.r_[np.zeros(N, np.uint8), np.ones(Q, np.uint8), np.zeros(pad, np.uint8)][:, None])
    x = prog.concat([rows_idx, rows_val, tag], axis=1)
    w = wi + wv + 1
    y = prog.benes(x, sw)
    prog.assert_zero(y[0, w - 1:w], f"{name}: the first element is a query")
    d = y[1:, :w - 1] ^ y[:-1, :w - 1]
    prog.assert_zero(prog.band(y[1:, w - 1:w], d), f"{name}: a query's value is not its index's table value")
    return vals


def lookup_witness(name: str, N: int, q_idx: np.ndarray, table_vals: np.ndarray) -> dict[str, np.ndarray]:
    """`<name>.vals` (the table rows at q_idx) and `<name>.sw` (a routing that puts each query after its row)."""
    Q = len(q_idx)
    n = _next_pow2(N + Q)
    key = np.r_[np.arange(N) * 2, np.asarray(q_idx, dtype=np.int64) * 2 + 1, np.zeros(n - N - Q, dtype=np.int64)]
    order = np.argsort(key, kind="stable")
    dest = np.empty(n, dtype=np.int64)
    dest[order] = np.arange(n)
    return {f"{name}.vals": table_vals[np.asarray(q_idx, dtype=np.int64)], f"{name}.sw": benes_route(dest)}


def permutation_check(prog: Program, name: str, values: Reg) -> None:
    """Assert that values (n, w) bits, n a power of two, are a permutation of 0 .. n-1 (routed into order)."""
    n, w = values.shape
    L = n.bit_length() - 1
    sw = prog.witness(f"{name}.sw", "bit", (2 * L - 1, n // 2))
    y = prog.benes(values, sw)
    want = ((np.arange(n)[:, None] >> np.arange(w)[None, :]) & 1).astype(np.uint8)
    prog.assert_eq(y, prog.const("bit", want), f"{name}: not a permutation")


def permutation_witness(name: str, values: np.ndarray) -> dict[str, np.ndarray]:
    return {f"{name}.sw": benes_route(np.asarray(values, dtype=np.int64))}
