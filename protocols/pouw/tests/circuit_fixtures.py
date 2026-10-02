"""Inputs for the circuit tests: a seeded `ncp-v2` call at a small shape, and for the tile checks' hiding Programs (`pc4`,
`pc8`) seeded salts and the outputs `leaves`' reference gives them."""
from __future__ import annotations

import random
import struct
import types

from verity.commitments.rowleaf import ROLE_W, ROLE_X
from verity_pouw.circuit import leaves as L
from verity_pouw.circuit import reference as R
from verity_pouw.schemes import pearl_c as C


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bf16(x: float) -> int:
    return f32(x) >> 16


def call(m: int, k: int, n: int, seed: int = 7, *, rowids=None, colids=None, a=None) -> R.Call:
    rng = random.Random(seed)
    if a is None:
        a = tuple(tuple(rng.randint(-64, 63) for _ in range(k)) for _ in range(m))
    b = tuple(tuple(rng.randint(-64, 63) for _ in range(k)) for _ in range(n))
    return R.Call(a, tuple(f32(rng.uniform(1e-3, 2e-2)) for _ in range(m)), b,
                  tuple(f32(rng.uniform(1e-3, 2e-2)) for _ in range(n)), tuple(bf16(rng.uniform(-1, 1)) for _ in range(n)),
                  bytes(rng.randrange(256) for _ in range(32)), forward=3, layer=5, site=2, weight=11,
                  rowids=tuple(range(1000, 1000 + m)) if rowids is None else tuple(rowids),
                  colids=tuple(range(n)) if colids is None else tuple(colids))


# ---- the hiding Programs ----------------------------------------------------------------------------------------------

#: the Definitions whose gates two units of a hiding Program may both compute (a row's commit string, in its row unit and
#: in each tile that reads it), and the packing primitives a tile reads its rows with
LEAF_SCOPES = ("PouwSha512Compress_v1", "PouwHm96Mask_v1", "PouwHm96Sha512_v1")
PACKERS = ("PouwPack16x2_v1", "PouwPack8x4_v1", "PouwPack4x8_v1")


def salts(m: int, n: int, tile: tuple[int, int], seed: int = 11) -> dict[str, list]:
    """Salts for a hiding Program: ``ya`` and ``yam`` per row of A, ``yb`` and ``ybm`` per row of B, ``yt`` per tile."""
    rng = random.Random(seed)
    y = lambda: bytes(rng.randrange(256) for _ in range(L.INST.salt_bytes))       # noqa: E731
    ms, ns = m // tile[0], n // tile[1]
    return {"ya": [y() for _ in range(m)], "yam": [y() for _ in range(m)], "yb": [y() for _ in range(n)],
            "ybm": [y() for _ in range(n)], "yt": [[y() for _ in range(ns)] for _ in range(ms)]}


def salt_args(s: dict[str, list]) -> dict[str, list[int]]:
    """Each salt input as the Program reads it: big-endian halves, row-major."""
    flat = lambda v: [x for r in v for x in (r if isinstance(r, list) else [r])]      # noqa: E731
    return {name: [h for y in flat(v) for h in L.halves(y)] for name, v in s.items()}


def hidden_outputs(tile: tuple[int, int], ok_a, rows_a, mids_a, ok_b, rows_b, mids_b, c, u, s) -> tuple[tuple, tuple]:
    """A hiding Program's call side and weight side outputs, flat as `evaluate` returns them, by `leaves.commit` and
    `pearl_c.leaf`: from the admission bits, each row's 16-bit words (input rows ``rows_*``, nvfp4-p or fp8-p rows
    ``mids_*``), the checked words C̃ ``c`` and U ``u`` (M rows of N) and the salts ``s``."""
    tm, tn = tile
    ms, ns = len(rows_a) // tm, len(rows_b) // tn
    bc = lambda role, w, y: L.halves(L.commit(role, w, y))                             # noqa: E731
    am = [bc(ROLE_X, w, y) for w, y in zip(mids_a, s["yam"])]
    bm = [bc(ROLE_W, w, y) for w, y in zip(mids_b, s["ybm"])]
    out = list(ok_a) + [h for w, y in zip(rows_a, s["ya"]) for h in bc(ROLE_X, w, y)] + [h for v in am for h in v]
    out += [h for rs in range(ms) for _cs in range(ns) for v in am[rs * tm:(rs + 1) * tm] for h in v]
    out += [h for _rs in range(ms) for cs in range(ns) for v in bm[cs * tn:(cs + 1) * tn] for h in v]
    for rs in range(ms):
        for cs in range(ns):
            part = lambda x: [list(x[rs * tm + i][cs * tn:(cs + 1) * tn]) for i in range(tm)]   # noqa: E731
            digest = C.leaf(types.SimpleNamespace(c=part(c), u=part(u)))
            out += bc(ROLE_X, L.words_tile(digest), s["yt"][rs][cs])
    weight = list(ok_b) + [h for w, y in zip(rows_b, s["yb"]) for h in bc(ROLE_W, w, y)] + [h for v in bm for h in v]
    return tuple(out), tuple(weight)


def recomputed_across(graph, owner) -> tuple[int, list[tuple[str, str]]]:
    """How many values two units both compute, and (Definition, primitive) of each that is not a leaf's gate."""
    across = [j for j, f in graph.recomputed if owner[j] != owner[f]]
    where = [(graph.scope_names[graph.scope[j]].split("{")[0], graph.prim_names[graph.prim[j]]) for j in across]
    return len(across), [w for w in where if w[0] not in LEAF_SCOPES and w[1] not in PACKERS]
