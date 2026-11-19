"""The stand-in Boolean bodies the suite's test primitives get (`stand_in_body`), registered with `verity.circuits.program` as
this module loads: the body of the lookup registered before it (else the catalog's) where it has one, a stand-in for a primitive a test defines, and a refusal for a
primitive of `verity` or an adapter with none, or one a test defines under such a primitive's id. Another suite's conftest,
or a script that writes difftest vectors, loads it by file, the file among the inputs of every suite that runs it."""
from __future__ import annotations

import zlib

import numpy as np

from verity.circuits import boolean as BL
from verity.circuits import program as PG
from verity.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.evaluation import batch as BA
from verity.circuits.evaluation.bits import evaluate_words
from verity.circuits.ops.boolean import versions as V
from verity.circuits.refs import Coll, Explicit, array_of

_STAND_INS: dict[str, object] = {}
#: the packages whose primitives are the catalog's: none of them gets a stand-in
CATALOG = ("verity", "verity_vllm", "verity_torch")
#: a primitive of at most this many input bits gets its truth table as its stand-in
TABLE_BITS = 12


class StandInMismatch(ValueError):
    """A stand-in body that computes another function than its primitive's word function on a random input."""


def _table(p: PrimitiveDefinition, k: int) -> np.ndarray:
    """``p``'s output word on each of its 2^k inputs, input bit i of row v being bit i of v (leaves in order, LSB first)."""
    params, _ = p.signature()
    v, at, arrays = np.arange(1 << k, dtype=np.uint64), 0, []
    for _n, t in params:
        leaves = []
        for lw in t.leaf_widths():
            leaves.append((v >> np.uint64(at)) & np.uint64((1 << lw) - 1))
            at += lw
        arrays.append(np.stack(leaves, axis=1).reshape((1 << k, *BA.type_shape(t))) if leaves else np.zeros((1 << k, 0), np.uint64))
    return np.asarray(BA.evaluate_batch(p, *arrays, kernel="reference"), dtype=np.uint64).reshape(1 << k)


def _held(p: PrimitiveDefinition, body) -> None:
    """``body``'s word view equals ``p``'s word function on 64 seeded random inputs; `StandInMismatch` by name if not."""
    params, _ = p.signature()
    args = BA.sample_uniform(np.random.default_rng(zlib.crc32(p.id.encode())), 64, [t for _, t in params])
    want = np.asarray(BA.evaluate_batch(p, *args, kernel="reference")).reshape(64, -1)
    got = np.asarray(evaluate_words(body, *args)).reshape(64, -1)
    bad = np.flatnonzero((want != got).any(axis=1))
    if bad.size:
        i = int(bad[0])
        raise StandInMismatch(f"the stand-in body of {p.id} computes another function than {p.id}: on "
                              f"{[a[i].tolist() for a in args]} it gives {got[i].tolist()}, {p.id} gives {want[i].tolist()}")


def _made(p: PrimitiveDefinition, table: np.ndarray | None):
    params, ret = p.signature()
    w = ret.w

    def body(B, S, *args):
        bits = [Coll(BL.BIT, Explicit((r,))) for a in args for r in a.refs]

        def gate(g, x, y):
            return B.call(g, BL.operands(x, y))

        def mux(col: np.ndarray, i: int):
            if (col == col[0]).all():
                return B.call(BL.constant(int(col[0])))
            h = len(col) // 2
            lo, hi = mux(col[:h], i - 1), mux(col[h:], i - 1)
            return gate(BL.XOR, lo, gate(BL.AND, bits[i], gate(BL.XOR, lo, hi)))

        if table is not None:
            out = [mux((table >> np.uint64(j)) & np.uint64(1), len(bits) - 1) for j in range(w)]
        else:
            cols: list[list[Coll]] = [[] for _ in range(w)]
            at = 0
            for _n, t in params:
                for lw in t.leaf_widths():
                    for j in range(min(lw, w)):
                        cols[j].append(bits[at + j])
                    at += lw
            out = [B.call(BL.xor_gate(len(c)), BL.operands(*c)) if len(c) > 1 else c[0] if c else B.call(BL.constant(0))
                   for c in cols]
        return out[0] if w == 1 else array_of(out)

    sig = (tuple((n, BL.bit_type(t)) for n, t in params), BL.bit_type(ret))
    return bind(CompositeDefinition(f"StandIn[{p.name}]", 1, (), lambda S: sig, body, register=False))


def stand_in_body(p: PrimitiveDefinition):
    """A test primitive's Boolean body: output bit j the Xor of bit j of every input leaf (absent past a leaf's width) where
    that is its function on random inputs, else its truth table where it has at most `TABLE_BITS` input bits, else
    `StandInMismatch`."""
    if p.id not in _STAND_INS:
        made = _made(p, None)
        try:
            _held(p, made)
        except StandInMismatch:
            k = sum(sum(t.leaf_widths()) for _, t in p.signature()[0])
            if k > TABLE_BITS:
                raise
            made = _made(p, _table(p, k))
            _held(p, made)
        _STAND_INS[p.id] = made
    return _STAND_INS[p.id]


def _of_catalog(p: PrimitiveDefinition) -> bool:
    return getattr(p.evaluate, "__module__", "").split(".")[0] in CATALOG


_BEFORE = PG._BODY_LOOKUP[0] if PG._BODY_LOOKUP else V.boolean_version


def _lookup(p: PrimitiveDefinition):
    try:
        return _BEFORE(p)
    except LookupError:
        if _of_catalog(p) or (p.id in REGISTRY and _of_catalog(REGISTRY[p.id])):
            raise
        return stand_in_body(p)


PG.register_boolean_bodies(_lookup)
