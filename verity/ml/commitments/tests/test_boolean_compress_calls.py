"""The hashing Definitions traced with calls (`trace.part`'s and `trace.definition`'s ``calls``): each SHA-512 compression in
the row-seg parts, the hm96 leaf and the frame-v3 leaf, node and leaf-at is a call of `sha512.Sha512CompressFolded` at its
arguments' pattern. Each computes what its builder's whole circuit (``traced``) computes, with the same ANDs, at the smallest
binding and at served ones; and the folded compression is `compress_ref` at every pattern circuit-check binds."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import const_bits
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments.gates import frame_v3 as F, hm96 as H, rowleaf as R
from verity.ml.commitments.gates.sha512 import IV, Sha512CompressFolded, compress_ref

ROW, SEG = "hm96-sha512/row/v2", "hm96-sha512/row-seg/v1"
#: (Definition, statics): the smallest bindings circuit-check takes and the served ones (a full 8,192-bit segment, a
#: 3,072-word bf16 row's outer hash, the leaf and node at the deployed depths' widths, the leaf-at block_check reads at)
CASES = [(R.Sha512RowSegSegment, (13,)), (R.Sha512RowSegSegment, (8192,)), (R.Sha512RowSegOuter, (13,)),
         (R.Sha512RowSegOuter, (49152,)), (H.Hm96Sha512Leaf, ()), (F.FrameV3Leaf, (1, ROW)), (F.FrameV3Leaf, (2, SEG)),
         (F.FrameV3Leaf, (3, ROW)), (F.FrameV3Node, (1,)), (F.FrameV3Node, (2,)), (F.FrameV3LeafAt, (1, 4, ROW)),
         (F.FrameV3LeafAt, (2, 4, ROW))]


def _ands(T: TR.Traced) -> int:
    """A trace's ANDs, each call's its folded circuit's."""
    n = T.counts["and"]
    for kind, _ in T.gates:
        if kind.startswith("call:"):
            _, name, P = kind.split(":")
            n += TR.FOLDED[name].traced(P).counts["and"]
    return n


def _flat(d):
    """``d`` as it was before calls: its builder's whole circuit, emitted."""
    if d.statics:
        return TR.part(d.name, d.statics, d.signature, d.build, d.doc, register=False)
    params, ret = d.signature(None)
    return TR.definition(d.name, d.version, d.build, params, ret, register=False)


def _words(bits) -> list[int]:
    return [sum(int(b) << j for j, b in enumerate(bits[64 * i:64 * i + 64])) for i in range(len(bits) // 64)]


@pytest.mark.parametrize("P", ["p0n1536", "iv"])
def test_the_folded_compression_is_the_reference(P: str) -> None:
    rng = random.Random(f"compress-folded-{P}")
    rows = np.array([[rng.randrange(2) for _ in range(1536 if P == "p0n1536" else 1024)] for _ in range(4)], dtype=np.uint8)
    if P == "iv":
        P = TR.encode_pattern(["o" if b.c else "z" for v in IV for b in const_bits(v, 64)] + [((i,), 0) for i in range(1024)])
        args = [[b.c for v in IV for b in const_bits(v, 64)] + row.tolist() for row in rows]
    else:
        args = [row.tolist() for row in rows]
    got = evaluate_bits(bind(Sha512CompressFolded, P=P), rows)
    for g, a in zip(got, args):
        assert _words(g.tolist()) == compress_ref(_words(a[:512]), _words(a[512:]))


@pytest.mark.slow
@pytest.mark.parametrize("d, statics", CASES, ids=[f"{d.name}{s}" for d, s in CASES])
def test_a_call_trace_computes_the_whole_circuit(d, statics) -> None:
    """Up to about two minutes a binding (the served segment's nine compressions traced whole and evaluated): what catches a
    pattern that names an argument wrong, which a smaller binding's constants may not reach, and an AND a call adds or drops."""
    S = dict(zip(d.statics, statics))
    flat, calls = d.traced(*statics), d.traced_calls(*statics)
    assert any(k.startswith("call:Sha512CompressFolded:") for k, _ in calls.gates)
    assert _ands(calls) == flat.counts["and"]
    params, _ = bind(d, **S).signature()
    rng = np.random.default_rng(194)
    xs = [rng.integers(0, 2, size=(3, t.leaves), dtype=np.uint8) for _, t in params]
    want = evaluate_bits(bind(_flat(d), **S), *xs)
    assert np.array_equal(evaluate_bits(bind(d, **S), *xs), want)
