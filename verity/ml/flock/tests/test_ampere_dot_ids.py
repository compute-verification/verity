"""The Ampere tensor-core step under both ids, the vLLM registry's ``AmpereBF16TcDot16_v1`` and core's ``_v2``: the same
function on the A100 / RTX 4090 captures and on random finite and non-finite words, and C-Flock lowers a program using either
to the same circuit, gate for gate."""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

from verity.ml.circuits.defs import PrimitiveDefinition, bind
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16
from verity.ml.circuits.silicon.total import tc_dot_total
from verity_flock import gf2
from verity_flock import ir_lower as IL

CAPTURES = Path(__file__).resolve().parents[2] / "circuits" / "ops" / "tests" / "fixtures" / "tc-total-2026-09-07"
#: the vLLM registry's _v1 (verity.ml.definitions.registry.prims) restated as its body is, so this backend test needs no integration
V1 = PrimitiveDefinition("AmpereBF16TcDot16", 1, P.AmpereBF16TcDot16.params, P.F32,
                         lambda acc, *ab: tc_dot_total(AMPERE_BF16_M16N8K16, acc, tuple(ab[:16]), tuple(ab[16:32])), register=False)
V2 = P.AmpereBF16TcDot16


def _captured() -> list[tuple[list[int], int]]:
    out = []
    for path in sorted(CAPTURES.glob("*/mma_special.reduced.json.gz")):
        with gzip.open(path, "rt") as f:
            doc = json.load(f)
        for c in doc["cases"] + doc["pos"]:
            ab = [int(c[k][i:i + 4], 16) for k in ("a", "b") for i in range(0, 64, 4)]
            out.append(([int(c["acc"], 16)] + ab, int(c["d"] if isinstance(c["d"], str) else c["d"][0], 16)))
    return out


def _random(n: int) -> list[list[int]]:
    """Finite words near one binade (cancelling sums), full-range finite words (overflowing groups), subnormals, zeros, infinities
    and NaN payloads in operands and accumulator, and raw bits."""
    rng = random.Random(20260928)
    special16 = [0, 0x8000, 1, 0x8001, 0x007F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F81, 0x3F80]
    special32 = [0, 0x80000000, 1, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC12345,
                 0x7F800001, 0x7FFFFFFF]

    def bf(kind: int) -> int:
        s = rng.getrandbits(1) << 15
        if kind == 0:
            return s | (rng.randint(120, 134) << 7) | rng.getrandbits(7)
        if kind == 1:
            return s | (rng.randint(200, 254) << 7) | rng.getrandbits(7)
        return rng.choice(special16) if rng.random() < 0.2 else rng.getrandbits(16)

    vecs = []
    for i in range(n):
        kind = i % 3
        acc = (rng.getrandbits(1) << 31) | (rng.randint(110, 140) << 23) | rng.getrandbits(23) if kind == 0 else \
            rng.choice(special32) if rng.random() < 0.2 else rng.getrandbits(32)
        vecs.append([acc] + [bf(kind) for _ in range(32)])
    return vecs


def _structure(C: gf2.Circuit) -> tuple:
    """Everything the circuit commits to: each bit's kind and operands, the table reads, the ports and the assertions."""
    data = [tuple(x.key() for x in d) if k == "and" else d for k, d in zip(C.kind, C.data)]
    ports = lambda m: {n: [b.key() for b in bits] for n, bits in m.items()}  # noqa: E731
    return C.kind, data, C.lookups, ports(C.inputs), ports(C.outputs), [(n, x.key()) for n, x in C.asserts]


def _step_circuit(prim) -> gf2.Circuit:
    C = gf2.Circuit()
    acc = C.input("acc", 32)
    ab = [C.input(f"ab{j}", 16) for j in range(32)]
    C.output("d", IL.PIECES[prim.id](C, acc, *ab))
    return C


def test_both_ids_are_one_function_on_the_captures_and_random_words():
    cap = _captured()
    assert len(cap) > 60_000
    assert [(v, d) for v, d in cap if not V1.evaluate(*v) == V2.evaluate(*v) == d] == []
    vecs = _random(3000)
    assert [v for v in vecs if V1.evaluate(*v) != V2.evaluate(*v)] == []
    C = _step_circuit(V2)
    T = len(cap)
    ins = {"acc": IL.slice_words([v[0] for v, _ in cap], 32)} | {f"ab{j}": IL.slice_words([v[1 + j] for v, _ in cap], 16) for j in range(32)}
    outs, _ = C.evaluate(ins, T)
    assert [int(x) for x in IL.unslice_words(outs["d"], T)] == [d for _, d in cap]


def test_both_ids_lower_to_the_same_circuit():
    assert V1.id == "AmpereBF16TcDot16_v1" and V2.id == "AmpereBF16TcDot16_v2"
    assert IL.TC_STEPS[V1.id] == IL.TC_STEPS[V2.id]
    assert _structure(_step_circuit(V1)) == _structure(_step_circuit(V2))
    whole = []
    for dot in (V1, V2):
        d = bind(G.GemmCoordinate, K=64, DOT=dot)
        C = gf2.Circuit()
        ins = [C.input(f"in{j}", 16) for j in range(128)]
        for k, o in enumerate(IL.lower(C, d, ins)):
            C.output(f"out{k}", o)
        whole.append(_structure(C))
    assert whole[0] == whole[1]
    lows = [IL.lowering(bind(G.GemmCoordinate, K=64, DOT=dot), "gemm-coordinate/k64/tc-step", unit_fn=unit)
            for dot in (V1, V2) for unit in (IL.tc_units, lambda t, p=dot.id: IL.tc_units(t, p))]
    assert {low.units.signature for low in lows[:2]} == {(V1.id,)} and {low.units.signature for low in lows[2:]} == {(V2.id,)}
    assert len({low.text for low in lows}) == 1
    assert len({str(_structure(low.circuit)) for low in lows}) == 1
