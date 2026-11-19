"""`attention_key_units`: `AttentionHead_v21` is `AttentionHead_v19`'s circuit, each key block's softmax step and P·V chunks a
unit of their own, and every unit fits a proof block's 256 rows at any key count of the class of record."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from verity.circuits.defs import bind
from verity.circuits.evaluation.reference import evaluate
from verity.commitments.periods import interface as IF

from verity_vllm.definitions.registry import attention_key_units as AK
from verity_vllm.definitions.registry import dead_free as DF
from verity_vllm.definitions.registry import hopper
from verity_vllm.definitions.registry import prims as P

RNG = np.random.default_rng(20261118)
FA2 = dict(DOT=hopper.HopperBF16WgmmaDot16, INV=P.Fa2InvSum)
#: Llama-3.2-1B's heads at the window's grain: D 64, kBlockN 192, a key's value 4 slices of 16 columns, P·V units of 48 keys
LLAMA = dict(D=64, BN=192, PK=48, VS=16)
#: the class of record's most keys
MAX_T = 1087
#: key counts at and around each unit boundary: a P chunk's (48), a block's (192), and the class of record's most
LLAMA_T = [1, 16, 48, 49, 50, 51, 192, 193, MAX_T]
STEPS = frozenset({"AttnSoftmaxStep_v1", "AttnPV_v1", "AttnHeadEnd_v1", "AttnBlockFirst_v1", "AttnBlock_v5"})
#: bf16 words at the edges: ±0, ±inf, NaNs, the largest finite, the smallest subnormal, ±1, one ulp above 1
BF16_EDGES = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F7F, 0xFF7F, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x3F81]


def _bf16(x) -> list[int]:
    return [int(w) for w in (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16)]


def _cases(D: int, T: int) -> list[tuple[list[int], list[int], list[int]]]:
    """(q, K rows, V rows): edge words mixed, random words, normals at three scales, and all -inf keys."""
    n = D + 2 * T * D
    rows = [[int(w) for w in RNG.choice(BF16_EDGES, n)], [int(w) for w in RNG.integers(0, 1 << 16, n)]]
    rows += [_bf16(RNG.normal(0, s, n)) for s in (0.1, 1.0, 30.0)]
    cases = [(r[:D], r[D:D + T * D], r[D + T * D:]) for r in rows]
    return cases + [(_bf16(RNG.normal(0, 1, D)), [0xFF80] * (T * D), _bf16(RNG.normal(0, 1, T * D)))]


def _circuit(fn, table: dict) -> tuple[Counter, list]:
    """``fn``'s gates with the Calls of `STEPS`' families written in place, order-free: each node by its Definition, axes,
    counts and operands (a parameter leaf or a node's output leaf, nodes interned in ``table``), with how many there are, and
    the return's leaves.  Equal for two Definitions, they are one circuit up to the order of its nodes: one function, gate
    for gate."""
    counts: Counter = Counter()

    def run(body, params):
        outs: list = []

        def src(r):
            s, i, j = r
            return params[i][j] if s == "p" else outs[i][j]

        for node in body.nodes:
            args = tuple(tuple(map(src, a.refs)) for a in node.args)
            if node.form == "call" and node.fn.base_id in STEPS:
                outs.append(run(node.fn.body, args))
                continue
            c = table.setdefault((node.form, node.fn.id, node.axes, node.n, node.n_xs, args), len(table))
            counts[c] += 1
            outs.append([("n", c, j) for j in range(node.out.leaves)])
        return [src(r) for r in body.ret]

    return counts, run(fn.body, [[("p", i, j) for j in range(t.leaves)] for i, (_, t) in enumerate(fn.body.params)])


def _heads(T: int, MF: int, D: int, BN: int, PK: int, VS: int):
    return (bind(AK.AttentionHeadV21, T=T, D=D, BN=BN, MASKED_FROM=MF, PK=PK, VS=VS, **FA2),
            bind(DF.AttentionHeadV19, T=T, D=D, BN=BN, MASKED_FROM=MF, **FA2))


def _same_circuit(new, old) -> None:
    table: dict = {}
    assert _circuit(new, table) == _circuit(old, table)


SMALL = [(1, 32, 0), (8, 32, 0), (15, 32, 0), (16, 32, 0), (17, 32, 0), (32, 32, 0), (40, 32, 1), (48, 32, 0), (48, 32, 1),
         (96, 32, 2), (17, 16, 1), (100, 64, 0)]


@pytest.mark.parametrize("T,BN,MF", SMALL)
@pytest.mark.parametrize("PK", [16, 32])
def test_head_v21_is_v19s_circuit_and_function(T, BN, MF, PK):
    """A first block of 1 ... BN keys, its last P chunk short or not, blocks below it guarded or not, chunks of one and two
    k-steps: v21's gates are v19's, and the two agree word for word."""
    new, old = _heads(T, MF, D=16, BN=BN, PK=PK, VS=4)
    _same_circuit(new, old)
    for q, k, v in _cases(16, T):
        assert evaluate(new, q, k, v) == evaluate(old, q, k, v)


@pytest.mark.slow
@pytest.mark.parametrize("T", LLAMA_T)
def test_head_v21_is_v19s_circuit_and_function_at_llama_geometry(T):
    """At Llama-3.2-1B's heads and every key count at a unit boundary up to the class of record's 1,087, with the first
    and the last block guarded: v21's gates are v19's, and the two agree word for word on edge words, random words,
    normals and all -inf keys."""
    nb = -(-T // LLAMA["BN"])
    for MF in sorted({0, nb - 1}):
        new, old = _heads(T, MF, **LLAMA)
        _same_circuit(new, old)
        for q, k, v in _cases(LLAMA["D"], T):
            assert evaluate(new, q, k, v) == evaluate(old, q, k, v)


def test_attention_v21_equals_v19():
    new = bind(AK.AttentionV21, T=48, NH=4, KVH=2, D=16, BN=32, MASKED_FROM=1, PK=16, VS=4, **FA2)
    old = bind(DF.AttentionV19, T=48, NH=4, KVH=2, D=16, BN=32, MASKED_FROM=1, **FA2)
    for _ in range(2):
        q, kc, vc = (_bf16(RNG.normal(0, 1, n)) for n in (4 * 16, 48 * 2 * 16, 48 * 2 * 16))
        assert evaluate(new, q, kc, vc) == evaluate(old, q, kc, vc)


def _unit_rows(fn) -> int:
    """A unit of ``fn``'s rows in and out with k cut a row a key and v a row a value slice, as the window reads them; refused
    (ValueError) where Flock's composer would refuse the proof block (`interface.check_proof_block`)."""
    cuts = {p: n for (p, _), n in zip(IF.layout(fn).rows, IF.layout(fn).outer) if p in ("k", "v")}
    IF.check_proof_block(fn, cuts)
    return len(IF.layout(fn).cut(cuts)) + len(IF.layout(fn).outs)


def _llama_units() -> dict[str, int]:
    """Every unit a head of Llama-3.2-1B's Calls at any key count, by id, with its rows: the softmax step at 1 ... BN keys,
    first or below with its block guarded or not, and P·V at 1 ... PK keys from zero, rescaled or carried."""
    D, BN, PK, VS, DOT = LLAMA["D"], LLAMA["BN"], LLAMA["PK"], LLAMA["VS"], FA2["DOT"]
    fns = [bind(AK.AttnSoftmaxStepV1, D=D, NVIS=n, FIRST=first, CHECK=check, DOT=DOT, PK=PK)
           for n in range(1, BN + 1) for first, check in ((True, True), (False, True), (False, False))]
    fns += [bind(AK.AttnPVV1, D=D, NV=n, FIRST=f, RESCALE=r, DOT=DOT, VS=VS)
            for n in range(1, PK + 1) for f, r in ((True, False), (False, True), (False, False))]
    fns.append(bind(DF.AttnHeadEndV1, D=D, INV=FA2["INV"]))
    return {fn.id: _unit_rows(fn) for fn in fns}


def test_every_unit_fits_a_proof_block_at_every_key_count():
    """Each unit a head Calls at T = 1 ... 1,087 keys is one of `_llama_units`, and each of those has a row past one SHA-512
    block and at most 256 rows: the softmax step 202 at a full block below the first (q, 192 keys, the running max and
    lane sums in; the max, lane sums, rescale factor and four P chunks out), P·V 196 at a rescaled chunk of 48 keys (O, the
    factor, P and 4 · 48 value slices in; O out)."""
    rows = _llama_units()
    assert max(rows.values()) <= IF.MAX_ROWS == 256
    soft = max(r for i, r in rows.items() if i.startswith("AttnSoftmaxStep_v1"))
    pv = max(r for i, r in rows.items() if i.startswith("AttnPV_v1"))
    assert (soft, pv) == (202, 196)
    called: set[str] = set()
    for T in range(1, MAX_T + 1):
        head = bind(AK.AttentionHeadV21, T=T, MASKED_FROM=0, **LLAMA, **FA2)
        called |= {n.fn.id for n in head.body.nodes}
        assert {n.form for n in head.body.nodes} == {"call"}
    assert called <= set(rows), sorted(called - set(rows))[:4]


def test_units_a_head_at_the_class_of_record():
    """At 1,087 keys a head is 6 blocks: the first of 127 keys (a softmax step and 3 P·V chunks), 5 full blocks below it
    (a softmax step and 4 chunks each) and the end, 30 units, where v19 is 7."""
    new, old = _heads(MAX_T, 0, **LLAMA)
    assert (len(new.body.nodes), len(old.body.nodes)) == (30, 7)


@pytest.mark.parametrize("fn,bindings,match", [
    (AK.AttnSoftmaxStepV1, dict(D=16, NVIS=4, FIRST=True, CHECK=False, PK=16), "FIRST needs CHECK"),
    (AK.AttnSoftmaxStepV1, dict(D=16, NVIS=4, FIRST=True, CHECK=True, PK=24), "PK must be a positive multiple of 16"),
    (AK.AttnPVV1, dict(D=16, NV=4, FIRST=True, RESCALE=True, VS=4), "FIRST and RESCALE exclude each other"),
    (AK.AttnPVV1, dict(D=16, NV=4, FIRST=True, RESCALE=False, VS=3), "VS must be a positive divisor of D"),
    (AK.AttentionHeadV21, dict(T=40, D=16, BN=40, MASKED_FROM=0, PK=16, VS=4, INV=P.Fa2InvSum), "BN must be"),
])
def test_refusals(fn, bindings, match):
    with pytest.raises(ValueError, match=match):
        bind(fn, DOT=FA2["DOT"], **bindings)
