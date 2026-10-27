"""`GumbelTokenSelectSharedGreedy_v1` / `GumbelTopPTokenSelectSharedGreedy_v1` (registry.sampling): `temp == 0` computed once and
read by both halves.  The same function as `GumbelTokenSelect_v1` / `GumbelTopPTokenSelect_v1` -- the same circuit once the second
`temp == 0` is merged into the first (every input), bit-equal on every 2^16 temperature pattern (every 16th with top-p) and on a
randomized sweep with temp 0, -0, +-inf and NaNs -- and no redundant gate under the partition checker."""
from __future__ import annotations

import random
import re

import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import sampling as SM
from verity.ml.definitions.registry import sampling_vocab
from verity.ml.circuits.cut import call_scope
from verity.ml.definitions import word as W

PAIRS = [(SM.GumbelTokenSelect, SM.GumbelTokenSelectSharedGreedy), (SM.GumbelTopPTokenSelect, SM.GumbelTopPTokenSelectSharedGreedy)]
SPECIAL_TEMPS = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001,
                 0xFFBFFFFF, 0x00000001, 0x80000001, 0x007FFFFF, 0x3F4CCCCD, 0x3F000000, 0x40000000]


def _canon(fn, table: dict) -> tuple[int, ...]:
    """Each returned leaf as an interned gate DAG (primitive, operands; inputs by position): equal ids, the same function."""
    c, call = call_scope(fn)
    lo, n = call.offset, fn.gates
    ids = [0] * n
    for j in range(n):
        g = c.gate(lo + j)
        ops = tuple(ids[o - lo] if lo <= o < lo + n else table.setdefault(("in", o), len(table)) for o in g.operands)
        ids[j] = table.setdefault((g.prim.id, ops), len(table))
    from verity.ml.circuits.layout import resolve
    return tuple(ids[resolve(call, r) - lo] for r in fn.body.ret)


def _args(old, V, rng, temp):
    logits = [rng.choice((0x7F80, 0xFF80, 0x7FC0, 0x0000, 0x8000)) if rng.random() < 0.1 else rng.getrandbits(16) for _ in range(V)]
    seed, pos = rng.getrandbits(64), rng.getrandbits(20)
    if old is SM.GumbelTokenSelect:
        return (logits, seed, pos, temp)
    top_p = rng.choice((SM.f32_bits(1.0), SM.f32_bits(0.9), SM.f32_bits(0.5), SM.f32_bits(0.05), rng.getrandbits(32)))
    return (logits, top_p, seed, pos, temp, rng.choice(SM.SPLIT_COUNTS))


@pytest.mark.parametrize("old,new", PAIRS)
@pytest.mark.parametrize("V", [2, 16])
def test_the_same_circuit_with_the_flag_merged(old, new, V):
    table: dict = {}
    assert _canon(bind(old, V=V), table) == _canon(bind(new, V=V), table)
    assert bind(old, V=V).gates - bind(new, V=V).gates == 2                  # the second F32Eq and its Const32[0]


@pytest.mark.parametrize("old,new", PAIRS)
@pytest.mark.slow
def test_bit_equal_on_every_temperature_pattern(old, new):
    rng = random.Random(0x5EED)
    fo, fn = bind(old, V=2), bind(new, V=2)
    step = 1 if old is SM.GumbelTokenSelect else 16                         # top-p: every sign, exponent and top 3 mantissa bits
    for hi in range(0, 1 << 16, step):
        temp = hi << 16 | (hi * 0x9E37) & 0xFFFF if hi & 1 else hi << 16
        a = _args(old, 2, rng, temp)
        assert evaluate(fo, *a) == evaluate(fn, *a), hex(temp)


@pytest.mark.parametrize("old,new", PAIRS)
def test_bit_equal_on_a_randomized_sweep(old, new):
    rng = random.Random(0xC0FFEE)
    fo, fn = bind(old, V=16), bind(new, V=16)
    for k in range(160):
        temp = SPECIAL_TEMPS[k % len(SPECIAL_TEMPS)] if k < 64 else rng.getrandbits(32)
        a = _args(old, 16, rng, temp)
        assert evaluate(fo, *a) == evaluate(fn, *a), (hex(temp), a)


@pytest.mark.parametrize("old,new", PAIRS)
def test_no_redundant_gate(old, new):
    for f, redundant in ((bind(old, V=16), 1), (bind(new, V=16), 0)):
        G = W.Graph(f)
        R = W.units(G, 16, 32)
        assert R["cut"].ok and R["kinds"] == ["output"] and len(G.recomputed) == redundant == R["cut"].detail["redundant_gates"]


def test_new_ids_carry_no_version_in_their_names():
    for old, new in PAIRS:
        name = new.id.rsplit("_v", 1)[0]
        assert new.id == f"{name}_v1" and not re.search(r"[Vv]\d", name) and name.startswith(old.id.rsplit("_v", 1)[0])
        assert sampling_vocab.SHARED_GREEDY[old.id] is new and bind(new, V=16).signature() == bind(old, V=16).signature()


@pytest.mark.parametrize("construction, shared", [(None, True), ("shared-greedy", True), ("greedy-check-per-stage", False)])
def test_the_target_binds_the_select_in_the_vocabulary_and_the_fold(construction, shared):
    """`TargetProfile.sampler_construction` (None = `shared-greedy`) at the vocabulary role the frontend rule asks for and at the fold's
    top-p select (`TokenSelectGumbelTopP._stage3`, the record's declared target); the rule: tests/program/test_derive_stochastic.py."""
    from verity_vllm.observe.events import DispatchEnter, DispatchExit, RunHeader, ViewDesc
    from verity_vllm.observe.fold.patterns.sampling import SCRATCH_TOPP_SELECT, TokenSelectGumbelTopP
    from verity_vllm.observe.fold.resolver import StepContext
    from verity_vllm.observe.fold.tree import TreeNode
    from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3
    from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET
    V = 64
    tp = DEFAULT_TARGET.with_overrides(sampler_construction=construction)
    for old, new in PAIRS:
        assert sampling_vocab.select_for(old, tp.effective_sampler_construction()) is (new if shared else old)
    select = SM.GumbelTopPTokenSelectSharedGreedy if shared else SM.GumbelTopPTokenSelect
    assert PROFILE_B1_EAGER_V3.kind("token_select_stochastic", V=V, CONSTRUCTION=tp.effective_sampler_construction()).id == bind(select, V=V).id
    logits = ViewDesc(1, 0, (1, V), (V, 1), "bfloat16", "cuda:0")
    local_argmax, idx, out = ViewDesc(2, 0, (1, 4), (4, 1), "int64", "cuda:0"), *(ViewDesc(a, 0, (1, 1), (1, 1), "int64", "cuda:0") for a in (3, 4))
    node = TreeNode(DispatchEnter(1, None, 0, "sample", "aten.gather.default", "aten::gather(Tensor self, int dim, Tensor index, *, bool "
                                  "sparse_grad=False) -> Tensor", [local_argmax, 1, idx], {}, 7, None), DispatchExit(1, out))
    header = RunHeader("verity-capture/log/v1", "2026-09-28T00:00:00Z", "toy", {"target": tp.to_json()} if construction else {}, {}, {})
    stage2 = {"local_argmax": local_argmax, "bf16": logits, "S": 4, "kernels": (), "specs": [("r0", 5, 7, 0.8, 0.9)]}
    ctx = StepContext(header, None, None, [], {}, {}, None, scratch={SCRATCH_TOPP_SELECT: {"stage2": {idx.alloc: stage2}}})
    assert TokenSelectGumbelTopP()._stage3(node, ctx).spec.id == bind(select, V=V).id
