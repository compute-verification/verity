"""The blockwise FP8 linear over its GEMM target's e4m3 k32 step, and its activation quant at G = 64.

* `ScaledMmFp8Block_v3{K,N,G,DOT}` and the shared-scale form's version 3 Definitions are the version 1 forms with the step lifted: at
  DOT = HopperE4m3QgmmaDot32 the same circuit gate for gate (so every word equal), at DOT = BlackwellE4m3QmmaDot32 another function;
  their version 4 forms are Boolean, agree with their word views, and are what the switch maps them to.
* `frontend.targets.scaled_mm_fp8_block_spec` binds by `GemmTarget.fp8_block_dot`: the version 1 forms on hopper (its Programs keep
  their digests), the version 3 forms over another target's step (blackwell_consumer's, which cites the cc 12.0 capture), and a refusal
  by name where a target records none.  The frontend rule and the fold's pattern ask it instead of comparing the compute capability with 9.
* `Fp8GroupQuant_v1{K,G}` at G = 64 (`fp8.QUANT_GROUPS`): a group's words are those of the G = 128 form over that group padded with
  zeros, and the rule and the fold bind it."""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.ops.prims import BlackwellE4m3QmmaDot32, HopperE4m3QgmmaDot32
from verity_vllm.observe.fold.patterns.fp8 import CutlassScaledMmFp8Block, PerTokenGroupFp8Quant
from verity_vllm.observe.fold.resolver import Compute, Unsupported, spec_label
from verity.ml.definitions import boolean as SW
from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET, TargetProfile
from verity.ml.definitions.registry import boolean_fp8 as BF
from verity.ml.definitions.registry import fp8 as F
from verity.ml.definitions.registry import fp8_vocab
from verity_vllm.program.frontend import gemm_targets as GT
from verity_vllm.program.frontend import targets as TG
from tests.program import test_fp8_profile as FP
from tests.program.test_boolean_fp8 import BF16_SPECIALS, _flat, _inputs
from tests.program.test_fp8_shared_scale import _canon, _Ctx

ONE, MINUS_ONE = 0x38, 0xB8          # e4m3 1.0, -1.0
F32_ONE = 0x3F800000
HOPPER = TargetProfile(compute_capability=(9, 0), num_sms=132, fp8_block_gemm="cutlass")
SM120 = TargetProfile(compute_capability=(12, 0), num_sms=188, fp8_block_gemm="cutlass")
ADA = TargetProfile(compute_capability=(8, 9), num_sms=128, fp8_block_gemm="cutlass")

def _word(defn, *args) -> int:
    return int(np.asarray(evaluate(defn, *args)).reshape(-1)[0])


# ---------------------------------------------------------------- the version 3 forms


def _v1_v3(K: int, N: int, G: int, dot=HopperE4m3QgmmaDot32):
    return [(bind(F.ScaledMmFp8BlockCoordinate, K=K, G=G), bind(F.ScaledMmFp8BlockCoordinateV3, K=K, G=G, DOT=dot)),
            (bind(F.ScaledMmFp8BlockCoordinateGivenScale, K=K, G=G), bind(F.ScaledMmFp8BlockCoordinateGivenScaleV3, K=K, G=G, DOT=dot)),
            (bind(F.ScaledMmFp8BlockSharedScaleTile, K=K, G=G), bind(F.ScaledMmFp8BlockSharedScaleTileV3, K=K, G=G, DOT=dot)),
            (bind(F.ScaledMmFp8Block, K=K, N=N, G=G), bind(F.ScaledMmFp8BlockV3, K=K, N=N, G=G, DOT=dot)),
            (bind(F.ScaledMmFp8BlockSharedScale, K=K, N=N, G=G), bind(F.ScaledMmFp8BlockSharedScaleV3, K=K, N=N, G=G, DOT=dot))]


@pytest.mark.parametrize("K,N", [(256, 384), (256, 320)])
def test_at_hoppers_step_each_version_3_form_is_the_version_1_circuit(K: int, N: int) -> None:
    """N = 320 ends mid-block (the ragged last weight block GLM-5.3's fused_qkv_a_proj has)."""
    for v1, v3 in _v1_v3(K, N, 128):
        assert v3.signature() == v1.signature() and v3.gates == v1.gates, v3.id
        table: dict = {}
        assert _canon(v1, table) == _canon(v3, table), v3.id


def test_at_blackwells_step_the_block_gemm_is_another_function() -> None:
    """One k block of four steps: 1 * 1, then 2^-9 * 2^-6 = 2^-15, then -1 * 1, then zeros.  Hopper's 14-bit adder truncates the
    second product against the accumulator, cc 12.0's 26-bit adder keeps it, and the cancellation brings it into bf16's range."""
    K = G = 128
    xq = [ONE] + [0] * 31 + [0x01] + [0] * 31 + [MINUS_ONE] + [0] * 63
    wq = [ONE] + [0] * 31 + [0x08] + [0] * 31 + [ONE] + [0] * 63
    hopper = _word(bind(F.ScaledMmFp8BlockCoordinate, K=K, G=G), xq, [F32_ONE], wq, [F32_ONE])
    assert _word(bind(F.ScaledMmFp8BlockCoordinateV3, K=K, G=G, DOT=HopperE4m3QgmmaDot32), xq, [F32_ONE], wq, [F32_ONE]) == hopper == 0
    sm120 = _word(bind(F.ScaledMmFp8BlockCoordinateV3, K=K, G=G, DOT=BlackwellE4m3QmmaDot32), xq, [F32_ONE], wq, [F32_ONE])
    assert sm120 == 0x3800                                                  # bf16(2^-15)
    row = evaluate(bind(F.ScaledMmFp8BlockSharedScaleV3, K=K, N=1, G=G, DOT=BlackwellE4m3QmmaDot32), xq, [F32_ONE], wq, [F32_ONE])
    assert [int(v) for v in np.asarray(row).reshape(-1)] == [sm120]


def test_block_gemm_picks_the_form_by_its_step() -> None:
    for scale, v1, v3 in (("per-coordinate", F.ScaledMmFp8Block, F.ScaledMmFp8BlockV3),
                          ("shared-per-block", F.ScaledMmFp8BlockSharedScale, F.ScaledMmFp8BlockSharedScaleV3)):
        assert fp8_vocab.block_gemm(256, 384, 128, scale) == bind(v1, K=256, N=384, G=128)
        assert fp8_vocab.block_gemm(256, 384, 128, scale, BlackwellE4m3QmmaDot32) == bind(v3, K=256, N=384, G=128, DOT=BlackwellE4m3QmmaDot32)
    assert fp8_vocab.SHARED_SCALE == {"ScaledMmFp8Block_v1": F.ScaledMmFp8BlockSharedScale}


# ---------------------------------------------------------------- the version 4 forms

BOOLEAN = [bind(BF.ScaledMmFp8BlockCoordinateV4, K=64, G=32, DOT=BG.BlackwellE4m3QmmaDot32),
           bind(BF.ScaledMmFp8BlockCoordinateV4, K=64, G=64, DOT=BG.BlackwellE4m3QmmaDot32),
           bind(BF.ScaledMmFp8BlockCoordinateV4, K=64, G=64, DOT=BG.HopperE4m3QgmmaDot32),
           bind(BF.ScaledMmFp8BlockCoordinateGivenScaleV4, K=64, G=32, DOT=BG.BlackwellE4m3QmmaDot32),
           bind(BF.ScaledMmFp8BlockSharedScaleTileV4, K=64, G=32, DOT=BG.BlackwellE4m3QmmaDot32),
           bind(BF.ScaledMmFp8BlockV4, K=64, N=48, G=32, DOT=BG.BlackwellE4m3QmmaDot32),
           bind(BF.ScaledMmFp8BlockSharedScaleV4, K=64, N=48, G=32, DOT=BG.BlackwellE4m3QmmaDot32)]


@pytest.mark.parametrize("target", BOOLEAN, ids=lambda t: t.id)
def test_the_version_4_form_agrees_with_its_word_view(target) -> None:
    word = TR.word_of(target)
    assert BL.is_boolean(target)
    assert (word.defn.name, word.defn.version) == (target.defn.name, 3)
    n = 64 if BS.compile(target).counts["and"] > 200_000 else 400
    inputs = _inputs(word, n, sum(map(ord, target.id)))
    got, ref = _flat(BS.evaluate_words(target, *inputs)), _flat(E.evaluate_batch(word, *inputs))
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{target.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"


@pytest.mark.parametrize("target", BOOLEAN, ids=lambda t: t.id)
def test_the_switch_maps_its_word_view_to_it(target) -> None:
    assert SW.boolean_version(TR.word_of(target)).id == target.id


# ---------------------------------------------------------------- the registry


@pytest.mark.parametrize("scale", [None, "shared-per-block", "per-coordinate"])
def test_hopper_binds_the_version_1_forms(scale) -> None:
    tp = HOPPER.with_overrides(fp8_scale_construction=scale)
    assert TG.fp8_block_dot(tp) is None
    v1 = F.ScaledMmFp8Block if scale == "per-coordinate" else F.ScaledMmFp8BlockSharedScale
    for K, N in ((2560, 6144), (6144, 2624)):
        assert TG.scaled_mm_fp8_block_spec(tp, K, N, 128) == bind(v1, K=K, N=N, G=128)


@pytest.mark.parametrize("cc", [(8, 9), (10, 0)])
def test_a_target_without_a_blockwise_step_is_refused(cc) -> None:
    tp = TargetProfile(compute_capability=cc)
    with pytest.raises(TG.UnregisteredTarget):
        TG.scaled_mm_fp8_block_spec(tp, 2560, 6144, 128)
    if cc != (10, 0):                                                     # cc 10.0 has no GEMM target at all
        with pytest.raises(TG.UnregisteredTarget, match=r"blockwise FP8 linear on ada \(8\.9\)"):
            TG.fp8_block_dot(tp)


def test_blackwell_consumer_records_its_blockwise_step_with_its_own_measurement() -> None:
    """cc 12.0's blockwise step is its `_C.cutlass_scaled_mm` blockwise kernel's own measurement, not the per-tensor FP8 linear's
    (`fp8_dot`): its evidence names the capture run and record and the cutlass cases it rests on."""
    gt = GT.REGISTERED["blackwell_consumer"]
    assert gt.fp8_block_dot == "BlackwellE4m3QmmaDot32"
    [ev] = [e for e in gt.evidence if e.startswith("the blockwise FP8 linear")]
    for needle in ("blackwell_fp8_capture", "r20261003-220018-3f6a", "0d346c79c642a0a1", "cutlass/*", "qkv_proj (K 4096, N 1280)",
                   "o_proj (K 1024, N 4096)", "M in {1, 2, 64, 65, 256, 257, 300}", "5,080,320 words",
                   "ScaledMmFp8Block_v3{DOT=BlackwellE4m3QmmaDot32_v1}"):
        assert needle in ev, needle
    assert {k for k, t in GT.REGISTERED.items() if t.fp8_block_dot} == {"hopper", "blackwell_consumer"}


def test_the_blockwise_step_is_keyed_on_its_own_field(monkeypatch) -> None:
    """Without `fp8_block_dot`, cc 12.0 is refused by name although its per-tensor and fused-MoE steps (`fp8_dot`, `fp8_moe_dot`)
    are the same step."""
    gt = dataclasses.replace(GT.REGISTERED["blackwell_consumer"], fp8_block_dot=None)
    monkeypatch.setitem(GT.REGISTERED, "blackwell_consumer", gt)
    assert gt.fp8_dot == gt.fp8_moe_dot == "BlackwellE4m3QmmaDot32"
    with pytest.raises(TG.UnregisteredTarget, match=r"blockwise FP8 linear on blackwell_consumer \(12\.0\): .* no registered Definition"):
        TG.scaled_mm_fp8_block_spec(SM120, 6144, 2624, 128)


@pytest.mark.parametrize("scale", ["shared-per-block", "per-coordinate"])
def test_blackwell_consumer_binds_the_version_3_forms_over_its_step(scale) -> None:
    assert TG.fp8_block_dot(SM120) is BlackwellE4m3QmmaDot32
    v3 = F.ScaledMmFp8BlockV3 if scale == "per-coordinate" else F.ScaledMmFp8BlockSharedScaleV3
    spec = TG.scaled_mm_fp8_block_spec(SM120, 6144, 2624, 128, scale)
    assert spec == bind(v3, K=6144, N=2624, G=128, DOT=BlackwellE4m3QmmaDot32) and "BlackwellE4m3QmmaDot32_v1" in spec.id


def test_the_vocabulary_binds_the_step_it_is_given() -> None:
    from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3 as V

    for scale, v1, v3 in (("per-coordinate", F.ScaledMmFp8Block, F.ScaledMmFp8BlockV3),
                          ("shared-per-block", F.ScaledMmFp8BlockSharedScale, F.ScaledMmFp8BlockSharedScaleV3)):
        assert V.kind("scaled_mm_fp8_block", K=256, N=384, G=128, SCALE=scale) == bind(v1, K=256, N=384, G=128)
        got = V.kind("scaled_mm_fp8_block", K=256, N=384, G=128, SCALE=scale, FP8_DOT=BlackwellE4m3QmmaDot32)
        assert got == bind(v3, K=256, N=384, G=128, DOT=BlackwellE4m3QmmaDot32)


# ---------------------------------------------------------------- the frontend rule


def _gate(tp):
    from verity_vllm.program.frontend.rules.vllm_bindings import clear_observations, observe_target_profile
    from verity_vllm.program.frontend.rules.vllm_bindings import fp8 as R

    clear_observations()
    try:
        observe_target_profile(tp)
        return R.CutlassScaledMmFp8BlockRule().applies(types.SimpleNamespace(args=(), kwargs={}), None)
    finally:
        clear_observations()


@pytest.mark.parametrize("cc", [(8, 9), (10, 0)])
def test_the_rule_refuses_a_target_without_a_blockwise_step(cc) -> None:
    reason = _gate(DEFAULT_TARGET.with_overrides(compute_capability=cc))
    assert reason.startswith("numerics_unregistered: ")
    if cc != (10, 0):
        assert "blockwise FP8 linear on ada (8.9)" in reason and "no registered Definition" in reason


def test_the_rule_passes_the_gate_where_a_step_is_recorded() -> None:
    """On cc 12.0, `VLLM_USE_DEEP_GEMM=0` also selects `CutlassFp8BlockScaledMMKernel`, whose `_C.cutlass_scaled_mm` is that target's
    own blockwise kernel: the kernel-class read-back cannot tell it from Hopper's, so the GEMM target decides."""
    assert _gate(DEFAULT_TARGET.with_overrides(compute_capability=(9, 0))) == "unexpected operator arity"
    assert _gate(DEFAULT_TARGET.with_overrides(compute_capability=(12, 0))) == "unexpected operator arity"


@pytest.mark.parametrize("scale", [None, "per-coordinate"])
def test_the_rule_binds_the_targets_step(monkeypatch, scale) -> None:
    from verity_vllm.program.frontend.rules.vllm_bindings import clear_observations, observe_target_profile
    from verity_vllm.program.frontend.rules.vllm_bindings import fp8 as R

    K, N = 256, 384
    monkeypatch.setattr(R, "shape", lambda ctx, v: (K, N))
    monkeypatch.setattr(R, "_untransposed", lambda ctx, v, a, b, name: name)
    monkeypatch.setattr(R, "rows_of", lambda coll: ([coll, coll], ()))
    monkeypatch.setattr(R, "from_rows", lambda outs, lead: outs)
    facts = {}
    for cc in ((9, 0), (12, 0)):
        clear_observations()
        try:
            observe_target_profile(DEFAULT_TARGET.with_overrides(compute_capability=cc, fp8_scale_construction=scale))
            ctx = _Ctx()
            R.CutlassScaledMmFp8BlockRule().emit(types.SimpleNamespace(args=("out", "a", "b", "a_s", "b_s")), ctx)
        finally:
            clear_observations()
        (facts[cc], _emitted), = ctx.records
        assert ctx.calls == [fp8_vocab.block_gemm(K, N, 128, scale or "shared-per-block", None if cc == (9, 0) else BlackwellE4m3QmmaDot32)] * 2
    assert "dot" not in facts[(9, 0)] and facts[(12, 0)]["dot"] == "BlackwellE4m3QmmaDot32_v1"


# ---------------------------------------------------------------- the fold's pattern


def _mm(tp):
    node = FP._node("_C.cutlass_scaled_mm.default", FP.MM_SCHEMA, FP.mm_args())
    return CutlassScaledMmFp8Block(group=FP.G).match(node, FP._ctx(cc=tp.compute_capability, target=tp))


def test_the_fold_refuses_a_target_without_a_blockwise_step() -> None:
    r = _mm(ADA)
    assert isinstance(r, Unsupported) and r.detail["compute_capability"] == [8, 9]
    assert r.reason == "blockwise FP8 linear on ada (8.9): this target's _C.cutlass_scaled_mm blockwise kernel has no registered Definition"


@pytest.mark.parametrize("scale", [None, "per-coordinate"])
def test_the_fold_binds_the_targets_step(scale) -> None:
    K, N, G = FP.K, FP.N, FP.G
    for tp, dot in ((HOPPER, None), (SM120, BlackwellE4m3QmmaDot32)):
        r = _mm(tp.with_overrides(fp8_scale_construction=scale))
        assert isinstance(r, Compute), r
        assert spec_label(r.spec) == spec_label(fp8_vocab.block_gemm(K, N, G, scale or "shared-per-block", dot))
    assert spec_label(r.spec).endswith(",DOT=BlackwellE4m3QmmaDot32_v1}")


# ---------------------------------------------------------------- the activation quant at G = 64


@pytest.mark.slow
def test_a_64_group_is_the_128_group_over_it_padded_with_zeros() -> None:
    """`_C.per_token_group_fp8_quant` runs one kernel at every group size; G = 128's Definition is exact-model-tested.  Zeros change no
    group's max (it starts at eps > 0), so each 64-wide group's e4m3 words and scale are the G = 128 form's over that group and 64 zeros."""
    rows, K = 96, 256
    rng = np.random.default_rng(64)
    x = np.concatenate([rng.normal(0, s, (rows, K)) for s in (1e-3, 1.0, 300.0)]).astype(np.float32)
    x = (x.view(np.uint32) >> 16).astype(np.uint16)
    planted = rng.random(x.shape) < 0.02
    x[planted] = rng.choice(BF16_SPECIALS, int(planted.sum()))
    y64 = np.asarray(E.evaluate_batch(bind(F.Fp8GroupQuant, K=K, G=64), x))         # [rows, K words | K/G scales]
    q64, s64 = y64[:, :K], y64[:, K:]
    groups = x.reshape(-1, 64)
    padded = np.concatenate([groups, np.zeros_like(groups)], axis=1)
    y128 = np.asarray(E.evaluate_batch(bind(F.Fp8GroupQuant, K=128, G=128), padded))
    q128, s128 = y128[:, :128], y128[:, 128]
    assert np.array_equal(q64.reshape(-1, 64), q128[:, :64]) and not q128[:, 64:].any()
    assert np.array_equal(s64.reshape(-1), s128)
    assert len(np.unique(s64)) > rows                                       # scales of every magnitude, not one saturated word
    assert F.QUANT_GROUPS == (64, 128)


def _quant_rule_ctx(K: int, G: int):
    import torch

    from verity_vllm.program.frontend.derive import Alloc
    from verity_vllm.program.frontend.rules.vllm_bindings import fp8 as R

    g = torch.fx.Graph()
    x, xq, xs = (g.placeholder(n) for n in ("x", "x_q", "x_s"))
    dtypes = {x: torch.bfloat16, xq: R.FP8_E4M3, xs: torch.float32}
    shapes = {x: (3, K), xq: (3, K), xs: (3, K // G)}
    ctx = types.SimpleNamespace(val=lambda v: Alloc(None, v), dtype_of=dtypes.get, shape_of_node=shapes.get)
    return (x, xq, xs), ctx


@pytest.mark.parametrize("G", [64, 128])
@pytest.mark.parametrize("ue8m0", [False, True])
def test_the_quant_rule_binds_every_registered_group(G: int, ue8m0: bool) -> None:
    """`use_ue8m0` selects the Definition the rule emits: False `fp8_group_quant` (Fp8GroupQuant_v1), True `fp8_group_quant_ue8m0`
    (Fp8GroupQuantUe8m0_v1, registry.deepgemm) -- the MoE experts' input quant when DeepGEMM is on, vLLM's sm_90 default."""
    from verity_vllm.program.frontend.rules.vllm_bindings import fp8 as R
    from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3 as V
    from verity.ml.definitions.registry import deepgemm as DG

    (x, xq, xs), ctx = _quant_rule_ctx(6144, G)
    node = types.SimpleNamespace(args=(x, xq, xs, G, 1e-10, -448.0, 448.0, ue8m0, False, False), kwargs={})
    assert R.PerTokenGroupFp8QuantRule().applies(node, ctx) is True
    assert V.kind("fp8_group_quant", K=6144, G=G) == bind(F.Fp8GroupQuant, K=6144, G=G)
    assert V.kind("fp8_group_quant_ue8m0", K=6144, G=G) == bind(DG.Fp8GroupQuantUe8m0, K=6144, G=G)


@pytest.mark.parametrize("G", [32, 96, 256, True])
def test_the_quant_rule_refuses_every_other_group(G) -> None:
    from verity_vllm.program.frontend.rules.vllm_bindings import fp8 as R

    (x, xq, xs), ctx = _quant_rule_ctx(6144, 64)
    node = types.SimpleNamespace(args=(x, xq, xs, G, 1e-10, -448.0, 448.0, False, False, False), kwargs={})
    assert R.PerTokenGroupFp8QuantRule().applies(node, ctx) == (
        f"numerics_unregistered: group_size={G!r}; Fp8GroupQuant_v1 is registered for the groups [64, 128]")


def _quant(group: int, G: int):
    xs = FP.view(FP.A_XS, (FP.M, FP.K // G), "float32", strides=(1, FP.M))
    node = FP._node("_C.per_token_group_fp8_quant.default", FP.QUANT_SCHEMA, FP.quant_args(G=G, xs=xs))
    return PerTokenGroupFp8Quant(group=group).match(node, FP._ctx())


def test_the_fold_resolves_the_quant_at_64() -> None:
    r = _quant(64, 64)
    assert isinstance(r, Compute), r
    assert spec_label(r.spec) == f"Fp8GroupQuant_v1{{K={FP.K},G=64}}" and r.outputs[1].view.shape == (FP.M, FP.K // 64)
    r = _quant(64, 128)
    assert isinstance(r, Unsupported) and "differs from the group this pattern resolves" in r.reason


@pytest.mark.parametrize("G", [32, 256])
def test_the_fold_refuses_an_unregistered_group(G: int) -> None:
    r = _quant(G, G)
    assert isinstance(r, Unsupported) and "registered" in r.reason and r.detail["registered"] == [64, 128]
