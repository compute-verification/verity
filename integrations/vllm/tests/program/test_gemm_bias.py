"""`GemmBias_v1{K,N}`: vLLM's batch-invariant linear with a bias (`matmul_kernel_persistent`, then `output + bias` in bf16) as ONE
Definition, one `GemmBiasCoordinate_v1` per output word, so the pre-bias GEMM word is interior to its coordinate's unit instead of a
Call boundary serving never commits (bf16 addition is not invertible, so no host source recovers it).

What is under test: the Definition is bit-equal to `BiasAdd_v1(Gemm_v1(x, W), b)` on edge and random vectors; the frontend binds a
Triton GEMM launch whose only reader is the linear's bias add (in its module, the next computing node) to it, and leaves every other
shape as `Gemm_v1` (+ `BiasAdd_v1`); `Q_word_v1` cuts it into one unit per output word with no committed interior word; the replay's
row kernel agrees.  The launch is the export's carrier (`verity_cba::triton_launch`), built here as in `test_harden_alias.py`.

On a target whose GEMM record adds the bias in cuBLASLt's FP32 epilogue the same pair is one `GemmBiasF32Epilogue_v1` per row at
M >= 2 and cuBLAS's gemv `GemvBiasF32_v1` at M = 1 (the served `F.linear(x, W, b)` is one kernel, so there is no pre-bias word to
commit and the module's value is that Call's); one own row under a declared launch context takes the context's rows at its request
step (the sampling events before it), and is refused where the context declares none.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from verity.evaluation import evaluate
from verity.ir.defs import bind
from verity_vllm.pipeline.build import applicability_conditions
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules import triton_iface as _ti  # noqa: F401  (defines verity_cba::triton_launch)
from verity_vllm.program.frontend.rules import vllm_bindings as VB
from verity_vllm.program.frontend.rules.vllm_bindings import gemm_pins_for, vllm_ruleset
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_launch_context, launch_rows_read
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.kernels import rows as RK
from verity_vllm.program.registry import b1, gemm_bias, gemv_bias
from verity_vllm.program.registry import gemm_targets as GT
from verity_vllm.program.registry import targets as TG
from verity_vllm.program.registry.serve3_reference import materialize
from verity_vllm.query import word as W

BF = torch.bfloat16
M, K, N = 3, 32, 8
PROF = dict(mode="profile", profile="b1-eager-v3")
#: bf16 edge words: +-0, +-smallest subnormal, largest subnormal, smallest normal, +-max finite, +-inf, quiet NaNs, a signalling NaN, +-1
EDGES = (0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0x3F80, 0xBF80)


def _vectors(n_random: int, seed: int):
    """(x[K], W[N*K], b[N]) word lists: every leaf the same edge word, then every leaf an edge word, then near-1.0 words."""
    rng = np.random.default_rng(seed)
    out = [([e] * K, [e] * (N * K), [e] * N) for e in EDGES]
    for _ in range(n_random):
        out.append(tuple(rng.choice(EDGES, n).tolist() for n in (K, N * K, N)))
        out.append(tuple(((rng.integers(0x3E00, 0x4040, n)) | (rng.integers(0, 2, n) << 15)).tolist() for n in (K, N * K, N)))
    return out


def test_gemm_bias_is_bit_equal_to_gemm_then_bias_add_on_edge_and_random_vectors():
    gb, g, ba = bind(gemm_bias.GemmBias, K=K, N=N), bind(b1.Gemm, K=K, N=N), bind(b1.BiasAdd, N=N)
    vecs = _vectors(24, 7)
    for x, w, b in vecs:
        assert list(evaluate(gb, x, w, b)) == list(evaluate(ba, list(evaluate(g, x, w)), b)), (x[:2], w[:2], b[:2])
    # the replay's row kernel is the same function wherever it evaluates (it declines non-finite words, like every bf16 row kernel)
    for x, w, b in vecs[len(EDGES):]:
        got = RK.gemm_bias({"K": K, "N": N}, (np.array(x, np.uint16), np.array(w, np.uint16), np.array(b, np.uint16)))["0"]
        want = np.array(evaluate(gb, x, w, b), dtype=np.uint16)
        fin = [(v & 0x7F80) != 0x7F80 for v in list(x) + list(w) + list(b) + want.tolist()]
        if all(fin):
            assert got.tolist() == want.tolist()


def test_q_word_cuts_gemm_bias_into_one_unit_per_output_word_with_no_committed_interior():
    rule = W.unit_rule(bind(gemm_bias.GemmBias, K=K, N=N))
    assert rule["units_per_call"] == N and rule["committed_interior_words"] == 0 and rule["max_out_bits"] == 16, rule


# ---------------------------------------------------------------------------------------------------------
# the frontend: the export's Triton launch followed by the linear's bias add
# ---------------------------------------------------------------------------------------------------------


def _gemm_meta(m: int, n: int, k: int) -> str:
    pins, _src = gemm_pins_for(m, n, k)
    scalars = {"M": m, "N": n, "K": k, "stride_am": k, "stride_ak": 1, "stride_bk": 1, "stride_bn": k, "stride_cm": n, "stride_cn": 1,
               "NUM_SMS": pins["NUM_SMS"], **pins["constexprs"]}
    return json.dumps({"kernel": pins["kernel"], "src_sha256": pins["src_sha256"], "triton": "test", "grid": [1], "scalars": scalars,
                       "tensor_args": ["a_ptr", "b_ptr", "c_ptr"], "constexprs": list(pins["constexprs"]),
                       "geometry": {"num_warps": pins["num_warps"], "num_stages": pins["num_stages"]}, "tensor_meta": [None, None, None]})


class _Linear(nn.Module):
    """`linear_batch_invariant(x, W, b)`: `c = empty; matmul_kernel_persistent(x, W.t(), c); c + b` (what the export records)."""

    def __init__(self, seed: int = 3) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter((torch.randn(N, K, generator=g) * 0.3).to(BF), requires_grad=False)
        self.bias = nn.Parameter(torch.randn(N, generator=g).to(BF), requires_grad=False)

    def gemm(self, x):
        c = torch.empty(x.shape[0], N, dtype=BF)
        meta = _gemm_meta(x.shape[0], N, K)
        torch.ops.verity_cba.triton_launch(json.loads(meta)["kernel"], [x, self.weight.t(), c], meta)
        return c

    def forward(self, x):
        return self.gemm(x) + self.bias


class _BiasFirst(_Linear):
    def forward(self, x):
        return self.bias + self.gemm(x)


class _NoBias(_Linear):
    def forward(self, x):
        return self.gemm(x)


class _PreBiasAlsoReturned(_Linear):
    """The pre-bias output has a second reader: it is a value of the Program, so the pair stays two Definitions."""

    def forward(self, x):
        c = self.gemm(x)
        return c + self.bias, c


class _OtherOpBetween(_Linear):
    """Another computation between the launch and the add: the add is not the launch's next computing node."""

    def forward(self, x):
        c = self.gemm(x)
        h = x + x
        return c + self.bias, h


def _derive(m: nn.Module, rows: int = M):
    return derive_report(m, (torch.empty(rows, K, dtype=BF),), **PROF, name=type(m).__name__, rules=vllm_ruleset())


def _hist(program) -> dict[str, int]:
    out: dict[str, int] = {}
    for n in program.fn.body.nodes:
        fam = n.fn.id.split("{")[0]
        out[fam] = out.get(fam, 0) + 1
    return out


def _bits(t: torch.Tensor) -> list[int]:
    return t.contiguous().view(torch.int16).numpy().astype(np.uint16).reshape(-1).tolist()


@pytest.mark.parametrize("cls", [_Linear, _BiasFirst])
def test_the_biased_batch_invariant_linear_derives_to_one_gemm_bias_per_row(cls):
    m = cls()
    rep = _derive(m)
    assert rep.ok, rep.unsupported
    assert _hist(rep.program) == {"GemmBias_v1": M}, _hist(rep.program)
    apps = [a for a in rep.provenance["applications"] if a["rule"] == "TritonGemm[vllm-bi]_v4"]
    assert len(apps) == 1 and apps[0]["preconditions_checked"]["bias_add"] and apps[0]["emitted"] == f"GemmBias_v1{{K={K},N={N}}} x{M}"
    assert not any(a["rule"].startswith("BroadcastBiasAdd") for a in rep.provenance["applications"])
    roles = {w["name"]: w.get("role") for w in rep.provenance["weights"]}
    assert rep.provenance["weight_roles"]["bias"]["role"] == "linear.bias" and "bias" in roles
    # the Program's value is the pair's, word for word, on every row
    x = (torch.randn(M, K, generator=torch.Generator().manual_seed(11)) * 0.8).to(BF)
    words = [_bits(x), [v for w in rep.provenance["weights"] for v in _bits(dict(m.named_parameters())[w["name"]])]]
    t = materialize(rep.program, words)
    got = [int(t[gid]) & 0xFFFF for gid in rep.program.circuit.root_outputs()]
    g, ba = bind(b1.Gemm, K=K, N=N), bind(b1.BiasAdd, N=N)
    wb, bb = _bits(m.weight), _bits(m.bias)
    want = [v for r in range(M) for v in evaluate(ba, list(evaluate(g, _bits(x[r]), wb)), bb)]
    assert got == want


@pytest.mark.parametrize("cls,want", [(_NoBias, {"Gemm_v1": M}),
                                      (_PreBiasAlsoReturned, {"Gemm_v1": M, "BiasAdd_v1": M}),
                                      (_OtherOpBetween, {"Gemm_v1": M, "BiasAdd_v1": M, "ResidualAdd_v1": M})])
def test_every_other_shape_keeps_gemm_and_bias_add(cls, want):
    rep = _derive(cls())
    assert rep.ok, rep.unsupported
    assert _hist(rep.program) == want, _hist(rep.program)
    apps = [a for a in rep.provenance["applications"] if a["rule"] == "TritonGemm[vllm-bi]_v4"]
    assert len(apps) == 1 and apps[0]["preconditions_checked"]["bias_add"] is None


# ---------------------------------------------------------------------------------------------------------
# a target whose GEMM record adds the bias in cuBLASLt's FP32 epilogue
# ---------------------------------------------------------------------------------------------------------

F32_EPILOGUE_TP = TargetProfile(compute_capability=(12, 0), num_sms=188)
NO_BIAS_RECORD_TP = TargetProfile(compute_capability=(9, 0), num_sms=132)


def _observing(tp: TargetProfile):
    saved = dict(VB._OBSERVED)
    VB.observe_target_profile(tp)
    return saved


@pytest.fixture
def f32_target():
    assert F32_EPILOGUE_TP.gemm_target().bias_epilogue == "f32"
    saved = _observing(F32_EPILOGUE_TP)
    yield TG.tensor_core_dot(F32_EPILOGUE_TP)
    VB._OBSERVED.clear()
    VB._OBSERVED.update(saved)


@pytest.fixture
def gemv_row(monkeypatch):
    """A (K, N) entry in the target's gemv table for this test's small linear."""
    monkeypatch.setitem(GT.GEMV_BIAS_TABLES[F32_EPILOGUE_TP.gemm_target().key], (K, N), (1, 8))
    return bind(gemv_bias.GemvBiasF32, K=K, N=N, V=1, T=8)


def _values(rep, m: nn.Module, x: torch.Tensor) -> list[int]:
    words = [_bits(x), [v for w in rep.provenance["weights"] for v in _bits(dict(m.named_parameters())[w["name"]])]]
    t = materialize(rep.program, words)
    return [int(t[gid]) & 0xFFFF for gid in rep.program.circuit.root_outputs()]


def _x(rows: int) -> torch.Tensor:
    return (torch.randn(rows, K, generator=torch.Generator().manual_seed(11)) * 0.8).to(BF)


@pytest.mark.parametrize("cls", [_Linear, _BiasFirst])
def test_on_an_f32_epilogue_target_the_biased_linear_is_one_gemm_bias_f32_epilogue_per_row(cls, f32_target):
    m = cls()
    rep = _derive(m)
    assert rep.ok, rep.unsupported
    spec = bind(gemm_bias.GemmBiasF32Epilogue, K=K, N=N, DOT=f32_target)
    assert spec.id == TG.gemm_bias_spec(F32_EPILOGUE_TP, K, N).id
    assert _hist(rep.program) == {"GemmBiasF32Epilogue_v1": M}, _hist(rep.program)
    apps = [a for a in rep.provenance["applications"] if a["rule"] == "TritonGemm[vllm-bi]_v4"]
    assert len(apps) == 1 and apps[0]["preconditions_checked"]["bias_add"] and apps[0]["emitted"] == f"{spec.id} x{M}"
    assert not any(a["rule"].startswith("BroadcastBiasAdd") for a in rep.provenance["applications"])
    assert rep.provenance["weight_roles"]["bias"]["role"] == "linear.bias"
    x = _x(M)
    assert _values(rep, m, x) == [v for r in range(M) for v in evaluate(spec, _bits(x[r]), _bits(m.weight), _bits(m.bias))]


def test_on_an_f32_epilogue_target_one_row_is_cublas_gemv(f32_target, gemv_row):
    m = _Linear()
    rep = _derive(m, rows=1)
    assert rep.ok, rep.unsupported
    assert gemv_row.id == TG.gemv_bias_spec(F32_EPILOGUE_TP, K, N).id
    assert _hist(rep.program) == {"GemvBiasF32_v1": 1}, _hist(rep.program)
    x = _x(1)
    assert _values(rep, m, x) == list(evaluate(gemv_row, _bits(x[0]), _bits(m.weight), _bits(m.bias)))


def test_on_an_f32_epilogue_target_one_row_outside_the_gemv_table_is_refused(f32_target):
    rep = _derive(_Linear(), rows=1)
    assert not rep.ok and rep.unsupported.reason == "numerics_unregistered", rep.unsupported
    assert "unsupported:gemv-table" in rep.unsupported.hint, rep.unsupported.hint


class _Steps(_Linear):
    """Three request steps of one row each, as the request wrapper runs them: the step's biased linear, then its sampling event."""

    def forward(self, x):
        return tuple(torch.argmax(self.gemm(x) + self.bias, dim=-1) for _ in range(3))


def _families(program) -> list[str]:
    return [n.fn.id.split("{")[0] for n in program.fn.body.nodes]


def test_under_a_declared_launch_context_one_own_row_takes_the_launch_rows_of_its_step(f32_target, gemv_row):
    VB.observe_launch_context(TG.parse_launch_context({"max_seqlen_q_by_step": [1, 1, 1], "rows_by_step": [3, 1, 2]}))
    rep = _derive(_Steps(), rows=1)
    assert rep.ok, rep.unsupported
    assert _families(rep.program) == ["GemmBiasF32Epilogue_v1", "TokenSelect_v1", "GemvBiasF32_v1", "TokenSelect_v1",
                                      "GemmBiasF32Epilogue_v1", "TokenSelect_v1"], _families(rep.program)
    assert launch_rows_read()
    cond = applicability_conditions(F32_EPILOGUE_TP, {}, {}, {}, None, None, active_launch_context())
    assert cond["launch_context"]["rows_by_step"] == {"0": 3, "1": 1, "2": 2}


def test_several_own_rows_do_not_read_the_launch_rows(f32_target):
    VB.observe_launch_context(TG.parse_launch_context([8, 1]))
    rep = _derive(_Linear())
    assert rep.ok and _hist(rep.program) == {"GemmBiasF32Epilogue_v1": M}, rep.unsupported or _hist(rep.program)
    assert not launch_rows_read()
    assert "rows_by_step" not in applicability_conditions(F32_EPILOGUE_TP, {}, {}, {}, None, None, active_launch_context())["launch_context"]


@pytest.mark.parametrize("spec,why", [([8, 1, 1], "declares max_seqlen_q only"),
                                      ({"max_seqlen_q_by_step": [1, 1, 1], "rows_by_step": "conflict"}, "'conflict'"),
                                      ({"max_seqlen_q_by_step": [1, 1, 1], "rows_by_step": [3]}, "names rows for steps [0]")])
def test_one_own_row_under_a_context_without_its_rows_is_refused(spec, why, f32_target, gemv_row):
    VB.observe_launch_context(TG.parse_launch_context(spec))
    rep = _derive(_Steps(), rows=1)
    assert not rep.ok and rep.unsupported.reason == "numerics_unregistered", rep.unsupported
    assert why in rep.unsupported.hint and "request step" in rep.unsupported.hint, rep.unsupported.hint


@pytest.mark.parametrize("cls,want", [(_NoBias, {"Gemm_v2": M}),
                                      (_PreBiasAlsoReturned, {"Gemm_v2": M, "BiasAdd_v1": M}),
                                      (_OtherOpBetween, {"Gemm_v2": M, "BiasAdd_v1": M, "ResidualAdd_v1": M})])
def test_on_an_f32_epilogue_target_every_other_shape_keeps_its_gemm(cls, want, f32_target):
    rep = _derive(cls())
    assert rep.ok, rep.unsupported
    assert _hist(rep.program) == want, _hist(rep.program)


def test_a_target_without_a_bias_record_keeps_gemm_and_bias_add():
    assert NO_BIAS_RECORD_TP.gemm_target().bias_epilogue is None
    saved = _observing(NO_BIAS_RECORD_TP)
    try:
        rep = _derive(_Linear())
    finally:
        VB._OBSERVED.clear()
        VB._OBSERVED.update(saved)
    assert rep.ok, rep.unsupported
    assert _hist(rep.program) == {"Gemm_v2": M, "BiasAdd_v1": M}, _hist(rep.program)
