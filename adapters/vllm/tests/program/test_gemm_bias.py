"""`GemmBias_v1{K,N}`: vLLM's batch-invariant linear with a bias (`matmul_kernel_persistent`, then `output + bias` in bf16) as ONE
Definition, one `GemmBiasCoordinate_v1` per output word, so the pre-bias GEMM word is interior to its coordinate's unit instead of a
Call boundary serving never commits (bf16 addition is not invertible, so no host source recovers it).

What is under test: the Definition is bit-equal to `BiasAdd_v1(Gemm_v1(x, W), b)` on edge and random vectors; the frontend binds a
Triton GEMM launch whose only reader is the linear's bias add (in its module, the next computing node) to it, and leaves every other
shape as `Gemm_v1` (+ `BiasAdd_v1`); `Q_word_v1` cuts it into one unit per output word with no committed interior word; the replay's
row kernel agrees.  The launch is the export's carrier (`verity_cba::triton_launch`), built here as in `test_harden_alias.py`.

`GemmBias_v2{K,N,DOT}` is the same pair with `Gemm_v2`'s k16 step: the frontend binds it where the target's record says
`gemm_bias_dot` (cc 12.0 and cc 9.0), at every M (vLLM's linear runs the same launch at M = 1); a family whose record lacks it keeps
`Gemm_v2` + `BiasAdd_v1`.  Either one-Call form is its module's one member `out`, which `query.manifest.bind` binds to the Linear's slot `0`.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules import triton_iface as _ti  # noqa: F401  (defines verity_cba::triton_launch)
from verity_vllm.program.frontend.rules import vllm_bindings as VB
from verity_vllm.program.frontend.rules.vllm_bindings import gemm_pins_for, vllm_ruleset
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity.ml.definitions.numerics import rows as RK
from verity.ml.definitions.registry import b1, gemm_bias, hopper
from verity_vllm.program.frontend import targets as TG
from verity.ml.definitions.numerics.serve3_reference import materialize
from verity_vllm.query import required as RQ
from verity.ml.definitions import word as W

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


DOTS = [b1.P.AmpereBF16TcDot16, hopper.HopperBF16WgmmaDot16]


@pytest.mark.parametrize("dot", DOTS, ids=lambda d: d.id)
def test_gemm_bias_v2_is_bit_equal_to_gemm_v2_then_bias_add_on_edge_and_random_vectors(dot):
    gb, g, ba = bind(gemm_bias.GemmBiasV2, K=K, N=N, DOT=dot), bind(b1.GemmV2, K=K, N=N, DOT=dot), bind(b1.BiasAdd, N=N)
    vecs = _vectors(24, 7)
    for x, w, b in vecs:
        assert list(evaluate(gb, x, w, b)) == list(evaluate(ba, list(evaluate(g, x, w)), b)), (x[:2], w[:2], b[:2])
    # Gemm_v2's row declines a non-finite word by raising, so only finite vectors reach it
    compared = 0
    for x, w, b in vecs[len(EDGES):]:
        want = np.array(evaluate(gb, x, w, b), dtype=np.uint16)
        if all((v & 0x7F80) != 0x7F80 for v in list(x) + list(w) + list(b) + want.tolist()):
            got = RK.gemm_bias_dot({"K": K, "N": N, "DOT": dot}, (np.array(x, np.uint16), np.array(w, np.uint16), np.array(b, np.uint16)))["0"]
            assert got.tolist() == want.tolist()
            compared += 1
    assert compared >= 24


@pytest.mark.parametrize("spec", [bind(gemm_bias.GemmBias, K=K, N=N), bind(gemm_bias.GemmBiasV2, K=K, N=N, DOT=hopper.HopperBF16WgmmaDot16)],
                         ids=lambda s: s.id.split("{")[0])
def test_the_replay_evaluates_a_one_call_biased_linear_from_a_committed_row_and_two_weight_slices(spec):
    from verity_vllm.check.replay.evaluate import evaluate as replay_evaluate
    from verity_vllm.query.manifest import format as RM
    x, w, b = _vectors(1, 5)[-1]
    out = replay_evaluate(spec.id.split("{")[0], RM.spec_statics(spec.id), [[np.array(x, np.uint16)], [], []],
                          [None, np.array(w, np.uint16), np.array(b, np.uint16)])["0"]
    assert out.tolist() == list(evaluate(spec, x, w, b))


def test_q_word_cuts_gemm_bias_v2_into_one_unit_per_output_word_with_no_committed_interior():
    rule = W.unit_rule(bind(gemm_bias.GemmBiasV2, K=K, N=N, DOT=hopper.HopperBF16WgmmaDot16))
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
    apps = [a for a in rep.provenance["applications"] if a["rule"] == "TritonGemm[vllm-bi]_v3"]
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
    apps = [a for a in rep.provenance["applications"] if a["rule"] == "TritonGemm[vllm-bi]_v3"]
    assert len(apps) == 1 and apps[0]["preconditions_checked"]["bias_add"] is None


# ---------------------------------------------------------------------------------------------------------
# a Gemm_v2 target: cc 12.0 and cc 9.0 bind the pair as GemmBias_v2
# ---------------------------------------------------------------------------------------------------------

SM120_TP = TargetProfile(compute_capability=(12, 0), num_sms=188)
H100_TP = TargetProfile(compute_capability=(9, 0), num_sms=132)
GEMM_V2_TARGETS = pytest.mark.parametrize("tp", [SM120_TP, H100_TP], ids=["cc12.0", "cc9.0"])


@pytest.fixture
def on_target():
    saved = dict(VB._OBSERVED)
    yield VB.observe_target_profile
    VB._OBSERVED.clear()
    VB._OBSERVED.update(saved)


def _values(rep, m: nn.Module, x: torch.Tensor) -> list[int]:
    words = [_bits(x), [v for w in rep.provenance["weights"] for v in _bits(dict(m.named_parameters())[w["name"]])]]
    t = materialize(rep.program, words)
    return [int(t[gid]) & 0xFFFF for gid in rep.program.circuit.root_outputs()]


def _pair(m: nn.Module, x: torch.Tensor, dot) -> list[int]:
    g, ba = bind(b1.GemmV2, K=K, N=N, DOT=dot), bind(b1.BiasAdd, N=N)
    return [v for r in range(x.shape[0]) for v in evaluate(ba, list(evaluate(g, _bits(x[r]), _bits(m.weight))), _bits(m.bias))]


@GEMM_V2_TARGETS
@pytest.mark.parametrize("rows", [M, 1])
@pytest.mark.parametrize("cls", [_Linear, _BiasFirst])
def test_the_biased_linear_derives_to_one_gemm_bias_v2_per_row(cls, rows, tp, on_target):
    on_target(tp)
    m = cls()
    rep = _derive(m, rows)
    assert rep.ok, rep.unsupported
    dot = TG.tensor_core_dot(tp)
    spec = TG.gemm_bias_spec(tp, K, N)
    assert dot is hopper.HopperBF16WgmmaDot16 and spec.id == bind(gemm_bias.GemmBiasV2, K=K, N=N, DOT=dot).id
    assert _hist(rep.program) == {"GemmBias_v2": rows}, _hist(rep.program)
    apps = [a for a in rep.provenance["applications"] if a["rule"] == "TritonGemm[vllm-bi]_v3"]
    assert len(apps) == 1 and apps[0]["preconditions_checked"]["bias_add"] and apps[0]["emitted"] == f"{spec.id} x{rows}"
    assert not any(a["rule"].startswith("BroadcastBiasAdd") for a in rep.provenance["applications"])
    assert rep.provenance["weight_roles"]["bias"]["role"] == "linear.bias"
    x = (torch.randn(rows, K, generator=torch.Generator().manual_seed(11)) * 0.8).to(BF)
    assert _values(rep, m, x) == _pair(m, x, dot)


@GEMM_V2_TARGETS
@pytest.mark.parametrize("cls,want", [(_NoBias, {"Gemm_v2": M}),
                                      (_PreBiasAlsoReturned, {"Gemm_v2": M, "BiasAdd_v1": M}),
                                      (_OtherOpBetween, {"Gemm_v2": M, "BiasAdd_v1": M, "ResidualAdd_v1": M})])
def test_every_other_shape_keeps_gemm_v2_and_bias_add(cls, want, tp, on_target):
    on_target(tp)
    rep = _derive(cls())
    assert rep.ok, rep.unsupported
    assert _hist(rep.program) == want, _hist(rep.program)


def test_a_family_whose_record_lacks_gemm_bias_dot_keeps_gemm_v2_then_bias_add(on_target, monkeypatch):
    monkeypatch.delitem(TG.TARGETS["hopper"], "gemm_bias_dot")
    on_target(H100_TP)
    m = _Linear()
    rep = _derive(m)
    assert rep.ok, rep.unsupported
    assert TG.gemm_bias_dot(H100_TP) is None and TG.gemm_bias_spec(H100_TP, K, N) is None
    assert _hist(rep.program) == {"Gemm_v2": M, "BiasAdd_v1": M}, _hist(rep.program)
    x = (torch.randn(M, K, generator=torch.Generator().manual_seed(11)) * 0.8).to(BF)
    assert _values(rep, m, x) == _pair(m, x, TG.tensor_core_dot(H100_TP))


# ---------------------------------------------------------------------------------------------------------
# the manifest: a one-Call biased linear has no call boundary
# ---------------------------------------------------------------------------------------------------------

def _qwen2_qkv_program(one_call: bool, rows: int = 2, layers: int = 2):
    """Qwen2's biased `qkv_proj` per layer and token row, named as a cc 12.0 Build names it, read by `o_proj`: one `GemmBias_v2`
    Call (`qkv_proj/add[r]`), or `Gemm_v2` (`qkv_proj/triton_launch_1[r]`) then `BiasAdd_v1` (`qkv_proj/add[r]/bias`) in one body."""
    from verity_vllm.query.units import ValueRef as R
    from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
    pre = "model.model.layers"
    names = [f"{pre}.{i}.self_attn.{p}" for i in range(layers) for p in ("qkv_proj.weight", "qkv_proj.bias", "o_proj.weight")]
    weights = InputView(-1, "weights", "{...}", tuple(Port(n, 64 * k, 64 * k + 64, 64, 16) for k, n in enumerate(names)))
    xs = InputView(-2, "x", f"Array<{rows * K},Value<16>>", (Port("x", 0, rows * K, rows * K, 16),))
    calls: list = []

    def add(spec, name, reads, n=N, width=16):
        calls.append(CallView(len(calls), spec, spec.split("{", 1)[0], "call", name, (Port("out", 0, n, n, width),), tuple(reads)))
        return len(calls) - 1

    last = None
    for r in range(rows):
        h = R(-2, "x")
        for i in range(layers):
            qkv, o = f"{pre}.{i}.self_attn.qkv_proj", f"{pre}.{i}.self_attn.o_proj"
            if one_call:
                y = add(f"GemmBias_v2{{K={K},N={N}}}", f"{qkv}/add[{r}]", [h, R(-1, f"{qkv}.weight"), R(-1, f"{qkv}.bias")])
            else:
                g = add(f"Gemm_v2{{K={K},N={N}}}", f"{qkv}/triton_launch_1[{r}]", [h, R(-1, f"{qkv}.weight")])
                y = add(f"BiasAdd_v1{{N={N}}}", f"{qkv}/add[{r}]/bias", [R(g, "out"), R(-1, f"{qkv}.bias")])
            last = add(f"Gemm_v2{{K={N},N={K}}}", f"{o}/triton_launch_2[{r}]", [R(y, "out"), R(-1, f"{o}.weight")], n=K)
            h = R(last, "out")
    add("ArgmaxTokenSelect_v1{N=1}", "argmax[0]", [R(last, "out")], n=1, width=32)
    return ProgramView(tuple(calls), (weights, xs), (R(len(calls) - 1, "out"),), "synthetic", "synthetic")


def test_a_one_call_biased_qkv_commits_one_member_and_no_call_boundary():
    """The Qwen2.5-1.5B B1 control on cc 12.0 (main without this binding) carried 3,584 more identities than the coverage Build: one
    `call_boundaries` row per (step, layer) at the pre-bias word `qkv_proj/triton_launch_1/out`, host-evaluated at Commit.  Under
    `Q_word` the one-Call form has none, and the module's one member `out` binds the Linear's committed slot `0`."""
    from verity_vllm.query.manifest.bind import Binder
    from verity_vllm.correspondence.reader_for_query import Correspondence
    res = {"program": {"digest": "d0", "function": "f"}, "attention_impls_observed": {},
           "target": {"declared": {"compute_capability": [12, 0]}}}
    man = {}
    for one_call in (True, False):
        P = _qwen2_qkv_program(one_call)
        man[one_call] = RQ.request_manifest(P, Correspondence(P, implementation_paths=["model.layers.0.self_attn.qkv_proj"]), res,
                                            request_id="r0").manifest
    rows = lambda m, fam: sorted((x["op_path"], x["output_member"], tuple(x["element_range"]), x.get("spec"))  # noqa: E731
                                 for x in m["identities"] if x["family"] == fam)
    qkv = [f"model.layers.{i}.self_attn.qkv_proj" for i in range(2)]
    assert rows(man[True], "call_boundaries") == []
    assert rows(man[False], "call_boundaries") == [(f"{q}/triton_launch_1", "out", (0, 2 * N), "Gemm_v2") for q in qkv]
    assert [r for r in rows(man[True], "instance_outputs") if r[0] in qkv] == [(q, "out", (0, 2 * N), "GemmBias_v2") for q in qkv]
    assert Binder.of_manifest(man[True], rank=0).bind(0, [f"{qkv[0]}/in", *(f"{q}/0" for q in qkv)]) == [
        (qkv[0], "in", 0, "label"), *((q, "out", 0, "slot") for q in qkv)]
    rest = lambda m: sorted(json.dumps({k: v for k, v in x.items() if k != "position"}, sort_keys=True) for x in m["identities"]  # noqa: E731
                            if x["family"] != "call_boundaries" and x["op_path"] not in qkv)
    assert rest(man[True]) == rest(man[False])
    assert len(man[False]["identities"]) - len(man[True]["identities"]) == len(qkv)
