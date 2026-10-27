"""The Build's lowering under `pouw:ncp-v2`, on a real `torch.export` of a small model.

The model has three linears. Each launches the batch-invariant Triton GEMM as vLLM's `linear_batch_invariant` does (the
export's carrier `verity_cba::triton_launch`, as in `test_gemm_bias.py`):
- qkv, a linear layer with its bias added after the GEMM;
- o, a linear layer without a bias;
- an lm_head that isn't a linear layer.

vLLM is not on the CPU test host, so the rule's `linear_types` is patched to this module's stand-in `LinearBase`.

- `torch.export` records each module's class as its qualified name, a string, and the rule resolves it (`module_type`).
- Under a target with `pouw:ncp-v2`, qkv (whose bias add is the GEMM's only reader) is one `NcpLinearRowBias_v1` per row and
  o one `NcpLinearRow_v1` per row, while the lm_head stays `Gemm_v1`. The Program's value is the circuit's y through the
  chain.
- Without `pouw:ncp-v2`, the same model is `GemmBias_v1`, `Gemm_v1`, `Gemm_v1`.
- `NcpLinearRowBias_v1` is `BiasAdd_v1` after `NcpLinearRow_v1`, bit for bit, and the replay's native row.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind
from verity_vllm.check.replay import pouw_circuit as RC
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules import triton_iface as _ti  # noqa: F401  (defines verity_cba::triton_launch)
from verity_vllm.program.frontend.rules.vllm_bindings import clear_observations, gemm_pins_for, observe_target_profile, vllm_ruleset
from verity_vllm.program.frontend.rules.vllm_bindings import triton_launches as TL
from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET
from verity.ml.definitions.registry import b1, pouw_rows
from verity.ml.definitions.numerics.serve3_reference import materialize

BF = torch.bfloat16
M, K, NQ, V = 3, 64, 64, 16
PROF = dict(mode="profile", profile="b1-eager-v3")
POUW = DEFAULT_TARGET.with_overrides(protocols=("pouw:ncp-v2", "sampled-proofs:rows-sha512"))


def _gemm_meta(m: int, n: int, k: int) -> str:
    pins, _src = gemm_pins_for(m, n, k)
    scalars = {"M": m, "N": n, "K": k, "stride_am": k, "stride_ak": 1, "stride_bk": 1, "stride_bn": k, "stride_cm": n, "stride_cn": 1,
               "NUM_SMS": pins["NUM_SMS"], **pins["constexprs"]}
    return json.dumps({"kernel": pins["kernel"], "src_sha256": pins["src_sha256"], "triton": "test", "grid": [1], "scalars": scalars,
                       "tensor_args": ["a_ptr", "b_ptr", "c_ptr"], "constexprs": list(pins["constexprs"]),
                       "geometry": {"num_warps": pins["num_warps"], "num_stages": pins["num_stages"]}, "tensor_meta": [None, None, None]})


def _gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """`matmul_persistent(x, W.t())` into a fresh `c`, as the export records it."""
    c = torch.empty(x.shape[0], w.shape[0], dtype=BF)
    meta = _gemm_meta(x.shape[0], w.shape[0], w.shape[1])
    torch.ops.verity_cba.triton_launch(json.loads(meta)["kernel"], [x, w.t(), c], meta)
    return c


def _param(g: torch.Generator, *shape: int, scale: float = 0.3) -> nn.Parameter:
    return nn.Parameter((torch.randn(*shape, generator=g) * scale).to(BF), requires_grad=False)


class LinearBase(nn.Module):
    """Stands in for vLLM's `LinearBase`: the rule's `linear_types` is patched to it."""


class QKVParallelLinear(LinearBase):
    def __init__(self, k: int, n: int, g: torch.Generator) -> None:
        super().__init__()
        self.weight, self.bias = _param(g, n, k), _param(g, n, scale=1.0)

    def forward(self, x):
        return _gemm(x, self.weight) + self.bias


class RowParallelLinear(LinearBase):
    def __init__(self, k: int, n: int, g: torch.Generator) -> None:
        super().__init__()
        self.weight = _param(g, n, k)

    def forward(self, x):
        return _gemm(x, self.weight)


class ParallelLMHead(nn.Module):
    def __init__(self, k: int, n: int, g: torch.Generator) -> None:
        super().__init__()
        self.weight = _param(g, n, k)

    def forward(self, x):
        return _gemm(x, self.weight)


class Tiny(nn.Module):
    def __init__(self, seed: int = 3) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.qkv, self.o, self.lm_head = QKVParallelLinear(K, NQ, g), RowParallelLinear(NQ, K, g), ParallelLMHead(K, V, g)

    def forward(self, x):
        return self.lm_head(self.o(self.qkv(x)))


@pytest.fixture
def linear_base(monkeypatch):
    monkeypatch.setattr(TL, "linear_types", lambda: (LinearBase,))


def _derive(m: nn.Module, target=None):
    clear_observations()
    try:
        if target is not None:
            observe_target_profile(target)
        return derive_report(m, (torch.empty(M, K, dtype=BF),), **PROF, name=type(m).__name__, rules=vllm_ruleset())
    finally:
        clear_observations()


def _hist(program) -> dict[str, int]:
    out: dict[str, int] = {}
    for n in program.fn.body.nodes:
        fam = n.fn.id.split("{")[0]
        out[fam] = out.get(fam, 0) + 1
    return out


def _bits(t: torch.Tensor) -> list[int]:
    return t.contiguous().view(torch.int16).numpy().astype(np.uint16).reshape(-1).tolist()


def test_torch_export_records_the_linear_class_as_its_qualified_name_and_the_rule_resolves_it():
    ep = torch.export.export(Tiny(), (torch.empty(M, K, dtype=BF),))
    launches = [n for n in ep.graph.nodes if n.op == "call_function" and "triton_launch" in str(n.target)]
    classes = [list(n.meta["nn_module_stack"].values())[-1][1] for n in launches]
    assert classes == [f"{__name__}.QKVParallelLinear", f"{__name__}.RowParallelLinear", f"{__name__}.ParallelLMHead"]
    assert [TL.module_type(c) for c in classes] == [QKVParallelLinear, RowParallelLinear, ParallelLMHead]
    assert TL.module_type(f"{__name__}.NoSuchClass") is None and TL.module_type("no.such.module.Class") is None


def test_under_pouw_the_linears_derive_to_pouw_rows_the_biased_one_fused_and_the_lm_head_stays_a_gemm(linear_base):
    m = Tiny()
    rep = _derive(m, POUW)
    assert rep.ok, rep.unsupported
    assert _hist(rep.program) == {"NcpLinearRowBias_v1": M, "NcpLinearRow_v1": M, "Gemm_v1": M}, _hist(rep.program)
    apps = {a["preconditions_checked"]["W_is_parameter"]: a for a in rep.provenance["applications"] if a["rule"] == "TritonGemm[vllm-bi]_v3"}
    assert apps["qkv.weight"]["emitted"] == f"NcpLinearRowBias_v1{{K={K},N={NQ}}} x{M}" and apps["qkv.weight"]["preconditions_checked"]["pouw"] == "ncp-v2"
    assert apps["o.weight"]["emitted"] == f"NcpLinearRow_v1{{K={NQ},N={K}}} x{M}"
    assert apps["lm_head.weight"]["emitted"] == f"Gemm_v1{{K={K},N={V}}} x{M}" and apps["lm_head.weight"]["preconditions_checked"]["pouw"] is None
    assert rep.provenance["weight_roles"]["qkv.bias"]["role"] == "linear.bias"
    # the Program's value is the circuit's y at qkv and o, then the lm_head's Gemm_v1, word for word on every row
    x = (torch.randn(M, K, generator=torch.Generator().manual_seed(11)) * 0.8).to(BF)
    params = dict(m.named_parameters())
    t = materialize(rep.program, [_bits(x), [v for w in rep.provenance["weights"] for v in _bits(params[w["name"]])]])
    got = [int(t[gid]) & 0xFFFF for gid in rep.program.circuit.root_outputs()]
    qkv, o = bind(pouw_rows.NcpLinearRowBias, K=K, N=NQ), bind(pouw_rows.NcpLinearRow, K=NQ, N=K)
    head = bind(b1.Gemm, K=K, N=V)
    want = []
    for r in range(M):
        h = list(evaluate(qkv, _bits(x[r]), _bits(m.qkv.weight), _bits(m.qkv.bias)))
        want += list(evaluate(head, list(evaluate(o, h, _bits(m.o.weight))), _bits(m.lm_head.weight)))
    assert got == want


def test_without_pouw_the_same_model_keeps_its_gemms(linear_base):
    rep = _derive(Tiny())
    assert rep.ok, rep.unsupported
    assert _hist(rep.program) == {"GemmBias_v1": M, "Gemm_v1": 2 * M}, _hist(rep.program)


def test_the_module_class_decides_which_launches_are_lowered(monkeypatch):
    """With only `RowParallelLinear` a linear layer, o is lowered and qkv keeps its `GemmBias_v1`."""
    monkeypatch.setattr(TL, "linear_types", lambda: (RowParallelLinear,))
    rep = _derive(Tiny(), POUW)
    assert rep.ok, rep.unsupported
    assert _hist(rep.program) == {"GemmBias_v1": M, "NcpLinearRow_v1": M, "Gemm_v1": M}, _hist(rep.program)


@pytest.mark.slow
def test_the_fused_row_is_bias_add_after_the_row_bit_for_bit_and_the_replays_native_row():
    rng = np.random.default_rng(4)
    fused, row, add = bind(pouw_rows.NcpLinearRowBias, K=K, N=NQ), bind(pouw_rows.NcpLinearRow, K=K, N=NQ), bind(b1.BiasAdd, N=NQ)

    def bf16(n, scale):
        return (np.asarray(rng.standard_normal(n) * scale, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    for scale in (1e-3, 1.0, 50.0, 1e-39, 0.0):
        x, w = bf16(K, scale), bf16(NQ * K, 0.5)
        for b in (bf16(NQ, 1.0), np.full(NQ, 0x8000, np.uint16), np.zeros(NQ, np.uint16)):
            got = list(evaluate(fused, [int(v) for v in x], [int(v) for v in w], [int(v) for v in b]))
            assert got == list(evaluate(add, list(evaluate(row, [int(v) for v in x], [int(v) for v in w])), [int(v) for v in b])), scale
            _a, s, _b, t, z = RC.int7_linear(x, w, NQ)
            assert got == RC.dequant(z, s, t, b), scale
