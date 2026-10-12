"""The Build's rules for GLM-5.3's routed MoE on Hopper, on synthetic modules that launch the same ops:

* `_moe_C.grouped_topk` (`GroupedTopkRule`) derives to `MoeRouterSigmoidBias_v1{E,TOPK,SCALE,KERNEL}` per token row, KERNEL the launch
  `registry.moe.router_kernel` picks, with the bias as a weight Input; another grouping, scoring or an unpinned launch refuses by name;
* `_C.silu_and_mul_per_block_quant` (`SiluAndMulPerBlockQuantRule`) derives to `SiluMulFp8BlockQuant_v1{I,G}` per row, equal to its twin;
* on cc 9.0 without batch invariance a block-FP8 `fused_moe_kernel` launch on a tuned wgmma tile derives to
  `MoeExpertFp8Gemm(W)_v1{...,KB,SWAP,DOT=HopperE4m3QgmmaDot32_v1}`, and an `mma.sync` tile or a SWAP_AB that is not `enable_swap_ab`
  refuses by name;
* the chain router -> w13 -> fused activation quant -> w2, as `TritonExperts` runs it for GLM-5.3-FP8, derives in one Program whose
  words are the twins' on the routed experts."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules import triton_iface as _ti  # noqa: F401  (defines verity_cba::triton_launch)
from verity_vllm.program.frontend.rules import vllm_bindings as VB
from verity_vllm.program.frontend.rules import vllm_iface as _vi  # noqa: F401  (defines the vllm_iface stand-ins)
from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset
from verity_vllm.program.frontend.rules.vllm_moe import MOE_GEMM_PINS
from verity_vllm.program.frontend.rules.vllm_moe_fp8 import MOE_FP8_CONSTEXPRS, wgmma_tile
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.kernels.fp8_moe_rows import moe_expert_fp8_block_row, silu_mul_fp8_block_quant_rows
from verity_vllm.program.kernels.serve3_reference import materialize
from verity_vllm.program.registry import moe as MOE

F8 = torch.float8_e4m3fn
PROF = dict(mode="profile", profile="b1-eager-v3")
H200 = TargetProfile(compute_capability=(9, 0), num_sms=132, batch_invariant=False, moe_backend="triton", fp8_block_gemm="cutlass")
DOT = "HopperE4m3QgmmaDot32_v1"
#: GLM-5.3's tuned H200 tile from M = 8 (BLOCK_SIZE_M 16 under SWAP_AB) and its block-FP8 numeric constexprs
TUNED = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32}
E, M, TOPK, H, I, G, RSF = 8, 2, 2, 128, 128, 128, 2.5
NV = M * TOPK


@pytest.fixture
def on_target():
    saved = dict(VB._OBSERVED)
    yield VB.observe_target_profile
    VB._OBSERVED.clear()
    VB._OBSERVED.update(saved)


def _e4m3(rng, *shape) -> np.ndarray:
    b = rng.integers(0, 256, shape).astype(np.uint8)
    b[(b & 0x7F) == 0x7F] ^= 1
    return b


def _f32(rng, lo, hi, *shape) -> np.ndarray:
    return np.exp(rng.uniform(np.log(lo), np.log(hi), shape)).astype(np.float32)


def _bf16(rng, *shape) -> torch.Tensor:
    return torch.from_numpy(rng.normal(0.0, 2.0, shape).astype(np.float32)).to(torch.bfloat16)


def _words(t: torch.Tensor) -> list[int]:
    if t.dtype == F8:
        return t.contiguous().view(torch.uint8).numpy().reshape(-1).astype(np.uint8).tolist()
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy().reshape(-1).astype(np.uint16).tolist()
    return t.contiguous().view(torch.int32).numpy().reshape(-1).astype(np.uint32).tolist()


def _derive(m: nn.Module, inputs):
    return derive_report(m, inputs, **PROF, name=type(m).__name__, rules=vllm_ruleset())


def _hint(rep) -> str:
    assert rep.program is None and rep.unsupported is not None
    return rep.unsupported.hint


def _run(rep, m: nn.Module, inputs) -> list[int]:
    params = dict(m.named_parameters())
    weights = [v for w in rep.provenance["weights"] for v in _words(params[w["name"]])]
    t = materialize(rep.program, [_words(x) for x in inputs] + ([weights] if rep.provenance["weights"] else []))
    return [int(t[gid]) for gid in rep.program.circuit.root_outputs()]


def _ids(rep) -> list[str]:
    return [n.fn.id for n in rep.program.fn.body.nodes]


# -- the router ------------------------------------------------------------------------------------------------------------


class _Router(nn.Module):
    def __init__(self, E: int, seed: int = 3, **call) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.bias = nn.Parameter(torch.from_numpy(rng.normal(0.0, 0.05, E).astype(np.float32)), requires_grad=False)
        self.call = {"n_group": 1, "topk_group": 1, "topk": TOPK, "renormalize": True, "scale": RSF, "scoring": 1, **call}

    def forward(self, logits):
        c = self.call
        return torch.ops.vllm_iface.grouped_topk(logits, c["n_group"], c["topk_group"], c["topk"], c["renormalize"], c["scale"], self.bias,
                                                 c["scoring"])


def _logits(E: int, rows: int = M, seed: int = 11) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(seed).normal(0.0, 1.5, (rows, E)).astype(np.float32))


@pytest.mark.parametrize("experts,kernel", [(8, "warp"), (256, "block")])
def test_grouped_topk_derives_the_sigmoid_router_per_row(experts, kernel, on_target):
    on_target(H200)
    m, x = _Router(experts), _logits(experts)
    rep = _derive(m, (x,))
    assert rep.program is not None, rep.unsupported
    fn = bind(MOE.MoeRouterSigmoidBias, E=experts, TOPK=TOPK, SCALE=RSF, KERNEL=kernel)
    assert _ids(rep) == [fn.id] * M
    got = _run(rep, m, (x,))
    bias = _words(m.bias)
    want_w, want_i = [], []
    for r in range(M):
        out = list(evaluate(fn, _words(x[r]), bias))
        want_w += out[:TOPK]
        want_i += out[TOPK:]
    assert got == want_w + want_i
    assert rep.provenance["weight_roles"]["bias"] == {"rule": "GroupedTopk[vllm-moe]_v1", "role": "router.e_score_correction_bias", "E": experts}


@pytest.mark.parametrize("call,needle", [
    ({"n_group": 2}, "numerics_unregistered: n_group=2"),
    ({"scoring": 0}, "scoring_func=0"),
    ({"renormalize": False}, "renormalize=False"),
    ({"topk": 16}, "numerics_unregistered: the grouped_topk launch at E=896, top 16 is not pinned"),
], ids=lambda v: v if isinstance(v, str) else "-".join(v))
def test_each_router_pin_refuses_by_name(call, needle, on_target):
    on_target(H200)
    experts = 896 if call.get("topk") == 16 else E
    hint = _hint(_derive(_Router(experts, **call), (_logits(experts),)))
    assert "GroupedTopk[vllm-moe]_v1" in hint and needle in hint, hint


# -- the fused activation quant ----------------------------------------------------------------------------------------------


class _ActQuant(nn.Module):
    def __init__(self, group: int = G, transposed: bool = False, ub: bool = False) -> None:
        super().__init__()
        self.group, self.transposed = group, transposed
        self.ub = nn.Parameter(torch.ones(1), requires_grad=False) if ub else None

    def forward(self, gu):
        rows, n = gu.shape
        q = torch.empty(rows, n // 2, dtype=F8)
        s = torch.empty(rows, n // 2 // self.group, dtype=torch.float32)
        torch.ops.vllm_iface.silu_and_mul_per_block_quant(q, gu, s, self.group, self.ub, self.transposed)
        return q, s


def test_silu_and_mul_per_block_quant_derives_one_row_form_per_row_equal_to_the_twin(on_target):
    on_target(H200)
    gu = _bf16(np.random.default_rng(4), NV, 2 * I)
    rep = _derive(_ActQuant(), (gu,))
    assert rep.program is not None, rep.unsupported
    assert _ids(rep) == [f"SiluMulFp8BlockQuant_v1{{I={I},G={G}}}"] * NV
    got = _run(rep, _ActQuant(), (gu,))
    q, s = silu_mul_fp8_block_quant_rows(np.array(_words(gu), dtype=np.uint16).reshape(NV, 2 * I), G=G)
    assert got == q.reshape(-1).tolist() + s.reshape(-1).tolist()


@pytest.mark.parametrize("kw,needle", [
    ({"group": 48}, "numerics_unregistered: group_size 48 is not a power of two dividing I=128"),
    ({"transposed": True}, "numerics_unregistered: is_scale_transposed=True"),
    ({"ub": True}, "numerics_unregistered: scale_ub is set"),
], ids=lambda v: v if isinstance(v, str) else "-".join(v))
def test_each_activation_pin_refuses_by_name(kw, needle, on_target):
    on_target(H200)
    hint = _hint(_derive(_ActQuant(**kw), (_bf16(np.random.default_rng(4), NV, 2 * I),)))
    assert "SiluAndMulPerBlockQuant[vllm-moe]_v1" in hint and needle in hint, hint


# -- the expert launches on a tuned Hopper tile, and the chain ---------------------------------------------------------------


def _meta(routed: bool, K: int, N: int, tiles: dict, warps: int = 4, stages: int = 3, **over) -> str:
    NB, KG = -(-N // G), K // G
    sc = {"N": N, "K": K, "EM": NV * 64, "num_valid_tokens": NV, "stride_am": K, "stride_ak": 1, "stride_be": N * K, "stride_bk": 1, "stride_bn": K,
          "stride_cm": N, "stride_cn": 1, "stride_asm": KG, "stride_ask": 1, "stride_bse": NB * KG, "stride_bsk": 1, "stride_bsn": KG,
          "stride_bbe": 0, "stride_bbn": 0, "group_n": G, "group_k": G, "naive_block_assignment": True, "MUL_ROUTED_WEIGHT": routed,
          "top_k": 1 if routed else TOPK, "b_bias_ptr": None, **MOE_FP8_CONSTEXPRS, **tiles,
          "SWAP_AB": tiles["BLOCK_SIZE_M"] < 64 and tiles["BLOCK_SIZE_N"] >= 64}
    sc.update(over)
    names = ["a_ptr", "b_ptr", "c_ptr", "a_scale_ptr", "b_scale_ptr", "expert_ids_ptr"] + (["topk_weights_ptr"] if routed else [])
    return json.dumps({"kernel": MOE_GEMM_PINS["kernel"], "src_sha256": MOE_GEMM_PINS["src_sha256"], "triton": "test", "grid": [1], "scalars": sc,
                       "tensor_args": names, "constexprs": [], "tensor_meta": [None] * len(names),
                       "geometry": {"num_warps": warps, "num_stages": stages}})


def _expert_stack(rng, N: int, K: int):
    w = nn.Parameter(torch.from_numpy(_e4m3(rng, E, N, K)).view(F8), requires_grad=False)
    ws = nn.Parameter(torch.from_numpy(_f32(rng, 1e-4, 1e-2, E, -(-N // G), K // G)), requires_grad=False)
    return w, ws


class _Up(nn.Module):
    """The w13 launch alone, the pair's expert ids an input (naive schedule)."""

    def __init__(self, tiles: dict = TUNED, **over) -> None:
        super().__init__()
        self.w13, self.w13_s = _expert_stack(np.random.default_rng(5), 2 * I, H)
        self.meta = _meta(False, H, 2 * I, tiles, **over)

    def forward(self, xq, sx, ids):
        c = torch.empty(M, TOPK, 2 * I, dtype=torch.bfloat16)
        torch.ops.verity_cba.triton_launch(MOE_GEMM_PINS["kernel"], [xq, self.w13, c, sx, self.w13_s, ids], self.meta)
        return c


class _Experts(nn.Module):
    """GLM-5.3's routed experts after the input quant, as `TritonExperts.apply` launches them: the router's (weights, ids) -> w13 ->
    `silu_and_mul_per_block_quant` -> w2 with the routed weights multiplied in."""

    def __init__(self, tiles: dict = TUNED) -> None:
        super().__init__()
        rng = np.random.default_rng(5)
        self.bias = nn.Parameter(torch.from_numpy(rng.normal(0.0, 0.05, E).astype(np.float32)), requires_grad=False)
        self.w13, self.w13_s = _expert_stack(rng, 2 * I, H)
        self.w2, self.w2_s = _expert_stack(rng, H, I)
        self.up, self.down = _meta(False, H, 2 * I, tiles), _meta(True, I, H, tiles)

    def forward(self, logits, xq, sx):
        g, ids = torch.ops.vllm_iface.grouped_topk(logits, 1, 1, TOPK, True, RSF, self.bias, 1)
        c1 = torch.empty(M, TOPK, 2 * I, dtype=torch.bfloat16)
        torch.ops.verity_cba.triton_launch(MOE_GEMM_PINS["kernel"], [xq, self.w13, c1, sx, self.w13_s, ids], self.up)
        q = torch.empty(NV, I, dtype=F8)
        s = torch.empty(NV, I // G, dtype=torch.float32)
        torch.ops.vllm_iface.silu_and_mul_per_block_quant(q, c1.view(-1, 2 * I), s, G, None, False)
        c3 = torch.empty(M, TOPK, H, dtype=torch.bfloat16)
        torch.ops.verity_cba.triton_launch(MOE_GEMM_PINS["kernel"], [q, self.w2, c3, s, self.w2_s, ids, g], self.down)
        return c3


def _activations(seed: int = 9):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(_e4m3(rng, M, H)).view(F8), torch.from_numpy(_f32(rng, 1e-3, 1e-1, M, H // G))


def _np(t: torch.Tensor) -> np.ndarray:
    return t.view(torch.uint8).numpy() if t.dtype == F8 else t.numpy().view(np.uint32)


def test_a_tuned_hopper_launch_binds_its_tile_and_the_hopper_step(on_target):
    on_target(H200)
    xq, sx = _activations()
    ids = torch.from_numpy(np.stack([np.random.default_rng(r).permutation(E)[:TOPK] for r in range(M)]).astype(np.int32))
    m = _Up()
    rep = _derive(m, (xq, sx, ids))
    assert rep.program is not None, rep.unsupported
    assert _ids(rep) == [f"MoeExpertFp8Gemm_v1{{E={E},K={H},N={2 * I},G={G},KB=128,SWAP=True,DOT={DOT}}}"] * NV
    got = [w & 0xFFFF for w in _run(rep, m, (xq, sx, ids))]
    w, ws = _np(m.w13), _np(m.w13_s)
    want = []
    for p in range(NV):
        e, t = int(ids.reshape(-1)[p]), p // TOPK
        words, nonfin = moe_expert_fp8_block_row(_np(xq)[t], _np(sx)[t], w[e], ws[e], G=G, KB=128, swap=True)
        assert not nonfin.any()
        want += words.tolist()
    assert got == want


@pytest.mark.parametrize("tiles,over,needle", [
    ({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1}, {},
     "the launch's tl.dot (BLOCK_SIZE_M 8 x BLOCK_SIZE_N 32, SWAP_AB False, 4 warps) is no wgmma tile"),
    (TUNED, {"SWAP_AB": False}, "SWAP_AB=False is not enable_swap_ab(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64) = True on cc 9.0"),
    ({**TUNED, "BLOCK_SIZE_K": 48}, {}, "BLOCK_SIZE_K=48 is not a multiple of the k32 step"),
], ids=["mma-sync-tile", "swap", "k-tile"])
def test_each_tuned_tile_pin_refuses_by_name(tiles, over, needle, on_target):
    on_target(H200)
    xq, sx = _activations()
    ids = torch.zeros(M, TOPK, dtype=torch.int32)
    hint = _hint(_derive(_Up(tiles, **over), (xq, sx, ids)))
    assert f"FusedMoeKernelFp8[vllm-moe]_v1: numerics_unregistered: {needle}" in hint, hint


def test_a_tuned_launch_off_hopper_refuses_by_name(on_target):
    on_target(TargetProfile(compute_capability=(12, 0), num_sms=188, batch_invariant=False, moe_backend="triton"))
    xq, sx = _activations()
    hint = _hint(_derive(_Up(), (xq, sx, torch.zeros(M, TOPK, dtype=torch.int32))))
    assert "modelled on cc 9.0 only (wgmma), not on cc 12.0" in hint, hint


def test_the_wgmma_tile_rule_matches_the_tuned_h200_entries():
    """GLM-5.3's TP8 shard on the H200 (`E=256,N=256,...NVIDIA_H200,...[128,128].json`): the M <= 3 entries are no wgmma tile, and
    every other is, with SWAP_AB below BLOCK_SIZE_M 64."""
    entries = {1: (8, 32), 2: (8, 32), 4: (8, 64), 8: (16, 64), 16: (16, 128), 512: (16, 128), 1024: (64, 128), 4096: (64, 128)}
    got = {m: wgmma_tile({"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "num_warps": 4}, bm < 64 and bn >= 64) for m, (bm, bn) in entries.items()}
    assert got == {1: False, 2: False, 4: True, 8: True, 16: True, 512: True, 1024: True, 4096: True}


def test_the_routed_experts_chain_derives_in_one_program_equal_to_the_twins(on_target):
    on_target(H200)
    x = _logits(E)
    xq, sx = _activations()
    m = _Experts()
    rep = _derive(m, (x, xq, sx))
    assert rep.program is not None, rep.unsupported
    router = bind(MOE.MoeRouterSigmoidBias, E=E, TOPK=TOPK, SCALE=RSF, KERNEL="warp")
    assert _ids(rep) == ([router.id] * M + [f"MoeExpertFp8Gemm_v1{{E={E},K={H},N={2 * I},G={G},KB=128,SWAP=True,DOT={DOT}}}"] * NV
                         + [f"SiluMulFp8BlockQuant_v1{{I={I},G={G}}}"] * NV
                         + [f"MoeExpertFp8GemmW_v1{{E={E},K={I},N={H},G={G},KB=128,SWAP=True,DOT={DOT}}}"] * NV)
    got = [w & 0xFFFF for w in _run(rep, m, (x, xq, sx))]
    bias = _words(m.bias)
    routes = [list(evaluate(router, _words(x[r]), bias)) for r in range(M)]
    gw = [int(w) for r in routes for w in r[:TOPK]]
    ids = [int(i) for r in routes for i in r[TOPK:]]
    w13, w13s, w2, w2s = (_np(t) for t in (m.w13, m.w13_s, m.w2, m.w2_s))
    up = np.stack([moe_expert_fp8_block_row(_np(xq)[p // TOPK], _np(sx)[p // TOPK], w13[ids[p]], w13s[ids[p]], G=G, KB=128, swap=True)[0]
                   for p in range(NV)])
    q, s = silu_mul_fp8_block_quant_rows(up.astype(np.uint16), G=G)
    want = []
    for p in range(NV):
        words, nonfin = moe_expert_fp8_block_row(q[p], s[p], w2[ids[p]], w2s[ids[p]], G=G, KB=128, swap=True, g=gw[p])
        assert not nonfin.any()
        want += words.tolist()
    assert got == want
