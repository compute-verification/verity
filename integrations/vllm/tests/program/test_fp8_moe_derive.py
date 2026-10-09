"""The block-FP8 `fused_moe_kernel` launches (`rules/vllm_moe_fp8.FusedMoeKernelFp8Rule`): on cc 12.0 each launch derives to one
`MoeExpertFp8Gemm(W)_v1{E,K,N,G,KB,SWAP,DOT=BlackwellE4m3QmmaDot32_v1}` Call per (token, slot) pair whose words are the twin's on the
routed expert's rows, the expert stack and its block scales are weight Inputs with their roles, and every pin refuses by name."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules import triton_iface as _ti  # noqa: F401  (defines verity_cba::triton_launch)
from verity_vllm.program.frontend.rules import vllm_bindings as VB
from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset
from verity_vllm.program.frontend.rules.vllm_moe import MOE_GEMM_PINS
from verity_vllm.program.frontend.rules.vllm_moe_fp8 import MOE_FP8_CONSTEXPRS
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.kernels.fp8_moe_rows import moe_expert_fp8_block_row
from verity_vllm.program.kernels.serve3_reference import materialize

E, M, TOPK, K, N, G = 3, 2, 2, 64, 40, 32
NB, KG, NV = -(-N // G), K // G, M * TOPK
F8 = torch.float8_e4m3fn
PROF = dict(mode="profile", profile="b1-eager-v3")
SM120 = TargetProfile(compute_capability=(12, 0), num_sms=188, moe_backend="triton")
FAMILY = {False: "MoeExpertFp8Gemm_v1", True: "MoeExpertFp8GemmW_v1"}
DOT = "BlackwellE4m3QmmaDot32_v1"


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


def _meta(routed: bool, **over) -> str:
    sc = {"N": N, "K": K, "EM": NV * 64, "num_valid_tokens": NV, "stride_am": K, "stride_ak": 1, "stride_be": N * K, "stride_bk": 1, "stride_bn": K,
          "stride_cm": N, "stride_cn": 1, "stride_asm": KG, "stride_ask": 1, "stride_bse": NB * KG, "stride_bsk": 1, "stride_bsn": KG,
          "stride_bbe": 0, "stride_bbn": 0, "group_n": G, "group_k": G, "naive_block_assignment": True, "MUL_ROUTED_WEIGHT": routed,
          "top_k": 1 if routed else TOPK, "b_bias_ptr": None, **MOE_FP8_CONSTEXPRS, **MOE_GEMM_PINS["tiles"]}
    sc.update(over)
    names = ["a_ptr", "b_ptr", "c_ptr", "a_scale_ptr", "b_scale_ptr", "expert_ids_ptr"] + (["topk_weights_ptr"] if routed else [])
    return json.dumps({"kernel": MOE_GEMM_PINS["kernel"], "src_sha256": MOE_GEMM_PINS["src_sha256"], "triton": "test", "grid": [1], "scalars": sc,
                       "tensor_args": names, "constexprs": [], "tensor_meta": [None] * len(names),
                       "geometry": {"num_warps": MOE_GEMM_PINS["num_warps"], "num_stages": MOE_GEMM_PINS["num_stages"]}})


class _Launch(nn.Module):
    """One `invoke_fused_moe_triton_kernel` launch on the naive schedule: the w13 launch (top_k = TOPK, the token rows) or, `routed`,
    the w2 one (top_k = 1, one quantised row per pair, the routed weights multiplied in)."""

    def __init__(self, routed: bool, ws_blocks: int = NB, ws_k: int = KG, seed: int = 5, **over) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.routed, self.meta = routed, _meta(routed, **over)
        self.w = nn.Parameter(torch.from_numpy(_e4m3(rng, E, N, K)).view(F8), requires_grad=False)
        self.ws = nn.Parameter(torch.from_numpy(_f32(rng, 1e-4, 1e-2, E, ws_blocks, ws_k)), requires_grad=False)

    def forward(self, xq, sx, ids, g=None):
        c = torch.empty(M, TOPK, N, dtype=torch.bfloat16)
        torch.ops.verity_cba.triton_launch(MOE_GEMM_PINS["kernel"], [xq, self.w, c, sx, self.ws, ids] + ([g] if self.routed else []), self.meta)
        return c


def _inputs(routed: bool, seed: int = 9):
    rng = np.random.default_rng(seed)
    rows = NV if routed else M
    ids = np.stack([rng.permutation(E)[:TOPK] for _ in range(M)]).astype(np.int32)
    out = [torch.from_numpy(_e4m3(rng, rows, K)).view(F8), torch.from_numpy(_f32(rng, 1e-3, 1e-1, rows, KG)), torch.from_numpy(ids)]
    if routed:
        out.append(torch.from_numpy(_f32(rng, 0.05, 0.9, M, TOPK)))
    return tuple(out)


def _words(t: torch.Tensor) -> list[int]:
    v = t.contiguous().view(torch.uint8) if t.dtype == F8 else t.contiguous().view(torch.int32)
    return v.numpy().astype(np.uint32 if t.dtype != F8 else np.uint8).reshape(-1).tolist()


def _derive(m: _Launch, inputs):
    return derive_report(m, inputs, **PROF, name=type(m).__name__, rules=vllm_ruleset())


def _hint(rep) -> str:
    assert rep.program is None and rep.unsupported is not None
    return rep.unsupported.hint


@pytest.mark.parametrize("routed", [False, True], ids=["w13", "w2"])
def test_a_block_fp8_launch_derives_one_pair_form_per_pair_equal_to_the_twin(routed, on_target):
    on_target(SM120)
    m, inputs = _Launch(routed), _inputs(routed)
    rep = _derive(m, inputs)
    assert rep.program is not None, rep.unsupported
    want_id = f"{FAMILY[routed]}{{E={E},K={K},N={N},G={G},KB=32,SWAP=False,DOT={DOT}}}"
    assert [n.fn.id for n in rep.program.fn.body.nodes] == [want_id] * NV
    params = dict(m.named_parameters())
    weights = [v for w in rep.provenance["weights"] for v in _words(params[w["name"]])]
    t = materialize(rep.program, [_words(x) for x in inputs] + [weights])
    got = [int(t[gid]) & 0xFFFF for gid in rep.program.circuit.root_outputs()]
    xq, sx, ids = (x.view(torch.uint8).numpy() if x.dtype == F8 else x.numpy() for x in inputs[:3])
    w, ws = m.w.view(torch.uint8).numpy(), m.ws.numpy().view(np.uint32)
    want = []
    for p in range(NV):
        e, row = int(ids.reshape(-1)[p]), p if routed else p // TOPK
        g = int(inputs[3].numpy().reshape(-1)[p:p + 1].view(np.uint32)[0]) if routed else None
        words, nonfin = moe_expert_fp8_block_row(xq[row], sx[row].view(np.uint32), w[e], ws[e], G=G, KB=32, g=g, dot="blackwell")
        assert not nonfin.any()
        want += words.tolist()
    assert got == want
    roles = rep.provenance["weight_roles"]
    assert roles["w"] == {"rule": "FusedMoeKernelFp8[vllm-moe]_v1", "role": "moe.expert_weight", "E": E, "K": K, "N": N}
    assert roles["ws"] == {"rule": "FusedMoeKernelFp8[vllm-moe]_v1", "role": "moe.expert_weight_scale", "E": E, "K": K, "N": N, "G": G}


@pytest.mark.parametrize("over,needle", [
    ({"use_fp8_w8a8": False}, "constexpr use_fp8_w8a8=False"),
    ({"SWAP_AB": True}, "constexpr SWAP_AB=True"),
    ({"per_channel_quant": True}, "constexpr per_channel_quant=True"),
    ({"USE_TD": True}, "constexpr USE_TD=True"),
    ({"group_n": 64}, "block_shape (group_n, group_k) = (64, 32)"),
    ({"BLOCK_SIZE_K": 64}, "tiling BLOCK_SIZE_K=64"),
    ({"stride_bsk": KG, "stride_bsn": 1}, "strides are not row-major: b_scale"),
], ids=lambda v: v if isinstance(v, str) else "-".join(v))
def test_each_launch_pin_refuses_by_name(over, needle, on_target):
    on_target(SM120)
    hint = _hint(_derive(_Launch(False, **over), _inputs(False)))
    assert f"FusedMoeKernelFp8[vllm-moe]_v1: numerics_unregistered: {needle}" in hint or f"FusedMoeKernelFp8[vllm-moe]_v1: {needle}" in hint, hint


def test_scales_on_another_block_grid_refuse(on_target):
    on_target(SM120)
    hint = _hint(_derive(_Launch(False, ws_blocks=-(-N // 128), ws_k=1), _inputs(False)))
    assert "FusedMoeKernelFp8[vllm-moe]_v1: scale shapes" in hint and f"want ({E}, {NB}, {KG})" in hint, hint


@pytest.mark.parametrize("cc", [(9, 0), (8, 9)])
def test_a_target_without_an_e4m3_step_refuses_by_name(cc, on_target):
    on_target(TargetProfile(compute_capability=cc))
    hint = _hint(_derive(_Launch(True), _inputs(True)))
    assert "FusedMoeKernelFp8[vllm-moe]_v1: numerics_unregistered: block-FP8 fused-MoE experts on" in hint and "no registered e4m3 step" in hint, hint


def test_the_padded_construction_does_not_own_the_fp8_launch(on_target):
    on_target(SM120.with_overrides(moe_construction="padded"))
    hint = _hint(_derive(_Launch(False), _inputs(False)))
    assert "FusedMoeKernelFp8[vllm-moe]_v1: moe_construction='padded'" in hint, hint


def test_the_dot_roles_bind_the_pair_forms_under_a_block_and_keep_their_notes():
    """The block-FP8 forms are optional statics of the `_dot` roles: the roles and their notes, and with them the vocabulary's
    version, are unchanged, and without `G` the roles bind what they always did."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.prims import HopperBF16WgmmaDot16
    from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3 as V
    from verity_vllm.program.registry import moe
    from verity_vllm.program.frontend.targets import FP8_DOTS
    dot = FP8_DOTS["BlackwellE4m3QmmaDot32"]
    assert V.kind("moe_expert_gemm_dot", E=E, K=K, N=N, DOT=dot, G=G, KB=32, SWAP=False).id == \
        f"MoeExpertFp8Gemm_v1{{E={E},K={K},N={N},G={G},KB=32,SWAP=False,DOT={DOT}}}"
    assert V.kind("moe_expert_gemm_w_dot", E=E, K=K, N=N, DOT=dot, G=G, KB=32, SWAP=False).id.startswith("MoeExpertFp8GemmW_v1{")
    assert V.kind("moe_expert_gemm_dot", E=E, K=K, N=N, DOT=dot, G=G) is None
    assert V.kind("moe_expert_gemm_dot", E=E, K=K, N=N, DOT=HopperBF16WgmmaDot16).id == \
        bind(moe.MoeExpertGemmV2, E=E, K=K, N=N, DOT=HopperBF16WgmmaDot16).id
    assert "MoeExpertFp8" not in json.dumps(V.to_json())
