"""GLM-5.3's routed MoE as the Build derives it at vLLM's sm_90 defaults (DeepGEMM's ue8m0 quant on): the input's
`Fp8GroupQuantUe8m0_v1` per token, `MoeRouterSigmoidBias_v1`, the w13 `MoeExpertFp8Gemm_v1` per pair, `SiluMul_v2`, the activation's
`Fp8GroupQuantUe8m0_v1` per pair, the w2 `MoeExpertFp8GemmW_v1`, `MoeSum_v1`.  The manifest sizes its `moe_block_stream` planes, the
replay index addresses each row at its plane, and the MoE source binds the same planes (`MOE_FP8_PLANES`): the three agree, and a row
that reads another pair's quant is named, never read."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from tests.program import test_sigmoid_moe_fp8_derive as S
from verity_vllm.check.replay import index as IX
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.manifest import format as fmt
from verity_vllm.query.program_view import from_instances
from verity_vllm.query.required import request_manifest

E, M, TOPK, H, I, G, NV = S.E, S.M, S.TOPK, S.H, S.I, S.G, S.NV
X = "model.model.layers.3.mlp.experts"
EPS = '{"f64":"0x1.0c6f7a0b5ed8dp-20"}'


class _Served(nn.Module):
    """`TritonExperts.apply` on GLM-5.3-FP8 with vLLM's default kernels, from the bf16 hidden states (`S._ExpertsDefault` with the
    unfused activation: `_C.silu_and_mul`, then the ue8m0 group quant of its output)."""

    def __init__(self) -> None:
        super().__init__()
        rng = np.random.default_rng(5)
        self.bias = nn.Parameter(torch.from_numpy(rng.normal(0.0, 0.05, E).astype(np.float32)), requires_grad=False)
        self.w13, self.w13_s = S._expert_stack(rng, 2 * I, H)
        self.w2, self.w2_s = S._expert_stack(rng, H, I)
        self.up, self.down = S._meta(False, H, 2 * I, S.TUNED), S._meta(True, I, H, S.TUNED)

    @staticmethod
    def _quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, s = torch.empty(*x.shape, dtype=S.F8), torch.empty(x.shape[0], x.shape[1] // G, dtype=torch.float32)
        torch.ops.vllm_iface.per_token_group_fp8_quant(x, q, s, G, 1e-10, -448.0, 448.0, True, False, False)
        return q, s

    def forward(self, logits, x):
        xq, sx = self._quant(x)
        g, ids = torch.ops.vllm_iface.grouped_topk(logits, 1, 1, TOPK, True, S.RSF, self.bias, 1)
        c1 = torch.empty(M, TOPK, 2 * I, dtype=torch.bfloat16)
        torch.ops.verity_cba.triton_launch(S.MOE_GEMM_PINS["kernel"], [xq, self.w13, c1, sx, self.w13_s, ids], self.up)
        c2 = torch.empty(NV, I, dtype=torch.bfloat16)
        torch.ops.vllm_iface.silu_and_mul(c2, c1.view(-1, 2 * I))
        q, s = self._quant(c2)
        c3 = torch.empty(M, TOPK, H, dtype=torch.bfloat16)
        torch.ops.verity_cba.triton_launch(S.MOE_GEMM_PINS["kernel"], [q, self.w2, c3, s, self.w2_s, ids, g], self.down)
        return c3


@pytest.fixture(scope="module")
def instances() -> dict:
    """The derived chain's instance rows under `model.model.layers.3.mlp.experts`, with the rows the CPU derive cannot emit appended:
    `MoeSum_v1` per token (`_moe_C.moe_sum` needs vLLM), the next layer's fused norm reading it, the logits and the token select."""
    from verity_vllm.pipeline.build import _instance_sequence
    from verity_vllm.program.compact import expanded_file_args
    from verity_vllm.program.frontend.rules import vllm_bindings as VB

    class Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.model = nn.Module()
            self.model.model.layers = nn.ModuleList([nn.Identity()] * 3 + [nn.Module()])
            self.model.model.layers[3].mlp = nn.Module()
            self.model.model.layers[3].mlp.experts = _Served()

        def forward(self, logits, x):
            return self.model.model.layers[3].mlp.experts(logits, x)

    saved = dict(VB._OBSERVED)
    VB.observe_target_profile(S.H200)
    try:
        rep = S._derive(Root(), (S._logits(E), S._bf16(np.random.default_rng(7), M, H)))
    finally:
        VB._OBSERVED.clear()
        VB._OBSERVED.update(saved)
    assert rep.program is not None, rep.unsupported
    rows = [dict(r, args=expanded_file_args(r["args"])) for r in _instance_sequence(rep.program)]
    downs = [r["i"] for r in rows if r["spec"].startswith("MoeExpertFp8GemmW")]

    def add(spec: str, name: str, args: list) -> int:
        rows.append({"i": len(rows), "spec": spec, "form": "call", "name": name, "args": args})
        return len(rows) - 1

    sums = [add(f"MoeSum_v1{{TOPK={TOPK},H={H}}}", f"{X}/moe_sum[{t}]", [[["n", downs[t * TOPK + k], 0, H] for k in range(TOPK)]])
            for t in range(M)]
    norms = [add(f"RMSNormFusedCuda_v2{{N={H},EPS={EPS}}}", f"model.model.layers.4.input_layernorm/fused_add_rms_norm[{t}]",
                 [[["n", s, 0, H]], [["p", 1, t * H, (t + 1) * H]], [["p", 1, 0, H]]]) for t, s in enumerate(sums)]
    lg = add(f"Gemm_v1{{K={H},N=8}}", "model.lm_head/linear[0]", [[["n", norms[-1], 0, H]], [["p", 1, 0, 8 * H]]])
    add("TokenSelect_v1{V=8}", "argmax[0]", [[["n", lg, 0, 8]]])
    params = [[n, str(t)] for n, t in rep.program.fn.body.params]
    return {"schema": "cb-a/instance-sequence/v1", "digest": "g" * 64, "params": params, "rows": rows}


def _families(inst: dict) -> list[str]:
    return [fmt.spec_family(r["spec"]) for r in inst["rows"]]


def test_the_derived_chain_is_the_served_paths_rows(instances):
    fams = _families(instances)
    assert fams[:4 * NV + 2 * M] == (["Fp8GroupQuantUe8m0_v1"] * M + ["MoeRouterSigmoidBias_v1"] * M + ["MoeExpertFp8Gemm_v1"] * NV
                                     + ["SiluMul_v2"] * NV + ["Fp8GroupQuantUe8m0_v1"] * NV + ["MoeExpertFp8GemmW_v1"] * NV)
    assert fams.count("MoeSum_v1") == M


def test_the_manifest_requires_the_ten_planes_the_source_binds_at_their_sizes(instances):
    g = fmt.moe_geometry(instances["rows"])[X]
    assert (g["M"], g["TOPK"], g["H"], g["I"], g["N_up"], g["G"], g["fp8"]) == (M, TOPK, H, I, 2 * I, G, True)
    assert (g["in_quant_rows"], g["act_quant_rows"], g["act_rows"]) == (M, NV, NV)     # H == I here: the input's quant is the first M
    want = {"router_w": M * TOPK, "router_ids": M * TOPK, "in_q": M * H, "in_scale": M * H // G, "up": NV * 2 * I, "act": NV * I,
            "act_q": NV * I, "act_scale": NV * I // G, "down": NV * H, "sum": M * H}
    assert [p for p, _n, _d in fmt.moe_identities(g)] == list(fmt.MOE_FP8_PLANES) == list(want)
    P = from_instances(instances)
    res = {"program": {"digest": "d0", "function": "f"}, "target": {"declared": {"compute_capability": [9, 0]}}}
    m = request_manifest(P, Correspondence(P), res, request_id="r0", rank=0, ranks=1).manifest
    assert m["complete"] and not m["unmodelled"], m["unmodelled"]
    planes = {x["output_member"]: x for x in m["identities"] if x["family"] == "moe_block_stream"}
    assert {k: x["element_range"][1] for k, x in planes.items()} == want and {x["op_path"] for x in planes.values()} == {"moe/L3"}
    assert {k: x["dtype"] for k, x in planes.items() if k.endswith(("_q", "_scale"))} == {"in_q": "w8", "in_scale": "f32", "act_q": "w8",
                                                                                         "act_scale": "f32"}
    assert not [x for x in m["identities"] if x["family"] == "call_boundaries"], "the block's rows are committed as its planes"
    out = [x for x in m["identities"] if x["family"] == "instance_outputs" and x["op_path"] == X]
    assert [(x["output_member"], x["spec"], x["element_range"][1]) for x in out] == [("out", "MoeSum_v1", M * H)]


def test_the_replay_index_addresses_every_row_at_the_plane_the_manifest_sizes(instances):
    pi = IX.ProgramIndex(instances)
    seen: dict[str, int] = {}
    kinds: list[str] = []
    for r in instances["rows"]:
        d = pi.describe(r)
        if d is None or d["op_path"] != "moe/L3":
            continue
        assert "plane_error" not in d and d["members"], (r["name"], d.get("plane_error"))
        kinds.append(d["plane_kind"])
        for member, width, _dtype in d["members"]:
            seen[member] = seen.get(member, 0) + width
    assert kinds == ["in_quant"] * M + ["router"] * M + ["up"] * NV + ["act"] * NV + ["act_quant"] * NV + ["down"] * NV
    g = fmt.moe_geometry(instances["rows"])[X]
    sizes = {p: n for p, n, _d in fmt.moe_identities(g)}
    sm = [pi.describe(r) for r in instances["rows"] if r["spec"].startswith("MoeSum")]
    seen["sum"] = sum(c["width"] for d in sm for c in d["copies"])
    assert seen == sizes and all(d["members"] == [("out", H, "bf16")] for d in sm)
    act_q = next(r for r in instances["rows"] if pi.describe(r).get("plane_kind") == "act_quant")
    assert pi.describe(act_q)["members"] == [("act_q", I, "w8"), ("act_scale", I // G, "f32")]


def _rewire(inst: dict, i: int, arg: int, j: int) -> dict:
    """Row i's operand `arg` read from row j (same slice)."""
    out = copy.deepcopy(inst)
    out["rows"][i]["args"][arg][0][1] = j
    return out


@pytest.mark.parametrize("case,needle", [
    ("up reads the other token's input quant", "up row 0 does not read in_quant row 0's (q, scale)"),
    ("down reads another pair's activation quant", "down row 0 does not read act_quant row 0's (q, scale)"),
    ("down reads the input quant", "down row 0 does not read act_quant row 0's (q, scale)"),
    ("activation quant reads another pair's act", "act_quant row 0 does not read act row 0"),
])
def test_a_plane_row_that_reads_another_pairs_quant_is_named(instances, case, needle):
    rows = instances["rows"]
    first = {k: next(r["i"] for r in rows if fmt.spec_family(r["spec"]) == k) for k in ("MoeExpertFp8Gemm_v1", "MoeExpertFp8GemmW_v1",
                                                                                       "SiluMul_v2")}
    in_q = [r["i"] for r in rows if fmt.spec_family(r["spec"]).startswith("Fp8GroupQuant")]
    up, down, act = first["MoeExpertFp8Gemm_v1"], first["MoeExpertFp8GemmW_v1"], first["SiluMul_v2"]
    act_q = in_q[M:]
    inst = {"up reads the other token's input quant": lambda: _rewire(_rewire(instances, up, 0, in_q[1]), up, 1, in_q[1]),
            "down reads another pair's activation quant": lambda: _rewire(_rewire(instances, down, 0, act_q[1]), down, 1, act_q[1]),
            "down reads the input quant": lambda: _rewire(_rewire(instances, down, 0, in_q[0]), down, 1, in_q[0]),
            "activation quant reads another pair's act": lambda: _rewire(instances, act_q[0], 0, act + 1)}[case]()
    pi = IX.ProgramIndex(inst)
    errs = [d["plane_error"] for r in inst["rows"] if (d := pi.describe(r)) and d.get("plane_error")]
    assert any(needle in e for e in errs), errs
    assert all(d["members"] == [] for r in inst["rows"] if (d := pi.describe(r)) and d.get("plane_error"))


def _rows(specs: list[tuple[str, int]], mod: str = X) -> list[dict]:
    out = []
    for spec, n in specs:
        out += [{"spec": spec, "name": f"{mod}/op[{k}]"} for k in range(n)]
    return out


def test_glm53s_shard_geometry_tells_the_two_quants_apart_by_width():
    """At GLM-5.3's TP8 shard (H = 6144, I = 256, E = 256, TOPK = 8, G = 128), one token: the input's quant is the K = H row, the
    activation's the K = I rows; the fused SiLU quant (DeepGEMM's ue8m0 off) leaves no `act` plane; OLMoE's bf16 block keeps six."""
    Hs, Is, Es, Ks, Gs = 6144, 256, 256, 8, 128
    head = [(f"Fp8GroupQuantUe8m0_v1{{K={Hs},G={Gs}}}", 1), (f"MoeRouterSigmoidBias_v1{{E={Es},TOPK={Ks},SCALE=2.5,KERNEL=block}}", 1),
            (f"MoeExpertFp8Gemm_v1{{E={Es},K={Hs},N={2 * Is},G={Gs},KB=128,SWAP=true,DOT=d}}", Ks)]
    tail = [(f"MoeExpertFp8GemmW_v1{{E={Es},K={Is},N={Hs},G={Gs},KB=128,SWAP=true,DOT=d}}", Ks), (f"MoeSum_v1{{TOPK={Ks},H={Hs}}}", 1)]
    served = fmt.moe_geometry(_rows(head + [(f"SiluMul_v2{{I={Is}}}", Ks), (f"Fp8GroupQuantUe8m0_v1{{K={Is},G={Gs}}}", Ks)] + tail))[X]
    assert (served["in_quant_rows"], served["act_quant_rows"], served["act_rows"], served["H"], served["I"]) == (1, Ks, Ks, Hs, Is)
    assert fmt.moe_identities(served) == [
        ("router_w", Ks, "f32"), ("router_ids", Ks, "i32"), ("in_q", Hs, "w8"), ("in_scale", Hs // Gs, "f32"), ("up", Ks * 2 * Is, "bf16"),
        ("act", Ks * Is, "bf16"), ("act_q", Ks * Is, "w8"), ("act_scale", Ks * Is // Gs, "f32"), ("down", Ks * Hs, "bf16"), ("sum", Hs, "bf16")]
    fused = fmt.moe_geometry(_rows(head + [(f"SiluMulFp8BlockQuant_v1{{I={Is},G={Gs}}}", Ks)] + tail))[X]
    assert [p for p, _n, _d in fmt.moe_identities(fused)] == [p for p in fmt.MOE_FP8_PLANES if p != "act"]
    assert fmt.moe_plane_kind("SiluMulFp8BlockQuant_v1", X) == "act_quant" and fmt.moe_plane_kind("Fp8GroupQuantUe8m0_v1", X) == "quant"
    assert fmt.moe_plane_kind("Fp8GroupQuantUe8m0_v1", "model.model.layers.3.mlp.shared_experts.gate_up_proj") is None
    olmoe = fmt.moe_geometry(_rows([("MoeRouterTopK_v1{E=64,TOPK=8,VPT=8}", 1), ("MoeExpertGemm_v1{E=64,K=2048,N=2048}", 8),
                                    ("SiluMul_v1{I=1024}", 8), ("MoeExpertGemmW_v1{E=64,K=1024,N=2048}", 8), ("MoeSum_v1{TOPK=8,H=2048}", 1)]))[X]
    assert [p for p, _n, _d in fmt.moe_identities(olmoe)] == list(fmt.MOE_PLANES) and not olmoe.get("fp8")
