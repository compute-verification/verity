"""The fused-MoE block of a served-compiled Program.  Inductor calls vLLM's `moe_forward` custom op as one opaque extern op; the Build
expands it into the Calls the op runs, the Call boundaries of the eager Program's MoE block (`MoeBlockSource`), each writing a value
the serving process commits from inside the op (`ServedCompiledSource._install_moe`, `c<piece>.moe/<plane>`, token rows):

  * the gate: the runner's `ReplicatedLinear`, `F.linear` -> `at::mm`, a cuBLAS GEMM of the op's hidden states against the gate
    weight, the extern `mm`'s route at (M, H, E) -> `logits` (E bf16);
  * the router, `topk_softmax` (`MoeRouterTopK_v1`, `MoeRouterTopKNorm_v1` under norm_topk_prob) -> `router_w` (TOPK f32), `router_ids`
    (TOPK i32);
  * the w13 launch of `fused_moe_kernel`: each (token, slot) pair's expert GEMM, `MoeExpertGemm` -> `up` (TOPK x 2I bf16, pair-major);
  * `_C.silu_and_mul` on each pair (`SiluMul_v2`) -> `act` (TOPK x I);
  * the w2 launch with the routed weight, `MoeExpertGemmW` -> `down` (TOPK x H);
  * `moe_sum` over the slots (`MoeSum_v1`) -> the op's return, the wrapper's own value.

The expert GEMMs are the bf16 path's `acc += tl.dot(a, b)`, one carried k16 chain, and on cc 9.0 every k16 step is Hopper's
(`_v2{DOT=HopperBF16WgmmaDot16}`) whichever config vLLM launched at M rows (`try_get_optimal_moe_config`, recorded per M by the
source, which must have launched the kernel at M): Triton lowers the dot to wgmma or to MMA v2's `mma.sync m16n8k16`, and on an H100
the latter's HMMA.16816 is `silicon.HOPPER_BF16_M16N8K16`, the wgmma k16 step's parameters, not the Ampere step (`_v1`
matched it on 27 of 1,344 rows), the binding the eager Program shares (`frontend.targets.moe_expert_dot`).
On cc 10.0 Triton lowers it to tcgen05.mma kind::f16 or to `mma.sync m16n8k16`, pinned to the same pipeline (silicon
`sm100.tcgen05.mma.bf16` and `sm100.mma.m16n8k16.bf16`), so the same Definitions bind.  cc 12.0 has neither wgmma nor tcgen05:
Triton lowers it to `mma.sync m16n8k16`, silicon `sm120.mma.m16n8k16.bf16`, the pipeline its cuBLAS GEMMs bind (`served_device`)
and its eager Program's MoE binds (`frontend.targets`' `blackwell_consumer`), so the same Definitions bind there too
(a run matched every MoeExpertGemm_v2 and MoeExpertGemmW_v2 call of Qwen3-30B-A3B's 14 steps on the RTX PRO 6000).
Another compute capability is refused.

The op takes its layer as an opaque argument: the serving process records the layer each piece's op ran (`moe_pieces`), and that
layer's facts (`served_match.moe_facts`) give its shapes and confirm it is the block expanded here.  Anything else (a fused or absent
gate, shared experts, routed scaling, another router or experts kernel, bias, a clamp, another activation, non-bf16 weights) is
refused, as is a piece with two of the op (its planes are named by piece).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from verity.ml.circuits.ops.prims import HopperBF16WgmmaDot16
from verity_vllm.acquire.inductor_templates import Unmodelled, bound
from verity.ml.definitions.registry import act
from verity.ml.definitions.registry import moe as M

PLANES = ("logits", "router_w", "router_ids", "up", "act", "down")
#: the compute capabilities whose fused_moe_kernel lowering `expert_definition` models
CC = ((9, 0), (10, 0), (12, 0))
#: what `served_match.moe_facts` must say of a layer for the expansion to be its computation
REQUIRED = {"gate_held": True, "fused_gate": False, "shared_experts": False, "routed_scaling": False, "softmax_router": True,
            "triton_experts": True, "bias": False, "clamp": False, "silu": True, "bf16": True}


def plan_key(E: int, K: int, N: int, routed: bool) -> str:
    """An expert GEMM's key in a step plan's `moe`, which its call carries (`"moe"`)."""
    return f"{E},{K},{N},{int(routed)}"


def expert_definition(E: int, K: int, N: int, routed: bool) -> dict[str, Any]:
    return bound(M.MoeExpertGemmWV2 if routed else M.MoeExpertGemmV2, E=E, K=K, N=N, DOT=HopperBF16WgmmaDot16)


def expert_plans(keys: set[str], configs: dict[str, dict[str, dict[str, Any]]]) -> Callable[[int], dict[str, Any]]:
    """M -> each expert GEMM's Definition at M rows by its `plan_key`, where the process launched fused_moe_kernel at M."""
    def plan(m: int) -> dict[str, Any]:
        out = {}
        for key in sorted(keys):
            E, K, N, r = map(int, key.split(","))
            if (configs.get(str(m)) or {}).get(str(bool(r))) is None:
                raise Unmodelled(f"no fused_moe_kernel launch at {m} rows (MUL_ROUTED_WEIGHT {bool(r)}) was recorded")
            out[key] = expert_definition(E, K, N, bool(r))
        return out
    return plan


def _block(layer: str, f: dict[str, Any] | None, x: dict[str, Any]) -> tuple[int, int, int, int]:
    """(E, TOPK, H, I) of `layer`, whose facts must be `REQUIRED`'s and whose hidden states are x's rows."""
    if f is None:
        raise Unmodelled(f"{layer}: the serving process recorded no facts of this fused-MoE layer")
    off = {k: f.get(k) for k, want in REQUIRED.items() if f.get(k) is not want}
    if off or f.get("renormalize") not in (True, False):
        raise Unmodelled(f"{layer}: {off or {'renormalize': f.get('renormalize')}}; the expansion models {REQUIRED}")
    (E, H), (E13, I2, H13), (E2, H2, I) = f["gate"]["shape"], f["w13"]["shape"], f["w2"]["shape"]
    if not (E == E13 == E2 == f["E"] and H == H13 == H2 == x["width"] and I2 == 2 * I) or not all(f[k]["weight"] for k in ("gate", "w13", "w2")):
        raise Unmodelled(f"{layer}: gate {f['gate']}, w13 {f['w13']}, w2 {f['w2']} on hidden states of width {x['width']}")
    return E, int(f["top_k"]), H, I


def _calls(p: int, x: dict[str, Any], out: str, f: dict[str, Any], dims: tuple[int, int, int, int]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The block's Calls in piece p on hidden states x, ending in the op's return `out`, and its planes' widths."""
    E, TOPK, H, I = dims
    lg, rw, ri, up, ac, down = (f"c{p:02d}.moe/{n}" for n in PLANES)
    widths = {lg: E, rw: TOPK, ri: TOPK, up: TOPK * 2 * I, ac: TOPK * I, down: TOPK * H}
    v = lambda name: {"value": name, "width": widths[name]}
    w = {k: {"weight": f[k]["weight"], "shape": list(f[k]["shape"])} for k in ("gate", "w13", "w2")}
    router = M.MoeRouterTopKNorm if f["renormalize"] else M.MoeRouterTopK
    calls = [{"op": "mm", "gemm": [H, E], "reads": [x, {**w["gate"], "t": True}], "writes": [lg]},
             {"op": "moe_router", **bound(router, E=E, TOPK=TOPK, VPT=int(f["VPT"])), "reads": [v(lg)], "writes": [rw, ri]},
             {"op": "moe_up", "moe": plan_key(E, H, 2 * I, False), "reads": [x, v(ri), w["w13"]], "writes": [up]},
             {"op": "moe_act", **bound(act.SiluMul, I=I), "reads": [v(up)], "writes": [ac]},
             {"op": "moe_down", "moe": plan_key(E, I, H, True), "reads": [v(ac), v(ri), v(rw), w["w2"]], "writes": [down]},
             {"op": "moe_sum", **bound(M.MoeSum, TOPK=TOPK, H=H), "reads": [v(down)], "writes": [out]}]
    return [{**c, "piece": p} for c in calls], widths


def expand(wired: dict[str, Any], moe: dict[str, Any] | None) -> tuple[dict[str, Any], Callable[[int], dict[str, Any]] | None]:
    """`served.wire`'s program with each fused-MoE op expanded into its Calls and its planes' values (before the op's return), and
    the step plans' expert GEMMs (`expert_plans`); (wired, None) when no piece calls the op.  `moe`: the serving process's record,
    {"layers": moe_facts, "pieces": moe_pieces, "configs": moe_configs, "cc": its compute capability}."""
    ops = [c for c in wired["calls"] if c["op"] == "moe_forward"]
    if not ops:
        if moe and moe.get("layers"):
            raise Unmodelled(f"the serving process has {len(moe['layers'])} fused-MoE layers, and no piece calls the fused-MoE op")
        return wired, None
    if not moe:
        raise Unmodelled("a piece calls the fused-MoE op, and the Build has no fused-MoE record of the serving process")
    if tuple(moe.get("cc") or ()) not in CC:
        raise Unmodelled(f"fused_moe_kernel's tl.dot lowering on compute capability {moe.get('cc')}: cc {', '.join(map(str, CC))} "
                         f"are modelled")
    p, n = Counter(c["piece"] for c in ops).most_common(1)[0]
    if n > 1:
        raise Unmodelled(f"piece {p} calls the fused-MoE op {n} times: its planes are named by piece")
    calls, planes, keys = [], {}, set()
    for c in wired["calls"]:
        if c["op"] != "moe_forward":
            calls.append(c)
            continue
        p, x, out = c["piece"], c["reads"][0], c["writes"][0]
        layer = moe["pieces"].get(str(p))
        if layer is None or c["layer"].get("name", layer) != layer:
            raise Unmodelled(f"piece {p}'s fused-MoE op names {c['layer']}, the serving process ran layer {layer}")
        f = moe["layers"].get(layer)
        block, widths = _calls(p, x, out, f, _block(layer, f, x))
        calls += block
        planes[out] = widths
        keys |= {b["moe"] for b in block if "moe" in b}
    values: dict[str, dict[str, Any]] = {}
    for name, val in wired["values"].items():
        for pn, width in planes.get(name, {}).items():
            if pn in wired["values"]:
                raise Unmodelled(f"{pn}: a value of the piece already has the name of a fused-MoE plane")
            values[pn] = {"width": width, "piece": val["piece"]}
        values[name] = val
    return {**wired, "calls": calls, "values": values}, expert_plans(keys, moe.get("configs") or {})
