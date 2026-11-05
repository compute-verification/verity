"""The fused-MoE block's value families as an acquisition source of the native committer (`NativeHostCommitter.attach_source`,
the FA2 `HiddenM1Source` precedent): the values the Program's MoE instances produce are materialised device tensors BETWEEN the block's
kernels, but inside ONE module's forward (`MoERunner`), in workspace buffers the next layer overwrites -- a module forward hook sees only
the final [M, H] sum.  This source wraps the three Python launch sites of the pinned vLLM and snapshots each producer's output into the
step stream right after its kernel (stream-ordered, before the workspace is reused), so every MoE value family the query needs is bound:

    ops.topk_softmax(w!, ids!, tei!, logits, ...)                -> moe/L<k>/router_w  (f32 [M, TOPK])   MoeRouterTopK_v1 outputs
                                                                    moe/L<k>/router_ids (i32 [M, TOPK])
    invoke_fused_moe_triton_kernel(A, B, C!, ..., MUL_ROUTED_WEIGHT=False, top_k=TOPK, ...) -> moe/L<k>/up   (bf16 [M, TOPK, 2I])  MoeExpertGemm_v1
    invoke_fused_moe_triton_kernel(A, B, C!, ..., MUL_ROUTED_WEIGHT=True,  top_k=1, ...)    -> moe/L<k>/act  (bf16 [M*TOPK, I]: A = silu_and_mul's
                                                                    output = the down launch's input)  SiluMul_v1 (MoE rows)
                                                                    moe/L<k>/down (bf16 [M, TOPK, H])  MoeExpertGemmW_v1
    ops.moe_sum(y, out!)                                         -> moe/L<k>/sum  (bf16 [M, H])        MoeSum_v1

and under block FP8 (below) the w13 launch's (A, A_scale) -> moe/L<k>/in_q, in_scale and the w2 launch's -> moe/L<k>/act_q, act_scale
(Fp8GroupQuantUe8m0_v1 outputs, the MoeExpertFp8Gemm(W)_v1 operands), with `act` the bf16 input of the quant that wrote the w2's A.

The grouped router (`GroupedTopKRouter._compute_routing` -> `grouped_topk_router.grouped_topk` -> `fused_grouped_topk` ->
`_custom_ops.grouped_topk`, DeepSeek-V3 / GLM `noaux_tc`) is bound at `grouped_topk_router.grouped_topk`, the name `_compute_routing`
looks up at call time: that function is `torch.compile`d (even under enforce_eager), so a wrapper on `_custom_ops.grouped_topk` would
sit inside the traced region.  Its return is the custom op's own (topk_values f32 [M, TOPK], topk_indices i32 [M, TOPK]), unchanged:

    grouped_topk(hidden_states, gating_output, topk, ..., e_score_correction_bias)  -> moe/L<k>/router_w, moe/L<k>/router_ids
        the outputs of `MoeRouterSigmoidBias_v1{E, TOPK, SCALE, KERNEL}` per token row; its inputs are `gating_output`'s row (the
        gate's f32 output) and `e_score_correction_bias`.  KERNEL is the launch (`router_kernel`), recorded per step.

With block-FP8 experts (`TritonExperts.apply`: `use_fp8_w8a8`, `block_shape` [128, 128]) both launches carry an A_scale: their A is
an e4m3 group quant, and the source binds each launch's (A, A_scale) -- the operands `xq`, `sx` of the w13 and w2 Programs -- as
`in_q`, `in_scale` (the input's quant, [M, H] e4m3 and [M, H/128] f32) and `act_q`, `act_scale` (the activation's, [M*TOPK, I] and
[M*TOPK, I/128]).  GLM-5.3's served path (DeepGEMM's ue8m0 quant on, vLLM's sm_90 default) runs `_C.silu_and_mul` into a bf16
activation and quantizes it with `torch.ops._C.per_token_group_fp8_quant`, as it does the input; the source wraps that op and binds
`act` = the input of the call whose e4m3 output is the w2 launch's A -- ten tensors per MoE layer (`MOE_FP8_PLANES`).  Without ue8m0
vLLM fuses the SiLU and the quant (`ops.silu_and_mul_per_block_quant`): no bf16 activation exists and `act` is not bound (nine).

A MoE module that does not all-reduce its output on a rank (GLM-5.3: `reduce_results=False`; the next layer's fused all-reduce norm
sums it) returns a value no other binder sees: on a TP rank (`world` > 1) the source binds it as `<module>/out` after the forward.

"Right after its kernel" is only true of a committer that copies at `_acquire`.  The gather collector
(`native_collect_v2b`) STAGES a descriptor and copies at the next flush point -- a module hook -- and the fused-MoE block has no hook
between its launches: vLLM's TritonExperts carves `intermediate_cache1` (up) and `intermediate_cache3` (down) out of the same workspace
(`_resize_cache(workspace2, ...)` twice; byte-for-byte alias when 2I == H), so the block's own w2 launch overwrites `up` before the next
layer's `input_layernorm` pre-flush reads it -> the committed `up` was the same layer's `down` (on the real OLMoE B1
record, 320/320 MoeExpertGemm_v1 VUs).  The source therefore asks the committer to flush right after staging `up`
(`committer.flush_after_plane("up")`: one gather launch per MoE layer per step on the gather collector, a no-op on the arena committer);
`act`, `down`, `sum` are overwritten only by the next layer, after that layer's pre-flush, and the router outputs are fresh allocations.

The Marlin experts (`MarlinExperts.apply` -> `fused_marlin_moe`, experts/marlin_moe.py: DeepSeek-V4's MXFP4 experts on Hopper)
launch `ops.moe_wna16_marlin_gemm(input, output, ...)` twice, the first with `mul_topk_weights` False:

    moe_wna16_marlin_gemm(hidden_states, cache1!, w13, ..., top_k=TOPK, mul_topk_weights=False) -> moe/L<k>/up   (bf16 [M*TOPK, 2I])
    moe_wna16_marlin_gemm(cache2, cache3!, w2, ..., a_scales, ..., top_k=1, mul_topk_weights=True) -> moe/L<k>/act (bf16 [M*TOPK, I])
                                                                    moe/L<k>/down (bf16 [M*TOPK, H])

and the sum is `MarlinExperts.moe_sum`'s `ops.moe_sum(input, output)`.  A launch is the w2 one only with `mul_topk_weights` True and
`top_k` 1, so a model that applies the router weight on the input (which swaps the flags) is refused, not bound under swapped names.
cache1 and cache3 are one buffer, as in TritonExperts, and
the modular kernel carves the block's output, which `moe_sum` writes, out of the workspace that holds cache2
(`MarlinExperts.workspace_shapes` swaps the two), so the source also flushes after the w2 launch (`flush_after_plane("act")`).

DeepSeek-V4's router (`FusedTopKBiasRouter._compute_routing`) calls `fused_topk_bias`, the name its module holds, with
`scoring_func` "sqrtsoftplus": on a layer with a hash table (`hash_indices_table`, the first `num_hash_layers`) it launches
`ops.topk_hash_softplus_sqrt`, elsewhere `dsv4_topk`'s Triton kernel when `can_use_dsv4_topk` holds.  Its return (topk_weights
f32 [M, TOPK], topk_ids i32 [M, TOPK]) is bound as router_w, router_ids, the kernel recorded per step.  A call without a hash
table that does not take `dsv4_topk` is refused (the scored router's Program is that kernel's); a call with another scoring
function is left to the launches it reaches (`topk_softmax`).

A manifest of record without `moe_block_stream` rows (DeepSeek-V4-Pro's: the Marlin launches and the routers are Calls under the MoE
module, `<module>/<fx node>`) is bound at those Calls: the source commits each value as `<module>/<label>` (`at_calls`), and
`bind.Binder` binds a label `PORTS` declares at its port of one of the module's call boundaries: up and down at the Marlin GEMMs,
act at `SiluMulClamp_v1`'s, sum at `MoeSum_v1`'s, router_w and router_ids at `dsv4_topk`'s, and on a hash layer router_w.hash at
`MoeRouterHash_v1`'s (a module without router rows of `format.MOE_PLANE_FAMILIES` has no planes: `format.moe_plane_modules`).  A hash
layer's ids are the module's `GatherI32x{V}` rows, committed as `ids`, and the token ids those gathers read as `in`
(`hash_router_input`); router_ids and ids are committed as their 32-bit words, as those rows are (`f32`).  `layer_of` tells
`identity_check` the source's names, a hash layer holding one more.  The call boundaries at those ports are this source's (`claims`,
one of `call_boundary_source.CLAIMS`): every TP rank attaches it (`pipeline.tp.rank_worker`), and the CPU gate counts them as the MoE
source's on several ranks when the manifest names no `moe_block_stream` plane (`taps.attached`).

`<k>` is the layer index of the MoE module whose forward is executing (forward pre-hooks on every `MoERunner`-family module set it, and
forward hooks clear it).  Exactly `planes` tensors per MoE layer per step are expected (six bf16, seven on Marlin with a scaled
activation, ten block FP8, fixed by the run's first w2 launch); `NativeHostCommitter.identity_check` verifies that count per layer per
step (fail closed:
a launch site that was not hit -- another kernel path, a fused variant -- is a missing tensor, never a silently smaller stream).  The
routing (which expert each token chose, the sorted schedule) is bound through `router_ids` and the pair-major `up`/`act`/`down` tensors:
the Program addresses pair p = t*TOPK + k, exactly the row order of these tensors, so the collected values map to instances by position
whatever the tokens chose (`rules/vllm_moe.py`; the sorted block schedule never enters the values).  Coverage class `moe_block_stream`.
"""
from __future__ import annotations

import importlib
import inspect
import re
import time
from collections.abc import Mapping
from typing import Any

from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources.partial_source import _moe_reduces
from verity_vllm.engine import hooks
from verity.ml.definitions.registry.moe import router_kernel
from verity_vllm.query.manifest.bind import Port
from verity_vllm.query.manifest.format import MOE_FP8_PLANES, MOE_PLANES

MOE_MODULE_CLASSES = {"MoERunner", "MoERunnerInterface", "FusedMoE", "SharedFusedMoE"}
TENSORS_PER_LAYER = len(MOE_PLANES)
COVERS_FAMILIES = ["MoeRouterTopK_v1", "MoeExpertGemm_v1", "MoeExpertGemm_v2", "SiluMul_v1", "SiluMul_v2", "MoeExpertGemmW_v1", "MoeExpertGemmW_v2", "MoeSum_v1",
                   "MoeRouterSigmoidBias_v1", "Fp8GroupQuant_v1", "Fp8GroupQuantUe8m0_v1", "MoeExpertFp8Gemm_v1", "MoeExpertFp8GemmW_v1",
                   "SiluMulFp8BlockQuant_v1"]
GROUP_QUANT_OP = "per_token_group_fp8_quant"     # torch.ops._C.per_token_group_fp8_quant(input, output_q, output_s, group_size, ...)
# grouped_topk(hidden_states, gating_output, topk, renormalize, num_expert_group, topk_group, scoring_func, routed_scaling_factor,
# e_score_correction_bias) with its defaults: the pinned signature, which `_compute_routing` calls by keyword
GROUPED_TOPK_PARAMS = (("hidden_states", None), ("gating_output", None), ("topk", None), ("renormalize", None), ("num_expert_group", 0),
                       ("topk_group", 0), ("scoring_func", "softmax"), ("routed_scaling_factor", 1.0), ("e_score_correction_bias", None))
TOPK_BIAS_MODULE = "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router"
#: moe_wna16_marlin_gemm's positional index of each argument the source reads, after (input, output)
MARLIN_ARGS = {"a_scales": 3, "top_k": 14, "mul_topk_weights": 15}
HASH_KERNEL = "topk_hash_softplus_sqrt"


def _call(family: str, port: str) -> Port:
    return Port(frozenset({family}), port, call_boundary=True)


#: `bind.Binder`'s declared labels at calls (`at_calls`): each label -> the port of the MoE module's Call whose value it commits
PORTS = {"router_w": _call("MoeRouterSqrtSoftplus_v1", "0"), "router_ids": _call("MoeRouterSqrtSoftplus_v1", "1"),
         "router_w.hash": _call("MoeRouterHash_v1", "out"), "up": _call("MarlinMoeExpertGemm_v1", "out"),
         "act": _call("SiluMulClamp_v1", "out"), "down": _call("MarlinMoeExpertGemmW_v1", "out"), "sum": _call("MoeSum_v1", "out")}
#: every label the source commits at calls under a MoE module
LABELS = frozenset({*MOE_FP8_PLANES, *PORTS, "ids", "in"})
#: (Definition, port) of every call boundary a label of `PORTS` binds
CALL_PORTS = frozenset((f, p.port) for p in PORTS.values() for f in p.families)


PLANE_NAME = re.compile(r"moe/L(\d+)/")


def binds_planes(manifest: Mapping[str, Any] | None) -> bool:
    """Whether the source binds the `moe_block_stream` planes (`moe/L<k>/<plane>`): with no manifest, or one that names them."""
    return manifest is None or any(r.get("family") == "moe_block_stream" for r in manifest.get("identities") or ())


def claims(r: Mapping[str, Any]) -> str | None:
    """`call_boundary_source.CLAIMS`: a call boundary at a port `PORTS` declares is this source's."""
    return (MoeBlockSource.short_name if r.get("family") == CB.FAMILY and
            (str(r.get("spec") or "").split("{", 1)[0], str(r.get("port") or r.get("output_member"))) in CALL_PORTS else None)


if claims not in CB.CLAIMS:
    CB.CLAIMS.append(claims)


def plane_layer(name: str) -> int | None:
    """The layer `k` of a plane's name (`moe/L<k>/<plane>`, the router-softmax tap's too); None for any other name."""
    m = PLANE_NAME.match(name)
    return int(m.group(1)) if m else None


def hash_router_input(input_tokens: Any, table: Any, topk: int) -> Any:
    """What a hash layer's K `GatherI32x{V}` calls per token read: each token's id in the table's dtype (as `fused_topk_bias` casts
    it), K times in a row (pair-major, the gathers' order)."""
    return input_tokens.to(table.dtype).repeat_interleave(topk)


def grouped_topk_args(a: tuple, k: dict) -> dict[str, Any]:
    """The arguments of one `grouped_topk` call by name (positional or keyword, defaults filled)."""
    out = {name: (a[i] if i < len(a) else default) for i, (name, default) in enumerate(GROUPED_TOPK_PARAMS)}
    out.update({name: v for name, v in k.items() if name in out})
    return out


def moe_layers_of(model: Any) -> dict[str, int]:
    """MoE module name -> layer index for every MoE-runner module of `model` (the outermost one per layer)."""
    out: dict[str, int] = {}
    for name, m in model.named_modules():
        names = {c.__name__ for c in type(m).__mro__}
        if names & MOE_MODULE_CLASSES and ".layers." in name:
            k = int(name.split(".layers.")[1].split(".")[0])
            if not any(n != name and name.startswith(n + ".") for n in out):
                out[name] = k
    return out


class MoeBlockSource:
    short_name = "moe"
    ns_name = "moe_block"

    def __init__(self, committer, model, *, world: int = 1, manifest: Mapping[str, Any] | None = None) -> None:
        import torch

        self.torch = torch
        self.c = committer
        self.layers = moe_layers_of(model)
        self.modules = [(n, m) for n, m in model.named_modules() if n in self.layers]
        self.at_calls = not binds_planes(manifest)
        #: layer -> its MoE module, the prefix of the layer's names at calls (a plane's is `moe/L<k>`)
        self.module_of = {k: n for n, k in self.layers.items()}
        #: the layers whose router read a hash table, at calls (each commits ids and in beside its weights)
        self.hash_layers: set[int] = set()
        # the MoE modules whose returned value this source binds as `<module>/out`: on a TP rank, those that do not all-reduce it
        self.outputs = {n for n, m in self.modules if world > 1 and not _moe_reduces(m, world)}
        self.cur_layer: int | None = None
        self._handles: list = []
        self._hooks = hooks.Hooks()
        self.active = False
        self.stats = {"launches": 0, "bytes": 0, "layers": len(self.layers), "tensors_per_layer": TENSORS_PER_LAYER, "acquire_s": 0.0,
                      "unattributed": 0, "shapes": {}, "plane_flushes": 0, "router": {}, "outputs": 0}
        # the committer decides (gather collector: a launch; arena committer: nothing staged -> False); absent = an old committer
        self._flush_after_plane = getattr(committer, "flush_after_plane", None)
        self.act_scaled: bool | None = None          # set by the run's first w2 launch: True when its A carries group scales (block FP8)
        self.planes: tuple[str, ...] = MOE_PLANES    # the planes every MoE layer binds per step, fixed with act_scaled
        self.router_steps: dict[str, list[int]] = {}  # grouped_topk kernel -> the steps that launched it
        self._quants: list[tuple[Any, Any]] = []      # (output_q, input) of the group quants since this layer's w13 launch
        self._fused_act = False                       # this layer ran silu_and_mul_per_block_quant since its w13 launch

    # -- patch points ------------------------------------------------------------------------------------------------------
    def _sites(self) -> list[tuple[Any, str]]:
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.fused_moe import fused_moe as fm
        # the ONE Triton launcher, at both names it is reached by: `fused_moe.invoke_fused_moe_triton_kernel` (the functional path's
        # `dispatch_fused_moe_kernel` looks it up in its module globals) and the engine's `experts.triton_moe` module, which imported the
        # name (TritonExperts.apply).  One binding per launch either way.
        sites: list[tuple[Any, str]] = [(ops, "topk_softmax"), (ops, "moe_sum"), (fm, "invoke_fused_moe_triton_kernel")]
        if hasattr(ops, "silu_and_mul_per_block_quant"):
            sites.append((ops, "silu_and_mul_per_block_quant"))
        if hasattr(ops, "moe_wna16_marlin_gemm"):
            sites.append((ops, "moe_wna16_marlin_gemm"))
        for modname, attr in (("vllm.model_executor.layers.fused_moe.experts.triton_moe", "invoke_fused_moe_triton_kernel"),
                              ("vllm.model_executor.layers.fused_moe.router.grouped_topk_router", "grouped_topk"),
                              (TOPK_BIAS_MODULE, "fused_topk_bias")):
            try:
                mod = importlib.import_module(modname)
            except Exception:  # noqa: BLE001
                continue
            if hasattr(mod, attr):
                sites.append((mod, attr))
        return sites

    @staticmethod
    def _fused_router(args: dict[str, Any]) -> bool:
        """Whether `grouped_topk` takes `fused_grouped_topk` (the custom op) for these arguments: the pinned function's own test."""
        from vllm import envs
        from vllm.platforms import current_platform
        return bool(envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK and current_platform.is_cuda() and args["num_expert_group"] <= 32
                    and args["topk"] <= 32 and args["e_score_correction_bias"] is not None)

    def install(self) -> None:
        me = self
        for name, m in self.modules:
            k = self.layers[name]

            def pre_hook(mod, args, _k=k):
                me.cur_layer = _k
                me._quants, me._fused_act = [], False

            def post_hook(mod, args, out, _name=name):
                if _name in me.outputs:
                    me._bind_output(_name, out)
                me.cur_layer = None

            self._handles.append(m.register_forward_pre_hook(pre_hook))
            self._handles.append(m.register_forward_hook(post_hook, always_call=True))
        for mod, attr in self._sites():
            self._hooks.wrap(mod, attr, lambda orig, attr=attr, mod=mod: self._wrap(attr, orig, mod))
        ops_c = self.torch.ops._C
        if hasattr(ops_c, GROUP_QUANT_OP):
            self._hooks.patch(ops_c, GROUP_QUANT_OP, self._group_quant(getattr(ops_c, GROUP_QUANT_OP)))
        self.active = True

    def out_site_names(self) -> list[str]:
        """The MoE modules whose output this source binds as `<module>/out`: expected modules of every step for `identity_check`."""
        return sorted(self.outputs)

    def ports(self) -> dict[str, Port]:
        return dict(PORTS) if self.at_calls else {}

    def layer_of(self, name: str) -> int | None:
        """The MoE layer of a name this source commits (not `<module>/out`); None for any other name."""
        if not self.at_calls:
            return plane_layer(name)
        prefix, _, label = name.rpartition("/")
        k = self.layers.get(prefix)
        return k if k is not None and label in LABELS else None

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._hooks.uninstall()
        self.active = False

    def on_step_begin(self) -> None:
        self.cur_layer = None
        self._quants = []

    # -- acquisition -------------------------------------------------------------------------------------------------------
    def _bind(self, what: str, t) -> None:
        job = getattr(self.c, "_cur", None)
        if job is None or t is None or not hasattr(t, "device") or t.device.type != "cuda":
            return
        if self.cur_layer is None:
            self.stats["unattributed"] += 1
            self.c.errors.append(f"moe source: {what} launched outside a MoE module forward (no layer attribution)"[:200])
            return
        t0 = time.perf_counter()
        name = f"{self.module_of[self.cur_layer] if self.at_calls else f'moe/L{self.cur_layer}'}/{what}"
        self.c._acquire(name, t, job)
        if what == "up" and self._flush_after_plane is not None and self._flush_after_plane("up"):
            self.stats["plane_flushes"] += 1             # `up` leaves the device before the block's w2 launch overwrites its workspace
        nb = t.numel() * t.element_size()
        self.stats["launches"] += 1
        self.stats["bytes"] += nb
        self.stats["acquire_s"] += time.perf_counter() - t0
        key = f"{what}:{tuple(t.shape)}:{str(t.dtype).replace('torch.', '')}"
        self.stats["shapes"][key] = self.stats["shapes"].get(key, 0) + 1

    def _bind_router(self, args: dict[str, Any], r: Any) -> None:
        """One `grouped_topk` call: its (topk_weights, topk_ids) as router_w / router_ids, and the kernel it launched (per step)."""
        job = getattr(self.c, "_cur", None)
        if job is None:
            return
        logits = args["gating_output"]
        if not self._fused_router(args):
            self.c.errors.append("moe source: grouped_topk took its torch path, not the fused custom op; router not bound"[:200])
            self.c.x03_expected = getattr(self.c, "x03_expected", 0) + 1
            return
        kernel = router_kernel(int(logits.shape[-1]), int(logits.shape[0]), int(args["topk"]))
        if kernel is None:
            self.c.errors.append(f"moe source: grouped_topk at {int(logits.shape[-1])} experts, top {args['topk']}: kernel not pinned"[:200])
        key = f"{args['scoring_func']}:{kernel}"
        self.stats["router"][key] = self.stats["router"].get(key, 0) + 1
        steps = self.router_steps.setdefault(str(kernel), [])
        step = int(getattr(job, "step", -1))
        if not steps or steps[-1] != step:
            steps.append(step)
        self._bind("router_w", r[0])
        self._bind("router_ids", r[1])

    def _refuse(self, why: str) -> None:
        self.c.errors.append(f"moe source: {why}"[:200])
        self.c.x03_expected = getattr(self.c, "x03_expected", 0) + 1

    def _bind_output(self, name: str, out: Any) -> None:
        """The MoE module's returned tensor as `<module>/out`.  It is a fresh tensor, so the staged reference holds its words until the
        next flush, which the next layer's fused all-reduce norm makes before writing its residual into it."""
        job = getattr(self.c, "_cur", None)
        if job is None or not hasattr(out, "device") or out.device.type != "cuda":
            if job is not None:
                self._refuse(f"{name} returned {type(out).__name__}, not one device tensor; {name}/out not bound")
            return
        self.c._acquire(f"{name}/out", out, job)
        self.stats["outputs"] += 1
        self.stats["bytes"] += out.numel() * out.element_size()

    def _group_quant(self, orig):
        me = self

        def per_token_group_fp8_quant(input, output_q, *a, **k):
            r = orig(input, output_q, *a, **k)
            if me.cur_layer is not None:
                me._quants.append((output_q, input))
            return r
        return per_token_group_fp8_quant

    @staticmethod
    def _same(q: Any, A: Any) -> bool:
        if q is A:
            return True
        ptr = getattr(q, "data_ptr", None)
        return ptr is not None and hasattr(A, "data_ptr") and ptr() == A.data_ptr() and q.numel() == A.numel()

    def _bind_w13(self, A: Any, scale: Any, C: Any) -> None:
        """The w13 launch: under block FP8 its (A, A_scale) is the input's group quant, `in_q` / `in_scale`; then `up`.  Only the group
        quants after it can be the activation's."""
        if scale is not None and hasattr(scale, "device"):
            self._bind("in_q", A)
            self._bind("in_scale", scale)
        self._bind("up", C)
        self._quants, self._fused_act = [], False

    def _bind_w2(self, A: Any, scale: Any, C: Any) -> None:
        """The w2 launch: unscaled, its A is the bf16 `act`; under block FP8 its (A, A_scale) is the activation's quant, `act_q` /
        `act_scale`, and `act` the input of the group quant that wrote A (none when vLLM fused the SiLU into the quant)."""
        scaled = scale is not None and hasattr(scale, "device")
        act = next((x for q, x in self._quants if self._same(q, A)), None) if scaled else A
        if scaled and act is None and not self._fused_act:
            self._refuse(f"the w2 launch's A at layer {self.cur_layer} is no {GROUP_QUANT_OP} output of the layer; act not bound")
        self._fix_planes(scaled, tuple(p for p in MOE_FP8_PLANES if p != "act" or act is not None) if scaled else MOE_PLANES)
        if act is not None:
            self._bind("act", act)
        if scaled:
            self._bind("act_q", A)
            self._bind("act_scale", scale)
        self._bind("down", C)
        self._quants = []

    def _bind_marlin_w2(self, act: Any, scale: Any, down: Any) -> None:
        """The Marlin w2 launch: its input is the bf16 `act`, with its a_scales as `act_scale` when the launch carries them."""
        scaled = scale is not None and hasattr(scale, "device")
        self._fix_planes(scaled, MOE_PLANES[:4] + ("act_scale",) + MOE_PLANES[4:] if scaled else MOE_PLANES)
        self._bind("act", act)
        if scaled:
            self._bind("act_scale", scale)
        self._bind("down", down)

    def _fix_planes(self, scaled: bool, planes: tuple[str, ...]) -> None:
        """The run's first w2 launch fixes the planes every layer binds (the per-layer count `identity_check` holds the steps to); a
        launch that binds others is an error."""
        if self.act_scaled is None:
            self.act_scaled, self.planes = scaled, planes
            self.stats["tensors_per_layer"] = len(planes)
        elif planes != self.planes:
            self._refuse(f"a w2 launch {'with' if scaled else 'without'} A_scale binding {len(planes)} planes in a run whose first bound "
                         f"{len(self.planes)}")

    def _bind_sqrtsoftplus(self, args: dict[str, Any], r: Any, mod: Any) -> None:
        """One `fused_topk_bias` call with `scoring_func` "sqrtsoftplus": its (topk_weights, topk_ids) as router_w / router_ids, and the
        kernel it launched (per step); a call without a hash table that `dsv4_topk` would not take is refused."""
        job = getattr(self.c, "_cur", None)
        if job is None or args["scoring_func"] != "sqrtsoftplus":
            return
        if args["hash_indices_table"] is not None:
            kernel = HASH_KERNEL
        else:
            idx = self.torch.int32 if args["indices_type"] is None else args["indices_type"]
            if not mod.can_use_dsv4_topk(args["gating_output"], args["e_score_correction_bias"], args["topk"], args["renormalize"], idx):
                self.c.errors.append("moe source: fused_topk_bias (sqrtsoftplus, no hash table) did not take dsv4_topk; router not bound")
                self.c.x03_expected = getattr(self.c, "x03_expected", 0) + 1
                return
            kernel = "dsv4_topk"
        key = f"sqrtsoftplus:{kernel}"
        self.stats["router"][key] = self.stats["router"].get(key, 0) + 1
        steps = self.router_steps.setdefault(kernel, [])
        step = int(getattr(job, "step", -1))
        if not steps or steps[-1] != step:
            steps.append(step)
        if not self.at_calls:
            self._bind("router_w", r[0])
            self._bind("router_ids", r[1])
            return
        ids = r[1].view(self.torch.float32)              # an i32 row of a Call is committed as its 32-bit words ("f32")
        if kernel != HASH_KERNEL:
            self._bind("router_w", r[0])
            self._bind("router_ids", ids)
            return
        if self.cur_layer is not None:
            self.hash_layers.add(self.cur_layer)
        self._bind("router_w.hash", r[0])
        self._bind("ids", ids)
        self._bind("in", hash_router_input(args["input_tokens"], args["hash_indices_table"], args["topk"]))

    def _wrap(self, attr: str, orig, mod: Any = None):
        me = self
        if attr == "moe_wna16_marlin_gemm":
            def moe_wna16_marlin_gemm(input, output, *a, **k):
                r = orig(input, output, *a, **k)
                def arg(name: str, i: int) -> Any:
                    return k[name] if name in k else (a[i] if len(a) > i else None)
                mtw, top_k = arg("mul_topk_weights", MARLIN_ARGS["mul_topk_weights"]), arg("top_k", MARLIN_ARGS["top_k"])
                if mtw is True and top_k == 1:
                    me._bind_marlin_w2(input, arg("a_scales", MARLIN_ARGS["a_scales"]), r)
                    # the block's output, which moe_sum writes next, is carved from the workspace that holds act
                    if getattr(me.c, "_cur", None) is not None and me._flush_after_plane is not None and me._flush_after_plane("act"):
                        me.stats["plane_flushes"] += 1
                elif mtw is False:
                    me._bind("up", r)
                else:
                    me.c.errors.append(f"moe source: a Marlin launch with mul_topk_weights {mtw!r}, top_k {top_k!r}; not bound"[:200])
                    me.c.x03_expected = getattr(me.c, "x03_expected", 0) + 1
                return r
            return moe_wna16_marlin_gemm
        if attr == "fused_topk_bias":
            sig = inspect.signature(orig)
            def fused_topk_bias(*a, **k):
                r = orig(*a, **k)
                b = sig.bind(*a, **k)
                b.apply_defaults()
                me._bind_sqrtsoftplus(b.arguments, r, mod)
                return r
            return fused_topk_bias
        if attr == "topk_softmax":
            def topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output, *a, **k):
                r = orig(topk_weights, topk_ids, token_expert_indices, gating_output, *a, **k)
                me._bind("router_w", topk_weights)
                me._bind("router_ids", topk_ids)
                return r
            return topk_softmax
        if attr == "moe_sum":
            def moe_sum(inp, out, *a, **k):
                r = orig(inp, out, *a, **k)
                me._bind("sum", out)
                return r
            return moe_sum
        if attr == "grouped_topk":
            def grouped_topk(*a, **k):
                r = orig(*a, **k)
                me._bind_router(grouped_topk_args(a, k), r)
                return r
            return grouped_topk
        if attr == "silu_and_mul_per_block_quant":
            def silu_and_mul_per_block_quant(*a, **k):
                me._fused_act = me._fused_act or me.cur_layer is not None
                return orig(*a, **k)
            return silu_and_mul_per_block_quant
        # the two Triton launchers share the positional prefix (A, B, C, A_scale, ..., mul_routed_weight, top_k, config)
        def launch(A, B, C, *a, **k):
            r = orig(A, B, C, *a, **k)
            mrw = k.get("mul_routed_weight")
            if mrw is None:
                # invoke_fused_moe_triton_kernel(A, B, C, A_scale, B_scale, topk_weights, sorted, expert_ids, npp, mul_routed_weight, top_k, ...)
                mrw = a[6] if len(a) > 6 else None
            scale = k["A_scale"] if "A_scale" in k else (a[0] if a else None)
            if mrw is True:
                me._bind_w2(A, scale, C)
            elif mrw is False:
                me._bind_w13(A, scale, C)
            else:
                me._refuse(f"{attr} launch with undetermined MUL_ROUTED_WEIGHT ({mrw!r}); not bound")
            return r
        return launch

    # -- declaration -------------------------------------------------------------------------------------------------------
    def declaration_classes(self) -> list[dict]:
        per_layer = self.stats["tensors_per_layer"]
        return [{"class": "moe_block_stream", "how": "bound-as-leaf", "covers_families": COVERS_FAMILIES, "rule": None, "count": None,
                 "bytes": self.stats["bytes"], "tensors": self.stats["launches"], "layers": self.stats["layers"], "tensors_per_layer": per_layer,
                 "plane_flushes": self.stats["plane_flushes"], "act_scaled": self.act_scaled, "planes": list(self.planes),
                 "at_calls": self.at_calls, "hash_layers": sorted(self.hash_layers),
                 "router": dict(self.stats["router"]), "router_kernel_steps": {k: list(v) for k, v in self.router_steps.items()},
                 "outputs": {"modules": len(self.outputs), "bound": self.stats["outputs"]},
                 "evidence": ("MoE block source: wraps vllm._custom_ops.topk_softmax / moe_sum / moe_wna16_marlin_gemm, "
                              "fused_moe.invoke_fused_moe_triton_kernel (also at experts.triton_moe), grouped_topk_router.grouped_topk, "
                              "fused_topk_bias_router.fused_topk_bias and torch.ops._C.per_token_group_fp8_quant; per MoE layer per step "
                              f"{', '.join(self.planes)} "
                              "snapshotted into the packed step stream (same run root) right after each kernel; the up plane is "
                              "flushed off the device right after staging (its workspace is the block's own w2 output: cache1/cache3 alias), "
                              "and on Marlin act and down after the w2 launch (moe_sum's output is carved from act's workspace) -- "
                              f"{self.stats['plane_flushes']} flushes; identity_check requires exactly "
                              f"{per_layer} per layer per step; value correspondence: verity_vllm/ops/moe_kernel_xcheck.py "
                              f"(registry.moe Definitions bit-exact vs the same kernels); shapes seen {self.stats['shapes']}")}]


__all__ = ["MoeBlockSource", "moe_layers_of", "router_kernel", "grouped_topk_args", "binds_planes", "claims", "hash_router_input",
           "MOE_MODULE_CLASSES", "TENSORS_PER_LAYER", "COVERS_FAMILIES", "GROUP_QUANT_OP", "HASH_KERNEL", "LABELS", "PORTS", "CALL_PORTS"]
