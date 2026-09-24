"""[R13 moe / MOE-03] The fused-MoE block's value families as an acquisition source of the native committer (`NativeHostCommitter.attach_source`,
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

[R16 moe FLUSH-03] "right after its kernel" is only true of a committer that copies at `_acquire`.  The gather collector
(`native_collect_v2b`) STAGES a descriptor and copies at the next flush point -- a module hook -- and the fused-MoE block has no hook
between its launches: vLLM's TritonExperts carves `intermediate_cache1` (up) and `intermediate_cache3` (down) out of the same workspace
(`_resize_cache(workspace2, ...)` twice; byte-for-byte alias when 2I == H), so the block's own w2 launch overwrites `up` before the next
layer's `input_layernorm` pre-flush reads it -> the committed `up` was the same layer's `down` (R16 lane-moe C2 v3 on the real OLMoE B1
record, 320/320 MoeExpertGemm_v1 VUs).  The source therefore asks the committer to flush right after staging `up`
(`committer.flush_after_plane("up")`: one gather launch per MoE layer per step on the gather collector, a no-op on the arena committer);
`act`, `down`, `sum` are overwritten only by the next layer, after that layer's pre-flush, and the router outputs are fresh allocations.

`<k>` is the layer index of the MoE module whose forward is executing (forward pre-hooks on every `MoERunner`-family module set it).  Exactly
six tensors per MoE layer per step are expected; `NativeHostCommitter.identity_check` verifies that count per layer per step (fail closed:
a launch site that was not hit -- another kernel path, a fused variant -- is a missing tensor, never a silently smaller stream).  The
routing (which expert each token chose, the sorted schedule) is bound through `router_ids` and the pair-major `up`/`act`/`down` tensors:
the Program addresses pair p = t*TOPK + k, exactly the row order of these tensors, so the collected values map to instances by position
whatever the tokens chose (`rules/vllm_moe.py`; the sorted block schedule never enters the values).  Coverage class `moe_block_stream`.
"""
from __future__ import annotations

import time
from typing import Any

MOE_MODULE_CLASSES = {"MoERunner", "MoERunnerInterface", "FusedMoE", "SharedFusedMoE"}
TENSORS_PER_LAYER = 6
COVERS_FAMILIES = ["MoeRouterTopK_v1", "MoeExpertGemm_v1", "SiluMul_v1", "MoeExpertGemmW_v1", "MoeSum_v1"]


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

    def __init__(self, committer, model) -> None:
        import torch

        self.torch = torch
        self.c = committer
        self.layers = moe_layers_of(model)
        self.modules = [(n, m) for n, m in model.named_modules() if n in self.layers]
        self.cur_layer: int | None = None
        self._handles: list = []
        self._patched: list[tuple[Any, str, Any]] = []
        self.active = False
        self.stats = {"launches": 0, "bytes": 0, "layers": len(self.layers), "tensors_per_layer": TENSORS_PER_LAYER, "acquire_s": 0.0,
                      "unattributed": 0, "shapes": {}, "plane_flushes": 0}
        # [FLUSH-03] the committer decides (gather collector: a launch; arena committer: nothing staged -> False); absent = an old committer
        self._flush_after_plane = getattr(committer, "flush_after_plane", None)

    # -- patch points ------------------------------------------------------------------------------------------------------
    def _sites(self) -> list[tuple[Any, str]]:
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.fused_moe import fused_moe as fm
        # the ONE Triton launcher, at both names it is reached by: `fused_moe.invoke_fused_moe_triton_kernel` (the functional path's
        # `dispatch_fused_moe_kernel` looks it up in its module globals) and the engine's `experts.triton_moe` module, which imported the
        # name (TritonExperts.apply).  One binding per launch either way.
        sites: list[tuple[Any, str]] = [(ops, "topk_softmax"), (ops, "moe_sum"), (fm, "invoke_fused_moe_triton_kernel")]
        try:
            from vllm.model_executor.layers.fused_moe.experts import triton_moe as tm
            if hasattr(tm, "invoke_fused_moe_triton_kernel"):
                sites.append((tm, "invoke_fused_moe_triton_kernel"))
        except Exception:  # noqa: BLE001
            pass
        return sites

    def install(self) -> None:
        me = self
        for name, m in self.modules:
            k = self.layers[name]

            def pre_hook(mod, args, _k=k):
                me.cur_layer = _k

            self._handles.append(m.register_forward_pre_hook(pre_hook))
        for mod, attr in self._sites():
            orig = getattr(mod, attr)
            self._patched.append((mod, attr, orig))
            setattr(mod, attr, self._wrap(attr, orig))
        self.active = True

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        for mod, attr, orig in reversed(self._patched):
            setattr(mod, attr, orig)
        self._patched = []
        self.active = False

    def on_step_begin(self) -> None:
        self.cur_layer = None

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
        name = f"moe/L{self.cur_layer}/{what}"
        self.c._acquire(name, t, job)
        if what == "up" and self._flush_after_plane is not None and self._flush_after_plane("up"):
            self.stats["plane_flushes"] += 1             # [FLUSH-03] `up` leaves the device before the block's w2 launch overwrites its workspace
        nb = t.numel() * t.element_size()
        self.stats["launches"] += 1
        self.stats["bytes"] += nb
        self.stats["acquire_s"] += time.perf_counter() - t0
        key = f"{what}:{tuple(t.shape)}:{str(t.dtype).replace('torch.', '')}"
        self.stats["shapes"][key] = self.stats["shapes"].get(key, 0) + 1

    def _wrap(self, attr: str, orig):
        me = self
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
        # the two Triton launchers share the positional prefix (A, B, C, ..., mul_routed_weight, top_k, config)
        def launch(A, B, C, *a, **k):
            r = orig(A, B, C, *a, **k)
            mrw = k.get("mul_routed_weight")
            if mrw is None:
                # invoke_fused_moe_triton_kernel(A, B, C, A_scale, B_scale, topk_weights, sorted, expert_ids, npp, mul_routed_weight, top_k, ...)
                mrw = a[6] if len(a) > 6 else None
            if mrw is True:
                me._bind("act", A)
                me._bind("down", C)
            elif mrw is False:
                me._bind("up", C)
            else:
                me.c.errors.append(f"moe source: {attr} launch with undetermined MUL_ROUTED_WEIGHT ({mrw!r}); not bound"[:200])
                me.c.x03_expected = getattr(me.c, "x03_expected", 0) + 1
            return r
        return launch

    # -- declaration -------------------------------------------------------------------------------------------------------
    def declaration_classes(self) -> list[dict]:
        return [{"class": "moe_block_stream", "how": "bound-as-leaf", "covers_families": COVERS_FAMILIES, "rule": None, "count": None,
                 "bytes": self.stats["bytes"], "tensors": self.stats["launches"], "layers": self.stats["layers"], "tensors_per_layer": TENSORS_PER_LAYER,
                 "plane_flushes": self.stats["plane_flushes"],
                 "evidence": ("MOE-03 source: wraps vllm._custom_ops.topk_softmax / moe_sum and fused_moe.invoke_fused_moe_triton_kernel (also "
                              "at experts.triton_moe); per MoE layer per step router_w, router_ids, up, act, down, sum "
                              "snapshotted into the packed step stream (same run root) right after each kernel; [FLUSH-03] the up plane is "
                              "flushed off the device right after staging (its workspace is the block's own w2 output: cache1/cache3 alias) -- "
                              f"{self.stats['plane_flushes']} flushes; identity_check requires exactly "
                              f"{TENSORS_PER_LAYER} per layer per step; value correspondence: verity_vllm/ops/moe_kernel_xcheck.py (cp-moe, "
                              f"registry.moe Definitions bit-exact vs the same kernels); shapes seen {self.stats['shapes']}")}]


__all__ = ["MoeBlockSource", "moe_layers_of", "MOE_MODULE_CLASSES", "TENSORS_PER_LAYER", "COVERS_FAMILIES"]
