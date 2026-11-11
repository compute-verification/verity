"""The router softmax (`query.router_softmax`) as an acquisition source: per MoE layer and step, the softmax words of every token row,
committed as `moe/L<k>/router_softmax` in the same step stream and run root as the MoE block's planes.  Opt-in (commit_delta
`--router-tap-so`, the row's ROUTER_TAP=1), never the record.

The layers are the required manifest's `router_softmax` identities.  Forward pre-hooks on the model's MoE modules (`moe_source.moe_layers_of`,
the block source's attribution) mark the layer that runs.  Inside a committed step and a manifest layer, `torch.ops._moe_C.topk_softmax`
is replaced by the tapped op of `ops/pod_router_tap.sh`, `verity_router_tap.topk_softmax_tap`: vLLM's `topkGating`, launched the same way,
whose threads also store the row's max, exponentials, reciprocal sum and probabilities (and under renormalize the selected probabilities
and the scale) into a float32 [rows, words] buffer.  Everywhere else the installed op runs.  The tapped op computes exactly the installed
op's outputs and every word equals the ordered router's committed value in the circuit (`properties.router_tap_exactness`).  The patch goes
through `engine.hooks` and `uninstall` undoes it.  A launch inside a manifest layer whose geometry is not the manifest's, or that carries a
correction bias (not in the circuit's router), runs the installed op and is a collector error.  vLLM's padding mask (`is_padding`, under
VLLM_MOE_SKIP_PADDING) is passed through: it changes a padding row's ids, never the softmax, and a padded launch has more rows than the
manifest's router Calls, which the committer's identity check refuses.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Mapping

from verity_vllm.acquire.router_tap_src.build_ext import NAME as TAP_NS
from verity_vllm.acquire.router_tap_src.build_ext import TAP_VERSION
from verity_vllm.acquire.sources.moe_source import moe_layers_of
from verity_vllm.engine import hooks
from verity_vllm.query import router_softmax as RS


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RouterSoftmaxSource:
    short_name = "router_softmax"

    def __init__(self, committer: Any, model: Any, layers: Mapping[int, Mapping[str, Any]], so_path: str) -> None:
        """`layers`: MoE layer -> its manifest geometry (`query.router_softmax.manifest_layers`); `so_path`: the tap build."""
        import torch

        if not layers:
            raise ValueError("router softmax: no MoE layer to tap")
        if not so_path:
            raise RuntimeError("router softmax: no tap build was given (--router-tap-so, ops/pod_router_tap.sh)")
        self.torch, self.c, self.model = torch, committer, model
        self.layers = {int(k): dict(v) for k, v in layers.items()}
        torch.ops.load_library(so_path)
        self.ns = getattr(torch.ops, TAP_NS)
        version = int(self.ns.tap_version())
        if version != TAP_VERSION:
            raise RuntimeError(f"router softmax: {so_path} is tap version {version}, this source launches version {TAP_VERSION}")
        self.so_path = so_path
        self.modules = moe_layers_of(model) if model is not None else {}
        self.words_per_token = sum(int(g["width"]) for g in self.layers.values())
        self._hooks = hooks.Hooks()
        self._handles: list[Any] = []
        self.cur_layer: int | None = None
        self._held: list[Any] = []
        self.stats: dict[str, Any] = {"tensors": 0, "bytes": 0, "rows": 0, "layers": len(self.layers), "errors": 0, "so": so_path,
                                      "so_sha256": _file_sha256(so_path)}
        self.active = False

    # ------------------------------------------------------------------------------------------------------------ install / uninstall
    def install(self) -> None:
        missing = sorted(set(self.layers) - set(self.modules.values()))
        if missing:
            raise RuntimeError(f"router softmax: the manifest's MoE layers {missing[:4]} are not MoE modules of the model")
        mods = dict(self.model.named_modules())
        for name, k in self.modules.items():
            self._handles.append(mods[name].register_forward_pre_hook(self._enter(k)))
            self._handles.append(mods[name].register_forward_hook(self._leave(k), always_call=True))
        self._hooks.patch(self.torch.ops._moe_C, "topk_softmax", self._wrap(self.torch.ops._moe_C.topk_softmax))
        self.active = True

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._hooks.uninstall()
        self.cur_layer = None
        self._held.clear()
        self.active = False

    def on_step_begin(self) -> None:
        self._held.clear()
        self.cur_layer = None

    def _enter(self, k: int) -> Any:
        def pre(_module: Any, _args: tuple) -> None:
            self.cur_layer = k

        return pre

    def _leave(self, k: int) -> Any:
        def post(_module: Any, _args: tuple, _out: Any) -> None:
            if self.cur_layer == k:
                self.cur_layer = None

        return post

    # ------------------------------------------------------------------------------------------------------------ acquisition
    def error(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.c.errors.append(msg[:300])

    def _wrap(self, orig: Any) -> Any:
        me, torch = self, self.torch

        def topk_softmax(topk_weights: Any, topk_indices: Any, token_expert_indices: Any, gating_output: Any, renormalize: bool = False,
                         bias: Any = None, is_padding: Any = None) -> None:
            job, layer = me.c._cur, me.cur_layer
            if job is None or layer is None or layer not in me.layers:
                return orig(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize, bias, is_padding)
            g = me.layers[layer]
            E, K = int(gating_output.shape[-1]), int(topk_weights.shape[-1])
            if (E, K, bool(renormalize)) != (g["experts"], g["topk"], g["renormalize"]) or bias is not None:
                me.error(f"router softmax: layer {layer} launched topk_softmax with E={E} TOPK={K} renormalize={bool(renormalize)} "
                         f"bias={bias is not None}; the manifest's router is E={g['experts']} TOPK={g['topk']} renormalize={g['renormalize']} "
                         "without bias")
                return orig(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize, bias, is_padding)
            t0 = time.perf_counter()
            rows = gating_output.numel() // E
            tap = torch.empty(rows, int(g["width"]), dtype=torch.float32, device=gating_output.device)
            me.ns.topk_softmax_tap(topk_weights, topk_indices, token_expert_indices, gating_output, bool(renormalize), None, is_padding, tap)
            me._held.append(tap)
            me.c._acquire(f"moe/L{layer}/{RS.MEMBER}", tap, job)
            n = int(tap.numel()) * 4
            me.stats["tensors"] += 1
            me.stats["bytes"] += n
            me.stats["rows"] += rows
            spans = getattr(me.c, "spans", None)
            if spans is not None:
                spans.add("acquisition", time.perf_counter() - t0, step=job.step, nbytes=n)
            return None

        return topk_softmax

    # ------------------------------------------------------------------------------------------------------------ the committer's view
    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        return {"per_token_bytes": 4 * self.words_per_token, "per_seq_bytes": 0, "tensors_per_step": len(self.layers)}

    def declaration_classes(self) -> list[dict]:
        return [{"class": RS.FAMILY, "how": "bound-as-leaf", "rule": None, "count": self.stats["tensors"], "bytes": self.stats["bytes"],
                 "tensors": self.stats["tensors"], "rows": self.stats["rows"], "errors": self.stats["errors"],
                 "evidence": (f"router-softmax source: `moe/L<k>/{RS.MEMBER}` of {len(self.layers)} MoE layers ({self.words_per_token} words "
                              f"per token row); tap build {self.so_path} (sha256 {self.stats['so_sha256']}); outputs bit-identical to the "
                              "installed topk_softmax and every word == the ordered router's committed value in the IR: "
                              "verity_vllm.properties.router_tap_exactness")}]


def attach(com: Any, so: str | None, manifest: Mapping[str, Any] | None, required_classes: tuple[str, ...], *,
           explicit_classes: bool = False, model: Any = None) -> tuple[tuple[str, ...], RouterSoftmaxSource | None]:
    """Attach the source to `com` when `so` (the tap build) is given; the required manifest must name `router_softmax` identities, else
    the Commit is refused (exit 3).  Returns the required classes with `router_softmax` added unless they were given explicitly."""
    if not so:
        return required_classes, None
    layers = RS.manifest_layers(manifest)
    if not layers:
        print("[delta] REFUSE: --router-tap-so but the required-value manifest names no router_softmax identity (manifest build --router-softmax)",
              flush=True)
        raise SystemExit(3)
    src = RouterSoftmaxSource(com, model if model is not None else getattr(com, "model", None), layers, so)
    com.attach_source(src)
    print(f"[delta] router-softmax source attached: {len(layers)} MoE layers, {src.words_per_token} words per token, tap {so} -> committer "
          f"{com.name}", flush=True)
    if not explicit_classes and RS.FAMILY not in required_classes:
        return (*required_classes, RS.FAMILY), src
    return required_classes, src


__all__ = ["RouterSoftmaxSource", "attach"]
