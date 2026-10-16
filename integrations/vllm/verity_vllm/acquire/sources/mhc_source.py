"""The mHC values a capture map's decoder layers and model compute, bound on one rank under the map's names (`commit.binder` `mhc`,
`verity_vllm.commit.capture_map`).

DeepSeek-V4's residual between layers is mHC's HC streams ([M, HC, H]), and the pinned vLLM computes every mix in four functions
of `vllm.model_executor.kernels.mhc.tilelang` that its model module imports and calls by their call-time globals
(vllm/models/deepseek_v4/nvidia/model.py): `DeepseekV4DecoderLayer.forward` calls `mhc_pre_broadcast_tilelang` on the embedding at
the first layer's attention side and `mhc_fused_post_pre_tilelang` at every other attention side and every ffn side, each with the
side's `hc_attn_fn` or `hc_ffn_fn`; `DeepseekV4Model.forward` calls `mhc_post_tilelang` after the last layer, then
`hc_head_fused_kernel_tilelang` with `hc_head_fn`.  None of them is a module.  This source wraps each in the module of the forward
that calls it and, on every call at a committer step, binds the call's returns (the residual, post mix, comb mix and sublayer input;
the post's and the head's one tensor) under the names of the map's taps at that function, then flushes them, since a later kernel
may write into the sublayer input.

A call belongs to the module and side whose `<side>_fn` parameter it was passed (layer i's `hc_attn_fn` is layer i's attention
side); the post passes none and is the model's.  `mhc_fused_post_pre_tilelang` takes its FMA path at `num_tokens <= 16`
(kernels/mhc/tilelang.py:518) and its TF32 path above, and the map's taps at it are on the hc routes: a call binds the taps of the
route the map's `hc` decision (`route_site`) gives for the call's own token count, and the route source holds the step's names to
the route it recorded at the decoder layer.  A call whose `fn` is no side's parameter fails the step.  Taps the map marks
`target_only` are bound on the target only (`route_source.on_target`).  `capture-map ids --binder mhc --names` prints what a step
binds.
"""
from __future__ import annotations

import inspect
import sys
from typing import Any, Mapping

import torch

from verity_vllm.commit import capture_map as CM
from verity_vllm.engine.hooks import Hooks

BINDER = "mhc"
PRE, POST_PRE, POST, HEAD = ("mhc_pre_broadcast_tilelang", "mhc_fused_post_pre_tilelang", "mhc_post_tilelang",
                             "hc_head_fused_kernel_tilelang")
FUNCTIONS = (PRE, POST_PRE, POST, HEAD)
DECISION = "hc"


def binds(cmap: Mapping[str, Any]) -> bool:
    return any((t.get("commit") or {}).get("binder") == BINDER for t in cmap["taps"])


def function_of(tap: Mapping[str, Any]) -> str:
    """The function whose return the tap is: its site's target after the path."""
    return tap["site"]["target"].rsplit(":", 1)[-1]


class MhcSource:
    short_name = "mhc"

    def __init__(self, com: Any, routes: Any, model: Any) -> None:
        self.c, self.cmap, self.env, self.on_target = com, routes.cmap, routes.env, routes.on_target
        taps = [t for t in self.cmap["taps"] if (t.get("commit") or {}).get("binder") == BINDER]
        unknown = sorted({function_of(t) for t in taps} - set(FUNCTIONS))
        if unknown:
            raise ValueError(f"capture map {self.cmap.get('id')}: the mhc source wraps no {unknown}")
        self.hc = next((s for s in CM.route_sites(self.cmap) if s.get("decision") == DECISION), None)
        if any(t.get("route") for t in taps) and (self.hc is None or self.hc["answer"] in ("called", "return", "none_returned")):
            raise ValueError(f"capture map {self.cmap.get('id')}: mhc taps on routes, and no {DECISION!r} decision on the token count")
        mods = dict(model.named_modules())
        #: (function, side, layer) -> [(return index, route, committed name)], the target_only taps on the target only
        self.names: dict[tuple[str, str, int | None], list[tuple[int, str | None, str]]] = {}
        #: id(<side>_fn) -> (side, layer); a function with no fn parameter -> its one (side, layer)
        self.owners: dict[int, tuple[str, int | None]] = {}
        self.unowned: dict[str, tuple[str, int | None]] = {}
        self.homes: list[Any] = []
        self.per_route: dict[str | None, list[Mapping[str, Any]]] = {}
        for t in taps:
            f, side = function_of(t), t["op"].rsplit(".", 1)[-1]
            layers = CM.tap_layers(self.cmap, t, self.env)
            for i in layers if layers is not None else [None]:
                owner = mods.get(CM.at_layer(t["module"], i))
                if owner is None:
                    raise ValueError(f"capture map {self.cmap.get('id')}: the model has no {CM.at_layer(t['module'], i)}")
                fn = getattr(owner, f"{side}_fn", None)
                if isinstance(fn, torch.Tensor):
                    self.owners[id(fn)] = (side, i)
                else:
                    self.unowned[f] = (side, i)
                home = sys.modules.get(inspect.unwrap(type(owner).forward).__module__)
                if not callable(getattr(home, f, None)):
                    raise ValueError(f"capture map {self.cmap.get('id')}: the module of {type(owner).__name__}.forward has no {f}")
                if home not in self.homes:
                    self.homes.append(home)
                if self.on_target or not t.get("target_only"):
                    self.names.setdefault((f, side, i), []).append((int(t["site"]["output"]), t.get("route"),
                                                                     CM.at_layer(t["commit"]["name"], i)))
            if self.on_target or not t.get("target_only"):
                self.per_route.setdefault(t.get("route"), []).append(t)
        self._hooks = Hooks()
        self.stats: dict[str, Any] = {}
        self.begin_run()

    # -- install / steps ------------------------------------------------------------------------------------------------------
    def install(self) -> None:
        used = {k[0] for k in self.names}
        for home in self.homes:
            for f in FUNCTIONS:
                if f in used and callable(getattr(home, f, None)):
                    self._hooks.wrap(home, f, self._call(f))

    def uninstall(self) -> None:
        self._hooks.uninstall()

    def on_step_begin(self) -> None:
        pass

    def begin_run(self) -> None:
        self.stats = {"homes": [h.__name__ for h in self.homes], "on_target": self.on_target, "calls": 0, "bound": 0, "unattributed": 0,
                      "by_route": {}}

    def declaration_classes(self) -> list[dict]:
        return []

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        """The taps this source binds, as bytes per token: those on no route, and the larger of the hc routes' (a step takes one)."""
        free = self.cmap.get("free_symbols", {})

        def size(taps: list[Mapping[str, Any]]) -> tuple[int, int]:
            per_token = n = 0
            for t in taps:
                words = 1
                for d in t["shape"][1:]:
                    words *= int(CM.dim(d, self.env, free))
                k = len(CM.tap_layers(self.cmap, t, self.env) or [None])
                per_token += words * torch.empty((), dtype=getattr(torch, t["dtype"])).element_size() * k
                n += k
            return per_token, n

        base = size(self.per_route.get(None, []))
        routed = max((size(ts) for r, ts in self.per_route.items() if r is not None), default=(0, 0))
        return {"per_token_bytes": base[0] + routed[0], "per_seq_bytes": 0, "tensors_per_step": base[1] + routed[1]}

    # -- binding ----------------------------------------------------------------------------------------------------------------
    def route(self, residual: torch.Tensor) -> tuple[str, ...]:
        """The hc routes of a `mhc_fused_post_pre_tilelang` call on `residual`: its token count as the function reads it
        (`residual.view(-1, hc_mult, hidden_size).shape[0]`) through the map's decision."""
        m = int(residual.numel()) // (int(residual.shape[-2]) * int(residual.shape[-1]))
        yes = bool(CM.evaluate(self.hc["answer"], {**self.env, "M": m}))
        return tuple(self.hc["true"] if yes else self.hc.get("false", ()))

    def bind(self, f: str, a: Mapping[str, Any], out: Any, job: Any) -> None:
        """One call of `f` with bound arguments `a` and return `out` at the committer's step `job`."""
        self.stats["calls"] += 1
        fn = a.get("fn")
        owner = self.owners.get(id(fn)) if fn is not None else self.unowned.get(f)
        if owner is None:
            self.stats["unattributed"] += 1
            self.c.errors.append(f"mhc: {f} called with an fn that is no side's parameter of the model"[:200])
            self.c.x03_expected = getattr(self.c, "x03_expected", 0) + 1
            return
        outs = out if isinstance(out, tuple) else (out,)
        routes = self.route(a["residual"]) if f == POST_PRE and self.hc is not None else ()
        for port, r, name in self.names.get((f, *owner), ()):
            if r is None or r in routes:
                self.c._acquire(name, outs[port], job)
                self.stats["bound"] += 1
                self.stats["by_route"][r or "-"] = self.stats["by_route"].get(r or "-", 0) + 1
        flush = getattr(self.c, "flush_after_plane", None)
        if flush is not None:
            flush(BINDER)

    def _call(self, f: str) -> Any:
        me = self

        def make(orig: Any) -> Any:
            sig = inspect.signature(orig)

            def call(*args: Any, **kwargs: Any) -> Any:
                out = orig(*args, **kwargs)
                job = getattr(me.c, "_cur", None)
                if job is not None:
                    me.bind(f, sig.bind(*args, **kwargs).arguments, out, job)
                return out
            return call
        return make


__all__ = ["MhcSource", "BINDER", "FUNCTIONS", "binds", "function_of"]
