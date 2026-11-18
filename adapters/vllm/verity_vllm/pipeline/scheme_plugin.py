"""Schemes turned on by name inside vLLM's own OpenAI server (``vllm serve``), each the scheme its name registers in the
one table (`verity.protocols.compose.schemes.get`), as ``VERITY_PROTOCOLS`` lists them, with ``VERITY_PROTOCOLS_CONFIG``.

`register`, a vLLM general plugin, is the engine's side, in the engine core's process: the adapter's memory hook
(`engine.kv_blocks`) on, and each engine core's schemes attached on the process's exchange (``process`` `hooks.ENGINE`)
once its model is loaded and before its scheduler builds the KV pool, the exchange answered with what vLLM's
configuration declares, the admission hook (`engine.admission.register`, keyed by the client's X-Request-Id) on its
scheduler, and what each scheme holds released (`hooks.Releases`) when the engine core shuts down. A scheme's
`hooks.RefuseInput` there aborts the input, its reason in the engine core's log.

vLLM loads its general plugins in each of its processes, so `register` imports vLLM only once ``VERITY_PROTOCOLS`` is
set, and the engine's side acts only in a process that builds an engine core.
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from typing import Any

from verity import hooks as H
from verity_vllm import messages, settings

__all__ = ["GOAL", "attach", "config", "declared", "register"]

#: the goal whose unbuilt parts raise `NotImplementedError` here
GOAL = "vllm/scheme-on-in-vllm-server"

_PLUGIN: list[Any] = []


def config(environ: Mapping[str, str] | None = None) -> tuple[list[str], dict[str, Any]]:
    """``VERITY_PROTOCOLS``'s names and ``VERITY_PROTOCOLS_CONFIG``'s object (``feed`` required with a name); no names,
    nothing."""
    names = [n for n in settings.get("VERITY_PROTOCOLS", environ) if n]
    if not names:
        return [], {}
    cfg = settings.get("VERITY_PROTOCOLS_CONFIG", environ)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("feed"), str) or not cfg["feed"]:
        raise ValueError("VERITY_PROTOCOLS_CONFIG: a JSON object with feed, the directory the server's front and its engine "
                         f"share, and each scheme's configuration under its name, not {cfg!r:.80}")
    return names, cfg


def context(process: str, name: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Scheme ``name``'s ``ctx`` in ``process`` (`hooks.FRONT`)."""
    return {"process": process, "feed": cfg["feed"], "config": dict(cfg.get(name) or {})}


def declared(vllm_config: Any) -> H.Declared:
    """What vLLM's configuration declares, the properties `engine.build.build_engine` declares of its own engine."""
    from verity_vllm.engine import served

    mc, pc = vllm_config.model_config, vllm_config.parallel_config
    return H.Declared("", {
        "model": mc.model, "execution": None, "eager": bool(mc.enforce_eager),
        "world": pc.tensor_parallel_size * pc.pipeline_parallel_size,
        "dtype": str(mc.dtype).removeprefix("torch."), "quantization": mc.quantization,
        "prefix_cache": bool(vllm_config.cache_config.enable_prefix_caching),
        "kernels": served.kernels({"quantization": mc.quantization}),
        "async_steps": bool(getattr(vllm_config.scheduler_config, "async_scheduling", False))})


def attach(exchange: messages.Exchange, vllm_config: Any, names: list[str], cfg: Mapping[str, Any]) -> list[Any]:
    """Each scheme ``names`` on ``exchange`` in the engine, then the exchange answered (`messages.Exchange.answer`); what
    each attach returned, in order. A tensor-parallel engine is refused by `GOAL`."""
    from verity.protocols.compose import schemes

    pc = vllm_config.parallel_config
    if pc.tensor_parallel_size * pc.pipeline_parallel_size != 1:
        raise NotImplementedError(f"{GOAL}: schemes inside vLLM's server on one GPU, not tensor_parallel_size "
                                  f"{pc.tensor_parallel_size} × pipeline_parallel_size {pc.pipeline_parallel_size}")
    got = [schemes.get(n).attach(exchange, context(H.ENGINE, n, cfg)) for n in names]
    if fixed := sorted({r.ranges for _, r in exchange.requests if isinstance(r, H.FixRanges)}):
        if len(fixed) > 1:
            raise RuntimeError(f"schemes fix the KV pool at {fixed} blocks: one size")
        vllm_config.cache_config.num_gpu_blocks_override = fixed[0]
    exchange.answer(declared(vllm_config))
    return got


def aborts(hook: Any, log: Callable[[str], Any] = lambda s: print(s, file=sys.stderr, flush=True),
           status: Any = None) -> Callable[[str, str], None]:
    """The engine's answer to a `hooks.RefuseInput`: input ``input``'s requests in ``hook``'s scheduler finished with
    ``status`` (vLLM's ``FINISHED_ABORTED`` by default), as vLLM's server finishes one whose client went away, and ``why``
    logged."""
    def refuse(input: str, why: str) -> None:
        rids = [rid for table in (hook.held, hook.live) for rid, i in table.items() if i == input]
        log(f"verity: input {input} refused ({len(rids)} request(s) aborted): {why}")
        if rids:
            if status is None:
                from vllm.v1.request import RequestStatus
                hook.sched.finish_requests(rids, RequestStatus.FINISHED_ABORTED)
            else:
                hook.sched.finish_requests(rids, status)
    return refuse


def register(environ: Mapping[str, str] | None = None, core: type | None = None) -> None:
    """The engine's side (the module docstring), as a vLLM general plugin with ``VERITY_PROTOCOLS`` set; ``core`` is
    vLLM's ``EngineCore`` unless given. Idempotent."""
    names, cfg = config(environ)
    if not names or _PLUGIN:
        return
    from verity_vllm.engine import admission, kv_blocks
    from verity_vllm.engine.hooks import Hooks

    if core is None:
        from vllm.v1.engine.core import EngineCore as core
    hooks = Hooks(tag="scheme-plugin")
    held: list[Any] = []

    def initialize(orig: Callable[..., Any]) -> Callable[..., Any]:
        def _initialize_kv_caches(self: Any, vllm_config: Any, *a: Any, **kw: Any) -> Any:
            held.extend(x for x in attach(messages.current(), vllm_config, names, cfg) if hasattr(x, "release"))
            return orig(self, vllm_config, *a, **kw)
        return _initialize_kv_caches

    def stop(orig: Callable[..., Any]) -> Callable[..., Any]:
        def shutdown(self: Any, *a: Any, **kw: Any) -> Any:
            while held:
                held.pop(0).release()
            return orig(self, *a, **kw)
        return shutdown

    hooks.wrap(core, "_initialize_kv_caches", initialize)
    hooks.wrap(core, "shutdown", stop)
    detach = kv_blocks.attach()
    admission.register(lambda exchange: None, refuse=aborts)
    _PLUGIN.append((hooks, detach))
