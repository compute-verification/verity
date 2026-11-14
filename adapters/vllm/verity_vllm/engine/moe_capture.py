"""The fused-MoE launches of chosen requests, taken in an engine that replays them at their served step shapes.

`install`, run in every worker (``engine.collective_rpc``), puts a `Capture` on the worker's model runner for the target
requests and the named MoE modules of its model. At every step that schedules a target it copies, for each of those
modules, the target's rows of the block's launches to the host, right after each launch (the block's workspace is the
next launch's):

    grouped_topk(...) -> (weights, ids)                              router_ids    i32 [TOPK]        the routed experts
    invoke_fused_moe_triton_kernel(A, B, C!, A_scale, ..., mul_routed_weight=False, top_k=TOPK, config)
                                                                     in_q          e4m3 [H]          A
                                                                     in_scale      f32 [H/G]         A_scale
                                                                     up            bf16 [TOPK, 2I]   C
    invoke_fused_moe_triton_kernel(A, B, C!, A_scale, _, topk_weights, ..., mul_routed_weight=True, top_k=1, config)
                                                                     act_q         e4m3 [TOPK, I]    A
                                                                     act_scale     f32 [TOPK, I/G]   A_scale
                                                                     topk_weights  f32 [TOPK]        the routed weights
                                                                     down          bf16 [TOPK, H]    C

each a token's rows (`KEPT`, by `acquire.sources.moe_source`'s plane names; a pair-major tensor's rows by pair), with
each launch's tile (its ``config``). `take`, run in every worker, writes each target's rows under
``<directory>/rank<r>/`` and returns their layout and sha256s. `uninstall`, run in every worker after `take`, takes the
capture off: its patches are `engine.hooks`' under `TAG`. `spans` holds every rank's pieces to the replay's own step
trace, `write` writes the record (`CAPTURE`) beside the ranks' files, and `read` reads one request's rows of one module
at one rank back from it.

The launch sites are the pinned vLLM's Python ones, which `acquire.sources.moe_source` wraps too (`FUSED_MOE_SITE`,
`TRITON_MOE_SITE`, `ROUTER_SITE`, `launch_arg`). A step vLLM replays from a CUDA graph runs none of them, so the
capture follows vLLM's capture of its graphs (`GRAPH_SITE`, `engine.graph_slots`): in each graph's capture a launch of
a named module copies its tensors, every row, into static slots, a copy the graph replays at every step, and after a
replayed step's forward the targets' rows are cut from the slots' first rows, with the tiles the capture launched at
(the padded size's). The step's graph is read from the runner's dispatch (`DISPATCH`), and each piece's is kept
(``dispatch``: its mode and padded tokens, ``NONE`` for a step that ran its launches in Python). So on an engine that
captures graphs the capture is installed before it does: `arm`, in a process before it builds its worker, installs
on the worker right before it warms up (`WORKER_SITE`). Installed later, it refuses the first replayed step of a
target, and so does a forward compiled past the launcher, rather than keep less.
Refused too: a module whose launches carry no ``A_scale`` (experts that are not block FP8), a router other than the
grouped one, a step that samples draft tokens or runs in micro-batches, and a runner whose module has no
``sort_batch_req_ids`` (the runner is vLLM's V2 model runner, `v1.worker.gpu.model_runner`, whose order the step's
rows are read in before the forward and held to its input batch after it).

What is kept is served values: nothing here logs or prints them, a refusal names shapes and rows only, and they are
written only under the directory `take` is given.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.engine import graph_slots, hooks, runner_step, step_trace

__all__ = ["CAPTURE", "DISPATCH", "FUSED_MOE_SITE", "GRAPH_SITE", "KEPT", "LAUNCH_ARGS", "ROUTER_SITE", "TAG",
           "TRITON_LAUNCH", "TRITON_MOE_SITE", "WORKER_SITE", "Capture", "arm", "disarm", "install", "launch_arg", "read",
           "spans", "take", "uninstall", "write"]

#: a capture's record, beside its ranks' files
CAPTURE = "capture.json"
#: the tag of a capture's patches of the launch sites (``hooks.live(TAG)``)
TAG = "moe-capture"
#: the one Triton launcher, by the name both of its modules hold it under
TRITON_LAUNCH = "invoke_fused_moe_triton_kernel"
#: the launcher's module, whose functional path looks the name up in its globals: (module, name)
FUSED_MOE_SITE = ("vllm.model_executor.layers.fused_moe.fused_moe", TRITON_LAUNCH)
#: the module of the Triton experts, which imported the name
TRITON_MOE_SITE = ("vllm.model_executor.layers.fused_moe.experts.triton_moe", TRITON_LAUNCH)
#: the grouped router, at the name its caller looks up when it routes
ROUTER_SITE = ("vllm.model_executor.layers.fused_moe.router.grouped_topk_router", "grouped_topk")
#: the class through whose ``capture`` vLLM captures every graph: (module, name)
GRAPH_SITE = ("vllm.v1.worker.gpu.cudagraph_utils", "CudaGraphManager")
#: the dispatch of a step to a graph, by the name the runner's own module holds it under
DISPATCH = "dispatch_cg_and_sync_dp"
#: the worker, whose ``compile_or_warm_up_model`` captures its runner's graphs: (module, name)
WORKER_SITE = ("vllm.v1.worker.gpu_worker", "Worker")
#: the launcher's positional index of each argument read from a launch, after (A, B, C):
#: invoke_fused_moe_triton_kernel(A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids,
#: num_tokens_post_padded, mul_routed_weight, top_k, config, ...)
LAUNCH_ARGS = {"A_scale": 0, "topk_weights": 2, "mul_routed_weight": 6, "top_k": 7, "config": 8}
#: what a request's file of one module holds, in this order: each a token's rows over the request's pieces
KEPT = ("in_q", "in_scale", "router_ids", "up", "act_q", "act_scale", "topk_weights", "down")

#: the capture on each runner of this process, by ``id(runner)``
_ON: dict[int, Capture] = {}
#: the slots of the captures taken off: a graph captured with a capture on replays its copies for as long as it
#: lives, so the tensors they write stay allocated with the process
_PARKED: list[graph_slots.Slots] = []
#: `arm`'s patch of the worker, where this process is armed
_ARMED: list[hooks.Hooks] = []


def launch_arg(a: Sequence[Any], k: Mapping[str, Any], name: str) -> Any:
    """The argument ``name`` (of `LAUNCH_ARGS`) of one launch called as ``launcher(A, B, C, *a, **k)``: its keyword, or
    its position; None where the call gives neither."""
    at = LAUNCH_ARGS[name]
    return k[name] if name in k else (a[at] if len(a) > at else None)


def _bits(t: Any) -> np.ndarray:
    """A copy of a tensor's elements on the host, as little-endian unsigned words of its element size."""
    import torch
    word = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(t.element_size())
    if word is None or t.is_complex():
        raise TypeError(f"a {t.dtype} tensor has no words of 1, 2, 4 or 8 bytes to keep")
    return t.contiguous().to("cpu", copy=True).view(word).numpy().view(f"<u{t.element_size()}")


def _sort(runner: Any) -> Callable[..., Sequence[str]]:
    """The batch order of ``runner``'s module, as it is now (the runner looks it up at each step)."""
    sort = getattr(sys.modules.get(type(runner).__module__), "sort_batch_req_ids", None)
    if not callable(sort) or not isinstance(getattr(runner, "decode_query_len", None), int):
        raise TypeError(f"{type(runner).__module__} has no sort_batch_req_ids, or its runner no decode_query_len: the "
                        "capture reads a step's rows in the V2 model runner's order")
    return sort


class Capture(runner_step.Part):
    """One runner's capture: at each step that schedules a target, the target's rows of each named module's launches,
    as `KEPT` names them, with the launches' tiles; ``pieces`` is each target's (first position, tokens) a step and
    ``dispatch`` the (mode, padded tokens) each of those steps ran at."""

    rank = runner_step.Rank.OBSERVE

    def __init__(self, runner: Any, targets: Sequence[str], modules: Mapping[str, Any]) -> None:
        self.runner, self.targets, self.names = runner, frozenset(targets), tuple(sorted(modules))
        self.pieces: dict[str, list[list[int]]] = {r: [] for r in self.targets}
        self.dispatch: dict[str, list[list[Any]]] = {r: [] for r in self.targets}
        self.kept: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
            r: {m: {f: [] for f in KEPT} for m in self.names} for r in self.targets}
        self.tiles: dict[str, dict[str, list[list[dict[str, int]]]]] = {r: {m: [] for m in self.names} for r in self.targets}
        #: field -> the torch dtype its words are the bits of, as the first launch gave it
        self.of: dict[str, str] = {}
        #: this step's targets: request -> (first row, end row, first position), and the step's rows; None between steps
        self.rows: dict[str, tuple[int, int, int]] | None = None
        self.total = 0
        #: the named module whose forward is running, and what each has launched this step
        self.at: str | None = None
        self.seen: dict[str, dict[str, Any]] = {}
        #: this step's dispatch: (mode, padded tokens, tokens, micro-batches); None before it
        self.step: tuple[str, int, int, int] | None = None
        #: (mode, padded tokens) -> module -> what its launches gave that graph's capture: each field's shape a token,
        #: its rows, TOPK and tiles; the fields' rows are in the slots
        self.graphs: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        self.slots = graph_slots.Slots(dropped=self.graphs.clear)
        self.hooks = hooks.Hooks(tag=TAG)
        graphs = importlib.import_module(GRAPH_SITE[0])
        sites: list[tuple[Any, str, Callable[..., Any]]] = [
            *((importlib.import_module(site), name, wrapper) for (site, name), wrapper in
              ((FUSED_MOE_SITE, self._launch), (TRITON_MOE_SITE, self._launch), (ROUTER_SITE, self._router))),
            (getattr(graphs, GRAPH_SITE[1], graphs), "capture", self.slots.capture_wrapper(self._mine)),
            (sys.modules[type(runner).__module__], DISPATCH, self._dispatched)]
        for owner, name, _ in sites:
            if not callable(getattr(owner, name, None)):
                raise TypeError(f"{owner.__name__} holds no {name}: the capture reads the pinned vLLM's launch sites, "
                                "its capture of its graphs and its dispatch of a step to one")
        self.handles = [h for name in self.names for h in (
            modules[name].register_forward_pre_hook(self._enter(name)),
            modules[name].register_forward_hook(self._leave, always_call=True))]
        for owner, name, wrapper in sites:
            self.hooks.wrap(owner, name, wrapper)
        self.remove = runner_step.add(runner, self)

    def off(self) -> None:
        """Take the capture off its runner's step, the launch sites and its modules. The rows stay with it, and the
        slots its graphs copy into stay allocated."""
        self.remove()
        self.hooks.uninstall()
        for h in self.handles:
            h.remove()
        self.handles = []
        if self.slots.slots:
            _PARKED.append(self.slots)

    # -- the step's graph --------------------------------------------------------------------------------------------------
    def _mine(self, manager: Any) -> bool:
        """Whether ``manager`` is the runner's own: a speculator's graphs are captured through the same class."""
        return manager is getattr(self.runner, "cudagraph_manager", None)

    def _dispatched(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        def dispatch(manager: Any, num_reqs: Any, num_tokens: Any, *a: Any, **k: Any) -> Any:
            out = orig(manager, num_reqs, num_tokens, *a, **k)
            if self.rows is not None and self._mine(manager):
                desc = out[0]
                self.step = (str(desc.cg_mode.name), int(desc.num_tokens), int(num_tokens), int(getattr(desc, "num_ubatches", 1)))
            return out
        return dispatch

    def _replayed(self, mode: str, pad: int, rows: Mapping[str, tuple[int, int, int]]) -> dict[str, dict[str, Any]]:
        """What a step replayed from the ``mode`` graph of ``pad`` tokens launched, as a step that ran its launches
        in Python leaves it in ``seen``: each named module's fields, the targets' rows cut from the slots' first rows,
        and the tiles its graph's capture launched at."""
        entries = self.slots.entries(mode, pad)
        if entries is None:
            raise TypeError(f"the step replays the {mode} CUDA graph of {pad} tokens, whose capture this capture did not "
                            "follow: it is installed before the engine captures its graphs (`arm`), or the engine is "
                            "an eager one")
        slot = {name: entry for entry in entries for name in (entry[1],)}
        out: dict[str, dict[str, Any]] = {}
        for name in self.names:
            meta = self.graphs.get((mode, pad), {}).get(name, {})
            lacks = [f for f in KEPT if f not in meta or f"{name}/{f}" not in slot]
            if lacks:
                raise TypeError(f"{name} gave no {', '.join(lacks)} in the capture of the {mode} CUDA graph of {pad} "
                                "tokens: another router or kernel path is not captured")
            seen: dict[str, Any] = {"tile13": meta["tile13"], "tile2": meta["tile2"]}
            for f in KEPT:
                value = self.slots.value(slot[f"{name}/{f}"], self.total, pad)
                seen[f] = {r: _bits(value[lo:hi]).reshape(hi - lo, *meta[f]) for r, (lo, hi, _) in rows.items()}
            out[name] = seen
        return out

    # -- the step's rows ---------------------------------------------------------------------------------------------------
    def before_forward(self, step: runner_step.Step) -> None:
        """The rows each target's tokens take in this step's launches, in the runner's batch order."""
        self.rows, self.at, self.seen, self.step = None, None, {}, None
        so = step.scheduler_output
        tokens = {r: int(n) for r, n in so.num_scheduled_tokens.items()}
        if not self.targets & tokens.keys():
            return
        drafts = getattr(so, "scheduled_spec_decode_tokens", None) or {}
        if any(drafts.values()):
            raise TypeError("the step verifies draft tokens: its rows are not one a scheduled token")
        start = step_trace.step_geometry(so)[0]
        rows, at = {}, 0
        for r in _sort(step.runner)(tokens, drafts, step.runner.decode_query_len):
            if r in self.targets:
                rows[r] = (at, at + tokens[r], start[r])
            at += tokens[r]
        self.rows, self.total = rows, at

    def after_forward(self, step: runner_step.Step) -> None:
        """Hold the rows read before the forward to the step's input batch and every named module to its three launches,
        then keep the step's rows."""
        rows, self.rows = self.rows, None
        at, self.step = self.step, None
        if rows is None:
            return
        batch = getattr(getattr(step.runner, "execute_model_state", None), "input_batch", None)
        if batch is None:
            raise TypeError("the runner kept no input batch of the step: the capture holds its rows to the batch's")
        n = int(batch.num_reqs)
        first = [int(x) for x in batch.query_start_loc_np[:n + 1]]
        laid = {r: (first[i], first[i + 1], int(batch.num_computed_tokens_np[i]))
                for i, r in enumerate(batch.req_ids[:n]) if r in rows}
        if laid != rows:
            raise ValueError(f"the step laid its requests out at (first row, end row, first position) {laid}, and the "
                             f"capture read its launches at {rows}")
        if at is None:
            raise TypeError(f"the step was not dispatched through {type(step.runner).__module__}'s {DISPATCH} with the "
                            "runner's own graph manager: the capture reads there whether a step replays a CUDA graph")
        mode, pad, tokens, ubatches = at
        if tokens != self.total or ubatches != 1:
            raise TypeError(f"the step was dispatched at {tokens} tokens in {ubatches} micro-batches and scheduled "
                            f"{self.total} rows: the capture reads one launch a module of the scheduled rows")
        if self.slots.errors:
            raise TypeError(f"the capture's slots: {'; '.join(self.slots.errors[:3])}")
        got = self._replayed(mode, pad, rows) if mode in graph_slots.GRAPH_MODES else self.seen
        for name in self.names:
            seen = got.get(name, {})
            lacks = [f for f in KEPT if f not in seen]
            if lacks:
                raise TypeError(f"{name} gave no {', '.join(lacks)} at this step's Python launch sites: a forward "
                                "compiled past the launcher launches at none, and another router or kernel path is "
                                "not captured")
            for r in rows:
                for f in KEPT:
                    self.kept[r][name][f].append(seen[f][r])
                self.tiles[r][name].append([seen["tile13"], seen["tile2"]])
        for r, (lo, hi, start) in rows.items():
            self.pieces[r].append([start, hi - lo])
            self.dispatch[r].append([mode, pad])
        self.seen = {}

    # -- the launches ------------------------------------------------------------------------------------------------------
    def _enter(self, name: str) -> Callable[..., None]:
        def enter(module: Any, args: Any) -> None:
            self.at = name
            if self.slots.capturing is not None:
                self.graphs.setdefault(self.slots.capturing, {}).pop(name, None)
        return enter

    def _tapped(self) -> bool:
        """Whether a launch running now is one to read: a named module's, in a step with a target or in the capture
        of one of the runner's graphs."""
        return self.at is not None and (self.rows is not None or self.slots.capturing is not None)

    def _launches(self) -> dict[str, Any]:
        """What the module at ``self.at`` has launched in this forward: a step's, or a graph's capture's."""
        graph = self.slots.capturing
        return (self.seen if graph is None else self.graphs.setdefault(graph, {})).setdefault(str(self.at), {})

    def _leave(self, module: Any, args: Any, out: Any) -> None:
        self.at = None

    def _keep(self, seen: dict[str, Any], field: str, t: Any, pairs: bool) -> None:
        """Keep each target's rows of ``t`` as ``field``: a token's row of a token-major tensor ([M, ...]), or with
        ``pairs`` its TOPK pairs' rows, of a token-major ([M, TOPK, ...]) or a pair-major one ([M * TOPK, ...]). In a
        graph's capture every row is copied into the field's slot instead, a row a token, and the copy is the graph's."""
        m, topk = seen["m"], seen["topk"]
        shape = tuple(getattr(t, "shape", ()))
        if shape[:1] == (m,) and (not pairs or shape[1:2] == (topk,)):
            per, token = 1, shape[1:]
        elif pairs and shape[:1] == (m * topk,):
            per, token = topk, (topk, *shape[1:])
        else:
            raise TypeError(f"{self.at}'s {field} has shape {shape} at {m} rows of {topk} routed experts")
        of = str(t.dtype).rpartition(".")[2]
        if self.of.setdefault(field, of) != of:
            raise TypeError(f"{self.at}'s {field} is {of} at this launch and was {self.of[field]} at an earlier one")
        graph, captured = self.slots.capturing, graph_slots.stream_capturing(t)
        if graph is None:
            assert self.rows is not None
            if captured:
                raise TypeError(f"{self.at}'s {field} is launched into a CUDA graph outside vLLM's capture of the "
                                "runner's graphs: its rows would be the graph's and not this step's")
            seen[field] = {r: _bits(t[lo * per:hi * per]).reshape(hi - lo, *token) for r, (lo, hi, _) in self.rows.items()}
            return
        entry = self.slots.stage((self.names.index(str(self.at)), KEPT.index(field)), f"{self.at}/{field}", t.reshape(m, -1), m)
        if captured:
            self.slots.plans[graph].append(entry)
        seen[field] = tuple(int(x) for x in token)

    def _router(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        def grouped_topk(*a: Any, **k: Any) -> Any:
            out = orig(*a, **k)
            if self._tapped():
                self._routed(out[1])
            return out
        return grouped_topk

    def _routed(self, ids: Any) -> None:
        """The router's ids of one forward of the module at ``self.at``, which fix its rows and TOPK."""
        import torch
        seen, graph = self._launches(), self.slots.capturing
        if "router_ids" in seen:
            raise TypeError(f"{self.at} routed twice in one forward")
        if ids.dim() != 2 or ids.is_floating_point() or (ids.shape[0] < self.total if graph is None else ids.shape[0] != graph[1]):
            raise TypeError(f"{self.at}'s router gave ids of shape {tuple(ids.shape)} ({ids.dtype}) at a "
                            + (f"step of {self.total} rows" if graph is None else f"graph of {graph[1]} tokens"))
        seen["m"], seen["topk"] = int(ids.shape[0]), int(ids.shape[1])
        self._keep(seen, "router_ids", ids.to(torch.int32), pairs=True)

    def _launch(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        def launch(A: Any, B: Any, C: Any, *a: Any, **k: Any) -> Any:
            out = orig(A, B, C, *a, **k)
            if self._tapped():
                self._launched(A, C, a, k)
            return out
        return launch

    def _launched(self, A: Any, C: Any, a: Sequence[Any], k: Mapping[str, Any]) -> None:
        """One launch of the module at ``self.at``: its A, A_scale and C, right after the kernel wrote C, and its tile."""
        seen = self._launches()
        weighted, scale, top_k, config = (launch_arg(a, k, n) for n in ("mul_routed_weight", "A_scale", "top_k", "config"))
        if weighted is not True and weighted is not False:
            raise TypeError(f"a launch of {self.at} with mul_routed_weight {type(weighted).__name__}: neither its w13 "
                            "nor its w2")
        which = "w2" if weighted else "w13"
        if "m" not in seen or weighted != ("up" in seen) or "down" in seen:
            raise TypeError(f"{self.at}'s {which} launch is out of order: a forward routes with grouped_topk, then "
                            "launches its w13, then its w2, each once")
        if scale is None or not hasattr(scale, "shape") or not isinstance(config, Mapping):
            raise TypeError(f"{self.at}'s {which} launch carries no A_scale or no config: its experts are not block FP8 "
                            "ones launched by the Triton launcher")
        tile = {str(key): int(v) for key, v in config.items()}
        if not weighted:
            if top_k != seen["topk"]:
                raise TypeError(f"{self.at}'s w13 launch has top_k {top_k} and its router {seen['topk']} experts a row")
            self._keep(seen, "in_q", A, pairs=False)
            self._keep(seen, "in_scale", scale, pairs=False)
            self._keep(seen, "up", C, pairs=True)
            seen["tile13"] = tile
            return
        weights = launch_arg(a, k, "topk_weights")
        if top_k != 1 or weights is None:
            raise TypeError(f"{self.at}'s w2 launch has top_k {top_k} and {'no' if weights is None else 'its'} "
                            "topk_weights: a w2 launch weights one pair a row")
        self._keep(seen, "act_q", A, pairs=True)
        self._keep(seen, "act_scale", scale, pairs=True)
        self._keep(seen, "topk_weights", weights, pairs=True)
        self._keep(seen, "down", C, pairs=True)
        seen["tile2"] = tile


def install(worker: Any, targets: Sequence[str], modules: Sequence[str]) -> int:
    """Put a `Capture` on ``worker``'s runner for ``targets`` (the requests' engine ids) and ``modules`` (the MoE
    modules to capture, by their names in the worker's model); its rank. On a worker whose runner replays CUDA graphs,
    before it captures them (`arm`)."""
    runner = worker.model_runner
    if id(runner) in _ON:
        raise ValueError("a capture is already on this worker's runner: `uninstall` it first")
    if not targets or not modules:
        raise ValueError("no request or no module to capture")
    named = dict(runner.get_model().named_modules())
    absent = sorted(set(modules) - named.keys())
    if absent:
        raise KeyError(f"the worker's model has no module named {', '.join(absent)}")
    _sort(runner)
    _ON[id(runner)] = Capture(runner, targets, {m: named[m] for m in modules})
    return int(worker.rank)


def arm(targets: Sequence[str], modules: Sequence[str]) -> None:
    """In a process that has built no worker yet: `install` on each worker it builds, right before the worker warms
    up, which is when its runner captures its graphs. `disarm` takes the patch off."""
    if _ARMED:
        raise ValueError("this process is already armed: `disarm` it first")
    if not targets or not modules:
        raise ValueError("no request or no module to capture")
    owner = getattr(importlib.import_module(WORKER_SITE[0]), WORKER_SITE[1], None)
    if not callable(getattr(owner, "compile_or_warm_up_model", None)):
        raise TypeError(f"{WORKER_SITE[0]} holds no {WORKER_SITE[1]}.compile_or_warm_up_model: the capture is installed "
                        "where the pinned vLLM's worker captures its graphs")

    def warm_up(orig: Callable[..., Any]) -> Callable[..., Any]:
        def compile_or_warm_up_model(worker: Any, *a: Any, **k: Any) -> Any:
            if id(worker.model_runner) not in _ON:
                install(worker, targets, modules)
            return orig(worker, *a, **k)
        return compile_or_warm_up_model

    into = hooks.Hooks(tag=TAG)
    into.wrap(owner, "compile_or_warm_up_model", warm_up)
    _ARMED.append(into)


def disarm() -> None:
    """Take `arm`'s patch of the worker off. A capture it installed stays on until `uninstall`."""
    while _ARMED:
        _ARMED.pop().uninstall()


def _on(worker: Any) -> Capture:
    capture = _ON.get(id(worker.model_runner))
    if capture is None:
        raise TypeError("no capture is on this worker's runner: `install` puts one there")
    return capture


def uninstall(worker: Any) -> int:
    """Take ``worker``'s capture off (its patches of the launch sites, its module hooks and its part of the runner's
    step); its rank. The rows go with the capture, so `take` comes first."""
    capture = _on(worker)
    capture.off()
    del _ON[id(worker.model_runner)]
    return int(worker.rank)


def take(worker: Any, directory: str) -> dict[str, Any]:
    """Write the worker's kept rows under ``<directory>/rank<r>/<i>/<j>.bin`` (i a target's place in sorted order, j a
    module's): a file holds its `KEPT` fields in order, each the request's tokens' rows. Returns the rank and, for each
    target a step scheduled, its ``pieces`` (first position, tokens), each piece's ``dispatch`` (mode, padded tokens)
    and each module's file (from ``directory``), sha256, fields (the words' numpy dtype, the torch dtype they are the
    bits of, a token's shape) and tiles (the w13 and w2 launch's, a piece)."""
    c, rank = _on(worker), int(worker.rank)
    out: dict[str, Any] = {}
    for i, rid in enumerate(sorted(c.targets)):
        if not c.pieces[rid]:
            continue
        d = Path(directory) / f"rank{rank}" / str(i)
        d.mkdir(parents=True, exist_ok=True)
        modules = {}
        for j, name in enumerate(c.names):
            arrays = {f: np.concatenate(c.kept[rid][name][f]) for f in KEPT}
            raw = b"".join(a.tobytes() for a in arrays.values())
            (d / f"{j}.bin").write_bytes(raw)
            modules[name] = {"file": f"rank{rank}/{i}/{j}.bin", "sha256": hashlib.sha256(raw).hexdigest(),
                             "fields": {f: {"dtype": a.dtype.str, "of": c.of[f], "shape": list(a.shape[1:])}
                                        for f, a in arrays.items()},
                             "tiles": c.tiles[rid][name]}
        out[rid] = {"pieces": c.pieces[rid], "dispatch": c.dispatch[rid], "modules": modules}
    return {"rank": rank, "requests": out}


def spans(takes: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """`write`'s spans from every rank's `take` and the replay's own step trace (`StepTrace` rows ``(request, first
    position, tokens)``): each captured request's pieces, the engine step and the dispatch of each, with every rank's
    modules. A rank whose pieces are not the trace's rows of the request, in step order, or whose steps ran at another
    dispatch than another rank's, is refused."""
    trace: dict[str, list[tuple[int, list[int]]]] = {}
    for s in steps:
        for rid, start, n in s["rows"]:
            trace.setdefault(rid, []).append((int(s["step"]), [start, n]))
    out: dict[str, dict[str, Any]] = {}
    for t in sorted(takes, key=lambda t: t["rank"]):
        for rid, mine in sorted(t["requests"].items()):
            pieces = [p for _, p in trace.get(rid, ())]
            if mine["pieces"] != pieces:
                raise ValueError(f"request {rid}: rank {t['rank']} captured (first position, tokens) {mine['pieces']} "
                                 f"and the replay's trace ran it at {pieces}")
            span = out.setdefault(rid, {"pieces": pieces, "steps": [s for s, _ in trace[rid]],
                                        "dispatch": mine["dispatch"], "ranks": {}})
            if mine["dispatch"] != span["dispatch"]:
                raise ValueError(f"request {rid}: rank {t['rank']} ran its steps at (mode, padded tokens) "
                                 f"{mine['dispatch']} and an earlier rank at {span['dispatch']}")
            span["ranks"][str(t["rank"])] = mine["modules"]
    return out


def write(directory: Path, spans: Mapping[str, Mapping[str, Any]]) -> Path:
    """The record of ``spans`` (`spans`') as ``directory/CAPTURE``, where ``directory`` is the one every rank's `take`
    wrote under: the order of a file's fields, and each request's pieces, steps, dispatch and ranks."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CAPTURE
    path.write_text(json.dumps({"kept": list(KEPT), "requests": spans}, indent=1))
    return path


def read(directory: Path, request: str, rank: int, module: str) -> dict[str, Any]:
    """One request's rows of ``module`` at ``rank`` from the capture written to ``directory``: each field of `KEPT` as
    its words ([T, *a token's shape], T the request's tokens over its pieces, in piece order), and under ``pieces``,
    ``steps``, ``dispatch``, ``tiles`` and ``of`` each piece's (first position, tokens), its engine step, its (mode,
    padded tokens), its (w13, w2) tiles, and each field's torch dtype. Refused by name: a request, rank, module or
    field the record lacks, and a file that is not the bytes the record's sha256 and shapes say."""
    doc = json.loads((directory / CAPTURE).read_text())
    if doc.get("kept") != list(KEPT):
        raise ValueError(f"{directory / CAPTURE}: its files hold {doc.get('kept')}, not {list(KEPT)}")
    span = doc["requests"].get(request)
    if span is None:
        raise KeyError(f"{directory / CAPTURE}: no request {request}")
    at = span["ranks"].get(str(rank), {}).get(module)
    if at is None:
        raise KeyError(f"{directory / CAPTURE}: request {request} has no {module} at rank {rank}")
    tokens = sum(n for _, n in span["pieces"])
    raw = (directory / at["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != at["sha256"]:
        raise ValueError(f"{directory / at['file']}: not the bytes the record's sha256 names")
    fields = at["fields"]
    if sorted(fields) != sorted(KEPT):
        raise KeyError(f"{directory / CAPTURE}: {module} of request {request} at rank {rank} has the fields "
                       f"{sorted(fields)}, not {sorted(KEPT)}")
    shapes = {f: (tokens, *fields[f]["shape"]) for f in KEPT}
    sizes = [np.dtype(fields[f]["dtype"]).itemsize * int(np.prod(shapes[f])) for f in KEPT]
    if sum(sizes) != len(raw):
        raise ValueError(f"{directory / at['file']}: {len(raw)} bytes, and its fields' shapes say {sum(sizes)}")
    out: dict[str, Any] = {"pieces": span["pieces"], "steps": span["steps"], "dispatch": span["dispatch"],
                           "tiles": at["tiles"], "of": {f: fields[f]["of"] for f in KEPT}}
    for i, f in enumerate(KEPT):
        out[f] = np.frombuffer(raw, dtype=fields[f]["dtype"], count=int(np.prod(shapes[f])),
                               offset=sum(sizes[:i])).reshape(shapes[f])
    return out
