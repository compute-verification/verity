"""The serve-tax arms' changes to vLLM's own OpenAI server, as a vLLM general plugin (`register`, entry point group
``vllm.general_plugins``, run in each of the server's processes): with ``VERITY_SERVETAX`` set (a JSON object: ``out``, the
directory its files go to, ``window_s``, and a true value for each change), every V1 scheduler built in the process gets

- ``record``: the scheduler's step trace (`verity_vllm.engine.step_trace.StepTrace`, the recorder serve-openai uses), its
  records stamped with the wall clock and written by a thread of its own to ``step_trace.PID.jsonl``;
- ``to_length``: each request decoded to its ``max_tokens`` (`to_length`), as vLLM decodes one sent with
  ``ignore_eos``, for requests that come through the network certifier, whose id request has no such field;
- ``release_order``: `ReleaseGate`, the requests added to the scheduler in the certifier's release order
  (`serve_admit.ReleaseOrder`), their release numbers read from their X-Request-Id (``{run}-{n}``, as `serve_tax`'s client
  numbers them, standing in for the certifier's headers);
- ``tokens``: `TokenCommit`, each window's tokens (every prompt id at its request's arrival and every sampled id) committed
  in ``rows-sha512`` rows on a thread of its own when the window closes, its roots in ``tokens.PID.jsonl``;
- ``boundary_rows`` (an object: ``check``, ``per_prompt``, ``per_output``, ``window_s``, ``drained``, ``inline``): the
  two-stage audit's boundary commitment (`verity_vllm.pipeline.boundary_windows.BoundaryWindows`), each window's finished
  requests committed by size class as the window closes, each class's public part in ``boundary.PID.jsonl``. A class's
  directory holds its requests' tokens and its salts, which are values: it goes under ``private``, the configuration's
  directory for what never leaves the node (``private``/boundary/), and under ``out`` only where none is named. With
  ``tables`` in it (``private``, ``layers``, ``capacity``, ``timeout_s``), each span's latent history tables too: rank 0
  of Model Runner V2 reads each step's rows (`_tables_runner`) and builds each class's tables once the window assigns
  its slots, keeping their rows and salts under the ``private`` it names, never under ``out`` or the run's directory. An
  engine whose cache can't hold the tables' row is refused as it starts, when the runner binds its caches (`_tables_guard`);

and with ``edges_s`` (seconds after the first request whose id holds ``edge_tag``) or ``every_s`` (an edge each
``every_s`` seconds of a load's serving) every engine core takes a replay block's boundary state at each edge
(`edges_of`, `boundary.Edges`): it drains, takes the state between two steps, and goes on. Each edge's record
(`edges.PID.jsonl`: the drain, the pause and its parts, the KV pool's root, sizes, and unless ``busy`` is false the GPU's
busy time each step since the last edge) is written off the engine loop. With the uniprocess executor, so are its
`boundary.dump_core` and block fingerprints (``edges/``) and its state (``kv_dir``/edge-K.state.pkl), and the KV pool's
bytes at the edges ``save_kv`` names to ``kv_dir``/edge-K.kv; from the first edge ``save_kv`` names to the last edge, each
step's sampled tokens and their logprobs' bits go to ``edges/outputs.PID.jsonl``. With a tensor-parallel executor each
rank keeps its part in its own process (`ranks_take`), copied to host memory on the ranks ``kv_copy`` names.

The tick's plugin (`verity_vllm.pipeline.tick_plugin`) is a separate entry point, loaded after this one, so its gate wraps
the scheduler outside these.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

ENV = "VERITY_SERVETAX"
#: how long the engine core waits on its input queue at a time while the release gate holds requests and nothing runs
POLL_S = 0.01
#: the model runners whose ``execute_model`` the KV edge follows: V1's and Model Runner V2's, whichever the engine builds
RUNNERS = ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner")

_PLUGIN: list[Any] = []


def config(environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    raw = (os.environ if environ is None else environ).get(ENV)
    if not raw:
        return None
    cfg = json.loads(raw)
    if not isinstance(cfg, dict) or not cfg.get("out"):
        raise ValueError(f"{ENV}: a JSON object with out and the changes to make, not {raw[:80]!r}")
    return cfg


def to_length(request: Any) -> None:
    """Hold a request to its ``max_tokens``, as vLLM holds one sent with ``ignore_eos``: its sampling parameters say
    so, name no end-of-sequence id, and keep no stop id. vLLM's scheduler stops a request at the parameters'
    end-of-sequence id, at one of their stop ids, or at its length; for a request sent without ``ignore_eos`` its
    front end sets the first to the tokenizer's and adds the generation config's to the second. The canonical id
    request has no ``ignore_eos`` and no stop ids of its own, so with both cleared a request that came as one stops
    where the same request sent to vLLM itself with ``ignore_eos`` stops, at its length. A request without sampling
    parameters (pooling) is left as it is."""
    params = getattr(request, "sampling_params", None)
    if params is not None:
        params.ignore_eos = True
        params._eos_token_id = None
        params.stop_token_ids = []


class ReleaseGate:
    """The scheduler's ``add_request`` in release order: each request's (run, n) from its engine id (``key``), submitted to
    a `serve_admit.ReleaseOrder` that forwards it to ``add`` once every earlier number of its run has come or passed (an
    abort of a held one, or ``hold_s`` without it). The order's skip timers fire from `poll`, which the engine core calls
    at each schedule and while it waits on its input with requests held."""

    def __init__(self, add: Callable[[Any], Any], abort: Callable[[list[str]], Any], *, hold_s: float = 5.0,
                 key: Callable[[str], tuple[str, int] | None], clock: Callable[[], float] = time.monotonic) -> None:
        from verity_vllm.pipeline.serve_admit import RELEASE_HEADER, RUN_HEADER, ReleaseOrder

        self.headers = (RELEASE_HEADER, RUN_HEADER)
        self.key_of, self.clock = key, clock
        self.timers: list[tuple[float, Callable[[], None]]] = []
        self.order = ReleaseOrder(lambda adm: add(adm.request), lambda engine_id: abort([engine_id]), hold_s=hold_s,
                                  schedule=lambda delay, fn: self.timers.append((self.clock() + delay, fn)))

    def add(self, request: Any) -> None:
        k = self.key_of(request.request_id)
        release = None
        if k is not None:
            release = self.order.key({self.headers[0]: str(k[1]), self.headers[1]: k[0]})
        self.order.submit(_Admission(request, request.request_id, release))

    def finished(self, ids: str | Iterable[str] | None) -> list[str] | None:
        """Of the requests the engine core finishes from outside (aborts), those not held here, for the scheduler."""
        if ids is None:
            return None
        ids = [ids] if isinstance(ids, str) else list(ids)
        held = {adm.engine_id for adm in self.order.held.values()}
        for rid in ids:
            if rid in held:
                self.order.abort(rid)
        return [rid for rid in ids if rid not in held]

    def poll(self) -> None:
        now = self.clock()
        due = [t for t in self.timers if t[0] <= now]
        if due:
            self.timers = [t for t in self.timers if t[0] > now]
            for _, fn in due:
                fn()

    def held(self) -> bool:
        return bool(self.order.held)


class _Admission:
    __slots__ = ("request", "engine_id", "release")

    def __init__(self, request: Any, engine_id: str, release: tuple[str, int] | None) -> None:
        self.request, self.engine_id, self.release = request, engine_id, release


#: vLLM's OpenAI server's engine request id around the client's X-Request-Id: ``cmpl-{X-Request-Id}-{choice}-{8 hex}``
ENGINE_ID = re.compile(r"^(?:cmpl|chatcmpl)-(.+)-\d+-[0-9a-f]{8}$")


def release_key(engine_id: str) -> tuple[str, int] | None:
    """(run, n) from an X-Request-Id ``{run}-{n}`` inside the server's engine id."""
    m = ENGINE_ID.match(engine_id)
    run, _, n = (m.group(1) if m else engine_id).rpartition("-")
    return (run, int(n)) if run and n.isdigit() else None


class TokenCommit:
    """Each window's tokens, 4 bytes each (little-endian u32) in the order the engine core saw them, committed as
    ``rows-sha512`` rows under a fresh salt key (`step_rows.keyed_tensor_root`) on a thread of its own when the window
    closes (`tick`, at the first schedule ``window_s`` after it opened, and at exit). ``sink`` gets each window's record."""

    def __init__(self, window_s: float, sink: Callable[[dict], None], clock: Callable[[], float] = time.monotonic,
                 commit: Callable[[int, bytes], bytes] | None = None) -> None:
        self.window_s, self.sink, self.clock, self.commit = float(window_s), sink, clock, commit or rows_root
        self.window = 0
        self.opened = clock()
        self.tokens: list[int] = []
        self.jobs: queue.SimpleQueue = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._work, name="token-commit", daemon=True)
        self.thread.start()

    def prompt(self, request: Any) -> None:
        self.tokens.extend(request.prompt_token_ids or ())

    def sampled(self, outputs: Any) -> None:
        for eco in (outputs or {}).values():
            for o in getattr(eco, "outputs", None) or ():
                if o.new_token_ids:
                    self.tokens.extend(o.new_token_ids)

    def tick(self) -> None:
        if self.clock() - self.opened >= self.window_s:
            self.close()

    def close(self) -> None:
        tokens, self.tokens = self.tokens, []
        self.jobs.put((self.window, self.opened, self.clock(), tokens))
        self.window += 1
        self.opened = self.clock()

    def _work(self) -> None:
        import array

        while (job := self.jobs.get()) is not None:
            window, opened, closed, tokens = job
            t = time.perf_counter()
            data = array.array("I", tokens).tobytes()
            root = self.commit(window, data)
            self.sink({"window": window, "tokens": len(tokens), "bytes": len(data), "root": root.hex(), "open_s": opened,
                       "close_s": closed, "hash_s": round(time.perf_counter() - t, 6)})

    def stop(self) -> None:
        if self.tokens:
            self.close()
        self.jobs.put(None)
        self.thread.join()


def rows_root(window: int, data: bytes) -> bytes:
    """``data``'s root as ``rows-sha512`` rows (`step_rows.keyed_tensor_root`) under a fresh salt key."""
    from verity_vllm.commit import step_rows

    entry = ["tokens", "u32", [len(data) // 4], len(data)]
    return step_rows.keyed_tensor_root(bytes(step_rows.ID_BYTES), window, entry, data, os.urandom(step_rows.KEY_BYTES))


class KvEdge:
    """At each window edge (`tick`, at the model runner's first step ``window_s`` after the last edge), the paged KV cache
    hashed where it lies: each distinct storage under its layers' tensors, viewed as rows of ``row_bytes``, hashed by the
    GPU's row-table kernels (`row_tables_device.step_roots`: SHA-512 row digests, hm96 leaves, frame trees) on their side
    stream, behind an event on the serving stream, in chunks of at most ``chunk_rows`` rows, so that nothing waits on the
    host. The kernels' work is per 1,024-byte segment, so the row size changes the planning on the host, not the hashing;
    their scratch is 64 bytes a segment, so at most ``inflight`` chunks are out at a time, the next launched at a later
    step once one is done. The whole pool is hashed, free blocks too: an upper bound on hashing the blocks live at the
    edge. A block that a step after the edge writes before the side stream reads it is hashed as written: a measurement of
    the cost, not a commitment of the state. ``sink`` gets each edge's record once its last chunk is done (bytes, kernel
    ms, the edge's span from its first launch to its last chunk's reaping). ``launch(edge, rows)`` hashes one chunk and
    returns what `row_tables_device.step_roots` returns (``ready()``, ``kernel_ms()``)."""

    def __init__(self, window_s: float, sink: Callable[[dict], None], row_bytes: int = 1 << 20, chunk_rows: int = 4096,
                 inflight: int = 2, clock: Callable[[], float] = time.monotonic,
                 launch: Callable[[int, Any], Any] | None = None) -> None:
        self.window_s, self.sink, self.clock, self.launch = float(window_s), sink, clock, launch or launch_rows
        self.row_bytes, self.chunk_rows, self.inflight = int(row_bytes), int(chunk_rows), int(inflight)
        self.edge = 0
        self.last = clock()
        self.todo: list[tuple[int, Any]] = []        # (edge, chunk) not yet launched
        self.out: list[tuple[int, Any, int]] = []    # (edge, pending, bytes) launched, not yet reaped
        self.edges: dict[int, dict[str, Any]] = {}   # edge -> its totals until its last chunk is reaped

    def rows(self, kv_caches: Iterable[Any]) -> list[Any]:
        """Each distinct storage under the KV cache as (n, row_bytes) int8 rows (a last partial row dropped)."""
        import torch

        seen: dict[int, Any] = {}
        for t in kv_caches:
            for x in (t if isinstance(t, (list, tuple)) else (t,)):
                s = x.untyped_storage()
                if s.data_ptr() in seen:
                    continue
                flat = torch.empty(0, dtype=torch.int8, device=x.device).set_(s)
                k = self.row_bytes
                seen[s.data_ptr()] = flat[: flat.numel() // k * k].view(-1, k)
        return list(seen.values())

    def tick(self, kv_caches: Iterable[Any]) -> None:
        self.reap()
        if self.clock() - self.last >= self.window_s:
            self.last = self.clock()
            chunks = [r[lo:lo + self.chunk_rows] for r in self.rows(kv_caches) for lo in range(0, r.shape[0], self.chunk_rows)]
            self.edges[self.edge] = {"chunks": len(chunks), "left": len(chunks), "bytes": 0, "kernel_ms": 0.0,
                                     "start": self.clock()}
            self.todo += [(self.edge, c) for c in chunks]
            self.edge += 1
        while self.todo and len(self.out) < self.inflight:
            edge, c = self.todo.pop(0)
            self.out.append((edge, self.launch(edge, c), c.shape[0] * c.shape[1]))

    def reap(self, wait: bool = False) -> None:
        while self.out and (wait or self.out[0][1].ready()):
            edge, p, nbytes = self.out.pop(0)
            e = self.edges[edge]
            e["bytes"] += nbytes
            e["kernel_ms"] += p.kernel_ms()
            e["left"] -= 1
            if e["left"] == 0:
                del self.edges[edge]
                ms = e["kernel_ms"]
                self.sink({"edge": edge, "bytes": e["bytes"], "chunks": e["chunks"], "kernel_ms": round(ms, 3),
                           "gb_per_s": round(e["bytes"] / ms / 1e6, 1) if ms else None,
                           "span_s": round(self.clock() - e["start"], 4), "t": time.time()})
            if wait and not self.out and self.todo:
                edge, c = self.todo.pop(0)
                self.out.append((edge, self.launch(edge, c), c.shape[0] * c.shape[1]))


def launch_rows(edge: int, rows: Any) -> Any:
    """One chunk of (n, K) int8 rows hashed by the row-table kernels on their side stream, timed."""
    import torch

    from verity_vllm.commit import row_tables_device as RTD

    ids = [("kv", i) for i in range(rows.shape[0])]
    scales = rows.new_zeros(rows.shape[0], dtype=torch.float32)
    return RTD.step_roots(bytes(64), edge, {"kv": (ids, rows, scales)}, bytes(32), timing=True)


def _kv_runner(execute: Callable[..., Any], cfg: Mapping[str, Any], capturing: Callable[[], bool] | None = None,
               edge_of: Callable[[], KvEdge] | None = None) -> Callable[..., Any]:
    """The model runner's ``execute_model``, then `KvEdge.tick` on its KV cache. The edge, and its window, start at the
    first step with a KV cache: the engine's startup runs the model before it has one (the memory profile) and while
    it captures CUDA graphs, when nothing may launch on another stream."""
    state: dict[str, Any] = {}
    if capturing is None:
        import torch

        capturing = torch.cuda.is_current_stream_capturing

    def make() -> KvEdge:
        edge = KvEdge(float(cfg.get("kv_window_s", cfg.get("window_s", 600))),
                      _lines(Path(cfg["out"]) / f"kv_edges.{os.getpid()}.jsonl"))
        atexit.register(edge.reap, True)
        return edge

    def execute_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        out = execute(self, *args, **kwargs)
        kv = getattr(self, "kv_caches", None)
        if kv and not capturing():
            if "edge" not in state:
                state["edge"] = (edge_of or make)()
            state["edge"].tick(kv)
        return out
    return execute_model


def _public(cfg: Mapping[str, Any]) -> tuple[Path, ...]:
    """The directories whose files leave the node: the serve's records and the run's own."""
    from verity_config.settings import SHARED

    return tuple(Path(p) for p in (cfg["out"], SHARED.RESEARCH_RUN_DIR) if p)


def _tp_rank() -> int:
    from vllm.distributed import get_tensor_model_parallel_rank

    return get_tensor_model_parallel_rank()


def rank0_tables(runner: Any, cfg: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    """Rank 0's history tables on ``runner``: its step hook at the configured layers, the channel to the boundary
    commitment and the builder that answers it on the runner's device, keeping each table's rows and salts."""
    from verity_vllm.pipeline import boundary_tables as BT

    tc, public = cfg["boundary_rows"]["tables"], _public(cfg)
    hook = BT.StepHook(runner.model, layers=BT.layers_of(tc["layers"]))
    keep = BT.Kept.of(tc, *public)
    return hook, BT.Channel.of(tc, *public), BT.Builder(hook.latents, device=runner.device, keep=keep)


def _tables_guard(initialize: Callable[..., Any], cfg: Mapping[str, Any],
                  rank: Callable[[], int] = _tp_rank) -> Callable[..., Any]:
    """Model Runner V2's ``initialize_kv_cache``, then, on rank 0, the history tables' guard on the caches it bound
    (`boundary_tables.StepHook`'s: an MLA cache that doesn't hold the tables' row, a configured layer that is no MLA
    layer). A refusal stops the engine as it starts, before it serves a request."""
    def initialize_kv_cache(self: Any, *args: Any, **kwargs: Any) -> Any:
        out = initialize(self, *args, **kwargs)
        if getattr(self, "kv_caches", None) and rank() == 0:
            from verity_vllm.pipeline import boundary_tables as BT

            BT.StepHook(self.model, layers=BT.layers_of(cfg["boundary_rows"]["tables"]["layers"]))
        return out
    return initialize_kv_cache


def _tables_runner(execute: Callable[..., Any], cfg: Mapping[str, Any], capturing: Callable[[], bool] | None = None,
                   rank: Callable[[], int] = _tp_rank, start: bool = True) -> Callable[..., Any]:
    """Model Runner V2's ``execute_model``, then, on rank 0, `boundary_tables.StepHook.after` on the step. Its tables
    start at the first step with a KV cache outside a capture and a dummy run, as the edge's do, with the builder
    answering the boundary commitment's channel on a thread of its own (``start``). A step whose tables can't start
    raises, and so does every step after it: no step is served with the tables off."""
    state: dict[str, Any] = {}
    if capturing is None:
        import torch

        capturing = torch.cuda.is_current_stream_capturing

    def execute_model(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        out = execute(self, scheduler_output, *args, **kwargs)
        if kwargs.get("dummy_run") or not getattr(self, "kv_caches", None) or capturing():
            return out
        if "hook" not in state:
            if rank() != 0:
                state["hook"] = None
            else:
                hook, channel, builder = rank0_tables(self, cfg)
                started = {"stop": channel.start(builder)} if start else {}
                state.update(hook=hook, channel=channel, builder=builder, **started)
        if state["hook"] is not None:
            state["hook"].after(self, scheduler_output)
        return out
    execute_model.state = state  # type: ignore[attr-defined]
    return execute_model


def _lines(path: Path) -> Callable[[dict], None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a")
    lock = threading.Lock()

    def write(rec: dict) -> None:
        with lock:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            fh.flush()
    return write


def _lines_behind(path: Path) -> Callable[[dict], None]:
    """`_lines`, written by a thread of its own: the caller only queues the line."""
    write = _lines(path)
    q: queue.SimpleQueue = queue.SimpleQueue()

    def work() -> None:
        while (rec := q.get()) is not None:
            write(rec)
    t = threading.Thread(target=work, name="servetax-lines", daemon=True)
    t.start()
    atexit.register(lambda: (q.put(None), t.join()))
    return q.put


def step_outputs_hook(trace: Any, sink: Callable[[dict], None], active: Callable[[], bool]) -> Callable[[Any], Any]:
    """A wrapper of the scheduler's ``update_from_output``, outside the step trace's (which forgets the step's index): while
    ``active()``, each step's sampled tokens and logprob bits (`boundary.step_outputs`) to ``sink`` under the trace's
    index of the step."""
    import boundary as B

    def update_from_output(orig):
        def wrapped(so, mro, *a, **kw):
            step = trace._pending.get(id(so))
            outs = orig(so, mro, *a, **kw)
            if step is not None and active():
                sink({"step": step, "out": B.step_outputs(outs)})
            return outs
        return wrapped
    return update_from_output


def edges_of(core: Any, cfg: Mapping[str, Any], clock: Callable[[], float] = time.monotonic) -> Any:
    """The engine core's `boundary.Edges` under ``cfg``: due every ``every_s`` seconds of a load's serving
    (`boundary.every`), or ``edges_s`` after the first request whose id holds ``edge_tag`` (`boundary.at_times`), each
    taking the boundary state as module `boundary` reads it, its record and files written as the module's docstring says.
    With the uniprocess executor the engine core takes every part itself (`local_take`); with a tensor-parallel one, each
    rank takes its own where it lies (`ranks_take`). Either hashes the KV pool once here, at the engine's first step, so
    that the hash kernels are built before the first edge and its pause is only the edge's."""
    import boundary as B
    from verity_vllm.engine.hooks import Hooks
    from verity_vllm.engine.step_trace import StepTrace

    sched = core.scheduler
    trace = next((x for x in getattr(sched, "_verity_servetax", ()) if isinstance(x, StepTrace)), None)
    if trace is None:
        raise RuntimeError("the edges need the step trace (record) on the scheduler")
    at = None if cfg.get("every_s") else [float(x) for x in cfg["edges_s"]]
    due, start = B.every(float(cfg["every_s"]), clock) if at is None else B.at_times(at, clock)
    tag = cfg.get("edge_tag")
    load: list[str | None] = [None]
    hooks = Hooks(tag="serve-tax-edges")

    def add_request(orig):
        def wrapped(request, *a, **kw):
            if tag is None or tag in request.request_id:
                key = release_key(request.request_id)
                load[0] = key[0] if key else None
                start(load[0])
            return orig(request, *a, **kw)
        return wrapped

    hooks.wrap(sched, "add_request", add_request)
    local = hasattr(core.model_executor, "driver_worker")
    take = (local_take if local else ranks_take)(core, cfg, trace, hooks, at, load, clock)
    return B.Edges(core, due, take, clock)


def local_take(core: Any, cfg: Mapping[str, Any], trace: Any, hooks: Any, at: list[float] | None, load: list,
               clock: Callable[[], float]) -> Callable[[Any, dict], None]:
    """`edges_of`'s take for the uniprocess executor: the KV pool copied to a pinned host buffer allocated here, once, and
    hashed, and the engine core's state with the runner's rows, written off the loop; at the edges ``save_kv`` names,
    the pool's bytes too, and from the first of them to the last edge each step's outputs (fixed times only)."""
    import pickle

    import numpy as np
    import torch

    import boundary as B

    out = Path(cfg["out"])
    d = out / "edges"
    d.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    sched, _kvm, pool, runner = B.parts(core)
    kv = B.KvPool(runner.kv_caches, len(pool.blocks))
    host = kv.host() if cfg.get("kv_copy", "all") not in (False, "none") else None
    save_kv = set(int(k) for k in cfg.get("save_kv", (0,)))
    kv_dir = Path(cfg.get("kv_dir") or d)
    kv_dir.mkdir(parents=True, exist_ok=True)
    record = _lines(out / f"edges.{pid}.jsonl")
    t = clock()
    kv.take()
    record({"hash_build_s": clock() - t})
    busy = B.Busy(lambda: torch.cuda.Event(enable_timing=True), first=trace.scheduled) if cfg.get("busy", True) else None
    runner._verity_busy = busy
    active = [False]
    last = {"step": trace.scheduled, "resumed_t": None}
    writer: list[threading.Thread | None] = [None]
    hooks.wrap(sched, "update_from_output", step_outputs_hook(trace, _lines_behind(d / f"outputs.{pid}.jsonl"),
                                                              lambda: active[0]))

    def take(e: B.Edges, info: dict) -> None:
        k = info["edge"]
        t0 = clock()
        torch.cuda.synchronize()
        t1 = clock()
        st = B.capture(core)
        st["next_step"], st["kv_layout"] = trace.scheduled, kv.layout()
        t2 = clock()
        dumped = B.dump_core(core) if cfg.get("dump", True) else None
        t3 = clock()
        if writer[0] is not None:
            writer[0].join()
        t4 = clock()
        kvr = kv.take(host)
        t5 = clock()
        fp = kvr.pop("fingerprint")
        kvr["copied_bytes"] = kvr["bytes"] if host is not None else 0
        st["kv"] = kvr
        busy_s = busy_steps = busy_steps_s = None
        if busy is not None:
            busy.reap(wait=True)
            busy_s, busy_steps = busy.total(last["step"], trace.scheduled)
            busy_steps_s = [round(x, 7) for x in busy.steps(last["step"], trace.scheduled)]
        rec = {"edge": k, "load": load[0], "next_step": trace.scheduled, "at_s": at[k] if at else None,
               "drained_unix": time.time() - (clock() - info["drained_t"]),
               "drain_s": info["drained_t"] - info["due_t"], "drain_calls": info["drain_calls"], "sync_s": t1 - t0,
               "state_s": t2 - t1, "dump_s": t3 - t2, "write_wait_s": t4 - t3, "kv_s": t5 - t4,
               "pause_s": t5 - info["drained_t"], "pause_production_s": (t5 - info["drained_t"]) - (t3 - t2) - (t4 - t3),
               "requests": len(sched.requests), "running": len(sched.running), "waiting": len(sched.waiting),
               "sizes": B.sizes(st), "kv": kvr, "since_last": {"from_step": last["step"], "busy_s": busy_s,
                                                              "busy_steps": busy_steps, "busy_steps_s": busy_steps_s,
                                                              "wall_s": (info["drained_t"] - last["resumed_t"])
                                                              if last["resumed_t"] is not None else None},
               "skipped": st["skipped"], "runner_skipped": st["runner"]["skipped"]}
        record(rec)
        last["step"], last["resumed_t"] = trace.scheduled, clock()
        active[0] = bool(at and save_kv) and min(save_kv) <= k < len(at) - 1

        def write() -> None:
            t = time.monotonic()
            with open(kv_dir / f"edge-{k}.state.pkl", "wb") as f:
                pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)
            if dumped is not None:
                (d / f"edge-{k}.dump.json").write_text(json.dumps(dumped))
            if fp is not None:
                np.save(d / f"edge-{k}.fp.npy", fp)
            got = {"edge": k, "state_write_s": time.monotonic() - t}
            if host is not None and k in save_kv:
                t = time.monotonic()
                path = kv_dir / f"edge-{k}.kv"
                got |= {"kv_file": str(path), "kv_write_bytes": B.write_host(host, str(path)), "kv_write_s": time.monotonic() - t}
            record(got)
        writer[0] = threading.Thread(target=write, name=f"edge-{k}-write")
        writer[0].start()
    return take


def ranks_take(core: Any, cfg: Mapping[str, Any], trace: Any, hooks: Any, at: list[float] | None, load: list,
               clock: Callable[[], float], key: Callable[[], bytes] = lambda: os.urandom(32),
               record: Callable[[dict], None] | None = None) -> Callable[[Any, dict], None]:
    """`edges_of`'s take for a tensor-parallel executor, whose ranks the engine core reaches only through
    ``collective_rpc``: the engine core's state taken here, and on every rank its KV pool hashed under a fresh key, the
    same on every rank, so that ranks holding the same bytes give the same root (`boundary.rank_take`), copied to host
    memory on the ranks ``kv_copy`` names (`boundary.KV_COPY`), and its runner rows and generators taken. Each part is
    kept where it was taken until the next edge replaces it; nothing is written but the edge's record, and the record's
    pause runs to the engine core's next step."""
    import boundary as B

    if cfg.get("save_kv"):
        raise RuntimeError("the edges of a tensor-parallel engine keep their KV bytes in host memory: no save_kv")
    sched = core.scheduler
    pool = sched.kv_cache_manager.block_pool
    rpc = core.model_executor.collective_rpc
    record = record or _lines_behind(Path(cfg["out"]) / f"edges.{os.getpid()}.jsonl")
    t = clock()
    init = rpc(B.rank_pool, args=(len(pool.blocks), cfg.get("kv_copy", "all")))
    record({"hash_build_s": clock() - t, "ranks": init})
    pool_bytes = sum(r["bytes"] for r in init)
    kept: dict[str, Any] = {}

    def take(e: B.Edges, info: dict) -> None:
        k = info["edge"]
        t0 = clock()
        kept["state"] = B.capture(core, runner=False)
        kept["state"]["next_step"] = trace.scheduled
        t1 = clock()
        ranks = sorted(rpc(B.rank_take, args=(key(),)), key=lambda r: r["rank"])
        t2 = clock()
        roots = {r["root"] for r in ranks}
        copied = [r["copy_ms"] for r in ranks if r["copy_ms"] is not None]
        rec = {"edge": k, "load": load[0], "next_step": trace.scheduled, "at_s": at[k] if at else None,
               "drained_unix": time.time() - (clock() - info["drained_t"]),
               "drain_s": info["drained_t"] - info["due_t"], "drain_calls": info["drain_calls"],
               "state_s": t1 - t0, "kv_s": t2 - t1,
               "requests": len(sched.requests), "running": len(sched.running), "waiting": len(sched.waiting),
               "kv": {"root": min(roots), "roots_equal": len(roots) == 1, "bytes": pool_bytes,
                      "copied_bytes": sum(r["copied_bytes"] for r in ranks),
                      "hash_kernel_ms": max(r["hash_kernel_ms"] for r in ranks), "copy_ms": max(copied, default=None),
                      "rank_kv_s": max(r["kv_s"] for r in ranks), "runner_s": max(r["runner_s"] for r in ranks)},
               "state_requests_bytes": len(kept["state"]["requests"]), "ranks": ranks}
        rec["pause_s"] = clock() - info["drained_t"]
        record(rec)
    return take


def _edge_step(orig: Callable[[Any], Any], cfg: Mapping[str, Any]) -> Callable[[Any], Any]:
    """`EngineCore.step_with_batch_queue` through the engine core's `boundary.Edges`, made at its first step (after startup,
    when its KV cache exists)."""
    def step_with_batch_queue(self: Any) -> Any:
        e = getattr(self, "_verity_edges", None)
        if e is None:
            e = self._verity_edges = edges_of(self, cfg)
        return e.step(orig)
    return step_with_batch_queue


def install(sched: Any, cfg: Mapping[str, Any]) -> list[Any]:
    """The changes ``cfg`` names on one scheduler, innermost first: the step trace, then each request's hold to its
    length, the tokens' commit and the boundary commitment, then the release gate (outside them, so that they see a
    request when the gate adds it)."""
    from verity_vllm.engine.hooks import Hooks
    from verity_vllm.engine.step_trace import StepTrace

    out = Path(cfg["out"])
    pid = os.getpid()
    done: list[Any] = []
    if cfg.get("record"):
        trace = StepTrace(out / f"step_trace.{pid}.jsonl", clock=time.time, writer=True).install(sched)
        done.append(trace)
    hooks = Hooks(tag="serve-tax")
    if cfg.get("to_length"):
        def held(orig):
            def wrapped(request, *a, **kw):
                to_length(request)
                return orig(request, *a, **kw)
            return wrapped

        hooks.wrap(sched, "add_request", held)
    if cfg.get("tokens"):
        tc = TokenCommit(float(cfg.get("window_s", 600)), _lines(out / f"tokens.{pid}.jsonl"))

        def add_request(orig):
            def wrapped(request, *a, **kw):
                tc.prompt(request)
                return orig(request, *a, **kw)
            return wrapped

        def update_from_output(orig):
            def wrapped(so, mro, *a, **kw):
                outs = orig(so, mro, *a, **kw)
                tc.sampled(outs)
                return outs
            return wrapped

        def schedule(orig):
            def wrapped(*a, **kw):
                tc.tick()
                return orig(*a, **kw)
            return wrapped

        hooks.wrap(sched, "add_request", add_request)
        hooks.wrap(sched, "update_from_output", update_from_output)
        hooks.wrap(sched, "schedule", schedule)
        atexit.register(tc.stop)
        done.append(tc)
    if cfg.get("boundary_rows"):
        from verity_vllm.pipeline.boundary_windows import BoundaryWindows

        sink = _lines(out / f"boundary.{pid}.jsonl")
        kept = Path(cfg["private"]) if cfg.get("private") else out
        released = release_key if cfg.get("release_order") else None
        done.append(BoundaryWindows.of(cfg["boundary_rows"], kept / "boundary", sink, released, public=_public(cfg))
                    .install(sched, hooks))
    if cfg.get("release_order"):
        add, finish = sched.add_request, sched.finish_requests
        gate = ReleaseGate(add, lambda ids: finish(ids, _aborted()), key=release_key, hold_s=float(cfg.get("hold_s", 5.0)))

        def gated_add(orig):
            return lambda request, *a, **kw: gate.add(request)

        def gated_schedule(orig):
            def wrapped(*a, **kw):
                gate.poll()
                return orig(*a, **kw)
            return wrapped

        def gated_finish(orig):
            return lambda request_ids, finished_status: orig(gate.finished(request_ids), finished_status)

        hooks.wrap(sched, "add_request", gated_add)
        hooks.wrap(sched, "schedule", gated_schedule)
        hooks.wrap(sched, "finish_requests", gated_finish)
        hooks.patch(sched, "_verity_release", gate)
        done.append(gate)
    if cfg.get("kv"):
        import torch

        from verity_vllm.commit import row_tables_device as RTD

        RTD.DeviceLibrary.of(torch, torch.cuda.current_device())
    done.append(hooks)
    return done


def _aborted() -> Any:
    from vllm.v1.request import RequestStatus

    return RequestStatus.FINISHED_ABORTED


def _init(init: Callable[..., None], cfg: Mapping[str, Any]) -> Callable[..., None]:
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        self._verity_servetax = install(self, cfg)
    return __init__


def _input(orig: Callable[[Any], None]) -> Callable[[Any], None]:
    """`EngineCoreProc._process_input_queue` while the release gate holds requests and nothing runs: wait on the input
    queue ``POLL_S`` at a time, firing the gate's skip timers between, instead of blocking until the next request."""
    def _process_input_queue(self: Any) -> None:
        gate = getattr(self.scheduler, "_verity_release", None)
        while gate is not None and gate.held() and not self.has_work() and self.is_running():
            try:
                req = self.input_queue.get(timeout=POLL_S)
            except queue.Empty:
                gate.poll()
                continue
            self._handle_client_request(*req)
            gate.poll()
        return orig(self)
    return _process_input_queue


def register() -> None:
    """Install `config`'s changes on every V1 scheduler this process builds. Idempotent."""
    cfg = config()
    if cfg is None or _PLUGIN:
        return
    from verity_vllm.engine.hooks import Hooks
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCoreProc

    hooks = Hooks(tag="serve-tax")
    hooks.wrap(Scheduler, "__init__", lambda init: _init(init, cfg))
    if cfg.get("release_order"):
        hooks.wrap(EngineCoreProc, "_process_input_queue", _input)
    if cfg.get("kv"):
        import importlib

        for name in RUNNERS:
            runner = getattr(importlib.import_module(name), "GPUModelRunner", None)
            if runner is not None:
                hooks.wrap(runner, "execute_model", lambda execute: _kv_runner(execute, cfg))
    if (cfg.get("boundary_rows") or {}).get("tables"):
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        hooks.wrap(GPUModelRunner, "initialize_kv_cache", lambda initialize: _tables_guard(initialize, cfg))
        hooks.wrap(GPUModelRunner, "execute_model", lambda execute: _tables_runner(execute, cfg))
    if cfg.get("edges_s") or cfg.get("every_s"):
        import boundary as B
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        hooks.wrap(EngineCore, "step_with_batch_queue", lambda orig: _edge_step(orig, cfg))
        if cfg.get("busy", True):
            B.busy_runner(GPUModelRunner, lambda r: getattr(r, "_verity_busy", None))
    _PLUGIN.append(hooks)
