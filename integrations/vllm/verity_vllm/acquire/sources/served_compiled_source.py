"""Served-default compiled acquisition: the values of a torch.compile + CUDA-graph engine at vLLM's serving defaults
(`-O2`: VLLM_COMPILE, Inductor, FULL_AND_PIECEWISE graphs, the V2 model runner, no batch invariance), acquired with no Python
hook inside graph replay.

What one engine step commits, in one canonical order (piece, sub, index):
  * piece 0's token inputs, `c00.input/ids` and `c00.input/positions` (sub -1), named from the compiled piece's placeholders;
  * every buffer Inductor materializes (sub 0): after each scheduled node's codegen the generated wrapper calls
    `run_intermediate_hooks('verity|<op>|<buf>', <buf>)` for each of the node's outputs that is allocated and still live (fused-away,
    freed, mutation-only, tuple and non-CUDA outputs are not values of the wrapper), so Triton, extern (cuBLAS `out=`), template,
    combo and fallback kernels are all covered -> `c<piece>.<op>/<buf>`;
  * each attention layer's output between pieces (sub 1), from its impl's `forward` -> `c<piece>.attention/<layer>`;
  * the logits, outside the compiled model, under the LogitsProcessor's module name (the committer adds the sampled ids).

Three execution paths give the committer the same sequence of (name, bytes):
  * graph capture (FULL: the whole forward under one `torch.cuda.graph`; PIECEWISE: each piece): the hook issues a device copy of
    the tensor into a static slot (one per (piece, sub, index, name), as many rows as the largest capture size), and the capture
    records the slots its (mode, padded size) writes.  The graph replays the copies every step.  Slots are allocated in the
    uncaptured warm-up pass vLLM runs before each capture, outside the graph pool; vLLM's memory-profiling captures are discarded
    with their slots.
  * a replayed step: nothing Python runs for captured work.  Live work inside the step (attention between PIECEWISE pieces, piece 0's
    inputs at its wrapper) copies into its slot the same way.  When the forward is complete (the logits) the step's slots are
    acquired in canonical order: `slot[:n]` for token-row tensors (n = the step's tokens; rows n.. are CUDA-graph padding, never a
    value), the whole slot otherwise.
  * an uncaptured step (NONE: e.g. a prefill above the largest capture size): each hook acquires at once, in the same order; a
    gathering committer flushes after each, since Inductor reuses a dead buffer's storage within its wrapper.

Kernels, fusion and autotuning are untouched: the hooks are wrapper lines added after Inductor fixed the kernels.  The generated
code must be produced with these hooks installed, so the engine compiles fresh: `install_hooks` turns Inductor's FX-graph and
AOTAutograd caches off (a hit would hand back a wrapper generated without the hooks), and vLLM's own is off for every compiled engine
(`engine.env.prepare_compiled_process_cache`).

Faults (negatives): `compiled_omit:<substring>` -- tensors whose name contains it are not acquired (coverage fails);
`compiled_late_drain` -- a replayed step acquires the previous replayed step's slot contents (`linkage` fails).
"""
from __future__ import annotations

import re
import time
from typing import Any

import torch

from verity_vllm.engine import hooks

PREFIX = "verity|"
GRAPH_MODES = ("FULL", "PIECEWISE")
#: The combined CUDA scheduler's per-node codegen entry points (`Scheduler._codegen` dispatches each node to one of them; extern
#: nodes go to `Scheduler.codegen_extern_call`).  The hooks are emitted after the call, for `scheduler.current_node`.
CODEGEN_METHODS = ("codegen_node", "codegen_template", "codegen_combo_kernel", "codegen_nested_reduction", "codegen_mix_order_reduction")
INPUT_NAMES = (("input_ids", "ids"), ("positions", "positions"))


def _capturing(val: torch.Tensor) -> bool:
    return val.is_cuda and torch.cuda.is_current_stream_capturing()


def _label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", text).strip("_") or "op"


def op_label(node: Any) -> str:
    """A kernel label for a scheduler node: an extern kernel's name (`mm`, `moe_forward`), else the origin op of its first output."""
    from torch._inductor import ir

    inner = getattr(node, "node", None)
    if isinstance(inner, ir.ExternKernel):
        name = getattr(inner, "python_kernel_name", None) or ""
        parts = [p for p in name.split(".") if p and p != "default"]
        return _label(parts[-1]) if parts else "extern"
    for buf in node.get_outputs():
        origin = buf.node.get_origin_node() if hasattr(buf.node, "get_origin_node") else None
        target = getattr(origin, "target", None)
        if target is not None:
            return _label(str(getattr(target, "_opname", None) or getattr(target, "__name__", target)))
    return "fused"


def hookable_outputs(node: Any, wrapper: Any, graph: Any) -> tuple[list[str], dict[str, int]]:
    """The output buffers of `node` that exist as CUDA tensors in the generated wrapper right after the node's code, and why the
    others don't."""
    from torch._inductor import ir

    mutation = getattr(ir, "MutationOutput", ())
    keep: list[str] = []
    skipped: dict[str, int] = {}
    for buf in node.get_outputs():
        name = buf.get_name()
        layout = getattr(buf.node, "layout", None)
        device = buf.node.get_device() if hasattr(buf.node, "get_device") else None
        why = ("removed" if name in graph.removed_buffers or name in graph.inplaced_to_remove else
               "freed" if name in wrapper.freed else
               "mutation" if isinstance(buf.node, mutation) else
               "no_tensor" if isinstance(layout, (ir.NoneLayout, ir.MultiOutputLayout)) else
               "not_cuda" if device is None or device.type != "cuda" else None)
        if why is None:
            keep.append(name)
        else:
            skipped[why] = skipped.get(why, 0) + 1
    return keep, skipped


def install_codegen_hooks(hook: Any, into: hooks.Hooks) -> dict[str, Any]:
    """Patch Inductor so every generated wrapper calls `hook(name, tensor)` on each materialized buffer right after the node that
    writes it.  Must run before the engine compiles; returns the live codegen counters."""
    from torch._functorch import config as fcfg
    from torch._inductor import config as icfg
    from torch._inductor import hooks as ih
    from torch._inductor import scheduler as isched
    from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
    from torch._inductor.virtualized import V

    into.patch(icfg, "fx_graph_cache", False)
    into.patch(fcfg, "enable_autograd_cache", False)
    stats: dict[str, Any] = {"nodes": 0, "hooks": 0, "skipped": {}}

    def emit() -> None:
        sched, wrapper = V.graph.scheduler, V.graph.wrapper_code
        node = getattr(sched, "current_node", None)
        if node is None or not getattr(wrapper, "supports_intermediate_hooks", False):
            return
        seen = wrapper.__dict__.setdefault("_verity_hooked", set())
        if node.get_name() in seen:
            return
        seen.add(node.get_name())
        names, skipped = hookable_outputs(node, wrapper, V.graph)
        op = op_label(node)
        stats["nodes"] += 1
        for why, k in skipped.items():
            stats["skipped"][why] = stats["skipped"].get(why, 0) + k
        for name in names:
            wrapper.writeline(f"run_intermediate_hooks({PREFIX + op + '|' + name!r}, {name})")
            stats["hooks"] += 1

    def after(orig: Any) -> Any:
        def run(self: Any, *a: Any, **kw: Any) -> Any:
            out = orig(self, *a, **kw)
            emit()
            return out
        return run

    for name in CODEGEN_METHODS:
        if hasattr(CUDACombinedScheduling, name):
            into.wrap(CUDACombinedScheduling, name, after)
    into.wrap(isched.Scheduler, "codegen_extern_call", after)
    if hook not in ih.INTERMEDIATE_HOOKS:
        into.append(ih.INTERMEDIATE_HOOKS, hook)
    return stats


class ServedCompiledSource:
    short_name = "served_compiled"

    def __init__(self, fault: str | None = None, into: hooks.Hooks | None = None) -> None:
        self.fault = fault
        self.omit = fault.split(":", 1)[1] if fault and fault.startswith("compiled_omit:") else None
        self.c: Any = None
        self._hooks = into if into is not None else hooks.Hooks()
        self.codegen: dict[str, Any] = {}
        self.piece = -1
        self.index: dict[int, int] = {}
        self.in_cgw = False
        self.max_rows = 0
        self.slots: dict[tuple, torch.Tensor] = {}
        self.static: set[tuple] = set()                                 # (order, name) captured once with a leading dim other than the padded size
        self.plans: dict[tuple[str, int], list[tuple]] = {}
        self.capturing: tuple[str, int] | None = None
        self.step_mode: str | None = None
        self.step_pad = 0
        self.step_tokens = 0
        self.live: list[tuple] = []
        self.drained = True
        self.stale: dict[int, torch.Tensor] = {}
        self.inputs: dict[int, list[tuple[int, str]]] = {}
        self.logits_name = "logits_processor"
        self.errors: list[str] = []
        self.stats: dict[str, Any] = {}
        self.stats_reset_step_counters()
        self.stats.update(capture_copies=0, slot_bytes=0, slot_resets=0, profiling_captures=0)

    # ------------------------------------------------------------------ installation (before the engine compiles and captures)

    def install_hooks(self) -> None:
        from vllm.compilation import cuda_graph as cg
        from vllm.compilation import piecewise_backend as pb
        from vllm.model_executor.layers.attention import attention as attn
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.v1.worker.gpu import cudagraph_utils as cgu
        from vllm.v1.worker.gpu import model_runner as mr

        src, into = self, self._hooks
        self.codegen = install_codegen_hooks(self._hook, into)

        def pb_call(orig):
            def call(self_pb, *args):
                if not src.in_cgw:
                    src._enter_piece(self_pb, args)
                return orig(self_pb, *args)
            return call

        def cgw_call(orig):
            def call(self_w, *args, **kw):
                runnable = getattr(self_w, "runnable", None)
                if getattr(runnable, "piecewise_compile_index", None) is None:
                    return orig(self_w, *args, **kw)
                src._enter_piece(runnable, args)
                src.in_cgw = True
                try:
                    return orig(self_w, *args, **kw)
                finally:
                    src.in_cgw = False
            return call

        def capture(orig):
            def run(self_m, create_forward_fn, *a, **kw):
                descs = [d for ds in getattr(self_m, "_capture_descs", {}).values() for d in ds]
                src._begin_capture(max((int(d.num_tokens) for d in descs), default=0))

                def create(desc, warmup):
                    fn = create_forward_fn(desc, warmup=warmup)
                    key = (desc.cg_mode.name, int(desc.num_tokens))

                    def forward_fn(cg_mode):
                        src.plans[key] = []
                        src.capturing = key
                        try:
                            return fn(cg_mode)
                        finally:
                            src.capturing = None
                    return forward_fn
                try:
                    return orig(self_m, create, *a, **kw)
                finally:
                    if getattr(self_m, "_capture_mem_samples", None) is not None:
                        src._end_profiling_capture()            # vLLM's memory profiling: its graphs are discarded, so are their slots
            return run

        def dispatch(orig):
            def run(manager, num_reqs, num_tokens, *a, **kw):
                out = orig(manager, num_reqs, num_tokens, *a, **kw)
                src._note_dispatch(out[0], int(num_tokens))
                return out
            return run

        def attention_init(orig):
            def init(self_a, *a, **kw):
                orig(self_a, *a, **kw)
                impl = getattr(self_a, "impl", None)
                if impl is not None:
                    into.patch(impl, "forward", src._attention_forward(impl.forward, str(getattr(self_a, "layer_name", "attn"))))
            return init

        def logits(orig):
            def fwd(self_l, *a, **kw):
                src.drain()
                out = orig(self_l, *a, **kw)
                job = src._job()
                if job is not None and isinstance(out, torch.Tensor):
                    src._acquire(src.logits_name, out, job)
                return out
            return fwd

        into.wrap(pb.PiecewiseBackend, "__call__", pb_call)
        into.wrap(cg.CUDAGraphWrapper, "__call__", cgw_call)
        into.wrap(cgu.CudaGraphManager, "capture", capture)
        into.wrap(mr, "dispatch_cg_and_sync_dp", dispatch)
        into.wrap(attn.Attention, "__init__", attention_init)
        into.wrap(LogitsProcessor, "forward", logits)

    def remove_hooks(self) -> None:
        """Undo `install_hooks`.  Code Inductor generated while they were installed keeps its hook lines."""
        self._hooks.uninstall()

    def note_engine(self, model: Any) -> None:
        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        self.logits_name = next((n for n, m in model.named_modules() if isinstance(m, LogitsProcessor)), self.logits_name)

    # ------------------------------------------------------------------ where the forward is

    def _enter_piece(self, backend: Any, args: tuple) -> None:
        self.piece = int(backend.piecewise_compile_index)
        self.index = {}
        if self.piece == 0:
            for i, name in self._inputs_of(backend):
                if i < len(args) and isinstance(args[i], torch.Tensor):
                    self._tap(-1, f"c00.input/{name}", args[i])

    def _inputs_of(self, backend: Any) -> list[tuple[int, str]]:
        got = self.inputs.get(id(backend))
        if got is None:
            graph = getattr(getattr(backend, "graph", None), "graph", None)
            places = [n.name for n in graph.nodes if n.op == "placeholder"] if graph is not None else []
            got = [(i, short) for i, p in enumerate(places) for key, short in INPUT_NAMES if key in p]
            if not got:
                self.errors.append(f"piece 0: no token-input placeholder among {places[:8]}")
            self.inputs[id(backend)] = got
        return got

    def _attention_forward(self, orig: Any, layer: str) -> Any:
        src = self

        def forward(*a: Any, **kw: Any) -> Any:
            out = orig(*a, **kw)
            value = kw.get("output")
            if value is None and len(a) > 6:
                value = a[6]
            if value is None:
                value = out
            if isinstance(value, torch.Tensor):
                src._tap(1, f"c{src.piece:02d}.attention/{layer}", value)
            return out
        return forward

    def _begin_capture(self, rows: int) -> None:
        if rows > self.max_rows:
            if self.slots:
                self.stats["slot_resets"] += 1
            self._drop_slots()
            self.max_rows = rows

    def _end_profiling_capture(self) -> None:
        self._drop_slots()
        self.max_rows = 0
        self.stats["profiling_captures"] += 1

    def _drop_slots(self) -> None:
        self.slots, self.plans, self.static = {}, {}, set()
        self.stats["slot_bytes"] = 0

    def _note_dispatch(self, desc: Any, num_tokens: int) -> None:
        if self._job() is None:
            return
        self.step_mode = desc.cg_mode.name
        self.step_pad = int(desc.num_tokens)
        self.step_tokens = num_tokens

    # ------------------------------------------------------------------ the hook: capture -> slot copy; live in a graph step -> slot copy; eager -> acquire

    def _job(self) -> Any:
        return getattr(self.c, "_cur", None) if self.c is not None else None

    def _hook(self, name: str, val: Any) -> None:
        if name.startswith(PREFIX) and isinstance(val, torch.Tensor):
            _, op, buf = name.split("|", 2)
            self._tap(0, f"c{self.piece:02d}.{op}/{buf}", val)

    def _tap(self, sub: int, name: str, val: torch.Tensor) -> None:
        if val.numel() == 0:
            return
        index = self.index.get(sub, 0)
        self.index[sub] = index + 1
        order = (self.piece, sub, index)
        if _capturing(val):
            if self.capturing is None:
                self.errors.append(f"{name}: captured outside a CudaGraphManager capture")
                return
            self.plans[self.capturing].append(self._stage(order, name, val, self.capturing[1]))
            self.stats["capture_copies"] += 1
            return
        if self.capturing is not None:
            # the uncaptured warm-up pass before each capture, and live work between PIECEWISE pieces during one: the slot is
            # allocated here, outside the graph's memory pool, so the capture only records the copy
            self._stage(order, name, val, self.capturing[1])
            return
        job = self._job()
        if job is None:
            self.stats["hook_calls_outside_step"] += 1                  # profiling, warm-up and capture warm-up runs
            return
        if self.step_mode in GRAPH_MODES:
            self.live.append(self._stage(order, name, val, self.step_pad))
            self.stats["live_copies"] += 1
            return
        self.stats["eager_acquisitions"] += 1
        self._acquire(name, val, job)
        if getattr(self.c, "gather", False):
            self.c._nc.flush()

    def _stage(self, order: tuple, name: str, val: torch.Tensor, n_pad: int) -> tuple:
        rows = val.dim() > 0 and int(val.shape[0]) == n_pad and n_pad <= self.max_rows
        if not rows:
            self.static.add((order, name))
        key = (order, name) if rows else (order, name, tuple(val.shape))
        slot = self.slots.get(key)
        if slot is None or slot.dtype != val.dtype or tuple(slot.shape[1:]) != tuple(val.shape[1:]):
            if slot is not None:
                self.errors.append(f"{name}: slot {tuple(slot.shape)}/{slot.dtype} reused for {tuple(val.shape)}/{val.dtype}")
            shape = (self.max_rows, *val.shape[1:]) if rows else tuple(val.shape)
            slot = torch.empty(shape, dtype=val.dtype, device=val.device)
            self.slots[key] = slot
            self.stats["slot_bytes"] += slot.numel() * slot.element_size()
        (slot[: val.shape[0]] if rows else slot).copy_(val)
        return order, name, slot, rows

    def _acquire(self, name: str, val: torch.Tensor, job: Any) -> None:
        if self.omit and self.omit in name:
            self.stats["omitted"] += 1
            return
        self.c._acquire(name, val, job)

    def drain(self) -> None:
        """Acquire this replayed step's slots in canonical order (once per step; a no-op for an eager step)."""
        if self.drained:
            return
        self.drained = True
        job = self._job()
        if job is None or self.step_mode not in GRAPH_MODES:
            return
        t0 = time.perf_counter()
        plan = self.plans.get((self.step_mode, self.step_pad))
        if plan is None:
            self.errors.append(f"step {job.step}: no captured plan for {self.step_mode} at {self.step_pad} tokens")
            return
        late = self.fault == "compiled_late_drain"
        n = self.step_tokens
        for order, name, slot, rows in sorted(plan + self.live, key=lambda e: e[0]):
            value = slot if not rows else slot[: self.step_pad] if (order, name) in self.static else slot[:n]
            if late:
                prev = self.stale.get(id(slot))
                self.stale[id(slot)] = value.clone()
                if prev is not None and prev.shape == value.shape:
                    value = prev
            self._acquire(name, value, job)
            self.stats["replay_drains"] += 1
        self.stats["replayed_steps"] += 1
        self.stats["padded_steps"] += int(n != self.step_pad)
        if getattr(self.c, "gather", False):
            self.c._nc.flush()
        self.stats["drain_s"] += time.perf_counter() - t0

    # ------------------------------------------------------------------ committer lifecycle

    def attach(self, committer: Any) -> None:
        self.c = committer
        committer.attach_source(self)
        committer.external_acquisition = True

    def expected_modules(self) -> list[str]:
        return []

    def extra_step_bytes(self, envelope: dict[str, Any], model: Any) -> dict[str, int]:  # noqa: ARG002
        """What a step adds to the committer's staging bound (`size_arena`), read from the slots the captures allocated: bytes per token
        row, the largest plan's fixed-shape tensors, and the tensor count."""
        per_token = sum(s.numel() * s.element_size() // self.max_rows for k, s in self.slots.items() if len(k) == 2) if self.max_rows else 0
        fixed = max((sum(slot.numel() * slot.element_size() for _o, _n, slot, rows in plan if not rows) for plan in self.plans.values()),
                    default=0)
        return {"per_token_bytes": per_token, "per_step_bytes": fixed, "tensors_per_step": len({k[:2] for k in self.slots})}

    def declaration_classes(self) -> list[dict]:
        return []

    def install(self) -> None:
        pass

    def uninstall(self) -> None:
        pass

    def on_step_begin(self) -> None:
        self.step_mode, self.step_pad, self.step_tokens = None, 0, 0
        self.live = []
        self.drained = False
        self.stats["steps"] += 1

    def on_step_end(self, job: Any) -> None:  # noqa: ARG002
        if self.step_mode is None:
            self.stats["steps_without_dispatch"] += 1
        elif self.step_mode not in GRAPH_MODES:
            self.stats["eager_steps"] += 1
        self.drain()

    def stats_reset_step_counters(self) -> None:
        for k in ("steps", "eager_steps", "replayed_steps", "padded_steps", "steps_without_dispatch", "eager_acquisitions", "live_copies",
                  "replay_drains", "hook_calls_outside_step", "omitted"):
            self.stats[k] = 0
        self.stats["drain_s"] = 0.0

    begin_run = stats_reset_step_counters

    def summary(self) -> dict[str, Any]:
        fixed = sorted({(k[1], k[2]) for k in self.slots if len(k) == 3})
        return {"short_name": self.short_name, "fault": self.fault, "max_rows": self.max_rows, "slots": len(self.slots),
                "plans": {f"{m}:{n}": len(p) for (m, n), p in sorted(self.plans.items())}, "codegen": dict(self.codegen),
                "fixed_shape": {"count": len(fixed), "sample": [f"{n}{list(s)}" for n, s in fixed[:12]]},
                "stats": dict(self.stats), "errors": list(self.errors)[:20]}


# ---------------------------------------------------------------------- the linkage check over committed bytes

def _committed(committer: Any, step: int, name: str) -> torch.Tensor | None:
    blk, _used = committer._gpu_blocks.get(step, (None, 0))
    meta = next((m for m in committer._layouts.get(step, ()) if m.name == name), None)
    if blk is None or meta is None:
        return None
    off = int(meta.dev_off) if int(getattr(meta, "dev_off", -1)) >= 0 else int(meta.stream_off)
    raw = blk[off: off + meta.nbytes]
    dtype = getattr(torch, str(meta.dtype).rsplit(".", 1)[-1])
    return raw.cpu().clone().view(dtype).reshape(meta.shape)


def linkage(committer: Any, ids_name: str = "c00.input/ids", sampled_name: str = "runner.sampler/sampled_token_ids") -> dict[str, Any]:
    """For every request that decodes at two consecutive engine steps, the token id committed as step s+1's input equals the token
    committed as step s's sample.  Read from the committed bytes (a stale or misplaced slot read fails here)."""
    rows = getattr(committer, "step_rows", {}) or {}
    checked, bad = 0, []
    for s in sorted(rows):
        nxt = rows.get(s + 1)
        if not nxt:
            continue
        sampled, ids = _committed(committer, s, sampled_name), _committed(committer, s + 1, ids_name)
        if sampled is None or ids is None:
            continue
        sampled = sampled.reshape(-1)
        here = {rid: (i, hi - lo) for i, (rid, lo, hi) in enumerate(rows[s])}
        for rid, lo, hi in nxt:
            i, k = here.get(rid, (None, 0))
            if i is None or k != 1 or hi - lo != 1 or i >= sampled.numel():
                continue
            checked += 1
            if int(ids[lo]) != int(sampled[i]):
                bad.append({"step": s + 1, "request": rid, "input": int(ids[lo]), "sampled": int(sampled[i])})
    return {"checked": checked, "mismatches": len(bad), "first": bad[:3], "passed": checked > 0 and not bad}
