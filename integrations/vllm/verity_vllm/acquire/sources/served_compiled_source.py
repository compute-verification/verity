"""Served-default compiled acquisition: the values of a torch.compile + CUDA-graph engine at vLLM's serving defaults
(`-O2`: VLLM_COMPILE, Inductor, FULL_AND_PIECEWISE graphs, the V2 model runner, no batch invariance), acquired with no Python
hook inside graph replay.

What one engine step commits, in one canonical order (piece, sub, index):
  * piece 0's token inputs, `c00.input/ids` and `c00.input/positions` (sub -1), named from the compiled piece's placeholders;
  * every buffer Inductor materializes (sub 0): after each scheduled node's codegen the generated wrapper (`inductor_hooks`) calls
    `run_intermediate_hooks('verity|<op>|<buf>', <buf>)` for each of the node's outputs that is allocated and still live (fused-away,
    freed, mutation-only, tuple and non-CUDA outputs are not values of the wrapper), so Triton, extern (cuBLAS `out=`), template,
    combo and fallback kernels are all covered -> `c<piece>.<op>/<buf>`;
  * each attention layer's output between pieces (sub 1), from its impl's `forward` -> `c<piece>.attention/<layer>`;
  * the fused-MoE block's values inside the opaque `vllm.moe_forward` op (sub 2), from its launch sites -> `c<piece>.moe/<plane>`;
  * the logits, outside the compiled model, under the LogitsProcessor's module name (the committer adds the sampled ids).

Three execution paths give the committer the same sequence of (name, bytes):
  * graph capture (FULL: the whole forward under one `torch.cuda.graph`; PIECEWISE: each piece): the hook issues a device copy of
    the tensor into a static slot (one per (piece, sub, index, name), as many rows as the largest capture size), and the capture
    records the slots its (mode, padded size) writes.  The graph replays the copies every step.  Slots are allocated in the
    uncaptured warm-up pass vLLM runs before each capture, outside the graph pool; vLLM's memory-profiling captures are discarded
    with their slots.  With `batch_copies` (the default) a captured hook records its copy and the wrapper's flush lines issue the
    recorded copies together, a multi-tensor copy per flush (one per word width) instead of one graph node per tensor; the flushes sit wherever
    a recorded buffer's bytes could next change (`install_codegen_hooks`), so each copy reads what the immediate copy would.  The
    fused-MoE taps copy at once: they fire inside the op, past every flush line.
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

import json
import os
import time
from typing import Any

import torch

from verity_vllm.acquire.sources.inductor_hooks import FLUSH, PREFIX, install_codegen_hooks
from verity_vllm.engine import hooks

GRAPH_MODES = ("FULL", "PIECEWISE")
INPUT_NAMES = (("input_ids", "ids"), ("positions", "positions"))
WORDS = ((8, torch.int64), (4, torch.int32), (2, torch.int16), (1, torch.uint8))


def _capturing(val: torch.Tensor) -> bool:
    return val.is_cuda and torch.cuda.is_current_stream_capturing()


class ServedCompiledSource:
    short_name = "served_compiled"

    def __init__(self, fault: str | None = None, into: hooks.Hooks | None = None, batch_copies: bool = True) -> None:
        self.fault = fault
        self.batch_copies = batch_copies                                # captured copies issued at the wrapper's flushes, as one copy each
        self.pending: list[tuple[torch.Tensor, torch.Tensor]] = []
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
        self.native_drain = False                                       # measurement: drain through the committer's slot plans
        self.native_plans: dict[tuple[str, int], int] = {}
        self.drained = True
        self.stale: dict[int, torch.Tensor] = {}
        self.inputs: dict[int, list[tuple[int, str]]] = {}
        self.logits_name = "logits_processor"
        self.modules: list[str] = []
        self.moe_configs: dict[str, dict[str, dict[str, Any]]] = {}     # token rows -> w13 ("False") / w2 ("True") -> Triton config
        self.moe_pieces: dict[str, str] = {}                            # piece -> the MoE layer its op ran
        self._moe_rows: int | None = None                               # the token rows of the running MoE block's router
        self.launched: dict[int, tuple[Any, Any]] = {}                  # id -> (autotuner, the config it launched in a step or a capture)
        self.errors: list[str] = []
        self.stats: dict[str, Any] = {}
        self.stats_reset_step_counters()
        self.stats.update(capture_copies=0, slot_bytes=0, slot_resets=0, profiling_captures=0, capture_flushes=0, strided_copies=0)

    # ------------------------------------------------------------------ installation (before the engine compiles and captures)

    def install_hooks(self) -> None:
        from vllm.compilation import cuda_graph as cg
        from vllm.compilation import piecewise_backend as pb
        from vllm.model_executor.layers.attention import attention as attn
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.v1.worker.gpu import cudagraph_utils as cgu
        from vllm.v1.worker.gpu import model_runner as mr

        src, into = self, self._hooks
        self.codegen = install_codegen_hooks(self._hook, into, batch=self.batch_copies)

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
                            src._end_capture(key)
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
        self._install_launch_record(into)
        self._install_moe(into)

    def _install_launch_record(self, into: hooks.Hooks) -> None:
        """The config each Inductor autotuner launches with in a served step or a capture (`launch_configs`), once per autotuner: the
        launcher `run`'s fast path cached at its first launch when that path is taken (whatever `launchers` holds since: a later
        precompile's rblock rescale re-makes two), else the one launcher its run settled on.  Compile-time tuning runs and Inductor's
        benchmark modules launch outside both."""
        from torch._inductor.runtime import triton_heuristics as th

        src = self

        def tuner_run(orig):
            def run(self_t, *a, **kw):
                fast = getattr(self_t, "_cached_launcher", None)
                out = orig(self_t, *a, **kw)
                if id(self_t) not in src.launched and (src.capturing is not None or src._job() is not None):
                    launchers = getattr(self_t, "launchers", None) or []
                    if (fast is not None and set(kw) <= {"stream", "benchmark_run"} and not kw.get("benchmark_run")
                            and not th.autograd_profiler._is_profiler_enabled and not th.get_active_debug_mode()):
                        src.launched[id(self_t)] = (self_t, fast.config)
                    elif len(launchers) == 1:
                        src.launched[id(self_t)] = (self_t, launchers[0].config)
                    else:
                        src.errors.append(f"{getattr(self_t, 'filename', None)}: launched holding {len(launchers)} launchers")
                return out
            return run

        into.wrap(th.CachingAutotuner, "run", tuner_run)

    def _install_moe(self, into: hooks.Hooks) -> None:
        """The fused-MoE block's values, which the opaque `vllm.moe_forward` op computes out of Inductor's sight: its launch sites
        (`MoeBlockSource`'s) tap their outputs as `c<piece>.moe/<plane>` (sub 2), token rows: the router's input logits (the gate's
        output inside the op), weights and ids at `topk_softmax`, `up` at the w13 launch, `act` (its A) and `down` at the w2 launch.
        `moe_sum` writes the op's return, which the wrapper's own hook commits.  Also recorded for the Build: the MoE layer each
        piece's op ran (`moe_pieces`, at the runner's `_forward_impl`: the compiled op takes its layer as an opaque argument) and the
        Triton config of each launch by its token rows and MUL_ROUTED_WEIGHT (`moe_configs`: the expert GEMMs' tensor-core step
        follows it).  A launch of other rows than its router's (a chunked block) is an error."""
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.fused_moe import fused_moe as fm
        from vllm.model_executor.layers.fused_moe.experts import triton_moe
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

        src = self

        def forward_impl(orig):
            def run(self_r, *a, **kw):
                name = str(self_r.layer_name)
                if (was := src.moe_pieces.setdefault(str(src.piece), name)) != name:
                    src.errors.append(f"c{src.piece:02d}: fused-MoE layers {was} and {name}")
                src._moe_rows = None
                return orig(self_r, *a, **kw)
            return run

        def router(orig):
            def topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output, *a, **kw):
                out = orig(topk_weights, topk_ids, token_expert_indices, gating_output, *a, **kw)
                src._moe_rows = rows = int(gating_output.shape[0])
                for plane, t in (("logits", gating_output), ("router_w", topk_weights), ("router_ids", topk_ids)):
                    src._moe_tap(plane, t, rows)
                return out
            return topk_softmax

        def expert(orig):
            def launch(A, B, C, *a, **kw):
                out = orig(A, B, C, *a, **kw)
                routed = kw["mul_routed_weight"] if "mul_routed_weight" in kw else (a[6] if len(a) > 6 else None)
                config = kw["config"] if "config" in kw else (a[8] if len(a) > 8 else None)
                rows = int(C.shape[0])
                if rows != src._moe_rows or routed not in (False, True) or not isinstance(config, dict):
                    src.errors.append(f"c{src.piece:02d}: a fused-MoE launch of {rows} rows (its router's {src._moe_rows}), "
                                      f"MUL_ROUTED_WEIGHT {routed!r}, config {config!r}")
                    return out
                cfg = dict(sorted(config.items()))
                if src.moe_configs.setdefault(str(rows), {}).setdefault(str(routed), cfg) != cfg:
                    src.errors.append(f"c{src.piece:02d}: two fused-MoE configs at {rows} rows, MUL_ROUTED_WEIGHT {routed}")
                if routed:
                    src._moe_tap("act", A, rows)
                    src._moe_tap("down", C, rows)
                else:
                    src._moe_tap("up", C, rows)
                return out
            return launch

        into.wrap(MoERunner, "_forward_impl", forward_impl)
        into.wrap(ops, "topk_softmax", router)
        for mod in (fm, triton_moe):
            into.wrap(mod, "invoke_fused_moe_triton_kernel", expert)

    def _moe_tap(self, plane: str, val: torch.Tensor, rows: int) -> None:
        # Copied at once, batch_copies or not: the op reuses its workspace within one call, past every flush line (TritonExperts'
        # `up` and `down` are both views of workspace2).
        self._tap(2, f"c{self.piece:02d}.moe/{plane}", val.reshape(rows, -1), defer=False)

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

    def _end_capture(self, key: tuple[str, int]) -> None:
        if self.pending:
            self.errors.append(f"capture {key[0]}:{key[1]}: {len(self.pending)} captured copies never flushed")
            self.pending = []

    def _end_profiling_capture(self) -> None:
        self._drop_slots()
        self.max_rows = 0
        self.stats["profiling_captures"] += 1

    def _drop_slots(self) -> None:
        self.slots, self.plans, self.static, self.pending = {}, {}, set(), []
        self.stats["slot_bytes"] = 0
        if self.native_plans and self.c is not None:
            self.c.clear_slot_plans()
        self.native_plans = {}

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
        if name == FLUSH:
            if self.pending:
                self._flush()
            return
        if name.startswith(PREFIX) and isinstance(val, torch.Tensor):
            _, op, buf = name.split("|", 2)
            self._tap(0, f"c{self.piece:02d}.{op}/{buf}", val)

    def _tap(self, sub: int, name: str, val: torch.Tensor, defer: bool = True) -> None:
        if val.numel() == 0:
            return
        index = self.index.get(sub, 0)
        self.index[sub] = index + 1
        order = (self.piece, sub, index)
        if _capturing(val):
            if self.capturing is None:
                self.errors.append(f"{name}: captured outside a CudaGraphManager capture")
                return
            defer = defer and bool(self.codegen.get("batch"))
            self.plans[self.capturing].append(self._stage(order, name, val, self.capturing[1], defer=defer))
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

    def _stage(self, order: tuple, name: str, val: torch.Tensor, n_pad: int, defer: bool = False) -> tuple:
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
        dst = slot[: val.shape[0]] if rows else slot
        if defer:
            self.pending.append((dst, val))                             # holds `val`, so its storage outlives the wrapper's `del`
        else:
            dst.copy_(val)
        return order, name, slot, rows

    def _flush(self) -> None:
        """Issue the recorded captured copies: the dense ones as one multi-tensor copy per word width (the widest that divides a copy's
        length and both its addresses), the strided ones one by one."""
        groups: dict[torch.dtype, tuple[list, list]] = {}
        for d, v in self.pending:
            if not (d.is_contiguous() and v.is_contiguous()):
                d.copy_(v)
                self.stats["strided_copies"] += 1
                continue
            nb = v.numel() * v.element_size()
            word = next(dt for w, dt in WORDS if nb % w == 0 and d.data_ptr() % w == 0 and v.data_ptr() % w == 0)
            dst, src = groups.setdefault(word, ([], []))
            dst.append(d.reshape(-1).view(word))
            src.append(v.reshape(-1).view(word))
        for dst, src in groups.values():
            torch._foreach_copy_(dst, src)
        self.stats["capture_flushes"] += 1
        self.pending = []

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
        if self.native_drain and not late and not self.omit and not self.live:
            key = (self.step_mode, self.step_pad)
            h = self.native_plans.get(key)
            if h is None:
                ordered = sorted(plan, key=lambda e: e[0])
                kinds = [0 if not rows else 2 if (order, name) in self.static else 1 for order, name, _slot, rows in ordered]
                h = self.native_plans[key] = self.c.slot_plan([e[1] for e in ordered], [e[2] for e in ordered], kinds)
            self.stats["replay_drains"] += self.c.stage_slot_plan(h, n, self.step_pad)
            self.stats["replayed_steps"] += 1
            self.stats["padded_steps"] += int(n != self.step_pad)
            if getattr(self.c, "gather", False):
                self.c._nc.flush()
            self.stats["drain_s"] += time.perf_counter() - t0
            return
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

    def use_build(self, build: dict[str, Any]) -> None:
        """The served Build of record (`acquire.served`): the modules of the values it names are the identities every committed step
        must cover (X-03, `identity_check`)."""
        p = build["program"]
        self.modules = sorted({n.split("/", 1)[0] for n in [*p["values"], p["logits"]["name"], p["sampled"]["name"]]})

    def expected_modules(self) -> list[str]:
        return list(self.modules)

    def extra_step_bytes(self, envelope: dict[str, Any], model: Any) -> dict[str, int]:
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

    def on_step_end(self, job: Any) -> None:
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

    def launch_configs(self) -> dict[str, Any]:
        """Each Triton kernel this process holds or launched an autotuner of, by the sha256 of its kernel module's text (the wrapper's
        `async_compile.triton` block, `inductor_templates.source_sha256`): the configs its autotuners compiled (``candidates``), the
        configs served steps and captures launched it with (``executed``, `_install_launch_record`), its live autotuners and the ones
        that launched.  A live autotuner's state is no record of a launch: the one a wrapper's call() runs need not be any module's
        global, nor even the live one of its kernel text (Qwen3-8B's combo kernels: one live autotuner each, never run).  The served
        Build binds each launch to its one executed config (`acquire.served`) and refuses a kernel launched with two."""
        import gc
        import hashlib

        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        def cfg(c: Any) -> dict[str, Any]:
            return {"kwargs": {str(k): v for k, v in (getattr(c, "kwargs", None) or {}).items()},
                    "num_warps": getattr(c, "num_warps", None), "num_stages": getattr(c, "num_stages", None)}

        frozen = gc.get_freeze_count()
        if frozen:                      # vLLM freezes the heap after capture, and get_objects skips what is frozen
            gc.unfreeze()
        try:
            live = [o for o in gc.get_objects() if issubclass(type(o), CachingAutotuner)]
        finally:
            if frozen:
                gc.freeze()
        out: dict[str, Any] = {}
        sha: dict[str, str] = {}

        def record(t: Any) -> dict[str, Any] | None:
            path = getattr(t, "filename", None)
            if not path or not os.path.isfile(path):
                return None
            if path not in sha:
                with open(path, "rb") as f:
                    sha[path] = hashlib.sha256(f.read()).hexdigest()
            rec = out.setdefault(sha[path], {"candidates": [], "executed": [], "autotuners": 0, "launched": 0})
            for c in [cfg(r.config) for r in getattr(t, "compile_results", None) or []]:
                if c not in rec["candidates"]:
                    rec["candidates"].append(c)
            return rec

        for t in live:
            if (rec := record(t)) is not None:
                rec["autotuners"] += 1
        for t, config in self.launched.values():
            rec = record(t)
            if rec is None:
                self.errors.append(f"an autotuner of {getattr(t, 'filename', None)!r} launched with no kernel module on disk")
                continue
            rec["launched"] += 1
            if (c := cfg(config)) not in rec["executed"]:
                rec["executed"].append(c)
        for rec in out.values():
            for k in ("candidates", "executed"):
                rec[k].sort(key=lambda c: json.dumps(c, sort_keys=True))
        return dict(sorted(out.items()))

    def summary(self) -> dict[str, Any]:
        fixed = sorted({(k[1], k[2]) for k in self.slots if len(k) == 3})
        return {"short_name": self.short_name, "fault": self.fault, "max_rows": self.max_rows, "slots": len(self.slots),
                "plans": {f"{m}:{n}": len(p) for (m, n), p in sorted(self.plans.items())}, "codegen": dict(self.codegen),
                "fixed_shape": {"count": len(fixed), "sample": [f"{n}{list(s)}" for n, s in fixed[:12]]},
                "stats": dict(self.stats), "errors": list(self.errors)[:20],
                **({"moe": {"pieces": dict(self.moe_pieces), "configs": self.moe_configs}} if self.moe_pieces else {})}


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
