"""Observe the in-process V2 `GPUModelRunner` step by step into the raw observation log (the capture), and build the run
header the log opens with.

Step observation.  In the pinned wheel the worker calls `runner.execute_model(scheduler_output)` (input preparation
+ model forward; the result stays in `runner.execute_model_state`) and then `runner.sample_tokens(...)`, which
calls `runner.sample(hidden_states, input_batch, grammar_output)` (lm_head GEMM, sampler).  Both instance methods
are wrapped (pattern `capture.py:679-720`); the observer mode is entered separately in each, with phase "forward"
(the runner's own input-preparation dispatcher calls / Triton kernels before the model's root forward carry phase
"other" when module hooks are on) and phase "sample".  Dummy / profile runs and empty steps are passed through.

Log order requires `StepStart` (whose facts are known only after input preparation) before the step's dispatch
events, and `SampleEnd` needs the sampled ids: the step's events are **buffered in memory**; the tiny per-step
buffers the model actually read (`input_ids`, `positions`, the first layer's slot mapping, `logits_indices`, the
gathered block table) and the sampled ids are copied D2H *asynchronously* into pinned host memory right after the
forward / after `sample`, behind a CUDA event, and the step is written out at the start of the next step (or at
close), when the runner itself has long synchronised on its own output copy.  The observer therefore adds no
synchronisation of its own, except `torch.cuda.synchronize()` at the end of a step for which snapshots were
requested, and inside one each time its copies fill the pinned arena (the arena's are written out, and it is reused).

Roots: every `named_parameters()` / `named_buffers()` entry (`Root(role=weight|buffer)`, incl. the rotary
`cos_sin_cache`); the runner's KV caches and input buffers are registered as `StorageAlloc(origin="external")`
with a `Note` naming them (they are not roots: their contents are produced inside the observed computation).
Prompt rows become `Root(role=prompt_token, view=<one-element row view of input_ids>, values=[id])`; decode rows
become `StepLink(step, row, request_id, src_step, value)` where `src_step` is the step whose `SampleEnd` produced
the token (tracked per request from the sampler outputs).

Snapshots (evidence path, `--snapshot-steps`): after every recorded dispatcher call / Triton launch of a listed
step, every non-root tensor argument / output is copied D2H (async, pinned arena, stream-ordered right after the
call) as the raw bytes of the storage range the view spans; the files are content-addressed (sha256) under
`values/<step>/` with an index in `snapshots.json`.  Tensors larger than `snapshot_max_bytes` whose one dimension
equals the KV cache's block count are snapshotted per (leading index, block-in-use) sub-view; other oversize
tensors are listed in the index as skipped.  `snapshot_max_bytes_op` (optional, lower) is the cap applied at
dispatcher-call events only: a large buffer produced by a Triton launch is then copied once, at the launch (where the
Compute's output snapshot is looked up), and not again at every aten view / custom-op call that touches it (the
pinned copies of one step are the capture's peak host memory; OLMoE's fused-MoE intermediates at a 2048-token
prefill are 64 MiB x ~23 events x 16 layers).  Skipped entries record which cap applied.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from verity_vllm.observe.events import (Event, Note, Root, RunEnd, RunHeader, SampleEnd, StepEnd, StepLink, StepStart, StorageAlloc,
                     ViewDesc)
from verity_vllm.observe.log import LogWriter
from verity_vllm.observe.observer import EmptyMode, ModuleNamer, Observer, Recorder, reenter_impl_name
from verity_vllm.observe.storage import StorageAdapter
from verity_vllm.observe import triton_adapter

from verity_vllm.engine import hooks
from verity_vllm.engine.build import Engine
from verity_vllm.observe.pinned import DEVICE_BACKED, IDENTITY_ALIAS_OPS, PinnedArena, device_backed, spanned_bytes, storage_bytes_view
from verity_vllm.engine.run_facts import model_config_subset, profile_manifest


SCHEMA = "verity-capture/log/v1"


@dataclass
class ObserverConfig:
    mode: str = "record"                 # "none" (baseline timing only) | "empty" (EmptyMode) | "record"
    reenter: bool = True
    triton: bool = True
    module_hooks: bool = True
    write_log: bool = True               # False: build every event, discard the buffer at step end (M0 rungs c-f)
    snapshot_steps: frozenset = frozenset()
    snapshot_dir: Path | None = None
    snapshot_max_bytes: int = 32 << 20
    snapshot_max_bytes_op: int | None = None   # cap at dispatcher-call events (None: snapshot_max_bytes); Triton launches keep the full cap
    arena_bytes: int = 1 << 30


@dataclass
class _Step:
    step: int
    events: list[Event] = field(default_factory=list)
    forward_s: float = 0.0
    sample_s: float = 0.0
    t_wall0: float = 0.0
    wall_s: float = 0.0
    facts: dict = field(default_factory=dict)
    input_ids_view: ViewDesc | None = None
    positions_view: ViewDesc | None = None
    sampled_view: ViewDesc | None = None
    logits_view: ViewDesc | None = None
    sampled_shape: tuple[int, int] = (0, 0)
    event: Any = None
    n_snap: int = 0


class Capture:
    """Install with `install()`, run the workload, `close()`.  One instance per run."""

    def __init__(self, engine: Engine, *, header: RunHeader | None, log_path: str | Path | None,
                 config: ObserverConfig | None = None, log: Callable[..., None] = print) -> None:
        self.engine = engine
        self.runner = engine.runner
        self.model = engine.model
        self.cfg = config or ObserverConfig()
        self.log = log
        self.header = header
        self.storage = StorageAdapter()
        self.writer: LogWriter | None = LogWriter(log_path) if (self.cfg.write_log and log_path is not None) else None
        self.rec = Recorder(self.storage, self._sink)
        self.mode = Observer(self.rec, reenter=self.cfg.reenter)
        self.namer: ModuleNamer | None = None
        self.root_allocs: set[int] = set()
        self.roots: list[Root] = []
        self.step_index = 0
        self._cur: _Step | None = None
        self._pending: _Step | None = None
        self._pending_token: dict[str, tuple[int, int, int]] = {}   # request -> (src_step, token, position it feeds)
        self.step_times: list[dict] = []
        self.sample_ids: dict[int, list[int]] = {}                   # step -> sampled ids
        self.step_rows: dict[int, list[tuple[str, int]]] = {}
        self.errors: list[dict] = []
        self.events_written = 0
        self.timing = {"flush_s": 0.0, "snapshot_s": 0.0, "sync_s": 0.0}
        self._hooks = hooks.Hooks()
        self._pin_cache: dict[str, torch.Tensor] = {}
        self._root_handles: list = []
        self._arena: PinnedArena | None = None
        self._snap_pending: list[dict] = []
        self._snap_index: list[dict] = []
        self._snap_bytes = 0
        self._snap_files: dict[str, str] = {}      # sha256 -> the file that holds these bytes (content-addressed across steps)
        self._kv_num_blocks: int | None = None
        self._kv_blocks_in_step: list[int] = []
        self.installed = False

    # ------------------------------------------------------------------ sinks

    def _sink(self, ev: Event) -> None:
        st = self._cur
        if st is not None:
            st.events.append(ev)
        elif self.writer is not None:
            self.writer.write(ev)
            self.events_written += 1

    def _write(self, ev: Event) -> None:
        if self.writer is not None:
            self.writer.write(ev)
            self.events_written += 1

    # ------------------------------------------------------------------ install / uninstall

    def install(self) -> None:
        if self.installed:
            return
        runner, model, cfg = self.runner, self.model, self.cfg
        if self.header is not None:
            self._write(self.header)
        self._install_roots()
        self._install_external_buffers()
        if cfg.mode == "record":
            if cfg.module_hooks:
                self.namer = ModuleNamer(self.rec, model)
                self._install_root_phase_hooks()
            if cfg.triton:
                triton_adapter.install(self.rec)
            if cfg.snapshot_steps:
                self.rec.after_call = self._after_call
                self.rec.after_launch = self._after_launch
        self._wrap_runner()
        self.installed = True

    def uninstall(self) -> None:
        if not self.installed:
            return
        self._finalize_pending()
        self._hooks.uninstall()
        if self.namer is not None:
            self.namer.remove()
            self.namer = None
        for h in getattr(self, "_root_handles", []):
            h.remove()
        self._root_handles = []
        if triton_adapter.is_installed():
            triton_adapter.set_recorder(None)
        self.rec.after_call = None
        self.rec.after_launch = None
        self.installed = False

    def close(self) -> dict:
        self.uninstall()
        stats = self.stats()
        if self.writer is not None:
            self._write(RunEnd(stats))
            self.writer.close()
        if self.cfg.snapshot_dir is not None and self.cfg.snapshot_steps:
            self._write_snapshot_index()
        return stats

    def stats(self) -> dict:
        return {"steps": self.step_index, "events_written": self.events_written, "recorder": dict(self.rec.counts),
                "encode_s": self.rec.encode_s, "storage": dict(self.storage.stats), "timing": dict(self.timing),
                "snapshot_bytes": self._snap_bytes, "snapshot_files": len(self._snap_files), "errors": self.errors[:20],
                "roots": len(self.roots), "step_times": self.step_times}

    def _install_roots(self) -> None:
        seen: set[int] = set()
        for role, items, origin in (("weight", list(self.model.named_parameters()), "param"),
                                    ("buffer", list(self.model.named_buffers()), "buffer")):
            for name, t in items:
                if not isinstance(t, torch.Tensor) or id(t) in seen:
                    continue
                seen.add(id(t))
                view = self.storage.view_of(t, origin=origin)
                self.root_allocs.add(view.alloc)
                for e in self.storage.events_pending():
                    self._write(e)
                r = Root(view, role, name)
                self.roots.append(r)
                self._write(r)

    def _install_external_buffers(self) -> None:
        named: dict[str, dict] = {}
        ib = getattr(self.runner, "input_buffers", None)
        for attr in ("input_ids", "positions"):
            t = getattr(ib, attr, None)
            if isinstance(t, torch.Tensor) and t.numel():
                v = self.storage.view_of(t, origin="external")
                named[f"input_buffers.{attr}"] = {"alloc": v.alloc, "shape": list(v.shape), "dtype": v.dtype}
        kvs = getattr(self.runner, "kv_caches", None) or []
        layer_of_alloc: dict[int, list[str]] = {}
        for name, mod in self.model.named_modules():
            kc = getattr(mod, "kv_cache", None)
            if isinstance(kc, torch.Tensor) and kc.numel():
                v = self.storage.view_of(kc, origin="external")
                layer_of_alloc.setdefault(v.alloc, []).append(name)
                named[f"kv_cache[{name}]"] = {"alloc": v.alloc, "shape": list(v.shape), "dtype": v.dtype, "offset": v.offset,
                                              "strides": list(v.strides)}
        for i, kc in enumerate(kvs):
            if isinstance(kc, torch.Tensor) and kc.numel():
                v = self.storage.view_of(kc, origin="external")
                named.setdefault(f"runner.kv_caches[{i}]", {"alloc": v.alloc, "shape": list(v.shape), "dtype": v.dtype})
        try:
            self._kv_num_blocks = int(self.runner.kv_cache_config.num_blocks)
        except (AttributeError, TypeError, ValueError):
            self._kv_num_blocks = None
        for e in self.storage.events_pending():
            self._write(e)
        try:
            block_size = int(self.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size)
        except (AttributeError, LookupError, TypeError, ValueError):
            block_size = None
        try:  # hybrid KV cache: the KV-cache groups (layer names, block size, spec kind) for the record
            groups_note = [{"layers": [str(x) for x in g.layer_names], "block_size": int(g.kv_cache_spec.block_size),
                            "spec": type(g.kv_cache_spec).__name__} for g in self.runner.kv_cache_config.kv_cache_groups]
        except (AttributeError, TypeError, ValueError):
            groups_note = None
        self._write(Note("external buffers (not roots): runner input buffers and KV caches",
                         {"buffers": named, "kv_num_blocks": self._kv_num_blocks, "block_size": block_size,
                          "kv_cache_groups": groups_note, "storage_weakref_frees": self.storage.use_weakrefs}))

    def _install_root_phase_hooks(self) -> None:
        rec = self.rec

        def pre(mod, args):
            if rec.active and rec.phase == "other":
                rec.phase = "forward"
                rec._root_forward = True

        def post(mod, args, out):
            if rec.active and getattr(rec, "_root_forward", False):
                rec.phase = "other"
                rec._root_forward = False

        self._root_handles = [self.model.register_forward_pre_hook(pre), self.model.register_forward_hook(post)]

    def _wrap_runner(self) -> None:
        runner, cap = self.runner, self
        orig_execute, orig_sample = runner.execute_model, runner.sample

        def execute_model(scheduler_output, *a, **kw):
            dummy = bool(kw.get("dummy_run", False)) or (len(a) >= 2 and bool(a[1]))
            if dummy or getattr(scheduler_output, "total_num_scheduled_tokens", 0) == 0:
                return orig_execute(scheduler_output, *a, **kw)
            cap._finalize_pending()
            st = cap._begin_step(scheduler_output)
            t0 = time.perf_counter()
            try:
                out = cap._run_observed(lambda: orig_execute(scheduler_output, *a, **kw), phase="forward")
            finally:
                st.forward_s = time.perf_counter() - t0
            if cap.cfg.mode == "record":
                cap._after_forward(st)
            return out

        def sample(hidden_states, input_batch, grammar_output, *a, **kw):
            st = cap._cur
            if st is None:
                return orig_sample(hidden_states, input_batch, grammar_output, *a, **kw)
            t0 = time.perf_counter()
            try:
                res = cap._run_observed(lambda: orig_sample(hidden_states, input_batch, grammar_output, *a, **kw), phase="sample")
            finally:
                st.sample_s = time.perf_counter() - t0
            if cap.cfg.mode == "record":
                cap._after_sample(st, res, input_batch)
            else:
                cap._end_timing_only(st)
            return res

        self._hooks.patch(runner, "execute_model", execute_model)
        self._hooks.patch(runner, "sample", sample)

        # the logits tensor is the return value of model.compute_logits (V2 runner: sample() -> compute_logits ->
        # sampler); record its ViewDesc only (no tensor reference is kept)
        model = self.model
        if hasattr(model, "compute_logits"):
            orig_cl = model.compute_logits

            def compute_logits(*a, **kw):
                out = orig_cl(*a, **kw)
                st = cap._cur
                if st is not None and cap.cfg.mode == "record" and isinstance(out, torch.Tensor):
                    try:
                        st.logits_view = cap.storage.view_of(out)
                    except Exception as e:  # noqa: BLE001
                        cap.errors.append({"where": "compute_logits", "step": st.step, "error": repr(e)[:200]})
                return out

            self._hooks.patch(model, "compute_logits", compute_logits)

    def _run_observed(self, fn: Callable[[], Any], *, phase: str) -> Any:
        cfg, rec = self.cfg, self.rec
        if cfg.mode == "none":
            return fn()
        if cfg.mode == "empty":
            with EmptyMode():
                return fn()
        rec.active = True
        rec.phase = ("other" if (phase == "forward" and self.namer is not None) else phase)
        rec._root_forward = False
        rec.open.clear()
        rec.module_stack.clear()
        try:
            with self.mode:
                return fn()
        finally:
            rec.active = False
            rec.phase = "other"

    # ------------------------------------------------------------------ step lifecycle

    def _begin_step(self, scheduler_output: Any) -> _Step:
        st = _Step(self.step_index)
        st.t_wall0 = time.perf_counter()
        st.facts["sched_req_ids"] = list(getattr(scheduler_output, "num_scheduled_tokens", {}).keys())
        self.rec.step = st.step
        self._cur = st
        if st.step in self.cfg.snapshot_steps and self.cfg.mode == "record":
            if self._arena is None:
                self._arena = PinnedArena(self.cfg.arena_bytes)
            self._arena.reset()
            self._snap_pending = []
            self._kv_blocks_in_step = []
        return st

    def _pin(self, name: str, n: int, dtype: torch.dtype, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        """A (cached, grow-only) pinned host buffer for the async D2H copies of the step facts."""
        shape = shape or (n,)
        need = max(int(np.prod(shape)), 1)
        t = self._pin_cache.get(name)
        if t is None or t.numel() < need or t.dtype != dtype:
            t = torch.empty(need, dtype=dtype, pin_memory=True)
            self._pin_cache[name] = t
        return t[:need].view(*shape)

    def _after_forward(self, st: _Step) -> None:
        """Host-side facts + async D2H copies of the buffers the model read (stream-ordered after the forward)."""
        try:
            ems = self.runner.execute_model_state
            ib = ems.input_batch
            from verity_vllm.engine.engine_driver import external_id

            n = int(ib.num_reqs)
            qsl = np.asarray(ib.query_start_loc_np[: n + 1]).astype(np.int64).copy()
            nct = np.asarray(ib.num_computed_tokens_np[:n]).astype(np.int64).copy()
            idx = np.asarray(ib.idx_mapping_np[:n]).astype(np.int64).copy()
            num_tokens = int(ib.num_tokens)
            num_padded = int(ib.num_tokens_after_padding)
            req_ids = [external_id(r) for r in ib.req_ids[:n]]
            rs = self.runner.req_states
            prompt_len = {rid: int(rs.prompt_len.np[int(idx[i])]) for i, rid in enumerate(req_ids)}
            bt = self.runner.block_tables
            nblocks = [int(bt.num_blocks.np[0, int(idx[i])]) for i in range(n)] if bt.num_kv_cache_groups else [0] * n
            block_size = int(self.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size)
            st.facts = {"n": n, "qsl": qsl, "nct": nct, "req_ids": req_ids, "num_tokens": num_tokens, "num_padded": num_padded,
                        "prompt_len": prompt_len, "nblocks": nblocks, "block_size": block_size,
                        "num_logits": int(ib.logits_indices.numel())}
            st.input_ids_view = self.storage.view_of(ib.input_ids)
            st.positions_view = self.storage.view_of(ib.positions)
            # async D2H of the tiny buffers the model actually read
            ids = self._pin("input_ids", num_tokens, ib.input_ids.dtype)
            ids.copy_(ib.input_ids[:num_tokens], non_blocking=True)
            pos = self._pin("positions", num_tokens, ib.positions.dtype)
            pos.copy_(ib.positions[:num_tokens], non_blocking=True)
            li = self._pin("logits_indices", int(ib.logits_indices.numel()), ib.logits_indices.dtype)
            li.copy_(ib.logits_indices, non_blocking=True)
            sm = ems.slot_mappings_by_layer
            slot_src = None
            if sm:
                first = next(iter(sm.values()))
                st.facts["slot_mapping_distinct"] = len({id(v) for v in sm.values()})
                st.facts["slot_mapping_layer"] = next(iter(sm.keys()))
                slot_src = first[:num_tokens]
                slots = self._pin("slots", num_tokens, first.dtype)
                slots.copy_(slot_src, non_blocking=True)
            if bt.num_kv_cache_groups and n:
                gathered = bt.input_block_tables[0]
                maxb = max(nblocks) if nblocks else 0
                if maxb:
                    bts = self._pin("block_tables", n * maxb, gathered.dtype, (n, maxb))
                    bts.copy_(gathered[:n, :maxb], non_blocking=True)
            # [r8 E-cleanup, hybrid KV cache] several KV-cache groups (vLLM hybrid KV-cache manager; Gemma-2: sliding-window
            # layers vs global layers): every group has its own block tables and slot mapping into the SAME physical KV
            # tensors, and the fields above are group 0's only.  Record every group (+ its layer names, so the fold can
            # address a layer's KV rows by ITS group).  Single-group runs record nothing extra (byte-identical logs).
            ngroups = int(bt.num_kv_cache_groups or 0)
            if ngroups > 1 and n:
                try:
                    groups_meta = []
                    kv_groups = self.runner.kv_cache_config.kv_cache_groups
                    for g in range(ngroups):
                        layers = [str(x) for x in getattr(kv_groups[g], "layer_names", [])]
                        nb_g = [int(bt.num_blocks.np[g, int(idx[i])]) for i in range(n)]
                        maxb_g = max(nb_g) if nb_g else 0
                        if maxb_g:
                            gathered_g = bt.input_block_tables[g]
                            bts_g = self._pin(f"block_tables_g{g}", n * maxb_g, gathered_g.dtype, (n, maxb_g))
                            bts_g.copy_(gathered_g[:n, :maxb_g], non_blocking=True)
                        sm_layer = next((ln for ln in layers if sm and ln in sm), None)
                        if sm_layer is not None:
                            slots_g = self._pin(f"slots_g{g}", num_tokens, sm[sm_layer].dtype)
                            slots_g.copy_(sm[sm_layer][:num_tokens], non_blocking=True)
                        groups_meta.append({"layers": layers, "block_size": int(kv_groups[g].kv_cache_spec.block_size),
                                            "nblocks": nb_g, "slot_layer": sm_layer})
                    st.facts["kv_groups"] = groups_meta
                except Exception as e:  # noqa: BLE001 - recorded, never silent: the StepStart then carries `ngroups` EMPTY
                    # groups, so every KV access of the step is Unsupported at resolve (REJECTED) instead of mis-addressed
                    self.errors.append({"where": "after_forward.kv_groups", "step": st.step, "error": repr(e)[:400]})
                    st.facts["kv_groups"] = [{"layers": [], "block_size": block_size, "nblocks": [0] * n, "slot_layer": None,
                                              "error": repr(e)[:400]} for _ in range(ngroups)]
            st.event = torch.cuda.Event()
            st.event.record()
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "after_forward", "step": st.step, "error": repr(e)[:400]})
            st.facts.setdefault("error", repr(e)[:400])

    def _after_sample(self, st: _Step, res: Any, input_batch: Any) -> None:
        try:
            so = res[0] if isinstance(res, tuple) else res
            sampled = so.sampled_token_ids
            st.sampled_view = self.storage.view_of(sampled)
            s2 = sampled.reshape(sampled.shape[0], -1) if sampled.dim() != 2 else sampled
            st.sampled_shape = (int(s2.shape[0]), int(s2.shape[1]))
            host = self._pin("sampled", int(s2.numel()), s2.dtype, st.sampled_shape)
            host.copy_(s2, non_blocking=True)
            if st.event is None:
                st.event = torch.cuda.Event()
            st.event.record()
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "after_sample", "step": st.step, "error": repr(e)[:400]})
        st.wall_s = time.perf_counter() - st.t_wall0
        if st.step in self.cfg.snapshot_steps and self.cfg.mode == "record":
            self._flush_snapshots(st)
        # the step is complete; it is written out at the next step start (or at close)
        self._cur = None
        self._pending = st
        self.step_index += 1
        # non-snapshot steps: the deferred write happens at the next execute_model; keep the buffer

    def _end_timing_only(self, st: _Step) -> None:
        """M0 rungs (a)/(b): no facts, no events; only the per-step timing."""
        st.wall_s = time.perf_counter() - st.t_wall0
        self.step_times.append({"step": st.step, "forward_s": st.forward_s, "sample_s": st.sample_s, "wall_s": st.wall_s,
                                "num_reqs": len(st.facts.get("sched_req_ids", [])), "events": 0, "snapshots": 0})
        self._cur = None
        self.step_index += 1

    def _finalize_pending(self) -> None:
        st = self._pending
        if st is None:
            return
        self._pending = None
        t0 = time.perf_counter()
        try:
            if st.event is not None:
                st.event.synchronize()
            self._emit_step(st)
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "finalize", "step": st.step, "error": repr(e)[:400]})
            # still write what we have so the log stays parseable
            self._write(Note("step finalisation failed", {"step": st.step, "error": repr(e)[:400]}))
            for ev in st.events:
                self._write(ev)
            self._write(StepEnd(st.step, st.forward_s, st.sample_s))
        st.events = []
        if self.writer is not None:
            self.writer.flush()
        self.timing["flush_s"] += time.perf_counter() - t0

    def _emit_step(self, st: _Step) -> None:
        f = st.facts
        if "n" not in f:
            raise RuntimeError(f"no step facts: {f.get('error')}")
        n, qsl, nct, req_ids = f["n"], f["qsl"], f["nct"], f["req_ids"]
        num_tokens = f["num_tokens"]
        ids = self._pin_cache["input_ids"][:num_tokens].tolist()
        pos_dev = self._pin_cache["positions"][:num_tokens].tolist()
        rows: list[tuple[str, int]] = []
        positions: list[int] = []
        for i in range(n):
            for k in range(int(qsl[i]), int(qsl[i + 1])):
                rows.append((req_ids[i], int(nct[i] + (k - qsl[i]))))
                positions.append(int(nct[i] + (k - qsl[i])))
        if pos_dev != positions:
            st.events.insert(0, Note("device positions differ from host-derived positions", {"step": st.step, "device": pos_dev[:8], "host": positions[:8]}))
            positions = [int(p) for p in pos_dev]
            rows = [(r, p) for (r, _), p in zip(rows, positions)]
        seq_lens = {req_ids[i]: int(nct[i] + (qsl[i + 1] - qsl[i])) for i in range(n)}
        num_computed = {req_ids[i]: int(nct[i]) for i in range(n)}
        nblocks = f["nblocks"]
        block_tables: dict[str, list[int]] = {}
        if max(nblocks, default=0) and "block_tables" in self._pin_cache:
            bt_host = self._pin_cache["block_tables"][: n * max(nblocks)].view(n, max(nblocks)).tolist()
            for i in range(n):
                block_tables[req_ids[i]] = [int(b) for b in bt_host[i][: nblocks[i]]]
        else:
            block_tables = {rid: [] for rid in req_ids}
        slot_mapping = self._pin_cache["slots"][:num_tokens].tolist() if "slots" in self._pin_cache and "slot_mapping_layer" in f else []
        nl = f["num_logits"]
        logits_indices = [int(x) for x in self._pin_cache["logits_indices"][:nl].tolist()]
        logits_rows = [(req_ids[i], int(nct[i] + (qsl[i + 1] - qsl[i]) - 1)) for i in range(n)]
        if len(logits_indices) != len(logits_rows):
            logits_rows = [rows[j] if 0 <= j < len(rows) else ("?", -1) for j in logits_indices]
        prompt_len = f["prompt_len"]
        phases = {"prefill" if p < prompt_len.get(r, 0) else "decode" for r, p in rows}
        phase_hint = phases.pop() if len(phases) == 1 else "mixed"
        kv_groups: list[dict] | None = None
        if f.get("kv_groups"):
            # [r8 E-cleanup, hybrid KV cache] one entry per KV-cache group (see `_after_forward`)
            kv_groups = []
            for g, meta in enumerate(f["kv_groups"]):
                nb_g = meta["nblocks"]
                tables: dict[str, list[int]] = {rid: [] for rid in req_ids}
                key = f"block_tables_g{g}"
                if max(nb_g, default=0) and key in self._pin_cache:
                    host = self._pin_cache[key][: n * max(nb_g)].view(n, max(nb_g)).tolist()
                    for i in range(n):
                        tables[req_ids[i]] = [int(b) for b in host[i][: nb_g[i]]]
                skey = f"slots_g{g}"
                sm_g = [int(s) for s in self._pin_cache[skey][:num_tokens].tolist()] if meta.get("slot_layer") and skey in self._pin_cache else []
                entry = {"layers": list(meta["layers"]), "block_size": int(meta["block_size"]), "block_tables": tables,
                         "slot_mapping": sm_g}
                if meta.get("error"):
                    entry["error"] = meta["error"]
                kv_groups.append(entry)
        start = StepStart(st.step, rows, [int(t) for t in ids], positions, seq_lens, num_computed, block_tables, f["block_size"],
                          [int(s) for s in slot_mapping], logits_rows, logits_indices, n, num_tokens, f["num_padded"],
                          st.input_ids_view, st.positions_view, phase_hint, kv_groups)
        # roots (prompt rows) and links (decode rows)
        roots: list[Root] = []
        links: list[StepLink] = []
        iv = st.input_ids_view
        for row, ((rid, p), tok) in enumerate(zip(rows, ids)):
            if p < prompt_len.get(rid, 0):
                if iv is not None:
                    rv = ViewDesc(iv.alloc, iv.offset + row * (iv.strides[0] if iv.strides else 1), (1,), (1,), iv.dtype, iv.device)
                    roots.append(Root(rv, "prompt_token", f"prompt[{rid}][{p}]", [int(tok)]))
            else:
                src = self._pending_token.get(rid)
                if src is not None and src[2] == p:
                    links.append(StepLink(st.step, row, rid, src[0], "eq", int(src[1])))
                    if int(src[1]) != int(tok):
                        st.events.insert(0, Note("StepLink value != input token", {"step": st.step, "row": row, "request_id": rid,
                                                                                  "link_value": int(src[1]), "input": int(tok)}))
                else:
                    st.events.insert(0, Note("decode row without a known source step", {"step": st.step, "row": row, "request_id": rid,
                                                                                         "position": p, "known": None if src is None else list(src)}))
        # sampled ids
        sampled_ids: list[int] = []
        if st.sampled_shape[0]:
            host = self._pin_cache["sampled"][: st.sampled_shape[0] * st.sampled_shape[1]].view(*st.sampled_shape)
            sampled_ids = [int(v) for v in host[:, 0].tolist()]
        sample_end = SampleEnd(st.step, st.logits_view, sampled_ids, st.sampled_view)
        for i, (rid, p) in enumerate(logits_rows):
            if i < len(sampled_ids):
                self._pending_token[rid] = (st.step, sampled_ids[i], p + 1)
        self.sample_ids[st.step] = sampled_ids
        self.step_rows[st.step] = rows
        # write: pending StorageAllocs (views referenced by StepStart / SampleEnd), StepStart, roots, links, the
        # buffered events, then the pending StorageFrees (deaths of this step's temporaries observed after the
        # step's events were buffered -- they must follow the StorageAllocs inside the buffer), SampleEnd, StepEnd
        pending = self.storage.events_pending()
        for e in pending:
            if isinstance(e, StorageAlloc):
                self._write(e)
        self._write(start)
        for r in roots:
            self._write(r)
        for l in links:
            self._write(l)
        for ev in st.events:
            self._write(ev)
        for e in pending:
            if not isinstance(e, StorageAlloc):
                self._write(e)
        self._write(sample_end)
        self._write(StepEnd(st.step, st.forward_s, st.sample_s))
        self.step_times.append({"step": st.step, "num_tokens": num_tokens, "num_reqs": n, "forward_s": st.forward_s,
                                "sample_s": st.sample_s, "wall_s": st.wall_s, "events": len(st.events), "phase": phase_hint,
                                "snapshots": st.n_snap})

    # ------------------------------------------------------------------ snapshots

    def _after_call(self, eid: int, op: str, args: tuple, kwargs: dict, out: Any) -> None:
        st = self._cur
        if st is None or st.step not in self.cfg.snapshot_steps:
            return
        if op in IDENTITY_ALIAS_OPS:      # no new value: its tensors are snapshotted where they are produced or read
            return
        try:
            from torch.utils._python_dispatch import _disable_current_modes

            cap = self.cfg.snapshot_max_bytes if self.cfg.snapshot_max_bytes_op is None else self.cfg.snapshot_max_bytes_op
            with _disable_current_modes():
                self._snap_struct(eid, "args", args, cap)
                for k, v in kwargs.items():
                    self._snap_struct(eid, f"kwargs.{k}", v, cap)
                self._snap_struct(eid, "outs", out, cap)
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "snapshot", "step": st.step, "id": eid, "error": repr(e)[:300]})

    def _after_launch(self, eid: int, fn: str, bound: dict) -> None:
        st = self._cur
        if st is None or st.step not in self.cfg.snapshot_steps:
            return
        try:
            from torch.utils._python_dispatch import _disable_current_modes

            with _disable_current_modes():
                for k, v in bound.items():
                    if isinstance(v, torch.Tensor):
                        self._snap_tensor(eid, f"args.{k}", v)
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "snapshot-triton", "step": st.step, "id": eid, "error": repr(e)[:300]})

    def _snap_struct(self, eid: int, path: str, x: Any, cap: int | None = None) -> None:
        if isinstance(x, torch.Tensor):
            self._snap_tensor(eid, path, x, cap=cap)
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                if isinstance(v, (torch.Tensor, list, tuple)):
                    self._snap_struct(eid, f"{path}[{i}]", v, cap)

    def _snap_tensor(self, eid: int, path: str, t: torch.Tensor, sub: dict | None = None, cap: int | None = None) -> None:
        if t.numel() == 0:
            return
        view = self.storage.view_of(t)
        if view.alloc in self.root_allocs:
            return
        full = self.cfg.snapshot_max_bytes
        cap = full if cap is None else cap
        off, nbytes = spanned_bytes(t)
        why = DEVICE_BACKED if device_backed(t) else None
        if why is None and nbytes > cap:
            nb = self._kv_num_blocks
            bdim = next((d for d, s in enumerate(t.shape) if nb and s == nb), None)
            if bdim is not None and sub is None:
                blocks = self._blocks_in_use()
                lead = [()] if bdim == 0 else [(k,) for k in range(int(t.shape[0]))] if bdim == 1 else None
                if lead is not None and blocks:
                    for l in lead:
                        for b in blocks:
                            v = t
                            for k in l:
                                v = v.select(0, k)
                            v = v.select(0, b)
                            self._snap_tensor(eid, path, v, sub={"dim": bdim, "index": list(l) + [b]}, cap=cap)
                    return
            why = f"spanned bytes {nbytes} > snapshot_max_bytes {cap}"
            if cap != full:
                why += f" (snapshot_max_bytes_op: dispatcher-call event; Triton launch events keep {full})"
        if why is not None:
            self._snap_index.append({"event": eid, "path": path, "view": view.to_json(), "dtype": view.dtype, "shape": list(view.shape),
                                     "skipped": why})
            return
        src = storage_bytes_view(t, off, nbytes)
        if self._snap_pending and not self._arena.fits(nbytes):
            self._flush_snapshots(self._cur)
        host = self._arena.take(nbytes)
        host.copy_(src, non_blocking=True)
        entry = {"event": eid, "path": path, "view": view.to_json(), "dtype": view.dtype, "shape": list(view.shape),
                 "strides": list(view.strides), "byte_offset": off, "nbytes": nbytes, "layout": "spanned-storage-bytes"}
        if sub is not None:
            entry["sub"] = sub
        self._snap_pending.append({"host": host, "entry": entry})
        if self._cur is not None:
            self._cur.n_snap += 1

    def _blocks_in_use(self) -> list[int]:
        """The physical KV blocks of every request scheduled in the current step (read once per snapshot step from the
        runner's block table, whose staged writes are applied at the start of execute_model; this read synchronises,
        which is allowed in snapshot steps only)."""
        if self._kv_blocks_in_step or self._cur is None:
            return self._kv_blocks_in_step
        try:
            bt = self.runner.block_tables
            rs = self.runner.req_states
            blocks: set[int] = set()
            # [r8 E-cleanup, hybrid KV cache] the blocks of EVERY KV-cache group: with several groups the physical tensors
            # are shared and a group-1 layer's rows live in group-1 blocks (group 0 alone left them un-snapshotted)
            for g in range(int(bt.num_kv_cache_groups or 1)):
                for rid in self._cur.facts.get("sched_req_ids", []):
                    ri = rs.req_id_to_index.get(rid)
                    if ri is None:
                        continue
                    count = int(bt.num_blocks.np[g, ri])
                    if count:
                        blocks.update(int(b) for b in bt.block_tables[g].gpu[ri, :count].cpu().tolist())
            self._kv_blocks_in_step = sorted(blocks)
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "blocks_in_use", "error": repr(e)[:300]})
            self._kv_blocks_in_step = []
        return self._kv_blocks_in_step

    def _flush_snapshots(self, st: _Step) -> None:
        """Wait for the step's copies and write them out: at a snapshot step's end, and whenever they fill the pinned arena mid-step."""
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        self.timing["sync_s"] += (t1 := time.perf_counter()) - t0
        d = Path(self.cfg.snapshot_dir) / "values" / str(st.step)
        d.mkdir(parents=True, exist_ok=True)
        for item in self._snap_pending:
            host: torch.Tensor = item["host"]
            entry = item["entry"]
            data = host.numpy().tobytes()
            sha = hashlib.sha256(data).hexdigest()
            # content-addressed across steps: identical bytes seen at an earlier step share that step's file.  (Before the
            # generalization sweep the entry pointed at values/<this step>/ even when the bytes had been written under an
            # earlier step: 4430 of 16334 index entries of a 16-step SmolLM2 capture named files that did not exist.)
            fname = self._snap_files.get(sha)
            if fname is None:
                fname = f"values/{st.step}/{sha[:16]}.bin"
                (Path(self.cfg.snapshot_dir) / fname).write_bytes(data)
                self._snap_files[sha] = fname
                self._snap_bytes += len(data)
            entry["sha256"] = sha
            entry["file"] = fname
            entry["step"] = st.step
            self._snap_index.append(entry)
        self._snap_pending = []
        self._arena.reset()
        self.timing["snapshot_s"] += time.perf_counter() - t1

    def _write_snapshot_index(self) -> None:
        p = Path(self.cfg.snapshot_dir) / "snapshots.json"
        p.write_text(json.dumps({"schema": "verity-capture/snapshots/v1", "layout": "spanned-storage-bytes",
                                 "note": "file = raw bytes of the storage range [byte_offset, byte_offset+nbytes) the view spans, "
                                         "copied D2H stream-ordered right after the call; content-addressed (sha256) so identical "
                                         "bytes share a file; 'sub' = sub-view (dim, index) of an oversize KV-cache tensor",
                                 "steps": sorted(self.cfg.snapshot_steps), "total_bytes": self._snap_bytes,
                                 "snapshot_max_bytes": self.cfg.snapshot_max_bytes, "snapshot_max_bytes_op": self.cfg.snapshot_max_bytes_op,
                                 "entries": self._snap_index}, separators=(",", ":")))


def make_header(engine: Engine, workload: dict, *, adapter_revision: str, wheel_sha256: str | None = None,
                vllm_commit: str | None = None, extra_notes: dict | None = None) -> RunHeader:
    manifest, pid, notes = profile_manifest(engine, adapter_revision=adapter_revision, wheel_sha256=wheel_sha256,
                                            vllm_commit=vllm_commit, sampling=workload["sampling"])
    notes["reenter_impls"] = {op: reenter_impl_name(op) for op in ("vllm.unified_attention_with_output", "vllm.unified_kv_cache_update")}
    notes["phases"] = "forward = inside the model's root forward; other = the runner's input preparation / post-forward work inside " \
                      "execute_model; sample = runner.sample (lm_head GEMM + sampler)"
    notes["observer_sync"] = "no observer-issued synchronisation; step facts + sampled ids are copied D2H asynchronously behind a " \
                             "CUDA event and read at the next step start; torch.cuda.synchronize only at the end of snapshot steps"
    if extra_notes:
        notes.update(extra_notes)
    model = {"repo": engine.cp["repo"], "revision": engine.cp["revision"], "local_path": engine.cp.get("local_path"),
             "case": engine.case, "config": model_config_subset(engine)}
    wl = {"requests": workload["requests"], "sampling": workload["sampling"], "source": workload.get("source"), "sha256": workload.get("sha256")}
    # n > 1: the request ids the runner sees are vLLM's children (`parallel_sampling.ParentRequest.get_child_info`:
    # id f"{index}_{request_id}", seed + index when seeded).  Recorded as a fact of the fork; the request record itself is unchanged.
    children = [{"request_id": f"{i}_{r['request_id']}", "parent": r["request_id"], "index": i,
                 "seed": None if r.get("seed") is None else int(r["seed"]) + i}
                for r in workload["requests"] for i in range(int(r.get("n") or 1)) if int(r.get("n") or 1) > 1]
    if children:
        wl["children"] = children
    return RunHeader(SCHEMA, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), pid, manifest, model, wl, notes)
