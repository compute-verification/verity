"""Native collection at the existing hook boundaries: the same committed values and layout as `NativeHostCommitter`.
This module is VALUE ACQUISITION (taking each required Value's bytes at its runtime occurrence, the module
hook) feeding the commitment; the `_flush_*` / `*_LEAVES` tables below are the hand-written runtime correspondence (which module
leaf produces which Value, and when its bytes are stable), until recorded correspondence + an acquisition plan replace them.

What differs from `NativeHostCommitter`'s Python hook path -- ONLY the serving-thread acquisition:

  native_host hook body (per tensor, Python)   this hook body (per tensor)
  --------------------------------------       -----------------------------------------------------------------------
  torch forward hook dispatch (_call_impl      wrapped `module.forward` (instance attribute; torch's hook-free fast path) -- flag `wrap`;
    slow path)                                   `wrap=False` keeps register_forward_hook for the ablation
  device/contiguity/dtype/shape reads,         `Collector.collect(name_id, tensor)`: ONE pybind call; the plan entry (name, dtype, shape,
    TensorMeta(...) construction, first_leaf     nbytes, arena offset, zero-gap) was PRECOMPUTED on the first step of this step-shape class
    arithmetic, dtype dict, perf_counter x2      and is checked, not rebuilt; then one cudaMemcpyAsync (D2D) on the PRODUCER's current
  ctypes cudaMemcpyAsync (+ memset for gaps)     stream (+ the same memset for the same gaps)
  per-step Python list of TensorMeta           per-plan cached record (plan_cache) + cached layout digest, built on the WORKER thread once per
                                                 plan version (identical fields => identical layout digest => identical step root)

Ordering: the copy is enqueued on the producer's current stream at the hook, i.e. after the kernel that produced the value and before any
later kernel on that stream can overwrite it in place (fused add-RMSNorm residual, RoPE in the qkv buffer) -- the known-safe placement.
No collector stream and no per-tensor event are needed for that; the step event (recorded on the same stream at step end) orders the
worker's side-stream hashing after every snapshot, exactly as in `NativeHostCommitter`.  A lifetime table may later license combining several
copies into one native call, only within proven-safe windows; this file does not do that.

The step's packed stream goes through a ring of device windows to the native worker (`native_collect_v2b`), which commits it under
rows-sha512 (`step_rows`): every 4 KiB row an hm96-sha512 hiding leaf in a frame-v3-sha512 tree, salted from the step's fresh key.
Retention, lazy openings, roots and the verifier are `NativeHostCommitter`'s.

Stress / negative-test knobs (never in headline runs; recorded when used):
  VERITY_CONSUMER_DELAY_MS=<ms>   the worker sleeps before consuming each step (delayed commitment consumer): backlog grows, the arena pool
                                  must grow instead of recycling a block early; openings must still verify 64/64
  VERITY_FAULT=release_before_keep  DELIBERATE ordering bug: the window slot is released for reuse BEFORE the retained copy (openings read
                                  it) is taken, and the copy is taken only after the ring refilled the slot (native wait_slot_reused):
                                  tree correct, retained bytes wrong -> openings (after release) FAIL named.
  VERITY_FAULT=drop1_dup0         DELIBERATE identity bug (a binding negative): every layer>0 input_layernorm binds Value /0 twice and drops /1
                                  -> counts and classes unchanged; the required-value coverage check MUST fail (Python wrappers are used for every module)
  VERITY_FAULT=skip_copy:<k>      DELIBERATE value bug: the k-th snapshot of every step is not issued (stale window bytes) -> the root must
                                  differ from a clean run's under the same row keys while the tree stays self-consistent (openings alone cannot see it)
  VERITY_FAULT=hang_after_dead    DELIBERATE hang: after the slot-wait timeout declared the collector dead, finalize
                                  blocks forever instead of failing the run closed -> the C++ dead-watchdog alone must end the process
                                  (rc 3 + <out>/COLLECTOR_DEAD_HUNG.json) within VERITY_COLLECT_DEAD_GRACE_S (default 120 s).
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any

import torch

from verity.ml.commitments.merkle import merkle_depth
from verity_vllm import settings
from verity_vllm.commit import step_rows
from verity_vllm.commit.committer.flush_points import GATHER_FLUSH_LEAVES, PRE_FLUSH_LEAVES
from verity_vllm.commit.committer.host_retention import (HIDDEN_M1_FAMILIES, RETAIN_EXCLUDE_ALIASES, _excluded_chunk_ranges,  # noqa: F401
                                                         _HostRanges, _HostWindows, outside, retain_exclude_families)
from verity_vllm.commit.committer.native_ext import ext
from verity_vllm.commit.committer.native_hidden import NativeHiddenM1Source, NativeHiddenM1SourceFA3
from verity_vllm.commit.committer.native_host import (
    NativeHostCommitter,
    TensorMeta,
    _StepJob,
    memtrace,
)
from verity_vllm.commit.committer.plan_cache import PlanCache, PlanMetas
from verity_vllm.commit.committer.topp_splits import (SPLITS_IDENTITY, TOPP_SPLIT_KERNELS, TOPP_SPLIT_MODULE, _SplitKernelProxy,  # noqa: F401
                                                      batch_top_p, install_split_kernel_proxies, split_count_of_launches, step_splits,
                                                      uninstall_split_kernel_proxies)

def plan_key_of(sched: Any, ntok: int, nreq: int, targets: bytes = b"") -> tuple[int, str]:
    """The collector's plan key for one engine step = the step's SHAPE CLASS, i.e. everything the staged byte layout depends on.

    (ntok, nreq) alone is not the layout: the FA3 hidden M1 export
    plane (`fa3_hidden_m1`) is sized by each request's KV length (BN-blocks of the sequence), so two decode steps with the same
    (ntok, nreq) stage different byte counts (the B8 mixed 1024/128 workload's steps 35 and 80: both ntok 4 / nreq 4, the same
    four live requests 45 positions apart, 78,892,288 vs 87,297,280 bytes), and a step that met another's plan fails CLOSED by name, as the
    bounded path must (a hit-plan step hashes in place and keeps no host copy to re-hash from).  So the key binds the per-request
    geometry as a multiset:

        key = blake2b-64( (ntok, nreq, sorted[(num_computed_tokens_i, num_scheduled_tokens_i) for every scheduled request]) ) & (2**63-1)

    Two steps with equal keys have equal per-request (KV length, new tokens) multisets, hence equal staged shapes for every plane whose
    size is a function of those (module outputs: ntok; hidden M1: KV blocks per request; token ids: ntok/nreq).  Row ORDER inside a
    tensor is not bound -- a leaf's frame domain binds the step's row count only, and the plan is a sequence of (name, bytes) in hook order.
    Between the warm-up (learn) run and the record run of the same declared, batch-invariant, greedy workload the schedule is identical,
    so every record step hits the plan its warm-up twin learned; the plan cache holds one entry per distinct step (<= steps x pairs).

    Some planes follow the engine step, not the geometry: the int8 source's Int8RowQuant_v1 sites and the call-boundary bodies are
    the manifest's, per engine step, and a served window restarts its steps at 0.  A bench manifest of 32 engine steps stages the
    quantizers' (x_q, s_a) at window steps 0..31 and none after, so a lone request after its batch-mates finish meets the same
    geometry at window step 25 in one window and 64 in the next, and stages different bytes.  `targets`, the digest of the names
    those sources commit at the step (`NativeCollectCommitter.step_targets`), is hashed in with the geometry; with none (b"") the key
    is the geometry's alone.
    Returns (key, mode): mode "geometry" when the scheduler output exposes the per-request fields (vLLM V1 `SchedulerOutput`:
    scheduled_new_reqs[].num_computed_tokens, scheduled_cached_reqs.{req_ids,num_computed_tokens}, num_scheduled_tokens{}), else
    "counts" = (ntok << 16) | nreq -- recorded on the run (`bounded_staging.plan_key_mode`) so a record never hides which one it used."""
    try:
        nst = sched.num_scheduled_tokens
        geo: list[tuple[int, int]] = []
        for r in getattr(sched, "scheduled_new_reqs", ()) or ():
            geo.append((int(r.num_computed_tokens), int(nst.get(r.req_id, 0))))
        cached = getattr(sched, "scheduled_cached_reqs", None)
        if cached is not None and getattr(cached, "req_ids", None) is not None:
            for i, rid in enumerate(cached.req_ids):
                geo.append((int(cached.num_computed_tokens[i]), int(nst.get(rid, 0))))
        if len(geo) != nreq or sum(g[1] for g in geo) != ntok:
            raise ValueError(f"scheduler output geometry {len(geo)} reqs / {sum(g[1] for g in geo)} tok != ({nreq}, {ntok})")
        geo.sort()
        what = (int(ntok), int(nreq), geo, bytes(targets)) if targets else (int(ntok), int(nreq), geo)
        h = hashlib.blake2b(repr(what).encode(), digest_size=8).digest()
        return int.from_bytes(h, "little") & 0x7FFF_FFFF_FFFF_FFFF, "geometry"
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return (int(ntok) << 16) | (int(nreq) & 0xFFFF), "counts"


# the first two zero-window overwriters and their flush leaves (GATHER_FLUSH_LEAVES, PRE_FLUSH_LEAVES): `flush_points`
# a third zero-window overwriter, INSIDE the fused-MoE block and invisible to every module hook: vLLM's TritonExperts
# carves `intermediate_cache1` (the w13 GEMM output = the `moe/L<k>/up` plane) and `intermediate_cache3` (the w2 GEMM output = `down`) out
# of the SAME workspace (`_resize_cache(workspace2, ...)` twice, offset 0 -> the two views alias byte for byte when 2I == H, OLMoE), so the
# block's own w2 launch overwrites `up` two launches after it was staged.  The MoE source (`moe_source.MoeBlockSource`) stages `up` right
# after the w13 launch, and the next module flush point is the NEXT layer's `input_layernorm` pre-flush, by which time the `up` bytes are the
# same layer's `down` bytes (on the OLMoE B1 record, without the flush: MoeExpertGemm_v1 320/320 VUs != F_V while committed up == committed
# down row for row; act/down/sum/router planes are correct, being overwritten only by the next layer, after the pre-flush).  So the MoE
# source asks the committer to flush right after staging `up` (`flush_after_plane`); one more gather launch per MoE layer per step, nothing
# for dense workloads.  Negative `VERITY_FAULT=late_flush_moe_up` drops that flush.
#: fused-MoE planes whose workspace the block itself overwrites before its next module hook: flush right after staging them
MOE_FLUSH_PLANES = frozenset({"up"})


WINDOW_MB, WINDOW_SLOTS = 256, 8
# rows-sha512 (`rows_hold_bytes`): the most of a step's h (a learning step's, before its length is known) and, apart, of its leaves (a
# step that keeps no levels, before its tree) the worker holds on the device.  Past it a learning step's later windows re-hash from
# host, and the held leaves fold into their windows' top nodes, so a step holds at most 2 x this beyond the ring and one window's
# transients, whatever its length.  A constant, not a share of the memory the engine leaves free.
ROWS_HOLD_BYTES = 1 << 30


def window_mb_default(windowed_worker: bool, retain: str) -> int:
    """VERITY_WINDOW_MB when it is unset: a ring of WINDOW_MB windows on the windowed worker (native worker + gather) unless it retains
    on the device, whose footprint is the per-step blocks; per-step blocks (0) on every other collector."""
    return WINDOW_MB if windowed_worker and retain != "device" else 0


class NativeCollectCommitter(NativeHostCommitter):
    """Arm 3.  Inherits the whole `NativeHostCommitter` path; replaces the serving-thread acquisition (see module docstring).

    `native_hooks=True` (collector v2, `native_collect_v2`): the module.forward wrappers are C++ callables (`HookFwd`) and the FA2
    M1 glue is `Fa2Wrap` -- no Python frame per producer; gather windows (P0/P1/P2 + step end) with 32 blocks per descriptor by default
    (size-adaptive, `auto_blocks`: up to 1024)."""

    def __init__(self, engine: Any, *, wrap: bool = True, profile: bool = False, build_dir: str | None = None, check_sizes: bool = True,
                 gather: bool = False, native_hooks: bool = False, auto_blocks: bool = False, native_worker: bool = False,
                 retain_levels: bool | None = None, commit_scheme: str | None = None,
                 rows_cap_blocks: int = 0, rows_cap_free: int = 2, rows_hold_bytes: int = ROWS_HOLD_BYTES, **kw: Any) -> None:
        kw.setdefault("fast_copy", True)
        kw.setdefault("gpu_tree", True)
        super().__init__(engine, **kw)
        self.wrap = bool(wrap)
        self.profile = bool(profile)
        self.gather = bool(gather)
        self.native_hooks = bool(native_hooks)
        # SplitsFor_v1 when the required manifest names `sampler_splits` (a derived-S Program; set by `taps.attach`): see `topp_splits.step_splits`
        self.prescribe_idle_splits = None
        if self.gather and not self.wrap:
            raise ValueError("gather mode needs wrap mode (flush points are module wrappers)")
        if self.native_hooks and (not self.wrap or self.profile):
            raise ValueError("native hooks need wrap mode and no per-hook profiling")
        self.auto_blocks = bool(auto_blocks) and self.gather
        self.native_worker = bool(native_worker)
        self._hasher = ext(build_dir).StepHasher() if self.native_worker else None
        # windowed staging: VERITY_WINDOW_MB=<W> (0 = off; unset = `window_mb_default`) lays the packed step stream over a ring of
        # VERITY_WINDOW_SLOTS device windows and absorbs each window as soon as the serving thread has filled it (see native_collect.cpp
        # "WINDOWED staging ring")
        win_mb = settings.VERITY_WINDOW_MB
        if win_mb is None:
            win_mb = window_mb_default(self.native_worker and self.gather, self.retain)
        self.win_bytes = int(float(win_mb) * (1 << 20)) // 256 * 256
        slots = settings.VERITY_WINDOW_SLOTS
        self.win_slots_req = WINDOW_SLOTS if slots is None else slots
        # rows-sha512: rows_cap_blocks=<n> (0 = off) hashes a window in launches of at most n blocks while at least rows_cap_free ring
        # slots are free (the serving thread is not waiting for one), leaving the other SMs to serving
        self.rows_cap_blocks = int(rows_cap_blocks)
        self.rows_cap_free = int(rows_cap_free)
        self.rows_hold_bytes = int(rows_hold_bytes)
        self.windowed = self.win_bytes > 0 and self.native_worker and self.gather
        if self.win_bytes > 0 and not self.windowed:
            raise ValueError("windowed staging needs native_collect_v2b (native worker + gather)")
        self.win_tensors: list[Any] = []
        self.win_scratch: Any | None = None
        # `retain_levels=False`: no tree level of a step is kept (on the windowed path each window's subtree stays on the device, no host
        # copy and no host sync per window); the roots are the same, and an opening takes its value and path from a replay
        # (`open(values_from=, paths_from_replay=True)`), which `verify` holds to this run's roots.  Unset, a windowed run that keeps no
        # values (retain=none, whose openings are replayed anyway) keeps no levels; every other run keeps them
        self.retain_levels = bool(retain_levels) if retain_levels is not None else not (self.windowed and self.retain == "none")
        if not self.retain_levels and not self.windowed:
            raise ValueError("retain_levels=False is implemented on the windowed path only (--window-mb > 0)")
        # the commitment scheme, rows-sha512 (`step_rows`: every step_rows.ROW_BYTES (4 KiB) chunk an hm96-sha512 hiding leaf in a
        # frame-v3-sha512 tree, 64-byte roots), the windowed worker's; a row is a chunk of the step's stream (`chunk`): the layout's
        # leaves, the windows' and the tree's.  Each step's salt key is fresh (`row_keys`); a replay that must reproduce a run's trees
        # takes that run's keys (`replay_row_keys`, set before the run and dropped by its finalize)
        if (commit_scheme or step_rows.NAME) != step_rows.NAME:
            raise ValueError(f"unknown commit scheme {commit_scheme!r}: the collector commits {step_rows.NAME}")
        if not self.windowed:
            raise ValueError(f"{step_rows.NAME} is implemented on the windowed worker (native worker + gather + --window-mb > 0)")
        self.replay_row_keys: dict[int, bytes] | None = None
        self._row_tn: Any | None = None
        self._row_consts = (step_rows.row_prefix(8 * step_rows.ROW_BYTES), step_rows.INST.salt_prefix,
                            step_rows.INST.leaf_prefix(step_rows.KEY))
        # BOUNDED staging (VERITY_STAGING_BOUNDED; unset = on whenever staging is windowed): a leaf's frame domain binds the step's
        # row count, which a LEARNING step (plan not yet known) only knows at step end -- unbounded, the windowed
        # path copies EVERY window of every step to pinned host (the learning re-hash source; the hit-plan "hedge" copy) and
        # returns none of it before the run's reset: the 3,545-token prefill of the B8 FP8 workload is 138 GiB of `fa3_hidden_m1` host
        # copies against a 239 GB cgroup.  Bounded:
        #   * a learning step's transient copies are capped by VERITY_LEARN_HOST_BUDGET_MB and given back to the pool right after the
        #     re-hash; over budget, a LEARN-ONLY pass (`learn_only`, the warm-up instrumented run) records the plan and discards the
        #     windows (no copy, no hash; placeholder commitment, never a record), a record run fails CLOSED by name;
        #   * a hit-plan step takes NO hedge copy: a plan mismatch is a named collector error (the plan of a prior identical step
        #     is the prescribed layout; an unexpected shape is a finding, not something to absorb silently);
        #   * retain=host + VERITY_RETAIN_EXCLUDE=<family,...> keeps every chunk EXCEPT the chunk-aligned interior of the excluded
        #     families (`_HostRanges`): those leaves open by replay against the original root, all others from host (C2's
        #     module outputs stay on host).
        bounded = settings.VERITY_STAGING_BOUNDED
        self.bounded = self.windowed if bounded is None else bounded
        self.learn_only = False                                  # set by the harness around the warm-up instrumented run
        self.learn_budget_bytes = int(settings.VERITY_LEARN_HOST_BUDGET_MB * (1 << 20))
        self.retain_exclude: set[str] = retain_exclude_families(settings.VERITY_RETAIN_EXCLUDE)
        self.retain_chunks: set[tuple[int, int]] = set()          # bounded retain=sample by CHUNK: (step, leaf index) -- a replay keeps only these
        if self.bounded and not self.windowed:
            raise ValueError("bounded staging (VERITY_STAGING_BOUNDED=1) is implemented on the windowed path only (--window-mb > 0)")
        if self.retain_exclude and not (self.bounded and self.retain == "host"):
            raise ValueError("VERITY_RETAIN_EXCLUDE needs VERITY_STAGING_BOUNDED=1 and retain=host (the excluded families open by replay)")
        self.learned_only_steps: list[int] = []                  # steps whose windows a learn-only pass discarded (plan recorded, no root)
        self.learned_only_bytes = 0
        self.transient_peak_bytes = 0                            # largest learning-step transient host copy (given back after the re-hash)
        self.excluded_bytes = 0                                  # retain=host + exclude: bytes NOT copied to host over the run
        self.plan_mismatches = 0
        # the model step runs on a HIGH-priority CUDA stream (VERITY_HI_PRIO, default on; 0 keeps the caller's) so the worker's SHA kernels
        # (side stream, default priority) yield SM slots to the serving kernels and the gather; fenced both ways with wait_stream at the
        # step boundary (the FA2 tap, the gather flushes and the step event all follow the current stream, so nothing else changes)
        self.hi_prio = bool(native_hooks) and settings.VERITY_HI_PRIO
        self._hi = torch.cuda.Stream(priority=-1) if self.hi_prio else None
        # the serving thread's step end as one native call (`end_step_full`; no step event on the windowed path, whose worker waits on the
        # windows' own), and on the windowed path a snapshot of the step's row order that the worker expands into `step_rows` unless a
        # source reads them at the step's end (`step_rows_at_step_end`); both leave the bytes and roots as they were
        self.native_end = True
        self.rows_off_thread = True
        self._rows_at_step_end = False
        if self.native_hooks:
            self.name = ("native_collect_v2" + ("" if self.gather else "_memcpy") + ("_auto" if self.auto_blocks else "") + ("_nw" if self.native_worker else "")
                         + ("_hp" if self.hi_prio else ""))
            self.hidden_source_cls = NativeHiddenM1Source
            self.hidden_source_cls_fa3 = NativeHiddenM1SourceFA3      # chosen by commit_delta when the tap is an FA3 (v9) build
        else:
            self.name = "native_collect" + ("_gather" if self.gather else "") + ("_wrap" if self.wrap else "_hook") + ("_prof" if self.profile else "")
        self._nc = ext(build_dir).Collector()
        self._nc.check_sizes = bool(check_sizes)
        self._nc.use_gather = self.gather
        if self.gather:
            self._nc.enable_gather(65536)
            # thread blocks per staged descriptor in the gather kernel (default 4; 32 with native hooks, the labelled-extra Llama geometry).
            # Size-adaptive, it is only a cap, 1024: a 256 MB prefill window holds a handful of descriptors, and at 32 blocks each the copy
            # fills a fraction of the SMs (B32 compiled prefill gather 12.2 ms/step at 32, 8.3 at 1024)
            blocks = settings.NATIVE_GATHER_BLOCKS
            self._nc.blocks_per_desc = blocks if blocks is not None else 1024 if self.auto_blocks else 32 if self.native_hooks else 4
            # size-adaptive blocks per window (cap = blocks_per_desc; ceil(max staged bytes / auto_bytes_per_block) otherwise)
            self._nc.auto_blocks = self.auto_blocks
            self._nc.auto_bytes_per_block = settings.NATIVE_GATHER_AUTO_BYTES
        self._names: list[str] = []
        self._name_ids: dict[str, int] = {}
        self._metas_cache = PlanCache()                          # (plan key, version) -> PlanMetas, for this run's and the previous run's plans
        self._wrapped: list[Any] = []
        self._plan_key = 0
        self._py_committed = 0
        self._prof_s = 0.0
        self.consumer_delay_s = settings.VERITY_CONSUMER_DELAY_MS / 1e3
        self.fault = settings.raw("VERITY_FAULT") or ""
        if self.fault.startswith("skip_copy:"):
            self._nc.fault_skip_copy_at = int(self.fault.split(":", 1)[1])
        if self.fault or self.consumer_delay_s:
            self.name = f"{self.name}-STRESS[{self.fault or ''}{'+delay%gms' % (self.consumer_delay_s * 1e3) if self.consumer_delay_s else ''}]"
        self.name = f"{self.name}+{step_rows.NAME}"
        self.extra_stats: dict[str, Any] = {}

    def native_ext(self) -> Any:
        """The collector's extension (`ext`), which the native hidden taps (`native_hidden`) build their C++ wrappers from."""
        return ext()

    # ------------------------------------------------------------------ names

    def _intern(self, name: str) -> int:
        nid = self._name_ids.get(name)
        if nid is None:
            nid = len(self._names)
            self._names.append(name)
            self._name_ids[name] = nid
        return nid

    # ------------------------------------------------------------------ install / uninstall (hook sites)

    def install(self) -> None:
        if self.installed:
            return
        self._check_skip()
        self._start_worker()
        self.gc_pauses.install()
        self._rows_at_step_end = any(getattr(src, "step_rows_at_step_end", False) for src in self.sources)
        for src in self.sources:
            src.install()
        if getattr(self, "external_acquisition", False):
            # compiled execution: the module boundaries are acquired by record ops inside the compiled graphs (compiled_source.CompiledGraphSource,
            # installed before vLLM compiled the model); wrapping the forwards here would never be traced and never run under graph replay
            pass
        elif self.wrap:
            for name, m in self._modules:
                pre = self._acquires_input(name, m)                     # by class (the embedding's ids) or by the manifest (a promoted member's consumer)
                if "forward" in m.__dict__:
                    raise RuntimeError(f"{name}: forward already wrapped")
                self._hooks.patch(m, "forward", self._make_native_wrapper(m.forward, name, pre) if (self.native_hooks and not self.fault)
                                  else self._make_wrapper(m.forward, name, pre))
                self._wrapped.append(m)
            for name, m in self._input_only_modules():                  # a consumer of no acquire class: input acquired, output not
                if "forward" in m.__dict__:
                    raise RuntimeError(f"{name}: forward already wrapped")
                self._hooks.patch(m, "forward", self._make_input_wrapper(m.forward, name))
                self._wrapped.append(m)
        else:
            for name, m in self._modules:
                self._handles.append(m.register_forward_hook(self._make_hook(name)))
                if self._acquires_input(name, m):
                    self._handles.append(m.register_forward_pre_hook(self._make_pre_hook(name)))
            for name, m in self._input_only_modules():
                self._handles.append(m.register_forward_pre_hook(self._make_pre_hook(name)))
        runner = self.runner
        for nm in ("execute_model", "sample"):
            if hasattr(runner, nm):
                self._orig[nm] = getattr(runner, nm)
        orig_exec = self._orig["execute_model"]
        me = self

        def execute_model(scheduler_output, *a, **kw):
            dummy = bool(kw.get("dummy_run", False)) or (len(a) >= 2 and bool(a[1]))
            ntok = getattr(scheduler_output, "total_num_scheduled_tokens", 0)
            if dummy or ntok == 0:
                return orig_exec(scheduler_output, *a, **kw)
            me._end_step()
            nreq = len(getattr(scheduler_output, "num_scheduled_tokens", ()) or ())
            # step-shape class -> plan key.  (ntok, nreq) alone is NOT the layout: the FA3 hidden M1 export plane is
            # sized by every request's KV length, so two decode steps with the same (ntok, nreq) differ in bytes (B8 mixed 1024/128 steps
            # 35 and 80: 78,892,288 vs 87,297,280; see `plan_key_of`).  The key binds the per-request
            # geometry (num_computed_tokens, num_scheduled_tokens) as a multiset; on the unbounded windowed path a miss still self-heals.
            # It binds too the names the step-keyed sources commit at the step about to begin (`step_targets`), which follow the window step.
            me._plan_key, me._plan_key_mode = plan_key_of(scheduler_output, int(ntok), int(nreq), me.step_targets(me._step_index))
            me._plan_shape = (int(ntok), int(nreq))
            if me.bounded and me._plan_key_mode != "geometry":
                # fail closed at the FIRST step, not mid-run: without the per-request geometry a hit-plan step can meet another layout
                raise RuntimeError("bounded staging needs the geometry plan key (per-request num_computed_tokens / num_scheduled_tokens); "
                                   f"this scheduler output ({type(scheduler_output).__name__}) does not expose them")
            me._cur_sched = scheduler_output                         # the step's scheduled tokens per request (row map, read at sample)
            me._step_min_bytes = me.step_min_bytes(int(ntok), int(nreq))   # pick a block big enough for THIS step (prefill vs decode)
            me._begin_step()
            if me.hi_prio:
                cur = torch.cuda.current_stream()
                hi = me._hi
                hi.wait_stream(cur)
                with torch.cuda.stream(hi):
                    res = orig_exec(scheduler_output, *a, **kw)
                cur.wait_stream(hi)
                return res
            return orig_exec(scheduler_output, *a, **kw)

        self._hooks.patch(runner, "execute_model", execute_model)
        if hasattr(runner, "gather_batch_req_state"):                          # new `vllm.v1.worker.gpu` runner: capture the step's sorted batch rows
            orig_gather = self._orig["gather_batch_req_state"] = runner.gather_batch_req_state

            def gather_batch_req_state(*a, **kw):
                res = orig_gather(*a, **kw)
                bs = res[0] if isinstance(res, tuple) and res else res
                if bs is not None and getattr(bs, "req_ids", None) is not None:
                    me._cur_batch = bs
                return res

            self._hooks.patch(runner, "gather_batch_req_state", gather_batch_req_state)
        if "sample" in self._orig:
            orig_sample = self._orig["sample"]
            # the top-p split count S of this step's sampling (manifest family `sampler_splits`, identity
            # `runner.sampler/splits`, one i32 per sequence) is a HOST constant of vLLM `_apply_topp_split` -- never on the device -- so it is
            # read from the split kernels' launches (constexpr kwarg `S`) by launch proxies on the module's kernel globals, not recomputed.
            self._split_launches: list[tuple] = []
            self._split_orig = install_split_kernel_proxies(self._split_launches)

            def sample(*a, **kw):
                del me._split_launches[:]
                res = orig_sample(*a, **kw)
                job = me._cur
                if job is not None:
                    try:
                        so = res[0] if isinstance(res, tuple) else res
                        st = getattr(so, "sampled_token_ids", None)
                        ib = a[1] if len(a) > 1 else kw.get("input_batch")
                        draw_srcs = [s for s in me.sources if hasattr(s, "on_sample")]      # a source of the draw's inputs (served_record)
                        owned = {n for s in draw_srcs for n in getattr(s, "own_names", ())}
                        if isinstance(st, torch.Tensor) and st.device.type == "cuda":
                            me._acquire("runner.sampler/sampled_token_ids", st, job)
                            for s in draw_srcs:
                                s.on_sample(ib, int(st.shape[0]), job)
                            # launched S, or (a derived-S Program's step on which vLLM skipped top-p) the prescribed S -- once per sampled row
                            S, why, counter = step_splits(list(me._split_launches), int(st.shape[0]), me.prescribe_idle_splits, lambda: batch_top_p(runner, a, kw),
                                                          lambda: torch.cuda.get_device_properties(st.device).multi_processor_count)
                            if why:
                                me.errors.append(f"sampler splits: {why}"[:200])
                            elif S is not None:
                                me._acquire(SPLITS_IDENTITY, torch.full((int(st.shape[0]),), S, dtype=torch.int32, device=st.device), job)
                                setattr(me, counter, getattr(me, counter, 0) + 1)
                                me.splits_seen = getattr(me, "splits_seen", {})
                                me.splits_seen[S] = me.splits_seen.get(S, 0) + 1
                        # the per-sequence SEED the sampler consumed this step (manifest family `sampler_seed`, identity
                        # `runner.sampler/seed`, one i64 per sequence): `sampling_states.seeds.gpu[input_batch.idx_mapping]` -- the very buffer
                        # `_gumbel_sample_kernel` reads (`seeds_ptr + req_state_idx`), gathered in the step's row order.  Bound only when the
                        # batch carries explicit seeds (a stochastic Program's registered runtime input); a greedy / unseeded step binds nothing.
                        ss = getattr(getattr(runner, "sampler", None), "sampling_states", None)
                        if "runner.sampler/seed" not in owned and ib is not None and ss is not None and hasattr(ss, "seeds") and hasattr(ib, "idx_mapping"):
                            idx_np = getattr(ib, "idx_mapping_np", None)
                            explicit = bool(ss.any_explicit_seed(idx_np)) if (idx_np is not None and hasattr(ss, "any_explicit_seed")) else False
                            if explicit:
                                seeds = ss.seeds.gpu[ib.idx_mapping.to(torch.long)]
                                me._acquire("runner.sampler/seed", seeds, job)
                                me.seed_bound_steps = getattr(me, "seed_bound_steps", 0) + 1
                    except Exception as e:  # noqa: BLE001
                        me.errors.append(f"sampler acquire: {e!r}"[:200])
                    me._end_step()
                return res

            self._hooks.patch(runner, "sample", sample)
        self.installed = True

    def uninstall(self) -> None:
        if not self.installed:
            return
        super().uninstall()
        self._wrapped = []
        uninstall_split_kernel_proxies(getattr(self, "_split_orig", None) or {})
        self._split_orig = {}

    def _flush_leaf(self, name: str) -> bool:
        """Legal collection points of the gather design (P0/P1/P2).  Negative test `VERITY_FAULT=late_flush`: the qkv_proj flush
        is dropped, so the staged qkv output is read at the o_proj hook -- AFTER RoPE rotated q/k in place (zero-hook window) -> the run
        root MUST differ from arm 2 (late-read regression: qkv_proj out is overwritten at hook+1)."""
        leaf = name.rsplit(".", 1)[-1]
        if self.fault == "late_flush" and leaf == "qkv_proj":
            return False
        if self.fault == "late_flush_qk" and leaf in ("q_norm", "k_norm"):       # negative: the QK-norm outputs meet RoPE before the o_proj flush
            return False
        return leaf in GATHER_FLUSH_LEAVES

    def _pre_flush_leaf(self, name: str) -> bool:
        """Flush BEFORE this module runs: the fused add-RMSNorm (`input_layernorm`, final `norm`) overwrites its `x`
        (the previous block's output) and `residual` (the previous `post_attention_layernorm/1`) in place, and a fused-MoE block has no
        down_proj hook to have flushed them.  Negative `VERITY_FAULT=late_flush_norm`: the pre-flush is dropped -> on an MoE model the
        committed `post_attention_layernorm/1` is the NEXT layer's residual."""
        leaf = name.rsplit(".", 1)[-1]
        if self.fault == "late_flush_norm":
            return False
        return leaf in PRE_FLUSH_LEAVES

    def _flush_plane(self, what: str) -> bool:
        """Does the staged fused-MoE plane `what` leave the device NOW (before the block's next launch)?  `up` lives in
        the workspace the block's own w2 launch overwrites (cache1/cache3 alias); the other planes are overwritten only by the next layer,
        after its `input_layernorm` pre-flush.  Negative `VERITY_FAULT=late_flush_moe_up`: the flush is dropped -> the committed `up` is
        the same layer's `down`."""
        if self.fault == "late_flush_moe_up" and what == "up":
            return False
        return what in MOE_FLUSH_PLANES

    def flush_after_plane(self, what: str) -> bool:
        """Acquisition sources call this right after `_acquire` of a fused-MoE plane; returns whether a gather launch was issued.  Non-gather
        collectors copy at the hook (nothing staged) -> False."""
        if not (self.gather and self._flush_plane(what)):
            return False
        self._nc.flush()
        return True

    def _with_pre_flush(self, fwd, name: str):
        """Wrap a hook body so the collector flushes what is staged before the module's forward runs (`_pre_flush_leaf`); a flush with
        nothing staged returns early in C++, so the pre-flush costs one Python frame on 1 + n_layers modules per step and no launch."""
        if not (self.gather and self._pre_flush_leaf(name)):
            return fwd
        nc = self._nc

        def fwd_pre_flush(*a, **kw):
            nc.flush()                       # the in-place consumer is about to run: everything staged by the previous block leaves now
            return fwd(*a, **kw)
        return fwd_pre_flush

    def _make_native_wrapper(self, orig, name: str, pre: bool):
        """Collector v2: the hook body is `HookFwd` (C++): one pybind dispatch per producer call, CPython type checks, name ids bound here."""
        nid_out = self._intern(name)
        nids_k = tuple(self._intern(f"{name}/{k}") for k in range(8))
        nid_in = self._intern(f"{name}/in") if pre else -1
        nid_st = self._intern(f"{name}/sampled_token_ids")
        flush = bool(self.gather and self._flush_leaf(name))
        return self._with_pre_flush(ext().HookFwd(self._nc, orig, nid_out, nids_k, nid_in, nid_st, flush), name)

    def _make_wrapper(self, orig, name: str, pre: bool):
        """The candidate hook body.  `orig` is the module's bound forward; after it returns, every tensor output goes to the native
        collector with a name id bound at install time (no string keys, no torch ops, no allocation on this path)."""
        me = self
        collect = self._nc.stage if self.gather else self._nc.collect
        flush = self._nc.flush if (self.gather and self._flush_leaf(name)) else None
        Tensor = torch.Tensor
        nid_out = self._intern(name)
        nids_k = tuple(self._intern(f"{name}/{k}") for k in range(8))
        nid_in = self._intern(f"{name}/in") if pre else None
        nid_st = self._intern(f"{name}/sampled_token_ids")
        profile = self.profile

        drop1_dup0 = self.fault == "drop1_dup0" and name.endswith(".input_layernorm") and ".layers.0." not in name

        def collect_out(out):
            if isinstance(out, Tensor):
                collect(nid_out, out)
            elif isinstance(out, (tuple, list)):
                if drop1_dup0 and len(out) >= 2 and isinstance(out[0], Tensor) and isinstance(out[1], Tensor):
                    collect(nids_k[0], out[0])       # the binding negative: member /1 dropped, member /0 bound twice -> same count
                    collect(nids_k[0], out[0])
                    return
                for k, o in enumerate(out):
                    if isinstance(o, Tensor):
                        collect(nids_k[k], o)
            else:
                st = getattr(out, "sampled_token_ids", None)
                if isinstance(st, Tensor):
                    collect(nid_st, st)
            if flush is not None:
                flush()                      # legal collection point (P0/P1/P2): one gather launch for everything staged since the last flush

        if pre:
            def fwd(*a, **kw):
                if me._cur is not None:
                    t0 = time.perf_counter() if profile else 0.0
                    try:
                        if a and isinstance(a[0], Tensor):
                            collect(nid_in, a[0])
                    except Exception as e:  # noqa: BLE001
                        me.errors.append(f"{name}/in: {e!r}"[:300])
                    if profile:
                        me._prof_s += time.perf_counter() - t0
                out = orig(*a, **kw)
                if me._cur is not None:
                    t0 = time.perf_counter() if profile else 0.0
                    try:
                        collect_out(out)
                    except Exception as e:  # noqa: BLE001
                        me.errors.append(f"{name}: {e!r}"[:300])
                    if profile:
                        me._prof_s += time.perf_counter() - t0
                return out
        elif profile:
            def fwd(*a, **kw):
                out = orig(*a, **kw)
                if me._cur is not None:
                    t0 = time.perf_counter()
                    try:
                        collect_out(out)
                    except Exception as e:  # noqa: BLE001
                        me.errors.append(f"{name}: {e!r}"[:300])
                    me._prof_s += time.perf_counter() - t0
                return out
        else:
            def fwd(*a, **kw):
                out = orig(*a, **kw)
                if me._cur is not None:
                    try:
                        collect_out(out)
                    except Exception as e:  # noqa: BLE001
                        me.errors.append(f"{name}: {e!r}"[:300])
                return out

        return self._with_pre_flush(fwd, name)

    def _make_input_wrapper(self, orig, name: str):
        """The INPUT-ONLY hook body of a consumer module the required manifest names for a promoted member and the committer
        did not select (no acquire class): its first positional tensor argument goes to the collector as `<name>/in` (`_acquire`: reference-integrity
        expected + committed accounting), its output is no identity and is not acquired.  One Python frame per call on the few such
        modules; a consumer of an acquire class takes the native `HookFwd` path with `nid_in` instead (`install`)."""
        me = self
        Tensor = torch.Tensor
        in_name = f"{name}/in"
        self._intern(in_name)

        def fwd(*a, **kw):
            job = me._cur
            if job is not None and a and isinstance(a[0], Tensor):
                try:
                    me._acquire(in_name, a[0], job)
                except Exception as e:  # noqa: BLE001
                    me.errors.append(f"{in_name}: {e!r}"[:300])
            return orig(*a, **kw)

        return fwd

    # ------------------------------------------------------------------ acquisition entry used by the Python hooks / the FA2 M1 source / the sampler

    def _acquire(self, name: str, t: Any, job: _StepJob) -> None:
        nid = self._name_ids.get(name)
        if nid is None:
            nid = self._intern(name)
        self._nc.note_expected(1)                                  # attempted acquisition (committed is counted by put)
        fn = self._nc.stage if self.gather else self._nc.collect
        try:
            if self.profile:
                t0 = time.perf_counter()
                fn(nid, t)
                self._prof_s += time.perf_counter() - t0
            else:
                fn(nid, t)
            self._py_committed += 1
        except RuntimeError:
            if not getattr(self._nc, "failure", ""):   # the run failed by name (abandon_step): refused, not committed, never raised into the forward
                raise

    def slot_plan(self, names: list[str], slots: list[Any], kinds: list[int]) -> int:
        """Register a replayed graph's slots once (kind 0: the whole slot, 1: its first n rows, 2: its first pad rows); the handle's
        `stage_slot_plan(handle, n, pad)` acquires them in this order in one native call, the same bytes and plan entries as `_acquire`
        of each view.  A handle holds its slots until `clear_slot_plans`."""
        return int(self._nc.add_slot_plan([self._intern(nm) for nm in names], list(slots), [int(k) for k in kinds]))

    def stage_slot_plan(self, handle: int, n: int, pad: int) -> int:
        try:
            return int(self._nc.stage_slot_plan(int(handle), int(n), int(pad)))
        except RuntimeError:
            if not getattr(self._nc, "failure", ""):
                raise
            return 0                     # refused after the run failed by name, as `_acquire`

    def clear_slot_plans(self) -> None:
        self._nc.clear_slot_plans()

    # ------------------------------------------------------------------ step boundaries (serving thread)

    def _record_step_rows(self, job: _StepJob, defer: bool = False) -> None:
        """BIND-01 (B>1 binding): the engine's row order of this step -- `InputBatch.req_ids` (the persistent batch after the step's
        `_update_states`) x `scheduler_output.num_scheduled_tokens` -> [(request_id, token_row_lo, token_row_hi)] in the flattened token
        order every per-token member uses; request-indexed members (logits, sampled ids) use the list index.  -> `self.step_rows[step]`.
        `defer`: only copy what the expansion reads (the batch's lists are reused by the next step) into `job.rows_snapshot`."""
        so = getattr(self, "_cur_sched", None)
        # Row-order source, in preference order: (a) the BatchReqState the new `vllm.v1.worker.gpu` runner builds per step
        # (`gather_batch_req_state`, captured by our wrapper -- rows sorted by the runner, NOT scheduler order); (b) the runner's
        # `execute_model_state.input_batch`; (c) the persistent `InputBatch` of the older `gpu_model_runner`.
        ib = getattr(self, "_cur_batch", None)
        src = "gather_batch_req_state"
        if ib is None:
            ems = getattr(self.runner, "execute_model_state", None)
            ib = getattr(ems, "input_batch", None)
            src = "execute_model_state.input_batch"
        if ib is None:
            ib = getattr(self.runner, "input_batch", None)
            src = "runner.input_batch"
        if so is None or ib is None:
            self.step_rows_why = f"so={type(so).__name__} ib=None runner={type(self.runner).__module__}.{type(self.runner).__name__}"
            return
        self._cur_batch = None
        nst = getattr(so, "num_scheduled_tokens", None) or {}
        req_ids = list(getattr(ib, "req_ids", []) or [])
        n = int(getattr(ib, "num_reqs", len(req_ids)) or len(req_ids))
        per_row = getattr(ib, "num_scheduled_tokens", None)
        self.step_rows_src = src
        if not hasattr(self, "step_rows"):
            self.step_rows = {}
        if defer:
            if per_row is not None and not isinstance(per_row, dict):
                try:
                    per_row = per_row.tolist() if hasattr(per_row, "tolist") else list(per_row)
                except TypeError:
                    per_row = None
            job.rows_snapshot = (req_ids, n, per_row, nst)                  # type: ignore[attr-defined]
            return
        self._expand_step_rows(int(job.step), req_ids, n, per_row, nst)

    def _expand_step_rows(self, step: int, req_ids: list, n: int, per_row: Any, nst: dict) -> None:
        try:
            per_row = [int(x) for x in per_row] if per_row is not None and not isinstance(per_row, dict) else None
        except TypeError:
            per_row = None
        rows, lo = [], 0
        for i, rid in enumerate(req_ids[:n]):
            if rid is None:
                continue
            k = int(per_row[i]) if per_row is not None and i < len(per_row) else int(nst.get(rid, 0))
            if k <= 0:
                continue
            rows.append((str(rid), lo, lo + k))
            lo += k
        self.step_rows[step] = rows

    def step_targets(self, step: int) -> bytes:
        """The digest of the sorted (name, count)s the sources that commit by engine step (`targets_at`) commit at `step`, b"" for none:
        part of its plan key.  Kept per step while the sources are the same: H1's int8 source names every linear at every step."""
        srcs = tuple(src for src in self.sources if hasattr(src, "targets_at"))
        if getattr(self, "_targets_of", None) != srcs:
            self._targets_of, self._targets_memo = srcs, {}
        sig = self._targets_memo.get(int(step))
        if sig is None:
            names = sorted(kv for src in srcs for kv in src.targets_at(step).items())
            sig = self._targets_memo[int(step)] = hashlib.blake2b(repr(names).encode(), digest_size=16).digest() if names else b""
        return sig

    def _begin_step(self) -> None:
        job = _StepJob(self._step_index, [], None, com=self._window_view(), seq=self._next_seq())
        self._cur = job
        self._step_index += 1
        for src in self.sources:
            src.on_step_begin()
        job.arena = None
        job.cursor = 0
        job.arena_ptr = 0
        job.windowed = True                                                     # type: ignore[attr-defined]
        job.done_evt = threading.Event()                                        # type: ignore[attr-defined]
        job.plan_used = int(self._nc.plan_used(self._plan_key))                 # type: ignore[attr-defined]  (-1: learning step, size unknown)
        # the metas of the plan this hit step is predicted to follow (None on a learning step): the worker needs them
        # BEFORE step end to know which chunks of a window belong to an excluded family (retain=host + exclude)
        job.plan_hint = (self._plan_key, int(self._nc.plan_version(self._plan_key))) if job.plan_used >= 0 else None  # type: ignore[attr-defined]
        job.hint_entries = self._nc.export_plan(self._plan_key) if self.retain_exclude and job.plan_hint and not self._plan_held(job.plan_hint) else None  # type: ignore[attr-defined]
        job.plan_shape = getattr(self, "_plan_shape", (0, 0))                    # type: ignore[attr-defined]  (ntok, nreq) for messages/records
        self._prof_s = 0.0
        self._nc.begin_step(self._plan_key, 0, 0, True, job.seq)                # its windows keyed by the job's seq: unique across windows
        self._q.put(job)                                                        # the worker absorbs windows WHILE the step runs
        if self.footprint is not None:
            _shape = getattr(self, "_plan_shape", (0, 0))
            self.footprint.event("begin_step", step=job.step, ntok=_shape[0], nreq=_shape[1], plan_key=int(self._plan_key), plan_key_mode=getattr(self, "_plan_key_mode", "counts"),
                                 step_min_bytes=int(getattr(self, "_step_min_bytes", 0) or 0),
                                 arena_block_bytes=int(job.arena.numel()) if job.arena is not None else 0, arena_total_bytes=self.arena.bytes_total, arena_free_blocks=[int(b.numel()) for b in self.arena.free],
                                 retained_device_total=self._retained_device_bytes())

    def _end_step(self) -> None:
        job = self._cur
        if job is None:
            return
        self._cur = None
        try:
            # the step's request row order (InputBatch after this step's _update_states; before the next)
            self._record_step_rows(job, defer=self.rows_off_thread and not self._rows_at_step_end)
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"step rows: {e!r}"[:200])
        t0 = time.perf_counter()
        for src in self.sources:
            fn = getattr(src, "on_step_end", None)
            if fn is not None:
                try:
                    fn(job)                  # compiled execution: drain the graph-written staging ring into this step (before the final flush)
                except Exception as e:  # noqa: BLE001
                    self.errors.append(f"{getattr(src, 'short_name', type(src).__name__)} on_step_end: {e!r}"[:300])
        if self.native_end:
            # P3 (step end) flush of the final norm outputs, logits and sampled ids, end_step, the dead-watchdog's progress, the hook errors
            used, n, changed, version, nbytes, issue_s, waits, wait_s, errs = self._nc.end_step_full(self.gather, self.native_hooks)
            if errs:
                self.errors.extend(str(e)[:300] for e in errs)
        else:
            if self.gather:
                self._nc.flush()                 # P3 (step end): final norm outputs, logits, sampled ids -- before the step event is recorded
            used, n, changed, version = self._nc.end_step()
            if getattr(self._nc, "win_dead", False):
                self._nc.progress()                 # dead-watchdog liveness: the serving thread still ends steps (fail-closed run in flight)
            if self.native_hooks:
                errs = self._nc.drain_errors()
                if errs:
                    self.errors.extend(str(e)[:300] for e in errs)
            st = self._nc.step_stats()
            nbytes, issue_s, waits, wait_s = st["bytes"], st["issue_s"], st.get("window_waits"), st.get("window_wait_s")
        job.cursor = int(used)
        key = self._plan_key
        # a learning step, or a hit on a plan the worker does not hold (not yet absorbed, or its step failed: plan_cache)
        entries = self._nc.export_plan(key) if changed or not self._plan_held((key, int(version))) else None
        job.plan = (key, int(version), entries)                         # type: ignore[attr-defined]
        job.acq_bytes = int(nbytes)
        self.staged_bytes += job.acq_bytes
        if not self.native_end:
            job.event = torch.cuda.Event()
            job.event.record(torch.cuda.current_stream())
        job.t_step_end = time.perf_counter()
        job.backlog = self._q.qsize()
        self.spans.add("movement", float(issue_s), step=job.step, nbytes=job.acq_bytes)   # native issue time (memcpy/memset calls) -- measured
        if waits:
            self.spans.add("wait", float(wait_s), step=job.step)   # blocked for a free window slot (the ring was full)
        if self.profile:
            self.spans.add("acquisition", self._prof_s - float(issue_s), step=job.step)  # Python wrapper + pybind overhead above the issue
        self.spans.add("acquisition", time.perf_counter() - t0, step=job.step)
        if self.footprint is not None:
            # the host side of the curve beside the device side: the pinned pool (total / in use) and this process's RSS
            # at every step end -- a component that accumulates the whole prefill shows up here as a pool that never stops growing
            _pool = getattr(self, "pool", None)
            _rss = None
            try:
                with open("/proc/self/statm") as _f:
                    _rss = int(_f.read().split()[1]) * 4096
            except OSError:
                pass
            self.footprint.event("end_step", step=job.step, used=int(job.cursor), staged_bytes=job.acq_bytes, backlog=job.backlog, tensors=int(n),
                                 pinned_pool_bytes=(int(_pool.bytes_total) if _pool is not None else None),
                                 pinned_pool_in_use=(int(_pool.bytes_in_use) if _pool is not None else None), rss_bytes=_rss)
        job.done_evt.set()                                                      # type: ignore[attr-defined]  (queued at step begin)

    # ------------------------------------------------------------------ worker side: metas from the plan (cached per plan version)

    def _plan_held(self, kv: tuple) -> bool:
        """Serving thread: whether the worker still holds plan ``kv``'s metas when it absorbs the step now ending. Each detached window
        it has yet to end retires a generation of its cache there (`_window_ended`) before it reaches this step: with one to go, only
        the current generation's plans survive; with more, none is counted on."""
        behind = self._windows_detached - self._windows_ended
        if behind <= 0:
            return kv in self._metas_cache
        return behind == 1 and kv in self._metas_cache.now

    def _metas_for(self, job: _StepJob, plan: tuple | None = None) -> PlanMetas:
        key, version, entries = plan or job.plan  # type: ignore[attr-defined]
        metas = self._metas_cache.get((key, version))
        if metas is None:
            if entries is None:
                raise RuntimeError(f"plan ({key}, {version}) not seen by the worker before a step that uses it")
            self._metas_cache[(key, version)] = metas = PlanMetas.of_entries(entries, self._names, self.chunk)
        return metas

    def layout_digest(self, metas: list[TensorMeta]) -> bytes:  # type: ignore[override]
        """The base class's function, computed once per plan: nothing changes a field it reads once the plan's record is built."""
        if not isinstance(metas, PlanMetas):
            return NativeHostCommitter.layout_digest(metas)
        if metas.digest is None:
            metas.digest = NativeHostCommitter.layout_digest(metas.scan())
        return metas.digest

    def _commit_step(self, job: _StepJob) -> None:
        if self.consumer_delay_s:
            time.sleep(self.consumer_delay_s)                      # STRESS: delayed commitment consumer
        if getattr(job, "windowed", False):                # queued at step BEGIN: plan / metas exist only after the step ended (inside)
            try:
                if self._nc.failure:                       # an earlier step was abandoned: this one fails by the run's name, uncommitted
                    raise RuntimeError(f"windowed staging: the collector failed earlier in this run ({self._nc.failure[:200]}) -- step {job.step} not committed")
                return self._commit_step_windowed(job)
            except Exception as e:  # noqa: BLE001
                self._nc.abandon_step(job.seq, f"worker failed on windowed step {job.step}: {e!r}"[:400], self._side.cuda_stream if self._side is not None else 0)
                raise           # the run failed by its first such name, and the step's slots went back after the side stream's reads of them
            finally:
                if getattr(self._nc, "win_dead", False):
                    self._nc.progress()                    # dead-watchdog liveness: the worker finished (or failed by name) a step
        raise RuntimeError(f"{step_rows.NAME}: step {job.step} was not staged through the window ring (enable_windows before the run)")

    # ------------------------------------------------------------------ windowed absorb / release
    def enable_windows(self, rec: dict[str, Any] | None = None) -> dict[str, Any]:
        """Allocate the device window ring (K slots x W) + one scratch window and switch the collector to windowed staging.  K = the requested
        slot count (>= 3): the ring is INDEPENDENT of the largest tensor -- `push_desc` acquires slots per window segment and, when the ring
        is exhausted, flushes the segments pushed so far and waits for the worker to hand a slot back, so a tensor larger than the whole
        ring (FA2 M1 of a long prefill) streams through it.  (Sizing K from the largest tensor, max(requested, ceil(bound / W) + 2), would make
        B=32 a 52-slot / 14 GB ring and let the engine run 32 decode steps ahead of the worker.)  The largest-tensor bound (prefill hidden bound / layers) is still reported for the record."""
        W = self.win_bytes
        C = self.chunk
        if W % C or (W // C) & (W // C - 1):
            raise ValueError(f"--window-mb: the window must hold a power of two of {C}-byte chunks (per-window subtrees == whole-step tree)")
        if self.retain == "device":
            raise ValueError("windowed staging retains on host or not at all (--retain host|none|sample); device retention is the footprint")
        self.win_log2_chunks = (W // C).bit_length() - 1
        largest = 0
        if rec:
            layers = int((rec.get("model") or {}).get("layers") or 1)
            largest = int(rec.get("prefill_hidden_bytes") or 0) // max(1, layers)
            largest = max(largest, int(rec.get("decode_step_bytes") or 0))
        K = max(self.win_slots_req, 3)
        free, total = torch.cuda.mem_get_info()
        need = (K + 1) * W
        if need > free - (512 << 20):
            raise RuntimeError(f"window ring does not fit: {K} x {W >> 20} MiB + scratch = {need >> 20} MiB but only {free >> 20} MiB free next to the engine; "
                               f"lower --window-slots / --window-mb or gpu_memory_utilization")
        self.win_tensors = [torch.empty(W, dtype=torch.uint8, device="cuda") for _ in range(K)]
        self.win_scratch = torch.empty(W, dtype=torch.uint8, device="cuda")
        self._nc.enable_windows([int(t.data_ptr()) for t in self.win_tensors], W)
        # BOUNDED staging with BACKPRESSURE, never a drop: when the ring is exhausted the serving thread blocks until the worker
        # hands a slot back; only after `win_wait_timeout_s` (default 600 s) does the acquisition FAIL -- named ("windowed staging: waited N s for a
        # free window slot ... the worker is not releasing windows"), recorded as a collector error, so the identity check turns the run into COMMIT FAIL
        # (never a warning).  VERITY_WIN_WAIT_TIMEOUT_S shortens it for the forced-stall negative (test_commit_fail_closed::test_e2e_windowed_stall_forces_commit_fail).
        if (wto := settings.VERITY_WIN_WAIT_TIMEOUT_S) is not None:
            self._nc.win_wait_timeout_s = wto
        # fail-FAST after the fail-closed name: the timeout above throws from the serving thread, but that throw needs the GIL back and a
        # Python frame that records the failure, and a process can instead sit for minutes at 0 % GPU with every thread in futex_wait.
        # So the C++ collector arms a GIL-free watchdog thread when it declares itself dead: unless the Python side
        # side keeps making progress (a step ends / the worker finishes a step -> progress()) or acknowledges (finalize, after the worker
        # join -> ack_dead: a run that fails closed normally is never touched) it stays quiet; VERITY_COLLECT_DEAD_GRACE_S (default 120 s)
        # without either and it writes <out>/COLLECTOR_DEAD_HUNG.json (path from VERITY_COLLECT_WATCHDOG_MARKER, set by commit_delta) and
        # _exit(3)s -- rc 3 is the "commit_delta refused/failed closed" code row_pod already reports as COMMIT FAIL.  Never a silent hang.
        grace = settings.VERITY_COLLECT_DEAD_GRACE_S
        marker = settings.VERITY_COLLECT_WATCHDOG_MARKER
        if hasattr(self._nc, "set_watchdog"):
            self._nc.set_watchdog(marker, grace)
        info = {"windowed": True, "window_bytes": W, "slots": K, "ring_bytes": K * W, "scratch_bytes": W, "largest_tensor_bound": largest,
                "backpressure": "block-until-slot-free (no drop)", "slot_wait_timeout_s": float(self._nc.win_wait_timeout_s),
                "dead_watchdog": {"grace_s": grace, "marker": marker or None, "rc": 3},
                "in_place_hashing": "hit-plan steps whole; learning steps keeping no bytes to each row's h in place up to rows_hold_bytes "
                                    "of h, leaves from h at step end; their later windows and others re-upload from host"}
        memtrace("window ring allocated")
        self.window_info = info
        return info

    # ------------------------------------------------------------------ rows-sha512 (step_rows)
    def _window_levels(self, n: int, j: int) -> tuple[list[int], int]:
        """Node counts of a step's levels 0 .. min(j, height) for n leaves, and the digest size."""
        width = 1 << merkle_depth(n)
        return [width >> L for L in range(min(j, width.bit_length() - 1) + 1)], step_rows.ROOT_BYTES

    def _row_table(self) -> Any:
        if self._row_tn is None:
            self._row_tn = torch.frombuffer(bytearray(step_rows.t_nibbles()), dtype=torch.uint8).cuda()
        return self._row_tn

    def _row_step(self, step: int, n: int, ks: bytes) -> Any:
        row_prefix, salt_prefix, leaf_prefix = self._row_consts
        return ext().RowStep(self._row_domain(step, n).domain_id, ks, row_prefix, salt_prefix, leaf_prefix, step_rows.SCHEMA, step_rows.ROW_BYTES)

    def _window_root(self, step: int, state: dict[str, Any], padded: int, side: Any) -> tuple[list[Any], bytes, float]:
        """The step's tree levels, Merkle root and tree ms from its absorbed windows (`state`, cleared): the upper tree over the per-window
        top nodes (level n_levels-1 of the step), its levels to pinned host too when the step keeps levels."""
        H = self._hasher
        if not padded:
            return [], self._row_domain(step, 0).empty_root(), 0.0
        top_span = 1 << (state["n_levels"] - 1)
        n_tops = state["width"] // top_span
        if state.get("leaves"):
            self._fold_leaves(state, side)
        if sum(int(t.shape[0]) for t in state["tops"]) != n_tops:
            raise RuntimeError(f"windowed absorb: {len(state['tops'])} window subtrees for {state['n_total']} leaves (expected {n_tops})")
        with torch.inference_mode(), torch.cuda.stream(side):
            tops = torch.cat(state["tops"]) if len(state["tops"]) > 1 else state["tops"][0]
        keep_lv = state["views"] is not None
        upper_dst = self.pool.take((2 * n_tops - 1) * 64) if keep_lv and self.retain_pinned else torch.empty(0, dtype=torch.uint8)
        upper, mr, tree_ms = H.rows_tree(state["rs"], tops, state["n_levels"] - 1, side.cuda_stream, keep_lv, upper_dst)
        levels = list(state["views"][:state["n_levels"] - 1]) + list(upper) if keep_lv else []
        del tops
        state.clear()
        return levels, mr, tree_ms

    def _hold_leaves(self, state: dict[str, Any], lv: Any, side: Any) -> None:
        """rows-sha512, a step that keeps no levels: window leaves `lv` wait on the device for the step's window levels, built in one pass
        of wide launches rather than one subtree a window; the leaves held so far fold first when `lv` would take them past
        `rows_hold_bytes`."""
        if state["leaves"] and sum(int(t.numel()) for t in state["leaves"]) + int(lv.numel()) > self.rows_hold_bytes:
            self._fold_leaves(state, side)
            self._win_leaf_folds = getattr(self, "_win_leaf_folds", 0) + 1
        state["leaves"].append(lv)

    def _fold_leaves(self, state: dict[str, Any], side: Any) -> None:
        """The held leaves, whole windows from the first not yet folded, to their windows' top nodes (level n_levels - 1), after the
        step's earlier tops; the held tensors are released once concatenated, before the levels above them are built."""
        n_levels = state["n_levels"]
        first = sum(int(t.shape[0]) for t in state["tops"]) << (n_levels - 1)
        with torch.inference_mode(), torch.cuda.stream(side):
            lv = torch.cat(state["leaves"]) if len(state["leaves"]) > 1 else state["leaves"][0]
            state["leaves"] = []
            state["tops"].append(self._hasher.rows_tree_window(state["rs"], lv, n_levels, first, side.cuda_stream, []))

    def _retain_from_transient(self, job: _StepJob, tblocks: list[tuple[int, Any, Any, int]], keep_set: Any, chunk_parts: Any,
                               ranges: list[tuple[int, int, Any]], side: Any) -> int:
        """Retention of a bounded LEARNING step, carved out of its transient copies (`tblocks`) now that the metas are known (retain=host:
        every chunk outside the excluded families' interiors; retain=sample: the challenged windows, or with `chunk_parts` only their
        challenged chunks), pinned -> pinned into `ranges`; then the transient blocks go back to the pool, so the step's host footprint
        returns to its retained bytes.  -> the bytes the exclusions kept off host"""
        W, C = self.win_bytes, self.chunk
        excl_bytes = 0
        with torch.inference_mode(), torch.cuda.stream(side):
            if self.retain == "host":
                excl = _excluded_chunk_ranges(job.metas, self.retain_exclude, C) if self.retain_exclude else []
                for kk, _hb, hv, n in tblocks:
                    base = kk * W
                    kept_b = 0
                    for a, b in outside(excl, base, base + n):
                        rv = self.pool.take(b - a)
                        rv.copy_(hv[a - base:b - base], non_blocking=True)
                        ranges.append((a, b, rv))
                        kept_b += b - a
                    excl_bytes += n - kept_b
            elif keep_set is not None:
                for kk, _hb, hv, n in tblocks:
                    if (job.step, kk) in keep_set:
                        if chunk_parts is not None:
                            base = kk * W
                            for a, b in chunk_parts(job.step, base, base + n):
                                rv = self.pool.take(b - a)
                                rv.copy_(hv[a - base:b - base], non_blocking=True)
                                ranges.append((a, b, rv))
                        else:
                            rv = self.pool.take(n)
                            rv.copy_(hv[:n], non_blocking=True)
                            ranges.append((kk * W, kk * W + n, rv))
            side.synchronize()
        for _kk, hb, _hv, _n in tblocks:
            self.pool.give_back(hb)
        return excl_bytes

    def _commit_step_windowed(self, job: _StepJob) -> None:
        """Worker: absorb the step WINDOW BY WINDOW while the serving thread is still filling later windows.  Per window: wait for the
        serving thread's complete-event, D2H the raw bytes to pinned host (retention / re-upload source), hash its leaves in place when the
        step's padded length, and so its frame domain, is already known (hit plan), hand the slot back.  After the step ended (learning
        step): frame the windows hashed to h in place, re-upload the other host windows one at a time through the scratch window and hash
        them with the now-known length; tree over all leaves; bind the root."""
        from verity_vllm.acquire.committer_api import StepCommitment

        side = self._side_stream()
        nc, H, W, C = self._nc, self._hasher, self.win_bytes, self.chunk
        if getattr(nc, "win_dead", False):
            # the serving thread's slot wait timed out earlier in this run: every stage() since was refused by name, so this step
            # has nothing staged -- fail it by the SAME name instead of tripping over an empty tree ("cudaMemcpyAsync(root): invalid argument").
            raise RuntimeError("windowed staging: collector dead after a slot-wait timeout earlier in this run (step not committed)")
        t0 = time.perf_counter()
        plan_used = int(getattr(job, "plan_used", -1))
        inplace = plan_used >= 0
        host_wins: list[tuple[int, Any, int]] = []                              # (k, pinned host view, bytes)
        transient: list[int] = []                                               # windows on host only for the re-hash (retain none|sample)
        j = self.win_log2_chunks                                                # W = C * 2^j: window k holds leaves [k 2^j, (k+1) 2^j)
        state: dict[str, Any] = {}                                              # per-step subtree state (reset when the plan length was wrong)
        ks = self._row_key(job.step)
        tn = self._row_table()

        def begin_tree(n_total: int) -> None:
            """Pinned host buffers for levels 0..n_levels-1 of the WHOLE step (n_L = the tree's width, a power of two, over 2^L); the
            per-window subtrees are copied straight into their slices (no host concat).  n_levels = min(j, height) + 1: below the window
            level the tree is a disjoint union of per-window subtrees; the upper tree is built over the per-window top nodes by
            `H.rows_tree` afterwards."""
            sizes, dig = self._window_levels(n_total, j)
            n_levels = len(sizes)
            state.update(rs=self._row_step(job.step, n_total, ks), width=sizes[0], absorbed=[], leaves=[])
            if not self.retain_levels:
                state.update(n_total=n_total, n_levels=n_levels, views=None, tops=[])
                return
            buf = self.pool.take(sum(sizes) * dig) if self.retain_pinned else torch.empty(sum(sizes) * dig, dtype=torch.uint8)
            views, off = [], 0
            for nL in sizes:
                v = buf[off:off + nL * dig]
                views.append(v.view(nL, dig))
                off += nL * dig
            state.update(n_total=n_total, n_levels=n_levels, views=views, tops=[])

        def row_leaves(src: Any, nbytes: int, k: int) -> Any:
            """rows-sha512: window k's leaves, its nbytes / ROW_BYTES rows at src and then pads to the window's end (or the tree's width); capped
            launches for the window (its leaves and subtree) while the ring has room."""
            first, rs = k << j, state["rs"]
            rs.max_blocks = self.rows_cap_blocks if self.rows_cap_blocks and nc.free_slots() >= self.rows_cap_free else 0
            return H.rows_leaves(rs, src, nbytes, first, min(1 << j, state["width"] - first), state["n_total"], tn, side.cuda_stream)[0]

        def absorb(k: int, lv_k: Any) -> None:
            """Subtree of window k (lv_k = its leaves, device) -> host level slices + top node kept on device."""
            n_k = int(lv_k.shape[0])
            views, n_levels = state["views"], state["n_levels"]
            state["absorbed"].append(k)
            if views is None:
                # no levels kept: the window's leaves (64 B a row) wait on the device for its tree (_window_root) up to rows_hold_bytes;
                # a window's own subtree is two launches of a few blocks each, bound by latency and by waiting for SMs
                self._hold_leaves(state, lv_k, side)
                return
            dsts = [views[L][k << (j - L):(k << (j - L)) + (n_k >> L)] for L in range(n_levels)]
            state["tops"].append(H.rows_tree_window(state["rs"], lv_k, n_levels, k << j, side.cuda_stream, dsts))

        # host copy of the raw window: retention (retain=host) and/or the re-upload source for learning steps
        keep_set = self.retain_windows if self.retain == "sample" else None     # retain=sample: only these (step, k) windows are kept
        # BOUNDED staging (see __init__): learning steps copy windows to TRANSIENT pool blocks under a budget (given back after
        # the re-hash), or discard them on a learn-only pass; hit steps take no hedge copy; retain=host + exclude keeps byte RANGES.
        bounded = self.bounded
        learning = plan_used < 0
        learn_only = bounded and learning and self.learn_only
        # rows on a learning step that keeps no bytes: a leaf's h needs only the step's salt key, so each window is hashed in place to its
        # h (64 B a row, kept on the device) and released with no host copy; its leaves follow from them once the step's length, and so
        # its frame domain, is known (rows_frame) -- the same leaves as the re-upload below.  Up to rows_hold_bytes of h: the window that
        # would pass it and the step's later ones take the host path below (copied, under bounded staging's budget, and re-uploaded)
        hfirst = learning and self.retain == "none"
        hwins: list[tuple[int, Any]] = []
        rs_h = self._row_step(job.step, 1, ks) if hfirst else None
        tblocks: list[tuple[int, Any, Any, int]] = []                           # bounded learning: (k, block, view, bytes) re-hash source
        tbytes = 0
        discard = False
        excl: list[tuple[int, int, str]] = []
        ranges: list[tuple[int, int, Any]] = []                                 # bounded retain=host (+ exclude): retained byte ranges
        excl_bytes = 0
        if bounded and not learning and self.retain == "host" and self.retain_exclude:
            # the metas of the plan this step follows: cached by the worker at the end of the learning step of the same (key, version) (it
            # absorbs steps in order), or, when it no longer holds that plan, built from the plan the serving thread exported at step begin
            hint = self._metas_for(job, (*job.plan_hint, job.hint_entries)) if getattr(job, "plan_hint", None) else None
            if hint is None:
                raise RuntimeError(f"bounded staging: hit step {job.step} has no cached metas for plan {getattr(job, 'plan_hint', None)} "
                                   "(cannot place the retain-exclude ranges)")
            excl = _excluded_chunk_ranges(hint, self.retain_exclude, C)

        # bounded retain=sample with the picks known by chunk (`retain_chunks`, set by openings_after_release for a replay): the retained
        # bytes of a window are the challenged chunks inside it, sorted, as absolute step-stream ranges
        chunk_picks: dict[int, list[int]] | None = None
        if bounded and self.retain == "sample" and getattr(self, "retain_chunks", None):
            chunk_picks = {}
            for st_, idx in self.retain_chunks:
                if st_ == job.step:
                    chunk_picks.setdefault((idx * C) // W, []).append(int(idx))

        def chunk_parts(step: int, lo: int, hi: int) -> list[tuple[int, int]]:
            idxs = sorted(set((chunk_picks or {}).get(lo // W, [])))
            return [(i * C, i * C + C) for i in idxs if lo <= i * C < hi]

        if inplace and plan_used > 0:
            begin_tree(-(-plan_used // C))
        k = 0
        leaf_ms = 0.0
        idle = 0.0                                                              # waiting for the serving thread (a window, the step's end)
        while True:
            tw = time.perf_counter()
            ok, used_k, slot = nc.wait_window(job.seq, k, side.cuda_stream)
            idle += time.perf_counter() - tw
            if not ok:
                if int(used_k) < 0 or getattr(nc, "win_dead", False):
                    # the serving thread's slot wait timed out mid-step: the remaining windows will never be marked, so
                    # fail THIS step by the same name now (rather than spin here until COMMIT_TIMEOUT).
                    raise RuntimeError(f"windowed staging: collector dead (slot-wait timeout) while absorbing step {job.step} window {k} -- step not committed")
                break
            used_k = int(used_k)
            padded_k = -(-used_k // C) * C
            st = self.win_tensors[int(slot)]
            # lifetime NEGATIVE on the windowed path (the default; the arena-path fault below never runs here): the slot
            # is handed back to the ring BEFORE the retained copy is taken and the copy is read only after the serving thread refilled the
            # slot -> the tree (hashed in place from the right bytes) is correct, the retained copy holds a later window's bytes ->
            # openings / openings-after-release MUST fail against the (correct) root: exit 3 named, never PASS.  In-place steps only.
            fault_lifetime = self.fault == "release_before_keep" and inplace and padded_k > 0
            with torch.inference_mode(), torch.cuda.stream(side):
                if padded_k > used_k:
                    st[used_k:padded_k].zero_()
                hv = None
                keep_win = self.retain == "host" or (keep_set is not None and (job.step, k) in keep_set)
                if hfirst and padded_k and sum(int(h.numel()) for _kk, h in hwins) + 64 * (padded_k // C) > self.rows_hold_bytes:
                    hfirst = False                                              # past rows_hold_bytes of h: this window on takes the host path
                if hfirst:
                    if padded_k:
                        rs_h.max_blocks = self.rows_cap_blocks if self.rows_cap_blocks and nc.free_slots() >= self.rows_cap_free else 0
                        hwins.append((k, H.rows_h(rs_h, st, padded_k, k << j, tn, side.cuda_stream)[0]))
                elif bounded and not fault_lifetime:
                    if learning:
                        # learning step: the window is copied to a TRANSIENT block (re-hash source; given back after) while the budget
                        # holds; over budget a learn-only pass drops everything copied so far and just consumes the ring (the plan is
                        # still recorded at step end), a record run fails closed by name -- never a whole-step host copy
                        if not discard and padded_k:
                            if tbytes + padded_k > self.learn_budget_bytes:
                                if not learn_only:
                                    raise RuntimeError(f"bounded staging: learning step {job.step} exceeds the transient host budget "
                                                       f"{self.learn_budget_bytes >> 20} MiB at window {k} ({tbytes + padded_k >> 20} MiB): "
                                                       "its plan was not pre-learned (warm-up learn-only pass) -- step not committed")
                                side.synchronize()                              # the in-flight D2H into the blocks given back
                                for _kk, hb, _hv, _n in tblocks:
                                    self.pool.give_back(hb)
                                tblocks.clear()
                                tbytes = 0
                                discard = True
                            else:
                                hb = self.pool.take_block(padded_k)
                                hvb = hb[:padded_k]
                                hvb.copy_(st[:padded_k], non_blocking=True)      # D2H pinned, side stream
                                tblocks.append((k, hb, hvb, padded_k))
                                tbytes += padded_k
                    elif padded_k and self.retain == "host":
                        base = k * W
                        kept_b = 0
                        for a, b in outside(excl, base, base + padded_k):
                            rv = self.pool.take(b - a)
                            rv.copy_(st[a - base:b - base], non_blocking=True)  # D2H pinned, side stream (only the retained chunks)
                            ranges.append((a, b, rv))
                            kept_b += b - a
                        excl_bytes += padded_k - kept_b
                    elif padded_k and keep_set is not None and (job.step, k) in keep_set:
                        if chunk_picks is not None:
                            # replay-to-open: keep ONLY the challenged chunks of this window (`chunk` bytes each) as byte ranges -- not the
                            # 256 MiB window; the replay's authentication paths come from its levels, so this is all a replay retains
                            base = k * W
                            for a, b in chunk_parts(job.step, base, base + padded_k):
                                rv = self.pool.take(b - a)
                                rv.copy_(st[a - base:b - base], non_blocking=True)
                                ranges.append((a, b, rv))
                        else:
                            hv = self.pool.take(padded_k)
                            hv.copy_(st[:padded_k], non_blocking=True)
                elif not fault_lifetime:                                        # unbounded: every window, the learning-step re-hash source
                    hv = self.pool.take(padded_k)
                    hv.copy_(st[:padded_k], non_blocking=True)                  # D2H pinned, side stream
                # in-place hashing only while the step stays inside the plan's length: a window running past it means the plan changed
                # mid-step (learning) -> the frame domain (the step's row count) is wrong -> the step is re-hashed from the host windows below
                if inplace and padded_k and state and (k << j) + padded_k // C <= state["n_total"]:
                    lv_k = row_leaves(st, padded_k, k)
                    absorb(k, lv_k)                                             # syncs the side stream: the slot's reads are done here
                    del lv_k
            if fault_lifetime:
                gen0 = int(nc.slot_generation(int(slot)))
                nc.release_slot(int(slot), side.cuda_stream)                    # BUG (deliberate): released with no retained copy taken yet
                if nc.wait_slot_reused(int(slot), gen0, 5.0, side.cuda_stream):  # the ring refilled the slot: its bytes are a later window's
                    with torch.inference_mode(), torch.cuda.stream(side):
                        hv = self.pool.take(padded_k)
                        hv.copy_(st[:padded_k], non_blocking=True)              # "retained" copy = the wrong occupant
                        side.synchronize()
                    self._fault_lifetime_windows = getattr(self, "_fault_lifetime_windows", 0) + 1
                else:
                    with torch.inference_mode(), torch.cuda.stream(side):       # no later window came (end of run): fall back to the right bytes
                        hv = self.pool.take(padded_k)
                        hv.copy_(st[:padded_k], non_blocking=True)
                        side.synchronize()
            else:
                nc.release_slot(int(slot), side.cuda_stream)                    # the next taker waits on this point of the side stream
            if hv is not None:
                host_wins.append((k, hv, padded_k))
                if not keep_win:
                    transient.append(k)                                         # re-hash source only: dropped from `keep` below
            k += 1
        tw = time.perf_counter()
        job.done_evt.wait()                                                     # type: ignore[attr-defined]  plan / cursor / metas known
        idle += time.perf_counter() - tw
        snap = getattr(job, "rows_snapshot", None)
        if snap is not None:
            try:
                self._expand_step_rows(int(job.step), *snap)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"step rows: {e!r}"[:200])
        job.metas = self._metas_for(job)
        used = int(job.cursor)
        padded = -(-used // C) * C
        plan_pad = -(-plan_used // C) * C if plan_used >= 0 else -1
        mismatch = plan_used >= 0 and plan_pad != padded                       # the plan changed mid-step (learning): in-place work is void
        if bounded and mismatch:
            # a hit step that ran past / short of its plan: the prescribed layout changed mid-step and, bounded, no host copy exists to
            # re-hash from.  A learn-only pass records the new plan and moves on (the step is not a record); a record run fails CLOSED by
            # name (the tree hashed in place is framed for the plan's row count -- it is NOT the commitment of this step's bytes).
            self.plan_mismatches += 1
            self.transient_peak_bytes = max(self.transient_peak_bytes, tbytes)
            if not self.learn_only:
                _shape = getattr(job, "plan_shape", (0, 0))
                raise RuntimeError(f"bounded staging: hit step {job.step} (plan ntok={_shape[0]} nreq={_shape[1]} key={job.plan[0]:#x}) ran "  # type: ignore[attr-defined]
                                   f"{padded} bytes against a plan of {plan_pad}: the prescribed layout changed mid-step and no host copy "
                                   "exists to re-hash from (step not committed; re-run the learn-only pass)")
            discard = True
        if discard:
            # learn-only outcome: the plan is exported (end_step), nothing is hashed; placeholder commitment (root b"") -- the warm-up
            # instrumented run is never a record, and a record run never reaches here (raised above / in the loop)
            side.synchronize()
            for _kk, hb, _hv, _n in tblocks:
                self.pool.give_back(hb)
            state.clear()
            if padded and self.retain_pinned:
                # prime the pinned pool with the LEVEL buffers the record run will take for this step (same sizes as begin_tree) and, under
                # retain=host (+ exclude), the same `take` sequence for the retained byte ranges -- all returned to the pool's free list by
                # reset() -- so the cudaHostAlloc cost (~0.4 s/GB) falls in the warm-up, not inside the measured pair.  `reserve` makes
                # the blocks these takes allocate in one burst, their pages faulted in over the pool's helper threads at once
                parts: list[tuple[int, int]] = []
                if self.retain == "host":
                    excl = _excluded_chunk_ranges(job.metas, self.retain_exclude, C) if self.retain_exclude else []
                    parts = [p for kk in range(-(-padded // W)) for p in outside(excl, kk * W, kk * W + min(W, padded - kk * W))]
                sizes, dig = self._window_levels(padded // C, j)
                self.pool.reserve(([dig * sum(sizes)] if self.retain_levels else []) + [b - a for a, b in parts])
                begin_tree(padded // C)
                state.clear()
                for a, b in parts:
                    self.pool.take(b - a)
            nc.finish_step(job.seq)
            self.learned_only_steps.append(job.step)
            self.learned_only_bytes += padded
            metas = job.metas
            self._layouts[job.step] = metas
            self._gpu_blocks[job.step] = (None, used)
            t2 = time.perf_counter()
            self.spans.add("hashing", t2 - t0, step=job.step, nbytes=0)
            if self.footprint is not None:
                self.footprint.event("learned_only", step=job.step, used=used, padded=padded)
            self._done[job.step] = StepCommitment(job.step, b"", b"", 0, metas.nbytes, len(metas), b"", job.t_step_end, t2, job.backlog)
            return
        if bounded and learning:
            host_wins = [(kk, hv, n) for kk, _hb, hv, n in tblocks]          # the re-hash source (transient blocks; given back below)
            self.transient_peak_bytes = max(self.transient_peak_bytes, tbytes)
        if padded and (not inplace or mismatch):
            # the step length was unknown (learning step) or the plan changed mid-step -> the leaves with the real length: the windows
            # hashed to h in place framed first (each h dropped once framed), then the host windows re-hashed
            if not host_wins and not hwins:
                raise RuntimeError(f"windowed absorb ({step_rows.NAME}): the plan changed mid-step and no host copy exists to re-hash from")
            state.clear()
            begin_tree(padded // C)
            self._win_hfirst_steps = getattr(self, "_win_hfirst_steps", 0) + bool(hwins)
            with torch.inference_mode(), torch.cuda.stream(side):
                while hwins:
                    kk, hs_k = hwins.pop(0)
                    first = kk << j
                    absorb(kk, H.rows_frame(state["rs"], hs_k, first, min(1 << j, state["width"] - first), state["n_total"], side.cuda_stream))
                for kk, hv, n in host_wins:
                    if not n:
                        continue
                    self.win_scratch[:n].copy_(hv[:n], non_blocking=True)     # H2D pinned; same stream as the leaf kernel -> ordered
                    lv_k = row_leaves(self.win_scratch, n, kk)
                    absorb(kk, lv_k)
                    del lv_k
            self._win_reupload_bytes = getattr(self, "_win_reupload_bytes", 0) + sum(n for _kk, _hv, n in host_wins)
        if padded:
            # the windows past the step's last row hold only pads (the tree is padded to a power of two): their subtrees, after the
            # data windows', which must all be in
            n_data = -(-state["n_total"] // (1 << j))
            if state["absorbed"] != list(range(n_data)):
                raise RuntimeError(f"windowed absorb ({step_rows.NAME}): windows {state['absorbed'][:8]} absorbed for {state['n_total']} leaves "
                                   f"(expected 0 .. {n_data - 1} in order)")
            with torch.inference_mode(), torch.cuda.stream(side):
                for kk in range(n_data, max(1, state["width"] >> j)):
                    absorb(kk, row_leaves(self.win_scratch, 0, kk))
        levels, mr, tree_ms = self._window_root(job.step, state, padded, side)
        nc.finish_step(job.seq)
        if bounded and learning:
            excl_bytes += self._retain_from_transient(job, tblocks, keep_set, chunk_parts if chunk_picks is not None else None, ranges, side)
            tblocks = []
            host_wins = []
        if bounded and self.retain in ("host", "sample"):
            if self.retain == "sample" and not learning and chunk_picks is None:
                kept = [(kk, hv, n) for kk, hv, n in host_wins if kk not in transient]
                keep = _HostWindows(kept, used, W, C) if kept else None
            else:
                ex_fams: dict[str, int] = {}
                for elo, ehi, fam in excl:
                    ex_fams[fam] = ex_fams.get(fam, 0) + (min(ehi, padded) - elo if elo < padded else 0)
                keep = _HostRanges(ranges, used, ex_fams, C) if ranges else None
            self.excluded_bytes += excl_bytes
        else:
            kept = [(kk, hv, n) for kk, hv, n in host_wins if kk not in transient] if self.retain in ("host", "sample") else []
            keep = _HostWindows(kept, used, W, C) if kept else None
        self._footprint_absorbed(job, padded, levels, keep)
        self._gpu_t["leaf_ms"] += leaf_ms
        self._gpu_t["tree_ms"] += tree_ms
        t1 = time.perf_counter()
        metas = job.metas
        nbytes = metas.nbytes
        n = padded // C
        layout_dig = self.layout_digest(metas)
        root = step_rows.step_root(self.program_digest, self._ctx_digest(job.step), self._geo_digest, layout_dig, n, mr)
        t2 = time.perf_counter()
        self.spans.add("hashing", t1 - t0, step=job.step, nbytes=nbytes, leaves=n)
        self.spans.add("worker_idle", idle, step=job.step)                     # inside `hashing`: the rest of it is the worker's own work
        self.spans.add("root_finalisation", t2 - t1, step=job.step)
        if self.retain_levels:
            self._gpu_levels[job.step] = levels
        self._gpu_blocks[job.step] = (keep, used)
        self._layouts[job.step] = metas
        self._done[job.step] = StepCommitment(job.step, root, mr, n, nbytes, len(metas), layout_dig, job.t_step_end, t2, job.backlog,
                                              hashing_s=t1 - t0, root_finalisation_s=t2 - t1)

    # ------------------------------------------------------------------ reporting

    # ------------------------------------------------------------------ window views (`NativeHostCommitter.detach`)

    SHARED_LAZY = NativeHostCommitter.SHARED_LAZY + ("_row_tn",)

    def _detach_into(self, view: Any) -> None:
        """The collector's run totals are the serving thread's (its step ends bump them) and the next window's reset clears them, so
        the view keeps them as they stand at the detach, with the run's failure so far."""
        super()._detach_into(view)
        view._py_committed = self._py_committed
        view._nc_at_detach = {"stats": dict(self._nc.stats()), "window_stats": dict(self._nc.window_stats()),
                              "failure": str(getattr(self._nc, "failure", "") or "")}

    def _window_ended(self, view: Any) -> None:
        """Worker: a step of the window it abandons fails the run by name after the next window's reset; the window's own failure is
        the one at its detach or this one. The worker's metas retire a generation here, where the old reset retired them."""
        if not view._nc_at_detach["failure"]:
            view._nc_at_detach["failure"] = str(getattr(self._nc, "failure", "") or "")
        self._metas_cache.end_run()

    def _nc_now(self) -> dict:
        """The collector's run totals and failure for this committer's window: a detached view's from its detach, else live."""
        snap = getattr(self, "_nc_at_detach", None)
        if snap is not None:
            return snap
        return {"stats": dict(self._nc.stats()), "window_stats": dict(self._nc.window_stats()),
                "failure": str(getattr(self._nc, "failure", "") or "")}

    def finalize(self):
        if getattr(self, "_nc_at_detach", None) is None:
            for src in self.sources:
                if hasattr(src, "sync_stats"):
                    src.sync_stats()     # a view's sources were synced, and copied, at its detach
        dead = bool(getattr(self._nc, "win_dead", False)) and hasattr(self._nc, "ack_dead")
        if dead:
            # the run reached finalize -> the identity check fails it closed (the error below is in the record `super().finalize()` builds
            # from self.errors, next to the drained slot-wait error).  The fail-fast watchdog stands down only AFTER super().finalize()
            # returned, i.e. after the worker was joined and the record built -- a join on a step that never completes
            # is exactly the hang it exists for (no progress for the grace -> rc 3 + <out>/COLLECTOR_DEAD_HUNG.json).
            if self.fault == "hang_after_dead":
                # DELIBERATE hang (negative test_e2e_collector_dead_hang_exits_rc3): the Python side never fails the run closed -- the
                # C++ watchdog alone must end the process with rc 3 within the grace.
                print("[collector] FAULT hang_after_dead: not acknowledging the dead collector (simulated hang)", flush=True)
                threading.Event().wait()
            self.errors.append("windowed staging: collector dead (slot-wait timeout) -- run failed closed at finalize")
        try:
            rc = super().finalize()
        finally:
            self.replay_row_keys = None
        if dead:
            self._nc.ack_dead()
        nc = self._nc_now()
        if failure := nc["failure"]:                        # read after the worker drained: a step it abandoned failed the run by this name
            self.errors.insert(0, f"collector failed: {failure}"[:300])
            rc.extra.update(errors=self.errors[:10], errors_n=len(self.errors), collector_failure=failure)
        st = nc["stats"]
        # reference integrity: what the manifest attempted vs what was bound (the Python `_acquire` path puts through stage/collect
        # directly, so its successes are added here); commit_delta refuses PASS unless they are equal and no error was recorded
        rc.extra["expected_tensors"] = int(st.get("expected", 0))
        rc.extra["committed_tensors"] = int(st.get("committed", 0)) + int(self._py_committed)
        rc.extra["windows"] = {**nc["window_stats"], **getattr(self, "window_info", {}),
                               "reupload_bytes": getattr(self, "_win_reupload_bytes", 0),
                               "hfirst_steps": getattr(self, "_win_hfirst_steps", 0),
                               "rows_hold_bytes": self.rows_hold_bytes,
                               "leaf_folds": getattr(self, "_win_leaf_folds", 0),
                               "fault_lifetime_windows": getattr(self, "_fault_lifetime_windows", 0)}    # >0 only under VERITY_FAULT=release_before_keep
        # bounded staging record: what the flag did this run (a learned-only step = a warm-up-only outcome; on a record run the
        # list MUST be empty -- commit_delta refuses PASS otherwise)
        rc.extra["bounded_staging"] = {"enabled": bool(self.bounded), "learn_only": bool(self.learn_only),
                                       "learn_budget_bytes": int(self.learn_budget_bytes),
                                       "learned_only_steps": list(self.learned_only_steps), "learned_only_bytes": int(self.learned_only_bytes),
                                       "transient_peak_bytes": int(self.transient_peak_bytes), "plan_mismatches": int(self.plan_mismatches),
                                       # how the plan key was formed: "geometry" = (ntok, nreq, multiset of per-request
                                       # (num_computed_tokens, num_scheduled_tokens)); "counts" = (ntok, nreq) only (older scheduler output)
                                       "plan_key_mode": getattr(self, "_plan_key_mode", "counts"),
                                       "retain_exclude": sorted(self.retain_exclude), "excluded_bytes": int(self.excluded_bytes),
                                       "host_retained_raw_bytes": sum(int(getattr(b, "retained_bytes", b.numel() if b is not None else 0))
                                                                      for b, _u in self._gpu_blocks.values() if b is not None and not getattr(b, "is_cuda", False)),
                                       # the admission this run was launched under (commit_delta's host bound + the DECLARED schedule it came
                                       # from; None = no admission computed) -- on the record, not only in host_admission_<kind>.json
                                       "host_admission": getattr(self, "host_admission", None)}
        rc.extra["native_collect"] = {**st, "wrap": self.wrap, "profile": self.profile, "gather": self.gather, "native_hooks": self.native_hooks,
                                      "blocks_per_desc": int(self._nc.blocks_per_desc) if self.gather else None,
                                      "plans_cached": len(self._metas_cache),
                                      "names": len(self._names), "consumer_delay_ms": self.consumer_delay_s * 1e3, "fault": self.fault,
                                      "cuda_runtime": int(ext().cuda_runtime_version()),
                                      "fa2_wrap": {s.short_name: s.stats for s in self.sources if hasattr(s, "sync_stats")}}
        return rc

    def reset(self) -> None:
        detached = self._detached
        super().reset()
        self.row_keys = {}
        self.step_rows = {}
        self._py_committed = 0
        self._nc.end_run()              # the collector's run totals, and the plans this run did not use (native_plans.h)
        if not detached:
            self._metas_cache.end_run()  # the worker's metas, by the same rule (plan_cache); a detached window's end retires them
        self.learned_only_steps = []    # per-run bounded-staging counters (the plans learned stay for the next run)
        self.learned_only_bytes = 0
        self.transient_peak_bytes = 0
        self.excluded_bytes = 0
        self.plan_mismatches = 0
        self._win_reupload_bytes = 0    # windowed: per-run counters of finalize's extra["windows"]
        self._win_hfirst_steps = 0
        self._win_leaf_folds = 0
        self._oar_deferred = []         # picks the in-run phases could not open from host (excluded families) -> openings_after_release
        for src in self.sources:
            fn = getattr(src, "begin_run", None)
            if fn is not None:
                fn()                    # per-run source counters (slots/hooks persist across the process's runs)
