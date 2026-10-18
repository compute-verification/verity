"""Candidate arm 3 (R10 acq-native): native collection at the existing hook boundaries — same committed values, same layout as CMT-3.
Glossary (Draft 3 §12): this module is VALUE ACQUISITION (taking each required Value's bytes at its runtime occurrence, the module
hook) feeding the commitment; the `_flush_*` / `*_LEAVES` tables below are the hand-written runtime correspondence (which module
leaf produces which Value, and when its bytes are stable) that Phase 3/4 replace with recorded correspondence + an acquisition plan.

What changes vs the frozen R9 complete path (`native_host_fast+hidden_m1_gputree`, arm 2) -- ONLY the serving-thread acquisition:

  R9 hook body (per tensor, Python)            candidate hook body (per tensor)
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
later kernel on that stream can overwrite it in place (fused add-RMSNorm residual, RoPE in the qkv buffer) -- CMT-3's known-safe placement.
No collector stream and no per-tensor event are needed for that; the step event (recorded on the same stream at step end) orders the
worker's side-stream hashing after every snapshot, exactly as in R9.  Owner 2's lifetime table may later license combining several copies
into one native call; this file does not do that (secondary bet, only within proven-safe windows).

Everything downstream (arena block lifecycle, GPU-resident SHA-256 chunk tree, retained block + lazy openings, roots, verifier, FA2 M1
export source) is inherited unchanged from `NativeHostCommitter`.

Correctness gate: the run root of this committer equals the run root of arm 2 on the same request (deterministic encoding; there is no
commitment randomness in CMT-3: program digest = sha256("cmt-integ/no-program-digest/" + run_id), ctx digest = run_id/step).

Stress / negative-test knobs (never in headline runs; recorded when used):
  VERITY_CONSUMER_DELAY_MS=<ms>   the worker sleeps before consuming each step (delayed commitment consumer): backlog grows, the arena pool
                                  must grow instead of recycling a block early; openings must still verify 64/64 and roots must still equal arm 2
  VERITY_FAULT=release_before_keep  DELIBERATE ordering bug: the arena block is released for reuse BEFORE the retained copy (openings read it)
                                  is taken, and the worker waits for the next step to land in it -> the openings check must FAIL loudly.
                                  [R15 capacity] On the WINDOWED path (rc3 default) the same fault releases the window slot before the
                                  retained copy and takes the copy only after the ring refilled the slot (native wait_slot_reused):
                                  tree correct, retained bytes wrong -> openings (after release) FAIL named; until R15 the flag was a
                                  silent no-op there (the windowed branch returned before the fault check).
  VERITY_FAULT=drop1_dup0         DELIBERATE identity bug (R14 BIND-01 negative, F-rev-01): every layer>0 input_layernorm binds Value /0 twice and drops /1
                                  -> counts and classes unchanged; the required-value coverage check MUST fail (Python wrappers are used for every module)
  VERITY_FAULT=skip_copy:<k>      DELIBERATE value bug: the k-th snapshot of every step is not issued (stale arena bytes) -> the root must differ
                                  from arm 2 (root-equality gate) while the tree stays self-consistent (openings alone cannot see it)
  VERITY_FAULT=hang_after_dead    [R16 sys, F-stoch-03] DELIBERATE hang: after the slot-wait timeout declared the collector dead, finalize
                                  blocks forever instead of failing the run closed -> the C++ dead-watchdog alone must end the process
                                  (rc 3 + <out>/COLLECTOR_DEAD_HUNG.json) within VERITY_COLLECT_DEAD_GRACE_S (default 120 s).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

import torch

from verity.ml.commitments.merkle import merkle_depth
from verity_vllm.commit import step_rows, step_rows_b3
from verity_vllm.commit.committer.flush_points import GATHER_FLUSH_LEAVES, PRE_FLUSH_LEAVES
from verity_vllm.commit.committer.host_retention import (HIDDEN_M1_FAMILIES, RETAIN_EXCLUDE_ALIASES, _excluded_chunk_ranges,  # noqa: F401
                                                         _HostRanges, _HostWindows, outside, retain_exclude_families)
from verity_vllm.acquire.sources.hidden_source import (
    HiddenM1Source,
    HiddenM1SourceFA3,
    fa2_kblock_n,
    fa2_swapped,
    fa2_tap_buffer_words,
    fa2_tap_words_per_seq,
    plan_fa2_tap_groups,
)
from verity_vllm.commit.committer.native_host import (
    GPU_CHUNK_BYTES,
    NativeHostCommitter,
    TensorMeta,
    _StepJob,
    memtrace,
)
from verity_vllm.commit.committer.plan_cache import PlanCache, PlanMetas
from verity_vllm.commit.committer.topp_splits import (SPLITS_IDENTITY, TOPP_SPLIT_KERNELS, TOPP_SPLIT_MODULE, _SplitKernelProxy,  # noqa: F401
                                                      batch_top_p, install_split_kernel_proxies, split_count_of_launches, step_splits,
                                                      uninstall_split_kernel_proxies)
from verity_vllm.commit.native_ranges import level_sizes

_EXT = None
HERE = os.path.dirname(os.path.abspath(__file__))


def plan_key_of(sched: Any, ntok: int, nreq: int) -> tuple[int, str]:
    """The collector's plan key for one engine step = the step's SHAPE CLASS, i.e. everything the staged byte layout depends on.

    [R17 fp8, F-fp8-PLANKEY] Until 01:28Z 2026-09-21 the key was (ntok << 16) | nreq.  That is not the layout: the FA3 hidden M1 export
    plane (`fa3_hidden_m1`) is sized by each request's KV length (BN-blocks of the sequence), so two decode steps with the same
    (ntok, nreq) stage different byte counts -- the B8 mixed 1024/128 record run hit it at steps 35 and 80 (both ntok 4 / nreq 4, the same
    four live requests 45 positions apart: 78,892,288 vs 87,297,280 bytes) and failed CLOSED by name, as the bounded path must (a hit-plan
    step hashes in place and keeps no host copy to re-hash from).  The key now binds the per-request geometry as a multiset:

        key = blake2b-64( (ntok, nreq, sorted[(num_computed_tokens_i, num_scheduled_tokens_i) for every scheduled request]) ) & (2**63-1)

    Two steps with equal keys have equal per-request (KV length, new tokens) multisets, hence equal staged shapes for every plane whose
    size is a function of those (module outputs: ntok; hidden M1: KV blocks per request; token ids: ntok/nreq).  Row ORDER inside a
    tensor is not bound -- v1 leaf headers bind the step length M only, and the plan is a sequence of (name, bytes) in hook order.
    Between the warm-up (learn) run and the record run of the same declared, batch-invariant, greedy workload the schedule is identical,
    so every record step hits the plan its warm-up twin learned; the plan cache holds one entry per distinct step (<= steps x pairs).
    Returns (key, mode): mode "geometry" when the scheduler output exposes the per-request fields (vLLM V1 `SchedulerOutput`:
    scheduled_new_reqs[].num_computed_tokens, scheduled_cached_reqs.{req_ids,num_computed_tokens}, num_scheduled_tokens{}), else
    "counts" = the old key -- recorded on the run (`bounded_staging.plan_key_mode`) so a record never hides which one it used."""
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
        import hashlib
        h = hashlib.blake2b(repr((int(ntok), int(nreq), geo)).encode(), digest_size=8).digest()
        return int.from_bytes(h, "little") & 0x7FFF_FFFF_FFFF_FFFF, "geometry"
    except Exception:
        return (int(ntok) << 16) | (int(nreq) & 0xFFFF), "counts"


def build_source_digest() -> str:
    """The source-set digest of THIS tree's collector (native_jit.source_set_digest with the toolchain versions) -- the key of its JIT dir."""
    from verity_vllm.commit.committer import native_jit

    return native_jit.source_set_digest(torch_version=str(torch.__version__), cuda_version=str(torch.version.cuda))


def ext(build_dir: str | None = None, verbose: bool = False):
    """JIT-build the C++ collector (pybind11 + cudart + .cu kernels) once per process.  `NATIVE_COLLECT_BUILD` (or `build_dir`) names the
    PARENT of the build directory; the source-set digest picks the child (`<parent>/<digest12>`, native_jit) -- [COORD R17-41 (1)] torch's
    mtime test is never the arbiter of staleness (record_v5/ship.sh preserves source mtimes; a pod-wide dir warmed by another tree's build
    was adopted by the r17-int-3 ship on cp-big).  Without a parent, torch's default cache dir is used, keyed the same way.  Concurrent jobs
    on one host serialise on the dir's build lock (`native_jit.jit_load`)."""
    global _EXT
    if _EXT is None:
        from verity_vllm.commit.committer import native_jit

        _EXT, bd, digest = native_jit.jit_load("verity_native_collect", native_jit.source_paths(HERE),
                                               parent=build_dir or os.environ.get("NATIVE_COLLECT_BUILD") or None,
                                               cflags=native_jit.CFLAGS, cuda_cflags=native_jit.CUDA_CFLAGS, verbose=verbose)
        missing = native_jit.missing_bindings(_EXT, os.path.join(HERE, "native_collect.cpp"))
        if missing:
            _EXT = None
            raise RuntimeError(f"native collector .so in {bd} does not expose {missing} bound by {HERE}/native_collect.cpp -- a stale build "
                               f"(source digest {digest[:12]}); remove the directory and rebuild")
        print(f"[jit] native collector source digest {digest[:12]} -> {bd}", file=sys.stderr, flush=True)
    return _EXT


# the first two zero-window overwriters and their flush leaves (GATHER_FLUSH_LEAVES, PRE_FLUSH_LEAVES): `flush_points`
# [R16 moe FLUSH-03] a third zero-window overwriter, INSIDE the fused-MoE block and invisible to every module hook: vLLM's TritonExperts
# carves `intermediate_cache1` (the w13 GEMM output = the `moe/L<k>/up` plane) and `intermediate_cache3` (the w2 GEMM output = `down`) out
# of the SAME workspace (`_resize_cache(workspace2, ...)` twice, offset 0 -> the two views alias byte for byte when 2I == H, OLMoE), so the
# block's own w2 launch overwrites `up` two launches after it was staged.  The MoE source (`moe_source.MoeBlockSource`) stages `up` right
# after the w13 launch, but the next flush point was the NEXT layer's `input_layernorm` pre-flush -> the committed `up` bytes were the same
# layer's `down` bytes (R16 lane-moe C2 v3 on the real OLMoE B1 record: MoeExpertGemm_v1 320/320 VUs != F_V while committed up == committed
# down row for row; act/down/sum/router planes correct, they are overwritten only by the next layer, after the pre-flush).  Fix: the MoE
# source asks the committer to flush right after staging `up` (`flush_after_plane`); one more gather launch per MoE layer per step, nothing
# for dense rows.  Negative `VERITY_FAULT=late_flush_moe_up` drops that flush (the pre-FLUSH-03 behaviour).
#: [FLUSH-03] fused-MoE planes whose workspace the block itself overwrites before its next module hook: flush right after staging them
MOE_FLUSH_PLANES = frozenset({"up"})


class NativeHiddenM1Source(HiddenM1Source):
    """R11 collector v2: the FA2 M1 tap glue as ONE C++ callable (`Fa2Wrap`, native_collect.cpp) in place of the Python `varlen_fwd`
    wrapper.  Same tapped kernel, same per-geometry buffer rule, same names/order into the same collector; stats synced at uninstall."""

    #: [R18 tap, F-r17-23] default cap on the FA2 tap's device residency (MiB); VERITY_FA2_TAP_CAP_MB=0 restores the unbounded buffers
    DEFAULT_TAP_CAP_MB = 2048

    def __init__(self, committer: "NativeCollectCommitter", so_path: str, ns: str = "verity_fa2_matAll") -> None:
        super().__init__(committer, so_path, ns)
        self._w = ext().Fa2Wrap(committer._nc, self.orig_op, self.tapped_op, committer._intern, ns)
        # R13 P0 (p0b): per-sequence compaction of multi-sequence launches (only the rows the kernel wrote are staged; B=1 unchanged)
        self._w.compact = os.environ.get("VERITY_FA2_COMPACT", "1").strip() not in ("0", "false", "no")
        self._w.zero_after = os.environ.get("VERITY_FA2_ZERO", "1").strip() not in ("0", "false", "no")
        self.short_name, self._w.ms = "hidden_m1", self.ms              # the MS class sizes every buffer and is a seventh compacted plane
        # [R18 tap, F-r17-23] bounded tap: one device buffer of <= cap bytes; a multi-sequence launch whose padded tap buffer exceeds it is
        # split into consecutive sequence groups (hidden_source.plan_fa2_tap_groups) tapped one after another into that buffer (C++ Fa2Wrap
        # seq_base / hold_launch / bound_words).  Names and bytes of the staged compacted rectangles are those of the unsplit launch.
        cap_mb = int(os.environ.get("VERITY_FA2_TAP_CAP_MB", str(self.DEFAULT_TAP_CAP_MB)).strip() or 0)
        self.tap_cap_bytes = max(0, cap_mb) << 20
        self._w.bound_words = self.tap_cap_bytes // 4
        self.telemetry_path: str | None = None           # commit_delta: <out>/fa2_tap_windows.jsonl (one line per tapped window / group)
        self._groups_cache: dict[tuple, Any] = {}         # per step: (cu_seqlens_q ptr, b, max_q, max_k) -> group plan (keeps the group tensors alive)
        self._tel = {"windows": 0, "split_launches": 0, "split_steps": set(), "peak_group_bytes": 0, "peak_unsplit_bytes": 0,
                     "torch_max_alloc": 0, "torch_alloc_peak_in_tap": 0, "largest_unbounded_bytes": 0, "cap_exceeded_seqs": 0}
        self._tel_fh = None

    def _wrapper(self):
        if self.tap_cap_bytes <= 0:
            return self._w
        return self._split_wrapper()

    def on_step_begin(self) -> None:
        self._w.on_step_begin()
        self._groups_cache.clear()

    # ------------------------------------------------------------------ [R18 tap] the split driver
    def _tel_line(self, rec: dict) -> None:
        if not self.telemetry_path:
            return
        try:
            if self._tel_fh is None:
                self._tel_fh = open(self.telemetry_path, "a", buffering=1)
            self._tel_fh.write(__import__("json").dumps(rec, separators=(",", ":")) + "\n")
        except Exception as e:  # noqa: BLE001
            self.c.errors.append(f"fa2 tap telemetry: {e!r}"[:200])
            self.telemetry_path = None

    def _split_wrapper(self):
        me, w, torch = self, self._w, self.torch

        def varlen_fwd(*args, **kwargs):
            c = me.c
            if len(args) < 21 or not bool(getattr(c._nc, "active", False)):
                return w(*args, **kwargs)                         # warm-up / dummy run: the installed kernel (C++ takes the same exit)
            q, k, out, cu_q = args[0], args[1], args[3], args[4]
            block_table = args[8]
            b = int(cu_q.numel()) - 1
            max_q, max_k, num_splits = int(args[10]), int(args[11]), int(args[20])
            H, Hk, D = int(q.shape[1]), int(k.shape[-2]), int(q.shape[-1])
            swapped = fa2_swapped(max_q, H, Hk, D, int(args[16]), int(args[17]), float(args[12]), args[9])
            if b < 2 or swapped or not w.compact or block_table is None:
                return w(*args, **kwargs)                         # unsplit paths: b == 1 whole-buffer identity, swapped decode, non-paged KV
            try:
                BN = fa2_kblock_n(D, num_splits)
            except RuntimeError:
                return w(*args, **kwargs)                         # the C++ wrapper refuses it by name (X-03)
            h, M = H, max_q
            NB = (max_k + BN - 1) // BN
            per_seq = fa2_tap_words_per_seq(h, M, NB, BN, D, me.ms)
            total_bytes = fa2_tap_buffer_words(b, h, M, NB, BN, D, me.ms) * 4
            if total_bytes > me._tel["largest_unbounded_bytes"]:
                me._tel["largest_unbounded_bytes"] = total_bytes  # what the unbounded path would have allocated for this launch
            if total_bytes <= me.tap_cap_bytes:
                me._tel["peak_unsplit_bytes"] = max(me._tel["peak_unsplit_bytes"], total_bytes)
                return w(*args, **kwargs)                         # fits: ONE group in the bounded buffer (C++ split_path), counters inline
            key = (int(cu_q.data_ptr()), b, max_q, max_k)
            plan = me._groups_cache.get(key)
            if plan is None:
                cu_host = cu_q.to("cpu")                          # one D2H per step (all layers share cu_seqlens_q)
                cu = [int(x) for x in cu_host.tolist()]
                groups = plan_fa2_tap_groups(b, per_seq, me.tap_cap_bytes // 4)
                plan = []
                for (g0, g1) in groups:
                    gw = fa2_tap_buffer_words(g1 - g0, h, M, NB, BN, D, me.ms) * 4
                    if gw > me.tap_cap_bytes:
                        me._tel["cap_exceeded_seqs"] += 1
                    plan.append({"g0": g0, "g1": g1, "t0": cu[g0], "t1": cu[g1], "bytes": gw,
                                 "cu_q": (cu_q[g0:g1 + 1] - cu_q[g0]).contiguous()})   # stays alive for the step (C++ seqlens cache keyed by ptr)
                me._groups_cache[key] = plan
                me._tel["split_steps"].add(int(getattr(c._cur, "step", -1)))
            if out is None:
                out = torch.empty_like(q)
            cu_k, seqused_k, leftpad_k, alibi = args[5], args[6], args[7], args[9]
            step = int(getattr(c._cur, "step", -1))
            layer = int(w.launches_in_step)
            results = []
            w.hold_launch = True
            try:
                for gi, g in enumerate(plan):
                    g0, g1, t0, t1 = g["g0"], g["g1"], g["t0"], g["t1"]
                    a = list(args)
                    a[0] = q[t0:t1]
                    a[3] = out[t0:t1]
                    a[4] = g["cu_q"]
                    if isinstance(cu_k, torch.Tensor) and cu_k.numel() == b + 1:
                        a[5] = (cu_k[g0:g1 + 1] - cu_k[g0]).contiguous()
                    if isinstance(seqused_k, torch.Tensor) and seqused_k.numel() == b:
                        a[6] = seqused_k[g0:g1]
                    if isinstance(leftpad_k, torch.Tensor) and leftpad_k.numel() == b:
                        a[7] = leftpad_k[g0:g1]
                    a[8] = block_table[g0:g1]
                    if isinstance(alibi, torch.Tensor) and alibi.dim() == 2 and alibi.shape[0] == b:
                        a[9] = alibi[g0:g1]
                    w.seq_base = g0
                    results.append(w(*a, **kwargs))
                    me._tel["windows"] += 1
                    me._tel["peak_group_bytes"] = max(me._tel["peak_group_bytes"], g["bytes"])
                    alloc = int(torch.cuda.memory_allocated())
                    mx = int(torch.cuda.max_memory_allocated())
                    me._tel["torch_alloc_peak_in_tap"] = max(me._tel["torch_alloc_peak_in_tap"], alloc)
                    me._tel["torch_max_alloc"] = max(me._tel["torch_max_alloc"], mx)
                    me._tel_line({"t": round(time.time(), 3), "step": step, "layer": layer, "group": gi, "groups": len(plan), "seqs": [g0, g1],
                                  "tokens": t1 - t0, "group_bytes": g["bytes"], "bound_bytes": me.tap_cap_bytes, "unbounded_launch_bytes": total_bytes,
                                  "windows": me._tel["windows"], "torch_alloc": alloc, "torch_max_alloc": mx})
            finally:
                w.hold_launch = False
                w.seq_base = 0
            w.end_launch()
            me._tel["split_launches"] += 1
            return me._merge_results(results, out, plan, H)

        return varlen_fwd

    @staticmethod
    def _merge_results(results: list, out: Any, plan: list, H: int):
        """The tapped op's return for the whole launch from the per-group returns: `out` was written in place (row ranges of the one
        tensor); softmax_lse is concatenated along its token dim ([H, total_q]) or batch dim ([b, H, M]); anything else (rng state, None)
        is taken from the first group.  Container type preserved."""
        import torch

        first = results[0]
        if isinstance(first, torch.Tensor):
            return out
        merged = []
        for j, t in enumerate(first):
            if j == 0 or not isinstance(t, torch.Tensor):
                merged.append(out if j == 0 else t)
                continue
            parts = [r[j] for r in results]
            if t.dim() == 2 and t.shape[0] == H and all(p.shape[1] == (g["t1"] - g["t0"]) for p, g in zip(parts, plan)):
                merged.append(torch.cat(parts, dim=1))
            elif t.dim() == 3 and all(p.shape[0] == (g["g1"] - g["g0"]) for p, g in zip(parts, plan)):
                merged.append(torch.cat(parts, dim=0))
            else:
                merged.append(t)
        return type(first)(merged) if isinstance(first, (list, tuple)) else merged

    def sync_stats(self) -> None:
        st = self._w.stats()
        self.stats["launches"] = int(st["launches"])
        self.stats["bytes"] = int(st["bytes"])
        self.stats["set_tap_s"] = float(st["wrap_s"])          # whole native wrapper wall (geometry + tap set + tapped op + stage), measured
        self.stats["geometries"] = {str(k): int(v) for k, v in dict(st["geometries"]).items()}
        self.stats["compacted_launches"] = int(st.get("compacted_launches", 0))
        self.stats["compact"] = bool(st.get("compact", False))
        self.stats["refused"] = int(st.get("refused", 0))       # R13 fa2: launches expected but not bound (uncovered tile / kernel skip) -> errors
        self.stats["tap_layout"] = int(st.get("tap_layout", 7))
        # [R18 tap, F-r17-23] the bounded-tap record: cap, the one bounded allocation, peaks, split counts (all bytes, all measured)
        self.stats["bounded_tap"] = {
            "cap_bytes": int(self.tap_cap_bytes), "bound_alloc_bytes": int(st.get("bound_alloc_bytes", 0)),
            "bound_used_peak_bytes": int(st.get("bound_used_peak_bytes", 0)), "geometry_buffers_bytes": int(st.get("geometry_buffers_bytes", 0)),
            "tap_resident_bytes": int(st.get("tap_resident_bytes", 0)), "split_launches": int(st.get("split_launches", 0)),
            "split_groups": int(st.get("split_groups", 0)), "cap_exceeded_launches": int(st.get("cap_exceeded_launches", 0)),
            "cap_exceeded_peak_bytes": int(st.get("cap_exceeded_peak_bytes", 0)), "windows": int(self._tel["windows"]),
            "split_steps": sorted(self._tel["split_steps"]), "peak_group_bytes": int(self._tel["peak_group_bytes"]),
            "largest_unbounded_launch_bytes": int(self._tel["largest_unbounded_bytes"]),
            "torch_alloc_peak_in_tap": int(self._tel["torch_alloc_peak_in_tap"]), "torch_max_memory_allocated": int(self._tel["torch_max_alloc"]),
            "bounded": bool(self.tap_cap_bytes > 0 and int(st.get("cap_exceeded_launches", 0)) == 0),
        }

    def declaration_classes(self) -> list[dict]:
        try:
            self.sync_stats()
        except Exception as e:  # noqa: BLE001
            self.c.errors.append(f"fa2 tap stats: {e!r}"[:200])
        cls = super().declaration_classes()
        if cls and "bounded_tap" in self.stats:
            cls[0]["bounded_tap"] = self.stats["bounded_tap"]
        return cls

    def uninstall(self) -> None:
        super().uninstall()
        self.sync_stats()
        if self.telemetry_path:
            try:
                import json as _json
                from pathlib import Path as _P
                _P(self.telemetry_path).with_name("fa2_tap_bound.json").write_text(_json.dumps(self.stats["bounded_tap"], indent=1) + "\n")
            except Exception as e:  # noqa: BLE001
                self.c.errors.append(f"fa2 tap summary: {e!r}"[:200])
        if self._tel_fh is not None:
            try:
                self._tel_fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._tel_fh = None


class NativeHiddenM1SourceFA3(HiddenM1SourceFA3):
    """R13 hopper: the FA3 M1 tap glue as ONE C++ callable (`Fa3Wrap`, native_collect.cpp) in place of the Python `fwd` wrapper of
    `hidden_source.HiddenM1SourceFA3` -- same tapped kernel (`verity_fa3_matReq.so`), same geometry / tile rule, same names into the same
    collector; stats synced at uninstall."""

    def __init__(self, committer: "NativeCollectCommitter", so_path: str, ns: str = "verity_fa3_matReq") -> None:
        super().__init__(committer, so_path, ns)
        self._w = ext().Fa3Wrap(committer._nc, self.orig_op, self.tapped_op, committer._intern, ns)
        # P0-03's rule for the FA3 tap: zero the staged whole buffer after staging on multi-sequence launches (B=1 unchanged)
        self._w.zero_after = os.environ.get("VERITY_FA3_ZERO", os.environ.get("VERITY_FA2_ZERO", "1")).strip() not in ("0", "false", "no")
        self.short_name, self._w.ms = "hidden_m1_fa3", self.ms

    def _wrapper(self):
        return self._w

    def on_step_begin(self) -> None:
        self._w.on_step_begin()

    def sync_stats(self) -> None:
        st = self._w.stats()
        self.stats["launches"] = int(st["launches"])
        self.stats["bytes"] = int(st["bytes"])
        self.stats["set_tap_s"] = float(st["wrap_s"])
        self.stats["geometries"] = {str(k): int(v) for k, v in dict(st["geometries"]).items()}
        self.stats["tiles"] = {str(k): int(v) for k, v in dict(st["tiles"]).items()}
        self.stats["refused"] = int(st["refused"])
        self.stats["tap_layout"] = int(st["tap_layout"])
        self.stats["zero_after"] = bool(st.get("zero_after", True))

    def uninstall(self) -> None:
        super().uninstall()
        self.sync_stats()


WINDOW_MB, WINDOW_SLOTS = 256, 8


def window_mb_default(windowed_worker: bool, retain: str) -> int:
    """VERITY_WINDOW_MB when it is unset: a ring of WINDOW_MB windows on the windowed worker (native worker + gather) unless it retains
    on the device, whose footprint is the per-step blocks; per-step blocks (0) on every other collector."""
    return WINDOW_MB if windowed_worker and retain != "device" else 0


class NativeCollectCommitter(NativeHostCommitter):
    """Arm 3.  Inherits the whole R9 complete path; replaces the serving-thread acquisition (see module docstring).

    R11 `native_hooks=True` (collector v2, `native_collect_v2`): the module.forward wrappers are C++ callables (`HookFwd`) and the FA2
    M1 glue is `Fa2Wrap` -- no Python frame per producer; gather windows (P0/P1/P2 + step end) with 32 blocks per descriptor by default
    (size-adaptive, `auto_blocks`: 8 KiB a block, up to 8192)."""

    def __init__(self, engine: Any, *, wrap: bool = True, profile: bool = False, build_dir: str | None = None, check_sizes: bool = True,
                 gather: bool = False, native_hooks: bool = False, auto_blocks: bool = False, native_worker: bool = False,
                 layout: str = "chunk-leaf-v1", retain_levels: bool | None = None, commit_scheme: str | None = None, **kw: Any) -> None:
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
        # P0 (R13) windowed staging: VERITY_WINDOW_MB=<W> (0 = off; unset = `window_mb_default`) lays the packed step stream over a ring of
        # VERITY_WINDOW_SLOTS device windows and absorbs each window as soon as the serving thread has filled it (see native_collect.cpp
        # "WINDOWED staging ring")
        win_mb = os.environ.get("VERITY_WINDOW_MB", "").strip() or window_mb_default(self.native_worker and self.gather, self.retain)
        self.win_bytes = int(float(win_mb) * (1 << 20)) // 256 * 256
        self.win_slots_req = int(os.environ.get("VERITY_WINDOW_SLOTS", WINDOW_SLOTS))
        # rows-sha512: VERITY_ROWS_CAP_BLOCKS=<n> (0 = off) hashes a window in launches of at most n blocks while at least
        # VERITY_ROWS_CAP_FREE ring slots are free (the serving thread is not waiting for one), leaving the other SMs to serving
        self.rows_cap_blocks = int(os.environ.get("VERITY_ROWS_CAP_BLOCKS", 0))
        self.rows_cap_free = int(os.environ.get("VERITY_ROWS_CAP_FREE", 2))
        self.windowed = self.win_bytes > 0 and self.native_worker and self.gather
        if self.win_bytes > 0 and not self.windowed:
            raise ValueError("windowed staging needs native_collect_v2b (native worker + gather)")
        self.win_tensors: list[Any] = []
        self.win_scratch: Any | None = None
        if layout not in ("chunk-leaf-v1", "chunk-leaf-v2"):
            raise ValueError(f"unknown leaf layout {layout!r} (chunk-leaf-v1 | chunk-leaf-v2)")
        self.layout = layout
        self.layout_v2 = layout == "chunk-leaf-v2"
        if self.layout_v2 and not self.windowed:
            raise ValueError("chunk-leaf-v2 is implemented on the windowed path only (--window-mb > 0)")   # header M := 0 (coordinator decision)
        # `retain_levels=False`: no tree level of a step is kept (on the windowed path each window's subtree stays on the device, no host
        # copy and no host sync per window); the roots are the same, and an opening takes its value and path from a replay
        # (`open(values_from=, paths_from_replay=True)`), which `verify` holds to this run's roots.  Unset, a windowed run that keeps no
        # values (retain=none, whose openings are replayed anyway) keeps no levels; every other run keeps them
        self.retain_levels = bool(retain_levels) if retain_levels is not None else not (self.windowed and self.retain == "none")
        if not self.retain_levels and not self.windowed:
            raise ValueError("retain_levels=False is implemented on the windowed path only (--window-mb > 0)")
        # the commitment scheme (`commit_scheme`, unset = vllm-v1): vllm-v1 (the record), or rows-sha512 (`step_rows`: every
        # step_rows.ROW_BYTES (4 KiB) chunk an hm96-sha512 hiding leaf in a frame-v3-sha512 tree, 64-byte roots), on the windowed worker under chunk-leaf-v1.
        # Each step's salt key is fresh (`row_keys`); a replay that must reproduce a run's trees takes that run's keys
        # (`replay_row_keys`, set before the run and dropped by its finalize).  rows-blake3 (`step_rows_b3`) is the hash loop's research
        # arm: rows-sha512's step with keyed BLAKE3 in every hash position; it commits and roots, and refuses openings
        self.commit_scheme = commit_scheme or "vllm-v1"
        if self.commit_scheme not in ("vllm-v1", step_rows.NAME, step_rows_b3.NAME):
            raise ValueError(f"unknown commit scheme {self.commit_scheme!r} (vllm-v1 | {step_rows.NAME} | {step_rows_b3.NAME})")
        self.rows = self.commit_scheme in (step_rows.NAME, step_rows_b3.NAME)
        self.rows_b3 = self.commit_scheme == step_rows_b3.NAME
        self._rows_mod = step_rows_b3 if self.rows_b3 else step_rows
        if self.rows and (not self.windowed or self.layout_v2 or self.hiding is not None):
            raise ValueError(f"{step_rows.NAME} is implemented on the windowed worker (native worker + gather + --window-mb > 0) under "
                             "chunk-leaf-v1 position leaves only")
        if self.rows:
            self.chunk = step_rows.ROW_BYTES      # a row is a chunk of the step's stream: the layout's leaves, the windows' and the tree's
        self.row_keys: dict[int, bytes] = {}
        self.replay_row_keys: dict[int, bytes] | None = None
        self._row_tn: Any | None = None
        if self.rows_b3:
            self._row_consts = (step_rows_b3.row_prefix(8 * step_rows.ROW_BYTES), step_rows_b3.SALT_PREFIX, step_rows_b3.leaf_prefix())
        else:
            self._row_consts = (step_rows.row_prefix(8 * step_rows.ROW_BYTES), step_rows.INST.salt_prefix,
                                step_rows.INST.leaf_prefix(step_rows.KEY)) if self.rows else None
        # [R17 fp8, F-fp8-B8MEM] BOUNDED staging (VERITY_STAGING_BOUNDED; unset = on whenever staging is windowed): under chunk-leaf-v1 the
        # leaf header binds the step length M, which a LEARNING step (plan not yet known) only knows at step end -- the windowed path
        # therefore copied EVERY window of every step to pinned host (the learning re-hash source; the hit-plan "hedge" copy) and
        # returned none of it before the run's reset: the 3,545-token prefill of the B8 FP8 row is 138 GiB of `fa3_hidden_m1` host
        # copies against a 239 GB cgroup (two OOM-kills, R16).  Bounded:
        #   * a learning step's transient copies are capped by VERITY_LEARN_HOST_BUDGET_MB and given back to the pool right after the
        #     re-hash; over budget, a LEARN-ONLY pass (`learn_only`, the warm-up instrumented run) records the plan and discards the
        #     windows (no copy, no hash; placeholder commitment, never a record), a record run fails CLOSED by name;
        #   * a hit-plan step takes NO hedge copy: a plan mismatch is a named collector error (the plan of a prior identical step
        #     is the prescribed layout; an unexpected shape is a finding, not something to absorb silently);
        #   * retain=host + VERITY_RETAIN_EXCLUDE=<family,...> keeps every chunk EXCEPT the chunk-aligned interior of the excluded
        #     families (`_HostRanges`): those leaves open by replay against the original root (R17-5), all others from host (C2's
        #     module outputs stay on host).
        self.bounded = (os.environ.get("VERITY_STAGING_BOUNDED", "").strip() or ("1" if self.windowed else "0")) in ("1", "true", "yes")
        self.learn_only = False                                  # set by the harness around the warm-up instrumented run
        self.learn_budget_bytes = int(float(os.environ.get("VERITY_LEARN_HOST_BUDGET_MB", "4096")) * (1 << 20))
        self.retain_exclude: set[str] = retain_exclude_families(os.environ.get("VERITY_RETAIN_EXCLUDE", ""))
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
        self.hi_prio = bool(native_hooks) and os.environ.get("VERITY_HI_PRIO", "1") == "1"
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
            self.hidden_source_cls_fa3 = NativeHiddenM1SourceFA3      # R13 hopper: chosen by commit_delta when the tap is an FA3 (v9) build
        else:
            self.name = "native_collect" + ("_gather" if self.gather else "") + ("_wrap" if self.wrap else "_hook") + ("_prof" if self.profile else "")
        self._nc = ext(build_dir).Collector()
        self._nc.check_sizes = bool(check_sizes)
        self._nc.use_gather = self.gather
        if self.gather:
            self._nc.enable_gather(65536)
            # thread blocks per staged descriptor in the gather kernel (R10 default 4; 32 = R10's labelled-extra Llama geometry, the R11 default).
            # Size-adaptive, it is only a cap (was 1024): a 256 MB prefill window holds a handful of descriptors, and at 32 blocks each the copy
            # filled a fraction of the SMs (B32 compiled prefill gather 12.2 -> 8.3 ms/step, r20261017-205629-7b49 vs r20261017-205439-5d0c).
            # 8 KiB a block, cap 8192: a launch is several waves of short blocks, so SMs that share no hash block take more of them; one wave
            # of 32 KiB blocks ended with its slowest block, on an SM beside a SHA block (H200 prefill gather 475 GB/s, 1.6 TB/s alone)
            self._nc.blocks_per_desc = int(os.environ.get("NATIVE_GATHER_BLOCKS", "8192" if self.auto_blocks else "32" if self.native_hooks else "4"))
            # R11 candidate B: size-adaptive blocks per window (cap = blocks_per_desc; ceil(max staged bytes / auto_bytes_per_block) otherwise)
            self._nc.auto_blocks = self.auto_blocks
            self._nc.auto_bytes_per_block = int(os.environ.get("NATIVE_GATHER_AUTO_BYTES", "8192"))
        self._names: list[str] = []
        self._name_ids: dict[str, int] = {}
        self._metas_cache = PlanCache()                          # (plan key, version) -> PlanMetas, for this run's and the previous run's plans
        self._wrapped: list[Any] = []
        self._plan_key = 0
        self._py_committed = 0
        self._prof_s = 0.0
        self.consumer_delay_s = float(os.environ.get("VERITY_CONSUMER_DELAY_MS", "0")) / 1e3
        self.fault = os.environ.get("VERITY_FAULT", "")
        if self.fault.startswith("skip_copy:"):
            self._nc.fault_skip_copy_at = int(self.fault.split(":", 1)[1])
        if self.fault or self.consumer_delay_s:
            self.name = f"{self.name}-STRESS[{self.fault or ''}{'+delay%gms' % (self.consumer_delay_s * 1e3) if self.consumer_delay_s else ''}]"
        if self.rows:
            self.name = f"{self.name}+{self.commit_scheme}"
        self.extra_stats: dict[str, Any] = {}

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
            # R15 `compiled`: the module boundaries are acquired by record ops inside the compiled graphs (compiled_source.CompiledGraphSource,
            # installed before vLLM compiled the model); wrapping the forwards here would never be traced and never run under graph replay
            pass
        elif self.wrap:
            for name, m in self._modules:
                pre = self._acquires_input(name, m)                     # by class (the embedding's ids) or [R19 int-20] by the manifest (a promoted member's consumer)
                if "forward" in m.__dict__:
                    raise RuntimeError(f"{name}: forward already wrapped")
                self._hooks.patch(m, "forward", self._make_native_wrapper(m.forward, name, pre) if (self.native_hooks and not self.fault)
                                  else self._make_wrapper(m.forward, name, pre))
                self._wrapped.append(m)
            for name, m in self._input_only_modules():                  # [R19 int-20] a consumer of no acquire class: input acquired, output not
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
            # step-shape class -> plan key.  [R17 fp8, F-fp8-PLANKEY] (ntok, nreq) alone is NOT the layout: the FA3 hidden M1 export plane is
            # sized by every request's KV length, so two decode steps with the same (ntok, nreq) differ in bytes (B8 mixed 1024/128 steps
            # 35 and 80: 78,892,288 vs 87,297,280 -- the bounded record run failed closed by name, 01:28Z).  The key binds the per-request
            # geometry (num_computed_tokens, num_scheduled_tokens) as a multiset; on the unbounded windowed path a miss still self-heals.
            me._plan_key, me._plan_key_mode = plan_key_of(scheduler_output, int(ntok), int(nreq))
            me._plan_shape = (int(ntok), int(nreq))
            if me.bounded and me._plan_key_mode != "geometry":
                # fail closed at the FIRST step, not mid-run: without the per-request geometry a hit-plan step can meet another layout
                raise RuntimeError("bounded staging needs the geometry plan key (per-request num_computed_tokens / num_scheduled_tokens); "
                                   f"this scheduler output ({type(scheduler_output).__name__}) does not expose them")
            me._cur_sched = scheduler_output                         # R14 BIND-01: the step's scheduled tokens per request (row map, read at sample)
            me._step_min_bytes = me.step_min_bytes(int(ntok), int(nreq))   # X-02: pick a block big enough for THIS step (prefill vs decode)
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
                try:
                    bs = res[0] if isinstance(res, tuple) else res
                    if bs is not None and getattr(bs, "req_ids", None) is not None:
                        me._cur_batch = bs
                except Exception:  # noqa: BLE001
                    pass
                return res

            self._hooks.patch(runner, "gather_batch_req_state", gather_batch_req_state)
        if "sample" in self._orig:
            orig_sample = self._orig["sample"]
            # [R16 stoch REQ-03] the top-p split count S of this step's sampling (manifest family `sampler_splits`, identity
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
                        if isinstance(st, torch.Tensor) and st.device.type == "cuda":
                            me._acquire("runner.sampler/sampled_token_ids", st, job)
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
                        # [R15 stoch] the per-sequence SEED the sampler consumed this step (manifest family `sampler_seed`, identity
                        # `runner.sampler/seed`, one i64 per sequence): `sampling_states.seeds.gpu[input_batch.idx_mapping]` -- the very buffer
                        # `_gumbel_sample_kernel` reads (`seeds_ptr + req_state_idx`), gathered in the step's row order.  Bound only when the
                        # batch carries explicit seeds (a stochastic Program's registered runtime input); a greedy / unseeded step binds nothing.
                        ib = a[1] if len(a) > 1 else kw.get("input_batch")
                        ss = getattr(getattr(runner, "sampler", None), "sampling_states", None)
                        if ib is not None and ss is not None and hasattr(ss, "seeds") and hasattr(ib, "idx_mapping"):
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
        uninstall_split_kernel_proxies(getattr(self, "_split_orig", None) or {})     # [R16 stoch REQ-03]
        self._split_orig = {}

    def _flush_leaf(self, name: str) -> bool:
        """Legal collection points of the gather design (owner 2's P0/P1/P2).  Negative test `VERITY_FAULT=late_flush`: the qkv_proj flush
        is dropped, so the staged qkv output is read at the o_proj hook -- AFTER RoPE rotated q/k in place (zero-hook window) -> the run
        root MUST differ from arm 2 (late-read regression; acq-life lifetimes.md: qkv_proj out is overwritten at hook+1)."""
        leaf = name.rsplit(".", 1)[-1]
        if self.fault == "late_flush" and leaf == "qkv_proj":
            return False
        if self.fault == "late_flush_qk" and leaf in ("q_norm", "k_norm"):       # [FLUSH-02 negative] the QK-norm outputs meet RoPE before the o_proj flush
            return False
        return leaf in GATHER_FLUSH_LEAVES

    def _pre_flush_leaf(self, name: str) -> bool:
        """[R15 moe FLUSH-02] flush BEFORE this module runs: the fused add-RMSNorm (`input_layernorm`, final `norm`) overwrites its `x`
        (the previous block's output) and `residual` (the previous `post_attention_layernorm/1`) in place, and a fused-MoE block has no
        down_proj hook to have flushed them.  Negative `VERITY_FAULT=late_flush_norm`: the pre-flush is dropped -> on an MoE model the
        committed `post_attention_layernorm/1` is the NEXT layer's residual (the r15-int-3 OLMoE record, ALERT M-0479)."""
        leaf = name.rsplit(".", 1)[-1]
        if self.fault == "late_flush_norm":
            return False
        return leaf in PRE_FLUSH_LEAVES

    def _flush_plane(self, what: str) -> bool:
        """[R16 moe FLUSH-03] does the staged fused-MoE plane `what` leave the device NOW (before the block's next launch)?  `up` lives in
        the workspace the block's own w2 launch overwrites (cache1/cache3 alias); the other planes are overwritten only by the next layer,
        after its `input_layernorm` pre-flush.  Negative `VERITY_FAULT=late_flush_moe_up`: the flush is dropped -> the committed `up` is
        the same layer's `down` (the pre-FLUSH-03 record)."""
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
            nc.flush()                       # [FLUSH-02] the in-place consumer is about to run: everything staged by the previous block leaves now
            return fwd(*a, **kw)
        return fwd_pre_flush

    def _make_native_wrapper(self, orig, name: str, pre: bool):
        """R11 v2: the hook body is `HookFwd` (C++): one pybind dispatch per producer call, CPython type checks, name ids bound here."""
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
                    collect(nids_k[0], out[0])       # F-rev-01 twin (R14 negative): member /1 dropped, member /0 bound twice -> same count
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
        """[R19 int-20] the INPUT-ONLY hook body of a consumer module the required manifest names for a promoted member and the committer
        did not select (no acquire class): its first positional tensor argument goes to the collector as `<name>/in` (`_acquire`: X-03
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

    # ------------------------------------------------------------------ acquisition entry used by R9 hooks / the FA2 M1 source / the sampler

    def _acquire(self, name: str, t: Any, job: _StepJob) -> None:
        nid = self._name_ids.get(name)
        if nid is None:
            nid = self._intern(name)
        self._nc.note_expected(1)                                  # X-03: attempted acquisition (committed is counted by put)
        fn = self._nc.stage if self.gather else self._nc.collect
        if self.profile:
            t0 = time.perf_counter()
            fn(nid, t)
            self._prof_s += time.perf_counter() - t0
        else:
            fn(nid, t)
        self._py_committed += 1

    def slot_plan(self, names: list[str], slots: list[Any], kinds: list[int]) -> int:
        """Register a replayed graph's slots once (kind 0: the whole slot, 1: its first n rows, 2: its first pad rows); the handle's
        `stage_slot_plan(handle, n, pad)` acquires them in this order in one native call, the same bytes and plan entries as `_acquire`
        of each view.  A handle holds its slots until `clear_slot_plans`."""
        return int(self._nc.add_slot_plan([self._intern(nm) for nm in names], list(slots), [int(k) for k in kinds]))

    def stage_slot_plan(self, handle: int, n: int, pad: int) -> int:
        return int(self._nc.stage_slot_plan(int(handle), int(n), int(pad)))

    def clear_slot_plans(self) -> None:
        self._nc.clear_slot_plans()

    # ------------------------------------------------------------------ step boundaries (serving thread)

    def _record_step_rows(self, job: _StepJob, defer: bool = False) -> None:
        """R14 BIND-01 (B>1 binding): the engine's row order of this step -- `InputBatch.req_ids` (the persistent batch after the step's
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

    def _begin_step(self) -> None:
        job = _StepJob(self._step_index, [], None)
        self._cur = job
        self._step_index += 1
        for src in self.sources:
            src.on_step_begin()
        if self.windowed:
            job.arena = None
            job.cursor = 0
            job.arena_ptr = 0
            job.windowed = True                                                 # type: ignore[attr-defined]
            job.done_evt = threading.Event()                                    # type: ignore[attr-defined]
            job.plan_used = int(self._nc.plan_used(self._plan_key))             # type: ignore[attr-defined]  (-1: learning step, size unknown)
            # [R17 fp8] the metas of the plan this hit step is predicted to follow (None on a learning step): the worker needs them
            # BEFORE step end to know which chunks of a window belong to an excluded family (retain=host + exclude)
            job.plan_hint = (self._plan_key, int(self._nc.plan_version(self._plan_key))) if job.plan_used >= 0 else None  # type: ignore[attr-defined]
            job.hint_entries = self._nc.export_plan(self._plan_key) if self.retain_exclude and job.plan_hint and job.plan_hint not in self._metas_cache else None  # type: ignore[attr-defined]
            job.plan_shape = getattr(self, "_plan_shape", (0, 0))                # type: ignore[attr-defined]  (ntok, nreq) for messages/records
            self._prof_s = 0.0
            self._nc.begin_step(self._plan_key, 0, 0, True, job.step)
            self._q.put(job)                                                    # the worker absorbs windows WHILE the step runs
        else:
            job.arena = self.arena.take(int(getattr(self, "_step_min_bytes", 0) or 0))
            job.cursor = 0
            job.arena_ptr = job.arena.data_ptr()
            memtrace("serving: step begin (block taken)", job.step)
            self._prof_s = 0.0
            self._nc.begin_step(self._plan_key, job.arena_ptr, int(job.arena.numel()), True)
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
        windowed = getattr(job, "windowed", False)
        try:
            # R14 BIND-01: the step's request row order (InputBatch after this step's _update_states; before the next)
            self._record_step_rows(job, defer=windowed and self.rows_off_thread and not self._rows_at_step_end)
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"step rows: {e!r}"[:200])
        t0 = time.perf_counter()
        for src in self.sources:
            fn = getattr(src, "on_step_end", None)
            if fn is not None:
                try:
                    fn(job)                  # R15 `compiled`: drain the graph-written staging ring into this step (before the final flush)
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
                self._nc.progress()                 # F-stoch-03 dead-watchdog liveness: the serving thread still ends steps (fail-closed run in flight)
            if self.native_hooks:
                errs = self._nc.drain_errors()
                if errs:
                    self.errors.extend(str(e)[:300] for e in errs)
            st = self._nc.step_stats()
            nbytes, issue_s, waits, wait_s = st["bytes"], st["issue_s"], st.get("window_waits"), st.get("window_wait_s")
        job.cursor = int(used)
        key = self._plan_key
        # a learning step, or a hit on a plan the worker does not hold (not yet absorbed, or its step failed: plan_cache)
        entries = self._nc.export_plan(key) if changed or (key, int(version)) not in self._metas_cache else None
        job.plan = (key, int(version), entries)                         # type: ignore[attr-defined]
        job.acq_bytes = int(nbytes)
        if not (windowed and self.native_end):
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
            # [coord R17-16 (i)] the host side of the curve beside the device side: the pinned pool (total / in use) and this process's RSS
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
        if windowed:
            job.done_evt.set()                                                  # type: ignore[attr-defined]  (queued at step begin)
        else:
            self._q.put(job)

    # ------------------------------------------------------------------ worker side: metas from the plan (cached per plan version)

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
                return self._commit_step_windowed(job)
            except Exception as e:  # noqa: BLE001
                self._nc.abandon_step(job.step)            # free the step's slots: the serving thread must not wait forever for this step
                print(f"[collector] worker failed on windowed step {job.step}: {e!r}"[:400], file=sys.stderr, flush=True)
                raise
            finally:
                if getattr(self._nc, "win_dead", False):
                    self._nc.progress()                    # F-stoch-03 dead-watchdog liveness: the worker finished (or failed by name) a step
        if self.rows:
            raise RuntimeError(f"{step_rows.NAME}: step {job.step} was not staged through the window ring (enable_windows before the run)")
        job.metas = self._metas_for(job)
        if self.fault == "release_before_keep":
            return self._commit_step_gpu_faulty(job)
        if self.native_worker and job.arena is not None and not job.overflow:
            return self._commit_step_native(job)
        return super()._commit_step(job)

    # ------------------------------------------------------------------ P0 (R13): windowed absorb / release
    def enable_windows(self, rec: dict[str, Any] | None = None) -> dict[str, Any]:
        """Allocate the device window ring (K slots x W) + one scratch window and switch the collector to windowed staging.  K = the requested
        slot count (>= 3): the ring is INDEPENDENT of the largest tensor -- `push_desc` acquires slots per window segment and, when the ring
        is exhausted, flushes the segments pushed so far and waits for the worker to hand a slot back, so a tensor larger than the whole
        ring (FA2 M1 of a long prefill) streams through it.  (Until 09:20Z K was max(requested, ceil(largest tensor bound / W) + 2); with
        the FA2-02 geometry bound that made B=32 a 52-slot / 14 GB ring and let the engine run 32 decode steps ahead of the worker.)  The
        largest-tensor bound (prefill hidden bound / layers) is still reported for the record."""
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
        # [R15 capacity] BOUNDED staging with BACKPRESSURE, never a drop: when the ring is exhausted the serving thread blocks until the worker
        # hands a slot back; only after `win_wait_timeout_s` (default 600 s) does the acquisition FAIL -- named ("windowed staging: waited N s for a
        # free window slot ... the worker is not releasing windows"), recorded as a collector error, so X-03 turns the run into COMMIT FAIL
        # (never a warning).  VERITY_WIN_WAIT_TIMEOUT_S shortens it for the forced-stall negative (test_commit_fail_closed::test_e2e_windowed_stall_forces_commit_fail).
        wto = os.environ.get("VERITY_WIN_WAIT_TIMEOUT_S")
        if wto:
            self._nc.win_wait_timeout_s = float(wto)
        # [R16 sys, F-stoch-03 M-0646] fail-FAST after the fail-closed name: the timeout above throws from the serving thread, but that throw
        # needs the GIL back and a Python frame that records the failure; on cp-dev1 (09:18Z) the process instead sat 13+ min at 0 % GPU with
        # every thread in futex_wait.  So the C++ collector arms a GIL-free watchdog thread when it declares itself dead: unless the Python side
        # side keeps making progress (a step ends / the worker finishes a step -> progress()) or acknowledges (finalize, after the worker
        # join -> ack_dead: a run that fails closed normally is never touched) it stays quiet; VERITY_COLLECT_DEAD_GRACE_S (default 120 s)
        # without either and it writes <out>/COLLECTOR_DEAD_HUNG.json (path from VERITY_COLLECT_WATCHDOG_MARKER, set by commit_delta) and
        # _exit(3)s -- rc 3 is the "commit_delta refused/failed closed" code row_pod already reports as COMMIT FAIL.  Never a silent hang.
        grace = float(os.environ.get("VERITY_COLLECT_DEAD_GRACE_S", "120") or 120)
        marker = os.environ.get("VERITY_COLLECT_WATCHDOG_MARKER", "")
        if hasattr(self._nc, "set_watchdog"):
            self._nc.set_watchdog(marker, grace)
        info = {"windowed": True, "window_bytes": W, "slots": K, "ring_bytes": K * W, "scratch_bytes": W, "largest_tensor_bound": largest,
                "backpressure": "block-until-slot-free (no drop)", "slot_wait_timeout_s": float(self._nc.win_wait_timeout_s),
                "dead_watchdog": {"grace_s": grace, "marker": marker or None, "rc": 3},
                "layout": "chunk-leaf-v2" if self.layout_v2 else "chunk-leaf-v1",
                "in_place_hashing": "all steps (v2)" if self.layout_v2 else
                ("hit-plan steps whole; learning steps keeping no bytes to each row's h in place, leaves from h at step end; others re-upload from host"
                 if self.rows else "hit-plan steps only (v1 binds the step length into every leaf); learning steps re-upload from host")}
        memtrace("window ring allocated")
        self.window_info = info
        return info

    # ------------------------------------------------------------------ rows-sha512 (step_rows)
    def _window_levels(self, n: int, j: int) -> tuple[list[int], int]:
        """Node counts of a step's levels 0 .. min(j, height) for n leaves, and the digest size."""
        if self.rows:
            width = 1 << merkle_depth(n)
            return [width >> L for L in range(min(j, width.bit_length() - 1) + 1)], step_rows.ROOT_BYTES
        return level_sizes(n, j), 32

    def _row_key(self, step: int) -> bytes:
        """Step `step`'s salt key: fresh from the OS generator, or the replayed run's."""
        if self.replay_row_keys is None:
            ks = os.urandom(self._rows_mod.KEY_BYTES)
        elif step in self.replay_row_keys:
            ks = self.replay_row_keys[step]
        else:
            raise RuntimeError(f"{step_rows.NAME} replay: the replayed run has no salt key for step {step}")
        self.row_keys[step] = ks
        return ks

    def _row_table(self) -> Any:
        if self._row_tn is None:
            table = step_rows_b3.t_table() if self.rows_b3 else step_rows.t_nibbles()
            self._row_tn = torch.frombuffer(bytearray(table), dtype=torch.uint8).cuda()
        return self._row_tn

    def _row_domain(self, step: int, n: int) -> Any:
        return step_rows.domain(self.program_digest, self._ctx_digest(step), self._geo_digest, n)

    def _row_step(self, step: int, n: int, ks: bytes) -> Any:
        row_prefix, salt_prefix, leaf_prefix = self._row_consts
        dom = self._row_domain(step, n)
        if self.rows_b3:
            return ext().RowStep(step_rows_b3.did(dom), ks, row_prefix, salt_prefix, leaf_prefix, step_rows_b3.SCHEMA, step_rows.ROW_BYTES, hash="blake3",
                                 hash_key=step_rows_b3.HASH_KEY)
        return ext().RowStep(dom.domain_id, ks, row_prefix, salt_prefix, leaf_prefix, step_rows.SCHEMA, step_rows.ROW_BYTES)

    def _level_path(self, levels: list, index: int) -> tuple:
        if not self.rows:
            return super()._level_path(levels, index)
        return tuple(bytes(levels[d][(index >> d) ^ 1].cpu().numpy().tobytes()) for d in range(len(levels) - 1))

    def _gpu_salt(self, step: int, index: int) -> bytes:
        return self._rows_mod.salt(self.row_keys[step], index) if self.rows else b""

    def _fold_run_root(self, roots: list[bytes]) -> bytes | None:
        if not self.rows:
            return super()._fold_run_root(roots)
        return step_rows.run_root(self.program_digest, self._geo_digest, roots) if all(len(r) == step_rows.ROOT_BYTES for r in roots) else None

    def verify(self, run: Any, opening: Any) -> bool:
        """rows-sha512: the row and its salt give the hiding leaf, the frame path folds it to the tree root, which binds into the step
        root, which is in the run root."""
        if not self.rows:
            return super().verify(run, opening)
        if self.rows_b3:
            raise NotImplementedError(f"verify: {step_rows_b3.NAME} is a research arm; it is never opened")
        try:
            sc = run.steps[opening.step]
            ident = self.identity_of(opening.step, opening.index)
        except (IndexError, KeyError):
            return False
        if opening.step not in self._gpu_blocks or ident != opening.identity or len(opening.value) != step_rows.ROW_BYTES:
            return False
        if len(opening.salt) != step_rows.SALT_BYTES or any(not isinstance(p, bytes) for p in opening.path):
            return False
        h = step_rows.hiding_leaf(opening.value, opening.salt, 8 * step_rows.ROW_BYTES)[1]
        merkle = step_rows.fold_path(self._row_domain(opening.step, sc.n_leaves), opening.index, h, opening.path)
        if merkle is None:
            return False
        root = step_rows.step_root(self.program_digest, self._ctx_digest(opening.step), self._geo_digest,
                                   self.layout_digest(self._layouts[opening.step]), sc.n_leaves, merkle)
        return root == sc.root and len(run.run_root) == step_rows.ROOT_BYTES and self._run_root(run) == run.run_root

    def _refuse_padding(self, padding: list) -> None:
        if padding and self.rows:
            raise ValueError(f"padding steps are vllm-v1 position-leaf blocks: not implemented under {step_rows.NAME}")
        super()._refuse_padding(padding)

    def open_range(self, step: int, lo: int, hi: int) -> Any:
        if self.rows:
            raise NotImplementedError(f"open_range: range openings are vllm-v1's; not implemented under {step_rows.NAME}")
        return super().open_range(step, lo, hi)

    def verify_range(self, run: Any, ro: Any, roots: dict[int, bytes] | None = None) -> bool:
        if self.rows:
            raise NotImplementedError(f"verify_range: range openings are vllm-v1's; not implemented under {step_rows.NAME}")
        return super().verify_range(run, ro, roots)

    def _leaf_record(self, steps: list) -> dict:
        rec = super()._leaf_record(steps)
        return {**rec, "commit_scheme": self.commit_scheme} if self.rows else rec

    def _window_root(self, step: int, state: dict[str, Any], padded: int, side: Any) -> tuple[list[Any], bytes, float]:
        """The step's tree levels, Merkle root and tree ms from its absorbed windows (`state`, cleared): the upper tree over the per-window
        top nodes (level n_levels-1 of the step), its levels to pinned host too when the step keeps levels."""
        H, rows = self._hasher, self.rows
        if not padded and rows:
            return [], self._row_domain(step, 0).empty_root(), 0.0
        if not padded:
            with torch.inference_mode(), torch.cuda.stream(side):
                lv_all = torch.zeros((0, 8), dtype=torch.int32, device="cuda")
            return H.tree(lv_all, side.cuda_stream, True, torch.empty(0, dtype=torch.uint8))
        top_span = 1 << (state["n_levels"] - 1)
        n_tops = state["width"] // top_span if rows else -(-state["n_total"] // top_span)
        if rows and state.get("leaves"):
            # the step's leaves, kept on the device by absorb: its window levels in one pass of wide launches, not one subtree a window
            with torch.inference_mode(), torch.cuda.stream(side):
                lv = torch.cat(state["leaves"]) if len(state["leaves"]) > 1 else state["leaves"][0]
                state["tops"] = [H.rows_tree_window(state["rs"], lv, state["n_levels"], 0, side.cuda_stream, [])]
            del lv
            state["leaves"] = []
        if sum(int(t.shape[0]) for t in state["tops"]) != n_tops:
            raise RuntimeError(f"windowed absorb: {len(state['tops'])} window subtrees for {state['n_total']} leaves (expected {n_tops})")
        with torch.inference_mode(), torch.cuda.stream(side):
            tops = torch.cat(state["tops"]) if len(state["tops"]) > 1 else state["tops"][0]
        if rows:
            keep_lv = state["views"] is not None
            upper_dst = self.pool.take((2 * n_tops - 1) * 64) if keep_lv and self.retain_pinned else torch.empty(0, dtype=torch.uint8)
            upper, mr, tree_ms = H.rows_tree(state["rs"], tops, state["n_levels"] - 1, side.cuda_stream, keep_lv, upper_dst)
            levels = list(state["views"][:state["n_levels"] - 1]) + list(upper) if keep_lv else []
        elif state["views"] is None:
            _upper, mr, tree_ms = H.tree(tops, side.cuda_stream, False, torch.empty(0, dtype=torch.uint8))   # its event sync orders every window's subtree
            levels = []
        else:
            tot, n = 0, int(tops.shape[0])
            while True:
                tot += n
                if n <= 1:
                    break
                n = (n + 1) // 2
            upper_dst = self.pool.take(tot * 32) if self.retain_pinned else torch.empty(0, dtype=torch.uint8)
            upper, mr, tree_ms = H.tree(tops, side.cuda_stream, True, upper_dst)
            levels = list(state["views"][:state["n_levels"] - 1]) + list(upper)
        del tops
        state.clear()
        return levels, mr, tree_ms

    def _commit_step_windowed(self, job: _StepJob) -> None:
        """Worker: absorb the step WINDOW BY WINDOW while the serving thread is still filling later windows.  Per window: wait for the
        serving thread's complete-event, D2H the raw bytes to pinned host (retention / re-upload source), hash its leaves in place when the
        step's padded length is already known (hit plan, or layout v2), hand the slot back.  After the step ended: (v1 learning step) re-upload
        the host windows one at a time through the scratch window and hash them with the now-known length; tree over all leaves; bind the
        root.  The logical stream, chunk indices, headers and tree are exactly those of the single-block path -> identical roots."""
        from verity_vllm.commit import scheme
        from verity_vllm.acquire.committer_api import StepCommitment

        self._gpu_ext_load()
        side = self._side_stream()
        nc, H, W, C = self._nc, self._hasher, self.win_bytes, self.chunk
        if getattr(nc, "win_dead", False):
            # [R15 capacity] the serving thread's slot wait timed out earlier in this run: every stage() since was refused by name, so this step
            # has nothing staged -- fail it by the SAME name instead of tripping over an empty tree ("cudaMemcpyAsync(root): invalid argument").
            raise RuntimeError("windowed staging: collector dead after a slot-wait timeout earlier in this run (step not committed)")
        t0 = time.perf_counter()
        plan_used = int(getattr(job, "plan_used", -1))
        inplace = self.layout_v2 or plan_used >= 0
        m_words = 0 if self.layout_v2 else (-(-plan_used // C) * C) // 4
        host_wins: list[tuple[int, Any, int]] = []                              # (k, pinned host view, bytes)
        pending: list[tuple[int, Any, Any]] = []                                # v2 learning steps: (k, host leaves, pool block) until the length is known
        transient: list[int] = []                                               # windows on host only for the v1 re-hash (retain none|sample)
        j = self.win_log2_chunks                                                # W = C * 2^j: window k holds leaves [k 2^j, (k+1) 2^j)
        state: dict[str, Any] = {}                                              # per-step subtree state (reset when the plan length was wrong)
        rows = self.rows
        ks = self._row_key(job.step) if rows else b""
        tn = self._row_table() if rows else None

        def begin_tree(n_total: int) -> None:
            """Pinned host buffers for levels 0..n_levels-1 of the WHOLE step (n_L = ceil(n / 2^L); rows-sha512: the tree's width, a power
            of two, over 2^L); the per-window subtrees are copied straight into their slices (no host concat).  n_levels = min(j, height)
            + 1: below the window level the tree is a disjoint union of per-window subtrees; the upper tree is built over the per-window
            top nodes by `H.tree` afterwards."""
            sizes, dig = self._window_levels(n_total, j)
            n_levels = len(sizes)
            if rows:
                state.update(rs=self._row_step(job.step, n_total, ks), width=sizes[0], absorbed=[], leaves=[])
            if not self.retain_levels:
                state.update(n_total=n_total, n_levels=n_levels, views=None, tops=[])
                return
            buf = self.pool.take(sum(sizes) * dig) if self.retain_pinned else torch.empty(sum(sizes) * dig, dtype=torch.uint8)
            views, off = [], 0
            for nL in sizes:
                v = buf[off:off + nL * dig]
                views.append(v.view(nL, dig) if rows else v.view(torch.int32).view(nL, 8))
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
            if rows:
                state["absorbed"].append(k)
                if views is None:
                    # no levels kept: the window's leaves (1/64 of its bytes) wait on the device for the step's tree (_window_root); a
                    # window's own subtree is two launches of a few blocks each, bound by latency and by waiting for SMs
                    state["leaves"].append(lv_k)
                    return
                dsts = [views[L][k << (j - L):(k << (j - L)) + (n_k >> L)] for L in range(n_levels)]
                state["tops"].append(H.rows_tree_window(state["rs"], lv_k, n_levels, k << j, side.cuda_stream, dsts))
                return
            if views is None:
                state["tops"].append(H.tree_window_top(lv_k, n_levels, side.cuda_stream))
                return
            dsts = []
            for L in range(n_levels):
                s0 = k << (j - L)
                dsts.append(views[L][s0:s0 + -(-n_k // (1 << L))])
            state["tops"].append(H.tree_window(lv_k, n_levels, side.cuda_stream, dsts))

        # host copy of the raw window: retention (retain=host) and/or the re-upload source for v1 learning steps; with layout v2 and
        # retain=none nothing leaves the device -- the window is hashed in place and released (openings by replay, p0b's path)
        keep_set = self.retain_windows if self.retain == "sample" else None     # retain=sample: only these (step, k) windows are kept
        # [R17 fp8] BOUNDED staging (see __init__): v1 learning steps copy windows to TRANSIENT pool blocks under a budget (given back after
        # the re-hash), or discard them on a learn-only pass; hit steps take no hedge copy; retain=host + exclude keeps byte RANGES.
        bounded = self.bounded and not self.layout_v2
        learning = plan_used < 0
        learn_only = bounded and learning and self.learn_only
        # rows on a learning step that keeps no bytes: a leaf's h needs only the step's salt key, so each window is hashed in place to its
        # h (1/64 of its bytes, kept on the device) and released with no host copy; its leaves follow from them once the step's length,
        # and so its frame domain, is known (rows_frame) -- the same leaves as the re-upload below, which only the other steps take.  With
        # no host copy there is no transient budget to exceed, so a bounded learn-only pass commits such a step instead of discarding it
        hfirst = rows and learning and self.retain == "none"
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
            ok, used_k, slot = nc.wait_window(job.step, k, side.cuda_stream)
            idle += time.perf_counter() - tw
            if not ok:
                if int(used_k) < 0 or getattr(nc, "win_dead", False):
                    # [R16 sys, F-stoch-03] the serving thread's slot wait timed out mid-step: the remaining windows will never be marked, so
                    # fail THIS step by the same name now (pin M-0662 (2): until R16 the worker spun here until COMMIT_TIMEOUT).
                    raise RuntimeError(f"windowed staging: collector dead (slot-wait timeout) while absorbing step {job.step} window {k} -- step not committed")
                break
            used_k = int(used_k)
            padded_k = -(-used_k // C) * C
            st = self.win_tensors[int(slot)]
            # [R15 capacity] lifetime NEGATIVE on the windowed path (the rc3 default; the arena-path fault below never runs here): the slot
            # is handed back to the ring BEFORE the retained copy is taken and the copy is read only after the serving thread refilled the
            # slot -> the tree (hashed in place from the right bytes) is correct, the retained copy holds a later window's bytes ->
            # openings / openings-after-release MUST fail against the (correct) root: exit 3 named, never PASS.  In-place steps only.
            fault_lifetime = self.fault == "release_before_keep" and inplace and padded_k > 0
            with torch.inference_mode(), torch.cuda.stream(side):
                if padded_k > used_k:
                    st[used_k:padded_k].zero_()
                hv = None
                keep_win = self.retain == "host" or (keep_set is not None and (job.step, k) in keep_set)
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
                            # replay-to-open (R17-5): keep ONLY the challenged chunks of this window (`chunk` bytes each) as byte ranges -- not the
                            # 256 MiB window; the replay's authentication paths come from its levels, so this is all a replay retains
                            base = k * W
                            for a, b in chunk_parts(job.step, base, base + padded_k):
                                rv = self.pool.take(b - a)
                                rv.copy_(st[a - base:b - base], non_blocking=True)
                                ranges.append((a, b, rv))
                        else:
                            hv = self.pool.take(padded_k)
                            hv.copy_(st[:padded_k], non_blocking=True)
                elif (keep_win or not self.layout_v2) and not fault_lifetime:   # v1 always needs the window for the learning-step re-hash
                    hv = self.pool.take(padded_k)
                    hv.copy_(st[:padded_k], non_blocking=True)                  # D2H pinned, side stream
                # in-place hashing only while the step stays inside the plan's length: a window running past it means the plan changed
                # mid-step (learning) -> the length in every header is wrong -> the step is re-hashed from the host windows below
                if inplace and padded_k and state and (k << j) + padded_k // C <= state["n_total"]:
                    if rows:
                        lv_k = row_leaves(st, padded_k, k)
                    else:
                        lv_k = H.leaves_window(st, padded_k, (k * W) // C, job.step, self.GPU_TREE_SRC, C, m_words, side.cuda_stream)
                    absorb(k, lv_k)                                             # syncs the side stream: the slot's reads are done here
                    del lv_k
                elif inplace and padded_k and self.layout_v2:
                    # v2 learning step: the header does not bind the step length, so the window is hashed in place now; its leaves (1/8 of
                    # the bytes) wait on PINNED HOST until the step's length is known and the subtrees can be laid into the host level slices
                    # (on device they were 1/8 of the step: 16 GB at B=64 -> OOM at step 0, 08:25Z); re-uploaded per window below
                    lv_k = H.leaves_window(st, padded_k, (k * W) // C, job.step, self.GPU_TREE_SRC, C, 0, side.cuda_stream)
                    hb = self.pool.take_block(int(lv_k.numel()) * 4)
                    hv_k = hb[:int(lv_k.numel()) * 4].view(torch.int32).view(lv_k.shape)
                    hv_k.copy_(lv_k, non_blocking=True)
                    side.synchronize()
                    del lv_k
                    pending.append((k, hv_k, hb))
            if fault_lifetime:
                gen0 = int(nc.slot_generation(int(slot)))
                nc.release_slot(int(slot), side.cuda_stream)                    # BUG (deliberate): released with no retained copy taken yet
                if nc.wait_slot_reused(int(slot), gen0, 5.0, side.cuda_stream):  # the ring refilled the slot: its bytes are a later window's
                    with torch.inference_mode(), torch.cuda.stream(side):
                        hv = self.pool.take(padded_k)
                        hv.copy_(st[:padded_k], non_blocking=True)              # "retained" copy = the wrong occupant
                        side.synchronize()
                    self._fault_lifetime_windows = getattr(self, "_fault_lifetime_windows", 0) + 1
                elif keep_win or not self.layout_v2:
                    with torch.inference_mode(), torch.cuda.stream(side):       # no later window came (end of run): fall back to the right bytes
                        hv = self.pool.take(padded_k)
                        hv.copy_(st[:padded_k], non_blocking=True)
                        side.synchronize()
            else:
                nc.release_slot(int(slot), side.cuda_stream)                    # the next taker waits on this point of the side stream
            if hv is not None:
                host_wins.append((k, hv, padded_k))
                if not keep_win:
                    transient.append(k)                                         # v1 re-hash source only: dropped from `keep` below
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
            # name (the tree hashed in place carries the plan's M in every header -- it is NOT the commitment of this step's bytes).
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
            nc.finish_step(job.step)
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
        rehash = bool(padded) and not self.layout_v2 and (not inplace or mismatch)
        if self.layout_v2 and mismatch and state:
            # v2: the plan changed mid-step.  Nothing hashed so far is wrong -- the header does not bind the step length and a FULL window's
            # subtree is a closed computation over its 2^j leaves; only the host level buffers were sized for the plan's length and the
            # per-level node offsets (k << (j - L)) do not depend on it.  So: new buffers for the real length, copy the absorbed prefix of
            # every level across (pinned -> pinned memcpy of <= 1/4 of the bytes absorbed so far), keep the window tops; windows beyond the
            # plan's length are in `pending` and are laid in below.  (Was: raise -> COMMIT FAIL at B=32 retain=none, 08:10Z.)
            old = dict(state)
            begin_tree(padded // C)
            if state["n_levels"] != old["n_levels"]:
                raise RuntimeError(f"windowed absorb (layout v2): plan {plan_pad} -> step {padded} bytes changes the tree height class "
                                   f"({old['n_levels']} -> {state['n_levels']} window levels); no host copy to re-hash from")
            for L in range(state["n_levels"] if state["views"] is not None else 0):
                n_copy = min(int(old["views"][L].shape[0]), int(state["views"][L].shape[0]))
                if n_copy:
                    state["views"][L][:n_copy].copy_(old["views"][L][:n_copy])
            state["tops"] = old["tops"]
            self._win_relayouts = getattr(self, "_win_relayouts", 0) + 1
        if rehash and hfirst:
            state.clear()
            begin_tree(padded // C)
            with torch.inference_mode(), torch.cuda.stream(side):
                for kk, hs_k in hwins:
                    first = kk << j
                    absorb(kk, H.rows_frame(state["rs"], hs_k, first, min(1 << j, state["width"] - first), state["n_total"], side.cuda_stream))
            hwins = []
            self._win_hfirst_steps = getattr(self, "_win_hfirst_steps", 0) + 1
        elif rehash:
            # v1: the step length was unknown (learning step) or the plan changed mid-step -> hash from the host windows with the real length
            if not host_wins:
                raise RuntimeError("windowed absorb (layout v1): the plan changed mid-step and no host copy exists to re-hash from")
            state.clear()
            pending = []
            begin_tree(padded // C)
            m_words = 0 if self.layout_v2 else padded // 4
            with torch.inference_mode(), torch.cuda.stream(side):
                for kk, hv, n in host_wins:
                    if not n:
                        continue
                    self.win_scratch[:n].copy_(hv[:n], non_blocking=True)     # H2D pinned; same stream as the leaf kernel -> ordered
                    if rows:
                        lv_k = row_leaves(self.win_scratch, n, kk)
                    else:
                        lv_k = H.leaves_window(self.win_scratch, n, (kk * W) // C, job.step, self.GPU_TREE_SRC, C, m_words, side.cuda_stream)
                    absorb(kk, lv_k)
                    del lv_k
            self._win_reupload_bytes = getattr(self, "_win_reupload_bytes", 0) + padded
        if pending:
            if not state:
                begin_tree(padded // C)
            with torch.inference_mode(), torch.cuda.stream(side):
                for kk, hv_k, hb in pending:
                    lv_k = hv_k.to("cuda", non_blocking=True)                   # H2D of the window's leaves (1/8 of its bytes), side stream
                    absorb(kk, lv_k)
                    del lv_k
            side.synchronize()                                                  # the H2Ds out of the blocks are done -> reusable
            for _kk, _hv_k, hb in pending:
                self.pool.give_back(hb)
            pending = []
        if rows and padded:
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
        nc.finish_step(job.step)
        if bounded and learning:
            # retention of a bounded LEARNING step is carved out of the transient copies now that the metas are known (retain=host:
            # every chunk outside the excluded families' interiors; retain=sample: the challenged windows), pinned -> pinned; then the
            # transient blocks go back to the pool -- the step's host footprint returns to its retained bytes
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
                            if chunk_picks is not None:
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
        root = (self._rows_mod if rows else scheme).step_root(self.program_digest, self._ctx_digest(job.step), self._geo_digest, layout_dig, n, mr)
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

    def _commit_step_native(self, job: _StepJob) -> None:
        """Candidate B worker step: `StepHasher.commit` (C++, GIL released) does wait-event, zero pad, chunk leaves, tree levels, retained
        copy, root D2H and the done-synchronisation -- the same sequence as `_commit_step_gpu`; Python only binds the root."""
        from verity_vllm.commit import scheme
        from verity_vllm.acquire.committer_api import StepCommitment

        self._gpu_ext_load()                                       # openings use HG.gpu_open on the levels (loaded once)
        side = self._side_stream()
        t0 = time.perf_counter()
        keep_mode = {"device": 0, "host": 1, "none": 2, "sample": 2}[self.retain]   # R13 P0 retention policy (see native_host.RETAIN_MODES)
        keep_dst = None
        levels_dst = torch.empty(0, dtype=torch.uint8)
        if self.retain != "device" and self.retain_pinned:
            # P0: pinned pool buffer for the step's tree levels (leaves n0 = padded/256 digests of 32 B, then ceil(n/2) per level)
            n0 = -(-int(job.cursor) // GPU_CHUNK_BYTES)
            tot, n = 0, n0
            while True:
                tot += n
                if n <= 1:
                    break
                n = (n + 1) // 2
            levels_dst = self.pool.take(tot * 32)
        if keep_mode == 1:
            # P0: host retention lands in PINNED memory (pool blocks, recycled per run) so the D2H runs at PCIe rate; pageable D2H measured
            # ~1 GB/s and made every decode step wait on the side stream (Llama 1024/128 B=1: hashing 7.1 s of an 8.6 s run)
            padded0 = -(-int(job.cursor) // GPU_CHUNK_BYTES) * GPU_CHUNK_BYTES
            keep_dst = self.pool.take(padded0) if self.retain_pinned else None
        levels, keep, mr, _padded, leaf_ms, tree_ms = self._hasher.commit(job.arena, job.cursor, job.step, self.GPU_TREE_SRC, GPU_CHUNK_BYTES,
                                                                          side.cuda_stream, job.event.cuda_event, keep_mode,
                                                                          keep_dst if keep_dst is not None else torch.empty(0, dtype=torch.uint8),
                                                                          self.retain != "device", levels_dst)     # levels -> pinned host inside the hasher
        if keep_mode == 2:
            keep = None
        if self.retain != "device" and levels and levels[0].is_cuda:
            levels = [lv.cpu() for lv in levels]                   # (fallback) authentication state to host; the device keeps only arena blocks
        self._footprint_absorbed(job, int(_padded), levels, keep)
        self.arena.release(job.arena)                              # after the done-event synchronisation inside commit()
        self._gpu_t["leaf_ms"] += leaf_ms
        self._gpu_t["tree_ms"] += tree_ms
        t1 = time.perf_counter()
        metas = job.metas
        nbytes = metas.nbytes
        n = int(levels[0].shape[0])
        layout_dig = self.layout_digest(metas)
        root = scheme.step_root(self.program_digest, self._ctx_digest(job.step), self._geo_digest, layout_dig, n, mr)
        t2 = time.perf_counter()
        self.spans.add("hashing", t1 - t0, step=job.step, nbytes=nbytes, leaves=n)
        self.spans.add("root_finalisation", t2 - t1, step=job.step)
        self._gpu_levels[job.step] = levels
        self._gpu_blocks[job.step] = (keep, job.cursor)
        self._layouts[job.step] = metas
        job.arena = None
        self._done[job.step] = StepCommitment(job.step, root, mr, n, nbytes, len(metas), layout_dig, job.t_step_end, t2, job.backlog,
                                              hashing_s=t1 - t0, root_finalisation_s=t2 - t1)

    def _commit_step_gpu_faulty(self, job: _StepJob) -> None:
        """DELIBERATE ordering bug (negative test only): the arena block is handed back for reuse before the retained copy is taken; the
        worker then waits until the serving thread has reused it.  The tree was computed over the step's bytes; the retained copy holds the
        next step's bytes -> openings must fail against the (correct) root.  Same code as `_commit_step_gpu` otherwise."""
        e = self._gpu_ext_load()
        side = self._side_stream()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.cuda.stream(side):
            side.wait_event(job.event)
            used = job.cursor
            padded = -(-used // GPU_CHUNK_BYTES) * GPU_CHUNK_BYTES
            if padded > used:
                job.arena[used:padded].zero_()
            for m in job.metas:
                m.stream_off = m.dev_off
                m.dev_block = None
            stream_i32 = job.arena[:padded].view(torch.int32)
            leaves = e.chunk_leaves(stream_i32, GPU_CHUNK_BYTES // 4, job.step, self.GPU_TREE_SRC, job.step, padded // 4, 0)
            levels = e.tree_levels(leaves)
            mr_t = levels[-1].to("cpu", non_blocking=True)
            torch.cuda.current_stream().synchronize()
            blk = job.arena
            self.arena.release(blk)                                # BUG: released before the retained copy exists
            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline and any(b is blk for b in self.arena.free):
                time.sleep(0.001)                                  # wait until the serving thread has taken (and is overwriting) the block
            time.sleep(0.05)
            keep = torch.empty(padded, dtype=torch.uint8, device=blk.device)
            keep.copy_(blk[:padded])                               # retained copy taken AFTER the block was reused
            torch.cuda.current_stream().synchronize()
        mr = mr_t.numpy().astype("uint32").tobytes()
        t1 = time.perf_counter()
        from verity_vllm.commit import scheme
        from verity_vllm.acquire.committer_api import StepCommitment

        n = int(leaves.shape[0])
        nbytes = sum(m.nbytes for m in job.metas)
        layout_dig = self.layout_digest(job.metas)
        root = scheme.step_root(self.program_digest, self._ctx_digest(job.step), self._geo_digest, layout_dig, n, mr)
        t2 = time.perf_counter()
        self.spans.add("hashing", t1 - t0, step=job.step, nbytes=nbytes, leaves=n)
        self.spans.add("root_finalisation", t2 - t1, step=job.step)
        self._gpu_levels[job.step] = levels
        self._gpu_blocks[job.step] = (keep, used)
        self._layouts[job.step] = job.metas
        job.arena = None
        self._done[job.step] = StepCommitment(job.step, root, mr, n, nbytes, len(job.metas), layout_dig, job.t_step_end, t2, job.backlog,
                                              hashing_s=t1 - t0, root_finalisation_s=t2 - t1)

    # ------------------------------------------------------------------ reporting

    def finalize(self):
        for src in self.sources:
            if hasattr(src, "sync_stats"):
                src.sync_stats()
        dead = bool(getattr(self._nc, "win_dead", False)) and hasattr(self._nc, "ack_dead")
        if dead:
            # F-stoch-03: the run reached finalize -> X-03 fails it closed (the error below is in the record `super().finalize()` builds
            # from self.errors, next to the drained slot-wait error).  The fail-fast watchdog stands down only AFTER super().finalize()
            # returned, i.e. after the worker was joined and the record built -- a join on a step that never completes (pin M-0662 (2))
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
        st = dict(self._nc.stats())
        # X-03 reference integrity: what the manifest attempted vs what was bound (the Python `_acquire` path puts through stage/collect
        # directly, so its successes are added here); commit_delta refuses PASS unless they are equal and no error was recorded
        rc.extra["expected_tensors"] = int(st.get("expected", 0))
        rc.extra["committed_tensors"] = int(st.get("committed", 0)) + int(self._py_committed)
        rc.extra["windows"] = {**dict(self._nc.window_stats()), **getattr(self, "window_info", {}),
                               "reupload_bytes": getattr(self, "_win_reupload_bytes", 0),
                               "hfirst_steps": getattr(self, "_win_hfirst_steps", 0),
                               "relayouts": getattr(self, "_win_relayouts", 0),
                               "fault_lifetime_windows": getattr(self, "_fault_lifetime_windows", 0)}    # >0 only under VERITY_FAULT=release_before_keep
        # [R17 fp8] bounded staging record: what the flag did this run (a learned-only step = a warm-up-only outcome; on a record run the
        # list MUST be empty -- commit_delta refuses PASS otherwise)
        rc.extra["bounded_staging"] = {"enabled": bool(self.bounded), "learn_only": bool(self.learn_only),
                                       "learn_budget_bytes": int(self.learn_budget_bytes),
                                       "learned_only_steps": list(self.learned_only_steps), "learned_only_bytes": int(self.learned_only_bytes),
                                       "transient_peak_bytes": int(self.transient_peak_bytes), "plan_mismatches": int(self.plan_mismatches),
                                       # [F-fp8-PLANKEY] how the plan key was formed: "geometry" = (ntok, nreq, multiset of per-request
                                       # (num_computed_tokens, num_scheduled_tokens)); "counts" = (ntok, nreq) only (older scheduler output)
                                       "plan_key_mode": getattr(self, "_plan_key_mode", "counts"),
                                       "retain_exclude": sorted(self.retain_exclude), "excluded_bytes": int(self.excluded_bytes),
                                       "host_retained_raw_bytes": sum(int(getattr(b, "retained_bytes", b.numel() if b is not None else 0))
                                                                      for b, _u in self._gpu_blocks.values() if b is not None and not getattr(b, "is_cuda", False)),
                                       # the admission this run was launched under (commit_delta's host bound + the DECLARED schedule it came
                                       # from; None = no admission computed) -- on the record, not only in host_admission_<kind>.json [F-r17b S-r17b-03 (iii)]
                                       "host_admission": getattr(self, "host_admission", None)}
        rc.extra["native_collect"] = {**st, "wrap": self.wrap, "profile": self.profile, "gather": self.gather, "native_hooks": self.native_hooks,
                                      "blocks_per_desc": int(self._nc.blocks_per_desc) if self.gather else None,
                                      "plans_cached": len(self._metas_cache),
                                      "names": len(self._names), "consumer_delay_ms": self.consumer_delay_s * 1e3, "fault": self.fault,
                                      "cuda_runtime": int(ext().cuda_runtime_version()),
                                      "fa2_wrap": {s.short_name: s.stats for s in self.sources if hasattr(s, "sync_stats")}}
        return rc

    def reset(self) -> None:
        super().reset()
        self.row_keys = {}
        self._py_committed = 0
        self._nc.end_run()              # the collector's run totals, and the plans this run did not use (native_plans.h)
        self._metas_cache.end_run()     # the worker's metas, by the same rule (plan_cache)
        self.learned_only_steps = []    # [R17 fp8] per-run bounded-staging counters (the plans learned stay for the next run)
        self.learned_only_bytes = 0
        self.transient_peak_bytes = 0
        self.excluded_bytes = 0
        self.plan_mismatches = 0
        self._win_reupload_bytes = 0    # windowed: per-run counters of finalize's extra["windows"]
        self._win_hfirst_steps = 0
        self._win_relayouts = 0
        self._oar_deferred = []         # picks the in-run stages could not open from host (excluded families) -> openings_after_release
        for src in self.sources:
            fn = getattr(src, "begin_run", None)
            if fn is not None:
                fn()                    # R15 compiled S3: per-run source counters (slots/hooks persist across the process's runs)
