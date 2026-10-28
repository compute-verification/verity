"""The native collector's hidden M1 taps: `hidden_source`'s FA2 and FA3 sources with their Python wrapper of the tapped kernel
replaced by one C++ callable each (`Fa2Wrap`, `Fa3Wrap` in native_collect.cpp, built from the committer's extension), and the FA2 tap's
bounded split driver, which taps a multi-sequence launch in sequence groups when its tap buffer would exceed the cap."""
from __future__ import annotations

import time
from typing import Any

from verity_vllm import settings
from verity_vllm.acquire.sources.hidden_source import (
    HiddenM1Source,
    HiddenM1SourceFA3,
    fa2_kblock_n,
    fa2_swapped,
    fa2_tap_buffer_words,
    fa2_tap_words_per_seq,
    plan_fa2_tap_groups,
)


class NativeHiddenM1Source(HiddenM1Source):
    """The FA2 M1 tap glue as ONE C++ callable (`Fa2Wrap`, native_collect.cpp) in place of the Python `varlen_fwd`
    wrapper.  Same tapped kernel, same per-geometry buffer rule, same names/order into the same collector; stats synced at uninstall."""

    #: default cap on the FA2 tap's device residency (MiB); VERITY_FA2_TAP_CAP_MB=0 restores the unbounded buffers
    DEFAULT_TAP_CAP_MB = 2048

    def __init__(self, committer: Any, so_path: str, ns: str = "verity_fa2_matAll") -> None:
        super().__init__(committer, so_path, ns)
        self._w = committer.native_ext().Fa2Wrap(committer._nc, self.orig_op, self.tapped_op, committer._intern, ns)
        # per-sequence compaction of multi-sequence launches (only the rows the kernel wrote are staged; B=1 unchanged)
        self._w.compact = settings.VERITY_FA2_COMPACT
        self._w.zero_after = settings.VERITY_FA2_ZERO
        self.short_name, self._w.ms = "hidden_m1", self.ms              # the MS class sizes every buffer and is a seventh compacted plane
        # bounded tap: one device buffer of <= cap bytes; a multi-sequence launch whose padded tap buffer exceeds it is
        # split into consecutive sequence groups (hidden_source.plan_fa2_tap_groups) tapped one after another into that buffer (C++ Fa2Wrap
        # seq_base / hold_launch / bound_words).  Names and bytes of the staged compacted rectangles are those of the unsplit launch.
        cap = settings.VERITY_FA2_TAP_CAP_MB
        cap_mb = self.DEFAULT_TAP_CAP_MB if cap is None else cap
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

    # ------------------------------------------------------------------ the split driver
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
                return w(*args, **kwargs)                         # the C++ wrapper refuses it by name
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

    #: a detached window's copy of the source (`window_views.window_source`): its stats stand as synced at the detach
    stats_frozen = False

    def sync_stats(self) -> None:
        if self.stats_frozen:
            return
        st = self._w.stats()
        self.stats["launches"] = int(st["launches"])
        self.stats["bytes"] = int(st["bytes"])
        self.stats["set_tap_s"] = float(st["wrap_s"])          # whole native wrapper wall (geometry + tap set + tapped op + stage), measured
        self.stats["geometries"] = {str(k): int(v) for k, v in dict(st["geometries"]).items()}
        self.stats["compacted_launches"] = int(st.get("compacted_launches", 0))
        self.stats["compact"] = bool(st.get("compact", False))
        self.stats["refused"] = int(st.get("refused", 0))       # launches expected but not bound (uncovered tile / kernel skip) -> errors
        self.stats["tap_layout"] = int(st.get("tap_layout", 7))
        # the bounded-tap record: cap, the one bounded allocation, peaks, split counts (all bytes, all measured)
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
            except OSError as e:
                self.c.errors.append(f"fa2 tap telemetry close: {e!r}"[:200])
            self._tel_fh = None


class NativeHiddenM1SourceFA3(HiddenM1SourceFA3):
    """The FA3 M1 tap glue as ONE C++ callable (`Fa3Wrap`, native_collect.cpp) in place of the Python `fwd` wrapper of
    `hidden_source.HiddenM1SourceFA3` -- same tapped kernel (`verity_fa3_matReq.so`), same geometry / tile rule, same names into the same
    collector; stats synced at uninstall."""

    def __init__(self, committer: Any, so_path: str, ns: str = "verity_fa3_matReq") -> None:
        super().__init__(committer, so_path, ns)
        self._w = committer.native_ext().Fa3Wrap(committer._nc, self.orig_op, self.tapped_op, committer._intern, ns)
        # P0-03's rule for the FA3 tap: zero the staged whole buffer after staging on multi-sequence launches (B=1 unchanged)
        zero = settings.VERITY_FA3_ZERO
        self._w.zero_after = settings.VERITY_FA2_ZERO if zero is None else zero
        self.short_name, self._w.ms = "hidden_m1_fa3", self.ms

    def _wrapper(self):
        return self._w

    def on_step_begin(self) -> None:
        self._w.on_step_begin()

    stats_frozen = False          # as `NativeHiddenM1Source.stats_frozen`

    def sync_stats(self) -> None:
        if self.stats_frozen:
            return
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
