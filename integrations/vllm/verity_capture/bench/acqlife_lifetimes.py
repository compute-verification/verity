"""R10 acq-life — MEASURED lifetimes of every value the reference committer (`native_host`, R9 CMT-3) collects per engine step.

DIAGNOSTIC ONLY (never in a headline timing path).  For every hook site of the frozen collector (same module set, same step
boundaries — `NativeHostCommitter` is imported for its selection logic, not installed) this records, on the pod:

  produced at    (step, hook ordinal, module name, module class, output index, dtype, shape, nbytes)
  location       data_ptr, storage data_ptr / nbytes, `_version` (guarded; inference tensors carry none), producing CUDA stream
  content        a snapshot copy at hook time (cudaMemcpyAsync D2D into a PRIVATE preallocated arena — no allocator traffic)
  first change   at every later boundary b (each later hook of the same step within a fine window, every layer boundary, the step
                 end, the next step's first hook, the next step's end) the bytes at the RECORDED POINTER RANGE are re-read (raw
                 cudaMemcpyAsync from the pointer, exactly what a deferred copy would read) and compared with the snapshot; the first
                 boundary at which they differ is the measured end of the safe window.  `--hold` keeps a reference to every tensor for
                 two steps: then only IN-PLACE writes can change the bytes; the difference between the two modes is allocator reuse.
  reuse          when a later collected tensor's pointer range overlaps an earlier one's, the reuse is attributed (collected-by-collected);
                 `--memhist` additionally records the caching allocator's history (alloc/free/segment events with stacks) for one step so
                 that reuse by UNCOLLECTED intermediates can be attributed to the allocating op.

Boundaries are named so that a batching collector can be specified against them: `hook+<k>` (k later hooks), `layer_end`, `step_end`,
`next_step_first_hook`, `next_step_end`, `survived` (unchanged through the whole horizon).

    PYTHONPATH=. /workspace/venv-cu129/bin/python -m verity_capture.bench.acqlife_lifetimes --out /workspace/cp/acqlife/lt \
        [--hold] [--memhist] [--hidden-so /workspace/cp/fa2/build/matAll/verity_fa2_matAll.so] [--fine-window 40]

Outputs (DIR): tensors.jsonl (one row per collected tensor per step), classes.json (aggregation per value class), streams.json,
memhist_step<k>.pickle (+ memhist_reuse.json), summary.json.  Every number is MEASURED on the pod unless labelled otherwise.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pickle
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from verity_vllm_adapter import profile as prof  # noqa: F401

prof.apply_env()

import torch  # noqa: E402

from verity_capture import vllm_adapter as va  # noqa: E402
from verity_capture.bench.commit_delta import run_workload  # noqa: E402
from verity_capture.experimental.commit_integ import native_host as nh  # noqa: E402
from verity_capture.experimental.commit_integ.spans import Spans  # noqa: E402

SCHEMA = "verity/acq-life/lifetimes/v1"
MEMCPY_D2D = 3


class _Rec:
    __slots__ = ("step", "ord", "name", "mcls", "dtype", "shape", "nbytes", "ptr", "sptr", "snbytes", "version", "stream", "t", "t_unix_us",
                 "snap_off", "first_change", "first_change_kind", "checks", "reused_by", "held", "layer", "hook_in_layer", "same_obj_as")

    def __init__(self, **kw: Any) -> None:
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def _layer_of(name: str) -> int | None:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


class LifetimeTracker:
    """Duck-compatible with the parts of NativeHostCommitter that HiddenM1Source touches (`_cur`, `_acquire`, `spans`)."""

    def __init__(self, engine: Any, *, hold: bool, fine_window: int, arena_bytes: int, horizon_steps: int = 2) -> None:
        self.engine, self.runner, self.model = engine, engine.runner, engine.model
        self.hold, self.fine_window, self.horizon = hold, fine_window, horizon_steps
        self.spans = Spans()
        self._cur: _Job | None = None
        self.sources: list = []
        self._handles: list = []
        self._orig: dict[str, Any] = {}
        self._had_attr: dict[str, bool] = {}
        self.step_index = 0
        self.rt, self.cudart_path = nh.NativeHostCommitter._load_cudart()
        self.arena = torch.empty(arena_bytes, dtype=torch.uint8, device="cuda")
        self.arena_ptr = self.arena.data_ptr()
        self.arena_cursor = 0
        self.scratch = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")     # largest single tensor (prefill FA2 stream 4.4 MB; margin)
        self.diff = torch.empty(64 << 20, dtype=torch.bool, device="cuda")
        self.max_checks = 400_000
        self.res = torch.zeros(self.max_checks, dtype=torch.int32, device="cuda")   # per-check mismatch counts (read once per step)
        self.res_n = 0
        self.checks_pending: list[tuple[_Rec, str, int]] = []                        # (rec, boundary label, res index)
        self.recs: list[_Rec] = []                                                   # all records, all steps
        self.alive: list[_Rec] = []                                                  # records whose first change is not yet found
        self.held_refs: dict[int, list[Any]] = defaultdict(list)                     # step -> tensor refs (hold mode)
        self.streams: dict[str, Any] = {"default": torch.cuda.default_stream().cuda_stream, "seen": {}}
        self.errors: list[str] = []
        self.noncontig: dict[str, int] = {}
        self.step_ends: dict[int, float] = {}
        self.hooks_per_step: dict[int, int] = {}
        mods = nh.NativeHostCommitter.__new__(nh.NativeHostCommitter)
        mods.model, mods.runner, mods.acquire_classes = self.model, self.runner, nh.ACQUIRE_CLASSES
        self._modules = nh.NativeHostCommitter._select_modules(mods)
        self._mcls_cache: dict[str, str] = {}

    # ---------------------------------------------------------------- install / step boundaries (same as the reference collector)
    def install(self) -> None:
        for src in self.sources:
            src.install()
        for name, m in self._modules:
            self._handles.append(m.register_forward_hook(self._make_hook(name)))
            if nh._mro_names(m) & nh.ACQUIRE_INPUT_CLASSES:
                self._handles.append(m.register_forward_pre_hook(self._make_pre_hook(name)))
        runner, me = self.runner, self
        for nm in ("execute_model", "sample"):
            if hasattr(runner, nm):
                self._had_attr[nm] = nm in runner.__dict__
                self._orig[nm] = getattr(runner, nm)
        orig_exec = self._orig["execute_model"]

        def execute_model(scheduler_output, *a, **kw):
            dummy = bool(kw.get("dummy_run", False)) or (len(a) >= 2 and bool(a[1]))
            if dummy or getattr(scheduler_output, "total_num_scheduled_tokens", 0) == 0:
                return orig_exec(scheduler_output, *a, **kw)
            me._end_step()
            me._begin_step()
            return orig_exec(scheduler_output, *a, **kw)

        runner.execute_model = execute_model
        if "sample" in self._orig:
            orig_sample = self._orig["sample"]

            def sample(*a, **kw):
                res = orig_sample(*a, **kw)
                job = me._cur
                if job is not None:
                    try:
                        so = res[0] if isinstance(res, tuple) else res
                        st = getattr(so, "sampled_token_ids", None)
                        if isinstance(st, torch.Tensor) and st.device.type == "cuda":
                            me._acquire("runner.sampler/sampled_token_ids", st, job)
                    except Exception as e:  # noqa: BLE001
                        me.errors.append(f"sampler acquire: {e!r}"[:200])
                    me._end_step()
                return res

            runner.sample = sample

    def uninstall(self) -> None:
        self._end_step()
        for src in self.sources:
            src.uninstall()
        for h in self._handles:
            h.remove()
        self._handles = []
        for nm, fn in self._orig.items():
            if self._had_attr.get(nm):
                setattr(self.runner, nm, fn)
            else:
                try:
                    delattr(self.runner, nm)
                except AttributeError:
                    pass
        self._orig = {}

    def _begin_step(self) -> None:
        self._cur = _Job(self.step_index)
        self.step_index += 1
        for src in self.sources:
            src.on_step_begin()
        # boundary: first hook of the new step is checked lazily in _acquire (label next_step_first_hook) for the previous step's records

    def _end_step(self) -> None:
        job = self._cur
        if job is None:
            return
        self._cur = None
        self.hooks_per_step[job.step] = job.n
        # boundary: step end — check every alive record produced in this step or the previous `horizon` steps
        for rec in list(self.alive):
            if rec.step == job.step:
                self._check(rec, "step_end")
            elif rec.step == job.step - 1:
                self._check(rec, "next_step_end")
            elif rec.step < job.step - 1:
                self._check(rec, f"step+{job.step - rec.step}_end")
        self._flush_checks()
        # retire records older than the horizon
        for rec in list(self.alive):
            if rec.step <= job.step - self.horizon:
                rec.first_change = rec.first_change or "survived"
                rec.first_change_kind = rec.first_change_kind or "none"
                self.alive.remove(rec)
        if self.hold:
            for s in [s for s in self.held_refs if s <= job.step - self.horizon]:
                del self.held_refs[s]
        self.step_ends[job.step] = time.perf_counter()

    # ---------------------------------------------------------------- acquisition = record + snapshot + boundary checks
    def _mcls(self, name: str) -> str:
        base = name.split("/")[0]
        c = self._mcls_cache.get(base)
        if c is None:
            c = "Sampler" if base == "runner.sampler" else "FA2M1" if base.startswith("fa2.m1") else "?"
            for n, m in self._modules:
                if n == base:
                    c = type(m).__name__
                    break
            self._mcls_cache[base] = c
        return c

    def _acquire(self, name: str, t: torch.Tensor, job: "_Job") -> None:
        if t.device.type != "cuda" or t.numel() == 0:
            return
        nbytes = t.numel() * t.element_size()
        if not t.is_contiguous():
            # a strided VIEW (prefill RoPE q/k inside the qkv buffer): track the storage SPAN the view covers (superset incl. the interleaved
            # columns) — its lifetime is the qkv buffer's; the frozen path materialises these via .contiguous() (torch slow path)
            span = (sum((s - 1) * st for s, st in zip(t.shape, t.stride())) + 1) * t.element_size()
            self.noncontig[name.rsplit(".", 1)[-1]] = self.noncontig.get(name.rsplit(".", 1)[-1], 0) + 1
            nbytes = span
        stream = torch.cuda.current_stream()
        sh = stream.cuda_stream
        self.streams["seen"][str(sh)] = self.streams["seen"].get(str(sh), 0) + 1
        try:
            ver = int(t._version)
        except Exception:  # noqa: BLE001
            ver = None
        try:
            sptr, snb = t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()
        except Exception:  # noqa: BLE001
            sptr, snb = None, None
        off = (self.arena_cursor + 255) & ~255
        if off + nbytes > self.arena.numel():
            off = 0                                                    # wrap (records older than the horizon are retired anyway)
        rc = self.rt.cudaMemcpyAsync(self.arena_ptr + off, t.data_ptr(), nbytes, MEMCPY_D2D, sh)
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync rc={rc}")
        self.arena_cursor = off + nbytes
        rec = _Rec(step=job.step, ord=job.n, name=name, mcls=self._mcls(name), dtype=str(t.dtype).replace("torch.", ""), shape=tuple(int(x) for x in t.shape),
                   nbytes=nbytes, ptr=t.data_ptr(), sptr=sptr, snbytes=snb, version=ver, stream=sh, t=time.perf_counter(), t_unix_us=int(time.time() * 1e6), snap_off=off,
                   first_change=None, first_change_kind=None, checks=0, reused_by=None, held=self.hold, layer=_layer_of(name), hook_in_layer=None,
                   same_obj_as=None)
        # reuse attribution: does this tensor's range overlap an earlier still-alive record's range (collected-by-collected)?
        lo, hi = rec.ptr, rec.ptr + nbytes
        for old in self.alive:
            if old.ptr < hi and lo < old.ptr + old.nbytes:
                if old.step == job.step and old.ptr == rec.ptr and old.nbytes == nbytes and old.mcls == rec.mcls and old.name == name:
                    continue
                if old.reused_by is None:
                    old.reused_by = f"s{job.step}/h{job.n}:{name}"
                if old.ptr == rec.ptr and old.nbytes == nbytes and old.first_change is None:
                    rec.same_obj_as = f"s{old.step}/h{old.ord}:{old.name}"
        if self.hold:
            self.held_refs[job.step].append(t)
        # boundary checks for earlier alive records: fine window (every later hook) / layer boundary / first hook of a new step
        is_layer_start = name.endswith("input_layernorm") or name.endswith("input_layernorm/0")
        for old in list(self.alive):
            if old.step == job.step:
                d = job.n - old.ord
                if d <= self.fine_window or is_layer_start:
                    self._check(old, f"hook+{d}" if d <= self.fine_window else "layer_end")
            elif old.step == job.step - 1 and job.n == 0:
                self._check(old, "next_step_first_hook")
        self.recs.append(rec)
        self.alive.append(rec)
        job.n += 1
        job.names.append(name)
        if job.n % 64 == 0:
            self._flush_checks()

    def _check(self, rec: _Rec, label: str) -> None:
        if rec.first_change is not None:
            return
        if self.res_n >= self.max_checks:
            self._flush_checks()
        n = rec.nbytes
        sh = torch.cuda.current_stream().cuda_stream
        rc = self.rt.cudaMemcpyAsync(self.scratch.data_ptr(), rec.ptr, n, MEMCPY_D2D, sh)
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync(check) rc={rc}")
        a = self.scratch[:n]
        b = self.arena[rec.snap_off:rec.snap_off + n]
        d = self.diff[:n]
        torch.ne(a, b, out=d)
        torch.sum(d, 0, dtype=torch.int32, out=self.res[self.res_n])
        self.checks_pending.append((rec, label, self.res_n))
        self.res_n += 1
        rec.checks += 1

    def _flush_checks(self) -> None:
        if not self.checks_pending:
            return
        vals = self.res[:self.res_n].cpu().tolist()
        # earliest boundary wins: pending list is in issue order (= boundary order)
        for rec, label, i in self.checks_pending:
            if rec.first_change is None and vals[i] != 0:
                rec.first_change = label
                rec.first_change_kind = "in_place" if (self.hold or rec.reused_by is None) else "reuse_or_in_place"
                if rec in self.alive:
                    self.alive.remove(rec)
        self.checks_pending = []
        self.res_n = 0

    def _make_hook(self, name: str):
        me = self

        def hook(module, args, output):
            job = me._cur
            if job is None:
                return None
            try:
                if isinstance(output, torch.Tensor):
                    me._acquire(name, output, job)
                elif isinstance(output, (tuple, list)):
                    for k, o in enumerate(output):
                        if isinstance(o, torch.Tensor):
                            me._acquire(f"{name}/{k}", o, job)
                else:
                    st = getattr(output, "sampled_token_ids", None)
                    if isinstance(st, torch.Tensor):
                        me._acquire(f"{name}/sampled_token_ids", st, job)
            except Exception as e:  # noqa: BLE001
                me.errors.append(f"{name}: {e!r}"[:300])
            return None

        return hook

    def _make_pre_hook(self, name: str):
        me = self

        def pre(module, args):
            job = me._cur
            if job is None or not args:
                return None
            try:
                if isinstance(args[0], torch.Tensor):
                    me._acquire(f"{name}/in", args[0], job)
            except Exception as e:  # noqa: BLE001
                me.errors.append(f"{name}/in: {e!r}"[:300])
            return None

        return pre

    def finish(self) -> None:
        self._end_step()
        torch.cuda.synchronize()
        self._flush_checks()
        for rec in self.alive:
            rec.first_change = rec.first_change or ("survived(horizon-cut)" if rec.step >= self.step_index - self.horizon else "survived")
            rec.first_change_kind = rec.first_change_kind or "none"
        self.alive = []


class _Job:
    __slots__ = ("step", "n", "names")

    def __init__(self, step: int) -> None:
        self.step, self.n, self.names = step, 0, []


# -------------------------------------------------------------------- value class of a record (the lifetime table's row key)
def value_class(rec: _Rec) -> str:
    name = rec.name
    base, _, idx = name.partition("/")
    leaf = base.rsplit(".", 1)[-1]
    if base.startswith("fa2.m1"):
        return "FA2.M1 export stream (R5-R8)"
    if leaf == "embed_tokens":
        return "embed_tokens/in (R1b input ids)" if idx == "in" else "embed_tokens out (R2b)"
    if base == "runner.sampler":
        return "sampled_token_ids (R2i)"
    if leaf in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
        return f"{leaf} out (R2a GEMM)"
    if leaf == "rotary_emb":
        return "rotary_emb/" + ("q" if idx == "0" else "k") + " (R2g RoPE, view of qkv)"
    if leaf == "attn":
        return "attn out (R2h)"
    if leaf == "act_fn":
        return "act_fn out (R2f SiLU*mul)"
    if leaf == "logits_processor":
        return "logits (R2a lm_head)"
    if leaf == "input_layernorm" and rec.layer == 0 and idx == "":
        return "layer0 input_layernorm out (R2c, Triton, not fused)"
    if leaf in ("input_layernorm", "post_attention_layernorm"):
        return f"{leaf}/{'normed' if idx == '0' else 'residual'} (R2d/R2e fused)"
    if leaf == "norm":
        return f"final norm/{'normed' if idx == '0' else 'residual'} (R2d/R2e fused)"
    return f"{leaf}/{idx}"


def _rank(label: str) -> int:
    """Order boundaries so that min/max over a class are meaningful (smaller = earlier)."""
    if label is None:
        return 10**6
    if label.startswith("hook+"):
        return int(label[5:])
    return {"layer_end": 1000, "step_end": 2000, "next_step_first_hook": 3000, "next_step_end": 4000}.get(label, 5000 if label.startswith("step+") else 9000)


def aggregate(recs: list[_Rec], hold: bool) -> dict:
    by: dict[str, list[_Rec]] = defaultdict(list)
    for r in recs:
        by[value_class(r)].append(r)
    out = {}
    for cls, rs in sorted(by.items()):
        labels = [r.first_change for r in rs]
        ranks = [_rank(l) for l in labels]
        lost_at_step_end = [r for r in rs if _rank(r.first_change) <= 2000]
        earliest = min(zip(ranks, labels))[1]
        latest = max(zip(ranks, labels))[1]
        prefill = [r for r in rs if r.step == 0]
        decode = [r for r in rs if r.step > 0]
        verdict = ("copy-at-hook-required" if any(_rank(l) <= 2000 for l in labels)
                   else "batchable-until-step_end" if all(_rank(l) > 2000 for l in labels) and any(_rank(l) < 9000 for l in labels)
                   else "persistent(≥2 steps)")
        out[cls] = {
            "tensors": len(rs), "per_step": round(len(rs) / max(1, len({r.step for r in rs})), 2), "bytes_total": sum(r.nbytes for r in rs),
            "bytes_per_step_decode": statistics.fmean([sum(r.nbytes for r in decode if r.step == s) for s in {r.step for r in decode}]) if decode else 0,
            "bytes_prefill": sum(r.nbytes for r in prefill),
            "first_change_earliest": earliest, "first_change_latest": latest,
            "first_change_hist": dict(sorted(((l or "None"), labels.count(l)) for l in set(labels))),
            "first_change_kind_hist": dict(sorted((k or "None", [r.first_change_kind for r in rs].count(k)) for k in {r.first_change_kind for r in rs})),
            "reused_by_collected": sum(1 for r in rs if r.reused_by),
            "reused_by_examples": sorted({(r.reused_by or "").split(":", 1)[-1] for r in rs if r.reused_by})[:4],
            "same_storage_as_earlier_collected": sum(1 for r in rs if r.same_obj_as),
            "lost_by_single_end_of_step_copy": {"tensors": len(lost_at_step_end), "bytes": sum(r.nbytes for r in lost_at_step_end)},
            "streams": sorted({str(r.stream) for r in rs}), "dtypes": sorted({r.dtype for r in rs}),
            "shapes_decode": sorted({str(list(r.shape)) for r in decode})[:3], "shapes_prefill": sorted({str(list(r.shape)) for r in prefill})[:3],
            "verdict": verdict, "mode": "hold-refs (in-place only)" if hold else "no-refs (in-place + allocator reuse)",
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="verity_capture/profiles/workload_cov_b0_b1_c256.json")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--case", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold", action="store_true", help="hold a reference to every collected tensor for 2 steps (isolates in-place writes)")
    ap.add_argument("--fine-window", type=int, default=40, help="check every alive record at each of the next k hooks (then layer/step boundaries)")
    ap.add_argument("--arena-mb", type=int, default=1536)
    ap.add_argument("--memhist", action="store_true", help="record the caching allocator history for steps 1-2 (pickle + reuse attribution)")
    ap.add_argument("--hidden-so", default=None)
    ap.add_argument("--engine-arg", action="append", default=None)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wl = va.load_workload(args.workload)
    engine_args = dict(wl.get("engine_args_required") or {})
    engine_args.update(va.parse_engine_args(args.engine_arg))
    eng = va.build_engine(args.checkpoints, max_num_seqs=8, case=va.case_of(wl, args.case), engine_args=engine_args)
    print(f"[acqlife] engine built in {eng.build_s:.1f}s ({eng.cp['repo']}); env {eng.env_flags}", flush=True)
    reqs, sampling = wl["requests"], wl["sampling"]
    for i in range(args.warmup):
        r = run_workload(eng, va.throwaway_requests(eng, reqs, seed=100 + i), sampling)
        print(f"[acqlife] warm-up {i}: {r['wall_s']:.3f}s {r['iterations']} it", flush=True)
    ref = run_workload(eng, reqs, sampling)                                # control tokens (must be reproduced under instrumentation)

    tr = LifetimeTracker(eng, hold=args.hold, fine_window=args.fine_window, arena_bytes=args.arena_mb << 20)
    if args.hidden_so:
        from verity_capture.experimental.commit_integ.hidden_source import HiddenM1Source

        tr.sources.append(HiddenM1Source(tr, args.hidden_so))
        print(f"[acqlife] hidden source attached: {args.hidden_so}", flush=True)
    if args.memhist:
        torch.cuda.memory._record_memory_history(max_entries=2_000_000)
    tr.install()
    t0 = time.perf_counter()
    r = run_workload(eng, reqs, sampling)
    tr.finish()
    t1 = time.perf_counter()
    tr.uninstall()
    snap = None
    if args.memhist:
        snap = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        with (out / "memhist.pickle").open("wb") as f:
            pickle.dump(snap, f)
    tokens_eq = r["tokens"] == ref["tokens"]
    print(f"[acqlife] instrumented run {t1 - t0:.2f}s ({r['iterations']} steps; control {ref['wall_s']:.3f}s); tokens identical: {tokens_eq}; "
          f"records {len(tr.recs)}; errors {tr.errors[:3]}", flush=True)

    rows = []
    for rec in tr.recs:
        rows.append({"step": rec.step, "ord": rec.ord, "name": rec.name, "class": value_class(rec), "module_class": rec.mcls, "layer": rec.layer, "dtype": rec.dtype,
                     "shape": list(rec.shape), "nbytes": rec.nbytes, "ptr": rec.ptr, "storage_ptr": rec.sptr, "storage_nbytes": rec.snbytes, "version": rec.version,
                     "stream": rec.stream, "t_unix_us": rec.t_unix_us, "first_change": rec.first_change, "first_change_kind": rec.first_change_kind, "checks": rec.checks,
                     "reused_by": rec.reused_by, "same_storage_as": rec.same_obj_as, "held": rec.held})
    with (out / "tensors.jsonl").open("w") as f:
        for rw in rows:
            f.write(json.dumps(rw) + "\n")
    classes = aggregate(tr.recs, args.hold)
    (out / "classes.json").write_text(json.dumps(classes, indent=1, default=str) + "\n")
    lost = [rec for rec in tr.recs if _rank(rec.first_change) <= 2000]
    per_step_lost = defaultdict(lambda: [0, 0])
    for rec in lost:
        per_step_lost[rec.step][0] += 1
        per_step_lost[rec.step][1] += rec.nbytes
    streams = {"default_stream": tr.streams["default"], "hook_streams_seen": tr.streams["seen"],
               "runner_side_streams": {k: str(getattr(eng.runner, k, None)) for k in ("async_output_copy_stream", "prepare_inputs_event", "transfer_event")},
               "note": "hook_streams_seen == {default} means every collected value is produced on the compute (default) stream"}
    (out / "streams.json").write_text(json.dumps(streams, indent=1, default=str) + "\n")
    summary = {
        "schema": SCHEMA, "label": "measured on pod", "mode": "hold" if args.hold else "no-hold", "fine_window": args.fine_window,
        "model": eng.cp["repo"], "case": eng.case, "workload": args.workload, "steps": r["iterations"], "hooks_per_step": tr.hooks_per_step,
        "tensors_total": len(tr.recs), "bytes_total": sum(x.nbytes for x in tr.recs), "tokens_identical_to_control": tokens_eq,
        "instrumented_wall_s": t1 - t0, "control_wall_s": ref["wall_s"], "errors": tr.errors[:20], "noncontiguous_views_tracked_as_span": tr.noncontig,
        "single_end_of_step_copy_would_lose": {"tensors": len(lost), "bytes": sum(x.nbytes for x in lost),
                                               "per_step": {str(k): {"tensors": v[0], "bytes": v[1]} for k, v in sorted(per_step_lost.items())},
                                               "by_class": {c: d["lost_by_single_end_of_step_copy"] for c, d in classes.items() if d["lost_by_single_end_of_step_copy"]["tensors"]}},
        "classes": {c: {k: d[k] for k in ("tensors", "per_step", "bytes_total", "first_change_earliest", "first_change_latest", "verdict", "reused_by_collected")} for c, d in classes.items()},
        "streams": streams, "versions": {"torch": torch.__version__, "cudart": tr.cudart_path, "gpu": torch.cuda.get_device_name(0)},
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if snap is not None:
        summary["memhist"] = attribute_memhist(snap, tr.recs, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("single_end_of_step_copy_would_lose", "tokens_identical_to_control", "hooks_per_step")}, indent=1)[:3000], flush=True)
    for c, d in classes.items():
        print(f"  {c:58s} n={d['tensors']:5d} {d['verdict']:28s} earliest={d['first_change_earliest']} latest={d['first_change_latest']} reused={d['reused_by_collected']} kinds={d['first_change_kind_hist']}", flush=True)
    print(f"[acqlife] wrote {out}", flush=True)


def attribute_memhist(snap: dict, recs: list[_Rec], out: Path) -> dict:
    """From the allocator history: for each collected pointer, the free event after its production and the next allocation covering the
    address (the reusing op's frame stack), if any.  Labelled measured (allocator events) — the op name is the top non-torch frame."""
    ev = snap.get("device_traces", [[]])
    traces = ev[0] if ev else []
    has_time = bool(traces) and "time_us" in traces[0]
    by_ptr: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for i, e in enumerate(traces):
        if e.get("action") in ("alloc", "free_completed", "free_requested"):
            by_ptr[e.get("addr")].append((i, e))
    summary = {"events": len(traces), "has_time_us": has_time, "collected_with_history": 0, "freed_within_horizon": 0, "freed_then_realloc": 0,
               "realloc_frames": defaultdict(int), "free_frames": defaultdict(int)}
    rows = []

    def _frame(e: dict) -> str | None:
        for fr in e.get("frames", []):
            fn = fr.get("filename", "")
            if ("vllm" in fn or "verity" in fn) and "/torch/" not in fn and "acqlife" not in fn:
                return f"{fn.rsplit('/', 1)[-1]}:{fr.get('line')} {fr.get('name')}"
        return None

    offsets: list[int] = []
    anchors: dict[int, int] = {}
    for k, rec in enumerate(recs):
        hist = by_ptr.get(rec.ptr)
        if not hist or not has_time:
            continue
        # anchor = the alloc event of THIS tensor: the alloc at its address closest to (and normally shortly before) the hook time; the
        # allocator clock and time.time() differ by a small offset (measured below), so match within ±5 ms and take the closest
        cands = [(abs(e["time_us"] - rec.t_unix_us), i) for i, e in hist if e["action"] == "alloc" and abs(e["time_us"] - rec.t_unix_us) < 5000]
        if cands:
            anchors[k] = min(cands)[1]
            offsets.append(traces[anchors[k]]["time_us"] - rec.t_unix_us)
    offset = int(statistics.median(offsets)) if offsets else 0
    summary["clock_offset_us_median"] = offset
    summary["anchored"] = len(anchors)
    for k, rec in enumerate(recs):
        hist = by_ptr.get(rec.ptr)
        if not hist:
            continue
        summary["collected_with_history"] += 1
        a = anchors.get(k)
        # the free of THIS block: first free_completed at the address after its alloc event (or, unanchored, after the offset-corrected hook time)
        freed_i = None
        for i, e in hist:
            if e["action"] == "free_completed" and ((a is not None and i > a) or (a is None and (not has_time or e["time_us"] >= rec.t_unix_us + offset - 200))):
                freed_i = i
                break
        realloc = None
        if freed_i is not None:
            summary["freed_within_horizon"] += 1
            for j in range(freed_i + 1, min(freed_i + 5000, len(traces))):
                e = traces[j]
                if e.get("action") == "alloc" and e["addr"] <= rec.ptr < e["addr"] + e.get("size", 0):
                    realloc = e
                    break
        frame = None
        if realloc is not None:
            summary["freed_then_realloc"] += 1
            frame = _frame(realloc)
            summary["realloc_frames"][frame or "?"] += 1
        if freed_i is not None:
            summary["free_frames"][_frame(traces[freed_i]) or "?"] += 1
        t_hook = rec.t_unix_us + offset
        rows.append({"step": rec.step, "ord": rec.ord, "name": rec.name, "ptr": rec.ptr, "anchored": a is not None, "freed": freed_i is not None,
                     "alloc_before_hook_us": (t_hook - traces[a]["time_us"]) if a is not None else None,
                     "free_after_us": (traces[freed_i]["time_us"] - t_hook) if (freed_i is not None and has_time) else None,
                     "free_frame": _frame(traces[freed_i]) if freed_i is not None else None,
                     "realloc_frame": frame, "realloc_size": realloc.get("size") if realloc else None,
                     "realloc_after_us": (realloc["time_us"] - t_hook) if (realloc is not None and has_time) else None})
    (out / "memhist_reuse.json").write_text(json.dumps(rows, indent=1) + "\n")
    summary["realloc_frames"] = dict(sorted(summary["realloc_frames"].items(), key=lambda kv: -kv[1])[:40])
    summary["free_frames"] = dict(sorted(summary["free_frames"].items(), key=lambda kv: -kv[1])[:40])
    return summary


if __name__ == "__main__":
    main()
