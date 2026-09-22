#!/usr/bin/env python
"""Minimal-recording A/B/C/D harness (lane COST-D, 2026-09-09; design: docs/vllm-poc/minimal-recording-design.md).

NOT RUN in phase 1 (no GPU, no paid compute).  Ready for a GPU pod: ``--dry-run`` and ``--selftest`` work
on a laptop without vllm/torch; every other mode needs the pod environment of ``scripts/replay_pod_setup.sh``.

Question (brief §3 / first-stage-boundary.md): what does the REQUIRED online work of plan P cost on top of
stock serving, separately from (i) the numerical profile and (ii) the diagnostic capture path?

Arms (one engine CONFIG per process — stock and profile engines cannot share a process — arms inside a
process are interleaved per repetition; the driver alternates processes):

    config `stock`   : A  stock vLLM (BI=0, CUDA graphs), no recorder                     layer 1
                       A' A + production recorder (StepTrace + per-request outputs)
    config `profile` : B  numerical profile (VLLM_BATCH_INVARIANT=1, enforce_eager), no recorder   layer 2
                       B' B + production recorder                                          (= ladder R2')
                       C  B' + BoundaryRecorder(plan P): module-level forward hooks that copy ONLY the plan's
                          first-boundary rows (first-stage-boundary.md §3) to pinned host memory, hash them
                          and (optionally) write them                                      REQUIRED online work
                       D  B' + full diagnostic capture (capture.py, hash-only or values)    today's capture path

Paired differences within a repetition (same process, same engine, prefix cache reset before every pass,
arms in a fresh random order every repetition) → mean, SD, 95 % t-CI of the difference AND of the ratio;
never a single delta.  Logged per pass: wall (CUDA-synchronised), generated/prompt tokens, bytes copied /
hashed / written by the recorder, peak RSS, GPU utilisation and SM clock samples (NVML, 50 ms), SM clock
lock state.  Every pass's tokens are compared with the first pass of the same arm (determinism) and the
arm-C/D token streams with arm B (recording must not change outputs).

    # phase-2 commands: see docs/vllm-poc/minimal-recording-design.md §5
    python scripts/minimal_recording_ab.py run --config profile --plan layer --shape b0 --reps 30 --out results/MINREC/...
    python scripts/minimal_recording_ab.py run --config stock   --shape b0 --reps 30 --out results/MINREC/...
    python scripts/minimal_recording_ab.py summarize results/MINREC/<run>/*/passes.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------------------
# shapes (representative; positions = prompt + outputs − 1 per request)
# --------------------------------------------------------------------------------------------------
SHAPES: dict[str, dict[str, Any]] = {
    "b0": {"n_requests": 4, "prompt_len": 32, "max_tokens": 16, "note": "the B0 workload shape (INT-PAD-2: 4 × 32/16)"},
    "b0x16": {"n_requests": 16, "prompt_len": 32, "max_tokens": 16, "note": "batch pressure at the B0 shape"},
    "long-prompt": {"n_requests": 4, "prompt_len": 512, "max_tokens": 32, "note": "prefill-heavy: boundary rows ∝ positions, few steps"},
    "decode-heavy": {"n_requests": 4, "prompt_len": 32, "max_tokens": 256, "note": "step-heavy: per-step hook overhead shows here"},
    "mixed": {"n_requests": 8, "prompt_len": 128, "max_tokens": 64, "note": "middle of the road"},
}

CONFIGS: dict[str, dict[str, Any]] = {
    # env is applied BEFORE importing vllm (same keys as scripts/replay_trusted_b0.py EXECS)
    "stock": {"env": {"VLLM_BATCH_INVARIANT": "0", "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                      "VLLM_USE_FLASHINFER_SAMPLER": "0"}, "enforce_eager": False, "arms": ["A", "A'"]},
    "profile": {"env": {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                        "VLLM_USE_FLASHINFER_SAMPLER": "0"}, "enforce_eager": True, "arms": ["B", "B'", "C:block", "C:layer", "D"]},
}

ARM_DESC = {
    "A": "stock, no recorder (layer 1 denominator)",
    "A'": "stock + production recorder (tokens, logprobs, step trace)",
    "B": "numerical profile (BI + eager), no recorder (layer 2)",
    "B'": "profile + production recorder (ladder R2')",
    "C:block": "profile + production recorder + BoundaryRecorder(block) — REQUIRED first boundary of P3/cu=block only",
    "C:layer": "profile + production recorder + BoundaryRecorder(layer) — REQUIRED first boundary of the layer plan only",
    "D": "profile + production recorder + full diagnostic capture (capture.py)",
}

# first-boundary rows per plan, as module hook points (first-stage-boundary.md §3.1).  A hook on a module
# receives its OUTPUT; the rows listed are the plan's cross-RU values produced there.
PLAN_HOOKS: dict[str, list[dict[str, Any]]] = {
    "whole": [],  # designated outputs only — the production recorder already records them
    "layer": [
        {"module": "layers.{l}", "rows": ["hidden", "residual"], "note": "decoder-layer output tuple (2 rows/position): block out + residual"},
    ],
    "block": [
        {"module": "layers.{l}.input_layernorm", "rows": ["residual"], "note": "residual forwarded LayerPre→LayerPost (fused add+norm port 1; layer 0: the embedding out)"},
        {"module": "layers.{l}.self_attn.qkv_proj", "rows": ["qkv"], "note": "qkv row (v read by Attention)"},
        {"module": "layers.{l}.self_attn.rotary_emb", "rows": ["q", "k"], "note": "post-RoPE q and k (k also read by later positions' Attention CUs)"},
        {"module": "layers.{l}.self_attn.attn", "rows": ["attn_out"], "note": "Attention → LayerPost"},
        {"module": "layers.{l}", "rows": ["hidden", "residual"], "note": "LayerPost → next LayerPre"},
    ],
    "operator": None,  # every row: arm C == arm D by definition (use --arms D)
}


# --------------------------------------------------------------------------------------------------
# statistics (pure python; ``--selftest`` exercises it)
# --------------------------------------------------------------------------------------------------
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
        20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t975(df: int) -> float:
    if df <= 0:
        return float("inf")
    keys = sorted(T975)
    for k in keys:
        if df <= k:
            return T975[k]
    return 1.96


def paired_ci(x: list[float], y: list[float]) -> dict[str, Any]:
    """95 % t-CI of mean(y − x) and of the ratio mean(y)/mean(x) (Fieller-free: delta method on paired
    differences, adequate at n ≥ 10; the bootstrap CI is also reported)."""
    assert len(x) == len(y) and len(x) >= 2
    n = len(x)
    d = [b - a for a, b in zip(x, y)]
    md, sd = statistics.mean(d), statistics.stdev(d)
    half = t975(n - 1) * sd / math.sqrt(n)
    mx = statistics.mean(x)
    rng = random.Random(20260909)
    boots = []
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(statistics.mean(y[i] for i in idx) / statistics.mean(x[i] for i in idx))
    boots.sort()
    return {
        "n": n, "mean_x": mx, "mean_y": statistics.mean(y), "mean_diff": md, "sd_diff": sd,
        "ci95_diff": [md - half, md + half], "ratio": statistics.mean(y) / mx,
        "ci95_ratio_delta": [1 + (md - half) / mx, 1 + (md + half) / mx],
        "ci95_ratio_bootstrap": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]],
        "resolves_1pct": half / mx < 0.01,
        "n_for_1pct": math.ceil((1.96 * sd / (0.01 * mx)) ** 2) if sd > 0 else 1,
    }


def summarize(paths: list[Path]) -> dict[str, Any]:
    passes = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                passes.append(json.loads(line))
    by = {}
    for r in passes:
        if r.get("warmup") or r.get("discard"):
            continue
        by.setdefault((r["config"], r["shape"], r.get("plan")), {}).setdefault(r["arm"], {})[r["rep"]] = r
    out = {}
    for (config, shape, plan), arms in by.items():
        base = "A" if config == "stock" else "B"
        if base not in arms:
            continue
        reps = sorted(arms[base])
        res = {}
        for arm, byrep in arms.items():
            if arm == base:
                continue
            common = [k for k in reps if k in byrep]
            if len(common) < 2:
                continue
            x = [arms[base][k]["wall_s"] for k in common]
            y = [byrep[k]["wall_s"] for k in common]
            toks = statistics.mean(byrep[k]["prompt_tokens"] + byrep[k]["generated_tokens"] for k in common)
            rec = lambda key: statistics.mean((byrep[k].get("recorder", {}) or {}).get(key, 0) or 0 for k in common)
            ci = paired_ci(x, y)
            res[f"{arm} vs {base}"] = ci | {
                "wall_mean_s": statistics.mean(y), "wall_sd_s": statistics.stdev(y), "wall_base_mean_s": statistics.mean(x),
                "delta_per_token_ms": 1000 * ci["mean_diff"] / toks, "tokens_per_pass": toks,
                "rows_recorded_mean": rec("rows"), "leaves_mean": rec("leaves"),
                "bytes_copied_mean": rec("bytes_copied"), "bytes_hashed_mean": rec("bytes_hashed"), "bytes_written_mean": rec("bytes_written"),
                "production_record_bytes_mean": rec("production_record_bytes"), "steptrace_bytes_mean": rec("steptrace_bytes"),
                "gpu_util_mean": statistics.mean((byrep[k].get("telemetry", {}) or {}).get("gpu_util_mean", 0) or 0 for k in common),
                "gpu_util_base_mean": statistics.mean((arms[base][k].get("telemetry", {}) or {}).get("gpu_util_mean", 0) or 0 for k in common),
                "tokens_identical_to_base": all(byrep[k].get("tokens_digest") == arms[base][k].get("tokens_digest") for k in common),
            }
        res["base"] = {"arm": base, "n": len(reps), "wall_mean_s": statistics.mean(arms[base][k]["wall_s"] for k in reps),
                       "wall_sd_s": statistics.stdev(arms[base][k]["wall_s"] for k in reps) if len(reps) > 1 else None,
                       "tokens_per_pass": statistics.mean(arms[base][k]["prompt_tokens"] + arms[base][k]["generated_tokens"] for k in reps)}
        out[f"{config}/{shape}/{plan}"] = res
    return out


# --------------------------------------------------------------------------------------------------
# GPU telemetry (NVML sampler thread)
# --------------------------------------------------------------------------------------------------
class Telemetry:
    def __init__(self, period_s: float = 0.05):
        self.period = period_s
        self.samples: list[tuple[float, int, int]] = []
        self._stop = threading.Event()
        self._t = None
        self.h = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001
            self.nv = None

    def clocks_locked(self) -> dict[str, Any]:
        if not self.h:
            return {"available": False}
        nv = self.nv
        try:
            return {"available": True, "sm_clock_mhz": nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_SM),
                    "sm_max_mhz": nv.nvmlDeviceGetMaxClockInfo(self.h, nv.NVML_CLOCK_SM),
                    "mem_clock_mhz": nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_MEM),
                    "name": nv.nvmlDeviceGetName(self.h) if isinstance(nv.nvmlDeviceGetName(self.h), str) else nv.nvmlDeviceGetName(self.h).decode(),
                    "note": "lock with `nvidia-smi -lgc <sm> -lmc <mem>` before the run; verify sm_clock ≈ constant across samples"}
        except Exception as e:  # noqa: BLE001
            return {"available": False, "error": repr(e)}

    def start(self):
        self.samples = []
        if not self.h:
            return
        self._stop.clear()

        def loop():
            nv = self.nv
            while not self._stop.is_set():
                try:
                    u = nv.nvmlDeviceGetUtilizationRates(self.h).gpu
                    c = nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_SM)
                    self.samples.append((time.perf_counter(), int(u), int(c)))
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(self.period)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._t:
            self._t.join(timeout=1)
        if not self.samples:
            return {"samples": 0}
        us = [s[1] for s in self.samples]
        cs = [s[2] for s in self.samples]
        return {"samples": len(us), "gpu_util_mean": statistics.mean(us), "gpu_util_max": max(us),
                "sm_clock_mean": statistics.mean(cs), "sm_clock_min": min(cs), "sm_clock_max": max(cs)}


# --------------------------------------------------------------------------------------------------
# BoundaryRecorder: module-level forward hooks that record ONLY the plan's first-boundary rows
# --------------------------------------------------------------------------------------------------
class BoundaryRecorder:
    """Arm C.  Copies the hooked modules' outputs to pinned host memory (async D2H on a side stream),
    hashes each step's rows (sha256, background thread) and optionally appends the raw bytes to a file.
    ``sink`` = none (copy only) | hash | file (copy + hash + write) — three rungs of the required work."""

    def __init__(self, model, plan: str, sink: str, out: Path, log=print):
        import torch  # noqa: F401

        self.plan, self.sink, self.out, self.log = plan, sink, out, log
        self.handles = []
        self.rows = 0
        self.bytes_copied = 0
        self.bytes_written = 0
        self.pending: list[Any] = []
        self.hash_thread = None
        self.digests: list[str] = []
        self._q: list[bytes] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.file = open(out / f"boundary-{plan}.bin", "ab") if sink == "file" else None
        self.stream = torch.cuda.Stream()
        spec = PLAN_HOOKS[plan]
        if spec is None:
            raise SystemExit("operator plan: arm C is arm D (full capture) — run --arms D")
        named = dict(model.named_modules())
        n_layers = sum(1 for n in named if n.startswith("model.layers.") and n.count(".") == 2) or sum(1 for n in named if n.startswith("layers.") and n.count(".") == 1)
        prefix = "model." if any(n.startswith("model.layers.") for n in named) else ""
        for h in spec:
            for l in range(n_layers):
                name = prefix + h["module"].format(l=l)
                mod = named.get(name)
                if mod is None:
                    raise SystemExit(f"hook point {name} not found in the model ({plan}); adjust PLAN_HOOKS")
                self.handles.append(mod.register_forward_hook(self._hook(name, h["rows"])))
        self.log(f"BoundaryRecorder({plan}, sink={sink}): {len(self.handles)} hooks on {n_layers} layers")
        if sink in ("hash", "file"):
            self.hash_thread = threading.Thread(target=self._hasher, daemon=True)
            self.hash_thread.start()

    def _hook(self, name: str, rows: list[str]):
        import torch

        def fn(module, inputs, output):
            outs = output if isinstance(output, (tuple, list)) else (output,)
            with torch.cuda.stream(self.stream):
                for t in outs[: len(rows)]:
                    if not torch.is_tensor(t):
                        continue
                    t2 = t.detach()
                    host = torch.empty(t2.shape, dtype=t2.dtype, pin_memory=True)
                    host.copy_(t2, non_blocking=True)
                    self.pending.append(host)
                    self.rows += t2.shape[0] if t2.dim() > 1 else 1
                    self.bytes_copied += t2.numel() * t2.element_size()

        return fn

    def end_step(self):
        """Call after each engine step (or once per pass): wait for D2H, hand the bytes to the hasher."""
        import torch

        self.stream.synchronize()
        if self.sink == "none":
            self.pending.clear()
            return
        with self._lock:
            for h in self.pending:
                self._q.append(h.numpy().tobytes())
        self.pending.clear()

    def _hasher(self):
        while not self._stop.is_set() or self._q:
            with self._lock:
                items, self._q = self._q, []
            for b in items:
                self.digests.append(hashlib.sha256(b).hexdigest())
                if self.file is not None:
                    self.file.write(b)
                    self.bytes_written += len(b)
            if not items:
                time.sleep(0.001)

    def close(self) -> dict[str, Any]:
        self.end_step()
        self._stop.set()
        if self.hash_thread:
            self.hash_thread.join()
        for h in self.handles:
            h.remove()
        if self.file:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()
        return {"plan": self.plan, "sink": self.sink, "rows": self.rows, "bytes_copied": self.bytes_copied,
                "bytes_written": self.bytes_written, "digests": len(self.digests)}


# --------------------------------------------------------------------------------------------------
# the run (pod only)
# --------------------------------------------------------------------------------------------------
@dataclass
class PassResult:
    config: str
    arm: str
    shape: str
    plan: str | None
    rep: int
    order: int
    wall_s: float
    prompt_tokens: int
    generated_tokens: int
    tokens_digest: str
    recorder: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    max_rss_mb: float | None = None
    warmup: bool = False
    discard: bool = False


def synthetic_requests(shape: dict[str, Any], vocab: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)

    class Req:
        def __init__(self, i, ids, mt):
            self.request_id = f"minrec-{i}"
            self.prompt_token_ids = ids
            self.prompt_len = len(ids)
            self.max_tokens = mt

    return [Req(i, rng.integers(3, vocab - 1, size=shape["prompt_len"]).tolist(), shape["max_tokens"]) for i in range(shape["n_requests"])]


def run(args) -> int:
    cfg = CONFIGS[args.config]
    for k in ("VLLM_USE_V2_MODEL_RUNNER", "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_BATCH_INVARIANT"):
        os.environ.pop(k, None)
    os.environ.update(cfg["env"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import resource

    import torch
    from vllm import LLM  # noqa: F401
    from verity_vllm_adapter import profile as prof
    from verity_vllm_adapter.boundary import PLAN_BOUNDARY, BoundaryRecorder
    from verity_vllm_adapter.driver import run_requests
    from verity_vllm_adapter.production import build_warmup, greedy_params, params_dict
    from verity_vllm_adapter.steptrace import StepTrace

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "log.txt", "a")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    tel = Telemetry()
    log("clocks:", json.dumps(tel.clocks_locked()))
    if args.require_locked_clocks and not tel.clocks_locked().get("available"):
        raise SystemExit("--require-locked-clocks: NVML not available")

    cp = prof.checkpoint_for(args.case, args.checkpoints)
    prof.verify_local_weights(cp)
    kw = prof.engine_kwargs(cp, replay=False, enforce_eager=cfg["enforce_eager"])
    kw["enable_prefix_caching"] = False  # the recorded pass forbids prefix hits (D1 §1); same for every arm
    t0 = time.perf_counter()
    llm = LLM(**kw)
    log(f"engine {args.config} built in {time.perf_counter() - t0:.1f}s; kwargs {json.dumps({k: str(v) for k, v in kw.items() if k != 'model'})}")
    runner = prof.get_model_runner(llm)  # the adapter's accessors (selective_prover.py uses the same)
    scheduler = prof.get_scheduler(llm)
    model = runner.model
    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 49152)) if model is not None else 49152

    if args.workload:  # the B0 request set (e2e/workload-B0-small.json): the prompts every B0 record served
        from verity_vllm_sampler.workload import load_workload

        wl = load_workload(args.workload)
        reqs = list(wl.requests)
        shape = {"n_requests": len(reqs), "prompt_len": sorted({r.prompt_len for r in reqs}), "max_tokens": sorted({r.max_tokens for r in reqs}),
                 "note": f"workload {args.workload}"}
        args.shape = "workload"
    else:
        shape = SHAPES[args.shape]
        reqs = synthetic_requests(shape, vocab, seed=20260909)
    arms = args.arms or cfg["arms"]
    # arm names: A, A' (stock); B, B', D (profile); C:<plan> = B' + BoundaryRecorder(plan) for plan in PLAN_BOUNDARY
    for a in arms:
        if a.startswith("C:") and a[2:] not in PLAN_BOUNDARY:
            raise SystemExit(f"unknown plan in arm {a}")
    if "C:whole" in arms:
        arms = [a for a in arms if a != "C:whole"]
        log("arm C:whole dropped: 'designated outputs only' has no activation rows beyond the production recorder (arm B')")

    prod_json_bytes = {}

    def production_record_bytes(outs_by_id) -> int:
        """Bytes of the production recorder's per-request records (production.jsonl lines): the designated outputs + params."""
        n = 0
        for r in reqs:
            c = outs_by_id[r.request_id].outputs[0]
            lps = [(lp.get(t).logprob if lp.get(t) is not None else None) for t, lp in zip(c.token_ids, c.logprobs or [], strict=False)]
            rec = {"request_id": r.request_id, "prompt_len": r.prompt_len, "prompt_token_ids": list(r.prompt_token_ids),
                   "sampling_params": params_dict(greedy_params(r.max_tokens, 1)), "output_token_ids": list(c.token_ids),
                   "finish_reason": c.finish_reason, "chosen_logprobs": lps}
            n += len(json.dumps(rec)) + 1
        return n

    def one_pass(arm: str, rep: int, order: int, warmup=False) -> PassResult:
        record = arm != "A" and arm != "B"
        trace = None
        brec = None
        cap = None
        saved = None
        if record and scheduler is not None:
            trace = StepTrace()
            trace.install(scheduler)
        if arm.startswith("C:"):
            brec = BoundaryRecorder(runner=runner, model=model, plan=arm[2:], out_dir=out / f"boundary-{arm[2:]}-rep{rep}", sink=args.sink, log=log)
        if arm == "D":
            from verity_vllm_adapter import capture as capmod

            # the diagnostic capture monkeypatches the runner and adds hooks to every module: snapshot so the pass can be undone
            saved = (runner.execute_model, runner.sample, {id(m): (dict(m._forward_pre_hooks), dict(m._forward_hooks)) for m in model.modules()})
            cap = capmod.Capture(runner=runner, model=model, profile_id="0000000000d1", run_dir=out / f"capture-{rep}", writer=None,
                                 hash_only=(args.sink != "file"), prompt_lens={r.request_id: len(r.prompt_token_ids) for r in reqs},
                                 arena_bytes=int(args.arena_gb * (1 << 30)), log=lambda *a: None)
        try:
            llm.reset_prefix_cache()
        except Exception:  # noqa: BLE001
            pass
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        tel.start()
        ev0.record()
        t = time.perf_counter()
        outs_by_id, timing = run_requests(llm, reqs, lambda r: greedy_params(r.max_tokens, 1))
        ev1.record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        gpu_span_s = ev0.elapsed_time(ev1) / 1000.0
        telem = tel.stop()
        rec_info = {}
        if brec is not None:
            s = brec.close()
            rec_info = {k: s[k] for k in ("plan", "first_boundary_rule", "sink", "rows", "bytes_copied", "bytes_hashed", "bytes_written", "leaves",
                                          "root_sha256_merkle", "steps", "slab_overflows", "timing_s")}
        if cap is not None:
            summ = cap.summary()
            rec_info = {"plan": "diagnostic/full capture", "rows": summ.get("leaves"), "leaves": summ.get("leaves"), "bytes_copied": cap.bytes_captured,
                        "bytes_hashed": cap.bytes_captured if args.sink != "file" else 0, "bytes_written": cap.bytes_captured if args.sink == "file" else 0,
                        "sink": args.sink, "timing_s": dict(cap.timing), "errors": len(cap.errors)}
            # undo the diagnostic capture: runner methods, module hooks, the MoE global
            runner.execute_model, runner.sample = saved[0], saved[1]
            for m in model.modules():
                pre, post = saved[2][id(m)]
                for k in list(m._forward_pre_hooks):
                    if k not in pre:
                        del m._forward_pre_hooks[k]
                for k in list(m._forward_hooks):
                    if k not in post:
                        del m._forward_hooks[k]
            capmod._MOE_HOOK.clear()
            del cap
        if trace is not None:
            trace.uninstall()
            rec_info["production_record_bytes"] = production_record_bytes(outs_by_id)
            rec_info["steptrace_bytes"] = len(json.dumps(trace.steps))
            rec_info["engine_steps"] = len(trace.steps)
        toks = [list(outs_by_id[r.request_id].outputs[0].token_ids) for r in reqs]
        digest = hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16]
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
        telem["gpu_span_s"] = gpu_span_s
        return PassResult(args.config, arm, args.shape, args.plan, rep, order, wall, sum(len(r.prompt_token_ids) for r in reqs),
                          sum(len(t) for t in toks), digest, rec_info, telem, rss, warmup)

    passes_f = open(out / "passes.jsonl", "a")

    def emit(pr: PassResult):
        passes_f.write(json.dumps(pr.__dict__) + "\n")
        passes_f.flush()

    # warm-up: synthetic prompts (never the workload's), every arm once, results discarded
    warm, _ = build_warmup(reqs, vocab, 16)
    run_requests(llm, warm, lambda r: greedy_params(r.max_tokens, 1))
    for arm in arms:
        pr = one_pass(arm, -1, -1, warmup=True)
        emit(pr)
        log(f"warm-up {arm}: {pr.wall_s:.3f}s rec={json.dumps({k: v for k, v in pr.recorder.items() if k != 'timing_s'})}")

    rng = random.Random(args.seed)
    first_digest: dict[str, str] = {}
    for rep in range(args.reps):
        order = arms[:]
        rng.shuffle(order)
        for i, arm in enumerate(order):
            pr = one_pass(arm, rep, i)
            pr.discard = rep < args.discard_first
            emit(pr)
            first_digest.setdefault(arm, pr.tokens_digest)
            det = pr.tokens_digest == first_digest[arm]
            log(f"rep {rep:3d} #{i} {arm:8s} {pr.wall_s:.4f}s gpu_span={pr.telemetry.get('gpu_span_s', 0):.4f}s rows={pr.recorder.get('rows', 0)} "
                f"bytes_hashed={pr.recorder.get('bytes_hashed', 0)} util={pr.telemetry.get('gpu_util_mean')} sm={pr.telemetry.get('sm_clock_mean')} det={det}")
    passes_f.close()
    summ = summarize([out / "passes.jsonl"])
    (out / "summary.json").write_text(json.dumps({"config": args.config, "shape": shape, "workload": args.workload, "arms": arms, "reps": args.reps,
                                                   "sink": args.sink, "clocks": tel.clocks_locked(), "gpu": prof.gpu_info() if hasattr(prof, "gpu_info") else None,
                                                   "engine_kwargs": {k: str(v) for k, v in kw.items() if k != "model"}, "results": summ}, indent=1))
    log(json.dumps(summ, indent=1))
    return 0


def dry_run(args) -> int:
    cfg = CONFIGS[args.config]
    arms = args.arms or cfg["arms"]
    shape = SHAPES[args.shape]
    plan = args.plan
    rows = None if plan in (None, "whole") else (PLAN_HOOKS[plan] and sum(len(h["rows"]) for h in PLAN_HOOKS[plan]))
    P = shape["n_requests"] * (shape["prompt_len"] + shape["max_tokens"] - 1)
    print(json.dumps({"config": args.config, "env": cfg["env"], "enforce_eager": cfg["enforce_eager"], "arms": {a: ARM_DESC[a] for a in arms},
                      "shape": shape, "positions": P, "plan": plan, "boundary_rows_per_layer_per_position": rows,
                      "hooks": PLAN_HOOKS.get(plan) if plan else None, "reps": args.reps, "discard_first": args.discard_first,
                      "passes_total": len(arms) * (args.reps + 1), "sink": args.sink}, indent=1))
    return 0


def selftest() -> int:
    rng = random.Random(1)
    x = [0.220 + rng.gauss(0, 0.004) for _ in range(30)]
    y = [xi * 1.012 + rng.gauss(0, 0.001) for xi in x]  # +1.2 % paired effect, mostly common noise
    r = paired_ci(x, y)
    assert r["ci95_ratio_delta"][0] > 1.0, r  # paired design resolves a 1.2 % effect at n=30
    z = [xi + rng.gauss(0, 0.004) for xi in x]  # independent noise of the same size: unresolved
    r2 = paired_ci(x, z)
    assert r2["ci95_ratio_delta"][0] < 1.0 < r2["ci95_ratio_delta"][1], r2
    print(json.dumps({"paired_effect_1.2pct": r, "no_effect_indep_noise": r2}, indent=1))
    print("selftest ok")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="pod only")
    r.add_argument("--config", required=True, choices=sorted(CONFIGS))
    r.add_argument("--arms", nargs="*", default=None, help="subset of the config's arms")
    r.add_argument("--plan", default=None, choices=["whole", "layer", "block", "operator"], help="plan P for arm C")
    r.add_argument("--shape", default="b0", choices=sorted(SHAPES))
    r.add_argument("--case", default="B0")
    r.add_argument("--checkpoints", default="/workspace/checkpoints")
    r.add_argument("--workload", default=None, help="workload JSON (e2e/workload-B0-small.json = the B0 request set); overrides --shape")
    r.add_argument("--reps", type=int, default=30)
    r.add_argument("--discard-first", type=int, default=1, help="repetitions discarded after warm-up")
    r.add_argument("--sink", default="hash", choices=["none", "hash", "file"], help="arm C/D recorder rung")
    r.add_argument("--arena-gb", type=float, default=8.0)
    r.add_argument("--seed", type=int, default=20260909)
    r.add_argument("--require-locked-clocks", action="store_true")
    r.add_argument("--out", required=True)
    r.add_argument("--dry-run", action="store_true", help="print the plan; no vllm import")
    s = sub.add_parser("summarize")
    s.add_argument("paths", nargs="+")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "summarize":
        print(json.dumps(summarize([Path(p) for p in a.paths]), indent=1))
        return 0
    if a.dry_run:
        return dry_run(a)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
