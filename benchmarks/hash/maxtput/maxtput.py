"""Served vLLM at its maximum throughput, stock and the Commit's arms on the same saturated load (kernels-maxtput).

verity_bench's runner measures a fixed batch (max_num_seqs 128, three to five engines resident at 10 GiB of KV each), so it can't
saturate the card. This driver keeps its engine (`verity_vllm.LLM`, execution ``compiled``: torch.compile with vLLM's own CUDA-graph
mode, prefix caching off) and circuits' arms unchanged (`served_commit.prof_arm`, the loop's `hash.loop.arms`), and changes the load
and the engine's scheduler and memory settings to vLLM's defaults for the card:

- max_num_seqs 1024 and max_num_batched_tokens unset (vLLM's ``LLM``-class defaults on a 70-160 GiB GPU: 1024 and 16384),
  gpu_memory_utilization 0.9 (vLLM's), max_model_len 4096 (the loop's preset);
- the load: ``--requests`` requests, all added at t=0 (``vllm bench serve``'s default request rate, inf), in served_commit's mixed
  rule (`verity_bench.vllm.presets.SweepPoint` traffic ``mixed``: prompts P and P/4 alternating, outputs O and O/2 every third)
  at ``vllm bench serve``'s random-dataset lengths (P 1024, O 128), greedy, ``ignore_eos``;
- every arm in its own engine process, one at a time on the card (no two fit at saturation's KV pool), interleaved by cycle:
  cycle 0 in ``--arms`` order, cycle 1 reversed, so each commit arm has a stock lifetime on both sides. The first stock engine
  sizes its KV pool as vLLM does; every later engine is pinned to the same pool (``kv_cache_memory_bytes``), so every arm
  schedules the same steps and is gated token for token against stock.

Each engine runs a learning run (``gate`` in cycle 0, ``warmup`` after: the committer learns its step plans; a learning run is
never timed), then ``--reps`` timed runs (``measure``), all on the same requests. A run's measures (``run_measures``):

- the steady window: from the first finished request to the last step that began with requests waiting (the queue is never
  empty inside it); output tokens/s, requests/s and scheduled (prefill + decode) tokens/s inside it;
- TTFT (from t=0, so it includes the queue wait, as ``vllm bench serve`` at rate inf) and TPOT ((last token - first) / (n - 1))
  per request, p50 and p99; whole-run output tokens/s, the Commit's finalize included;
- the saturation evidence: waiting requests at each step (its minimum in the window), running requests and KV-cache usage, and
  the GPU's utilization over the window from ``--gpu-log`` (``nvidia-smi --query-gpu=timestamp,utilization.gpu,... -lms``, which
  job.sh runs beside the driver).

    python -m hash.maxtput.maxtput --model qwen3-8b --arms stock,sha512,sha512-layer --requests 2048 --out "$OUT"

Writes OUT/maxtput.json after every engine (so a cap that kills the run keeps what finished) and exits 3 when a gate failed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MODELS = {
    "qwen3-8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "llama31-8b": ("unsloth/Llama-3.1-8B", "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814"),
}
#: the loop's arm names (collect.py's ARM) -> the arm object; stock has none
ARMS = {
    "stock": None,
    "sha512": "served_commit.prof_arm:SERVED_COMMIT_ROWS",
    "sha512-layer": "served_commit.prof_arm:SERVED_COMMIT_ROWS_LAYER",
    "blake3": "hash.loop.arms:SERVED_COMMIT_ROWS_BLAKE3",
}
MAX_NUM_SEQS = 1024


def requests(n: int, prompt: int, output: int, vocab: int) -> list[SimpleNamespace]:
    """served_commit's mixed rule at (prompt, output), every request at t=0, token ids as verity_bench's `_requests` draws them."""
    point = f"maxtput-n{n}-p{prompt}-o{output}"
    seed = int.from_bytes(hashlib.sha256(point.encode()).digest()[:8], "big")
    out = []
    for i in range(n):
        p = prompt if i % 2 == 0 else max(16, prompt // 4)
        o = output if i % 3 else max(8, output // 2)
        ids = [3 + (seed + i * 1_000_003 + j * 97_409) % (vocab - 4) for j in range(p)]
        out.append(SimpleNamespace(request_id=f"{point}-{i}", prompt_token_ids=ids, max_tokens=o, seed=i))
    return out


def pct(values: list[float], q: int) -> float | None:
    if len(values) < 2:
        return values[0] if values else None
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def run_measures(steps: list[dict], first: dict[str, float], done: dict[str, float], ntok: dict[str, int],
                 sched_log: list[tuple[float, int]], end_s: float) -> dict[str, Any]:
    """One run's measures from its per-step records (``t``: seconds from t=0 when the step returned, ``new``: output tokens it
    returned, ``fin``: requests it finished, ``waiting`` / ``running`` / ``kv``: the scheduler's state after it)."""
    gen = sum(ntok.values())
    t_first_finish = min(done.values())
    lo = next(i for i, s in enumerate(steps) if s["t"] >= t_first_finish)
    hi = lo
    for i in range(lo + 1, len(steps)):
        if steps[i - 1]["waiting"] > 0:
            hi = i
    win = steps[lo + 1:hi + 1]
    t0, t1 = steps[lo]["t"], steps[hi]["t"]
    dt = t1 - t0
    ttft = sorted(first.values())
    tpot = sorted((done[r] - first[r]) / (ntok[r] - 1) for r in done if ntok.get(r, 0) > 1)
    sched_tokens = sum(n for t, n in sched_log if t0 < t <= t1)
    m: dict[str, Any] = {
        "requests": len(done), "generated_tokens": gen, "wall_s": end_s, "steps": len(steps),
        "whole_output_tokens_per_s": gen / end_s, "whole_requests_per_s": len(done) / end_s,
        "window": {"from_step": lo, "to_step": hi, "from_s": t0, "to_s": t1, "seconds": dt, "steps": len(win)},
        "ttft_p50_s": pct(ttft, 50), "ttft_p99_s": pct(ttft, 99), "ttft_mean_s": statistics.fmean(ttft),
        "tpot_p50_s": pct(tpot, 50), "tpot_p99_s": pct(tpot, 99), "tpot_mean_s": statistics.fmean(tpot),
        "last_step_s": steps[-1]["t"],
    }
    if dt > 0 and win:
        m["output_tokens_per_s"] = sum(s["new"] for s in win) / dt
        m["requests_per_s"] = sum(s["fin"] for s in win) / dt
        m["scheduled_tokens_per_s"] = sched_tokens / dt
        m["waiting_min"] = min(steps[i - 1]["waiting"] for i in range(lo + 1, hi + 1))
        m["running_mean"] = statistics.fmean(s["running"] for s in win)
        m["running_min"], m["running_max"] = min(s["running"] for s in win), max(s["running"] for s in win)
        kv = [s["kv"] for s in win if s.get("kv") is not None]
        if kv:
            m["kv_usage_mean"], m["kv_usage_max"] = statistics.fmean(kv), max(kv)
        gaps = [win[i]["t"] - win[i - 1]["t"] for i in range(1, len(win))]
        if gaps:
            m["step_s_p50"], m["step_s_max"] = pct(gaps, 50), max(gaps)
    return m


def gpu_util(log: Path | None, start_epoch: float, end_epoch: float) -> dict[str, Any] | None:
    """Mean and minimum of nvidia-smi's utilization.gpu (and the mean SM clock) over [start, end] epoch seconds."""
    if log is None or not log.is_file():
        return None
    util, clk = [], []
    with log.open() as f:
        for row in csv.reader(f):
            if len(row) < 3 or not row[0].strip()[:4].isdigit():
                continue
            try:
                ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                u, c = float(row[1].split()[0]), float(row[2].split()[0])
            except ValueError:
                continue
            if start_epoch <= ts <= end_epoch:
                util.append(u)
                clk.append(c)
    if not util:
        return None
    return {"samples": len(util), "util_mean_pct": statistics.fmean(util), "util_min_pct": min(util),
            "util_p10_pct": pct(sorted(util), 10), "sm_clock_mean_mhz": statistics.fmean(clk)}


class Engine:
    """One arm's engine in this process, as verity_bench's `_VllmEngine` builds it, with vLLM's scheduler defaults."""

    def __init__(self, arm_name: str, model: str, kv_bytes: int | None, gpu_util_frac: float, max_model_len: int) -> None:
        from verity_vllm.engine import env as engine_env

        engine_env.apply_compile_fresh()
        engine_env.apply_engine(None)
        import torch
        import vllm  # noqa: F401
        from verity_bench.vllm.arms import Arm, Target, experimental_arm, plugin
        from verity_vllm.engine.hooks import Hooks, live
        from verity_vllm.pipeline.llm import LLM

        spec = ARMS[arm_name]
        self.arm = experimental_arm(spec) if spec else Arm("stock", precision="bfloat16")
        self.obj = plugin(self.arm)
        if spec and spec.startswith("served_commit.prof_arm:"):
            # RootsArm dumps each distinct run layout to out/layouts.jsonl, ~10 MB a process at saturation: not kept here
            from served_commit import prof_arm

            class _Every(set):
                def __contains__(self, _: object) -> bool:
                    return True

            prof_arm.RootsArm.dumped = _Every()
        self.target = Target(self.arm, None, None, Hooks(tag=f"bench:{self.arm.id}"))
        if self.obj is not None:
            before = {id(e) for e in live()}
            self.obj.install(self.target)
            owned = {id(e) for e in self.target.hooks.entries}
            if [e for e in live() if id(e) not in before and (id(e) not in owned or e.tag != self.target.hooks.tag)]:
                raise RuntimeError(f"arm {self.arm.id!r} patched outside target.hooks")
        name, revision = MODELS[model]
        kw: dict[str, Any] = {"tensor_parallel_size": 1, "max_num_seqs": MAX_NUM_SEQS, "max_model_len": max_model_len,
                              "gpu_memory_utilization": gpu_util_frac, "enforce_eager": False, "execution": "compiled"}
        if kv_bytes is not None:
            kw["kv_cache_memory_bytes"] = int(kv_bytes)
        t0 = time.perf_counter()
        self.llm = LLM(name, revision=revision, **kw)
        self.target.engine, self.target.model = self.llm, self.llm.model
        if self.obj is not None:
            self.obj.ready(self.target)
        self.build_s = time.perf_counter() - t0
        vc = self.llm.vllm.llm_engine.vllm_config
        mc, cc, sc = vc.model_config, vc.cache_config, vc.scheduler_config
        per_token = 2 * mc.get_num_layers(vc.parallel_config) * mc.get_num_kv_heads(vc.parallel_config) * mc.get_head_size() * 2
        self.identity = {
            "arm": arm_name, "arm_id": self.arm.id, "module_spec": spec, "vllm_version": vllm.__version__,
            "gpu": torch.cuda.get_device_name(0), "model": name, "revision": revision,
            "max_num_seqs": int(sc.max_num_seqs), "max_num_batched_tokens": int(sc.max_num_batched_tokens),
            "max_model_len": int(mc.max_model_len), "enable_prefix_caching": bool(cc.enable_prefix_caching),
            "async_scheduling": getattr(sc, "async_scheduling", None), "gpu_memory_utilization": gpu_util_frac,
            "kv_cache_memory_bytes_pinned": kv_bytes, "num_gpu_blocks": int(cc.num_gpu_blocks), "block_size": int(cc.block_size),
            "kv_cache_tokens": int(cc.num_gpu_blocks) * int(cc.block_size),
            "kv_bytes": int(cc.num_gpu_blocks) * int(cc.block_size) * per_token,
            "effective": dict(self.llm.engine_record.get("effective") or {}), "build_s": self.build_s,
            "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
            "windows": dict(getattr(self.obj, "windows", None) or {}),
        }
        self.vocab = int(mc.hf_config.vocab_size)

    def run(self, phase: str, reqs: list[SimpleNamespace]) -> dict[str, Any]:
        import torch
        from vllm import SamplingParams
        from verity_vllm.engine.arrivals import scheduler_of

        eng = self.llm.vllm.llm_engine
        sched = scheduler_of(self.llm.vllm)
        if sched is None:
            raise RuntimeError("the in-process V1 scheduler is not reachable: no saturation record")
        sched_log: list[tuple[float, int]] = []
        orig = sched.schedule
        clock = time.perf_counter
        start = [0.0]

        def schedule(*a: Any, **k: Any) -> Any:
            out = orig(*a, **k)
            sched_log.append((clock() - start[0], int(getattr(out, "total_num_scheduled_tokens", 0))))
            return out

        kvm = getattr(sched, "kv_cache_manager", None)
        self.target.phase, self.target.point = phase, "maxtput"
        if self.obj is not None:
            self.obj.begin_run(self.target)
        for r in reqs:
            eng.add_request(request_id=r.request_id, prompt={"prompt_token_ids": list(r.prompt_token_ids)},
                            params=SamplingParams(max_tokens=r.max_tokens, ignore_eos=True, detokenize=False, seed=r.seed,
                                                  temperature=0.0))
        torch.cuda.synchronize()
        sched.schedule = schedule
        steps: list[dict] = []
        first: dict[str, float] = {}
        done: dict[str, float] = {}
        ntok: dict[str, int] = {}
        tokens: dict[str, tuple[int, ...]] = {}
        epoch0 = time.time()
        start[0] = clock()
        try:
            while eng.has_unfinished_requests():
                outs = eng.step()
                t = clock() - start[0]
                new = fin = 0
                for o in outs:
                    rid = o.request_id
                    n = len(o.outputs[0].token_ids) if o.outputs else 0
                    k = ntok.get(rid, 0)
                    if n > k:
                        if k == 0:
                            first[rid] = t
                        new += n - k
                        ntok[rid] = n
                    if o.finished:
                        done[rid] = t
                        fin += 1
                        tokens[rid] = tuple(int(x) for x in o.outputs[0].token_ids)
                kv = getattr(kvm, "usage", None) if kvm is not None else None
                steps.append({"t": t, "new": new, "fin": fin, "waiting": len(sched.waiting), "running": len(sched.running),
                              "kv": float(kv) if isinstance(kv, (int, float)) else None})
            if self.obj is not None:
                self.obj.end_run(self.target)
            torch.cuda.synchronize()
            end_s = clock() - start[0]
        finally:
            sched.schedule = orig
        if len(done) != len(reqs):
            raise RuntimeError(f"{len(reqs) - len(done)} requests did not finish")
        body = json.dumps({k: list(v) for k, v in sorted(tokens.items())}, separators=(",", ":")).encode()
        rec: dict[str, Any] = {"phase": phase, "epoch0": epoch0, "token_digest": hashlib.sha256(body).hexdigest(),
                               "measures": run_measures(steps, first, done, ntok, sched_log, end_s)}
        w = rec["measures"]["window"]
        rec["window_epoch"] = [epoch0 + w["from_s"], epoch0 + w["to_s"]]
        rec["finalize_s"] = end_s - steps[-1]["t"]
        if self.obj is not None:
            rec["extras"] = self.obj.extras(self.target)
        stride = max(1, len(steps) // 400)
        rec["steps_sampled"] = [{**s, "i": i} for i, s in enumerate(steps) if i % stride == 0 or i == len(steps) - 1]
        return rec

    def gate(self) -> dict[str, Any]:
        if self.obj is None:
            return {"passed": True, "provided": False}
        got = self.obj.gate(self.target)
        return dict(got) if isinstance(got, dict) else {"passed": bool(got)}

    def close(self) -> None:
        try:
            if self.obj is not None:
                self.obj.close()
        finally:
            self.target.hooks.uninstall()
            self.llm.close()


def _child(conn: Any, arm: str, model: str, kv_bytes: int | None, a: dict[str, Any], phases: list[str]) -> None:
    out: dict[str, Any] = {"arm": arm, "runs": []}
    eng = None
    try:
        eng = Engine(arm, model, kv_bytes, a["gpu_util"], a["max_model_len"])
        out["identity"] = eng.identity
        reqs = requests(a["requests"], a["prompt"], a["output"], eng.vocab)
        for phase in phases:
            out["runs"].append(eng.run(phase, reqs))
            r = out["runs"][-1]["measures"]
            print(f"[maxtput] {arm} {phase}: {r.get('output_tokens_per_s', 0):.1f} out tok/s in window "
                  f"({r['window']['seconds']:.1f} s, waiting min {r.get('waiting_min')}), whole {r['whole_output_tokens_per_s']:.1f}, "
                  f"TPOT p50 {1e3 * (r['tpot_p50_s'] or 0):.1f} ms", flush=True)
        out["gate"] = eng.gate()
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["trace"] = traceback.format_exc()[-6000:]
    finally:
        if eng is not None:
            try:
                eng.close()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
    conn.send(out)
    conn.close()


def lifetime(arm: str, model: str, kv_bytes: int | None, a: dict[str, Any], phases: list[str]) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(child, arm, model, kv_bytes, a, phases))
    p.start()
    child.close()
    try:
        got = parent.recv()
    except EOFError:
        got = {"arm": arm, "runs": [], "error": "engine process died"}
    p.join()
    got["exitcode"] = p.exitcode
    return got


def gates(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every engine's own gate, the KV pools equal, and every run's tokens equal to stock's."""
    out = []
    ref = next((r["token_digest"] for e in doc["engines"] if e["arm"] == "stock" for r in e["runs"]), None)
    pool = next((e["identity"]["kv_cache_tokens"] for e in doc["engines"] if e.get("identity")), None)
    for e in doc["engines"]:
        where = {"arm": e["arm"], "cycle": e["cycle"]}
        if e.get("error"):
            out.append({**where, "kind": "engine", "passed": False, "detail": e["error"]})
            continue
        out.append({**where, "kind": "arm", "passed": bool(e["gate"].get("passed")),
                    "detail": {k: e["gate"].get(k) for k in ("runs", "failed", "warmups") if k in e["gate"]}})
        out.append({**where, "kind": "kv-pool", "passed": e["identity"]["kv_cache_tokens"] == pool,
                    "tokens": e["identity"]["kv_cache_tokens"], "control_tokens": pool})
        for r in e["runs"]:
            out.append({**where, "kind": "tokens", "phase": r["phase"], "passed": ref is not None and r["token_digest"] == ref,
                        "digest": r["token_digest"], "control_digest": ref})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=sorted(MODELS), default="qwen3-8b")
    ap.add_argument("--arms", default="stock,sha512,sha512-layer", help=f"comma-separated, stock first; of {','.join(ARMS)}")
    ap.add_argument("--requests", type=int, default=2048)
    ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--output", type=int, default=128)
    ap.add_argument("--reps", type=int, default=1, help="timed runs an engine lifetime")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--gpu-util", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--kv-bytes", type=int, default=None, help="pin every engine's KV pool (default: the first stock engine's)")
    ap.add_argument("--gpu-log", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    if not arms or arms[0] != "stock" or any(x not in ARMS for x in arms):
        ap.error(f"--arms must start with stock and name arms of {sorted(ARMS)}")
    a.out.mkdir(parents=True, exist_ok=True)
    opts = {"requests": a.requests, "prompt": a.prompt, "output": a.output, "gpu_util": a.gpu_util,
            "max_model_len": a.max_model_len}
    doc: dict[str, Any] = {"schema": "kernels-maxtput/v1", "model": a.model, "arms": arms, "options": {**opts, "reps": a.reps,
                           "cycles": a.cycles, "max_num_seqs": MAX_NUM_SEQS},
                           "load": "served_commit mixed rule (prompts P, P/4 alternating; outputs O, O/2 every third) at P, O; "
                                   "all requests at t=0 (request rate inf); greedy, ignore_eos",
                           "engines": []}
    kv = a.kv_bytes
    dst = a.out / "maxtput.json"
    for cycle in range(a.cycles):
        for arm in (arms if cycle % 2 == 0 else arms[::-1]):
            phases = ["gate" if cycle == 0 else "warmup"] + ["measure"] * a.reps
            print(f"[maxtput] cycle {cycle} {arm}: kv pin {kv}", flush=True)
            e = lifetime(arm, a.model, kv, opts, phases)
            e["cycle"] = cycle
            for r in e.get("runs", ()):
                r["gpu"] = gpu_util(a.gpu_log, *r["window_epoch"])
            doc["engines"].append(e)
            if e.get("error"):
                print(f"[maxtput] {arm} cycle {cycle} failed: {e['error']}\n{e.get('trace', '')}", flush=True)
            elif kv is None:
                kv = e["identity"]["kv_bytes"]
            doc["gates"] = gates(doc)
            tmp = dst.with_name(dst.name + ".tmp")
            tmp.write_text(json.dumps(doc, indent=1) + "\n")
            os.replace(tmp, dst)
    ok = all(g["passed"] for g in doc["gates"])
    print(f"[maxtput] gates {'passed' if ok else 'FAILED'}: "
          + "; ".join(f"{g['arm']}/{g['cycle']} {g['kind']}" for g in doc["gates"] if not g["passed"]), flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
