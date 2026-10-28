"""Per-token timing traces of real vLLM serving, for the network certifier's calibration (`network-certifier-calibrate`).

One job (`run.sh`, under `research run --queue` on one leased GPU) starts vLLM's OpenAI-compatible server on that GPU from
the node's local weights, replays recorded request mixes open-loop with a streaming client on the same host, and records
on the monotonic clock when each request was sent and when each streamed chunk arrived. The client stands where a certifier
in front of the server would: every streamed token is one egress frame, ready when its chunk arrives, and a request is
⌈bytes/B⌉ ingress frames that arrive when it is sent.

**The load.** A mix names workloads of `integrations/vllm/workloads/` (`MIXES`) with a period each. Instance k of a
workload starts at k·period plus up to half a period of jitter, and its requests follow at their recorded
`arrival_offset_s`: the rule of the rebuilt traces behind decision D. Several workloads in one mix superpose. Every
request runs to its `max_tokens` (`ignore_eos`). Its first prompt token is replaced by a seeded random one, so that no
instance hits another's prefix cache. The requests sent before t0, the warm-up after model load and CUDA graph capture,
only load the engine. The trace starts at t0 and keeps whole windows.

**Modes.** `plaintext` is vLLM's stock server. `commit` serves the same load from vLLM's engine inside the recorder, built
as the Commit stage builds it (batch-invariant, eager), with `native_collect`, the served committer, hashing every Call
output of every step on the GPU; each window is one commitment run, finalized at the window's end. `bi-eager` is that
engine without the committer. Both stamp the tokens each engine step adds when the step returns, and add `steps.jsonl`
(each step's end, duration and time in garbage collection) and, for `commit`, `commits.jsonl` (each run's finalization
stall). Their monitor runs in a child process, so that the engine's own pauses stay in the trace. Before the replay,
with the schedule and every other input built, the recorder freezes its heap (`gc.freeze()`, `gc` in `meta.json`), so
that full collections don't walk the whole run's requests, which a served engine never holds at once. Offline traces
recorded without `gc` were taken on a higher GC floor (a full collection before t0) and compare
against a different floor than later ones. A step or a finalization that takes `STALL_S` is hung: every thread's stack
goes to stderr and the job exits, rather than holding its GPU to the lease's end.

**What a job writes** to `--out`:
- `requests.jsonl`, one line per request as it finishes, so an evicted job keeps what it finished;
- `samples.jsonl`, every 2 s: the host's load average, the CPU use (ours and everyone else's) of the NUMA nodes of the
  GPU and of the job's CPUs, the GPU's utilisation, clocks and compute processes, and the longest stall of a 10 ms
  heartbeat thread and of the client's event loop (the monitor's own, in the offline modes);
- `meta.json`: host, GPU, CPUs, versions, model, server arguments, the mix and t0;
- `server.log`, and at the end `trace.json` (the calibration format, `verity.core.protocols.certifier.calibration`) and
  `summary.json`.

**Exit status.** 0 when the chunk ran to its deadline. A SIGTERM (an eviction: the lease reclaimed the GPU) stops it;
the job still converts what it recorded, sets `ended.evicted` in `meta.json`, and exits 143, so that a queue requeues
the chunk rather than counting it done.

**`convert`** rebuilds `trace.json` from those files, and drops what isn't honest timing of an idle GPU:
- a heartbeat or event-loop stall longer than `--freeze-ms` (SIGSTOP, a paused scope, a saturated host) drops the
  windows it overlaps and the next one, where the backlog it left drains; a stall longer than `--freeze-rest-ms` drops
  every window from its first to the chunk's end, since nothing shows a longer stall's backlog drains as fast as a short
  one's.
  `--freeze-scope chunk` is the old rule, for sensitivity: any such stall voids the whole chunk;
- a window is dropped when another process computed on our GPU, when the rest of the NUMA node kept more than
  `--max-others-busy` of its CPUs busy or co-tenants more than `--max-cotenant-busy` of the job's own (its cpuset's
  busy share less our processes' CPU time; `--cpu-rule total` counts our use too, against `--max-affinity-busy`), when
  a sample came over `--freeze-ms` late or none came after it, or when one of its requests failed;
- the chunk is cut at dropped windows into contiguous segments, one configuration each.

**`merge`** joins chunks' traces into one for the calibration tool.

    python record.py run --out DIR --model unsloth/Llama-3.2-1B --mix b16=3,b4=1,b64=10 --minutes 55
    python record.py convert DIR [DIR ...]
    python record.py merge OUT.json DIR|TRACE.json ...
"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import gc
import json
import math
import multiprocessing
import os
import random
import resource
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from importlib import metadata
from pathlib import Path

WORKLOADS = Path(__file__).resolve().parents[2] / "integrations/vllm/workloads"
MIXES = {
    "b16": "llama32-1b__bf16__l40s__tp1__b16__i1024__o128__mixed-arrivals__greedy__bi-eager.json",
    "b4": "llama32-1b__bf16__l40s__tp1__b4__i256__o32__mixed-arrivals__greedy__bi-eager.json",
    "b64": "llama32-1b__bf16__l40s__tp1__b64__i1024__o128__mixed__greedy__bi-eager.json",
}
CHECKPOINTS = WORKLOADS.parent / "manifests/checkpoints.json"
ROLES = {"unsloth/Llama-3.2-1B": "LLAMA32_1B", "Qwen/Qwen2.5-0.5B": "QWEN05"}
OFFLINE = {"bi-eager": False, "commit": True}
BUCKET_MS, WINDOW_BUCKETS, FRAME_BYTES, QUEUE_CAPACITY = 100, 600, 4096, 1 << 16
SAMPLE_S, HEARTBEAT_S = 2.0, 0.01
FREEZE_MS, MAX_OTHERS_BUSY, MAX_AFFINITY_BUSY, MAX_COTENANT_BUSY, TAIL_MS = 1000.0, 0.5, 0.9, 0.9, 10_000
FREEZE_REST_MS = 5000.0  # a longer stall drops the rest of its chunk (`convert` in the module docstring)
CPU_RULES, FREEZE_SCOPES = ("cotenants", "total"), ("window", "chunk")
STALL_S = 120  # an offline engine step or finalize this long is hung: every thread's stack goes to stderr and the job exits

clock = time.monotonic


def ms(seconds: float) -> float:
    return round(1000 * seconds, 3)


def percentiles(xs: Sequence[float], ps: Iterable[float] = (50, 99, 99.9)) -> dict:
    """Nearest-rank percentiles and the maximum, as the calibration tool takes them."""
    s = sorted(xs)
    if not s:
        return {}
    return {f"p{p:g}": s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))] for p in ps} | {"max": s[-1]}


def parse_mix(text: str) -> list[tuple[str, float]]:
    """`b16=3,b4=1` -> [("b16", 3.0), ("b4", 1.0)]: workload names of `MIXES` and their periods in seconds."""
    out = []
    for part in text.split(","):
        name, _, period = part.partition("=")
        if name not in MIXES or not period:
            raise ValueError(f"mix entry {part!r}: want NAME=PERIOD_S with NAME in {sorted(MIXES)}")
        out.append((name, float(period)))
    return out


def load_workload(name: str) -> list[dict]:
    from verity_vllm.pipeline.workload_file import workload_file

    reqs = json.loads(workload_file(WORKLOADS / MIXES[name]).read_text())["requests"]
    return [{"index": r["index"], "offset_s": r["arrival_offset_s"], "prompt": r["prompt_token_ids"],
             "max_tokens": r["max_tokens"]} for r in reqs]


def schedule(mix: Sequence[tuple[str, list[dict], float]], start_s: float, end_s: float, seed: int) -> list[dict]:
    """Open-loop arrivals, in seconds from t0: instance k of each workload starts at start_s + k·period + U(0, period/2)
    while that is before end_s, and each of its requests follows at its offset. Each request's first prompt token is
    replaced by a seeded random token, so that no request shares a cached prefix."""
    out = []
    for name, reqs, period in mix:
        rng = random.Random(f"{seed}/{name}")
        k = 0
        while (base := start_s + k * period + rng.uniform(0, period / 2)) < end_s:
            for r in reqs:
                prompt = [rng.randrange(1000, 100_000), *r["prompt"][1:]]
                out.append({"rid": f"{name}.{k}.{r['index']}", "workload": name, "instance": k, "index": r["index"],
                            "at_s": base + r["offset_s"], "prompt": prompt, "max_tokens": r["max_tokens"]})
            k += 1
    return sorted(out, key=lambda q: (q["at_s"], q["rid"]))


class Stream:
    """One request's streamed response: each `data:` line of the server-sent events is stamped on arrival with the
    number of tokens it carries (`return_token_ids`)."""

    def __init__(self, t0: float):
        self.t0, self.chunks, self.done = t0, [], False

    def line(self, raw: bytes) -> None:
        if not raw.startswith(b"data: "):
            return
        t = clock()
        if raw.startswith(b"data: [DONE]"):
            self.done = True
            return
        choices = json.loads(raw[6:]).get("choices") or [{}]
        if n := len(choices[0].get("token_ids") or ()):
            self.chunks.append([ms(t - self.t0), n])


class JsonLines:
    """JSON lines, flushed per line so that a killed job keeps every line written."""

    def __init__(self, path: Path, mode: str = "w"):
        self.fh = open(path, mode, encoding="utf-8")  # noqa: SIM115  (open for the job's life)

    def __call__(self, rec: dict) -> None:
        self.fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


class Collections:
    """A `gc.callbacks` entry that adds up the time this interpreter spends collecting garbage; `take()` returns the
    time since the last take. One collection runs at a time, under the GIL, so starts and stops pair."""

    def __init__(self) -> None:
        self.s = self.began = 0.0

    def __call__(self, phase: str, info: dict) -> None:
        if phase == "start":
            self.began = clock()
        else:
            self.s += clock() - self.began

    def take(self) -> float:
        s, self.s = self.s, 0.0
        return s


def request_body(model: str, q: dict) -> bytes:
    return json.dumps({"model": model, "prompt": q["prompt"], "max_tokens": q["max_tokens"], "temperature": 0.0,
                       "top_p": 1.0, "ignore_eos": True, "stream": True, "return_token_ids": True}).encode()


def request_record(q: dict, body: bytes) -> dict:
    rec = {k: q[k] for k in ("rid", "workload", "instance", "index", "max_tokens")}
    return rec | {"prompt_len": len(q["prompt"]), "request_bytes": len(body), "sched_ms": ms(q["at_s"])}


async def send(session, url: str, model: str, q: dict, t0: float, sink: Callable[[dict], None]) -> None:
    body = request_body(model, q)
    rec = request_record(q, body)
    stream = Stream(t0)
    rec["send_ms"] = ms(clock() - t0)
    try:
        async with session.post(url, data=body, headers={"Content-Type": "application/json"}) as r:
            rec["http"] = r.status
            async for raw in r.content:
                stream.line(raw)
    except Exception as e:  # noqa: BLE001  (recorded: the request's window is dropped)
        rec["error"] = repr(e)[:300]
    rec["chunks"], rec["n_tokens"] = stream.chunks, sum(n for _, n in stream.chunks)
    rec["ok"] = rec.get("http") == 200 and stream.done and rec["n_tokens"] == q["max_tokens"]
    sink(rec)


async def replay(port: int, model: str, requests: list[dict], t0: float, drain_until: float, sink, stall: dict) -> dict:
    """Send each request at its time, streaming every response; stop sending on SIGTERM. Returns how it ended."""
    import aiohttp

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    loop.add_signal_handler(signal.SIGTERM, stop.set)

    async def lag() -> None:
        while True:
            t = clock()
            await asyncio.sleep(HEARTBEAT_S)
            stall["loop"] = max(stall["loop"], clock() - t - HEARTBEAT_S)

    watcher = asyncio.create_task(lag())
    url = f"http://127.0.0.1:{port}/v1/completions"
    ended = {"stopped_ms": None}
    timeout = aiohttp.ClientTimeout(total=None, sock_read=300)
    # A fresh connection per request: a reused keep-alive connection the server has just closed fails the request.
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, force_close=True), timeout=timeout) as session:
        tasks = []
        for q in requests:
            if (delay := t0 + q["at_s"] - clock()) > 0:
                try:
                    await asyncio.wait_for(stop.wait(), delay)
                except TimeoutError:
                    pass
            if stop.is_set():
                ended["stopped_ms"] = ms(clock() - t0)
                break
            tasks.append(asyncio.create_task(send(session, url, model, q, t0, sink)))
        _, pending = await asyncio.wait(tasks, timeout=max(1.0, drain_until - clock())) if tasks else (set(), set())
        for t in pending:
            t.cancel()
        ended["unfinished"] = len(pending)
    watcher.cancel()
    return ended


def offline_engine(args: argparse.Namespace, commit: bool, meta: dict, mix: Sequence[tuple[str, list[dict], float]]):
    """vLLM's engine in this process as the Commit stage builds it (`verity_vllm.engine.build`: batch-invariant, eager,
    no prefix cache), and with `commit` the served committer (`native_collect_v2b` under rows-sha512, the Commit's scheme),
    which commits every Call output of every step as hm96-sha512 rows in the step's frame-v3-sha512 tree on the GPU,
    keeping neither the values nor the tree levels, with a 64-byte root per window. At about 1.4 MB of Call outputs per
    token, hashing on host threads (`native_host`) can't keep up with the mixes. The staging is sized from the mix's
    envelope as the Commit stage arms it (`size_arena`) over the windowed worker's device ring (`enable_windows`), and
    refused here, before any request, when it doesn't fit beside the engine."""
    from verity_vllm.engine import vllm_adapter as va

    eng = va.build_engine(CHECKPOINTS, max_num_seqs=args.max_num_seqs, case=ROLES[args.model], execution="bi-eager",
                          engine_args={"max_model_len": args.max_model_len,
                                       "max_num_batched_tokens": args.max_num_batched_tokens,
                                       "gpu_memory_utilization": args.gpu_memory_utilization})
    meta["engine"] = {"kwargs": {k: v for k, v in eng.kwargs.items() if k not in ("model", "tokenizer")},
                      "env": eng.env_flags, "effective": eng.effective, "build_s": round(eng.build_s, 1)}
    if not commit:
        return eng, None
    from verity_vllm.commit.committer.native_collect import NativeCollectCommitter

    com = NativeCollectCommitter(eng, run_id="network-traces", retain_values=False, arena_bytes=None, wrap=True, gather=True,
                                 native_hooks=True, auto_blocks=True, native_worker=True)
    t = clock()
    com.register_weights()
    rows = [r for _, reqs, _ in mix for r in reqs]
    arena = com.size_arena({"batch": args.max_num_seqs, "max_prompt": max(len(r["prompt"]) for r in rows),
                            "max_new": max(r["max_tokens"] for r in rows),
                            "max_num_batched_tokens": args.max_num_batched_tokens, "max_model_len": args.max_model_len})
    windows = com.enable_windows(arena)
    meta["committer"] = {"name": com.name, "commit_scheme": com.commit_scheme, "retain_values": False,
                         "run_s": WINDOW_BUCKETS * BUCKET_MS / 1000, "register_weights_s": round(clock() - t, 1),
                         "arena": {k: arena.get(k) for k in ("decode_block_bytes", "prefill_block_bytes", "blocks_preallocated",
                                                             "step_bound_bytes", "per_token_bytes")}, "windows": windows}
    return eng, com


def offline_replay(eng, com, requests: list[dict], t0: float, drain_until: float, sink, steps, commits) -> dict:
    """Admit each request between engine steps once its time has come, step while any runs, and stamp the tokens each
    step adds at the step's return. A request's send time is its scheduled arrival; `admit_ms` is when the loop took
    it. With a committer, each window is one commitment run: at every window boundary the loop finalizes the run's root
    (waiting for the hashing to drain) and starts the next, and that stall is part of the trace."""
    from verity_vllm.engine.engine_driver import external_id
    from vllm import SamplingParams

    core = eng.llm.llm_engine
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    window_s = WINDOW_BUCKETS * BUCKET_MS / 1000
    pending, live = list(reversed(requests)), {}
    ended: dict = {"stopped_ms": None}
    next_run = t0 if com is not None else math.inf
    if com is not None:
        com.reset()
        com.install()
    paused = Collections()
    gc.callbacks.append(paused)
    try:
        while (now := clock()) < drain_until and (pending or live):
            if stop.is_set():
                ended["stopped_ms"] = ms(now - t0)
                break
            while pending and t0 + pending[-1]["at_s"] <= now:
                q = pending.pop()
                live[q["rid"]] = request_record(q, request_body("served", q)) | {
                    "send_ms": ms(q["at_s"]), "admit_ms": ms(now - t0), "chunks": [], "n_tokens": 0}
                core.add_request(q["rid"], {"prompt_token_ids": q["prompt"]},
                                 SamplingParams(max_tokens=q["max_tokens"], temperature=0.0, ignore_eos=True))
            if now >= next_run:
                faulthandler.dump_traceback_later(STALL_S, exit=True)
                rc = com.finalize()
                faulthandler.cancel_dump_traceback_later()
                done = clock()
                com.uninstall()
                com.reset()
                com.install()
                commits({"t_ms": ms(now - t0), "finalize_ms": ms(done - now), "restart_ms": ms(clock() - done),
                         "steps": len(rc.steps), "bytes": sum(s.n_bytes for s in rc.steps),
                         "root_ready_delay_ms": ms(rc.root_ready_delay_s), "run_root": rc.run_root.hex(),
                         "errors": rc.extra.get("errors_n"), "gc_ms": ms(paused.take()),
                         # the collector's plans the reset before this window dropped, those still pending at its end, and the
                         # window's step that freed the last of them (native_plans.h)
                         **{k: rc.extra.get("native_collect", {}).get(k)
                            for k in ("plans_dropped", "plans_pending", "plans_pending_emptied_step")}})
                next_run += window_s
                continue
            if not core.has_unfinished_requests():
                stop.wait(max(0.0, min(t0 + pending[-1]["at_s"] if pending else drain_until, next_run) - clock()))
                continue
            faulthandler.dump_traceback_later(STALL_S, exit=True)
            outs = core.step()
            faulthandler.cancel_dump_traceback_later()
            t = clock()
            if com is not None and com.errors:
                raise RuntimeError(f"the committer failed a step, keeping its staging block: {com.errors[0]}")
            n = 0
            for o in outs:
                rec = live.get(external_id(o.request_id))
                if rec is None:
                    continue
                if (k := len(o.outputs[0].token_ids) - rec["n_tokens"]) > 0:
                    rec["chunks"].append([ms(t - t0), k])
                    rec["n_tokens"] += k
                    n += k
                if o.finished:
                    rec["ok"] = rec["n_tokens"] == rec["max_tokens"]
                    sink(live.pop(rec["rid"]))
            steps({"t_ms": ms(t - t0), "dur_ms": ms(t - now), "running": len(live), "tokens": n, "gc_ms": ms(paused.take())})
    finally:
        gc.callbacks.remove(paused)
    ended["unfinished"] = len(live)
    if com is not None:
        com.uninstall()
    return ended


def own_pids(root: int | None = None) -> set[int]:
    import psutil

    me = psutil.Process(root)
    return {me.pid, *(c.pid for c in me.children(recursive=True))}


class Monitor(threading.Thread):
    """Every SAMPLE_S: load, the CPU use of `cpus` (ours and the rest's) and of the job's own CPUs, the GPU's state and
    compute processes, and the longest heartbeat and event-loop stalls since the last sample. In a child process
    (`watch`), `root` is the recorder's pid: the job's processes are its tree, and the monitor stops when it exits."""

    def __init__(self, sink, gpu: dict, cpus: list[int], stall: dict, root: int | None = None, halt=None):
        super().__init__(daemon=True)
        self.sink, self.gpu, self.cpus, self.stall, self.root = sink, gpu, cpus, stall, root
        self.halt = halt if halt is not None else threading.Event()

    def heartbeat(self) -> None:
        while not self.halt.is_set():
            t = clock()
            time.sleep(HEARTBEAT_S)
            self.stall["hb"] = max(self.stall["hb"], clock() - t - HEARTBEAT_S)

    def run(self) -> None:
        import psutil
        import pynvml

        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        mine = pynvml.nvmlDeviceGetHandleByUUID(self.gpu["uuid"])
        threading.Thread(target=self.heartbeat, daemon=True).start()
        procs: dict[int, psutil.Process] = {}
        last: dict[int, float] = {}

        def own_cpu_s() -> float:
            """CPU seconds our processes used since the last call, summed per process: one that exits takes its
            unread time with it, one that starts counts from zero."""
            seen, total = {}, 0.0
            for pid in own_pids(self.root):
                try:
                    c = procs.setdefault(pid, psutil.Process(pid)).cpu_times()
                except psutil.Error:
                    procs.pop(pid, None)
                    continue
                seen[pid] = c.user + c.system
                total += seen[pid] - last.get(pid, 0.0)
            last.clear()
            last.update(seen)
            return total

        aff = sorted(os.sched_getaffinity(0))

        def numa() -> tuple[float, float, float, float]:
            per = psutil.cpu_times(percpu=True)
            busy = [c.user + c.nice + c.system + c.irq + c.softirq + c.steal for c in per]
            total = [sum(c) - c.guest - c.guest_nice for c in per]
            return (sum(busy[i] for i in self.cpus), sum(total[i] for i in self.cpus),
                    sum(busy[i] for i in aff), sum(total[i] for i in aff))

        prev_t, (prev_busy, prev_total, prev_abusy, prev_atotal) = clock(), numa()
        own_cpu_s()
        while not self.halt.wait(SAMPLE_S) and (self.root is None or os.getppid() == self.root):
            try:
                t, (busy, total, abusy, atotal), own = clock(), numa(), own_cpu_s()
                span = max(1e-9, total - prev_total)
                ncpu = len(self.cpus)
                others = max(0.0, (busy - prev_busy) - own) / span
                pids = {p.pid: p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(mine)}
                ours = own_pids(self.root)
                util = pynvml.nvmlDeviceGetUtilizationRates(mine)
                sample = {
                    "mono": t, "dt_s": round(t - prev_t, 3), "load": list(os.getloadavg()),
                    "numa_busy": round((busy - prev_busy) / span, 4), "others_busy": round(others, 4),
                    "own_cores": round(own / max(1e-9, t - prev_t), 3), "cpus": ncpu,
                    "affinity_busy": round((abusy - prev_abusy) / max(1e-9, atotal - prev_atotal), 4),
                    "gpu_util": util.gpu, "gpu_mem_util": util.memory,
                    "sm_mhz": pynvml.nvmlDeviceGetClockInfo(mine, pynvml.NVML_CLOCK_SM),
                    "mem_mhz": pynvml.nvmlDeviceGetClockInfo(mine, pynvml.NVML_CLOCK_MEM),
                    "temp_c": pynvml.nvmlDeviceGetTemperature(mine, pynvml.NVML_TEMPERATURE_GPU),
                    "power_w": round(pynvml.nvmlDeviceGetPowerUsage(mine) / 1000, 1),
                    "throttle": pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(mine),
                    "foreign_pids": sorted(p for p in pids if p not in ours),
                    "gpus_util": [pynvml.nvmlDeviceGetUtilizationRates(h).gpu for h in handles],
                    "hb_max_ms": ms(self.stall["hb"]), "loop_max_ms": ms(self.stall["loop"]),
                }
            except Exception as e:  # noqa: BLE001  (a failed sample is a gap the trace drops; a dead monitor would blind it)
                print(f"monitor: a sample failed: {e!r}", file=sys.stderr, flush=True)
                continue
            self.sink(sample)
            self.stall["hb"] = self.stall["loop"] = 0.0
            prev_t, prev_busy, prev_total, prev_abusy, prev_atotal = t, busy, total, abusy, atotal


def watch(path: Path, gpu: dict, cpus: list[int], root: int, halt) -> None:
    """The offline modes' monitor, in a child process: their engine shares the recorder's interpreter, and its own
    pauses (a garbage collection, a step holding the GIL) are the served system's timing, which a heartbeat in that
    interpreter would take for a frozen host. A stopped scope stops this process too."""
    samples = JsonLines(path)
    try:
        Monitor(samples, gpu, cpus, {"hb": 0.0, "loop": 0.0}, root=root, halt=halt).run()
    finally:
        samples.close()


def cpulist(text: str) -> list[int]:
    """The kernel's CPU list format, `0-3,8` -> [0, 1, 2, 3, 8]."""
    out = []
    for part in text.strip().split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return out


def numa_cpus(node: int) -> list[int]:
    return cpulist(Path(f"/sys/devices/system/node/node{node}/cpulist").read_text())


def identity() -> dict:
    """The host, the leased GPU (CUDA_VISIBLE_DEVICES, PCI bus order), its NUMA node, and the job's CPUs. `watched` are
    the CPUs of the NUMA nodes of the GPU and of the job's CPUs, whose load the monitor reads."""
    import pynvml

    pynvml.nvmlInit()
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    h = pynvml.nvmlDeviceGetHandleByUUID(dev) if dev.startswith("GPU-") else pynvml.nvmlDeviceGetHandleByIndex(int(dev))
    bus = pynvml.nvmlDeviceGetPciInfo(h).busId
    bus = (bus.decode() if isinstance(bus, bytes) else bus).lower()[-12:]
    numa = int(Path(f"/sys/bus/pci/devices/{bus}/numa_node").read_text())
    numa = max(numa, 0)
    gpu = {"index": pynvml.nvmlDeviceGetIndex(h), "uuid": pynvml.nvmlDeviceGetUUID(h), "name": pynvml.nvmlDeviceGetName(h),
           "pci": bus, "numa": numa, "driver": pynvml.nvmlSystemGetDriverVersion()}
    gpu = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in gpu.items()}
    versions = {}
    for dist in ("vllm", "torch", "aiohttp"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    affinity = sorted(os.sched_getaffinity(0))
    nodes = {numa} | {n for n in range(len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))))
                      if set(numa_cpus(n)) & set(affinity)}
    return {"host": socket.gethostname(), "gpu": gpu, "affinity": affinity, "numa_nodes": sorted(nodes),
            "watched": sorted({c for n in nodes for c in numa_cpus(n)}),
            "versions": versions, "python": sys.version.split()[0],
            "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "GPU_LEASE_UUID", "RESEARCH_RUN_ID",
                                                   "GPU_LEASE_WHO", "VLLM_BATCH_INVARIANT")}}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(server: subprocess.Popen, port: int, deadline: float) -> None:
    while clock() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"the vLLM server exited with {server.returncode} before it was ready (server.log)")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError("the vLLM server was not ready before the job's deadline")


def stop_server(server: subprocess.Popen) -> None:
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def evicted(*_) -> None:
    raise SystemExit(143)


def local_model(name: str, hf_home: Path) -> str:
    """A directory as given, or a repo id's snapshot in the Hugging Face cache: `refs/main`'s, else the only one (a
    copied cache can lack `refs`, which an offline vLLM then can't resolve)."""
    if Path(name).is_dir():
        return name
    repo = hf_home / "hub" / ("models--" + name.replace("/", "--"))
    ref = repo / "refs" / "main"
    snaps = [ref.read_text().strip()] if ref.exists() else sorted(p.name for p in (repo / "snapshots").glob("*"))
    if len(snaps) != 1:
        raise RuntimeError(f"{name}: {len(snaps)} snapshots in {repo}, and no refs/main")
    return str(repo / "snapshots" / snaps[0])


def run(args: argparse.Namespace) -> int:
    start = clock()
    deadline = start + 60 * args.minutes
    if args.mode in OFFLINE and args.model not in ROLES:
        raise ValueError(f"--mode {args.mode} builds a pinned checkpoint: one of {sorted(ROLES)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mix = [(name, load_workload(name), period) for name, period in parse_mix(args.mix)]
    meta = identity() | {"mode": args.mode, "model": args.model, "mix": args.mix, "seed": args.seed,
                         "workloads": {name: MIXES[name] for name, _, _ in mix}, "minutes": args.minutes,
                         "warmup_s": args.warmup_s, "grid": {"bucket_ms": BUCKET_MS, "window_buckets": WINDOW_BUCKETS,
                                                             "frame_bytes": FRAME_BYTES}}
    port, served, server, offline = free_port(), "served", None, args.mode in OFFLINE
    model = local_model(args.model, Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")))
    server_argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
                   "--served-model-name", served, "--host", "127.0.0.1", "--port", str(port), "--seed", "0",
                   "--max-model-len", str(args.max_model_len), "--disable-uvicorn-access-log", *args.server_arg]
    if not offline:
        meta["server_argv"] = server_argv[1:]
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    stall = {"hb": 0.0, "loop": 0.0}
    requests = JsonLines(out / "requests.jsonl")
    steps, commits = (JsonLines(out / "steps.jsonl"), JsonLines(out / "commits.jsonl")) if offline else (None, None)
    if offline:
        spawn = multiprocessing.get_context("spawn")
        samples, halt = None, spawn.Event()
        monitor = spawn.Process(target=watch, args=(out / "samples.jsonl", meta["gpu"], meta["watched"], os.getpid(), halt),
                                daemon=True)
    else:
        samples, halt = JsonLines(out / "samples.jsonl"), threading.Event()
        monitor = Monitor(samples, meta["gpu"], meta["watched"], stall, halt=halt)
    monitor.start()
    signal.signal(signal.SIGTERM, evicted)
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))  # a connection per in-flight request, and a backlog has thousands
    if not offline:
        with open(out / "server.log", "wb") as log:
            server = subprocess.Popen(server_argv, stdout=log, stderr=subprocess.STDOUT)
    ended: dict = {"stopped_ms": None, "unfinished": 0}
    try:
        if offline:
            eng, com = offline_engine(args, OFFLINE[args.mode], meta, mix)
        else:
            wait_ready(server, port, deadline - args.drain_s - 120)
        ready = clock()
        t0 = ready + args.warmup_s
        window_s = WINDOW_BUCKETS * BUCKET_MS / 1000
        windows = int((deadline - args.drain_s - t0) // window_s)
        if windows < 1:
            raise RuntimeError(f"the server took {ready - start:.0f} s to start, leaving no whole window")
        reqs = schedule(mix, -args.warmup_s, windows * window_s, args.seed)
        meta |= {"t0_mono": t0, "t0_unix": time.time() + (t0 - clock()), "startup_s": round(ready - start, 1),
                 "windows": windows, "requests": len(reqs)}
        if offline:
            gc.collect()  # cyclic garbage that a freeze caught would never be freed
            gc.freeze()
            meta["gc"] = {"frozen_before_t0": gc.get_freeze_count()}
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        if offline:
            ended = offline_replay(eng, com, reqs, t0, deadline - 20, requests, steps, commits)
        else:
            ended = asyncio.run(replay(port, served, reqs, t0, deadline - 20, requests, stall))
    except SystemExit as e:
        if e.code != 143:
            raise
        ended["stopped_ms"] = ms(clock() - meta.get("t0_mono", clock()))
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        halt.set()
        if server is not None:
            stop_server(server)
        if offline:
            monitor.join(SAMPLE_S + 30)
        for f in (requests, samples, steps, commits):
            if f is not None:
                f.close()
    # Only SIGTERM sets stopped_ms: before the replay (`evicted`), or in it, where it stops sending.
    ended["evicted"] = ended["stopped_ms"] is not None
    meta["ended"] = ended
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    convert(out)
    return 143 if ended["evicted"] else 0


def build_trace(records: Sequence[dict], samples: Sequence[dict], *, name: str, windows: int, labels: dict | None = None,
                stopped_ms: float | None = None, ingress: str = "text", freeze_ms: float = FREEZE_MS,
                max_others_busy: float = MAX_OTHERS_BUSY, max_affinity_busy: float = MAX_AFFINITY_BUSY,
                freeze_scope: str = "window", freeze_rest_ms: float = FREEZE_REST_MS, cpu_rule: str = "cotenants",
                max_cotenant_busy: float = MAX_COTENANT_BUSY,
                affinity_cpus: int | None = None) -> tuple[list[dict], dict]:
    """The calibration configurations of one chunk, and what was kept and dropped.

    `records` are `requests.jsonl`'s and `samples` `samples.jsonl`'s, both with times in ms from t0 (`t_ms` for a sample,
    which covers the `dt_s` before it). Windows are T = WINDOW_BUCKETS buckets of BUCKET_MS from t0; the chunk has
    `windows` of them, fewer when it was stopped at `stopped_ms`. A request's frames:
    - egress: one per token, index λ its position, `anchor` the bucket of its send time and `ready` the first bucket
      that starts after its chunk arrived; in canonical order, by ready bucket, then anchor (an engine's batch puts
      older requests first), then arrival;
    - ingress: ⌈bytes/B⌉ frames at its send time, where bytes is a text prompt's 4·prompt_len + 512 (`ingress="text"`,
      the rebuilt traces' rule) or the token-id JSON body sent (`"body"`).

    A heartbeat or event-loop stall over `freeze_ms` (a stopped process stops its heartbeat too) drops the windows it
    overlaps and the one after, while the backlog it left drains, and one over `freeze_rest_ms` every window from its
    first to the chunk's end. With `freeze_scope="chunk"` a stall from t0 to `TAIL_MS` after the last window, while its
    last responses stream, freezes the chunk: it gives no configuration. A window is dropped when a sample overlapping
    it saw a foreign process on the GPU, the watched CPUs' other users busier than `max_others_busy`, co-tenants busier
    than the job's CPU rule allows, or came over `freeze_ms` late while the heartbeat ran (the host stalled the
    monitor), when it ends after the last sample (the monitor stopped), or when a request sent in it failed; each run of
    kept windows is one configuration, its times rebased to its first window.

    The job's CPU rule: with `cpu_rule="cotenants"`, others' share of the job's `affinity_cpus` CPUs, the sample's
    `affinity_busy` less `own_cores / affinity_cpus` (our processes' CPU time), over `max_cotenant_busy`. Serving that
    saturates its own CPUs is that deployment's honest timing; others taking them is not. Without `affinity_cpus` our use
    can't be told apart, and all of it counts. With `cpu_rule="total"`, `affinity_busy` itself over `max_affinity_busy`."""
    if cpu_rule not in CPU_RULES:
        raise ValueError(f"cpu_rule is one of {CPU_RULES}, not {cpu_rule!r}")
    if freeze_scope not in FREEZE_SCOPES:
        raise ValueError(f"freeze_scope is one of {FREEZE_SCOPES}, not {freeze_scope!r}")

    def cotenants(s: dict) -> float:
        own = s["own_cores"] / affinity_cpus if affinity_cpus else 0.0
        return max(0.0, s.get("affinity_busy", 0.0) - own)

    def crowded(s: dict) -> str | None:
        if cpu_rule == "total":
            return "job's CPUs saturated" if s.get("affinity_busy", 0.0) > max_affinity_busy else None
        return "co-tenants on the job's CPUs" if cotenants(s) > max_cotenant_busy else None

    window_ms = WINDOW_BUCKETS * BUCKET_MS
    if stopped_ms is not None:
        windows = min(windows, max(0, int(stopped_ms // window_ms)))
    end_ms = windows * window_ms
    measured = [s for s in samples if s["t_ms"] >= 0 and s["t_ms"] - 1000 * s["dt_s"] < end_ms + TAIL_MS]
    stalls = [max(s["hb_max_ms"], s["loop_max_ms"]) for s in measured]
    gaps = [1000 * s["dt_s"] - 1000 * SAMPLE_S for s in measured]
    frozen = (freeze_scope == "chunk" and max(stalls, default=0.0) > freeze_ms) or (windows > 0 and not measured)
    dropped: dict[int, str] = {}
    for s, gap, stall in zip(measured, gaps, stalls):
        why = ("foreign GPU process" if s["foreign_pids"] else
               "stalled" if stall > freeze_ms else
               "NUMA node busy" if s["others_busy"] > max_others_busy else
               crowded(s) or ("monitor late" if gap > freeze_ms else None))
        if why:
            lo, hi = s["t_ms"] - max(1000 * s["dt_s"], stall), s["t_ms"]
            last = windows - 1 if stall > freeze_rest_ms else int(hi // window_ms) + (stall > freeze_ms)
            for w in range(max(0, int(lo // window_ms)), min(windows - 1, last) + 1):
                dropped.setdefault(w, why)
    for w in range(int(max((s["t_ms"] for s in measured), default=0.0) // window_ms), windows):
        dropped.setdefault(w, "unmonitored")
    kept = [r for r in records if 0 <= r["send_ms"] < end_ms]
    for r in kept:
        if not r["ok"]:
            dropped.setdefault(int(r["send_ms"] // window_ms), "request failed")
    segments, w = [], 0
    while w < windows and not frozen:
        if w in dropped:
            w += 1
            continue
        lo = w
        while w < windows and w not in dropped:
            w += 1
        segments.append((lo, w))
    configs = []
    for lo, hi in segments:
        base = lo * window_ms
        reqs = [r for r in kept if lo * window_ms <= r["send_ms"] < hi * window_ms]
        frames, arrivals = [], []
        for r in reqs:
            anchor = int((r["send_ms"] - base) // BUCKET_MS)
            i = 0
            for t, n in r["chunks"]:
                for _ in range(n):
                    frames.append((int((t - base) // BUCKET_MS) + 1, anchor, t, r["rid"], i))
                    i += 1
            size = 4 * r["prompt_len"] + 512 if ingress == "text" else r["request_bytes"]
            arrivals += [round(r["send_ms"] - base, 3)] * math.ceil(size / FRAME_BYTES)
        frames.sort()
        configs.append({
            "name": f"{name}/w{lo}-{hi}",
            "egress": [{"link": "responses",
                        "frames": [{"session": s, "index": i, "anchor": a, "ready": rd} for rd, a, _, s, i in frames]}],
            "ingress": [{"link": "requests", "arrival_ms": sorted(arrivals)}],
            "labels": (labels or {}) | {"windows": [lo, hi], "ingress_bytes": ingress},
        })
    ttft = [r["chunks"][0][0] - r["send_ms"] for r in kept if r["ok"]]
    itl = [b[0] - a[0] for r in kept if r["ok"] for a, b in zip(r["chunks"], r["chunks"][1:])]
    summary = {
        "name": name, "windows_planned": windows, "frozen": frozen, "freeze_scope": freeze_scope,
        "windows_kept": sum(hi - lo for lo, hi in segments), "segments": segments,
        "windows_dropped": {str(k): v for k, v in sorted(dropped.items())},
        "requests": len(kept), "requests_failed": sum(not r["ok"] for r in kept),
        "frames": sum(len(c["egress"][0]["frames"]) for c in configs),
        "ttft_ms": percentiles(ttft), "inter_chunk_ms": percentiles(itl),
        "tokens_per_chunk_max": max((n for r in kept for _, n in r["chunks"]), default=0),
        "stall_ms": percentiles(stalls, (50, 99)), "stall_ms_max": max(stalls, default=0.0),
        "sample_gap_ms_max": max(gaps, default=0.0),
        "others_busy": percentiles([s["others_busy"] for s in measured], (50, 99)),
        "affinity_busy": percentiles([s.get("affinity_busy", 0.0) for s in measured], (50, 99)),
        "cpu_rule": cpu_rule, "cotenant_busy": percentiles([cotenants(s) for s in measured], (50, 99)),
        "load1": percentiles([s["load"][0] for s in measured], (50, 99)),
        "own_cores": percentiles([s["own_cores"] for s in measured], (50, 99)),
        "sm_mhz": percentiles([s["sm_mhz"] for s in measured], (1, 50)),
        "foreign_pids": sorted({p for s in measured for p in s["foreign_pids"]}),
    }
    return configs, summary


def trace_doc(configs: list[dict]) -> dict:
    return {"bucket_ms": BUCKET_MS, "window_buckets": WINDOW_BUCKETS, "frame_bytes": FRAME_BYTES,
            "queue_capacity": QUEUE_CAPACITY, "configurations": configs}


def read_lines(path: Path) -> list[dict]:
    """JSON lines, without a last line that a kill cut short."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else ():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def convert(out: Path, **rules) -> dict:
    """`trace.json` and `summary.json` of one job's directory, under `build_trace`'s drop rules (`rules`)."""
    meta = json.loads((out / "meta.json").read_text())
    if "t0_mono" not in meta:
        summary = {"frozen": False, "windows_planned": 0, "windows_kept": 0, "error": "the job ended before t0"}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        return summary
    if not (out / "requests.jsonl").exists():
        raise FileNotFoundError(f"{out / 'requests.jsonl'}: missing, though `run` creates it before t0; restore it (an evicted "
                                f"run file's run-files.evicted.json names its artifact) rather than convert every window as idle")
    samples = [s | {"t_ms": ms(s["mono"] - meta["t0_mono"])} for s in read_lines(out / "samples.jsonl")]
    run_id = meta["env"].get("RESEARCH_RUN_ID") or out.name
    labels = {"mode": meta["mode"], "model": meta["model"], "mix": meta["mix"], "host": meta["host"],
              "gpu": meta["gpu"]["name"], "gpu_uuid": meta["gpu"]["uuid"], "run": run_id}
    configs, summary = build_trace(read_lines(out / "requests.jsonl"), samples, name=f"{meta['mode']}/{run_id}",
                                   windows=meta["windows"], labels=labels, stopped_ms=(meta.get("ended") or {}).get("stopped_ms"),
                                   affinity_cpus=len(meta.get("affinity") or ()) or None, **rules)
    summary |= {"labels": labels, "startup_s": meta.get("startup_s"), "ended": meta.get("ended")}
    (out / "trace.json").write_text(json.dumps(trace_doc(configs), separators=(",", ":")))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def merge(paths: Sequence[Path]) -> dict:
    """One trace from chunks' traces (a trace file, or a job's out directory holding trace.json)."""
    configs = []
    for p in paths:
        doc = json.loads((p / "trace.json" if p.is_dir() else p).read_text())
        grid = {k: doc[k] for k in ("bucket_ms", "window_buckets", "frame_bytes", "queue_capacity")}
        if grid != {k: v for k, v in trace_doc([]).items() if k != "configurations"}:
            raise ValueError(f"{p}: grid {grid} differs from this recorder's")
        configs += doc["configurations"]
    return trace_doc(configs)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="record.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="serve on this host's GPU and record one chunk")
    r.add_argument("--out", required=True)
    r.add_argument("--model", required=True, help="a model in the local Hugging Face cache (HF_HOME)")
    r.add_argument("--mix", default="b16=3,b4=1,b64=10", help="NAME=PERIOD_S,... over " + ", ".join(MIXES))
    r.add_argument("--mode", default="plaintext", help="plaintext (the OpenAI server), or one of the offline engine's "
                   + ", ".join(OFFLINE) + f" (models {', '.join(ROLES)})")
    r.add_argument("--minutes", type=float, default=55, help="the whole job, startup included")
    r.add_argument("--warmup-s", type=float, default=90, help="load before t0 that the trace leaves out")
    r.add_argument("--drain-s", type=float, default=60, help="time kept after the last window for in-flight requests")
    r.add_argument("--max-model-len", type=int, default=2048)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--server-arg", action="append", default=[], help="passed to the vLLM server, e.g. --server-arg=--enforce-eager")
    r.add_argument("--max-num-seqs", type=int, default=256, help="the offline engine's")
    r.add_argument("--max-num-batched-tokens", type=int, default=8192, help="the offline engine's (the stock server's own)")
    r.add_argument("--gpu-memory-utilization", type=float, default=0.7,
                   help="the offline engine's, leaving the committer's prefill-sized staging block room beside it")
    c = sub.add_parser("convert", help="rebuild trace.json and summary.json in job directories")
    c.add_argument("dirs", nargs="+", type=Path)
    c.add_argument("--ingress-bytes", choices=("text", "body"), default="text")
    c.add_argument("--freeze-ms", type=float, default=FREEZE_MS)
    c.add_argument("--max-others-busy", type=float, default=MAX_OTHERS_BUSY)
    c.add_argument("--cpu-rule", choices=CPU_RULES, default="cotenants",
                   help="drop a window when co-tenants took over --max-cotenant-busy of the job's CPUs (cotenants), or "
                        "when they were over --max-affinity-busy busy, the job's own use included (total)")
    c.add_argument("--max-cotenant-busy", type=float, default=MAX_COTENANT_BUSY)
    c.add_argument("--max-affinity-busy", type=float, default=MAX_AFFINITY_BUSY)
    c.add_argument("--freeze-scope", choices=FREEZE_SCOPES, default="window",
                   help="a stall over --freeze-ms drops the windows it overlaps and the next (window), or voids the "
                        "whole chunk (chunk, the old rule, for sensitivity)")
    c.add_argument("--freeze-rest-ms", type=float, default=FREEZE_REST_MS,
                   help="with --freeze-scope window, a stall over this drops the rest of its chunk")
    m = sub.add_parser("merge", help="join chunks' traces into one")
    m.add_argument("out", type=Path)
    m.add_argument("inputs", nargs="+", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    if args.cmd == "convert":
        for d in args.dirs:
            s = convert(d, ingress=args.ingress_bytes, freeze_ms=args.freeze_ms, max_others_busy=args.max_others_busy,
                        max_affinity_busy=args.max_affinity_busy, freeze_scope=args.freeze_scope,
                        freeze_rest_ms=args.freeze_rest_ms, cpu_rule=args.cpu_rule,
                        max_cotenant_busy=args.max_cotenant_busy)
            print(f"{d}: kept {s['windows_kept']}/{s['windows_planned']} windows, frozen={s['frozen']}")
        return 0
    args.out.write_text(json.dumps(merge(args.inputs), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
