"""The serve tax: what each change to the production server costs on every token, against stock compiled vLLM at the
integration's pin, on the same node, workload and SLO.

    serve_tax.py plan --model-id ID --c4 C4.json.gz --rates NAME=R,... [--span-s 90] [--requests N] --out PLAN.jsonl
    serve_tax.py run --arm ARM --model-id ID --c4 C4.json.gz --rates NAME=R,... [--checkpoint DIR] [--reps 3] [--span-s 90]
                     [--requests N] [--plan PLAN.jsonl] [--window-s S] [--kv-window-s S]
                     [--tick-ms SAT,FREE | --tick-from STEP_TRACE|RECORD_ARM_DIR] --out DIR
    serve_tax.py metrics DIR [--slo-ttft-s S --slo-tpot-s S]
    serve_tax.py table DIR... --out DIR [--slo-ttft-s S --slo-tpot-s S]
    serve_tax.py tick STEP_TRACE|RECORD_ARM_DIR [--load saturated]
    serve_tax.py fit ARM_DIR --budget-s S --reps N
    serve_tax.py checkpoint --model-id ID

`run` serves one arm (`ARMS`) on one server lifetime of vLLM's own OpenAI server (`vllm.entrypoints.openai.api_server`):
a warm-up load, then each named load REPS times, rep by rep (light, medium, saturated, light, ...), each from an empty prefix
cache (the dev server's ``POST /reset_prefix_cache``). A load is the trace corpus's request plan (C4 prompts of 16 to 1,024
tokens, about 60% behind one of eight shared prefixes, caps 64 or 256: `traffic.plan`) under Poisson arrivals at its rate for
``--span-s`` seconds, sent by the corpus's streaming client (`corpus.drive`), every request decoding to its cap
(``ignore_eos``) so that every arm serves the same tokens. A plan whose requests carry their own sampling (the replay
ladder's production plan: ShareGPT, multi-turn and 8k-32k-token prompts, each greedy or at temperature 0.7 with its seed
by the public rule) is served with it, each request stopping at its end of text or its cap (`serve_body`). Each load's DIR/<load>/rep-<r>/client.jsonl holds the client's
record of each request (send, each streamed chunk's time and tokens, done), and meta.json the load's rate, start and count,
and its schedule: the engine's counters on ``/metrics`` before and after it (`SCHEDULE`).

The arms, each one engine configuration; each adds its change to the arm it names as its parent:

- ``compiled``: stock vLLM at the pin, its defaults (torch.compile and CUDA graphs, chunked prefill, prefix caching,
  async scheduling), FP8 where the checkpoint is FP8;
- ``eager``: ``--enforce-eager``, the tax if commits or replay force eager;
- ``bi-compiled``: ``compiled`` with ``VLLM_BATCH_INVARIANT=1``, vLLM's batch-invariant kernels;
- ``bi-eager``: ``eager`` with ``VLLM_BATCH_INVARIANT=1``, the committing path's execution with nothing committed;
- ``record``: the scheduler's step trace (`verity_vllm.engine.step_trace.StepTrace`, the recorder serve-openai uses), in
  the engine core, written by a thread of its own (`servetax_plugin`);
- ``release-order``: requests added in the network certifier's release order (`serve_admit.ReleaseOrder` in the engine
  core, `servetax_plugin.ReleaseGate`), the client standing in for the certifier, which numbers them in send order;
- ``tick``: the certified tick (`serve_admit.TickGate` under vLLM's server, `verity_vllm.pipeline.tick_plugin`), τ_sat and
  τ_free given or calibrated on a ``record`` trace (`tick`), the client standing in for the certifier (`corpus.StandIn`);
- ``tokens``: each window's tokens committed: every prompt id and every sampled id, 4 bytes a token, in ``rows-sha512``
  rows (`verity_vllm.commit.step_rows`), hashed off the engine loop at each window's close (`servetax_plugin.TokenCommit`).
- ``kv``: the paged KV cache committed at window edges (every ``--kv-window-s``): the whole pool, between two steps, in
  ``rows-sha512`` rows of 1 MiB hashed on the GPU on a side stream ordered after the step by an event, with no hook inside
  a CUDA graph (`servetax_plugin.KvEdge`);
- ``tokens-only``: ``record`` with each Commit period's tokens committed as ``tokens`` commits them, without the tick;
- ``boundary``: ``tokens-only`` with the boundary commitment at each replay block's edge (every ``--edges-every-s``
  seconds of a load): the engine drains, the KV pool is hashed where it lies by the row-table kernels under a fresh key
  and copied to pinned host memory (on the ranks ``--kv-copy`` names), and the scheduler's state and each rank's runner
  rows and generators are taken, all kept until the next edge (`servetax_plugin.edges_of`, `boundary.rank_take`).
- ``boundary-rows``: ``record`` with prefix caching off and the two-stage audit's boundary commitment at serve, each
  window's finished requests committed by size class (`verity_vllm.pipeline.boundary_rows`), its object (``check``,
  ``per_prompt``, ``per_output``, ``drained``, ``inline``) ``--boundary-rows``'s JSON, at ``--window-s``; the server is
  held after its loads until its windows' commits account for every finished request (`await_commits`).

`plan` fixes the requests every arm serves, so a job's arms and reps read one file (``--plan``).

`metrics` reads each load's client records: throughput (output tokens and requests a second over the load's span, its
first send to its last done), TTFT and ITL (the gaps between a request's streamed chunks) at p50 and p99, and goodput, the
requests a second that met the SLO (TTFT at most ``--slo-ttft-s`` and time per output token after the first at most
``--slo-tpot-s``), with the GPUs' busy seconds over the load (NVML's utilization integrated: `GpuBusy`) and the edges
that fell in it. `table` lays the arms' metrics side by side per load, as means over reps with 95% Student-t intervals,
and each arm's tax: its compute per token over stock compiled's at the same load, less one (compiled's throughput over
the arm's), against arm ``compiled`` and against its parent; and each arm's edges, their pause and its parts. `fit` says
how many reps of an arm like an earlier one fit in the seconds a job has left.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRACES = ROOT / "benchmarks/network_e2e/traces"
sys.path.insert(0, str(ROOT / "verity" / "configuration" / "config"))
sys.path.append(str(ROOT))
from verity_config.settings import SHARED  # noqa: E402 (from the tree's verity/configuration/config, put on sys.path above)
SCHEMA = "verity-bench/serve-tax/v0"
PLUGIN_ENV = "VERITY_SERVETAX"
TICK_MIDDLEWARE = "verity_vllm.pipeline.tick_plugin.TickFeed"
TICK_PLUGIN = ROOT / "adapters/vllm/verity_vllm/pipeline/tick_plugin.py"
#: two-sided 95% Student-t quantiles by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 8: 2.306, 10: 2.228, 20: 2.086, 30: 2.042}


@dataclass(frozen=True)
class Arm:
    parent: str | None
    what: str
    args: tuple[str, ...] = ()
    plugin: tuple[str, ...] = ()
    tick: bool = False
    #: the server's environment over the job's (VLLM_BATCH_INVARIANT is "0" unless an arm sets it)
    env: dict = field(default_factory=dict)
    #: the corpus's sampling (`corpus.SAMPLING`) and whether each request asks for its sampled tokens' logprobs
    sampling: str = "greedy"
    logprobs: bool = False
    #: a replay block's boundary state taken at each of ``--edges-s`` (`servetax_plugin.edges_of`), whether each edge
    #: also dumps every leaf of the engine core's state for the restore's check (`boundary.dump_core`), and whether each
    #: step's GPU busy time is timed for the replay's comparison (`boundary.Busy`)
    edges: bool = False
    edge_dump: bool = True
    edge_busy: bool = True


BATCH_INVARIANT = {"VLLM_BATCH_INVARIANT": "1"}
ARMS = {
    "compiled": Arm(None, "stock vLLM, compiled (torch.compile, CUDA graphs)"),
    "eager": Arm("compiled", "stock vLLM, eager", ("--enforce-eager",)),
    "bi-compiled": Arm("compiled", "+ vLLM's batch-invariant kernels", env=BATCH_INVARIANT),
    "bi-eager": Arm("eager", "eager with vLLM's batch-invariant kernels: the committing path's execution (TargetProfile's "
                    "bi-eager), nothing committed", ("--enforce-eager",), env=BATCH_INVARIANT),
    "record": Arm("compiled", "+ the scheduler's step trace (StepTrace)", plugin=("record",)),
    "release-order": Arm("record", "+ admission in the certifier's release order", plugin=("record", "release_order")),
    "tick": Arm("record", "+ admission on the certified tick", plugin=("record",), tick=True),
    "tokens": Arm("tick", "+ each window's tokens committed (rows-sha512)", plugin=("record", "tokens"), tick=True),
    "kv": Arm("tokens", "+ the paged KV cache hashed in place at each window edge", plugin=("record", "tokens", "kv"),
              tick=True),
    "restore-base": Arm("record", "+ seeded sampling, each sampled token's logprob returned", plugin=("record",),
                        sampling="seeded", logprobs=True),
    "restore-pause": Arm("restore-base", "+ a replay block's boundary state taken at each edge (drained, copied, hashed, "
                         "kept), as a server would: no leaf dump, no per-step timing", plugin=("record",), sampling="seeded",
                         logprobs=True, edges=True, edge_dump=False, edge_busy=False),
    "restore": Arm("restore-base", "+ a replay block's boundary state taken at each edge (drained, copied, hashed, kept) "
                   "and every leaf of the engine core's state dumped there for the restore's check",
                   plugin=("record",), sampling="seeded", logprobs=True, edges=True),
    "restore-unseeded": Arm("restore", "the restore arm with requests that carry no seed",
                            plugin=("record",), sampling="unseeded", logprobs=True, edges=True),
    "tokens-only": Arm("record", "+ each Commit period's tokens committed (rows-sha512), without the tick",
                       plugin=("record", "tokens")),
    "boundary": Arm("tokens-only", "+ the boundary commitment at each replay block's edge, as a server would take it: "
                    "drained, the KV pool hashed by the row-table kernels and copied to host memory, the scheduler's "
                    "state and each rank's runner rows and generators taken and kept; no leaf dump, no per-step timing",
                    plugin=("record", "tokens"), edges=True, edge_dump=False, edge_busy=False),
    "boundary-rows": Arm("record", "+ the two-stage boundary commitment at serve: each window's finished requests committed "
                         "by size class (boundary_rows), prefix caching off", ("--no-enable-prefix-caching",),
                         plugin=("record", "boundary_rows")),
}
#: what every request's body adds to the corpus's (`corpus`), set by `run` from its arm
BODY_EXTRA: dict = {}
#: the corpus's seeded sampling without the request's seed, as production clients send it: vLLM then samples each
#: request's seed from the engine's numpy generator when the request reaches the model runner
UNSEEDED = "unseeded"


@dataclass(frozen=True)
class Model:
    repo: str
    revision: str
    served: str
    args: tuple[str, ...]
    #: the SLO goodput counts against: (TTFT s, time per output token s)
    slo: tuple[float, float]
    env: dict = field(default_factory=dict)

    def checkpoint(self) -> Path:
        hub = Path(SHARED.HF_HOME or "/workspace/hf") / "hub"
        return hub / f"models--{self.repo.replace('/', '--')}" / "snapshots" / self.revision


#: every arm's engine: vLLM's defaults but for the context (the plan's longest request is 1,024 + 256 tokens), the batch
#: cap (pinned, so that a saturated load hits the same cap on every vLLM build) and the seed
BASE_ARGS = ("--max-model-len", "4096", "--max-num-seqs", "256", "--seed", "0", "--disable-uvicorn-access-log")
MODELS = {
    "moonlight": Model("moonshotai/Moonlight-16B-A3B-Instruct", "4e735b07a89f73647dfab71ab91b840f362ede5b", "moonlight-16b",
                       ("--dtype", "bfloat16", "--trust-remote-code"), (2.0, 0.08)),
    "llama8b": Model("unsloth/Llama-3.1-8B", "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814", "llama-3.1-8b",
                     ("--dtype", "bfloat16"), (2.0, 0.08)),
    "qwen3-30b": Model("Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", "qwen3-30b-a3b",
                       ("--dtype", "bfloat16"), (2.0, 0.08)),
    "glm53": Model("zai-org/GLM-5.3", "aca966e4e02791568aa6a4ced368624b3d897f42", "glm-5.3",
                   ("--tensor-parallel-size", "8"), (2.0, 0.08)),
}


def corpus():
    """The trace corpus's client and plan (`benchmarks/network_e2e/traces`), with every request decoding to its cap."""
    if str(TRACES) not in sys.path:
        sys.path.insert(0, str(TRACES))
    import corpus as C

    if not getattr(C, "_serve_tax", False):
        C.request_body = functools.partial(serve_body, C.request_body)
        C._serve_tax = True
    return C


def serve_body(body: Callable[..., dict], served: str, r: dict, rid: str, sampling: str) -> dict:
    """The corpus's request body (``body``) as the serve tax sends it: decoding to its cap, with `BODY_EXTRA`, and under
    `UNSEEDED` the seeded sampling's parameters without the request's seed. A plan request that carries its own sampling
    (the replay ladder's production plan, `prod_plan`) is sent with it instead, its seed by the public seed rule
    (`corpus.request_seed`) when it is seeded, and stops at its end of text or its cap."""
    if "sampling" in r:
        b = {"model": served, "prompt": r["prompt_ids"], "max_tokens": r["max_tokens"], **r["sampling"], "stream": True,
             "stream_options": {"include_usage": True}, "return_token_ids": True, "user": rid}
        if r.get("seeded"):
            b["seed"] = corpus().request_seed(rid)
        return {**b, **BODY_EXTRA}
    b = body(served, r, rid, "seeded" if sampling == UNSEEDED else sampling)
    if sampling == UNSEEDED:
        b.pop("seed", None)
    return {**b, "ignore_eos": True, **BODY_EXTRA}


# --------------------------------------------------------------------------------------------------------- the server

def plugin_site(site: Path, tick: bool) -> Path:
    """A directory whose .dist-info declares `servetax_plugin.register` and, under the tick, `tick_plugin.register` as vLLM
    general plugins, in that order: the tick's scheduler wrapper is then outside the step trace's, so the trace sees a
    request when the gate adds it."""
    d = site / "verity_serve_tax-0.0.dist-info"
    d.mkdir(parents=True, exist_ok=True)
    (d / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-serve-tax\nVersion: 0.0\n")
    (d / "entry_points.txt").write_text("[vllm.general_plugins]\nverity_serve_tax = servetax_plugin:register\n" +
                                        ("verity_tick = verity_vllm.pipeline.tick_plugin:register\n" if tick else ""))
    return site


def base_args(max_model_len: int | None = None) -> tuple[str, ...]:
    """`BASE_ARGS`, with the context limit ``max_model_len`` where one is given (a plan with longer prompts)."""
    if max_model_len is None:
        return BASE_ARGS
    i = BASE_ARGS.index("--max-model-len")
    return (*BASE_ARGS[:i + 1], str(max_model_len), *BASE_ARGS[i + 2:])


def server_argv(arm: Arm, model: Model, checkpoint: Path, port: int, num_gpu_blocks: int | None = None,
                max_model_len: int | None = None) -> list[str]:
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(checkpoint), "--served-model-name",
            model.served, "--host", "127.0.0.1", "--port", str(port), *model.args, *base_args(max_model_len), *arm.args,
            *(["--num-gpu-blocks-override", str(num_gpu_blocks)] if num_gpu_blocks else []),
            *(["--middleware", TICK_MIDDLEWARE] if arm.tick else [])]


def server_env(arm: Arm, d: Path, site: Path, tick: dict | None, window_s: float, kv_window_s: float,
               edges: dict | None = None, boundary_rows: dict | None = None) -> dict[str, str]:
    env = dict(os.environ, VLLM_SERVER_DEV_MODE="1", VLLM_BATCH_INVARIANT="0",
               PYTHONPATH=os.pathsep.join([str(site), str(HERE), str(ROOT / "adapters/vllm"), str(ROOT),
                                           *SHARED.PYTHONPATH]))
    env.update(arm.env)
    if "boundary_rows" in arm.plugin and not (isinstance(boundary_rows, dict) and boundary_rows.get("check")):
        raise SystemExit("the boundary-rows arm needs --boundary-rows, a JSON object naming its check template "
                         "(run.sh: BOUNDARY_ROWS)")
    if arm.plugin:
        rows = ({"boundary_rows": {"window_s": window_s, **boundary_rows}}
                if "boundary_rows" in arm.plugin and boundary_rows else {})
        env[PLUGIN_ENV] = json.dumps({"out": str(d), "window_s": window_s, "kv_window_s": kv_window_s,
                                      **{k: True for k in arm.plugin}, **(edges if arm.edges and edges else {}), **rows})
    if arm.tick:
        if tick is None:
            raise SystemExit("the tick arms need τ: --tick-ms SAT,FREE or --tick-from STEP_TRACE")
        env["VERITY_TICK"] = json.dumps({k: tick[k] for k in ("bucket_us", "sat_us", "free_us")} | {
            "hold_ms": 5000, "feed": str(d / "tick-feed.jsonl"), "log": str(d)})
    return env


def wait_ready(server: subprocess.Popen, port: int, startup_s: float) -> None:
    deadline, last = time.monotonic() + startup_s, None
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"the server exited with {server.returncode} before it was ready (server.log)")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError as e:
            last = e
        time.sleep(1.0)
    raise RuntimeError(f"the server was not ready in {startup_s:g} s (its health check last: {last!r})")


def reset_prefix_cache(port: int, tries: int = 30) -> bool:
    """Empty the server's prefix cache between loads (a dev-mode route); False if it never could (blocks still held)."""
    for _ in range(tries):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/reset_prefix_cache", method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            if json.loads(r.read() or b"{}").get("success", True):
                return True
        time.sleep(1.0)
    return False


#: a load's schedule, from the engine's counters on the server's Prometheus /metrics (summed over engines): its steps
#: (each step with outputs observes ``iteration_tokens_total`` once), the tokens they scheduled and sampled, the prompt
#: tokens the prefix cache served, and preemptions
SCHEDULE = {"steps": "vllm:iteration_tokens_total_count", "step_tokens": "vllm:iteration_tokens_total_sum",
            "generation_tokens": "vllm:generation_tokens_total", "prompt_tokens": "vllm:prompt_tokens_total",
            "prompt_tokens_cached": "vllm:prompt_tokens_cached_total", "preemptions": "vllm:num_preemptions_total"}


def counters(text: str) -> dict[str, float]:
    """SCHEDULE's counters in a Prometheus text exposition, each summed over its label sets."""
    tot = dict.fromkeys(SCHEDULE.values(), 0.0)
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in tot:
            tot[name] += float(line.rsplit(" ", 1)[1])
    return {k: tot[v] for k, v in SCHEDULE.items()}


def scrape(port: int) -> dict[str, float]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
        return counters(r.read().decode())


# `research cancel` and `--timeout` SIGKILL the job 20 s after its SIGTERM, and the server runs in a session of its own:
# a terminated arm kills the server's group before then, or the engine core and its workers keep the GPUs
TERM_GRACE_S = 5.0


def stop(server: subprocess.Popen, grace_s: float = 120.0) -> str:
    """Stop the server and every process of its group (the engine core), the server given ``grace_s`` to exit on SIGINT:
    ``killed`` if any of the group was left after the server exited, else ``exited``."""
    if server.poll() is None:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(grace_s)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    try:
        os.killpg(server.pid, signal.SIGKILL)
    except ProcessLookupError:
        return "exited"
    return "killed"


# ----------------------------------------------------------------------------------------------------------- the plan

def load_docs(c4: Path, checkpoint: Path, n: int) -> list[list[int]]:
    """The first n documents of a C4 shard as the checkpoint's token ids: its tokenizer.json where it has one (the corpus's
    reader), else its own tokenizer through transformers (Moonlight's tiktoken)."""
    if (checkpoint / "tokenizer.json").exists():
        return corpus().load_docs(c4, checkpoint / "tokenizer.json", n)
    import gzip

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    out = []
    with gzip.open(c4, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(tok.encode(json.loads(line)["text"], add_special_tokens=False))
            if len(out) == n:
                break
    return out


def plan_of(f: Path, c4: Path, checkpoint: Path, n: int, docs: int, seed: int) -> list[dict]:
    """The plan's first n requests from f, made there first (`traffic.plan` over the shard's first ``docs`` documents)
    unless f exists; refused if f holds fewer than n, never replaced."""
    if f.exists():
        plan = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if len(plan) < n:
            raise SystemExit(f"the plan at {f} holds {len(plan)} requests and these loads may send {n}: give a longer "
                             "plan, or fewer --rates, a shorter --span-s or --requests")
        return plan[:n]
    plan = corpus().traffic.plan(load_docs(c4, checkpoint, docs), n, seed)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in plan))
    return plan


def plan_need(rates: dict[str, float], span_s: float, requests: int | None) -> int:
    """Requests a run of these loads may send: a load's arrivals over its span, with room for Poisson's spread."""
    return requests or max(math.ceil(r * span_s * 1.5) + 64 for r in rates.values())


def parse_rates(text: str) -> dict[str, float]:
    out = {}
    for part in filter(None, text.split(",")):
        name, _, r = part.partition("=")
        out[name] = float(r)
    return out


# ------------------------------------------------------------------------------------------------------------ run

def nvml() -> tuple[object | None, str | None]:
    """NVML's bindings (vLLM's own copy, else nvidia-ml-py's), initialized, or None and why not (neither installed, or no
    driver answered)."""
    import importlib
    import importlib.util

    def found(m: str) -> bool:
        parts = m.split(".")
        return all(importlib.util.find_spec(".".join(parts[:k])) is not None for k in range(1, len(parts) + 1))

    name = next((m for m in ("vllm.third_party.pynvml", "pynvml") if found(m)), None)
    if name is None:
        return None, "no NVML bindings (vllm.third_party.pynvml, pynvml)"
    lib = importlib.import_module(name)
    try:
        lib.nvmlInit()
    except Exception as e:  # noqa: BLE001 -- NVMLError's class lives in the module that failed to initialize
        return None, f"{name}.nvmlInit: {e!r}"
    return lib, None


class GpuBusy:
    """The busy GPU-seconds of a load: each visible GPU's NVML utilization (the share of its last sample period in which
    a kernel ran) read every ``period_s`` on a thread of its own, integrated over the time between reads. The visible
    GPUs are ``CUDA_VISIBLE_DEVICES``'s, by index or UUID, else all of the node's."""

    def __init__(self, lib: object | None, visible: str | None = None, period_s: float = 0.2,
                 clock: Callable[[], float] = time.monotonic) -> None:
        import threading

        self.lib, self.period_s, self.clock = lib, period_s, clock
        self.handles: list = []
        if lib is not None:
            ids = [x.strip() for x in (visible or "").split(",") if x.strip()] if visible else \
                [str(i) for i in range(lib.nvmlDeviceGetCount())]
            self.handles = [lib.nvmlDeviceGetHandleByUUID(x) if x.startswith(("GPU-", "MIG-")) else
                            lib.nvmlDeviceGetHandleByIndex(int(x)) for x in ids]
        self.busy = [0.0] * len(self.handles)
        self.last: float | None = None
        self.reads = 0
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def sample(self) -> None:
        now = self.clock()
        util = [self.lib.nvmlDeviceGetUtilizationRates(h).gpu / 100.0 for h in self.handles]
        if self.last is not None:
            for i, u in enumerate(util):
                self.busy[i] += u * (now - self.last)
        self.last = now
        self.reads += 1

    def __enter__(self) -> GpuBusy:
        if self.handles:
            import threading

            self.sample()
            self.thread = threading.Thread(target=self._run, name="serve-tax-gpu-busy", daemon=True)
            self.thread.start()
        return self

    def _run(self) -> None:
        while not self.stop.wait(self.period_s):
            self.sample()

    def __exit__(self, *exc: object) -> None:
        if self.thread is not None:
            self.stop.set()
            self.thread.join()
            self.sample()

    def result(self) -> dict | None:
        if not self.handles:
            return None
        return {"gpus": len(self.handles), "busy_gpu_s": round(sum(self.busy), 3),
                "busy_s_per_gpu": [round(b, 3) for b in self.busy], "reads": self.reads, "period_s": self.period_s}


def drive_load(port: int, served: str, plan: list[dict], rate: float, span_s: float, tag: str, d: Path, drain_s: float,
               tick: dict | None, requests: int | None = None, sampling: str = "greedy", lib: object | None = None) -> dict:
    """One load: the plan's first requests under Poisson arrivals at ``rate`` for ``span_s`` (or ``requests`` of them),
    sent by `corpus.drive`; its client records in d/client.jsonl, and with NVML's bindings ``lib`` the GPUs' busy time
    over it (`GpuBusy`)."""
    C = corpus()
    d.mkdir(parents=True, exist_ok=True)
    arr = C.traffic.arrivals(plan, rate)
    k = requests or max(1, sum(1 for a in arr if a <= span_s))
    reqs, arr = plan[:k], arr[:k]
    ids = [f"{tag}-{r['i']:05d}" for r in reqs]
    t = time.monotonic()
    with GpuBusy(lib, SHARED.CUDA_VISIBLE_DEVICES) as busy:
        got = asyncio.run(C.drive(port, served, reqs, arr, ids, d / "client.jsonl", drain_s, sampling,
                                  *([tick["bucket_us"] / 1e6] if tick else [])))
    meta = {"tag": tag, "rate": rate, "span_s": span_s, "requests": k, "arrivals_span_s": round(arr[-1], 3),
            "wall_s": round(time.monotonic() - t, 3), **got, "gpu": busy.result()}
    recs = [json.loads(x) for x in (d / "client.jsonl").read_text().splitlines() if x.strip()]
    meta["unserved"] = len(recs) - len(served_records(recs))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def await_commits(d: Path, n: int, wait_s: float, clock: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep) -> dict:
    """Hold the server until the boundary-rows arm's sink lines in ``d`` (``boundary.PID.jsonl``, one a class, committed or
    failed) account for ``n`` spans, the requests its loads finished, or ``wait_s`` passes: each window commits on a thread
    of the engine core's own, and the server's stop leaves the engine core seconds."""
    t0 = clock()
    while True:
        got = sum(int(json.loads(x).get("spans", 0)) for f in sorted(d.glob("boundary.*.jsonl"))
                  for x in f.read_text().splitlines() if x.strip())
        if got >= n or clock() - t0 >= wait_s:
            return {"spans": got, "finished": n, "waited_s": round(clock() - t0, 1), "complete": got >= n}
        sleep(1.0)


def open_files() -> int:
    """Raise this process's open-file limit to its hard limit and return it: a saturated load holds a connection open for
    each request in the server's queue, past the usual soft limit of 1,024."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def versions() -> dict:
    from importlib import metadata

    return {dist: next((d.version for d in metadata.distributions(name=dist)), None)
            for dist in ("vllm", "torch", "aiohttp", "transformers")}


def file_sha256(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


#: the ranks that copy their KV pool to host memory at an edge (`boundary.KV_COPY`, which this module doesn't import)
KV_COPY = ("all", "first", "none")


def tp_size(model: Model) -> int:
    args = list(model.args)
    return int(args[args.index("--tensor-parallel-size") + 1]) if "--tensor-parallel-size" in args else 1


def edge_config(arm: Arm, model: Model, a: argparse.Namespace, rates: dict[str, float]) -> dict | None:
    """The plugin's edges (`servetax_plugin.edges_of`) for an arm that takes the boundary state: one each
    ``--edges-every-s`` seconds of each load, or at ``--edges-s`` after the first load's first request; the KV bytes kept
    at ``--save-kv``'s edges (by default the first, at fixed times; none, every so often), which a tensor-parallel engine
    keeps in host memory instead."""
    if not arm.edges:
        return None
    if bool(a.edges_s) == bool(a.edges_every_s):
        raise SystemExit(f"arm {a.arm} takes the boundary state at --edges-s S,S,... or every --edges-every-s S: one of them")
    save = a.save_kv if a.save_kv is not None else ("" if a.edges_every_s else "0")
    save_kv = [int(x) for x in save.split(",") if x.strip()]
    if save_kv and tp_size(model) > 1:
        raise SystemExit(f"{a.model_id} runs at TP {tp_size(model)}: its edges keep the KV bytes in host memory (no --save-kv)")
    when = ({"every_s": a.edges_every_s} if a.edges_every_s else
            {"edges_s": [float(x) for x in a.edges_s.split(",")], "edge_tag": f"-{next(iter(rates))}1-"})
    return {**when, "save_kv": save_kv, "kv_dir": str(Path(a.kv_dir) / a.arm) if a.kv_dir else None, "kv_copy": a.kv_copy,
            "dump": arm.edge_dump, "busy": arm.edge_busy}


def run(a: argparse.Namespace) -> int:
    termed: list[int] = []

    def on_term(s: int, _frame: object) -> None:
        termed.append(s)
        raise SystemExit(128 + s)

    signal.signal(signal.SIGTERM, on_term)
    nofile = open_files()
    arm, model = ARMS[a.arm], MODELS[a.model_id]
    checkpoint = Path(a.checkpoint) if a.checkpoint else model.checkpoint()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rates = parse_rates(a.rates)
    t = time.monotonic()
    plan = plan_of(Path(a.plan) if a.plan else out / "plan.jsonl", Path(a.c4), checkpoint,
                   plan_need(rates, a.span_s, a.requests), a.docs, a.seed)
    print(f"[serve-tax] plan: {len(plan)} requests in {time.monotonic() - t:.1f} s", flush=True)
    tick = tick_args(a.tick_from, a.tick_ms) if arm.tick else None
    srv = out / "server"
    srv.mkdir(exist_ok=True)
    site = plugin_site(out / ".site", arm.tick)
    port = corpus().free_port()
    argv = server_argv(arm, model, checkpoint, port, a.num_gpu_blocks, a.max_model_len)
    BODY_EXTRA.clear()
    BODY_EXTRA.update({"logprobs": 0} if arm.logprobs else {})
    edges = edge_config(arm, model, a, rates)
    lib, no_nvml = nvml()
    if no_nvml:
        print(f"[serve-tax] no busy GPU-seconds: {no_nvml}", flush=True)
    meta = {"schema": SCHEMA, "arm": a.arm, "what": arm.what, "parent": arm.parent, "model_id": a.model_id,
            "model": f"{model.repo}@{model.revision}", "checkpoint": str(checkpoint), "server_argv": argv[1:],
            "plugin": list(arm.plugin), "env": arm.env, "tick": tick, "window_s": a.window_s, "kv_window_s": a.kv_window_s, "rates": rates, "reps": a.reps, "plan": a.plan,
            "plan_sha256": file_sha256(Path(a.plan)) if a.plan and Path(a.plan).is_file() else None,
            "span_s": a.span_s, "c4": a.c4, "seed": a.seed, "sampling": arm.sampling, "logprobs": arm.logprobs, "edges": edges,
            "gpu_busy": lib is not None, "no_gpu_busy": no_nvml, "versions": versions(), "host": os.uname().nodename,
            "run_id": SHARED.RESEARCH_RUN_ID, "gpus": SHARED.CUDA_VISIBLE_DEVICES, "open_files": nofile, "loads": []}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    t = time.monotonic()
    with open(srv / "server.log", "wb") as log:
        server = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                  env=server_env(arm, srv, site, tick, a.window_s, a.kv_window_s, edges, a.boundary_rows),
                                  start_new_session=True)
    rc = 0
    try:
        wait_ready(server, port, a.startup_s)
        meta["startup_s"] = round(time.monotonic() - t, 1)
        print(f"[serve-tax] {a.arm}: up in {meta['startup_s']} s", flush=True)
        finished = 0
        if a.warm_s > 0:
            w = drive_load(port, model.served, plan, min(rates.values()), a.warm_s, "warm", out / "warm", a.drain_s, tick,
                           sampling=arm.sampling)
            finished += w["requests"] - w["unfinished"]
            print(f"[serve-tax] warm-up: {w['requests']} requests, unfinished {w['unfinished']}", flush=True)
        for rep in range(1, a.reps + 1):
            for name, rate in rates.items():
                ok = reset_prefix_cache(port)
                d = out / name / f"rep-{rep}"
                before = scrape(port)
                m = drive_load(port, model.served, plan, rate, a.span_s, f"{name}{rep}", d, a.drain_s, tick, a.requests,
                               arm.sampling, lib)
                after = scrape(port)
                m |= {"load": name, "rep": rep, "prefix_cache_reset": ok,
                      "schedule": {k: after[k] - before[k] for k in after}}
                (d / "meta.json").write_text(json.dumps(m, indent=1))
                meta["loads"].append({k: m[k] for k in ("load", "rep", "rate", "requests", "unfinished", "wall_s", "t0_unix")})
                (out / "meta.json").write_text(json.dumps(meta, indent=1))
                print(f"[serve-tax] {name} rep {rep}: {m['requests']} requests at {rate:g}/s, wall {m['wall_s']} s, "
                      f"unfinished {m['unfinished']}, unserved {m['unserved']}", flush=True)
                finished += m["requests"] - m["unfinished"]
        if "boundary_rows" in arm.plugin:
            meta["boundary_commits"] = await_commits(srv, finished, a.commit_wait_s)
            print(f"[serve-tax] boundary commits: {meta['boundary_commits']}", flush=True)
    except Exception as e:  # noqa: BLE001 -- the record says why the arm stopped
        meta["error"] = f"{type(e).__name__}: {e}"
        print(f"[serve-tax] {a.arm}: FAILED {meta['error']}", flush=True)
        rc = 1
    finally:
        meta["server_stop"] = stop(server, TERM_GRACE_S if termed else 120.0)
        meta["server_rc"] = server.returncode
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        shutil.rmtree(out / ".site", ignore_errors=True)
    return rc


# ------------------------------------------------------------------------------------------------------- the tick's τ

def busy_cycle(recs: list[dict]) -> dict:
    """The engine's mean busy cycle over a step trace with times (`StepTrace` with a clock): from one step's output to the
    next's, between consecutive steps that scheduled tokens, where a request was in the engine (arrived, not yet finished)
    once the first's output was back, as `summary.busy_cycle` reads the corpus's traces."""
    flight: set[str] = set()
    outs: list[tuple[float, bool]] = []
    for r in recs:
        if r.get("kind") == "arrive":
            flight.add(r["request"])
        elif r.get("kind") == "step":
            flight -= set(r.get("finished") or ())
            outs.append((float(r["t"]), bool(flight)))
        elif r.get("kind") == "finish":
            flight.discard(r["request"])
    cycles = [b[0] - a[0] for a, b in zip(outs, outs[1:]) if a[1]]
    return {"cycles": len(cycles), "idle": max(0, len(outs) - 1 - len(cycles)),
            "mean_us": round(statistics.fmean(cycles) * 1e6, 1) if cycles else None}


def tick_args(source: str | None, given_ms: str | None, load: str = "saturated") -> dict:
    """The certified tick's constants in µs: τ_b the corpus's bucket, τ_sat and τ_free the corpus's multiples of the
    mean busy cycle of a ``record`` step trace (a file, or a ``record`` arm's dir: its trace over ``load``'s first rep),
    or given as ``SAT,FREE`` ms."""
    if not TICK_PLUGIN.is_file():
        raise SystemExit(f"the tick arms need the certified tick ({TICK_PLUGIN.name}), which this tree lacks: run them "
                         "from a tree that has --admit tick")
    C = corpus()
    if (source is None) == (given_ms is None):
        raise SystemExit("the tick arms take τ from one of --tick-from STEP_TRACE|RECORD_DIR and --tick-ms SAT,FREE")
    if given_ms is not None:
        sat, free = (float(x) for x in given_ms.split(","))
        return {"bucket_us": C.TICK_BUCKET_MS * 1000, "sat_us": round(sat * 1000), "free_us": round(free * 1000),
                "calibrated_from": None}
    src = Path(source)
    if src.is_dir():
        traces = step_traces(src)
        meta = json.loads((src / "meta.json").read_text())
        at = [x for x in meta["loads"] if x["load"] == load and x["rep"] == 1]
        if not traces or not at:
            raise SystemExit(f"{src}: no step trace or no {load} load to calibrate the tick on")
        recs = window_trace(traces[0], at[0]["t0_unix"], at[0]["t0_unix"] + at[0]["wall_s"])
        where = {"trace": str(traces[0]), "load": load, "t0_unix": at[0]["t0_unix"], "wall_s": at[0]["wall_s"]}
    else:
        recs = trace_records(src)
        where = {"trace": str(src)}
    c = busy_cycle(recs)
    if not c["mean_us"]:
        raise SystemExit(f"{source}: no busy cycle to calibrate the tick on")
    return {"bucket_us": C.TICK_BUCKET_MS * 1000, "sat_us": round(C.TICK_SAT * c["mean_us"]),
            "free_us": round(C.TICK_FREE * c["mean_us"]), "calibrated_from": {**where, **c}}


def step_traces(arm_dir: Path) -> list[Path]:
    """An arm's step traces, one per engine core, plain or gzipped (run.sh gzips them once the job's table is made)."""
    return sorted((arm_dir / "server").glob("step_trace.*.jsonl*"))


def trace_records(trace: Path) -> list[dict]:
    import gzip

    with (gzip.open if trace.suffix == ".gz" else open)(trace, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def window_trace(trace: Path, t0: float, t1: float) -> list[dict]:
    """The step trace's records with times in [t0, t1] (one load's), its header first."""
    return in_window(trace_records(trace), t0, t1)


def in_window(recs: list[dict], t0: float, t1: float) -> list[dict]:
    return recs[:1] + [r for r in recs[1:] if "t" in r and t0 <= float(r["t"]) <= t1]


# ------------------------------------------------------------------------------------------------------------ metrics

def pct(xs: list[float], q: float) -> float | None:
    """The q-quantile of xs by linear interpolation between order statistics (None when empty)."""
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def served_records(recs: list[dict]) -> list[dict]:
    """The client records of requests served to their end."""
    return [r for r in recs if r.get("status") == 200 and not r.get("error") and r.get("done_s") is not None]


def load_metrics(recs: list[dict], slo: tuple[float, float]) -> dict:
    """One load's metrics from its client records (module docstring)."""
    ok = served_records(recs)
    if not ok:
        return {"sent": len(recs), "ok": 0}
    span = max(r["send_unix"] + r["done_s"] for r in ok) - min(r["send_unix"] for r in recs)
    tokens = sum(len(r["output_ids"]) for r in ok)
    ttft = [r["first_token_s"] for r in ok if r.get("first_token_s") is not None]
    itl = [b[0] - a[0] for r in ok for a, b in zip(r["chunks"], r["chunks"][1:])]
    good = 0
    tpots = []
    for r in ok:
        n, ft = len(r["output_ids"]), r.get("first_token_s")
        tpot = (r["done_s"] - ft) / (n - 1) if ft is not None and n > 1 else 0.0
        tpots.append(tpot)
        good += ft is not None and ft <= slo[0] and tpot <= slo[1]
    return {"sent": len(recs), "ok": len(ok), "output_tokens": tokens, "span_s": span,
            "output_tok_per_s": tokens / span, "req_per_s": len(ok) / span, "goodput_req_per_s": good / span,
            "slo_attained": good / len(recs), "ttft_p50_s": pct(ttft, 0.5), "ttft_p99_s": pct(ttft, 0.99),
            "itl_p50_s": pct(itl, 0.5), "itl_p99_s": pct(itl, 0.99), "tpot_p50_s": pct(tpots, 0.5),
            "tpot_p99_s": pct(tpots, 0.99)}


def edge_cost(recs: list[dict], windows: list[tuple[float, float]], before_s: float = 10.0) -> list[dict]:
    """What a load's client saw around each of ``windows`` (an edge's start and pause, in the load's seconds; a record's
    chunk times count from its send): the requests streaming across it (sent before it, done after it) and the longest
    gap between consecutive chunks that overlaps it, each request's; and the TTFT of the requests sent from ``before_s``
    before it to its end. Given another arm's records and the same windows, the same requests at the same moments without
    the edge."""
    ok = served_records(recs)
    out = []
    for t, pause in windows:
        gaps = []
        for r in ok:
            if r["send_s"] < t < r["send_s"] + r["done_s"]:
                ts = [r["send_s"], *(r["send_s"] + c[0] for c in r["chunks"])]
                gaps.append(max((b - a for a, b in zip(ts, ts[1:]) if b >= t and a <= t + pause), default=0.0))
        ttft = [r["first_token_s"] for r in ok
                if t - before_s <= r["send_s"] <= t + pause and r.get("first_token_s") is not None]
        out.append({"t_s": t, "pause_s": pause, "in_flight": len(gaps), "gap_p50_s": pct(gaps, 0.5),
                    "gap_max_s": max(gaps, default=None), "sent": len(ttft), "ttft_p50_s": pct(ttft, 0.5),
                    "ttft_p99_s": pct(ttft, 0.99), "ttft_max_s": max(ttft, default=None)})
    return out


def edge_windows(arm_dir: Path, load: str, rep: int = 1) -> list[tuple[float, float]]:
    """A restore arm's edges as `edge_cost` windows in ``load``'s rep ``rep``'s seconds: each from its drain's end, as
    long as its pause."""
    t0 = json.loads((arm_dir / load / f"rep-{rep}" / "meta.json").read_text())["t0_unix"]
    return [(r["drained_unix"] - t0, r["pause_s"]) for r in edge_records(arm_dir)]


def schedule_metrics(s: dict | None) -> dict:
    """A load's schedule from its counters (`SCHEDULE`): its engine steps, the tokens each scheduled and the rows each
    sampled (one per request that emitted a token), its preemptions and the prompt's share served from the prefix cache."""
    if not s or not s.get("steps"):
        return {}
    return {"steps": s["steps"], "tokens_per_step": s["step_tokens"] / s["steps"],
            "rows_per_step": s["generation_tokens"] / s["steps"], "preemptions": s["preemptions"],
            "prompt_cached_share": s["prompt_tokens_cached"] / s["prompt_tokens"] if s["prompt_tokens"] else None}


def edge_records(arm_dir: Path) -> list[dict]:
    """An arm's edges (`servetax_plugin.edges_of`'s records with a step), in order."""
    recs = [json.loads(x) for f in sorted((arm_dir / "server").glob("edges.*.jsonl"))
            for x in f.read_text().splitlines() if x.strip()]
    return sorted((r for r in recs if "next_step" in r), key=lambda r: r["edge"])


def gpu_metrics(m: dict, output_tokens: int | None) -> dict:
    """A load's busy GPU-seconds (`GpuBusy`), and per thousand output tokens."""
    g = m.get("gpu")
    if not g:
        return {}
    return {"busy_gpu_s": g["busy_gpu_s"],
            "busy_gpu_s_per_ktok": g["busy_gpu_s"] / (output_tokens / 1e3) if output_tokens else None}


def load_edges(edges: list[dict], m: dict) -> dict:
    """The edges that fell in a load (from its first send to its wall time's end): how many, and the engine's stall at
    each, the drain to an empty batch queue plus the pause after it, at its median and longest."""
    t0 = m.get("t0_unix")
    if not edges or t0 is None:
        return {}
    stall = [r["drain_s"] + r["pause_s"] for r in edges if t0 <= r["drained_unix"] <= t0 + m["wall_s"]]
    return {"edges": len(stall), "edge_stall_p50_s": pct(stall, 0.5), "edge_stall_max_s": max(stall, default=None),
            "edge_stall_s": sum(stall)}


def edge_summary(edges: list[dict]) -> dict | None:
    """An arm's edges outside its warm-up: how many, each part's median (seconds, the hash kernels' and the copy's
    milliseconds: their longest rank's at TP > 1), the pause's p99 and longest, the bytes hashed and copied, and whether
    every rank's KV root agreed at every edge."""
    es = [r for r in edges if r.get("load") != "warm"]
    if not es:
        return None

    def med(f: Callable[[dict], float | None]) -> float | None:
        return pct([x for x in map(f, es) if x is not None], 0.5)

    pause = [r["pause_s"] for r in es]
    return {"edges": len(es), "pause_p50_s": pct(pause, 0.5), "pause_p99_s": pct(pause, 0.99), "pause_max_s": max(pause),
            "drain_p50_s": med(lambda r: r["drain_s"]), "state_p50_s": med(lambda r: r["state_s"]),
            "kv_p50_s": med(lambda r: r["kv_s"]), "hash_kernel_ms_p50": med(lambda r: r["kv"]["hash_kernel_ms"]),
            "copy_ms_p50": med(lambda r: r["kv"]["copy_ms"]), "runner_s_p50": med(lambda r: r["kv"].get("runner_s")),
            "stall_p50_s": med(lambda r: r["drain_s"] + r["pause_s"]), "kv_bytes": es[0]["kv"]["bytes"],
            "copied_bytes": es[0]["kv"].get("copied_bytes"), "roots_equal": all(r["kv"].get("roots_equal", True) for r in es)}


def arm_metrics(d: Path, slo: tuple[float, float] | None = None) -> dict:
    """An arm's run dir: its meta and each load's metrics by rep, with the GPUs' busy time and the edges that fell in
    it, and its edges' summary (`edge_summary`)."""
    meta = json.loads((d / "meta.json").read_text())
    slo = slo or MODELS[meta["model_id"]].slo
    edges = edge_records(d)
    loads: dict[str, list[dict]] = {}
    for name in meta["rates"]:
        for rep in sorted((d / name).glob("rep-*"), key=lambda p: int(p.name.split("-")[1])):
            f = rep / "client.jsonl"
            if not f.exists():
                continue
            recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            m = json.loads((rep / "meta.json").read_text())
            lm = load_metrics(recs, slo)
            loads.setdefault(name, []).append({"rep": int(rep.name.split("-")[1]), "rate": m["rate"],
                                               "unfinished": m.get("unfinished"), **lm,
                                               **schedule_metrics(m.get("schedule")),
                                               **gpu_metrics(m, lm.get("output_tokens")), **load_edges(edges, m)})
    return {"schema": SCHEMA + "/metrics", "arm": meta["arm"], "parent": meta.get("parent"), "what": meta.get("what"),
            "run_id": meta.get("run_id"), "model_id": meta["model_id"], "slo": {"ttft_s": slo[0], "tpot_s": slo[1]},
            "startup_s": meta.get("startup_s"), "error": meta.get("error"), "tick": meta.get("tick"), "loads": loads,
            "edges": edge_summary(edges)}


def summary(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    out = {"mean": statistics.fmean(xs), "n": len(xs)}
    if len(xs) >= 2:
        sd = statistics.stdev(xs)
        half = T95[max(k for k in T95 if k <= len(xs) - 1)] * sd / math.sqrt(len(xs))
        out |= {"sd": sd, "ci95": [out["mean"] - half, out["mean"] + half]}
    return out


KEYS = ("output_tok_per_s", "req_per_s", "goodput_req_per_s", "slo_attained", "ttft_p50_s", "ttft_p99_s", "itl_p50_s",
        "itl_p99_s", "steps", "tokens_per_step", "rows_per_step", "preemptions", "prompt_cached_share", "busy_gpu_s",
        "busy_gpu_s_per_ktok", "edges", "edge_stall_p50_s", "edge_stall_max_s", "edge_stall_s")


def table(arms: list[dict]) -> dict:
    """Each arm's metrics per load over its reps, and its tax (module docstring) against ``compiled`` and its parent."""
    by = {a["arm"]: a for a in arms}
    rows = []
    for a in arms:
        for load, reps in a["loads"].items():
            row = {"arm": a["arm"], "load": load, "run_id": a["run_id"], "reps": len(reps),
                   "rate": reps[0]["rate"] if reps else None, "sent": sum(r["sent"] for r in reps),
                   "unserved": sum(r["sent"] - r["ok"] for r in reps), **{k: summary([r.get(k) for r in reps]) for k in KEYS}}
            for ref_name, key in (("compiled", "tax_vs_compiled"), (a.get("parent"), "tax_vs_parent")):
                ref = by.get(ref_name)
                if ref is None or ref_name == a["arm"] or load not in ref["loads"]:
                    continue
                mine = summary([r.get("output_tok_per_s") for r in reps])
                theirs = summary([r.get("output_tok_per_s") for r in ref["loads"][load]])
                good_mine = summary([r.get("goodput_req_per_s") for r in reps])
                good_theirs = summary([r.get("goodput_req_per_s") for r in ref["loads"][load]])
                row[key] = {"of": ref_name, "throughput": tax(theirs, mine), "goodput": tax(good_theirs, good_mine)}
            rows.append(row)
    return {"schema": SCHEMA + "/table", "slo": arms[0]["slo"] if arms else None, "rows": rows,
            "arms": {a["arm"]: {k: a.get(k) for k in ("parent", "what", "run_id", "startup_s", "error", "tick", "edges")}
                     for a in arms}}


def tax(ref: dict, arm: dict) -> dict | None:
    """ref's mean over arm's, less one: the arm's extra compute per token; its interval by the delta method over both
    means' relative standard errors, only where each side has two reps or more (one rep has no spread to go by)."""
    if not ref.get("n") or not arm.get("n") or not arm["mean"]:
        return None
    r = ref["mean"] / arm["mean"]
    out = {"value": r - 1, "n": [ref["n"], arm["n"]]}
    if all(x["n"] >= 2 and x["mean"] for x in (ref, arm)):
        parts = [(x["sd"] / x["mean"]) ** 2 / x["n"] for x in (ref, arm)]
        df = min(ref["n"], arm["n"]) - 1
        half = T95[max(k for k in T95 if k <= df)] * r * math.sqrt(sum(parts))
        out["ci95"] = [r - half - 1, r + half - 1]
    return out


def _f(s: dict, scale: float = 1.0, digits: int = 4) -> str:
    if not s or not s.get("n"):
        return "—"
    return f"{s['mean'] * scale:.{digits}g}"


def _tax(t: dict | None) -> str:
    if not t:
        return "—"
    s = f"{t['value']:+.1%}"
    return s + (f" [{t['ci95'][0]:+.1%}, {t['ci95'][1]:+.1%}]" if "ci95" in t else "")


def markdown(t: dict) -> str:
    slo = t["slo"] or {}
    lines = [f"SLO: TTFT ≤ {slo.get('ttft_s')} s and time per output token ≤ {slo.get('tpot_s')} s. Tax: the arm's compute "
             "per token over the reference's at the same load, less one, from output tokens a second (goodput's beside it). "
             "Unserved: requests sent and not served over the load's reps; a tax beside one isn't over the same requests.",
             "Steps: the engine's steps over the load, with the tokens each scheduled and the rows each sampled.",
             "Busy GPU-s: the GPUs' NVML utilization integrated over the load, summed over GPUs, and per thousand output "
             "tokens. Edges: the replay-block edges in the load, and the engine's stall at each (the drain plus the pause) "
             "at its median and longest.",
             "", "| arm | load | rate (req/s) | unserved | out tok/s | req/s | goodput req/s | SLO met | TTFT p50 / p99 (ms) | "
             "ITL p50 / p99 (ms) | steps | tokens / rows a step | busy GPU-s (per 1k tok) | edges (stall p50 / max, s) | "
             "tax vs compiled (tok/s) | tax vs compiled (goodput) | tax vs parent (tok/s) | run |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in t["rows"]:
        vc, vp = r.get("tax_vs_compiled") or {}, r.get("tax_vs_parent") or {}
        edges = (f"{_f(r.get('edges'), 1, 3)} ({_f(r.get('edge_stall_p50_s'), 1, 3)} / {_f(r.get('edge_stall_max_s'), 1, 3)})"
                 if (r.get("edges") or {}).get("n") else "—")
        lines.append(f"| {r['arm']} | {r['load']} | {r['rate']:.3g} | {r['unserved']} of {r['sent']} | "
                     f"{_f(r['output_tok_per_s'])} | {_f(r['req_per_s'])} | "
                     f"{_f(r['goodput_req_per_s'])} | {_f(r['slo_attained'], 100, 3)}% | "
                     f"{_f(r['ttft_p50_s'], 1e3)} / {_f(r['ttft_p99_s'], 1e3)} | {_f(r['itl_p50_s'], 1e3)} / "
                     f"{_f(r['itl_p99_s'], 1e3)} | {_f(r.get('steps'), 1, 5)} | "
                     f"{_f(r.get('tokens_per_step'), 1, 4)} / {_f(r.get('rows_per_step'), 1, 3)} | "
                     f"{_f(r.get('busy_gpu_s'), 1, 4)} ({_f(r.get('busy_gpu_s_per_ktok'), 1, 3)}) | {edges} | "
                     f"{_tax(vc.get('throughput'))} | {_tax(vc.get('goodput'))} | "
                     f"{_tax(vp.get('throughput'))}{' (' + vp['of'] + ')' if vp else ''} | {r['run_id']} |")
    es = {name: a["edges"] for name, a in t.get("arms", {}).items() if a.get("edges")}
    if es:
        lines += ["", "Each arm's edges outside its warm-up: the pause from the drain's end to the next step, its parts' "
                  "medians (at TP > 1 the longest rank's), the bytes hashed and copied at an edge, and whether every rank's KV "
                  "root agreed.", "",
                  "| arm | edges | pause p50 / p99 / max (s) | drain p50 (s) | state p50 (s) | KV p50 (s) | "
                  "hash kernels p50 (ms) | copy p50 (ms) | runner rows p50 (s) | GB hashed / copied | roots equal |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for name, e in es.items():
            lines.append(f"| {name} | {e['edges']} | {_n(e['pause_p50_s'])} / {_n(e['pause_p99_s'])} / {_n(e['pause_max_s'])} | "
                         f"{_n(e['drain_p50_s'])} | {_n(e['state_p50_s'])} | {_n(e['kv_p50_s'])} | "
                         f"{_n(e['hash_kernel_ms_p50'])} | {_n(e['copy_ms_p50'])} | {_n(e['runner_s_p50'])} | "
                         f"{_n(_gb(e['kv_bytes']), 4)} / {_n(_gb(e['copied_bytes']), 4)} | {'yes' if e['roots_equal'] else 'NO'} |")
    return "\n".join(lines) + "\n"


def _n(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}g}"


def _gb(x: float | None) -> float | None:
    return None if x is None else x / 1e9


#: what an arm spends beside its loads' wall time: each load's prefix-cache reset and /metrics scrapes, and the
#: server's stop
LOAD_OVERHEAD_S, STOP_S = 5.0, 60.0


def reps_that_fit(meta: dict, warm_wall_s: float, budget_s: float, reps: int, margin: float = 1.1) -> int:
    """How many of ``reps`` reps an arm like the one ``meta`` records fits in ``budget_s`` seconds: its startup, its
    warm-up's ``warm_wall_s`` and its longest rep's loads, each ``margin`` times as long; ``reps`` when ``meta`` has no
    rep to go by, else 0 to ``reps``."""
    per_rep: dict[int, float] = {}
    for x in meta.get("loads") or ():
        per_rep[x["rep"]] = per_rep.get(x["rep"], 0.0) + x["wall_s"] + LOAD_OVERHEAD_S
    if not per_rep or meta.get("startup_s") is None:
        return reps
    fixed = (meta["startup_s"] + warm_wall_s + STOP_S) * margin
    return max(0, min(reps, int((budget_s - fixed) // (max(per_rep.values()) * margin))))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=sorted(ARMS))
    r.add_argument("--model-id", required=True, choices=sorted(MODELS))
    r.add_argument("--checkpoint")
    r.add_argument("--c4", required=True)
    r.add_argument("--rates", required=True, help="NAME=REQ_PER_S,... in the order each rep serves them")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--span-s", type=float, default=90.0, help="each load's arrivals span")
    r.add_argument("--requests", type=int, help="each load's request count instead of --span-s's")
    r.add_argument("--warm-s", type=float, default=20.0, help="a warm-up load at the lowest rate first (0: none)")
    r.add_argument("--commit-wait-s", type=float, default=900.0,
                   help="the boundary-rows arm: how long the server is held after its loads for its commits (await_commits)")
    r.add_argument("--drain-s", type=float, default=600.0)
    r.add_argument("--startup-s", type=float, default=1800.0)
    r.add_argument("--window-s", type=float, default=600.0, help="the Commit period the tokens arm closes at")
    r.add_argument("--kv-window-s", type=float, default=600.0, help="the window edge the kv arm hashes the KV cache at")
    r.add_argument("--boundary-rows", type=json.loads, metavar="JSON",
                   help="the boundary-rows arm's object: check, per_prompt, per_output, drained, inline")
    r.add_argument("--plan", help="the plan's file, shared by a job's arms (default: OUT/plan.jsonl)")
    r.add_argument("--docs", type=int, default=20000, help="the shard's documents a plan draws from")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--tick-from", metavar="STEP_TRACE|RECORD_DIR", help="τ for the tick arms (ignored by the others)")
    r.add_argument("--tick-ms", metavar="SAT,FREE")
    r.add_argument("--num-gpu-blocks", type=int, help="the KV pool's blocks (vLLM's --num-gpu-blocks-override)")
    r.add_argument("--max-model-len", type=int, help="the engines' context limit (default: BASE_ARGS's)")
    r.add_argument("--edges-s", metavar="S,S,...", help="the restore arms' edges, seconds after the first load's first request")
    r.add_argument("--edges-every-s", type=float, metavar="S", help="an edge each S seconds of each load's serving instead")
    r.add_argument("--save-kv", metavar="K,...", help="the edges whose KV bytes and state a restore arm keeps "
                   "(default: 0 with --edges-s, none with --edges-every-s)")
    r.add_argument("--kv-dir", help="where a restore arm keeps them, in a directory named for the arm (default: its server/edges)")
    r.add_argument("--kv-copy", default="all", choices=KV_COPY,
                   help="the ranks that copy their KV pool to host memory at an edge (first: rank 0, for ranks that hold the same bytes)")
    r.add_argument("--out", required=True)
    pl = sub.add_parser("plan", help="make the plan a job's arms share, before any GPU is held")
    for flag in ("--model-id", "--checkpoint", "--c4", "--rates", "--out"):
        pl.add_argument(flag, required=flag != "--checkpoint")
    pl.add_argument("--span-s", type=float, default=90.0)
    pl.add_argument("--requests", type=int)
    pl.add_argument("--docs", type=int, default=20000)
    pl.add_argument("--seed", type=int, default=0)
    for name in ("metrics", "table"):
        p = sub.add_parser(name)
        p.add_argument("dirs", nargs="+", type=Path)
        p.add_argument("--slo-ttft-s", type=float)
        p.add_argument("--slo-tpot-s", type=float)
        if name == "table":
            p.add_argument("--out", type=Path)
    k = sub.add_parser("tick")
    k.add_argument("source", metavar="STEP_TRACE|RECORD_DIR")
    k.add_argument("--load", default="saturated")
    e = sub.add_parser("edges", help="each edge of a restore arm as its client saw it, beside other arms at the same moments")
    e.add_argument("arm", type=Path)
    e.add_argument("others", nargs="*", type=Path)
    e.add_argument("--load", default="medium")
    e.add_argument("--rep", type=int, default=1)
    f = sub.add_parser("fit", help="how many reps of an arm like ARM_DIR's fit in the seconds left (reps_that_fit)")
    f.add_argument("arm", type=Path, metavar="ARM_DIR")
    f.add_argument("--budget-s", type=float, required=True)
    f.add_argument("--reps", type=int, required=True)
    c = sub.add_parser("checkpoint", help="the model's pinned snapshot under HF_HOME")
    c.add_argument("--model-id", required=True, choices=sorted(MODELS))
    a = ap.parse_args(argv)
    if a.cmd == "edges":
        windows = edge_windows(a.arm, a.load, a.rep)
        for d in (a.arm, *a.others):
            f = d / a.load / f"rep-{a.rep}" / "client.jsonl"
            recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            for k_, row in enumerate(edge_cost(recs, windows)):
                print(json.dumps({"arm": d.name, "edge": k_, **row}))
        return 0
    if a.cmd == "run":
        return run(a)
    if a.cmd == "fit":
        meta = json.loads((a.arm / "meta.json").read_text())
        warm = a.arm / "warm" / "meta.json"
        print(reps_that_fit(meta, json.loads(warm.read_text())["wall_s"] if warm.exists() else 0.0, a.budget_s, a.reps))
        return 0
    if a.cmd == "checkpoint":
        print(MODELS[a.model_id].checkpoint())
        return 0
    if a.cmd == "plan":
        ck = Path(a.checkpoint) if a.checkpoint else MODELS[a.model_id].checkpoint()
        n = plan_need(parse_rates(a.rates), a.span_s, a.requests)
        print(f"[serve-tax] plan: {len(plan_of(Path(a.out), Path(a.c4), ck, n, a.docs, a.seed))} requests in {a.out}")
        return 0
    if a.cmd == "tick":
        print(json.dumps(tick_args(a.source, None, a.load)))
        return 0
    slo = (a.slo_ttft_s, a.slo_tpot_s) if a.slo_ttft_s and a.slo_tpot_s else None
    arms = [json.loads(d.read_text()) if d.is_file() else arm_metrics(d, slo) for d in a.dirs]
    if a.cmd == "metrics":
        print("\n".join(json.dumps(x) for x in arms))
        return 0
    t = table(arms)
    md = markdown(t)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "table.json").write_text(json.dumps(t, indent=1))
        (a.out / "table.md").write_text(md)
    print(md, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
