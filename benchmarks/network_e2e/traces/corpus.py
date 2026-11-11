"""A corpus of real vLLM V1 scheduler traces under varied traffic: plain vLLM's OpenAI server, a scheduler hook
(`vllm_trace_plugin`), and an async streaming client with Poisson arrivals.

    corpus.py run --out DIR --model-id llama32_1b --c4 C4.json.gz [--variant seeded] [--checkpoint DIR] [--traces heavy,...]
                  [--requests 1000] [--rate R] [--admit tick (--tick-from TRACE_DIR | --tick-ms SAT,FREE)]
    corpus.py summarize DIR
    corpus.py compare --out DIR CORPUS[:TRACE,...] ...
    corpus.py tick TRACE_DIR

`run` tokenizes C4 documents with the checkpoint's tokenizer, makes one plan of requests (`traffic.plan`) and serves it once
per trace in `TRACES`, each on a fresh server (so every trace starts with a cold prefix cache), into DIR/<trace>/:
``raw/`` holds the hook's files and ``client.jsonl`` the client's. `summarize` (`summary.py`) joins them into each trace's
``steps.jsonl`` and ``requests.jsonl``, compares every trace's output ids request by request, and writes DIR/summary.json
and DIR/README.md. `compare` does the same comparison across corpora that serve the same plan.

`HOSTILE`'s traces test batch invariance against a malicious scheduler. Each serves heavy's load on one server (chunked
prefill, an 8192-token budget): ``honest`` under vLLM's own scheduler, the others under `vllm_adversarial_plugin`'s hostile
one, with its seed and prefix-cache mode. If a request's output ids moved with the hostile choices, `compare` shows it.

`CUT`'s traces serve honest's server and load with the carried pool's cut at every window edge
(`verity_vllm.pipeline.cut_plugin`, configured by ``VERITY_CUT``): at each edge every running request is preempted and
continues in the next window, hitting its own cached blocks. Each open request is held at what the server has sent of it
(`cut_plugin.SentFeed`, a middleware inside the tick's, writes ``raw/sent.jsonl``). `compare` against ``honest`` shows
whether a cut moves any request's output ids, and each edge's record (``raw/cut.PID.jsonl``) holds its hold, its cut set
and its cost.

`WARDEN`'s traces serve a trace's server and load through the active warden: `warden_serve.py serve` in front of the
server, a process beside it on its host (``warden beside the server, not on the auditor host``), and the client
(`drive_warden`) speaking cells through the relay's shim. The relay, the client and the server each run on cores of
their own (`cpu_plan`), the relay under `warden_probe.py` (its every tick, publish and collection) and the host's load
sampled every 100 ms (`host_sample.py`), so that a late tick can be told apart from a stalled client. DIR/<trace>/warden/
holds what the relay published (its deployment, certified records and connection log), the probes' records and the
shim's record of every connection; `summarize` runs `warden_serve.py check` on it first. Which relay served (its
egress from the release a, or from the admission s) is the trace's meta.json `warden.egress_base`
(`egress_base`). Every trace's server also loads `vllm_accept_stamp`, which records each request's
TCP accept, headers and body in the API server, so that the order requests arrived in can be told apart on each side of
the wire.

``--admit tick`` serves every trace of the run under the certified tick (`verity_vllm.pipeline.serve_admit`, its
``tick``): the server loads `tick_plugin.register` (`verity_vllm.pipeline.tick_plugin`) as a general plugin beside the
tracer, configured by ``VERITY_TICK``, and `tick_plugin.TickFeed` as its outermost middleware, which writes the release
headers and the watermark the server gets to its feed (``raw/tick-feed.jsonl``). In a `WARDEN` trace they come from the
network certifier in front of the server: it numbers each request it releases, in its release order, with the bucket of
its release, and streams the watermark to the server (`warden_serve.py serve --watermark`), its bucket τ_b
(`tick_for`). In every other trace the client stands in for the network certifier, which isn't in front of the server
(`StandIn`): it numbers the requests in the order it sends them, puts each in the bucket of its send (`TICK_BUCKET_MS`
buckets from the trace's origin), sends both in the release headers, and streams the watermark's line for each bucket as
it closes. The hostile schedulers' traces don't take the tick. τ_sat and τ_free are the engine class's: given
(``--tick-ms``), or calibrated on a trace of the same server on the same host (``--tick-from``, `tick_from`, which `tick`
prints): `TICK_SAT` and `TICK_FREE` times its mean busy cycle (`summary.busy_cycle`). The trace's meta.json records them
with the origin, the run, the watermark's outcome and where the releases came from (``tick``), and requests.jsonl each
request's ``release`` and ``bucket`` (a `WARDEN` trace's as the server got them, checked against the network certifier's
log: `summary.feed_releases`), which `replay.read_corpus_trace` reads.

Every model runs eager under the integration's engine switches (`verity_vllm.engine.env.ENGINE`) but the in-process engine
core, which the OpenAI server does not have: Model Runner V2 and, but for GLM-5.3, batch invariance
(``VLLM_BATCH_INVARIANT=1``, the ``bi-eager`` execution, `verity_vllm.engine.served`). A corpus's
`VARIANTS` entry sets its sampling and whether async scheduling is on (the profile has it off). Seeded sampling seeds each
request by `request_seed` from its id, which is the same in every trace, as `engine.served_seed` fixes a public seed at
``add_request``. Model Runner V2 samples a sampled token's Gumbel noise from Philox keyed by that seed and the token's
position. `MODELS`:

* ``llama32_1b``: unsloth/Llama-3.2-1B in bf16, vLLM's own attention on FlashAttention 2 (`FA2`). On Hopper vLLM picks
  FlashAttention 3, whose tile depends on the step's longest query, so a row's attention changes with the step it is in
  even under batch invariance; FlashAttention 2's doesn't, and it is what vLLM picks below Hopper.
* ``llama31_8b_int8``: the int8_exact linears (``--quantization verity_int8``) with vLLM's own attention on `FA2`. The
  int8 attention (``attention_backend CUSTOM``) refuses a request whose query rows are neither its whole window nor one
  row (`verity_vllm.engine.int8_ops._requests`), which is every prefix-cache hit and every later prefill chunk.
* ``glm53``: GLM-5.3 (block-FP8 ``GlmMoeDsaForCausalLM``) at TP8 on an 8-GPU node, vLLM's own kernels, as the
  integration's GLM53 rows serve it (``VLLM_BATCH_INVARIANT=0``). Its rates are measured on its node, so `run` takes
  ``--rate``.

A request stops on the checkpoint's end-of-text ids as vLLM takes them (`stop_ids`), which the corpus's meta.json records
as ``eos_token_ids``.
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import fcntl
import gc
import gzip
import hashlib
import itertools
import json
import math
import os
import re
import resource
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import traffic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EOS = (128001,)
#: the integration's engine switches under the bi-eager execution, less VLLM_ENABLE_V1_MULTIPROCESSING (the OpenAI server's
#: engine core is always its own process)
ENGINE_ENV = {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
              "VLLM_BATCH_INVARIANT": "1"}
#: --max-num-seqs pinned, so the batch cap a saturating trace hits is the corpus's, not the vLLM version's default
BASE_ARGS = ["--enforce-eager", "--enable-prefix-caching", "--max-model-len", "2048", "--max-num-seqs", "256", "--seed", "0",
             "--disable-uvicorn-access-log", "--middleware", "vllm_accept_stamp.AcceptStamp"]
#: name -> request sampling parameters; `seeded` requests also carry `request_seed` of their id
SAMPLING = {"greedy": {"temperature": 0.0}, "seeded": {"temperature": 0.8, "top_p": 0.95, "top_k": 50}}
#: name -> (sampling, async scheduling)
VARIANTS = {"greedy": ("greedy", False), "seeded": ("seeded", False), "async-greedy": ("greedy", True),
            "async-seeded": ("seeded", True)}
#: `engine.served_seed`'s name for its public default seed; `request_seed` hashes a request id under it
SEED_NAME = b"verity-vllm/served-default-seed/v1"
WHOLE_PREFILLS = ["--no-enable-chunked-prefill", "--max-num-batched-tokens", "8192"]
#: the small KV cache, in tokens; its bytes are the model's (`kv_bytes_per_token`)
KV_SMALL_TOKENS = 8192
#: name -> (server arguments, what it is for); the small KV cache's bytes are added per model
TRACES = {
    "light": (WHOLE_PREFILLS, "light load, whole prefills"),
    "medium": (WHOLE_PREFILLS, "medium load, whole prefills"),
    "heavy": (WHOLE_PREFILLS, "batch-saturating load (256 sequences, a waiting queue), whole prefills"),
    "chunked": (["--enable-chunked-prefill", "--max-num-batched-tokens", "512"],
                "chunked prefill with a 512-token step budget, near its throughput"),
    "smallkv": (["--enable-chunked-prefill", "--max-num-batched-tokens", "2048"],
                f"chunked prefill and a {KV_SMALL_TOKENS}-token KV cache, so the scheduler preempts"),
}
ORDER = ("heavy", "chunked", "smallkv", "medium", "light")
#: adversarial trace -> its scheduler: None, the honest one, or (VERITY_ADV_CACHE, VERITY_ADV_SEED) of a hostile one
#: (`vllm_adversarial_plugin`); every one serves heavy's load with chunked prefill on, so that a hostile scheduler may
#: split any prefill
HOSTILE = {"honest": None, "refuse-1": ("refuse", 1), "refuse-2": ("refuse", 2), "refuse-3": ("refuse", 3),
           "shortest-4": ("shortest", 4), "refuse-5": ("refuse", 5), "shortest-6": ("shortest", 6)}
CACHE_WHAT = {"refuse": "prefix-cache hits refused at random", "shortest": "prefix-cache hits cut to their first block"}
TRACES |= {name: (["--enable-chunked-prefill", "--max-num-batched-tokens", "8192"],
                  "heavy's load, chunked prefill (8192-token budget), the honest scheduler" if h is None else
                  f"heavy's load, chunked prefill (8192-token budget), a hostile scheduler (seed {h[1]}: random "
                  f"preemptions, hold-backs, order and prefill splits; {CACHE_WHAT[h[0]]})")
           for name, h in HOSTILE.items()}
#: cut trace -> its ``VERITY_CUT`` less the log (`verity_vllm.pipeline.cut_plugin.cut_config`): honest's server and load,
#: every open request cut at each window edge
CUT = {"cut-8": {"every_steps": 8}, "cut-64": {"every_steps": 64}, "cut-1s": {"every_s": 1.0}}
TRACES |= {name: (TRACES["honest"][0], f"{TRACES['honest'][1]}, cut at every window edge ({json.dumps(c)})")
           for name, c in CUT.items()}
#: warden trace -> (the trace whose server and load it serves, D in buckets). The client reaches the server only through the
#: active warden (`verity.protocols.certifier.verifier.warden`, run by `benchmarks/network_traces/warden_serve.py serve`), in
#: cells. A late egress cell fails its connection closed and vLLM aborts the request, so D, from the warden's egress base
#: (`egress_base`), covers the slowest request's send to last token without the warden (heavy 10.7 s, small KV 55.7 s,
#: async greedy). Under ``--admit tick`` it is kept, so that a tick trace's network certifier has an arrival trace's
#: schedule: the gate adds to a request at most a bucket and its clock's lag behind the engine (τ_sat below a saturated
#: step's time) before its admission, and stretches each engine step of its life to at most τ_free (`serve_admit`),
#: which on these traces' busy cycles is estimated, not measured, to stay inside D's headroom over those figures. A cell
#: D doesn't cover fails its connection late, which the summary counts (`summary.warden_summary`'s ``failed``), so it
#: can't pass unseen.
WARDEN = {"warden-heavy": ("heavy", 300), "warden-smallkv": ("smallkv", 1200)}
WARDEN_WHERE = "warden beside the server, not on the auditor host"
#: a stall of the client's event loop it records (`drive_warden`), in seconds: a fifth of a bucket
LOOP_LAG_S = 0.02
#: the descriptors the client holds beside its connections' (the server's and warden's pipes, the loop's, the logs')
FD_MARGIN = 256
#: the warden's schedule less D: lanes and queue hold every request of a 1000-request plan at once, so the warden never
#: throttles the load nor refuses; one 1 KiB cell a bucket each way keeps the padding of a thousand connections' D in
#: memory; and the bytes egress form, as vLLM streams its text completions (`warden`'s "The egress form")
WARDEN_SCHEDULE = {"cell_bytes": 1024, "slots": 1, "bucket_ms": 100, "lanes": 1024, "queue": 1024, "window": 600,
                   "ingress_cells": 64, "ingress_buckets": 50, "egress_cells": 4096, "egress": "bytes"}
#: seconds the warden runs past the last arrival's D, for the last responses' egress cells
WARDEN_TAIL_S = 60.0
#: seconds the trace waits for the warden past its last window: a spawned publisher may still be committing the last
#: windows' records (`warden_probe`'s `publisher`)
WARDEN_END_S = 900.0
TRACES |= {name: (TRACES[base][0], f"{TRACES[base][1]}; through the active warden, D = {d} buckets of 100 ms "
                                   f"({WARDEN_WHERE})")
           for name, (base, d) in WARDEN.items()}
#: the certified tick (``--admit tick``): the network certifier's bucket τ_b in ms, and τ_sat, τ_free as multiples of a
#: calibrating trace's mean busy cycle (`tick_from`)
TICK_BUCKET_MS = 100
TICK_SAT, TICK_FREE = 0.9, 1.1
#: how long the gate waits on a released request that hasn't reached the engine, or on a silent watermark, before it goes on
TICK_HOLD_MS = 5000
#: `tick_plugin.TickFeed`, the server's side of the release headers and the watermark (the network certifier's or
#: `StandIn`'s)
TICK_MIDDLEWARE = "verity_vllm.pipeline.tick_plugin.TickFeed"
CUT_MIDDLEWARE = "verity_vllm.pipeline.cut_plugin.SentFeed"


@dataclass(frozen=True)
class Model:
    repo: str
    revision: str
    served: str
    what: str
    args: tuple[str, ...]
    env: dict
    #: trace -> Poisson rate in requests/s, so that light keeps a handful of rows running and heavy outruns 256 sequences
    rates: dict

    def checkpoint(self) -> Path:
        hub = Path(os.environ.get("HF_HOME", "/workspace/hf")) / "hub"
        return hub / f"models--{self.repo.replace('/', '--')}" / "snapshots" / self.revision


#: vLLM's own attention on FlashAttention 2 on every GPU, the version whose output doesn't depend on the step's shape
FA2 = ("--attention-config", '{"flash_attn_version": 2}')

MODELS = {
    # a decode step takes about 20 ms from 10 to 180 rows and whole prefills run at about 105k tokens/s; 8192 KV tokens
    # serve about 7 requests/s
    "llama32_1b": Model("unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "llama-3.2-1b",
                        "Llama-3.2-1B (G-A's LLAMA32_1B), bf16, the bi-eager engine, vLLM's own attention",
                        ("--dtype", "bfloat16", *FA2), {},
                        {"light": 4.0, "medium": 20.0, "heavy": 300.0, "chunked": 40.0, "smallkv": 10.0}),
    # a decode step takes about 32 ms from 5 to 50 rows and whole prefills run at about 37k tokens/s
    "llama31_8b_int8": Model("verity-int8/Llama-3.1-8B-INT8", "723d342ac443b040931242c768df342b0f6f15a7", "llama-3.1-8b-int8",
                             "Llama-3.1-8B INT8 (int8_exact linears, `--quantization verity_int8`), the bi-eager engine, "
                             "vLLM's own attention", ("--quantization", "verity_int8", *FA2), {"VERITY_INT8_PARTS": "linears"},
                             {"light": 2.0, "medium": 10.0, "heavy": 100.0, "chunked": 20.0, "smallkv": 10.0}),
    "glm53": Model("zai-org/GLM-5.3", "aca966e4e02791568aa6a4ced368624b3d897f42", "glm-5.3",
                   "GLM-5.3 (the integration's GLM53, block-FP8 GlmMoeDsaForCausalLM) at TP8, eager, vLLM's own kernels "
                   "(not batch invariant)", ("--tensor-parallel-size", "8"), {"VLLM_BATCH_INVARIANT": "0"}, {}),
}


def kv_bytes_per_token(checkpoint: Path) -> int:
    """K and V of every layer for one token, in the checkpoint's 16-bit dtype."""
    c = json.loads((checkpoint / "config.json").read_text())
    head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    return 2 * c["num_hidden_layers"] * c.get("num_key_value_heads", c["num_attention_heads"]) * head_dim * 2


def stop_ids(checkpoint: Path) -> list[int]:
    """The ids vLLM stops a request on when it sets no stop strings or ids: the tokenizer's end-of-text token, then the
    generation config's ``eos_token_id`` (an int or a list; config.json's where there is no generation_config.json), each
    once; `EOS` where the checkpoint names none."""
    ids: list[int] = []
    tc = checkpoint / "tokenizer_config.json"
    eos = json.loads(tc.read_text()).get("eos_token") if tc.exists() else None
    eos = eos.get("content") if isinstance(eos, dict) else eos
    if eos is not None and (checkpoint / "tokenizer.json").exists():
        added = json.loads((checkpoint / "tokenizer.json").read_text()).get("added_tokens") or ()
        ids += [t["id"] for t in added if t.get("content") == eos][:1]
    for name in ("generation_config.json", "config.json"):
        if (checkpoint / name).exists():
            e = json.loads((checkpoint / name).read_text()).get("eos_token_id")
            if e is not None:
                ids += [e] if isinstance(e, int) else list(e)
                break
    return list(dict.fromkeys(int(i) for i in ids)) or list(EOS)


def request_seed(rid: str) -> int:
    """The public seed rule: the first 63 bits of SHA-512 over `SEED_NAME`, "/" and the request id."""
    return int.from_bytes(hashlib.sha512(SEED_NAME + b"/" + rid.encode()).digest()[:8], "big") >> 1


def request_id(i: int) -> str:
    """A plan request's X-Request-Id, the same in every trace."""
    return f"req-{i:05d}"


def request_body(served: str, r: dict, rid: str, sampling: str) -> dict:
    body = {"model": served, "prompt": r["prompt_ids"], "max_tokens": r["max_tokens"], **SAMPLING[sampling],
            "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True, "user": rid}
    if sampling == "seeded":
        body["seed"] = request_seed(rid)
    return body


def clock() -> float:
    return time.monotonic()


def load_docs(c4: Path, tokenizer: Path, n: int) -> list[list[int]]:
    """The first n documents of a C4 jsonl.gz shard, as token ids (no special tokens)."""
    from tokenizers import Tokenizer

    texts = []
    with gzip.open(c4, "rt", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if len(texts) == n:
                break
    tok = Tokenizer.from_file(str(tokenizer))
    return [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]


def plugin_site(site: Path, tick: bool = False, cut: bool = False) -> Path:
    """A directory whose .dist-info declares `vllm_adversarial_plugin.register` and `vllm_trace_plugin.register` as vLLM
    general plugins, in that order: vLLM calls them in the file's order, so the tracer wraps the adversary's `schedule`
    and records the step after the adversary has put back what it held. With `cut`, `cut_plugin.register` between them,
    so that the tracer records the step after the cut. With `tick`, `tick_plugin.register` too, whose gate wraps each
    scheduler built after them, so that the tracer sees a request when the gate adds it, and adds it through the cut's
    numbering in release order."""
    d = site / "verity_trace_corpus-0.0.dist-info"
    d.mkdir(parents=True, exist_ok=True)
    (d / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-trace-corpus\nVersion: 0.0\n")
    (d / "entry_points.txt").write_text("[vllm.general_plugins]\nverity_adversary = vllm_adversarial_plugin:register\n" +
                                        ("verity_cut = verity_vllm.pipeline.cut_plugin:register\n" if cut else "") +
                                        "verity_trace = vllm_trace_plugin:register\n" +
                                        ("verity_tick = verity_vllm.pipeline.tick_plugin:register\n" if tick else ""))
    return site


class StandIn:
    """The network certifier's side of ``--admit tick`` where none is in front of the server: what it sends in the release
    headers and the watermark (`verity_vllm.pipeline.serve_admit`), from the client's sends. `release` numbers a request
    N, in the order of the calls, and puts it in the bucket of `now`, counted from `origin` in buckets of `bucket_s`;
    `close` gives the watermark's next line, ``t n`` with n the releases of buckets up to t, once bucket t has closed
    (`due`). A release after a bucket's line goes to a later bucket whatever `now` says (an event loop may run a timer a
    little early), so every line counts exactly the releases of its buckets."""

    def __init__(self, bucket_s: float, origin: float, run: str | None = None) -> None:
        self.bucket_s, self.origin, self.run = bucket_s, origin, run or secrets.token_hex(8)
        self.buckets: list[int] = []
        self.closed = -1

    def release(self, now: float) -> tuple[int, int]:
        a = max(math.floor((now - self.origin) / self.bucket_s), self.closed + 1)
        self.buckets.append(a)
        return len(self.buckets) - 1, a

    def due(self) -> float:
        """When the next line's bucket closes."""
        return self.origin + (self.closed + 2) * self.bucket_s

    def close(self) -> bytes:
        self.closed += 1
        return b"%d %d\n" % (self.closed, bisect.bisect_right(self.buckets, self.closed))

    def done(self, sent: int) -> bool:
        """Every one of `sent` requests is released and its bucket's line is out: the watermark may end."""
        return len(self.buckets) == sent and (not self.buckets or self.closed >= self.buckets[-1])


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
    raise RuntimeError("the vLLM server was not ready in time")


def stop(server: subprocess.Popen) -> None:
    """SIGINT, which vLLM's server takes as a shutdown of its engine core too, then SIGKILL to whatever is left of its
    session (it was started in one of its own), so the next trace gets the whole GPU."""
    if server.poll() is None:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    try:
        os.killpg(server.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def tick_modules():
    """The vLLM integration's `serve_admit` (the release headers and the watermark's path) and `tick_plugin` (the server's
    plugin and middleware, and ``VERITY_TICK``), imported only for ``--admit tick``."""
    p = str(ROOT / "adapters/vllm")
    if p not in sys.path:
        sys.path.insert(0, p)
    from verity_vllm.pipeline import serve_admit, tick_plugin
    return serve_admit, tick_plugin


async def drive(port: int, served: str, requests: list[dict], arrivals: list[float], ids: list[str], out: Path,
                drain_s: float, sampling: str, tick_bucket_s: float | None = None) -> dict:
    """Send each request at its arrival (seconds from now) and stream its tokens; one client.jsonl line per request. With
    `tick_bucket_s`, stand in for the network certifier (`StandIn`): release each request as it is sent, its number and
    bucket in the release headers and in client.jsonl (``release``, ``bucket``), and stream the watermark to the server's
    ``POST /verity/watermark`` from the origin until every request is released and its bucket's line is out."""
    import aiohttp

    url = f"http://127.0.0.1:{port}/v1/completions"
    loop = asyncio.get_running_loop()
    lines = open(out, "w", encoding="utf-8")
    t0 = loop.time()
    t0_unix = time.time()
    cert = StandIn(tick_bucket_s, t0) if tick_bucket_s else None
    sa = tick_modules()[0] if cert is not None else None

    async def one(session, r: dict, at: float, rid: str) -> None:
        await asyncio.sleep(max(0.0, t0 + at - loop.time()))
        sent = loop.time()
        rec = {"id": rid, "i": r["i"], "planned_s": round(at, 6), "send_s": round(sent - t0, 6),
               "send_unix": t0_unix + (sent - t0), "max_tokens": r["max_tokens"], "output_ids": [], "chunks": []}
        headers = {"X-Request-Id": rid}
        if cert is not None:
            n, a = cert.release(sent)
            rec["release"], rec["bucket"] = n, a
            headers |= {sa.RELEASE_HEADER: str(n), sa.RUN_HEADER: cert.run, sa.BUCKET_HEADER: str(a)}
        body = request_body(served, r, rid, sampling)
        rec["seed"] = body.get("seed")
        try:
            async with session.post(url, json=body, headers=headers) as resp:
                rec["status"] = resp.status
                if resp.status != 200:
                    rec["error"] = (await resp.text())[:2000]
                    return
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    msg = json.loads(line[6:])
                    if msg.get("usage"):
                        rec["usage"] = msg["usage"]
                    for ch in msg.get("choices") or ():
                        toks = ch.get("token_ids") or []
                        if toks:
                            now = round(loop.time() - sent, 6)
                            rec["chunks"].append([now, len(toks)])
                            rec["output_ids"].extend(toks)
                            rec.setdefault("first_token_s", now)
                        if ch.get("finish_reason"):
                            rec["finish_reason"] = ch["finish_reason"]
                            rec["stop_reason"] = ch.get("stop_reason")
            rec["done_s"] = round(loop.time() - sent, 6)
        except asyncio.CancelledError:
            rec["error"] = "cancelled at the drain deadline"
            raise
        except Exception as e:  # noqa: BLE001 -- a failed request is a record, not the end of the trace
            rec["error"] = f"{type(e).__name__}: {e}"[:2000]
        finally:
            lines.write(json.dumps(rec, separators=(",", ":")) + "\n")
            lines.flush()

    async def watermark(session) -> dict:
        async def body():
            while not cert.done(len(requests)):
                await asyncio.sleep(max(0.0, cert.due() - loop.time()))
                yield cert.close()

        try:
            async with session.post(f"http://127.0.0.1:{port}{sa.WATERMARK_PATH}", data=body(),
                                    headers={sa.RUN_HEADER: cert.run}) as resp:
                return {"status": resp.status, "answer": (await resp.text())[:2000], "lines": cert.closed + 1}
        except Exception as e:  # noqa: BLE001 -- the gate opens on a broken watermark, which its log records
            return {"error": f"{type(e).__name__}: {e}"[:2000], "lines": cert.closed + 1}

    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None, sock_connect=60)
    # The first streamed line carries the prompt's token ids (return_token_ids); aiohttp refuses a line over twice
    # read_bufsize (LineTooLong), which its default 256 KiB puts at about 115k prompt tokens.
    async with aiohttp.ClientSession(connector=conn, timeout=timeout, read_bufsize=2 ** 22) as session:
        mark = asyncio.create_task(watermark(session)) if cert is not None else None
        tasks = [asyncio.create_task(one(session, r, at, rid)) for r, at, rid in zip(requests, arrivals, ids)]
        done, pending = await asyncio.wait(tasks, timeout=max(arrivals) + drain_s)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        tick = {}
        if mark is not None:
            if not (await asyncio.wait({mark}, timeout=TICK_HOLD_MS / 1000))[0]:
                mark.cancel()
            (wm,) = await asyncio.gather(mark, return_exceptions=True)
            tick = {"tick_run": cert.run, "watermark": wm if isinstance(wm, dict) else
                    {"error": f"cut at the drain: {type(wm).__name__}", "lines": cert.closed + 1}}
    lines.close()
    return {"t0_unix": t0_unix, "sent": len(tasks), "unfinished": len(pending), "wall_s": round(loop.time() - t0, 3), **tick}


def http_request(body: dict, rid: str, host: str) -> bytes:
    """The bytes of `drive`'s request on a connection of its own (`Connection: close`), as the warden's client sends them."""
    data = json.dumps(body, separators=(",", ":")).encode()
    return (f"POST /v1/completions HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nX-Request-Id: {rid}\r\n"
            f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n").encode() + data


def read_response(data: bytes, times: list[float], payload: int, sent: float) -> dict:
    """What `drive` records of a streamed completion, read from the bytes the warden's cells carried: `data`, and `times`,
    when each `DATA` or `END` cell arrived (cell k carries bytes kP ... (k+1)P - 1, P = `payload`). A server-sent event's
    tokens arrived with the cell that completed it, so `chunks` has the seconds from `sent` to that cell's arrival."""
    head, sep, _ = data.partition(b"\r\n\r\n")
    m = re.match(rb"HTTP/1\.[01] (\d{3})", head)
    rec: dict = {"status": int(m.group(1)) if m else None, "output_ids": [], "chunks": []}
    headers = {k.strip().lower(): v.strip() for k, _, v in (line.partition(b":") for line in head.split(b"\r\n")[1:])}
    start = len(head) + len(sep)
    pieces: list[tuple[int, bytes]] = []  # each body piece and its offset in `data`
    if b"chunked" in headers.get(b"transfer-encoding", b"").lower():
        i = start
        while (j := data.find(b"\r\n", i)) >= 0:
            try:
                n = int(data[i:j].split(b";")[0], 16)
            except ValueError:
                break
            if n == 0:
                break
            pieces.append((j + 2, data[j + 2:j + 2 + n]))
            i = j + 4 + n
    else:
        pieces.append((start, data[start:]))
    body = b"".join(p for _, p in pieces)
    if rec["status"] != 200:
        rec["error"] = body[:2000].decode(errors="replace")
        return rec
    bases = list(itertools.accumulate((len(p) for _, p in pieces), initial=0))

    def arrived(k: int) -> float:
        n = bisect.bisect_right(bases, k) - 1
        cell = (pieces[n][0] + k - bases[n]) // payload
        return round(times[min(cell, len(times) - 1)] - sent, 6)

    pos = 0
    while (end := body.find(b"\n\n", pos)) >= 0:
        line, pos = body[pos:end].strip(), end + 2
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        msg = json.loads(line[6:])
        if msg.get("usage"):
            rec["usage"] = msg["usage"]
        for ch in msg.get("choices") or ():
            toks = ch.get("token_ids") or []
            if toks:
                now = arrived(end + 1)
                rec["chunks"].append([now, len(toks)])
                rec["output_ids"].extend(toks)
                rec.setdefault("first_token_s", now)
            if ch.get("finish_reason"):
                rec["finish_reason"] = ch["finish_reason"]
                rec["stop_reason"] = ch.get("stop_reason")
    if times:
        rec["done_s"] = round(times[-1] - sent, 6)
    return rec


def warden_modules():
    """`verity`'s warden and `warden_serve` (benchmarks/network_traces), imported only for a warden trace."""
    for p in (ROOT, ROOT / "benchmarks/network_traces"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from verity.protocols.certifier.verifier import warden

    import warden_serve
    return warden, warden_serve


async def drive_warden(deployment: dict, served: str, requests: list[dict], arrivals: list[float], ids: list[str], out: Path,
                       drain_s: float, sampling: str, cpus: list[int] | None = None) -> dict:
    """`drive` through the warden: from the warden's origin, each request at its arrival on a connection of its own to the
    warden's listener, sent as cells and read through the client shim (`warden.cells`, `warden.Shim`), to the warden's
    close. client.jsonl has `drive`'s fields (`read_response`) and each request's warden ordinal and outcome, and
    warden/client.json the shim's record of every connection in `warden_serve.connection_log`'s form, for its `check`.

    The shim times each cell when the event loop reads it, so the loop must not stall: on `cpus` when given (`cpu_plan`),
    every thread of the process with it (numpy's among them), with the garbage collector off and every object made
    before the drive frozen out of its reach (stalls every connection shares put cell reads late), and the descriptor
    table grown to hold every connection before the first is opened
    (`reserve_fds`). warden/client_loop.json records what the loop did: every stall of `LOOP_LAG_S` or more ([seconds
    after the origin, ms]), every collection that still ran, and the table's size."""
    from warden_probe import Collections

    warden, ws = warden_modules()
    params = warden.WardenParams(**deployment["params"])
    B, origin, listen = params.cell_bytes, deployment["origin_unix"], tuple(deployment["listen"])
    host = "%s:%d" % tuple(deployment["upstream"])
    loop = asyncio.get_running_loop()
    affinity = os.sched_getaffinity(0)
    if cpus:
        pin_threads(cpus)
    fds = reserve_fds(len(requests) + FD_MARGIN)
    gc.collect()
    gc.freeze()
    gc.disable()
    collections = Collections(clock=time.time)
    gc.callbacks.append(collections)
    lags: list[list[float]] = []

    async def watch(step: float = 0.01) -> None:
        nxt = loop.time() + step
        while True:
            await asyncio.sleep(max(0.0, nxt - loop.time()))
            late = loop.time() - nxt
            if late >= LOOP_LAG_S:
                lags.append([round(time.time() - origin, 3), round(late * 1000, 1)])
            nxt = max(nxt + step, loop.time())

    watcher = asyncio.create_task(watch())
    try:
        await asyncio.sleep(max(0.0, origin - time.time()))
        lines = open(out, "w", encoding="utf-8")
        t0 = loop.time()
        t0_unix = time.time()
        conns: dict[int, dict] = {}

        async def one(k: int, r: dict, at: float, rid: str) -> None:
            await asyncio.sleep(max(0.0, t0 + at - loop.time()))
            body = request_body(served, r, rid, sampling)
            request = http_request(body, rid, host)
            sent = time.time()
            rec = {"id": rid, "i": r["i"], "planned_s": round(at, 6), "send_s": round(loop.time() - t0, 6), "send_unix": sent,
                   "max_tokens": r["max_tokens"], "seed": body.get("seed"), "output_ids": [], "chunks": []}
            shim = warden.Shim(B, time.time)
            try:
                reader, writer = await asyncio.open_connection(*listen)
                try:
                    writer.write(warden.cells(request, B))
                    await shim.read(reader)
                finally:
                    writer.close()
                rec["warden"] = {"ordinal": shim.ordinal, "outcome": shim.outcome}
                times = [t for t, kind in shim.arrivals if kind in (warden.DATA, warden.END)]
                rec |= read_response(bytes(shim.data), times, params.payload, sent)
                if shim.outcome != "end":
                    rec.setdefault("error", f"the warden's connection ended: {shim.outcome}")
                conns[k] = ws.connection_log(k, round(t0_unix - origin + at, 6), sent - origin, request, shim, origin)
            except asyncio.CancelledError:
                rec["error"] = "cancelled at the drain deadline"
                raise
            except Exception as e:  # noqa: BLE001 -- a failed request is a record, not the end of the trace
                rec["error"] = f"{type(e).__name__}: {e}"[:2000]
            finally:
                conns.setdefault(k, {"index": k, "error": rec.get("error")})
                lines.write(json.dumps(rec, separators=(",", ":")) + "\n")
                lines.flush()

        tasks = [asyncio.create_task(one(k, r, at, rid)) for k, (r, at, rid) in enumerate(zip(requests, arrivals, ids))]
        done, pending = await asyncio.wait(tasks, timeout=max(arrivals) + drain_s)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        lines.close()
        doc = {"connections": [conns.get(k, {"index": k, "error": "no record"}) for k in range(len(tasks))]}
        (out.parent / "warden" / "client.json").write_text(json.dumps(doc))
        return {"t0_unix": t0_unix, "sent": len(tasks), "unfinished": len(pending), "wall_s": round(loop.time() - t0, 3)}
    finally:
        watcher.cancel()
        gc.callbacks.remove(collections)
        gc.enable()
        gc.unfreeze()
        if cpus:
            pin_threads(affinity)
        (out.parent / "warden" / "client_loop.json").write_text(json.dumps(
            {"cpus": cpus, "gc": "off, everything before the drive frozen", "watch_ms": 10, "lag_ms_at_least": LOOP_LAG_S * 1000,
             "fds": fds, "lags": lags, "max_lag_ms": max((ms for _, ms in lags), default=None), **collections.doc(origin)}))


def reserve_fds(n: int) -> int:
    """Grow the process's descriptor table to hold `n` descriptors (at most RLIMIT_NOFILE's soft limit) now; returns its
    size. The kernel grows it as descriptors pass 64, 128, 256 …, and in a process of more than one thread each growth
    waits for an RCU grace period (`expand_fdtable`), the growing thread asleep in D: on a loaded node, every connection's
    cell read a bucket late (`warden_serve.reserve_fds`). The table never shrinks."""
    n = min(n, resource.getrlimit(resource.RLIMIT_NOFILE)[0])
    fd = os.open(os.devnull, os.O_RDONLY)
    try:
        os.close(fcntl.fcntl(fd, fcntl.F_DUPFD, n - 1))
    finally:
        os.close(fd)
    return n


def pin_threads(cpus) -> None:
    """Every thread of this process on `cpus` (`sched_setaffinity` of pid 0 moves only the calling thread)."""
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
        except ProcessLookupError:
            pass


def cpu_plan(cpus: list[int]) -> dict | None:
    """The job's cores split for a warden trace, so that nothing else runs on the warden's or the client's: the warden the
    last one (the last two from 8 cores, its publisher beside its ticker), the client the one before, and
    the server, its engine core and the host sampler the rest. None under 4 cores (nothing is pinned)."""
    cpus = sorted(cpus)
    if len(cpus) < 4:
        return None
    k = 2 if len(cpus) >= 8 else 1
    return {"server": cpus[:-k - 1], "client": [cpus[-k - 1]], "warden": cpus[-k:]}


def taskset(cpus: list[int] | None) -> list[str]:
    return ["taskset", "-c", ",".join(map(str, cpus))] if cpus else []


def egress_base(warden) -> str:
    """Which point the relay's D counts from, read from its module's docstring: egress cell i at
    a + D + i/r from the release a (the ingress end), or at s + D + i/r from the admission s."""
    doc = warden.__doc__ or ""
    if "before a + D" in doc:
        return "release a (the ingress end): egress cell i at a + D + i/r"
    if "before s + D" in doc:
        return "admission s: egress cell i at s + D + i/r"
    return "unknown (the warden's docstring names neither a + D nor s + D)"


def start_warden(d: Path, port: int, delay: int, span_s: float, key: Path, env: dict,
                 cpus: list[int] | None = None, watermark: bool = False) -> tuple[subprocess.Popen, dict]:
    """`warden_serve.py serve` in front of the server at `port`, a process beside it on `cpus`, through `warden_probe.py`
    (its ticks, publishes and collections to d/warden/ticks.json), writing to d/warden/ (its log to serve.log) with a
    fresh session and the LMS key in `key`, for enough windows to cover the last arrival's D and `WARDEN_TAIL_S`; with
    `watermark`, also streaming the watermark to the server (``--watermark``). Returns the process and its deployment
    record."""
    params = WARDEN_SCHEDULE | {"delay": delay}
    tau = params["bucket_ms"] / 1000
    windows = math.ceil((span_s + delay * tau + WARDEN_TAIL_S) / (params["window"] * tau))
    w = d / "warden"
    w.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "benchmarks/network_traces/warden_serve.py"), "keys", "--dir", str(key)],
                   check=True, stdout=subprocess.DEVNULL, env=env)
    flags = [x for k, v in params.items() for x in (f"--{k.replace('_', '-')}", str(v))] + ["--watermark"] * watermark
    with open(w / "serve.log", "w") as log:
        proc = subprocess.Popen([*taskset(cpus), sys.executable, str(HERE / "warden_probe.py"), "serve", "--upstream",
                                 f"127.0.0.1:{port}", "--listen", "127.0.0.1:0", "--out", str(w), "--key", str(key),
                                 "--session", secrets.token_hex(32), "--windows", str(windows), "--start-in", "2", *flags],
                                stdout=subprocess.PIPE, stderr=log, text=True, env=env)
    if "listening" not in json.loads(proc.stdout.readline() or "{}"):
        proc.wait()
        raise RuntimeError(f"the warden exited with {proc.returncode} before it listened (warden/serve.log)")
    return proc, json.loads((w / "deployment.json").read_text())


def check_warden(d: Path) -> int:
    """`warden_serve.py check` of a warden trace's d/warden (its certificates, the audit of its records and the client's
    cells against them), into d/warden/check.json."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]))
    with open(d / "warden" / "check.log", "w") as log:
        return subprocess.run([sys.executable, str(ROOT / "benchmarks/network_traces/warden_serve.py"), "check", "--run",
                               str(d / "warden")], stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def rate_of(model: Model, name: str, rate: float | None) -> float | None:
    """A trace's Poisson rate: `rate` where given, else the model's for the trace's load (None if it has none)."""
    return rate if rate is not None else model.rates.get("heavy" if name in HOSTILE or name in CUT
                                                         else WARDEN.get(name, (name,))[0])


def tick_args(trace: str | None, given_ms: str | None) -> dict:
    """``--admit tick``'s constants in microseconds (`serve_admit.Tick`'s fields) and where τ came from: calibrated on a
    trace's directory (`tick_from`), or given as ``SAT,FREE`` in ms."""
    if (trace is None) == (given_ms is None):
        raise SystemExit("--admit tick takes τ_sat and τ_free from one of --tick-from TRACE_DIR and --tick-ms SAT,FREE")
    if trace is not None:
        return tick_from(Path(trace))
    sat, free = (float(x) for x in given_ms.split(","))
    if min(sat, free) <= 0:
        raise SystemExit(f"--tick-ms {given_ms}: τ_sat and τ_free are positive milliseconds")
    return {"bucket_us": TICK_BUCKET_MS * 1000, "sat_us": round(sat * 1000), "free_us": round(free * 1000),
            "calibrated_from": None}


def tick_from(d: Path) -> dict:
    """The certified tick's constants calibrated on `d`, a summarized trace of the same server on the same host: τ_b
    `TICK_BUCKET_MS`, and τ_sat and τ_free `TICK_SAT` and `TICK_FREE` times its mean busy cycle (`summary.busy_cycle`), in
    microseconds, with where they came from."""
    import summary

    c = summary.busy_cycle(summary.read_jsonl(d / "steps.jsonl"), summary.read_jsonl(d / "requests.jsonl"))
    if not c["cycles"]:
        raise SystemExit(f"{d}: no busy cycle to calibrate the tick on (is the trace summarized?)")
    return {"bucket_us": TICK_BUCKET_MS * 1000, "sat_us": round(TICK_SAT * c["mean_us"]),
            "free_us": round(TICK_FREE * c["mean_us"]), "calibrated_from": {"trace": str(d), **c}}


def tick_for(traces: list[str], trace: str | None, given_ms: str | None) -> dict:
    """``--admit tick``'s constants for a run of `traces` (`tick_args`). Refused for the hostile schedulers' traces, and
    for a `WARDEN` trace unless its network certifier's bucket is τ_b, since that certifier's release buckets and
    watermark lines are then the tick's buckets."""
    if bad := [n for n in traces if n in HOSTILE]:
        raise SystemExit(f"--admit tick doesn't serve the hostile schedulers' traces {bad}")
    tick = tick_args(trace, given_ms)
    if (ws := [n for n in traces if n in WARDEN]) and WARDEN_SCHEDULE["bucket_ms"] * 1000 != tick["bucket_us"]:
        raise SystemExit(f"--admit tick: the network certifier's bucket, {WARDEN_SCHEDULE['bucket_ms']} ms "
                         f"(WARDEN_SCHEDULE), isn't τ_b, {tick['bucket_us']} µs, so {ws} would number releases in buckets "
                         "the tick doesn't count")
    return tick


def run_trace(name: str, out: Path, model: Model, checkpoint: Path, requests: list[dict], site: Path,
              args: argparse.Namespace) -> dict:
    server_args, why = TRACES[name]
    sampling, async_scheduling = VARIANTS[args.variant]
    base, delay = WARDEN.get(name, (name, None))
    rate = rate_of(model, name, getattr(args, "rate", None))
    tick = getattr(args, "tick", None)
    hostile = HOSTILE.get(name)
    adversary = {"VERITY_ADV_CACHE": hostile[0], "VERITY_ADV_SEED": str(hostile[1])} if hostile else {}
    if (cut := CUT.get(name)) is not None:
        raw = out / name / "raw"
        adversary["VERITY_CUT"] = json.dumps({**cut, "log": str(raw), "feed": str(raw / "sent.jsonl")})
    if base == "smallkv":
        server_args = [*server_args, "--kv-cache-memory-bytes", str(KV_SMALL_TOKENS * kv_bytes_per_token(checkpoint))]
    d = out / name
    (d / "raw").mkdir(parents=True, exist_ok=True)
    port = free_port()
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(checkpoint), "--served-model-name",
            model.served, "--host", "127.0.0.1", "--port", str(port), *model.args, *BASE_ARGS,
            "--async-scheduling" if async_scheduling else "--no-async-scheduling", *server_args,
            *(["--middleware", CUT_MIDDLEWARE] if cut is not None else []),
            *(["--middleware", TICK_MIDDLEWARE] if tick else [])]
    engine_env = {**ENGINE_ENV, **model.env}
    env = dict(os.environ, **engine_env, **adversary, VERITY_TRACE_DIR=str(d / "raw"),
               PYTHONPATH=os.pathsep.join([str(site), str(HERE), str(ROOT / "adapters/vllm"), str(ROOT),
                                           *filter(None, [os.environ.get("PYTHONPATH")])]))
    if tick:
        env[tick_modules()[1].TICK_ENV] = json.dumps({**{k: tick[k] for k in ("bucket_us", "sat_us", "free_us")},
                                                      "hold_ms": TICK_HOLD_MS, "feed": str(d / "raw" / "tick-feed.jsonl"),
                                                      "log": str(d / "raw")})
    arr = traffic.arrivals(requests, rate)
    ids = [request_id(r["i"]) for r in requests]
    cpus = cpu_plan(list(os.sched_getaffinity(0))) if delay is not None else None
    meta = {"trace": name, "rate": rate, "why": why, "server_argv": argv[1:], "engine_env": engine_env,
            "adversary": {"cache": hostile[0], "seed": hostile[1]} if hostile else None, "cut": cut,
            "requests": len(requests), "arrivals_span_s": round(arr[-1], 3)}
    if delay is not None:
        meta["cpus"] = cpus or f"not pinned: the job has {len(os.sched_getaffinity(0))} cores"
    t = clock()
    warden = sampler = None
    key = Path(tempfile.mkdtemp(prefix="trace-corpus-warden-")) if delay is not None else None
    with open(d / "server.log", "wb") as log:
        server = subprocess.Popen([*taskset(cpus and cpus["server"]), *argv], stdout=log, stderr=subprocess.STDOUT, env=env,
                                  start_new_session=True)
    try:
        wait_ready(server, port, clock() + args.startup_s)
        meta["startup_s"] = round(clock() - t, 1)
        print(f"[corpus] {name}: server up in {meta['startup_s']} s; {len(requests)} requests at {rate}/s", flush=True)
        if delay is None:
            meta |= asyncio.run(drive(port, model.served, requests, arr, ids, d / "client.jsonl", args.drain_s, sampling,
                                      tick["bucket_us"] / 1e6 if tick else None))
            if tick:
                meta["tick"] = {**tick, "hold_ms": TICK_HOLD_MS, "origin_unix": meta["t0_unix"],
                                "run": meta.pop("tick_run"), "watermark": meta.pop("watermark"),
                                "where": "the client's stand-in network certifier (corpus.StandIn)"}
        else:
            wenv = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]))
            warden, dep = start_warden(d, port, delay, arr[-1], key, wenv, cpus and cpus["warden"], watermark=bool(tick))
            end = dep["origin_unix"] + dep["windows"] * dep["params"]["window"] * dep["params"]["bucket_ms"] / 1000
            meta["warden"] = {"where": WARDEN_WHERE, "egress_base": egress_base(warden_modules()[0]), "params": dep["params"],
                              "digest": dep["digest"], "windows": dep["windows"], "origin_unix": dep["origin_unix"],
                              "listen": dep["listen"], "slack_s": dep.get("slack"),
                              "key": "a scratch LMS key made for this trace and deleted after it (its public half is in "
                                     "warden/deployment.json)"}
            if tick:
                meta["tick"] = {**tick, "hold_ms": TICK_HOLD_MS, "origin_unix": dep["origin_unix"], "run": dep["run_id"],
                                "where": f"the network certifier ({WARDEN_WHERE}; warden_serve.py serve --watermark)"}
            pid = d / "warden" / "publisher.pid"
            sampler = subprocess.Popen([*taskset(cpus and cpus["server"]), sys.executable, str(HERE / "host_sample.py"),
                                        str(d / "warden" / "host.jsonl"), ",".join(map(str, sorted(os.sched_getaffinity(0)))),
                                        f"warden:{warden.pid}", f"client:{os.getpid()}", f"server={server.pid}",
                                        *([f"publisher:{pid.read_text()}"] if pid.exists() else [])])
            print(f"[corpus] {name}: warden ({WARDEN_WHERE}) on {dep['listen']}, D {delay} from the "
                  f"{meta['warden']['egress_base'].split(':')[0]}, {dep['windows']} windows; cores {meta['cpus']}", flush=True)
            meta |= asyncio.run(drive_warden(dep, model.served, requests, arr, ids, d / "client.jsonl", args.drain_s,
                                             sampling, cpus and cpus["client"]))
            try:
                meta["warden"]["rc"] = warden.wait(max(0.0, end - time.time()) + WARDEN_END_S)
            except subprocess.TimeoutExpired:
                meta["warden"]["rc"] = f"still running {WARDEN_END_S:g} s past its last window"
            meta["warden"]["ended_past_last_window_s"] = round(time.time() - end, 3)
            if tick:
                log = d / "warden" / "warden.json"
                meta["tick"]["watermark"] = (json.loads(log.read_text()).get("watermark") if log.exists() else
                                             {"error": "the warden wrote no warden.json"})
    finally:
        if warden is not None and warden.poll() is None:
            warden.terminate()
            warden.wait()
        if sampler is not None:
            sampler.terminate()
            sampler.wait()
        stop(server)
        if key is not None:
            shutil.rmtree(key, ignore_errors=True)
        meta["server_rc"] = server.returncode
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[corpus] {name}: done in {round(clock() - t)} s, unfinished {meta.get('unfinished')}, rc {server.returncode}"
          + (f", watermark {meta['tick'].get('watermark')}" if "tick" in meta else ""), flush=True)
    return meta


def versions() -> dict:
    from importlib import metadata

    out = {}
    for dist in ("vllm", "torch", "aiohttp", "tokenizers"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def _terminated(signum, frame):
    raise SystemExit(128 + signum)


def run(args: argparse.Namespace) -> int:
    # SIGTERM (run.sh's deadline) unwinds through run_trace's `finally`, which stops the server: it runs in a session of its own
    signal.signal(signal.SIGTERM, _terminated)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    model = MODELS[args.model_id]
    checkpoint = Path(args.checkpoint) if args.checkpoint else model.checkpoint()
    traces = [n for n in args.traces.split(",") if n]
    if bad := [n for n in traces if n not in TRACES]:
        raise SystemExit(f"unknown traces {bad}; known: {sorted(TRACES)}")
    if missing := [n for n in traces if rate_of(model, n, args.rate) is None]:
        raise SystemExit(f"{args.model_id} has no rate for {missing}: give --rate")
    args.tick = None
    if args.admit == "tick":
        args.tick = tick_for(traces, args.tick_from, args.tick_ms)
        print(f"[corpus] the certified tick: {json.dumps(args.tick)}", flush=True)
    elif args.tick_from or args.tick_ms:
        raise SystemExit("--tick-from and --tick-ms go with --admit tick")
    t = clock()
    docs = load_docs(Path(args.c4), checkpoint / "tokenizer.json", args.docs)
    requests = traffic.plan(docs, args.requests, args.seed)
    print(f"[corpus] plan: {len(requests)} requests over {len(docs)} documents in {round(clock() - t, 1)} s", flush=True)
    with open(out / "plan.jsonl", "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    sampling, async_scheduling = VARIANTS[args.variant]
    meta = {"model_id": args.model_id, "model": f"{model.repo}@{model.revision}", "model_what": model.what,
            "variant": args.variant, "sampling": {"name": sampling, **SAMPLING[sampling],
                                                  "seed_rule": request_seed.__doc__ if sampling == "seeded" else None},
            "async_scheduling": async_scheduling, "engine_env": {**ENGINE_ENV, **model.env}, "rate": args.rate,
            "admit": args.admit, "tick": args.tick,
            "checkpoint": str(checkpoint), "c4": args.c4, "docs": args.docs, "seed": args.seed, "requests": args.requests,
            "eos_token_ids": stop_ids(checkpoint), "traffic": {"lengths": traffic.LENGTHS, "prefixes": traffic.PREFIXES,
                                                               "shared": traffic.SHARED, "at_end": traffic.AT_END,
                                                               "gap": traffic.GAP, "caps": traffic.CAPS},
            "versions": versions(), "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "GPU_LEASE_UUID",
                                                                            "GPU_LEASE_WHO", "RESEARCH_RUN_ID")},
            "host": socket.gethostname(), "traces": {}}
    rc = 0
    for name in traces:
        meta["traces"][name] = {"error": "cut: the run ended during this trace"}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        try:
            meta["traces"][name] = run_trace(name, out, model, checkpoint, requests,
                                             plugin_site(out / ".site", tick=args.tick is not None,
                                                         cut=any(n in CUT for n in traces)), args)
        except Exception as e:  # noqa: BLE001 -- the next trace still runs
            meta["traces"][name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[corpus] {name}: FAILED {e}", flush=True)
            rc = 1
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--model-id", required=True, choices=sorted(MODELS))
    r.add_argument("--variant", default="greedy", choices=sorted(VARIANTS))
    r.add_argument("--checkpoint", help="the checkpoint's directory (default: the model's snapshot under $HF_HOME/hub)")
    r.add_argument("--c4", required=True, help="a C4 jsonl.gz shard")
    r.add_argument("--docs", type=int, default=20000)
    r.add_argument("--requests", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--traces", default=",".join(ORDER))
    r.add_argument("--startup-s", type=float, default=600)
    r.add_argument("--drain-s", type=float, default=600)
    r.add_argument("--rate", type=float, help="every trace's Poisson rate in requests/s (default: the model's per trace)")
    r.add_argument("--admit", choices=("arrival", "tick"), default="arrival",
                   help="arrival: vLLM adds each request as it comes; tick: the certified tick (serve_admit)")
    r.add_argument("--tick-from", metavar="TRACE_DIR", help="calibrate τ_sat and τ_free on this summarized trace of the "
                                                            "same server on this host (`tick`)")
    r.add_argument("--tick-ms", metavar="SAT,FREE", help="τ_sat and τ_free in ms")
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    c = sub.add_parser("compare", help="compare output ids request by request across summarized corpora of one plan")
    c.add_argument("--out", required=True, help="where compare.json and COMPARE.md go")
    c.add_argument("corpora", nargs="+", help="CORPUS[:TRACE,...], a summarized corpus and, optionally, which traces")
    k = sub.add_parser("tick", help="the certified tick's constants calibrated on a summarized trace (`tick_from`)")
    k.add_argument("trace", help="a summarized trace's directory (steps.jsonl, requests.jsonl)")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    if args.cmd == "tick":
        print(json.dumps(tick_from(Path(args.trace))))
        return 0
    import summary

    if args.cmd == "summarize":
        for w in sorted(Path(args.dir).glob("*/warden/deployment.json")):
            if not (w.parent / "check.json").exists():
                print(f"[corpus] {w.parent.parent.name}: warden check rc {check_warden(w.parent.parent)}", flush=True)
        summary.summarize(Path(args.dir))
        return 0
    out = summary.compare_corpora([summary.corpus_spec(x) for x in args.corpora], Path(args.out))
    print(json.dumps({k: out[k] for k in ("requests", "identical", "verdict")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
