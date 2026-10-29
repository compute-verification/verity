"""A corpus of real vLLM V1 scheduler traces under varied traffic: plain vLLM's OpenAI server, a scheduler hook
(`vllm_trace_plugin`), and an async streaming client with Poisson arrivals.

    corpus.py run --out DIR --model-id llama32_1b --c4 C4.json.gz [--variant seeded] [--checkpoint DIR] [--traces heavy,...]
                  [--requests 1000]
    corpus.py summarize DIR
    corpus.py compare --out DIR CORPUS[:TRACE,...] ...

`run` tokenizes C4 documents with the checkpoint's tokenizer, makes one plan of requests (`traffic.plan`) and serves it once
per trace in `TRACES`, each on a fresh server (so every trace starts with a cold prefix cache), into DIR/<trace>/:
``raw/`` holds the hook's files and ``client.jsonl`` the client's. `summarize` (`summary.py`) joins them into each trace's
``steps.jsonl`` and ``requests.jsonl``, compares every trace's output ids request by request, and writes DIR/summary.json
and DIR/README.md. `compare` does the same comparison across corpora that serve the same plan.

`HOSTILE`'s traces test batch invariance against a malicious scheduler. Each serves heavy's load on one server (chunked
prefill, an 8192-token budget): ``honest`` under vLLM's own scheduler, the others under `vllm_adversarial_plugin`'s hostile
one, with its seed and prefix-cache mode. If a request's output ids moved with the hostile choices, `compare` shows it.

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

Every model runs the integration's batch-invariant eager engine (the ``bi-eager`` execution, `verity_vllm.engine.served`):
``VLLM_BATCH_INVARIANT=1``, eager, Model Runner V2 and the integration's other engine switches
(`verity_vllm.engine.env.ENGINE`) but the in-process engine core, which the OpenAI server does not have. A corpus's
`VARIANTS` entry sets its sampling and whether async scheduling is on (the profile has it off). Seeded sampling seeds each
request by `request_seed` from its id, which is the same in every trace, as `engine.served_seed` fixes a public seed at
``add_request``. Model Runner V2 draws a sampled token's Gumbel noise from Philox keyed by that seed and the token's
position. `MODELS`:

* ``llama32_1b``: unsloth/Llama-3.2-1B in bf16, vLLM's own attention.
* ``llama31_8b_int8``: the int8_exact linears (``--quantization verity_int8``) with vLLM's own attention. The int8 attention
  (``attention_backend CUSTOM``) refuses a request whose query rows are neither its whole window nor one row
  (`verity_vllm.engine.int8_ops._requests`), which is every prefix-cache hit and every later prefill chunk.
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
#: warden trace -> (the trace whose server and load it serves, D in buckets). The client reaches the server only through the
#: active warden (`verity.core.protocols.certifier.warden`, run by `benchmarks/network_traces/warden_serve.py serve`), in
#: cells. A late egress cell fails its connection closed and vLLM aborts the request, so D, from the warden's egress base
#: (`egress_base`), covers the slowest request's send to last token without the warden (heavy 10.7 s, small KV 55.7 s,
#: async greedy).
WARDEN = {"warden-heavy": ("heavy", 300), "warden-smallkv": ("smallkv", 1200)}
WARDEN_WHERE = "warden beside the server, not on the auditor host"
#: a stall of the client's event loop it records (`drive_warden`), in seconds: a fifth of a bucket
LOOP_LAG_S = 0.02
#: the descriptors the client holds beside its connections' (the server's and warden's pipes, the loop's, the logs')
FD_MARGIN = 256
#: the warden's schedule less D: lanes and queue hold every request of a 1000-request plan at once, so the warden never
#: throttles the load nor refuses; one 1 KiB cell a bucket each way keeps the padding of a thousand connections' D in memory
WARDEN_SCHEDULE = {"cell_bytes": 1024, "slots": 1, "bucket_ms": 100, "lanes": 1024, "queue": 1024, "window": 600,
                   "ingress_cells": 64, "ingress_buckets": 50, "egress_cells": 4096}
#: seconds the warden runs past the last arrival's D, for the last responses' egress cells
WARDEN_TAIL_S = 60.0
#: seconds the trace waits for the warden past its last window: a spawned publisher may still be committing the last
#: windows' records (`warden_probe`'s `publisher`)
WARDEN_END_S = 900.0
TRACES |= {name: (TRACES[base][0], f"{TRACES[base][1]}; through the active warden, D = {d} buckets of 100 ms "
                                   f"({WARDEN_WHERE})")
           for name, (base, d) in WARDEN.items()}


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


MODELS = {
    # a decode step takes about 20 ms from 10 to 180 rows and whole prefills run at about 105k tokens/s; 8192 KV tokens
    # serve about 7 requests/s
    "llama32_1b": Model("unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "llama-3.2-1b",
                        "Llama-3.2-1B (G-A's LLAMA32_1B), bf16, the bi-eager engine, vLLM's own attention",
                        ("--dtype", "bfloat16"), {},
                        {"light": 4.0, "medium": 20.0, "heavy": 300.0, "chunked": 40.0, "smallkv": 10.0}),
    # a decode step takes about 32 ms from 5 to 50 rows and whole prefills run at about 37k tokens/s
    "llama31_8b_int8": Model("verity-int8/Llama-3.1-8B-INT8", "723d342ac443b040931242c768df342b0f6f15a7", "llama-3.1-8b-int8",
                             "Llama-3.1-8B INT8 (int8_exact linears, `--quantization verity_int8`), the bi-eager engine, "
                             "vLLM's own attention", ("--quantization", "verity_int8"), {"VERITY_INT8_PARTS": "linears"},
                             {"light": 2.0, "medium": 10.0, "heavy": 100.0, "chunked": 20.0, "smallkv": 10.0}),
}


def kv_bytes_per_token(checkpoint: Path) -> int:
    """K and V of every layer for one token, in the checkpoint's 16-bit dtype."""
    c = json.loads((checkpoint / "config.json").read_text())
    head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    return 2 * c["num_hidden_layers"] * c.get("num_key_value_heads", c["num_attention_heads"]) * head_dim * 2


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


def plugin_site(site: Path) -> Path:
    """A directory whose .dist-info declares `vllm_adversarial_plugin.register` and `vllm_trace_plugin.register` as vLLM
    general plugins, in that order: vLLM calls them in the file's order, so the tracer wraps the adversary's `schedule`
    and records the step after the adversary has put back what it held."""
    d = site / "verity_trace_corpus-0.0.dist-info"
    d.mkdir(parents=True, exist_ok=True)
    (d / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-trace-corpus\nVersion: 0.0\n")
    (d / "entry_points.txt").write_text("[vllm.general_plugins]\nverity_adversary = vllm_adversarial_plugin:register\n"
                                        "verity_trace = vllm_trace_plugin:register\n")
    return site


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


async def drive(port: int, served: str, requests: list[dict], arrivals: list[float], ids: list[str], out: Path,
                drain_s: float, sampling: str) -> dict:
    """Send each request at its arrival (seconds from now) and stream its tokens; one client.jsonl line per request."""
    import aiohttp

    url = f"http://127.0.0.1:{port}/v1/completions"
    loop = asyncio.get_running_loop()
    lines = open(out, "w", encoding="utf-8")
    t0 = loop.time()
    t0_unix = time.time()

    async def one(session, r: dict, at: float, rid: str) -> None:
        await asyncio.sleep(max(0.0, t0 + at - loop.time()))
        sent = loop.time()
        rec = {"id": rid, "i": r["i"], "planned_s": round(at, 6), "send_s": round(sent - t0, 6),
               "send_unix": t0_unix + (sent - t0), "max_tokens": r["max_tokens"], "output_ids": [], "chunks": []}
        body = request_body(served, r, rid, sampling)
        rec["seed"] = body.get("seed")
        try:
            async with session.post(url, json=body, headers={"X-Request-Id": rid}) as resp:
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

    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None, sock_connect=60)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        tasks = [asyncio.create_task(one(session, r, at, rid)) for r, at, rid in zip(requests, arrivals, ids)]
        done, pending = await asyncio.wait(tasks, timeout=max(arrivals) + drain_s)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    lines.close()
    return {"t0_unix": t0_unix, "sent": len(tasks), "unfinished": len(pending), "wall_s": round(loop.time() - t0, 3)}


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
    from verity.core.protocols.certifier import warden

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
                 cpus: list[int] | None = None) -> tuple[subprocess.Popen, dict]:
    """`warden_serve.py serve` in front of the server at `port`, a process beside it on `cpus`, through `warden_probe.py`
    (its ticks, publishes and collections to d/warden/ticks.json), writing to d/warden/ (its log to serve.log) with a
    fresh session and the LMS key in `key`, for enough windows to cover the last arrival's D and `WARDEN_TAIL_S`. Returns
    the process and its deployment record."""
    params = WARDEN_SCHEDULE | {"delay": delay}
    tau = params["bucket_ms"] / 1000
    windows = math.ceil((span_s + delay * tau + WARDEN_TAIL_S) / (params["window"] * tau))
    w = d / "warden"
    w.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "benchmarks/network_traces/warden_serve.py"), "keys", "--dir", str(key)],
                   check=True, stdout=subprocess.DEVNULL, env=env)
    flags = [x for k, v in params.items() for x in (f"--{k.replace('_', '-')}", str(v))]
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


def run_trace(name: str, out: Path, model: Model, checkpoint: Path, requests: list[dict], site: Path,
              args: argparse.Namespace) -> dict:
    server_args, why = TRACES[name]
    sampling, async_scheduling = VARIANTS[args.variant]
    base, delay = WARDEN.get(name, (name, None))
    rate = model.rates["heavy" if name in HOSTILE else base]
    hostile = HOSTILE.get(name)
    adversary = {"VERITY_ADV_CACHE": hostile[0], "VERITY_ADV_SEED": str(hostile[1])} if hostile else {}
    if base == "smallkv":
        server_args = [*server_args, "--kv-cache-memory-bytes", str(KV_SMALL_TOKENS * kv_bytes_per_token(checkpoint))]
    d = out / name
    (d / "raw").mkdir(parents=True, exist_ok=True)
    port = free_port()
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(checkpoint), "--served-model-name",
            model.served, "--host", "127.0.0.1", "--port", str(port), *model.args, *BASE_ARGS,
            "--async-scheduling" if async_scheduling else "--no-async-scheduling", *server_args]
    engine_env = {**ENGINE_ENV, **model.env}
    env = dict(os.environ, **engine_env, **adversary, VERITY_TRACE_DIR=str(d / "raw"),
               PYTHONPATH=os.pathsep.join([str(site), str(HERE), str(ROOT / "integrations/vllm"), str(ROOT),
                                           *filter(None, [os.environ.get("PYTHONPATH")])]))
    arr = traffic.arrivals(requests, rate)
    ids = [request_id(r["i"]) for r in requests]
    cpus = cpu_plan(list(os.sched_getaffinity(0))) if delay is not None else None
    meta = {"trace": name, "rate": rate, "why": why, "server_argv": argv[1:], "engine_env": engine_env,
            "adversary": {"cache": hostile[0], "seed": hostile[1]} if hostile else None,
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
            meta |= asyncio.run(drive(port, model.served, requests, arr, ids, d / "client.jsonl", args.drain_s, sampling))
        else:
            wenv = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]))
            warden, dep = start_warden(d, port, delay, arr[-1], key, wenv, cpus and cpus["warden"])
            end = dep["origin_unix"] + dep["windows"] * dep["params"]["window"] * dep["params"]["bucket_ms"] / 1000
            meta["warden"] = {"where": WARDEN_WHERE, "egress_base": egress_base(warden_modules()[0]), "params": dep["params"],
                              "digest": dep["digest"], "windows": dep["windows"], "origin_unix": dep["origin_unix"],
                              "listen": dep["listen"], "slack_s": dep.get("slack"),
                              "key": "a scratch LMS key made for this trace and deleted after it (its public half is in "
                                     "warden/deployment.json)"}
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
    print(f"[corpus] {name}: done in {round(clock() - t)} s, unfinished {meta.get('unfinished')}, rc {server.returncode}",
          flush=True)
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
    t = clock()
    docs = load_docs(Path(args.c4), checkpoint / "tokenizer.json", args.docs)
    requests = traffic.plan(docs, args.requests, args.seed)
    print(f"[corpus] plan: {len(requests)} requests over {len(docs)} documents in {round(clock() - t, 1)} s", flush=True)
    with open(out / "plan.jsonl", "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    traces = [n for n in args.traces.split(",") if n]
    if bad := [n for n in traces if n not in TRACES]:
        raise SystemExit(f"unknown traces {bad}; known: {sorted(TRACES)}")
    sampling, async_scheduling = VARIANTS[args.variant]
    meta = {"model_id": args.model_id, "model": f"{model.repo}@{model.revision}", "model_what": model.what,
            "variant": args.variant, "sampling": {"name": sampling, **SAMPLING[sampling],
                                                  "seed_rule": request_seed.__doc__ if sampling == "seeded" else None},
            "async_scheduling": async_scheduling,
            "checkpoint": str(checkpoint), "c4": args.c4, "docs": args.docs, "seed": args.seed, "requests": args.requests,
            "eos_token_ids": list(EOS), "traffic": {"lengths": traffic.LENGTHS, "prefixes": traffic.PREFIXES,
                                                    "shared": traffic.SHARED, "at_end": traffic.AT_END, "gap": traffic.GAP,
                                                    "caps": traffic.CAPS},
            "versions": versions(), "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "GPU_LEASE_UUID",
                                                                            "GPU_LEASE_WHO", "RESEARCH_RUN_ID")},
            "host": socket.gethostname(), "traces": {}}
    rc = 0
    for name in traces:
        meta["traces"][name] = {"error": "cut: the run ended during this trace"}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        try:
            meta["traces"][name] = run_trace(name, out, model, checkpoint, requests, plugin_site(out / ".site"), args)
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
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    c = sub.add_parser("compare", help="compare output ids request by request across summarized corpora of one plan")
    c.add_argument("--out", required=True, help="where compare.json and COMPARE.md go")
    c.add_argument("corpora", nargs="+", help="CORPUS[:TRACE,...], a summarized corpus and, optionally, which traces")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
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
