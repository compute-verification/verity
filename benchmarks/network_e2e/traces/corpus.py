"""A corpus of real vLLM V1 scheduler traces under varied traffic: plain vLLM's OpenAI server, the int8_exact linears, and a
scheduler hook (`vllm_trace_plugin`), driven by an async streaming client with Poisson arrivals.

    corpus.py run --out DIR --model CKPT_DIR --c4 C4.json.gz [--traces heavy,medium,chunked,smallkv,light] [--requests 1000]
    corpus.py summarize DIR

`run` tokenizes C4 documents with the checkpoint's tokenizer, makes one plan of requests (`traffic.plan`) and serves it once
per trace in `TRACES`, each on a fresh server (so every trace starts with a cold prefix cache), into DIR/<trace>/:
``raw/`` holds the hook's files and ``client.jsonl`` the client's. `summarize` (`summary.py`) joins them into each trace's
``steps.jsonl`` and ``requests.jsonl`` and writes DIR/summary.json and DIR/README.md.

The server is the engine of vLLM's int8 rows (`integrations/vllm/workloads/*__int8__*`): ``--quantization verity_int8``,
batch invariance (``VLLM_BATCH_INVARIANT=1``), eager, async scheduling off, Model Runner V2 and the integration's other
engine switches (`verity_vllm.engine.env.ENGINE`) but the in-process engine core, which the OpenAI server does not have.
Attention is vLLM's own: the int8 attention (``attention_backend CUSTOM``) refuses a request whose query rows are neither
its whole window nor one row (`verity_vllm.engine.int8_ops._requests`), which is every prefix-cache hit and every later
prefill chunk.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import resource
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import traffic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SERVED = "llama-3.1-8b-int8"
EOS = (128001,)
#: the integration's engine switches for vLLM's int8 rows, less VLLM_ENABLE_V1_MULTIPROCESSING (the OpenAI server's engine
#: core is always its own process)
ENGINE_ENV = {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
              "VLLM_BATCH_INVARIANT": "1", "VERITY_INT8_PARTS": "linears"}
#: --max-num-seqs pinned, so the batch cap a saturating trace hits is the corpus's, not the vLLM version's default
BASE_ARGS = ["--quantization", "verity_int8", "--enforce-eager", "--no-async-scheduling", "--enable-prefix-caching",
             "--max-model-len", "2048", "--max-num-seqs", "256", "--seed", "0", "--disable-uvicorn-access-log"]
WHOLE_PREFILLS = ["--no-enable-chunked-prefill", "--max-num-batched-tokens", "8192"]
KV_SMALL_BYTES = 1 << 30
#: name -> (Poisson rate in requests/s, server arguments, what it is for). On one RTX PRO 6000 a decode step takes about
#: 32 ms from 5 to 50 rows and whole prefills run at about 37k tokens/s (the 60-request smoke, r20261024-045656-4c66), so
#: 2/s keeps a handful of rows running and 100/s outruns 256 sequences and queues.
TRACES = {
    "light": (2.0, WHOLE_PREFILLS, "light load, whole prefills"),
    "medium": (10.0, WHOLE_PREFILLS, "medium load, whole prefills"),
    "heavy": (100.0, WHOLE_PREFILLS, "batch-saturating load (256 sequences, a waiting queue), whole prefills"),
    "chunked": (20.0, ["--enable-chunked-prefill", "--max-num-batched-tokens", "512"],
                "chunked prefill with a 512-token step budget, near its throughput"),
    "smallkv": (10.0, ["--enable-chunked-prefill", "--max-num-batched-tokens", "2048", "--kv-cache-memory-bytes",
                       str(KV_SMALL_BYTES)], "chunked prefill and a 1 GiB KV cache (8192 tokens), so the scheduler preempts"),
}
ORDER = ("heavy", "medium", "chunked", "smallkv", "light")


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
    """A directory whose .dist-info declares `vllm_trace_plugin.register` as a vLLM general plugin."""
    d = site / "verity_trace_corpus-0.0.dist-info"
    d.mkdir(parents=True, exist_ok=True)
    (d / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-trace-corpus\nVersion: 0.0\n")
    (d / "entry_points.txt").write_text("[vllm.general_plugins]\nverity_trace = vllm_trace_plugin:register\n")
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


async def drive(port: int, requests: list[dict], arrivals: list[float], ids: list[str], out: Path, drain_s: float) -> dict:
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
        body = {"model": SERVED, "prompt": r["prompt_ids"], "max_tokens": r["max_tokens"], "temperature": 0.0,
                "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True, "user": rid}
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


def run_trace(name: str, out: Path, model: str, requests: list[dict], site: Path, args: argparse.Namespace) -> dict:
    rate, server_args, why = TRACES[name]
    d = out / name
    (d / "raw").mkdir(parents=True, exist_ok=True)
    port = free_port()
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--served-model-name", SERVED,
            "--host", "127.0.0.1", "--port", str(port), *BASE_ARGS, *server_args]
    env = dict(os.environ, **ENGINE_ENV, VERITY_TRACE_DIR=str(d / "raw"),
               PYTHONPATH=os.pathsep.join([str(site), str(HERE), str(ROOT / "integrations/vllm"), str(ROOT),
                                           *filter(None, [os.environ.get("PYTHONPATH")])]))
    arr = traffic.arrivals(requests, rate)
    ids = [f"{name}-{r['i']:05d}" for r in requests]
    meta = {"trace": name, "rate": rate, "why": why, "server_argv": argv[1:], "engine_env": ENGINE_ENV,
            "requests": len(requests), "arrivals_span_s": round(arr[-1], 3)}
    t = clock()
    with open(d / "server.log", "wb") as log:
        server = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        wait_ready(server, port, clock() + args.startup_s)
        meta["startup_s"] = round(clock() - t, 1)
        print(f"[corpus] {name}: server up in {meta['startup_s']} s; {len(requests)} requests at {rate}/s", flush=True)
        meta |= asyncio.run(drive(port, requests, arr, ids, d / "client.jsonl", args.drain_s))
    finally:
        stop(server)
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


def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    t = clock()
    docs = load_docs(Path(args.c4), Path(args.model) / "tokenizer.json", args.docs)
    requests = traffic.plan(docs, args.requests, args.seed)
    print(f"[corpus] plan: {len(requests)} requests over {len(docs)} documents in {round(clock() - t, 1)} s", flush=True)
    with open(out / "plan.jsonl", "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    traces = [n for n in args.traces.split(",") if n]
    if bad := [n for n in traces if n not in TRACES]:
        raise SystemExit(f"unknown traces {bad}; known: {sorted(TRACES)}")
    meta = {"model": args.model, "c4": args.c4, "docs": args.docs, "seed": args.seed, "requests": args.requests,
            "eos_token_ids": list(EOS), "traffic": {"lengths": traffic.LENGTHS, "prefixes": traffic.PREFIXES,
                                                    "shared": traffic.SHARED, "at_end": traffic.AT_END, "gap": traffic.GAP,
                                                    "caps": traffic.CAPS},
            "versions": versions(), "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "GPU_LEASE_UUID",
                                                                            "GPU_LEASE_WHO", "RESEARCH_RUN_ID")},
            "host": socket.gethostname(), "traces": {}}
    rc = 0
    for name in traces:
        try:
            meta["traces"][name] = run_trace(name, out, args.model, requests, plugin_site(out / ".site"), args)
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
    r.add_argument("--model", required=True, help="the checkpoint's directory")
    r.add_argument("--c4", required=True, help="a C4 jsonl.gz shard")
    r.add_argument("--docs", type=int, default=20000)
    r.add_argument("--requests", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--traces", default=",".join(ORDER))
    r.add_argument("--startup-s", type=float, default=600)
    r.add_argument("--drain-s", type=float, default=600)
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    import summary

    summary.summarize(Path(args.dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
