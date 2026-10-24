"""A warden run (`warden_serve.py rehearse`'s files) over a served window's traffic, for `a5_warden.py --run`: the requests of
a `verity-vllm serve-openai` run's closed Commit window K, sent through the warden in the order the run recorded them, and an
upstream that answers each with the tokens the server streamed it, so the warden's cells carry that window's requests and
responses.

* **The requests** (`traffic`): each request of ``requests.jsonl`` with rows in window K, as a streamed ``/v1/completions`` body:
  the served model (``serve.json``'s ``served_model_name``), what its ingress record says it brought, and its prompt as the
  token ids the server recorded (``prompt_ids``), from ``--start`` at the times the run recorded (`arrivals`), or ``--gap``
  seconds apart.
* **The responses** (`replay_endpoint`): each request, matched by its prompt's token ids, answered with the ids the server
  streamed it (``streamed_ids``), one server-sent event a token in vLLM's completion form (``token_ids``, no text), its finish
  reason the recorded one, then its usage and ``[DONE]``. serve-openai records token ids, not the HTTP bytes it sent, so the
  bytes are rendered again from them; a request the window doesn't hold is answered 404.
* **The run** (`rehearse`): as `warden_serve.py rehearse` runs one (a key in scratch, ``serve`` and ``client`` as processes, then
  ``check``), with that traffic and that upstream. ``--out`` holds the warden's files (``deployment.json``, the records,
  ``cells.jsonl``; ``cells.salts`` is witness) and ``check.json``.

    python benchmarks/one_stage/warden_replay.py rehearse --serve-run RUN --serve-window K --out DIR [--gap S] [--token-ms 20] [schedule]
    python benchmarks/one_stage/warden_replay.py endpoint --serve-run RUN --serve-window K --listen HOST:PORT [--token-ms 20]

``--serve-window`` is the served run's Commit window; ``--window`` is the warden's own, in buckets (`WardenParams`).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TRACES = HERE.parent / "network_traces"
sys.path.insert(0, str(TRACES))

import warden_serve as ws  # noqa: E402
from verity.core.protocols.certifier.warden import WardenParams  # noqa: E402


def served_requests(run: str | Path, window: int) -> list[dict]:
    """The requests of ``requests.jsonl`` with rows in Commit window ``window``, in the order the run recorded them."""
    lines = (Path(run) / "requests.jsonl").read_text().splitlines()
    out = [q for q in map(json.loads, filter(str.strip, lines)) if int(window) in (q.get("windows") or [])]
    if not out:
        raise SystemExit(f"{run}/requests.jsonl holds no request with rows in window {window}")
    return out


def served_model(run: str | Path) -> str:
    serve = json.loads((Path(run) / "serve.json").read_text())
    return str(serve.get("served_model_name") or serve["model"])


def body_of(q: Mapping[str, Any], model: str) -> dict:
    """A recorded request as the streamed ``/v1/completions`` body the client sends: what its ingress brought, its prompt as
    the recorded token ids."""
    brought = {k: v for k, v in ((q.get("ingress") or {}).get("brought") or {}).items() if k != "prompt"}
    return {"model": model, **brought, "prompt": [int(t) for t in q["prompt_ids"]], "stream": True,
            "stream_options": {"include_usage": True}}


def arrivals(run: str | Path, requests: list[Mapping[str, Any]]) -> list[float]:
    """Each request's time as the run recorded it, in seconds after the earliest: the end of the engine step its first row ran
    in (``step_trace.jsonl``'s ``t_end``), since serve-openai records no arrival time."""
    lines = (Path(run) / "step_trace.jsonl").read_text().splitlines()
    ends = {(int(d["window"]), int(d["step"])): d.get("t_end") for d in map(json.loads, filter(str.strip, lines))}
    out = []
    for q in requests:
        w, s = (int(x) for x in q["rows"][0][:2])
        if ends.get((w, s)) is None:
            raise SystemExit(f"{run}/step_trace.jsonl records no end for window {w}'s step {s}, request {q.get('http_id')}'s first")
        out.append(float(ends[w, s]))
    return [t - min(out) for t in out]


def traffic(run: str | Path, window: int, *, start_s: float = 1.0, gap_s: float | None = None) -> list[dict]:
    """`warden_serve`'s traffic form, {"at", "body"} a request, of window ``window``'s requests in the order they arrive: at
    the times the run recorded (`arrivals`), or ``gap_s`` apart in the order it recorded them."""
    model, requests = served_model(run), served_requests(run, window)
    at = arrivals(run, requests) if gap_s is None else [i * gap_s for i in range(len(requests))]
    return [{"at": round(start_s + t, 3), "body": body_of(q, model)} for t, q in sorted(zip(at, requests), key=lambda tq: tq[0])]


def answers(requests: list[Mapping[str, Any]]) -> dict[tuple[int, ...], list[Mapping[str, Any]]]:
    """Each prompt's recorded requests, in order: a prompt sent twice is answered in the order the run recorded."""
    out: dict[tuple[int, ...], list[Mapping[str, Any]]] = {}
    for q in requests:
        out.setdefault(tuple(int(t) for t in q["prompt_ids"]), []).append(q)
    return out


def events(q: Mapping[str, Any], model: str) -> list[dict]:
    """The server-sent events that answer ``q``: a token each (its id, no text), the recorded finish reason on the last, then the
    usage."""
    ids, cid = [int(t) for t in q["streamed_ids"]], f"cmpl-{q.get('http_id') or q.get('engine_id')}"
    finish = (q.get("finish") or {}).get("reason") or "stop"
    out = [{"id": cid, "object": "text_completion", "created": 0, "model": model,
            "choices": [{"index": 0, "text": "", "token_ids": [t], "logprobs": None,
                         "finish_reason": finish if k == len(ids) - 1 else None, "stop_reason": None}]}
           for k, t in enumerate(ids)]
    n = len(q["prompt_ids"])
    return out + [{"id": cid, "object": "text_completion", "created": 0, "model": model, "choices": [],
                   "usage": {"prompt_tokens": n, "completion_tokens": len(ids), "total_tokens": n + len(ids)}}]


async def replay_endpoint(reader: asyncio.StreamReader, writer, *, held: dict, model: str, sleep=asyncio.sleep,
                          first_s: float = 0.1, token_s: float = 0.02) -> None:
    """One request's recorded answer (`events`), its first event ``first_s`` after the request and one every ``token_s``; 404 for
    a prompt ``held`` has no answer left for."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
        m = re.search(rb"(?im)^content-length:\s*(\d+)", head)
        body = json.loads(await reader.readexactly(int(m.group(1)))) if m else {}
        queue = held.get(tuple(int(t) for t in body.get("prompt") or []))
        if not queue:
            doc = b'{"error": "no recorded request of the window has this prompt"}'
            writer.write(b"HTTP/1.1 404 Not Found\r\ncontent-type: application/json\r\nconnection: close\r\n"
                         b"content-length: %d\r\n\r\n%s" % (len(doc), doc))
            await writer.drain()
            return
        q = queue.pop(0)
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream; charset=utf-8\r\nconnection: close\r\n"
                     b"transfer-encoding: chunked\r\n\r\n")
        await sleep(first_s)
        for doc in events(q, model):
            writer.write(ws._sse(doc))
            await writer.drain()
            await sleep(token_s)
        writer.write(b"e\r\ndata: [DONE]\n\n\r\n0\r\n\r\n")
        await writer.drain()
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError, ValueError):
        pass
    finally:
        writer.close()


def endpoint(args) -> int:
    held, model = answers(served_requests(args.serve_run, args.serve_window)), served_model(args.serve_run)

    async def run() -> None:
        server = await asyncio.start_server(
            lambda rd, wr: replay_endpoint(rd, wr, held=held, model=model, token_s=args.token_ms / 1000), *args.listen)
        ws.cs._announce({"listening": [args.listen[0], server.sockets[0].getsockname()[1]], "model": model,
                         "requests": sum(map(len, held.values()))})
        await server.serve_forever()

    asyncio.run(run())
    return 0


def rehearse(args) -> int:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    params = ws._params(args)
    sent = traffic(args.serve_run, args.serve_window, start_s=args.start, gap_s=args.gap)
    (out / "traffic.json").write_text(json.dumps(sent))
    warden = [sys.executable, str(TRACES / "warden_serve.py")]
    scratch = Path(tempfile.mkdtemp(prefix="warden-replay-"))
    procs: list[subprocess.Popen] = []
    try:
        ws.cs.load_key(scratch / "warden", args.height)
        up = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "endpoint", "--serve-run", str(args.serve_run),
                               "--serve-window", str(args.serve_window), "--listen", "127.0.0.1:0", "--token-ms", str(args.token_ms)],
                              stdout=subprocess.PIPE, text=True)
        procs.append(up)
        upstream = tuple(json.loads(up.stdout.readline())["listening"])
        windows = math.ceil((sent[-1]["at"] + args.tail) / (params.window * params.bucket_ms / 1000))
        with open(out / "serve.log", "w") as log:
            serving = subprocess.Popen(warden + ["serve", "--upstream", "%s:%d" % upstream, "--listen", "127.0.0.1:0", "--out",
                                                 str(out), "--key", str(scratch / "warden"), "--session", secrets.token_hex(32),
                                                 "--windows", str(windows), "--start-in", str(args.start_in), "--slack",
                                                 str(args.slack), *ws._flags(args)],
                                       stdout=subprocess.PIPE, stderr=log, text=True)
            procs.append(serving)
            if "listening" not in json.loads(serving.stdout.readline() or "{}"):
                raise SystemExit("serve exited before listening (serve.log)")
            with open(out / "client.log", "w") as clog:
                client_exit = subprocess.run(warden + ["client", "--deployment", str(out / "deployment.json"), "--traffic",
                                                       str(out / "traffic.json"), "--out", str(out)],
                                             stdout=clog, stderr=subprocess.STDOUT).returncode
            serve_exit = serving.wait()
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        shutil.rmtree(scratch)
    result = ws.check(out) | {"served": {"run": str(args.serve_run), "window": args.serve_window, "requests": len(sent)},
                              "upstream": list(upstream), "model": sent[0]["body"]["model"],
                              "exit": {"serve": serve_exit, "client": client_exit}}
    (out / "check.json").write_text(json.dumps(result, indent=1))
    print(ws.table(result), flush=True)
    return 0 if result["accepted"] else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="warden_replay.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    e, r = sub.add_parser("endpoint", help="the window's recorded answers"), sub.add_parser("rehearse", help="the warden run")
    for q in (e, r):
        q.add_argument("--serve-run", required=True, type=Path, help="a serve-openai run dir (serve.json, requests.jsonl)")
        q.add_argument("--serve-window", required=True, type=int, help="the served run's closed Commit window K")
        q.add_argument("--token-ms", type=float, default=20.0, help="milliseconds between a response's events")
    e.add_argument("--listen", required=True, type=ws.cs._address, help="HOST:PORT (port 0: any)")
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--start", type=float, default=1.0, help="the first request's time after the origin, in seconds")
    r.add_argument("--gap", type=float, help="seconds between requests (default: the times the run recorded, `arrivals`)")
    r.add_argument("--tail", type=float, default=10.0, help="seconds the run lasts past the last request's time")
    r.add_argument("--height", type=int, default=ws.cs.HEIGHT, help="the LMS key's tree height")
    r.add_argument("--start-in", type=float, default=2.0, help="the origin, seconds after the warden is up")
    r.add_argument("--slack", type=float, default=0.025, help="seconds a tick may run late before it faults")
    for f in dataclasses.fields(WardenParams):
        r.add_argument(f"--{f.name.replace('_', '-')}", type=int, default=ws.SCHEDULE[f.name])
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return endpoint(args) if args.command == "endpoint" else rehearse(args)


if __name__ == "__main__":
    raise SystemExit(main())
