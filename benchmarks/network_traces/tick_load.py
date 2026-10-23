"""tick-load: the network certifier's tick lateness in a served rehearsal on a stub endpoint, beside a CPU hog confined to
this process's own CPUs.

    python tick_load.py --out DIR [--hogs-per-cpu 3] [--requests 400] [--in-flight 4] [--tokens 64]
        [--token-ms 25] [--delta-buckets 52] [--profile planned|served] [--strategies honest[,late,extra]]

It starts a stub OpenAI streaming endpoint on loopback (`/v1/models`, `/tokenize`, and `/v1/completions`, each
completion `max_tokens` chunks `--token-ms` apart, side by side, as vLLM decodes the requests in flight), then
`--hogs-per-cpu` busy loops per CPU of this process's affinity (the run's cpuset), each pinned to that set with
`taskset`, then `certifier_serve.py rehearse` against the stub (`--strategies`, honest by default, on N2's
`--profile`), and stops the hogs. It writes
`ticks.json` in `--out`: the CPUs and hogs, the rehearsal's exit and verdict, and per grid the ticks and their lateness
(median, p99, p99.9, max in ms, the ticks past the slack with their window and bucket, and the windows the honest run
was refused on). The rehearsal's own outputs are under `--out`/rehearsal.

Run it inside one `research run --on auto --cpus N`, so the hog loads only the run's own cores.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Endpoint:
    """The stub endpoint: one model, a tokenizer that makes a token of every byte, and streamed completions paced
    `token_s` apart, every connection on its own."""

    def __init__(self, token_s: float) -> None:
        self.token_s = token_s

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            n = re.search(rb"content-length: (\d+)", head, re.I)
            body = json.loads(await reader.readexactly(int(n.group(1)))) if n else None
            if head.startswith(b"POST /v1/completions "):
                await self.complete(body, writer)
            else:
                doc = ({"data": [{"id": "stub-model"}]} if head.startswith(b"GET /v1/models ") else
                       {"tokens": list(body["prompt"].encode()), "count": len(body["prompt"].encode())})
                data = json.dumps(doc).encode()
                writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n%s"
                             % (len(data), data))
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    async def complete(self, body: dict, writer: asyncio.StreamWriter) -> None:
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
        loop = asyncio.get_running_loop()
        start = loop.time()
        for k in range(body["max_tokens"]):
            await asyncio.sleep(max(0.0, start + (k + 1) * self.token_s - loop.time()))
            p = b'data: {"choices":[{"text":" t%d"}]}\n\n' % k
            writer.write(b"%x\r\n%s\r\n" % (len(p), p))
            await writer.drain()
        done = b"data: [DONE]\n\n"
        writer.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(done), done))
        await writer.drain()


def serve_endpoint(token_s: float) -> int:
    """Start the stub on a loopback port in a thread of its own; its port."""
    ready: list = []
    started = threading.Event()

    def main() -> None:
        async def go() -> None:
            server = await asyncio.start_server(Endpoint(token_s).handle, "127.0.0.1", 0)
            ready.append(server.sockets[0].getsockname()[1])
            started.set()
            await server.serve_forever()

        asyncio.run(go())

    threading.Thread(target=main, daemon=True).start()
    started.wait()
    return ready[0]


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def publisher(timing: dict, log: dict, buckets: int) -> dict:
    """The publisher's calls (`publisher_timing.json`, on the proxy's clock): wall and thread CPU seconds of each, and
    per grid the lateness of the ticks whose wait [deadline, deadline + lateness] overlaps a call's [start, end] against
    the rest's."""
    calls = [c for c in timing["calls"] if "end" in c]
    spans = sorted((c["start"], c["end"]) for c in calls)
    def ms(xs: list[float]) -> dict:
        return {"n": len(xs), "median_ms": _quantile(xs, 0.5) if xs else None,
                "p99_ms": _quantile(xs, 0.99) if xs else None, "max_ms": max(xs) if xs else None}

    out = {"switchinterval_s": timing["switchinterval"], "calls": len(calls),
           "wall": ms([(c["end"] - c["start"]) * 1000 for c in calls]),
           "keep_cpu": ms([c["keep_cpu"] * 1000 for c in calls]),
           "certify_cpu": ms([c["certify_cpu"] * 1000 for c in calls]),
           "queued": ms([(c["start"] - c["submit"]) * 1000 for c in calls])}
    for grid, lateness in log["lateness_ms"].items():
        tau = timing["bucket_ms"][grid] / 1000
        over, rest, late = [], [], []
        for k, x in enumerate(lateness):
            t = timing["start"] * buckets + k
            a = timing["origin"] + t * tau
            b = a + max(x, 0.0) / 1000
            hit = [s for s in spans if s[0] < b and s[1] > a]
            (over if hit else rest).append(x)
            if x > 10:
                late.append([t, t // buckets, t % buckets, x, bool(hit)])
        out[grid] = {"overlapping": ms(over), "not_overlapping": ms(rest), "past_10ms": late}
    return out


def summary(run: Path) -> dict:
    """Per grid, the honest run's tick lateness (`proxy.json.gz`), and its refused windows (`check.json`)."""
    log = json.loads(gzip.open(run / "proxy.json.gz").read())
    buckets = json.loads((run / "calibration.json").read_text())["grid"]["window_buckets"]
    slack = log["slack_ms"]
    out = {"slack_ms": slack}
    for grid, lateness in log["lateness_ms"].items():
        late = [[t, t // buckets, t % buckets, x] for t, x in enumerate(lateness) if x > slack]
        out[grid] = {"ticks": len(lateness), "median_ms": _quantile(lateness, 0.5), "p99_ms": _quantile(lateness, 0.99),
                     "p999_ms": _quantile(lateness, 0.999), "max_ms": max(lateness),
                     "past_slack": late, "worst": sorted(enumerate(lateness), key=lambda tx: -tx[1])[:8]}
    for grid, lateness in log["lateness_ms"].items():
        pos = {"first": [x for t, x in enumerate(lateness) if t % buckets == 0],
               "last": [x for t, x in enumerate(lateness) if t % buckets == buckets - 1],
               "other": [x for t, x in enumerate(lateness) if 0 < t % buckets < buckets - 1]}
        out[grid]["by_position"] = {k: {"n": len(v), "median_ms": _quantile(v, 0.5), "p99_ms": _quantile(v, 0.99),
                                        "max_ms": max(v)} for k, v in pos.items() if v}
    if (run / "publisher_timing.json").exists():
        out["publisher"] = publisher(json.loads((run / "publisher_timing.json").read_text()), log, buckets)
    check = json.loads((run / "check.json").read_text()) if (run / "check.json").exists() else {}
    out["verdict"] = check.get("verdict")
    out["refused_windows"] = {row["window"]: {d: a["raw"] for d, a in row["audit"].items() if a["raw"]}
                              for row in check.get("rows", []) if not row["n2"]["accepted"]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tick_load.py", description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hogs-per-cpu", type=int, default=3)
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--in-flight", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--token-ms", type=float, default=25.0)
    ap.add_argument("--delta-buckets", type=int, default=52)
    ap.add_argument("--profile", default="planned", help="N2's profile, planned or served (`certifier_serve.py plan`)")
    ap.add_argument("--strategies", default="honest", help="the rehearsal's strategies, honest, late, extra")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0))
    port = serve_endpoint(a.token_ms / 1000)
    cpu_list = ",".join(map(str, cpus))
    hogs = [subprocess.Popen(["taskset", "-c", cpu_list, sys.executable, "-c", "while True: pass"])
            for _ in range(a.hogs_per_cpu * len(cpus))]
    try:
        rc = subprocess.run([sys.executable, str(HERE / "certifier_serve.py"), "rehearse",
                             "--upstream", f"http://127.0.0.1:{port}", "--requests", str(a.requests),
                             "--in-flight", str(a.in_flight), "--tokens", str(a.tokens),
                             "--delta-buckets", str(a.delta_buckets), "--profile", a.profile,
                             "--strategies", a.strategies, "--out", str(a.out / "rehearsal")],
                            env=os.environ | {"CERTIFIER_SERVE_TIMING": "1"}).returncode
    finally:
        for h in hogs:
            h.kill()
        for h in hogs:
            h.wait()
    doc = {"cpus": cpus, "hogs": len(hogs), "rehearse_exit": rc, "token_ms": a.token_ms, "requests": a.requests,
           "in_flight": a.in_flight, "delta_buckets": a.delta_buckets, "profile": a.profile,
           "strategies": a.strategies}
    run = a.out / "rehearsal" / "honest"
    if (run / "proxy.json.gz").exists():
        doc |= summary(run)
    (a.out / "ticks.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc.get(k) for k in ("verdict", "refused_windows")} |
                     {g: {k: doc[g][k] for k in ("ticks", "p999_ms", "max_ms")} | {"past_slack": len(doc[g]["past_slack"])}
                      for g in ("egress", "ingress") if g in doc}), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
