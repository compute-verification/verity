"""The link a batched PoUS round runs over, measured as the round itself sees it: one side answers like the prover, the other
times like the memory challenger.

    net.py serve --port P [--listen ADDR] [--minutes M]              # the prover's side: answers until M minutes pass
    net.py probe --host H --port P --out DIR [--batches 1,64,...] [--reps R] [--bulk-mb MB] [--streams N]

A round is `u32be b ‖ b × u64be index` out and `b × BLOCK` bytes back (2,056, P2 v4's block), the time from the request's
send to the last byte's receipt on `time.perf_counter_ns`, as `challenger_party.serve` reads it. `probe` writes `net.json`:
each batch size's round times (`quantiles`), the TCP connect time (one RTT), and the bulk goodput of `--bulk-mb` MB sent by
the prover's side over each of `--streams` connections at once. `b = 0xFFFFFFFF` asks for `u64be n` bulk bytes instead.
The socket buffers are the kernel's defaults (both sides' `net.ipv4.tcp_rmem` / `tcp_wmem` are recorded).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import threading
import time
from pathlib import Path

BLOCK = 2_056
BULK = 0xFFFF_FFFF


def _read(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise ConnectionError(f"the peer closed after {len(b)} of {n} bytes")
    return b


def _sysctls() -> dict[str, str]:
    out = {}
    for k in ("net/ipv4/tcp_rmem", "net/ipv4/tcp_wmem", "net/core/rmem_max", "net/core/wmem_max",
              "net/ipv4/tcp_congestion_control"):
        try:
            out[k] = Path("/proc/sys", k).read_text().strip()
        except OSError:
            out[k] = None
    return out


def _answer(conn: socket.socket, blob: bytes) -> None:
    with conn, conn.makefile("rwb") as f:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            try:
                b = int.from_bytes(_read(f, 4), "big")
            except ConnectionError:
                return
            if b == BULK:
                n = int.from_bytes(_read(f, 8), "big")
                while n > 0:
                    f.write(blob[:min(n, len(blob))])
                    n -= min(n, len(blob))
                f.flush()
                continue
            _read(f, 8 * b)
            f.write(blob[:b * BLOCK])
            f.flush()


def serve(port: int, listen: str, minutes: float) -> None:
    blob = os.urandom(BLOCK * 65_536)
    with socket.create_server((listen, port)) as srv:
        srv.settimeout(1.0)
        print(json.dumps({"listening": port, "host": socket.gethostname(), "sysctl": _sysctls()}), flush=True)
        end = time.monotonic() + 60 * minutes
        while time.monotonic() < end:
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            threading.Thread(target=_answer, args=(conn, blob), daemon=True).start()


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    at = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    return {"n": len(s), "min": s[0], "p50": statistics.median(s), "p90": at(0.9), "p99": at(0.99), "max": s[-1]}


def _connect(host: str, port: int) -> tuple[socket.socket, float]:
    t = time.perf_counter_ns()
    s = socket.create_connection((host, port), timeout=30)
    ms = (time.perf_counter_ns() - t) / 1e6
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s, ms


def _bulk(host: str, port: int, nbytes: int, out: list) -> None:
    s, _ = _connect(host, port)
    with s, s.makefile("rwb") as f:
        t = time.perf_counter_ns()
        f.write(BULK.to_bytes(4, "big") + nbytes.to_bytes(8, "big"))
        f.flush()
        got = 0
        while got < nbytes:
            chunk = f.read1(min(1 << 20, nbytes - got))
            if not chunk:
                raise ConnectionError("bulk: the peer closed early")
            got += len(chunk)
        out.append((time.perf_counter_ns() - t) / 1e9)


def probe(host: str, port: int, out: Path, batches: list[int], reps: int, bulk_mb: int, streams: list[int]) -> dict:
    doc: dict = {"host": socket.gethostname(), "peer": f"{host}:{port}", "block_bytes": BLOCK, "sysctl": _sysctls()}
    connects = []
    for _ in range(5):
        s, ms = _connect(host, port)
        s.close()
        connects.append(ms)
    doc["connect_ms"] = quantiles(connects)
    s, _ = _connect(host, port)
    rounds = {}
    with s, s.makefile("rwb") as f:
        for b in batches:
            xs = []
            for _ in range(reps):
                req = b.to_bytes(4, "big") + bytes(8 * b)
                t = time.perf_counter_ns()
                f.write(req)
                f.flush()
                _read(f, b * BLOCK)
                xs.append((time.perf_counter_ns() - t) / 1e6)
            rounds[str(b)] = quantiles(xs) | {"bytes": b * BLOCK}
            print(f"b={b}: p50 {rounds[str(b)]['p50']:.3f} ms, max {rounds[str(b)]['max']:.3f} ms", flush=True)
    doc["round_ms"] = rounds
    bulk = {}
    for n in streams:
        secs: list[float] = []
        each = bulk_mb * (1 << 20) // n
        t = time.perf_counter_ns()
        ths = [threading.Thread(target=_bulk, args=(host, port, each, secs)) for _ in range(n)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        wall = (time.perf_counter_ns() - t) / 1e9
        bulk[str(n)] = {"bytes": each * n, "wall_s": wall, "MB_per_s": each * n / wall / 1e6, "per_stream_s": secs}
        print(f"bulk {n} stream(s): {bulk[str(n)]['MB_per_s']:.1f} MB/s over {wall:.2f} s", flush=True)
    doc["bulk"] = bulk
    out.mkdir(parents=True, exist_ok=True)
    (out / "net.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--listen", default="0.0.0.0")
    s.add_argument("--minutes", type=float, default=15)
    p = sub.add_parser("probe")
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batches", default="1,16,64,256,1024,4096,16384")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--bulk-mb", type=int, default=256)
    p.add_argument("--streams", default="1,4")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        serve(a.port, a.listen, a.minutes)
    else:
        probe(a.host, a.port, a.out, [int(x) for x in a.batches.split(",")], a.reps, a.bulk_mb,
              [int(x) for x in a.streams.split(",")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
