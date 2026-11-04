"""The streamed-label wipe's timing on one node: the labels a GPU generates, what reaches host memory and the network
stack, and the challenge round trips, so the fill time and the challenger's GPU budget are measured
(`verity/core/protocols/pous/PROTOCOL.md`, "The streamed-label wipe").

`bench.py all --bin WIPELABELS --scratch DIR --out DIR` runs every step and writes `timing.json`:

- `gates`: the GPU's labels against `streamed_wipe.label`, at the start of the index range and across the index's
  word boundary.  A failed gate times nothing.
- `gpu`: `N = 40,625,000,000` labels (the 2.6 TB deployed fill) on the one GPU, stored to device memory and XORed
  (compute alone); then generate-and-copy to pinned host memory on two streams.
- `link`: the NIC's line rate from sysfs, TCP over loopback at 1 to 16 connections, and `ib_write_bw` through the HCA's
  loopback when perftest is installed.  Between the nodes only TCP 22 passes, so no inter-node stream is measured.
- `rtt`: a prover process holding `2^28` labels (16 GiB, dumped by the GPU) in memory answers `k = 74,000` challenges
  over loopback TCP at batch sizes 1 (the sequential reveal the guarantee covers) to 74,000, every answer checked.
- `session`: one whole session of the reference (`streamed_wipe.WipeMemoryChallenger`, test signer) against the honest
  prover in its own process over TCP: a `2^20`-label fill streamed, then 74,000 sequential challenges, signed, and
  judged by `wipe_verdict.judge` (the Lean verdict, `streamed-wipe-verdict`, unless `--verdict` says otherwise), each
  side's verdict recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import mmap
import multiprocessing as mp
import os
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "verity/core/protocols/pous/tests")]

from verity.core.protocols.pous import streamed_wipe as sw  # noqa: E402

WIPE_VERDICT = ROOT / "verity/core/devices/memory_challenger/lean/wipe_verdict.py"
SECRET, SERVER = b"streamed-wipe bench secret", b"bench-node/bench"
SESSION = hashlib.sha512(b"streamed-wipe bench session").digest()[:32]
TABLE_LABELS = 1 << 28
SESSION_LABELS = 1 << 20
BATCHES = (1, 16, 256, 4096, sw.DEPLOYED_K)


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if not r:
            raise ConnectionError("peer closed")
        got += r
    return buf


def _bin_json(cmd: list[str]) -> tuple[dict, list[str]]:
    out = subprocess.run(cmd, check=False, capture_output=True, text=True)
    lines = out.stdout.splitlines()
    if out.returncode or not lines:
        raise RuntimeError(f"{cmd[1]} exited {out.returncode}: {out.stderr.strip()[-400:]}")
    return json.loads(lines[0]), lines[1:]


def gates(binary: str, key: bytes) -> dict:
    out = {}
    for start, count in ((0, 64), ((1 << 40) - 3, 8), ((1 << 64) - 4, 4)):
        head, lines = _bin_json([binary, "check", "--key", key.hex(), "--start", str(start), "--count", str(count)])
        ok = [bytes.fromhex(x) == sw.label(key, start + t) for t, x in enumerate(lines)]
        out[str(start)] = {"count": count, "host_mismatches": head["host_mismatches"], "reference_matches": sum(ok),
                           "device": head["device"]}
        if head["host_mismatches"] or not all(ok) or len(ok) != count:
            raise SystemExit(f"GATE FAILED at start {start}: {out[str(start)]}")
    return out


def gpu(binary: str, key: bytes) -> dict:
    rate, _ = _bin_json([binary, "rate", "--key", key.hex(), "--labels", str(sw.DEPLOYED_LABELS), "--chunk", str(1 << 26)])
    stream, _ = _bin_json([binary, "stream", "--key", key.hex(), "--labels", str(1 << 32), "--chunk", str(1 << 22)])
    return {"rate": rate, "stream": stream,
            "fill_s_at_store_rate": sw.DEPLOYED_FILL_BYTES / (rate["store_GBps"] * 1e9)}


def _sink(port: int, total: int, ready) -> None:
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    ready.set()
    conn, _ = srv.accept()
    buf = bytearray(16 << 20)
    view, got = memoryview(buf), 0
    while got < total:
        r = conn.recv_into(view)
        if not r:
            break
        got += r
    conn.sendall(b"k")
    conn.close()


def _source(port: int, total: int, go) -> None:
    s = socket.create_connection(("127.0.0.1", port))
    buf = memoryview(os.urandom(16 << 20))
    go.wait()
    sent = 0
    while sent < total:
        n = min(len(buf), total - sent)
        s.sendall(buf[:n])
        sent += n
    s.recv(1)
    s.close()


def tcp_loopback(conns: int, per_conn: int, base_port: int) -> dict:
    ctx = mp.get_context("fork")
    readies = [ctx.Event() for _ in range(conns)]
    go = ctx.Event()
    sinks = [ctx.Process(target=_sink, args=(base_port + i, per_conn, readies[i])) for i in range(conns)]
    for p in sinks:
        p.start()
    for r in readies:
        r.wait()
    sources = [ctx.Process(target=_source, args=(base_port + i, per_conn, go)) for i in range(conns)]
    for p in sources:
        p.start()
    time.sleep(0.5)  # every source connected before the clock starts
    t0 = time.perf_counter()
    go.set()
    for p in sources:
        p.join()
    dt = time.perf_counter() - t0
    for p in sinks:
        p.join()
    return {"connections": conns, "bytes": conns * per_conn, "s": dt, "GBps": conns * per_conn / dt / 1e9}


def link(base_port: int) -> dict:
    out: dict = {"interfaces": {}, "infiniband": {}}
    for i in sorted(Path("/sys/class/net").iterdir()):
        try:
            out["interfaces"][i.name] = {"speed_mbps": int((i / "speed").read_text()), "mtu": int((i / "mtu").read_text())}
        except (OSError, ValueError):
            pass
    for p in sorted(Path("/sys/class/infiniband").glob("*/ports/*")) if Path("/sys/class/infiniband").exists() else []:
        out["infiniband"][str(p)] = {"rate": (p / "rate").read_text().strip(), "state": (p / "state").read_text().strip()}
    out["tcp_loopback"] = [tcp_loopback(c, (8 << 30) // c if c > 1 else 8 << 30, base_port + 100 * c)
                           for c in (1, 4, 8, 16)]
    out["ib_write_bw"] = ib_write_bw()
    return out


def ib_write_bw() -> dict:
    exe = shutil.which("ib_write_bw")
    if exe is None:
        return {"skipped": "perftest's ib_write_bw is not installed"}
    dev = next(iter(sorted(p.name for p in Path("/sys/class/infiniband").iterdir())), None)
    if dev is None:
        return {"skipped": "no RDMA device"}
    args = [exe, "-d", dev, "-s", str(1 << 20), "-D", "5", "--report_gbits", "-q", "4", "-F"]
    srv = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(1.0)  # perftest's server listens before its client connects
    cli = subprocess.run([*args, "127.0.0.1"], capture_output=True, text=True, timeout=60, check=False)
    try:
        srv_out = srv.communicate(timeout=30)[0]
    except subprocess.TimeoutExpired:
        srv.kill()
        srv_out = srv.communicate()[0]
    return {"device": dev, "rc": cli.returncode, "client": cli.stdout[-1500:] + cli.stderr[-500:], "server": srv_out[-800:]}


def prover_main(args: argparse.Namespace) -> None:
    """The server's side: answer challenges from a dumped table, or keep a fill it is streamed (the honest prover)."""
    if args.table:
        f = open(args.table, "rb")
        store = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        store.madvise(mmap.MADV_WILLNEED)
        for off in range(0, len(store), 1 << 12):  # every page resident before the first challenge
            store[off]
    else:
        store = bytearray(args.labels * sw.LABEL_BYTES)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print("ready", flush=True)
    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    L = sw.LABEL_BYTES
    while True:
        kind = _recv_exact(conn, 1)
        if kind == b"Q":
            break
        if kind == b"F":
            start, count = struct.unpack(">QI", _recv_exact(conn, 12))
            store[L * start:L * (start + count)] = _recv_exact(conn, L * count)
            continue
        (count,) = struct.unpack(">I", _recv_exact(conn, 4))
        idx = struct.unpack(f">{count}Q", _recv_exact(conn, 8 * count))
        conn.sendall(b"".join(store[L * i:L * i + L] for i in idx))
    conn.close()


def _spawn_prover(port: int, *, table: str | None = None, labels: int = 0) -> subprocess.Popen:
    cmd = [sys.executable, __file__, "prover", "--port", str(port)]
    cmd += ["--table", table] if table else ["--labels", str(labels)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    assert p.stdout is not None and p.stdout.readline().strip() == "ready"
    return p


def _connect(port: int) -> socket.socket:
    s = socket.create_connection(("127.0.0.1", port))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s


def _ask(s: socket.socket, idx) -> bytes:
    s.sendall(b"C" + struct.pack(f">I{len(idx)}Q", len(idx), *idx))
    return bytes(_recv_exact(s, sw.LABEL_BYTES * len(idx)))


def _stats(xs: list[int]) -> dict:
    xs = sorted(xs)
    return {"n": len(xs), "min_ns": xs[0], "p50_ns": xs[len(xs) // 2], "p99_ns": xs[min(len(xs) - 1, len(xs) * 99 // 100)],
            "max_ns": xs[-1], "mean_ns": statistics.fmean(xs)}


def rtt(binary: str, key: bytes, scratch: Path, port: int) -> dict:
    table = scratch / "labels.bin"
    dump, _ = _bin_json([binary, "dump", "--key", key.hex(), "--labels", str(TABLE_LABELS), "--chunk", str(1 << 24),
                         "--out", str(table)])
    prover = _spawn_prover(port, table=str(table))
    s = _connect(port)
    out: dict = {"table_labels": TABLE_LABELS, "dump": dump, "k": sw.DEPLOYED_K, "batches": {}}
    try:
        for batch in BATCHES:
            reps = 20 if batch == sw.DEPLOYED_K else 1
            sessions = []
            for rep in range(reps):
                idx = sw.draw_indices(os.urandom(32), SERVER, SESSION, TABLE_LABELS, sw.DEPLOYED_K)
                expect = [sw.label(key, i) for i in idx]
                per, wrong = [], 0
                t0 = time.perf_counter_ns()
                for first in range(0, len(idx), batch):
                    part = idx[first:first + batch]
                    a = time.perf_counter_ns()
                    got = _ask(s, part)
                    per.append(time.perf_counter_ns() - a)
                    wrong += sum(got[64 * t:64 * t + 64] != expect[first + t] for t in range(len(part)))
                total = time.perf_counter_ns() - t0
                if wrong:
                    raise SystemExit(f"RTT FAILED: {wrong} wrong answers at batch {batch}")
                sessions.append({"total_ns": total, "per_batch": _stats(per)})
            out["batches"][str(batch)] = {"reps": reps, "sessions": sessions,
                                          "total_ns_min": min(x["total_ns"] for x in sessions),
                                          "total_ns_median": statistics.median(x["total_ns"] for x in sessions)}
        s.sendall(b"Q")
        with open(table, "rb") as f, mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as store:
            for o in range(0, len(store), mmap.PAGESIZE):  # a fresh mapping faults on every page; the prover's doesn't
                store[o]
            looks = []
            for _ in range(5):
                idx = sw.draw_indices(os.urandom(32), SERVER, SESSION, TABLE_LABELS, sw.DEPLOYED_K)
                a = time.perf_counter_ns()
                b"".join(store[64 * i:64 * i + 64] for i in idx)
                looks.append(time.perf_counter_ns() - a)
        out["lookup_74000_ns"] = _stats(looks)
    finally:
        s.close()
        prover.wait(timeout=60)
        table.unlink()
    return out


def _wipe_verdict():
    spec = importlib.util.spec_from_file_location("streamed_wipe_lean_verdict", WIPE_VERDICT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def session(port: int, verdict: str) -> dict:
    import pous_test_signer as ts

    wv = _wipe_verdict()
    if verdict != "python":
        wv.server()  # the Lean verdict built before the session: where it can't be, the step stops here
    k, n = sw.DEPLOYED_K, SESSION_LABELS
    params = sw.WipeParams(n, k, 1, 50_000_000, (("host-dram", n * sw.LABEL_BYTES),))
    with tempfile.TemporaryDirectory(prefix="streamed-wipe-bench-") as scratch:
        signer = ts.TestSigner(b"streamed-wipe bench", height=2, directory=scratch)
        c = sw.WipeMemoryChallenger(SECRET, SERVER, SESSION, params, signer, log=ts.session_log(scratch))
        prover = _spawn_prover(port, labels=n)
        s = _connect(port)
        try:
            t0 = time.perf_counter_ns()
            for start, data in c.fill(1 << 14):
                s.sendall(b"F" + struct.pack(">QI", start, len(data) // sw.LABEL_BYTES) + data)
            t1 = time.perf_counter_ns()
            c.run(lambda idx: _ask(s, idx))
            t2 = time.perf_counter_ns()
            s.sendall(b"Q")
        finally:
            s.close()
            prover.wait(timeout=60)
        signed = c.close(egress_bits=0)
    t3 = time.perf_counter_ns()
    v, sides = wv.judge(signed, c.commitment, trusted={signer.public_key}, schemes=ts.SCHEMES, decide=verdict)
    t4 = time.perf_counter_ns()
    per = [r.elapsed_ns for r in signed.batches]
    return {"labels": n, "k": k, "ended": signed.ended, "accepted": v.accepted, "reasons": list(v.reasons),
            "verdict": verdict, "verdicts": sides, "verdict_s": (t4 - t3) / 1e9,
            "fill_s": (t1 - t0) / 1e9, "challenges_s": (t2 - t1) / 1e9, "per_round": _stats(per),
            "limit_ns": params.limit_ns}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all")
    a.add_argument("--bin", required=True)
    a.add_argument("--scratch", required=True, type=Path)
    a.add_argument("--out", required=True, type=Path)
    a.add_argument("--port", type=int, default=47100)
    a.add_argument("--verdict", choices=("lean", "both", "python"), default=None,
                   help="who decides the session's verdict (wipe_verdict.py; its VERDICT by default)")
    p = sub.add_parser("prover")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--table")
    p.add_argument("--labels", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "prover":
        prover_main(args)
        return
    key = sw.wipe_key(SECRET, SERVER, SESSION)
    res: dict = {"schema": "verity-pous/streamed-wipe-timing/v1", "label_tag": sw.LABEL_TAG.decode().rstrip("\0"),
                 "deployed": {"fill_bytes": sw.DEPLOYED_FILL_BYTES, "labels": sw.DEPLOYED_LABELS, "k": sw.DEPLOYED_K}}
    path = args.out / "timing.json"

    def save() -> None:
        path.write_text(json.dumps(res, indent=1) + "\n")

    for name, step in (("gates", lambda: gates(args.bin, key)), ("gpu", lambda: gpu(args.bin, key)),
                       ("rtt", lambda: rtt(args.bin, key, args.scratch, args.port)),
                       ("session", lambda: session(args.port + 1, args.verdict or _wipe_verdict().VERDICT)),
                       ("link", lambda: link(args.port + 2))):
        t = time.perf_counter()
        res[name] = step()
        res[name + "_wall_s"] = time.perf_counter() - t
        save()
        print(name, json.dumps(res[name])[:600], flush=True)
    g, rt = res["gpu"], res["rtt"]["batches"]
    line_GBps = max((v["speed_mbps"] for v in res["link"]["interfaces"].values()), default=0) / 8e3
    res["summary"] = {
        "gpu_label_GBps": g["rate"]["store_GBps"], "gpu_label_compute_GBps": g["rate"]["xor_GBps"],
        "gpu_to_host_GBps": g["stream"]["GBps"], "deployed_fill_on_one_gpu_s": g["rate"]["store_s"],
        "nic_line_GBps": line_GBps, "fill_s_at_line_rate": sw.DEPLOYED_FILL_BYTES / (line_GBps * 1e9) if line_GBps else None,
        "servers_one_gpu_labels_at_line_rate": g["rate"]["store_GBps"] / line_GBps if line_GBps else None,
        "tcp_loopback_best_GBps": max(x["GBps"] for x in res["link"]["tcp_loopback"]),
        "batched_74000_rtt_ms_min": rt[str(sw.DEPLOYED_K)]["total_ns_min"] / 1e6,
        "batched_74000_rtt_ms_median": rt[str(sw.DEPLOYED_K)]["total_ns_median"] / 1e6,
        "sequential_74000_s": rt["1"]["total_ns_min"] / 1e9,
        "sequential_round_p50_us": rt["1"]["sessions"][0]["per_batch"]["p50_ns"] / 1e3,
        "sequential_round_p99_us": rt["1"]["sessions"][0]["per_batch"]["p99_ns"] / 1e3,
        "reference_session_accepted": res["session"]["accepted"],
    }
    save()
    print("summary", json.dumps(res["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
