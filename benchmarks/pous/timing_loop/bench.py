"""PoUS's timing loop measured three ways against one responder on the same host: how much of the window
(Δ = `audit.DEADLINE_NS` plus the RTT allowance) each memory challenger architecture spends.

    python benchmarks/pous/timing_loop/bench.py --out DIR --scratch DIR --lean-exe EXE [--cpu-responder] [--segments S]
        [--rounds N] [--pings N] [--loads idle,load] [--verifier-cpu C] [--helper-cpu C] [--responder-cpu C] [--limit-ns L]

- **python**: today's loop, `memory_challenger.py`'s timed audit in a Python process (`loops.py python`).
- **lean-drive**: a Lean driver (`timing-loop drive`) asks a Python worker (`loops.py helper`) over stdin and stdout for each
  challenge, the reveal and the verdict, while Python keeps the socket: the Lean-to-Python hop on every round.
- **lean-own**: Lean owns the socket and the clock (`timing-loop own`, `Std.Internal.UV.TCP`), sends indices drawn here as
  `audit.MemoryChallenger` draws them, and checks each answer byte for byte against the store. `lean-own-2cpu` is the same loop
  with the memory challenger CPU's sibling allowed too, so Lean's event-loop thread need not share the loop's CPU; `lean-own-digest`
  checks each answer as `VerifierKey.check` does, its leaf digest in Lean (`Flock.Sha256`), on one CPU.

The responder is `pous_native.cu`'s over a store `C` encoded on the GPU from a random `W` (`--segments` segments of the band),
or with `--cpu-responder` the harness's `ControlResponder` over random bytes. `idle` runs each loop alone; `load` runs the
harness's load beside it (`harness.py`'s POUS forward: decode all of `C`, the weight set's GEMMs at `T = --load-tokens`, zero,
synchronized), as the honest audit runs. Every loop is pinned (`--verifier-cpu`; the worker on `--helper-cpu`; the
responder's thread on `--responder-cpu`).

A round's latency is the memory challenger's own reading, from just before the send to the whole answer (python: `challenge()` to
`respond()`, as `MemoryChallenger` judges it); its overhead is the latency less the responder's reported server time; its gap is
the time from that receipt to the next round's send within one audit (the check and the next draw, on a sequential reveal's
critical path). `out/summary.json` holds each configuration's quantiles, the fractions over 0.5 and 0.85 ms and the machine;
`out/rounds/*.npz` every round.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "band_gpu"))

import live  # noqa: E402

SCHEME = "band-chain/d12/v1"
LOOPS = ("python", "lean-drive", "lean-own", "lean-own-2cpu", "lean-own-digest")
THRESHOLDS_NS = (500_000, 850_000)
WEIGHT_SETS = HERE.parent / "weight_sets.json"


def quantiles(xs: np.ndarray) -> dict[str, float]:
    if not len(xs):
        return {}
    s = np.sort(xs)
    at = lambda q: int(s[min(len(s) - 1, int(q * len(s)))])  # noqa: E731
    return {"n": int(len(s)), "min": int(s[0]), "p50": at(0.5), "p99": at(0.99), "p99.9": at(0.999), "p99.99": at(0.9999),
            "max": int(s[-1]), "mean": float(s.mean())}


def over(xs: np.ndarray) -> dict[str, float]:
    return {f"over_{t}_ns": float((xs > t).mean()) for t in THRESHOLDS_NS} | {f"count_over_{t}_ns": int((xs > t).sum())
                                                                             for t in THRESHOLDS_NS}


def cpu_times(cpus: list[int]) -> dict[int, list[int]]:
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        name, *vals = line.split()
        if name.startswith("cpu") and name[3:].isdigit() and int(name[3:]) in cpus:
            out[int(name[3:])] = [int(v) for v in vals]
    return out


def busy_fraction(before: dict[int, list[int]], after: dict[int, list[int]]) -> dict[str, float]:
    out = {}
    for c in before:
        d = [b - a for a, b in zip(before[c], after[c])]
        total = sum(d)
        out[str(c)] = 1 - (d[3] + d[4]) / total if total else 0.0
    return out


def cmd_out(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def machine(a: argparse.Namespace) -> dict[str, Any]:
    lscpu = cmd_out(["lscpu"]) or ""
    model = next((ln.split(":", 1)[1].strip() for ln in lscpu.splitlines() if ln.startswith("Model name")), None)
    siblings = {c: Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read_text().strip()
                for c in (a.verifier_cpu, a.helper_cpu, a.responder_cpu) if c is not None
                and Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").exists()}
    return {"hostname": platform.node(), "cpu_model": model, "kernel": platform.release(), "python": sys.version.split()[0],
            "lean_toolchain": (HERE / "lean" / "lean-toolchain").read_text().strip(),
            "lean_version": cmd_out(["lean", "--version"]), "affinity": sorted(os.sched_getaffinity(0)),
            "cpus": {"verifier": a.verifier_cpu, "helper": a.helper_cpu, "responder": a.responder_cpu},
            "thread_siblings": siblings, "clocksource": cmd_out(["cat", "/sys/devices/system/clocksource/clocksource0/current_clocksource"]),
            "cgroup": cmd_out(["cat", "/proc/self/cgroup"])}


def store(a: argparse.Namespace, scheme: Any) -> tuple[bytes, Any]:
    """`C` on the host, and what the responder and the load need: the GPU backend, `C` and its tags on the device."""
    p = scheme.params
    if a.cpu_responder:
        return os.urandom(p.blocks * p.block_bytes), None
    import torch

    from benchmarks.pous.band_gpu import harness

    backend = harness.gpu(scheme, a.build_dir, harness.KERNELS)
    g = torch.Generator(device=backend.device).manual_seed(20261005)
    w = torch.randint(0, 256, (p.segments, p.blocks_per_segment, p.block_bytes), dtype=torch.uint8, device=backend.device,
                      generator=g)
    tags = backend.tags(harness.SALT, list(range(p.segments)), backend.device)
    c = backend.encode(w, tags)
    torch.cuda.synchronize()
    return c.contiguous().cpu().numpy().tobytes(), {"backend": backend, "c": c, "tags": tags, "w": w}


def load_fn(a: argparse.Namespace, gpu: dict[str, Any] | None, used: int) -> Any:
    if gpu is None or "load" not in a.loads:
        return None
    import torch

    from benchmarks.pous.band_gpu import harness

    backend, c, tags = gpu["backend"], gpu["c"], gpu["tags"]
    ws = next(w for w in json.loads(WEIGHT_SETS.read_text())["weight_sets"] if w["id"] == a.weight_set)
    pl = harness.placements(ws["tensors"], used)
    xs = harness.inputs(pl, {a.load_tokens}, backend.device)

    def forward() -> None:
        flat = backend.decode(c, tags).reshape(-1)
        harness.forward(flat, pl, xs[a.load_tokens])
        flat.zero_()
        torch.cuda.synchronize()

    forward()
    return forward


def responder(a: argparse.Namespace, gpu: dict[str, Any] | None, c_host: bytes, width: int) -> Any:
    if gpu is None:
        from benchmarks.pous.band_gpu import harness

        return harness.ControlResponder(width, lambda i: (c_host[i * width:(i + 1) * width], 0, 1_000, 0))
    return gpu["backend"].responder(gpu["c"], a.responder_cpu)


def loop_cmd(loop: str, a: argparse.Namespace, files: dict[str, Path], audits: int, k: int, raw: Path) -> list[str]:
    py = [sys.executable, str(HERE / "loops.py")]
    common = ["--ready", str(files["ready"]), "--leaves", str(files["leaves"]), "--k", str(k), "--limit-ns", str(a.limit_ns)]
    pin = lambda cpus: ["taskset", "-c", cpus] if cpus else []  # noqa: E731
    vcpu = "" if a.verifier_cpu is None else str(a.verifier_cpu)
    if loop == "python":
        return py + ["python", *common, "--audits", str(audits), "--pings", str(a.pings), "--out", str(raw)] + (
            ["--verifier-cpu", vcpu] if vcpu else [])
    if loop == "lean-drive":
        return pin(vcpu) + [a.lean_exe, "drive", "--k", str(k), "--audits", str(audits), "--pings", str(a.pings),
                            "--limit-ns", str(a.limit_ns), "--out", str(raw), "--"] + py + ["helper", *common] + (
            ["--cpu", str(a.helper_cpu)] if a.helper_cpu is not None else [])
    sib = sibling_list(a.verifier_cpu) if loop == "lean-own-2cpu" and vcpu else vcpu
    digest = ["--check", "digest", "--leaves", str(files["leaves"]), "--prefixes", str(files["prefixes"])]
    return pin(sib) + [a.lean_exe, "own", "--port", "{port}", "--challenges", str(files["challenges"]), "--store",
                       str(files["store"]), "--block", str(files["block"]), "--k", str(k), "--pings", str(a.pings),
                       "--limit-ns", str(a.limit_ns), "--out", str(raw)] + (digest if loop == "lean-own-digest" else [])


def sibling_list(cpu: int) -> str:
    p = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    return p.read_text().strip() if p.exists() else str(cpu)


def cpu_list(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def lean_rows(raw: Path, width: int) -> tuple[np.ndarray, np.ndarray]:
    b = np.frombuffer(raw.read_bytes(), dtype="<u8").astype(np.int64)
    pings = int(b[0])
    ping_w = 5
    ps = b[1:1 + ping_w * pings].reshape(-1, ping_w)
    n = int(b[1 + ping_w * pings])
    rs = b[2 + ping_w * pings:].reshape(n, width)
    return ps, rs


def rounds_of(loop: str, raw: Path, k: int) -> dict[str, np.ndarray]:
    """Every loop's rounds in one shape: latency, server and device ns, overhead, the gap to the next send (within an audit,
    else -1), correctness, and the ping round trips."""
    if loop == "python":
        z = np.load(raw)
        latency = z["received"] - z["start"]
        if not np.array_equal(latency, z["elapsed_ns"]):
            raise RuntimeError("python: the recorded clock disagrees with the memory challenger's elapsed_ns")
        nxt_send, this_recv = z["sent"][1:], z["answered"][:-1]
        out = {"index": z["index"], "latency_ns": latency, "server_ns": z["server_ns"], "device_ns": z["device_ns"],
               "wire_ns": z["answered"] - z["sent"], "correct": z["correct"], "check_draw_ns":
               np.append(z["start"][1:] - z["received"][:-1], -1), "gap_raw": np.append(nxt_send - this_recv, -1),
               "ping_ns": z["ping_answered"] - z["ping_sent"]}
    elif loop == "lean-drive":
        ps, rs = lean_rows(raw, 11)
        index, d0, d1, t0, t1, v1, server, device, wrtt, correct, _ = rs.T
        out = {"index": index, "latency_ns": t1 - t0, "server_ns": server, "device_ns": device, "wire_ns": wrtt,
               "correct": correct, "draw_hop_ns": d1 - d0, "verdict_hop_ns": v1 - t1,
               "gap_raw": np.append(t0[1:] - t1[:-1], -1), "ping_ns": ps[:, 1] - ps[:, 0], "ping_wire_ns": ps[:, 4]}
    else:
        ps, rs = lean_rows(raw, 7)
        index, t0, t1, server, device, ok, recvs = rs.T
        out = {"index": index, "latency_ns": t1 - t0, "server_ns": server, "device_ns": device, "correct": ok,
               "recvs": recvs, "gap_raw": np.append(t0[1:] - t1[:-1], -1), "ping_ns": ps[:, 1] - ps[:, 0]}
    n = len(out["latency_ns"])
    last = (np.arange(n) % k) == k - 1
    out["gap_ns"] = np.where(last, -1, out.pop("gap_raw"))
    out["overhead_ns"] = out["latency_ns"] - out["server_ns"]
    return out


def summarize(r: dict[str, np.ndarray]) -> dict[str, Any]:
    gap = r["gap_ns"][r["gap_ns"] >= 0]
    out = {"rounds": int(len(r["latency_ns"])), "wrong": int((r["correct"] == 0).sum()),
           "latency_ns": quantiles(r["latency_ns"]), "latency_over": over(r["latency_ns"]),
           "overhead_ns": quantiles(r["overhead_ns"]), "server_ns": quantiles(r["server_ns"]),
           "device_ns": quantiles(r["device_ns"]), "gap_ns": quantiles(gap),
           "round_plus_gap_ns": quantiles((r["latency_ns"] + np.maximum(r["gap_ns"], 0))[r["gap_ns"] >= 0]),
           "ping_ns": quantiles(r["ping_ns"])}
    for f in ("wire_ns", "check_draw_ns", "draw_hop_ns", "verdict_hop_ns", "ping_wire_ns", "recvs"):
        if f in r:
            x = r[f][r[f] >= 0] if f == "check_draw_ns" else r[f]
            out[f] = quantiles(x)
    if "wire_ns" in r:
        out["latency_less_wire_ns"] = quantiles(r["latency_ns"] - r["wire_ns"])
    return out


def run_one(loop: str, cmd: list[str], a: argparse.Namespace, load: Any, start_responder: Any, out: Path,
            tag: str) -> dict[str, Any]:
    resp = start_responder()
    cmd = [str(resp.port()) if x == "{port}" else x for x in cmd]
    watch = [c for c in (a.verifier_cpu, a.helper_cpu, a.responder_cpu) if c is not None]
    before, t_start = cpu_times(watch), time.monotonic()
    forwards = 0
    with (out / "logs" / f"{tag}.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env())
        if load is None:
            stdout, _ = proc.communicate()
        else:
            while proc.poll() is None:
                load()
                forwards += 1
            stdout = proc.stdout.read()
    seconds = time.monotonic() - t_start
    after = cpu_times(watch)
    counts = {"answered": int(resp.answered()), "pings": int(resp.pings())}
    resp.stop()
    line = next((ln for ln in stdout.splitlines() if ln.startswith("RESULT ")), None)
    return {"rc": proc.returncode, "cmd": cmd, "seconds": seconds, "forwards_completed": forwards, "responder": counts,
            "verdicts": json.loads(line.split(" ", 1)[1]) if line else None, "cpu_busy": busy_fraction(before, after)}


def env() -> dict[str, str]:
    paths = [str(ROOT), str(ROOT / "verity" / "protocols" / "accounting" / "space" / "pous"), os.environ.get("PYTHONPATH", "")]
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in paths if p)}


def main(argv: list[str] | None = None) -> int:
    from benchmarks.pous.band_gpu.harness import SALT
    from verity.protocols.accounting.space.pous import audit, protocol, schemes
    from verity.protocols.accounting.space.pous.labels import salt_value, tag
    from verity.protocols.accounting.space.pous.verifier import VerifierKey, leaf_hash

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--lean-exe", required=True)
    ap.add_argument("--cpu-responder", action="store_true")
    ap.add_argument("--segments", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=100_000)
    ap.add_argument("--pings", type=int, default=20_000)
    ap.add_argument("--loads", default="idle,load")
    ap.add_argument("--loops", default=",".join(LOOPS))
    ap.add_argument("--load-tokens", type=int, default=1)
    ap.add_argument("--weight-set", default="qwen2.5-0.5b/linears+head")
    ap.add_argument("--verifier-cpu", type=int, default=None)
    ap.add_argument("--helper-cpu", type=int, default=None)
    ap.add_argument("--responder-cpu", type=int, default=None)
    ap.add_argument("--limit-ns", type=int, default=audit.DEADLINE_NS + 350_000)
    ap.add_argument("--build-dir", default=None)
    a = ap.parse_args(argv)
    out, scratch = Path(a.out), Path(a.scratch)
    for d in (out / "rounds", out / "logs", scratch):
        d.mkdir(parents=True, exist_ok=True)
    a.build_dir = a.build_dir or str(scratch / "native")
    loads = [x for x in a.loads.split(",") if x]
    if a.cpu_responder and "load" in loads:
        raise SystemExit("--cpu-responder runs idle only: the harness's load is the GPU's")

    here = set(os.sched_getaffinity(0))
    measured = {c for c in (a.verifier_cpu, a.helper_cpu, a.responder_cpu) if c is not None}
    reserved = {int(x) for c in measured for x in cpu_list(sibling_list(c))} & here
    if reserved and here - reserved:
        os.sched_setaffinity(0, here - reserved)
    scheme = live.scheme_of(SCHEME, schemes.get(SCHEME).params.with_segments(a.segments))
    p = scheme.params
    width = p.block_bytes
    t = time.monotonic()
    c_host, gpu = store(a, scheme)
    leaves = [leaf_hash(p, SALT, i, c_host[i * width:(i + 1) * width]) for i in range(p.blocks)]
    vk = VerifierKey.from_leaves(p, SALT, leaves)
    k = scheme.certificate.k if scheme.certificate is not None else audit.CHALLENGES
    audits = math.ceil(a.rounds / k)
    files = {"ready": scratch / "ready.json", "leaves": scratch / "leaves.bin", "store": scratch / "store.bin",
             "challenges": scratch / "challenges.bin", "prefixes": scratch / "prefixes.bin", "block": width}
    files["leaves"].write_bytes(b"".join(leaves))
    files["store"].write_bytes(c_host)
    files["prefixes"].write_bytes(b"".join(b"\x00" + tag(salt_value(SALT), s).to_bytes(32, "little") + v.to_bytes(2, "little")
                                           for s in range(p.segments) for v in range(p.blocks_per_segment)))
    draws = []
    for _ in range(audits):
        key = protocol.challenge_key(os.urandom(32), vk)
        draws += [key.uniform(p.blocks, j) for j in range(k)]
    files["challenges"].write_bytes(np.array(draws, dtype="<u8").tobytes())
    setup_s = time.monotonic() - t
    load = load_fn(a, gpu, p.blocks * width)
    facts: dict[str, Any] = {"machine": machine(a)}
    if gpu is not None:
        facts["gpu_before"] = gpu["backend"].facts()
    summary: dict[str, Any] = {"schema": "verity-pous/timing-loop/v1", "scheme": scheme.name, "params": p.manifest(), "k": k,
                               "audits": audits, "limit_ns": a.limit_ns, "deadline_ns": audit.DEADLINE_NS,
                               "responder": "ControlResponder (CPU)" if gpu is None else "pous_native.cu (GPU)",
                               "load": {"weight_set": a.weight_set, "tokens": a.load_tokens}, "setup_s": setup_s,
                               "configs": {}}
    for ld in loads:
        for loop in [x for x in a.loops.split(",") if x]:
            tag = f"{loop}-{ld}"
            raw = out / "rounds" / (f"{tag}.npz" if loop == "python" else f"{tag}.bin")
            cmd = loop_cmd(loop, a, files, audits, k, raw)

            def start() -> Any:
                r = responder(a, gpu, c_host, width)
                files["ready"].write_text(json.dumps({"port": int(r.port()), "salt": SALT.hex(), "scheme": scheme.name,
                                                      "params": p.manifest()}))
                return r

            run = run_one(loop, cmd, a, load if ld == "load" else None, start, out, tag)
            rec = {**run}
            if run["rc"] == 0 and raw.exists():
                r = rounds_of(loop, raw, k)
                rec.update(summarize(r))
                if loop != "python":
                    np.savez_compressed(out / "rounds" / f"{tag}.npz", **r)
                    raw.unlink()
            summary["configs"][tag] = rec
            print(f"{tag}: rc={run['rc']} {json.dumps({q: rec.get('latency_ns', {}).get(q) for q in ('p50', 'p99', 'p99.9', 'max')})}",
                  flush=True)
    if gpu is not None:
        facts["gpu_after"] = gpu["backend"].facts()
    summary["facts"] = facts
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "table.md").write_text(table(summary))
    return 0 if all(c["rc"] == 0 and c.get("wrong") == 0 for c in summary["configs"].values()) else 1


def table(s: dict[str, Any]) -> str:
    us = lambda ns: "" if ns is None else f"{ns / 1e3:,.1f}"  # noqa: E731
    head = ("| configuration | n | latency p50 / p99 / p99.9 / max (µs) | overhead p50 / p99 / p99.9 / max (µs) | "
            "gap p50 / p99 / p99.9 / max (µs) | > 0.5 ms | > 0.85 ms |\n|---|---|---|---|---|---|---|\n")
    rows = []
    for tag, c in s["configs"].items():
        if "latency_ns" not in c:
            rows.append(f"| {tag} | failed (rc {c['rc']}) | | | | | |")
            continue
        q = lambda d: " / ".join(us(d.get(x)) for x in ("p50", "p99", "p99.9", "max"))  # noqa: E731
        o = c["latency_over"]
        rows.append(f"| {tag} | {c['rounds']:,} | {q(c['latency_ns'])} | {q(c['overhead_ns'])} | {q(c['gap_ns'])} | "
                    f"{o['over_500000_ns']:.2e} ({o['count_over_500000_ns']}) | {o['over_850000_ns']:.2e} "
                    f"({o['count_over_850000_ns']}) |")
    return head + "\n".join(rows) + "\n"


if __name__ == "__main__":
    sys.exit(main())
