"""A long-running proof server on one GPU, as production would run one: K `flock-circuit prove --zk --gpu --stream`
processes on the device, each proving a stream of sampled statements at the planner's mix of proof-block types against one
long-lived `serve --zk` per type, the statement cache (`FC_STMT_CACHE=1`) and verify-ahead (`FC_VERIFY_AHEAD`) on, until
each prover's statement-cache hit rate is flat.

    python benchmarks/one_stage/proof_server.py mix    --sweep DIR --case R6x4 --scope "world year" --beta B --out MIX.json
    python benchmarks/one_stage/proof_server.py run    --mix MIX.json --staged Z --provers K --bin BIN --out DIR [--cpu]
    python benchmarks/one_stage/proof_server.py report --mix MIX.json --sweep DIR --runs DIR... --out REPORT.json

* **Mix** (`mix`, where BREAKEVEN's sweep runs): the planner's design at a case, scope and beta
  (`optimize.two_stage_design` through the sweep in `--sweep`), and each cell's statements a sampled replay block
  (`optimize.two_stage_statements`): the blocks it takes, its statement cap, the seconds the line prices it at, its share
  of proving and, for each cell `benchmarks/served_units/stage.py` stages (`TYPES`), its type.
* **Statements** (`--staged`, `stationary_gpu.sh`'s layout: `Z/<type>/c<n>/stage`): each type staged at a quarter, half and
  all of a 2^35-bit statement on one built circuit; K provers take the statements of full/K blocks, the most K fit on
  one device.
* **Streams**: per type, its statements a sampled replay block at the run's size s, taken/s, in the planner's ratio
  (smooth weighted round robin, each prover from its own phase), each statement whole (every staged unit drawn) except
  one in taken/s, the type's remainder in that replay block, of a uniform size below s. Each prover's stream starts with
  `--warm` sessions, one of each type at least; its sessions run back to back, a saturated server.
* **Verifiers**: one `serve --zk` per type for the whole run, `--concurrency K (1 + FC_VERIFY_AHEAD)`, serving the type's
  draws in turn (`--draw-file` with `draws`), its own statement cache on.
* **Measured**: each session's record with its type, the prover's statement-cache hits and misses, and the loop's wait for
  its verdict past verify-ahead (`verdict_wait_s`); the device's busy and memory (`nvidia-smi`, every 0.5 s) and each
  process's device memory; each prover's and the verifiers' summed resident memory (peaks sampled from /proc); the
  cgroup's memory peak and OOM kills. The steady window is from the last prover's last warm verdict to the first prover's
  last: there, statements a second by type, the device's busy share, and each type's GPU-seconds a statement (a
  session's span on its prover over K, the device shared by K at once).
* **Report** (`report`): per K the steady numbers, and the β they imply: the line's break-even β (`sweep.solve` at FP4's
  weights) with each measured type's proving replaced by its GPU-seconds a statement over s blocks (its statement s
  blocks, the verifier's fixed seconds kept as the line prices them), beside the line's own.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from zk_cost import Busy, Peak, environ, free_port, jsonl, read_jsonl  # noqa: E402

#: the cells `benchmarks/served_units/stage.py` stages, by type: a census type by its descriptor, a fused per-expert
#: block by its read's and GEMM's
FUSED = {"fr-w13": ("EmbeddingRead_v1{V=256,H=3072}", "MoeExpertFp8BlockGemm_v4"),
         "fr-w2": ("EmbeddingRead_v1{V=256,H=4096}", "MoeExpertFp8BlockGemmW_v4")}
#: K provers' statement size: the staged count of a type at full/K
SIZE = {1: "full", 2: "half", 4: "quarter"}


def type_of(block: str, census: dict[str, tuple[str, int]]) -> str | None:
    for t, (read, gemm) in FUSED.items():
        if block.startswith(f"Fused[{read} + {gemm}{{"):
            return t
    return next((t for t, (spec, _) in census.items() if spec == block), None)


# ---- mix ------------------------------------------------------------------------------------------------------------------

def line_at(sweep: Path, case: str, scope: str, beta: float):
    """The case's line through the sweep, and its design at ``beta``: (sweep, the line, the scope, the model, the cells,
    the row, the registration span)."""
    sys.path[:0] = [str(sweep)]
    import sweep as S
    from common import SCOPES, c, f1, tc
    from verity.core.protocols.sampled_proofs import optimize as opt
    L = S.build(S.cases()[case])
    _name, i, (tb, fl, oh), _comp, conf, replay_gpu_s, extra, *more = L
    s = (more[0] if more else {}).get("scopes", SCOPES)[scope]
    m = f1.model(i, tb, s, oh, fl)
    replay = (m.replay.cost * tc.batch(conf, s, c)) if replay_gpu_s is None else replay_gpu_s
    g = tc.Grain(m.replay.replay_blocks, m.cells, replay, m.served)
    span = g.replay_blocks * tc.registration_gpu_seconds(conf.registration, c) / s.gpu_seconds
    row = opt.two_stage_design(g.cells, g.replay_blocks, m.delta, budget=(beta - extra) * s.gpu_seconds,
                               replay=g.replay, h=m.byte_s, served=m.served, union=conf.form == "patterns",
                               registration=span)
    return S, L, s, i, list(g.cells), row, span, opt


def mix(a) -> int:
    from dataclasses import replace
    S, L, s, i, cells, row, span, opt = line_at(Path(a.sweep), a.case, a.scope, a.beta)
    sys.path.insert(0, str(HERE.parent / "served_units"))
    from stage import CENSUS
    strata = [x for x in i["partition"].value["strata"] if x["per_token"] > 0]
    assert len(strata) == len(cells), (len(strata), len(cells))
    split = [replace(x, stratum=0 if opt._size_class(x.bits) <= row["merged"] else opt._size_class(x.bits)) for x in cells]
    st = opt.two_stage_statements(split, row["k"], row["q"], span)
    total = sum(r["prove"] for r in st)
    out = [{"block": x["block"], "type": type_of(x["block"], CENSUS), "kind": x["kind"], "k_log": x["k_log"],
            "cap": c.cap, "taken": r["taken"], "shared": r["shared"], "fixed_s": c.fixed, "block_s": c.prove,
            "prove_s": r["prove"], "share": r["prove"] / total}
           for x, c, r in zip(strata, split, st) if r["taken"] > 0]
    out.sort(key=lambda r: -r["share"])
    from verity.core.protocols.pous.wipe_monitoring import PERIODS
    seconds = dict(PERIODS)[s.period]
    rec = {"case": a.case, "scope": a.scope, "beta": a.beta, "q": row["q"], "replay_sample": row["replay_sample"],
           "period_s": seconds, "replay_blocks_s": row["replay_sample"] / seconds, "gpus": s.gpu_seconds / seconds,
           "gpu_seconds": s.gpu_seconds, "prove_gpu_s": row["prove"], "total_gpu_s": row["total"],
           "covered": sum(r["share"] for r in out if r["type"]), "cells": out}
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(f"MIX {a.case} {a.scope} beta {a.beta}: {len(out)} cells, staged types cover {rec['covered']:.1%} of proving; "
          + ", ".join(f"{r['type']} {r['share']:.1%}" for r in out if r["type"]), flush=True)
    return 0


# ---- run ------------------------------------------------------------------------------------------------------------------

def staged(z: Path, K: int) -> dict[str, dict]:
    """Each staged type's statement for K provers: the count of full/K blocks on its built circuit (full: its largest)."""
    out = {}
    for td in sorted(p for p in z.iterdir() if p.is_dir()):
        sts = {json.loads(f.read_text())["n"]: f.parent for f in td.glob("c*/stage/stage.json")}
        if not sts:
            continue
        full = max(sts)
        n = max([c for c in sts if c <= full // K] or [min(sts)])
        rec = json.loads((sts[n] / "stage.json").read_text())
        out[rec["unit"]] = {"c": n, "full": full, "dir": str(sts[n]), "pin": (sts[n] / "pin.txt").read_text().strip(),
                            "k_log": rec["k_log"], "definition": rec["definition"],
                            "circuit_bytes": (sts[n] / "circuit.txt").stat().st_size}
    return out


def taken_of(spec: str) -> dict[str, float]:
    """Each staged type's blocks a sampled replay block: from a `mix` file, or inline as ``type=blocks,...``."""
    if not Path(spec).exists():
        return {t: float(x) for t, x in (e.split("=") for e in spec.split(","))}
    out: dict[str, float] = {}
    for r in json.loads(Path(spec).read_text())["cells"]:
        if r["type"]:
            out[r["type"]] = out.get(r["type"], 0.0) + r["taken"]
    return out


def swrr(weights: dict[str, float], n: int, phase: int) -> list[str]:
    """``n`` picks in ``weights``' ratio by smooth weighted round robin, from pick ``phase`` of the sequence."""
    cur, tot, out = {t: 0.0 for t in weights}, sum(weights.values()), []
    for j in range(phase + n):
        for t, w in weights.items():
            cur[t] += w
        t = max(cur, key=lambda x: (cur[x], x))
        cur[t] -= tot
        if j >= phase:
            out.append(t)
    return out


def route_of(types: dict[str, dict], K: int) -> dict[str, int]:
    """Each type's prover when types are routed: the largest circuits first, each to the prover holding the fewest bytes,
    so the K provers hold every circuit once between them."""
    held, out = [0] * K, {}
    for t in sorted(types, key=lambda t: -types[t]["circuit_bytes"]):
        out[t] = held.index(min(held))
        held[out[t]] += types[t]["circuit_bytes"]
    return out


def streams(types: dict[str, dict], weights: dict[str, float], K: int, n: int, warm: int, seed: int,
            route: dict[str, int] | None = None):
    """Each prover's sessions (type, drawn units) and each type's draws, in the order its verifier serves them; routed,
    a prover's sessions are its own types' in their mix ratio."""
    import numpy as np
    rng = np.random.default_rng(seed)
    order = sorted(types, key=lambda t: -weights[t])
    per, draws = [], {t: [] for t in types}
    for p in range(K):
        own = [t for t in order if route is None or route[t] == p]
        if not own:
            per.append([])
            continue
        warmup = (own * (warm // len(own) + 1))[:max(warm, len(own))]
        seq = [(t, None) for t in warmup]
        mine = {t: weights[t] for t in own}
        for t in swrr(mine, n, phase=0 if route else p * max(1, n // K)):
            c = types[t]["c"]
            remainder = c > 1 and rng.random() < 1.0 / max(1.0, types[t]["rb"])
            seq.append((t, int(rng.integers(1, c)) if remainder else None))
        per.append(seq)
    for seq in per:
        for t, r in seq:
            c = types[t]["c"]
            units = list(range(c)) if r is None else sorted(int(u) for u in rng.choice(c, size=r, replace=False))
            draws[t].append({"law": "subset", "population": c, "k": len(units), "units": units})
    return per, draws


class Gpu:
    """Each process's device memory (`nvidia-smi --query-compute-apps`) every ``ms`` milliseconds."""

    def __init__(self, ms: int = 1000) -> None:
        dev = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
        self.peak: dict[int, int] = {}
        self._p = subprocess.Popen(["nvidia-smi", "-i", dev, "--query-compute-apps=pid,used_memory",
                                    "--format=csv,noheader,nounits", "-lms", str(ms)], stdout=subprocess.PIPE, text=True)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        for line in self._p.stdout:
            try:
                pid, mib = (int(x) for x in line.split(","))
            except ValueError:
                continue
            self.peak[pid] = max(self.peak.get(pid, 0), mib)

    def stop(self) -> dict[int, int]:
        self._p.terminate()
        self._p.wait(timeout=30)
        return self.peak


class Guard:
    """Stops the run's process groups once the host's free memory falls under ``floor_gib``, so host RAM ends the run as
    a recorded limit and not as an OOM kill. Free is ``limit_gib`` less every visible process's resident set when given
    (a RunPod container shows no memory cgroup, and its /proc/meminfo is the host's); else the cgroup's room under
    memory.max; else MemAvailable."""

    def __init__(self, pids: list[int], floor_gib: float, limit_gib: float = 0.0, every: float = 1.0) -> None:
        self.pids, self.floor, self.hit, self.low_gib = pids, floor_gib * 2**30, None, None
        self.limit, self.used_peak_gib = limit_gib * 2**30, 0.0
        self._stop = threading.Event()
        self._every = every
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def free(self) -> float | None:
        if self.limit:
            used = 0
            for d in Path("/proc").iterdir():
                try:
                    used += int((d / "statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") if d.name.isdigit() else 0
                except (OSError, IndexError, ValueError):
                    pass
            self.used_peak_gib = max(self.used_peak_gib, used / 2**30)
            return self.limit - used
        try:
            rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
            d = Path("/sys/fs/cgroup") / rel.lstrip("/")
            cap = (d / "memory.max").read_text().strip()
            if cap != "max":
                return int(cap) - int((d / "memory.current").read_text())
        except (OSError, IndexError, ValueError):
            pass
        try:
            return int(next(x for x in Path("/proc/meminfo").read_text().splitlines()
                            if x.startswith("MemAvailable")).split()[1]) * 1024
        except (OSError, StopIteration):
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            free = self.free()
            if free is not None:
                self.low_gib = min(self.low_gib if self.low_gib is not None else 1e18, free / 2**30)
                if free < self.floor and self.hit is None:
                    self.hit = time.time()
                    for pid in self.pids:
                        try:
                            os.killpg(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
            self._stop.wait(self._every)

    def stop(self) -> None:
        self._stop.set()
        self._t.join()


def cgroup() -> dict:
    """The job's cgroup v2 memory peak, OOM kills and CPU seconds, where readable."""
    out = {}
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        d = Path("/sys/fs/cgroup") / rel.lstrip("/")
        for name in ("memory.peak", "memory.max"):
            f = d / name
            if f.exists():
                out[name] = f.read_text().strip()
        ev = d / "memory.events"
        if ev.exists():
            out["oom_kill"] = int(dict(x.split() for x in ev.read_text().splitlines()).get("oom_kill", 0))
        cs = d / "cpu.stat"
        if cs.exists():
            out["cpu_usage_s"] = int(dict(x.split() for x in cs.read_text().splitlines())["usage_usec"]) / 1e6
    except (OSError, IndexError, KeyError, ValueError):
        pass
    return out


def run(a) -> int:
    K, out = a.provers, Path(a.out)
    scratch = Path(a.scratch) if a.scratch else out
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    taken = taken_of(a.mix)
    types = {t: v for t, v in staged(Path(a.staged), K).items() if t in taken and (not a.types or t in a.types.split(","))}
    if not types:
        raise SystemExit(f"proof_server: no staged type of the mix under {a.staged}")
    for t, v in types.items():
        v["rb"] = taken[t] / v["c"]
    weights = {t: v["rb"] for t, v in types.items()}
    ahead = int(os.environ.get("FC_VERIFY_AHEAD") or 0)
    route = route_of(types, K) if a.route and K > 1 else None
    per, draws = streams(types, weights, K, a.sessions, a.warm, a.seed, route)
    per = [seq for seq in per if seq]
    K = len(per)
    warms = [len(seq) - a.sessions for seq in per]
    zk = [] if a.plain else ["--zk"]
    env = {"FC_STMT_CACHE": "1", "FC_HOST_WITNESS": None, "FC_FIREWALL": "0", **({"FC_COINS": "tree"} if a.plain else {})}
    (out / "types.json").write_text(json.dumps(types, indent=1))
    serves, ports, logs = {}, {}, {}
    peak_v = Peak()
    for t, v in types.items():
        (scratch / f"draws-{t}.json").write_text(json.dumps({"draws": draws[t]}))
        ports[t] = free_port()
        logs[t] = open(out / f"serve-{t}.log", "w")
        d = Path(v["dir"])
        serves[t] = subprocess.Popen(
            [a.bin, "serve", *zk, "--listen", f"127.0.0.1:{ports[t]}", "--out", str(scratch / "sessions" / t),
             "--instances", str(d / "pub.bin"), "--circuit", str(d / "circuit.txt"), "--pin", v["pin"],
             "--operator", "proof-server", "--draw-file", str(scratch / f"draws-{t}.json"),
             "--concurrency", str(K * (1 + ahead)), "--sessions", str(len(draws[t]))],
            stdout=logs[t], stderr=subprocess.STDOUT, text=True, env=environ(env), start_new_session=True)
        peak_v.roots.append(serves[t].pid)
    for i, seq in enumerate(per):
        with open(out / f"stream-{i}.jsonl", "w") as f:
            for t, _ in seq:
                d = Path(types[t]["dir"])
                f.write(json.dumps({"circuit": str(d / "circuit.txt"), "instances": str(d / "inst.bin"),
                                    "verifier": f"127.0.0.1:{ports[t]}", "label": t}) + "\n")
    deadline = time.monotonic() + a.deadline
    for t in types:
        while "SERVING" not in (out / f"serve-{t}.log").read_text():
            if serves[t].poll() is not None or time.monotonic() > deadline:
                raise SystemExit(f"proof_server: {t}'s verifier didn't start: {(out / f'serve-{t}.log').read_text()[-600:]}")
            time.sleep(0.2)
    busy, gpu = (None, None) if a.cpu else (Busy(), Gpu())
    peaks = [Peak() for _ in range(K)]
    cmd = [a.bin, "prove", *zk, *([] if a.cpu else ["--gpu"])]
    lives: list[dict] = []
    lock = threading.Lock()
    provers = []
    t_start = time.time()
    for i in range(K):
        p = subprocess.Popen([*cmd, "--warm", str(warms[i]), "--stream", str(out / f"stream-{i}.jsonl")], stdout=subprocess.PIPE,
                             stderr=open(out / f"prove{i}.err", "w"), text=True, env=environ({**env, "FC_VERIFY_AHEAD": str(ahead)}),
                             start_new_session=True)
        peaks[i].roots.append(p.pid)
        provers.append(p)

    def read(i: int, p: subprocess.Popen) -> None:
        with open(out / f"prove{i}.out", "w") as fo:
            for line in p.stdout:
                fo.write(line)
                if line.startswith("LIVE\t"):
                    v = json.loads(line[5:])
                    keep = {x: v.get(x) for x in ("run", "warm", "label", "accepted", "why", "e2e_s", "witness_s",
                                                  "prove_total_s", "wait_s", "verdict_wait_s", "instances", "k_log",
                                                  "stmt_cache", "verify_ahead")}
                    with lock:
                        lives.append({"p": i, "t_end": round(time.time(), 3), **keep})
                        jsonl(out / "live.jsonl", lives[-1])

    readers = [threading.Thread(target=read, args=(i, p), daemon=True) for i, p in enumerate(provers)]
    for r in readers:
        r.start()
    guard = Guard([p.pid for p in provers] + [p.pid for p in serves.values()], a.mem_floor, a.mem_limit)
    timed_out = False
    for p in provers:
        try:
            p.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
    for p in provers + list(serves.values()):
        if p.poll() is None:
            if not timed_out:
                try:
                    p.wait(timeout=120)
                    continue
                except subprocess.TimeoutExpired:
                    pass
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=60)
    for r in readers:
        r.join(timeout=60)
    guard.stop()
    for f in logs.values():
        f.close()
    samples = busy.stop() if busy else []
    gpu_peak = gpu.stop() if gpu else {}
    host_kb = [pk.stop() for pk in peaks]
    serve_kb = peak_v.stop()
    verdicts = []
    for t in types:
        for line in (out / f"serve-{t}.log").read_text().splitlines():
            if line.startswith("SESSION\t"):
                s = json.loads(line.split("\t", 1)[1])
                verdicts.append({"type": t, "verify_s": s.get("verify_s"), "accepted": s.get("accepted"),
                                 "stmt_cache": s.get("stmt_cache")})
    rec = summarize(lives, samples, types, K)
    rec |= {"provers": K, "verify_ahead": ahead, "sessions": a.sessions, "warm": warms,
            "route": route, "cpu": a.cpu, "plain": a.plain,
            "timed_out": timed_out, "rc": [p.returncode for p in provers], "wall_s": round(time.time() - t_start, 3),
            "host_limit": {"stopped_at_s": guard.hit and round(guard.hit - t_start, 1), "floor_gib": a.mem_floor,
                           "limit_gib": a.mem_limit or None, "min_free_gib": guard.low_gib and round(guard.low_gib, 1),
                           "used_peak_gib": round(guard.used_peak_gib, 1) if a.mem_limit else None},
            "types": {t: {k: v[k] for k in ("c", "full", "k_log", "rb")} for t, v in types.items()},
            "host_peak_gib": [round(x / 2**20, 2) for x in host_kb], "verifiers_peak_gib": round(serve_kb / 2**20, 2),
            "gpu_peak_mib": {str(p.pid): gpu_peak.get(p.pid) for p in provers}, "gpu_peak_all_mib": gpu_peak,
            "gpu_mem_max_mib": max((m for *_, m in samples), default=None), "cgroup": cgroup(),
            "verify_s": by_type([(v["type"], v["verify_s"]) for v in verdicts if v["verify_s"] is not None]),
            "verifier_cache": {t: next((v["stmt_cache"] for v in reversed(verdicts) if v["type"] == t), None) for t in types},
            "accepted": all(x["accepted"] is True for x in lives) and len(lives) == sum(map(len, per))
            and len(verdicts) == sum(len(x) for x in draws.values()) and all(v["accepted"] is True for v in verdicts)}
    with open(out / "busy.csv", "w") as f:
        f.writelines(f"{t:.3f},{u},{m}\n" for t, u, m in samples)
    (out / "server.json").write_text(json.dumps(rec, indent=1))
    print(f"SERVER K={K}: accepted {rec['accepted']}, steady {rec.get('steady_s')} s, {rec.get('statements_s')} statements/s, "
          f"busy {rec.get('busy')}, hit rate windows {rec.get('hit_windows')}, host peak {rec['host_peak_gib']} GiB, "
          f"verifiers {rec['verifiers_peak_gib']} GiB, device {rec['gpu_peak_mib']} MiB", flush=True)
    return 0 if rec["accepted"] else 1


def by_type(pairs) -> dict[str, dict]:
    out: dict[str, list[float]] = {}
    for t, x in pairs:
        out.setdefault(t, []).append(x)
    return {t: {"n": len(xs), "mean": round(sum(xs) / len(xs), 4), "max": round(max(xs), 4)} for t, xs in out.items()}


def summarize(lives: list[dict], samples, types: dict, K: int) -> dict:
    """The steady window's numbers: from the last prover's last warm verdict to the first prover's last."""
    ends = {i: sorted(x["t_end"] for x in lives if x["p"] == i) for i in range(K)}
    warm_end = {i: max([x["t_end"] for x in lives if x["p"] == i and x["warm"]] or [0.0]) for i in range(K)}
    lo, hi = max(warm_end.values()), min((e[-1] if e else 0.0) for e in ends.values())
    if hi <= lo or not all(sum(1 for t in e if t > lo) >= 2 for e in ends.values()):
        return {}
    inside = [x for x in lives if lo < x["t_end"] <= hi and not x["warm"]]
    span = hi - lo
    # each session's span on its prover: from its previous record to its own, the device shared by K at once
    gpu_s: list[tuple[str, float]] = []
    for i in range(K):
        mine = sorted((x for x in lives if x["p"] == i), key=lambda x: x["t_end"])
        for prev, x in zip(mine, mine[1:]):
            if lo < x["t_end"] <= hi and not x["warm"]:
                gpu_s.append((x["label"], (x["t_end"] - prev["t_end"]) / K))
    rate = {t: round(sum(1 for x in inside if x["label"] == t) / span, 4) for t in types}
    # the mix's throughput on one GPU: each measured type's GPU-seconds a statement, weighted by its statements a sampled
    # replay block (rb), so the window's own type ratio (routing, stream ends) doesn't enter
    g = by_type(gpu_s)
    tot = sum(types[t]["rb"] for t in g)
    per_stmt = sum(types[t]["rb"] / tot * g[t]["mean"] for t in g) if g else None
    blocks = {t: round(sum(x["instances"] or 0 for x in inside if x["label"] == t) / span, 2) for t in types}
    busy = [u for t, u, _ in samples if lo <= t <= hi]
    # each prover's statement-cache hit rate over consecutive windows of its timed sessions
    windows = []
    for i in range(K):
        mine = [x["stmt_cache"] for x in sorted(lives, key=lambda x: x["t_end"]) if x["p"] == i and not x["warm"]
                and x.get("stmt_cache")]
        step = max(1, len(mine) // 4)
        pts = mine[step - 1::step]
        prev = ([x["stmt_cache"] for x in sorted(lives, key=lambda x: x["t_end"]) if x["p"] == i and x["warm"]
                 and x.get("stmt_cache")] or [{"hits": 0, "misses": 0}])[-1]
        # a record's counters are read when it is emitted, after verify-ahead's next lookups: a window can be empty
        rates = []
        for c in pts:
            dh, dm = c["hits"] - prev["hits"], c["misses"] - prev["misses"]
            if dh + dm:
                rates.append(round(dh / (dh + dm), 4))
                prev = c
        windows.append(rates)
    waits = [x["verdict_wait_s"] for x in inside if x.get("verdict_wait_s") is not None]
    return {"steady_s": round(span, 3), "statements_s": round(len(inside) / span, 4), "statements_s_by_type": rate,
            "blocks_s_by_type": blocks, "busy": round(sum(busy) / len(busy) / 100, 4) if busy else None,
            "gpu_s_by_type": g, "hit_windows": windows,
            "statements_s_at_mix": round(1 / per_stmt, 4) if per_stmt else None,
            "statements_s_at_mix_by_type": {t: round(types[t]["rb"] / tot / per_stmt, 4) for t in g} if per_stmt else {},
            "flat": all(len(w) >= 2 and abs(w[-1] - w[-2]) < 0.01 for w in windows),
            "verdict_wait_s": round(sum(waits), 3), "verdict_wait_share": round(sum(waits) / (span * K), 4) if span else None,
            "e2e_s_by_type": by_type([(x["label"], x["e2e_s"]) for x in inside if x.get("e2e_s") is not None])}


# ---- report ---------------------------------------------------------------------------------------------------------------

def report(a) -> int:
    mixrec = json.loads(Path(a.mix).read_text())
    S, *_ = line_at(Path(a.sweep), mixrec["case"], mixrec["scope"], mixrec["beta"])
    from levers import WEIGHTS
    sys.path.insert(0, str(HERE.parent / "served_units"))
    from stage import CENSUS
    case = S.cases()[mixrec["case"]]
    if not case.verify:
        raise SystemExit(f"proof_server report: {mixrec['case']} prices no verifier (its reprice step is the one replaced)")
    out = {"case": mixrec["case"], "scope": mixrec["scope"], "line_beta": {}, "runs": []}
    for w, target in WEIGHTS.items():
        out["line_beta"][w] = S.solve(S.build(case), target, mixrec["scope"], case.scale)
    apply = S.reprice.apply

    def build(rec: dict, measured: float):
        """The case's line as the sweep builds it, each measured type's proving its GPU-seconds a statement (times
        ``measured``) over its statement's blocks, the verifier's fixed seconds kept."""
        gs = rec.get("gpu_s_by_type") or {}

        def measured_apply(strata, **kw):
            new = []
            for x, y in zip(apply(strata, **kw), apply(strata, **(kw | {"verify": 0.0}))):
                t = type_of(x["block"], CENSUS)
                if t in gs and t in rec["types"]:
                    c = rec["types"][t]["c"]
                    x = x | {"seconds_per_block": gs[t]["mean"] * measured / c, "statement_blocks": c,
                             "fixed_seconds": x["fixed_seconds"] - y["fixed_seconds"], "priced": "measured"}
                new.append(x)
            return new

        S.reprice.apply = measured_apply
        try:
            return S.build(case)
        finally:
            S.reprice.apply = apply

    recs = []
    for d in a.runs:
        rec = json.loads((Path(d) / "server.json").read_text()) | {"dir": d}
        if (Path(d) / "live.jsonl").exists():
            samples = [(float(t), None if u == "None" else float(u), m)
                       for t, u, m in (x.split(",") for x in (Path(d) / "busy.csv").read_text().split())]
            rec |= summarize(read_jsonl(Path(d) / "live.jsonl"), samples, rec["types"], rec["provers"]) or {}
        recs.append(rec)
    if a.combine:
        # one line from several runs' types (each type from the first run that measured it), e.g. the type groups
        # whose circuits don't fit one host together, each run at the same K
        m: dict = {"dir": ",".join(r["dir"] for r in recs), "provers": "+".join(str(r["provers"]) for r in recs),
                   "types": {}, "gpu_s_by_type": {}}
        for r in recs:
            for t, g in (r.get("gpu_s_by_type") or {}).items():
                if t not in m["gpu_s_by_type"]:
                    m["gpu_s_by_type"][t], m["types"][t] = g, r["types"][t]
        g, ty = m["gpu_s_by_type"], m["types"]
        tot = sum(ty[t]["rb"] for t in g)
        per_stmt = sum(ty[t]["rb"] / tot * g[t]["mean"] for t in g) if g else None
        m["statements_s_at_mix"] = round(1 / per_stmt, 4) if per_stmt else None
        m["statements_s_at_mix_by_type"] = {t: round(ty[t]["rb"] / tot / per_stmt, 4) for t in g} if per_stmt else {}
        recs.append(m)
    share = {c["type"]: c["share"] for c in mixrec["cells"] if c["type"]}
    for rec in recs:
        gs = rec.get("gpu_s_by_type") or {}
        r = {"dir": rec["dir"], "provers": rec["provers"], "statements_s": rec.get("statements_s"), "busy": rec.get("busy"),
             "statements_s_at_mix": rec.get("statements_s_at_mix"),
             "statements_s_at_mix_by_type": rec.get("statements_s_at_mix_by_type"), "gpu_s_by_type": gs,
             "statement_blocks": {t: rec["types"][t]["c"] for t in gs},
             "covered": round(sum(share.get(t, 0.0) for t in gs), 4),
             # the measured types' proving GPUs over all GPUs at the line's sampling rate: a ratio, the same for any
             # monitoring period and scope
             "prover_gpu_share": round(sum(mixrec["replay_blocks_s"] * rec["types"][t]["rb"] * gs[t]["mean"] for t in gs)
                                       / mixrec["gpus"], 6),
             "beta": {w: S.solve(build(rec, 1.0), t, mixrec["scope"], case.scale) for w, t in WEIGHTS.items()},
             "beta_prover4": {w: S.solve(build(rec, 0.25), t, mixrec["scope"], case.scale) for w, t in WEIGHTS.items()}}
        out["runs"].append(r)
        print(f"REPORT K={r['provers']} ({r['covered']:.1%} of proving measured): {r['statements_s_at_mix']} statements/s "
              f"at the mix, beta {r['beta']} (the line's {out['line_beta']}), with a 4x prover {r['beta_prover4']}",
              flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["mix", "run", "report"])
    ap.add_argument("--sweep", help="BREAKEVEN's sweep directory (benchmarks/spot_figures/breakeven)")
    ap.add_argument("--case", default="R6x4")
    ap.add_argument("--scope", default="world year")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--mix", help="a `mix` file, or (run) each type's blocks a sampled replay block, type=blocks,...")
    ap.add_argument("--scratch", help="(run) the verifiers' sessions and the draws, which hold proofs and unit lists")
    ap.add_argument("--staged", help="the staged statements, Z/<type>/c<n>/stage")
    ap.add_argument("--types", default="", help="only these staged types (comma-separated)")
    ap.add_argument("--provers", type=int, default=1)
    ap.add_argument("--sessions", type=int, default=120, help="each prover's timed sessions")
    ap.add_argument("--warm", type=int, default=0, help="each prover's warm sessions (at least one a type)")
    ap.add_argument("--route", action="store_true",
                    help="each type to one prover, the K provers holding every circuit once between them (K > 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deadline", type=float, default=3 * 3600, help="seconds before the provers are stopped")
    ap.add_argument("--mem-floor", type=float, default=8.0, help="GiB of free host memory under which the run is stopped")
    ap.add_argument("--mem-limit", type=float, default=0.0,
                    help="the host's memory in GiB where no cgroup shows its limit (a RunPod pod's memoryInGb)")
    ap.add_argument("--bin")
    ap.add_argument("--cpu", action="store_true", help="the CPU prover (no --gpu, no device sampling)")
    ap.add_argument("--plain", action="store_true", help="Flock's plain path, a comparison (no --zk, FC_COINS=tree)")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--combine", action="store_true", help="(report) also one line from all the runs' measured types")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"mix": mix, "run": run, "report": report}[a.command](a)


if __name__ == "__main__":
    raise SystemExit(main())
