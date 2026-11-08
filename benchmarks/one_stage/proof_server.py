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

from zk_cost import Busy, Peak, environ, free_port, jsonl  # noqa: E402

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
                            "k_log": rec["k_log"], "definition": rec["definition"]}
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


def streams(types: dict[str, dict], weights: dict[str, float], K: int, n: int, warm: int, seed: int):
    """Each prover's sessions (type, drawn units) and each type's draws, in the order its verifier serves them."""
    import numpy as np
    rng = np.random.default_rng(seed)
    order = sorted(types, key=lambda t: -weights[t])
    per, draws = [], {t: [] for t in types}
    for p in range(K):
        warmup = (order * (warm // len(order) + 1))[:max(warm, len(order))]
        seq = [(t, None) for t in warmup]
        for t in swrr(weights, n, phase=p * max(1, n // K)):
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
    per, draws = streams(types, weights, K, a.sessions, a.warm, a.seed)
    warm = len(per[0]) - a.sessions
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
    cmd = [a.bin, "prove", *zk, *([] if a.cpu else ["--gpu"]), "--warm", str(warm)]
    lives: list[dict] = []
    lock = threading.Lock()
    provers = []
    t_start = time.time()
    for i in range(K):
        p = subprocess.Popen([*cmd, "--stream", str(out / f"stream-{i}.jsonl")], stdout=subprocess.PIPE,
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
    rec = summarize(lives, samples, types, K, warm, a.sessions)
    rec |= {"provers": K, "verify_ahead": ahead, "sessions": a.sessions, "warm": warm, "cpu": a.cpu, "plain": a.plain,
            "timed_out": timed_out, "rc": [p.returncode for p in provers], "wall_s": round(time.time() - t_start, 3),
            "types": {t: {k: v[k] for k in ("c", "full", "k_log", "rb")} for t, v in types.items()},
            "host_peak_gib": [round(x / 2**20, 2) for x in host_kb], "verifiers_peak_gib": round(serve_kb / 2**20, 2),
            "gpu_peak_mib": {str(p.pid): gpu_peak.get(p.pid) for p in provers}, "gpu_peak_all_mib": gpu_peak,
            "gpu_mem_max_mib": max((m for *_, m in samples), default=None), "cgroup": cgroup(),
            "verify_s": by_type([(v["type"], v["verify_s"]) for v in verdicts if v["verify_s"] is not None]),
            "verifier_cache": {t: next((v["stmt_cache"] for v in reversed(verdicts) if v["type"] == t), None) for t in types},
            "accepted": all(x["accepted"] is True for x in lives) and len(lives) == K * len(per[0])
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


def summarize(lives: list[dict], samples, types: dict, K: int, warm: int, n: int) -> dict:
    """The steady window's numbers: from the last prover's last warm verdict to the first prover's last."""
    ends = {i: sorted(x["t_end"] for x in lives if x["p"] == i) for i in range(K)}
    if not all(len(e) > warm + 1 for e in ends.values()):
        return {}
    lo, hi = max(e[warm - 1] for e in ends.values()), min(e[-1] for e in ends.values())
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
    blocks = {t: round(sum(x["instances"] or 0 for x in inside if x["label"] == t) / span, 2) for t in types}
    busy = [u for t, u, _ in samples if lo <= t <= hi]
    # each prover's statement-cache hit rate over consecutive windows of its timed sessions
    windows = []
    for i in range(K):
        mine = [x["stmt_cache"] for x in sorted(lives, key=lambda x: x["t_end"]) if x["p"] == i and not x["warm"]
                and x.get("stmt_cache")]
        step = max(1, len(mine) // 4)
        pts = mine[step - 1::step]
        prev = next((x["stmt_cache"] for x in sorted(lives, key=lambda x: x["t_end"]) if x["p"] == i and x["warm"]
                     and x.get("stmt_cache")), {"hits": 0, "misses": 0})
        rates = []
        for c in pts:
            dh, dm = c["hits"] - prev["hits"], c["misses"] - prev["misses"]
            rates.append(round(dh / (dh + dm), 4) if dh + dm else None)
            prev = c
        windows.append(rates)
    waits = [x["verdict_wait_s"] for x in inside if x.get("verdict_wait_s") is not None]
    return {"steady_s": round(span, 3), "statements_s": round(len(inside) / span, 4), "statements_s_by_type": rate,
            "blocks_s_by_type": blocks, "busy": round(sum(busy) / len(busy) / 100, 4) if busy else None,
            "gpu_s_by_type": by_type(gpu_s), "hit_windows": windows,
            "flat": all(len(w) >= 2 and None not in w[-2:] and abs(w[-1] - w[-2]) < 0.01 for w in windows),
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

    for d in a.runs:
        rec = json.loads((Path(d) / "server.json").read_text())
        r = {"dir": d, "provers": rec["provers"], "statements_s": rec.get("statements_s"), "busy": rec.get("busy"),
             "beta": {w: S.solve(build(rec, 1.0), t, mixrec["scope"], case.scale) for w, t in WEIGHTS.items()},
             "beta_prover4": {w: S.solve(build(rec, 0.25), t, mixrec["scope"], case.scale) for w, t in WEIGHTS.items()}}
        out["runs"].append(r)
        print(f"REPORT K={r['provers']}: beta {r['beta']} (the line's {out['line_beta']}), with a 4x prover "
              f"{r['beta_prover4']}", flush=True)
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deadline", type=float, default=3 * 3600, help="seconds before the provers are stopped")
    ap.add_argument("--bin")
    ap.add_argument("--cpu", action="store_true", help="the CPU prover (no --gpu, no device sampling)")
    ap.add_argument("--plain", action="store_true", help="Flock's plain path, a comparison (no --zk, FC_COINS=tree)")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"mix": mix, "run": run, "report": report}[a.command](a)


if __name__ == "__main__":
    raise SystemExit(main())
