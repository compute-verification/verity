"""C-Flock's zero-knowledge session cost on sm_120, for the sampling planner's cost model (`sampled_proofs.costfit`): A6's
W8A8 coordinates (`ScaledMmS8Coordinate_v1{K, DOT=BlackwellS8ImmaDot32_v1}`) staged in statements of c units, each
session proving k of them drawn by the verifier's Lean, with `flock-circuit prove --zk --gpu`, and the Lean verifier on the
sessions.

    python benchmarks/one_stage/zk_cost.py stage  --lean flock-verify --out DIR [--ks 4096,14336] [--sizes 1,8,64,512]
    python benchmarks/one_stage/zk_cost.py prove  --bin flock-circuit --out DIR [--arms fw,nofw] [--cold 2] [--runs 3] [--cpu]
    python benchmarks/one_stage/zk_cost.py verify --lean flock-verify --out DIR
    python benchmarks/one_stage/zk_cost.py fit    --out DIR [--run RUN_ID]

* **Statements.** For each K and c, a one-template program of c units (`mixed_population_program`), its
  `Q_template_instances` partition, and one statement staged by A6's `stage_template` bound to it (stand-in values; a
  Boolean circuit's cost doesn't depend on them). Its stage seconds exclude building the circuit, which a template pays
  once.
* **Draws.** k in {1, 8, 64, c} (`--draws`), at most c, each by Lean `flock-verify draw --stratified k` over the statement's
  program and partition, served from that file (`serve --zk --draw-file`). `--draws c` proves every unit: the cost of c
  drawn units in one statement.
* **Sessions.** Per point and arm, `--cold` processes proving one session each (`--runs 1 --warm 0`: one touched
  statement's prover today) and one proving `--warm` + `--runs` sessions (the in-process session cost). Arm `fw` is the
  design's path: the firewall's send check on every session (`FC_FIREWALL=1`), which rechecks the host's witness; `nofw`
  is the same with the check off. Both take the host's witness (`FC_HOST_WITNESS=1`), as proof_throughput's ZK point does,
  so their difference is the firewall's.
* **Verification.** Lean `verify --zk --partition --program` on each cold session of arm `fw` and the first timed session
  of each warm process, one at a time.
* **Fit.** Weighted least squares in relative error, with standard errors from the residuals: per K, the cold process
  `a + a_c c + a_k k` (GPU-s), the in-process session `g + g_c c + g_k k`, the process's setup (its wall less its session),
  staging `s + s_c c` and verification `v + v_c c + v_k k` (CPU-s); and across K, `costfit`'s features (one, ands,
  ands*staged, ands*drawn). `observations.json` holds `costfit.Observation`s: prove is a cold process's wall, verify the
  single-threaded verifier's CPU time (its wall stretches on a loaded node).
* **CPU** (`--cpu`, `zk_cost_cpu.sh`): the same sessions by `flock-circuit prove --zk` on the node's CPUs, as the CPU nodes
  prove a block's draw. Each session record carries `peak_rss_kb`, the sampled peak of the prover's, its live verifier's
  and the firewall's Lean's resident memory summed, which is what a node holds for one session.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from math import ceil, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a0  # noqa: E402
import a6  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402
from verity.core.protocols.sampled_proofs.optimize import SHA512_ANDS  # noqa: E402

#: ANDs of the pieces of a unit with its rows' openings (benchmarks/sampling_plan's inputs: its pins, hm96 PROTOCOL §5)
S8_STEP, EPILOGUE, SHA512, HM96 = 4_190, 5_000, SHA512_ANDS, 116_240


def unit_ands(K: int) -> int:
    """One coordinate's ANDs with its rows' openings, as `costfit` counts them: K/32 s8 steps, the epilogue, two K-byte
    operand rows and three one-block rows (the scales and the output)."""
    return K // 32 * S8_STEP + EPILOGUE + 2 * (HM96 + SHA512 * ceil((K + 17) / 128)) + 3 * (HM96 + SHA512)


def draws(c: int, spec: str = "1,8,64,c") -> list[int]:
    """The draw sizes of a statement of c units: ``spec``'s entries, each a count or ``c`` (every unit), those at most c."""
    return sorted({k for k in (c if x == "c" else int(x) for x in spec.split(",")) if k <= c})


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Peak:
    """The peak of the summed RSS of some processes and all their descendants, sampled every `every` seconds in a thread:
    a firewalled session's prover, its live verifier (`serve`) and the firewall's Lean child peak together, and their sum is
    what a node must hold for the session (ru_maxrss gives only the largest one process)."""

    def __init__(self, every: float = 0.5) -> None:
        import threading
        self.roots: list[int] = []
        self.peak_kb = 0
        self._stop = threading.Event()
        self._every = every
        self._page_kb = os.sysconf("SC_PAGE_SIZE") // 1024
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _tree(self) -> set[int]:
        seen, todo = set(), list(self.roots)
        while todo:
            pid = todo.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                for tid in os.listdir(f"/proc/{pid}/task"):
                    todo += [int(c) for c in Path(f"/proc/{pid}/task/{tid}/children").read_text().split()]
            except OSError:
                pass
        return seen

    def _run(self) -> None:
        while not self._stop.is_set():
            kb = 0
            for pid in self._tree():
                try:
                    kb += int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * self._page_kb
                except (OSError, IndexError, ValueError):
                    pass
            self.peak_kb = max(self.peak_kb, kb)
            self._stop.wait(self._every)

    def stop(self) -> int:
        self._stop.set()
        self._t.join()
        return self.peak_kb


def timed(cmd: list[str], stdout, stderr, env: dict | None = None,
          peak: Peak | None = None) -> tuple[int, dict, list[tuple[float, str]]]:
    """Run ``cmd``; its exit code, wall and CPU seconds and peak RSS, and each stdout line with the seconds since start;
    ``peak`` also samples its process tree."""
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr, text=True, env={**os.environ, **(env or {})})
    if peak is not None:
        peak.roots.append(p.pid)
    lines = []
    for line in p.stdout:
        lines.append((time.monotonic() - t0, line.rstrip("\n")))
        stdout.write(line)
    _, status, ru = os.wait4(p.pid, 0)
    p.returncode = os.waitstatus_to_exitcode(status)
    return p.returncode, {"wall_s": time.monotonic() - t0, "user_s": ru.ru_utime, "sys_s": ru.ru_stime,
                          "maxrss_kb": ru.ru_maxrss}, lines


def jsonl(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


# ---- stage ----------------------------------------------------------------------------------------------------------------

def stage(a) -> int:
    out = Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    plan = []
    for K in (int(x) for x in a.ks.split(",")):
        d = a6.template(K)
        for c in (int(x) for x in a.sizes.split(",")):
            sd = out / f"k{K}-c{c}"
            sd.mkdir(parents=True, exist_ok=True)
            program = P.mixed_population_program([(d, c)])
            query = P.template_instances_query([d])
            part = P.partition(program, query)
            (sd / "partition.json").write_text(json.dumps(part, sort_keys=True))
            (sd / "program.json").write_bytes(canonical_json(encode_program(program)))
            ru0 = os.times()
            [st] = a6.stage_template(K, c, 0, {"partition": P.digest(part), "program": part["program"]}, sd / "stage", None)
            ru1 = os.times()
            rec = {"K": K, "c": c, "dir": str(st["dir"]), "pin": st["pin"], "ands": st["ands"], "unit_ands": unit_ands(K),
                   "unit_nnz": st["unit_nnz"], "k_log": st["meta"].get("k_log"), "cap": st["cap"], "twin": st["twin"],
                   "build_s": st["seconds"]["build"], "stage_s": st["seconds"]["stage"],
                   "stage_cpu_s": round(ru1.user + ru1.system - ru0.user - ru0.system, 2)}
            jsonl(out / "records" / "stage.jsonl", rec)
            for k in draws(c, a.draws):
                p = a0.run([a.lean, "draw", "--program", str(sd / "program.json"), "--partition", str(sd / "partition.json"),
                            "--stratified", str(k)], sd / f"draw-k{k}.log")
                line = next((x for x in p.stdout.splitlines() if x.startswith("DRAW ")), None)
                if line is None:
                    raise SystemExit(f"Lean draw k={k} on k{K}-c{c}: no DRAW line (rc {p.returncode}): {p.stderr[-400:]}")
                units = sorted(json.loads(line.split(" ", 1)[1])["units"])
                (sd / f"draw-k{k}.json").write_text(json.dumps({"law": "subset", "population": c, "k": len(units), "units": units}))
                plan.append({"K": K, "c": c, "k": k, "dir": str(sd), "stage_dir": str(st["dir"]), "pin": st["pin"]})
            print(f"ZKCOST staged K={K} c={c}: {st['ands']} ANDs a unit (formula {unit_ands(K)}), k_log {rec['k_log']}, "
                  f"cap {st['cap']}, twin {st['twin']['agrees']}, stage {rec['stage_s']} s", flush=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    return 0


# ---- prove ----------------------------------------------------------------------------------------------------------------

def session(a, pt: dict, arm: str, mode: str, rep: int, out: Path) -> dict:
    """One prover process against its own `serve --zk`, proving ``warm + runs`` sessions of the point's draw."""
    sd, d, c, k = Path(pt["dir"]), Path(pt["stage_dir"]), pt["c"], pt["k"]
    warm, runs = (0, 1) if mode == "cold" else (a.warm, a.runs)
    o = out / f"k{pt['K']}-c{c}-k{k}" / f"{arm}-{mode}{rep}"
    o.mkdir(parents=True, exist_ok=True)
    port = free_port()
    env = {"FC_HOST_WITNESS": "1", "FC_FIREWALL": "1" if arm == "fw" else "0",
           **(a0.firewall_env({"partition_file": sd / "partition.json", "program_file": sd / "program.json"}) if arm == "fw" else {})}
    serve_log = open(o / "serve.log", "w")
    serve = subprocess.Popen([a.bin, "serve", "--zk", "--listen", f"127.0.0.1:{port}", "--out", str(o / "sessions"),
                              "--instances", str(d / f"pub-{c}.bin"), "--circuit", str(d / "circuit.txt"), "--tables", str(d / "tables"),
                              "--pin", pt["pin"], "--operator", "zk-cost", "--draw-file", str(sd / f"draw-k{k}.json"),
                              "--sessions", str(warm + runs)], stdout=serve_log, stderr=subprocess.STDOUT, text=True,
                             env={**os.environ, **env})
    peak = Peak()
    peak.roots.append(serve.pid)
    serve_usage = None
    try:
        while "SERVING" not in (o / "serve.log").read_text() and serve.poll() is None:
            time.sleep(0.2)
        with open(o / "prove.out", "w") as fo, open(o / "prove.err", "w") as fe:
            rc, usage, lines = timed([a.bin, "prove", "--zk", *([] if a.cpu else ["--gpu"]), "--verifier", f"127.0.0.1:{port}",
                                      "--instances", str(d / f"inst-{c}.bin"), "--circuit", str(d / "circuit.txt"),
                                      "--tables", str(d / "tables"), "--runs", str(runs), "--warm", str(warm)], fo, fe, env, peak)
        deadline = time.monotonic() + 900
        while serve.returncode is None and time.monotonic() < deadline:
            pid, status, ru = os.wait4(serve.pid, os.WNOHANG)
            if pid:
                serve.returncode = os.waitstatus_to_exitcode(status)
                serve_usage = {"serve_user_s": ru.ru_utime, "serve_sys_s": ru.ru_stime, "serve_maxrss_kb": ru.ru_maxrss}
            else:
                time.sleep(0.5)
    finally:
        if serve.poll() is None:
            serve.terminate()
            serve.wait(timeout=60)
        serve_log.close()
        peak_kb = peak.stop()
    lives = []
    for t, line in lines:
        if line.startswith("LIVE\t"):
            v = json.loads(line[5:])
            fw = {}
            for b in v.get("buckets") or []:
                for key, x in (b or {}).items():
                    if (key.startswith("t.firewall") or key.startswith("cpu.firewall")) and isinstance(x, (int, float)):
                        fw[key] = fw.get(key, 0.0) + float(x)
            lives.append({"t_emit_s": round(t, 3), "firewall": fw, "proof_bytes": sum(v.get("proof_bytes") or [0]),
                          **{x: v.get(x) for x in ("run", "warm", "accepted", "why", "prover_stopped", "e2e_s", "witness_s",
                                                   "witness_prebuilt_s", "prove_total_s", "wait_s", "wire_sent", "wire_recv",
                                                   "k_log", "dense_m", "instances", "gpu", "prover")}})
    built = next((float(x.rsplit("built in ", 1)[1].rstrip("s")) for x in (o / "prove.err").read_text().splitlines()
                  if x.startswith("statement ") and " built in " in x), None)
    sessions = sorted((o / "sessions").glob("*/session.json"), key=lambda f: f.stat().st_mtime)
    served = [json.loads(f.read_text()) for f in sessions]
    rec = {"K": pt["K"], "c": c, "k": k, "arm": arm, "mode": mode, "rep": rep, "warm": warm, "runs": runs, "rc": rc,
           "gpu_uuid": os.environ.get("GPU_LEASE_UUID"), "cpu": bool(a.cpu), "threads": os.environ.get("RAYON_NUM_THREADS"),
           **{x: round(y, 3) for x, y in {**usage, **(serve_usage or {})}.items()}, "peak_rss_kb": peak_kb,
           "built_s": built, "lives": lives, "dir": str(o),
           "sessions": [str(f.parent) for f in sessions],
           "serve_verify_s": [((r or {}).get("verdict") or {}).get("verify_s") for r in served],
           "accepted": rc == 0 and len(lives) == warm + runs and all(x["accepted"] is True for x in lives)}
    print(f"ZKCOST prove K={pt['K']} c={c} k={k} {arm} {mode}{rep}: rc {rc}, wall {usage['wall_s']:.2f} s, built {built} s, "
          f"e2e {[x['e2e_s'] and round(x['e2e_s'], 2) for x in lives]}, k_log {[x['k_log'] for x in lives]}, "
          f"dense_m {[x['dense_m'] for x in lives]}, peak {peak_kb / 2**20:.1f} GiB (prover, serve and firewall summed), "
          f"accepted {rec['accepted']}"
          + ("" if rec["accepted"] else f" ({[x['why'] for x in lives]}; {(o / 'prove.err').read_text()[-300:]!r})"), flush=True)
    return rec


def prove(a) -> int:
    out = Path(a.out)
    plan = json.loads((out / "plan.json").read_text())
    if a.points is not None:
        plan = [plan[int(i)] for i in a.points.split(",")]
    if not a.cpu:
        q = "name,uuid,driver_version,memory.total,compute_cap,clocks.max.sm,power.limit"
        (out / "records" / "gpu.csv").write_text(subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv"],
                                                                 capture_output=True, text=True).stdout)
    ok, arms, tried = True, a.arms.split(","), set()
    # an arm whose first point failed every session stays dropped across --points calls
    earlier = read_jsonl(out / "records" / "prove.jsonl")
    for arm in list(arms):
        mine = [r for r in earlier if r["arm"] == arm]
        if mine:
            tried.add(arm)
            first = (mine[0]["K"], mine[0]["c"], mine[0]["k"])
            if not any(r["accepted"] for r in mine if (r["K"], r["c"], r["k"]) == first):
                arms.remove(arm)
    for pt in plan:
        for arm in list(arms):
            recs = [session(a, pt, arm, "cold", rep, out / "points") for rep in range(a.cold)]
            if a.runs:
                recs.append(session(a, pt, arm, "warm", 0, out / "points"))
            for rec in recs:
                jsonl(out / "records" / "prove.jsonl", rec)
                ok &= rec["accepted"]
            if arm not in tried and not any(r["accepted"] for r in recs):
                print(f"ZKCOST arm {arm} dropped: every session of its first point failed", flush=True)
                arms.remove(arm)
            tried.add(arm)
    return 0 if ok else 1


# ---- verify ---------------------------------------------------------------------------------------------------------------

def verify(a) -> int:
    out = Path(a.out)
    keys: dict[str, dict] = {}
    ok = True
    for r in read_jsonl(out / "records" / "prove.jsonl"):
        if not r["accepted"]:
            continue
        sd = out / f"k{r['K']}-c{r['c']}"
        d = sd / "stage" / "c0"
        if str(sd) not in keys:
            p = a0.run([a.lean, "archive-put", str(sd / "archive"), str(d / "circuit.txt"), str(d / f"pub-{r['c']}.bin"),
                        str(sd / "partition.json"), str(sd / "program.json")], sd / "archive-put.log")
            keys[str(sd)] = {Path(x.split()[1]).name: x.split()[0] for x in p.stdout.splitlines() if len(x.split()) == 2}
        kk = keys[str(sd)]
        picks = r["sessions"] if r["mode"] == "cold" and r["arm"] == "fw" else \
            r["sessions"][r["warm"]:r["warm"] + 1] if r["mode"] == "warm" else []
        for s in picks:
            log = Path(r["dir"]) / f"lean-verify-{Path(s).name}"
            with open(f"{log}.log", "w") as fo, open(f"{log}.err", "w") as fe:
                rc, usage, lines = timed([a.lean, "verify", "--statement", a0.STATEMENT_ID, "--archive", str(sd / "archive"),
                                          "--circuit", kk.get("circuit.txt", ""), "--public", kk.get(f"pub-{r['c']}.bin", ""),
                                          "--partition", kk.get("partition.json", ""), "--program", kk.get("program.json", ""),
                                          "--zk", "--session", s], fo, fe)
            line = next((x for _, x in lines if x.startswith("VERDICT ")), None)
            v = json.loads(line.split(" ", 1)[1]) if line else {"accepted": False, "why": f"no verdict (rc {rc})"}
            rec = {"K": r["K"], "c": r["c"], "k": r["k"], "arm": r["arm"], "mode": r["mode"], "rep": r["rep"], "session": s,
                   "accepted": v.get("accepted") is True, "why": v.get("why"), "partition": v.get("partition"),
                   **{x: round(y, 3) for x, y in usage.items()}}
            jsonl(out / "records" / "verify.jsonl", rec)
            ok &= rec["accepted"]
            print(f"ZKCOST verify K={r['K']} c={r['c']} k={r['k']} {r['arm']} {r['mode']}{r['rep']}: {rec['accepted']} "
                  f"{rec['why'] or ''} {usage['wall_s']:.2f} s wall, {usage['user_s'] + usage['sys_s']:.2f} s CPU", flush=True)
    return 0 if ok else 1


# ---- fit ------------------------------------------------------------------------------------------------------------------

def wls(rows: list[tuple[list[float], float]], names: list[str]) -> dict:
    """Least squares in relative error (weights 1/y^2): coefficients, their standard errors and the relative residual."""
    x = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=float)
    w = 1.0 / y ** 2
    scale = np.abs(x).max(axis=0)
    scale[scale == 0] = 1.0
    xs = x / scale
    a = xs.T @ (w[:, None] * xs)
    coef_s = np.linalg.solve(a, xs.T @ (w * y))
    res = y - xs @ coef_s
    dof = len(y) - len(names)
    s2 = float(w @ res ** 2) / dof if dof > 0 else float("nan")
    cov = s2 * np.linalg.inv(a) if dof > 0 else np.full_like(a, np.nan)
    se = np.sqrt(np.diag(cov)) / scale
    coef = coef_s / scale
    return {"coef": {n: float(c) for n, c in zip(names, coef)}, "se": {n: float(s) for n, s in zip(names, se)},
            "sigma": sqrt(s2) if dof > 0 else None, "points": len(y), "dof": dof}


def fit(a) -> int:
    out = Path(a.out)
    stages = {(r["K"], r["c"]): r for r in read_jsonl(out / "records" / "stage.jsonl")}
    proves = [r for r in read_jsonl(out / "records" / "prove.jsonl") if r["accepted"]]
    verifies = [r for r in read_jsonl(out / "records" / "verify.jsonl") if r["accepted"]]
    ks = sorted({K for K, _ in stages})
    arms = sorted({r["arm"] for r in proves})
    result: dict = {"run": a.run, "unit_ands": {K: unit_ands(K) for K in ks},
                    "staged_ands": {K: next(r["ands"] for (k_, _), r in stages.items() if k_ == K) for K in ks}, "fits": {}}

    def put(name, rows, names):
        # a single K makes "one" and "ands" one column, and --draws c makes c and k one: such a fit has no solution
        x = np.array([r[0] for r in rows], dtype=float)
        scale = np.abs(x).max(axis=0) if len(rows) else None
        if len(rows) > len(names) and np.linalg.matrix_rank(x / np.where(scale == 0, 1.0, scale)) == len(names):
            result["fits"][name] = wls(rows, names)

    for K in ks:
        put(f"stage/K{K}", [([1, c], r["stage_s"]) for (k_, c), r in stages.items() if k_ == K], ["one", "c"])
        whole = all(r["c"] == r["k"] for r in proves if r["K"] == K)
        ck, ckn = (lambda r: [1, r["k"]], ["one", "k"]) if whole else (lambda r: [1, r["c"], r["k"]], ["one", "c", "k"])
        vs = [r for r in verifies if r["K"] == K]
        put(f"verify/K{K}", [(ck(r), r["wall_s"]) for r in vs], ckn)
        put(f"verify_cpu/K{K}", [(ck(r), r["user_s"] + r["sys_s"]) for r in vs], ckn)
        for arm in arms:
            cold = [r for r in proves if r["K"] == K and r["arm"] == arm and r["mode"] == "cold"]
            put(f"cold/{arm}/K{K}", [(ck(r), r["wall_s"]) for r in cold], ckn)
            put(f"setup/{arm}/K{K}", [([1, r["c"]], r["wall_s"] - r["lives"][0]["e2e_s"]) for r in cold], ["one", "c"])
            timed_ = [(r, x) for r in proves if r["K"] == K and r["arm"] == arm and r["mode"] == "warm"
                      for x in r["lives"] if not x["warm"]]
            put(f"session/{arm}/K{K}", [(ck(r), x["e2e_s"]) for r, x in timed_], ckn)
    def features(K: int, c: int, k: int) -> list[float]:
        return [1, unit_ands(K), unit_ands(K) * c, unit_ands(K) * k]

    joint = ["one", "ands", "ands*staged", "ands*drawn"]
    for arm in arms:
        cold = [r for r in proves if r["arm"] == arm and r["mode"] == "cold"]
        put(f"cold/{arm}/joint", [(features(r["K"], r["c"], r["k"]), r["wall_s"]) for r in cold], joint)
        put(f"setup/{arm}/joint", [(features(r["K"], r["c"], r["k"])[:3], r["wall_s"] - r["lives"][0]["e2e_s"]) for r in cold],
            joint[:3])
        fw = [(r, x) for r in proves if r["arm"] == arm and r["mode"] == "warm" for x in r["lives"] if not x["warm"]]
        put(f"session/{arm}/joint", [(features(r["K"], r["c"], r["k"]), x["e2e_s"]) for r, x in fw], joint)
        if fw and any(x["firewall"] for _, x in fw):
            result.setdefault("firewall", {})[arm] = {
                f"K{K}-c{c}-k{k}": {key: float(np.median([x["firewall"].get(key, 0.0) for r, x in fw if (r["K"], r["c"], r["k"]) == (K, c, k)]))
                                    for key in sorted({key for _, x in fw for key in x["firewall"]})}
                for K, c, k in sorted({(r["K"], r["c"], r["k"]) for r, _ in fw})}
    put("verify/joint", [(features(r["K"], r["c"], r["k"]), r["user_s"] + r["sys_s"]) for r in verifies], joint)
    put("stage/joint", [(features(K, c, 1)[:3], r["stage_s"]) for (K, c), r in stages.items()], joint[:3])

    # costfit.Observations of the design's path (arm fw): a point's cold processes' mean wall, the mean CPU time of Lean's
    # verification of its sessions (either arm's: the firewall doesn't change what the verifier checks) and its statement's
    # staging
    obs = []
    arm = "fw" if "fw" in arms else (arms[0] if arms else None)
    for K, c, k in sorted({(r["K"], r["c"], r["k"]) for r in proves if r["arm"] == arm and r["mode"] == "cold"}):
        cold = [r["wall_s"] for r in proves if (r["K"], r["c"], r["k"], r["arm"], r["mode"]) == (K, c, k, arm, "cold")]
        ver = [r["user_s"] + r["sys_s"] for r in verifies if (r["K"], r["c"], r["k"]) == (K, c, k)]
        obs.append({"source": a.run or "zk_cost", "template": f"ScaledMmS8Coordinate_v1{{K={K}}}", "ands": unit_ands(K),
                    "staged": c, "drawn": k, "prove": float(np.mean(cold)),
                    "verify": float(np.mean(ver)) if ver else None, "stage": stages[(K, c)]["stage_s"]})
    result["observations_arm"] = arm
    (out / "observations.json").write_text(json.dumps(obs, indent=1))
    (out / "fit.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    lines = ["| fit | coefficient | value | ± SE | points | σ (relative) |", "|---|---|---|---|---|---|"]
    for name, f in result["fits"].items():
        sigma = "-" if f["sigma"] is None else f"{f['sigma']:.3f}"
        for i, (n, v) in enumerate(f["coef"].items()):
            head = (name, f["points"], sigma) if i == 0 else ("", "", "")
            lines.append(f"| {head[0]} | {n} | {v:.4g} | {f['se'][n]:.2g} | {head[1]} | {head[2]} |")
    (out / "fit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("ZKCOST " + json.dumps({n: {k: f"{v:.4g} ± {f['se'][k]:.2g}" for k, v in f["coef"].items()}
                                  for n, f in result["fits"].items()}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["stage", "prove", "verify", "fit"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin")
    ap.add_argument("--lean")
    ap.add_argument("--ks", default="4096,14336")
    ap.add_argument("--sizes", default="1,8,64,512")
    ap.add_argument("--draws", default="1,8,64,c", help="stage: the draw sizes a statement is proved at, c meaning all its units")
    ap.add_argument("--points", default=None, help="prove: only these indices of plan.json (one GPU lease a point)")
    ap.add_argument("--arms", default="fw,nofw", help="fw: the firewall's send check on (the design's path); nofw: off")
    ap.add_argument("--cold", type=int, default=2, help="one-session prover processes a point and arm")
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3, help="the warm process's timed sessions; 0: no warm process")
    ap.add_argument("--cpu", action="store_true", help="prove: on the node's CPUs (no --gpu), as the CPU nodes prove a draw")
    ap.add_argument("--run", default=os.environ.get("RESEARCH_RUN_ID"), help="fit: the run id the observations name")
    a = ap.parse_args()
    if a.command in ("stage", "verify") and not a.lean or a.command == "prove" and not a.bin:
        ap.error(f"{a.command} needs {'--lean' if a.command != 'prove' else '--bin'}")
    return {"stage": stage, "prove": prove, "verify": verify, "fit": fit}[a.command](a)


if __name__ == "__main__":
    raise SystemExit(main())
