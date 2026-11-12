"""Flock's zero-knowledge session cost on sm_120, for the sampling planner's cost model (`spot.costfit`): A6's
W8A8 coordinates (`ScaledMmS8Coordinate_v1{K, DOT=BlackwellS8ImmaDot32_v1}`) staged in statements of c units, each
session proving k of them sampled by the verifier's Lean, with `flock-circuit prove --zk --gpu`, and the Lean verifier on the
sessions.

    python benchmarks/one_stage/zk_cost.py stage  --lean flock-verify --out DIR [--ks 4096,14336] [--sizes 1,8,64,512]
    python benchmarks/one_stage/zk_cost.py stage  --out DIR --units "fp8-coord:K=6144 router-row" [--sizes half,full] [--draws c]
    python benchmarks/one_stage/zk_cost.py prove  --bin flock-circuit --out DIR [--arms fw,nofw] [--cold 2] [--runs 3] [--cpu]
    python benchmarks/one_stage/zk_cost.py verify --lean flock-verify --out DIR
    python benchmarks/one_stage/zk_cost.py fit    --out DIR [--run RUN_ID]
    python benchmarks/one_stage/zk_cost.py overhead --out DIR [--card rtx-pro-6000-server]

* **Statements.** For each K and c, a one-template program of c units (`mixed_population_program`), its
  `Q_template_instances` partition, and one statement staged by A6's `stage_template` bound to it (stand-in values; a
  Boolean circuit's cost doesn't depend on them). Its stage seconds exclude building the circuit, which a template pays
  once.
* **Served proof-block types** (`--units`): instead of A6's coordinates, each of `benchmarks/served_units/stage.py`'s types
  (GLM-5.3's at TP8, synthetic values), staged by that script bound to the Program of its instances and that Program's
  partition, which it writes beside the statement: `--sizes` takes counts or `full` (the most instances under the
  prover's 2^35 witness bits, from a probe staging's k_log, whose 2 instances a size of 2 reuses) and `half`. Their
  samples are uniform subsets the server reads (`serve --draw-file`), every instance at k = c, and Lean verifies the
  newest session of each point and arm against the type's own partition and Program, which it must accept: a refusal,
  whatever its reason, is a failed check.
  A type staged as a templated statement (a table read: its stage's ``unit.tmpl``) is proved, served and verified at
  `TMPL_STATEMENT` on its unit rows, placed once by ``--tmpl-unit-bin`` (`flock-tmpl-unit`).
* **Samples.** k in {1, 8, 64, c} (`--draws`), at most c, each by Lean `flock-verify draw --stratified k` over the statement's
  program and partition, on a development challenger's coins opened after the statement's partition digest and pin
  (`a0.challenger_draw`), served from that file (`serve --zk --draw-file`). `--draws c` proves every unit: the cost of c
  sampled proof blocks in one statement. An entry `k:arm+arm` runs only those arms at k (`c:dev+nofw 8:dev+nofw+fw`).
* **Sessions.** Per point and arm, `--cold` processes proving one session each (`--runs 1 --warm 0`: one touched
  statement's prover) and one proving `--warm` + `--runs` sessions (the in-process session cost), the whole plan
  `--passes` times with each family's sizes in ABBA order, so that a drift in the node's load falls on every size alike.
  Each session records its prover's and ZK server's user and system seconds (their process trees', between the prover's
  LIVE lines) and the node's load, so that a session slowed by a busy node shows as one. Arm `fw` is the
  design's path: the ZK server's send check on every session (`FC_FIREWALL=1`), which rechecks the host's witness; `nofw`
  is the same with the check off. Both take the host's witness (`FC_HOST_WITNESS=1`), as proof_throughput's ZK point does,
  so their difference is the ZK server's. Arm `dev` leaves both unset: the device builds the witness, which the ZK server
  can't recheck (flock-circuit refuses `FC_FIREWALL=1` without the host's witness).
* **Overhead** (`overhead`): per served type and arm, the timed sessions' end-to-end seconds at k = c fitted as
  `fixed + marginal c` over its sizes, with the marginal's standard error, each of the session's parts (`PARTS`: witness,
  proving, the wait on the ZK server, the rest) fitted the same way, and a dot product's proof overhead: its GPU-seconds a
  proof block over the GPU-seconds the same card's datasheet dense peak needs for the block's K MACs in its operand format
  (`verity.hardware.cards.rate`; `--card none` on a CPU node, whose witness and proving run on its CPUs rather than a device).
  Per statement and arm, its sessions at every k it drew, k = c among them, fitted over k (`drawn`): the statement's
  fixed seconds and a sampled block's, the deployed sample's price; `fw` less `nofw` at the same samples is the send
  check's increment. Its `observations.json` holds each arm's `costfit.Observation`s, prove the median timed session.
* **Verification.** Lean `verify --zk --partition --program` on each cold session of arm `fw` and the first timed session
  of each warm process, one at a time.
* **Fit.** Weighted least squares in relative error, with standard errors from the residuals: per K, the cold process
  `a + a_c c + a_k k` (GPU-s), the in-process session `g + g_c c + g_k k`, the process's setup (its wall less its session),
  staging `s + s_c c` and verification `v + v_c c + v_k k` (CPU-s); and across K, `costfit`'s features (one, ands,
  ands*staged, ands*sampled). `observations.json` holds `costfit.Observation`s: prove is a cold process's wall, verify the
  single-threaded verifier's CPU time (its wall stretches on a loaded node).
* **CPU** (`--cpu`, `zk_cost_cpu.sh`): the same sessions by `flock-circuit prove --zk` on the node's CPUs, as the CPU nodes
  prove a block's sample. Each session record carries `peak_rss_kb`, the sampled peak of the prover's, its live verifier's
  and the ZK server's Lean's resident memory summed, which is what a node holds for one session.
"""
from __future__ import annotations

import argparse
import hashlib
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
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import a0  # noqa: E402
from verity.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.protocols.spot.verifier.one_stage import draw as D, partition as P  # noqa: E402
from verity.protocols.spot.optimize import SHA512_ANDS  # noqa: E402

#: a templated statement's id (`verity_flock.circuit.TMPL_STATEMENT`): its prover and server read the placed unit
TMPL_STATEMENT = "verity/flock-circuit/tmpl"

#: ANDs of the pieces of a unit with its rows' openings (benchmarks/sampling_plan's inputs: its pins, hm96 PROTOCOL §5)
S8_STEP, EPILOGUE, SHA512, HM96 = 4_190, 5_000, SHA512_ANDS, 116_240
#: the prover's witness bits a statement (70-class-sweep.sh's MAX_STATEMENT_BITS): a `full` statement's
MAX_BITS = 1 << 35
#: the served types whose proof block is one output coordinate of a dot product, K MACs (their static ``K``), and their
#: operands' format; a census name's stage.json has no statics
DOT = {"fp8-coord": "e4m3", "s8-coord": "s8"}
STAGE_PY = REPO / "benchmarks" / "served_units" / "stage.py"


def unit_ands(K: int) -> int:
    """One coordinate's ANDs with its rows' openings, as `costfit` counts them: K/32 s8 steps, the epilogue, two K-byte
    operand rows and three one-block rows (the scales and the output)."""
    return K // 32 * S8_STEP + EPILOGUE + 2 * (HM96 + SHA512 * ceil((K + 17) / 128)) + 3 * (HM96 + SHA512)


def draws(c: int, spec: str = "1,8,64,c") -> list[tuple[int, list[str] | None]]:
    """The sample sizes of a statement of c units: ``spec``'s entries, each a count or ``c`` (every unit), those at most c,
    each with the arms that run it (``k:arm+arm``; None: every arm)."""
    got: dict[int, set | None] = {}
    for x in spec.split(","):
        n, _, arms = x.partition(":")
        k = c if n == "c" else int(n)
        if k <= c:
            prev = got.get(k, set())
            got[k] = None if not arms or prev is None else prev | set(arms.split("+"))
    return [(k, None if got[k] is None else sorted(got[k])) for k in sorted(got)]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def proc_cpu(pids) -> tuple[float, float, int]:
    """The user and system seconds of ``pids`` so far, each with its reaped children's (``/proc/PID/stat``'s utime + cutime,
    stime + cstime), and how many had exited before their stat was read: their seconds reach their parent's once it reaps
    them."""
    u = s = gone = 0
    for pid in pids:
        try:
            f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            gone += 1
            continue
        u, s = u + int(f[11]) + int(f[13]), s + int(f[12]) + int(f[14])
    tck = os.sysconf("SC_CLK_TCK")
    return u / tck, s / tck, gone


def session_cpu(marks: list[dict], names: tuple[str, ...]) -> list[dict]:
    """Each session's share of a prover's marks (`Peak.mark`: one at its start, then one at each LIVE line): each named root's
    process tree's user and system seconds since the mark before, the node's 1-minute load at the session's end, and how
    many of the trees' processes exited between being listed and read at that end."""
    return [{"load1": round(b["load1"], 2), "gone": sum(x[2] for x in b["cpu"]),
             **{f"{n}_{k}_s": round(b["cpu"][i][j] - a["cpu"][i][j], 3) for i, n in enumerate(names)
                for j, k in enumerate(("user", "sys"))}}
            for a, b in zip(marks, marks[1:])]


class Peak:
    """The peak of the summed RSS of some processes and all their descendants, sampled every `every` seconds in a thread:
    a ZK server session's prover, its live verifier (`serve`) and the ZK server's Lean child peak together, and their sum is
    what a node must hold for the session (ru_maxrss gives only the largest one process). `mark` records each root's tree's
    CPU seconds, so that a session's are its own rather than its process's."""

    def __init__(self, every: float = 0.5) -> None:
        import threading
        self.roots: list[int] = []
        self.marks: list[dict] = []
        self.peak_kb = 0
        self._stop = threading.Event()
        self._every = every
        self._page_kb = os.sysconf("SC_PAGE_SIZE") // 1024
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def mark(self, t: float) -> None:
        self.marks.append({"t": round(t, 3), "load1": os.getloadavg()[0],
                           "cpu": [proc_cpu(self._tree([r])) for r in self.roots]})

    def _tree(self, roots: list[int] | None = None) -> set[int]:
        seen, todo = set(), list(self.roots if roots is None else roots)
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
    ``peak`` also samples its process tree and marks its roots' CPU seconds at the start and at each LIVE line."""
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr, text=True, env=a0.environ(env))
    if peak is not None:
        peak.roots.append(p.pid)
        peak.mark(0.0)
    lines = []
    for line in p.stdout:
        lines.append((time.monotonic() - t0, line.rstrip("\n")))
        if peak is not None and line.startswith("LIVE\t"):
            peak.mark(lines[-1][0])
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

def stage_served(spec: str, n: int, d: Path) -> dict:
    """``n`` instances of a served type (`benchmarks/served_units/stage.py`) into ``d``; its stage.json, with the CPU seconds."""
    d.mkdir(parents=True, exist_ok=True)
    ru0 = os.times()
    p = a0.run([sys.executable, str(STAGE_PY), spec, str(n), str(d)], d.parent / f"{d.name}.log", timeout=6 * 3600)
    ru1 = os.times()
    if p.returncode != 0 or not (d / "stage.json").exists():
        raise SystemExit(f"stage.py {spec} {n}: rc {p.returncode}: {p.stderr[-600:]}")
    return {**json.loads((d / "stage.json").read_text()),
            "stage_cpu_s": round(ru1.children_user + ru1.children_system - ru0.children_user - ru0.children_system, 2)}


def stage_units(a, out: Path) -> list[dict]:
    """Each ``--units`` type at each ``--sizes`` entry (a count, ``full`` or ``half``), and its samples' files."""
    plan = []
    for spec in a.units.split():
        tag = spec.lower().replace(":", "-").replace(",", "-").replace("=", "")
        probe = stage_served(spec, 2, out / f"{tag}-probe")
        per = (1 << probe["k_log"]) // (probe.get("vus_per_block") or 1)
        full = max(1, MAX_BITS // per)
        for size in a.sizes.split(","):
            want = {"full": full, "half": max(1, full // 2)}.get(size) or int(size)
            sd = out / f"{tag}-c{want}"
            if want == probe["n"] and (out / f"{tag}-probe").is_dir():
                sd.mkdir(parents=True, exist_ok=True)
                (out / f"{tag}-probe").rename(sd / "stage")
                st = probe
            else:
                st = stage_served(spec, want, sd / "stage")
            c = st["n"]
            name = spec.partition(":")[0]
            rec = {"unit": spec, "tag": tag, "c": c, "full": full, "ands": st["ands"], "k_log": st["k_log"],
                   "vus_per_block": st.get("vus_per_block"), "definition": st["definition"], "class": st.get("class"),
                   "statics": st["statics"], "macs": st["statics"]["K"] if name in DOT else 0, "format": DOT.get(name),
                   "stage_s": st["seconds"]["total"], "stage_cpu_s": st["stage_cpu_s"], "maxrss_gb": st["maxrss_gb"],
                   "circuit_sha512": st["circuit_sha512"], "twin": st["twin"]}
            jsonl(out / "records" / "stage.jsonl", rec)
            rng = np.random.default_rng(int.from_bytes(st["circuit_sha512"][:16].encode(), "little"))
            for k, arms in draws(c, a.draws):
                units = list(range(c)) if k == c else sorted(int(u) for u in rng.choice(c, size=k, replace=False))
                (sd / f"draw-k{k}.json").write_text(json.dumps({"law": "subset", "population": c, "k": k, "units": units}))
                std = sd / "stage"
                part, program = std / "partition.json", std / "program.json"
                plan.append({"unit": spec, "tag": tag, "c": c, "k": k, "arms": arms, "dir": str(sd), "stage_dir": str(std),
                             "pin": (std / "pin.txt").read_text().strip(), "inst": str(std / "inst.bin"),
                             "pub": str(std / "pub.bin"), "partition": str(part) if part.exists() else None,
                             "program": str(program) if program.exists() else None})
            print(f"ZKCOST staged {spec} c={c} (full {full}): {st['ands']} ANDs a unit, k_log {st['k_log']}, "
                  f"{st['definition']}, stage {st['seconds']['total']} s", flush=True)
    return plan


def stage(a) -> int:
    out = Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    if a.units:
        (out / "plan.json").write_text(json.dumps(stage_units(a, out), indent=1))
        return 0
    import a6
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
            statement = hashlib.sha512(canonical_json({"partition": P.digest(part), "pin": st["pin"]})).hexdigest()
            for k, arms in draws(c, a.draws):
                law = D.derive(program, query, D.parse_design(f"stratified:{k}"))
                p = a0.challenger_draw(a.lean, ["--program", str(sd / "program.json"), "--partition",
                                                str(sd / "partition.json"), "--stratified", str(k)], law, c, statement,
                                       sd / f"draw-k{k}.log")
                line = next((x for x in p.stdout.splitlines() if x.startswith("DRAW ")), None)
                if line is None:
                    raise SystemExit(f"Lean draw k={k} on k{K}-c{c}: no DRAW line (rc {p.returncode}): {p.stderr[-400:]}")
                units = sorted(json.loads(line.split(" ", 1)[1])["units"])
                (sd / f"draw-k{k}.json").write_text(json.dumps({"law": "subset", "population": c, "k": len(units), "units": units}))
                plan.append({"K": K, "c": c, "k": k, "arms": arms, "dir": str(sd), "stage_dir": str(st["dir"]), "pin": st["pin"]})
            print(f"ZKCOST staged K={K} c={c}: {st['ands']} ANDs a unit (formula {unit_ands(K)}), k_log {rec['k_log']}, "
                  f"cap {st['cap']}, twin {st['twin']['agrees']}, stage {rec['stage_s']} s", flush=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    return 0


# ---- prove ----------------------------------------------------------------------------------------------------------------

def placed_unit(a, d: Path) -> list[str]:
    """A templated stage's prover and server arguments: its unit rows placed at their positions (``d/unit.placed``, once,
    by ``--tmpl-unit-bin`` from the stage's template and layout mode) and its tables; none for a flat stage."""
    if not (d / "unit.tmpl").exists():
        return []
    placed = d / "unit.placed"
    if not placed.exists():
        if not a.tmpl_unit_bin:
            raise SystemExit(f"{d} stages a templated statement, whose prover reads --tmpl-unit-bin's placed unit")
        p = a0.run([a.tmpl_unit_bin, str(d / "unit.tmpl"), "--layout", (d / "unit.layout").read_text().strip(),
                    "--out", str(placed)], d / "tmpl-unit.log")
        if p.returncode:
            raise SystemExit(f"flock-tmpl-unit: exit {p.returncode} ({d / 'tmpl-unit.log'})")
    (d / "tables").mkdir(exist_ok=True)
    return ["--tmpl-unit", str(placed)]


def session(a, pt: dict, arm: str, mode: str, rep: int, out: Path) -> dict:
    """One prover process against its own `serve --zk`, proving ``warm + runs`` sessions of the point's sample."""
    sd, d, c, k = Path(pt["dir"]), Path(pt["stage_dir"]), pt["c"], pt["k"]
    warm, runs = (0, 1) if mode == "cold" else (a.warm, a.runs)
    o = out / (f"{pt['tag']}-c{c}-k{k}" if "unit" in pt else f"k{pt['K']}-c{c}-k{k}") / f"{arm}-{mode}{rep}"
    o.mkdir(parents=True, exist_ok=True)
    port = free_port()
    if "unit" in pt:
        inst, pub = pt["inst"], pt["pub"]
        tmpl = placed_unit(a, d)
        tables = ["--tables", str(d / "tables")] if (d / "tables").is_dir() else []
        tables += tmpl
        fw_files = {"partition_file": pt["partition"], "program_file": pt.get("program"),
                    **({"statement": TMPL_STATEMENT} if tmpl else {})}
    else:
        inst, pub, tables = str(d / f"inst-{c}.bin"), str(d / f"pub-{c}.bin"), ["--tables", str(d / "tables")]
        fw_files = {"partition_file": sd / "partition.json", "program_file": sd / "program.json"}
    env = {"FC_HOST_WITNESS": None if arm == "dev" else "1", "FC_FIREWALL": "1" if arm == "fw" else "0",
           **(a0.firewall_env(fw_files) if arm == "fw" else {})}
    serve_log = open(o / "serve.log", "w")
    serve = subprocess.Popen([a.bin, "serve", "--zk", "--listen", f"127.0.0.1:{port}", "--out", str(o / "sessions"),
                              "--instances", pub, "--circuit", str(d / "circuit.txt"), *tables,
                              "--pin", pt["pin"], "--operator", "zk-cost", "--draw-file", str(sd / f"draw-k{k}.json"),
                              "--sessions", str(warm + runs)], stdout=serve_log, stderr=subprocess.STDOUT, text=True,
                             env=a0.environ(env))
    peak = Peak()
    peak.roots.append(serve.pid)
    serve_usage = None
    try:
        while "SERVING" not in (o / "serve.log").read_text() and serve.poll() is None:
            time.sleep(0.2)
        with open(o / "prove.out", "w") as fo, open(o / "prove.err", "w") as fe:
            rc, usage, lines = timed([a.bin, "prove", "--zk", *([] if a.cpu else ["--gpu"]), "--verifier", f"127.0.0.1:{port}",
                                      "--instances", inst, "--circuit", str(d / "circuit.txt"),
                                      *tables, "--runs", str(runs), "--warm", str(warm)], fo, fe, env, peak)
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
                                                   "witness_prebuilt_s", "prove_total_s", "wait_s", "wait_e2e_s", "calls",
                                                   "wire_sent", "wire_recv", "k_log", "dense_m", "instances", "gpu", "prover")}})
    for x, cpu in zip(lives, session_cpu(peak.marks, ("serve", "prover"))):
        x["cpu_s"] = cpu
    built = next((float(x.rsplit("built in ", 1)[1].rstrip("s")) for x in (o / "prove.err").read_text().splitlines()
                  if x.startswith("statement ") and " built in " in x), None)
    sessions = sorted((o / "sessions").glob("*/session.json"), key=lambda f: f.stat().st_mtime)
    served = [json.loads(f.read_text()) for f in sessions]
    rec = {"K": pt.get("K"), "unit": pt.get("unit"), "tag": pt.get("tag"),
           "c": c, "k": k, "arm": arm, "mode": mode, "rep": rep, "warm": warm, "runs": runs, "rc": rc,
           "gpu_uuid": os.environ.get("GPU_LEASE_UUID"), "cpu": bool(a.cpu), "threads": os.environ.get("RAYON_NUM_THREADS"),
           **{x: round(y, 3) for x, y in {**usage, **(serve_usage or {})}.items()}, "peak_rss_kb": peak_kb,
           "built_s": built, "lives": lives, "dir": str(o),
           "sessions": [str(f.parent) for f in sessions],
           "serve_verify_s": [((r or {}).get("verdict") or {}).get("verify_s") for r in served],
           "accepted": rc == 0 and len(lives) == warm + runs and all(x["accepted"] is True for x in lives)}
    print(f"ZKCOST prove {pt.get('unit') or 'K=%s' % pt['K']} c={c} k={k} {arm} {mode}{rep}: rc {rc}, wall {usage['wall_s']:.2f} s, built {built} s, "
          f"e2e {[x['e2e_s'] and round(x['e2e_s'], 2) for x in lives]}, k_log {[x['k_log'] for x in lives]}, "
          f"dense_m {[x['dense_m'] for x in lives]}, peak {peak_kb / 2**20:.1f} GiB (prover, serve and firewall summed), "
          f"accepted {rec['accepted']}"
          + ("" if rec["accepted"] else f" ({[x['why'] for x in lives]}; {(o / 'prove.err').read_text()[-300:]!r})"), flush=True)
    return rec


def passes(plan: list[dict], n: int) -> list[tuple[int, dict]]:
    """The plan's points over ``n`` passes, each with its pass: a family's points (one served type's sizes, or A6's
    coordinates) in plan order on even passes and reversed on odd ones (ABBA), so that a drift in the node's load over the
    run falls on each size alike."""
    fams: dict[str, list[dict]] = {}
    for pt in plan:
        fams.setdefault(pt.get("tag") or "s8-coordinate-v1", []).append(pt)
    return [(p, pt) for p in range(n) for pts in fams.values() for pt in (pts if p % 2 == 0 else pts[::-1])]


def prove(a) -> int:
    out = Path(a.out)
    plan = json.loads((out / "plan.json").read_text())
    if a.points is not None:
        plan = [plan[int(i)] for i in a.points.split(",")]
    if not a.cpu:
        q = "name,uuid,driver_version,memory.total,compute_cap,clocks.max.sm,power.limit"
        (out / "records" / "gpu.csv").write_text(subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv"],
                                                                 capture_output=True, text=True).stdout)
    ok, arms = True, a.arms.split(",")
    # an arm whose first point of a family (A6's coordinates, or one served type) failed every session stays dropped for that
    # family across --points calls
    family = lambda r: r.get("tag") or "s8-coordinate-v1"  # noqa: E731
    tried, dropped = set(), set()
    earlier = read_jsonl(out / "records" / "prove.jsonl")
    for key in sorted({(r["arm"], family(r)) for r in earlier}):
        mine = [r for r in earlier if (r["arm"], family(r)) == key]
        tried.add(key)
        first = (mine[0]["c"], mine[0]["k"])
        if not any(r["accepted"] for r in mine if (r["c"], r["k"]) == first):
            dropped.add(key)
    for p, pt in passes(plan, a.passes):
        for arm in [x for x in arms if pt.get("arms") is None or x in pt["arms"]]:
            key = (arm, family(pt))
            if key in dropped:
                continue
            recs = [session(a, pt, arm, "cold", rep, out / "points") for rep in range(p * a.cold, (p + 1) * a.cold)]
            if a.runs:
                recs.append(session(a, pt, arm, "warm", p, out / "points"))
            for rec in recs:
                jsonl(out / "records" / "prove.jsonl", rec)
                ok &= rec["accepted"]
            if key not in tried and not any(r["accepted"] for r in recs):
                print(f"ZKCOST arm {arm} dropped for {key[1]}: every session of its first point failed", flush=True)
                dropped.add(key)
            tried.add(key)
    return 0 if ok else 1


# ---- verify ---------------------------------------------------------------------------------------------------------------

def lean_outcome(v: dict) -> str:
    """A Lean verdict as what happened: `accepted`, or `refused` for any reason."""
    return "accepted" if v.get("accepted") is True else "refused"


def verify_unit(a, r: dict, plan: dict) -> dict:
    """Lean on the newest session of a served type's warm process, with the statement's own partition and Program. Its record
    keeps Lean's `accepted` and `why`."""
    pt = plan[(r["tag"], r["c"], r["k"])]
    s, d = r["sessions"][-1], Path(pt["stage_dir"])
    tmpl = (d / "unit.tmpl").exists()
    log = Path(r["dir"]) / f"lean-verify-{Path(s).name}"
    with open(f"{log}.log", "w") as fo, open(f"{log}.err", "w") as fe:
        rc, usage, lines = timed([a.lean, "verify", "--zk", "--statement", TMPL_STATEMENT if tmpl else a0.STATEMENT_ID,
                                  "--circuit", str(d / "circuit.txt"), "--public", pt["pub"],
                                  *(["--tables", str(d / "tables")] if tmpl else []),
                                  *(["--partition", pt["partition"]] if pt["partition"] else []),
                                  *(["--program", pt["program"]] if pt.get("program") else []), "--session", s], fo, fe)
    line = next((x for _, x in lines if x.startswith("VERDICT ")), None)
    v = json.loads(line.split(" ", 1)[1]) if line else {"accepted": False, "why": f"no verdict (rc {rc})"}
    why = v.get("why") or ""
    rec = {"unit": r["unit"], "tag": r["tag"], "c": r["c"], "k": r["k"], "arm": r["arm"], "mode": r["mode"], "rep": r["rep"],
           "session": s, "accepted": v.get("accepted") is True, "why": v.get("why"), **{x: round(y, 3) for x, y in usage.items()}}
    jsonl(Path(a.out) / "records" / "verify.jsonl", rec)
    print(f"ZKCOST verify {r['unit']} c={r['c']} k={r['k']} {r['arm']}: {lean_outcome(rec)}"
          f"{f' ({why[:120]})' if why else ''} {usage['wall_s']:.2f} s wall", flush=True)
    return rec


def verify(a) -> int:
    out = Path(a.out)
    keys: dict[str, dict] = {}
    # every session is verified at its statement's Program and must be accepted
    accepted, verdicts = True, []
    proves = read_jsonl(out / "records" / "prove.jsonl")
    plan = {(p["tag"], p["c"], p["k"]): p for p in json.loads((out / "plan.json").read_text()) if "unit" in p}
    largest: dict[tuple, dict] = {}   # a served type's largest point a warm process of each arm proved
    for r in proves:
        if r.get("unit") and r["accepted"] and r["mode"] == "warm":
            key = (r["tag"], r["arm"])
            if key not in largest or (r["c"], r["k"]) >= (largest[key]["c"], largest[key]["k"]):
                largest[key] = r
    for r in largest.values():
        verdicts.append(verify_unit(a, r, plan))
        accepted &= verdicts[-1]["accepted"]
    for r in proves:
        if not r["accepted"] or r.get("unit"):
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
                   "accepted": v.get("accepted") is True, "why": v.get("why"), "partition": v.get("partition"), **{x: round(y, 3) for x, y in usage.items()}}
            jsonl(out / "records" / "verify.jsonl", rec)
            verdicts.append(rec)
            accepted &= rec["accepted"]
            print(f"ZKCOST verify K={r['K']} c={r['c']} k={r['k']} {r['arm']} {r['mode']}{r['rep']}: "
                  f"{'accepted' if rec['accepted'] else 'not accepted'} {rec['why'] or ''} {usage['wall_s']:.2f} s wall, "
                  f"{usage['user_s'] + usage['sys_s']:.2f} s CPU", flush=True)
    outcomes = [lean_outcome(v) for v in verdicts]
    print(f"ZKCOST verify: Lean accepted {outcomes.count('accepted')} of {len(outcomes)} sessions and refused "
          f"{outcomes.count('refused')}", flush=True)
    return 0 if accepted else 1


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
    # A6's coordinates only: the served types' fit is `overhead`'s
    stages = {(r["K"], r["c"]): r for r in read_jsonl(out / "records" / "stage.jsonl") if not r.get("unit")}
    proves = [r for r in read_jsonl(out / "records" / "prove.jsonl") if r["accepted"] and not r.get("unit")]
    verifies = [r for r in read_jsonl(out / "records" / "verify.jsonl") if r["accepted"] and not r.get("unit")]
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
    # verification of its sessions (either arm's: the ZK server doesn't change what the verifier checks) and its statement's
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


# ---- overhead -------------------------------------------------------------------------------------------------------------

#: the definition `overhead` reports, in a reader's words
DEFINITION = ("proof overhead: the GPU-seconds the prover takes to prove a proof block, over the GPU-seconds the same GPU "
              "needs to compute the block's MACs at its datasheet dense peak for the block's operand format")


def line(points: dict[int, list[float]]) -> dict:
    """``seconds = fixed + marginal n`` over every session at each n, by least squares, with the marginal's standard error
    from the residuals (one n: no slope; two sessions: no error)."""
    xs = [(n, y) for n, ys in points.items() for y in ys]
    if len(points) < 2:
        return {"fixed_s": None, "marginal_s": None, "marginal_se_s": None, "points": len(xs)}
    a, y = np.array([[1.0, n] for n, _ in xs]), np.array([y for _, y in xs])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    dof = len(xs) - 2
    se = sqrt(float(((y - a @ coef) ** 2).sum()) / dof * np.linalg.inv(a.T @ a)[1, 1]) if dof > 0 else None
    return {"fixed_s": float(coef[0]), "marginal_s": float(coef[1]), "marginal_se_s": se, "points": len(xs)}


#: a session's parts of its end-to-end seconds, from its LIVE record: the witness, the proving, the wait on the ZK server's live
#: verifier inside them (`wait_e2e_s`; `wait_s` adds the final verdict's, after the proofs are sent), and the rest of the
#: prover's host work. On a CPU node the first two run on its CPUs, on a GPU node on the device
PARTS = {"witness": lambda x: x.get("witness_s") or 0.0, "proving": lambda x: x["prove_total_s"],
         "wait": lambda x: x["wait_e2e_s"],
         "rest": lambda x: x["e2e_s"] - (x.get("witness_s") or 0.0) - x["prove_total_s"] - x["wait_e2e_s"]}


def overhead(a) -> int:
    from verity.hardware import cards
    out = Path(a.out)
    # --card none: a CPU node's sessions, which no GPU's peak divides
    card = None if a.card == "none" else cards.load(a.card)
    stages = [r for r in read_jsonl(out / "records" / "stage.jsonl") if r.get("unit")]
    proves = [r for r in read_jsonl(out / "records" / "prove.jsonl") if r.get("unit") and r["accepted"]]
    verifies = [r for r in read_jsonl(out / "records" / "verify.jsonl") if r.get("unit")]
    gpu = (out / "records" / "gpu.csv").read_text().splitlines()[1:2] if (out / "records" / "gpu.csv").exists() else []
    if gpu:
        name, cc = gpu[0].split(",")[0].strip(), gpu[0].split(",")[4].strip().replace(".", "")
        if (card or {}).get("arch", "").rstrip("a") != f"sm_{cc}":
            raise SystemExit(f"overhead: the run's GPU is {name} (sm_{cc}) but --card {a.card} is {(card or {}).get('arch')}; "
                             "pass the card it ran on (zk_cost.sh's ZK_CARD, e.g. h200-sxm-141gb for an H200)")
    types: dict[str, dict] = {}
    obs: dict[str, list[dict]] = {}
    for tag in dict.fromkeys(r["tag"] for r in stages):
        st = {r["c"]: r for r in stages if r["tag"] == tag}
        s0 = st[max(st)]
        peak = float(cards.rate(card, fmt=s0["format"])) if s0["macs"] and card else None
        t = {"unit": s0["unit"], "definition": s0["definition"], "class": s0.get("class"), "ands": s0["ands"],
             "k_log": s0["k_log"], "full": s0["full"], "sizes": sorted(st), "macs": s0["macs"], "format": s0["format"],
             "peak_mac_per_s": peak, "stage_s": {c: r["stage_s"] for c, r in st.items()}, "arms": {}, "drawn": {}}
        for arm in sorted({r["arm"] for r in proves if r["tag"] == tag}):
            mine = [r for r in proves if r["tag"] == tag and r["arm"] == arm]
            whole = {c: [x for r in mine if r["mode"] == "warm" and r["c"] == r["k"] == c for x in r["lives"] if not x["warm"]]
                     for c in st}
            whole = {c: xs for c, xs in whole.items() if xs}
            if not whole:
                continue
            e2e = {c: [x["e2e_s"] for x in xs] for c, xs in whole.items()}
            cmax = max(whole)
            med = float(np.median(e2e[cmax]))
            arm_rec = {**line(e2e), "e2e_s": {c: float(np.median(v)) for c, v in e2e.items()},
                       "prove_total_s": {c: float(np.median([x["prove_total_s"] for x in xs])) for c, xs in whole.items()},
                       "witness_s": {c: float(np.median([x["witness_s"] or 0.0 for x in xs])) for c, xs in whole.items()},
                       "dense_m": {c: xs[0]["dense_m"] for c, xs in whole.items()}, "amortized_s": med / cmax,
                       "cold_wall_s": {c: [r["wall_s"] for r in mine if r["mode"] == "cold" and r["c"] == r["k"] == c] for c in whole},
                       "peak_rss_kb": max(r["peak_rss_kb"] for r in mine),
                       # records from before the harness kept wait_e2e_s have no parts
                       "parts": {n: {**line({c: [f(x) for x in xs] for c, xs in whole.items()}),
                                     "s": {c: float(np.median([f(x) for x in xs])) for c, xs in whole.items()}}
                                 for n, f in PARTS.items()}
                       if all(x.get("wait_e2e_s") is not None for xs in whole.values() for x in xs) else {},
                       "lean": [{"c": v["c"], "k": v["k"], "accepted": v["accepted"], "why": v["why"]}
                                for v in verifies if v["tag"] == tag and v["arm"] == arm]}
            cpu = {c: [x["cpu_s"] for x in xs if x.get("cpu_s")] for c, xs in whole.items()}
            if any(cpu.values()):
                arm_rec["cpu_s"] = {c: {k: float(np.median([y[k] for y in ys])) for k in ys[0]} for c, ys in cpu.items() if ys}
            if peak:
                native = s0["macs"] / peak
                arm_rec["overhead_amortized"] = arm_rec["amortized_s"] / native
                if arm_rec["marginal_s"]:
                    arm_rec["overhead_marginal"] = arm_rec["marginal_s"] / native
            t["arms"][arm] = arm_rec
        # per statement and arm, its sessions fitted over the sampled blocks k (k = c among them): the statement's fixed
        # seconds and a sampled block's
        for c in st:
            fits = {}
            for arm in ("dev", "nofw", "fw"):
                pts: dict[int, list[float]] = {}
                for r in proves:
                    if r["tag"] == tag and r["arm"] == arm and r["mode"] == "warm" and r["c"] == c:
                        pts.setdefault(r["k"], []).extend(x["e2e_s"] for x in r["lives"] if not x["warm"])
                if pts:
                    fits[arm] = {**line(pts), "e2e_s": {k: float(np.median(v)) for k, v in pts.items()}}
            if fits:
                t["drawn"][c] = fits
        types[tag] = t
        # costfit.Observations per arm: each statement and sample's median timed session end to end (a warm prover's; the live
        # verifier's waits in it), Lean's CPU time where it verified that point and arm, and the statement's staging (CPU-s)
        for arm in sorted({r["arm"] for r in proves if r["tag"] == tag}):
            for c, k in sorted({(r["c"], r["k"]) for r in proves if r["tag"] == tag and r["arm"] == arm}):
                xs = [x["e2e_s"] for r in proves if (r["tag"], r["arm"], r["mode"], r["c"], r["k"]) == (tag, arm, "warm", c, k)
                      for x in r["lives"] if not x["warm"]]
                ver = [v["user_s"] + v["sys_s"] for v in verifies if (v["tag"], v["arm"], v["c"], v["k"]) == (tag, arm, c, k)]
                if xs and c in st:
                    obs.setdefault(arm, []).append(
                        {"source": a.run or "zk_cost", "template": s0["definition"], "ands": s0["ands"], "staged": c, "drawn": k,
                         "prove": float(np.median(xs)), "verify": float(np.mean(ver)) if ver else None,
                         "stage": st[c].get("stage_cpu_s", st[c]["stage_s"])})
    rec = {"schema": "verity/proof-overhead/v1", "definition": DEFINITION, "card": a.card, "gpu": gpu, "run": a.run,
           "settings": "flock-circuit prove --zk --gpu, one warm process (its timed sessions' end-to-end seconds), every staged "
                       "proof block drawn (k = c); dev: the device's witness, nofw: the host's, fw: the host's and the ZK server's "
                       "send check (FC_FIREWALL=1)", "types": types}
    (out / "overhead.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    (out / "observations.json").write_text(json.dumps(obs, indent=1))
    rows = ["| type | definition | arm | sizes | fixed s | marginal ms | amortized ms | overhead (marginal) | overhead (amortized) | Lean |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    fmt = lambda x, s=1.0, f=".3g": "-" if x is None else format(x * s, f)  # noqa: E731
    pm = lambda f: fmt(f["marginal_s"], 1e3) + ("" if f.get("marginal_se_s") is None else f" ± {fmt(f['marginal_se_s'], 1e3, '.2g')}")  # noqa: E731
    cpu_rows = ["| type | arm | c | prover user s | prover sys s | serve user s | serve sys s | load (1 min) |",
                "|---|---|---:|---:|---:|---:|---:|---:|"]
    for tag, t in types.items():
        for arm, r in t["arms"].items():
            lean = ",".join(lean_outcome(v) for v in r["lean"]) or "-"
            rows.append(f"| {tag} | {t['definition']} | {arm} | {','.join(map(str, t['sizes']))} | {fmt(r['fixed_s'])} | "
                        f"{pm(r)} | {fmt(r['amortized_s'], 1e3)} | {fmt(r.get('overhead_marginal'))} | "
                        f"{fmt(r.get('overhead_amortized'))} | {lean} |")
            for n, f in r["parts"].items():
                rows.append(f"| {tag} | its {n} | {arm} | {','.join(map(str, t['sizes']))} | {fmt(f['fixed_s'])} | {pm(f)} "
                            "| - | - | - | - |")
            for c, m in (r.get("cpu_s") or {}).items():
                cpu_rows.append(f"| {tag} | {arm} | {c} | {m['prover_user_s']:.1f} | {m['prover_sys_s']:.1f} | "
                                f"{m['serve_user_s']:.1f} | {m['serve_sys_s']:.1f} | {m['load1']:.1f} |")
        for c, fits in t["drawn"].items():
            for arm, f in fits.items():
                if f["marginal_s"] is not None:
                    rows.append(f"| {tag} | drawn at c={c} | {arm} | k {','.join(map(str, sorted(f['e2e_s'])))} | "
                                f"{fmt(f['fixed_s'])} | {fmt(f['marginal_s'], 1e3)} | - | - | - | - |")
            if "fw" in fits and "nofw" in fits:
                ks = sorted(set(fits["fw"]["e2e_s"]) & set(fits["nofw"]["e2e_s"]))
                d = {k: fits["fw"]["e2e_s"][k] - fits["nofw"]["e2e_s"][k] for k in ks}
                if ks:
                    rows.append(f"| {tag} | send check at c={c} | fw - nofw | k {','.join(map(str, ks))} | {fmt(d[ks[0]])} | "
                                f"{fmt((d[ks[-1]] - d[ks[0]]) / (ks[-1] - ks[0]) if len(ks) > 1 else None, 1e3)} "
                                "| - | - | - | - |")
    if len(cpu_rows) > 2:
        rows += ["", "Each timed session's CPU seconds (medians), its prover's and its ZK server's process trees:", "", *cpu_rows]
    (out / "overhead.md").write_text(DEFINITION + "\n\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["stage", "prove", "verify", "fit", "overhead"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin")
    ap.add_argument("--lean")
    ap.add_argument("--tmpl-unit-bin", help="prove: flock-tmpl-unit, which places a templated stage's unit rows")
    ap.add_argument("--ks", default="4096,14336")
    ap.add_argument("--units", default="", help="stage: served types (stage.py's TYPE[:S=V,...]), space-separated, instead of --ks")
    ap.add_argument("--sizes", default=None, help="stage: staged units a statement (default 1,8,64,512; with --units half,full)")
    ap.add_argument("--card", default="rtx-pro-6000-server",
                    help="overhead: the card whose datasheet peak divides (verity.hardware.cards); none: a CPU node's sessions")
    ap.add_argument("--draws", default="1,8,64,c", help="stage: the sample sizes a statement is proved at, c meaning all its units")
    ap.add_argument("--points", default=None, help="prove: only these indices of plan.json (one GPU lease a point)")
    ap.add_argument("--arms", default="fw,nofw", help="fw: the ZK server's send check on (the design's path); nofw: off; "
                                                      "dev: off, the device's witness")
    ap.add_argument("--cold", type=int, default=2, help="one-session prover processes a point and arm")
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3, help="the warm process's timed sessions; 0: no warm process")
    ap.add_argument("--passes", type=int, default=1, help="prove: the plan this many times, each family's sizes in ABBA order")
    ap.add_argument("--cpu", action="store_true", help="prove: on the node's CPUs (no --gpu), as the CPU nodes prove a sample")
    ap.add_argument("--run", default=os.environ.get("RESEARCH_RUN_ID"), help="fit: the run id the observations name")
    a = ap.parse_args()
    a.sizes = a.sizes or ("half,full" if a.units else "1,8,64,512")
    if a.command == "verify" and not a.lean or a.command == "stage" and not (a.lean or a.units) \
            or a.command == "prove" and not a.bin:
        ap.error(f"{a.command} needs {'--lean' if a.command != 'prove' else '--bin'}")
    return {"stage": stage, "prove": prove, "verify": verify, "fit": fit, "overhead": overhead}[a.command](a)


if __name__ == "__main__":
    raise SystemExit(main())
