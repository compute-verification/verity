"""SPOT's proof server's rate on one device: each staged type's full statement proved by K streaming provers on the device
(`verity_flock.proof_server.ProofServer`'s ``provers``), its sessions back to back, each K in turn.

    python benchmarks/one_stage/server_rate.py --bin flock-circuit --staged Z --types frw13,frw2 --provers 1,2,4 --out DIR \\
        --device D [--sessions 24] [--warm 2] [--ahead 2] [--lean flock-verify [--lean-sessions 2]] [--tmpl-unit-bin B] \\
        [--wide-leaf]

* **Statements**: ``Z/<type>/c<n>/stage`` (`stationary_gpu.sh`'s layout), the type's largest n, every unit drawn: a
  sampled replay block at the full statement. A templated one (``verity/flock-circuit/tmpl`` in its circuit's META, as
  `served_units/stage.py` stages a table read) is proved and verified with its unit's rows at their positions, which
  ``--tmpl-unit-bin`` (`flock-tmpl-unit`) places once into ``stage/unit.placed`` from the circuit's template section.
* **The statement version**: fast100's, or with ``--wide-leaf`` the wide level-0 leaf's (`flock-circuit --wide-leaf`),
  which every prover and verifier of the row then carries; the row says which (``wide_leaf``), and its Lean verdict is
  asked under the version's own statement name (`WIDE_LEAF`). The two versions of a staged statement are two rows,
  each run of the script one version. A row is held to the version its sessions ran: each finished session's
  ``identity.profile`` (`PROFILES`) is counted in the row's ``profile`` beside the one the row asked for, and one that
  differs voids the row, either way. A `flock-circuit` that does not know ``--wide-leaf`` ignores it and proves
  fast100's statement, accepted.
* **Load**: ``--warm`` sessions a prover, all finished before any timed one is submitted (a prover and its verifier load
  the circuit on their first session; ``ready_s``), then ``--sessions`` timed ones (at least 8 K) with K (ahead + 2)
  outstanding.
* **Measured**, for each type and K: statements a second over the steady window, from the K-th timed session's filing by
  its verifier to the last but K's (the clock is the verifier's: a prover reports a session only once it collects its
  verdict, ``verify_ahead`` sessions later), each prover's sessions in it (``idle_provers``: those with none); the device's
  busy share (`nvidia-smi` utilization.gpu every 200 ms, its mean over the window) and its
  memory peak; host memory (each process's RSS every 0.5 s: each prover's peak, the provers' summed peak, the
  verifiers'); the provers' CPU seconds a statement and the node's load over the window; each window session's prover
  timings (``e2e_s`` from Register to the last proof, ``prove_total_s`` its device phases, ``front_s`` the difference); the
  verifier's seconds a statement (``verify_s``: the mean over the window of each session's, from its verifier's record).
* **What prices a row**: ``gpus``, the devices its provers held, so a statement costs ``gpus / statements_per_s``
  GPU-seconds; ``stage.blocks``, the proof blocks of the proved statement, and ``stage.full``, the most one statement of
  its circuit holds (`statement_blocks`); ``flat``, whether the provers' statement caches had stopped filling, with the
  hit series and the rule that decide it (``cache``; `cache_flat`); and ``cores``, the CPUs the row's processes could run
  on: their ``count``, their ``list`` and the ``numa_nodes`` that hold them (`cores`).
* **The verdict of record** (``--lean``): at K = 1, the last ``--lean-sessions`` sessions get Lean `flock-verify verify
  --zk` on their record on the server's verdict pool, one at a time and after the steady window, each with its own CPU
  seconds and peak RSS (`os.wait4`), the circuit and the public file archived once a type. With ``--wide-leaf`` the
  verdict is asked under the version's own name, ``verity/flock-circuit+wide-leaf`` or
  ``verity/flock-circuit/tmpl+wide-leaf``, and the name alone selects the version: `flock-verify` has no
  ``--wide-leaf`` argument, takes a wide-leaf name only with ``--zk`` and with no ``--rec-…`` argument, and refuses
  every other name that ends ``+wide-leaf`` as an unknown statement (exit 2). A `flock-verify` whose `Flock/Tags.lean`
  does not list the two names refuses them the same way, and a row then records that refusal for each session
  (`no_verdict`), not a verdict on the proof.

Writes DIR/rate.json (a row a type and K, as each ends) and DIR/<type>-k<K>/ (the server's logs and session records; no
circuit, prover file or proof). A type whose server fails at some K is recorded with its error, and the next goes on.

``FC_VERIFIER_CPUS`` in the environment (a CPU list within the run's affinity) runs the verifiers on those CPUs and the
provers on the rest (`ProofServer`'s placement). Either way a row keeps it (``verifier_cpus``) and the verifiers' CPU
seconds a statement over the timed sessions beside the provers' (``verifier_cpu_s_per_statement``).

A row is **void** when any session was refused (a refused session files fast, so it would raise the rate), any of the
server's processes exited nonzero, a prover's log reports the device out of memory, or a finished session ran another
statement version than the row's: it has no rate (``statements_per_s`` is null), ``void`` lists the reasons, and a
``VOID`` line says so. The run exits 1 when any row is void or failed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "verity/protocols/flock/prover/python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity.protocols.spot.verifier.one_stage.proofs import verifier_env
from verity_flock import circuit as FC
from verity_flock.proof_server import ProofServer, Statement

STATEMENT_ID = "verity/flock-circuit"
#: what the wide level-0 leaf's version adds to a statement's name, flat or templated, for the Lean verifier
WIDE_LEAF = "+wide-leaf"
#: the profile a session's identity names under each statement version (`flock-circuit`'s ``identity``), by ``--wide-leaf``
PROFILES = {False: "fast100", True: "fast100-wide-leaf"}
#: flock-circuit's statement shape: m = k_log + nbl is at most 35 (`a6.M_MAX`)
M_MAX = 35
#: the rule that decides a row's ``flat``, kept in the row's ``cache``
FLAT_RULE = ("every prover's statement-cache hit rate over the last quarter of its timed sessions is within 0.01 of the "
             "third quarter's")
PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024
#: where the kernel lists each NUMA node's CPUs
NODES = Path("/sys/devices/system/node")
# the CUDA prover prints the runtime's error string ("FFI CUDA error out of memory at ...") before it fails the session
DEVICE_OOM = re.compile(r"out of memory", re.IGNORECASE)


def cpu_s(pids) -> tuple[float, list[int]]:
    """The user and system seconds of ``pids`` so far (`/proc/PID/stat`), and the pids already gone, which count nothing."""
    s, gone = 0, []
    for pid in pids:
        try:
            f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            gone.append(pid)
            continue
        s += int(f[11]) + int(f[12])
    return s / os.sysconf("SC_CLK_TCK"), gone


def tmpl_unit(d: Path, binary: str | None) -> Path | None:
    """A templated statement's unit rows at their positions (``d/unit.placed``), as `a0.templated` places a stager's:
    `flock-tmpl-unit` from the circuit file's template section and META's layout, read off the file's head. None for a
    flat statement."""
    with open(d / "circuit.txt") as f:
        head = [f.readline() for _ in range(3)]
        tag = head[2].split()
        if len(tag) == 2 and tag[0] == FC.TMPL_SECTION and tag[1].isdigit():
            head += [f.readline() for _ in range(int(tag[1]))]
    text = "".join(head)
    meta = FC.meta_of(text)
    if meta.get("statement") != FC.TMPL_STATEMENT:
        return None
    if not binary:
        raise RuntimeError(f"{d}: a templated statement is proved with its unit placed by flock-tmpl-unit: give --tmpl-unit-bin")
    placed = d / "unit.placed"
    if not placed.exists():
        (d / "unit.tmpl").write_text(FC.template_of(text))
        part = d / "unit.placed.part"
        p = subprocess.run([binary, str(d / "unit.tmpl"), "--layout", meta["layout"], "--out", str(part)],
                           capture_output=True, text=True)
        (d / "tmpl-unit.log").write_text(p.stdout + p.stderr)
        if p.returncode:
            raise RuntimeError(f"flock-tmpl-unit: exit {p.returncode} ({d / 'tmpl-unit.log'})")
        part.rename(placed)
    return placed


def staged(z: Path, t: str, tmpl_unit_bin: str | None = None) -> tuple[Statement, int, Path]:
    d = max((z / t).glob("c*/stage"), key=lambda p: int(p.parent.name[1:]))
    n = json.loads((d / "stage.json").read_text())["n"]
    return Statement(pin=(d / "pin.txt").read_text().strip(), circuit=d / "circuit.txt", tables=d / "tables",
                     instances=d / "inst.bin", public=d / "pub.bin", tmpl_unit=tmpl_unit(d, tmpl_unit_bin)), n, d


def statement_blocks(stage: dict) -> dict:
    """The proof blocks of a staged statement (its ``n`` instances) and the most one statement of its circuit holds, by
    the bound `a6.py` chunks a population at (``vus_per_block`` instances in each of 2^(M_MAX - k_log) blocks; None
    without the circuit's shape)."""
    k, g = stage.get("k_log"), stage.get("vus_per_block") or 1
    return {"blocks": stage["n"], "full": g << (M_MAX - k) if isinstance(k, int) and k <= M_MAX else None}


def cache_flat(sessions: list[tuple[str, bool, dict | None]]) -> tuple[bool, dict]:
    """Whether the provers' statement caches had stopped filling over the timed sessions (`FLAT_RULE`), and the hit series
    that says so. ``sessions``: each session's prover, whether it was a warm one, and its record's ``stmt_cache``: the
    prover's lookups so far and the sampled statements it holds, each built once, so a statement held is a miss.

    A prover's timed sessions, in the order of its lookups, are cut into four quarters. A quarter's hit rate is
    1 - held / lookups over what it added to the quarter before it (the first: to the prover's last warm session); a record's
    counters are read when it is emitted, after the lookups of the sessions built ahead of it, so a quarter's lookups are
    its sessions' only roughly. A prover with under four timed sessions, without the counters, or with no lookup in either
    of its last two quarters isn't flat."""
    by: dict[str, dict[bool, list[tuple[int, int]]]] = {}
    for prover, warm, c in sessions:
        mine = by.setdefault(prover, {True: [], False: []})
        if c and isinstance(c.get("lookups"), int) and isinstance(c.get("held"), int):
            mine[warm].append((c["lookups"], c["held"]))
    series, flat = {}, bool(by)
    for prover, mine in sorted(by.items()):
        timed, start = sorted(mine[False]), max(mine[True], default=(0, 0))
        ends = [timed[len(timed) * q // 4 - 1] for q in range(1, 5)] if len(timed) >= 4 else []
        added = [(e[0] - p[0], e[1] - p[1]) for p, e in zip([start, *ends], ends)]
        series[prover] = {"warm": {"lookups": start[0], "held": start[1]}, "timed": len(timed), "quarters": [
            {"lookups": n, "held": h, "hit_rate": round(1 - h / n, 4) if n > 0 else None} for n, h in added]}
        # |h3 / n3 - h4 / n4| <= 0.01, in integers
        flat = flat and len(added) == 4 and added[2][0] > 0 and added[3][0] > 0 and (
            100 * abs(added[2][1] * added[3][0] - added[3][1] * added[2][0]) <= added[2][0] * added[3][0])
    return flat, {"rule": FLAT_RULE, "provers": series}


class Device:
    """The device's utilization and memory every ``ms`` milliseconds, each sample with its monotonic time."""

    def __init__(self, dev: str, ms: int = 200) -> None:
        self.samples: list[tuple[float, float, float]] = []
        self.unparsed: list[str] = []
        self._p = subprocess.Popen(["nvidia-smi", "-i", dev, "--query-gpu=utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits", f"--loop-ms={ms}"], stdout=subprocess.PIPE, text=True)
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()

    def _read(self) -> None:
        for line in self._p.stdout:
            try:
                u, m = (float(x) for x in line.split(","))
            except ValueError:
                self.unparsed.append(line.strip())
                continue
            self.samples.append((time.monotonic(), u, m))

    def stop(self) -> None:
        self._p.terminate()
        self._p.wait()
        self._t.join()

    def busy(self, t0: float, t1: float) -> float | None:
        u = [s[1] for s in self.samples if t0 <= s[0] <= t1]
        return round(statistics.fmean(u) / 100, 4) if u else None

    def peak_mib(self) -> float | None:
        return max((s[2] for s in self.samples), default=None)


class Host:
    """Each of a server's processes' RSS every ``every`` seconds: each one's peak, and the peaks of the provers' and the
    verifiers' sums."""

    def __init__(self, ps: ProofServer, every: float = 0.5) -> None:
        self.ps, self.peak, self.provers_kb, self.verifiers_kb, self.gone = ps, {}, 0, 0, set()
        self._stop = threading.Event()
        self._every = every
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            sums = {"prover": 0, "verifier": 0}
            for name, pid in self.ps.pids.items():
                try:
                    kb = int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * PAGE_KB
                except FileNotFoundError:
                    self.gone.add(name)
                    continue
                self.peak[name] = max(self.peak.get(name, 0), kb)
                sums[name.split("-", 1)[0]] = sums.get(name.split("-", 1)[0], 0) + kb
            self.provers_kb = max(self.provers_kb, sums["prover"])
            self.verifiers_kb = max(self.verifiers_kb, sums["verifier"])
            self._stop.wait(self._every)

    def stop(self) -> dict:
        self._stop.set()
        self._t.join()
        gib = lambda kb: round(kb / 2**20, 2)  # noqa: E731
        return {"provers_gib": gib(self.provers_kb), "verifiers_gib": gib(self.verifiers_kb),
                "each_gib": {k: gib(v) for k, v in sorted(self.peak.items())}, "exited_while_sampled": sorted(self.gone)}


def void_reasons(refused: int, sessions: int, exits: dict[str, int], out: Path) -> list[str]:
    """Why a row's rate can't stand (empty when it can): a refused session, a server process that exited nonzero, or a
    prover whose log (``out/prover-*.log``) reports the device out of memory."""
    why = []
    if refused:
        why.append(f"{refused} of {sessions} sessions refused")
    if bad := sorted(f"{k} rc {v}" for k, v in exits.items() if v != 0):
        why.append("exited nonzero: " + ", ".join(bad))
    if oom := sorted(p.stem for p in out.glob("prover-*.log") if DEVICE_OOM.search(p.read_text(errors="replace"))):
        why.append("out of device memory: " + ", ".join(oom))
    return why


def version_reasons(asked: str, ran: dict[str, int]) -> list[str]:
    """Why a row's sessions are not the row's statement version (empty when they are): ``ran`` counts the finished
    sessions by the profile their identity names ("none" for a record that names none), ``asked`` is the row's. It holds
    both ways: a wide-leaf row over fast100 sessions, and a fast100 row over wide-leaf ones."""
    other = {p: c for p, c in ran.items() if p != asked}
    if not other:
        return []
    why = (f"{sum(other.values())} of {sum(ran.values())} finished sessions ran another statement version than the row's "
           f"{asked!r}: " + ", ".join(f"{c} at {p!r}" for p, c in sorted(other.items())))
    if asked == PROFILES[True] and PROFILES[False] in other:
        why += "; a flock-circuit that does not know --wide-leaf ignores it: build the prover from a tree whose flock-circuit has it"
    return [why]


def throttled() -> dict:
    """The cgroup's CPU throttling so far (cgroup v2's cpu.stat; none without it)."""
    p = Path("/sys/fs/cgroup/cpu.stat")
    return {k: int(v) for k, v in (x.split() for x in p.read_text().splitlines())} if p.exists() else {}


def cpu_list(cpus) -> str:
    """CPUs as the kernel writes a list: in order, neighbours as a range ("72-81,84")."""
    runs: list[list[int]] = []
    for c in sorted(cpus):
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def cores(cpus=None, nodes: Path | None = None) -> dict:
    """The CPUs a row's processes can run on: this process's own (its affinity; every process of the row runs within
    it: unplaced, each inherits it; with ``FC_VERIFIER_CPUS``, the verifiers run on ``verifier_cpus`` and the provers on
    the rest), as their count, their list and the NUMA nodes that hold any of them (`NODES`' ``nodeN/cpulist``; none on a
    machine that lists no nodes)."""
    mine, held = set(os.sched_getaffinity(0) if cpus is None else cpus), []
    for f in (nodes or NODES).glob("node[0-9]*/cpulist"):
        parts = f.read_text().strip().split(",")
        if any(mine & set(range(int(p.split("-")[0]), int(p.split("-")[-1]) + 1)) for p in parts if p):
            held.append(int(f.parent.name[4:]))
    return {"count": len(mine), "list": cpu_list(mine), "numa_nodes": sorted(held)}


def no_verdict(statement: str, rc: int, said: str) -> str:
    """Why a `flock-verify verify` that printed no VERDICT line gave none, from its exit code and what it wrote
    (``said``). A statement name it does not know is said as that: the refusal is of the name, before any proof is read,
    and a wide-leaf name is known only to a `flock-verify` whose `Flock/Tags.lean` lists it."""
    if "unknown statement" not in said:
        return f"no verdict (rc {rc})"
    why = f"no verdict (rc {rc}): this flock-verify does not know the statement {statement!r} and refused it unread"
    if statement.endswith(WIDE_LEAF):
        why += ("; the wide level-0 leaf's names are known only to a flock-verify built from a tree whose "
                "Flock/Tags.lean lists them, so this says nothing of the proof")
    return why


class Lean:
    """Lean `flock-verify verify --zk` on a session's record, with its own CPU seconds and peak RSS; the circuit, public
    file and tables archived once, in scratch outside ``out`` (`close` removes it); ``statement``, the one `verify` sets
    up. `verify` runs in `verifier_env`: under the verifier's allocator setting unless this process's environment sets
    its own, which both the seconds and the peak read."""

    def __init__(self, lean: str, d: Path, out: Path, statement: str = STATEMENT_ID) -> None:
        self.lean, self.out, self.statement = lean, out, statement
        self.archive = Path(tempfile.mkdtemp(prefix="server-rate-archive-"))
        tables = sorted(str(f) for f in (d / "tables").glob("*") if f.is_file()) if (d / "tables").is_dir() else []
        p = subprocess.run([lean, "archive-put", str(self.archive), str(d / "circuit.txt"), str(d / "pub.bin"), *tables],
                           capture_output=True, text=True)
        (out / "archive-put.log").write_text(p.stdout + p.stderr)
        self.keys = {Path(x.split()[1]).name: x.split()[0] for x in p.stdout.splitlines() if len(x.split()) == 2}
        self._one = threading.Lock()

    def close(self) -> None:
        shutil.rmtree(self.archive, ignore_errors=True)

    def __call__(self, s) -> dict:
        cmd = [self.lean, "verify", "--statement", self.statement, "--archive", str(self.archive),
               "--circuit", self.keys.get("circuit.txt", ""), "--public", self.keys.get("pub.bin", ""), "--zk",
               "--session", str(s.dir)]
        log = self.out / f"lean-{s.label.replace('/', '-')}.log"
        with self._one, open(log, "w") as f:
            t = time.monotonic()
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=f, text=True, env=verifier_env())
            stdout = p.stdout.read()
            _, status, ru = os.wait4(p.pid, 0)
            p.returncode = os.waitstatus_to_exitcode(status)
            wall = time.monotonic() - t
            f.write(stdout)
        line = next((x for x in stdout.splitlines() if x.startswith("VERDICT ")), None)
        v = (json.loads(line.split(" ", 1)[1]) if line
             else {"accepted": False, "why": no_verdict(self.statement, p.returncode, log.read_text(errors="replace"))})
        return {"accepted": v.get("accepted") is True, "why": v.get("why"), "wall_s": round(wall, 2),
                "cpu_s": round(ru.ru_utime + ru.ru_stime, 2), "user_s": round(ru.ru_utime, 2),
                "maxrss_gib": round(ru.ru_maxrss / 2**20, 2)}


def one(a, dev: str, t: str, K: int) -> dict:
    st, n, d = staged(Path(a.staged), t, a.tmpl_unit_bin)
    out = Path(a.out) / f"{t}-k{K}"
    out.mkdir(parents=True, exist_ok=True)
    statement = (STATEMENT_ID if st.tmpl_unit is None else FC.TMPL_STATEMENT) + (WIDE_LEAF if a.wide_leaf else "")
    lean = Lean(a.lean, d, out, statement) if a.lean and K == 1 and a.lean_sessions else None
    draw = {"law": "subset", "population": n, "k": n, "units": list(range(n))}
    timed_n = max(a.sessions, 8 * K)
    total, window = K * a.warm + timed_n, K * (a.ahead + 2)
    room, lock, done = threading.Semaphore(window), threading.Lock(), []
    gpu, t_start, cpu0 = Device(dev), time.monotonic(), throttled()
    wall = time.time() - t_start
    devices, stage = [dev], json.loads((d / "stage.json").read_text())
    rec: dict = {"type": t, "provers": K, "wide_leaf": a.wide_leaf, "gpus": len(devices), "cores": cores(), "n": n, "pin": st.pin,
                 "verify_ahead": a.ahead,
                 "warm": a.warm, "sessions": timed_n, "device": dev, "stage": stage | statement_blocks(stage)}
    ps = ProofServer(a.bin, out, devices=devices, provers=K, verify_ahead=a.ahead, verdicts=1, wide_leaf=a.wide_leaf,
                     env={"FC_FIREWALL": "0", "FC_HOST_WITNESS": None, "FC_PHASE_TIMES": "1"})
    host = Host(ps)
    provers_cpu = lambda: cpu_s(pid for name, pid in ps.pids.items() if name.startswith("prover-"))  # noqa: E731
    verifiers_cpu = lambda: cpu_s(pid for name, pid in ps.pids.items() if name.startswith("verifier-"))  # noqa: E731

    def wait(s, i: int) -> None:
        try:
            s.result()
            err = None
        except Exception as e:  # noqa: BLE001 -- recorded with the session, the rest go on
            err = str(e)[:600]
        with lock:
            done.append({"t": time.monotonic(), "i": i, "s": s, "error": err, "load1": os.getloadavg()[0]})
        room.release()

    waits, judged = [], []

    def go(i: int) -> None:
        room.acquire()
        fn = lean if lean is not None and i >= total - a.lean_sessions else None
        s = ps.submit(st, draw, f"{t}/k{K}/{i}", verdict=fn)
        if fn is not None:
            judged.append(s)
        waits.append(threading.Thread(target=wait, args=(s, i), daemon=True))
        waits[-1].start()

    try:
        # every prover and its verifier load the circuit and finish their warm sessions before any timed one is
        # submitted, so the window never times a prover still loading
        for i in range(K * a.warm):
            go(i)
        for w in waits:
            w.join()
        rec["ready_s"] = round(time.monotonic() - t_start, 1)
        cpu_ready, _ = provers_cpu()
        vcpu_ready, _ = verifiers_cpu()
        for i in range(K * a.warm, total):
            go(i)
        for w in waits:
            w.join()
        cpu_end, gone = provers_cpu()
        vcpu_end, vgone = verifiers_cpu()
        rec["prover_cpu_s_per_statement"] = round((cpu_end - cpu_ready) / timed_n, 2)
        rec["cpu_gone"] = gone
        rec["verifier_cpus"] = ps.verifier_cpus
        rec["verifier_cpu_s_per_statement"] = round((vcpu_end - vcpu_ready) / timed_n, 2)
        rec["verifier_cpu_gone"] = vgone
    finally:
        rec["exits"] = ps.close()
        rec["host"] = host.stop()
        gpu.stop()
    rec["wall_s"] = round(time.monotonic() - t_start, 1)
    rec["smi_unparsed"] = gpu.unparsed[:8]
    done.sort(key=lambda r: r["t"])
    bad = [r for r in done if r["error"] or not r["s"].accepted]
    rec["accepted"] = len(done) - len(bad)
    # the first 8 refused sessions; refused_n counts them all
    rec["refused"] = [{"label": r["s"].label, "error": r["error"], "why": (r["s"].live or {}).get("why")} for r in bad][:8]
    rec["refused_n"] = len(bad)
    rec["warm_per_prover"] = dict(Counter(r["s"].prover for r in done if r["i"] < K * a.warm))
    # the clock is each session's filing by its verifier (its record directory's mtime): a prover reports a session
    # only once it collects its verdict, ``verify_ahead`` sessions later, and its last ones all at once
    for r in done:
        r["filed"] = r["s"].dir.stat().st_mtime - wall if r["s"].dir is not None and r["s"].dir.exists() else None
    timed = sorted((r for r in done if r["i"] >= K * a.warm and r["filed"] is not None), key=lambda r: r["filed"])
    rec["unfiled"] = sum(1 for r in done if r["filed"] is None)
    i0, i1 = K - 1, len(timed) - 1 - K
    if i1 - i0 < 2:
        i0, i1 = 0, len(timed) - 1
    w0, w1 = timed[i0], timed[i1]
    span = w1["filed"] - w0["filed"]
    win = [r for r in timed[i0 + 1:i1 + 1] if r["s"].live]
    med = lambda k: round(statistics.median(v), 4) if (v := [r["s"].live.get(k) for r in win  # noqa: E731
                                                             if isinstance(r["s"].live.get(k), (int, float))]) else None
    rec |= {"statements_per_s": round((i1 - i0) / span, 4) if span > 0 else None, "window_s": round(span, 2),
            "window_sessions": i1 - i0, "busy": gpu.busy(w0["filed"], w1["filed"]), "device_peak_mib": gpu.peak_mib(),
            "load1": [round(w0["load1"], 1), round(w1["load1"], 1)],
            "throttled": {k: throttled().get(k, 0) - cpu0.get(k, 0) for k in ("nr_throttled", "throttled_usec")},
            "median": {k: med(k) for k in ("e2e_s", "prove_total_s", "witness_s", "wait_s", "verdict_wait_s", "tp_s")},
            "prebuilt_used": sum(1 for r in win if r["s"].live.get("witness_prebuilt_s") is not None),
            "per_prover": dict(sorted(Counter(r["s"].prover for r in win).items()))}
    rec["idle_provers"] = K - len(rec["per_prover"])
    if verify := [v for r in win if isinstance(v := (r["s"].served or {}).get("verify_s"), (int, float))]:
        rec["verify_s"] = round(statistics.fmean(verify), 4)
    rec["flat"], rec["cache"] = cache_flat([(r["s"].prover, r["i"] < K * a.warm, (r["s"].live or {}).get("stmt_cache"))
                                            for r in done])
    if rec["median"]["e2e_s"] is not None and rec["median"]["prove_total_s"] is not None:
        rec["median"]["front_s"] = round(rec["median"]["e2e_s"] - rec["median"]["prove_total_s"], 4)
    if lean is not None:
        rec["lean"] = []
        for s in judged:
            try:
                rec["lean"].append({"label": s.label, **s.verdict()})
            except Exception as e:  # noqa: BLE001 -- a failed verdict is recorded, not raised
                rec["lean"].append({"label": s.label, "error": str(e)[:400]})
        lean.close()
    # read before the records are thinned below: what each finished session says it ran
    ran = Counter(str(((r["s"].live or {}).get("identity") or {}).get("profile") or "none") for r in done if r["s"].live)
    rec["profile"] = {"asked": PROFILES[a.wide_leaf], "ran": dict(sorted(ran.items()))}
    with open(out / "sessions.jsonl", "w") as f:
        for r in done:
            live = {k: v for k, v in (r["s"].live or {}).items() if k not in ("verdict", "buckets", "roots", "unit_draw", "identity")}
            f.write(json.dumps({"i": r["i"], "t": round(r["t"] - t_start, 3),
                                "filed": None if r["filed"] is None else round(r["filed"] - t_start, 3),
                                "prover": r["s"].prover, "error": r["error"], "live": live}, default=str) + "\n")
    with open(out / "device.tsv", "w") as f:
        f.writelines(f"{x - t_start:.3f}\t{u:g}\t{m:g}\n" for x, u, m in gpu.samples)
    if why := void_reasons(len(bad), len(done), rec["exits"], out) + version_reasons(rec["profile"]["asked"], rec["profile"]["ran"]):
        rec |= {"void": why, "statements_per_s": None}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bin", required=True)
    ap.add_argument("--staged", required=True)
    ap.add_argument("--types", required=True, help="comma-separated staged type directories, each run at every K")
    ap.add_argument("--provers", default="1,2,4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", required=True, help="the GPU's index (gpu-lease's CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--sessions", type=int, default=24, help="timed sessions a K (at least 8 K)")
    ap.add_argument("--warm", type=int, default=2, help="warm sessions a prover")
    ap.add_argument("--ahead", type=int, default=2)
    ap.add_argument("--lean", default=None)
    ap.add_argument("--lean-sessions", type=int, default=2)
    ap.add_argument("--tmpl-unit-bin", default=None, help="flock-tmpl-unit: places a templated statement's unit")
    ap.add_argument("--wide-leaf", action="store_true",
                    help="the wide level-0 leaf's statement version: every prover and verifier runs with --wide-leaf")
    ap.add_argument("--deadline", type=float, default=0, help="UTC epoch seconds after which no type and K starts")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in a.types.split(","):
        for K in (int(k) for k in a.provers.split(",")):
            if a.deadline and time.time() > a.deadline:
                rows.append({"type": t, "provers": K, "skipped": "past the deadline"})
                continue
            try:
                r = one(a, a.device, t, K)
            except Exception as e:  # noqa: BLE001 -- the next type and K go on
                r = {"type": t, "provers": K, "error": f"{type(e).__name__}: {str(e)[:800]}"}
            rows.append(r)
            (out / "rate.json").write_text(json.dumps(rows, indent=1, default=str))
            print("RATE " + json.dumps({k: r.get(k) for k in ("type", "provers", "statements_per_s", "busy", "device_peak_mib",
                                                              "host", "accepted", "flat", "median", "lean", "void", "error")},
                                       default=str), flush=True)
            if r.get("void"):
                print(f"VOID {t} K={K}: " + "; ".join(r["void"]), flush=True)
    if failed := [f"{r['type']} K={r['provers']}" for r in rows if r.get("void") or r.get("error")]:
        print(f"server_rate: no rate for {', '.join(failed)} (void or failed: rate.json says why)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
