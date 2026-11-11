"""SPOT's proof server's rate on one device: each staged type's full statement proved by K streaming provers on the device
(`verity_flock.proof_server.ProofServer`'s ``provers``), its sessions back to back, each K in turn.

    python benchmarks/one_stage/server_rate.py --bin flock-circuit --staged Z --types frw13,frw2 --provers 1,2,4 --out DIR \\
        --device D [--sessions 24] [--warm 2] [--ahead 2] [--lean flock-verify [--lean-sessions 2]]

* **Statements**: ``Z/<type>/c<n>/stage`` (`stationary_gpu.sh`'s layout), the type's largest n, every unit drawn: a
  sampled replay block at the full statement.
* **Load**: ``--warm`` sessions a prover, all finished before any timed one is submitted (a prover and its verifier load
  the circuit on their first session; ``ready_s``), then ``--sessions`` timed ones (at least 8 K) with K (ahead + 2)
  outstanding.
* **Measured**, for each type and K: statements a second over the steady window, from the K-th timed session's filing by
  its verifier to the last but K's (the clock is the verifier's: a prover reports a session only once it collects its
  verdict, ``verify_ahead`` sessions later), each prover's sessions in it (``idle_provers``: those with none); the device's
  busy share (`nvidia-smi` utilization.gpu every 200 ms, its mean over the window) and its
  memory peak; host memory (each process's RSS every 0.5 s: each prover's peak, the provers' summed peak, the
  verifiers'); the provers' CPU seconds a statement and the node's load over the window; each window session's prover
  timings (``e2e_s`` from Register to the last proof, ``prove_total_s`` its device phases, ``front_s`` the difference).
* **The verdict of record** (``--lean``): at K = 1, the last ``--lean-sessions`` sessions get Lean `flock-verify verify
  --zk` on their record on the server's verdict pool, one at a time and after the steady window, each with its own CPU
  seconds and peak RSS (`os.wait4`), the circuit and the public file archived once a type.

Writes DIR/rate.json (a row a type and K, as each ends) and DIR/<type>-k<K>/ (the server's logs and session records; no
circuit, prover file or proof). A type whose server fails at some K is recorded with its error, and the next goes on.

A row is **void** when any session was refused (a refused session files fast, so it would raise the rate), any of the
server's processes exited nonzero, or a prover's log reports the device out of memory: it has no rate
(``statements_per_s`` is null), ``void`` lists the reasons, and a ``VOID`` line says so. The run exits 1 when any row is
void or failed.
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

from verity_flock.proof_server import ProofServer, Statement

STATEMENT_ID = "verity/flock-circuit"
PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024
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


def staged(z: Path, t: str) -> tuple[Statement, int, Path]:
    d = max((z / t).glob("c*/stage"), key=lambda p: int(p.parent.name[1:]))
    n = json.loads((d / "stage.json").read_text())["n"]
    return Statement(pin=(d / "pin.txt").read_text().strip(), circuit=d / "circuit.txt", tables=d / "tables",
                     instances=d / "inst.bin", public=d / "pub.bin"), n, d


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


def throttled() -> dict:
    """The cgroup's CPU throttling so far (cgroup v2's cpu.stat; none without it)."""
    p = Path("/sys/fs/cgroup/cpu.stat")
    return {k: int(v) for k, v in (x.split() for x in p.read_text().splitlines())} if p.exists() else {}


class Lean:
    """Lean `flock-verify verify --zk` on a session's record, with its own CPU seconds and peak RSS; the circuit, public
    file and tables archived once, in scratch outside ``out`` (`close` removes it)."""

    def __init__(self, lean: str, d: Path, out: Path) -> None:
        self.lean, self.out = lean, out
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
        cmd = [self.lean, "verify", "--statement", STATEMENT_ID, "--archive", str(self.archive),
               "--circuit", self.keys.get("circuit.txt", ""), "--public", self.keys.get("pub.bin", ""), "--zk",
               "--session", str(s.dir)]
        log = self.out / f"lean-{s.label.replace('/', '-')}.log"
        with self._one, open(log, "w") as f:
            t = time.monotonic()
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=f, text=True)
            stdout = p.stdout.read()
            _, status, ru = os.wait4(p.pid, 0)
            p.returncode = os.waitstatus_to_exitcode(status)
            wall = time.monotonic() - t
            f.write(stdout)
        line = next((x for x in stdout.splitlines() if x.startswith("VERDICT ")), None)
        v = json.loads(line.split(" ", 1)[1]) if line else {"accepted": False, "why": f"no verdict (rc {p.returncode})"}
        return {"accepted": v.get("accepted") is True, "why": v.get("why"), "wall_s": round(wall, 2),
                "cpu_s": round(ru.ru_utime + ru.ru_stime, 2), "user_s": round(ru.ru_utime, 2),
                "maxrss_gib": round(ru.ru_maxrss / 2**20, 2)}


def one(a, dev: str, t: str, K: int) -> dict:
    st, n, d = staged(Path(a.staged), t)
    out = Path(a.out) / f"{t}-k{K}"
    out.mkdir(parents=True, exist_ok=True)
    lean = Lean(a.lean, d, out) if a.lean and K == 1 and a.lean_sessions else None
    draw = {"law": "subset", "population": n, "k": n, "units": list(range(n))}
    timed_n = max(a.sessions, 8 * K)
    total, window = K * a.warm + timed_n, K * (a.ahead + 2)
    room, lock, done = threading.Semaphore(window), threading.Lock(), []
    gpu, t_start, cpu0 = Device(dev), time.monotonic(), throttled()
    wall = time.time() - t_start
    rec: dict = {"type": t, "provers": K, "n": n, "pin": st.pin, "verify_ahead": a.ahead, "warm": a.warm,
                 "sessions": timed_n, "device": dev, "stage": json.loads((d / "stage.json").read_text())}
    ps = ProofServer(a.bin, out, devices=[dev], provers=K, verify_ahead=a.ahead, verdicts=1,
                     env={"FC_FIREWALL": "0", "FC_HOST_WITNESS": None, "FC_PHASE_TIMES": "1"})
    host = Host(ps)
    provers_cpu = lambda: cpu_s(pid for name, pid in ps.pids.items() if name.startswith("prover-"))  # noqa: E731

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
        for i in range(K * a.warm, total):
            go(i)
        for w in waits:
            w.join()
        cpu_end, gone = provers_cpu()
        rec["prover_cpu_s_per_statement"] = round((cpu_end - cpu_ready) / timed_n, 2)
        rec["cpu_gone"] = gone
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
    with open(out / "sessions.jsonl", "w") as f:
        for r in done:
            live = {k: v for k, v in (r["s"].live or {}).items() if k not in ("verdict", "buckets", "roots", "unit_draw", "identity")}
            f.write(json.dumps({"i": r["i"], "t": round(r["t"] - t_start, 3),
                                "filed": None if r["filed"] is None else round(r["filed"] - t_start, 3),
                                "prover": r["s"].prover, "error": r["error"], "live": live}, default=str) + "\n")
    with open(out / "device.tsv", "w") as f:
        f.writelines(f"{x - t_start:.3f}\t{u:g}\t{m:g}\n" for x, u, m in gpu.samples)
    if why := void_reasons(len(bad), len(done), rec["exits"], out):
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
                                                              "host", "accepted", "median", "lean", "void", "error")},
                                       default=str), flush=True)
            if r.get("void"):
                print(f"VOID {t} K={K}: " + "; ".join(r["void"]), flush=True)
    if failed := [f"{r['type']} K={r['provers']}" for r in rows if r.get("void") or r.get("error")]:
        print(f"server_rate: no rate for {', '.join(failed)} (void or failed: rate.json says why)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
