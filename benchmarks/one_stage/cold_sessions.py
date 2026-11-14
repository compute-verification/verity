"""One cold session a member of a kept window, behind the ZK server, on a device or on the CPU: each session's seconds by
part, its peak memory and its Lean verdict, or the text it was refused with.

    python benchmarks/one_stage/cold_sessions.py plan --window W --members K,K,...
        # each member's statement from the window's files (``summary.json``'s members, the public file's header), or exit 1
        # naming what is missing
    python benchmarks/one_stage/cold_sessions.py prove --window W --interface F --bin B --zk-server L --member K
        --arm gpu|cpu [--device D --card UUID] --out O [--timeout S]
        # one session on a proof server of its own: O/<arm>/session-m<k>/cold.json and a COLD line
    python benchmarks/one_stage/cold_sessions.py verify --window W --interface F --lean V --member K --arm A --out O
        # Lean's verdict on that session (`a0.lean_verify`): O/<arm>/session-m<k>/verdict.json and a VERDICT-COLD line
    python benchmarks/one_stage/cold_sessions.py table --out O
        # O/table.json and O/table.md, a row a member

A window is an audit's ``audit/window`` (`a5.audit`'s): ``m<k>/`` (``circuit.txt``, ``inst-N.bin``, ``pub-N.bin``,
``registered.json`` where the member reads registered values), each session's sample (``session-m<k>/``, `a0.SAMPLE_FILE`), ``partition.json``,
``program.json`` and ``summary.json``; ``--interface`` is the window's registered circuit record. Each session is the one
`a5.proof_server` proves an audit's member in, behind the ZK server: a `ProofServer` of its own (so its prover and its
verifier start cold), ``verify_ahead=0``, its Lean session loop set up as `a0.prover_env` says, on the member's own sample
from the audit; on ``--arm gpu`` its prover is ``prove --zk --gpu`` on ``--device``. Its records are filed as the audit's
(``session-m<k>/sessions``), its processes' logs in ``session-m<k>/proof-server``.

Parts of a session (``cold.json``): ``submitted_to_records_s`` (submission to both sides' records, `Session.seconds`); each
side's ``statement ... built in`` (``prover_built_s``, ``verifier_built_s``: the circuit's load and the statement's build);
the prover's ``LIVE`` record (``e2e_s``, ``witness_s``, ``prove_total_s``, ``prove_s``, ``wait_e2e_s``) and ``outside_s``,
``e2e_s`` less the witness, the proving and the wait on the verifier: the ZK server's part of the clock. With the
stamping wrappers of `member_seconds.wrap` as the binaries, the stamps give each process's start, so ``loop_start_s``
(the prover's start to its Lean loop's) and ``before_clock_s`` (the loop's start to the clock's) are read from them.
Memory: ``host_peak_gib``, the summed RSS of every process of the server's trees (the provers', the verifiers' and the
Lean loops'), sampled every half second, and each tree's; ``device_peak_mib``, `server_rate.Device`'s on ``--card``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a0  # noqa: E402
import member_seconds as MS  # noqa: E402
import server_rate as SR  # noqa: E402
from verity_flock.proof_server import ProofServer, ProofServerError, Statement  # noqa: E402

LIVE_KEYS = ("e2e_s", "witness_s", "prove_total_s", "prove_s", "wait_s", "wait_e2e_s", "k_log", "dense_m", "accepted", "why")


def members(window: Path) -> dict[int, dict]:
    return {m["k"]: m for m in json.loads((window / "summary.json").read_text())["members"]}


def statement(window: Path, interface: Path, k: int) -> dict:
    """Member ``k``'s statement as `a5.audit` hands it to `a0.lean_verify` and `a0.prover_env`."""
    d = window / f"m{k}"
    reg = d / "registered.json"
    return {"dir": d, "partition_file": window / "partition.json", "program_file": window / "program.json",
            **({"registered_file": reg} if reg.exists() else {}), "interface_file": interface, "archived": {}}


def plan(a: argparse.Namespace) -> int:
    window, rows, bad = Path(a.window), members(Path(a.window)), []
    for k in a.members:
        m, d = rows.get(k), window / f"m{k}"
        if m is None:
            bad.append(f"m{k}: not in summary.json")
            continue
        need = [d / "circuit.txt", d / f"inst-{m['n']}.bin", d / f"pub-{m['n']}.bin", window / f"session-m{k}" / a0.SAMPLE_FILE]
        if missing := [str(f) for f in need if not f.exists()]:
            bad.append(f"m{k}: missing {missing}")
            continue
        pin = a0.public_header(d, m["n"]).get("circuit_sha512")
        print("PLAN " + json.dumps({"k": k, "n": m["n"], "template": m["template"], "pin": pin,
                                    "circuit_bytes": (d / "circuit.txt").stat().st_size,
                                    "registered": (d / "registered.json").exists(),
                                    "sample": json.loads(need[3].read_text())}), flush=True)
    for f in (window / "partition.json", window / "program.json"):
        if not f.exists():
            bad.append(f"missing {f}")
    for b in bad:
        print(f"PLAN-MISSING {b}", flush=True)
    return 1 if bad else 0


def processes() -> dict[int, tuple[int, int]]:
    """Every process's parent and RSS in KiB, from one `ps`."""
    out = {}
    for line in subprocess.run(["ps", "-e", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout.splitlines():
        pid, ppid, rss = (int(x) for x in line.split())
        out[pid] = (ppid, rss)
    return out


def tree_rss(roots: dict[int, str]) -> dict[str, int]:
    """The summed RSS in KiB of each root's tree (itself and every descendant), by the root's name."""
    procs, kids = processes(), {}
    for pid, (ppid, _) in procs.items():
        kids.setdefault(ppid, []).append(pid)
    sums = {}
    for root, name in roots.items():
        todo, seen = [root], set()
        while todo:
            p = todo.pop()
            if p in seen or p not in procs:
                continue
            seen.add(p)
            sums[name] = sums.get(name, 0) + procs[p][1]
            todo += kids.get(p, [])
    return sums


class Trees:
    """The summed RSS of the server's process trees every ``every`` seconds: the peak of the whole and of each tree."""

    def __init__(self, ps: ProofServer, every: float = 0.5) -> None:
        self.ps, self.every, self.peak_kb, self.by_tree = ps, every, 0, {}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            sums = tree_rss({pid: name for name, pid in self.ps.pids.items()})
            self.peak_kb = max(self.peak_kb, sum(sums.values()))
            for n, kb in sums.items():
                self.by_tree[n] = max(self.by_tree.get(n, 0), kb)

    def stop(self) -> dict:
        self._stop.set()
        self._t.join()
        return {"host_peak_gib": round(self.peak_kb / 2**20, 2),
                "host_peak_gib_by_tree": {n: round(kb / 2**20, 2) for n, kb in sorted(self.by_tree.items())}}


def built(log: Path) -> float | None:
    m = MS.BUILT.search(log.read_text(errors="replace")) if log.exists() else None
    return float(m.group(1)) if m else None


def stamped(stamps: Path, prover_pid: int | None) -> dict:
    """From the wrappers' stamps: the prover's start, its Lean loop's (a loop whose parent is the prover)."""
    if prover_pid is None or not stamps.exists():
        return {}
    st = MS.stamps(stamps.read_text())
    prover = next((s for s in st if s["pid"] == prover_pid), None)
    loop = next((s for s in st if s["program"] == "loop" and s["ppid"] == prover_pid), None)
    return {"prover_start_ns": prover and prover["ns"], "loop_start_ns": loop and loop["ns"]}


def session_dir(out: Path, arm: str, k: int) -> Path:
    return out / arm / f"session-m{k}"


def prove(a: argparse.Namespace) -> int:
    window, k = Path(a.window), a.member
    m = members(window)[k]
    st = statement(window, Path(a.interface), k)
    sdir = session_dir(Path(a.out), a.arm, k)
    sdir.mkdir(parents=True, exist_ok=True)
    sample = json.loads((window / f"session-m{k}" / a0.SAMPLE_FILE).read_text())
    (sdir / a0.SAMPLE_FILE).write_text(json.dumps(sample))
    env = {**a0.prover_env(a.zk_server, st), "FC_PHASE_TIMES": "1"}
    devices = [a.device] if a.arm == "gpu" else None
    ps = ProofServer(a.bin, sdir / "proof-server", devices=devices, env=env, verify_ahead=0, verdicts=1,
                     challenger=a0.own_challenger(zk=True))

    def stop(sig, _frame) -> None:
        # the server's processes run in sessions of their own: a lease's SIGTERM to this one alone would leave them running
        for pid in ps.pids.values():
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                print(f"cold_sessions: pid {pid} had already exited at signal {sig}", file=sys.stderr, flush=True)
        os._exit(128 + sig)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(int(a.timeout))
    card = SR.Device(a.card) if a.card else None
    host = Trees(ps)
    rec = {"k": k, "n": m["n"], "template": m["template"], "arm": a.arm, "device": a.device if devices else None,
           "card": a.card, "circuit_bytes": (st["dir"] / "circuit.txt").stat().st_size, "sample": sample,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.time()
    pin = a0.public_header(st["dir"], m["n"])["circuit_sha512"]
    s = ps.submit(Statement(pin=pin, circuit=st["dir"] / "circuit.txt", tables=st["dir"] / "tables",
                            instances=st["dir"] / f"inst-{m['n']}.bin", public=st["dir"] / f"pub-{m['n']}.bin"),
                  sample, str(sdir.resolve()), out=sdir / "sessions")
    try:
        s.result()
        rec["error"] = None
    except ProofServerError as e:
        rec["error"] = str(e)[-4000:]
    rec["session_wall_s"] = round(time.time() - t0, 1)
    pids = ps.pids
    rec["exits"] = ps.close()
    signal.alarm(0)
    rec |= host.stop()
    if card is not None:
        card.stop()
        rec["device_peak_mib"] = card.peak_mib()
    rec["submitted_to_records_s"] = s.seconds
    live = s.live or {}
    rec["live"] = {x: live.get(x) for x in LIVE_KEYS if x in live}
    rec["served"] = {x: (s.served or {}).get(x) for x in ("accepted", "why", "session") if x in (s.served or {})}
    rec["accepted"] = s.accepted and rec["error"] is None
    rec["refusal"] = None if rec["accepted"] else (rec["error"] or live.get("why") or (s.served or {}).get("why"))
    logs = {p.stem.split(".")[0]: p for p in (sdir / "proof-server").glob("*.log")}
    rec["prover_built_s"] = next((built(p) for n, p in logs.items() if n.startswith("prover")), None)
    rec["verifier_built_s"] = next((built(p) for n, p in logs.items() if n.startswith("verifier")), None)
    rec["outside_s"] = outside(live)
    prover_pid = next((pid for n, pid in pids.items() if n.startswith("prover")), None)
    sp = stamped(Path(a.stamps), prover_pid) if a.stamps else {}
    if sp.get("prover_start_ns") and sp.get("loop_start_ns"):
        rec["loop_start_s"] = round((sp["loop_start_ns"] - sp["prover_start_ns"]) / 1e9, 3)
        if isinstance(live.get("e2e_s"), (int, float)) and s.seconds is not None:
            # the clock's start: the records' time less e2e and the prover's wait after its clock
            records_ns = (t0 + s.seconds) * 1e9
            rec["before_clock_s"] = round((records_ns - live["e2e_s"] * 1e9 - sp["loop_start_ns"]) / 1e9, 3)
            rec["prover_start_after_submit_s"] = round(sp["prover_start_ns"] / 1e9 - t0, 3)
    (sdir / "cold.json").write_text(json.dumps(rec, indent=1))
    print("COLD " + json.dumps(rec), flush=True)
    return 0


def outside(live: dict) -> float | None:
    """``e2e_s`` less the witness, the proving and the wait on the verifier inside the clock (`session_split.rest`)."""
    if not all(isinstance(live.get(k), (int, float)) for k in ("e2e_s", "prove_total_s", "wait_e2e_s")):
        return None
    return round(live["e2e_s"] - (live.get("witness_s") or 0.0) - live["prove_total_s"] - live["wait_e2e_s"], 3)


def verify(a: argparse.Namespace) -> int:
    window, k = Path(a.window), a.member
    m = members(window)[k]
    st = statement(window, Path(a.interface), k)
    sdir = session_dir(Path(a.out), a.arm, k)
    t0 = time.time()
    v = a0.lean_verify(a.lean, st, m["n"], sdir, zk=True)
    v = {"k": k, "arm": a.arm, "verdict_wall_s": round(time.time() - t0, 1),
         "lean_peak_gib": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 2), **v}
    (sdir / "verdict.json").write_text(json.dumps(v, indent=1))
    print("VERDICT-COLD " + json.dumps(v), flush=True)
    return 0


def table(a: argparse.Namespace) -> int:
    out, rows = Path(a.out), {}
    for f in sorted(out.glob("*/session-m*/cold.json")):
        r = json.loads(f.read_text())
        v = f.with_name("verdict.json")
        r["verdict"] = json.loads(v.read_text()) if v.exists() else None
        rows.setdefault(r["k"], {})[r["arm"]] = r
    (out / "table.json").write_text(json.dumps(rows, indent=1))

    def cell(r: dict | None) -> str:
        if r is None:
            return "not run"
        if not r["accepted"]:
            return "refused: " + re.sub(r"\s+", " ", str(r["refusal"]))[:160].replace("|", "/")
        lv, v = r["live"], r["verdict"] or {}
        verdict = (f"verdict {v['verdict_wall_s']} s {'accepted' if v.get('accepted') else 'REFUSED'}"
                   if v else "no verdict")
        return (f"wall {r['session_wall_s']} s; built P {r['prover_built_s']} V {r['verifier_built_s']}; loop start "
                f"{r.get('loop_start_s')}; before clock {r.get('before_clock_s')}; e2e {lv.get('e2e_s')} (witness "
                f"{lv.get('witness_s')}, proving {lv.get('prove_total_s')}, wait {lv.get('wait_e2e_s')}, outside "
                f"{r.get('outside_s')}); {verdict}; host {r['host_peak_gib']} GiB"
                + (f"; card {r.get('device_peak_mib')} MiB" if r["arm"] == "gpu" else ""))
    lines = ["| member | template | circuit bytes | GPU session | CPU session |", "|---|---|---|---|---|"]
    for k in sorted(rows):
        g, c = rows[k].get("gpu"), rows[k].get("cpu")
        any_ = g or c
        lines.append(f"| m{k} | `{any_['template'].split('{')[0]}` | {any_['circuit_bytes']:,} | {cell(g)} | {cell(c)} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def ks(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--window", required=True)
    p.add_argument("--members", type=ks, required=True)
    for name in ("prove", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--window", required=True)
        p.add_argument("--interface", required=True, help="the window's registered circuit record (verify's --interface)")
        p.add_argument("--member", type=int, required=True)
        p.add_argument("--arm", choices=["gpu", "cpu"], required=True)
        p.add_argument("--out", required=True)
        if name == "prove":
            p.add_argument("--bin", required=True, help="flock-circuit (or its stamping wrapper)")
            p.add_argument("--zk-server", required=True, help="the ZK server's Lean session loop (or its stamping wrapper)")
            p.add_argument("--device", type=int, default=0, help="with --arm gpu: the CUDA index the prover proves on")
            p.add_argument("--card", help="the card's UUID, for its memory (nvidia-smi -i)")
            p.add_argument("--stamps", help="member_seconds.wrap's stamps.txt")
            p.add_argument("--timeout", type=float, default=3600, help="seconds before the session's processes are killed")
        else:
            p.add_argument("--lean", required=True)
    p = sub.add_parser("table")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"plan": plan, "prove": prove, "verify": verify, "table": table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
