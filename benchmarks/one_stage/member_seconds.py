"""An audit's seconds a member outside its prover's clock and in Lean's verify before its first check, split by what is
read: where a member's time goes that is neither proving nor checking, on an audit `a5.audit` ran (its ``out``), each
session behind the ZK server or not.

    python benchmarks/one_stage/member_seconds.py wrap DIR PROVER VERIFIER LOOP
        # a stamping wrapper of each binary under DIR (`PROGRAMS`: the prover, Lean's verifier and the ZK server's Lean
        # loop), their three paths printed: give the audit these
    python benchmarks/one_stage/member_seconds.py measure ROOT --stamped DIR [--reps 3] [--no-alone]
        # every audit under ROOT (a directory holding session-m<k>/): <audit>/member-seconds.json and a MEMBER-SECONDS line;
        # run it before the audit's logs are copied, which loses their modification times

`a5_tp.sh` does both under ``A5_MEMBER_SECONDS=1``.

* **Stamps** (`wrap`): each wrapper appends its start (``NAME PID PPID EPOCH ARGV``) to DIR/stamps.txt and execs the
  binary, so a process keeps its pid and its signals. A process's end is its log's modification time, which the audit
  writes when the process exits (`a0.run`).
* **The session** (`session`), per member with a process pair. Before the prover's clock: the verifier's process ahead of
  the prover's (its load, then the audit's poll for ``SERVING``), the prover's own load to where it starts the ZK
  server's Lean loop, and from there to the clock (the loop's reads, the connection and its pings). The clock's start is
  the prover's exit less ``e2e_s`` and its wait on the verdict after the clock (``wait_s`` less ``wait_e2e_s``), so the
  prover's exit is counted before the clock. Each side's load is its ``statement ... built in`` line.
* **Lean** (`lean`), per member: the sample's check, ``archive-put``, and ``verify`` split by its own lines (``inputs:``,
  the files read by key and the circuit parsed; its setup's line, `SETUP`; the session's line, whose remainder is the
  check).
  Before the first check is the member's own ``archive-put`` and ``verify`` less the check. The audit's one
  ``archive-put`` of the window's files before its sessions is the window's (`window`), counted once. A member the
  window's kept verifier checked (`a0.VerifyMembers`) has no ``verify`` process: its seconds are from its line to its
  answer, on its log's first lines, and the kept process's life is the window's.
* **Each reader alone** (`alone`, unless ``--no-alone``), from the member's stamped commands on the member's files, the
  median of ``--reps``: the verifier's and the prover's load (the prover's ``statement`` on the public file and on the
  prover's file), the Lean loop's start to ``LISTENING`` with and without the window's partition and program, its setup at
  the member's sample (``session --schedule``, which a session runs inside the clock), and ``archive-put`` of each file
  alone and of none (a Lean process's start), and ``verify`` on the member's session, a process of its own. A reader
  that fails is its error, never a figure. ``verify`` alone is also the guard of the audit's verdicts: it must print the
  verdict line the audit has for the member, byte for byte, whichever way the audit verified it, or ``measure`` exits 1.
  A reader run alone starts in this process's environment as it is, as the audit's kept verifier does: under the
  allocator's default unless the environment sets a value. A ``verify`` process of the audit's own runs under the
  verifier's allocator setting (`proofs.verifier_env`), so its seconds and ``verify`` alone's are under two settings.

It reads no value: the logs, the records' times, file sizes and the stamps. Seconds are rounded to a millisecond.
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
import time
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

#: the audit's three programs: the prover, Lean's verifier and the ZK server's Lean loop
PROGRAMS = PROVER, VERIFIER, LOOP = ("flock-circuit", "flock-verify", "flock-firewall")
#: the verifier's command that checks a session's sample, and the names its `verify` gives its setup's line
SAMPLE_CHECK, SETUP = "draw-test", ("setup (drawn)", "setup")
STAMPS, REAL = "stamps.txt", "real.json"
WRAPPER = """#!/bin/bash
echo "{name} $$ $PPID ${{EPOCHREALTIME/,/.}} $*" >> {stamps}
exec {real} "$@"
"""
BUILT = re.compile(r"^statement \S+ .* built in ([\d.]+)s", re.MULTILINE)
LEAN_MS = re.compile(r"^(.+): (\d+) ms$", re.MULTILINE)
#: a member's verify log from the window's kept verifier (`a0.VerifyMembers.verify`): its place, and its seconds
KEPT_MEMBER = re.compile(r"^# run as member (\d+) of: ", re.MULTILINE)
TOOK = re.compile(r"^# rc \S+ in ([0-9.]+) s$", re.MULTILINE)
#: the clock the readers run alone are timed with
clock = time.perf_counter


def wrap(d: Path, binaries: list[str]) -> list[Path]:
    """A wrapper of each of ``binaries`` (`PROGRAMS`, in that order) under ``d``, which stamps its start in ``d/stamps.txt``."""
    d.mkdir(parents=True, exist_ok=True)
    real = {name: str(Path(b).resolve()) for name, b in zip(PROGRAMS, binaries, strict=True)}
    (d / REAL).write_text(json.dumps(real, indent=1, sort_keys=True))
    for name, b in real.items():
        (d / name).write_text(WRAPPER.format(name=name, stamps=d / STAMPS, real=b))
        (d / name).chmod(0o755)
    return [d / name for name in PROGRAMS]


def stamps(text: str) -> list[dict]:
    """The wrappers' lines: each process's program, pid, parent, start in ns and arguments."""
    out = []
    for line in text.splitlines():
        f = line.split(" ")
        if len(f) >= 5 and f[0] in PROGRAMS:
            out.append({"program": f[0], "pid": int(f[1]), "ppid": int(f[2]), "ns": int(Decimal(f[3]) * 10**9), "argv": f[4:]})
    return out


def flag(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv[:-1] else None


def under(path: str | None, d: Path) -> bool:
    return path is not None and os.path.abspath(path).startswith(os.path.abspath(d) + os.sep)


def same(path: str, f: Path) -> bool:
    return os.path.abspath(path) == os.path.abspath(f)


def ended(log: Path) -> int | None:
    """When the process ``log`` records exited: the log's modification time, in ns."""
    return log.stat().st_mtime_ns if log.exists() else None


def last(found: list[dict], by: int | None) -> dict | None:
    """The latest of ``found`` started by ``by``: the process a log written at ``by`` records; none where there is no log."""
    found = [s for s in found if by is not None and s["ns"] <= by]
    return max(found, key=lambda s: s["ns"]) if found else None


def text(f: Path) -> str:
    return f.read_text(errors="replace") if f.exists() else ""


def live(log: str) -> dict:
    return next((json.loads(x[5:]) for x in log.splitlines() if x.startswith("LIVE\t")), {})


def built(log: str) -> float | None:
    m = BUILT.findall(log)
    return float(m[-1]) if m else None


def sub(a: int | None, b: int | None) -> float | None:
    """``a`` less ``b``, two times in ns, in seconds."""
    return None if a is None or b is None else round((a - b) / 1e9, 3)


def less(a: float | None, *bs: float | None) -> float | None:
    return None if a is None or any(b is None for b in bs) else round(a - sum(bs), 3)


def lean_lines(log: str, session: str | None) -> dict:
    """`verify`'s own lines of ``log``: ``inputs``, the session's setup and the session's whole, in seconds."""
    got = {name: int(ms) / 1000 for name, ms in LEAN_MS.findall(log.split("--- stderr", 1)[-1])}
    setup = next((got[name] for name in SETUP if name in got), None)
    whole = got.get(session) if session else None
    return {"inputs_s": got.get("inputs"), "setup_s": setup, "session_s": whole, "check_s": less(whole, setup)}


def kept_member(log: str) -> dict | None:
    """A verify log the window's kept verifier wrote (`a0.VerifyMembers.verify`): the member's place, the `verify` command
    its line stood for (``argv``, from ``verify`` on, as a stamp's) and its seconds; None for a `verify` process's log."""
    head = log.split("--- stdout", 1)[0]
    member, took = KEPT_MEMBER.search(head), TOOK.search(head)
    if member is None:
        return None
    argv = head.splitlines()[0].split(" ")
    return {"member": int(member.group(1)), "argv": argv[argv.index("verify"):] if "verify" in argv else [],
            "s": took and float(took.group(1))}


def found(audit: Path, k: int, all_stamps: list[dict]) -> dict:
    """Member ``k``'s honest processes among the stamps: the ones its logs record."""
    m, s = audit / f"m{k}", audit / f"session-m{k}"
    exited = ended(s / "prove.log")
    pair = [x for x in all_stamps if x["program"] == PROVER and under(flag(x["argv"], "--circuit"), m)]
    prove = last([x for x in pair if x["argv"][0] == "prove"], exited)
    serve = last([x for x in pair if x["argv"][0] == "serve" and under(flag(x["argv"], "--out"), s)], prove and prove["ns"])
    child = [x for x in all_stamps if x["program"] == LOOP and prove is not None and x["ppid"] == prove["pid"]
             and x["ns"] >= prove["ns"]]
    verify = [x for x in all_stamps if x["program"] == VERIFIER and len(x["argv"]) > 1]
    return {"serve": serve, "prove": prove,
            "tap": last([x for x in child if x["argv"][0] == "tap"], exited),
            "loop": last([x for x in child if x["argv"][0] == "session"], exited),
            "sample-check": last([x for x in verify if x["argv"][0] == SAMPLE_CHECK and same(x["argv"][1], s / "u1-u3.cases.json")],
                                 ended(s / "u1-u3.lean.log")),
            "archive-put": last([x for x in verify if x["argv"][0] == "archive-put" and under(x["argv"][1], s)],
                                ended(s / "archive-put.log")),
            "verify": last([x for x in verify if x["argv"][0] == "verify" and under(flag(x["argv"], "--session"), s)],
                           ended(s / "lean-verify.log"))}


def session(audit: Path, k: int, at: dict) -> dict:
    """Member ``k``'s session outside its prover's clock (the module docstring's second bullet)."""
    s = audit / f"session-m{k}"
    lv = live(text(s / "prove.log"))
    ns = {name: x and x["ns"] for name, x in at.items()}
    prover = sub(ended(s / "prove.log"), ns["prove"])
    after = less(lv.get("wait_s"), lv.get("wait_e2e_s"))
    before = less(prover, lv.get("e2e_s"), after)
    to_loop = sub(ns["loop"], ns["prove"])
    return {"verifier_ahead_s": sub(ns["prove"], ns["serve"]), "verifier_built_s": built(text(s / "serve.log")),
            "prover_s": prover, "prover_built_s": built(text(s / "prove.log")), "tap_at_s": sub(ns["tap"], ns["prove"]),
            "to_loop_s": to_loop, "loop_to_clock_s": less(before, to_loop), "before_clock_s": before,
            "e2e_s": lv.get("e2e_s"), "witness_s": lv.get("witness_s"), "prove_total_s": lv.get("prove_total_s"),
            "after_clock_wait_s": after, "accepted": lv.get("accepted")}


def lean(audit: Path, k: int, at: dict) -> dict:
    """Member ``k``'s Lean checks (the module docstring's third bullet)."""
    s = audit / f"session-m{k}"
    ns = {name: x and x["ns"] for name, x in at.items()}
    put, whole = sub(ended(s / "archive-put.log"), ns["archive-put"]), sub(ended(s / "lean-verify.log"), ns["verify"])
    # a member whose files the window archived (`window`) puts nothing, and has no log of a put
    own = 0.0 if not (s / "archive-put.log").exists() else put
    log = text(s / "lean-verify.log")
    kept = kept_member(log)
    verify = kept or at["verify"]
    if kept:    # no process of its own: its seconds are from its line to its answer
        whole = kept["s"]
    lines = lean_lines(log, verify and flag(verify["argv"], "--session"))
    return {"waited_s": sub(ns["sample-check"], ended(s / "prove.log")),
            "sample_check_s": sub(ended(s / "u1-u3.lean.log"), ns["sample-check"]), "archive_put_s": put, "verify_s": whole, **lines,
            "verify_rest_s": less(whole, lines["inputs_s"], lines["session_s"]),
            "before_first_check_s": less(None if own is None or whole is None else own + whole, lines["check_s"]),
            "kept_member": kept and kept["member"]}


def window(audit: Path, all_stamps: list[dict]) -> dict:
    """What Lean does once for the window: `a5.audit`'s one ``archive-put`` of every sampled member's files, before the
    first session, and its kept verifier's life, from its start to its exit after the last member; no figure for an audit
    whose members each put their own, or each have a ``verify`` of their own."""
    d = audit / "lean-verify"
    mine = [x for x in all_stamps if x["program"] == VERIFIER and len(x["argv"]) > 1]
    end, closed = ended(d / "archive-put.log"), ended(d / "verify-members.log")
    at = last([x for x in mine if x["argv"][0] == "archive-put" and same(x["argv"][1], d / "archive")], end)
    kept = last([x for x in mine if x["argv"][0] == "verify-members" and same(flag(x["argv"], "--archive") or "", d / "archive")],
                closed)
    return {"archive_put_s": sub(end, at and at["ns"]), "files": at and len(at["argv"]) - 2,
            "verify_members_s": sub(closed, kept and kept["ns"])}


def sizes(audit: Path, k: int) -> dict:
    m = audit / f"m{k}"
    files = {"circuit": m / "circuit.txt", "partition": audit / "partition.json", "program": audit / "program.json",
             "registered": m / "registered.json", **{f"public:{f.name}": f for f in sorted(m.glob("pub-*.bin"))},
             **{f"prover:{f.name}": f for f in sorted(m.glob("inst-*.bin"))},
             **{f"table:{f.name}": f for f in sorted((m / "tables").glob("*"))}}
    return {name: f.stat().st_size for name, f in files.items() if f.is_file()}


def timed(cmd: list[str], stdin: str | None = None) -> tuple[float, subprocess.CompletedProcess]:
    t = clock()
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=3600)
    return clock() - t, p


def listening(cmd: list[str]) -> float:
    """Seconds from the Lean loop's start to its ``LISTENING`` line; it ends when its standard input closes."""
    t = clock()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = p.stdout.readline()
        s = clock() - t
        if not line.startswith("LISTENING "):
            raise RuntimeError(f"no LISTENING line: {line.strip()!r} {p.stderr.read()[-300:]!r}")
        return s
    finally:
        p.stdin.close()
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        p.stdout.close()
        p.stderr.close()


def median(run: Callable[[], float], reps: int) -> dict:
    """``run``'s seconds ``reps`` times: their median and each, or the error of the first that fails."""
    got = []
    try:
        for _ in range(reps):
            got.append(run())
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        return {"error": str(e)[-400:]}
    return {"s": round(statistics.median(got), 3), "each": [round(x, 3) for x in got]}


def kept(argv: list[str], flags: tuple[str, ...]) -> list[str]:
    return [x for name in flags if (v := flag(argv, name)) is not None for x in (name, v)]


def alone(audit: Path, k: int, at: dict, real: dict, reps: int) -> dict:
    """Each reader of member ``k``'s files run alone (the module docstring's fourth bullet)."""
    m, s = audit / f"m{k}", audit / f"session-m{k}"
    out: dict = {}
    for side, name in (("serve", "verifier_load"), ("prove", "prover_load")):
        if at[side] is None:
            continue
        argv, loads = at[side]["argv"], []
        cmd = [real[PROVER], "statement", *(["--zk"] if "--zk" in argv else []),
               *kept(argv, ("--instances", "--circuit", "--tmpl-unit", "--tables", "--pin"))]

        def statement(cmd: list[str] = cmd, loads: list[float] = loads) -> float:
            sec, p = timed(cmd)
            if p.returncode or built(p.stderr) is None:
                raise RuntimeError(f"statement exited {p.returncode}: {p.stderr[-300:]}")
            loads.append(built(p.stderr))
            return sec
        out[name] = median(statement, reps)
        if "error" not in out[name]:
            out[name]["built_s"] = round(statistics.median(loads), 3)
    # the loop's own public file is the prover's copy, gone once the loop has started: the verifier's is the same file
    public = at["serve"] and flag(at["serve"]["argv"], "--instances")
    if at["loop"] is not None and public:
        argv = at["loop"]["argv"]
        base = [real[LOOP], "session", *kept(argv, ("--statement", "--circuit")), "--public", public,
                *kept(argv, ("--tables", "--session-tables"))]
        window = kept(argv, ("--partition", "--program"))
        out["loop_start"] = median(lambda: listening([*base, "--verifier", "127.0.0.1:9", *window]), reps)
        out["loop_start_no_window"] = median(lambda: listening([*base, "--verifier", "127.0.0.1:9"]), reps)
        sample = text(s / "draw.json")

        def schedule() -> float:
            sec, p = timed([*base, *window, "--schedule"], sample)
            if p.returncode or not p.stdout.startswith("ok "):
                raise RuntimeError(f"session --schedule exited {p.returncode}: {p.stdout[:200]} {p.stderr[-300:]}")
            return sec
        out["loop_setup_at_sample"] = median(schedule, reps)
    if at["archive-put"] is not None:
        def put(files: list[str]) -> Callable[[], float]:
            def run() -> float:
                d = tempfile.mkdtemp(prefix="member-seconds-")
                try:
                    sec, p = timed([real[VERIFIER], "archive-put", d, *files])
                finally:
                    shutil.rmtree(d, ignore_errors=True)
                if p.returncode:
                    raise RuntimeError(f"archive-put exited {p.returncode}: {p.stderr[-300:]}")
                return sec
            return run
        # a later member of a template links the circuit its first member put (`a0.lean_verify`): here every member's is put
        files = at["archive-put"]["argv"][2:]
        circuit = str(m / "circuit.txt")
        out["archive_put"] = {"none": median(put([]), reps),
                              **{Path(f).name: median(put([f]), reps) for f in ([] if circuit in files else [circuit]) + files}}
    log = text(s / "lean-verify.log")
    verify = kept_member(log) or at["verify"]
    if verify is not None:
        want, split = [x for x in log.splitlines() if x.startswith("VERDICT ")], []

        def alone_verify() -> float:
            sec, p = timed([real[VERIFIER], *verify["argv"]])
            if [x for x in p.stdout.splitlines() if x.startswith("VERDICT ")] != want:
                raise RuntimeError(f"verify alone (exit {p.returncode}) printed another verdict than the audit has for this "
                                   f"member: {p.stdout[:300]} {p.stderr[-300:]}")
            split.append(lean_lines("--- stderr\n" + p.stderr, flag(verify["argv"], "--session")))
            return sec
        out["verify"] = median(alone_verify, reps)
        if "error" not in out["verify"]:
            out["verify"].update({x: statistics.median(r[x] for r in split) if all(r[x] is not None for r in split) else None
                                  for x in ("inputs_s", "setup_s", "check_s")})
    return out


def members_of(audit: Path) -> list[int]:
    return sorted(int(d.name[9:]) for d in audit.glob("session-m*") if d.is_dir() and d.name[9:].isdigit())


def audits(root: Path) -> list[Path]:
    return sorted({d.parent for d in root.rglob("session-m*") if d.is_dir() and d.name[9:].isdigit()})


#: the logs whose modification times are their processes' ends
LOGS = {"prove": "prove.log", "sample-check": "u1-u3.lean.log", "archive-put": "archive-put.log", "verify": "lean-verify.log"}


def measure(audit: Path, all_stamps: list[dict]) -> dict:
    """``audit``'s members' seconds: a record a member with a session, in the members' order."""
    summary = json.loads(text(audit / "summary.json") or "{}")
    named = {e["k"]: e for e in summary.get("members", []) if isinstance(e, dict) and "k" in e}
    waits = {v.get("sigma"): v.get("session_s") for v in (summary.get("timing") or {}).values() if isinstance(v, dict)}
    members = []
    for k in members_of(audit):
        at, s = found(audit, k, all_stamps), audit / f"session-m{k}"
        sigma = live(text(s / "prove.log")).get("sigma")
        members.append({"k": k, "template": named.get(k, {}).get("template"), "n": named.get(k, {}).get("n"),
                        "bytes": sizes(audit, k), "starts_ns": {name: x and x["ns"] for name, x in at.items()},
                        "ends_ns": {name: ended(s / log) for name, log in LOGS.items()},
                        "session": {**session(audit, k, at), "session_s": waits.get(sigma) if sigma else None},
                        "lean": lean(audit, k, at)})
    for a, b in zip(members, members[1:]):
        a["to_next_member_s"] = sub(b["starts_ns"]["serve"], a["ends_ns"]["prove"])
    return {"audit": audit.name, "zk_server": bool(summary.get("firewall")), "stamped": bool(all_stamps),
            "staging": summary.get("staging"), "window": window(audit, all_stamps), "members": members}


def readers(audit: Path, got: dict, all_stamps: list[dict], real: dict, reps: int) -> bool:
    """Adds each member's readers run alone (`alone`) to `measure`'s ``got``; False when one could not be run at all, or
    `verify` alone did not print the audit's verdict of a member."""
    ok = True
    for rec in got["members"]:
        try:
            rec["alone"] = alone(audit, rec["k"], found(audit, rec["k"], all_stamps), real, reps)
            ok = ok and "error" not in rec["alone"].get("verify", {})
        except Exception as e:  # noqa: BLE001  (the audit's own seconds are kept whatever a reader's command does)
            rec["alone"], ok = {"error": repr(e)[-400:]}, False
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="member_seconds.py", description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("wrap", help="a stamping wrapper of each binary")
    w.add_argument("dir", type=Path)
    w.add_argument("binaries", nargs=3, metavar="BIN", help="the prover, Lean's verifier and the ZK server's Lean loop, in that order")
    m = sp.add_parser("measure", help="every audit under ROOT")
    m.add_argument("root", type=Path)
    m.add_argument("--stamped", type=Path, required=True, help="wrap's DIR")
    m.add_argument("--reps", type=int, default=3)
    m.add_argument("--no-alone", action="store_true", help="the audit's own seconds only: no reader run alone")
    a = ap.parse_args(argv)
    if a.cmd == "wrap":
        print(*wrap(a.dir, a.binaries))
        return 0
    if not (a.stamped / STAMPS).exists():
        print(f"member_seconds: no {a.stamped / STAMPS}: run the audit on wrap's binaries", file=sys.stderr)
        return 1
    all_stamps = stamps(text(a.stamped / STAMPS))
    real = None if a.no_alone else json.loads((a.stamped / REAL).read_text())
    found_audits = audits(a.root)
    if not found_audits:
        print(f"member_seconds: no audit under {a.root} (a directory holding session-m<k>/)", file=sys.stderr)
        return 1
    ok = True
    for audit in found_audits:
        got = measure(audit, all_stamps)
        (audit / "member-seconds.json").write_text(json.dumps(got, indent=1, sort_keys=True))
        if real is not None:
            ok = readers(audit, got, all_stamps, real, a.reps) and ok
            (audit / "member-seconds.json").write_text(json.dumps(got, indent=1, sort_keys=True))
        print("MEMBER-SECONDS " + json.dumps({"audit": audit.name, "window_archive_put_s": got["window"]["archive_put_s"], "members": [
            {"k": r["k"], "template": r["template"], "before_clock_s": r["session"]["before_clock_s"],
             "verifier_ahead_s": r["session"]["verifier_ahead_s"], "e2e_s": r["session"]["e2e_s"],
             "before_first_check_s": r["lean"]["before_first_check_s"], "check_s": r["lean"]["check_s"],
             "verify_s": r["lean"]["verify_s"], "verify_alone": r.get("alone", {}).get("verify")}
            for r in got["members"]]}))
    if not ok:
        print("member_seconds: a member's readers could not be run alone, or `verify` alone printed another verdict than "
              "the audit's (its `alone` is the error)", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
