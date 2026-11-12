"""The Lean verifier's seconds a member of a run before its first check: one process a member (`verify`) against one
kept process for them all (`verify-members`, which reads the run's program once).

    python3 benchmarks/verifier_cost/flock_members.py --lean BIN --argv FILE --out DIR [--members 3] [--rounds 2] [--core C]

FILE holds one `verify` call, an argument a line, `verify` first: a member's call as its caller makes it, with the run's
`--program` (and `--archive` where the call names its files by digest). The member is handed in `--members` times over.
`verify-members` takes that as so many members, each a call of its own (one `verify` call would refuse the second for
its reused coin key), so each costs what a member with that program and a statement of that size costs. A round is the
members alone, a process each, then the kept process on the same argument lists, a line a member, the next written once
the last is answered; the rounds alternate so both ways see the node's load, which each row records.

Per member and way, from the verifier's own stage lines on standard error: `inputs_ms` (its files read, hashed and
parsed before any session), `setup_ms` (its session's statement set up), `before_check_ms`, their sum, the seconds a
member waits before its first check, and `session_ms` (the setup and the check). `wall_s` is the process's for a member
alone, and for a kept member the time from its line written to its `MEMBER` line read: the first member's holds the
process's start and its one read of the program. A kept process's peak RSS is the whole process's, and a kept member's
`rss_kb` is the process's resident memory when that member's `MEMBER` line arrives: what the process holds, beside
whatever runs next to it, until the next member's line.

Every process runs in `verifier_env`, as the verifier's Python callers start it: under the verifier's allocator setting
unless this job's environment sets a value of its own. The setting changes the memory a process holds and its seconds,
so `members.json` records the value the processes ran under (`env`); figures under two values are not one figure.

The guard, without which no figure counts: each kept member's `VERDICT` lines are byte for byte those `verify` printed
for it alone in the same round, its `MEMBER` line carries that exit status and its index, every member is accepted,
and the kept process exits 0. `DIR/members.json` (`verity/flock-members/v1`) has the rows, the kept processes, the
guard's failures and `summary`, the medians a way: alone, the kept process's first member, and its later members.
Exit 3 when the guard fails, after the file is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from verity.protocols.spot.verifier.one_stage.proofs import VERIFIER_ENV, verifier_env

SCHEMA = "verity/flock-members/v1"
STAGE = re.compile(r"^(.+): (\d+) ms$")
ONCE = ("--archive", "--program")


def stages(err: str) -> list[dict]:
    """The stage lines of a process's standard error, a dict a `verify` body: `inputs_ms`, `setup_ms`, `session_ms`."""
    calls: list[dict] = []
    for line in err.splitlines():
        m = STAGE.match(line)
        if not m:
            continue
        name, ms = m.group(1), int(m.group(2))
        if name == "inputs":
            calls.append({"inputs_ms": ms, "setup_ms": 0, "session_ms": 0})
        elif calls:
            key = "setup_ms" if name == "setup" or name.startswith("setup (") else "session_ms"
            calls[-1][key] += ms
    for c in calls:
        c["before_check_ms"] = c["inputs_ms"] + c["setup_ms"]
    return calls


def once_a_run(args: list[str]) -> list[str]:
    """What `verify-members` is given once: the member's own `--archive` and `--program`."""
    given: list[str] = []
    for flag in ONCE:
        if flag in args:
            given += [flag, args[args.index(flag) + 1]]
    if "--program" not in given:
        sys.exit("flock_members: the call names no --program; a kept process has nothing to keep")
    return given


def reaped(p: subprocess.Popen) -> tuple[int, int]:
    """The exit status and peak RSS in kB of a child whose pipes are read to their end."""
    _, status, usage = os.wait4(p.pid, 0)
    p.returncode = os.waitstatus_to_exitcode(status)
    return p.returncode, usage.ru_maxrss


def resident_kb(pid: int) -> int:
    """A live process's resident memory in kB (`VmRSS` of its `/proc` status)."""
    [line] = [x for x in Path(f"/proc/{pid}/status").read_text().splitlines() if x.startswith("VmRSS:")]
    return int(line.split()[1])


def alone(cmd: list[str], args: list[str], out: str, env: dict[str, str]) -> dict:
    """One `verify` on a member's arguments in `env`, its output in `out`.out and its stage lines in `out`.err."""
    t0 = time.monotonic()
    with open(f"{out}.out", "wb") as o, open(f"{out}.err", "wb") as e:
        p = subprocess.Popen([*cmd, "verify", *args], stdout=o, stderr=e, env=env)
        rc, rss = reaped(p)
    return {"exit": rc, "wall_s": round(time.monotonic() - t0, 3), "max_rss_kb": rss, "verdict": Path(f"{out}.out").read_bytes()}


def kept(cmd: list[str], once: list[str], lines: list[bytes], out: str, env: dict[str, str]) -> tuple[dict, list[dict]]:
    """One `verify-members` on `lines` in `env`: the process, and an answer a member reached (its lines up to `MEMBER`, its
    wall)."""
    answers: list[dict] = []
    closed = None
    t0 = time.monotonic()
    with open(f"{out}.err", "wb") as e:
        p = subprocess.Popen([*cmd, "verify-members", *once], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=e,
                             env=env)
        assert p.stdin is not None and p.stdout is not None
        for line in lines:
            t1 = time.monotonic()
            try:
                p.stdin.write(line)
                p.stdin.flush()
            except BrokenPipeError as e:
                closed = f"its input closed before member {len(answers)}'s line was written ({e})"
                break
            answer: list[bytes] = []
            while (got := p.stdout.readline()):
                answer.append(got)
                if got.startswith(b"MEMBER "):
                    break
            if not answer or not answer[-1].startswith(b"MEMBER "):
                break
            answers.append({"lines": answer, "wall_s": round(time.monotonic() - t1, 3), "rss_kb": resident_kb(p.pid)})
        try:
            p.stdin.close()
        except BrokenPipeError as e:
            closed = closed or f"its input closed before its end ({e})"
        rest = p.stdout.read()
        rc, rss = reaped(p)
    Path(f"{out}.out").write_bytes(b"".join(b"".join(a["lines"]) for a in answers) + rest)
    return {"exit": rc, "wall_s": round(time.monotonic() - t0, 3), "max_rss_kb": rss, "answered": len(answers),
            "closed": closed}, answers


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lean", required=True, help="the Lean verifier's executable")
    ap.add_argument("--argv", required=True, help="one verify call, an argument a line, `verify` first")
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--core", type=int, default=None, help="pin every process to this core (default: the last it may use)")
    a = ap.parse_args()
    argv = Path(a.argv).read_text().splitlines()
    if not argv or argv[0] != "verify":
        sys.exit(f"flock_members: {a.argv} is not a verify call")
    args = argv[1:]
    once = once_a_run(args)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    core = a.core if a.core is not None else max(os.sched_getaffinity(0))
    cmd = ["taskset", "-c", str(core), a.lean]
    env = verifier_env()
    line = json.dumps(args).encode() + b"\n"
    rows: list[dict] = []
    processes: list[dict] = []
    failures: list[str] = []
    for r in range(a.rounds):
        singles = []
        for i in range(a.members):
            load = os.getloadavg()[0]
            one = alone(cmd, args, f"{out}/r{r}.alone{i}", env)
            got = stages(Path(f"{out}/r{r}.alone{i}.err").read_text(errors="replace"))
            if len(got) != 1:
                failures.append(f"round {r}: member {i} alone printed {len(got)} `inputs:` lines, not 1")
            if one["exit"] != 0:
                failures.append(f"round {r}: member {i} alone exited {one['exit']}: only an accepted member is timed")
            singles.append(one)
            rows.append({"round": r, "way": "alone", "member": i, "load": load, **(got[0] if got else {}),
                         **{k: one[k] for k in ("wall_s", "exit", "max_rss_kb")}})
            print(f"FM {json.dumps(rows[-1])}", flush=True)
        load = os.getloadavg()[0]
        process, answers = kept(cmd, once, [line] * a.members, f"{out}/r{r}.kept", env)
        got = stages(Path(f"{out}/r{r}.kept.err").read_text(errors="replace"))
        processes.append({"round": r, "load": load, **process})
        print(f"FM {json.dumps({'way': 'kept-process', **processes[-1]})}", flush=True)
        if process["exit"] != 0 or process["closed"] or len(answers) != a.members or len(got) != a.members:
            failures.append(f"round {r}: the kept process exited {process['exit']} with {len(answers)} of {a.members} members "
                            f"answered and {len(got)} `inputs:` lines" + (f"; {process['closed']}" if process["closed"] else ""))
        for i, answer in enumerate(answers):
            verdict, last = b"".join(answer["lines"][:-1]), answer["lines"][-1]
            want = b'MEMBER {"exit":%d,"index":%d}\n' % (singles[i]["exit"], i)
            if verdict != singles[i]["verdict"]:
                failures.append(f"round {r}: member {i}'s kept verdict (sha256 {hashlib.sha256(verdict).hexdigest()[:16]}) is not "
                                f"verify's alone ({hashlib.sha256(singles[i]['verdict']).hexdigest()[:16]})")
            if last != want:
                failures.append(f"round {r}: member {i}'s line is {last!r}, not {want!r}")
            rows.append({"round": r, "way": "kept", "member": i, "load": load, **(got[i] if i < len(got) else {}),
                         "wall_s": answer["wall_s"], "rss_kb": answer["rss_kb"],
                         "exit": json.loads(last[len(b"MEMBER "):]).get("exit")})
            print(f"FM {json.dumps(rows[-1])}", flush=True)

    def of(way: str, key: str, members) -> float | None:
        return median([row[key] for row in rows if row["way"] == way and row["member"] in members and key in row])

    later = range(1, a.members)
    summary = {f"{name}_{key}": of(way, key, members)
               for name, way, members in (("alone", "alone", range(a.members)), ("kept_first", "kept", (0,)),
                                          ("kept_later", "kept", later))
               for key in ("before_check_ms", "inputs_ms", "setup_ms", "session_ms", "wall_s", "rss_kb")}
    doc = {"schema": SCHEMA, "lean_sha256": hashlib.sha256(Path(a.lean).read_bytes()).hexdigest(), "members": a.members,
           "rounds": a.rounds, "core": core, "env": {name: env[name] for name in VERIFIER_ENV}, "once": once, "rows": rows,
           "kept": processes,
           "guard": {"ok": not failures, "failures": failures}, "summary": summary}
    (out / "members.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"FM {json.dumps({'guard': doc['guard'], 'summary': summary})}", flush=True)
    return 0 if not failures else 3


if __name__ == "__main__":
    sys.exit(main())
