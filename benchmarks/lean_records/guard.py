#!/usr/bin/env python3
"""The Lean tools' guard: a records run of `security/Proofs` (`lean_changed.py --records security/Proofs --rebuild`, with
its requirers) on one kept records tree of a node, a warm-up and then rounds of three arms, each on a one-line lemma of
its own added to a `security/Proofs` module, so that no arm reuses another's build:

  A  the base tools (``--base``: each of `TOOLS` as base-check.py, ...; one the base hasn't is absent) at the job's
     reservation
  B  this tree's tools at the job's reservation
  C  this tree's tools at ``--low-gb`` (RESEARCH_RUN_MEM_GB = CHECK_RUN_MEM_GB)

It holds one slot of the node's `build` Lean pool for the whole job and runs every arm on that slot's tree, with
LEAN_SLOTS pointed at a file that doesn't exist so that no arm takes a second slot. Each arm reads, every second, what its
process tree holds (RssAnon + RssShmem), the `lean` processes in it, and the user and system CPU of the tools' own Python
processes (their children aside); and at its end the tree's CPU (`wait4`) and its log (`read_log`). A process that exits
while the sampler reads it counts in the arm's ``gone``.

An arm is clean where the first `security/Proofs` build compiled modules: one whose lemma an earlier run on the node built
restores every module from the node's artifact cache, which its `sampled:` line shows (`read_log`). The verdict
(`summarize`) is over clean arms alone and judges B against A on the build step's seconds and the tools' own CPU, each
within 1.1 of A's median, and each kind's peak against its reservation. An arm's whole seconds are reported and judge
nothing: the audit's spot-check recompiles a random sample of the modules its build restored (`check.py`'s `SAMPLE_RNG`),
whose seconds differ several-fold from one sample to the next, and they move with the node's load.

    python3 guard.py --out DIR --base DIR [--rounds N] [--first-lemma K] [--low-gb GB]

Lemma k is arm k's, from ``--first-lemma``: a job after another on the same node starts past the lemmas the first built.
Writes ``--out``/guard.json, and per arm a timestamped log and its memory series. The README has the run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path.cwd()
LEAN = ROOT / "tools/verity/lean"
#: the files of the tools an arm runs, the plan's rules and figures among them
TOOLS = ("check.py", "lean_changed.py", "lean_audit.py", "reserve.py", "reserve.json")
MODULE = ROOT / "security/Proofs/Flock/SamplingGeneric/TwoAt.lean"
END = "end SamplingDeploy"
PKG = "security/Proofs"
EVERY_S = 1.0
#: the largest file of an arm's reports kept in $OUT
KEEP_BYTES = 256 * 1024
#: B within this share of A's median, on each judged reading
WITHIN = 1.1
#: the readings the verdict judges, which the node's load and the spot-check's sample move least
JUDGED = ("build_s", "tool_cpu_s")

AUDIT_LINE = re.compile(r"AUDIT (\S+): ")
STAGED = re.compile(r"staged: (\d+) of its (\d+) keyed modules")
SAMPLED = re.compile(r"sampled: \d+ of the (\d+) modules its build didn't compile, recompiled in ([\d.]+) s")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: this tree's plan: what says a run reserved, in which unit, and the figures
RESERVE = load("guard_reserve", Path(__file__).resolve().parents[2] / "tools/verity/lean/reserve.py")


def say(s: str) -> None:
    print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} guard: {s}", flush=True)


def read_log(text: str, pkg: str = PKG) -> dict:
    """What a records run's output says of ``pkg``'s audits, one per check.py run that audited it: the modules the first
    one's build compiled (its `staged:` line's keyed modules less its `sampled:` line's modules the build didn't compile),
    whether that is any (``clean``), and the seconds of every one's spot-check (``spot_s``)."""
    audits, cur = [], None
    for line in text.splitlines():
        if m := AUDIT_LINE.search(line):
            cur = {} if m[1] == pkg or m[1].endswith(f"/{pkg}") else None
            if cur is not None:
                audits.append(cur)
        elif cur is not None and (m := STAGED.search(line)):
            cur["keyed"] = int(m[2])
        elif cur is not None and (m := SAMPLED.search(line)):
            cur["undone"], cur["spot_s"] = int(m[1]), float(m[2])
    first = audits[0] if audits else {}
    compiled = first["keyed"] - first["undone"] if {"keyed", "undone"} <= first.keys() else None
    return {"clean": bool(compiled), "compiled": compiled,
            "spot_s": round(sum(a.get("spot_s", 0.0) for a in audits), 1), "audits": len(audits)}


def tree_pids(root: int, gone: set[int]) -> list[int]:
    """``root`` and its descendants; a process that exits while its `stat` is read goes into ``gone``."""
    kids: dict[int, list[int]] = {}
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                ppid = int((p / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                gone.add(int(p.name))
                continue
            kids.setdefault(ppid, []).append(int(p.name))
    out, todo = [], [root]
    while todo:
        q = todo.pop()
        out.append(q)
        todo.extend(kids.get(q, []))
    return out


def held(pid: int, gone: set[int]) -> tuple[int, str]:
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except OSError:
        gone.add(pid)
        return 0, ""
    kb = sum(int(ln.split()[1]) for ln in lines if ln.startswith(("RssAnon:", "RssShmem:")))
    return kb * 1024, next((ln.split()[1] for ln in lines if ln.startswith("Name:")), "")


def ticks(pid: int, gone: set[int]) -> int | None:
    """The process's own user and system CPU so far, in clock ticks (`/proc/PID/stat`'s utime and stime)."""
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12])
    except (OSError, ValueError, IndexError):
        gone.add(pid)
        return None


def tool(pid: int, name: str, seen: dict[int, bool], gone: set[int]) -> bool:
    """Whether ``pid`` is a Python process of the tools (its command line names `tools/verity/lean/`)."""
    if pid not in seen:
        try:
            seen[pid] = name.startswith("python") and b"tools/verity/lean/" in Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            gone.add(pid)
            seen[pid] = False
    return seen[pid]


def sample(pid: int, t0: float, series: Path, done: threading.Event, most: dict) -> None:
    """Every `EVERY_S` until ``done``: the process tree under ``pid``, its total, its largest process and its `lean` count,
    into ``series``, the most of each into ``most``, the last CPU ticks of each of the tools' processes (``own``), and
    each process that exited while it was read (``gone``)."""
    seen: dict[int, bool] = {}
    gone = most["gone"]
    with series.open("w") as s:
        s.write("t_s\tgib\tlean\tlargest_gib\n")
        while not done.is_set():
            total, lean, largest = 0, 0, 0
            for q in tree_pids(pid, gone):
                b, name = held(q, gone)
                total += b
                lean += name == "lean"
                largest = max(largest, b)
                if tool(q, name, seen, gone) and (t := ticks(q, gone)) is not None:
                    most["own"][q] = max(most["own"].get(q, 0), t)
            t = time.monotonic() - t0
            s.write(f"{t:.0f}\t{total / 2**30:.2f}\t{lean}\t{largest / 2**30:.2f}\n")
            s.flush()
            if total > most["bytes"]:
                most.update(bytes=total, at=t, lean_at=lean)
            most.update(lean=max(most["lean"], lean), largest=max(most["largest"], largest))
            done.wait(EVERY_S)


def arm(n: int, kind: str, k: int, tree: Path, tools: dict[str, bytes | None], original: str, out_dir: Path,
        low_gb: int) -> dict:
    for f in TOOLS:
        if tools[f] is None:
            (LEAN / f).unlink(missing_ok=True)
        else:
            (LEAN / f).write_bytes(tools[f])
    head, sep, tail = original.rpartition(END)
    MODULE.write_text(f"{head}theorem guardLeanRecords_{k} : ({k} : Nat) + 0 = {k} := rfl\n\n{sep}{tail}")
    env = {x: v for x, v in os.environ.items() if x not in ("CHECK_SLOT_MEM_GB", "LEAN_NUM_THREADS", "LEAN_SLOT")}
    env |= {"LEAN_SLOTS": str(out_dir / "no-lean-slots"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    if kind == "C":
        env |= {"RESEARCH_RUN_MEM_GB": str(low_gb), "CHECK_RUN_MEM_GB": str(low_gb)}
    out, log, series = out_dir / f"arm{n}-{kind}", out_dir / f"arm{n}-{kind}.log", out_dir / f"arm{n}-{kind}.mem.tsv"
    reserved = {v: env[v] for v, _ in RESERVE.RESERVATIONS if v in env}
    say(f"arm {n} ({kind}, lemma {k}) starts; reservation {reserved}")
    t0 = time.monotonic()
    p = subprocess.Popen([sys.executable, str(LEAN / "lean_changed.py"), "--records", PKG, "--tree", str(tree), "--out",
                          str(out), "--rebuild"], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, text=True, errors="replace")
    done, most = threading.Event(), {"bytes": 0, "lean": 0, "at": 0.0, "lean_at": 0, "largest": 0, "own": {}, "gone": set()}
    th = threading.Thread(target=sample, args=(p.pid, t0, series, done, most), daemon=True)
    th.start()
    with log.open("w") as lg:
        for line in p.stdout:
            if not line.startswith("lean-slot:"):
                lg.write(f"{time.monotonic() - t0:8.1f} {line}")
    _, status, usage = os.wait4(p.pid, 0)
    rc = p.returncode = os.waitstatus_to_exitcode(status)
    seconds = time.monotonic() - t0
    done.set()
    th.join()
    reps = []
    for r in sorted(out.rglob("report.json")):
        try:
            d = json.loads(r.read_text())
        except (OSError, ValueError) as e:
            reps.append({"at": str(r.relative_to(out)), "unread": f"{type(e).__name__}: {e}"})
            continue
        reps.append({"at": str(r.relative_to(out)), **{x: d.get(x) for x in
                     ("package", "status", "build_threads", "seconds", "peak_gib")}})
    for f in [f for f in out.rglob("*") if f.is_file() and f.stat().st_size > KEEP_BYTES]:
        f.unlink()
    got = read_log(log.read_text(errors="replace"))
    proofs = next((r for r in reps if r["at"] == "audit-Proofs/report.json"), {})
    rec = {"n": n, "arm": kind, "lemma": k, "rc": rc, "seconds": round(seconds, 1), "clean": got["clean"],
           "compiled": got["compiled"], "spot_s": got["spot_s"], "rest_s": round(seconds - got["spot_s"], 1),
           "build_s": (proofs.get("seconds") or {}).get("build"),
           "tool_cpu_s": round(sum(most["own"].values()) / os.sysconf("SC_CLK_TCK"), 1),
           "tree_cpu_s": round(usage.ru_utime + usage.ru_stime, 1),
           "peak_gib": round(most["bytes"] / 2**30, 2), "peak_at_s": round(most["at"]), "lean_at_peak": most["lean_at"],
           "max_lean": most["lean"], "largest_gib": round(most["largest"] / 2**30, 2), "gone": len(most["gone"]),
           "reserved": reserved, "reports": reps}
    say(f"arm {n} ({kind}) rc {rc}: {rec['seconds']} s ({rec['spot_s']} of them spot-checks), build {rec['build_s']} s, "
        f"clean {rec['clean']} ({rec['compiled']} compiled), the tools' own CPU {rec['tool_cpu_s']} s, the tree's "
        f"{rec['tree_cpu_s']} s; peak {rec['peak_gib']} GiB with {rec['lean_at_peak']} lean, at most {rec['max_lean']} "
        f"lean at once, largest process {rec['largest_gib']} GiB")
    return rec


def gib_of(reserved: dict[str, str]) -> float | None:
    unit = dict(RESERVE.RESERVATIONS)
    vals = [float(v) if unit[k] == "GiB" else RESERVE.gib(float(v)) for k, v in reserved.items()]
    return min(vals) if vals else None


def median(rs: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rs if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def summarize(recs: list[dict]) -> tuple[dict, dict]:
    """Per kind, over its clean arms: the medians of the whole seconds, the seconds without the spot-checks, the build
    step, the tools' own CPU and the tree's, and the peaks. The verdict: B within `WITHIN` of A's median on each of
    `JUDGED`, and each kind's peak within its reservation. The whole seconds judge nothing (the module docstring)."""
    by = {k: [r for r in recs if r["arm"] == k and r["clean"]] for k in "ABC"}
    summary = {}
    for k, rs in by.items():
        summary[k] = {"n": len(rs), "passed": sum(r["rc"] == 0 for r in rs),
                      **{f"median_{x}": median(rs, x) for x in ("seconds", "rest_s", *JUDGED, "tree_cpu_s")},
                      "peak_gib": max((r["peak_gib"] for r in rs), default=None),
                      "max_lean": max((r["max_lean"] for r in rs), default=None),
                      "largest_gib": max((r["largest_gib"] for r in rs), default=None),
                      "reserved_gib": gib_of(rs[0]["reserved"]) if rs else None}
    a = summary["A"]
    verdict = {f"B_{x}_within_{WITHIN}_A": bool(a[f"median_{x}"] and summary["B"][f"median_{x}"] is not None
                                                and summary["B"][f"median_{x}"] <= WITHIN * a[f"median_{x}"])
               for x in JUDGED}
    for k in "BC":
        s = summary[k]
        verdict[f"{k}_peak_within_reservation"] = bool(s["reserved_gib"] and s["peak_gib"] is not None
                                                       and s["peak_gib"] <= s["reserved_gib"])
    return summary, verdict


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rounds", type=int, default=3, help="rounds of the three arms after the warm-up")
    ap.add_argument("--first-lemma", type=int, default=0, help="the first arm's lemma, past those the node has built")
    ap.add_argument("--low-gb", type=int, default=RESERVE.BUILD_HEAVY_GIB,
                    help="arm C's reservation, in GB: by default a heavy build's figure read as GB, less than its GiB, so "
                    "C's build runs fewer jobs")
    ap.add_argument("--base", type=Path, required=True, help="the directory holding arm A's tools, each of TOOLS the base "
                    "has as base-FILE (in a run, its inputs)")
    ap.add_argument("--out", type=Path, required=True, help="where guard.json and each arm's log and series go (in a run, "
                    "$OUT)")
    a = ap.parse_args(argv)
    out_dir, base = a.out, a.base
    out_dir.mkdir(parents=True, exist_ok=True)
    common = load("guard_common", ROOT / "tools/verity/check/common.py")
    ls = load("guard_lean_slot", ROOT / "tools/verity/check/lean_slot.py")
    if not ls.SLOTS.exists() or "build" not in ls.parse(ls.SLOTS.read_text()):
        say("this node has no `build` Lean pool, so no kept records tree to measure on")
        return 3
    say(f"cores {common.cores()}; waiting for a `build` slot")
    name, i, fd = ls.wait("build", by="lean", log=lambda s: None, or_any=False)
    tree = common.cache_dir() / "lean-records" / f"{name}-{i}"
    say(f"holding a `build` slot; its tree is lean-records/{name}-{i} (kept: {tree.is_dir()})")
    here = {f: (LEAN / f).read_bytes() for f in TOOLS}
    based = {f: (base / f"base-{f}").read_bytes() if (base / f"base-{f}").is_file() else None for f in TOOLS}
    original = MODULE.read_text()
    if not original.rstrip().endswith(END):
        say(f"{MODULE} doesn't end in {END!r}")
        return 3
    plan = [("warm", here)] + [(kind, based if kind == "A" else here) for _ in range(a.rounds) for kind in "ABC"]
    recs = []
    try:
        for n, (kind, tools) in enumerate(plan):
            recs.append(arm(n, kind, a.first_lemma + n, tree, tools, original, out_dir, a.low_gb))
            (out_dir / "guard.json").write_text(json.dumps({"cores": common.cores(), "arms": recs}, indent=1) + "\n")
    finally:
        for f in TOOLS:
            (LEAN / f).write_bytes(here[f])
        MODULE.write_text(original)
        os.close(fd)
    summary, verdict = summarize(recs)
    (out_dir / "guard.json").write_text(json.dumps({"cores": common.cores(), "summary": summary, "verdict": verdict,
                                                    "arms": recs}, indent=1) + "\n")
    for k in "ABC":
        say(f"{k}: {summary[k]}")
    say(f"verdict {verdict}")
    return 0 if all(r["rc"] == 0 for r in recs) and all(verdict.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
