#!/usr/bin/env python3
"""`cir223/cross.py --main STAGED [--jobs N] SPEC...`: try the crossing of open heads over a staged layout move with the
checkout's own `tools/move/restack.py`, on a node, in a clone that holds every commit named. It changes no branch of any
remote and pushes nothing: each result is a commit of this clone, bundled into `$OUT/crossed.bundle`.

A SPEC is one of

    NAME=COMMIT            the command every owner runs: `restack.py COMMIT --main STAGED --kept --dry-run --json`
    NAME=COMMIT@PARENT     a stack's child, crossed after PARENT in this run so that its result holds PARENT's crossed
                           result: for each move of STAGED in order, the move is the script step of PARENT's own crossing
                           (the commit whose message says `Restack-move: MOVE`), and the last merge is onto PARENT's result
    NAME=COMMIT@PARENT:onto  the child with `restack.py COMMIT --main STAGED --onto PARENT'S RESULT`, as it is

Specs with no parent run side by side, at most `--jobs` at a time; a child starts when its parent has ended.

`$OUT`: `summary.txt` (what happened, in lines), `results.json` (one record a spec), `NAME*.json` and `NAME*.err`
(restack.py's own output), `NAME.before.txt` and `NAME.after.txt` (the change's files before and after), `crossed.bundle`.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import re
import resource
import subprocess
import sys
import threading
import time

RS = "tools/move/restack.py"
KEPT = os.path.expanduser("~/.cache/verity-check/restack-generated")
MOVE_LINE = re.compile(r"^(Move|Layout-move): (tools/move/[\w.-]+\.toml)((?: --no-lock)?)$", re.M)
LINE_SCRIPT = {"Move": "tools/move/rename.py", "Layout-move": "tools/move/layout.py"}
ENV = dict(os.environ)
if not subprocess.run(["git", "config", "--get", "user.name"], capture_output=True, text=True).stdout.strip():
    # a node's clone has no identity of its own, and restack.py's commits need one
    ENV.update(GIT_AUTHOR_NAME="circuits", GIT_AUTHOR_EMAIL="circuits@verity.invalid",
               GIT_COMMITTER_NAME="circuits", GIT_COMMITTER_EMAIL="circuits@verity.invalid")
OUT = os.environ.get("OUT") or "out"
LOCK = threading.Lock()
T0 = time.time()


def say(text: str) -> None:
    with LOCK:
        for line in text.splitlines() or [""]:
            line = f"[+{time.time() - T0:6.0f}s] {line}"
            print(line, flush=True)
            with open(os.path.join(OUT, "summary.txt"), "a") as f:
                f.write(line + "\n")


def git(*args: str, ok: tuple[int, ...] = (0,)) -> str:
    p = subprocess.run(["git", *args], capture_output=True, text=True, env=ENV, errors="replace")
    if p.returncode not in ok:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout.strip()


def holds(a: str, b: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b], env=ENV).returncode == 0


def kept_files() -> set[str]:
    return {os.path.basename(p) for p in glob.glob(os.path.join(KEPT, "*.json"))}


def restack(tag: str, args: list[str]) -> tuple[dict | None, float, int]:
    """Run restack.py with ``args``; its JSON's one record (None when it printed none), the seconds and its exit."""
    t = time.time()
    before = kept_files()
    with open(os.path.join(OUT, f"{tag}.json"), "w") as o, open(os.path.join(OUT, f"{tag}.err"), "w") as e:
        rc = subprocess.run([sys.executable, RS, *args, "--kept", "--dry-run", "--json"], stdout=o, stderr=e, env=ENV).returncode
    secs = time.time() - t
    try:
        with open(os.path.join(OUT, f"{tag}.json")) as f:
            r = json.load(f)[0]
    except (OSError, ValueError, IndexError):
        r = None
    with open(os.path.join(OUT, f"{tag}.err")) as f:
        err = f.read().strip().splitlines()
    new = sorted(kept_files() - before)
    read = sum("as checked before" in line for line in err)
    say(f"{tag}: restack.py {' '.join(args)}\n{tag}: exit {rc} after {secs:.0f} s; kept checks read {read}, kept files new on "
        f"the node since it started {len(new)}; largest child so far {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024} MB")
    for line in err[-12:]:
        say(f"{tag}: err | {line[:400]}")
    if r is None:
        say(f"{tag}: no JSON record")
        return None, secs, rc
    say(f"{tag}: status {r.get('status')} step {r.get('step')} result {r.get('result')} head {r.get('head')}")
    say(f"{tag}: move {r.get('move')} base {r.get('base')} onto {r.get('onto')} crossed {r.get('crossed')}")
    say(f"{tag}: plain merge {r['plain']['status']} ({len(r['plain']['files'])} files) {' '.join(r['plain']['files'][:40])}")
    for f in r.get("files", []):
        say(f"{tag}: file {f}")
    if r.get("error"):
        say(f"{tag}: error {r['error'][:3000]}")
    for s in r.get("steps", []):
        say(f"{tag}: step {s['step']} commit {s['commit']} restored {s.get('restored')}")
        for line in s.get("output", []):
            say(f"{tag}:    | {line[:400]}")
    return r, secs, rc


def shortstat(a: str, b: str) -> str:
    return " ".join(git("diff", "--shortstat", a, b).split()) or "no difference"


def facts(name: str, commit: str, result: str, before_from: str | None, after_from: str, main: str) -> dict:
    """What the result holds and how its change reads against ``after_from``, beside the change before the crossing."""
    f = {"holds_commit": holds(commit, result), "holds_main": holds(main, result), "holds_after_from": holds(after_from, result),
         "after": shortstat(after_from, result)}
    with open(os.path.join(OUT, f"{name}.after.txt"), "w") as o:
        o.write(git("diff", "--stat=240", "--stat-count=400", "-M", after_from, result) + "\n")
    if before_from:
        f["before"] = shortstat(before_from, commit)
        with open(os.path.join(OUT, f"{name}.before.txt"), "w") as o:
            o.write(git("diff", "--stat=240", "--stat-count=400", "-M", before_from, commit) + "\n")
    changed = [p for p in git("diff", "--name-only", "--no-renames", after_from, result).splitlines() if p]
    marked = []
    for i in range(0, len(changed), 200):
        got = git("grep", "-l", "-E", "^(<<<<<<<|>>>>>>>) ", result, "--", *changed[i:i + 200], ok=(0, 1))
        marked += [line.split(":", 1)[1] for line in got.splitlines() if ":" in line]
    f["changed_files"], f["conflict_markers"] = len(changed), marked
    return f


def moves_of(main: str, head: str) -> list[tuple[str, str, list[str]]]:
    log = git("log", "--topo-order", "--reverse", "--no-merges", "--format=%H%x00%B%x1e", main, f"^{head}")
    found = []
    for rec in filter(None, (x.strip() for x in log.split("\x1e"))):
        commit, body = rec.split("\x00", 1)
        if m := MOVE_LINE.search(body):
            found.append((commit, LINE_SCRIPT[m[1]], ["--map", m[2], *m[3].split()]))
    return found


def plain(name: str, commit: str, main: str, base1: str) -> dict:
    r, secs, rc = restack(name, [commit, "--main", main])
    rec = {"name": name, "kind": "plain", "commit": commit, "seconds": round(secs), "exit": rc, "restack": r}
    if r and r.get("result") and r.get("status") in ("clean", "locks-stale", "pins-stale"):
        rec["result"] = r["result"]
        rec["facts"] = facts(name, commit, r["result"], base1 if holds(base1, commit) else None, main, main)
        say(f"{name}: the result {r['result'][:12]} holds the commit crossed: {rec['facts']['holds_commit']}, holds the staged head: "
            f"{rec['facts']['holds_main']}; its change before (against the commit below the move): {rec['facts'].get('before')}; "
            f"after (against the staged head): {rec['facts']['after']}; files with conflict markers among those changed: "
            f"{len(rec['facts']['conflict_markers'])}")
    return rec


def child_onto(name: str, commit: str, parent: dict, main: str) -> dict:
    lp = parent.get("result")
    if not lp:
        say(f"{name}: its parent {parent['name']} has no crossed result; not tried")
        return {"name": name, "kind": "child-onto", "commit": commit, "parent": parent["name"], "skipped": "parent has no result"}
    r, secs, rc = restack(name, [commit, "--main", main, "--onto", lp])
    rec = {"name": name, "kind": "child-onto", "commit": commit, "parent": parent["name"], "parent_result": lp,
           "seconds": round(secs), "exit": rc, "restack": r}
    if r and r.get("result") and r.get("status") in ("clean", "locks-stale", "pins-stale"):
        rec["result"] = r["result"]
        rec["facts"] = facts(name, commit, r["result"], parent["commit"] if holds(parent["commit"], commit) else None, lp, main)
        say(f"{name}: the result {r['result'][:12]} holds the parent's crossed result: {rec['facts']['holds_after_from']}; "
            f"before (child against parent): {rec['facts'].get('before')}; after (crossed child against crossed parent): "
            f"{rec['facts']['after']}")
    return rec


def child_steps(name: str, commit: str, parent: dict, main: str) -> dict:
    lp, lc = parent.get("result"), parent["commit"]
    rec = {"name": name, "kind": "child-steps", "commit": commit, "parent": parent["name"], "parent_result": lp, "moves": []}
    if not lp:
        say(f"{name}: its parent {parent['name']} has no crossed result; not tried")
        return {**rec, "skipped": "parent has no result"}
    tip, moves, t = commit, moves_of(main, lc), time.time()
    for n, (m, script, margs) in enumerate(moves, 1):
        steps = git("log", "--format=%H", f"--grep=^Restack-move: {m}", lp, "--not", lc, main).split()
        if not steps:
            say(f"{name}: move {n} of {len(moves)}, {m[:12]}: no script step of {parent['name']}'s crossing names it; stopped")
            return {**rec, "seconds": round(time.time() - t), "stopped": f"no script step for {m}"}
        s = steps[-1]
        args = [tip, "--move-commit", s, "--script", script, *[f"--script-arg={a}" for a in margs], "--main", main]
        args += ["--onto", lp] if n == len(moves) else ["--no-pins"]
        r, secs, rc = restack(f"{name}.x{n}", args)
        rec["moves"].append({"move": m, "script_step": s, "seconds": round(secs), "exit": rc, "restack": r})
        rec["restack"] = r
        if not (r and r.get("result") and r.get("status") in ("clean", "locks-stale", "pins-stale", "up-to-date")):
            return {**rec, "seconds": round(time.time() - t), "stopped": f"move {n}"}
        tip = r["result"]
    rec.update(result=tip, seconds=round(time.time() - t))
    rec["facts"] = facts(name, commit, tip, lc if holds(lc, commit) else None, lp, main)
    say(f"{name}: the crossed child {tip[:12]} holds the parent's crossed result {lp[:12]}: {rec['facts']['holds_after_from']}; "
        f"holds the commit crossed: {rec['facts']['holds_commit']}; holds the staged head: {rec['facts']['holds_main']}; before "
        f"(child against parent): {rec['facts'].get('before')}; after (crossed child against crossed parent): {rec['facts']['after']}; "
        f"files with conflict markers among those changed: {len(rec['facts']['conflict_markers'])}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--main", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    main_sha = git("rev-parse", f"{a.main}^{{commit}}")
    specs = []
    for spec in a.specs:
        name, _, rest = spec.partition("=")
        commit, _, parent = rest.partition("@")
        parent, _, how = parent.partition(":")
        specs.append({"name": name, "commit": git("rev-parse", f"{commit}^{{commit}}"), "parent": parent or None, "how": how})
    all_moves = moves_of(main_sha, specs[0]["commit"])
    base1 = git("rev-parse", f"{all_moves[0][0]}^") if all_moves else main_sha
    uv = subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip()
    say(f"staged {main_sha}; the commit below the first move {base1}; moves {[m[0][:12] for m in all_moves]}")
    say(f"python {sys.executable} {sys.version.split()[0]}; {uv}; {git('--version')}; cpus {os.cpu_count()}, usable "
        f"{len(os.sched_getaffinity(0))}; host {os.uname().nodename}; jobs {a.jobs}")
    names = sorted(set(re.findall(r"\bmerge=([\w.-]+)", open(".gitattributes").read())))
    defined = [n for n in names if git("config", "--get", f"merge.{n}.driver", ok=(0, 1))]
    say(f"restack.py sha256 {subprocess.run(['sha256sum', RS], capture_output=True, text=True).stdout[:16]}; merge drivers "
        f".gitattributes names {len(names)}, defined {len(defined)}; kept checks on the node now {len(kept_files())}")
    for key in ("merge.vllm-allowlist.driver", "merge.fixture-map.driver"):
        say(f"{key}: {git('config', '--get', key, ok=(0, 1))[:200]}")
    done: dict[str, dict] = {}
    events = {s["name"]: threading.Event() for s in specs}
    gate = threading.Semaphore(a.jobs)

    def run(s: dict) -> dict:
        try:
            if s["parent"]:
                events[s["parent"]].wait()
            with gate:
                if not s["parent"]:
                    rec = plain(s["name"], s["commit"], main_sha, base1)
                elif s["how"] == "onto":
                    rec = child_onto(s["name"], s["commit"], done[s["parent"]], main_sha)
                else:
                    rec = child_steps(s["name"], s["commit"], done[s["parent"]], main_sha)
        except Exception as e:  # one spec's failure is its record, not the run's
            say(f"{s['name']}: the driver failed: {e!r}")
            rec = {"name": s["name"], "commit": s["commit"], "driver_error": repr(e)}
        done[s["name"]] = rec
        events[s["name"]].set()
        with LOCK:
            with open(os.path.join(OUT, "results.json"), "w") as f:
                json.dump(done, f, indent=1)
        return rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
        list(pool.map(run, specs))
    refs, excl = [], {main_sha, *(s["commit"] for s in specs)}
    for name, rec in done.items():
        tips = {"": rec.get("result")}
        r = rec.get("restack") or {}
        if not rec.get("result") and r.get("steps"):
            tips["-partial"] = r["steps"][-1]["commit"]
        for suffix, tip in tips.items():
            if tip and tip not in excl:
                ref = f"cir223/{name}{suffix}"
                git("branch", "-f", ref, tip)
                refs.append(ref)
    if refs:
        git("bundle", "create", "-q", os.path.join(OUT, "crossed.bundle"), *refs, *[f"^{c}" for c in sorted(excl)])
        say(f"bundle: {len(refs)} refs, {os.path.getsize(os.path.join(OUT, 'crossed.bundle')) // 1024} KiB: {' '.join(refs)}")
    for name, rec in done.items():
        r = rec.get("restack") or {}
        say(f"RESULT {name}: {rec.get('kind')} status {r.get('status') or rec.get('skipped') or rec.get('driver_error')} "
            f"step {r.get('step')} result {rec.get('result')} stopped {rec.get('stopped')} seconds {rec.get('seconds')} "
            f"files {' '.join(r.get('files') or [])}")
    say(f"total {time.time() - T0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
