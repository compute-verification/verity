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
    NAME=COMMIT:named      the head as it is: `--move-commit` for each move of `--below`..STAGED it does not hold, and
                           `--onto STAGED` (with `@PARENT`, started when PARENT has ended)
    NAME=COMMIT:named-attrs  as `:named`, with the vLLM lists' attribute lines of the commit below the first move (their
                           old paths) given through `core.attributesFile`
    NAME=COMMIT:pins       restack.py's pins step alone, on a merge made by hand
    NAME=COMMIT:given=RESULT  nothing is run: COMMIT was crossed by someone else as RESULT, which its children are crossed over

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
import tempfile
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
# The vLLM lists' merge driver in either layout, set in the same command as restack.py (a clone's own config names the
# moved path only; in a checkout of the moved tree this runs the same script).
ENV.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="merge.vllm-allowlist.driver",
           GIT_CONFIG_VALUE_0='f=adapters/vllm/tests/lint/merge_allowlist.py; [ -f "$f" ] || '
                              'f=integrations/vllm/tests/lint/merge_allowlist.py; python3 "$f" %O %A %B %P')
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


def restack(tag: str, args: list[str], env: dict | None = None) -> tuple[dict | None, float, int]:
    """Run restack.py with ``args``; its JSON's one record (None when it printed none), the seconds and its exit."""
    t = time.time()
    before = kept_files()
    with open(os.path.join(OUT, f"{tag}.json"), "w") as o, open(os.path.join(OUT, f"{tag}.err"), "w") as e:
        rc = subprocess.run([sys.executable, RS, *args, "--kept", "--dry-run", "--json"], stdout=o, stderr=e, env=env or ENV).returncode
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
    pm = r.get("plain") or {}
    say(f"{tag}: plain merge {pm.get('status')} ({len(pm.get('files') or [])} files) {' '.join((pm.get('files') or [])[:40])}")
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
    if r and r.get("status") == "conflicted" and r.get("step") == "base" and r.get("crossed"):
        # A stop at a later move's `base` names no commit: cross the moves it passed again, alone, for the head a mend starts from.
        args = [commit, "--main", main, "--no-pins"]
        for m in r["crossed"]:
            args += ["--move-commit", m]
        p, psecs, _ = restack(f"{name}.partial", args)
        if p and p.get("result"):
            rec["partial"], rec["partial_seconds"] = p["result"], round(psecs)
            say(f"{name}: crossed over {[m[:12] for m in r['crossed']]} alone it is {p['result']} ({p.get('status')}); the stop is "
                f"its merge with {r.get('base')}")
    if r and r.get("result") and r.get("status") in ("clean", "locks-stale", "pins-stale"):
        rec["result"] = r["result"]
        rec["facts"] = facts(name, commit, r["result"], base1 if holds(base1, commit) else None, main, main)
        say(f"{name}: the result {r['result'][:12]} holds the commit crossed: {rec['facts']['holds_commit']}, holds the staged head: "
            f"{rec['facts']['holds_main']}; its change before (against the commit below the move): {rec['facts'].get('before')}; "
            f"after (against the staged head): {rec['facts']['after']}; files with conflict markers among those changed: "
            f"{len(rec['facts']['conflict_markers'])}")
    return rec


def old_attributes(name: str, below: str) -> dict:
    """ENV with `core.attributesFile` naming a file of the vLLM lists' lines of ``below``'s `.gitattributes` (the paths before
    the move): restack.py merges with `git merge-tree` in this clone, whose own `.gitattributes` names the moved paths only,
    so a list at its old path would get a text merge at the `base` step."""
    lines = [x for x in git("show", f"{below}:.gitattributes").splitlines()
             if x.startswith("integrations/vllm/") and "merge=vllm-allowlist" in x]
    path = os.path.join(tempfile.mkdtemp(prefix="cir223-attrs-"), "attributes")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    say(f"{name}: core.attributesFile holds {len(lines)} line(s) of {below[:12]}'s .gitattributes: {' | '.join(lines)}")
    return {**ENV, "GIT_CONFIG_COUNT": "2", "GIT_CONFIG_KEY_1": "core.attributesFile", "GIT_CONFIG_VALUE_1": path}


def named(name: str, commit: str, main: str, moves: list[str], attrs_from: str | None = None) -> dict:
    """A head as it is (not merged with the commit below the move), each move named: `--move-commit M... --onto MAIN`."""
    args = [commit, "--main", main, "--onto", main]
    for m in moves:
        if not holds(m, commit):
            args += ["--move-commit", m]
    r, secs, rc = restack(name, args, old_attributes(name, attrs_from) if attrs_from else None)
    rec = {"name": name, "kind": "named", "commit": commit, "seconds": round(secs), "exit": rc, "restack": r}
    if r and r.get("status") == "conflicted" and r.get("step") == "base" and r.get("crossed"):
        # A stop at a later move's `base` names no commit: cross the moves it passed again, alone, for the head a mend starts from.
        pargs = [commit, "--main", main, "--no-pins"]
        for m in r["crossed"]:
            pargs += ["--move-commit", m]
        p, psecs, _ = restack(f"{name}.partial", pargs)
        if p and p.get("result"):
            rec["partial"], rec["partial_seconds"] = p["result"], round(psecs)
            say(f"{name}: crossed over {[m[:12] for m in r['crossed']]} alone it is {p['result']} ({p.get('status')}); the stop is "
                f"its merge with {r.get('base')}")
    if r and r.get("result") and r.get("status") in ("clean", "locks-stale", "pins-stale"):
        rec["result"] = r["result"]
        rec["facts"] = facts(name, commit, r["result"], None, main, main)
        say(f"{name}: the result {r['result'][:12]} holds the commit crossed: {rec['facts']['holds_commit']}, holds the staged head: "
            f"{rec['facts']['holds_main']}; after (against the staged head): {rec['facts']['after']}")
    return rec


def pins(name: str, commit: str, main: str) -> dict:
    """restack.py's own pins step on a merge made by hand: its `regen_pins` on ``commit``, and the commit it would add."""
    sys.path.insert(0, os.path.dirname(RS))
    import restack as R  # the staged tree's
    os.environ.update({k: v for k, v in ENV.items() if k.startswith("GIT_")})
    t = time.time()
    tree, output, fresh = R.regen_pins(commit, sys.executable, None)
    result = commit if tree is None else R.commit_tree(tree, [commit], f"restack: regenerate stale pins ({R.PINS} --regen)\n", None)
    rec = {"name": name, "kind": "pins", "commit": commit, "seconds": round(time.time() - t), "result": result,
           "restack": {"status": "clean" if fresh else "pins-stale", "step": "pins", "result": result},
           "pins_changed": tree is not None, "pins_fresh": fresh, "pins_output": output}
    rec["facts"] = facts(name, commit, result, None, main, main)
    say(f"{name}: pins.py --regen on {commit[:12]}: every pin fresh after: {fresh}; it changed the tree: {tree is not None}; the "
        f"result {result[:12]} holds the staged head: {rec['facts']['holds_main']}; against the staged head: {rec['facts']['after']}; "
        f"files with conflict markers among those changed: {len(rec['facts']['conflict_markers'])}")
    for line in output[-8:]:
        say(f"{name}:    | {line[:300]}")
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


def child_steps(name: str, commit: str, parent: dict, main: str, all_moves: list[tuple[str, str, list[str]]]) -> dict:
    lp, lc = parent.get("result"), parent["commit"]
    rec = {"name": name, "kind": "child-steps", "commit": commit, "parent": parent["name"], "parent_result": lp, "moves": []}
    if not lp:
        say(f"{name}: its parent {parent['name']} has no crossed result; not tried")
        return {**rec, "skipped": "parent has no result"}
    tip, moves, t = commit, [m for m in all_moves if not holds(m[0], lc)], time.time()
    # The parent's own script steps, one a move, oldest first: those on the first-parent line from its result back to its head
    # (a parent that is itself a child names its own parent's steps in their trailers, so they are not found by the move's id).
    own = git("log", "--first-parent", "--reverse", "--format=%H", "--grep=^Restack-move: ", lp, "--not", lc).split()
    if len(own) != len(moves):
        say(f"{name}: {parent['name']}'s crossing has {len(own)} script steps for {len(moves)} moves; stopped")
        return {**rec, "seconds": 0, "stopped": f"{len(own)} script steps for {len(moves)} moves"}
    for n, ((m, script, margs), s) in enumerate(zip(moves, own), 1):
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
    ap.add_argument("--below", help="a commit below every move to cross (default: the first spec's commit)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    main_sha = git("rev-parse", f"{a.main}^{{commit}}")
    specs = []
    for spec in a.specs:
        name, _, rest = spec.partition("=")
        rest, _, how = rest.partition(":")
        commit, _, parent = rest.partition("@")
        after = [x["name"] for x in specs if parent and not how and x["parent"] == parent and not x["how"]]
        specs.append({"name": name, "commit": git("rev-parse", f"{commit}^{{commit}}"), "parent": parent or None, "how": how,
                      "after": after[-1] if after else None})
    all_moves = moves_of(main_sha, git("rev-parse", f"{a.below}^{{commit}}") if a.below else specs[0]["commit"])
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
            # one child of a parent at a time, in the order given (the first pays the check of the parent's script steps, the
            # next read it); a spec with a parent and a `how` only waits for the parent to end
            if s.get("after"):
                events[s["after"]].wait()
            with gate:
                if s["how"].startswith("given="):
                    # a head someone else crossed and pushed: its result is taken as it is, as its children's parent result
                    got = git("rev-parse", f"{s['how'][6:]}^{{commit}}")
                    rec = {"name": s["name"], "kind": "given", "commit": s["commit"], "result": got, "given": True,
                           "restack": {"status": "given"}}
                    say(f"{s['name']}: given: {s['commit'][:12]} crossed as {got}; it holds that head: {holds(s['commit'], got)}; "
                        f"holds main: {holds(main_sha, got)}; holds each move: {[holds(m[0], got) for m in all_moves]}")
                elif s["how"] == "named":
                    rec = named(s["name"], s["commit"], main_sha, [m[0] for m in all_moves])
                elif s["how"] == "named-attrs":
                    rec = named(s["name"], s["commit"], main_sha, [m[0] for m in all_moves], attrs_from=base1)
                elif s["how"] == "pins":
                    rec = pins(s["name"], s["commit"], main_sha)
                elif not s["parent"]:
                    rec = plain(s["name"], s["commit"], main_sha, base1)
                elif s["how"] == "onto":
                    rec = child_onto(s["name"], s["commit"], done[s["parent"]], main_sha)
                else:
                    rec = child_steps(s["name"], s["commit"], done[s["parent"]], main_sha, all_moves)
        except Exception as e:  # one spec's failure is its record, not the run's
            say(f"{s['name']}: the driver failed: {e!r}")
            rec = {"name": s["name"], "commit": s["commit"], "driver_error": repr(e)}
        done[s["name"]] = rec
        if rec.get("result") and not rec.get("given"):
            # each result is bundled as it is made (`NAME.bundle`), for a reader who doesn't wait for the run's end
            try:
                ref = f"cir223/{s['name']}"
                git("branch", "-f", ref, rec["result"])
                skip = {main_sha, *(x["commit"] for x in specs), *(d["result"] for d in list(done.values()) if d.get("given"))}
                skip.discard(rec["result"])
                git("bundle", "create", "-q", os.path.join(OUT, f"{s['name']}.bundle"), ref, *[f"^{c}" for c in sorted(skip)])
                say(f"{s['name']}: bundled as {s['name']}.bundle: {rec['result']}")
            except Exception as e:
                say(f"{s['name']}: its own bundle failed: {e!r}")
        events[s["name"]].set()
        with LOCK:
            with open(os.path.join(OUT, "results.json"), "w") as f:
                json.dump(done, f, indent=1)
        return rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
        list(pool.map(run, specs))
    by_commit: dict[str, list[dict]] = {}
    for rec in done.values():
        if rec.get("result"):
            by_commit.setdefault(rec["commit"], []).append(rec)
    for commit, recs in by_commit.items():
        for other in recs[1:]:
            same = git("rev-parse", f"{recs[0]['result']}^{{tree}}") == git("rev-parse", f"{other['result']}^{{tree}}")
            say(f"TREES {recs[0]['name']} {recs[0]['result'][:12]} and {other['name']} {other['result'][:12]} (both from "
                f"{commit[:12]}): {'the same tree' if same else 'DIFFERENT TREES: ' + shortstat(recs[0]['result'], other['result'])}")
    for rec in done.values():
        if rec.get("kind") == "child-steps" and rec.get("result"):
            say(f"STACK {rec['name']} {rec['result'][:12]} holds its parent {rec['parent']}'s result {rec['parent_result'][:12]}: "
                f"{holds(rec['parent_result'], rec['result'])}; holds its own head {rec['commit'][:12]}: "
                f"{holds(rec['commit'], rec['result'])}; holds main {main_sha[:12]}: {holds(main_sha, rec['result'])}")
    refs, excl = [], {main_sha, *(s["commit"] for s in specs)}
    excl |= {rec["result"] for rec in done.values() if rec.get("given")}
    for name, rec in done.items():
        tips = {"": rec.get("result")}
        r = rec.get("restack") or {}
        if rec.get("partial"):
            tips["-partial"] = rec["partial"]
        elif not rec.get("result") and r.get("steps"):
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
