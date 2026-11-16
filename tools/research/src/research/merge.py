"""`research merge REF [--into BRANCH] [--tool check] [-m MESSAGE] [--push] [--dry-run]`: the merge gate.
`research merge --train A B C [--on MACHINE --project P] [--command CMD] [--push]`: several refs as one checked candidate (`train`).
`research merge --train --prepare BRANCH [--onto BASE] N@SHA ...`: build a train and push it as BRANCH for its check (`prepare`).
A train (built either way) is refused when it loses lines from its base that no ref removes in a commit of its own, as a merge
in a ref's history that dropped them would make it (`reverted`), but for the paths of `--allow-removal PATH`.
`research merge --sweep [--into BRANCH] [--remote origin] [--dry-run]`: close the pull requests that landed and delete their branches
(`sweep`).
`research merge --mirror [SHA]`: push SHA (default main's tip) to node 1's bare repository's main, as --push does (`push_mirror`).
`research merge --canary [--tool check] [--json]`: the gates' self-test, run daily: attempts the merge gate, the push guard and the
train gate must refuse and a control they must admit, in scratch stores (`canary`).
`research merge --label-landed [REV]`: write the `landed` label on each run the landings after REV name (`label_landed`).

Run in the checkout of the target branch.  It merges the commit REF names (`git merge --no-ff`) only if the evidence store holds
a passing attempt of the tool (default `check`) for that exact commit: `state` done, rc 0, source commit REF's, tree clean, as
the attempt's own record says (the index only finds it); a run of the command the target's declaration of the tool names
(`Tool.command`, in a check slot or not, followed by nothing but check.py's own flags: `unlike`); and its own result, of a kind
that declaration produces (`Tool.produces`: `check/v1` for check; an attempt with no result, or whose record or result can't
be read, passes no gate) and not in `REFUSED_KINDS` (the Lean fast path's `lean-fast/v1`, which skips the kernel replay), with
the verdict passed and no step listed twice.  A sharded check (`check.py --shard K/N`, one Attempt per shard, its result a
`check-shard/v1`) passes only through the Attempt of `check.py --combine RUN...` on the same commit, and only when the gate,
reading the shards itself (`shard_set`), finds them a passing check of that clean commit (one plan, K = 1..N once each, the
commit's own `tools/check/`, every part once, every job its `ci.toml` puts on the gate, no step failed) and the combine's
steps theirs; a shard alone passes nothing.  A job off the gate (drive's policy, policy.toml's `off_gate`) is required of no
check, whole or sharded, and a recorded check doesn't run it: it runs after landing, in drive's after-landing run, and its red
there names its culprit.  With a usable remote the catalog's attempts are refreshed first, so an attempt
recorded on another machine counts.  Refused as well: a
target checkout that is not on BRANCH or has tracked changes, and a REF the target has moved past (the target's tip is not an
ancestor of REF) unless merging REF leaves REF's own tree (`tree_identical`: what the target gained is in REF already, as when a
train checked on the train ahead of it lands after that one), since otherwise the merged tree is not the one that was checked;
merge the target into the branch and check the new tip.  The steps a change requires are read from the target's declaration and
the diff from its tip either way.  The merge commit carries a `Check: <attempt id>` trailer.  Exit 0 merged (or would, with
--dry-run), 1 a failure past the gate (git's merge, the push, the sweep, a train's offending ref), 2 usage, 3 refused
(`research.gates.refuse`: the gate, the push guard, a train or its build).

With --verdicts (opt-in) the same refusals hold before any verdict, and in place of a passing attempt of REF, REF is
admitted when the target's `check.py --plan REF --packs-only --base HEAD` (`PLAN`, `verdict_gate`) shows every part it
requires passed under its key at REF, each by a pass that a recorded check's verdict pack holds: an attempt of the tool that
ended, ran the declared command whole or as a shard, and lists that pack, by sha256, among its run files (`pack_refusal`); or an
after-landing run's (`check.py --deferred`, the declared command with `--deferred STEPS`, recorded as no tool: `post_landing`).
Every Lake package's audit and every step the change requires (`required_steps`) is among those parts, so a skip passes
nothing; a part with no key is never passed, a build step (`plan.IMPLIED`) is `implied` only by the cited passes of the parts
that read what it builds (its `implied_by`), and a pass kept only in a machine's cache is cited by nothing.
The merge commit carries `Verdicts: <attempt id>...`.  It takes no --train or --push: the push guard takes only a passing
attempt of the tree.  The plan's keys hold the machine it runs on (Python, the base image, the tools on the PATH), so it finds
the checks' passes only run on the kind of machine they ran on: `--plan-on M1,M2` (drive.toml's `on`) makes it on each of those
check nodes in turn, a short `research run` (`check.py --plan --on`), from packs gathered where the gate runs, and admits on the
first plan fully passed.

After a history rewrite (`commitmap`: an accepted `commit-map/v1` in the store), an attempt of an old commit counts for its image when
the rewrite kept the tree, and REF is refused when it holds a commit the rewrite replaced: merging old history would bring it back.

After a merge lands, and on its own with --sweep, every open pull request whose head is on the remote's copy of BRANCH is closed
with a comment naming the commit it landed in, and every open pull request whose base branch is on it is retargeted onto BRANCH.
A closed pull request's head branch is deleted from the remote unless an open pull request still uses it, or it moved past the
head that landed (GitHub's "delete head branch" fires only on merges made on GitHub, and trains merge locally).
The remote's branch, fetched first, is the truth, never the checkout. With --push, a merge that lands is pushed to the remote
(never forced), what the tool's declaration runs after a landing (`Tool.after_land`) is started, detached, on the landed commit,
and then it is swept; without it the sweep runs before anyone pushes, finds nothing the merge landed, and
`research merge --sweep` after the push does it. Whoever lands a train passes --push. Once origin has main, the
same commit goes to vy-nebius-cpu-4's bare repository, whose main its builds check out (`vy-lean-sandbox checkout`): a plain
push over the ssh `research run --on` uses, bounded by MIRROR_TIMEOUT_S. One that fails is a line with the command that reruns
it, and the merge stands; $RESEARCH_MERGE_MIRROR names another machine, and empty skips it.
These after-landing steps (the spec DM, the mirror, `after_land`, the Lean cache publish when a Lean package changed,
`lean_publish`, and one line per landing in the merge thread: what it merged
and its `Checked:` run, `report_landings`; a `landed` label on each run a landing's trailer names, `label_landed`) follow the remote's main, not the push: AFTER_LAND_REF on the
remote is the main they last ran for, and every `research merge` that can change something (all but --dry-run, --mirror,
--canary and --guard-push) catches them up, once, for whatever landed past it since, however it was pushed (`caught_up`).

A train's ref may be pinned: `N@SHA` (pull request N; quote `#N@SHA`, or the shell reads a comment) or `BRANCH@SHA` is fetched
from the remote and refused unless its head there is still SHA, so what merges is the head that was reviewed (`pinned`); the gate
reads each again when the tip lands (`unstacked`). It reads no ready mark and no grant, since the train's own check is the gate;
the owner's ready mark is what admits a PR to a train (`research queue`), not part of this gate.
A ref whose merge onto the train so far conflicts only on import lines (two PRs adding modules to one root import list) keeps
both sides, ours first; a conflict on any other line refuses it, and so does a JSON file whose lines git merged into something
that doesn't parse (`_merge`). Generated files never keep merged text: every merge runs the tree's own
`tools/check/pins.py --merged` before its commit (`regenerate`), which runs `lake update` in each Lake package both sides'
Lake changes reach, re-points its dependency bundle's pin, regenerates every stale pin, and resolves a conflict in any of those
files; the merge commit names what it rewrote in a `Regenerated:` trailer, which `reverted` reads as removals the train means.

A push that lands a change to what a guarantee says (a `lean-audit.json`'s guarantees or the definitions they read) sends the
spec-change DM of what changed, and nothing waits on its reader (`spec_alert`). A DM that fails is printed and goes to #agent-alerts,
and SPEC_ALERTED_REF stays where it was, so every later `research merge` sends it again and `--sweep` fails until it goes.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .gates import FINDING, OK, REFUSED, git_env, refuse, worktrees_shared

PROG = "research merge"


class Refused(Exception):
    pass


def _git(repo: Path, *args: str, check: bool = True, lazy: bool = True) -> subprocess.CompletedProcess:
    env = git_env() if lazy else git_env(GIT_NO_LAZY_FETCH="1")
    with worktrees_shared(repo, env) if args[:1] == ("fetch",) else contextlib.nullcontext():
        r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, errors="surrogateescape", env=env)
    if check and r.returncode:
        raise Refused(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    return r


REFUSED_KINDS = frozenset({"lean-fast/v1"})
SLOT_WRAP = ("python3", "tools/check/slot.py")
SLOT_FLAGS, SLOT_OPTS = ("--priority",), ("--rank", "--by", "--twin", "--wait-max", "--run-max")
CHECK_FLAGS = ("--no-cache", "--circuit-cache", "--keep-going")


def invocation(argv, command) -> tuple[str, object]:
    if not command:
        return "refused", "the target's declaration of the tool names no command to hold it to"
    if not (isinstance(argv, list) and all(isinstance(x, str) for x in argv)):
        return "refused", "its record has no argv"
    run = argv
    if tuple(argv[:2]) == SLOT_WRAP:
        opts, seen = argv[2:], set()
        while opts[:1] != ["--"] and opts[:1] and opts[0] not in seen and (
                opts[0] in SLOT_FLAGS or (opts[0] in SLOT_OPTS and len(opts) > 1)):
            seen.add(opts[0])
            opts = opts[1:] if opts[0] in SLOT_FLAGS else opts[2:]
        run = opts[1:] if opts[:1] == ["--"] else argv
    if run[:len(command)] != list(command):
        return "refused", f"its command `{shlex.join(argv)}` isn't the target's declared `{shlex.join(command)}`"
    tail = run[len(command):]
    if tail[:1] == ["--combine"]:
        ids = tail[1:]
        return ("combine", ids) if ids and not any(x.startswith("-") for x in ids) else (
            "refused", f"its command gives the declared one `{shlex.join(tail)}`, which check.py doesn't take")
    rest = [x for x in tail if x not in CHECK_FLAGS]
    if "--verdicts-in" in rest[:-1]:
        i = rest.index("--verdicts-in")
        rest = rest[:i] + rest[i + 2:]
    mode: tuple[str, object] = ("check", None)
    if "--shard" in rest[:-1] and (m := re.fullmatch(r"(?:auto:)?([1-9]\d*)/([1-9]\d*)", rest[rest.index("--shard") + 1])):
        i = rest.index("--shard")
        mode, rest = ("shard", (int(m.group(1)), int(m.group(2)))), rest[:i] + rest[i + 2:]
    if rest and not (rest[0] == "--cores" and len(rest) == 2 and rest[1].isdigit()):
        return "refused", f"its command gives the declared one `{shlex.join(rest)}`, which check.py doesn't take"
    return mode


def post_landing(argv, command) -> str | None:
    if not (isinstance(argv, list) and argv.count("--deferred") == 1) or (i := argv.index("--deferred")) + 1 >= len(argv):
        return None
    steps = argv[i + 1]
    return steps if re.fullmatch(r"[a-z0-9-]+(,[a-z0-9-]+)*", steps) and invocation(argv[:i] + argv[i + 2:], command)[0] == "check" \
        else None


def unlike(argv, command) -> str | None:
    mode, arg = invocation(argv, command)
    if mode == "shard":
        return (f"it is shard {arg[0]}/{arg[1]} of a sharded check, which passes no gate alone: `check.py --record --combine` "
                "its set")
    if mode == "combine":
        return "it combines shards, which only the gate's own reading of them admits"
    return arg if mode == "refused" else None


SHARD_KIND = "check-shard/v1"


def _ci_toml(repo: Path, commit: str) -> dict | None:
    import tomllib

    from .store import tools_registry

    path = tools_registry.resolve("tools/check/ci.toml",
                                  lambda p: _git(repo, "cat-file", "-e", f"{commit}:{p}", check=False).returncode == 0)
    r = _git(repo, "show", f"{commit}:{path or 'tools/check/ci.toml'}", check=False)
    try:
        return tomllib.loads(r.stdout) if r.returncode == 0 else None
    except (tomllib.TOMLDecodeError, TypeError):
        return None


def declared(repo: Path, commit: str) -> list[dict] | None:
    jobs = doc.get("job", []) if (doc := _ci_toml(repo, commit)) is not None else None
    if not isinstance(jobs, list) or not all(isinstance(t, dict) and isinstance(t.get("name"), str) for t in jobs):
        return None
    return jobs


def shard_plans(repo: Path, commit: str) -> dict[int, list[list[str]]] | None:
    plans = doc.get("plan", []) if (doc := _ci_toml(repo, commit)) is not None else None
    if not isinstance(plans, list):
        return None
    out = {}
    for p in plans:
        shards = p.get("shards") if isinstance(p, dict) else None
        if not (isinstance(shards, list) and all(isinstance(s, list) and all(isinstance(x, str) for x in s) for s in shards)):
            return None
        out.setdefault(len(shards), shards)
    return out


def declared_jobs(repo: Path, commit: str) -> list[str] | str:
    from . import policy

    pol = policy.at(repo, commit)
    if isinstance(pol, str):
        return pol
    jobs = [t["name"] for t in declared(repo, commit) or ()
            if t.get("gate", True) is not False and t["name"] not in pol.off_gate]
    if not jobs:
        return f"{commit[:12]}'s tools/check/ci.toml can't be read for the jobs a check runs"
    return jobs


def _read(store, run_id: str, tool: str | None, commit: str | None):
    try:
        doc = store.get_attempt(run_id)
    except Exception as e:  # noqa: BLE001  (a guard never fails open)
        return f"its attempt can't be read ({type(e).__name__}: {e})"
    if not isinstance(doc, dict):
        return "its attempt can't be read"
    st, src = doc.get("status") or {}, doc.get("source") or {}
    said = {"tool": (doc.get("tool") or {}).get("name"), "commit": src.get("commit"), "dirty": src.get("dirty"),
            "state": st.get("state"), "rc": st.get("rc")}
    want = {"tool": tool, "commit": commit, "dirty": False, "state": "done", "rc": 0}
    if off := [f"{k} {said[k]}" for k, v in want.items() if v is not None and said[k] != v]:
        return f"its record says {', '.join(off)}, not the index's passing check"
    art = (doc.get("outputs") or {}).get("result")
    try:
        return doc, art, store.get_manifest(art) if art else None
    except Exception as e:  # noqa: BLE001  (a guard never fails open)
        return f"its result can't be read ({type(e).__name__}: {e})"


def shard_set(store, repo: Path, commit: str, ids: list[str], tool: str = "check",
              command: tuple[str, ...] = ()) -> tuple[list[dict] | None, str | None]:
    if not ids or len(set(ids)) != len(ids):
        return None, "a set of shards names each shard's attempt once" if ids else "no shard is named"
    config = _git(repo, "rev-parse", "--verify", "-q", f"{commit}:tools/check", check=False).stdout.strip()
    if not config:
        return None, f"{commit[:12]} has no tools/check/ to hold the shards' config to"
    jobs = declared_jobs(repo, commit)
    if isinstance(jobs, str):
        return None, jobs
    shards: dict[int, tuple[str, dict, list[dict]]] = {}
    plans = set()
    for rid in ids:
        got = _read(store, rid, tool, commit)
        if isinstance(got, str):
            return None, f"{rid}: {got}"
        doc, art, result = got
        mode, arg = invocation(doc.get("argv"), command)
        if mode != "shard":
            return None, f"{rid}: {arg if mode == 'refused' else 'it is not a shard (`check.py --shard K/N`)'}"
        kind = result.kind if result else None
        if kind != SHARD_KIND:
            return None, f"{rid}: its result is {f'a {kind}' if kind else 'missing'}, not a {SHARD_KIND}"
        meta = result.meta if isinstance(result.meta, dict) else {}
        sh = meta.get("shard") if isinstance(meta.get("shard"), dict) else {}
        steps = meta.get("steps", [])
        if meta.get("run_id") != rid:
            return None, f"{rid}: its result names run {meta.get('run_id')}"
        if (meta.get("validation") or {}).get("status") != "passed":
            return None, f"{rid}: its result's verdict is {(meta.get('validation') or {}).get('status')}, not passed"
        if why := twice(steps):
            return None, f"{rid}: {why}"
        if (sh.get("k"), sh.get("n")) != arg:
            return None, f"{rid}: its result is shard {sh.get('k')}/{sh.get('n')}, its command {arg[0]}/{arg[1]}"
        if sh.get("commit") != commit or sh.get("config") != config:
            return None, (f"{rid}: its result checked commit {sh.get('commit')} with config {sh.get('config')}, not {commit[:12]}'s "
                          f"tools/check/ ({config[:12]})")
        if not all(isinstance(sh.get(f), list) and all(isinstance(p, str) for p in sh[f]) for f in ("parts", "all")):
            return None, f"{rid}: its result doesn't list its parts and every part"
        if bad := [s["step"] for s in steps if s.get("status") not in ("passed", "skipped")]:
            return None, f"{rid}: {', '.join(bad)} didn't pass"
        ran = [p for s in steps for p in (s.get("parts") or [])]
        if not all(isinstance(p, str) for p in ran) or sorted(ran) != sorted(sh["parts"]):
            return None, f"{rid}: its steps ran {', '.join(map(str, ran)) or 'nothing'}, not its shard's parts"
        if arg[0] in shards:
            return None, f"shard {arg[0]}/{arg[1]} is named twice ({shards[arg[0]][0]}, {rid})"
        shards[arg[0]] = (rid, sh, steps)
        plans.add((arg[1], json.dumps(sh["all"]), (sh.get("plan") or {}).get("digest") if isinstance(sh.get("plan"), dict) else None))
    if len(plans) != 1:
        return None, ("the shards are of different plans (their N, every part they say an unsharded check covers, or their auto "
                      "plan's digest)")
    (n, _, _), = plans
    if missing := [k for k in range(1, n + 1) if k not in shards]:
        return None, f"shard {', '.join(f'{k}/{n}' for k in missing)} is not among them"
    every = shards[1][1]["all"]
    covered = [p for k in sorted(shards) for p in shards[k][1]["parts"]]
    if len(set(every)) != len(every):
        return None, "their list of every part names one twice"
    if sorted(covered) != sorted(every):
        again = sorted({p for p in covered if covered.count(p) > 1})
        return None, (f"their parts aren't every part once (twice: {', '.join(again) or 'none'}; in no shard: "
                      f"{', '.join(sorted(set(every) - set(covered))) or 'none'}; not a part: "
                      f"{', '.join(sorted(set(covered) - set(every))) or 'none'})")
    if uncovered := [j for j in jobs if not any(p == j or p.startswith(j + ":") for p in every)]:
        return None, f"no shard runs {', '.join(uncovered)}, which {commit[:12]}'s ci.toml puts on the gate"
    names = {p.partition(":")[0] for p in every}
    common = [[s["step"] for s in shards[k][2] if not s.get("parts") and s["step"] not in names] for k in sorted(shards)]
    if any(sorted(c) != sorted(common[0]) for c in common):
        return None, "the shards ran different steps outside their parts"
    out = [{"step": s, "status": "passed" if all(x.get("status") == "passed" for k in shards for x in shards[k][2] if x["step"] == s)
            else "skipped", "shards": [shards[k][0] for k in sorted(shards)]} for s in common[0]]
    part = {p: {"step": p, "status": s.get("status"), "shard": k, "run_id": shards[k][0], "seconds": s.get("seconds")}
            for k in sorted(shards) for s in shards[k][2] for p in (s.get("parts") or [])}
    return out + [part[p] for p in every], None


def declaration(repo: Path, tool: str):
    """Read from the checkout, not from the commit being merged; None when it has none."""
    from .store import tools_registry

    try:
        return tools_registry.from_tree(tool, repo)
    except Exception:  # noqa: BLE001  (a declaration that can't be read is none)
        return None


def produced(repo: Path, tool: str) -> frozenset[str]:
    """None without a declaration, so that no attempt passes."""
    return frozenset(getattr(declaration(repo, tool), "produces", None) or ())


def declared_command(repo: Path, tool: str) -> tuple[str, ...]:
    """None without a declaration, so that no attempt passes."""
    return tuple(getattr(declaration(repo, tool), "command", None) or ())


def refused(store, run_id: str, kinds: frozenset[str], tool: str | None = None, commit: str | None = None,
            command: tuple[str, ...] | None = None, repo: Path | None = None) -> str | None:
    got = _read(store, run_id, tool, commit)
    if isinstance(got, str):
        return got
    doc, art, result = got
    mode, ids = invocation(doc.get("argv"), command)
    if mode != "combine" and (why := unlike(doc.get("argv"), command)):
        return why
    kind = result.kind if result else None
    if kind in REFUSED_KINDS:
        return f"its result is a {kind}, which no merge accepts"
    if kind not in kinds:
        return ((f"its result is a {kind}" if kind else f"its result {art} has no kind" if art else "it published no result") +
                f", and the target's declaration of the tool produces {', '.join(sorted(kinds)) or 'nothing'}")
    meta = result.meta if isinstance(result.meta, dict) else {}
    if meta.get("run_id") != run_id:
        return f"its result names run {meta.get('run_id')}, not this one"
    validation = meta.get("validation")
    verdict = validation.get("status") if isinstance(validation, dict) else None
    if verdict != "passed":
        return f"its result's verdict is {verdict}, not passed"
    if why := twice(meta.get("steps", [])):
        return why
    if mode == "combine":
        if repo is None or commit is None:
            return "it combines shards, which this gate reads only in the target's checkout, for a commit"
        steps, why = shard_set(store, repo, commit, ids, tool or "check", command or ())
        if why:
            return f"its shards aren't a passing check: {why}"
        seen = sorted((s["step"], s.get("status")) for s in meta.get("steps", []))
        if seen != sorted((s["step"], s["status"]) for s in steps):
            return "its steps aren't the ones its shards make"
    return None


def twice(steps) -> str | None:
    if not (isinstance(steps, list) and all(isinstance(s, dict) and isinstance(s.get("step"), str) for s in steps)):
        return "its result has steps this gate can't read"
    names = [s["step"] for s in steps]
    if again := next((n for i, n in enumerate(names) if n in names[:i]), None):
        return f"its result lists step {again} more than once"
    return None


def passing(store, commit: str, tool: str = "check", *, kinds: frozenset[str], command: tuple[str, ...], tree: str | None = None,
            maps=None, repo: Path | None = None) -> list[str]:
    rows = store.index.sql("SELECT id FROM attempts WHERE tool = ? AND source_commit = ? AND source_dirty = 0 AND state = 'done' "
                           "AND rc = 0 ORDER BY start_utc DESC", (tool, commit))
    found = [(r[0], commit) for r in rows]
    for old in (maps.equivalent(commit)[1:] if maps else []):
        rows = store.index.sql("SELECT id, manifest_json FROM attempts WHERE tool = ? AND source_commit = ? AND source_dirty = 0 "
                               "AND state = 'done' AND rc = 0 ORDER BY start_utc DESC", (tool, old))
        for rid, doc in rows:
            recorded = (json.loads(doc).get("source") or {}).get("tree") if doc else None
            if tree is None or recorded in (None, tree):
                found.append((rid, old))
    return [rid for rid, c in found if refused(store, rid, kinds, tool, c, command, repo) is None]


def tree_identical(repo: Path, commit: str) -> bool:
    r = _git(repo, "merge-tree", "--write-tree", "HEAD", commit, check=False)
    return r.returncode == 0 and r.stdout.split("\n", 1)[0].strip() == _git(repo, "rev-parse", f"{commit}^{{tree}}").stdout.strip()


LOCK = "lean-audit.json"


def _run_files(store, run_id: str) -> str | None:
    out = (store.get_attempt(run_id) or {}).get("outputs") or {}
    return out.get("run_files") or out.get("run_record")


def _under_run(path: str) -> str:
    if "/lean-audit/" in path:
        return "lean-audit/" + path.split("/lean-audit/", 1)[1]
    return path if path.startswith("lean-audit/") else f"lean-audit/{path}"


AUDIT = "lean-audit/audit.json"
REPAIRED = frozenset({"tools/verity/lean/unread-caps.json", "tests/test_lean_packages.py"})


class Unreadable(Exception):
    """A check regenerated lock records, or its Lean report can't say whether it did, and the records can't be read."""


def regenerated_records(store, run_id: str, dest: Path | None = None) -> dict[str, tuple[str, Path | None]] | None:
    if not (art := _run_files(store, run_id)):
        return None
    try:
        rows = {f["path"]: f["sha256"] for f in store.listing(art)}
    except Exception:  # noqa: BLE001  (no listing: the caller reads the live run, else holds)
        return None
    if AUDIT not in rows:
        return None
    with tempfile.TemporaryDirectory() as d:
        try:
            audit = json.loads((store.fetch(art, Path(d), paths=[AUDIT]) / AUDIT).read_text())
        except (OSError, ValueError) as e:
            raise Unreadable(f"{run_id}'s {AUDIT} in the store can't be read ({type(e).__name__})") from e
    want = _regenerated_in(run_id, audit)
    if missing := sorted(pkg for pkg, p in want.items() if p not in rows):
        raise Unreadable(f"{run_id}'s {AUDIT} says it regenerated {', '.join(missing)}'s lock, which its files in the store lack")
    got = store.fetch(art, dest, paths=sorted(want.values())) if dest is not None and want else None
    return {pkg: (rows[p], (got / p) if got is not None else None) for pkg, p in want.items()}


def _regenerated_in(run_id: str, audit) -> dict[str, str]:
    regen = audit.get("regenerated") if isinstance(audit, dict) else None
    fixed = audit.get("repaired") if isinstance(audit, dict) else None
    if not isinstance(regen or {}, dict) or not isinstance(fixed or {}, dict):
        raise Unreadable(f"{run_id}'s {AUDIT} has a `regenerated` or `repaired` that isn't an object")
    if odd := sorted(set(fixed or {}) - REPAIRED):
        raise Unreadable(f"{run_id}'s {AUDIT} says it repaired {', '.join(odd)}, which no lock run repairs (`REPAIRED`)")
    return {str(pkg): _under_run(str(p)) for pkg, p in (regen or {}).items()} | {
        str(f): _under_run(str(p)) for f, p in (fixed or {}).items()}


def fetch_run(rid: str, names: list[str], cwd: Path | None = None, refresh: bool = False) -> tuple[Path, dict] | None:
    also = [x for n in names for x in ("--also", n)]
    r = subprocess.run([sys.executable, "-m", "research", "status", rid, "--json", *(["--refresh"] if refresh else []), *also],
                       cwd=cwd, capture_output=True, text=True)
    try:
        row = json.loads(r.stdout)["runs"][0]
        return Path(row["dir"]), row
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def live_records(rid: str, fetch=fetch_run) -> dict[str, tuple[str, Path | None]] | None:
    if (got := fetch(rid, [AUDIT])) is None:
        return None
    try:
        audit = json.loads((got[0] / AUDIT).read_text())
    except OSError:
        return None
    except ValueError as e:
        raise Unreadable(f"{rid}'s {AUDIT} on its node can't be parsed ({e})") from e
    if not (want := _regenerated_in(rid, audit)):
        return {}
    got = fetch(rid, sorted(set(want.values())))
    if missing := sorted(pkg for pkg, p in want.items() if got is None or not (got[0] / p).is_file()):
        raise Unreadable(f"{rid}'s {AUDIT} says it regenerated {', '.join(missing)}'s lock, which can't be fetched from its "
                         "run dir on its node")
    return {pkg: (hashlib.sha256((got[0] / p).read_bytes()).hexdigest(), got[0] / p) for pkg, p in want.items()}


def lean_report(store, rid: str, fetch=fetch_run):
    if (art := _run_files(store, rid)) and AUDIT in {f["path"] for f in store.listing(art)}:
        with tempfile.TemporaryDirectory() as d:
            return json.loads((store.fetch(art, Path(d), paths=[AUDIT]) / AUDIT).read_text())
    if (got := fetch(rid, [AUDIT])) is not None:
        return json.loads((got[0] / AUDIT).read_text())
    return None


def bookkeeping_warnings(store, rid: str, fetch=fetch_run) -> list[str]:
    try:
        audit = lean_report(store, rid, fetch)
        warned = audit.get("bookkeeping") if isinstance(audit, dict) else None
        return [str(w) for w in warned] if isinstance(warned, list) else []
    except Exception as e:  # noqa: BLE001  (warnings only: no report, no warnings)
        print(f"research merge: lock run {rid}'s bookkeeping warnings unread ({type(e).__name__}: {e}); its lock commit "
              f"lists none", file=sys.stderr)
        return []


def _blob_sha256(repo: Path, commit: str, path: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{path}"], capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else ""


def _ran_on(store, rid: str, commit: str, fetch) -> bool:
    rows = store.index.sql("SELECT source_commit, source_dirty FROM attempts WHERE id = ?", (rid,))
    if rows:
        return rows[0][0] == commit and not rows[0][1]
    got = fetch(rid, []) if fetch is not None else None
    return got is not None and got[1].get("tree_sha") == commit


def regenerated_parent(repo: Path, commit: str, store, *, tool: str, kinds: frozenset[str], command: tuple[str, ...],
                       maps=None, deferred: str | None = None, fetch=None) -> str | None:
    parents = _git(repo, "rev-list", "--parents", "-n", "1", commit).stdout.split()[1:]
    if len(parents) != 1:
        return None
    parent = parents[0]
    files = _git(repo, "diff", "--name-only", "--no-renames", parent, commit).stdout.split()
    said = {f.strip() for f in _git(repo, "log", "-1", "--format=%(trailers:key=Regenerated,valueonly,separator=%x2C)",
                                     commit).stdout.replace("\n", ",").split(",") if f.strip()}
    if not files or set(files) != said or not all(os.path.basename(f) == LOCK or f in REPAIRED for f in files):
        return None
    if deferred is not None:
        checked = re.findall(r"^Checked: (\S+)", _git(repo, "log", "-1", "--format=%B", commit).stdout, re.M)
        ids = [deferred] if checked == [deferred] and _ran_on(store, deferred, parent, fetch) else []
    else:
        tree = _git(repo, "rev-parse", f"{parent}^{{tree}}").stdout.strip()
        ids = passing(store, parent, tool, kinds=kinds, command=command, tree=tree, maps=maps, repo=repo)
    lock = {f if f in REPAIRED else os.path.dirname(f) or ".": _blob_sha256(repo, commit, f) for f in files}
    for rid in ids:
        try:
            recs = regenerated_records(store, rid)
            if recs is None and fetch is not None:
                recs = live_records(rid, fetch)
        except Exception:  # noqa: BLE001  (`Unreadable`, or the store failing: nothing to vouch with)
            continue
        if recs and {(pkg or "."): sha for pkg, (sha, _) in recs.items()} == lock:
            return rid
    return None


def _admissible(repo: Path, ref: str, into: str, store, tool: str, log, remote: str):
    commit = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()
    head = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
    if head != into:
        raise Refused(f"the checkout is on {head or 'a detached HEAD'}, not {into}")
    if _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise Refused(f"the {into} checkout has uncommitted changes")
    if _git(repo, "merge-base", "--is-ancestor", "HEAD", commit, check=False).returncode and not tree_identical(repo, commit):
        raise Refused(f"{into} has moved past {commit[:12]}: merge {into} into the branch, then record `{tool}` on its new tip")
    if store.remote is not None:
        try:
            store.refresh(attempts_only=True)
        except Exception as e:  # noqa: BLE001  (the local catalog still answers; say it may be behind)
            log(f"research merge: refresh from the remote failed ({type(e).__name__}: {e}); using the local catalog")
    from .commitmap import Maps  # noqa: PLC0415

    maps = Maps.load(store)
    if old := maps.old_history(repo, commit, "HEAD"):
        raise Refused(f"{commit[:12]} holds {len(old)} commit(s) a history rewrite replaced (e.g. {old[0][:12]}): merging it would bring the "
                      f"old history back; move the branch onto the rewritten one with `research git remap` and push that")
    if why := unstacked(repo, commit, "HEAD", remote, log):
        raise Refused("; ".join(why))
    return commit, maps


def gate(repo: Path, ref: str, into: str, store, tool: str = "check", log=print, remote: str = "origin") -> tuple[str, str]:
    commit, maps = _admissible(repo, ref, into, store, tool, log, remote)
    tree = _git(repo, "rev-parse", f"{commit}^{{tree}}").stdout.strip()
    kinds, command = produced(repo, tool), declared_command(repo, tool)
    ids = passing(store, commit, tool, kinds=kinds, command=command, tree=tree, maps=maps, repo=repo)
    if not ids and (via := regenerated_parent(repo, commit, store, tool=tool, kinds=kinds, command=command, maps=maps)):
        ids = [via]
    if not ids:
        rows = store.index.sql("SELECT id, state, rc, source_dirty FROM attempts WHERE tool = ? AND source_commit = ?", (tool, commit))
        why = {r[0]: refused(store, r[0], kinds, tool, commit, command, repo) for r in rows if r[1] == "done" and r[2] == 0 and not r[3]}
        seen = "; ".join(f"{r[0]} {r[1]} rc={r[2]}{' dirty' if r[3] else ''}{f' refused: {why[r[0]]}' if why.get(r[0]) else ''}"
                         for r in rows) or "none"
        raise Refused(f"no passing `{tool}` attempt for {commit} in the evidence store (attempts of it: {seen}); "
                      f"record one: `uv run python tools/check/check.py --record` on that commit")
    need = required_steps(repo, "HEAD", commit, tool)
    if not need:
        return commit, ids[0]
    why = []
    for rid in ids:
        missing = [s for s in need if step_status(store, rid, s) != "passed"]
        if not missing:
            return commit, rid
        why.append(f"{rid}: {', '.join(f'{s} {step_status(store, rid, s)}' for s in missing)}")
    raise Refused(f"{commit[:12]} changes {', '.join(sorted(set(p for ps in need.values() for p in ps)))}, where `{tool}` must have run "
                  f"{', '.join(need)}; no attempt did ({'; '.join(why)}). Record it with what those steps need (for `check`: "
                  f"`uv run python tools/check/check.py --record --on MACHINE` sends the agreement inputs)")


PLAN_KIND = "check-plan/v1"
PLAN = ("uv", "run", "--locked", "--all-packages", "--extra", "torch-cpu", "python", "tools/check/check.py", "--plan")
PACK_STATES = ("done", "failed", "cancelled")
PACKS = ("verdicts.tar.gz", "verdicts-reused.tar.xz")


def run_plan(repo: Path, commit: str, base: str, also: tuple[str, ...] = (), on: str | None = None) -> dict:
    r = subprocess.run([*PLAN, commit, "--json", "--packs-only", "--base", base, *(x for rid in also for x in ("--from", rid)),
                        *(["--on", on] if on else [])], cwd=repo, capture_output=True, text=True)
    try:
        doc = json.loads(r.stdout)
    except ValueError:
        doc = None
    if not isinstance(doc, dict):
        raise Refused(f"`check.py --plan {commit[:12]}{f' --on {on}' if on else ''}` printed no plan (exit {r.returncode}): "
                      f"{(r.stderr or r.stdout).strip()[-400:]}")
    return doc


def pack_refusal(store, rid: str, sha256: str, tool: str, command: tuple[str, ...]) -> str | None:
    try:
        doc = store.get_attempt(rid)
    except Exception as e:  # noqa: BLE001  (a guard never fails open)
        return f"its attempt can't be read ({type(e).__name__}: {e})"
    if not isinstance(doc, dict):
        return "its attempt can't be read"
    mode, why = invocation(doc.get("argv"), command)
    if mode == "refused" and post_landing(doc.get("argv"), command):
        mode = "deferred"
    if (name := (doc.get("tool") or {}).get("name")) != tool and mode != "deferred":
        return f"it is a `{name}` attempt, not `{tool}`"
    if (state := (doc.get("status") or {}).get("state")) not in PACK_STATES:
        return f"it is {state}, not ended"
    if mode not in ("check", "shard", "deferred"):
        return why if mode == "refused" else "it combines shards, and holds no pack of its own"
    out = doc.get("outputs") or {}
    for art in dict.fromkeys(a for a in (out.get("run_files"), out.get("run_record")) if a):
        try:
            rows = {f["path"]: f.get("sha256") for f in store.listing(art)}
        except Exception as e:  # noqa: BLE001  (no listing: nothing to cite)
            return f"its run files can't be listed ({type(e).__name__}: {e})"
        if any(rows.get(name) == sha256 for name in PACKS):
            return None
    return f"its run files hold no {' or '.join(PACKS)} of that sha256"


def plan_refusal(doc: dict, store, commit: str, tree: str, tool: str, command: tuple[str, ...], need: dict) -> tuple[list[str], str | None]:
    if doc.get("kind") != PLAN_KIND or doc.get("commit") != commit or doc.get("tree") != tree:
        return [], (f"the plan is a {doc.get('kind')} of {str(doc.get('commit'))[:12]} (tree {str(doc.get('tree'))[:12]}), not a "
                    f"{PLAN_KIND} of the clean {commit[:12]}")
    parts = doc.get("parts")
    if not isinstance(parts, list) or not parts or not all(isinstance(p, dict) and isinstance(p.get("part"), str) for p in parts):
        return [], "the plan lists no parts this gate can read"
    by = {p["part"]: p for p in parts}
    if missing := [s for s in need if s not in by]:
        return [], f"the plan has no part {', '.join(missing)}, which {commit[:12]}'s change requires"
    wanted = [p for p in parts if p.get("required", True) or p["part"] in need or p["part"].startswith("lean-audit")]
    resting = {n for p in wanted if p.get("state") == "implied" and isinstance(p.get("implied_by"), list) for n in p["implied_by"]}
    wanted += [p for p in parts if p["part"] in resting and p not in wanted]
    judged: dict[tuple[str, str], str | None] = {}
    cited, bad = set(), []
    for p in wanted:
        if p.get("state") == "implied":
            rests = p.get("implied_by")
            if not isinstance(rests, list) or not rests or not all(isinstance(n, str) and n in by for n in rests):
                bad.append(f"{p['part']} implied (by no part this plan lists)")
            elif short := [n for n in rests if by[n].get("state") != "passed"]:
                bad.append(f"{p['part']} implied ({', '.join(short[:3])} not passed)")
            continue
        if p.get("state") != "passed":
            bad.append(f"{p['part']} {p.get('state')}" + (f" ({p['why']})" if p.get("why") else ""))
            continue
        why = "no pass found in a recorded check's pack"
        for f in p.get("found") or []:
            if not isinstance(f, dict) or not f.get("attempt") or not f.get("pack_sha256"):
                continue
            k = (f["attempt"], f["pack_sha256"])
            if k not in judged:
                judged[k] = pack_refusal(store, *k, tool, command)
            if judged[k] is None:
                cited.add(f["attempt"])
                why = None
                break
            why = f"{f['attempt']}: {judged[k]}"
        if why:
            bad.append(f"{p['part']} ({why})")
    if bad:
        return [], f"{len(bad)} of {len(wanted)} parts have no cited pass under their key: {'; '.join(bad[:8])}" + (
            f"; and {len(bad) - 8} more" if len(bad) > 8 else "")
    return sorted(cited), None


def verdict_gate(repo: Path, ref: str, into: str, store, tool: str = "check", log=print, remote: str = "origin",
                 plan=None, also: tuple[str, ...] = (), on: tuple[str, ...] = ()) -> tuple[str, list[str]]:
    commit, _maps = _admissible(repo, ref, into, store, tool, log, remote)
    tree = _git(repo, "rev-parse", f"{commit}^{{tree}}").stdout.strip()
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    command, need = declared_command(repo, tool), required_steps(repo, "HEAD", commit, tool)
    whys = []
    for m in on or (None,):
        try:
            doc = (plan or run_plan)(repo, commit, head, also, on=m)
        except Refused as e:
            whys.append(str(e))
            continue
        cited, why = plan_refusal(doc, store, commit, tree, tool, command, need)
        if not why:
            return commit, cited
        whys.append(f"{f'on {m}: ' if m else ''}{why}")
    raise Refused(f"{commit[:12]}'s plan isn't fully passed by recorded checks: {' | '.join(whys)}")


ZERO = "0" * 40
HOOK = """#!/bin/sh
# installed by `research merge`: a push that moves {into} needs a passing `{tool}` attempt of the tree it lands; other pushes pass
lines=$(cat)
printf '%s\\n' "$lines" | awk '$3 == "refs/heads/{into}" {{ f = 1 }} END {{ exit !f }}' || exit 0
if [ ! -x {python} ] || [ ! -d {src} ]; then
  echo "research merge: push guard: {python} or {src} is gone; refusing the push to {into} (re-run research merge to reinstall)" >&2
  exit 1
fi
printf '%s\\n' "$lines" | PYTHONPATH={src} exec {python} -m research merge --guard-push --into {into} --tool {tool}
"""


def unchecked_pushes(repo: Path, lines: list[str], store, into: str = "main", tool: str = "check") -> list[str]:
    bad = []
    kinds, command = produced(repo, tool), declared_command(repo, tool)
    for line in lines:
        f = line.split()
        if len(f) != 4 or f[2] != f"refs/heads/{into}" or f[1] == ZERO:
            continue
        tip = f[1]
        tree = _git(repo, "rev-parse", f"{tip}^{{tree}}").stdout.strip()
        parents = _git(repo, "rev-list", "--parents", "-n", "1", tip).stdout.split()[1:]
        if not any(_git(repo, "rev-parse", f"{x}^{{tree}}").stdout.strip() == tree
                   and passing(store, x, tool, kinds=kinds, command=command, tree=tree, repo=repo) for x in [tip, *parents[1:]]) \
                and not regenerated_parent(repo, tip, store, tool=tool, kinds=kinds, command=command):
            bad.append(tip)
    return bad


CANARY_SCHEMA = "research/merge-canary/v1"
CANARY = (("no-result", "a passing check that published no result", True),
          ("wrong-kind", "a passing check whose result is a bench-result/v1", True),
          ("refused-kind", "a passing check whose result is the Lean fast path's lean-fast/v1", True),
          ("unreadable-record", "a passing check whose attempt record can't be read", True),
          ("disagreeing-record", "a check the index lists as passing whose record says rc 1", True),
          ("wrong-command", "a passing result of `bash -c '<the declared command> || true'`", False),
          ("failed-verdict", "a run of the declared command whose result's verdict is failed", False),
          ("repeated-step", "a run of the declared command whose result lists a step as passed, then as failed", True),
          ("control", "a run of the declared command whose passing result is of a kind the declaration produces", True))


def canary(repo: Path, tool: str = "check") -> dict:
    from .jobs import trains
    from .runs import new_run_id
    from .store import tools_registry
    from .store.local import LocalStore

    module = (tools_registry.locate(tool, repo) or tools_registry.REGISTRY.get(tool, tool).rpartition(":"))[0]
    rel = Path(*module.split(".")).with_suffix(".py")
    if not (repo / rel).is_file():
        raise Refused(f"this checkout has no {rel}, its declaration of `{tool}`: run the canary in the target's checkout")
    kinds = sorted(produced(repo, tool))
    command = list(declared_command(repo, tool))
    candidate = "a change no step of the declaration covers\n"
    ident = ("-c", "user.name=research merge canary", "-c", "user.email=canary@research.invalid", "-c", "commit.gpgsign=false",
             "-c", "core.hooksPath=/dev/null")
    cases = []
    with tempfile.TemporaryDirectory(prefix="research-merge-canary-") as tmp:
        scratch = Path(tmp) / "repo"
        (scratch / rel).parent.mkdir(parents=True)
        (scratch / rel).write_bytes((repo / rel).read_bytes())
        _git(Path(tmp), "init", "-q", "-b", "main", str(scratch))
        _git(scratch, "add", "-A")
        _git(scratch, *ident, "commit", "-q", "-m", "the canary's target")
        base = _git(scratch, "rev-parse", "HEAD").stdout.strip()
        _git(scratch, "checkout", "-q", "-b", "candidate")
        (scratch / "candidate.txt").write_text(candidate)
        _git(scratch, "add", "candidate.txt")
        _git(scratch, *ident, "commit", "-q", "-m", "the canary's candidate")
        commit = _git(scratch, "rev-parse", "HEAD").stdout.strip()
        _git(scratch, "checkout", "-q", "main")
        for case, what, trained in CANARY:
            store = LocalStore(Path(tmp) / f"store-{case}", config_path=Path(tmp) / "no-remote.toml")
            kind = {"no-result": None, "wrong-kind": "bench-result/v1", "refused-kind": "lean-fast/v1"}.get(
                case, kinds[0] if kinds else "check/v1")
            rid = new_run_id()
            steps = [{"step": "suites", "status": "passed"}] + [{"step": "suites", "status": "failed"}] * (case == "repeated-step")
            meta = {"schema": "research/result/v0.1", "kind": kind, "run_id": rid, "steps": steps,
                    "validation": {"status": "failed" if case == "failed-verdict" else "passed", "checker": f"{PROG} --canary"}}
            art = store.put_artifact(kind, meta) if kind else None
            record = {
                "schema": "research/attempt/v0.1", "id": rid, "tool": {"name": tool, "version": None}, "derivation": "drv:" + "0" * 64,
                "reuse_key": None, "argv": ["bash", "-c", f"{shlex.join(command)} || true"] if case == "wrong-command" else command,
                "params": {}, "key_params": [], "inputs": {}, "conditions": {}, "key_conditions": [],
                "source": {"commit": commit, "dirty": False}, "closure": None, "execution": {}, "cost": None,
                "status": {"state": "done", "rc": 0}, "outputs": {"result": art} if art else {}, "telemetry": None}
            store.put_attempt(record)
            if case in ("unreadable-record", "disagreeing-record"):
                store.attempt_path(rid).unlink()
                store.attempt_path(rid).write_text("{" if case == "unreadable-record" else
                                                   json.dumps(record | {"status": {"state": "done", "rc": 1}}))
            try:
                gate(scratch, commit, "main", store, tool, log=lambda *_: None)
                at_gate = "admitted"
            except Refused as e:
                at_gate = f"refused: {e}"
            pushed = unchecked_pushes(scratch, [f"refs/heads/main {commit} refs/heads/main {base}"], store, "main", tool)
            at_train = None
            if trained and tool == "check":
                try:
                    on_disk = json.loads(store.attempt_path(rid).read_text())
                except ValueError:
                    on_disk = None
                svc = trains.TrainService(
                    github=trains.FakeGitHub(commits={base: {"parents": [], "files": {trains.TOOL_PY: (repo / trains.TOOL_PY).read_text()}},
                                                      commit: {"parents": [base], "files": {"candidate.txt": candidate}}},
                                             refs={"main": base}),
                    store=trains.FakeStore(attempts={rid: on_disk} if isinstance(on_disk, dict) else {},
                                           manifests={art: {"kind": kind, "meta": meta}} if art else {}))
                why = svc.gate(trains.Train(id=1, prs=[], base=base, tip=commit, job=None, land=False, requirements={},
                                            by="canary", run_id=rid))
                at_train = f"refused: {why}" if why else "admitted"
            want = "admitted" if case == "control" else "refused"
            seen = [at_gate, "refused" if pushed else "admitted", at_train]
            cases.append({"case": case, "attempt": what, "want": want, "gate": at_gate, "push_guard": seen[1], "train_gate": at_train,
                          "ok": all(s.split(":")[0] == want for s in seen if s is not None)})
    return {"schema": CANARY_SCHEMA, "tool": tool, "produces": kinds, "command": command, "cases": cases,
            "ok": all(c["ok"] for c in cases)}


def install_push_guard(repo: Path, into: str = "main", tool: str = "check") -> Path | None:
    hooks = Path(_git(repo, "rev-parse", "--git-path", "hooks").stdout.strip())
    hook = hooks if hooks.is_absolute() else repo / hooks
    hook = hook / "pre-push"
    if hook.exists():
        return None
    hook.parent.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parents[1]
    hook.write_text(HOOK.format(into=into, tool=tool, python=shlex.quote(sys.executable), src=shlex.quote(str(src))))
    hook.chmod(0o755)
    return hook


def required_steps(repo: Path, base: str, commit: str, tool: str) -> dict[str, list[str]]:
    rules = getattr(declaration(repo, tool), "merge_requires", None)
    if not rules:
        return {}
    changed = [p for p in _git(repo, "diff", "--no-renames", "--name-only", "-z", base, commit).stdout.split("\0") if p]
    out = {}
    for step, rule in rules.items():
        if hit := rule_touches(rule, changed):
            out[step] = hit
    return out


def rule_touches(rule: list[str], changed: list[str]) -> list[str]:
    import fnmatch

    prefixes = [r for r in rule if not r.startswith("!")]
    ignored = [r[1:] for r in rule if r.startswith("!")]

    def counts(c: str) -> bool:
        return not any(fnmatch.fnmatchcase(c if "/" in g else c.rsplit("/", 1)[-1], g) for g in ignored)

    return [p for p in prefixes if any((c == p.rstrip("/") or c.startswith(p.rstrip("/") + "/")) and counts(c) for c in changed)]


def step_status(store, run_id: str, step: str) -> str:
    try:
        art = (store.get_attempt(run_id) or {}).get("outputs", {}).get("result")
        doc = store.get_manifest(art).meta if art else {}
    except Exception as e:  # noqa: BLE001  (an unreadable result admits nothing)
        return f"unreadable ({type(e).__name__})"
    steps = [s for s in doc.get("steps", []) if isinstance(s, dict)]
    mine = [s for s in steps if s.get("step") == step]
    sh = next((x for x in (doc.get("shard"), doc.get("deferred")) if isinstance(x, dict) and x.get("all")), None)
    if sh and any(s.get("parts") for s in mine):
        every = [p for p in sh["all"] if isinstance(p, str) and p.partition(":")[0] == step]
        if sorted(p for s in mine for p in s.get("parts") or ()) != sorted(every):
            return "partial"
    if mine:
        return mine[0].get("status", "absent")
    parts = [s.get("status") or "absent" for s in steps if str(s.get("step", "")).startswith(step + ":")]
    return next((x for x in parts if x != "passed"), "passed") if parts else "absent"


def _record(tree: Path, tool: str, command: list[str], on: str | None, project: str | None, log=print,
            sends: list[str] = ()) -> str:
    from . import runs as R

    rid = R.new_run_id()
    base = [sys.executable, "-m", "research", "run", "--id", rid, "--tool", tool, "--cwd", "source",
            *(["--project", project] if project else [])]
    if on:
        r = subprocess.run([*base, "--on", on, "--source", str(tree), *[x for f in sends for x in ("--send", f)], "--", *command],
                           cwd=tree, capture_output=True, text=True)
        if r.returncode:
            raise Refused(f"`research run --on {on}` failed: {(r.stderr or r.stdout).strip()[-400:]}")
        status = R.runs_root(None) / rid / "status.json"
        while True:
            time.sleep(POLL_S)
            subprocess.run([sys.executable, "-m", "research", "fetch", rid], cwd=tree, capture_output=True)
            try:
                if json.loads(status.read_text()).get("state") in ("done", "failed", "cancelled"):
                    break
            except (OSError, ValueError):
                pass
    else:
        subprocess.run([*base, "--", *command], cwd=tree)
    log(f"research merge: `{tool}` recorded as {rid}")
    return rid


POLL_S = 20


PIN = re.compile(r"#?([\w./-]+)@([0-9a-f]{7,40})")


def pinned(repo: Path, ref: str, remote: str) -> tuple[str, str]:
    m = PIN.fullmatch(ref)
    if not m:
        return ref, _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()
    name, want = m.groups()
    label, src = (f"#{name}", f"pull/{name}/head") if name.isdigit() else (name, name)
    _git(repo, "fetch", "-q", remote, src)
    head = _git(repo, "rev-parse", "FETCH_HEAD").stdout.strip()
    if not head.startswith(want):
        raise Refused(f"{label} has moved: its head on {remote} is {head[:12]}, not {want}; review the new head and name it")
    return label, head


IMPORT_LINE = re.compile(r"\s*(import\s|from\s+\S+\s+import\s)")


def keep_both_imports(text: str) -> str | None:
    out: list[str] = []
    ours: list[str] = []
    theirs: list[str] = []
    side, hunks = None, 0
    for line in text.splitlines(keepends=True):
        if side is None and line.startswith("<<<<<<< "):
            side, ours, theirs = "ours", [], []
        elif side == "ours" and line.rstrip("\r\n") == "=======":
            side = "theirs"
        elif side == "theirs" and line.startswith(">>>>>>> "):
            if not all(IMPORT_LINE.match(x) or not x.strip() for x in ours + theirs):
                return None
            out += ours
            seen = {x.rstrip() for x in ours}
            for x in theirs:
                if x.rstrip() not in seen:
                    seen.add(x.rstrip())
                    out.append(x)
            side, hunks = None, hunks + 1
        elif side is None:
            out.append(line)
        else:
            (ours if side == "ours" else theirs).append(line)
    return "".join(out) if hunks and side is None else None


def _strict_json(text: str) -> bool:
    def pairs(kv: list) -> dict:
        if len({k for k, _ in kv}) != len(kv):
            raise ValueError("a key repeats")
        return dict(kv)

    try:
        json.loads(text, object_pairs_hook=pairs)
        return True
    except ValueError:
        return False


def broken_json(tree: Path, theirs: str) -> list[str]:
    mb = _git(tree, "merge-base", "HEAD^1", theirs).stdout.strip()
    changed = [set(_git(tree, "diff", "--name-only", "-z", mb, rev).stdout.split("\0")) for rev in ("HEAD^1", theirs)]

    def text(rev: str, path: str) -> str | None:
        r = _git(tree, "show", f"{rev}:{path}", check=False)
        return None if r.returncode else r.stdout

    bad = []
    for p in sorted(x for x in changed[0] & changed[1] if x.endswith(".json")):
        merged = text("HEAD", p)
        if merged is not None and not _strict_json(merged) and all(
                _strict_json(t) for t in (text("HEAD^1", p), text(theirs, p)) if t is not None):
            bad.append(p)
    return bad


PINS_SCRIPT = "tools/check/pins.py"


def regenerate(tree: Path, ours: str, theirs: str, conflicted: list[str]) -> list[str]:
    script = tree / PINS_SCRIPT
    if not script.is_file() or "--merged" not in script.read_text():
        if conflicted:
            raise Refused(f"a conflict in {', '.join(conflicted)} holds more than import lines")
        return []
    p = subprocess.run([sys.executable, PINS_SCRIPT, "--merged", ours, theirs, "--conflicted", *conflicted], cwd=tree,
                       capture_output=True, text=True)
    try:
        doc = json.loads((p.stdout.strip().splitlines() or ["{}"])[-1])
    except ValueError:
        doc = {}
    notes = doc.get("notes") or [(p.stderr or p.stdout).strip()[-300:]]
    for n in notes:
        print(f"research merge: {PINS_SCRIPT} --merged: {n}", file=sys.stderr)
    if p.returncode == 3:
        raise Refused(f"a conflict in {', '.join(conflicted)} holds more than import lines ({'; '.join(notes)})")
    if p.returncode:
        raise Refused(f"`{PINS_SCRIPT} --merged` exited {p.returncode}: {'; '.join(notes)}")
    regenerated = doc.get("regenerated", [])
    if regenerated:
        _git(tree, "add", "--", *regenerated)
    return regenerated


def _merge(tree: Path, label: str, sha: str, msg: str) -> None:
    ours = _git(tree, "rev-parse", "HEAD").stdout.strip()
    r = _git(tree, "-c", "merge.conflictStyle=merge", "merge", "--no-ff", "--no-commit", sha, check=False)
    conflicted = [p for p in _git(tree, "diff", "--name-only", "-z", "--diff-filter=U").stdout.split("\0") if p]
    if r.returncode and not conflicted:
        _git(tree, "merge", "--abort", check=False)
        raise Refused(f"{label} does not merge onto the train so far: {(r.stderr or r.stdout).strip()[-300:]}")
    resolved, rest = {}, []
    for p in conflicted:
        try:
            new = keep_both_imports((tree / p).read_text())
        except (OSError, UnicodeDecodeError):
            new = None
        if new is None:
            rest.append(p)
        else:
            resolved[p] = new
    for p, new in resolved.items():
        (tree / p).write_text(new)
    if resolved:
        _git(tree, "add", "--", *resolved)
    try:
        regenerated = regenerate(tree, ours, sha, rest)
    except Refused as e:
        _git(tree, "merge", "--abort", check=False)
        raise Refused(f"{label} does not merge onto the train so far ({e}): {(r.stderr or r.stdout).strip()[-300:]}") from None
    trailers = "".join([f"\nImports: {', '.join(resolved)}" if resolved else "",
                        f"\nRegenerated: {', '.join(regenerated)}" if regenerated else ""])
    if _git(tree, "rev-parse", "-q", "--verify", "MERGE_HEAD", check=False).returncode == 0:
        _git(tree, "commit", "-q", "-m", msg + trailers)
    if bad := broken_json(tree, sha):
        raise Refused(f"merging {label} onto the train leaves {', '.join(bad)} unparseable as JSON (or with a key twice in one "
                      "object), though both sides parse: git merged their lines; merge the train so far into it and fix that by hand")


def _chain(tree: Path, pins: list[tuple[str, str]], stamp: str) -> list[str]:
    chain = []
    for label, sha in pins:
        _merge(tree, label, sha, f"Merge {label} ({sha[:8]})\n\nTrain: {stamp}")
        chain.append(_git(tree, "rev-parse", "HEAD").stdout.strip())
    return chain


MERGED_PR = re.compile(r"Merge #(\d+) \(([0-9a-f]{7,40})\)")


def adds_nothing(repo: Path, commit: str, head: str, lazy: bool = True) -> bool:
    r = _git(repo, "merge-tree", "--write-tree", commit, head, check=False, lazy=lazy)
    return r.returncode == 0 and r.stdout.split("\n", 1)[0].strip() == _git(repo, "rev-parse", f"{commit}^{{tree}}").stdout.strip()


def unstacked(repo: Path, commit: str, base: str, remote: str, log=print) -> list[str]:
    out, pins = [], []
    for line in _git(repo, "log", "--first-parent", "--format=%P%x00%s", f"{base}..{commit}").stdout.splitlines():
        parents, _, subject = line.partition("\0")
        if m := MERGED_PR.fullmatch(subject):
            onto, sha = (parents.split() + ["", ""])[:2]
            if sha.startswith(m[2]):
                pins.append((int(m[1]), sha, onto))
            else:
                out.append(f"#{m[1]}'s merge names {m[2]} but merges {sha[:12] or 'nothing'}")
    if not pins:
        return out
    r = _git(repo, "ls-remote", remote, "refs/pull/*/head", check=False)
    if r.returncode:
        raise Refused(f"the pull requests {commit[:12]} merges can't be judged: {remote}'s pull request heads can't be "
                      f"read ({(r.stderr or r.stdout).strip()[-200:]})")
    prs: dict[str, list[int]] = {}
    for line in r.stdout.splitlines():
        h, _, ref = line.partition("\t")
        prs.setdefault(h, []).append(int(ref.split("/")[2]))
    for n, sha, onto in pins:
        r = _git(repo, "fetch", "-q", remote, f"pull/{n}/head", check=False)
        head = None if r.returncode else _git(repo, "rev-parse", "FETCH_HEAD").stdout.strip()
        if head != sha:
            if head and adds_nothing(repo, commit, head):
                log(f"research merge: #{n}'s head moved from {sha[:12]} to {head[:12]}, which adds nothing to {commit[:12]} "
                    f"(merging it leaves the tip's tree); it lands as {sha[:12]}, and the sweep closes it")
            else:
                out.append(f"#{n} was merged at {sha[:12]}, but its head on {remote} " + (
                    f"is now {head[:12]}: rebuild the train on the new head and check that" if head else
                    f"can't be read ({(r.stderr or r.stdout).strip()[-200:]})"))
                continue
        for h in _git(repo, "rev-list", f"{onto}..{sha}").stdout.split():
            for m in prs.get(h, []) if h != sha else ():
                if m != n and _git(repo, "merge-base", "--is-ancestor", h, onto, check=False).returncode:
                    out.append(f"#{n} at {sha[:12]} is stacked on #{m} at {h[:12]}, which is neither on main nor pinned below "
                               f"it in the tip at that head: pin {m}@{h[:9]} before it, or rebase #{n} off it")
    return out


PREFLIGHT = ["tools/check/check.py", "--preflight"]


def preflight_argv() -> list[str]:
    import importlib.util
    import shutil

    uv = shutil.which("uv")
    if importlib.util.find_spec("pytest") is not None or uv is None:
        return [sys.executable, *PREFLIGHT]
    return [uv, "run", "--locked", "python", *PREFLIGHT]


def _removals(text: str) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    path = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            path = None
        elif line.startswith("--- "):
            path = line[6:] if line.startswith("--- a/") else None
        elif line.startswith("-") and path is not None and line[1:].strip():
            out.setdefault(path, Counter())[line[1:]] += 1
    return out


def reverted(repo: Path, base: str, tip: str, pins: list[tuple[str, str]], allow: tuple[str, ...] = ()) -> list[str]:
    own: Counter = Counter()
    for _label, sha in pins:
        log = _git(repo, "log", "--no-merges", "-p", "-U0", "--no-renames", "--no-color", "--format=%x00%(trailers:key=Restack-move,valueonly)",
                   f"{base}..{sha}").stdout
        for commit in log.split("\x00")[1:]:
            trailer, _, patch = commit.partition("\n")
            if not trailer.strip():
                for lines in _removals(patch).values():
                    own += lines
    out = []
    regen = _git(repo, "log", "--format=%(trailers:key=Regenerated,valueonly,separator=%x2C)", f"{base}..{tip}").stdout
    allow = (*allow, *{p for t in re.split(r"[,\n]", regen) if (p := t.strip())})
    for path, lines in sorted(_removals(_git(repo, "diff", "-U0", "--no-renames", "--no-color", base, tip).stdout).items()):
        extra = lines - own
        own -= lines
        if extra and path not in allow:
            first = next(iter(extra)).strip()
            out.append(f"{path} loses {sum(extra.values())} line(s) no train pull request removes in a commit of its own "
                       f"(e.g. {first[:80]!r})")
    return out


REVERTED_HINT = (". A merge in a ref's history dropped them: merge the target into it again, keeping both sides, or pass "
                 "--allow-removal PATH for a removal the train means")


def prepare(repo: Path, refs: list[str], branch: str, onto: str, remote: str, lint: list[str], store,
            allow_removal: tuple[str, ...] = ()) -> tuple[str, str]:
    """Builds a train for its check without recording or landing anything."""
    r = _git(repo, "fetch", "-q", remote, onto, check=False)
    base = _git(repo, "rev-parse", "--verify", f"{'FETCH_HEAD' if r.returncode == 0 else onto}^{{commit}}").stdout.strip()
    pins = [pinned(repo, ref, remote) for ref in refs]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tree = Path(tempfile.mkdtemp(prefix="research-train-"))
    _git(repo, "worktree", "add", "-q", "--detach", str(tree), base)
    try:
        tip = _chain(tree, pins, stamp)[-1]
        if why := unstacked(repo, tip, base, remote):
            raise Refused("; ".join(why) + f": the merge gate would refuse the train ({tip[:12]}); nothing was pushed")
        if lost := reverted(repo, base, tip, pins, allow_removal):
            raise Refused(f"the train {tip[:12]} would revert what landed: " + "; ".join(lost) + REVERTED_HINT + "; nothing was pushed")
        if subprocess.run(lint, cwd=tree).returncode:
            raise Refused(f"`{shlex.join(lint)}` failed on the train ({tip[:12]}); nothing was pushed")
        _git(repo, "push", "-q", remote, f"{tip}:refs/heads/{branch}")
        got = (_git(repo, "ls-remote", remote, f"refs/heads/{branch}").stdout.split() or ["nothing"])[0]
        if got != tip:
            raise Refused(f"{remote}'s {branch} is {got[:12]} after the push, not the train {tip[:12]}")
        return tip, _git(repo, "rev-parse", f"{tip}^{{tree}}").stdout.strip()
    finally:
        _git(repo, "worktree", "remove", "--force", str(tree), check=False)


def train(repo: Path, refs: list[str], into: str, store, tool: str, command: list[str], on: str | None = None,
          project: str | None = None, log=print, spec: str | None = None, sends: list[str] = (),
          remote: str = "origin", allow_removal: tuple[str, ...] = ()) -> tuple[str | None, str | None]:
    head = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
    if head != into or _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise Refused(f"the checkout must be a clean {into}")
    pins = [pinned(repo, ref, remote) for ref in refs]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tree = Path(tempfile.mkdtemp(prefix="research-train-"))
    branch = f"research-train/{stamp}"
    _git(repo, "worktree", "add", "-q", "-b", branch, str(tree), "HEAD")
    try:
        chain = _chain(tree, pins, stamp)
        if lost := reverted(repo, "HEAD", chain[-1], pins, allow_removal):
            raise Refused(f"the train {chain[-1][:12]} would revert what landed: " + "; ".join(lost) + REVERTED_HINT + "; nothing was checked")
        if why := unstacked(repo, chain[-1], "HEAD", remote, log):
            raise Refused("; ".join(why) + "; nothing was checked")
        need = required_steps(repo, "HEAD", chain[-1], tool)
        if need and on and not sends:
            raise Refused(f"the train changes {', '.join(sorted(set(p for ps in need.values() for p in ps)))}, where `{tool}` must run "
                          f"{', '.join(need)}: pass what it needs with --send (for `check`: `uv run python tools/check/check.py "
                          f"--agreement-files` prints the flags)")

        def passes(k: int) -> bool:
            _git(tree, "checkout", "-q", "--detach", chain[k - 1])
            _record(tree, spec or tool, command, on, project, log, sends)
            try:
                gate(repo, chain[k - 1], into, store, tool, log=lambda *_: None, remote=remote)
                ok = True
            except Refused as e:
                ok = False
                log(f"research merge: {e}")
            log(f"research merge: train prefix {k}/{len(chain)} ({chain[k - 1][:12]}): `{tool}` {'passed' if ok else 'FAILED'}")
            return ok

        lo, hi, bad = 0, len(chain), None
        if not passes(hi):
            while hi - lo > 1:
                mid = (lo + hi) // 2
                lo, hi = (mid, hi) if passes(mid) else (lo, mid)
            bad = refs[hi - 1]
        land = chain[hi - 1] if bad is None else (chain[lo - 1] if lo else None)
        if land is not None:
            gate(repo, land, into, store, tool, log, remote)
            _git(repo, "merge", "--ff-only", "-q", land)
        return land, bad
    finally:
        _git(repo, "worktree", "remove", "--force", str(tree), check=False)
        _git(repo, "branch", "-D", branch, check=False)


GH_TIMEOUT_S = 120


def _gh_cli(repo: Path, *args: str) -> str:
    try:
        r = subprocess.run(["gh", *args], cwd=repo, capture_output=True, text=True, timeout=GH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise Refused(f"gh {' '.join(args[:2])}: no answer in {GH_TIMEOUT_S} s") from None
    if r.returncode:
        raise Refused(f"gh {' '.join(args[:2])}: {(r.stderr or r.stdout).strip()[-300:]}")
    return r.stdout


def landed_in(repo: Path, commit: str, tip: str) -> str:
    after = set(_git(repo, "rev-list", "--ancestry-path", f"{commit}..{tip}").stdout.split())
    landed = commit
    for c in _git(repo, "rev-list", "--first-parent", "-n", str(len(after) + 1), tip).stdout.split():
        if c not in after:
            return commit if c == commit else landed
        landed = c
    return landed


ENDED_QUERY = ("query($owner: String!, $name: String!) { repository(owner: $owner, name: $name) { pullRequests(states: [MERGED, "
               "CLOSED], first: 100, orderBy: {field: UPDATED_AT, direction: DESC}) { nodes { number state headRefName headRefOid "
               "isCrossRepository } } } }")


def sweep(repo: Path, into: str = "main", remote: str = "origin", dry_run: bool = False, gh=None, log=print,
          keep: Iterable[str] = ()) -> tuple[list[int], list[int]]:
    gh = gh or (lambda *a: _gh_cli(repo, *a))
    _git(repo, "fetch", "-q", remote, f"+refs/heads/{into}:refs/remotes/{remote}/{into}")
    tip = _git(repo, "rev-parse", f"refs/remotes/{remote}/{into}").stdout.strip()

    # a head the clone lacks once `into` is fetched isn't on it, so a partial clone fetches none to say so: a lazy fetch, with no
    # negotiation, brings that head's whole history, and git's auto maintenance never repacks such packs
    def on(sha: str) -> bool:
        return _git(repo, "merge-base", "--is-ancestor", sha, tip, check=False, lazy=False).returncode == 0

    local = _git(repo, "rev-parse", "--verify", "-q", f"refs/heads/{into}", check=False).stdout.strip()
    if local and not on(local):
        log(f"research merge: {into} has commits {remote}/{into} lacks; push it, then `research merge --sweep` closes what they land")
    prs = sorted(json.loads(gh("pr", "list", "--state", "open", "--limit", "1000", "--json",
                               "number,headRefName,headRefOid,baseRefName")), key=lambda p: p["number"])
    done = [p for p in prs if p["headRefName"] != into and on(p["headRefOid"])]
    rest = [p for p in prs if p not in done]
    bases = sorted({p["baseRefName"] for p in rest if p["baseRefName"] != into})
    tips = {}
    if bases:
        out = _git(repo, "ls-remote", remote, *[f"refs/heads/{b}" for b in bases]).stdout
        tips = {ref.removeprefix("refs/heads/"): sha for sha, ref in (line.split() for line in out.splitlines())}
    moved = [p for p in rest if p["baseRefName"] in tips and on(tips[p["baseRefName"]])]
    pinned = {int(m[1]): parents.split()[1] for parents, _, subject in (line.partition("\0") for line in _git(
        repo, "log", "-n", "5000", "--format=%P%x00%s", "--grep", r"^Merge #[0-9]", tip).stdout.splitlines())
        if (m := MERGED_PR.fullmatch(subject)) and len(parents.split()) == 2}
    done += [p for p in rest if p not in moved and p["number"] in pinned and adds_nothing(repo, tip, p["headRefOid"], lazy=False)]
    rest = [p for p in rest if p not in done]
    verb = "would " if dry_run else ""
    for p in moved:
        log(f"research merge: {verb}retarget #{p['number']} from {p['baseRefName']} (on {remote}/{into}) onto {into}")
        if not dry_run:
            gh("pr", "edit", str(p["number"]), "--base", into)
    used = {p["headRefName"] for p in rest} | {p["baseRefName"] for p in rest if p not in moved}
    for p in done:
        later = not on(p["headRefOid"])
        landed = landed_in(repo, pinned[p["number"]] if later else p["headRefOid"], tip)
        subject = _git(repo, "log", "-1", "--format=%s", landed).stdout.strip()
        note = f"Merged into {into} at {landed[:12]} ({subject})." + (
            f" Its head since, {p['headRefOid'][:12]}, adds nothing to {into}." if later else "")
        log(f"research merge: {verb}close #{p['number']} ({p['headRefName']}): {note}")
        if not dry_run:
            gh("pr", "close", str(p["number"]), "--comment", note)
        head = p["headRefName"]
        if head in used or later:
            continue
        log(f"research merge: {verb}delete {remote}'s branch {head} (#{p['number']}'s head, on {into})")
        if not dry_run:
            r = _git(repo, "push", "-q", f"--force-with-lease=refs/heads/{head}:{p['headRefOid']}", remote, f":refs/heads/{head}",
                     check=False)
            if r.returncode:
                log(f"research merge: kept {head}: {(r.stderr or r.stdout).strip()[-200:]}")
    ended = json.loads(gh("api", "graphql", "-F", "owner={owner}", "-F", "name={repo}", "-f", f"query={ENDED_QUERY}",
                          "--jq", ".data.repository.pullRequests.nodes"))
    heads = {ref.removeprefix("refs/heads/"): sha for sha, ref in (
        line.split() for line in _git(repo, "ls-remote", "--heads", remote).stdout.splitlines())}
    keep = set(keep)
    for p in ended:
        head, closed = p["headRefName"], p.get("state") == "CLOSED"
        if (head in used or head in KEEP_BRANCHES or p.get("isCrossRepository") or heads.get(head) != p["headRefOid"]
                or (closed and (head in keep or RUN_BRANCH_RE.fullmatch(head))) or not (closed or on(p["headRefOid"]))):
            continue
        log(f"research merge: {verb}delete {remote}'s branch {head} "
            + (f"(closed #{p['number']}'s head, unmerged)" if closed else f"(merged #{p['number']}'s head, on {into})"))
        if not dry_run:
            r = _git(repo, "push", "-q", f"--force-with-lease=refs/heads/{head}:{p['headRefOid']}", remote, f":refs/heads/{head}",
                     check=False)
            if r.returncode and "remote ref does not exist" not in (r.stderr or ""):
                log(f"research merge: kept {head}: {(r.stderr or r.stdout).strip()[-200:]}")
    if not done and not moved:
        log(f"research merge: no open pull request is on {remote}/{into} ({tip[:12]}) or based on a branch that is")
    return [p["number"] for p in done], [p["number"] for p in moved]


CHECKED = "checked"
KEEP_BRANCHES = frozenset({"main", CHECKED, "next", "queue-shadow"})
RUN_BRANCH_RE = re.compile(r"cursor/verity-[a-z0-9-]+-[0-9a-f]{4}")
MERGED_IDLE_S, UNMERGED_IDLE_S = 86400, 3 * 86400
SWEEP_LIMIT = 200


def sweep_branches(repo: Path, remote: str = "origin", into: str = "main", now: float | None = None, dry_run: bool = False,
                   gh=None, log=print, chunk: int = 100, keep: Iterable[str] = (), limit: int = SWEEP_LIMIT) -> tuple[int, int]:
    gh = gh or (lambda *a: _gh_cli(repo, *a))
    now = time.time() if now is None else now
    _git(repo, "fetch", "-q", "--prune", remote, f"+refs/heads/*:refs/sweep/{remote}/*")
    tip = _git(repo, "rev-parse", f"refs/sweep/{remote}/{into}").stdout.strip()
    prs = json.loads(gh("pr", "list", "--state", "open", "--limit", "1000", "--json", "headRefName,baseRefName"))
    used = {p["headRefName"] for p in prs} | {p["baseRefName"] for p in prs} | set(keep)
    rows = _git(repo, "for-each-ref", "--format=%(refname:lstrip=3) %(objectname) %(committerdate:unix)",
                f"refs/sweep/{remote}/").stdout.splitlines()
    delete, archive = [], []
    for name, sha, when in (r.split() for r in rows):
        if name == into or name in KEEP_BRANCHES or name in used or RUN_BRANCH_RE.fullmatch(name):
            continue
        idle = now - int(when)
        if _git(repo, "merge-base", "--is-ancestor", sha, tip, check=False).returncode == 0:
            if idle > MERGED_IDLE_S:
                delete.append((idle, name, sha))
        elif idle > UNMERGED_IDLE_S:
            archive.append((idle, name, sha))
    found = len(delete), len(archive)
    delete, archive = ([(n, s) for _, n, s in sorted(x, key=lambda r: -r[0])[:limit]] for x in (delete, archive))
    verb = "would " if dry_run else ""
    log(f"research merge: {verb}delete {len(delete)} of {found[0]} branch(es) on {into} a day idle, {verb}archive {len(archive)} "
        f"of {found[1]} unmerged branch(es) three days idle to refs/archive/heads/")
    if delete:
        log(f"research merge: {verb}delete: {' '.join(n for n, _ in delete)}")
    if archive:
        log(f"research merge: {verb}archive: {' '.join(n for n, _ in archive)}")
    if not dry_run:
        for i in range(0, len(delete), chunk):
            part = delete[i:i + chunk]
            _git(repo, "push", "-q", *(f"--force-with-lease=refs/heads/{n}:{s}" for n, s in part), remote,
                 *(f":refs/heads/{n}" for n, _ in part), check=False)
        for i in range(0, len(archive), chunk):
            part = archive[i:i + chunk]
            _git(repo, "push", "-q", "--atomic", *(f"--force-with-lease=refs/heads/{n}:{s}" for n, s in part), remote,
                 *(f"{s}:refs/archive/heads/{n}" for n, s in part), *(f":refs/heads/{n}" for n, _ in part), check=False)
    for line in _git(repo, "for-each-ref", "--format=%(refname)", f"refs/sweep/{remote}/").stdout.split():
        _git(repo, "update-ref", "-d", line, check=False)
    return len(delete), len(archive)


def _sweep_after(repo: Path, into: str, remote: str, keep: Iterable[str] = ()) -> None:
    if _git(repo, "remote", "get-url", remote, check=False).returncode:
        print(f"research merge: no remote {remote!r}; pull request sweep skipped")
        return
    try:
        sweep(repo, into, remote, keep=keep)
    except (Refused, ValueError, KeyError) as e:
        print(f"research merge: pull request sweep failed ({e}); run `research merge --sweep` once GitHub is reachable",
              file=sys.stderr)


MIRROR, MIRROR_ENV, MIRROR_TIMEOUT_S = "vy-nebius-cpu-4", "RESEARCH_MERGE_MIRROR", 120


def mirror_name() -> str | None:
    name = os.environ.get(MIRROR_ENV, MIRROR).strip()
    return None if name.lower() in ("", "none", "off") else name


def _mirror_target(name: str) -> tuple[list[str], str]:
    from . import remote

    m = remote.resolve_machine(name, "verity")
    t = remote.ssh_host_target(m)
    return remote.ssh_argv(t["host"], t["port"], user=t["user"]), f"{m['root']}/{remote.POD_GIT_REPO}"


def push_mirror(repo: Path, sha: str, name: str, target=None, run=subprocess.run) -> bool:
    """A plain push, so only a fast-forward."""
    from .remote import git_push_url

    rerun = f"cd {shlex.quote(str(repo))} && research merge --mirror {sha}"
    try:
        ssh, bare = (target or _mirror_target)(name)
        r = run(["git", "-c", "protocol.ext.allow=always", "push", "-q", "--no-verify", git_push_url(ssh, bare), f"{sha}:refs/heads/main"],
                cwd=repo, capture_output=True, text=True, timeout=MIRROR_TIMEOUT_S)
        why = None if r.returncode == 0 else f"git push exited {r.returncode}: {' '.join((r.stderr or r.stdout).split())[-300:]}"
    except subprocess.TimeoutExpired:
        why = f"no answer within {MIRROR_TIMEOUT_S} s"
    except Exception as e:                                      # noqa: BLE001: a machine that can't be reached fails only this push
        why = f"{type(e).__name__}: {e}"
    if why:
        print(f"research merge: main ({sha[:12]}) not pushed to {name} ({why}); rerun: {rerun}", file=sys.stderr)
        return False
    print(f"research merge: pushed main ({sha[:12]}) to {name}:{bare}")
    return True


LEAN_PUBLISH, LEAN_PUBLISH_ENV, LEAN_PUBLISH_LOG, LEAN_PUBLISH_TIMEOUT_S = (
    "/data/infra/lean_publish.sh", "RESEARCH_LEAN_PUBLISH", "/data/drive/lean-publish.log", 120)


def lean_publish(repo: Path, start: str, tip: str, log=print, run=subprocess.run, clock=time.monotonic) -> bool:
    from .queue import lean_change

    script, what = os.environ.get(LEAN_PUBLISH_ENV, LEAN_PUBLISH).strip(), f"research merge: the Lean cache publish of {tip[:12]}"
    if not script:
        log(f"{what} is off (${LEAN_PUBLISH_ENV} is empty)")
        return False
    try:
        touched = lean_change(repo, tip, base=start)
    except Exception as e:                                      # noqa: BLE001: a diff that can't be read skips only the publish
        log(f"{what} not started: {start[:12]}..{tip[:12]} can't be diffed ({type(e).__name__}: {e})")
        return False
    if not touched:
        log(f"{what} not started: {start[:12]}..{tip[:12]} touches no Lean package")
        return False
    if not Path(script).is_file():
        log(f"{what} not started: {script} is absent")
        return False
    t0 = clock()
    try:
        r = run([script, tip], cwd=repo, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=LEAN_PUBLISH_TIMEOUT_S)
        why = None if r.returncode == 0 else f"exited {r.returncode}: {' '.join((r.stderr or r.stdout or '').split())[-300:]}"
    except subprocess.TimeoutExpired:
        why = f"no exit within {LEAN_PUBLISH_TIMEOUT_S} s"
    except Exception as e:                                      # noqa: BLE001: a script that can't run fails only the publish
        why = f"{type(e).__name__}: {e}"
    took = clock() - t0
    if why:
        log(f"{what} not started: {script} {why} ({took:.1f} s); its log is {LEAN_PUBLISH_LOG}; rerun: {script} {tip}")
        return False
    log(f"{what} started by {script} in {took:.1f} s; its minutes and bytes are in {LEAN_PUBLISH_LOG}")
    return True


AFTER_LAND_THEN_REMOVE = 'r=$1; shift; "$@"; s=$?; cd / && git -C "$r" worktree remove --force "$0"; exit $s'


def _own_tree(repo: Path, commit: str) -> Path | None:
    """A detached worktree, since the caller's checkout moves on under it."""
    import shutil

    from .queue import worktrees_locked
    from .settings import train_trees

    if _git(repo, "rev-parse", "--git-common-dir", check=False).returncode:
        return None
    if root := train_trees():
        os.makedirs(root, exist_ok=True)
    tree = Path(tempfile.mkdtemp(prefix=f"research-after-land-{commit[:12]}-", dir=root))
    with worktrees_locked(repo):
        added = _git(repo, "worktree", "add", "-q", "--detach", str(tree), commit, check=False).returncode == 0
    if not added:
        shutil.rmtree(tree, ignore_errors=True)
        return None
    return tree


def after_land(repo: Path, tool: str, commit: str, log=print, spawn=subprocess.Popen, attempt: str = "") -> list[str] | None:
    from .store import tools_registry

    try:
        argv = [*tools_registry.from_tree(tool, repo).after_land]
    except Exception:  # noqa: BLE001  (a tool the checkout does not declare runs nothing after a landing)
        return None
    if not argv:
        return None
    argv += [commit, "--attempt", attempt] if attempt else [commit]
    out = Path(tempfile.gettempdir()) / f"research-after-land-{commit[:12]}.log"
    tree = _own_tree(repo, commit)
    run = argv if tree is None else ["sh", "-c", AFTER_LAND_THEN_REMOVE, str(tree), str(repo), *argv]
    try:
        with open(out, "ab") as f:
            spawn(run, cwd=tree or repo, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as e:
        if tree is not None:
            _git(repo, "worktree", "remove", "--force", str(tree), check=False)
        log(f"research merge: after the landing, `{shlex.join(argv)}` did not start ({e}); run it by hand")
        return argv
    log(f"research merge: after the landing, started `{shlex.join(argv)}` in {tree or repo} (log {out})")
    return argv


AFTER_LAND_REF = "refs/research/after-land"
SPEC_ALERTED_REF = "refs/research/spec-alerted"


def _remote_refs(repo: Path, remote: str, *refs: str) -> dict[str, str] | None:
    r = _git(repo, "ls-remote", remote, *refs, check=False)
    return None if r.returncode else {ref: sha for sha, ref in (line.split() for line in r.stdout.splitlines())}


def _move(repo: Path, remote: str, ref: str, was: str, tip: str, log=print) -> bool:
    p = _git(repo, "push", "-q", f"--force-with-lease={ref}:{was}", remote, f"{tip}:{ref}", check=False)
    if p.returncode:
        log(f"research merge: {remote}'s {ref} wasn't moved to {tip[:12]} ({(p.stderr or p.stdout).strip()[-200:]}); "
            "another lander may have caught up first")
    return not p.returncode


def caught_up(repo: Path, remote: str, tool: str = "check", since: str = "", log=print,
              runs: dict[str, str] | None = None) -> str | None:
    from .spec_alert import alert

    refs = _remote_refs(repo, remote, "refs/heads/main", AFTER_LAND_REF, SPEC_ALERTED_REF)
    if refs is None:
        log(f"research merge: {remote}'s main can't be read; the steps after a landing wait for the next `research merge`")
        return None
    tip, mark, alerted = refs.get("refs/heads/main", ""), refs.get(AFTER_LAND_REF, ""), refs.get(SPEC_ALERTED_REF, "")
    start = mark or since
    spec_from = alerted or start
    if not tip or (tip == mark and spec_from in ("", tip)):
        return None
    if start or spec_from:
        fetched = _git(repo, "fetch", "-q", remote, "refs/heads/main", *(r for r, s in ((AFTER_LAND_REF, mark),
                                                                                         (SPEC_ALERTED_REF, alerted)) if s),
                       check=False)
        if fetched.returncode:
            log(f"research merge: {remote}'s main or its markers can't be fetched; the steps after a landing wait for the next "
                "`research merge`")
            return None
    if spec_from and spec_from != tip:
        if _git(repo, "merge-base", "--is-ancestor", spec_from, tip, check=False).returncode:
            log(f"research merge: the spec-change DM last went up to {spec_from[:12]}, which isn't in {remote}'s main "
                f"({tip[:12]}); no DM was sent: send it by hand, then `git push {remote} {tip}:{SPEC_ALERTED_REF}`")
        elif alert(repo, spec_from, tip, remote=remote, log=log):
            _move(repo, remote, SPEC_ALERTED_REF, alerted, tip, log)
        else:
            log(f"research merge: the spec DM over {spec_from[:12]}..{tip[:12]} wasn't sent; {SPEC_ALERTED_REF} stays at "
                f"{spec_from[:12]}, so the next `research merge` or `research merge --sweep` sends it again")
    elif not alerted:
        _move(repo, remote, SPEC_ALERTED_REF, "", tip, log)
    if tip == mark:
        return None
    if start:
        if _git(repo, "merge-base", "--is-ancestor", start, tip, check=False).returncode:
            log(f"research merge: the steps after a landing last ran for {start[:12]}, which isn't in {remote}'s main "
                f"({tip[:12]}); nothing ran: run them by hand, then `git push {remote} {tip}:{AFTER_LAND_REF}`")
            return None
        if mark and mark != since:
            n = len(_git(repo, "rev-list", "--first-parent", f"{mark}..{tip}").stdout.split())
            log(f"research merge: the steps after a landing last ran for {mark[:12]}; catching up {n} commit(s) of main to {tip[:12]}")
        if remote == "origin" and (name := mirror_name()):
            push_mirror(repo, tip, name)
        after_land(repo, tool, tip, log=log, attempt=(runs or {}).get(tip, ""))
        if remote == "origin":
            lean_publish(repo, start, tip, log=log)
            report_landings(repo, start, tip, log=log, runs=runs)
            label_landed(repo, start, tip, log=log, runs=runs)
    if _move(repo, remote, AFTER_LAND_REF, mark, tip, log) and not start:
        log(f"research merge: {remote} had no {AFTER_LAND_REF}; set it to main ({tip[:12]}): the steps after a landing are "
            "caught up from here, and nothing before it was looked at")
    return tip if start else None


MERGED_REF = re.compile(r"Merge (#\d+|origin/\S+|[0-9a-f]{12,40}) \([0-9a-f]{7,40}\)|^Queue: (#\d+) ")
CHECK_TRAILER = re.compile(r"^Check(?:ed)?: (.*)$", re.M)
RUN_ID = re.compile(r"r20\d{6}-(?:\d{6}|\*)-[0-9a-f]{4}")


def _merged_name(ref: str, prs: dict[str, int]) -> str:
    if ref.startswith("#"):
        return ref
    if (branch := ref.removeprefix("origin/")) != ref:
        return f"#{prs[branch]}" if branch in prs else branch
    sha = next((k for k in prs if len(k) == 40 and k.startswith(ref)), None)
    return f"#{prs[sha]}" if sha else ref[:12]


def landing_lines(repo: Path, start: str, tip: str, prs: dict[str, int] | None = None,
                  runs: dict[str, str] | None = None) -> list[str]:
    prs = prs or {}
    lines = []
    for c in _git(repo, "rev-list", "--first-parent", "--reverse", f"{start}..{tip}").stdout.split():
        parents = _git(repo, "rev-list", "--parents", "-n", "1", c).stdout.split()[1:]
        subject, body = _git(repo, "log", "-1", "--format=%s%x00%b", c).stdout.split("\0", 1)
        subjects = [subject]
        if len(parents) == 2:
            subjects += _git(repo, "log", "--first-parent", "--format=%s", f"{parents[0]}..{parents[1]}").stdout.splitlines()
        merged = []
        for s in subjects:
            for m in MERGED_REF.finditer(s):
                if m[2] is None and len(parents) == 2 and parents[1].startswith(m[1]):
                    continue
                if (ref := _merged_name(m[1] or m[2], prs)) not in merged:
                    merged.append(ref)
        what = " + ".join(reversed(merged)) if merged else f"“{subject}”"
        found = [r for t in CHECK_TRAILER.findall(body) for r in RUN_ID.findall(t)] or RUN_ID.findall((runs or {}).get(c, ""))
        lines.append(f"landed on main: {c[:12]} {what}; checked by {', '.join(dict.fromkeys(found)) or 'no run (no Check: trailer)'}")
    return lines


LANDED_BY = "research-merge"


def landed_runs(repo: Path, start: str, tip: str, runs: dict[str, str] | None = None) -> list[tuple[str, str, str]]:
    out = []
    for c in _git(repo, "rev-list", "--first-parent", "--reverse", f"{start}..{tip}" if start else tip).stdout.split():
        body, ct = _git(repo, "log", "-1", "--format=%b%x00%ct", c).stdout.rsplit("\0", 1)
        ts = datetime.fromtimestamp(int(ct), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        found = [r for t in CHECK_TRAILER.findall(body) for r in RUN_ID.findall(t)] or RUN_ID.findall((runs or {}).get(c, ""))
        out += [(r, c, ts) for r in dict.fromkeys(found)]
    return out


def _resolve_run(remote, run: str) -> str | None:
    if "*" not in run:
        return run
    if remote is None:
        return None
    date, _, suffix = run.split("-")
    hits = [k.rsplit("/", 1)[-1].removesuffix(".json") for k in remote.list(f"attempts/{date}-") if k.endswith(f"-{suffix}.json")]
    return hits[0] if len(hits) == 1 else None


def label_landed(repo: Path, start: str, tip: str, *, store=None, log=print, runs: dict[str, str] | None = None) -> bool:
    try:
        from .store import labels_sync as LS
        from .store.local import LocalStore

        found = landed_runs(repo, start, tip, runs)
        if not found:
            return True
        st = store if store is not None else LocalStore(None)
        done, unresolved = [], []
        for run, commit, ts in found:
            if (rid := _resolve_run(st.remote, run)) is None:
                unresolved.append(run)
                continue
            d = rid[1:16]
            ts = max(ts, f"{d[:4]}-{d[4:6]}-{d[6:8]}T{d[9:11]}:{d[11:13]}:{d[13:15]}Z")
            if not any(lab.key == "landed" and lab.value == commit and lab.by == LANDED_BY for lab in st.labels(rid)):
                st.add_label(rid, "landed", commit, LANDED_BY, ts=ts)
            done.append(rid)
        if unresolved:
            log(f"research merge: no `landed` label on {', '.join(unresolved)}: the id names no single run on the remote")
        if st.remote is None:
            log(f"research merge: {len(done)} `landed` label(s) stay in {st.root} (no remote): `research data push --pending` uploads them")
            return False
        failed = LS.push_targets(st, sorted(set(done)))["failed"] if done else []
        log(f"research merge: {len(done)} `landed` label(s) on the check runs of {start[:12] or 'the root'}..{tip[:12]}"
            + (f"; {len(failed)} not pushed: {failed[0]}" if failed else ""))
        return not failed and not unresolved
    except Exception as e:                              # noqa: BLE001: a label that fails leaves the landings as they are
        print(f"research merge: the `landed` labels over {start[:12]}..{tip[:12]} weren't written: {type(e).__name__}: {e}",
              file=sys.stderr)
        return False


REVERTS = re.compile(r"^This reverts commit ([0-9a-f]{40})", re.M)
CHECKED_HEADS_SQL = """SELECT DISTINCT lv.value, lt.value FROM records l
JOIN fields lt ON lt.id = l.id AND lt.path = 'target'
JOIN fields lv ON lv.id = l.id AND lv.path = 'value'
JOIN fields rs ON rs.id = lt.value AND rs.path IN ('source.commit', 'execution.remote.source_sha')
WHERE l.type = 'label' AND l.kind = 'landed' AND rs.value = lv.value"""


def checked_heads(repo: Path, tip: str, idx) -> list[tuple[str, str]]:
    line = _git(repo, "rev-list", "--first-parent", "--reverse", tip).stdout.split()
    pos = {c: i for i, c in enumerate(line)}
    found: dict[str, str] = {}
    for commit, run in sorted(idx.query(CHECKED_HEADS_SQL)[1]):
        if commit in pos:
            found.setdefault(commit, run)
    heads = sorted(found, key=pos.__getitem__)
    bad = set()
    for r in REVERTS.findall(_git(repo, "log", "--first-parent", "--format=%B", tip).stdout):
        if r in pos and (h := next((h for h in heads if pos[h] >= pos[r]), None)):
            bad.add(h)
    return [(h, found[h]) for h in heads if h not in bad]


def main_line(repo: Path, tip: str, idx) -> list[str]:
    return [c for c, _ in reversed(checked_heads(repo, tip, idx))]


def report_landings(repo: Path, start: str, tip: str, *, api=None, reg: dict | None = None, gh=None, log=print,
                    runs: dict[str, str] | None = None) -> bool:
    from .monitors import MERGE_THREAD
    from .slack import FRONT_DOOR, QUIET, Slack, _footer, load_registry, registry_path

    try:
        gh = gh or (lambda *a: _gh_cli(repo, *a))
        try:
            prs = {k: p["number"] for p in sorted(json.loads(gh("pr", "list", "--state", "all", "--limit", "300", "--json",
                                                                 "number,headRefName,headRefOid")), key=lambda p: p["number"])
                   for k in (p["headRefName"], p["headRefOid"])}
        except Exception:                               # noqa: BLE001: without GitHub a branch is named by its branch
            prs = {}
        lines = landing_lines(repo, start, tip, prs, runs)
        if not lines:
            return True
        reg = reg if reg is not None else load_registry(registry_path())
        api = api if api is not None else Slack(os.environ.get("SLACK_BOT_TOKEN"))
        api.call("chat.postMessage", channel=reg["channels"][FRONT_DOOR]["id"], thread_ts=MERGE_THREAD,
                 text="\n".join(lines) + _footer((QUIET,)), username="research merge", unfurl_links=False)
        log(f"research merge: posted {len(lines)} landing(s) to the merge thread")
        return True
    except Exception as e:                              # noqa: BLE001: a report that fails leaves the landings as they are
        print(f"research merge: the landings {start[:12]}..{tip[:12]} weren't posted to the merge thread: {type(e).__name__}: {e}",
              file=sys.stderr)
        return False


def spec_alert_lag(repo: Path, remote: str) -> str:
    refs = _remote_refs(repo, remote, "refs/heads/main", SPEC_ALERTED_REF) or {}
    tip, alerted = refs.get("refs/heads/main", ""), refs.get(SPEC_ALERTED_REF, "")
    if not alerted or alerted == tip:
        return ""
    return (f"{remote}'s {SPEC_ALERTED_REF} is {alerted[:12]}, behind main ({tip[:12]}): the spec DM over that range "
            "hasn't been sent")


DRIVE_ENV = "RESEARCH_DRIVE"
DRIVE_ONLY = ("main", "next", CHECKED)


def on_drive() -> bool:
    return os.environ.get(DRIVE_ENV) == "1"


def off_drive(ref: str, verb: str) -> str:
    return (f"`{ref}` is moved only by drive's App, and this isn't drive (no {DRIVE_ENV}=1): run it on the control pod, "
            f"`/data/infra/drive.sh {verb}`")


def _push_and_sweep(repo: Path, into: str, remote: str, push: bool, tool: str = "check") -> int:
    if push and into in DRIVE_ONLY and not on_drive():
        print(f"research merge: not pushed: {off_drive(into, 'merge ...')}", file=sys.stderr)
        return 1
    if push:
        from .spec_alert import remote_head

        was = remote_head(repo, remote, into) if into == "main" else ""
        r = _git(repo, "push", "-q", remote, f"refs/heads/{into}:refs/heads/{into}", check=False)
        if r.returncode:
            print(f"research merge: push of {into} to {remote} refused, nothing swept: {(r.stderr or r.stdout).strip()[-300:]}",
                  file=sys.stderr)
            return 1
        print(f"research merge: pushed {into} to {remote}")
        if into != "main":
            after_land(repo, tool, _git(repo, "rev-parse", "HEAD").stdout.strip())
        elif not caught_up(repo, remote, tool, since=was) and not was:
            print(f"research merge: no {remote}/{into} before the push to compare with; no spec change was looked for, "
                  f"so read this push's lean-audit.json changes yourself", file=sys.stderr)
    _sweep_after(repo, into, remote)
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[1])
    ap.add_argument("ref", nargs="*", help="the commit to merge (a branch, tag or sha); with --train, several, in merge order")
    ap.add_argument("--sweep", action="store_true", help="merge nothing: close the open pull requests on the remote's --into and "
                    "retarget the ones based on a branch that is (with --dry-run: say what it would do)")
    ap.add_argument("--remote", default="origin", help="the remote whose --into branch the sweep reads, that pinned train refs are "
                    "fetched from and --prepare pushes to (default origin)")
    ap.add_argument("--train", action="store_true", help="merge the refs as one candidate, record the tool on it, fast-forward on "
                    "a pass, bisect to the offending ref on a failure")
    ap.add_argument("--on", default=None, metavar="MACHINE", help="--train: record the tool on this machine (research run --on)")
    ap.add_argument("--project", default=None, help="--train: the project the run belongs to (required with --on)")
    ap.add_argument("--command", default=None, help="--train: the workload command (default the tool declaration's); refused "
                    "unless the target's declared command, with check.py's own flags at most, since the gate takes no other")
    ap.add_argument("--send", action="append", default=[], metavar="FILE", help="--train --on: ship this file with every run (research run --send)")
    ap.add_argument("--prepare", default=None, metavar="BRANCH", help="--train: record and land nothing; build the train, run --lint "
                    "on it and push it to --remote as BRANCH (never forced), for `tools/check/train.sh BRANCH MACHINE`")
    ap.add_argument("--onto", default=None, metavar="BASE", help="--prepare: the branch on --remote (or commit) the train starts "
                    "from (default --into; a train ahead's prep branch stacks this one on it)")
    ap.add_argument("--lint", default=None, metavar="CMD", help="--prepare: what must pass on the train before it's pushed "
                    "(default `python tools/check/check.py --preflight`: under this interpreter when it has pytest, else `uv run --locked`)")
    ap.add_argument("--allow-removal", action="append", default=[], metavar="PATH", help="--train: a path where the train may "
                    "lose lines none of its refs removes in a commit of its own (`reverted`; by default the train is refused)")
    ap.add_argument("--into", default="main", help="the branch checked out here that receives the merge (default main)")
    ap.add_argument("--push", action="store_true", help="once the merge lands, push --into to --remote (never forced), then sweep; "
                    "without it the sweep runs before the push and can close nothing the merge landed")
    ap.add_argument("--tool", default="check", help="the tool whose passing attempt the merge requires (default check)")
    ap.add_argument("-m", "--message", default=None, help="the merge commit's message (default `Merge REF (sha)`)")
    ap.add_argument("--mirror", action="store_true", help=f"merge nothing: push REF (default --into's tip) to the main of "
                    f"${MIRROR_ENV}'s bare repository (default {MIRROR}'s), as --push does after the remote")
    ap.add_argument("--label-landed", action="store_true", help="merge nothing: write and push the `landed` label on each run a "
                    "Check: or Checked: trailer on --into's first-parent line names, after REF (default the whole line); a rerun "
                    "writes nothing new")
    ap.add_argument("--dry-run", action="store_true", help="check the gate, merge nothing")
    ap.add_argument("--store", default=None, help="evidence store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--guard-push", action="store_true", help="the pre-push hook `research merge` installs: read git's pre-push "
                    "lines on stdin and refuse a push that adds to --into a commit with no passing --tool attempt of its tree")
    ap.add_argument("--canary", action="store_true", help="merge nothing: put the attempts every gate must refuse (CANARY: no "
                    "result, a result of another kind or a refused one, a record that can't be read or disagrees with the index, "
                    "another command, a failed verdict, a step listed twice) and a control it must admit to the gate, the push guard "
                    "and the train gate, in scratch stores, under this checkout's declaration of --tool; exit 1 unless all are "
                    "refused and the control admitted")
    ap.add_argument("--json", action="store_true", help=f"--canary: print its report as JSON ({CANARY_SCHEMA})")
    ap.add_argument("--verdicts", action="store_true", help="admit REF when the target's `check.py --plan` of it shows every part "
                    "passed under its key by a recorded check's verdict pack, cited in a `Verdicts:` trailer, in place of a passing "
                    "attempt of REF itself (`verdict_gate`); not with --train or --push, since the push guard takes only the latter")
    ap.add_argument("--verdicts-from", action="append", default=[], metavar="RUN", help="--verdicts: also look passes up in "
                    "RUN's pack (`check.py --plan --from`), beside the newest recorded checks'")
    ap.add_argument("--plan-on", default=None, metavar="M1,M2", help="--verdicts: make the plan on each of these check nodes in "
                    "turn (`check.py --plan --on`: a short `research run` there, whose keys are that machine's; give the machines "
                    "the checks ran on, drive.toml's `on`), and admit on the first fully passed")
    a = ap.parse_args(argv)
    if (a.verdicts_from or a.plan_on) and not a.verdicts:
        ap.error("--verdicts-from and --plan-on go with --verdicts")
    if a.verdicts and (a.train or a.push or a.sweep or a.canary or a.guard_push or a.mirror or a.label_landed):
        ap.error("--verdicts gates one ref, without --train, --push or another mode")
    from .store.local import LocalStore

    repo = Path.cwd()
    if a.json and not a.canary:
        ap.error("--json goes with --canary")
    if a.canary:
        if a.ref or a.train or a.sweep or a.guard_push or a.mirror:
            ap.error("--canary takes no refs and no other mode")
        try:
            report = canary(repo, a.tool)
        except Refused as e:
            return refuse(PROG, str(e))
        if a.json:
            print(json.dumps(report, indent=1))
        for c in report["cases"]:
            gates = (("the gate", c["gate"]), ("the push guard", c["push_guard"]), ("the train gate", c["train_gate"]))
            seen = ", ".join(f"{name} {got.split(':')[0]} it" if got else f"{name} doesn't judge it" for name, got in gates)
            wrong = "; ".join(f"{name}: {got}" for name, got in gates if got and got.split(":")[0] != c["want"])
            print(f"research merge: canary: {c['case']} ({c['attempt']}): {seen}" + ("" if c["ok"] else f"; it must be {c['want']}: "
                  f"{wrong}"), file=sys.stderr if a.json else sys.stdout)
        if not report["ok"]:
            print(f"research merge: canary FAILED: a gate admits what it must refuse, or refuses the control (`--tool {a.tool}` "
                  f"produces {', '.join(report['produces']) or 'nothing'})", file=sys.stderr)
        return OK if report["ok"] else FINDING
    if a.guard_push:
        bad = unchecked_pushes(repo, sys.stdin.read().splitlines(), LocalStore(a.store), a.into, a.tool)
        for c in bad:
            refuse(PROG, f"the push: {c[:12]} would reach {a.into} with no passing `{a.tool}` attempt of its tree; land it with "
                         "`research merge` (the hook fails closed: `git push --no-verify` only with root's yes)")
        return REFUSED if bad else OK
    if a.label_landed:
        if len(a.ref) > 1 or a.train or a.sweep or a.mirror:
            ap.error("--label-landed takes at most one ref and no other mode")
        tip = _git(repo, "rev-parse", "--verify", f"refs/heads/{a.into}^{{commit}}").stdout.strip()
        return 0 if label_landed(repo, a.ref[0] if a.ref else "", tip, store=LocalStore(a.store)) else 1
    if not a.sweep and not a.dry_run and (hook := install_push_guard(repo, a.into, a.tool)):
        print(f"research merge: installed the push guard {hook}", file=sys.stderr)
    if a.mirror:
        if len(a.ref) > 1 or a.train or a.sweep:
            ap.error("--mirror takes at most one ref")
        if not (name := mirror_name()):
            print(f"research merge: ${MIRROR_ENV} turns the mirror off", file=sys.stderr)
            return 1
        r = _git(repo, "rev-parse", "--verify", f"{a.ref[0] if a.ref else 'refs/heads/' + a.into}^{{commit}}", check=False)
        if r.returncode:
            print(f"research merge: no commit {a.ref[0] if a.ref else a.into}", file=sys.stderr)
            return 1
        return 0 if push_mirror(repo, r.stdout.strip(), name) else 1
    if (a.into == "main" and not a.dry_run and (a.sweep or a.ref)
            and _git(repo, "remote", "get-url", a.remote, check=False).returncode == 0):
        caught_up(repo, a.remote, a.tool)
    if a.sweep:
        if a.ref or a.train:
            ap.error("--sweep takes no refs")
        try:
            sweep(repo, a.into, a.remote, a.dry_run)
        except (Refused, ValueError, KeyError) as e:
            print(f"research merge: sweep failed: {e}", file=sys.stderr)
            return 1
        if a.into == "main" and not a.dry_run and (lag := spec_alert_lag(repo, a.remote)):
            print(f"research merge: {lag}; rerun `research merge --sweep` once Slack is reachable", file=sys.stderr)
            return 1
        return 0
    if not a.ref:
        ap.error("a ref to merge is required (or --sweep)")
    if a.prepare:
        if not a.train:
            ap.error("--prepare needs --train")
        onto = a.onto or a.into
        lint = shlex.split(a.lint) if a.lint else preflight_argv()
        try:
            tip, tree = prepare(repo, a.ref, a.prepare, onto, a.remote, lint, LocalStore(a.store), tuple(a.allow_removal))
        except Refused as e:
            return refuse(PROG, str(e))
        print(f"research merge: train {' '.join(a.ref)} on {onto}: pushed {a.remote}/{a.prepare} = {tip[:12]} (tree {tree[:12]}); "
              f"check it with `tools/check/train.sh {tip[:12]} MACHINE`, land it with `research merge {a.prepare}`")
        return 0
    if a.train:
        from .store import tools_registry

        decl = tools_registry.load(a.tool, repo)
        command = shlex.split(a.command) if a.command else list(decl.command)
        if why := unlike(command, declared_command(repo, decl.name)):
            return refuse(PROG, f"the train's check would pass no gate: {why}; nothing was checked")
        try:
            land, bad = train(repo, a.ref, a.into, LocalStore(a.store), decl.name, command, a.on, a.project, spec=a.tool, sends=a.send,
                              remote=a.remote, allow_removal=tuple(a.allow_removal))
        except Refused as e:
            return refuse(PROG, str(e))
        moved = f"{a.into} fast-forwarded to {land[:12]}" if land else f"{a.into} unchanged"
        print(f"research merge: train {' '.join(a.ref)}: " + (f"`{decl.name}` passed; {moved}" if bad is None else f"{bad} breaks it; {moved}"))
        pushed = _push_and_sweep(repo, a.into, a.remote, a.push, a.tool) if land else 0
        return OK if bad is None and not pushed else FINDING
    if len(a.ref) != 1:
        ap.error("several refs need --train")
    a.ref = a.ref[0]
    try:
        if a.verdicts:
            commit, cited = verdict_gate(repo, a.ref, a.into, LocalStore(a.store), a.tool, remote=a.remote,
                                         also=tuple(a.verdicts_from), on=tuple(m for m in (a.plan_on or "").split(",") if m))
            attempt, trailer = f"the packs of {', '.join(cited)}, part by part", f"Verdicts: {' '.join(cited)}"
        else:
            commit, attempt = gate(repo, a.ref, a.into, LocalStore(a.store), a.tool, remote=a.remote)
            trailer = f"Check: {attempt}"
    except Refused as e:
        return refuse(PROG, str(e))
    if a.dry_run:
        print(f"research merge: {a.ref} ({commit[:12]}) may be merged into {a.into}: `{a.tool}` passed in {attempt}")
        return 0
    msg = (a.message or f"Merge {a.ref} ({commit[:8]})") + f"\n\n{trailer}"
    r = _git(repo, "merge", "--no-ff", "-m", msg, commit, check=False)
    if r.returncode:
        print(f"research merge: git merge failed: {(r.stderr or r.stdout).strip()}", file=sys.stderr)
        return 1
    same = _git(repo, "rev-parse", "HEAD^{tree}").stdout == _git(repo, "rev-parse", f"{commit}^{{tree}}").stdout
    print(f"research merge: merged {a.ref} ({commit[:12]}) into {a.into}; `{a.tool}` passed in {attempt}"
          + ("" if same else "; the merged tree differs from the checked one" + (", so it isn't pushed" if a.push else "")))
    if not same:
        _sweep_after(repo, a.into, a.remote)
        return 1
    return _push_and_sweep(repo, a.into, a.remote, a.push, a.tool)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
