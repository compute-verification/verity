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

Run in the checkout of the target branch.  It merges the commit REF names (`git merge --no-ff`) only if the evidence store holds
a passing attempt of the tool (default `check`) for that exact commit: `state` done, rc 0, source commit REF's, tree clean, as
the attempt's own record says (the index only finds it); a run of the command the target's declaration of the tool names
(`Tool.command`, in a check slot or not, followed by nothing but check.py's own flags: `unlike`); and its own result, of a kind
that declaration produces (`Tool.produces`: `check/v1` for check; an attempt with no result, or whose record or result can't
be read, passes no gate) and not in `REFUSED_KINDS` (the Lean fast path's `lean-fast/v1`, which skips the kernel replay), with
the verdict passed and no step listed twice.  A sharded check (`check.py --shard K/N`, one Attempt per shard, its result a
`check-shard/v1`) passes only through the Attempt of `check.py --combine RUN...` on the same commit, and only when the gate,
reading the shards itself (`shard_set`), finds them a passing check of that clean commit (one plan, K = 1..N once each, the
commit's own `tools/verity/check/`, every part once, every job of its `ci.toml`, no step failed) and the combine's steps theirs; a
shard alone passes nothing.  With a usable remote the catalog's attempts are refreshed first, so an attempt
recorded on another machine counts.  Refused as well: a
target checkout that is not on BRANCH or has tracked changes, and a REF the target has moved past (the target's tip is not an
ancestor of REF) unless merging REF leaves REF's own tree (`tree_identical`: what the target gained is in REF already, as when a
train checked on the train ahead of it lands after that one), since otherwise the merged tree is not the one that was checked;
merge the target into the branch and check the new tip.  The steps a change requires are read from the target's declaration and
the diff from its tip either way.  The merge commit carries a `Check: <attempt id>` trailer.  Exit 0 merged (or would, with
--dry-run), 1 a failure past the gate (git's merge, the push, the sweep, a train's offending ref), 2 usage, 3 refused
(`research.gates.refuse`: the gate, the push guard, a train or its build).

After a history rewrite (`commitmap`: an accepted `commit-map/v1` in the store), an attempt of an old commit counts for its image when
the rewrite kept the tree, and REF is refused when it holds a commit the rewrite replaced: merging old history would bring it back.

After a merge lands, and on its own with --sweep, every open pull request whose head is on the remote's copy of BRANCH is closed
with a comment naming the commit it landed in, and every open pull request whose base branch is on it is retargeted onto BRANCH.
A closed pull request's head branch is deleted from the remote unless an open pull request still uses it, or it moved past the
head that landed (GitHub's "delete head branch" fires only on merges made on GitHub, and trains merge locally).
The remote's branch, fetched first, is the truth, never the checkout. With --push, a merge that lands is pushed to the remote
(never forced), what the tool's declaration runs after a landing (`Tool.after_land`) is started, detached, on the landed commit,
and then it is swept; without it the sweep runs before anyone pushes, finds nothing the merge landed, and
`research merge --sweep` after the push does it. Whoever lands a train passes --push (Daniel, 1 Oct). Once origin has main, the
same commit goes to vy-nebius-1's bare repository, whose main node 1's builds check out (`vy-lean-sandbox checkout`): a plain
push over the ssh `research run --on` uses, bounded by MIRROR_TIMEOUT_S. One that fails is a line with the command that reruns
it, and the merge stands; $RESEARCH_MERGE_MIRROR names another machine, and empty skips it.
These after-landing steps (the spec DM, the mirror, `after_land`, and one line per landing in the merge thread: what it merged
and its `Checked:` run, `report_landings`) follow the remote's main, not the push: AFTER_LAND_REF on the
remote is the main they last ran for, and every `research merge` that can change something (all but --dry-run, --mirror,
--canary and --guard-push) catches them up, once, for whatever landed past it since, however it was pushed (`caught_up`).

A train's ref may be pinned: `N@SHA` (pull request N; quote `#N@SHA`, or the shell reads a comment) or `BRANCH@SHA` is fetched
from the remote and refused unless its head there is still SHA, so what merges is the head that was reviewed (`pinned`); the gate
reads each again when the tip lands (`unstacked`). It reads no ready mark and no grant, since the train's own check is the gate
(Daniel, 6 and 8 Oct); the owner's ready mark is what admits a PR to a train (`research queue`), not part of this gate.
A ref whose merge onto the train so far conflicts only on import lines (two PRs adding modules to one root import list) keeps
both sides, ours first; a conflict on any other line refuses it, and so does a JSON file whose lines git merged into something
that doesn't parse (`_merge`). Generated files never keep merged text: every merge runs the tree's own
`tools/verity/check/pins.py --merged` before its commit (`regenerate`), which runs `lake update` in each Lake package both sides'
Lake changes reach, re-points its dependency bundle's pin, regenerates every stale pin, and resolves a conflict in any of those
files; the merge commit names what it rewrote in a `Regenerated:` trailer, which `reverted` reads as removals the train means.

A push that lands a change to what a guarantee says (a `lean-audit.json`'s guarantees or the definitions they read) DMs Daniel
what changed, and nothing waits on him (`spec_alert`; his ruling, 3 Oct). A DM that fails is printed and goes to #agent-alerts,
and SPEC_ALERTED_REF stays where it was, so every later `research merge` sends it again and `--sweep` fails until it goes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from .gates import FINDING, OK, REFUSED, git_env, refuse

PROG = "research merge"


class Refused(Exception):
    pass


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, errors="surrogateescape", env=git_env())
    if check and r.returncode:
        raise Refused(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    return r


#: result kinds that pass no merge gate, whatever `--tool` recorded them and whatever its declaration says it produces:
#: `lean-fast/v1` is the Lean fast path's (`tools/verity/lean/lean_changed.py --records`), an audit without the kernel replay,
#: which every merge check keeps
REFUSED_KINDS = frozenset({"lean-fast/v1"})
#: what `check.py --record --on` puts before the declared command (its `slot_wrap`): one of the machine's check slots, then
#: `--priority`, `--by NAME` and `--twin PAIR[@HOME]` when given, then `--`
SLOT_WRAP = ("python3", "tools/verity/check/slot.py")
#: check.py's own flags, as its `main` reads them in a recorded run: these anywhere, `--verdicts-in PACK` once, and then nothing
#: or `--cores N`
CHECK_FLAGS = ("--no-cache", "--keep-going")


def invocation(argv, command) -> tuple[str, object]:
    """What an attempt's ``argv`` ran of the target's declared ``command``: ("check", None), ``command`` itself, in a check slot
    (`SLOT_WRAP`) or not, followed by nothing but check.py's own flags (`CHECK_FLAGS`); ("shard", (K, N)), the same with
    `--shard K/N` once (`tools/verity/check/check.py`'s sharding); ("combine", [run ids]), ``command`` with `--combine RUN...` and
    nothing else; or ("refused", why) for anything else."""
    if not command:
        return "refused", "the target's declaration of the tool names no command to hold it to"
    if not (isinstance(argv, list) and all(isinstance(x, str) for x in argv)):
        return "refused", "its record has no argv"
    run = argv
    if tuple(argv[:2]) == SLOT_WRAP:
        opts = argv[2:]
        opts = opts[1:] if opts[:1] == ["--priority"] else opts
        opts = opts[2:] if opts[:1] == ["--by"] and len(opts) > 1 else opts
        opts = opts[2:] if opts[:1] == ["--twin"] and len(opts) > 1 else opts
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
    if "--shard" in rest[:-1] and (m := re.fullmatch(r"([1-9]\d*)/([1-9]\d*)", rest[rest.index("--shard") + 1])):
        i = rest.index("--shard")
        mode, rest = ("shard", (int(m.group(1)), int(m.group(2)))), rest[:i] + rest[i + 2:]
    if rest and not (rest[0] == "--cores" and len(rest) == 2 and rest[1].isdigit()):
        return "refused", f"its command gives the declared one `{shlex.join(rest)}`, which check.py doesn't take"
    return mode


def unlike(argv, command) -> str | None:
    """Why an attempt's ``argv`` isn't a whole run of the target's declared ``command``, or None (`invocation`'s "check"): a
    shard passes no gate alone, and a combine only through `refused`'s reading of its shards."""
    mode, arg = invocation(argv, command)
    if mode == "shard":
        return (f"it is shard {arg[0]}/{arg[1]} of a sharded check, which passes no gate alone: `check.py --record --combine` "
                "its set")
    if mode == "combine":
        return "it combines shards, which only the gate's own reading of them admits"
    return arg if mode == "refused" else None


#: a sharded check's shard result (`tools/verity/check/check.py --shard`): never a kind the tool produces, read only by `shard_set`
SHARD_KIND = "check-shard/v1"


def declared_jobs(repo: Path, commit: str) -> list[str] | str:
    """The job names ``commit``'s own `tools/verity/check/ci.toml` declares, or why they can't be read."""
    import tomllib

    from .store import tools_registry

    path = tools_registry.resolve("tools/verity/check/ci.toml",
                                  lambda p: _git(repo, "cat-file", "-e", f"{commit}:{p}", check=False).returncode == 0)
    r = _git(repo, "show", f"{commit}:{path or 'tools/verity/check/ci.toml'}", check=False)
    try:
        jobs = [t["name"] for t in tomllib.loads(r.stdout).get("job", [])] if r.returncode == 0 else None
    except (tomllib.TOMLDecodeError, TypeError, KeyError, AttributeError):
        jobs = None
    if not jobs or not all(isinstance(j, str) for j in jobs):
        return f"{commit[:12]}'s tools/verity/check/ci.toml can't be read for the jobs a check runs"
    return jobs


def _read(store, run_id: str, tool: str | None, commit: str | None):
    """(record, result id, result manifest or None) of the attempt, or why it can't pass whatever it ran: its record can't be
    read, or doesn't say it is a passing ``tool`` attempt of the clean ``commit``, or its result can't be read."""
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
    """(the steps of the one check the shard attempts ``ids`` make, None) when together they are a passing check of the clean
    ``commit``, else (None, why).  Each must be a passing ``tool`` attempt of ``commit`` whose command is the declared
    ``command`` with `--shard K/N` (`invocation`), whose own result is a passed `SHARD_KIND` naming that shard, ``commit`` and
    its config (``commit``'s tree of `tools/verity/check/`), with no step but passed or skipped and its steps' parts its shard's parts;
    together, one N with K = 1..N once each, one list of every part, which their parts make exactly once and which covers every
    job ``commit``'s `ci.toml` declares, and the same steps outside their parts (the preflight), passed in each.  The steps: those
    once, then each part as its shard's step ran it."""
    if not ids or len(set(ids)) != len(ids):
        return None, "a set of shards names each shard's attempt once" if ids else "no shard is named"
    config = _git(repo, "rev-parse", "--verify", "-q", f"{commit}:tools/verity/check", check=False).stdout.strip()
    if not config:
        return None, f"{commit[:12]} has no tools/verity/check/ to hold the shards' config to"
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
                          f"tools/verity/check/ ({config[:12]})")
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
        plans.add((arg[1], json.dumps(sh["all"])))
    if len(plans) != 1:
        return None, "the shards are of different plans (their N, or every part they say an unsharded check covers)"
    (n, _), = plans
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
        return None, f"no shard runs {', '.join(uncovered)}, which {commit[:12]}'s ci.toml declares"
    names = {p.partition(":")[0] for p in every}
    common = [[s["step"] for s in shards[k][2] if not s.get("parts")] for k in sorted(shards)]
    if any(sorted(c) != sorted(common[0]) for c in common) or names & set(common[0]):
        return None, "the shards ran different steps outside their parts, or a job outside its parts"
    out = [{"step": s, "status": "passed" if all(x.get("status") == "passed" for k in shards for x in shards[k][2] if x["step"] == s)
            else "skipped", "shards": [shards[k][0] for k in sorted(shards)]} for s in common[0]]
    part = {p: {"step": p, "status": s.get("status"), "shard": k, "run_id": shards[k][0], "seconds": s.get("seconds")}
            for k in sorted(shards) for s in shards[k][2] for p in (s.get("parts") or [])}
    return out + [part[p] for p in every], None


def declaration(repo: Path, tool: str):
    """The target's own declaration of ``tool``, read from the checkout (not from the commit being merged), or None when the
    checkout has none."""
    from .store import tools_registry

    try:
        return tools_registry.from_tree(tool, repo)
    except Exception:  # noqa: BLE001  (a declaration that can't be read is none)
        return None


def produced(repo: Path, tool: str) -> frozenset[str]:
    """The result kinds the target's declaration of ``tool`` produces (`Tool.produces`); none when it has no declaration, so
    that no attempt passes."""
    return frozenset(getattr(declaration(repo, tool), "produces", None) or ())


def declared_command(repo: Path, tool: str) -> tuple[str, ...]:
    """The command the target's declaration of ``tool`` runs (`Tool.command`); none when it has no declaration, so that no
    attempt passes."""
    return tuple(getattr(declaration(repo, tool), "command", None) or ())


def refused(store, run_id: str, kinds: frozenset[str], tool: str | None = None, commit: str | None = None,
            command: tuple[str, ...] | None = None, repo: Path | None = None) -> str | None:
    """Why the attempt can't pass the gate whatever the index says of it, or None: its record can't be read, or doesn't say
    what the index row did (a passing ``tool`` attempt of the clean ``commit``), or ran something other than ``command`` (the
    tool's `declared_command`, `unlike`; with none, nothing passes), or its result can't be read, is in `REFUSED_KINDS`, is not
    of a kind in ``kinds`` (the tool's `produced`; an attempt that published no result has none), is another run's, has a
    verdict other than passed, or lists a step twice (the gates would read it two ways).  A `--combine` of shards passes only
    when, read in ``repo`` (the target's checkout), its shards are a passing check of ``commit`` (`shard_set`) and its steps are
    the ones they make."""
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
    """Why a result's ``steps`` can't be read one way, or None: each is an object naming its step, and no step is listed twice
    (`step_status` would read the first and the train gate the last)."""
    if not (isinstance(steps, list) and all(isinstance(s, dict) and isinstance(s.get("step"), str) for s in steps)):
        return "its result has steps this gate can't read"
    names = [s["step"] for s in steps]
    if again := next((n for i, n in enumerate(names) if n in names[:i]), None):
        return f"its result lists step {again} more than once"
    return None


def passing(store, commit: str, tool: str = "check", *, kinds: frozenset[str], command: tuple[str, ...], tree: str | None = None,
            maps=None, repo: Path | None = None) -> list[str]:
    """Ids of the attempts of ``tool`` that passed on the clean tree of ``commit``, newest first, less those `refused` names
    (``kinds`` and ``command``: the result kinds that may pass and the command that must have run, the target's `produced` and
    `declared_command`).  After a history rewrite (`commitmap`), an attempt of an old commit counts for its image when the map
    says the rewrite kept the tree and, where the attempt recorded its tree, that tree is ``tree``."""
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


def after_landing(store, commit: str, tool: str = "check", *, command: tuple[str, ...]) -> str | None:
    """The newest attempt that passed ``tool``'s after-landing jobs on the clean ``commit`` (`check.py --after-landing`, which
    the landing hook `landed.py` records on each landing of main), or None.  Citations and tables read only commits it names
    (Daniel, 12 Oct).  ``command`` is the target's `declared_command`; the attempt ran it with `--after-landing` and check.py's
    own flags, and its result is a passed `check/v1`.  No merge gate takes such an attempt (`invocation` refuses the flag)."""
    rows = store.index.sql("SELECT id FROM attempts WHERE tool = ? AND source_commit = ? AND source_dirty = 0 AND state = 'done' "
                           "AND rc = 0 ORDER BY start_utc DESC", (tool, commit))
    for (rid,) in rows:
        got = _read(store, rid, tool, commit)
        if isinstance(got, str):
            continue
        doc, _art, result = got
        argv = doc.get("argv")
        if not (isinstance(argv, list) and "--after-landing" in argv) or unlike([x for x in argv if x != "--after-landing"], command):
            continue
        meta = result.meta if result is not None and isinstance(result.meta, dict) else {}
        if result.kind == "check/v1" and meta.get("run_id") == rid and (meta.get("validation") or {}).get("status") == "passed":
            return rid
    return None


def tree_identical(repo: Path, commit: str) -> bool:
    """Whether merging ``commit`` into the checkout's HEAD, which it does not contain, gives ``commit``'s own tree: HEAD's
    changes are all in ``commit`` already (a train checked on the train ahead of it, which landed as a merge commit of the same
    tree), so the tree that would land is the one that was checked."""
    r = _git(repo, "merge-tree", "--write-tree", "HEAD", commit, check=False)
    return r.returncode == 0 and r.stdout.split("\n", 1)[0].strip() == _git(repo, "rev-parse", f"{commit}^{{tree}}").stdout.strip()


def gate(repo: Path, ref: str, into: str, store, tool: str = "check", log=print, remote: str = "origin") -> tuple[str, str]:
    """(commit, attempt id) when ``ref`` may be merged into ``into``; `Refused` naming why otherwise, including a pull request it
    merges whose head on ``remote`` moved, or that is stacked on one neither on ``into`` nor below it (`unstacked`)."""
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
    tree = _git(repo, "rev-parse", f"{commit}^{{tree}}").stdout.strip()
    kinds, command = produced(repo, tool), declared_command(repo, tool)
    ids = passing(store, commit, tool, kinds=kinds, command=command, tree=tree, maps=maps, repo=repo)
    if not ids:
        rows = store.index.sql("SELECT id, state, rc, source_dirty FROM attempts WHERE tool = ? AND source_commit = ?", (tool, commit))
        why = {r[0]: refused(store, r[0], kinds, tool, commit, command, repo) for r in rows if r[1] == "done" and r[2] == 0 and not r[3]}
        seen = "; ".join(f"{r[0]} {r[1]} rc={r[2]}{' dirty' if r[3] else ''}{f' refused: {why[r[0]]}' if why.get(r[0]) else ''}"
                         for r in rows) or "none"
        raise Refused(f"no passing `{tool}` attempt for {commit} in the evidence store (attempts of it: {seen}); "
                      f"record one: `uv run python tools/verity/check/check.py --record` on that commit")
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
                  f"`uv run python tools/verity/check/check.py --record --on MACHINE` sends the agreement inputs)")


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
    """The tips a push (git's pre-push lines: `local_ref local_sha remote_ref remote_sha`) would move ``into`` to without a
    passing ``tool`` attempt of their tree, the merge gate's rule: the tip's own attempt (a `--train` fast-forward), or, as a
    `research merge` commit has, its merged parent's when the two trees are equal."""
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
                   and passing(store, x, tool, kinds=kinds, command=command, tree=tree, repo=repo) for x in [tip, *parents[1:]]):
            bad.append(tip)
    return bad


CANARY_SCHEMA = "research/merge-canary/v1"
#: `--canary`'s attempts of one commit, each alone in a store: (case, what it is, whether the train gate judges it: that gate
#: holds its job's own run to the record and the result's kind and steps, not to the command or the verdict); every gate that
#: judges a case must refuse all but the control
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
    """The `research/merge-canary/v1` report of `--canary`: each CANARY attempt, alone in a scratch store with no remote, put to
    the gate and the push guard for a commit of a scratch repository whose declaration of ``tool`` is ``repo``'s own, and, for
    `check`, to the train gate (`jobs.trains.TrainService.gate`, on its fakes holding the same declaration, record and result);
    ``ok`` when every gate that judges a case refuses every bad one and admits the control.  Nothing of ``repo``'s or the
    evidence store's is read or written beyond that declaration."""
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
    """The pre-push hook (`--guard-push`) in ``repo``'s hooks directory, unless one is there already; its path when written."""
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
    """The steps the target's own declaration of ``tool`` (`Tool.merge_requires`, read from the checkout, not from the commit being
    merged) requires for the change ``base``..``commit``: {step: the prefixes it touches}."""
    rules = getattr(declaration(repo, tool), "merge_requires", None)
    if not rules:
        return {}
    changed = [p for p in _git(repo, "diff", "--no-renames", "--name-only", "-z", base, commit).stdout.split("\0") if p]   # a rename's source too
    out = {}
    for step, rule in rules.items():
        if hit := rule_touches(rule, changed):
            out[step] = hit
    return out


def rule_touches(rule: list[str], changed: list[str]) -> list[str]:
    """The prefixes of a `Tool.merge_requires` rule that ``changed`` touches.  A rule is path prefixes and `!` patterns: a
    changed file counts when it is under a prefix and matches no pattern (one without a `/` is matched against the file's name,
    one with a `/` against its path), so `["verity/ml/flock/", "verity/core/service/flock/", "!README.md"]` leaves out every
    README.md under either."""
    import fnmatch

    prefixes = [r for r in rule if not r.startswith("!")]
    ignored = [r[1:] for r in rule if r.startswith("!")]

    def counts(c: str) -> bool:
        return not any(fnmatch.fnmatchcase(c if "/" in g else c.rsplit("/", 1)[-1], g) for g in ignored)

    return [p for p in prefixes if any((c == p.rstrip("/") or c.startswith(p.rstrip("/") + "/")) and counts(c) for c in changed)]


def step_status(store, run_id: str, step: str) -> str:
    """The status ``step`` has in the attempt's result.json ``steps`` ("absent" when it has none)."""
    try:
        art = (store.get_attempt(run_id) or {}).get("outputs", {}).get("result")
        doc = store.get_manifest(art).meta if art else {}       # a result artifact's document is its manifest's meta
    except Exception as e:  # noqa: BLE001  (an unreadable result admits nothing)
        return f"unreadable ({type(e).__name__})"
    return next((s.get("status", "absent") for s in doc.get("steps", []) if s.get("step") == step), "absent")


def _record(tree: Path, tool: str, command: list[str], on: str | None, project: str | None, log=print,
            sends: list[str] = ()) -> str:
    """Record ``tool`` on the commit checked out in ``tree`` (``research run --tool``; ``--on`` a machine: launched there and
    polled until it ends); returns the run id."""
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


#: a pinned train ref: pull request N or a branch of the remote, at the head that was reviewed
PIN = re.compile(r"#?([\w./-]+)@([0-9a-f]{7,40})")


def pinned(repo: Path, ref: str, remote: str) -> tuple[str, str]:
    """(how the train's merge message names ``ref``, the commit it merges).  ``N@SHA`` (pull request N) and ``BRANCH@SHA`` are
    fetched from ``remote`` and refused unless the head there is still SHA; any other ref is the commit it names here."""
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


#: a line a train's merge takes from both sides of a conflict: a Lean or Python import
IMPORT_LINE = re.compile(r"\s*(import\s|from\s+\S+\s+import\s)")


def keep_both_imports(text: str) -> str | None:
    """``text``, a file git left with conflict markers (`merge` style), with each conflict resolved to both sides, ours and then
    the lines of theirs ours lacks, when every line on either side of every conflict is an import or blank; None when one holds
    anything else, or there is no conflict to resolve."""
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
    """Whether ``text`` parses as JSON with no key twice in one object (a text merge of two added records can repeat a key)."""
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
    """The JSON files whose text the merge at ``tree``'s HEAD merged (both sides changed them, as two PRs' guarantees change one
    `lean-audit.json`) and that don't parse (`_strict_json`) though both sides' did."""
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


#: the tree's own list of generated files and how each is regenerated after a merge (`pins.py --merged`)
PINS_SCRIPT = "tools/verity/check/pins.py"


def regenerate(tree: Path, ours: str, theirs: str, conflicted: list[str]) -> list[str]:
    """Run the tree's `pins.py --merged OURS THEIRS` on the merge under way in ``tree`` and stage what it rewrote; the paths.
    Refused when a conflict is in a file it doesn't generate or a regeneration failed. A tree without it (one that predates
    it) resolves nothing."""
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
    """Merge ``sha`` onto the worktree ``tree``'s HEAD (`--no-ff`).  A conflict made only of import lines keeps both sides
    (`keep_both_imports`), named in the commit's `Imports:` trailer; any other refuses.  A merged JSON file must parse
    (`broken_json`)."""
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
    """Merge each (label, commit) in order onto the worktree ``tree``'s HEAD, one `--no-ff` merge each (`_merge`); the chain's
    commits, which are the train's prefixes.  Refused at the first that doesn't merge."""
    chain = []
    for label, sha in pins:
        _merge(tree, label, sha, f"Merge {label} ({sha[:8]})\n\nTrain: {stamp}")
        chain.append(_git(tree, "rev-parse", "HEAD").stdout.strip())
    return chain


#: the subject `_chain` gives a pinned pull request's merge onto a train
MERGED_PR = re.compile(r"Merge #(\d+) \(([0-9a-f]{7,40})\)")


def adds_nothing(repo: Path, commit: str, head: str) -> bool:
    """Whether merging ``head`` into ``commit`` leaves ``commit``'s own tree: what ``head`` has, ``commit`` has already (a merge of
    `main` into a pull request the tip already holds)."""
    r = _git(repo, "merge-tree", "--write-tree", commit, head, check=False)
    return r.returncode == 0 and r.stdout.split("\n", 1)[0].strip() == _git(repo, "rev-parse", f"{commit}^{{tree}}").stdout.strip()


def unstacked(repo: Path, commit: str, base: str, remote: str, log=print) -> list[str]:
    """The structural part of the merge gate, which no label can fix: why ``commit`` can't land on ``base`` for a pull request it
    merges above ``base`` (a first-parent `Merge #N (SHA)` commit `_chain` wrote, whose second parent is the pinned head).  The
    merge names one head and merges another; N's head on ``remote`` is no longer that head, closed or not, and merging the new one
    would change ``commit`` (`adds_nothing`; a head that only merged `main` in lands as the pinned one); or N is stacked on another
    pull request whose head on ``remote`` is in what N adds, so neither on main nor pinned below N at that head.  What can't be
    read refuses."""
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
            # a pull request whose head is N's own pinned head lands whole with N: it is no base left behind
            for m in prs.get(h, []) if h != sha else ():
                # rev-list's walk can stop early on skewed commit dates and list a commit `onto` has
                if m != n and _git(repo, "merge-base", "--is-ancestor", h, onto, check=False).returncode:
                    out.append(f"#{n} at {sha[:12]} is stacked on #{m} at {h[:12]}, which is neither on main nor pinned below "
                               f"it in the tip at that head: pin {m}@{h[:9]} before it, or rebase #{n} off it")
    return out


#: what `prepare` runs on a train before pushing it: check's preflight (`uv.lock`, the repository's lints)
PREFLIGHT = ["tools/verity/check/check.py", "--preflight"]


def preflight_argv() -> list[str]:
    """`prepare`'s default lint: the preflight under this interpreter when it has pytest, which the lints need and a system
    python3 lacks, else under `uv run --locked`, as check itself runs."""
    import importlib.util
    import shutil

    uv = shutil.which("uv")
    if importlib.util.find_spec("pytest") is not None or uv is None:
        return [sys.executable, *PREFLIGHT]
    return [uv, "run", "--locked", "python", *PREFLIGHT]


def _removals(text: str) -> dict[str, Counter]:
    """The lines a `git diff -U0 --no-renames` (or `git log -p` of the same form) removes, by the path they're removed from;
    blank lines left out."""
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
    """What the train ``tip`` removes from ``base`` that no ref of ``pins`` removes in a commit of its own (a non-merge commit
    of ``base..SHA``; a layout move's restack commit, trailer `Restack-move`, counts as none): a line a merge in a ref's history
    dropped, as #1142's merge of main dropped #1012's lines and its train `25157eb17` silently undid #1012. Lines match by
    content, wherever a ref's commit removed them, so a removal survives a move of its file. One line per path, none for a
    path in ``allow`` or in a merge's `Regenerated:` trailer (a generated file, rewritten by its command: `regenerate`)."""
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


#: how a train that `reverted` refuses goes in
REVERTED_HINT = (". A merge in a ref's history dropped them: merge the target into it again, keeping both sides, or pass "
                 "--allow-removal PATH for a removal the train means")


def prepare(repo: Path, refs: list[str], branch: str, onto: str, remote: str, lint: list[str], store,
            allow_removal: tuple[str, ...] = ()) -> tuple[str, str]:
    """Build a train for its check without recording or landing anything: ``refs`` (`pinned`), merged in order onto ``onto`` as ``remote`` has it (the target, or the prep branch of a train ahead, to stack
    on it) in a scratch worktree, refused at the first that doesn't merge, when the merge gate's structural checks (`unstacked`)
    would refuse it whatever its labels, or when the result loses lines no ref removes itself (`reverted`, but for the paths of
    ``allow_removal``); ``lint`` run on the result there; its tip pushed to
    ``remote`` as ``branch``, never forced, and read back.  Every step names the scratch worktree or ``repo``'s object store,
    never its checkout.  Returns (the tip, its tree)."""
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
    """Merge ``refs`` in order onto ``into``'s tip as one candidate (a chain of `--no-ff` merges in a scratch worktree), record
    `tool` on it once and fast-forward ``into`` to it when it passes: the tree that lands is the one checked.  When it fails,
    bisect the chain's prefixes (each an exact commit, checked the same way) to the first ref whose merge makes it fail, and
    fast-forward to the longest prefix that passed.  ``tool`` is the tool's name, ``spec`` how `research run --tool` loads it
    (default the name).  Returns (the commit ``into`` now has, when it moved; the offending ref)."""
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
                          f"{', '.join(need)}: pass what it needs with --send (for `check`: `uv run python tools/verity/check/check.py "
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


def _gh_cli(repo: Path, *args: str) -> str:
    r = subprocess.run(["gh", *args], cwd=repo, capture_output=True, text=True)
    if r.returncode:
        raise Refused(f"gh {' '.join(args[:2])}: {(r.stderr or r.stdout).strip()[-300:]}")
    return r.stdout


def landed_in(repo: Path, commit: str, tip: str) -> str:
    """The commit on ``tip``'s first-parent line that brought ``commit`` in: ``commit`` itself when it is on that line, otherwise
    the oldest commit of the line that contains it (the merge that landed it)."""
    after = set(_git(repo, "rev-list", "--ancestry-path", f"{commit}..{tip}").stdout.split())
    landed = commit
    for c in _git(repo, "rev-list", "--first-parent", "-n", str(len(after) + 1), tip).stdout.split():
        if c not in after:
            return commit if c == commit else landed
        landed = c
    return landed


def sweep(repo: Path, into: str = "main", remote: str = "origin", dry_run: bool = False, gh=None, log=print) -> tuple[list[int], list[int]]:
    """Close every open pull request whose head is on ``remote``'s ``into``, or whose pinned head a train landed and whose head since
    adds nothing to it (`adds_nothing`; its branch is kept), with a comment naming the commit it landed in, and
    retarget onto ``into`` every open pull request whose base branch's tip is on it, deleting each closed one's head branch that no
    open pull request uses.  Idempotent: what it closed is no longer open, and what it retargeted is based on ``into``.  Returns
    (closed, retargeted) PR numbers, ascending."""
    gh = gh or (lambda *a: _gh_cli(repo, *a))
    _git(repo, "fetch", "-q", remote, f"+refs/heads/{into}:refs/remotes/{remote}/{into}")
    tip = _git(repo, "rev-parse", f"refs/remotes/{remote}/{into}").stdout.strip()

    def on(sha: str) -> bool:
        return _git(repo, "merge-base", "--is-ancestor", sha, tip, check=False).returncode == 0

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
    # a PR whose pinned head landed in a train and whose head since adds nothing to `into` (it merged `main` in)
    pinned = {int(m[1]): parents.split()[1] for parents, _, subject in (line.partition("\0") for line in _git(
        repo, "log", "-n", "5000", "--format=%P%x00%s", "--grep", r"^Merge #[0-9]", tip).stdout.splitlines())
        if (m := MERGED_PR.fullmatch(subject)) and len(parents.split()) == 2}
    done += [p for p in rest if p not in moved and p["number"] in pinned and adds_nothing(repo, tip, p["headRefOid"])]
    rest = [p for p in rest if p not in done]
    verb = "would " if dry_run else ""
    for p in moved:
        log(f"research merge: {verb}retarget #{p['number']} from {p['baseRefName']} (on {remote}/{into}) onto {into}")
        if not dry_run:
            gh("pr", "edit", str(p["number"]), "--base", into)
    # a closed PR's head branch goes too, unless an open PR still uses it (as its head, or as a base the retargets left)
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
            # the lease keeps a branch that moved past the PR's landed head: a commit pushed after the landing isn't on `into`
            r = _git(repo, "push", "-q", f"--force-with-lease=refs/heads/{head}:{p['headRefOid']}", remote, f":refs/heads/{head}",
                     check=False)
            if r.returncode:
                log(f"research merge: kept {head}: {(r.stderr or r.stdout).strip()[-200:]}")
    if not done and not moved:
        log(f"research merge: no open pull request is on {remote}/{into} ({tip[:12]}) or based on a branch that is")
    return [p["number"] for p in done], [p["number"] for p in moved]


def _sweep_after(repo: Path, into: str, remote: str) -> None:
    """The sweep after a merge landed: skipped without the remote; a failure is reported and leaves the merge's result alone."""
    if _git(repo, "remote", "get-url", remote, check=False).returncode:
        print(f"research merge: no remote {remote!r}; pull request sweep skipped")
        return
    try:
        sweep(repo, into, remote)
    except (Refused, ValueError, KeyError) as e:
        print(f"research merge: pull request sweep failed ({e}); run `research merge --sweep` once GitHub is reachable",
              file=sys.stderr)


MIRROR, MIRROR_ENV, MIRROR_TIMEOUT_S = "vy-nebius-1", "RESEARCH_MERGE_MIRROR", 120


def mirror_name() -> str | None:
    name = os.environ.get(MIRROR_ENV, MIRROR).strip()
    return None if name.lower() in ("", "none", "off") else name


def _mirror_target(name: str) -> tuple[list[str], str]:
    """``name``'s ssh argv, as `research deploy` and `research run --on` reach it, and its bare repository."""
    from . import remote

    m = remote.resolve_machine(name, "verity")
    t = remote.ssh_host_target(m)
    return remote.ssh_argv(t["host"], t["port"], user=t["user"]), f"{m['root']}/{remote.POD_GIT_REPO}"


def push_mirror(repo: Path, sha: str, name: str, target=None, run=subprocess.run) -> bool:
    """``sha`` to the main of ``name``'s bare repository: a plain push, so only a fast-forward, past this checkout's own pre-push
    guard, which judges pushes that move origin's main (a hand landing has no check attempt of its tree, and the guard refused
    node 1's copy of 61f64824c and 24aa9a9a4, 7 Oct). False, after one line naming the failure and the command that reruns it,
    when it doesn't land."""
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


def after_land(repo: Path, tool: str, commit: str, log=print, spawn=subprocess.Popen) -> list[str] | None:
    """Start what the target's own declaration of ``tool`` runs after a landing (`Tool.after_land`, read from the checkout as
    `required_steps` reads `merge_requires`) on ``commit``, in a session of its own, so it outlives this process and nothing it
    does changes the merge's result.  Returns the argv started, None when the declaration names none."""
    from .store import tools_registry

    try:
        argv = [*tools_registry.from_tree(tool, repo).after_land]
    except Exception:  # noqa: BLE001  (a tool the checkout does not declare runs nothing after a landing)
        return None
    if not argv:
        return None
    argv.append(commit)
    out = Path(tempfile.gettempdir()) / f"research-after-land-{commit[:12]}.log"
    try:
        with open(out, "ab") as f:
            spawn(argv, cwd=repo, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as e:
        log(f"research merge: after the landing, `{shlex.join(argv)}` did not start ({e}); run it by hand")
        return argv
    log(f"research merge: after the landing, started `{shlex.join(argv)}` (log {out})")
    return argv


#: the remote's ref holding the main commit the after-landing steps last ran for
AFTER_LAND_REF = "refs/research/after-land"
#: the remote's ref holding the main commit Daniel was last sent the spec change up to: it moves only when the DM went, so one
#: that failed is sent again by the next `research merge`, and `--sweep` fails while it lags main
SPEC_ALERTED_REF = "refs/research/spec-alerted"


def _remote_refs(repo: Path, remote: str, *refs: str) -> dict[str, str] | None:
    r = _git(repo, "ls-remote", remote, *refs, check=False)
    return None if r.returncode else {ref: sha for sha, ref in (line.split() for line in r.stdout.splitlines())}


def _move(repo: Path, remote: str, ref: str, was: str, tip: str, log=print) -> bool:
    """Push ``tip`` to ``remote``'s ``ref``, leased on ``was`` (the one read) so two landers don't both advance it."""
    p = _git(repo, "push", "-q", f"--force-with-lease={ref}:{was}", remote, f"{tip}:{ref}", check=False)
    if p.returncode:
        log(f"research merge: {remote}'s {ref} wasn't moved to {tip[:12]} ({(p.stderr or p.stdout).strip()[-200:]}); "
            "another lander may have caught up first")
    return not p.returncode


def caught_up(repo: Path, remote: str, tool: str = "check", since: str = "", log=print) -> str | None:
    """Run the after-landing steps once for what landed on ``remote``'s main since AFTER_LAND_REF there (or ``since``, main before
    the caller's own push, when the remote has no marker yet): push main's tip to node 1's mirror, start ``tool``'s `after_land`
    on the tip (a later landing's `post_train.py` tests the ones before it too), post each landing to the merge thread
    (`report_landings`), then move the marker to the tip.  The spec DM
    (`spec_alert`) goes over SPEC_ALERTED_REF..tip (AFTER_LAND_REF's start when that marker is new) and moves its marker only
    when it was sent, so a failed DM is retried by every later run until it goes.  Without a marker or ``since`` it only sets
    the markers.  Nothing raises, since a landing that happened stands.  Returns the tip the steps ran for."""
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
            log(f"research merge: Daniel was last sent the spec change up to {spec_from[:12]}, which isn't in {remote}'s main "
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
        after_land(repo, tool, tip, log=log)
        if remote == "origin":
            report_landings(repo, start, tip, log=log)
    if _move(repo, remote, AFTER_LAND_REF, mark, tip, log) and not start:
        log(f"research merge: {remote} had no {AFTER_LAND_REF}; set it to main ({tip[:12]}): the steps after a landing are "
            "caught up from here, and nothing before it was looked at")
    return tip if start else None


#: what a landing merged, in its own subject or its train's (`prepare`): a pull request, a branch of origin, or a commit
MERGED_REF = re.compile(r"Merge (#\d+|origin/\S+|[0-9a-f]{12,40}) \([0-9a-f]{7,40}\)")
CHECK_TRAILER = re.compile(r"^Check(?:ed)?: (.*)$", re.M)
#: a run id, or the lander's `r20261010-*-9827` for one whose launch time it didn't copy
RUN_ID = re.compile(r"r20\d{6}-(?:\d{6}|\*)-[0-9a-f]{4}")


def _merged_name(ref: str, prs: dict[str, int]) -> str:
    if ref.startswith("#"):
        return ref
    if (branch := ref.removeprefix("origin/")) != ref:
        return f"#{prs[branch]}" if branch in prs else branch
    sha = next((k for k in prs if len(k) == 40 and k.startswith(ref)), None)
    return f"#{prs[sha]}" if sha else ref[:12]


def landing_lines(repo: Path, start: str, tip: str, prs: dict[str, int] | None = None) -> list[str]:
    """One line per commit on ``tip``'s first-parent line after ``start``: what it merged (pull requests, or the branches and
    commits ``prs``, head branch or head sha to number, doesn't map to one) and the runs its `Check:` or `Checked:` trailer
    names.  Read from main's log, since a pull request's state can't stand in: a stacked member such as #1706 reads closed on
    GitHub without having landed on its own."""
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
                if len(parents) == 2 and parents[1].startswith(m[1]):
                    continue                            # the train's own tip, whose merges name what it holds
                if (ref := _merged_name(m[1], prs)) not in merged:
                    merged.append(ref)
        what = " + ".join(reversed(merged)) if merged else f"“{subject}”"
        runs = [r for t in CHECK_TRAILER.findall(body) for r in RUN_ID.findall(t)]
        lines.append(f"landed on main: {c[:12]} {what}; checked by {', '.join(dict.fromkeys(runs)) or 'no run (no Check: trailer)'}")
    return lines


def report_landings(repo: Path, start: str, tip: str, *, api=None, reg: dict | None = None, gh=None, log=print) -> bool:
    """Post `landing_lines` over ``start``..``tip`` to the merge thread, one line per landing, so every landing is reported
    whoever pushed it (two of 10 Oct's were missing from the lander's own reports).  True when posted or there was nothing to
    post; a failure is printed and returns False, and nothing raises, since the landings stand."""
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
        lines = landing_lines(repo, start, tip, prs)
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
    """Why the spec DM is behind ``remote``'s main ("" when it isn't, or there is no marker yet to be behind)."""
    refs = _remote_refs(repo, remote, "refs/heads/main", SPEC_ALERTED_REF) or {}
    tip, alerted = refs.get("refs/heads/main", ""), refs.get(SPEC_ALERTED_REF, "")
    if not alerted or alerted == tip:
        return ""
    return (f"{remote}'s {SPEC_ALERTED_REF} is {alerted[:12]}, behind main ({tip[:12]}): the spec DM over that range "
            "hasn't been sent")


def _push_and_sweep(repo: Path, into: str, remote: str, push: bool, tool: str = "check") -> int:
    """After a merge landed on ``into`` here: with ``push``, push it to ``remote`` (never forced), DM Daniel the spec change it
    landed on main (`spec_alert`), start what ``tool`` runs after a landing (`after_land`) and sweep, since the sweep closes only
    what is on the remote; a refused push sweeps nothing and fails. Without ``push`` the sweep finds nothing the merge landed,
    says so, and `research merge --sweep` after the push does it. On main the steps are `caught_up`'s, from the marker (or main
    before the push), so they cover a landing pushed another way too; the mirror can fail without failing the merge."""
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
                    "on it and push it to --remote as BRANCH (never forced), for `tools/verity/check/train.sh BRANCH MACHINE`")
    ap.add_argument("--onto", default=None, metavar="BASE", help="--prepare: the branch on --remote (or commit) the train starts "
                    "from (default --into; a train ahead's prep branch stacks this one on it)")
    ap.add_argument("--lint", default=None, metavar="CMD", help="--prepare: what must pass on the train before it's pushed "
                    "(default `python tools/verity/check/check.py --preflight`: under this interpreter when it has pytest, else `uv run --locked`)")
    ap.add_argument("--allow-removal", action="append", default=[], metavar="PATH", help="--train: a path where the train may "
                    "lose lines none of its refs removes in a commit of its own (`reverted`; by default the train is refused)")
    ap.add_argument("--into", default="main", help="the branch checked out here that receives the merge (default main)")
    ap.add_argument("--push", action="store_true", help="once the merge lands, push --into to --remote (never forced), then sweep; "
                    "without it the sweep runs before the push and can close nothing the merge landed")
    ap.add_argument("--tool", default="check", help="the tool whose passing attempt the merge requires (default check)")
    ap.add_argument("-m", "--message", default=None, help="the merge commit's message (default `Merge REF (sha)`)")
    ap.add_argument("--mirror", action="store_true", help=f"merge nothing: push REF (default --into's tip) to the main of "
                    f"${MIRROR_ENV}'s bare repository (default {MIRROR}'s), as --push does after the remote")
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
    a = ap.parse_args(argv)
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
              f"check it with `tools/verity/check/train.sh {tip[:12]} MACHINE`, land it with `research merge {a.prepare}`")
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
        commit, attempt = gate(repo, a.ref, a.into, LocalStore(a.store), a.tool, remote=a.remote)
    except Refused as e:
        return refuse(PROG, str(e))
    if a.dry_run:
        print(f"research merge: {a.ref} ({commit[:12]}) may be merged into {a.into}: `{a.tool}` passed in {attempt}")
        return 0
    msg = (a.message or f"Merge {a.ref} ({commit[:8]})") + f"\n\nCheck: {attempt}"
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
