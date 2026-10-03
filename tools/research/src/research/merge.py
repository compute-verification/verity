"""`research merge REF [--into BRANCH] [--tool check] [-m MESSAGE] [--push] [--dry-run]`: the merge gate.
`research merge --train A B C [--on MACHINE --project P] [--command CMD] [--push]`: several refs as one checked candidate (`train`).
`research merge --train --prepare BRANCH [--onto BASE] N@SHA ...`: build a train and push it as BRANCH for its check (`prepare`).
`research merge --sweep [--into BRANCH] [--remote origin] [--dry-run]`: close the pull requests that landed and delete their branches
(`sweep`).
`research merge --canary [--tool check] [--json]`: the gates' self-test, run daily: attempts the merge gate, the push guard and the
train gate must refuse and a control they must admit, in scratch stores (`canary`).

Run in the checkout of the target branch.  It merges the commit REF names (`git merge --no-ff`) only if the evidence store holds
a passing attempt of the tool (default `check`) for that exact commit: `state` done, rc 0, source commit REF's, tree clean, as
the attempt's own record says (the index only finds it); a run of the command the target's declaration of the tool names
(`Tool.command`, in a check slot or not, followed by nothing but check.py's own flags: `unlike`); and its own result, of a kind
that declaration produces (`Tool.produces`: `check/v1` for check; an attempt with no result, or whose record or result can't
be read, passes no gate) and not in `REFUSED_KINDS` (the Lean fast path's `lean-fast/v1`, which skips the kernel replay), with
the verdict passed and no step listed twice.  With a usable remote the catalog's attempts are refreshed first, so an attempt
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
(never forced) and then swept; without it the sweep runs before anyone pushes, finds nothing the merge landed, and
`research merge --sweep` after the push does it. Whoever lands a train passes --push (Daniel, 1 Oct).

A train's ref may be pinned: `N@SHA` (pull request N; quote `#N@SHA`, or the shell reads a comment) or `BRANCH@SHA` is fetched
from the remote and refused unless its head there is still SHA, so what merges is the head that was reviewed (`pinned`).
A ref whose merge onto the train so far conflicts only on import lines (two PRs adding modules to one root import list) keeps
both sides, ours first; a conflict on any other line refuses it, and so does a JSON file whose lines git merged into something
that doesn't parse (`_merge`).  Before a train is built, each ref whose change needs the statement-reviewer grant under
`research queue`'s rules (a change to the `pins` or `reads` of a `lean-audit.json`) is refused until pull request N holds it at
that head (`ungranted`).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .gates import FINDING, OK, REFUSED, git_env, refuse

PROG = "research merge"


class Refused(Exception):
    pass


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, env=git_env())
    if check and r.returncode:
        raise Refused(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    return r


#: result kinds that pass no merge gate, whatever `--tool` recorded them and whatever its declaration says it produces:
#: `lean-fast/v1` is the Lean fast path's (`tools/check/lean_changed.py --records`), an audit without the kernel replay,
#: which every merge check keeps
REFUSED_KINDS = frozenset({"lean-fast/v1"})
#: what `check.py --record --on` puts before the declared command (its `slot_wrap`): one of the machine's check slots, then
#: `--priority` and `--by NAME` when given, then `--`
SLOT_WRAP = ("python3", "tools/check/slot.py")
#: check.py's own flags, as its `main` reads them in a recorded run: these anywhere, `--verdicts-in PACK` once, and then nothing
#: or `--cores N`
CHECK_FLAGS = ("--no-cache", "--keep-going")


def unlike(argv, command) -> str | None:
    """Why an attempt's ``argv`` isn't a run of the target's declared ``command``, or None: it is ``command`` itself, in a check
    slot (`SLOT_WRAP`) or not, followed by nothing but check.py's own flags (`CHECK_FLAGS`)."""
    if not command:
        return "the target's declaration of the tool names no command to hold it to"
    if not (isinstance(argv, list) and all(isinstance(x, str) for x in argv)):
        return "its record has no argv"
    run = argv
    if tuple(argv[:2]) == SLOT_WRAP:
        opts = argv[2:]
        opts = opts[1:] if opts[:1] == ["--priority"] else opts
        opts = opts[2:] if opts[:1] == ["--by"] and len(opts) > 1 else opts
        run = opts[1:] if opts[:1] == ["--"] else argv
    if run[:len(command)] != list(command):
        return f"its command `{shlex.join(argv)}` isn't the target's declared `{shlex.join(command)}`"
    rest = [x for x in run[len(command):] if x not in CHECK_FLAGS]
    if "--verdicts-in" in rest[:-1]:
        i = rest.index("--verdicts-in")
        rest = rest[:i] + rest[i + 2:]
    if rest and not (rest[0] == "--cores" and len(rest) == 2 and rest[1].isdigit()):
        return f"its command gives the declared one `{shlex.join(rest)}`, which check.py doesn't take"
    return None


def declaration(repo: Path, tool: str):
    """The target's own declaration of ``tool``, read from the checkout (not from the commit being merged), or None when the
    checkout has none."""
    from .store import tools_registry

    module, _, attr = tools_registry.REGISTRY.get(tool, tool).rpartition(":")
    try:
        return tools_registry._from_file(module, attr, repo)
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
            command: tuple[str, ...] | None = None) -> str | None:
    """Why the attempt can't pass the gate whatever the index says of it, or None: its record can't be read, or doesn't say
    what the index row did (a passing ``tool`` attempt of the clean ``commit``), or ran something other than ``command`` (the
    tool's `declared_command`, `unlike`; with none, nothing passes), or its result can't be read, is in `REFUSED_KINDS`, is not
    of a kind in ``kinds`` (the tool's `produced`; an attempt that published no result has none), is another run's, has a
    verdict other than passed, or lists a step twice (the gates would read it two ways)."""
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
    if why := unlike(doc.get("argv"), command):
        return why
    art = (doc.get("outputs") or {}).get("result")
    try:
        result = store.get_manifest(art) if art else None
    except Exception as e:  # noqa: BLE001  (a guard never fails open)
        return f"its result can't be read ({type(e).__name__}: {e})"
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
    return twice(meta.get("steps", []))


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
            maps=None) -> list[str]:
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
    return [rid for rid, c in found if refused(store, rid, kinds, tool, c, command) is None]


def tree_identical(repo: Path, commit: str) -> bool:
    """Whether merging ``commit`` into the checkout's HEAD, which it does not contain, gives ``commit``'s own tree: HEAD's
    changes are all in ``commit`` already (a train checked on the train ahead of it, which landed as a merge commit of the same
    tree), so the tree that would land is the one that was checked."""
    r = _git(repo, "merge-tree", "--write-tree", "HEAD", commit, check=False)
    return r.returncode == 0 and r.stdout.split("\n", 1)[0].strip() == _git(repo, "rev-parse", f"{commit}^{{tree}}").stdout.strip()


def gate(repo: Path, ref: str, into: str, store, tool: str = "check", log=print) -> tuple[str, str]:
    """(commit, attempt id) when ``ref`` may be merged into ``into``; `Refused` naming why otherwise."""
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
    tree = _git(repo, "rev-parse", f"{commit}^{{tree}}").stdout.strip()
    kinds, command = produced(repo, tool), declared_command(repo, tool)
    ids = passing(store, commit, tool, kinds=kinds, command=command, tree=tree, maps=maps)
    if not ids:
        rows = store.index.sql("SELECT id, state, rc, source_dirty FROM attempts WHERE tool = ? AND source_commit = ?", (tool, commit))
        why = {r[0]: refused(store, r[0], kinds, tool, commit, command) for r in rows if r[1] == "done" and r[2] == 0 and not r[3]}
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
                   and passing(store, x, tool, kinds=kinds, command=command, tree=tree) for x in [tip, *parents[1:]]):
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

    rel = Path(*tools_registry.REGISTRY.get(tool, tool).rpartition(":")[0].split(".")).with_suffix(".py")
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
    changed = [p for p in _git(repo, "diff", "--name-only", base, commit).stdout.splitlines() if p]
    out = {}
    for step, rule in rules.items():
        if hit := rule_touches(rule, changed):
            out[step] = hit
    return out


def rule_touches(rule: list[str], changed: list[str]) -> list[str]:
    """The prefixes of a `Tool.merge_requires` rule that ``changed`` touches.  A rule is path prefixes and `!` patterns: a
    changed file counts when it is under a prefix and matches no pattern (one without a `/` is matched against the file's name,
    one with a `/` against its path), so `["backends/flock/", "!README.md"]` leaves out every README.md there."""
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
    """The JSON files whose text the merge at ``tree``'s HEAD merged (both sides changed them, as two PRs' pins change one
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


def _merge(tree: Path, label: str, sha: str, msg: str) -> None:
    """Merge ``sha`` onto the worktree ``tree``'s HEAD (`--no-ff`).  A conflict made only of import lines keeps both sides
    (`keep_both_imports`), named in the commit's `Imports:` trailer; any other refuses.  A merged JSON file must parse
    (`broken_json`)."""
    r = _git(tree, "-c", "merge.conflictStyle=merge", "merge", "--no-ff", "-m", msg, sha, check=False)
    if r.returncode:
        conflicted = [p for p in _git(tree, "diff", "--name-only", "-z", "--diff-filter=U").stdout.split("\0") if p]
        resolved = {}
        for p in conflicted:
            try:
                resolved[p] = keep_both_imports((tree / p).read_text())
            except (OSError, UnicodeDecodeError):
                resolved[p] = None
        if not conflicted or None in resolved.values():
            _git(tree, "merge", "--abort", check=False)
            other = [p for p, v in resolved.items() if v is None]
            raise Refused(f"{label} does not merge onto the train so far"
                          + (f" (a conflict in {', '.join(other)} holds more than import lines)" if other else "")
                          + f": {(r.stderr or r.stdout).strip()[-300:]}")
        for p, new in resolved.items():
            (tree / p).write_text(new)
        _git(tree, "add", "--", *resolved)
        _git(tree, "commit", "-q", "-m", f"{msg}\nImports: {', '.join(resolved)}")
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


#: the grants of `research queue`'s rules (role: what it attests) a train's pull request must hold at its pinned head before its
#: check is spent; the lander applies the rules' other grants by hand
TRAIN_GRANTS = {"statement-reviewer": "a named reviewer read what `tools/lean/audit.py --update` prints for its changed pins and "
                                      "the definitions they read (lane contract §5)"}


def ungranted(repo: Path, pins: list[tuple[str, str]], base: str, store, log=print) -> list[str]:
    """Why each train ref whose change from ``base`` needs a TRAIN_GRANTS grant under the queue's rules (as ``base`` has them;
    none when it has no rules file) can't go in: pull request N lacks the `grant` label on `pr:N@SHA` at its pinned head (and,
    for a lean-audit.json change, a `lean_review` that printed it whole as the same statements: `research.queue.reviewed`), or
    the ref isn't a pinned pull request, which is all a grant can be on."""
    from .queue import RULES, RULES_BEFORE_MOVE, QueueError, Rules, StoreGrants, grant_target

    for path in (RULES, RULES_BEFORE_MOVE):
        r = _git(repo, "show", f"{base}:{path}", check=False)
        if r.returncode == 0:
            break
    else:
        return []
    grants = StoreGrants(store, lambda s: log(s.replace("research queue:", "research merge:", 1)))
    out = []
    try:
        rules = Rules.parse(r.stdout, path)
        lean = {g.role for g in rules.grants if g.lean_audit}
        for label, sha in pins:
            pr = int(label[1:]) if label.startswith("#") else None
            need = sorted(rules.needs(repo, base, sha, (lambda: grants.reviews(pr, sha)) if pr else list) & set(TRAIN_GRANTS))
            if need and pr is None:
                out.append(f"{label} needs the {', '.join(need)} grant ({path}), which only a pull request's pinned head can hold: "
                           f"name it as N@SHA")
            elif missing := [g for g in need if g not in grants(pr, sha)]:
                target = grant_target(pr, sha)
                out += [f"{label} needs the {g} grant at {sha[:12]} ({path}: {TRAIN_GRANTS[g]}); once it has it, "
                        f"`research data label {target} grant {g} --by REVIEWER` (a new push needs a new one)" + (
                            f". A change `audit.py --update` (`--moved MOVES.json` for a move) prints only as the same statements "
                            f"needs none: `research data put --kind lean-review/v1 --file OUT/audit.json`, then `research data "
                            f"label {target} lean_review ART --by LANE`" if g in lean else "") for g in missing]
    except QueueError as e:
        raise Refused(f"the queue's rules ({path}): {e}") from e
    return out


#: what `prepare` runs on a train before pushing it: check's preflight (`uv.lock`, the repository's lints)
PREFLIGHT = ["tools/check/check.py", "--preflight"]


def preflight_argv() -> list[str]:
    """`prepare`'s default lint: the preflight under this interpreter when it has pytest, which the lints need and a system
    python3 lacks, else under `uv run --locked`, as check itself runs."""
    import importlib.util
    import shutil

    uv = shutil.which("uv")
    if importlib.util.find_spec("pytest") is not None or uv is None:
        return [sys.executable, *PREFLIGHT]
    return [uv, "run", "--locked", "python", *PREFLIGHT]


def prepare(repo: Path, refs: list[str], branch: str, onto: str, remote: str, lint: list[str], store) -> tuple[str, str]:
    """Build a train for its check without recording or landing anything: ``refs`` (`pinned`), refused while one lacks a grant
    (`ungranted`), merged in order onto ``onto`` as ``remote`` has it (the target, or the prep branch of a train ahead, to stack
    on it) in a scratch worktree, refused at the first that doesn't merge; ``lint`` run on the result there; its tip pushed to
    ``remote`` as ``branch``, never forced, and read back.  Every step names the scratch worktree or ``repo``'s object store,
    never its checkout.  Returns (the tip, its tree)."""
    r = _git(repo, "fetch", "-q", remote, onto, check=False)
    base = _git(repo, "rev-parse", "--verify", f"{'FETCH_HEAD' if r.returncode == 0 else onto}^{{commit}}").stdout.strip()
    pins = [pinned(repo, ref, remote) for ref in refs]
    if why := ungranted(repo, pins, base, store):
        raise Refused("; ".join(why) + "; nothing was built")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tree = Path(tempfile.mkdtemp(prefix="research-train-"))
    _git(repo, "worktree", "add", "-q", "--detach", str(tree), base)
    try:
        tip = _chain(tree, pins, stamp)[-1]
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
          remote: str = "origin") -> tuple[str | None, str | None]:
    """Merge ``refs`` in order onto ``into``'s tip as one candidate (a chain of `--no-ff` merges in a scratch worktree), record
    `tool` on it once and fast-forward ``into`` to it when it passes: the tree that lands is the one checked.  When it fails,
    bisect the chain's prefixes (each an exact commit, checked the same way) to the first ref whose merge makes it fail, and
    fast-forward to the longest prefix that passed.  ``tool`` is the tool's name, ``spec`` how `research run --tool` loads it
    (default the name).  Returns (the commit ``into`` now has, when it moved; the offending ref)."""
    head = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
    if head != into or _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise Refused(f"the checkout must be a clean {into}")
    pins = [pinned(repo, ref, remote) for ref in refs]
    if why := ungranted(repo, pins, "HEAD", store, log):
        raise Refused("; ".join(why) + "; nothing was checked")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tree = Path(tempfile.mkdtemp(prefix="research-train-"))
    branch = f"research-train/{stamp}"
    _git(repo, "worktree", "add", "-q", "-b", branch, str(tree), "HEAD")
    try:
        chain = _chain(tree, pins, stamp)
        need = required_steps(repo, "HEAD", chain[-1], tool)
        if need and on and not sends:
            raise Refused(f"the train changes {', '.join(sorted(set(p for ps in need.values() for p in ps)))}, where `{tool}` must run "
                          f"{', '.join(need)}: pass what it needs with --send (for `check`: `uv run python tools/check/check.py "
                          f"--agreement-files` prints the flags)")

        def passes(k: int) -> bool:
            _git(tree, "checkout", "-q", "--detach", chain[k - 1])
            _record(tree, spec or tool, command, on, project, log, sends)
            try:
                gate(repo, chain[k - 1], into, store, tool, log=lambda *_: None)
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
            gate(repo, land, into, store, tool, log)
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
    """Close every open pull request whose head is on ``remote``'s ``into`` (with a comment naming the commit it landed in) and
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
    verb = "would " if dry_run else ""
    for p in moved:
        log(f"research merge: {verb}retarget #{p['number']} from {p['baseRefName']} (on {remote}/{into}) onto {into}")
        if not dry_run:
            gh("pr", "edit", str(p["number"]), "--base", into)
    # a closed PR's head branch goes too, unless an open PR still uses it (as its head, or as a base the retargets left)
    used = {p["headRefName"] for p in rest} | {p["baseRefName"] for p in rest if p not in moved}
    for p in done:
        landed = landed_in(repo, p["headRefOid"], tip)
        subject = _git(repo, "log", "-1", "--format=%s", landed).stdout.strip()
        note = f"Merged into {into} at {landed[:12]} ({subject})."
        log(f"research merge: {verb}close #{p['number']} ({p['headRefName']}): {note}")
        if not dry_run:
            gh("pr", "close", str(p["number"]), "--comment", note)
        head = p["headRefName"]
        if head in used:
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


def _push_and_sweep(repo: Path, into: str, remote: str, push: bool) -> int:
    """After a merge landed on ``into`` here: with ``push``, push it to ``remote`` (never forced) and sweep, since the sweep closes
    only what is on the remote; a refused push sweeps nothing and fails. Without ``push`` the sweep finds nothing the merge
    landed, says so, and `research merge --sweep` after the push does it."""
    if push:
        r = _git(repo, "push", "-q", remote, f"refs/heads/{into}:refs/heads/{into}", check=False)
        if r.returncode:
            print(f"research merge: push of {into} to {remote} refused, nothing swept: {(r.stderr or r.stdout).strip()[-300:]}",
                  file=sys.stderr)
            return 1
        print(f"research merge: pushed {into} to {remote}")
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
    ap.add_argument("--into", default="main", help="the branch checked out here that receives the merge (default main)")
    ap.add_argument("--push", action="store_true", help="once the merge lands, push --into to --remote (never forced), then sweep; "
                    "without it the sweep runs before the push and can close nothing the merge landed")
    ap.add_argument("--tool", default="check", help="the tool whose passing attempt the merge requires (default check)")
    ap.add_argument("-m", "--message", default=None, help="the merge commit's message (default `Merge REF (sha)`)")
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
        if a.ref or a.train or a.sweep or a.guard_push:
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
    if a.sweep:
        if a.ref or a.train:
            ap.error("--sweep takes no refs")
        try:
            sweep(repo, a.into, a.remote, a.dry_run)
        except (Refused, ValueError, KeyError) as e:
            print(f"research merge: sweep failed: {e}", file=sys.stderr)
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
            tip, tree = prepare(repo, a.ref, a.prepare, onto, a.remote, lint, LocalStore(a.store))
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
                              remote=a.remote)
        except Refused as e:
            return refuse(PROG, str(e))
        moved = f"{a.into} fast-forwarded to {land[:12]}" if land else f"{a.into} unchanged"
        print(f"research merge: train {' '.join(a.ref)}: " + (f"`{decl.name}` passed; {moved}" if bad is None else f"{bad} breaks it; {moved}"))
        pushed = _push_and_sweep(repo, a.into, a.remote, a.push) if land else 0
        return OK if bad is None and not pushed else FINDING
    if len(a.ref) != 1:
        ap.error("several refs need --train")
    a.ref = a.ref[0]
    try:
        commit, attempt = gate(repo, a.ref, a.into, LocalStore(a.store), a.tool)
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
    return _push_and_sweep(repo, a.into, a.remote, a.push)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
