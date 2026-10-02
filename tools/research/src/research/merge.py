"""`research merge REF [--into BRANCH] [--tool check] [-m MESSAGE] [--push] [--dry-run]`: the merge gate.
`research merge --train A B C [--on MACHINE --project P] [--command CMD] [--push]`: several refs as one checked candidate (`train`).
`research merge --train --prepare BRANCH [--onto BASE] N@SHA ...`: build a train and push it as BRANCH for its check (`prepare`).
`research merge --sweep [--into BRANCH] [--remote origin] [--dry-run]`: close the pull requests that landed and delete their branches
(`sweep`).

Run in the checkout of the target branch.  It merges the commit REF names (`git merge --no-ff`) only if the evidence store holds
a passing attempt of the tool (default `check`) for that exact commit: `state` done, rc 0, source commit REF's, tree clean.  With
a usable remote the catalog's attempts are refreshed first, so an attempt recorded on another machine counts.  Refused as well: a
target checkout that is not on BRANCH or has tracked changes, and a REF the target has moved past (the target's tip is not an
ancestor of REF) unless merging REF leaves REF's own tree (`tree_identical`: what the target gained is in REF already, as when a
train checked on the train ahead of it lands after that one), since otherwise the merged tree is not the one that was checked;
merge the target into the branch and check the new tip.  The steps a change requires are read from the target's declaration and
the diff from its tip either way.  The merge commit carries a `Check: <attempt id>` trailer.  Exit 0 merged (or would, with
--dry-run), 1 refused, 2 usage.

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


class Refused(Exception):
    pass


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if check and r.returncode:
        raise Refused(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    return r


def passing(store, commit: str, tool: str = "check", *, tree: str | None = None, maps=None) -> list[str]:
    """Ids of the attempts of ``tool`` that passed on the clean tree of ``commit``, newest first.  After a history rewrite
    (`commitmap`), an attempt of an old commit counts for its image when the map says the rewrite kept the tree and, where the
    attempt recorded its tree, that tree is ``tree``."""
    rows = store.index.sql("SELECT id FROM attempts WHERE tool = ? AND source_commit = ? AND source_dirty = 0 AND state = 'done' "
                           "AND rc = 0 ORDER BY start_utc DESC", (tool, commit))
    ids = [r[0] for r in rows]
    for old in (maps.equivalent(commit)[1:] if maps else []):
        rows = store.index.sql("SELECT id, manifest_json FROM attempts WHERE tool = ? AND source_commit = ? AND source_dirty = 0 "
                               "AND state = 'done' AND rc = 0 ORDER BY start_utc DESC", (tool, old))
        for rid, doc in rows:
            recorded = (json.loads(doc).get("source") or {}).get("tree") if doc else None
            if tree is None or recorded in (None, tree):
                ids.append(rid)
    return ids


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
    ids = passing(store, commit, tool, tree=tree, maps=maps)
    if not ids:
        rows = store.index.sql("SELECT id, state, rc, source_dirty FROM attempts WHERE tool = ? AND source_commit = ?", (tool, commit))
        seen = "; ".join(f"{r[0]} {r[1]} rc={r[2]}{' dirty' if r[3] else ''}" for r in rows) or "none"
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
    for line in lines:
        f = line.split()
        if len(f) != 4 or f[2] != f"refs/heads/{into}" or f[1] == ZERO:
            continue
        tip = f[1]
        tree = _git(repo, "rev-parse", f"{tip}^{{tree}}").stdout.strip()
        parents = _git(repo, "rev-list", "--parents", "-n", "1", tip).stdout.split()[1:]
        if not any(_git(repo, "rev-parse", f"{x}^{{tree}}").stdout.strip() == tree and passing(store, x, tool, tree=tree)
                   for x in [tip, *parents[1:]]):
            bad.append(tip)
    return bad


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
    from .store import tools_registry

    module, _, attr = tools_registry.REGISTRY.get(tool, tool).rpartition(":")
    try:
        rules = tools_registry._from_file(module, attr, repo).merge_requires
    except Exception:  # noqa: BLE001  (a tool the checkout does not declare has no path rules)
        return {}
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


def _chain(tree: Path, pins: list[tuple[str, str]], stamp: str) -> list[str]:
    """Merge each (label, commit) in order onto the worktree ``tree``'s HEAD, one `--no-ff` merge each; the chain's commits, which
    are the train's prefixes.  Refused at the first that doesn't merge."""
    chain = []
    for label, sha in pins:
        r = _git(tree, "merge", "--no-ff", "-m", f"Merge {label} ({sha[:8]})\n\nTrain: {stamp}", sha, check=False)
        if r.returncode:
            _git(tree, "merge", "--abort", check=False)
            raise Refused(f"{label} does not merge onto the train so far: {(r.stderr or r.stdout).strip()[-300:]}")
        chain.append(_git(tree, "rev-parse", "HEAD").stdout.strip())
    return chain


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


def prepare(repo: Path, refs: list[str], branch: str, onto: str, remote: str, lint: list[str]) -> tuple[str, str]:
    """Build a train for its check without recording or landing anything: ``refs`` (`pinned`) merged in order onto ``onto`` as
    ``remote`` has it (the target, or the prep branch of a train ahead, to stack on it) in a scratch worktree, refused at the first
    that doesn't merge; ``lint`` run on the result there; its tip pushed to ``remote`` as ``branch``, never forced, and read back.
    Every step names the scratch worktree or ``repo``'s object store, never its checkout.  Returns (the tip, its tree)."""
    r = _git(repo, "fetch", "-q", remote, onto, check=False)
    base = _git(repo, "rev-parse", "--verify", f"{'FETCH_HEAD' if r.returncode == 0 else onto}^{{commit}}").stdout.strip()
    pins = [pinned(repo, ref, remote) for ref in refs]
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
    ap = argparse.ArgumentParser(prog="research merge", description=__doc__.split("\n\n")[1])
    ap.add_argument("ref", nargs="*", help="the commit to merge (a branch, tag or sha); with --train, several, in merge order")
    ap.add_argument("--sweep", action="store_true", help="merge nothing: close the open pull requests on the remote's --into and "
                    "retarget the ones based on a branch that is (with --dry-run: say what it would do)")
    ap.add_argument("--remote", default="origin", help="the remote whose --into branch the sweep reads, that pinned train refs are "
                    "fetched from and --prepare pushes to (default origin)")
    ap.add_argument("--train", action="store_true", help="merge the refs as one candidate, record the tool on it, fast-forward on "
                    "a pass, bisect to the offending ref on a failure")
    ap.add_argument("--on", default=None, metavar="MACHINE", help="--train: record the tool on this machine (research run --on)")
    ap.add_argument("--project", default=None, help="--train: the project the run belongs to (required with --on)")
    ap.add_argument("--command", default=None, help="--train: the workload command (default the tool declaration's)")
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
    a = ap.parse_args(argv)
    from .store.local import LocalStore

    repo = Path.cwd()
    if a.guard_push:
        bad = unchecked_pushes(repo, sys.stdin.read().splitlines(), LocalStore(a.store), a.into, a.tool)
        for c in bad:
            print(f"research merge: refused push: {c[:12]} would reach {a.into} with no passing `{a.tool}` attempt of its tree; "
                  "land it with `research merge` (the hook fails closed: `git push --no-verify` only with root's yes)", file=sys.stderr)
        return 1 if bad else 0
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
            tip, tree = prepare(repo, a.ref, a.prepare, onto, a.remote, lint)
        except Refused as e:
            print(f"research merge: refused: {e}", file=sys.stderr)
            return 1
        print(f"research merge: train {' '.join(a.ref)} on {onto}: pushed {a.remote}/{a.prepare} = {tip[:12]} (tree {tree[:12]}); "
              f"check it with `tools/check/train.sh {tip[:12]} MACHINE`, land it with `research merge {a.prepare}`")
        return 0
    if a.train:
        from .store import tools_registry

        decl = tools_registry.load(a.tool, repo)
        command = shlex.split(a.command) if a.command else list(decl.command)
        try:
            land, bad = train(repo, a.ref, a.into, LocalStore(a.store), decl.name, command, a.on, a.project, spec=a.tool, sends=a.send,
                              remote=a.remote)
        except Refused as e:
            print(f"research merge: refused: {e}", file=sys.stderr)
            return 1
        moved = f"{a.into} fast-forwarded to {land[:12]}" if land else f"{a.into} unchanged"
        print(f"research merge: train {' '.join(a.ref)}: " + (f"`{decl.name}` passed; {moved}" if bad is None else f"{bad} breaks it; {moved}"))
        pushed = _push_and_sweep(repo, a.into, a.remote, a.push) if land else 0
        return 0 if bad is None and not pushed else 1
    if len(a.ref) != 1:
        ap.error("several refs need --train")
    a.ref = a.ref[0]
    try:
        commit, attempt = gate(repo, a.ref, a.into, LocalStore(a.store), a.tool)
    except Refused as e:
        print(f"research merge: refused: {e}", file=sys.stderr)
        return 1
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
