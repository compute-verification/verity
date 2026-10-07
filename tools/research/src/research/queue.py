"""`research queue`: the merge queue (infra plan change 5).  `next` is `main` plus every admitted PR, one merge commit each, in the
order they were admitted; `main` fast-forwards to the newest commit on `next` that `research merge`'s gate accepts: a passing
`check` attempt of that exact commit.

    research queue status [--json]                 each open PR: its place in `next`, what it waits for, or why it was refused
    research queue sync [--dry-run]                admit and drop PRs, merge them onto `next`, push it
    research queue land [--dry-run]                fast-forward `main` to the newest commit on `next` that passed `check`, push it
    research queue eject PR --reason TEXT          keep the PR's current head out of `next`; a new push re-admits it
    research queue ready PR (--on MACHINE [--priority] | --local) [--by LANE]
                                                   the owner's word that the PR is ready at its current head (a new push needs it again),
                                                   once the quick tier passes there: as a run on MACHINE in a check slot
                                                   (`RemoteQuickTier`; `auto` or M1,M2: the one with a free slot), or here
                                                   (`QuickTier`); exit 3 when it never started (no slot within its wait)
    research queue judge RUN [--on MACHINE] [--by LANE]
                                                   the mark `ready --on` writes, from its quick-tier run RUN that already ended
                                                   (its caller gone), for the open PR at the head the tier ran at, or at its tree
    common flags: [--shadow] [--workdir DIR] [--rules FILE] [--repo OWNER/NAME] [--remote URL]

Admission.  A PR is admitted while it is open against `main`, has an owner, carries no hold label, is marked ready at its head, and
has every grant its change needs.  Ready is a store label `ready = true` on `pr:{number}@{head sha}` that `research queue ready PR`
writes, so a new push needs a new one (a restack too: the tier ran on the old tree; not a head whose tree is a marked head's,
as a merge of `main` that changes no file leaves it: `ready` marks that one without a tier), and only after the quick tier passes at that
head: `suites.py --quick` on the suites the PR changes, from a clean checkout, either as a `research run` on a machine (`--on`,
`RemoteQuickTier`: in one of its check slots, so several PRs' tiers run at once and the caller only waits) or here (`--local`,
`QuickTier`); a failure refuses and names the tests, and a remote run whose result can't be read, or that reports another head, is a
failure.  Once the queue lands trains it replaces the
merge request (until then `main` moves only by the coordinator's trains, so ask on Slack too).  Draft state is ignored: agents open PRs as drafts, and
the ready label is what says one is done.  The hold label keeps a PR out whatever else holds.  The owner is the lane its body names
on a `Lane {name}` line, else the Cursor agent id its body carries.  The rules file (`ci/queue.toml` on
`main`, `tools/check/queue.toml` until the repository move, or `--rules`) says which grants a change needs, read from what merging
it into `main` would land (`main`'s tree against the one `git merge-tree` writes; a head that conflicts needs every grant): a role for changed files
under listed paths (`merge_requires`' syntax, `!` patterns included), and who gives each (`by`, which `status` prints with the label command to a PR waiting on it).  A grant is a
store label `grant = {role}` on the target `pr:{number}@{head sha}`.  It carries to a new head of the PR that only restacks it or
merges `main` into it: one whose change against `main` is the granted head's, every file from the same bytes to the same bytes, its
`lean-audit.json` change included (`carry`, Daniel's ruling of 2 Oct); the queue prints the head each grant carried from, or why one
didn't.  Any other push needs new grants, and so does a head whose history git can't compare (no merge base with `main`, or
several: no one diff against one says what it changes).  Grants and ready are read after
pulling the target's labels (for a carry, every head's) from the store's remote; when that pull can't happen (no remote here, one
that's unusable, a failed pull) the command fails with `STALE`, never answering from this machine's labels alone.

`next`.  Its first-parent chain above `main` is the queue, and each merge commit's trailers name its PR, head and owner: git holds
the queue's state.  A sync keeps the longest prefix of that chain which still matches the admitted PRs (same PR, same head, same
order) and merges the rest on top.  So a new push moves its PR to the tip, a dropped PR takes the commits after it along (they're
merged again), and a `main` that moved outside the queue rebuilds `next` on it.  A PR whose head contains another open PR's head
goes in after it and waits while that one waits.  A PR that doesn't merge cleanly is left out of that sync, and its owner is told
which files conflict and which queued PRs touched them.

Landing.  `land` walks `next` down from its tip and fast-forwards `main` to the first commit the gate accepts: `main` gets exactly
the tree that was checked, with no new merge commit.  A push refused because `main` moved leaves `main` alone; the next sync rebuilds
`next` on the new `main`.  A landing that changes what a guarantee says DMs Daniel what changed (`spec_alert`).

Notices.  A refusal (a conflict, an ejection) is sent once per PR head: a handoff in the owner's lane directory of the notes, or in
`lanes/coordinator/` for an owner known only by agent id.

`--shadow` builds `queue-shadow` instead of `next`, never pushes `main` and sends no notices, so the queue can run beside the trains
with no effect on them.

Files.  The queue works only in its own clone (`--workdir`, default ~/.research/queue/verity), and fetches with forced refspecs: a
local copy of a branch that wasn't force-fetched once put a train on a stale base.  `state.json` beside the clone keeps ejections and
the notices sent; losing it costs a retry.  One queue command runs at a time per workdir (`queue.lock`).  The clone authenticates as
the caller's checkout does (`credential_args`): a cloud VM sets its credential helper in the checkout's own config, which a clone
elsewhere never reads.  Git never prompts, so a remote the queue can't read fails the command instead of answering from what the
clone fetched last.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from . import quick_tier
from .gates import git_env
from .store import ids, vocab

BRANCH, SHADOW_BRANCH = "next", "queue-shadow"
RULES = "ci/queue.toml"
#: where the rules are until the repository move puts `tools/check/` at `ci/`
RULES_BEFORE_MOVE = "tools/check/queue.toml"
IDENTITY = {"GIT_AUTHOR_NAME": "research queue", "GIT_AUTHOR_EMAIL": "queue@research.invalid",
            "GIT_COMMITTER_NAME": "research queue", "GIT_COMMITTER_EMAIL": "queue@research.invalid"}
LANE_RE = re.compile(r"^[ \t>*_]*Lane[:\s]+`?([A-Za-z0-9][A-Za-z0-9._-]*)`?", re.M)
AGENT_RE = re.compile(r"\bbc-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
TRAILER_RE = re.compile(r"^(PR|Head|Owner): (.+)$", re.M)
SLUG_RE = re.compile(r"github\.com[:/]+([^/]+/[^/]+?)(?:\.git)?/?$")
AUTH_FAILED_RE = re.compile(r"could not read (Username|Password)|Authentication failed|unable to get password")


class QueueError(Exception):
    pass


def _git(repo: Path, *args: str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, capture_output=True, text=True,
                       errors="surrogateescape", env=git_env(GIT_TERMINAL_PROMPT="0", **(env or {})))
    if check and r.returncode:
        raise QueueError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()[-400:]}")
    return r


def _out(repo: Path, *args: str) -> str:
    return _git(repo, *args).stdout.strip()


def credential_args(repo: Path, remote: str) -> tuple[str, ...]:
    """`git -c` options that let the queue's own clone authenticate to `remote` as the checkout `repo` does: the `credential.*`
    settings its own config has for the remote (a cloud VM's helper lives there), else `gh auth git-credential` when no helper
    applies to the remote at all and `gh` is on PATH.  None when git's global config already serves it, or the remote isn't http."""
    if not re.match(r"https?://", remote):
        return ()
    found = _git(repo, "config", "--local", "--get-urlmatch", "credential", remote, check=False).stdout.splitlines()
    conf = dict(line.split(" ", 1) for line in found if " " in line)
    if not conf.get("credential.helper"):
        if _git(repo, "config", "--get-urlmatch", "credential.helper", remote, check=False).stdout.strip() or not (gh := shutil.which("gh")):
            return ()
        conf = {"credential.helper": f"!{shlex.quote(gh)} auth git-credential"}
    return ("-c", "credential.helper=", *(o for k, v in conf.items() for o in ("-c", f"{k}={v}")))


# ---- the open PRs -------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PR:
    number: int
    head: str
    title: str = ""
    body: str = ""
    draft: bool = False
    labels: tuple[str, ...] = ()
    base: str = "main"

    @classmethod
    def from_api(cls, d: dict) -> "PR":
        return cls(number=int(d["number"]), head=d["head"]["sha"], title=d.get("title") or "", body=d.get("body") or "",
                   draft=bool(d.get("draft")), labels=tuple(sorted(x["name"] for x in d.get("labels") or [])),
                   base=(d.get("base") or {}).get("ref", "main"))

    @property
    def owner(self) -> str | None:
        """The lane a `Lane {name}` line names, else the Cursor agent id the body carries (its PR footer), else None."""
        if m := LANE_RE.search(self.body):
            return m.group(1)
        if m := AGENT_RE.search(self.body):
            return m.group(0)
        return None


def _json_values(text: str) -> Iterable:
    """The JSON values `gh api --paginate` prints one after another (one array per page)."""
    dec, i = json.JSONDecoder(), 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        v, i = dec.raw_decode(text, i)
        yield v


class GitHub:
    """The open PRs of a repository, through `gh api` (the cloud VMs' authenticated GitHub CLI)."""

    def __init__(self, repo: str):
        self.repo = repo

    def open_prs(self) -> list[PR]:
        r = subprocess.run(["gh", "api", "--paginate", f"repos/{self.repo}/pulls?state=open&per_page=100"], capture_output=True, text=True)
        if r.returncode:
            raise QueueError(f"gh api repos/{self.repo}/pulls: {(r.stderr or r.stdout).strip()[-300:]}")
        return [PR.from_api(d) for page in _json_values(r.stdout) for d in (page if isinstance(page, list) else [page])]


# ---- rules and grants ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    role: str
    paths: tuple[str, ...] = ()        # `merge_requires`' syntax: a changed file under a prefix and matching no `!` pattern needs the role
    by: str = ""                       # who gives the grant, as `queue status` tells a waiting owner


@dataclass(frozen=True)
class Rules:
    grants: tuple[Rule, ...] = ()
    hold_label: str = "hold"
    source: str = RULES

    @classmethod
    def parse(cls, text: str, source: str = RULES) -> "Rules":
        doc = tomllib.loads(text)
        unknown = set(doc) - {"hold_label", "grant"}
        if unknown:
            raise QueueError(f"{source}: unknown keys {sorted(unknown)}")
        rules = []
        for i, g in enumerate(doc.get("grant", [])):
            if set(g) - {"role", "paths", "by"}:
                raise QueueError(f"{source}: grant {i}: unknown keys {sorted(set(g) - {'role', 'paths', 'by'})}")
            if (why := vocab.check(vocab.GRANT, g.get("role"))) is not None:
                raise QueueError(f"{source}: grant {i}: {why}")
            if not g.get("paths"):
                raise QueueError(f"{source}: grant {i} ({g['role']}) names no paths")
            rules.append(Rule(g["role"], tuple(g["paths"]), str(g.get("by", ""))))
        return cls(tuple(rules), doc.get("hold_label", "hold"), source)

    def asks(self, pr: int, head: str, roles: set[str]) -> str:
        """Who gives each of ``roles`` (its rule's `by`) and the label that grants it, for a PR waiting on them."""
        by = {r.role: r.by for r in self.grants}
        who = ", ".join(f"{r} from {by[r]}" if by.get(r) else f"{r} (no `by` in {self.source})" for r in sorted(roles))
        return f"{who}; granted with `research data label {grant_target(pr, head)} grant ROLE --by LANE` ({self.source})"

    def needs(self, repo: Path, base: str, head: str, log: Callable[[str], None] = lambda s: None) -> set[str]:
        """The roles whose grants merging `head` into `base` needs: judged on what would land, `base`'s tree against the tree `git
        merge-tree` writes for the merge, so `base`'s own changes since `head` forked never count, whatever merge bases they share
        (several, after a criss-cross, merge as a virtual base, as `git merge` does).  A head
        that doesn't merge cleanly needs every role, with a line saying why: no tree says what merging it changes, and a wait (the
        queue) or a refusal of that ref (a train) is how both callers hold a PR back."""
        from .merge import rule_touches

        r = _git(repo, "merge-tree", "--write-tree", "--no-messages", base, head, check=False)
        if r.returncode:
            why = "conflicts with it" if r.returncode == 1 else f"can't be merged into it ({(r.stderr or r.stdout).strip()[-200:]})"
            log(f"research queue: {head[:12]} {why}, so no tree says what merging it into {_named(base)} changes: it needs every "
                f"grant the rules name until it merges cleanly")
            return {r.role for r in self.grants}
        merged = r.stdout.split("\n", 1)[0]
        changed = [p for p in _git(repo, "diff", "--no-renames", "--name-only", "-z", "--ignore-submodules=none", base, merged).stdout.split("\0") if p]   # a rename's source too
        return {r.role for r in self.grants if rule_touches(list(r.paths), changed)}


def grant_target(pr: int, head: str) -> str:
    return f"pr:{pr}@{head}"


def _is_audit(path: str) -> bool:
    return path == "lean-audit.json" or path.endswith("/lean-audit.json")


def merge_bases(repo: Path, base: str, head: str) -> list[str]:
    """Every best common ancestor of ``base`` and ``head`` (`git merge-base --all`); none when they share no history or git can't
    say.  Several when ``head`` merged a commit that ``base`` merged too since (a criss-cross): `git merge` then diffs against a
    virtual base built from them all, so no one diff against one of them says what merging ``head`` changes."""
    r = _git(repo, "merge-base", "--all", base, head, check=False)
    return [] if r.returncode else r.stdout.split()


def _named(base: str) -> str:
    return base[:12] if re.fullmatch(r"[0-9a-f]{40}", base) else base


def _shown(path: str) -> str:
    """``path`` as text that prints: the bytes of a name that isn't UTF-8 (kept as surrogates by `_git`) as backslash escapes."""
    return path.encode("utf-8", "surrogateescape").decode("utf-8", "backslashreplace")


def own_change(repo: Path, base: str, head: str) -> dict[str, tuple[str, ...]] | None:
    """What ``head`` changes against its merge base with ``base``: {path: (old mode, new mode, old blob, new blob)}, a rename as a
    deletion and an addition, a submodule as the commit it pins (whatever `diff.ignoreSubmodules` or a `.gitmodules` says to
    ignore).  None when git can't say (``head`` isn't in ``repo``, or hasn't exactly one merge base with ``base``: `merge_bases`)."""
    mbs = merge_bases(repo, base, head)
    if len(mbs) != 1:
        return None
    r = _git(repo, "diff", "--raw", "--no-renames", "--no-abbrev", "--ignore-submodules=none", "-z", mbs[0], head, check=False)
    if r.returncode:
        return None
    fields = r.stdout.split("\0")
    return {path: tuple(meta.lstrip(":").split()[:4]) for meta, path in zip(fields[0::2], fields[1::2])}


ZERO_BLOB = "0" * 40


def rebased(repo: Path, was: tuple[str, ...], now: tuple[str, ...]) -> bool:
    """Whether one path's change ``now`` is the change ``was`` replayed onto another base (`own_change` entries: modes and blobs,
    before and after): ``was``'s edit, merged into ``now``'s base the way `git merge-file` merges three versions, applies cleanly
    and gives ``now``'s bytes exactly.  So the same hunks carry onto a base that changed elsewhere in the file, and hunks that land
    on other lines, or that the base changed under, don't.  The same modes are required, and a file added, deleted or pinned as a
    submodule (no three versions to merge) is ``was == now`` alone."""
    if was == now:
        return True
    if was[:2] != now[:2] or "160000" in was[:2] or ZERO_BLOB in (*was[2:], *now[2:]):
        return False
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for name, blob in (("ours", now[2]), ("base", was[2]), ("theirs", was[3])):
            r = subprocess.run(["git", "cat-file", "blob", blob], cwd=repo, capture_output=True, env=git_env())
            if r.returncode:
                return False
            (f := Path(tmp) / name).write_bytes(r.stdout)
            files.append(str(f))
        merged = subprocess.run(["git", "merge-file", "-p", "-q", *files], cwd=repo, capture_output=True, env=git_env())
        want = subprocess.run(["git", "cat-file", "blob", now[3]], cwd=repo, capture_output=True, env=git_env())
    return merged.returncode == 0 and want.returncode == 0 and merged.stdout == want.stdout


def carry(repo: Path, base: str, old: str, new: str) -> tuple[bool, str]:
    """Whether a grant on ``old``, another head of a PR, counts at its head ``new``, and why (Daniel's
    rulings: 2 Oct, one carries across a restack or a merge of `main` when the new head's `lean-audit.json` change is identical to
    the one granted; 6 Oct, 8:03 AM PDT, it carries when the PR's own patch is byte-identical at both heads, even if `main` moved
    under it, `lean-audit.json` included).

    The rule: against its merge base with ``base`` (the `main` the gate compares with now, so a parent that landed counts as
    `main`; for a PR stacked on another in a train, the ref below it there, so a parent that moved underneath it by adding
    commits or merging `main` doesn't count as its change), ``new`` changes exactly the paths ``old`` changes, and each the same way (`rebased`: its edit, replayed onto
    ``new``'s base by a three-way merge, gives ``new``'s bytes).  A `lean-audit.json` that `main` changed elsewhere meanwhile
    still carries; one where `main` changed the lines the PR edits doesn't, since replaying the edit then conflicts or gives other
    bytes.  That is what a rebase onto `main`, a restack
    after a parent landed, or a merge of `main` leaves, so it carries only to a head whose patch is the one that was granted,
    whatever its history: with one merge base, a merge commit can hold edits of its own and a rebase rewrites every commit, but
    neither can make two different patches compare equal.  Nothing else carries: a head that changes any file more, less or
    differently, or one git can't compare (``old`` isn't in ``repo``, no merge base, or several, where a merge can drop a change
    `main` merged without any diff against one of them showing it), needs new grants."""
    for h in (old, new):
        if _git(repo, "cat-file", "-e", f"{h}^{{commit}}", check=False).returncode:
            return False, f"{h[:12]} isn't in {repo}, so its change can't be compared (fetch it)"
    name = _named(base)
    for h in (old, new):
        if not (mbs := merge_bases(repo, base, h)):
            return False, f"{h[:12]} shares no history with {name}"
        if len(mbs) > 1:
            return False, (f"{h[:12]} has {len(mbs)} merge bases with {name} ({', '.join(m[:12] for m in mbs)}), so no one diff says "
                           f"what merging it changes; merging {name} into it again gives it one")
    was, now = own_change(repo, base, old), own_change(repo, base, new)
    if was is None or now is None:
        return False, f"git can't diff {(old if was is None else new)[:12]} against its merge base with {name}"
    if differ := sorted(p for p in was.keys() | now.keys() if p not in was or p not in now or not rebased(repo, was[p], now[p])):
        if audits := [p for p in differ if _is_audit(p)]:
            return False, f"its lean-audit.json patch isn't byte-identical ({', '.join(map(_shown, audits))})"
        more = f" and {len(differ) - 3} more" if len(differ) > 3 else ""
        return False, f"its change against {name} differs ({', '.join(map(_shown, differ[:3]))}{more})"
    def below(a: str, b: str) -> bool:
        return _git(repo, "merge-base", "--is-ancestor", a, b, check=False).returncode == 0

    if not below(old, new):
        how = "restack"
    elif below(base, new):
        how = "merge of main"
    elif (r := _git(repo, "rev-list", "--no-merges", "-n1", new, f"^{old}", f"^{base}", check=False)).returncode == 0 and \
            not r.stdout.strip():
        how = "merge of an ancestor of main"
    else:
        how = "commits on top of it that cancel out"
    return True, f"{how}; lean-audit.json change identical" if any(map(_is_audit, now)) else f"{how}; no lean-audit.json change"


class StoreGrants:
    """Grants from the evidence store: `grant` labels on `pr:{n}@{head}`, pulled from the remote first.  Without that pull it raises:
    a cloud VM's shell that lost its store secrets has no remote, and its local labels are whatever it last pulled.  Given the
    clone ``repo`` and the ``base`` its PRs merge onto, a grant on another head of the PR counts where `carry`
    says it does (`carried`), with a line naming the head it carried from."""

    def __init__(self, store, log: Callable[[str], None] = print, repo: Path | None = None, base: str = "origin/main"):
        self.store, self.log, self.repo, self.base = store, log, repo, base
        self._pulled: set[int] = set()

    def __call__(self, pr: int, head: str) -> set[str]:
        return {str(lab.value) for lab in self._labels(pr, head, vocab.GRANT)}

    def ready(self, pr: int, head: str) -> bool:
        """Whether someone asserted `ready = true` on `pr:{n}@{head}`."""
        return any(str(lab.value).lower() == "true" for lab in self._labels(pr, head, vocab.READY))

    def carried(self, pr: int, head: str, roles: set[str], onto: str | None = None) -> set[str]:
        """Which of ``roles`` a `grant` on another head of pull request ``pr`` gives ``head`` (`carry`), each head's change taken
        against ``onto`` (default the ``base`` it merges onto; for a PR stacked on another in a train, the ref below it there)."""
        return {str(lab.value) for _, lab in self._carried(pr, head, vocab.GRANT, roles, onto)} if roles else set()

    def _carried(self, pr: int, head: str, key: str, values: set[str] | None, onto: str | None = None) -> list[tuple]:
        """(the head it's on, the label) for each ``key`` label (whose value is in ``values``, unless None) on another head of pull
        request ``pr`` from which `carry` onto ``onto`` (default ``base``) says it counts at ``head``; a line for each head with such
        labels, carried or not."""
        if self.repo is None:
            return []
        out = []
        for old, labs in self._other_heads(pr, head, key).items():
            if not (labs := [lab for lab in labs if values is None or str(lab.value) in values]):
                continue
            what = ", ".join(sorted({f"{key} {lab.value}" for lab in labs}))
            ok, why = carry(self.repo, onto or self.base, old, head)
            if ok:
                self.log(f"research queue: {what} carried from {grant_target(pr, old)} to {head[:12]} ({why})")
                out += [(old, lab) for lab in labs]
            else:
                self.log(f"research queue: {what} on {grant_target(pr, old)} doesn't carry to {head[:12]}: {why}")
        return out

    def _other_heads(self, pr: int, head: str, key: str) -> dict[str, list]:
        """{head: its ``key`` labels} for each head of pull request ``pr`` but ``head`` that has some, after pulling every head's
        labels from the store's remote (once per PR); without that pull it raises, as `_labels` does."""
        from .store import labels_sync

        prefix = grant_target(pr, "")
        if pr not in self._pulled:
            try:
                labels_sync.pull_prefix(self.store, prefix)
            except Exception as e:  # noqa: BLE001  (no pull, no carry: local labels alone may be stale)
                raise QueueError(f"STALE: the labels of {prefix}* weren't pulled from the store's remote ({type(e).__name__}: {e}), "
                                 f"so grants on #{pr}'s other heads here may be old; not carrying any") from None
            self._pulled.add(pr)
        out = {}
        for d in sorted((Path(self.store.root) / "labels").glob(f"{ids.target_dirname(prefix)}*")):
            old = d.name.rpartition("@")[2]
            if old != head and ids.is_pr(grant_target(pr, old)) and (labs := self.store.labels(grant_target(pr, old), key=key)):
                out[old] = labs
        return out

    def _labels(self, pr: int, head: str, key: str) -> list:
        from .store import labels_sync
        from .store.local import StoreError

        target = grant_target(pr, head)
        try:
            labels_sync.read_through(self.store, [target], lambda s: self.log(f"research queue: {s}"), strict=True)
        except StoreError as e:
            raise QueueError(f"STALE: the labels of {target} weren't pulled from the store's remote ({e}), so ready marks and grants "
                             f"here may be old; not answering from them") from None
        return self.store.labels(target, key=key)


# ---- the queue ----------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Member:
    """One PR on `next`: its merge commit, and the head it merged."""
    pr: int
    head: str
    owner: str
    commit: str


@dataclass
class Entry:
    """One open PR's standing in a sync."""
    pr: PR
    state: str = "ready"                  # ready | queued | waiting | refused
    why: str = ""
    after: tuple[int, ...] = ()           # open PRs whose heads its head contains
    position: int | None = None           # 1-based, on `next`
    commit: str | None = None


def _message(pr: PR) -> str:
    return f"Queue: #{pr.number} {pr.title}".rstrip() + f"\n\nPR: #{pr.number}\nHead: {pr.head}\nOwner: {pr.owner}\n"


class Queue:
    def __init__(self, workdir: Path, *, remote: str, github, grants: Callable[[int, str], set[str]], ready: Callable[[int, str], bool],
                 store=None, shadow: bool = False, rules: Rules | None = None, notes: Path | None = None, log: Callable[[str], None] = print,
                 auth: tuple[str, ...] = (),
                 carry: Callable[[int, str, set[str]], set[str]] = lambda n, h, roles: set()):
        self.workdir, self.remote, self.github, self.grants, self.ready, self.store = Path(workdir), remote, github, grants, ready, store
        self.carry = carry
        self.shadow, self.log, self._rules, self.auth = shadow, log, rules, auth
        self.branch = SHADOW_BRANCH if shadow else BRANCH
        self.notes = Path(notes or os.environ.get("RESEARCH_NOTES") or Path.home() / ".research" / "notes")
        self.state_path = self.workdir.parent / f"{self.workdir.name}.state.json"

    # -- the clone and its state

    def _remote(self, cwd: Path, *args: str, ok: tuple[int, ...] | None = (0,)) -> subprocess.CompletedProcess:
        """A git command that talks to the remote, with `auth` (`credential_args`); one that exits outside `ok` raises."""
        r = _git(cwd, *self.auth, *args, check=False)
        if ok is not None and r.returncode not in ok:
            err = (r.stderr or r.stdout).strip()[-400:]
            hint = ("; no credential helper applies to the remote outside the caller's checkout: set one in git's global config, "
                    "or put an authenticated `gh` on PATH") if not self.auth and AUTH_FAILED_RE.search(err) else ""
            raise QueueError(f"git {' '.join(args)}: {err}{hint}")
        return r

    def _clone(self) -> None:
        if not (self.workdir / ".git").exists():
            self.workdir.parent.mkdir(parents=True, exist_ok=True)
            self._remote(self.workdir.parent, "clone", "-q", self.remote, str(self.workdir))
        # the caller's origin as it is now: a token embedded in the URL the clone was made from rotates
        _git(self.workdir, "remote", "set-url", "origin", self.remote)
        _git(self.workdir, "merge", "--abort", check=False)
        _git(self.workdir, "reset", "-q", "--hard")
        _git(self.workdir, "clean", "-qfdx")

    def fetch(self, prs: Iterable[PR] = ()) -> None:
        """Forced refspecs only: `main`, the queue's branch when it exists, and each open PR's head as `refs/queue/pr/{n}`."""
        specs = ["+refs/heads/main:refs/remotes/origin/main", *(f"+refs/pull/{p.number}/head:refs/queue/pr/{p.number}" for p in prs)]
        self._remote(self.workdir, "fetch", "-q", "--prune", "origin", *specs)
        # exit 2: no such branch; any other failure is the remote's, never an empty queue
        if self._remote(self.workdir, "ls-remote", "--exit-code", "origin", f"refs/heads/{self.branch}", ok=(0, 2)).returncode == 0:
            self._remote(self.workdir, "fetch", "-q", "origin", f"+refs/heads/{self.branch}:refs/remotes/origin/{self.branch}")
        else:
            _git(self.workdir, "update-ref", "-d", f"refs/remotes/origin/{self.branch}", check=False)

    def load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {"ejected": {}, "notified": [], "conflicts": {}}

    def save_state(self, state: dict) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
        tmp.replace(self.state_path)

    def rules(self) -> Rules:
        if self._rules is None:
            for path in (RULES, RULES_BEFORE_MOVE):
                r = _git(self.workdir, "show", f"origin/main:{path}", check=False)
                if r.returncode == 0:
                    self._rules = Rules.parse(r.stdout, path)
                    break
            else:
                raise QueueError(f"`main` has no {RULES}: pass --rules FILE")
        return self._rules

    def chain(self) -> list[Member]:
        """The members on the queue's branch above `main`, oldest first; none when the branch is missing or `main` isn't below it."""
        ref = f"refs/remotes/origin/{self.branch}"
        if _git(self.workdir, "rev-parse", "--verify", "-q", ref, check=False).returncode:
            return []
        if _git(self.workdir, "merge-base", "--is-ancestor", "origin/main", ref, check=False).returncode:
            return []
        out = []
        log = _out(self.workdir, "log", "--first-parent", "--reverse", "--format=%H%x00%B%x1e", f"origin/main..{ref}")
        for rec in filter(None, (x.strip() for x in log.split("\x1e"))):
            commit, body = rec.split("\x00", 1)
            t = dict(TRAILER_RE.findall(body))
            if not {"PR", "Head", "Owner"} <= set(t):
                return []                                   # a commit the queue didn't make: rebuild from `main`
            out.append(Member(int(t["PR"].lstrip("#")), t["Head"], t["Owner"], commit))
        return out

    # -- admission

    def entries(self, prs: list[PR], state: dict) -> dict[int, Entry]:
        rules = self.rules()
        heads = {p.head: p.number for p in prs}
        out: dict[int, Entry] = {}
        for p in prs:
            e = out[p.number] = Entry(p)
            ejected = state["ejected"].get(str(p.number))
            if rules.hold_label in p.labels:
                e.state, e.why = "waiting", f"held (`{rules.hold_label}` label)"
            elif p.owner is None:
                e.state, e.why = "waiting", "no owner: add a `Lane {name}` line to the PR body"
            elif ejected and ejected["head"] == p.head:
                e.state, e.why = "refused", f"ejected at {p.head[:8]}: {ejected['why']} (a new push re-admits it)"
            elif not self.ready(p.number, p.head):
                e.state, e.why = "waiting", f"not marked ready at {p.head[:8]}: `research queue ready {p.number} --on MACHINE` (or --local)"
            elif missing := self.missing(rules, p):
                e.state, e.why = "waiting", (f"waiting for the {', '.join(sorted(missing))} grant at {p.head[:8]}: "
                                             f"{rules.asks(p.number, p.head, missing)}")
            below = _out(self.workdir, "rev-list", f"origin/main..{p.head}").split()
            e.after = tuple(sorted(heads[h] for h in below if h in heads and heads[h] != p.number))
        changed = True
        while changed:                                      # a PR waits while a PR whose commits it contains isn't admitted
            changed = False
            for e in out.values():
                if e.state == "ready" and (blocking := [a for a in e.after if out[a].state != "ready"]):
                    e.state, e.why, changed = "waiting", f"contains #{blocking[0]}, which isn't admitted", True
        return out

    def missing(self, rules: Rules, p: PR) -> set[str]:
        """The roles ``p``'s change needs that no grant gives at its head, on it or carried to it from another (`carry`)."""
        need = rules.needs(self.workdir, "origin/main", p.head, log=self.log)
        missing = need - self.grants(p.number, p.head)
        return missing - self.carry(p.number, p.head, missing) if missing else missing

    def order(self, chain: list[Member], entries: dict[int, Entry]) -> list[int]:
        """Members that keep their head keep their order; the rest follow by PR number, each after the PRs it contains."""
        ready = {n for n, e in entries.items() if e.state == "ready"}
        kept = [m.pr for m in chain if m.pr in ready and entries[m.pr].pr.head == m.head]
        out: list[int] = []

        def place(n: int) -> None:
            if n in out:
                return
            for a in entries[n].after:
                place(a)
            out.append(n)

        for n in kept + sorted(ready - set(kept)):
            place(n)
        return out

    # -- building `next`

    def _merge(self, tip: str, pr: PR, members: list[Member]) -> tuple[str | None, str]:
        _git(self.workdir, "checkout", "-q", "--detach", tip)
        r = _git(self.workdir, "merge", "--no-ff", "--no-edit", "-m", _message(pr), pr.head, check=False, env=IDENTITY)
        if r.returncode == 0:
            return _out(self.workdir, "rev-parse", "HEAD"), ""
        files = _out(self.workdir, "diff", "--name-only", "--diff-filter=U").split()
        _git(self.workdir, "merge", "--abort", check=False)
        _git(self.workdir, "reset", "-q", "--hard", tip)
        if not files:
            return None, f"doesn't merge onto `{self.branch}`: {(r.stderr or r.stdout).strip()[-200:]}"
        by = [f"#{m.pr}" for m in members
              if set(files) & set(_out(self.workdir, "diff", "--name-only", f"{m.commit}^1", m.commit).split())]
        return None, f"conflicts with `{self.branch}` in {', '.join(files)}" + (f" (touched by {', '.join(by)})" if by else "")

    def sync(self, dry_run: bool = False) -> tuple[list[Member], dict[int, Entry]]:
        self._clone()
        state = self.load_state()
        prs = [p for p in self.github.open_prs() if p.base == "main"]
        self.fetch(prs)
        chain = self.chain()
        entries = self.entries(prs, state)
        order = self.order(chain, entries)
        want = [(n, entries[n].pr.head) for n in order]
        k = next((i for i, (m, w) in enumerate(zip(chain, want)) if (m.pr, m.head) != w), min(len(chain), len(want)))
        members = chain[:k]
        tip = members[-1].commit if members else _out(self.workdir, "rev-parse", "origin/main")
        dropped: set[int] = set()
        conflicts: dict[str, dict] = {}
        for n in order[k:]:
            e = entries[n]
            if blocked := [a for a in e.after if a in dropped]:
                e.state, e.why = "waiting", f"contains #{blocked[0]}, which didn't merge"
                dropped.add(n)
                continue
            commit, why = self._merge(tip, e.pr, members)
            if commit is None:
                e.state, e.why = "refused", why
                dropped.add(n)
                conflicts[str(n)] = {"head": e.pr.head, "why": why}
                self.notify(e, "conflict", why, state, dry_run)
                continue
            tip = commit
            members.append(Member(n, e.pr.head, e.pr.owner or "", commit))
        for i, m in enumerate(members, 1):
            entries[m.pr].state, entries[m.pr].position, entries[m.pr].commit = "queued", i, m.commit
        old = _git(self.workdir, "rev-parse", "--verify", "-q", f"refs/remotes/origin/{self.branch}", check=False).stdout.strip()
        if tip != old and not dry_run:
            self._remote(self.workdir, "push", "-q", f"--force-with-lease=refs/heads/{self.branch}:{old}", "origin",
                         f"{tip}:refs/heads/{self.branch}")
        self.log(f"research queue: `{self.branch}` at {tip[:12]}: {len(members)} PR(s) above `main`"
                 + (f", kept {k}, merged {len(members) - k}" if members else "") + (" (dry run: not pushed)" if dry_run else ""))
        if not dry_run:
            state["conflicts"] = conflicts
            self.save_state(state)
        return members, entries

    def eject(self, pr: int, reason: str, by: str) -> None:
        self._clone()
        prs = {p.number: p for p in self.github.open_prs()}
        if pr not in prs:
            raise QueueError(f"#{pr} isn't an open PR")
        state = self.load_state()
        state["ejected"][str(pr)] = {"head": prs[pr].head, "why": reason, "by": by, "at": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())}
        self.notify(Entry(prs[pr]), "eject", f"ejected from `{self.branch}`: {reason}", state, False)
        self.save_state(state)

    # -- landing

    def land(self, dry_run: bool = False) -> Member | None:
        """Fast-forward `main` to the newest commit on the queue's branch that the gate accepts, and push it (never with --shadow)."""
        from .merge import Refused, gate

        self._clone()
        self.fetch()
        chain = self.chain()
        if not chain:
            self.log(f"research queue: nothing on `{self.branch}` above `main`")
            return None
        _git(self.workdir, "checkout", "-q", "-B", "main", "refs/remotes/origin/main")
        store = self.store
        if store is not None and store.remote is not None:
            try:
                store.refresh(attempts_only=True)
            except Exception as e:  # noqa: BLE001  (the local catalog still answers)
                self.log(f"research queue: refreshing attempts failed ({type(e).__name__}: {e}); using the local catalog")
            store = _Refreshed(store)
        newest = None
        for m in reversed(chain):
            try:
                _, attempt = gate(self.workdir, m.commit, "main", store, "check", log=lambda *_: None)
            except Refused as e:
                newest = newest or f"#{m.pr} at {m.commit[:12]}: {e}"
                continue
            prs = " ".join(f"#{x.pr}" for x in chain[: chain.index(m) + 1])
            if dry_run or self.shadow:
                self.log(f"research queue: would land {prs} ({m.commit[:12]}, check {attempt})" + (" [shadow]" if self.shadow else ""))
                return m
            from .spec_alert import alert, remote_head

            was = (remote_head(self.workdir, "origin", "main", self.auth)
                   or _git(self.workdir, "rev-parse", "refs/heads/main").stdout.strip())
            _git(self.workdir, "merge", "-q", "--ff-only", m.commit)
            r = self._remote(self.workdir, "push", "-q", "origin", "main:refs/heads/main", ok=None)
            if r.returncode:
                _git(self.workdir, "reset", "-q", "--hard", "refs/remotes/origin/main")
                self.log(f"research queue: pushing `main` to {m.commit[:12]} was refused ({r.stderr.strip()[-200:]}); "
                         f"the next sync rebuilds `{self.branch}` on the new `main`")
                return None
            self.log(f"research queue: landed {prs} on `main` at {m.commit[:12]} (check {attempt})")
            alert(self.workdir, was, m.commit, prs=[x.pr for x in chain[: chain.index(m) + 1]], log=self.log)
            return m
        self.log(f"research queue: no commit on `{self.branch}` has a passing check yet (newest: {newest})")
        return None

    # -- notices

    def notify(self, e: Entry, kind: str, text: str, state: dict, dry_run: bool) -> None:
        """One handoff per PR head and kind, to the owner's lane (or `coordinator/` for an owner known only by agent id)."""
        key = f"{e.pr.number}@{e.pr.head}:{kind}"
        if key in state["notified"]:
            return
        owner = e.pr.owner or "unknown"
        lane = "coordinator" if AGENT_RE.fullmatch(owner) or owner == "unknown" else owner
        if self.shadow or dry_run:
            self.log(f"research queue: would tell {owner} (lanes/{lane}/): #{e.pr.number} {text}")
            return
        stamp = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
        d = self.notes / "lanes" / lane
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{stamp}-handoff-from-merge-queue-{e.pr.number}-{kind}.md"
        p.write_text(f"lane: {lane} · kind: handoff · from: merge-queue · to: {owner} · created: {stamp}\n\n"
                     f"# #{e.pr.number} {kind}: {text.split(':')[0]}\n\n- **PR:** #{e.pr.number} {e.pr.title}\n"
                     f"- **Head:** `{e.pr.head}`\n- **What happened:** {text}\n"
                     f"- **Next:** push a fix; a new head is admitted again at the tip of `{self.branch}`.\n")
        state["notified"].append(key)
        if os.environ.get("RESEARCH_NOTES_SYNC") == "1":
            subprocess.run([sys.executable, "-m", "research", "notes", "sync", "--path", f"lanes/{lane}"], capture_output=True)
        self.log(f"research queue: told {owner}: {p}")

    # -- status

    def status(self) -> dict[int, Entry]:
        self._clone()
        state = self.load_state()
        prs = [p for p in self.github.open_prs() if p.base == "main"]
        self.fetch(prs)
        entries = self.entries(prs, state)
        for i, m in enumerate(self.chain(), 1):
            if m.pr in entries and entries[m.pr].pr.head == m.head:
                entries[m.pr].state, entries[m.pr].position, entries[m.pr].commit = "queued", i, m.commit
        conflicts = state.get("conflicts", {})
        for n, e in entries.items():
            if e.state == "ready" and (c := conflicts.get(str(n))) and c["head"] == e.pr.head:
                e.state, e.why = "refused", f"{c['why']} (at the last sync)"
            elif e.state == "ready":
                e.why = "admitted: ready for a train (the queue does not land trains yet)"
        return entries


class _Refreshed:
    """A store whose attempts were just refreshed: the gate reads it without refreshing again for every commit it tries."""
    remote = None

    def __init__(self, store):
        self._store = store

    def __getattr__(self, name):
        return getattr(self._store, name)


# ---- command line -------------------------------------------------------------------------------------------------------------------

def _lock(workdir: Path):
    workdir.parent.mkdir(parents=True, exist_ok=True)
    f = open(workdir.parent / f"{workdir.name}.lock", "w")  # noqa: SIM115  (held until the process exits)
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise QueueError(f"another `research queue` holds {f.name}") from None
    return f


def _render(entries: dict[int, Entry]) -> str:
    rank = {"queued": 0, "ready": 1, "waiting": 2, "refused": 3}
    rows = sorted(entries.values(), key=lambda e: (rank[e.state], e.position or 0, e.pr.number))
    lines = []
    for e in rows:
        where = f"{e.position:>3}" if e.position else "  -"
        lines.append(f"{where}  #{e.pr.number:<5} {e.pr.head[:8]}  {e.state:<8} {(e.pr.owner or '-')[:24]:<24} {e.why}")
    return "\n".join(lines) or "no open PRs"


#: the quick tier, run from the checkout's root: the suites the PR changes since `--changed REV`, without the tests marked `slow`
SUITES = quick_tier.SUITES
QUICK_TIER = ["uv", "run", "--locked", "--extra", "torch-cpu", "python", SUITES, "--quick"]
#: the Lake packages whose builds the suites' tests run, built before `suites.py` as `check` builds them before those suites: each
#: test worker runs `lake build` there, and workers building one from cold at once fail (`lake build` exit 1, a binary missing)
LEAN_BUILDS = ("backends/flock/verifier/lean",)


class QuickTier:
    """`suites.py --quick --changed {merge base with main}` at a PR's head, from a clean checkout (`quick_tier.tier`, judged by
    `quick_tier.judge`): one worktree of the caller's repository per machine under `root`, checked out at the head and cleaned of
    everything but its `.venv` and the `.lake` of each of `LEAN_BUILDS` for each PR (one PR at a time: `{name}.lock`), so its
    environment syncs only what moved and Lake rebuilds only what changed.  Each of `LEAN_BUILDS` is built first (with the PATH's
    `lake`, else elan's; none, or a failed build, refuses).  Passes are cached by input (`suites.py`), so a re-run after a failure
    runs only what failed."""

    #: the worktree's name after the hash of the caller's git common dir: a remote tier ships its own (`-ship`), so a local tier,
    #: which holds its worktree's lock for the whole tier, never holds up a remote one's checkout from the same repository
    WORKTREE_SUFFIX = ""

    def __init__(self, repo: Path, remote: str, root: Path | None = None):
        self.repo, self.remote = Path(repo), remote
        self.root = Path(root or Path.home() / ".research" / "queue" / "ready")

    def __call__(self, pr: PR) -> str | None:
        """None when the tier passes at ``pr``'s head; else what failed, for the refusal."""
        with self._checkout(pr) as (wt, base):
            if base is None:
                return f"{pr.head[:12]} isn't on {self.remote} (pushed again since?)"
            out = self.root / "logs" / f"{pr.number}-{pr.head[:12]}"
            return quick_tier.judge(quick_tier.tier(wt, out, pr.head, base, QUICK_TIER, LEAN_BUILDS), pr.head)

    @contextlib.contextmanager
    def _checkout(self, pr: PR):
        """Under the worktree's lock: the worktree at ``pr``'s head, cleaned, and the head's merge base with `main` (None when the
        head isn't on the remote)."""
        common = Path(_out(self.repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
        wt = self.root / (hashlib.sha256(str(common).encode()).hexdigest()[:12] + self.WORKTREE_SUFFIX)
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / f"{wt.name}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _git(self.repo, "fetch", "-q", self.remote, "+refs/heads/main:refs/queue/ready/main",
                 f"+refs/pull/{pr.number}/head:refs/queue/ready/pr")
            if _git(self.repo, "cat-file", "-e", f"{pr.head}^{{commit}}", check=False).returncode:
                yield wt, None
                return
            _git(self.repo, "worktree", "prune")
            if not (wt / ".git").exists():
                shutil.rmtree(wt, ignore_errors=True)
                _git(self.repo, "worktree", "add", "-q", "--detach", "--force", str(wt), pr.head)
            else:
                _git(wt, "checkout", "-q", "--detach", "--force", pr.head)
            _git(wt, "clean", "-qffdx", "-e", "/.venv", *(x for d in LEAN_BUILDS for x in ("-e", f"/{d}/.lake")))
            yield wt, _out(self.repo, "merge-base", "refs/queue/ready/main", pr.head)


#: what a remote quick tier may take once it holds a check slot (`slot.py --run-max`), how long it may wait in the machine's line for
#: one first (`slot.py --wait-max`; the run's own timeout on the machine is the two together), and how much longer the caller waits
#: for its records to say it ended (custody's upload, a machine briefly unreachable) before it gives up on the run
REMOTE_TIMEOUT_S = 3 * 3600
REMOTE_WAIT_S = 6 * 3600
REMOTE_GRACE_S = 1800
REMOTE_POLL_S = 30
#: slot.py's exit when no check slot came within --wait-max (its NOT_STARTED): the tier never ran
SLOT_NOT_STARTED = 75
#: `research queue ready`'s exit when the tier never started: nothing failed, and nothing is marked
READY_NOT_STARTED = 3


class NotStarted(str):
    """Why a quick tier never ran (no check slot within its wait, no machine to run it on): not a failure of the PR's head."""


def place(spec: str, repo: Path, read: Callable[[list[str]], dict] | None = None) -> tuple[str | None, str]:
    """`--on auto` (check.py's AUTO) or `--on M1,M2`: the machine `check.py --on` would take for a lane check now (`choose` over
    each machine's `read_slots`, from the caller's checkout), with the Lean audit slots left out, since the quick tier takes none;
    None when no machine's slots could be read. And why, with each machine skipped."""
    s = importlib.util.spec_from_file_location("queue_check", repo / "tools" / "check" / "check.py")
    check = importlib.util.module_from_spec(s)
    sys.modules[s.name] = check
    s.loader.exec_module(check)
    names = list(check.AUTO) if spec == "auto" else [n for n in spec.split(",") if n]
    docs = {n: ({**d, "lean": {"pools": None}} if isinstance(d, dict) else d) for n, d in (read or check.read_slots)(names).items()}
    got, why, skipped = check.choose(names, docs)
    return got, why + "".join(f"; skipped {s}" for s in skipped)


def _launch(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True)


class RemoteQuickTier(QuickTier):
    """The quick tier as a `research run` on machine ``on`` (`auto` or `M1,M2`: the one `place` picks when the tier starts): a
    worktree of its own (`WORKTREE_SUFFIX`), checked out and cleaned as `QuickTier` does, is the run's `--source`, and the worktree's
    lock is held only while it is checked out and shipped, so several PRs' tiers run at once.  On the machine the run waits for a
    free check slot (`tools/check/slot.py` from the caller's checkout, sent with the run, as `check.py --record --on` takes one) and
    runs `quick_tier.py` (sent too) in the shipped tree.  The caller polls the run's records (`research fetch`) until it ends and
    judges the facts in its result.json as the local tier's are judged; a run that ends without them, with facts from another run
    or head, or other than `done` with exit 0, is a failure.  A run that waited `wait_s` for a slot and gave up never started
    (`NotStarted`), and neither did one with no machine to run on.  The tier has `timeout_s` from taking its slot (`--run-max`), so
    the run on the machine is bounded by the two together (`--timeout`), and the caller's wait by that plus REMOTE_GRACE_S of
    ``clock`` time (a fetch can take its ssh's own timeout, so polls are not counted), past which it ends with the poll under way."""

    WORKTREE_SUFFIX = "-ship"

    def __init__(self, repo: Path, remote: str, on: str, *, by: str = "", project: str = "verity", root: Path | None = None,
                 slot: Path | None = None, priority: bool = False, timeout_s: float = REMOTE_TIMEOUT_S, wait_s: float = REMOTE_WAIT_S,
                 poll_s: float = REMOTE_POLL_S, launch: Callable[[list[str], Path], subprocess.CompletedProcess] = _launch,
                 fetch: Callable[[Path], dict] | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, place: Callable[[str, Path], tuple[str | None, str]] = place):
        super().__init__(repo, remote, root)
        self.on, self.by, self.project, self.poll_s = on, by, project, poll_s
        self.timeout_s, self.wait_s = timeout_s, wait_s
        self.priority = priority
        self.slot = Path(slot or self.repo / "tools" / "check" / "slot.py")
        self.launch, self.fetch, self.sleep, self.clock, self.place = launch, fetch, sleep, clock, place

    def __call__(self, pr: PR) -> str | None:
        from . import runs as R

        on = self.on
        if not self.slot.is_file():
            return f"{self.slot} isn't there to take a check slot on {on} with (run it from a checkout of the repository)"
        if on == "auto" or "," in on:
            got, why = self.place(on, self.repo)
            if got is None:
                return NotStarted(f"--on {on}: {why}")
            print(f"research queue: --on {on}: {got}: {why}", flush=True)
            on = got
        job = Path(quick_tier.__file__)
        rid = R.new_run_id()
        with self._checkout(pr) as (wt, base):
            if base is None:
                return f"{pr.head[:12]} isn't on {self.remote} (pushed again since?)"
            args = [*(x for d in LEAN_BUILDS for x in ("--lean", d)), "--head", pr.head, "--base", base, "--", *QUICK_TIER]
            opts = ["--priority"] * self.priority + [*(["--by", self.by] if self.by else []), "--wait-max", f"{self.wait_s:g}",
                                                     "--run-max", f"{self.timeout_s:g}"]
            script = (f'i="$RESEARCH_RUN_DIR/inputs"; exec python3 "$i/{self.slot.name}" {shlex.join(opts)} '
                      f'-- python3 "$i/{job.name}" {shlex.join(args)}')
            r = self.launch([sys.executable, "-m", "research", "run", "--id", rid, "--on", on, "--project", self.project,
                             "--source", str(wt), "--cwd", "source", "--send", str(job), "--send", str(self.slot), "--require-result",
                             "--timeout", f"{self.wait_s + self.timeout_s:g}", "--custody-ttl", str(math.ceil(self.wait_s + self.timeout_s + REMOTE_GRACE_S)),
                             "--declared-output", "out/*", "--", "sh", "-c", script], wt)
        if r.returncode:
            return f"`research run --on {on}` failed (exit {r.returncode}): {(r.stderr or r.stdout or '').strip()[-400:]}"
        print(f"research queue: the quick tier runs as {rid} on {on}, in a check slot; waiting for it to end "
              f"(`research inspect {rid}`)", flush=True)
        d = R.runs_root() / rid
        status, why = self._wait(d, on)
        return why or judge_run(d, rid, on, pr.head, status)

    def _wait(self, d: Path, on: str) -> tuple[dict, str | None]:
        """The run's status.json once it says the run ended, from its records fetched every `poll_s` until the wait's deadline; or why
        the wait gave up."""
        from . import remote
        from . import runs as R

        fetch = self.fetch or remote.fetch
        status: dict = {}
        bound = self.wait_s + self.timeout_s + REMOTE_GRACE_S
        deadline = self.clock() + bound
        while self.clock() < deadline:
            self.sleep(self.poll_s)
            try:
                rec = fetch(d)
            except Exception as e:  # noqa: BLE001  (a fetch that fails is a poll that saw nothing)
                rec = {"last_fetch_error": {"stderr": f"{type(e).__name__}: {e}"}}
            status = _json_file(d / "status.json")
            if status.get("state") in R.TERMINAL:
                return status, None
            if gone := rec.get("launcher_exited"):
                return status, f"run {d.name}'s runner on {on} exited before the run ended ({gone.get('last_line') or 'no log line'})"
        return status, (f"run {d.name} on {on} didn't end within {bound:g} s (last seen "
                        f"{status.get('state') or 'without a status'}): `research inspect {d.name}`")


def tier_report(d: Path, rid: str, on: str, status: dict):
    """Run ``rid``'s result.json (its records in ``d``, its `status.json` ``status``) as a `quick_tier` report, and None; or None and
    why it isn't one."""
    from .result import check

    res = check(d, rid)
    if not res.present and status.get("rc") == SLOT_NOT_STARTED:
        return None, NotStarted(f"run {rid} on {on} waited for a check slot until its --wait-max and never ran the tier "
                                f"(`research inspect {rid}`)")
    if not res.present:
        return None, f"run {rid} on {on} ended {status.get('state')} (exit {status.get('rc')}) without a result.json (`research inspect {rid}`)"
    if not res.valid:
        return None, f"run {rid}'s result.json is unreadable: {'; '.join(res.errors)[:400]}"
    if (kind := res.doc.get("kind")) != quick_tier.KIND:
        return None, f"run {rid}'s result is a {kind or 'result with no kind'}, not a {quick_tier.KIND} report"
    return res, None


def judge_report(res, rid: str, on: str, head: str, status: dict) -> str | None:
    """None when the report ``res`` of run ``rid``, ended with ``status``, shows the tier passed at ``head``; else what failed."""
    if (why := quick_tier.judge(res.doc.get("tier"), head, f"its output is in run {rid}'s log on {on}")) is not None:
        return why
    if status.get("state") != "done" or status.get("rc") != 0 or res.validation_status != "passed":
        return (f"run {rid}'s facts show a pass, but it ended {status.get('state')} (exit {status.get('rc')}) with validation "
                f"{res.validation_status}")
    return None


def judge_run(d: Path, rid: str, on: str, head: str, status: dict) -> str | None:
    """None when ended run ``rid`` (records in ``d``) reports a pass of the quick tier at ``head``; else what failed: a run that ends
    without the facts, with facts from another run or head, or other than `done` with exit 0, is a failure."""
    res, why = tier_report(d, rid, on, status)
    return why or judge_report(res, rid, on, head, status)


def _json_file(p: Path) -> dict:
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def same_tree_mark(store, repo: Path, remote: str, pr: PR) -> str | None:
    """Another head of ``pr`` marked ready whose tree is ``pr.head``'s (a merge of `main` that changes no file), or None.  The tier
    at that head ran these exact files, and when a clean merge of `main` leaves the tree as it was, the files the new head changes
    against `main` are among those the marked head changed against its merge base, so the tier would run no suite it didn't.  None
    when either commit isn't here after fetching the PR's head (a rebase drops the old one), and QueueError when the PR's labels
    can't be pulled from the store's remote."""
    _fetch_heads(repo, remote, [pr])
    if not (want := _tree(repo, pr.head)):
        return None
    marked = StoreGrants(store, repo=repo)._other_heads(pr.number, pr.head, vocab.READY)
    return next((old for old, labs in sorted(marked.items()) if any(str(lab.value).lower() == "true" for lab in labs)
                 and _tree(repo, old) == want), None)


def same_tree_prs(repo: Path, remote: str, prs: list[PR], head: str) -> list[PR]:
    """The PRs of ``prs`` whose head has ``head``'s tree, after fetching their heads (none when ``head`` isn't here)."""
    _fetch_heads(repo, remote, prs)
    want = _tree(repo, head)
    return [p for p in prs if want and _tree(repo, p.head) == want] if prs else []


def _fetch_heads(repo: Path, remote: str, prs: list[PR]) -> None:
    """Each PR's head into a ref of its own, so a tier's fetch under the worktree's lock never meets this one."""
    if prs:
        _git(repo, "fetch", "-q", remote, *(f"+refs/pull/{p.number}/head:refs/queue/same-tree/{p.number}" for p in prs), check=False)


def _tree(repo: Path, commit: str) -> str:
    return _git(repo, "rev-parse", "-q", "--verify", f"{commit}^{{tree}}", check=False).stdout.strip()


def ready(github, store, number: int | None, by: str, tier: Callable[[PR], str | None]) -> int:
    """`research queue ready PR`: run the quick tier at the PR's current head (`tier`, a `RemoteQuickTier` or a `QuickTier`), and once
    it passes write `ready = true` on `pr:{n}@{that head}` and push it to the store's remote.  A head whose tree is that of another
    head marked ready (`same_tree_mark`) is marked without running the tier again."""
    pr = next((p for p in github.open_prs() if p.number == number), None) if number is not None else None
    if pr is None:
        print(f"research queue ready: {'#' + str(number) if number is not None else 'a PR number'} is not an open PR", file=sys.stderr)
        return 2
    if (repo := getattr(tier, "repo", None)) is not None:
        try:
            old = same_tree_mark(store, repo, tier.remote, pr)
        except QueueError as e:
            print(f"research queue: {e}; running the quick tier", file=sys.stderr)
            old = None
        if old:
            print(f"research queue: #{pr.number}'s head {pr.head[:12]} has the tree of {old[:12]}, marked ready after its quick tier "
                  f"passed: the same files, so the tier isn't run again", flush=True)
            return _mark(store, pr, by, None, "ready")
    print(f"research queue: the quick tier at #{pr.number}'s head {pr.head[:12]}: `suites.py --quick` on the suites it changes", flush=True)
    return _mark(store, pr, by, tier(pr), "ready")


def judge(github, store, ref: str, by: str, on: str | None = None, fetch: Callable[[Path], dict] | None = None,
          runs_root: Path | None = None, repo: Path | None = None, url: str | None = None) -> int:
    """`research queue judge RUN`: what `ready --on` does once its run ends, for a quick-tier run that already ended (its waiting
    client gone): fetch its records (`research fetch`; a run another VM launched is adopted from its machine, ``on`` or each), judge
    them as `RemoteQuickTier` does, at the head the tier ran at, and mark the open PR at that head ready, or, when none is there now
    and ``repo`` (a clone of ``url``) is given, the one open PR whose head has that head's tree (`same_tree_mark`'s reason).
    Refused when the run can't be fetched, hasn't ended, isn't a quick tier, failed, or no open PR is at that head or tree now."""
    from . import cli, remote
    from . import runs as R

    def refuse(why: str) -> int:
        print(f"research queue judge: {why}", file=sys.stderr)
        return 1

    try:
        d = cli._run_dir(R.runs_root(runs_root), ref, on)
        rec = (fetch or remote.fetch)(d)
    except (FileNotFoundError, LookupError, remote.Refused) as e:
        return refuse(str(e))
    rid, machine, status = d.name, rec.get("machine") or "its machine", _json_file(d / "status.json")
    if status.get("state") not in R.TERMINAL:
        err = (rec.get("last_fetch_error") or {}).get("stderr")
        return refuse(f"run {rid} hasn't ended (last seen {status.get('state') or 'without a status'}"
                      f"{'; its machine is unreachable: ' + err.strip()[-200:] if err else ''}): `research inspect {rid}`")
    res, why = tier_report(d, rid, machine, status)
    if why:
        return refuse(why)
    head = tier.get("head") if isinstance(tier := res.doc.get("tier"), dict) else None
    open_prs = github.open_prs()
    prs = [p for p in open_prs if p.head == head]
    if not prs and repo is not None and isinstance(head, str):
        if len(prs := same_tree_prs(repo, url, open_prs, head)) == 1:
            print(f"research queue: #{prs[0].number}'s head {prs[0].head[:12]} has the tree of {head[:12]}, where run {rid} ran the "
                  f"quick tier: the same files", flush=True)
    if len(prs) != 1:
        return refuse(f"run {rid} ran the quick tier at {str(head)[:12]}, " + (
            f"the head of each of {', '.join('#' + str(p.number) for p in prs)}: mark one with `research queue ready`" if prs else
            "which is no open PR's head now, nor its tree (pushed again since, or closed?)"))
    return _mark(store, prs[0], by, judge_report(res, rid, machine, head, status), "judge")


def _mark(store, pr: PR, by: str, failed: str | None, command: str) -> int:
    """`ready = true` on `pr:{n}@{head}`, pushed to the store's remote, unless the quick tier ``failed`` there (the refusal)."""
    from .store.cli import _label_write_through

    if isinstance(failed, NotStarted):
        print(f"research queue {command}: #{pr.number} is not marked ready at {pr.head[:12]}: the quick tier never started: {failed}. "
              f"Nothing failed: run it again (`--on auto` takes the machine with a free check slot)", file=sys.stderr)
        return READY_NOT_STARTED
    if failed is not None:
        print(f"research queue {command}: #{pr.number} is not marked ready at {pr.head[:12]}: the quick tier failed: {failed}. Push a "
              f"fix, or run it again: passes are cached by input, so a re-run on the same cache runs only what failed", file=sys.stderr)
        return 1
    target = grant_target(pr.number, pr.head)
    store.add_label(target, vocab.READY, True, by)
    said, lost = _label_write_through(store, target)
    print(f"research queue: #{pr.number} is ready at {pr.head[:12]} ({target}); {said}")
    return 1 if lost else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research queue", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["status", "sync", "land", "eject", "ready", "judge"])
    ap.add_argument("pr", nargs="?", help="eject, ready: the PR number; judge: the quick tier's run id")
    ap.add_argument("--reason", help="eject: why, for the owner's notice")
    ap.add_argument("--by", default=os.environ.get("RESEARCH_LANE"),
                    help="eject, ready, judge: who ejects it (default merge-queue) or says it's ready (no default: the owner's lane), "
                         "default $RESEARCH_LANE")
    ap.add_argument("--shadow", action="store_true", help=f"build `{SHADOW_BRANCH}`, never push `main`, send no notices")
    ap.add_argument("--dry-run", action="store_true", help="sync/land: decide and report, push nothing, tell no one")
    ap.add_argument("--workdir", type=Path, default=Path.home() / ".research" / "queue" / "verity", help="the queue's own clone")
    ap.add_argument("--remote", default=None, help="the repository to clone (default: this checkout's origin)")
    ap.add_argument("--repo", default=None, help="OWNER/NAME on GitHub (default: from the remote's URL)")
    ap.add_argument("--rules", type=Path, default=None, help=f"the admission rules (default: {RULES} on `main`)")
    ap.add_argument("--store", default=None, help="evidence store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--json", action="store_true", help="status: as JSON")
    ap.add_argument("--on", metavar="MACHINE", help="ready: run the quick tier as a `research run` there, in a check slot (`auto` or "
                                                    "M1,M2: the one with a free check slot, as `check.py --on` picks); judge: look "
                                                    "for a run another VM launched there only")
    ap.add_argument("--local", action="store_true", help="ready: run the quick tier here")
    ap.add_argument("--priority", action="store_true",
                    help="ready --on: wait for the check slot as a priority waiter (`slot.py --priority`), as train checks do; "
                         "only with the slot's owners' grant for that PR")
    a = ap.parse_args(argv)
    if a.dry_run and a.command not in ("sync", "land"):
        ap.error(f"--dry-run is for sync and land only: `{a.command}` would act for real")
    if a.command == "ready" and bool(a.on) == a.local:
        ap.error("ready takes one of --on MACHINE (the quick tier as a run there, in a check slot) and --local (here)")
    if a.priority and not (a.command == "ready" and a.on):
        ap.error("--priority is for ready --on MACHINE: the check slot's line is the machine's")
    if a.command == "judge" and not a.pr:
        ap.error("judge needs the quick tier's run id")
    if a.command in ("ready", "judge") and not a.by:
        ap.error(f"{a.command} writes the owner's ready mark: pass --by <lane> (or set $RESEARCH_LANE)")
    a.by = a.by or "merge-queue"
    if a.command != "judge" and a.pr is not None:
        if not a.pr.isdigit():
            ap.error(f"{a.command}: {a.pr!r} is not a PR number")
        a.pr = int(a.pr)
    from .store.local import LocalStore

    try:
        remote = a.remote or _out(Path.cwd(), "remote", "get-url", "origin")
        repo = a.repo or ((m := SLUG_RE.search(remote)) and m.group(1))
        if not repo:
            raise QueueError(f"can't tell OWNER/NAME from {remote!r}: pass --repo")
        store = LocalStore(a.store)
        if a.command == "judge":
            here = _git(Path.cwd(), "rev-parse", "--show-toplevel", check=False).stdout.strip()
            return judge(GitHub(repo), store, a.pr, a.by, on=a.on, repo=Path(here) if here else None, url=remote)
        if a.command == "ready":
            top = Path(_out(Path.cwd(), "rev-parse", "--show-toplevel"))
            tier = RemoteQuickTier(top, remote, a.on, by=a.by, priority=a.priority) if a.on else QuickTier(top, remote)
            return ready(GitHub(repo), store, a.pr, a.by, tier)
        labels = StoreGrants(store, log=lambda s: print(s, file=sys.stderr), repo=a.workdir)
        q = Queue(a.workdir, remote=remote, github=GitHub(repo), grants=labels, ready=labels.ready, store=store, shadow=a.shadow,
                  rules=Rules.parse(a.rules.read_text(), str(a.rules)) if a.rules else None,
                  auth=credential_args(Path.cwd(), remote), carry=labels.carried)
        lock = _lock(a.workdir)  # noqa: F841
        if a.command == "status":
            entries = q.status()
            if a.json:
                print(json.dumps([{"pr": e.pr.number, "head": e.pr.head, "owner": e.pr.owner, "draft": e.pr.draft, "state": e.state,
                                   "why": e.why, "position": e.position, "commit": e.commit, "after": list(e.after)}
                                  for e in sorted(entries.values(), key=lambda e: e.pr.number)], indent=1))
            else:
                print(_render(entries))
            return 0
        if a.command == "sync":
            q.sync(dry_run=a.dry_run)
            return 0
        if a.command == "land":
            return 0 if q.land(dry_run=a.dry_run) else 1
        if a.pr is None or not a.reason:
            ap.error("eject needs a PR number and --reason")
        q.eject(a.pr, a.reason, a.by)
        q.sync()
        return 0
    except QueueError as e:
        print(f"research queue: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
