"""`research queue`: the merge queue (infra plan change 5).  `next` is `main` plus every admitted PR, one merge commit each, in the
order they were admitted; `main` fast-forwards to the newest commit on `next` that `research merge`'s gate accepts: a passing
`check` attempt of that exact commit.

    research queue status [--json]                 each open PR: its place in `next`, what it waits for, or why it was refused
    research queue sync [--dry-run]                admit and drop PRs, merge them onto `next`, push it
    research queue land [--dry-run]                fast-forward `main` to the newest commit on `next` that passed `check`, push it
    research queue eject PR --reason TEXT [--by HANDLE]
                                                   keep the PR's current head out of `next`; a new push or ready mark re-admits it;
                                                   posted as HANDLE to @ci and the owner in the merge thread (exit 1 if it couldn't be)
    research queue ready PR (--on MACHINE | --local | --mark) [--by LANE]
                                                   the owner's word that the PR is ready at its current head (a new push needs it again),
                                                   once the quick tier passes there: as a run on MACHINE in a check slot
                                                   (`RemoteQuickTier`; `auto` or M1,M2: the one with a free slot), or here
                                                   (`QuickTier`); exit 3 when it never started (no slot within its wait); or, with
                                                   --mark, on the owner's word alone (`MarkOnly`)
    research queue drive --on MACHINE [--max-running N] [--train-size N] [--marked-after TS] [--first N,...] [--dry-run]
                                                   one tick of the landing automation (`research.drive`): land what passed, sync,
                                                   check `next` in trains with train.sh, eject what one PR broke, tell @ci the rest
    research queue judge RUN [--on MACHINE] [--by LANE]
                                                   the mark `ready --on` writes, from its quick-tier run RUN that already ended
                                                   (its caller gone), for the open PR at the head the tier ran at, or at its tree
    common flags: [--shadow] [--workdir DIR] [--rules FILE] [--repo OWNER/NAME] [--remote URL]

Admission.  A PR is admitted while it is open against `main`, has an owner, carries no hold label, and is marked ready at its head;
no change waits on a per-head grant (Daniel, 8 Oct).  Ready is a store label `ready = true` on `pr:{number}@{head sha}` that
`research queue ready PR` writes, so a new push needs a new one (a restack too: the tier ran on the old tree; not a head whose tree is a marked head's,
as a merge of `main` that changes no file leaves it: `ready` marks that one without a tier), and only after the quick tier passes at that
head: `suites.py --quick` on the suites the PR changes, from a clean checkout, either as a `research run` on a machine (`--on`,
`RemoteQuickTier`: in one of its check slots, so several PRs' tiers run at once and the caller only waits) or here (`--local`,
`QuickTier`); a failure refuses and names the tests, and a remote run whose result can't be read, or that reports another head, is a
failure.  `--mark` (`MarkOnly`) writes the mark with no tier, and the train's full check is the head's first test.  Once the queue lands trains it replaces the
merge request (until then `main` moves only by the coordinator's trains, so ask on Slack too).  Draft state is ignored: agents open PRs as drafts, and
the ready label is what says one is done.  The hold label keeps a PR out whatever else holds.  The owner is the lane its body names
on a `Lane {name}` line, else the Cursor agent id its body carries.  The rules file (`ci/queue.toml` on
`main`, `tools/verity/check/queue.toml` until the repository move, or `--rules`) names the hold label.  Ready marks are read after
pulling the target's labels from the store's remote; when that pull can't happen (no remote here, one
that's unusable, a failed pull) the command fails with `STALE`, never answering from this machine's labels alone.

`next`.  Its first-parent chain above `main` is the queue, and each merge commit's trailers name its PR, head and owner: git holds
the queue's state.  A sync keeps the longest prefix of that chain which still matches the admitted PRs (same PR, same head, same
order) and merges the rest on top.  So a new push moves its PR to the tip, a dropped PR takes the commits after it along (they're
merged again), and a `main` that moved outside the queue rebuilds `next` on it.  A PR whose head contains another open PR's head
goes in after it and waits while that one waits.  A PR that doesn't merge cleanly is left out of that sync, and its owner is told
which files conflict and which queued PRs touched them.

Landing.  `land` walks `next` down from its tip and fast-forwards `main` to the first commit the gate accepts: `main` gets exactly
the tree that was checked, with no new merge commit.  A push refused because `main` moved leaves `main` alone; the next sync rebuilds
`next` on the new `main`.  A landing that changes what a guarantee says DMs Daniel what changed (`spec_alert`).  `drive`
(`research.drive`) is the loop's tick that needs no agent: it launches the checks of `next`'s trains, lands what passed (onto a
moved `main` too, when the two share no file), ejects the PR a failure points at alone and tells `@ci` what it refuses.

Notices.  A refusal (a conflict, an ejection) is posted in the merge thread to `@ci` and the owner's handle, once per PR head:
drive tells each conflict its sync finds, and `eject` posts as its ejector.  Nothing is written to the notes: a handoff note
there reached its owner only when the VM that ran the sync pushed its notes, inside that lane's commit (13 Oct: @lean's
held notes carried conflict handoffs for ci's and circuits' PRs).

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
#: where the rules are until the repository move puts `tools/verity/check/` at `ci/`
RULES_BEFORE_MOVE = "tools/verity/check/queue.toml"
IDENTITY = {"GIT_AUTHOR_NAME": "research queue", "GIT_AUTHOR_EMAIL": "queue@research.invalid",
            "GIT_COMMITTER_NAME": "research queue", "GIT_COMMITTER_EMAIL": "queue@research.invalid"}
DATES = ("GIT_AUTHOR_DATE", "GIT_COMMITTER_DATE")
#: a transfer from the remote under 1 KB/s for 2 min aborts: git has no timeout, and a stalled HTTPS fetch held a drive tick
#: for 10 min (13 Oct 02:33Z), the next tick retrying it
STALL = ("-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=120")
LANE_RE = re.compile(r"^[ \t>*_]*Lane[:\s]+`?([A-Za-z0-9][A-Za-z0-9._-]*)`?", re.M)
REGISTRY = Path(".agents/skills/using-slack/registry.json")
AGENT_RE = re.compile(r"\bbc-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
TRAILER_RE = re.compile(r"^(PR|Head|Owner): (.+)$", re.M)
SLUG_RE = re.compile(r"github\.com[:/]+([^/]+/[^/]+?)(?:\.git)?/?$")
AUTH_FAILED_RE = re.compile(r"could not read (Username|Password)|Authentication failed|unable to get password")


class QueueError(Exception):
    pass


def _git(repo: Path, *args: str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    full = git_env(GIT_TERMINAL_PROMPT="0", **(env or {}))
    if env is IDENTITY:                     # the queue's merges carry the time they're made: a date a caller pins stops `log --since`
        for k in DATES:
            full.pop(k, None)
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, capture_output=True, text=True,
                       errors="surrogateescape", env=full)
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
    branch: str = ""

    @classmethod
    def from_api(cls, d: dict) -> "PR":
        return cls(number=int(d["number"]), head=d["head"]["sha"], title=d.get("title") or "", body=d.get("body") or "",
                   draft=bool(d.get("draft")), labels=tuple(sorted(x["name"] for x in d.get("labels") or [])),
                   base=(d.get("base") or {}).get("ref", "main"), branch=d["head"].get("ref") or "")

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


# ---- rules and labels ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Rules:
    hold_label: str = "hold"
    source: str = RULES
    #: ready marks older than this don't count (`--marked-after` overrides it), so every VM's queue admits what drive admits
    marked_after: str | None = None

    @classmethod
    def parse(cls, text: str, source: str = RULES) -> "Rules":
        doc = tomllib.loads(text)
        unknown = set(doc) - {"hold_label", "marked_after"}
        if unknown:
            raise QueueError(f"{source}: unknown keys {sorted(unknown)}")
        cut = doc.get("marked_after")
        if cut is not None and not (isinstance(cut, str) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", cut)):
            raise QueueError(f"{source}: marked_after {cut!r}: a UTC time as YYYY-MM-DDTHH:MM:SSZ, as labels record it")
        return cls(doc.get("hold_label", "hold"), source, cut)


def label_target(pr: int, head: str) -> str:
    return f"pr:{pr}@{head}"


class StoreLabels:
    """Ready marks from the evidence store: `ready` labels on `pr:{n}@{head}`, pulled from the remote first.  Without that pull it
    raises: a cloud VM's shell that lost its store secrets has no remote, and its local labels are whatever it last pulled."""

    def __init__(self, store, log: Callable[[str], None] = print, since: str | None = None):
        self.store, self.log, self.since = store, log, since
        self._pulled: set[int] = set()
        self._newest: dict[tuple[int, str], object] = {}
        self._fresh: set[str] = set()

    def prefetch(self, heads: Iterable[tuple[int, str]]) -> None:
        """Pull the labels of each `pr:{n}@{head}` from the store's remote at once, so `ready` answers each from them without a
        pull of its own (12 Oct: one listing per open PR, in turn, was 10 s of every drive tick).  A failed pull leaves each
        target to its own, which says why."""
        from .store import labels_sync

        targets = [label_target(n, h) for n, h in heads]
        if not targets or getattr(self.store, "remote", None) is None:
            return
        try:
            labels_sync.pull_targets(self.store, targets, jobs=labels_sync.SYNC_JOBS)
        except Exception:  # noqa: BLE001
            return
        self._fresh.update(targets)

    def ready(self, pr: int, head: str) -> bool:
        """Whether the newest `ready` label on `pr:{n}@{head}` is `true`, written at or after ``since`` (`--marked-after`) when
        given.  `research queue eject` writes `ready = false` there, so an ejection holds on every VM's queue until the owner
        marks it again or pushes a new head (12 Oct: #1824's ejection lived in the ejecting VM's state, and drive re-admitted it)."""
        labs = sorted(self._labels(pr, head, vocab.READY), key=lambda lab: lab.ts)
        self._newest[(pr, head)] = labs[-1] if labs else None
        return bool(labs) and str(labs[-1].value).lower() == "true" and (self.since is None or labs[-1].ts >= self.since)

    def marked_by(self, pr: int, head: str) -> str | None:
        """Who wrote the newest `ready` label `ready` read on `pr:{n}@{head}`."""
        lab = self._newest.get((pr, head))
        return None if lab is None else lab.by

    def marked_at(self, pr: int, head: str) -> str | None:
        """When the newest `ready` label `ready` read on `pr:{n}@{head}` was written."""
        lab = self._newest.get((pr, head))
        return None if lab is None else lab.ts

    def before_cutoff(self, pr: int, head: str) -> str | None:
        """When the newest `ready` label `ready` read on `pr:{n}@{head}` is `true` but older than ``since``, its time: a mark
        drive ignores, which `status` names so the owner re-marks or closes the PR (12 Oct: #1784, #1786, #1801)."""
        lab = self._newest.get((pr, head))
        if lab is None or self.since is None or str(lab.value).lower() != "true" or lab.ts >= self.since:
            return None
        return lab.ts

    def _other_heads(self, pr: int, head: str, key: str) -> dict[str, list]:
        """{head: its ``key`` labels} for each head of pull request ``pr`` but ``head`` that has some, after pulling every head's
        labels from the store's remote (once per PR); without that pull it raises, as `_labels` does."""
        from .store import labels_sync

        prefix = label_target(pr, "")
        if pr not in self._pulled:
            try:
                labels_sync.pull_prefix(self.store, prefix)
            except Exception as e:  # noqa: BLE001  (no pull: local labels alone may be stale)
                raise QueueError(f"STALE: the labels of {prefix}* weren't pulled from the store's remote ({type(e).__name__}: {e}), "
                                 f"so ready marks on #{pr}'s other heads here may be old") from None
            self._pulled.add(pr)
        out = {}
        for d in sorted((Path(self.store.root) / "labels").glob(f"{ids.target_dirname(prefix)}*")):
            old = d.name.rpartition("@")[2]
            if old != head and ids.is_pr(label_target(pr, old)) and (labs := self.store.labels(label_target(pr, old), key=key)):
                out[old] = labs
        return out

    def _labels(self, pr: int, head: str, key: str) -> list:
        from .store import labels_sync
        from .store.local import StoreError

        target = label_target(pr, head)
        if target in self._fresh:
            return self.store.labels(target, key=key)
        try:
            labels_sync.read_through(self.store, [target], lambda s: self.log(f"research queue: {s}"), strict=True)
        except StoreError as e:
            raise QueueError(f"STALE: the labels of {target} weren't pulled from the store's remote ({e}), so ready marks here may "
                             f"be old; not answering from them") from None
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
    def __init__(self, workdir: Path, *, remote: str, github, ready: Callable[[int, str], bool],
                 store=None, shadow: bool = False, rules: Rules | None = None, log: Callable[[str], None] = print,
                 auth: tuple[str, ...] = (), first: tuple[int, ...] = (),
                 before_cutoff: Callable[[int, str], str | None] | None = None,
                 marked_by: Callable[[int, str], str | None] | None = None,
                 marked_at: Callable[[int, str], str | None] | None = None,
                 prefetch: Callable[[list[tuple[int, str]]], None] | None = None):
        self.workdir, self.remote, self.github, self.ready, self.store = Path(workdir), remote, github, ready, store
        self.shadow, self.log, self._rules, self.auth, self.first = shadow, log, rules, auth, tuple(first)
        self.before_cutoff, self.marked_by, self.marked_at, self.prefetch = before_cutoff, marked_by, marked_at, prefetch
        #: how many members at the bottom of `next` a running check covers (`Drive`): `first` goes above them, so no reorder
        #: takes a running check's commit off `next`
        self.hold = 0
        self.branch = SHADOW_BRANCH if shadow else BRANCH
        self.state_path = self.workdir.parent / f"{self.workdir.name}.state.json"

    # -- the clone and its state

    def _remote(self, cwd: Path, *args: str, ok: tuple[int, ...] | None = (0,)) -> subprocess.CompletedProcess:
        """A git command that talks to the remote, with `auth` (`credential_args`); one that exits outside `ok` raises."""
        r = _git(cwd, *STALL, *self.auth, *args, check=False)
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
        # `.gitattributes` names this driver for every lean-audit.json; without it in the clone's config git merges the records
        # as text, and any two PRs that each add a record conflict (12 Oct: #1806 dropped from `next`)
        from .clone import RECORD_DRIVER

        for k, v in RECORD_DRIVER.items():
            _git(self.workdir, "config", k, v)
        _git(self.workdir, "merge", "--abort", check=False)
        _git(self.workdir, "reset", "-q", "--hard")
        _git(self.workdir, "clean", "-qfdx")

    def fetch(self, prs: Iterable[PR] = ()) -> dict[int, str]:
        """Forced refspecs only: `main`, the queue's branch when it exists, and each open PR's head as `refs/queue/pr/{n}`;
        return each fetched pull ref's commit, which can briefly differ from GitHub's API head."""
        prs = list(prs)
        specs = ["+refs/heads/main:refs/remotes/origin/main", *(f"+refs/pull/{p.number}/head:refs/queue/pr/{p.number}" for p in prs)]
        self._remote(self.workdir, "fetch", "-q", "--prune", "origin", *specs)
        # exit 2: no such branch; any other failure is the remote's, never an empty queue
        if self._remote(self.workdir, "ls-remote", "--exit-code", "origin", f"refs/heads/{self.branch}", ok=(0, 2)).returncode == 0:
            self._remote(self.workdir, "fetch", "-q", "origin", f"+refs/heads/{self.branch}:refs/remotes/origin/{self.branch}")
        else:
            _git(self.workdir, "update-ref", "-d", f"refs/remotes/origin/{self.branch}", check=False)
        return {p.number: _out(self.workdir, "rev-parse", f"refs/queue/pr/{p.number}") for p in prs}

    def fetch_prs(self, prs: list[PR]) -> tuple[list[PR], dict[int, Entry]]:
        """Fetch ``prs`` and separate any whose API head disagrees with its pull ref, so one GitHub propagation race cannot stop
        the queue.  Such a PR waits outside ``next`` until a later snapshot agrees rather than admitting an unverified old head."""
        fetched = self.fetch(prs)
        consistent, waiting = [], {}
        for p in prs:
            if fetched[p.number] == p.head:
                consistent.append(p)
                continue
            why = f"GitHub's API head {p.head[:12]} and pull ref {fetched[p.number][:12]} disagree; retrying next sync"
            waiting[p.number] = Entry(p, state="waiting", why=why)
            self.log(f"research queue: #{p.number}'s API head {p.head[:12]} and fetched pull ref "
                     f"{fetched[p.number][:12]} disagree; retrying next sync")
        return consistent, waiting

    def load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {"ejected": {}, "conflicts": {}}

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

    def _marked_again(self, p: PR, ejected: dict) -> bool:
        """Whether ``p``'s owner marked it ready after its ejection at this head: the store's newest mark decides on every VM, so
        this queue's own record of the ejection can't hold a PR that mark admits (12 Oct: #1736, re-marked at its ejected head,
        stayed off `next` on the ejecting VM's queue only)."""
        if self.marked_at is None or not self.ready(p.number, p.head):
            return False
        when, at = self.marked_at(p.number, p.head), ejected.get("at", "")
        if len(at) == len("2026-10-12T21:09Z"):
            at = at[:-1] + ":59Z"                       # recorded to the minute: a mark in that minute may predate the ejection
        return when is not None and when > at

    def entries(self, prs: list[PR], state: dict) -> dict[int, Entry]:
        rules = self.rules()
        heads = {p.head: p.number for p in prs}
        out: dict[int, Entry] = {}
        if self.prefetch is not None:
            self.prefetch([(p.number, p.head) for p in prs if rules.hold_label not in p.labels and p.owner is not None])
        for p in prs:
            e = out[p.number] = Entry(p)
            ejected = state["ejected"].get(str(p.number))
            if rules.hold_label in p.labels:
                e.state, e.why = "waiting", f"held (`{rules.hold_label}` label)"
            elif p.owner is None:
                e.state, e.why = "waiting", "no owner: add a `Lane {name}` line to the PR body"
            elif ejected and ejected["head"] == p.head and not self._marked_again(p, ejected):
                e.state, e.why = "refused", f"ejected at {p.head[:8]}: {ejected['why']} (a new push or ready mark re-admits it)"
            elif self.ready(p.number, p.head):
                pass
            elif self.before_cutoff and (when := self.before_cutoff(p.number, p.head)):
                e.state, e.why = "waiting", (f"ignored: marked before cutoff (its ready mark at {p.head[:8]} is from {when}, "
                                             f"before `marked_after`): re-mark it or close it")
            else:
                e.state, e.why = "waiting", f"not marked ready at {p.head[:8]}: `research queue ready {p.number} --on MACHINE` (or --local; --mark without the tier)"
            below = _out(self.workdir, "rev-list", f"origin/main..{p.head}").split()
            e.after = tuple(sorted(heads[h] for h in below if h in heads and heads[h] != p.number))
        changed = True
        while changed:                                      # a PR waits while a PR whose commits it contains isn't admitted
            changed = False
            for e in out.values():
                if e.state == "ready" and (blocking := [a for a in e.after if out[a].state != "ready"]):
                    e.state, e.why, changed = "waiting", f"contains #{blocking[0]}, which isn't admitted", True
        return out

    def order(self, chain: list[Member], entries: dict[int, Entry]) -> list[int]:
        """Members that keep their head keep their order; the rest follow by PR number, each after the PRs it contains.  The PRs
        in `first` (`--first`) that are ready go below all of them, in the order given, so the next train starts with them; but
        above the bottom `hold` members, which a running check covers."""
        ready = {n for n, e in entries.items() if e.state == "ready"}
        kept = [m.pr for m in chain if m.pr in ready and entries[m.pr].pr.head == m.head]
        held: list[int] = []
        for m in chain[:self.hold]:
            if m.pr not in kept:
                break
            held.append(m.pr)
        out: list[int] = []

        def place(n: int) -> None:
            if n in out:
                return
            for a in entries[n].after:
                place(a)
            out.append(n)

        for n in held + [f for f in self.first if f in ready] + kept + sorted(ready - set(kept)):
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

    def handle(self, pr: PR) -> str | None:
        """The Slack handle of the lane that owns ``pr``, from main's registry (`.agents/skills/using-slack/registry.json`): the
        lane its body names, the agent id it carries, the ready mark's author, else the agent whose id ends in its branch's
        suffix (`cursor/<name>-4292`); None when none of them is a handle."""
        r = _git(self.workdir, "show", f"origin/main:{REGISTRY}", check=False)
        try:
            reg = json.loads(r.stdout).get("handles", []) if r.returncode == 0 else []
        except (ValueError, AttributeError):
            return None
        names = {h.get("handle") for h in reg if isinstance(h, dict)}
        holders = {a: h["handle"] for h in reg if isinstance(h, dict) and "handle" in h
                   for a in AGENT_RE.findall(str(h.get("holder", "")))}
        by = self.marked_by(pr.number, pr.head) if self.marked_by else None
        suffix = pr.branch.rpartition("-")[2] if "-" in pr.branch else ""
        tail = sorted({n for a, n in holders.items() if len(suffix) == 4 and a.endswith(suffix)})
        for c in (pr.owner, holders.get(pr.owner or ""), by, tail[0] if len(tail) == 1 else None):
            if c in names:
                return c
        return None

    def sync(self, dry_run: bool = False) -> tuple[list[Member], dict[int, Entry]]:
        self._clone()
        state = self.load_state()
        prs = [p for p in self.github.open_prs() if p.base == "main"]
        prs, waiting = self.fetch_prs(prs)
        chain = self.chain()
        entries = self.entries(prs, state)
        entries.update(waiting)
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
                conflicts[str(n)] = {"head": e.pr.head, "why": why, "handle": self.handle(e.pr)}
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

    def eject(self, pr: int, reason: str, by: str) -> PR:
        self._clone()
        prs = {p.number: p for p in self.github.open_prs()}
        if pr not in prs:
            raise QueueError(f"#{pr} isn't an open PR")
        state = self.load_state()
        state["ejected"][str(pr)] = {"head": prs[pr].head, "why": reason, "by": by, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if self.store is not None and not self.shadow:
            from .store.cli import _label_write_through

            target = label_target(pr, prs[pr].head)
            self.store.add_label(target, vocab.READY, False, by, ref=reason)
            said, lost = _label_write_through(self.store, target)
            self.log(f"research queue: #{pr}'s ready mark at {prs[pr].head[:12]} is withdrawn ({target}); {said}")
            if lost:
                raise QueueError(f"#{pr}'s withdrawn mark didn't reach the store's remote, so other queues may still admit it: {said}")
        self.save_state(state)
        return prs[pr]

    def tell_ejected(self, p: PR, reason: str, by: str, post: Callable[..., bool] | None = None) -> int:
        """A hand ejection's notice, posted as ``by`` in the merge thread to `@ci` and the lane that owns ``p`` (only a handle's
        holder may post as it).  Drive's
        own ejections are told by `Driver.refuse`.  1, with the text to send, when the post failed."""
        from .drive import CI, _merge_thread, merge_thread_post

        owner = self.handle(p)
        to = tuple(dict.fromkeys((CI, *([f"@{owner}"] if owner else []))))
        text = (f"{' '.join(to)} queue: #{p.number} at {p.head[:8]} is ejected from `{self.branch}` by {by}: {reason}. "
                f"Fix it and re-mark it; the next sync rebuilds `{self.branch}` without it")
        if self.shadow:
            self.log(f"research queue: would tell {' '.join(to)}: {text} [shadow]")
            return 0
        if (post or merge_thread_post)(text, to, sender=by):
            self.log(f"research queue: told {' '.join(to)} as @{by}: #{p.number} ejected")
            return 0
        print(f"research queue: #{p.number} is ejected, but posting its notice as @{by} failed, so no one was told. Send it as "
              f"your own handle: research msg send --as HANDLE --thread {_merge_thread()} --to {' '.join(to)} --text {text!r}",
              file=sys.stderr)
        return 1

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

    # -- status

    def status(self) -> dict[int, Entry]:
        self._clone()
        state = self.load_state()
        prs = [p for p in self.github.open_prs() if p.base == "main"]
        prs, waiting = self.fetch_prs(prs)
        entries = self.entries(prs, state)
        entries.update(waiting)
        for i, m in enumerate(self.chain(), 1):
            if m.pr in entries and entries[m.pr].pr.head == m.head:
                entries[m.pr].state, entries[m.pr].position, entries[m.pr].commit = "queued", i, m.commit
        conflicts = state.get("conflicts", {})
        for n, e in entries.items():
            if e.state == "ready" and (c := conflicts.get(str(n))) and c["head"] == e.pr.head:
                e.state, e.why = "refused", f"{c['why']} (at the last sync)"
            elif e.state == "ready":
                e.why = f"admitted: the next sync merges it onto `{self.branch}`, and drive checks and lands it in a train"
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
LEAN_BUILDS = ("verity/core/service/flock",)


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
#: the remote tier's command, slot.py from the shipped tree as the launcher of a machine with /etc/vy/runs-via-slot accepts it
#: (pods/nebius/slot_only.py's `through_slot`, which the suites that read this module may not read)
SLOT = ("python3", "tools/verity/check/slot.py")
#: `research queue ready`'s exit when the tier never started: nothing failed, and nothing is marked
READY_NOT_STARTED = 3


class NotStarted(str):
    """Why a quick tier never ran (no check slot within its wait, no machine to run it on): not a failure of the PR's head."""


def place(spec: str, repo: Path, read: Callable[[list[str]], dict] | None = None) -> tuple[str | None, str]:
    """`--on auto` (check.py's AUTO) or `--on M1,M2`: the machine `check.py --on` would take for a lane check now (`choose` over
    each machine's `read_slots`, from the caller's checkout), with the Lean audit slots left out, since the quick tier takes none;
    None when no machine's slots could be read. And why, with each machine skipped."""
    from .store import tools_registry

    path = tools_registry.resolve("tools/verity/check/check.py", lambda p: (repo / p).is_file()) or "tools/verity/check/check.py"
    s = importlib.util.spec_from_file_location("queue_check", repo / path)
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
    free check slot (the shipped tree's `tools/verity/check/slot.py`, as `check.py --record --on` takes one: the run's command is
    slot.py's, which a machine with /etc/vy/runs-via-slot requires) and runs `quick_tier.py` (sent with the run) in that tree.
    The caller polls the run's records (`research fetch`) until it ends and judges the facts in its result.json as the local tier's are judged; a run that ends without them, with facts from another run
    or head, or other than `done` with exit 0, is a failure.  A run that waited `wait_s` for a slot and gave up never started
    (`NotStarted`), and neither did one with no machine to run on.  The tier has `timeout_s` from taking its slot (`--run-max`), so
    the run on the machine is bounded by the two together (`--timeout`), and the caller's wait by that plus REMOTE_GRACE_S of
    ``clock`` time (a fetch can take its ssh's own timeout, so polls are not counted), past which it ends with the poll under way."""

    WORKTREE_SUFFIX = "-ship"

    def __init__(self, repo: Path, remote: str, on: str, *, by: str = "", project: str = "verity", root: Path | None = None,
                 timeout_s: float = REMOTE_TIMEOUT_S, wait_s: float = REMOTE_WAIT_S,
                 poll_s: float = REMOTE_POLL_S, launch: Callable[[list[str], Path], subprocess.CompletedProcess] = _launch,
                 fetch: Callable[[Path], dict] | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, place: Callable[[str, Path], tuple[str | None, str]] = place):
        super().__init__(repo, remote, root)
        self.on, self.by, self.project, self.poll_s = on, by, project, poll_s
        self.timeout_s, self.wait_s = timeout_s, wait_s
        self.launch, self.fetch, self.sleep, self.clock, self.place = launch, fetch, sleep, clock, place

    def __call__(self, pr: PR) -> str | None:
        from . import runs as R

        on = self.on
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
            opts = [*(["--by", self.by] if self.by else []), "--wait-max", f"{self.wait_s:g}", "--run-max", f"{self.timeout_s:g}"]
            r = self.launch([sys.executable, "-m", "research", "run", "--id", rid, "--on", on, "--project", self.project,
                             "--source", str(wt), "--cwd", "source", "--send", str(job), "--require-result",
                             "--timeout", f"{self.wait_s + self.timeout_s:g}", "--custody-ttl", str(math.ceil(self.wait_s + self.timeout_s + REMOTE_GRACE_S)),
                             "--declared-output", "out/*", "--", *SLOT, *opts,
                             "--", "sh", "-c", f'exec python3 "$RESEARCH_RUN_DIR/inputs/{job.name}" {shlex.join(args)}'], wt)
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
            if rec.get("launcher_exited") or rec.get("runner_gone"):
                # a fetch copies status.json before it probes the runner, which can record its end and exit in between
                # (r20261010-134738-7625 passed and read as gone): fetch once more before calling the run lost
                try:
                    fetch(d)
                except Exception:  # noqa: BLE001
                    pass
                status = _json_file(d / "status.json")
                if status.get("state") in R.TERMINAL:
                    return status, None
            if gone := rec.get("launcher_exited"):
                return status, f"run {d.name}'s runner on {on} exited before the run ended ({gone.get('last_line') or 'no log line'})"
            if gone := rec.get("runner_gone"):
                return status, f"run {d.name} on {on} ended without recording it: {gone['why']}"
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
    marked = StoreLabels(store)._other_heads(pr.number, pr.head, vocab.READY)
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


class MarkOnly:
    """`ready --mark`: the owner's word alone, with no quick tier, so marking needs no machine and no wait; the train's full check
    tests the head.  A ready post with no mark behind it cost #1650 5.5 h (infra's review, 9 Oct)."""

    def __call__(self, pr: PR) -> None:
        return None


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
    if isinstance(tier, MarkOnly):
        print(f"research queue: #{pr.number} at {pr.head[:12]}: marked on {by}'s word, with no quick tier", flush=True)
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
    target = label_target(pr.number, pr.head)
    store.add_label(target, vocab.READY, True, by)
    said, lost = _label_write_through(store, target)
    print(f"research queue: #{pr.number} is ready at {pr.head[:12]} ({target}); {said}")
    return 1 if lost else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research queue", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["status", "sync", "land", "eject", "ready", "judge", "drive"])
    ap.add_argument("pr", nargs="?", help="eject, ready: the PR number; judge: the quick tier's run id")
    ap.add_argument("--reason", help="eject: why, for the owner's notice")
    ap.add_argument("--by", default=os.environ.get("RESEARCH_LANE"),
                    help="eject, ready, judge: who ejects it, the handle its notice is posted as (default merge-queue, which no "
                         "one holds, so the notice fails) or says it's ready (no default: the owner's lane), "
                         "default $RESEARCH_LANE")
    ap.add_argument("--shadow", action="store_true", help=f"build `{SHADOW_BRANCH}`, never push `main`, send no notices")
    ap.add_argument("--dry-run", action="store_true", help="sync/land/drive: decide and report, push and launch nothing, tell no one")
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
    ap.add_argument("--mark", action="store_true", help="ready: mark it with no quick tier (the train's full check tests it)")
    ap.add_argument("--max-running", type=int, default=2, help="drive: the most checks of `next` its launches keep running at once")
    ap.add_argument("--train-size", type=int, default=4, help="drive: PRs per checked train on `next`")
    ap.add_argument("--marked-after", metavar="YYYY-MM-DDTHH:MM:SSZ", help="status/sync/land/drive: admit only ready marks written at "
                                                                           "or after this time (default: `marked_after` in the rules file)")
    ap.add_argument("--first", metavar="N[,N...]", default="", help="sync/drive: put these ready PRs at the bottom of `next`, in this "
                                                                    "order, ahead of members that keep their place (a fix the next "
                                                                    "train needs)")
    a = ap.parse_args(argv)
    if a.first and not re.fullmatch(r"\d+(,\d+)*", a.first):
        ap.error(f"--first {a.first!r}: PR numbers separated by commas")
    if a.marked_after and not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", a.marked_after):
        ap.error(f"--marked-after {a.marked_after!r}: a UTC time as YYYY-MM-DDTHH:MM:SSZ, as labels record it")
    if a.dry_run and a.command not in ("sync", "land", "drive"):
        ap.error(f"--dry-run is for sync, land and drive only: `{a.command}` would act for real")
    if a.command == "ready" and bool(a.on) + a.local + a.mark != 1:
        ap.error("ready takes one of --on MACHINE (the quick tier as a run there, in a check slot), --local (here) and --mark (no tier)")
    if a.command == "judge" and not a.pr:
        ap.error("judge needs the quick tier's run id")
    if a.command == "drive" and not a.on:
        ap.error("drive needs --on MACHINE (train.sh's: a machine, or `auto`) for the checks it launches")
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
            tier = MarkOnly() if a.mark else RemoteQuickTier(top, remote, a.on, by=a.by) if a.on else QuickTier(top, remote)
            return ready(GitHub(repo), store, a.pr, a.by, tier)
        labels = StoreLabels(store, log=lambda s: print(s, file=sys.stderr), since=a.marked_after)
        q = Queue(a.workdir, remote=remote, github=GitHub(repo), ready=labels.ready, store=store, shadow=a.shadow,
                  rules=Rules.parse(a.rules.read_text(), str(a.rules)) if a.rules else None,
                  auth=credential_args(Path.cwd(), remote), first=tuple(int(n) for n in a.first.split(",") if n),
                  before_cutoff=labels.before_cutoff, marked_by=labels.marked_by, marked_at=labels.marked_at,
                  prefetch=labels.prefetch)
        lock = _lock(a.workdir)  # noqa: F841
        if not a.marked_after:
            if a.rules is None:
                q._clone()
                q.fetch()
            labels.since = q.rules().marked_after
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
        if a.command == "drive":
            from .drive import Driver

            Driver(q, on=a.on, max_running=a.max_running, train_size=a.train_size).tick(dry_run=a.dry_run)
            return 0
        if a.pr is None or not a.reason:
            ap.error("eject needs a PR number and --reason")
        told = q.tell_ejected(q.eject(a.pr, a.reason, a.by), a.reason, a.by)
        q.sync()
        return told
    except QueueError as e:
        print(f"research queue: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
