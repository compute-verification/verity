"""`research queue`: the merge queue (infra plan change 5).  `next` is `main` plus every admitted PR, one merge commit each, in the
order they were admitted; `main` fast-forwards to the newest commit on `next` that `research merge`'s gate accepts: a passing
`check` attempt of that exact commit.

    research queue status [--json]                 each open PR: its place in `next`, what it waits for, or why it was refused
    research queue sync [--dry-run]                admit and drop PRs, merge them onto `next`, push it
    research queue land [--dry-run]                fast-forward `main` to the newest commit on `next` that passed `check`, push it
    research queue eject PR --reason TEXT          keep the PR's current head out of `next`; a new push re-admits it
    research queue ready PR [--by LANE]            the owner's word that the PR is ready at its current head (a new push needs it again)
    common flags: [--shadow] [--workdir DIR] [--rules FILE] [--repo OWNER/NAME] [--remote URL]

Admission.  A PR is admitted while it is open against `main`, has an owner, carries no hold label, is marked ready at its head, and
has every grant its change needs.  Ready is a store label `ready = true` on `pr:{number}@{head sha}` that `research queue ready PR`
writes, so a new push needs a new one; it replaces the merge-request note.  Draft state is ignored: agents open PRs as drafts, and
the ready label is what says one is done.  The hold label keeps a PR out whatever else holds.  The owner is the lane its body names
on a `Lane {name}` line, else the Cursor agent id its body carries.  The rules file (`ci/queue.toml` on
`main`, `tools/check/queue.toml` until the repository move, or `--rules`) says which grants a change needs: a role for changed files
under listed paths (`merge_requires`' syntax, `!` patterns included), or for a change to listed sections of any `lean-audit.json`
(`pins`, `reads`).  A grant is a store label `grant = {role}` on the target `pr:{number}@{head sha}`, so a new push needs new
grants.  Grants and ready are read after pulling the target's labels from the store's remote; when that pull fails the local
labels decide, so a grant only the remote has keeps the PR waiting.

`next`.  Its first-parent chain above `main` is the queue, and each merge commit's trailers name its PR, head and owner: git holds
the queue's state.  A sync keeps the longest prefix of that chain which still matches the admitted PRs (same PR, same head, same
order) and merges the rest on top.  So a new push moves its PR to the tip, a dropped PR takes the commits after it along (they're
merged again), and a `main` that moved outside the queue rebuilds `next` on it.  A PR whose head contains another open PR's head
goes in after it and waits while that one waits.  A PR that doesn't merge cleanly is left out of that sync, and its owner is told
which files conflict and which queued PRs touched them.

Landing.  `land` walks `next` down from its tip and fast-forwards `main` to the first commit the gate accepts: `main` gets exactly
the tree that was checked, with no new merge commit.  A push refused because `main` moved leaves `main` alone; the next sync rebuilds
`next` on the new `main`.

Notices.  A refusal (a conflict, an ejection) is sent once per PR head: a handoff in the owner's lane directory of the notes, or in
`lanes/coordinator/` for an owner known only by agent id.

`--shadow` builds `queue-shadow` instead of `next`, never pushes `main` and sends no notices, so the queue can run beside the trains
with no effect on them.

Files.  The queue works only in its own clone (`--workdir`, default ~/.research/queue/verity), and fetches with forced refspecs: a
local copy of a branch that wasn't force-fetched once put a train on a stale base.  `state.json` beside the clone keeps ejections and
the notices sent; losing it costs a retry.  One queue command runs at a time per workdir (`queue.lock`).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

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


class QueueError(Exception):
    pass


def _git(repo: Path, *args: str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    if check and r.returncode:
        raise QueueError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()[-400:]}")
    return r


def _out(repo: Path, *args: str) -> str:
    return _git(repo, *args).stdout.strip()


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
    lean_audit: tuple[str, ...] = ()   # a change to one of these top-level sections of any `lean-audit.json` needs the role


@dataclass(frozen=True)
class Rules:
    grants: tuple[Rule, ...] = ()
    hold_label: str = "hold"

    @classmethod
    def parse(cls, text: str, source: str = RULES) -> "Rules":
        doc = tomllib.loads(text)
        unknown = set(doc) - {"hold_label", "grant"}
        if unknown:
            raise QueueError(f"{source}: unknown keys {sorted(unknown)}")
        rules = []
        for i, g in enumerate(doc.get("grant", [])):
            if set(g) - {"role", "paths", "lean_audit"}:
                raise QueueError(f"{source}: grant {i}: unknown keys {sorted(set(g) - {'role', 'paths', 'lean_audit'})}")
            if (why := vocab.check(vocab.GRANT, g.get("role"))) is not None:
                raise QueueError(f"{source}: grant {i}: {why}")
            if not (g.get("paths") or g.get("lean_audit")):
                raise QueueError(f"{source}: grant {i} ({g['role']}) names neither paths nor lean_audit sections")
            rules.append(Rule(g["role"], tuple(g.get("paths", ())), tuple(g.get("lean_audit", ()))))
        return cls(tuple(rules), doc.get("hold_label", "hold"))

    def needs(self, repo: Path, base: str, head: str) -> set[str]:
        """The roles whose grants the change `base`...`head` (from their merge base) needs."""
        from .merge import rule_touches

        mb = _out(repo, "merge-base", base, head)
        changed = [p for p in _out(repo, "diff", "--name-only", mb, head).splitlines() if p]
        roles = {r.role for r in self.grants if r.paths and rule_touches(list(r.paths), changed)}
        audits = [p for p in changed if p == "lean-audit.json" or p.endswith("/lean-audit.json")]
        for r in (r for r in self.grants if r.lean_audit and r.role not in roles):
            if any(_section(repo, mb, p, s) != _section(repo, head, p, s) for p in audits for s in r.lean_audit):
                roles.add(r.role)
        return roles


def _section(repo: Path, rev: str, path: str, section: str):
    r = _git(repo, "show", f"{rev}:{path}", check=False)
    if r.returncode:
        return None
    try:
        return json.loads(r.stdout).get(section)
    except (ValueError, AttributeError):
        return r.stdout                                  # an unparseable file counts as changed whenever its bytes do


def grant_target(pr: int, head: str) -> str:
    return f"pr:{pr}@{head}"


class StoreGrants:
    """Grants from the evidence store: `grant` labels on `pr:{n}@{head}`, pulled from the remote first when there is one."""

    def __init__(self, store, log: Callable[[str], None] = print):
        self.store, self.log = store, log

    def __call__(self, pr: int, head: str) -> set[str]:
        return {str(lab.value) for lab in self._labels(pr, head, vocab.GRANT)}

    def ready(self, pr: int, head: str) -> bool:
        return any(str(lab.value).lower() == "true" for lab in self._labels(pr, head, vocab.READY))

    def _labels(self, pr: int, head: str, key: str) -> list:
        from .store import labels_sync

        target = grant_target(pr, head)
        labels_sync.read_through(self.store, [target], lambda s: self.log(f"research queue: {s}"))
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
                 store=None, shadow: bool = False, rules: Rules | None = None, notes: Path | None = None, log: Callable[[str], None] = print):
        self.workdir, self.remote, self.github, self.grants, self.ready, self.store = Path(workdir), remote, github, grants, ready, store
        self.shadow, self.log, self._rules = shadow, log, rules
        self.branch = SHADOW_BRANCH if shadow else BRANCH
        self.notes = Path(notes or os.environ.get("RESEARCH_NOTES") or Path.home() / ".research" / "notes")
        self.state_path = self.workdir.parent / f"{self.workdir.name}.state.json"

    # -- the clone and its state

    def _clone(self) -> None:
        if not (self.workdir / ".git").exists():
            self.workdir.parent.mkdir(parents=True, exist_ok=True)
            _git(self.workdir.parent, "clone", "-q", self.remote, str(self.workdir))
        # the caller's origin as it is now: a token embedded in the URL the clone was made from rotates
        _git(self.workdir, "remote", "set-url", "origin", self.remote)
        _git(self.workdir, "merge", "--abort", check=False)
        _git(self.workdir, "reset", "-q", "--hard")
        _git(self.workdir, "clean", "-qfdx")

    def fetch(self, prs: Iterable[PR] = ()) -> None:
        """Forced refspecs only: `main`, the queue's branch when it exists, and each open PR's head as `refs/queue/pr/{n}`."""
        specs = ["+refs/heads/main:refs/remotes/origin/main", *(f"+refs/pull/{p.number}/head:refs/queue/pr/{p.number}" for p in prs)]
        _git(self.workdir, "fetch", "-q", "--prune", "origin", *specs)
        if _git(self.workdir, "ls-remote", "--exit-code", "origin", f"refs/heads/{self.branch}", check=False).returncode == 0:
            _git(self.workdir, "fetch", "-q", "origin", f"+refs/heads/{self.branch}:refs/remotes/origin/{self.branch}")
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
                e.state, e.why = "waiting", f"not marked ready at {p.head[:8]}: `research queue ready {p.number}`"
            elif missing := rules.needs(self.workdir, "origin/main", p.head) - self.grants(p.number, p.head):
                e.state, e.why = "waiting", f"waiting for the {', '.join(sorted(missing))} grant at {p.head[:8]}"
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
            _git(self.workdir, "push", "-q", f"--force-with-lease=refs/heads/{self.branch}:{old}", "origin",
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
            _git(self.workdir, "merge", "-q", "--ff-only", m.commit)
            r = _git(self.workdir, "push", "-q", "origin", "main:refs/heads/main", check=False)
            if r.returncode:
                _git(self.workdir, "reset", "-q", "--hard", "refs/remotes/origin/main")
                self.log(f"research queue: pushing `main` to {m.commit[:12]} was refused ({r.stderr.strip()[-200:]}); "
                         f"the next sync rebuilds `{self.branch}` on the new `main`")
                return None
            self.log(f"research queue: landed {prs} on `main` at {m.commit[:12]} (check {attempt})")
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
                e.why = "admitted: merges at the next sync"
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


def ready(github, store, number: int | None, by: str) -> int:
    """`research queue ready PR`: write `ready = true` on `pr:{n}@{its current head}`, and push it to the store's remote."""
    from .store.cli import _label_write_through

    pr = next((p for p in github.open_prs() if p.number == number), None) if number is not None else None
    if pr is None:
        print(f"research queue ready: {'#' + str(number) if number is not None else 'a PR number'} is not an open PR", file=sys.stderr)
        return 2
    target = grant_target(pr.number, pr.head)
    store.add_label(target, vocab.READY, True, by)
    print(f"research queue: #{pr.number} is ready at {pr.head[:12]} ({target}); {_label_write_through(store, target)}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research queue", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["status", "sync", "land", "eject", "ready"])
    ap.add_argument("pr", nargs="?", type=int, help="eject, ready: the PR number")
    ap.add_argument("--reason", help="eject: why, for the owner's notice")
    ap.add_argument("--by", default=os.environ.get("RESEARCH_LANE") or "merge-queue", help="eject, ready: who ejects it or says it's ready")
    ap.add_argument("--shadow", action="store_true", help=f"build `{SHADOW_BRANCH}`, never push `main`, send no notices")
    ap.add_argument("--dry-run", action="store_true", help="sync/land: decide and report, push nothing, tell no one")
    ap.add_argument("--workdir", type=Path, default=Path.home() / ".research" / "queue" / "verity", help="the queue's own clone")
    ap.add_argument("--remote", default=None, help="the repository to clone (default: this checkout's origin)")
    ap.add_argument("--repo", default=None, help="OWNER/NAME on GitHub (default: from the remote's URL)")
    ap.add_argument("--rules", type=Path, default=None, help=f"the admission rules (default: {RULES} on `main`)")
    ap.add_argument("--store", default=None, help="evidence store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--json", action="store_true", help="status: as JSON")
    a = ap.parse_args(argv)
    from .store.local import LocalStore

    try:
        remote = a.remote or _out(Path.cwd(), "remote", "get-url", "origin")
        repo = a.repo or ((m := SLUG_RE.search(remote)) and m.group(1))
        if not repo:
            raise QueueError(f"can't tell OWNER/NAME from {remote!r}: pass --repo")
        store = LocalStore(a.store)
        if a.command == "ready":
            return ready(GitHub(repo), store, a.pr, a.by)
        labels = StoreGrants(store)
        q = Queue(a.workdir, remote=remote, github=GitHub(repo), grants=labels, ready=labels.ready, store=store, shadow=a.shadow,
                  rules=Rules.parse(a.rules.read_text(), str(a.rules)) if a.rules else None)
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
