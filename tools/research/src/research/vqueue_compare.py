r"""`python3 -m research.vqueue_compare`: where the verified queue's shadow and drive's old path agree and disagree, over a
window (`--since`, `--until`, UTC ISO; by default from the log's first decision to now).

It reads the shadow's decision log (`--log`: one JSON object a line, `research.vqueue`'s actions as its loop applied them;
unknown keys ignored, a line that isn't one reported), and the old path from git in a clone (`--repo`): drive's `Queue: #N`
merges on `origin/checked`'s first-parent line (`queue.queue_commit`: each one's `PR` and `Head` trailers; its commit time is
its landing's, since drive merges a member again as it lifts it and lands it in that tick), the `git revert`s there that undo
one (an odd chain of `This reverts commit C`; a `Revert "Queue: #N` subject naming no commit undoes #N's newest landing), and
`origin/main`'s first-parent line.  When `main` reached each commit isn't in git: `--main-moves` gives it, `<UTC ISO> <commit>`
lines (a hex prefix of `PREFIX` digits or more names the one commit of `main`'s line it starts, and a commit off that line
counts as the newest commit of the line it holds); the earliest line is where `main` was then, and each later one reaches the
commits above every line before it.  GitHub's push activity on `main` is such a file:

    gh api 'repos/OWNER/REPO/activity?ref=refs/heads/main&activity_type=push&per_page=100' --paginate \
        --jq '.[] | "\(.timestamp) \(.after)"'

Without it `main`'s arrival is unknown, and the old path's `main` is `main` now.  Each PR's ready mark is the newest `ready =
true` label at a head at or before the event measured from it (`research queue stats` reads the same labels; a head with no
label of its own has no mark here), and an old-path ejection is a head whose newest `ready` label is `false` (`research queue
eject`'s, its reason in `ref`).  `--old-reds` names the old path's after-landing reds, JSON `{tip: [step, ...]}`.

The report (`--json` for the whole of it):

  per PR         each PR either path acted on in the window: its mark; the shadow's first build with it, then its `advance`
                 (when it would have reached `main`), its last `eject` (why, the failed tasks) or `skip`; the old path's
                 landing on `checked`, its revert or ejection, and when `main` reached it.  The shadow is every decision up
                 to `--until`, so a PR's row can show what it did before the window
  aggregates     the count, p50 and p90 of mark -> shadow advance and of mark -> old-path `main`, over the advances and
                 arrivals in the window; the PRs the shadow put on `main` in the window that the old path hadn't by
                 `--until`, and the reverse
  disagreements  (a) the shadow ejected a PR at a head the old path landed and didn't revert;
                 (b) the shadow advanced a candidate holding a PR at a head the old path reverted or ejected;
                 (c) a ready PR the shadow never batched, with the last reason it logged: the PR's `skip`, else a `build`
                     with no sha (no member merged, so nothing was built) after its mark;
                 (d) a candidate the shadow passed whose PRs above `main` (its members and those of the candidates below it),
                     each at its head, are exactly what `checked` landed above that `main` up to a tip `--old-reds` names red
  liveness       the longest a candidate pending during the window waited from its build to its advance or drop (or
                 `--until`), the builds dropped stale (a `drop` with no outcome, but `on main`'s), the retries that enqueued
                 again, the rings, and the actions that failed (a decision with an `error`, which nothing else counts)
  soundness      whether every `advance` was to a candidate the log built, whose base was the logged `main` (or the candidate
                 advanced before it in the same tick), and no outcome logged for it failed: `main` moves only to a passed tree
                 on itself

Exit 0 when the paths agree and every advance was sound, 1 when not, 2 when the inputs can't be read.  `parse_log`,
`parse_chain`, `parse_moves`, `parse_reds`, `compare` and `render` are pure; `main` reads git, the files and the labels
(`store_labels`, which tests replace)."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable

from .digest import REVERT_SUBJECT
from .merge import REVERTS
from .queue import STALL, queue_commit
from .queue_stats import _hours, p90, ts
from .vqueue import FAIL, ON_MAIN, PASS

#: the loop's decision log on the control pod
LOG = Path("/data/vqueue/decisions.jsonl")
BUILD, ADVANCE, DROP, EJECT, RETRY, RING, SKIP = "build", "advance", "drop", "eject", "retry", "ring", "skip"
#: the decisions that name the PRs the shadow acted on; a drop, retry or ring names a candidate
NAMING = (BUILD, ADVANCE, EJECT, SKIP)
#: the shortest commit prefix `--main-moves` and `--old-reds` may give
PREFIX = 7
CHAIN_FORMAT = "%H%x00%P%x00%ct%x00%B%x1e"
#: a batch reader of `ready` labels: [(PR, head)] -> {(PR, head): its labels}, leaving out a head it couldn't read
Labels = Callable[[list[tuple[int, str]]], dict[tuple[int, str], list]]


def clock() -> datetime:
    return datetime.now(UTC)


def when(s: str) -> datetime:
    """A UTC ISO-8601 time: with `Z`, an offset, or neither (UTC)."""
    t = ts(s)
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def iso(t: datetime | None) -> str | None:
    return None if t is None else t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- the decision log -----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    at: datetime
    tick: int
    shadow: bool
    main: str | None
    action: str
    sha: str | None
    base: str | None
    members: tuple[int, ...]
    heads: tuple[str, ...]
    number: int | None
    why: str
    outcome: dict | None
    failed: tuple[dict, ...]
    line: int
    error: str = ""                    # the action failed, and the loop's tick stopped there
    enqueued: bool | None = None       # a retry's: False when it only says why it didn't enqueue

    @property
    def prs(self) -> tuple[int, ...]:
        """The PRs it names: an eject's or a skip's `number`, else its members."""
        if self.number is not None and (self.action in (EJECT, SKIP) or not self.members):
            return (self.number,)
        return self.members

    def head(self, pr: int) -> str | None:
        """``pr``'s head here: at its place in `members`, or the one head of a decision naming one PR."""
        if len(self.heads) == len(self.members) and pr in self.members:
            return self.heads[self.members.index(pr)]
        return self.heads[0] if len(self.heads) == 1 and self.prs == (pr,) else None


def _entry(d: object, line: int) -> Entry:
    if not isinstance(d, dict):
        raise TypeError("not a JSON object")
    outcome, number = d.get("outcome"), d.get("number")
    if outcome is not None and not isinstance(outcome, dict):
        raise TypeError(f"outcome {outcome!r} isn't an object")
    return Entry(at=when(d["at"]), tick=int(d.get("tick") or 0), shadow=bool(d.get("shadow")), main=d.get("main"),
                 action=str(d["action"]), sha=d.get("sha"), base=d.get("base"),
                 members=tuple(int(n) for n in d.get("members") or ()), heads=tuple(str(h) for h in d.get("heads") or ()),
                 number=None if number is None else int(number), why=str(d.get("why") or ""), outcome=outcome,
                 failed=tuple(f for f in d.get("failed") or () if isinstance(f, dict)), line=line,
                 error=str(d.get("error") or ""), enqueued=None if d.get("enqueued") is None else bool(d["enqueued"]))


def parse_log(text: str) -> tuple[list[Entry], list[tuple[int, str]]]:
    """The log's decisions, oldest first, and (line, why) of each other non-blank line: one the loop is still writing, or junk."""
    out, bad = [], []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(_entry(json.loads(line), i))
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            bad.append((i, f"{type(e).__name__}: {e}"[:200]))
    out.sort(key=lambda e: (e.at, e.line))
    return out, bad


@dataclass
class Candidate:
    sha: str
    base: str | None
    built: datetime
    members: dict[int, str | None]
    ended: Entry | None = None             # its advance or drop
    passed: Entry | None = None            # its advance, or a drop recording a pass
    outcome: str | None = None             # the state of the outcome a drop recorded for it


@dataclass
class Shadow:
    """The decisions up to the window's end: each candidate by its sha, and the first or every decision naming each PR, with
    its head."""
    cands: dict[str, Candidate] = field(default_factory=dict)
    built: dict[int, tuple[Entry, str | None]] = field(default_factory=dict)
    advanced: dict[int, tuple[Entry, str | None]] = field(default_factory=dict)
    ejects: dict[int, list[tuple[Entry, str | None]]] = field(default_factory=dict)
    skips: dict[int, list[tuple[Entry, str | None]]] = field(default_factory=dict)
    heads: dict[int, list[str]] = field(default_factory=dict)
    advances: list[Entry] = field(default_factory=list)
    nobuild: list[Entry] = field(default_factory=list)     # why nothing was built: a build no member merged into
    stale: list[Entry] = field(default_factory=list)       # drops with no outcome, of a candidate not on `main`
    retries: list[Entry] = field(default_factory=list)     # those that enqueued again
    rings: list[Entry] = field(default_factory=list)
    errors: list[Entry] = field(default_factory=list)      # actions that failed: none of them took effect
    unattributed: list[Entry] = field(default_factory=list)   # advances naming no members, of a candidate the log didn't build


def shadow(entries: list[Entry]) -> Shadow:
    s = Shadow()
    for e in entries:
        if e.error:
            s.errors.append(e)
            if not (e.action == BUILD and e.sha):          # a build naming its sha was pushed: it is a candidate still
                continue
        c = s.cands.get(e.sha or "")
        for pr in e.prs if e.action in NAMING else ():
            if (h := e.head(pr)) and h not in s.heads.setdefault(pr, []):
                s.heads[pr].append(h)
        if e.action == BUILD and not e.sha:
            s.nobuild.append(e)
        elif e.action == BUILD and c is None:
            c = s.cands[e.sha] = Candidate(e.sha, e.base, e.at, {pr: e.head(pr) for pr in e.members})
            for pr, h in c.members.items():
                s.built.setdefault(pr, (e, h))
        elif e.action == ADVANCE:
            s.advances.append(e)
            members = {pr: e.head(pr) for pr in e.members} or (c.members if c else {})
            if not members:
                s.unattributed.append(e)
            for pr, h in members.items():
                s.advanced.setdefault(pr, (e, h))
            if c is not None:
                c.ended, c.passed = c.ended or e, c.passed or e
        elif e.action == DROP:
            state = (e.outcome or {}).get("state")
            if e.outcome is None and e.why != ON_MAIN:
                s.stale.append(e)
            if c is not None:
                c.ended, c.outcome = c.ended or e, c.outcome or state
                c.passed = c.passed or (e if state == PASS else None)
        elif e.action in (EJECT, SKIP):
            for pr in e.prs:
                (s.ejects if e.action == EJECT else s.skips).setdefault(pr, []).append((e, e.head(pr)))
            if e.action == SKIP and not e.prs:
                s.nobuild.append(e)
        elif e.action == RETRY and e.enqueued is not False:
            s.retries.append(e)
        elif e.action == RING:
            s.rings.append(e)
    return s


def holds(s: Shadow, sha: str) -> tuple[dict[int, str | None], str | None]:
    """What candidate ``sha`` holds above `main`, its members and those of each candidate below it ({PR: head}), and the base
    the bottom one was built on."""
    out: dict[int, str | None] = {}
    c, base, seen = s.cands.get(sha), None, set()
    while c is not None and c.sha not in seen:
        seen.add(c.sha)
        for pr, h in c.members.items():
            out.setdefault(pr, h)
        base, c = c.base, s.cands.get(c.base or "")
    return out, base


# ---- git ------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Landing:
    """A `Queue: #N` merge on `checked`'s first-parent line: #``pr`` at ``head``, and the revert that undid it, if one did."""
    pr: int
    head: str
    commit: str
    at: datetime
    undone: str | None = None
    undone_at: datetime | None = None


@dataclass
class Chain:
    """`checked`'s first-parent line, oldest first: each commit's time, its landings by merge commit, and the commits each
    revert on it names."""
    commits: list[str]
    times: dict[str, datetime]
    landings: dict[str, Landing]
    undoes: dict[str, tuple[str, ...]]
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {c: i for i, c in enumerate(self.commits)}


def _undone(c: str, by: dict[str, str]) -> str | None:
    """The revert that undoes ``c``: the first of an odd chain of reverts from it (a revert of that revert relands it)."""
    n, x = 0, c
    while x in by:
        x, n = by[x], n + 1
    return by[c] if n % 2 else None


def parse_chain(log: str) -> Chain:
    """`checked`'s line from `git log --first-parent --format=CHAIN_FORMAT` (newest first)."""
    recs = []
    for rec in log.split("\x1e"):
        if rec.strip():
            commit, parents, ct, body = rec.lstrip("\n").split("\x00", 3)
            recs.append((commit, parents, datetime.fromtimestamp(int(ct), UTC), body))
    commits, times, made, undoes = [], {}, {}, {}
    for commit, parents, at, body in reversed(recs):
        if (m := queue_commit(commit, parents, body)) is not None:
            made[commit] = m
        named = tuple(REVERTS.findall(body))
        if not named and (subject := REVERT_SUBJECT.match(body)):
            last = next((c for c in reversed(commits) if c in made and made[c].pr == int(subject[1])), None)
            named = (last,) if last else ()
        if named:
            undoes[commit] = named
        commits.append(commit)
        times[commit] = at
    by: dict[str, str] = {}
    for r, targets in undoes.items():
        for t in targets:
            by.setdefault(t, r)
    landings = {}
    for c, m in made.items():
        u = _undone(c, by)
        landings[c] = Landing(m.pr, m.head, c, times[c], u, times.get(u) if u else None)
    return Chain(commits, times, landings, undoes)


def resolve(c: str, commits: list[str], index: dict[str, int]) -> str | None:
    """The one commit of ``commits`` that ``c`` is, or starts with when it is a hex prefix of `PREFIX` digits or more."""
    c = c.strip().lower()
    if c in index:
        return c
    if len(c) < PREFIX or c.strip("0123456789abcdef"):
        return None
    hits = [x for x in commits if x.startswith(c)]
    return hits[0] if len(hits) == 1 else None


def parse_moves(text: str, line: list[str], onto: Callable[[str], str | None] | None = None,
                ) -> tuple[list[tuple[datetime, str]], list[str]]:
    """``text``'s `<UTC ISO> <commit>` lines (blank and `#` lines skipped), each commit resolved on `main`'s first-parent
    ``line``, and why each other line was left out.  A commit `main` reached off that line (one `checked` merged later as a
    second parent) is read as ``onto``'s answer, the newest commit of the line it holds."""
    index = {c: i for i, c in enumerate(line)}
    moves, warn = [], []
    for i, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split()
        try:
            at = when(parts[0])
        except ValueError as e:
            warn.append(f"--main-moves line {i}: {raw.strip()!r}: {e}; want `<UTC ISO> <commit>`: left out")
            continue
        c = resolve(parts[1], line, index) if len(parts) == 2 else None
        if c is None and onto is not None and len(parts) == 2:
            c = onto(parts[1])
        if c is None:
            warn.append(f"--main-moves line {i}: {raw.strip()!r} names no one commit of `main`'s first-parent line: left out")
        else:
            moves.append((at, c))
    return moves, warn


def arrivals(line: list[str], moves: list[tuple[datetime, str]]) -> dict[str, datetime]:
    """When `main` reached each commit of its first-parent ``line`` (oldest first), from ``moves`` ((time, commit of ``line``)):
    the earliest is where `main` was then, and each later one reaches the commits above every move before it, up to its own."""
    index = {c: i for i, c in enumerate(line)}
    out: dict[str, datetime] = {}
    top = None
    for at, c in sorted(moves, key=lambda m: (m[0], index[m[1]])):
        i = index[c]
        for j in range(i + 1 if top is None else top + 1, i + 1):
            out[line[j]] = at
        top = i if top is None else max(top, i)
    return out


def parse_reds(doc: object, chain: Chain) -> tuple[dict[str, list[str]], list[str]]:
    """``doc``'s {tip: [step, ...]}, each tip resolved on `checked`'s line, and why each other tip was left out."""
    if not isinstance(doc, dict):
        return {}, ["--old-reds isn't a JSON object {tip: [step, ...]}: no red is compared"]
    reds, warn = {}, []
    for tip, steps in doc.items():
        c = resolve(str(tip), chain.commits, chain.index)
        if c is None:
            warn.append(f"--old-reds: {tip} names no one commit of `checked`'s first-parent line: left out")
        else:
            reds[c] = [str(x) for x in steps] if isinstance(steps, list) else [str(steps)]
    return reds, warn


def same_prs(chain: Chain, base: str, held: dict[int, str | None], reds: dict[str, list[str]]) -> str | None:
    """The first tip of ``reds`` above ``base`` on `checked`'s line whose landings above ``base``, net of the reverts between,
    are exactly ``held`` ({PR: head, None where unknown}); a revert there of a commit not above ``base`` makes it another tree."""
    lo = chain.index[base]
    tips = sorted((chain.index[t], t) for t in reds if chain.index.get(t, -1) > lo)
    up: dict[str, Landing] = {}
    other, at = False, lo
    for i, tip in tips:
        for c in chain.commits[at + 1:i + 1]:
            if c in chain.landings:
                up[c] = chain.landings[c]
            for t in chain.undoes.get(c, ()):
                if t in up:
                    del up[t]
                elif chain.index.get(t, -1) <= lo or t in chain.undoes:
                    other = True
        at = i
        got = {x.pr: x.head for x in up.values()}
        if not other and got.keys() == held.keys() and all(h is None or got[pr] == h for pr, h in held.items()):
            return tip
    return None


# ---- labels ---------------------------------------------------------------------------------------------------------------------

def _true(lab) -> bool:
    return str(lab.value).lower() == "true"


def marked(labs: Iterable, at: datetime | None = None) -> datetime | None:
    """The newest `ready = true` label's time among ``labs``, at or before ``at`` when given."""
    times = [when(lab.ts) for lab in labs if _true(lab)]
    return max((t for t in times if at is None or t <= at), default=None)


def ejection(labs: Iterable):
    """The newest of ``labs`` when it is `research queue eject`'s `ready = false`, its reason in `ref` (`queue.ejection`)."""
    last = max(labs, key=lambda lab: lab.ts, default=None)
    return last if last is not None and str(last.value).lower() == "false" and last.ref else None


def store_labels(root: str | None) -> Labels:
    """The batch reader `main` uses: the `ready` labels on each `pr:{n}@{head}` in the evidence store at ``root``, pulled from
    its remote first as drive reads them (`queue.StoreLabels`); a head whose labels can't be pulled is left out, and said."""
    from .queue import QueueError, StoreLabels
    from .store import vocab
    from .store.local import LocalStore

    def read(heads: list[tuple[int, str]]) -> dict[tuple[int, str], list]:
        labels = StoreLabels(LocalStore(root), log=lambda s: None)
        labels.prefetch(heads)
        out, unread = {}, []
        for pr, head in heads:
            try:
                out[(pr, head)] = labels._labels(pr, head, vocab.READY)
            except QueueError as e:
                unread.append(f"pr:{pr}@{head[:12]}: {e}")
        if unread:
            print(f"research vqueue-compare: the `ready` labels of {len(unread)} of {len(heads)} head(s) weren't read, so their "
                  f"marks are unknown; the first: {unread[0]}", file=sys.stderr)
        return out

    return read


# ---- the comparison -------------------------------------------------------------------------------------------------------------

def _stats(hours: list[float]) -> dict:
    return {"n": len(hours), "p50_h": statistics.median(hours), "p90_h": p90(hours)} if hours else {"n": 0}


def _failed(e: Entry) -> list[dict]:
    return [{"task": f.get("task"), "key": f.get("key")} for f in e.failed]


def compare(entries: list[Entry], chain: Chain, line: list[str], arrived: dict[str, datetime] | None,
            reds: dict[str, list[str]], labels: Labels, since: datetime, until: datetime) -> dict:
    """The report over [``since``, ``until``]: ``line`` is `main`'s first-parent line, oldest first, and ``arrived`` is
    `arrivals`' (None without `--main-moves`).  The shadow is every decision up to ``until`` (a candidate built before the
    window is still what an advance in it moved `main` to), and the window picks what is reported: the PRs either path acted
    on in it, the advances, ejects, drops, retries and rings in it, and the candidates pending during it."""
    def inside(t: datetime | None) -> bool:
        return t is not None and since <= t <= until

    window = [e for e in entries if inside(e.at)]
    s = shadow([e for e in entries if e.at <= until])
    advances = [e for e in s.advances if inside(e.at)]
    advanced = {pr: (e, head) for pr, (e, head) in s.advanced.items() if inside(e.at)}
    on_main = set(line)
    landed: dict[int, list[Landing]] = {}
    for x in chain.landings.values():
        landed.setdefault(x.pr, []).append(x)

    def arrival(x: Landing) -> datetime | None:
        return None if arrived is None else arrived.get(x.commit)

    def live(pr: int, head: str | None = None) -> Landing | None:
        return next((x for x in reversed(landed.get(pr, ())) if x.undone is None and head in (None, x.head)), None)

    def reverted(pr: int, head: str | None = None) -> Landing | None:
        return None if live(pr, head) else next((x for x in reversed(landed.get(pr, ())) if x.undone and head in (None, x.head)),
                                                None)

    def on_main_by_until(x: Landing) -> bool:
        return (a := arrival(x)) is not None and a <= until if arrived is not None else x.commit in on_main

    old = {x.pr for x in chain.landings.values() if inside(x.at) or inside(arrival(x)) or inside(x.undone_at)}
    named = {pr for e in window if e.action in NAMING for pr in e.prs}
    scope = sorted(named | set(advanced) | old)
    heads = sorted({(pr, h) for pr in scope for h in [*s.heads.get(pr, ()), *(x.head for x in landed.get(pr, ()))]})
    got = labels(heads)
    unread = [f"pr:{pr}@{h[:12]}" for pr, h in heads if (pr, h) not in got]

    def mark(pr: int, head: str | None, at: datetime | None = None) -> datetime | None:
        return marked(got.get((pr, head), ()), at) if head else None

    def old_eject(pr: int, head: str | None):
        lab = ejection(got.get((pr, head), ())) if head and live(pr, head) is None else None
        return lab and {"at": iso(when(lab.ts)), "by": lab.by, "why": lab.ref or "", "head": head}

    rows = []
    for pr in scope:
        b, adv, x, r = s.built.get(pr), s.advanced.get(pr), live(pr), reverted(pr)
        ej, sk = (s.ejects.get(pr) or [None])[-1], (s.skips.get(pr) or [None])[-1]
        if x:
            head = x.head
        elif adv and adv[1]:
            head = adv[1]
        else:
            head = s.heads[pr][-1] if s.heads.get(pr) else r.head if r else None
        events = [(e.at, h) for e, h in filter(None, (b, adv))] + ([(x.at, x.head)] if x else [])
        first = min((at for at, h in events if h == head), default=None)
        rows.append({
            "pr": pr, "head": head, "marked": iso(mark(pr, head, first)),
            "first_build": b and {"at": iso(b[0].at), "sha": b[0].sha, "head": b[1]},
            "advance": adv and {"at": iso(adv[0].at), "sha": adv[0].sha, "head": adv[1]},
            "eject": ej and {"at": iso(ej[0].at), "head": ej[1], "why": ej[0].why, "failed": _failed(ej[0])},
            "skip": sk and {"at": iso(sk[0].at), "head": sk[1], "why": sk[0].why},
            "landed": x and {"at": iso(x.at), "commit": x.commit, "head": x.head},
            "reverted": r and {"at": iso(r.undone_at), "commit": r.undone, "landed": r.commit, "head": r.head},
            "ejected": old_eject(pr, head),
            "main": iso(arrival(x)) if x else None, "on_main": bool(x and x.commit in on_main)})

    to_advance = [h for pr, (e, head) in advanced.items() if (h := _hours(mark(pr, head, e.at), e.at)) is not None]
    old_main, old_ever, to_main = set(), set(), []
    for pr in scope:
        if (x := live(pr)) is None:
            continue
        if on_main_by_until(x):
            old_ever.add(pr)
        if inside(arrival(x)) if arrived is not None else x.commit in on_main:
            old_main.add(pr)
            if (h := _hours(mark(pr, x.head, x.at), arrival(x))) is not None:
                to_main.append(h)

    out: list[dict] = []
    for pr, xs in sorted(s.ejects.items()):
        for e, h in {h: (e, h) for e, h in xs if inside(e.at)}.values():
            if (x := live(pr, h)) is not None:
                out.append({"kind": "a", "pr": pr, "head": x.head,
                            "shadow": {"eject": iso(e.at), "why": e.why, "failed": _failed(e)},
                            "old": {"landed": iso(x.at), "commit": x.commit}})
    flagged: set[tuple[int, str | None]] = set()
    for e in advances:
        held = holds(s, e.sha or "")[0] if e.sha in s.cands else {}
        for pr in e.members:
            if (h := e.head(pr)) is not None or pr not in held:
                held[pr] = h
        for pr, h in held.items():
            if (pr, h) in flagged:
                continue
            seen = {"advance": iso(e.at), "sha": e.sha}
            if (r := reverted(pr, h)) is not None:
                out.append({"kind": "b", "pr": pr, "head": r.head, "shadow": seen,
                            "old": {"reverted": iso(r.undone_at), "revert": r.undone, "landed": r.commit}})
            elif ej := old_eject(pr, h):
                out.append({"kind": "b", "pr": pr, "head": h, "shadow": seen, "old": {"ejected": ej["at"], "by": ej["by"],
                                                                                      "why": ej["why"]}})
            else:
                continue
            flagged.add((pr, h))
    for row in rows:
        pr = row["pr"]
        if pr in s.built or pr in s.advanced or pr in s.ejects:
            continue
        m = when(row["marked"]) if row["marked"] else None
        if not (landed.get(pr) or s.skips.get(pr) or m):
            continue
        if sk := (s.skips.get(pr) or [None])[-1]:
            why = {"skip": iso(sk[0].at), "why": sk[0].why}
        elif nb := [e for e in s.nobuild if m is None or e.at >= m]:
            why = {"no_build": iso(nb[-1].at), "why": nb[-1].why}
        else:
            why = {}
        out.append({"kind": "c", "pr": pr, "head": row["head"], "marked": row["marked"], "shadow": why})
    for c in s.cands.values():
        if c.passed is None or not inside(c.passed.at):
            continue
        held, base = holds(s, c.sha)
        if base in chain.index and (tip := same_prs(chain, base, held, reds)):
            out.append({"kind": "d", "pr": None, "head": None, "sha": c.sha, "base": base, "prs": sorted(held),
                        "shadow": {"passed": iso(c.passed.at)}, "old": {"tip": tip, "red": reds[tip]}})

    longest = None
    if pending := [c for c in s.cands.values() if c.ended is None or c.ended.at >= since]:
        c = max(pending, key=lambda c: (c.ended.at if c.ended else until) - c.built)
        longest = {"hours": _hours(c.built, c.ended.at if c.ended else until), "sha": c.sha, "members": sorted(c.members),
                   "built": iso(c.built), "ended": c.ended and {"at": iso(c.ended.at), "action": c.ended.action}}

    bad, before = [], {}
    for e in advances:
        cand = s.cands.get(e.sha or "")
        base = e.base or (cand.base if cand else None)
        if base is None or base not in (e.main, before.get((e.tick, e.main))):
            bad.append({"at": iso(e.at), "sha": e.sha, "base": base, "main": e.main,
                        "why": "its base isn't the logged `main`" if base else "it names no base"})
        elif cand is None:
            bad.append({"at": iso(e.at), "sha": e.sha, "base": base, "main": e.main, "why": "the log holds no build of it"})
        elif cand.outcome == FAIL:
            bad.append({"at": iso(e.at), "sha": e.sha, "base": base, "main": e.main, "why": "the outcome logged for it failed"})
        before[(e.tick, e.main)] = e.sha

    warn = [f"the `ready` labels of {len(unread)} head(s) weren't read, so their marks are unknown: {', '.join(unread[:5])}"] \
        if unread else []
    warn += [f"advance of {e.sha} at {iso(e.at)} names no members, and the log has no build of it: its PRs aren't counted"
             for e in s.unattributed if inside(e.at)]
    if not window:
        warn.append("no decision in the window")
    actions: dict[str, int] = {}
    for e in window:
        actions[e.action] = actions.get(e.action, 0) + 1
    passed = [e for e in advances if e.sha in s.cands and s.cands[e.sha].outcome == PASS]
    stale, retries, rings, errors = ([e for e in xs if inside(e.at)] for xs in (s.stale, s.retries, s.rings, s.errors))
    return {
        "window": {"since": iso(since), "until": iso(until)},
        "log": {"decisions": len(window), "ticks": len({e.tick for e in window}), "actions": actions,
                "not_shadow": sum(not e.shadow for e in window)},
        "main_moves": arrived is not None,
        "prs": rows,
        "aggregates": {"mark_to_advance": _stats(to_advance), "mark_to_main": _stats(to_main) if arrived is not None else None,
                       "shadow_only": sorted(set(advanced) - old_ever), "old_only": sorted(old_main - set(s.advanced))},
        "disagreements": out,
        "liveness": {"longest_pending": longest,
                     "dropped_stale": [{"at": iso(e.at), "sha": e.sha, "why": e.why} for e in stale],
                     "retries": [{"at": iso(e.at), "sha": e.sha, "why": e.why} for e in retries],
                     "rings": [{"at": iso(e.at), "sha": e.sha, "why": e.why} for e in rings],
                     "failed_actions": [{"at": iso(e.at), "action": e.action, "sha": e.sha, "error": e.error} for e in errors]},
        "soundness": {"advances": len(advances), "passed": len(passed), "violations": bad, "ok": not bad},
        "warnings": warn,
    }


# ---- the text -------------------------------------------------------------------------------------------------------------------

def hm(t: str | None) -> str:
    return f"{when(t):%m-%d %H:%MZ}" if t else "-"


def short(sha: str | None) -> str:
    return (sha or "?")[:12]


def plain(text: str, n: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 1] + "…"


def tasks(failed: list[dict]) -> str:
    return ", ".join(f"{f['task']} ({f['key']})" if f.get("key") else str(f["task"]) for f in failed) or "none named"


def _table(rows: list[list[str]]) -> list[str]:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return ["  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]


def _cells(r: dict) -> list[str]:
    shadow = (f"advance {hm(r['advance']['at'])} {short(r['advance']['sha'])}" if r["advance"] else
              f"eject {hm(r['eject']['at'])}: {plain(r['eject']['why'], 40)}" if r["eject"] else
              f"built {short(r['first_build']['sha'])}" if r["first_build"] else
              f"skip {hm(r['skip']['at'])}: {plain(r['skip']['why'], 40)}" if r["skip"] else "-")
    old = (f"landed {hm(r['landed']['at'])} {short(r['landed']['commit'])}" if r["landed"] else
           f"reverted {hm(r['reverted']['at'])} {short(r['reverted']['commit'])}" if r["reverted"] else
           f"ejected {hm(r['ejected']['at'])}" if r["ejected"] else "-")
    main = hm(r["main"]) if r["main"] else "yes, time unknown" if r["on_main"] else "-"
    build = f"{hm(r['first_build']['at'])}" if r["first_build"] else "-"
    return [f"#{r['pr']}", short(r["head"]) if r["head"] else "-", hm(r["marked"]), build, shadow, old, main]


def _dis(d: dict) -> str:
    at = f"#{d['pr']} at {short(d['head'])}" if d["pr"] is not None else ""
    sh, old = d["shadow"], d.get("old", {})
    if d["kind"] == "a":
        return (f"(a) {at}: the shadow ejected it {hm(sh['eject'])} ({plain(sh['why'])}; failed: {tasks(sh['failed'])}), and "
                f"the old path landed it {hm(old['landed'])} as {short(old['commit'])}")
    if d["kind"] == "b":
        did = (f"reverted it {hm(old['reverted'])} ({short(old['revert'])} undoes {short(old['landed'])})" if "reverted" in old
               else f"ejected it {hm(old['ejected'])} by {old['by']}: {plain(old['why'])}")
        return f"(b) {at}: the shadow advanced {short(sh['sha'])} with it {hm(sh['advance'])}, and the old path {did}"
    if d["kind"] == "c":
        why = (f"its skip {hm(sh['skip'])}: {plain(sh['why'])}" if "skip" in sh else
               f"no build {hm(sh['no_build'])}: {plain(sh['why'])}" if "no_build" in sh else "nothing logged")
        return f"(c) #{d['pr']}: ready (marked {hm(d['marked'])}) and never batched; the last reason: {why}"
    return (f"(d) {short(d['sha'])} passed {hm(sh['passed'])} with {' '.join(f'#{n}' for n in d['prs'])} on "
            f"{short(d['base'])}, and the old path's check of {short(old['tip'])}, the same PRs on it, was red: "
            f"{', '.join(old['red'])}")


def _spread(name: str, v: dict | None) -> str:
    if v is None:
        return f"{name:28} unknown: no --main-moves"
    return f"{name:28} " + (f"n={v['n']}  p50 {v['p50_h']:.2f} h  p90 {v['p90_h']:.2f} h" if v["n"] else "n=0")


def _prs(prs: list[int]) -> str:
    return f"{len(prs)}" + (f" ({' '.join(f'#{n}' for n in prs)})" if prs else "")


def render(rep: dict) -> str:
    log, agg, live, sound = rep["log"], rep["aggregates"], rep["liveness"], rep["soundness"]
    acts = ", ".join(f"{n} {a}" for a, n in sorted(log["actions"].items()))
    head = f"{log['decisions']} decision(s) in {log['ticks']} tick(s)" + (f" ({acts})" if acts else "")
    if log["not_shadow"]:
        head += f", {log['not_shadow']} not shadow"
    lines = [f"vqueue compare {hm(rep['window']['since'])} to {hm(rep['window']['until'])}: {head}; {len(rep['prs'])} PR(s) seen"]
    if rep["prs"]:
        lines += _table([["PR", "head", "marked", "first build", "shadow", "old path", "main"], *map(_cells, rep["prs"])])
    lines += [_spread("mark -> shadow advance", agg["mark_to_advance"]), _spread("mark -> old-path main", agg["mark_to_main"])]
    lines.append(f"on main by the shadow only: {_prs(agg['shadow_only'])}; by the old path only: {_prs(agg['old_only'])}"
                 + ("" if rep["main_moves"] else " (the old path's: on `main` now)"))
    dis = rep["disagreements"]
    lines.append(f"disagreements: {len(dis)}")
    lines += [f"  {_dis(d)}" for d in sorted(dis, key=lambda d: (d["kind"], d["pr"] or 0))]
    if lp := live["longest_pending"]:
        end = f"{lp['ended']['action']} {hm(lp['ended']['at'])}" if lp["ended"] else "still pending"
        lines.append(f"longest pending: {lp['hours']:.2f} h, {short(lp['sha'])} ({' '.join(f'#{n}' for n in lp['members'])}) "
                     f"built {hm(lp['built'])}, {end}")
    else:
        lines.append("longest pending: no candidate built")
    lines.append(f"builds dropped stale: {len(live['dropped_stale'])}; infra retries: {len(live['retries'])}; "
                 f"rings: {len(live['rings'])}; failed actions: {len(live['failed_actions'])}")
    for f in live["failed_actions"][:5]:
        what = f"{f['action']} {short(f['sha'])}" if f["sha"] else f["action"]
        lines.append(f"  {what} {hm(f['at'])} failed: {plain(f['error'])}")
    if sound["ok"]:
        lines.append(f"soundness: every advance ({sound['advances']}) was to a candidate the log built on the logged main, none "
                     f"with a failed outcome ({sound['passed']} with a logged pass)")
    else:
        lines.append(f"soundness: VIOLATED by {len(sound['violations'])} of {sound['advances']} advance(s):")
        lines += [f"  {short(v['sha'])} {hm(v['at'])}: {v['why']} (base {short(v['base'])}, main {short(v['main'])})"
                  for v in sound["violations"]]
    lines += [f"warning: {w}" for w in rep["warnings"]]
    return "\n".join(lines)


# ---- the command ----------------------------------------------------------------------------------------------------------------

class GitError(RuntimeError):
    pass


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if r.returncode:
        raise GitError(f"git {' '.join(args)} in {repo}: {(r.stderr or r.stdout).strip()[-400:]}")
    return r.stdout


def _onto(repo: Path, line: list[str]) -> Callable[[str], str | None]:
    """The newest commit of `main`'s first-parent ``line`` that a commit holds, down that commit's own first-parent line;
    None for one git doesn't have (the move is then left out, and said)."""
    index = set(line)

    def onto(c: str) -> str | None:
        r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=10000", f"{c}^{{commit}}", "--"], cwd=repo,
                           capture_output=True, text=True)
        return None if r.returncode else next((x for x in r.stdout.split() if x in index), None)

    return onto


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python3 -m research.vqueue_compare", description=__doc__.split("\n\n")[0],
                                 epilog="exit 0: the paths agree and every advance was sound; 1: not; 2: an input can't be read")
    ap.add_argument("--log", type=Path, default=LOG, help=f"the shadow's decision log (default {LOG})")
    ap.add_argument("--repo", type=Path, default=Path("."), help="a clone whose --checked and --main are fetched (default .)")
    ap.add_argument("--checked", default="origin/checked", help="the old path's landing branch (default origin/checked)")
    ap.add_argument("--main", default="origin/main", help="`main` (default origin/main)")
    ap.add_argument("--fetch", action="store_true", help="fetch origin's `main` and `checked` into --repo first")
    ap.add_argument("--since", help="the window's start, UTC ISO (default: the log's first decision)")
    ap.add_argument("--until", help="the window's end, UTC ISO (default: now)")
    ap.add_argument("--main-moves", type=Path, help="`<UTC ISO> <commit>` lines: when `main` reached each commit")
    ap.add_argument("--old-reds", type=Path, help="JSON {tip: [step, ...]}: the old path's after-landing reds")
    ap.add_argument("--store", default=None, help="evidence store root for the `ready` labels (default $RESEARCH_STORE or "
                                                  "~/.research/store)")
    ap.add_argument("--json", action="store_true", help="the whole report as JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    try:
        if a.fetch:
            _git(a.repo, *STALL, "fetch", "-q", "origin", "main", "checked")
        chain = parse_chain(_git(a.repo, "log", "--first-parent", f"--format={CHAIN_FORMAT}", a.checked, "--"))
        line = _git(a.repo, "rev-list", "--first-parent", a.main, "--").split()[::-1]
        entries, bad = parse_log(a.log.read_text(errors="replace"))
        moves, warn = parse_moves(a.main_moves.read_text(), line, _onto(a.repo, line)) if a.main_moves else ([], [])
        reds, also = parse_reds(json.loads(a.old_reds.read_text()), chain) if a.old_reds else ({}, [])
        until = when(a.until) if a.until else clock()
        since = when(a.since) if a.since else entries[0].at if entries else until
    except (GitError, OSError, ValueError) as e:
        print(f"research vqueue-compare: {e}", file=sys.stderr)
        return 2
    rep = compare(entries, chain, line, arrivals(line, moves) if a.main_moves else None, reds, store_labels(a.store), since,
                  until)
    rep["log"]["unreadable"] = [{"line": i, "why": why} for i, why in bad]
    rep["warnings"] = [*(f"log line {i} isn't a decision ({why}): left out" for i, why in bad), *warn, *also, *rep["warnings"]]
    print(json.dumps(rep, indent=1) if a.json else render(rep))
    return 0 if rep["soundness"]["ok"] and not rep["disagreements"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
