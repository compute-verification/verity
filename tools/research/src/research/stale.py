"""Stale PRs (TOP-90 row 36): an open PR that isn't marked ready at its head and has been idle `stale_days` (drive.toml) is
stale.  Idle is since the latest of its head commit's date and the ready (an eject's `ready = false` too) and review labels on
that head.  `research queue status` says so, drive rings its owner once a day, and at `stale_close_days` drive closes it with
`close_comment` unless it carries the `hold` label, is a draft that says "keep open", or is on `next`; its branch stays
(`merge.sweep_branches` leaves it, `Driver._sweep`)."""
from __future__ import annotations

import calendar
import re
import subprocess
import sys
import time
from typing import Callable, Collection, Iterable

from .queue import PR, Queue, QueueError, _git

DAY_S = 86400
#: a draft whose title or body says this is never closed for being idle
KEEP_OPEN_RE = re.compile(r"\bkeep[ -]open\b", re.I)


def close_comment(days: int) -> str:
    return (f"Closed by drive after {days} days unmarked and idle (TOP-90 row 36). The branch stays: reopen the PR, push or mark "
            "it ready, and it queues again.")


def keep_open(pr: PR) -> bool:
    return pr.draft and bool(KEEP_OPEN_RE.search(f"{pr.title}\n{pr.body}"))


def _ts(stamp: str | None) -> float | None:
    """A label's UTC time (`YYYY-MM-DDTHH:MM:SSZ`) as seconds."""
    try:
        return float(calendar.timegm(time.strptime(stamp.rstrip("Z")[:19], "%Y-%m-%dT%H:%M:%S"))) if stamp else None
    except ValueError:
        print(f"stale: a label's time {stamp!r} isn't YYYY-MM-DDTHH:MM:SSZ, so it doesn't count", file=sys.stderr)
        return None


def pull_refs(q: Queue, prs: Iterable[PR]) -> dict[int, str]:
    """{number: its `refs/queue/pr/N`} after fetching the pull ref of each of ``prs`` whose ref in the queue's clone isn't at its
    head: the sync fetches only PRs based on `main`."""
    def refs() -> dict[int, str]:
        out = _git(q.workdir, "for-each-ref", "--format=%(refname:lstrip=3) %(objectname)", "refs/queue/pr/").stdout
        return {int(n): sha for n, sha in (line.split() for line in out.splitlines()) if n.isdigit()}

    have = refs()
    if missing := [p.number for p in prs if have.get(p.number) != p.head]:
        q._remote(q.workdir, "fetch", "-q", "origin", *(f"+refs/pull/{n}/head:refs/queue/pr/{n}" for n in missing), ok=None)
        have = refs()
    return have


def commit_times(repo, shas: Iterable[str]) -> dict[str, float]:
    """{commit: its committer date} of each of ``shas`` the repository has."""
    shas = sorted(set(shas))
    if not shas:
        return {}
    r = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"], cwd=repo, input="\n".join(shas) + "\n",
                       capture_output=True, text=True)
    present = [line.split()[0] for line in r.stdout.splitlines() if line.endswith(" commit")]
    if not present:
        return {}
    out = _git(repo, "show", "-s", "--format=%H %ct", *present).stdout
    return {sha: float(ct) for sha, ct in (line.split() for line in out.splitlines())}


def stale_prs(q: Queue, prs: Iterable[PR], now: float, days: int,
              reviewed_at: Callable[[int, str], list[str]] | None = None, keep: Collection[int] = ()) -> dict[int, float]:
    """{number: when it went idle} of each of ``prs`` that is stale at ``now`` (the module's rule; ``days`` 0: none), but one
    carrying the hold label or listed in ``keep`` (drive.toml's `stale_keep`).  ``reviewed_at``: the times of the review labels on a PR's head.  A PR whose labels can't be read
    is left out, and logged."""
    if days <= 0:
        return {}
    hold = q.rules().hold_label
    prs = [p for p in prs if hold not in p.labels and p.number not in keep]
    pull_refs(q, prs)
    born = commit_times(q.workdir, (p.head for p in prs))
    cut = now - days * DAY_S
    old = [p for p in prs if p.head in born and born[p.head] <= cut]
    if old and q.prefetch is not None:
        q.prefetch([(p.number, p.head) for p in old])
    out = {}
    for p in old:
        try:
            if q.ready(p.number, p.head):
                continue
            stamps = [q.marked_at(p.number, p.head) if q.marked_at else None, *(reviewed_at(p.number, p.head) if reviewed_at else ())]
        except QueueError as e:
            q.log(f"research queue: #{p.number}'s labels can't be read, so it isn't judged stale now: {e}")
            continue
        since = max([born[p.head], *(t for s in stamps if (t := _ts(s)) is not None)])
        if since <= cut:
            out[p.number] = since
    return out
