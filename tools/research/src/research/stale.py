from __future__ import annotations

import calendar
import subprocess
import sys
import time
from typing import Callable, Collection, Iterable

from .queue import PR, Queue, QueueError, _git

DAY_S = 86400


def _ts(stamp: str | None) -> float | None:
    try:
        return float(calendar.timegm(time.strptime(stamp.rstrip("Z")[:19], "%Y-%m-%dT%H:%M:%S"))) if stamp else None
    except ValueError:
        print(f"stale: a label's time {stamp!r} isn't YYYY-MM-DDTHH:MM:SSZ, so it doesn't count", file=sys.stderr)
        return None


def pull_refs(q: Queue, prs: Iterable[PR]) -> dict[int, str]:
    def refs() -> dict[int, str]:
        out = _git(q.workdir, "for-each-ref", "--format=%(refname:lstrip=3) %(objectname)", "refs/queue/pr/").stdout
        return {int(n): sha for n, sha in (line.split() for line in out.splitlines()) if n.isdigit()}

    have = refs()
    if missing := [p.number for p in prs if have.get(p.number) != p.head]:
        q._remote(q.workdir, "fetch", "-q", "origin", *(f"+refs/pull/{n}/head:refs/queue/pr/{n}" for n in missing), ok=None)
        have = refs()
    return have


def commit_times(repo, shas: Iterable[str]) -> dict[str, float]:
    shas = sorted(set(shas))
    if not shas:
        return {}
    r = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"], cwd=repo, input="\n".join(shas) + "\n",
                       capture_output=True, text=True)
    present = [line.split()[0] for line in r.stdout.splitlines() if line.endswith(" commit")]
    if not present:
        return {}
    out = _git(repo, "log", "--no-walk=unsorted", "--format=%H %ct", *present).stdout
    return {sha: float(ct) for sha, ct in (line.split() for line in out.splitlines())}


def stale_prs(q: Queue, prs: Iterable[PR], now: float, days: int,
              reviewed_at: Callable[[int, str], list[str]] | None = None, keep: Collection[int] = ()) -> dict[int, float]:
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
