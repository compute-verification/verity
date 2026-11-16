"""The Goals page as the merge queue's goal gate reads it (`research.queue.goal_statuses`), with no network: `serve` puts a page
behind the route (`research.goals.Route`) and this machine's copy of it under a test's directory (`research.queue.GOALS_CACHE`)."""
from __future__ import annotations

from pathlib import Path

from research import goals as G
from research import queue as Q
from research.slack import SlackError

#: a page with a goal of each status: each goal's id to its status
PAGE = {"ci/faster-landing": "on-track", "ci/merge-speed": "off-track", "ci/old-landing": "done", "proofs/zk": "on-track"}
#: what the route raises while the page is down
DOWN = f"GET {G.GOALS}: the route is unreachable: a test's page is down"


class Page:
    """The route: ``goals`` (each id to its status) at each read, or `DOWN` while ``goals`` is None; ``reads`` counts the reads."""

    def __init__(self, goals: dict[str, str] | None):
        self.goals, self.reads = goals, 0

    def call(self, body: dict | None = None) -> list[dict]:
        assert body is None, "the goal gate only reads the page"
        self.reads += 1
        if self.goals is None:
            raise SlackError(DOWN)
        return [{"id": k, "status": v, "title": k} for k, v in self.goals.items()]


def serve(monkeypatch, tmp: Path, goals: dict[str, str] | None) -> Page:
    """``goals`` behind the Goals page's route (None: the page is down), and this machine's copy at ``tmp``/cache/goals.json,
    which no read has written yet: the route."""
    page = Page(goals)
    monkeypatch.setattr(G, "Route", lambda: page)
    monkeypatch.setattr(Q, "GOALS_CACHE", tmp / "cache" / "goals.json")
    return page
