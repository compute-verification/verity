"""`research digest`: a lane's sync on one screen, every source faked and the clock fixed."""
from __future__ import annotations

import json

from research import digest as D

NOW = 1792300000.0
LANE = "memory-accounting"


def row(pr: int, state: str, head: str = "a" * 40, owner: str = LANE, why: str = "", position: int | None = None) -> dict:
    return {"pr": pr, "head": head, "owner": owner, "state": state, "why": why, "position": position}


class Fake:
    """Sources over plain data; `fail` names the ones that raise."""

    def __init__(self) -> None:
        self.bells: list[dict] = []
        self.replies: dict[str, list[dict]] = {}
        self.main = ("m1", [])
        self.rows: list[dict] = []
        self.states: dict[int, str] = {}
        self.note_files: tuple[str, list[str]] = ("n1", [])
        self.fail: set[str] = set()

    def _check(self, name: str) -> None:
        if name in self.fail:
            raise RuntimeError(f"{name} is down")

    def sources(self) -> D.Sources:
        def bells(since: str) -> list[dict]:
            self._check("doorbells")
            return [m for m in self.bells if float(m["ts"]) > float(since)]

        def thread(channel: str, ts: str, since: str) -> list[dict]:
            return [m for m in self.replies.get(ts, []) if float(m["ts"]) > float(since)]

        def landings(last):
            self._check("landings")
            return self.main

        return D.Sources(doorbells=bells, thread=thread, landings=landings, queue=lambda: self.rows,
                         pr_state=lambda n: self.states.get(n, ""), notes=lambda last: self.note_files, now=lambda: NOW)


def test_a_doorbell_shows_once_and_only_from_the_last_hour_on_a_first_digest():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    f.bells = [{"ts": f"{NOW - 7200:.6f}", "text": "two hours ago"}, {"ts": f"{NOW - 60:.6f}", "text": "a minute ago"}]
    out, errors = D.digest(LANE, s, f.sources())
    assert not errors
    assert "## Doorbells (1)" in out and any("a minute ago" in x for x in out)
    assert not any("two hours ago" in x for x in out)
    out, _ = D.digest(LANE, s, f.sources())
    assert not any(x.startswith("## Doorbells") for x in out)


def test_a_thread_shows_other_handles_replies_but_not_the_lanes_own_nor_a_reply_its_doorbell_showed():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    s["threads"] = {"pairing": ["agent-coordination", "1.0"]}
    f.bells = [{"ts": f"{NOW - 10:.6f}", "text": f"from ci, thread 1.0, reply {NOW - 20:.6f}: @memory-accounting queue"}]
    f.replies["1.0"] = [
        {"ts": f"{NOW - 30:.6f}", "username": LANE, "text": "our own status"},
        {"ts": f"{NOW - 20:.6f}", "username": "ci", "text": "already rung"},
        {"ts": f"{NOW - 15:.6f}", "username": "compute-accounting", "text": "new from them"},
    ]
    out, _ = D.digest(LANE, s, f.sources())
    threads = out[out.index("## Threads") + 1:]
    assert len([x for x in threads if x.startswith("- pairing")]) == 1
    assert "new from them" in threads[0]


def test_landings_name_the_queue_merges_and_a_lane_pr_that_left_the_queue_says_whether_it_landed():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    f.rows = [row(10, "queued", position=3), row(11, "queued"), row(12, "waiting", owner="ci")]
    D.digest(LANE, s, f.sources())
    assert set(s["queue"]) == {"10", "11"}
    f.main = ("m2", ["Queue: #10 the first", "a direct commit", "Queue: #12 another lane's"])
    f.rows, f.states = [], {11: "CLOSED"}
    out, _ = D.digest(LANE, s, f.sources())
    assert "## Landed on main (2)" in out
    assert "- #10 landed" in out and "- #11 left the queue (CLOSED)" in out
    assert s["main"] == "m2" and s["queue"] == {}


def test_a_watched_pr_triggers_when_it_lands_or_its_state_changes_but_not_when_first_seen():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    s["watch"] = {"20": "merge main into #21", "30": "re-mark #31"}
    f.rows = [row(20, "queued", owner="ci"), row(30, "queued", owner="ci")]
    out, _ = D.digest(LANE, s, f.sources())
    assert not any("TRIGGER" in x for x in out)
    f.rows = [row(30, "refused", owner="ci", why="conflicts with next")]
    f.main = ("m2", ["Queue: #20 ci's change"])
    out, _ = D.digest(LANE, s, f.sources())
    assert "- TRIGGER #20 landed: merge main into #21" in out
    assert any(x.startswith("- TRIGGER #30 is now refused") and "re-mark #31" in x for x in out)


def test_a_failing_source_is_reported_keeps_its_cursor_and_hides_nothing_else():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    s["doorbell"] = f"{NOW - 100:.6f}"
    f.fail = {"doorbells"}
    f.rows = [row(10, "queued")]
    f.note_files = ("n2", ["lanes/memory-accounting/20261018T0500Z-report-x.md"])
    out, errors = D.digest(LANE, s, f.sources())
    assert errors == ["doorbells: doorbells is down"]
    assert s["doorbell"] == f"{NOW - 100:.6f}"
    assert "## Our PRs: 1 queued" in out and "- 20261018T0500Z-report-x.md" in out


def test_the_watch_list_and_threads_are_edited_in_the_state_file(tmp_path, capsys):
    p = tmp_path / "d.json"
    assert D.main(["--as", f"@{LANE}", "--state", str(p), "--watch", "#2446", "merge main into #2421"]) == 0
    assert D.main(["--as", LANE, "--state", str(p), "--thread", "cards", "9.5", "--channel", "ask-daniel"]) == 0
    s = json.loads(p.read_text())
    assert s["watch"] == {"2446": "merge main into #2421"} and s["threads"] == {"cards": ["ask-daniel", "9.5"]}
    assert D.main(["--as", LANE, "--state", str(p), "--unwatch", "2446", "--drop-thread", "cards"]) == 0
    s = json.loads(p.read_text())
    assert s["watch"] == {} and s["threads"] == {}
    capsys.readouterr()
