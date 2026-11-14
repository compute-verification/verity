from __future__ import annotations

import json
from datetime import UTC, datetime

from research import linear, response

REG = {"handles": [{"handle": h} for h in ("top", "ci", "infra", "lean", "vllm", "proofs", "soundness", "kernels",
                                           "dashboard-goals")]}
HANDLES = {h["handle"] for h in REG["handles"]}
T0 = datetime(2026, 11, 14, 5, 0, tzinfo=UTC).timestamp()
APP = {"LINEAR_CLIENT_ID": "dummy-client", "LINEAR_CLIENT_SECRET": "dummy-secret"}


def at(minutes: float) -> str:
    return datetime.fromtimestamp(T0 + minutes * 60, UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def app(minutes: float, issue: str, by: str, body: str, cid: str = "") -> dict:
    """A comment through `linear comment --as BY` (or `--worker`): the app's createAsUser name, no user."""
    return {"id": cid or f"{issue}@{minutes}", "createdAt": at(minutes), "body": body, "user": None,
            "botActor": {"userDisplayName": by}, "issue": {"identifier": issue, "url": f"https://linear.app/x/issue/{issue}"}}


def person(minutes: float, issue: str, body: str, name: str = "Daniel Reuter") -> dict:
    return {"id": f"{issue}@{minutes}", "createdAt": at(minutes), "body": body, "user": {"name": name}, "botActor": None,
            "issue": {"identifier": issue, "url": f"https://linear.app/x/issue/{issue}"}}


def test_the_writer_is_the_apps_as_name_else_the_handle_a_persons_key_signs_with_else_unknown():
    """Every lane writes through one app, so Linear's user is the same for all: the lane is the createAsUser name. A
    comment under a person's key is the handle its body signs with after its leading names, and else nobody's."""
    w = lambda c: response.writer(c, HANDLES)[0]  # noqa: E731
    assert w(app(0, "CI-1", "vllm", "@top @ci vllm (lead), 10:37 PM PST. **Done.**")) == "vllm"
    assert w(app(0, "CI-1", "vllm/integrator", "@ci a note")) == "vllm/integrator"
    assert w(person(0, "CI-1", "@top @ci dashboard-goals, Fri 13 Nov 9:32 PM PST. **ci is the only lead left**")) == "dashboard-goals"
    assert w(person(0, "CI-1", "RULING. vllm (lead), Fri 13 Nov 5:04 AM PST (13:04Z), to the mark")) == "vllm"
    assert w(person(0, "CI-1", "top, Fri 13 Nov 10:40 PM PST. Read.")) == "top"
    assert response.writer(person(0, "CI-1", "@lean: @ci the body"), HANDLES) == ("lean", "@ci the body")  # a personal key's
    assert w(person(0, "CI-1", "Please look at this, thanks")) == response.UNKNOWN
    assert w(person(0, "CI-1", "stranger (lead), 10:00 PM PST. not a handle")) == response.UNKNOWN


def comments() -> list[dict]:
    return [
        app(0, "CI-1", "infra", "@ci @top infra, Fri 13 Nov 9:00 PM PST. **One ask.**"),             # 2 asks
        app(5, "CI-1", "top", "@infra top, 9:05 PM PST. Read."),                                      # answers top's; rings infra
        app(20, "CI-1", "ci/WORKER", "@infra a worker's note"),                                       # answers ci's; rings no one
        app(30, "LEAN-2", "lean", "@lean @proofs lean (lead), 4:00 AM PST (12:00Z), says another time"),  # lean's own name: no ask
        app(90, "LEAN-2", "proofs", "proofs (lead). Answered."),                                      # 60 min after
        app(40, "LEAN-3", "lean", "@top @ci @infra @vllm @proofs lean, an announcement to five"),     # 5 wide asks
        app(41, "LEAN-3", "ci", "ci, read."),
        app(45, "VLLM-4", "vllm", "@soundness vllm, an ask nobody answers here"),
        app(50, "CI-5", "soundness", "soundness, answered on another issue"),                         # not VLLM-4
        person(60, "INF-6", "@kernels Please look at this"),                                         # unknown asker
        person(61, "INF-6", "kernels (lead), 9:01 PM PST. Looked."),                                  # kernels, by signature
    ]


def test_each_handle_a_ring_names_is_an_ask_answered_by_its_lanes_next_comment_on_the_issue():
    now = T0 + 120 * 60
    found, left = response.asks(comments(), REG, now)
    got = [(a["issue"], a["by"], a["to"], a["minutes"], a["open"], a["wide"]) for a in found]
    assert got == [
        ("CI-1", "infra", "ci", 20.0, False, False),           # by ci's worker: the lane answers
        ("CI-1", "infra", "top", 5.0, False, False),           # @top counts like every other handle
        ("CI-1", "top", "infra", 115.0, True, False),          # open at its age, from Linear's clock
        ("LEAN-2", "lean", "proofs", 60.0, False, False),      # the text's 12:00Z is not the clock; lean's own name is no ask
        ("LEAN-3", "lean", "top", 80.0, True, True),
        ("LEAN-3", "lean", "ci", 1.0, False, True),
        ("LEAN-3", "lean", "infra", 80.0, True, True),
        ("LEAN-3", "lean", "vllm", 80.0, True, True),
        ("LEAN-3", "lean", "proofs", 80.0, True, True),
        ("VLLM-4", "vllm", "soundness", 75.0, True, False),    # an answer on another issue answers nothing
        ("INF-6", "unknown", "kernels", 1.0, False, False),
    ]
    assert left == {"unknown": {"Daniel Reuter": 1}, "worker_rings": 1}
    assert found[0]["rung"] == "2026-11-14T05:00:00Z" and found[0]["answered"] == "2026-11-14T05:20:00Z"
    assert [a["to"] for a in response.asks(comments(), REG, now, since=T0 + 40 * 60)[0]][:2] == ["top", "ci"]


def test_the_figures_count_an_open_ask_at_its_age_and_split_narrow_rings_from_wide():
    assert response.percentile([], 90) is None
    assert response.percentile([5, 1, 3, 2, 4], 50) == 3 and response.percentile(list(range(1, 11)), 90) == 9
    now = T0 + 120 * 60
    found, left = response.asks(comments(), REG, now)
    rep = response.report(found, left, T0, now, open_over=30)
    assert rep["all"]["narrow"] == {"received": 6, "answered": 4, "open": 2, "median": 20.0, "p90": 115.0}
    assert rep["all"]["wide"] == {"received": 5, "answered": 1, "open": 4, "median": 80.0, "p90": 80.0}
    assert rep["lanes"]["infra"]["narrow"]["open"] == 1 and rep["lanes"]["infra"]["wide"]["received"] == 1
    assert [(a["issue"], a["to"]) for a in rep["open_over"]][:2] == [("CI-1", "infra"), ("LEAN-3", "top")]
    assert set(response.report(found, left, T0, now, 30, lane="top")["lanes"]) == {"top"}
    json.dumps(rep)


def test_the_table_reads_pacific_with_its_label():
    now = T0 + 120 * 60
    found, left = response.asks(comments(), REG, now)
    text = response.render(response.report(found, left, T0, now, open_over=30))
    assert "from Fri 13 Nov 9:00 PM PST (05:00Z) to Fri 13 Nov 11:00 PM PST (07:00Z)" in text
    assert "all lanes                6        4     2      20    115" in text
    assert "CI-1       @infra              rung by top Fri 13 Nov 9:05 PM PST (05:05Z), open 115 min" in text
    assert "1 comments under Daniel Reuter" in text and "not counted: 1 workers' comments" in text


def test_comments_pages_through_every_comment_since_the_windows_start(tmp_path):
    pages = [{"nodes": [app(0, "CI-1", "top", "a")], "pageInfo": {"hasNextPage": True, "endCursor": "c1"}},
             {"nodes": [app(1, "CI-1", "ci", "b")], "pageInfo": {"hasNextPage": False, "endCursor": None}}]
    sent = []

    def transport(url: str, data: bytes, headers: dict) -> dict:
        if url == linear.TOKEN_URL:
            return {"access_token": "dummy-token", "expires_in": 3600}
        sent.append(json.loads(data)["variables"])
        return {"data": {"comments": pages[len(sent) - 1]}}

    lin = linear.Linear(env=APP, transport=transport, now=lambda: T0, cache=tmp_path / "token.json")
    got = lin.comments("2026-11-14T05:00:00Z")
    assert [c["body"] for c in got] == ["a", "b"]
    assert sent == [{"f": {"createdAt": {"gt": "2026-11-14T05:00:00Z"}}, "after": None},
                    {"f": {"createdAt": {"gt": "2026-11-14T05:00:00Z"}}, "after": "c1"}]
