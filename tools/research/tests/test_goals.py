"""`research goals` against a fake route: set adds a goal or changes only the fields given, confirm and done write it
back, replace sends exactly the file's goals, show prints them by due date, and a refusal comes back as one line."""
from __future__ import annotations

import json

import pytest

from research import goals
from research.slack import RouteError

GOAL = {"id": "demos/both", "title": "Both demos pass on one served H200 run", "status": "off-track",
        "due": "2026-11-05T10:00-08:00", "tags": ["remote-demo"], "owner": "vllm", "headline": True,
        "description": "The verdicts come back.", "waitingOn": [{"goal": "demos/pouw"}]}


class Fake:
    """The route: the page's goals, and every body posted to it."""

    def __init__(self, page: list[dict] | None = None, refuse: str | None = None):
        self.page = [dict(g, updatedAt="2026-11-02T04:00:00Z", updatedBy="bc-top") for g in page or []]
        self.posted: list[dict] = []
        self.refuse = refuse

    def call(self, body: dict | None = None) -> list[dict]:
        if body is None:
            return self.page
        if self.refuse:
            raise RouteError(f"POST {goals.GOALS}: refused: {self.refuse}", 400)
        self.posted.append(body)
        if "set" in body:
            self.page = [g for g in self.page if g["id"] != body["set"]["id"]] + [body["set"]]
        return self.page


def test_set_changes_only_the_fields_given_and_drops_what_the_store_added(capsys):
    fake = Fake([GOAL])
    assert goals.main(["set", "demos/both", "--status", "on-track", "--no-wait"], fake) == 0
    sent = fake.posted[-1]["set"]
    assert sent == {k: v for k, v in GOAL.items() if k != "waitingOn"} | {"status": "on-track"}
    assert "Thu 5 Nov, 10:00 AM" in capsys.readouterr().out


def test_wait_replaces_the_links_tier_overrides_the_derived_one_and_blockers_say_why():
    fake = Fake([GOAL])
    assert goals.main(["set", "demos/both", "--wait", "goal:vllm/partition", "--wait",
                       "ask:dbd7c37c-12c8-4a1f-b279-1c99cb922c39", "--tier", "2", "--owner", "proofs",
                       "--blockers", "Main isn't moving."], fake) == 0
    sent = fake.posted[-1]["set"]
    assert sent["waitingOn"] == [{"goal": "vllm/partition"}, {"ask": "dbd7c37c-12c8-4a1f-b279-1c99cb922c39"}]
    assert sent["tier"] == 2 and sent["owner"] == "proofs" and sent["blockers"] == "Main isn't moving."
    fake = Fake([GOAL | {"tier": 2, "blockers": "x"}])
    assert goals.main(["set", "demos/both", "--no-tier", "--no-blockers"], fake) == 0
    assert "tier" not in fake.posted[-1]["set"] and "blockers" not in fake.posted[-1]["set"]
    assert goals.main(["set", "demos/both", "--wait", "partition"], fake) == 1


def test_a_new_goal_needs_title_status_and_due(capsys, tmp_path):
    fake = Fake()
    assert goals.main(["set", "infra/x", "--title", "Verity holds no infra"], fake) == 1
    assert "needs --status, --due" in capsys.readouterr().err
    (tmp_path / "d.md").write_text("Every infra path is gone.")
    assert goals.main(["set", "infra/x", "--title", "Verity holds no infra", "--status", "on-track", "--due",
                       "2026-11-06", "--tag", "infra", "--description", f"@{tmp_path / 'd.md'}"], fake) == 0
    assert fake.posted[-1]["set"] == {"id": "infra/x", "title": "Verity holds no infra", "status": "on-track",
                                      "due": "2026-11-06", "tags": ["infra"], "description": "Every infra path is gone."}


def test_confirm_writes_it_back_unchanged_and_done_marks_it_done():
    fake = Fake([GOAL])
    assert goals.main(["confirm", "demos/both"], fake) == 0
    assert fake.posted[-1] == {"set": GOAL}
    assert goals.main(["done", "demos/both"], fake) == 0
    assert fake.posted[-1]["set"]["status"] == "done"
    assert goals.main(["done", "nope"], fake) == 1


def test_replace_sends_exactly_the_files_goals(tmp_path):
    fake = Fake()
    path = tmp_path / "goals.json"
    path.write_text(json.dumps([GOAL | {"updatedBy": "x"}]))
    assert goals.main(["replace", str(path)], fake) == 0
    assert fake.posted == [{"replace": [GOAL]}]
    path.write_text(json.dumps({"goals": []}))
    assert goals.main(["replace", str(path)], fake) == 1


def test_show_lists_every_goal_by_due_date_with_links_moves_and_tier_overrides(capsys):
    subgoal = {"id": "vllm/partition", "title": "Partition on main", "status": "on-track", "due": "2026-11-02",
               "tags": ["remote-demo"], "tier": 2}
    fake = Fake([GOAL | {"wasDue": "2026-11-03"}, subgoal])
    assert goals.main(["show"], fake) == 0
    out = capsys.readouterr().out
    assert "Thu 5 Nov, 10:00 AM" in out and "off-track" in out and "waits on demos/pouw" in out
    assert "(was Tue 3 Nov, end of day)" in out and "demos/both" in out
    assert "Partition on main  remote-demo  (tier 2)" in out


def test_a_refusal_is_one_line_naming_the_field(capsys):
    fake = Fake([GOAL], refuse="due must be a date (2026-11-05) or a time with its offset (2026-11-05T10:00-08:00)")
    assert goals.main(["set", "demos/both", "--due", "Thursday"], fake) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "due must be a date" in err[0]


def test_a_route_error_other_than_400_is_one_line_with_its_code(monkeypatch):
    import io
    import urllib.error

    def refuse(*a, **k):
        raise urllib.error.HTTPError(goals.GOALS, 403, "Forbidden", {}, io.BytesIO(b'{"error": "identity_refused"}'))

    route = goals.Route()
    monkeypatch.setattr(route._oidc, "send", lambda fn: fn("t"))
    monkeypatch.setattr(goals, "http_json", refuse)
    with pytest.raises(RouteError) as e:
        route.call()
    assert e.value.status == 403 and "HTTP 403 identity_refused" in str(e.value)


def test_a_site_key_reads_the_page_without_an_identity_and_writes_nothing(monkeypatch):
    import io
    import urllib.error

    sent = []

    def answer(method, url, data, headers):
        sent.append((method, headers["Authorization"]))
        return {"goals": [GOAL]}

    monkeypatch.setattr(goals, "http_json", answer)
    monkeypatch.setattr(goals._Identity, "send", lambda *a: pytest.fail("a read with a site key asks for no identity"))
    route = goals.Route(key="a-dummy-site-key")
    assert route.call() == [GOAL] and sent == [("GET", "Bearer a-dummy-site-key")]
    with pytest.raises(goals.SlackError, match="a site key only reads"):
        route.call({"remove": "demos/both"})
    assert len(sent) == 1

    def refuse(*a, **k):
        raise urllib.error.HTTPError(goals.GOALS, 401, "Unauthorized", {}, io.BytesIO(b'{"error": "invalid_key"}'))

    monkeypatch.setattr(goals, "http_json", refuse)
    with pytest.raises(RouteError, match="rejected the site key") as e:
        route.call()
    assert e.value.status == 401


@pytest.mark.parametrize("argv", [["set", "x", "--status", "maybe"], ["set", "x", "--wait", "goal:a", "--no-wait"],
                                  ["set", "x", "--tier", "2", "--no-tier"], ["set", "x", "--tier", "12"]])
def test_the_command_line_refuses_what_the_site_would(argv):
    with pytest.raises(SystemExit):
        goals.main(argv, Fake())


def test_priority_sets_or_clears_a_goal_s_priority_as_its_own_write():
    fake = Fake([GOAL])
    assert goals.main(["priority", "demos/both", "1"], fake) == 0
    assert fake.posted[-1] == {"priority": {"id": "demos/both", "priority": 1}}
    assert goals.main(["priority", "demos/both", "clear"], fake) == 0
    assert fake.posted[-1] == {"priority": {"id": "demos/both", "priority": None}}
    stored = GOAL | {"priority": 1, "prioritySet": True, "prioritySetBy": "top", "prioritySetAt": "2026-11-04T21:40:00.000Z"}
    fake = Fake([stored])
    assert goals.main(["set", "demos/both", "--title", "Reworded"], fake) == 0
    assert not {"priority", "prioritySet", "prioritySetBy", "prioritySetAt"} & set(fake.posted[-1]["set"]), "a goal write never sends priority back"
