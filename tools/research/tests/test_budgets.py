"""budgets.toml (pods/budgets.py): what a line is, which line covers a pod, reading it from the notes origin (a pushed line counts,
a local edit does not), and the gate `research pods create` passes against the guard's own state."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research.pods import budgets as B

T0 = B.parse_time("2026-09-23T20:00:00Z")
GOOD = """
[guard]
project = ["vy"]
balance_floor = 25
exempt = ["vy-control-"]

[budgets]
"vy-a-" = { cap_usd = 3, expires = "2026-09-24T20:00Z", max_pod_hours = 4, by = "daniel 2026-09-23" }
"vy-a-big-" = { cap_usd = 100, expires = 2026-09-24T20:00:00Z, max_pod_hours = 48 }
"""


def test_a_file_parses_into_lines_and_guard_settings():
    b = B.parse(GOOD)
    assert b.project == ("vy",) and b.exempt == ("vy-control-",) and b.balance_floor == 25 and b.terminate_uncovered and b.uncovered_grace_min == 15
    a = b.lines["vy-a-"]
    assert (a.cap_usd, a.max_pod_hours, a.by, a.expires) == (3.0, 4.0, "daniel 2026-09-23", T0 + 86400)
    assert b.lines["vy-a-big-"].expires == T0 + 86400, "a TOML datetime and an ISO string are the same expiry"
    assert a.live(T0) and not a.live(T0 + 86400)


def test_the_longest_prefix_covers_and_other_names_are_not_the_projects():
    b = B.parse(GOOD)
    assert b.covering("vy-a-big-1-veritor-campaign").prefix == "vy-a-big-"
    assert b.covering("vy-a-1-veritor-campaign").prefix == "vy-a-"
    assert b.covering("vy-b-1") is None and b.is_project("vy-b-1") and not b.is_project("ligerito-h100")
    assert b.is_exempt("vy-control-verity")


@pytest.mark.parametrize("text, problem", [
    ('[budgets]\n"vy-a-" = { cap = 3, expires = "2026-09-24T20:00Z", max_pod_hours = 4 }', "cap not a line key"),
    ('[budgets]\n"vy-a-" = { cap_usd = 3, max_pod_hours = 4 }', "missing expires"),
    ('[budgets]\n"vy-a-" = { cap_usd = 3, expires = "2026-09-24T20:00", max_pod_hours = 4 }', "not ISO 8601 with a zone"),
    ('[budgets]\n"vy-a-" = { cap_usd = -1, expires = "2026-09-24T20:00Z", max_pod_hours = 4 }', "cap_usd is not a number"),
    ('[budgets]\n"vy-a-" = { cap_usd = 3, expires = "2026-09-24T20:00Z", max_pod_hours = 0 }', "max_pod_hours is not a positive"),
    ('[guard]\nprojects = ["vy"]', "'projects' is not a setting"),
    ('[guard]\nterminate_uncovered = "yes"', "terminate_uncovered is not true or false"),
    ('[budget]\n', "unknown table or key 'budget'"),
    ('[budgets\n', "does not parse"),
])
def test_a_file_with_any_bad_line_is_rejected_whole(text, problem):
    with pytest.raises(B.BudgetError, match=problem):
        B.parse(GOOD.split("[budgets]")[0] + "\n" + text if text.startswith("[budgets]") else text)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def notes(tmp_path):
    """A notes clone with an origin; git's identity is the test's."""
    origin, clone = tmp_path / "origin.git", tmp_path / "notes"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True, capture_output=True)
    for k, v in (("user.name", "t"), ("user.email", "t@t"), ("commit.gpgsign", "false")):
        _git(clone, "config", k, v)
    _git(clone, "checkout", "-q", "-b", "main")
    (clone / "budgets.toml").write_text(GOOD)
    _git(clone, "add", "budgets.toml")
    _git(clone, "commit", "-q", "-m", "budgets")
    _git(clone, "push", "-q", "origin", "main")
    return clone


def test_a_notes_clone_is_read_from_its_origin_so_only_a_pushed_line_counts(notes):
    b, where = B.load(notes)
    assert set(b.lines) == {"vy-a-", "vy-a-big-"} and "origin/main" in where
    (notes / "budgets.toml").write_text(GOOD + '"vy-new-" = { cap_usd = 1, expires = "2026-09-24T20:00Z", max_pod_hours = 1 }\n')
    assert "vy-new-" not in B.load(notes)[0].lines, "committed or not, a line nobody pushed is not a line"
    _git(notes, "commit", "-qam", "vy-new-")
    assert "vy-new-" not in B.load(notes)[0].lines
    _git(notes, "push", "-q", "origin", "main")
    assert "vy-new-" in B.load(notes)[0].lines
    assert B.load(notes / "budgets.toml")[1] == str(notes / "budgets.toml"), "a file path is read as it is"


# ---- the create gate --------------------------------------------------------------------------------------------------------------------

def guard_state(**lines_state) -> dict:
    b = B.parse(GOOD)
    return {"last_ok_poll": T0, "budgets": {**b.view(), "source": "notes origin/main abc"},
            "lines": {p: {"spent": 0.0, "tripped": None, **lines_state.get(p.strip("-").replace("-", "_"), {})} for p in b.lines}}


def test_the_gate_lets_a_covered_name_under_a_fresh_guard():
    assert B.gate("vy-a-1-veritor-campaign", 4, guard_state(), T0 + 60) is None


def test_the_gate_refuses_while_the_guard_is_down():
    why = B.gate("vy-a-1-veritor-campaign", 1, guard_state(), T0 + B.STALE_S + 1, host="vy-control-verity")
    assert "last completed a poll 3.0 min ago" in why and "nothing new starts unguarded" in why
    assert "never" in B.gate("vy-a-1", 1, {**guard_state(), "last_ok_poll": None}, T0)


def test_the_gate_refuses_while_the_guard_cannot_refresh_its_budgets():
    state = guard_state()
    state["budgets"] |= {"read_utc": "2026-10-09T08:12:43Z", "error": "OSError: git symbolic-ref --short HEAD: fatal: ref HEAD is not a symbolic ref"}
    why = B.gate("vy-a-1-veritor-campaign", 1, state, T0 + 60, host="vy-control-verity")
    assert "cannot read budgets.toml" in why and "2026-10-09T08:12:43Z" in why and "HEAD is on its branch" in why


def test_the_gate_refuses_a_name_no_line_covers_and_says_how_to_add_one():
    why = B.gate("vy-b-1-veritor-campaign", 1, guard_state(), T0)
    assert "no budget line covers 'vy-b-1-veritor-campaign'" in why and '"vy-b-1-" = { cap_usd = 5' in why and "push it" in why


def test_the_gate_refuses_an_expired_or_spent_line_and_a_lease_past_its_max_age():
    assert "expired at 2026-09-24T20:00:00Z" in B.gate("vy-a-1", 1, {**guard_state(), "last_ok_poll": T0 + 86400}, T0 + 86400)
    assert "at its cap ($3.00 of $3)" in B.gate("vy-a-1", 1, guard_state(vy_a={"spent": 3.0}), T0)
    assert "at its cap" in B.gate("vy-a-1", 1, guard_state(vy_a={"spent": 1.0, "tripped": {"at": "t"}}), T0)
    assert "--max-hours 5 is over the line 'vy-a-''s max_pod_hours 4" in B.gate("vy-a-1", 5, guard_state(), T0)
    assert B.gate("vy-a-big-1", 5, guard_state(vy_a={"spent": 3.0}), T0) is None, "the longest prefix is the one judged"


def test_check_create_fails_closed_when_the_guards_state_cannot_be_read():
    def unreachable():
        raise OSError("ssh: connect to host 1.2.3.4 port 22: Connection refused")

    why, line = B.check_create("vy-a-1", 1, reader=unreachable)
    assert "cannot read the budgets guard's state" in why and "nothing new starts unguarded" in why and line is None
    assert B.check_create("vy-a-1", 1, reader=lambda: (guard_state(), T0)) == (None, "vy-a-")


def test_read_guard_state_reads_a_local_file(tmp_path, monkeypatch):
    import json

    f = tmp_path / "state.json"
    f.write_text(json.dumps(guard_state()))
    monkeypatch.setenv("RESEARCH_GUARD_HOST", "local")
    monkeypatch.setenv("RESEARCH_GUARD_STATE", str(f))
    state, _clock = B.read_guard_state()
    assert state["budgets"]["lines"]["vy-a-"]["cap_usd"] == 3.0
