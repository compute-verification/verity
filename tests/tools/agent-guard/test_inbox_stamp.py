"""`inbox.py send` refuses a message whose first line is stamped ahead of the clock, as `research msg` does (research's
`timefmt.stamp_ahead`): the lean lane's inbox lines ran minutes ahead while the rule was kept by hand."""
from __future__ import annotations

import datetime as dt
import importlib
from pathlib import Path

import pytest

GUARD = (Path(__file__).resolve().parents[3] / "tools" / "agent-guard")
AGENT = "bc-77777777-7777-4777-8777-777777777777"
NOW = dt.datetime(2026, 11, 15, 9, 55, 20, tzinfo=dt.UTC)                              # 1:55 AM PST


@pytest.fixture
def inbox(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(GUARD))
    mod = importlib.import_module("inbox")
    monkeypatch.setattr(mod.send, "__defaults__", (tmp_path / "box",))          # `send`'s inbox, bound when it was defined
    return mod


def test_send_refuses_a_stamp_ahead_of_the_clock_and_queues_nothing(inbox, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        inbox.main(["send", AGENT, "--local", "--text", "lean (lead), 2:05 AM PST: continue"], now=lambda: NOW)
    assert str(e.value) == ("inbox: the first line says 2:05 AM PST and it is 1:55 AM PST: write the time "
                            "`TZ=America/Los_Angeles date` gives now, leave the time out, or, if it is a time to come, put "
                            "`by`, `at` or `ETA` before it")
    assert not (tmp_path / "box").exists()
    body = tmp_path / "m.md"
    body.write_text("lean (lead), 1:55 AM PST: continue")
    assert inbox.main(["send", AGENT, "--local", "--file", str(body)], now=lambda: NOW) == 0
    assert "lean (lead), 1:55 AM PST: continue" in (tmp_path / "box" / f"{AGENT}.md").read_text()
