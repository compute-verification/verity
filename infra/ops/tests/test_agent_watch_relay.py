"""Infra's relay posts each line the agent watch adds to events.jsonl on the control pod, once, from the offset it last posted."""
import importlib.util
import json
from pathlib import Path

import pytest

RELAY = Path(__file__).resolve().parents[2] / "control" / "agent_watch_relay.py"


@pytest.fixture
def relay(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("agent_watch_relay", RELAY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    log, posts = {"text": ""}, []
    monkeypatch.setattr(m, "OFFSET", tmp_path / "offset")
    monkeypatch.setattr(m, "DRY", False)
    monkeypatch.setattr(m, "pod", lambda cmd: log["text"])
    monkeypatch.setattr(m.slack, "_post", lambda api, ch, handle, text, *a, **k: posts.append(text) or "1.0")
    m.log_text, m.posts = log, posts
    return m


def line(run: str) -> str:
    return json.dumps({"kind": "held", "agent": "bc-x", "name": "x", "run": run, "since": "2026-10-18T20:00:00Z", "minutes": 15,
                       "action": "cancelled and continued"})


def test_a_line_still_being_written_waits_for_the_next_pass(relay):
    relay.log_text["text"] = line("run-1") + "\n" + line("run-2")[:30]
    relay.once()
    relay.log_text["text"] = line("run-1") + "\n" + line("run-2") + "\n"
    relay.once()
    assert [p.count("run-1") + 2 * p.count("run-2") for p in relay.posts] == [1, 2], "each posted once, whole"
    assert relay.OFFSET.read_text() == "2"


def test_a_log_replaced_with_a_shorter_one_is_posted_from_the_top(relay):
    relay.OFFSET.write_text("5")
    relay.log_text["text"] = line("run-a") + "\n" + line("run-b") + "\n"
    relay.once()
    assert len(relay.posts) == 2 and relay.OFFSET.read_text() == "2"


def test_a_refused_post_is_posted_again_next_pass_and_the_relay_goes_on(relay, monkeypatch):
    relay.log_text["text"] = line("run-1") + "\n" + line("run-2") + "\n"

    def refuse_second(api, ch, handle, text, *a, **k):
        if "run-2" in text:
            raise relay.slack.SlackError("ratelimited")
        relay.posts.append(text)
    monkeypatch.setattr(relay.slack, "_post", refuse_second)
    monkeypatch.setattr(relay.sys, "argv", ["agent_watch_relay.py", "--once"])
    assert relay.main() == 0
    assert relay.OFFSET.read_text() == "1"
