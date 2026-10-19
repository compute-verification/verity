"""The deployment's shape: the build's vendored copy is the repository's, the pins Vercel installs are the suite's, and
vercel.json wires the functions, the queue and the crons the code expects."""
from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest
from vercel.queue import Message, MessageMetadata, RetryAfter
from vercel.queue.testing import clear_subscriptions

from conftest import CHAT, LUKE, REPO
from slack_events import consumer, deps, vendor

SUITE = Path(__file__).resolve().parents[1]
SERVICE = SUITE / "service"


def load(monkeypatch, name: str):
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(f"slack_events_api_{name}", SERVICE / "api" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_build_vendors_slack_py_and_the_registry_byte_for_byte(tmp_path):
    spec = importlib.util.spec_from_file_location("slack_events_build", SERVICE / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    assert build.main([str(tmp_path)]) == 0
    assert (tmp_path / "research_slack.py").read_bytes() == (REPO / "tools/research/src/research/slack.py").read_bytes()
    assert (tmp_path / "registry.json").read_bytes() == (REPO / ".agents/skills/using-slack/registry.json").read_bytes()
    assert vendor.slack().__file__ in (str(SERVICE / "_vendor" / "research_slack.py"),
                                       str(REPO / "tools/research/src/research/slack.py"))


def test_the_bundled_registry_passes_research_msgs_own_check():
    reg = vendor.registry()
    assert "ask-the-team" in reg["channels"] and any(h["handle"] == "top" for h in reg["handles"])


def test_requirements_txt_pins_what_the_suite_tests():
    pins = (SERVICE / "requirements.txt").read_text().split()
    project = tomllib.loads((SUITE / "pyproject.toml").read_text())["project"]
    assert pins == project["dependencies"] and all("==" in p for p in pins)


def test_vercel_json_wires_the_functions_the_queue_and_the_crons():
    v = json.loads((SERVICE / "vercel.json").read_text())
    fn = v["functions"]
    assert fn["api/consume.py"]["experimentalTriggers"] == [{"type": "queue/v2beta", "topic": deps.TOPIC}]
    assert fn["api/consume.py"]["maxDuration"] < consumer.LEASE_S       # a lease outlives the delivery that holds it
    assert {c["path"]: c["schedule"] for c in v["crons"]} == {"/api/cron/watchdog": "*/5 * * * *",
                                                             "/api/cron/reconcile": "*/10 * * * *"}
    assert v["framework"] is None and v["buildCommand"] == "python3 build.py"
    assert not (SERVICE / "pyproject.toml").exists()               # Vercel's uv build would read it, and the workspace's
    assert (SERVICE / ".python-version").read_text().strip() == "3.13"


def test_api_index_serves_every_route(monkeypatch):
    app = load(monkeypatch, "index").app
    paths = {r.path for r in app.routes}
    assert {"/api/slack/events", "/api/slack/interactivity", "/api/slack/health", "/api/cron/watchdog",
            "/api/cron/reconcile"} <= paths


def delivery(payload: dict, n: int) -> Message:
    return Message(payload=payload, metadata=MessageMetadata(
        message_id="m1", delivery_count=n, created_at=datetime(2027, 1, 15, tzinfo=timezone.utc), topic=deps.TOPIC,
        consumer_group="slack-events"))


def test_api_consume_runs_the_consumer_and_turns_retry_into_retry_after(world, monkeypatch):
    clear_subscriptions()
    monkeypatch.setattr(deps, "production", lambda: world.deps)
    mod = load(monkeypatch, "consume")
    try:
        post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
        payload = world.event(CHAT, post, "EvQ")
        assert world.deps.store.claim(consumer.event_key("EvQ"), "processing:other", consumer.LEASE_S)
        with pytest.raises(RetryAfter):
            mod.on_message(delivery(payload, 1))
        world.clock.t += consumer.LEASE_S
        assert mod.on_message(delivery(payload, 2)) is None
        assert world.deps.store.get(consumer.event_key("EvQ")) == "done" and len(world.rung("top")) == 1
    finally:
        clear_subscriptions()
