"""The deployment's shape: the build's vendored copy is the repository's, the pins Vercel installs are the suite's, the
queue's subscriber is declared as Vercel documents for Python, and vercel.json wires the functions and crons."""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest
import anyio
from vercel.queue import Message, MessageMetadata, RetryAfter

from conftest import CHAT, LUKE, REPO
from slack_events import consumer, deps, subscriber, vendor

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


def test_the_service_pyproject_pins_what_the_suite_tests_and_declares_the_subscriber():
    """Vercel installs service/pyproject.toml and reads its `[[tool.vercel.subscribers]]` (the vercel-queue README's
    push-mode setup): one entry, `module:function`, naming the `@subscribe` function on the topic the app publishes to."""
    service = tomllib.loads((SERVICE / "pyproject.toml").read_text())
    suite = tomllib.loads((SUITE / "pyproject.toml").read_text())["project"]
    pins = service["project"]["dependencies"]
    assert pins == suite["dependencies"] and all("==" in p for p in pins)
    assert not any(p.startswith(("upstash", "redis")) for p in pins)
    [sub] = service["tool"]["vercel"]["subscribers"]
    assert sub == {"entrypoint": "slack_events.subscriber:on_message"}
    module, func = sub["entrypoint"].split(":")
    fn = getattr(importlib.import_module(module), func)
    assert fn is subscriber.on_message and inspect.iscoroutinefunction(fn.__wrapped__ if hasattr(fn, "__wrapped__") else fn)
    assert not (SERVICE / "requirements.txt").exists()


def test_vercel_json_wires_the_functions_and_the_crons_and_no_queue_trigger():
    v = json.loads((SERVICE / "vercel.json").read_text())
    assert set(v["functions"]) == {"api/index.py"}
    assert "experimentalTriggers" not in json.dumps(v)                  # the subscriber's trigger is generated from pyproject
    assert {c["path"]: c["schedule"] for c in v["crons"]} == {"/api/cron/watchdog": "*/5 * * * *",
                                                             "/api/cron/reconcile": "*/10 * * * *"}
    assert v["framework"] is None and v["buildCommand"] == "python3 build.py"
    assert (SERVICE / ".python-version").read_text().strip() == "3.13"
    assert not (SERVICE / "api" / "consume.py").exists()


def test_api_index_serves_every_route(monkeypatch):
    app = load(monkeypatch, "index").app
    paths = {r.path for r in app.routes}
    assert {"/api/slack/events", "/api/slack/interactivity", "/api/slack/health", "/api/cron/watchdog",
            "/api/cron/reconcile"} <= paths


def delivery(payload: dict, n: int) -> Message:
    return Message(payload=payload, metadata=MessageMetadata(
        message_id="m1", delivery_count=n, created_at=datetime(2027, 1, 15, tzinfo=timezone.utc), topic=deps.TOPIC,
        consumer_group="slack-events"))


def test_the_subscriber_runs_the_consumer_and_turns_retry_into_retry_after(world, monkeypatch):
    monkeypatch.setattr(deps, "production", lambda: world.deps)
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    payload = world.event(CHAT, post, "EvQ")
    world.slack.fail = {"chat.postMessage"}
    with pytest.raises(RetryAfter) as r:
        anyio.run(subscriber.on_message, delivery(payload, 1))
    assert r.value.timeout_seconds == consumer.backoff(1)
    world.slack.fail = set()
    assert anyio.run(subscriber.on_message, delivery(payload, 2)) is None
    assert len(world.rung("top")) == 1
