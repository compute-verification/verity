"""The deployment's shape: the build's vendored copy is the repository's, the pins Vercel installs are the suite's, the
queue is consumed by polling (the `drain` cron), and vercel.json wires the function and the crons."""
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


def test_the_service_pyproject_pins_what_the_suite_tests_and_declares_no_push_subscriber():
    """Vercel installs service/pyproject.toml.  It declares no `[[tool.vercel.subscribers]]`: this build never compiles
    one, so nothing would consume the topic through it, and if a later build did, a second consumer group would route
    every post twice beside the drain's."""
    service = tomllib.loads((SERVICE / "pyproject.toml").read_text())
    suite = tomllib.loads((SUITE / "pyproject.toml").read_text())["project"]
    pins = service["project"]["dependencies"]
    assert pins == suite["dependencies"] and all("==" in p for p in pins)
    assert not any(p.startswith(("upstash", "redis")) for p in pins)
    assert "tool" not in service
    assert not (SERVICE / "requirements.txt").exists()


def test_vercel_json_wires_the_function_and_the_crons_with_the_drain_every_minute_and_no_queue_trigger():
    v = json.loads((SERVICE / "vercel.json").read_text())
    assert set(v["functions"]) == {"api/index.py"}
    assert "experimentalTriggers" not in json.dumps(v)                  # the drain polls; nothing is pushed
    assert {c["path"]: c["schedule"] for c in v["crons"]} == {"/api/cron/drain": "* * * * *",
                                                             "/api/cron/watchdog": "*/5 * * * *",
                                                             "/api/cron/reconcile": "*/10 * * * *"}
    assert v["framework"] is None and v["buildCommand"] == "python3 build.py && mkdir -p public && echo slack-events > public/index.txt"   # a custom build with no framework needs a non-empty output dir
    assert v["ignoreCommand"] == "python3 ignore.py"
    assert (SERVICE / ".python-version").read_text().strip() == "3.13"
    assert not (SERVICE / "api" / "consume.py").exists()


def test_api_index_serves_every_route(monkeypatch):
    app = load(monkeypatch, "index").app
    paths = {r.path for r in app.routes}
    assert {"/api/slack/events", "/api/slack/interactivity", "/api/slack/health", "/api/cron/watchdog",
            "/api/cron/reconcile", "/api/cron/drain"} <= paths


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


class FakeDelivery:
    """The SDK's Delivery as `drain` uses it: entering gives the message; a clean exit acknowledges it, and RetryAfter
    schedules it again and is swallowed."""

    def __init__(self, message: Message, log: list):
        self.message, self.log = message, log

    async def __aenter__(self) -> Message:
        return self.message

    async def __aexit__(self, typ, exc, tb):
        if exc is None:
            self.log.append(("ack", self.message.message_id))
        elif isinstance(exc, RetryAfter):
            self.log.append(("retry", self.message.message_id, exc.timeout_seconds))
            return True
        return None


def test_the_drain_handles_each_claimed_message_until_the_topic_is_empty(world, monkeypatch):
    monkeypatch.setattr(deps, "production", lambda: world.deps)
    posts = [world.slack.add(CHAT, text=f"question {i}", user=LUKE) for i in range(3)]
    batches, log, asked = [[delivery(world.event(CHAT, p, f"Ev{i}"), 1) for i, p in enumerate(posts[:2])],
                           [delivery(world.event(CHAT, posts[2], "Ev2"), 1)], []], [], []

    async def receive(topic, group, *, limit, lease_duration):
        asked.append((topic, group, limit, lease_duration))
        for m in batches.pop(0):
            yield FakeDelivery(m, log)

    assert anyio.run(subscriber.drain, world.deps, receive) == {"handled": 3, "failed": 0}
    assert [a[:2] for a in asked] == [(deps.TOPIC, subscriber.CONSUMER_GROUP)] * 3 and batches == []
    assert [x[0] for x in log] == ["ack"] * 3 and len(world.rung("top")) == 3


def test_a_retry_is_rescheduled_not_acknowledged_and_the_drain_stops_at_its_budget(world, monkeypatch):
    monkeypatch.setattr(deps, "production", lambda: world.deps)
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    world.slack.fail = {"chat.postMessage"}
    log, ticks = [], iter([0.0, 1.0, subscriber.BUDGET_S + 1])

    async def receive(topic, group, *, limit, lease_duration):
        yield FakeDelivery(delivery(world.event(CHAT, post, "EvR"), 1), log)

    assert anyio.run(lambda: subscriber.drain(world.deps, receive, lambda: next(ticks))) == {"handled": 1, "failed": 0}
    assert log == [("retry", "m1", consumer.backoff(1))]


def test_a_git_deployment_builds_only_main_and_only_when_the_service_changed():
    """Git deployments build main alone, filtered by the service's inputs (ignore.py), so a landing deploys itself.  Console's CLI deploy (no branch) and an unanswerable diff both build."""
    ignore = load_module("ignore")
    asked = []

    def diff(answer):
        return lambda base: asked.append(base) or answer
    assert ignore.decide({}, diff(False))[0] == 1 and asked == []
    assert ignore.decide({"VERCEL_GIT_COMMIT_REF": "cursor/x-1234"}, diff(True))[0] == 0 and asked == []
    assert ignore.decide({"VERCEL_GIT_COMMIT_REF": "main", "VERCEL_GIT_PREVIOUS_SHA": "abc"}, diff(False))[0] == 0
    assert ignore.decide({"VERCEL_GIT_COMMIT_REF": "main", "VERCEL_GIT_PREVIOUS_SHA": "abc"}, diff(True))[0] == 1
    assert ignore.decide({"VERCEL_GIT_COMMIT_REF": "main", "VERCEL_GIT_PREVIOUS_SHA": "abc"}, diff(None))[0] == 1
    assert ignore.decide({"VERCEL_GIT_COMMIT_REF": "main"}, diff(False))[0] == 1 and asked == ["abc", "abc", "abc"]  # main's first
    assert set(ignore.INPUTS) >= {f"{p}" for p in build_sources()}


def test_ignore_reads_the_diff_from_git(tmp_path):
    import subprocess
    ignore = load_module("ignore")
    git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a], cwd=tmp_path,
                                    check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    (tmp_path / "README").write_text("x")
    git("add", "-A"); git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "README").write_text("y")
    git("commit", "-qam", "docs only")
    assert ignore.changed(base, tmp_path) is False
    (tmp_path / "infra" / "slack-events").mkdir(parents=True)
    (tmp_path / "infra" / "slack-events" / "a.py").write_text("x")
    git("add", "-A"); git("commit", "-qm", "the service")
    assert ignore.changed(base, tmp_path) is True and ignore.changed("0" * 40, tmp_path) is None


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"slack_events_service_{name}", SERVICE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_sources():
    return load_module("build").SOURCES.values()
