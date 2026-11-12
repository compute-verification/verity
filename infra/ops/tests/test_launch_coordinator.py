"""`launch_coordinator.py tell` (infra/cursor) sends one message to each coordinator's chat: one request an agent, to the
holder the registry names, top left out unless named; a dry run sends nothing and needs no key; a bad handle or an empty
message refuses before any request; one target's failure stops none of the others."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("launch_coordinator", ROOT / "infra" / "cursor" / "launch_coordinator.py")
L = sys.modules["launch_coordinator"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

KEY = "dummy-key"
TOP, LEAN, CI, VLLM, OTHER = (f"bc-{n:08x}-0000-0000-0000-000000000000" for n in range(1, 6))
DESKTOP = "f7772a12-0000-0000-0000-000000000000"


class Api:
    """A stand-in for `call`: records each request and answers by agent id (`refuse` maps one to its status and body)."""

    def __init__(self, refuse: dict | None = None):
        self.calls, self.refuse = [], refuse or {}

    def __call__(self, method, path, key, body=None):
        self.calls.append((method, path, key, body))
        agent = path.split("/")[3]
        if agent in self.refuse:
            return self.refuse[agent]
        return 202, {"delivery": "steered", "runId": "run-1"}

    def agents(self) -> list[str]:
        return [path.split("/")[3] for _, path, _, _ in self.calls]


@pytest.fixture
def api(tmp_path, monkeypatch):
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"handles": [
        {"handle": "top", "holder": TOP}, {"handle": "lean", "holder": LEAN}, {"handle": "ci", "holder": CI},
        {"handle": "vllm", "holder": VLLM}, {"handle": "console", "holder": DESKTOP}, {"handle": "unheld"}]}))
    monkeypatch.setattr(L, "REGISTRY", reg)
    monkeypatch.setenv("CURSOR_API_KEY", KEY)
    a = Api()
    monkeypatch.setattr(L, "call", a)
    return a


def tell(*argv: str) -> None:
    L.main(["tell", *argv])


def test_each_coordinator_but_top_gets_exactly_one_request_carrying_the_text(api, capsys):
    tell("--text", "Ruling: ship it.")
    assert api.agents() == [LEAN, CI, VLLM], "registry order; top, a desktop agent and an unheld handle left out"
    assert all(c == ("POST", f"/v1/agents/{a}/runs", KEY, {"prompt": {"text": "Ruling: ship it."}, "delivery": "steer"})
               for c, a in zip(api.calls, api.agents()))
    assert capsys.readouterr().out.splitlines() == [f"{h}\t{a}\tsent (steered)" for h, a in
                                                     (("lean", LEAN), ("ci", CI), ("vllm", VLLM))]


def test_no_steer_sends_a_plain_follow_up(api, capsys):
    api.refuse[CI] = 201, {"run": {"id": "run-9"}}
    tell("--text", "x", "--only", "ci", "--no-steer")
    assert api.calls == [("POST", f"/v1/agents/{CI}/runs", KEY, {"prompt": {"text": "x"}})]
    assert capsys.readouterr().out == f"ci\t{CI}\tsent (run run-9)\n"


def test_a_dry_run_sends_nothing_and_needs_no_key(api, monkeypatch, capsys):
    monkeypatch.delenv("CURSOR_API_KEY")
    tell("--text", "Ruling: ship it.", "--dry-run")
    out = capsys.readouterr().out
    assert api.calls == []
    assert out.splitlines()[:3] == [f"{h}\t{a}\thttps://cursor.com/agents/{a}" for h, a in
                                    (("lean", LEAN), ("ci", CI), ("vllm", VLLM))]
    assert out.count("Ruling: ship it.") == 1 and out.endswith("Ruling: ship it.\n")


def test_the_message_can_come_from_a_file(api, tmp_path):
    (tmp_path / "msg.md").write_text("Ruling from a file.\n")
    tell("--file", str(tmp_path / "msg.md"), "--only", "lean")
    assert [c[3]["prompt"]["text"] for c in api.calls] == ["Ruling from a file.\n"]


def test_only_skip_and_agent_choose_the_targets(api):
    tell("--text", "x", "--only", "top", "ci")
    assert api.agents() == [TOP, CI], "--only names top too"
    api.calls.clear()
    tell("--text", "x", "--skip", "ci", "--skip", "vllm")
    assert api.agents() == [LEAN]
    api.calls.clear()
    tell("--text", "x", "--only", "lean", "--agent", OTHER, LEAN)
    assert api.agents() == [LEAN, OTHER], "an --agent id adds a target once"


@pytest.mark.parametrize("argv", [["--only", "nobody"], ["--skip", "console"], ["--only", "lean", "unheld"],
                                  ["--agent", "not-an-id"]])
def test_an_unknown_handle_or_agent_refuses_before_any_request(api, argv):
    with pytest.raises(SystemExit) as exit_:
        tell("--text", "x", *argv)
    assert api.calls == [] and "nothing sent" in str(exit_.value)
    if argv[0] != "--agent":
        assert str(exit_.value).endswith("gives a cloud agent: top, lean, ci, vllm")


@pytest.mark.parametrize("text", ["", "  \n\t"])
def test_an_empty_message_refuses_before_any_request(api, text):
    with pytest.raises(SystemExit) as exit_:
        tell("--text", text)
    assert api.calls == [] and str(exit_.value) == "the message is empty; nothing sent"


def test_one_failing_target_stops_none_of_the_others_and_the_exit_is_nonzero(api, monkeypatch, capsys):
    api.refuse[LEAN] = 409, {"code": "agent_archived"}
    real = L.call

    def unreachable(method, path, key, body=None):
        if CI in path:
            raise OSError("connection refused")
        return real(method, path, key, body)
    monkeypatch.setattr(L, "call", unreachable)
    with pytest.raises(SystemExit) as exit_:
        tell("--text", "x")
    assert api.agents() == [LEAN, VLLM], "ci's request raised before the stand-in recorded it; vllm was still sent"
    assert capsys.readouterr().out.splitlines() == [
        f'lean\t{LEAN}\tFAILED HTTP 409 {{"code": "agent_archived"}}',
        f"ci\t{CI}\tFAILED not reached: connection refused",
        f"vllm\t{VLLM}\tsent (steered)"]
    assert str(exit_.value) == (f'2 of 3 not sent:\nlean {LEAN}: HTTP 409 {{"code": "agent_archived"}}\n'
                                f"ci {CI}: not reached: connection refused")


def test_a_send_needs_the_key(api, monkeypatch):
    monkeypatch.delenv("CURSOR_API_KEY")
    with pytest.raises(SystemExit, match="set CURSOR_API_KEY"):
        tell("--text", "x")
    assert api.calls == []


def test_the_registry_names_each_coordinator_by_its_cloud_agent():
    held = L.coordinators()
    assert "top" in held and "console" not in held and "dashboard" not in held, "desktop agents get no follow-ups"
    assert all(L.CLOUD_AGENT.fullmatch(a) for a in held.values())
