"""`.cursor/hooks.json`'s Task hook keeps a coordinator from waiting on a local worker in the foreground, which holds its turn
(and so every Slack, Linear and timer event) until the worker ends, and keeps a cloud agent from
writing Linear through its MCP connector, which acts as the person who connected it."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "tools/agent-guard/task_hook.py"
HOLDER = "bc-8f4b652b-17bc-5055-a120-b56940b7c7a9"
WORKER = "bc-00000000-0000-0000-0000-000000000000"


def answer(tmp_path, conversation_id, tool_input, tool_name="Task") -> dict:
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"handles": [{"handle": "proofs", "holder": f"the proofs coordinator ({HOLDER})"}]}))
    env = {**os.environ, "AGENT_GUARD_REGISTRY": str(reg), "AGENT_GUARD_TASK_LOG": str(tmp_path / "task.jsonl")}
    out = subprocess.run([sys.executable, HOOK], input=json.dumps({"hook_event_name": "preToolUse", "tool_name": tool_name,
                                                                     "conversation_id": conversation_id, "tool_input": tool_input}),
                         capture_output=True, text=True, env=env, check=True).stdout
    return json.loads(out)


def test_cursor_runs_the_hook_before_every_task_call_direct_or_dynamic():
    entries = json.loads((ROOT / ".cursor/hooks.json").read_text())["hooks"]["preToolUse"]
    assert {e["command"] for e in entries} == {"python3 tools/agent-guard/task_hook.py"}
    assert sorted(e["matcher"] for e in entries) == ["CallDynamicTool", "Task"]


def test_a_task_call_through_the_dynamic_tool_wrapper_is_judged_like_a_direct_one(tmp_path):
    fg = {"subagent_type": "generalPurpose", "prompt": "x"}
    assert answer(tmp_path, HOLDER, {"namespace": "cursor", "toolName": "Task", "arguments": fg},
                  "CallDynamicTool")["permission"] == "deny"
    bg = {**fg, "environment": "cloud", "run_in_background": True}
    assert answer(tmp_path, HOLDER, {"namespace": "cursor", "toolName": "Task", "arguments": bg},
                  "CallDynamicTool") == {"permission": "allow"}
    recs = [json.loads(ln) for ln in (tmp_path / "task.jsonl").read_text().splitlines()]
    assert [(r["tool_name"], r["deny"], r["environment"]) for r in recs] == [("CallDynamicTool", True, None),
                                                                             ("CallDynamicTool", False, "cloud")]


def test_other_dynamic_tools_pass_unlogged(tmp_path):
    a = answer(tmp_path, HOLDER, {"namespace": "Linear", "toolName": "get_issue", "arguments": {"id": "INF-1"}},
               "CallDynamicTool")
    assert a == {"permission": "allow"} and not (tmp_path / "task.jsonl").exists()


@pytest.mark.parametrize("conversation_id", [HOLDER, WORKER])
@pytest.mark.parametrize("tool", ["save_issue", "save_comment", "save_status_update", "delete_comment", "mark_notification"])
def test_a_cloud_agents_linear_connector_write_is_refused_with_the_apps_way(tmp_path, conversation_id, tool):
    a = answer(tmp_path, conversation_id, {"namespace": "Linear", "toolName": tool, "arguments": {"id": "INF-1"}},
               "CallDynamicTool")
    assert a["permission"] == "deny" and "acts as Daniel" in a["agent_message"] and f"`{tool}`" in a["agent_message"]
    assert "research msg ask --as ME" in a["agent_message"] and "createAsUser" in a["agent_message"]
    assert not (tmp_path / "task.jsonl").exists()


@pytest.mark.parametrize("conversation_id,tool_input", [
    (HOLDER, {"namespace": "Linear", "toolName": "list_issues", "arguments": {}}),
    (WORKER, {"namespace": "Linear", "toolName": "search_documentation", "arguments": {}}),
    (WORKER, {"namespace": "Linear", "toolName": "extract_images", "arguments": {}}),
    ("6f1d2c3e-0000-4000-8000-000000000000", {"namespace": "Linear", "toolName": "save_issue", "arguments": {}}),
    (None, {"namespace": "Linear", "toolName": "save_issue", "arguments": {}}),
    (WORKER, {"namespace": "Notion", "toolName": "notion-update-page", "arguments": {}}),
])
def test_linear_reads_a_local_sessions_writes_and_other_connectors_pass(tmp_path, conversation_id, tool_input):
    assert answer(tmp_path, conversation_id, tool_input, "CallDynamicTool") == {"permission": "allow"}


@pytest.mark.parametrize("tool_input", [
    {"subagent_type": "generalPurpose", "prompt": "x"},
    {"subagent_type": "generalPurpose", "prompt": "x", "environment": "local", "run_in_background": False},
    {"subagent_type": "best-of-n-runner", "prompt": "x"},
    {"subagent_type": "debug", "prompt": "x"},
])
def test_a_coordinators_local_worker_in_the_foreground_is_refused_with_the_way_to_run_it(tmp_path, tool_input):
    a = answer(tmp_path, HOLDER, tool_input)
    assert a["permission"] == "deny" and "@proofs" in a["agent_message"] and 'environment: "cloud"' in a["agent_message"]
    assert "run_in_background: true" in a["agent_message"]


@pytest.mark.parametrize("conversation_id,tool_input", [
    (HOLDER, {"subagent_type": "generalPurpose", "prompt": "x", "environment": "cloud"}),
    (HOLDER, {"subagent_type": "generalPurpose", "prompt": "x", "run_in_background": True}),
    (HOLDER, {"subagent_type": "explore", "prompt": "x"}),
    (HOLDER, {"subagent_type": "cursor-guide", "prompt": "x", "run_in_background": False}),
    (WORKER, {"subagent_type": "generalPurpose", "prompt": "x"}),
    (None, {"subagent_type": "generalPurpose", "prompt": "x"}),
    (HOLDER, "not an object"),
])
def test_cloud_workers_foreground_lookups_and_other_agents_pass(tmp_path, conversation_id, tool_input):
    assert answer(tmp_path, conversation_id, tool_input) == {"permission": "allow"}


def test_each_verdict_is_logged_without_the_prompt(tmp_path):
    answer(tmp_path, HOLDER, {"subagent_type": "generalPurpose", "prompt": "a secret plan", "run_in_background": False})
    (rec,) = [json.loads(ln) for ln in (tmp_path / "task.jsonl").read_text().splitlines()]
    assert rec == {"conversation_id": HOLDER, "tool_name": "Task", "deny": True, "input_keys": ["prompt", "run_in_background", "subagent_type"],
                   "environment": None, "subagent_type": "generalPurpose", "run_in_background": False}
