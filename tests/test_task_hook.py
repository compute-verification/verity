"""`.cursor/hooks.json`'s Task hook keeps a coordinator from waiting on a local worker in the foreground, which holds its turn
(and so every Slack, Linear and timer event) until the worker ends (proofs' rule 1, 18 Oct)."""
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


def answer(tmp_path, conversation_id, tool_input) -> dict:
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"handles": [{"handle": "proofs", "holder": f"the proofs coordinator ({HOLDER})"}]}))
    env = {**os.environ, "AGENT_GUARD_REGISTRY": str(reg), "AGENT_GUARD_TASK_LOG": str(tmp_path / "task.jsonl")}
    out = subprocess.run([sys.executable, HOOK], input=json.dumps({"hook_event_name": "preToolUse", "tool_name": "Task",
                                                                     "conversation_id": conversation_id, "tool_input": tool_input}),
                         capture_output=True, text=True, env=env, check=True).stdout
    return json.loads(out)


def test_cursor_runs_the_hook_before_every_task_call():
    (entry,) = json.loads((ROOT / ".cursor/hooks.json").read_text())["hooks"]["preToolUse"]
    assert entry["command"] == "python3 tools/agent-guard/task_hook.py" and entry["matcher"] == "Task"


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
    assert rec == {"conversation_id": HOLDER, "deny": True, "input_keys": ["prompt", "run_in_background", "subagent_type"],
                   "environment": None, "subagent_type": "generalPurpose", "run_in_background": False}
