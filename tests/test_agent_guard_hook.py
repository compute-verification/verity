"""`tools/agent-guard/hook.sh` runs origin/main's hooks and Slack registry whatever the branch has: a branch that tracks
`.cursor/hooks.json` runs hooks from its own tree, so without it a branch cut before a guard landed would never run that
guard (a cloud agent on such a branch could write Linear through the connector, as Daniel)."""
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
HOLDER = "bc-8f4b652b-17bc-5055-a120-b56940b7c7a9"
REGISTRY = ".agents/skills/using-slack/registry.json"
ALLOW_ALL = 'import json\nprint(json.dumps({"permission": "allow"}))\n'


def test_an_old_branch_runs_main_s_hooks_and_registry_and_its_own_without_main(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("AGENT_GUARD_")}
    env.update(GIT_ENV, AGENT_GUARD_CACHE=str(tmp_path / "cache"), AGENT_GUARD_TASK_LOG=str(tmp_path / "task.jsonl"),
               AGENT_GUARD_MIN_AVAIL="0")
    ws = tmp_path / "ws"

    def git(*a):
        return subprocess.run(["git", *a], cwd=ws, check=True, capture_output=True, text=True, env=env).stdout

    def hook(name, event):
        p = subprocess.run(["sh", "tools/agent-guard/hook.sh", name], cwd=ws, input=json.dumps(event), capture_output=True,
                           text=True, env=env, check=True)
        return json.loads(p.stdout)["permission"]

    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(tmp_path / "origin.git")], check=True, env=env)
    subprocess.run(["git", "clone", "-q", str(tmp_path / "origin.git"), str(ws)], check=True, env=env, capture_output=True)
    shutil.copytree(ROOT / "tools/agent-guard", ws / "tools/agent-guard")
    (ws / REGISTRY).parent.mkdir(parents=True)
    (ws / REGISTRY).write_text(json.dumps({"handles": [{"handle": "proofs", "holder": f"proofs ({HOLDER})"}]}))
    git("add", "-A")
    git("commit", "-q", "-m", "main")
    git("push", "-q", "origin", "HEAD:main")
    git("checkout", "-q", "-b", "old")
    (ws / "tools/agent-guard/task_hook.py").write_text(ALLOW_ALL)
    (ws / "tools/agent-guard/shell_hook.py").write_text(ALLOW_ALL)
    (ws / REGISTRY).write_text(json.dumps({"handles": []}))
    git("commit", "-q", "-am", "an old branch: hooks without today's guards, a registry without the holder")
    git("fetch", "-q", "origin")

    linear = {"hook_event_name": "preToolUse", "tool_name": "CallDynamicTool", "conversation_id": HOLDER,
              "tool_input": {"namespace": "Linear", "toolName": "save_issue", "arguments": {}}}
    task = {"hook_event_name": "preToolUse", "tool_name": "Task", "conversation_id": HOLDER,
            "tool_input": {"subagent_type": "generalPurpose", "prompt": "x"}}
    shell = {"hook_event_name": "beforeShellExecution", "command": "env | cut -d= -f1", "cwd": str(ws)}
    assert [hook("task_hook.py", linear), hook("task_hook.py", task), hook("shell_hook.py", shell)] == ["deny"] * 3
    assert hook("shell_hook.py", {**shell, "command": "ls"}) == "allow"
    assert len([d for d in (tmp_path / "cache").iterdir() if d.is_dir() and not d.name.startswith(".")]) == 1

    git("update-ref", "-d", "refs/remotes/origin/main")
    assert [hook("task_hook.py", linear), hook("shell_hook.py", shell)] == ["allow"] * 2
