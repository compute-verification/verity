"""One line, once a session, when the agent guard is not running on this agent's VM.

A cloud agent's session runs the hooks of the branch its VM booted on. Where Cursor could load no shell hook from that
branch's `.cursor/hooks.json`, nothing refuses a command that would print a secret and no inbox message arrives, and nothing
else says so: every command just works. `tools/agent-guard/inbox.py guard` decides whether a shell hook has run for the
agent here and writes the line (what is off, why, the fix); `research` asks it as an agent's first command on a VM starts
(`at_main.reexec`, the first thing a command-line `research` does) and keeps a mark, so its later commands ask nothing.
Only an agent is asked: a person, a pod and a node have no `CURSOR_CONVERSATION_ID`.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Mapping

TOOL = Path(__file__).resolve().parents[4] / "tools/agent-guard/inbox.py"
#: a conversation id as the hook reads it, and so the only kind the tool answers for; it also names the mark's file
AGENT_ID = re.compile(r"(bc-)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
ASK_TIMEOUT_S = 20


def notice(env: Mapping[str, str], tool: Path = TOOL, marks: Path | None = None, run: Callable = subprocess.run,
           out=None) -> str | None:
    """Print the guard's line for this agent, if it has one and this session hasn't been told; what was printed, or None.

    `env` is the process's environment: the agent is its `CURSOR_CONVERSATION_ID`, and its checkout `RESEARCH_CHECKOUT`
    (the agent VM's `research` sets it) or the working directory. A process under a test (`PYTEST_CURRENT_TEST`, which a
    test's subprocesses inherit) asks nothing: a test reads only what its suite declares, and nothing of the VM it happens
    to run on. The mark is per agent, under the temporary directory, which a new VM starts without; an answer that couldn't
    be had is said and not marked, so the next command asks again."""
    agent = env.get("CURSOR_CONVERSATION_ID", "").strip()
    if env.get("PYTEST_CURRENT_TEST") or not AGENT_ID.fullmatch(agent) or not tool.is_file():
        return None
    mark = (marks or Path(tempfile.gettempdir()) / "research-guard-notice") / agent
    if mark.exists():
        return None

    def say(text: str) -> str:
        print(f"research: {text}", file=out or sys.stderr)
        return f"research: {text}"

    try:
        p = run([sys.executable, str(tool), "guard", agent, "--repo", env.get("RESEARCH_CHECKOUT") or "."],
                capture_output=True, text=True, timeout=ASK_TIMEOUT_S, env=dict(env), check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return say(f"could not ask whether the agent guard runs on this VM (`{tool} guard`): {e}")
    line = " ".join(p.stdout.split())
    if p.returncode not in (0, 1) or (p.returncode == 1) != bool(line):
        return say(f"could not ask whether the agent guard runs on this VM: `{tool} guard` exited {p.returncode}: "
                   f"{(line or p.stderr.strip())[-300:]}")
    try:
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.touch()
    except OSError as e:
        say(f"could not write {mark}, so the next command asks again: {e}")
    return say(line) if line else None
