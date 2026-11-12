"""The agent guard's settings registry: each environment variable the inbox and its hooks share is read here, once.

Standard library only, and read at import: every hook run and every `inbox.py` command is a fresh process. The hooks run
this directory's files side by side (`boot.sh` and `hook.sh` copy it whole), so a sibling `import settings` finds it.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

#: where queued messages wait, as `<conversation_id>.md`
INBOX = os.environ.get("AGENT_INBOX", "/tmp/agent-inbox")
#: seconds between the hook's background `inbox.py pull` for one agent; `off` stops it
INBOX_POLL_S = os.environ.get("AGENT_INBOX_POLL_S", "30")
#: one line per hook run: which hook, the event's name, its conversation, tool name and key names (never a value)
EVENT_LOG = os.environ.get("AGENT_GUARD_EVENT_LOG", "/tmp/agent-guard-events.jsonl")
#: the log rotates at this size and keeps one older file, so it holds between one and two times this much: enough to reach
#: back past `worktree_guard.GONE_S` on a VM many agents share, which is as far as `reclaim` has to see
EVENT_LOG_MAX_BYTES = 4 << 20
#: the head of a line `record_event` wrote for an agent's hook run: its time, the hook and the agent
RUN = re.compile(r'^\{"t": (\d+), "hook": "([^"]*)", "event": (?:"[^"]*"|null), "conversation_id": "([^"]+)"', re.M)
#: this agent's conversation id where Cursor's shell sets it: the agent `inbox.py selftest` checks when none is named
CONVERSATION_ID = os.environ.get("CURSOR_CONVERSATION_ID", "")
#: a regex for the push URLs the push rules judge, in place of verity's and infra's on GitHub (`worktree_guard.QUEUE_URL`)
PUSH_REMOTES = os.environ.get("AGENT_GUARD_PUSH_REMOTES", "")


def record_event(hook: str, event: object) -> None:
    """Append what `hook` was handed to EVENT_LOG, keeping the last EVENT_LOG_MAX_BYTES or so; `inbox.py check` reads it to
    say whether this VM's shell hook runs for an agent, and its key names show which event shape Cursor sent."""
    e = event if isinstance(event, dict) else {}
    tool_input = e.get("tool_input")
    line = json.dumps({"t": int(time.time()), "hook": hook, "event": e.get("hook_event_name"),
                       "conversation_id": e.get("conversation_id"), "tool_name": e.get("tool_name"),
                       "keys": sorted(e) if isinstance(event, dict) else type(event).__name__,
                       "tool_input_keys": sorted(tool_input) if isinstance(tool_input, dict) else None})
    try:
        if os.path.isfile(EVENT_LOG) and os.path.getsize(EVENT_LOG) > EVENT_LOG_MAX_BYTES:
            os.replace(EVENT_LOG, EVENT_LOG + ".1")
        with open(EVENT_LOG, "a") as f:
            f.write(line + "\n")
    except OSError as e:                                # the hook's verdict never waits on its log
        print(f"agent-guard: event not recorded in {EVENT_LOG}: {e}", file=sys.stderr)


def recorded_runs() -> list[tuple[int, str, str]]:
    """(time, hook, agent) of every hook run EVENT_LOG still holds for an agent, in the file it last rotated to and in itself:
    the VM's one record of when each agent last ran a command."""
    runs = []
    for path in (EVENT_LOG + ".1", EVENT_LOG):
        if os.path.isfile(path):
            with open(path, errors="replace") as f:
                runs += [(int(t), hook, agent) for t, hook, agent in RUN.findall(f.read())]
    return runs
