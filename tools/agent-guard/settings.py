"""The agent guard's settings registry: each environment variable the inbox and its hooks share is read here, once.

Standard library only, and read at import: every hook run and every `inbox.py` command is a fresh process. The hooks run
this directory's files side by side (`boot.sh` and `hook.sh` copy it whole), so a sibling `import settings` finds it.
"""
from __future__ import annotations

import json
import os
import sys
import time

#: where queued messages wait, as `<conversation_id>.md`
INBOX = os.environ.get("AGENT_INBOX", "/tmp/agent-inbox")
#: seconds between the hook's background `inbox.py pull` for one agent; `off` stops it
INBOX_POLL_S = os.environ.get("AGENT_INBOX_POLL_S", "30")
#: one line per hook run: which hook, the event's name, its conversation, tool name and key names (never a value)
EVENT_LOG = os.environ.get("AGENT_GUARD_EVENT_LOG", "/tmp/agent-guard-events.jsonl")
EVENT_LOG_MAX_BYTES = 1 << 20
#: this agent's conversation id where Cursor's shell sets it: the agent `inbox.py selftest` checks when none is named
CONVERSATION_ID = os.environ.get("CURSOR_CONVERSATION_ID", "")
#: a regex for the push URLs the push rules judge, in place of verity's and infra's on GitHub (`worktree_guard.QUEUE_URL`)
PUSH_REMOTES = os.environ.get("AGENT_GUARD_PUSH_REMOTES", "")
#: the secrets Cursor injects into this VM, by name, and those of them this process's environment lacks: what
#: `kept_session.py start` reads before it starts a tmux server (a value is never read out)
INJECTED = [n for n in os.environ.get("CLOUD_AGENT_INJECTED_SECRET_NAMES", "").split(",") if n]
INJECTED_UNSET = [n for n in INJECTED if not os.environ.get(n)]


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
