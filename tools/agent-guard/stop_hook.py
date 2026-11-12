#!/usr/bin/env python3
"""Cursor's `stop` hook (`.cursor/hooks.json`): it offers a message still waiting when an agent's turn ends to the runtime
as the turn's follow-up. No delivery relies on it.

A message reaches an agent at its next shell command, which the shell hook (`shell_hook.py`) refuses once with it. A cloud
agent whose turn ends with a message waiting starts no new turn for it, so it is reached only by resuming it, and
`inbox.py send` tells every sender so. This hook stays installed so that a runtime that honours a `stop` hook's follow-up
delivers with no further change.

When a turn completes, it pulls the agent's pushed messages (`inbox.py pull`, its git calls bounded by GIT_TIMEOUT_S) and
hands everything queued for the agent to Cursor as `followup_message`, for the runtime to submit as the agent's next
prompt. An aborted or failed turn gets nothing. A message is taken once, by whichever hook takes it first, and appended to
`<conversation_id>.delivered` like the shell hook's: one this hook hands to a runtime that drops the follow-up is left only
there.
"""
import json
import subprocess
import sys
from pathlib import Path

import inbox
import shell_hook
from settings import INBOX_POLL_S, record_event

#: each git call of the pull; the hook's own timeout in hooks.json bounds the whole run
GIT_TIMEOUT_S = 10
NOTE = ("MESSAGE FROM YOUR COORDINATOR, delivered when your turn ended by the stop hook (tools/agent-guard/stop_hook.py). "
        "Act on it before you end your run.\n\n")


def main() -> int:
    try:
        event = json.loads(sys.stdin.read() or "{}")
    except ValueError as e:                             # nothing to deliver to: the turn ends as it would
        print(f"stop_hook: the event isn't JSON ({e}); nothing delivered", file=sys.stderr)
        event = {}
    event = event if isinstance(event, dict) else {}
    record_event("stop_hook.py", event)
    conversation = str(event.get("conversation_id") or "")
    if event.get("status") != "completed" or not inbox.AGENT_ID.fullmatch(conversation):
        print("{}")
        return 0
    root = next(iter(event.get("workspace_roots") or []), "")
    repo = shell_hook.checkout(root, str(Path.cwd()))
    if INBOX_POLL_S != "off":
        inbox.GIT_TIMEOUT_S = GIT_TIMEOUT_S
        try:
            inbox.pull(conversation, Path(repo))
        except (RuntimeError, OSError, subprocess.SubprocessError) as e:      # what is already queued still goes out
            print(f"stop_hook: inbox pull failed: {type(e).__name__}: {e}", file=sys.stderr)
    try:
        message = shell_hook.delivery(conversation)
    except OSError as e:                                # the shell hook's next command tries again
        print(f"stop_hook: the inbox couldn't be read: {e}", file=sys.stderr)
        message = None
    print(json.dumps({"followup_message": NOTE + message} if message else {}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
