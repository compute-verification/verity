#!/usr/bin/env python3
"""Cursor's `preToolUse` hook for the Task tool (`.cursor/hooks.json`, matcher `Task`): a coordinator runs its workers as cloud
agents, not as local subagents on its own VM.

Slack, Linear and timer events reach an agent only while its turn is free, and a local subagent, in the background or not, keeps
its parent's turn open until it ends: on 18 Oct the proofs coordinator heard nothing for ten hours while six local workers ran
(note:proofs/20261018T1550Z-friction-coordinator-turn-held-by-local-subagents). So for an agent that holds a handle (its id is
in a `holder` of the Slack registry, `.agents/skills/using-slack/registry.json`) it refuses a Task call unless the call sets
`environment: cloud`, or asks for one of the short read-only lookups in `LOOKUPS` in the foreground. Every other agent's Task
calls pass. Reads Cursor's hook JSON on stdin and prints its verdict as JSON on stdout; an input it can't read passes. Each
verdict is appended to $AGENT_GUARD_TASK_LOG (default /tmp/agent-guard-task.jsonl) with the call's environment, subagent type
and background flag, never its prompt.
"""
import json
import os
import re
import sys
from pathlib import Path

REGISTRY = Path(os.environ.get("AGENT_GUARD_REGISTRY",
                               Path(__file__).resolve().parents[2] / ".agents/skills/using-slack/registry.json"))
LOG = Path(os.environ.get("AGENT_GUARD_TASK_LOG", "/tmp/agent-guard-task.jsonl"))
LOOKUPS = {"explore", "cursor-guide", "ci-investigator", "videoReview"}
AGENT_ID = re.compile(r"bc-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def holders() -> dict[str, str]:
    """Agent id -> the handle it holds."""
    try:
        reg = json.loads(REGISTRY.read_text())
    except (OSError, ValueError):
        return {}
    return {a: h.get("handle", "?") for h in reg.get("handles") or [] for a in AGENT_ID.findall(str(h.get("holder", "")))}


def reason(event: dict) -> str | None:
    handle = holders().get(str(event.get("conversation_id") or ""))
    args = event.get("tool_input")
    if handle is None or not isinstance(args, dict):
        return None
    if args.get("environment") == "cloud":
        return None
    if args.get("subagent_type") in LOOKUPS and not args.get("run_in_background"):
        return None
    return (f"Refused by tools/agent-guard/task_hook.py: you hold @{handle}, and a local subagent keeps your turn open until it "
            "ends, so no Slack, Linear or timer event reaches you meanwhile (proofs heard nothing for ten hours on 18 Oct). Run "
            "the worker as a cloud agent (Task with environment: \"cloud\"; it gets its own VM and branch) or as a detached "
            "`research run`, then end your turn. Short read-only lookups in the foreground still run locally: "
            f"{', '.join(sorted(LOOKUPS))}.")


def main() -> int:
    try:
        event = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        event = {}
    event = event if isinstance(event, dict) else {}
    why = reason(event)
    args = event.get("tool_input") if isinstance(event.get("tool_input"), dict) else {}
    try:
        with open(LOG, "a") as f:
            f.write(json.dumps({"conversation_id": event.get("conversation_id"), "deny": bool(why), "input_keys": sorted(args),
                                **{k: args.get(k) for k in ("environment", "subagent_type", "run_in_background")}}) + "\n")
    except OSError:
        pass
    print(json.dumps({"permission": "deny", "agent_message": why, "user_message": why} if why else {"permission": "allow"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
