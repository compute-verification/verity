#!/usr/bin/env python3
"""Cursor's `preToolUse` hook for the Task tool (`.cursor/hooks.json`, matchers `Task` and `CallDynamicTool`, the wrapper a
session that offers Task as a dynamic tool calls it through): a coordinator never waits on a local worker in the foreground.

Slack, Linear and timer events reach an agent only while its turn is free, and a foreground subagent holds its parent's turn
until it ends; one in the background doesn't. Local workers also share the coordinator's VM, where one OOM kills them all
(note:proofs/20261018T1550Z-friction-coordinator-turn-held-by-local-subagents, rule 1: cloud workers where the Task tool offers
them, background always). So for an agent that holds a handle (its id is in a `holder` of the Slack registry,
`.agents/skills/using-slack/registry.json`) it refuses a local Task call in the foreground, unless it is one of the short
read-only lookups in `LOOKUPS`. Cloud and background calls pass, and so does every other agent's. Reads Cursor's hook JSON on stdin and prints its verdict as JSON on stdout; an input it can't read passes. Each
verdict is appended to $AGENT_GUARD_TASK_LOG (default /tmp/agent-guard-task.jsonl) with the call's environment, subagent type
and background flag, never its prompt.

It also refuses a cloud agent's (a `bc-` conversation's) write through Linear's MCP connector, which acts as Daniel, so the
write would show as his: every Linear tool but the reads in `LINEAR_READS`. The refusal names the app's way to do it (the
using-slack skill). Reads pass, and so does a local session's write, Daniel's own among them. These verdicts aren't logged.
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
LINEAR_READS = ("get_", "list_", "search_", "extract_")


def holders() -> dict[str, str]:
    """Agent id -> the handle it holds."""
    try:
        reg = json.loads(REGISTRY.read_text())
    except (OSError, ValueError):
        return {}
    return {a: h.get("handle", "?") for h in reg.get("handles") or [] for a in AGENT_ID.findall(str(h.get("holder", "")))}


def task_input(event: dict):
    """The Task call's input; where Task is a dynamic tool it arrives as `CallDynamicTool` with toolName `Task`, and a
    dynamic call of any other tool has none."""
    args = event.get("tool_input")
    if event.get("tool_name") == "CallDynamicTool":
        return args.get("arguments") if isinstance(args, dict) and args.get("toolName") == "Task" else None
    return args


def reason(event: dict) -> str | None:
    handle = holders().get(str(event.get("conversation_id") or ""))
    args = task_input(event)
    if handle is None or not isinstance(args, dict):
        return None
    if args.get("environment") == "cloud" or args.get("run_in_background") or args.get("subagent_type") in LOOKUPS:
        return None
    return (f"Refused by tools/agent-guard/task_hook.py: you hold @{handle}, and a foreground subagent holds your turn until it "
            "ends, so no Slack, Linear or timer event reaches you meanwhile. Run the worker as a cloud agent (Task with "
            "environment: \"cloud\"; its own VM, so an OOM there can't take yours) where your Task tool offers it, else with "
            "run_in_background: true, or as a detached `research run`; then end your turn. Short read-only lookups run in the "
            f"foreground: {', '.join(sorted(LOOKUPS))}.")


def linear_write(event: dict) -> str | None:
    """Why a cloud agent's call of a Linear connector tool that writes is refused, else None."""
    args = event.get("tool_input")
    if event.get("tool_name") != "CallDynamicTool" or not isinstance(args, dict):
        return None
    tool = str(args.get("toolName") or "")
    if ("linear" not in str(args.get("namespace") or "").lower() or tool.startswith(LINEAR_READS)
            or not AGENT_ID.fullmatch(str(event.get("conversation_id") or ""))):
        return None
    return (f"Refused by tools/agent-guard/task_hook.py: Linear's MCP connector acts as Daniel, so `{tool}` would show as "
            "his. Write through the app, as your handle (the using-slack skill): `research msg ask --as ME --to @OWNER "
            "--file BRIEF.md` files an issue; `research msg linear state|edit|place ISSUE ...` changes one and `research "
            "msg linear comment ISSUE TEXT --as ME` comments; `research.linear.Linear().gql(...)` with `createAsUser` does "
            "what no verb does. The connector's reads pass.")


def main() -> int:
    try:
        event = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        event = {}
    event = event if isinstance(event, dict) else {}
    if why := linear_write(event):
        print(json.dumps({"permission": "deny", "agent_message": why, "user_message": why}))
        return 0
    why, args = reason(event), task_input(event)
    if event.get("tool_name") == "CallDynamicTool" and args is None:
        print(json.dumps({"permission": "allow"}))
        return 0
    args = args if isinstance(args, dict) else {}
    try:
        with open(LOG, "a") as f:
            f.write(json.dumps({"conversation_id": event.get("conversation_id"), "tool_name": event.get("tool_name"),
                                "deny": bool(why), "input_keys": sorted(args),
                                **{k: args.get(k) for k in ("environment", "subagent_type", "run_in_background")}}) + "\n")
    except OSError:
        pass
    print(json.dumps({"permission": "deny", "agent_message": why, "user_message": why} if why else {"permission": "allow"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
