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
#: one line per hook run: which hook, the event's name, its conversation, tool name and key names, and of the payload's
#: other values only EVENT_VALUES
EVENT_LOG = os.environ.get("AGENT_GUARD_EVENT_LOG", "/tmp/agent-guard-events.jsonl")
#: the log rotates at this size and keeps one older file, so it holds between one and two times this much: enough to reach
#: back past `worktree_guard.GONE_S` on a VM many agents share, which is as far as `reclaim` has to see
EVENT_LOG_MAX_BYTES = 4 << 20
#: the payload's values a line keeps, each where the payload has it as a string, cut at EVENT_VALUE_MAX characters: the
#: model the payload names, and the id of the turn it belongs to
EVENT_VALUES = ("model", "generation_id")
EVENT_VALUE_MAX = 64
#: the head of a line `record_event` wrote for an agent's hook run: its time, the hook and the agent
RUN = re.compile(r'^\{"t": (\d+), "hook": "([^"]*)", "event": (?:"[^"]*"|null), "conversation_id": "([^"]+)"', re.M)
#: what a command carries, in its text, to pass where the model rule (`shell_hook.model_reason`) would refuse it: for a
#: lead that has decided the turn stays on its model. Read in the command's text, as `AGENT_GUARD_LOCAL_OK=1` is, since
#: a hook is handed the command and never the environment of the shell that runs it
MODEL_OK = "AGENT_GUARD_MODEL_OK=1"
#: the `event` of the line the model rule adds to EVENT_LOG for a turn it settles: refused once, or passed by MODEL_OK
MODEL_REFUSAL, MODEL_PASS = "model_refusal", "model_ok"
#: a model's family, the name a payload gives it without its effort: `claude-fable-5-2` of `claude-fable-5-2-xhigh`
FAMILY = re.compile(r"claude-[a-z]+-\d+-\d+")
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
    say whether this VM's shell hook runs for an agent, its key names show which event shape Cursor sent, and `inbox.py
    turns` reads from it the model each turn's payloads name. Of the payload's other values a line keeps EVENT_VALUES
    alone: never the command, a path or an address, nor anything of `tool_input` (a Task call's `model` argument is the
    worker's, which `task_hook.py` logs)."""
    e = event if isinstance(event, dict) else {}
    tool_input = e.get("tool_input")
    values = {k: e[k][:EVENT_VALUE_MAX] for k in EVENT_VALUES if isinstance(e.get(k), str) and e[k]}
    line = json.dumps({"t": int(time.time()), "hook": hook, "event": e.get("hook_event_name"),
                       "conversation_id": e.get("conversation_id"), "tool_name": e.get("tool_name"), **values,
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


def agent_events(agent: str) -> list[dict]:
    """EVENT_LOG's lines for `agent`, oldest first: those of the file it last rotated to, then its own. A line counts
    when it parses as a JSON object whose `conversation_id` is `agent`; one that doesn't hold the id as text is skipped
    unparsed, since this runs on a shell command and the log holds up to twice EVENT_LOG_MAX_BYTES, and one that holds it
    but is not JSON is skipped and said on stderr. A missing file is no lines; any other OSError rises to the caller."""
    events = []
    for path in (EVENT_LOG + ".1", EVENT_LOG):
        if not os.path.isfile(path):
            continue
        with open(path, errors="replace") as f:
            lines = [line for line in f if agent in line]
        for line in lines:
            try:
                e = json.loads(line)
            except ValueError as err:
                print(f"agent-guard: a line of {path} is not JSON, skipped: {err}", file=sys.stderr)
                continue
            if isinstance(e, dict) and e.get("conversation_id") == agent:
                events.append(e)
    return events


def model_family(model: str) -> str:
    """`model`'s family (FAMILY), which the model rule compares, so that a lead moved to another effort is never told; a
    name of no family is its own."""
    found = FAMILY.match(model)
    return found.group() if found else model


def launch_model(events: list[dict]) -> str | None:
    """The model an agent was launched on, as this machine's log has it (`events`: its lines, oldest first): the model
    its first turn here names. Its first line names that turn, and the first line of that turn that names a model, the
    model. None where the agent has no line, its first line names no turn (a hook from before the log kept one recorded
    it) or no line of its first turn names a model: then no launch model is on record, and the model rule
    (`shell_hook.model_reason`) refuses nothing.

    The rule is wrong where the log starts after the launch: it rotated the launch turn out, or the agent came to this
    machine later (a wake or a resume boots a new machine, whose log holds none of its earlier turns). Nothing on the
    machine tells such a first turn from a launch, so there the first turn the log holds stands for the launch. In that
    turn the rule refuses nothing, since its model is the one on record; in a later turn here on another model it
    refuses once and names the model on record, which is why the refusal says to go by the brief where the brief names
    another launch model."""
    first = events[0].get("generation_id") if events else None
    if not isinstance(first, str) or not first:
        return None
    return next((e["model"] for e in events
                 if e.get("generation_id") == first and isinstance(e.get("model"), str) and e["model"]), None)
