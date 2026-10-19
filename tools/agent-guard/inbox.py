#!/usr/bin/env python3
"""`inbox.py`: tell a running agent on this VM something. The shell hook (`shell_hook.py`) hands it over.

    python3 tools/agent-guard/inbox.py send AGENT_ID --text 'TEXT'   (or --file PATH; `--file -` reads stdin)
    python3 tools/agent-guard/inbox.py list                          each agent's pending and delivered messages (a
                                                                     pending one with `since` its oldest message's stamp)

A message waits in `$AGENT_INBOX` (default `/tmp/agent-inbox`) as `AGENT_ID.md`, and messages sent before delivery queue up
in it. The agent's next shell command is refused once with them as the refusal's text, and the agent re-runs the command.
AGENT_ID is the agent's conversation id: a subagent's is the `bc-…` id its Task call returned. Only agents whose shell
commands pass through this VM's hook can be reached this way. Message an agent on another VM through its coordinator, on
Slack.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

INBOX = Path(os.environ.get("AGENT_INBOX", "/tmp/agent-inbox"))
STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\dT\d\d:\d\dZ)\]$", re.M)


def send(agent: str, text: str, inbox: Path = INBOX) -> Path:
    if not agent or "/" in agent or agent.startswith("."):
        raise SystemExit(f"inbox: {agent!r} is no agent id (a conversation id such as bc-…)")
    if not text.strip():
        raise SystemExit("inbox: the message is empty")
    inbox.mkdir(parents=True, exist_ok=True)
    box = inbox / f"{agent}.md"
    stamp = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
    with open(box, "a") as f:
        f.write(f"[{stamp}]\n{text.rstrip()}\n\n")
    return box


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send")
    s.add_argument("agent")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--file")
    sub.add_parser("list")
    a = ap.parse_args(argv)
    if a.cmd == "send":
        text = a.text if a.text is not None else (sys.stdin.read() if a.file == "-" else Path(a.file).read_text())
        print(f"queued for {a.agent}: {send(a.agent, text)}")
        return 0
    if not INBOX.is_dir():
        print(f"inbox: {INBOX} is empty")
        return 0
    for p in sorted(INBOX.iterdir()):
        if p.suffix in (".md", ".delivered"):
            line = f"{'pending  ' if p.suffix == '.md' else 'delivered'} {p.stem}  {p.stat().st_size} bytes"
            if p.suffix == ".md" and (m := STAMP.search(p.read_text(errors="replace"))):
                line += f"  since {m[1]}"
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
