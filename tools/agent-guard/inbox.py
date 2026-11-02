#!/usr/bin/env python3
"""`inbox.py`: tell a running agent something, on this VM or any other. The shell hook (`shell_hook.py`) hands it over.

    python3 tools/agent-guard/inbox.py send AGENT_ID --text 'TEXT'   (or --file PATH; `--file -` reads stdin)
    python3 tools/agent-guard/inbox.py send AGENT_ID --local ...     only this VM's inbox, no push
    python3 tools/agent-guard/inbox.py pull AGENT_ID [--repo DIR]    move AGENT_ID's pushed messages into this VM's inbox
    python3 tools/agent-guard/inbox.py list                          each agent's pending and delivered messages (a
                                                                     pending one with `since` its oldest message's stamp)

`send` pushes the message as a commit to origin's branch `inbox/AGENT_ID` (one file, `inbox.md`), from the checkout it runs
in (`--repo`). Wherever AGENT_ID runs, its shell hook starts `pull` in the background at most every $AGENT_INBOX_POLL_S
seconds (default 30): `pull` appends the branch's new messages to this VM's inbox and pushes a commit that empties the file,
so a message reaches one VM once. A message waits in `$AGENT_INBOX` (default `/tmp/agent-inbox`) as `AGENT_ID.md`, and
messages sent before delivery queue up in it. The agent's next shell command is refused once with them as the refusal's text,
and the agent re-runs the command, so a pushed message arrives within a poll interval and two of the agent's commands.
`--local` writes this VM's inbox directly and reaches only agents whose shell commands pass through this VM's hook.

AGENT_ID is the agent's conversation id: a cloud agent's or a subagent's is the `bc-…` id its Task call returned.
"""
from __future__ import annotations

import argparse
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

INBOX = Path(os.environ.get("AGENT_INBOX", "/tmp/agent-inbox"))
STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\dT\d\d:\d\dZ)\](?: [0-9a-f]{12})?$", re.M)
HEAD = re.compile(r"^\[\d{4}-\d\d-\d\dT\d\d:\d\dZ\] ([0-9a-f]{12})$", re.M)
FILE = "inbox.md"
PUSH_TRIES = 4
LOCK_STALE_S = 300


def agent_id(agent: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", agent or ""):
        raise SystemExit(f"inbox: {agent!r} is no agent id (a conversation id such as bc-…)")
    return agent


def entry(text: str, ident: str | None = None) -> str:
    if not text.strip():
        raise SystemExit("inbox: the message is empty")
    stamp = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
    return f"[{stamp}]{' ' + ident if ident else ''}\n{text.rstrip()}\n\n"


def send(agent: str, text: str, inbox: Path = INBOX) -> Path:
    """Queue `text` for `agent` in this VM's inbox."""
    inbox.mkdir(parents=True, exist_ok=True)
    box = inbox / f"{agent_id(agent)}.md"
    with open(box, "a") as f:
        f.write(entry(text))
    return box


def git(repo: Path, *args: str, stdin: str | None = None) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], input=stdin, capture_output=True, text=True, timeout=120)
    if r.returncode:
        raise RuntimeError(f"git {args[0]} exited {r.returncode}: {r.stderr.strip()[-400:]}")
    return r.stdout


def tip(repo: Path, agent: str) -> str | None:
    out = git(repo, "ls-remote", "origin", f"refs/heads/inbox/{agent}").split()
    return out[0] if out else None


def remote_text(repo: Path, agent: str, sha: str) -> str:
    git(repo, "fetch", "-q", "--no-write-fetch-head", "origin", f"+refs/heads/inbox/{agent}:refs/agent-inbox/{agent}")
    return git(repo, "cat-file", "blob", f"{sha}:{FILE}")


def commit(repo: Path, agent: str, text: str, parent: str | None, message: str) -> str:
    """Push a commit whose `inbox.md` is `text` onto `inbox/agent` at `parent`; a moved branch refuses it (no force)."""
    blob = git(repo, "hash-object", "-w", "--stdin", stdin=text).strip()
    tree = git(repo, "mktree", stdin=f"100644 blob {blob}\t{FILE}\n").strip()
    sha = git(repo, "commit-tree", tree, *(["-p", parent] if parent else []), "-m", message).strip()
    git(repo, "push", "-q", "origin", f"{sha}:refs/heads/inbox/{agent}")
    return sha


def push(agent: str, text: str, repo: Path) -> str:
    """Append `text` to origin's `inbox/agent`, retrying while another push moves the branch; the new tip."""
    item, err = entry(text, secrets.token_hex(6)), None
    for _ in range(PUSH_TRIES):
        try:
            at = tip(repo, agent)
            return commit(repo, agent, (remote_text(repo, agent, at) if at else "") + item, at, f"inbox: a message for {agent}")
        except RuntimeError as e:
            err = e
    raise SystemExit(f"inbox: could not push to origin's inbox/{agent} after {PUSH_TRIES} tries: {err}")


def split(text: str) -> list[tuple[str, str]]:
    """`(id, block)` for each message in a pushed inbox, in order."""
    heads = list(HEAD.finditer(text))
    return [(m[1], text[m.start():heads[i + 1].start() if i + 1 < len(heads) else len(text)]) for i, m in enumerate(heads)]


def pull(agent: str, repo: Path, inbox: Path = INBOX) -> int:
    """Move the messages on origin's `inbox/agent` into this VM's inbox, then empty the branch's file; how many arrived.

    A message already in the inbox (by its id) isn't added again, so a pull whose emptying push lost a race repeats safely."""
    agent_id(agent)
    inbox.mkdir(parents=True, exist_ok=True)
    lock = inbox / f"{agent}.pulling"
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        if time.time() - lock.stat().st_mtime < LOCK_STALE_S:
            return 0
        lock.unlink(missing_ok=True)
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    try:
        seen = inbox / f"{agent}.seen"
        at = tip(repo, agent)
        if at is None or (seen.exists() and seen.read_text().strip() == at):
            return 0
        items = split(remote_text(repo, agent, at))
        here = "".join(p.read_text(errors="replace") for p in (inbox / f"{agent}.md", inbox / f"{agent}.delivered") if p.exists())
        known = set(HEAD.findall(here))
        fresh = [block for ident, block in items if ident not in known]
        if fresh:
            with open(inbox / f"{agent}.md", "a") as f:
                f.write("".join(fresh))
        if items:
            at = commit(repo, agent, "", at, f"inbox: {agent} took {len(items)} message(s)")
        seen.write_text(at + "\n")
        return len(fresh)
    finally:
        lock.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send")
    s.add_argument("agent")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--file")
    s.add_argument("--local", action="store_true", help="this VM's inbox only, no push")
    s.add_argument("--repo", type=Path, default=Path("."), help="the checkout whose origin carries the message")
    p = sub.add_parser("pull")
    p.add_argument("agent")
    p.add_argument("--repo", type=Path, default=Path("."))
    sub.add_parser("list")
    a = ap.parse_args(argv)
    if a.cmd == "send":
        text = a.text if a.text is not None else (sys.stdin.read() if a.file == "-" else Path(a.file).read_text())
        if a.local:
            print(f"queued for {a.agent}: {send(a.agent, text)}")
        else:
            print(f"pushed for {a.agent}: origin's inbox/{agent_id(a.agent)} at {push(a.agent, text, a.repo)[:12]}; its "
                  "shell hook delivers it within a poll interval")
        return 0
    if a.cmd == "pull":
        print(f"inbox: {pull(a.agent, a.repo)} new message(s) for {a.agent}")
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
