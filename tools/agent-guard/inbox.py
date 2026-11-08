#!/usr/bin/env python3
"""`inbox.py`: tell a running agent something, on this VM or any other. The shell hook (`shell_hook.py`) hands it over.

    python3 tools/agent-guard/inbox.py send AGENT_ID --text 'TEXT'   (or --file PATH; `--file -` reads stdin)
    python3 tools/agent-guard/inbox.py send AGENT_ID --local ...     only this VM's inbox, no push
    python3 tools/agent-guard/inbox.py pull AGENT_ID [--repo DIR]    move AGENT_ID's pushed messages into this VM's inbox
    python3 tools/agent-guard/inbox.py list                          each agent's pending and delivered messages (a
                                                                     pending one with `since` its oldest message's stamp)
    python3 tools/agent-guard/inbox.py status AGENT_ID [--repo DIR]  whether AGENT_ID's VM has taken its pushed messages;
                                                                     exit 1 when one has waited past STALE_S unpulled
    python3 tools/agent-guard/inbox.py check AGENT_ID                on AGENT_ID's own VM: whether this VM's hook polls
                                                                     for it; exit 1, with how to read the branch, if not

`send` pushes the message as a commit to origin's branch `inbox/AGENT_ID` (one file, `inbox.md`), from the checkout it runs
in (`--repo`). Wherever AGENT_ID runs, its shell hook starts `pull` in the background at most every $AGENT_INBOX_POLL_S
seconds (default 30): `pull` appends the branch's new messages to this VM's inbox and pushes a commit that empties the file,
so a message reaches one VM once. A message waits in `$AGENT_INBOX` (default `/tmp/agent-inbox`) as `AGENT_ID.md`, and
messages sent before delivery queue up in it. The agent's next shell command is refused once with them as the refusal's text,
and the agent re-runs the command, so a pushed message arrives within a poll interval and two of the agent's commands. An
agent whose turn has ended (FINISHED or IDLE) runs no command and takes nothing until it is resumed, with Task or a follow-up.
`--local` writes this VM's inbox directly and reaches only agents whose shell commands pass through this VM's hook.

AGENT_ID is the agent's conversation id: a cloud agent's or a subagent's is the `bc-…` id its Task call returned.
"""
from __future__ import annotations

import argparse
import calendar
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

import settings

INBOX = Path(settings.INBOX)
STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\dT\d\d:\d\dZ)\](?: [0-9a-f]{12})?$", re.M)
HEAD = re.compile(r"^\[\d{4}-\d\d-\d\dT\d\d:\d\dZ\] ([0-9a-f]{12})$", re.M)
FILE = "inbox.md"
PUSH_TRIES = 4
LOCK_STALE_S = 300
#: a pushed message no VM has taken after this long: its target's hook isn't polling (INF-608: BINDKSAT's branch held every
#: message from 04:33Z to 07:32Z while each `send` said only "pushed")
STALE_S = 600


def agent_id(agent: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", agent or ""):
        raise SystemExit(f"inbox: {agent!r} is no agent id (a conversation id such as bc-…)")
    return agent


def entry(text: str, ident: str | None = None) -> str:
    if not text.strip():
        raise SystemExit("inbox: the message is empty")
    stamp = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
    return f"[{stamp}]{' ' + ident if ident else ''}\n{text.rstrip()}\n\n"


#: a conversation id as the hook reads it (`conversation_id`): a UUID, `bc-` before it for a cloud agent or subagent
AGENT_ID = re.compile(r"(bc-)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def full_id(agent: str) -> str:
    if not AGENT_ID.fullmatch(agent or ""):
        raise SystemExit(f"inbox: {agent!r} is no full agent id; the hook delivers only to the whole conversation id, "
                         "e.g. bc-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (the id the Task call returned), never a prefix")
    return agent


def send(agent: str, text: str, inbox: Path = INBOX) -> Path:
    """Queue `text` for `agent` in this VM's inbox."""
    full_id(agent)
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
    full_id(agent)
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


def unpulled(repo: Path, agent: str) -> tuple[int, str | None]:
    """How many messages origin's `inbox/agent` holds that no VM has taken, and the oldest one's stamp."""
    at = tip(repo, agent)
    if at is None:
        return 0, None
    stamps = STAMP.findall(remote_text(repo, agent, at))
    return len(stamps), (stamps[0] if stamps else None)


def age_s(stamp: str, now: float | None = None) -> float:
    return (time.time() if now is None else now) - calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%MZ"))


def stale_note(agent: str, n: int, oldest: str) -> str:
    return (f"{n} message(s) on origin's inbox/{agent} are still unpulled, the oldest from {oldest}: its VM's shell hook isn't "
            f"polling (its turn ended, FINISHED or IDLE, so it runs no shell command until resumed with Task; or a hook older "
            f"than the inbox). The agent reads them with `git fetch origin inbox/{agent} && git show FETCH_HEAD:{FILE}`, and "
            f"`inbox.py check {agent}` on its VM says why")


#: a target that ended its turn runs no shell command, so its hook never pulls (INF-608: MEMPOOL finished and marked #8132
#: before circuits' two stop messages reached it); a sender without the Cursor API can't see that, so `send` always says it
UNTIL_RESUMED = ("it arrives only while {agent}'s turn runs shell commands: if {agent} has ended its turn (FINISHED or IDLE), "
                 "resume it with Task (or a follow-up), which delivers at once")


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
    st = sub.add_parser("status")
    st.add_argument("agent")
    st.add_argument("--repo", type=Path, default=Path("."))
    ck = sub.add_parser("check")
    ck.add_argument("agent")
    a = ap.parse_args(argv)
    if a.cmd == "send":
        text = a.text if a.text is not None else (sys.stdin.read() if a.file == "-" else Path(a.file).read_text())
        if a.local:
            print(f"queued for {a.agent}: {send(a.agent, text)}")
            return 0
        try:
            before = unpulled(a.repo, full_id(a.agent))
        except RuntimeError:
            before = (0, None)
        print(f"pushed for {a.agent}: origin's inbox/{full_id(a.agent)} at {push(a.agent, text, a.repo)[:12]}; its "
              f"shell hook delivers it within a poll interval, but {UNTIL_RESUMED.format(agent=a.agent)}")
        if before[1] and age_s(before[1]) > STALE_S:
            print(f"inbox: NOT DELIVERED YET: {stale_note(a.agent, before[0], before[1])}", file=sys.stderr)
        return 0
    if a.cmd == "status":
        n, oldest = unpulled(a.repo, full_id(a.agent))
        if not n:
            print(f"inbox: origin's inbox/{a.agent} is empty: its VM took every message")
            return 0
        if age_s(oldest) > STALE_S:
            print(f"inbox: NOT DELIVERED: {stale_note(a.agent, n, oldest)}")
            return 1
        print(f"inbox: {n} message(s) on origin's inbox/{a.agent}, the oldest from {oldest}: within its hook's poll interval")
        return 0
    if a.cmd == "check":
        mark, log = INBOX / f"{agent_id(a.agent)}.polled", INBOX / f"{a.agent}.pull.log"
        every = settings.INBOX_POLL_S
        tail = log.read_text(errors="replace").strip().splitlines()[-1:] if log.exists() else []
        if every != "off" and mark.exists() and time.time() - mark.stat().st_mtime < float(every) + 60:
            print(f"inbox: this VM's hook polls for {a.agent} every {every} s" + (f"; its last pull said: {tail[0]}" if tail else ""))
            return 0
        why = ("AGENT_INBOX_POLL_S is off" if every == "off" else
               f"no poll for {a.agent} in {INBOX} within {every} s, so this VM's shell hook doesn't deliver its id")
        print(f"inbox: NOT POLLING: {why}" + (f" (last pull: {tail[0]})" if tail else "")
              + f". Read your messages with `git fetch origin inbox/{a.agent} && git show FETCH_HEAD:{FILE}`")
        return 1
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
