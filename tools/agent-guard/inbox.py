#!/usr/bin/env python3
"""`inbox.py`: tell a running agent something, on this VM or any other. The shell hook (`shell_hook.py`) hands it over.

    python3 tools/agent-guard/inbox.py send AGENT_ID --text 'TEXT'   (or --file PATH; `--file -` reads stdin)
    python3 tools/agent-guard/inbox.py send AGENT_ID --wait S ...    then say whether a VM took it within S seconds; exit 1
                                                                     when none did: its hook is not taking its messages
    python3 tools/agent-guard/inbox.py send AGENT_ID --local ...     only this VM's inbox, no push
    python3 tools/agent-guard/inbox.py pull AGENT_ID [--repo DIR]    move AGENT_ID's pushed messages into this VM's inbox
    python3 tools/agent-guard/inbox.py list                          each agent's pending and delivered messages (a
                                                                     pending one with `since` its oldest message's stamp)
    python3 tools/agent-guard/inbox.py status AGENT_ID [--repo DIR]  whether AGENT_ID's VM has taken its pushed messages;
                                                                     exit 1 when one has waited past STALE_S unpulled
    python3 tools/agent-guard/inbox.py check AGENT_ID [--repo DIR]   on AGENT_ID's own VM: whether this VM's hook polls
                                                                     for it, and whether Cursor reads a shell hook here; exit
                                                                     1, with how to read the branch, if not
    python3 tools/agent-guard/inbox.py guard [AGENT_ID] [--repo DIR] on your own VM: nothing when a shell hook has run for
                                                                     you here; else one line (the guard and the inbox are
                                                                     not running, why, the fix) and exit 1. `research`
                                                                     prints it once a session
    python3 tools/agent-guard/inbox.py selftest [AGENT_ID]           on your own VM ($CURSOR_CONVERSATION_ID by default):
                                                                     the checks of `check`, then send yourself a message your
                                                                     next commands must deliver: delivery end to end

`send` pushes the message as a commit to origin's branch `inbox/AGENT_ID` (one file, `inbox.md`), from the checkout it runs
in (`--repo`). Wherever AGENT_ID runs, its shell hook starts `pull` in the background at most every $AGENT_INBOX_POLL_S
seconds (default 30): `pull` appends the branch's new messages to this VM's inbox and pushes a commit that empties the file,
so a message reaches one VM once. A message waits in `$AGENT_INBOX` (default `/tmp/agent-inbox`) as `AGENT_ID.md`, and
messages sent before delivery queue up in it. The agent's next shell command is refused once with them as the refusal's text,
and the agent re-runs the command, so a pushed message arrives within a poll interval and two of the agent's commands; one
still waiting when its turn completes is the turn's follow-up (`stop_hook.py`). An agent whose turn has ended (FINISHED or
IDLE) runs no command and takes nothing until it is resumed, with Task or a follow-up. `send` says when earlier messages to
the agent are still uncollected on origin, and `send --wait S` whether a VM took this one: taken is on the agent's VM, to be
delivered at its next command, and not yet acted on.
`--local` writes this VM's inbox directly and reaches only agents whose shell commands pass through this VM's hook.

Which hooks a session runs is fixed when its VM boots, by its branch's tracked `.cursor/hooks.json`, and nothing on a
running VM changes it. A shell hook that ran for an agent leaves its event in the guard's event log, or (a copy from before
that log) the inbox's poll mark: `check`, `selftest` and `guard` read both.

AGENT_ID is the agent's conversation id: a cloud agent's or a subagent's is the `bc-…` id its Task call returned.
"""
from __future__ import annotations

import argparse
import calendar
import json
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
#: a pushed message no VM has taken after this long: its target's hook isn't polling (one branch held every message for three
#: hours while each `send` said only "pushed")
STALE_S = 600
#: each git call's bound; the stop hook, which runs inside Cursor's hook timeout, lowers it
GIT_TIMEOUT_S = 120


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
    r = subprocess.run(["git", "-C", str(repo), *args], input=stdin, capture_output=True, text=True, timeout=GIT_TIMEOUT_S)
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


def push(agent: str, text: str, repo: Path, ident: str | None = None) -> str:
    """Append `text` to origin's `inbox/agent` as the message `ident`, retrying while another push moves the branch; the new
    tip."""
    full_id(agent)
    item, err = entry(text, ident or secrets.token_hex(6)), None
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


#: seconds between `send --wait`'s looks at origin, and the longest it may wait: no tool call blocks more than 20 minutes
WAIT_EVERY_S, WAIT_MAX_S = 5.0, 1200.0


def taken(repo: Path, agent: str, ident: str) -> bool:
    """Whether a VM has taken the message `ident` off origin's `inbox/agent`: `pull` empties the branch's file."""
    at = tip(repo, agent)
    return at is None or ident not in HEAD.findall(remote_text(repo, agent, at))


def wait_taken(repo: Path, agent: str, ident: str, max_s: float, sleep=time.sleep, clock=time.monotonic) -> float | None:
    """Seconds until a VM took the message `ident`, looking every WAIT_EVERY_S for `max_s`; None when none took it. A look
    git couldn't make says so and counts as not taken."""
    start = clock()
    while True:
        try:
            if taken(repo, agent, ident):
                return clock() - start
        except RuntimeError as e:
            print(f"inbox: could not read origin's inbox/{agent}: {e}", file=sys.stderr)
        left = max_s - (clock() - start)
        if left <= 0:
            return None
        sleep(min(WAIT_EVERY_S, left))


def age_s(stamp: str, now: float | None = None) -> float:
    return (time.time() if now is None else now) - calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%MZ"))


def stale_note(agent: str, n: int, oldest: str, earlier: bool = False) -> str:
    head = (f"{n} earlier message(s) uncollected since {oldest}: this agent's VM is not pulling" if earlier else
            f"{n} message(s) on origin's inbox/{agent} are still unpulled, the oldest from {oldest}: its VM's shell hook isn't "
            f"polling")
    return (head + " (its turn ended, FINISHED or IDLE, so it runs no shell command until resumed with Task; or its VM runs a "
            f"hook older than the inbox, or none). The agent reads them with `git fetch origin inbox/{agent} && git show "
            f"FETCH_HEAD:{FILE}`, and `inbox.py check {agent}` on its VM says why")


#: a target that ended its turn runs no shell command, so its hook never pulls (a worker finished and marked its PR ready
#: before two stop messages reached it); a sender without the Cursor API can't see that, so `send` always says it
UNTIL_RESUMED = ("it arrives only while {agent}'s turn runs shell commands: if {agent} has ended its turn (FINISHED or IDLE), "
                 "resume it with Task (or a follow-up), which delivers at once")


#: Cursor reads hooks.json as JSONC and strips these before parsing it, inside strings too: a `/*` in a command (a glob such as
#: `dir/*/`) deletes everything up to the next `*/`, and so silently drops or reorders hooks
JSONC_COMMENTS = (re.compile(r"//.*$", re.M), re.compile(r"/\*[\s\S]*?\*/"))


def parsed(text: str) -> object:
    """`text` as JSON, or why it isn't."""
    try:
        return json.loads(text)
    except ValueError as e:
        return f"nothing it can parse ({e})"


def as_cursor_reads(text: str) -> object:
    """The hooks config Cursor takes from a hooks.json's text, or why it takes none."""
    for comment in JSONC_COMMENTS:
        text = comment.sub("", text)
    return parsed(text)


def shell_hook_in(text: str) -> bool:
    """Whether Cursor reads a shell hook from a hooks.json's `text`."""
    read = as_cursor_reads(text)
    hooks = (read.get("hooks") or {}).get("beforeShellExecution") or [] if isinstance(read, dict) else []
    return any("shell_hook.py" in str(h.get("command")) for h in hooks if isinstance(h, dict))


def hooks_file(repo: Path) -> Path | None:
    """The one hooks.json a cloud agent's session reads: the one at the top of the checkout holding `repo`; one in a
    directory above the checkout (a workspace root that holds it) is not read. None when `repo` is in no checkout."""
    try:
        return Path(git(repo, "rev-parse", "--show-toplevel").strip()) / ".cursor/hooks.json"
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        print(f"inbox: {repo.resolve()} is in no checkout: {e}", file=sys.stderr)
        return None


def file_problems(repo: Path) -> list[str]:
    """How Cursor reads the hooks.json of `repo`'s checkout (`hooks_file`) otherwise than as its JSON, and whether it reads
    no shell hook from it."""
    path = hooks_file(repo)
    if path is None or not path.is_file():
        return [f"Cursor reads no shell hook: {repo.resolve()} is in no checkout" if path is None else
                f"Cursor reads no shell hook: there is no {path}"]
    problems, text = [], path.read_text(errors="replace")
    read = as_cursor_reads(text)
    if read != parsed(text):
        steps = (read.get("hooks") or {}) if isinstance(read, dict) else {}
        problems.append(f"Cursor reads {path} otherwise than its JSON says: it strips a `/*` or `//` inside it as a comment"
                        f", leaving {read if isinstance(read, str) else 'the steps ' + ', '.join(sorted(steps))}")
    if not shell_hook_in(text):
        problems.append(f"Cursor reads no shell hook from {path}")
    return problems


def hook_seen(agent: str) -> bool:
    """Whether a shell hook has run for `agent` on this VM: EVENT_LOG holds an event of it, or the inbox its poll mark. Only
    the shell hook writes that mark, and a copy from before the event log writes it too: hook.sh runs origin/main's copy,
    which can be older than the checkout's. Another hook's event shows nothing: from a hooks.json it misreads, Cursor can
    run the Task hook on shell commands and no shell hook."""
    for log in (Path(settings.EVENT_LOG + ".1"), Path(settings.EVENT_LOG)):
        if log.exists() and any('"hook": "shell_hook.py"' in ln and f'"conversation_id": "{agent}"' in ln
                                for ln in log.read_text(errors="replace").splitlines()):
            return True
    return (INBOX / f"{agent}.polled").exists()


#: Cursor reads a VM's hooks once, when the VM boots, from its branch's tracked hooks.json: nothing done on a running VM
#: changes which hooks it runs
AT_BOOT = "A VM reads its hooks once, when it boots, from its branch's .cursor/hooks.json"


def hook_problems(agent: str, repo: Path) -> list[str]:
    """Why this VM's shell hook may not run for `agent`: what `file_problems` finds at `repo`, and no trace of a shell hook
    run for `agent` (`hook_seen`), which this command's own hook would have left."""
    problems = file_problems(repo)
    if not hook_seen(agent):
        problems.append(f"no shell hook has run for {agent} on this VM: {settings.EVENT_LOG} has no event of it and {INBOX} no "
                        f"poll mark of it, though this command's own hook would have left both. {AT_BOOT}: only a VM booted "
                        "after that file is fixed runs it")
    return problems


def guard_off(agent: str, repo: Path) -> str | None:
    """One line for `agent` on its own VM when no shell hook has run for it here: the guard and the inbox are not running,
    why (its branch's hooks.json, read at `repo`'s checkout), and the fix; None when one has run."""
    if hook_seen(agent):
        return None
    head = (f"no shell hook has run for {agent} on this VM, so the guard that refuses a command printing a secret and the "
            f"inbox are not running here. {AT_BOOT}")
    if hooks_file(repo) is None:
        return f"{head}. Run `python3 tools/agent-guard/inbox.py guard {agent}` in your checkout for why, and the fix"
    problems = file_problems(repo)
    try:
        main = git(repo, "show", "origin/main:.cursor/hooks.json")
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        main = f"unread ({e})"
    new_vm = "a VM booted from the branch after that runs them"
    if not problems:
        why = "this branch's loads as written now, so it changed after this VM booted, or Cursor did not load it"
        fix = f"Push the branch: {new_vm}"
    elif shell_hook_in(main) and as_cursor_reads(main) == parsed(main):
        why, fix = problems[0], f"Merge origin/main, whose hooks.json loads, and push: {new_vm}"
    else:
        why, fix = problems[0], ("origin/main's does not load either, as of this checkout's last fetch: fetch it, merge it "
                                 f"once it loads, and push: {new_vm}")
    return (f"{head}: {why}. {fix}. Until then `python3 tools/agent-guard/inbox.py check {agent}` says how to read your "
            "messages by hand")


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


def main(argv: list[str] | None = None, sleep=time.sleep, clock=time.monotonic) -> int:
    """`sleep` and `clock` are what `send --wait` waits by."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send")
    s.add_argument("agent")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--file")
    s.add_argument("--local", action="store_true", help="this VM's inbox only, no push")
    s.add_argument("--wait", type=float, metavar="S", help=f"then say whether a VM took it within S seconds (at most "
                                                           f"{WAIT_MAX_S:g}); exit 1 when none did")
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
    ck.add_argument("--repo", type=Path, default=Path("."))
    for name in ("guard", "selftest"):
        sf = sub.add_parser(name)
        sf.add_argument("agent", nargs="?", default=settings.CONVERSATION_ID)
        sf.add_argument("--repo", type=Path, default=Path("."))
    a = ap.parse_args(argv)
    if a.cmd == "guard":
        line = guard_off(full_id(a.agent), a.repo)
        if line:
            print(line)
        return 1 if line else 0
    if a.cmd == "selftest":
        problems = hook_problems(full_id(a.agent), a.repo)
        token = secrets.token_hex(4)
        at = push(a.agent, f"inbox self-check {token}: a message you sent yourself reached you, so this VM delivers your "
                           "inbox. Nothing to do.", a.repo)
        (INBOX / f"{a.agent}.polled").unlink(missing_ok=True)
        for p in problems:
            print(f"inbox: PROBLEM: {p}")
        print(f"inbox: self-check {token} pushed to origin's inbox/{a.agent} at {at[:12]}. Run `sleep 10`, then `true`: the "
              f"`true` should be refused with \"inbox self-check {token}\", which is delivery end to end. If it isn't, "
              f"`inbox.py check {a.agent}` says why.")
        return 1 if problems else 0
    if a.cmd == "send":
        text = a.text if a.text is not None else (sys.stdin.read() if a.file == "-" else Path(a.file).read_text())
        if a.local:
            print(f"queued for {a.agent}: {send(a.agent, text)}")
            return 0
        if a.wait is not None and not 0 < a.wait <= WAIT_MAX_S:
            raise SystemExit(f"inbox: --wait {a.wait:g} is not within 0 to {WAIT_MAX_S:g} seconds (no tool call blocks more than "
                             "20 minutes)")
        try:
            before = unpulled(a.repo, full_id(a.agent))
        except RuntimeError:
            before = (0, None)
        ident = secrets.token_hex(6)
        print(f"pushed for {a.agent}: origin's inbox/{full_id(a.agent)} at {push(a.agent, text, a.repo, ident)[:12]}; its "
              f"shell hook delivers it within a poll interval, but {UNTIL_RESUMED.format(agent=a.agent)}"
              + ("" if a.wait is not None else ". `--wait S` says whether its VM took it within S seconds"), flush=True)
        n, oldest = before
        if n and oldest and age_s(oldest) > STALE_S:
            print(f"inbox: NOT DELIVERED YET: {stale_note(a.agent, n, oldest, earlier=True)}", file=sys.stderr)
        elif n and oldest:
            print(f"inbox: {n} earlier message(s) uncollected since {oldest}; unpulled past {STALE_S // 60} min means "
                  f"{a.agent}'s VM is not pulling (`inbox.py status {a.agent}`)", file=sys.stderr)
        if a.wait is None:
            return 0
        waited = wait_taken(a.repo, a.agent, ident, a.wait, sleep, clock)
        if waited is None:
            print(f"inbox: NOT TAKEN in {a.wait:g} s: no VM pulled {ident} off origin's inbox/{a.agent}, so its hook is not "
                  f"taking its messages now: {a.agent} is inside one long command or has ended its turn, or its VM runs no "
                  f"shell hook (`inbox.py guard {a.agent}` on its VM says). It stays on the branch for a later pull; to "
                  "stop its work now, cancel its runs yourself", file=sys.stderr)
            return 1
        print(f"inbox: taken {waited:.0f} s after the push: {a.agent}'s VM pulled {ident}, and its next shell command "
              "delivers it. Taken is not acted on: its runs and its issue show that")
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
        problems = hook_problems(a.agent, a.repo)
        for p in problems:
            print(f"inbox: PROBLEM: {p}")
        if every != "off" and mark.exists() and time.time() - mark.stat().st_mtime < float(every) + 60:
            print(f"inbox: this VM's hook polls for {a.agent} every {every} s" + (f"; its last pull said: {tail[0]}" if tail else ""))
            return 1 if problems else 0
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
