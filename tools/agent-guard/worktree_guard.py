#!/usr/bin/env python3
"""One agent per checkout (INF-59; the rule is AGENTS.md's, #2416): the shell hook's judgment of `git checkout`, `git switch` and
`git push`, and the branch registry it keeps.

    python3 tools/agent-guard/worktree_guard.py hand BRANCH AGENT_ID   give a branch you created to another agent (its new owner)
    python3 tools/agent-guard/worktree_guard.py list                   each worktree's owner and each registered branch's

A worktree belongs to the first agent (Cursor conversation) that runs a git command in it: the coordinator's `/workspace`, or
the worktree a worker made with `git worktree add`. The owner is stamped in the worktree's git dir (`agent-owner`), and the
branch it had then is registered to it. The hook then refuses, from any other agent:
  - switching that worktree's branch (`git checkout BRANCH|-b|-B`, `git switch`), since everything installed editable from it
    (the `research` CLI) follows the switch; checking out files (`git checkout -- PATH`, `git checkout REF -- PATH`) passes;
  - pushing a branch registered to another agent: one it created (`checkout -b`, `switch -c`, `worktree add -b`) or the one its
    worktree had when it claimed it, until that agent hands it over (`hand`). An unregistered branch passes.
The registry is `$AGENT_GUARD_STATE/branches.json` (default /tmp/agent-guard), one VM's: an agent on a VM of its own is never
judged. Where a command names no directory (`git -C DIR`, or a `cd DIR` before it in the same command) the hook judges the
event's cwd, else the workspace root, since the shell's remembered directory never reaches it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

STATE = Path(os.environ.get("AGENT_GUARD_STATE", "/tmp/agent-guard"))
OWNER_FILE = "agent-owner"


def git(directory: str, *args: str) -> str | None:
    try:
        p = subprocess.run(["git", "-C", directory, *args], capture_output=True, text=True, timeout=5,
                           env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def registry() -> dict[str, str]:
    try:
        d = json.loads((STATE / "branches.json").read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def register(branch: str, agent: str, *, replace: bool = False) -> None:
    reg = registry()
    if branch and (replace or branch not in reg):
        reg[branch] = agent
        STATE.mkdir(parents=True, exist_ok=True)
        tmp = STATE / f"branches.json.{os.getpid()}"
        tmp.write_text(json.dumps(reg, indent=1, sort_keys=True))
        os.replace(tmp, STATE / "branches.json")


def owner(gitdir: str, agent: str, directory: str) -> str:
    """The worktree's owner, claiming it (and registering its branch) for `agent` when it has none."""
    f = Path(gitdir) / OWNER_FILE
    try:
        return f.read_text().strip() or agent
    except OSError:
        pass
    try:
        with open(f, "x") as out:
            out.write(agent + "\n")
    except FileExistsError:
        return f.read_text().strip() or agent
    except OSError:
        return agent
    register(git(directory, "symbolic-ref", "--short", "-q", "HEAD") or "", agent)
    return agent


def invocations(commands: list[list[str]], cwd: str) -> list[tuple[str, list[str]]]:
    """(directory, git arguments after its global options) of each `git` among `commands` (unwrapped simple commands), tracking
    `cd DIR` and `git -C DIR`."""
    out = []
    for words in commands:
        if not words:
            continue
        if words[0] == "cd" and len(words) > 1:
            cwd = os.path.normpath(os.path.join(cwd, os.path.expanduser(words[1])))
            continue
        if os.path.basename(words[0]) != "git":
            continue
        d, args = cwd, words[1:]
        while args and args[0].startswith("-"):
            if args[0] == "-C" and len(args) > 1:
                d, args = os.path.normpath(os.path.join(d, os.path.expanduser(args[1]))), args[2:]
            elif args[0] in ("-c", "--git-dir", "--work-tree", "--namespace") and len(args) > 1:
                args = args[2:]
            else:
                args = args[1:]
        if args:
            out.append((d, args))
    return out


def created_branch(sub: str, args: list[str]) -> str | None:
    """The branch a checkout/switch/worktree add creates, if any."""
    flags = {"checkout": ("-b", "-B"), "switch": ("-c", "-C", "--create", "--force-create"), "worktree": ("-b", "-B")}.get(sub, ())
    for i, a in enumerate(args[:-1]):
        if a in flags:
            return args[i + 1]
    return None


def switches_branch(directory: str, sub: str, args: list[str]) -> bool:
    if sub == "switch":
        return True
    if sub != "checkout" or "--" in args:
        return False
    if created_branch(sub, args):
        return True
    rest = [a for a in args if not a.startswith("-")]
    return len(rest) == 1 and git(directory, "rev-parse", "--verify", "-q", rest[0] + "^{commit}") is not None


def push_targets(directory: str, args: list[str]) -> list[str]:
    rest = [a for a in args if not a.startswith("-")]
    specs = rest[1:]
    current = git(directory, "symbolic-ref", "--short", "-q", "HEAD")
    out = []
    for s in specs or ["HEAD"]:
        dst = s.lstrip("+").split(":")[-1] or s.lstrip("+").split(":")[0]
        dst = current if dst == "HEAD" else dst
        if dst:
            out.append(dst.removeprefix("refs/heads/"))
    return out


def reason(commands: list[list[str]], agent: str, cwd: str) -> str | None:
    """Why the agent may not run these git commands here, or None. Registers the branches an allowed command creates."""
    if not agent:
        return None
    for d, args in invocations(commands, cwd):
        gitdir = git(d, "rev-parse", "--absolute-git-dir")
        if gitdir is None:
            continue
        sub = args[0]
        mine = owner(gitdir, agent, d)
        if sub in ("checkout", "switch") and switches_branch(d, sub, args[1:]) and mine != agent:
            top = git(d, "rev-parse", "--show-toplevel") or d
            return (f"{top} is agent {mine}'s checkout, and switching its branch moves everything installed from it under that "
                    f"agent (the `research` CLI follows it). Work in a checkout of your own: `git -C {top} worktree add "
                    f"../wt-<name> -b <branch>` (or `git worktree add ../wt-<name> <existing-branch>`), then run git there "
                    f"(`git -C ../wt-<name> ...`, or `cd ../wt-<name> && ...` in the same command). One agent per checkout "
                    f"(AGENTS.md).")
        if sub == "push":
            reg = registry()
            for b in push_targets(d, args[1:]):
                if reg.get(b, agent) != agent:
                    return (f"branch {b} belongs to agent {reg[b]} (it created it, or its checkout had it): push a branch of "
                            f"your own (`git push -u origin <your-branch>`), or have its owner hand it to you: `python3 "
                            f"tools/agent-guard/worktree_guard.py hand {b} {agent}`.")
        if (b := created_branch(sub, args[1:])) and sub in ("checkout", "switch", "worktree"):
            register(b, agent)
    return None


def main(argv: list[str]) -> int:
    if argv[:1] == ["hand"] and len(argv) == 3:
        register(argv[1], argv[2], replace=True)
        print(f"worktree_guard: {argv[1]} is now {argv[2]}'s")
        return 0
    if argv[:1] == ["list"]:
        print(json.dumps(registry(), indent=1, sort_keys=True))
        return 0
    print(__doc__.split("\n\n")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
