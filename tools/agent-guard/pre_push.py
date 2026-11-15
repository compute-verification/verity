#!/usr/bin/env python3
"""git's `pre-push` hook on an agent VM: the branch rule the shell hook applies to a typed `git push` (worktree_guard.py),
applied to every branch a push updates, whoever runs it. The shell hook reads only the command line an agent types, so a
`git push` inside a script, a tool or any child process reaches this hook alone.

`install.sh` writes `/usr/local/lib/agent-guard/hooks/pre-push`, which runs this file, and the `git` it puts ahead of
/usr/bin (git_guard.sh) gives every `git push` on the VM `-c core.hooksPath=` that directory (`AGENT_GUARD_GIT_HOOKS` names
another), in every worktree and clone, after the caller's own `-c` and with `--no-verify` dropped. The hook then runs the
pre-push hook git would have run without it (`AGENT_GUARD_CHAIN_HOOKS`, the hooks directory the push had: none under
`--no-verify` or `core.hooksPath=/dev/null`), with the same arguments and input, and git takes that hook's verdict.

The pusher is the agent whose shell started it: `CURSOR_CONVERSATION_ID`, which Cursor sets in every agent's shell and every
process under it inherits. A push is judged when that is set and it goes to a repository whose `main`, `checked` and `next`
drive lands, compute-verification/verity or compute-verification/infra on GitHub (`worktree_guard.queue_remote`: the URL git
hands the hook, whole, in https, ssh or scp form, with or without `.git` and a credential; `AGENT_GUARD_PUSH_REMOTES`, a
regular expression, names others in their place): for each `refs/heads/` branch it creates, moves, forces or deletes,
the push is refused when the branch is `main`, `checked` or `next` (a lander's repair of `next` alone passes with
`AGENT_GUARD_PUSH_NEXT_OK='REASON'` in its environment, logged as the typed one is), or registered to another agent
(`$AGENT_GUARD_STATE/branches.json`) that hasn't handed it over; an unregistered branch passes. Without
`CURSOR_CONVERSATION_ID` nothing is judged: no agent started the push (a person, a boot script), and the queue's pushes on
the control pod, CI tooling and a person's laptop run no wrapper and so no hook at all. A push to another repository is not
judged either: the notes' `main`, which every agent's `research notes sync` pushes, a lane's notes branch, a fork, a node's bare
repository, a scratch clone. The URL can hold a token, so nothing prints it.
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.dont_write_bytecode = True                      # no __pycache__ beside the hook: boot.sh copies its directory file by file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worktree_guard  # noqa: E402


def branches(lines: list[str]) -> list[str]:
    """The branches git's pre-push lines (`LOCAL_REF LOCAL_SHA REMOTE_REF REMOTE_SHA`) update, deletions included."""
    out = []
    for line in lines:
        fields = line.split()
        if len(fields) == 4 and fields[2].startswith("refs/heads/"):
            out.append(fields[2].removeprefix("refs/heads/"))
    return out


def reason(remote: str, url: str, lines: list[str], env: dict[str, str]) -> str | None:
    """Why this push may not run (module doc), or None."""
    agent = env.get("CURSOR_CONVERSATION_ID", "").strip()
    if not agent or not worktree_guard.queue_remote(url, env.get("AGENT_GUARD_PUSH_REMOTES")):
        return None
    pushed = branches(lines)
    targets = worktree_guard.PROTECTED & set(pushed)
    if targets and not worktree_guard.next_repaired(targets, env.get("AGENT_GUARD_PUSH_NEXT_OK", ""),
                                                    {"agent": agent, "dir": os.getcwd(), "remote": remote, "by": "pre-push"}):
        return worktree_guard.PROTECTED_REASON
    return worktree_guard.owner_reason(pushed, agent)


def chained(argv: list[str], data: bytes, env: dict[str, str]) -> int:
    """The verdict of the pre-push hook git would have run without this one (`AGENT_GUARD_CHAIN_HOOKS`), 0 when there is none
    or it is this one again."""
    hooks = env.get("AGENT_GUARD_CHAIN_HOOKS", "")
    hook = os.path.join(hooks, "pre-push") if hooks else ""
    if not hook or not os.path.isfile(hook) or not os.access(hook, os.X_OK):
        return 0
    return subprocess.run([hook, *argv], input=data, check=False, env={**env, "AGENT_GUARD_CHAIN_HOOKS": ""}).returncode


def main(argv: list[str]) -> int:
    data = sys.stdin.buffer.read()
    env = dict(os.environ)
    try:
        why = reason(argv[0] if argv else "", argv[1] if len(argv) > 1 else "", data.decode("utf-8", "replace").splitlines(), env)
    except (OSError, ValueError) as e:                 # as the shell hook's checkout guard does: a bug here stops no push
        print(f"pre-push (tools/agent-guard/pre_push.py): not judged: {type(e).__name__}: {e}", file=sys.stderr)
        why = None
    if why:
        print(f"pre-push (tools/agent-guard/pre_push.py): refused: {why}", file=sys.stderr)
        return 1
    return chained(argv, data, env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
