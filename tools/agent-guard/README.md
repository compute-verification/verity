# The agent guard

What stands between an agent on a Cursor VM and three mistakes: printing a secret, running a heavy build on the VM it
shares, and blocking in one call so long that it misses its messages. It also carries messages to a running agent and
records the VM's OOM kills. Nothing in `verity/` reads it. The scripts are standard library only and run as files,
side by side; it is not a package.

## What other code loads

| File | What it is | Named by |
|---|---|---|
| `boot.sh` | Puts the guard on a VM whatever branch is checked out: it writes `.cursor/hooks.json` where the branch tracks none, copies origin/main's guard out of the checkout, runs `install.sh` and starts `oom_watch.py`. | The Cloud Agent environment's start, on every boot; the team skill; `infra/cursor` |
| `hook.sh` | `hook.sh NAME` runs origin/main's copy of the hook NAME on Cursor's event, so a branch cut before a guard landed still runs it. | `.cursor/hooks.json`, for every hook |
| `shell_hook.py` | The shell hook. It refuses a command that would print secrets, a local build or unselected test run, a call that would block more than 20 minutes, and a branch switch or push that belongs to another agent. It delivers a waiting inbox message by refusing the next command once. | `hook.sh`; the root's tests |
| `inbox.py` | Messages to a running agent, on this VM or another. | The team skill; `research digest` |
| `research.sh` | The `research` on an agent VM's PATH: it runs origin/main's `tools/research/src/research/at_main.py`, which runs the checkout's `research` or, when the checkout is only behind, origin/main's. | `install.sh`; `at_main.py` |

The rest: `task_hook.py` (a coordinator never waits on a local worker in the foreground, and a cloud agent writes
Linear through `research msg`), `stop_hook.py` (a message still waiting when a turn ends becomes its follow-up),
`worktree_guard.py` (one agent per checkout), `env_guard.py` and `git_guard.sh` (an `env` that redacts values and a
`git` that keeps credentials out of remote URLs, which `install.sh` puts ahead of the system's), and `settings.py`
(every environment variable the hooks share, read once).

## Commands

~~~text
git -C /workspace show origin/main:tools/agent-guard/boot.sh | bash -s -- /workspace    put the guard on this VM
python3 tools/agent-guard/inbox.py send AGENT_ID --text 'TEXT'     tell a running agent something
python3 tools/agent-guard/inbox.py list                            each agent's pending and delivered messages
python3 tools/agent-guard/inbox.py check AGENT_ID                  on its own VM: whether its hook polls for it
python3 tools/agent-guard/worktree_guard.py hand BRANCH AGENT_ID   give a branch you created to another agent
python3 tools/agent-guard/worktree_guard.py list                   each worktree's owner
python3 tools/agent-guard/oom_watch.py scan                        the OOM kills still in the kernel's buffer
uv run tools/verity/check/suites.py tools/agent-guard              its tests
~~~

A heavy command the shell hook refuses runs under `research run --on auto` instead.
