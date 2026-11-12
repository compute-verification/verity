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
| `shell_hook.py` | The shell hook. It refuses a command that would print secrets (the environment, a key or env file, or any process's full command line, whatever it is piped into), a local build or unselected test run, a call that would block more than 20 minutes, a branch switch that belongs to another agent, and a push to verity or infra that updates `main`, `checked`, `next` or another agent's branch; each rule also reads what the command has another program run: what tmux types into a pane (`send-keys`) or starts (`new-session`, `new-window`, `split-window`, `respawn-pane`, `run-shell` and the like), and what `bash -c`, `env`, `xargs`, `eval`, `watch`, `nohup`, `setsid` or `timeout` runs. A script file's contents are not read. A listing that another machine runs over `ssh` or `research pods ssh` is refused unfiltered and passes when piped straight into a `grep` that keeps only matching lines. It delivers a waiting inbox message by refusing the next command once. | `hook.sh`; the root's tests |
| `pre_push.py` | git's `pre-push` hook: the branch rule a typed push meets, applied to every branch a push updates, so a push from a script or a child process meets it too. It judges a push to `compute-verification/verity` or `compute-verification/infra`, the repositories whose `main`, `checked` and `next` drive lands, by an agent's process (one with `CURSOR_CONVERSATION_ID`, which every agent's shell sets and its children inherit): refused when it updates `main`, `checked` or `next`, or a branch registered to another agent (`worktree_guard.py hand` hands one over); an unregistered branch passes. A push without that id (a person's, the queue's) and one to any other repository (the notes, whose `main` every agent's `research notes sync` pushes, a fork, a node) are not judged. It then runs the hook the push would have run without it. | `install.sh` writes `/usr/local/lib/agent-guard/hooks/pre-push`, which `git_guard.sh` sets as `core.hooksPath` for every `git push`, through `--no-verify` and a caller's own `core.hooksPath` |
| `procs.py` | This machine's processes, with each command line's token-like runs cut out: what the shell hook's refusal of a listing points to, beside `ps -eo pid,etimes,comm` and `pgrep -l`. | The shell hook's refusal |
| `inbox.py` | Messages to a running agent, on this VM or another. | The team skill; `research digest` |
| `research.sh` | The `research` on an agent VM's PATH: it runs origin/main's `tools/research/src/research/at_main.py`, which runs the checkout's `research` or, when the checkout is only behind, origin/main's. | `install.sh`; `at_main.py` |

The rest: `task_hook.py` (a coordinator never waits on a local worker in the foreground, and a cloud agent writes
Linear through `research msg`), `stop_hook.py` (a message still waiting when a turn ends becomes its follow-up),
`worktree_guard.py` (one agent per checkout), `env_guard.py` and `git_guard.sh` (an `env` that redacts values and a
`git` that keeps credentials out of remote URLs and gives every push `pre_push.py`, which `install.sh` puts ahead of
the system's), and `settings.py` (every environment variable the hooks share, read once).

The push rules, typed (`worktree_guard.py`) or in `pre_push.py`, cover the queue's two repositories alone
(`worktree_guard.queue_remote`: their GitHub URLs exactly, in https, ssh or scp form, with or without `.git` and a credential;
`AGENT_GUARD_PUSH_REMOTES` names others in their place). The registry is the VM's own and names a branch without its
repository. The typed rule reads a push's words as git does (options anywhere, by any prefix, with their values; git's
own `-C`, `-c` and the like), and places a push by three things only: the directory the hook is handed, git's own `-C`,
and a URL or path given as its repository. It follows no `cd`: a `cd`, `pushd`, `popd` or `source` anywhere in the
command, run or not, or a repository named by `GIT_DIR`, `GIT_WORK_TREE` or `--git-dir`, leaves the push unplaced
(`git -C <checkout> push ...` places it): such a push that names no branch is refused for that, with the push it accepts
(`git -C <checkout> push -u origin HEAD`, or the branch by name). The branch-switch rule refuses a switch it can't place. It judges as a push
to the queue's repositories one it can't place, or whose first word is neither a remote, a URL nor a path, an option git
doesn't know, or git configuration set elsewhere in the command. It can't see what earlier commands left in the shell (an
exported `GIT_DIR`, an alias); `pre_push.py`, which reads the URL git pushes to, can. Not covered by the
push rule: a push that runs `/usr/bin/git` by path, a git alias for `push`, `VERITY_GIT_GUARD=off`, and a client other
than git. On a git before 2.31 a push is judged but runs no other pre-push hook.

## Commands

~~~text
git -C /workspace show origin/main:tools/agent-guard/boot.sh | bash -s -- /workspace    put the guard on this VM
python3 tools/agent-guard/inbox.py send AGENT_ID --text 'TEXT'     tell a running agent something
python3 tools/agent-guard/inbox.py list                            each agent's pending and delivered messages
python3 tools/agent-guard/inbox.py check AGENT_ID                  on its own VM: whether its hook polls for it
python3 tools/agent-guard/procs.py [PATTERN]                       processes and their command lines, tokens cut
python3 tools/agent-guard/worktree_guard.py hand BRANCH AGENT_ID   give a branch you created to another agent
python3 tools/agent-guard/worktree_guard.py list                   each worktree's owner
python3 tools/agent-guard/oom_watch.py scan                        the OOM kills still in the kernel's buffer
uv run tools/verity/check/suites.py tools/agent-guard              its tests
~~~

A heavy command the shell hook refuses runs under `research run --on auto` instead.
