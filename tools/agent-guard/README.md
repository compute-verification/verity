# The agent guard

An agent on a Cursor VM shares that VM, and its shell, with secrets and with other agents. The agent guard is what stands
between such an agent and three mistakes: printing a secret, running a heavy build on the VM it shares, and blocking in
one call so long that it misses its messages. Cursor runs it as a hook on every shell command, and it refuses a command
that would make one of those mistakes, saying what to run instead. It also keeps each branch and checkout to the agent
that made it, keeps a commit under the machine's own identity, carries messages to a running agent, and records the VM's
OOM kills.

Nothing in `verity/` reads it. The scripts are the standard library alone and run as files, side by side; it is not a
package.

## Try it

~~~sh
python3 tools/agent-guard/procs.py python3
~~~

This lists this machine's `python3` processes with their command lines, every token-like run cut out. It is what the
shell hook points to when it refuses a command that would print a process's full command line, and it needs nothing.

## Commands

~~~text
git -C /workspace show origin/main:tools/agent-guard/boot.sh | bash -s -- /workspace    put the guard on this VM
python3 tools/agent-guard/inbox.py send AGENT_ID --text 'TEXT'     tell a running agent something
python3 tools/agent-guard/inbox.py list                            each agent's pending and delivered messages
python3 tools/agent-guard/inbox.py check AGENT_ID                  on its own VM: whether its hook polls for it
python3 tools/agent-guard/inbox.py turns [AGENT_ID]                on its own VM: the model each of its turns names in the event log
python3 tools/agent-guard/procs.py [PATTERN]                       processes and their command lines, tokens cut
python3 tools/agent-guard/worktree_guard.py hand BRANCH AGENT_ID   give a branch you created to another agent
python3 tools/agent-guard/worktree_guard.py reclaim DIR            a coordinator: take back a gone agent's checkout
python3 tools/agent-guard/worktree_guard.py list                   each registered branch's owner
python3 tools/agent-guard/oom_watch.py scan                        the OOM kills still in the kernel's buffer
tmux new-session -d -s vm-secrets                                  the session this VM keeps, by hand (boot starts it)
uv run tools/check/suites.py tools/agent-guard              its tests
~~~

A heavy command the shell hook refuses runs under `research run --on auto` instead.

[`DESIGN.md`](DESIGN.md) says what each file is and who loads it, and states every rule the hooks keep: what each refuses,
how it reads a command, and what it does not cover.
