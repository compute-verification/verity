# The agent guard

What stands between an agent on a Cursor VM and three mistakes: printing a secret, running a heavy build on the VM it
shares, and blocking in one call so long that it misses its messages. It also carries messages to a running agent and
records the VM's OOM kills. Nothing in `verity/` reads it. The scripts are standard library only and run as files,
side by side; it is not a package.

## What other code loads

| File | What it is | Named by |
|---|---|---|
| `boot.sh` | Puts the guard on a VM whatever branch is checked out: it writes `.cursor/hooks.json` where the branch tracks none, copies origin/main's guard out of the checkout, runs `install.sh`, and starts `oom_watch.py` and the tmux session the VM keeps (`kept_session.py start`). | The Cloud Agent environment's start, on every boot; the team skill; `infra/cursor` |
| `hook.sh` | `hook.sh NAME` runs origin/main's copy of the hook NAME on Cursor's event, so a branch cut before a guard landed still runs it. | `.cursor/hooks.json`, for every hook |
| `shell_hook.py` | The shell hook. It refuses a command that would print secrets (the environment, a key or env file, or any process's full command line, whatever it is piped into), a local build or unselected test run, a call that would block more than 20 minutes, a branch switch that belongs to another agent, a push to verity or infra that updates `main`, `checked`, `next` or another agent's branch, a `git` in a checkout of verity or infra that names an author or committer of its own, a `git config --global` that writes the machine's identity, a command that leaves an author's or committer's name set in the shell (`export GIT_AUTHOR_NAME=...`), and a `tmux kill-server` or `tmux kill-session` that would end the VM's tmux server or the session it keeps; each rule also reads what the command has another program run: what tmux types into a pane (`send-keys`) or starts (`new-session`, `new-window`, `split-window`, `respawn-pane`, `run-shell` and the like), and what `bash -c`, `env`, `xargs`, `eval`, `watch`, `nohup`, `setsid` or `timeout` runs. A script file's contents are not read. A listing that another machine runs over `ssh` or `research pods ssh` is refused unfiltered and passes when piped straight into a `grep` that keeps only matching lines. It delivers a waiting inbox message by refusing the next command once. | `hook.sh`; the root's tests |
| `pre_push.py` | git's `pre-push` hook: the branch rule a typed push meets, applied to every branch a push updates, so a push from a script or a child process meets it too. It judges a push to `compute-verification/verity` or `compute-verification/infra`, the repositories whose `main`, `checked` and `next` drive lands, by an agent's process (one with `CURSOR_CONVERSATION_ID`, which every agent's shell sets and its children inherit): refused when it updates `main`, `checked` or `next`, or a branch registered to another agent (`worktree_guard.py hand` hands one over); an unregistered branch passes. A push without that id (a person's, the queue's) and one to any other repository (the notes, whose `main` every agent's `research notes sync` pushes, a fork, a node) are not judged. It then runs the hook the push would have run without it. | `install.sh` writes `/usr/local/lib/agent-guard/hooks/pre-push`, which `git_guard.sh` sets as `core.hooksPath` for every `git push`, through `--no-verify` and a caller's own `core.hooksPath` |
| `kept_session.py` | The tmux session a VM keeps, `vm-secrets`, so that its tmux server never ends: the server's environment holds the secrets it started with, and a shell that lost them takes them back from it. `start` starts it; the shell hook's rule against ending it is here too. | `boot.sh`; the shell hook; `research doctor`, whose `secrets` check names the session |
| `procs.py` | This machine's processes, with each command line's token-like runs cut out: what the shell hook's refusal of a listing points to, beside `ps -eo pid,etimes,comm` and `pgrep -l`. | The shell hook's refusal |
| `inbox.py` | Messages to a running agent, on this VM or another. | The team skill; `research digest` |
| `research.sh` | The `research` on an agent VM's PATH: it runs origin/main's `tools/research/src/research/at_main.py`, which runs the checkout's `research` or, when the checkout is only behind, origin/main's. | `install.sh`; `at_main.py` |

The rest: `task_hook.py` (a coordinator never waits on a local worker in the foreground, and a cloud agent writes
Linear through `research msg`; each Task call's verdict, model and resume flag, never its prompt, go to
`$AGENT_GUARD_TASK_LOG`, default `/tmp/agent-guard-task.jsonl`), `stop_hook.py` (it offers a message still waiting when a
turn ends to the runtime as a follow-up, which no delivery relies on: a message reaches an agent at its next shell
command, and one its VM holds unshown when its turn ends may be gone by its next turn, so an agent whose turn has ended
is resumed with the text itself), `worktree_guard.py` (one agent per checkout), `env_guard.py` and `git_guard.sh` (an
`env` that redacts values and a `git` that keeps credentials out of remote URLs and gives every push `pre_push.py`,
which `install.sh` puts ahead of the system's), and `settings.py` (every environment variable the hooks share, read once).

The event log (`settings.record_event`; `$AGENT_GUARD_EVENT_LOG`, default `/tmp/agent-guard-events.jsonl`) holds one line
for each run of the shell, Task and stop hooks: when, which hook, the event's name, its conversation id and tool name, the
payload's key names, and of the payload's other values only `model` and `generation_id` (the turn's id), each where the
payload has it as a string, cut at 64 characters; never the command, a path or an address. Past about 4 MB it is renamed
to `.1` and started again. `inbox.py check`, `selftest` and `guard` read from it whether a shell hook has run for an agent,
and `inbox.py turns` the model each of an agent's turns names there. Each VM's log is its own.

A checkout belongs to the first agent that runs git in it, and the stamp outlives the agent. `worktree_guard.py reclaim
DIR` gives the checkout holding DIR, and every branch registered to its owner, to the VM's coordinator once that owner
is gone. Both tests read what the VM records and nothing the caller says, so a running worker is never evicted on a
guess and a worker can't take another's checkout. The coordinator is an agent that holds a handle in origin/main's
Slack registry, which the hook and the command both read (a checkout's own copy decides only where origin/main can't
be read): the hook takes the caller from Cursor's event, which no command line sets, and refuses a `reclaim` typed by
any other agent; the command stamps only an id that holds a handle, so a made-up id gives the checkout to a
coordinator, never to the agent that made it up. Gone is no hook run in two hours, six times the 20 minutes one
command may block, in the record every hook run adds a line to (`settings.EVENT_LOG`: each shell command, each Task,
dynamic-tool and MCP call, each turn's end). The record must reach back those two hours, since it rotates by size, and
must hold a shell-hook run of the caller's within them, since where no shell hook runs every agent would read as gone;
short of either, nothing is taken back. A worker whose turn ended over two hours ago reads as gone: its files stay, and
`hand` gives a branch back. A refused branch switch or push names `reclaim` to a coordinator once it would succeed.

The push rules, typed (`worktree_guard.py`) or in `pre_push.py`, cover the queue's two repositories alone
(`worktree_guard.queue_remote`: their GitHub URLs exactly, in https, ssh or scp form, with or without `.git` and a credential;
`AGENT_GUARD_PUSH_REMOTES` names others in their place). The registry is the VM's own and names a branch without its
repository. `pre_push.py` reads the commits such a push adds as well, whatever made them: it refuses one that adds a
commit whose author or committer is not the checkout's configured identity (`git config user.name` and `git config
user.email` where the push runs), names the commits, and says how to make them again on a new branch. A push adds what
the remote lacks as the checkout knows it, the commits reachable from what it pushes and from no remote-tracking ref of
that remote, so a merge of the remote's `main` adds the merge alone and `git fetch` comes first where a ref is stale; a
push that adds commits from a checkout with no configured identity, or whose commits git can't list, is refused too.
Where a node made the commits, the run that made them names this checkout's identity, as `tools/move/README.md` shows.
One other identity passes (`OTHERS`), the queue's own: `research queue cross` makes a crossing's commits under it and
pushes them with the caller's git, so an agent's crossing meets this rule, and a test of research's holds the entry to
the queue's definition. A head crossed by hand under any other name is refused.
The typed rule reads a push's words as git does (options anywhere, by any prefix, with their values; git's
own `-C`, `-c` and the like), and places a push by three things only: the directory the hook is handed, git's own `-C`,
and a URL or path given as its repository. It follows no `cd`: a `cd`, `pushd` or `popd` anywhere in the command, run
or not, leaves the push unplaced unless its own `-C` is absolute (`cd /tmp && git -C /abs/checkout push ...` runs in
`/abs/checkout` wherever the shell is; a relative `-C` or one through `$` does not). A `source` or `.`, which can change
what git reads as well as where it runs, or a repository named by `GIT_DIR`, `GIT_WORK_TREE` or `--git-dir`, leaves it
unplaced whatever its `-C`. An unplaced push that names no branch is refused for that, with the push it accepts
(`git -C <checkout> push -u origin HEAD`, or the branch by name). The branch-switch rule refuses a switch it can't place. It judges as a push
to the queue's repositories one it can't place, or whose first word is neither a remote, a URL nor a path, an option git
doesn't know, or git configuration set elsewhere in the command. It can't see what earlier commands left in the shell (an
exported `GIT_DIR`, an alias); `pre_push.py`, which reads the URL git pushes to, can. Not covered by the
push rule: a push that runs `/usr/bin/git` by path, a git alias for `push`, `VERITY_GIT_GUARD=off`, and a client other
than git. On a git before 2.31 a push is judged but runs no other pre-push hook.

The identity rule (`worktree_guard.identity_reason`) covers a checkout whose `origin` is one of those two repositories, a
worktree of one included. A commit there carries the machine's configured identity, so a `git` that names an author or
committer of its own is refused: `-c` or `--config-env` of `user.name`, `user.email` or the same two keys of `author` and
`committer`; `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`, `GIT_COMMITTER_NAME` or `GIT_COMMITTER_EMAIL` assigned ahead of the
command; `git commit --author`; and a `git config KEY VALUE` that writes one of those keys to the checkout's own
configuration. The refusal says to commit with the machine's configured identity and pass no name or address. A read
passes (`git log --author`, `git config user.email`, `git config --get user.email`, `git var GIT_AUTHOR_IDENT`), and so
does any other repository: a scratch one takes any identity. The machine's identity sits in its global configuration, so
a `git config --global KEY VALUE` of one of those keys is refused wherever it is typed, unless the command itself names
another file as the global configuration (`HOME=DIR git config --global ...`); `git config --global user.name` reads it.

Git reads `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`, `GIT_COMMITTER_NAME` and `GIT_COMMITTER_EMAIL` from its environment in
every repository, and a shell hands a variable it holds to every later command: a name exported for a commit in a scratch
repository is the author of the next commit in a checkout too, by a `git` that names none. The hook reads a command,
never a shell's environment, so it refuses the command that leaves one of the four set in the shell that runs it
(`shell_hook.shell_identity_reason`), wherever it is typed: `export NAME=VALUE` and `export NAME`; `declare`, `typeset`
and `readonly` of one (`declare -p NAME` prints it and passes); and an assignment with no command after it (`NAME=VALUE`
alone), which reaches git where the name is already exported or `set -a` is on, or once an `export` names it. This rule
alone refuses in doubt, since a commit under a wrong name can't be taken back and a refused command is typed again: what a
subshell or `bash -c` runs is refused though the name may die with it; a text with a `#` in it, where the hook's reading
of a command stops, is read line by line as well; and what the hook can't split, the text or a line, is cut at its
operators and line ends with its quotes left on, so there an export that is only the words of a quoted string can be
refused too. The refusal says where
a scratch repository's name goes, ahead of the one command (`GIT_AUTHOR_NAME=NAME ... git -C <scratch> commit ...`) or on
it (`-c user.name=NAME -c user.email=ADDRESS`), and that `unset GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME
GIT_COMMITTER_EMAIL` clears one the shell holds. The names ahead of one command pass this rule and are judged by the one
above, where that git runs; `unset`, a read (`echo "$GIT_AUTHOR_NAME"`) and an `export` of any other variable
(`GIT_AUTHOR_DATE`) pass.

The rule for a `git` that names an identity places a command by the directory the hook is handed and git's own `-C`, and, alone among the rules, follows a
plain `cd` (`shell_hook.cd_dirs`): `cd CHECKOUT && git ...` is judged in CHECKOUT and `cd SCRATCH && git ...` in SCRATCH.
A plain `cd` is `cd DIR` and no more, between `&&`, `;` or a line's end, DIR a directory that exists, written out, taken
from where the `cd` before it left the shell; a git inside `( ... )` or what `bash -c` runs is placed by the `cd`s before
it there. Where the `cd`s leave the directory in doubt the command passes: any other form of `cd` (`cd "$D"`, `cd ~`, `cd
-`, `cd DIR || exit`), and everything after a keyword (`if`, `do`, `{`), a `)`, a `&` or a `$(`, or with an unquoted
backquote anywhere. Not covered: a command the hook still can't place (`pushd`, `source`, `GIT_DIR`, `--git-dir`), a
variable the shell held before the guard arrived (`unset` clears it), one a script exports (`source FILE`, a line in
`~/.bashrc`) or whose name the shell builds (`GIT_{AUTHOR,COMMITTER}_NAME`, `export "$line"`), `git config --global
--unset|--add|--replace-all`, `--system` and `--file`, `--author` cut short, an alias, and what a script runs.

The tmux rule (`kept_session.py`) covers the VM's own tmux server, the one a `tmux` with no `-L NAME` or `-S PATH` talks
to; a server of a command's own passes whatever it ends there. `tmux kill-server` is refused always. A `tmux
kill-session` is refused when it would end `vm-secrets` (named exactly, by the start of its name or by a pattern, as
tmux reads a target), when it would end the server's last session, and when it names no session the hook can read (no
`-t`, a variable, a window or a pane). A `tmux kill-session -a` ends every session but one, and the hook can't tell
other agents' sessions from the caller's: it is refused whatever its target, `vm-secrets` included, while the server has
more than one session. Any other kill passes, as does `kill-session -C`, which ends nothing. For a `kill-session` the
hook asks tmux which sessions there are; where tmux gives no answer it refuses what the command's text could mean for
`vm-secrets`, and every `-a`. Each refusal says to close your own sessions by name. It can't see which server a pane's
own tmux talks to (a kill typed into a pane is judged as the VM's), a `TMUX` or `TMUX_TMPDIR` the command sets, or a
session ended some other way (its last window killed, its shell's `exit`, a signal). `kept_session.py start` starts the
session only where the server it would start holds every secret the VM injects; a server it starts gives each later
session the `CURSOR_*` variables and `PATH` of the shell that starts that session, since the boot's environment names no
agent.

Every rule of the shell hook reads a command as the shell runs it (`shell_hook.words_of`), typed or in what a carrier
has run: a comment hides the rest of its own line and nothing after it, a `#` inside a word or a quoted string
(`issue#12`, a URL's fragment) hides nothing, and a `\` at a line's end joins the next line to it, though never to a
comment. Quotes are what Python's `shlex` takes for them, which leaves three things read otherwise than the shell reads
them: a comment inside backticks runs to its line's end, a quote nested in a `$(...)` inside another ends that one, and
a `$'...'` ends at an escaped quote inside it. A command `shlex` can't split, with a quote nothing closes and no `#`
before it, is judged on its raw text.

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
uv run tools/verity/check/suites.py tools/agent-guard              its tests
~~~

A heavy command the shell hook refuses runs under `research run --on auto` instead.
