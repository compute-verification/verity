#!/usr/bin/env python3
"""One agent per checkout (the rule is AGENTS.md's): the shell hook's judgment of `git checkout`, `git switch` and
`git push`, and the branch registry it keeps.

    python3 tools/agent-guard/worktree_guard.py hand BRANCH AGENT_ID   give a branch you created to another agent (its new owner)
    python3 tools/agent-guard/worktree_guard.py reclaim DIR            as the VM's coordinator, take back the checkout DIR, and
                                                                       the branches its owner holds, once that owner is gone
    python3 tools/agent-guard/worktree_guard.py list                   each registered branch's owner

A worktree belongs to the first agent (Cursor conversation) that runs a git command in it: the coordinator's `/workspace`, or
the worktree a worker made with `git worktree add`. The owner is stamped in the worktree's git dir (`agent-owner`), and the
branch it had then is registered to it. The stamp outlives its agent, so the VM's coordinator takes a checkout back from an
owner that is gone (`reclaim`): the coordinator is an agent that holds a handle in the Slack registry (`task_hook.holders`,
origin/main's copy where `hook.sh` runs the hook), and gone is no hook run of the owner's in the last GONE_S, by the VM's
record of every hook run (`settings.recorded_runs`), which must reach back that far and hold a shell-hook run of the caller's
(`gone_reason`). The checkout and every branch registered to its owner are then the coordinator's. The hook refuses a
`reclaim` typed by an agent that holds no handle, the command judges the rest when it runs, and a checkout goes to no agent
but one that holds a handle. The hook then refuses, from any other agent:
  - switching that worktree's branch (`git checkout BRANCH|-b|-B`, `git switch`), since everything installed editable from it
    (the `research` CLI) follows the switch; checking out files (`git checkout -- PATH`, `git checkout REF -- PATH`) passes;
  - pushing a branch registered to another agent: one it created (`checkout -b`, `switch -c`, `worktree add -b`) or the one its
    worktree had when it claimed it, until that agent hands it over (`hand`). An unregistered branch passes.
For every agent, whatever it owns, it also refuses a push that would update `main`, `checked` or `next` (only drive and the lander move
them), and a `git config` that prints values, since the checkout's http.*.extraheader carries a GitHub token.
A lander's hand repair of `next` under the queue lock names its reason on the push itself (`AGENT_GUARD_PUSH_NEXT_OK='REASON'
git push ...`), and is logged to `$AGENT_GUARD_STATE/push-next-ok.jsonl`. Nothing lets a push to `main` through.
It refuses, too, a push, branch switch or worktree add with `$` or a backquote in its words, whose branch the shell names only
after the hook has read it. A push the agent doesn't type (a script's, a tool's) meets the same push rule, `PROTECTED_REASON`
and `owner_reason`, in git's pre-push hook (pre_push.py), branch by branch.
The push rules (the three branches, another agent's branch, an unexpanded push) cover only a push to the repositories whose
`main`, `checked` and `next` drive lands, QUEUE_REPOS on GitHub (`queue_remote`; `AGENT_GUARD_PUSH_REMOTES`, a regular
expression, names others in their place): another repository's `main` is its own (the notes' `main`, which every agent's
`research notes sync` pushes), and the registry names a branch without its repository. A push's words are read as git
reads them (`read_push`: options anywhere, by any prefix, with their values), and one whose repository the guard can't tell
is judged as one to them (`judged`): run outside a repository, a first word that is neither a remote, a URL nor a path, an
option git doesn't know, or git configuration another command in the line may set.
A git command runs, for every rule, in the event's cwd (else the workspace root) joined with its own `git -C DIR`, and
nowhere else (`indexed_invocations`). A `cd`, `pushd`, `popd` or `source` anywhere in the command, run or not, or a repository
named by `GIT_DIR`, `GIT_WORK_TREE` or `--git-dir`, makes its directory unknown (`moves`): a push is then judged as one to
the queue's repositories with every branch its current one, refused for what the guard can't tell unless its own words name
a protected branch (`unplaced_reason`), and a branch switch is refused.
The registry is `$AGENT_GUARD_STATE/branches.json` (default /tmp/agent-guard), one VM's: an agent on a VM of its own is never
judged.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from settings import CONVERSATION_ID, EVENT_LOG, PUSH_REMOTES, recorded_runs
from task_hook import REGISTRY, holders

STATE = Path(os.environ.get("AGENT_GUARD_STATE", "/tmp/agent-guard"))
OWNER_FILE = "agent-owner"
#: an owner with no hook run for this long is gone: six times the longest one command may block (shell_hook.BLOCK_CAP_S), so
#: an agent that is waiting on a command never reads as gone
GONE_S = 2 * 3600
RECLAIM = "python3 tools/agent-guard/worktree_guard.py reclaim"

#: the repositories whose `main`, `checked` and `next` drive's App lands, and whose branches the registry names
QUEUE_REPOS = ("compute-verification/verity", "compute-verification/infra")
#: a GitHub URL of one of them, whole: https or ssh (with any user, credential or port) or scp form, with or without `.git`
QUEUE_URL = (r"(?i)^(?:(?:https?|ssh|git)://(?:[^@/\s]*@)?github\.com(?::\d+)?/|(?:[\w.-]+@)?github\.com:/?)"
             rf"(?:{'|'.join(re.escape(r) for r in QUEUE_REPOS)})(?:\.git)?/?$")


def git(directory: str, *args: str, env: dict[str, str] | None = None) -> str | None:
    try:
        p = subprocess.run(["git", "-C", directory, *args], capture_output=True, text=True, timeout=5,
                           env={**os.environ, **(env or {}), "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"})
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


def stamp(gitdir: str) -> str:
    """The agent stamped as the owner of the worktree whose git dir is `gitdir`, "" when it has none; it claims nothing."""
    f = Path(gitdir) / OWNER_FILE
    return f.read_text().strip() if f.is_file() else ""


def ago(t: float, now: float) -> str:
    return f"{(now - t) / 60:.0f} min ago ({time.strftime('%H:%M', time.gmtime(t))} UTC)"


def gone_reason(owner: str, caller: str, now: float | None = None) -> str:
    """Why the VM's record of hook runs (`settings.recorded_runs`) can't show agent `owner` gone, or "" when it does: gone is
    no hook run of its own in the last GONE_S. The record must hold a shell-hook run of `caller` in that time, since on a VM
    whose shell hook runs for no one every agent would read as gone, and must reach back that far, since it rotates by size
    and an owner it holds no run of may have run one just before its oldest line."""
    now, window = time.time() if now is None else now, f"{GONE_S / 3600:g} h"
    try:
        runs = recorded_runs()
    except OSError as e:
        return f"the VM's record of hook runs is unreadable ({e}), so it shows no agent gone"
    if not any(agent == caller and hook == "shell_hook.py" and t > now - GONE_S for t, hook, agent in runs):
        return (f"{EVENT_LOG} holds no shell-hook run of yours in the last {window}, so it can't show that another agent ran "
                f"none: no shell hook records your commands on this VM (`python3 tools/agent-guard/inbox.py guard` says why, "
                f"and the fix)")
    if (since := min(t for t, _, _ in runs)) > now - GONE_S:        # `runs` holds the caller's, so it is not empty
        return (f"the VM's record of hook runs ({EVENT_LOG}) reaches back only {ago(since, now)}, and {owner} may have run a "
                f"command before that: gone is no hook run in {window}. Try again after "
                f"{time.strftime('%H:%M', time.gmtime(since + GONE_S))} UTC")
    if (last := max((t for t, _, agent in runs if agent == owner), default=0)) > now - GONE_S:
        return (f"agent {owner} ran a command on this VM {ago(last, now)}, so it is not gone (gone is no hook run in {window}): "
                f"ask it for what you need, or work in a checkout of your own")
    return ""


def reclaim_note(owner: str, agent: str, top: str) -> str:
    """What a refusal that names `owner`, the agent stamped on the checkout `top`, adds for `agent`: the command that takes the
    checkout back, when `agent` is the VM's coordinator and `owner` is gone; else ""."""
    if (handle := holders().get(agent)) is None or gone_reason(owner, agent):
        return ""
    return (f" Agent {owner} has run no command on this VM in the last {GONE_S / 3600:g} h, and you hold @{handle}: take back "
            f"{top} and the branches {owner} holds with `{RECLAIM} {top}`.")


def not_coordinator(agent: str, directory: str) -> str:
    """The refusal of a `reclaim` by an agent that holds no handle."""
    return (f"only the VM's coordinator takes a checkout back, and {agent or 'this shell (no CURSOR_CONVERSATION_ID)'} holds no "
            f"handle in the Slack registry ({REGISTRY}): ask your coordinator to run `{RECLAIM} {directory}`, or work in a "
            f"checkout of your own (`git worktree add ../wt-<name> -b <branch>`)")


def reclaim(directory: str, agent: str, now: float | None = None) -> tuple[int, str]:
    """Give the checkout holding `directory`, and every branch registered to its stamped owner, to `agent`: (0, what changed),
    or (1, why not). Only an agent that holds a handle takes a checkout back, only one stamped to another agent, and only
    once that owner is gone (`gone_reason`)."""
    if (handle := holders().get(agent)) is None:
        return 1, not_coordinator(agent, directory)
    gitdir, top = git(directory, "rev-parse", "--absolute-git-dir"), git(directory, "rev-parse", "--show-toplevel")
    if gitdir is None:
        return 1, f"{directory} is in no checkout: name a checkout's directory"
    top = top or directory
    if not (owner := stamp(gitdir)):
        return 1, (f"{top} has no owner to take it back from: a checkout belongs to the first agent that runs git in it, so run "
                   f"`git -C {top} status` and it is yours")
    if owner == agent:
        return 0, f"{top} is already {agent}'s"
    if why := gone_reason(owner, agent, now):
        return 1, f"{top} stays agent {owner}'s: {why}"
    tmp = Path(gitdir) / f"{OWNER_FILE}.{os.getpid()}"
    tmp.write_text(agent + "\n")
    os.replace(tmp, Path(gitdir) / OWNER_FILE)
    held = sorted(b for b, a in registry().items() if a == owner)
    for b in held:
        register(b, agent, replace=True)
    return 0, (f"{top} is now {agent}'s (@{handle}); it was {owner}'s, gone by the VM's record of hook runs"
               + (f". So are the branches {owner} held: {', '.join(held)}" if held else ""))


def reclaim_reason(commands: list[list[str]], agent: str) -> str | None:
    """A `worktree_guard.py reclaim DIR` typed by an agent that holds no handle: the command reads who runs it from its
    environment, which the caller sets, and the hook from Cursor's event, which it doesn't."""
    for words in commands:
        for i, w in enumerate(words[:-1]):
            if w.endswith("worktree_guard.py") and words[i + 1] == "reclaim":
                return None if agent in holders() else not_coordinator(agent, " ".join(words[i + 2:i + 3]) or "DIR")
    return None


def hand_reason(commands: list[list[str]], agent: str, cwd: str = "") -> str | None:
    """A `worktree_guard.py hand BRANCH AGENT_ID` from an agent other than the one BRANCH is registered to: only its owner gives
    a branch away, or any agent could take another's branch by handing it to itself. Where the checkout the command runs in
    (`cwd`) is that owner's and it is gone, the VM's coordinator is told how to take both back (`reclaim_note`)."""
    reg = registry()
    for words in commands:
        for i, w in enumerate(words[:-2]):
            if w.endswith("worktree_guard.py") and words[i + 1] == "hand" and reg.get(b := words[i + 2], agent) != agent:
                return (f"branch {b} belongs to agent {reg[b]}, so only that agent can hand it on: ask it to run `python3 "
                        f"tools/agent-guard/worktree_guard.py hand {b} <agent-id>`."
                        + (held_note(reg[b], agent, cwd) or " If it is gone, push a branch of your own."))
    return None


def held_note(owner: str, agent: str, directory: str) -> str:
    """`reclaim_note` for the checkout holding `directory`, when `owner` is the agent stamped on it; else ""."""
    gitdir = git(directory, "rev-parse", "--absolute-git-dir") if directory else None
    if gitdir is None or stamp(gitdir) != owner:
        return ""
    return reclaim_note(owner, agent, git(directory, "rev-parse", "--show-toplevel") or directory)


#: git's own options that take the next word as their value (`git -h`; git takes none of its own options by a prefix)
GIT_VALUED = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env", "--attr-source"}
#: git's own options that change which configuration a command reads: the guard's own git is given them too
GIT_CONFIGURING = {"-c", "--config-env", "--bare"}
#: git's own options, and the variables, that name the repository or worktree in place of the directory git runs in
DIRECTORY_OPTIONS = {"--git-dir", "--work-tree"}
DIRECTORY_VARIABLES = {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"}
#: what moves the shell to another directory, wherever it sits in a command and whether or not it runs
MOVES = {"cd", "pushd", "popd", "source"}
#: the words after which the shell starts a command
HEADS = {"{", "(", "then", "do", "else", "elif", "!", "time"}


class Git(NamedTuple):
    i: int                  # its index among the commands
    d: str                  # the directory it runs in: the hook's cwd and its own `-C`
    args: list[str]         # its arguments after git's own options
    options: list[str]      # git's own options among GIT_CONFIGURING, with their values
    unknown: str            # why `d` may not be where it runs or what it reads, or ""


def moves(commands: list[list[str]]) -> str:
    """Why no git command among `commands` (unwrapped simple commands) can be placed by the hook's cwd and its own `-C`, or "":
    a `cd`, `pushd`, `popd` or `source` anywhere among them (in any list, pipeline, subshell, body, function or carrier, run or
    not), a `.`, or a command the shell names only later (`$C DIR`)."""
    for words in commands:
        heads = [w for j, w in enumerate(words) if j == 0 or words[j - 1] in HEADS]
        if hit := next((w for w in words if w in MOVES), None):
            return f"the command has a `{hit}` in it, and the guard follows none, wherever it sits"
        if hit := next((w for w in heads if w == "." or any(c in w for c in UNEXPANDED)), None):
            return f"the command runs `{hit}`, which may move it to another directory"
    return ""


def indexed_invocations(commands: list[list[str]], cwd: str, envs: list[dict[str, str]] | None = None) -> list[Git]:
    """Each `git` among `commands` (unwrapped simple commands), run in the hook's `cwd` and its own `git -C DIR`, nothing else;
    `envs[i]` is the git variables command `i` runs under (shell_hook.git_env). Its directory is unknown (`Git.unknown`) when
    the command moves (`moves`), or when it names its repository by DIRECTORY_OPTIONS or DIRECTORY_VARIABLES."""
    moved, out = moves(commands), []
    for i, words in enumerate(commands):
        if not words or os.path.basename(words[0]) != "git":
            continue
        env = envs[i] if envs is not None and i < len(envs) else {}
        d, args, options = cwd, words[1:], []
        unknown = moved or next((f"the command sets `{n}`" for n in sorted(DIRECTORY_VARIABLES & set(env))), "")
        while args and args[0].startswith("-"):
            a, valued = args[0], args[0] in GIT_VALUED and len(args) > 1
            name = a.partition("=")[0]
            if a == "-C" and valued:
                d = os.path.normpath(os.path.join(d, os.path.expanduser(args[1])))
            elif name in GIT_CONFIGURING:
                options += args[:1 + valued]
            elif name in DIRECTORY_OPTIONS:
                unknown = unknown or f"it names its repository with `{name}`"
            args = args[1 + valued:]
        if args:
            out.append(Git(i, d, args, options, unknown))
    return out


def created_branch(sub: str, args: list[str]) -> str | None:
    """The branch a checkout/switch/worktree add creates, if any."""
    flags = {"checkout": ("-b", "-B"), "switch": ("-c", "-C", "--create", "--force-create"), "worktree": ("-b", "-B")}.get(sub, ())
    for i, a in enumerate(args[:-1]):
        if a in flags:
            return args[i + 1]
    return None


def switches_branch(directory: str, sub: str, args: list[str], known: bool = True) -> bool:
    """Whether `git SUB ARGS` switches the branch; a `checkout` of one name does when the name is a commit in `directory`, and
    may when the directory is not `known`."""
    if sub == "switch":
        return True
    if sub != "checkout" or "--" in args:
        return False
    if created_branch(sub, args):
        return True
    rest = [a for a in args if not a.startswith("-")]
    return len(rest) == 1 and (not known or git(directory, "rev-parse", "--verify", "-q", rest[0] + "^{commit}") is not None)


#: `git push`'s long options (`git push -h`); git takes each by any prefix that names one alone, and `--no-` before each
PUSH_LONG = ("verbose", "quiet", "repo", "all", "branches", "mirror", "delete", "tags", "dry-run", "porcelain", "force",
             "force-with-lease", "force-if-includes", "recurse-submodules", "thin", "receive-pack", "exec", "set-upstream",
             "progress", "prune", "no-verify", "verify", "follow-tags", "signed", "atomic", "push-option", "ipv4", "ipv6")
#: those that take a value, as `=VALUE` or the next word (`--force-with-lease` and `--signed` take one only after `=`)
PUSH_VALUED = {"repo", "recurse-submodules", "receive-pack", "exec", "push-option"}
#: its single-letter options; `-o` (`--push-option`) takes a value, the rest of its word or the next one
PUSH_SHORT = set("vqdnfu46o")
#: the options that push every branch, the queue's among them
PUSH_EVERY = {"all", "branches", "mirror"}
#: a repository named as a URL (`scheme://`, `host:path`) or a path, rather than as a remote of the checkout
REPOSITORY_FORM = re.compile(r"://|^[^/:]+:|^[~./]|/")


@dataclass
class Push:
    """A `git push` read as git's option parser reads it."""
    d: str
    options: list[str]                  # git's own options the guard's git is given too (Git.options)
    env: dict[str, str]                 # the GIT_* variables, HOME and XDG_CONFIG_HOME the command is given
    placed: bool = True                 # whether it runs in `d` (not Git.unknown)
    repository: str | None = None       # its first word that isn't an option, else `--repo`'s value; None: the default
    refspecs: list[str] = field(default_factory=list)
    every: bool = False                 # PUSH_EVERY
    unsure: str = ""                    # why the guard can't tell where it goes, if it can't


def long_option(name: str) -> tuple[str, bool] | None:
    """The PUSH_LONG option `--NAME` gives and whether it is negated (`--no-`), or None when it names none or several."""
    if name in PUSH_LONG:
        return name, False
    if name.startswith("no-") and name[3:] in PUSH_LONG:
        return name[3:], True
    hits = [(n, False) for n in PUSH_LONG if n.startswith(name)]
    hits += [(n, True) for n in PUSH_LONG if name.startswith("no-") and n.startswith(name[3:])]
    return hits[0] if len(hits) == 1 else None


def read_push(g: Git, env: dict[str, str]) -> Push:
    """`git push` (`g.args[1:]`) read as git reads it: its options may come anywhere before a `--`, a long one by any prefix that
    names one alone, and a value in the option's word or the next one; the first word that is neither is the repository, the
    rest its refspecs. `env` holds `$` for a variable another command may set (shell_hook.git_env)."""
    p = Push(g.d, g.options, env, placed=not g.unknown)
    words, i, given = g.args[1:], 0, []
    while i < len(words):
        a, i = words[i], i + 1
        if a == "--":
            given += words[i:]
            break
        if a.startswith("--"):
            name, eq, value = a[2:].partition("=")
            if (hit := long_option(name)) is None:
                p.unsure = f"git reads `{a}` as no option of `git push`, or as several"
                continue
            full, negated = hit
            if full in PUSH_VALUED and not negated and not eq:
                value, i = (words[i] if i < len(words) else ""), i + 1
            if full == "repo" and not negated:
                p.repository = value
            p.every = full in PUSH_EVERY and not negated or p.every and not (negated and full in PUSH_EVERY)
        elif a.startswith("-") and len(a) > 1:
            for at, letter in enumerate(a[1:], 2):
                if letter not in PUSH_SHORT:
                    p.unsure = f"git reads `-{letter}` as no option of `git push`"
                if letter == "o":
                    i += at == len(a)
                    break
        else:
            given.append(a)
    if given:
        p.repository, p.refspecs = given[0], given[1:]
    if any(c in w for w in [*g.options, *p.env.values()] for c in UNEXPANDED):
        p.unsure = "the git configuration it runs under is set elsewhere in the command, or named through `$`"
    if g.unknown:
        p.unsure = g.unknown
    return p


def push_targets(p: Push, unplaced: str = "*") -> list[str]:
    """The branches `p` updates, as their names under refs/heads/ (`*` in one stands for any branch); its current branch is
    `unplaced` when the guard can't place the push, and `*` when it is detached."""
    current = git(p.d, *p.options, "symbolic-ref", "--short", "-q", "HEAD", env=p.env) if p.placed else None
    out = []
    for s in p.refspecs or ["HEAD"]:
        s = s.lstrip("+")
        if s.startswith("^"):
            continue
        src, colon, dst = s.partition(":")
        dst = (dst or (src if colon and src else "*" if colon else "")) if colon else src
        dst = (current or ("*" if p.placed else unplaced)) if dst in ("HEAD", "@") else dst
        out.append(dst.removeprefix("refs/").removeprefix("heads/"))
    return out


#: Branches only drive and the lander push, from their own processes rather than an agent's shell
PROTECTED = {"main", "checked", "next"}


def queue_remote(url: str, given: str | None = None) -> bool:
    """Whether a push to `url` goes to one of QUEUE_REPOS (QUEUE_URL), or, with `given` (AGENT_GUARD_PUSH_REMOTES), matches
    that regular expression instead."""
    return bool(re.search(given or QUEUE_URL, url))


def push_url(p: Push) -> str | None:
    """The URL `p` pushes to, as git rewrites it (`insteadOf`, `pushurl`), or None when it can't be told: its repository is a
    remote of the checkout or a URL or path (REPOSITORY_FORM, or a path that exists), else the current branch's push remote,
    `remote.pushDefault`, its upstream's remote or `origin`. A first word that is none of these means the words were misread."""
    def probe(*args: str) -> str | None:
        return git(p.d, *p.options, *args, env=p.env)

    remote = p.repository
    if remote is None:
        current = probe("symbolic-ref", "--short", "-q", "HEAD")
        keys = ([f"branch.{current}.pushRemote"] if current else []) + ["remote.pushDefault"] + (
            [f"branch.{current}.remote"] if current else [])
        remote = next((r for k in keys if (r := probe("config", "--get", k))), "origin")
    elif remote not in (probe("remote") or "").split():
        if not (REPOSITORY_FORM.search(remote) or os.path.exists(os.path.join(p.d, os.path.expanduser(remote)))):
            return None
        return probe("ls-remote", "--get-url", remote)
    return probe("remote", "get-url", "--push", remote)


def judged(p: Push) -> str | None:
    """Whether the push rules judge `p` (module doc): None when it goes to another repository; else "" when it goes to one of
    QUEUE_REPOS, or why the guard can't tell where it goes, which it then judges as a push to them."""
    if p.unsure:
        return p.unsure
    if (url := push_url(p)) is None:
        return (f"`{p.repository}` is no remote of {p.d}, nor a URL or a path" if p.repository else
                f"{p.d} names no remote to push to")
    return "" if queue_remote(url, PUSH_REMOTES) else None


def judged_note(why: str) -> str:
    """What a refusal of a push judged for `why` (`judged`) adds: how to name the repository so the guard can tell it."""
    return (f" The guard judged it as a push to verity or infra because {why}. It places a push only by the directory the "
            f"command starts in, git's own `-C` and a URL or path given as the repository, and a push to another repository "
            f"passes once it can: name the checkout and the remote on git itself, with no `cd` in the command, `git -C "
            f"<checkout> push <remote> <refspec>`.") if why else ""


def unplaced_reason(why: str) -> str:
    """The refusal of a push whose current branch the guard can't tell, for `why` (Git.unknown), and the push it accepts."""
    return (f"the guard can't tell which checkout this push runs in, so which branch it pushes, because {why}. It places a push "
            f"only by the directory the command starts in and git's own `-C`: name the checkout on git itself, with no `cd` in "
            f"the command, `git -C <checkout> push -u origin HEAD`, or name the branch, `git push -u origin <your-branch>`.")

#: `git config` forms that list keys: an installation token sits in http.*.extraheader's value and in url.*.insteadof's name,
#: so even `--name-only` prints it
CONFIG_DUMPS = {"--list", "-l", "--get-regexp", "--show-origin", "--get-all", "--get-urlmatch"}
CREDENTIAL_KEYS = ("extraheader", "insteadof", "token", "password", "credential")


PROTECTED_REASON = ("this push would update `main`, `checked` or `next`, which only drive and the lander move (AGENTS.md). Push "
                    "your own branch instead: `git push origin HEAD:refs/heads/<your-branch>` (create it first with "
                    "`git switch -c <your-branch>` if this checkout is on main).")


def next_repaired(targets: set[str], ok: str, record: dict) -> bool:
    """Whether a push to the protected `targets` is a lander's hand repair of `next` alone, named by its reason `ok`
    (AGENT_GUARD_PUSH_NEXT_OK); `record`, with the reason, then goes to `$AGENT_GUARD_STATE/push-next-ok.jsonl`."""
    if not (ok and targets == {"next"}):
        return False
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "push-next-ok.jsonl", "a") as f:
        f.write(json.dumps({**record, "reason": ok}) + "\n")
    return True


def owner_reason(branches: list[str], agent: str, directory: str = "") -> str | None:
    """Why `agent` may not push `branches`: the first that is registered to another agent, until that agent hands it over.
    An unregistered branch passes. Where the checkout the push runs in (`directory`) is that agent's and it is gone, the VM's
    coordinator is told how to take both back (`held_note`)."""
    reg = registry()
    for b in (n for given in branches for n in (fnmatch.filter(reg, given) if "*" in given else [given])):
        if reg.get(b, agent) != agent:
            return (f"branch {b} belongs to agent {reg[b]} (it created it, or its checkout had it): push a branch of "
                    f"your own (`git push -u origin <your-branch>`), or have its owner hand it to you: `python3 "
                    f"tools/agent-guard/worktree_guard.py hand {b} {agent}`." + held_note(reg[b], agent, directory))
    return None


def main_or_config_reason(invocations_: list[Git], pushes: dict[int, Push], next_ok: list[str], agent: str) -> str | None:
    """Whatever agent runs it: a push that would update `main` or `next` (a worker whose VM comes back from a snapshot reset
    to `main` would push onto it, and drive would rebuild every queued PR), or a `git config` that prints credential-bearing
    values."""
    for i, d, args, _, _ in invocations_:
        sub, ok = args[0], next_ok[i] if i < len(next_ok) else ""
        if sub == "push" and (why := judged(p := pushes[i])) is not None:
            named, targets = ({b for b in PROTECTED if any(fnmatch.fnmatchcase(b, t) for t in push_targets(p, unplaced))}
                              for unplaced in ("", "*"))
            if not p.every and next_repaired(targets, ok, {"agent": agent, "dir": d, "args": args}):
                continue
            if p.every or named:
                return PROTECTED_REASON + judged_note(why)
            if targets:
                return unplaced_reason(why)
        if sub == "config":
            rest = args[1:]
            names = [a for a in rest if not a.startswith("-")]
            if set(rest) & CONFIG_DUMPS or (len(names) == 1 and any(k in names[0].lower() for k in CREDENTIAL_KEYS)):
                return ("this `git config` would print a GitHub token: it sits in http.*.extraheader's value and in "
                        "url.*.insteadof's name, so `--name-only` lists it too (INF-105, INF-297). Read the one key you need: "
                        "`git config --get user.name`.")
    return None


#: what the shell expands after the hook has read the command: a branch or directory named through one can't be judged
UNEXPANDED = ("$", "`")


def unexpanded_reason(commands: list[list[str]], indexed: list[Git], pushes: dict[int, Push]) -> str | None:
    """Whatever agent runs it: a push, branch switch or worktree add whose words the shell has yet to expand. The hook sees
    `refs/heads/$B`, not the branch it becomes, so it can't hold that branch to its owner or to `main`, `checked` and `next`.
    Of `git worktree` only `add` creates or checks out a branch (and a subcommand the shell names only later may be it): a
    `remove`, `prune`, `list`, `move`, `lock` or `repair` over a variable passes."""
    for i, _, args, _, _ in indexed:
        sub = args[0]
        if sub not in ("push", "checkout", "switch", "worktree") or (sub == "checkout" and "--" in args):
            continue
        if sub == "worktree" and args[1:2] != ["add"] and not any(c in w for w in args[1:2] for c in UNEXPANDED):
            continue
        remote = pushes[i].repository or "" if sub == "push" else ""
        if sub == "push" and not any(c in remote for c in UNEXPANDED) and judged(pushes[i]) is None:
            continue
        if words := [w for w in commands[i][1:] if any(c in w for c in UNEXPANDED)]:
            literal = "git worktree add ../wt-<name> -b <branch>" if sub == "worktree" else "git push origin HEAD:refs/heads/<your-branch>"
            return (f"this `git {sub}` names `{words[0]}`, which the shell expands only after the guard has read it, so the guard "
                    f"can't check which branch or checkout it is. Name it literally (`{literal}`).")
    return None


def reason(commands: list[list[str]], agent: str, cwd: str, next_ok: list[str] | None = None,
           envs: list[dict[str, str]] | None = None) -> str | None:
    """Why the agent may not run these git commands here, or None. Registers the branches an allowed command creates.
    `next_ok[i]` is the AGENT_GUARD_PUSH_NEXT_OK reason `commands[i]` was given, if any; `envs[i]` the git variables it runs
    under (shell_hook.git_env)."""
    indexed = indexed_invocations(commands, cwd, envs)
    pushes = {g.i: read_push(g, envs[g.i] if envs is not None and g.i < len(envs) else {}) for g in indexed if g.args[0] == "push"}
    if why := unexpanded_reason(commands, indexed, pushes) or main_or_config_reason(indexed, pushes, next_ok or [], agent):
        return why
    if not agent:
        return None
    if why := hand_reason(commands, agent, cwd) or reclaim_reason(commands, agent):
        return why
    for i, d, args, _, unknown in indexed:
        sub = args[0]
        if unknown and sub in ("checkout", "switch") and switches_branch(d, sub, args[1:], known=False):
            return (f"the guard can't tell whose checkout this `git {sub}` switches, because {unknown}. Name the checkout on git "
                    f"itself, with no `cd` in the command: `git -C <checkout> {sub} ...` (files: `git -C <checkout> checkout -- "
                    f"PATH`). One agent per checkout (AGENTS.md).")
        gitdir = None if unknown else git(d, "rev-parse", "--absolute-git-dir")
        if gitdir is None and not unknown:
            continue
        mine = owner(gitdir, agent, d) if gitdir else agent
        if sub in ("checkout", "switch") and switches_branch(d, sub, args[1:]) and mine != agent:
            top = git(d, "rev-parse", "--show-toplevel") or d
            return (f"{top} is agent {mine}'s checkout, and switching its branch moves everything installed from it under that "
                    f"agent (the `research` CLI follows it). Work in a checkout of your own: `git -C {top} worktree add "
                    f"../wt-<name> -b <branch>` (or `git worktree add ../wt-<name> <existing-branch>`), then run git there "
                    f"with `git -C ../wt-<name> ...`. One agent per checkout (AGENTS.md)." + reclaim_note(mine, agent, top))
        if sub == "push" and (judgment := judged(pushes[i])) is not None and (
                why := owner_reason(push_targets(pushes[i]), agent, "" if unknown else d)):
            return why + judged_note(judgment)
        if (b := created_branch(sub, args[1:])) and sub in ("checkout", "switch", "worktree"):
            register(b, agent)
    return None


def main(argv: list[str]) -> int:
    if argv[:1] == ["hand"] and len(argv) == 3:
        register(argv[1], argv[2], replace=True)
        print(f"worktree_guard: {argv[1]} is now {argv[2]}'s")
        return 0
    if argv[:1] == ["reclaim"] and len(argv) == 2:
        code, said = reclaim(argv[1], CONVERSATION_ID)
        print(f"worktree_guard: {said}", file=sys.stderr if code else sys.stdout)
        return code
    if argv[:1] == ["list"]:
        print(json.dumps(registry(), indent=1, sort_keys=True))
        return 0
    print(__doc__.split("\n\n")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
