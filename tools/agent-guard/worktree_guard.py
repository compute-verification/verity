#!/usr/bin/env python3
"""One agent per checkout (the rule is AGENTS.md's): the shell hook's judgment of `git checkout`, `git switch` and
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
option git doesn't know, a `cd` before it that may not hold (`indexed_invocations`), or git variables the command sets.
The registry is `$AGENT_GUARD_STATE/branches.json` (default /tmp/agent-guard), one VM's: an agent on a VM of its own is never
judged. Where a command names no directory (`git -C DIR`, or a `cd DIR` before it in the same command) the hook judges the
event's cwd, else the workspace root, since the shell's remembered directory never reaches it.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from settings import PUSH_REMOTES

STATE = Path(os.environ.get("AGENT_GUARD_STATE", "/tmp/agent-guard"))
OWNER_FILE = "agent-owner"

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


def hand_reason(commands: list[list[str]], agent: str) -> str | None:
    """A `worktree_guard.py hand BRANCH AGENT_ID` from an agent other than the one BRANCH is registered to: only its owner gives
    a branch away, or any agent could take another's branch by handing it to itself."""
    reg = registry()
    for words in commands:
        for i, w in enumerate(words[:-2]):
            if w.endswith("worktree_guard.py") and words[i + 1] == "hand" and reg.get(b := words[i + 2], agent) != agent:
                return (f"branch {b} belongs to agent {reg[b]}, so only that agent can hand it on: ask it to run `python3 "
                        f"tools/agent-guard/worktree_guard.py hand {b} <agent-id>`. If it is gone, push a branch of your own.")
    return None


#: git's own options that take the next word as their value (`git -h`; git takes none of its own options by a prefix)
GIT_VALUED = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env", "--attr-source"}
#: git's own options that change which repository or configuration a command reads: the guard's own git is given them too
GIT_CONFIGURING = {"-c", "--git-dir", "--work-tree", "--config-env", "--bare"}
#: the operator shell_hook.flattened gives a command that a carrier runs: a `cd` there never reaches the commands after it
CARRIED = "carried"


class Git(NamedTuple):
    i: int                  # its index among the commands
    d: str                  # the directory it runs in
    args: list[str]         # its arguments after git's own options
    options: list[str]      # git's own options among GIT_CONFIGURING, with their values
    settled: bool           # whether `d` is known: no `cd` before it that might not hold, or an absolute `-C`


def indexed_invocations(commands: list[list[str]], cwd: str, ops: list[str] | None = None) -> list[Git]:
    """Each `git` among `commands` (unwrapped simple commands), tracking `cd DIR`, `pushd DIR` and `git -C DIR`. `ops[i]` is the
    shell operator after command `i` (CARRIED for one a carrier runs); a `cd` holds for the commands after it only when it is
    typed alone between `;` or `&&` (not piped, sent to the background, in a carrier or before a `)` that may close it), and
    any other change of directory (`cd -`, `popd`, `source`, a `cd` with an option or a `$`) leaves the directory unsettled."""
    out, settled, moved, before = [], True, False, ";"
    for i, words in enumerate(commands):
        op = ops[i] if ops is not None and i < len(ops) else None
        if words and words[0] in ("cd", "pushd", "popd", "source", "."):
            if words[0] in ("cd", "pushd"):
                cwd = os.path.normpath(os.path.join(cwd, os.path.expanduser(words[1] if len(words) > 1 else "~")))
            settled = settled and (words[0] == "cd" and len(words) == 2 and not words[1].startswith("-")
                                   and not any(c in words[1] for c in UNEXPANDED) and op in ("&&", ";")
                                   and "|" not in before.replace("||", ""))
            moved = True
        elif words and os.path.basename(words[0]) == "git":
            d, args, options, anchored = cwd, words[1:], [], False
            while args and args[0].startswith("-"):
                a, valued = args[0], args[0] in GIT_VALUED and len(args) > 1
                if a == "-C" and valued:
                    path = os.path.expanduser(args[1])
                    d, anchored = os.path.normpath(os.path.join(d, path)), anchored or os.path.isabs(path)
                elif a.partition("=")[0] in GIT_CONFIGURING:
                    options += args[:1 + valued]
                args = args[1 + valued:]
            if args:
                out.append(Git(i, d, args, options, settled or anchored))
        if moved and op is not None and ")" in op:
            settled = False
        if op != CARRIED:
            before = op or ";"
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


def read_push(g: Git, env: dict[str, str] | None) -> Push:
    """`git push` (`g.args[1:]`) read as git reads it: its options may come anywhere before a `--`, a long one by any prefix that
    names one alone, and a value in the option's word or the next one; the first word that is neither is the repository, the
    rest its refspecs. `env` is None when an earlier command may have changed the git variables it runs under."""
    p = Push(g.d, g.options, env or {})
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
    if not g.settled:
        p.unsure = "a `cd` before it may not hold (in a subshell, a pipe, a carrier, after `||`), so its directory is unknown"
    if env is None or any(c in w for w in [*g.options, *p.env.values()] for c in UNEXPANDED):
        p.unsure = "the git configuration it runs under is set in the command, or named through `$`"
    return p


def push_targets(p: Push) -> list[str]:
    """The branches `p` updates, as their names under refs/heads/ (`*` in one stands for any branch)."""
    current = git(p.d, *p.options, "symbolic-ref", "--short", "-q", "HEAD", env=p.env)
    out = []
    for s in p.refspecs or ["HEAD"]:
        s = s.lstrip("+")
        if s.startswith("^"):
            continue
        src, colon, dst = s.partition(":")
        dst = (dst or (src if colon and src else "*" if colon else "")) if colon else src
        dst = (current or "*") if dst in ("HEAD", "@") else dst
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
    return (f" The guard judged it as a push to verity or infra because {why}. A push to another repository passes once the "
            f"guard can tell it: name the checkout and the remote, `git -C <checkout> push <remote> <refspec>`.") if why else ""

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


def owner_reason(branches: list[str], agent: str) -> str | None:
    """Why `agent` may not push `branches`: the first that is registered to another agent, until that agent hands it over.
    An unregistered branch passes."""
    reg = registry()
    for b in (n for given in branches for n in (fnmatch.filter(reg, given) if "*" in given else [given])):
        if reg.get(b, agent) != agent:
            return (f"branch {b} belongs to agent {reg[b]} (it created it, or its checkout had it): push a branch of "
                    f"your own (`git push -u origin <your-branch>`), or have its owner hand it to you: `python3 "
                    f"tools/agent-guard/worktree_guard.py hand {b} {agent}`.")
    return None


def main_or_config_reason(invocations_: list[Git], pushes: dict[int, Push], next_ok: list[str], agent: str) -> str | None:
    """Whatever agent runs it: a push that would update `main` or `next` (a worker whose VM comes back from a snapshot reset
    to `main` would push onto it, and drive would rebuild every queued PR), or a `git config` that prints credential-bearing
    values."""
    for i, d, args, _, _ in invocations_:
        sub, ok = args[0], next_ok[i] if i < len(next_ok) else ""
        if sub == "push" and (why := judged(p := pushes[i])) is not None:
            targets = {b for b in PROTECTED if any(fnmatch.fnmatchcase(b, t) for t in push_targets(p))}
            if not p.every and next_repaired(targets, ok, {"agent": agent, "dir": d, "args": args}):
                continue
            if p.every or targets:
                return PROTECTED_REASON + judged_note(why)
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
    `refs/heads/$B`, not the branch it becomes, so it can't hold that branch to its owner or to `main`, `checked` and `next`."""
    for i, _, args, _, _ in indexed:
        sub = args[0]
        if sub not in ("push", "checkout", "switch", "worktree") or (sub == "checkout" and "--" in args):
            continue
        remote = pushes[i].repository or "" if sub == "push" else ""
        if sub == "push" and not any(c in remote for c in UNEXPANDED) and judged(pushes[i]) is None:
            continue
        if words := [w for w in commands[i][1:] if any(c in w for c in UNEXPANDED)]:
            return (f"this `git {sub}` names `{words[0]}`, which the shell expands only after the guard has read it, so the guard "
                    f"can't check which branch or checkout it is. Name it literally (`git push origin HEAD:refs/heads/<your-branch>`).")
    return None


def reason(commands: list[list[str]], agent: str, cwd: str, next_ok: list[str] | None = None, ops: list[str] | None = None,
           envs: list[dict[str, str] | None] | None = None) -> str | None:
    """Why the agent may not run these git commands here, or None. Registers the branches an allowed command creates.
    `next_ok[i]` is the AGENT_GUARD_PUSH_NEXT_OK reason `commands[i]` was given, if any; `ops[i]` the shell operator after it
    (`indexed_invocations`); `envs[i]` the git variables it is given (`read_push`)."""
    indexed = indexed_invocations(commands, cwd, ops)
    pushes = {g.i: read_push(g, envs[g.i] if envs is not None and g.i < len(envs) else {}) for g in indexed if g.args[0] == "push"}
    if why := unexpanded_reason(commands, indexed, pushes) or main_or_config_reason(indexed, pushes, next_ok or [], agent):
        return why
    found = [(g.i, g.d, g.args) for g in indexed]
    if not agent:
        return None
    if why := hand_reason(commands, agent):
        return why
    for i, d, args in found:
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
        if sub == "push" and (judgment := judged(pushes[i])) is not None and (
                why := owner_reason(push_targets(pushes[i]), agent)):
            return why + judged_note(judgment)
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
