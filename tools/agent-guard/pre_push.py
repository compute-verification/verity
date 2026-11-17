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
drive lands, compute-verification/verity or compute-verification/platform (or its old name, infra) on GitHub (`worktree_guard.queue_remote`: the URL git
hands the hook, whole, in https, ssh or scp form, with or without `.git` and a credential; `AGENT_GUARD_PUSH_REMOTES`, a
regular expression, names others in their place): for each `refs/heads/` branch it creates, moves, forces or deletes,
the push is refused when the branch is `main`, `checked` or `next` (a lander's repair of `next` alone passes with
`AGENT_GUARD_PUSH_NEXT_OK='REASON'` in its environment, logged as the typed one is), or registered to another agent
(`$AGENT_GUARD_STATE/branches.json`) that hasn't handed it over; an unregistered branch passes. Without
`CURSOR_CONVERSATION_ID` nothing is judged: no agent started the push (a person, a boot script), and the queue's pushes on
the control pod, CI tooling and a person's laptop run no wrapper and so no hook at all. A push to another repository is not
judged either: the notes' `main`, which every agent's `research notes sync` pushes, a lane's notes branch, a fork, a node's bare
repository, a scratch clone. The URL can hold a token, so nothing prints it.

A judged push is refused, too, when it adds a commit whose author or committer is not the checkout's configured identity
(`authors_reason`): `git config user.name` and `git config user.email` as git's configuration files give them where the
push runs, and not as the push's own command line or environment sets them (`git -c user.name=NAME push`, a
`GIT_CONFIG_COUNT`). Git takes a commit's names from its environment before its configuration, so a name a script
exported, a shell held or a tool gave reaches a commit whoever typed it, and this hook reads the commits themselves. A
push adds what the remote lacks as the checkout knows it: for each line git hands the
hook, the commits reachable from its local sha and from no remote-tracking ref of that remote (`refs/remotes/REMOTE/`),
nor from the remote sha any line gives. A merge of the remote's `main` or of a branch another agent pushed adds the merge
alone, a new branch is held to what it adds to the remote's other branches, and a deletion adds nothing; a push that names
its repository by URL has no such refs, and a commit fetched some other way (by URL, by sha, a pull request's ref) is held
as new until `git fetch REMOTE` brings a ref that holds it. The committer must be the checkout's identity; the author must
be too, or an identity that already authors a commit on the remote's `main` as the checkout last fetched it
(`refs/remotes/REMOTE/main`, read only once a commit's author differs), with an address that is not made up (`made_up`)
nor the checkout's under another name: so a commit carried from a real author the remote has (a cherry-pick, a rebase) passes, and a name nobody has seen on
`main` is refused. The refusal says to carry another author's commits by a merge, and gives the `filter-branch` that
writes again only the names it found made up, in only the commits it names by sha. A push that adds commits
is refused as well where the checkout has no configured identity, or where git can't list them (it fails, or takes more
than `GIT_S`): a push that can't be judged is the one this rule is for. `OTHERS` holds the identities let through beside
the checkout's own, each with its reason: one, the queue's own, under which `research queue cross` makes a crossing's
commits before the caller's git pushes them. A head crossed by hand under any other name is refused.
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.dont_write_bytecode = True                      # no __pycache__ beside the hook: boot.sh copies its directory file by file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worktree_guard  # noqa: E402

#: an identity other than the checkout's own whose commits a judged push may add, `"NAME <ADDRESS>": "why it is let through"`
OTHERS: dict[str, str] = {
    "research queue <queue@research.invalid>": "the identity `research queue cross` gives a crossing's commits (`IDENTITY` in "
                                               "tools/research/src/research/queue.py), which the caller's own git then pushes",
}
#: seconds one `git` of the hook's may take: the list of what a push adds is the long one
GIT_S = 30
#: what git reads a commit's names from before its configuration
NAMES = "GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL"
#: how many of the commits a refusal names
SHOWN = 5
#: how many commits the refusal's `filter-branch` names by sha, so that it stays a command one can copy; the push after it
#: is refused again and names the rest
MENDED = 50


def branches(lines: list[str]) -> list[str]:
    """The branches git's pre-push lines (`LOCAL_REF LOCAL_SHA REMOTE_REF REMOTE_SHA`) update, deletions included."""
    out = []
    for line in lines:
        fields = line.split()
        if len(fields) == 4 and fields[2].startswith("refs/heads/"):
            out.append(fields[2].removeprefix("refs/heads/"))
    return out


def git(*args: str, env: dict[str, str], check: bool = True) -> str:
    """What `git ARGS` prints where the push runs (git starts the hook there, with its repository in `env`). Where git can't
    start, takes more than GIT_S or, under `check`, fails, this raises what `subprocess.run` does."""
    return subprocess.run(["git", *args], capture_output=True, check=check, timeout=GIT_S,
                          env={**env, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}).stdout.decode("utf-8", "replace")


def identity(env: dict[str, str]) -> str | None:
    """The checkout's configured identity as a commit carries it, `NAME <ADDRESS>`, or None where it has no name or no
    address: `user.name` and `user.email` as git's configuration files give them here. A value of the scope git calls the
    command's (a `-c` of the push's own command line, a `GIT_CONFIG_COUNT` in its environment) is not the checkout's."""
    found = []
    for key in ("user.name", "user.email"):
        rows = git("config", "--show-scope", "--get-all", key, env=env, check=False).split("\n")   # an unset key: it exits 1
        files = [value.strip() for scope, _, value in (row.partition("\t") for row in rows) if scope not in ("", "command")]
        found.append(files[-1] if files else "")
    name, address = found
    return f"{name} <{address}>" if name and address else None


def address(who: str) -> str:
    """`NAME <ADDRESS>`'s address, lower case."""
    return who.rpartition("<")[2].removesuffix(">").lower()


def made_up(who: str) -> bool:
    """Whether `NAME <ADDRESS>`'s address is one no mail reaches: a domain with no dot in it (`ci@x`, `coordinator@verity`),
    or one under a name RFC 2606 and RFC 6761 reserve (`invalid`, `test`, `example`, `localhost`) or RFC 6762's `local`."""
    domain = address(who).rpartition("@")[2]
    return "." not in domain or domain.rpartition(".")[2] in ("invalid", "test", "example", "localhost", "local")


def main_authors(remote: str, env: dict[str, str]) -> set[str] | None:
    """Every `NAME <ADDRESS>` that authors a commit on the remote's `main` as this checkout last fetched it
    (`refs/remotes/REMOTE/main`), or None where no such ref is here."""
    listed = git("log", "--format=%an <%ae>", f"refs/remotes/{remote}/main", "--", env=env, check=False)
    return set(listed.splitlines()) or None                                    # a ref that is there has a commit


def authors_reason(remote: str, lines: list[str], env: dict[str, str]) -> str | None:
    """Why this push adds a commit that the checkout's identity did not write, as author or as committer, or that can't be
    held to it (module doc), or None."""
    updates = [fields for line in lines if len(fields := line.split()) == 4]
    named = bool(remote) and not worktree_guard.REPOSITORY_FORM.search(remote)        # a URL can hold a token: never shown
    there = "<remote>" if not named else remote
    held = [*([f"--remotes={remote}"] if named else []), *(old for *_, old in updates if set(old) != {"0"})]
    added: dict[str, tuple[str, str, str]] = {}
    try:
        for _, new, _, _ in updates:
            if set(new) == {"0"}:
                continue
            listed = git("rev-list", "--ignore-missing", "--format=%H%x00%an <%ae>%x00%cn <%ce>", new, "--not", *held, env=env)
            for row in listed.split("\n"):
                sha, _, names = row.partition("\0")                                   # rev-list's own `commit SHA` lines have none
                if names:
                    author, _, committer = names.partition("\0")
                    added.setdefault(sha, (author, committer, new))
        me = identity(env) if added else None
        ours = {me, *OTHERS}
        # A commit carried from another author (a cherry-pick, a rebase) keeps its author and has the pusher as committer.
        # It passes where that author already authors a commit on the remote's `main`: a real name the remote has, which
        # this push does not bring in. A made-up address (`made_up`) is never let through as an author, however many
        # commits under it `main` holds; the committer is always the checkout's, so names a shell exported are refused on it.
        # The checkout's own address under another name is made up too: a tool or a shell gave the name.
        def fake(name: str) -> bool:
            return name not in ours and (made_up(name) or address(name) == address(me or ""))

        real = {name for pair in added.values() for name in pair[:2] if me and name not in ours and not fake(name)}
        known = (main_authors(remote, env) if named and real else None) or set()
        absent = bool(named and real) and not known
    except (OSError, subprocess.SubprocessError) as e:
        return (f"git could not list the commits this push adds ({type(e).__name__}: it failed, could not start or took more "
                f"than {GIT_S} s), so they can't be held to this checkout's identity. Push again; after `git fetch {there}` "
                "it has fewer to list.")
    if not added:
        return None
    if me is None:
        return ("this push adds commits, and this checkout has no identity to hold them to: `git config user.name` or `git "
                "config user.email` prints nothing here. The machine's identity is in its global configuration: push where "
                "git reads it, with no `GIT_CONFIG_GLOBAL` or `HOME` that names another.")
    def admitted(author: str) -> bool:
        return author in ours or (author in real and author in known)

    wrong = [(sha, author, committer, new) for sha, (author, committer, new) in added.items()
             if committer not in ours or not admitted(author)]
    if not wrong:
        return None
    # another author's work: carried under this checkout's identity as committer, that author's own commit (one real name
    # as author and committer), or committed by a real identity `main` has; every other refused commit took a made-up
    # name, and only those `filter-branch` mends
    theirs = [sha for sha, author, committer, _ in wrong
              if author in real and (committer in ours or author == committer or (committer in known and admitted(author)))]

    def written(author: str, committer: str) -> str:
        said = [f"{role} {''.join(c if c.isprintable() else '?' for c in name)}" for role, name, held in
                (("author", author, admitted(author)), ("committer", committer, committer in ours)) if not held]
        return f"author and committer {said[0].removeprefix('author ')}" if len(said) == 2 and author == committer else (
            ", ".join(said))

    shown = "; ".join(f"{sha[:12]} {written(author, committer)}" for sha, author, committer, _ in wrong[:SHOWN])
    rest = f"; and {len(wrong) - SHOWN} more" if len(wrong) > SHOWN else ""
    said = [f"this push adds {len(wrong)} commit{'s' * (len(wrong) > 1)} that this checkout's identity, {me}, did not "
            f"write, as author or as committer: {shown}{rest}. First `git fetch {there}`: a commit the remote already holds "
            "is not held to it once a ref of the remote's here holds it, and the push may then pass."]
    if not named:
        said.append("This push names its repository by URL, so no ref of it here counts: push to the remote by its name.")
    if absent:
        said.append(f"`refs/remotes/{remote}/main` is not here, so no author can be held to {remote}'s `main`: `git fetch "
                    f"{remote} main` first, and a commit whose author already authors one there passes under this "
                    "checkout's identity as committer.")
    if theirs:
        said.append("A commit whose author is another real name is that author's own, or was carried from that author by a "
                    "rebase or a cherry-pick under this checkout's identity as committer: another author's commits are "
                    "carried by a merge, not by a "
                    "rebase or a cherry-pick, so on a new branch, never by a forced push, run `git merge "
                    "<their-branch-or-commit>` on the branch or commit the remote holds them on; where the remote holds "
                    "none of an author's commits yet, the first push of that author's own commits is the author's to make.")
    made = [row for row in wrong if row[0] not in theirs]
    if made:
        # only the names this rule found made up, in only the commits it found them in, named by sha: a real person's
        # commit on the same branch, or a real author a pick kept, comes out as it was
        mended = made[:MENDED]
        reset = [(role, [sha for sha, author, committer, _ in mended if wrong_name(author, committer)])
                 for role, wrong_name in (("AUTHOR", lambda author, _: fake(author)),
                                          ("COMMITTER", lambda _, committer: committer not in ours))]
        cases = "; ".join(f'case "$GIT_COMMIT" in {"|".join(shas)}) unset GIT_{role}_NAME GIT_{role}_EMAIL;; esac'
                          for role, shas in reset if shas)
        more = (f" It names the first {MENDED} of the {len(made)} by sha, so that it stays a command one can copy: push "
                "<new-branch> after it, and the refusal names the rest by their new shas." if len(made) > MENDED else "")
        said.append(f"The {'other ' * bool(theirs)}commits, each with a committer that is not this checkout's identity or "
                    f"an author whose address is made up (a domain with no dot, or under `invalid`, `test`, `example`, "
                    f"`localhost` or `local`) or is this checkout's under another name, took made-up names from git's "
                    f"environment or from a tool. Clear the shell's, `unset {NAMES}`, and make them again on a new branch, "
                    f"never by a forced push: `git switch -c <new-branch> {made[0][3][:12]}`, then "
                    f"`FILTER_BRANCH_SQUELCH_WARNING=1 "
                    f"git filter-branch -f --env-filter '{cases}' -- <new-branch> --not --remotes={there}`, which writes "
                    "again, in the commits it names by sha, each such committer and author under this checkout's identity, "
                    "and leaves every other name, and each commit's tree, message and dates, as they were; then push "
                    f"<new-branch>.{more}")
    return " ".join(said)


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
    return worktree_guard.owner_reason(pushed, agent) or authors_reason(remote, lines, env)


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
