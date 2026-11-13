"""A push an agent's script or child process makes meets the branch rule a typed `git push` meets: the git guard (git_guard.sh)
gives every push the pre-push hook (pre_push.py), which refuses a branch registered to another agent and `main`, `checked`
and `next`, and then runs the hook the push would have run without it, for a push to the repositories whose `main`,
`checked` and `next` drive lands (verity and infra). A push no agent started (a person's, the queue's) is not judged, and
neither is one to another repository (the notes, whose `main` every agent's `research notes sync` pushes). A judged push is
refused, too, when it adds a commit that the checkout's configured identity did not write, as author or as committer."""
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1]
ME, OTHER = "bc-11111111-1111-4111-8111-111111111111", "bc-22222222-2222-4222-8222-222222222222"
REAL_GIT = "/usr/bin/git"
#: the identity the miniature VM's git configuration holds, and another, as a script's exports leave it in a shell
OWN = "Own Machine <own@machine.invalid>"
TRIAL = {"GIT_AUTHOR_NAME": "trial", "GIT_AUTHOR_EMAIL": "trial@invalid", "GIT_COMMITTER_NAME": "trial",
         "GIT_COMMITTER_EMAIL": "trial@invalid"}
AS_TRIAL = "author and committer trial <trial@invalid>"


class VM:
    """An agent VM in miniature: `git` on PATH is the guard, whose pushes go through a hooks directory with pre_push.py in it;
    `origin` stands in for verity on GitHub (AGENT_GUARD_PUSH_REMOTES), `node` for a node's bare repository. Its identity
    (OWN) is in its global configuration, as a VM's is, and nothing in its environment names another."""

    def __init__(self, tmp: Path):
        self.tmp, self.origin, self.node, self.ws = tmp, tmp / "origin.git", tmp / "node.git", tmp / "ws"
        self.state, bin_, hooks = tmp / "state", tmp / "bin", tmp / "hooks"
        for d in (bin_, hooks, self.state):
            d.mkdir()
        (bin_ / "git").write_text(f"#!/bin/sh\nexec bash {GUARD / 'git_guard.sh'} \"$@\"\n")
        (hooks / "pre-push").write_text(f"#!/bin/sh\nexec python3 {GUARD / 'pre_push.py'} \"$@\"\n")
        for f in (bin_ / "git", hooks / "pre-push"):
            f.chmod(0o755)
        self.config = tmp / "gitconfig"
        self.config.write_text("[user]\n\tname = Own Machine\n\temail = own@machine.invalid\n")
        base = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "AGENT_GUARD_", "CURSOR_", "VERITY_"))}
        self.env = {**base, "GIT_CONFIG_GLOBAL": str(self.config), "GIT_CONFIG_NOSYSTEM": "1",
                    "PATH": f"{bin_}:{base.get('PATH', '/usr/bin:/bin')}", "VERITY_REAL_GIT": REAL_GIT,
                    "AGENT_GUARD_GIT_HOOKS": str(hooks), "AGENT_GUARD_STATE": str(self.state),
                    "AGENT_GUARD_PUSH_REMOTES": str(self.origin)}
        for bare in (self.origin, self.node):
            self.git("init", "-q", "--bare", "-b", "main", str(bare), cwd=tmp)
        self.git("clone", "-q", str(self.origin), str(self.ws), cwd=tmp)
        self.git("commit", "-q", "--allow-empty", "-m", "a")
        self.git("remote", "add", "node", str(self.node))
        (self.state / "branches.json").write_text(json.dumps({"theirs": OTHER, "mine": ME}))
        script = tmp / "push.sh"
        script.write_text("#!/bin/bash\n# what a lead's helper does: the shell hook never sees this push\ngit push \"$@\"\n")
        script.chmod(0o755)
        self.script = script

    def git(self, *args: str, cwd: Path | None = None, **env: str) -> str:
        return subprocess.run([REAL_GIT, *args], cwd=cwd or self.ws, check=True, capture_output=True, text=True,
                              env={**self.env, **env}).stdout

    def push(self, *args: str, agent: str | None = ME, **env: str) -> subprocess.CompletedProcess:
        """`push.sh ARGS` run by `agent`'s shell (None: no agent's), as a script run from the agent's command would be."""
        e = {**self.env, **env, **({"CURSOR_CONVERSATION_ID": agent} if agent else {})}
        return subprocess.run([str(self.script), *args], cwd=self.ws, capture_output=True, text=True, env=e)

    def shell(self, script: str, agent: str | None = ME, **env: str) -> subprocess.CompletedProcess:
        """`script` run by bash in the checkout, by `agent`'s shell: its exports reach every git after them."""
        e = {**self.env, **env, **({"CURSOR_CONVERSATION_ID": agent} if agent else {})}
        return subprocess.run(["bash", "-c", script], cwd=self.ws, capture_output=True, text=True, env=e)

    def commit(self, message: str, *args: str, **env: str) -> str:
        """An empty commit in the checkout, under the VM's identity unless `args` or `env` name another: its id."""
        self.git("commit", "-q", "--allow-empty", "-m", message, *args, **env)
        return self.git("rev-parse", "HEAD").strip()

    def who(self, revs: str, bare: Path | None = None) -> set[str]:
        """Each `AUTHOR | COMMITTER` among `revs`."""
        return set(self.git("log", "--format=%an <%ae> | %cn <%ce>", *revs.split(), cwd=bare or self.ws).splitlines())

    def heads(self, bare: Path | None = None) -> set[str]:
        out = self.git("for-each-ref", "--format=%(refname:short)", "refs/heads", cwd=bare or self.origin)
        return set(out.split())


@pytest.fixture
def vm(tmp_path):
    return VM(tmp_path)


def test_a_scripts_push_to_another_agents_branch_is_refused_naming_its_owner_and_the_hand_over(vm):
    p = vm.push("origin", "HEAD:refs/heads/theirs")
    assert p.returncode != 0 and "theirs" not in vm.heads()
    assert f"branch theirs belongs to agent {OTHER}" in p.stderr
    assert f"python3 tools/agent-guard/worktree_guard.py hand theirs {ME}" in p.stderr
    ours = [line for line in p.stderr.splitlines() if line.startswith("pre-push")]
    assert ours and not any(str(vm.origin) in line for line in ours), "the hook never prints the push URL, which can hold a token"


def test_a_scripts_push_to_its_own_or_an_unregistered_branch_passes(vm):
    assert vm.push("origin", "HEAD:refs/heads/mine").returncode == 0
    assert vm.push("-u", "origin", "HEAD:refs/heads/unregistered").returncode == 0
    assert {"mine", "unregistered"} <= vm.heads()


def test_a_branch_handed_over_passes(vm):
    subprocess.run(["python3", str(GUARD / "worktree_guard.py"), "hand", "theirs", ME], check=True, capture_output=True,
                   env=vm.env)
    assert vm.push("origin", "HEAD:refs/heads/theirs").returncode == 0


def test_a_delete_and_a_force_push_are_judged_as_the_typed_rule_judges_them(vm):
    assert vm.push("origin", "HEAD:refs/heads/theirs", "HEAD:refs/heads/mine", agent=None).returncode == 0
    vm.git("commit", "-q", "--allow-empty", "--amend", "-m", "b")       # not a fast-forward of either
    for args in (["--delete", "theirs"], [":theirs"], ["--force", "HEAD:refs/heads/theirs"], ["+HEAD:theirs"],
                 ["--force-with-lease", "HEAD:theirs"], ["HEAD:refs/heads/mine", "+HEAD:refs/heads/theirs"]):
        p = vm.push("origin", *args)
        assert p.returncode != 0 and f"belongs to agent {OTHER}" in p.stderr, args
    assert "theirs" in vm.heads()
    assert vm.push("origin", "+HEAD:refs/heads/mine").returncode == 0
    assert vm.push("origin", "--delete", "mine").returncode == 0 and "mine" not in vm.heads()


def test_a_scripts_push_to_main_checked_or_next_is_refused_and_a_named_repair_of_next_passes(vm):
    for ref in ("main", "checked", "next"):
        p = vm.push("origin", f"HEAD:refs/heads/{ref}")
        assert p.returncode != 0 and "only drive and the lander move" in p.stderr, ref
    assert vm.push("--all", "origin").returncode != 0
    repair = vm.push("origin", "HEAD:refs/heads/next", AGENT_GUARD_PUSH_NEXT_OK="restore next under the queue lock")
    assert repair.returncode == 0 and "next" in vm.heads()
    logged = json.loads((vm.state / "push-next-ok.jsonl").read_text().splitlines()[-1])
    assert logged["reason"] == "restore next under the queue lock" and logged["agent"] == ME and logged["by"] == "pre-push"
    assert vm.push("origin", "HEAD:refs/heads/main", AGENT_GUARD_PUSH_NEXT_OK="x").returncode != 0, "nothing lets main through"


def test_a_push_no_agent_started_is_not_judged_so_a_persons_and_the_queues_pass(vm):
    """No CURSOR_CONVERSATION_ID: a person at a terminal, or the queue's own pushes (on the control pod it runs no git guard
    at all; here, where it would, the hook still leaves it alone)."""
    for args in (["HEAD:refs/heads/theirs"], ["HEAD:refs/heads/main"], ["HEAD:refs/heads/next"], ["--delete", "theirs"]):
        assert vm.push("origin", *args, agent=None).returncode == 0, args
    assert vm.push("origin", "HEAD:refs/heads/next", agent="").returncode == 0, "an empty id is no agent"


def test_a_push_to_a_remote_other_than_github_is_not_judged(vm):
    """`research` pushes main to a node's bare repository on the agent's behalf; the branch rule is GitHub's."""
    assert vm.push("node", "HEAD:refs/heads/main").returncode == 0 and "main" in vm.heads(vm.node)
    assert vm.push(str(vm.node), "HEAD:refs/heads/theirs").returncode == 0


def test_the_hook_the_push_would_have_run_still_runs_and_neither_no_verify_nor_a_hooks_path_skips_the_rule(vm):
    """`research merge` installs its own pre-push hook in a checkout; tools pass `-c core.hooksPath=/dev/null` or
    `--no-verify` to skip a checkout's hooks, which skips that hook and never the branch rule."""
    own = vm.ws / ".git" / "hooks" / "pre-push"
    own.write_text(f"#!/bin/sh\ncat > {vm.tmp / 'own-ran'}\necho \"own hook: $1\" >&2\nexit ${{OWN_EXIT:-0}}\n")
    own.chmod(0o755)
    ok = vm.push("origin", "HEAD:refs/heads/mine")
    assert ok.returncode == 0 and "own hook: origin" in ok.stderr
    assert (vm.tmp / "own-ran").read_text().split()[2] == "refs/heads/mine", "it reads the push's lines as git gave them"
    assert vm.push("origin", "HEAD:refs/heads/unregistered", OWN_EXIT="1").returncode != 0, "its refusal stands"
    assert vm.push("origin", "HEAD:refs/heads/free", agent=None, OWN_EXIT="1").returncode != 0, "whoever pushes"
    (vm.tmp / "own-ran").unlink()
    for args in (["--no-verify", "origin", "HEAD:refs/heads/theirs"], ["origin", "--no-verify", "HEAD:refs/heads/theirs"]):
        p = vm.push(*args)
        assert p.returncode != 0 and "belongs to agent" in p.stderr, args
    assert vm.push("--no-verify", "origin", "HEAD:refs/heads/free", OWN_EXIT="1").returncode == 0
    assert not (vm.tmp / "own-ran").exists(), "--no-verify still skips the checkout's own hook"
    e = {**vm.env, "CURSOR_CONVERSATION_ID": ME}
    p = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "push", "origin", "HEAD:refs/heads/theirs"], cwd=vm.ws,
                       capture_output=True, text=True, env=e)
    assert p.returncode != 0 and "belongs to agent" in p.stderr
    p = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "push", "origin", "HEAD:refs/heads/free2"], cwd=vm.ws,
                       capture_output=True, text=True, env={**e, "OWN_EXIT": "1"})
    assert p.returncode == 0 and not (vm.tmp / "own-ran").exists()


def test_a_pushs_options_never_hide_where_it_goes_from_the_hook(vm):
    """git hands the hook the remote and the URL it pushes to, whatever words named them, so an option that takes a value
    (`--recurse-submodules check`, `--receive-pack P`, `--repo R`) moves nothing the hook reads."""
    for args in (["--recurse-submodules", "check", "origin", "HEAD:refs/heads/main"],
                 ["--receive-pack", "git-receive-pack", "origin", "HEAD:refs/heads/main"], ["--repo", "origin"],
                 ["--repo=origin"], ["--push-opt=x", "origin", "HEAD:refs/heads/theirs"]):
        p = vm.push(*args)
        assert p.returncode != 0 and ("only drive and the lander move" in p.stderr or "belongs to agent" in p.stderr), args
    assert not vm.heads()


def test_a_child_process_push_is_judged_like_a_scripts(vm):
    code = "import subprocess, sys; sys.exit(subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/theirs']).returncode)"
    p = subprocess.run(["python3", "-c", code], cwd=vm.ws, capture_output=True, text=True,
                       env={**vm.env, "CURSOR_CONVERSATION_ID": ME})
    assert p.returncode != 0 and f"belongs to agent {OTHER}" in p.stderr


def test_without_the_hooks_directory_the_guard_pushes_as_git_does(vm, tmp_path):
    p = vm.push("origin", "HEAD:refs/heads/theirs", AGENT_GUARD_GIT_HOOKS=str(tmp_path / "absent"))
    assert p.returncode == 0, "a VM where install.sh put no hook"


def test_install_writes_the_hook_the_git_guard_runs():
    text = (GUARD / "install.sh").read_text()
    assert "hooks=/usr/local/lib/agent-guard/hooks" in text and "/pre_push.py" in text
    assert "${AGENT_GUARD_GIT_HOOKS:-/usr/local/lib/agent-guard/hooks}" in (GUARD / "git_guard.sh").read_text()


#: a GitHub installation token's shape, from parts, as the URL a Cursor VM's insteadOf rewrites to carries one
TOKEN = "gh" "s_" "dummyDUMMYdummy0123456789abcdefABCDEF"
NOTES = "https://github.com/compute-verification/research-notes.git"
VERITY = ["https://github.com/compute-verification/verity.git", "https://github.com/compute-verification/verity",
          "git@github.com:compute-verification/verity.git", "ssh://git@github.com/compute-verification/verity.git",
          f"https://x-access-token:{TOKEN}@github.com/compute-verification/verity"]
INFRA = ["https://github.com/compute-verification/infra.git", f"https://x-access-token:{TOKEN}@github.com/compute-verification/infra"]
NOT_QUEUE = [NOTES, f"https://x-access-token:{TOKEN}@github.com/compute-verification/research-notes.git",
             "https://github.com/someone/verity-fork", "https://github.com/compute-verification/verity-notes",
             "https://github.com/someone/compute-verification/verity", "https://github.com/compute-verification/verity.git.bak",
             "git@github.com:someone/verity.git"]
ZERO, A, B = "0" * 40, "1" * 40, "2" * 40


def hook(url: str, *lines: str, agent: str | None = ME, tmp: Path, **env: str) -> subprocess.CompletedProcess:
    """pre_push.py as git runs it for a push to `url` (`origin`), fed git's lines, with the default repositories."""
    base = {k: v for k, v in os.environ.items() if not k.startswith(("AGENT_GUARD_", "CURSOR_"))}
    (tmp / "state").mkdir(exist_ok=True)
    (tmp / "state" / "branches.json").write_text(json.dumps({"theirs": OTHER, "lanes/infra": OTHER}))
    e = {**base, "AGENT_GUARD_STATE": str(tmp / "state"), **({"CURSOR_CONVERSATION_ID": agent} if agent else {}), **env}
    return subprocess.run(["python3", str(GUARD / "pre_push.py"), "origin", url], input="".join(f"{x}\n" for x in lines),
                          capture_output=True, text=True, env=e)


MAIN, NEXT = f"refs/heads/main {A} refs/heads/main {B}", f"HEAD {A} refs/heads/next {B}"
LANE, DELETE = f"refs/heads/lanes/infra {A} refs/heads/lanes/infra {B}", f"(delete) {ZERO} refs/heads/theirs {B}"


def test_a_notes_sync_push_of_main_to_the_notes_passes(tmp_path):
    """`research notes sync` pushes the notes' `main` from a child of the agent's shell (notes_sync.py, through a token URL),
    and `research notes push` a lane's branch, which may share a name with a registered verity branch."""
    for url in NOT_QUEUE[:2]:
        for refs in (MAIN, NEXT, LANE, DELETE):
            assert hook(url, refs, tmp=tmp_path).returncode == 0, (url, refs)


def test_the_same_refs_to_verity_are_refused(tmp_path):
    for url in VERITY:
        for refs, said in ((MAIN, "only drive and the lander move"), (NEXT, "only drive and the lander move"),
                           (f"refs/heads/x {A} refs/heads/checked {ZERO}", "only drive and the lander move"),
                           (LANE, f"belongs to agent {OTHER}"), (DELETE, f"belongs to agent {OTHER}")):
            p = hook(url, refs, tmp=tmp_path)
            assert p.returncode == 1 and said in p.stderr, (url, refs)
    assert hook(VERITY[0], MAIN, agent=None, tmp=tmp_path).returncode == 0, "no agent's push is judged"


def test_the_infra_repositorys_main_is_refused(tmp_path):
    for url in INFRA:
        p = hook(url, MAIN, tmp=tmp_path)
        assert p.returncode == 1 and "only drive and the lander move" in p.stderr, url


def test_a_url_that_only_holds_the_repositorys_name_elsewhere_is_not_verity(tmp_path):
    for url in NOT_QUEUE[2:]:
        for refs in (MAIN, LANE):
            assert hook(url, refs, tmp=tmp_path).returncode == 0, (url, refs)


def test_a_url_with_a_token_in_it_is_matched_and_never_printed(tmp_path):
    for url in (VERITY[4], INFRA[1]):
        p = hook(url, MAIN, LANE, tmp=tmp_path)
        assert p.returncode == 1 and "refused" in p.stderr
        assert TOKEN not in p.stderr + p.stdout and "x-access-token" not in p.stderr + p.stdout


def test_agent_guard_push_remotes_names_other_repositories_in_the_defaults_place(tmp_path):
    line = f"refs/heads/main {A} refs/heads/main {B}"
    assert hook(NOTES, line, tmp=tmp_path, AGENT_GUARD_PUSH_REMOTES=r"research-notes").returncode == 1
    assert hook(VERITY[0], line, tmp=tmp_path, AGENT_GUARD_PUSH_REMOTES=r"research-notes").returncode == 0


def test_an_inbox_message_pushed_to_verity_passes(vm):
    """`inbox.py send` commits a message onto `inbox/AGENT` and pushes it to the checkout's origin, verity, through `git`."""
    code = ("import sys; sys.dont_write_bytecode = True; sys.path.insert(0, sys.argv[1]); import inbox; from pathlib import Path; "
            "print(inbox.commit(Path(sys.argv[2]), sys.argv[3], 'hello', None, 'message'))")
    p = subprocess.run(["python3", "-c", code, str(GUARD), str(vm.ws), OTHER], capture_output=True, text=True,
                       env={**vm.env, "CURSOR_CONVERSATION_ID": ME})
    assert p.returncode == 0, p.stderr
    assert f"inbox/{OTHER}" in vm.heads()


EXPORTS = "export " + " ".join(f"{name}={value}" for name, value in TRIAL.items())


def test_a_commit_under_another_author_is_refused_naming_it_and_the_checkouts_identity(vm):
    sha = vm.commit("another's", "--author=trial <trial@invalid>")
    p = vm.push("origin", "HEAD:refs/heads/mine")
    assert p.returncode != 0 and "mine" not in vm.heads()
    assert f"this push adds 1 commit that this checkout's identity, {OWN}, did not write" in p.stderr
    assert f"{sha[:12]} author trial <trial@invalid>." in p.stderr and "committer trial" not in p.stderr


def test_a_commit_under_another_committer_is_refused(vm):
    sha = vm.commit("carried", GIT_COMMITTER_NAME="trial", GIT_COMMITTER_EMAIL="trial@invalid")
    p = vm.push("origin", "HEAD:refs/heads/mine")
    assert p.returncode != 0 and "mine" not in vm.heads()
    assert f"{sha[:12]} committer trial <trial@invalid>." in p.stderr and "author trial" not in p.stderr


def test_names_exported_in_the_shell_that_commits_and_pushes_are_refused(vm):
    """A scratch script's `export` of the four names outlives the scratch repository: every later `git commit` in that shell
    takes them, in a checkout too, and the push from the same shell is the first thing that reads what they wrote."""
    p = vm.shell(f"{EXPORTS}\ngit commit -q --allow-empty -m one && git commit -q --allow-empty -m two\n"
                 "git push origin HEAD:refs/heads/mine\n")
    assert p.returncode != 0 and not vm.heads()
    assert "this push adds 2 commits that" in p.stderr and p.stderr.count(AS_TRIAL) == 2
    assert vm.who("-2") == {"trial <trial@invalid> | trial <trial@invalid>"}


def test_a_branch_whose_commits_are_all_the_checkouts_passes_new_and_moved(vm):
    vm.commit("one"), vm.commit("two")
    assert vm.push("origin", "HEAD:refs/heads/mine").returncode == 0
    vm.commit("three")
    assert vm.push("origin", "HEAD:refs/heads/mine").returncode == 0
    assert vm.who("mine", vm.origin) == {f"{OWN} | {OWN}"}


def remotes_line(vm: VM) -> str:
    """A branch `line` on the remote with one commit under another identity (as its `main` holds the queue's), the checkout
    back on a new branch `work` cut from before it: that commit's id."""
    vm.git("switch", "-q", "-c", "line")
    theirs = vm.commit("the remote's own", **TRIAL)
    vm.git("push", "-q", "origin", "line")
    vm.git("switch", "-q", "-c", "work", "main")
    return theirs


def test_a_merge_of_a_branch_the_remote_already_has_passes_once_a_ref_of_the_remotes_here_holds_it(vm):
    theirs = remotes_line(vm)
    vm.commit("mine")
    vm.git("merge", "-q", "--no-ff", "-m", "Merge line", "origin/line")
    vm.git("update-ref", "-d", "refs/remotes/origin/line")       # as after a pull by URL: the commit is here, no ref holds it
    p = vm.push("origin", "HEAD:refs/heads/work")
    assert p.returncode != 0 and f"{theirs[:12]} {AS_TRIAL}." in p.stderr and "`git fetch origin`" in p.stderr
    vm.git("fetch", "-q", "origin")
    assert vm.push("origin", "HEAD:refs/heads/work").returncode == 0
    assert vm.git("rev-list", "--count", "--merges", "work", cwd=vm.origin).strip() == "1"


def test_a_cherry_pick_of_the_remotes_commit_passes_under_the_checkouts_identity_and_not_under_anothers(vm):
    """A cherry-pick is a new commit with its source's author and the picker as committer."""
    vm.git("switch", "-q", "-c", "line")
    ours, theirs = vm.commit("ours"), vm.commit("another's", **TRIAL)
    vm.git("push", "-q", "origin", "line")
    vm.git("switch", "-q", "-c", "work", "main")
    vm.commit("mine")
    vm.git("cherry-pick", "--allow-empty", ours)
    assert vm.git("rev-parse", "HEAD").strip() != ours and vm.push("origin", "HEAD:refs/heads/work").returncode == 0
    vm.git("cherry-pick", "--allow-empty", theirs)
    picked = vm.git("rev-parse", "HEAD").strip()
    p = vm.push("origin", "HEAD:refs/heads/work")
    assert p.returncode != 0 and f"{picked[:12]} author trial <trial@invalid>." in p.stderr


def test_another_names_commits_pass_to_a_repository_that_is_not_the_queues(vm):
    vm.commit("another's", **TRIAL)
    assert vm.push("node", "HEAD:refs/heads/work").returncode == 0 and "work" in vm.heads(vm.node)
    assert vm.push(str(vm.node), "HEAD:refs/heads/again").returncode == 0


def test_another_names_commits_pass_in_a_push_no_agent_started(vm):
    vm.commit("another's", **TRIAL)
    assert vm.push("origin", "HEAD:refs/heads/work", agent=None).returncode == 0 and "work" in vm.heads()
    assert vm.push("origin", "HEAD:refs/heads/again", agent="").returncode == 0, "an empty id is no agent"


def test_a_deletion_adds_nothing(vm):
    vm.git("push", "-q", "origin", "HEAD:refs/heads/mine", "HEAD:refs/heads/free")
    vm.commit("another's", **TRIAL)
    assert vm.push("origin", "--delete", "mine").returncode == 0 and "mine" not in vm.heads()
    vm.config.write_text("")
    assert vm.push("origin", ":free").returncode == 0 and "free" not in vm.heads(), "nor does it need an identity"


def test_a_new_branch_is_held_to_what_it_adds_to_the_remotes_other_branches_not_to_its_whole_history(vm):
    for i in range(3):
        vm.commit(f"the remote's own {i}", **TRIAL)
    vm.git("push", "-q", "origin", "HEAD:refs/heads/main")
    vm.git("switch", "-q", "-c", "work")
    vm.commit("mine")
    assert vm.push("-u", "origin", "work").returncode == 0, "git hands the hook no remote sha for a new branch"
    vm.git("switch", "-q", "-c", "other")
    sha = vm.commit("another's", **TRIAL)
    p = vm.push("-u", "origin", "other")
    assert p.returncode != 0 and "this push adds 1 commit that" in p.stderr and f"{sha[:12]} {AS_TRIAL}." in p.stderr


def test_a_refusal_names_five_commits_and_counts_the_rest_and_what_it_says_to_run_makes_them_again(vm):
    """Seven commits under another name, a merge of the remote's branch mended by hand among them. The refusal's own
    commands, run in a shell that still holds the names, give a branch that passes: the same seven under the checkout's
    identity, the merge and its mend kept, and the remote's commit, under the other name too, left as it is."""
    theirs = remotes_line(vm)
    change = "echo $i > f$i && git add f$i && git commit -q -m $i"
    made = vm.shell(f"{EXPORTS}\nfor i in 1 2 3; do {change}; done\n"
                    "git merge -q --no-ff --no-commit origin/line && echo mended > m && git add m && git commit -q -m merge\n"
                    f"for i in 4 5 6; do {change}; done\n", agent=None)
    assert made.returncode == 0, made.stderr
    old = vm.git("rev-parse", "HEAD").strip()
    p = vm.shell(f"{EXPORTS}\ngit push origin work")
    assert p.returncode != 0 and "work" not in vm.heads()
    assert "this push adds 7 commits that" in p.stderr and p.stderr.count(AS_TRIAL) == 5 and "; and 2 more." in p.stderr
    assert theirs[:12] not in p.stderr, "the remote's own commit is not one this push adds"
    said = re.findall(r"`([^`]+)`", p.stderr)
    unset, switch, again = (next(c for c in said if c.startswith(w)) for w in ("unset ", "git switch ", "FILTER_BRANCH_"))
    assert old[:12] in switch
    run = "\n".join([EXPORTS, unset, switch, again, "git push origin again"]).replace("<new-branch>", "again")
    done = vm.shell(run)
    assert done.returncode == 0, done.stderr
    assert vm.who("again --not line", vm.origin) == {f"{OWN} | {OWN}"}
    assert vm.git("rev-parse", "again^{tree}", cwd=vm.origin) == vm.git("rev-parse", f"{old}^{{tree}}")
    assert vm.git("rev-list", "--merges", "--format=%P", "again", cwd=vm.origin).split()[-1] == theirs
    assert vm.git("log", "--format=%s", "again", "--not", "line", cwd=vm.origin).split() == "6 5 4 merge 3 2 1".split()


def test_a_push_that_adds_commits_from_a_checkout_with_no_configured_identity_is_refused(vm):
    vm.config.write_text("[user]\n\tname = Own Machine\n")
    p = vm.push("origin", "HEAD:refs/heads/mine")
    assert p.returncode != 0 and "mine" not in vm.heads()
    assert "this checkout has no identity to hold them to" in p.stderr
    vm.config.write_text("[user]\n\tname = Own Machine\n\temail = own@machine.invalid\n")
    assert vm.push("origin", "HEAD:refs/heads/mine").returncode == 0


def test_a_push_whose_commits_git_cannot_list_is_refused(tmp_path):
    p = hook(VERITY[0], f"refs/heads/x {A} refs/heads/x {ZERO}", tmp=tmp_path, GIT_DIR=str(tmp_path / "no-repository"))
    assert p.returncode == 1 and "git could not list the commits this push adds" in p.stderr
    assert hook(VERITY[0], f"(delete) {ZERO} refs/heads/x {B}", tmp=tmp_path, GIT_DIR=str(tmp_path / "no-repository")).returncode == 0


def test_a_name_given_on_a_command_line_or_in_the_environments_configuration_is_not_the_checkouts(vm):
    """A wrapper that puts `-c user.name=...` on every git of a script puts it on the push too, and a script that exports a
    configuration (GIT_CONFIG_COUNT) leaves it in the shell that commits and pushes: neither is what the checkout's files
    hold."""
    sha = vm.commit("another's", **TRIAL)
    p = vm.shell("git -c user.name=trial -c user.email=trial@invalid push origin HEAD:refs/heads/mine")
    assert p.returncode != 0 and f"{sha[:12]} {AS_TRIAL}." in p.stderr and f"this checkout's identity, {OWN}," in p.stderr
    vm.git("switch", "-q", "-c", "other", "main")
    p = vm.shell("export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=user.name GIT_CONFIG_VALUE_0=trial GIT_CONFIG_KEY_1=user.email "
                 "GIT_CONFIG_VALUE_1=trial@invalid\ngit commit -q --allow-empty -m one\ngit push origin HEAD:refs/heads/other\n")
    assert vm.who("-1") == {"trial <trial@invalid> | trial <trial@invalid>"}
    assert p.returncode != 0 and AS_TRIAL in p.stderr and f"this checkout's identity, {OWN}," in p.stderr and not vm.heads()


def test_a_tag_is_held_as_a_branch_is(vm):
    sha = vm.commit("another's", **TRIAL)
    vm.git("tag", "v1")
    p = vm.push("origin", "v1")
    assert p.returncode != 0 and f"{sha[:12]} {AS_TRIAL}." in p.stderr
    assert not vm.git("for-each-ref", "refs/tags", cwd=vm.origin)


def test_a_push_by_url_is_held_to_its_lines_alone_and_the_refusal_never_prints_the_url(vm):
    sha = vm.commit("another's", **TRIAL)
    p = vm.push(str(vm.origin), "HEAD:refs/heads/mine")
    ours = [line for line in p.stderr.splitlines() if line.startswith("pre-push")]
    assert p.returncode != 0 and len(ours) == 1 and f"{sha[:12]} {AS_TRIAL}." in ours[0]
    assert str(vm.origin) not in ours[0] and "--remotes=<remote>" in ours[0] and "push to the remote by its name" in ours[0]


#: the one identity let through beside the checkout's, as `research queue cross` gives it to git for a crossing's commits, and
#: a name of a lane's own, as a head crossed by hand carries
QUEUES = "research queue <queue@research.invalid>"
QUEUE = {"GIT_AUTHOR_NAME": "research queue", "GIT_AUTHOR_EMAIL": "queue@research.invalid",
         "GIT_COMMITTER_NAME": "research queue", "GIT_COMMITTER_EMAIL": "queue@research.invalid"}
BY_HAND = {"GIT_AUTHOR_NAME": "restack carry", "GIT_AUTHOR_EMAIL": "noreply@invalid",
           "GIT_COMMITTER_NAME": "restack carry", "GIT_COMMITTER_EMAIL": "noreply@invalid"}


def crossed(vm: VM, names: dict[str, str]) -> tuple[str, str, list[str]]:
    """A head the agent pushed to its branch `mine`, crossed over a move the remote's `main` landed since, as `research queue
    cross` crosses one: a merge of the move's parent, then the head mapped across the move with the move as a second parent,
    each made by `commit-tree` with `names` in git's environment. The head as pushed, the crossed head, and the two commits
    the crossing made."""
    vm.git("push", "-q", "origin", "main")
    vm.git("switch", "-q", "-c", "work")
    old = vm.commit("the lane's own change")
    assert vm.push("origin", "HEAD:refs/heads/mine").returncode == 0
    vm.git("switch", "-q", "main")
    base, move = vm.commit("the move's parent", **TRIAL), vm.commit("the move", **TRIAL)
    vm.git("push", "-q", "origin", "main")
    tree = vm.git("rev-parse", f"{old}^{{tree}}").strip()
    merge = vm.git("commit-tree", tree, "-p", old, "-p", base, "-m", f"Merge {base}", **names).strip()
    new = vm.git("commit-tree", tree, "-p", merge, "-p", move, "-m", f"Mapped across {move}", **names).strip()
    assert set(vm.git("rev-list", new, "--not", old, "origin/main").split()) == {merge, new}
    return old, new, [merge, new]


def cross_push(vm: VM, old: str, new: str) -> subprocess.CompletedProcess:
    """The push `research queue cross --push` makes with the caller's own git, a lease on the old head."""
    return vm.push(f"--force-with-lease=refs/heads/mine:{old}", "origin", f"{new}:refs/heads/mine")


def test_a_head_crossed_as_the_queue_crosses_one_passes_under_the_queues_identity(vm):
    old, new, _ = crossed(vm, QUEUE)
    assert vm.who(f"{new} --not {old} origin/main") == {f"{QUEUES} | {QUEUES}"}
    p = cross_push(vm, old, new)
    assert p.returncode == 0, p.stderr
    assert vm.git("rev-parse", "refs/heads/mine", cwd=vm.origin).strip() == new


def test_the_same_head_crossed_by_hand_under_a_name_of_a_lanes_own_is_refused_naming_its_commits(vm):
    old, new, made = crossed(vm, BY_HAND)
    p = cross_push(vm, old, new)
    assert p.returncode != 0 and vm.git("rev-parse", "refs/heads/mine", cwd=vm.origin).strip() == old
    assert "this push adds 2 commits that" in p.stderr
    assert all(f"{sha[:12]} author and committer restack carry <noreply@invalid>" in p.stderr for sha in made)


def test_the_one_identity_let_through_beside_the_checkouts_is_the_queues_and_an_allowance_is_one_entry(vm):
    load = "import json, sys; sys.dont_write_bytecode = True; sys.path.insert(0, sys.argv[1]); import pre_push; "
    listed = subprocess.run(["python3", "-c", load + "print(json.dumps(pre_push.OTHERS))", str(GUARD)], cwd=vm.ws,
                            capture_output=True, text=True, check=True, env=vm.env).stdout
    assert list(json.loads(listed)) == [QUEUES] and "research queue cross" in json.loads(listed)[QUEUES]
    code = (load + "pre_push.OTHERS.update({who: 'a tool that commits as itself' for who in sys.argv[4:]}); "
            "sys.exit(pre_push.main(sys.argv[2:4]))")

    def judged(sha: str, *others: str) -> int:
        return subprocess.run(["python3", "-c", code, str(GUARD), "origin", str(vm.origin), *others], cwd=vm.ws,
                              input=f"refs/heads/work {sha} refs/heads/work {ZERO}\n", capture_output=True, text=True,
                              env={**vm.env, "CURSOR_CONVERSATION_ID": ME}).returncode

    start = vm.git("rev-parse", "HEAD").strip()
    for near in ({**QUEUE, "GIT_AUTHOR_EMAIL": "queue@invalid"}, {**QUEUE, "GIT_COMMITTER_NAME": "queue"}, BY_HAND):
        assert judged(vm.commit("near the queue's, and not it", **near)) == 1, near
        vm.git("reset", "-q", "--hard", start)
    assert judged(vm.commit("the queue's", **QUEUE)) == 0
    vm.git("reset", "-q", "--hard", start)
    sha = vm.commit("a tool's", **TRIAL)
    assert judged(sha) == 1 and judged(sha, "trial <other@invalid>") == 1
    assert judged(sha, "trial <trial@invalid>") == 0
