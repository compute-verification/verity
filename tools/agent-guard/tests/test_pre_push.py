"""A push an agent's script or child process makes meets the branch rule a typed `git push` meets: the git guard (git_guard.sh)
gives every push the pre-push hook (pre_push.py), which refuses a branch registered to another agent and `main`, `checked`
and `next`, and then runs the hook the push would have run without it, for a push to the repositories whose `main`,
`checked` and `next` drive lands (verity and infra). A push no agent started (a person's, the queue's) is not judged, and
neither is one to another repository (the notes, whose `main` every agent's `research notes sync` pushes)."""
import json
import os
import subprocess
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1]
GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
ME, OTHER = "bc-11111111-1111-4111-8111-111111111111", "bc-22222222-2222-4222-8222-222222222222"
REAL_GIT = "/usr/bin/git"


class VM:
    """An agent VM in miniature: `git` on PATH is the guard, whose pushes go through a hooks directory with pre_push.py in it;
    `origin` stands in for verity on GitHub (AGENT_GUARD_PUSH_REMOTES), `node` for a node's bare repository."""

    def __init__(self, tmp: Path):
        self.tmp, self.origin, self.node, self.ws = tmp, tmp / "origin.git", tmp / "node.git", tmp / "ws"
        self.state, bin_, hooks = tmp / "state", tmp / "bin", tmp / "hooks"
        for d in (bin_, hooks, self.state):
            d.mkdir()
        (bin_ / "git").write_text(f"#!/bin/sh\nexec bash {GUARD / 'git_guard.sh'} \"$@\"\n")
        (hooks / "pre-push").write_text(f"#!/bin/sh\nexec python3 {GUARD / 'pre_push.py'} \"$@\"\n")
        for f in (bin_ / "git", hooks / "pre-push"):
            f.chmod(0o755)
        base = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "AGENT_GUARD_", "CURSOR_", "VERITY_"))}
        self.env = {**base, **GIT_ENV, "PATH": f"{bin_}:{base.get('PATH', '/usr/bin:/bin')}", "VERITY_REAL_GIT": REAL_GIT,
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

    def git(self, *args: str, cwd: Path | None = None) -> str:
        return subprocess.run([REAL_GIT, *args], cwd=cwd or self.ws, check=True, capture_output=True, text=True,
                              env=self.env).stdout

    def push(self, *args: str, agent: str | None = ME, **env: str) -> subprocess.CompletedProcess:
        """`push.sh ARGS` run by `agent`'s shell (None: no agent's), as a script run from the agent's command would be."""
        e = {**self.env, **env, **({"CURSOR_CONVERSATION_ID": agent} if agent else {})}
        return subprocess.run([str(self.script), *args], cwd=self.ws, capture_output=True, text=True, env=e)

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
