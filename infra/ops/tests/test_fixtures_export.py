"""infra/control/fixtures_export.sh and its helper fixtures_export.py, run as the control pod runs them: a copy of each
under a made-up /data/infra, an origin with pull request heads that the script clones without blobs, as the pod's clone is,
and directories for the private store's remote and for the public export. `research data export` is a stub that records
what it was given, except in the tests that say the real one runs."""
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from research.store.local import LocalStore, manifest_key

CONTROL = Path(__file__).resolve().parents[3] / "infra" / "control"
SCRIPT, HELPER = CONTROL / "fixtures_export.sh", CONTROL / "fixtures_export.py"
REGISTRY = "fixtures/artifacts.json"
GIT = shutil.which("git")

#: `python3.12 -m research data export ...` in the stubbed tests: one JSON line per call, and nothing sent anywhere
STUB = """
import hashlib, json, os, subprocess, sys
from pathlib import Path
argv = sys.argv[1:]
repo = Path(argv[argv.index("--repo") + 1])
files = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file() and ".git" not in p.relative_to(repo).parts)
head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
call = {"argv": argv, "cwd": os.getcwd(), "pythonpath": os.environ.get("PYTHONPATH"),
        "bucket_sha256": hashlib.sha256(os.environ.get("RESEARCH_EXPORT_BUCKET", "").encode()).hexdigest(),
        "repo": str(repo), "head": head, "files": files,
        "registry": json.loads((repo / "fixtures" / "artifacts.json").read_text())}
with open(CALLS, "a") as f:
    f.write(json.dumps(call) + "\\n")
print("stub export:", " ".join(argv))
"""
#: git, after whose every fetch into a clone somebody else fetches pull request 7's head into the same clone
INTERLOPER = """#!/bin/sh
{git} "$@"; rc=$?
[ "$1" = -C ] && case " $* " in *" fetch "*) {git} -C "$2" fetch -q origin pull/7/head ;; esac
exit $rc
"""
#: how the queue's loop takes the bucket's name from the deployed script, whose path is $1
QUEUE_GREP = "grep -o 'RESEARCH_EXPORT_BUCKET=[^ ]*' \"$1\""
QUEUE_READS = QUEUE_GREP + " | head -1 | cut -d= -f2"


def git(*args, cwd):
    return subprocess.run([GIT, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def text_of(f: Path) -> str:
    """A file's text with the bucket's name blanked, so that an assertion over the text that fails doesn't print the name."""
    return re.sub(r"(RESEARCH_EXPORT_BUCKET=)[^ ]*", r"\1NAME", f.read_text())


class Pod:
    """The made-up pod and origin. A registry is written as {name: path}: the fixture `name` at `path`, tracked there
    unless the path is under fetched/, when its bytes are only in the private store."""

    def __init__(self, tmp: Path, *, real: bool = False):
        self.tmp, self.origin = tmp, tmp / "origin"
        self.data, self.root, self.bin = tmp / "data" / "infra", tmp / "root", tmp / "bin"
        self.clone, self.store = self.data / "fixtures-export-repo", self.data / "fixtures-export-store"
        self.log, self.script = self.data / "fixtures-export.log", self.data / "fixtures_export.sh"
        self.vault, self.public, self.calls_file = tmp / "vault", tmp / "public", tmp / "exports.jsonl"
        for d in (self.origin, self.data, self.root / ".config" / "verity", self.bin, tmp / "t", tmp / "home", tmp / "src"):
            d.mkdir(parents=True)
        self.cfg = tmp / "store.toml"
        self.cfg.write_text(f'[remote]\ntype = "fs"\npath = "{self.vault}"\n[export]\ntype = "fs"\npath = "{self.public}"\n')
        (self.root / ".config" / "verity" / "r2.env").write_text(f"RESEARCH_STORE_CONFIG={self.cfg}\n")
        #: where a lane puts its fixtures, pushing each fetched one to the private store's remote
        self.lane = LocalStore(tmp / "lane", config_path=self.cfg)
        self.ids: dict[str, str] = {}
        self.deploy(SCRIPT.read_text())
        shutil.copy(HELPER, self.data / "fixtures_export.py")
        stub = tmp / "stub_export.py"
        stub.write_text(STUB.replace("CALLS", repr(str(self.calls_file))))
        route = "" if real else f'[ "$1 $2 $3 $4" = "-m research data export" ] && exec "{sys.executable}" "{stub}" "$@"\n'
        self.shim("python3.12", f'#!/bin/sh\n{route}exec "{sys.executable}" "$@"\n')
        self.shim("sleep", '#!/bin/sh\nkill "$PPID"\n')        # the loop's sleep ends the loop: one pass, and no test waits
        git("init", "-q", "-b", "main", cwd=self.origin)
        # a clone without blobs fetches each blob it reads from origin, which must allow that
        git("config", "uploadpack.allowFilter", "true", cwd=self.origin)
        git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=self.origin)
        self.main = self.commit("main", {"tools/research/src/.keep": "", REGISTRY: {}, "README": "main\n"})

    def deploy(self, text: str) -> None:
        """Install a script as /data/infra/fixtures_export.sh, its pod paths and its origin replaced by the made-up ones."""
        at = "--no-checkout git@github.com:compute-verification/verity.git"
        found = all(s in text for s in (at, "/data/infra/", "/root/"))
        assert found, "the script names its origin, /data/infra/ and /root/"
        text = text.replace("/data/infra", str(self.data)).replace("/root/", f"{self.root}/")
        self.script.write_text(text.replace(at, f"--no-checkout file://{self.origin}"))
        self.script.chmod(0o755)

    def shim(self, name: str, text: str) -> None:
        (self.bin / name).write_text(text)
        (self.bin / name).chmod(0o755)

    def fixture(self, name: str, path: str) -> str:
        """The id of fixture ``name``, which is of its bytes and its file's name; one registered under fetched/ is pushed to
        the private store's remote."""
        if name not in self.ids:
            f = self.tmp / "src" / Path(path).name
            f.write_bytes(name.encode() * 64)
            self.ids[name] = self.lane.put_artifact("fixture/v1", {}, None, file=f)
            if path.startswith("fetched/"):
                assert self.lane.push(self.ids[name])["preserved"]
        return self.ids[name]

    def commit(self, branch: str, files: dict) -> str:
        """A commit on ``branch`` (which starts at main's tip if new) writing ``files``: text, a link to a Path, or for a
        registry {name: path}."""
        born = subprocess.run([GIT, "rev-parse", "-q", "--verify", "HEAD"], cwd=self.origin, capture_output=True).returncode == 0
        if born:
            new = subprocess.run([GIT, "rev-parse", "-q", "--verify", f"refs/heads/{branch}"], cwd=self.origin, capture_output=True).returncode
            git("checkout", "-q", *(["-b", branch, "main"] if new else [branch]), cwd=self.origin)
        for path, content in files.items():
            p = self.origin / path
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, dict):
                for name, where in content.items():
                    if not where.startswith("fetched/"):
                        (self.origin / where).parent.mkdir(parents=True, exist_ok=True)
                        (self.origin / where).write_bytes(name.encode() * 64)
                content = json.dumps(self.registry(content), indent=2) + "\n"
            p.symlink_to(content) if isinstance(content, Path) else p.write_text(content)
        git("add", "-A", cwd=self.origin)
        git("commit", "-q", "-m", f"{branch}: {', '.join(files)}", cwd=self.origin)
        head = git("rev-parse", "HEAD", cwd=self.origin)
        git("checkout", "-q", "main", cwd=self.origin)
        if branch == "main":
            self.main = head
        return head

    def pr(self, n: int, files: dict) -> str:
        """A commit on pull request ``n``'s branch, lane/N, and its new head, which origin's refs/pull/N/head follows."""
        head = self.commit(f"lane/{n}", files)
        git("update-ref", f"refs/pull/{n}/head", head, cwd=self.origin)
        return head

    def env(self) -> dict[str, str]:
        """This machine's environment with the made-up pod's commands ahead of its own."""
        path = os.pathsep.join([str(self.bin), *os.get_exec_path()])
        return {**os.environ, "PATH": path, "TMPDIR": str(self.tmp / "t"), "HOME": str(self.tmp / "home")}

    def run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.script), *args], env=self.env(), capture_output=True, text=True)

    def calls(self) -> list[dict]:
        return [json.loads(line) for line in self.calls_file.read_text().splitlines()] if self.calls_file.exists() else []

    def pr_calls(self) -> list[dict]:
        """The exports run for a pull request's head: every call but main's, which reads the clone itself."""
        return [c for c in self.calls() if c["repo"] != str(self.clone)]

    def registry(self, fixtures: dict[str, str]) -> dict[str, str]:
        """The registry document for {name: path}."""
        return {self.fixture(name, path): path for name, path in fixtures.items()}

    def clean(self) -> bool:
        """After a one-off the clone is as a pass left it: no worktree, no temporary directory, no ref but main's, and main
        checked out, which is what the queue's loop runs from it."""
        return (git("worktree", "list", cwd=self.clone).count("\n") == 0 and not list((self.tmp / "t").iterdir())
                and git("for-each-ref", "--format=%(refname)", "refs/export/", cwd=self.clone) in ("", "refs/export/main")
                and git("rev-parse", "HEAD", cwd=self.clone) == self.main)

    def public_names(self) -> set[str]:
        """The fixtures whose manifest the public export holds."""
        return {name for name, art in self.ids.items() if (self.public / manifest_key(art.split(":", 1)[1])).is_file()}


@pytest.fixture
def pod(tmp_path):
    return Pod(tmp_path)


A, B, C, D = {"a": "pkg/fixtures/a.bin"}, {"b": "fetched/b.bin"}, {"c": "pkg/fixtures/c.bin"}, {"d": "fetched/d.bin"}


def test_a_fetch_of_another_ref_between_the_loops_fetch_and_its_checkout_changes_nothing(pod):
    """The loop, one pass of it (its sleep ends it), with somebody else fetching pull request 7's head into the same
    clone after each of the script's fetches: the pass still checks out and exports main, and it is the only export, so the
    pull request's id reaches none. The log line of the pass is the success marker the cloud watch reads."""
    pod.commit("main", {REGISTRY: A})
    other = pod.pr(7, {REGISTRY: A | C})
    pod.shim("git", INTERLOPER.format(git=GIT))
    pod.run()
    log = pod.log.read_text()
    (main,) = pod.calls()
    assert main["repo"] == str(pod.clone) and main["head"] == pod.main != other and main["registry"] == pod.registry(A), log
    assert main["argv"][-2:] == ["--repo", str(pod.clone)] and git("rev-parse", "HEAD", cwd=pod.clone) == pod.main
    watch = tomllib.loads((CONTROL / "cloud_watch.toml").read_text())
    (marker,) = [loop["pattern"] for loop in watch["loop"] if loop["name"] == "Fixture export"]
    assert len(re.findall(marker, log, re.M)) == 1 and re.search(rf"^\S+Z pass rc=0 {pod.main[:7]}", log, re.M), log


def test_the_interloper_does_move_what_a_checkout_of_the_last_fetch_reads(pod):
    """The control for the test above: under the same interloper, a fetch of main and then a checkout of whatever the clone
    fetched last lands on the pull request's head, not on main."""
    other = pod.pr(7, {"README": "a pull request\n"})
    pod.shim("git", INTERLOPER.format(git=GIT))
    git("clone", "-q", f"file://{pod.origin}", str(pod.clone), cwd=pod.tmp)
    subprocess.run(["sh", "-c", 'git -C "$1" fetch -q origin main && git -C "$1" checkout -q -f --detach FETCH_HEAD', "sh", str(pod.clone)],
                   env=pod.env(), check=True)
    assert git("rev-parse", "HEAD", cwd=pod.clone) == other != pod.main


def test_neither_the_script_nor_its_helper_reads_the_clones_last_fetch():
    """No FETCH_HEAD in either: the script's checkout names the ref it fetched main into."""
    script, helper = text_of(SCRIPT), text_of(HELPER)
    assert "FETCH_HEAD" not in script and "FETCH_HEAD" not in helper
    assert "fetch -q origin +refs/heads/main:refs/export/main && " in script and "checkout -q -f --detach refs/export/main" in script
    assert script.count(" checkout -") == 1 and helper.count('"checkout"') == 1, "one checkout in each, by name"


def test_what_the_queues_loop_takes_from_the_script_is_as_it_takes_it(pod):
    """The queue's loop runs an export of its own on the pod with three things of this script's. The bucket: it reads the
    name from the deployed script's text, so the script holds one line of that form, and what the loop reads there is what
    the script's own exports are given. The export store's path. And the clone's checkout, whose `research` it runs: main's
    after a pass and after a one-off."""
    pod.commit("main", {REGISTRY: A})
    pod.pr(7, {REGISTRY: A | C})
    assert pod.run("once").returncode == 0 and pod.run("pr", "7").returncode == 0
    sh = ["sh", "-c"]
    forms = len(subprocess.run([*sh, QUEUE_GREP, "sh", str(pod.script)], capture_output=True, text=True, check=True).stdout.splitlines())
    name = subprocess.run([*sh, QUEUE_READS, "sh", str(pod.script)], capture_output=True, text=True, check=True).stdout.strip()
    read, nothing = hashlib.sha256(name.encode()).hexdigest(), hashlib.sha256(b"").hexdigest()
    del name
    assert forms == 1 and read != nothing, "one line names the bucket in the form the queue's loop reads"
    given = [c["bucket_sha256"] for c in pod.calls()]
    assert given == [read, read], "main's export and the one-off's are given the name the queue's loop reads"
    assert all(c["argv"][:6] == ["-m", "research", "data", "export", "--store", str(pod.store)] for c in pod.calls())
    script = text_of(SCRIPT)
    assert "\nREPO=/data/infra/fixtures-export-repo\n" in script and "\nSTORE=/data/infra/fixtures-export-store\n" in script
    assert pod.clean() and (pod.clone / "tools" / "research" / "src").is_dir()


def test_a_dry_pass_passes_the_dry_run_on_and_leaves_mains_checkout_as_it_is(pod):
    """`once --dry-run` after main moved: the export gets --dry-run, and the clone stays on the main it had."""
    pod.commit("main", {REGISTRY: A})
    assert pod.run("once").returncode == 0
    before = pod.main
    pod.commit("main", {"README": "main moved\n"})
    p = pod.run("once", "--dry-run")
    assert p.returncode == 0, p.stdout + p.stderr
    (main,) = pod.calls()[1:]
    assert main["argv"][-3:] == ["--repo", str(pod.clone), "--dry-run"] and main["head"] == before != pod.main
    assert git("rev-parse", "HEAD", cwd=pod.clone) == before and f"pass rc=0 {before[:7]}\n" in p.stdout
    assert pod.run("once", "--fresh").returncode == 2 and pod.run("twice").returncode == 2 and len(pod.calls()) == 2


def test_the_one_off_exports_only_the_pull_requests_new_ids_and_leaves_the_clone_as_it_was(pod):
    """`pr N`: a registry cut to the one id the head registers and main doesn't, in a worktree at the head that holds that
    registry and the one tracked fixture, exported as main's is, with main's code from main's checkout. The ref and the
    worktree are gone afterwards. A pull request that registers nothing new, one whose registry can't be read, one that
    doesn't exist and a number that isn't one export nothing."""
    pod.commit("main", {REGISTRY: A | B})
    assert pod.run("once").returncode == 0
    head = pod.pr(7, {REGISTRY: A | B | C, "tools/research/src/research/__main__.py": "raise SystemExit('a head runs nothing')\n"})
    pod.pr(8, {"README": "no fixture\n"})
    pod.pr(9, {REGISTRY: "<<<<<<< ours\n{}\n"})
    p = pod.run("pr", "7")
    assert p.returncode == 0, p.stdout + p.stderr
    main, (one,) = pod.calls()[0], pod.pr_calls()
    assert one["head"] == head and one["registry"] == pod.registry(C), "only what main lacks"
    assert one["files"] == [REGISTRY, "pkg/fixtures/c.bin"], "nothing else of the head is checked out"
    assert one["argv"][:8] == main["argv"][:8] and one["argv"][8:] == ["--repo", one["repo"], "--no-previous"]
    assert one["cwd"] == main["cwd"] == str(pod.clone / "tools" / "research") and one["pythonpath"] == "src", "main's code"
    assert f"pr 7 at {head[:9]}: 1 new id(s) against main {pod.main[:9]}\n  {pod.ids['c']}  pkg/fixtures/c.bin\n" in p.stdout
    assert f"pr 7 at {head[:9]}: export rc=0\n" in p.stdout and pod.clean()
    p = pod.run("pr", "8")
    assert p.returncode == 0 and "registers nothing that main" in p.stdout, p.stdout + p.stderr
    p = pod.run("pr", "9")
    assert p.returncode == 1 and f"pr 9: {REGISTRY} is not JSON" in p.stdout, p.stdout + p.stderr
    assert pod.run("pr", "99").returncode == 1 and pod.run("pr", "seven").returncode == 2 and pod.run("pr").returncode == 2
    assert len(pod.pr_calls()) == 1 and pod.clean()


def test_the_one_off_exports_no_id_mains_registry_has_ever_named(pod):
    """Main registered `b` once and has dropped it. A head that branched before the drop still holds `b`, and another,
    branched after it, registers `b` again: `b` is in each registry and not in main's, but main has named it, so it is
    no head's to export. Only `c`, which the first added, is exported."""
    pod.commit("main", {REGISTRY: A | B})
    pod.pr(7, {REGISTRY: A | B | C})
    pod.commit("main", {REGISTRY: A})
    pod.pr(9, {REGISTRY: A | B})
    assert pod.run("once").returncode == 0
    p = pod.run("pr", "7")
    assert p.returncode == 0 and [c["registry"] for c in pod.pr_calls()] == [pod.registry(C)], p.stdout + p.stderr
    p = pod.run("pr", "9")
    assert p.returncode == 0 and "registers nothing that main" in p.stdout and len(pod.pr_calls()) == 1, p.stdout + p.stderr


def test_a_head_whose_new_entries_lead_out_of_its_worktree_is_refused_whole(pod):
    """A new entry at a path outside the repository, or at a link the head tracks, would have the export hash the pod's
    own files. Neither head gets an export, not even of the good new id it registers beside the bad one."""
    pod.commit("main", {REGISTRY: A})
    assert pod.run("once").returncode == 0
    secret = pod.root / ".config" / "verity" / "r2.env"
    for n in (5, 6):
        pod.pr(n, {REGISTRY: A | C})
    good = pod.registry(A | C)
    link = pod.pr(5, {REGISTRY: json.dumps(good | {"art:" + "5" * 64: "pkg/fixtures/link"}), "pkg/fixtures/link": secret})
    out = pod.pr(6, {REGISTRY: json.dumps(good | {"art:" + "6" * 64: "../../root/.config/verity/r2.env"})})
    for n, at, why in ((5, link, "pkg/fixtures/link is a link, which the export would follow out of the worktree"),
                       (6, out, "its registry names a path outside the repository (../../root/.config/verity/r2.env)")):
        p = pod.run("pr", str(n))
        assert p.returncode == 1 and f"pr {n} at {at[:9]}: 2 new id(s)" in p.stdout and f"pr {n} at {at[:9]}: {why}" in p.stdout, p.stdout
    assert pod.pr_calls() == [] and pod.clean()


def test_a_one_off_waits_for_the_pass_that_holds_the_lock(pod):
    """While something holds the pass lock, as a pass of the loop does, `pr N` says that it waits and starts no export;
    once the lock is free it runs."""
    pod.commit("main", {REGISTRY: A})
    assert pod.run("once").returncode == 0
    pod.pr(7, {REGISTRY: A | C})
    held = os.open(pod.root / "fixtures-export-pass.lock", os.O_WRONLY | os.O_CREAT)
    fcntl.flock(held, fcntl.LOCK_EX)
    with subprocess.Popen([str(pod.script), "pr", "7"], env=pod.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
        try:
            first = p.stdout.readline()
            waited = "waiting for the pass that holds" in first and pod.pr_calls() == []
        finally:
            os.close(held)                                  # the lock is free
        rest = p.stdout.read()
    assert waited and p.returncode == 0 and len(pod.pr_calls()) == 1, first + rest


def test_the_one_offs_dry_run_names_the_ids_and_sends_nothing(tmp_path):
    """With the real `research data export` and a directory as the public export: `pr N --dry-run` names the head's two new
    ids, one tracked and one fetched, and the export stays empty; without --dry-run those two land there and main's don't,
    since the one-off passes only the head's, and a second run sends no object. Nothing in the head's tree ran, though it
    holds a `research` of its own."""
    pod = Pod(tmp_path, real=True)
    pod.commit("main", {REGISTRY: A | B})
    ran = tmp_path / "ran"
    head = pod.pr(7, {REGISTRY: A | B | C | D,
                      "tools/research/src/research/__init__.py": f"open({str(ran)!r}, 'w').close()\n",
                      "tools/research/src/research/__main__.py": f"open({str(ran)!r}, 'w').close()\n"})
    git("clone", "-q", "--filter=blob:none", f"file://{pod.origin}", str(pod.clone), cwd=pod.tmp)
    p = pod.run("pr", "7", "--dry-run")
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"pr 7 at {head[:9]}: 2 new id(s) against main {pod.main[:9]}\n" in p.stdout
    assert f"  {pod.ids['c']}  pkg/fixtures/c.bin\n" in p.stdout and f"  {pod.ids['d']}  fetched/d.bin\n" in p.stdout
    assert pod.ids["a"] not in p.stdout and pod.ids["b"] not in p.stdout
    assert "(dry run): 2 id(s)" in p.stderr and "export rc=0 (dry run: nothing sent)" in p.stdout
    assert not [f for f in pod.public.rglob("*") if f.is_file()], "a dry run sent something"
    assert pod.clean()
    sent = r"export to \S+: (\d+) id\(s\), \d+ blob\(s\); uploaded (\d+) object"        # a manifest and a blob for each id
    for uploaded in ("4", "0"):                             # the export's record of what landed covers a one-off's ids too
        p = pod.run("pr", "7")
        assert p.returncode == 0 and re.findall(sent, p.stderr) == [("2", uploaded)], p.stdout + p.stderr
        assert pod.public_names() == {"c", "d"} and not ran.exists() and pod.clean()
