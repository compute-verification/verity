"""infra/control/fixtures_export.sh and its helper fixtures_export.py, run as the control pod runs them: a copy of each
under a made-up /data/infra, an origin with pull request heads that the script clones without blobs, as the pod's clone is,
and directories for the private store's remote and for the public export. `research data export` is a stub that records
what it was given, except in the tests that say the real one runs."""
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from research.queue import label_target
from research.store import labels_sync, vocab
from research.store.local import LocalStore, manifest_key

CONTROL = Path(__file__).resolve().parents[3] / "infra" / "control"
SCRIPT, HELPER = CONTROL / "fixtures_export.sh", CONTROL / "fixtures_export.py"
REGISTRY = "fixtures/artifacts.json"
GIT = shutil.which("git")

#: `python3.12 -m research data export ...` in the stubbed tests: one JSON line per call, and nothing sent anywhere
STUB = """
import json, os, subprocess, sys
from pathlib import Path
argv = sys.argv[1:]
repo = Path(argv[argv.index("--repo") + 1])
files = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file() and ".git" not in p.relative_to(repo).parts)
head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
call = {"argv": argv, "cwd": os.getcwd(), "pythonpath": os.environ.get("PYTHONPATH"),
        "bucket": bool(os.environ.get("RESEARCH_EXPORT_BUCKET")), "repo": str(repo), "head": head, "files": files,
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


def git(*args, cwd):
    return subprocess.run([GIT, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


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
        #: where a lane puts its fixtures and writes its marks, pushing both to the private store's remote
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
        assert at in text and "/data/infra/" in text and "/root/" in text
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

    def mark(self, n: int, head: str, ready: bool = True, *, minute: int = 0) -> None:
        """A `ready` label on pr:N@HEAD, as `research queue ready` (or, false, `eject`) leaves it on the store's remote."""
        target = label_target(n, head)
        self.lane.add_label(target, vocab.READY, ready, "lane", ts=f"2026-01-01T00:{minute:02d}:00Z")
        assert labels_sync.push_targets(self.lane, [target])["failed"] == []

    def run(self, *args: str) -> subprocess.CompletedProcess:
        env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "TMPDIR": str(self.tmp / "t"), "HOME": str(self.tmp / "home")}
        return subprocess.run([str(self.script), *args], env=env, capture_output=True, text=True)

    def calls(self) -> list[dict]:
        return [json.loads(line) for line in self.calls_file.read_text().splitlines()] if self.calls_file.exists() else []

    def heads_calls(self) -> list[dict]:
        """The exports run for pull request heads: every call but main's, which reads the clone itself."""
        return [c for c in self.calls() if c["repo"] != str(self.clone)]

    def registry(self, fixtures: dict[str, str]) -> dict[str, str]:
        """The registry document for {name: path}."""
        return {self.fixture(name, path): path for name, path in fixtures.items()}

    def clean(self) -> bool:
        """No worktree, no temporary directory and no one-off ref is left behind."""
        return (git("worktree", "list", cwd=self.clone).count("\n") == 0 and not list((self.tmp / "t").iterdir())
                and not git("for-each-ref", "refs/export/once/", cwd=self.clone))

    def public_names(self) -> set[str]:
        """The fixtures whose manifest the public export holds."""
        return {name for name, art in self.ids.items() if (self.public / manifest_key(art.split(":", 1)[1])).is_file()}


@pytest.fixture
def pod(tmp_path):
    return Pod(tmp_path)


A, B, C, D = {"a": "pkg/fixtures/a.bin"}, {"b": "fetched/b.bin"}, {"c": "pkg/fixtures/c.bin"}, {"d": "fetched/d.bin"}


def test_a_marked_heads_new_ids_are_exported_and_mains_are_not_passed_twice(pod):
    """One pass: main's registry from main's checkout, then for the marked pull request a registry cut to the two ids its
    head registers and main doesn't, in a worktree at the head that holds that registry and the one tracked fixture. Both
    exports run main's code from main's checkout."""
    pod.commit("main", {REGISTRY: A | B})
    head = pod.pr(7, {REGISTRY: A | B | C | D, "tools/research/src/research/__main__.py": "raise SystemExit('a head runs nothing')\n"})
    pod.mark(7, head)
    p = pod.run("once")
    assert p.returncode == 0, p.stdout + p.stderr
    main, (one,) = pod.calls()[0], pod.heads_calls()
    assert len(pod.calls()) == 2 and main["repo"] == str(pod.clone) and main["head"] == pod.main
    assert main["registry"] == pod.registry(A | B)
    assert main["argv"][-2:] == ["--repo", str(pod.clone)] and "--no-previous" not in main["argv"]
    assert one["registry"] == pod.registry(C | D), "only what main lacks"
    assert one["head"] == head and one["files"] == [REGISTRY, "pkg/fixtures/c.bin"], "nothing else of the head is checked out"
    assert one["argv"][:8] == main["argv"][:8] and one["argv"][8:] == ["--repo", one["repo"], "--no-previous"]
    for call in (main, one):
        assert call["cwd"] == str(pod.clone / "tools" / "research") and call["pythonpath"] == "src" and call["bucket"], "main's code"
    assert f"pr 7 at {head[:9]}: 2 new id(s), marked ready\n" in p.stdout and f"pr 7 at {head[:9]}: export rc=0\n" in p.stdout
    assert f"pass rc=0 {pod.main[:7]}" in p.stdout and "heads pass rc=0\n" in p.stdout
    assert pod.clean()


def test_an_unmarked_or_an_ejected_head_is_not_exported(pod):
    """Neither a head nobody marked nor one whose newest `ready` label is false (`research queue eject`) gets a worktree
    or an export; a mark on an older head of the pull request, which registered fewer ids, doesn't count for the new one."""
    pod.commit("main", {REGISTRY: A})
    pod.pr(8, {REGISTRY: A | C})
    ejected = pod.pr(9, {REGISTRY: A | D})
    pod.mark(9, ejected, minute=1)
    pod.mark(9, ejected, False, minute=2)
    pod.mark(10, pod.pr(10, {REGISTRY: A | {"e": "pkg/fixtures/e.bin"}}))
    pod.pr(10, {REGISTRY: A | {"e": "pkg/fixtures/e.bin", "f": "pkg/fixtures/f.bin"}})
    p = pod.run("once")
    assert p.returncode == 0, p.stdout + p.stderr
    assert [c["repo"] for c in pod.calls()] == [str(pod.clone)], "main's export and no other"
    for n, count in ((8, 1), (9, 1), (10, 2)):
        assert re.search(rf"^pr {n} at [0-9a-f]{{9}}: {count} new id\(s\), not marked ready: not exported$", p.stdout, re.M), p.stdout
    assert "3 pull request(s) on a branch of origin, 3 registering ids main lacks" in p.stdout and pod.clean()


def test_a_pass_that_cannot_pull_the_marks_exports_no_head_and_fails(pod):
    """The mark is pulled from the store's remote each pass. Without a remote the pass doesn't answer from the labels its
    store already holds: the head it exported a pass ago is not exported, and the heads pass fails, while main's goes on."""
    pod.commit("main", {REGISTRY: A})
    pod.mark(7, pod.pr(7, {REGISTRY: A | C}))
    p = pod.run("once")
    assert p.returncode == 0 and len(pod.heads_calls()) == 1, p.stdout + p.stderr
    pod.cfg.unlink()
    p = pod.run("once")
    assert p.returncode == 1 and len(pod.heads_calls()) == 1 and len(pod.calls()) == 3, p.stdout + p.stderr
    assert re.search(r"^pr 7 at [0-9a-f]{9}: 1 new id\(s\), not exported: QueueError: STALE: ", p.stdout, re.M), p.stdout
    assert "pass rc=0 " in p.stdout and "heads pass rc=1\n" in p.stdout


def test_a_head_behind_main_republishes_nothing_that_main_dropped(pod):
    """Two heads branched when main registered `b`; main has dropped it since. `b` is in their registries and not in
    main's, but it was in their merge base's, so it is not theirs: only `c`, which one of them added, is exported."""
    pod.commit("main", {REGISTRY: A | B})
    pod.mark(7, pod.pr(7, {REGISTRY: A | B | C}))
    pod.mark(8, pod.pr(8, {"README": "a head that changes no registry\n"}))
    pod.commit("main", {REGISTRY: A})
    p = pod.run("once")
    assert p.returncode == 0, p.stdout + p.stderr
    assert [c["registry"] for c in pod.heads_calls()] == [pod.registry(C)]
    assert "2 pull request(s) on a branch of origin, 1 registering ids main lacks" in p.stdout


def test_a_mark_carries_to_a_later_head_that_registers_the_same_new_ids_and_so_does_an_ejection(pod):
    """A push on top of the marked head that leaves its new ids as they were (a merge of main, a fix) needs no new mark,
    as in the queue; an ejection at the marked head holds at the later one too."""
    pod.commit("main", {REGISTRY: A})
    marked = pod.pr(7, {REGISTRY: A | C})
    pod.mark(7, marked, minute=1)
    later = pod.pr(7, {"README": "a fix\n"})
    p = pod.run("once")
    assert p.returncode == 0, p.stdout + p.stderr
    (one,) = pod.heads_calls()
    assert one["head"] == later != marked and one["registry"] == pod.registry(C)
    assert f"#7 at {later[:8]} takes the `ready` labels at {marked[:8]}" in p.stdout
    pod.mark(7, marked, False, minute=2)
    p = pod.run("once")
    assert p.returncode == 0 and len(pod.heads_calls()) == 1 and "not marked ready: not exported" in p.stdout, p.stdout + p.stderr


def test_a_head_whose_registry_cannot_be_read_stops_no_other_head(pod):
    """A head with a registry that isn't JSON (a conflict left in it) is named and passed over; the marked head beside it
    is still exported and the pass succeeds."""
    pod.commit("main", {REGISTRY: A})
    pod.pr(6, {REGISTRY: "<<<<<<< ours\n{}\n"})
    head = pod.pr(7, {REGISTRY: A | C})
    pod.mark(7, head)
    p = pod.run("once")
    assert p.returncode == 0, p.stdout + p.stderr
    assert [c["head"] for c in pod.heads_calls()] == [head] and "; registry unreadable at the head of 6\n" in p.stdout


def test_a_head_whose_new_entries_lead_out_of_its_worktree_is_refused_whole(pod):
    """A new entry at a path outside the repository, or at a link the head tracks, would have the export hash the pod's
    own files. Neither head gets an export, the heads pass fails naming each, and the marked head beside them is exported."""
    pod.commit("main", {REGISTRY: A})
    secret = pod.tmp / "root" / ".config" / "verity" / "r2.env"
    link = pod.pr(5, {REGISTRY: json.dumps(pod.registry(A) | {"art:" + "5" * 64: "pkg/fixtures/link"}), "pkg/fixtures/link": secret})
    out = pod.pr(6, {REGISTRY: json.dumps(pod.registry(A) | {"art:" + "6" * 64: "../../root/.config/verity/r2.env"})})
    head = pod.pr(7, {REGISTRY: A | C})
    for n, at in ((5, link), (6, out), (7, head)):
        pod.mark(n, at)
    p = pod.run("once")
    assert p.returncode == 1 and [c["head"] for c in pod.heads_calls()] == [head], p.stdout + p.stderr
    assert f"pr 5 at {link[:9]}: pkg/fixtures/link is a link, which the export would follow out of the worktree" in p.stdout
    assert f"pr 6 at {out[:9]}: its registry names a path outside the repository (../../root/.config/verity/r2.env)" in p.stdout
    assert "heads pass rc=1\n" in p.stdout and pod.clean()


def test_a_fetch_of_another_ref_between_the_loops_fetch_and_its_checkout_changes_nothing(pod):
    """The loop, one pass of it (its sleep ends it), with somebody else fetching pull request 7's head into the same
    clone after each of the script's fetches: main's pass still checks out and exports main, and the unmarked pull
    request's id reaches no export. The log line of main's pass is the success marker the cloud watch reads."""
    pod.commit("main", {REGISTRY: A})
    other = pod.pr(7, {REGISTRY: A | C})
    pod.shim("git", INTERLOPER.format(git=GIT))
    pod.run()
    log = pod.log.read_text()
    main = pod.calls()[0]
    assert main["repo"] == str(pod.clone) and main["head"] == pod.main != other and main["registry"] == pod.registry(A), log
    assert all(pod.ids["c"] not in c["registry"] for c in pod.calls()), "the pull request's id was exported"
    assert git("rev-parse", "HEAD", cwd=pod.clone) == pod.main
    watch = tomllib.loads((CONTROL / "cloud_watch.toml").read_text())
    (marker,) = [loop["pattern"] for loop in watch["loop"] if loop["name"] == "Fixture export"]
    assert len(re.findall(marker, log, re.M)) == 1 and re.search(rf"^\S+Z pass rc=0 {pod.main[:7]}", log, re.M), log
    assert len(pod.calls()) == 1 and re.search(r"^\S+Z heads pass rc=0$", log, re.M), log


def test_the_interloper_does_move_what_a_checkout_of_the_last_fetch_reads(pod):
    """The control for the test above: under the same interloper, a fetch of main and then a checkout of whatever the clone
    fetched last lands on the pull request's head, not on main."""
    other = pod.pr(7, {"README": "a pull request\n"})
    pod.shim("git", INTERLOPER.format(git=GIT))
    git("clone", "-q", f"file://{pod.origin}", str(pod.clone), cwd=pod.tmp)
    subprocess.run(["sh", "-c", 'git -C "$1" fetch -q origin main && git -C "$1" checkout -q -f --detach FETCH_HEAD', "sh", str(pod.clone)],
                   env={**os.environ, "PATH": f"{pod.bin}:{os.environ['PATH']}"}, check=True)
    assert git("rev-parse", "HEAD", cwd=pod.clone) == other != pod.main


def test_neither_the_script_nor_its_helper_reads_the_clones_last_fetch():
    """No FETCH_HEAD in either: the script's checkout names the ref it fetched main into."""
    for f in (SCRIPT, HELPER):
        assert "FETCH_HEAD" not in f.read_text(), f.name
    script = SCRIPT.read_text()
    assert "fetch -q origin +refs/heads/main:refs/export/main && " in script and "checkout -q -f --detach refs/export/main" in script
    assert script.count(" checkout -") == 1 and HELPER.read_text().count('"checkout"') == 1, "one checkout in each, by name"


def test_a_dry_pass_passes_the_dry_run_on_and_leaves_mains_checkout_as_it_is(pod):
    """`once --dry-run` after main moved: both exports get --dry-run, and the clone stays on the main it had."""
    pod.commit("main", {REGISTRY: A})
    assert pod.run("once").returncode == 0
    before = pod.main
    head = pod.pr(7, {REGISTRY: A | C})
    pod.mark(7, head)
    pod.commit("main", {"README": "main moved\n"})
    p = pod.run("once", "--dry-run")
    assert p.returncode == 0, p.stdout + p.stderr
    main, one = pod.calls()[1:]
    assert main["argv"][-3:] == ["--repo", str(pod.clone), "--dry-run"] and main["head"] == before != pod.main
    assert one["argv"][-4:] == ["--repo", one["repo"], "--no-previous", "--dry-run"] and one["head"] == head
    assert git("rev-parse", "HEAD", cwd=pod.clone) == before and f"pr 7 at {head[:9]}: export rc=0 (dry run: nothing sent)" in p.stdout
    assert pod.run("once", "--fresh").returncode == 2 and pod.run("twice").returncode == 2 and len(pod.calls()) == 3


def test_the_one_off_exports_one_pull_requests_new_ids_whether_or_not_it_is_marked(pod):
    """`pr N`: the unmarked head's new id, from a ref and a worktree of its own that are gone afterwards, with main's
    checkout left where it was; a pull request that registers nothing new, one that doesn't exist and a number that isn't
    one export nothing."""
    pod.commit("main", {REGISTRY: A})
    assert pod.run("once").returncode == 0
    head = pod.pr(7, {REGISTRY: A | C})
    pod.pr(8, {"README": "no fixture\n"})
    p = pod.run("pr", "7")
    assert p.returncode == 0, p.stdout + p.stderr
    (one,) = pod.heads_calls()
    assert one["head"] == head and one["registry"] == pod.registry(C)
    assert one["argv"][-3:] == ["--repo", one["repo"], "--no-previous"] and one["files"] == [REGISTRY, "pkg/fixtures/c.bin"]
    assert f"pr 7 at {head[:9]}: 1 new id(s) against main {pod.main[:9]}\n  {pod.ids['c']}  pkg/fixtures/c.bin\n" in p.stdout
    assert pod.clean() and git("rev-parse", "HEAD", cwd=pod.clone) == pod.main and not git("for-each-ref", "refs/export/pr/", cwd=pod.clone)
    p = pod.run("pr", "8")
    assert p.returncode == 0 and "registers nothing that main" in p.stdout and len(pod.heads_calls()) == 1
    assert pod.run("pr", "99").returncode == 1 and pod.run("pr", "seven").returncode == 2 and pod.run("pr").returncode == 2
    assert len(pod.heads_calls()) == 1 and pod.clean()


def test_the_one_offs_dry_run_names_the_ids_and_sends_nothing(tmp_path):
    """With the real `research data export` and a directory as the public export: `pr N --dry-run` names the head's two new
    ids, one tracked and one fetched, and the export stays empty; without --dry-run those two land there and main's don't,
    since the one-off passes only the head's."""
    pod = Pod(tmp_path, real=True)
    pod.commit("main", {REGISTRY: A | B})
    head = pod.pr(7, {REGISTRY: A | B | C | D})
    git("clone", "-q", "--filter=blob:none", f"file://{pod.origin}", str(pod.clone), cwd=pod.tmp)
    p = pod.run("pr", "7", "--dry-run")
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"pr 7 at {head[:9]}: 2 new id(s) against main {pod.main[:9]}\n" in p.stdout
    assert f"  {pod.ids['c']}  pkg/fixtures/c.bin\n" in p.stdout and f"  {pod.ids['d']}  fetched/d.bin\n" in p.stdout
    assert pod.ids["a"] not in p.stdout and pod.ids["b"] not in p.stdout
    assert "(dry run): 2 id(s)" in p.stderr and "export rc=0 (dry run: nothing sent)" in p.stdout
    assert not [f for f in pod.public.rglob("*") if f.is_file()], "a dry run sent something"
    assert pod.clean()
    p = pod.run("pr", "7")
    assert p.returncode == 0 and pod.public_names() == {"c", "d"}, p.stdout + p.stderr
    assert pod.clean()


def test_a_pass_sends_mains_fixtures_and_a_marked_heads_and_the_next_pass_sends_nothing(tmp_path):
    """With the real `research data export`: main's two fixtures and the marked head's two reach the public export, the
    unmarked head's doesn't, and nothing in the head's tree ran, though it holds a `research` of its own. A second pass
    sends no object: the export's record of what landed covers main's ids and the head's."""
    pod = Pod(tmp_path, real=True)
    pod.commit("main", {REGISTRY: A | B})
    ran = tmp_path / "ran"
    head = pod.pr(7, {REGISTRY: A | B | C | D,
                      "tools/research/src/research/__init__.py": f"open({str(ran)!r}, 'w').close()\n",
                      "tools/research/src/research/__main__.py": f"open({str(ran)!r}, 'w').close()\n"})
    pod.pr(8, {REGISTRY: A | B | {"e": "fetched/e.bin"}})
    pod.mark(7, head)
    p = pod.run("once")
    assert p.returncode == 0, p.stdout + p.stderr
    assert pod.public_names() == {"a", "b", "c", "d"} and not ran.exists()
    sent = r"export to \S+: (\d+) id\(s\), \d+ blob\(s\); uploaded (\d+) object"        # a manifest and a blob for each id
    assert re.findall(sent, p.stderr) == [("2", "4"), ("2", "4")], p.stderr
    p = pod.run("once")
    assert p.returncode == 0, p.stdout + p.stderr
    assert re.findall(sent, p.stderr) == [("2", "0"), ("2", "0")], p.stderr
    assert pod.public_names() == {"a", "b", "c", "d"} and pod.clean()
