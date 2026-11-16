"""infra/control/fixtures_export.sh's candidate pass: each of origin's `queue/<sha12>` branches has its registry exported with
main's code, once, and origin needs no `next`."""
import hashlib
import os
import re
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "infra" / "control" / "fixtures_export.sh"
STUB = """#!/bin/sh
for a; do [ "$prev" = --repo ] && repo=$a; prev=$a; done
head=$(git -C "$repo" rev-parse HEAD)
echo "$(pwd) $repo $head $(cat "$repo/fixtures/artifacts.json") $(printf %s "$RESEARCH_EXPORT_BUCKET" | sha256sum | cut -c1-16)" >> "$RECORD"
! grep -qx "$head" "$FAIL" 2>/dev/null
"""


def git(*args, cwd):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def commit(origin, registry, branch):
    git("checkout", "-q", "-B", branch, "main", cwd=origin)
    (origin / "fixtures" / "artifacts.json").write_text(registry)
    git("commit", "-q", "-am", registry, cwd=origin)
    sha = git("rev-parse", "HEAD", cwd=origin)
    git("checkout", "-q", "main", cwd=origin)
    git("branch", "-q", "-m", branch, f"queue/{sha[:12]}", cwd=origin)
    return sha


class Pod:
    def __init__(self, tmp):
        self.tmp, self.origin, self.repo = tmp, tmp / "origin", tmp / "repo"
        (self.origin / "fixtures").mkdir(parents=True)
        (self.origin / "tools" / "research").mkdir(parents=True)
        (self.origin / "tools" / "research" / "x").write_text("")
        git("init", "-q", "-b", "main", cwd=self.origin)
        (self.origin / "fixtures" / "artifacts.json").write_text("main")
        git("add", "-A", cwd=self.origin)
        git("commit", "-q", "-m", "main", cwd=self.origin)
        git("clone", "-q", str(self.origin), str(self.repo), cwd=tmp)
        (tmp / "bin").mkdir()
        (tmp / "bin" / "python3.12").write_text(STUB)
        (tmp / "bin" / "python3.12").chmod(0o755)
        (tmp / "r2.env").write_text("")
        self.record, self.fail, self.done = tmp / "record", tmp / "fail", tmp / "done"

    def candidates(self):
        """One candidate pass, its log lines, and each export as (cwd, --repo, HEAD there, its registry, the bucket's hash)."""
        fn = re.search(r"^candidates\(\) \{\n.*?^\}\n", SCRIPT.read_text(), re.M | re.S).group(0)
        fn = fn.replace("/root/.config/verity/r2.env", str(self.tmp / "r2.env"))
        env = {**os.environ, "PATH": f"{self.tmp / 'bin'}:{os.environ['PATH']}", "REPO": str(self.repo), "W": str(self.tmp / "w"),
               "DONE": str(self.done), "RECORD": str(self.record), "FAIL": str(self.fail)}
        self.record.unlink(missing_ok=True)
        out = subprocess.run(["sh", "-c", f"{fn}candidates", str(SCRIPT)], env=env, capture_output=True, text=True, check=True)
        runs = [tuple(line.split()) for line in self.record.read_text().splitlines()] if self.record.exists() else []
        return out.stdout.splitlines(), runs


def bucket_hash():
    """The main pass's bucket, as the stub records it: the candidate pass must export to the same one."""
    bucket = re.search(r"PYTHONPATH=src RESEARCH_EXPORT_BUCKET=(\S+) timeout", SCRIPT.read_text()).group(1)
    return hashlib.sha256(bucket.encode()).hexdigest()[:16]


def test_each_candidate_is_exported_at_its_own_tree_with_mains_code_to_the_main_passs_bucket(tmp_path):
    pod = Pod(tmp_path)
    a, b = commit(pod.origin, "a", "x"), commit(pod.origin, "b", "y")
    assert not git("for-each-ref", "refs/heads/next", "refs/heads/checked", cwd=pod.origin)
    log, runs = pod.candidates()
    code, w = str(pod.repo / "tools" / "research"), str(tmp_path / "w")
    assert sorted(runs) == sorted([(code, w, a, "a", bucket_hash()), (code, w, b, "b", bucket_hash())])
    assert sorted(line.split(" ", 1)[1] for line in log) == sorted([f"candidate {a} pass rc=0", f"candidate {b} pass rc=0"])
    assert not (tmp_path / "w").exists() and git("worktree", "list", "--porcelain", cwd=pod.repo).count("worktree ") == 1


def test_a_candidate_exported_whole_is_not_exported_again_a_failed_one_is_and_a_dropped_one_is_not(tmp_path):
    pod = Pod(tmp_path)
    a, b = commit(pod.origin, "a", "x"), commit(pod.origin, "b", "y")
    pod.fail.write_text(f"{b}\n")
    log, runs = pod.candidates()
    assert sorted(r[2] for r in runs) == sorted([a, b]) and f"candidate {b} pass rc=1" in " ".join(log)
    pod.fail.unlink()
    git("branch", "-q", "-D", f"queue/{a[:12]}", cwd=pod.origin)
    c = commit(pod.origin, "c", "z")
    _, runs = pod.candidates()
    assert sorted(r[2] for r in runs) == sorted([b, c])
    git("branch", "-q", "-D", f"queue/{b[:12]}", cwd=pod.origin)
    assert pod.candidates() == ([], [])
    assert a not in git("for-each-ref", "refs/remotes/origin/queue/", cwd=pod.repo)
