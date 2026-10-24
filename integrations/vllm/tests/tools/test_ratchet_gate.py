"""`tools/ratchet_gate.py` on toy repositories: a branch's own commits judged together against its merge base, each queue merge
on its own, a new or moved ratchet, the changes it doesn't judge (tests and docs only), the exemptions (a revert, a pure move
the base's move_check certifies), and its command line."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "ratchet_gate.py"
_spec = importlib.util.spec_from_file_location("ratchet_gate", SCRIPT)
G = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("ratchet_gate", G)
_spec.loader.exec_module(G)

P01 = "integrations/vllm/tests/lint/allowlists/p01_core.json"
#: a stand-in for move_check: certified when every change is a 100% rename
MOVE_CHECK = """\
import subprocess, sys
out = subprocess.run(["git", "diff", "--name-status", "-M100%", sys.argv[1], sys.argv[2]], capture_output=True, text=True).stdout
sys.exit(0 if out and all(line.startswith("R100") for line in out.splitlines()) else 1)
"""
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def _entry(file: str, symbol: str) -> dict:
    return {"file": file, "kind": "k", "symbol": symbol, "detail": "d"}


def _allowlist(*entries: dict) -> str:
    return json.dumps({"rule": "p01", "entries": list(entries)})


class Repo:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir()
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.root, env=ENV, check=True, capture_output=True, text=True).stdout.strip()

    def commit(self, message: str, files: dict[str, str | None]) -> str:
        for path, text in files.items():
            p = self.root / path
            if text is None:
                self.git("rm", "-q", path)
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            self.git("add", path)
        self.git("commit", "-q", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD")

    def queue(self, pr: int, title: str, head: str) -> str:
        """the queue's merge of ``head`` onto the current branch (`research.queue._message`)."""
        self.git("merge", "-q", "--no-ff", "-m", f"Queue: #{pr} {title}\n\nPR: #{pr}\nHead: {head}\nOwner: lane\n", head)
        return self.git("rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Repo:
    r = Repo(tmp_path / "repo")
    r.commit("base", {"integrations/vllm/verity_vllm/a.py": "A = 1\n", "integrations/vllm/verity_vllm/b.py": "B = 1\n",
                      P01: _allowlist(_entry("verity_vllm/a.py", "A"), _entry("verity_vllm/b.py", "B")),
                      "tools/verity/check/move_check.py": MOVE_CHECK})
    monkeypatch.chdir(r.root)
    return r


def _verdicts(doc: dict) -> list[tuple[str, str]]:
    return [(c["name"], c["verdict"]) for c in doc["changes"]]


def test_a_branch_is_judged_against_its_merge_base_and_must_retire_an_exception(repo):
    base = repo.git("rev-parse", "main")
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("one", {"integrations/vllm/verity_vllm/a.py": "A = 2\n"})
    head = repo.commit("two", {"integrations/vllm/verity_vllm/c.py": "C = 1\n"})
    doc = G.gate("main", "pr")
    [c] = doc["changes"]
    assert (c["base"], c["commit"], c["verdict"], c["before"], c["after"]) == (base, head, "failed", 2, 2)
    assert c["source"] == ["integrations/vllm/verity_vllm/a.py", "integrations/vllm/verity_vllm/c.py"]
    assert "retire at least 1" in c["why"]
    assert c["fix"][0] == f"{P01}: verity_vllm/a.py k A d"          # the entry in a file it already changes comes first
    assert c["fix"][-1] == f"each ratchet's entries, fewest first: {P01} 2"
    assert doc["failures"] == [{"commit": head, "pr": None, "why": c["why"], "fix": c["fix"]}]
    repo.commit("three", {P01: _allowlist(_entry("verity_vllm/b.py", "B"))})
    assert _verdicts(G.gate("main", "pr")) == [(f"{base[:12]}..{repo.git('rev-parse', 'HEAD')[:12]}", "passed")]


def test_main_moving_on_keeps_the_branch_judged_against_its_merge_base(repo):
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("pr", {"integrations/vllm/verity_vllm/a.py": "A = 2\n", P01: _allowlist(_entry("verity_vllm/a.py", "A"))})
    repo.git("checkout", "-q", "main")
    repo.commit("main moves", {"integrations/vllm/verity_vllm/b.py": "B = 2\n", "other.txt": "x\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"], c["source"]) == ("passed", 2, 1, ["integrations/vllm/verity_vllm/a.py"])


def test_each_queue_merge_is_one_pr_against_the_one_below(repo):
    base = repo.git("rev-parse", "main")
    repo.git("checkout", "-q", "-b", "one")
    one = repo.commit("retires A", {"integrations/vllm/verity_vllm/a.py": "A = 2\n", P01: _allowlist(_entry("verity_vllm/b.py", "B"))})
    repo.git("checkout", "-q", "-b", "two", base)
    two = repo.commit("adds code", {"integrations/vllm/verity_vllm/c.py": "C = 1\n"})
    repo.git("checkout", "-q", "-b", "three", base)
    three = repo.commit("tests only", {"integrations/vllm/tests/test_c.py": "def test(): pass\n"})
    repo.git("checkout", "-q", "-b", "next", base)
    m1, m2, m3 = repo.queue(1, "retire", one), repo.queue(2, "feature", two), repo.queue(3, "tests", three)
    doc = G.gate("main", "next")
    assert _verdicts(doc) == [("#1", "passed"), ("#2", "failed"), ("#3", "untouched")]
    assert [(c["base"], c["before"], c["after"]) for c in doc["changes"][:2]] == [(base, 2, 1), (m1, 1, 1)]
    assert [(f["commit"], f["pr"]) for f in doc["failures"]] == [(m2, 2)]
    assert doc["changes"][2]["commit"] == m3


def test_a_new_ratchet_is_its_own_baseline_and_a_moved_one_is_still_compared(repo):
    repo.git("checkout", "-q", "-b", "pr")
    moved = "integrations/vllm/tests/lint/moved/allowlists/p01_core.json"
    repo.commit("adds p02, moves p01 and retires from it", {
        "integrations/vllm/verity_vllm/a.py": "A = 2\n", P01: None, moved: _allowlist(_entry("verity_vllm/b.py", "B")),
        "integrations/vllm/tests/p02_allowlist.json": _allowlist(*(_entry("verity_vllm/a.py", str(i)) for i in range(5)))})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"]) == ("passed", 2, 1)


def test_an_allowlist_left_unreadable_fails(repo):
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("breaks it", {"integrations/vllm/verity_vllm/a.py": "A = 2\n", P01: "{"})
    [c] = G.gate("main", "pr")["changes"]
    assert c["verdict"] == "failed" and f"leaves {P01} unreadable" in c["why"]


def test_docs_and_tests_under_the_package_are_not_source():
    assert G.is_source("integrations/vllm/verity_vllm/engine/x.py")
    assert G.is_source("integrations/vllm/verity_vllm/kernels/k.cu")
    assert not any(map(G.is_source, ["integrations/vllm/verity_vllm/README.md", "integrations/vllm/verity_vllm/x/test_y.py",
                                      "integrations/vllm/verity_vllm/x/tests/y.py", "integrations/vllm/verity_vllm/conftest.py",
                                      "integrations/vllm/tests/test_x.py", "integrations/vllm/tools/scorecard.py"]))
    assert G.is_ratchet("integrations/vllm/tests/lint/allowlists/p07.json")
    assert G.is_ratchet("integrations/vllm/tests/by_name_allowlist.json")
    assert not G.is_ratchet("integrations/vllm/tests/fixtures/x.json")


def test_a_revert_and_a_pure_move_are_exempt(repo):
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit('Revert "Queue: #9 retire B"', {"integrations/vllm/verity_vllm/b.py": "B = 0\n",
                                                P01: _allowlist(*(_entry("verity_vllm/b.py", str(i)) for i in range(3)))})
    assert _verdicts(G.gate("main", "pr"))[0][1] == "exempt"
    repo.git("checkout", "-q", "-b", "move", "main")
    repo.git("mv", "integrations/vllm/verity_vllm/a.py", "integrations/vllm/verity_vllm/moved_a.py")
    repo.commit("move a", {})
    [c] = G.gate("main", "move")["changes"]
    assert (c["verdict"], c["why"]) == ("exempt", f"a pure move ({G.MOVE_CHECK} certifies it)")
    repo.git("checkout", "-q", "-b", "next", "main")
    repo.queue(4, 'Revert "x"', "pr")
    assert _verdicts(G.gate("main", "next")) == [("#4", "exempt")]


def test_the_command_line(repo):
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("feature", {"integrations/vllm/verity_vllm/c.py": "C = 1\n"})
    run = lambda *a: subprocess.run([sys.executable, str(SCRIPT), *a], cwd=repo.root, capture_output=True, text=True)  # noqa: E731
    r = run("main", "pr")
    assert r.returncode == 1 and "failed: changes 1 source file(s)" in r.stdout and f"{P01} 2" in r.stdout, r.stdout
    r = run("main", "pr", "--json")
    assert r.returncode == 1 and json.loads(r.stdout)["kind"] == G.KIND
    assert run("main", "main").returncode == 0
    r = run("main", "no-such-ref")
    assert r.returncode == 2 and "ratchet gate: git rev-parse" in r.stderr
