"""`tools/ratchet_gate.py` on toy repositories: a branch's own commits judged together against its merge base, each queue merge
on its own, no ratchet growing (by entries, per ratchet, or by an entry's count), a new, moved or renamed ratchet, the changes
it doesn't judge, the base's lint code run on the change's tree (a loosened, deleted or bypassed lint fails, naming what it
still finds; a tightened one passes once its findings are fixed), the exemptions (a revert, a pure move the base's
move_check certifies), and its command line."""
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
P02 = "integrations/vllm/tests/lint/allowlists/p02_more.json"
LINT = "integrations/vllm/tests/lint/test_p01_core.py"
#: a stand-in for move_check: certified when every change is a 100% rename
MOVE_CHECK = """\
import subprocess, sys
out = subprocess.run(["git", "diff", "--name-status", "-M100%", sys.argv[1], sys.argv[2]], capture_output=True, text=True).stdout
sys.exit(0 if out and all(line.startswith("R100") for line in out.splitlines()) else 1)
"""
#: a toy lint in the real ones' shape: a library file matching WORD is a violation, keyed by its file
TOY_LINT = '''\
import json
import re
from pathlib import Path

import pytest

NAME = "p01_core"
WORD = re.compile(r"BAD")
HERE = Path(__file__).resolve().parent


def found():
    return sorted(f"verity_vllm/{p.name}" for p in (HERE.parents[1] / "verity_vllm").glob("*.py") if WORD.search(p.read_text()))


@pytest.mark.slow
def test_no_new_violations():
    allowed = {e["file"] for e in json.loads((HERE / "allowlists" / f"{NAME}.json").read_text())["entries"]}
    new = [f for f in found() if f not in allowed]
    assert not new, "P1: not in its allowlist:\\n" + "\\n".join(f"  {f} [k]" for f in new)


def test_allowlist_has_no_stale_entries():
    assert False, "the gate runs no stale-entry test"
'''
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def _entry(file: str, symbol: str, count: int = 1) -> dict:
    return {"file": file, "kind": "k", "symbol": symbol, "detail": "d", **({"count": count} if count != 1 else {})}


def _allowlist(*entries: dict) -> str:
    """one entry a line, as the lints' allowlists are kept"""
    return '{"rule": "p01", "entries": [\n' + ",\n".join(json.dumps(e) for e in entries) + "\n]}\n"


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


@pytest.fixture
def linted(repo) -> Repo:
    """``repo`` with the toy lint, whose two violations (a.py, b.py) P01 allows."""
    repo.commit("lint", {LINT: TOY_LINT, "integrations/vllm/verity_vllm/a.py": "A = 'BAD'\n",
                         "integrations/vllm/verity_vllm/b.py": "B = 'BAD'\n"})
    repo.git("checkout", "-q", "-b", "pr")
    return repo


def _verdicts(doc: dict) -> list[tuple[str, str]]:
    return [(c["name"], c["verdict"]) for c in doc["changes"]]


def test_a_branch_is_judged_against_its_merge_base_and_may_not_grow_a_ratchet(repo):
    base = repo.git("rev-parse", "main")
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("one", {"integrations/vllm/verity_vllm/a.py": "A = 2\n"})
    assert _verdicts(G.gate("main", "pr")) == [(f"{base[:12]}..{repo.git('rev-parse', 'HEAD')[:12]}", "untouched")]
    head = repo.commit("two", {"integrations/vllm/verity_vllm/c.py": "C = 1\n",
                               P01: _allowlist(_entry("verity_vllm/a.py", "A"), _entry("verity_vllm/b.py", "B"),
                                               _entry("verity_vllm/c.py", "C"))})
    doc = G.gate("main", "pr")
    [c] = doc["changes"]
    assert (c["base"], c["commit"], c["verdict"], c["before"], c["after"], c["touched"]) == (base, head, "failed", 2, 3, [P01])
    assert c["why"].startswith(f"grows {P01} 2 -> 3 entries: no change adds a ratchet exception")
    assert c["fix"] == [f'{P01}: new {G.key(_entry("verity_vllm/c.py", "C"))}']
    assert doc["failures"] == [{"commit": head, "pr": None, "why": c["why"], "fix": c["fix"]}]
    repo.commit("three: the entry moves with the code", {"integrations/vllm/verity_vllm/b.py": None,
                                                        P01: _allowlist(_entry("verity_vllm/a.py", "A"),
                                                                        _entry("verity_vllm/c.py", "C"))})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"]) == ("passed", 2, 2)


def test_no_recorded_count_grows_and_one_ratchet_does_not_pay_for_another(repo):
    repo.commit("p02, and a count", {P02: _allowlist(_entry("verity_vllm/a.py", "X")),
                                     P01: _allowlist(_entry("verity_vllm/a.py", "A", 2), _entry("verity_vllm/b.py", "B"))})
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("raises a count, retires another", {P01: _allowlist(_entry("verity_vllm/a.py", "A", 3))})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"]) == ("failed", 3, 2)
    assert c["why"].startswith(f'grows {P01} count 2 -> 3 for {G.key(_entry("verity_vllm/a.py", "A"))}')
    repo.git("checkout", "-q", "-b", "trade", "main")
    repo.commit("retires from p01, adds to p02", {P01: _allowlist(_entry("verity_vllm/b.py", "B")),
                                                 P02: _allowlist(_entry("verity_vllm/a.py", "X"), _entry("verity_vllm/b.py", "Y"))})
    [c] = G.gate("main", "trade")["changes"]
    assert (c["verdict"], c["before"], c["after"]) == ("failed", 3, 3) and f"{P02} 1 -> 2 entries" in c["why"]
    repo.git("checkout", "-q", "-b", "lower", "main")
    repo.commit("lowers a count", {P01: _allowlist(_entry("verity_vllm/a.py", "A"), _entry("verity_vllm/b.py", "B"))})
    assert _verdicts(G.gate("main", "lower"))[0][1] == "passed"


def test_main_moving_on_keeps_the_branch_judged_against_its_merge_base(repo):
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("pr", {"integrations/vllm/verity_vllm/a.py": "A = 2\n", P01: _allowlist(_entry("verity_vllm/a.py", "A"))})
    repo.git("checkout", "-q", "main")
    repo.commit("main moves", {"integrations/vllm/verity_vllm/b.py": "B = 2\n", "other.txt": "x\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"], c["touched"]) == ("passed", 2, 1, [P01])


def test_each_queue_merge_is_one_pr_against_the_one_below(repo):
    base = repo.commit("p02", {P02: _allowlist(_entry("verity_vllm/a.py", "X"))})
    repo.git("checkout", "-q", "-b", "one")
    one = repo.commit("retires A", {"integrations/vllm/verity_vllm/a.py": "A = 2\n", P01: _allowlist(_entry("verity_vllm/b.py", "B"))})
    repo.git("checkout", "-q", "-b", "two", base)
    two = repo.commit("adds code and its exception", {"integrations/vllm/verity_vllm/c.py": "C = 1\n",
                                                      P02: _allowlist(_entry("verity_vllm/a.py", "X"), _entry("verity_vllm/c.py", "C"))})
    repo.git("checkout", "-q", "-b", "three", base)
    three = repo.commit("code and tests only", {"integrations/vllm/verity_vllm/d.py": "D = 1\n",
                                                "integrations/vllm/tests/test_c.py": "def test(): pass\n"})
    repo.git("checkout", "-q", "-b", "next", base)
    m1 = repo.queue(1, "retire", one)
    m2 = repo.queue(2, "feature", two)
    m3 = repo.queue(3, "tests", three)
    doc = G.gate("main", "next")
    assert _verdicts(doc) == [("#1", "passed"), ("#2", "failed"), ("#3", "untouched")]
    assert [(c["base"], c["before"], c["after"]) for c in doc["changes"][:2]] == [(base, 3, 2), (m1, 2, 3)]
    assert [(f["commit"], f["pr"]) for f in doc["failures"]] == [(m2, 2)]
    assert doc["changes"][2]["commit"] == m3


def test_a_new_ratchet_is_its_own_baseline_and_a_moved_or_renamed_one_is_still_compared(repo):
    repo.git("checkout", "-q", "-b", "pr")
    moved = "integrations/vllm/tests/lint/moved/allowlists/p01_core.json"
    repo.commit("adds p02, moves p01 and retires from it", {
        "integrations/vllm/verity_vllm/a.py": "A = 2\n", P01: None, moved: _allowlist(_entry("verity_vllm/b.py", "B")),
        "integrations/vllm/tests/p02_allowlist.json": _allowlist(*(_entry("verity_vllm/a.py", str(i)) for i in range(5)))})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"]) == ("passed", 2, 1)
    repo.git("checkout", "-q", "-b", "slug", "main")
    by_slug = "integrations/vllm/tests/lint/allowlists/core.json"
    repo.commit("names p01 by its slug and grows it", {P01: None, by_slug: _allowlist(
        _entry("verity_vllm/a.py", "A"), _entry("verity_vllm/b.py", "B"), _entry("verity_vllm/c.py", "C"))})
    [c] = G.gate("main", "slug")["changes"]
    assert (c["verdict"], c["before"], c["after"]) == ("failed", 2, 3) and c["why"].startswith(f"grows {by_slug} 2 -> 3")


def test_an_allowlist_left_unreadable_fails(repo):
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("breaks it", {"integrations/vllm/verity_vllm/a.py": "A = 2\n", P01: "{"})
    [c] = G.gate("main", "pr")["changes"]
    assert c["verdict"] == "failed" and f"leaves {P01} unreadable" in c["why"]


def test_what_is_a_ratchet_lint_code_and_a_slug():
    assert G.is_ratchet("integrations/vllm/tests/lint/allowlists/p07.json")
    assert G.is_ratchet("integrations/vllm/tests/by_name_allowlist.json")
    assert not G.is_ratchet("integrations/vllm/tests/fixtures/x.json")
    assert G.is_lint_code(LINT) and G.is_lint_code("integrations/vllm/tests/lint/_ratchet.py")
    assert not any(map(G.is_lint_code, [P01, "integrations/vllm/tests/lint/README.md", "integrations/vllm/tests/test_x.py"]))
    assert {G.slug(P01), G.slug(LINT), G.slug("x/allowlists/core.json"), G.slug("x/test_core.py")} == {"core"}
    assert G.lint_selection([LINT, P02, "integrations/vllm/tests/by_name_allowlist.json"]) == ["core", "more"]
    assert G.lint_selection([LINT, "integrations/vllm/tests/lint/_ratchet.py"]) is None


def test_a_revert_and_a_pure_move_are_exempt(linted):
    linted.commit('Revert "Queue: #9 retire B"', {"integrations/vllm/verity_vllm/c.py": "C = 'BAD'\n",
                                                  P01: _allowlist(*(_entry(f"verity_vllm/{f}.py", f) for f in "abc"))})
    assert _verdicts(G.gate("main", "pr"))[0][1] == "exempt"
    linted.git("checkout", "-q", "-b", "move", "main")
    linted.git("mv", P01, "integrations/vllm/tests/lint/allowlists/p01_other.json")
    linted.commit("move p01's allowlist off its lint's name", {})
    [c] = G.gate("main", "move")["changes"]
    assert (c["verdict"], c["why"], c["lints"]) == ("exempt", f"a pure move ({G.MOVE_CHECK} certifies it)", ["p01_core"])
    linted.git("checkout", "-q", "-b", "next", "main")
    linted.queue(4, 'Revert "x"', "pr")
    assert _verdicts(G.gate("main", "next")) == [("#4", "exempt")]


def test_a_retired_entry_passes_once_the_base_lint_agrees_it_is_gone(linted):
    linted.commit("fixes a.py and retires its entry", {"integrations/vllm/verity_vllm/a.py": "A = 'ok'\n",
                                                       P01: _allowlist(_entry("verity_vllm/b.py", "B"))})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"], c["lints"]) == ("passed", 2, 1, ["p01_core"])
    linted.commit("drops b.py's entry with b.py unchanged", {P01: _allowlist()})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", ["p01_core: verity_vllm/b.py [k]"])
    assert "the base's lint code" in c["why"] and "finds 1 violation(s)" in c["why"]


@pytest.mark.parametrize("loosen", [
    {LINT: TOY_LINT.replace('r"BAD"', 'r"WORSE"')},
    {LINT: None},
    {LINT: TOY_LINT.replace("if WORD.search(p.read_text())", "if WORD.search(p.read_text()) and False"),
     "integrations/vllm/tests/__init__.py": "raise SystemExit(0)\n"},
], ids=["threshold", "deleted", "bypassed"])
def test_loosening_a_lint_to_drop_its_entries_fails_naming_what_the_base_lint_finds(linted, loosen):
    head = linted.commit("loosens the lint and drops its entries", {**loosen, P01: _allowlist()})
    doc = G.gate("main", "pr")
    [c] = doc["changes"]
    assert (c["verdict"], c["before"], c["after"], c["lints"]) == ("failed", 2, 0, ["p01_core"])
    assert c["fix"] == ["p01_core: verity_vllm/a.py [k]", "p01_core: verity_vllm/b.py [k]"]
    assert doc["failures"] == [{"commit": head, "pr": None, "why": c["why"], "fix": c["fix"]}]


def test_deleting_a_lint_and_its_allowlist_passes_once_nothing_is_left(linted):
    linted.commit("fixes both", {"integrations/vllm/verity_vllm/a.py": "A = 1\n", "integrations/vllm/verity_vllm/b.py": "B = 1\n",
                                 LINT: None, P01: None})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"], c["lints"]) == ("passed", 2, 0, ["p01_core"])


def test_tightening_a_lint_passes_once_its_findings_are_fixed_and_fails_if_they_grow_the_allowlist(linted):
    linted.git("checkout", "-q", "main")
    linted.commit("c.py", {"integrations/vllm/verity_vllm/c.py": "C = 'BAR'\n"})
    linted.git("checkout", "-q", "-b", "tighten")
    tighter = {LINT: TOY_LINT.replace('r"BAD"', 'r"BA[DR]"')}
    linted.commit("tightens the lint and fixes c.py", {**tighter, "integrations/vllm/verity_vllm/c.py": "C = 'ok'\n"})
    [c] = G.gate("main", "tighten")["changes"]
    assert (c["verdict"], c["before"], c["after"], c["lints"]) == ("passed", 2, 2, ["p01_core"])
    linted.git("checkout", "-q", "-b", "tighten-and-allow", "main")
    linted.commit("tightens the lint and allows c.py", {**tighter, P01: _allowlist(
        *(_entry(f"verity_vllm/{f}.py", f) for f in "abc"))})
    [c] = G.gate("main", "tighten-and-allow")["changes"]
    assert (c["verdict"], c["before"], c["after"]) == ("failed", 2, 3) and c["why"].startswith(f"grows {P01} 2 -> 3")


def test_shared_lint_code_runs_every_lint_and_an_error_is_named(linted):
    linted.commit("a helper", {"integrations/vllm/tests/lint/_helper.py": "X = 1\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"]) == ("passed", ["p01_core"])
    assert G.violations([{"lint": "p10_size", "test": "test_no_new_x", "error": True,
                          "message": "Traceback:\n  File x\nFileNotFoundError: gone.py\n"}]) == [
        "p10_size: test_no_new_x raised FileNotFoundError: gone.py"]


def test_the_command_line(repo):
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("feature", {"integrations/vllm/verity_vllm/c.py": "C = 1\n",
                            P01: _allowlist(_entry("verity_vllm/a.py", "A"), _entry("verity_vllm/b.py", "B"),
                                            _entry("verity_vllm/c.py", "C"))})
    run = lambda *a: subprocess.run([sys.executable, str(SCRIPT), *a], cwd=repo.root, capture_output=True, text=True)  # noqa: E731
    r = run("main", "pr")
    assert r.returncode == 1 and f"failed: grows {P01} 2 -> 3 entries" in r.stdout and "verity_vllm/c.py" in r.stdout, r.stdout
    r = run("main", "pr", "--json")
    assert r.returncode == 1 and json.loads(r.stdout)["kind"] == G.KIND
    assert run("main", "main").returncode == 0
    r = run("main", "no-such-ref")
    assert r.returncode == 2 and "ratchet gate: git rev-parse" in r.stderr
