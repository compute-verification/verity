"""`tools/check/added_lines.py`: what a change's added lines bring of retired terms, issue ids and run ids, counted per file by
the repository lints as the base has them, on a toy repository holding this tree's two lints and a glossary of its own."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "check" / "added_lines.py"
LINTS = ("tests/lint/test_repo_lane_text.py", "tests/lint/test_repo_retired_terms.py")
#: one retired word, which no real row names
GLOSSARY = '# Glossary\n\n## Older names\n\n| Old name, where it stays | Term |\n|---|---|\n| "zorblax" | widget |\n'
#: an issue id and a run id, built so that no literal one stands in this file
ISSUE = "-".join(("CI", "4850"))
RUN = "r" + "-".join(("20261025", "031300", "0fcc"))
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}



def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                      # dataclasses look their module up while the class is built
    spec.loader.exec_module(mod)
    return mod


D = _module("added_lines_declaration", ROOT / "tools" / "check" / "declaration.py")
CHECK = _module("added_lines_check", ROOT / "tools" / "check" / "check.py")


class Repo:
    def __init__(self, path: Path):
        self.path = path
        path.mkdir()
        self.git("init", "-q", "-b", "main")
        self.base = self.commit({**{p: (ROOT / p).read_text() for p in LINTS}, "GLOSSARY.md": GLOSSARY,
                                 "a.py": "X = 1  # a widget\n", "old.py": f"# the zorblax of {ISSUE}\nY = 2\n"})

    def git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.path, env=ENV, check=True, capture_output=True, text=True).stdout.strip()

    def commit(self, files: dict[str, str | None], moves: dict[str, str] | None = None) -> str:
        for old, new in (moves or {}).items():
            (self.path / new).parent.mkdir(parents=True, exist_ok=True)
            self.git("mv", old, new)
        for path, text in files.items():
            if text is None:
                self.git("rm", "-q", path)
                continue
            (self.path / path).parent.mkdir(parents=True, exist_ok=True)
            (self.path / path).write_text(text)
            self.git("add", path)
        self.git("commit", "-q", "-m", "change")
        return self.git("rev-parse", "HEAD")

    def change(self, files: dict[str, str | None], moves: dict[str, str] | None = None) -> str:
        """A branch off the base with ``files`` (None: deleted) and ``moves`` ({old: new}): its head."""
        self.git("checkout", "-q", "--detach", self.base)
        return self.commit(files, moves)

    def judge(self, head: str, *extra: str) -> tuple[int, dict]:
        r = subprocess.run([sys.executable, str(SCRIPT), self.base, head, "--json", *extra], cwd=self.path, env=ENV,
                           capture_output=True, text=True)
        assert r.returncode in (0, 1, 2), r.stderr
        return r.returncode, json.loads(r.stdout)


def named(doc: dict) -> list[tuple[str, str, int]]:
    return [(f["path"], f["kind"], s["line"]) for f in doc["failures"] for s in f["sites"]]


def test_an_added_retired_term_in_a_touched_file_is_named_with_its_line_and_the_lint_s_fix(tmp_path):
    repo = Repo(tmp_path / "r")
    rc, doc = repo.judge(repo.change({"a.py": "X = 1  # a widget\nZ = 'the zorblax'\n"}))
    assert rc == 1 and doc["verdict"] == "failed" and named(doc) == [("a.py", "retired term", 2)], doc
    [f] = doc["failures"]
    assert f["why"] == ("a.py gains 1 retired term(s), 0 -> 1: a.py:2 'zorblax' is retired; say widget (GLOSSARY.md) in "
                        "`Z = 'the zorblax'`"), f
    assert f["fix"][0].startswith("Say the term the row gives (GLOSSARY.md's older names)"), f
    text = subprocess.run([sys.executable, str(SCRIPT), repo.base, "HEAD"], cwd=repo.path, env=ENV, capture_output=True,
                          text=True)
    assert text.returncode == 1 and "    a.py:2 'zorblax' is retired; say widget (GLOSSARY.md)  | Z = 'the zorblax'" in text.stdout


def test_a_pure_rename_and_an_edited_line_keeping_its_old_word_pass(tmp_path):
    repo = Repo(tmp_path / "r")
    rc, doc = repo.judge(repo.change({}, {"old.py": "pkg/new.py"}))
    assert rc == 0 and doc["verdict"] == "passed" and doc["failures"] == [], doc
    rc, doc = repo.judge(repo.change({"old.py": f"# the zorblax of {ISSUE}, said again\nY = 3\n"}))
    assert rc == 0 and doc["failures"] == [], doc


def test_a_rename_that_adds_a_use_is_named_for_the_added_one_alone(tmp_path):
    repo = Repo(tmp_path / "r")
    head = repo.change({"pkg/new.py": f"# the zorblax of {ISSUE}\nY = 2\nW = 'zorblax'\n"}, {"old.py": "pkg/new.py"})
    assert repo.git("diff", "-M", "--name-status", repo.base, head).startswith("R")
    rc, doc = repo.judge(head)
    assert rc == 1 and named(doc) == [("pkg/new.py", "retired term", 3)], doc
    assert doc["failures"][0]["before"] == 1 and doc["failures"][0]["after"] == 2


def test_an_issue_id_and_a_run_id_added_in_a_comment_are_named_and_in_code_strings_or_data_they_are_not(tmp_path):
    """The ids are lane text, which the lane-text lint reads in prose alone: a comment, a docstring, Markdown; a string in code
    and a JSON file are data, as that lint reads them."""
    repo = Repo(tmp_path / "r")
    head = repo.change({"a.py": f"X = 1  # a widget\n# see {ISSUE}, run {RUN}\nKEY = '{ISSUE} {RUN}'\n",
                        "rows.json": json.dumps({"issue": ISSUE, "run": RUN}), "notes.md": f"Ran {RUN}.\n"})
    rc, doc = repo.judge(head)
    assert rc == 1 and sorted(named(doc)) == [("a.py", "issue id", 2), ("a.py", "run id", 2), ("notes.md", "run id", 1)], doc
    lane = [f for f in doc["failures"] if f["kind"] != "retired term"]
    assert all(f["fix"][0].startswith("Say what is true now") for f in lane), lane


def test_archive_and_what_the_lints_skip_are_not_read(tmp_path):
    repo = Repo(tmp_path / "r")
    head = repo.change({"archive/x.py": f"# zorblax {ISSUE} {RUN}\n", "pkg/tests/fixtures/y.py": f"# zorblax {ISSUE}\n",
                        "uv.lock": f"zorblax {ISSUE}\n"})
    assert repo.judge(head)[0] == 0


def test_a_move_between_files_is_no_addition_and_a_copy_is(tmp_path):
    """A site the change takes out of one file and puts in another, its line edited on the way, leaves the tree holding no more
    of it; a copy of it does."""
    repo = Repo(tmp_path / "r")
    assert repo.judge(repo.change({"old.py": "Y = 2\n", "b.py": f"# zorblax of {ISSUE}, moved\n"}))[0] == 0
    rc, doc = repo.judge(repo.change({"b.py": f"# the zorblax of {ISSUE}\nY = 2\n"}))
    assert rc == 1 and sorted(named(doc)) == [("b.py", "issue id", 1), ("b.py", "retired term", 1)], doc


def test_a_change_to_the_lints_or_the_glossary_can_t_relax_its_own_judge(tmp_path):
    """The base's copies judge: a head that empties the lane-text lint's `sites` and drops the glossary's row is judged by the
    lints and the row as the base has them."""
    repo = Repo(tmp_path / "r")
    lane = (repo.path / LINTS[0]).read_text() + "\n\ndef sites(text, path):\n    return []\n"
    head = repo.change({LINTS[0]: lane, "GLOSSARY.md": GLOSSARY.split('| "zorblax"')[0],
                        "a.py": f"X = 1  # a widget, {ISSUE}\nZ = 'zorblax'\n"})
    rc, doc = repo.judge(head)
    assert rc == 1 and sorted(named(doc)) == [("a.py", "issue id", 1), ("a.py", "retired term", 2)], doc


def test_a_base_without_the_lints_can_t_judge(tmp_path):
    repo = Repo(tmp_path / "r")
    repo.base = repo.change({"GLOSSARY.md": None, **dict.fromkeys(LINTS)})
    rc, doc = repo.judge(repo.commit({"a.py": f"# {ISSUE}\n"}))
    assert rc == 2 and doc["verdict"] == "cannot-judge" and "has no GLOSSARY.md" in doc["why"], doc


def test_it_is_off_the_merge_gate_and_a_warning_wherever_it_runs(tmp_path, monkeypatch):
    """A red holds nothing: policy.toml takes the step off the merge gate, and drive.toml runs the script at a ready mark as a
    warn gate, whose failure the mark prints and lands all the same; and the step runs the script as the
    run's base has it, skipping a base without one."""
    job = next(j for j in D.load() if j.name == "added-lines")
    assert not job.gate and job.name not in {j.name for j in D.gated(D.load())}
    drive = tomllib.loads((ROOT / "tools" / "check" / "drive.toml").read_text())
    assert "tools/check/added_lines.py" in drive["warn_gates"]
    assert "tools/check/added_lines.py" not in drive["landing_gates"]

    check = CHECK
    repo = Repo(tmp_path / "r")
    head = repo.change({"a.py": f"X = 1  # {ISSUE}\n"})
    monkeypatch.setattr(check, "ROOT", repo.path)
    out = tmp_path / "run"
    (out / "inputs").mkdir(parents=True)
    (out / "inputs" / check.BASE_FILE).write_text(f"{repo.base}\n")
    assert check.added_lines(out) == f"skipped: its base {repo.base} has no {check.ADDED_LINES}"
    repo.git("checkout", "-q", "--detach", repo.base)
    with_script = repo.commit({check.ADDED_LINES: SCRIPT.read_text()})
    (out / "inputs" / check.BASE_FILE).write_text(f"{with_script}\n")
    repo.git("checkout", "-q", "--detach", with_script)
    head = repo.commit({"a.py": f"X = 1  # {ISSUE}\n", check.ADDED_LINES: "import sys\nsys.exit(0)\n"})
    argv = check.added_lines(out)
    assert not isinstance(argv, str) and argv[4] == with_script, argv
    r = subprocess.run(argv, cwd=repo.path, env=ENV, capture_output=True, text=True)
    assert r.returncode == 1 and f"a.py:1 linear: {ISSUE}" in r.stdout, r.stdout + r.stderr
    assert repo.git("rev-parse", "HEAD") == head
    (out / "inputs" / check.BASE_FILE).write_text("0" * 40 + "\n")
    assert check.added_lines(out) == f"skipped: no base: {'0' * 40} names no commit here"
