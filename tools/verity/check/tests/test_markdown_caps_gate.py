"""`gates/markdown_caps.py`, drive's landing gate for the Markdown caps: it applies `tests/test_repository.py`'s caps and fixes,
read at BASE, to each `.md` file that changed, at its size in TOP."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
GATE = ROOT / "tools/verity/check/gates/markdown_caps.py"
CAPS = ROOT / "tests/test_repository.py"


def _repository():
    spec = importlib.util.spec_from_file_location("repository_invariants", CAPS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


def _commit(repo: Path, files: dict[str, str | None]) -> str:
    for rel, text in files.items():
        if text is None:
            (repo / rel).unlink()
            continue
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "change")
    return _git(repo, "rev-parse", "HEAD")


def _gate(repo: Path, base: str, top: str) -> tuple[int, dict]:
    r = subprocess.run([sys.executable, str(GATE), base, top, "--json"], cwd=repo, capture_output=True, text=True)
    return r.returncode, json.loads(r.stdout) if r.returncode in (0, 1) else {"stderr": r.stderr}


def test_the_gate_applies_the_caps_and_fixes_of_test_repository_to_each_changed_markdown_file(tmp_path):
    """Its caps are test_repository's own (`markdown_cap`), on every kind of file that has one, either side of the cap."""
    rep = _repository()
    skill, split = ".agents/skills/x/SKILL.md", next(iter(rep.SPLIT_DOCUMENT_DIRS)) + "entry.md"
    sizes = {"README.md": rep.MD_SIZE_CAP, "a/README.md": rep.MD_SIZE_CAP + 1, "a/PROTOCOL.md": rep.MD_SIZE_CAP + 1,
             "b/PROTOCOL.md": rep.PROTOCOL_SIZE_CAP + 2, skill: rep.SKILL_SIZE_CAP + 3, split: rep.MD_SIZE_CAP + 4,
             "big.txt": rep.MD_SIZE_CAP * 2, "kept.md": rep.MD_SIZE_CAP + 5, "gone.md": rep.MD_SIZE_CAP + 6}
    _git(tmp_path, "init", "-q", "-b", "main")
    base = _commit(tmp_path, {"tests/test_repository.py": CAPS.read_text(), "kept.md": "k" * sizes["kept.md"],
                              "gone.md": "g" * sizes["gone.md"]})
    top = _commit(tmp_path, {**{p: "x" * n for p, n in sizes.items() if p not in ("kept.md", "gone.md")}, "gone.md": None})
    rc, doc = _gate(tmp_path, base, top)
    changed = [p for p in sizes if p.endswith(".md") and p not in ("kept.md", "gone.md")]
    assert sorted((f["path"], f["size"], f["cap"]) for f in doc["files"]) == sorted(
        (p, sizes[p], rep.markdown_cap(p)) for p in changed)
    over = [p for p in changed if rep.markdown_over_cap(p, sizes[p])]
    assert rc == 1 and sorted(f["path"] for f in doc["failures"]) == sorted(over) == sorted(
        ["a/README.md", "b/PROTOCOL.md", skill, split])
    for f in doc["failures"]:
        p, cap = f["path"], rep.markdown_cap(f["path"])
        assert f["why"] == f"{p} is {sizes[p]} bytes at {top[:12]}, over its {cap}-byte cap (tests/test_repository.py)"
        assert f["fix"] == [f"trim {p} to {cap} bytes or fewer ({sizes[p] - cap} over): "
                            + rep.markdown_over_cap(p, sizes[p]).split("; ", 1)[1]]
    rc, doc = _gate(tmp_path, base, base)
    assert rc == 0 and doc["files"] == doc["failures"] == []


def test_the_caps_are_read_at_base_so_a_change_cannot_raise_its_own(tmp_path):
    rep = _repository()
    _git(tmp_path, "init", "-q", "-b", "main")
    base = _commit(tmp_path, {"tests/test_repository.py": CAPS.read_text()})
    raised = CAPS.read_text().replace("MD_SIZE_CAP = 48 * 1024", "MD_SIZE_CAP = 96 * 1024")
    assert raised != CAPS.read_text()
    top = _commit(tmp_path, {"tests/test_repository.py": raised, "AGENTS.md": "a" * (rep.MD_SIZE_CAP + 206)})
    rc, doc = _gate(tmp_path, base, top)
    assert rc == 1 and [(f["path"], f["cap"]) for f in doc["failures"]] == [("AGENTS.md", rep.MD_SIZE_CAP)]
    assert _gate(tmp_path, top, top)[0] == 0


def test_a_base_without_the_caps_is_unable_to_judge(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    base = _commit(tmp_path, {"a.txt": "a\n"})
    rc, doc = _gate(tmp_path, base, _commit(tmp_path, {"a.md": "a\n"}))
    assert rc == 2 and "tests/test_repository.py" in doc["stderr"]
