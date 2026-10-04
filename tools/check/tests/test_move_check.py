"""`tools/check/move_check.py`: which commit ranges it certifies as pure moves, on tiny git repositories."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GIT = shutil.which("git")
spec = importlib.util.spec_from_file_location("check_move_check", ROOT / "tools" / "check" / "move_check.py")
MC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MC)

ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}      # not this machine's git config
PYPROJECT = 'name = "pous"\n[tool.verity.tests]\ninputs = ["protocols/pous/data", "census/hardware.json"]\n'
LAKEFILE = '[[require]]\nname = "verity"\npath = "../../../packages/verity/lean"\n'


def _git(repo: Path, *args: str) -> str:
    return subprocess.run([GIT, "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=repo, check=True,
                          capture_output=True, text=True, env=ENV).stdout.strip()


def _write(repo: Path, files: dict[str, str]) -> None:
    for p, text in files.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(text)


def _repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo, files)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _move(repo: Path, old: str, new: str) -> None:
    (repo / new).parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "mv", old, new)


def _check(repo: Path, base: str, extra: dict[str, str] | None = None) -> dict:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "move")
    return MC.certify(base, _git(repo, "rev-parse", "HEAD"), extra, cwd=str(repo))


BASE_FILES = {
    "protocols/pous/src/verity_pous/__init__.py": "",
    "protocols/pous/src/verity_pous/codec.py": "def encode(x):\n    return x + 1\n",
    "protocols/pous/lean/Pous.lean": "theorem t : 1 = 1 := rfl\n",
    "protocols/pous/lean/lean-audit.json": '{"guarantees": {"Pous.t": "abc"}}\n',
    "protocols/pouw/src/verity_pouw/x.py": "X = 1\n",
    "tests/test_pous.py": "from verity_pous.codec import encode\nimport verity_pous.codec as c\n\n\ndef test():\n    assert encode(1) == 2\n",
}


def test_pure_rename_is_certified(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous", "security_proofs/pous")
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert len(r["renames"]) == 4 and not r["edits"]
    # the map is the moved directory, not `protocols`, since `protocols/pouw` stayed
    assert r["map"] == {"protocols/pous": "security_proofs/pous"}


def test_rename_with_a_one_byte_edit_is_refused(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/src/verity_pous/codec.py": "def encode(x):\n    return x + 2\n"})
    r = _check(repo, base)
    assert not r["certified"]
    [f] = r["refusals"]
    assert f["path"] == "security_proofs/pous/src/verity_pous/codec.py" and f["old"] == "protocols/pous/src/verity_pous/codec.py"
    assert "-    return x + 1" in f["hunk"] and "+    return x + 2" in f["hunk"]


def test_pyproject_path_edit_is_certified(tmp_path):
    repo, base = _repo(tmp_path, {**BASE_FILES, "protocols/pous/pyproject.toml": PYPROJECT,
                                  "pyproject.toml": '[tool.uv.workspace]\nmembers = ["protocols/pous", "protocols/pouw"]\n'})
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/pyproject.toml": PYPROJECT.replace("protocols/pous/data", "security_proofs/pous/data"),
                  "pyproject.toml": '[tool.uv.workspace]\nmembers = ["security_proofs/pous", "protocols/pouw"]\n'})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert {e["path"] for e in r["edits"]} == {"pyproject.toml", "security_proofs/pous/pyproject.toml"}


def test_path_edit_to_a_directory_that_did_not_move_is_refused(tmp_path):
    repo, base = _repo(tmp_path, {**BASE_FILES, "pyproject.toml": 'members = ["protocols/pous", "protocols/pouw"]\n'})
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"pyproject.toml": 'members = ["security_proofs/pous", "security_proofs/pouw"]\n'})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "pyproject.toml" and "security_proofs/pouw" in f["hunk"]


def test_non_path_edit_in_pyproject_is_refused(tmp_path):
    repo, base = _repo(tmp_path, {**BASE_FILES, "protocols/pous/pyproject.toml": PYPROJECT})
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/pyproject.toml":
                  PYPROJECT.replace("protocols/pous/data", "security_proofs/pous/data").replace('"pous"', '"pous2"')})
    r = _check(repo, base)
    [f] = r["refusals"]
    assert f["path"] == "security_proofs/pous/pyproject.toml" and f["reason"] == "edit beyond path strings"
    assert f["hunk"].splitlines()[1:] == ['-name = "pous"', '+name = "pous2"']


def test_import_line_rewrite_is_certified(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous/src/verity_pous", "security_proofs/pous/src/pous")
    _write(repo, {"tests/test_pous.py": BASE_FILES["tests/test_pous.py"].replace("verity_pous", "pous")})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert r["edits"] == [{"path": "tests/test_pous.py", "kind": "import"}]


def test_import_of_a_module_that_did_not_move_is_refused(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous/src/verity_pous", "security_proofs/pous/src/pous")
    _write(repo, {"tests/test_pous.py": BASE_FILES["tests/test_pous.py"].replace("verity_pous.codec", "pous.other")})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "tests/test_pous.py" and f["reason"] == "edit beyond import lines"


def test_a_new_file_is_refused(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/NEW.md": "hello\n"})
    [f] = _check(repo, base)["refusals"]
    assert (f["path"], f["status"], f["reason"]) == ("security_proofs/pous/NEW.md", "A", "added")


def test_names_with_spaces_and_odd_characters(tmp_path):
    files = {"old dir/a file.txt": "one\n", "old dir/tab\there.txt": "two\n", 'old dir/quo"te.txt': "three\n"}
    repo, base = _repo(tmp_path, files)
    _move(repo, "old dir", "new dir")
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert sorted(x["new"] for x in r["renames"]) == ["new dir/a file.txt", 'new dir/quo"te.txt', "new dir/tab\there.txt"]
    assert r["map"] == {"old dir": "new dir"}


def test_lean_package_move_with_relative_require(tmp_path):
    files = {**BASE_FILES, "packages/verity/lean/Verity.lean": "def v := 1\n", "protocols/pous/lean/lakefile.toml": LAKEFILE}
    repo, base = _repo(tmp_path, files)
    _move(repo, "packages/verity/lean", "verity/lean")
    _move(repo, "protocols/pous/lean", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/lakefile.toml": LAKEFILE.replace("../../../packages/verity/lean", "../../verity/lean")})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert r["edits"] == [{"path": "security_proofs/pous/lakefile.toml", "old": "protocols/pous/lean/lakefile.toml", "kind": "path"}]


def test_byte_identical_lakefile_whose_relative_require_broke_is_refused(tmp_path):
    files = {**BASE_FILES, "packages/verity/lean/Verity.lean": "def v := 1\n", "protocols/pous/lean/lakefile.toml": LAKEFILE}
    repo, base = _repo(tmp_path, files)
    _move(repo, "protocols/pous/lean", "security_proofs/pous")
    [f] = _check(repo, base)["refusals"]
    assert f["reason"] == "relative path now resolves elsewhere"


def test_lean_audit_pin_change_is_refused(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/lean/lean-audit.json": '{"guarantees": {"Pous.t": "abd"}}\n'})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "security_proofs/pous/lean/lean-audit.json"


def test_explicit_map_and_cli(tmp_path, capsys, monkeypatch):
    repo, base = _repo(tmp_path, {**BASE_FILES, "AGENTS.md": "See `protocols/pous/lean/Pous.lean`.\n"})
    _write(repo, {"AGENTS.md": "See `elsewhere/Pous.lean`.\n"})
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "doc")
    monkeypatch.chdir(repo)
    assert MC.main([base, "HEAD", "--json"]) == 1
    assert not json.loads(capsys.readouterr().out)["certified"]
    assert MC.main([base, "HEAD", "--map", "protocols/pous/lean=elsewhere"]) == 0
    assert capsys.readouterr().out.startswith("move-check: certified")
