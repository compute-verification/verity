"""`pins`: a pin is stale when a file it records changed, `--regen` runs its command, and a file that records a Lean source's
digest without being listed is found; on a toy repository."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import json
import os
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pins_under_test", HERE.parent / "pins.py")
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)

ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
GEN = """import hashlib, json, pathlib
src = pathlib.Path("pkg/A.lean")
pathlib.Path("vec.json").write_text(json.dumps({"inputs": {"pkg/A.lean": hashlib.sha256(src.read_bytes()).hexdigest()}}))
"""


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "A.lean").write_text("def a := 1\n")
    (tmp_path / "pkg" / "lean-toolchain").write_text("leanprover/lean4:v4\n")
    (tmp_path / "pkg" / "README.md").write_text("Regenerate:\n\nsha256sum lean-toolchain \\\n  A.lean > TRUSTED.sha256\n")
    (tmp_path / "gen.py").write_text(GEN)
    (tmp_path / ".gitignore").write_text("fetched/\n")
    subprocess.run(["sh", "-c", "sha256sum lean-toolchain A.lean > TRUSTED.sha256"], cwd=tmp_path / "pkg", check=True)
    subprocess.run(["python3", "gen.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, env=ENV)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, env=ENV)
    monkeypatch.setattr(P, "PINS", (P.Pin("pkg/TRUSTED.sha256", None, "pkg", None),
                                    P.Pin("vec.json", "inputs", "", "python3 gen.py")))
    return tmp_path


def test_fresh_pins_are_not_stale_and_an_edit_makes_both_stale_naming_their_commands(repo):
    assert P.stale(repo) == []
    (repo / "pkg" / "A.lean").write_text("def a := 2\n")
    rows = P.stale(repo)
    assert [(pin.path, s) for pin, s in rows] == [("pkg/TRUSTED.sha256", ["pkg/A.lean"]), ("vec.json", ["pkg/A.lean"])]
    report = P.report(rows, repo)
    assert "run `cd pkg && sha256sum lean-toolchain A.lean > TRUSTED.sha256`" in report
    assert "run `python3 gen.py`" in report


def test_regen_runs_each_stale_pins_command_and_leaves_nothing_stale(repo, capsys):
    (repo / "pkg" / "A.lean").write_text("def a := 2\n")
    assert P.main(["--regen", "--root", str(repo)]) == 0
    assert P.stale(repo) == []
    assert json.loads((repo / "vec.json").read_text())["inputs"]["pkg/A.lean"] == sha(b"def a := 2\n")
    assert f"{sha(b'def a := 2' + bytes([10]))}  A.lean" in (repo / "pkg" / "TRUSTED.sha256").read_text()


def test_a_command_that_fails_leaves_its_pin_stale_and_regen_says_so(repo, monkeypatch, capsys):
    monkeypatch.setattr(P, "PINS", (P.Pin("vec.json", "inputs", "", "exit 3"),))
    (repo / "pkg" / "A.lean").write_text("def a := 2\n")
    assert P.main(["--regen", "--root", str(repo)]) == 1
    err = capsys.readouterr().err
    assert "vec.json: its command exited 3" in err and "vec.json is stale (pkg/A.lean): run `exit 3`" in err


def test_a_recorded_file_that_is_gone_is_stale_unless_git_ignores_it(repo, monkeypatch):
    doc = {"inputs": {"pkg/A.lean": sha((repo / "pkg/A.lean").read_bytes()), "fetched/big.bin": "0" * 64, "gone.txt": "0" * 64}}
    (repo / "vec.json").write_text(json.dumps(doc))
    assert [s for _, s in P.stale(repo)] == [["gone.txt"]]
    (repo / "fetched").mkdir()
    (repo / "fetched" / "big.bin").write_bytes(b"x")
    assert [s for _, s in P.stale(repo)] == [["fetched/big.bin", "gone.txt"]]


def test_a_tracked_file_recording_a_lean_sources_digest_must_be_listed(repo, monkeypatch):
    assert P.unlisted(repo) == {}
    (repo / "notes.json").write_text(json.dumps({"toolchain": sha((repo / "pkg/lean-toolchain").read_bytes())}))
    (repo / "lean-audit.json").write_text(json.dumps({"digest": sha((repo / "pkg/A.lean").read_bytes())}))
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=ENV)
    assert P.unlisted(repo) == {"notes.json": ["pkg/lean-toolchain"]}
    monkeypatch.setattr(P, "PINS", ())
    assert P.unlisted(repo) == {"notes.json": ["pkg/lean-toolchain"], "pkg/TRUSTED.sha256": ["pkg/A.lean", "pkg/lean-toolchain"],
                                "vec.json": ["pkg/A.lean"]}


def test_a_readme_without_the_command_is_refused(repo):
    (repo / "pkg" / "README.md").write_text("no command here\n")
    with pytest.raises(SystemExit, match="no `sha256sum ... > TRUSTED.sha256` command"):
        P.PINS[0].shell(repo)
