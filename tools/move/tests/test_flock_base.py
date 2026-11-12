"""`flock_base.py` on a toy tree: a base package Flock's package requires by path, a re-export, and importers."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

MOVE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("flock_base_for_tests", MOVE / "flock_base.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

FLOCK = B.FLOCK
TREE = {
    f"{FLOCK}/lakefile.toml": 'name = "flock_verifier"\n\n[[require]]\nname = "core_base"\npath = "../../../circuits/base"\n\n'
                              '[[require]]\nname = "fetched"\ngit = "https://example.invalid/x"\n\n[[lean_lib]]\nname = "Flock"\n',
    f"{FLOCK}/Flock/Hash.lean": "module\n\npublic import Base.Hash\n\n/-! A re-export. -/\n",
    f"{FLOCK}/Flock/Verify.lean": "module\n\npublic import Flock.Hash\nimport all Flock.Hash\npublic import Flock.Record\n"
                                  "-- import Flock.Hash is how it was\n\ndef verify := 1\n",
    f"{FLOCK}/Flock/Record.lean": "module\n\ndef record := 1\n",
    "verity/circuits/base/lakefile.toml": 'name = "core_base"\n\n[[lean_lib]]\nname = "Base"\n',
    "verity/circuits/base/Base/Hash.lean": "module\n\nnamespace Flock\nprivate def round := 1\nend Flock\n",
    "verity/circuits/base/Base/Pow2.lean": "module\n\ndef pow := 1\n",
    "security/Proofs/P.lean": "import Flock.Hash\n\nopen private round from Flock.Hash\nopen private record from Flock.Record\n",
    "security/Proofs/Q.lean": "open private round Round.busy natJ?\n  from Flock.Hash\n\ndef from_ := 1\n",
}


def tree(tmp_path: Path) -> Path:
    for f, text in TREE.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text)
    for args in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "toy"]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True)
    return tmp_path


def test_the_moved_modules_are_read_from_the_lakefiles(tmp_path):
    """A module of a package required by path was Flock's; one of ``NEW`` never was, and a fetched package has none."""
    root = tree(tmp_path)
    assert B.moved(root) == {"Hash": "Base.Hash"}
    assert B.moves(root) == {"_private.Flock.Hash": "_private.Base.Hash"}


def test_private_names_the_module_a_declaration_moved_to_and_no_other(tmp_path):
    root = tree(tmp_path)
    assert B.private(root) == [("security/Proofs/P.lean", "Flock.Hash", "Base.Hash"),
                               ("security/Proofs/Q.lean", "Flock.Hash", "Base.Hash")]
    assert (root / "security/Proofs/P.lean").read_text() == (
        "import Flock.Hash\n\nopen private round from Base.Hash\nopen private record from Flock.Record\n")
    assert (root / "security/Proofs/Q.lean").read_text() == (
        "open private round Round.busy natJ?\n  from Base.Hash\n\ndef from_ := 1\n")
    assert B.private(root) == []


def test_rewrite_renames_each_import_and_deletes_the_re_export(tmp_path):
    root = tree(tmp_path)
    lines, gone = B.rewrite(root, dry=True)
    assert (dict(lines), gone) == ({"security/Proofs": 1, f"{FLOCK}/Flock": 2}, [f"{FLOCK}/Flock/Hash.lean"])
    assert (root / gone[0]).exists() and "import Flock.Hash" in (root / "security/Proofs/P.lean").read_text()
    assert B.rewrite(root) == (lines, gone)
    assert not (root / gone[0]).exists()
    assert (root / FLOCK / "Flock/Verify.lean").read_text() == (
        "module\n\npublic import Base.Hash\nimport all Base.Hash\npublic import Flock.Record\n"
        "-- import Flock.Hash is how it was\n\ndef verify := 1\n")
    assert (root / "security/Proofs/P.lean").read_text().startswith("import Base.Hash\n")
    assert B.rewrite(root) == ({}, [])


def test_a_lock_allows_the_module_where_it_is():
    lock = {"layers": {"X": {"imports": ["Flock.Hash", "Flock.Record", "Flock", "Lean"], "except": ["Flock.Hash"], "why": "w"}}}
    assert B.layers(lock, {"Hash": "Base.Hash"}) == 2
    assert lock["layers"]["X"] == {"imports": ["Base.Hash", "Flock.Record", "Flock", "Lean"], "except": ["Base.Hash"], "why": "w"}
