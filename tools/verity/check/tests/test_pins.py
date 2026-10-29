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


FOUND = "Regenerate:\n\nsha256sum lean-toolchain A.lean \\\n  $(find Defs -name '*.lean' | LC_ALL=C sort -u) > TRUSTED.sha256\n"


def test_a_file_in_a_pins_scope_that_it_doesnt_list_makes_it_stale_and_its_command_lists_it(repo):
    """The pin's command finds the `.lean` files of a tree: one added there after the pin was written (in a new directory)
    makes the pin stale while every file it lists is unchanged, `--stale` names it with the command, and `--regen` writes
    the pin that lists it and passes."""
    (repo / "pkg" / "README.md").write_text(FOUND)
    (repo / "pkg" / "Defs").mkdir()
    (repo / "pkg" / "Defs" / "B.lean").write_text("def b := 1\n")
    assert P.main(["--regen", "--root", str(repo)]) == 0 and P.stale(repo) == []
    (repo / "pkg" / "Defs" / "Claims").mkdir()
    (repo / "pkg" / "Defs" / "Claims" / "C.lean").write_text("def c := 1\n")
    assert P.PINS[0].gap(repo) == (["pkg/Defs/Claims/C.lean"], [])
    assert [(pin.path, s) for pin, s in P.stale(repo)] == [("pkg/TRUSTED.sha256", ["pkg/Defs/Claims/C.lean"])]
    assert P.main(["--stale", "--root", str(repo)]) == 1
    assert P.report(P.stale(repo), repo) == (
        "pkg/TRUSTED.sha256 is stale (its command hashes 1 file it doesn't list: pkg/Defs/Claims/C.lean): run `cd pkg && "
        "sha256sum lean-toolchain A.lean $(find Defs -name '*.lean' | LC_ALL=C sort -u) > TRUSTED.sha256`")
    assert P.main(["--regen", "--root", str(repo)]) == 0
    assert P.stale(repo) == [] and P.main(["--stale", "--root", str(repo)]) == 0
    assert (repo / "pkg" / "TRUSTED.sha256").read_text().splitlines()[2:] == [
        f"{sha(b'def b := 1' + bytes([10]))}  Defs/B.lean", f"{sha(b'def c := 1' + bytes([10]))}  Defs/Claims/C.lean"]


def test_a_file_the_pin_lists_that_its_command_no_longer_hashes_makes_it_stale(repo):
    (repo / "pkg" / "README.md").write_text(FOUND)
    (repo / "pkg" / "Defs").mkdir()
    (repo / "pkg" / "Defs" / "B.lean").write_text("def b := 1\n")
    assert P.main(["--regen", "--root", str(repo)]) == 0
    (repo / "pkg" / "README.md").write_text(FOUND.replace("Defs", "Other"))
    assert [s for _, s in P.stale(repo)] == [["pkg/Defs/B.lean"]]
    assert "(it lists 1 file its command doesn't hash: pkg/Defs/B.lean)" in P.report(P.stale(repo), repo)


def test_a_readme_without_the_command_is_refused(repo):
    (repo / "pkg" / "README.md").write_text("no command here\n")
    with pytest.raises(SystemExit, match="no `sha256sum ... > TRUSTED.sha256` command"):
        P.PINS[0].shell(repo)


#: `lake update` as far as a manifest goes: every package the lakefile requires, those its path requirements require after it
#: (their `dir` joined from here, as Lake writes it), unnormalized and in that order
FAKE_LAKE = r'''#!/usr/bin/env python3
import json, os, sys, tomllib
assert sys.argv[1:] == ["update"], sys.argv
def entries(d, rel, inherited):
    out = []
    for r in tomllib.load(open(os.path.join(d, "lakefile.toml"), "rb")).get("require", []):
        if "path" in r:
            out.append({"type": "path", "name": r["name"], "dir": os.path.join(rel, r["path"]), "inherited": inherited})
            out += entries(os.path.join(d, r["path"]), os.path.join(rel, r["path"]), True)
        else:
            out.append({"type": "git", "name": r["name"], "url": r["git"], "rev": r["rev"], "inherited": inherited})
    return out
seen, pkgs = set(), []
for e in entries(".", "", False):
    if e["name"] not in seen:
        seen.add(e["name"])
        pkgs.append(e)
open("lake-manifest.json", "w").write(json.dumps({"version": "1.1.0", "packagesDir": ".lake/packages", "packages": pkgs}, indent=1) + "\n")
'''
ID = {**ENV, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
GIT = '[[require]]\nname = "mathlib"\ngit = "https://example.invalid/mathlib"\nrev = "' + "a" * 40 + '"\n'


def _commit(root: Path, msg: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, env=ENV)
    subprocess.run(["git", "commit", "-qm", msg], cwd=root, check=True, env=ID)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, env=ENV, capture_output=True, text=True).stdout.strip()


def _lake(root: Path, *pkgs: str) -> None:
    for p in pkgs:
        subprocess.run(["lake", "update"], cwd=root / p, check=True)


@pytest.fixture
def lake_repo(tmp_path, monkeypatch):
    """`top` requires `base` by path, `base` requires Mathlib by git; a pin records top's manifest."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "lake").write_text(FAKE_LAKE)
    (bin_ / "lake").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    root = tmp_path / "repo"
    for d, lakefile in (("base", f'name = "base"\n\n{GIT}\n# the rest of base\n\n\n\n\n'),
                        ("top", 'name = "top"\n\n[[require]]\nname = "base"\npath = "../base"\n')):
        (root / d).mkdir(parents=True)
        (root / d / "lakefile.toml").write_text(lakefile)
        (root / d / "lean-toolchain").write_text("leanprover/lean4:v4\n")
    (root / "gen.py").write_text(GEN.replace("pkg/A.lean", "top/lake-manifest.json"))
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True, env=ENV)
    _lake(root, "base", "top")
    subprocess.run(["python3", "gen.py"], cwd=root, check=True)
    base = _commit(root, "base")
    monkeypatch.setattr(P, "PINS", (P.Pin("vec.json", "inputs", "", "python3 gen.py"),))
    return root, base


def _side(root: Path, base: str, name: str, before: bool, top: bool = True) -> str:
    """A branch off ``base`` whose `base` package requires a new package ``name`` by path (first in its lakefile, or last), with
    base's manifest regenerated there, and top's and the pin unless not ``top`` (as #1524 and #1497 left Security/Proofs')."""
    subprocess.run(["git", "checkout", "-q", "-b", name, base], cwd=root, check=True, env=ENV)
    (root / name).mkdir()
    (root / name / "lakefile.toml").write_text(f'name = "{name}"\n')
    req = f'[[require]]\nname = "{name}"\npath = "../{name}"\n'
    lf = (root / "base" / "lakefile.toml").read_text()
    head, _, rest = lf.partition("\n\n")
    (root / "base" / "lakefile.toml").write_text(f"{head}\n\n{req}\n{rest}" if before else lf + req)
    _lake(root, "base", *(["top"] if top else []))
    subprocess.run(["python3", "gen.py"], cwd=root, check=True)
    return _commit(root, name)


def test_a_merge_of_two_branches_each_adding_a_lake_dependency_gets_the_manifests_lake_writes_and_fresh_pins(lake_repo):
    """Two branches that each add a path dependency to `security_properties` merge the manifests' text, and a dependent
    package's manifest would hold neither. The merge reruns `lake update` in both packages, `base` first, takes OURS for
    the conflicted manifests before it, and the pin of top's manifest is rewritten by its command, not found stale later."""
    root, base = lake_repo
    x, y = _side(root, base, "x", True), _side(root, base, "y", False, top=False)
    subprocess.run(["git", "checkout", "-q", x], cwd=root, check=True, env=ENV)
    m = subprocess.run(["git", "merge", "--no-commit", "--no-ff", y], cwd=root, env=ID, capture_output=True, text=True)
    conflicted = subprocess.run(["git", "diff", "--name-only", "--diff-filter=U"], cwd=root, env=ENV, capture_output=True,
                                text=True).stdout.split()
    assert m.returncode == 0 and conflicted == [], m.stdout
    names = [e["name"] for e in json.loads((root / "top" / "lake-manifest.json").read_text())["packages"]]
    assert "y" not in names, "git's merge alone leaves top's manifest without y"
    assert P.lake_affected(root, x, y) == ["base", "top"]
    rc, doc = P.merged(root, x, y, conflicted)
    assert rc == 0, doc
    names = [e["name"] for e in json.loads((root / "top" / "lake-manifest.json").read_text())["packages"]]
    assert names == ["base", "x", "mathlib", "y"]
    fresh = (root / "top" / "lake-manifest.json").read_bytes()
    _lake(root, "base", "top")
    assert (root / "top" / "lake-manifest.json").read_bytes() == fresh, "what `lake update` writes"
    assert P.stale(root) == [] and {"top/lake-manifest.json", "vec.json", *conflicted} <= set(doc["regenerated"])


def test_a_conflict_in_a_file_merged_doesn_t_generate_exits_3(lake_repo):
    root, base = lake_repo
    rc, doc = P.merged(root, base, base, ["base/lakefile.toml"])
    assert rc == 3 and "base/lakefile.toml" in doc["notes"][0]


def test_a_manifest_that_fetches_the_same_packages_in_another_order_keeps_its_bundle_under_its_new_key(lake_repo, monkeypatch):
    """Security/Proofs on 0fcc's fix: `lake update` reordered the fetched entries, so the dependency key moved and the bundle,
    the same packages at the same revisions, was pinned under it. A manifest that fetches anything else is only named."""
    root, base = lake_repo
    git_entry = {"type": "git", "name": "m2", "url": "u", "rev": "b" * 40}
    doc = json.loads((root / "top" / "lake-manifest.json").read_text())
    doc["packages"].append(git_entry)
    (root / "top" / "lake-manifest.json").write_text(json.dumps(doc) + "\n")
    tc = (root / "top" / "lean-toolchain").read_bytes()
    old = P.deps_key((root / "top" / "lake-manifest.json").read_text(), tc)
    (root / P.LEAN_DEPS).parent.mkdir(parents=True)
    pin = {"package": "top", "sha256": "s", "bytes": 1, "run": "r"}
    (root / P.LEAN_DEPS).write_text(json.dumps({"deps": {old: pin}, "schema": "x"}))
    side = _commit(root, "two fetched packages")
    doc["packages"] = [doc["packages"][-1], *doc["packages"][:-1]]
    (root / "top" / "lake-manifest.json").write_text(json.dumps(doc) + "\n")
    new = P.deps_key((root / "top" / "lake-manifest.json").read_text(), tc)
    notes: list[str] = []
    assert new != old and P.repoint(root, "top", [side], notes)
    assert json.loads((root / P.LEAN_DEPS).read_text())["deps"] == {new: pin} and "pinned as" in notes[0]
    doc["packages"][0] = {**git_entry, "rev": "c" * 40}
    (root / "top" / "lake-manifest.json").write_text(json.dumps(doc) + "\n")
    notes.clear()
    assert not P.repoint(root, "top", [side], notes) and "fetches its dependencies cold" in notes[0]
