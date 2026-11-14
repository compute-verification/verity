"""`tools/verity/lean/moved.py` on toy trees: where a package path of an earlier layout is after the recorded moves, which
`setup.sh` asks before it enters a package (the Cloud Agent environment's install command names a path the repository
can't change), and a build directory left at a package's former path carried to where the package is. No network, no elan.

The old names here are the point, so no layout map rewrites this file ([rewrite] exclude)."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
LEAN = HERE.parent
spec = importlib.util.spec_from_file_location("lean_moved", LEAN / "moved.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

#: three recorded moves, oldest first: out of `verity/`, an area renamed, and the directory's own rename
EARLIER = [{"verity/Security/": "Security/", "tools/lean/": "tools/verity/lean/", "tools/lean/audit.py": "tools/verity/lean/check.py"},
           {"Security/Specs/": "Security/Models/", "Security/Specs.lean": "Security/Models.lean"}]
RENAME = {"Security/": "security/", "tests/toy/Security/": "tests/toy/security/"}
PACKAGES = ("Proofs", "Properties")


def tree(tmp: Path, top: str, moves: list[dict[str, str]]) -> Path:
    """A repository whose analysis packages are under ``top``, with ``moves`` recorded."""
    root = tmp / "repo"
    for d in (top, *(f"{top}/{p}" for p in PACKAGES), f"{top}/Models"):
        (root / d).mkdir(parents=True)
    for d in (top, *(f"{top}/{p}" for p in PACKAGES)):
        (root / d / "lakefile.toml").write_text('name = "x"\n')
    (root / M.MOVES).parent.mkdir(parents=True)
    (root / M.MOVES).write_text(json.dumps({"about": "toy", "moves": [{"paths": m} for m in moves]}))
    return root


def test_a_package_path_of_any_earlier_layout_is_where_the_package_is_now(tmp_path):
    root = tree(tmp_path, "security", [*EARLIER, RENAME])
    for old, new in [("Security", "security"), ("Security/Proofs", "security/Proofs"), ("verity/Security", "security"),
                     ("verity/Security/Properties/", "security/Properties"), ("Security/Specs", "security/Models"),
                     ("security", "security"), ("./security/Proofs", "security/Proofs")]:
        assert M.resolve(old, root) == new, old
    (root / "Security" / ".lake").mkdir(parents=True)
    assert M.resolve("Security", root) == "security", "the old directory a build left behind is not the package"


def test_a_path_no_move_knows_or_that_leads_to_no_directory_is_refused(tmp_path):
    root = tree(tmp_path, "security", [*EARLIER, RENAME])
    for path in ("Nowhere", "Securit", "Security2/Proofs", "Security/Gone", "verity/Security/Specs.lean", "../repo/security",
                 "/abs/Security", "tests/toy/Security"):
        assert M.resolve(path, root) is None, path


def test_before_the_move_the_old_path_is_the_directory_and_the_new_one_is_unknown(tmp_path):
    """The map's change lands before the generated commit: until the record has the move, nothing is taken through it."""
    root = tree(tmp_path, "Security", EARLIER)
    assert M.resolve("Security", root) == "Security" and M.resolve("verity/Security/Proofs", root) == "Security/Proofs"
    assert M.resolve("security", root) is None
    (root / M.MOVES).unlink()
    assert M.resolve("Security", root) == "Security" and M.resolve("verity/Security", root) is None, "a tree with no record"
    (root / M.MOVES).write_text("{")
    with pytest.raises(ValueError):                    # a record that doesn't read is not "no moves"
        M.resolve("Security", root)


def test_former_paths_are_listed_latest_first():
    moves = [*EARLIER, RENAME]
    assert M.former("security/Proofs", moves) == ["Security/Proofs", "verity/Security/Proofs"]
    assert M.former("security/Models", moves) == ["Security/Models", "Security/Specs", "verity/Security/Specs"]
    assert M.former("tools/verity/lean", moves) == ["tools/lean"] and M.former("verity/core", moves) == []
    assert M.now("tools/lean/audit.py", moves) == "tools/verity/lean/check.py"


def _run(root: Path, *argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    script = root / "tools" / "verity" / "lean" / "moved.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(LEAN / "moved.py", script)
    return subprocess.run([sys.executable, str(script), *argv], cwd=cwd or root, capture_output=True, text=True)


def test_the_command_prints_the_directory_from_the_root_or_exits_1(tmp_path):
    root = tree(tmp_path, "security", [*EARLIER, RENAME])
    r = _run(root, "Security/Proofs")
    assert (r.returncode, r.stdout) == (0, "security/Proofs\n"), r.stderr
    r = _run(root, str(root / "verity" / "Security"), cwd=tmp_path)
    assert (r.returncode, r.stdout) == (0, "security\n"), "an absolute path, from another directory"
    r = _run(root, "Nowhere")
    assert r.returncode == 1 and r.stdout == "" and "no recorded layout move" in r.stderr


FAKE = "#!/bin/sh\necho \"$(basename \"$0\") $* in $(pwd)\" >> \"$STATE\"\nexit 0\n"


def test_the_install_command_s_old_package_path_sets_up_the_moved_package(tmp_path):
    """`bash tools/lean/setup.sh Security`, as the environment's install command runs it, on a tree where the package is
    `security/`: the shim passes the path on, the script takes it through the moves and says so, and Lake runs in the
    package. A path no move knows fails before anything is installed."""
    root = tree(tmp_path, "security", [*EARLIER, RENAME])
    (root / "security" / "lean-toolchain").write_text("leanprover/lean4:v9.9.9\n")
    for rel in ("tools/lean/setup.sh", "tools/verity/lean/setup.sh", "tools/verity/lean/moved.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(LEAN.parents[2] / rel, root / rel)
    (root / ".gitignore").write_text(".lake/\n")
    for argv in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"]):
        subprocess.run(["git", *argv], cwd=root, check=True, capture_output=True)
    _builds(root, {"Security": "spec", "Security/Proofs": "proofs"})      # what the machine built before the move
    bin_, state = tmp_path / "bin", tmp_path / "state"
    bin_.mkdir()
    for name in ("elan", "lean", "lake"):
        (bin_ / name).write_text(FAKE)
        (bin_ / name).chmod(0o755)
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "HOME": str(tmp_path), "STATE": str(state), "LEAN_CACHE": "off",
           "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    r = subprocess.run(["bash", "tools/lean/setup.sh", "Security"], cwd=root, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "setup.sh: Security is security since the layout moves" in r.stdout
    assert "lean: security/Proofs: its build directory moved with it from Security/Proofs (a layout move)" in r.stdout
    assert (root / "security" / ".lake" / "build" / "olean").read_text() == "spec" and not (root / "Security").exists()
    log = state.read_text().splitlines()
    assert f"lake env true in {root / 'security'}" in log and log[0] == f"elan toolchain install leanprover/lean4:v9.9.9 in {root / 'security'}"
    state.unlink()
    r = subprocess.run(["bash", "tools/lean/setup.sh", "Nowhere"], cwd=root, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "no recorded layout move" in r.stderr and not state.exists()


def _builds(root: Path, at: dict[str, str]) -> None:
    for d, name in at.items():
        (root / d / ".lake" / "build").mkdir(parents=True)
        (root / d / ".lake" / "build" / "olean").write_text(name)


def test_a_build_directory_left_at_a_former_path_moves_with_its_package(tmp_path):
    """A move renames tracked files; `.lake` is not one, so it stays where the package was and the package starts with
    none: every module would compile again. Each package's, nested ones included, is carried to where the package is."""
    root = tree(tmp_path, "security", [*EARLIER, RENAME])
    _builds(root, {"Security": "spec", "Security/Proofs": "proofs", "verity/Security/Properties": "older", "security/Properties": "kept"})
    (root / "Security" / "Definitions").mkdir()
    (root / "Security" / "Definitions" / "A.lean").write_text("-- a stale copy, or someone's untracked work\n")
    said: list[str] = []
    pkgs = ["security", "security/Proofs", "security/Properties", "verity/core"]
    assert M.carry(root, pkgs, M.recorded(root), said.append) == {"Security": "security", "Security/Proofs": "security/Proofs"}
    assert [(root / p / ".lake" / "build" / "olean").read_text() for p in pkgs[:3]] == ["spec", "proofs", "kept"]
    assert said == ["lean: security: its build directory moved with it from Security (a layout move)",
                    "lean: security/Proofs: its build directory moved with it from Security/Proofs (a layout move)"]
    assert not (root / "Security" / "Proofs").exists() and (root / "Security" / "Definitions" / "A.lean").is_file(), \
        "an emptied directory goes; what a checkout still holds there stays"
    assert (root / "verity" / "Security" / "Properties" / ".lake").is_dir(), "a package with a build directory keeps its own"
    assert M.carry(root, pkgs, M.recorded(root), said.append) == {} and len(said) == 2


def test_a_tree_of_copies_loses_the_former_directory_and_an_older_path_is_reached(tmp_path):
    """A slot's tree holds only copies of tracked files beside the builds: `prune` removes the former directory with them.
    A tree last used two layouts ago has its build at the older path."""
    root = tree(tmp_path, "security", [*EARLIER, RENAME])
    _builds(root, {"verity/Security": "spec", "verity/Security/Proofs": "proofs"})
    (root / "verity" / "Security" / "A.lean").write_text("-- a copy\n")
    got = M.carry(root, ["security", "security/Proofs"], M.recorded(root), lambda s: None, prune=True)
    assert got == {"verity/Security": "security", "verity/Security/Proofs": "security/Proofs"}
    assert (root / "security" / "Proofs" / ".lake" / "build" / "olean").read_text() == "proofs"
    assert not (root / "verity" / "Security").exists() and (root / "verity").is_dir()
