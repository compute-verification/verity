"""`tools/verity/lean/module_keys.py`: a module's sha256 key reads everything its compilation reads, a module Lake would read
otherwise has none, and a build restores a module only from the entry under its own key whose every artifact has the sha256
the entry names."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_module_keys_under_test", HERE.parent / "module_keys.py")
MK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MK)

TOOLCHAIN = "leanprover/lean4:v4.34.0\n"


def _write(root: Path, files: dict[str, str]) -> None:
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)


def _workspace(tmp: Path, more: dict[str, str] | None = None) -> Path:
    """A root package `r` (library `R`, its default target) requiring `d` (library `D`) by path and fetching `mathlib`, with
    ``more`` files over it."""
    files = {
        "r/lakefile.toml": 'name = "r"\ndefaultTargets = ["R"]\n\n[[lean_lib]]\nname = "R"\n',
        "r/lake-manifest.json": json.dumps({"name": "r", "packages": [
            {"type": "path", "name": "d", "dir": "../d"}, {"type": "git", "name": "mathlib", "rev": "abc"}]}),
        "r/lean-toolchain": TOOLCHAIN,
        "r/R.lean": "import R.A\nimport D.X\nimport Mathlib.Logic.Basic\n",
        "r/R/A.lean": "/- A. -/\nimport D.X\n\ntheorem a : True := trivial\n",
        "r/R/B.lean": "import R.A\n",
        "d/lakefile.toml": 'name = "d"\n\n[[lean_lib]]\nname = "D"\n',
        "d/D/X.lean": "theorem x : True := trivial\n",
    } | (more or {})
    _write(tmp, files)
    return tmp / "r"


def test_a_module_system_header_s_imports_are_in_its_key(tmp_path):
    """Every import form of Lean's module system (`headers.py`) puts the imported module's key in the importer's: a key that
    skipped `public import` would restore a module built against another version of what it imports."""
    root = _workspace(tmp_path, {"r/R.lean": "module\n\npublic import R.A\nmeta import D.X\nimport Mathlib.Logic.Basic\n",
                                 "r/R/A.lean": "/- A. -/\nmodule\n\npublic meta import D.X\n\npublic section\n",
                                 "r/R/B.lean": "module\nimport all R.A\n", "d/D/X.lean": "module\n"})
    k = MK.keys(root, ["R.B"])
    assert k["R"].imports == ("D.X", "R.A") and k["R.A"].imports == ("D.X",) and k["R.B"].imports == ("R.A",)
    was = {m: v.key for m, v in k.items()}
    (root.parent / "d" / "D" / "X.lean").write_text("module\n\npublic def x := 1\n")
    assert {m for m, v in MK.keys(root, ["R.B"]).items() if v.key != was[m]} == {"D.X", "R.A", "R", "R.B"}


def test_a_key_changes_with_what_the_module_s_compilation_reads_and_not_with_where_it_is(tmp_path):
    root = _workspace(tmp_path / "one")
    k = MK.keys(root)
    assert list(k) == ["D.X", "R.A", "R"] and k["R"].imports == ("D.X", "R.A") and k["D.X"].package.name == "d"
    assert "R.B" not in k and "R.B" in MK.keys(root, ["R.B"]), "a module only `lake build +R.B` builds"
    other = tmp_path / "two"
    shutil.copytree(tmp_path / "one", other)
    assert {m: v.key for m, v in MK.keys(other / "r").items()} == {m: v.key for m, v in k.items()}

    def changed(path: str, text: str) -> set[str]:
        f = other / path
        was = f.read_text()
        f.write_text(text)
        now = MK.keys(other / "r")
        f.write_text(was)
        return {m for m in k if now[m].key != k[m].key}

    assert changed("d/D/X.lean", "theorem x : True := by trivial\n") == {"D.X", "R.A", "R"}
    assert changed("r/R/A.lean", "import D.X\n-- a comment\n") == {"R.A", "R"}
    assert changed("r/R.lean", "import R.A\nimport D.X\nimport Mathlib.Logic.Basic\n-- a comment\n") == {"R"}
    assert changed("d/lakefile.toml", 'name = "d"\n\n[leanOptions]\nautoImplicit = false\n\n[[lean_lib]]\nname = "D"\n') \
        == {"D.X", "R.A", "R"}
    assert changed("r/lean-toolchain", "leanprover/lean4:v4.35.0\n") == set(k)
    manifest = json.loads((root / "lake-manifest.json").read_text())
    manifest["packages"][1]["rev"] = "abd"
    assert changed("r/lake-manifest.json", json.dumps(manifest)) == set(k), "a fetched package's modules"


def test_a_module_lake_could_read_otherwise_has_no_key(tmp_path):
    """No key, so it is built: a header the parser can't read, an import a workspace file holds that no library claims or
    that two packages define, a package whose lakefile it doesn't follow, and every module importing one."""
    root = _workspace(tmp_path / "a", {"r/R/A.lean": "/- never closed\nimport D.X\n"})
    assert list(MK.keys(root)) == ["D.X"]
    root = _workspace(tmp_path / "b", {"r/R/A.lean": "import Stray\n", "r/Stray.lean": ""})
    assert list(MK.keys(root)) == ["D.X"]
    root = _workspace(tmp_path / "c", {"d/lakefile.toml": 'name = "d"\n\n[[lean_lib]]\nname = "D"\nroots = ["D", "R.A"]\n',
                                       "d/R/A.lean": ""})
    assert list(MK.keys(root)) == ["D.X"], "R.A is in two packages"
    root = _workspace(tmp_path / "d")
    (tmp_path / "d" / "d" / "lakefile.toml").rename(tmp_path / "d" / "d" / "lakefile.lean")
    assert MK.keys(root) == {} and MK.workspace(root) == f"{root / '../d'} has no lakefile.toml"
    for i, lakefile in enumerate(('name = "d"\n\n[[lean_lib]]\nname = "D"\nsrcDir = "src"\n',
                                  'name = "d"\n\n[[lean_lib]]\nname = "D"\nglobs = ["D.+"]\n',
                                  'name = "d"\n\n[[lean_lib]]\nname = "D"\nroots = "D"\n')):
        root = _workspace(tmp_path / f"e{i}", {"d/lakefile.toml": lakefile})
        assert MK.keys(root) == {}, lakefile


def test_a_package_s_input_files_are_in_the_key_of_each_of_its_modules(tmp_path):
    """A module may read a file as it compiles (`include_str`), which Lake tracks only as an `[[input_file]]` or
    `[[input_dir]]` target: editing a pinned checkpoint's sha512 in the `models.json` a module embeds that way left its key
    as it was. Each file such a target names, every file under a directory, is in the key of each module of its package;
    a target this reading can't follow leaves the package without keys, so its modules are built."""
    lakefile = ('name = "r"\ndefaultTargets = ["R"]\n\n[[input_file]]\nname = "models"\npath = "models.json"\ntext = true\n\n'
                '[[input_dir]]\nname = "data"\npath = "data"\n\n[[lean_lib]]\nname = "R"\nneeds = ["models", "data"]\n')
    files = {"r/lakefile.toml": lakefile, "r/models.json": '{"checkpoint_sha512": "aa"}\n', "r/data/a/b.txt": "b\n"}
    root = _workspace(tmp_path / "ok", files)
    k = MK.keys(root, ["R.B"])
    assert sorted(k) == ["D.X", "R", "R.A", "R.B"]

    def changed(path: str, text: str) -> set[str]:
        f = tmp_path / "ok" / path
        was = f.read_text() if f.exists() else None
        f.write_text(text)
        now = MK.keys(root, ["R.B"])
        if was is None:
            f.unlink()
        else:
            f.write_text(was)
        return {m for m in k if now[m].key != k[m].key}

    mine = {"R", "R.A", "R.B"}
    assert changed("r/models.json", '{"checkpoint_sha512": "ab"}\n') == mine
    assert changed("r/data/a/b.txt", "c\n") == mine
    assert changed("r/data/c.txt", "") == mine, "a file added under the directory"
    for i, (more, why) in enumerate([
            ({"r/lakefile.toml": lakefile.replace('"models.json"', '"gone.json"')}, "its input_file models has no gone.json"),
            ({"r/lakefile.toml": lakefile.replace('["models", "data"]', '["d/other"]')}, "names a target that isn't"),
            ({"r/lakefile.toml": lakefile + '\n[[extern_lib]]\nname = "x"\n'}, "doesn't follow (extern_lib)")]):
        root = _workspace(tmp_path / f"no{i}", files | more)
        assert MK.keys(root) == {} and why in MK.workspace(root), why
    root = _workspace(tmp_path / "link", files)
    (root / "data" / "l").symlink_to(root / "data" / "a")
    assert MK.keys(root) == {} and "holds a symbolic link" in MK.workspace(root)


def _setup(root: Path, mod: str, pkg: str, arts: dict[str, Path]) -> None:
    f = root / ".lake" / "build" / "ir" / MK.path_of(mod, ".setup.json")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"name": mod, "package": pkg, "isModule": False, "plugins": [], "dynlibs": [], "options": {},
                             "importArts": {i: {"olean": str(p)} for i, p in arts.items()}}))


def test_a_build_is_indexed_only_where_lake_read_its_header_as_the_parser_did(tmp_path):
    root = _workspace(tmp_path)
    pkgs = MK.workspace(root)[0]
    keyed, owners = MK.keys(root), {}

    def lib(p, m):
        return tmp_path / p / ".lake" / "build" / "lib" / "lean" / MK.path_of(m, ".olean")

    mathlib = tmp_path / "r" / ".lake" / "packages" / "mathlib" / ".lake" / "build" / "lib" / "lean" / "Mathlib.olean"
    right = {"D.X": lib("d", "D.X"), "R.A": lib("r", "R.A"), "Mathlib": mathlib, "Init": Path("/elan/lib/lean/Init.olean")}
    assert MK.indexable("R", keyed, pkgs, owners) == "no setup.json"
    _setup(root, "R", "r", right)
    assert MK.indexable("R", keyed, pkgs, owners) is None
    _setup(root, "R", "r", right | {"R.B": lib("r", "R.B")})
    assert "Lake imports ['R.B']" in MK.indexable("R", keyed, pkgs, owners)
    _setup(root, "R", "r", {i: p for i, p in right.items() if i != "D.X"})
    assert "and not ['D.X']" in MK.indexable("R", keyed, pkgs, owners)
    _setup(root, "R", "r", right | {"D.X": lib("r", "D.X")})
    assert MK.indexable("R", keyed, pkgs, owners) == "D.X is built in r, not d"
    assert owners[str(lib("r", "D.X").parent)] == "r" and owners[str(mathlib.parent)] is None, "one lookup a directory"
    _setup(root, "R", "d", right)
    assert "setup.json names 'R' of 'd'" == MK.indexable("R", keyed, pkgs, owners)


def _entry(store: Path, k, mod: str, h: str, files: dict[str, bytes]) -> dict:
    """``mod``'s index entry under ``k``, with its artifacts, as `LakeArtifacts.put` writes it."""
    (store / "artifacts").mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        (store / "artifacts" / n).write_bytes(b)
    e = {"module": mod, "scope": k.package.name, "hash": h, "outputs": {"o": [n for n in files if n.endswith(".olean")],
         "i": next(n for n in files if n.endswith(".ilean"))}, "sha256": {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}}
    (store / MK.INDEX).mkdir(exist_ok=True)
    (store / MK.INDEX / f"{k.key}.json").write_text(json.dumps(e))
    return e


def test_a_build_stages_a_module_only_under_its_key_with_every_artifact_s_sha256(tmp_path):
    root = _workspace(tmp_path / "ws")
    keyed = MK.keys(root)
    store = tmp_path / "store"
    x = _entry(store, keyed["D.X"], "D.X", "1" * 16, {"a" * 16 + ".olean": b"X", "b" * 16 + ".ilean": b"x"})
    _entry(store, keyed["R.A"], "R.A", "2" * 16, {"c" * 16 + ".olean": b"A", "d" * 16 + ".ilean": b"a"})
    _entry(store, keyed["R"], "R", "3" * 16, {"e" * 16 + ".olean": b"R", "f" * 16 + ".ilean": b"r"})
    (store / "artifacts" / ("c" * 16 + ".olean")).write_bytes(b"not what R.A's entry names")
    staged, n = MK.stage(store, root, [], tmp_path / "cache")
    assert n == 3 and staged == {("d", "1" * 16): "D.X", ("r", "3" * 16): "R"}, "R.A's artifact differs"
    assert json.loads((tmp_path / "cache" / "outputs" / "d" / f"{'1' * 16}.json").read_text()) == x["outputs"]
    assert sorted(p.name for p in (tmp_path / "cache" / "outputs").rglob("*.json")) == ["1" * 16 + ".json", "3" * 16 + ".json"]
    assert (tmp_path / "cache" / "artifacts" / ("a" * 16 + ".olean")).stat().st_ino == \
        (store / "artifacts" / ("a" * 16 + ".olean")).stat().st_ino, "a hard link on one file system"
    (root / "R.lean").write_text("import R.A\nimport D.X\n")
    staged, _ = MK.stage(store, root, [], tmp_path / "cache2")
    assert staged == {("d", "1" * 16): "D.X"}, "R's source changed: its key has no entry"
    e = json.loads((store / MK.INDEX / f"{keyed['D.X'].key}.json").read_text())
    (store / MK.INDEX / f"{keyed['D.X'].key}.json").write_text(json.dumps(e | {"module": "R.A"}))
    assert MK.stage(store, root, [], tmp_path / "cache3")[0] == {}, "an entry for another module under D.X's key"
    (store / MK.INDEX / f"{keyed['D.X'].key}.json").write_text(json.dumps(e)[:-1])
    assert MK.stage(store, root, [], tmp_path / "cache4")[0] == {}
    assert not (store / MK.INDEX / f"{keyed['D.X'].key}.json").exists(), "an entry that isn't JSON is removed"


def test_a_build_stages_under_the_store_s_lock(tmp_path, monkeypatch):
    """`LakeArtifacts.prune` takes the lock exclusively, so it can't remove an artifact while one is checked."""
    root = _workspace(tmp_path / "ws")
    store = tmp_path / "store"
    _entry(store, MK.keys(root)["D.X"], "D.X", "1" * 16, {"a" * 16 + ".olean": b"X", "b" * 16 + ".ilean": b"x"})
    held, digest = [], MK.digest

    def checked(f: Path) -> str:
        with open(store / MK.LOCK, "a") as other:
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held.append(False)
            except BlockingIOError:
                held.append(True)
        return digest(f)

    monkeypatch.setattr(MK, "digest", checked)
    assert MK.stage(store, root, [], tmp_path / "cache")[0] == {("d", "1" * 16): "D.X"}
    assert held == [True, True]


def test_two_modules_under_one_input_hash_of_one_package_are_both_left_out(tmp_path):
    """Lake finds a module in its cache by its package and input hash alone: staging both would let Lake restore one from
    the other's outputs."""
    root = _workspace(tmp_path / "ws")
    keyed = MK.keys(root)
    store = tmp_path / "store"
    _entry(store, keyed["D.X"], "D.X", "1" * 16, {"a" * 16 + ".olean": b"X", "b" * 16 + ".ilean": b"x"})
    _entry(store, keyed["R.A"], "R.A", "2" * 16, {"c" * 16 + ".olean": b"A", "d" * 16 + ".ilean": b"a"})
    _entry(store, keyed["R"], "R", "2" * 16, {"e" * 16 + ".olean": b"R", "f" * 16 + ".ilean": b"r"})
    staged, _ = MK.stage(store, root, [], tmp_path / "cache")
    assert staged == {("d", "1" * 16): "D.X"}


def test_a_module_with_the_input_hash_of_another_staged_one_is_named(tmp_path):
    root = _workspace(tmp_path / "ws")
    lib = root / ".lake" / "build" / "lib" / "lean"
    for mod, h in (("R", "3" * 16), ("R/A", "2" * 16), ("R/B", "3" * 16)):
        (lib / f"{mod}.trace").parent.mkdir(parents=True, exist_ok=True)
        (lib / f"{mod}.trace").write_text(json.dumps({"depHash": h, "outputs": {}}))
    staged = {("r", "3" * 16): "R", ("d", "2" * 16): "D.X"}
    assert [(m, o) for m, _, o in MK.misrestored(staged, root)] == [("R.B", "R")], "R.A's hash is staged in another package"
    assert MK.misrestored({("r", "3" * 16): "R", ("r", "2" * 16): "R.A"}, root)[0][1] == lib / "R" / "B.trace"
    (lib / "R" / "B.trace").write_text(json.dumps({"depHash": "3" * 16, "outputs": {}})[:-3])
    assert [(m, o) for m, _, o in MK.misrestored(staged, root)] == [("R.B", "R")], "a trace that isn't whole JSON"


def test_only_the_staged_directories_of_processes_that_are_gone_are_swept(tmp_path):
    store, proc = tmp_path / "store", tmp_path / "proc"
    (proc / "12").mkdir(parents=True)
    for d in ("12-Security", "13-Security", "other"):
        (store / MK.STAGE / d).mkdir(parents=True)
    MK.sweep(store, proc)
    assert sorted(os.listdir(store / MK.STAGE)) == ["12-Security", "other"]
