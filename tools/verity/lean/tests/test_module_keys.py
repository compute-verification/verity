"""`tools/verity/lean/module_keys.py`: a module's key is the task graph's key (`contract.key`) of its compile, which reads
everything the compile reads: its source, its package's settings and input files, the toolchain and fetched packages, and
the outputs of the workspace modules it loads, at the levels Lean loads them, so a rebuild that leaves those outputs as they
were leaves its key. A module Lake would read otherwise has none, and a build restores a module, in import order, only from
the entry under its own key whose every artifact has the digest the entry names."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_module_keys_under_test", HERE.parent / "module_keys.py")
MK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MK)
C = MK.C
#: the vectors the task graph's contract reproduces, which Verity and the executor share
VECTORS = HERE.parents[1] / "graph" / "tests" / "key_vectors.json"

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


def _out(m: str, key: str, o: str) -> str:
    return C.digest(f"{key} {o}".encode())


def _built(keyed: dict, out=_out) -> dict[str, str]:
    """{module: its key} as a build of ``keyed`` in import order gives them, each output of each module ``out(module, its
    key, the output)``: by default a digest of its key, as though every output changed with what its module compiled from."""
    digests: dict[tuple[str, str], str] = {}
    keys = {}
    for m, k in keyed.items():
        keys[m] = MK.key(m, keyed, digests)
        digests |= {(m, o): out(m, keys[m], o) for o in MK.read(MK.ALL, k.module)}
    return keys


def test_the_key_rule_is_the_contract_s_and_reproduces_its_vectors():
    """`module_keys` keys through the task graph's contract (`contract.key`), loaded from its file, and that contract gives
    the digests and keys `key_vectors.json` pins: the cache and the task graph hash a task one way."""
    assert MK.CONTRACT == HERE.parents[1] / "graph" / "verity_graph" / "contract.py"
    frozen = json.loads(VECTORS.read_text())
    for f in frozen["files"]:
        assert C.digest(bytes.fromhex(f["bytes_hex"])) == f["digest"], f["path"]
    fd = {f["path"]: f["digest"] for f in frozen["files"]}
    for e in frozen["envs"]:
        assert C.env_digest(e["spec"], fd) == e["digest"], e["name"]
    assert len(frozen["keys"]) >= 10
    for v in frozen["keys"]:
        assert C.canonical(C.key_doc(v["task"], fd, v["read_digests"], v["env_digest"])).decode() == v["canonical"], v["name"]
        assert C.key(v["task"], fd, v["read_digests"], v["env_digest"]) == v["key"], v["name"]


def test_a_module_s_key_is_the_contract_s_key_of_its_compile(tmp_path):
    """Its task: kind `lean-module/v3`, its module and package, its source and its package's lakefile by path from the root
    package, its outputs, the root's toolchain and manifest as env, and each output it reads of a module it loads by
    digest."""
    root = _workspace(tmp_path)
    keyed = MK.keys(root)
    digests = {("D.X", "olean"): C.digest(b"X"), ("R.A", "olean"): C.digest(b"A")}
    task = {"id": "lean-module:R.lean", "kind": "lean-module/v3", "argv": [], "vars": {"module": "R", "package": "r"},
            "inputs": ["R.lean", "lakefile.toml"],
            "reads": [{"task": "lean-module:../d/D/X.lean", "output": "olean"}, {"task": "lean-module:R/A.lean", "output": "olean"}],
            "outputs": [{"name": "olean", "path": ".lake/build/lib/lean/R.olean"},
                        {"name": "ilean", "path": ".lake/build/lib/lean/R.ilean"}]}
    files = {p: C.file_digest(root / p) for p in ("R.lean", "lakefile.toml")}
    env = C.env_digest(["lake-manifest.json", "lean-toolchain"],
                       {f: C.file_digest(root / f) for f in ("lake-manifest.json", "lean-toolchain")})
    reads = {"lean-module:../d/D/X.lean": {"olean": C.digest(b"X")}, "lean-module:R/A.lean": {"olean": C.digest(b"A")}}
    assert MK.key("R", keyed, digests) == C.key(task, files, reads, env)
    assert MK.key("R", keyed, {("D.X", "olean"): C.digest(b"X")}) is None, "a read whose output isn't known yet"
    assert MK.key("D.X", keyed, {}) == C.key(keyed["D.X"].task | {"reads": []}, keyed["D.X"].files, {}, env)
    assert keyed["D.X"].task["inputs"] == ["../d/D/X.lean", "../d/lakefile.toml"]


#: `module` files importing one another in every form: R imports R.A publicly, R.M for its code and all of R.B; R.A
#: imports R.P publicly and R.Q privately, R.M and R.B each privately; R.L is a legacy file over R.A
LEVELS = {
    "r/R.lean": "module\n\npublic import R.A\nmeta import R.M\nimport all R.B\n",
    "r/R/A.lean": "module\n\npublic import R.P\nimport R.Q\n",
    "r/R/P.lean": "module\n\npublic theorem p : True := trivial\n",
    "r/R/Q.lean": "module\n",
    "r/R/M.lean": "module\n\nimport R.Q\n",
    "r/R/B.lean": "module\n\nimport R.P\n",
    "r/R/L.lean": "import R.A\n",
}


def test_a_module_file_reads_what_lean_loads_of_its_imports_and_a_legacy_file_every_part(tmp_path):
    """As Lean 4.34 loads an import (`importModulesCore`): a `module` file reads the public `olean` of its imports and of
    what their `public import`s reach, the code (`ir.sig`, `ir`) of what a `meta import` reaches, and every part of an
    `import all`'s target; a legacy file every part of every module it imports, transitively. A read is in the key by its
    digest, so the key changes with exactly the outputs read."""
    keyed = MK.keys(_workspace(tmp_path, LEVELS), ["R.L"])
    assert MK.loads("R", keyed) == {"R.A": MK.PUBLIC, "R.P": MK.PUBLIC, "R.M": MK.PUBLIC | MK.CODE, "R.Q": MK.CODE,
                                    "R.B": MK.PUBLIC | MK.ALL}, \
        "R.A's private import R.Q loads only for R.M's code, and R.B's private import R.P at R.B's public level"
    assert MK.loads("R.A", keyed) == {"R.P": MK.PUBLIC, "R.Q": MK.PUBLIC}, "a module compiled loads all its imports' public part"
    assert MK.loads("R.L", keyed) == dict.fromkeys(("R.A", "R.P", "R.Q"), MK.ALL)
    digests = {(m, o): C.digest(f"{m} {o}".encode()) for m, k in keyed.items() for o in MK.read(MK.ALL, k.module)}

    def readers(m: str, o: str) -> set[str]:
        other = digests | {(m, o): C.digest(b"another build")}
        return {r for r in keyed if MK.key(r, keyed, other) != MK.key(r, keyed, digests)}

    assert readers("R.P", "olean.private") == {"R.L"}
    assert readers("R.P", "olean") == {"R", "R.A", "R.B", "R.L"}
    assert readers("R.Q", "ir") == {"R", "R.L"}, "R loads R.Q for R.M's code"
    assert readers("R.Q", "olean.server") == {"R.L"}
    assert readers("R.B", "olean.private") == {"R"}, "R imports all of R.B"
    assert readers("R.A", "olean.private") == {"R.L"}
    assert readers("R.M", "ir.sig") == {"R"}


def test_a_rebuild_that_leaves_an_import_s_outputs_as_they_were_leaves_its_importers_keys(tmp_path):
    """A proof edited in a `module` file changes its `olean.private` alone: every module that reads only the public part
    keeps its key, so its entry restores it, and a legacy file over it, which reads every part, is built."""
    root = _workspace(tmp_path, LEVELS)

    def out(m: str, key: str, o: str) -> str:
        return C.digest(f"{m} {o}".encode()) if m == "R.P" and o != "olean.private" else _out(m, key, o)

    was = _built(MK.keys(root, ["R.L"]), out)
    (root / "R" / "P.lean").write_text("module\n\npublic theorem p : True := by trivial\n")
    now = _built(MK.keys(root, ["R.L"]), out)
    assert {m for m in was if now[m] != was[m]} == {"R.P", "R.L"}
    assert {m for m, k in _built(MK.keys(root, ["R.L"])).items() if k != was[m]} == {"R.P", "R.A", "R", "R.B", "R.L"}, \
        "with every output of R.P changed, each module reading R.P changes, and R through R.A and R.B"


def test_a_module_system_header_s_imports_are_in_its_key(tmp_path):
    """Every import form of Lean's module system (`headers.py`) puts the imported module's outputs in the importer's key: a
    key that skipped `public import` would restore a module built against another version of what it imports."""
    root = _workspace(tmp_path, {"r/R.lean": "module\n\npublic import R.A\nmeta import D.X\nimport Mathlib.Logic.Basic\n",
                                 "r/R/A.lean": "/- A. -/\nmodule\n\npublic meta import D.X\n\npublic section\n",
                                 "r/R/B.lean": "module\nimport all R.A\n", "d/D/X.lean": "module\n"})
    k = MK.keys(root, ["R.B"])
    assert k["R"].imports == ("D.X", "R.A") and k["R.A"].imports == ("D.X",) and k["R.B"].imports == ("R.A",)
    was = _built(k)
    (root.parent / "d" / "D" / "X.lean").write_text("module\n\npublic def x := 1\n")
    assert {m for m, v in _built(MK.keys(root, ["R.B"])).items() if v != was[m]} == {"D.X", "R.A", "R", "R.B"}


def test_a_key_changes_with_what_the_module_s_compilation_reads_and_not_with_where_it_is(tmp_path):
    root = _workspace(tmp_path / "one")
    k = MK.keys(root)
    assert list(k) == ["D.X", "R.A", "R"] and k["R"].imports == ("D.X", "R.A") and k["D.X"].package.name == "d"
    assert "R.B" not in k and "R.B" in MK.keys(root, ["R.B"]), "a module only `lake build +R.B` builds"
    keys = _built(k)
    other = tmp_path / "two"
    shutil.copytree(tmp_path / "one", other)
    assert _built(MK.keys(other / "r")) == keys

    def changed(path: str, text: str) -> set[str]:
        f = other / path
        was = f.read_text()
        f.write_text(text)
        now = _built(MK.keys(other / "r"))
        f.write_text(was)
        return {m for m in k if now[m] != keys[m]}

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
    assert {"models.json", "data/a/b.txt"} <= set(k["R.A"].task["inputs"]) and k["D.X"].task["inputs"] == \
        ["../d/D/X.lean", "../d/lakefile.toml"]
    keys = _built(k)

    def changed(path: str, text: str) -> set[str]:
        f = tmp_path / "ok" / path
        was = f.read_text() if f.exists() else None
        f.write_text(text)
        now = _built(MK.keys(root, ["R.B"]))
        if was is None:
            f.unlink()
        else:
            f.write_text(was)
        return {m for m in k if now[m] != keys[m]}

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


def test_a_build_holds_the_outputs_a_key_reads_where_lake_leaves_them(tmp_path):
    root = _workspace(tmp_path, {"r/R.lean": "import R.A\n", "r/R/A.lean": "module\n"})
    keyed = MK.keys(root)
    lib = root / ".lake" / "build" / "lib" / "lean"
    _write(lib, {"R/A.olean": "A", "R/A.ir": "ir", "R.olean": "R"})
    assert MK.read_digests(keyed, {("R", "olean"): "given"}) == {("R.A", "olean"): C.digest(b"A"),
                                                                  ("R.A", "ir"): C.digest(b"ir"), ("R", "olean"): "given"}


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


def _entry(store: Path, key: str, scope: str, mod: str, h: str, files: dict[str, bytes]) -> dict:
    """``mod``'s index entry under ``key``, with its artifacts, as `LakeArtifacts.put` writes it."""
    (store / "artifacts").mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        (store / "artifacts" / n).write_bytes(b)
    e = {"module": mod, "scope": scope, "hash": h, "outputs": {"o": [n for n in files if n.endswith(".olean")],
         "i": next(n for n in files if n.endswith(".ilean"))}, "digests": {n: C.digest(b) for n, b in files.items()}}
    MK.entry_path(store, key).parent.mkdir(exist_ok=True)
    MK.entry_path(store, key).write_text(json.dumps(e))
    return e


def _stock(store: Path, keyed: dict, entries: dict[str, tuple[str, dict[str, bytes]]]) -> dict[str, str]:
    """Index ``entries`` ({module: (Lake's input hash, {artifact: its bytes})}, in import order), each under the key the
    digests the entries before it name give it, as `LakeArtifacts.put` indexes a build holding those artifacts; {module: its
    key}."""
    named: dict[tuple[str, str], str] = {}
    out = {}
    for m, (h, files) in entries.items():
        out[m] = MK.key(m, keyed, named)
        e = _entry(store, out[m], keyed[m].package.name, m, h, files)
        named |= {(m, n.split(".", 1)[1]): d for n, d in e["digests"].items()}
    return out


X = {"a" * 16 + ".olean": b"X", "b" * 16 + ".ilean": b"x"}
A = {"c" * 16 + ".olean": b"A", "d" * 16 + ".ilean": b"a"}
R = {"e" * 16 + ".olean": b"R", "f" * 16 + ".ilean": b"r"}


def test_a_build_stages_a_module_only_under_its_key_with_every_artifact_s_digest(tmp_path):
    root = _workspace(tmp_path / "ws")
    keyed = MK.keys(root)
    store = tmp_path / "store"
    k = _stock(store, keyed, {"D.X": ("1" * 16, X), "R.A": ("2" * 16, A), "R": ("3" * 16, R)})
    staged, n = MK.stage(store, root, [], tmp_path / "cache")
    assert n == 3 and staged == {("d", "1" * 16): "D.X", ("r", "2" * 16): "R.A", ("r", "3" * 16): "R"}
    x = json.loads(MK.entry_path(store, k["D.X"]).read_text())
    assert json.loads((tmp_path / "cache" / "outputs" / "d" / f"{'1' * 16}.json").read_text()) == x["outputs"]
    assert sorted(p.name for p in (tmp_path / "cache" / "outputs").rglob("*.json")) == [f"{c * 16}.json" for c in "123"]
    assert (tmp_path / "cache" / "artifacts" / ("a" * 16 + ".olean")).stat().st_ino == \
        (store / "artifacts" / ("a" * 16 + ".olean")).stat().st_ino, "a hard link on one file system"
    (store / "artifacts" / ("c" * 16 + ".olean")).write_bytes(b"not what R.A's entry names")
    assert MK.stage(store, root, [], tmp_path / "cache2")[0] == {("d", "1" * 16): "D.X"}, \
        "R.A's artifact differs, so R.A is built, and R, which reads it, too"
    (store / "artifacts" / ("c" * 16 + ".olean")).write_bytes(b"A")
    (root / "R.lean").write_text("import R.A\nimport D.X\n")
    assert MK.stage(store, root, [], tmp_path / "cache3")[0] == {("d", "1" * 16): "D.X", ("r", "2" * 16): "R.A"}, \
        "R's source changed: its key has no entry"
    MK.entry_path(store, k["D.X"]).write_text(json.dumps(x | {"module": "R.A"}))
    assert MK.stage(store, root, [], tmp_path / "cache4")[0] == {}, \
        "an entry for another module under D.X's key, and every other module reads D.X"
    MK.entry_path(store, k["D.X"]).write_text(json.dumps(x)[:-1])
    assert MK.stage(store, root, [], tmp_path / "cache5")[0] == {}
    assert not MK.entry_path(store, k["D.X"]).exists(), "an entry that isn't JSON is removed"


def test_a_build_keys_each_module_from_the_digests_its_imports_entries_name(tmp_path):
    """The stage keys a module once each module it reads has its entry, from the digests those entries name: an import
    rebuilt to the same outputs leaves its importers restored, and one rebuilt to other outputs leaves them to be built."""
    root = _workspace(tmp_path / "ws")
    store = tmp_path / "store"
    _stock(store, MK.keys(root), {"D.X": ("1" * 16, X), "R.A": ("2" * 16, A), "R": ("3" * 16, R)})
    (root.parent / "d" / "D" / "X.lean").write_text("theorem x : True := by trivial\n")
    _stock(store, MK.keys(root), {"D.X": ("4" * 16, X)})
    assert MK.stage(store, root, [], tmp_path / "cache")[0] == \
        {("d", "4" * 16): "D.X", ("r", "2" * 16): "R.A", ("r", "3" * 16): "R"}, "D.X's outputs are as they were"
    _stock(store, MK.keys(root), {"D.X": ("5" * 16, {"9" * 16 + ".olean": b"X'", "b" * 16 + ".ilean": b"x"})})
    assert MK.stage(store, root, [], tmp_path / "cache2")[0] == {("d", "5" * 16): "D.X"}, "D.X's olean is another"


def test_a_build_stages_under_the_store_s_lock(tmp_path, monkeypatch):
    """`LakeArtifacts.prune` takes the lock exclusively, so it can't remove an artifact while one is checked."""
    root = _workspace(tmp_path / "ws")
    store = tmp_path / "store"
    _stock(store, MK.keys(root), {"D.X": ("1" * 16, X)})
    held, digest = [], MK.digest

    def checked(f: Path) -> str:
        if tmp_path / "cache" in f.parents:
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
    store = tmp_path / "store"
    _stock(store, MK.keys(root), {"D.X": ("1" * 16, X), "R.A": ("2" * 16, A), "R": ("2" * 16, R)})
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
