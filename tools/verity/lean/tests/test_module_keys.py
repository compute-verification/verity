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
import threading
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


def _canon(doc) -> str:
    return C.digest(json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def test_a_module_s_key_is_the_contract_s_key_of_its_compile(tmp_path):
    """Its task: kind `lean-module/v4`; its module, its package and the library that owns it; its source and its package's
    lakefile, each by its package's name and its path there, the lakefile's digest that of its settings without
    `defaultTargets` and each library's `roots` and `globs`; its outputs; as env the root's toolchain and its manifest's
    fetched entries, each whole but `inherited`; and each output it reads of a module it loads by digest."""
    root = _workspace(tmp_path)
    keyed = MK.keys(root)
    digests = {("D.X", "olean"): C.digest(b"X"), ("R.A", "olean"): C.digest(b"A")}
    task = {"id": "lean-module:r/R.lean", "kind": "lean-module/v4", "argv": [],
            "vars": {"module": "R", "package": "r", "library": "R"}, "inputs": ["r/R.lean", "r/lakefile.toml"],
            "reads": [{"task": "lean-module:d/D/X.lean", "output": "olean"}, {"task": "lean-module:r/R/A.lean", "output": "olean"}],
            "outputs": [{"name": "olean", "path": "r/.lake/build/lib/lean/R.olean"},
                        {"name": "ilean", "path": "r/.lake/build/lib/lean/R.ilean"}]}
    files = {"r/R.lean": C.file_digest(root / "R.lean"), "r/lakefile.toml": _canon({"name": "r", "lean_lib": [{"name": "R"}]})}
    parts = {"lean-toolchain": C.file_digest(root / "lean-toolchain"),
             "fetched packages": _canon({"manifest": {}, "fetched": [{"type": "git", "name": "mathlib", "rev": "abc"}]})}
    env = C.env_digest(["fetched packages", "lean-toolchain"], parts)
    reads = {"lean-module:d/D/X.lean": {"olean": C.digest(b"X")}, "lean-module:r/R/A.lean": {"olean": C.digest(b"A")}}
    assert MK.key("R", keyed, digests) == C.key(task, files, reads, env)
    assert keyed["R"].parts == parts == MK.env_parts(root)
    assert MK.key("R", keyed, {("D.X", "olean"): C.digest(b"X")}) is None, "a read whose output isn't known yet"
    assert MK.key("D.X", keyed, {}) == C.key(keyed["D.X"].task | {"reads": []}, keyed["D.X"].files, {}, env)
    assert keyed["D.X"].task["inputs"] == ["d/D/X.lean", "d/lakefile.toml"]
    assert keyed["D.X"].task["vars"] == {"module": "D.X", "package": "d", "library": "D"}


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


#: a root package of two libraries with different options: `R` (its default target, which also owns the module `T`) and `S`
LIBS = ('name = "r"\ndefaultTargets = ["R"]\n\n[[lean_lib]]\nname = "R"\nroots = ["R", "T"]\n\n'
        '[[lean_lib]]\nname = "S"\nroots = ["S"]\n\n[lean_lib.leanOptions]\nautoImplicit = true\n')


def _changed(root: Path, keys: dict[str, str], path: str, text: str, extra=()) -> set[str]:
    """The modules of ``keys`` whose key is another with ``text`` at ``path`` of ``root``'s tree (then put back)."""
    f = root.parent / path
    was = f.read_text() if f.exists() else None
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    now = _built(MK.keys(root, extra))
    f.unlink() if was is None else f.write_text(was)
    return {m for m in keys if now.get(m) != keys[m]}


def test_every_setting_of_a_lakefile_is_in_the_key_but_the_lists_of_what_a_build_compiles(tmp_path):
    """The key leaves out of a lakefile only `defaultTargets` and each library's `roots` (and `globs`, which this reading
    refuses): they say which modules a build compiles, and Lake's input hash of a module reads none of them. Anything
    else moves the key of every module of the package, a setting this reading has never met included."""
    root = _workspace(tmp_path, {"r/lakefile.toml": LIBS, "r/S.lean": "", "r/R/A.lean": ""})
    build = ["R", "S", "R.B"]
    keys = _built(MK.keys(root, build))
    mine = {"R", "R.A", "R.B", "S"}
    assert set(keys) == mine | {"D.X"}
    for same in (LIBS.replace('defaultTargets = ["R"]', 'defaultTargets = ["R", "S"]'),
                 LIBS.replace('defaultTargets = ["R"]\n', ""),
                 LIBS.replace('roots = ["S"]', 'roots = ["S", "U"]'),
                 LIBS.replace('roots = ["R", "T"]\n', "")):
        assert same != LIBS and _changed(root, keys, "r/lakefile.toml", same, build) == set(), same
    for other in (LIBS.replace('name = "r"', 'name = "r"\nversion = "0.1.0"'),
                  LIBS.replace('name = "r"', 'name = "r"\nmoreLeanArgs = ["-DautoImplicit=true"]'),
                  LIBS.replace('name = "r"', 'name = "r"\nweakLeanArgs = ["-Dtrace.x=true"]'),
                  LIBS.replace('name = "r"', 'name = "r"\nbuildType = "debug"'),
                  LIBS.replace('name = "r"', 'name = "r"\nplatformIndependent = true'),
                  LIBS.replace('name = "r"', 'name = "r"\nsomethingNobodyThoughtOf = 1'),
                  LIBS.replace("autoImplicit = true", "autoImplicit = false"),
                  LIBS.replace('name = "S"', 'name = "S"\nmoreLeanArgs = ["-DautoImplicit=true"]'),
                  LIBS.replace('name = "S"', 'name = "S"\nsomethingNobodyThoughtOf = 1'),
                  LIBS + '\n[[lean_exe]]\nname = "tool"\nroot = "Tool"\n',
                  LIBS + '\n[[require]]\nname = "d"\npath = "../d"\n',
                  LIBS + '\n[leanOptions]\nrelaxedAutoImplicit = false\n'):
        assert _changed(root, keys, "r/lakefile.toml", other, build) == mine, other
    lib = 'name = "d"\n\n[[lean_lib]]\nname = "D"\n'
    assert _changed(root, keys, "d/lakefile.toml", lib + 'roots = ["D", "E"]\n', build) == set()
    assert _changed(root, keys, "d/lakefile.toml", lib + 'leanOptions = {autoImplicit = false}\n', build) == {"D.X", "R"}, \
        "a path dependency's setting moves its modules' keys, and through what they read their importers'"


def test_where_a_required_package_is_is_in_no_key(tmp_path):
    """A lakefile's `require` says which package and, by its `path`, where it is. Where decides nothing a compile reads
    but that package's modules, which a key reads by their outputs' digests, each under a path from its own package's
    directory: a required package moved, with the `path` and the manifest's `dir` rewritten, rekeys no module. Which
    packages a lakefile requires, and each other field of a `require`, still rekeys the package's modules."""
    req = '\n[[require]]\nname = "d"\npath = "../d"\n'
    root = _workspace(tmp_path, {"r/lakefile.toml": LIBS + req})
    keys = _built(MK.keys(root))
    mine = {"R", "R.A"}
    assert set(keys) == mine | {"D.X"}
    assert _changed(root, keys, "r/lakefile.toml", LIBS + req.replace("../d", "../r/../d")) == set()
    fetched = '\n[[require]]\nname = "mathlib"\ngit = "https://example.org/mathlib"\nrev = "abc"\n'
    for other in (LIBS, LIBS + req.replace('name = "d"', 'name = "e"'), LIBS + req + fetched,
                  LIBS + req + 'scope = "s"\n', LIBS + req + 'somethingNobodyThoughtOf = 1\n'):
        assert _changed(root, keys, "r/lakefile.toml", other) == mine, other
    with_fetched = _built(MK.keys(_workspace(tmp_path, {"r/lakefile.toml": LIBS + req + fetched})))
    assert _changed(root, with_fetched, "r/lakefile.toml", LIBS + req + fetched.replace("abc", "abd")) == mine
    (root / "lakefile.toml").write_text(LIBS + req.replace("../d", "../moved/to/d"))
    (root / "lake-manifest.json").write_text(_manifest(root, lambda doc: doc["packages"][0].update(dir="../moved/to/d")))
    (tmp_path / "moved" / "to").mkdir(parents=True)
    shutil.move(tmp_path / "d", tmp_path / "moved" / "to" / "d")
    assert _built(MK.keys(root)) == keys


def test_a_module_that_moves_to_a_library_with_other_options_has_another_key(tmp_path):
    """A library sets its modules' options, and `roots` say which library owns a module: with `roots` out of the key, the
    key names the owner. Lake compiles a module that several targets of its package own with the settings of whichever
    a build reaches it through, so it has a key, naming them all, only where they differ in nothing a compile reads."""
    root = _workspace(tmp_path, {"r/lakefile.toml": LIBS, "r/S.lean": "", "r/T.lean": "", "r/R/A.lean": ""})
    build = ["S", "T", "R.B"]
    keyed = MK.keys(root, build)
    assert {m: k.task["vars"]["library"] for m, k in keyed.items()} == \
        {"D.X": "D", "R.A": "R", "R.B": "R", "R": "R", "S": "S", "T": "R"}
    keys = _built(keyed)
    (root / "lakefile.toml").write_text(LIBS.replace('roots = ["R", "T"]', 'roots = ["R"]').replace('roots = ["S"]', 'roots = ["S", "T"]'))
    now = MK.keys(root, build)
    assert now["T"].task["vars"]["library"] == "S" and now["T"].files == keyed["T"].files, "the lakefile's digest is as it was"
    assert {m for m in keys if _built(now)[m] != keys[m]} == {"T"}
    (root / "lakefile.toml").write_text(LIBS.replace('roots = ["S"]', 'roots = ["S", "R.A"]'))
    assert sorted(MK.keys(root, build)) == ["D.X", "S", "T"], "R.A is R's and S's, so it and its importers have no key"
    exe = '\n[[lean_exe]]\nname = "tool"\nexeName = "t"\nsupportInterpreter = true\n'
    (root / "lakefile.toml").write_text(LIBS + exe + 'root = "Tool"\n')
    (root / "Tool.lean").write_text("import R.A\n")
    assert MK.keys(root, ["Tool"])["Tool"].task["vars"]["library"] == "lean_exe:tool"
    (root / "lakefile.toml").write_text(LIBS + exe + 'root = "R.A"\n')
    owned = MK.keys(root, build)
    assert owned["R.A"].task["vars"]["library"] == "R+lean_exe:tool", "a library's module that is an executable's root"
    for differ in (exe + 'root = "R.A"\nneeds = ["x"]\n', exe + 'root = "R.A"\nsomethingNobodyThoughtOf = 1\n',
                   exe + 'root = "R.A"\nleanOptions = {autoImplicit = true}\n', exe + 'root = "S"\n'):
        (root / "lakefile.toml").write_text(LIBS + differ)
        assert not {"R.A", "S"} <= set(MK.keys(root, build)), differ


def _manifest(root: Path, change) -> str:
    doc = json.loads((root / "lake-manifest.json").read_text())
    change(doc)
    return json.dumps(doc)


def test_the_env_is_the_toolchain_and_every_fetched_entry_of_the_root_s_manifest(tmp_path):
    """Of the manifest a key reads every fetched package's entry, whole but `inherited`, and every other field but the
    root's `name`, its `packagesDir` and the entries of the packages required by path, whose directories `workspace`
    reads: a field this reading has never met moves every key, and so does a fetched package added to the root, also
    the key of a module of `d`, which has no manifest to say what it fetches."""
    root = _workspace(tmp_path, {"r/lake-manifest.json": json.dumps({
        "version": "1.2.0", "name": "r", "packagesDir": ".lake/packages", "packages": [
            {"type": "path", "name": "d", "dir": "../d", "inherited": False, "configFile": "lakefile.toml"},
            {"type": "git", "name": "mathlib", "rev": "abc", "url": "https://example.org/mathlib", "inherited": False,
             "configFile": "lakefile.lean", "subDir": None, "scope": "", "inputRev": "v1", "manifestFile": "lake-manifest.json"},
            {"type": "git", "name": "batteries", "rev": "def", "url": "https://example.org/batteries", "inherited": True}]})})
    keys = _built(MK.keys(root))

    def changed(change) -> set[str]:
        return _changed(root, keys, "r/lake-manifest.json", _manifest(root, change))

    for same in (lambda d: d.update(name="another"), lambda d: d.update(packagesDir="elsewhere"),
                 lambda d: d["packages"][1].update(inherited=True), lambda d: d["packages"].reverse(),
                 lambda d: d["packages"][0].update(dir="../r/../d", inherited=True, manifestFile="m.json")):
        assert changed(same) == set()
    for field, value in (("rev", "abd"), ("url", "https://example.org/other"), ("configFile", "lakefile.toml"), ("subDir", "sub"),
                         ("scope", "s"), ("inputRev", "v2"), ("manifestFile", "other.json"), ("name", "Mathlib"),
                         ("somethingNobodyThoughtOf", 1)):
        assert changed(lambda d: d["packages"][1].update({field: value})) == set(keys), field
    assert changed(lambda d: d.update(version="1.3.0")) == set(keys)
    assert changed(lambda d: d.update(somethingNobodyThoughtOf=1)) == set(keys)
    assert changed(lambda d: d["packages"].append({"type": "git", "name": "extra", "rev": "0"})) == set(keys)
    assert changed(lambda d: d["packages"].pop()) == set(keys)
    assert changed(lambda d: d["packages"][2].update(type="another kind")) == set(keys)


def _fetching(tmp: Path) -> tuple[Path, dict]:
    """`_workspace`, where `d` says what it fetches (`mathlib`) and `r` fetches `arklib` too; and `mathlib`'s entry."""
    mathlib = {"type": "git", "name": "mathlib", "rev": "abc"}
    root = _workspace(tmp, {
        "r/lake-manifest.json": json.dumps({"name": "r", "packages": [
            {"type": "path", "name": "d", "dir": "../d"}, mathlib, {"type": "git", "name": "arklib", "rev": "1"}]}),
        "d/lake-manifest.json": json.dumps({"name": "d", "packages": [mathlib | {"inherited": False}]}),
        "d/lean-toolchain": TOOLCHAIN})
    return root, mathlib


def test_a_module_has_one_key_under_every_root_that_pins_what_its_package_fetches_alike(tmp_path):
    """A key's paths start at the module's own package, and its env is the root's toolchain and the root's pins of the
    packages the module's own package fetches: a path dependency's module has one key under the dependency itself and
    under each root that pins those alike, whatever else the root fetches, another under a root that pins one of them
    otherwise, and none under a root that doesn't pin one. The root's own modules read every pin of the root."""
    root, mathlib = _fetching(tmp_path)
    _write(tmp_path, {"q/lakefile.toml": 'name = "q"\ndefaultTargets = ["Q"]\n\n[[lean_lib]]\nname = "Q"\n',
                      "q/lake-manifest.json": json.dumps({"name": "q", "packagesDir": "p", "packages": [
                          mathlib | {"inherited": True}, {"type": "path", "name": "d", "dir": "../r/../d"}]}),
                      "q/lean-toolchain": TOOLCHAIN, "q/Q.lean": "import D.X\n"})

    def under(r: str, change=None) -> dict[str, str]:
        if change is not None:
            (tmp_path / r / "lake-manifest.json").write_text(_manifest(tmp_path / r, change))
        return _built(MK.keys(tmp_path / r, ["D.X"]))

    q, d = under("q"), under("d")["D.X"]
    assert under("r")["D.X"] == d == q["D.X"], "r fetches arklib, which d doesn't"
    more = under("q", lambda doc: doc["packages"].append({"type": "git", "name": "doc-gen4", "rev": "2"}))
    assert more["D.X"] == d and more["Q"] != q["Q"]
    assert under("q", lambda doc: doc["packages"][0].update(rev="abd"))["D.X"] != d
    assert under("q", lambda doc: doc["packages"].pop(0)) == {}, "q pins no mathlib: its Q imports D.X, which has no key"
    assert "pins no mathlib, which" in MK.env_parts(tmp_path / "q", tmp_path / "d")
    assert under("d", lambda doc: doc["packages"][0].update(rev="abd"))["D.X"] != d, "d as its own root"
    assert under("r")["D.X"] == d, "what d's manifest pins is not in a key under r, only which packages it names"


def test_the_keys_of_a_root_s_own_modules_name_every_pin_of_the_root(tmp_path):
    """The env of a root's own modules is the one every module of its workspace had before a path dependency's read
    less: the root's toolchain and all its manifest pins, whatever the packages it requires by path fetch. So no entry
    under a root's own module's key is left behind."""
    root, mathlib = _fetching(tmp_path)
    keyed = MK.keys(root)
    whole = {"lean-toolchain": C.digest(TOOLCHAIN.encode()),
             "fetched packages": _canon({"manifest": {}, "fetched": [{"type": "git", "name": "arklib", "rev": "1"}, mathlib]})}
    own = {m: k for m, k in keyed.items() if k.package.name == "r"}
    assert set(own) == {"R", "R.A"} and MK.env_parts(root) == MK.env_parts(root, root) == whole
    assert {(k.env, tuple(k.parts.items())) for k in own.values()} == {(MK.workspace(root)[1], tuple(whole.items()))}
    assert MK.workspace(root)[1] == C.env_digest(sorted(whole), whole)
    less = whole | {"fetched packages": _canon({"manifest": {}, "fetched": [mathlib]})}
    assert keyed["D.X"].parts == less and keyed["D.X"].env == C.env_digest(sorted(less), less)
    assert MK.fetches(tmp_path / "d") == {"mathlib"} and MK.fetches(root) == {"mathlib", "arklib"}
    for unread in ("{", "[]", '{"packages": 1}', '{"packages": ["mathlib"]}', '{"packages": [{"type": "git"}]}',
                   '{"packages": [{"type": "git", "name": ["mathlib"]}]}', None):
        if unread is None:
            (tmp_path / "d" / "lake-manifest.json").unlink()
        else:
            (tmp_path / "d" / "lake-manifest.json").write_text(unread)
        again = MK.keys(root)
        assert {m: (k.task, k.env) for m, k in again.items() if m != "D.X"} == {m: (k.task, k.env) for m, k in own.items()}
        assert isinstance(MK.fetches(tmp_path / "d"), str) and again["D.X"].parts == whole, \
            f"a package whose manifest doesn't say what it fetches ({unread}): every pin of the root"


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


def test_a_workspace_lake_could_configure_otherwise_has_no_keys(tmp_path):
    """No keys where Lake would read a package from somewhere this reading doesn't: a `lakefile.lean` beside the
    `lakefile.toml` (Lake reads the `lakefile.lean`), a path entry configured by another file or without a directory,
    entries in the root's `.lake` that Lake takes over the manifest's, or two packages of one name."""
    root = _workspace(tmp_path / "a", {"d/lakefile.lean": ""})
    assert MK.keys(root) == {} and "has a lakefile.lean beside its lakefile.toml" in MK.workspace(root)
    for i, (change, why) in enumerate([
            (lambda d: d["packages"][0].update(configFile="other.toml"), "d is configured by other.toml"),
            (lambda d: d["packages"][0].pop("dir"), "a package required by path has no directory"),
            (lambda d: d.update(packages={}), "its packages are not a list of entries")]):
        root = _workspace(tmp_path / f"b{i}")
        (root / "lake-manifest.json").write_text(_manifest(root, change))
        assert MK.keys(root) == {} and why in MK.workspace(root), why
    root = _workspace(tmp_path / "c", {"r/.lake/package-overrides.json": "[]"})
    assert MK.keys(root) == {} and "overrides the manifest's entries" in MK.workspace(root)
    root = _workspace(tmp_path / "d", {"d/lakefile.toml": 'name = "r"\n\n[[lean_lib]]\nname = "D"\n'})
    assert MK.keys(root) == {} and "two packages are named r" in MK.workspace(root)


def test_a_workspace_whose_compiles_load_native_code_has_no_keys(tmp_path):
    """A package or a library that precompiles its modules, or names plugins or dynlibs, makes a compile load native
    code, and a key reads none of its bytes. Such a package has no keys, and neither has a root that requires it by
    path, whose modules import its own. The defaults (`false`, no plugin) load nothing and change nothing."""
    lib = '\n[[lean_lib]]\nname = "D"\n'
    for i, (lakefile, why) in enumerate([
            (f'name = "d"\nprecompileModules = true\n{lib}', "precompileModules"),
            (f'name = "d"\n{lib}precompileModules = true\n', "precompileModules"),
            (f'name = "d"\n{lib}plugins = ["d/Plug:shared"]\n', "plugins"),
            (f'name = "d"\n{lib}dynlibs = ["d/Ffi:shared"]\n', "dynlibs"),
            (f'name = "d"\nmoreLeanArgs = ["--load-dynlib=libffi.so"]\n{lib}', "--load-dynlib=libffi.so"),
            (f'name = "d"\n{lib}weakLeanArgs = ["--plugin=p.so"]\n', "--plugin=p.so")]):
        root = _workspace(tmp_path / f"n{i}", {"d/lakefile.toml": lakefile})
        assert MK.keys(root) == {}, lakefile
        assert MK.workspace(root).endswith(f"{why} makes a compile load native code, which no key reads")
    root = _workspace(tmp_path / "ok", {"d/lakefile.toml": f'name = "d"\nprecompileModules = false\n{lib}plugins = []\n'})
    assert set(MK.keys(root)) == set(MK.keys(_workspace(tmp_path / "plain")))


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
    assert {"r/models.json", "r/data/a/b.txt"} <= set(k["R.A"].task["inputs"]) and k["D.X"].task["inputs"] == \
        ["d/D/X.lean", "d/lakefile.toml"]
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
    right = {"D.X": lib("d", "D.X"), "R.A": lib("r", "R.A"), "Mathlib": mathlib}
    assert MK.indexable("R", keyed, pkgs, owners) == "no setup.json"
    _setup(root, "R", "r", right)
    assert MK.indexable("R", keyed, pkgs, owners) is None
    _setup(root, "R", "r", right | {"R.B": lib("r", "R.B")})
    assert "Lake imports ['R.B']" in MK.indexable("R", keyed, pkgs, owners)
    _setup(root, "R", "r", {i: p for i, p in right.items() if i != "D.X"})
    assert "and not ['D.X']" in MK.indexable("R", keyed, pkgs, owners)
    _setup(root, "R", "r", right | {"D.X": lib("r", "D.X")})
    assert MK.indexable("R", keyed, pkgs, owners) == "D.X is built in r, not d"
    assert owners["r", str(lib("r", "D.X").parent)] == "r" and owners["r", str(mathlib.parent)] is None, "one lookup a directory"
    _setup(root, "R", "d", right)
    assert "setup.json names 'R' of 'd'" == MK.indexable("R", keyed, pkgs, owners)


def test_a_build_that_read_its_tree_at_a_fixed_path_is_indexed_as_one_at_the_trees_own(tmp_path):
    """A sandboxed build reads its tree at a fixed path (`sandbox.sh --at`), so every path Lake wrote into a `setup.json`
    is under it. Told where the build read the tree (``at``), `indexable` reads each as the tree's and gives the verdicts
    it gives a build at the tree's own path; untold, it finds no package's build there and refuses the module."""
    root = _workspace(tmp_path)
    pkgs = MK.workspace(root)[0]
    keyed, owners = MK.keys(root), {}
    fixed = Path("/var/tmp/verity")
    at = (str(fixed), str(tmp_path))

    def lib(p, m):
        return fixed / p / ".lake" / "build" / "lib" / "lean" / MK.path_of(m, ".olean")

    mathlib = fixed / "r" / ".lake" / "packages" / "mathlib" / ".lake" / "build" / "lib" / "lean" / "Mathlib.olean"
    right = {"D.X": lib("d", "D.X"), "R.A": lib("r", "R.A"), "Mathlib": mathlib}
    _setup(root, "R", "r", right)
    assert MK.indexable("R", keyed, pkgs, owners, at) is None
    assert owners["r", str(lib("d", "D.X").parent)] == "d" and owners["r", str(mathlib.parent)] is None
    assert MK.indexable("R", keyed, pkgs) == f"{lib('d', 'D.X').parent} is in no root's fetched packages"
    _setup(root, "R", "r", right | {"D.X": lib("r", "D.X")})
    assert MK.indexable("R", keyed, pkgs, owners, at) == "D.X is built in r, not d"
    _setup(root, "R", "r", right | {"Mathlib": fixed / "p/.lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean"})
    assert MK.indexable("R", keyed, pkgs, {}, at) == f"{tmp_path / 'p'} has no keys, and a compile read mathlib under {tmp_path / 'p'}"
    elsewhere = {"D.X": tmp_path / "d/.lake/build/lib/lean/D/X.olean", "R.A": lib("r", "R.A"), "Mathlib": mathlib}
    _setup(root, "R", "r", elsewhere)
    assert MK.indexable("R", keyed, pkgs, {}, at, own := set()) is None, "a path outside the fixed one is read as it is"
    assert own == {"R"}, "and counted, for `put` to log"
    _setup(root, "R", "r", right)
    assert MK.indexable("R", keyed, pkgs, {}, at, own := set()) is None and own == set()
    _setup(root, "R", "r", elsewhere)
    MK.indexable("R", keyed, pkgs, {}, None, own)
    assert own == set(), "a build that bound nothing is not counted"


def test_a_build_is_indexed_only_where_it_read_the_fetched_packages_as_its_root_pins_them(tmp_path):
    """A key names the fetched packages by its root's pins. A package required by path is compiled again by a build of
    each root that requires it, over that root's packages: `r`'s modules by a build of `p`, which requires `r`. That
    compile is one of what `r`'s key names only when `p` has `r`'s toolchain and pins every package `r` fetches as `r`
    does; Lake lists no module of the toolchain, so a module read anywhere else is refused."""
    root = _workspace(tmp_path)
    pkgs = MK.workspace(root)[0]
    keyed = MK.keys(root)
    mine = {"D.X": tmp_path / "d/.lake/build/lib/lean/D/X.olean", "R.A": root / ".lake/build/lib/lean/R/A.olean"}

    def other(entries: list[dict], toolchain: str = TOOLCHAIN, package: str = "mathlib") -> str | None:
        _write(tmp_path, {"p/lake-manifest.json": json.dumps({"name": "p", "packagesDir": "elsewhere", "packages": entries}),
                          "p/lean-toolchain": toolchain})
        _setup(root, "R", "r", mine | {"Mathlib": tmp_path / "p/.lake/packages" / package / ".lake/build/lib/lean/Mathlib.olean"})
        return MK.indexable("R", keyed, pkgs)

    same = {"type": "git", "name": "mathlib", "rev": "abc"}
    more = [{"type": "path", "name": "r", "dir": "../r"}, {"type": "git", "name": "«doc-gen4»", "rev": "1"}]
    assert other([same | {"inherited": True}, *more]) is None, "whether a root requires a package itself is in no key"
    assert "pins mathlib otherwise than" in other([same | {"rev": "abd"}, *more])
    assert "pins mathlib otherwise than" in other(more)
    assert "has another toolchain than" in other([same, *more], "leanprover/lean4:v4.35.0\n")
    assert "does not fetch doc-gen4, which a compile read under" in other([same, *more], package="doc-gen4")
    (tmp_path / "p/lean-toolchain").unlink()
    assert "p has no keys" in MK.indexable("R", keyed, pkgs)
    _setup(root, "R", "r", mine | {"Mathlib": root / ".lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean",
                                    "Init": Path("/elan/lib/lean/Init.olean")})
    assert MK.indexable("R", keyed, pkgs) == "/elan/lib/lean is in no root's fetched packages"


def test_a_build_of_a_dependency_is_indexed_where_it_read_the_packages_the_dependency_fetches(tmp_path):
    """A module of `d`, which fetches `mathlib`, has one key under `d` and under `r`, which fetches `arklib` too
    (`_fetching`). `d`'s own build compiles it over `d`'s packages, and a build of `r` or of another root requiring `d`
    over that root's: each is a compile of what the key under `r` names where the root it ran under has `r`'s
    toolchain and pins `mathlib` as `r` does, whatever else `r` or that root fetches."""
    root, mathlib = _fetching(tmp_path)
    pkgs, keyed = MK.workspace(root)[0], MK.keys(root)
    _write(tmp_path, {"d/D/X.lean": "import Mathlib.Logic.Basic\n"})

    def read(under: str, entries: list[dict] | None = None, toolchain: str = TOOLCHAIN) -> str | None:
        if entries is not None:
            _write(tmp_path, {f"{under}/lake-manifest.json": json.dumps({"name": under, "packages": entries}),
                              f"{under}/lean-toolchain": toolchain})
        _setup(tmp_path / "d", "D.X", "d", {
            "Mathlib.Logic.Basic": tmp_path / under / ".lake/packages/mathlib/.lake/build/lib/lean/Mathlib/Logic/Basic.olean"})
        return MK.indexable("D.X", keyed, pkgs)

    assert read("d") is None, "d's own build: it fetches no arklib, and D.X's key names none"
    assert read("r") is None
    assert read("p", [mathlib, {"type": "path", "name": "d", "dir": "../d"}, {"type": "git", "name": "other", "rev": "3"}]) is None
    assert "pins mathlib otherwise than" in read("p", [mathlib | {"rev": "abd"}])
    assert "has another toolchain than" in read("p", [mathlib], "leanprover/lean4:v4.35.0\n")
    assert "pins mathlib otherwise than" in read("d", [mathlib | {"rev": "abd"}]), "d's tree at another pin than r's"
    assert MK.keys(root)["D.X"].env == keyed["D.X"].env, "which r's key doesn't name: it reads r's pin"


def test_a_compile_that_read_a_package_its_own_package_does_not_fetch_is_never_indexed(tmp_path):
    """Under `r`, which fetches `arklib`, Lake would resolve an import of `d`'s module in `arklib`, which `d` doesn't
    fetch. The module's key names `mathlib`'s pin alone, so no entry stands under it for a compile that read `arklib`,
    wherever: it is refused, and one of `r`'s own modules that read the same directory is indexed."""
    root, _ = _fetching(tmp_path)
    _write(tmp_path, {"d/D/X.lean": "import ArkLib.Basic\n", "r/R/B.lean": "import ArkLib.Basic\n"})
    pkgs, keyed, owners = MK.workspace(root)[0], MK.keys(root, ["R.B"]), {}
    arklib = {"ArkLib.Basic": root / ".lake/packages/arklib/.lake/build/lib/lean/ArkLib/Basic.olean"}
    assert "D.X" in keyed and keyed["D.X"].parts != keyed["R.B"].parts
    _setup(root, "R.B", "r", arklib)
    assert MK.indexable("R.B", keyed, pkgs, owners) is None
    _setup(tmp_path / "d", "D.X", "d", arklib)
    why = MK.indexable("D.X", keyed, pkgs, owners)
    assert why == f"{tmp_path / 'd'} does not fetch arklib, which a compile read under {root}", "after R.B, over one owners"
    assert MK.indexable("R.B", keyed, pkgs, owners) is None


def test_a_directory_s_refusal_is_never_read_half_written_by_another_thread(tmp_path, monkeypatch):
    """`put` asks `indexable` of a build's modules on several threads over one ``owners``. While one thread reads the two
    manifests that refuse a directory, another asking of the same directory must not find it there as one the root
    pins: it would index a module compiled over another root's pins. The first thread waits inside `pinned_as` until
    the second has looked the directory up, and both refuse."""
    root = _workspace(tmp_path)
    pkgs = MK.workspace(root)[0]
    keyed = MK.keys(root)
    _write(tmp_path, {"p/lake-manifest.json": json.dumps({"name": "p", "packages": [
        {"type": "git", "name": "mathlib", "rev": "another"}]}), "p/lean-toolchain": TOOLCHAIN})
    _setup(root, "R", "r", {"D.X": tmp_path / "d/.lake/build/lib/lean/D/X.olean", "R.A": root / ".lake/build/lib/lean/R/A.olean",
                            "Mathlib": tmp_path / "p/.lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean"})
    inside, looked = threading.Event(), threading.Event()
    first = threading.current_thread()
    pinned_as = MK.pinned_as

    class Owners(dict):
        def __contains__(self, d):
            if threading.current_thread() is not first and "packages" in d[1]:
                looked.set()
            return super().__contains__(d)

    def slow(root, of, here):
        if threading.current_thread() is first:
            inside.set()
            looked.wait()
        return pinned_as(root, of, here)

    monkeypatch.setattr(MK, "pinned_as", slow)
    owners, got = Owners(), []

    def second():
        inside.wait()
        got.append(MK.indexable("R", keyed, pkgs, owners))

    other = threading.Thread(target=second, daemon=True)
    other.start()
    got.append(MK.indexable("R", keyed, pkgs, owners))
    other.join()
    assert [why is not None and "pins mathlib otherwise than" in why for why in got] == [True, True], got


def test_a_module_file_is_indexed_where_lake_handed_its_compile_the_modules_lean_loads(tmp_path):
    """Lake hands a `module` file's compile the modules Lean loads of its imports, which are the ones its key reads:
    not what one of its imports imports privately. `R.A` imports `R.Q` privately, so of `R.A`'s importers only one that
    needs its code reaches `R.Q`. A build is indexed where the `setup.json` lists exactly those; one that listed the
    whole closure, or left a loaded module out, read its header another way than the key did."""
    root = _workspace(tmp_path, LEVELS | {"r/R/U.lean": "module\n\npublic import R.A\n"})
    pkgs = MK.workspace(root)[0]
    keyed = MK.keys(root, ["R.U", "R.L"])
    lib = root / ".lake" / "build" / "lib" / "lean"
    assert keyed["R.U"].closure == {"R.A", "R.P", "R.Q"} and set(MK.loads("R.U", keyed)) == {"R.A", "R.P"}
    _setup(root, "R.U", "r", {m: lib / MK.path_of(m, ".olean") for m in ("R.A", "R.P")})
    assert MK.indexable("R.U", keyed, pkgs) is None
    _setup(root, "R.U", "r", {m: lib / MK.path_of(m, ".olean") for m in ("R.A", "R.P", "R.Q")})
    assert "Lake imports ['R.Q'] and not []" in MK.indexable("R.U", keyed, pkgs)
    _setup(root, "R.U", "r", {"R.A": lib / "R" / "A.olean"})
    assert "Lake imports [] and not ['R.P']" in MK.indexable("R.U", keyed, pkgs)
    loaded = set(MK.loads("R", keyed))
    assert loaded == {"R.A", "R.P", "R.M", "R.Q", "R.B"}, "`R` needs `R.M`'s code, so `R.M`'s private import too"
    _setup(root, "R", "r", {m: lib / MK.path_of(m, ".olean") for m in loaded})
    assert MK.indexable("R", keyed, pkgs) is None
    _setup(root, "R.L", "r", {m: lib / MK.path_of(m, ".olean") for m in ("R.A", "R.P")})
    assert "and not ['R.Q']" in MK.indexable("R.L", keyed, pkgs), "a legacy file loads everything it imports, transitively"


#: `R.F` imports `R.D` by a `public import` and by a private `meta import`, which `keys` folds into one `public meta
#: import`; `R.D` imports `R.S` privately, and `R.T` imports `R.F`
FOLD = {"r/R/T.lean": "module\n\npublic import R.F\n", "r/R/F.lean": "module\n\npublic import R.D\nmeta import R.D\n",
        "r/R/D.lean": "module\n\nimport R.S\n", "r/R/S.lean": "module\n"}


def test_a_build_lake_handed_less_than_its_key_reads_is_indexed_where_the_rest_is_read_for_its_code_alone(tmp_path):
    """`R.T`'s key reads `R.S`'s code through the fold, where Lean loads `R.D`'s code only for `R.F`'s own compile, so Lake
    hands `R.T` no `R.S`. Refused for that, `R.T` would be compiled in every build: a key that reads more than its compile
    is indexed. One that left out a module the key reads more than the code of, or handed one it doesn't read, is not."""
    root = _workspace(tmp_path, FOLD)
    pkgs, keyed = MK.workspace(root)[0], MK.keys(root, ["R.T"])
    lib = root / ".lake" / "build" / "lib" / "lean"

    def handed(*ms: str) -> str | None:
        _setup(root, "R.T", "r", {m: lib / MK.path_of(m, ".olean") for m in ms})
        return MK.indexable("R.T", keyed, pkgs)

    assert MK.loads("R.T", keyed) == {"R.F": MK.PUBLIC, "R.D": MK.PUBLIC | MK.CODE, "R.S": MK.CODE}
    assert handed("R.F", "R.D") is None and handed("R.F", "R.D", "R.S") is None
    assert handed("R.F", "R.S") == "Lake imports [] and not ['R.D'] of the workspace"
    assert handed("R.F", "R.D", "R.A") == "Lake imports ['R.A'] and not [] of the workspace"


def _entry(store: Path, key: str, scope: str, mod: str, h: str, files: dict[str, bytes], inputs: dict | None = None) -> dict:
    """``mod``'s index entry under ``key``, with its artifacts, as `LakeArtifacts.put` writes it: with what its key read
    (``inputs``), it is also the entry the module was last indexed under."""
    (store / "artifacts").mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        (store / "artifacts" / n).write_bytes(b)
    e = {"schema": MK.SCHEMA, "module": mod, "scope": scope, "hash": h,
         "outputs": {"o": [n for n in files if n.endswith(".olean")], "i": next(n for n in files if n.endswith(".ilean"))},
         "digests": {n: C.digest(b) for n, b in files.items()}, "inputs": inputs}
    MK.entry_path(store, key).parent.mkdir(exist_ok=True)
    MK.entry_path(store, key).write_text(json.dumps(e))
    if inputs is not None:
        MK.last_path(store, scope, mod).parent.mkdir(parents=True, exist_ok=True)
        MK.last_path(store, scope, mod).write_text(json.dumps({"key": key}))
    return e


def _stock(store: Path, keyed: dict, entries: dict[str, tuple[str, dict[str, bytes]]]) -> dict[str, str]:
    """Index ``entries`` ({module: (Lake's input hash, {artifact: its bytes})}, in import order), each under the key the
    digests the entries before it name give it, as `LakeArtifacts.put` indexes a build holding those artifacts; {module: its
    key}."""
    named: dict[tuple[str, str], str] = {}
    out = {}
    for m, (h, files) in entries.items():
        out[m] = MK.key(m, keyed, named)
        e = _entry(store, out[m], keyed[m].package.name, m, h, files, MK.record(m, keyed, named))
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
    for f in [*(store / "artifacts").iterdir(), *(store / MK.INDEX).iterdir()]:
        os.utime(f, ns=(10**18, 10**18))
    staged, n = MK.stage(store, root, [], tmp_path / "cache")
    assert n == 3 and staged == {("d", "1" * 16): "D.X", ("r", "2" * 16): "R.A", ("r", "3" * 16): "R"}
    assert {f.stat().st_mtime_ns for f in (store / "artifacts").iterdir()} == {10**18}, \
        "a tree that restored an artifact holds this file: a newer time on it would be an output newer than its `.hash`"
    assert all(MK.entry_path(store, h).stat().st_mtime_ns > 10**18 for h in k.values()), "its use is its entry's"
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


def test_an_entry_records_what_its_key_read_and_a_miss_says_which_inputs_differ(tmp_path):
    """Each entry records its key's inputs (`record`): its vars, the env's parts, each file's digest and what it read of
    each module it loads. A build that doesn't restore a module says why against the entry the module was last indexed
    under: every input that differs, in one phrase, or that it reads a module that isn't restored. The first alone was
    read as the cause of a miss whose source had changed too."""
    files = {"r/lakefile.toml": LIBS, "r/R/A.lean": "import D.X\n", "r/S.lean": "", "r/T.lean": ""}
    root = _workspace(tmp_path / "ws", files)
    store = tmp_path / "store"
    keyed = MK.keys(root, ["T"])
    stock = {"D.X": ("1" * 16, X), "R.A": ("2" * 16, A), "R": ("3" * 16, R),
             "T": ("4" * 16, {"7" * 16 + ".olean": b"T", "8" * 16 + ".ilean": b"t"})}
    k = _stock(store, keyed, stock)
    e = json.loads(MK.entry_path(store, k["R"]).read_text())
    x, a = (C.digest(" ".join([C.digest(b)]).encode())[7:19] for b in (b"X", b"A"))
    assert e["inputs"] == {"vars": {"module": "R", "package": "r", "library": "R"}, "env": keyed["R"].parts,
                           "files": {f: keyed["R"].files[f] for f in ("r/R.lean", "r/lakefile.toml")},
                           "reads": {"D.X": x, "R.A": a}}
    assert MK.last(store, "r", "R") == e["inputs"] and MK.last(store, "r", "R.B") is None

    caches = iter(range(99))

    def missed(extra=("T",)) -> tuple[dict, dict]:
        why: dict = {}
        MK.stage(store, root, extra, tmp_path / "cache" / str(next(caches)), report=why)
        return why["missed"], why["first"]

    assert missed() == ({}, {})
    assert missed(["T", "R.B"]) == ({"never indexed here": 1}, {"never indexed here": "R.B"})
    (root / "R" / "A.lean").write_text("import D.X\n-- edited\n")
    assert missed() == ({"its source": 1, "reads a module not restored": 1},
                        {"its source": "R.A", "reads a module not restored": "R"})
    (root / "R" / "A.lean").write_text("import D.X\n")
    (root / "lakefile.toml").write_text(LIBS.replace('name = "r"', 'name = "r"\nbuildType = "debug"'))
    assert missed()[0] == {"its lakefile": 2, "reads a module not restored": 1}
    (root / "lakefile.toml").write_text(LIBS.replace('roots = ["R", "T"]', 'roots = ["R"]').replace('roots = ["S"]', 'roots = ["S", "T"]'))
    assert missed() == ({"its library": 1}, {"its library": "T"})
    (root / "lakefile.toml").write_text(LIBS)
    (root / "lean-toolchain").write_text("leanprover/lean4:v4.35.0\n")
    assert missed()[0] == {"the toolchain": 2, "reads a module not restored": 2}
    (root / "lean-toolchain").write_text(TOOLCHAIN)
    (root / "lake-manifest.json").write_text(_manifest(root, lambda d: d["packages"][1].update(rev="abd")))
    assert missed()[0] == {"the fetched packages": 2, "reads a module not restored": 2}
    (root / "T.lean").write_text("-- edited\n")
    assert missed() == ({"the fetched packages": 1, "the fetched packages and its source": 1, "reads a module not restored": 2},
                        {"the fetched packages": "D.X", "the fetched packages and its source": "T",
                         "reads a module not restored": "R.A"}), "T's pins and its source both differ, and the line names both"
    (root / "T.lean").write_text("")
    (root / "lake-manifest.json").write_text(_manifest(root, lambda d: d["packages"][1].update(rev="abc")))
    _stock(store, MK.keys(root, ["T"]), {"D.X": ("5" * 16, {"9" * 16 + ".olean": b"X'", "b" * 16 + ".ilean": b"x"})})
    assert missed() == ({"an output it reads": 1, "reads a module not restored": 1},
                        {"an output it reads": "R.A", "reads a module not restored": "R"}), "D.X's olean is another"
    _stock(store, MK.keys(root, ["T"]), {"D.X": ("1" * 16, X)})
    MK.entry_path(store, k["R"]).unlink()
    assert missed() == ({"never indexed here": 1}, {"never indexed here": "R"}), "the entry it was last indexed under is gone"
    _stock(store, MK.keys(root, ["T"]), stock)
    (store / "artifacts" / ("c" * 16 + ".olean")).write_bytes(b"not what R.A's entry names")
    assert missed()[0] == {"an artifact of its entry differs": 1, "reads a module not restored": 1}
    was = e["inputs"]
    assert MK.differs(was, was) == MK.GONE and MK.differs(was, None) == MK.NEVER
    other = {"vars": was["vars"] | {"library": "S"}, "env": {p: "0" for p in was["env"]},
             "files": {"r/R.lean": "0", "r/lakefile.toml": "0", "r/notes.txt": "0"}, "reads": {}}
    assert MK.differs(was, other) == ("its library, the toolchain, the fetched packages, its source, its lakefile, "
                                      "an input file and an output it reads")
    assert MK.differs(was, was | {"reads": {}}) == "an output it reads"
    assert MK.differs(was, was | {"files": was["files"] | {"r/notes.txt": "0"}, "reads": {}}) == "an input file and an output it reads"


def _held(keyed: dict, mod: str, files: dict[str, bytes], h: str = "0" * 16) -> Path:
    """``mod`` as its package's build holds it once Lake compiled or restored it: each artifact of ``files`` where Lake
    leaves it, and the trace naming them; the trace."""
    build = keyed[mod].package.build()
    for n, b in files.items():
        f = MK.file_of(build, mod, n.split(".", 1)[1])
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b)
    trace = build / "lib" / "lean" / MK.path_of(mod, ".trace")
    trace.write_text(json.dumps({"depHash": h, "outputs": {
        "o": [n for n in files if n.endswith(".olean")], "i": next(n for n in files if n.endswith(".ilean"))}}))
    return trace


B = {"1" * 16 + ".olean": b"B", "2" * 16 + ".ilean": b"b"}


def test_a_build_takes_the_trace_of_each_module_it_holds_in_a_version_the_cache_does_not(tmp_path):
    """Lake keeps a module whose trace has its input hash over what was staged for it, and only what a run compiles is
    indexed. So a module an earlier build left in the tree in a version the cache lacks would be restored nowhere else,
    and one the cache holds another version of would sit under modules restored against that other version. Each loses
    its trace before the build: Lake then restores the one and compiles the other, which the build's `put` indexes.
    A module with no key is left as it is, and without ``settled`` so is every module."""
    root = _workspace(tmp_path / "ws")
    store = tmp_path / "store"
    keyed = MK.keys(root, ["R.B"])
    _stock(store, keyed, {"D.X": ("1" * 16, X), "R.A": ("2" * 16, A), "R": ("3" * 16, R)})
    traces = {"D.X": _held(keyed, "D.X", X), "R": _held(keyed, "R", R),
              "R.A": _held(keyed, "R.A", {"9" * 16 + ".olean": b"another A", "d" * 16 + ".ilean": b"a"})}
    other = root / ".lake" / "build" / "lib" / "lean" / "R" / "NoKey.trace"
    other.write_text(json.dumps({"depHash": "7" * 16, "outputs": {"o": ["8" * 16 + ".olean"]}}))
    caches = iter(range(99))

    def stage(settled: bool = True) -> dict:
        why: dict = {}
        MK.stage(store, root, ["R.B"], tmp_path / "cache" / str(next(caches)), report=why, settled=settled)
        return why

    assert "took" not in stage(False) and all(t.exists() for t in traces.values())
    why = stage()
    assert (why["took"], why["left"]) == ({MK.REPLACED: 1}, 0), "R.B has no trace: nothing to take"
    assert not traces["R.A"].exists() and traces["D.X"].exists() and traces["R"].exists() and other.exists()
    traces["R.B"] = _held(keyed, "R.B", B)
    assert stage()["took"] == {MK.ABSENT: 1} and not traces["R.B"].exists(), "R.B: its key is not in the index"
    assert stage()["took"] == {} and traces["D.X"].exists() and traces["R"].exists() and other.exists()
    traces["R"].write_text('{"depHash": "3333')
    assert stage()["took"] == {} and not traces["R"].exists(), "a trace that isn't JSON goes: Lake takes it for none"


def test_a_build_takes_no_trace_in_the_build_of_a_package_it_requires_by_path(tmp_path):
    """`d`'s build is `d`'s: an audit of `d` seals its files, and other packages' builds and tests read them while `r` is
    staged and built. A trace taken there would have `r`'s Lake compile or restore `D.X` over them. So staging `r`
    leaves that build whole: on a cache that holds nothing for these keys, where `D.X`'s trace names a version the cache
    lacks; on one that holds another version of it, where the trace names other artifacts than its entry; and with a
    trace that isn't JSON. `r`'s own modules are settled as before: on the cold cache each is keyed on what the tree
    holds of `D.X` and compiled."""
    root = _workspace(tmp_path / "ws")
    store = tmp_path / "store"
    keyed = MK.keys(root, ["R.B"])
    traces = {m: _held(keyed, m, files) for m, files in (("D.X", X), ("R.A", A), ("R", R), ("R.B", B))}
    build = keyed["D.X"].package.build()
    caches = iter(range(99))

    def held() -> dict:
        return {f: (f.stat().st_ino, f.stat().st_mtime_ns, f.read_bytes()) for f in build.rglob("*") if f.is_file()}

    def took() -> dict:
        why: dict = {}
        MK.stage(store, root, ["R.B"], tmp_path / "cache" / str(next(caches)), report=why, settled=True)
        return why["took"]

    was = held()
    assert took() == {MK.ABSENT: 3} and held() == was, "R.A, R and R.B lose their traces; D.X, which the cache lacks too, keeps its"
    assert not any(traces[m].exists() for m in ("R.A", "R", "R.B"))
    _stock(store, keyed, {"D.X": ("5" * 16, {"9" * 16 + ".olean": b"X'", "b" * 16 + ".ilean": b"x"})})
    assert took() == {} and held() == was, "the cache holds another version of D.X than the tree"
    traces["D.X"].write_text('{"depHash": "1111')
    was = held()
    assert took() == {} and held() == was, "a trace of d's that isn't JSON is d's build's to remove"


def test_a_build_settles_the_build_of_a_path_dependency_its_run_does_not_audit(tmp_path):
    """`d` is in the tree only as `r`'s path dependency, and its build there is a copy of a kept build that no build of
    `d`'s own will settle. Left as it is, `D.X`, which the cache lacks, is never indexed, and every module of `r` that
    loads it is recorded as one the cache could not index, in each build that takes such a copy. Named to the stage
    step, `d`'s build is settled with `r`'s: `D.X` loses its trace where the cache lacks it or holds another version,
    and keeps it where the cache holds what the tree does. The first build of a run that reaches a module there settles
    it; a later one of the same run leaves it, since an earlier build has compiled it and the cache holds it only when
    the run ends."""
    root = _workspace(tmp_path / "ws")
    d, store = (tmp_path / "ws" / "d").resolve(), tmp_path / "store"
    keyed = MK.keys(root, ["R.B"])
    held = {m: files for m, files in (("D.X", X), ("R.A", A), ("R", R), ("R.B", B))}
    traces = {m: _held(keyed, m, files) for m, files in held.items()}
    caches = iter(range(99))

    def took(seen: set, also=(d,)) -> dict:
        why: dict = {}
        MK.stage(store, root, ["R.B"], tmp_path / "cache" / str(next(caches)), report=why, settled=True, also=also, seen=seen)
        return why["took"]

    run: set = set()
    assert took(run) == {MK.ABSENT: 4} and not any(t.exists() for t in traces.values()), "D.X goes with r's three"
    assert run == {(d, "D.X")}
    traces = {m: _held(keyed, m, files) for m, files in held.items()}     # the run's first build compiled all four
    assert took(run) == {MK.ABSENT: 3} and traces["D.X"].exists(), "a second build of the run leaves what the first settled"
    _stock(store, keyed, {"D.X": ("1" * 16, X), "R.A": ("2" * 16, A), "R": ("3" * 16, R)})
    traces = {m: _held(keyed, m, files) for m, files in held.items()}
    assert took(set()) == {MK.ABSENT: 1} and traces["D.X"].exists(), "the cache holds D.X as the tree does; R.B it lacks"
    traces["D.X"] = _held(keyed, "D.X", {"9" * 16 + ".olean": b"another X", "b" * 16 + ".ilean": b"x"})
    assert took(set(), also=()) == {} and traces["D.X"].exists(), "not named: d's build stays whole"
    assert took(set(), also=(root,)) == {} and traces["D.X"].exists(), "the root named is no path dependency"
    assert took(set()) == {MK.REPLACED: 1} and not traces["D.X"].exists(), "another version than the cache's: Lake restores it"


def test_a_module_above_one_that_is_compiled_keeps_its_trace_when_the_cache_holds_it_as_the_tree_does(tmp_path):
    """A module above one that isn't staged has no key until that one is compiled. It is keyed on what the tree holds
    of it, its key if the compile gives those bytes again: indexed under that key with the artifacts its trace names,
    it keeps its trace (Lake compiles it anyway if the module below comes out otherwise); else it is compiled too, so a
    cold tree is settled in one build, not one level of imports a build."""
    root = _workspace(tmp_path / "ws", {"r/R/A.lean": "import D.X\n-- edited since the cache indexed it\n"})
    store = tmp_path / "store"
    keyed = MK.keys(root, ["R.B"])
    _stock(store, keyed, {"D.X": ("1" * 16, X)})
    traces = {m: _held(keyed, m, files) for m, files in (("D.X", X), ("R.A", A), ("R", R), ("R.B", B))}
    held = MK.read_digests({m: keyed[m] for m in ("R.A", "R", "R.B")},
                           {("D.X", n.split(".", 1)[1]): C.digest(b) for n, b in X.items()})
    _entry(store, MK.key("R", keyed, held), "r", "R", "3" * 16, R)
    why: dict = {}
    MK.stage(store, root, ["R.B"], tmp_path / "cache", report=why, settled=True)
    assert why["took"] == {MK.ABSENT: 2} and why["left"] == 0
    assert {m for m, t in traces.items() if t.exists()} == {"D.X", "R"}, "R.A and R.B are compiled; R is the cache's"


def test_a_module_the_cache_could_not_index_is_left_to_lake_with_every_module_above_it(tmp_path):
    """`put` records each module a build compiled and it could not index, under the key it compiled under. Taking such a
    module's trace would compile it, and every module above it, in every build, for nothing: it is left to Lake, as a
    module without a key is. The record stands under the code that refused the module (`rules`): one written by other
    code, which may index the module now, lets it be compiled once more."""
    root = _workspace(tmp_path / "ws", {"r/R/C.lean": "import R.B\n"})
    store = tmp_path / "store"
    keyed = MK.keys(root, ["R.C"])
    _stock(store, keyed, {"D.X": ("1" * 16, X), "R.A": ("2" * 16, A), "R": ("3" * 16, R)})
    traces = {m: _held(keyed, m, files) for m, files in (
        ("D.X", X), ("R.A", A), ("R", R), ("R.B", B), ("R.C", {"5" * 16 + ".olean": b"C", "6" * 16 + ".ilean": b"c"}))}
    named = {(m, n.split(".", 1)[1]): C.digest(b) for m, files in (("D.X", X), ("R.A", A), ("R", R)) for n, b in files.items()}
    marker = MK.unstored_path(store, MK.key("R.B", keyed, named))
    marker.parent.mkdir()
    record = {"module": "R.B", "scope": "r", "why": "setup.json names another", "by": MK.rules()}
    marker.write_text(json.dumps(record))
    why: dict = {}
    MK.stage(store, root, ["R.C"], tmp_path / "cache-1", report=why, settled=True)
    assert (why["took"], why["left"]) == ({}, 2) and all(t.exists() for t in traces.values()), "R.B, and R.C, which loads it"
    marker.write_text(json.dumps(record | {"by": C.digest(b"the code before a change to what is refused")}))
    MK.stage(store, root, ["R.C"], tmp_path / "cache-2", report=why, settled=True)
    assert (why["took"], why["left"]) == ({MK.ABSENT: 2}, 0) and not traces["R.B"].exists() and not traces["R.C"].exists()


def test_a_damaged_record_that_a_module_could_not_be_indexed_is_removed_where_it_is_read(tmp_path):
    key = C.digest(b"a key")
    marker = MK.unstored_path(tmp_path, key)
    assert MK.unstored(tmp_path, key) is None, "none was written"
    marker.parent.mkdir()
    marker.write_text(json.dumps({"module": "R.B", "scope": "r", "why": "", "by": MK.rules()}))
    assert MK.unstored(tmp_path, key) == marker
    marker.write_text('{"module": "R.B", "sco')
    assert MK.unstored(tmp_path, key) is None and not marker.exists()


def test_an_entry_of_another_schema_is_never_staged(tmp_path):
    """An entry written under an earlier schema has no `schema` of this one: were its key ever computed again, it would
    still not be read."""
    root = _workspace(tmp_path / "ws")
    store = tmp_path / "store"
    k = _stock(store, MK.keys(root), {"D.X": ("1" * 16, X)})
    f = MK.entry_path(store, k["D.X"])
    e = json.loads(f.read_text())
    assert MK.valid(e) and MK.stage(store, root, [], tmp_path / "c1")[0] == {("d", "1" * 16): "D.X"}
    for other in ({k: v for k, v in e.items() if k != "schema"}, e | {"schema": "lean-module/v3"}):
        f.write_text(json.dumps(other))
        assert not MK.valid(other) and MK.stage(store, root, [], tmp_path / "c2")[0] == {}


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
