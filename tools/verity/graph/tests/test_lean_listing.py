"""The lean-module, lean-lock and lean-env listings on a toy checkout of Lake packages laid out as Security's are: a package
whose `srcDir` is its parent, a requirement by path, a `proved_in`, a lock's `roots`, every import form, `module` files and
the legacy files importing them, what each loads of its imports at each level, the envs the packages compile in, and the
trees that make no graph."""
from __future__ import annotations

import json
import os

import pytest
from conftest import toy_tree

from verity_graph import ROOT, graph
from verity_graph.families import lean_env, lean_lock, lean_module
from verity_graph.runners.lean_env import TOOLS as ENV_TOOLS
from verity_graph.runners.lean_module import TOOLS

LISTERS = {"lean-module": lean_module.tasks, "lean-lock": lean_lock.tasks, "lean-env": lean_env.tasks}
TOOLCHAIN = "leanprover/lean4:v4.34.0\n"
#: what each toy module's task reads of its env: Sec and Sec/Proofs fetch nothing and share a toolchain, so one env, Sec's
ENV = {"task": "lean-env:Sec", "output": "axioms"}
#: the toy's `module` files: Lean refuses a `module` importing a legacy file, so these import only each other
MODULES = ("Sec/Definitions.lean", "Sec/Definitions/A.lean", "Sec/Proofs/B.lean", "Sec/Proofs/C.lean", "Sec/Proofs/D.lean",
           "Sec/Proofs/H.lean", "Sec/Proofs/J.lean")
#: what a task reads of a `module` file's build at each level it loads it at
AT = {"public": ("olean",), "code": ("olean", "ir.sig", "ir"), "all": ("olean", *lean_module.PARTS)}


def manifest(name: str, *paths: tuple[str, str], fetched: tuple[str, ...] = (), inherited: tuple[str, ...] = (),
             revs: dict[str, str] | None = None) -> str:
    return json.dumps({"version": "1.1.0", "packagesDir": ".lake/packages", "name": name, "lakeDir": ".lake",
                       "packages": [*({"type": "git", "scope": "leanprover-community", "name": n,
                                       "rev": (revs or {}).get(n, "0" * 40),
                                       "url": f"https://github.com/leanprover-community/{n}", "inputRev": "v4.34.0",
                                       "manifestFile": "lake-manifest.json", "inherited": n in inherited,
                                       "configFile": "lakefile.lean"}
                                     for n in fetched),
                                    *({"type": "path", "scope": "", "name": n, "manifestFile": "lake-manifest.json",
                                       "inherited": False, "dir": d, "configFile": "lakefile.toml"} for n, d in paths)]})


FILES = {
    "Sec/lakefile.toml": 'name = "sec"\ndefaultTargets = ["Definitions"]\n\n[leanOptions]\nautoImplicit = false\n\n'
                         '[[lean_lib]]\nname = "Definitions"\n',
    "Sec/lake-manifest.json": manifest("sec"),
    "Sec/lean-toolchain": TOOLCHAIN,
    "Sec/lean-audit.json": json.dumps({"guarantees": [], "proved_in": ["Proofs"]}),
    "Sec/Definitions.lean": "module\n\npublic import Definitions.A\n",
    "Sec/Definitions/A.lean": "/- a comment /- nested -/ -/\nmodule\n\npublic import Init.Data.List\n\ndef a := 1\n",
    "Sec/Scratch.lean": "def scratch := 0\n",
    "Sec/Proofs/lakefile.toml": 'name = "proofs"\nsrcDir = ".."\ndefaultTargets = ["Proofs"]\n\n[[require]]\nname = "sec"\n'
                                'path = ".."\n\n[[lean_lib]]\nname = "Proofs"\n',
    "Sec/Proofs/lake-manifest.json": manifest("proofs", ("sec", "..")),
    "Sec/Proofs/lean-toolchain": TOOLCHAIN,
    "Sec/Proofs/lean-audit.json": json.dumps({"guarantees": [], "roots": ["Proofs.G", "Proofs.H", "Proofs.J"]}),
    "Sec/Proofs.lean": "import Proofs.B\nimport Proofs.E\nimport Mathlib.Logic.Basic\n",
    "Sec/Proofs/B.lean": "module -- the module system\n/- not a doc comment, which would end the header -/\n"
                         "public import Definitions.A\nmeta import Proofs.C\nimport all Definitions\n"
                         "public meta import Init.Data.List\n\ntheorem b : True := trivial\n",
    "Sec/Proofs/C.lean": "module\n\npublic import Definitions\nimport Proofs.D\n",
    "Sec/Proofs/D.lean": "module\n\nprelude\n",
    "Sec/Proofs/E.lean": "import Scratch\nimport Proofs.Missing\n",
    "Sec/Proofs/F.lean": "theorem f : True := trivial\n",
    "Sec/Proofs/G.lean": "import Proofs.D\n",
    "Sec/Proofs/H.lean": "module\n\nimport Proofs.C\n",
    "Sec/Proofs/J.lean": "module\n\nimport all Proofs.C\n",
    "tools/verity/lean/tests/fixtures/ctl/lakefile.toml": 'name = "ctl"\n\n[[lean_lib]]\nname = "Ctl"\n',
    "tools/verity/lean/tests/fixtures/ctl/lean-audit.json": "{}",
    "tools/verity/lean/tests/fixtures/ctl/Ctl.lean": "def c := 0\n",
    "tools/verity/lean/check.py": "",
    "tools/verity/check/common.py": "",
    "tools/verity/graph/verity_graph/contract.py": "",
    **{f: "" for f in (*TOOLS, *ENV_TOOLS)},
    f"{graph.RUNNERS}/lean_module.py": "",
    f"{graph.RUNNERS}/lean_lock.py": "",
    f"{graph.RUNNERS}/lean_env.py": "",
}
#: packages on other envs than Sec's: Ext fetches Mathlib, as Ext/More does beside a requirement of Ext by path, which
#: leaves the env Ext's; Old fetches nothing on another toolchain
OTHER_ENVS = {
    "Ext/lakefile.toml": 'name = "ext"\ndefaultTargets = ["Ext"]\n\n[[require]]\nname = "mathlib"\n'
                         'git = "https://github.com/leanprover-community/mathlib"\n\n[[lean_lib]]\nname = "Ext"\n',
    "Ext/lake-manifest.json": manifest("ext", fetched=("mathlib",)),
    "Ext/lean-toolchain": TOOLCHAIN,
    "Ext/Ext.lean": "import Mathlib.Data.Nat.Basic\n",
    "Ext/More/lakefile.toml": 'name = "more"\ndefaultTargets = ["More"]\n\n[[require]]\nname = "ext"\npath = ".."\n\n'
                              '[[require]]\nname = "mathlib"\ngit = "https://github.com/leanprover-community/mathlib"\n\n'
                              '[[lean_lib]]\nname = "More"\n',
    "Ext/More/lake-manifest.json": manifest("more", ("ext", ".."), fetched=("mathlib",)),
    "Ext/More/lean-toolchain": TOOLCHAIN,
    "Ext/More/More.lean": "import Ext\n",
    "Old/lakefile.toml": 'name = "old"\ndefaultTargets = ["Old"]\n\n[[lean_lib]]\nname = "Old"\n',
    "Old/lake-manifest.json": manifest("old"),
    "Old/lean-toolchain": "leanprover/lean4:v4.33.0\n",
    "Old/Old.lean": "def old := 0\n",
}


def listed(tmp_path, **changes) -> dict:
    tree = toy_tree(tmp_path / "tree", (ROOT / graph.SPEC).read_text(), {**FILES, **changes})
    return {t["id"]: t for t in graph.listing(tree, listers=LISTERS)["tasks"]}


def ordered(rs) -> list[dict]:
    return sorted(rs, key=lambda r: (r["task"], r["output"]))


def built(*files: str) -> list[dict]:
    """What a lock reads of each of ``files``' tasks, in the core's order: every part of a `module` file (`MODULES`), the
    `olean` of a legacy one."""
    return ordered({"task": f"lean-module:{f}", "output": o} for f in files
                   for o in (AT["all"] if f in MODULES else ("olean",)))


def loaded(at: dict[str, str]) -> list[dict]:
    """What a module's task reads of each task of ``at`` at its level there (`AT`), in the core's order: of a `module` file
    that level's parts, of a legacy one its `olean`, and of each its `axioms`; and its env's `axioms`."""
    return ordered([ENV, *({"task": f"lean-module:{f}", "output": o} for f, level in at.items()
                           for o in (*(AT[level] if f in MODULES else ("olean",)), "axioms"))])


def test_modules_of_packages_with_a_parent_src_dir_read_their_imports_transitively(tmp_path):
    ts = listed(tmp_path)
    assert sorted(i for i in ts if i.startswith("lean-module:")) == [f"lean-module:Sec/{f}" for f in (
        "Definitions.lean", "Definitions/A.lean", "Proofs.lean", "Proofs/B.lean", "Proofs/C.lean", "Proofs/D.lean",
        "Proofs/E.lean", "Proofs/G.lean", "Proofs/H.lean", "Proofs/J.lean")], \
        "the default targets and the lock's roots (Proofs.G, H and J) are built; a file no library claims (Scratch), a " \
        "module nothing builds (Proofs.F) and the auditor's fixtures have no task"
    b = ts["lean-module:Sec/Proofs/B.lean"]
    assert b["vars"] == {"module": "Proofs.B", "package": "proofs", "options": "{}"}
    assert ts["lean-module:Sec/Definitions/A.lean"]["vars"]["options"] == '{"autoImplicit":false}'
    assert b["inputs"] == sorted(["Sec/Proofs/B.lean", "Sec/Proofs/lakefile.toml", *TOOLS,
                                  *graph.runner_sources("lean-module")])
    assert b["reads"] == loaded({"Sec/Definitions.lean": "all", "Sec/Definitions/A.lean": "code", "Sec/Proofs/C.lean": "code",
                                 "Sec/Proofs/D.lean": "code"}), \
        "`import all` loads every part of Definitions; `meta import` loads the code of C and of all C imports, privately " \
        "too (Proofs.D); the toolchain's are the env's"
    assert b["outputs"] == [{"name": n, "path": f"out/{p}"} for n, p in (
        ("axioms", "axioms/Proofs/B.json"), ("ilean", "lean/Proofs/B.ilean"), ("ir", "lean/Proofs/B.ir"),
        ("ir.sig", "lean/Proofs/B.ir.sig"), ("olean", "lean/Proofs/B.olean"),
        ("olean.private", "lean/Proofs/B.olean.private"), ("olean.server", "lean/Proofs/B.olean.server"))], \
        "a `module` file's task outputs the parts lean writes beside its olean"
    assert b["about"] == "compile and kernel-replay Sec/Proofs/B.lean" and b["env"] == "lean" and b["key"] is None
    e = ts["lean-module:Sec/Proofs/E.lean"]
    assert e["reads"] == [ENV], \
        "an import no package claims (Scratch) or no file holds (Proofs.Missing) is left for the compile to fail on"
    assert [o["name"] for o in e["outputs"]] == ["axioms", "ilean", "olean"], "a legacy file's task outputs no parts"
    assert ts["lean-module:Sec/Proofs/D.lean"]["reads"] == [ENV], "a module that imports nothing reads its env"
    assert ts["lean-module:Sec/Proofs/G.lean"]["reads"] == [ENV, *(
        {"task": "lean-module:Sec/Proofs/D.lean", "output": o}
        for o in ("axioms", "ir", "ir.sig", "olean", "olean.private", "olean.server"))], \
        "a legacy file reads every part of a `module` file it imports, as Lake gives it them, and its record"
    assert ts["lean-module:Sec/Proofs.lean"]["reads"] == loaded({f"Sec/{f}": "all" for f in (
        "Definitions.lean", "Definitions/A.lean", "Proofs/B.lean", "Proofs/C.lean", "Proofs/D.lean", "Proofs/E.lean")}), \
        "and of every module it imports, transitively; Mathlib's imports are the env's"


def test_a_dotted_lean_option_is_one_option_as_lake_reads_it(tmp_path):
    lakefile = FILES["Sec/lakefile.toml"].replace("autoImplicit = false\n",
                                                  "autoImplicit = false\nbackward.proofsInPublic = true\n")
    ts = listed(tmp_path, **{"Sec/lakefile.toml": lakefile})
    assert ts["lean-module:Sec/Definitions/A.lean"]["vars"]["options"] == \
        '{"autoImplicit":false,"backward.proofsInPublic":true}', \
        "TOML reads a dotted key as a nested table and Lake joins its keys into one option's name: `lean --setup` " \
        "refuses a table as an option's value (`invalid LeanOptionValue type`)"


def test_a_module_file_reads_the_public_part_of_what_it_imports(tmp_path):
    reads = {i.removeprefix("lean-module:Sec/"): t["reads"] for i, t in listed(tmp_path).items()}
    assert reads["Proofs/C.lean"] == loaded({"Sec/Definitions.lean": "public", "Sec/Definitions/A.lean": "public",
                                            "Sec/Proofs/D.lean": "public"}), \
        "a `module` file loads the public part of each module it imports and of what those import by `public import`"
    assert reads["Proofs/H.lean"] == loaded({"Sec/Proofs/C.lean": "public", "Sec/Definitions.lean": "public",
                                            "Sec/Definitions/A.lean": "public"}), \
        "and nothing of what they import privately (C's Proofs.D)"
    assert reads["Proofs/J.lean"] == loaded({"Sec/Proofs/C.lean": "all", "Sec/Definitions.lean": "public",
                                            "Sec/Definitions/A.lean": "public", "Sec/Proofs/D.lean": "public"}), \
        "`import all` loads every part of the module it names and the public part of all that one imports"
    assert reads["Definitions.lean"] == loaded({"Sec/Definitions/A.lean": "public"})


def test_a_lock_reads_every_package_its_records_read(tmp_path):
    ts = listed(tmp_path)
    assert sorted(i for i in ts if i.startswith("lean-lock:")) == ["lean-lock:Sec", "lean-lock:Sec/Proofs"]
    for pkg in ("Sec", "Sec/Proofs"):
        t = ts[f"lean-lock:{pkg}"]
        assert t["reads"] == built(*(i.removeprefix("lean-module:") for i in ts if i.startswith("lean-module:"))), \
            "Sec names Sec/Proofs as proved_in, and Sec/Proofs requires Sec by path: each reads both builds"
        assert {"Sec/Proofs/B.lean", "Sec/Definitions/A.lean", "Sec/lean-audit.json", "Sec/Proofs/lean-audit.json",
                "tools/verity/lean/check.py", "tools/verity/check/common.py",
                "tools/verity/graph/verity_graph/contract.py"} <= set(t["inputs"])
        assert t["outputs"] == [{"name": "records", "path": f"out/{pkg}/lean-audit.json"}]


def test_a_lock_the_audit_refuses_is_listed_for_its_own_task_to_fail(tmp_path):
    ts = listed(tmp_path, **{"Sec/lean-audit.json": "[]", "Sec/Proofs/lean-audit.json": json.dumps({"proved_in": ["../../.."]})})
    assert ts["lean-lock:Sec"]["reads"] == built("Sec/Definitions.lean", "Sec/Definitions/A.lean"), \
        "a lock that isn't an object names no proved_in, as lean_audit.provers reads it"
    assert len({r["task"] for r in ts["lean-lock:Sec/Proofs"]["reads"]}) == 7, "a proved_in outside the tree adds no package"


def test_the_listers_share_one_reading_of_the_imports_while_every_file_it_read_is_unchanged(tmp_path):
    tree = toy_tree(tmp_path / "tree", (ROOT / graph.SPEC).read_text(), FILES)
    files = graph.tracked(tree)
    first = lean_module.imports(tree, files)
    assert lean_module.imports(tree, files) is first, "lean-module, lean-lock and lean-env read the headers once a listing"
    (tree / "Sec/Proofs/C.lean").write_text("module\n\npublic import Definitions\n")
    edited = lean_module.imports(tree, files)
    assert "Sec/Proofs/D.lean" not in edited[2]["Sec/Proofs/C.lean"], "a header edited under the same files is read again"
    assert edited == lean_module._imports(tree, files)
    (tree / "Sec/Proofs/lean-audit.json").write_text(json.dumps({"guarantees": [], "roots": ["Proofs.G"]}))
    rooted = lean_module.imports(tree, files)
    assert "Sec/Proofs/H.lean" not in rooted[2] and "Sec/Proofs/H.lean" in edited[2], \
        "and so is the tree after a lock's `roots` change"
    assert rooted == lean_module._imports(tree, files)


def test_one_env_per_toolchain_and_fetched_packages_each_module_reading_its_package_s(tmp_path):
    ts = listed(tmp_path, **OTHER_ENVS)
    assert sorted(i for i in ts if i.startswith("lean-env:")) == ["lean-env:Ext", "lean-env:Old", "lean-env:Sec"], \
        "Sec and Sec/Proofs fetch nothing on one toolchain, Ext/More fetches what Ext does (its requirement by path aside), " \
        "Old has a toolchain of its own; an env is named after its first package, and the auditor's fixtures have none"
    sec = ts["lean-env:Sec"]
    assert sec["vars"] == {"package": "Sec", "scope": "whole"}
    assert sec["inputs"] == sorted(["Sec/lake-manifest.json", "Sec/lean-toolchain", *ENV_TOOLS,
                                    *graph.runner_sources("lean-env")])
    assert sec["reads"] == [] and sec["key"] is not None, "the toolchain's env reads nothing, so it is keyed when listed"
    assert sec["outputs"] == [{"name": "axioms", "path": "out/lean-env/Sec/axioms.json"}]
    assert sec["about"] == "make Sec's lean env and kernel-replay its modules (scope whole)" and sec["env"] == "lean"
    ext = ts["lean-env:Ext"]
    assert ext["inputs"][:2] == ["Ext/lake-manifest.json", "Ext/lean-toolchain"]
    assert ext["reads"] == [{"task": "lean-env:Sec", "output": "axioms"}] and ext["key"] is None, \
        "Ext's env is layered on the toolchain's, Sec's, and keyed once that one's record is"
    assert ext["outputs"] == [{"name": "axioms", "path": "out/lean-env/Ext/axioms.json"}]
    assert ts["lean-env:Old"]["reads"] == [], "no env of Old's toolchain fetches less than Old's"
    envs = {i: [r for r in t["reads"] if r["task"].startswith("lean-env:")] for i, t in ts.items()
            if i.startswith("lean-module:")}
    of = {"lean-module:Ext/Ext.lean": "Ext", "lean-module:Ext/More/More.lean": "Ext", "lean-module:Old/Old.lean": "Old"}
    assert envs == {i: [{"task": f"lean-env:{of.get(i, 'Sec')}", "output": "axioms"}] for i in envs}, \
        "each module's task reads its own package's env, once"
    assert {"task": "lean-module:Ext/Ext.lean", "output": "axioms"} in ts["lean-module:Ext/More/More.lean"]["reads"]


def fetching(name: str, lib: str, **pins) -> dict[str, str]:
    """A package at ``lib`` of one module, ``lib``, whose manifest is `manifest(name, **pins)`."""
    return {f"{lib}/lakefile.toml": f'name = "{name}"\ndefaultTargets = ["{lib}"]\n\n[[lean_lib]]\nname = "{lib}"\n',
            f"{lib}/lake-manifest.json": manifest(name, **pins), f"{lib}/lean-toolchain": TOOLCHAIN,
            f"{lib}/{lib}.lean": "import Mathlib.Logic.Basic\n"}


#: beside `OTHER_ENVS`: Props pins Mathlib as Ext does but through a requirement's manifest (`inherited`), Big pins ArkLib
#: too, and Fork pins ArkLib and another Mathlib
LAYERED = {**fetching("props", "Props", fetched=("mathlib",), inherited=("mathlib",)),
           **fetching("big", "Big", fetched=("mathlib", "arklib")),
           **fetching("fork", "Fork", fetched=("mathlib", "arklib"), revs={"mathlib": "1" * 40})}


def test_packages_pinning_the_same_packages_share_an_env_and_one_pinning_more_is_layered_on_it(tmp_path, monkeypatch):
    ts = listed(tmp_path / "whole", **OTHER_ENVS, **LAYERED)
    assert {i: t["reads"] for i, t in ts.items() if i.startswith("lean-env:")} == {
        "lean-env:Big": [{"task": "lean-env:Ext", "output": "axioms"}],
        "lean-env:Ext": [{"task": "lean-env:Sec", "output": "axioms"}],
        "lean-env:Fork": [{"task": "lean-env:Sec", "output": "axioms"}],
        "lean-env:Old": [], "lean-env:Sec": []}, \
        "Props shares Ext's env, `inherited` aside; Big's is layered on Ext's, the largest it holds; Fork's Mathlib isn't " \
        "Ext's, so its env is layered on the toolchain's alone; Old's toolchain is its own"
    assert {"task": "lean-env:Ext", "output": "axioms"} in ts["lean-module:Props/Props.lean"]["reads"]
    assert {"task": "lean-env:Big", "output": "axioms"} in ts["lean-module:Big/Big.lean"]["reads"]
    assert len({t["outputs"][0]["path"] for i, t in ts.items() if i.startswith("lean-env:")}) == 5, \
        "each env's record has a path of its own, as one layered on another reads that one's"
    monkeypatch.setattr(lean_env, "SCOPE", "imports")
    ts = listed(tmp_path / "imports", **OTHER_ENVS, **LAYERED)
    assert all(t["reads"] == [] for i, t in ts.items() if i.startswith("lean-env:")), \
        "the imports scope layers no env, since `--closure-of` walks an import into the env below"


def test_the_whole_scope_keys_an_env_on_its_files_and_the_imports_scope_on_the_headers(tmp_path, monkeypatch):
    more = {"Sec/Proofs/G.lean": "import Proofs.D\nimport Std.Data.HashMap\n"}
    whole = listed(tmp_path / "whole", **OTHER_ENVS)
    assert listed(tmp_path / "whole-more", **OTHER_ENVS, **more)["lean-env:Sec"] == whole["lean-env:Sec"], \
        "in the whole scope a header that imports more of the env leaves its task as it was"
    monkeypatch.setattr(lean_env, "SCOPE", "imports")
    ts = listed(tmp_path / "imports", **OTHER_ENVS)
    assert {i: json.loads(t["vars"].pop("modules")) for i, t in ts.items() if i.startswith("lean-env:")} == {
        "lean-env:Sec": ["Init", "Init.Data.List", "Mathlib.Logic.Basic"], "lean-env:Ext": ["Init", "Mathlib.Data.Nat.Basic"],
        "lean-env:Old": ["Init"]}, \
        "the env's modules that the headers of the modules the check builds name, and Init but under `prelude` (Proofs.D); " \
        "a module a package claims (Proofs.Missing) or holds (Scratch) is no env's, nor is one nothing builds (Proofs.F)"
    assert ts["lean-env:Sec"]["vars"] == {"package": "Sec", "scope": "imports"}
    got = listed(tmp_path / "imports-more", **OTHER_ENVS, **more)["lean-env:Sec"]
    assert json.loads(got["vars"]["modules"]) == ["Init", "Init.Data.List", "Mathlib.Logic.Basic", "Std.Data.HashMap"]
    assert got["key"] != ts["lean-env:Sec"]["key"], "in the imports scope the same header change reruns it"


def test_a_scope_that_isn_t_one_makes_no_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(lean_env, "SCOPE", "closure")
    with pytest.raises(graph.GraphError, match="SCOPE is 'closure': it is one of whole, imports"):
        listed(tmp_path)


@pytest.mark.slow
def test_this_tree_s_modules_and_imports_are_those_the_lean_build_cache_keys():
    """On this checkout, whose packages no toy stands in for: the modules listed are those `module_keys.keys` keys from each
    package's default targets and lock roots, each importing what its key reads, so a task and the cache see one module
    one way. The cache alone keys no module that targets with different settings own (`module_keys.Package.owner`), nor
    what imports one."""
    pkgs, owner, direct, _system, _env = lean_module.imports(ROOT, graph.tracked(ROOT))
    keyed = {}
    for d in pkgs:
        ks = lean_module.mk.keys(ROOT / d, lean_module.mk.policy_roots(ROOT / d))
        keyed |= {os.path.relpath(k.package.file(m), ROOT): sorted(os.path.relpath(ks[i].package.file(i), ROOT)
                                                                   for i in k.imports) for m, k in ks.items()}
    unowned = {f for f, (d, mod) in owner.items() if f in direct and pkgs[d].lake.owner(mod) is None}
    while more := {f for f, edges in direct.items() if f not in unowned and unowned & set(edges)}:
        unowned |= more
    assert keyed == {f: list(edges) for f, edges in direct.items() if f not in unowned}
    assert len(unowned) < len(direct) // 100, "few enough that the cache still restores a build"


@pytest.mark.slow
def test_this_tree_s_locks_read_every_file_their_audit_can():
    """On this checkout: each lock's inputs are every file an audit of its package can read (`lean_audit.inputs`), so a
    change to a package its records read reruns it, as a Flock change reruns Security's."""
    files = graph.tracked(ROOT)
    ts = lean_lock.tasks(ROOT, files)
    assert len(ts) == len(lean_module.la.packages(files))
    for t in ts:
        assert t["inputs"] == lean_module.la.inputs(t["vars"]["package"], files), t["id"]


@pytest.mark.parametrize("changes, refusal", [
    ({"Sec/Proofs/D.lean": "import Proofs.C\n"}, "an import cycle"),
    ({"Sec/Other/lakefile.toml": 'name = "other"\nsrcDir = ".."\n\n[[lean_lib]]\nname = "Definitions"\n',
      "Sec/Other/lake-manifest.json": manifest("other"), "Sec/Other/lean-toolchain": TOOLCHAIN}, "a module of both"),
    ({"Sec/Proofs/lakefile.lean": "import Lake\n"}, "lakefile.toml only"),
    ({"Sec/lakefile.toml": 'name = "sec"\nmoreLeanArgs = ["-DmaxHeartbeats=0"]\n\n[[lean_lib]]\nname = "Definitions"\n'},
     "moreLeanArgs changes how Lake compiles"),
    ({"Sec/lakefile.toml": 'name = "sec"\n\n[[lean_lib]]\nname = "Definitions"\nglobs = ["Definitions.+"]\n'},
     "globs of its own"),
    ({"Sec/lean-audit.json": "["}, "Sec/lean-audit.json isn't JSON .*`roots`"),
    ({"Sec/Notes/lean-audit.json": "{"}, "Sec/Notes/lean-audit.json isn't JSON .*`proved_in`"),
])
def test_trees_that_make_no_graph(tmp_path, changes, refusal):
    with pytest.raises(graph.GraphError, match=refusal):
        listed(tmp_path, **changes)
