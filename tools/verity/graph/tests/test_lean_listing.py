"""The lean-module and lean-lock listings on a toy checkout of Lake packages laid out as Security's are: a package whose
`srcDir` is its parent, a requirement by path, a `proved_in`, a lock's `roots`, every import form, `module` files and the
legacy files importing them, transitive reads, and the trees that make no graph."""
from __future__ import annotations

import json
import os

import pytest
from conftest import toy_tree

from verity_graph import ROOT, graph
from verity_graph.families import lean_lock, lean_module
from verity_graph.runners.lean_module import TOOLS

LISTERS = {"lean-module": lean_module.tasks, "lean-lock": lean_lock.tasks}
TOOLCHAIN = "leanprover/lean4:v4.34.0\n"
#: the toy's `module` files: Lean refuses a `module` importing a legacy file, so B imports only these
MODULES = ("Sec/Definitions.lean", "Sec/Definitions/A.lean", "Sec/Proofs/B.lean", "Sec/Proofs/C.lean", "Sec/Proofs/D.lean")


def manifest(name: str, *paths: tuple[str, str]) -> str:
    return json.dumps({"version": "1.1.0", "packagesDir": ".lake/packages", "name": name, "lakeDir": ".lake",
                       "packages": [{"type": "path", "scope": "", "name": n, "manifestFile": "lake-manifest.json",
                                     "inherited": False, "dir": d, "configFile": "lakefile.toml"} for n, d in paths]})


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
    "Sec/Proofs/lean-audit.json": json.dumps({"guarantees": [], "roots": ["Proofs.G"]}),
    "Sec/Proofs.lean": "import Proofs.B\nimport Proofs.E\nimport Mathlib.Logic.Basic\n",
    "Sec/Proofs/B.lean": "module -- the module system\n/- not a doc comment, which would end the header -/\n"
                         "public import Definitions.A\nmeta import Proofs.C\nimport all Proofs.D\n"
                         "public meta import Init.Data.List\n\ntheorem b : True := trivial\n",
    "Sec/Proofs/C.lean": "module\n\npublic import Definitions\nimport Proofs.D\n",
    "Sec/Proofs/D.lean": "module\n\nprelude\n",
    "Sec/Proofs/E.lean": "import Scratch\nimport Proofs.Missing\n",
    "Sec/Proofs/F.lean": "theorem f : True := trivial\n",
    "Sec/Proofs/G.lean": "import Proofs.D\n",
    "tools/verity/lean/tests/fixtures/ctl/lakefile.toml": 'name = "ctl"\n\n[[lean_lib]]\nname = "Ctl"\n',
    "tools/verity/lean/tests/fixtures/ctl/lean-audit.json": "{}",
    "tools/verity/lean/tests/fixtures/ctl/Ctl.lean": "def c := 0\n",
    "tools/verity/lean/check.py": "",
    "tools/verity/check/common.py": "",
    **{f: "" for f in TOOLS},
    f"{graph.RUNNERS}/lean_module.py": "",
    f"{graph.RUNNERS}/lean_lock.py": "",
}


def listed(tmp_path, **changes) -> dict:
    tree = toy_tree(tmp_path / "tree", (ROOT / graph.SPEC).read_text(), {**FILES, **changes})
    return {t["id"]: t for t in graph.listing(tree, listers=LISTERS)["tasks"]}


def built(*files: str) -> list[dict]:
    """What an importer reads of each of ``files``' tasks, in the core's order: every part of a `module` file (`MODULES`),
    the `olean` of a legacy one."""
    return sorted(({"task": f"lean-module:{f}", "output": o} for f in files
                   for o in (("olean", *lean_module.PARTS) if f in MODULES else ("olean",))),
                  key=lambda r: (r["task"], r["output"]))


def test_modules_of_packages_with_a_parent_src_dir_read_their_imports_transitively(tmp_path):
    ts = listed(tmp_path)
    assert sorted(i for i in ts if i.startswith("lean-module:")) == [f"lean-module:Sec/{f}" for f in (
        "Definitions.lean", "Definitions/A.lean", "Proofs.lean", "Proofs/B.lean", "Proofs/C.lean", "Proofs/D.lean",
        "Proofs/E.lean", "Proofs/G.lean")], \
        "the default targets and the lock's roots (Proofs.G) are built; a file no library claims (Scratch), a module " \
        "nothing builds (Proofs.F) and the auditor's fixtures have no task"
    b = ts["lean-module:Sec/Proofs/B.lean"]
    assert b["vars"] == {"module": "Proofs.B", "package": "proofs", "options": "{}"}
    assert ts["lean-module:Sec/Definitions/A.lean"]["vars"]["options"] == '{"autoImplicit":false}'
    assert b["inputs"] == sorted(["Sec/Proofs/B.lean", "Sec/Proofs/lakefile.toml", *TOOLS,
                                  *graph.runner_sources("lean-module")])
    assert b["reads"] == built("Sec/Definitions.lean", "Sec/Definitions/A.lean", "Sec/Proofs/C.lean", "Sec/Proofs/D.lean"), \
        "public, meta and all imports are read, each with what it imports; the toolchain's are the env's"
    assert b["outputs"] == [{"name": n, "path": f"out/{p}"} for n, p in (
        ("axioms", "axioms/Proofs/B.json"), ("ilean", "lean/Proofs/B.ilean"), ("ir", "lean/Proofs/B.ir"),
        ("ir.sig", "lean/Proofs/B.ir.sig"), ("olean", "lean/Proofs/B.olean"),
        ("olean.private", "lean/Proofs/B.olean.private"), ("olean.server", "lean/Proofs/B.olean.server"))], \
        "a `module` file's task outputs the parts lean writes beside its olean"
    assert b["about"] == "compile and kernel-replay Sec/Proofs/B.lean" and b["env"] == "lean" and b["key"] is None
    e = ts["lean-module:Sec/Proofs/E.lean"]
    assert e["reads"] == [], \
        "an import no package claims (Scratch) or no file holds (Proofs.Missing) is left for the compile to fail on"
    assert [o["name"] for o in e["outputs"]] == ["axioms", "ilean", "olean"], "a legacy file's task outputs no parts"
    assert ts["lean-module:Sec/Proofs/D.lean"]["key"] is not None, "a module that reads nothing is keyed when listed"
    assert ts["lean-module:Sec/Proofs/G.lean"]["reads"] == [
        {"task": "lean-module:Sec/Proofs/D.lean", "output": o}
        for o in ("ir", "ir.sig", "olean", "olean.private", "olean.server")], \
        "a legacy file reads every part of a `module` file it imports, as Lake gives it them"
    assert ts["lean-module:Sec/Proofs.lean"]["reads"] == built(
        *(f"Sec/{f}" for f in ("Definitions.lean", "Definitions/A.lean", "Proofs/B.lean", "Proofs/C.lean", "Proofs/D.lean",
                               "Proofs/E.lean"))), "Mathlib's imports are the env's"


def test_a_lock_reads_every_package_its_records_read(tmp_path):
    ts = listed(tmp_path)
    assert sorted(i for i in ts if i.startswith("lean-lock:")) == ["lean-lock:Sec", "lean-lock:Sec/Proofs"]
    for pkg in ("Sec", "Sec/Proofs"):
        t = ts[f"lean-lock:{pkg}"]
        assert t["reads"] == built(*(i.removeprefix("lean-module:") for i in ts if i.startswith("lean-module:"))), \
            "Sec names Sec/Proofs as proved_in, and Sec/Proofs requires Sec by path: each reads both builds"
        assert {"Sec/Proofs/B.lean", "Sec/Definitions/A.lean", "Sec/lean-audit.json", "Sec/Proofs/lean-audit.json",
                "tools/verity/lean/check.py", "tools/verity/check/common.py"} <= set(t["inputs"])
        assert t["outputs"] == [{"name": "records", "path": f"out/{pkg}/lean-audit.json"}]


def test_a_lock_the_audit_refuses_is_listed_for_its_own_task_to_fail(tmp_path):
    ts = listed(tmp_path, **{"Sec/lean-audit.json": "[]", "Sec/Proofs/lean-audit.json": json.dumps({"proved_in": ["../../.."]})})
    assert ts["lean-lock:Sec"]["reads"] == built("Sec/Definitions.lean", "Sec/Definitions/A.lean"), \
        "a lock that isn't an object names no proved_in, as lean_audit.provers reads it"
    assert len({r["task"] for r in ts["lean-lock:Sec/Proofs"]["reads"]}) == 7, "a proved_in outside the tree adds no package"


@pytest.mark.slow
def test_this_tree_s_modules_and_imports_are_those_the_lean_build_cache_keys():
    """On this checkout, whose packages no toy stands in for: the modules listed are those `module_keys.keys` keys from each
    package's default targets and lock roots, each importing what its key reads, so a task and the cache see one module
    one way."""
    pkgs, _owner, direct, _system = lean_module.imports(ROOT, graph.tracked(ROOT))
    keyed = {}
    for d in pkgs:
        ks = lean_module.mk.keys(ROOT / d, lean_module.mk.policy_roots(ROOT / d))
        keyed |= {os.path.relpath(k.package.file(m), ROOT): sorted(os.path.relpath(ks[i].package.file(i), ROOT)
                                                                   for i in k.imports) for m, k in ks.items()}
    assert keyed == direct


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
