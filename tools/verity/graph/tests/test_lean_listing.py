"""The lean-module and lean-lock listings on a toy checkout of Lake packages laid out as Security's are: a package whose
`srcDir` is its parent, a requirement by path, a `proved_in`, a lock's `roots`, every import form, transitive reads, and
the trees that make no graph."""
from __future__ import annotations

import json
import os

import pytest
from conftest import toy_tree

from verity_graph import ROOT, graph
from verity_graph.families import lean_lock, lean_module

LISTERS = {"lean-module": lean_module.tasks, "lean-lock": lean_lock.tasks}
TOOLCHAIN = "leanprover/lean4:v4.34.0\n"


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
    "Sec/Definitions.lean": "import Definitions.A\n",
    "Sec/Definitions/A.lean": "/- a comment /- nested -/ -/\nimport Mathlib.Logic.Basic\n\ndef a := 1\n",
    "Sec/Scratch.lean": "def scratch := 0\n",
    "Sec/Proofs/lakefile.toml": 'name = "proofs"\nsrcDir = ".."\ndefaultTargets = ["Proofs"]\n\n[[require]]\nname = "sec"\n'
                                'path = ".."\n\n[[lean_lib]]\nname = "Proofs"\n',
    "Sec/Proofs/lake-manifest.json": manifest("proofs", ("sec", "..")),
    "Sec/Proofs/lean-toolchain": TOOLCHAIN,
    "Sec/Proofs/lean-audit.json": json.dumps({"guarantees": [], "roots": ["Proofs.G"]}),
    "Sec/Proofs.lean": "import Proofs.B\nimport Proofs.E\n",
    "Sec/Proofs/B.lean": "module -- the module system\n/-! docs -/\npublic import Definitions.A\nmeta import Proofs.C\n"
                         "import all Proofs.D\npublic meta import Init.Data.List\n\ntheorem b : True := trivial\n",
    "Sec/Proofs/C.lean": "import Definitions\nimport Proofs.D\n",
    "Sec/Proofs/D.lean": "prelude\n",
    "Sec/Proofs/E.lean": "import Scratch\nimport Proofs.Missing\n",
    "Sec/Proofs/F.lean": "theorem f : True := trivial\n",
    "Sec/Proofs/G.lean": "import Proofs.D\n",
    "tools/verity/lean/tests/fixtures/ctl/lakefile.toml": 'name = "ctl"\n\n[[lean_lib]]\nname = "Ctl"\n',
    "tools/verity/lean/tests/fixtures/ctl/lean-audit.json": "{}",
    "tools/verity/lean/tests/fixtures/ctl/Ctl.lean": "def c := 0\n",
    "tools/verity/lean/check.py": "",
    "tools/verity/check/common.py": "",
    f"{graph.RUNNERS}/lean_module.py": "",
    f"{graph.RUNNERS}/lean_lock.py": "",
}


def listed(tmp_path, **changes) -> dict:
    tree = toy_tree(tmp_path / "tree", (ROOT / graph.SPEC).read_text(), {**FILES, **changes})
    return {t["id"]: t for t in graph.listing(tree, listers=LISTERS)["tasks"]}


def olean(*files: str) -> list[dict]:
    return [{"task": f"lean-module:{f}", "output": "olean"} for f in sorted(files)]


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
    assert b["inputs"] == sorted(["Sec/Proofs/B.lean", "Sec/Proofs/lakefile.toml", *graph.runner_sources("lean-module")])
    assert b["reads"] == olean("Sec/Definitions.lean", "Sec/Definitions/A.lean", "Sec/Proofs/C.lean", "Sec/Proofs/D.lean"), \
        "public, meta and all imports are read, each with what it imports; the toolchain's and Mathlib's are the env's"
    assert b["outputs"] == [{"name": "axioms", "path": "out/axioms/Proofs/B.json"},
                            {"name": "ilean", "path": "out/lean/Proofs/B.ilean"},
                            {"name": "olean", "path": "out/lean/Proofs/B.olean"}]
    assert b["about"] == "compile and kernel-replay Sec/Proofs/B.lean" and b["env"] == "lean" and b["key"] is None
    assert ts["lean-module:Sec/Proofs/E.lean"]["reads"] == [], \
        "an import no package claims (Scratch) or no file holds (Proofs.Missing) is left for the compile to fail on"
    assert ts["lean-module:Sec/Proofs/D.lean"]["key"] is not None, "a module that reads nothing is keyed when listed"
    assert {r["task"] for r in ts["lean-module:Sec/Proofs.lean"]["reads"]} == {
        f"lean-module:Sec/{f}" for f in ("Definitions.lean", "Definitions/A.lean", "Proofs/B.lean", "Proofs/C.lean",
                                         "Proofs/D.lean", "Proofs/E.lean")}


def test_a_lock_reads_every_package_its_records_read(tmp_path):
    ts = listed(tmp_path)
    assert sorted(i for i in ts if i.startswith("lean-lock:")) == ["lean-lock:Sec", "lean-lock:Sec/Proofs"]
    for pkg in ("Sec", "Sec/Proofs"):
        t = ts[f"lean-lock:{pkg}"]
        assert t["reads"] == olean(*(i.removeprefix("lean-module:") for i in ts if i.startswith("lean-module:"))), \
            "Sec names Sec/Proofs as proved_in, and Sec/Proofs requires Sec by path: each reads both builds"
        assert {"Sec/Proofs/B.lean", "Sec/Definitions/A.lean", "Sec/lean-audit.json", "Sec/Proofs/lean-audit.json",
                "tools/verity/lean/check.py", "tools/verity/check/common.py"} <= set(t["inputs"])
        assert t["outputs"] == [{"name": "records", "path": f"out/{pkg}/lean-audit.json"}]


def test_a_lock_the_audit_refuses_is_listed_for_its_own_task_to_fail(tmp_path):
    ts = listed(tmp_path, **{"Sec/lean-audit.json": "[", "Sec/Proofs/lean-audit.json": json.dumps({"proved_in": ["../../.."]})})
    assert ts["lean-lock:Sec"]["reads"] == olean("Sec/Definitions.lean", "Sec/Definitions/A.lean"), \
        "a lock that isn't JSON names no proved_in, as lean_audit.provers reads it"
    assert len(ts["lean-lock:Sec/Proofs"]["reads"]) == 7, "a proved_in outside the tree adds no package"


@pytest.mark.slow
def test_this_tree_s_modules_and_imports_are_those_the_lean_build_cache_keys():
    """On this checkout, whose packages no toy stands in for: the modules listed are those `module_keys.keys` keys from each
    package's default targets and lock roots, each importing what its key reads, so a task and the cache see one module
    one way."""
    pkgs, _owner, direct = lean_module.imports(ROOT, graph.tracked(ROOT))
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
])
def test_trees_that_make_no_graph(tmp_path, changes, refusal):
    with pytest.raises(graph.GraphError, match=refusal):
        listed(tmp_path, **changes)
