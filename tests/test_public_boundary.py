"""The public monorepo's boundary (`tools/verity/check/public_boundary.py`): no file the one-way export copies needs a file
it doesn't, beyond the edges `public_boundary_allowed.txt` lists, which only shrinks; and the export's dry run
(`public_export.py`), whose manifests lose only the entries naming what it doesn't copy."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "verity" / "check" / "public_boundary.py"
EXPORT = ROOT / "tools" / "verity" / "check" / "public_export.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PB = _load("public_boundary_under_test", SCRIPT)
PE = _load("public_export_under_test", EXPORT)


def test_no_public_file_needs_one_the_export_does_not_copy() -> None:
    """Every edge from a public file to one that isn't public is listed. The check runs in a process of its own, which
    reads the tree's files in forked workers."""
    r = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_allowed_is_sorted_unique_edges() -> None:
    lines = PB.read_allowed()
    assert lines, "public_boundary_allowed.txt lists no edge: delete it with the last one, and this test"
    assert lines == sorted(set(lines)), "public_boundary_allowed.txt is sorted with each edge once (`--prune` writes it so)"
    bad = [x for x in lines if not re.fullmatch(r"\S+ -> \S+", x)]
    assert not bad, f"public_boundary_allowed.txt lines that aren't `SOURCE -> TARGET`: {bad}"


def test_every_public_group_covers_a_file() -> None:
    """Each group of the decided boundary holds a tracked file, so a renamed directory can't empty one unseen."""
    boundary, tree = PB.Boundary.load(), PB.Tree(ROOT)
    empty = [g for g, r in boundary.groups.items() if not any(r.fullmatch(f) for f in tree.files)]
    assert not empty, f"public groups whose globs cover no tracked file: {empty}; fix their globs in public_boundary.toml"


BOUNDARY = """
when_closed = ["tests"]
[public]
a = ["verity/pub"]
b = ["verity/other"]
[private]
team = ["tools/secret", "verity/pub/hidden.py", "verity/pub/pkg/hidden_mod.py"]
[[needed]]
why = "structure"
paths = ["**/{lakefile.toml,pyproject.toml,__init__.py}"]
via = ["package"]
[[needed]]
why = "shared, for a"
paths = ["shared"]
from = ["a"]
"""

#: one file for each kind of edge, from the public `verity/pub` into `tools/priv`, which no glob makes public
TREE = {
    "verity/pub/lakefile.toml": 'name = "pub"\n[[lean_lib]]\nname = "Pub"\n[[require]]\nname = "priv"\npath = "../../tools/priv"\n',
    "verity/pub/Pub.lean": "import Priv\n",
    "tools/priv/lakefile.toml": 'name = "priv"\n[[lean_lib]]\nname = "Priv"\n',
    "tools/priv/Priv.lean": "def x := 1\n",
    "tools/priv/pyproject.toml": ('[project]\nname = "privtool"\n[project.scripts]\nresearch = "privpkg:main"\n'
                                  '[tool.hatch.build.targets.wheel]\npackages = ["privpkg"]\n'),
    "tools/priv/privpkg/__init__.py": "def main():\n    pass\n",
    "tools/priv/data.json": "{}\n",
    "tools/privbench/pyproject.toml": '[project]\nname = "privbench"\n',
    "verity/pub/pyproject.toml": ('[project]\nname = "pub"\ndependencies = ["privtool>=1"]\n'
                                  '[project.optional-dependencies]\nbench = ["privtool", "privbench"]\n'
                                  '[project.scripts]\npub-run = "privpkg:main"\n'),
    "verity/pub/imports.py": "import json\nimport privpkg\n",
    "verity/pub/reads.py": "from pathlib import Path\n\nDATA = Path(__file__).resolve().parents[2] / 'tools' / 'priv' / 'data.json'\n",
    "verity/pub/runs.py": "import subprocess\n\nsubprocess.run(['research', 'status'], check=True)\n",
    "verity/pub/run.sh": "#!/bin/bash\nset -e\nresearch status x\npython3 tools/priv/privpkg/__init__.py\n",
    "verity/pub/pkg/__init__.py": "",
    "verity/pub/pkg/mod.py": "X = 1\n",
    "verity/pub/pkg/hidden_mod.py": "Z = 1\n",
    "verity/pub/hidden.py": "Y = 1\n",
    "verity/pub/sees.py": "import hidden\n",
    "verity/pub/from_pkg.py": "from pkg import mod, hidden_mod\nfrom pkg.mod import X\n",
    "tools/priv/privpkg/sub.py": "",
    "verity/pub/from_priv.py": "from privpkg import sub\n",
    "verity/pub/wants.py": "import shared.s\n",
    "verity/other/wants.py": "import shared.t\n",
    "shared/__init__.py": "",
    "shared/s.py": "import shared.u\n",
    "shared/t.py": "",
    "shared/u.py": "",
    "tests/test_ok.py": "import json\n",
    "tests/test_bad.py": "import privpkg\n",
}


@pytest.fixture
def judged(tmp_path: Path):
    for f, text in TREE.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text, encoding="utf-8")
    (tmp_path / "boundary.toml").write_text(BOUNDARY, encoding="utf-8")
    tree = PB.Tree(tmp_path, files=TREE)
    return tmp_path, tree, PB.judge(tree, PB.Boundary.load(tmp_path / "boundary.toml"))


@pytest.mark.parametrize(("source", "kind", "target"), [
    ("verity/pub/Pub.lean", "lean import", "tools/priv/Priv.lean"),
    ("verity/pub/imports.py", "python import", "tools/priv/privpkg/__init__.py"),
    ("verity/pub/pyproject.toml", "python dependency", "tools/priv/pyproject.toml"),
    ("verity/pub/reads.py", "path", "tools/priv/data.json"),
    ("verity/pub/runs.py", "command", "tools/priv/pyproject.toml"),
    ("verity/pub/run.sh", "command", "tools/priv/pyproject.toml"),
    ("verity/pub/run.sh", "path", "tools/priv/privpkg/__init__.py"),
    ("verity/pub/sees.py", "python import", "verity/pub/hidden.py"),
    ("verity/pub/from_pkg.py", "python import", "verity/pub/pkg/hidden_mod.py"),
])
def test_each_kind_of_need_is_an_edge(judged, source: str, kind: str, target: str) -> None:
    _, _, j = judged
    assert (source, kind, target) in {(e.source, e.kind, e.target) for e in j.edges}, "\n".join(map(str, j.edges))


def test_a_from_import_names_its_module_only_past_an_open_package(judged) -> None:
    """`from M import a` needs M and the module M.a: while M's edge is open, M.a's is the same cut."""
    _, _, j = judged
    assert {(e.source, e.target) for e in j.edges if e.source.startswith("verity/pub/from_")} == {
        ("verity/pub/from_pkg.py", "verity/pub/pkg/hidden_mod.py"),
        ("verity/pub/from_priv.py", "tools/priv/privpkg/__init__.py")}


def test_what_the_exports_rewrite_drops_is_cut_not_open(judged) -> None:
    """A path `require`, an extra's entry and a console script are edges the export drops from the manifest, so they count
    as closed by it; a target the file also needs another way stays open."""
    _, _, j = judged
    assert {(e.source, e.kind, e.target) for e in j.cut} == {
        ("verity/pub/lakefile.toml", "lake require", "tools/priv/lakefile.toml"),
        ("verity/pub/pyproject.toml", "python extra", "tools/privbench/pyproject.toml"),
        ("verity/pub/pyproject.toml", "python script", "tools/priv/privpkg/__init__.py")}
    assert not [e for e in j.edges if e.kind in PB.EXPORT_CUTS]
    assert ("verity/pub/pyproject.toml", "python dependency", "tools/priv/pyproject.toml") in {
        (e.source, e.kind, e.target) for e in j.edges}


def test_needed_files_follow_their_entry(judged) -> None:
    """A `needed` file is public only when a public file of a group its entry names needs it, by a kind its entry names,
    and then pulls in the rest of its entry; a `private` glob wins over a public one."""
    _, _, j = judged
    assert {"verity/pub/pyproject.toml", "verity/pub/lakefile.toml", "verity/pub/pkg/__init__.py"} <= j.public.keys()
    assert {"shared/s.py", "shared/u.py", "shared/__init__.py"} <= j.public.keys()
    assert "shared/t.py" not in j.public
    assert ("verity/other/wants.py", "shared/t.py") in {(e.source, e.target) for e in j.edges}
    assert "tools/priv/pyproject.toml" not in j.public, "a structure file pulled by a dependency, not by its package"
    assert "verity/pub/hidden.py" not in j.public


def test_a_test_is_public_only_when_all_it_needs_is(judged) -> None:
    _, _, j = judged
    assert j.public.get("tests/test_ok.py") == "when closed"
    assert "tests/test_bad.py" not in j.public
    assert not [e for e in j.edges if e.source.startswith("tests/")]


def test_the_command_refuses_a_new_edge_and_only_shrinks(judged, capsys) -> None:
    root, _, j = judged
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", *TREE], cwd=root, check=True)
    args = ["--root", str(root), "--boundary", str(root / "boundary.toml"), "--allowed", str(root / "allowed.txt")]
    assert PB.main(args) == 1
    out = capsys.readouterr().out
    assert PB.FIX in out and "verity/pub/Pub.lean:1: lean import Priv -> tools/priv/Priv.lean" in out
    assert PB.main([*args, "--record"]) == 0
    assert PB.read_allowed(root / "allowed.txt") == sorted(e.key for e in j.edges)
    assert PB.main(args) == 0
    (root / "verity/pub/Pub.lean").write_text("-- no import\n", encoding="utf-8")
    assert PB.main([*args, "--prune"]) == 0
    assert "verity/pub/Pub.lean -> tools/priv/Priv.lean" not in PB.read_allowed(root / "allowed.txt")
    (root / "verity/pub/Pub.lean").write_text("import Priv\n", encoding="utf-8")
    capsys.readouterr()
    assert PB.main(args) == 1, "an edge the list dropped can't come back"
    assert "lean import Priv" in capsys.readouterr().out


@pytest.mark.parametrize(("files", "read"), [
    ({"x/lakefile.toml": 'name = "x"\n[[lean_lib]]\nname = "X"\nsrcDir = "y"\n', "x/X.lean": "import Y\n"}, "x/X.lean"),
    ({"x/lakefile.toml": 'name = "x"\n[[lean_lib]]\nname = "X"\n', "x/X.lean": "/- open\nimport Y\n"}, "x/X.lean"),
    ({"x/a.py": "import (\n"}, "x/a.py"),
    ({"x/pyproject.toml": "[project\n", "x/a.py": ""}, "x/a.py"),
])
def test_a_file_it_cannot_read_fails_closed(tmp_path: Path, files: dict[str, str], read: str) -> None:
    """A need it can't read is an error, not a file that needs nothing."""
    for f, text in files.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="can't|isn't"):
        PB.Tree(tmp_path, files=files).edges(read)


@pytest.mark.parametrize(("script", "line"), [
    ("echo \"a\nresearch status\"; research run x\n", 2),
    ("set -e\nresearch keep \"$X\" --owner o \\\n  --why \"y\" > /dev/null 2>&1\n", 2),
])
def test_a_shell_command_over_lines_is_still_read(tmp_path: Path, script: str, line: int) -> None:
    """A line `shlex` refuses alone, one a quote leaves open or a backslash continues, still names its command."""
    files = {"tools/priv/pyproject.toml": TREE["tools/priv/pyproject.toml"], "verity/pub/x.sh": script}
    for f, text in files.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text, encoding="utf-8")
    assert [(e.line, e.target) for e in PB.Tree(tmp_path, files=files).edges("verity/pub/x.sh")] == [
        (line, "tools/priv/pyproject.toml")]


LAKEFILE = '''name = "pub"
srcDir = "src"
defaultTargets = ["Pub", "Agg", "gone-exe", "tool"]

# a dependency the export copies
[[require]]
name = "kept"
path = "../kept"

# one it doesn't
[[require]]
name = "priv"
path = "../../tools/priv"

[[require]]
name = "mathlib"
git = "https://github.com/leanprover-community/mathlib4"
rev = "v4.0.0"

[[lean_lib]]
name = "Pub"
roots = ["Pub.A", "Pub.B",
  "Pub.Gone"]

# an aggregate root the export lacks, over a module it copies
[[lean_lib]]
name = "Agg"

[[lean_lib]]
name = "Nothing"

[[lean_exe]]
name = "gone-exe"
root = "Gone.Main"

[[lean_exe]]
name = "tool"
root = "Tool"

[leanOptions]
autoImplicit = false
'''
LAKE_COPIED = {"verity/kept/lakefile.toml", "verity/pub/src/Pub/A.lean", "verity/pub/src/Pub/B/C.lean",
               "verity/pub/src/Agg/X.lean", "verity/pub/src/Tool.lean"}


def test_a_lakefile_loses_only_what_names_a_file_the_export_does_not_copy() -> None:
    """A root stays while the export copies a module at or below it; a library or executable with none goes, and so does a
    default target whose root file the export lacks, whose modules then build one by one."""
    new, dropped = PE.rewrite_lakefile(LAKEFILE, "verity/pub", LAKE_COPIED)
    assert dropped == [("require", "priv"), ("root", "Pub.Gone"), ("lean_lib", "Nothing"), ("lean_exe", "gone-exe"),
                       ("defaultTarget", "Pub"), ("defaultTarget", "Agg"), ("defaultTarget", "gone-exe")]
    old = tomllib.loads(LAKEFILE)
    assert tomllib.loads(new) == old | {
        "defaultTargets": ["tool"], "require": [old["require"][0], old["require"][2]],
        "lean_lib": [{"name": "Pub", "roots": ["Pub.A", "Pub.B"]}, {"name": "Agg"}], "lean_exe": [old["lean_exe"][1]]}
    assert "# a dependency the export copies" in new and "# an aggregate root" in new
    assert "# one it doesn't" not in new, "a dropped table's comment goes with it"
    assert PE.lean_targets(new, "verity/pub", old, LAKE_COPIED) == {
        "default": ["tool"], "modules": ["Agg.X", "Pub.A", "Pub.B.C"], "missing_roots": ["Pub.B", "Agg"]}


def test_a_manifest_loses_the_path_packages_the_export_does_not_copy() -> None:
    doc = {"version": "1.1.0", "packages": [
        {"type": "path", "name": "kept", "dir": "../kept", "configFile": "lakefile.toml"},
        {"type": "path", "name": "priv", "dir": "../../tools/priv", "configFile": "lakefile.toml"},
        {"type": "path", "name": "via", "dir": "../../tools/priv/../../verity/kept", "configFile": "lakefile.toml",
         "inherited": True},
        {"type": "git", "name": "mathlib", "url": "https://github.com/leanprover-community/mathlib4", "rev": "abc"}]}
    new, changed = PE.rewrite_manifest(json.dumps(doc), "verity/pub", LAKE_COPIED)
    assert changed == [("package", "priv"), ("dir", "via")]
    assert [(p["name"], p.get("dir")) for p in json.loads(new)["packages"]] == [
        ("kept", "../kept"), ("via", "../kept"), ("mathlib", None)]


WORKSPACE = '''[project]
name = "root"
dependencies = ["numpy"]

[tool.uv]
conflicts = [
    [{ extra = "cpu" }, { package = "pubdist", extra = "gpu" }],
    [{ extra = "cpu" }, { package = "privdist", extra = "gpu" }],
]

[tool.uv.workspace]
members = ["verity/pub", "tools/priv"]

[tool.uv.sources]
pubdist = { workspace = true }
privdist = { workspace = true }
torch = { index = "cpu" }

[dependency-groups]
# the dev group
dev = ["pytest>=8", "pubdist", "privdist"]

[tool.pytest.ini_options]
testpaths = ["tests", "tools/priv/tests"]

[tool.verity.tests]
untraced = [
    "tests/test_a.py",
    "tests/test_gone.py",
]
'''


def test_the_workspace_loses_what_names_a_distribution_or_path_the_export_does_not_copy() -> None:
    """A line of a nested array that starts with `[` is no table's header."""
    new, dropped = PE.rewrite_pyproject(WORKSPACE, "", {"pyproject.toml", "verity/pub/pyproject.toml", "tests/test_a.py"},
                                        {"pubdist", "root"}, {"pubdist", "privdist", "root"})
    assert sorted(dropped) == sorted([
        ("conflict", json.dumps([{"extra": "cpu"}, {"package": "privdist", "extra": "gpu"}])), ("member", "tools/priv"),
        ("source", "privdist"), ("dependency-groups.dev", "privdist"), ("testpaths", "tools/priv/tests"),
        ("untraced", "tests/test_gone.py")])
    old = tomllib.loads(WORKSPACE)
    assert tomllib.loads(new) == old | {
        "tool": {"uv": {"conflicts": old["tool"]["uv"]["conflicts"][:1], "workspace": {"members": ["verity/pub"]},
                        "sources": {"pubdist": {"workspace": True}, "torch": {"index": "cpu"}}},
                 "pytest": {"ini_options": {"testpaths": ["tests"]}}, "verity": {"tests": {"untraced": ["tests/test_a.py"]}}},
        "dependency-groups": {"dev": ["pytest>=8", "pubdist"]}}
    assert "# the dev group" in new


MEMBER = '''[project]
name = "pubdist"
dependencies = ["verity", "hardpriv"]

[project.optional-dependencies]
torch = ["torch>=2.4"]
bench = ["privdist"]

[tool.uv.sources]
verity = { workspace = true }
hardpriv = { workspace = true }
privdist = { workspace = true }

[tool.verity.tests]
untraced = ["tests/test_kept.py", "tests/test_gone.py"]
'''


def test_a_members_extras_lose_what_the_export_does_not_copy_but_a_dependency_and_its_source_stay() -> None:
    """A dropped extra takes its source with it, or uv would look for the name on the package index; a `[project]
    dependencies` entry is an edge the build fails on, so it and its source stay, at the member and at the root."""
    copied = {"pyproject.toml", "verity/pub/pyproject.toml", "verity/pub/tests/test_kept.py"}
    new, dropped = PE.rewrite_pyproject(MEMBER, "verity/pub", copied, {"pubdist", "verity"},
                                        {"pubdist", "verity", "hardpriv", "privdist"}, {"hardpriv"})
    assert dropped == [("optional-dependencies.bench", "privdist"), ("source", "privdist"), ("untraced", "tests/test_gone.py")]
    old = tomllib.loads(MEMBER)
    assert tomllib.loads(new) == old | {
        "project": old["project"] | {"optional-dependencies": {"torch": ["torch>=2.4"], "bench": []}},
        "tool": {"uv": {"sources": {"verity": {"workspace": True}, "hardpriv": {"workspace": True}}},
                 "verity": {"tests": {"untraced": ["tests/test_kept.py"]}}}}
    root, _ = PE.rewrite_pyproject(WORKSPACE.replace("privdist = { workspace = true }", "hardpriv = { workspace = true }"),
                                   "", {"pyproject.toml"}, set(), {"hardpriv"}, {"hardpriv"})
    assert tomllib.loads(root)["tool"]["uv"]["sources"]["hardpriv"] == {"workspace": True}


@pytest.mark.parametrize("text", [
    '[tool.uv]\nconflicts = [\n    [\n        { package = "privdist" },\n        { extra = "cpu" },\n    ],\n]\n',
    '[dependency-groups]\ndev = ["privdist", { include-group = "x" }]\n',
])
def test_a_rewrite_it_cannot_lay_out_fails_closed(text: str) -> None:
    """A layout the line-wise rewrite would get wrong is an error, never a manifest that reads as something else."""
    with pytest.raises(ValueError, match="public_export"):
        PE.rewrite_pyproject(text, "", {"pyproject.toml"}, set(), {"privdist"})


LISTED = {"pkg/run.sh", "pkg/private.sh", "pkg/m/__init__.py", "pkg/m/gone.py", "pkg/gone/a.py", "pkg/tests/t.py",
          "lib/x.py", "lib/hidden.py"}
COPIED = LISTED - {"pkg/private.sh", "pkg/m/gone.py", "pkg/gone/a.py", "lib/hidden.py"}


def test_which_files_are_lists_of_the_tree_s_paths() -> None:
    assert PE.path_list("pkg/tests/lint/allowlists/p06.json") == ("allowlist", "pkg")
    assert PE.path_list("tests/lint/allowlists/repo_ruff.json") == ("allowlist", "")
    assert PE.path_list("pkg/tests/census_roots.txt") == ("roots", "pkg")
    assert PE.path_list("pkg/tests/dead_code_keep.json") == ("keep", "pkg")
    assert PE.path_list("pkg/tests/lint/test_p06.py") is None
    assert PE.path_list("pkg/attests/census_roots.txt") is None


def test_an_allowlist_loses_the_entries_naming_a_file_the_export_does_not_copy() -> None:
    """An entry names its file from the directory holding the list's `tests/`, or else from the root, as the lint reads it;
    one naming nothing tracked stays, for the lint's own stale-entry test."""
    lines = ['    {"file": "run.sh", "kind": "k", "count": 2},\n', '    {"file": "private.sh", "kind": "k"},\n',
             '    {"file": "private.sh", "kind": "j"},\n', '    {"file": "lib/x.py", "kind": "k"},\n',
             '    {"file": "untracked.py", "kind": "k"},\n', '    {"file": "lib/hidden.py", "kind": "k"}\n']
    text = '{\n  "rule": "r",\n  "entries": [\n' + "".join(lines) + "  ]\n}\n"
    new, dropped = PE.rewrite_path_list(text, "pkg/tests/lint/allowlists/r.json", LISTED, COPIED)
    assert dropped == [("file", "private.sh"), ("file", "lib/hidden.py")]
    assert new == '{\n  "rule": "r",\n  "entries": [\n' + "".join(lines[i] for i in (0, 3)) + lines[4].rstrip(",\n") + "\n  ]\n}\n"
    assert PE.rewrite_path_list(new, "pkg/tests/lint/allowlists/r.json", LISTED, COPIED) == (new, [])


def test_census_roots_lose_the_shells_directories_and_modules_the_export_does_not_copy() -> None:
    text = ("# roots\nrun.sh        # kept\nprivate.sh    # gone\n**/*_negatives.sh\ngone/\ntests/\n"
            "m             # a package\nm.gone\nlib.x\nlib.hidden\nnothing.sh\n")
    new, dropped = PE.rewrite_path_list(text, "pkg/tests/census_roots.txt", LISTED, COPIED)
    assert dropped == [("root", "gone/"), ("root", "lib.hidden"), ("root", "m.gone"), ("root", "private.sh")]
    assert new == "# roots\nrun.sh        # kept\n**/*_negatives.sh\ntests/\nm             # a package\nlib.x\nnothing.sh\n"


def test_the_keep_list_loses_the_modules_the_export_does_not_copy() -> None:
    doc = {"schema": "s", "modules": {"m": "why", "lib.x": "why", "m.gone": "why"}}
    new, dropped = PE.rewrite_path_list(json.dumps(doc, indent=1) + "\n", "pkg/tests/dead_code_keep.json", LISTED, COPIED)
    assert dropped == [("module", "m.gone")]
    assert json.loads(new) == {"schema": "s", "modules": {"m": "why", "lib.x": "why"}}


def test_a_list_it_cannot_lay_out_fails_closed() -> None:
    text = json.dumps({"rule": "r", "entries": [{"file": "run.sh"}, {"file": "private.sh"}]})
    with pytest.raises(ValueError, match="public_export: the rewrite of pkg/tests/lint/allowlists/r.json"):
        PE.rewrite_path_list(text, "pkg/tests/lint/allowlists/r.json", LISTED, COPIED)


def test_the_export_is_one_fresh_commit_of_the_public_set(judged, tmp_path_factory) -> None:
    root, _, _ = judged
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", *TREE], cwd=root, check=True)
    tree, boundary = PE.PB.Tree(root, files=TREE), PE.PB.Boundary.load(root / "boundary.toml")
    j = PE.PB.judge(tree, boundary)
    dest = tmp_path_factory.mktemp("export")
    got = PE.export(tree, boundary, j, dest, {})

    def git(*args: str) -> list[str]:
        return subprocess.run(["git", *args], cwd=dest, capture_output=True, text=True, check=True).stdout.split()

    assert sorted(git("ls-files")) == sorted(j.public)
    assert git("rev-list", "--count", "HEAD") == ["1"]
    assert got["rewrites"] == {"verity/pub/lakefile.toml": ["require priv"],
                               "verity/pub/pyproject.toml": ["optional-dependencies.bench privtool",
                                                             "optional-dependencies.bench privbench", "script pub-run"]}
    assert [e.split(": ", 1)[1] for e in got["edges_cut"]] == [
        "lake require priv -> tools/priv/lakefile.toml", "python extra privbench -> tools/privbench/pyproject.toml",
        "python script pub-run = privpkg:main -> tools/priv/privpkg/__init__.py"]
    assert "pub-run" not in (dest / "verity/pub/pyproject.toml").read_text(encoding="utf-8")
    assert got["lean_targets"]["verity/pub"]["modules"] == ["Pub"]
    assert "priv" not in (dest / "verity/pub/lakefile.toml").read_text(encoding="utf-8")


def test_the_export_refuses_an_edge_counted_cut_that_its_rewrite_kept(judged, tmp_path_factory, monkeypatch) -> None:
    """The boundary counts a lake `require` or an extra's entry closed only because the export drops it; an export that kept
    one fails rather than ship the edge."""
    root, tree, j = judged
    monkeypatch.setattr(PE, "rewrite_lakefile", lambda text, here, copied: (text, []))
    with pytest.raises(ValueError, match="rewrite kept them: .*lake require priv"):
        PE.export(tree, PE.PB.Boundary.load(root / "boundary.toml"), j, tmp_path_factory.mktemp("export"), {})


def test_the_export_ships_the_fetched_fixtures_its_boundary_names(judged, tmp_path_factory, monkeypatch) -> None:
    """A fetched fixture (a registry entry git doesn't track) that the boundary's `fixtures` names is fetched and committed as
    tracked files, a tree's every file; the other public fetched fixtures are listed as not shipped, and a private one isn't
    listed. One the fetch leaves missing, or with a file over a tracked file's limit, is refused by name."""
    root, _, _ = judged
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", *TREE], cwd=root, check=True)
    (root / "boundary.toml").write_text('fixtures = ["verity/pub/data"]\n' + BOUNDARY, encoding="utf-8")
    reg = {"art:" + c * 64: p for c, p in (("1", "verity/pub/data/w.json"), ("2", "verity/pub/data/set"),
                                           ("3", "verity/pub/other.bin"), ("4", "tools/secret/s.json"),
                                           ("5", "verity/pub/imports.py"))}
    (root / "fixtures").mkdir()
    (root / "fixtures" / "artifacts.json").write_text(json.dumps(reg), encoding="utf-8")
    sizes = {"verity/pub/data/w.json": 3, "verity/pub/data/set/a": 1, "verity/pub/data/set/b/c": 2}
    asked: list[list[str]] = []

    def fetch(r: Path, paths: list[str], log: Path) -> str:
        asked.append(paths)
        for f, n in sizes.items():
            (r / f).parent.mkdir(parents=True, exist_ok=True)
            (r / f).write_bytes(b"x" * n)
        return ""

    monkeypatch.setattr(PE, "fetch", fetch)
    tree, boundary = PE.PB.Tree(root, files=TREE), PE.PB.Boundary.load(root / "boundary.toml")
    shipped, unshipped = PE.fixtures(tree, boundary, tmp_path_factory.mktemp("logs"))
    assert asked == [["verity/pub/data/set", "verity/pub/data/w.json"]]
    assert shipped == {"verity/pub/data/set": ["verity/pub/data/set/a", "verity/pub/data/set/b/c"],
                       "verity/pub/data/w.json": ["verity/pub/data/w.json"]}
    assert unshipped == ["verity/pub/other.bin"]
    dest = tmp_path_factory.mktemp("export")
    PE.export(tree, boundary, PE.PB.judge(tree, boundary), dest, shipped)
    listed = subprocess.run(["git", "ls-files"], cwd=dest, capture_output=True, text=True, check=True).stdout.split()
    assert set(sizes) <= set(listed) and (dest / "verity/pub/data/set/b/c").read_bytes() == b"xx"

    sizes["verity/pub/data/set/b/c"] = PE.REPOSITORY.BLOB_LIMIT + 1
    with pytest.raises(ValueError, match=r"public_export: verity/pub/data/set/b/c: over 256 KiB, the most a tracked file holds"):
        PE.fixtures(tree, boundary, tmp_path_factory.mktemp("logs"))
    monkeypatch.setattr(PE, "fetch", lambda r, paths, log: "verity/pub/data/w.json")
    with pytest.raises(RuntimeError, match=r"public_export: verity/pub/data/w.json: fixtures the export ships, missing or stale"):
        PE.fixtures(tree, boundary, tmp_path_factory.mktemp("logs"))


def test_the_export_refuses_a_destination_it_would_overwrite(tmp_path: Path) -> None:
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "f").write_text("", encoding="utf-8")
    for dest, root in ((tmp_path / "in", tmp_path), (tmp_path / "full", tmp_path / "elsewhere")):
        with pytest.raises(SystemExit):
            PE.main([str(dest), "--root", str(root)])
    assert not (tmp_path / "in").exists()


def test_leftovers_are_counted_by_area_and_kind(tmp_path: Path) -> None:
    (tmp_path / "verity/pub/x").mkdir(parents=True)
    (tmp_path / "verity/pub/x/a.py").write_text("# art:abc1, from bc-0123456789 on cursor/proofs-pubsplit-67ca\n",
                                                encoding="utf-8")
    (tmp_path / "verity/pub/x/b.bin").write_bytes(b"\0art:abc1")
    got = PE.scrub(tmp_path, ["verity/pub/x/a.py", "verity/pub/x/b.bin"])
    assert got["by_area"] == {"verity/pub/x": {"art": 1, "agent": 1, "branch": 1}}
    assert got["not_read_binary_or_large"] == ["verity/pub/x/b.bin"]


def test_known_residue_is_reported_by_its_sites_under_its_paths(tmp_path: Path) -> None:
    doc = {"residue": [{"what": "the ledger's field", "pattern": r"\bperiod\b[^\n]*\blane\b", "paths": ["verity/pub"],
                        "until": "the v3 ledger"}]}
    for f, text in {"verity/pub/a.py": '{"period": "p", "lane": "x"}\nlane = 1\n', "verity/other/b.py": "period lane\n"}.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text, encoding="utf-8")
    got = PE.residue(tmp_path, ["verity/pub/a.py", "verity/other/b.py"], PE.PB.Boundary.load(doc))
    assert got == [{"what": "the ledger's field", "until": "the v3 ledger", "sites": 1, "files": {"verity/pub/a.py": 1}}]
    with pytest.raises(ValueError, match="exactly what, pattern, paths and until"):
        PE.PB.Boundary.load({"residue": [{"what": "x", "pattern": "x", "paths": []}]})
    assert [r.what for r in PE.PB.Boundary.load().residue], "this tree's boundary names its known residue"


def test_third_party_pins_are_read_without_fetching(tmp_path: Path) -> None:
    files = {"a/lake-manifest.json": json.dumps({"packages": [
                 {"type": "git", "name": "mathlib", "url": "https://github.com/leanprover-community/mathlib4", "rev": "abc"},
                 {"type": "path", "name": "b", "dir": "../b"}]}),
             "a/setup.sh": "git clone -q https://github.com/succinctlabs/flock.git\ncd flock && git checkout -q b684b12\n",
             "uv.lock": ('version = 1\n[[package]]\nname = "numpy"\nversion = "2.0.0"\nsource = { registry = "https://pypi.org/simple" }\n'
                         '[[package]]\nname = "pub"\nversion = "0.1.0"\nsource = { editable = "verity/pub" }\n')}
    for f, text in files.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text, encoding="utf-8")
    got = PE.licenses_static(tmp_path, sorted(files))
    assert [(p["name"], p["rev"]) for p in got["lean"]] == [("mathlib", "abc")]
    assert got["python_locked"] == ["numpy==2.0.0"]
    assert [(u["url"], u["rev"]) for u in got["upstream"]] == [("https://github.com/succinctlabs/flock", "b684b12")]


@pytest.mark.slow
def test_this_trees_export_rewrites_every_manifest_it_copies(tmp_path: Path) -> None:
    """This tree's export, in a process of its own: each manifest it rewrites reads as the old one less the entries it
    drops, or the export fails. A manifest laid out in a way the rewrite doesn't know fails here, not on the export's day."""
    r = subprocess.run([sys.executable, str(EXPORT), str(tmp_path / "export"), "--report", str(tmp_path / "report.json")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["rewrites"] and report["lean_targets"]


def test_the_build_shares_one_set_of_dependencies_where_pins_agree(tmp_path: Path, monkeypatch) -> None:
    """The package pinning the most goes first and fetches; a later one whose pins agree uses its `.lake/packages`, and
    each builds its default targets and its modules one by one. `elan` and `lake` are stand-ins that record their calls."""
    bin_dir, calls = tmp_path / "elan-bin", tmp_path / "calls"
    bin_dir.mkdir()
    for name in ("elan", "lake"):
        (bin_dir / name).write_text(
            f'#!/bin/sh\necho "{name} $(basename "$PWD") $*" >> {calls}\n'
            'if [ "$1" = env ]; then mkdir -p .lake/packages/mathlib && echo "Apache License" > .lake/packages/mathlib/LICENSE; fi\n',
            encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    monkeypatch.setattr(PE, "ELAN", bin_dir)
    dest = tmp_path / "export"
    pins = {"mathlib": {"type": "git", "name": "mathlib", "rev": "abc"}, "aesop": {"type": "git", "name": "aesop", "rev": "def"}}
    for pkg, names in (("big", ["mathlib", "aesop"]), ("small", ["mathlib"])):
        (dest / pkg).mkdir(parents=True)
        (dest / pkg / "lean-toolchain").write_text("leanprover/lean4:v4.0.0\n", encoding="utf-8")
        (dest / pkg / "lake-manifest.json").write_text(json.dumps({"packages": [pins[n] for n in names]}), encoding="utf-8")
    targets = {"small": {"default": ["S"], "modules": [], "missing_roots": []},
               "big": {"default": [], "modules": ["B.X", "B.Y"], "missing_roots": ["B"]}}
    got = PE.build(dest, tmp_path / "logs", targets)
    assert [(b["package"], b["ok"], b["shares_packages"]) for b in got["lean"]] == [("big", True, False), ("small", True, True)]
    assert (dest / "small" / ".lake" / "packages").resolve() == (dest / "big" / ".lake" / "packages").resolve()
    assert calls.read_text(encoding="utf-8").splitlines() == [
        "elan big toolchain install leanprover/lean4:v4.0.0", "lake big env true", "lake big exe cache get",
        "lake big build +B.X +B.Y", "lake small env true", "lake small build S"]
    assert got["licenses"]["lean"] == {"mathlib": "Apache-2.0 (LICENSE)"}
