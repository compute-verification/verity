"""The public monorepo's boundary (`tools/verity/check/public_boundary.py`): no file the one-way export copies needs a file
it doesn't, beyond the edges `public_boundary_allowed.txt` lists, which only shrinks; and the export's dry run
(`public_export.py`), whose manifests lose only the entries naming what it doesn't copy."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
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


#: the vLLM integration's Pearl-C path, and the kernel directories a run's `kernel` setting names to it
PEARL_C = ROOT / "adapters" / "vllm" / "verity_vllm" / "protocols" / "pouw" / "pearl_c"
PEARL_C_KERNELS = ("benchmarks/pouw/pearl_c_sm120", "benchmarks/pouw/pearl_c_sm90")
INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]+"([^"]+)"', re.MULTILINE)


def _loaded(module: Path) -> set[str]:
    """The files a module's `_load` calls take from a kernel directory: each call's ``"NAME.py"``."""
    calls = [n for n in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_load"]
    return {c.value for n in calls for c in ast.walk(n)
            if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.endswith(".py")}


def test_the_pearl_c_kernels_the_integration_loads_are_public() -> None:
    """`verity_vllm.protocols.pouw.pearl_c.kernel` loads ``run.py``, ``fixture.py`` and the cubin from the directory a run's
    ``kernel`` setting names, and `pearl_c.grouped` the ``grouped.py`` beside that ``run.py``: paths a run gives, which the
    check reads no edge for. So a `public` glob covers each kernel directory's loaded files and what builds its cubin:
    ``build.sh``, the source it compiles and every file that source includes, which the check doesn't read either. What
    those files import, reach and run in turn are edges the check reads."""
    boundary, tree = PB.Boundary.load(), PB.Tree(ROOT)
    always, beside = _loaded(PEARL_C / "kernel.py"), _loaded(PEARL_C / "grouped.py")
    assert {"run.py", "fixture.py"} <= always and beside, (
        f"pearl_c's `_load` calls name {sorted(always)} and {sorted(beside)}: read here what it loads as it names it now")
    needs: set[str] = set()
    for d in PEARL_C_KERNELS:
        needs |= {f"{d}/{n}" for n in always} | {f"{d}/{n}" for n in beside if f"{d}/{n}" in tree.files} | {f"{d}/build.sh"}
        todo = [f"{d}/{n}" for n in re.findall(r"\$HERE/([\w.-]+\.cu)\b", tree.text(f"{d}/build.sh"))]
        assert todo, f"{d}/build.sh compiles no `$HERE/NAME.cu`: read here the source its cubin is built from"
        while todo:
            f = todo.pop()
            if f in tree.files and f not in needs:
                todo += [PB._rel(os.path.join(os.path.dirname(f), i)) for i in INCLUDE.findall(tree.text(f))]
            needs.add(f)
    private = sorted(f for f in needs if f not in tree.files or boundary.is_private(f) or boundary.group(f) is None)
    assert not private, ("the public vLLM integration serves Pearl-C from files the export doesn't copy; name each in "
                         f"public_boundary.toml's `benchmarks` group (a boundary decision), or stop loading it: {private}")


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


#: a public distribution that depends on two that stay private: one for its private files alone, one a public file uses
DEPENDS = {
    "tools/t/pyproject.toml": ('[project]\nname = "tool"\ndependencies = [\n    "store",\n    "keys>=1",\n]\n'
                            '[tool.uv.sources]\nstore = { workspace = true }\nkeys = { workspace = true }\n'),
    "tools/t/check.py": "import keyspkg\n",
    "tools/t/setup.sh": '#!/bin/bash\nif [ -f tools/t/run.py ]; then python3 tools/t/run.py; fi\n',
    "tools/t/run.py": "import storepkg\n",
    "store/pyproject.toml": '[project]\nname = "store"\n[tool.hatch.build.targets.wheel]\npackages = ["storepkg"]\n',
    "store/storepkg/__init__.py": "",
    "keys/pyproject.toml": '[project]\nname = "keys"\n[tool.hatch.build.targets.wheel]\npackages = ["keyspkg"]\n',
    "keys/keyspkg/__init__.py": "",
}
DEPENDS_BOUNDARY = """
[public]
a = ["tools/t/{check.py,setup.sh}"]
[private]
team = ["store", "keys"]
[[needed]]
why = "structure"
paths = ["**/pyproject.toml"]
[[optional]]
edge = "tools/t/setup.sh -> tools/t/run.py"
why = "the script runs it only where it is"
"""


def _depends(root: Path, boundary: str = DEPENDS_BOUNDARY):
    for f, text in DEPENDS.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text, encoding="utf-8")
    tree, boundary = PB.Tree(root, files=DEPENDS), PB.Boundary.load(tomllib.loads(boundary))
    return tree, boundary, PB.judge(tree, boundary)


def test_a_dependency_no_public_file_uses_is_cut_and_one_a_public_file_uses_stays_open(tmp_path: Path) -> None:
    """A pyproject's dependency on a distribution that isn't public is closed by the export while no public file of its
    own distribution reaches a file of that one: the private files it is declared for stay behind. One a public file
    uses stays open, beside that file's own edge."""
    _, _, j = _depends(tmp_path)
    assert [(e.source, e.kind, e.what) for e in j.cut] == [("tools/t/pyproject.toml", "python dependency", "store")]
    assert {(e.source, e.kind, e.target) for e in j.edges} == {
        ("tools/t/pyproject.toml", "python dependency", "keys/pyproject.toml"),
        ("tools/t/check.py", "python import", "keys/keyspkg/__init__.py")}


def test_the_export_drops_a_dependency_no_public_file_uses_with_its_source(tmp_path_factory) -> None:
    root = tmp_path_factory.mktemp("tree")
    tree, boundary, j = _depends(root)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", *DEPENDS], cwd=root, check=True)
    dest = tmp_path_factory.mktemp("export")
    got = PE.export(tree, boundary, j, dest, {})
    assert got["rewrites"] == {"tools/t/pyproject.toml": ["dependency store", "source store"]}
    assert tomllib.loads((dest / "tools/t/pyproject.toml").read_text(encoding="utf-8")) == {
        "project": {"name": "tool", "dependencies": ["keys>=1"]}, "tool": {"uv": {"sources": {"keys": {"workspace": True}}}}}
    assert [e.split(": ", 1)[1] for e in got["edges_cut"]] == ["python dependency store -> store/pyproject.toml"]


def test_the_export_refuses_a_dependency_counted_cut_that_its_rewrite_kept(tmp_path_factory, monkeypatch) -> None:
    """The boundary counts a dependency no public file uses closed only because the export drops it from the
    pyproject; an export whose rewrite kept it fails rather than ship the edge."""
    root = tmp_path_factory.mktemp("tree")
    tree, boundary, j = _depends(root)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", *DEPENDS], cwd=root, check=True)
    monkeypatch.setattr(PE, "rewrite_pyproject", lambda text, *rest: (text, []))
    with pytest.raises(ValueError, match="rewrite kept them: .*python dependency store"):
        PE.export(tree, boundary, j, tmp_path_factory.mktemp("export"), {})


def test_the_export_writes_a_controls_file_that_only_loses_a_policys_lane_fields(tmp_path_factory) -> None:
    """A controls file in which no control goes and a control's `policy` names an owner is still written anew: the
    export drops nothing it would report, and the file it ships carries no owner."""
    root = tmp_path_factory.mktemp("tree")
    controls = {"listed": {"policy": {"guarantees": {"P.t": {"owner": "@lane"}}}, "fails": "open: P.S is"},
                "plain": {"write": {"A.lean": "x"}, "fails": "sorry: "}}
    files = {f: json.dumps(controls, indent=1) + "\n" for f in PE.CONTROLS[:1]}
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", *files], cwd=root, check=True)
    tree, boundary = PB.Tree(root, files=files), PB.Boundary.load(tomllib.loads('[public]\na = ["tools"]\n'))
    dest = tmp_path_factory.mktemp("export")
    got = PE.export(tree, boundary, PB.judge(tree, boundary), dest, {})
    (f,) = files
    assert f not in got["rewrites"]
    assert json.loads((dest / f).read_text(encoding="utf-8")) == {
        "listed": {"policy": {"guarantees": {"P.t": {}}}, "fails": "open: P.S is"}, "plain": controls["plain"]}


def test_an_optional_edge_is_not_open_and_each_is_one_edge_with_its_reason(tmp_path: Path) -> None:
    """An edge the boundary lists under `[[optional]]` is one its source runs without, so it is not open; the judgment
    names those the tree has. Without the entry the edge is open."""
    _, _, j = _depends(tmp_path)
    assert [str(e) for e in j.optional] == ["tools/t/setup.sh:2: path tools/t/run.py -> tools/t/run.py"]
    assert "tools/t/setup.sh -> tools/t/run.py" not in {e.key for e in j.edges}
    _, _, bare = _depends(tmp_path, DEPENDS_BOUNDARY.partition("[[optional]]")[0])
    assert "tools/t/setup.sh -> tools/t/run.py" in {e.key for e in bare.edges} and bare.optional == []
    for entry in ({"edge": "tools/t/setup.sh -> tools/t/run.py"}, {"edge": "tools/t/setup.sh", "why": "w"},
                  {"edge": "a -> b", "why": " "}, {"edge": "a -> b", "why": "w", "until": "x"}):
        with pytest.raises(ValueError, match="optional"):
            PB.Boundary.load({"optional": [entry]})


def test_every_optional_edge_of_the_boundary_is_one_this_tree_has() -> None:
    """A listed optional edge the tree no longer has is a line to delete: the list states what is true now."""
    boundary, tree = PB.Boundary.load(), PB.Tree(ROOT)
    stale = [key for key in boundary.optional
             if key not in {e.key for e in tree.edges(key.partition(" -> ")[0])}]
    assert not stale, f"`[[optional]]` edges no file has; delete them from public_boundary.toml: {stale}"


def test_the_lean_checks_controls_ship_for_an_unowned_lock() -> None:
    """A control's `policy` is laid over its fixture's lock, which the export ships with no owner: the export strips each
    `policy` of them too, drops a control that needs an owned lock, and leaves a file with neither as it stands."""
    controls = {"plain": {"write": {"A.lean": "x"}, "fails": "sorry: "},
                "listed": {"policy": {"guarantees": {"P.t": {"owner": "@lane"}}, "open_seed": {}}, "fails": "open: P.S is"},
                "ticket": {"policy": {"open": {"P.S": {"due": "soon"}}}, "fails": "open: P.S's entry", "owned": True}}
    text = json.dumps(controls, indent=1) + "\n"
    new, dropped = PE.rewrite_controls(text)
    assert dropped == [("control", "ticket")]
    assert json.loads(new) == {"plain": controls["plain"], "listed": {"policy": {"guarantees": {"P.t": {}}}, "fails": "open: P.S is"}}
    plain = json.dumps({"plain": controls["plain"]}) + "\n"
    assert PE.rewrite_controls(plain) == (plain, [])
    spelled = {"spelled": {"fails": 'open: list it, writing "P.S": {"owner": "@lane", "digest": '}}
    with pytest.raises(ValueError, match="the controls spelled spells an owner"):
        PE.rewrite_controls(json.dumps(spelled))
    assert {f"tools/verity/lean/tests/fixtures/{f}" for f in PE.LEAN_CHECK.CONTROLS.values()} == set(PE.CONTROLS)
    for f in PE.CONTROLS:
        assert (ROOT / f).is_file(), f


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


def test_a_path_anchored_on_a_package_is_an_edge(tmp_path: Path, capsys) -> None:
    """`resources.files("pkg")` is the directory an import of `pkg` from the file finds (a package's own, the one holding a
    module, a namespace package's itself), so a chain on it that reaches a directory the export doesn't copy is an edge:
    refused while the list lacks it, allowed once it is listed."""
    files = {"verity/pub/pkg/__init__.py": "",
             "verity/pub/pkg/config.py": ('from importlib import resources\nfrom pathlib import Path\n\n'
                                          'ROOT = Path(str(resources.files("pkg"))).resolve().parent\n'
                                          'SESSION = ROOT.parent.parent / "tools" / "priv" / "session"\n'),
             "tools/priv/session/run.py": ""}
    for f, text in files.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text, encoding="utf-8")
    (tmp_path / "boundary.toml").write_text(BOUNDARY, encoding="utf-8")
    tree, edge = PB.Tree(tmp_path, files=files), "verity/pub/pkg/config.py:5: path tools/priv/session -> tools/priv/session"
    assert [tree._package(n, "verity/pub/pkg/config.py") for n in ("pkg", "pkg.config", "tools.priv.session", "pkg.gone",
                                                                 "json")] == ["verity/pub/pkg", "verity/pub/pkg",
                                                                              "tools/priv/session", None, None]
    assert [str(e) for e in PB.judge(tree, PB.Boundary.load(tmp_path / "boundary.toml")).edges] == [edge]
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", *files], cwd=tmp_path, check=True)
    args = ["--root", str(tmp_path), "--boundary", str(tmp_path / "boundary.toml"), "--allowed", str(tmp_path / "allowed.txt")]
    assert PB.main(args) == 1
    assert edge in capsys.readouterr().out
    (tmp_path / "allowed.txt").write_text("verity/pub/pkg/config.py -> tools/priv/session\n", encoding="utf-8")
    assert PB.main(args) == 0


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


class _Texts:
    """A tree as `generated_roots` reads it: its files' texts."""

    def __init__(self, files: dict[str, str]):
        self.files = files

    def text(self, f: str) -> str:
        return self.files[f]


def test_a_root_file_the_export_lacks_is_the_imports_of_the_modules_below_it() -> None:
    """The export writes a library's root only where it copies modules below it and not the file: every such module but
    those the package's lock exempts, as a `module` when each of them is one. The lakefile then keeps its default target.
    A root only the lock names is written the same way, since the Lean check builds each of the lock's roots by name."""
    lakefile = ('name = "p"\nsrcDir = "src"\ndefaultTargets = ["Agg", "Mixed"]\n[[lean_lib]]\nname = "Agg"\n'
                '[[lean_lib]]\nname = "Has"\n[[lean_lib]]\nname = "Mixed"\nroots = ["Mixed.In"]\n')
    files = {
        "p/lakefile.toml": lakefile,
        "p/lean-audit.json": json.dumps({"roots": ["Agg", "Has", "Has.Sub", "Has.None"], "exempt": {"Agg.Probe": "a probe"}}),
        "p/src/Agg/A.lean": "module\n\npublic import Agg.B.C\n",
        "p/src/Agg/B/C.lean": "/- a comment -/\nmodule\n",
        "p/src/Agg/Probe/Run.lean": "def main : IO Unit := pure ()\n",
        "p/src/AggOther.lean": "module\n",
        "p/src/Has.lean": "import Has.X\n",
        "p/src/Has/X.lean": "def x := 1\n",
        "p/src/Has/Y.lean": "def y := 1\n",
        "p/src/Has/Sub/Z.lean": "module\n",
        "p/src/Mixed/In/X.lean": "module\n",
        "p/src/Mixed/In/Y.lean": "def y := 1\n",
    }
    got = PE.generated_roots(_Texts(files), set(files))
    assert got == {
        "p/src/Agg.lean": ("module\n\npublic import Agg.A\npublic import Agg.B.C\n\n"
                           "/-! The modules under `Agg` that this tree holds. -/\n"),
        "p/src/Has/Sub.lean": "module\n\npublic import Has.Sub.Z\n\n/-! The modules under `Has.Sub` that this tree holds. -/\n",
        "p/src/Mixed/In.lean": ("import Mixed.In.X\nimport Mixed.In.Y\n\n"
                                "/-! The modules under `Mixed.In` that this tree holds. -/\n")}
    for text in got.values():
        assert PE.PB.HEADERS.header(text) is not None
    built = set(files) | set(got)
    assert PE.rewrite_lakefile(lakefile, "p", built) == (lakefile, [])
    targets = PE.lean_targets(lakefile, "p", tomllib.loads(lakefile), built)
    assert targets["default"] == ["Agg", "Mixed"] and targets["missing_roots"] == []
    assert PE.generated_roots(_Texts(files), set(files) - {"p/src/Agg/A.lean", "p/src/Agg/B/C.lean"}).keys() == {
        "p/src/Has/Sub.lean", "p/src/Mixed/In.lean"}, "a root with only exempt modules below it has nothing to import"
    assert PE.generated_roots(_Texts(files), set(files) - {"p/lean-audit.json"}).keys() == {
        "p/src/Agg.lean", "p/src/Mixed/In.lean"}, "without the lock, no root is the lock's"


def test_a_root_whose_modules_are_each_a_root_is_dropped_and_not_written() -> None:
    """A root with no copied file whose copied modules are each named as a root too would import nothing the build
    lacks, and its file would stand under the name of a module the export leaves out: the export writes none, and the
    lakefile and the lock lose the root. A module below it that the lock doesn't name becomes a root of the lock, so the
    Lean check still seals it. A root with one module that is no root is written as before."""
    lakefile = ('name = "p"\ndefaultTargets = ["Lib"]\n[[lean_lib]]\nname = "Lib"\n'
                'roots = ["Lib.Real", "Lib.Real.A", "Lib.Real.B", "Lib.Agg"]\n')
    lock = {"roots": ["Lib.Real", "Lib.Real.A", "Lib.Agg"]}
    files = {
        "p/lakefile.toml": lakefile, "p/lean-audit.json": json.dumps(lock),
        "p/Lib/Real/A.lean": "module\n", "p/Lib/Real/B.lean": "module\n", "p/Lib/Agg/X.lean": "module\n",
    }
    tree, copied = _Texts(files), set(files)
    assert PE.spare_roots(tree, copied) == {"p": {"Lib.Real": ["Lib.Real.A", "Lib.Real.B"]}}
    got = PE.generated_roots(tree, copied)
    assert set(got) == {"p/Lib/Agg.lean"}
    new, dropped = PE.rewrite_lakefile(lakefile, "p", copied | set(got), ["Lib.Real"])
    assert tomllib.loads(new)["lean_lib"][0]["roots"] == ["Lib.Real.A", "Lib.Real.B", "Lib.Agg"]
    assert dropped == [("root", "Lib.Real")]
    targets = PE.lean_targets(new, "p", tomllib.loads(lakefile), copied | set(got))
    assert targets["default"] == ["Lib"] and targets["missing_roots"] == []

    class Has:
        def has(self, here, kind, name):
            return True

    text = PE.LEAN_CHECK.text(lock)
    new, dropped = PE.rewrite_lock(text, "p", Has(), spare={"Lib.Real": ["Lib.Real.A", "Lib.Real.B"]})
    assert json.loads(new)["roots"] == ["Lib.Real.A", "Lib.Agg", "Lib.Real.B"] and dropped == [("roots", "Lib.Real")]
    assert PE.rewrite_lock(text, "p", Has(), spare={"Lib.Other": ["Lib.Other.A"]}) == (text, [])
    files["p/Lib/Real/C.lean"] = "module\n"
    assert PE.spare_roots(_Texts(files), set(files)) == {}
    assert set(PE.generated_roots(_Texts(files), set(files))) == {"p/Lib/Agg.lean", "p/Lib/Real.lean"}


@pytest.mark.parametrize(("text", "alone"), [
    ("", True),
    ("module\n\npublic import A.B\nimport C\n", True),
    ("/- why -/\nmodule\n\npublic import A.B\n\n@[expose] public section\n", True),
    ("import A\n\n/-! The modules under `A`. /- nested -/ -/\n-- a line\npublic section\n/- last -/\n", True),
    ("import A\n\n@[expose] public section\n\nnamespace A\n", False),
    ("import A\n\n@[expose] public section\n\n@[expose] public section\n", False),
    ("import A\n\nsection Named\n", False),
    ("import A\n\n/-- a doc -/\ndef x := 1\n", False),
    ("import A\n\nabbrev TAG : String := \"tag/v1\"\n", False),
    ("import A\n\n/-! left open\n", False),
    ("import\n", False),
])
def test_a_file_of_imports_alone_declares_nothing(text: str, alone: bool) -> None:
    """A file is imports alone when it is its header, comments and module docs, and at most one line opening an unnamed
    section with nothing in it. Anything else may declare something, and a file that doesn't read counts as that."""
    assert PE.imports_alone(text) is alone


def _unsealed(read, files, here: str) -> list[str]:
    """The modules of the package at ``here``, among ``files`` (texts by ``read``), that the Lean check would not seal:
    under a root of its lock, not under its `exempt`, and reached from no root by the imports of ``files``."""
    lock = json.loads(read(f"{here}/lean-audit.json"))
    src = PE._norm(here, PE.tomllib.loads(read(f"{here}/lakefile.toml")).get("srcDir", "."))
    prefix = f"{src}/" if src else ""
    at = {f[len(prefix):].removesuffix(".lean").replace("/", "."): f for f in files
          if f.endswith(".lean") and f.startswith(prefix)}

    def under(names, m: str) -> bool:
        return any(m == n or m.startswith(f"{n}.") for n in names)

    reached, todo = set(), [r for r in lock["roots"] if r in at]
    while todo:
        if (m := todo.pop()) not in reached:
            reached.add(m)
            todo += [i for i in PE.PB.HEADERS.imports(read(at[m])) or [] if i in at]
    return sorted(m for m in at if under(lock["roots"], m) and not under(lock.get("exempt", {}), m) and m not in reached)


def test_no_generated_file_stands_under_the_name_of_a_file_that_declares_something() -> None:
    """The export writes a root's file under a tracked file's name only when that file is imports alone: a file of
    imports must not stand in for a module with content. A root of the lock alone whose tracked file declares something
    is dropped, its modules still built by the root above, so the check seals every copied module as before, and a
    library's refuses the export, naming the file. The module a lock names as its assumptions' is no root, so it is not
    written: the root above imports the modules under it."""
    lakefile = 'name = "p"\n[[lean_lib]]\nname = "Lib"\n'
    files = {
        "p/lakefile.toml": lakefile,
        "p/lean-audit.json": json.dumps({"roots": ["Lib", "Lib.Table"], "assumptions_module": "Lib.Assumptions"}),
        "p/Lib.lean": "module\n\npublic import Lib.Defs\npublic import Lib.Private\n\n@[expose] public section\n",
        "p/Lib/Defs.lean": "module\n",
        "p/Lib/Private.lean": "module\n",
        "p/Lib/Table.lean": "module\n\npublic import Lib.Table.Part\n\nabbrev TAG : String := \"tag/v1\"\n",
        "p/Lib/Table/Part.lean": "module\n",
        "p/Lib/Assumptions.lean": "module\n\npublic import Lib.Assumptions.Hash\n\ndef Cut : Prop := True\n",
        "p/Lib/Assumptions/Hash.lean": "module\n",
    }
    tree = _Texts(files)
    copied = set(files) - {"p/Lib.lean", "p/Lib/Private.lean", "p/Lib/Table.lean", "p/Lib/Assumptions.lean"}
    assert PE.generated_roots(tree, copied) == {
        "p/Lib.lean": ("module\n\npublic import Lib.Assumptions.Hash\npublic import Lib.Defs\n"
                       "public import Lib.Table.Part\n\n/-! The modules under `Lib` that this tree holds. -/\n")}
    assert PE.spare_roots(tree, copied) == {"p": {"Lib.Table": ["Lib.Table.Part"]}}

    class Has:
        def has(self, here, kind, name):
            return True

    new, dropped = PE.rewrite_lock(PE.LEAN_CHECK.text(json.loads(files["p/lean-audit.json"])), "p", Has(),
                                   spare={"Lib.Table": ["Lib.Table.Part"]})
    assert json.loads(new)["roots"] == ["Lib"] and dropped == [("roots", "Lib.Table")]
    assert json.loads(new)["assumptions_module"] == "Lib.Assumptions"
    wrote = {f: files[f] for f in copied} | PE.generated_roots(tree, copied) | {"p/lean-audit.json": new}
    assert _unsealed(wrote.get, set(wrote), "p") == []
    assert _unsealed((wrote | {"p/Lib.lean": "module\n\npublic import Lib.Defs\n"}).get, set(wrote), "p") == [
        "Lib.Assumptions.Hash", "Lib.Table.Part"], "a root above that didn't import them would leave them unsealed"
    files["p/Lib.lean"] += "\ndef x := 1\n"
    with pytest.raises(ValueError, match=r"public_export: p/Lib\.lean: .* more than imports and comments"):
        PE.generated_roots(_Texts(files), copied)


def test_a_module_only_files_the_export_leaves_out_import_is_a_root_of_the_lock_it_writes() -> None:
    """The Lean check seals the modules under a lock's roots and builds only what the roots import, so a copied module
    that only files the export leaves out import would have no build: the export lists it as a root. A module the lock
    exempts, one outside its roots, one a root reaches, one a listed module imports and one no file imports are not
    listed, and a lock with none is left as it is."""
    files = {
        "p/lakefile.toml": 'name = "p"\nsrcDir = "src"\n[[lean_lib]]\nname = "Lib"\n',
        "p/src/Lib.lean": "module\n\npublic import Lib.A\n",
        "p/src/Lib/A.lean": "module\n\npublic import Lib.B\npublic import Other.X\n",
        "p/src/Lib/B.lean": "module\n",
        "p/src/Lib/Leaf.lean": "module\n\npublic import Lib.Leaf.Below\n",
        "p/src/Lib/Leaf/Below.lean": "module\n",
        "p/src/Lib/Alone.lean": "module\n",
        "p/src/Lib/Probe/Run.lean": "def main : IO Unit := pure ()\n",
        "p/src/Other/X.lean": "module\n",
        "p/src/Lib/Hidden.lean": "module\n\npublic import Lib.Leaf\npublic import Lib.Leaf.Below\npublic import Lib.Probe.Run\n",
        "q/Lib/Elsewhere.lean": "module\n",
    }
    tracked, lock = set(files), {"roots": ["Lib"], "exempt": {"Lib.Probe": "a probe"}}
    copied = tracked - {"p/src/Lib/Hidden.lean"}
    assert PE.unreached(files.get, copied, "p", lock, tracked) == ["Lib.Leaf"]
    assert PE.unreached(files.get, copied, "p", lock | {"roots": ["Lib", "Lib.Leaf"]}, tracked) == []
    assert PE.unreached(files.get, copied - {"p/src/Lib/A.lean"}, "p", lock, tracked) == ["Lib.B", "Lib.Leaf"]
    assert PE.unreached(files.get, tracked, "p", lock, tracked) == [], "nothing is left out, so nothing is cut off"
    assert PE.unreached(files.get, copied, "q", lock, tracked) == [], "no lakefile, so no package to read"

    class Has:
        def has(self, here, kind, name):
            return True

    def more(out):
        return PE.unreached(files.get, copied, "p", out, tracked)

    text = PE.LEAN_CHECK.text(lock)
    new, dropped = PE.rewrite_lock(text, "p", Has(), more)
    assert json.loads(new) == lock | {"roots": ["Lib", "Lib.Leaf"]} and dropped == []
    assert PE.rewrite_lock(new, "p", Has(), more) == (new, [])


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
    monkeypatch.setattr(PE, "rewrite_lakefile", lambda text, here, copied, spare=(): (text, []))
    with pytest.raises(ValueError, match="rewrite kept them: .*lake require priv"):
        PE.export(tree, PE.PB.Boundary.load(root / "boundary.toml"), j, tmp_path_factory.mktemp("export"), {})


LOCK_BOUNDARY = """
when_closed = ["tests"]
[public]
a = ["lean"]
[private]
team = ["lean/Lib/Hidden.lean", "lean/Lib/Hidden/Deep.lean"]
"""

#: a Lake package the export copies less `Lib.Hidden` and one of the two modules under it
LOCK_TREE = {
    "lean/lakefile.toml": 'name = "lib"\n[[lean_lib]]\nname = "Lib"\n',
    "lean/Lib.lean": "import Lib.Shown\n",
    "lean/Lib/Shown.lean": ("namespace Lib\nstructure Shape where\n  side : Nat\ntheorem Kept : True := trivial\n"
                            "theorem _root_.Lib.Rooted : True := trivial\ndef walk : Nat := go 0\n"
                            "where go (n : Nat) : Nat := n\nend Lib\n"),
    "lean/Lib/Hidden.lean": ("namespace Lib\ntheorem Withheld : True := trivial\ndef walk.aside : Nat := 0\n"
                             "theorem Twice : True := trivial\nend Lib\n"),
    "lean/Lib/Hidden/Deep.lean": "namespace Lib.Inner\ntheorem Deep : True := trivial\nend Lib.Inner\n",
    "lean/Lib/Hidden/Open.lean": "namespace Lib\ntheorem Open : True := trivial\ntheorem Twice : True := trivial\nend Lib\n",
}


def _record(**more) -> dict:
    return {"owner": "@lane", "signature": [], "assumptions": [],
            "premises": {"assumptions": [], "domain": [], "open": [], "reads": []}, "type_hash": "00"} | more


def _lock() -> dict:
    return {
        "format": 3,
        "roots": ["Lib"],
        "exempt": {"Lib.Hidden": "its runners", "Lib.Hidden.Deep": "a runner"},
        "runs": {"Lib.Hidden": {"why": "vectors"}, "Lib.Hidden.Open": {"why": "vectors"}},
        "open": {"Lib.Kept": {"owner": "@lane", "digest": "01", "issue": "LANE-1", "due": "2026-01-01"},
                 "Lib.Withheld": {"owner": "@lane", "digest": "02", "issue": "LANE-2", "due": "2026-01-02"}},
        "default_owner": "@lane",
        "escapes": {"Lib.walk.go": "its loop", "Lib.walk.aside": "a partial def"},
        "layers": {"Lib": {"imports": ["Lib", "Mathlib", "Lib.Hidden", "Lib.Hidden.Deep"],
                           "why": "all of it"}},
        "four_names": ["Lib", "Lib.Inner"],
        "four_names_except": {"Lib.Twice": "named before the rule"},
        "guarantees": {"Lib.Rooted": _record(premises={"assumptions": [], "domain": [], "open": ["Lib.Kept"],
                                                       "reads": ["Lib.Shape.mk"]}),
                       "Lib.Withheld": _record(premises={"assumptions": [], "domain": [], "open": ["Lib.Withheld"],
                                                         "reads": []}),
                       "Lib.Outgrown": _record()},
        "reads": {"Lib.Shown": {"digest": "03", "definitions": {"Lib.Shape": "04", "Lib.Shape.casesOn": "05"}},
                  "Lib.Hidden": {"digest": "06", "definitions": {"Lib.walk.aside": "07"}},
                  "Lib.Hidden.Deep": {"digest": "08", "definitions": {}}},
    }


def _lean(root: Path, lock: dict, boundary: str = LOCK_BOUNDARY):
    """The tree of `LOCK_TREE` with ``lock`` as its package's, tracked, and its judgment under ``boundary``."""
    files = LOCK_TREE | {"lean/lean-audit.json": PE.LEAN_CHECK.text(lock)}
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text, encoding="utf-8")
    (root / "boundary.toml").write_text(boundary, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", *files], cwd=root, check=True)
    tree, loaded = PE.PB.Tree(root, files=files), PE.PB.Boundary.load(root / "boundary.toml")
    return tree, loaded, PE.PB.judge(tree, loaded)


def test_the_export_writes_a_lock_that_states_only_what_it_copies(tmp_path_factory) -> None:
    """An entry stays only if a copied file declares its name, is its module, or lies under its prefix, and none carries
    its owner, issue or due date; the rest of the lock reads as it did."""
    tree, boundary, j = _lean(tmp_path_factory.mktemp("tree"), _lock())
    assert "lean/Lib/Hidden.lean" not in j.public and "lean/Lib/Hidden/Deep.lean" not in j.public
    assert {"lean/Lib/Hidden/Open.lean", "lean/lean-audit.json"} <= set(j.public)
    dest = tmp_path_factory.mktemp("export")
    got = PE.export(tree, boundary, j, dest, {})
    wrote = (dest / "lean/lean-audit.json").read_text(encoding="utf-8")
    source = _lock()
    public = {
        "format": 3,
        "roots": ["Lib"],
        "exempt": {"Lib.Hidden": "its runners"},
        "runs": {"Lib.Hidden.Open": {"why": "vectors"}},
        "open": {"Lib.Kept": {"digest": "01"}},
        "escapes": {"Lib.walk.go": "its loop"},
        "layers": {"Lib": {"imports": ["Lib", "Mathlib", "Lib.Hidden"], "why": "all of it"}},
        "four_names": ["Lib"],
        "four_names_except": {"Lib.Twice": "named before the rule"},
        "guarantees": {"Lib.Rooted": {k: v for k, v in source["guarantees"]["Lib.Rooted"].items() if k != "owner"}},
        "reads": {"Lib.Shown": source["reads"]["Lib.Shown"]},
    }
    assert json.loads(wrote) == public
    assert wrote == PE.LEAN_CHECK.text(public)
    for word in ("Withheld", "Deep", "Outgrown", "aside", "Inner", "@lane", "LANE-", "2026-01"):
        assert word not in wrote
    assert got["rewrites"]["lean/lean-audit.json"] == [
        "exempt Lib.Hidden.Deep", "runs Lib.Hidden", "open Lib.Withheld", "escapes Lib.walk.aside",
        "layers.*.imports Lib.Hidden.Deep", "four_names Lib.Inner", "guarantees Lib.Withheld", "guarantees Lib.Outgrown",
        "reads Lib.Hidden", "reads Lib.Hidden.Deep"]
    report = got["locks"]["lean/lean-audit.json"]
    assert report["kept"] == {"roots": 1, "exempt": 1, "runs": 1, "open": 1, "escapes": 1, "layers": 1,
                              "layers.*.imports": 3, "four_names": 1, "four_names_except": 1, "guarantees": 1,
                              "reads": 1}
    assert report["dropped"] == {"exempt": 1, "runs": 1, "open": 1, "escapes": 1, "layers.*.imports": 1, "four_names": 1,
                                 "guarantees": 2, "reads": 2}
    assert report["undeclared"] == ["guarantees: Lib.Outgrown"] and report["witness_withheld"] == []
    assert report["also_withheld"] == ["four_names_except: Lib.Twice"]
    assert report["lane_fields"] == 3 + 2 * 3 + 1
    assert report["sites"] == {} and report["site_list"] == []
    assert report["spelled"] == {} and report["spelled_list"] == []
    assert report["bytes"] == [len(PE.LEAN_CHECK.text(source)), len(wrote)]


def test_the_export_writes_the_root_file_its_boundary_leaves_out(tmp_path_factory) -> None:
    """A library whose root file is private is still one the export's lock can name as a root: the export writes the file,
    commits it with what it copied and reports it, and no root is missing."""
    private = LOCK_BOUNDARY.replace('team = [', 'team = ["lean/Lib.lean", ')
    tree, boundary, j = _lean(tmp_path_factory.mktemp("tree"), _lock(), private)
    assert "lean/Lib.lean" not in j.public and "lean/Lib/Shown.lean" in j.public
    dest = tmp_path_factory.mktemp("export")
    got = PE.export(tree, boundary, j, dest, {})
    assert (dest / "lean/Lib.lean").read_text(encoding="utf-8") == (
        "import Lib.Shown\n\n/-! The modules under `Lib` that this tree holds. -/\n"), "less what the lock exempts"
    assert got["generated_roots"] == ["lean/Lib.lean"]
    assert got["lean_targets"]["lean"] == {"default": [], "modules": ["Lib", "Lib.Hidden.Open", "Lib.Shown"],
                                           "missing_roots": []}
    assert json.loads((dest / "lean/lean-audit.json").read_text(encoding="utf-8"))["roots"] == ["Lib"]
    tracked = subprocess.run(["git", "ls-files"], cwd=dest, capture_output=True, text=True, check=True).stdout.split()
    assert sorted(tracked) == sorted([*j.public, "lean/Lib.lean"])


def test_the_export_leaves_a_lock_that_is_public_as_it_stands(tmp_path_factory) -> None:
    lock = {"format": 3, "roots": ["Lib"], "layers": {"Lib": {"imports": ["Lib", "Mathlib"], "why": "all of it"}}}
    tree, boundary, j = _lean(tmp_path_factory.mktemp("tree"), lock)
    dest = tmp_path_factory.mktemp("export")
    got = PE.export(tree, boundary, j, dest, {})
    assert (dest / "lean/lean-audit.json").read_bytes() == (tree.root / "lean/lean-audit.json").read_bytes()
    assert "lean/lean-audit.json" not in got["rewrites"]
    assert got["locks"]["lean/lean-audit.json"]["dropped"] == {} and got["locks"]["lean/lean-audit.json"]["roots_added"] == []


def test_the_export_lists_a_module_only_a_file_it_leaves_out_imports_as_a_root(tmp_path_factory, monkeypatch) -> None:
    hidden = "import Lib.Hidden.Open\n" + LOCK_TREE["lean/Lib/Hidden.lean"]
    monkeypatch.setitem(LOCK_TREE, "lean/Lib/Hidden.lean", hidden)
    lock = {"format": 3, "roots": ["Lib"], "layers": {"Lib": {"imports": ["Lib", "Mathlib"], "why": "all of it"}}}
    tree, boundary, j = _lean(tmp_path_factory.mktemp("tree"), lock)
    dest = tmp_path_factory.mktemp("export")
    got = PE.export(tree, boundary, j, dest, {})
    assert json.loads((dest / "lean/lean-audit.json").read_text(encoding="utf-8")) == lock | {"roots": ["Lib", "Lib.Hidden.Open"]}
    assert got["locks"]["lean/lean-audit.json"]["roots_added"] == ["Lib.Hidden.Open"]
    assert got["locks"]["lean/lean-audit.json"]["dropped"] == {}


def test_the_export_refuses_a_lock_it_wrote_that_names_what_no_copied_file_has(tmp_path_factory, monkeypatch) -> None:
    """The export reads back each lock it wrote: one that still names a declaration or a module no copied file has, or
    carries an owner, fails by name rather than ship."""
    tree, boundary, j = _lean(tmp_path_factory.mktemp("tree"), _lock())
    monkeypatch.setattr(PE, "rewrite_lock", lambda text, here, held, more=None, spare=None: (text, []))
    with pytest.raises(ValueError) as e:
        PE.export(tree, boundary, j, tmp_path_factory.mktemp("export"), {})
    said = str(e.value)
    assert "lean/lean-audit.json, as the export wrote it, still names what no copied file has" in said
    for named in ("exempt: Lib.Hidden.Deep", "runs: Lib.Hidden;", "open: Lib.Withheld", "escapes: Lib.walk.aside",
                  "layers[Lib].imports: Lib.Hidden.Deep", "four_names: Lib.Inner", "guarantees: Lib.Withheld",
                  "guarantees: Lib.Outgrown", "reads: Lib.Hidden;", "reads: Lib.Hidden.Deep",
                  "guarantees[Lib.Withheld].premises.open: Lib.Withheld", "guarantees.Lib.Rooted.owner: a lane field",
                  "open.Lib.Kept.due: a lane field", "default_owner: a lane field"):
        assert named in said


def test_the_export_refuses_a_public_guarantee_that_rests_on_a_premise_it_does_not_copy(tmp_path_factory) -> None:
    lock = _lock()
    lock["guarantees"]["Lib.Rooted"]["premises"]["open"] = ["Lib.Withheld"]
    tree, boundary, j = _lean(tmp_path_factory.mktemp("tree"), lock)
    with pytest.raises(ValueError, match=r"cite what it doesn't copy.*guarantees\[Lib.Rooted\].premises.open: Lib.Withheld"):
        PE.export(tree, boundary, j, tmp_path_factory.mktemp("export"), {})


def test_the_export_drops_a_witness_entry_whose_witness_it_does_not_copy(tmp_path_factory) -> None:
    lock = _lock() | {"witnesses": {"Lib.Rooted": {"honest": ["Lib.Kept"], "nontrivial": "Lib.Withheld"}}}
    tree, boundary, j = _lean(tmp_path_factory.mktemp("tree"), lock)
    dest = tmp_path_factory.mktemp("export")
    got = PE.export(tree, boundary, j, dest, {})
    wrote = json.loads((dest / "lean/lean-audit.json").read_text(encoding="utf-8"))
    assert wrote["witnesses"] == {} and "Lib.Rooted" in wrote["guarantees"]
    assert "witnesses.*.nontrivial Lib.Rooted" in got["rewrites"]["lean/lean-audit.json"]
    assert got["locks"]["lean/lean-audit.json"]["dropped"]["witnesses.*.nontrivial"] == 1
    assert got["locks"]["lean/lean-audit.json"]["witness_withheld"] == ["witnesses.*.nontrivial: Lib.Rooted"]


def test_the_export_drops_a_runner_whose_generator_it_does_not_copy(tmp_path_factory) -> None:
    """A run's data is what its `generate` writes, so a runner whose generator names a file the export leaves out goes
    from the lock; its module stays exempt."""
    lock = _lock()
    lock["runs"]["Lib.Hidden.Open"]["generate"] = ["python3", "{repo}/lean/Lib/Hidden.lean", "{repo}", "--out", "{out}"]
    tree, boundary, j = _lean(tmp_path_factory.mktemp("tree"), lock)
    dest = tmp_path_factory.mktemp("export")
    got = PE.export(tree, boundary, j, dest, {})
    wrote = json.loads((dest / "lean/lean-audit.json").read_text(encoding="utf-8"))
    assert wrote["runs"] == {} and wrote["exempt"] == {"Lib.Hidden": "its runners"}
    report = got["locks"]["lean/lean-audit.json"]
    assert report["generator_withheld"] == ["runs.*.generate: Lib.Hidden.Open"] and report["witness_withheld"] == []
    lock["runs"]["Lib.Hidden.Open"]["generate"][1] = "Lib/Shown.lean"
    tree, boundary, j = _lean(tmp_path_factory.mktemp("tree"), lock)
    dest = tmp_path_factory.mktemp("export")
    got = PE.export(tree, boundary, j, dest, {})
    assert json.loads((dest / "lean/lean-audit.json").read_text(encoding="utf-8"))["runs"] == {
        "Lib.Hidden.Open": lock["runs"]["Lib.Hidden.Open"]}
    assert got["locks"]["lean/lean-audit.json"]["generator_withheld"] == []


def _exported(tmp_path_factory):
    """`_lock()`'s tree exported: (its root, the export, the exported lock's path there)."""
    root, dest = tmp_path_factory.mktemp("tree"), tmp_path_factory.mktemp("export")
    PE.export(*_lean(root, _lock()), dest, {})
    return root, dest, "lean/lean-audit.json"


def _update(change, failures: list[tuple[str, str | None]] = (), entries: bool = True):
    """An exported check's update, as `records` runs one: it writes ``change`` of the lock and reports ``failures``, each
    a code and the entry it is about."""
    def run(dest: Path, packages: list[str], out: Path, log: Path) -> dict:
        assert packages == ["lean"]
        at = dest / "lean" / "lean-audit.json"
        lock = json.loads(at.read_text(encoding="utf-8"))
        change(lock)
        at.write_text(PE.LEAN_CHECK.text(lock), encoding="utf-8")
        out.mkdir(parents=True)
        (out / "audit.json").write_text(json.dumps({"packages": [{
            "package": str(dest / "lean"), "failures": [f"{code}: {entry}" for code, entry in failures],
            "failure_codes": [code for code, _ in failures]} | (
                {"failure_entries": [entry for _, entry in failures]} if entries else {})]}), encoding="utf-8")
        return {"exit": 1 if failures else 0, "seconds": 0.0, "log": log.name}
    return run


def test_the_exports_own_check_writes_its_records_and_states_nothing_otherwise(tmp_path_factory) -> None:
    """The exported tree's check writes the records only a build of that tree knows (what a module declares that a
    guarantee reads), and the lock keeps each guarantee and premise as its source states it. A statement the update
    finds no guarantee for is listed as unstated, an entry it reports stale goes, and the export is still one commit,
    holding the locks as written."""
    root, dest, f = _exported(tmp_path_factory)

    def reads(lock: dict) -> None:
        lock["reads"]["Lib.Shown"]["definitions"] = {"Lib.Shape": "04"}
        lock["reads_exempt"] = {"Lib.Inner": "read as it stands", "Lib.Shown": "read as it stands"}

    got = PE.records(root, dest, tmp_path_factory.mktemp("logs"), [f], _update(reads, [
        ("shape.unstated", "Lib.Guarantees.S"), ("reads.exempt-stale", "Lib.Inner")]))
    wrote = json.loads((dest / f).read_text(encoding="utf-8"))
    assert wrote["reads"]["Lib.Shown"]["definitions"] == {"Lib.Shape": "04"}
    assert wrote["shape"]["frozen"] == {"Lib.Guarantees.S": "unstated: no theorem in this tree proves it"}
    assert wrote["reads_exempt"] == {"Lib.Shown": "read as it stands"}
    assert got["unstated"] == {f: ["Lib.Guarantees.S"]} and got["stale"] == {f: {"reads.exempt-stale": ["Lib.Inner"]}}
    log = subprocess.run(["git", "log", "--format=%H"], cwd=dest, capture_output=True, text=True, check=True).stdout.split()
    assert log == [got["commit"]]
    assert subprocess.run(["git", "status", "--porcelain"], cwd=dest, capture_output=True, text=True).stdout == ""


def test_the_export_refuses_a_lock_its_check_wrote_that_restates_a_guarantee(tmp_path_factory) -> None:
    """An update records whatever the build shows, so a guarantee whose statement reads differently on the exported
    tree would ship restated. One planted difference refuses the export, naming the lock, the guarantee and the field."""
    root, dest, f = _exported(tmp_path_factory)

    def restate(lock: dict) -> None:
        lock["guarantees"]["Lib.Rooted"]["type_hash"] = "ff"

    with pytest.raises(ValueError, match=r"1 place\(s\) where a lock the exported tree's check wrote states a guarantee or "
                                         r"a premise otherwise .*: lean/lean-audit.json: guarantees\[Lib.Rooted\]: its "
                                         r"record differs in type_hash"):
        PE.records(root, dest, tmp_path_factory.mktemp("logs"), [f], _update(restate))


def test_the_export_refuses_what_its_records_do_not_settle(tmp_path_factory) -> None:
    """A failure the update leaves that is neither an unstated statement nor a stale entry is one the exported tree's
    Lean check would report, so it refuses the export, by lock, code and entry. So does a report the export can't
    read: none at all, or one without each failure's entry."""
    root, dest, f = _exported(tmp_path_factory)

    def same(lock: dict) -> None:
        pass

    with pytest.raises(ValueError, match=r"1 failure\(s\) of the exported tree's Lean check that its records don't "
                                         r"settle: lean/lean-audit.json: witnesses.missing \(Lib.Rooted\)"):
        PE.records(root, dest, tmp_path_factory.mktemp("logs"), [f],
                   _update(same, [("shape.unstated", "Lib.Guarantees.S"), ("witnesses.missing", "Lib.Rooted")]))
    with pytest.raises(ValueError, match="settle: lean/lean-audit.json: build"):
        PE.records(root, dest, tmp_path_factory.mktemp("logs"), [f], _update(same, [("build", None)]))
    with pytest.raises(ValueError, match="without the code and the entry of each"):
        PE.records(root, dest, tmp_path_factory.mktemp("logs"), [f],
                   _update(same, [("shape.unstated", "Lib.Guarantees.S")], entries=False))
    with pytest.raises(RuntimeError, match="wrote no .*audit.json"):
        PE.records(root, dest, tmp_path_factory.mktemp("logs"), [f],
                   lambda dest, packages, out, log: {"exit": 2, "seconds": 0.0, "log": log.name})


def test_a_lakefile_needs_its_packages_lock(tmp_path: Path) -> None:
    """A Lake package's lock is one of its files: the Lean check of the public tree refuses a package without one, so
    the lakefile needs it as it needs its manifest and its toolchain."""
    files = {"p/lakefile.toml": 'name = "p"\n', "p/lake-manifest.json": "{}\n", "p/lean-toolchain": "t\n",
             "p/lean-audit.json": "{}\n", "q/lakefile.toml": 'name = "q"\n'}
    for f, text in files.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text, encoding="utf-8")
    tree = PE.PB.Tree(tmp_path, files=files)
    assert {(e.kind, e.target) for e in tree.edges("p/lakefile.toml")} == {
        ("package", "p/lake-manifest.json"), ("package", "p/lean-toolchain"), ("package", "p/lean-audit.json")}
    assert tree.edges("q/lakefile.toml") == []


def test_what_the_copied_files_hold_of_the_trees_lean(tmp_path_factory) -> None:
    tree, _, j = _lean(tmp_path_factory.mktemp("tree"), _lock())
    held, L = PE.Held(tree, set(j.public)), PE.PUBLIC_LOCK
    has = {kind: [n for n in names if held.has("lean", kind, n)] for kind, names in {
        L.DECLARATION: ["Lib.Kept", "Lib.Rooted", "Lib.Shape", "Lib.Withheld", "Lib.walk.go", "Lib.Inner.Deep"],
        L.CONSTANT: ["Lib.walk", "Lib.walk.go", "Lib.Shape.mk", "Lib.walk.aside", "Lib.Withheld.go", "Lib.Shape.mk.inner"],
        L.NAMESPACE: ["Lib", "Lib.Shape", "Lib.Inner", "Other"],
        L.MODULE: ["Lib", "Lib.Shown", "Lib.Hidden", "Lib.Hidden.Open", "Lib.Hidden.Deep", "Mathlib"],
        L.PREFIX: ["Lib", "Lib.Shown", "Lib.Hidden", "Lib.Hidden.Open", "Lib.Hidden.Deep", "Mathlib"],
        L.IMPORT: ["Lib", "Lib.Shown", "Lib.Hidden", "Lib.Hidden.Open", "Lib.Hidden.Deep", "Mathlib", "Mathlib.Tactic"],
        L.PACKAGE: [".", "../lean", "../gone"],
        L.ARGUMENT: ["python3", "--out", "{out}", "{repo}", "Lib/Shown.lean", "{repo}/lean/Lib/Shown.lean", "Lib/Hidden",
                     "Lib/Hidden.lean", "{repo}/lean/Lib/Hidden.lean", "Lib/Hidden/Deep.lean", "../lean/Lib/Hidden.lean"],
    }.items()}
    assert has == {
        L.DECLARATION: ["Lib.Kept", "Lib.Rooted", "Lib.Shape"],
        L.CONSTANT: ["Lib.walk", "Lib.walk.go", "Lib.Shape.mk"],
        L.NAMESPACE: ["Lib", "Lib.Shape"],
        L.MODULE: ["Lib", "Lib.Shown", "Lib.Hidden.Open"],
        L.PREFIX: ["Lib", "Lib.Shown", "Lib.Hidden", "Lib.Hidden.Open"],
        L.IMPORT: ["Lib", "Lib.Shown", "Lib.Hidden", "Lib.Hidden.Open", "Mathlib", "Mathlib.Tactic"],
        L.PACKAGE: [".", "../lean"],
        L.ARGUMENT: ["python3", "--out", "{out}", "{repo}", "Lib/Shown.lean", "{repo}/lean/Lib/Shown.lean", "Lib/Hidden"],
    }
    assert [n for n in ("Lib.Outgrown", "Lib.Withheld", "Lib.Kept") if held.undeclared(L.DECLARATION, n)] == ["Lib.Outgrown"]
    assert [m for m in ("Gone", "Lib.Hidden", "Lib") if held.undeclared(L.MODULE, m)] == ["Gone"]
    assert [m for m in ("Gone", "Lib.Hidden.Deep") if held.undeclared(L.PREFIX, m)] == ["Gone"]
    with pytest.raises(ValueError, match="a kind it doesn't know"):
        held.has("lean", "theorem", "Lib.Kept")


def test_every_section_the_lean_check_writes_has_its_rule_for_the_export() -> None:
    """A section added to the lock's schema (`check.ORDER`) with no rule in `public_lock.py` would refuse every lock that
    holds it on the export's day; it fails here first."""
    L = PE.PUBLIC_LOCK
    ruled = {s.partition(".")[0] for s in L.RULED}
    assert not sorted(set(PE.LEAN_CHECK.ORDER) - ruled), "give each its rule in tools/verity/lean/public_lock.py"
    assert ruled <= set(PE.LEAN_CHECK.ORDER) | {"four_names_except", "guarantee_lemmas"}


def test_a_locks_leftovers_are_named_by_section_and_entry() -> None:
    lock = {"format": 3, "exempt": {"Lib.Run": "a runner, since #4321"}, "deployed": ["Lib.G"],
            "shape": {"frozen": {"Lib.G": "frozen: see note:lean/20260101T0000Z-report-x"}, "events": ["Lib.E"]}}
    sites = PE.PUBLIC_LOCK.leftovers(lock, PE.lane_text)
    assert [(section, entry, kind) for section, entry, kind, _ in sites] == [("exempt", "Lib.Run", "pr"),
                                                                             ("shape.frozen", "Lib.G", "note")]
    assert sites[0][3] == "#4321" and sites[1][3].startswith("note:")


def _known(boundary, pattern: str):
    """``boundary`` with the text of the package's lock that ``pattern`` matches as known residue."""
    entry = PE.PB.Residue("known", re.compile(pattern), PE.PB.globs(["lean/lean-audit.json"]), "the change that removes it")
    return boundary._replace(residue=(*boundary.residue, entry))


#: a reason that spells what only the files left out have (a declaration, a module, a file), and what a copied file has
SPELLS = ("runs beside Lib.Withheld and Lib.Hidden.Deep (lean/Lib/Hidden.lean, lean/Lib/Hidden/); Lib.Kept, Lib.Twice, "
          "Lib.Hidden.Open, Mathlib.Tactic and lean/Lib/Shown.lean are another matter.")


def test_the_export_refuses_a_lock_whose_kept_text_names_what_it_leaves_out(tmp_path_factory) -> None:
    """A reason is prose, so the export never rewords it: one that still spells a declaration, a module or a path only
    the files left out have refuses the export by file, section, entry and text, and not for what a copied file has."""
    lock = {"format": 3, "roots": ["Lib"], "exempt": {"Lib.Hidden": SPELLS}, "shape": {"frozen": {"Lib.Kept": "as Lib.Withheld"}}}
    tree, boundary, j = _lean(tmp_path_factory.mktemp("tree"), lock)
    dest = tmp_path_factory.mktemp("export")
    with pytest.raises(ValueError) as e:
        PE.export(tree, boundary, j, dest, {})
    said = str(e.value)
    for named in ("4 place(s)", "lean/lean-audit.json: exempt[Lib.Hidden] holds declaration 'Lib.Withheld'",
                  "lean/lean-audit.json: exempt[Lib.Hidden] holds module 'Lib.Hidden.Deep'",
                  "lean/lean-audit.json: exempt[Lib.Hidden] holds path 'lean/Lib/Hidden.lean'",
                  "lean/lean-audit.json: shape.frozen[Lib.Kept] holds declaration 'Lib.Withheld'",
                  "reword each in the lock itself", "[[residue]]"):
        assert named in said
    for kept in ("Lib.Twice", "Lib.Hidden.Open", "Mathlib.Tactic", "Shown.lean"):
        assert kept not in said
    assert not (dest / ".git").exists(), "a refused export commits nothing"


def test_the_export_refuses_a_lock_whose_kept_text_holds_lane_text(tmp_path_factory) -> None:
    lock = _lock()
    lock["layers"]["Lib"]["why"] = "all of it, since #1234"
    lock["exempt"]["Lib.Hidden"] = "its runners (Daniel, 5 Oct)"
    lock["exempt"]["Lib.Hidden.Deep"] = "a runner, since #99"
    tree, boundary, j = _lean(tmp_path_factory.mktemp("tree"), lock)
    with pytest.raises(ValueError) as e:
        PE.export(tree, boundary, j, tmp_path_factory.mktemp("export"), {})
    said = str(e.value)
    assert "2 place(s)" in said and "lean/lean-audit.json: layers[Lib] holds pr '#1234'" in said
    assert "lean/lean-audit.json: exempt[Lib.Hidden] holds attribution '(Daniel, 5 Oct)'" in said
    assert "#99" not in said, "an entry the export drops is not its to refuse"


def test_the_export_ships_what_the_boundary_lists_as_known_residue_and_reports_it(tmp_path_factory) -> None:
    """A `[[residue]]` entry covers a lock's text by the lock's path and the text its pattern matches: the export ships
    that and reports it, and still refuses what no entry covers."""
    lock = {"format": 3, "roots": ["Lib"], "exempt": {"Lib.Hidden": SPELLS},
            "shape": {"frozen": {"Lib.Kept": "as Lib.Withheld, since #1234"}}}
    tree, boundary, j = _lean(tmp_path_factory.mktemp("tree"), lock)
    with pytest.raises(ValueError) as e:
        PE.export(tree, _known(boundary, r"^Lib\.|^lean/"), j, tmp_path_factory.mktemp("export"), {})
    assert "1 place(s)" in str(e.value) and "shape.frozen[Lib.Kept] holds pr '#1234'" in str(e.value)
    elsewhere = boundary._replace(residue=(PE.PB.Residue("known", re.compile("."), PE.PB.globs(["other"]), "a change"),))
    with pytest.raises(ValueError, match=r"5 place\(s\)"):
        PE.export(tree, elsewhere, j, tmp_path_factory.mktemp("export"), {})
    dest = tmp_path_factory.mktemp("export")
    got = PE.export(tree, _known(boundary, r"^Lib\.|^lean/|^#1234$"), j, dest, {})
    assert json.loads((dest / "lean/lean-audit.json").read_text(encoding="utf-8")) == lock
    report = got["locks"]["lean/lean-audit.json"]
    assert report["spelled"] == {"exempt": 3, "shape.frozen": 1}
    assert report["spelled_list"] == [
        ["exempt", "Lib.Hidden", "declaration", "Lib.Withheld"], ["exempt", "Lib.Hidden", "module", "Lib.Hidden.Deep"],
        ["exempt", "Lib.Hidden", "path", "lean/Lib/Hidden.lean"],
        ["shape.frozen", "Lib.Kept", "declaration", "Lib.Withheld"]]
    assert report["sites"] == {"shape.frozen": 1} and report["site_list"] == [["shape.frozen", "Lib.Kept", "pr", "#1234"]]


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
    drops, or the export fails. A manifest laid out in a way the rewrite doesn't know fails here, not on the export's day.
    Each root file it writes under a tracked file's name replaces a file of imports alone. Where it drops a root of a
    lock (the identity package's `Identity.Domains`, whose file is the other protocols' table and stays behind), the
    check still seals every module it copies there: each is reached from a root the written lock keeps, the modules
    `unreached` lists among them."""
    r = subprocess.run([sys.executable, str(EXPORT), str(tmp_path / "export"), "--report", str(tmp_path / "report.json")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["rewrites"] and report["lean_targets"]
    standing = [f for f in report["generated_roots"] if (ROOT / f).is_file()]
    assert standing, "this tree's export writes root files under tracked files' names; the next line holds each to the rule"
    assert [f for f in standing if not PE.imports_alone((ROOT / f).read_text(encoding="utf-8"))] == []
    dest = tmp_path / "export"
    copied = set(subprocess.run(["git", "ls-files"], cwd=dest, capture_output=True, text=True, check=True).stdout.split())
    assert "Identity.Domains" in report["spare_roots"]["verity/commitments/identity"]
    for here in report["spare_roots"]:
        assert _unsealed(lambda f: (dest / f).read_text(encoding="utf-8"), copied, here) == [], here


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
