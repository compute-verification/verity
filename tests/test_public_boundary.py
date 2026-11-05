"""The public monorepo's boundary (`tools/verity/check/public_boundary.py`): no file the one-way export copies needs a file
it doesn't, beyond the edges `public_boundary_allowed.txt` lists, which only shrinks."""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "verity" / "check" / "public_boundary.py"
_spec = importlib.util.spec_from_file_location("public_boundary_under_test", SCRIPT)
PB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PB)


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
team = ["tools/secret", "verity/pub/hidden.py"]
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
    "verity/pub/pyproject.toml": ('[project]\nname = "pub"\ndependencies = ["privtool>=1"]\n'
                                  '[project.scripts]\npub-run = "privpkg:main"\n'),
    "verity/pub/imports.py": "import json\nimport privpkg\n",
    "verity/pub/reads.py": "from pathlib import Path\n\nDATA = Path(__file__).resolve().parents[2] / 'tools' / 'priv' / 'data.json'\n",
    "verity/pub/runs.py": "import subprocess\n\nsubprocess.run(['research', 'status'], check=True)\n",
    "verity/pub/run.sh": "#!/bin/bash\nset -e\nresearch status x\npython3 tools/priv/privpkg/__init__.py\n",
    "verity/pub/pkg/__init__.py": "",
    "verity/pub/pkg/mod.py": "X = 1\n",
    "verity/pub/hidden.py": "Y = 1\n",
    "verity/pub/sees.py": "import hidden\n",
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
    ("verity/pub/lakefile.toml", "lake require", "tools/priv/lakefile.toml"),
    ("verity/pub/imports.py", "python import", "tools/priv/privpkg/__init__.py"),
    ("verity/pub/pyproject.toml", "python dependency", "tools/priv/pyproject.toml"),
    ("verity/pub/pyproject.toml", "python import", "tools/priv/privpkg/__init__.py"),
    ("verity/pub/reads.py", "path", "tools/priv/data.json"),
    ("verity/pub/runs.py", "command", "tools/priv/pyproject.toml"),
    ("verity/pub/run.sh", "command", "tools/priv/pyproject.toml"),
    ("verity/pub/run.sh", "path", "tools/priv/privpkg/__init__.py"),
    ("verity/pub/sees.py", "python import", "verity/pub/hidden.py"),
])
def test_each_kind_of_need_is_an_edge(judged, source: str, kind: str, target: str) -> None:
    _, _, j = judged
    assert (source, kind, target) in {(e.source, e.kind, e.target) for e in j.edges}, "\n".join(map(str, j.edges))


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
