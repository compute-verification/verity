"""`source_identity.closure()` is the integration's import closure: `source_identity.code_identity` over it keys the hot Commit worker and
names the TP Commit's tree, so a Verity core file outside it would be code those keys never see."""
from __future__ import annotations

from pathlib import Path

import verity

from verity_vllm.pipeline import source_identity as sid


def test_closure_covers_every_core_file():
    """Every directory the import system finds `verity` in (one per member that ships part of it), not a hand-written one."""
    pats = [sid.glob_re(p) for p in sid.closure()]
    files = sid.list_source_files(sid.TREE)
    portions = [Path(p).resolve().relative_to(sid.TREE.resolve()).as_posix() for p in verity.__path__]
    assert portions
    for portion in portions:
        core = [p for p in files if p.startswith(portion + "/")]
        outside = [p for p in core if not any(rx.match(p) for rx in pats)]
        assert core and not outside, f"core files under {portion} outside source_identity.closure(): {outside}"


def test_glob_re_crosses_directories_only_at_double_star():
    assert sid.glob_re("verity/**").match("verity/ml/circuits/ops.py")
    assert sid.glob_re("**/x.py").match("x.py") and sid.glob_re("**/x.py").match("a/b/x.py")
    assert not sid.glob_re("verity/*.py").match("verity/ml/ops.py") and sid.glob_re("verity/*.py").match("verity/ops.py")
    assert not sid.glob_re("integrations/vllm/**").match("integrations/vllmx/a.py")


def test_a_tree_without_git_lists_what_its_gitignore_leaves(tmp_path):
    (tmp_path / "out" / "gen").mkdir(parents=True)
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    for rel in ("out/gen/a.py", "pkg/__pycache__/m.cpython-312.pyc", "pkg/m.py", "pkg/m.pyc"):
        (tmp_path / rel).write_text("")
    (tmp_path / ".gitignore").write_text("out/\n*.pyc\n")
    assert sid.list_source_files(tmp_path) == [".gitignore", "pkg/m.py"]
