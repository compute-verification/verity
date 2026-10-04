"""`tools/migrate/split_dense.py` on synthetic trees: what it moves and rewrites, that a rerun changes nothing, that a tree
already split (a branch rebased past it) only has its leftover references rewritten, and each loud refusal by name."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools/migrate/split_dense.py"


def _load():
    spec = importlib.util.spec_from_file_location("split_dense", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["split_dense"] = mod
    spec.loader.exec_module(mod)
    return mod


SD = _load()
REG = "integrations/vllm/verity_vllm/program/registry"
KER = "integrations/vllm/verity_vllm/program/kernels"
OLD = "verity_vllm.program." + "registry"

TWIN = f'''"""Adapter:  verity-vllm properties-admission produce --adapter {OLD}.rope_difftest --n 4"""
from verity.ir.defs import CompositeDefinition
from . import b1
from .prims import BF16

SPEC = CompositeDefinition("RoPEDifftest", 1, ("NH",), None, lambda B, S: None, register=False)
'''

FILES = {
    f"{REG}/__init__.py": "",
    f"{REG}/prims.py": "BF16 = 'bf16'\n",
    f"{REG}/b1.py": f'CONFORMANCE = "B0 store sweep, `{OLD}.rmsnorm_fused_sweep`"\n',
    f"{REG}/sampling.py": "",
    **{f"{REG}/{m}.py": f"import numpy as np\nfrom verity.ir.defs import composite\nfrom {OLD} import prims as P, b1\n"
       for m in SD.DEFINITION_MODULES},
    f"{REG}/rope_difftest.py": TWIN,
    f"{REG}/rope_gptj_difftest.py": "from verity_vllm.program.registry import boolean_rope\n",
    f"{REG}/rmsnorm_fused_sweep.py": "from verity_vllm.config import ROOT\n",
    f"{KER}/__init__.py": "",
    f"{KER}/derived_rows.py": (f"def rows():\n    \"\"\"Delegates to `rmsnorm_fused_sweep.fused_rows`.\"\"\"\n"
                               f"    from {OLD}.rmsnorm_fused_sweep import fused_rows\n    return fused_rows\n"),
    "integrations/vllm/tests/test_x.py": (f"from {OLD} import b1, rope_difftest as RD, prims\n"
                                          f"from {OLD} import rope_gptj_difftest\n"
                                          f"\"\"\"`registry.rope_difftest` and `registry.moe_difftest`\"\"\"\n"),
    "integrations/vllm/tests/census_roots.txt": f"{OLD}.rope_difftest     # adapter\n{OLD}.moe_difftest      # adapter\n",
    "integrations/vllm/tests/by_name_allowlist.json": (
        '[\n'
        '  {"file": "verity_vllm/program/registry/lifted.py", "kind": "table"},\n'
        '  {"file": "verity_vllm/program/registry/rmsnorm_fused_sweep.py", "kind": "model-name"}\n'
        ']\n'),
    "integrations/vllm/tests/lint/allowlists/p08_facts.json": (
        '{"entries": [\n    {"file": "verity_vllm/program/registry/rmsnorm_fused_sweep.py", "kind": "model"}\n]}\n'),
    "integrations/vllm/verity_vllm/pipeline/cli.py": f'COMMANDS = {{"rmsnorm-fused-sweep": ("{OLD}.rmsnorm_fused_sweep", "")}}\n',
    "integrations/vllm/tools/migrate/notes.py": f"# names {OLD}.rope_difftest on purpose\n",
}


def _tree(root: Path, files: dict[str, str] = FILES) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def _snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*")) if p.is_file() and ".git" not in p.parts}


def test_moves_the_twins_and_rewrites_every_reference(tmp_path):
    root = _tree(tmp_path)
    assert SD.main(["--root", str(root)]) == 0
    for name in SD.MOVES:
        assert not (root / f"{REG}/{name}.py").exists()
        assert (root / f"{KER}/{name}.py").exists()
    moved = (root / f"{KER}/rope_difftest.py").read_text()
    assert "from verity_vllm.program.registry import b1\n" in moved
    assert "from verity_vllm.program.registry.prims import BF16\n" in moved
    assert "--adapter verity_vllm.program.kernels.rope_difftest" in moved
    assert (root / "integrations/vllm/tests/test_x.py").read_text() == (
        f"from {OLD} import b1, prims\nfrom verity_vllm.program.kernels import rope_difftest as RD\n"
        "from verity_vllm.program.kernels import rope_gptj_difftest\n"
        "\"\"\"`kernels.rope_difftest` and `registry.moe_difftest`\"\"\"\n")
    assert (root / f"{KER}/derived_rows.py").read_text() == FILES[f"{KER}/derived_rows.py"].replace(
        f"{OLD}.rmsnorm_fused_sweep", "verity_vllm.program.kernels.rmsnorm_fused_sweep")
    assert (root / "integrations/vllm/tests/census_roots.txt").read_text() == (
        "verity_vllm.program.kernels.rope_difftest     # adapter\n"
        "verity_vllm.program.kernels.rope_gptj_difftest     # properties-admission --adapter: imported by name\n"
        f"{OLD}.moe_difftest      # adapter\n")
    assert json.loads((root / "integrations/vllm/tests/by_name_allowlist.json").read_text()) == [
        {"file": "verity_vllm/program/registry/lifted.py", "kind": "table"}]
    assert json.loads((root / "integrations/vllm/tests/lint/allowlists/p08_facts.json").read_text()) == {
        "entries": [{"file": "verity_vllm/program/kernels/rmsnorm_fused_sweep.py", "kind": "model"}]}
    assert "verity_vllm.program.kernels.rmsnorm_fused_sweep" in (root / "integrations/vllm/verity_vllm/pipeline/cli.py").read_text()
    assert (root / f"{REG}/b1.py").read_text() == FILES[f"{REG}/b1.py"]
    assert (root / "integrations/vllm/tools/migrate/notes.py").read_text() == FILES["integrations/vllm/tools/migrate/notes.py"]
    for m in SD.DEFINITION_MODULES:
        assert (root / f"{REG}/{m}.py").read_text() == FILES[f"{REG}/{m}.py"]


def test_rerun_changes_nothing_and_check_passes(tmp_path):
    root = _tree(tmp_path)
    assert SD.main(["--root", str(root), "--check"]) == 1
    assert SD.main(["--root", str(root)]) == 0
    after = _snapshot(root)
    assert SD.main(["--root", str(root), "--check"]) == 0
    assert SD.plan(root).empty()
    assert SD.main(["--root", str(root)]) == 0
    assert _snapshot(root) == after


def test_on_a_split_tree_only_the_leftover_references_are_rewritten(tmp_path):
    """A branch rebased past the split: the modules are moved, but the branch added a new reference to an old path."""
    root = _tree(tmp_path)
    assert SD.main(["--root", str(root)]) == 0
    late = root / "integrations/vllm/tests/test_late.py"
    late.write_text(f"from {OLD}.rope_gptj_difftest import SHAPES\n")
    p = SD.plan(root)
    assert p.moves == [] and set(p.edits) == {late}
    assert SD.main(["--root", str(root)]) == 0
    assert late.read_text() == "from verity_vllm.program.kernels.rope_gptj_difftest import SHAPES\n"


def test_in_a_git_work_tree_the_move_is_a_rename(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    root = _tree(tmp_path)
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "tree"], check=True)
    assert SD.main(["--root", str(root)]) == 0
    status = subprocess.run(git + ["status", "--porcelain"], check=True, capture_output=True, text=True).stdout
    assert f"R  {REG}/rmsnorm_fused_sweep.py -> {KER}/rmsnorm_fused_sweep.py" in status
    assert (root / f"{KER}/rope_difftest.py").read_text().startswith('"""Adapter:  verity-vllm properties-admission produce '
                                                                      '--adapter verity_vllm.program.kernels.rope_difftest')


def _refused(root: Path, *fragments: str) -> None:
    before = _snapshot(root)
    with pytest.raises(SD.SplitError) as e:
        SD.plan(root)
    for f in fragments:
        assert f in str(e.value)
    assert SD.main(["--root", str(root)]) == 2
    assert _snapshot(root) == before


def test_refuses_a_module_at_both_paths(tmp_path):
    root = _tree(tmp_path, {**FILES, f"{KER}/rope_difftest.py": TWIN})
    _refused(root, "rope_difftest: both")


def test_refuses_a_module_at_neither_path(tmp_path):
    root = _tree(tmp_path, {k: v for k, v in FILES.items() if not k.endswith("/rope_gptj_difftest.py")})
    _refused(root, "rope_gptj_difftest: neither")


@pytest.mark.parametrize("body, found", [
    ("from verity.ir.defs import composite\n\n@composite('X', 1, [], None)\ndef X(B, S):\n    pass\n", "@composite X"),
    ("from verity.ir.defs import CompositeDefinition\nX = CompositeDefinition('X', 1, (), None, None)\n",
     "CompositeDefinition(...) without register=False"),
])
def test_refuses_to_move_a_module_that_registers_a_definition(tmp_path, body, found):
    root = _tree(tmp_path, {**FILES, f"{REG}/rmsnorm_fused_sweep.py": body})
    _refused(root, "rmsnorm_fused_sweep registers Definitions", found)


@pytest.mark.parametrize("line, mod", [
    ("from verity_vllm.config import ROOT\n", "verity_vllm.config.ROOT"),
    ("from . import spec\n", f"{OLD}.spec"),
    ("import torch\n", "torch"),
    ("def f():\n    from verity_vllm.program.kernels import twins\n", "verity_vllm.program.kernels.twins"),
])
def test_refuses_a_definition_module_that_imports_a_non_definition_module(tmp_path, line, mod):
    root = _tree(tmp_path, {**FILES, f"{REG}/boolean_norms.py": FILES[f"{REG}/boolean_norms.py"] + line})
    _refused(root, "Definition module boolean_norms imports a non-Definition module", mod)


def test_refuses_attribute_access_it_cannot_rewrite(tmp_path):
    use = f"from verity_vllm.program import registry\nCASES = registry.rope_difftest.cases\n"
    root = _tree(tmp_path, {**FILES, "integrations/vllm/tests/test_attr.py": use})
    _refused(root, "test_attr.py", "registry.rope_difftest")


def test_refuses_a_census_root_without_its_place(tmp_path):
    root = _tree(tmp_path, {**FILES, "integrations/vllm/tests/census_roots.txt": f"{OLD}.moe_difftest      # adapter\n"})
    _refused(root, "census_roots.txt: 0 lines start with 'verity_vllm.program.kernels.rope_difftest'",
             "verity_vllm.program.kernels.rope_gptj_difftest has no place")


def test_refuses_a_kept_file_that_imports_a_moved_module(tmp_path):
    kept = FILES[f"{REG}/b1.py"] + "from . import rope_difftest\n"
    root = _tree(tmp_path, {**FILES, f"{REG}/b1.py": kept})
    _refused(root, "b1.py is kept as it is but imports a moved module", f"{OLD}.rope_difftest")
