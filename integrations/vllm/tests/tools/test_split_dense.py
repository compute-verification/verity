"""`tools/migrate/split_dense.py` on a small tree laid out as main was before the split: it moves the family's twins and
vocabulary, rewrites each import form, reruns as a no-op, and refuses by name a tree it cannot split."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "migrate" / "split_dense.py"
_spec = importlib.util.spec_from_file_location("split_dense", SCRIPT)
SD = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("split_dense", SD)
_spec.loader.exec_module(SD)

REG = "verity_vllm/program/registry"
CONSUMER = '''"""Reads `registry.rope_difftest` and verity_vllm.program.registry.gemm_targets."""
from verity_vllm.program.registry import gemm_targets as GT
from verity_vllm.program.registry import b1, rope_difftest  # noqa: F401
from verity_vllm.program.registry.gemm_targets import REGISTERED  # noqa: E402


def f():
    from verity_vllm.program.registry import rmsnorm_fused_sweep
    return rmsnorm_fused_sweep, "verity_vllm.program.registry.rope_gptj_difftest"
'''
WANT = '''"""Reads `difftests.rope_difftest` and verity_vllm.program.frontend.gemm_targets."""
from verity_vllm.program.frontend import gemm_targets as GT
from verity_vllm.program.registry import b1
from verity_vllm.properties.difftests import rope_difftest  # noqa: F401
from verity_vllm.program.frontend.gemm_targets import REGISTERED  # noqa: E402


def f():
    from verity_vllm.program.kernels import rmsnorm_fused_sweep
    return rmsnorm_fused_sweep, "verity_vllm.properties.difftests.rope_gptj_difftest"
'''
CYCLE = '{"file": "verity_vllm", "kind": "module-cycle", "symbol": "<package>", "detail": "%s"}'
P09 = CYCLE % "program.frontend.target_profile <-> program.registry.gemm_targets" + "\n"


def _tree(root: Path) -> Path:
    v = root / "integrations" / "vllm"
    for d in (REG, "verity_vllm/program/frontend", "verity_vllm/program/kernels", "verity_vllm/properties"):
        (v / d).mkdir(parents=True)
    for name in SD.DEFINITION_MODULES + ("b1", "rope_gptj_difftest", "rmsnorm_fused_sweep", "gemm_targets"):
        (v / REG / f"{name}.py").write_text(f'"""{name}"""\n')
    (v / REG / "rope_difftest.py").write_text("from . import b1\nfrom .prims import BF16\n")
    (v / "verity_vllm/consumer.py").write_text(CONSUMER)
    (v / "tests").mkdir()
    (v / "tests/allow.json").write_text('[{"file": "verity_vllm/program/registry/gemm_targets.py"}]\n')
    (v / "tests/p09_layering.json").write_text(P09)
    (root / "CHANGELOG.md").write_text("- moved verity_vllm.program.registry.gemm_targets\n")
    return v


def test_the_split_moves_rewrites_and_reruns_as_a_no_op(tmp_path):
    v = _tree(tmp_path)
    done = SD.apply(tmp_path, log=lambda *_: None)
    assert [m.name for m in done["moves"]] == [m.name for m in SD.MOVES]
    for m in SD.MOVES:
        assert not (v / m.old_path).exists() and (v / m.new_path).exists()
    assert (v / "verity_vllm/consumer.py").read_text() == WANT
    assert (v / "tests/allow.json").read_text() == '[{"file": "verity_vllm/program/frontend/gemm_targets.py"}]\n'
    assert (v / "verity_vllm/properties/difftests/rope_difftest.py").read_text() == (
        "from verity_vllm.program.registry import b1\nfrom verity_vllm.program.registry.prims import BF16\n")
    assert (v / "tests/p09_layering.json").read_text() == CYCLE % "program.frontend.gemm_targets <-> program.frontend.target_profile" + "\n"
    assert (v / "verity_vllm/properties/difftests/__init__.py").exists()
    assert "registry.gemm_targets" in (tmp_path / "CHANGELOG.md").read_text()
    again = SD.plan(tmp_path)
    assert again == {"moves": [], "edits": {}}


def test_a_module_in_both_places_is_refused(tmp_path):
    v = _tree(tmp_path)
    SD.apply(tmp_path, log=lambda *_: None)
    (v / REG / "gemm_targets.py").write_text("")
    with pytest.raises(SD.SplitError, match="gemm_targets: both"):
        SD.plan(tmp_path)


def test_a_module_in_neither_place_is_refused(tmp_path):
    v = _tree(tmp_path)
    (v / REG / "rope_gptj_difftest.py").unlink()
    with pytest.raises(SD.SplitError, match="rope_gptj_difftest: neither"):
        SD.plan(tmp_path)


def test_an_import_it_cannot_rewrite_is_refused_and_nothing_changes(tmp_path):
    v = _tree(tmp_path)
    (v / "verity_vllm/consumer.py").write_text("from verity_vllm.program.registry import (b1,\n    gemm_targets)\n")
    with pytest.raises(SD.SplitError, match="parenthesized import"):
        SD.apply(tmp_path, log=lambda *_: None)
    assert (v / REG / "gemm_targets.py").exists()


def test_a_relative_import_of_a_moved_module_in_the_registry_is_refused(tmp_path):
    v = _tree(tmp_path)
    (v / REG / "b1.py").write_text("from .gemm_targets import REGISTERED\n")
    with pytest.raises(SD.SplitError, match="relative import of gemm_targets"):
        SD.plan(tmp_path)
