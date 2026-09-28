"""`tools/migrate/split_prims_serve.py`'s reference rewrite on small sources, and the split's state on this tree (every move
made, nothing half-made)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "migrate" / "split_prims_serve.py"


@pytest.fixture(scope="module")
def S():
    spec = importlib.util.spec_from_file_location("split_prims_serve", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("split_prims_serve", mod)
    spec.loader.exec_module(mod)
    return mod


REG = "verity_vllm.program.registry"


def test_a_moved_module_import_is_split_off(S):
    src = f"from {REG} import b1, targets as T  # noqa: F401\n"
    assert S.rewrite_python(src) == (f"from {REG} import b1\n"
                                     f"from verity_vllm.program.frontend import targets as T  # noqa: F401\n")


def test_moved_names_are_imported_from_their_new_module(S):
    src = f"from {REG}.lifted import check_lag, lag_from_header\n"
    assert S.rewrite_python(src) == (f"from {REG}.lifted import check_lag\n"
                                     "from verity_vllm.program.lifting.lifted_records import lag_from_header\n")


def test_an_alias_used_only_for_moved_names_is_replaced(S):
    src = f"def f(x):\n    from {REG} import b1_tp2 as TP2\n    return TP2.collective_kind(x)\n"
    assert S.rewrite_python(src) == ("def f(x):\n    from verity_vllm.program.frontend import collective_kinds\n"
                                     "    return collective_kinds.collective_kind(x)\n")


def test_an_alias_still_used_keeps_its_import(S):
    src = (f"from {REG} import b1\nfrom {REG} import prims as P\n\n\n"
           "def g(monkeypatch):\n    monkeypatch.setattr(P, \"_rms\", None)\n    return P._fa2(), P.BF16_NAN_TORCH\n")
    out = S.rewrite_python(src)
    assert f"from {REG} import b1, prims as P\nfrom verity_vllm.program.kernels import prims_twins\n" in out
    assert 'monkeypatch.setattr(prims_twins, "_rms", None)' in out
    assert "return prims_twins._fa2(), P.BF16_NAN_TORCH" in out


def test_python_in_a_string_is_rewritten(S):
    src = f'CODE = """\nfrom {REG} import prims as P, targets as T\nprint(T.DOTS)\n"""\n'
    assert S.rewrite_python(src) == (f'CODE = """\nfrom {REG} import prims as P\n'
                                     'from verity_vllm.program.frontend import targets as T\nprint(T.DOTS)\n"""\n')


def test_short_forms_are_rewritten_in_comments_only(S):
    src = 'X = "registry.targets.fp8_block_dot"  # see registry.targets\n'
    assert S.rewrite_python(src) == 'X = "registry.targets.fp8_block_dot"  # see frontend.targets\n'


def test_full_paths_are_rewritten_everywhere(S):
    src = f'M = "{REG}.conformance"\nP = "integrations/vllm/verity_vllm/program/registry/targets.py"\n'
    assert S.rewrite_python(src) == ('M = "verity_vllm.program.frontend.conformance"\n'
                                     'P = "integrations/vllm/verity_vllm/program/frontend/targets.py"\n')


@pytest.mark.parametrize("src", [
    "from .targets import attention_spec\n",
    f"from {REG}.lifted import *\n",
])
def test_what_it_cannot_rewrite_fails(S, src):
    with pytest.raises(S.SplitError):
        S.rewrite_python(src, f"integrations/vllm/{REG.replace('.', '/')}/x.py")


def test_the_rewrite_is_idempotent(S):
    src = (f"from {REG} import b1, prims as P, targets as T\n\n\n"
           "def f():\n    return P._rms(), T.DOTS, b1.Serve  # registry.targets\n")
    once = S.rewrite_python(src)
    assert S.rewrite_python(once) == once


def test_census_roots_declare_the_difftests_directory(S):
    text = "a.b\nverity_vllm.properties.difftests.collectives_difftest      # by name\nc.d\n"
    out = S.census_roots(text)
    assert out.splitlines()[1].startswith("verity_vllm/properties/difftests/ ")
    assert "collectives_difftest" not in out
    assert S.census_roots(out) == out


def test_this_tree_is_split(S):
    """Every move is made and none is half-made (`_state` fails on a half-made one), so `apply` here plans no move."""
    wholes, parts = S._state(S.REPO)
    assert (wholes, parts) == ([], [])
