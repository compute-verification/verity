"""`verity.ml.circuits.refs.runs` (the reference-run decomposition the derivation frontend and program families share) is
pure and lives in `refs`, so `verity_vllm.program.frontend.rules.family.match_program` imports without torch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from verity.ml.circuits.refs import Affine, Concat, Explicit, Strided, runs


def test_runs_merges_adjacent_and_keeps_order():
    assert runs(Affine("n", 3, 4, 8)) == [("n", 3, 4, 12)]
    assert runs(Concat([Affine("n", 3, 4, 8), Affine("n", 3, 12, 2), Affine("p", 0, 0, 1)])) == [("n", 3, 4, 14), ("p", 0, 0, 1)]
    assert runs(Explicit((("n", 1, 5), ("n", 1, 6), ("n", 2, 0)))) == [("n", 1, 5, 7), ("n", 2, 0, 1)]
    # a [2][3] row-major view with row stride 10 -> two unit-stride rows
    assert runs(Strided("n", 0, 100, 3, 10, 1, 6)) == [("n", 0, 100, 103), ("n", 0, 110, 113)]
    # a column gather (inner stride != 1) is one run per leaf
    assert runs(Strided("n", 0, 0, 2, 1, 4, 4)) == [("n", 0, 0, 1), ("n", 0, 4, 5), ("n", 0, 1, 2), ("n", 0, 5, 6)]
    assert runs(Explicit(())) == []


def test_family_imports_without_torch():
    """`match_program` is the capture wave's M4 structure comparison, run on a laptop without torch."""
    code = ("import sys; sys.modules['torch'] = None\n"
            "from verity_vllm.program.frontend.rules.family import match_program, expand_leaves\n"
            "from verity_vllm.program.frontend.rules.family import _runs\n"
            "from verity.ml.circuits.refs import runs\n"
            "assert _runs is runs\n"
            "print('ok')")
    repo = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_derive_alias_when_torch_available():
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    from verity_vllm.program.frontend import derive
    assert derive._runs is runs
