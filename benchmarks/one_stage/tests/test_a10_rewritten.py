"""A10's registered tensors that vLLM rewrites at load (`Committed.rewritten`) open against their committed tensor roots under the
weights tree's scheme (`cpu_replay.tensor_root_of`): under rows-sha512, a tensor composed as vLLM rewrites it opens and one composed
otherwise is mismatched by name."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent


def _a10(monkeypatch) -> types.ModuleType:
    """a10 over stand-ins for a9 and a11, whose module-level names it binds and `rewritten` never reads: the real ones import the
    served units and circuit-check, which this suite does not read.  Loaded under its own name, so no other test imports it."""
    monkeypatch.syspath_prepend(str(HERE.parent))
    for name, attrs in (("a9", ("capture_words",)), ("a11", ("ROW", "fold_values", "operands", "save_values"))):
        stub = types.ModuleType(name)
        for a in attrs:
            setattr(stub, a, None)
        monkeypatch.setitem(sys.modules, name, stub)
    spec = importlib.util.spec_from_file_location("a10_under_test", HERE.parent / "a10.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_rewritten_tensor_opens_against_its_rows_sha512_tensor_root(monkeypatch, tmp_path):
    from verity_vllm.check import fp8_ue8m0
    from verity_vllm.commit import scheme, step_rows

    a10 = _a10(monkeypatch)
    geo, wk = bytes(range(32)), bytes(range(64, 128))
    served = {"model.a.weight": np.arange(8, dtype=np.uint16), "model.b.weight": np.arange(8, 16, dtype=np.uint16)}
    names = [[n, "bfloat16", [8], 16] for n in served]
    roots = [step_rows.tensor_root(geo, i, e, served[e[0]].view(np.uint8), wk) for i, e in enumerate(names)]
    tree = step_rows.weights_tree(geo, names, scheme.names_digest(names), roots, wk)

    class Rewrites:
        def __init__(self, _names, _roles, _compose, _config):
            pass

        def covers(self, _name):
            return True

        def compose(self, name, _shape):
            return (served[name] if name == "model.a.weight" else served[name] + 1), "rewritten"

    monkeypatch.setattr(fp8_ue8m0, "build_roles", lambda _d: {n: {} for n in served})
    monkeypatch.setattr(fp8_ue8m0, "Rewrites", Rewrites)
    words: dict = {}
    out = a10.Committed.rewritten(tree, SimpleNamespace(config={}), words, tmp_path, "", {"world": 1, "rank": 0})
    assert (out["covered"], out["opened"], out["mismatched"]) == (2, 1, ["model.b.weight"])
    assert list(words) == ["model.a.weight"] and np.array_equal(words["model.a.weight"], served["model.a.weight"])
