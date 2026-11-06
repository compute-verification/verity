"""difftest.py (protected): the one-call evaluator agrees with a numpy reference on BiasAdd (bf16 add, RNE) and is
deterministic; the word generators cover the required edge kinds; `check` fails when an edge kind is missing."""
from __future__ import annotations

import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.properties import admission as difftest
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import b1


def _bf16_add_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fa = (a.astype(np.uint32) << 16).view(np.float32)
    fb = (b.astype(np.uint32) << 16).view(np.float32)
    s = (fa + fb).astype(np.float32)
    # round-to-nearest-even to bf16
    u = s.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) >> 16
    return u.astype(np.uint16)


def test_evaluate_spec_bias_add_matches_numpy_reference():
    rng = np.random.default_rng(0)
    N = 8
    y = difftest.words_for("random", [N], "bfloat16", rng)
    b = difftest.words_for("random", [N], "bfloat16", rng)
    spec = bind(b1.BiasAdd, N=N)
    got = difftest.evaluate_spec(spec, [y, b])
    exp = _bf16_add_ref(y, b)
    finite = ~np.isin(exp & 0x7F80, [0x7F80])
    assert np.array_equal(np.array(got, np.uint16)[finite], exp[finite])
    assert got == difftest.evaluate_spec(spec, [y, b])


@pytest.mark.parametrize("kind", difftest.REQUIRED_EDGE_KINDS + ("random",))
def test_word_generators(kind):
    w = difftest.words_for(kind, [3, 5], "bfloat16", np.random.default_rng(1))
    assert w.dtype == np.uint16 and w.shape == (3, 5)
    if kind == "subnormal":
        assert np.all((w & 0x7F80) == 0) and np.all((w & 0x007F) != 0)
    if kind == "zeros":
        assert np.all((w & 0x7FFF) == 0)


def test_check_requires_edge_kinds(tmp_path, monkeypatch):
    # a fake adapter for BiasAdd whose cases are random-only -> admission fails for missing edge kinds
    mod = types.ModuleType("fake_adapter")
    mod.SEMANTIC_ID = "test/BiasAdd@v1"
    mod.cases = lambda rng, i: {"statics": {"N": 4}, "inputs": [{"shape": [4], "dtype": "bfloat16", "kind": "random"}] * 2}
    mod.spec = lambda st: bind(b1.BiasAdd, N=st["N"])
    monkeypatch.setitem(sys.modules, "fake_adapter", mod)
    d = tmp_path / "dt"
    (d / "case_0").mkdir(parents=True)
    rng = np.random.default_rng(3)
    y = difftest.words_for("random", [4], "bfloat16", rng); b = difftest.words_for("random", [4], "bfloat16", rng)
    np.save(d / "case_0" / "in_0.npy", y); np.save(d / "case_0" / "in_1.npy", b)
    np.save(d / "case_0" / "out_0.npy", _bf16_add_ref(y, b))
    rec = {"i": 0, "statics": {"N": 4}, "inputs": [{"shape": [4], "dtype": "bfloat16", "kind": "random"}] * 2,
           "outputs": [{"shape": [4], "dtype": "bfloat16"}], "edge": False}
    (d / "manifest.json").write_text(json.dumps({"seed": 3, "n": 1, "cases": [rec]}))
    rep = difftest.check("fake_adapter", d)
    assert rep["differential"] == {"passed": 1, "total": 1}
    assert rep["status"] == "fail" and set(rep["edge_kinds_missing"]) == set(difftest.REQUIRED_EDGE_KINDS)


def test_produce_writes_bf16_outputs_as_words(tmp_path, monkeypatch):
    # a production op returning bfloat16 must cross to numpy as uint16 words (numpy has no bf16;
    # `Tensor.numpy()` raised "Got unsupported ScalarType BFloat16" and every bf16-output Definition was un-difftestable)
    torch = pytest.importorskip("torch")
    mod = types.ModuleType("fake_bf16_adapter")
    mod.SEMANTIC_ID = "test/BiasAdd@v1"
    mod.cases = lambda rng, i: {"statics": {"N": 4}, "inputs": [{"shape": [4], "dtype": "bfloat16", "kind": "random"}] * 2}
    mod.production = lambda st, ins: [ins[0] + ins[1]]
    mod.spec = lambda st: bind(b1.BiasAdd, N=st["N"])
    monkeypatch.setitem(sys.modules, "fake_bf16_adapter", mod)
    man = difftest.produce("fake_bf16_adapter", 1, 3, tmp_path / "dt", device="cpu")
    out = np.load(tmp_path / "dt" / "case_0" / "out_0.npy")
    assert man["cases"][0]["outputs"] == [{"shape": [4], "dtype": "bfloat16"}] and out.dtype == np.uint16 and out.shape == (4,)
    y = np.load(tmp_path / "dt" / "case_0" / "in_0.npy"); b = np.load(tmp_path / "dt" / "case_0" / "in_1.npy")
    assert np.array_equal(out, _bf16_add_ref(y, b))
