"""CPU checks for the cc 9.0 CUDA RMSNorm capture's models, verdict and orchestration."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import json

import numpy as np

from tests.properties import glm53_cuda_rmsnorm_capture_gpu as X
from verity_vllm.engine.env import SERVED_DEFAULTS
from verity.ml.definitions.numerics import cuda_rmsnorm_rows as R
from verity_vllm.properties.record import REGISTRY, digest_of


def _case(kind: str = "rms_norm", m: int = 1, n: int = 6144) -> dict:
    return {"kind": kind, "M": m, "N": n, "block": X.M.launch_block(m, n, fused=kind != "rms_norm"),
            "eps": X.EPS_BY_WIDTH[n],
            "launch_ok": True, "mismatch": 0, "residual_mismatch": 0, "deterministic": True,
            "alternatives": {"batch_invariant_block": 1, "linear_f64_reduction": 1, "sqrt_div_epilogue": 1}}


def _doc() -> dict:
    return {"gate": {"ok": True}, "served": {"ok": True}, "sass": {"ok": True},
            "cases": [_case(kind, m, n) for n in X.WIDTHS for m in X.MS for kind in X.KINDS], "errors": {}}


def _stub_envs(monkeypatch, **switches) -> None:
    import sys
    import types

    envs = types.ModuleType("vllm.envs")
    for k, v in {"VLLM_BATCH_INVARIANT": False, "VLLM_USE_OINK_OPS": False, **switches}.items():
        setattr(envs, k, v)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)


def _through_torch(words: np.ndarray) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(words).view(np.int16)).view(torch.bfloat16)
    return X._u16(tensor)


def test_capture_bitcasts_real_torch_bf16_tensors_to_words() -> None:
    words = np.array([[0x0000, 0x8000, 0x0001, 0x7F80], [0xFF80, 0x7FC1, 0x3F80, 0xBF80]], np.uint16)
    got = _through_torch(words)
    assert got.dtype == np.uint16
    assert np.array_equal(got, words)


def test_verdict_requires_all_shapes_words_launches_determinism_and_probes() -> None:
    assert X._verdict(_doc())
    for field, value in (("launch_ok", False), ("mismatch", 1), ("residual_mismatch", 1), ("deterministic", False)):
        doc = _doc()
        doc["cases"][0][field] = value
        assert not X._verdict(doc)
    doc = _doc()
    doc["cases"][0]["alternatives"]["sqrt_div_epilogue"] = 0
    for case in doc["cases"][1:]:
        case["alternatives"]["sqrt_div_epilogue"] = 0
    assert not X._verdict(doc)
    assert not X._verdict({**_doc(), "cases": _doc()["cases"][:-1]})
    assert not X._verdict({**_doc(), "errors": {"x": "boom"}})
    assert not X._verdict({**_doc(), "served": {"ok": False}})
    assert not X._verdict({k: v for k, v in _doc().items() if k != "served"})


def test_served_needs_batch_invariance_and_oink_off(monkeypatch) -> None:
    _stub_envs(monkeypatch)
    assert X.served() == {"VLLM_BATCH_INVARIANT": False, "VLLM_USE_OINK_OPS": False, "ok": True}
    for k in ("VLLM_BATCH_INVARIANT", "VLLM_USE_OINK_OPS"):
        _stub_envs(monkeypatch, **{k: True})
        assert X.served()["ok"] is False


def test_planted_rows_and_numpy_models_include_both_in_place_outputs() -> None:
    x, res, w = X.operands(42, 64, 7)
    assert x.shape == res.shape == (42, 64) and w.shape == (64,)
    assert {0x0001, 0x0080, 0x7F7F, 0x7F80, 0x7FC1} <= set(x.reshape(-1))
    eps = X.EPS_BY_WIDTH[6144]
    got = R.fused_add_rms_norm_rows(x, res, w, eps, 64)
    assert got["out"].shape == got["residual"].shape == x.shape
    assert got["out"].dtype == got["residual"].dtype == np.uint16
    alt = X.alternatives("fused_add_rms_norm", x, res, w, eps, 64, got)
    assert set(alt) == {"batch_invariant_block", "linear_f64_reduction", "sqrt_div_epilogue"}
    assert all(isinstance(v, int) and v >= 0 for v in alt.values())


def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys) -> None:
    """`main` with the launches and the SASS build stood in for, every served-default switch inherited set: cleared before vLLM
    reads them, its switches read back, the SASS built with a served engine's toolchain, GO; with batch invariance read back on, or
    the SASS build failing, NO-GO."""
    import os

    from verity_vllm.pipeline.cli import MACHINE

    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    sass_ok = [True]

    def stub_sass(out):
        assert os.environ["PATH"].split(":")[:2] == [os.path.dirname(X.sys.executable), MACHINE["RunEnv.cuda_bin"]]
        return {"ok": sass_ok[0], "nvcc": ["Build cuda_13.0"]}

    monkeypatch.setattr(X, "sass", stub_sass)
    monkeypatch.setenv("PATH", os.environ["PATH"])
    monkeypatch.delenv("CUDA_HOME", raising=False)
    for k in SERVED_DEFAULTS:
        monkeypatch.setenv(k, "0")
    _stub_envs(monkeypatch)

    def stub_case(work, kind, m, n):
        assert not any(k in os.environ for k in SERVED_DEFAULTS)
        words = np.array([[0x0000, 0x8000, 0x7FC1, 0x3F80]], np.uint16)
        assert np.array_equal(_through_torch(words), words)
        return _case(kind, m, n)

    monkeypatch.setattr(X, "one_case", stub_case)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path)]) == 0
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["ok"] and doc["digest"] == digest_of(doc) and len(doc["cases"]) == 24
    assert doc["env_cleared"] == {k: "0" for k in SERVED_DEFAULTS} and doc["served"]["ok"] is True
    assert doc["env"]["CUDA_HOME"] == MACHINE["RunEnv.cuda_home"]
    assert {c["eps"] for c in doc["cases"] if c["N"] == 6144} == {1e-5}
    assert {c["eps"] for c in doc["cases"] if c["N"] == 7168} == {1e-6}
    assert "GLM53-CUDA-RMSNORM GO served=True sass=True cases=24 errors=0" in capsys.readouterr().out
    sass_ok[0] = False
    assert X.main(["--out", str(out), "--scratch", str(tmp_path)]) == 1
    assert "GLM53-CUDA-RMSNORM NO-GO served=True sass=False" in capsys.readouterr().out
    sass_ok[0] = True
    _stub_envs(monkeypatch, VLLM_BATCH_INVARIANT=True)
    assert X.main(["--out", str(out), "--scratch", str(tmp_path)]) == 1
    assert "GLM53-CUDA-RMSNORM NO-GO served=False" in capsys.readouterr().out


def test_the_sass_source_reduces_with_vllms_operator() -> None:
    """vLLM's cub_helpers.h CubAddOp in both block reductions: CCCL 3 (CUDA 13) has no cub::Sum, which the reproduction named
    until the served toolchain was CUDA 13's."""
    src = X.SASS_SOURCE.read_text()
    assert src.count("Reduce(variance, CubAddOp{}, blockDim.x)") == 2 and "cub::Sum{}" not in src
    includes = [line for line in src.splitlines() if line.startswith("#include")]
    assert "#if CUB_VERSION >= 200800" in src and "#include <cub/cub.cuh>" not in includes


def test_device_gate_failure_runs_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(X, "served_defaults", lambda: {})
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": False, "checks": {"cc": False}})
    monkeypatch.setattr(X, "one_case", lambda *a: (_ for _ in ()).throw(AssertionError("must not launch")))
    assert X.main(["--out", str(tmp_path)]) == 3


def test_record_is_registered() -> None:
    assert REGISTRY["glm53_cuda_rmsnorm_capture"].schema == X.SCHEMA
