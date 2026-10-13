"""CPU checks for the H200 CUDA RMSNorm capture's models, verdict and orchestration."""
from __future__ import annotations

import json

import numpy as np

from tests.properties import glm53_cuda_rmsnorm_capture_gpu as X
from verity_vllm.program.kernels import cuda_rmsnorm_rows as R
from verity_vllm.properties.record import REGISTRY, digest_of


def _case(kind: str = "rms_norm", m: int = 1, n: int = 6144) -> dict:
    return {"kind": kind, "M": m, "N": n, "block": X.M.launch_block(m, n, fused=kind != "rms_norm"),
            "launch_ok": True, "mismatch": 0, "residual_mismatch": 0, "deterministic": True,
            "alternatives": {"batch_invariant_block": 1, "linear_f64_reduction": 1, "sqrt_div_epilogue": 1}}


def _doc() -> dict:
    return {"gate": {"ok": True}, "sass": {"ok": True},
            "cases": [_case(kind, m, n) for n in X.WIDTHS for m in X.MS for kind in X.KINDS], "errors": {}}


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


def test_planted_rows_and_numpy_models_include_both_in_place_outputs() -> None:
    x, res, w = X.operands(42, 64, 7)
    assert x.shape == res.shape == (42, 64) and w.shape == (64,)
    assert {0x0001, 0x0080, 0x7F7F, 0x7F80, 0x7FC1} <= set(x.reshape(-1))
    got = R.fused_add_rms_norm_rows(x, res, w, X.EPS, 64)
    assert got["out"].shape == got["residual"].shape == x.shape
    assert got["out"].dtype == got["residual"].dtype == np.uint16


def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X, "sass", lambda out: {"ok": True})
    monkeypatch.setattr(X, "one_case", lambda work, kind, m, n: _case(kind, m, n))
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path)]) == 0
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["ok"] and doc["digest"] == digest_of(doc) and len(doc["cases"]) == 24
    assert "GLM53-CUDA-RMSNORM GO cases=24 errors=0" in capsys.readouterr().out


def test_device_gate_failure_runs_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": False, "checks": {"cc": False}})
    monkeypatch.setattr(X, "one_case", lambda *a: (_ for _ in ()).throw(AssertionError("must not launch")))
    assert X.main(["--out", str(tmp_path)]) == 3


def test_record_is_registered() -> None:
    assert REGISTRY["glm53_cuda_rmsnorm_capture"].schema == X.SCHEMA
