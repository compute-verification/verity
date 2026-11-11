"""`stage.py`'s fused all-reduce + RMSNorm instances on a capture written by `capture.py`'s recorder around a stand-in
``vllm::flashinfer_trtllm_fused_allreduce_norm``: the op's schema at vLLM d9105ea80, writing in place the way vLLM's op drives
FlashInfer's MNNVL kernel, with the values of `tp_allreduce_check.mnnvl_norm_rows` (the numpy model of the kernel). Given
``norm_out``, as `fused_allreduce_rms_norm` passes it, the new residual goes into ``allreduce_in``, the normed rows into
``norm_out``, and ``residual`` keeps the old residual though the schema marks it written. Without ``norm_out``, the new residual
goes into ``residual`` and the normed rows into ``allreduce_in``. Every staged instance must pass the match gate."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.ml.circuits.evaluation import evaluate_batch  # noqa: E402
from verity.ml.definitions.registry import tp_allreduce as TA  # noqa: E402
from verity.ml.definitions.registry import tp_allreduce_check as K  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
W, T, H, EPS, STEP, LAYER = 8, 2, 256, 1e-5, 0, 3
OP = "flashinfer_trtllm_fused_allreduce_norm"
SCHEMA = (f"{OP}(Tensor(a0!) allreduce_in, Tensor(a1!) residual, Tensor rms_gamma, float rms_eps, SymInt world_size, "
          "bool launch_with_pdl, bool fp32_acc, SymInt max_token_num, SymInt pattern_code, Tensor(a9!)? norm_out=None, "
          "Tensor(a10!)? quant_out=None, Tensor(a11!)? scale_out=None, Tensor? scale_factor=None, float weight_bias=0.) -> ()")
#: what the stand-in reduces: every rank's partials, (W, T, H) bf16 words, set by the test before the ranks run
PARTS: dict[str, np.ndarray] = {}


def _bf16(rng, shape, scale: float) -> np.ndarray:
    return ((rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _t(words: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(words).view(np.int16)).view(torch.bfloat16)


def _w(t) -> np.ndarray:
    return t.view(torch.int16).numpy().view(np.uint16)


def _stand_in(allreduce_in, residual, rms_gamma, rms_eps, world_size, launch_with_pdl, fp32_acc, max_token_num, pattern_code,
              norm_out=None, quant_out=None, scale_out=None, scale_factor=None, weight_bias=0.0):
    parts = PARTS["parts"]
    M, N = residual.shape
    if not any(np.array_equal(_w(allreduce_in), p) for p in parts):
        raise AssertionError("allreduce_in is no rank's partial")
    normed, new_res = K.mnnvl_norm_rows(parts, _w(residual), _w(rms_gamma), TA.mnnvl_shot(M, N, world_size),
                                        TA.mnnvl_grid(M, N)[1], rms_eps, True)
    residual_out, norm_dst = (allreduce_in, norm_out) if norm_out is not None else (residual, allreduce_in)
    residual_out.copy_(_t(new_res))
    norm_dst.copy_(_t(normed))


@pytest.fixture(scope="module")
def fused_op():
    lib = torch.library.Library("vllm", "FRAGMENT")
    lib.define(SCHEMA)
    lib.impl(OP, _stand_in, "CPU")
    yield getattr(torch.ops.vllm, OP).default
    lib._destroy()


def _load(name: str, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "setrecursionlimit", lambda n: None)
    spec = importlib.util.spec_from_file_location(f"spot_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def _capture(tmp: Path, monkeypatch, fused_op, norm_out_given: bool):
    """Each rank's call recorded by capture.py's recorder, committed in the layout stage.py's `Capture` checks."""
    capture = _load("capture", monkeypatch)
    S = capture._State({"capture": str(tmp), "steps": [STEP], "mla_tokens": 0, "open": {}, "keep": "all", "min_free_gib": 0})
    monkeypatch.setattr(capture, "S", S)
    rng = np.random.default_rng(7 + norm_out_given)
    PARTS["parts"] = parts = _bf16(rng, (W, T, H), 0.5)
    res, gamma = _bf16(rng, (T, H), 2.0), _bf16(rng, (H,), 0.1)
    roots = []
    for r in range(W):
        S.rank, S.events, S.recording, S.whole, S.layer = r, [], True, True, LAYER
        x, rs, g = _t(parts[r]).clone(), _t(res).clone(), _t(gamma).clone()
        out = torch.empty_like(x) if norm_out_given else None
        with capture.Recorder():
            fused_op(x, rs, g, EPS, W, True, True, 42, 1, out)
        capture._drain()
        sd = tmp / f"rank{r}" / f"step{STEP}"
        sd.mkdir(parents=True, exist_ok=True)
        ops = "".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in S.events).encode()
        (sd / "ops.jsonl").write_bytes(ops)
        (tmp / f"rank{r}" / "params.json").write_bytes(b"{}")
        root = _sha(f"verity/spot/rank-root\0{_sha(ops)}\0{_sha(b'{}')}".encode())
        (sd / "meta.json").write_text(json.dumps({"scheduled": {"req": T}, "ops_sha512": _sha(ops), "params_sha512": _sha(b"{}"),
                                                  "rank_root": root, "check": {}}))
        roots.append(root)
    step_root = _sha(("verity/spot/step-root\0" + "\0".join(roots)).encode())
    for r in range(W):
        (tmp / f"rank{r}" / f"step{STEP}" / "commit.json").write_text(json.dumps(
            {"root": step_root, "coins": "00" * 64, "rank_roots": roots, "layers": {"moe": [LAYER]}, "opened": {}}))
    return parts, res, gamma


@pytest.mark.parametrize("norm_out_given", [True, False], ids=["norm_out", "no_norm_out"])
def test_every_fused_instance_passes_the_match_gate(tmp_path, monkeypatch, fused_op, norm_out_given: bool) -> None:
    parts, res, gamma = _capture(tmp_path, monkeypatch, fused_op, norm_out_given)
    stage = _load("stage", monkeypatch)
    cap = stage.Capture(tmp_path, STEP)
    (e0,) = [e for e in cap.events[0] if e["op"] == f"vllm::{OP}"]
    want_normed, want_res = K.mnnvl_norm_rows(parts, res, gamma, 1, TA.mnnvl_grid(T, H)[1], EPS, True)
    new_res, normed = stage.fused_ar_norm_outputs(e0)
    assert np.array_equal(cap.tensor(0, new_res), want_res) and np.array_equal(cap.tensor(0, normed), want_normed)
    if norm_out_given:
        assert e0["out"]["residual"]["sha512"] == e0["in"]["residual"]["sha512"], "the recorder saw residual written"
    insts = list(stage.fused_ar_norm_instances(cap))
    assert sorted({i[0] for i in insts}) == ["AllReduceResidualMnnvl_W8_S1", "RmsNormMnnvl_W8_S1"] and len(insts) == 3 * T
    for label, block, key, args_of, want_of in insts:
        if label.startswith("RmsNorm"):
            assert f"CLUSTER={TA.mnnvl_grid(T, H)[1]}," in block, block
        args = [stage.words(a)[None] for a in args_of()]
        got = np.asarray(evaluate_batch(stage.block_fn(block), *args)).reshape(-1)
        want = np.concatenate([stage.words(a) for a in want_of()])
        assert np.array_equal(got, want), (key, int(np.nonzero(got != want)[0][0]))
        if norm_out_given and label.startswith("AllReduceResidual"):
            t, c = key[1], key[2]
            old = stage.words(cap.tensor(0, e0["out"]["residual"])[t, c * (H // 2):(c + 1) * (H // 2)])
            assert not np.array_equal(got, old), "the old residual buffer would have passed the gate"
