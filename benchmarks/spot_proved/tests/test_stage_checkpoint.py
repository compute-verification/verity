"""`stage.py --checkpoint`: a weight the capture didn't keep is composed from the checkpoint and read only when its SHA-512 is the one
the step's ``params.json`` committed (any other composition is an error, never "not kept"), and the DeepGEMM linears and MLA's
absorbed weights take vLLM's load-time rewrite while the routed experts don't."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
GOOD = np.arange(6, dtype=np.uint16).reshape(2, 3)


def _stage(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "setrecursionlimit", lambda n: None)
    spec = importlib.util.spec_from_file_location("spot_stage", HERE / "stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cap(stage, composed: np.ndarray, checkpoint: str | None = "ckpt"):
    cap = stage.Capture.__new__(stage.Capture)
    cap.d, cap.step, cap.ranks, cap.checkpoint = Path("/nonexistent"), 7, [0], checkpoint
    cap.commit = {0: {"opened": {}}}
    cap.params = {0: {"w": {"shape": [2, 3], "dtype": "torch.bfloat16", "sha512": hashlib.sha512(GOOD.tobytes()).hexdigest()}}}
    cap._params, cap._ck, cap._lock, cap._composers = {}, None, threading.Lock(), {0: lambda n: composed}
    return cap


def test_a_composed_weight_is_read_only_at_its_committed_digest(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    assert np.array_equal(_cap(stage, GOOD.copy()).param(0, "w"), GOOD)
    with pytest.raises(LookupError, match="not the committed bytes") as bad:
        _cap(stage, GOOD + 1).param(0, "w")
    assert not str(bad.value).startswith("not kept")
    with pytest.raises(LookupError, match="not in an opened layer"):
        _cap(stage, GOOD.copy(), checkpoint=None).param(0, "w")


def test_the_deepgemm_linears_and_mla_absorb_take_the_load_rewrite_and_the_routed_experts_do_not(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    e4, f32, bf = "torch.float8_e4m3fn", "torch.float32", "torch.bfloat16"
    L = "model.layers.3."
    params = {L + "self_attn.q_b_proj.weight": e4, L + "self_attn.q_b_proj.weight_scale_inv": f32,
              L + "self_attn.W_UK_T": bf, L + "self_attn.W_UV": bf, L + "mlp.shared_experts.down_proj.weight": e4,
              L + "mlp.shared_experts.down_proj.weight_scale_inv": f32, L + "mlp.experts.w13_weight": e4,
              L + "mlp.experts.w13_weight_scale_inv": f32, L + "mlp.gate.weight": bf, L + "input_layernorm.weight": bf}
    roles = {k: v["role"] for k, v in stage.load_roles({k: {"dtype": d} for k, d in params.items()}).items()}
    assert roles == {L + "self_attn.q_b_proj.weight": "fp8_w_requant_ue8m0", L + "self_attn.q_b_proj.weight_scale_inv": "fp8_ws_requant_ue8m0",
                     L + "self_attn.W_UK_T": "mla.absorb", L + "self_attn.W_UV": "mla.absorb",
                     L + "mlp.shared_experts.down_proj.weight": "fp8_w_requant_ue8m0",
                     L + "mlp.shared_experts.down_proj.weight_scale_inv": "fp8_ws_requant_ue8m0"}
