"""spot_proved's staging of MLA's latent absorbs (`stage.bmm_instances`) on an in-memory capture, through the match gate: each
``aten::bmm`` of a row's heads against ``W_UK_T`` or ``W_UV`` is staged per token row on `BatchedGemm_v1`, agrees with
`bf16_gemm_check.bmm_rows` and fails on a flipped word; a call in the DSA indexer's decode-padded layout (padded decode rows,
then the prefill rows) maps its rows to the step's tokens and leaves its padding out; any other bmm is counted, not staged."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from verity.definitions.registry import bf16_gemm_check as BC

ROOT = Path(__file__).resolve().parents[3]
B, K, N, PADDED, TOKENS, PREFILL = 2, 32, 16, 8, 6, 3


def _stage(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "setrecursionlimit", lambda n: None)
    spec = importlib.util.spec_from_file_location("spot_stage", ROOT / "benchmarks" / "spot_proved" / "stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bf16(rng, shape) -> np.ndarray:
    f = (rng.standard_normal(shape).astype(np.float32) * np.float32(0.5)).view(np.uint32)
    return ((f + np.uint32(0x8000)) >> np.uint32(16)).astype(np.uint16)


def _cap(stage):
    C = stage.Capture
    t: dict = {}

    class _Cap:
        rows, real_rows, tokens, padded = C.rows, C.real_rows, C.tokens, C.padded

        def opened(self, e, weights=False):
            return True

        def tensor(self, r, ref):
            return t[ref["sha512"]] if "sha512" in ref else t[ref["param"]]

    def ref(name, a, param=False):
        t[name] = a
        return {"param" if param else "sha512": name, "shape": list(a.shape), "dtype": "torch.bfloat16"}

    rng = np.random.default_rng(5)
    uk, uv = _bf16(rng, (B, K, N)), _bf16(rng, (B, N, K))
    q, o = _bf16(rng, (B, PADDED, K)), _bf16(rng, (B, TOKENS - PREFILL + 1 + PREFILL, N))
    events = [{"seq": 0, "op": "_C::top_k_per_row_prefill", "layer": 0, "in": {"numRows": PREFILL}, "out": {}},
              {"seq": 1, "op": "aten::bmm", "layer": 0, "in": {"self": ref("q", q), "mat2": ref("W_UK_T", uk, True)},
               "out": {"ret0": ref("ql", BC.bmm_rows(q, uk))}},
              {"seq": 2, "op": "aten::bmm", "layer": 0, "in": {"self": ref("o", o), "mat2": ref("W_UV", uv, True)},
               "out": {"ret0": ref("ov", BC.bmm_rows(o, uv))}},
              {"seq": 3, "op": "aten::bmm", "layer": 0, "in": {"self": ref("o", o), "mat2": ref("o2", o)}, "out": {"ret0": ref("o3", o)}},
              {"seq": 4, "op": "model_output", "layer": None, "in": {}, "out": {"hidden": {"shape": [PADDED, 6144]}}}]
    cap = _Cap()
    cap.ranks, cap.events, cap.meta, cap.only, cap.t = [0], {0: events}, {0: {"scheduled": {"a": TOKENS}}}, None, t
    return cap


def _profile():
    from verity_vllm import TargetProfile
    return TargetProfile(compute_capability=(9, 0), batch_invariant=False)


def test_mlas_absorbs_are_gated_per_token_row_and_the_decode_padded_layout_leaves_its_padding_out(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    cap = _cap(stage)
    unstaged: dict = {}
    insts = list(stage.bmm_instances(cap, _profile(), unstaged))
    assert unstaged["AtenBmm_other"]["calls"] == [{"rank": 0, "seq": 3}]
    by_seq = {s: [i for i in insts if i[2][1] == s] for s in (1, 2)}
    assert [i[2][2] for i in by_seq[1]] == list(range(TOKENS)) and [i[2][2] for i in by_seq[2]] == list(range(TOKENS))
    assert {i[0] for i in by_seq[1]} == {f"BatchedGemm_B{B}_K{K}_N{N}"} and {i[0] for i in by_seq[2]} == {f"BatchedGemm_B{B}_K{N}_N{K}"}
    o = cap.t["o"]
    assert [int(np.nonzero((o.transpose(1, 0, 2) == i[3]()[0]).all(axis=(1, 2)))[0][0]) for i in by_seq[2]] == [0, 1, 2, 4, 5, 6]
    for label, spec, key, args_of, want_of in insts:
        g, _ = stage.gate_instance(spec, key, args_of, want_of, 4_000_000)
        assert g["agree"] is True, (label, key, g)
    cap.t["ov"] = cap.t["ov"].copy()
    cap.t["ov"][1, 5, 2] ^= 1
    token4 = next(i for i in by_seq[2] if i[2][2] == 4)
    g, _ = stage.gate_instance(token4[1], token4[2], token4[3], token4[4], 4_000_000)
    assert g["agree"] is False and g["first_diff"] == K + 2  # head 1's word 2: W_UV's call has K words a head
