"""`numerics.fp8_twins.np_online_weight_quant`, the CPU re-quantisation of a bf16 checkpoint's weights under vLLM's
`Fp8PerTensorOnlineLinearMethod`, gives the words an H100 production run committed (profile 0ca394ec17b9, layer 0's
qkv_proj of the FP8 B1 program, `registry.fp8`).

`fixtures/fp8_online_weight_quant_l0_qkv.npz`:
* `w`, `rows`: rows of the checkpoint's fused [2048, 1536] bf16 qkv_proj weight (Qwen/Qwen2.5-1.5B at revision
  8faed761d45a263340a0528343f099c05c9a4323, `model.safetensors` sha256 a961db72e75d52b1…, layer 0's q_proj, k_proj and
  v_proj weights stacked along dim 0; the stacked tensor's bytes have sha256 f1a1617d411206b1…): the committed rows and
  row 254, which holds the tensor's largest |w|, so the scale over these rows is the whole tensor's;
* `cols`, `w_q`, `s_w`: the production run's committed e4m3 rows at `cols` and its committed f32 weight scale, copied from
  the vLLM integration's `tests/program/data/fp8_production_row_l0_qkv.npz`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verity.ml.definitions.numerics import fp8_twins as T

FIXTURE = Path(__file__).parent / "fixtures" / "fp8_online_weight_quant_l0_qkv.npz"


def test_online_weight_quant_is_the_committed_words():
    d = np.load(FIXTURE)
    w_q, s_w = T.np_online_weight_quant(d["w"])
    at = [d["rows"].tolist().index(c) for c in d["cols"].tolist()]
    assert np.float32(s_w).view(np.uint32) == d["s_w"][0]
    assert np.array_equal(w_q[at], d["w_q"])


def test_the_scale_is_the_whole_tensors():
    """Without row 254 the committed rows alone give another scale and other words: the fixture's extra row is what makes
    its scale the tensor's."""
    d = np.load(FIXTURE)
    at = [d["rows"].tolist().index(c) for c in d["cols"].tolist()]
    w_q, s_w = T.np_online_weight_quant(d["w"][at])
    assert np.float32(s_w).view(np.uint32) != d["s_w"][0]
    assert not np.array_equal(w_q, d["w_q"])
