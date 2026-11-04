"""`numerics.fp8_twins.np_online_weight_quant`, the CPU re-quantisation of a bf16 checkpoint's weights under vLLM's
`Fp8PerTensorOnlineLinearMethod`, gives the words an H100 production run committed for layer 0's qkv_proj of the FP8 B1
program (`registry.fp8`).

`fixtures/fp8_online_weight_quant_l0_qkv.npz`:
* `w`, `rows`: rows of the checkpoint's fused [2048, 1536] bf16 qkv_proj weight (`CHECKPOINT`: layer 0's q_proj, k_proj and
  v_proj weights stacked along dim 0): the committed rows and the row that holds the tensor's largest |w|, so the scale
  over these rows is the whole tensor's;
* `cols`, `w_q`, `s_w`: the production run's committed e4m3 rows at `cols` and its committed f32 weight scale, from the vLLM
  integration's `PRODUCTION`.

`python verity/ml/definitions/tests/test_fp8_twins.py --write` rewrites the fixture from those two sources.
"""
from __future__ import annotations

import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

from verity.ml.definitions.numerics import fp8_twins as T

FIXTURE = Path(__file__).parent / "fixtures" / "fp8_online_weight_quant_l0_qkv.npz"
CHECKPOINT = "https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/8faed761d45a263340a0528343f099c05c9a4323/model.safetensors"
PRODUCTION = Path(__file__).resolve().parents[4] / "integrations/vllm/tests/program/data/fp8_production_row_l0_qkv.npz"


def test_online_weight_quant_is_the_committed_words():
    d = np.load(FIXTURE)
    w_q, s_w = T.np_online_weight_quant(d["w"])
    at = [d["rows"].tolist().index(c) for c in d["cols"].tolist()]
    assert np.float32(s_w).view(np.uint32) == d["s_w"][0]
    assert np.array_equal(w_q[at], d["w_q"])


def test_the_scale_is_the_whole_tensors():
    """Without the row of the tensor's largest |w| the committed rows alone give another scale and other words: the
    fixture's extra row is what makes its scale the tensor's."""
    d = np.load(FIXTURE)
    at = [d["rows"].tolist().index(c) for c in d["cols"].tolist()]
    w_q, s_w = T.np_online_weight_quant(d["w"][at])
    assert np.float32(s_w).view(np.uint32) != d["s_w"][0]
    assert not np.array_equal(w_q, d["w_q"])


def _qkv_weight() -> np.ndarray:
    """`CHECKPOINT`'s layer-0 q_proj, k_proj and v_proj weights stacked along dim 0, as bf16 words, read by byte range."""
    def read(lo: int, hi: int) -> bytes:
        with urllib.request.urlopen(urllib.request.Request(CHECKPOINT, headers={"Range": f"bytes={lo}-{hi - 1}"})) as r:
            return r.read()

    n = struct.unpack("<Q", read(0, 8))[0]
    header = json.loads(read(8, 8 + n))
    parts = []
    for p in "qkv":
        t = header[f"model.layers.0.self_attn.{p}_proj.weight"]
        lo, hi = t["data_offsets"]
        parts.append(np.frombuffer(read(8 + n + lo, 8 + n + hi), dtype="<u2").reshape(t["shape"]))
    return np.concatenate(parts)


def write() -> None:
    w = _qkv_weight()
    prod = np.load(PRODUCTION)
    top = int(np.abs((w.astype(np.uint32) << 16).view(np.float32)).max(axis=1).argmax())
    rows = np.array(sorted({*prod["cols"].tolist(), top}), dtype=np.int64)
    np.savez_compressed(FIXTURE, rows=rows, w=w[rows], cols=prod["cols"], w_q=prod["w_rows"], s_w=prod["sw"])


if __name__ == "__main__" and "--write" in sys.argv:
    write()
