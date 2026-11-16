"""The encoded-stack KV-append capture without a GPU: its CPU references against the word codec's and the Definition's.

The GPU run is ``tests/properties/encoded_kv_append_capture_gpu.py`` on one cc 12.0 GPU (`stack.sh` MODE=capture).
Here: the payload and word-order mapping equal `kv.py`'s byte layout and the append kernel's source indexing, the
encoded reference round-trips through ``kv.decode_word`` at the appended store indices, and the Definition comparison
(`KvCacheAppendE4m3_v1`'s CPU reference on the same inputs) flags exactly the elements the 8-bit quantization would
change and none the 16-bit cache stores exactly.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "encoded_kv_append_capture_gpu", HERE / "properties" / "encoded_kv_append_capture_gpu.py")
X = sys.modules["encoded_kv_append_capture_gpu"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)
KV = X.KV

HEADS, DIM = KV.KV_HEADS, KV.HEAD_DIM


def rows(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 1 << 16, HEADS * DIM).astype(np.uint16),
            rng.integers(0, 1 << 16, HEADS * DIM).astype(np.uint16))


def test_payload_words_are_k_then_v_bytes():
    """Word w's payload is bytes [w*S, (w+1)*S) of K of every head then V of every head, little-endian."""
    f = KV.field("p1024")
    k, v = rows(0)
    data = k.astype("<u2").tobytes() + v.astype("<u2").tobytes()
    words = X.payload_words(k, v, f)
    s = f.word_bytes
    assert len(words) == KV.token_layer_words(f)
    assert all(m == int.from_bytes(data[s * w:s * (w + 1)], "little") for w, m in enumerate(words))


def test_pg_wi_matches_the_kernel_source_mapping():
    """The kernel reads word (g, j) from the K row (j < hw/2) or the V row at g*DIM + (j % (hw/2))*(wb/2) elements;
    pg_wi must send exactly those bytes to the payload word that holds them."""
    f = KV.field("p1024")
    wb = f.word_bytes
    hw = 4 * DIM // wb                                  # the library's: hw * wb == 4 * HEAD_DIM bytes, a (slot, head)'s K and V
    head_words, half = DIM * 2 // wb, KV.token_layer_words(f) // 2
    k, v = rows(1)
    words = X.payload_words(k, v, f)
    seen = set()
    for g in range(HEADS):
        for j in range(hw):
            wi = X.pg_wi(g, j, hw, head_words, half)
            seen.add(wi)
            src = k if j < hw // 2 else v
            start = g * DIM + (j % (hw // 2)) * (wb // 2)
            got = src[start:start + wb // 2].astype("<u2").tobytes()
            assert int.from_bytes(got, "little") == words[wi]
    assert seen == set(range(KV.token_layer_words(f)))


def test_encode_reference_round_trips():
    """The reference words decode back to the payloads at the appended store indices (p512 for speed)."""
    key = KV.key("p512", b"\x07" * KV.SEED_BYTES, 1, mask=True)
    k, v = rows(2)
    slot, layer, layers, base = 37, 5, 32, 1_000_000
    ref = X.encode_reference(k, v, slot, layer, layers, base, key)
    tlw = KV.token_layer_words(key.field)
    w0 = base + (slot * layers + layer) * tlw
    want = X.payload_words(k, v, key.field)
    assert [KV.decode_word(y, w0 + w, key) for w, y in enumerate(ref)] == want
    assert ref != want  # the append writes codec words, not the payloads


def test_definition_report_flags_exactly_the_lossy_elements():
    """At scale 1.0: a value E4M3 represents exactly is kept, NaN maps to NaN, and the first element the 8-bit
    quantization would change is localized with its stored word and Definition byte."""
    one = X.F32_ONE
    k = np.array([0x3F80, 0x7FC1, 0x0000, 0x3F81], dtype=np.uint16)   # 1.0, NaN, +0, 1.0078125
    v = np.array([0xBF80, 0x4280], dtype=np.uint16)                   # -1.0, 64.0
    r = X.definition_report(k, v, one, one)
    assert r["elements"] == 6
    assert r["lossy_under_definition"] == 1
    assert r["first_difference"] == {"which": "key", "element": 3, "stored_bf16_word": "0x3f81",
                                     "stored_value": X.bf16_value(0x3F81), "definition_byte": "0x38",
                                     "definition_value": 1.0}


def test_definition_report_dequantizes_by_the_scale():
    """At scale 2.0 the Definition stores e4m3(x / 2) and its reader sees that times 2: 1.0 and 1.25 survive
    (their halves are exact E4M3 values), 1.0078125 does not."""
    two = 0x40000000
    r = X.definition_report(np.array([0x3F80], dtype=np.uint16), np.array([0x3FA0], dtype=np.uint16), two, two)
    assert r["elements"] == 2 and r["lossy_under_definition"] == 0
    r = X.definition_report(np.array([0x3F81], dtype=np.uint16), np.array([], dtype=np.uint16), two, two)
    assert r["lossy_under_definition"] == 1
