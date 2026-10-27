"""verity_vllm.program.predict.window_circuit's int8 linears (VLLM-33's G3): an exact-int8 checkpoint served with its linears
int8 (serve-openai's verity_int8) is the same class circuit with each decoder linear its quantizer unit and s8 GEMM slices,
pinned on `test_window_circuit`'s TINY class, windows and grain.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest
from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import CLASS, GRAIN, OTHER, WINDOW, _blocks, _label, _labels, _source, held
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict import window_builder as WB
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.common import _arr
from verity_vllm.program.predict.window_circuit import window_circuit

from verity.ml.windows import interface as IF


def _registered():
    """TINY's exact-int8 checkpoint on a target serving its linears int8."""
    phi = S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": {**S.TINY, "quantization_config": dict(INT8_EXACT_QUANT)}},
                target={**S.RTXPRO6000, "int8_exact": "linears"})
    return S.Registered.of(phi, default_seed=7)


def test_an_int8_linear_is_its_quantizer_and_s8_slices_at_the_gemms_weight_words():
    """Each int8 linear is one `Int8RowQuant_v1{K}` unit and its `ScaledMmS8_v1{K, N}` in slices of 512 // (K / 2) columns (an
    s8 weight row is K / 2 words: 16 columns at K = 64, 8 at K = 128), each slice's ``x`` and ``sx`` one quantizer unit's codes
    and scale and its ``w`` and ``sw`` registered reads of the linear's s8 rows and f32 row scales (layer 0's q/k/v, o and
    gate/up, each its own weight and scales from row 0, a slice a row on: 512 words of codes, 32 of scales); the LM head
    stays bf16 `Gemm_v2`.
    Two windows of the class are still one circuit, blocks included, with the weights pinned no input row is free, and every
    wire reads its own slot at its step or before."""
    reg = _registered()
    circuit = window_circuit(reg, S.compile_steps(reg, *S._session(*WINDOW)), CLASS, GRAIN)
    other = window_circuit(reg, S.compile_steps(reg, *S._session(*OTHER)), CLASS, GRAIN)
    assert IF.circuit_bytes(circuit.description()) == IF.circuit_bytes(other.description())
    n = {_label(fn): k for fn, k in circuit.templates}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    assert {t: k for t, k in n.items() if t.startswith(("Int8RowQuant", "ScaledMmS8", "Gemm"))} == {
        "Int8RowQuant_v1{K=64}": 60, "Int8RowQuant_v1{K=128}": 20, "ScaledMmS8_v1{K=64,N=16}": 560,
        "ScaledMmS8_v1{K=128,N=8}": 160, "Gemm_v2{K=64,N=8}": 320, "Gemm_v2{K=64,N=1}": 10}
    # slices a quantizer: K = 64's q/k/v (N 128), o (64) and gate/up (256) at 16 columns, K = 128's down (64) at 8
    for (K, N), slices in (((64, 16), [8, 4, 16]), ((128, 8), [8])):
        mm, q = ids[f"ScaledMmS8_v1{{K={K},N={N}}}"], ids[f"Int8RowQuant_v1{{K={K}}}"]
        x = [_source(circuit, mm, "x", u) for u in range(n[f"ScaledMmS8_v1{{K={K},N={N}}}"])]
        sx = [_source(circuit, mm, "sx", u) for u in range(len(x))]
        assert {(w, o) for w, _, o in x} == {(q, "o0")} and {(w, o) for w, _, o in sx} == {(q, "o1")}
        assert [u for _, u, _ in x] == [u for _, u, _ in sx]
        assert sorted(Counter(u for _, u, _ in x).values()) == sorted(slices * 20)
    mm = ids["ScaledMmS8_v1{K=64,N=16}"]
    reads = {(r.param, r.port, r.base, r.step[-1]) for r in circuit.registered.reads if r.reader == mm}
    layer = "model.layers.0.self_attn.qkv_proj", "model.layers.0.self_attn.o_proj", "model.layers.0.mlp.gate_up_proj"
    assert {(WB.weight_param(f"{w}.{t}", n), p, 0, n) for w in layer for t, p, n in (("weight", "w", 512),
                                                                                     ("weight_scale", "sw", 32))} <= reads
    assert len(circuit.registered.params) == 24
    assert {p.split("/", 1)[0] for p in circuit.registered.params} == {WC.WEIGHTS}
    assert IF.input_count(json.loads(json.dumps(circuit.description())), **held(circuit))["free"] == 0
    assert _blocks(circuit) == _blocks(other)
    labels = _labels(circuit)
    assert all((lab >= 0).all() for lab in labels.values())
    for w in circuit.wires:
        i = np.arange(w.units, dtype=np.int64)
        reader, writer = labels[w.reader][w.at + i // w.cut[1]], labels[w.writer][w.source(i)]
        assert (reader[:, 0] == writer[:, 0]).all() and (writer[:, 1] <= reader[:, 1]).all(), w.doc()


def test_a_word_is_whole_leaves_of_one_byte_width():
    """Weight and node-output reads are u16 words: s8 leaves two a word, f32 two words a leaf; a read starting inside a word,
    and a field whose leaves don't fill whole words, are refused."""
    spans = WC._spans("w", [("a", _arr((4,), 8)), ("b", _arr((2,), 32))])
    assert spans == [(0, 0, 8), (4, 2, 32), (6, 6, 0)]
    assert [WC._word(spans, leaf, "r") for leaf in (0, 2, 4, 5, 6)] == [0, 1, 2, 4, 6]
    with pytest.raises(Unexpressible, match="r: leaf 1 starts inside a u16 word of 8-bit leaves"):
        WC._word(spans, 1, "r")
    with pytest.raises(Unexpressible, match=r"w a's 3 leaves are \[8\] bits, not one byte width filling whole u16 words"):
        WC._spans("w", [("a", _arr((3,), 8))])
