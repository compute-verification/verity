"""verity_vllm.program.predict.window_circuit's int8 linears: an exact-int8 checkpoint served with its linears
int8 (serve-openai's verity_int8) is the same class circuit with each decoder linear its quantizer unit and s8 GEMM slices,
pinned on `test_window_circuit`'s TINY class and windows, at 512 weight words a slice.

The norm after the attention and the one after the MLP are one `RMSNormFusedCuda_v2` template, whose ``x`` o_proj's 4 slices
write as 4 rows and down_proj's 8 as 8, and a port is cut once for every writer. So the window holds the first as that
template, ``x`` cut 4 ways, and cuts the second at its residual add: one `RMSNormFusedCudaAdd_v2{R=8}` part, ``x`` cut 8 ways,
and its tail. The units and registered reads are pinned on its lowering, the one its weights registration commits
(`window_circuit.lower`).
"""
from __future__ import annotations

import json
import re

import numpy as np
import pytest
from verity.commitments import hm96
from verity.commitments import registered_value as RV
from verity.commitments.periods import interface as IF

from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import (
    CLASS,
    ROPE,
    SCAN_PIN,
    FULL,
    _alone,
    _checkpoint,
    _label,
    _refused,
    _register,
    held,
    wired,
    wired_matches_by_hand,
)
from verity_vllm.pipeline import advice as ADV
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict import window_builder as WB
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.common import _arr
from verity_vllm.program.predict.scan_state import SCAN_STATE
from verity_vllm.program.predict.window_circuit import Grain, window_circuit

#: 512 weight words a slice, so an s8 slice at K = 64 is one of TINY's 16-word heads
GRAIN = Grain(gemm_words=512, rows=256, select=64)
NORM = 'RMSNormFusedCuda_v2{N=64,EPS={"f64":"0x1.4f8b588e368f1p-17"}}'


def _registered():
    """TINY's exact-int8 checkpoint on a target serving its linears int8."""
    phi = S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": {**S.TINY, "quantization_config": dict(INT8_EXACT_QUANT)}},
                target={**S.RTXPRO6000, "int8_exact": "linears"})
    return S.Registered.of(phi, default_seed=7)


def _lowered(reg):
    return WC.lower(reg, CLASS, None, GRAIN).circuit


def test_an_int8_linear_is_its_quantizer_and_s8_slices_at_the_gemms_weight_words():
    """Each int8 linear is its quantizer and its `ScaledMmS8_v1{K, N}` in slices of 512 // (K / 2) columns (an s8 weight row
    is K / 2 words: 16 columns at K = 64, 8 at K = 128), each slice's ``w`` and ``sw`` registered reads of the linear's s8
    rows, recut from its u16 words to bytes, and f32 row scales, registered at `WC.WIDE_FORMAT` (layer 0's q/k/v, o and gate/up, each its own weight and scales from row 0, a slice a row on:
    512 words of codes, 32 of scales); the LM head stays bf16 `Gemm_v2`. Each quantizer is one `Int8RowQuant_v3{K, R}`
    unit reading ``x`` as its writers' R rows: q/k/v's and gate/up's the norm's row (R = 1), down_proj's the SiLU's (R = 1)
    and o_proj's the attention's 4 heads (R = 4).
    The window holds the norm both o_proj and down_proj write, cut at its residual add after down_proj (module docstring);
    its lowering registers the weights and the scan state."""
    reg = _registered()
    window = window_circuit(reg, S.compile_steps(reg, *S._session(*FULL)), CLASS, GRAIN)
    add = "RMSNormFusedCudaAdd_v2{N=64,C=64,AT=0,R=8}"
    assert {t: k for fn, k in window.templates if (t := _label(fn)).startswith("RMSNormFused")} == {
        "RMSNormFusedCuda_v2{N=64}": 20, add: 20, "RMSNormFusedCudaTail_v1{N=64}": 20}
    assert {t: c for t, c in window.cuts.items() if t.startswith("RMSNormFused")} == {NORM: {"x": 4}, add: {"x": 8}}
    circuit = _lowered(reg)
    n = {_label(fn): k for fn, k in circuit.templates}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    assert {t: k for t, k in n.items() if t.startswith(("Int8", "ScaledMmS8", "Gemm"))} == {
        "Int8RowQuant_v3{K=64,R=1}": 40, "Int8RowQuant_v3{K=64,R=4}": 20, "Int8RowQuant_v3{K=128,R=1}": 20,
        "ScaledMmS8_v1{K=64,N=16}": 560, "ScaledMmS8_v1{K=128,N=8}": 160,
        "Gemm_v2{K=64,N=8}": 320, "Gemm_v2{K=64,N=1}": 10}
    assert (sum(circuit.units.values()), len(circuit.wires)) == (1568, 173)
    mm = ids["ScaledMmS8_v1{K=64,N=16}"]
    reads = {(r.param, r.port, r.base, r.step[-1]) for r in circuit.registered.reads if r.reader == mm}
    layer = "model.layers.0.self_attn.qkv_proj", "model.layers.0.self_attn.o_proj", "model.layers.0.mlp.gate_up_proj"
    assert {(WB.weight_param(f"{w}.{t}", n), p, 0, n) for w in layer for t, p, n in (("weight", "w", 512),
                                                                                     ("weight_scale", "sw", 32))} <= reads
    params = dict(circuit.registered.params)
    assert circuit.registered.formats == {p: WC.WIDE_FORMAT for p in params if "weight_scale" in p} != {}
    assert params.pop(SCAN_STATE) == 16 * 4548 and len(params) == 25
    assert {p.split("/", 1)[0] for p in params} == {WC.WEIGHTS}
    assert IF.input_count(json.loads(json.dumps(circuit.description())), **held(circuit)) == {
        "units": 1568, "inputs": 10547104, "wired": 1385248, "registered": 9161856, "ingress": 0, "free": 0}


@pytest.mark.parametrize("flip, param", [("model.layers.1.mlp.down_proj.weight", "model.layers.1.mlp.down_proj.weight@512"),
                                         ("model.layers.0.self_attn.v_proj.weight_scale",
                                          "model.layers.0.self_attn.qkv_proj.weight_scale@32")])
def test_the_committers_registered_roots_hold_every_int8_weight_read_but_the_rotary_tables(tmp_path, flip, param):
    """On TINY's exact-int8 checkpoint the committer's registration (`weights_registration` at the window's grain), held to the verifier's weights record (`registration_pins`), has a hiding root for each weight parameter the
    class circuit registers, each linear's s8 codes and f32 row scales their own, but the rotary table: pinned at them,
    and the scan state at its own root, every weight read but the rotary table's is registered. A committer serving one
    byte changed in a linear's codes, or in its scales, refuses to register that weight, and a window that serves it is
    refused by that parameter's name alone, publishing nothing of it. Pinned on the lowering (module docstring), whose
    node-output reads are wired."""
    reg = _registered()
    doc = json.loads(json.dumps(_lowered(reg).description()))
    d, man = _checkpoint(tmp_path, quant=INT8_EXACT_QUANT)
    registration, rpath = _register(tmp_path / "committer", reg, d, checkpoints_manifest=man, grain=GRAIN)
    pins = WW.registration_pins(registration, WW.weights_record(reg, d, man, grain=GRAIN))
    assert (ADV.registered_weights(registration), len(pins), list(registration["unrooted"])) == (pins, 24, [ROPE])
    assert set(pins) == set(_lowered(reg).registered.params) - {ROPE, SCAN_STATE}
    rope, both = _alone(doc, ROPE), {**pins, **SCAN_PIN}
    held_ = IF.input_count(doc, pins=both, roots=both)
    assert held_ == {"units": 1568, "inputs": 10547104, "wired": 1385248, "registered": 9161856 - rope, "ingress": 0,
                     "free": rope}
    served = WW.record_words(reg, *_checkpoint(tmp_path / "served", flip=flip, quant=INT8_EXACT_QUANT))
    refused = {param.rpartition("@")[0]: "served words that aren't the checkpoint's"}
    with pytest.raises(ValueError, match=re.escape(f"no weight registered: {refused} ")):
        WW.weights_registration(reg, tmp_path / "refused.salts", d, served, checkpoints_manifest=man, grain=GRAIN)
    with pytest.raises(ValueError) as e:
        WW.weight_values(reg, served, rpath, GRAIN)
    assert _refused(e) == [f"{WC.WEIGHTS}/{param}"]


@pytest.mark.slow
def test_the_int8_class_wired_program_reads_the_s8_weights_as_registered_words_and_is_its_members_fed_by_hand():
    """Each s8 slice (`ScaledMmS8_v1`) reads its weight rows as the weights registration's u16 words recut to bytes and its
    row scales as 32-bit leaves, so the class circuit is one wired Program with no free row. Slow (about two minutes): only the
    whole int8 class holds the s8 reads."""
    circuit = WC.class_circuit(_registered(), CLASS, None, GRAIN)
    prog, _ = wired(circuit)
    assert prog.annotations["free"] == []
    wired_matches_by_hand(circuit, prog)


def test_an_int8_linears_row_scales_commit_and_open_as_32_bit_leaves():
    """A row-scale parameter's registered value at its format's width (`WC.WIDE_FORMAT`) holds each row's u16 words as
    32-bit leaves, low word first, under a domain naming ``word_bits`` 32: its entry says so, the class circuit registers
    its words, and a row opens its root through its path and binds its scales; the same row under the 16-bit domain does
    not open it."""
    circuit = _lowered(_registered())
    p = min(q for q in circuit.registered.params if "weight_scale" in q)
    assert IF.value_format(circuit.registered, p) == (IF.V2, 32)
    w = WW._param(p)[1]
    scales = np.arange(1, 1 + 3 * w // 2, dtype=np.float32) / 64
    v = WW.weight_value("ab" * 64, p, scales.view(np.uint16), word_bits=32)
    assert v.entry() == {"value": p, "root": v.root.hex(), "leaves": 3, "words": w // 2, "word_bits": 32,
                         "domain": RV.domain("ab" * 64, p, w // 2, 3, 32).domain_id.hex()}
    assert v.words * v.leaves * v.word_bits // 16 == 3 * w
    rows, salts, head = v.statement([1])
    assert rows == [tuple(scales.view(np.uint32)[w // 2:w].tolist())] and head["positions"] == [1]
    assert v.bcs[1] == hm96.SHA512.commit_string(hm96.SHA512.default_key, RV.row_digest(rows[0], 32, v.schema), salts[0])

    def opens(dom) -> bool:
        digest, i = RV.leaf(dom, 1, v.bcs[1], v.schema), 1
        for level, sib in enumerate(bytes.fromhex(h) for h in [head["paths"][0][k:k + 128]
                                                               for k in range(0, len(head["paths"][0]), 128)]):
            digest = dom.node(level, i >> 1, *((digest, sib) if i % 2 == 0 else (sib, digest)))
            i >>= 1
        return digest == v.root

    assert opens(v.domain) and not opens(RV.domain("ab" * 64, p, w, 3))


def test_a_word_is_whole_leaves_of_one_byte_width():
    """Weight and node-output reads are u16 words: s8 leaves two a word, f32 two words a leaf; a read starting inside a word,
    and a field whose leaves don't fill whole words, are refused."""
    spans = WB._spans("w", [("a", _arr((4,), 8)), ("b", _arr((2,), 32))])
    assert spans == [(0, 0, 8), (4, 2, 32), (6, 6, 0)]
    assert [WB._word(spans, leaf, "r") for leaf in (0, 2, 4, 5, 6)] == [0, 1, 2, 4, 6]
    with pytest.raises(Unexpressible, match="r: leaf 1 starts inside a u16 word of 8-bit leaves"):
        WB._word(spans, 1, "r")
    with pytest.raises(Unexpressible, match=r"w a's 3 leaves are \[8\] bits, not one byte width filling whole u16 words"):
        WB._spans("w", [("a", _arr((3,), 8))])
