"""verity_vllm.program.predict.window_circuit's int8 linears: an exact-int8 checkpoint served with its linears
int8 (serve-openai's verity_int8) is the same class circuit with each decoder linear its quantizer unit and s8 GEMM slices,
pinned on `test_window_circuit`'s TINY class and windows, at 512 weight words a slice.

The norm after the attention and the one after the MLP are one `RMSNormFusedCuda_v2` template, whose ``x`` o_proj's 4 slices
write as 4 rows and down_proj's 8 as 8, and a port is cut once for every writer. So the window holds the first as that
template, ``x`` cut 4 ways, and cuts the second at its residual add: one `RMSNormFusedCudaAdd_v1` part, ``x`` cut 8 ways,
and its tail. The units and registered reads are pinned on its unwired lowering, the one its weights registration commits
(`window_circuit.lower`, ``wired=False``).
"""
from __future__ import annotations

import json
import re

import pytest
from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import CLASS, KEY, ROPE, SCAN_PIN, WINDOW, _alone, _checkpoint, _label, held
from verity_vllm.pipeline import advice as ADV
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict import window_builder as WB
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.common import _arr
from verity_vllm.program.predict.scan_state import SCAN_STATE
from verity_vllm.program.predict.window_circuit import Grain, window_circuit

from verity.ml.windows import interface as IF

#: 512 weight words a slice, so an s8 slice at K = 64 is one of TINY's 16-word heads
GRAIN = Grain(gemm_words=512, rows=256, select=64)
NORM = 'RMSNormFusedCuda_v2{N=64,EPS={"f64":"0x1.4f8b588e368f1p-17"}}'


def _registered():
    """TINY's exact-int8 checkpoint on a target serving its linears int8."""
    phi = S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": {**S.TINY, "quantization_config": dict(INT8_EXACT_QUANT)}},
                target={**S.RTXPRO6000, "int8_exact": "linears"})
    return S.Registered.of(phi, default_seed=7)


def _unwired(reg):
    return WC.lower(reg, CLASS, None, GRAIN, wired=False).circuit


def test_an_int8_linear_is_its_quantizer_and_s8_slices_at_the_gemms_weight_words():
    """Each int8 linear is its quantizer and its `ScaledMmS8_v1{K, N}` in slices of 512 // (K / 2) columns (an s8 weight row
    is K / 2 words: 16 columns at K = 64, 8 at K = 128), each slice's ``w`` and ``sw`` registered reads of the linear's s8
    rows and f32 row scales (layer 0's q/k/v, o and gate/up, each its own weight and scales from row 0, a slice a row on:
    512 words of codes, 32 of scales); the LM head stays bf16 `Gemm_v2`. Each quantizer is one `Int8RowQuant_v3{K, R}`
    unit reading ``x`` as its writers' R rows: q/k/v's and gate/up's the norm's row (R = 1), down_proj's the SiLU's (R = 1)
    and o_proj's the attention's 4 heads (R = 4).
    The window holds the norm both o_proj and down_proj write, cut at its residual add after down_proj (module docstring);
    its unwired lowering registers the weights and the scan state."""
    reg = _registered()
    window = window_circuit(reg, S.compile_steps(reg, *S._session(*WINDOW)), CLASS, GRAIN, provable=False)
    add = "RMSNormFusedCudaAdd_v1{N=64,C=64,AT=0}"
    assert {t: k for fn, k in window.templates if (t := _label(fn)).startswith("RMSNormFused")} == {
        "RMSNormFusedCuda_v2{N=64}": 20, add: 20, "RMSNormFusedCudaTail_v1{N=64}": 20}
    assert {t: c for t, c in window.cuts.items() if t.startswith("RMSNormFused")} == {NORM: {"x": 4}, add: {"x": 8}}
    circuit = _unwired(reg)
    n = {_label(fn): k for fn, k in circuit.templates}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    assert {t: k for t, k in n.items() if t.startswith(("Int8", "ScaledMmS8", "Gemm"))} == {
        "Int8RowQuant_v3{K=64,R=1}": 40, "Int8RowQuant_v3{K=64,R=4}": 20, "Int8RowQuant_v3{K=128,R=1}": 20,
        "ScaledMmS8_v1{K=64,N=16}": 560, "ScaledMmS8_v1{K=128,N=8}": 160,
        "Gemm_v2{K=64,N=8}": 320, "Gemm_v2{K=64,N=1}": 10}
    assert (sum(circuit.units.values()), circuit.wires, circuit.cuts) == (1568, (), {})
    mm = ids["ScaledMmS8_v1{K=64,N=16}"]
    reads = {(r.param, r.port, r.base, r.step[-1]) for r in circuit.registered.reads if r.reader == mm}
    layer = "model.layers.0.self_attn.qkv_proj", "model.layers.0.self_attn.o_proj", "model.layers.0.mlp.gate_up_proj"
    assert {(WB.weight_param(f"{w}.{t}", n), p, 0, n) for w in layer for t, p, n in (("weight", "w", 512),
                                                                                     ("weight_scale", "sw", 32))} <= reads
    params = dict(circuit.registered.params)
    assert params.pop(SCAN_STATE) == 16 * 4548 and len(params) == 25
    assert {p.split("/", 1)[0] for p in params} == {WC.WEIGHTS}
    assert IF.input_count(json.loads(json.dumps(circuit.description())), **held(circuit)) == {
        "units": 1568, "inputs": 10547104, "wired": 0, "registered": 9161856, "ingress": 0, "free": 1385248}


@pytest.mark.parametrize("flip, param", [("model.layers.1.mlp.down_proj.weight", "model.layers.1.mlp.down_proj.weight@512"),
                                         ("model.layers.0.self_attn.v_proj.weight_scale",
                                          "model.layers.0.self_attn.qkv_proj.weight_scale@32")])
def test_the_committers_registered_roots_hold_every_int8_weight_read_but_the_rotary_tables(tmp_path, flip, param):
    """On TINY's exact-int8 checkpoint the committer's registration (`weights_registration` at the window's grain under its
    key), held to the verifier's weights record (`registration_pins`), has a hiding root for each weight parameter the
    class circuit registers, each linear's s8 codes and f32 row scales their own, but the rotary table: pinned at them,
    and the scan state at its own root, every weight read but the rotary table's is registered. A committer serving one
    byte changed in a linear's codes, or in its scales, refuses to register that weight, and a window that serves it
    commits another root for that parameter alone, whose reads are then not registered. Pinned on the unwired lowering
    (module docstring), whose node-output reads are free for want of a wire."""
    reg = _registered()
    doc = json.loads(json.dumps(_unwired(reg).description()))
    d, man = _checkpoint(tmp_path, quant=INT8_EXACT_QUANT)
    registration = WW.weights_registration(reg, KEY, d, checkpoints_manifest=man, grain=GRAIN)
    pins = WW.registration_pins(registration, WW.weights_record(reg, d, man, grain=GRAIN))
    assert (ADV.registered_weights(registration), len(pins), list(registration["unrooted"])) == (pins, 24, [ROPE])
    assert set(pins) == set(_unwired(reg).registered.params) - {ROPE, SCAN_STATE}
    rope, both = _alone(doc, ROPE), {**pins, **SCAN_PIN}
    held_ = IF.input_count(doc, pins=both, roots=both)
    assert held_ == {"units": 1568, "inputs": 10547104, "wired": 0, "registered": 9161856 - rope, "ingress": 0,
                     "free": 1385248 + rope}
    served = WW.record_words(reg, *_checkpoint(tmp_path / "served", flip=flip, quant=INT8_EXACT_QUANT))
    refused = {param.rpartition("@")[0]: "served words that aren't the checkpoint's"}
    with pytest.raises(ValueError, match=re.escape(f"no weight registered: {refused} ")):
        WW.weights_registration(reg, KEY, d, served, checkpoints_manifest=man, grain=GRAIN)
    roots = {p: v.root.hex() for p, v in WW.weight_values(reg, served, KEY, GRAIN).items()}
    assert [p for p in pins if pins[p] != roots[p]] == [f"{WC.WEIGHTS}/{param}"]
    got, alone = IF.input_count(doc, pins=both, roots={**roots, **SCAN_PIN}), _alone(doc, f"{WC.WEIGHTS}/{param}")
    assert (got["registered"], got["free"]) == (held_["registered"] - alone, held_["free"] + alone) and alone > 0


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
