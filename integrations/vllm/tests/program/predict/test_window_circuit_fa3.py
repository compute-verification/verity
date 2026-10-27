"""verity_vllm.program.predict.window_circuit on sm_90 (VLLM-127): the H200's Program binds FA3's `Attention_v4`, which has
no per-key-block cut, so each head is one unit of `AttentionHead_v4{T}` reading its KV group's keys and values each as one
`AllGather_v1{T, D}` row while T is at most `KEYS`, and past it the attention is refused by name. Pinned on
`test_window_circuit`'s TINY class, windows and grain.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import CLASS, GRAIN, OTHER, WINDOW, _blocks, _bytes, _label, _labels, _source
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_circuit import class_circuit, window_circuit

from verity.ml.windows import interface as IF

#: the H200's declared target (`pins/NVIDIA_H200.json`): FA3, so `Attention_v4` at D = 16's kBlockN of 128
H200 = {"compute_capability": [9, 0], "num_sms": 132}
HEAD = "AttentionHead_v4{{T={},D=16,BN=128,MASKED_FROM=0}}"


def _registered():
    return S.Registered.of(S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": S.TINY}, target=H200), default_seed=7)


def test_an_fa3_attention_is_its_heads_each_reading_its_kv_groups_gathers():
    """Position p's attention (T = p + 1) is `AttentionHead_v4{T}`, a unit a head (2 slots, 2 layers, 4 heads: 16 a T).
    At position 4 slot 0's heads 0 and 1 read KV group 0's keys as one `AllGather_v1{5, 16}` row over the RoPE'd keys of
    positions 0..4 (RoPE units 24p + 8), slot 1's head 0 its own (24p + 10), and the values one over each position's
    gathered value slices. Two windows of the class are one circuit, blocks included, with the weights pinned no input row
    is free, and every wire reads its own slot at its step or before."""
    reg = _registered()
    circuit = window_circuit(reg, S.compile_steps(reg, *S._session(*WINDOW)), CLASS, GRAIN)
    other = window_circuit(reg, S.compile_steps(reg, *S._session(*OTHER)), CLASS, GRAIN)
    assert _bytes(circuit) == _bytes(other) and _blocks(circuit) == _blocks(other)
    n = {_label(fn): k for fn, k in circuit.templates}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    assert {t: k for t, k in n.items() if t.startswith(("Attention", "AttnBlock", "AttnHeadEnd"))} == {
        HEAD.format(t): 16 for t in range(1, 6)}
    head, rope = ids[HEAD.format(5)], ids["RoPEHead_v1{D=16}"]
    gather5, pair = ids["AllGather_v1{WORLD=5,N=16}"], ids["AllGather_v1{WORLD=2,N=8}"]
    assert [_source(circuit, head, "q", u) for u in range(8)] == [(rope, 96 + u, "o0") for u in range(8)]
    assert [_source(circuit, head, "k", u) for u in range(8)] == [(gather5, u // 2, "o0") for u in range(8)]
    assert [_source(circuit, head, "v", u) for u in range(8)] == [(gather5, 4 + u // 2, "o0") for u in range(8)]
    assert [_source(circuit, gather5, "parts", 0, r) for r in range(5)] == [(rope, 24 * p + 8, "o0") for p in range(5)]
    assert [_source(circuit, gather5, "parts", 2, r) for r in range(5)] == [(rope, 24 * p + 10, "o0") for p in range(5)]
    assert {_source(circuit, gather5, "parts", 4, r)[0] for r in range(5)} == {pair}
    assert IF.input_count(json.loads(json.dumps(circuit.description())), pins={WC.WEIGHTS})["free"] == 0
    labels = _labels(circuit)
    assert all((lab >= 0).all() for lab in labels.values())
    for w in circuit.wires:
        i = np.arange(w.units, dtype=np.int64)
        reader, writer = labels[w.reader][w.at + i // w.cut[1]], labels[w.writer][w.source(i)]
        assert (reader[:, 0] == writer[:, 0]).all() and (writer[:, 1] <= reader[:, 1]).all(), w.doc()


def test_an_fa3_head_past_keys_is_refused_by_name(monkeypatch):
    """A head of more keys than `KEYS` (here 4, so position 4's T of 5) is refused by name: only v18 and v19 are cut by key
    block."""
    monkeypatch.setattr(WC, "KEYS", 4)
    with pytest.raises(Unexpressible, match=r"Attention_v4\{T=5,.*\}: a head of 5 keys, past the 4 one `AllGather_v1` reads; "
                                            r"the window's lowering cuts only `Attention_v18`'s and v19's heads by key block, "
                                            r"not v4's"):
        class_circuit(_registered(), CLASS, None, GRAIN)
