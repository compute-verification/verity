"""verity_vllm.program.predict.window_circuit on sm_90 (VLLM-127): the H200's Program binds FA3's `Attention_v4`, which the
lowering cuts by key block as `Attention_v20` (`CUTS`): a head's units are its first visited block `AttnBlockFirst_v1` and its
end `AttnHeadEnd_v1` at FA3's dot and reciprocal, and past one key block (TINY's kBlockN on sm_90 is 128) each full block
`AttnBlock_v4{FIRST=False, CHECK=False}`, reading that block's gather, made once. An attention version with no cut
(`Attention_v2`, which `fa3_construction = "guard-every-block"` binds) is a unit a head of its own head Definition, one
`AllGather_v1{T, D}` a KV group, while T is at most `KEYS`, and past it is refused by name. With
``VERITY_VLLM_FA3_LLAMA_1B=1``, int8 Llama-3.2-1B on sm_90 lowers whole at 4,32,4 and at the class of record. Pinned on
`test_window_circuit`'s TINY class, windows and grain.
"""
from __future__ import annotations

import json
import os
import resource
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import (CLASS, GRAIN, OTHER, WINDOW, _blocks, _bytes, _label, _labels, _source,
                                                       _unit_rows)
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_circuit import CLASS_OF_RECORD, Grain, WindowClass, class_circuit, window_circuit

from verity.ml.windows import interface as IF

#: the H200's declared target (`pins/NVIDIA_H200.json`): FA3, so `Attention_v4` at D = 16's kBlockN of 128
H200 = {"compute_capability": [9, 0], "num_sms": 132}
DOT, INV = "DOT=HopperBF16WgmmaDot16_v1", "INV=Fa3InvSum_v1"
LLAMA_1B = Path(__file__).resolve().parent / "fixtures/llama32-1b.config.json"


def _registered(target: dict = H200, config: dict = S.TINY):
    return S.Registered.of(S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": config}, target=target), default_seed=7)


def _wired_in_slot_and_step(circuit) -> None:
    labels = _labels(circuit)
    assert all((lab >= 0).all() for lab in labels.values())
    for w in circuit.wires:
        i = np.arange(w.units, dtype=np.int64)
        reader, writer = labels[w.reader][w.at + i // w.cut[1]], labels[w.writer][w.source(i)]
        assert (reader[:, 0] == writer[:, 0]).all() and (writer[:, 1] <= reader[:, 1]).all(), w.doc()


def test_an_fa3_head_is_v20s_calls_over_its_kv_groups_gathers():
    """Position p's heads (T = p + 1, one key block) are `Attention_v20`'s Calls, `AttnBlockFirst_v1{NVIS=T}` at FA3's dot and
    `AttnHeadEnd_v1` at FA3's reciprocal, a unit a head each (2 slots, 2 layers, 4 heads: 16 a T). At position 4 slot 0's
    heads 0 and 1 read KV group 0's keys as one `AllGather_v1{5, 16}` row over the RoPE'd keys of positions 0..4 (RoPE units
    24p + 8), slot 1's head 0 its own (24p + 10), and the end reads its block's running sum and output. Two windows of the
    class are one circuit, blocks included, with the weights pinned no input row is free, and every wire reads its own slot
    at its step or before."""
    reg = _registered()
    circuit = window_circuit(reg, S.compile_steps(reg, *S._session(*WINDOW)), CLASS, GRAIN)
    other = window_circuit(reg, S.compile_steps(reg, *S._session(*OTHER)), CLASS, GRAIN)
    assert _bytes(circuit) == _bytes(other) and _blocks(circuit) == _blocks(other)
    n = Counter()
    for fn, k in circuit.templates:
        if fn.id.startswith(("Attention", "AttnBlock", "AttnHeadEnd")):
            n[fn.id] += k
    assert n == {**{f"AttnBlockFirst_v1{{D=16,NVIS={t},BN=128,{DOT}}}": 16 for t in range(1, 6)}, f"AttnHeadEnd_v1{{D=16,{INV}}}": 80}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    first, end = ids["AttnBlockFirst_v1{D=16,NVIS=5,BN=128}"], ids["AttnHeadEnd_v1{D=16}"]
    rope, gather5, pair = ids["RoPEHead_v1{D=16}"], ids["AllGather_v1{WORLD=5,N=16}"], ids["AllGather_v1{WORLD=2,N=8}"]
    assert [_source(circuit, first, "q", u) for u in range(8)] == [(rope, 96 + u, "o0") for u in range(8)]
    assert [_source(circuit, first, "k", u) for u in range(8)] == [(gather5, u // 2, "o0") for u in range(8)]
    assert [_source(circuit, first, "v", u) for u in range(8)] == [(gather5, 4 + u // 2, "o0") for u in range(8)]
    assert [_source(circuit, gather5, "parts", 0, r) for r in range(5)] == [(rope, 24 * p + 8, "o0") for p in range(5)]
    assert [_source(circuit, gather5, "parts", 2, r) for r in range(5)] == [(rope, 24 * p + 10, "o0") for p in range(5)]
    assert {_source(circuit, gather5, "parts", 4, r)[0] for r in range(5)} == {pair}
    assert [_source(circuit, end, p, 64 + 5) for p in ("l", "o")] == [(first, 5, "o1"), (first, 5, "o2")]
    assert IF.input_count(json.loads(json.dumps(circuit.description())), pins={WC.WEIGHTS})["free"] == 0
    _wired_in_slot_and_step(circuit)


@pytest.mark.slow
def test_past_one_key_block_an_fa3_head_reads_its_full_blocks_gather_made_once():
    """At T 129 (one Program of 129 positions at TINY of 2,048 positions, a grain of 256 rows) a head is its first block
    `AttnBlockFirst_v1{NVIS=1}` over key 128, the full block below it `AttnBlock_v4{CHECK=false}` (FA3's one-row decode guards
    only the first block it visits) over keys 0..127, reading the first block's m, l and o, and its end, reading the full
    block's. The full block's keys are one `AllGather_v1{128, 16}` a KV group, the row position 127's
    `AttnBlockFirst_v1{NVIS=128}` read when the block filled: a layer's 8 such gathers, no unit past the grain's rows and no
    free input row."""
    reg = _registered(config={**S.TINY, "max_position_embeddings": 2048})
    circuit = class_circuit(reg, WindowClass(1, 128, 2), 1, Grain(gemm_words=512, rows=256, select=64))
    n = {_label(fn): k for fn, k in circuit.templates}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    block, gather = "AttnBlock_v4{D=16,NVIS=128,FIRST=false,BN=128,CHECK=false}", "AllGather_v1{WORLD=128,N=16}"
    assert (n[block], n[gather], n["AttnBlockFirst_v1{D=16,NVIS=1,BN=128}"], n["AttnHeadEnd_v1{D=16}"]) == (8, 8, 16, 129 * 8)
    assert not [lab for lab in n if lab.startswith(("AllGather_v1{WORLD=129", "AttentionHead")) or "CHECK=true" in lab]
    first, full, end = ids["AttnBlockFirst_v1{D=16,NVIS=1,BN=128}"], ids[block], ids["AttnHeadEnd_v1{D=16}"]
    rope, filled = ids["RoPEHead_v1{D=16}"], ids["AttnBlockFirst_v1{D=16,NVIS=128,BN=128}"]
    assert _source(circuit, first, "k", 8) == (rope, 12 * 128 + 4, "o0")
    assert _source(circuit, full, "k", 0) == _source(circuit, full, "k", 1) == _source(circuit, filled, "k", 0) == (ids[gather], 0, "o0")
    assert [_source(circuit, full, p, 0) for p in ("m_prev", "l_prev", "o_prev")] == [(first, 8, o) for o in ("o0", "o1", "o2")]
    assert [_source(circuit, end, p, 8 * 128) for p in ("l", "o")] == [(full, 0, "o1"), (full, 0, "o2")]
    assert max(_unit_rows(circuit).values()) == 129 and IF.input_count(circuit.description(), pins={WC.WEIGHTS})["free"] == 0


#: FA3 declared to guard every block: `Attention_v2`, which has no per-key-block cut
GUARDED = {**H200, "fa3_construction": "guard-every-block"}
HEAD = "AttentionHead_v2{{T={},D=16,BN=128}}"


def test_an_attention_with_no_cut_is_its_heads_each_reading_its_kv_groups_gathers():
    """`Attention_v2`'s position p (T = p + 1) is `AttentionHead_v2{T}`, a unit a head (16 a T), whose keys and values are one
    `AllGather_v1{T, 16}` row a KV group, the same gathers v20's first block reads. Two windows of the class are one circuit,
    no input row is free, and every wire reads its own slot at its step or before."""
    reg = _registered(GUARDED)
    circuit = window_circuit(reg, S.compile_steps(reg, *S._session(*WINDOW)), CLASS, GRAIN)
    other = window_circuit(reg, S.compile_steps(reg, *S._session(*OTHER)), CLASS, GRAIN)
    assert _bytes(circuit) == _bytes(other) and _blocks(circuit) == _blocks(other)
    n = {_label(fn): k for fn, k in circuit.templates}
    assert {t: k for t, k in n.items() if t.startswith(("Attention", "AttnBlock", "AttnHeadEnd"))} == {
        HEAD.format(t): 16 for t in range(1, 6)}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    head, rope, gather5 = ids[HEAD.format(5)], ids["RoPEHead_v1{D=16}"], ids["AllGather_v1{WORLD=5,N=16}"]
    assert [_source(circuit, head, "q", u) for u in range(8)] == [(rope, 96 + u, "o0") for u in range(8)]
    assert [_source(circuit, head, "k", u) for u in range(8)] == [(gather5, u // 2, "o0") for u in range(8)]
    assert [_source(circuit, head, "v", u) for u in range(8)] == [(gather5, 4 + u // 2, "o0") for u in range(8)]
    assert IF.input_count(json.loads(json.dumps(circuit.description())), pins={WC.WEIGHTS})["free"] == 0
    _wired_in_slot_and_step(circuit)


def test_a_head_with_no_cut_past_keys_is_refused_by_name(monkeypatch):
    """A head of more keys than `KEYS` (here 4, so position 4's T of 5) of a version with no cut is refused by name."""
    monkeypatch.setattr(WC, "KEYS", 4)
    with pytest.raises(Unexpressible, match=r"Attention_v2\{T=5,.*\}: a head of 5 keys, past the 4 one `AllGather_v1` reads; "
                                            r"the window's lowering cuts a head by key block only at `Attention` versions "
                                            r"4, 18, 19, 20 \(`CUTS`\), not 2"):
        class_circuit(_registered(GUARDED), CLASS, None, GRAIN)


@pytest.mark.skipif(not os.environ.get("VERITY_VLLM_FA3_LLAMA_1B"),
                    reason="int8 Llama-3.2-1B on sm_90 at 4,32,4 and the class of record: the class of record some 25 minutes "
                           "and 45 GB on a pod; set VERITY_VLLM_FA3_LLAMA_1B=1")
@pytest.mark.parametrize("cls", [WindowClass(4, 32, 4), CLASS_OF_RECORD], ids=["4,32,4", "class-of-record"])
def test_int8_llama_3_2_1b_lowers_on_sm_90(cls):
    """int8 Llama-3.2-1B's window circuit on the H200 (exact-int8 linears, FA3's heads at D 64 and kBlockN 192) at CIR-95's
    grain: every head `Attention_v20`'s Calls (no whole `AttentionHead_v4`, no guarded full block), no unit past 256 rows in
    and out, and no free input row. Its numbers (units, free inputs, the most rows a unit, peak RSS, the heads' census) are
    printed as a JSON line."""
    config = {**json.loads(LLAMA_1B.read_text()), "quantization_config": dict(INT8_EXACT_QUANT)}
    circuit = class_circuit(_registered({**H200, "int8_exact": "linears"}, config), cls)
    rows = max(sum(map(len, IF.unit_rows(fn, circuit.cuts.get(IF.descriptor_id(fn))))) for fn, _ in circuit.templates)
    free = IF.input_count(circuit.description(), pins={WC.WEIGHTS})["free"]
    heads = Counter()
    for fn, k in circuit.templates:
        if fn.id.startswith(("Attention", "AttnBlock", "AttnHeadEnd")):
            heads[fn.id.split("{")[0] + ("" if fn.base_id != "AttnBlock_v4" else f"{{CHECK={str(fn.bindings['CHECK']).lower()}}}")] += k
    print(json.dumps({"class": [cls.slots, cls.prompt, cls.outputs, cls.lanes, cls.every], "positions": cls.positions,
                      "units": sum(circuit.units.values()), "templates": len(circuit.templates), "wires": len(circuit.wires),
                      "reads": len(circuit.registered.reads), "entries": len(circuit.blocks.inference), "free": free,
                      "most_rows": rows, "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
                      "heads": dict(heads)}, sort_keys=True), flush=True)
    assert set(heads) <= {"AttnBlockFirst_v1", "AttnBlock_v4{CHECK=false}", "AttnHeadEnd_v1"}
    assert rows <= WC._GRAIN.rows and free == 0
