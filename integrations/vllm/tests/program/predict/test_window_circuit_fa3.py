"""verity_vllm.program.predict.window_circuit on sm_90: the H200's Program binds FA3's `Attention_v4`, which the
lowering cuts by key block as `Attention_v22` (`CUTS`): a head's units are its first visited block's softmax
`AttnBlockSoftmax_v1` and P·V groups `AttnBlockPv_v1` and its end `AttnHeadEnd_v1` at FA3's dot and reciprocal, and past one
key block (TINY's kBlockN on sm_90 is 128) each full block's softmax `AttnBlockSoftmax_v2{FIRST=False, CHECK=False}` and its
groups. A softmax reads its keys as rows of the RoPE'd keys and a group its values as the value projection's slices, a key a
row, so a unit's rows are bounded by its block's keys. An attention version with no cut (`Attention_v2`, which
`fa3_construction = "guard-every-block"` binds) is a unit a head of its own head Definition, reading the same rows, refused
by name past the grain's rows. With ``VERITY_VLLM_FA3_LLAMA_1B=1``, int8 Llama-3.2-1B on sm_90 at 4,32,4 lowers with every
reader wired and every unit a proof block C-Flock's composer takes. Pinned on `test_window_circuit`'s TINY class, its served
traffic and grain.
"""
from __future__ import annotations

import json
import os
import re
import resource
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import (CLASS, GRAIN, OTHER, PV, SOFTMAX, WINDOW, _blocks, _bytes, _label,
                                                       _labels, _source, held)
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_circuit import Grain, WindowClass, class_circuit, window_circuit

from verity.ml.windows import interface as IF

#: the H200's declared target (`pins/NVIDIA_H200.json`): FA3, so `Attention_v4` at D = 16's kBlockN of 128
H200 = {"compute_capability": [9, 0], "num_sms": 132}
DOT, INV = "DOT=HopperBF16WgmmaDot16_v1", "INV=Fa3InvSum_v1"
LLAMA_1B = Path(__file__).resolve().parent / "fixtures/llama32-1b.config.json"
REFUSED = "templates the window can't hold, each port cut once a writer row: "


def _registered(target: dict = H200, config: dict = S.TINY):
    return S.Registered.of(S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": config}, target=target), default_seed=7)


def _wired_in_slot_and_step(circuit) -> None:
    labels = _labels(circuit)
    assert all((lab >= 0).all() for lab in labels.values())
    for w in circuit.wires:
        i = np.arange(w.units, dtype=np.int64)
        reader, writer = labels[w.reader][w.at + i // w.cut[1]], labels[w.writer][w.source(i)]
        assert (reader[:, 0] == writer[:, 0]).all() and (writer[:, 1] <= reader[:, 1]).all(), w.doc()


def _reads_its_keys_and_values(circuit, head: str, values: str | None = None) -> None:
    """Position 4's heads (T = 5), ``head`` reading the keys and ``values`` (``head`` itself where None) the values: slot 0's
    heads 0 and 1 read KV group 0's keys a row a position, the RoPE'd keys of positions 0..4 (RoPE units 24p + 8), and its
    values the value projection's slice 128p + 6; slot 1's head 0 its own (24p + 10 and 128p + 14). A head's queries are
    its own RoPE unit's."""
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    tid, rope, v = ids[head], ids["RoPEHead_v1{D=16}"], ids["Gemm_v2{K=64,N=16}"]
    vid = ids[values] if values else tid
    assert {**circuit.cuts[tid], **circuit.cuts[vid]} == {"k": 5, "v": 5}
    assert [_source(circuit, tid, "q", u) for u in range(8)] == [(rope, 96 + u, "o0") for u in range(8)]
    for u, k, val in ((0, 8, 6), (1, 8, 6), (2, 9, 7), (4, 10, 14)):
        assert [_source(circuit, tid, "k", u, p) for p in range(5)] == [(rope, 24 * p + k, "o0") for p in range(5)]
        assert [_source(circuit, vid, "v", u, p) for p in range(5)] == [(v, 128 * p + val, "o0") for p in range(5)]


def test_an_fa3_head_is_v22s_calls_reading_a_key_and_a_value_row_a_position():
    """Position p's heads (T = p + 1, one key block) are `Attention_v22`'s Calls, `AttnBlockSoftmax_v1{NVIS=T}` and its one
    P·V group `AttnBlockPv_v1{N=T}` at FA3's dot and `AttnHeadEnd_v1` at FA3's reciprocal, a unit a head each (2 slots,
    2 layers, 4 heads: 16 a T); the softmax reads its keys and the group its values a row a position
    (`_reads_its_keys_and_values`) and its softmax's P, and the end the softmax's running sum and the group's output. Under
    both test traffics (`WINDOW`, `OTHER`) the class is one circuit, blocks included, with the weights and the scan state
    pinned no input row is free, and every wire reads its own slot at its step or before."""
    reg = _registered()
    circuit = window_circuit(reg, S.compile_steps(reg, *S._session(*WINDOW)), CLASS, GRAIN, provable=False)
    other = window_circuit(reg, S.compile_steps(reg, *S._session(*OTHER)), CLASS, GRAIN, provable=False)
    assert _bytes(circuit) == _bytes(other) and _blocks(circuit) == _blocks(other)
    n = Counter()
    for fn, k in circuit.templates:
        if fn.id.startswith(("Attention", "AttnBlock", "AttnHeadEnd")):
            n[fn.id] += k
    soft = f"AttnBlockSoftmax_v1{{{{D=16,NVIS={{}},FIRST=True,{DOT},CHECK=True,PK=128,LAST=True}}}}"
    pv = f"AttnBlockPv_v1{{{{D=16,N={{}},VS=16,{DOT},FIRST=True}}}}"
    assert n == {**{soft.format(t): 16 for t in range(1, 6)}, **{pv.format(t): 16 for t in range(1, 6)},
                 f"AttnHeadEnd_v1{{D=16,{INV}}}": 80}
    assert not [fn.id for fn, _ in circuit.templates if fn.id.startswith("AllGather")]
    soft, pv = SOFTMAX.format(5), PV.format(5)
    _reads_its_keys_and_values(circuit, soft, pv)
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    soft, pv, end = ids[soft], ids[pv], ids["AttnHeadEnd_v1{D=16}"]
    assert [_source(circuit, pv, "p", u) for u in range(8)] == [(soft, u, "o1") for u in range(8)]
    assert [_source(circuit, end, p, 64 + 5) for p in ("l", "o")] == [(soft, 5, "o0"), (pv, 5, "o0")]
    assert IF.input_count(json.loads(json.dumps(circuit.description())), **held(circuit))["free"] == 0
    _wired_in_slot_and_step(circuit)


def test_an_fa3_key_block_is_units_within_the_grains_rows_at_every_key_count():
    """At T 128 and 129 (one Program of 129 positions at TINY of 2,048 positions) a head's first block's softmax reads 127
    or 128 keys' rows and its group as many values', and the full block below key 128 (`AttnBlockSoftmax_v2{CHECK=false}`:
    FA3's one-row decode guards only the first block it visits; LAST, its m not out) 128 of each besides its running state,
    its largest unit 135 rows, where a unit a block held 263."""
    reg = _registered(config={**S.TINY, "max_position_embeddings": 2048})
    circuit = class_circuit(reg, WindowClass(1, 128, 2), 1, GRAIN, provable=False)
    rows = {_label(fn): sum(map(len, IF.unit_rows(fn, circuit.cuts.get(IF.descriptor_id(fn))))) for fn, _ in circuit.templates}
    assert max(rows.items(), key=lambda kv: kv[1]) == (
        "AttnBlockSoftmax_v2{D=16,NVIS=128,FIRST=false,CHECK=false,PK=128,LAST=true}", 135)
    assert (rows[SOFTMAX.format(128)], rows["AttnBlockPv_v1{D=16,N=128,VS=16,FIRST=false}"]) == (131, 131)


#: FA3 declared to guard every block: `Attention_v2`, which has no per-key-block cut
GUARDED = {**H200, "fa3_construction": "guard-every-block"}
HEAD = "AttentionHead_v2{{T={},D=16,BN=128}}"


def test_an_attention_with_no_cut_is_its_heads_each_reading_a_key_and_a_value_row_a_position():
    """`Attention_v2`'s position p (T = p + 1) is `AttentionHead_v2{T}`, a unit a head (16 a T), reading the keys and values
    v20's first block reads (`_reads_its_keys_and_values`). Two windows of the class are one circuit, no input row is free,
    and every wire reads its own slot at its step or before."""
    reg = _registered(GUARDED)
    circuit = window_circuit(reg, S.compile_steps(reg, *S._session(*WINDOW)), CLASS, GRAIN, provable=False)
    other = window_circuit(reg, S.compile_steps(reg, *S._session(*OTHER)), CLASS, GRAIN, provable=False)
    assert _bytes(circuit) == _bytes(other) and _blocks(circuit) == _blocks(other)
    n = {_label(fn): k for fn, k in circuit.templates}
    assert {t: k for t, k in n.items() if t.startswith(("Attention", "AttnBlock", "AttnHeadEnd"))} == {
        HEAD.format(t): 16 for t in range(1, 6)}
    _reads_its_keys_and_values(circuit, HEAD.format(5))
    assert IF.input_count(json.loads(json.dumps(circuit.description())), **held(circuit))["free"] == 0
    _wired_in_slot_and_step(circuit)


def test_a_head_with_no_cut_past_the_grains_rows_is_refused_by_name():
    """A head of a version with no cut is one unit of its keys' and values' rows besides its query and output: at 10 rows a
    unit, position 4's T of 5 (12 rows) is refused by name. The norm of 12 rows is cut at its residual add instead, one part
    of 10 rows and a tail of 4, and a token select of 128 logits a unit is 8 rows a combine."""
    with pytest.raises(Unexpressible, match=re.escape(
            f"1 {REFUSED}AttentionHead_v2{{T=5,D=16,BN=128,{DOT},{INV}}}: 12 rows in and out, past a unit's 10")):
        class_circuit(_registered(GUARDED), CLASS, None, Grain(gemm_words=1024, rows=10, select=128))


@pytest.mark.skipif(not os.environ.get("VERITY_VLLM_FA3_LLAMA_1B"),
                    reason="int8 Llama-3.2-1B on sm_90 at 4,32,4, minutes on a pod; set VERITY_VLLM_FA3_LLAMA_1B=1")
def test_int8_llama_3_2_1b_on_sm_90_lowers_every_unit_to_a_proof_block():
    """int8 Llama-3.2-1B's window circuit on the H200 (exact-int8 linears, FA3's heads at D 64 and kBlockN 192) at the default
    grain and 4,32,4 lowers with every unit within 256 rows: each P·V group reads its values a slice a row of the value
    projection (`AttnBlockPv_v1{VS}`), and the norm after down_proj, whose x down_proj's slices write as 512 rows where
    o_proj's write 128, is cut at its residual add (8 parts of 256 and the tail). Every unit is a proof block C-Flock's
    composer takes (`interface.check_proof_block`, which the lowering runs), each head's softmax steps and P·V groups among
    them: at 4 outputs a collector's rows and an entry's are under 112 bytes, so each is its position's combine's
    (`SelectCollect_v1`, the last's `SelectEntry_v1`), which reads its group units' bests and indices as rows of 334 and 668
    bytes. Its heads are `Attention_v22`'s Calls (no whole `AttentionHead_v4`, no guarded full block). Its numbers (units,
    templates, registered reads, the heads' census, each head family's most rows and rows' k_log, the most rows and k_log of
    a proof block, peak RSS) are printed as a JSON line."""
    config = {**json.loads(LLAMA_1B.read_text()), "quantization_config": dict(INT8_EXACT_QUANT)}
    reg, cls = _registered({**H200, "int8_exact": "linears"}, config), WindowClass(4, 32, 4)
    circuit = class_circuit(reg, cls)
    heads, n, most, proved, rows = Counter(), {_label(fn): k for fn, k in circuit.templates}, {}, {}, {}
    for fn, k in circuit.templates:
        cuts = circuit.cuts.get(IF.descriptor_id(fn))
        proved[_label(fn)], rows[_label(fn)] = IF.check_proof_block(fn, cuts), sum(map(len, IF.unit_rows(fn, cuts)))
        if fn.id.startswith(("Attention", "AttnBlock", "AttnHeadEnd")):
            fam = fn.id.split("{")[0] + ("" if fn.base_id != "AttnBlockSoftmax_v2"
                                         else f"{{CHECK={str(fn.bindings['CHECK']).lower()}}}")
            heads[fam] += k
            most[fam] = max(most.get(fam, (0, 0)), (rows[_label(fn)], proved[_label(fn)]))
    print(json.dumps({"class": [cls.slots, cls.prompt, cls.outputs, cls.lanes, cls.every], "positions": cls.positions,
                      "units": sum(n.values()), "templates": len(n), "reads": len(circuit.registered.reads),
                      "heads": dict(heads), "heads_most_rows_k_log": most, "rows": max(rows.values()),
                      "k_log": max(proved.values()),
                      "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)}, sort_keys=True), flush=True)
    assert max(rows.values()) <= IF.MAX_ROWS and max(proved.values()) <= IF.K_MAX
    assert {lab.split("{")[0] for lab in n} >= {"SelectCollect_v1", "SelectEntry_v1"}
    assert not {lab.split("{")[0] for lab in n} & {"TokenSelectCombine_v2", "OutCollect_v1", "SlotEntry_v1"}
    assert set(heads) <= {"AttnBlockSoftmax_v1", "AttnBlockSoftmax_v2{CHECK=false}", "AttnBlockPv_v1", "AttnHeadEnd_v1"}
    assert {"AttnBlockSoftmax_v1", "AttnBlockPv_v1", "AttnHeadEnd_v1"} <= set(most)
    tail = n["RMSNormFusedCudaTail_v1{N=2048}"]
    assert tail and [n[f"RMSNormFusedCudaAdd_v1{{N=2048,C=256,AT={at}}}"] for at in range(8)] == [tail] * 8
