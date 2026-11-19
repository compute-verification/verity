"""verity_vllm.program.predict.window_circuit on sm_90: the H200's Program binds FA3's `Attention_v4`, which the
lowering cuts by key block as `Attention_v20` (`CUTS`): a head's units are its first visited block `AttnBlockFirst_v1` and its
end `AttnHeadEnd_v1` at FA3's dot and reciprocal, and past one key block (TINY's kBlockN on sm_90 is 128) each full block
`AttnBlock_v4{FIRST=False, CHECK=False}`. A block reads its keys and values as rows of the RoPE'd keys and the value
projection's slices, a key a row, so past 126 positions a Program's first block holds more than a unit's 256 rows and it is
refused by name with its bound (BF16 Llama-3.2-1B's, 5 rows a key, past 50). An
attention version with no cut (`Attention_v2`, which `fa3_construction = "guard-every-block"` binds) is a unit a head of its
own head Definition, reading the same rows, refused by name past the grain's rows. BF16 Llama-3.2-1B at 4,27,2 lowers wired,
untied and cut at Pearl+'s linears. With ``VERITY_VLLM_FA3_LLAMA_1B=1``,
int8 Llama-3.2-1B on sm_90 at 4,32,4 lowers with every reader wired and every unit a proof block Flock's composer takes.
Pinned on `test_window_circuit`'s TINY class, windows and grain.
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
from verity.commitments.periods import interface as IF

from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import (
    CLASS,
    GRAIN,
    FULL,
    FULL_LATER,
    _blocks,
    _bytes,
    _label,
    _labels,
    _source,
    held,
)
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict import window_builder as WB
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_circuit import (
    Grain,
    PearlTables,
    WindowClass,
    check_provable,
    class_circuit,
    window_circuit,
)

#: the H200's declared target (its device pin, `bf16_gemm_check.device_pin`): FA3, so `Attention_v4` at D = 16's kBlockN of 128
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


def _reads_its_keys_and_values(circuit, head: str) -> None:
    """Position 4's heads (T = 5): slot 0's heads 0 and 1 read KV group 0's keys a row a position, the RoPE'd keys of
    positions 0..4 (RoPE units r(p) + 8, r(p) the units before position p: 24 a position, 16 at the prompt's positions 0
    to 2, which have no last-layer queries), and its values the value projection's slice g(p) + 12 (g(p): 128 a position,
    64 at those three, the queries', keys' and values' slices an allocation each); slot 1's head 0 its own (r(p) + 10
    and g(p) + 14). A head's queries are its own RoPE unit's."""
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    tid, rope, v = ids[head], ids["RoPEHead_v1{D=16}"], ids["Gemm_v2{K=64,N=16}"]
    assert circuit.cuts[tid] == {"k": 5, "v": 5}
    r, g = ([a * min(p, 3) + b * max(p - 3, 0) for p in range(5)] for a, b in ((16, 24), (64, 128)))
    assert [_source(circuit, tid, "q", u) for u in range(8)] == [(rope, r[4] + u, "o0") for u in range(8)]
    for u, k, val in ((0, 8, 12), (1, 8, 12), (2, 9, 13), (4, 10, 14)):
        assert [_source(circuit, tid, "k", u, p) for p in range(5)] == [(rope, r[p] + k, "o0") for p in range(5)]
        assert [_source(circuit, tid, "v", u, p) for p in range(5)] == [(v, g[p] + val, "o0") for p in range(5)]


def test_an_fa3_head_is_v20s_calls_reading_a_key_and_a_value_row_a_position():
    """Position p's heads (T = p + 1, one key block) are `Attention_v20`'s Calls, `AttnBlockFirst_v1{NVIS=T}` at FA3's dot and
    `AttnHeadEnd_v1` at FA3's reciprocal, a unit a head each (2 slots, 2 layers, 4 heads: 16 a T, 8 at T 1 to 3, the prompt's
    positions before its last, whose last layer's attention no token reads); the first block reads its
    keys and values a row a position (`_reads_its_keys_and_values`), and the end its block's running sum and output. Two
    windows of the fixed load are one circuit, blocks included, with the weights and the scan state pinned no input row is free,
    and every wire reads its own slot at its step or before."""
    reg = _registered()
    circuit = window_circuit(reg, S.compile_steps(reg, *S._session(*FULL)), CLASS, GRAIN)
    other = window_circuit(reg, S.compile_steps(reg, *S._session(*FULL_LATER)), CLASS, GRAIN)
    assert _bytes(circuit) == _bytes(other) and _blocks(circuit) == _blocks(other)
    n: Counter[str] = Counter()
    for fn, k in circuit.templates:
        if fn.id.startswith(("Attention", "AttnBlock", "AttnHeadEnd")):
            n[fn.id] += k
    assert n == {**{f"AttnBlockFirst_v1{{D=16,NVIS={t},BN=128,{DOT}}}": 16 if t > 3 else 8 for t in range(1, 6)},
                 f"AttnHeadEnd_v1{{D=16,{INV}}}": 56}
    assert not [fn.id for fn, _ in circuit.templates if fn.id.startswith("AllGather")]
    _reads_its_keys_and_values(circuit, "AttnBlockFirst_v1{D=16,NVIS=5,BN=128}")
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    first, end = ids["AttnBlockFirst_v1{D=16,NVIS=5,BN=128}"], ids["AttnHeadEnd_v1{D=16}"]
    assert [_source(circuit, end, p, 8 * 3 + 16 + 5) for p in ("l", "o")] == [(first, 5, "o1"), (first, 5, "o2")]
    assert IF.input_count(json.loads(json.dumps(circuit.description())), **held(circuit))["free"] == 0
    _wired_in_slot_and_step(circuit)


def test_past_126_positions_an_fa3_program_is_refused_by_name_with_its_bound():
    """At T 127 (one Program of 129 positions at TINY of 2,048 positions) a head's first block reads 127 keys' rows and as
    many values', more than a unit's 256 rows with its query and outputs: refused by name at position 126, naming the
    bound, 126 positions."""
    reg = _registered(config={**S.TINY, "max_position_embeddings": 2048})
    with pytest.raises(Unexpressible, match=re.escape(
            "class Program 0: position 126, past the 126 positions a Program holds at this model and grain: its first key "
            "block at 127 keys is 258 rows in and out (4 and 2 a key), past a unit's 256")):
        class_circuit(reg, WindowClass(1, 128, 2), 1, GRAIN)


def test_past_50_positions_a_llama_3_2_1b_program_on_the_h200_is_refused_by_name_with_its_bound():
    """BF16 Llama-3.2-1B on the H200 at the default grain: a key block reads 5 rows a key, its key's and its value's,
    besides its query and its three outputs, so a Program of 50 positions lowers and one of 51 is refused at position
    50, naming the bound. One layer holds it: a layer's key blocks are its own."""
    reg = _registered(config={**json.loads(LLAMA_1B.read_text()), "num_hidden_layers": 1})
    class_circuit(reg, WindowClass(1, 49, 2), 1)
    with pytest.raises(Unexpressible, match=re.escape(
            "class Program 0: position 50, past the 50 positions a Program holds at this model and grain: its first key "
            "block at 51 keys is 259 rows in and out (4 and 5 a key), past a unit's 256")):
        class_circuit(reg, WindowClass(1, 50, 2), 1)


def test_unequal_prompts_at_one_boundary_lower_wired_in_an_order(monkeypatch):
    """BF16 Llama-3.2-1B of two layers on the H200: requests of prompts 8, 5 and 2 at one boundary lower to their Programs
    in an order of their ranges. Where one Program's prompt position is dead past its last layer's attention and the next
    one's isn't, a node allocates for fewer slots, and wires coalesced across those allocations (one `interface.coalesce`
    over the window's) give a range a box past what it reads: there `interface.ordered` finds a cycle."""
    reg = _registered(config={**json.loads(LLAMA_1B.read_text()), "num_hidden_layers": 2})
    cls = WindowClass(3, 8, 2)
    requests = tuple(WC.Request(0, j, p, 2) for j, p in enumerate((8, 5, 2)))
    low = WC.lower(reg, cls, None, requests=requests)
    assert [p.positions for p in low.programs] == [9, 6, 3] and low.circuit.order
    monkeypatch.setattr(WB._Builder, "allocation_wires", lambda self: self.wires)
    with pytest.raises(ValueError, match="cycle: no order of the ranges"):
        WC.lower(reg, cls, None, requests=requests)


#: FA3 declared to guard every block: `Attention_v2`, which has no per-key-block cut
GUARDED = {**H200, "fa3_construction": "guard-every-block"}
HEAD = "AttentionHead_v2{{T={},D=16,BN=128}}"


def test_an_attention_with_no_cut_is_its_heads_each_reading_a_key_and_a_value_row_a_position():
    """`Attention_v2`'s position p (T = p + 1) is `AttentionHead_v2{T}`, a unit a head (16 a T, 8 at the prompt's T 1 to 3),
    reading the keys and values
    v20's first block reads (`_reads_its_keys_and_values`). Two windows of the fixed load are one circuit, no input row is free,
    and every wire reads its own slot at its step or before."""
    reg = _registered(GUARDED)
    circuit = window_circuit(reg, S.compile_steps(reg, *S._session(*FULL)), CLASS, GRAIN)
    other = window_circuit(reg, S.compile_steps(reg, *S._session(*FULL_LATER)), CLASS, GRAIN)
    assert _bytes(circuit) == _bytes(other) and _blocks(circuit) == _blocks(other)
    n = {_label(fn): k for fn, k in circuit.templates}
    assert {t: k for t, k in n.items() if t.startswith(("Attention", "AttnBlock", "AttnHeadEnd"))} == {
        HEAD.format(t): 16 if t > 3 else 8 for t in range(1, 6)}
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


#: Pearl+'s scheme a tied run's class circuit is cut at (`served_verdicts.sh`'s PEARLPLUS_ARM)
PEARL = "pearl-c-h100-v1-h3"
ADD = "RMSNormFusedCudaAdd_v2{{N=2048,C=256,AT={},R={}}}"


@pytest.mark.slow
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_bf16_llama_3_2_1b_on_sm_90_lowers_wired_at_4_27_2(tied):
    """BF16 Llama-3.2-1B's class circuit on the H200 at 4,27,2, from its configuration alone, lowers wired with every unit a
    proof block. Untied, o_proj's `Gemm_v2{K=2048, N=8}` slices write the norm after the attention's x as 256 rows and
    down_proj's `Gemm_v2{K=8192, N=4}` the next norm's as 512, so both are cut at their residual add, their parts reading 32
    and 64 rows, a template each. Cut at Pearl+'s linears, every norm's x is `LinearOut_v3` rows and no norm is cut."""
    reg = _registered(config=json.loads(LLAMA_1B.read_text()))
    circuit = class_circuit(reg, WindowClass(4, 27, 2), tables=PearlTables(PEARL) if tied else None)
    check_provable(circuit)
    n = {_label(fn): k for fn, k in circuit.templates}
    norms = {lab: k for lab, k in n.items() if lab.startswith("RMSNormFused")}
    if tied:
        assert (len(n), norms) == (51, {"RMSNormFusedCuda_v2{N=2048}": 3584})
        return
    assert {lab for lab in n if lab.startswith("Gemm_v2")} >= {"Gemm_v2{K=2048,N=8}", "Gemm_v2{K=8192,N=4}"}
    assert (len(n), norms) == (63, {"RMSNormFusedCudaTail_v1{N=2048}": 3584,
                                    **{ADD.format(at, r): 1792 for at in range(8) for r in (32, 64)}})
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    assert all(circuit.cuts[ids[ADD.format(at, r)]] == {"x": r} for at in range(8) for r in (32, 64))


@pytest.mark.skipif(not os.environ.get("VERITY_VLLM_FA3_LLAMA_1B"),
                    reason="int8 Llama-3.2-1B on sm_90 at 4,32,4, minutes on a pod; set VERITY_VLLM_FA3_LLAMA_1B=1")
def test_int8_llama_3_2_1b_on_sm_90_lowers_every_unit_to_a_proof_block():
    """int8 Llama-3.2-1B's window circuit on the H200 (exact-int8 linears, FA3's heads at D 64 and kBlockN 192) at the default
    grain and 4,32,4 lowers with every unit within 256 rows: each first block reads its values a slice a row of the value
    projection (`AttnBlockFirst_v2{VS=32}`), and the norm after down_proj, whose x down_proj's slices write as 256 rows where
    o_proj's write 64, is cut at its residual add (8 parts of 256, each 32 rows, and the tail). Every unit is a proof block Flock's
    composer takes (`check_provable`): at 4 outputs a collector's rows and an entry's
    are under 112 bytes, so each is its position's combine's (`SelectCollect_v1`, the last's `SelectEntry_v1`), which reads
    its group units' bests and indices as rows of 334 and 668 bytes. Its heads are `Attention_v20`'s Calls (no whole `AttentionHead_v4`, no guarded full
    block). Its numbers (units, templates, registered reads, the heads' census, the most rows and k_log of a proof block,
    peak RSS) are printed as a JSON line."""
    config = {**json.loads(LLAMA_1B.read_text()), "quantization_config": dict(INT8_EXACT_QUANT)}
    reg, cls = _registered({**H200, "int8_exact": "linears"}, config), WindowClass(4, 32, 4)
    circuit = class_circuit(reg, cls)
    check_provable(circuit)
    heads: Counter[str]
    heads, n = Counter(), {_label(fn): k for fn, k in circuit.templates}
    for fn, k in circuit.templates:
        if fn.id.startswith(("Attention", "AttnBlock", "AttnHeadEnd")):
            heads[fn.id.split("{")[0] + ("" if fn.base_id != "AttnBlock_v4" else f"{{CHECK={str(fn.bindings['CHECK']).lower()}}}")] += k
    cuts = [circuit.cuts.get(IF.descriptor_id(fn)) for fn, _ in circuit.templates]
    rows = max(sum(map(len, IF.unit_rows(fn, c))) for (fn, _), c in zip(circuit.templates, cuts))
    k_log = max(IF.check_proof_block(fn, c) for (fn, _), c in zip(circuit.templates, cuts))
    print(json.dumps({"class": [cls.slots, cls.prompt, cls.outputs, cls.lanes, cls.every], "positions": cls.positions,
                      "units": sum(n.values()), "templates": len(n), "reads": len(circuit.registered.reads),
                      "heads": dict(heads), "rows": rows, "k_log": k_log,
                      "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)}, sort_keys=True), flush=True)
    assert rows <= IF.MAX_ROWS and k_log <= IF.K_MAX
    assert {lab.split("{")[0] for lab in n} >= {"SelectCollect_v1", "SelectEntry_v1"}
    assert not {lab.split("{")[0] for lab in n} & {"TokenSelectCombine_v2", "OutCollect_v1", "SlotEntry_v1"}
    assert set(heads) <= {"AttnBlockFirst_v2", "AttnBlock_v4{CHECK=false}", "AttnHeadEnd_v1"}
    tail = n["RMSNormFusedCudaTail_v1{N=2048}"]
    assert tail and [n[f"RMSNormFusedCudaAdd_v2{{N=2048,C=256,AT={at},R=32}}"] for at in range(8)] == [tail] * 8
