"""verity_vllm.program.predict.window_circuit (at its fixed slots): a served window is its class's one Circuit of the
catalog's units, a class Program a slot whatever the traffic, every input row wired from its writer's row or a registered read
of the weights or the network certifier's scan state rows, and nothing free; a slot's outputs leave as its entry of the scan's
Programs' row. Pinned on a TINY class of 2 slots, 4 prompt ids and 2 outputs (5 positions a Program), at a grain small
enough to slice TINY's GEMMs and cut its ports; the class's circuit, its blocks included, is the same for windows that differ
in everything, their request count included. At the class of record's slots (8, 4 at each of buckets 0 and 16) with TINY's
caps, a 3-request and an 8-request window are one circuit. Every reader reads its writers' rows as they are, a port cut
once a writer row. A head is `AttentionHead_v19`'s Calls, a unit each: a key block reads a key's row and a value's a key,
so past 126 keys it holds more than a unit's 256 rows, and TINY at 129 positions is refused by name.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.check.test_weights_of_record import _bf16, write_safetensors
from tests.program.predict import test_steps as S
from verity_vllm.check import weights_of_record as WR
from verity_vllm.pipeline import advice as ADV
from verity_vllm.program import model as M
from verity_vllm.program.predict import scan_state as SS
from verity_vllm.program.predict import window_builder as WB
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_circuit import CLASS_OF_RECORD, Grain, WindowClass, class_circuit, window_circuit
from verity_vllm.records import weights_registration as WRG

from verity.core.protocols.certifier import ingress_scan as IS
from verity.core.protocols.pouw import audit as AU
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.definitions.registry import io_shell as IO
from verity.ml.windows import interface as IF
from verity.ml.windows import wiring as WI

GRAIN = Grain(gemm_words=1024, rows=256, select=64)
CLASS = WindowClass(2, 4, 2)
#: the class of record's slots (8, 4 a boundary at buckets 0 and 16) at TINY's caps
SLOTS = WindowClass(8, 4, 2, lanes=4, every=16)
G2 = {**S.GREEDY, "max_tokens": 2}
#: q0's prompt of 4 at step 0, q1's prompt of 3 chunked 2 then 1 across steps 0 and 1; each keeps two outputs
WINDOW = ([(4, 2, G2), (3, 2, G2)], [[(0, 0, 4), (1, 0, 2)], [(0, 4, 1), (1, 2, 1)], [(1, 3, 1)]])
#: another window of the class: q0's prompt of 3 at step 0 keeps two outputs, q1's prompt of 2 arrives at step 1 and keeps one
OTHER = ([(3, 2, G2), (2, 1, {**S.GREEDY, "max_tokens": 1})], [[(0, 0, 3)], [(0, 3, 1), (1, 0, 2)]])
ONE = ([(4, 2, G2)], [[(0, 0, 4)], [(0, 4, 1)]])
#: eight requests of prompts 1..4, every one prefilled at step 0 and decoded once at step 1
EIGHT = ([(1 + i % 4, 2, G2) for i in range(8)], [[(i, 0, 1 + i % 4) for i in range(8)], [(i, 1 + i % 4, 1) for i in range(8)]])
#: three requests: two at step 0, the third (one output) arriving at step 1
THREE = ([(4, 2, G2), (2, 2, G2), (3, 1, {**S.GREEDY, "max_tokens": 1})], [[(0, 0, 4), (1, 0, 2)], [(0, 4, 1), (1, 2, 1), (2, 0, 3)]])
EMB, GROUPS = "EmbeddingRead_v1{V=256,H=64}", "TokenSelectGroups_v1{C=64,G=16}"
FEED = "TokenFeed_v1{NP=4,FIRST=false}"
#: the scan state read at TINY's 2 lanes (4,548 words a row), each position's select with its collector (the first
#: position's, every later one's) and the last's with the slot's entry
SCAN, ENTRY = "ScanSlots_v2{NS=4548,NP=4,IDB=17,K0=12,K=45}", "SelectEntry_v1{V=256,C=16,KO=2,FIRST=false,IDB=17,NP=4}"
COLLECT0, COLLECT = "SelectCollect_v1{V=256,C=16,KO=2,FIRST=true}", "SelectCollect_v1{V=256,C=16,KO=2,FIRST=false}"


def held(circuit) -> dict[str, dict[str, str]]:
    """`interface.input_count`'s pins and the window's roots holding every registered parameter of ``circuit``: each
    weight's own, ``weights/<label>@<words>``, and the scan state's, `scan_state.SCAN_STATE`, pinned by the verifier at
    the root the window's registration records for it. Each pin here is the window's own root, the scan state's included,
    which is why a count under it reports zero free inputs: the verifier pins the scan state at the network certifier's
    scan rows' root (`SCAN_PIN`), and each slot's occupancy, prompt length and token ids are free unless the window reads
    that state."""
    roots = {p: "ab" * 64 for p in circuit.registered.params}
    return {"pins": roots, "roots": roots}


def _label(fn) -> str:
    return fn.id.split("{")[0] + "{" + ",".join(f"{k}={json.dumps(v)}" for k, v in fn.bindings.items() if type(v) in (int, bool)) + "}"


def _circuit(window, grain: Grain = GRAIN, cls: WindowClass = CLASS, placed=None):
    reg = S._registered()
    return window_circuit(reg, S.compile_steps(reg, *S._session(*window)), cls, grain, placed=placed, provable=False)


def _bytes(circuit) -> bytes:
    return IF.circuit_bytes(circuit.description())


def _blocks(circuit) -> bytes:
    """The circuit's blocks as its record carries them, less the proof block's partition digest: that is its population
    program's, which its templates fix (`Circuit.blocks_record`, pinned in the windows suite), and building it here would
    run the embedding's population program."""
    return IF.circuit_bytes(circuit.blocks.doc(sum(circuit.units.values()), ""))


@pytest.fixture(scope="module")
def window():
    """The window's Circuit and its templates' ids by label."""
    circuit = _circuit(WINDOW)
    return circuit, {_label(fn): tid for fn, _ in circuit.templates for tid in [IF.descriptor_id(fn)]}


def _source(circuit, tid: str, port: str, unit: int, row: int = 0) -> tuple[str, int, str]:
    """The writer row (template, unit, output) that ``unit``'s ``port`` row ``row`` is wired from."""
    (w,) = [w for w in circuit.wires if (w.reader, w.port) == (tid, port) and w.at <= unit < w.at + w.reader_units
            and w.cut[0] <= row < sum(w.cut)]
    return w.writer, int(w.writers(unit, 1, row)[0]), w.out


def test_windows_of_a_class_are_one_circuit_whatever_their_request_count(window):
    """The privacy property (only the slot count is public): WINDOW and OTHER differ in their requests' arrival
    steps, prompt lengths (4 and 3 against 3 and 2) and output lengths, and a window of one request in its request count too,
    and their circuits' descriptions (units, wires, cuts, registered reads), blocks, templates and external ports are
    byte for byte the class's circuit: the blocks' slots are the class's 2 whatever the window's requests."""
    circuit, _ = window
    cls = class_circuit(S._registered(), CLASS, None, GRAIN, provable=False)
    for other in (_circuit(OTHER), _circuit(ONE), cls):
        assert _bytes(other) == _bytes(circuit) and _blocks(other) == _blocks(circuit)
        assert (other.ids, other.external) == (circuit.ids, circuit.external)
        assert (other.blocks.slots, other.blocks.steps) == (2, 5)


def test_a_3_request_and_an_8_request_window_at_the_class_of_records_slots_are_one_circuit():
    """At the class of record's slots, 4 a boundary at buckets 0 and 16 (at TINY's caps of 4 prompt ids and 2 outputs), a
    window of three requests (two anchored at bucket 0, one at 16) and one of eight (four at each) emit the
    same circuit, the same descriptor digest (SHA-512 of its description's bytes) and the same blocks: eight class Programs,
    each slot's `ScanSlots_v2` unit reading its lane of the scan's state row before its anchor (rows 0 and 16 of the 4-lane
    scan's 9,090 words) and its last position making its entry of the scan's Programs' row."""
    eight = _circuit(EIGHT, cls=SLOTS, placed=[(0, 1)] * 4 + [(16, 17)] * 4)
    three = _circuit(THREE, cls=SLOTS, placed=[(0, 1), (16, 18), (16, 31)])
    assert hashlib.sha512(_bytes(three)).hexdigest() == hashlib.sha512(_bytes(eight)).hexdigest()
    assert (three.ids, three.external, _blocks(three)) == (eight.ids, eight.external, _blocks(eight))
    assert (eight.blocks.slots, eight.blocks.steps) == (8, 5)
    n = {_label(fn): k for fn, k in eight.templates}
    scan = SCAN.replace("NS=4548", "NS=9090")
    assert (n[scan], n[FEED], n[EMB], n[ENTRY]) == (8, 8 * 3, 8 * SLOTS.positions, 8)
    assert [(r.at, r.radix, r.base, r.step) for r in eight.registered.reads if r.param == SS.SCAN_STATE] == [
        (0, (4, 2), 0, (0, 16 * 9090))]
    assert eight.registered.params[SS.SCAN_STATE] == 32 * 9090 and not eight.ingress


def test_the_class_of_record_takes_both_windows_to_its_one_class_circuit(monkeypatch):
    """The class of record (`CLASS_OF_RECORD`: EVERY 16, 4 slots a boundary, 8 a 32-bucket window) is 8 slots of 1,024
    prompt ids and 64 outputs, lane-major: slot 2j + b is boundary b's lane j at bucket 16b. A 3-request and an 8-request
    window are each taken to the one class circuit, every slot's Program, with the same arguments (its lowering, at heads of
    up to 1,087 keys, is refused past a key block's 126: `test_past_126_keys_a_key_block_is_refused_by_name`)."""
    assert (CLASS_OF_RECORD.slots, CLASS_OF_RECORD.lanes, CLASS_OF_RECORD.every, CLASS_OF_RECORD.positions) == (8, 4, 16, 1087)
    assert (CLASS_OF_RECORD.boundaries, CLASS_OF_RECORD.buckets) == ((0, 16), 32)
    assert [(CLASS_OF_RECORD.anchor(k), CLASS_OF_RECORD.lane(k)) for k in range(8)] == [(16 * (k % 2), k // 2) for k in range(8)]
    calls = []
    monkeypatch.setattr(WC, "class_circuit", lambda *a: calls.append(a[1:]) or "the class circuit")
    reg = S._registered()
    for window, placed in ((THREE, [(0, 1), (0, 2), (16, 31)]), (EIGHT, [(0, 1)] * 4 + [(16, 17)] * 4)):
        assert window_circuit(reg, S.compile_steps(reg, *S._session(*window)), CLASS_OF_RECORD, placed=placed) == "the class circuit"
    assert calls == [(CLASS_OF_RECORD, None, WC._GRAIN, WC.NAME, True)] * 2


def test_a_request_anchored_at_16_with_64_outputs_is_expressible(monkeypatch):
    """A slot's Program hands all its outputs to the scan at its anchor bucket and the scan carries them across
    windows (`SC_GROUP`), so a request at the class of record's second boundary (bucket 16) keeping all 64 outputs, its last
    at bucket 16 + D + 64 / r = 59 past the 32-bucket window's end, is taken to the one class circuit."""
    calls = []
    monkeypatch.setattr(WC, "class_circuit", lambda *a: calls.append(a[1:]) or "the class circuit")
    reg = S._registered()
    window = ([(1, 64, {**S.GREEDY, "max_tokens": 64})], [[(0, 0, 1)]] + [[(0, p, 1)] for p in range(1, 64)])
    compiled = S.compile_steps(reg, *S._session(*window))
    assert [(u.LP, u.T + 1) for u in compiled.requests if u is not None] == [(1, 64)]
    assert window_circuit(reg, compiled, CLASS_OF_RECORD, placed=[(16, 59)]) == "the class circuit"
    assert calls == [(CLASS_OF_RECORD, None, WC._GRAIN, WC.NAME, True)]


def test_the_description_grows_with_the_class_programs_nodes_not_its_slots():
    """Each allocation's slots sit side by side in its template, so every slot's copy of a wire or read pattern is one: the
    class of record's slots at 2 and at 8 Programs have the same 118 wires besides the scan's, 88 registered reads (87
    of the weights, one of the scan state), root reads, openings and 59 inference entries, the 8 Programs 4 times the units.
    The scan's wires are a lane's outputs (lane-major, so a lane's slots follow on), each output named for its lane: its
    token, slot, head and the entry's occupancy word (4 at 2 Programs, both lane 0's, and 4 a lane at 8), whatever the
    boundaries."""
    reg = S._registered()
    two, eight = (class_circuit(reg, SLOTS, n, GRAIN, provable=False) for n in (2, None))
    shape = [(len(c.wires), len(c.registered.reads), len(c.registered.roots), len(c.registered.openings),
              len(c.blocks.inference)) for c in (two, eight)]
    assert shape == [(122, 88, 2, 1, 59), (134, 88, 2, 1, 59)] and two.ids == eight.ids
    scan = IF.descriptor_id(next(fn for fn, _ in eight.templates if fn.name == "ScanSlots"))
    assert [sum(w.writer == scan for w in c.wires) for c in (two, eight)] == [4, 4 * SLOTS.lanes]
    assert sum(n for _, n in eight.templates) == 4 * sum(n for _, n in two.templates) == 5552
    assert [[(r.at, r.radix, r.base, r.step) for r in c.registered.reads if r.param == SS.SCAN_STATE] for c in (two, eight)] == [
        [(0, (2,), 0, (16 * 9090,))], [(0, (4, 2), 0, (0, 16 * 9090))]]


def test_the_circuit_is_a_class_program_a_slot_in_units_by_template_and_wire_patterns(window):
    """Each slot is the class Program, its 5 positions each the whole stack, the LM head and the token select: TINY's GEMMs
    sliced 16 columns a unit (1,024 weight words), o_proj's 8 (its own template, whose ``x`` is its 4 heads' rows) and
    down_proj's 8 at K 128, its RoPE a unit a head, its attention a unit a head a Call of `AttentionHead_v19`'s body
    (position p's first key block of p + 1 keys, `AttnBlockFirst_v1{NVIS=p + 1}`, and its end), its token select four
    units of four groups each and a combine, its first position's token, slot, occupancy word and head its lane's of the scan state row
    (`ScanSlots_v2`), positions 1..3's tokens `TokenFeed_v1` units, each position's combine its select collected
    (`SelectCollect_v1`), and its last position's its entry too (`SelectEntry_v1`). A port read as m writer rows is cut m ways, each
    row its writer's as it is (a key block's keys and values a row a position, the fused norm's ``x`` a row a slice), and
    the wiring is a pattern per run, the two slots' one, not a wire per unit, but for the scan's outputs, named for their
    lane. The external rows are the scan state rows the slots read and their entries."""
    circuit, ids = window
    label = {tid: name for name, tid in ids.items()}
    attn = "AttnBlockFirst_v1{{D=16,NVIS={},BN=128}}"
    assert {label[IF.descriptor_id(fn)]: n for fn, n in circuit.templates} == {
        EMB: 10, SCAN: 2, FEED: 6, COLLECT0: 2, COLLECT: 6, ENTRY: 2, "RMSNormTriton_v5{N=64}": 10,
        "Gemm_v2{K=64,N=16}": 640, "Gemm_v2{K=64,N=8}": 160, "Gemm_v2{K=128,N=8}": 160, "Gemm_v2{K=64,N=1}": 10,
        "RoPEHead_v1{D=16}": 120, **{attn.format(t): 16 for t in range(1, 6)}, "AttnHeadEnd_v1{D=16}": 80,
        "RMSNormFusedCuda_v2{N=64}": 40, "SiluMul_v2{I=128}": 20, GROUPS: 40}
    assert {label[t]: c for t, c in circuit.cuts.items()} == {
        **{attn.format(t): {"k": t, "v": t} for t in range(2, 6)}, "Gemm_v2{K=64,N=8}": {"x": 4},
        "RMSNormFusedCuda_v2{N=64}": {"x": 8}, "SiluMul_v2{I=128}": {"gu": 16}, GROUPS: {"x": 4},
        **{s: {"best": 4, "idx": 4} for s in (COLLECT0, COLLECT, ENTRY)}}
    family = {tid: name.split("{")[0] for name, tid in ids.items()}
    assert Counter((family[w.reader], w.port, family[w.writer], w.out) for w in circuit.wires) == {
        ("AttnBlockFirst_v1", "k", "RoPEHead_v1", "o0"): 5, ("AttnBlockFirst_v1", "q", "RoPEHead_v1", "o0"): 5,
        ("AttnBlockFirst_v1", "v", "Gemm_v2", "o0"): 5, ("AttnHeadEnd_v1", "l", "AttnBlockFirst_v1", "o1"): 5,
        ("AttnHeadEnd_v1", "o", "AttnBlockFirst_v1", "o2"): 5, ("EmbeddingRead_v1", "tok", "ScanSlots_v2", "o0"): 1,
        ("EmbeddingRead_v1", "tok", "ScanSlots_v2", "o4"): 1, ("EmbeddingRead_v1", "tok", "TokenFeed_v1", "o0"): 1,
        ("EmbeddingRead_v1", "tok", "SelectCollect_v1", "o0"): 1, ("Gemm_v2", "x", "AttnHeadEnd_v1", "o0"): 1,
        ("Gemm_v2", "x", "RMSNormFusedCuda_v2", "o0"): 16, ("Gemm_v2", "x", "RMSNormTriton_v5", "o0"): 5,
        ("Gemm_v2", "x", "SiluMul_v2", "o0"): 1, ("RMSNormFusedCuda_v2", "res", "EmbeddingRead_v1", "o0"): 5,
        ("RMSNormFusedCuda_v2", "res", "RMSNormFusedCuda_v2", "o1"): 5, ("RMSNormFusedCuda_v2", "x", "Gemm_v2", "o0"): 20,
        ("RMSNormTriton_v5", "x", "EmbeddingRead_v1", "o0"): 1, ("RoPEHead_v1", "x", "Gemm_v2", "o0"): 20,
        ("SelectCollect_v1", "best", "TokenSelectGroups_v1", "o0"): 2, ("SelectCollect_v1", "head", "ScanSlots_v2", "o3"): 1,
        ("SelectCollect_v1", "head", "ScanSlots_v2", "o7"): 1, ("SelectCollect_v1", "idx", "TokenSelectGroups_v1", "o1"): 2,
        ("SelectCollect_v1", "state", "SelectCollect_v1", "o1"): 2, ("SelectCollect_v1", "x0", "Gemm_v2", "o0"): 2,
        ("SelectEntry_v1", "best", "TokenSelectGroups_v1", "o0"): 1, ("SelectEntry_v1", "idx", "TokenSelectGroups_v1", "o1"): 1,
        ("SelectEntry_v1", "occ", "ScanSlots_v2", "o2"): 1, ("SelectEntry_v1", "occ", "ScanSlots_v2", "o6"): 1,
        ("SelectEntry_v1", "state", "SelectCollect_v1", "o1"): 1, ("SelectEntry_v1", "x0", "Gemm_v2", "o0"): 1,
        ("SiluMul_v2", "gu", "Gemm_v2", "o0"): 1, ("TokenFeed_v1", "sel", "SelectCollect_v1", "o0"): 2,
        ("TokenFeed_v1", "slot", "ScanSlots_v2", "o1"): 1, ("TokenFeed_v1", "slot", "ScanSlots_v2", "o5"): 1,
        ("TokenFeed_v1", "slot", "TokenFeed_v1", "o1"): 1, ("TokenSelectGroups_v1", "x", "Gemm_v2", "o0"): 1}
    assert circuit.external == (f"{ids[SCAN]}/state", f"{ids[ENTRY]}/o0")


def test_a_positions_token_is_its_feed_below_the_prompt_cap_and_the_previous_select_past_it(window):
    """Each slot's request is its lane's of the scan state row before its anchor (both slots at bucket 0, lanes 0 and 1, row 0
    of the registered scan state), read by its `ScanSlots_v2` unit, whose lane k is its outputs ``o4k`` to ``o4k+3``: the
    first position's token and slot (its `TokenFeed_v1`'s), the occupancy word and the head. Position 0's embedding reads
    its lane's token and position 1's feed its lane's slot; each later feed reads its predecessor's slot and the previous
    position's select, and position p's embedding reads its feed's token for p < 4 and position p - 1's select at p = 4. A
    template's units are a position's slots side by side (slot k's position p at 2p + k)."""
    circuit, ids = window
    feed, emb, first, rest, scan = ids[FEED], ids[EMB], ids[COLLECT0], ids[COLLECT], ids[SCAN]
    assert not circuit.ingress and [r.doc() for r in circuit.registered.reads if r.param == SS.SCAN_STATE] == [
        {"reader": scan, "port": "state", "param": SS.SCAN_STATE, "at": 0, "radix": [2], "base": 0, "step": [0]}]
    assert [_source(circuit, feed, "slot", u) for u in range(6)] == [(scan, 0, "o1"), (scan, 1, "o5")] + [
        (feed, u, "o1") for u in range(4)]
    assert [_source(circuit, feed, "sel", u) for u in range(6)] == [(first, 0, "o0"), (first, 1, "o0")] + [
        (rest, u, "o0") for u in range(4)]
    assert [_source(circuit, emb, "tok", u) for u in range(10)] == [(scan, 0, "o0"), (scan, 1, "o4")] + [
        (feed, u, "o0") for u in range(6)] + [(rest, 4, "o0"), (rest, 5, "o0")]


def test_a_slots_outputs_are_its_selects_collected_into_its_entry(window):
    """Position p of slot k's select is collected in the unit that makes it (`SelectCollect_v1`, its state ``o1``: the
    first position's from its lane's scan head ``o4k+3``, every later one's from its predecessor's), and its last
    position's `SelectEntry_v1` collects its own select on from the last collector and makes the entry, reading its lane's
    occupancy word (the scan unit's ``o4k+2``), so an empty slot's entry is zero and no select is masked apart from it.
    Each reads its position's four group units' bests and indices, a row each. The entries, not the selects, are the
    circuit's external rows."""
    circuit, ids = window
    scan, groups = ids[SCAN], ids[GROUPS]
    first, rest, entry = ids[COLLECT0], ids[COLLECT], ids[ENTRY]
    assert [_source(circuit, first, "head", u) for u in (0, 1)] == [(scan, 0, "o3"), (scan, 1, "o7")]
    assert [_source(circuit, rest, "state", u) for u in range(6)] == [(first, 0, "o1"), (first, 1, "o1")] + [
        (rest, u, "o1") for u in range(4)]
    assert [_source(circuit, entry, "occ", u) for u in (0, 1)] == [(scan, 0, "o2"), (scan, 1, "o6")]
    assert [_source(circuit, entry, "state", u) for u in (0, 1)] == [(rest, 4, "o1"), (rest, 5, "o1")]
    assert [_source(circuit, first, "idx", 1, r) for r in range(4)] == [(groups, 2 * r + 1, "o1") for r in range(4)]
    assert [_source(circuit, entry, "best", 1, r) for r in range(4)] == [(groups, 33 + 2 * r, "o0") for r in range(4)]
    assert [w for _, w in IF.layout(circuit.template(entry)).outs] == [5]
    assert all(f"{t}/o0" not in circuit.external for t in (first, rest)) and f"{entry}/o0" in circuit.external


def test_a_slots_entry_stops_at_the_registered_models_stop_ids(window):
    """The entry's ``STOP`` is `stop_ids`, the registered model's config's ``eos_token_id`` (an id, a list of them or none):
    TINY declares none, Llama-3.2-1B one (128001), and a list is its set in order."""
    circuit, ids = window
    assert circuit.template(ids[ENTRY]).bindings["STOP"] == WC.stop_ids(S._registered()) == ()
    assert WC.stop_ids(S._registered({**S.TINY, "eos_token_id": 128001})) == (128001,)
    assert WC.stop_ids(S._registered({**S.TINY, "eos_token_id": [9, 3, 9]})) == (3, 9)


def state_row(scan, fields) -> list[int]:
    """A scan state row of ``scan``'s class (`ingress_scan.layout`'s u16 words): lane j's occupancy word, prompt length,
    ``max_tokens`` and ids ``fields[j]``, every other field zero."""
    lay, x = IS.layout(scan), 0
    for j, (occ, lp, m, ids) in enumerate(fields):
        x |= occ << lay[f"{j}.occ"][0] | lp << lay[f"{j}.fields"][0] | m << lay[f"{j}.max_tokens"][0]
        x |= sum(v << lay[f"{j}.ids"][0] + scan.IDB * IO.position(scan.K0, scan.K, p) for p, v in enumerate(ids))
    return [x >> 16 * k & 0xFFFF for k in range(SS.words(scan))]


def test_the_scan_state_row_is_ingress_scans_and_its_wide_read_is_one_row_a_slot():
    """The scan state row a slot reads is `ingress_scan.layout`'s, its head then each lane's fields, ``occ`` (the slot's
    occupancy word, 1 when its Program starts at the row's boundary) among them: at the class of record (4 lanes, KO 64,
    the scan's cells of record) 9,353 u16 words, lane j's fields 37,387 bits apart. `ScanSlots_v2` reads it whole, one
    registered row a slot, and gives every lane's first token (2 words), slot (2,050 at 1,024 prompt ids), occupancy word
    (2) and head (4), of which a slot's lane's four are wired: its unit is 1 + 4·4 rows, 12 of them unread. A row packed by
    the layout (`state_row`) reads back each lane's fields, ids at `io_shell.position`, and its first position's token and
    slot (`io_shell.scan_reference`): an empty lane's token 0 and prompt length -1 past it."""
    scan = CLASS_OF_RECORD.scan
    lay = IS.layout(scan)
    assert (SS.words(scan), scan.K0, scan.K, scan.IDB, scan.KO) == (9353, 12, 45, 17, 64)
    assert [k.split(".", 1)[1] for k in lay if k.startswith("0.")] == [f for f, _ in IS.slot_layout(scan)]
    assert [lay[f"{j}.occ"][0] - lay["0.occ"][0] for j in range(4)] == [37387 * j for j in range(4)]
    bridge = SS.bridge(scan, CLASS_OF_RECORD.prompt)
    lay_ = IF.layout(bridge)
    assert lay_.rows == (("state", 9353),) and [w for _, w in lay_.outs] == [2, 2050, 2, 4] * 4
    assert sum(map(len, IF.unit_rows(bridge))) == 1 + 4 * 4
    small = SLOTS.scan
    fields = [(1, 3, 2, [7, 0x1FFFF, 0, 9]), (0, 0, 0, [0] * 4), (1, 1, 2, [5, 0, 0, 0]), (0, 0, 0, [0] * 4)]
    row = state_row(small, fields)
    assert len(row) == SS.words(small)
    b = SS.bridge(small, SLOTS.prompt).bindings
    got = IO.slots_reference(row, 4, 17, small.K0, small.K, b["OCC"], b["LP"], b["MAX"], b["IDS"])
    assert got == [([occ, lp, *ids], [lp, m]) for occ, lp, m, ids in fields]
    empty = (0, [0xFFFFFFFF, 0, 0, 0, 0], 0, [0, 0])
    assert IO.scan_reference(row, 4, 17, small.K0, small.K, b["OCC"], b["LP"], b["MAX"], b["IDS"]) == [
        (7, [2, 0x1FFFF, 0, 9, 0], 1, [3, 2]), empty, (5, [0, 0, 0, 0, 0], 1, [1, 2]), empty]


def test_the_weights_and_the_embedding_are_registered_reads(window):
    """Each weight a port reads is its own registered parameter at the port's row width (`WB.weight_param`), and the scan
    state is one more (the window's 16 buckets' rows of 4,548 words): GEMM slices, norms and RoPE read their rows from
    their weight's at public positions, and the embedding is `EmbeddingRead_v1`'s opening of the table over its V rows of
    H, keyed by ``tok`` and pinned by its verdict, the table a parameter of its own (`WB.table_param`, its rows ``row-seg``
    leaves). With TINY's tied embeddings the LM head's slices read the embedding at 1,024 words a row, and the token
    select's ``x0`` unit its first row at 64, the table's width but a ``row/v2`` read; o_proj's slices, 8 columns, read
    its weight at 512. Both slots' reads of a node are one pattern (RoPE's a slot digit of step 0)."""
    circuit, ids = window
    reg = circuit.registered
    emb, cs = "model.embed_tokens.weight", "model.layers.0.self_attn.rotary_emb.cos_sin_cache"
    assert {p: n for p, n in reg.params.items() if "layers.1" not in p} == {
        WB.weight_param(emb, 1024): 16384, WB.weight_param(emb, 64): 16384, WB.table_param(emb, 64): 16384,
        WB.weight_param(cs, 16): 1024, WB.weight_param("model.norm.weight", 64): 64, SS.SCAN_STATE: 16 * 4548,
        **{WB.weight_param(f"model.layers.0.{w}.weight", n): k for w, n, k in (
            ("input_layernorm", 64, 64), ("post_attention_layernorm", 64, 64), ("self_attn.qkv_proj", 1024, 8192),
            ("self_attn.o_proj", 512, 4096), ("mlp.gate_up_proj", 1024, 16384), ("mlp.down_proj", 1024, 8192))}}
    assert len(reg.params) == 18
    assert Counter((r.reader, r.port) for r in reg.reads) == {  # a layer's reads its own, a pattern a position
        (ids["Gemm_v2{K=64,N=16}"], "w"): 25, (ids["Gemm_v2{K=64,N=8}"], "w"): 10, (ids["Gemm_v2{K=128,N=8}"], "w"): 10,
        (ids["Gemm_v2{K=64,N=1}"], "w"): 1, (ids["RMSNormFusedCuda_v2{N=64}"], "w"): 20, (ids["RMSNormTriton_v5{N=64}"], "w"): 1,
        (ids["RoPEHead_v1{D=16}"], "cs"): 20, (ids[SCAN], "state"): 1}
    table = f"table/{WB.weight_param(emb, 64)}+0"
    assert [t.doc() for t in reg.tables] == [{"name": table, "param": WB.table_param(emb, 64), "offset": 0, "words": 64,
                                              "rows": 256}]
    assert sorted(r.doc().items() for r in reg.roots) == sorted(
        {"reader": ids[EMB], "port": p, "table": table, "of": p, "at": 0, "units": 10}.items() for p in ("domain", "root"))
    assert [o.doc() for o in reg.openings] == [{"reader": ids[EMB], "table": table, "key": "tok",
                                                "ports": ["row", "salt", "path"], "verdict": "o1", "at": 0, "units": 10}]
    assert [(r.param, r.at, r.radix, r.base, r.step) for r in reg.reads if r.reader == ids["Gemm_v2{K=64,N=1}"]] == [
        (WB.weight_param(emb, 64), 0, (10,), 0, (0,))]
    rope = [(r.param, r.at, r.radix, r.base, r.step) for r in reg.reads if r.reader == ids["RoPEHead_v1{D=16}"]]
    assert rope[:4] == [(WB.weight_param(cs, 16), 0, (2, 4), 0, (0, 0)), (WB.weight_param(cs, 16), 8, (2, 2), 0, (0, 0)),
                        (WB.weight_param(cs, 16), 12, (2, 4), 0, (0, 0)), (WB.weight_param(cs, 16), 20, (2, 2), 0, (0, 0))]


def _whole_rows(circuit: IF.Circuit) -> None:
    """Every registered read of ``circuit`` reads whole rows of its parameter, a weight's named for the port's width (the
    scan state's rows are its NS words)."""
    reg = circuit.registered
    for r in reg.reads:
        lay = IF.layout(circuit.template(r.reader))
        words = dict(lay.rows + lay.outs)[r.port]
        assert (r.param == SS.SCAN_STATE or r.param.endswith(f"@{words}")) and reg.params[r.param] % words == 0, r.doc()
        assert r.base % words == 0 and all(s % words == 0 for s in r.step), r.doc()


def test_every_weight_read_is_whole_rows_of_its_own_weight_at_its_ports_width(window):
    """`interface._reads_of` takes a read's rows as whole rows of its parameter, its base and steps multiples of the port's
    words, which one parameter for the whole weights record (a norm's 64 words between two GEMMs' 512-word rows) breaks. A
    read whose weight isn't whole rows of its port is refused."""
    circuit, ids = window
    _whole_rows(circuit)
    b = WB._Builder(GRAIN)
    b.weights = (("a", 0, 96), ("b", 96, 64))
    with pytest.raises(Unexpressible, match="whole rows of the port"):
        b.weight(0, 64, 64)
    with pytest.raises(Unexpressible, match="spans two weights"):
        b.weight(64, 160, 32)
    assert b.weight(96, 160, 64) == (WB.weight_param("b", 64), 0) and b.params == {WB.weight_param("b", 64): 64}


@pytest.mark.slow
def test_the_window_forms_its_members_each_read_port_a_table_of_its_weights_rows(window):
    """`interface.members` forms the window's members (as the record's verifier does, `window_of_record`): every read covers
    whole members and reads whole rows, so a GEMM slice's ``w`` is a shared-row table of its weight's rows. Slow (about 70 s
    on one core) for the population Program `members` builds, `EmbeddingRead_v1`'s Merkle path traced once; the cheap
    check of the same rows is `test_every_weight_read_is_whole_rows_of_its_own_weight_at_its_ports_width`."""
    circuit, ids = window
    ms = IF.members(circuit, 64)
    assert sum(m.n for m in ms) == sum(n for _, n in circuit.templates) and len(ms) == 113
    gemm = [m for m in ms if m.template == ids["Gemm_v2{K=64,N=8}"]]
    assert gemm and all(dict(m.tables)["w"] <= m.n for m in gemm)


@pytest.mark.slow
def test_core_registers_every_unit_of_the_window(window):
    """Core's registration of the window (`partition.evaluate` of its population program and template query, the record's
    verifier's) accepts every unit, its ranges the circuit's: every template has a gate, so no gather or row copy stands
    in for its reader's ports, and an instance commits within the widest public class. Slow (about a minute) for the
    population program, `EmbeddingRead_v1`'s Merkle path traced once; no cheaper test runs core's own refusals."""
    circuit, _ = window
    part = P.evaluate(*circuit.program())
    assert {t: n for _k, t, _lo, n in part.ranges} == {IF.descriptor_id(fn): n for fn, n in circuit.templates}


def test_a_head_is_its_key_blocks_calls_reading_a_key_and_a_value_row_a_position(window):
    """Position 4 (T 5) at layer 0: a head is `AttentionHead_v19`'s Calls, its one key block `AttnBlockFirst_v1{NVIS=5}` and
    its end `AttnHeadEnd_v1`, which reads the block's running sum and output. The block reads its KV group's keys and values
    as they are, a row a position (``k`` and ``v`` cut 5): slot 0's heads 0 and 1 share KV group 0, whose keys are RoPE units
    24p + 8 of positions 0..4 (a position's 24 are two layers of each slot's 4 query and 2 key heads, side by side) and whose
    values each position's qkv slice 6 (GEMM unit 128p + 6: a position's 128 are its allocations' slices, each both slots'
    side by side, layer 0's qkv first); slot 0's head 2 reads group 1's (24p + 9, 128p + 7) and slot 1's head 0 (unit 4)
    its own (24p + 10, 128p + 14)."""
    circuit, ids = window
    first, end = ids["AttnBlockFirst_v1{D=16,NVIS=5,BN=128}"], ids["AttnHeadEnd_v1{D=16}"]
    rope, gemm = ids["RoPEHead_v1{D=16}"], ids["Gemm_v2{K=64,N=16}"]
    for u, k, v, q in ((0, 8, 6, 96), (1, 8, 6, 97), (2, 9, 7, 98), (4, 10, 14, 100)):
        assert [_source(circuit, first, "k", u, r) for r in range(5)] == [(rope, 24 * p + k, "o0") for p in range(5)]
        assert [_source(circuit, first, "v", u, r) for r in range(5)] == [(gemm, 128 * p + v, "o0") for p in range(5)]
        assert _source(circuit, first, "q", u) == (rope, q, "o0")
    assert [_source(circuit, end, p, 64 + 0) for p in ("l", "o")] == [(first, 0, "o1"), (first, 0, "o2")]
    assert [_source(circuit, end, p, 64 + 5) for p in ("l", "o")] == [(first, 5, "o1"), (first, 5, "o2")]


def test_past_126_keys_a_key_block_is_refused_by_name():
    """A key block reads its keys' rows and its values', a row a key each, besides its query and its three outputs, so past
    126 keys it holds more than a unit's 256 rows: one Program of 126 positions at TINY of 2,048 positions lowers, its
    largest unit exactly 256 rows, and at 129 (position 128 the first past one key block of 128) the first blocks of 127 and
    128 keys and the full block `AttnBlock_v5{NVIS=128}` are refused by name, together."""
    reg, grain = S._registered({**S.TINY, "max_position_embeddings": 2048}), Grain(gemm_words=1024, rows=256, select=64)
    assert max(_unit_rows(class_circuit(reg, WindowClass(1, 125, 2), 1, grain, provable=False)).values()) == 256
    first = "AttnBlockFirst_v1{{D=16,NVIS={},BN=128,DOT=HopperBF16WgmmaDot16_v1}}: {} rows in and out, past a unit's 256"
    with pytest.raises(Unexpressible, match=re.escape(
            f"3 templates the window can't hold, each port cut once a writer row: {first.format(127, 258)}; "
            f"{first.format(128, 260)}; AttnBlock_v5{{D=16,NVIS=128,FIRST=False,BN=128,DOT=HopperBF16WgmmaDot16_v1,"
            "CHECK=False}: 263 rows in and out, past a unit's 256")):
        class_circuit(reg, WindowClass(1, 128, 2), 1, grain, provable=False)


def _labels(circuit) -> dict[str, np.ndarray]:
    """Each template's units' (slot, step), by descriptor id: its inference entries expanded, an entry's digits in mixed radix
    with the first the slowest."""
    out = {tid: np.full((n, 2), -1, dtype=np.int64) for tid, n in circuit.units.items()}
    for e in circuit.blocks.inference:
        i, lab = np.arange(e.units, dtype=np.int64), np.zeros((e.units, 2), dtype=np.int64)
        lab[:, 1] = e.step
        for r, a in zip(reversed(e.radix), reversed(e.axes)):
            if a != "-":
                lab[:, int(a == "step")] += i % r
            i //= r
        out[e.template][e.at:e.at + e.units] = lab
    return out


def test_the_blocks_put_each_unit_at_its_slot_and_its_positions_step(window):
    """The inference block (`interface.Blocks`): the class's 2 slots are its slots and its 5 positions its steps, each
    allocation's units an entry of its slots' runs side by side, like entries merged. Position p's embedding and select
    with its collector are step p, its feed (p = 1..3) step p, a slot's scan read step 0 and its entry step 4, a head's Calls
    over T keys step T - 1 (their digits layer, slot, head; the ends one template over every step), and the entries tile
    every template.
    Every wire's writer is of its reader's slot at its step or before: a slot reads only its own units, a position only its
    own and earlier positions'."""
    circuit, ids = window
    label = {tid: name for name, tid in ids.items()}
    assert (circuit.blocks.slots, circuit.blocks.steps, len(circuit.blocks.inference)) == (2, 5, 59)
    by: dict[str, list] = {}
    for e in circuit.blocks.inference:
        by.setdefault(label[e.template], []).append((e.at, e.radix, e.axes, e.step))
    assert set(by) == set(ids)
    assert by[EMB] == [(0, (5, 2), ("step", "slot"), 0)]
    assert by[SCAN] == by[COLLECT0] == [(0, (2,), ("slot",), 0)]
    assert by[FEED] == by[COLLECT] == [(0, (3, 2), ("step", "slot"), 1)] and by[ENTRY] == [(0, (2,), ("slot",), 4)]
    att = "AttnBlockFirst_v1{{D=16,NVIS={},BN=128}}"
    assert by[att.format(5)] == [(0, (2, 2, 4), ("-", "slot", "-"), 4)]
    assert by["AttnHeadEnd_v1{D=16}"] == [(0, (5, 2, 2, 4), ("step", "-", "slot", "-"), 0)]
    assert [{s for *_, s in by[att.format(t)]} for t in range(1, 6)] == [{t} for t in range(5)]
    assert by["RoPEHead_v1{D=16}"][:2] == [(0, (2, 4), ("slot", "-"), 0), (8, (2, 2), ("slot", "-"), 0)]
    labels = _labels(circuit)
    assert all((lab >= 0).all() for lab in labels.values())
    for w in circuit.wires:
        i = np.arange(w.units, dtype=np.int64)
        reader, writer = labels[w.reader][w.at + i // w.cut[1]], labels[w.writer][w.source(i)]
        assert (reader[:, 0] == writer[:, 0]).all() and (writer[:, 1] <= reader[:, 1]).all(), w.doc()


def test_inference_entries_are_the_slots_runs_side_by_side():
    """`window_builder.inference` takes a template's runs in unit order (a slot's are allocated before the next slot's, each allocation's
    slots side by side), merges its units of one (slot, step) into runs, takes each ``slots`` runs (slots 0, 1, ... of one
    length at one step) as an entry, and merges like entries that follow on at one step under a ``-`` digit, then at
    consecutive steps under a ``step`` digit; units that aren't such runs are refused by name."""
    assert WB.inference({"t": [(0, 1, 0, 0), (1, 1, 0, 0), (2, 2, 1, 0), (4, 2, 0, 1), (6, 2, 1, 1)]}, 2) == [
        IF.InferenceEntry("t", 0, (2, 2, 2), ("step", "slot", "-"), 0)]
    for spans in ([(0, 1, 0, 3), (1, 1, 1, 3), (2, 1, 0, 3), (3, 1, 1, 3)], [(0, 1, 0, 3), (2, 1, 0, 3), (1, 1, 1, 3), (3, 1, 1, 3)]):
        assert WB.inference({"t": spans}, 2) == [IF.InferenceEntry("t", 0, (2, 2), ("-", "slot"), 3)]
    with pytest.raises(Unexpressible, match=r"t: units from 0 are not 2 slots' runs of one length at one step"):
        WB.inference({"t": [(0, 2, 0, 0), (2, 1, 1, 0)]}, 2)


def test_every_input_row_is_wired_or_registered_counted_from_the_description(window):
    """`interface.input_count` over the circuit's description (JSON, no Program lowered), each registered parameter pinned
    by the verifier itself before the run at the root the window's registration records for it (`held`): each weight's (one
    a weight and row width, ``weights/<label>@<words>``) and the scan state's, which the network certifier's scan proofs
    hold. It finds no free input row and no ingress, so a proof of the window's units leaves nothing but the declared advice
    to the prover. The registered reads, table reads and openings are of the weights (the GEMM slices', norms' and RoPE
    tables' rows; the embedding table's root, domain and rows) but for the two slots' scan state rows (4,548 words each):
    with the weights' pins alone those are free, with no pin all of them are, a weight whose window root is another's is
    free by its own reads, and a pin of ``weights`` holds none of them."""
    circuit, _ = window
    doc, reg = json.loads(json.dumps(circuit.description())), circuit.registered
    assert {r.param for r in reg.reads} | {t.param for t in reg.tables} == set(reg.params)
    assert {p.split("/", 1)[0] for p in reg.params if p != SS.SCAN_STATE} == {WC.WEIGHTS} and SS.SCAN_STATE in reg.params
    scan = 2 * 4548 * 16
    count = {"units": 1388, "inputs": 16340384, "wired": 1607968, "registered": 14586880 + scan, "ingress": 0, "free": 0}
    pins, roots = held(circuit).values()
    assert IF.input_count(doc, pins=pins, roots=roots) == count
    weights = {p: r for p, r in pins.items() if p != SS.SCAN_STATE}
    assert IF.input_count(doc, pins=weights, roots=roots) == {**count, "registered": 14586880, "free": scan}
    free = {**count, "registered": 0, "free": 14586880 + scan}
    assert IF.input_count(doc) == IF.input_count(doc, pins=pins) == free
    assert IF.input_count(doc, pins={WC.WEIGHTS: "ab" * 64}, roots={**roots, WC.WEIGHTS: "ab" * 64}) == free
    one = WB.weight_param("model.norm.weight", 64)
    alone = IF.input_count(doc, pins={one: pins[one]}, roots=roots)["registered"]
    other = IF.input_count(doc, pins=pins, roots={**roots, one: "cd" * 64})["registered"]
    assert 0 < alone and other == count["registered"] - alone


#: TINY's rotary table's parameter, the one weight with no byte digest in the weights record (held to the analytic table)
ROPE = WB.weight_param("model.layers.0.self_attn.rotary_emb.cos_sin_cache", 16)
#: the scan state's pin, which no weights registration gives: the network certifier's scan rows' root
SCAN_PIN = {SS.SCAN_STATE: "ef" * 64}
#: the committer's secret key, which its weights' row salts are drawn under (`window_weights.weights_registration`)
KEY = bytes(range(32))


def _alone(doc, param: str) -> int:
    """The bits ``param``'s reads register, pinned alone at the window's root for it."""
    return IF.input_count(doc, pins={param: "ab" * 64}, roots={param: "ab" * 64})["registered"]


def _pins(tmp_path, key: bytes = KEY) -> dict[str, str]:
    """The verifier's pins of TINY's pinned checkpoint at the test grain: the committer's registration under ``key``, held to
    the verifier's weights record (`registration_pins`)."""
    reg, (d, man) = S._registered(), _checkpoint(tmp_path)
    return WW.registration_pins(WW.weights_registration(reg, key, d, checkpoints_manifest=man, grain=GRAIN),
                                WW.weights_record(reg, d, man, grain=GRAIN))


def test_the_committers_registered_roots_hold_every_weight_read_but_the_rotary_tables(window, tmp_path):
    """The committer's registration of its weights (`weights_registration` at the window's grain under its key, of the pinned
    checkpoint), held by the verifier to its own weights record's shapes (`registration_pins`) and recorded by the window's
    registration (`advice.registered_weights`), holds every weight read, the embedding's table's and openings included, but
    the rotary table's, which a registration not given the engine's served table leaves unrooted: with the scan state
    pinned at its own root, the window's free bits are the rotary table's reads alone. The verifier's record states the
    rotary table's shape all the same. Each root is core's `Registered.commit` of the weight's rows (a table's ``row-seg`` rows) salted
    `audit.hm96_salt` of the committer's key, a hiding commitment: the verifier's record names no root, another key commits
    another root for every weight, and a window that serves the checkpoint's words commits the registered roots
    (`weight_values`)."""
    circuit, _ = window
    doc = json.loads(json.dumps(circuit.description()))
    reg, (d, man) = S._registered(), _checkpoint(tmp_path)
    registration = WW.weights_registration(reg, KEY, d, checkpoints_manifest=man, grain=GRAIN)
    record = WW.weights_record(reg, d, man, grain=GRAIN)
    pins = WW.registration_pins(registration, record)
    assert ADV.registered_weights(registration) == pins
    weights = set(circuit.registered.params) - {SS.SCAN_STATE}
    assert set(pins) == weights - {ROPE} and list(registration["unrooted"]) == [ROPE] and not record["unrooted"]
    assert set(record["registered_entries"]) == weights
    assert "registered_roots" not in record and not [e for e in record["registered_entries"].values() if "root" in e]
    assert registration["grain"] == record["grain"] == vars(GRAIN) and registration["program"] == record["program_digest"]
    rope, both = _alone(doc, ROPE), {**pins, **SCAN_PIN}
    assert rope == 2 * 5 * 2 * 6 * 16 * 16          # each slot's position's layer's q and k heads, a row of 16 words
    assert IF.input_count(doc, pins=both, roots=both) == {"units": 1388, "inputs": 16340384, "wired": 1607968,
                                                         "registered": 14586880 + 2 * 4548 * 16 - rope, "ingress": 0,
                                                         "free": rope}
    ck, nonces = WR.CheckpointIndex(d, checkpoints_manifest=man, model="org/m", revision="r"), WW.weight_nonces(weights)
    emb = "model.embed_tokens.weight"
    for name, p, schema in (("model.norm.weight", WB.weight_param("model.norm.weight", 64), G.SCHEMA_V2),
                            (emb, WB.weight_param(emb, 64), G.SCHEMA_V2), (emb, WB.table_param(emb, 64), G.SCHEMA_SEG)):
        rows = ck.array(name).view(np.uint16).reshape(-1, 64).tolist()
        want = G.Registered.commit(registration["program"], p, rows, [AU.hm96_salt(KEY, nonces[p], j) for j in range(len(rows))],
                                   schema=schema)
        assert want.entry() == registration["values"][p]
    other = _pins(tmp_path / "other", bytes(32))
    assert set(other) == set(pins) and not [p for p in pins if other[p] == pins[p]]
    served = WW.weight_values(reg, WW.record_words(reg, d, man), KEY, GRAIN)
    assert {p: v.root.hex() for p, v in served.items()} == pins


@pytest.mark.parametrize("weight, served_as", [
    ("model.layers.0.self_attn.k_proj.weight", ["model.layers.0.self_attn.qkv_proj.weight@1024"]),
    ("model.embed_tokens.weight", ["model.embed_tokens.weight@1024", "model.embed_tokens.weight@64",
                                   "model.embed_tokens.weight@64/table"]),
])
def test_a_changed_weight_byte_is_never_registered_and_leaves_exactly_that_weights_reads_free(window, tmp_path, weight, served_as):
    """A committer serving a checkpoint one byte of ``weight`` away from the pinned one refuses to register that weight (its
    served words aren't the pinned shards'), and only that one; a window that serves it under the registered key commits
    another root for each parameter of the weight it is served as (k_proj's byte is the fused qkv's, the tied embedding's
    is its table's, the LM head's and ``x0``'s) and the registered root for every other, so those reads alone open no
    registered read: they go free (the scan state pinned at its own root)."""
    circuit, _ = window
    doc = json.loads(json.dumps(circuit.description()))
    reg, (d, man) = S._registered(), _checkpoint(tmp_path)
    pins = WW.registration_pins(WW.weights_registration(reg, KEY, d, checkpoints_manifest=man, grain=GRAIN),
                                WW.weights_record(reg, d, man, grain=GRAIN))
    served = WW.record_words(reg, *_checkpoint(tmp_path / "served", flip=weight))
    refused = {served_as[0].rpartition("@")[0]: "served words that aren't the checkpoint's"}
    with pytest.raises(ValueError, match=re.escape(f"no weight registered: {refused} ")):
        WW.weights_registration(reg, KEY, d, served, checkpoints_manifest=man, grain=GRAIN)
    roots = {p: v.root.hex() for p, v in WW.weight_values(reg, served, KEY, GRAIN).items()}
    changed = [f"{WC.WEIGHTS}/{p}" for p in served_as]
    assert set(roots) == set(pins) and sorted(p for p in pins if pins[p] != roots[p]) == changed
    free = _alone(doc, ROPE) + sum(_alone(doc, p) for p in changed)
    assert IF.input_count(doc, pins={**pins, **SCAN_PIN}, roots={**roots, **SCAN_PIN})["free"] == free > _alone(doc, ROPE)


def test_the_engines_rotary_table_registers_within_its_bound_and_a_wrong_one_is_refused(window, tmp_path):
    """Given the words the engine serves of the rotary table (``tables``), the committer registers it as it registers a
    weight, once they are within the analytic table's bound of config.json (`check.weights_of_record`'s check of a served
    table, `model.check_cos_sin`), and the verifier holds its entry to the shape its own record declares: with the scan
    state pinned at its own root no input bit is free, and a window serving those words commits the registered root. A
    table of another rope theta, of another length or of 32-bit words is refused by name, nothing registered; the
    verifier refuses a registration whose rotary entry commits a table of another length, or that drops it without naming
    it unrooted; and a window serving another table than the registered one commits another root, so the rotary table's
    reads, and only those, go free."""
    circuit, _ = window
    doc = json.loads(json.dumps(circuit.description()))
    reg, (d, man) = S._registered(), _checkpoint(tmp_path)
    record = WW.weights_record(reg, d, man, grain=GRAIN)
    field = next(f for f in record["fields"] if f.get("pinned") == "analytic")
    C, table = field["analytic"]["C"], M.cos_sin_table(field["analytic"]["C"])
    registration = WW.weights_registration(reg, KEY, d, checkpoints_manifest=man, grain=GRAIN, tables={"cos_sin": table})
    pins, weights = WW.registration_pins(registration, record), set(circuit.registered.params) - {SS.SCAN_STATE}
    assert set(pins) == weights and not registration["unrooted"] and ADV.registered_weights(registration) == pins
    unrooted = WW.weights_registration(reg, KEY, d, checkpoints_manifest=man, grain=GRAIN)
    assert {p: e for p, e in registration["values"].items() if p != ROPE} == unrooted["values"]
    both = {**pins, **SCAN_PIN}
    assert IF.input_count(doc, pins=both, roots=both) == {"units": 1388, "inputs": 16340384, "wired": 1607968,
                                                         "registered": 14586880 + 2 * 4548 * 16, "ingress": 0, "free": 0}
    served = {**WW.record_words(reg, d, man), field["field"]: table}
    assert {p: v.root.hex() for p, v in WW.weight_values(reg, served, KEY, GRAIN).items()} == pins

    other = M.cos_sin_table({**C, "ROPE_THETA": 2 * C["ROPE_THETA"]})
    for bad, why in [(other, r"\d+ of \d+ served words outside the analytic bound"),
                     (table[: len(table) // 2], re.escape(f"vs analytic {field['shape']}")),
                     (table.astype(np.float32), "served float32 words, not the table's 16-bit ones")]:
        with pytest.raises(ValueError, match=f"no weight registered: .*{why}"):
            WW.weights_registration(reg, KEY, d, checkpoints_manifest=man, grain=GRAIN, tables={"cos_sin": bad})
    values = registration["values"]
    short = WW.weight_value(registration["program"], ROPE, table[: len(table) // 2], KEY, WW.weight_nonces(weights)[ROPE])
    for bad, why in [({**registration, "values": {**values, ROPE: short.entry()}}, f"entries for ['{ROPE}'] aren't the record's shapes"),
                     ({**registration, "values": {p: e for p, e in values.items() if p != ROPE}}, f"registers ['{ROPE}'] unlike")]:
        with pytest.raises(ValueError, match=re.escape(why)):
            WW.registration_pins(bad, record)
    roots = {p: v.root.hex() for p, v in WW.weight_values(reg, {**served, field["field"]: other}, KEY, GRAIN).items()}
    assert sorted(p for p in pins if roots[p] != pins[p]) == [ROPE]
    assert IF.input_count(doc, pins=both, roots={**roots, **SCAN_PIN})["free"] == _alone(doc, ROPE) > 0


def _checkpoint(tmp_path, seed: int = 3, flip: str | None = None, quant: dict | None = None) -> tuple[str, str]:
    """TINY's checkpoint as safetensors shards of its HF names (random bf16 weights from ``seed``, the low bit of ``flip``'s
    first element flipped) and a manifest pinning it at `test_steps._registered`'s model and revision: the checkpoint
    directory and the manifest. With ``quant`` (its config's ``quantization_config``), each linear is an exact-int8 one's,
    ``weight`` I8 [N, K] and ``weight_scale`` F32 [N, 1] (`engine.int8_checkpoint`)."""
    T, rng = S.TINY, np.random.default_rng(seed)
    H, I, NH, KVH, V = T["hidden_size"], T["intermediate_size"], T["num_attention_heads"], T["num_key_value_heads"], T["vocab_size"]
    D = H // NH
    t = {"model.embed_tokens.weight": _bf16(rng, V, H), "model.norm.weight": _bf16(rng, H)}
    for i in range(T["num_hidden_layers"]):
        p = f"model.layers.{i}."
        t |= {p + "self_attn.q_proj.weight": _bf16(rng, NH * D, H), p + "self_attn.k_proj.weight": _bf16(rng, KVH * D, H),
              p + "self_attn.v_proj.weight": _bf16(rng, KVH * D, H), p + "self_attn.o_proj.weight": _bf16(rng, H, NH * D),
              p + "mlp.gate_proj.weight": _bf16(rng, I, H), p + "mlp.up_proj.weight": _bf16(rng, I, H),
              p + "mlp.down_proj.weight": _bf16(rng, H, I), p + "input_layernorm.weight": _bf16(rng, H),
              p + "post_attention_layernorm.weight": _bf16(rng, H)}
    t = {n: ("BF16", a) for n, a in t.items()}
    for n in [n for n in t if quant and n.endswith("_proj.weight")]:
        (N, K), s = t[n][1].shape, n.removesuffix("weight") + "weight_scale"
        t[n], t[s] = ("I8", rng.integers(-127, 128, (N, K), dtype=np.int8)), ("F32", rng.random((N, 1), dtype=np.float32))
    if flip:
        a = t[flip][1]
        a.view(f"u{a.itemsize}").flat[0] ^= 1
    d = tmp_path / "snap"
    d.mkdir(parents=True, exist_ok=True)
    write_safetensors(str(d / "model.safetensors"), t)
    (d / "config.json").write_text(json.dumps({**T, "rope_theta": 10000.0, **({"quantization_config": quant} if quant else {})}))
    man = tmp_path / "checkpoints.json"
    model = S._registered().phi.model
    man.write_text(json.dumps({"checkpoints": [{"repo": model["repo"], "revision": model["revision"], "local_path": str(d),
                                                "files": {"model.safetensors": {"sha256": WR.sha256_file(str(d / "model.safetensors"))}}}]}))
    return str(d), str(man)


def test_the_verifier_holds_the_committers_registration_to_its_weights_records_shapes(tmp_path):
    """`weights_record`, the verifier's: every field of the class Program's weights record (TINY's 23: each layer's fused qkv
    and gate-up, o and down projections and norms, the embedding, the final norm, the rotary table and the attention
    scales), in the Program's order, composed from the checkpoint's pinned shards, so it is of record, and for each weight
    parameter the class circuit registers at the grain the record names its entry's shape (value, leaves, words, domain;
    the rotary table's from its declared shape) and no root. One key's registration of one checkpoint is one registration;
    the verifier pins its roots only where it registers exactly the record's parameters at their shapes, but those it
    names unrooted (`registration_pins`), and a window
    records only a registration (`advice.registered_weights`): never a weights record, nor one of another model than the
    run served. A key that isn't 32 bytes draws no salt; a committer registers no checkpoint its manifest pins as another
    model, and a shard the manifest doesn't pin leaves the checkpoint not of record, which no committer registers."""
    reg = S._registered()
    d, man = _checkpoint(tmp_path)
    rec = WW.weights_record(reg, d, man, grain=GRAIN)
    assert (rec["of_record"], rec["complete"], rec["n_fields"], rec["unresolved"]) == (True, True, 23, [])
    program = WC.class_program(reg, WW.WEIGHTS_CLASS).program
    assert [f["field"] for f in rec["fields"]] == [n for n, _ in dict(program.fn.signature()[0])[WC.WEIGHTS].fields]
    every, shapes = class_circuit(reg, WW.WEIGHTS_CLASS, grain=GRAIN, provable=False).registered.params, rec["registered_entries"]
    params = {p: every[p] for p in WW.weight_params(every)}
    assert set(every) - set(params) == {SS.SCAN_STATE}
    assert rec["grain"] == vars(GRAIN) and set(shapes) | set(rec["unrooted"]) == set(params)
    assert len(shapes) == len(params) == 17 and not rec["unrooted"]
    assert not [p for p, e in shapes.items() if set(e) != {"value", "leaves", "words", "domain"} or e["value"] != p
                or e["leaves"] * e["words"] != params[p]]
    registration = WW.weights_registration(reg, KEY, d, checkpoints_manifest=man, grain=GRAIN)
    assert registration["model"] == {"repo": "org/m", "revision": "r"} and set(registration["values"]) == set(shapes) - {ROPE}
    pins = WW.registration_pins(registration, rec)
    assert ADV.registered_weights(registration) == ADV.registered_weights(registration, {"model": "org/m", "revision": "r"}) == pins
    d2, man2 = _checkpoint(tmp_path / "again")
    assert WW.weights_registration(reg, KEY, d2, checkpoints_manifest=man2, grain=GRAIN) == registration
    values, one = registration["values"], WB.weight_param("model.norm.weight", 64)
    for bad, record, why in [
            ({**registration, "schema": WR.SCHEMA}, rec, f"field 'schema' is {WR.SCHEMA!r}, not {WRG.SCHEMA}"),
            (registration, {k: v for k, v in rec.items() if k != "registered_entries"}, "the record states none"),
            ({**registration, "values": {p: e for p, e in values.items() if p != one}}, rec, f"registers ['{one}'] unlike"),
            ({**registration, "values": {p: e for p, e in values.items() if p != one},
              "unrooted": {**registration["unrooted"], one: "no byte digest"}}, rec, f"registers ['{one}'] unlike"),
            ({**registration, "values": {**values, one: {**values[one], "leaves": 2 * values[one]["leaves"]}}}, rec,
             f"entries for ['{one}'] aren't the record's shapes"),
            ({**registration, "values": {**values, one: {**values[one], "root": "ab"}}}, rec,
             f"field 'values.{one}' doesn't name its parameter with a root of 64 bytes"),
            ({**registration, "values": {**values, one: {**values[one], "salt": "00" * 192}}}, rec,
             f"field 'values.{one}.salt' is not a field of the record")]:
        with pytest.raises(ValueError, match=re.escape(why)):
            WW.registration_pins(bad, record)
    for bad, serve, why in [(rec, None, "never a weights record"),
                            ({**registration, "values": {"x": {**values[one], "value": "x"}}}, None, "registers ['x'], not weight"),
                            (registration, {"model": "org/other", "revision": "r"}, "the run served org/other@r")]:
        with pytest.raises(ValueError, match=re.escape(why)):
            ADV.registered_weights(bad, serve)
    with pytest.raises(ValueError, match="32-byte key"):
        WW.weights_registration(reg, KEY[:16], d, checkpoints_manifest=man, grain=GRAIN)
    other = tmp_path / "other.json"
    other.write_text(Path(man).read_text().replace('"repo": "org/m"', '"repo": "org/other"'))
    with pytest.raises(ValueError, match=re.escape("manifest entry is org/other@r, not the registered model org/m@r")):
        WW.weights_registration(reg, KEY, d, checkpoints_manifest=str(other), grain=GRAIN)
    shard = Path(d) / "model.safetensors"
    body = bytearray(shard.read_bytes())
    body[-1] ^= 1
    shard.write_bytes(bytes(body))
    unpinned = WW.weights_record(reg, d, man, grain=GRAIN)
    assert unpinned["of_record"] is False and unpinned["root_of_record"] != rec["root_of_record"]
    with pytest.raises(ValueError, match="the checkpoint is not of record"):
        WW.weights_registration(reg, KEY, d, checkpoints_manifest=man, grain=GRAIN)
    assert WR.SCHEMA == rec["schema"]


def test_what_the_class_doesnt_take_is_refused_by_name():
    """A window past its class (three requests in two slots, a prompt past the cap, more outputs than the cap), a stochastic
    request (its Program would read a seed) and a unit past the grain's rows (at 12, position 4's key block's 14) are
    refused, and so is a Program count the class has no slots for."""
    reg = S._registered()

    def refused(window, cls=CLASS, grain=GRAIN):
        return lambda: window_circuit(reg, S.compile_steps(reg, *S._session(*window)), cls, grain, provable=False)

    three = ([(2, 2, G2)] * 3, [[(0, 0, 2), (1, 0, 2), (2, 0, 2)], [(0, 2, 1), (1, 2, 1), (2, 2, 1)]])
    long = ([(5, 2, G2)], [[(0, 0, 5)], [(0, 5, 1)]])
    many = ([(2, 3, {**S.GREEDY, "max_tokens": 3})], [[(0, 0, 2)], [(0, 2, 1)], [(0, 3, 1)]])
    stoch = ([(4, 2, {**S.STOCH, "max_tokens": 2})], [[(0, 0, 4)], [(0, 4, 1)]])
    for window, match in ((three, "3 requests in a window of a class of 2 slots"), (long, "5 prompt ids and 2 outputs, past"),
                          (many, "2 prompt ids and 3 outputs, past its class's 4 and 2"), (stoch, "a stochastic request")):
        with pytest.raises(Unexpressible, match=match):
            refused(window)()
    with pytest.raises(Unexpressible, match=re.escape(
            "1 templates the window can't hold, each port cut once a writer row: AttnBlockFirst_v1{D=16,NVIS=5,BN=128,"
            "DOT=HopperBF16WgmmaDot16_v1}: 14 rows in and out, past a unit's 12")):
        refused(WINDOW, grain=Grain(gemm_words=1024, rows=12, select=64))()
    for n in (0, 3):
        with pytest.raises(Unexpressible, match=f"{n} Programs in a window of a class of 2 slots"):
            class_circuit(reg, CLASS, n, GRAIN)


def test_a_window_its_slots_dont_place_is_refused_by_name():
    """At a class of one slot at each of buckets 0 and 16 (a 32-bucket window), a window placed as the network certifier's
    scan placed it is the class's circuit, a last output past the window's end included (the scan carries it across
    windows); refused by name are two requests at one boundary, an anchor that is no boundary and a last output before its
    anchor. A placement that isn't one a request is a caller's error."""
    cls = WindowClass(2, 4, 2, lanes=1, every=16)
    assert (cls.boundaries, cls.buckets, cls.anchor(0), cls.anchor(1)) == ((0, 16), 32, 0, 16)
    ok = _circuit(WINDOW, cls=cls, placed=[(0, 15), (16, 31)])
    assert _bytes(ok) == _bytes(class_circuit(S._registered(), cls, None, GRAIN, provable=False))
    assert _bytes(_circuit(WINDOW, cls=cls, placed=[(0, 15), (16, 47)])) == _bytes(ok)
    assert [(r.at, r.radix, r.base, r.step) for r in ok.registered.reads if r.param == SS.SCAN_STATE] == [
        (0, (2,), 0, (16 * 2277,))]
    for placed, match in (
            (None, "2 requests anchored at bucket 0, past the 1 slots a boundary has"),
            ([(16, None), (16, None)], "2 requests anchored at bucket 16, past the 1 slots"),
            ([(0, None), (8, None)], r"request 1: anchored at bucket 8, none of its class's boundaries \[0, 16\]"),
            ([(16, 3), (0, None)], "request 0: its last output at bucket 3, before its anchor 16")):
        with pytest.raises(Unexpressible, match=match):
            _circuit(WINDOW, cls=cls, placed=placed)
    with pytest.raises(ValueError, match="1 placements for the window's 2 requests"):
        _circuit(WINDOW, cls=cls, placed=[(0, None)])
    for bad in ({"lanes": 3}, {"every": 0}):
        with pytest.raises(ValueError, match="lanes dividing its slots"):
            WindowClass(8, 4, 2, **bad)


def _unit_rows(circuit) -> dict[str, int]:
    """Each template's rows in and out under the circuit's cuts, as C-Flock's `UnitStager` takes them
    (`interface.unit_rows`)."""
    return {_label(fn): sum(map(len, IF.unit_rows(fn, circuit.cuts.get(IF.descriptor_id(fn))))) for fn, _ in circuit.templates}


def test_no_unit_holds_more_rows_in_and_out_than_the_grain(window):
    """C-Flock's composer gives each row a unit reads or writes an hm96 slot, so at Llama-3.2-1B's grain a unit fits its
    2^27-bit block with at most 256 rows in and out. A port is cut once a writer row, so a unit's rows are its writers' rows,
    its registered reads and its outputs: at the test grain the largest are SiluMul's (its 16 gate/up slices and its output,
    17), position 4's key block's (5 key rows, 5 value rows, its query and three outputs, 14), the fused norm's (8 down or
    o_proj slices, its residual, its weight and two outputs, 12) and the scan's (its state row and two lanes' four outputs,
    9). At 16 rows SiluMul's 17 are its body's halves instead, `SiluMul_v2{I=64}` of 8 gate/up slices each. No unit holds
    more than the grain's rows."""
    circuit, _ = window
    rows = _unit_rows(circuit)
    assert (rows["SiluMul_v2{I=128}"], rows["AttnBlockFirst_v1{D=16,NVIS=5,BN=128}"], rows["RMSNormFusedCuda_v2{N=64}"],
            rows[SCAN]) == (17, 14, 12, 9)
    assert max(rows.values()) == 17 <= GRAIN.rows
    small = _circuit(WINDOW, Grain(gemm_words=1024, rows=16, select=64))
    n, rows = {_label(fn): k for fn, k in small.templates if fn.name == "SiluMul"}, _unit_rows(small)
    assert n == {"SiluMul_v2{I=64}": 40} and rows["SiluMul_v2{I=64}"] == 9 and max(rows.values()) == 14


def _widest_row(circuit) -> dict[str, int]:
    """Each template's widest row in or out under the circuit's cuts, in bytes, as C-Flock's `UnitStager` frames them (bit
    rows of exactly their words, `interface.unit_rows`)."""
    out = {}
    for fn, _ in circuit.templates:
        ins, _outs = IF.unit_rows(fn, circuit.cuts.get(IF.descriptor_id(fn)))
        out[_label(fn)] = max([b // 8 for _p, _s, b, _ps in ins] + [2 * w for _o, w in IF.layout(fn).outs])
    return out


def test_a_class_too_small_to_prove_is_refused_naming_each_template_c_flock_wouldnt_compose(window):
    """C-Flock's composer refuses a proof block none of whose rows is over 111 bytes (no row then has a compression before its
    last SHA-512 block, so the block's SHA range holds no slot), one of more than 256 rows, or one past 2^27 bits
    (`interface.check_proof_block`), and the class builder refuses such a unit at lowering, naming each. TINY's class is too
    small to prove: 12 templates have no wider row, its heads' of 16 values (RoPE, the key blocks, the head end), the token
    select's group units at 64 logits a unit (4 groups), its combines with the I/O shell's words at 4 prompt ids and 2
    outputs (the collectors, the entry) and the feed, so its tests lower it with ``provable=False``. Every other template has
    a wider row."""
    circuit, _ = window
    short = {lab: b for lab, b in _widest_row(circuit).items() if b <= 111}
    assert short == {"RoPEHead_v1{D=16}": 32, **{f"AttnBlockFirst_v1{{D=16,NVIS={t},BN=128}}": 64 for t in range(1, 6)},
                     "AttnHeadEnd_v1{D=16}": 64, GROUPS: 32, COLLECT0: 16, COLLECT: 16, FEED: 20, ENTRY: 16}
    with pytest.raises(Unexpressible) as refused:
        class_circuit(S._registered(), CLASS, None, GRAIN)
    why = str(refused.value)
    assert why.startswith(f"{len(short)} templates C-Flock's composer wouldn't prove: ")
    named = {_label(fn) for fn, _ in circuit.templates if re.search(re.escape(IF.descriptor_id(fn)) + "(: | cut )", why)}
    assert named == set(short)


#: the smallest test class C-Flock proves every unit of: a head of 64 values (one of 16 has no row past 111 bytes at any
#: class), prompts of 27 ids (the feed's slot 112 bytes) and one output, GEMM slices of 64 columns (a key's value one row)
#: and a token select of one unit of 512 logits (32 groups: its indices a row of 128 bytes, which its collector and the
#: entry share its unit with)
PROVABLE = {**S.TINY, "hidden_size": 128, "intermediate_size": 256, "num_attention_heads": 2, "num_key_value_heads": 1,
            "vocab_size": 512}
PROVABLE_CLASS, PROVABLE_GRAIN = WindowClass(2, 27, 1), Grain(gemm_words=8192, rows=256, select=512)


def test_the_smallest_provable_test_class_lowers_every_unit_to_a_proof_block():
    """At `PROVABLE`'s model, class and grain the class builder's check passes (`interface.check_proof_block`): its 3,996
    units of 43 templates each have a row past 111 bytes and are proof blocks C-Flock's composer takes, its token selects
    a `TokenSelectGroups_v1{512, 16}` unit and a combine with the position's collector a position a slot, the last
    position's with the entry."""
    circuit = class_circuit(S._registered(PROVABLE), PROVABLE_CLASS, None, PROVABLE_GRAIN)
    n = {_label(fn): k for fn, k in circuit.templates}
    assert (sum(n.values()), len(n)) == (3996, 43) and min(_widest_row(circuit).values()) > 111
    assert all(IF.check_proof_block(fn, circuit.cuts.get(IF.descriptor_id(fn))) <= 27 for fn, _ in circuit.templates)
    P = PROVABLE_CLASS.positions
    assert n["TokenSelectGroups_v1{C=512,G=16}"] == 2 * P and (
        n["SelectCollect_v1{V=512,C=16,KO=1,FIRST=true}"], n["SelectCollect_v1{V=512,C=16,KO=1,FIRST=false}"],
        n["SelectEntry_v1{V=512,C=16,KO=1,FIRST=false,IDB=17,NP=27}"]) == (2, 2 * (P - 2), 2)


def test_a_token_select_is_its_function_at_the_grains_units_of_groups():
    """At 128 logits a unit, each of the circuit's ten token selects is two `TokenSelectGroups_v1{C=128, G=16}` units, each
    reading 8 LM-head slices' rows as they are (its ``x`` cut 8) and writing its 8 groups' bests and indices a row each, and a
    `TokenSelectCombine_v2{V=256, C=16}` reading the two units' rows (its position's `SelectCollect_v1` or, at the last,
    `SelectEntry_v1`): the Program's own body (chunks of 64) at groups of 16. A grain whose select isn't whole units of V, or
    whose units aren't whole groups, is refused."""
    circuit = _circuit(WINDOW, Grain(gemm_words=1024, rows=256, select=128))
    n = {_label(fn): k for fn, k in circuit.templates}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    groups, combines = ids["TokenSelectGroups_v1{C=128,G=16}"], [ids[s] for s in (COLLECT0, COLLECT, ENTRY)]
    assert (n["TokenSelectGroups_v1{C=128,G=16}"], sum(n[s] for s in (COLLECT0, COLLECT, ENTRY))) == (20, 10) and GROUPS not in n
    assert circuit.cuts[groups] == {"x": 8} and all(circuit.cuts[c] == {"best": 2, "idx": 2} for c in combines)
    assert {_source(circuit, groups, "x", u, r)[0] for u in range(20) for r in range(8)} == {ids["Gemm_v2{K=64,N=16}"]}
    assert [w for _, w in IF.layout(circuit.template(groups)).outs] == [8, 16]
    assert IF.input_count(circuit.description(), **held(circuit))["free"] == 0
    for select in (96, 8):
        with pytest.raises(Unexpressible, match=rf"256 logits aren't units of {select}, each groups of 16"):
            _circuit(WINDOW, Grain(gemm_words=1024, rows=256, select=select))


#: TINY at Llama's MLP ratio (intermediate four times hidden) and GEMM slices of 512 weight words: down_proj writes 32 rows of
#: 2 columns where o_proj writes 16 of 4, and a value GEMM's slice is half a key's value
SPLIT, SPLIT_GRAIN = {**S.TINY, "intermediate_size": 256}, Grain(gemm_words=512, rows=256, select=64, norm=16)
NORM, TAIL = "RMSNormFusedCuda_v2{N=64}", "RMSNormFusedCudaTail_v1{N=64}"


def _add(at: int) -> str:
    return f"RMSNormFusedCudaAdd_v1{{N=64,C=16,AT={at}}}"


@pytest.fixture(scope="module")
def split():
    return class_circuit(S._registered(SPLIT), CLASS, None, SPLIT_GRAIN, provable=False)


def test_a_fused_norm_read_at_another_row_count_is_cut_at_its_residual_add_in_order_of_at(split):
    """The norm after o_proj reads its x as o_proj's 16 rows, so the norm after down_proj (the next layer's input norm and the
    final norm), whose x is down_proj's 32, is cut at its residual add: 4 `RMSNormFusedCudaAdd_v1{N=64, C=16, AT}` units, part
    AT reading x's elements [16·AT, 16·(AT+1)) as 8 of down_proj's rows and res whole, and `RMSNormFusedCudaTail_v1{N=64}`,
    its r row j part AT = j's sum and its w a weight read. The GEMMs read the tail's normed row and the next residual add its
    r, as they read v2's. A first key block whose keys' values are two value-GEMM rows each is `AttnBlockFirst_v2{VS=8}`, its
    v cut a row a slice. No input row is free."""
    n = {_label(fn): k for fn, k in split.templates}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in split.templates}
    tail, down = ids[TAIL], ids["Gemm_v2{K=256,N=2}"]
    assert n[NORM] == n[TAIL] == 20 and [n[_add(at)] for at in range(4)] == [20] * 4
    assert (split.cuts[ids[NORM]], split.cuts[tail]) == ({"x": 16}, {"r": 4})
    assert all(_source(split, tail, "r", u, j) == (ids[_add(j)], u, "o0") for u in range(20) for j in range(4))
    for u in range(20):
        xs = [_source(split, ids[_add(at)], "x", u, r) for at in range(4) for r in range(8)]
        assert {w for w, _, _ in xs} == {down} and [k for _, k, _ in xs] == list(range(xs[0][1], xs[0][1] + 32))
    assert all(split.cuts[ids[_add(at)]] == {"x": 8} for at in range(4))
    readers = {(o, _label(split.template(w.reader)), w.port) for w in split.wires if w.writer == tail for o in [w.out]}
    assert readers == {("o0", "Gemm_v2{K=64,N=8}", "x"), ("o0", "Gemm_v2{K=64,N=1}", "x"), ("o1", NORM, "res")}
    first = {lab: split.cuts[ids[lab]]["v"] for lab in n if lab.startswith("AttnBlockFirst")}
    assert first == {f"AttnBlockFirst_v2{{D=16,NVIS={t},BN=128,VS=8}}": 2 * t for t in range(1, 6)}
    assert IF.input_count(split.description(), **held(split))["free"] == 0


def test_a_tail_wired_out_of_order_of_at_is_refused_by_the_wiring_check(split):
    """The tail is `RMSNormFusedCuda_v2` on its parts' sums in order of AT, and out of that order it normalizes another row. A
    window of one norm cut as the class circuit cuts it (its parts and tail, the tail's r wired as the circuit wires it)
    commits each sum and the tail's row of it as one public string, so the verifier's wiring check holds the tail's r to its
    writers on the public bodies, and the same rows read in a swapped order are refused there."""
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in split.templates}
    tail, v2 = split.template(ids[TAIL]), split.template(ids[NORM])
    parts = [split.template(ids[_add(at)]) for at in range(4)]
    order = {w.cut[0]: w.writer for w in split.wires if (w.reader, w.port) == (ids[TAIL], "r")}
    norm = IF.Circuit("norm", tuple((fn, 1) for fn in (*parts, tail)), cuts={ids[TAIL]: {"r": 4}},
                      wires=tuple(IF.Wire(ids[TAIL], "r", order[j], "o0", at=0, radix=(1,), base=0, step=(0,), cut=(j, 1))
                                  for j in range(4)))
    rng = np.random.default_rng(5)
    x, res, w = (_bf16(rng, 1, 64) for _ in range(3))

    def ev(fn, *args):
        return np.asarray(evaluate_batch(fn, *args, kernel="reference")).astype(np.uint16)

    sums = [ev(p, x[:, 16 * at:16 * (at + 1)], res).reshape(1, 16) for at, p in enumerate(parts)]
    out = ev(tail, np.concatenate(sums, axis=1), w).reshape(1, 128)
    assert np.array_equal(out, ev(v2, x, res, w).reshape(1, 128))
    assert not np.array_equal(ev(tail, np.concatenate([sums[1], sums[0], *sums[2:]], axis=1), w).reshape(1, 128), out)
    win = IF.Window(norm, "ab" * 64, {"law": "stratified", "k": 2}, 1)
    for at, p in enumerate(parts):
        win.record(p, [x[:, 16 * at:16 * (at + 1)], res], [sums[at]])
    win.record(tail, [None, w], [out[:, :64], out[:, 64:]])
    assert win.mismatches() == {}
    c, members = win.close(salt_key=bytes(32)), win.window.members
    publics = {m.key: c.public_body(k) for k, m in enumerate(members)}
    assert WI.check(members, publics) == {}
    (t,) = [m for m in members if m.wired]
    rows, swap = {q: (o, ws) for q, o, ws in t.wired}, {"r#0": "r#1", "r#1": "r#0"}
    moved = [replace(m, wired=tuple((q, *rows[swap.get(q, q)]) for q, _o, _ws in m.wired)) for m in members]
    assert set(WI.check(moved, publics)) == {t.key}
