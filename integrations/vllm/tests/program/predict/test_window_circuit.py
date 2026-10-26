"""verity_vllm.program.predict.window_circuit (V3b at ASK-122's fixed slots): a served window is its class's one Circuit of the
catalog's units, a class Program a slot whatever the traffic, every input row wired from its writer's row or a registered read
of the weights or the network certifier's scan state rows, and nothing free; a slot's outputs leave as its entry of the scan's
Programs' row. Pinned on a TINY class of 2 slots, 4 prompt ids and 2 outputs (5 positions a Program), at a grain small
enough to slice TINY's GEMMs and cut its ports; the class's circuit, its blocks included, is the same for windows that differ
in everything, their request count included. At the class of record's slots (8, 4 at each of buckets 0 and 16) with TINY's
caps, a 3-request and an 8-request window are one circuit. A head is
`AttentionHead_v19`'s Calls, a unit each: past 128 keys (TINY at 2,048 positions, a grain of 256 rows) its full key blocks'
gathers are made once and read by every later position, and the class of record itself lowers at TINY (slow).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
import pytest
from tests.program.predict import test_steps as S
from verity_vllm.program.predict import scan_state as SS
from verity_vllm.program.predict import window_builder as WB
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_circuit import CLASS_OF_RECORD, Grain, WindowClass, class_circuit, window_circuit

from verity.core.protocols.certifier import ingress_scan as IS
from verity.ml.definitions.registry import io_shell as IO
from verity.ml.windows import interface as IF

GRAIN = Grain(gemm_words=512, rows=9, select=64)
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
EMB, COMBINE = "EmbeddingRead_v1{V=256,H=64}", "TokenSelectCombine_v2{V=256,C=64}"
FIRST, FEED = "TokenFeed_v1{NP=4,FIRST=true}", "TokenFeed_v1{NP=4,FIRST=false}"
ROW, MASK = "SlotRow_v1{NP=4}", "SlotMask_v1{}"
#: the scan state read at TINY's 2 lanes (4,548 words a row), each position's collector and a slot's entry
SCAN, ENTRY = "ScanSlots_v1{NS=4548,NP=4,IDB=17,K0=12,K=45}", "SlotEntry_v1{KO=2,IDB=17,NP=4}"
COLLECT0, COLLECT = "OutCollect_v1{KO=2,FIRST=true}", "OutCollect_v1{KO=2,FIRST=false}"
#: the verifier's pins: the weights record's root and the scan state's
PINS = {WC.WEIGHTS, SS.SCAN_STATE}


def _label(fn) -> str:
    return fn.id.split("{")[0] + "{" + ",".join(f"{k}={json.dumps(v)}" for k, v in fn.bindings.items() if type(v) in (int, bool)) + "}"


def _circuit(window, grain: Grain = GRAIN, cls: WindowClass = CLASS, placed=None):
    reg = S._registered()
    return window_circuit(reg, S.compile_steps(reg, *S._session(*window)), cls, grain, placed=placed)


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
    """The privacy property (ASK-122: only the slot count is public): WINDOW and OTHER differ in their requests' arrival
    steps, prompt lengths (4 and 3 against 3 and 2) and output lengths, and a window of one request in its request count too,
    and their circuits' descriptions (units, wires, cuts, registered reads), blocks, templates and external ports are
    byte for byte the class's circuit: the blocks' slots are the class's 2 whatever the window's requests."""
    circuit, _ = window
    cls = class_circuit(S._registered(), CLASS, None, GRAIN)
    for other in (_circuit(OTHER), _circuit(ONE), cls):
        assert _bytes(other) == _bytes(circuit) and _blocks(other) == _blocks(circuit)
        assert (other.ids, other.external) == (circuit.ids, circuit.external)
        assert (other.blocks.slots, other.blocks.steps) == (2, 5)


def test_a_3_request_and_an_8_request_window_at_the_class_of_records_slots_are_one_circuit():
    """At the class of record's slots, 4 a boundary at buckets 0 and 16 (at TINY's caps of 4 prompt ids and 2 outputs), a
    window of three requests (two anchored at bucket 0, one at 16) and one of eight (four at each) emit the
    same circuit, the same descriptor digest (SHA-512 of its description's bytes) and the same blocks: eight class Programs,
    each slot's reading its lane of the scan's state row before its anchor (rows 0 and 16 of the 4-lane scan's 9,090 words)
    and making its entry of the scan's Programs' row."""
    eight = _circuit(EIGHT, cls=SLOTS, placed=[(0, 1)] * 4 + [(16, 17)] * 4)
    three = _circuit(THREE, cls=SLOTS, placed=[(0, 1), (16, 18), (16, 31)])
    assert hashlib.sha512(_bytes(three)).hexdigest() == hashlib.sha512(_bytes(eight)).hexdigest()
    assert (three.ids, three.external, _blocks(three)) == (eight.ids, eight.external, _blocks(eight))
    assert (eight.blocks.slots, eight.blocks.steps) == (8, 5)
    n = {_label(fn): k for fn, k in eight.templates}
    scan = SCAN.replace("NS=4548", "NS=9090")
    assert (n[scan], n[ROW], n[FIRST], n[MASK], n[EMB], n[ENTRY]) == (8, 8, 8, 8 * SLOTS.positions, 8 * SLOTS.positions, 8)
    assert [(r.at, r.radix, r.base, r.step) for r in eight.registered.reads if r.param == SS.SCAN_STATE] == [
        (0, (4, 2), 0, (0, 16 * 9090))]
    assert eight.registered.params[SS.SCAN_STATE] == 32 * 9090 and not eight.ingress


def test_the_class_of_record_takes_both_windows_to_its_one_class_circuit(monkeypatch):
    """The class of record (`CLASS_OF_RECORD`, network-accounting's pick for ASK-122: EVERY 16, 4 slots a boundary, 8 a
    32-bucket window) is 8 slots of 1,024 prompt ids and 64 outputs, lane-major: slot 2j + b is boundary b's lane j at
    bucket 16b. A 3-request and an 8-request window are each taken to the one class circuit, every slot's Program, with the
    same arguments (its lowering is `test_the_class_of_record_lowers_within_the_grain_and_its_description_is_its_nodes`')."""
    assert (CLASS_OF_RECORD.slots, CLASS_OF_RECORD.lanes, CLASS_OF_RECORD.every, CLASS_OF_RECORD.positions) == (8, 4, 16, 1087)
    assert (CLASS_OF_RECORD.boundaries, CLASS_OF_RECORD.buckets) == ((0, 16), 32)
    assert [(CLASS_OF_RECORD.anchor(k), CLASS_OF_RECORD.lane(k)) for k in range(8)] == [(16 * (k % 2), k // 2) for k in range(8)]
    calls = []
    monkeypatch.setattr(WC, "class_circuit", lambda *a: calls.append(a[1:]) or "the class circuit")
    reg = S._registered()
    for window, placed in ((THREE, [(0, 1), (0, 2), (16, 31)]), (EIGHT, [(0, 1)] * 4 + [(16, 17)] * 4)):
        assert window_circuit(reg, S.compile_steps(reg, *S._session(*window)), CLASS_OF_RECORD, placed=placed) == "the class circuit"
    assert calls == [(CLASS_OF_RECORD, None, WC._GRAIN, WC.NAME)] * 2


def test_a_request_anchored_at_16_with_64_outputs_is_expressible(monkeypatch):
    """VLLM-126: a slot's Program hands all its outputs to the scan at its anchor bucket and the scan carries them across
    windows (`SC_GROUP`), so a request at the class of record's second boundary (bucket 16) keeping all 64 outputs, its last
    at bucket 16 + D + 64 / r = 59 past the 32-bucket window's end, is taken to the one class circuit."""
    calls = []
    monkeypatch.setattr(WC, "class_circuit", lambda *a: calls.append(a[1:]) or "the class circuit")
    reg = S._registered()
    window = ([(1, 64, {**S.GREEDY, "max_tokens": 64})], [[(0, 0, 1)]] + [[(0, p, 1)] for p in range(1, 64)])
    compiled = S.compile_steps(reg, *S._session(*window))
    assert [(u.LP, u.T + 1) for u in compiled.requests if u is not None] == [(1, 64)]
    assert window_circuit(reg, compiled, CLASS_OF_RECORD, placed=[(16, 59)]) == "the class circuit"
    assert calls == [(CLASS_OF_RECORD, None, WC._GRAIN, WC.NAME)]


def test_the_description_grows_with_the_class_programs_nodes_not_its_slots():
    """Each allocation's slots sit side by side in its template, so every slot's copy of a wire or read pattern is one: the
    class of record's slots at 2 and at 8 Programs have the same 151 wires besides the scan read's, 64 registered reads (63
    of the weights, one of the scan state), root reads, openings and 83 inference entries, the 8 Programs 4 times the units.
    The scan read's wires are 2 a lane its Programs reach (its row and its head; lane-major, so a lane's slots follow on):
    2 at 2 Programs, both lane 0's, and 8 at 8, whatever the boundaries."""
    reg = S._registered()
    two, eight = class_circuit(reg, SLOTS, 2, GRAIN), class_circuit(reg, SLOTS, None, GRAIN)
    shape = [(len(c.wires), len(c.registered.reads), len(c.registered.roots), len(c.registered.openings),
              len(c.blocks.inference)) for c in (two, eight)]
    assert shape == [(153, 64, 2, 1, 83), (159, 64, 2, 1, 83)] and two.ids == eight.ids
    scan = IF.descriptor_id(next(fn for fn, _ in eight.templates if fn.name == "ScanSlots"))
    assert [sum(w.writer == scan for w in c.wires) for c in (two, eight)] == [2, 2 * SLOTS.lanes]
    assert sum(n for _, n in eight.templates) == 4 * sum(n for _, n in two.templates) == 10192
    assert [[(r.at, r.radix, r.base, r.step) for r in c.registered.reads if r.param == SS.SCAN_STATE] for c in (two, eight)] == [
        [(0, (2,), 0, (16 * 9090,))], [(0, (4, 2), 0, (0, 16 * 9090))]]


def test_the_circuit_is_a_class_program_a_slot_in_units_by_template_and_wire_patterns(window):
    """Each slot is the class Program, its 5 positions each the whole stack, the LM head and the token select: TINY's GEMMs
    sliced 8 columns a unit, its RoPE a unit a head, its attention a unit a head a Call of `AttentionHead_v19`'s body (position
    p's first key block of p + 1 keys, `AttnBlockFirst_v1{NVIS=p + 1}`, and its end), its token select four chunks and a
    combine, positions 0..3's tokens `TokenFeed_v1` units, its row a `SlotRow_v1` of its lane of the scan state row
    (`ScanSlots_v1`), each position's select masked (`SlotMask_v1`) and collected (`OutCollect_v1`), and its last position's
    `SlotEntry_v1`; a port read from several writers is cut, its groups gathered by `AllGather_v1` units, and the wiring is a
    pattern per run, the two slots' one, not a wire per unit, but for the scan read's two a lane. The external rows are the
    scan state rows the slots read and their entries."""
    circuit, ids = window
    label = {tid: name for name, tid in ids.items()}
    assert {label[IF.descriptor_id(fn)]: n for fn, n in circuit.templates} == {
        EMB: 10, FIRST: 2, FEED: 6, SCAN: 2, ROW: 2, MASK: 10, COLLECT0: 2, COLLECT: 8, ENTRY: 2,
        "RMSNormTriton_v5{N=64}": 10, "Gemm_v2{K=64,N=8}": 1440,
        "RoPEHead_v1{D=16}": 120, **{f"AttnBlockFirst_v1{{D=16,NVIS={t},BN=128}}": 16 for t in range(1, 6)},
        "AttnHeadEnd_v1{D=16}": 80, "RMSNormFusedCuda_v2{N=64}": 40, "SiluMul_v2{I=128}": 20, "Gemm_v2{K=128,N=4}": 320, "TokenSelectChunk_v2{C=64}": 40,
        COMBINE: 10, "Gemm_v2{K=64,N=1}": 10, "AllGather_v1{WORLD=2,N=8}": 40, "AllGather_v1{WORLD=2,N=16}": 16,
        "AllGather_v1{WORLD=3,N=16}": 16, "AllGather_v1{WORLD=4,N=16}": 36, "AllGather_v1{WORLD=5,N=16}": 16,
        "AllGather_v1{WORLD=8,N=8}": 120, "AllGather_v1{WORLD=4,N=1}": 10, "AllGather_v1{WORLD=4,N=8}": 40,
        "AllGather_v1{WORLD=8,N=4}": 40}
    assert {label[t]: c for t, c in circuit.cuts.items()} == {
        "RMSNormFusedCuda_v2{N=64}": {"x": 2}, "RoPEHead_v1{D=16}": {"x": 2}, "SiluMul_v2{I=128}": {"gu": 4},
        COMBINE: {"idx": 4}, **{g: {"parts": int(g.split("WORLD=")[1].split(",")[0])} for g in ids if g.startswith("AllGather")}}
    family = {tid: name.split("{")[0] for name, tid in ids.items()}
    assert Counter((family[w.reader], w.port, family[w.writer], w.out) for w in circuit.wires) == {
        ("AllGather_v1", "parts", "AllGather_v1", "o0"): 8, ("AllGather_v1", "parts", "AttnHeadEnd_v1", "o0"): 1,
        ("AllGather_v1", "parts", "Gemm_v2", "o0"): 5, ("AllGather_v1", "parts", "RoPEHead_v1", "o0"): 8,
        ("AllGather_v1", "parts", "TokenSelectChunk_v2", "o0"): 1, ("AttnBlockFirst_v1", "k", "AllGather_v1", "o0"): 4,
        ("AttnBlockFirst_v1", "k", "RoPEHead_v1", "o0"): 1, ("AttnBlockFirst_v1", "q", "RoPEHead_v1", "o0"): 5,
        ("AttnBlockFirst_v1", "v", "AllGather_v1", "o0"): 5, ("AttnHeadEnd_v1", "l", "AttnBlockFirst_v1", "o1"): 5,
        ("AttnHeadEnd_v1", "o", "AttnBlockFirst_v1", "o2"): 5, ("EmbeddingRead_v1", "tok", "TokenFeed_v1", "o0"): 2,
        ("EmbeddingRead_v1", "tok", "TokenSelectCombine_v2", "o0"): 1, ("Gemm_v2", "x", "AllGather_v1", "o0"): 10,
        ("Gemm_v2", "x", "RMSNormFusedCuda_v2", "o0"): 16, ("Gemm_v2", "x", "RMSNormTriton_v5", "o0"): 5,
        ("Gemm_v2", "x", "SiluMul_v2", "o0"): 1, ("RMSNormFusedCuda_v2", "res", "EmbeddingRead_v1", "o0"): 5,
        ("RMSNormFusedCuda_v2", "res", "RMSNormFusedCuda_v2", "o1"): 5, ("RMSNormFusedCuda_v2", "x", "AllGather_v1", "o0"): 20,
        ("RMSNormTriton_v5", "x", "EmbeddingRead_v1", "o0"): 1, ("RoPEHead_v1", "x", "Gemm_v2", "o0"): 20,
        ("SiluMul_v2", "gu", "AllGather_v1", "o0"): 1, ("SlotMask_v1", "occ", "SlotRow_v1", "o0"): 1,
        ("SlotMask_v1", "tok", "TokenSelectCombine_v2", "o0"): 1, ("TokenFeed_v1", "sel", "TokenSelectCombine_v2", "o0"): 1,
        ("TokenFeed_v1", "slot", "SlotRow_v1", "o1"): 1, ("TokenFeed_v1", "slot", "TokenFeed_v1", "o1"): 2,
        ("TokenSelectChunk_v2", "x", "AllGather_v1", "o0"): 1, ("TokenSelectCombine_v2", "best", "AllGather_v1", "o0"): 1,
        ("TokenSelectCombine_v2", "idx", "TokenSelectChunk_v2", "o1"): 1, ("TokenSelectCombine_v2", "x0", "Gemm_v2", "o0"): 1,
        ("SlotRow_v1", "row", "ScanSlots_v1", "o0"): 1, ("SlotRow_v1", "row", "ScanSlots_v1", "o2"): 1,
        ("OutCollect_v1", "head", "ScanSlots_v1", "o1"): 1, ("OutCollect_v1", "head", "ScanSlots_v1", "o3"): 1,
        ("OutCollect_v1", "state", "OutCollect_v1", "o0"): 2, ("OutCollect_v1", "tok", "SlotMask_v1", "o0"): 2,
        ("SlotEntry_v1", "occ", "SlotRow_v1", "o0"): 1, ("SlotEntry_v1", "state", "OutCollect_v1", "o0"): 1}
    assert circuit.external == (f"{ids[SCAN]}/state", f"{ids[ENTRY]}/o0")


def test_a_positions_token_is_its_feed_below_the_prompt_cap_and_the_previous_select_past_it(window):
    """Each slot's row is its lane's of the scan state row before its anchor (both slots at bucket 0, lanes 0 and 1, row 0 of
    the registered scan state), read by its `ScanSlots_v1` and opened by its `SlotRow_v1` into the slot its first position's
    feed reads; each later feed reads its predecessor's slot and the previous position's select, and position p's embedding
    reads its feed's token for p < 4 and position p - 1's select at p = 4. A template's units are a position's slots side by
    side (slot k's position p at 2p + k)."""
    circuit, ids = window
    first, feed, emb, comb, row, scan = ids[FIRST], ids[FEED], ids[EMB], ids[COMBINE], ids[ROW], ids[SCAN]
    assert not circuit.ingress and [r.doc() for r in circuit.registered.reads if r.param == SS.SCAN_STATE] == [
        {"reader": scan, "port": "state", "param": SS.SCAN_STATE, "at": 0, "radix": [2], "base": 0, "step": [0]}]
    assert [_source(circuit, row, "row", u) for u in (0, 1)] == [(scan, 0, "o0"), (scan, 1, "o2")]
    assert [_source(circuit, first, "slot", u) for u in (0, 1)] == [(row, 0, "o1"), (row, 1, "o1")]
    assert [_source(circuit, feed, "slot", u) for u in range(6)] == [(first, 0, "o1"), (first, 1, "o1")] + [
        (feed, u, "o1") for u in range(4)]
    assert [_source(circuit, feed, "sel", u) for u in range(6)] == [(comb, u, "o0") for u in range(6)]
    assert [_source(circuit, emb, "tok", u) for u in range(10)] == [(first, 0, "o0"), (first, 1, "o0")] + [
        (feed, u, "o0") for u in range(6)] + [(comb, 6, "o0"), (comb, 7, "o0")]


def test_a_slots_outputs_are_its_masked_selects_collected_into_its_entry(window):
    """Position p of slot k's select is masked by its slot's occupancy word (`SlotMask_v1`, `SlotRow_v1`'s ``o0``) and
    collected (`OutCollect_v1`: the first position's from its lane's scan head, every later one's from its predecessor's);
    its last position's `SlotEntry_v1` reads the last collector and the same occupancy word, so an empty slot's entry is zero.
    The entries, not the selects or the masked outputs, are the circuit's external rows."""
    circuit, ids = window
    mask, comb, row, scan = ids[MASK], ids[COMBINE], ids[ROW], ids[SCAN]
    first, rest, entry = ids[COLLECT0], ids[COLLECT], ids[ENTRY]
    assert [_source(circuit, mask, "occ", u) for u in range(10)] == [(row, u % 2, "o0") for u in range(10)]
    assert [_source(circuit, mask, "tok", u) for u in range(10)] == [(comb, u, "o0") for u in range(10)]
    assert [_source(circuit, first, "head", u) for u in (0, 1)] == [(scan, 0, "o1"), (scan, 1, "o3")]
    assert [_source(circuit, first, "tok", u) for u in (0, 1)] == [(mask, 0, "o0"), (mask, 1, "o0")]
    assert [_source(circuit, rest, "tok", u) for u in range(8)] == [(mask, 2 + u, "o0") for u in range(8)]
    assert [_source(circuit, rest, "state", u) for u in range(8)] == [(first, 0, "o0"), (first, 1, "o0")] + [
        (rest, u, "o0") for u in range(6)]
    assert [_source(circuit, entry, "occ", u) for u in (0, 1)] == [(row, 0, "o0"), (row, 1, "o0")]
    assert [_source(circuit, entry, "state", u) for u in (0, 1)] == [(rest, 6, "o0"), (rest, 7, "o0")]
    assert not {f"{comb}/o0", f"{mask}/o0"} & set(circuit.external) and f"{entry}/o0" in circuit.external


def test_a_slots_entry_stops_at_the_registered_models_stop_ids(window):
    """The entry's ``STOP`` is `stop_ids`, the registered model's config's ``eos_token_id`` (an id, a list of them or none):
    TINY declares none, Llama-3.2-1B one (128001), and a list is its set in order."""
    circuit, ids = window
    assert circuit.template(ids[ENTRY]).bindings["STOP"] == WC.stop_ids(S._registered()) == ()
    assert WC.stop_ids(S._registered({**S.TINY, "eos_token_id": 128001})) == (128001,)
    assert WC.stop_ids(S._registered({**S.TINY, "eos_token_id": [9, 3, 9]})) == (3, 9)


def test_the_scan_state_row_is_pr2s_form_and_its_wide_read_is_one_row_a_slot():
    """The scan state row a slot reads (`scan_state.layout`) is `ingress_scan`'s head and slots with ``occ``, a 0/1 word,
    before ``n`` (network-accounting's PR2 form): at the class of record (4 lanes, KO 64, w63's cells) 9,353 u16 words, lane
    j's fields 37,387 bits apart. `ScanSlots_v1` reads it whole, one registered row a slot, and gives every lane's
    `SlotRow_v1` row (2,052 words at 1,024 prompt ids) and head (4), of which a slot's two are wired: its unit is 1 + 2·4
    rows, 3 rows and 3 heads of them unread. A row packed by `layout` reads back each lane's fields, ids at `io_shell.position`."""
    scan = CLASS_OF_RECORD.scan
    lay, base = SS.layout(scan), [f for f, _ in IS.slot_layout(scan)]
    assert (SS.words(scan), scan.K0, scan.K, scan.IDB, scan.KO) == (9353, 12, 45, 17, 64)
    assert [k.split(".", 1)[1] for k in lay if k.startswith("0.")] == base[:base.index("n")] + ["occ"] + base[base.index("n"):]
    assert [lay[f"{j}.occ"][0] - lay["0.occ"][0] for j in range(4)] == [37387 * j for j in range(4)]
    bridge = SS.bridge(scan, CLASS_OF_RECORD.prompt)
    lay_ = IF.layout(bridge)
    assert lay_.rows == (("state", 9353),) and [w for _, w in lay_.outs] == [2052, 4] * 4
    assert sum(map(len, IF.unit_rows(bridge))) == 1 + 2 * 4
    small = SLOTS.scan
    lay, x = SS.layout(small), 0
    fields = [(1, 3, 2, [7, 0x1FFFF, 0, 9]), (0, 0, 0, [0] * 4), (1, 1, 2, [5, 0, 0, 0]), (0, 0, 0, [0] * 4)]
    for j, (occ, lp, m, ids) in enumerate(fields):
        x |= occ << lay[f"{j}.occ"][0] | lp << lay[f"{j}.fields"][0] | m << lay[f"{j}.max_tokens"][0]
        x |= sum(v << lay[f"{j}.ids"][0] + 17 * IO.position(small.K0, small.K, p) for p, v in enumerate(ids))
    row = [x >> 16 * k & 0xFFFF for k in range(SS.words(small))]
    b = SS.bridge(small, SLOTS.prompt).bindings
    got = IO.slots_reference(row, 4, 17, small.K0, small.K, b["OCC"], b["LP"], b["MAX"], b["IDS"])
    assert got == [([occ, lp, *ids], [lp, m]) for occ, lp, m, ids in fields]


def test_the_weights_and_the_embedding_are_registered_reads(window):
    """The weights record and the scan state are the registered parameters (the scan state the window's 16 buckets' rows of
    4,548 words): GEMM slices, norms and RoPE read their rows from the weights at public positions, and the embedding is
    `EmbeddingRead_v1`'s opening of the table over its first V·H words, keyed by ``tok`` and pinned by its verdict. With
    TINY's tied embeddings the LM head's first row, which the token select's ``x0`` unit reads, is the table's first row.
    Both slots' reads of a node are one pattern (RoPE's a slot digit of step 0)."""
    circuit, ids = window
    reg = circuit.registered
    assert reg.params == {"weights": 91472, SS.SCAN_STATE: 16 * 4548}
    assert Counter((r.reader, r.port) for r in reg.reads) == {
        (ids["Gemm_v2{K=64,N=8}"], "w"): 30, (ids["Gemm_v2{K=128,N=4}"], "w"): 1, (ids["Gemm_v2{K=64,N=1}"], "w"): 1,
        (ids["RMSNormFusedCuda_v2{N=64}"], "w"): 10, (ids["RMSNormTriton_v5{N=64}"], "w"): 1, (ids["RoPEHead_v1{D=16}"], "cs"): 20,
        (ids[SCAN], "state"): 1}
    assert [t.doc() for t in reg.tables] == [{"name": "embedding@0", "param": "weights", "offset": 0, "words": 64, "rows": 256}]
    assert sorted(r.doc().items() for r in reg.roots) == sorted(
        {"reader": ids[EMB], "port": p, "table": "embedding@0", "of": p, "at": 0, "units": 10}.items() for p in ("domain", "root"))
    assert [o.doc() for o in reg.openings] == [{"reader": ids[EMB], "table": "embedding@0", "key": "tok",
                                                "ports": ["row", "salt", "path"], "verdict": "o1", "at": 0, "units": 10}]
    assert [(r.at, r.radix, r.base, r.step) for r in reg.reads if r.reader == ids["Gemm_v2{K=64,N=1}"]] == [(0, (10,), 0, (0,))]
    rope = [(r.at, r.radix, r.base, r.step) for r in reg.reads if r.reader == ids["RoPEHead_v1{D=16}"]]
    assert rope[:4] == [(0, (2, 4), 90432, (0, 0)), (8, (2, 2), 90432, (0, 0)), (12, (2, 4), 90432, (0, 0)),
                        (20, (2, 2), 90432, (0, 0))]


def test_a_head_is_its_key_blocks_calls_over_its_kv_groups_gathers(window):
    """Position 4 (T 5) at layer 0: a head is `AttentionHead_v19`'s Calls, its one key block `AttnBlockFirst_v1{NVIS=5}` and
    its end `AttnHeadEnd_v1`, which reads the block's running sum and output. Slot 0's heads 0 and 1 share KV group 0, whose
    keys are one `AllGather_v1{5, 16}` row over the RoPE'd keys of positions 0..4 (RoPE units 24p + 8: a position's 24 are
    two layers of each slot's 4 query and 2 key heads, side by side) and whose values one over each position's value row
    (its two GEMM slices gathered at its position); slot 1's head 0 (unit 4) reads its own gathers over its own keys
    (24p + 10). Position 4's gathers are a layer's keys then values, each a slot's two groups side by side."""
    circuit, ids = window
    first, end = ids["AttnBlockFirst_v1{D=16,NVIS=5,BN=128}"], ids["AttnHeadEnd_v1{D=16}"]
    rope, gather5, pair = ids["RoPEHead_v1{D=16}"], ids["AllGather_v1{WORLD=5,N=16}"], ids["AllGather_v1{WORLD=2,N=8}"]
    assert _source(circuit, first, "k", 0) == _source(circuit, first, "k", 1) == (gather5, 0, "o0")
    assert _source(circuit, first, "k", 2) == (gather5, 1, "o0")
    assert [_source(circuit, gather5, "parts", 0, r) for r in range(5)] == [(rope, 24 * p + 8, "o0") for p in range(5)]
    assert _source(circuit, first, "q", 0) == (rope, 96, "o0")
    assert _source(circuit, first, "k", 4) == (gather5, 2, "o0") and _source(circuit, first, "q", 4) == (rope, 100, "o0")
    assert [_source(circuit, gather5, "parts", 2, r) for r in range(5)] == [(rope, 24 * p + 10, "o0") for p in range(5)]
    assert _source(circuit, first, "v", 0) == (gather5, 4, "o0")
    assert {_source(circuit, gather5, "parts", 4, r)[0] for r in range(5)} == {pair}
    assert [_source(circuit, end, p, 64 + 0) for p in ("l", "o")] == [(first, 0, "o1"), (first, 0, "o2")]
    assert [_source(circuit, end, p, 64 + 5) for p in ("l", "o")] == [(first, 5, "o1"), (first, 5, "o2")]


@pytest.fixture(scope="module")
def long():
    """One Program of 129 positions at TINY of 2,048 positions and a grain of 256 rows: position 128 (T 129) is the first
    past one key block of 128."""
    reg = S._registered({**S.TINY, "max_position_embeddings": 2048})
    circuit = class_circuit(reg, WindowClass(1, 128, 2), 1, Grain(gemm_words=512, rows=256, select=64))
    return circuit, {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}


@pytest.mark.slow
def test_past_one_key_block_a_head_reads_its_full_blocks_gather_made_once(long):
    """At T 129 (position 128) a head is its first block `AttnBlockFirst_v1{NVIS=1}` over key 128 (RoPE's row itself), the
    full block below it `AttnBlock_v5{CHECK=false}` (FA2's one-row decode guards only the first block it visits) over keys
    0..127, reading the first block's m, l and o, and its end, reading the full block's. The full block's keys are one
    `AllGather_v1{128, 16}` a KV group, the same row position 127's `AttnBlockFirst_v1{NVIS=128}` read when the block filled:
    a layer's 8 such gathers (2 layers, keys and values, 2 groups), no gather of 129, and no unit past the grain's rows."""
    circuit, ids = long
    n = {_label(fn): k for fn, k in circuit.templates}
    block, gather = "AttnBlock_v5{D=16,NVIS=128,FIRST=false,BN=128,CHECK=false}", "AllGather_v1{WORLD=128,N=16}"
    assert (n[block], n[gather], n["AttnBlockFirst_v1{D=16,NVIS=1,BN=128}"], n["AttnHeadEnd_v1{D=16}"]) == (8, 8, 16, 129 * 8)
    assert not [lab for lab in n if lab.startswith("AllGather_v1{WORLD=129") or "CHECK=true" in lab]
    first, full, end = ids["AttnBlockFirst_v1{D=16,NVIS=1,BN=128}"], ids[block], ids["AttnHeadEnd_v1{D=16}"]
    rope, filled = ids["RoPEHead_v1{D=16}"], ids["AttnBlockFirst_v1{D=16,NVIS=128,BN=128}"]
    assert _source(circuit, first, "k", 8) == (rope, 12 * 128 + 4, "o0") and _source(circuit, first, "q", 8) == (rope, 12 * 128, "o0")
    assert _source(circuit, full, "k", 0) == _source(circuit, full, "k", 1) == _source(circuit, filled, "k", 0) == (ids[gather], 0, "o0")
    assert [_source(circuit, ids[gather], "parts", 0, r) for r in (0, 1, 127)] == [(rope, 12 * p + 4, "o0") for p in (0, 1, 127)]
    assert _source(circuit, full, "q", 0) == (rope, 12 * 128, "o0")
    assert [_source(circuit, full, p, 0) for p in ("m_prev", "l_prev", "o_prev")] == [(first, 8, o) for o in ("o0", "o1", "o2")]
    assert [_source(circuit, end, p, 8 * 128) for p in ("l", "o")] == [(full, 0, "o1"), (full, 0, "o2")]
    assert max(_unit_rows(circuit).values()) == 129 and IF.input_count(circuit.description(), pins=PINS)["free"] == 0


@pytest.mark.slow
def test_the_class_of_record_lowers_within_the_grain_and_its_description_is_its_nodes(monkeypatch):
    """The class of record itself (8 slots of 1,087 positions, a head over up to 1,087 keys: 8 full key blocks and a first
    of 63) lowers at TINY's geometry and a grain of 256 rows: no unit past the grain's rows (the largest a full block's
    gather, its 128 parts and its output), every input row wired or registered, and its 8 Programs' 8 x 297,800 units (4
    times 2 Programs') over 2 Programs' 33,284 wires besides the scan read's (2 a lane its Programs reach), 13,048 registered
    reads and 13,066 inference entries, which grow with the class Program's 26,088 nodes (its heads' Calls among them), not
    with its units or its slots. The baseline is 2
    Programs, as in `test_the_description_grows_with_the_class_programs_nodes_not_its_slots`: one Program's entries have a
    slot digit of 1, under which a step's unlike allocations run together (2,194 entries here), and from 2 on every
    allocation's slots are one entry's slot digit, so the entries are one list at any slot count. About a minute and a half
    and 3 GB: the class of record's own lowering, which no cheaper class reaches (each full block's gather read by every
    later position, the 4,088 full-block Calls a head)."""
    reg = S._registered({**S.TINY, "max_position_embeddings": 2048})
    grain, built, program = Grain(gemm_words=512, rows=256, select=64), WC.class_program(reg, CLASS_OF_RECORD), WC.class_program
    monkeypatch.setattr(WC, "class_program", lambda r, cls: built if cls == CLASS_OF_RECORD else program(r, cls))
    two, circuit = class_circuit(reg, CLASS_OF_RECORD, 2, grain), class_circuit(reg, CLASS_OF_RECORD, None, grain)
    n = {_label(fn): k for fn, k in circuit.templates}
    assert (n["AttnBlock_v5{D=16,NVIS=128,FIRST=false,BN=128,CHECK=false}"], n["AllGather_v1{WORLD=128,N=16}"]) == (
        8 * 2 * 4 * 4088, 8 * 8 * 8)
    assert max(_unit_rows(circuit).values()) == 129
    assert sum(n.values()) == 4 * sum(k for _, k in two.templates) == 8 * 297_800
    shape = [(len(c.wires), len(c.registered.reads), len(c.blocks.inference)) for c in (two, circuit)]
    assert shape == [(33286, 13048, 13066), (33292, 13048, 13066)] and len(built.program.fn.body.nodes) == 26088
    assert IF.input_count(circuit.description(), pins=PINS)["free"] == 0


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
    allocation's units an entry of its slots' runs side by side, like entries merged. Position p's embedding, select, mask
    and collector are step p, its feed (p = 1..3) step p, a slot's scan read and row step 0 and its entry step 4, a head's
    Calls over T keys step T - 1 (their digits layer, slot, head; the ends one template over every step; the key and value
    gathers' a layer's keys or values, the slot, the KV group), and the entries tile every template.
    Every wire's writer is of its reader's slot at its step or before: a slot reads only its own units, a position only its
    own and earlier positions'."""
    circuit, ids = window
    label = {tid: name for name, tid in ids.items()}
    assert (circuit.blocks.slots, circuit.blocks.steps, len(circuit.blocks.inference)) == (2, 5, 83)
    by: dict[str, list] = {}
    for e in circuit.blocks.inference:
        by.setdefault(label[e.template], []).append((e.at, e.radix, e.axes, e.step))
    assert set(by) == set(ids)
    assert by[EMB] == by[COMBINE] == by[MASK] == [(0, (5, 2), ("step", "slot"), 0)]
    assert by[FIRST] == by[SCAN] == by[ROW] == by[COLLECT0] == [(0, (2,), ("slot",), 0)]
    assert (by[FEED], by[COLLECT], by[ENTRY]) == ([(0, (3, 2), ("step", "slot"), 1)], [(0, (4, 2), ("step", "slot"), 1)],
                                                  [(0, (2,), ("slot",), 4)])
    att = "AttnBlockFirst_v1{{D=16,NVIS={},BN=128}}"
    assert by[att.format(5)] == [(0, (2, 2, 4), ("-", "slot", "-"), 4)]
    assert by["AttnHeadEnd_v1{D=16}"] == [(0, (5, 2, 2, 4), ("step", "-", "slot", "-"), 0)]
    assert by["AllGather_v1{WORLD=5,N=16}"] == [(0, (4, 2, 2), ("-", "slot", "-"), 4)]
    assert [{s for *_, s in by[att.format(t)]} for t in range(1, 6)] == [{t} for t in range(5)]
    assert by["RoPEHead_v1{D=16}"][:2] == [(0, (2, 4), ("slot", "-"), 0), (8, (2, 2), ("slot", "-"), 0)]
    labels = _labels(circuit)
    assert all((lab >= 0).all() for lab in labels.values())
    for w in circuit.wires:
        i = np.arange(w.units, dtype=np.int64)
        reader, writer = labels[w.reader][w.at + i // w.cut[1]], labels[w.writer][w.source(i)]
        assert (reader[:, 0] == writer[:, 0]).all() and (writer[:, 1] <= reader[:, 1]).all(), w.doc()


def test_inference_entries_are_the_slots_runs_side_by_side():
    """`_inference` takes a template's runs in unit order (a slot's are allocated before the next slot's, each allocation's
    slots side by side), merges its units of one (slot, step) into runs, takes each ``slots`` runs (slots 0, 1, ... of one
    length at one step) as an entry, and merges like entries that follow on at one step under a ``-`` digit, then at
    consecutive steps under a ``step`` digit; units that aren't such runs are refused by name."""
    assert WC._inference({"t": [(0, 1, 0, 0), (1, 1, 0, 0), (2, 2, 1, 0), (4, 2, 0, 1), (6, 2, 1, 1)]}, 2) == [
        IF.InferenceEntry("t", 0, (2, 2, 2), ("step", "slot", "-"), 0)]
    for spans in ([(0, 1, 0, 3), (1, 1, 1, 3), (2, 1, 0, 3), (3, 1, 1, 3)], [(0, 1, 0, 3), (2, 1, 0, 3), (1, 1, 1, 3), (3, 1, 1, 3)]):
        assert WC._inference({"t": spans}, 2) == [IF.InferenceEntry("t", 0, (2, 2), ("-", "slot"), 3)]
    with pytest.raises(Unexpressible, match=r"t: units from 0 are not 2 slots' runs of one length at one step"):
        WC._inference({"t": [(0, 2, 0, 0), (2, 1, 1, 0)]}, 2)


def test_every_input_row_is_wired_or_registered_counted_from_the_description(window):
    """`interface.input_count` over the circuit's description (JSON, no Program lowered) at the verifier's pins, the weights
    record's root, fixed before the run, and the scan state's, which the network certifier's scan proofs hold, finds no free
    input row and no ingress, so a proof of the window's units leaves nothing but the declared advice to the prover. The
    registered reads, table reads and openings are of the weights (the GEMM slices', norms' and RoPE tables' rows; the
    embedding table's root, domain and rows) but for the two slots' scan state rows (4,548 words each): with the weights'
    pin alone those are free, and with no pin all of them are."""
    circuit, _ = window
    doc, reg = json.loads(json.dumps(circuit.description())), circuit.registered
    assert {r.param for r in reg.reads} | {t.param for t in reg.tables} == set(reg.params) == PINS
    scan = 2 * 4548 * 16
    count = {"units": 2548, "inputs": 17572128, "wired": 2839712, "registered": 14586880 + scan, "ingress": 0, "free": 0}
    assert IF.input_count(doc, pins=PINS) == count
    assert IF.input_count(doc, pins={WC.WEIGHTS}) == {**count, "registered": 14586880, "free": scan}
    assert IF.input_count(doc) == {**count, "registered": 0, "free": 14586880 + scan}


def test_what_the_class_doesnt_take_is_refused_by_name():
    """A window past its class (three requests in two slots, a prompt past the cap, more outputs than the cap), a stochastic
    request (its Program would read a seed) and a gather past the grain's rows (position 0's token select reads a chunk of 64
    logits from 8 LM-head slices) are refused, and so is a Program count the class has no slots for."""
    reg = S._registered()

    def refused(window, cls=CLASS, grain=GRAIN):
        return lambda: window_circuit(reg, S.compile_steps(reg, *S._session(*window)), cls, grain)

    three = ([(2, 2, G2)] * 3, [[(0, 0, 2), (1, 0, 2), (2, 0, 2)], [(0, 2, 1), (1, 2, 1), (2, 2, 1)]])
    long = ([(5, 2, G2)], [[(0, 0, 5)], [(0, 5, 1)]])
    many = ([(2, 3, {**S.GREEDY, "max_tokens": 3})], [[(0, 0, 2)], [(0, 2, 1)], [(0, 3, 1)]])
    stoch = ([(4, 2, {**S.STOCH, "max_tokens": 2})], [[(0, 0, 4)], [(0, 4, 1)]])
    for window, match in ((three, "3 requests in a window of a class of 2 slots"), (long, "5 prompt ids and 2 outputs, past"),
                          (many, "2 prompt ids and 3 outputs, past its class's 4 and 2"), (stoch, "a stochastic request")):
        with pytest.raises(Unexpressible, match=match):
            refused(window)()
    with pytest.raises(Unexpressible, match=r"a gather of 8 rows of \[8\] words: rows of one width, at most 2 \(with its"):
        refused(WINDOW, grain=Grain(gemm_words=512, rows=3, select=64))()
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
    assert _bytes(ok) == _bytes(class_circuit(S._registered(), cls, None, GRAIN))
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


def test_no_unit_holds_more_rows_in_and_out_than_the_grain():
    """C-Flock's composer gives each row a unit reads or writes an hm96 slot, so at Llama-3.2-1B's grain a unit fits its
    2^27-bit block with at most 256 rows in and out, and a gather of 256 parts (257 rows) is refused when staged. The cut
    counts a gather's output row: SiluMul's 1,024 gate/up slices are gathered 128 at a time (8 gathers, not 4 of 256), and
    the fused norm's 512 down slices 128 at a time (4, not 2 of 256), its 128 o-proj slices 32 at a time. TINY at 16 rows is
    the same case, 32 gate/up slices and 16 down: 8 at a time, where 16 a gather would make 17 rows. No unit of the circuit
    holds more than the grain's rows, and at the test grain's 9 the largest hold exactly 9."""
    assert WB._cut(("SiluMul_v2", "gu"), {1024: 16}, 255, 256) == 8
    assert WB._cut(("RMSNormFusedCuda_v2", "x"), {128: 16, 512: 16}, 252, 256) == 4
    circuit = _circuit(WINDOW, Grain(gemm_words=512, rows=16, select=64))
    label = {IF.descriptor_id(fn): _label(fn) for fn, _ in circuit.templates}
    assert {label[t]: c for t, c in circuit.cuts.items() if not label[t].startswith("AllGather")} == {
        "RMSNormFusedCuda_v2{N=64}": {"x": 2}, "RoPEHead_v1{D=16}": {"x": 2}, "SiluMul_v2{I=128}": {"gu": 4},
        COMBINE: {"idx": 4}}
    rows = _unit_rows(circuit)
    assert max(rows.values()) <= 16 and max(c["parts"] for c in circuit.cuts.values() if "parts" in c) == 8
    assert max(_unit_rows(_circuit(WINDOW)).values()) == GRAIN.rows


def test_a_token_select_is_its_function_at_the_grains_chunk():
    """At 128 logits a chunk, each of the circuit's ten token selects is two `TokenSelectChunk_v2{C=128}` units, each reading
    16 LM-head slices through one gather, and a `TokenSelectCombine_v2{V=256, C=128}`: the Program's own body (chunks of 64)
    at another C."""
    circuit = _circuit(WINDOW, Grain(gemm_words=512, rows=17, select=128))
    n = {_label(fn): k for fn, k in circuit.templates}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    assert (n["TokenSelectChunk_v2{C=128}"], n["TokenSelectCombine_v2{V=256,C=128}"]) == (20, 10)
    assert not [t for t in n if t.startswith(("TokenSelectChunk_v2{C=64}", "TokenSelectCombine_v2{V=256,C=64}"))]
    assert {_source(circuit, ids["TokenSelectChunk_v2{C=128}"], "x", u)[0] for u in range(20)} == {ids["AllGather_v1{WORLD=16,N=8}"]}
    assert IF.input_count(circuit.description(), pins=PINS)["free"] == 0
