"""verity_vllm.program.predict.window_rows: a window's rows, every unit of `window_circuit.lower`'s served circuit given
its output rows from its requests' values. Pinned on `test_window_circuit`'s TINY class (2 slots, 4 prompt ids, 2
outputs) serving `LANES`' two requests (5 and 4 positions) at its grain, TINY's bf16 checkpoint on the H200 (FA3's `Attention_v4`, the class 4,32,4's
attention; its exact-int8 window waits on its norm's reader, `test_window_circuit_int8`), with random weights registered
under a committer's key and a scan state whose row at bucket 0 carries the two slots' requests; and the network
attester's scan of its own traffic (`scan_state_rows.scan_state`), which starts each request in the slot its anchor and lane give, at
TINY's slots and the class of record's.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import torch
import pytest
from verity.circuits import supplied as SU
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.commitments.gates import embedding as E
from verity.commitments.periods import interface as IF
from verity.commitments.periods import rows as CS
from verity.protocols.proof.spot.prover import docket as DK
from verity.protocols.proof.spot.verifier.one_stage import docket as R, registered as CR
from verity.commitments import registered_value as RV
from verity.protocols.accounting.compute.pouw.verifier.audit import hm96_salt
from verity.protocols.accounting.network.prover import ingress_scan as IS
from verity.protocols.accounting.network.prover import scan_state_rows as SR
from verity.protocols.accounting.network.verifier import network_attester as W
from verity.protocols.accounting.network.verifier.id_form import id_request

from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import (
    CLASS,
    GRAIN,
    SPLIT,
    SPLIT_GRAIN,
    entry_of,
    state_row,
)
from tests.program.predict.test_window_circuit import SLOTS as SLOTS_CLASS
from tests.program.predict.test_window_circuit_fa3 import H200, _wired_in_slot_and_step
from verity_vllm.check.replay.checks import check_rows
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.definitions.registry import io_shell as IO
from verity_vllm.program.instances_file import read_instances
from verity_vllm.program.predict import inference_blocks as IB
from verity_vllm.program.predict import scan_state as SS
from verity_vllm.program.predict import window_builder as WB
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_plan as WP
from verity_vllm.program.predict import window_rows as WR
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.program.predict.naming import write_build
from verity_vllm.program.predict.steps import compile_steps
from verity_vllm.program.predict.window_lowering import GROUP

PROGRAM = "test/model"
DESIGN = {"law": "stratified", "k": 2}
RUN = "ab" * 64
#: the lanes of the scan state row both slots read (bucket 0's), each (occupancy, prompt length, max_tokens, ids): slot 0 a
#: prompt of 4 (its first output from position 3's select), slot 1 a prompt of 3, each keeping 2 outputs
LANES = [(1, 4, 2, [17, 3, 250, 9]), (1, 3, 2, [5, 77, 42, 0])]
#: each slot's row, its lane's (occupancy, prompt length, ids)
SLOTS = tuple([occ, lp, *ids] for occ, lp, _m, ids in LANES)
#: the window's scan state rows, a bucket each: bucket 0's the slots' lanes, every later one's another request in each lane
STATE = [state_row(CLASS.scan, LANES)] + [state_row(CLASS.scan, [(1, 1, 1, [99, 0, 0, 0])] * 2)] * (CLASS.buckets - 1)
#: the window's requests as its circuit lowers them, a lane each of bucket 0
REQUESTS = tuple(WC.Request(CLASS.anchor(k), CLASS.lane(k), lp, m) for k, (_o, lp, m, _ids) in enumerate(LANES))
#: the nodes prefill computes at its last prompt position only: the final norm, the LM head and the token select
PRUNED = 3
#: the class Program's nodes a position
NODES = 24
EMBED = "model.embed_tokens.weight"
TABLE = WB.table_param(EMBED, 64)


def _bf16(x) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _weights(fn, seed: int = 3) -> dict[str, np.ndarray]:
    """Finite random weights, each field's elements in their unsigned width: s8 codes, small positive f32 scales (an
    attention's scales 1), norms near 1, the rotary cache in [-1, 1]."""
    rng, out = np.random.default_rng(seed), {}
    for label, ty in dict(fn.signature()[0])[WC.WEIGHTS].fields:
        (bits,), n = ty.width_set(), ty.leaves
        if bits == 8:
            out[label] = rng.integers(-127, 128, n).astype(np.int8).view(np.uint8)
        elif bits == 32:
            out[label] = (np.ones(1) if n == 1 else rng.uniform(5e-3, 2e-2, n)).astype(np.float32).view(np.uint32)
        elif "cos_sin" in label:
            out[label] = _bf16(rng.uniform(-1, 1, n))
        elif "norm" in label:
            out[label] = _bf16(1 + 0.1 * rng.standard_normal(n))
        else:
            out[label] = _bf16(0.5 * rng.standard_normal(n))
    return out


def _registered(config=S.TINY):
    """``config`` (TINY) at its bf16 checkpoint on the H200."""
    return S.Registered.of(S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": config}, target=H200), default_seed=7)


@dataclass(frozen=True)
class Unserved:
    """A request of ``LP`` prompt ids the engine computed none of, at ``positions``: every node of its `Slot` its row
    kernel's, the oracle `_served` gives a request's served rows from."""
    LP: int
    positions: int

    def __call__(self, *_: Any) -> None:
        return None


def unserved_slots(low, state, weights, cls=CLASS) -> list[WR.Slot]:
    """Each of ``low``'s Programs' `Slot` of its lane's row (`scan_slots`), every node computed (`Unserved`)."""
    return [WR.Slot(P.fn, cls, row, weights, Unserved(row[1], P.positions))
            for P, (row, _) in zip(low.programs, WR.scan_slots(low, state))]


def _window(config=S.TINY, grain=GRAIN, cls=CLASS, lanes=LANES):
    """The served circuit of ``cls`` lowered (a request a lane of bucket 0's ``lanes``: `REQUESTS`), its weights, its
    registered values (each weight's parameter under fresh salts, and the scan state's, bucket 0's row ``lanes`` and
    every later one another request in each lane (`STATE`), committed as a stand-in for the network attester's, in
    ``row/v2`` rows as the scan's units commit their state rows, as `Rows` and `Window` take them) and each request's
    `Slot` of its lane's row (`scan_slots`), every node computed."""
    later = [(1, 1, 1, [99] + [0] * (len(lanes[0][3]) - 1))] * len(lanes)
    state_rows = [state_row(cls.scan, lanes)] + [state_row(cls.scan, later)] * (cls.buckets - 1)
    requests = tuple(WC.Request(cls.anchor(k), cls.lane(k), lp, m) for k, (_o, lp, m, _ids) in enumerate(lanes))
    low = WC.lower(_registered(config), cls, None, grain, requests=requests)
    fields = _weights(low.fn)
    state = RV.Registered.commit(PROGRAM, SS.SCAN_STATE, state_rows,
                                 [hm96_salt(bytes(32), 0, j) for j in range(len(state_rows))], schema=CR.SCHEMA_V2)
    values = {**WW.weight_values(_registered(config), fields, None, grain, keep_rows=True), SS.SCAN_STATE: state}

    def weights(label: str, lo: int, hi: int) -> np.ndarray:
        return fields[label][lo:hi]

    return low, fields, values, weights, unserved_slots(low, state_rows, weights, cls)


@pytest.fixture(scope="module")
def window():
    return _window()


def _position(fn, family: str) -> list[int]:
    """Each position's node of ``family`` (its first)."""
    out: dict[int, int]
    pos, out = WR.positions(fn), {}
    for j, n in enumerate(fn.body.nodes):
        if n.fn.name == family:
            out.setdefault(pos[j], j)
    return [out[p] for p in sorted(out)]


def _gt(a: int, b: int) -> bool:
    f = (np.array([a, b], dtype=np.uint32) << 16).view(np.float32)
    return bool(f[0] > f[1])


def _i32(xs) -> list[int]:
    return np.asarray(xs, dtype=np.uint16).view("<u4").tolist()


def test_a_slots_tokens_are_its_prompt_then_its_selects(window):
    """A slot's token at a position is its prompt id below its prompt length and the previous position's select past it, each
    `TokenFeed_v1` carrying the slot on; each request's Program is its own positions of 24 nodes (an embedding opens each),
    5 and 4."""
    low, *_, slots = window
    assert [len(P.fn.body.nodes) for P in low.programs] == [120, 96] and low.fn is low.programs[0].fn
    for s, row, n in zip(slots, SLOTS, (5, 4)):
        lp, sel = row[1], _position(s.fn, "TokenSelect")
        assert sorted(set(WR.positions(s.fn))) == list(range(n))
        picks = [int(s.values[j][0][0]) for j in sel]
        assert s.tokens == row[2:2 + lp] + picks[lp - 1:n - 1] and all(0 <= t < 256 for t in picks)
        assert s.slots[0] == row[1:] and s.carried[0] == [lp - 1, *row[3:], 0]
        assert s.source == [WR.COMPUTED] * 24 * n


def test_every_unit_holds_its_nodes_value_or_its_definitions_function(window):
    """Every unit is filled: an embedding read's row is the table's at its token and its opening (row, salt and path, as
    `gates.embedding.arguments` encodes them) opens to the table's root, its verdict 1; each token select's group units, ``x0``
    and combine give the node's token, a row of its collector; each feed (every position but the first below its prompt
    length) holds its token and carried slot; each slot's lane's scan unit its lane's first token and
    carried slot, occupancy word and head of the state row at its anchor (`io_shell.scan_reference`), its last collector
    (the last position's) its outputs from its prompt's last position on and its entry the read bit (its
    occupancy word's bit 0), the 2 outputs' count and ids; each head's first
    key block (`Attention_v20`'s cut) is evaluated, and its end's Definition on that block's state is the node's value, the
    row kernel's. With no served values every unit is computed or evaluated."""
    low, fields, tables, _w, slots = window
    rows = WR.Rows(low, slots, tables)
    assert {int(x) for v in rows.source.values() for x in v} == {WR.COMPUTED, WR.EVALUATED}
    table = tables[TABLE]
    root, dom = table.root, table.domain.domain_id
    for P, s, row in zip(low.programs, slots, SLOTS):
        for p, j in enumerate(_position(P.fn, "Embedding")):
            (_, _, r), = P.outs[j]
            t = s.tokens[p]
            assert rows.outs[r.tid][0][r.unit].tolist() == list(table.rows[t]) and rows.outs[r.tid][1][r.unit].tolist() == [1]
            ins = {q: rows.inputs[r.tid][q][r.unit] for q in ("row", "salt", "path")}
            salt, path = (np.ascontiguousarray(ins[q]).view("<u8").astype(">u8").tobytes() for q in ("salt", "path"))
            path = [path[k:k + 64] for k in range(0, len(path), 64)]
            assert E.reference(256, t, root, dom, ins["row"].tolist(), salt, path) == (list(table.rows[t]), 1)
            want = E.arguments(256, t, root, dom, ins["row"].tolist(), salt, path)
            for q, bits in zip(("row", "salt", "path"), want[3:]):
                w = 16 if q == "row" else 64
                e = np.ascontiguousarray(ins[q]).view(f"<u{w // 8}").astype(np.uint64)
                assert [(int(x) >> k) & 1 for x in e for k in range(w)] == bits
            feed = P.tokens[p]
            assert (feed is None) == (p == 0 or p >= P.L)
            if feed is not None:
                f = rows.outs[feed[0]]
                assert _i32(f[0][feed[1]]) == [t] and _i32(f[1][feed[1]]) == s.carried[p]
        for j, (units, x0) in P.selects.items():
            token = int(s.values[j][0][0])
            run, run_i = int(rows.outs[x0[0]][0][x0[1]][0]), 0
            for c, (ct, cu) in enumerate(units):
                bests, idxs = rows.outs[ct][0][cu].tolist(), _i32(rows.outs[ct][1][cu])
                assert len(bests) == len(idxs) == 64 // GROUP
                for g, (best, idx) in enumerate(zip(bests, idxs)):
                    if _gt(best, run):
                        run, run_i = best, c * 64 + g * GROUP + idx
            (_, _, r), = P.outs[j]
            assert run_i == token and _i32(rows.outs[r.tid][0][r.unit]) == [token]
        st, su = P.scan_unit
        lane = [[row[2]], s.carried[0], row[:1], [row[1], 2]]
        assert [_i32(rows.outs[st][k][su]) for k in range(4)] == lane
        tok, slot, occ, head = WR._lane(P.scan, STATE, P.anchor, IO.scan_reference)
        assert lane == [[tok], slot, [occ], head]
        lp, outs = row[1], [int(s.values[j][0][0]) for j in P.selects]
        (ct, cu), (et, eu) = P.collects[-1], P.entry_unit
        assert len(P.collects) == lp - 1 + CLASS.outputs and len(outs) == P.positions - lp + 1 == CLASS.outputs
        assert _i32(rows.outs[ct][0][cu]) == [outs[-1]]
        assert _i32(rows.outs[ct][1][cu]) == [-CLASS.outputs & 0xFFFFFFFF, 2, *outs]
        assert sum(int(w) << 16 * k for k, w in enumerate(rows.outs[et][0][eu])) == 1 | 2 << 1 | outs[0] << 33 | outs[1] << 50
        assert len(P.heads) == 2 * P.positions - (lp - 1)
        for j, (plan, calls) in P.heads.items():
            (first, _), (end, _) = plan
            (ft, fu), (et, eu) = calls
            assert first.name == "AttnBlockFirst" and end.name == "AttnHeadEnd" and rows.source[ft][fu] == WR.EVALUATED
            lw, ow = (np.ascontiguousarray(rows.outs[ft][k][fu:fu + 4]).view("<u4") for k in (1, 2))
            node = np.asarray(evaluate_batch(end, lw, ow)).reshape(4, -1).astype(np.uint16)
            assert np.array_equal(node, rows.outs[et][0][eu:eu + 4])
            assert np.array_equal(node.reshape(-1), WR._words(s.values[j])) and node.any()


def test_a_slots_row_is_its_lanes_of_the_scan_state_row_at_its_anchor(window):
    """Each slot's row and head are its lane's of the scan state row its lane's `ScanSlots_v5` reads, the one at its anchor
    (`scan_slots`; both of TINY's slots at bucket 0, lanes 0 and 1): a slot of another row than its lane's, a class
    circuit, slots of other Programs than the circuit's, or no scan state's value, are refused by name."""
    low, _f, values, weights, slots = window
    assert WR.scan_slots(low, STATE) == [(r, [r[1], 2]) for r in SLOTS]
    other = [*SLOTS[0][:-1], SLOTS[0][-1] + 1]
    with pytest.raises(ValueError, match="request 0's Program: its slot's row isn't lane 0's of the scan state row at bucket 0"):
        WR.Rows(low, [WR.Slot(low.fn, CLASS, other, weights, Unserved(4, 5)), slots[1]], values)
    with pytest.raises(ValueError, match="slot 0's values are of another Program than the circuit's slot 0"):
        WR.Rows(low, slots[::-1], values)
    with pytest.raises(ValueError, match="a class circuit's rows: the committer fills a served window's circuit"):
        WR.Rows(WC.lower(_registered(), CLASS, None, GRAIN), slots, values)
    with pytest.raises(ValueError, match="a Program of 5 positions for a request served at 4"):
        WR.Slot(low.fn, CLASS, SLOTS[0], weights, Unserved(4, 4))
    with pytest.raises(ValueError, match=re.escape(f"give the scan state's registered value, {SS.SCAN_STATE}, with its rows")):
        WR.Rows(low, slots, {p: v for p, v in values.items() if p != SS.SCAN_STATE})


def test_a_standin_scan_state_gives_each_slot_its_row_and_max_tokens(window):
    """Where no scan ran (`prove_path`'s window commit), `scan_state_rows.standin_rows` is the scan state rows that give each slot its row
    and ``max_tokens`` at its anchor and lane, every other field zero: bucket 0's row is `STATE`'s, which holds the slots'
    lanes alone. It is a per-window value in its reading port's rows, each row under 192 fresh bytes from the OS, so a
    second commit of the same rows opens the same rows under other salts and another root; `Rows` takes it as the scan
    state."""
    low, _f, values, _w, slots = window
    heads = [(r, m) for r, (_o, _lp, m, _ids) in zip(SLOTS, LANES)]
    v, again = (WR.state_value(PROGRAM, CLASS, SR.standin_rows(CLASS, heads), window=RUN) for _ in range(2))
    assert WR.scan_slots(low, v.rows) == WR.scan_slots(low, STATE) and v.rows[0] == tuple(STATE[0])
    assert (v.leaves, v.words, v.schema, v.window) == (CLASS.buckets, SS.words(CLASS.scan), CR.SCHEMA_V2, RUN)
    salts = [bytes(s) for s in (*v.salts, *again.salts)]
    assert list(again.rows) == list(v.rows) and len(set(salts)) == 2 * CLASS.buckets
    assert {len(s) for s in salts} == {192} and again.entry()["root"] != v.entry()["root"]
    ref = RV.Registered.commit(PROGRAM, SS.SCAN_STATE, list(v.rows), list(v.salts), schema=CR.SCHEMA_V2, window=RUN)
    assert ref.entry() == v.entry()
    WR.Rows(low, slots, {**values, SS.SCAN_STATE: v})


def _traffic(cls: WC.WindowClass, prompts, arrivals) -> tuple[list[bytes], dict[int, tuple[int, ...]]]:
    """Id requests (`id_form.id_request`, ``max_tokens`` 2) of ``prompts`` arriving at buckets ``arrivals``, request i
    answered ids 100 + i and 200 + i, through the network attester at ``cls``'s scan schedule for the transfer blocks of
    3 execution phases of its buckets (`scan_traffic.serve`, network accounting's tests' driver): its registered ingress rows and each request's outputs by
    ordinal."""
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("scan_traffic", Path(__file__).parents[5] / "verity" / "protocols" / "accounting" / "network" / "verifier" / "scan_traffic.py")
    assert spec is not None and spec.loader is not None
    ST = sys.modules["scan_traffic"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ST)
    scan = cls.scan
    params = W.NetworkAttesterParams(cell_bytes=scan.B, slots=scan.r, bucket_ms=100, delay=scan.D, lanes=scan.lanes,
                            queue=scan.lanes, window=cls.buckets, ingress_cells=scan.C, ingress_buckets=scan.WITHIN,
                            egress_cells=scan.KO + 1, every=scan.EVERY)
    reqs = [id_request(model="org/m", user=f"user-{i}", prompt=list(p), max_tokens=2, seed=i, temperature=0.0, top_p=1.0,
                       frame_bytes=scan.B, offset=0) for i, p in enumerate(prompts)]
    served = ST.serve(params, reqs, arrivals, [1 << 20] * len(reqs), [(100 + i, 200 + i) for i in range(len(reqs))], 3)
    return served.rows()[0], served.outputs


def test_the_scans_state_starts_each_request_at_its_anchor_in_its_lane():
    """`scan_state_rows.scan_state` on the network attester's traffic at the class of record's slots (`SLOTS_CLASS`: 4 lanes, boundaries
    at buckets 0 and 16 of a 32-bucket window, TINY's bounds): three requests admitted at bucket 0 are anchored at bucket 16
    (WITHIN 12 on), in lanes 0, 1 and 2, so window 0's class Programs 1, 3 and 5 start them (slot nb·j + b); a fourth
    admitted at 5 is anchored at 32 (`scan_state_rows.scan_anchors`), window 1's first boundary, in lane 3: window 1's Program 6. Each
    started slot's lane of the row at its anchor holds its prompt and every other none; the rows are the scan's, from the
    empty state's, and carry what the scan does past the slots' rows, so a stand-in (`scan_state_rows.standin_rows`) giving the same
    slots the same rows is another value. A prompt past the class's prompt bound (the scan's KI) and rows short of the
    window's last are refused."""
    prompts = ((17, 3, 250, 9), (5, 77, 42), (1, 2), (9, 8, 7, 6))
    ingress, outputs = _traffic(SLOTS_CLASS, prompts, (0, 0, 0, 5))
    scan = IS.Class(**SLOTS_CLASS.scan.shape)
    assert SR.scan_anchors(SLOTS_CLASS, ingress, outputs) == {1: 16, 2: 16, 3: 16, 4: 32}
    for w, want in ((0, [None, 1, None, 2, None, 3, None, None]), (1, [None] * 6 + [4, None])):
        rows, starts = SR.scan_state(SLOTS_CLASS, ingress, outputs, w * SLOTS_CLASS.buckets)
        assert starts == want and len(rows) == SLOTS_CLASS.buckets
        heads = []
        for k, q in enumerate(starts):
            p = prompts[q - 1] if q else ()
            assert IS.token_row(scan, IS.decode(scan, rows[SLOTS_CLASS.anchor(k)]))[SLOTS_CLASS.lane(k)] == \
                ((1, len(p), p) if q else (0, 0, ()))
            heads.append(([int(bool(q)), len(p), *p, *[0] * (SLOTS_CLASS.prompt - len(p))], 2 if q else 0))
        got = WR.state_value(PROGRAM, SLOTS_CLASS, rows, window=RUN)
        standin = WR.state_value(PROGRAM, SLOTS_CLASS, SR.standin_rows(SLOTS_CLASS, heads), window=RUN)
        assert got.rows != standin.rows and (got.leaves, got.words) == (standin.leaves, standin.words)
    assert rows[0] != IS.encode(scan, IS.empty(scan))
    assert SR.scan_state(SLOTS_CLASS, ingress, outputs, 0)[0][0] == IS.encode(scan, IS.empty(scan))
    with pytest.raises(ValueError, match="refuses the network attester's ingress rows"):
        SR.scan_state(WC.WindowClass(8, 3, 2, lanes=4, every=16), ingress, outputs)
    with pytest.raises(ValueError, match="give the ingress rows to it, 63, not 40"):
        SR.scan_state(SLOTS_CLASS, ingress[:40], outputs, SLOTS_CLASS.buckets)


def test_the_scans_state_value_gives_the_window_its_slots(window):
    """The window's scan state as the network attester's scan gives it: `LANES`' two requests, admitted at bucket 0 of
    TINY's 16-bucket windows, are anchored at bucket 16, window 1's one boundary, in lanes 0 and 1, so class Programs 0
    and 1 start them; the scan's rows, committed (`state_value`), give each slot its row and head as the stand-in's
    `STATE` does (`scan_slots`), and `Rows` takes the scan's value with the slots of those rows."""
    low, _f, values, _w, slots = window
    ingress, outputs = _traffic(CLASS, [ids[:lp] for _o, lp, _m, ids in LANES], (0, 0))
    rows, starts = SR.scan_state(CLASS, ingress, outputs, CLASS.buckets)
    assert starts == [1, 2]
    v = WR.state_value(PROGRAM, CLASS, rows, window=RUN)
    assert WR.scan_slots(low, v.rows) == WR.scan_slots(low, STATE) == [(r, [r[1], 2]) for r in SLOTS]
    WR.Rows(low, slots, {**values, SS.SCAN_STATE: v})


def test_a_full_windows_entry_j_is_the_request_the_scan_started_in_slot_j_whatever_the_programs_order():
    """A full window of two requests of 4 and 2 prompt ids, in either lane order, through the network attester's scan:
    TINY's window 1 starts the first in slot 0 and the second in slot 1. The served Programs are the longest first, and
    entry j (unit j of the window's one `SlotEntry_v1`) is slot j's Program's: at slot j's certified prompt length it
    counts the request's 2 outputs, and read as the other slot's, whose length differs, it is KO + 1, which the scan
    refuses. Swapping the lanes swaps the Programs' order and the entries' prompt lengths, not the entries' slots."""
    scan, prompts = IS.Class(**CLASS.scan.shape), ((17, 3, 250, 9), (5, 77))
    for lanes in (prompts, prompts[::-1]):
        ingress, outputs = _traffic(CLASS, lanes, (0, 0))
        rows, starts = SR.scan_state(CLASS, ingress, outputs, CLASS.buckets)
        assert starts == [1, 2]
        slots = IS.token_row(scan, IS.decode(scan, rows[0]))
        assert [s[:3] for s in slots] == [(1, len(p), p) for p in lanes]
        low = WC.lower(S._registered(), CLASS, None, GRAIN, requests=tuple(WC.Request(0, j, len(p), 2)
                                                                          for j, p in enumerate(lanes)))
        assert [p.positions for p in low.programs] == [5, 3]
        by_entry = sorted(range(2), key=lambda k: low.programs[k].entry_unit[1])
        assert [low.programs[k].lane for k in by_entry] == [0, 1] and [low.programs[k].L for k in by_entry] == [len(p) for p in lanes]
        for j, k in enumerate(by_entry):
            assert entry_of(low, k, slots[j][1], 2)[1] == 2
            assert entry_of(low, k, slots[1 - j][1], 2)[1] == CLASS.outputs + 1


def test_a_window_of_lanes_1_and_3_of_four_has_an_entry_a_lane_and_the_scan_takes_them():
    """Requests in lanes 1 and 3 of a class of four lanes: the window's one `SlotEntry_v1` has four units in lane order,
    lanes 0 and 2 each on its occupancy word and a collector of its own (`OutCollect_v1{FIRST}`), both in lane 1's
    Program's slot, so no wire crosses slots. Their entries read 0 and the occupied lanes' read 1 with their 2 outputs,
    and the network attester's scan step takes that Programs' row at the lanes' anchor, where it refuses the two
    occupied lanes' entries as the first two of the row."""
    cls = WC.WindowClass(4, 4, 2)
    lanes = [(0, 0, 0, [0] * 4), (1, 3, 2, [5, 77, 42, 0]), (0, 0, 0, [0] * 4), (1, 2, 2, [9, 8, 0, 0])]
    requests = tuple(WC.Request(cls.anchor(k), cls.lane(k), lp, m) for k, (occ, lp, m, _ids) in enumerate(lanes) if occ)
    reg = _registered()
    low = WC.lower(reg, cls, None, GRAIN, requests=requests)
    _wired_in_slot_and_step(low.circuit)
    assert [(s, k) for s, k, _c, _e in low.empty] == [(0, 0), (2, 0)] and low.programs[0].lane == 1
    rows_ = [state_row(cls.scan, lanes)] * cls.buckets
    fields = _weights(low.fn)
    state = RV.Registered.commit(PROGRAM, SS.SCAN_STATE, rows_, [hm96_salt(bytes(32), 0, j) for j in range(len(rows_))],
                                 schema=CR.SCHEMA_V2)
    values = {**WW.weight_values(reg, fields, None, GRAIN, keep_rows=True), SS.SCAN_STATE: state}
    slots = [WR.Slot(P.fn, cls, row, lambda label, lo, hi: fields[label][lo:hi], Unserved(row[1], P.positions))
             for P, (row, _) in zip(low.programs, WR.scan_slots(low, rows_))]
    rows = WR.Rows(low, slots, values)
    et = low.programs[0].entry_unit[0]
    assert rows.n[et] == 4 and [P.entry_unit[1] for P in low.programs] == [1, 3]
    scan = IS.Class(**cls.scan.shape)
    bits = [w >> j & 1 for e in rows.outs[et][0] for w in e for j in range(16)]
    got = IS.prog_decode(scan, bits)
    assert [(x, n) for x, n, _ids in got] == [(0, 0), (1, 2), (0, 0), (1, 2)]
    st = IS.decode(scan, [w >> j & 1 for w in rows_[0] for j in range(16)][:IS.state_bits(scan)])
    for j in (1, 3):
        st.slots[j] = replace(st.slots[j], live=1, done=1, q=j, anchor=st.now)
    st.last = 3
    row = bytes(scan.S * scan.B)
    assert IS.step(scan, row, st, got) is not None
    assert IS.step(scan, row, st, [got[1], got[3], IS.NONE, IS.NONE]) is None


def test_a_chunks_best_is_its_first_maximum_above_minus_infinity():
    """`TokenSelectChunk_v2`'s scan from (-inf, 0): a NaN is never taken (not even first), equal maxima keep the first, and a
    chunk of no number above -inf is (-inf, 0)."""
    nan, inf, ninf = 0x7FC0, 0x7F80, 0xFF80
    assert WR.chunk_best(_bf16([1.0, 3.0, 3.0, 2.0])) == (int(_bf16(3.0)), 1)
    assert WR.chunk_best(np.array([nan, *_bf16([-2.0, -1.0])], dtype=np.uint16)) == (int(_bf16(-1.0)), 2)
    assert WR.chunk_best(np.array([ninf, nan, ninf], dtype=np.uint16)) == (ninf, 0)
    assert WR.chunk_best(np.array([ninf, inf], dtype=np.uint16)) == (inf, 1)


def _served(low, s, T: int, pruned: int = PRUNED, also: tuple[int, ...] = (), whole: tuple[int, ...] = ()):
    """Slot ``s``'s values as a request's served rows (`Served`): its prompt's positions in prefill, but for the last
    ``pruned`` nodes of a position and the nodes ``also`` at its last prompt position only, and ``T`` decode steps, each
    member's elements by row under op ``op<k>``, op by op. A pruned node in ``whole`` is committed at every prompt
    position, as the engine computes it, though its Program's one row is the last position's."""
    nodes, pos, lp = low.fn.body.nodes, WR.positions(low.fn), s.row[1]
    K = len(nodes) // (max(pos) + 1)
    rows, store = [], {}
    for k in range(K):
        fields = WR._fields(nodes[k].fn)
        names = ["out"] if len(fields) == 1 else [str(i) for i in range(len(fields))]
        members = [(m, f.leaves, "x") for m, f in zip(names, fields)]
        prefill = [lp - 1] if k >= K - pruned or k in also else list(range(lp))
        for step, ps in [(0, prefill), *((d, [lp - 1 + d]) for d in range(1, T + 1))]:
            for r, p in enumerate(ps):
                rows.append({"spec": descriptor_id(nodes[p * K + k].fn), "step": step, "op_path": f"op{k}", "members": members,
                             "row": r})
            ps = list(range(lp)) if step == 0 and k in whole else ps
            for i, m in enumerate(names):
                store[(step, f"op{k}", m)] = np.concatenate([s.values[p * K + k][i] for p in ps])
    return rows, store


def _read(store):
    """`Served`'s ``read`` and ``bound`` over ``store``."""
    def read(step, op, m, lo, hi):
        a = store.get((step, op, m))
        return (None, f"no {op}/{m} at step {step}") if a is None else (a[lo:hi], "")

    def bound(step, op, m):
        a = store.get((step, op, m))
        return 0 if a is None else a.size
    return read, bound


def test_a_served_value_is_the_nodes_value_and_the_rest_are_computed(window):
    """A request's served rows give each node the engine computed its value: every prompt position's, the pruned nodes at
    the last prompt position's only, and its decode steps'; every other node's value is computed from them. A served value
    is taken as served (a changed row stays changed), an embedding that isn't the table's row and a select that isn't its
    logits' first maximum are refused by name, as is a row of another Definition."""
    low, _f, tables, weights, slots = window
    s0 = slots[0]
    rows, store = _served(low, s0, T=1)
    served = WR.Served(rows, *_read(store))
    assert served.LP == 4 and len(served.prefill) == NODES and len(served.decode) == 1
    got = WR.Slot(low.fn, CLASS, SLOTS[0], weights, served)
    K = NODES
    want = [WR.SERVED if p == 4 or k < K - PRUNED or p == 3 else WR.COMPUTED for p in range(5) for k in range(K)]
    assert got.source == want
    assert all(np.array_equal(a, b) for x, y in zip(got.values, s0.values) for a, b in zip(x, y)) and got.tokens == s0.tokens
    rows_ = WR.Rows(low, [got, slots[1]], tables)
    (_, _, r), = low.programs[0].outs[K + 1]
    assert rows_.source[r.tid][r.unit] == WR.SERVED

    norm = dict(store)
    norm[(0, "op1", "out")] = norm[(0, "op1", "out")] ^ np.uint16(1)
    changed = WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(rows, *_read(norm)))
    assert np.array_equal(changed.values[1][0], store[(0, "op1", "out")][:64] ^ np.uint16(1))

    bad = dict(store)
    bad[(0, "op0", "out")] = bad[(0, "op0", "out")] ^ np.uint16(1)
    with pytest.raises(ValueError, match="node 0 .* at position 0: served as .*not the table's row at its token"):
        WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(rows, *_read(bad)))
    pick = dict(store)
    pick[(1, f"op{K - 1}", "out")] = (pick[(1, f"op{K - 1}", "out")] + 1) % 256
    with pytest.raises(ValueError, match="at position 4: served as .*not the first maximum of its logits"):
        WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(rows, *_read(pick)))
    other = [{**d, "spec": "SiluMul_v2{I=128}"} if (d["op_path"], d["step"]) == ("op2", 1) else d for d in rows]
    with pytest.raises(ValueError, match="position 4's node 2 is .*served as SiluMul_v2"):
        WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(other, *_read(store)))
    with pytest.raises(ValueError, match="the served request's 4 prompt ids, its slot row's 3"):
        WR.Slot(low.fn, CLASS, SLOTS[1], weights, served)
    dropped = [d for d in rows if (d["op_path"], d["step"]) != ("op2", 1)]
    with pytest.raises(ValueError, match=r"step 1: none of the prefill's op \('op2', .*\) \(1 ops\)"):
        WR.Served(dropped, *_read(store))
    twice = rows + [{**d} for d in rows if (d["op_path"], d["step"]) == ("op2", 1)]
    with pytest.raises(ValueError, match=r"step 1: op \('op2', .*\) served at two rows"):
        WR.Served(twice, *_read(store))


def test_the_rows_print_their_units_by_source_and_their_readers_counts(window, capsys):
    """`Rows` prints one ``WINDOW-ROWS`` line: every unit by source, and the reads, failures and members not retained of
    the reader its served slots read through, summed once however many slots share it; ``opened`` is None with no reader."""
    from collections import Counter
    from types import SimpleNamespace

    low, _f, tables, weights, slots = window
    rows, store = _served(low, slots[0], T=1)
    reader = SimpleNamespace(stats=Counter(reads=7, leaves=9, failed=0, not_retained=2))
    got = WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(rows, *_read(store), reader))
    filled = WR.Rows(low, [got, slots[1]], tables)
    line, = [x for x in capsys.readouterr().err.splitlines() if x.startswith("WINDOW-ROWS ")]
    doc = json.loads(line.removeprefix("WINDOW-ROWS "))
    assert doc == filled.counts() and doc["opened"] == {"reads": 7, "failed": 0, "not_retained": 2}
    assert sum(doc["units"].values()) == sum(filled.n.values()) and doc["units"]["served"] > 0
    WR.Rows(low, slots, tables)
    assert json.loads(capsys.readouterr().err.split("WINDOW-ROWS ", 1)[1])["opened"] is None


def _located(store, prefix: str = ""):
    """`Served`'s ``locate`` over ``store``: a member's elements as its bytes, under ``prefix`` and its op and member's name."""
    from verity_vllm.check.replay.opening import Located

    def locate(step, op, m, lo, hi):
        a = store.get((step, op, m))
        if a is None:
            return None, f"no {op}/{m} at step {step}"
        return Located(step, f"{prefix}{op}/{m}", lo * a.itemsize, hi * a.itemsize, f"uint{8 * a.itemsize}"), ""
    return locate


def test_a_plan_is_the_rows_sources_and_feeds_each_served_row_from_where_it_is_committed(window):
    """`Plan`, from the lowering and each slot's served record with no value read, gives every unit the source `Rows` gives
    it, and each run of rows a served node fills is the committed bytes it names, cut at the port's row width: the words
    `Rows` puts there, which `stage` lays out from the retained blocks as the row hash reads them. A slot with no ``where``
    (a stand-in) is computed throughout, and a served record with no ``locate`` is refused by name."""
    low, _f, tables, weights, slots = window
    served, got, held = [], [], {}
    for k, s in enumerate(slots):
        rows, store = _served(low, s, T=max(WR.positions(low.programs[k].fn)) + 1 - s.row[1])
        held.update({(step, f"s{k}:{op}/{m}"): a for (step, op, m), a in store.items()})
        served.append(WR.Served(rows, *_read(store), locate=_located(store, f"s{k}:")))
        got.append(WR.Slot(low.programs[k].fn, CLASS, SLOTS[k], weights, served[-1]))
    filled = WR.Rows(low, got, tables)
    plan = WP.Plan(low, served)
    assert all(np.array_equal(plan.source[t], filled.source[t]) for t in filled.source)
    assert plan.describe()["units"] == filled.counts()["units"] and filled.counts()["units"]["served"] > 0
    fed, blocks = 0, {key: torch.from_numpy(np.ascontiguousarray(a).view(np.uint8).reshape(-1)) for key, a in held.items()}
    for f in plan.feeds:
        want = filled.outs[f.tid][int(f.out[1:])][f.unit:f.unit + f.rows]
        words = np.concatenate([np.frombuffer(held[(x.step, x.name)].view(np.uint8)[x.lo:x.hi].tobytes(), f"<u{int(x.dtype[4:]) // 8}")
                                .astype(f"<u{x.bits // 8}").view("<u2") for x in f.spans])
        assert np.array_equal(words, want.reshape(-1)), f
        rows = WP.stage(f, lambda step, name: (blocks[(step, name)], 0)).numpy()
        assert rows.shape == (f.rows, -(-2 * f.words // 16) * 16) and not rows[:, 2 * f.words:].any()
        assert np.array_equal(np.ascontiguousarray(rows[:, :2 * f.words]).view("<u2"), want.reshape(f.rows, f.words)), f
        fed += f.rows
    assert fed == plan.describe()["fed_rows"] > 0

    stand = WP.Plan(low, slots)
    assert not stand.feeds and all(np.array_equal(stand.source[t], WR.Rows(low, slots, tables).source[t]) for t in stand.source)
    with pytest.raises(ValueError, match="give its `locate`"):
        WP.Plan(low, [WR.Served(rows, *_read(store)) for rows, store in [_served(low, slots[0], T=1)]] + [slots[1]])


def test_a_served_prefill_is_read_by_op_however_its_rows_interleave(window):
    """A Build's prefill runs position by position and prunes a node that isn't the last of a position's (the last layer's
    query side, its keys and values kept for the cache), so an op's first row can come after a later node's: its ops are
    read in the order of its last prompt position's rows, the node order, and each decode step's rows by op in whatever
    order they come. A node pruned at an earlier prompt position is computed."""
    low, _f, _t, weights, slots = window
    s0, K = slots[0], NODES
    mid, lp = K - PRUNED - 2, s0.row[1]
    rows, store = _served(low, s0, T=1, also=(mid,))
    pruned = {mid, *range(K - PRUNED, K)}

    def at(d):
        k = int(d["op_path"][2:])
        return (lp - 1 if k in pruned else int(d["row"]), k)

    interleaved = sorted((d for d in rows if d["step"] == 0), key=at) + [d for d in rows if d["step"] == 1][::-1]
    assert [d["op_path"] for d in interleaved].index(f"op{mid}") > [d["op_path"] for d in interleaved].index(f"op{mid + 1}")
    served = WR.Served(interleaved, *_read(store))
    assert [v[0]["op_path"] for v in served.prefill] == [f"op{k}" for k in range(K)]
    assert [d["op_path"] for d in served.decode[0]] == [f"op{k}" for k in range(K)]
    got = WR.Slot(low.fn, CLASS, SLOTS[0], weights, served)
    assert got.source == [WR.COMPUTED if p < lp - 1 and k in pruned else WR.SERVED for p in range(5) for k in range(K)]
    assert all(np.array_equal(a, b) for x, y in zip(got.values, s0.values) for a, b in zip(x, y))


def test_a_pruned_node_committed_at_every_prompt_position_is_read_at_the_last(window):
    """The engine computes the last layer and the final norm at every prompt position, so the record binds them a row a
    position though the Program computes them at the last one only, while it gathers the logits a row a request: a pruned
    node's one row is its member's last prompt row where the record binds a row a prompt position, else its only one.
    Bound at another number of rows, it is refused by name."""
    low, _f, _t, weights, slots = window
    s0, K = slots[0], NODES
    mid, norm, lp = K - PRUNED - 2, K - PRUNED, s0.row[1]
    rows, store = _served(low, s0, T=1, also=(mid,), whole=(mid, norm))
    ours = {key: a for key, a in store.items() if key[:2] == (0, f"op{norm}")}
    assert all(a.size == lp * f.leaves for a, f in zip(ours.values(), WR._fields(low.fn.body.nodes[norm].fn)))
    assert not np.array_equal(s0.values[norm][0], s0.values[(lp - 1) * K + norm][0])
    got = WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(rows, *_read(store)))
    assert got.source[(lp - 1) * K + norm] == WR.SERVED
    assert all(np.array_equal(a, b) for x, y in zip(got.values, s0.values) for a, b in zip(x, y))
    two = {**store, **{key: a[:2 * (a.size // lp)] for key, a in ours.items()}}
    with pytest.raises(ValueError, match=rf"op{norm}: its one prefill row's members bound at \[2.0\] rows, not a row a prompt "
                                         rf"position \({lp}\) nor one"):
        WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(rows, *_read(two)))


def test_a_narrowed_linears_rows_are_its_node_read_whole_at_every_prompt_position(window):
    """The registered compiler narrows the last layer's fused qkv on the prompt rows before the last to its keys' and
    values' columns (`program.prune.narrowed`), which the manifest spells ``out@<width>`` beside the last row's whole
    ``out``, its one row there: both are the node's prefill, read a row a prompt position from the whole rows the record
    binds under ``out``, since the engine computes every one. Bound at the last row only, the node is refused by name, and
    a narrowed op with no whole row beside it is refused as an op of neither a row a prompt position nor the last one."""
    low, _f, _t, weights, slots = window
    s0, K, nodes = slots[0], NODES, low.fn.body.nodes
    query = [j for j in range(K) if nodes[j].fn.name == "RoPE"][-2]
    k, lp, op = query - 1, s0.row[1], f"op{query - 1}"
    N = WR._fields(nodes[k].fn)[0].leaves
    w = N - nodes[query].fn.bindings["NHEADS"] * nodes[query].fn.bindings["D"]
    rows, store = _served(low, s0, T=1)
    m, _n, dt = next(d for d in rows if d["op_path"] == op)["members"][0]

    def spelt(d):
        if (d["op_path"], d["step"]) != (op, 0):
            return d
        return {**d, "row": 0} if d["row"] == lp - 1 else {**d, "members": [(f"{m}@{w}", w, dt)]}

    program = [spelt(d) for d in rows]
    assert sorted(d["row"] for d in program if (d["op_path"], d["step"], d["members"][0][0]) == (op, 0, f"{m}@{w}")) \
        == list(range(lp - 1))
    record = {**store, (0, op, f"{m}@{w}"): store[(0, op, m)].reshape(lp, N)[:, N - w:].ravel()}
    served = WR.Served(program, *_read(record))
    assert len(served.prefill) == K and [(d["row"], d["members"][0][0]) for d in served.prefill[k]] == [(p, m) for p in range(lp)]
    got = WR.Slot(low.fn, CLASS, SLOTS[0], weights, served)
    assert got.source == [WR.COMPUTED if p < lp - 1 and j >= K - PRUNED else WR.SERVED for p in range(lp + 1) for j in range(K)]
    assert all(np.array_equal(a, b) for x, y in zip(got.values, s0.values) for a, b in zip(x, y))

    last = {**record, (0, op, m): store[(0, op, m)][(lp - 1) * N:]}
    with pytest.raises(ValueError, match=rf"{op}: a narrowed linear's node, read whole at every prompt position, its widest "
                                         rf"members bound at \[1.0\] rows, not a row a prompt position \({lp}\)"):
        WR.Served(program, *_read(last))
    alone = [d for d in program if (d["op_path"], d["step"], d["members"][0][0]) != (op, 0, m)]
    with pytest.raises(ValueError, match=rf"{op}: prefill rows \[{', '.join(map(str, range(lp - 1)))}\], not one a prompt "
                                         r"position nor the last one's"):
        WR.Served(alone, *_read(record))


def test_a_call_boundary_is_matched_at_its_site_whatever_its_fx_node_counts_to(window):
    """A call boundary's op path names its fx node, which counts on over the request's steps (the served int8 linears'
    ``…/row_quant`` to ``…/row_quant_63`` at the prefill, ``…/row_quant_64`` on at the first decode step): a decode step's
    op is the prefill's at its site, its op path less the fx node, by its ordinal among the step's ops there. One missing
    there is refused."""
    low, _f, _t, weights, slots = window
    s0 = slots[0]
    rows, store = _served(low, s0, T=1)
    by: dict[tuple, list[int]] = {}
    for d in rows:
        by.setdefault(tuple(m[0] for m in d["members"]), []).append(int(d["op_path"][2:]))
    a, b = next(sorted(set(v))[:2] for v in by.values() if len(set(v)) > 1)

    def spell(op: str, step: int) -> str:
        k = int(op[2:])
        n = 2 * step + (k == b)
        return op if k not in (a, b) else "m/row_quant" + (f"_{n}" if n else "")

    rows = [{**d, "op_path": spell(d["op_path"], d["step"])} for d in rows]
    store = {(st, spell(op, st), m): v for (st, op, m), v in store.items()}
    served = WR.Served(rows, *_read(store))
    assert [served.prefill[k][0]["op_path"] for k in (a, b)] == ["m/row_quant", "m/row_quant_1"]
    assert [served.decode[0][k]["op_path"] for k in (a, b)] == ["m/row_quant_2", "m/row_quant_3"]
    got = WR.Slot(low.fn, CLASS, SLOTS[0], weights, served)
    assert all(np.array_equal(x, y) for u, w in zip(got.values, s0.values) for x, y in zip(u, w))
    with pytest.raises(ValueError, match=r"step 1: none of the prefill's op \('m/row_quant_1', .*\) \(1 ops\)"):
        WR.Served([d for d in rows if d["op_path"] != "m/row_quant_3"], *_read(store))


class _Index:
    """A `ProgramIndex` of served rows (`_served`'s): each row's spec, and the rest as its ``describe`` gives it."""

    def __init__(self, rows) -> None:
        self.served = rows
        self.rows = [{"i": i, "spec": d["spec"]} for i, d in enumerate(rows)]
        self.inst: dict[str, Any] = {}

    def describe(self, r):
        return {k: v for k, v in self.served[r["i"]].items() if k != "spec"}

    def value_families(self, i):
        return {m[0]: "instance_outputs" for m in self.served[i]["members"]}


def _record(store, rid: str, arrive: int, spell) -> CommittedStore:
    """``store``'s values (`_served`'s) as a record's `CommittedStore`: request ``rid``'s at engine step ``arrive`` plus its
    step, bound under op path ``spell(op, engine step)``."""
    entries, blobs = [], {}
    for (step, op, m), a in store.items():
        e = arrive + step
        name = f"{spell(op, e)}/{m}"
        entries.append({"name": name, "element_offset": 0, "encoding": {"dtype": a.dtype.name},
                        "identity": {"request_id": rid, "step": e, "op_path": spell(op, e), "output_member": m,
                                     "element_range": [0, a.size]}})
        blobs[(e, name)] = a.tobytes()
    return CommittedStore({"entries": entries}, lambda step, name, lo, hi: blobs[(step, name)][lo:hi])


def test_a_later_arrivals_call_boundary_is_read_under_the_fx_node_the_record_binds_it_under(window):
    """The serve commits a batch's call boundary under one fx node, counted over its earliest arrival's steps (the served
    int8 linears' ``…/row_quant_64`` at engine step 1 for every request then in the batch), so a request that arrived later
    has its Program's call boundary at another count: it is read under the record's one op path of its module and op for
    the request at that engine step. Where its module calls its op twice at a step, or the record binds the request under
    two such op paths there or none, its read is refused by name."""
    low, _f, _t, weights, slots = window
    s0 = slots[0]
    rows, store = _served(low, s0, T=1)
    by: dict[tuple, list[int]] = {}
    for d in rows:
        by.setdefault(tuple(m[0] for m in d["members"]), []).append(int(d["op_path"][2:]))
    members, (a, b) = next((ms, sorted(set(v))[:2]) for ms, v in by.items() if len(set(v)) > 1)

    def spelling(module):
        def spell(op: str, step: int) -> str:
            k = int(op[2:])
            n = 2 * step + (k == b)
            return op if k not in (a, b) else f"{module(k)}/row_quant" + (f"_{n}" if n else "")
        return spell

    def served(spell, record):
        program = [{**d, "op_path": spell(d["op_path"], d["step"])} for d in rows]
        return WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served.of_record(_Index(program), record, "late", 1))

    apart = spelling(lambda k: f"m{k}")
    record = _record(store, "late", 1, apart)
    assert record.ops("late", 1)[f"m{a}/row_quant_2"] == set(members)
    assert not record.has("late", 1, f"m{a}/row_quant", members[0])
    got = served(apart, record)
    assert all(np.array_equal(x, y) for u, w in zip(got.values, s0.values) for x, y in zip(u, w))

    one = spelling(lambda k: "m")
    with pytest.raises(ValueError, match=r"position 0's node \d+: no committed identity for \(late, step 1, m/row_quant/"):
        served(one, _record(store, "late", 1, one))
    twice = _record({**store, **{(0, "again", m): store[(0, f"op{a}", m)] for m in members}}, "late", 1,
                    lambda op, step: f"m{a}/row_quant_9" if op == "again" else apart(op, step))
    assert {f"m{a}/row_quant_2", f"m{a}/row_quant_9"} <= set(twice.ops("late", 1))
    with pytest.raises(ValueError, match=rf"position 0's node \d+: no committed identity for \(late, step 1, m{a}/row_quant/"):
        served(apart, twice)
    none = _record({key: v for key, v in store.items() if key[:2] != (0, f"op{b}")}, "late", 1, apart)
    with pytest.raises(ValueError, match=rf"no committed identity for \(late, step 1, m{b}/row_quant_1/"):
        served(apart, none)


@pytest.mark.usefixtures("code_versions_stood_in")
def test_a_requests_checks_are_no_served_op_and_any_other_row_without_a_member_is_refused(tmp_path):
    """A request's Program ends on its length rule (`StopCheck_v1`), a check its instances header names: it computes the
    Program's verdict from the request's sampled ids, the engine never runs it and the record binds no member of it. So
    `Served.of_record` takes the request's ops from its Program's other rows, however the request finished, and a row with
    no committed member that the header names no check is refused by name. Each prompt is one id, a prefill of one row an
    op, and no record is given: `of_record` reads none before a value is asked."""
    reg = S._registered()
    advice, ingress = S._session([(1, 3, S.GREEDY), (1, 3, {**S.GREEDY, "max_tokens": 3})],
                                 [[(0, 0, 1), (1, 0, 1)], [(0, 1, 1), (1, 1, 1)], [(0, 2, 1), (1, 2, 1)]])
    c = compile_steps(reg, advice, ingress, named=True)
    assert sorted(u.stop["finish"] for u in c.requests) == [1, 2]
    for u in c.requests:
        d = tmp_path / str(u.stop["finish"])
        write_build(str(d), c.programs[u.digest], c.named[u.digest])
        pi = ProgramIndex(read_instances(str(d / "instances.json.gz"), expand=False), program_dir=str(d))
        (_what, i, _lo, _hi), = check_rows(pi)
        rule = pi.rows[i]
        assert i == len(pi.rows) - 1 and rule["spec"].startswith("StopCheck_v1{")
        assert not (pi.describe(rule) or {}).get("members")
        served = WR.Served.of_record(pi, None, "q", 0)
        assert (served.LP, len(served.decode)) == (1, pi.n_steps - 1)
        assert [x["spec"] for x in served.rows] == [r["spec"] for r in pi.rows[:-1]]
        del pi.inst["checks"]
        with pytest.raises(ValueError, match=rf"q: row {i} \(StopCheck_v1\{{.*\}}\) has no committed member"):
            WR.Served.of_record(pi, None, "q", 0)


def test_a_node_served_under_other_statics_is_held_to_its_class_nodes_value(window):
    """FA3's key block follows its launch's query rows, so the engine's prefill attention is `Attention_v4` at another
    ``BN`` than the class Program's one-row decode's: a node served under other statics of its class node's Definition is
    taken where its value is the class node's on its operands, and refused by name where it isn't."""
    low, _f, _t, weights, slots = window
    s0, k = slots[0], _position(low.fn, "Attention")[0]
    rows, store = _served(low, s0, T=1)
    op = f"op{k}"
    rows = [{**d, "spec": re.sub(r"(?<=[{,])BN=(\d+)", lambda m: f"BN={int(m[1]) + 64}", d["spec"])}
            if (d["op_path"], d["step"]) == (op, 0) else d for d in rows]
    prefill = [d for d in rows if (d["op_path"], d["step"]) == (op, 0)]
    assert len(prefill) == s0.row[1] and all(d["spec"] != descriptor_id(low.fn.body.nodes[k].fn) for d in prefill)
    got = WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(rows, *_read(store)))
    assert got.source[k] == WR.SERVED
    assert all(np.array_equal(x, y) for u, w in zip(got.values, s0.values) for x, y in zip(u, w))
    m = prefill[0]["members"][0][0]
    bad = dict(store)
    bad[(0, op, m)] = bad[(0, op, m)] ^ np.uint16(1)
    with pytest.raises(ValueError, match=rf"node {k} .* at position 0: served as .*not its Definition's value on its "
                                         r"operands \(served as Attention_v4\{"):
        WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(rows, *_read(bad)))


def test_the_registered_values_are_each_weights_rows_and_the_scan_state(window):
    """Each weight's parameter is its words in rows of its reading port's (``row/v2``), the table's its V rows of H
    (``row-seg``), each committed under salts of its own, 192 a row and no two alike: the table and a weight are core's
    commits of those rows under the weights record's program and those salts, and the entries name nothing but each value's size, domain and hiding
    root. The scan state is no value of the committer's but the network attester's, a row a bucket of the window. Each
    value is in its reading ports' rows, which a window's commit refuses otherwise. Another parameter is refused."""
    low, fields, got, *_ = window
    assert set(got) == set(low.circuit.registered.params)
    reads = low.circuit.registered.reads
    lays = {r.reader: dict(IF.layout(low.circuit.template(r.reader)).schemas) for r in reads}
    assert {(r.param, got[r.param].schema) for r in reads} == {(r.param, lays[r.reader][r.port]) for r in reads}
    t = got[TABLE]
    assert (t.leaves, t.words, t.schema, t.window) == (256, 64, CR.SCHEMA_SEG, None) and t.schema == E.SCHEMA
    assert np.array_equal(t.rows[:].reshape(-1), fields[EMBED])
    program = WC.class_program(_registered(), WW.WEIGHTS_CLASS.positions).digest
    assert {v.program for p, v in got.items() if p != SS.SCAN_STATE} == {program}
    assert (got[SS.SCAN_STATE].leaves, got[SS.SCAN_STATE].words) == (CLASS.buckets, SS.words(CLASS.scan))
    for name, rows, schema in ((TABLE, fields[EMBED].reshape(-1, 64), CR.SCHEMA_SEG),
                               (WB.weight_param(EMBED, 1024), fields[EMBED].reshape(-1, 1024), CR.SCHEMA_V2)):
        salts = list(got[name].salts)
        assert len(salts) == len(rows) == len(set(salts)) and {len(x) for x in salts} == {192}
        ref = RV.Registered.commit(program, name, rows.tolist(), salts, schema=schema)
        assert got[name].entry() == ref.entry() and got[name].levels == ref.levels and got[name].schema == schema
        assert set(got[name].entry()) == {"value", "root", "leaves", "words", "domain"}
    down = got[WB.weight_param("model.layers.0.mlp.down_proj.weight", 1024)].rows[:]
    assert down.shape == (8, 1024) and np.array_equal(down.reshape(-1), fields["model.layers.0.mlp.down_proj.weight"])
    with pytest.raises(ValueError, match="'weights/x@3/rows' is no weight parameter"):
        WW.weight_value(program, "weights/x@3/rows", np.zeros(3, np.uint16))


def _as_leaves(x: np.ndarray, width: int, leaves: int) -> np.ndarray:
    """A port's ``width``-bit elements as its Definition's ``leaves``: the elements themselves, or each element's bits, least
    significant first, where the port's leaves are bits."""
    if x.shape[1] == leaves:
        return x
    return ((x.astype(np.uint64)[:, :, None] >> np.arange(width, dtype=np.uint64)) & 1).reshape(len(x), -1)


def _packed(bits: np.ndarray, width: int) -> np.ndarray:
    """``(n, k · width)`` bits as ``k`` ``width``-bit elements, least significant first (`_as_leaves`' inverse)."""
    return (bits.reshape(len(bits), -1, width) << np.arange(width, dtype=np.uint64)).sum(axis=2)


def test_a_bit_leaf_ports_elements_are_its_bits_least_significant_first():
    """`_packed` against `SlotEntry_v1`'s entry, words of bit leaves, which pack to `entry_reference`'s words; and
    each lane's `ScanSlots_v5`, whose ports are words, on a state row of u16 words gives `scan_reference`'s lane."""
    row: np.ndarray = np.asarray([state_row(CLASS.scan, LANES)], dtype=np.uint64)
    for scan in SS.bridge(CLASS.scan, CLASS.prompt):
        (_, t), = scan.signature()[0]
        out = np.asarray(evaluate_batch(scan, _as_leaves(row, 16, t.leaves), kernel="reference")).reshape(-1).astype(np.uint64)
        k, s, o, h = WR._lane(scan, row.tolist(), 0, IO.scan_reference)
        assert scan.name == "ScanSlots" and out.tolist() == [k, *s, o, *h]
    entry = bind(IO.SlotEntry, KO=2, IDB=17, NP=4, STOP=())
    state = [(2 - 4 - 2) & IO.M32, 2, 17, 3]
    bits = np.asarray(evaluate_batch(entry, np.array([[1]], np.uint32), np.array([state], np.uint32), kernel="reference"))
    words = _packed(bits.reshape(1, -1).astype(np.uint64), 16)[0].tolist()
    assert words == IO.entry_reference(1, state, 2, 17, 4, ()) and words[:3] == [1 | 2 << 1, 0, 17 << 1]


@pytest.mark.slow
def test_a_windows_rows_are_what_its_definitions_compute_on_its_committed_inputs(window):
    """The rows recorded into a `Window` (its registered values under the committer's key, the tables' supplied) are
    every unit's Definition on the inputs the window commits: its wired rows, its registered reads' rows and its openings',
    each word template's outputs under the reference evaluator, each embedding read's by `gates.embedding.reference`; the
    window closes over them, labelled by step, slot and form. What no cheaper test reaches: the circuit's own wires and reads
    resolved by the window, against every Definition. Slow (minutes) for the window's members, which trace the embedding
    read's population Program, and the reference evaluator on every unit."""
    low, _f, values, weights, slots = window
    rows_, store = _served(low, slots[0], T=1)
    s0 = WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(rows_, *_read(store)))
    rows = WR.Rows(low, [s0, slots[1]], values)
    win = IF.Window(low.circuit, RUN, DESIGN, 64, registered=values, forms=WR.FORMS)
    win.supply()
    rows.record(win)
    bad: dict[str, list[int]] = {}
    for fn, n in low.circuit.templates:
        tid = descriptor_id(fn)
        lay = IF.layout(fn)
        ins = win.input_words(tid, 0, n)
        if fn.name == "EmbeddingRead":
            t = ins[0].view("<u4").reshape(-1)
            for u in range(n):
                path = np.ascontiguousarray(ins[5][u]).view("<u8").astype(">u8").tobytes()
                h, ok = E.reference_opening(256, int(t[u]), *(np.ascontiguousarray(ins[g][u]).view("<u8").astype(">u8").tobytes()
                                                      for g in (1, 2)), ins[3][u].tolist(),
                                    np.ascontiguousarray(ins[4][u]).view("<u8").astype(">u8").tobytes(),
                                    [path[k:k + 64] for k in range(0, len(path), 64)])
                if h != rows.outs[tid][0][u].tolist() or [ok] != rows.outs[tid][1][u].tolist():
                    bad.setdefault(tid, []).append(u)
            continue
        params, ret = fn.signature()
        args = [_as_leaves(np.ascontiguousarray(x).view(f"<u{w // 8}").reshape(n, -1), w, t.leaves)
                for x, w, (_, t) in zip(ins, lay.widths, params)]
        flat = np.asarray(evaluate_batch(fn, *args, kernel="reference")).reshape(n, -1).astype(np.uint64)
        got, want, at = [], [], 0
        for o, w, f in zip(rows.outs[tid], lay.widths[len(lay.rows):], getattr(ret, "fields", None) or (ret,)):
            got.append(np.ascontiguousarray(o).view(f"<u{w // 8}").reshape(n, -1).astype(np.uint64))
            part, at = flat[:, at:at + f.leaves], at + f.leaves
            want.append(part if part.shape[1] == got[-1].shape[1] else _packed(part, w))
        got, want = np.concatenate(got, axis=1), np.concatenate(want, axis=1)
        if (off := np.flatnonzero((got != want).any(axis=1))).size:
            bad[tid] = off.tolist()
    assert bad == {}
    c = win.close()
    assert c is not None
    labels = win.labels()
    seen = {lab["form"] for m in labels["members"] for *_, lab in m["labels"]}
    assert seen == set(WR.FORMS) and all(m["labels"] for m in labels["members"])
    assert {int(x) for v in rows.source.values() for x in v} == {WR.SERVED, WR.COMPUTED, WR.EVALUATED}


def test_a_cut_norms_parts_hold_its_sum_and_its_tail_the_nodes_value():
    """At a class whose norm after down_proj is cut at its residual add (`test_window_circuit.SPLIT`), each add part holds its
    words of the node's sum, which is `RMSNormFusedCudaAdd_v2` on the node's x and res, and the tail the node's value, which
    is `RMSNormFusedCudaTail_v1` on the parts' sums in order of AT; every unit is filled."""
    low, _f, values, _w, slots = _window(SPLIT, SPLIT_GRAIN)
    rows = WR.Rows(low, slots, values)
    nodes, seen = low.fn.body.nodes, 0
    for P, s in zip(low.programs, slots):
        for j, parts in P.norms.items():
            x, res = (s.operand(a) for a in nodes[j].args[:2])
            sums = [rows.outs[r.tid][0][r.unit] for r in parts]
            for at, (r, got) in enumerate(zip(parts, sums)):
                add = rows.fns[r.tid]
                assert add.bindings["AT"] == at and np.array_equal(
                    np.asarray(evaluate_batch(add, x[None, 16 * at:16 * (at + 1)], res[None])).reshape(-1), got)
            (_, _, normed), (_, _, r) = P.outs[j]
            tail = rows.fns[normed.tid]
            want = np.asarray(evaluate_batch(tail, np.concatenate(sums)[None], s.operand(nodes[j].args[2])[None])).reshape(-1)
            assert np.array_equal(want, WR._words(s.values[j])) and np.array_equal(rows.outs[r.tid][1][r.unit], want[64:])
            seen += 1
    assert seen == sum(2 * P.positions - (P.L - 1) for P in low.programs) == 13


@pytest.mark.slow
@pytest.mark.parametrize("cls, lanes", [(WC.WindowClass(1, 1, 1), [(1, 1, 1, [17])]), (CLASS, LANES)], ids=["1,1,1", "2,4,2"])
def test_a_windows_scan_state_is_a_run_input_committed_whole_and_registered_by_no_pin(cls, lanes, tmp_path):
    """The scan's state is C's run input node, all ``buckets`` rows, of which the window's lanes read row 0: the close
    commits its tree ``input/n<w>`` from the scan state's value, the rows no unit reads included, in the domain C gives
    the node, and the registration registers it as a run input, never as a value a pin holds. Written as `prove_path`
    writes it, with no index salt given, its index salt is 192 fresh bytes read back from the window's
    ``index-salt.bin`` under the index leaf it registers. The record registers the embedding table and its two anchors
    (`supplied.anchors`), and the Python verifier takes it whole, handed what the audit hands it (the deployment's
    registration record, `deployment_record`, on the development path, and the partition of the records), but R8 refuses
    the scan's state run input where two lanes read it, as only the variant with no members does. Slow (minutes) for the
    window's close, which no cheaper path to `Window.write` skips."""
    low, _f, values, _w, slots = _window(cls=cls, lanes=lanes)
    c, words = low.circuit, SS.words(cls.scan)
    assert c.registered.values == () and c.registered.run == (SS.SCAN_STATE,)
    program, digest = c.program()[0], IB.program_digest(c)
    names = [p for p, _ in program.fn.signature()[0]]
    w = names.index(SS.SCAN_STATE)
    assert dict(program.fn.signature()[0])[SS.SCAN_STATE].leaves == cls.buckets * words and program.bindings[w] == "run"
    heads = [(r, m) for r, (_o, _lp, m, _ids) in zip(([occ, lp, *ids] for occ, lp, _m, ids in lanes), lanes)]
    state = WR.state_value(digest, cls, SR.standin_rows(cls, heads), window=RUN)
    vals = {**values, SS.SCAN_STATE: state}
    win = IF.Window(c, RUN, DESIGN, 64, registered=vals, forms=WR.FORMS)
    reads = [r for r in win.runs if r.param == f"input/n{w}"]
    assert [(r.at, r.base, r.step) for r in reads] == [(0, 0, (words,))] * cls.lanes
    win.supply()
    WR.Rows(low, slots, vals).record(win)
    win.write(win.close(), tmp_path, {}, registrar=DK)
    rec = json.loads((tmp_path / "registration.json").read_text())
    kept = (tmp_path / CS.INDEX_SALT).read_bytes()
    assert len(kept) == CS.SALT_BYTES and kept not in {bytes(s) for s in state.salts}
    assert CS.index_leaf((tmp_path / "index.json").read_bytes(), kept).hex() == rec["window"]["index_leaf"]
    assert SS.SCAN_STATE not in rec["window"].get("registered_roots", {})
    assert SS.SCAN_STATE not in [e["value"] for e in rec.get("registered", [])]
    tree = win.inputs[f"input/n{w}"]
    want = IF.run_input_domain(program, digest, w, words, RUN)
    assert (tree.leaves, tree.domain.domain_id) == (cls.buckets, want.domain_id) and want.count == cls.buckets
    assert np.array_equal(CS.value_rows(tree, range(cls.buckets)), CS.value_rows(state, range(cls.buckets)))
    doc = json.loads((tmp_path / "circuit.json").read_text())
    assert f"input/n{w}" in {r["param"] for r in doc["run_inputs"]} and doc["registered"]["run"] == [SS.SCAN_STATE]
    anchors = SU.anchors(program)
    assert anchors == (f"{TABLE}/domain", f"{TABLE}/root")
    assert {e["value"]: e for e in rec["registered"]} == {v: win.registered[v].entry() for v in (*anchors, TABLE)}
    model = _registered().phi.model
    deployed = WW.deployment_record(WW.WeightsRegistration(
        program=values[TABLE].program, repo=model["repo"], revision=model["revision"], grain=asdict(GRAIN),
        values={p: v.entry() for p, v in values.items() if p != SS.SCAN_STATE}, unrooted={}).dump())
    of_record = R.registration_parsed({"record": json.dumps(deployed).encode().hex(), "head": R.DEVELOPMENT}, None, None,
                                      None)
    mine, _doc = IF.of_records(tmp_path, DESIGN, IF.resolve)
    got = R.check(rec, program, win.window.query, DESIGN, CS.SCHEME,
                  {r["port"]: {"leaves": r["leaves"], "domain": r["domain"]} for r in rec["roots"]},
                  leaf_layer=rec.get("leaf_layer"), partition=mine.partition,
                  registered={v: win.registered[v].words for v in anchors}, schemas={v: CR.SCHEMA_V2 for v in anchors},
                  weights_of_record=of_record[:2])
    assert set(got.codes) == ({"R8-shared-input"} if cls.lanes > 1 else set()), got
    assert cls.lanes == 1 or got.detail["R8-shared-input"]["inputs"] == [w], got


class _Kept:
    """A window's `record` as `Rows.record` calls it, keeping each call's template, label and element bytes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str, list[bytes | None], list[bytes]]] = []

    def record(self, fn, ins, outs, label=None) -> None:
        self.calls.append((fn.id, len(outs[0]), json.dumps(label), [None if a is None else a.tobytes() for a in ins],
                           [a.tobytes() for a in outs]))


def _rows(requests, lanes):
    """The served circuit of ``requests`` at a scan state whose row at bucket 0 is ``lanes`` (`STATE`'s later rows), its
    registered values and its `Rows` of every node computed."""
    reg = _registered()
    low = WC.lower(reg, CLASS, None, GRAIN, requests=requests)
    rows_ = [state_row(CLASS.scan, lanes), *STATE[1:]]
    fields = _weights(low.fn)
    state = RV.Registered.commit(PROGRAM, SS.SCAN_STATE, rows_, [hm96_salt(bytes(32), 0, j) for j in range(len(rows_))],
                                 schema=CR.SCHEMA_V2)
    values = {**WW.weight_values(reg, fields, None, GRAIN, keep_rows=True), SS.SCAN_STATE: state}
    return low, values, WR.Rows(low, unserved_slots(low, rows_, lambda label, lo, hi: fields[label][lo:hi]), values)


#: one request in lane 0 (a prompt of 3 keeping 1 output: 3 positions), lane 1 empty
ONE = [(1, 3, 1, [5, 77, 42, 0]), (0, 0, 0, [0, 0, 0, 0])]


def test_a_window_of_fewer_requests_than_slots_is_its_one_requests_rows():
    """A window serving one request in the class's two slots lowers to that request's Program alone, of its 3 positions,
    and its rows fill every unit of it once (`Rows` refuses otherwise) from that request's values: no slot for the lane
    that served nothing, whose collector and entry are in the request's slot, and no position past the request's end.
    `record` hands the window every unit."""
    low, _values, rows = _rows((WC.Request(CLASS.anchor(0), CLASS.lane(0), 3, 1),), ONE)
    assert len(low.programs) == 1 and low.programs[0].positions == 3 and len(low.fn.body.nodes) == 3 * NODES
    assert low.circuit.blocks.slots == 1
    kept = _Kept()
    rows.record(kept)
    counts: dict[str, int] = {}
    for fid, n, *_ in kept.calls:
        counts[fid] = counts.get(fid, 0) + n
    want: dict[str, int] = {}
    for t, n in rows.n.items():
        want[rows.fns[t].id] = want.get(rows.fns[t].id, 0) + n
    assert counts == want


@pytest.mark.slow
def test_a_window_of_fewer_requests_than_slots_commits(tmp_path):
    """The one-request window (`ONE`) closed and written as `prove_path`'s window commit writes it: its circuit of one
    Program commits, its registration naming the embedding table and its two anchors alone, its scan state a run input. What the quick test above can't reach: `Window.close` and
    `Window.write` on a window not of the class's shape. Slow (minutes) for the close."""
    low, values, rows = _rows((WC.Request(CLASS.anchor(0), CLASS.lane(0), 3, 1),), ONE)
    win = IF.Window(low.circuit, RUN, DESIGN, 64, registered=values, forms=WR.FORMS)
    win.supply()
    rows.record(win)
    win.write(win.close(), tmp_path, {}, registrar=DK)
    rec = json.loads((tmp_path / "registration.json").read_text())
    assert [e["value"] for e in rec["registered"]] == [*SU.anchors(low.circuit.program()[0]), TABLE]
    assert SS.SCAN_STATE not in rec["window"].get("registered_roots", {})


def _private(seed: int):
    """The window of `REQUESTS` (one advice) under weights drawn at ``seed``: its registered values and its `Rows`, every
    node computed. Two seeds give one circuit and two sets of private values."""
    reg = _registered()
    low = WC.lower(reg, CLASS, None, GRAIN, requests=REQUESTS)
    fields = _weights(low.fn, seed)
    state = RV.Registered.commit(PROGRAM, SS.SCAN_STATE, STATE, [hm96_salt(bytes(32), 0, j) for j in range(len(STATE))],
                                 schema=CR.SCHEMA_V2)
    values = {**WW.weight_values(reg, fields, None, GRAIN, keep_rows=True), SS.SCAN_STATE: state}
    return low, values, WR.Rows(low, unserved_slots(low, STATE, lambda label, lo, hi: fields[label][lo:hi]), values)


def test_a_windows_labels_are_its_step_slot_and_form_alone():
    """One advice under two sets of private values (`_private`): its rows differ, and `Rows.record` hands the window the
    same runs under byte-identical labels, each only its step, slot and form. Every unit is in one run, and both forms
    and both slots are there."""
    (_la, _va, a), (_lb, _vb, b) = _private(3), _private(4)
    assert any(not np.array_equal(x, y) for t in a.outs for x, y in zip(a.outs[t], b.outs[t]))
    ka, kb = _Kept(), _Kept()
    a.record(ka)
    b.record(kb)
    runs = [(fid, n, lab) for fid, n, lab, *_ in ka.calls]
    assert runs == [(fid, n, lab) for fid, n, lab, *_ in kb.calls]
    labels = [json.loads(lab) for *_, lab in runs]
    assert all(set(lab) == {"step", "slot", "form"} for lab in labels)
    assert {lab["form"] for lab in labels} == set(WR.FORMS) and {lab["slot"] for lab in labels} == {0, 1}
    counts: dict[str, int] = {}
    for fid, n, _ in runs:
        counts[fid] = counts.get(fid, 0) + n
    want: dict[str, int] = {}
    for t, n in a.n.items():
        want[a.fns[t].id] = want.get(a.fns[t].id, 0) + n
    assert counts == want


#: the keys of `Window.write`'s public JSON whose values are hiding commitments or digests over them, and the commit's seconds
HIDING = {"root", "roots", "registered_roots", "paths", "registered_paths", "public_sha512", "leaf_layer",
          "registration_digest", "seconds"}
PRIVATE = re.compile(r"[mtv]\d+-private\.bin|index\.json|index-salt\.bin")


def _hide(x: Any, role: str) -> Any:
    if isinstance(x, dict):
        return {k: _hide(v, role) for k, v in x.items()}
    if isinstance(x, list):
        return [_mask(v) if isinstance(v, dict) else _hide(v, role) for v in x]
    return f"<{role}>"


def _mask(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _hide(v, k) if k in HIDING else _mask(v) for k, v in x.items()}
    return [_mask(v) for v in x] if isinstance(x, list) else x


def _public(c, d) -> dict[str, Any]:
    """Window dir ``d``'s public files, each hiding commitment and digest over one replaced by its role (`HIDING`; a public
    body's ``b || c`` strings by their length), each prover-only file by its name once it is its owner's alone."""
    out: dict[str, Any] = {}
    for f in sorted(d.iterdir()):
        if PRIVATE.fullmatch(f.name):
            assert f.stat().st_mode & 0o077 == 0, f"{f.name} is readable by others"
            out[f.name] = "private"
        elif f.name in ("registration.json", "window.json"):
            out[f.name] = _mask(json.loads(f.read_text()))
        elif f.suffix == ".bin":
            k = int(f.name[1:].split("-")[0])
            ports = ([c.window.row_tables[k].port] if f.name[0] == "t"
                     else [c.window.members[k].port(p) for p, _ in c.window.members[k].all_rows])
            body, bc = f.read_bytes(), sum(len(c.bc[p]) for p in ports)
            out[f.name] = (len(body), body[bc:])
        else:
            out[f.name] = f.read_bytes()
    return out


@pytest.mark.slow
def test_one_advice_under_two_sets_of_private_values_has_one_public_record(tmp_path):
    """One advice under two sets of private values (`_private`), each closed and written by `Window.write` as
    `prove_path`'s window commit writes it, under one index salt: every public file is byte for byte the other's once each
    hiding commitment (a root, a row's ``b || c``, a registered read's path) and each digest over one is replaced by its
    role, and the index and its salt are prover only. What the labels test can't reach: everything else `Window.write`
    makes public. Slow (minutes) for the two windows' closes."""
    pub, salt = {}, os.urandom(CS.SALT_BYTES)
    for seed in (3, 4):
        low, values, rows = _private(seed)
        win = IF.Window(low.circuit, RUN, DESIGN, 64, registered=values, forms=WR.FORMS)
        win.supply()
        rows.record(win)
        c = win.close()
        prov = {"window_circuit": {"class": [CLASS.slots, CLASS.prompt, CLASS.outputs, CLASS.lanes, CLASS.every], "window": 0}}
        win.write(c, tmp_path / str(seed), prov, registrar=DK, index_salt=salt)
        pub[seed] = _public(c, tmp_path / str(seed))
    assert pub[3] == pub[4]
    assert {"index.json", "index-salt.bin", "circuit.json", "registration.json"} <= set(pub[3])


def test_a_windows_leaf_salts_are_each_one_position_of_its_key_and_its_strings_the_ones_close_makes():
    """`leaf_strings`: row r of port q of template t takes the key's stream at index (t, q), leaf r (blocks 3r .. 3r + 2,
    as `salt_stream.engine.fill` writes them), so no position of the stream serves two rows of a window, no index's head
    passes the engine's ``HEAD_MAX``, and `close` handed them makes the roots it makes from the same salts."""
    from verity.circuits.ops import gemm as G
    from verity.commitments import randomness as RND
    from verity.commitments.cuda.salt_stream.engine import HEAD_MAX

    from verity_vllm.program.predict.window_rows import leaf_strings

    coord, acc = bind(G.ScaledMmS8Coordinate, K=64, **G.GEMM_SM120_S8), bind(G.GemmAccumulatorS8, K=64, **G.GEMM_SM120_S8)
    rng = np.random.default_rng(0)
    win = IF.Window(IF.Circuit("test", ((coord, 10), (acc, 3))), "ab" * 64, {"law": "stratified", "k": 2}, 4)
    for fn, n in ((coord, 10), (acc, 3)):
        lay = IF.layout(fn)
        ins = [rng.integers(0, 1 << 16, (n, w), dtype=np.uint16).view(f"<u{lay.widths[g] // 8}")
               for g, (_p, w) in enumerate(lay.rows)]
        outs = evaluate_batch(fn, *ins, kernel="reference")
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        win.record(fn, ins, [np.asarray(o).astype(f"<u{w // 8}") for o, w in zip(outs, lay.widths[len(lay.rows):])])
    key = RND.derive(bytes(64), "verity/window-commit/salts/v1", {"window": "ab" * 64})
    got, rows = leaf_strings(win, key), win.leaf_rows()
    assert set(got) == set(rows) and {lr.template for lr in rows.values()} == {0, 1}
    heads = {k: key.head((lr.template, lr.port)) for k, lr in rows.items()}
    positions = [(heads[k], 3 * u + j) for k, lr in rows.items() for u in lr.units for j in range(3)]
    assert len(set(positions)) == len(positions) == 3 * sum(len(lr.rows) for lr in rows.values())
    assert max(len(h) for h in heads.values()) <= HEAD_MAX
    salts = [ys[i:i + CS.SALT_BYTES] for ys, _ in got.values() for i in range(0, len(ys), CS.SALT_BYTES)]
    assert len(set(salts)) == len(salts)
    k, lr = next(iter(rows.items()))
    assert got[k][0][CS.SALT_BYTES:2 * CS.SALT_BYTES] == key.block((lr.template, lr.port), 3) + key.block(
        (lr.template, lr.port), 4) + key.block((lr.template, lr.port), 5)
    by_member = {p: s for k, lr in rows.items() for p, s in lr.split(got[k][0], CS.SALT_BYTES).items()}
    assert win.close(strings=got).roots == win.close(salts_of=by_member).roots
