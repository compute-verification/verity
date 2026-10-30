"""verity_vllm.program.predict.window_rows: a window's rows, every unit of `window_circuit.lower`'s class circuit given its
output rows from its slots' values. Pinned on `test_window_circuit`'s TINY class (2 slots, 4 prompt ids, 2 outputs: 5
positions a Program) at its grain, TINY's bf16 checkpoint on the H200 (FA3's `Attention_v4`, the class 4,32,4's
attention; its exact-int8 window waits on its norm's reader, `test_window_circuit_int8`), with random weights registered
under a committer's key and a scan state whose row at bucket 0 carries the two slots' requests.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import CLASS, GRAIN, SPLIT, SPLIT_GRAIN, state_row
from tests.program.predict.test_window_circuit_fa3 import H200
from verity_vllm.program.predict import scan_state as SS
from verity_vllm.program.predict import window_builder as WB
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_rows as WR
from verity_vllm.program.predict import window_weights as WW

from verity.core.protocols.pouw.audit import hm96_salt
from verity.core.protocols.sampled_proofs.one_stage import registered as CR
from verity.core.protocols.sampled_proofs.one_stage import registration as REG
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.commitments.gates import embedding as E
from verity.ml.definitions.registry import io_shell as IO
from verity.ml.windows import interface as IF

PROGRAM = "test/model"
LAW = {"law": "stratified", "k": 2}
RUN = "ab" * 64
#: the lanes of the scan state row both slots read (bucket 0's), each (occupancy, prompt length, max_tokens, ids): slot 0 a
#: prompt of 4 (its first output from position 3's select), slot 1 a prompt of 3, each keeping 2 outputs
LANES = [(1, 4, 2, [17, 3, 250, 9]), (1, 3, 2, [5, 77, 42, 0])]
#: each slot's row, its lane's (occupancy, prompt length, ids)
SLOTS = tuple([occ, lp, *ids] for occ, lp, _m, ids in LANES)
#: the window's scan state rows, a bucket each: bucket 0's the slots' lanes, every later one's another request in each lane
STATE = [state_row(CLASS.scan, LANES)] + [state_row(CLASS.scan, [(1, 1, 1, [99, 0, 0, 0])] * 2)] * (CLASS.buckets - 1)
#: the same window at another occupancy and other lengths: slot 0 empty, slot 1 a prompt of 2 keeping 1 output
OTHER_LANES = [(0, 0, 0, [0, 0, 0, 0]), (1, 2, 1, [5, 77, 0, 0])]
OTHER = tuple([occ, lp, *ids] for occ, lp, _m, ids in OTHER_LANES)
#: the nodes prefill computes at its last prompt position only: the final norm, the LM head and the token select
PRUNED = 3
#: the class Program's nodes a position
NODES = 24
EMBED = "model.embed_tokens.weight"
TABLE = WB.table_param(EMBED, 64)
#: the committer's secret key
KEY = bytes(range(32))


def _bf16(x) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _weights(fn) -> dict[str, np.ndarray]:
    """Finite random weights, each field's elements in their unsigned width: s8 codes, small positive f32 scales (an
    attention's scales 1), norms near 1, the rotary cache in [-1, 1]."""
    rng, out = np.random.default_rng(3), {}
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


def _window(config=S.TINY, grain=GRAIN):
    """The class circuit lowered, its weights, its registered values (each weight's parameter under `KEY`, and the scan
    state's, `STATE` committed as a stand-in for the network certifier's, in ``row/v2`` rows as the scan's units commit
    their state rows, as `Rows` and `Window` take them) and each slot's `Slot` of its lane's row (`scan_slots`)."""
    low = WC.lower(_registered(config), CLASS, None, grain, provable=False)
    fields = _weights(low.fn)
    state = CR.Registered.commit(PROGRAM, SS.SCAN_STATE, STATE, [hm96_salt(bytes(32), 0, j) for j in range(len(STATE))],
                                 schema=CR.SCHEMA_V2)
    values = {**WW.weight_values(_registered(config), fields, KEY, grain, keep_rows=True), SS.SCAN_STATE: state}

    def weights(label: str, lo: int, hi: int) -> np.ndarray:
        return fields[label][lo:hi]

    return low, fields, values, weights, [WR.Slot(low.fn, CLASS, row, weights) for row, _ in WR.scan_slots(low, STATE)]


@pytest.fixture(scope="module")
def window():
    return _window()


def _position(fn, family: str) -> list[int]:
    """Each position's node of ``family`` (its first)."""
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
    `TokenFeed_v1` carrying the slot on; the class Program's nodes are 5 positions of 24 (an embedding opens each)."""
    low, *_, slots = window
    sel = _position(low.fn, "TokenSelect")
    assert len(low.fn.body.nodes) == 120 and sorted(set(WR.positions(low.fn))) == list(range(5))
    for s, row in zip(slots, SLOTS):
        lp = row[1]
        picks = [int(s.values[j][0][0]) for j in sel]
        assert s.tokens == row[2:2 + lp] + picks[lp - 1:4] and all(0 <= t < 256 for t in picks)
        assert s.slots[0] == row[1:] and s.carried[0] == [lp - 1, *row[3:], 0]
        assert s.source == [WR.COMPUTED] * 120


def test_every_unit_holds_its_nodes_value_or_its_definitions_function(window):
    """Every unit is filled: an embedding read's row is the table's at its token and its opening (row, salt and path, as
    `gates.embedding.arguments` encodes them) opens to the table's root, its verdict 1; each token select's group units, ``x0``
    and combine give the node's token, every position's but the last's a row of its collector; each feed (every position
    but the first below the prompt cap) holds its token and carried slot; each slot's scan read every lane's first token and
    carried slot, occupancy word and head of the state row at its anchor (`io_shell.scan_reference`), its last collector
    (the next-to-last position's) the outputs so far from its prompt's last position on and its entry the read bit (its
    occupancy word's bit 0), the 2 outputs' count and ids; each head's first
    key block (`Attention_v20`'s cut) is evaluated, and its end's Definition on that block's state is the node's value, the
    row kernel's. With no served values every unit is computed or evaluated."""
    low, fields, tables, _w, slots = window
    rows = WR.Rows(low, slots, tables)
    assert {int(x) for v in rows.source.values() for x in v} == {WR.COMPUTED, WR.EVALUATED}
    emb = _position(low.fn, "Embedding")
    table = tables[TABLE]
    root, dom = table.root, table.domain.domain_id
    for P, s, row in zip(low.programs, slots, SLOTS):
        for p, j in enumerate(emb):
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
            assert (feed is None) == (p == 0 or p >= CLASS.prompt)
            if feed is not None:
                f = rows.outs[feed[0]]
                assert _i32(f[0][feed[1]]) == [t] and _i32(f[1][feed[1]]) == s.carried[p]
        for j, (units, x0) in P.selects.items():
            token = int(s.values[j][0][0])
            run, run_i = int(rows.outs[x0[0]][0][x0[1]][0]), 0
            for c, (ct, cu) in enumerate(units):
                bests, idxs = rows.outs[ct][0][cu].tolist(), _i32(rows.outs[ct][1][cu])
                assert len(bests) == len(idxs) == 64 // WC.GROUP
                for g, (best, idx) in enumerate(zip(bests, idxs)):
                    if _gt(best, run):
                        run, run_i = best, c * 64 + g * WC.GROUP + idx
            assert run_i == token and (j in P.outs) == (j != max(P.selects))
            if j in P.outs:
                (_, _, r), = P.outs[j]
                assert _i32(rows.outs[r.tid][0][r.unit]) == [token]
        st, su = P.scan_unit
        lanes = [[[r[2]], s_.carried[0], r[:1], [r[1], 2]] for r, s_ in zip(SLOTS, slots)]
        assert [_i32(rows.outs[st][k][su]) for k in range(8)] == [v for lane in lanes for v in lane]
        assert lanes == [[[tok], slot, [occ], head] for tok, slot, occ, head in WR._lanes(P, STATE, IO.scan_reference)]
        lp, picks = row[1], [int(s.values[j][0][0]) for j in P.selects]
        outs, (ct, cu), (et, eu) = picks[lp - 1:lp + 1], P.collects[-1], P.entry_unit
        assert len(P.collects) + 1 == len(picks) == CLASS.positions and len(outs) == CLASS.outputs
        n = min(CLASS.outputs, CLASS.positions - lp)
        assert _i32(rows.outs[ct][0][cu]) == [picks[-2]]
        assert _i32(rows.outs[ct][1][cu]) == [(lp - CLASS.positions) & 0xFFFFFFFF, 2, *[0] * (CLASS.outputs - n), *outs[:n]]
        assert sum(int(w) << 16 * k for k, w in enumerate(rows.outs[et][0][eu])) == 1 | 2 << 1 | outs[0] << 33 | outs[1] << 50
        assert len(P.heads) == 2 * (CLASS.prompt + CLASS.outputs - 1)
        for j, (plan, calls) in P.heads.items():
            (first, _), (end, _) = plan
            (ft, fu), (et, eu) = calls
            assert first.name == "AttnBlockFirst" and end.name == "AttnHeadEnd" and rows.source[ft][fu] == WR.EVALUATED
            l, o = (np.ascontiguousarray(rows.outs[ft][k][fu:fu + 4]).view("<u4") for k in (1, 2))
            node = np.asarray(evaluate_batch(end, l, o)).reshape(4, -1).astype(np.uint16)
            assert np.array_equal(node, rows.outs[et][0][eu:eu + 4])
            assert np.array_equal(node.reshape(-1), WR._words(s.values[j])) and node.any()


def test_a_slots_row_is_its_lanes_of_the_scan_state_row_at_its_anchor(window):
    """Each slot's row and head are its lane's of the scan state row its `ScanSlots_v2` reads, the one at its anchor
    (`scan_slots`; both of TINY's slots at bucket 0, lanes 0 and 1): slots given out of their lanes' order, or no scan
    state's value, are refused by name."""
    low, _f, values, _w, slots = window
    assert WR.scan_slots(low, STATE) == [(r, [r[1], 2]) for r in SLOTS]
    with pytest.raises(ValueError, match="class Program 0: its slot's row isn't lane 0's of the scan state row at bucket 0"):
        WR.Rows(low, slots[::-1], values)
    with pytest.raises(ValueError, match=re.escape(f"give the scan state's registered value, {SS.SCAN_STATE}, with its rows")):
        WR.Rows(low, slots, {p: v for p, v in values.items() if p != SS.SCAN_STATE})


def test_a_standin_scan_state_gives_each_slot_its_row_and_max_tokens(window):
    """Where no scan ran (`prove_path`'s window commit), `standin_state` is the scan state rows that give each slot its row
    and ``max_tokens`` at its anchor and lane, every other field zero: bucket 0's row is `STATE`'s, which holds the slots'
    lanes alone. It is committed under the committer's key at `STATE_NONCE` plus the window's index, a per-window value in
    its reading port's rows, and `Rows` takes it as the scan state."""
    low, _f, values, _w, slots = window
    v = WR.standin_state(PROGRAM, CLASS, [(r, m) for r, (_o, _lp, m, _ids) in zip(SLOTS, LANES)], KEY, 1, window=RUN)
    assert WR.scan_slots(low, v.rows) == WR.scan_slots(low, STATE) and v.rows[0] == tuple(STATE[0])
    assert (v.leaves, v.words, v.schema, v.window) == (CLASS.buckets, SS.words(CLASS.scan), CR.SCHEMA_V2, RUN)
    assert bytes(v.salts[0]) == hm96_salt(KEY, WR.STATE_NONCE + 1, 0) and WR.STATE_NONCE > max(WW.weight_nonces(values).values())
    WR.Rows(low, slots, {**values, SS.SCAN_STATE: v})


def test_a_chunks_best_is_its_first_maximum_above_minus_infinity():
    """`TokenSelectChunk_v2`'s scan from (-inf, 0): a NaN is never taken (not even first), equal maxima keep the first, and a
    chunk of no number above -inf is (-inf, 0)."""
    nan, inf, ninf = 0x7FC0, 0x7F80, 0xFF80
    assert WR.chunk_best(_bf16([1.0, 3.0, 3.0, 2.0])) == (int(_bf16(3.0)), 1)
    assert WR.chunk_best(np.array([nan, *_bf16([-2.0, -1.0])], dtype=np.uint16)) == (int(_bf16(-1.0)), 2)
    assert WR.chunk_best(np.array([ninf, nan, ninf], dtype=np.uint16)) == (WR.NEG_INF_BF16, 0)
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
    (``row-seg``), each committed under the committer's key at a nonce of its own: the table and a weight are core's commits
    of those rows under the weights record's program and `audit.hm96_salt`'s salts at `window_weights.weight_nonces`' (the
    registration's roots, `weights_registration`), and the entries name nothing but each value's size, domain and hiding
    root. The scan state is no value of the committer's but the network certifier's, a row a bucket of the window. Each
    value is in its reading ports' rows, which a window's commit refuses otherwise. Another parameter is refused."""
    low, fields, got, *_ = window
    assert set(got) == set(low.circuit.registered.params)
    reads = low.circuit.registered.reads
    lays = {r.reader: dict(IF.layout(low.circuit.template(r.reader)).schemas) for r in reads}
    assert {(r.param, got[r.param].schema) for r in reads} == {(r.param, lays[r.reader][r.port]) for r in reads}
    t = got[TABLE]
    assert (t.leaves, t.words, t.schema, t.window) == (256, 64, CR.SCHEMA_SEG, None) and t.schema == E.SCHEMA
    assert np.array_equal(t.rows[:].reshape(-1), fields[EMBED])
    nonces, program = WW.weight_nonces(got), WC.class_program(_registered(), WW.WEIGHTS_CLASS).digest
    assert set(nonces) == set(got) - {SS.SCAN_STATE} and {v.program for p, v in got.items() if p in nonces} == {program}
    assert (got[SS.SCAN_STATE].leaves, got[SS.SCAN_STATE].words) == (CLASS.buckets, SS.words(CLASS.scan))
    for name, rows, schema in ((TABLE, fields[EMBED].reshape(-1, 64), CR.SCHEMA_SEG),
                               (WB.weight_param(EMBED, 1024), fields[EMBED].reshape(-1, 1024), CR.SCHEMA_V2)):
        salts = [hm96_salt(KEY, nonces[name], j) for j in range(len(rows))]
        ref = CR.Registered.commit(program, name, rows.tolist(), salts, schema=schema)
        assert got[name].entry() == ref.entry() and got[name].levels == ref.levels and got[name].schema == schema
        assert set(got[name].entry()) == {"value", "root", "leaves", "words", "domain"}
    assert WR.index_salt(KEY, 3) == hm96_salt(KEY, WR.INDEX_NONCE + 3, 0) and WR.INDEX_NONCE >= WR.STATE_NONCE + (1 << 32)
    assert len({bytes(v.salts[0]) for v in got.values()} | {WR.index_salt(KEY, 0)}) == len(got) + 1
    with pytest.raises(ValueError, match="a window's index is a u32"):
        WR.index_salt(KEY, 1 << 32)
    down = got[WB.weight_param("model.layers.0.mlp.down_proj.weight", 1024)].rows[:]
    assert down.shape == (8, 1024) and np.array_equal(down.reshape(-1), fields["model.layers.0.mlp.down_proj.weight"])
    with pytest.raises(ValueError, match="'weights/x@3/rows' is no weight parameter"):
        WW.weight_value(program, "weights/x@3/rows", np.zeros(3, np.uint16), KEY, 0)


def _as_leaves(x: np.ndarray, width: int, leaves: int) -> np.ndarray:
    """A port's ``width``-bit elements as its Definition's ``leaves``: the elements themselves, or each element's bits, least
    significant first, where the port's leaves are bits (`ScanSlots_v2`'s state words)."""
    if x.shape[1] == leaves:
        return x
    return ((x.astype(np.uint64)[:, :, None] >> np.arange(width, dtype=np.uint64)) & 1).reshape(len(x), -1)


def _packed(bits: np.ndarray, width: int) -> np.ndarray:
    """``(n, k · width)`` bits as ``k`` ``width``-bit elements, least significant first (`_as_leaves`' inverse)."""
    return (bits.reshape(len(bits), -1, width) << np.arange(width, dtype=np.uint64)).sum(axis=2)


def test_a_bit_leaf_ports_elements_are_its_bits_least_significant_first():
    """`_as_leaves` and `_packed` against `ScanSlots_v2` and `SlotEntry_v1`, whose ports are words of bit leaves: a state row
    gives `scan_reference`'s lanes through them, and an entry's bits pack to `entry_reference`'s words."""
    scan = SS.bridge(CLASS.scan, CLASS.prompt)
    row = np.asarray([state_row(CLASS.scan, LANES)], dtype=np.uint64)
    (_, t), = scan.signature()[0]
    out = np.asarray(evaluate_batch(scan, _as_leaves(row, 16, t.leaves), kernel="reference")).reshape(1, -1).astype(np.uint64)
    b = scan.bindings
    ref = IO.scan_reference(row[0].tolist(), b["NP"], b["IDB"], b["K0"], b["K"], b["OCC"], b["LP"], b["MAX"], b["IDS"])
    assert scan.name == "ScanSlots" and _packed(out, 32)[0].tolist() == [v for k, s, o, h in ref for v in (k, *s, o, *h)]
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
    win = IF.Window(low.circuit, RUN, LAW, 64, registered=values, forms=WR.FORMS)
    win.supply(CR)
    rows.record(win)
    bad = {}
    for fn, n in low.circuit.templates:
        tid = descriptor_id(fn)
        lay = IF.layout(fn)
        ins = win.input_words(tid, 0, n)
        if fn.name == "EmbeddingRead":
            t = ins[0].view("<u4").reshape(-1)
            for u in range(n):
                path = np.ascontiguousarray(ins[5][u]).view("<u8").astype(">u8").tobytes()
                h, ok = E.reference(256, int(t[u]), *(np.ascontiguousarray(ins[g][u]).view("<u8").astype(">u8").tobytes()
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
        if (u := np.flatnonzero((got != want).any(axis=1))).size:
            bad[tid] = u.tolist()
    assert bad == {}
    c = win.close(salt_key=bytes(32))
    assert c is not None
    labels = win.labels()
    seen = {lab["form"] for m in labels["members"] for *_, lab in m["labels"]}
    assert seen == set(WR.FORMS) and all(m["labels"] for m in labels["members"])
    assert {int(x) for v in rows.source.values() for x in v} == {WR.SERVED, WR.COMPUTED, WR.EVALUATED}


def test_a_cut_norms_parts_hold_its_sum_and_its_tail_the_nodes_value():
    """At a class whose norm after down_proj is cut at its residual add (`test_window_circuit.SPLIT`), each add part holds its
    words of the node's sum, which is `RMSNormFusedCudaAdd_v1` on the node's x and res, and the tail the node's value, which
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
    assert seen == 2 * 2 * CLASS.positions


class _Runs:
    """A window's `record` as `Rows.record` calls it, keeping each run's template, unit count and label."""

    def __init__(self) -> None:
        self.runs: list[tuple[str, int, Any]] = []

    def record(self, fn, ins, outs, label) -> None:
        self.runs.append((descriptor_id(fn), len(outs[0]), label))


def _runs(rows) -> list[tuple[str, int, Any]]:
    got = _Runs()
    rows.record(got)
    return got.runs


def _two(window) -> dict[str, tuple[Any, list[WR.Slot]]]:
    """The window at two occupancies, each its stand-in scan state (`standin_state`, window 0's under `KEY`) and slots:
    `LANES` with slot 0 served (its request's values), and `OTHER_LANES`, every node computed."""
    low, _f, _v, weights, slots = window
    rows_, store = _served(low, slots[0], T=1)
    s0 = WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(rows_, *_read(store)))
    state = {k: WR.standin_state(PROGRAM, CLASS, [([occ, lp, *ids], m) for occ, lp, m, ids in lanes], KEY, 0, window=RUN)
             for k, lanes in (("served", LANES), ("other", OTHER_LANES))}
    return {"served": (state["served"], [s0, slots[1]]),
            "other": (state["other"], [WR.Slot(low.fn, CLASS, r, weights) for r in OTHER])}


def test_a_windows_labels_are_its_class_structure_alone(window):
    """`Rows.record` labels each run by its step, slot and form, all three the lowering's, so the window at two
    occupancies (slot 0 served at a prompt of 4 and slot 1 at 3, each keeping 2 outputs, or slot 0 empty and slot 1 at 2
    keeping 1) records the same runs under byte-identical labels, though its rows and its units' sources differ. Every
    unit is in one run, and both forms and both slots are there."""
    low, _f, values, *_ = window
    rows = {k: WR.Rows(low, ss, {**values, SS.SCAN_STATE: st}) for k, (st, ss) in _two(window).items()}
    a, b = rows["served"], rows["other"]
    assert any(not np.array_equal(a.source[t], b.source[t]) for t in a.source)
    assert any(not np.array_equal(x, y) for t in a.outs for x, y in zip(a.outs[t], b.outs[t]))
    runs = _runs(a)
    assert json.dumps(runs) == json.dumps(_runs(b))
    assert all(set(lab) == {"step", "slot", "form"} for *_, lab in runs)
    assert {lab["form"] for *_, lab in runs} == set(WR.FORMS) and {lab["slot"] for *_, lab in runs} >= {0, 1}
    assert {t: sum(n for u, n, _ in runs if u == t) for t in a.n} == a.n


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


def _public(c, d: Path) -> dict[str, Any]:
    """Window dir ``d``'s public files, each hiding commitment and digest over one replaced by its role (`HIDING`; a public
    body's ``b || c`` strings by their length, its table refs kept), each prover-only file by its name once it is its
    owner's alone."""
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
def test_two_windows_differing_in_occupancy_and_lengths_have_one_public_record(window, tmp_path):
    """The window at two occupancies (`_two`), each closed and written by `Window.write` as `prove_path`'s window commit
    writes it (its stand-in scan state a value of the window's, its class and index public, its index salt window 0's
    under `KEY`), has one public record: every public file is byte for byte the other's once each hiding commitment (a
    root, a row's ``b || c``, a registered read's path) and each digest over one (a public body's, the leaf layer, the
    registration's) is replaced by its role. The labels, the index's hiding leaf, the unit files, the table refs, the
    domains and the registration's provenance all agree, and the index and its salt are prover only. What the labels test
    can't reach: everything else `Window.write` makes public. Slow (minutes) for the two windows' closes, as the test
    above."""
    low, _f, values, *_ = window
    pub = {}
    for name, (state, slots) in _two(window).items():
        vals = {**values, SS.SCAN_STATE: state}
        win = IF.Window(low.circuit, RUN, LAW, 64, registered=vals, forms=WR.FORMS)
        win.supply(CR)
        WR.Rows(low, slots, vals).record(win)
        c = win.close()
        prov = {"window_circuit": {"class": [CLASS.slots, CLASS.prompt, CLASS.outputs, CLASS.lanes, CLASS.every], "window": 0},
                "activations": {"program": WC.NAME, "values": {SS.SCAN_STATE: state.entry()}}}
        win.write(c, tmp_path / name, prov, registrar=REG, index_salt=WR.index_salt(KEY, 0))
        pub[name] = _public(c, tmp_path / name)
    assert pub["served"] == pub["other"]
    assert {"index.json", "index-salt.bin", "circuit.json", "registration.json"} <= set(pub["served"])
