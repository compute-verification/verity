"""verity_vllm.program.predict.window_rows: a window's rows, every unit of `window_circuit.lower`'s class circuit given its
output rows from its slots' values. Pinned on `test_window_circuit`'s TINY class (2 slots, 4 prompt ids, 2 outputs: 5
positions a Program) at its grain, TINY's exact-int8 checkpoint on the H200 (FA3's `Attention_v4`, the class 4,32,4's
attention), with random weights registered under a committer's key.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import CLASS, GRAIN
from tests.program.predict.test_window_circuit_fa3 import H200
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict import window_builder as WB
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_rows as WR

from verity.core.protocols.pouw.audit import hm96_salt
from verity.core.protocols.sampled_proofs.one_stage import registered as CR
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.commitments.gates import embedding as E
from verity.ml.windows import interface as IF

PROGRAM = "test/model"
LAW = {"law": "stratified", "k": 2}
RUN = "ab" * 64
#: slot 0 a prompt of 4 (its last output from position 3's select), slot 1 a prompt of 3
SLOTS = ([1, 4, 17, 3, 250, 9], [1, 3, 5, 77, 42, 0])
#: the nodes prefill computes at its last prompt position only: the final norm, the LM head and the token select
PRUNED = 3
EMBED = "model.embed_tokens.weight"
TABLE = WB.table_param(EMBED, 64)
#: the committer's secret key, and the slot rows' salt nonce past every weight parameter's
KEY, SLOT_NONCE = bytes(range(32)), 1 << 20


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


@pytest.fixture(scope="module")
def window():
    """The class circuit lowered, its weights, its registered values under `KEY` (each weight's parameter and the slot
    rows, as `Rows` and `Window` take them) and each slot's `Slot`."""
    phi = S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": {**S.TINY, "quantization_config": dict(INT8_EXACT_QUANT)}},
                target={**H200, "int8_exact": "linears"})
    low = WC.lower(S.Registered.of(phi, default_seed=7), CLASS, None, GRAIN)
    fields = _weights(low.fn)
    values = {**WR.weight_values(low.circuit, PROGRAM, fields.__getitem__, KEY),
              WC.SLOT_ROWS: WR.slot_rows_value(PROGRAM, SLOTS, KEY, SLOT_NONCE, window=RUN)}

    def weights(label: str, lo: int, hi: int) -> np.ndarray:
        return fields[label][lo:hi]

    return low, fields, values, weights, [WR.Slot(low.fn, CLASS, row, weights) for row in SLOTS]


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
    `TokenFeed_v1` carrying the slot on; the class Program's nodes are 5 positions of 32 (an embedding opens each)."""
    low, *_, slots = window
    sel = _position(low.fn, "TokenSelect")
    assert len(low.fn.body.nodes) == 160 and sorted(set(WR.positions(low.fn))) == list(range(5))
    for s, row in zip(slots, SLOTS):
        lp = row[1]
        picks = [int(s.values[j][0][0]) for j in sel]
        assert s.tokens == row[2:2 + lp] + picks[lp - 1:4] and all(0 <= t < 256 for t in picks)
        assert s.slots[0] == row[1:] and s.carried[0] == [lp - 1, *row[3:], 0]
        assert s.source == [WR.COMPUTED] * 160


def test_every_unit_holds_its_nodes_value_or_its_definitions_function(window):
    """Every unit is filled: an embedding read's row is the table's at its token and its opening (row, salt and path, as
    `gates.embedding.arguments` encodes them) opens to the table's root, its verdict 1; each token select's chunks, ``x0``
    and combine give the node's token, and its mask the token for an occupied slot; each feed holds its token and carried
    slot, each slot row its row opened. With no served values every unit is computed or evaluated."""
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
            feed, slot_row = P.tokens[p]
            assert (feed is None) == (p >= CLASS.prompt) and (slot_row is None) == (p > 0)
            if feed is not None:
                f = rows.outs[feed[0]]
                assert _i32(f[0][feed[1]]) == [t] and _i32(f[1][feed[1]]) == s.carried[p]
            if slot_row is not None:
                assert _i32(rows.outs[slot_row[0]][0][slot_row[1]]) == row[:1]
                assert _i32(rows.outs[slot_row[0]][1][slot_row[1]]) == row[1:]
        for j, (chunks, x0, mask) in P.selects.items():
            token = int(s.values[j][0][0])
            run, run_i = int(rows.outs[x0[0]][0][x0[1]][0]), 0
            for c, (ct, cu) in enumerate(chunks):
                best, idx = int(rows.outs[ct][0][cu][0]), _i32(rows.outs[ct][1][cu])[0]
                if _gt(best, run):
                    run, run_i = best, c * 64 + idx
            assert run_i == token
            assert _i32(rows.outs[mask[0]][0][mask[1]]) == [token]
            (_, _, r), = P.outs[j]
            assert _i32(rows.outs[r.tid][0][r.unit]) == [token]


def test_a_chunks_best_is_its_first_maximum_above_minus_infinity():
    """`TokenSelectChunk_v2`'s scan from (-inf, 0): a NaN is never taken (not even first), equal maxima keep the first, and a
    chunk of no number above -inf is (-inf, 0)."""
    nan, inf, ninf = 0x7FC0, 0x7F80, 0xFF80
    assert WR.chunk_best(_bf16([1.0, 3.0, 3.0, 2.0])) == (int(_bf16(3.0)), 1)
    assert WR.chunk_best(np.array([nan, *_bf16([-2.0, -1.0])], dtype=np.uint16)) == (int(_bf16(-1.0)), 2)
    assert WR.chunk_best(np.array([ninf, nan, ninf], dtype=np.uint16)) == (WR.NEG_INF_BF16, 0)
    assert WR.chunk_best(np.array([ninf, inf], dtype=np.uint16)) == (inf, 1)


def _served(low, s, T: int, pruned: int = PRUNED):
    """Slot ``s``'s values as a request's served rows (`Served`): its prompt's positions in prefill, but for the last
    ``pruned`` nodes of a position at its last prompt position only, and ``T`` decode steps, each member's elements by row
    under op ``op<k>``."""
    nodes, pos, lp = low.fn.body.nodes, WR.positions(low.fn), s.row[1]
    K = len(nodes) // (max(pos) + 1)
    rows, store = [], {}
    for k in range(K):
        fields = WR._fields(nodes[k].fn)
        names = ["out"] if len(fields) == 1 else [str(i) for i in range(len(fields))]
        members = [(m, f.leaves, "x") for m, f in zip(names, fields)]
        prefill = [lp - 1] if k >= K - pruned else list(range(lp))
        for step, ps in [(0, prefill), *((d, [lp - 1 + d]) for d in range(1, T + 1))]:
            for r, p in enumerate(ps):
                rows.append({"spec": descriptor_id(nodes[p * K + k].fn), "step": step, "op_path": f"op{k}", "members": members,
                             "row": r})
            for i, m in enumerate(names):
                store[(step, f"op{k}", m)] = np.concatenate([s.values[p * K + k][i] for p in ps])
    return rows, store


def _read(store):
    def read(step, op, m, lo, hi):
        a = store.get((step, op, m))
        return (None, f"no {op}/{m} at step {step}") if a is None else (a[lo:hi], "")
    return read


def test_a_served_value_is_the_nodes_value_and_the_rest_are_computed(window):
    """A request's served rows give each node the engine computed its value: every prompt position's, the pruned nodes at
    the last prompt position's only, and its decode steps'; every other node's value is computed from them. A served value
    is taken as served (a changed row stays changed), an embedding that isn't the table's row and a select that isn't its
    logits' first maximum are refused by name, as is a row of another Definition."""
    low, _f, tables, weights, slots = window
    s0 = slots[0]
    rows, store = _served(low, s0, T=1)
    served = WR.Served(rows, _read(store))
    assert served.LP == 4 and len(served.prefill) == 32 and len(served.decode) == 1
    got = WR.Slot(low.fn, CLASS, SLOTS[0], weights, served)
    K = 32
    want = [WR.SERVED if p == 4 or k < K - PRUNED or p == 3 else WR.COMPUTED for p in range(5) for k in range(K)]
    assert got.source == want
    assert all(np.array_equal(a, b) for x, y in zip(got.values, s0.values) for a, b in zip(x, y)) and got.tokens == s0.tokens
    rows_ = WR.Rows(low, [got, slots[1]], tables)
    (_, _, r), = low.programs[0].outs[K + 1]
    assert rows_.source[r.tid][r.unit] == WR.SERVED

    norm = dict(store)
    norm[(0, "op1", "out")] = norm[(0, "op1", "out")] ^ np.uint16(1)
    changed = WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(rows, _read(norm)))
    assert np.array_equal(changed.values[1][0], store[(0, "op1", "out")][:64] ^ np.uint16(1))

    bad = dict(store)
    bad[(0, "op0", "out")] = bad[(0, "op0", "out")] ^ np.uint16(1)
    with pytest.raises(ValueError, match="node 0 .* at position 0: served as .*not the table's row at its token"):
        WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(rows, _read(bad)))
    pick = dict(store)
    pick[(1, f"op{K - 1}", "out")] = (pick[(1, f"op{K - 1}", "out")] + 1) % 256
    with pytest.raises(ValueError, match="at position 4: served as .*not the first maximum of its logits"):
        WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(rows, _read(pick)))
    other = [{**d, "spec": "Gemm_v2{K=1,N=1}"} if (d["op_path"], d["step"]) == ("op2", 1) else d for d in rows]
    with pytest.raises(ValueError, match="position 4's node 2 is .*served as Gemm_v2"):
        WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(other, _read(store)))
    with pytest.raises(ValueError, match="the served request's 4 prompt ids, its slot row's 3"):
        WR.Slot(low.fn, CLASS, SLOTS[1], weights, served)
    swapped = [d for d in rows if d["step"] == 0] + sorted((d for d in rows if d["step"] == 1), key=lambda d: d["op_path"])
    with pytest.raises(ValueError, match="step 1's 32 rows are not the prefill's 32 ops in its order"):
        WR.Served(swapped, _read(store))


def test_the_registered_values_are_each_weights_rows_and_the_slot_rows(window):
    """Each weight's parameter is its words in rows of its reading port's (``row/v2``), the table's its V rows of H
    (``row-seg``), and the slot rows the slots' rows of i32s under the window's root, each committed under the committer's
    key at a nonce of its own: the table and a weight are core's commits of those rows under `audit.hm96_salt`'s salts,
    and the entries name nothing but each value's size, domain and hiding root. Another parameter is refused."""
    low, fields, got, *_ = window
    assert set(got) == set(low.circuit.registered.params)
    t = got[TABLE]
    assert (t.leaves, t.words, t.schema, t.window) == (256, 64, CR.SCHEMA_SEG, None) and t.schema == E.SCHEMA
    assert np.array_equal(t.rows[:].reshape(-1), fields[EMBED])
    names = sorted(n for n in got if n != WC.SLOT_ROWS)
    for name, rows, schema in ((TABLE, fields[EMBED].reshape(-1, 64), CR.SCHEMA_SEG),
                               (WB.weight_param(EMBED, 512), fields[EMBED].reshape(-1, 512), CR.SCHEMA_V2)):
        salts = [hm96_salt(KEY, names.index(name), j) for j in range(len(rows))]
        ref = CR.Registered.commit(PROGRAM, name, rows.tolist(), salts, schema=schema)
        assert got[name].entry() == ref.entry() and got[name].levels == ref.levels and got[name].schema == schema
        assert set(got[name].entry()) == {"value", "root", "leaves", "words", "domain"}
    s = got[WC.SLOT_ROWS]
    assert (s.leaves, s.words, s.window, s.schema) == (2, 12, RUN, CR.SCHEMA_V2) and _i32(s.rows[:][1]) == SLOTS[1]
    assert len({bytes(v.salts[0]) for v in got.values()}) == len(got)
    s8 = got[WB.weight_param("model.layers.0.mlp.down_proj.weight", 512)].rows[:]
    assert s8.shape == (8, 512) and np.array_equal(s8.reshape(-1).view(np.uint8), fields["model.layers.0.mlp.down_proj.weight"])
    odd = SimpleNamespace(registered=SimpleNamespace(params={"weights/x@3/rows": 3}))
    with pytest.raises(ValueError, match="no weight's registered parameter nor the slot rows"):
        WR.weight_values(odd, PROGRAM, fields.__getitem__, KEY)


@pytest.mark.slow
def test_a_windows_rows_are_what_its_definitions_compute_on_its_committed_inputs(window):
    """The rows recorded into a `Window` (its registered values under the committer's key, the tables' supplied) are
    every unit's Definition on the inputs the window commits: its wired rows, its registered reads' rows and its openings',
    each word template's outputs under the reference evaluator, each embedding read's by `gates.embedding.reference`; the
    window closes over them, labelled by step and source. What no cheaper test reaches: the circuit's own wires and reads
    resolved by the window, against every Definition. Slow (minutes) for the window's members, which trace the embedding
    read's population Program, and the reference evaluator on every unit."""
    low, _f, values, weights, slots = window
    rows_, store = _served(low, slots[0], T=1)
    s0 = WR.Slot(low.fn, CLASS, SLOTS[0], weights, WR.Served(rows_, _read(store)))
    rows = WR.Rows(low, [s0, slots[1]], values)
    win = IF.Window(low.circuit, RUN, LAW, 64, registered=values)
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
        args = [np.ascontiguousarray(x).view(f"<u{w // 8}").reshape(n, -1) for x, w in zip(ins, lay.widths)]
        want = np.asarray(evaluate_batch(fn, *args, kernel="reference")).reshape(n, -1).astype(np.uint64)
        got = np.concatenate([np.ascontiguousarray(o).view(f"<u{w // 8}").reshape(n, -1).astype(np.uint64)
                              for o, w in zip(rows.outs[tid], lay.widths[len(lay.rows):])], axis=1)
        if (u := np.flatnonzero((got != want).any(axis=1))).size:
            bad[tid] = u.tolist()
    assert bad == {}
    c = win.close(salt_key=bytes(32))
    assert c is not None
    labels = win.labels()
    seen = {lab["source"] for m in labels["members"] for *_, lab in m["labels"]}
    assert seen == {"served", "computed", "evaluated"} and all(m["labels"] for m in labels["members"])
