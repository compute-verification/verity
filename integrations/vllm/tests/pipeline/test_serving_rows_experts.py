"""A MoE layer served cut per expert (`pipeline.serving_rows`' ``slots``, ``read`` and ``expert`` layouts, `expert_rows`): two
tokens' router rows and their four (token, slot) pairs of one ``MoeExpertFp8Gemm_v1`` layer at E = 8, K = 64, N = 4, G = KB = 32,
in blocks of two rows, and a window of the router's slots, the reads of the blocks' rows and scale rows and the block GEMMs:

* the units are laid out and cut as the module docstring says, each read's key wired to its token's router slot and each block
  GEMM's rows to the reads of its block;
* the members' words are each template's reference on its rows, each read's row, salt and path open its table's committed root at
  the pair's expert, and the block GEMMs' outputs are the pairs' served words;
* the reads' roots, domain ids and verdicts are committed by reference to the tables' supplied values, and a listed record of the
  window counts the tables' reads registered exactly where their roots are pinned.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from verity_flock.class_statement import UnitStager
from verity.core.protocols.sampled_proofs.one_stage import registered as G1
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.prims import HopperE4m3MmaV2Dot32 as MV
from verity.ml.commitments.gates import embedding as EMB
from verity.ml.definitions.numerics import fp8_moe_rows as FR
from verity.ml.definitions.registry import boolean_fp8_moe as BM
from verity.ml.definitions.registry import boolean_moe as BMOE
from verity.ml.definitions.registry import fp8_moe_experts as X
from verity.ml.definitions.registry import moe as MOE
from verity.ml.windows import interface as IF
from verity.ml.windows import rows as CS
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import expert_rows as EX
from verity_vllm.pipeline import expert_tables as ET
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import weight_rows as WT
from verity_vllm.query.program_view import spec_statics

E, TOPK, K, N, G, KB, NR, T = 8, 2, 64, 4, 32, 32, 2, 2
NB, KG, B, P = -(-N // G), K // G, N // NR, T * TOPK
ROUTER = descriptor_id(bind(MOE.MoeRouterSigmoidBias, E=E, TOPK=TOPK, SCALE=2.5, KERNEL="block"))
GEMM = descriptor_id(bind(X.MoeExpertFp8Gemm, E=E, K=K, N=N, G=G, KB=KB, SWAP=False, DOT=MV))
SLOTS = descriptor_id(bind(BMOE.MoeRouterSigmoidBiasSlots, E=E, TOPK=TOPK, SCALE=2.5, KERNEL="block"))
BLOCK = descriptor_id(bind(BM.MoeExpertFp8BlockGemmV4, K=K, N=NR, G=G, KB=KB, SWAP=False, DOT=BG.HopperE4m3MmaV2Dot32))
READ_W, READ_WS = descriptor_id(EMB.read(E, NR * K // 2)), descriptor_id(EMB.read(E, 2 * K // G))
TMPLS = [SLOTS, READ_W, READ_WS, BLOCK]
CAPS = {SLOTS: 4, READ_W: 4, READ_WS: 4, BLOCK: 3}
LAYER = "model.layers.3.mlp"
W, WS = f"{LAYER}.experts.w13_weight", f"{LAYER}.experts.w13_weight_scale_inv"
COMMITTER = {"key": bytes(range(32)), "program": "ab" * 64}


def _e4m3(rng: np.random.Generator, *shape: int) -> np.ndarray:
    b = rng.integers(0, 256, shape, dtype=np.uint8)
    b[(b & 0x7F) == 0x7F] ^= 1
    return b


def _f32(rng: np.random.Generator, lo: float, hi: float, *shape: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(lo), np.log(hi), shape)).astype(np.float32).view(np.uint32)


class Index:
    """The rows' Program: its ``weights`` struct of the layer's expert weights and scales, and its router rows' outputs, the
    weights ``[0, TOPK)`` and the ids ``[TOPK, 2 TOPK)`` of each."""
    pnames = ["weights"]
    fields = [(W, E * N * K, [E, N, K], 8), (WS, E * NB * KG, [E, NB, KG], 32)]

    def weight_slice(self, lo: int, hi: int):
        at = {(0, E * N * K): (W, [E, N, K]), (E * N * K, E * N * K + E * NB * KG): (WS, [E, NB, KG])}.get((lo, hi))
        return None if at is None else (at[0], 0, hi - lo, at[1])

    def weight_bits(self, label: str) -> int:
        return 8 if label == W else 32

    def member_at(self, j: int, lo: int, hi: int):
        d = {"op_path": f"{LAYER}.gate"}
        return (d, "topk_ids", lo - TOPK, hi - TOPK, "i32") if lo >= TOPK else (d, "topk_weights", lo, hi, "f32")


@pytest.fixture
def layer(monkeypatch):
    """The two tokens' router rows and their pairs' rows (token-major, each token's slots in order), each operand by its argument
    group, and the store of their served outputs: the router's on its logits and bias, each pair's its twin on its expert's rows."""
    rng = np.random.default_rng(0xC185)
    pi = Index()
    w, ws = _e4m3(rng, E * N * K), _f32(rng, 1e-4, 1e-2, E * NB * KG)
    ops = {json.dumps([["p", 0, 0, w.size]]): w, json.dumps([["p", 0, w.size, w.size + ws.size]]): ws}
    words = {"topk_weights": [], "topk_ids": [], "out": []}
    routers, pairs, experts = [], [], []
    bias = (rng.standard_normal(E) * 1e-2).astype(np.float32).view(np.uint32)
    ops[json.dumps([["x", 90, 0, E]])] = bias
    for t in range(T):
        logits = (rng.standard_normal(E) * 2).astype(np.float32).view(np.uint32)
        ops[json.dumps([["x", 80 + t, 0, E]])] = logits
        slots = evaluate(bind(MOE.MoeRouterSigmoidBiasSlots, E=E, TOPK=TOPK, SCALE=2.5, KERNEL="block"), logits.tolist(), bias.tolist())
        words["topk_weights"] += slots[:TOPK]
        words["topk_ids"] += slots[TOPK:]
        r = {"i": 10 + t, "spec": ROUTER, "args": [[["x", 80 + t, 0, E]], [["x", 90, 0, E]]]}
        routers.append(("r0", pi, r, {"statics": spec_statics(ROUTER), "op_path": f"{LAYER}.gate", "row": t, "step": 5,
                                      "members": [("topk_weights", TOPK, "f32"), ("topk_ids", TOPK, "i32")]}))
        for k in range(TOPK):
            p, e = len(pairs), int(slots[TOPK + k])
            xq, sx = _e4m3(rng, K), _f32(rng, 1e-3, 1e-1, KG)
            args = [[["x", 100 + p, 0, K]], [["x", 200 + p, 0, KG]], [["n", 10 + t, TOPK + k, TOPK + k + 1]],
                    [["p", 0, 0, w.size]], [["p", 0, w.size, w.size + ws.size]]]
            ops.update({json.dumps(args[0]): xq, json.dumps(args[1]): sx, json.dumps(args[2]): np.array([e], np.int32)})
            y, nonfinite = FR.moe_expert_fp8_block_row(xq, sx, w.reshape(E, N, K)[e], ws.reshape(E, NB, KG)[e], G=G, KB=KB,
                                                       dot="mmav2")
            assert not nonfinite.any()
            words["out"] += y.tolist()
            pairs.append(("r0", pi, {"i": 20 + p, "spec": GEMM, "args": args},
                          {"statics": spec_statics(GEMM), "op_path": f"{LAYER}.experts", "row": p, "step": 7,
                           "members": [("out", N, "bf16")]}))
            experts.append(e)
    assert len(set(experts)) > 1
    dtype = {"topk_weights": np.uint32, "topk_ids": np.int32, "out": np.uint16}
    held = {m: np.array(v, dtype=np.uint64).astype(dtype[m]) for m, v in words.items()}

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            return held[member][lo:hi], None

    monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, g, bits=None: [ops[json.dumps(g)]])
    rows = [routers if t == SLOTS else pairs for t in TMPLS]
    return SimpleNamespace(rows=rows, store=Store(), w=w, ws=ws, experts=experts, out=held["out"], ops=ops)


def cut(rows):
    plan = EX.Plan(TMPLS, rows)
    return plan, EX.wire(EX.cut(SRW.members(TMPLS, rows, None, CAPS), plan), plan)


def committer(plan):
    params = [param for t in plan.what if plan.what[t][0] != "gemm" for _at, _n, _b, param, _l in plan.tables(t)]
    return {**COMMITTER, "nonces": ET.nonces(params, 0)}


def stager(t: str) -> UnitStager:
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = SRW.template(t).fn, "experts-test", {}, CR.unit_rows(SRW.template(t))[0]
    return st


def test_a_pair_row_holds_its_blocks_and_reads_and_a_router_row_its_slots():
    hits = dict(SRW.unit_templates(GEMM))
    words = {t: SRW.template(t).word for t in TMPLS}
    assert hits[words[BLOCK]] == B and hits[words[READ_W]] == B and hits[words[READ_WS]] == NB
    assert dict(SRW.unit_templates(ROUTER))[words[SLOTS]] == 1
    assert [SRW.template(t).layout for t in TMPLS] == ["slots", "read", "read", "expert"]
    assert [EX.kind(t, spec_statics(GEMM)) for t in TMPLS] == [("slots", 1), ("w", NR), ("ws", G), ("gemm", NR)]


def test_the_units_are_cut_by_table_and_slot_and_wired_to_the_router_and_the_reads(layer):
    plan, ms = cut(layer.rows)
    assert {t: sum(m.n for m in ms if m.template == t) for t in TMPLS} == {SLOTS: T, READ_W: P * B, READ_WS: P * NB, BLOCK: P * B}
    base = {t: min(m.base for m in ms if m.template == t) for t in TMPLS}
    pair = {(20 + TOPK * t + k): (t, k) for t in range(T) for k in range(TOPK)}
    # a read's units are table-major, ((l B + b) TOPK + k) T + t; a block GEMM's slot-major, ((l TOPK + k) T + t) B + b
    for t, order in ((READ_W, lambda tok, k, b: (b * TOPK + k) * T + tok), (READ_WS, lambda tok, k, b: k * T + tok),
                     (BLOCK, lambda tok, k, b: (k * T + tok) * B + b)):
        got = {u: (pair[plan.key(t, u)[1]], b) for u, (_p, b) in enumerate(plan.units[t])}
        assert all(order(tok, k, b) == u for u, ((tok, k), b) in got.items())
    assert [(m.n, m.base - base[m.template]) for m in ms if m.template == READ_W] == [(T, j * T) for j in range(B * TOPK)]
    assert [(m.n, m.base - base[m.template]) for m in ms if m.template == BLOCK] == [(3, 0), (1, 3), (3, 4), (1, 7)]
    for m in ms:
        wired = {port: (out, writers) for port, out, writers in m.wired}
        if m.template == SLOTS:
            assert wired == {}
            continue
        units = range(m.base - base[m.template], m.base - base[m.template] + m.n)
        keys = [(pair[plan.key(m.template, u)[1]], plan.units[m.template][u][1]) for u in units]
        if m.template in (READ_W, READ_WS):
            (out, writers), = wired.values()
            assert list(wired) == ["in0"] and out == f"o{TOPK + keys[0][0][1]}"
            assert list(writers) == [base[SLOTS] + tok for (tok, _k), _b in keys]
            continue
        assert sorted(wired) == ["in2", "in3"] and wired["in2"][0] == wired["in3"][0] == "o0"
        assert list(wired["in2"][1]) == [base[READ_W] + (b * TOPK + k) * T + tok for (tok, k), b in keys]
        assert list(wired["in3"][1]) == [base[READ_WS] + (b * NR // G * TOPK + k) * T + tok for (tok, k), b in keys]


def test_the_members_are_their_references_the_reads_open_and_the_blocks_are_the_served_words(layer):
    plan, ms = cut(layer.rows)
    tables = EX.commit_tables(plan, layer.store, None, None, SRW._slices, committer(plan))
    assert sorted(tables) == sorted([EX.table_param(W, NR * K // 2, b) for b in range(B)] + [EX.table_param(WS, 2 * K // G, 0)])
    got = SRW.read_all(ms, layer.rows, layer.store, None, tables=tables)
    at = {}
    for j, m in enumerate(ms):
        for i in range(m.n):
            at[m.base + i] = (j, i)
    for m, (g, entries) in zip(ms, got):
        if m.template in (SLOTS, BLOCK):
            stager(m.template).reference_check(g["rows"], g["outs"])
        else:
            for i in range(m.n):
                r = {q: g["rows"][q][i] for q, _w in m.rows}
                tok = int(r["in0"].view("<u4")[0])
                digests = [r["in5"][32 * d:32 * (d + 1)].view("<u8").astype(">u8").tobytes() for d in range(len(r["in5"]) // 32)]
                h, ok = EMB.reference(E, tok, r["in1"].view("<u8").astype(">u8").tobytes(), r["in2"].view("<u8").astype(">u8").tobytes(),
                                      r["in3"].tolist(), r["in4"].view("<u8").astype(">u8").tobytes(), digests)
                assert ok == 1 and h == g["outs"]["o0"][i].tolist() and g["outs"]["o1"][i].tolist() == [1]
        for port, out, writers in m.wired:
            for i, wr in enumerate(writers):
                j, k = at[wr]
                assert np.array_equal(g["rows"][port][i], got[j][0]["outs"][out][k])
        held = [u for e in entries for u in range(max(e[0], m.base), min(e[1], m.base + m.n))]
        assert held == list(range(m.base, m.base + m.n))
    served = np.concatenate([g["outs"]["o0"].reshape(-1, NR) for m, (g, _e) in zip(ms, got) if m.template == BLOCK])
    blocks = {u: (p, b) for u, (p, b) in enumerate(plan.units[BLOCK])}
    words = np.zeros((P, N), dtype=np.uint16)
    for u, row in enumerate(served):
        p, b = blocks[u]
        words[p, b * NR:(b + 1) * NR] = row
    assert words.reshape(-1).tolist() == layer.out.tolist()


def test_the_reads_roots_domains_and_verdicts_are_committed_by_reference_and_count_registered_where_pinned(layer):
    plan, ms = cut(layer.rows)
    tables = EX.commit_tables(plan, layer.store, None, None, SRW._slices, committer(plan))
    reg = EX.declarations(plan)
    assert {(r.reader, r.port, r.of, r.at, r.units) for r in reg.roots} == {
        (t, p, of, at, n) for t in (READ_W, READ_WS) for at, n, _b, _param, _l in plan.tables(t)
        for p, of in (("in1", "root"), ("in2", "domain"))}
    values = {**tables, **IF.supplied(reg, {p: v.entry() for p, v in tables.items()}, "cd" * 64, module=G1)}
    got = SRW.read_all(ms, layer.rows, layer.store, None, tables=tables)
    data, entries, roots, salts_of = WT.by_reference(SimpleNamespace(members=ms, row_tables=()), reg, [g for g, _e in got], values)
    assert roots == {p: v.root.hex() for p, v in tables.items()}
    for m, d in zip(ms, data):
        if m.template not in (READ_W, READ_WS):
            assert m.key not in entries and not any(isinstance(v, CS.ValueRows) for v in d["rows"].values())
            continue
        assert isinstance(d["rows"]["in1"], CS.ValueRows) and isinstance(d["rows"]["in2"], CS.ValueRows)
        assert sorted(entries[m.key]) == ["in1", "in2", "o1"] and entries[m.key]["o1"]["positions"] == [0] * m.n
        assert salts_of[m.port("o1")] == CS.value_salts(values[entries[m.key]["o1"]["value"]], np.zeros(m.n, dtype=np.int64))
    pins = {p: v.root.hex() for p, v in tables.items()}
    doc = {"templates": [[t, sum(m.n for m in ms if m.template == t)] for t in TMPLS], "members": [IF._member_doc(m) for m in ms],
           "registered": reg.doc()}
    resolve = {t: SRW.template(t).fn for t in TMPLS}.__getitem__
    counted = IF.input_count(doc, resolve, pins=pins, roots=pins)
    words = {t: dict(next(m for m in ms if m.template == t).rows) for t in TMPLS}
    free = 16 * (T * (words[SLOTS]["in0"] + words[SLOTS]["in1"]) + P * B * (words[BLOCK]["in0"] + words[BLOCK]["in1"]))
    assert counted["free"] == free and counted["registered"] > 0
    assert IF.input_count(doc, resolve, pins=None, roots=pins)["registered"] == 0


def test_a_read_of_a_table_the_committer_did_not_commit_and_a_pair_routed_outside_the_window_are_refused(layer):
    plan, ms = cut(layer.rows)
    tables = EX.commit_tables(plan, layer.store, None, None, SRW._slices, committer(plan))
    gone = EX.table_param(WS, 2 * K // G, 0)
    with pytest.raises(ValueError, match="not among the committer's tables"):
        SRW.read_all(ms, layer.rows, layer.store, None, tables={p: v for p, v in tables.items() if p != gone})
    with pytest.raises(ValueError, match="names no nonce"):
        EX.commit_tables(plan, layer.store, None, None, SRW._slices, {**COMMITTER, "nonces": {}})
    rows = [rs[1:] if t == SLOTS else rs for t, rs in zip(TMPLS, layer.rows)]
    with pytest.raises(ValueError, match="routed by a row outside"):
        cut(rows)
