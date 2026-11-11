"""A MoE layer served cut per expert (`pipeline.serving_rows`' ``slots``, ``read``, ``expert`` and ``fused`` layouts,
`expert_rows`): two tokens' router rows and their four (token, slot) pairs of one MoE layer at E = 8, K = 64, N = 4, G = KB = 32,
in blocks of two rows, its pair rows ``MoeExpertFp8Gemm_v1`` (w13) or ``MoeExpertFp8GemmW_v1`` (w2, its routed weight from the
router's slot), and a window of the router's slots, the reads of the blocks' rows and scale rows and the block GEMMs, or of the
router's slots, the reads of the scale rows and the fused blocks:

* the units are laid out and cut as the module docstring says, each read's and fused block's key wired to its token's router slot
  and each block GEMM's rows, and each fused block's scale row, to the reads of its block;
* a read is staged as the verifier's read template, every other unit flat (`chunk_rows.statement`);
* the members' words are each template's reference on its rows, each read's and fused block's row, salt and path open its table's
  committed root at the pair's expert, and the block GEMMs' and fused blocks' outputs are the pairs' served words;
* the roots, domain ids and verdicts are committed by reference to the tables' supplied values, and a listed record of the window
  counts the tables' reads registered exactly where their roots are pinned;
* the committer takes its tables' program and nonces from the rank's registration, read through its record
  (`records.expert_tables`), each of two ranks' windows its own, and a fused block refuses another rank's tables.
"""
from __future__ import annotations

import json
import subprocess
import sys
from functools import cache
from types import SimpleNamespace

import numpy as np
import pytest
from verity_flock.class_statement import UnitStager
from verity.protocols.spot.verifier.one_stage import registered as G1
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.definitions.ops.boolean import gemm as BG
from verity.definitions.ops.prims import HopperE4m3MmaV2Dot32 as MV
from verity.definitions.ops.prims import HopperE4m3QgmmaDot32 as HP
from verity.commitments.gates import embedding as EMB
from verity.definitions.numerics import fp8_moe_rows as FR
from verity.definitions.registry import boolean_fp8_moe as BM
from verity.definitions.registry import boolean_moe as BMOE
from verity.definitions.registry import fp8_moe_experts as X
from verity.definitions.registry import moe as MOE
from verity.commitments.windows import interface as IF
from verity.commitments.windows import rows as CS
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import expert_rows as EX
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import weight_rows as WT
from verity_vllm.query.program_view import spec_statics
from verity_vllm.records import expert_tables as XT
from verity_vllm.records.fields import RecordError

E, TOPK, K, N, G, KB, NR, T = 8, 2, 64, 4, 32, 32, 2, 2
NB, KG, B, P = -(-N // G), K // G, N // NR, T * TOPK
ROUTER = descriptor_id(bind(MOE.MoeRouterSigmoidBias, E=E, TOPK=TOPK, SCALE=2.5, KERNEL="block"))
SLOTS = descriptor_id(bind(BMOE.MoeRouterSigmoidBiasSlots, E=E, TOPK=TOPK, SCALE=2.5, KERNEL="block"))
READ_W, READ_WS = descriptor_id(EMB.read(E, NR * K // 2)), descriptor_id(EMB.read(E, 2 * K // G))
LAYER = "model.layers.3.mlp"


@cache
def ids(proj: str, n: int = N, nr: int = NR) -> SimpleNamespace:
    """Projection ``proj``'s pair row of n rows, and its block GEMM and fused block of nr rows, as served: w13 unweighted on MMA v2,
    w2 weighted and swapped on QGMMA."""
    row, block, fused, swap, dot, bdot, twin = {
        "w13": (X.MoeExpertFp8Gemm, BM.MoeExpertFp8BlockGemmV4, BM.MoeExpertFp8ReadGemmV1, False, MV, BG.HopperE4m3MmaV2Dot32, "mmav2"),
        "w2": (X.MoeExpertFp8GemmW, BM.MoeExpertFp8BlockGemmWV4, BM.MoeExpertFp8ReadGemmWV1, True, HP, BG.HopperE4m3QgmmaDot32, "hopper"),
    }[proj]
    return SimpleNamespace(
        row=descriptor_id(bind(row, E=E, K=K, N=n, G=G, KB=KB, SWAP=swap, DOT=dot)),
        block=descriptor_id(bind(block, K=K, N=nr, G=G, KB=KB, SWAP=swap, DOT=bdot)),
        fused=descriptor_id(bind(fused, READ=EMB.read(E, nr * K // 2), K=K, N=nr, G=G, KB=KB, SWAP=swap, DOT=bdot)),
        swap=swap, twin=twin, routed=proj == "w2", w=f"{LAYER}.experts.{proj}_weight", ws=f"{LAYER}.experts.{proj}_weight_scale_inv")


PROJ = {proj: ids(proj) for proj in ("w13", "w2")}
GEMM, BLOCK, W, WS = PROJ["w13"].row, PROJ["w13"].block, PROJ["w13"].w, PROJ["w13"].ws
TMPLS = [SLOTS, READ_W, READ_WS, BLOCK]
CAPS = {SLOTS: 4, READ_W: 4, READ_WS: 4, BLOCK: 3}
COMMITTER = {"key": bytes(range(32)), "program": "ab" * 64}


def fused(proj: str, n: int = N, nr: int = NR) -> tuple[list[str], dict[str, int]]:
    """A fused window's templates and chunk caps: the router's slots, the reads of the scale rows and the fused blocks."""
    t = ids(proj, n, nr).fused
    return [SLOTS, READ_WS, t], {SLOTS: 4, READ_WS: 4, t: 4}


def _e4m3(rng: np.random.Generator, *shape: int) -> np.ndarray:
    b = rng.integers(0, 256, shape, dtype=np.uint8)
    b[(b & 0x7F) == 0x7F] ^= 1
    return b


def _f32(rng: np.random.Generator, lo: float, hi: float, *shape: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(lo), np.log(hi), shape)).astype(np.float32).view(np.uint32)


class Index:
    """The rows' Program: its ``weights`` struct of the layer's expert weights and scales (projection ``proj``'s, n rows), and its
    router rows' outputs, the weights ``[0, TOPK)`` and the ids ``[TOPK, 2 TOPK)`` of each."""
    pnames = ["weights"]

    def __init__(self, proj: str = "w13", n: int = N) -> None:
        self.w, self.ws, nb = PROJ[proj].w, PROJ[proj].ws, -(-n // G)
        self.fields = [(self.w, E * n * K, [E, n, K], 8), (self.ws, E * nb * KG, [E, nb, KG], 32)]
        self.at = {(0, E * n * K): (self.w, [E, n, K]), (E * n * K, E * n * K + E * nb * KG): (self.ws, [E, nb, KG])}

    def weight_slice(self, lo: int, hi: int):
        at = self.at.get((lo, hi))
        return None if at is None else (at[0], 0, hi - lo, at[1])

    def weight_bits(self, label: str) -> int:
        return 8 if label == self.w else 32

    def member_at(self, j: int, lo: int, hi: int):
        d = {"op_path": f"{LAYER}.gate"}
        return (d, "topk_ids", lo - TOPK, hi - TOPK, "i32") if lo >= TOPK else (d, "topk_weights", lo, hi, "f32")


def build(monkeypatch, proj: str = "w13", rank: int = 0, n: int = N) -> SimpleNamespace:
    """Rank ``rank``'s two tokens' router rows and their pairs' rows of projection ``proj`` and n rows (token-major, each token's
    slots in order), each operand by its argument group, and the store of their served outputs: the router's on its logits and
    bias, each pair's its twin on its expert's rows (and its slot's weight, w2's).  Each rank's weights are its own; the serving
    rows read this rank's operands from here on."""
    rng = np.random.default_rng(0xC185 + 0x100 * rank + (proj != "w13") + (n != N) * 0x10)
    pr, pi, nb = ids(proj, n), Index(proj, n), -(-n // G)
    w, ws = _e4m3(rng, E * n * K), _f32(rng, 1e-4, 1e-2, E * nb * KG)
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
                    *([[["n", 10 + t, k, k + 1]]] if pr.routed else []), [["p", 0, 0, w.size]], [["p", 0, w.size, w.size + ws.size]]]
            ops.update({json.dumps(args[0]): xq, json.dumps(args[1]): sx, json.dumps(args[2]): np.array([e], np.int32)})
            if pr.routed:
                ops[json.dumps(args[3])] = np.array([slots[k]], np.uint32)
            y, nonfinite = FR.moe_expert_fp8_block_row(xq, sx, w.reshape(E, n, K)[e], ws.reshape(E, nb, KG)[e], G=G, KB=KB,
                                                       swap=pr.swap, g=int(slots[k]) if pr.routed else None, dot=pr.twin)
            assert not nonfinite.any()
            words["out"] += y.tolist()
            pairs.append(("r0", pi, {"i": 20 + p, "spec": pr.row, "args": args},
                          {"statics": spec_statics(pr.row), "op_path": f"{LAYER}.experts", "row": p, "step": 7,
                           "members": [("out", n, "bf16")]}))
            experts.append(e)
    assert len(set(experts)) > 1
    dtype = {"topk_weights": np.uint32, "topk_ids": np.int32, "out": np.uint16}
    held = {m: np.array(v, dtype=np.uint64).astype(dtype[m]) for m, v in words.items()}

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            return held[member][lo:hi], None

    def slices(_pi, _s, _w, _rid, _a, g, bits=None):
        return [ops[json.dumps(g)]]

    monkeypatch.setattr(SRW, "_slices", slices)
    return SimpleNamespace(routers=routers, pairs=pairs, store=Store(), slices=slices, w=w, ws=ws, experts=experts,
                           out=held["out"], ops=ops, n=n)


def rows_of(tmpls, ns):
    return [ns.routers if t == SLOTS else ns.pairs for t in tmpls]


@pytest.fixture
def layer(monkeypatch):
    """Rank 0's w13 rows (`build`), each template of ``TMPLS`` its own."""
    ns = build(monkeypatch)
    ns.rows = rows_of(TMPLS, ns)
    return ns


def cut(rows, tmpls=TMPLS, caps=CAPS):
    plan = EX.Plan(tmpls, rows)
    return plan, EX.wire(EX.cut(SRW.members(tmpls, rows, None, caps), plan), plan)


def committer(plan, rank=0):
    params = [param for t in plan.what if plan.what[t][0] != "gemm" for _at, _n, _b, param, _l in plan.tables(t)]
    return {**COMMITTER, "nonces": {p: (3 << 61) + (rank << 20) + i for i, p in enumerate(sorted(params))}}


def stager(t: str) -> UnitStager:
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = SRW.template(t).fn, "experts-test", {}, CR.unit_rows(SRW.template(t))[0]
    return st


def ports(t: str) -> dict[str, str]:
    """Template ``t``'s input ports by parameter name."""
    return {nm: f"in{i}" for i, (nm, _ty) in enumerate(SRW.template(t).fn.signature()[0])}


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


def opened(rows, i):
    """The row unit ``i``'s read ports (``in0`` .. ``in5``) open to at their key, or None where they don't open their root."""
    def digest(a):
        return a.view("<u8").astype(">u8").tobytes()

    r = {q: rows[q][i] for q in ("in0", "in1", "in2", "in3", "in4", "in5")}
    h, ok = EMB.reference(E, int(r["in0"].view("<u4")[0]), digest(r["in1"]), digest(r["in2"]), r["in3"].tolist(), digest(r["in4"]),
                          [digest(r["in5"][32 * d:32 * (d + 1)]) for d in range(len(r["in5"]) // 32)])
    return h if ok == 1 else None


def check_words(plan, ms, got, block, ns, nr=NR, gemm=None):
    """Each member's words are its template's reference on its rows (a read's: its row, salt and path open its root at its key, to
    its output row; a fused block's, by its parts: its read's ports open to its row, and the reference of ``gemm``, its block GEMM,
    on that row is its output), every verdict 1, each wired row its writer's output row, each unit in an index entry, and block
    template ``block``'s outputs, each at its pair's block of nr rows, layer ``ns``'s served words."""
    at = {m.base + i: (j, i) for j, m in enumerate(ms) for i in range(m.n)}
    for m, (g, entries) in zip(ms, got):
        layout = SRW.template(m.template).layout
        if layout in ("read", "fused"):
            row = g["rows"]["in3"] if layout == "fused" else g["outs"]["o0"]
            assert all(opened(g["rows"], i) == row[i].tolist() for i in range(m.n))
        if layout == "fused":
            fu = ports(m.template)
            stager(gemm).reference_check({p: g["rows"][fu["row" if nm == "w" else nm]] for nm, p in ports(gemm).items()},
                                         {"o0": g["outs"]["o0"]})
        elif layout != "read":
            stager(m.template).reference_check(g["rows"], g["outs"])
        assert "o1" not in g["outs"] or m.template == SLOTS or (g["outs"]["o1"] == 1).all()
        for port, o, writers in m.wired:
            for i, wr in enumerate(writers):
                j, k = at[wr]
                assert np.array_equal(g["rows"][port][i], got[j][0]["outs"][o][k])
        held = [u for e in entries for u in range(max(e[0], m.base), min(e[1], m.base + m.n))]
        assert held == list(range(m.base, m.base + m.n))
    served = np.concatenate([g["outs"]["o0"].reshape(-1, nr) for m, (g, _e) in zip(ms, got) if m.template == block])
    words = np.zeros((P, ns.n), dtype=np.uint16)
    for u, row in enumerate(served):
        p, b = plan.units[block][u]
        words[p, b * nr:(b + 1) * nr] = row
    assert words.reshape(-1).tolist() == ns.out.tolist()


def test_the_members_are_their_references_the_reads_open_and_the_blocks_are_the_served_words(layer):
    plan, ms = cut(layer.rows)
    tables = EX.commit_tables(plan, layer.store, None, None, SRW._slices, committer(plan))
    assert sorted(tables) == sorted([EX.table_param(W, NR * K // 2, b) for b in range(B)] + [EX.table_param(WS, 2 * K // G, 0)])
    check_words(plan, ms, SRW.read_all(ms, layer.rows, layer.store, None, tables=tables), BLOCK, layer)


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


def test_the_committer_reads_its_tables_registration_through_its_record(layer, tmp_path):
    """A window's committer takes its key, program and each table's nonce from the rank's registration as the record reads it,
    which refuses another schema, an unknown field, a table without its own nonce, another rank's registration and a missing key."""
    plan, _ms = cut(layer.rows)
    want = committer(plan)
    tables = EX.commit_tables(plan, layer.store, None, None, SRW._slices, want)
    doc = XT.ExpertTables(program=COMMITTER["program"], repo="test/moe", revision="0" * 40, world=1, rank=0, prefix="",
                          blocks={"w13": NR}, values={p: v.entry() for p, v in tables.items()}, nonces=want["nonces"]).dump()
    (tmp_path / "key").write_bytes(COMMITTER["key"])
    (tmp_path / "tables-rank0.json").write_text(json.dumps(doc))
    spec = {"key": str(tmp_path / "key"), "registration": str(tmp_path / "tables-rank{rank}.json")}
    assert EX.of_request(spec) == want and EX.of_request(None) is None
    assert XT.ExpertTables.of(json.loads(json.dumps(doc)), "t").dump() == doc
    first = sorted(doc["nonces"])[0]
    for bad, why in [({**doc, "schema": "verity-vllm/weights-registration/v1"}, "schema"), ({**doc, "format": 1}, "format"),
                     ({**doc, "nonces": {p: n for p, n in doc["nonces"].items() if p != first}}, "nonces"),
                     ({**doc, "nonces": dict.fromkeys(doc["nonces"], 7)}, "nonces")]:
        with pytest.raises(RecordError, match=why):
            XT.ExpertTables.of(bad, "t")
    with pytest.raises(ValueError, match="not of rank 1"):
        EX.of_request({**spec, "registration": str(tmp_path / "tables-rank0.json")}, 1)
    with pytest.raises(ValueError, match="--tables-key"):
        EX.of_request({**spec, "key": None})


@pytest.mark.parametrize("proj", PROJ)
def test_a_pair_row_holds_its_fused_blocks(proj):
    pr = PROJ[proj]
    assert dict(SRW.unit_templates(pr.row))[pr.fused] == B
    assert SRW.template(pr.fused).layout == "fused" and SRW.template(pr.fused).word == pr.fused
    assert EX.kind(pr.fused, spec_statics(pr.row)) == ("fused", NR)


FRESH = "import sys\nfrom verity_vllm.pipeline import serving_templates as ST\nprint(*(ST.template(t).layout for t in sys.argv[1:]))"


def test_a_fresh_process_resolves_a_partitions_read_and_fused_templates():
    """A process that resolves a partition's templates before anything imported the read's module (a verifier's circuit, a cap
    builder) resolves its reads and fused blocks: this module imports `gates.embedding` itself, so only a fresh interpreter sees it."""
    r = subprocess.run([sys.executable, "-c", FRESH, READ_WS, PROJ["w2"].fused], capture_output=True, text=True, check=True)
    assert r.stdout.split() == ["read", "fused"]


def test_a_reads_statement_is_the_verifiers_read_template_and_every_other_templates_is_flat():
    """`chunk_rows.statement`: a table read's (``read``, a block's row or scale row) is the templated statement of the verifier's
    read template at `chunk_rows.READ_LAYOUT`, the one template `Flock.ReadNet.check` accepts; a router slot's, a block GEMM's and
    a fused block's are flat, their table reads laid out as `chunk_rows.reads` says."""
    for t in (READ_W, READ_WS):
        assert (SRW.template(t).layout, CR.statement(SRW.template(t))) == ("read", {"layout": CR.READ_LAYOUT, "structural": True})
    for t in (SLOTS, BLOCK, *(pr.fused for pr in PROJ.values())):
        tp = SRW.template(t)
        assert tp.layout != "read" and CR.statement(tp) == {"reads": CR.reads(tp)}, t


def test_the_circuit_record_states_each_templates_statement_as_compose_composes_it(monkeypatch):
    """`serving_rows.circuit_record` records each template's `chunk_rows.statement`, the statement ``compose_circuit.py
    --template`` composes and a5's composer (`records_audit.composer`) reads back: a read's layout mode and ``structural``,
    every other template's ``reads``."""
    monkeypatch.setattr(IF, "templates_of", lambda _w: [])
    monkeypatch.setattr(SRW, "registered", lambda _ms, _rows: None)
    monkeypatch.setattr(IF, "circuit_record", lambda _w, _name, _templates, sts, **_kw: sts)
    tmpls = [SLOTS, READ_W, READ_WS, BLOCK, PROJ["w13"].fused]
    sts = SRW.circuit_record(SimpleNamespace(members=[SimpleNamespace(template=t, schemas=True) for t in tmpls], row_tables=()), [])
    for t in tmpls:
        assert {k: v for k, v in sts[t].items() if k not in ("in_rows", "out_rows")} == CR.statement(SRW.template(t)), t
    assert "reads" not in sts[READ_WS] and sts[BLOCK]["reads"] == CR.reads(SRW.template(BLOCK))


@pytest.mark.parametrize("proj", PROJ)
def test_the_fused_blocks_are_cut_as_reads_and_wired_to_the_router_and_the_scale_reads(monkeypatch, proj):
    tmpls, caps = fused(proj)
    FU = tmpls[-1]
    plan, ms = cut(rows_of(tmpls, build(monkeypatch, proj)), tmpls, caps)
    assert {t: sum(m.n for m in ms if m.template == t) for t in tmpls} == {SLOTS: T, READ_WS: P * NB, FU: P * B}
    base = {t: min(m.base for m in ms if m.template == t) for t in tmpls}
    pair = {(20 + TOPK * t + k): (t, k) for t in range(T) for k in range(TOPK)}
    got = {u: (pair[plan.key(FU, u)[1]], b) for u, (_p, b) in enumerate(plan.units[FU])}
    assert all((b * TOPK + k) * T + tok == u for u, ((tok, k), b) in got.items())
    assert [(m.n, m.base - base[FU]) for m in ms if m.template == FU] == [(T, j * T) for j in range(B * TOPK)]
    for m in (m for m in ms if m.template == FU):
        keys = [got[u] for u in range(m.base - base[FU], m.base - base[FU] + m.n)]
        k, routers = keys[0][0][1], [base[SLOTS] + tok for (tok, _k), _b in keys]
        want = {"in0": (f"o{TOPK + k}", routers),
                ports(FU)["ws"]: ("o0", [base[READ_WS] + (b * NR // G * TOPK + k) * T + tok for (tok, k), b in keys]),
                **({ports(FU)["g"]: (f"o{k}", routers)} if PROJ[proj].routed else {})}
        assert {p: (o, list(w)) for p, o, w in m.wired} == want


@pytest.mark.parametrize("proj", PROJ)
def test_the_fused_blocks_are_their_references_open_their_tables_and_are_the_served_words(monkeypatch, proj):
    tmpls, caps = fused(proj)
    pr, ns = PROJ[proj], build(monkeypatch, proj)
    rows = rows_of(tmpls, ns)
    plan, ms = cut(rows, tmpls, caps)
    tables = EX.commit_tables(plan, ns.store, None, None, SRW._slices, committer(plan))
    assert sorted(tables) == sorted([EX.table_param(pr.w, NR * K // 2, b) for b in range(B)] + [EX.table_param(pr.ws, 2 * K // G, 0)])
    check_words(plan, ms, SRW.read_all(ms, rows, ns.store, None, tables=tables), tmpls[-1], ns, gemm=pr.block)


def test_a_fused_blocks_scale_row_is_its_own_g_blocks(monkeypatch):
    """At N = 64 in blocks of 16 rows, four blocks a pair over two G-blocks: each fused block's ``ws`` is wired to the read of its
    own G-block's scale row, and its words are its parts' references there."""
    n, nr = 64, 16
    tmpls, caps = fused("w2", n, nr)
    pr, ns, FU = ids("w2", n, nr), build(monkeypatch, "w2", n=n), tmpls[-1]
    rows = rows_of(tmpls, ns)
    plan, ms = cut(rows, tmpls, caps)
    base = {t: min(m.base for m in ms if m.template == t) for t in tmpls}
    pair = {(20 + TOPK * t + k): (t, k) for t in range(T) for k in range(TOPK)}
    for m in (m for m in ms if m.template == FU):
        keys = [(pair[plan.key(FU, u)[1]], plan.units[FU][u][1]) for u in range(m.base - base[FU], m.base - base[FU] + m.n)]
        assert dict((p, list(w)) for p, _o, w in m.wired)[ports(FU)["ws"]] == [
            base[READ_WS] + (b * nr // G * TOPK + k) * T + tok for (tok, k), b in keys]
    assert {b * nr // G for _p, b in plan.units[FU]} == {0, 1}
    tables = EX.commit_tables(plan, ns.store, None, None, SRW._slices, committer(plan))
    check_words(plan, ms, SRW.read_all(ms, rows, ns.store, None, tables=tables), FU, ns, nr=nr, gemm=pr.block)


def test_a_fused_blocks_tables_are_declared_on_it_committed_by_reference_and_count_registered_where_pinned(monkeypatch):
    """The fused block declares its table as a read does, its root, domain id and verdict are committed by reference, and with
    the tables pinned its free inputs are its activations alone: its rows are an opening, never a free input or an output."""
    tmpls, caps = fused("w2")
    FU, ns = tmpls[-1], build(monkeypatch, "w2")
    rows = rows_of(tmpls, ns)
    plan, ms = cut(rows, tmpls, caps)
    tables = EX.commit_tables(plan, ns.store, None, None, SRW._slices, committer(plan))
    reg = EX.declarations(plan)
    assert {(o.reader, o.key, o.ports, o.verdict, o.at, o.units) for o in reg.openings} == {
        (t, "in0", ("in3", "in4", "in5"), "o1", at, n) for t in (READ_WS, FU) for at, n, _b, _param, _l in plan.tables(t)}
    values = {**tables, **IF.supplied(reg, {p: v.entry() for p, v in tables.items()}, "cd" * 64, module=G1)}
    got = SRW.read_all(ms, rows, ns.store, None, tables=tables)
    data, entries, roots, _salts = WT.by_reference(SimpleNamespace(members=ms, row_tables=()), reg, [g for g, _e in got], values)
    assert roots == {p: v.root.hex() for p, v in tables.items()}
    for m, d in zip(ms, data):
        assert (m.key in entries) == (m.template != SLOTS)
        if m.template != SLOTS:
            assert sorted(entries[m.key]) == ["in1", "in2", "o1"] and isinstance(d["rows"]["in1"], CS.ValueRows)
    pins = {p: v.root.hex() for p, v in tables.items()}
    doc = {"templates": [[t, sum(m.n for m in ms if m.template == t)] for t in tmpls], "members": [IF._member_doc(m) for m in ms],
           "registered": reg.doc()}
    resolve = {t: SRW.template(t).fn for t in tmpls}.__getitem__
    words = {t: dict(next(m for m in ms if m.template == t).rows) for t in tmpls}
    free = 16 * (T * (words[SLOTS]["in0"] + words[SLOTS]["in1"]) + P * B * (words[FU][ports(FU)["xq"]] + words[FU][ports(FU)["sx"]]))
    counted = IF.input_count(doc, resolve, pins=pins, roots=pins)
    assert counted["free"] == free and counted["registered"] > 0
    assert IF.input_count(doc, resolve, pins=None, roots=pins)["registered"] == 0


def test_a_pair_row_cut_both_fused_and_apart_is_refused(layer):
    for extra in ([PROJ["w13"].fused], [READ_W, PROJ["w13"].fused]):
        tmpls = [SLOTS, READ_WS, BLOCK, *extra] if len(extra) == 1 else [SLOTS, READ_WS, *extra]
        with pytest.raises(ValueError, match="both fused"):
            EX.Plan(tmpls, rows_of(tmpls, layer))


def test_two_ranks_windows_each_commit_and_open_their_own_registration_and_a_fused_block_refuses_the_other(monkeypatch, tmp_path):
    """Two ranks' windows of the fused cut, each rank's weights its own: each committer is its rank's registration as the record
    reads it, each window's members are their references on its own tables, and a fused block given the other rank's tables is
    refused, since their rows at its expert are not its pair row's."""
    tmpls, caps = fused("w2")
    (tmp_path / "key").write_bytes(COMMITTER["key"])
    spec = {"key": str(tmp_path / "key"), "registration": str(tmp_path / "tables-rank{rank}.json")}
    ranks = []
    for rank in range(2):
        ns = build(monkeypatch, "w2", rank)
        rows = rows_of(tmpls, ns)
        plan, ms = cut(rows, tmpls, caps)
        want = {**committer(plan, rank), "program": f"{rank + 1:02x}" * 64}
        tables = EX.commit_tables(plan, ns.store, None, None, ns.slices, want)
        doc = XT.ExpertTables(program=want["program"], repo="test/moe", revision="0" * 40, world=2, rank=rank, prefix="",
                              blocks={"w2": NR}, values={p: v.entry() for p, v in tables.items()}, nonces=want["nonces"]).dump()
        (tmp_path / f"tables-rank{rank}.json").write_text(json.dumps(doc))
        assert EX.of_request(spec, rank) == want
        ranks.append((ns, rows, plan, ms, tables))
    assert sorted(ranks[0][4]) == sorted(ranks[1][4]) and all(ranks[0][4][p].root != ranks[1][4][p].root for p in ranks[0][4])
    for ns, rows, plan, ms, tables in ranks:
        monkeypatch.setattr(SRW, "_slices", ns.slices)
        check_words(plan, ms, SRW.read_all(ms, rows, ns.store, None, tables=tables), tmpls[-1], ns, gemm=PROJ["w2"].block)
    with pytest.raises(ValueError, match="is not row"):
        SRW.read_all(ranks[1][3], ranks[1][1], ranks[1][0].store, None, tables=ranks[0][4])


@pytest.mark.slow
def test_the_fused_blocks_words_are_the_fused_templates_own_reference(monkeypatch):
    """Each fused unit's words evaluated on the fused template's own gates, its read inside: about a minute, the bit-sliced
    evaluation of the read's hashes. It catches what `check_words`' references on the fused block's parts can't: words the
    fused template itself, as a window stages it, does not take to the served words and a verdict of 1."""
    tmpls, caps = fused("w2")
    ns = build(monkeypatch, "w2")
    rows = rows_of(tmpls, ns)
    plan, ms = cut(rows, tmpls, caps)
    got = SRW.read_all(ms, rows, ns.store, None, tables=EX.commit_tables(plan, ns.store, None, None, SRW._slices, committer(plan)))
    for m, (g, _e) in zip(ms, got):
        if m.template == tmpls[-1]:
            stager(m.template).reference_check(g["rows"], g["outs"])
            assert (g["outs"]["o1"] == 1).all()
