"""DSA's long rows by table reads in a served window (`pipeline.history_rows`: the ``history`` layout and its writer, the
committer's `commit.history_tables`): one request's ``DsaLatentRow_v1`` rows at layers 3 and 5 over two windows, a prefill of
five tokens and then a decode step of one, and in each window long rows of TOPK = 4 history reads:

* a long row is TOPK reads at V = 2^17, the ``history`` layout, staged as the verifier's read template and cut where its table
  changes; with ``histories`` on, the writer's ``o0`` is row-seg, and off, its members are as before and a history read is
  refused;
* every latent row is appended to its (request, layer) table from the table's edge, read in the window or not, the second
  window's from the first's, and each read opens its table's root at its key, to the latent row there (the placeholder past
  the table's length);
* the reads' roots, domain ids and verdicts are committed by reference to the tables' supplied values, the writer's rows as
  their table's strings (hashed once), and the verifier recomputes each table's root from the writer's public
  strings, the second window's after the first window's held strings, and holds the root the window registers to it;
* each template's statement states every row under the schema its members commit it under (`stated_rows.unstated`): the
  writer's ``o0`` row-seg where the window commits history tables, v2 where it doesn't.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from verity.protocols.spot.verifier.one_stage import draw as D
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.commitments import history as HT
from verity.commitments.gates import embedding as EMB
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.definitions.registry import boolean_dsa_norm as BDN
from verity.definitions.registry import dsa_norm as DN
from verity.commitments.windows import history as HW
from verity.commitments.windows import interface as IF
from verity.commitments.windows import rows as CS
from verity_vllm.commit import history_tables as H
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import history_rows as HR
from verity_vllm.pipeline import serving_rows as SRW

from tests.pipeline.stated_rows import unstated, unstated_in

KVL, R, EPS, TOPK, WORDS = 512, 64, 1e-5, 4, 576
LATENT = bind(DN.DsaLatentRow, KVL=KVL, R=R, EPS=EPS)
LAT = descriptor_id(bind(BDN.DsaLatentRowV2, KVL=KVL, R=R, EPS=EPS))
READ = descriptor_id(EMB.read(HT.CAPACITY, WORDS))
TMPLS, CAPS = [LAT, READ], {LAT: 4, READ: 3}
PROGRAM, RUN = "ab" * 64, "cd" * 32
#: each window's steps: its latent rows' tokens, and its long rows, each (layer, its token, its keys)
WINDOWS = [
    {"step": 0, "tokens": 5, "reads": [(3, 3, [0, 2, 3, 9]), (3, 4, [1, 3, 4, 0]), (5, 3, [3, 2, 1, 0]), (5, 4, [4, 4, 0, 7])]},
    {"step": 1, "tokens": 1, "reads": [(5, 0, [5, 0, 3, 2])]},
]
LAYERS = (3, 5)


def _bf16(rng: np.random.Generator, n: int, scale: float, at: float = 0.0) -> np.ndarray:
    return ((rng.standard_normal(n) * scale + at).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def op(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.kv_a_layernorm"


def build(monkeypatch) -> SimpleNamespace:
    """Each window's rows, each template of ``TMPLS`` its own (the writer's latent rows, layer-major, then the long rows), the
    store of their served latent rows (each its `LATENT` on its kv, the layer's norm weight and its cache row) and every latent
    row by (layer, position); the serving rows read their operands from here on."""
    rng = np.random.default_rng(0xD5A)
    ops, held, latent, windows, i = {}, {}, {}, [], 0
    norm = {layer: _bf16(rng, KVL, 0.05, 1.0) for layer in LAYERS}
    for layer in LAYERS:
        ops[json.dumps([["p", layer, 0, KVL]])] = norm[layer]
    for w in WINDOWS:
        writes, reads = [], []
        for layer in LAYERS:
            for t in range(w["tokens"]):
                kv, cs = _bf16(rng, KVL + R, 1.0), _bf16(rng, R, 0.5)
                args = [[["x", 1000 + i, 0, KVL + R]], [["p", layer, 0, KVL]], [["x", 2000 + i, 0, R]]]
                ops[json.dumps(args[0])], ops[json.dumps(args[2])] = kv, cs
                y = np.asarray(evaluate(LATENT, kv.tolist(), norm[layer].tolist(), cs.tolist()), dtype=np.uint16)
                held[(w["step"], op(layer), t)] = y
                latent[(layer, len([k for k in latent if k[0] == layer]))] = y
                writes.append(("r0", None, {"i": i, "spec": LATENT.id, "args": args},
                               {"statics": {"KVL": KVL, "R": R, "EPS": EPS}, "op_path": op(layer), "row": t, "step": w["step"],
                                "members": [("latent", WORDS, "bf16")]}))
                i += 1
        for layer, t, keys in w["reads"]:
            args = [[["x", 3000 + i, 0, TOPK]], [["n", 0, 0, 1]]]
            ops[json.dumps(args[0])] = np.asarray(keys, dtype=np.int32)
            spec = f"DsaHistoryReads_v1{{H={WORDS},TOPK={TOPK},V={len([k for k in latent if k[0] == layer])}}}"
            reads.append(("r0", None, {"i": i, "spec": spec, "args": args},
                          {"statics": {"V": len([k for k in latent if k[0] == layer]), "TOPK": TOPK, "H": WORDS},
                           "op_path": f"model.layers.{layer}.self_attn.indexer.reads", "row": t, "step": w["step"],
                           "members": [("hist", TOPK * WORDS, "bf16")]}))
            i += 1
        windows.append([writes, reads])

    class Store:
        def words(self, rid, step, op_path, member, lo, hi):
            got = held.get((step, op_path, lo // WORDS))
            return (None, "not served") if got is None or hi - lo != WORDS else (got, None)

    def slices(_pi, _s, _w, _rid, _a, g, bits=None):
        return [ops[json.dumps(g)]]

    monkeypatch.setattr(SRW, "_slices", slices)
    return SimpleNamespace(windows=windows, store=Store(), latent=latent)


@pytest.fixture
def served(monkeypatch):
    return build(monkeypatch)


def cut(rows, histories=True, caps=CAPS):
    plan = HR.Plan(TMPLS, rows, histories)
    return plan, HR.cut(SRW.members(TMPLS, rows, None, caps, HR.seg(TMPLS, histories)), plan)


def binding(ms) -> CS.Binding:
    n = {t: sum(m.n for m in ms if m.template == t) for t in TMPLS}
    return CS.Binding(*CS.population_program([(SRW.template(t).fn, n[t]) for t in TMPLS], several=True),
                      D.parse_design("stratified:2"), RUN, tuple(ms), (), "", None)


def opens(t: HT.HistoryTable, g, i):
    """Read unit ``i``'s row, salt and path open table ``t``'s root at its key: -> its key and the row it opens to, or None."""
    def raw(a):
        return np.asarray(a).view("<u8").astype(">u8").tobytes()

    key = int(np.asarray(g["rows"]["in0"][i]).view("<u4")[0])
    path = raw(g["rows"]["in5"][i])
    h, ok = EMB.reference(HT.CAPACITY, key, t.root, t.domain.domain_id, np.asarray(g["rows"]["in3"][i]).tolist(),
                          raw(g["rows"]["in4"][i]), [path[64 * d:64 * (d + 1)] for d in range(len(path) // 64)])
    return (key, h) if ok == 1 else None


def test_a_long_row_is_topk_reads_cut_by_table_and_its_writer_commits_its_tables_leaves(served):
    rows = served.windows[0]
    assert SRW.template(LAT).layout == "row" and HR.writer(LAT) and SRW.template(READ).layout == "history"
    assert SRW.template(descriptor_id(EMB.read(8, WORDS))).layout == "read"
    assert CR.statement(SRW.template(READ)) == {"layout": CR.READ_LAYOUT, "structural": True}
    plan, ms = cut(rows)
    assert [(m.template, m.n) for m in ms] == [(LAT, 4), (LAT, 4), (LAT, 2)] + [(READ, 3), (READ, 3), (READ, 2)] * 2
    assert {m.schema("o0") for m in ms if m.template == LAT} == {SEG} and {m.schema("o0") for m in ms if m.template == READ} == {V2}
    assert plan.keys() == [("r0", 3), ("r0", 5)] and [k for _a, _n, k in plan.tables[READ]] == plan.keys()
    assert plan.units[READ] == [(p, j) for p in range(4) for j in range(TOPK)]
    off = SRW.members([LAT], rows[:1], None, CAPS)
    assert {m.schema("o0") for m in off} == {V2} and HR.Plan([LAT], rows[:1], False).keys() == []
    with pytest.raises(ValueError, match="commits no history tables"):
        HR.Plan(TMPLS, rows, False)


def test_a_request_served_cached_prompt_rows_appends_its_own_rows_and_its_reads_are_refused_by_name(served):
    """A table holds only rows its request computed: a request served a prefix-cache hit appends its own latent rows from
    the table's edge (leaf i its position C + i), and its reads, which open the composed table its replay builds, are
    refused by name rather than opened against this one."""
    writes, reads = served.windows[0]
    warm = SimpleNamespace(kv_prefix={"C": 4, "input": "prefix"})
    assert HR.cached(None) == 0 and HR.cached(SimpleNamespace(kv_prefix=None)) == 0 and HR.cached(warm) == 4
    with pytest.raises(ValueError, match="request r0 is served 4 cached prompt rows"):
        HR.Plan(TMPLS, [writes, [(x[0], warm, *x[2:]) for x in reads]], True)
    plan = HR.Plan([LAT], [[(x[0], warm, *x[2:]) for x in writes]], True)
    assert plan.keys() == [("r0", 3), ("r0", 5)] and [len(us) for us in plan.written.values()] == [5, 5]


def light(ms) -> SimpleNamespace:
    """A window of members ``ms`` as `serving_rows.window_values` reads it, with no population program to build (a real
    `rows.Binding` lowers the Boolean latent row and the V = 2^17 read, minutes on a CPU)."""
    return SimpleNamespace(members=tuple(ms), row_tables=(), partition_digest="ef" * 64)


def statements(monkeypatch, ms, reg=None) -> dict:
    """The statements `serving_rows.circuit_record` records for members ``ms``, by template, taken off a `light` window
    whose registered declarations are ``reg``: the record's other fields need its population program."""
    monkeypatch.setattr(IF, "templates_of", lambda _w: [])
    monkeypatch.setattr(SRW, "registered", lambda *_a: reg)
    monkeypatch.setattr(IF, "circuit_record", lambda _w, _name, _templates, sts, **_kw: sts)
    return SRW.circuit_record(light(ms), [])


def test_every_member_commits_each_port_under_the_schema_its_statement_states(served, monkeypatch):
    """The rule (`stated_rows.unstated`) over every member of each window, with history tables and without: where the
    window commits them the writer's ``o0`` is stated row-seg as it is committed, its table's leaves, and every other row
    v2; where it doesn't, the same template's ``o0`` is a v2 row in both."""
    for rows in served.windows:
        _plan, ms = cut(rows)
        sts = statements(monkeypatch, ms)
        assert unstated(ms, sts) == []
        assert [s for s, _ports in sts[LAT]["out_rows"]] == [SEG] and {s for _p, s, *_x in sts[LAT]["in_rows"]} == {V2}
        assert {s for s, _ports in sts[READ]["out_rows"]} | {s for _p, s, *_x in sts[READ]["in_rows"]} == {V2}
    off = SRW.members([LAT], served.windows[0][:1], None, CAPS)
    sts = statements(monkeypatch, off)
    assert unstated(off, sts) == [] and [s for s, _ports in sts[LAT]["out_rows"]] == [V2]


def test_a_record_whose_members_and_declared_tables_disagree_on_the_row_seg_rows_is_refused(served, monkeypatch):
    """`serving_rows.circuit_record` refuses a template whose members commit a row under two schemas (it has one statement),
    and a declared history table whose writer's row (`interface.seg_rows`) the members commit v2; `chunk_rows.unit_rows`
    refuses a row-seg row that is none of the template's, and a row named with no template, which would otherwise read as
    another template's and leave the row v2."""
    rows = served.windows[0]
    _plan, ms = cut(rows)
    with pytest.raises(ValueError, match="a template has one statement"):
        statements(monkeypatch, [*ms, CR.unit_member(SRW.template(LAT), 1)])
    declared = SimpleNamespace(histories=(SimpleNamespace(writer=LAT, out=HR.OUT),), doc=dict)
    assert statements(monkeypatch, ms, declared)[LAT]["out_rows"] == [(SEG, (0,))]
    with pytest.raises(ValueError, match="are not committed row-seg"):
        statements(monkeypatch, SRW.members([LAT], rows[:1], None, CAPS), declared)
    with pytest.raises(ValueError, match=r"\['o1'\] is no row of its"):
        CR.unit_rows(SRW.template(LAT), [(LAT, "o1")])
    assert CR.unit_rows(SRW.template(LAT), [(READ, "o1")]) == CR.unit_rows(SRW.template(LAT))
    for bare in (lambda: CR.unit_rows(SRW.template(LAT), [HR.OUT]), lambda: SRW.unit_member(LAT, 1, seg=[HR.OUT])):
        with pytest.raises(TypeError, match=r"names each row as \(template, row\)"):
            bare()


def recomputed(h: IF.History, ms, strings_of, held) -> list[bytes]:
    """History ``h``'s strings as the verifier orders them (`history.order`, as `ml.windows.history.strings` does): a unit run's
    its writer's committed strings, a held run's the strings ``held`` of the table's earlier windows."""
    writers = sorted((m for m in ms if m.template == h.writer), key=lambda m: m.base)
    mine = b"".join(strings_of[m.port(h.out)] for m in writers)
    units = sum(m.n for m in writers)
    srcs = HT.order(h.runs, h.length, h.writer, h.out, units, {v: len(x) for v, x in held.items()})
    return [mine[s[1] * CS.BC_BYTES:(s[1] + 1) * CS.BC_BYTES] if s[0] == "unit" else held[s[1]][s[2]] for s in srcs]


def windows(served, commit=None):
    """Both windows over one committer's tables, each window's members and what it commits (`serving_rows.window_values`),
    checked as the module docstring says; ``commit`` (members, values) -> what else to check of the window."""
    hs, held = H.Histories(PROGRAM), {}
    for k, rows in enumerate(served.windows):
        plan, ms = cut(rows)
        edges = {key: hs.edge(*key) for key in plan.keys()}
        got = SRW.window_values((commit or {}).get("window", light)(ms), rows, served.store, None, histories=hs, on=True)
        data, entries, roots, salts_of, strings_of, _idx, hreg = got
        assert {key: hs.edge(*key) for key in plan.keys()} == {key: edges[key] + WINDOWS[k]["tokens"] for key in edges}
        for key, t in hs.tables.items():
            assert all(np.array_equal(t.rows[p], served.latent[(key[1], p)]) for p in range(len(t.rows)))
        base = min(x.base for x in ms if x.template == READ)
        for m, d in zip(ms, data):
            if m.template != READ:
                assert len(strings_of[m.port("o0")]) == m.n * CS.BC_BYTES and m.key not in entries
                continue
            assert isinstance(d["rows"]["in1"], CS.ValueRows) and isinstance(d["rows"]["in2"], CS.ValueRows)
            assert sorted(entries[m.key]) == ["in1", "in2", "o1"] and (d["outs"]["o1"] == 1).all()
            for i in range(m.n):
                p, j = plan.units[READ][m.base - base + i]
                layer, _t, keys = WINDOWS[k]["reads"][p]
                want = served.latent.get((layer, keys[j]), np.zeros(WORDS, dtype=np.uint16))
                assert opens(hs.tables[("r0", layer)], d, i) == (keys[j], want.tolist())
                assert d["outs"]["o0"][i].tolist() == want.tolist()
        writers = sorted((m for m in ms if m.template == LAT), key=lambda m: m.base)
        assert b"".join(strings_of[m.port("o0")] for m in writers) == b"".join(
            hs.strings("r0", key[1], range(edges[key], hs.edge(*key))) for key in plan.keys())
        assert roots == {H.name(*key): hs.tables[key].root.hex() for key in plan.keys()}
        assert [h.name for h in hreg.histories] == [H.name(*key) for key in plan.keys()]
        assert IF.seg_rows(hreg) == HR.seg(TMPLS, True) == {(LAT, "o0")}
        assert [h.runs[0].get("held") for h in hreg.histories] == [h.name if k else None for h in hreg.histories]
        for h in hreg.histories:
            bcs = recomputed(h, ms, strings_of, held)
            HT.check(PROGRAM, h.name, WORDS, bcs, bytes.fromhex(roots[h.name]), HT.CAPACITY)
            held[h.name] = bcs
        if commit is not None:
            commit["check"](rows, ms, data, roots, salts_of, strings_of, hreg)
    assert hs.retained("r0") == 2 * 6 * H.ROW_BYTES and hs.end("r0") == 2 * 6 * H.ROW_BYTES and hs.requests() == []


def test_every_latent_row_is_appended_from_its_edge_and_each_read_opens_its_table_and_the_verifier_recomputes_its_root(served):
    windows(served)


@pytest.mark.skipif(not os.environ.get("VERITY_VLLM_HISTORY_E2E"),
                    reason="VERITY_VLLM_HISTORY_E2E=1: about 5 minutes on a CPU, building the Boolean DsaLatentRow_v2 and the "
                           "V = 2^17 read's population program")
@pytest.mark.slow
def test_the_window_commits_the_writers_rows_as_its_tables_strings_and_the_verifier_pins_each_root(served):
    """The cheaper test recomputes each root from the declared runs over the strings `window_values` gives `rows.commit`;
    this one commits the window (`rows.commit`, its strings given and not), pins each table at the root the window
    registers (`history.pins`) and holds that root to the one over its writers' public bodies and the strings held of its
    earlier windows (`history.params`, `history.roots_of`), as the verifier does."""
    held: dict[str, list[bytes]] = {}

    def check(rows, ms, data, roots, salts_of, strings_of, hreg):
        nonlocal held
        w = binding(ms)
        c = CS.commit(w, data, salt_key=bytes(32), salts_of=salts_of, strings_of=strings_of)
        full = CS.commit(w, data, salt_key=bytes(32), salts_of=salts_of)
        assert full.roots == c.roots and full.bc == c.bc
        publics = {m.key: c.public_body(i) for i, m in enumerate(ms)}
        doc = json.loads(IF.circuit_bytes(SRW.circuit_record(w, rows, None, hreg)))
        assert unstated_in(doc) == []
        pins = HW.pins(doc, roots)
        assert pins == {h.name: {"value": h.name, "root": roots[h.name], "leaves": HT.CAPACITY, "words": WORDS,
                                 "domain": HT.domain(PROGRAM, h.name, WORDS, HT.CAPACITY).domain_id.hex()} for h in hreg.histories}
        checked = HT.roots_of(CS.leaf_layer(publics), HW.params(doc, w, publics, pins, held))
        assert {name: root.hex() for name, (root, _domain) in checked.items()} == roots
        held = {h.name: HW.strings(h, w, publics, held) for h in hreg.histories}

    windows(served, {"window": binding, "check": check})


def test_a_window_commits_history_tables_exactly_when_its_partition_file_says_so(served):
    rows = served.windows[0]
    _plan, ms = cut(rows)
    w = light(ms)
    with pytest.raises(ValueError, match="give both or neither"):
        SRW.window_values(w, rows, served.store, None, on=True)
    with pytest.raises(ValueError, match="give both or neither"):
        SRW.window_values(w, rows, served.store, None, histories=H.Histories(PROGRAM), on=False)
    with pytest.raises(ValueError, match="no latent rows"):
        HR.declarations(HR.Plan([READ], rows[1:], True), H.Histories(PROGRAM), {})
    unwritten = [rows[0], [(*x[:3], {**x[3], "op_path": x[3]["op_path"].replace("layers.5.", "layers.7.")}) for x in rows[1]]]
    with pytest.raises(ValueError, match="model.layers.7.self_attn.latent, which holds no rows"):
        HR.declarations(HR.Plan(TMPLS, unwritten, True), H.Histories(PROGRAM), {})
    keyless = [rows[0], [(*x[:3], {**x[3], "statics": {**x[3]["statics"], "TOPK": TOPK + 1}}) for x in rows[1]]]
    _p, km = cut(keyless, caps={LAT: 4, READ: 5})
    with pytest.raises(ValueError, match="its tok is 4 positions, not its TOPK = 5"):
        SRW.window_values(light(km), keyless, served.store, None, histories=H.Histories(PROGRAM), on=True)
    _p, um = cut(rows, caps={READ: 3})
    with pytest.raises(ValueError, match="not one row o0"):
        SRW.window_values(light(um), rows, served.store, None, histories=H.Histories(PROGRAM), on=True)
