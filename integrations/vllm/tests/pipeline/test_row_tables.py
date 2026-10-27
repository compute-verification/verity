"""The row-table layout of the serving rows (`pipeline.serving_rows.with_row_tables`, `ml.windows.rows.RowTable`) and PoUW's
interface to it (`commit.row_tables`): one hm96-sha512/row-seg/v1 table per (engine step, linear) in the canonical order, whose
root is `verity.ml.commitments.row_table`'s under its serve-time domain, chunks whose x rows are its rows (the same strings) read by
offset, a ScaledMmS8Coordinate replay from the table that reproduces every output, the step roots serving computes at its step
boundary equal to the window's (each the step's subtree of the window's table, which is the tree over them), the verifier's
registration and cross-check through the step roots, step openings that verify and tampered rows that don't, and the bytes the
layout commits (the files it writes; Llama-3.1-8B's B32 decode step)."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as OG
from verity.ml.commitments import row_table as RT
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registered as REG, registration as R
from verity_vllm.commit import index as IX, row_tables as RTB
from verity.ml.windows import rows as CS, tables as TB
from verity_vllm.pipeline import serving_rows as SRW

DOT = '{"fn":"BlackwellS8ImmaDot32_v1"}'
K, CAP, RUN = 64, 7, "ab" * 32
#: the serve-time context (the live coin commitment's root) and the run's row key
CTX, ROW_KEY = bytes(range(64)), bytes(range(32, 64))
COORD = f"ScaledMmS8Coordinate_v1{{K={K},DOT={DOT}}}"
DEFN = bind(OG.ScaledMmS8Coordinate, K=K, **OG.GEMM_SM120_S8)
QKV, UP = "model.layers.0.self_attn.qkv_proj", "model.layers.0.mlp.gate_up_proj"
N = {QKV: 6, UP: 4}
ARRIVALS = {"r0": 0, "r1": 1}
#: (request, step, token row, program row, op_path), in the window's order: r0's prefill (two tokens) and first decode step,
#: and r1's prefill, which arrives at engine step 1 and so shares r0's decode step's tables
ROWS = [("r0", 0, 0, 10, QKV), ("r0", 0, 1, 11, QKV), ("r0", 0, 0, 12, UP), ("r0", 0, 1, 13, UP), ("r0", 1, 0, 20, QKV),
        ("r0", 1, 0, 21, UP), ("r1", 0, 0, 10, QKV), ("r1", 0, 0, 11, UP)]
UNITS = sum(N[op] for *_, op in ROWS)


def row_of(rid: str, engine_step: int, op: str, tok: int) -> int:
    """The index in ``ROWS`` of ``rid``'s token row ``tok`` of ``op`` at ``engine_step``."""
    return next(u for u, (q, step, t, _i, o) in enumerate(ROWS) if (q, ARRIVALS[q] + step, o, t) == (rid, engine_step, op, tok))


class Served:
    """Every row's operands and outputs, the outputs ScaledMmS8Coordinate's reference over them."""

    def __init__(self, seed: int = 3) -> None:
        rng = np.random.default_rng(seed)
        self.rows, self.ops, self.y = [], {}, {}
        self.W = {op: rng.integers(-128, 128, (n, K)).astype(np.int8) for op, n in N.items()}
        self.sw = {op: (np.abs(rng.standard_normal(n)) * 0.02).astype(np.float32) for op, n in N.items()}
        for u, (rid, step, tok, i, op) in enumerate(ROWS):
            n, w = N[op], 1000 + list(N).index(op)
            r = {"i": i, "spec": f"ScaledMmS8_v1{{K={K},N={n},DOT={DOT}}}",
                 "args": [[["n", u, 0, K]], [["n", u, K, K + 1]], [["p", w, 0, n * K]], [["p", w, n * K, n * K + n]]]}
            d = {"statics": {"K": K, "N": n}, "op_path": op, "members": [("y", n, "bf16")], "row": tok, "step": step}
            self.rows.append((rid, None, r, d))
            x, sx = rng.integers(-128, 128, K).astype(np.int8), np.array([abs(rng.standard_normal()) / 127], np.float32)
            got = E.evaluate_batch(DEFN, np.repeat(x.view(np.uint8)[None], n, 0), np.repeat(sx.view(np.uint32), n),
                                   self.W[op].view(np.uint8), self.sw[op].view(np.uint32), kernel="reference")
            for g, a in zip(r["args"], (x.view(np.uint8), sx.view(np.uint32), self.W[op].reshape(-1).view(np.uint8), self.sw[op].view(np.uint32))):
                self.ops[(rid, json.dumps(g))] = a
            self.y.setdefault((rid, ARRIVALS[rid] + step, op), {})[tok] = np.asarray(got, dtype=np.uint16).reshape(-1)

    def operand(self, _pi, _store, _w, rid, _arrive, group):
        return self.ops[(rid, json.dumps(group))]

    def words(self, rid, step, op, _member, lo, hi):
        y = self.y[(rid, step, op)]
        return np.concatenate([y[t] for t in sorted(y)])[lo:hi], None

    def x_of(self, u: int) -> tuple[bytes, int]:
        """Row ``u``'s s8 codes and its scale's bits."""
        rid, *_ = ROWS[u]
        args = self.rows[u][2]["args"]
        return self.ops[(rid, json.dumps(args[0]))].tobytes(), int(self.ops[(rid, json.dumps(args[1]))][0])

    def outputs(self, engine_step: int) -> dict[str, tuple[list[tuple[str, int]], np.ndarray, np.ndarray]]:
        """What the serving loop holds at ``engine_step``'s boundary: each linear's int8 quantize output, its batch rows in the order
        the requests were scheduled (r1 first: not the canonical order)."""
        out: dict[str, tuple[list, list, list]] = {}
        for u in sorted(range(len(ROWS)), key=lambda u: (ROWS[u][0] != "r1", u)):
            rid, step, tok, _i, op = ROWS[u]
            if ARRIVALS[rid] + step == engine_step:
                codes, bits = self.x_of(u)
                ids, xs, ss = out.setdefault(op, ([], [], []))
                ids.append((rid, tok))
                xs.append(np.frombuffer(codes, dtype=np.int8))
                ss.append(np.uint32(bits).view(np.float32))
        return {op: (ids, np.stack(xs), np.array(ss, dtype=np.float32)) for op, (ids, xs, ss) in out.items()}


def window(row_tables: bool = True) -> tuple[CS.Window, Served, list]:
    s = Served()
    ms = SRW.members([COORD], [s.rows], [UNITS], {COORD: CAP})
    tables: tuple = ()
    if row_tables:
        ms, tables = SRW.with_row_tables(ms, [COORD], [s.rows], ARRIVALS)
    w = CS.Window(*CS.population_program([(DEFN, UNITS)], several=True), D.parse_law("stratified:2"), RUN, tuple(ms), tables,
                  CTX.hex() if row_tables else "")
    return w, s, ms


def committed(monkeypatch, row_tables: bool = True, **kw) -> tuple[CS.Committed, Served, list]:
    w, s, ms = window(row_tables)
    monkeypatch.setattr(SRW, "_operand", s.operand)
    got = SRW._read_s8(ms, s.rows, s, None, ARRIVALS)
    return CS.commit(w, [g[0] for g in got], salt_key=bytes(32), row_salt_key=ROW_KEY, **kw), s, got


def test_one_table_per_engine_step_and_linear_in_the_canonical_order_and_each_chunk_links_the_token_rows_it_reads():
    w, _s, ms = window()
    assert [(t.port, t.keys) for t in w.row_tables] == [
        (f"row-table/0/{UP}", (("r0", 0, 12), ("r0", 1, 13))), (f"row-table/0/{QKV}", (("r0", 0, 10), ("r0", 1, 11))),
        (f"row-table/1/{UP}", (("r0", 0, 21), ("r1", 0, 11))), (f"row-table/1/{QKV}", (("r0", 0, 20), ("r1", 0, 10)))]
    assert all(t.words == SRW.s8_words(K) and RTB.layer(t.name) == 0 for t in w.row_tables)
    assert (w.segments, w.places) == (((0, (0, 1)), (1, (2, 3))), (0, 1, 0, 1))
    # units: r0 qkv t0 [0, 6), t1 [6, 12), up t0 [12, 16), t1 [16, 20), r0 decode qkv [20, 26), up [26, 30), r1 qkv [30, 36), up [36, 40)
    # each chunk at most CAP units of one table: a chunk ends at the cap or where the next token row is another table's
    assert [(m.base, m.n, m.links, m.link_units) for m in ms] == [
        (0, 7, ((1, 0), (1, 1)), (6, 1)), (7, 5, ((1, 1),), (5,)), (12, 7, ((0, 0), (0, 1)), (4, 3)), (19, 1, ((0, 1),), (1,)),
        (20, 6, ((3, 0),), (6,)), (26, 4, ((2, 0),), (4,)), (30, 6, ((3, 1),), (6,)), (36, 4, ((2, 1),), (4,))]
    assert [m.key for m in ms] == [f"{COORD}/c{j}" for j in range(8)]
    assert ms[2].link_refs == (0, 0, 0, 0, 1, 1, 1)
    assert all(dict(m.tables) == {"x": len(m.links), "w": m.n} for m in ms)
    assert w.row_tables[2].rank("r1", 0) == 1 and w.row_tables[2].rank("r0", 0) == 0
    with pytest.raises(KeyError):
        w.row_tables[2].rank("r1", 1)


def test_the_tables_do_not_depend_on_the_programs_row_order():
    """r1's rows listed first: the same tables, and every link still names its unit's own token row."""
    s = Served()
    perm = [6, 7, 0, 1, 2, 3, 4, 5]
    rows = [s.rows[u] for u in perm]
    ms, tables = SRW.with_row_tables(SRW.members([COORD], [rows], [UNITS], {COORD: CAP}), [COORD], [rows], ARRIVALS)
    assert tables == window()[0].row_tables
    units = [u for u in perm for _ in range(N[ROWS[u][4]])]
    for m in ms:
        mine = list(dict.fromkeys(units[m.base:m.base + m.n]))
        assert [(tables[t].keys[r][:2], tables[t].segment, tables[t].name) for t, r in m.links] == \
            [((ROWS[u][0], ROWS[u][2]), ARRIVALS[ROWS[u][0]] + ROWS[u][1], ROWS[u][4]) for u in mine]


def test_each_tables_root_is_the_cpu_references_under_its_serve_time_domain_and_its_chunks_carry_its_strings(monkeypatch):
    c, s, got = committed(monkeypatch)
    w = c.window
    for step, ks in w.segments:
        tabs = [w.row_tables[k] for k in ks]
        assert b"".join(c.salts[t.port] for t in tabs) == CS.segment_salts(ROW_KEY, step, sum(t.n for t in tabs))
    for t, tab in enumerate(w.row_tables):
        units = [row_of(rid, tab.segment, tab.name, tok) for rid, tok, _i in tab.keys]
        salts = c.salts[tab.port]
        assert len(salts) == 192 * tab.n
        ref = RT.commit(RT.table_domain(CTX, tab.segment, w.places[t], tab.n), [RT.s8_row(*s.x_of(u)) for u in units],
                        [salts[i * 192:(i + 1) * 192] for i in range(tab.n)])
        assert c.table_roots[tab.port] == ref.root and c.bc[tab.port] == b"".join(ref.strings) and c.table_public(t) == c.bc[tab.port]
        assert c.domains[tab.port].domain_id == ref.domain.domain_id and tab.port not in c.roots
    for k, m in enumerate(w.members):
        x = c.bc[m.port("x")]
        assert [x[j * 128:(j + 1) * 128] for j in range(len(m.links))] == \
            [c.bc[w.row_tables[t].port][r * 128:(r + 1) * 128] for t, r in m.links]
        assert c.public_body(k)[:128 * len(m.links)] == x


def test_the_step_roots_serving_computes_at_its_step_boundary_are_the_windows_and_the_windows_table_is_the_tree_over_them(monkeypatch):
    c, s, _got = committed(monkeypatch)
    w = c.window
    served = {step: RTB.step_roots(CTX, step, s.outputs(step), ROW_KEY) for step, _ in w.segments}
    for step, ks in w.segments:
        tabs = [w.row_tables[k] for k in ks]
        assert served[step].tables == tuple((t.name, c.table_roots[t.port]) for t in tabs)
        assert served[step].root == c.segment_roots[step] == RT.step_tree(CTX, step, [c.table_roots[t.port] for t in tabs]).root
        ref = RT.commit_step(CTX, step, [([RT.s8_row(*s.x_of(row_of(rid, step, t.name, tok))) for rid, tok, _i in t.keys],
                                          [c.salts[t.port][i * 192:(i + 1) * 192] for i in range(t.n)]) for t in tabs])
        assert ref.root == served[step].root and len(served[step].root) == 64
    assert RTB.step_roots(CTX, 1, s.outputs(1), ROW_KEY, workers=2) == served[1]
    assert RTB.step_roots(bytes(64), 1, s.outputs(1), ROW_KEY).root != served[1].root
    assert RTB.step_roots(CTX, 1, s.outputs(1), bytes(32)).root != served[1].root
    # the window's table: the tree over the step roots, each the step's subtree, nothing committed again
    win = RT.RootTree(w.tables_domain, [served[st].root for st, _ in w.segments])
    assert c.roots[CS.TABLE_PORT] == win.root and c.domains[CS.TABLE_PORT] == w.tables_domain
    for i, (step, _ks) in enumerate(w.segments):
        o = win.open([i])
        assert o.roots == (served[step].root,) and RT.verify_roots(c.roots[CS.TABLE_PORT], w.tables_domain.domain_id, len(w.segments), o)
    # the window's commit holds its steps to the roots serving recorded before their coins
    recorded = {step: sr.root for step, sr in served.items()}
    assert committed(monkeypatch, segment_roots=recorded)[0].roots == c.roots
    with pytest.raises(ValueError, match=r"segments \[1\]: the window's segment roots are not the ones recorded"):
        committed(monkeypatch, segment_roots={**recorded, 1: bytes(64)})
    w2, _, _ = window()
    with pytest.raises(ValueError, match="give row_salt_key"):
        CS.commit(w2, [g[0] for g in _got], salt_key=bytes(32), segment_roots=recorded)
    with pytest.raises(ValueError, match="twice in one step"):
        RTB.step_roots(CTX, 1, {QKV: ([("r0", 0), ("r0", 0)], np.zeros((2, K), np.int8), np.zeros(2, np.float32))}, ROW_KEY)
    with pytest.raises(ValueError, match="even K"):
        RTB.step_roots(CTX, 1, {QKV: ([("r0", 0)], np.zeros((1, K + 1), np.int8), np.zeros(1, np.float32))}, ROW_KEY)


def test_the_s8_coordinate_replay_reads_each_units_x_row_from_the_table(monkeypatch):
    c, s, got = committed(monkeypatch)
    for k, m in enumerate(c.window.members):
        refs = got[k][0]["refs"]
        x, wr, o0 = c.rows[m.port("x")], c.rows[m.port("w")], c.rows[m.port("o0")]
        xb = np.stack([x[refs[i, 0]].astype("<u2").view(np.uint8) for i in range(m.n)])
        wb = wr.astype("<u2").view(np.uint8)
        out = E.evaluate_batch(DEFN, np.ascontiguousarray(xb[:, :K]), np.ascontiguousarray(xb[:, K:]).view("<u4").reshape(-1),
                               np.ascontiguousarray(wb[:, :K]), np.ascontiguousarray(wb[:, K:]).view("<u4").reshape(-1), kernel="reference")
        assert np.array_equal(np.asarray(out, dtype=np.uint16), o0[:, 0])
        for i in range(m.n):
            _lo, _hi, rid, step, _layer, op, _member, tok, _i = next(e for e in got[k][1] if e[0] <= m.base + i < e[1])
            assert xb[i].tobytes() == RT.s8_row(*s.x_of(row_of(rid, step, op, tok)))[0]


def test_the_registration_holds_the_tree_over_the_step_roots_and_the_verifier_ties_it_to_the_chunks(monkeypatch, tmp_path):
    c, s, got = committed(monkeypatch)
    win = CS.write(c, tmp_path, {"row": 101}, IX.index_doc(c.window.members, [g[1] for g in got], RUN), registrar=R)
    rec = json.loads((tmp_path / "registration.json").read_text())
    mine, _, _ = window()
    ports = CS.expected_ports(mine)
    assert ports[CS.TABLE_PORT]["leaves"] == 2 and not {t.port for t in mine.row_tables} & set(ports)
    assert {r["port"] for r in rec["roots"]} == set(ports)
    assert next(r for r in rec["roots"] if r["port"] == CS.TABLE_PORT)["root"] == c.roots[CS.TABLE_PORT].hex()
    publics = {m.key: (tmp_path / f"m{k}-public.bin").read_bytes() for k, m in enumerate(mine.members)}
    v = R.check(rec, mine.program, mine.query, mine.law, CS.SCHEME, ports, leaf_layer=CS.leaf_layer(publics))
    assert v.ok, v.detail
    roots, live = {r["port"]: r["root"] for r in rec["roots"]}, dict(c.segment_roots)
    assert TB.check_row_tables(mine, publics, roots) == {} and TB.check_row_tables(mine, publics, roots, live) == {}
    assert [e["port"] for e in win["row_tables"]] == [t.port for t in mine.row_tables]
    assert [e["step_index"] for e in win["row_tables"]] == [0, 1, 0, 1]
    assert win["row_table_steps"]["steps"] == [{"engine_step": 0, "tables": [0, 1], "root": c.segment_roots[0].hex()},
                                               {"engine_step": 1, "tables": [2, 3], "root": c.segment_roots[1].hex()}]
    assert win["members"][2]["row_links"] == [[0, 0], [0, 1]] and win["members"][2]["shared_rows"] == {"x": 2, "w": 7, "o0": 7}
    assert win["members"][2]["registered"][CS.LINKED]["value"] == mine.row_tables[0].port
    assert win["members"][2]["registered"][CS.LINKED]["positions"] == [0, 1]
    # each linked chunk's --registered file: its x rows open its one table's root at the positions its units read
    reads = TB.registered_reads(mine, publics, registered=REG)
    assert set(reads) == {m.key for m in mine.members if m.links}
    for k, m in enumerate(mine.members):
        t, port, head = m.links[0][0], reads[m.key]["ports"][CS.LINKED], win["members"][k]["registered"][CS.LINKED]
        tab = mine.row_tables[t]
        assert reads[m.key]["format"] == REG.READS_V1 and set(reads[m.key]) == {"format", "ports"}
        assert set(port) == {"value", "root", "leaves", "words", "word_bits", "domain", "row_schema", "reads", "reads_from"}
        assert (port["value"], port["root"], port["leaves"], port["words"], port["reads_from"]) == \
            (tab.port, c.table_roots[tab.port].hex(), tab.n, SRW.s8_words(K), m.base)
        assert port["reads"] == [head["positions"][j] for j in m.link_refs] == [m.links[j][1] for j in m.link_refs]
        dom = mine.table_domain(t)
        assert port["domain"] == dom.domain_id.hex() and port["row_schema"] == REG.SCHEMA_SEG
        root, x = bytes.fromhex(port["root"]), publics[m.key]
        for j, (pos, path) in enumerate(zip(head["positions"], head["paths"], strict=True)):
            bc, path = x[j * CS.BC_BYTES:(j + 1) * CS.BC_BYTES], bytes.fromhex(path)
            assert REG.verify_read(dom, root, pos, bc, path, REG.SCHEMA_SEG)
            assert not REG.verify_read(dom, bytes(64), pos, bc, path, REG.SCHEMA_SEG)
            assert not REG.verify_read(dom, root, pos ^ 1, bc, path, REG.SCHEMA_SEG)
            assert not REG.verify_read(dom, root, pos, bc, path, REG.SCHEMA)
    with pytest.raises(ValueError, match="do not hold"):
        TB.registered_reads(mine, {k: v for k, v in publics.items() if k != mine.members[0].key}, registered=REG)
    # a chunk whose x string is the only copy of a table row and is not that row: step 0's root, against the recorded one
    k0, t0, t1 = mine.members[0].key, mine.row_tables[0].port, mine.row_tables[1].port
    flip = lambda body, at: body[:at] + bytes([body[at] ^ 1]) + body[at + 1:]  # noqa: E731
    once = dict(publics, **{k0: flip(publics[k0], 5)})                 # m0's x row 0, table 1's row 0, read by m0 alone
    assert TB.check_row_tables(mine, once, roots) == {CS.TABLE_PORT: "the members' strings are not its root's rows"}
    assert TB.check_row_tables(mine, once, roots, live) == {f"{CS.TABLE_PORT}/0": "the members' strings are not the segment root recorded"}
    twice = dict(publics, **{k0: flip(publics[k0], 200)})              # m0's x row 1, table 1's row 1, which m1 reads too
    assert set(TB.check_row_tables(mine, twice, roots)) == {t1, CS.TABLE_PORT} and set(TB.check_row_tables(mine, twice, roots, live)) == {t1}
    # a registered root that is not the tree over the step roots, a recorded step root the window's are not, a chunk missing
    other = dict(roots, **{CS.TABLE_PORT: "00" * 64})
    assert TB.check_row_tables(mine, publics, other, live) == {CS.TABLE_PORT: "its root is not the tree over the recorded segment roots"}
    assert set(TB.check_row_tables(mine, publics, roots, {**live, 1: bytes(64)})) == {f"{CS.TABLE_PORT}/1", CS.TABLE_PORT}
    assert TB.check_row_tables(mine, {k: v for k, v in publics.items() if k != k0}, roots) == {
        k0: "its public body is 0 bytes, short of its 2 x strings", t1: "a row no member's", CS.TABLE_PORT: "a segment whose tables do not hold"}
    assert t0 not in TB.check_row_tables(mine, {k: v for k, v in publics.items() if k != k0}, roots)
    copies, _, _ = window(row_tables=False)
    assert "R3-ports" in R.check(rec, copies.program, copies.query, copies.law, CS.SCHEME, CS.expected_ports(copies),
                                 leaf_layer=CS.leaf_layer(publics)).codes


def test_the_audit_record_carries_each_drawn_chunks_reads_beside_the_roots_lean_reports(monkeypatch, tmp_path):
    """`reported_reads`: each drawn chunk's table and root from the verifier's own reads file beside Lean's `registered`
    finding, held only when every verdict accepted and reports exactly its file's port, value and root."""
    c, _s, got = committed(monkeypatch)
    CS.write(c, tmp_path, {"row": 101}, IX.index_doc(c.window.members, [g[1] for g in got], RUN), registrar=R)
    mine, _, _ = window()
    reads = TB.registered_reads(mine, {m.key: (tmp_path / f"m{k}-public.bin").read_bytes() for k, m in enumerate(mine.members)},
                                 registered=REG)
    k0, k4 = mine.members[0].key, mine.members[4].key
    lean = lambda key: {"accepted": True, "of_record": True,  # noqa: E731
                        "registered": {p: {"value": e["value"], "root": e["root"]} for p, e in reads[key]["ports"].items()}}
    rec = TB.reported_reads(reads, {k0: lean(k0), k4: lean(k4)})
    t1 = mine.row_tables[1]
    assert rec["ok"] and rec["chunks"][k0] == {"reads": {CS.LINKED: {"value": t1.port, "root": c.table_roots[t1.port].hex()}},
                                               "reported": lean(k0)["registered"], "ok": True}
    assert json.loads(json.dumps(rec)) == rec
    another_root = {CS.LINKED: {**lean(k0)["registered"][CS.LINKED], "root": "00" * 64}}
    for bad in ({**lean(k0), "registered": another_root}, {**lean(k0), "registered": lean(k4)["registered"]},
                {k: v for k, v in lean(k0).items() if k != "registered"}, {**lean(k0), "accepted": False}):
        assert not TB.reported_reads(reads, {k0: bad, k4: lean(k4)})["ok"]
    assert not TB.reported_reads({k4: reads[k4]}, {k0: lean(k0), k4: lean(k4)})["ok"]


def test_a_chunk_spanning_two_row_tables_is_refused_on_the_verifier_side(monkeypatch, tmp_path):
    """The chunker cuts at table boundaries, and a prover that builds its own chunks can't bring the old form back: the verifier's
    reads file names one root, and no row of another table opens it at any position through any of its paths (Lean's
    `Registered.opens`, through its Python twin `registered.verify_read`; `verity/ml/flock/tests/test_lean_registered_reads.py`
    runs the executable verifier on such a table). `registered_reads` writes no file for a member linking two tables, whatever
    built it."""
    c, _s, got = committed(monkeypatch)
    CS.write(c, tmp_path, {"row": 101}, IX.index_doc(c.window.members, [g[1] for g in got], RUN), registrar=R)
    w = c.window
    bc = lambda t, r: c.bc[w.row_tables[t].port][r * CS.BC_BYTES:(r + 1) * CS.BC_BYTES]  # noqa: E731
    trees = {t: RT.RowTable(w.table_domain(t), [bc(t, r) for r in range(w.row_tables[t].n)]) for t in range(len(w.row_tables))}
    span = [(1, 1), (0, 0)]         # the old chunker's chunk: table 1's row 1, then table 0's row 0
    for held in (0, 1):
        dom, root, n = w.table_domain(held), c.table_roots[w.row_tables[held].port], w.row_tables[held].n
        assert [REG.verify_read(dom, root, r, bc(t, r), trees[t].path(r), REG.SCHEMA_SEG) for t, r in span] == [t == held for t, _ in span]
        t, r = next(x for x in span if x[0] != held)
        assert not any(REG.verify_read(dom, root, p, bc(t, r), trees[held].path(q), REG.SCHEMA_SEG) for p in range(n) for q in range(n))
    mine, _, _ = window()
    publics = {m.key: (tmp_path / f"m{k}-public.bin").read_bytes() for k, m in enumerate(mine.members)}
    old = copy.copy(mine.members[1])
    object.__setattr__(old, "links", tuple(span))
    object.__setattr__(old, "link_units", (4, 1))
    spanned = copy.copy(mine)
    object.__setattr__(spanned, "members", (mine.members[0], old, *mine.members[2:]))
    with pytest.raises(ValueError, match="several row tables"):
        TB.registered_reads(spanned, publics, registered=REG)


def test_pouws_interface_finds_a_table_and_its_step_openings_verify_and_tampered_rows_fail(monkeypatch, tmp_path):
    c, s, got = committed(monkeypatch)
    CS.write(c, tmp_path, {"row": 101}, IX.index_doc(c.window.members, [g[1] for g in got], RUN), registrar=R)
    rec = json.loads((tmp_path / "registration.json").read_text())
    mine, _, _ = window()
    live = dict(c.segment_roots)
    prover, verifier = RTB.of_dir(tmp_path), RTB.of_registration(mine, rec, live)
    key = lambda t: (t.spec, t.domain_id, t.k, t.index, t.tables, t.context, t.step_root)  # noqa: E731
    assert [key(t) for t in prover] == [key(t) for t in verifier]
    t = RTB.find(verifier, 1, 0, "qkv_proj")
    assert t is RTB.find(verifier, 1, 0, QKV) and t.port == f"row-table/1/{QKV}" and t.step_root == live[1]
    assert (t.leaves, t.rank("r1", 0), t.rank("r0", 0), t.index, t.tables) == (2, 1, 0, 1, 2)
    for ranks in ([1], [0, 1]):
        o = RTB.opening(tmp_path, t, ranks)
        rows = RTB.witness(tmp_path, t, ranks)
        assert RTB.verify(t, o) and RTB.verify(t, o, rows) and o.root == c.table_roots[t.port]
    assert RT.StepOpening.of_json(json.loads(json.dumps(o.to_json()))) == o
    assert rows[1][0] == RT.s8_row(*s.x_of(6))[0] and rows[1][1] == 8 * K + 32
    row, bits, salt = rows[1]
    assert not RTB.verify(t, o, {**rows, 1: (bytes([row[0] ^ 1]) + row[1:], bits, salt)})
    assert not RTB.verify(t, o, {**rows, 1: (row, bits, rows[0][2])})
    assert not RTB.verify(RTB.find(verifier, 0, 0, "qkv_proj"), o)
    assert not RTB.verify(RTB.find(verifier, 1, 0, "gate_up_proj"), o)
    up = RTB.opening(tmp_path, RTB.find(prover, 1, 0, "gate_up_proj"), [0, 1])
    assert not RTB.verify(t, RT.StepOpening(o.table, up.rows)) and not RTB.verify(t, RT.StepOpening(up.table, o.rows))
    with pytest.raises(LookupError, match="0 row tables"):
        RTB.find(verifier, 2, 0, "qkv_proj")
    with pytest.raises(LookupError, match="2 row tables"):
        RTB.find(verifier + [t], 1, 0, "qkv_proj")
    with pytest.raises(ValueError, match="does not hold"):
        RTB.of_registration(mine, dict(rec, roots=[r for r in rec["roots"] if r["port"] != CS.TABLE_PORT]), live)
    with pytest.raises(ValueError, match="not the tree over the step roots serving recorded"):
        RTB.of_registration(mine, rec, {**live, 0: live[1]})
    (tmp_path / f"t{t.k}-public.bin").write_bytes(RTB._strings(tmp_path, RTB.find(prover, 0, 0, "qkv_proj"))[0] * 2)
    with pytest.raises(ValueError, match="not row-table/1"):
        RTB.opening(tmp_path, t, [0])


def test_a_window_or_commit_whose_links_do_not_hold_is_refused(monkeypatch):
    w, s, ms = window()
    prog, query, ctx = w.program, w.query, CTX.hex()
    with pytest.raises(ValueError, match="every row some member's"):
        CS.Window(prog, query, w.law, RUN, w.members, w.row_tables + (RTB.row_table(9, QKV, COORD, SRW.s8_words(K), [("r9", 0, 1)]),), ctx)
    with pytest.raises(ValueError, match="no row of"):
        CS.Window(prog, query, w.law, RUN, w.members, w.row_tables[:3], ctx)
    bad = replace(w.row_tables[0], words=3)
    with pytest.raises(ValueError, match="34 words"):
        CS.Window(prog, query, w.law, RUN, w.members, (bad,) + w.row_tables[1:], ctx)
    t = w.row_tables
    with pytest.raises(ValueError, match="by segment, rising"):
        CS.Window(prog, query, w.law, RUN, w.members, t[2:] + t[:2], ctx)
    with pytest.raises(ValueError, match="row keys each once"):
        CS.Window(prog, query, w.law, RUN, w.members, t[:2] + (replace(t[2], keys=(t[2].keys[0],) * 2),) + t[3:], ctx)
    # within a segment the order is the integration's: vLLM's is refused otherwise on its side
    RTB.check_order(t)
    with pytest.raises(ValueError, match=r"row_tables.table_key"):
        RTB.check_order((t[1], t[0]) + t[2:])
    with pytest.raises(ValueError, match=r"row_tables.table_key"):
        RTB.check_order(t[:2] + (replace(t[2], keys=t[2].keys[::-1]),) + t[3:])
    for no in ("", "ab" * 32, "AB" * 64):
        with pytest.raises(ValueError, match="names its context"):
            CS.Window(prog, query, w.law, RUN, w.members, t, no)
    with pytest.raises(ValueError, match="one per row"):
        CS.Member(COORD, (("x", 34), ("w", 34)), (("o0", 1),), 2, tables=(("x", 2), ("w", 2)), links=((0, 0),))
    member = lambda links, units: CS.Member(COORD, (("x", 34), ("w", 34)), (("o0", 1),), 3,  # noqa: E731
                                            tables=(("x", len(links)), ("w", 3)), links=links, link_units=units)
    assert member(((0, 0), (0, 1)), (2, 1)).link_refs == (0, 0, 1)
    with pytest.raises(ValueError, match="several row tables"):
        member(((0, 0), (1, 0)), (2, 1))
    for units in ((), (3,), (2, 2), (3, 0)):
        with pytest.raises(ValueError, match="link units"):
            member(((0, 0), (0, 1)), units)
    monkeypatch.setattr(SRW, "_operand", s.operand)
    got = SRW._read_s8(ms, s.rows, s, None, ARRIVALS)
    data = [g[0] for g in got]
    data[1]["rows"]["x"] = data[1]["rows"]["x"].copy()
    data[1]["rows"]["x"][0, 0] ^= 1                                  # chunk 1 reads table 1's row 1 otherwise than chunk 0
    with pytest.raises(ValueError, match="as another member reads it"):
        CS.commit(w, data, salt_key=bytes(32))


@pytest.mark.parametrize("row_tables", [False, True])
def test_window_bytes_are_the_bytes_the_window_writes(monkeypatch, tmp_path, row_tables):
    c, _s, got = committed(monkeypatch, row_tables)
    CS.write(c, tmp_path, {"row": 101}, IX.index_doc(c.window.members, [g[1] for g in got], RUN), registrar=R)
    size = lambda pat: sum(p.stat().st_size for p in Path(tmp_path).glob(pat))  # noqa: E731
    b = CS.window_bytes(c.window.members, c.window.row_tables)
    assert (b["public"], b["private"]) == (size("[mt]*-public.bin"), size("[mt]*-private.bin"))
    assert (b["tables_public"], b["tables_private"]) == (size("t*-public.bin"), size("t*-private.bin"))
    x = 2 * SRW.s8_words(K) + 192
    if row_tables:
        assert b["x_private"] == x * (sum(len(m.links) for m in c.window.members) + 8)
    else:
        assert b["x_private"] == x * UNITS and b["x_public"] == 128 * UNITS


#: Llama-3.1-8B's int8 linears (K, N), its layers, and the chunk caps of record
LLAMA8B = {"self_attn.qkv_proj": (4096, 6144), "self_attn.o_proj": (4096, 4096), "mlp.gate_up_proj": (4096, 28672), "mlp.down_proj": (14336, 4096)}
LAYERS, B = 32, 32
CAPS = {4096: 2048, 14336: 1024}


def llama_b32_decode_step() -> tuple[list[str], list[list], dict[str, int]]:
    """One decode step of B = 32 requests: per request, per layer, its four linears' ScaledMmS8_v1 rows (one token row each)."""
    tmpls = [f"ScaledMmS8Coordinate_v1{{K={k},DOT={DOT}}}" for k in CAPS]
    rows: list[list] = [[], []]
    for q in range(B):
        for layer in range(LAYERS):
            for j, (lin, (k, n)) in enumerate(LLAMA8B.items()):
                d = {"statics": {"K": k, "N": n}, "op_path": f"model.layers.{layer}.{lin}", "members": [("y", n, "bf16")], "row": 0, "step": 1}
                rows[k == 14336].append((f"r{q}", None, {"i": 4 * layer + j}, d))
    return tmpls, rows, {t: CAPS[k] for t, k in zip(tmpls, CAPS)}


def test_the_bytes_of_llama_8bs_b32_decode_step_n_copies_and_one_table():
    """Llama-3.1-8B, B = 32, one decode step: the ScaledMmS8Coordinate units' committed bytes in each layout, from the layout
    rule `window_bytes` holds to the written files (above).  Every chunk is one token row's (the caps divide every N)."""
    tmpls, rows, caps = llama_b32_decode_step()
    ms = SRW.members(tmpls, rows, None, caps)
    linked, tables = SRW.with_row_tables(ms, tmpls, rows, {f"r{q}": 0 for q in range(B)})
    before, after = CS.window_bytes(ms), CS.window_bytes(linked, tables)
    units = B * LAYERS * sum(n for _, n in LLAMA8B.values())
    xrow = {k: 2 * SRW.s8_words(k) + 192 for k in CAPS}
    per_tl = {k: sum(n for kk, n in LLAMA8B.values() if kk == k) for k in CAPS}
    chunks = B * LAYERS * sum(n // CAPS[k] for k, n in LLAMA8B.values())
    assert (sum(m.n for m in ms), len(ms), len(tables), sum(t.n for t in tables)) == (units, chunks, LAYERS * 4, B * LAYERS * 4)
    assert all(len(m.links) == 1 for m in linked)
    assert before["x_public"] == 128 * units == 5_637_144_576
    assert before["x_private"] == B * LAYERS * sum(per_tl[k] * xrow[k] for k in CAPS) == 231_970_177_024
    table_rows = B * LAYERS * sum(xrow[k] for k, _ in LLAMA8B.values())
    chunk_rows = B * LAYERS * sum(n // CAPS[k] * xrow[k] for k, n in LLAMA8B.values())
    assert (after["tables_public"], after["tables_private"]) == (128 * B * LAYERS * 4, table_rows) == (524_288, 28_065_792)
    assert after["x_private"] == table_rows + chunk_rows == 171_094_016
    assert after["x_public"] == 128 * (B * LAYERS * 4 + chunks) == 3_538_944
    # w and o0 stay per unit; the refs (u32 per unit and port) are the layout's one new cost
    assert after["public"] - after["x_public"] == before["public"] - before["x_public"] + 12 * units
    assert after["private"] - after["x_private"] == before["private"] - before["x_private"]
    print(json.dumps({"units": units, "chunks": chunks, "row_tables": len(tables), "before": before, "after": after}, indent=1))
