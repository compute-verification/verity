"""The row-table layout of the serving rows (`pipeline.serving_rows.with_row_tables`, `commit.serving_rows.RowTable`) and PoUW's
interface to it (`commit.row_tables`): one hm96-sha512/row-seg/v1 table per (engine step, linear) whose root is
`verity.ml.commitments.row_table`'s, chunks whose x rows are its rows (the same strings) read by offset, a ScaledMmS8Coordinate
replay from the table that reproduces every output, the verifier's registration and cross-check, openings that verify and
tampered rows that don't, and the bytes the layout commits (the files it writes; Llama-3.1-8B's B32 decode step)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as OG
from verity.ml.commitments import row_table as RT
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity_vllm.commit import row_tables as RTB, serving_rows as CS
from verity_vllm.pipeline import serving_rows as SRW

DOT = '{"fn":"BlackwellS8ImmaDot32_v1"}'
K, CAP, RUN = 64, 7, "ab" * 32
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


def window(row_tables: bool = True) -> tuple[CS.Window, Served, list]:
    s = Served()
    ms = SRW.members([COORD], [s.rows], [UNITS], {COORD: CAP})
    tables: tuple = ()
    if row_tables:
        ms, tables = SRW.with_row_tables(ms, [COORD], [s.rows], ARRIVALS)
    w = CS.Window(*CS.population_program([(DEFN, UNITS)], several=True), D.parse_law("stratified:2"), RUN, tuple(ms), tables)
    return w, s, ms


def committed(monkeypatch, row_tables: bool = True) -> tuple[CS.Committed, Served, list]:
    w, s, ms = window(row_tables)
    monkeypatch.setattr(SRW, "_operand", s.operand)
    got = SRW._read_s8(ms, s.rows, s, None, ARRIVALS)
    return CS.commit(w, [g[0] for g in got], salt_key=bytes(32)), s, got


def test_one_table_per_engine_step_and_linear_and_each_chunk_links_the_token_rows_it_reads():
    w, _s, ms = window()
    assert [(t.port, t.rows) for t in w.row_tables] == [
        (f"row-table/0/{QKV}", (("r0", 0, 10), ("r0", 1, 11))), (f"row-table/0/{UP}", (("r0", 0, 12), ("r0", 1, 13))),
        (f"row-table/1/{QKV}", (("r0", 0, 20), ("r1", 0, 10))), (f"row-table/1/{UP}", (("r0", 0, 21), ("r1", 0, 11)))]
    assert all(t.words == SRW.s8_words(K) and t.layer == 0 for t in w.row_tables)
    # units: r0 qkv t0 [0, 6), t1 [6, 12), up t0 [12, 16), t1 [16, 20), r0 decode qkv [20, 26), up [26, 30), r1 qkv [30, 36), up [36, 40)
    assert [(m.base, m.n, m.links) for m in ms] == [
        (0, 7, ((0, 0), (0, 1))), (7, 7, ((0, 1), (1, 0))), (14, 7, ((1, 0), (1, 1), (2, 0))), (21, 7, ((2, 0), (3, 0))),
        (28, 7, ((3, 0), (2, 1))), (35, 5, ((2, 1), (3, 1)))]
    assert all(dict(m.tables) == {"x": len(m.links), "w": m.n} for m in ms)
    assert w.row_tables[2].rank("r1", 0) == 1 and w.row_tables[2].rank("r0", 0) == 0
    with pytest.raises(KeyError):
        w.row_tables[2].rank("r1", 1)


def test_each_tables_root_is_the_cpu_references_and_its_chunks_carry_its_strings(monkeypatch):
    c, s, got = committed(monkeypatch)
    w = c.window
    for t, tab in enumerate(w.row_tables):
        units = [row_of(rid, tab.engine_step, tab.op_path, tok) for rid, tok, _i in tab.rows]
        salts = c.salts[tab.port]
        ref = RT.commit(w.domain(tab.port, RT.SCHEMA, tab.n), [RT.s8_row(*s.x_of(u)) for u in units],
                        [salts[i * 192:(i + 1) * 192] for i in range(tab.n)])
        assert c.roots[tab.port] == ref.root and c.bc[tab.port] == b"".join(ref.strings) and c.table_public(t) == c.bc[tab.port]
    for k, m in enumerate(w.members):
        x = c.bc[m.port("x")]
        assert [x[j * 128:(j + 1) * 128] for j in range(len(m.links))] == \
            [c.bc[w.row_tables[t].port][r * 128:(r + 1) * 128] for t, r in m.links]
        assert c.public_body(k)[:128 * len(m.links)] == x


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


def test_the_registration_holds_the_tables_and_the_verifier_ties_them_to_the_chunks(monkeypatch, tmp_path):
    c, s, got = committed(monkeypatch)
    win = CS.write(c, tmp_path, {"row": 101}, CS.index_doc(c.window.members, [g[1] for g in got], RUN))
    rec = json.loads((tmp_path / "registration.json").read_text())
    mine, _, _ = window()
    ports = CS.expected_ports(mine)
    assert {t.port for t in mine.row_tables} <= set(ports) and {r["port"] for r in rec["roots"]} == set(ports)
    publics = {m.key: (tmp_path / f"m{k}-public.bin").read_bytes() for k, m in enumerate(mine.members)}
    v = R.check(rec, mine.program, mine.query, mine.law, CS.SCHEME, ports, leaf_layer=CS.leaf_layer(publics))
    assert v.ok, v.detail
    roots = {r["port"]: r["root"] for r in rec["roots"]}
    assert CS.check_row_tables(mine, publics, roots) == {}
    assert [e["port"] for e in win["row_tables"]] == [t.port for t in mine.row_tables]
    assert win["members"][2]["row_links"] == [[1, 0], [1, 1], [2, 0]] and win["members"][2]["shared_rows"] == {"x": 3, "w": 7, "o0": 7}
    # a chunk whose x string is not its table's row, a registered root that is another table's, a table no chunk reads
    k0 = mine.members[0].key
    flipped = dict(publics, **{k0: publics[k0][:200] + bytes([publics[k0][200] ^ 1]) + publics[k0][201:]})
    assert set(CS.check_row_tables(mine, flipped, roots)) == {mine.row_tables[0].port}
    t0, t1 = mine.row_tables[0].port, mine.row_tables[1].port
    assert set(CS.check_row_tables(mine, publics, dict(roots, **{t0: roots[t1]}))) == {t0}
    assert CS.check_row_tables(mine, {k: v for k, v in publics.items() if k != k0}, roots) == {
        k0: "its public body is 0 bytes, short of its 2 x strings", t0: "a row no member's"}
    copies, _, _ = window(row_tables=False)
    assert "R3-ports" in R.check(rec, copies.program, copies.query, copies.law, CS.SCHEME, CS.expected_ports(copies),
                                 leaf_layer=CS.leaf_layer(publics)).codes


def test_pouws_interface_finds_a_table_and_its_openings_verify_and_tampered_rows_fail(monkeypatch, tmp_path):
    c, s, got = committed(monkeypatch)
    CS.write(c, tmp_path, {"row": 101}, CS.index_doc(c.window.members, [g[1] for g in got], RUN))
    rec = json.loads((tmp_path / "registration.json").read_text())
    mine, _, _ = window()
    prover, verifier = RTB.of_dir(tmp_path), RTB.of_registration(mine, rec)
    assert [(t.spec, t.root, t.domain_id, t.k) for t in prover] == [(t.spec, t.root, t.domain_id, t.k) for t in verifier]
    t = RTB.find(verifier, 1, 0, "qkv_proj")
    assert t is RTB.find(verifier, 1, 0, QKV) and t.port == f"row-table/1/{QKV}" and t.root == c.roots[t.port]
    assert (t.leaves, t.rank("r1", 0), t.rank("r0", 0)) == (2, 1, 0)
    for ranks in ([1], [0, 1]):
        o = RTB.opening(tmp_path, t, ranks)
        rows = RTB.witness(tmp_path, t, ranks)
        assert RTB.verify(t, o) and RTB.verify(t, o, rows)
    assert rows[1][0] == RT.s8_row(*s.x_of(6))[0] and rows[1][1] == 8 * K + 32
    row, bits, salt = rows[1]
    assert not RTB.verify(t, o, {**rows, 1: (bytes([row[0] ^ 1]) + row[1:], bits, salt)})
    assert not RTB.verify(t, o, {**rows, 1: (row, bits, rows[0][2])})
    assert not RTB.verify(RTB.find(verifier, 0, 0, "qkv_proj"), o)
    with pytest.raises(LookupError, match="0 row tables"):
        RTB.find(verifier, 2, 0, "qkv_proj")
    with pytest.raises(LookupError, match="2 row tables"):
        RTB.find(verifier + [t], 1, 0, "qkv_proj")
    with pytest.raises(ValueError, match="does not hold"):
        RTB.of_registration(mine, dict(rec, roots=[r for r in rec["roots"] if r["port"] != t.port]))
    (tmp_path / f"t{t.k}-public.bin").write_bytes(RTB._strings(tmp_path, RTB.find(prover, 0, 0, "qkv_proj"))[0] * 2)
    with pytest.raises(ValueError, match="not row-table/1"):
        RTB.opening(tmp_path, t, [0])


def test_a_window_or_commit_whose_links_do_not_hold_is_refused(monkeypatch):
    w, s, ms = window()
    prog, query = w.program, w.query
    with pytest.raises(ValueError, match="every row some member's"):
        CS.Window(prog, query, w.law, RUN, w.members, w.row_tables + (CS.RowTable(9, QKV, COORD, SRW.s8_words(K), (("r9", 0, 1),)),))
    with pytest.raises(ValueError, match="no row of"):
        CS.Window(prog, query, w.law, RUN, w.members, w.row_tables[:3])
    bad = CS.RowTable(0, QKV, COORD, 3, w.row_tables[0].rows)
    with pytest.raises(ValueError, match="34 words"):
        CS.Window(prog, query, w.law, RUN, w.members, (bad,) + w.row_tables[1:])
    with pytest.raises(ValueError, match="one per row"):
        CS.Member(COORD, (("x", 34), ("w", 34)), (("o0", 1),), 2, tables=(("x", 2), ("w", 2)), links=((0, 0),))
    monkeypatch.setattr(SRW, "_operand", s.operand)
    got = SRW._read_s8(ms, s.rows, s, None, ARRIVALS)
    data = [g[0] for g in got]
    data[1]["rows"]["x"] = data[1]["rows"]["x"].copy()
    data[1]["rows"]["x"][0, 0] ^= 1                                  # chunk 1 reads table 0's row 1 otherwise than chunk 0
    with pytest.raises(ValueError, match="as another member reads it"):
        CS.commit(w, data, salt_key=bytes(32))


@pytest.mark.parametrize("row_tables", [False, True])
def test_window_bytes_are_the_bytes_the_window_writes(monkeypatch, tmp_path, row_tables):
    c, _s, got = committed(monkeypatch, row_tables)
    CS.write(c, tmp_path, {"row": 101}, CS.index_doc(c.window.members, [g[1] for g in got], RUN))
    size = lambda pat: sum(p.stat().st_size for p in Path(tmp_path).glob(pat))  # noqa: E731
    b = CS.window_bytes(c.window.members, c.window.row_tables)
    assert (b["public"], b["private"]) == (size("[mt]*-public.bin"), size("[mt]*-private.bin"))
    assert (b["tables_public"], b["tables_private"]) == (size("t*-public.bin"), size("t*-private.bin"))
    x = 2 * SRW.s8_words(K) + 192
    if row_tables:
        assert b["x_private"] == x * (sum(len(m.links) for m in c.window.members) + 8)
    else:
        assert b["x_private"] == x * UNITS and b["x_public"] == 128 * UNITS


#: Llama-3.1-8B's int8 linears (K, N), its layers, and the chunk caps of record (r20261018-084708-1e58's partition file)
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
