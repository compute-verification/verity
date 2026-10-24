"""`circuit.h1window`: a window's Program derives one unit per tile under its templates' instance query, one template per K,
numbered call by call and row block major, and `unit_reads` gives each unit's rows of the window's parameters from the
Program's references: its call's A block of its K, its linear's B rows (the same in every step), its z/y record, the salt,
its position and the roots, keyed by its K's template's parameters, so a mixed-K window registers in one `registered.reads`.
A stand-in tile of the zy unit's parameter shape (`h1zy`: rows of K codes and a scale, the record, a receipt as wide as K
makes it and ``ok``) keeps the Program small; `benchmarks/pouw/tests/test_h1_prove.py` runs the real one."""
from __future__ import annotations

import cProfile
import pstats
import random

import pytest

from verity.ml.circuits import ports as PT
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.ports import In, Out, Ports
from verity.ml.circuits.refs import Coll, coll_concat
from verity.ml.circuits.types import Array, Record, Value

from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.pouw.circuit import h1zy as Z
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.core.protocols.sampled_proofs.one_stage import registered as G

V16, BIT = Value(16), Value(1)
XOR1 = PrimitiveDefinition("H1WindowTestXor1", 1, (("a", BIT), ("b", BIT)), BIT, lambda a, b: a ^ b, register=False)
LOW1 = PrimitiveDefinition("H1WindowTestLow1", 1, (("v", V16),), BIT, lambda v: v & 1, register=False)
#: a receipt fold: the coin is its key, its return a sink
FOLD1 = PrimitiveDefinition("H1WindowTestFold1", 1, (("key", BIT), ("x", BIT)), BIT, lambda key, x: key ^ x, register=False,
                            ports=Ports(params={"key": In(kind="coin")}, ret=Out(sink=True)))
#: the stand-in tile's ports, as the zy unit's: the salt its coin, its position, and its receipt a sink
TILE_PORTS = Ports(params={"salt": In(kind="coin"), "position": In(kind="position")}, ret={"receipt": Out(sink=True)})
R, C = 4, 8
TILE = (R, C)


def _first(c: Coll) -> Coll:
    return Coll(BIT, c.refs.slice(0, 1))


def _tile_sig(S, row=None, record=None, out=HW.OUTPUTS):
    row, record = (row or Z.ROW)(S.K), (record or Z.RECORD)(S.R, S.C)
    return ((("a", Array(S.R, row)), ("b", Array(S.C, row)), ("zy", record), ("salt", Array(4, BIT)), ("position", V16),
             ("roots", Array(2, Array(4, V16)))), Record((out[0], Array(S.K // 2, BIT)), (out[1], BIT)))


def _tile_body(B, S, a, b, zy, salt, position, roots, sig=_tile_sig):
    x = B.call(XOR1, B.call(XOR1, _first(a[0]), _first(b[0])), B.call(XOR1, B.call(LOW1, position), B.call(LOW1, roots[0][0])))
    w = sig(S)[1].fields[0][1].leaves
    receipt = coll_concat(Array(w, BIT),
                          [B.call(FOLD1, _first(salt), B.call(XOR1, x, Coll(BIT, a[min(1, S.R - 1)].refs.slice(i, i + 1))))
                           for i in range(w)])
    return coll_concat(sig(S)[1], [receipt, B.call(XOR1, x, _first(zy))])


def _test_tile(name: str, sig=_tile_sig, ports=TILE_PORTS):
    return CompositeDefinition(name, 0, ("K", "R", "C"), sig, lambda B, S, *args: _tile_body(B, S, *args, sig=sig),
                               register=False, ports=ports)


TestTile = _test_tile("H1WindowTestTile")


def unit(k: int, tile: tuple[int, int] = TILE):
    return bind(TestTile, K=k, R=tile[0], C=tile[1])


def _words(k: int) -> int:
    """An A or B row's 16-bit words at K: its K codes and its f32 scale (`h1zy.ROW`)."""
    return k // 2 + 2


#: a layer's weights at two contractions, as down_proj's is the intermediate size
LINEARS = (("q_proj", 32, 16), ("k_proj", 32, 8), ("down_proj", 64, 8))
#: two steps, each its own A block of each K
CALLS = tuple((lin, s) for s in range(2) for lin in ("q_proj", "k_proj", "down_proj"))


def _expected(linears, calls, m, tile=TILE):
    """Each tile's (K, first A row of a/K, first B row of b/K) in the window's numbering, written out independently of the
    Program: each K's linears' B rows once, linear after linear, and every call of a linear reading them."""
    (r, c), k_of, n_of, b0, at = tile, {}, {}, {}, {}
    for name, k, n in linears:
        k_of[name], n_of[name], b0[name] = k, n, at.get(k, 0)
        at[k] = b0[name] + n
    return [(k_of[lin], blk * m + r * ti, b0[lin] + c * tj)
            for lin, blk in calls for ti in range(m // r) for tj in range(n_of[lin] // c)]


def _reads(want, tile=TILE):
    return [{**{f"a_k{k}[{i}]": (f"a/{k}", a0 + i) for i in range(tile[0])},
             **{f"b_k{k}[{j}]": (f"b/{k}", b0 + j) for j in range(tile[1])},
             "zy": ("zy", u), "salt": ("salt", 0), "position": ("position", u), "roots": ("roots", 0),
             "receipt": ("receipt", u), "ok": ("ok", u)}
            for u, (k, a0, b0) in enumerate(want)]


def _b0(reads: dict) -> tuple[str, int]:
    """A unit's read of its first B row, at its K's key."""
    return next(v for key, v in reads.items() if key.startswith("b_k") and key.endswith("[0]"))


def test_a_mixed_k_window_with_shared_b_derives_exactly_each_calls_tiles():
    m = HW.slot(5, TILE)
    prog = HW.program(LINEARS, CALLS, m, TILE, unit=unit)
    q = HW.query(prog)
    tp = P.evaluate(prog, q)
    want = _expected(LINEARS, CALLS, m)
    ids = dict(zip((32, 64), q["params"]["templates"]))
    assert q["name"] == "Q_template_instances" and q["params"]["templates"] == [
        P.template_instance_query(HW.template(k, unit(k)))["params"]["template"] for k in (32, 64)]
    assert [[n for n, _t in t.signature()[0]] for _k, t in prog.fn.bindings["TILES"]] == [
        ["a_k32", "b_k32", *HW.SHARED], ["a_k64", "b_k64", *HW.SHARED]]
    assert m == 8 and tp.population == len(want) == 2 * (2 * 2 + 2 * 1 + 2 * 1)
    assert [(tid, first, n) for _k, tid, first, n in tp.ranges] == [(ids[k], u, 1) for u, (k, _a, _b) in enumerate(want)]
    sig = dict(prog.fn.signature()[0])
    assert [name for name, _t in prog.fn.signature()[0]] == list(HW.roles((64, 32)))
    assert (sig["a/32"].n, sig["b/32"].n, sig["a/64"].n, sig["b/64"].n, sig["zy"].n, sig["position"].n) == (
        2 * m, 24, 2 * m, 8, len(want), len(want))
    assert sig["zy"].elem == Z.RECORD(R, C) and sig["a/32"].elem == Z.ROW(32) and sig["b/64"].elem == Z.ROW(64)
    assert [t.n for t in prog.fn.signature()[1].fields] == [4, 2, 2] * 2
    assert [t.elem.fields[0][1].n for t in prog.fn.signature()[1].fields] == [16, 16, 32] * 2
    assert [HW.unit_reads(prog, tp, u) for u in range(tp.population)] == _reads(want)
    steps = [[_b0(HW.unit_reads(prog, tp, u)) for u in range(s * len(want) // 2, (s + 1) * len(want) // 2)] for s in (0, 1)]
    assert steps[0] == steps[1], "every step's units read the linears' one copy of B"


def test_a_mixed_k_window_registers_in_one_read_each_k_under_its_own_ports():
    """One `registered.reads` over a window of K = 32 and K = 64 (a/K and b/K each a registered value, a row K/2 + 2 words,
    and zy one of 3 R C words): every unit's A and B rows under its K's template's keys and its record under ``zy``, exactly
    the rows `unit_reads` gives, and no row for a unit of the other K; and in the ``--registered`` file each K's ports carry
    their own value's width and domain."""
    m = HW.slot(5, TILE)
    prog = HW.program(LINEARS, CALLS, m, TILE, unit=unit)
    q = HW.query(prog)
    tp = P.evaluate(prog, q)
    values = {**{f"{t}/{k}": _words(k) for k in (32, 64) for t in "ab"}, "zy": 3 * R * C}
    got = G.reads(prog, q, values, {v: G.SCHEMA_V2 for v in values})
    want = [HW.unit_reads(prog, tp, u) for u in range(tp.population)]
    assert sorted(got) == sorted([f"a_k{k}[{i}]" for k in (32, 64) for i in range(R)]
                                 + [f"b_k{k}[{j}]" for k in (32, 64) for j in range(C)] + ["zy"])
    for key, entry in got.items():
        assert entry["rows"] == [w[key][1] if key in w else None for w in want], key
        assert all(w[key][0] == entry["value"] for w in want if key in w), key
    assert got["zy"]["rows"] == list(range(tp.population)), "unit u reads row u of zy, whatever its K"
    hit = [[r is not None for r in got[f"a_k{k}[0]"]["rows"]] for k in (32, 64)]
    assert [sum(h) for h in hit] == [12, 4] and all(x != y for x, y in zip(*hit)), "each unit reads one K's rows"
    roots = {v: f"{i:02x}" * 64 for i, v in enumerate(values)}
    ports = G.reads_file(prog, q, values, roots, schemas={v: G.SCHEMA_V2 for v in values})["ports"]
    a32, a64, zy = ports["a_k32[0]"], ports["a_k64[0]"], ports["zy"]
    assert (a32["value"], a32["words"], a32["root"], a64["value"], a64["words"], a64["root"]) == (
        "a/32", 18, roots["a/32"], "a/64", 34, roots["a/64"])
    assert (zy["value"], zy["words"], zy["leaves"]) == ("zy", 96, 16)
    assert len({ports[f"{t}_k{k}[0]"]["domain"] for k in (32, 64) for t in "ab"}) == 4


def test_each_unit_reads_its_a_rows_and_record_at_its_calls_tables():
    """A unit's A rows are committed in its step's A table of its K and its record in its call's z table before its round's
    coin, so `table_reads` re-bases them there: tile t of call c, the j-th of its step's calls of K, on rows
    R ti .. R ti + R - 1 of its A block, reads A row i at row M j + R ti + i of ``a/<K>@<s>`` and its record at row t of
    call c's z table; its B rows and the rest are `unit_reads`'. At each step count that divides the calls."""
    m = HW.slot(5, TILE)
    prog = HW.program(LINEARS, CALLS, m, TILE, prefix="h1/w/", unit=unit)
    tp = P.evaluate(prog, HW.query(prog))
    want, units, k_of = _expected(LINEARS, CALLS, m), HW.call_units(prog), {name: k for name, k, _n in LINEARS}
    assert units == [(0, 4), (4, 2), (6, 2), (8, 4), (12, 2), (14, 2)]
    for steps in (1, 2, 3, 6):
        per = len(CALLS) // steps
        for c, (u0, n) in enumerate(units):
            lin, blk = CALLS[c]
            k, s = k_of[lin], c // per
            mine = [d for d in range(s * per, (s + 1) * per) if k_of[CALLS[d][0]] == k]
            assert HW.step_of(prog, c, steps) == (s, len(mine), m * mine.index(c))
            a = f"h1/w/a/{k}@{s}"
            assert (HW.table(prog, "a", c, steps), HW.table(prog, "zy", c)) == (a, f"h1/w/zy@{c}")
            for t in range(n):
                got, base = HW.table_reads(prog, tp, u0 + t, steps), HW.unit_reads(prog, tp, u0 + t)
                a0 = want[u0 + t][1]
                assert HW.call_of(prog, u0 + t) == (c, t)
                assert got == {**base, **{f"a_k{k}[{i}]": (a, m * mine.index(c) + a0 - blk * m + i) for i in range(R)},
                               "zy": (f"h1/w/zy@{c}", t)}
                assert all(m * mine.index(c) <= got[f"a_k{k}[{i}]"][1] < m * (mine.index(c) + 1) for i in range(R))
    with pytest.raises(ValueError, match="no unit"):
        HW.call_of(prog, tp.population)
    with pytest.raises(ValueError, match="a call's tables"):
        HW.table(prog, "b", 0)
    with pytest.raises(ValueError, match="a call's tables"):
        HW.table(prog, "a", 0)
    with pytest.raises(ValueError, match="equal blocks"):
        HW.table(prog, "a", 0, 4)


def test_a_calls_statement_opens_its_units_rows_at_its_tables_and_at_no_other_calls():
    """`table_reads_file` holds a statement to one call's units. With step 1's A table of K = 32 (calls 3 and 4, M rows
    each, call after call), each call's z table and the weights' b/32 committed as ``row/v2`` registered values under
    their own names, every A row, B row and record a unit of either call reads opens its port's root at its position
    (`registered.verify_read`, the check ``--registered`` makes of a table row), call 3's A rows in the table's first M
    and call 4's in its next M, a unit of another call reads nothing, and another step's A table, of the same rows under
    the same salts, opens none of them: its leaves are framed under its own name's domain."""
    m = HW.slot(5, TILE)
    prog = HW.program(LINEARS, CALLS, m, TILE, unit=unit)
    tp, digest, rng = P.evaluate(prog, HW.query(prog)), P.program_digest(prog), random.Random(7)

    def commit(value: str, rows: int, words: int) -> G.Registered:
        return G.Registered.commit(digest, value, [[rng.randrange(1 << 16) for _ in range(words)] for _ in range(rows)],
                                   schema=G.SCHEMA_V2)

    a = HW.table(prog, "a", 3, 2)
    assert a == HW.table(prog, "a", 4, 2) == "a/32@1"
    tabs = {a: commit(a, 2 * m, _words(32)), "b/32": commit("b/32", 24, _words(32))}
    twin = G.Registered.commit(digest, HW.table(prog, "a", 0, 2), tabs[a].rows, tabs[a].salts, schema=G.SCHEMA_V2)
    assert twin.bcs == tabs[a].bcs and twin.root != tabs[a].root
    for c, j in ((3, 0), (4, 1)):
        u0, n = HW.call_units(prog)[c]
        zy = HW.table(prog, "zy", c)
        tabs[zy] = commit(zy, n, 3 * R * C)
        file = HW.table_reads_file(prog, tp, c, {v: r.root.hex() for v, r in tabs.items()}, ports=HW.window_ports(prog),
                                   steps=2)
        assert file["format"] == G.READS and sorted(file["ports"]) == sorted(
            [f"a_k32_{i}" for i in range(R)] + [f"b_k32_{j}" for j in range(C)] + ["zy"])
        for port, e in file["ports"].items():
            reg, dom = tabs[e["value"]], G.domain(digest, e["value"], e["words"], e["leaves"])
            assert (e["leaves"], e["words"], e["domain"], e["row_schema"]) == (reg.leaves, reg.words,
                                                                                 dom.domain_id.hex(), G.SCHEMA_V2), port
            assert all(r is None for u, r in enumerate(e["reads"]) if not u0 <= u < u0 + n), port
            assert len(e["reads"]) == u0 + n < tp.population, "no trailing nulls: a unit past reads reads none"
            for u in range(u0, u0 + n):
                p = e["reads"][u]
                assert G.verify_read(dom, reg.root, p, reg.bcs[p], reg.path(p), G.SCHEMA_V2), (port, u)
                if e["value"] == a:
                    assert m * j <= p < m * (j + 1), (port, u)
                    assert not G.verify_read(dom, reg.root, p, twin.bcs[p], twin.path(p), G.SCHEMA_V2), (port, u)
    with pytest.raises(ValueError, match="roots the verifier does not hold"):
        HW.table_reads_file(prog, tp, 3, {"b/32": "00" * 64}, steps=2)


def test_a_served_windows_call_tables_bind_its_open_root_and_the_weights_none():
    """Under a served window's open root, `table_reads_file` gives the call's A and z tables the domains that bind it, the
    window's own rows (PRF-127), and the weights' b/32 the one that binds none: the domains credit gives them
    (`credit.domain_of`), and the same reads as with no window."""
    m = HW.slot(5, TILE)
    prog = HW.program(LINEARS, CALLS, m, TILE, unit=unit)
    tp, digest, window = P.evaluate(prog, HW.query(prog)), P.program_digest(prog), "ab" * 64
    c = 3
    roots = {HW.table(prog, "a", c, 2): "01" * 64, HW.table(prog, "zy", c): "02" * 64, "b/32": "03" * 64}
    bare = HW.table_reads_file(prog, tp, c, roots, ports=HW.window_ports(prog), steps=2)
    file = HW.table_reads_file(prog, tp, c, roots, ports=HW.window_ports(prog), window=window, steps=2)
    assert sorted(file["ports"]) == sorted(bare["ports"])
    for port, e in file["ports"].items():
        held = e["value"] == "b/32"
        assert e["domain"] == G.domain(digest, e["value"], e["words"], e["leaves"],
                                       window=None if held else window).domain_id.hex(), port
        assert e["domain"] == Cr.domain_of(digest, window, e), port
        assert (e["domain"] == bare["ports"][port]["domain"]) == held, port
        assert {k: v for k, v in e.items() if k != "domain"} == {k: v for k, v in bare["ports"][port].items()
                                                                  if k != "domain"}, port


def test_a_calls_receipt_table_is_its_statements_output_row_o0_when_registered():
    """With call c's receipt table ``receipt@<c>`` among the roots, `table_reads_file` reads its units' receipts there as the
    statement's output row o0 (`RECEIPT_PORT`), unit u0 + t at row t, one row of the tile's receipt width in 16-bit words
    under the window-bound domain the served path commits it under, and every read opens that tree; without it the file
    has no o0, and a tile whose receipt is no whole 16-bit words is refused rather than read at a truncated width."""
    m = HW.slot(5, TILE)
    prog = HW.program(LINEARS, CALLS, m, TILE, unit=unit)
    tp, digest, rng, w, c = P.evaluate(prog, HW.query(prog)), P.program_digest(prog), random.Random(13), "ab" * 64, 3
    u0, n = HW.call_units(prog)[c]
    rc = HW.table(prog, HW.RECEIPTS, c)
    assert rc == f"receipt@{c}"
    roots = {HW.table(prog, "a", c, 2): "01" * 64, HW.table(prog, "zy", c): "02" * 64, "b/32": "03" * 64}
    assert HW.RECEIPT_PORT not in HW.table_reads_file(prog, tp, c, roots, ports=HW.window_ports(prog), window=w,
                                                      steps=2)["ports"]
    words = 32 // 2 // 16
    reg = G.Registered.commit(digest, rc, [[rng.randrange(1 << 16) for _ in range(words)] for _ in range(n)],
                              schema=G.SCHEMA_V2, window=w)
    file = HW.table_reads_file(prog, tp, c, {**roots, rc: reg.root.hex()}, ports=HW.window_ports(prog), window=w,
                               steps=2)["ports"]
    e = file[HW.RECEIPT_PORT]
    assert (e["value"], e["leaves"], e["words"], e["root"], e["domain"]) == (
        rc, n, words, reg.root.hex(), reg.domain.domain_id.hex())
    assert e["domain"] == Cr.domain_of(digest, w, e)
    assert e["reads"] == [u - u0 if u0 <= u < u0 + n else None for u in range(u0 + n)] and u0 + n < tp.population
    assert all(G.verify_read(reg.domain, reg.root, t, reg.bcs[t], reg.path(t), G.SCHEMA_V2) for t in range(n))

    odd_tile = _test_tile("H1WindowTestOddReceiptTile", sig=lambda S: _tile_sig(S)[:1] + (
        Record((HW.OUTPUTS[0], Array(S.K // 2 + 1, BIT)), (HW.OUTPUTS[1], BIT)),))
    odd = HW.program(LINEARS, CALLS, m, TILE, unit=lambda k: bind(odd_tile, K=k, R=R, C=C))
    with pytest.raises(ValueError, match="no receipt of whole 16-bit words"):
        HW.table_reads_file(odd, P.evaluate(odd, HW.query(odd)), c, {**roots, rc: reg.root.hex()}, steps=2)


def test_a_window_at_k_32_and_322_reads_each_row_under_its_own_port():
    """`window_ports` gives each A and B read of a window of K = 32 and K = 322 its own port: without the separator, A row
    20 at K = 32 (``a_k32[20]``) and A row 0 at K = 322 (``a_k322[0]``) would both be ``a_k3220``, and the ``--registered``
    file would hold one port for two reads. A tile of one B row reads it under the parameter's name, port row 0's."""
    tile = (32, 32)
    prog = HW.program((("q_proj", 32, 32), ("down_proj", 322, 32)), (("q_proj", 0), ("down_proj", 0)), 32, tile,
                      unit=lambda k: unit(k, tile))
    q = HW.query(prog)
    values = {f"{t}/{k}": _words(k) for k in (32, 322) for t in "ab"}
    schemas = {v: G.SCHEMA_V2 for v in values}
    reads = G.reads(prog, q, values, schemas)
    ports = HW.window_ports(prog)
    assert set(ports) == set(reads) and len(reads) == 2 * (32 + 32)
    assert len(set(ports.values())) == len(ports)
    assert (ports["a_k32[20]"], ports["a_k322[0]"], ports["b_k322[31]"]) == ("a_k32_20", "a_k322_0", "b_k322_31")
    assert len({key.replace("[", "").replace("]", "") for key in reads}) < len(reads), "unseparated, two reads share a port"
    roots = {v: f"{i:02x}" * 64 for i, v in enumerate(values)}
    file = G.reads_file(prog, q, values, roots, ports=ports, schemas=schemas)["ports"]
    assert sorted(file) == sorted(ports.values())
    assert all(file[ports[key]]["reads"] == reads[key]["rows"] for key in reads)

    one = HW.program((("q_proj", 32, 2),), (("q_proj", 0),), 2, (2, 1), unit=lambda k: unit(k, (2, 1)))
    got = G.reads(one, HW.query(one), {"a/32": _words(32), "b/32": _words(32)}, {"a/32": G.SCHEMA_V2, "b/32": G.SCHEMA_V2})
    assert set(HW.window_ports(one)) == set(got) and HW.window_ports(one)["b_k32"] == "b_k32_0"


#: what a unit reads that is no registered value: its coin, its position, its roots and its outputs
UNREGISTERED = ("salt", "position", "roots", *HW.OUTPUTS)


@pytest.mark.parametrize("tile", [(1, C), (R, 1)], ids=["R=1", "C=1"])
def test_a_lone_row_is_read_under_its_parameters_name_as_registered_reads_keys_it(tile):
    """With R = 1 (or C = 1) a unit's one A (or B) row is its whole parameter, which `registered.reads` keys by the
    parameter's name (``a_k<K>``), not ``a_k<K>[0]``: `unit_reads` and `table_reads` key it so too, at the rows
    `registered.reads` gives, and `table_reads_file` through `window_ports` alone names its port ``a_k<K>_0``, with no
    alias of the two keys."""
    lone = "a" if tile[0] == 1 else "b"
    prog = HW.program(LINEARS, CALLS, HW.slot(5, tile), tile, unit=lambda k: unit(k, tile))
    q = HW.query(prog)
    tp = P.evaluate(prog, q)
    values = {**{f"{t}/{k}": _words(k) for k in (32, 64) for t in "ab"}, "zy": 3 * tile[0] * tile[1]}
    got = G.reads(prog, q, values, {v: G.SCHEMA_V2 for v in values})
    want = [HW.unit_reads(prog, tp, u) for u in range(tp.population)]
    assert {f"{lone}_k32", f"{lone}_k64"} <= set(got) and not [k for k in got if k.startswith(f"{lone}_k") and "[" in k]
    assert set(got) == {key for w in want for key in w if key not in UNREGISTERED}
    for key, e in got.items():
        assert e["rows"] == [w[key][1] if key in w else None for w in want], key
    ports = HW.window_ports(prog)
    assert set(ports) == set(got) - {"zy"} and ports[f"{lone}_k32"] == f"{lone}_k32_0"
    c = 3
    assert set(HW.table_reads(prog, tp, HW.call_units(prog)[c][0], 2)) == set(want[HW.call_units(prog)[c][0]])
    roots = {HW.table(prog, "a", c, 2): "01" * 64, HW.table(prog, "zy", c): "02" * 64, "b/32": "03" * 64}
    file = HW.table_reads_file(prog, tp, c, roots, ports=ports, steps=2)["ports"]
    assert sorted(file) == sorted([ports[key] for key in got if "_k32" in key] + ["zy"])
    assert file[f"{lone}_k32_0"]["value"] == (HW.table(prog, "a", c, 2) if lone == "a" else "b/32")


def test_under_a_windows_open_root_a_calls_file_opens_the_tables_committed_in_that_window():
    """``window``: with the window's open root W, `table_reads_file` gives the call's A and z tables the registered domain
    that binds W (`registered.domain`'s ``window``), as the served path commits them, and the weights' b/32 the one that
    binds none; without it every domain binds none, as before. Every read opens the trees committed so, and the call's A
    table committed in another window, of the same rows under the same salts, opens none of them at its own root."""
    m = HW.slot(5, TILE)
    prog = HW.program(LINEARS, CALLS, m, TILE, unit=unit)
    tp, digest, rng = P.evaluate(prog, HW.query(prog)), P.program_digest(prog), random.Random(11)
    c, w, other = 3, "ab" * 64, "cd" * 64
    a, zy = HW.table(prog, "a", c, 2), HW.table(prog, "zy", c)
    u0, n = HW.call_units(prog)[c]

    def commit(value: str, rows: int, words: int, window: str | None) -> G.Registered:
        return G.Registered.commit(digest, value, [[rng.randrange(1 << 16) for _ in range(words)] for _ in range(rows)],
                                   schema=G.SCHEMA_V2, window=window)

    tabs = {a: commit(a, 2 * m, _words(32), w), zy: commit(zy, n, 3 * R * C, w), "b/32": commit("b/32", 24, _words(32), None)}
    roots = {v: r.root.hex() for v, r in tabs.items()}
    file = HW.table_reads_file(prog, tp, c, roots, ports=HW.window_ports(prog), window=w, steps=2)["ports"]
    bare = HW.table_reads_file(prog, tp, c, roots, ports=HW.window_ports(prog), steps=2)["ports"]
    twin = G.Registered.commit(digest, a, tabs[a].rows, tabs[a].salts, schema=G.SCHEMA_V2, window=other)
    assert twin.bcs == tabs[a].bcs and twin.root != tabs[a].root
    for port, e in file.items():
        reg = tabs[e["value"]]
        assert e["domain"] == reg.domain.domain_id.hex(), port
        assert (e["domain"] == bare[port]["domain"]) == (e["value"] == "b/32"), port
        assert {k: v for k, v in e.items() if k != "domain"} == {k: v for k, v in bare[port].items() if k != "domain"}
        assert bare[port]["domain"] == G.domain(digest, e["value"], e["words"], e["leaves"]).domain_id.hex(), port
        for u in range(u0, u0 + n):
            p = e["reads"][u]
            assert G.verify_read(reg.domain, reg.root, p, reg.bcs[p], reg.path(p), G.SCHEMA_V2), (port, u)
            if e["value"] == a:
                assert not G.verify_read(reg.domain, twin.root, p, twin.bcs[p], twin.path(p), G.SCHEMA_V2), (port, u)


def _operations(steps: int) -> int:
    """The function calls of building a window of `LINEARS` over ``steps`` steps (`program`, `query`, `P.evaluate`) and
    reading every unit's rows (`unit_reads`)."""
    calls = tuple((lin, s) for s in range(steps) for lin in ("q_proj", "k_proj", "down_proj"))
    profile = cProfile.Profile()
    profile.enable()
    prog = HW.program(LINEARS, calls, 8, TILE, unit=unit)
    tp = P.evaluate(prog, HW.query(prog))
    [HW.unit_reads(prog, tp, u) for u in range(tp.population)]
    profile.disable()
    return pstats.Stats(profile).total_calls


def test_a_window_and_its_units_reads_cost_linearly_in_its_calls():
    """Eight times the calls cost at most 8 × 1.5 times the function calls, counted rather than timed. It fails when the
    window's build re-derives a per-call fact for each tile (the Builder's E_REINTERPRET check of the window's result
    summing its fields' leaves at each read, `types.field_starts`), or `unit_reads` scans the window's nodes for each unit
    (`h1window._scan`): either alone grows this window's count 15 to 23 times, and it took an 8B served window of 2,304
    calls about an hour."""
    _operations(1)
    small, large = _operations(8), _operations(64)
    assert large <= 8 * 1.5 * small, f"{large} calls at 192 window calls, {small} at 24"


def test_a_one_k_window_is_h1_proves_every_call_reading_one_a_block():
    linears = (("k_proj", 32, 16), ("q_proj", 32, 16))
    calls = (("k_proj", 0), ("q_proj", 0))
    prog = HW.program(linears, calls, 8, TILE, unit=unit)
    tp = P.evaluate(prog, HW.query(prog))
    want = _expected(linears, calls, 8)
    assert tp.population == len(want) == 8 and len(HW.query(prog)["params"]["templates"]) == 1
    assert [HW.unit_reads(prog, tp, u)["b_k32[0]"] for u in range(8)] == [("b/32", b) for _k, _a, b in want]


def test_the_parameters_take_the_names_a_served_path_gives_them():
    prog = HW.program(LINEARS, CALLS, 8, TILE, prefix="h1/w/", unit=unit)
    tp = P.evaluate(prog, HW.query(prog))
    assert [n for n, _t in prog.fn.signature()[0]] == ["h1/w/" + r for r in HW.roles((32, 64))]
    assert {n: p.kind for n, p in prog.fn.defn.ports(prog.fn.S()).params.items()} == {
        "h1/w/salt": "coin", "h1/w/position": "position"}, "the salt is the units' coin, and position their positions"
    got = HW.unit_reads(prog, tp, 5)
    assert got["a_k32[1]"] == ("h1/w/a/32", 4 + 1) and got["b_k32[0]"] == ("h1/w/b/32", 16)
    assert got["receipt"] == ("h1/w/receipt", 5)
    assert P.program_digest(prog) != P.program_digest(HW.program(LINEARS, CALLS, 8, TILE, unit=unit))


def test_every_units_receipt_is_a_sink_so_the_window_is_no_seed():
    """With the salt its coin port and the positions its position port, a window whose return held no sink would be a
    seed, whose body no port rule follows (`ports._coin_role`), and the tiles' coin flow would go unchecked: each unit's
    receipt, exactly its leaves (``ok`` is none), is a sink of the window's result, and a unit that returns none is refused."""
    prog = HW.program(LINEARS, CALLS, 8, TILE, unit=unit)
    want, at = [], 0
    for t in prog.fn.signature()[1].fields:
        for _ in range(t.n):
            want.append((at, at + t.elem.fields[0][1].leaves))
            at += t.elem.leaves
    assert [(lo, hi) for lo, hi, o, _w in PT.port_types(prog.fn).ret if o.sink] == want and len(want) == 16
    assert PT._coin_role(prog.fn) is None
    bare = _test_tile("H1WindowTestBareTile", ports=Ports(params={"salt": In(kind="coin")}))
    with pytest.raises(ValueError, match="returns no sink"):
        HW.program(LINEARS, CALLS, 8, TILE, unit=lambda k: bind(bare, K=k, R=R, C=C))


def test_the_slot_is_a_power_of_two_of_at_least_the_tiles_rows():
    assert [HW.slot(n, (4, 8)) for n in (1, 4, 5, 8, 9, 16, 17)] == [4, 4, 8, 8, 16, 16, 32]
    assert [HW.slot(n, (32, 32)) for n in (1, 32, 33)] == [32, 32, 64]
    with pytest.raises(ValueError, match="at least one token row"):
        HW.slot(0, TILE)


def test_the_tile_is_a_parameter_and_each_k_binds_its_own_unit():
    """`tile_unit` is the zy unit over the H1 tile at its K and R × C (`h1_tile`): its rows are the tile's codes and their
    scale, its record z then y, and it returns the receipt and ``ok``."""
    served, smoke = HW.tile_unit(4096, (32, 32)), HW.tile_unit(64, (4, 8))
    a, b, zy = (t for _n, t in served.signature()[0][:3])
    assert served.defn.id == "PouwH1TileZy_v1" and served.bindings["TILE"] == HW.h1_tile(4096, (32, 32))
    assert (a.n, b.n, a.elem.leaves, zy.leaves, len(served.bindings["TILE"].bindings["PAIRS"])) == (
        32, 32, 8 * 4096 + 32, 48 * 32 * 32, 32 * 32)
    a, b, zy = (t for _n, t in smoke.signature()[0][:3])
    assert (a.n, b.n, a.elem.leaves, zy.leaves, len(smoke.bindings["TILE"].bindings["PAIRS"])) == (4, 8, 8 * 64 + 32, 1536, 32)
    assert [n for n, _t in smoke.signature()[0]] == list(HW.TILE_PARAMS)
    assert [n for n, _t in smoke.signature()[1].fields] == list(HW.OUTPUTS) and smoke.signature()[1].fields[1][1].leaves == 1
    assert HW.tile_unit(14336, (32, 32)).bindings["TILE"].bindings["EXACT"].bindings["K"] == 14336


@pytest.mark.slow
def test_the_real_tile_composes_at_mixed_k_and_at_the_served_32_by_32():
    """The H1 tile unit itself (`tile_unit`): a two-step window of K = 64 and K = 128 linears derives each call's tiles
    under its two templates, and the served path's 32 × 32 tile one unit a 32 × 32 block. What the stand-in can't show is
    that the real tile's bit-leaf rows and K-wide receipts compose into the window's parameters and result. The Program
    traces the tile's seed and noise circuits, about 35 s at any K or tile, and each partition about 7 s."""
    linears = (("q_proj", 64, 16), ("k_proj", 64, 8), ("down_proj", 128, 8))
    calls = tuple((lin, s) for s in range(2) for lin in ("q_proj", "k_proj", "down_proj"))
    prog = HW.program(linears, calls, 8, TILE)
    tp = P.evaluate(prog, HW.query(prog))
    want = _expected(linears, calls, 8)
    ids = dict(zip((64, 128), HW.query(prog)["params"]["templates"]))
    assert [(tid, first, n) for _k, tid, first, n in tp.ranges] == [(ids[k], u, 1) for u, (k, _a, _b) in enumerate(want)]
    assert [HW.unit_reads(prog, tp, u) for u in range(tp.population)] == _reads(want)
    receipts = [t.elem.fields[0][1].n for t in prog.fn.signature()[1].fields]
    assert receipts[2] > receipts[0], "down_proj's receipt is K-wide"

    served = HW.program((("q_proj", 64, 64),), (("q_proj", 0),), HW.slot(20, (32, 32)), (32, 32))
    tp = P.evaluate(served, HW.query(served))
    want = _expected((("q_proj", 64, 64),), (("q_proj", 0),), 32, (32, 32))
    assert tp.population == len(want) == 2
    assert [HW.unit_reads(served, tp, u) for u in range(2)] == _reads(want, (32, 32))


@pytest.mark.parametrize("linears,calls,m,tile,why", [
    (LINEARS, CALLS, 6, TILE, "power of two"),
    (LINEARS, CALLS, 2, TILE, "power of two"),
    ((("q_proj", 32, 12),), (("q_proj", 0),), 8, TILE, "whole blocks"),
    (LINEARS, (("q_proj", 0), ("k_proj", 2), ("down_proj", 0)), 8, TILE, "every A block of K = 32"),
    (LINEARS, CALLS + (("v_proj", 0),), 8, TILE, "every call is of a linear"),
    (LINEARS + (("v_proj", 32, 8),), CALLS, 8, TILE, "every linear is called"),
    (LINEARS + (("q_proj", 64, 8),), CALLS, 8, TILE, "each named once"),
    (LINEARS, (), 8, TILE, "each named once, and calls"),
    (LINEARS, CALLS, 8, (8, 8), "no 8 × 8 tile"),
])
def test_a_layout_off_its_fixed_slots_is_refused(linears, calls, m, tile, why):
    with pytest.raises(ValueError, match=why):
        HW.program(linears, calls, m, tile, unit=unit)


def test_a_unit_whose_shared_parameters_differ_by_k_is_refused():
    def odd_sig(S):
        params, ret = _tile_sig(S)
        return params[:3] + (("salt", Array(8, BIT)),) + params[4:], ret

    odd = _test_tile("H1WindowTestOddTile", odd_sig)
    with pytest.raises(ValueError, match="takes other"):
        HW.program(LINEARS, CALLS, 8, TILE, unit=lambda k: unit(k) if k == 32 else bind(odd, K=k, R=R, C=C))


def _z_alone(r: int, c: int) -> Record:
    return Record(("z", Array(r * c, Z.S32)))


@pytest.mark.parametrize("name,sig,why", [
    ("H1WindowTestZAlone", lambda S: _tile_sig(S, record=_z_alone), "a record of z alone leaves the served y unproved"),
    ("H1WindowTestYAlone", lambda S: _tile_sig(S, record=lambda r, c: Record(("y", Array(r * c, Z.BF16)))), "512-bit record"),
    ("H1WindowTestCodesOnly", lambda S: _tile_sig(S, row=lambda k: Array(k, Z.S8)), "and the row's f32 scale"),
    ("H1WindowTestZOut", lambda S: _tile_sig(S, out=("z", "receipt")), "not \\('receipt', 'ok'\\)"),
])
def test_a_unit_that_doesnt_prove_its_record_against_its_scaled_rows_is_refused(name, sig, why):
    """The window's unit reads its record as z then y and its rows with their scales, and returns its receipt and ``ok``: a
    unit whose record port carries z alone (the y it would leave unproved is the served one), or y alone, or whose rows have
    no scale to dequantize by, or that returns z (the record's z, not a unit output, is the served value), is refused."""
    with pytest.raises(ValueError, match=why):
        HW.program(LINEARS, CALLS, 8, TILE, unit=lambda k: bind(_test_tile(name, sig), K=k, R=R, C=C))


def test_a_window_whose_receipts_padding_spills_into_another_block_is_staged():
    """The H1 tile's receipt is an output row of 8 (KS + m) bytes, and the Lean verifier takes a v2 row of any length
    (#3148): a window of the real tile unit at K = 384, 416, 896 and 928, whose receipts end past byte 111 of a block, is
    staged."""
    ks = (384, 416, 896, 928)
    assert [HW.tile_unit(k, TILE).signature()[1].fields[0][1].leaves // 8 for k in ks] == [112, 120, 240, 248]
    prog = HW.program(tuple((f"l{k}", k, C) for k in ks), tuple((f"l{k}", 0) for k in ks), R, TILE)
    assert HW.call_units(prog) == [(0, 1), (1, 1), (2, 1), (3, 1)]
