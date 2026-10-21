"""`circuit.h1window`: a window's Program derives one unit per tile under its templates' instance query, one template per K,
numbered call by call and row block major, and `unit_reads` gives each unit's rows of the window's parameters from the
Program's references: its call's A block of its K, its linear's B rows (the same in every step), the salt, its position and
the roots, keyed by its K's template's parameters, so a mixed-K window registers in one `registered.reads`. A stand-in
tile of the H1 tile's parameter shape, its rows and its output as wide as K makes them, keeps the Program small;
`benchmarks/pouw/tests/test_h1_prove.py` runs the real one."""
from __future__ import annotations

import pytest

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array, Value

from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.core.protocols.sampled_proofs.one_stage import registered as G

V16 = Value(16)
XOR16 = PrimitiveDefinition("H1WindowTestXor16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: a ^ b, register=False)
R, C = 4, 8
TILE = (R, C)


def _tile_sig(S):
    row = Array(S.K // 16, V16)
    return ((("a", Array(S.R, row)), ("b", Array(S.C, row)), ("salt", Array(4, V16)), ("position", V16),
             ("roots", Array(2, Array(4, V16)))), Array(S.K // 32, V16))


def _tile_body(B, S, a, b, salt, position, roots):
    x = B.call(XOR16, B.call(XOR16, a[0][0], b[0][0]), B.call(XOR16, salt[0], B.call(XOR16, position, roots[0][0])))
    return coll_concat(Array(S.K // 32, V16), [B.call(XOR16, x, a[1][i]) for i in range(S.K // 32)])


TestTile = CompositeDefinition("H1WindowTestTile", 0, ("K", "R", "C"), _tile_sig, _tile_body, register=False)


def unit(k: int, tile: tuple[int, int] = TILE):
    return bind(TestTile, K=k, R=tile[0], C=tile[1])


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
             "salt": ("salt", 0), "position": ("position", u), "roots": ("roots", 0), "z": ("z", u), "receipt": ("receipt", u)}
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
    assert (sig["a/32"].n, sig["b/32"].n, sig["a/64"].n, sig["b/64"].n, sig["position"].n) == (2 * m, 24, 2 * m, 8, len(want))
    assert [t.n for t in prog.fn.signature()[1].fields] == [4, 2, 2] * 2
    assert [t.elem.n for t in prog.fn.signature()[1].fields] == [1, 1, 2] * 2
    assert [HW.unit_reads(prog, tp, u) for u in range(tp.population)] == _reads(want)
    steps = [[_b0(HW.unit_reads(prog, tp, u)) for u in range(s * len(want) // 2, (s + 1) * len(want) // 2)] for s in (0, 1)]
    assert steps[0] == steps[1], "every step's units read the linears' one copy of B"


def test_a_mixed_k_window_registers_in_one_read_each_k_under_its_own_ports():
    """One `registered.reads` over a window of K = 32 and K = 64 (a/K and b/K each a registered value, a row K/16 words):
    every unit's A and B rows under its K's template's keys, exactly the rows `unit_reads` gives, and no row for a unit of
    the other K; and in the ``--registered`` file each K's ports carry their own value's width and domain."""
    m = HW.slot(5, TILE)
    prog = HW.program(LINEARS, CALLS, m, TILE, unit=unit)
    q = HW.query(prog)
    tp = P.evaluate(prog, q)
    values = {f"{t}/{k}": k // 16 for k in (32, 64) for t in "ab"}
    got = G.reads(prog, q, values, {v: G.SCHEMA_V2 for v in values})
    want = [HW.unit_reads(prog, tp, u) for u in range(tp.population)]
    assert sorted(got) == sorted([f"a_k{k}[{i}]" for k in (32, 64) for i in range(R)]
                                 + [f"b_k{k}[{j}]" for k in (32, 64) for j in range(C)])
    for key, entry in got.items():
        assert entry["rows"] == [w[key][1] if key in w else None for w in want], key
        assert all(w[key][0] == entry["value"] for w in want if key in w), key
    hit = [[r is not None for r in got[f"a_k{k}[0]"]["rows"]] for k in (32, 64)]
    assert [sum(h) for h in hit] == [12, 4] and all(x != y for x, y in zip(*hit)), "each unit reads one K's rows"
    roots = {v: f"{i:02x}" * 64 for i, v in enumerate(values)}
    ports = G.reads_file(prog, q, values, roots, schemas={v: G.SCHEMA_V2 for v in values})["ports"]
    a32, a64 = ports["a_k32[0]"], ports["a_k64[0]"]
    assert (a32["value"], a32["words"], a32["root"], a64["value"], a64["words"], a64["root"]) == (
        "a/32", 2, roots["a/32"], "a/64", 4, roots["a/64"])
    assert len({ports[f"{t}_k{k}[0]"]["domain"] for k in (32, 64) for t in "ab"}) == 4


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
    assert list(prog.fn.defn.ports(prog.fn.S()).params) == ["h1/w/salt"], "the salt is the units' coin"
    got = HW.unit_reads(prog, tp, 5)
    assert got["a_k32[1]"] == ("h1/w/a/32", 4 + 1) and got["b_k32[0]"] == ("h1/w/b/32", 16)
    assert got["receipt"] == ("h1/w/receipt", 5)
    assert P.program_digest(prog) != P.program_digest(HW.program(LINEARS, CALLS, 8, TILE, unit=unit))


def test_the_slot_is_a_power_of_two_of_at_least_the_tiles_rows():
    assert [HW.slot(n, (4, 8)) for n in (1, 4, 5, 8, 9, 16, 17)] == [4, 4, 8, 8, 16, 16, 32]
    assert [HW.slot(n, (32, 32)) for n in (1, 32, 33)] == [32, 32, 64]
    with pytest.raises(ValueError, match="at least one token row"):
        HW.slot(0, TILE)


def test_the_tile_is_a_parameter_and_each_k_binds_its_own_unit():
    served, smoke = HW.tile_unit(4096, (32, 32)), HW.tile_unit(64, (4, 8))
    a, b = (t for _n, t in served.signature()[0][:2])
    assert (a.n, b.n, a.elem.leaves, len(served.bindings["PAIRS"])) == (32, 32, 8 * 4096, 32 * 32)
    a, b = (t for _n, t in smoke.signature()[0][:2])
    assert (a.n, b.n, a.elem.leaves, len(smoke.bindings["PAIRS"])) == (4, 8, 8 * 64, 4 * 8)
    assert HW.tile_unit(14336, (32, 32)).bindings["EXACT"].bindings["K"] == 14336


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
    receipts = [t.elem.fields[1][1].n for t in prog.fn.signature()[1].fields]
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
    def odd(k):
        if k == 32:
            return unit(k)
        return bind(CompositeDefinition("H1WindowTestOddTile", 0, ("K",), lambda S: (
            (("a", Array(R, Array(S.K // 16, V16))), ("b", Array(C, Array(S.K // 16, V16))), ("salt", Array(8, V16)),
             ("position", V16), ("roots", Array(2, Array(4, V16)))), V16),
            lambda B, S, a, b, salt, position, roots: B.call(XOR16, a[0][0], salt[0]), register=False), K=k)
    with pytest.raises(ValueError, match="takes other"):
        HW.program(LINEARS, CALLS, 8, TILE, unit=odd)
