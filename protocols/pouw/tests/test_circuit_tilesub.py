"""Tile units cut into sub-tiles (`verity_pouw.circuit.tilesub`, ``Shape.tile_sub``): a row's chunk digests and the tile leaf
over them are `PearlTileDigest` composed, `pearl_c.leaf` byte for byte and the pinned tile-chunks rows; the cut roots give
today's roots' outputs, plain and hiding, for Pearl-C on the H100, sm_120 and sm_120 unpromoted and for Pearl-C4, with and
without the row cut, with the same computing gates, and their partition computes every value in exactly one unit; a sub-tile
is one call of today's tile template at a × b, its outputs wired one per committed row; each piece's ports are committed rows
of `leaves`' schemas, short or whole blocks, and none reads a public input region; a drawn sub-tile is proved with its whole
tile; and with ``tile_sub`` unset nothing is the cut's."""
from __future__ import annotations

import json
import random
import types
from collections import Counter
from dataclasses import replace

import pytest

import circuit_fixtures as CF
import pouw_rows_vectors as V
from verity.commitments.rowleaf import ROLE_X, bit_row, row_bytes
from verity.evaluation import evaluate
from verity.ir import partition_object as PO
from verity.ir.codec import descriptor_id
from verity.ir.cut import CallGraph
from verity.ir.program import Program
from verity.randomness import derive
from verity_pouw.circuit import leaves as L
from verity_pouw.circuit import pc4, pc8, plan, rowk, tilesub
from verity_pouw.circuit.anchors import public_ports
from verity_pouw.protocol import DomainError
from verity_pouw.schemes import pearl_c as C

PINNED = json.loads(V.OUT.read_text())


def _halves(words) -> list[int]:
    """A PearlRowChunks output's 32-bit words as the tile-chunks row's 16-bit words, low half first."""
    return [h for w in words for h in (w & 0xFFFF, w >> 16)]


def _flat(rows) -> list[int]:
    return [v for r in rows for v in r]


# ---- the digest units --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tm, tn, sb", [(1, 1, 1), (2, 4, 2), (2, 33, 1), (3, 64, 16), (1, 128, 8)])
def test_the_digest_units_compose_to_the_tile_digest(tm, tn, sb):
    """Each row's `PearlRowChunks` is its tile-chunks row (`leaves.row_chunk_digests`, then zero words), and
    `PearlTileLeaf` over the rows' is `pearl_c.leaf` and `PearlTileDigest`'s output: a partial chunk, one to four chunks a
    row, several rows."""
    rng = random.Random(tm * 1000 + tn * 10 + sb)
    c, u = ([[rng.getrandbits(32) for _ in range(tn)] for _ in range(tm)] for _ in range(2))
    rows = [evaluate(L.row_chunks(tn, sb), c[r], u[r]) for r in range(tm)]
    for r, out in enumerate(rows):
        assert _halves(out) == L.words_tile_chunks(L.row_chunk_digests(c[r], u[r]))
    got = evaluate(L.tile_leaf(tm, tn), _flat(rows))
    assert L.digest_bytes(got) == C.leaf(types.SimpleNamespace(c=c, u=u))
    assert got == evaluate(L.tile_digest(tm, tn), _flat(c), _flat(u))


@pytest.mark.parametrize("case", [c for c in PINNED["cases"] if c["row"] == "tile-chunks"], ids=lambda c: c["case"])
def test_the_gates_compute_the_pinned_tile_chunks_rows(case):
    """`PearlRowChunks` gives the pinned tile-chunks row from the case's C̃ and U."""
    x = V.inputs(case["case"], "tile-chunks", case["k"])
    words = _halves(evaluate(L.row_chunks(case["k"], 1), x["c"], x["u"]))
    assert L.commit("tile-chunks", words, V.shake(case["case"], "salt", L.INST.salt_bytes)).hex() == case["bc"]


def test_each_row_is_its_elements_bits():
    """sub-cu's words are `rowleaf.bit_row`'s of a sub-tile row's C̃ then U words and tile-chunks' of its digests' bytes then
    zeros, each of its schema's word count; their pieces write them as outputs (sub-cu a sub-tile's C̃ and U, row by row),
    and PearlRowChunks' zero words are constants in gates."""
    rng = random.Random(5)
    for b in tilesub.SUBS:
        c, u = [rng.getrandbits(32) for _ in range(b)], [rng.getrandbits(32) for _ in range(b)]
        words = L.words_sub_cu(c, u)
        assert (row_bytes(words, 16), 16 * len(words)) == bit_row((c, 32), (u, 32)) and len(words) == L.ROW_WORDS["sub-cu"](b)
    for n in range(1, L.TILE_CHUNKS + 1):
        ds = [bytes(rng.getrandbits(8) for _ in range(32)) for _ in range(n)]
        words = L.words_tile_chunks(ds)
        assert row_bytes(words, 16) == b"".join(ds) + bytes(32 * (L.TILE_CHUNKS - n)) and len(words) == 64
    for tn, sb in ((4, 2), (33, 1), (128, 16)):
        b = L.row_chunks(tn, sb).body
        n = 8 * -(-tn // C.CHUNK)
        fill = [b.ret[i][1] for i in range(n, L.CHUNKS_U32)]
        assert len(fill) == L.CHUNKS_U32 - n
        assert {(b.nodes[j].fn.family, b.nodes[j].fn.evaluate()) for j in fill} <= {("Const", 0)}
    for bad in ([], [0] * 5):
        with pytest.raises(ValueError):
            L.words_tile_chunks([bytes(32)] * len(bad))
    with pytest.raises(ValueError):
        L.words_sub_cu([0], [0, 0])
    for args in ((6, 4), (160, 16), (0, 1)):
        with pytest.raises(ValueError):
            L.row_chunks(*args)
    with pytest.raises(ValueError):
        L.tile_leaf(1, 129)


def test_each_cut_row_is_short_or_whole_blocks():
    """Every sub-cu row (4b words, b in `SUBS`) and the tile-chunks row are one SHA-512 block with its padding (at most 55
    words) or whole blocks (64 | words), the v2 rows C-Flock's widened scope takes."""
    for n in [L.ROW_WORDS["sub-cu"](b) for b in tilesub.SUBS] + [L.ROW_WORDS["tile-chunks"](0)]:
        assert 2 * n + 17 <= 128 or n % 64 == 0, n
    assert L.ROW_SCHEMA["sub-cu"] == L.ROW_SCHEMA["tile-chunks"] == L.ROW_SCHEMA["tile"]


# ---- the roots ---------------------------------------------------------------------------------------------------------

SHAPES = [pc8.Shape(4, 128, 8, tile=(2, 4), dev="h100", tile_sub=(2, 2)),
          pc8.Shape(4, 128, 8, tile=(2, 4), dev="sm120", tile_sub=(2, 2)),
          pc8.Shape(4, 64, 8, tile=(2, 4), dev="sm120-unpromoted", tile_sub=(2, 2)),
          pc8.Shape(2, 128, 4, tile=(2, 4), dev="sm120", tile_sub=(1, 1)),
          pc4.Shape(4, 64, 8, tile=(2, 4), tile_sub=(2, 2)),
          pc4.Shape(4, 64, 8, tile=(2, 4), tile_sub=(1, 1)),
          pc4.Shape(1, 64, 33, tile=(1, 33), tile_sub=(1, 1)),
          pc8.Shape(2, 384, 4, tile=(1, 2), dev="sm120", row_k=128, tile_sub=(1, 1)),
          pc4.Shape(2, 256, 4, tile=(1, 2), row_k=64, tile_sub=(1, 2))]
IDS = [f"{getattr(s, 'dev', 'pc4')}-{s.tile_sub[0]}x{s.tile_sub[1]}{'-rowk' if s.row_k else ''}-n{s.n}" for s in SHAPES]


def _mod(shape):
    return pc8 if isinstance(shape, pc8.Shape) else pc4


def _sub_tile(shape):
    return (tilesub.sub_tile8(shape.k, shape.tile_sub, shape.dev) if _mod(shape) is pc8
            else tilesub.sub_tile4(shape.k, shape.tile_sub))


def _inputs(shape) -> tuple:
    return (CF.inputs8(shape.dev, shape.k, shape.m, shape.n, "pc8-tilesub") if _mod(shape) is pc8
            else CF.inputs4(shape.k, shape.m, shape.n, "pc4-tilesub"))


@pytest.mark.parametrize("hidden", [False, True], ids=["plain", "hidden"])
@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
def test_the_cut_roots_give_todays_outputs(shape, hidden):
    """The cut's `call` and `check` give today's roots' outputs on a scheme's honest block (the call reading the weight's
    committed rows), plain and hiding, so a hiding root's tile digests are `PearlTileDigest`'s; the weight is the shape's
    without the cut."""
    mod, cut = _mod(shape), replace(shape, hidden=hidden)
    today = replace(cut, tile_sub=None)
    a, ea, b, eb, fa, fb, g = _inputs(shape)
    assert mod.weight(cut).id == mod.weight(today).id
    w = evaluate(mod.weight(today), b, eb, fa, fb, g)
    n, k = shape.n, shape.k
    committed = (w[:n * k], w[n * k:-n]) if mod is pc8 else (w[:n * k], w[n * k:n * k + n * k // 16], w[n * k + n * k // 16:-n])
    assert evaluate(mod.call(cut), a, ea, fa, fb, *committed) == evaluate(mod.call(today), a, ea, fa, fb, *committed)
    got = evaluate(mod.check(cut), a, ea, b, eb, fa, fb, g)
    assert got == evaluate(mod.check(today), a, ea, b, eb, fa, fb, g)
    assert got[-(shape.m + n):] == (1,) * (shape.m + n)
    if hidden:
        c, u = (evaluate(mod.check(replace(today, hidden=False)), a, ea, b, eb, fa, fb, g)[f * shape.m * n:(f + 1) * shape.m * n]
                for f in range(2))
        rows = lambda x: [x[i * n:(i + 1) * n] for i in range(shape.m)]  # noqa: E731
        assert got[:-(shape.m + n)] == CF.tile_digests(shape.tile, rows(c), rows(u))


def _counts(shape, side: str) -> dict[str, int]:
    (tm, tn), (sa, sb), (m, n) = shape.tile, shape.tile_sub, (shape.m, shape.n)
    rows = {"row_b": n, "row_a": m}
    if shape.row_k is None:
        out = {r: rows[r] for r in pc4.ROLES[side] if r in rows}
    else:
        segs = shape.k // shape.row_k
        per = {"head": 1, "first": 1, "form": segs - 1, "tail": 1}
        out = {f"{r}_{p}": rows[r] * c for r in rows if r in pc4.ROLES[side] for p, c in per.items() if c}
    out["tile"] = (m // sa) * (n // sb)
    if shape.hidden:
        out |= {"row_digest": m * (n // tn), "tile_leaf": (m // tm) * (n // tn)}
    return out


@pytest.mark.parametrize("hidden", [False, True], ids=["plain", "hidden"])
@pytest.mark.parametrize("side", ["check", "call"])
@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
def test_the_cut_computes_every_value_in_exactly_one_unit(shape, side, hidden):
    """A unit per sub-tile, and under hiding one per row of A and tile column (its chunk digests) and one per tile (its
    leaf), each on its grid; every sub-tile is `tilesub`'s sub-tile at a × b; `validate_unit_cut` certifies every gate once
    with no recompute, the partition object verifies, and the cut computes today's gates, its only extra free gates
    constants (each unit's own: a sub-tile's zero, a row digest's padding and the tile-chunks row's zero words)."""
    mod, cut = _mod(shape), replace(shape, hidden=hidden)
    build = {"check": mod.check, "call": mod.call}[side]
    fn = build(cut)
    prog, q = Program(fn), mod.query(cut, side)
    us = mod.units(fn)
    assert Counter(u.role for u in us) == _counts(cut, side)
    (tm, tn), (sa, sb), (m, n) = shape.tile, shape.tile_sub, (shape.m, shape.n)
    grid = {"tile": (m // sa, n // sb), "row_digest": (m, n // tn), "tile_leaf": (m // tm, n // tn)}
    for role in tilesub.ROLES:
        rs, cs = grid[role]
        assert [u.where for u in us if u.role == role] == ([(r, c) for r in range(rs) for c in range(cs)]
                                                           if role in _counts(cut, side) else [])
    sub_tile = _sub_tile(shape)
    assert {nd.fn.id for nd in fn.body.nodes if nd.fn.defn is sub_tile.defn} == {sub_tile.id}
    assert {u.template for u in us if u.role == "tile"} == {descriptor_id(sub_tile)}
    part = PO.evaluate(prog, q)
    assert [t for _k, t, _a, n_ in part.ranges for _ in range(n_)] == [u.template for u in us]
    assert PO.verify(PO.build(prog, q), prog).ok
    g, _owner, verdict = mod.unit_cut(fn)
    assert verdict.ok, verdict.codes
    assert verdict.detail["certified"] == verdict.detail["computed_gates"]
    whole = CallGraph(build(replace(cut, tile_sub=None)))
    computing = lambda gr: Counter(gr.prim_names[p] for p in gr.prim)  # noqa: E731
    assert computing(g) == computing(whole)
    extra = g.free_by_prim - whole.free_by_prim
    assert all("Const" in p for p in extra) and not whole.free_by_prim - g.free_by_prim, extra


def test_the_deployed_sub_tile_is_todays_tile_template_at_2x2():
    """`TILE_SUB` passes on the scheme's 64 × 64 tile, plain and hiding, and its sub-tile calls today's tile template at
    2 × 2, the one #983's tie pins at 1 × 1 under the other value."""
    assert tilesub.TILE_SUB == (2, 2) and set(tilesub.TILE_SUB) <= set(tilesub.SUBS)
    for hidden in (False, True):
        for sub in (tilesub.TILE_SUB, (1, 1)):
            pc8.Shape(64, 4096, 64, hidden=hidden, tile_sub=sub).check()
            pc4.Shape(64, 4096, 64, hidden=hidden, tile_sub=sub).check()
    shape = pc8.Shape(2, 128, 2, tile=(2, 2), dev="h100", tile_sub=(1, 1), hidden=True)
    assert tilesub.unit_templates8(shape, "call")[1:] == [tilesub.sub_tile8(128, (1, 1), "h100"), L.row_chunks(2, 1),
                                                          L.tile_leaf(2, 2)]
    for fn, tile in ((tilesub.sub_tile8(4096, tilesub.TILE_SUB, "sm120"), pc8.templates(4096, (2, 2), "sm120")["tile"]),
                     (tilesub.sub_tile4(4096, tilesub.TILE_SUB), pc4.templates(4096, (2, 2))["tile"])):
        assert [nd.fn for nd in fn.body.nodes] == [tile]


@pytest.mark.parametrize("mod, args", [(pc8, (128, (2, 2), "h100")), (pc8, (64, (1, 4), "sm120-unpromoted")),
                                       (pc4, (64, (2, 2))), (pc4, (64, (1, 1)))], ids=["h100-2x2", "unpromoted-1x4", "pc4-2x2",
                                                                                        "pc4-1x1"])
def test_a_sub_tile_is_todays_tile_template_and_wiring(mod, args):
    """A sub-tile is one call of today's tile template at a × b and nothing else, with its parameters: its output r is row r
    of the tile's C̃ then row r of its U, a reference to each word and no gate, so it is 4b words, one sub-cu row a port (at
    a = 1 too, never a run of two ports); evaluated, it is today's tile's C̃ and U row by row."""
    fn = (tilesub.sub_tile8 if mod is pc8 else tilesub.sub_tile4)(*args)
    tile, (a, b) = mod.templates(*args)["tile"], args[1]
    (params, ret), (tile_params, _) = fn.signature(), tile.signature()
    assert [nd.fn for nd in fn.body.nodes] == [tile] and fn.gates == tile.gates and params == tile_params
    assert len(ret.fields) == a and all(sum(t.leaf_widths()) == 16 * L.ROW_WORDS["sub-cu"](b) for t in ret.fields)
    assert list(fn.body.ret) == [("n", 0, f * a * b + r * b + j) for r in range(a) for f in range(2) for j in range(b)]
    rng = random.Random(a * 10 + b)
    x = [[rng.getrandbits(w) for w in t.leaf_widths()] for _nm, t in params]
    got, want = evaluate(fn, *x), evaluate(tile, *x)
    c, u = want[:a * b], want[a * b:]
    assert list(got) == [v for r in range(a) for v in c[r * b:(r + 1) * b] + u[r * b:(r + 1) * b]]
    for r in range(a):
        assert _halves(got[2 * r * b:2 * (r + 1) * b]) == L.words_sub_cu(c[r * b:(r + 1) * b], u[r * b:(r + 1) * b])


# ---- the ports, the anchors and the layout --------------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
def test_the_ports_are_each_pieces_committed_rows(shape):
    """Each tile piece's ports (`tilesub.ports`) list every parameter and output once: a sub-tile its a rows of A's and b
    rows of B's data rows and its a sub-cu rows, output r row r's alone, a row digest its TN / b sub-cu rows and its
    tile-chunks row, the leaf its TM tile-chunks rows and today's tile row; the rows' are today's or the row cut's."""
    cut = replace(shape, hidden=True)
    ports = cut.ports
    mod = _mod(shape)
    rows = (rowk.PORTS8 if mod is pc8 else rowk.PORTS4) if shape.row_k else mod.PORTS
    assert set(ports) == set(rows) | set(tilesub.ROLES)
    assert all(ports[r] == rows[r] for r in rows if r != "tile")
    (tm, tn), (sa, sb) = shape.tile, shape.tile_sub
    a_ports, b_ports = (("ac", "pa"), ("bc", "pb")) if mod is pc8 else (("ac", "as", "pa"), ("bc", "bs", "pb"))
    assert [r for r in ports["tile"]["rows"] if r[0] == "sub-cu"] == [("sub-cu", ROLE_X, (r,)) for r in range(sa)]
    CF.check_ports(_sub_tile(shape), ports["tile"], cut.derivations, shape.k,
                   dict.fromkeys(a_ports, sa) | dict.fromkeys(b_ports, sb), {"sub-cu": sb})
    CF.check_ports(L.row_chunks(tn, sb), ports["row_digest"], cut.derivations, shape.k, {"c": tn // sb, "u": tn // sb},
                   {"sub-cu": sb})
    CF.check_ports(L.tile_leaf(tm, tn), ports["tile_leaf"], cut.derivations, shape.k, {"d": tm})


@pytest.mark.parametrize("shape", [SHAPES[1], SHAPES[4], SHAPES[7]], ids=[IDS[1], IDS[4], IDS[7]])
def test_no_tile_piece_reads_a_public_input_region(shape):
    """The tile pieces read committed rows only, and the rows' public input ports and anchors' rules are the shape's
    without the tile cut."""
    for hidden in (False, True):
        cut = replace(shape, hidden=hidden)
        today = replace(cut, tile_sub=None)
        for role in tilesub.ROLES:
            assert cut.ports[role]["regions"] == {} and public_ports(cut, role) == {}
        for role in today.ports:
            if role != "tile":
                assert public_ports(cut, role) == public_ports(today, role)
        assert cut.derivations is today.derivations and cut.seeds is today.seeds


def test_the_layout_proves_each_drawn_sub_tile_with_its_whole_tile():
    """`plan.tile_layout` with the cut: the tile stratum is the sub-tiles at the W_ref given, and a sub-tile's closure is
    itself, its tile's other sub-tiles, under hiding the tile's row digests and leaf, then its tile's rows of A; its weight
    rows are its tile's rows of B; a draw proves them, the row units each once, and opens nothing."""
    for hidden in (False, True):
        shape = pc4.Shape(4, 64, 8, tile=(2, 4), tile_sub=(2, 2), hidden=hidden)
        lay = plan.tile_layout(shape, 1000)
        lay.check()
        us, ws = lay.units, pc4.units(pc4.weight(shape))
        subs = [u for u in us if u.role == "tile"]
        assert len(subs) == 8 and set(lay.closure) == set(lay.b_rows) == {t.index for t in subs}
        for t in subs:
            rs, cs = t.where[0], t.where[1] // 2
            own = [(r, w) for r, w in [(us[i].role, us[i].where) for i in lay.closure[t.index]]]
            want = [("tile", (rs, 2 * cs + j)) for j in range(2)]
            if hidden:
                want += [("row_digest", (2 * rs + i, cs)) for i in range(2)] + [("tile_leaf", (rs, cs))]
            want += [("row_a", (2 * rs + i,)) for i in range(2)]
            assert own[0] == ("tile", t.where) and sorted(own) == sorted(want)
            assert [(ws[j].role, ws[j].where) for j in lay.b_rows[t.index]] == [("row_b", (4 * cs + j,)) for j in range(4)]
        strata = [(s.role, s.work) for s in lay.strata]
        assert strata == [("row_a", 0), ("tile", 8 * 1000)] + ([("row_digest", 0), ("tile_leaf", 0)] if hidden else [])
        for law in (plan.CountLaw(3), plan.WorkLaw(2)):
            p = plan.draw(derive(bytes(32), plan.DRAW_DOMAIN, {"test": "tilesub"}), lay, law)
            drawn = [u for u in p.drawn if us[u].role == "tile"]
            assert drawn and p.openings == () and p.proved_y == ()
            assert set(p.proved) == set(p.drawn) | {i for t in drawn for i in lay.closure[t]}
            assert p.proved_b == tuple(sorted({j for t in drawn for j in lay.b_rows[t]}))


# ---- the domain ---------------------------------------------------------------------------------------------------------

def test_shapes_off_the_cut_are_refused():
    """tile_sub is a pair of `SUBS` members (ints, not bools) dividing the tile, and a hiding cut's tile rows are at most
    four chunks (TN <= 128), where one tile-chunks row holds their digests."""
    for shape in (pc8.Shape(2, 128, 4, tile=(2, 4), tile_sub=(3, 1)), pc8.Shape(2, 128, 4, tile=(1, 4), tile_sub=(2, 2)),
                  pc8.Shape(2, 128, 4, tile=(2, 4), tile_sub=(2, 8)), pc8.Shape(2, 128, 4, tile=(2, 4), tile_sub=(True, 1)),
                  pc8.Shape(2, 128, 4, tile=(2, 4), tile_sub=[2, 2]), pc8.Shape(2, 128, 4, tile=(2, 4), tile_sub=(2, 2, 2)),
                  pc8.Shape(2, 128, 4, tile=(2, 4), tile_sub=(0, 1)), pc8.Shape(2, 128, 4, tile=(2, 4), tile_sub=(2.0, 2)),
                  pc4.Shape(64, 64, 64, tile_sub=(32, 32)), pc4.Shape(4, 64, 8, tile=(2, 4), tile_sub=(1, 3)),
                  pc4.Shape(1, 64, 192, tile=(1, 192), tile_sub=(1, 16), hidden=True)):
        for build in (_mod(shape).check, _mod(shape).call):
            with pytest.raises(DomainError):
                build(shape)
    pc4.Shape(1, 64, 192, tile=(1, 192), tile_sub=(1, 16)).check()
    pc4.Shape(1, 64, 128, tile=(1, 128), tile_sub=(1, 16), hidden=True).check()


def test_off_the_cut_nothing_is_the_cuts():
    """With ``tile_sub`` unset the roots, queries, units, ports and anchors are today's (or the row cut's): no root, template
    or role of the tile cut's."""
    cut_ids = {id(d) for d in tilesub.ROOTS} | {id(d) for d in (tilesub.Pc8SubTileDef, tilesub.Pc4SubTileDef,
                                                                 L.PearlRowChunksDef, L.PearlTileLeafDef)}
    assert len(cut_ids) == 16 + 4
    for mod, shape in ((pc8, pc8.Shape(4, 128, 8, tile=(2, 4))), (pc4, pc4.Shape(4, 64, 8, tile=(2, 4))),
                       (pc8, pc8.Shape(2, 256, 2, tile=(1, 2), row_k=128)), (pc4, pc4.Shape(2, 128, 2, tile=(1, 2), row_k=64))):
        for shape_ in (shape, replace(shape, hidden=True)):
            for side in ("weight", "call", "check"):
                fn = getattr(mod, side)(shape_)
                assert id(fn.defn) not in cut_ids and all(id(nd.fn.defn) not in cut_ids for nd in fn.body.nodes)
                assert not {u.role for u in mod.units(fn)} & {"row_digest", "tile_leaf"}
            assert shape_.ports is ((rowk.PORTS8 if mod is pc8 else rowk.PORTS4) if shape.row_k else mod.PORTS)
    for mod, shape in ((pc8, pc8.Shape(4, 128, 8, tile=(2, 4))), (pc4, pc4.Shape(4, 64, 8, tile=(2, 4)))):
        args = (shape.dev,) if mod is pc8 else ()
        assert json.dumps(mod.query(shape, "check")) == json.dumps(
            PO.template_instances_query([mod.templates(shape.k, shape.tile, *args)[r] for r in mod.ROLES["check"]]))
        assert mod.Shape.ports is mod.PORTS and mod.Shape.derivations is mod.DERIVATIONS
        cut = replace(shape, tile_sub=(1, 1))
        assert cut.ports is tilesub.ports(cut) and cut.derivations is mod.DERIVATIONS
        both = replace(cut, row_k=64 if mod is pc4 else 128, tile_sub=(2, 2))
        assert both.ports is tilesub.ports(both) and both.ports["row_a_head"] is (rowk.PORTS8 if mod is pc8 else
                                                                                  rowk.PORTS4)["row_a_head"]
