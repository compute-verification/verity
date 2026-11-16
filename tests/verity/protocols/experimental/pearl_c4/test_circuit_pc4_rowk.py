"""Pearl-C4's row units cut along k (`pc4_rowk`, ``Shape.row_k``), on `test_circuit_rowk`'s helpers: on every row,
admitted or refused, a row's head, forms and tail in sequence give the whole row unit's outputs, rows of A and of B; the
cut roots give today's roots' outputs with the same computing gates, and their partition computes every value in exactly
one unit; each piece's ports are committed rows of `leaves`' schemas and public input regions, a form's its segment of F_A
and F_B under the slice rule, and the head commits the words its readers read; and with ``row_k`` unset nothing is the
cut's."""
from __future__ import annotations

from dataclasses import replace

import pytest

import pearl_c4_vectors as V4
import test_circuit_rowk as R
from verity.circuits.evaluation import evaluate
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import leaves as L
from verity.protocols.experimental.pearl_c4.prover import pc4, pc4_rowk
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import plan, rowk
from verity.protocols.experimental.pearl_c4.prover import pearl_c4 as F
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_kw as P
from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import DomainError

SEED = R.SEED
K4, KS4 = 256, 64
SHAPE = pc4.Shape(2, K4, 2, tile=(1, 1), row_k=KS4)
SLICE_RULES = {f"{a}-slice": f"verity/pouw/anchor/pearl-c4/{a}/slice/v2" for a in ("fa", "fb")}


def _rows4(k: int) -> dict[str, list[int]]:
    """`test_circuit_pc4`'s rows at depth k, their words placed in later segments."""
    base = [V4.row_codes("pc4-rowk", "A", i, k) for i in range(3)]
    at = lambda w, updates: [updates.get(t, v) for t, v in enumerate(w)]  # noqa: E731
    mid, last = (k // 2) // 8 * 8, k - 3
    spike = lambda v: at(base[0], {last: P.f32_to_bf16(v)})  # noqa: E731
    return {"honest-0": base[0], "honest-1": base[1], "negated": [w ^ 0x8000 for w in base[2]],
            "inf-unsampled": at(base[0], {last: 0x7F80}), "minus-nan-sampled": at(base[0], {mid: 0xFFC1}),
            "spike-2^16": spike(2.0 ** 16), "spike-256": spike(256.0), "max-unsampled": at(base[0], {mid + 1: 0x7F7F}),
            "zero": [0] * k, "tiny": [0x0001] * k, "huge": [0x7F7F] * k}


@pytest.mark.parametrize("side", ["a", "b"])
def test_pearl_c4_pieces_give_the_whole_rows_outputs(side):
    """Pearl-C4's head (its admission whole: `pc4._stats`), forms and tail of a row give `Pc4RowA`'s (`Pc4RowB`'s) codes,
    scale bytes, P and ok on every row: a segment is whole 64-position atoms, so each form's chains carried from the one
    before are the whole row's."""
    fa, fb = R._flat(F.basis(SEED, F.ROLE_FA, K4)), R._flat(F.basis(SEED, F.ROLE_FB, K4))
    own, other = (fa, fb) if side == "a" else (fb, fa)
    extra = [] if side == "a" else [R._flat(F.gram(F.NVFP4, SEED, K4))]
    whole = pc4.templates(K4)[f"row_{side}"]
    pieces = pc4_rowk.pieces4(K4, KS4, side)
    verdicts = {}
    for i, (name, x) in enumerate(_rows4(K4).items()):
        e = F.line(SEED, F.ROLE_EA if side == "a" else F.ROLE_EB, i)
        want = evaluate(whole, x, e, own, other, *extra)
        ok, got = R._in_sequence(pieces, KS4, rowk.HEAD4, 2, x, (), (e,), own, other, extra)
        assert got + ok == want, name
        verdicts[name] = want[-1]
    assert {n for n, v in verdicts.items() if v} == {"honest-0", "honest-1", "negated", "spike-256"}


def _inputs4(k: int, m: int, n: int):
    a = [w for i in range(m) for w in V4.row_codes("pc4-rowk", "A", i, k)]
    b = [w for j in range(n) for w in V4.row_codes("pc4-rowk", "B", j, k)]
    ea, eb = (R._flat(F.line(SEED, role, i) for i in range(c)) for role, c in ((F.ROLE_EA, m), (F.ROLE_EB, n)))
    return a, ea, b, eb, R._flat(F.basis(SEED, F.ROLE_FA, k)), R._flat(F.basis(SEED, F.ROLE_FB, k)), R._flat(F.gram(F.NVFP4, SEED, k))


@pytest.mark.parametrize("hidden", [False, True], ids=["plain", "hidden"])
def test_the_cut_roots_give_todays_outputs(hidden):
    """The cut's `weight`, `call` and `check` give today's roots' outputs on an honest block (the call reading the
    weight's committed rows: codes, scale bytes and P), plain and hiding."""
    R.roots_give_todays_outputs(pc4, replace(SHAPE, hidden=hidden), _inputs4(SHAPE.k, 2, 2), lambda w, n, k: (
        w[:n * k], w[n * k:n * k + n * k // 16], w[n * k + n * k // 16:-n]))


@pytest.mark.parametrize("hidden", [False, True], ids=["plain", "hidden"])
@pytest.mark.parametrize("side", ["check", "weight", "call"])
def test_the_cut_computes_every_value_in_exactly_one_unit(side, hidden):
    """`test_circuit_rowk`'s on Pearl-C4's cut, whose forms decode their segment's BF16 words again (`Bf16ToF32`,
    wiring)."""
    R.cut_computes_every_value_once(pc4, replace(SHAPE, hidden=hidden), side, {"Bf16ToF32_v1"})


def test_the_layout_proves_each_rows_pieces_with_its_tiles():
    """`plan.tile_layout` on a cut shape: a tile's closure is the tile and its rows of A's units (head, forms, tail, row by
    row), and its weight rows are its rows of B's in the weight's Program."""
    shape = pc4.Shape(4, K4, 6, tile=(2, 3), row_k=KS4)
    lay = plan.tile_layout(shape, 1)
    lay.check()
    us, ws = pc4.units(pc4.call(shape)), pc4.units(pc4.weight(shape))
    for t in (u for u in us if u.role == "tile"):
        rs, cs = t.where
        assert lay.closure[t.index][0] == t.index
        assert sorted(us[i].where[0] for i in lay.closure[t.index][1:]) == sorted(
            [rs * 2 + i for i in range(2)] * (K4 // KS4 + 2))
        assert all(us[i].role.startswith("row_a") for i in lay.closure[t.index][1:])
        assert sorted(ws[j].where[0] for j in lay.b_rows[t.index]) == sorted([cs * 3 + j for j in range(3)] * (K4 // KS4 + 2))


def _pieces_of(shape) -> dict[str, list]:
    return R.pieces_by_role(lambda s: pc4_rowk.pieces4(shape.k, shape.row_k, s))


def test_the_ports_are_each_pieces_committed_rows():
    """Each piece's ports (`pc4_rowk.PORTS4`) list every parameter and output once, as Pearl-C's do."""
    R.ports_are_committed_rows(pc4, SHAPE, _pieces_of(SHAPE), "nvfp4-codes")


@pytest.mark.parametrize("side", ["a", "b"])
def test_the_head_commits_the_row_its_readers_read(side):
    """On every row, admitted or refused, the head's pc4-head ports, each padded, are `leaves.words_pc4_head` of its
    values: the very words the first form, each form and the tail read from that row."""
    R.head_commits_its_readers_row(SHAPE, side, _pieces_of(SHAPE), rowk.HEAD4, "pc4-head", _rows4(SHAPE.k),
                                   lambda i: (), lambda h: L.words_pc4_head(h["alpha"], h["ae"]),
                                   lambda seen: len(seen) > 2)


def test_each_heads_ok_word_is_constant_in_gates():
    """The head's ok word, as Pearl-C's tail's: `SelectBf16` of a `BitAnd`."""
    R.ok_is_a_select_of_an_and(pc4_rowk.pieces4(128, 64, "a")[0], pc4_rowk.pieces4(128, 64, "b")[0])


def test_a_form_reads_its_segment_of_the_bases():
    """A form's F_A and F_B regions are its segment's lines of today's whole regions, under the slice rule; its E line
    and the tail's G are today's; a segment outside the row is refused."""
    R.form_reads_its_segment(SHAPE, *R._anchors(SHAPE, F.PearlC4("nvfp4", hashing="h0")))


def test_slice_rules_exist_only_under_the_cut():
    """The cut's shape names today's five rules and the two slice rules; the class and a shape without ``row_k`` name
    today's five only."""
    R.slice_rules_only_under_the_cut(pc4, SHAPE, SLICE_RULES)


def test_shapes_off_the_cut_are_refused():
    """row_k is a positive multiple of 64 that divides k, and `ROW_K` is one."""
    assert rowk.ROW_K % pc4.ATOM == 0
    pc4.Shape(1, 4096, 1, tile=(1, 1), row_k=rowk.ROW_K).check()
    for shape in (pc4.Shape(1, 256, 1, tile=(1, 1), row_k=32), pc4.Shape(1, 192, 1, tile=(1, 1), row_k=128),
                  pc4.Shape(1, 256, 1, tile=(1, 1), row_k=True)):
        for build in (pc4.check, pc4.weight, pc4.call):
            with pytest.raises(DomainError):
                build(shape)


def test_off_the_cut_nothing_is_the_cuts():
    """With ``row_k`` unset the roots, queries and units are today's: no root, template or role of the cut's."""
    R.nothing_is_the_cuts(pc4_rowk, 8 + 5, pc4, pc4.Shape(4, 64, 6, tile=(2, 3)), ())
