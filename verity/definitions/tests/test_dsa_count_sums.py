"""`boolean_dsa_topk_mask.DsaCountSum_v1`'s builder commits the sum it returns (`_committed`), which changes no Definition and
adds rows to a unit lowered as a served unit's stager lowers it (`verity_flock.class_statement.unit_lowering`).  Each bit of
a sum that is a form is a column held to that form by a row of its own (A the form, B the constant one), and that row alone
fails when a prover sets the bit wrong and follows it through; the commit rows are the sums' and no other's; a unit computes
what it computes with no sum committed; and its nonzeros grow with its keys, not faster."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.commitments.windows import interface as IF
from verity.definitions.registry import boolean_dsa_topk_mask as BM
from verity.definitions.registry import dsa_topk as T
from verity.definitions.registry import dsa_topk_mask_cut as X
from verity.definitions.registry import dsa_topk_reads as R
from verity.definitions.registry.boolean_dsa import own_gates

N, AT, TOPK = 64, 16, 16
UNITS = {"positions": bind(R.DsaMaskPositionsV2, N=N, AT=AT, K=TOPK), "block": bind(X.DsaTopKMaskBlockV2, N=N, AT=AT, TOPK=TOPK),
         "whole": bind(X.DsaTopKMaskWholeV2, N=N, TOPK=TOPK)}


def _lowering(fn):
    """``fn`` lowered on its rows, a row a parameter, as `UnitStager` lowers a served unit."""
    from verity_flock import class_statement as CS

    return CS.unit_lowering(fn, "unit/count-sums", [(p, ps) for p, _schema, _bits, ps in IF.unit_rows(fn)[0]])[0]


def _commit_rows(lay) -> list[int]:
    """The layout's commit rows (`forms.Circuit.commit`: A the form, B the constant one alone) among its ANDs' rows; the
    units here assert nothing, so no other row there has that B."""
    from verity_flock import ir_lower as IL

    assert lay.counts["asserts"] == 0
    return [i for i in range(lay.in_words * IL.WORD, lay.out_col * IL.WORD) if list(IL._cols_of(lay.rows_b[i])) == [lay.const]]


def _unsatisfied(lay, words, wrong: Sequence[int] = ()) -> dict[int, int]:
    """The rows ``(A_i . z)(B_i . z) = z_i`` that witnesses z fail on one instance, its input ports' ``words``, each with the
    witnesses that fail it (bit t of an int the t-th), as `ir_lower.evaluate_layout` computes z and checks it.  Witness t is
    that of a prover who sets column ``wrong[t]`` to the other bit and computes every later column from it; with no
    ``wrong``, the one honest witness."""
    from verity_flock import ir_lower as IL

    ones = (1 << max(1, len(wrong))) - 1
    lie = {row: 1 << t for t, row in enumerate(wrong)}
    z = [0] * lay.useful
    for at, width, w in zip(lay.in_port_cols, lay.in_bits, words):
        z[at:at + width] = [ones * (int(w) >> b & 1) for b in range(width)]
    z[lay.const] = ones

    def product(i: int) -> int:
        a = b = 0
        for k in IL._cols_of(lay.rows_a[i]):
            a ^= z[k]
        for k in IL._cols_of(lay.rows_b[i]):
            b ^= z[k]
        return a & b

    for i in range(lay.in_words * IL.WORD, lay.useful):
        if i != lay.const:
            z[i] = product(i) ^ lie.get(i, 0)
    return {i: failed for i in range(lay.useful) if (failed := product(i) ^ z[i])}


def _instances(kind: str, rows: int = 16) -> np.ndarray:
    """``rows`` instances of a unit of `UNITS`, a u16 word a column in its rows' order.  A mask unit's scores are of three
    values, so keys tie at the threshold, its n inside the block, at its edges and past it, a block's threshold one of the
    three keys and its r from 0 past TOPK; a positions unit's masks are sparse, dense, empty and full."""
    rng = np.random.default_rng(265)
    if kind == "positions":
        bits = rng.random((rows, 16 * -(-N // 16))) < np.array([0.1, 0.5, 0.9, 0.2])[np.arange(rows) % 4, None]
        bits[0], bits[1] = False, True
        return (bits.reshape(rows, -1, 16) << np.arange(16)).sum(axis=2).astype(np.uint64)
    lo = AT if kind == "block" else 0
    pool = rng.integers(0, 1 << 32, size=3, dtype=np.uint64)
    scores = rng.choice(pool, size=(rows, N))
    n = np.array([(0, lo, lo + 1, lo + TOPK, lo + N, lo + N + 3)[t] if t < 6 else lo + rng.integers(N + 1) for t in range(rows)],
                 dtype=np.uint64)
    cols = [scores & 0xFFFF, scores >> 16, n & 0xFFFF, n >> 16]
    x = np.concatenate([np.stack(cols[:2], axis=2).reshape(rows, -1), np.stack(cols[2:], axis=1)], axis=1)
    if kind == "whole":
        return x
    theta = np.array([T.topk_key(int(s)) for s in rng.choice(pool, size=rows)], dtype=np.uint64)
    r = rng.integers(0, TOPK + 2, size=rows, dtype=np.uint64)
    return np.concatenate([x, np.stack([theta & 0xFFFF, theta >> 16, r], axis=1)], axis=1)


def test_a_sums_rows_at_two_bits() -> None:
    """`DsaCountSum_v1{W=2}` as its builder lowers it: its two ANDs, c_1 = a_0 b_0 and t = (a_1 + c_1)(b_1 + c_1), then a
    commit row for each bit of the sum, A its form and B the constant one: a_0 + b_0, a_1 + b_1 + c_1 and the carry out
    c_1 + t.  The word it returns is those three committed bits."""
    from verity_flock import gf2
    from verity_flock import ir_lower as IL

    C = gf2.Circuit()
    leaves = [C.input(f"in{j}", 1) for j in range(4)]
    out = [w for (w,) in IL.lower(C, bind(BM.DsaCountSum, W=2), leaves)]
    a, b = [leaves[0][0], leaves[1][0]], [leaves[2][0], leaves[3][0]]
    c1, t, *sums = (gf2.L(frozenset((v,))) for v in range(4, len(C.kind)))
    assert [C.kind[v] for v in range(4, len(C.kind))] == ["and"] * 5 and [w.key() for w in out] == [s.key() for s in sums]
    rows = [(x.key(), y.key()) for x, y in C.data[4:]]
    assert rows[:2] == [(a[0].key(), b[0].key()), ((a[1] ^ c1).key(), (b[1] ^ c1).key())]
    assert rows[2:] == [(f.key(), gf2.ONE.key()) for f in (a[0] ^ b[0], a[1] ^ b[1] ^ c1, c1 ^ t)]


@pytest.mark.parametrize("kind", UNITS)
def test_a_sum_bit_set_wrong_fails_its_own_row(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """On a satisfied instance of each unit, a prover who sets one committed sum bit to the other value and computes every
    later column from it fails exactly one row, that bit's own: the sum is held to its operands and is no advice.  So for
    every sum bit of the unit, a witness each.  Flock's evaluator (`Lowering.evaluate`'s ``flip``, the bit alone changed)
    refuses the first, the middle and the last of them too.  The commit rows are the count sums': with the builder main
    had, the unit has none."""
    low = _lowering(UNITS[kind])
    lay, x = low.layout, _instances(kind)[7:8]
    commits = _commit_rows(lay)
    assert commits and _unsatisfied(lay, x[0]) == {} and int(low.evaluate(x)[2]) == 1
    assert _unsatisfied(lay, x[0], wrong=commits) == {row: 1 << t for t, row in enumerate(commits)}
    for row in (commits[0], commits[len(commits) // 2], commits[-1]):
        assert int(low.evaluate(x, flip=(row, 0))[2]) == 0
    monkeypatch.setattr(BM.DsaCountSum, "build", own_gates(BM.DsaCountSum))
    assert _commit_rows(_lowering(UNITS[kind]).layout) == []


@pytest.mark.parametrize("kind", UNITS)
def test_a_unit_computes_what_it_did_with_no_sum_committed(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each unit's laid-out rows, run on instances with more keys than TOPK, are all satisfied and give the outputs of the
    unit lowered with no sum committed (the builder main had): the same function, with the sums' rows more and the same
    ANDs otherwise.  The outputs say something: a whole mask has min(n, TOPK) members, a block's masks and the positions
    differ from instance to instance."""
    x = _instances(kind)
    full = (1 << len(x)) - 1
    low = _lowering(UNITS[kind])
    got, _cut, ok = low.evaluate(x)
    monkeypatch.setattr(BM.DsaCountSum, "build", own_gates(BM.DsaCountSum))
    plain = _lowering(UNITS[kind])
    want, _cut, ok_plain = plain.evaluate(x)
    assert int(ok) == int(ok_plain) == full and np.array_equal(got, want)
    commits = len(_commit_rows(low.layout))
    assert low.layout.counts["ands"] == plain.layout.counts["ands"] + commits
    assert len({r.tobytes() for r in got}) > len(x) // 2
    if kind == "whole":
        n = x[:, 2 * N] | x[:, 2 * N + 1] << np.uint64(16)
        members = [sum(bin(int(w)).count("1") for w in r) for r in got]
        assert members == np.minimum(n, TOPK).tolist()


def test_a_units_nonzeros_grow_with_its_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The positions of 256 keys stage in under 6 times the nonzeros of 64 keys' (4.8 times): a row that reads a count reads
    the sum's W + 1 columns.  With no sum committed it is over 6 times (11.4): a sum is then an XOR over the ANDs of the
    subtree under it, and each row that reads it carries all of them."""
    def nonzeros(n: int) -> int:
        c = _lowering(bind(R.DsaMaskPositionsV2, N=n, AT=0, K=TOPK)).layout.counts
        return c["nnz_a"] + c["nnz_b"]

    assert nonzeros(256) < 6 * nonzeros(64)
    monkeypatch.setattr(BM.DsaCountSum, "build", own_gates(BM.DsaCountSum))
    assert nonzeros(256) > 6 * nonzeros(64)
