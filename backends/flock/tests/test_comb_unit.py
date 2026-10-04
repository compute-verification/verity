"""A real C-Flock unit circuit (``rope-head/neox-bf16/pair``, 6,145 rows in a 2^13 slot) through com_P and the comb gadget:
its share of the lincheck fold, from com_P's rows, is the Lean verifier's (`fixtures/comb_rope_share.json`, its
``project (baseOf …)`` at two reps' coins), as the reference and as the gadget's circuit; a changed nonzero changes the share,
and the changed row no longer opens com_P."""
from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path

import numpy as np

from verity.ml.boolean import comb as K
from verity.ml.boolean.forms import Circuit
from verity_flock import com_p
from verity_flock import ir_lower as IL
from verity_numerical.bench import lowerings
from verity_numerical.bench import templates as TM

FIX = json.loads((Path(__file__).parent / "fixtures" / "comb_rope_share.json").read_text())
#: the class's shape for this unit: 2,420 entries a bucket (its fullest holds 2,409), 13 blocks a row (605 entries)
SHAPE = com_p.shape(13, 2420, 6144, 13)


@cache
def _unit():
    lay = lowerings.registry("C-Flock").module("rope-head").lowering(TM.subcircuit("rope-head", D=64)).layout
    text = IL.netlist(lay, "rope-head/neox-bf16/pair")
    return text, [list(map(int, r)) for r in lay.rows_a], [list(map(int, r)) for r in lay.rows_b], lay.const


@cache
def _rows():
    _, ra, rb, _ = _unit()
    return com_p.rows(SHAPE, ra, rb)


def _points() -> list[K.Point]:
    h = lambda s: int(s, 16)  # noqa: E731
    return [K.Point(h(r["alpha"]), [h(x) for x in r["lw"]], [h(x) for x in r["rho"]], [h(x) for x in r["rr"]]) for r in FIX["reps"]]


def _lean_shares() -> list[list[int]]:
    return [[int(x, 16) for x in r["share"]] for r in FIX["reps"]]


def _share(p: K.Point, rows) -> list[int]:
    buckets, a, b = K.unpack(SHAPE, rows)
    return K.template_fold(p, [K.values([p], es)[0] for es in buckets], K.column([p], SHAPE.COL, a, b)[0], SHAPE.COL)


def test_the_fixture_is_this_unit():
    text, _, _, const = _unit()
    assert hashlib.sha256(text.encode()).hexdigest() == FIX["netlist_sha256"]
    assert (const, SHAPE.R, SHAPE.E, SHAPE.rows) == (6144, 4, 605, 258)


def test_com_p_holds_the_unit_circuit():
    """com_P's rows give back the unit's matrices (each cell an odd number of times), and the statement's entry list."""
    _, ra, rb, _ = _unit()
    odd = lambda rows: [sorted(c for c in set(r) if r.count(c) % 2) for r in rows]  # noqa: E731
    n = 1 << SHAPE.SL
    got_a, got_b = com_p.matrices(SHAPE, _rows())
    assert got_a == odd(ra) + [[]] * (n - len(ra)) and got_b == odd(rb) + [[]] * (n - len(rb))
    ents = com_p.hidden_entries(SHAPE, _rows())
    real = [t for t in ents if t[3]]
    assert len(ents) == SHAPE.nnz and len(real) == sum(map(len, got_a)) + sum(map(len, got_b))
    assert real == sorted(real, key=lambda t: (t[0], t[2], t[1])) and ents[len(real):] == [(0, 0, 0, 0)] * (len(ents) - len(real))


def test_rope_share_reference_is_the_lean_verifiers():
    for p, want in zip(_points(), _lean_shares()):
        assert _share(p, _rows()) == want


def test_rope_share_circuit_is_the_lean_verifiers():
    """The gadget's entry circuit, bit-sliced over all of com_P's 154,880 entry slots (padding included), summed per bucket,
    plus the column's circuit: the Lean verifier's share at both reps."""
    ps = _points()
    buckets, a, b = K.unpack(SHAPE, _rows())
    ents = np.array([e for es in buckets for e in es], dtype=np.uint64)
    T, W = len(ents), SHAPE.W
    C = Circuit()
    C.output("o", [x for X in K.build_entry(C, SHAPE.SL, ps, C.input("e", W)) for x in X])
    col = lambda j: int.from_bytes(np.packbits(((ents >> np.uint64(j)) & np.uint64(1)).astype(np.uint8), bitorder="little").tobytes(), "little")  # noqa: E731
    out, _ = C.evaluate({"e": [col(j) for j in range(W)]}, T)
    span = SHAPE.bound
    sums = [[0] * 64 for _ in ps]
    for k, x in enumerate(out["o"]):
        r, bit = divmod(k, 128)
        for s in range(64):
            sums[r][s] |= (((x >> (s * span)) & ((1 << span) - 1)).bit_count() & 1) << bit
    CC = K.column_circuit(SHAPE.SL, SHAPE.COL, ps)
    bits = a + b
    o, _ = CC.evaluate({f"in{j}": bits[16 * j:16 * j + 16] for j in range(len(bits) // 16)}, 1)
    cols = [K.f128_of([sum(o[f"out{8 * r + w}"][j] << j for j in range(16)) for w in range(8)]) for r in range(len(ps))]
    for r, (p, want) in enumerate(zip(ps, _lean_shares())):
        assert K.template_fold(p, sums[r], cols[r], SHAPE.COL) == want


def test_a_changed_nonzero_changes_the_share_or_breaks_the_binding():
    """Flip one bit of a real entry in com_P's row: the share is the changed matrices' (no longer the unit's), and the changed
    row's leaf does not open the registered root at its position, while the registered row does."""
    from verity_sampled_proofs.one_stage.registered import Registered, verify_read

    rows = _rows()
    salts = [hashlib.shake_256(b"comb-test-salt" + k.to_bytes(4, "little")).digest(192) for k in range(len(rows))]
    reg = Registered.commit("rec/class/rope-head-d64/test", com_p.VALUE, rows, salts)
    k = 5 * SHAPE.R  # bucket 5's first row
    bad = [list(r) for r in rows]
    bad[k][0] ^= 1 << 2  # entry 0's row bit 0
    p, want = _points()[0], _lean_shares()[0]
    changed = _share(p, bad)
    assert changed != want
    ba, bb = com_p.matrices(SHAPE, bad)
    assert changed == K.fold_dense(p.alpha, 0, p.lw, p.rho, list(reversed(p.rr)), ba, bb, 0)

    bc = Registered.commit("rec/class/rope-head-d64/test", com_p.VALUE, [bad[k]], [salts[k]]).bcs[0]
    assert verify_read(reg.domain, reg.root, k, reg.bcs[k], reg.path(k))
    assert not verify_read(reg.domain, reg.root, k, bc, reg.path(k))
