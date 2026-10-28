"""`pouw_native`: `ncp-v2`'s committed values computed natively are the Programs', unit by unit: each call's and its
weight side's."""
from __future__ import annotations

import numpy as np
import pytest
from verity.ml.protocols.pouw.circuit import partition as CP
from verity.ml.protocols.pouw.circuit import reference as CR

from verity.ml.circuits.evaluation import evaluate
from verity_vllm.protocols.pouw import circuit as PC
from verity_vllm.protocols.pouw import native as PN


class Counted:
    """Quantized rows that record every read."""

    def __init__(self, b, t):
        self.b, self.t, self.n, self.k, self.reads = b, t, b.shape[0], b.shape[1], []

    def rows(self, lo, hi):
        self.reads.append((lo, hi))
        return self.b[lo:hi], self.t[lo:hi]


def _call(rng, m, k, n):
    b = rng.integers(-63, 64, (n, k)).astype(np.int8)
    t = (np.abs(rng.standard_normal(n)).astype(np.float32) * 0.01 + 1e-3).view(np.uint32)
    return PC.Call(rng.integers(-63, 64, (m, k)).astype(np.int8),
                   (np.abs(rng.standard_normal(m)).astype(np.float32) * 0.01 + 1e-3).view(np.uint32), Counted(b, t),
                   (rng.standard_normal(n).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16), bytes(range(32)),
                   forward=3, layer=2, site=0, weight=2, rowids=np.arange(100, 100 + m, dtype=np.uint32),
                   colids=np.arange(n, dtype=np.uint32))


@pytest.fixture(scope="module")
def three_strips():
    """A call whose tree has two nodes (three strips), evaluated both ways."""
    call = _call(np.random.default_rng(0), 48, 64, 32)
    return call, PN.Native(call, "ncp-v2"), PC.evaluate_ir("ncp-v2", call)


@pytest.mark.slow
def test_every_unit_is_the_programs_outputs_numbering_and_wiring(three_strips):
    call, nat, (c, tr) = three_strips
    tp, units = c.partition, CP.units(call.shape, PC.kind_of("ncp-v2"))
    assert nat.population == tp.population == len(units)
    for u in range(tp.population):
        role, where = nat.locate(u)
        assert units[u].role == role and nat.index(role, *where) == u
        assert role in ("node", "key") or tuple(units[u].where) == where
        assert [int(v) for v in nat.outputs(u)] == [tr[g] for g in tp.outputs(u)], (u, role, where)
        flat = [v for arg in nat.args(u, nat.outputs) for v in (arg if isinstance(arg, list) else [arg])]
        assert flat == [tr[g] for g in tp.inputs(u)], (u, role, where)


@pytest.mark.slow
def test_every_y_unit_is_the_weight_programs_outputs_numbering_and_wiring(three_strips):
    call = three_strips[0]
    c, tr = PC.evaluate_weight_ir("ncp-v2", call)
    nw, tp = PN.NativeWeight(call.b, "ncp-v2", call.salt, call.weight, call.colids), c.weight_partition
    assert nw.population == tp.population == call.shape.n // 16
    for u in range(tp.population):
        assert [int(v) for v in nw.outputs(u)] == [tr[g] for g in tp.outputs(u)], u
        assert [v for arg in nw.args(u) for v in arg] == [tr[g] for g in tp.inputs(u)], u
    assert list(evaluate(nw.template(1), *nw.args(1))) == [int(v) for v in nw.outputs(1)]


@pytest.mark.slow
def test_the_commit_is_the_ir_commit_and_the_reference(three_strips):
    call, _nat, _ir = three_strips
    nat, ir = PN.commit_call(0, "ncp-v2", call), PC.commit_call_ir(0, "ncp-v2", call)
    assert (nat.leaves, nat.inputs, nat.root) == (ir.leaves, ir.inputs, ir.root) and np.array_equal(nat.y, ir.y)
    assert (nat.side.leaves, nat.side.inputs, nat.side.root) == (ir.side.leaves, ir.side.inputs, ir.side.root)
    assert PC.call_leaf(nat) == PC.call_leaf(ir) and nat.side.reads == ir.side.reads
    assert nat.side.reads == PN.NativeWeight.of(nat.side).reads(block=16) and PC.inputs_digest(call, nat.side.reads) == nat.inputs
    assert all(np.array_equal(a, b) for cs in range(2) for a, b in zip(nat.read(cs), ir.read(cs)))
    assert [[int(v) for v in r] for r in nat.y] == CR.evaluate(call.reference(), PC.kind_of("ncp-v2")).y


def test_each_template_evaluates_from_its_regenerated_inputs_to_its_outputs(three_strips):
    _call_, nat, _ir = three_strips
    for u in (0, nat.index("node", 1), nat.index("key"), nat.index("form_x", 2), nat.index("tile", 2, 1),
              nat.index("dequant", 1, 0, 200)):
        assert list(evaluate(nat.template(u), *nat.args(u, nat.outputs))) == [int(v) for v in nat.outputs(u)], nat.locate(u)


def test_the_weight_is_read_once_in_blocks_and_the_block_does_not_change_the_commit():
    """The weight side reads the rows once, and each call on it once more (for t and its tiles), in blocks: each block once,
    in whatever order the pool's workers take them."""
    call = _call(np.random.default_rng(3), 16, 64, 80)
    nw = PN.NativeWeight(call.b, "ncp-v2", call.salt, call.weight, call.colids)
    side = nw.commit(block=32)
    assert sorted(call.b.reads) == [(0, 32), (32, 64), (64, 80)]
    assert (side.leaves, side.inputs, side.root) == (lambda o: (o.leaves, o.inputs, o.root))(nw.commit(block=256))
    call.b.reads.clear()
    a = PN.Native(call, "ncp-v2").commit(0, side, block=32)
    assert sorted(call.b.reads) == [(0, 32), (32, 64), (64, 80)]
    b = PN.Native(call, "ncp-v2").commit(0, side, block=256)
    assert (a.leaves, a.inputs, a.root) == (b.leaves, b.inputs, b.root) and np.array_equal(a.y, b.y)


@pytest.mark.parametrize("threads", [1, 3, 256])
def test_the_pools_threads_and_their_blocks_do_not_change_the_commit(threads, monkeypatch):
    """A pool of one thread per CPU cuts the weight into `block_for` blocks, at least `MIN_BLOCK` rows, so that every thread has
    one: 256 threads read a 400-row weight 64 rows at a time, one thread in `BLOCK`s; either commit is the other's, bit for bit."""
    call = _call(np.random.default_rng(5), 16, 64, 400)
    nw = PN.NativeWeight(call.b, "ncp-v2", call.salt, call.weight, call.colids)
    ref = PN.Native(call, "ncp-v2", side=nw).commit(0, nw.commit(block=PN.BLOCK), block=PN.BLOCK)
    monkeypatch.setattr(PN, "THREADS", threads)
    PN._pool.cache_clear()
    try:
        call = _call(np.random.default_rng(5), 16, 64, 400)
        got = PN.commit_call(0, "ncp-v2", call)
    finally:
        PN._pool.cache_clear()
    assert PN.block_for(400) == {1: 256, 3: 144, 256: 64}[threads]
    assert max(hi - lo for lo, hi in call.b.reads) == PN.block_for(400)
    assert (got.leaves, got.inputs, got.root) == (ref.leaves, ref.inputs, ref.root) and np.array_equal(got.y, ref.y)
    assert (got.side.leaves, got.side.root) == (ref.side.leaves, ref.side.root)


def test_a_tampered_value_changes_exactly_its_units_leaf_and_its_opening():
    call = _call(np.random.default_rng(4), 32, 64, 16)
    honest = PN.commit_call(0, "ncp-v2", call)
    nat = PN.Native(call, "ncp-v2")
    for u, j in ((nat.index("tile", 1, 0), 7), (nat.index("dequant", 0, 0, 3), 0)):
        rec = PN.commit_call(0, "ncp-v2", call, tamper={(u, j): int(honest.opening(u)[j]) ^ 1}, side=honest.side)
        diff = [v for v in range(nat.population) if rec.leaf(v) != honest.leaf(v)]
        assert diff == [u] and int(rec.opening(u)[j]) == int(honest.opening(u)[j]) ^ 1
        assert PC.unit_leaf(rec.opening(u)) == rec.leaf(u)
        served = rec.y.astype(np.int64) ^ honest.y.astype(np.int64)
        assert served.tolist() == ([[1 if (i, c) == (0, 3) else 0 for c in range(16)] for i in range(32)]
                                   if nat.locate(u)[0] == "dequant" else np.zeros((32, 16), int).tolist())
    side = PN.commit_weight("ncp-v2", call, tamper={(0, 3000): int(honest.side.opening(0)[3000]) ^ 1})
    assert [v for v in range(side.units) if side.leaf(v) != honest.side.leaf(v)] == [0]
    assert int(side.opening(0)[3000]) == int(honest.side.opening(0)[3000]) ^ 1 and PC.unit_leaf(side.opening(0)) == side.leaf(0)


def test_dequant_over_arrays_is_the_primitives_on_every_edge():
    """Zero and negative zero, subnormal and huge scales, Z at the int32 extremes and past 2^24, and a NaN product."""
    specials = np.array([0, 0x80000000, 1, 0x007FFFFF, 0x00800000, 0x3F800000, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000,
                         0x33D6BF95, 0x3C010204], np.uint32)
    zs = np.array([0, 1, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, (1 << 24) + 1, (1 << 25) + 3, 0xFFF00001], np.uint32)
    biases = np.array([0x8000, 0, 0x3F80, 0xBF80, 0x7F80, 0x0001, 0x7FC0], np.uint16)
    rng = np.random.default_rng(5)
    s = np.concatenate([specials, rng.integers(0, 1 << 32, 20, dtype=np.uint64).astype(np.uint32)])
    for z in zs:
        for b in biases:
            got = PN.dequant(z, s[:, None], s[None, :], b)
            want = [[CR.dequant(int(z), int(si), int(ti), int(b)) for ti in s] for si in s]
            assert got.tolist() == want, (hex(z), hex(b))
