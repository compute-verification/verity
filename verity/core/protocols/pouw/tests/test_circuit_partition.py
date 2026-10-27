"""`ncp-v2`'s partition: the template query, the invariant with no recompute, the Z separator and the width rule with the
separator port pinned by the protocol."""
from __future__ import annotations

from collections import Counter

import pytest

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.cut import CallGraph, check_cut
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, array_of, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple

from verity.core.protocols.pouw.circuit import ncp2
from verity.core.protocols.pouw.circuit import partition as P

SHAPE = ncp2.Shape(32, 64, 32)


def test_the_template_query_applies_and_numbers_units_as_core_does():
    prog, q = P.program(SHAPE), P.query(SHAPE)
    part = PO.evaluate(prog, q)
    us = P.units(SHAPE)
    assert part.population == len(us)
    assert [t for _k, t, _a, n in part.ranges for _ in range(n)] == [u.template for u in us]
    assert Counter(u.role for u in us) == {"digest": 2, "node": 1, "key": 1, "form_x": 2, "tile": 4, "dequant": 1024}
    assert PO.verify(PO.build(prog, q), prog).ok


def test_the_weight_program_is_its_y_units():
    """A weight's Y strips and c₀ are formed once per run and weight, in their own Program."""
    prog, q = P.weight_program(SHAPE), P.weight_query(SHAPE)
    assert PO.evaluate(prog, q).population == SHAPE.n // 16
    assert PO.verify(PO.build(prog, q), prog).ok


def test_a_one_strip_call_has_no_node_units():
    shape = ncp2.Shape(16, 64, 32)
    assert all(u.role != "node" for u in P.units(shape))
    assert PO.verify(PO.build(P.program(shape), P.query(shape)), P.program(shape)).ok


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["shift24", "word"])
def test_the_unit_cut_computes_every_value_in_exactly_one_unit(kind):
    """`validate_unit_cut` over the whole call, for both P's: every gate certified once, reads across units
    through committed values, and no value computed in two units (`gate-recomputed`)."""
    uc = P.unit_cut(SHAPE, kind)
    assert uc.verdict.ok, uc.verdict.codes
    assert uc.verdict.detail["certified"] == uc.verdict.detail["computed_gates"]


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["shift24", "word"])
def test_the_z_separator_width(kind):
    """Every path from a PoUW unit to y crosses a tile's Z: m·n values of 32 bits, 8,192 bits per tile."""
    sep = P.separator(SHAPE, kind)
    assert sep.ok and sep.reached_pouw == 0 and sep.words_delivered == 0
    assert sep.z_gates == SHAPE.m * SHAPE.n
    assert sep.width_bits == 32 * SHAPE.m * SHAPE.n and sep.per_tile_bits == 8192


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["shift24", "word"])
def test_the_width_rule_with_the_pinned_port(kind):
    """With the tiles' Z pinned as the separator, only the dequantization units reach y, and each fits the width rule."""
    wr = P.width_rule(SHAPE, kind)
    assert wr.ok and not wr.too_wide
    us = P.units(SHAPE, kind)
    assert {us[u].role for u in wr.checked} == {"dequant"}
    assert wr.exempt == 10 and wr.separator_bits == 32 * SHAPE.m * SHAPE.n
    bits = P.boundary_bits(SHAPE, kind)
    assert bits["dequant"] == {16: 1024} and bits["tile"] == {(11 * 256 + 256) * 32: 4}


def _served_word_body(B, S, a, s, ys, c0s, t, bias, salt, fwd, pos, wid, rid):
    """`ncp2.linear`'s body with one change: dequantization reads each tile's last credited word instead of its Z, the
    wiring a pinned port must reject (a word leaf is not a port)."""
    m, k, n = S.M, S.K, S.N
    ms, ns = m // 16, n // 16
    tm = ncp2.templates(k, S.PERM)
    strips = a.reshape(Array(ms, Array(16, Array(k, ncp2.W.I7))))
    dig = B.batch(tm["digest"], strips, axes=(0,))
    root = ncp2.rfc6962(0, ms, lambda i: dig[i], lambda left, right: B.call(ncp2.NcpNode, left, right))
    key = B.call(ncp2.NcpKey, salt, fwd, pos, wid, root)
    xs = B.batch(tm["form_x"], strips, key, rid.reshape(Array(ms, Array(16, ncp2.U32))), axes=(0, None, 0))
    tiles = [B.batch(tm["tile"], xs[rs], ys, c0s, axes=(None, 0, 0)) for rs in range(ms)]
    per = (3 * k // 16 - 1) * 256
    out = {}
    for rs in range(ms):
        for cs in range(ns):
            last = Coll(Array(256, ncp2.U32), tiles[rs][cs][0].refs.slice(per - 256, per))
            s_rep = Coll(Array(256, ncp2.F32), concat([s[16 * rs + r].refs for r in range(16) for _ in range(16)]))
            t_rep = Coll(Array(256, ncp2.F32), concat([t[16 * cs + c].refs for _ in range(16) for c in range(16)]))
            b_rep = Coll(Array(256, ncp2.BF16), concat([bias[16 * cs + c].refs for _ in range(16) for c in range(16)]))
            out[rs, cs] = B.batch(ncp2.NcpDequant, last, s_rep, t_rep, b_rep, axes=(0, 0, 0, 0))
    y = Coll(Array(m, Array(n, ncp2.BF16)), concat([out[i // 16, j // 16][16 * (i % 16) + j % 16].refs for i in range(m) for j in range(n)]))
    return tuple_of(y, array_of([tiles[rs][cs][0] for rs in range(ms) for cs in range(ns)]))


@pytest.mark.slow
def test_a_word_leaf_wired_to_the_served_output_fails_the_width_rule():
    """The same templates, but y read from a credited word: the tiles now reach y other than through their pinned Z, and so
    does every PoUW unit upstream of them, so all of them are width-checked, and fail."""
    fn = bind(CompositeDefinition("NcpLinearServedWord", 1, ("M", "K", "N", "PERM"), ncp2._linear_sig, _served_word_body,
                                  register=False), M=SHAPE.m, K=SHAPE.k, N=SHAPE.n, PERM="shift24")
    wr = P.width_rule_of(fn, SHAPE, "shift24", SHAPE.m * SHAPE.n)
    us = P.units_of(fn, SHAPE)
    assert not wr.ok
    assert {us[u].role for u, _ob, _og in wr.too_wide} == set(P.POUW_ROLES) - {"form_y"}
    # the verifier never takes this Program: it builds its own and compares digests
    from verity.ml.circuits.codec import encode_program
    from verity.ml.circuits.program import Program
    from verity.core.protocols.pouw.circuit import traced_as
    from verity.core.protocols.pouw.protocol import DomainError

    t = traced_as("ncp-v2-shift24", SHAPE.m, SHAPE.k, SHAPE.n)
    t.check_descriptor(encode_program(Program(t.definition)))
    with pytest.raises(DomainError):
        t.check_descriptor(encode_program(Program(fn)))


def test_the_width_rule_is_over_the_protocols_program_and_its_served_outputs():
    """`width_rule` takes no Definition and no served count: those are the protocol's. A variant's check refuses an empty
    served set, which would pass vacuously."""
    with pytest.raises(ValueError):
        P.width_rule_of(ncp2.linear(SHAPE), SHAPE, "shift24", 0)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["shift24", "word"])
def test_calls_on_one_weight_compose_without_recompute(kind):
    """Two calls on one weight in one root: the weight's Y units once, each call's own units reading them. With Y per call
    the second call recomputed the first's whole Y unit, and with the key's per-call lanes last two of its XORs; now no value is computed in two units."""
    uc = P.window_cut(ncp2.Shape(16, 64, 16), 2, kind)
    assert uc.verdict.ok, uc.verdict.codes
    assert Counter(u.role for u in uc.units) == {"form_y": 1, "digest": 2, "key": 2, "form_x": 2, "tile": 2, "dequant": 512}


def _naive_form_x_body(B, S, rows, key, rid):
    """`NcpFormX`'s body with E₁'s row index in one lane only, the layout the recompute check rejects."""
    e = ncp2.Emit(B)
    k, pi = S.K, ncp2.perm(S.PERM, S.K)
    key_lanes = [(key[2 * t], key[2 * t + 1]) for t in range(4)]
    out = []
    for r in range(16):
        halves = ncp2.shake256(e, [(rid[r], 0)] + ncp2.keccak_lanes(ncp2.TAG_E1) + key_lanes, k)
        nz = ncp2._noise(e, halves, k)
        a = [rows[r][j] for j in range(k)]
        out.append(array_of([e.call(ncp2.W.PouwFormX, a[j], nz[j]) for j in range(k)]
                            + [e.call(ncp2.W.PouwFormX, a[j], nz[pi[j]]) for j in range(k)]
                            + [e.call(ncp2.W.PouwFormX3, a[j], nz[j], nz[pi[j]]) for j in range(k)]))
    return array_of(out)


def _two_strips(form_x):
    """Two X units of one call, reading the same key: a root of two instances and nothing else."""
    k = 64
    sig = lambda S: ((("rows", Array(2, Array(16, Array(k, ncp2.W.I7)))), ("key", Array(8, ncp2.U32)),  # noqa: E731
                      ("rowids", Array(2, Array(16, ncp2.U32)))), Array(2, Array(16, Array(3 * k, ncp2.W.U8))))
    return bind(CompositeDefinition("TwoStrips", 0, (), sig, lambda B, S, rows, key, rid: B.batch(form_x, rows, key, rid, axes=(0, None, 0)),
                                    register=False))


def _check(fn):
    g = CallGraph(fn)
    per = fn.body.nodes[0].fn.gates
    return check_cut(g, [g.keep[c] // per for c in range(g.n)], limits=None)


@pytest.mark.slow
def test_the_index_in_every_column_is_what_keeps_the_xof_recompute_free():
    """With the row index in one lane, round 1 of every row's Keccak computes values of the key alone, the same in every X
    unit of the call: a recompute across units. With it in each of the five columns' first lane, there is none."""
    naive = bind(CompositeDefinition("NcpFormXNaive", 1, ("K", "PERM"), ncp2.NcpFormXDef.signature, _naive_form_x_body,
                                     register=False), K=64, PERM="shift24")
    assert "gate-recomputed" in _check(_two_strips(naive)).codes
    assert _check(_two_strips(ncp2.templates(64, "shift24")["form_x"])).ok
