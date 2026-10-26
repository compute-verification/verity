"""`ncp-v2`'s Program (`verity.ml.protocols.pouw.circuit.ncp2`) evaluates to the native reference, which is the MVP's own arithmetic on
the two-level key's noise."""
from __future__ import annotations

import hashlib

import pytest

import circuit_fixtures as F
from verity.ml.circuits.evaluation import evaluate
from verity.ml.protocols.pouw.circuit import ncp2
from verity.ml.protocols.pouw.circuit import reference as R
from verity.core.protocols.pouw.protocol import DomainError
from verity.ml.protocols.pouw.schemes.ncp import assemble, checked_route_u, shift_perm


def _outputs(shape, call, kind="shift24"):
    out = evaluate(ncp2.linear(shape, kind), *call.args())
    mn = shape.m * shape.n
    per = (shape.steps - 1) * 256
    words = [list(out[mn + i * per:mn + (i + 1) * per]) for i in range((shape.m // 16) * (shape.n // 16))]
    return list(out[:mn]), words


def test_reference_z_is_the_integer_product():
    call = F.call(32, 64, 32)
    assert R.evaluate(call).z == R.product(call)


def test_reference_words_are_the_mvps_route_u():
    """The reference's credited words are `schemes.ncp`'s route-U checked values on the two-level key's E₁ and F₁."""
    call = F.call(32, 64, 32, seed=11)
    ref = R.evaluate(call)
    k = 64
    e1 = [R.e1_row(ref.key, call.rowids[i], k) for i in range(32)]
    f1 = [R.f1_col(call, call.colids[j], k) for j in range(32)]
    pi, sigma = shift_perm(k)
    checked = checked_route_u(assemble([list(r) for r in call.a], [list(r) for r in call.b], e1, f1, pi, sigma))
    assert ref.words[1] == [checked[r][16 + c][t] for t in range(11) for r in range(16) for c in range(16)]


def test_the_root_is_rfc6962_over_the_strip_digests():
    call = F.call(48, 64, 32)
    d = [R.strip_digest(call.a[16 * i:16 * (i + 1)]) for i in range(3)]
    assert R.root(call.a) == R.node(R.node(d[0], d[1]), d[2])
    assert R.root(call.a[:16]) == d[0]
    assert d[0] == hashlib.sha512(ncp2.TAG_STRIP + b"".join(R.row_digest(r) for r in call.a[:16])).digest()


def test_index_binding_separates_identical_strips():
    """Two identical A strips (a request batched twice, or padding) get different noise and different words, because the
    global row index is in E₁'s key (X-SPC-30)."""
    strip = tuple(tuple((7 * i + j) % 128 - 64 for j in range(64)) for i in range(16))
    call = F.call(32, 64, 32, a=strip + strip)
    ref = R.evaluate(call)
    assert R.e1_row(ref.key, call.rowids[0], 64) != R.e1_row(ref.key, call.rowids[16], 64)
    assert ref.words[0] != ref.words[2]
    same = F.call(32, 64, 32, a=strip + strip, rowids=list(range(16)) * 2)
    assert R.evaluate(same).words[0] == R.evaluate(same).words[2]         # the binding is what separates them


def test_the_key_binds_every_anchored_field():
    call = F.call(32, 64, 32)
    base = R.key(call, R.root(call.a))
    for field, value in (("forward", 4), ("layer", 6), ("site", 3), ("weight", 12), ("salt", bytes(32))):
        other = R.Call(**{**call.__dict__, field: value})
        assert R.key(other, R.root(other.a)) != base, field


@pytest.mark.parametrize("kind", ["shift24", "word"])
def test_the_program_evaluates_to_the_reference(kind):
    """Both P's: `ncp-v2`'s word P and `ncp-v2-shift24`'s shift by 24 (X-SPC-98)."""
    shape = ncp2.Shape(32, 64, 32)
    call = F.call(32, 64, 32)
    ref = R.evaluate(call, kind)
    out = evaluate(ncp2.linear(shape, kind), *call.args(kind))
    mn, per = shape.m * shape.n, (shape.steps - 1) * 256
    y, words = list(out[:mn]), [list(out[mn + i * per:mn + (i + 1) * per]) for i in range(4)]
    assert y == [v for row in ref.y for v in row]
    assert words == ref.words


@pytest.mark.slow
def test_the_program_evaluates_to_the_reference_on_an_unbalanced_tree():
    """Three strips: the tree has two node units, and one strip's path is the root alone."""
    shape = ncp2.Shape(48, 64, 32)
    call = F.call(48, 64, 32, seed=5)
    ref = R.evaluate(call)
    y, words = _outputs(shape, call)
    assert y == [v for row in ref.y for v in row]
    assert words == ref.words


def test_the_weight_program_evaluates_to_the_reference():
    """`NcpWeight` commits each column's Y and c₀ as `schemes.ncp.assemble` has them, which the call reads."""
    call = F.call(32, 64, 32)
    ys, c0s = R.weight_outputs(call)
    assert list(evaluate(ncp2.weight(ncp2.Shape(32, 64, 32)), *call.weight_args())) == [v for col in ys for v in col] + c0s


@pytest.mark.slow
def test_a_window_evaluates_to_its_calls():
    """Two calls on one weight in one root (`NcpWindow`) give each call's reference outputs."""
    shape = ncp2.Shape(16, 64, 16)
    one = F.call(16, 64, 16, seed=3)
    two = R.Call(**{**F.call(16, 64, 16, seed=4).__dict__, "b": one.b, "t": one.t, "bias": one.bias, "salt": one.salt,
                    "weight": one.weight, "colids": one.colids, "forward": one.forward + 1})
    args = one.weight_args()[:4] + [list(one.t), list(one.bias), [x & 0x7F for c in (one, two) for r in c.a for x in r],
                                    [v for c in (one, two) for v in c.s]]
    args += [[w for c in (one, two) for w in (c.forward & 0xFFFFFFFF, c.forward >> 32)],
             [w for c in (one, two) for w in (c.position & 0xFFFFFFFF, c.position >> 32)], [i for c in (one, two) for i in c.rowids]]
    out = evaluate(ncp2.window(shape, 2), *args)
    mn, per = 256, (shape.steps - 1) * 256
    for j, c in enumerate((one, two)):
        ref = R.evaluate(c)
        assert list(out[j * mn:(j + 1) * mn]) == [v for row in ref.y for v in row]
        assert list(out[2 * mn + j * per:2 * mn + (j + 1) * per]) == ref.words[0]


def test_shapes_outside_the_circuit_are_refused():
    with pytest.raises(DomainError):
        ncp2.linear(ncp2.Shape(24, 64, 32))
    with pytest.raises(DomainError):
        ncp2.linear(ncp2.Shape(32, 48, 32))
