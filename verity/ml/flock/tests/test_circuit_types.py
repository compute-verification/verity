"""Circuit types (`verity_flock.circuit_types`): a two-level adder evaluates to the sum and equals its expansion; the digest is
RFC 8785 JSON under SHA-512 and ignores names; table reads read their table; and `validate` refuses each broken rule."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

from verity.ml.circuits import library
from verity_flock import circuit_types as CT
from verity_flock import gf2

T = 64
MASK = (1 << T) - 1
#: contents and digests for a second implementation (the docs site's) to recheck: ``python test_circuit_types.py --write``
VECTORS = Path(__file__).with_name("circuit_type_vectors.json")


def _full_adder(names=("a", "b", "c")) -> CT.CircuitType:
    C = gf2.Circuit()
    a, b, c = (C.input(n, 1)[0] for n in names)
    return CT.from_gf2(C, [a, b, c], [a ^ b ^ c, C.AND(a, b) ^ C.AND(c, a ^ b)], ((3, 1),), ((2, 1),), library.PROGRAM, {})


def _adder(fa: CT.CircuitType, k: int = 4) -> CT.CircuitType:
    """a + b over k-bit a, b: k calls of the full adder, the first carry the constant 0."""
    items, n, carry, sums = [], 2 * k, ((), 0), []
    for i in range(k):
        items.append((CT.CALL, 0, (((i,), 0), ((k + i,), 0), carry)))
        sums.append(((n,), 0))
        carry = ((n + 1,), 0)
        n += 2
    return CT.CircuitType(library.PROGRAM, ((k, 1), (k, 1)), ((k + 1, 1),), (fa.digest(),), (), tuple(items), tuple(sums + [carry]))


def _bits(xs, width):
    return [sum(((x >> j) & 1) << v for v, x in enumerate(xs)) for j in range(width)]


def _ints(bits):
    return [sum(((b >> v) & 1) << j for j, b in enumerate(bits)) for v in range(T)]


@pytest.fixture(scope="module")
def adder():
    fa = _full_adder()
    add4 = _adder(fa)
    types = {fa.digest(): fa, add4.digest(): add4}
    for t in (fa, add4):
        CT.validate(t, types.__getitem__)
    return types, add4


def test_a_call_tree_computes_the_sum(adder):
    types, add4 = adder
    rng = random.Random(1)
    a, b = [rng.randrange(16) for _ in range(T)], [rng.randrange(16) for _ in range(T)]
    out = CT.evaluate(types, add4.digest(), _bits(a, 4) + _bits(b, 4), T, lambda *_: 0)
    assert _ints(out) == [x + y for x, y in zip(a, b)]


def test_expansion_equals_the_tree_and_gf2s_own_adder(adder):
    types, add4 = adder
    flat = CT.expand(types, add4.digest())
    assert not flat.calls and len(flat.items) == 4 * 2
    C = gf2.Circuit()
    a, b = C.input("a", 4), C.input("b", 4)
    ref = CT.from_gf2(C, a + b, gf2.add(C, a, b, width=5), ((4, 1), (4, 1)), ((5, 1),), library.PROGRAM, {})
    rng = random.Random(2)
    ins = [rng.getrandbits(T) for _ in range(8)]
    got = CT.evaluate(types, add4.digest(), ins, T, lambda *_: 0)
    assert CT.evaluate({"f": flat}, "f", ins, T, lambda *_: 0) == got
    assert CT.evaluate({"r": ref}, "r", ins, T, lambda *_: 0) == got


def test_the_digest_is_sha512_over_rfc8785_json_and_ignores_names():
    fa = _full_adder()
    assert fa.canonical() == json.dumps(fa.content(), sort_keys=True, separators=(",", ":")).encode()
    assert fa.digest() == hashlib.sha512(b"verity/circuit-type/v1\0" + fa.canonical()).hexdigest()
    assert _full_adder(("x", "y", "z")).digest() == fa.digest()
    assert CT.from_content(json.loads(fa.canonical())) == fa


def test_a_callee_is_listed_once_however_many_calls(adder):
    _, add4 = adder
    assert len(add4.calls) == 1 and sum(1 for it in add4.items if it[0] == CT.CALL) == 4


def test_a_table_read_reads_its_table():
    rng = random.Random(3)
    words = [rng.getrandbits(8) for _ in range(64)]
    sha = hashlib.sha512(b"".join(w.to_bytes(4, "little") for w in words)).hexdigest()
    C = gf2.Circuit()
    x = C.input("x", 6)
    v = C.lookup("toy", x, 8)
    t = CT.from_gf2(C, x, [v[j] ^ x[0] for j in range(8)], ((1, 6),), ((1, 8),), library.PROGRAM, {"toy": (sha, 6, 8)})
    CT.validate(t, {}.__getitem__)
    assert [it[0] for it in t.items] == [CT.READ] and t.tables == ((sha, 6, 8),)
    xs = [rng.randrange(64) for _ in range(T)]
    out = CT.evaluate({"t": t}, "t", _bits(xs, 6), T, lambda s, i: words[i])
    assert _ints(out) == [words[i] ^ (0xFF if i & 1 else 0) for i in xs]


def _refused(t, callee=None, match=None):
    with pytest.raises(CT.InvalidType, match=match):
        CT.validate(t, callee or {}.__getitem__)


def test_validate_refuses_each_broken_rule(adder):
    types, add4 = adder
    fa = types[add4.calls[0]]
    ex2 = library.table("ex2")
    base = dict(label=library.PROGRAM, ins=((2, 1),), outs=((1, 1),), calls=(), tables=())
    one = CT.CircuitType(items=((CT.AND, ((0,), 0), ((1,), 0)),), outputs=(((2,), 0),), **base)
    CT.validate(one, {}.__getitem__)
    _refused(CT.CircuitType(**{**base, "label": "public"}, items=one.items, outputs=one.outputs), match="label")
    _refused(CT.CircuitType(items=((CT.AND, ((2,), 0), ((1,), 0)),), outputs=(((2,), 0),), **base), match="earlier wires")
    _refused(CT.CircuitType(items=((CT.AND, ((1, 0), 0), ((1,), 0)),), outputs=(((2,), 0),), **base), match="ascending")
    _refused(CT.CircuitType(items=one.items, outputs=(((2,), 0), ((0,), 1)), **base), match="output forms")
    _refused(CT.CircuitType(items=one.items + ((CT.AND, ((0,), 0), ((2,), 1)),), outputs=(((2,), 0),), **base), match="dead")
    _refused(CT.CircuitType(items=((5, 0, 0),), outputs=(((0,), 0),), **base), match="only AND")
    bad_call = CT.CircuitType(library.PROGRAM, ((2, 1),), ((2, 1),), (fa.digest(),), (), ((CT.CALL, 0, (((0,), 0), ((1,), 0))),),
                              (((2,), 0), ((3,), 0)))
    _refused(bad_call, types.__getitem__, match="passes 2 bits")
    lib_calls_prog = CT.CircuitType(library.LIBRARY, ((3, 1),), ((2, 1),), (fa.digest(),), (),
                                    ((CT.CALL, 0, (((0,), 0), ((1,), 0), ((2,), 0))),), (((3,), 0), ((4,), 0)))
    _refused(lib_calls_prog, types.__getitem__, match="calls a program type")
    toy = ("0" * 128, 2, 1)
    lib_reads_prog = CT.CircuitType(library.LIBRARY, ((2, 1),), ((1, 1),), (), (toy,), ((CT.READ, 0, (((0,), 0), ((1,), 0))),),
                                    (((2,), 0),))
    _refused(lib_reads_prog, match="program table")
    wrong_shape = CT.CircuitType(library.LIBRARY, ((22, 1),), ((31, 1),), (), ((ex2.sha512, 22, 31),),
                                 ((CT.READ, 0, tuple(((i,), 0) for i in range(22))),), tuple(((22 + j,), 0) for j in range(31)))
    _refused(wrong_shape, match="index and")
    order = CT.CircuitType(library.PROGRAM, ((8, 1),), ((4, 1),), ("a" * 128, fa.digest()), (),
                           ((CT.CALL, 1, (((0,), 0), ((1,), 0), ((2,), 0))), (CT.CALL, 1, (((3,), 0), ((4,), 0), ((5,), 0)))),
                           (((8,), 0), ((9,), 0), ((10,), 0), ((11,), 0)))
    _refused(order, {fa.digest(): fa, "a" * 128: fa}.__getitem__, match="first use")


def _ex2_read() -> CT.CircuitType:
    ex2 = library.table("ex2")
    return CT.CircuitType(library.LIBRARY, ((1, 23),), ((1, 31),), (), ((ex2.sha512, 23, 31),),
                          ((CT.READ, 0, tuple(((i,), 0) for i in range(23))),), tuple(((23 + j,), 0) for j in range(31)))


def test_a_library_read_of_a_library_table_is_valid():
    CT.validate(_ex2_read(), {}.__getitem__)


def _vectors() -> list[dict]:
    fa = _full_adder()
    return [{"name": name, "content": t.content(), "digest": t.digest()}
            for name, t in (("full adder", fa), ("4-bit adder of 4 full-adder calls", _adder(fa)), ("ex2 read", _ex2_read()))]


def test_the_vectors_are_current():
    assert json.loads(VECTORS.read_text()) == _vectors()


if __name__ == "__main__" and "--write" in sys.argv:
    VECTORS.write_text(json.dumps(_vectors(), indent=1) + "\n")
