"""PoUW alone: a synthetic profile and a stub reader stand in for sampled proofs."""

from fractions import Fraction

import pytest

from verity.evaluation import evaluate_region
from verity.ir import Array, Program, Tuple, Value, bind, composite
from verity.proofs import IntegrityProfile
from verity_pouw import NeedUnmet, NoisedDot, PoUW, Rejected, noise

K = 8
KEYS = {"gpu0": b"k0" * 16, "gpu1": b"k1" * 16}
V = Array(K, Value(16))
HEADER = b"h" * 32


@composite("PouwTestOneTile", 1, [], lambda S: (tuple((n, V) for n in "xwef"), Tuple(Value(16), Value(16))))
def OneTile(B, S, x, w, e, f):
    return B.call(bind(NoisedDot, K=K), x, w, e, f)


PROGRAM = Program(OneTile)
SPAN = (0, PROGRAM.circuit.size)
REQUESTS = {"r0": ("gpu0", HEADER)}
LOG = {"pouw/header/r0": (0.0, HEADER)}.__getitem__


def profile(**kw):
    base = dict(programs=(), prescribed=frozenset({"noise"}), linked=True, n_r=1000, n_v=4, p=Fraction(1), k=4,
                delta=0.01, replayed=(0, 0))
    return IntegrityProfile(**{**base, **kw})


@pytest.fixture
def epoch():
    pouw = PoUW(KEYS, K, Fraction(1), 0.01)
    pouw.issue("r0", "gpu0", HEADER, LOG)
    values = {g: (3 * g + 1) & 0xFFFF for g in range(2 * K)}
    values.update(pouw.prescribed(REQUESTS, {"s0": PROGRAM}, {"s0": {"r0": SPAN}})["s0"])
    evaluate_region(PROGRAM.circuit, PROGRAM.circuit[0:SPAN[1]], values)
    return pouw, values


def test_seeds_bind_key_device_header_and_step():
    pouw = PoUW(KEYS, K, Fraction(1), 0.01)
    s = pouw.seed("gpu0", b"h")
    assert s not in {pouw.seed("gpu1", b"h"), pouw.seed("gpu0", b"g")}
    assert noise(s, "s0", K) != noise(s, "s1", K)


def test_seeds_are_issued_only_for_registered_headers():
    with pytest.raises(Rejected, match="header"):
        PoUW(KEYS, K, Fraction(1), 0.01).issue("r0", "gpu0", b"other", LOG)


def test_noised_dot_serves_the_plain_product(epoch):
    pouw, values = epoch
    (t,) = pouw.tiles(PROGRAM, SPAN)
    assert values[t.returns[0]] == sum(values[i] * values[K + i] for i in range(K)) & 0xFFFF


def test_tickets_verify_through_the_reader_and_nothing_else(epoch):
    pouw, values = epoch
    tickets = pouw.tickets("s0", PROGRAM, {"r0": SPAN}, values)
    args = (REQUESTS, {"s0": PROGRAM}, {"s0": {"r0": SPAN}})
    assert pouw.verify(tickets, *args, lambda s, g: values[g]) == {"gpu0": 1}
    for read, tks in ((lambda s, g: None, tickets), (lambda s, g: values[g], tickets * 2)):
        with pytest.raises(Rejected):
            pouw.verify(tks, *args, read)


def test_claim_reads_what_it_needs_off_the_profile(epoch):
    pouw, _ = epoch
    (exact,) = pouw.claim({"gpu0": 1000}, {"gpu0": 4000}, profile(), tiles_per_vu=1)
    (loose,) = pouw.claim({"gpu0": 1000}, {"gpu0": 4000}, profile(p=Fraction(1, 4), k=2), tiles_per_vu=1)
    assert exact.forgeable == 0 and 0 < loose.forgeable and loose.macs < exact.macs <= exact.declared_macs
    assert exact.events == {"integrity": 0.01, "pouw/tickets/gpu0": 0.01}
    with pytest.raises(NeedUnmet):
        pouw.claim({"gpu0": 1}, {"gpu0": 1}, profile(prescribed=frozenset()), tiles_per_vu=1)
