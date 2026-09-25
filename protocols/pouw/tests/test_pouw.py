"""PoUW alone: a synthetic allowance and a stub authenticated reader stand in for sampled proofs."""

from fractions import Fraction

import pytest

from verity.evaluation import evaluate_region
from verity.ir import Array, Program, Tuple, Value, bind, composite
from verity.resource_accounting import IntegrityAllowance
from verity_pouw import NoisedDot, PoUW, Rejected, Request, noise, seed

K = 8
KEYS = {"gpu0": b"k0" * 16, "gpu1": b"k1" * 16}
V = Array(K, Value(16))


@composite("PouwTestOneTile", 1, [], lambda S: (tuple((n, V) for n in "xwef"), Tuple(Value(16), Value(16))))
def OneTile(B, S, x, w, e, f):
    return B.call(bind(NoisedDot, K=K), x, w, e, f)


@pytest.fixture(scope="module")
def tile():
    program = Program(OneTile)
    pouw = PoUW(program, K=K, p=Fraction(1), eta=0.01)
    req = Request("r0", "gpu0", b"h" * 32, (0, program.circuit.size))
    log = {"pouw/header/r0": (1, req.header)}.__getitem__
    noise_values = pouw.prescribed([req], KEYS, log)
    xw = {g: (3 * g + 1) & 0xFFFF for g in range(2 * K)}
    values = {**xw, **noise_values}
    evaluate_region(program.circuit, program.circuit[0:program.circuit.size], values)
    return pouw, req, values, xw


def test_seed_is_bound_to_key_device_and_header():
    s = seed(KEYS["gpu0"], "gpu0", b"h")
    assert s == seed(KEYS["gpu0"], "gpu0", b"h")
    assert s not in {seed(KEYS["gpu1"], "gpu0", b"h"), seed(KEYS["gpu0"], "gpu1", b"h"), seed(KEYS["gpu0"], "gpu0", b"g")}


def test_noised_dot_serves_the_plain_product(tile):
    pouw, _, values, xw = tile
    (t,) = pouw.tiles
    assert values[t.returns[0]] == sum(xw[i] * xw[K + i] for i in range(K)) & 0xFFFF
    e, f = noise(seed(KEYS["gpu0"], "gpu0", b"h" * 32), K)
    assert [values[g] for g in t.args[2 * K:]] == e + f


def test_prescribed_requires_the_registered_header(tile):
    pouw, req, _, _ = tile
    with pytest.raises(Rejected, match="header"):
        pouw.prescribed([req], KEYS, {"pouw/header/r0": (1, b"x" * 32)}.__getitem__)


def test_tickets_verify_through_the_reader_and_nothing_else(tile):
    pouw, req, values, _ = tile
    tickets = pouw.tickets([req], KEYS, values, open_=lambda g: None)
    honest = lambda g, opening: values[g]  # noqa: E731
    assert pouw.verify(tickets, [req], keys=KEYS, read=honest) == {"gpu0": 1}
    with pytest.raises(Rejected):
        pouw.verify(tickets, [req], keys=KEYS, read=lambda g, opening: None)
    with pytest.raises(Rejected):
        pouw.verify(tickets * 2, [req], keys=KEYS, read=honest)
    other_device = Request("r0", "gpu1", req.header, req.interval)
    lossy = PoUW(pouw.program, K=K, p=Fraction(1, 1 << 20), eta=0.01)
    with pytest.raises(Rejected, match="losing"):
        lossy.verify(tickets, [other_device], keys=KEYS, read=honest)


def test_claim_charges_the_allowance_worst_case_to_every_device():
    pouw = PoUW(Program(OneTile), K=K, p=Fraction(1, 4), eta=0.01)
    exact = IntegrityAllowance(n_r=1000, n_v=4, p=Fraction(1), k=4, delta=0.01)
    loose = IntegrityAllowance(n_r=1000, n_v=4, p=Fraction(1, 4), k=2, delta=0.01)
    (a,) = pouw.claim({"gpu0": 1000}, exact, tiles_per_vu=1)
    (b,) = pouw.claim({"gpu0": 1000}, loose, tiles_per_vu=1)
    assert a.forgeable == 0 and 0 < b.forgeable and b.macs < a.macs < 4000 * K
    assert pouw.tiles_lower_bound(0) == 0
    assert pouw.tiles_lower_bound(100) < pouw.tiles_lower_bound(1000) < 4000
