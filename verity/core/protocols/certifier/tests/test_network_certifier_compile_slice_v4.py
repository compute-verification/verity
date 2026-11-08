"""The compile slice charging the length terms (`compile_slice_v4.CompileSlice_v4`): CompileSlice_v1 with
ChargeLink_v2{D, 0, 0, K} in place of ChargeLink_v1{D}. Its charge is v1's for the window's decisions plus each certified
arrival's length term, bitlen(max_tokens) + 1 bits, read from ``cert``'s certified output length and max_tokens. On both
demonstration slices and on window 212 of the served run the circuit accepts that charge and refuses one bit less, and a
max_tokens the certified output length doesn't fit is refused by ``code``. A max_tokens other than the certified one is
another ``cert`` row, so the b‖c of ``cert`` to the certified record refuses it."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import compile_slice as CS
from verity.core.protocols.certifier.tests.test_network_certifier_compile_slice import BOOK
from verity.core.protocols.certifier.tests.test_network_certifier_compile_slice_w212 import CARRY, OWN, _load
from verity.ml.protocols.certifier import compile_slice_v4 as V4
from verity.ml.protocols.certifier import window_charge as WC


@pytest.fixture(scope="module", params=["demo", "demo_async"])
def x(request):
    st, sl = getattr(V4, request.param)()
    return SimpleNamespace(st=st, S=SimpleNamespace(**st), sl=sl, v1=getattr(CS, request.param)()[1])


def _tails(sl) -> int:
    return sum(WC.length_bits(m) for m in sl.max_tokens)


def _with_mt(S, c: CS.Case, slot: int, mt: int) -> SimpleNamespace:
    """`c`'s rows with ``cert``'s max_tokens in `slot` set to `mt`."""
    F = CS.fields(S, V4.CODE)["cert"]
    at = sum(n * w for _name, n, w in F[:-1]) + slot * WC.MW
    cert = list(c.rows["cert"])
    cert[at:at + WC.MW] = [(mt >> j) & 1 for j in range(WC.MW)]
    return SimpleNamespace(rows={**c.rows, "cert": cert})


def test_the_charge_is_v1s_and_the_length_tails(x):
    """The honest unit is accepted (0x3F) at v1's charge of the window's decisions (`compile_slice.case` from the same
    book's priors) plus each arrival's length term, and the circuit agrees with the reference at that charge and one
    bit below it, where only ``charge`` (bit 3) is refused."""
    honest = V4.case(x.S, x.sl, BOOK)
    assert honest.verdict == V4.ACCEPT
    assert honest.length == CS.case(x.S, x.v1, BOOK).length + _tails(x.sl)
    low = V4.case(x.S, x.sl, BOOK, charge=honest.length - 1)
    assert low.verdict == V4.ACCEPT & ~(1 << 3)
    assert V4.evaluate(x.st, [honest, low]) == [honest.verdict, low.verdict]


def test_an_output_length_past_max_tokens_is_refused(x):
    """A slot's max_tokens below its certified output length's width, 2^bitlen(mt) <= olen, leaves the length term
    uncodable: the reference refuses ``code`` and leaves ``charge`` unspecified, and the circuit clears bit 2."""
    honest = V4.case(x.S, x.sl, BOOK)
    k, (_id, _P, O, *_cap) = max(enumerate(x.sl.arrivals), key=lambda a: a[1][2])
    assert O >= 2
    short = _with_mt(x.S, honest, k, O >> 1)
    flags = V4.check(x.S, short.rows)
    assert flags["code"] is False and flags["charge"] is None
    assert V4.evaluate(x.st, [short])[0] & (1 << 2) == 0


def test_max_tokens_is_cert_s(x):
    """A lower max_tokens that still fits the output length charges less, and the circuit accepts it at that charge:
    the length slots aren't the circuit's to hold, ``cert``'s b‖c to the certified record is, and the row differs."""
    honest = V4.case(x.S, x.sl, BOOK)
    k = max(range(len(x.sl.arrivals)), key=lambda j: x.sl.max_tokens[j] - x.sl.arrivals[j][2].bit_length())
    mt = (1 << x.sl.arrivals[k][2].bit_length()) - 1
    assert WC.length_bits(mt) < WC.length_bits(x.sl.max_tokens[k])
    cut = replace(x.sl, max_tokens=tuple(mt if j == k else m for j, m in enumerate(x.sl.max_tokens)))
    lower = V4.case(x.S, cut, BOOK)
    assert lower.verdict == V4.ACCEPT and lower.length < honest.length
    assert lower.rows["cert"] != honest.rows["cert"]
    assert V4.evaluate(x.st, [lower]) == [V4.ACCEPT]


@pytest.mark.slow
def test_w212_charges_v1s_segment_and_its_length_tails():
    """Window 212 at its own statics (`compile_slice_w212.json`), its three requests' max_tokens their certified output
    lengths (each ran to its cap: ``cap`` = 1 in `compile_slice_w212_class_async.json`): from the session's carry into
    w212, the unit charges v1's 27 bits plus 5 + 4 + 5 for the length terms, and the circuit accepts that charge and
    refuses one bit less."""
    w = _load(OWN)
    sl = V4.Slice(w.sl.arrivals, w.sl.decisions, w.sl.steps, w.sl.tokens, tuple(a[2] for a in w.sl.arrivals))
    v1 = CS.case(w.S, w.sl, BOOK, served=w.served, carry=CARRY)
    kw = {"served": w.served, "carry": WC.WindowCarry(CARRY.counts, CARRY.run)}
    honest = V4.case(w.S, sl, BOOK, **kw)
    assert v1.length == 27 and _tails(sl) == 14
    assert honest.verdict == V4.ACCEPT and honest.length == v1.length + _tails(sl)
    low = V4.case(w.S, sl, BOOK, charge=honest.length - 1, **kw)
    assert V4.evaluate(w.st, [honest, low]) == [V4.ACCEPT, V4.ACCEPT & ~(1 << 3)]
