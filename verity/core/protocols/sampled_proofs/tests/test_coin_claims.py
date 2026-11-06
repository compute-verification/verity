"""Every coin source is named by a claim instance (`challenger.coin_claim`): its ``claim``, a ``uniform/<instance>``
assumption of `verity.core.claims`.  The Lean verifier's `coins` is A3's ``uniform/io-getrandombytes``; a served
challenger refuses a source no claim instance names (`served_claim`), `os.urandom` among them, and a misnamed one is
refused wherever it is read; a test's seeded stream stays unnamed, and the challenger keeps the claim it sampled under."""

from __future__ import annotations

import os
import random

import pytest

from verity.core import claims
from verity.core.protocols.sampled_proofs import challenger as C
from verity.core.protocols.sampled_proofs.one_stage import verifier as V

A3 = "uniform/io-getrandombytes"


def test_the_lean_verifiers_coins_are_named_by_a3():
    assert C.served_claim(V.coins) == V.CLAIM == A3
    assert claims.get(A3).name == "Independent uniform output of IO.getRandomBytes"


@pytest.mark.parametrize("source", [os.urandom, random.Random(0).randbytes, lambda n: bytes(n)])
def test_a_coin_without_a_claim_instance_is_refused_for_a_served_challenger(source):
    assert C.coin_claim(source) is None
    with pytest.raises(C.CoinError, match="coins-without-claim"):
        C.served_claim(source)
    assert C.LiveChallenger("verity/test/coins/v1", 0, [("x", 1)], random.Random(0).randbytes).claim is None


@pytest.mark.parametrize("claim", ["cr/sha-512", "uniform/no-such-source", "uniform", "io-getrandombytes", 7])
def test_a_misnamed_source_is_refused(claim):
    with pytest.raises(C.CoinError, match="uniform/<instance>"):
        C.named(claim, random.Random(0).randbytes)

    def source(n: int) -> bytes:
        return bytes(n)
    source.claim = claim
    with pytest.raises(C.CoinError, match="uniform/<instance>"):
        C.coin_claim(source)
    with pytest.raises(C.CoinError, match="uniform/<instance>"):
        C.LiveChallenger("verity/test/coins/v1", 0, [("x", 1)], source)


def test_a_named_source_draws_its_own_bytes_and_the_challenger_keeps_its_claim():
    source = C.named(A3, random.Random(1).randbytes)
    assert C.served_claim(source) == A3 and source(8) == random.Random(1).randbytes(8)
    ch = C.LiveChallenger("verity/test/coins/v1", 0, [("x", 2)], C.named(A3, random.Random(2).randbytes))
    assert ch.claim == A3
    assert ch.commitment == C.LiveChallenger("verity/test/coins/v1", 0, [("x", 2)], random.Random(2).randbytes).commitment
