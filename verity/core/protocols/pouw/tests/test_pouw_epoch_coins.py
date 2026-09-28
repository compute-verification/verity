"""PoUW's epoch coins (`coins`, `audit.Epoch.start`): the vectors are pinned and each pinned opening opens against its
root; the salt is the challenger's salt coin, opened after the weights' root; a foreign, tampered, mislabelled,
early or other-epoch coin is refused, for the salt and for the draw; and a challenger opens each coin once."""

from __future__ import annotations

import dataclasses
import json
import random

import pytest

import epoch_coins as E
import epoch_coins_vectors as V
from verity.core.protocols.pouw import coins as EC
from verity.core.protocols.pouw.audit import Epoch, Prover, Sampled, Verifier, Workload, salt_of, weights_root
from verity.core.protocols.pouw.protocol import DomainError, Matmul, Shape
from verity.core.protocols.pouw.schemes import ncp

PINNED = json.loads(V.OUT.read_text())


def test_the_vectors_are_pinned():
    assert V.vectors() == PINNED


@pytest.mark.parametrize("case", PINNED["cases"], ids=lambda c: c["case"])
def test_each_pinned_opening_opens_against_its_root(case):
    commitment = EC.CoinCommitment.from_json(case["commitment"])
    for doc, after in zip(case["openings"], (case["weights_root"], case["transcript_root"]), strict=True):
        o = EC.CoinOpening.from_json(doc)
        assert EC.coin_reason(commitment, o, o.label, bytes.fromhex(after)) is None
        assert EC.leaf_value(o.coin, o.salt).hex() == case["values"][o.label] and len(o.path) == 1
    salt_coin = EC.CoinOpening.from_json(case["openings"][0])
    assert salt_of(commitment, salt_coin.coin, bytes.fromhex(case["weights_root"])).hex() == case["salt"]


def setup():
    s, rng = ncp.NCP(k_min=64, n_min=64), random.Random(3)
    work = Workload((Matmul(0, Shape(32, 64, 64), 7),))
    weights = {7: [ncp.int7_bytes([rng.randint(-64, 63) for _ in range(64)]) for _ in range(64)]}
    acts = {0: [ncp.int7_bytes([rng.randint(-64, 63) for _ in range(64)]) for _ in range(32)]}
    prover = Prover(s, weights)
    return s, work, acts, prover, prover.weight_roots


def test_the_lifecycle_takes_the_salt_and_the_draw_from_the_challenger():
    """The challenger commits before the epoch, opens the salt coin after the weights' root and the draw coin after the
    transcript's root; the epoch's salt is `salt_of` that coin, and the audit draws from the draw coin."""
    s, work, acts, prover, roots = setup()
    ch = E.challenger()
    opening = ch.open("salt", weights_root(roots))
    epoch = Epoch.start(work, roots, ch.commitment, opening)
    assert epoch.coins == ch.commitment and epoch.salt == salt_of(ch.commitment, opening.coin, weights_root(roots))
    assert epoch.salt != Epoch.unverified(work, roots, opening.coin).salt
    tr = prover.prove(epoch, acts)
    audit = Verifier(s, epoch).audit(prover, tr, Sampled(4), ch)
    assert audit.accepted and len(audit.selected) == 4
    draw = E.challenger().open("draw", tr.commitment.root)
    assert audit.selected == Sampled(4).select(s, epoch, tr, draw.coin, Verifier(s, epoch).refs, prover.digest)


def test_the_salt_refuses_a_coin_that_does_not_open_against_the_epochs_commitment():
    s, work, acts, prover, roots = setup()
    root = weights_root(roots)
    honest = E.challenger().open("salt", root)
    refused = (
        ("does not open against the epoch's coin commitment", E.challenger(b"a foreign challenger").open("salt", root)),
        ("does not open against the epoch's coin commitment", dataclasses.replace(honest, coin=bytes(32))),
        ("does not open against the epoch's coin commitment", dataclasses.replace(honest, salt=bytes(192))),
        ("is the epoch's 'draw' coin, not its 'salt' coin", E.challenger().open("draw", root)),
        ("was not opened after what the verifier recorded", E.challenger().open("salt", b"before the weights")),
        ("is epoch 8's, not epoch 7's", E.challenger(epoch=8).open("salt", root)),
        ("opening is malformed", dataclasses.replace(honest, coin=honest.coin[:31])),
    )
    for reason, opening in refused:
        with pytest.raises(DomainError, match=f"^the salt's coin {reason}"):
            Epoch.start(work, roots, E.COINS, opening)
    with pytest.raises(DomainError, match="every weight"):
        Epoch.start(work, {}, E.COINS, honest)
    assert Epoch.start(work, roots, E.COINS, honest).salt == salt_of(E.COINS, honest.coin, root)


def test_the_draw_refuses_bare_bytes_and_a_foreign_coin_on_an_epoch_with_coins():
    s, work, acts, prover, roots = setup()
    epoch = Epoch.start(work, roots, E.COINS, E.challenger().open("salt", weights_root(roots)))
    tr = prover.prove(epoch, acts)
    v = Verifier(s, epoch)
    for source, reason in ((b"verifier secret", "draws from its draw coin, not from bare bytes"),
                           (E.challenger(b"a foreign challenger"), "the draw's coin does not open"),
                           (E.Answering(E.challenger().open("draw", b"before the transcript")), "not opened after")):
        audit = v.audit(prover, tr, Sampled(4), source)
        assert not audit.accepted and audit.selected == () and reason in audit.verdicts[0].reason
    bare = Epoch.unverified(work, roots, b"seed")
    audit = Verifier(s, bare).audit(prover, prover.prove(bare, acts), Sampled(4), E.challenger())
    assert not audit.accepted and "no coin commitment" in audit.verdicts[0].reason


def test_a_challenger_opens_each_coin_once_and_only_its_own_labels():
    ch = E.challenger()
    ch.open("salt", b"the weights' root")
    with pytest.raises(DomainError, match="already open"):
        ch.open("salt", b"the weights' root")
    with pytest.raises(DomainError, match="an epoch's coins"):
        ch.open("noise", b"")
    assert E.challenger().commitment == ch.commitment != E.challenger(b"another").commitment


def test_a_fresh_challenger_draws_from_the_operating_system():
    a, b = EC.Challenger(7), EC.Challenger(7)
    assert a.commitment != b.commitment and len(a.commitment.root) == 64
    with pytest.raises(ValueError, match="returned"):
        EC.Challenger(7, lambda n: bytes(n - 1))
