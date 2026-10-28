"""PoUW's epoch coins (`coins`, `audit.Epoch.start`): the vectors are pinned and each pinned opening opens against its
root; the salt is the challenger's salt coin, opened after the weights' root; a foreign, tampered, mislabelled,
early or other-epoch coin is refused, for the salt and for the draw; and a challenger opens each coin once."""

from __future__ import annotations

import dataclasses
import json
import os
import random
import secrets
import sys
from pathlib import Path

import pytest

import epoch_coins as E
import epoch_coins_vectors as V
from verity.core.protocols.pouw import coins as EC
from verity.core.protocols.pouw import verifier
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.pouw.audit import Epoch, Workload, salt_of, weights_root
from verity.core.protocols.pouw.protocol import DomainError, Matmul, Shape
from verity.core.protocols.pouw.schemes import ncp

# the clear audit is a reference tool (benchmarks/pouw/clear), imported by name
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "benchmarks" / "pouw" / "clear"))
from clear_audit import Prover, Sampled, Verifier  # noqa: E402

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


def test_a_challenger_has_no_entropy_of_its_own():
    """A challenger draws only from the entropy it is handed (a served one's is the Lean verifier of record's,
    `verifier.coins`; `lean/check/tests/test_difftest.py` holds it to the Lean): there is no default, so none reads
    Python's generator, and an entropy that answers short is refused."""
    with pytest.raises(TypeError, match="entropy"):
        EC.Challenger(7)
    with pytest.raises(TypeError, match="entropy"):
        Wd.live_challenger(7, 2)
    a, b = E.challenger(b"a"), E.challenger(b"b")
    assert a.commitment != b.commitment and len(a.commitment.root) == 64
    with pytest.raises(ValueError, match="returned"):
        EC.Challenger(7, lambda n: bytes(n - 1))


def test_a_served_challengers_coins_are_named_by_a_claim_instance():
    """The Lean verifier of record's `coins` is named by A3 (`verifier.CLAIM`, ``uniform/io-getrandombytes``),
    and a challenger keeps the claim instance its entropy carries; one without a claim instance, `os.urandom` among
    them, is refused as a served challenger's source (`challenger.served_claim`)."""
    assert Ch.served_claim(verifier.coins) == verifier.CLAIM == "uniform/io-getrandombytes"
    named = Ch.named(verifier.CLAIM, random.Random(0).randbytes)
    assert EC.Challenger(7, named).claim == Wd.live_challenger(7, 2, named).claim == verifier.CLAIM
    assert EC.Challenger(7, random.Random(0).randbytes).claim is None
    for source in (os.urandom, random.Random(0).randbytes):
        with pytest.raises(Ch.CoinError, match="coins-without-claim"):
            Ch.served_claim(source)


def test_a_served_window_draws_every_coin_from_its_entropy_and_none_from_the_os(monkeypatch):
    """The epoch's salt and draw coins and a live window's salt, round and draw coins, and every hm96 salt beside them,
    come from the challenger's named entropy alone: with the operating system's generators refusing, they are drawn,
    opened and checked against their commitments."""
    def refuse(*_a, **_k):
        raise AssertionError("a served coin read the operating system's randomness")
    for owner, name in ((os, "urandom"), (os, "getrandom"), (secrets, "token_bytes"), (random, "SystemRandom")):
        monkeypatch.setattr(owner, name, refuse)
    named = Ch.named(verifier.CLAIM, random.Random(3).randbytes)
    epoch = EC.Challenger(0, named)
    for label, after in (("salt", b"w" * 64), ("draw", b"t" * 64)):
        assert EC.coin_reason(epoch.commitment, epoch.open(label, after), label, after) is None
    live = Wd.live_challenger(0, 2, Ch.named(verifier.CLAIM, random.Random(4).randbytes))
    for label, count in Wd.live_families(2):
        rec = Ch.LiveRecord(live, label, clock=lambda: 0.0)
        roots = [bytes([i]) * 64 for i in range(count)]
        for root in roots:
            rec.record(root)
        assert Ch.order_reason(live.commitment, label, roots, rec.openings) is None


#: The PoUW and proof-system modules that read the operating system's randomness, each with why it isn't a coin, or
#: the claim instance that names it
OS_RANDOMNESS = {
    "pouw/circuit/anchors.py": "ncp-v2's window secret (`Ledger.open`'s `_fresh_source`, `secrets`), named by A5's "
                               "uniform/python-secrets",
    "sampled_proofs/one_stage/verifier.py": "a placeholder nonce in the JSON it writes to the Lean (`_dumps`), no coin",
}


def test_the_served_path_takes_no_os_randomness_for_its_coins():
    """No PoUW or proof-system module reads `os.urandom`, `os.getrandom`, `secrets` or `random.SystemRandom` but those
    `OS_RANDOMNESS` names: every coin comes from a challenger's named entropy, the Lean verifier of record's `coins` on
    the served path."""
    import ast
    root = Path(EC.__file__).resolve().parents[1]
    found = set()
    for path in [*(root / "pouw").rglob("*.py"), *(root / "sampled_proofs").rglob("*.py")]:
        rel = path.relative_to(root).as_posix()
        if "/tests/" in f"/{rel}" or "/.lake/" in f"/{rel}":
            continue
        for node in ast.walk(ast.parse(path.read_text(), rel)):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and (node.value.id, node.attr) in {("os", "urandom"), ("os", "getrandom"), ("random", "SystemRandom")}
                    or isinstance(node, ast.alias) and node.name.split(".")[0] == "secrets"
                    or isinstance(node, ast.ImportFrom) and node.module == "secrets"):
                found.add(rel)
    assert found == set(OS_RANDOMNESS), sorted(found ^ set(OS_RANDOMNESS))
