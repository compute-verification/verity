"""PoUS's session checked inside proofs (`pous.verifier.session_proofs`): the clear checks. A key the configuration doesn't
pin, a signature that doesn't verify over the message, an unknown scheme, a coin that doesn't open, a session over another
store than the registered one and an index past the store each reject."""

from __future__ import annotations

import dataclasses
import hashlib
import random

from verity.commitments import session_coins as SC
from verity.commitments.crypto import lms
from verity.protocols.accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.accounting.memory.pous.verifier import session_proofs as V

RNG = random.Random(3)
K, BLOCKS, BLOCK = 5, 64, 2056
PARAMS = lms.Params(64, 1, 2)
SCHEMES = {mc.LMS: mc.lms_verify(PARAMS)}
"""The deployed scheme's id at a two-level tree, so a test key signs quickly; the checks read only the id."""


def _rand(n: int) -> bytes:
    return bytes(RNG.getrandbits(8) for _ in range(n))


STORE = {"value": "store", "root": _rand(64).hex(), "leaves": BLOCKS}
KEY, OTHER = (lms.keygen(PARAMS, hashlib.sha512(b"session-proofs/" + w).digest()[:64]) for w in (b"pinned", b"other"))


def _pin(key: lms.PrivateKey) -> str:
    return hashlib.sha512(lms.public_key(key)).hexdigest()


def _sign(key: lms.PrivateKey, session: bytes, m: bytes) -> bytes:
    return lms.sign_at(key, 0, lms.memory_attester_message(key.I, session, m)).encode()


def _session(k: int = K, drawn: int = BLOCKS):
    """A session of `k` rounds over the registered `STORE`, its coins drawn over `drawn` blocks, its records, its salt and
    its message signed under `KEY`."""
    session = _rand(32)
    tree = mc.CoinTree.draw(b"source", session, drawn, k, key=mc.coin_key(_rand(mc.HM96.key_bytes)),
                            salts=[_rand(mc.HM96.salt_bytes) for _ in range(k)])
    c = mc.SessionCommitment(session, tree.root, tree.key, BLOCKS, k, 854_000, BLOCK, bytes.fromhex(STORE["root"]))
    rounds = [mc.Round(j, tree.indices[j], 10 * j, 10 * j + 7, _rand(64)) for j in range(k)]
    return c, [tree.opening(j) for j in range(k)], rounds, _rand(mc.RECORD_SALT_BYTES)


def _clear(c, openings, m=None, key=KEY, sig=None, scheme=mc.LMS, pinned=None, store=STORE, rounds=None, salt=None
           ) -> tuple[str, ...]:
    m = mc.message(c, salt, SC.COMPLETE, rounds) if m is None else m
    sig = _sign(key, c.session, m) if sig is None else sig
    return V.clear_reasons(c, openings, scheme, lms.public_key(key), m, sig, store=store,
                           pinned={_pin(KEY)} if pinned is None else pinned, schemes=SCHEMES)


def _unsalted(c, rounds) -> bytes:
    head = SC.header(c.session, c.coin_root, hashlib.sha512(c.coin_key).digest(), c.binding, c.blocks, c.k, c.limit_ns, c.block_bytes, SC.COMPLETE,
                     len(rounds))
    return hashlib.sha512(b"verity-pous/memory-challenger/session/v1\0" + len(head).to_bytes(8, "big") + head
                          + mc.records_digest(rounds)).digest()


def test_the_clear_checks_reject() -> None:
    """A session's pinned key, its scheme, its signed message and its openings pass; a key the configuration doesn't pin
    (its own signature verifying), an unknown scheme and a round whose coin doesn't open each reject."""
    c, openings, rounds, salt = _session()
    clear = lambda o=openings, **kw: _clear(c, o, rounds=rounds, salt=salt, **kw)  # noqa: E731
    assert clear() == ()
    assert clear(key=OTHER) == (V.UNPINNED,)
    assert clear(pinned=set()) == (V.UNPINNED,)
    assert clear(scheme="test/x") == ("signature scheme 'test/x' is not accepted",)
    bad = openings[:2] + [dataclasses.replace(openings[2], index=(openings[2].index + 1) % BLOCKS)] + openings[3:]
    assert "round 2's index is not its committed coin" in clear(bad)


def test_a_message_the_signature_does_not_sign_is_refused() -> None:
    """The signature is the attester's over the salted message of its own records: the message of records that differ
    in one word, under another salt or without a salt (the unsalted tag), each fails to verify by name."""
    c, openings, rounds, salt = _session()
    sig = _sign(KEY, c.session, mc.message(c, salt, SC.COMPLETE, rounds))
    word = dataclasses.replace(rounds[1], digest=bytes([rounds[1].digest[0] ^ 1]) + rounds[1].digest[1:])
    for m in (mc.message(c, salt, SC.COMPLETE, [rounds[0], word, *rounds[2:]]),
              mc.message(c, bytes([salt[0] ^ 1]) + salt[1:], SC.COMPLETE, rounds), _unsalted(c, rounds)):
        assert _clear(c, openings, m=m, sig=sig) == (V.BAD_SIGNATURE,)
    assert _clear(c, openings, m=mc.message(c, salt, SC.COMPLETE, rounds), sig=sig[:-1]) == (V.BAD_SIGNATURE,)


def test_a_session_over_another_store_is_refused() -> None:
    """The store the session's openings pin is the registered one, whose root the setup published: a session whose
    signed binding is another root, or whose block count is not the store's rows, is refused by name."""
    c, openings, rounds, salt = _session()
    other = {**STORE, "root": _rand(64).hex()}
    (why,) = _clear(c, openings, rounds=rounds, salt=salt, store=other)
    assert why.startswith(V.OTHER_STORE) and "is not the store's root" in why
    (why,) = _clear(c, openings, rounds=rounds, salt=salt, store={**STORE, "leaves": BLOCKS + 1})
    assert why.startswith(V.OTHER_STORE) and f"it audits {BLOCKS} blocks, the store has {BLOCKS + 1}" in why


def test_an_index_past_the_store_is_refused() -> None:
    """The store's opening returns a verdict of 1 for a key past its rows, so the index bound is a clear check: coins
    that open, drawn over twice the store's blocks, are refused by name at each round past it."""
    c, openings, rounds, salt = _session(k=16, drawn=2 * BLOCKS)
    past = [o.j for o in openings if o.index >= BLOCKS]
    assert past and _clear(c, openings, rounds=rounds, salt=salt) == tuple(
        f"{V.PAST_STORE}round {j}'s index {openings[j].index} is not below the store's {BLOCKS} blocks" for j in past)
