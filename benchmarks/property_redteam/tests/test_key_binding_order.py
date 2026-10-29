"""KeyBinding's ideal step against the codec's odd symmetry (KB-5).

KB-1, a page appended after a pass's seed carrying that pass's receipts of other blocks, was once accepted by the
audit. Its regression test is poub's `test_append_order.py`, where the seed chains every append
received before the pass, so the page is rejected."""

from __future__ import annotations

import hashlib

from verity.core.protocols.poub import codec

P512 = codec.Codec(codec.P512, 2, False)


def test_unmasked_two_round_words_agree_up_to_sign_on_one_key_relation_and_the_mask_breaks_it() -> None:
    """KeyBinding's δ counts two words whose round inputs agree up to sign. Under p512/k2, unmasked, words (m, i, b)
    and (m', i', b') encode to y and p − y whenever t + t' ≡ −2·c_1 (t = τ·i + b) and m' ≡ −m − 2·c_0 + 2·c_1 (mod p),
    since ρ(p − x) = p − ρ(x): the event is that key relation, which under the seed's key (b = 0) is
    τ·(i + i') ≡ −2·c_1, and under bound keys needs offsets a writer gets only from SHA-512. One unit off the relation,
    the twin is unrelated; under the default (p1024, masked, k = 3) σ_K breaks the same construction."""
    seed = hashlib.sha512(b"verity/property-redteam/key-binding-sign/v0\0").digest()[:codec.SEED_BYTES]
    m, i, i2, b = 0x1234567 << 300, 3, 5, 12345
    for c, related in ((P512, True), (codec.DEFAULT, False)):
        k, p = codec.key(seed, c), c.field.p
        t2 = (-2 * k.c[1] - (k.tau * i + b)) % p
        b2 = (t2 - k.tau * i2) % p
        m2 = (-m - 2 * k.c[0] + 2 * k.c[1]) % p
        y = codec.encode_word(m, i, k, b)
        assert (codec.encode_word(m2, i2, k, b2) == p - y) is related
        assert codec.encode_word(m2, i2, k, b2 + 1) != p - y
