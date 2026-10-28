"""A block's key: under the seed's key alone, the writer of a block written after the codec seed picks its encoding and
holds nothing for it (the attack, against the reference); under the block's bound key, which binds its payload's
digest, that takes a search over SHA-512."""

from __future__ import annotations

import copy
import dataclasses
import hashlib

import pytest

from remote_bandwidth_fakes import Clock, Entropy
from verity.core.protocols.remote_bandwidth import audit, codec, coins, receipts
from verity.core.protocols.remote_bandwidth.audit import Params, Session
from verity.ml.commitments.rowleaf import sha512_row_v2_digest

SEED = hashlib.sha512(b"verity/remote-bandwidth/test-key-binding/v0\0").digest()[:codec.SEED_BYTES]
NODE = b"node-under-test"
WEIGHTS, KV = 0, 1
PICKED = bytes(codec.BLOCK_BYTES)
P512 = codec.Codec(codec.P512, 2, False)


class Memory:
    """A cheating prover's memory of encoded blocks: those it holds, and for any other the picked encoding, made from
    nothing."""

    def __init__(self, held: dict[int, bytes]) -> None:
        self.held = held

    def __getitem__(self, j: int) -> bytes:
        return self.held[j] if j in self.held else bytes(codec.BLOCK_BYTES)


def weights() -> bytes:
    return b"".join(hashlib.sha512(b"weights %d" % u).digest()[:63] + b"\0" for u in range(codec.BLOCK_BYTES // 64))


def one_decode(block: bytes, j: int, k: codec.Key, d: bytes) -> bytes:
    """The payload block `j`'s words give under the bound key of digest `d`, whatever that payload's own digest is: the
    writer's one decode."""
    f, b = k.codec.field, codec.binding(d, j, k)
    s = f.word_bytes
    return b"".join(codec.decode_word(int.from_bytes(block[s * w:s * (w + 1)], "little"), f.block_words * j + w, k, b)
                    .to_bytes(s, "little") for w in range(f.block_words))


def window(store: audit.Store, prover_store: audit.Store, passes: int = 2) -> tuple[audit.Verdict, list, list]:
    """One window of `passes` whole-store passes, every receipt drawn: the verifier's verdict on `prover_store`'s
    openings, the draws and the passes."""
    reg = store.registration
    p = Params(blocks=reg.blocks, deadline_ns=10, windows=1, samples=passes * reg.blocks, delta=1e-6, window_ns=10)
    clock = Clock()
    s = Session(p, reg, None, coins.local(reg.digest, 0, p.windows, Entropy(b"coins"), appends=1), not_before_ns=0,
                clock=clock)
    prover = audit.Prover(prover_store, None, s.commitment, entropy=Entropy(b"receipt salts"))
    challenge, previous, out = s.start(), reg.digest, []
    for i in range(passes):
        out.append(prover.run(challenge, i, previous))
        assert s.receive(0, i, out[-1].root, step=i) is None
        previous = out[-1].root
    clock.now = p.window_ns
    assert s.boundary() is None
    clock.now = s.deadline(0) + 1
    draws = s.close(0)
    return s.verify(0, [prover.open(out[i], q) for i, q in draws]), list(draws), out


def test_under_the_seeds_key_a_block_written_after_the_seed_takes_the_encoding_its_writer_picks() -> None:
    """The status quo on the encoded KV path: the codec seed is fixed when the store registers, and a KV block written
    later is encoded under the seed's key, from its address alone.  Its writer, a circuit that can learn the key, picks
    the block's encoding (all zeros) and writes the payload one decode of it gives.  The store then holds the picked
    encoding, every pass's sketch-sha512 receipt of the block is the receipt of zeros, and a prover that holds only the
    weights' block passes a window whose openings include the KV block's receipts.  The registration is built with its
    seed first, as a KV block's seed precedes its payload."""
    k = codec.key(SEED)
    kv = codec.decode(PICKED, KV, k)
    assert kv is not None and kv != PICKED and codec.encode(kv, KV, k) == PICKED
    payloads = [weights(), kv]
    salts = [Entropy(b"payload salts")(receipts.SALT_BYTES) for _ in payloads]
    root = audit.payload_tree(NODE, payloads, salts).commitment.root
    reg = audit.Registration(NODE, 2, root, SEED, codec.DEFAULT, receipts.DEFAULT, False, receipts.LAYOUT,
                             receipts.SKETCH_ROWS)
    store = audit.Store(reg, payloads, salts)
    assert store.blocks[KV] == PICKED

    cheater = copy.copy(store)
    cheater.payloads, cheater.blocks = None, Memory({WEIGHTS: store.blocks[WEIGHTS]})
    verdict, draws, passes = window(store, cheater)
    assert verdict == audit.Verdict(0, True)
    assert sum(passes[i].plan.block(q) == KV for i, q in draws) == len(passes)
    for ps in passes:
        assert ps.receipts[KV] == receipts.SketchReceipter(ps.seed)(KV, PICKED)


@pytest.mark.parametrize("c", [codec.DEFAULT, P512], ids=lambda c: c.name)
def test_a_bound_block_round_trips_under_the_digest_kept_beside_it(c) -> None:
    """The bound key's encoding decodes under the payload's digest, which is its payload leaf's x_j, and under nothing
    else: not the seed's key, another digest or another address.  One changed byte moves every word of the encoding,
    since the digest enters every word's key."""
    k, m = codec.key(SEED, c), weights()
    d = codec.digest(m)
    assert d == sha512_row_v2_digest(m, 8 * codec.BLOCK_BYTES)
    e = codec.encode(m, KV, k, bound=True)
    assert codec.decode(e, KV, k, d) == m
    assert e != codec.encode(m, KV, k) and codec.decode(e, KV, k) != m
    assert codec.decode(e, KV, k, codec.digest(PICKED)) is None and codec.decode(e, KV + 1, k, d) is None
    s = c.field.word_bytes
    changed = codec.encode(b"\1" + m[1:], KV, k, bound=True)
    assert all(e[s * w:s * (w + 1)] != changed[s * w:s * (w + 1)] for w in range(c.field.block_words))
    assert codec.decode((c.field.p + 1).to_bytes(s, "little") + e[s:], KV, k, d) is None
    with pytest.raises(ValueError, match="digest"):
        codec.decode(e, KV, k, d[:-1])
    with pytest.raises(ValueError, match="codec seed"):
        codec.binding(d, KV, dataclasses.replace(k, seed=None))


def test_an_append_s_coin_keys_a_bound_block_under_a_derivation_index_of_its_own() -> None:
    """The coin form: `K_j = H(seed ‖ j ‖ d_j ‖ c_a)`, the codec's framing at derivation index 2^32 − 4, then j, d_j
    and the coin, so no key of the form without a coin (2^32 − 3) is one of its keys.  Each coin gives another key,
    and a block decodes only under the coin it was encoded under.  A coin that isn't 32 bytes, a coin on a block under
    the seed's key, or one decoded without its digest is refused, and a codec's rounds stay below every derivation
    index."""
    k, m = codec.key(SEED, P512), weights()
    d, coin = codec.digest(m), hashlib.sha512(b"an append's coin").digest()[:codec.COIN_BYTES]
    message = (codec.DOMAIN + b"\4p512" + SEED + ((1 << 32) - 4).to_bytes(4, "big") + KV.to_bytes(8, "big") + d
               + coin + bytes(4))
    assert codec.binding(d, KV, k, coin) == int.from_bytes(hashlib.sha512(message).digest(), "little") % P512.field.p
    other = bytes(codec.COIN_BYTES)
    assert len({codec.binding(d, KV, k), codec.binding(d, KV, k, coin), codec.binding(d, KV, k, other)}) == 3
    e = codec.encode(m, KV, k, bound=True, coin=coin)
    assert codec.decode(e, KV, k, d, coin) == m
    assert codec.decode(e, KV, k, d) is None and codec.decode(e, KV, k, d, other) is None
    with pytest.raises(ValueError, match="coin is 32 bytes"):
        codec.binding(d, KV, k, coin[:-1])
    with pytest.raises(ValueError, match="bound block"):
        codec.encode(m, KV, k, coin=coin)
    with pytest.raises(ValueError, match="under its digest"):
        codec.decode(e, KV, k, coin=coin)
    assert codec.Codec(codec.P512, codec.COIN_INDEX - 1, False).rounds == (1 << 32) - 5
    with pytest.raises(ValueError, match="at least"):
        codec.Codec(codec.P512, codec.COIN_INDEX, False)


def test_under_the_bound_key_a_picked_encoding_has_no_payload_without_a_search() -> None:
    """The same move under the bound key.  For the picked encoding and any digest d the writer tries, one decode under
    d's key gives a payload whose own digest is not d (a try matches with chance about 2^-512), so `decode` refuses the
    pair, and writing that payload stores another encoding, whose receipts are not the zeros'.  The tries: the digests
    of zeros and of the seed's-key payload, then 32 steps of d ← digest(one decode under d), the search for a fixed
    point, then 30 others."""
    k = codec.key(SEED)
    tries = [codec.digest(PICKED), codec.digest(codec.decode(PICKED, KV, k))]
    for _ in range(32):
        tries.append(codec.digest(one_decode(PICKED, KV, k, tries[-1])))
    tries += [hashlib.sha512(b"try %d" % t).digest() for t in range(30)]
    assert len(set(tries)) == len(tries) == 64
    for d in tries:
        m = one_decode(PICKED, KV, k, d)
        assert codec.digest(m) != d and codec.decode(PICKED, KV, k, d) is None
    seed = receipts.seed(bytes(receipts.CHALLENGE_BYTES), 0, bytes(receipts.DIGEST_BYTES))
    for d in tries[:2]:
        stored = codec.encode(one_decode(PICKED, KV, k, d), KV, k, bound=True)
        assert stored != PICKED
        assert receipts.SketchReceipter(seed)(KV, stored) != receipts.SketchReceipter(seed)(KV, PICKED)
