"""The bandwidth audit: its tags, one window end to end over a real P2 store, and every way a window is rejected."""

from __future__ import annotations

import hashlib
import itertools
import math
import struct
from dataclasses import replace
from fractions import Fraction

import pytest

import bandwidth_vectors as bv
from verity.core.protocols.bandwidth import audit, tags
from verity.core.protocols.pous import protocol, schemes
from verity.core.protocols.pous.schemes.p2 import ROOT_CALL_NS, P2Params
from verity.ml.crypto import blake3
from verity.ml.randomness import frame

TAGS = sorted(tags.TAGS)
P = tags.MERSENNE31


def _ticks(step: int):
    counter = itertools.count(0, step)
    return lambda: next(counter)


def _params(tag: str, samples: int = 8, deadline_ns: int = 1_000_000) -> audit.Params:
    return audit.Params(tag, samples, deadline_ns, rtt_allowance_ns=100_000)


def _run(store: audit.Store, vk, params: audit.Params, *, step: int = 1_000, window: int = 0,
         prover=None) -> tuple[audit.Verdict, tuple[int, ...]]:
    """One window, the clock advancing `step` ns per reading; `prover(seed) -> TagCommitment` defaults to the honest
    pass."""
    verifier = audit.Verifier(vk, params, b"verifier secret", clock=_ticks(step))
    w = verifier.window(window)
    seed = w.start()
    commitment = (prover or (lambda s: audit.tag_pass(store, params.tag, s)))(seed)
    drawn = w.commit(commitment.root)
    return w.verdict([commitment.open(i) for i in drawn]), drawn


@pytest.fixture(scope="module")
def p2_store():
    """A real P2 store: W encoded under the setup's salt, and the verifier's `vk` from the public-encoder setup."""
    sch = schemes.P2(P2Params(128, 173, 8, 2))
    w = bytes(range(256))[: sch.params.w_bytes]
    com = protocol.commit(sch, w)
    salt = protocol.draw_salt(b"setup secret", com)
    encoded = protocol.encode(sch, w, salt)
    vk = protocol.verifier_key(sch, com, salt, mode=protocol.PUBLIC, encoded=encoded)
    store = audit.Store.from_segments(sch.params, salt, encoded)
    assert store.root == vk.root
    return store, vk


@pytest.fixture(scope="module")
def big_store():
    """37 blocks of 2,056 bytes, P2's width: not a power of two, and a short last leaf for the polynomial tag."""
    store = bv.store(37)
    return store, bv.vk_of(store)


# --- tags ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", TAGS)
def test_prefix_is_one_sha512_block(name: str) -> None:
    for domain in (tags.TAG_DOMAIN, tags.LEAF_DOMAIN):
        assert len(tags.prefix(domain, name, bv.SEED)) == 128
    with pytest.raises(ValueError):
        tags.prefix(tags.TAG_DOMAIN, name, b"short")


def test_random_oracle_tag_is_its_formula() -> None:
    block = bv.block(5)
    want = hashlib.sha512(tags.prefix(tags.TAG_DOMAIN, "ro-sha512/v1", bv.SEED) + (5).to_bytes(8, "big") + block)
    assert tags.get("ro-sha512/v1").keyed(bv.SEED)(5, block) == want.digest()


def test_blake3_tag_is_its_formula() -> None:
    block = bv.block(5)
    key = hashlib.sha512(tags.prefix(tags.TAG_DOMAIN, "ro-blake3/v1", bv.SEED)).digest()[:32]
    assert tags.get("ro-blake3/v1").key(bv.SEED) == key
    assert tags.get("ro-blake3/v1").keyed(bv.SEED)(5, block) == blake3.keyed(key, (5).to_bytes(8, "big") + block)


@pytest.mark.parametrize("name", TAGS)
def test_every_tag_is_seed_first(name: str) -> None:
    """`H(σ, j, E_j)`: σ is framed inside the first 128 bytes the tag hashes (the SHA-512 prefix, or the input of the
    BLAKE3 key every compression starts from), ahead of `j` and the block, and the tag changes with σ alone."""
    head = tags.prefix(tags.TAG_DOMAIN, name, bv.SEED)
    assert len(head) == 128 and frame(bv.SEED) in head and head.index(frame(bv.SEED)) + len(frame(bv.SEED)) <= 128
    block, other = bv.block(2), bytes(range(32, 64))
    assert tags.get(name).keyed(bv.SEED)(2, block) != tags.get(name).keyed(other)(2, block)


def test_only_the_random_oracle_tags_are_certified() -> None:
    assert {n for n in TAGS if tags.get(n).certified} == {"ro-blake3/v1", "ro-sha512/v1"}


def test_polynomial_tag_is_the_polynomial_at_the_block_point() -> None:
    t = tags.get("gf31-poly/v1")
    block = bv.block(3)
    words = [w for (w,) in struct.iter_unpack("<I", block)]
    g = t.point(bv.SEED, 3)
    assert 2 <= g <= P - 1
    want = sum(w % P * pow(g, len(words) - 1 - i, P) for i, w in enumerate(words)) % P
    assert t.keyed(bv.SEED)(3, block) == want.to_bytes(4, "big")


def test_points_reject_values_at_or_past_p_minus_two() -> None:
    """With a modulus near 2^30 about half the 31-bit values are rejected: the point is 2 plus the first value below
    p − 2, in counter order, and some block's first value is rejected."""
    p = (1 << 30) + 3
    head = hashlib.sha512(tags.prefix(tags.TAG_DOMAIN, "gf31-poly/v1", bv.SEED))
    first_rejected = 0
    for j in range(16):
        d = hashlib.sha512(tags.prefix(tags.TAG_DOMAIN, "gf31-poly/v1", bv.SEED) + j.to_bytes(8, "big")
                           + (0).to_bytes(4, "big")).digest()
        values = [int.from_bytes(d[k:k + 4], "big") >> 1 for k in range(0, 64, 4)]
        first_rejected += values[0] >= p - 2
        assert tags._point(head, p, j) == 2 + next(x for x in values if x < p - 2)
    assert first_rejected


def test_points_differ_per_block_and_per_seed() -> None:
    t = tags.get("gf31-poly/v1")
    assert len({t.point(bv.SEED, j) for j in range(64)}) == 64
    assert t.point(bv.SEED, 0) != t.point(bytes(32), 0)


def test_polynomial_tag_binds_words_only_mod_p() -> None:
    """The word map is not injective: w and w + p agree, so the tag binds ⌈W·log2 p⌉ bits of a block, not all of it."""
    t = tags.get("gf31-poly/v1")
    a = bytearray(bv.block(9))
    w = int.from_bytes(a[:4], "little") % P
    b = bytearray(a)
    b[:4] = (w + P).to_bytes(4, "little")
    a[:4] = w.to_bytes(4, "little")
    keyed = t.keyed(bv.SEED)
    assert keyed(9, bytes(a)) == keyed(9, bytes(b))
    assert t.info_bits(2056) == 15_934 < 8 * 2056
    assert t.guess(2056) == Fraction(513, P - 2)


def test_tags_refuse_blocks_they_do_not_define() -> None:
    with pytest.raises(ValueError):
        tags.get("gf31-poly/v1").check_block(2054)
    with pytest.raises(ValueError):
        tags.get("nope/v1")
    assert tags.get("ro-sha512/v1").info_bits(2056) == 16_448


@pytest.mark.parametrize("name", TAGS)
def test_tag_tree_packs_a_leaf_of_tags(name: str) -> None:
    t = tags.get(name)
    ys = [t.keyed(bv.SEED)(j, bv.block(j, 16)) for j in range(37)]
    leaves = tags.leaves(t, bv.SEED, ys)
    per = tags.tags_per_leaf(t)
    assert per == t.per_leaf and len(leaves) == -(-37 // per)
    head = tags.prefix(tags.LEAF_DOMAIN, name, bv.SEED)
    last = len(leaves) - 1
    assert leaves[last] == hashlib.sha512(head + last.to_bytes(8, "big") + b"".join(ys[last * per:])).digest()


# --- one window ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", TAGS)
def test_honest_window_over_a_p2_store_is_accepted(p2_store, name: str) -> None:
    store, vk = p2_store
    verdict, drawn = _run(store, vk, _params(name))
    assert verdict.accepted and verdict.failure is None and len(drawn) == 8
    assert verdict.public() == {"accepted": True}


@pytest.mark.parametrize("name", TAGS)
def test_honest_window_at_p2_width_is_accepted(big_store, name: str) -> None:
    store, vk = big_store
    verdict, drawn = _run(store, vk, _params(name, samples=37))
    assert verdict.accepted and len(set(drawn)) > 1


def test_last_leaf_opens_short(big_store) -> None:
    store, _ = big_store
    seed = bytes(32)
    commitment = audit.tag_pass(store, "gf31-poly/v1", seed)
    assert len(commitment.open(36).leaf_tags) == 5 and len(commitment.open(0).leaf_tags) == 16


def test_samples_come_after_the_commitment_from_the_window_key(big_store) -> None:
    store, vk = big_store
    params = _params("ro-sha512/v1")
    _, drawn = _run(store, vk, params, window=3)
    key = audit.window_key(b"verifier secret", vk, 3)
    assert drawn == audit.samples(key, 37, 8)
    assert audit.seed(key) != audit.seed(audit.window_key(b"verifier secret", vk, 4))


def test_late_commitment_is_rejected_and_draws_nothing(big_store) -> None:
    store, vk = big_store
    params = _params("gf31-poly/v1")
    verdict, drawn = _run(store, vk, params, step=params.limit_ns + 1)
    assert drawn == () and not verdict.accepted and verdict.failure == "late"
    assert verdict.elapsed_ns == params.limit_ns + 1
    on_time, _ = _run(store, vk, params, step=params.limit_ns)
    assert on_time.accepted


def test_malformed_root_is_rejected(big_store) -> None:
    _, vk = big_store
    w = audit.Verifier(vk, _params("ro-sha512/v1"), b"s", clock=_ticks(1)).window(0)
    w.start()
    assert w.commit(b"\0" * 63) == () and w.verdict().failure == "malformed"


class _Corrupt(audit.TagCommitment):
    """A prover whose tags are wrong on the blocks `bad` names (it skipped them and guessed)."""

    def __init__(self, store, tag, seed, bad):
        honest = audit.tag_pass(store, tag, seed)
        ys = [bytes(len(y)) if bad(j) else y for j, y in enumerate(honest.ys)]
        super().__init__(store, tag, seed, ys)


@pytest.mark.parametrize("name", TAGS)
def test_guessed_tags_are_caught(big_store, name: str) -> None:
    store, vk = big_store
    verdict, _ = _run(store, vk, _params(name), prover=lambda s: _Corrupt(store, name, s, lambda j: True))
    assert not verdict.accepted and verdict.failure == "tag-mismatch"
    assert verdict.public() == {"accepted": False}


def test_a_skipped_fraction_passes_only_when_no_sample_lands_in_it(big_store) -> None:
    """Skipping blocks `0..k-1` passes exactly the windows whose samples all miss them."""
    store, vk = big_store
    params = _params("gf31-poly/v1", samples=4)
    outcomes = set()
    for window in range(12):
        verdict, drawn = _run(store, vk, params, window=window,
                              prover=lambda s: _Corrupt(store, params.tag, s, lambda j: j < 9))
        assert verdict.accepted == all(i >= 9 for i in drawn)
        outcomes.add(verdict.accepted)
    assert outcomes == {True, False}


def test_wrong_block_fails_the_store_path(big_store) -> None:
    store, vk = big_store
    params = _params("ro-sha512/v1")
    w = audit.Verifier(vk, params, b"s", clock=_ticks(1)).window(0)
    c = audit.tag_pass(store, params.tag, w.start())
    drawn = w.commit(c.root)
    openings = [c.open(i) for i in drawn]
    openings[0] = replace(openings[0], block=bytes(2056))
    assert w.verdict(openings).failure == "store-path"


def test_wrong_tag_path_fails(big_store) -> None:
    store, vk = big_store
    params = _params("gf31-poly/v1")
    w = audit.Verifier(vk, params, b"s", clock=_ticks(1)).window(0)
    c = audit.tag_pass(store, params.tag, w.start())
    drawn = w.commit(c.root)
    openings = [c.open(i) for i in drawn]
    openings[0] = replace(openings[0], tag_path=(bytes(64),) * len(openings[0].tag_path))
    assert w.verdict(openings).failure == "tag-path"


def test_openings_must_answer_the_samples_in_order(big_store) -> None:
    store, vk = big_store
    params = _params("ro-sha512/v1")
    for mutate in (lambda o: o[1:], lambda o: o[::-1] if o[0].index != o[-1].index else o[1:],
                   lambda o: [replace(o[0], leaf_tags=o[0].leaf_tags * 2), *o[1:]]):
        w = audit.Verifier(vk, params, b"s", clock=_ticks(1)).window(0)
        c = audit.tag_pass(store, params.tag, w.start())
        drawn = w.commit(c.root)
        assert w.verdict(mutate([c.open(i) for i in drawn])).failure == "malformed"


def test_a_window_runs_once(big_store) -> None:
    store, vk = big_store
    verifier = audit.Verifier(vk, _params("ro-sha512/v1"), b"s", clock=_ticks(1))
    w = verifier.window(0)
    with pytest.raises(RuntimeError):
        verifier.window(0)
    with pytest.raises(RuntimeError):
        w.commit(bytes(64))
    with pytest.raises(RuntimeError):
        w.verdict()
    c = audit.tag_pass(store, "ro-sha512/v1", w.start())
    with pytest.raises(RuntimeError):
        w.start()
    drawn = w.commit(c.root)
    with pytest.raises(RuntimeError):
        w.commit(c.root)
    first = w.verdict([c.open(i) for i in drawn])
    assert first.accepted and w.verdict([]) is first


def test_params_refuse_a_window_that_fits_a_root() -> None:
    audit.Params("ro-sha512/v1", 1, 2 * ROOT_CALL_NS - 1)
    with pytest.raises(ValueError, match="rebuild floor"):
        audit.Params("ro-sha512/v1", 1, 2 * ROOT_CALL_NS)
    with pytest.raises(ValueError, match="rebuild floor"):
        audit.Params("ro-sha512/v1", 1, 1_000_000, rtt_allowance_ns=ROOT_CALL_NS * 2 - 1_000_000)
    with pytest.raises(ValueError):
        audit.Params("ro-sha512/v1", 0, 1)


def test_verifier_refuses_a_tag_the_store_cannot_carry() -> None:
    store = bv.store(4, 18)
    with pytest.raises(ValueError):
        audit.Verifier(bv.vk_of(store), _params("gf31-poly/v1"), b"s")


# --- the bound -------------------------------------------------------------------------------------------------------


def test_the_deployed_samples_are_the_proofs() -> None:
    """`s = 1,849` is the fewest samples whose sampling term `((k − 1)/N)^s` is at most `2^-42` at `ε_s = 1/64`."""
    blocks = 1 << 19
    k = audit.certified_blocks(blocks)
    assert k == 516_096 == math.ceil(Fraction(63, 64) * blocks)
    assert audit.samples_for(blocks, -42) == audit.SAMPLES == 1849
    assert ((k - 1) / blocks) ** 1849 <= 2.0 ** -42 < ((k - 1) / blocks) ** 1848


@pytest.mark.parametrize("name", ["ro-blake3/v1", "ro-sha512/v1"])
def test_bound_is_the_guarantee_at_the_proofs_point(name: str) -> None:
    params = audit.Params(name, audit.SAMPLES, 1_000_000, rtt_allowance_ns=100_000)
    b = audit.bound(params, 1 << 19, 2056, chip_bytes=208_000_000, side_bytes_per_s=63e9)
    k = 516_096
    assert b.side_bytes == 69_300_000 and b.slack_bits == 40 * k + 41
    assert b.ingress_bytes == (k * 2056 * 8 - 8 * 208_000_000 - 8 * 69_300_000 - (40 * k + 41)) // 8
    assert b.error_compression_log2 == -41 and b.error_guess <= 1849 * 2.0 ** -256
    assert b.error < 2.0 ** -40 and 0.7e9 < b.ingress_bytes < (1 << 19) * 2056


def test_the_fast_check_has_no_bound() -> None:
    with pytest.raises(ValueError, match="uncertified"):
        audit.bound(audit.Params("gf31-poly/v1", audit.SAMPLES, 1_000_000), 1 << 19, 2056, chip_bytes=0,
                    side_bytes_per_s=0)
