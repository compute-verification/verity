"""The bandwidth audit: its tags, one window end to end over a real P2 store, and every way a window is rejected."""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import shutil
import struct
from dataclasses import dataclass, replace
from fractions import Fraction

import pytest

import bandwidth_vectors as bv
from verity.core.protocols.bandwidth import audit, tags
from verity.core.protocols.pous import protocol, schemes
from verity.core.protocols.pous.schemes.p2 import ROOT_CALL_NS, P2Params
from verity.core.protocols.pous.verifier import VerifierKey
from verity.ml.crypto import blake3
from verity.ml.randomness import frame

TAGS = sorted(tags.TAGS)
P = tags.MERSENNE31


def _ticks(step: int):
    counter = itertools.count(0, step)
    return lambda: next(counter)


class _Clock:
    """A clock the test sets: every reading is `now`."""

    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


def _nonces(label: bytes = b""):
    """Deterministic nonces for a test, distinct for each label and draw: the verifier's own randomness, replayed."""
    counter = itertools.count()
    return lambda n: hashlib.sha512(b"test nonce " + label + next(counter).to_bytes(8, "big")).digest()[:n]


def _params(tag: str, samples: int = 8, deadline_ns: int = 1_000_000) -> audit.Params:
    return audit.Params(tag, samples, deadline_ns, rtt_allowance_ns=100_000)


@pytest.fixture
def state_of(tmp_path):
    """`state_of(vk)`: a new store's challenger state, in a state directory of its own, with test nonces; closed after
    the test."""
    opened = []

    def make(vk, nonces=None):
        states = tmp_path / f"states-{len(opened)}"
        states.mkdir()
        audit.ChallengerState.create(states, vk)
        opened.append(audit.ChallengerState(states, vk, nonces=nonces or _nonces(states.name.encode())))
        return opened[-1]

    yield make
    for s in opened:
        s.close()


@dataclass(frozen=True)
class _Setup:
    """A device's store and its `vk`, with what the verifier knows of its salt: the device's id, the commitment to `W`
    and the source the salt was sampled from.  Unpacks as `(store, vk)`."""

    store: audit.Store
    vk: VerifierKey
    device: bytes
    commitment: bytes
    salt_source: bytes

    def __iter__(self):
        return iter((self.store, self.vk))

    def verifier(self, params: audit.Params, state, *, source: bytes = b"s", clock=None) -> audit.Verifier:
        return audit.Verifier(self.vk, params, source, state, device=self.device, commitment=self.commitment,
                              salt_source=self.salt_source, clock=clock or _ticks(1))


def _run(setup: _Setup, params: audit.Params, state, *, step: int = 1_000,
         prover=None) -> tuple[audit.Verdict, tuple[int, ...]]:
    """One window, the clock advancing `step` ns per reading; `prover(seed) -> TagCommitment` defaults to the honest
    pass."""
    w = setup.verifier(params, state, source=b"verifier secret", clock=_ticks(step)).window()
    seed = w.start()
    commitment = (prover or (lambda s: audit.tag_pass(setup.store, params.tag, s)))(seed)
    drawn = w.commit(commitment.root)
    return w.verdict([commitment.open(i) for i in drawn]), drawn


def _p2_setup(device: bytes, salt=None) -> _Setup:
    """A real P2 store: one W encoded under `device`'s salt (or `salt(source, com)`), and the verifier's `vk` from the
    public-encoder setup."""
    sch = schemes.P2(P2Params(128, 173, 8, 2))
    w = bytes(range(256))[: sch.params.w_bytes]
    com, source = protocol.commit(sch, w), b"setup secret"
    salt = salt(source, com) if salt else audit.draw_salt(source, com, device)
    encoded = protocol.encode(sch, w, salt)
    vk = protocol.verifier_key(sch, com, salt, mode=protocol.PUBLIC, encoded=encoded)
    store = audit.Store.from_segments(sch.params, salt, encoded)
    assert store.root == vk.root
    return _Setup(store, vk, device, com, source)


@pytest.fixture(scope="module")
def p2_store():
    return _p2_setup(bv.DEVICE)


@pytest.fixture(scope="module")
def big_store():
    """37 labels of 2,056 bytes, P2's width: not a power of two, and a short last leaf for the polynomial tag."""
    store = bv.store(37)
    return _Setup(store, bv.vk_of(store), bv.DEVICE, bv.COMMITMENT, bv.SOURCE)


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
    BLAKE3 key every compression starts from), ahead of `j` and the label, and the tag changes with σ alone."""
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
    p − 2, in counter order, and some label's first value is rejected."""
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
    """The word map is not injective: w and w + p agree, so the tag binds ⌈W·log2 p⌉ bits of a label, not all of it."""
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
def test_honest_window_over_a_p2_store_is_accepted(p2_store, state_of, name: str) -> None:
    verdict, drawn = _run(p2_store, _params(name), state_of(p2_store.vk))
    assert verdict.accepted and verdict.failure is None and len(drawn) == 8
    assert verdict.public() == {"accepted": True}


@pytest.mark.parametrize("name", TAGS)
def test_honest_window_at_p2_width_is_accepted(big_store, state_of, name: str) -> None:
    verdict, drawn = _run(big_store, _params(name, samples=37), state_of(big_store.vk))
    assert verdict.accepted and len(set(drawn)) > 1


def test_last_leaf_opens_short(big_store) -> None:
    store, _ = big_store
    seed = bytes(32)
    commitment = audit.tag_pass(store, "gf31-poly/v1", seed)
    assert len(commitment.open(36).leaf_tags) == 5 and len(commitment.open(0).leaf_tags) == 16


def test_samples_come_after_the_commitment_from_the_window_key(big_store, state_of) -> None:
    """The window's key binds its tag and the nonce the verifier sampled for it, and changes with either."""
    vk = big_store.vk
    params = _params("ro-sha512/v1")
    nonce = bytes(range(32))
    _, drawn = _run(big_store, params, state_of(vk, nonces=lambda n: nonce))
    key = audit.window_key(b"verifier secret", vk, params.tag, nonce)
    assert drawn == audit.samples(key, 37, 8)
    other_tag = audit.window_key(b"verifier secret", vk, "gf31-poly/v1", nonce)
    other_nonce = audit.window_key(b"verifier secret", vk, params.tag, bytes(32))
    assert len({audit.seed(k) for k in (key, other_tag, other_nonce)}) == 3
    for bad in (bytes(31), bytes(33), bytearray(32), 0):
        with pytest.raises(ValueError):
            audit.window_key(b"verifier secret", vk, params.tag, bad)


def test_late_commitment_is_rejected_and_draws_nothing(big_store, state_of) -> None:
    params = _params("gf31-poly/v1")
    state = state_of(big_store.vk)
    verdict, drawn = _run(big_store, params, state, step=params.limit_ns + 1)
    assert drawn == () and not verdict.accepted and verdict.failure == "late"
    assert verdict.elapsed_ns == params.limit_ns + 1
    on_time, _ = _run(big_store, params, state, step=params.limit_ns)
    assert on_time.accepted


def test_malformed_root_is_rejected(big_store, state_of) -> None:
    w = big_store.verifier(_params("ro-sha512/v1"), state_of(big_store.vk)).window()
    w.start()
    assert w.commit(b"\0" * 63) == () and w.verdict().failure == "malformed"


class _Corrupt(audit.TagCommitment):
    """A prover whose tags are wrong on the labels `bad` names (it skipped them and guessed)."""

    def __init__(self, store, tag, seed, bad):
        honest = audit.tag_pass(store, tag, seed)
        ys = [bytes(len(y)) if bad(j) else y for j, y in enumerate(honest.ys)]
        super().__init__(store, tag, seed, ys)


@pytest.mark.parametrize("name", TAGS)
def test_guessed_tags_are_caught(big_store, state_of, name: str) -> None:
    store, vk = big_store
    verdict, _ = _run(big_store, _params(name), state_of(vk), prover=lambda s: _Corrupt(store, name, s, lambda j: True))
    assert not verdict.accepted and verdict.failure == "tag-mismatch"
    assert verdict.public() == {"accepted": False}


def test_a_skipped_fraction_passes_only_when_no_sample_lands_in_it(big_store, state_of) -> None:
    """Skipping labels `0..k-1` passes exactly the windows whose samples all miss them."""
    store, vk = big_store
    params = _params("gf31-poly/v1", samples=4)
    outcomes = set()
    state = state_of(vk)
    for _ in range(12):
        verdict, drawn = _run(big_store, params, state,
                              prover=lambda s: _Corrupt(store, params.tag, s, lambda j: j < 9))
        assert verdict.accepted == all(i >= 9 for i in drawn)
        outcomes.add(verdict.accepted)
    assert outcomes == {True, False}


def test_wrong_block_fails_the_store_path(big_store, state_of) -> None:
    store, vk = big_store
    params = _params("ro-sha512/v1")
    w = big_store.verifier(params, state_of(vk)).window()
    c = audit.tag_pass(store, params.tag, w.start())
    drawn = w.commit(c.root)
    openings = [c.open(i) for i in drawn]
    openings[0] = replace(openings[0], block=bytes(2056))
    assert w.verdict(openings).failure == "store-path"


def test_wrong_tag_path_fails(big_store, state_of) -> None:
    store, vk = big_store
    params = _params("gf31-poly/v1")
    w = big_store.verifier(params, state_of(vk)).window()
    c = audit.tag_pass(store, params.tag, w.start())
    drawn = w.commit(c.root)
    openings = [c.open(i) for i in drawn]
    openings[0] = replace(openings[0], tag_path=(bytes(64),) * len(openings[0].tag_path))
    assert w.verdict(openings).failure == "tag-path"


def test_openings_must_answer_the_samples_in_order(big_store, state_of) -> None:
    store, vk = big_store
    params = _params("ro-sha512/v1")
    state = state_of(vk)
    for mutate in (lambda o: o[1:], lambda o: o[::-1] if o[0].index != o[-1].index else o[1:],
                   lambda o: [replace(o[0], leaf_tags=o[0].leaf_tags * 2), *o[1:]]):
        w = big_store.verifier(params, state).window()
        c = audit.tag_pass(store, params.tag, w.start())
        drawn = w.commit(c.root)
        assert w.verdict(mutate([c.open(i) for i in drawn])).failure == "malformed"


def test_a_window_runs_once(big_store, state_of) -> None:
    store, vk = big_store
    verifier = big_store.verifier(_params("ro-sha512/v1"), state_of(vk))
    w = verifier.window()
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


def test_one_window_over_a_store_is_open_at_a_time(big_store, state_of) -> None:
    """While one window over a store is open the verifier refuses to start another, which stays unstarted and starts
    once the first has its commitment: one read of each label never answers two windows."""
    store, vk = big_store
    params = _params("ro-blake3/v1")
    clock = _Clock()
    verifier = big_store.verifier(params, state_of(vk), clock=clock)
    first, second = verifier.window(), verifier.window()
    c1 = audit.tag_pass(store, params.tag, first.start())
    with pytest.raises(RuntimeError, match="open"):
        second.start()
    clock.now = params.limit_ns
    with pytest.raises(RuntimeError, match="open"):
        second.start()
    drawn = first.commit(c1.root)
    assert first.verdict([c1.open(i) for i in drawn]).accepted
    c2 = audit.tag_pass(store, params.tag, second.start())
    clock.now += params.limit_ns
    assert second.verdict([c2.open(i) for i in second.commit(c2.root)]).accepted
    third = verifier.window()
    third.start()
    third.commit(b"short")
    fourth = verifier.window()
    fourth.open()
    with pytest.raises(RuntimeError, match="open"):
        fourth.open()
    clock.now += 10 * params.limit_ns
    with pytest.raises(RuntimeError, match="open"):
        verifier.window().open()
    c4 = audit.tag_pass(store, params.tag, fourth.start())
    assert fourth.verdict([c4.open(i) for i in fourth.commit(c4.root)]).accepted
    verifier.window().start()


def test_one_devices_commitment_fails_anothers_window(p2_store, state_of) -> None:
    """Two devices holding one `W` under one setup secret hold different stores, seeds and tags: device a's
    commitment, or its store tagged under b's seed, fails b's window, while each device passes its own, both open."""
    store_a, vk_a = p2_store
    b = _p2_setup(audit.device_id(bv.DEVICES[1]))
    store_b, vk_b = b
    assert store_a.salt != store_b.salt and store_a.blocks != store_b.blocks and vk_a.root != vk_b.root
    params = _params("ro-blake3/v1")
    va, vb = (s.verifier(params, state_of(s.vk), source=b"window secret") for s in (p2_store, b))
    win_a, win_b = va.window(), vb.window()
    seed_a, seed_b = win_a.start(), win_b.start()
    assert seed_a != seed_b
    ca = audit.tag_pass(store_a, params.tag, seed_a)
    assert win_a.verdict([ca.open(i) for i in win_a.commit(ca.root)]).accepted
    assert win_b.verdict([ca.open(i) for i in win_b.commit(ca.root)]).failure == "store-path"
    win_b = vb.window()
    forged = audit.tag_pass(store_a, params.tag, win_b.start())
    assert win_b.verdict([forged.open(i) for i in win_b.commit(forged.root)]).failure == "store-path"
    win_b = vb.window()
    cb = audit.tag_pass(store_b, params.tag, win_b.start())
    assert win_b.verdict([cb.open(i) for i in win_b.commit(cb.root)]).accepted


def test_a_store_not_salted_for_its_device_is_refused(p2_store, state_of) -> None:
    """The verifier takes only a store whose salt is `draw_salt`'s for its device: one set up under PoUS's own salt,
    another device's store, or a store salted under another commitment or source is refused before any window."""
    params = _params("ro-blake3/v1")
    pous_salted = _p2_setup(bv.DEVICE, salt=protocol.draw_salt)
    assert pous_salted.vk.salt == protocol.draw_salt(pous_salted.salt_source, pous_salted.commitment)
    other = audit.device_id(bv.DEVICES[1])
    for setup in (pous_salted, replace(p2_store, device=other), replace(p2_store, commitment=bytes(64)),
                  replace(p2_store, salt_source=b"another secret")):
        with pytest.raises(ValueError, match="salt"):
            setup.verifier(params, state_of(setup.vk))
    for device in (bv.DEVICES[0], bv.DEVICE.hex(), bv.DEVICE[:15]):
        with pytest.raises(ValueError, match="UUID"):
            replace(p2_store, device=device).verifier(params, state_of(p2_store.vk))
    assert p2_store.verifier(params, state_of(p2_store.vk)).window()


def test_a_device_has_one_id_its_gpus_uuid() -> None:
    """A device's id is its GPU's 16-byte UUID, read from nvidia-smi's one spelling; any other spelling, and any
    id of another form, is refused, so two spellings of one GPU never give two salts."""
    uuid = "GPU-7e28035c-3c3b-b7aa-d655-0dad098bab62"
    device = audit.device_id(uuid)
    assert device == bytes.fromhex("7e28035c3c3bb7aad6550dad098bab62") and len(device) == audit.DEVICE_BYTES
    for other in (uuid.upper(), "GPU-" + uuid[4:].upper(), uuid[4:], "GPU-7e28035c3c3bb7aad6550dad098bab62",
                  "{" + uuid + "}", uuid + "\n", " " + uuid, "MIG-" + uuid[4:], uuid[:-1], uuid + "0", uuid.encode(), device):
        with pytest.raises(ValueError):
            audit.device_id(other)
    com = bytes(64)
    assert audit.draw_salt(b"s", com, device) != audit.draw_salt(b"s", com, audit.device_id(bv.DEVICES[0]))
    for other in (uuid, device.hex(), device[:15], device + b"\0", bytearray(device), None):
        with pytest.raises(ValueError):
            audit.draw_salt(b"s", com, other)


def test_the_challenger_reads_a_devices_id_from_the_driver() -> None:
    """`read_device` asks nvidia-smi for the GPU's UUID and MIG mode and reads the UUID through `device_id`: a whole GPU
    (MIG off, or without MIG) gives its 16 bytes; a GPU in MIG mode, a selector naming a slice, another spelling, or
    any answer that is not one row of a UUID and a mode is refused."""
    uuid = "GPU-7e28035c-3c3b-b7aa-d655-0dad098bab62"
    calls: list = []

    def smi(out: str):
        return lambda args: (calls.append(list(args)), out)[1]

    assert audit.read_device("1", smi=smi(f"{uuid}, Disabled\n")) == audit.device_id(uuid)
    assert calls == [["-i", "1", "--query-gpu=uuid,mig.mode.current", "--format=csv,noheader"]]
    assert audit.read_device(uuid, smi=smi(f"{uuid}, [N/A]\n")) == audit.device_id(uuid)
    for gpu, out in (("1", f"{uuid}, Enabled\n"), ("1", f"{uuid.upper()}, Disabled\n"), ("1", ""),
                     ("1", f"{uuid}, Disabled\n{bv.DEVICES[0]}, Disabled\n"), ("1", f"{uuid}\n"),
                     ("1", f"MIG-{uuid[4:]}, Disabled\n"), ("1", f"{uuid},Disabled\n"),
                     ("MIG-" + uuid[4:], f"{uuid}, Disabled\n"), ("", f"{uuid}, Disabled\n")):
        with pytest.raises(ValueError):
            audit.read_device(gpu, smi=smi(out))


def test_a_window_past_its_limit_frees_the_store_and_is_late(big_store, state_of) -> None:
    store, vk = big_store
    params = _params("ro-blake3/v1")
    clock = _Clock()
    verifier = big_store.verifier(params, state_of(vk), clock=clock)
    first, second = verifier.window(), verifier.window()
    s1 = first.start()
    clock.now = params.limit_ns + 1
    s2 = second.start()
    c1 = audit.tag_pass(store, params.tag, s1)
    assert first.commit(c1.root) == () and first.verdict().failure == "late"
    c2 = audit.tag_pass(store, params.tag, s2)
    assert second.verdict([c2.open(i) for i in second.commit(c2.root)]).accepted


# --- the challenger's state ------------------------------------------------------------------------------------------


def test_the_state_records_a_window_before_its_seed(big_store, tmp_path, monkeypatch) -> None:
    """`open NONCE TAG` is written and fsynced before `σ` is derived, by `start` or by `open` ahead of it, and
    `close NONCE` follows the commitment."""
    store, vk = big_store
    audit.ChallengerState.create(tmp_path, vk)
    path = tmp_path / audit.state_name(vk)
    events: list = []
    fsync, seed = audit.os.fsync, audit.seed
    monkeypatch.setattr(audit.os, "fsync", lambda fd: (events.append(path.read_text().splitlines()[-1]), fsync(fd))[1])
    monkeypatch.setattr(audit, "seed", lambda key: (events.append("seed"), seed(key))[1])
    nonces = iter((bytes(range(32)), bytes(range(1, 33))))
    a, b = bytes(range(32)).hex(), bytes(range(1, 33)).hex()
    with audit.ChallengerState(tmp_path, vk, nonces=lambda n: next(nonces)) as state:
        verifier = big_store.verifier(_params("ro-blake3/v1"), state)
        w = verifier.window()
        s = w.start()
        assert events == [f"open {a} ro-blake3/v1", "seed"]
        c = audit.tag_pass(store, "ro-blake3/v1", s)
        assert w.verdict([c.open(i) for i in w.commit(c.root)]).accepted
        w = verifier.window()
        w.open()
        assert events[-1] == f"open {b} ro-blake3/v1"
        w.start()
        assert events[-2:] == [f"open {b} ro-blake3/v1", "seed"]
    assert path.read_text().splitlines()[1:] == [f"open {a} ro-blake3/v1", f"close {a}", f"open {b} ro-blake3/v1",
                                                 f"close {b}"]


def test_the_state_fails_closed(big_store, tmp_path) -> None:
    """A state that is missing, unreadable, another store's, held by another challenger or changed under this one
    opens no window, and a store's state is never made again."""
    store, vk = big_store
    states = tmp_path / "states"
    states.mkdir()
    path = states / audit.state_name(vk)
    with pytest.raises(audit.StateError, match="missing"):
        audit.ChallengerState(states, vk)
    audit.ChallengerState.create(states, vk)
    with pytest.raises(audit.StateError, match="never made again"):
        audit.ChallengerState.create(states, vk)
    other = bv.store(5)
    renamed = tmp_path / "renamed"
    renamed.mkdir()
    (renamed / audit.state_name(bv.vk_of(other))).write_bytes(path.read_bytes())
    with pytest.raises(audit.StateError, match="another store"):
        audit.ChallengerState(renamed, bv.vk_of(other))
    with audit.ChallengerState(states, vk) as state:
        with pytest.raises(audit.StateError, match="held"):
            audit.ChallengerState(states, vk)
        with pytest.raises(ValueError, match="challenger state"):
            replace(big_store, store=other, vk=bv.vk_of(other)).verifier(_params("ro-blake3/v1"), state)
    head = path.read_bytes()
    a, b = bytes(32).hex().encode(), bytes(range(32)).hex().encode()
    for i, body in enumerate((b"", head[:-1], head + b"open " + a + b" ro-blake3/v1", head + b"open " + a + b" nope/v1\n",
                              head + b"open " + a + b" ro-blake3/v1\nclose " + a + b"\nopen " + a + b" ro-blake3/v1\n",
                              head + b"close " + a + b"\n", head + b"open " + a + b" gf31-poly/v1\nopen " + b + b" gf31-poly/v1\n",
                              head + b"open " + b.upper() + b" ro-blake3/v1\n", head.replace(b"/v1 ", b"/v0 "))):
        bad = tmp_path / f"bad-{i}"
        bad.mkdir()
        (bad / audit.state_name(vk)).write_bytes(body)
        with pytest.raises(audit.StateError, match="unreadable"):
            audit.ChallengerState(bad, vk)
    with audit.ChallengerState(states, vk) as state:
        v = big_store.verifier(_params("ro-blake3/v1"), state)
        with open(path, "ab") as f:
            f.write(b"open " + a + b" ro-blake3/v1\n")
        with pytest.raises(audit.StateError, match="changed under"):
            v.window().start()


def test_a_store_has_one_challenger_state(big_store, tmp_path) -> None:
    """A store's state is the file its root and salt name in the challenger's state directory, beside other stores'.
    A second `create` for the store is refused, and a copy or hard link of the state under another name, a symbolic
    link, a subdirectory or a stray file makes `create` and opening refuse the directory, for every store, so no second
    state of a store can serve a second challenger whose windows overlap the first's; nor can a link in the state's
    place."""
    store, vk = big_store
    other, third = bv.vk_of(bv.store(5)), bv.vk_of(bv.store(6))
    states = tmp_path / "states"
    states.mkdir()
    audit.ChallengerState.create(states, vk)
    audit.ChallengerState.create(states, other)
    path = states / audit.state_name(vk)
    names = sorted(f"{v.root.hex()}-{v.salt.hex()}.state" for v in (vk, other))
    assert sorted(p.name for p in states.iterdir()) == names
    with pytest.raises(audit.StateError, match="never made again"):
        audit.ChallengerState.create(states, vk)
    with audit.ChallengerState(states, vk), audit.ChallengerState(states, other):
        pass
    extra = states / "copy.state"
    for stray in (lambda: shutil.copy(path, extra), lambda: os.link(path, extra), lambda: extra.symlink_to(path),
                  extra.mkdir, lambda: extra.write_bytes(b"notes\n")):
        stray()
        for make in (lambda: audit.ChallengerState(states, vk), lambda: audit.ChallengerState(states, other),
                     lambda: audit.ChallengerState.create(states, third)):
            with pytest.raises(audit.StateError, match="one challenger state|only states"):
                make()
        if extra.is_dir() and not extra.is_symlink():
            extra.rmdir()
        else:
            extra.unlink()
    real = tmp_path / "real.state"
    path.rename(real)
    path.symlink_to(real)
    with pytest.raises(audit.StateError, match="link"):
        audit.ChallengerState(states, vk)
    path.unlink()
    real.rename(path)
    with audit.ChallengerState(states, vk):
        pass
    assert len(path.read_text().splitlines()) == 1 and not (states / audit.state_name(third)).exists()


def test_a_window_open_when_its_challenger_stops_never_passes(big_store, tmp_path) -> None:
    """A window still open when its state closes, or when its challenger dies, closes with it: its commitment is
    refused, a window opened but not started never starts, and reopening the state records the dead challenger's
    window closed."""
    store, vk = big_store
    audit.ChallengerState.create(tmp_path, vk)
    path = tmp_path / audit.state_name(vk)
    params = _params("ro-blake3/v1")
    with audit.ChallengerState(tmp_path, vk, nonces=_nonces(b"closed")) as state:
        w = big_store.verifier(params, state).window()
        c = audit.tag_pass(store, params.tag, w.start())
    with pytest.raises(audit.StateError, match="closed"):
        w.commit(c.root)
    with audit.ChallengerState(tmp_path, vk, nonces=_nonces(b"opened")) as state:
        opened = big_store.verifier(params, state).window()
        opened.open()
    with pytest.raises(audit.StateError, match="closed"):
        opened.start()
    state = audit.ChallengerState(tmp_path, vk, nonces=_nonces(b"died"))
    w = big_store.verifier(params, state).window()
    c = audit.tag_pass(store, params.tag, w.start())
    state._file.close()
    nonce = w._nonce.hex()
    assert path.read_text().splitlines()[-1] == f"open {nonce} {params.tag}"
    with audit.ChallengerState(tmp_path, vk):
        assert path.read_text().splitlines()[-1] == f"close {nonce}"
    with pytest.raises(audit.StateError, match="closed"):
        w.commit(c.root)
    assert len(path.read_text().splitlines()) == 7


# --- no window's samples are reused, across verifiers, tags and restarts ----------------------------------------------


def _forge(store: audit.Store, tag: str, seed: bytes, known: set[int]) -> audit.TagCommitment:
    """A prover that tags only the labels in `known` (samples it learned elsewhere) and guesses the rest."""
    return _Corrupt(store, tag, seed, lambda j: j not in known)


def test_a_second_verifier_of_the_store_repeats_no_window(big_store, state_of) -> None:
    """Two verifiers of one tag over one store, both counting from their first window, share the store's state, so
    their windows draw different seeds and samples, and samples learned from one pass none of the other's windows; a
    second state over the same file is refused while the first runs."""
    store, vk = big_store
    params = _params("ro-blake3/v1")
    state = state_of(vk)
    first, second = (big_store.verifier(params, state) for _ in range(2))
    w1 = first.window()
    c1 = audit.tag_pass(store, params.tag, w1.start())
    drawn1 = w1.commit(c1.root)
    assert w1.verdict([c1.open(i) for i in drawn1]).accepted
    w2 = second.window()
    seed2 = w2.start()
    assert seed2 != c1.seed
    forged = _forge(store, params.tag, seed2, set(drawn1))
    drawn2 = w2.commit(forged.root)
    assert drawn2 != drawn1 and not set(drawn2) <= set(drawn1)
    assert w2.verdict([forged.open(i) for i in drawn2]).failure == "tag-mismatch"
    with pytest.raises(audit.StateError, match="held"):
        audit.ChallengerState(state.directory, vk)


def test_another_tags_window_reveals_no_samples(big_store, state_of) -> None:
    """The fast check's window and the certified tag's window over one store, both the first of their verifiers, draw
    different samples: the samples a `gf31-poly/v1` window reveals after its commitment pass no `ro-blake3/v1` window,
    and a prover that tags only those labels is rejected."""
    store, vk = big_store
    state = state_of(vk)
    fast, certified = (big_store.verifier(_params(t), state) for t in ("gf31-poly/v1", "ro-blake3/v1"))
    wf = fast.window()
    cf = audit.tag_pass(store, "gf31-poly/v1", wf.start())
    revealed = wf.commit(cf.root)
    assert wf.verdict([cf.open(i) for i in revealed]).accepted
    wc = certified.window()
    forged = _forge(store, "ro-blake3/v1", wc.start(), set(revealed))
    drawn = wc.commit(forged.root)
    assert drawn != revealed and not set(drawn) <= set(revealed)
    assert wc.verdict([forged.open(i) for i in drawn]).failure == "tag-mismatch"


def test_a_restarted_challenger_repeats_no_window_and_refuses_a_replayed_root(big_store, tmp_path) -> None:
    """After a restart the store's state holds every nonce sampled before it: a challenger whose randomness repeats one
    opens no window, and a fresh window's seed and samples are new, so the root and openings of a window from before
    the restart, replayed without reading the store, are refused: the replayed leaves are bound to the old seed, and the
    old openings answer other samples."""
    store, vk = big_store
    params = _params("ro-blake3/v1")
    audit.ChallengerState.create(tmp_path, vk)
    path = tmp_path / audit.state_name(vk)
    with audit.ChallengerState(tmp_path, vk, nonces=_nonces(b"before")) as state:
        w = big_store.verifier(params, state).window()
        old = audit.tag_pass(store, params.tag, w.start())
        old_drawn = w.commit(old.root)
        assert w.verdict([old.open(i) for i in old_drawn]).accepted
    with audit.ChallengerState(tmp_path, vk, nonces=_nonces(b"before")) as state:
        with pytest.raises(audit.StateError, match="never drawn twice"):
            big_store.verifier(params, state).window().start()
    with audit.ChallengerState(tmp_path, vk, nonces=_nonces(b"after")) as state:
        w = big_store.verifier(params, state).window()
        assert w.start() != old.seed
        drawn = w.commit(old.root)
        assert drawn != old_drawn
        assert w.verdict([old.open(i) for i in drawn]).failure == "tag-path"
        w = big_store.verifier(params, state).window()
        w.start()
        assert w.commit(old.root) != old_drawn
        assert w.verdict([old.open(i) for i in old_drawn]).failure == "malformed"
    assert sum(r.startswith("open ") for r in path.read_text().splitlines()) == 3


def test_params_refuse_a_window_that_fits_a_root() -> None:
    audit.Params("ro-sha512/v1", 1, 2 * ROOT_CALL_NS - 1)
    with pytest.raises(ValueError, match="rebuild floor"):
        audit.Params("ro-sha512/v1", 1, 2 * ROOT_CALL_NS)
    with pytest.raises(ValueError, match="rebuild floor"):
        audit.Params("ro-sha512/v1", 1, 1_000_000, rtt_allowance_ns=ROOT_CALL_NS * 2 - 1_000_000)
    with pytest.raises(ValueError):
        audit.Params("ro-sha512/v1", 0, 1)


def test_verifier_refuses_a_tag_the_store_cannot_carry(big_store, state_of) -> None:
    store = bv.store(4, 18)
    vk = bv.vk_of(store)
    with pytest.raises(ValueError, match="u32 words"):
        replace(big_store, store=store, vk=vk).verifier(_params("gf31-poly/v1"), state_of(vk))


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
    assert audit.QUERIES_LOG2 == 64 and b.side_bytes == 69_300_000 and b.slack_bits == 64 * k + 41
    assert b.ingress_bytes == (k * 2056 * 8 - 8 * 208_000_000 - 8 * 69_300_000 - (64 * k + 41)) // 8
    assert b.error_compression_log2 == -41 and b.error_guess <= 1849 * 2.0 ** -256
    assert b.error < 2.0 ** -40 and 0.7e9 < b.ingress_bytes < (1 << 19) * 2056


def test_the_fast_check_has_no_bound() -> None:
    with pytest.raises(ValueError, match="uncertified"):
        audit.bound(audit.Params("gf31-poly/v1", audit.SAMPLES, 1_000_000), 1 << 19, 2056, chip_bytes=0,
                    side_bytes_per_s=0)
