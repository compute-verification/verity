"""The remote bandwidth audit over PoUS's P2 store (`p2.py`): the registration from PoUS's setup of a 9-block C*
(`fixtures/p2-c-star.bin`, the prover's bytes), the unit sketch and its per-block partials, an honest session
accepted and recorded, and each way an opening goes wrong rejected."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from pathlib import Path

import pytest

from verity.core.protocols.pous import setup
from verity.core.protocols.pous.schemes import p2 as pous_p2
from verity.core.protocols.pous.schemes.p2 import P2Params
from verity.core.protocols.remote_bandwidth import audit, codec, coins, p2, receipts, timing
from verity.ml.randomness import frame

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE = b"the verifier's secret"
NODE = b"p2-node-under-test"
SCHEDULE = timing.honest(1000, 2, 300, gap_ns=10, down_ns=20, up_ns=30)
A = 20 + 30 + 300 + 10


class Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


class Entropy:
    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


@pytest.fixture(scope="module")
def fixture() -> dict:
    out = json.loads((FIXTURES / "p2-store.json").read_text())
    data = (FIXTURES / "p2-c-star.bin").read_bytes()
    assert hashlib.sha512(data).hexdigest() == out["c_star_sha512"]
    out["blocks"] = [data[p2.BLOCK_BYTES * j:p2.BLOCK_BYTES * (j + 1)] for j in range(len(data) // p2.BLOCK_BYTES)]
    m = out["params"]
    out["pous"] = P2Params(m["w"], m["c0"], m["blocks_per_segment"], m["segments"], m["expander"])
    return out


@pytest.fixture(scope="module")
def vk(fixture):
    """PoUS's setup on the verifier's side: C* decoded segment by segment and held to the commitment to W."""
    params, blocks, n = fixture["pous"], fixture["blocks"], fixture["pous"].blocks_per_segment
    commitment, salt = bytes.fromhex(fixture["commitment"]), bytes.fromhex(fixture["salt"])
    return setup.verifier_key_from_encoding(params, None, commitment, salt,
                                            [blocks[n * s:n * (s + 1)] for s in range(params.segments)])


@pytest.fixture(scope="module")
def store(fixture, vk) -> p2.Store:
    reg = p2.register(SOURCE, NODE, bytes.fromhex(fixture["commitment"]), vk)
    return p2.Store(reg, fixture["blocks"])


def params(reg: p2.Registration, **kw) -> audit.Params:
    base = dict(blocks=reg.blocks, deadline_ns=A, windows=2, samples=2, delta=1e-6, window_ns=1000, codec=reg.codec,
                receipt=reg.receipt, layout=reg.layout, rows=reg.rows)
    return audit.Params(**{**base, **kw})


def drive(store: p2.Store, p: audit.Params, run=None):
    clock = Clock()
    reg = store.registration
    s = audit.AuditRun(p, reg, None, coins.local(reg.digest, 0, p.windows, Entropy(b"coins")), not_before_ns=0,
                       clock=clock)
    prover = p2.Prover(store, entropy=Entropy(b"receipt salts"))
    events = sorted([(0, 0, -1)] + [(b, 0, t) for t, b in enumerate(SCHEDULE.boundaries)]
                    + [(a, 1, (t, i)) for t, i, a in SCHEDULE.arrivals])
    challenges, passes, previous = {}, {}, store.registration.digest
    for step, (at, kind, x) in enumerate(events):
        clock.now = at
        if kind == 0:
            coin = s.start() if x < 0 else s.boundary()
            if coin is not None:
                challenges[x + 1] = coin
            continue
        t, i = x
        ps = (run or (lambda pr, ch, i_, prev: pr.run(ch, i_, prev)))(prover, challenges[t], i, previous)
        passes[t, i], previous = ps, ps.root
        assert s.receive(t, i, ps.root, ps.plan.runs, step=step) is None
    clock.now = s.deadline(p.windows - 1) + 1
    return s, prover, passes


def verdicts(s, prover, passes, open_=None):
    out = []
    for t in range(s.params.windows):
        draws = s.close(t)
        out.append(s.verify(t, [(open_ or prover.open)(passes[t, i], q) for i, q in draws]))
        if not out[-1].accepted:
            break
    return out


def test_the_codec_is_pous_registered_point(fixture) -> None:
    assert codec.P2_POINT == pous_p2.POINT[:2] and codec.P2_EXPANDER == pous_p2.SHA512_CTR
    c = p2.codec_of(fixture["pous"])
    assert c.name == fixture["pous"].codec.name == "p2-16448/v4" and c.blocks == 9
    assert c.block_encode_squarings == 2 * (((fixture["pous"].p + 1) // 4).bit_length() - 1) == 32_890
    assert p2.BLOCK_BYTES == fixture["pous"].block_bytes and p2.UNIT_SECTORS == 257
    assert p2.UNIT_BYTES % p2.SECTOR_BYTES == 0 and p2.BLOCK_BYTES % p2.SECTOR_BYTES != 0
    with pytest.raises(ValueError, match="registered point"):
        p2.codec_of(P2Params(16448, 21065, 3, 3, pous_p2.CHACHA8))
    with pytest.raises(ValueError, match="registered point"):
        codec.P2Codec(3, 3, w=8192)


def test_the_registration_is_pous_setups(fixture, vk, store) -> None:
    reg = store.registration
    assert reg.salt == setup.draw_salt(SOURCE, reg.commitment) == bytes.fromhex(fixture["salt"])
    assert reg.root == vk.root == bytes.fromhex(fixture["root"]) and reg.blocks == 2 and reg.rows == 16
    c = reg.codec
    assert reg.digest == hashlib.sha512(
        p2.REGISTRATION_DOMAIN + frame(NODE) + struct.pack(">IIIQ", c.w, c.c0, 3, 3) + frame("sha512-ctr-v4")
        + reg.commitment + reg.salt + reg.root + frame("p2-unit4/v0") + frame("sketch-sha512")
        + (16).to_bytes(4, "big")).digest()
    assert len({reg.digest, dataclasses.replace(reg, rows=64).digest, dataclasses.replace(reg, node=b"x").digest}) == 3
    other = setup.draw_salt(b"another secret", reg.commitment)
    with pytest.raises(ValueError, match="under the salt"):
        p2.register(b"another secret", NODE, reg.commitment, vk)
    assert other != reg.salt
    with pytest.raises(ValueError, match="not the registered C"):
        p2.Store(reg, fixture["blocks"][1:] + fixture["blocks"][:1])
    with pytest.raises(ValueError, match="at least one unit"):
        p2.Registration(NODE, codec.P2Codec(3, 1), reg.commitment, reg.salt, reg.root)


def test_a_units_sketch_is_its_blocks_partials_summed(store) -> None:
    seed = hashlib.sha512(b"a pass seed").digest()
    sk = p2.Sketcher(seed)
    signs = sk.signs()
    assert len(signs) == 16 and {len(r) for r in signs} == {1028}
    assert signs != receipts.sign_bits(seed, 16) and signs[0][:512] != receipts.sign_bits(seed, 16)[0]
    for g in range(store.registration.blocks):
        unit = store.unit(g)
        partials = [sk.block(4 * g + b, store.blocks[4 * g + b]) for b in range(4)]
        sigma = sk.total(partials)
        assert sigma == sk.sketch(unit) and sk.receipt(g, sigma) == sk(g, unit)
        row = 3
        want = sum((-1 if signs[row][x // 8] >> (x % 8) & 1 else 1) * (v - 256 if v > 127 else v)
                   for x, v in enumerate(unit))
        assert struct.unpack("<16i", sigma)[row] == want
        assert sk(g, unit) == hashlib.sha512(receipts.sketch_prefix(seed) + g.to_bytes(8, "big") + sigma).digest()
        x = store.blocks[4 * g + 2]
        bits = signs[0][p2.BLOCK_SIGN_BYTES * 2:p2.BLOCK_SIGN_BYTES * 3]
        want0 = sum((-1 if bits[u // 8] >> (u % 8) & 1 else 1) * (v - 256 if v > 127 else v) for u, v in enumerate(x))
        assert struct.unpack("<16i", partials[2])[0] == want0


def test_an_honest_session_over_the_p2_store_is_accepted(store) -> None:
    s, prover, passes = drive(store, params(store.registration))
    assert all(v.accepted for v in verdicts(s, prover, passes))
    r = s.record()
    pub = r.public()
    assert pub["block_bytes"] == p2.UNIT_BYTES and pub["blocks"] == 2 and pub["codec"] == "p2-16448/v4"
    assert pub["codec_params"] == {"blocks_per_segment": 3, "c0": 21065, "expander": "sha512-ctr-v4", "segments": 3,
                                   "w": 16448}
    assert pub["layout"] == "p2-unit4/v0" and pub["receipts"] == [2 * n for n in SCHEDULE.passes]
    with pytest.raises(ValueError, match="P2 store waits on the squaring ceiling"):
        audit.ledger(r, capacity_bytes_per_s=1e12, chip_bytes=0, rebuild=audit.Rebuild(r.codec, 0.0))


def test_receipts_from_the_decodes_partials_are_the_passes(store) -> None:
    """The serving side: π of each block as it is decoded, in any order, σ per unit, then y; the pass committed from
    them is the reference's, and accepted."""

    def decode_side(prover, challenge, i, previous):
        seed = receipts.seed(coins.coin(challenge), i, previous)
        sk, plan = p2.Sketcher(seed), receipts.Plan.whole(2)
        partials = {j: sk.block(j, store.blocks[j]) for j in reversed(range(8))}
        ys = [sk.receipt(g, sk.total([partials[4 * g + b] for b in range(4)])) for g in plan]
        ps = prover.commit(challenge.index, i, seed, plan, ys)
        assert ps.receipts == prover.pass_under(challenge.index, i, seed).receipts
        return ps

    s, prover, passes = drive(store, params(store.registration), decode_side)
    assert all(v.accepted for v in verdicts(s, prover, passes))


@pytest.mark.parametrize("fault, failure", [
    ("swap", "payload-path"), ("path", "payload-path"), ("three", "malformed"), ("receipts", "receipt"),
    ("leaf", "receipt-path"), ("salt", "malformed"),
])
def test_a_bad_opening_is_rejected(store, fault, failure) -> None:
    def zeros(prover, challenge, i, previous):
        seed = receipts.seed(coins.coin(challenge), i, previous)
        sk = p2.Sketcher(seed)
        return prover.commit(challenge.index, i, seed, receipts.Plan.whole(2),
                             [sk(g, bytes(p2.UNIT_BYTES)) for g in range(2)])

    s, prover, passes = drive(store, params(store.registration), zeros if fault == "receipts" else None)

    def open_(ps, q):
        o = prover.open(ps, q)
        if fault == "swap":
            return dataclasses.replace(o, blocks=(o.blocks[1], o.blocks[0], *o.blocks[2:]))
        if fault == "path":
            return dataclasses.replace(o, paths=((o.paths[0][0][::-1], *o.paths[0][1:]), *o.paths[1:]))
        if fault == "three":
            return dataclasses.replace(o, blocks=o.blocks[:3], paths=o.paths[:3])
        if fault == "leaf":
            return dataclasses.replace(o, leaf_salt=bytes(len(o.leaf_salt)))
        if fault == "salt":
            return dataclasses.replace(o, leaf_salt=o.leaf_salt[:-1])
        return o

    v = verdicts(s, prover, passes, open_)
    assert not v[-1].accepted and v[-1].failure == failure and s.outcome == failure


def test_the_tail_past_the_last_unit_is_in_no_plan(store) -> None:
    s, _, _ = drive(store, params(store.registration))
    assert s.registration.blocks == 2 and store.registration.codec.blocks == 9
    with pytest.raises(ValueError):
        receipts.Plan(((2, 1),), s.registration.blocks)


def test_a_p2_store_takes_no_append_no_tiles_and_no_sloth_session(store) -> None:
    """One session type over either codec: over a P2 store its blocks are units, an append is refused, check 3's unit
    (which decodes) has no draws to take, no tiles bind, and the registration and codec are P2's together."""
    reg = store.registration
    s = audit.AuditRun(params(reg), reg, None, coins.local(reg.digest, 0, 2, Entropy(b"c")), not_before_ns=0,
                       clock=Clock())
    assert s.block_bytes == p2.UNIT_BYTES
    s.start()
    with pytest.raises(audit.SessionError, match="P2 store's openings are checked on PoUS's vk"):
        s.drawn(0)
    v = s.append(bytes(64), 1)
    assert isinstance(v, audit.Verdict) and v.failure == "append"
    with pytest.raises(ValueError, match="takes None"):
        audit.AuditRun(params(reg), reg, audit.NoiseTiles(), coins.local(reg.digest, 0, 2, Entropy(b"c")),
                       not_before_ns=0)
    sloth = audit.register(SOURCE, NODE, 2, bytes(64), nonce=bytes(32))
    with pytest.raises(ValueError, match="P2 Registration"):
        audit.AuditRun(params(reg), sloth, None, coins.local(sloth.digest, 0, 2, Entropy(b"c")), not_before_ns=0)
    with pytest.raises(ValueError, match="not the registration's"):
        audit.AuditRun(audit.Params(blocks=2, deadline_ns=A, windows=2), reg, None,
                       coins.local(reg.digest, 0, 2, Entropy(b"c")), not_before_ns=0)
    for kw in (dict(layout=receipts.STORE_ORDER), dict(receipt=receipts.BLOCK_SHA512, rows=0), dict(bind=True)):
        with pytest.raises(ValueError, match="P2 store's session"):
            params(reg, **kw)
