"""`poub.opening`: check 3 by the opening unit's reference on a small p512/k2 store. An honest window is
accepted; the same window over a store that lost its first drawn block is rejected on codec at that draw; an uncertified
plan row fails the plan link and a public row not the verifier's own is malformed. A plan row's ``b || c`` is the proof
system's (pinned from C-Flock's commit strings)."""

from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest

from verity.core.protocols.poub import audit, codec, coins, opening as O, receipts
from verity.core.protocols.poub.receipts import Plan

N, L = 32, 16
NODE, SOURCE = b"opening-node", b"the verifier's secret"
P512 = codec.Codec(codec.P512, 2, False)
#: C-Flock's ``b || c`` (`verity_flock.circuit._commit_strings`) of the plan row 64..127 of 1,024 bits under `SALT`
PLAN_BC = ("12f7869e085f13e475ea68ff04f83289655b686aa1c56da778cc299af82f355882e4e6870600f12fa17ba27a530593488ad2df0e5745cf04c3"
           "f28e1caf352a9f425fcf1a31e3f75bce33b749894811a7b0ded5eed561fc3e18dee59b0f24f0806e3369646067a9612ea4ff35315e63e53bf4"
           "fbe9f2bce4d0b79f9cffa203ecd4")
SALT = hashlib.shake_256(b"verity/poub/opening/test plan salt").digest(receipts.SALT_BYTES)


class Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


class Entropy:
    """A deterministic stand-in for the operating system's generator: SHA-512 in counter mode under `label`."""

    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


def payload(j: int) -> bytes:
    """64-byte words with their top byte cleared, so every word is below p512's p."""
    return b"".join(hashlib.sha512(b"%d/%d" % (j, u)).digest()[:63] + b"\0" for u in range(codec.BLOCK_BYTES // 64))


@pytest.fixture(scope="module")
def store() -> O.Store:
    payloads = [payload(j) for j in range(N)]
    salt = Entropy(b"payload salts")
    ys = [salt(receipts.SALT_BYTES) for _ in range(N)]
    root = audit.payload_tree(NODE, payloads, ys).commitment.root
    return O.Store(audit.register(SOURCE, NODE, N, root, P512, nonce=bytes(32)), payloads, ys)


def window(store: O.Store) -> SimpleNamespace:
    """One window of N / L passes, pass i the stripe of L blocks from L i, closed and drawn; the same coins every call, so
    the same draws over any store of the same registration."""
    reg = store.registration
    p = audit.Params(blocks=N, deadline_ns=10**6, windows=1, samples=4, delta=1e-3, window_ns=10, codec=P512)
    clock = Clock()
    s = O.Session(p, reg, None, coins.local(reg.digest, 0, p.windows, Entropy(b"coins"), appends=1), not_before_ns=0,
                  clock=clock)
    prover = audit.Prover(store, None, s.commitment, entropy=Entropy(b"receipt salts"))
    challenge, previous, passes = s.start(), reg.digest, {}
    for i in range(N // L):
        runs = ((L * i, L),)
        ps = prover.run(challenge, i, previous, Plan(runs, N), head=store.head)
        assert s.receive(0, i, ps.root, runs, step=i) is None
        passes[0, i], previous = ps, ps.root
    clock.now += p.window_ns
    s.boundary()
    clock.now = s.deadline(0) + 1
    assert not isinstance(s.close(0), audit.Verdict)
    salt = Entropy(b"plan salts")
    rows = {i: list(range(L * i, L * (i + 1))) for i in range(N // L)}
    salts = {i: salt(receipts.SALT_BYTES) for i in rows}
    c3 = SimpleNamespace(codec=P512, P=L, DR=O.depth(receipts.leaves(L)), DP=O.depth(N), DV=1, plan_rows=rows,
                         plan_salts=salts, plan_bcs={i: O.commit(16 * L, rows[i], salts[i]) for i in rows},
                         values=O.Values(bytes(64), None, N, ()))
    return SimpleNamespace(session=s, passes=passes, c3=c3)


def test_a_plan_rows_commitment_is_the_proof_systems() -> None:
    assert O.commit(16 * 64, list(range(64, 128)), SALT).hex() == PLAN_BC
    with pytest.raises(ValueError, match="1024 bits is 64 u16 words, not 63"):
        O.commit(16 * 64, list(range(64, 127)), SALT)


def test_an_honest_window_is_accepted(store) -> None:
    w = window(store)
    assert len(w.session.drawn(0)) == 4
    assert O.reference(w.session, store, w.passes, w.c3) == [audit.Verdict(0, True)]


def test_a_lost_block_fails_on_codec_at_its_draw(store) -> None:
    """The store's block at the first draw replaced by other bytes before the passes: the prover receipts what it holds,
    the coins draw the same positions, and the unit's verdict on that draw is codec."""
    d0 = window(store).session.drawn(0)[0]
    lost = copy.copy(store)
    lost.blocks = list(store.blocks)
    lost.blocks[d0.block] = hashlib.shake_256(b"a lost block").digest(codec.BLOCK_BYTES)
    w = window(lost)
    assert w.session.drawn(0)[0].block == d0.block
    assert O.reference(w.session, lost, w.passes, w.c3) == [audit.Verdict(0, False, "codec", (d0.index, d0.position))]


def test_a_window_with_no_draws_is_malformed(store) -> None:
    """A window whose draws are empty (none a session makes) is `malformed` at `decide`, as at `verify`."""
    w = window(store)
    w.session.windows[0].draws = ()
    assert w.session.decide(0, []) == audit.Verdict(0, False, "malformed")


def test_the_plan_link_and_the_public_rows(store) -> None:
    w = window(store)
    d = w.session.drawn(0)[0]
    S = O.statics(P512, d.leaf_size, L, w.c3.DR, w.c3.DP, w.c3.DV)
    a = O.witness(d, w.passes[d.window, d.index], store, w.c3.plan_rows[d.step], S, w.c3.values)
    seed = store.registration.codec_seed
    salt, bc = w.c3.plan_salts[d.step], w.c3.plan_bcs[d.step]
    assert O.check(d, a, seed, S, w.c3.values, salt, bc) is None
    other = 1 - d.step
    assert O.check(d, a, seed, S, w.c3.values, salt, w.c3.plan_bcs[other]) == "plan"
    assert O.check(d, a, seed, S, w.c3.values, w.c3.plan_salts[other], bc) == "plan"
    for key in O.public(d, seed, S, w.c3.values):
        off = dict(a, **{key: [a[key][0] ^ 1, *a[key][1:]]})
        assert O.check(d, off, seed, S, w.c3.values, salt, bc) == "malformed", key
