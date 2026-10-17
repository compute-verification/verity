"""KeyBinding's game against the remote audit's order of appends and passes.

The stacked guarantees (`Security/Models/Accounting/Guarantees/Stacking.lean`) read a session's store as
`view (choice ω) ω`, with ω drawn before the session's coins: "a write that adapts to the coins is not in this game".
The audit appends a KV page mid-window and plans it in the pass whose seed existed before the page's payload was
committed (`Session.append` need only precede the root of the pass that holds it; `test_appends.honest_session` does
this on the honest path). Such a page can hold that pass's sketches of other blocks, 64 bytes a block at 16 rows, from
which their receipts in the pass follow without reading them: a carrier whose contents are a function of the coins,
which `RateSketch.Provenance` (inputs a function of the store alone) cannot state."""

from __future__ import annotations

import hashlib

from verity.core.protocols.remote_bandwidth import audit, codec, coins, receipts
from verity.core.protocols.remote_bandwidth.audit import Params, Session
from verity.core.protocols.remote_bandwidth.receipts import Plan

N = 8
NODE = b"node-under-test"
SOURCE = b"the verifier's secret"
P512 = codec.Codec(codec.P512, 2, False)


class Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


class Entropy:
    """SHA-512 in counter mode under `label`, in place of the operating system's generator."""

    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


def payload(label: bytes) -> bytes:
    return b"".join(hashlib.sha512(label + b"/%d" % u).digest()[:63] + b"\0" for u in range(codec.BLOCK_BYTES // 64))


def salts(label: bytes, n: int) -> list[bytes]:
    entropy = Entropy(label)
    return [entropy(receipts.SALT_BYTES) for _ in range(n)]


def carrier(store: audit.Store, seed: bytes) -> bytes:
    """A KV page's payload: the registered blocks' sketches under the pass seed `seed`, zero-padded to a block."""
    sketch = receipts.SketchReceipter(seed).sketch
    body = b"".join(sketch(store.blocks[j]) for j in range(N))
    return body + bytes(codec.BLOCK_BYTES - len(body))


def relayed(page: bytes, seed: bytes) -> list[bytes]:
    """The registered blocks' receipts under `seed`, from the page's sketches alone."""
    head, width = receipts.SketchReceipter(seed).head, 4 * receipts.SKETCH_ROWS
    out = []
    for j in range(N):
        h = head.copy()
        h.update(j.to_bytes(8, "big"))
        h.update(page[width * j:width * (j + 1)])
        out.append(h.digest())
    return out


def test_a_page_appended_after_a_pass_seed_carries_that_pass_s_receipts_of_the_registered_blocks() -> None:
    """Window 0: pass 0 over the registered store; then a page holding pass 1's sketches of the registered blocks,
    made from seed_(0,1) = H(σ_0, 1, root_(0,0)) and appended at index N; then pass 1 over the registered blocks and the
    page, whose registered receipts are computed from the page's 512 bytes of sketches, not from the blocks. Every
    receipt is drawn and opened (the openings' blocks from the prover's host copy), and the window is accepted. A
    page made the same way from pass 0's seed, the only seed before root_(0,0), relays none of pass 1's receipts: the
    page is a function of the session's coin σ_0 and of the prover's root, after both."""
    payloads = [payload(b"weights %d" % j) for j in range(N)]
    ys = salts(b"payload salts", N)
    root = audit.payload_tree(NODE, payloads, ys).commitment.root
    store = audit.Store(audit.register(SOURCE, NODE, N, root, P512, nonce=bytes(32)), payloads, ys)
    reg = store.registration
    p = Params(blocks=N, deadline_ns=10**6, windows=1, samples=10**6, delta=1e-6, window_ns=10, codec=reg.codec)
    clock = Clock()
    s = Session(p, reg, None, coins.Challenger(reg.digest, 0, p.coins, entropy=Entropy(b"coins")), not_before_ns=0,
                clock=clock)
    prover = audit.Prover(store, None, s.commitment, entropy=Entropy(b"receipt salts"))
    challenge = s.start()

    pass0 = prover.run(challenge, 0, reg.digest)
    assert s.receive(0, 0, pass0.root) is None
    seed1 = receipts.seed(challenge.coin, 1, pass0.root)

    page = carrier(store, seed1)
    a = store.append([page], salts(b"page salts", 1))
    assert s.append(a.root, a.count) == N
    plan = Plan(((0, N + 1),), store.extent)
    honest = prover.pass_under(0, 1, seed1, plan).receipts
    held = codec.decode(store.blocks[N], N, reg.key, a.digests[0])
    assert held == page
    ys1 = tuple(relayed(held, seed1)) + (receipts.SketchReceipter(seed1)(N, store.blocks[N]),)
    assert ys1 == honest
    assert relayed(carrier(store, pass0.seed), seed1) != honest[:N]
    assert codec.BLOCK_BYTES // (4 * receipts.SKETCH_ROWS) == 64

    leaf_salts = tuple(Entropy(b"pass 1 salts")(receipts.SALT_BYTES) for _ in range(receipts.leaves(plan.length)))
    pass1 = audit.Pass(0, 1, seed1, plan, ys1, leaf_salts, receipts.tree(seed1, plan, ys1, leaf_salts))
    assert s.receive(0, 1, pass1.root, plan.runs) is None

    clock.now = p.window_ns
    assert s.boundary() is None
    clock.now = s.deadline(0) + 1
    draws = s.close(0)
    passes = {0: pass0, 1: pass1}
    assert len(draws) == 2 * N + 1
    assert s.verify(0, [prover.open(passes[i], q) for i, q in draws]).accepted


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
