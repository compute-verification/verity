"""KeyBinding's game against the audit's order of appends and passes, the red team's KB-1
(note:20261017T1859Z-report-property-redteam §R4.2): a KV page appended after a pass's seed can hold that pass's
sketches of other blocks, 64 bytes a block at 16 rows, and their receipts follow from the page's bytes without reading
them.  A pass's seed is its own live coin, released after the root before it (`Session.seed`), and the session takes
no append while a pass's coin is out, so no block in a pass's plan was appended after its seed left, and a page
appended before it is independent of it."""

from __future__ import annotations

import copy
import hashlib

import pytest

from test_appends import N, Entropy, Run, pages, salts, store
from verity.core.protocols.remote_bandwidth import audit, codec, coins, receipts
from verity.core.protocols.remote_bandwidth.receipts import Plan


def carrier(st: audit.Store, seed: bytes) -> bytes:
    """A KV page's payload: the registered blocks' sketches under the pass seed `seed`, zero-padded to a block."""
    sketch = receipts.SketchReceipter(seed).sketch
    body = b"".join(sketch(st.blocks[j]) for j in range(N))
    return body + bytes(codec.BLOCK_BYTES - len(body))


def relayed(page: bytes, seed: bytes) -> tuple[bytes, ...]:
    """The registered blocks' receipts under `seed`, from the page's sketches alone."""
    head, width = receipts.SketchReceipter(seed).head, 4 * receipts.SKETCH_ROWS
    out = []
    for j in range(N):
        h = head.copy()
        h.update(j.to_bytes(8, "big"))
        h.update(page[width * j:width * (j + 1)])
        out.append(h.digest())
    return tuple(out)


def relay(r: Run, page_seed: bytes, plan: Plan) -> audit.Pass:
    """Pass 1 over the registered blocks and the page at N, its registered receipts relayed from the page and the
    page's own receipt honest, all under `page_seed`."""
    page = codec.decode(r.store.blocks[N], N, r.store.registration.key, r.store.appends[0].digests[0], r.coins[N])
    ys = relayed(page, page_seed) + (receipts.SketchReceipter(page_seed)(N, r.store.blocks[N]),)
    leaf = Entropy(b"pass 1 salts")
    leaf_salts = tuple(leaf(receipts.SALT_BYTES) for _ in range(receipts.leaves(plan.length)))
    return audit.Pass(0, 1, page_seed, plan, ys, leaf_salts, receipts.tree(page_seed, plan, ys, leaf_salts))


def v0_seed(r: Run, i: int, previous: bytes) -> bytes:
    """Pass i's seed as seed/v0 had it, SHA512(SEED_DOMAIN ‖ σ_0 ‖ i ‖ root before): what the prover could compute,
    and grind, before sending its root."""
    return hashlib.sha512(b"verity/remote-bandwidth/seed/v0\0" + coins.coin(r.challenges[0]) + i.to_bytes(4, "big")
                          + previous).digest()


def test_a_page_appended_after_a_pass_s_seed_left_is_out_of_turn() -> None:
    """The red team's order under live coins: pass 0 over the registered store, then pass 1's coin asked for, and a
    page holding pass 1's sketches of the registered blocks under that coin appended before pass 1's root.  The session
    takes no append while a pass's coin is out: the append is out of turn (`order`) and rejects the session, so the
    page never joins the store a plan of pass 1 may hold."""
    r = Run(store(), windows=1)
    assert r.run() is None
    seed = coins.seed(r.s.seed())
    a = r.store.append([carrier(r.store, seed)], salts(b"page salts", 1))
    assert r.s.append(a.root, a.count) == audit.Verdict(0, False, "order")
    assert r.s.failed == audit.Verdict(0, False, "order") and r.s.extent == N


def test_a_page_appended_before_a_pass_s_seed_relays_none_of_its_receipts() -> None:
    """A page appended between passes is committed before the next pass's coin leaves, so it holds sketches under no
    seed but a guess: here the seed seed/v0 would have given pass 1, H(σ_0, 1, root_(0,0)), under which the page relays
    all N registered receipts (the attack the red team measured, which the session accepted at #2381).  Pass 1's seed
    is its coin, drawn after the page: the relayed receipts are none of the pass's, the pass sent under the guess fails
    its receipts' paths, and the honest pass over the page is accepted."""
    r = Run(store(), windows=1)
    assert r.run() is None
    guess = v0_seed(r, 1, r.passes[0, 0].root)
    a = r.store.append([carrier(r.store, guess)], salts(b"page salts", 1))
    assert r.commit(a.root, a.count) == N
    plan = Plan(((0, N + 1),), r.store.extent)
    attack = relay(r, guess, plan)
    assert attack.receipts == r.prover.pass_under(0, 1, guess, plan).receipts

    cheat = copy.deepcopy(r)
    coin = cheat.s.seed()
    page = codec.decode(r.store.blocks[N], N, r.store.registration.key, a.digests[0], r.coins[N])
    assert coins.seed(coin) != guess
    assert not set(relayed(page, coins.seed(coin))) & set(cheat.prover.pass_under(0, 1, coins.seed(coin), plan).receipts)
    assert cheat.s.receive(0, 1, attack.root, plan.runs, step=1) is None
    cheat.passes[0, 1] = attack
    cheat.boundary()
    verdict = cheat.finish()[0]
    assert not verdict.accepted and verdict.failure == "receipt-path" and verdict.at[0] == 1

    assert r.run(plan) is None and r.passes[0, 1].seed == r.s.windows[0].seeds[1] != guess
    r.boundary()
    assert r.finish() == [audit.Verdict(0, True)]


@pytest.mark.parametrize("appended", ["between", "during"])
def test_a_pass_s_plan_holds_only_blocks_appended_before_its_coin(appended) -> None:
    """Pass 1 reads two pages appended after pass 0.  A third page appended between passes 0 and 1 joins the store
    and pass 1 is accepted; one sent while pass 1's coin is out is out of turn (`order`)."""
    r = Run(store(), windows=1)
    assert r.run() is None
    r.append(2)
    plan = Plan(((0, N + 2),), N + 2)
    if appended == "between":
        r.append(1)
        assert r.run(plan) is None
        assert r.s.windows[0].heads[1] == r.head
        r.boundary()
        assert r.finish() == [audit.Verdict(0, True)]
    else:
        r.s.seed()
        a = r.store.append(pages(r.store.extent, 1), salts(b"late", 1))
        assert r.s.append(a.root, a.count) == audit.Verdict(0, False, "order")


def test_the_chain_over_a_session_s_appends_and_each_pass_s_seed_is_its_coin() -> None:
    root, other = bytes(range(64)), bytes(range(64, 128))
    h1 = audit.append_head(None, N, 2, root)
    assert h1 == hashlib.sha512(b"verity/remote-bandwidth/appends/v0\0" + bytes(64) + N.to_bytes(8, "big")
                                + (2).to_bytes(8, "big") + root).digest()
    h2 = audit.append_head(h1, N + 2, 1, other)
    assert h2 == hashlib.sha512(b"verity/remote-bandwidth/appends/v0\0" + h1 + (N + 2).to_bytes(8, "big")
                                + (1).to_bytes(8, "big") + other).digest()
    st = store()
    assert st.head is None
    a, b = st.append(pages(N, 2), salts(b"a", 2)), st.append(pages(N + 2, 1), salts(b"b", 1))
    assert st.head == audit.append_head(audit.append_head(None, N, 2, a.root), N + 2, 1, b.root)
    for bad in ((None, N, 0, root), (None, -1, 1, root), (None, N, 1, root[:63]), (root[:63], N, 1, root)):
        with pytest.raises(ValueError):
            audit.append_head(*bad)

    r = Run(store(), windows=1)
    assert r.run() is None and r.run() is None
    opened = r.s.coins.opened(coins.PASS)
    assert [r.passes[0, i].seed for i in (0, 1)] == [coins.seed(o) for o in opened] == r.s.windows[0].seeds
    assert [o.after for o in opened] == [r.s.coins.window, r.passes[0, 0].root]
