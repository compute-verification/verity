"""KeyBinding's game against the audit's order of appends and passes, the red team's KB-1: a KV page appended after
a pass's seed can hold that pass's sketches of other blocks, 64 bytes a block at 16 rows, and their receipts follow
from the page's bytes without reading them.  A pass's seed binds every append received before its root (`audit.chained`), so no block in its plan was
appended after its seed, and such a page would be a fixed point of SHA-512."""

from __future__ import annotations

import copy
import hashlib

import pytest

from test_appends import N, Entropy, Run, pages, salts, store
from verity.core.protocols.poub import audit, codec, coins, receipts
from verity.core.protocols.poub.receipts import Plan


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


def test_a_page_appended_after_a_pass_s_seed_relays_none_of_its_receipts() -> None:
    """The red team's order: pass 0 over the registered store, then a page holding pass 1's sketches of the
    registered blocks under pass 1's seed as it was before this rule, H(σ_0, 1, root_(0,0)), appended at N, then pass 1
    over all N + 1 blocks.  Under that seed the page relays all N registered receipts (the attack the red team measured,
    which the session accepted before this rule).  Pass 1's seed binds the page, so the relayed receipts are none of
    the pass's, a pass sent under the old seed fails its receipts' paths, and the honest pass over the page is accepted."""
    r = Run(store(), windows=1)
    assert r.run() is None
    root0, coin = r.passes[0, 0].root, coins.coin(r.challenges[0])
    unbound = receipts.seed(coin, 1, root0)
    a = r.store.append([carrier(r.store, unbound)], salts(b"page salts", 1))
    assert r.commit(a.root, a.count) == N
    plan = Plan(((0, N + 1),), r.store.extent)
    attack = relay(r, unbound, plan)
    assert attack.receipts == r.prover.pass_under(0, 1, unbound, plan).receipts

    bound = receipts.seed(coin, 1, audit.chained(root0, r.head))
    assert r.head == r.store.head and bound != unbound
    page = codec.decode(r.store.blocks[N], N, r.store.registration.key, a.digests[0], r.coins[N])
    assert not set(relayed(page, bound)) & set(r.prover.pass_under(0, 1, bound, plan).receipts)

    cheat = copy.deepcopy(r)
    assert cheat.s.receive(0, 1, attack.root, plan.runs, step=1) is None
    cheat.passes[0, 1] = attack
    cheat.boundary()
    verdict = cheat.finish()[0]
    assert not verdict.accepted and verdict.failure == "receipt-path" and verdict.at[0] == 1

    assert r.run(plan) is None and r.passes[0, 1].seed == bound
    r.boundary()
    assert r.finish() == [audit.Verdict(0, True)]


def test_a_page_appended_before_the_root_it_chains_from_relays_nothing_either() -> None:
    """Why the rule is the seed's and not the order of arrival: the prover computes pass 0 and holds its root,
    builds the page from H(σ_0, 1, root_(0,0)), appends it, and only then sends root_(0,0), so the page arrives before
    the root pass 1 chains from.  Every seed whose root comes after the append binds it, pass 0's too: the prover's
    pass 0, computed before the page existed, fails, and one that bound the page would need the page before its own
    root."""
    r = Run(store(), windows=1)
    pass0 = r.prover.run(r.challenges[0], 0, r.previous)
    unbound = receipts.seed(coins.coin(r.challenges[0]), 1, pass0.root)
    a = r.store.append([carrier(r.store, unbound)], salts(b"page salts", 1))
    assert r.commit(a.root, a.count) == N
    assert r.s.receive(0, 0, pass0.root, step=0) is None
    plan = Plan(((0, N + 1),), r.store.extent)
    attack = relay(r, unbound, plan)
    assert r.s.receive(0, 1, attack.root, plan.runs, step=1) is None
    r.passes = {(0, 0): pass0, (0, 1): attack}
    r.boundary()
    verdict = r.finish()[0]
    assert not verdict.accepted and verdict.failure == "receipt-path" and verdict.at[0] == 0


@pytest.mark.parametrize("computed", ["before", "after"])
def test_a_pass_s_seed_binds_every_append_sent_before_its_root(computed) -> None:
    """Pass 1 reads two pages; a third is appended and not read.  Computed before the third page's append, which
    then goes out ahead of its root, the pass fails its receipts' paths; computed after it, the pass is accepted."""
    r = Run(store(), windows=1)
    assert r.run() is None
    r.append(2)
    plan = Plan(((0, N + 2),), N + 2)
    if computed == "before":
        stale = r.prover.run(r.challenges[0], 1, r.previous, plan, head=r.head)
        r.append(1)
        assert r.s.receive(0, 1, stale.root, plan.runs, step=1) is None
        r.passes[0, 1] = stale
        r.boundary()
        verdict = r.finish()[0]
        assert not verdict.accepted and verdict.failure == "receipt-path" and verdict.at[0] == 1
    else:
        r.append(1)
        assert r.run(plan) is None
        r.boundary()
        assert r.finish() == [audit.Verdict(0, True)]


def test_the_chain_over_a_session_s_appends_and_a_session_without_appends_keeps_its_seeds() -> None:
    root, other = bytes(range(64)), bytes(range(64, 128))
    h1 = audit.append_head(None, N, 2, root)
    assert h1 == hashlib.sha512(b"verity/poub/appends/v0\0" + bytes(64) + N.to_bytes(8, "big")
                                + (2).to_bytes(8, "big") + root).digest()
    h2 = audit.append_head(h1, N + 2, 1, other)
    assert h2 == hashlib.sha512(b"verity/poub/appends/v0\0" + h1 + (N + 2).to_bytes(8, "big")
                                + (1).to_bytes(8, "big") + other).digest()
    assert audit.chained(root, None) == root
    assert audit.chained(root, h2) == hashlib.sha512(b"verity/poub/chained/v0\0" + root + h2).digest()
    st = store()
    assert st.head is None
    a, b = st.append(pages(N, 2), salts(b"a", 2)), st.append(pages(N + 2, 1), salts(b"b", 1))
    assert st.head == audit.append_head(audit.append_head(None, N, 2, a.root), N + 2, 1, b.root)
    for bad in ((None, N, 0, root), (None, -1, 1, root), (None, N, 1, root[:63]), (root[:63], N, 1, root)):
        with pytest.raises(ValueError):
            audit.append_head(*bad)
    with pytest.raises(ValueError):
        audit.chained(root[:63], h1)

    r = Run(store(), windows=1)
    assert r.run() is None and r.run() is None
    assert r.passes[0, 0].seed == receipts.seed(coins.coin(r.challenges[0]), 0, r.store.registration.digest)
    assert r.passes[0, 1].seed == receipts.seed(coins.coin(r.challenges[0]), 1, r.passes[0, 0].root)
