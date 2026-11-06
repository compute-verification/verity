"""The append coin: an appended block's bound key takes a coin the challenger releases after the append's root arrives,
b_j(c_a) = H(seed ‖ j ‖ d_j ‖ c_a), so no choice the prover makes before committing the append's digests can depend on
the blocks' encodings.  Each coin is live, committed before the session and hidden until it opens, once, after its
append's root: a retry never resamples one, and a coin no root follows is a failure in the record."""

from __future__ import annotations

import hashlib

import pytest

from test_appends import N, Run, honest_session, pages, salts, store
from verity.core.protocols.remote_bandwidth import audit, codec, coins
from verity.core.protocols.sampled_proofs.challenger import CoinError


def test_an_honest_session_opens_one_live_coin_an_append_and_each_page_decodes_under_its_own() -> None:
    """The honest session's two appends open the challenger's append coins 0 and 1, each after its append's root.
    Every page decodes under its digest and its own append's coin, under neither the pre-coin form
    (`codec.encode(..., bound=True)` with no coin) nor the other append's coin, and every window is accepted."""
    r = Run(store())
    assert r.s.coins.appends == 0
    r.append(4)
    assert r.s.coins.appends == 1
    r = honest_session()
    assert r.s.coins.appends == 2 and set(r.coins) == {N, N + 4}
    a, b = r.coins[N], r.coins[N + 4]
    assert a != b and len(a) == len(b) == codec.COIN_BYTES
    assert [coins.coin(o) for o in r.s.coins.opened(coins.APPEND)] == [a, b]
    assert r.s.coins.afters(coins.APPEND) == tuple(x[2].root for x in r.s._appends)
    assert not {a, b} & {coins.coin(o) for o in r.s.coins.opened(coins.CHALLENGE) + r.s.coins.opened(coins.OPENINGS)}
    assert [x[3] for x in r.s._appends] == [a, b]
    k = r.store.registration.key
    for j, own, other in [(j, a, b) for j in range(N, N + 4)] + [(j, b, a) for j in range(N + 4, N + 6)]:
        d = r.store.committed(j)[2]
        assert codec.decode(r.store.blocks[j], j, k, d, own) == pages(j, 1)[0]
        assert codec.decode(r.store.blocks[j], j, k, d, other) is None
        assert codec.decode(r.store.blocks[j], j, k, d) is None
    assert r.finish() == [audit.Verdict(0, True), audit.Verdict(1, True)]


@pytest.mark.parametrize("early", ["pre-coin", "guessed", "reused"])
def test_a_prover_that_encodes_an_append_before_its_coin_is_rejected(early) -> None:
    """A prover that encodes its pages before the verifier's coin for them arrives: under the pre-coin bound form,
    under a coin it picked itself, or under the coin of an earlier append.  Its receipts are of those blocks, but each
    decodes under no key the verifier holds for it, so the first draw on a page is the session's `codec` rejection."""
    st = store()
    r = Run(st, windows=1)
    first = r.append(1)
    assert r.run(change=(((N, 1),), ())) is None
    a = st.append(pages(N + 1, 2), salts(b"early", 2))
    k = st.registration.key
    if early == "pre-coin":
        del st.pending[a.first]
        st.blocks[a.first:] = [codec.encode(m, a.first + u, k, bound=True) for u, m in enumerate(pages(N + 1, 2))]
    else:
        st.encode(a.first, r.coins[first.first] if early == "reused" else hashlib.sha512(b"picked").digest()[:32])
    assert r.commit(a.root, a.count) == N + 1 and r.coins[N + 1] != r.coins[N]
    assert r.run(change=(((N + 1, 2),), ())) is None
    r.boundary()
    assert r.finish() == [audit.Verdict(0, False, "codec", (1, N + 1))]


def test_an_append_sent_again_draws_no_coin_and_rejects_the_session() -> None:
    """A retry never resamples: the same append root sent a second time, as a prover that wanted another coin would,
    is the session's `coin` rejection, and the challenger opens nothing for it."""
    r = Run(store())
    a = r.append(2)
    assert r.s.append(a.root, a.count) == audit.Verdict(0, False, "coin")
    assert r.s.coins.appends == 1 and r.s.extent == N + 2
    with pytest.raises(audit.SessionError, match="coin"):
        r.s.append(bytes(64), 1)


def test_an_append_no_root_follows_is_a_failure_at_the_last_close() -> None:
    """An append's coin is used once a root follows it: the append joins that root's window.  One after window 0's
    last root joins window 1; one after the session's last root never joins a window, and closing the last window
    is the session's `coin` rejection, before its openings coin is released."""
    r = Run(store(), windows=2)
    assert r.run() is None
    r.boundary()
    r.append(1)
    assert r.run(change=(((N, 1),), ())) is None
    r.append(1)
    r.boundary()
    r.clock.now = r.s.deadline(1) + 1
    assert r.s.close(0) and r.s.verify(0, [r.prover.open(r.passes[0, i], q) for i, q in r.s.windows[0].draws]).accepted
    assert r.s.close(1) == audit.Verdict(1, False, "coin")
    assert r.s.coins.appends == 2 and r.s.windows[1].draws is None and r.s.outcome == "coin"
    assert r.s.coins.opened(coins.OPENINGS)[0].index == 0 and len(r.s.coins.opened(coins.OPENINGS)) == 1


def test_a_session_that_reserves_no_append_coins_has_no_append_family_and_refuses_an_append() -> None:
    """A session reserves only the coins it will use (`coins.families`): with no append coins its layout has no append
    family, an append is the session's `coin` rejection with nothing opened, and asking its coins for one is refused."""
    r = Run(store(), appends=0)
    assert r.s.commitment.families == ((coins.CHALLENGE, 2), (coins.OPENINGS, 2))
    assert r.s.append(hashlib.sha512(b"page").digest(), 1) == audit.Verdict(0, False, "coin")
    assert r.s.coins.appends == 0 and r.s.coins.opened(coins.APPEND) == () and r.s.extent == N
    with pytest.raises(CoinError, match="no append family"):
        r.s.coins.append(bytes(64))


def test_an_append_after_the_last_window_s_roots_are_final_draws_no_coin() -> None:
    """Once the last window's roots are final no root can follow an append, so its coin could never be used: the
    append is the session's `coin` rejection, and the challenger opens nothing."""
    r = Run(store(), windows=1)
    assert r.run() is None
    r.boundary()
    r.clock.now = r.s.deadline(0) + 1
    assert r.s.close(0)
    assert r.s.append(hashlib.sha512(b"late").digest(), 1) == audit.Verdict(0, False, "coin")
    assert r.s.coins.appends == 0 and r.s.extent == N
