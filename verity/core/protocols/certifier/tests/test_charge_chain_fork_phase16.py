"""REDTEAM phase 16, target 4 against #3801 (cursor/enc-charge-chain-061a @bcae38a4a3): ChargeLink_v1's carry across
windows, where window w's row 0 is window w-1's row 1 under the same salt, window 0's row 0 is the book's priors under a
public salt, and `check_chain` refuses a break (CH-rows / CH-first / CH-carry).

The PR's own `test_a_reset_carry_is_refused` covers a restart before window 1. These add the two angles the lead named
beside it, and the boundary the chain draws:

- a FORK, two windows claiming one carry-out: within a session a window that reuses an earlier window's carry-out
  (skipping the real predecessor's) is refused by CH-carry; a second session forked from a mid-chain carry-out (rather
  than the priors) is refused by CH-first. The chain is a line anchored at the priors, so neither fork is representable.
- a RESTART placed deeper, at window 2 of a three-window chain, is refused by CH-carry wherever it sits.
- a SKIPPED window is the chain's boundary: `check_chain` binds carry CONTINUITY, not the window COUNT (a truncated
  session, and an empty one, pass) nor the per-window charge (a no-op window whose carry-out equals its carry-in chains
  cleanly). So dropping a window, or a window that charges nothing while it served, is caught only by the caller pinning
  the sealed-window count and by `ChargeLink`/`ChargeCount`'s `en` binding -- not by the chain. (check_chain has no
  caller in the tree yet.)
"""
from __future__ import annotations

import json
from pathlib import Path

from verity.core.protocols.certifier import arith_encode as AE, charge_chain as CC
from verity.core.protocols.sampled_proofs.one_stage import produced
from verity.ml.commitments import hm96

FIXTURE = Path(__file__).parent / "fixtures" / "arith_encode_vectors.json"
PROGRAM, REGISTRATION = "charge-chain-test", "ab" * 64


def _ints(s):
    return [int(c) for c in s]


def _two_windows():
    """The async-kv-pressure fixture stream's two windows: N, book, [(en, bits)]."""
    for s in json.loads(FIXTURE.read_text())["streams"]:
        if s["source"].startswith("async-kv-pressure") and "decisions" in s:
            dec = s["decisions"]
            return dec["cut"]["N"], dec["book"], [(_ints(en), _ints(bits)) for en, bits, _l in dec["cut"]["windows"]]
    raise AssertionError("no async-kv-pressure stream")


def _commit(row0, row1, salt0, salt1=None):
    return produced.commit(PROGRAM, CC.VALUE, [CC.row(row0), CC.row(row1)], REGISTRATION,
                           [salt0, salt1 if salt1 is not None else hm96.SHA512.fresh_salt()], CC.WORD_BITS, CC.SCHEMA)


def _honest():
    """A valid chain: window 0 from the priors, window 1 from window 0's carry-out, and a third honest continuation."""
    N, book, wins = _two_windows()
    ses = CC.session(N, book, wins)
    v0 = CC.commit_window(PROGRAM, book, CC.carry_of(ses[0][2]), REGISTRATION, None)
    v1 = CC.commit_window(PROGRAM, book, CC.carry_of(ses[1][2]), REGISTRATION, v0)
    # a third window continuing honestly from v1 (its carry-out value is immaterial to the structural chain)
    v2 = CC.commit_window(PROGRAM, book, CC.carry_of(ses[1][2]), REGISTRATION, v1)
    assert CC.check_chain([CC.openings(v) for v in (v0, v1, v2)], PROGRAM, book) is None
    return book, v0, v1, v2


def test_a_fork_reusing_a_carry_out_within_a_session_is_refused():
    """Two windows claiming one carry-out: window 2's carry-in reuses window 0's carry-out, skipping window 1's. The
    chain is a line (window w's carry-in must be window w-1's carry-out), so this is refused by CH-carry."""
    book, v0, v1, _v2 = _honest()
    # a window whose carry-in row/salt are window 0's carry-out: it "claims" v0's carry-out as its own carry-in
    fork = _commit(CC.unrow(v0.rows[1]), CC.unrow(v1.rows[1]), v0.salts[1])
    assert CC.openings(fork)[0] == CC.openings(v0)[1]                       # it does reuse v0's carry-out opening
    got = CC.check_chain([CC.openings(v0), CC.openings(v1), CC.openings(fork)], PROGRAM, book)
    assert got == "CH-carry: window 2's carry-in isn't window 1's carry-out"


def test_a_second_session_forked_from_a_mid_chain_carry_out_is_refused():
    """The other fork: a fresh session that starts from a mid-chain carry-out rather than the book's priors. Every
    session must begin at the priors under their public salt (CH-first), so a carry-out can't seed a new session."""
    book, v0, v1, v2 = _honest()
    # start a session at window 1 (its carry-in is v0's carry-out, not the priors)
    assert CC.check_chain([CC.openings(v1), CC.openings(v2)], PROGRAM, book) == \
        "CH-first: the session's first carry isn't the book's priors under their public salt"


def test_a_restart_placed_at_window_two_is_refused():
    """A restart deeper in the chain: window 2's carry-in reset to the book's priors (the cheap undercount). Refused by
    CH-carry, whether the reset row sits under a fresh salt or under the public start salt."""
    book, v0, v1, _v2 = _honest()
    prior = CC.encode(AE.first_carry(book))
    for salt in (hm96.SHA512.fresh_salt(), CC.start_salt(PROGRAM)):
        reset = _commit(prior, CC.unrow(v1.rows[1]), salt)
        got = CC.check_chain([CC.openings(v0), CC.openings(v1), CC.openings(reset)], PROGRAM, book)
        assert got == "CH-carry: window 2's carry-in isn't window 1's carry-out"


def test_the_chain_binds_continuity_not_the_count_or_the_per_window_charge():
    """The chain's boundary. `check_chain` accepts a truncated session (one window, or none), so a dropped window is
    not its to catch -- the sealed-window count is the caller's. And a no-op window, whose carry-out equals its
    carry-in, chains cleanly, so a window that charged nothing while it served is caught by `ChargeLink`/`ChargeCount`'s
    `en` binding, not by the chain."""
    book, v0, v1, v2 = _honest()
    ops = [CC.openings(v) for v in (v0, v1, v2)]
    assert CC.check_chain(ops, PROGRAM, book) is None
    assert CC.check_chain(ops[:1], PROGRAM, book) is None and CC.check_chain([], PROGRAM, book) is None  # count not bound
    # a no-op window between v0 and its continuation: carry-out row == carry-in row (it charges nothing)
    noop = _commit(CC.unrow(v0.rows[1]), CC.unrow(v0.rows[1]), v0.salts[1])
    after = _commit(CC.unrow(noop.rows[1]), CC.unrow(v1.rows[1]), noop.salts[1])
    assert CC.check_chain([CC.openings(v0), CC.openings(noop), CC.openings(after)], PROGRAM, book) is None
    assert CC.unrow(noop.rows[0]) == CC.unrow(noop.rows[1])                 # the no-op: carry-out is its carry-in
