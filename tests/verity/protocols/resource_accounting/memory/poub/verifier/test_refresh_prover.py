"""The reference prover's store under refresh: a re-key is of a registered run or a fresh append's slot whole, and its
blocks name their origin's position; and with refresh off, the prover's bytes are as before.  Its sessions under
refresh are held to the Lean verifier (`lean/test_refresh_prover.py`)."""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
from test_appends import N, honest_session, pages, salts, store


def test_a_rekey_rejects_an_origin_outside_the_store_or_other_than_a_fresh_append_whole():
    s = store()
    a = s.append(pages(s.extent, 3), salts(b"page salts", 3), slot=0)
    s.rekey(("append", 0), 3, 1)
    s.rekey(("registered", 0), N, 2)
    for origin, count in ((("append", 1), 3), (("append", 0), 2), (("registered", N - 1), 2), (("registered", 0), 0)):
        with pytest.raises(ValueError, match="never of a re-key"):
            s.rekey(origin, count, 3)
    assert s.origin_position(a.first + a.count + 1) == a.first + 1 and s.origin_position(a.first) is None


def test_with_refresh_off_the_reference_prover_s_bytes_are_as_before():
    """`test_appends.honest_session`, which never re-keys under refresh, opened whole: its pass roots and openings
    hash to the digest they had before refresh existed, and no opening names an origin."""
    r = honest_session()
    h = hashlib.sha512()
    for (t, i), ps in sorted(r.passes.items()):
        h.update(ps.root)
        for q in range(ps.plan.length):
            o = r.prover.open(ps, q)
            assert o.origin_position is None
            h.update(repr(tuple(getattr(o, f.name) for f in dataclasses.fields(o) if f.name != "origin_position"))
                     .encode())
    assert all(a.origin is None and a.root == a.tree.commitment.root for a in r.store.appends)
    assert h.hexdigest()[:32] == OFF_DIGEST


OFF_DIGEST = "44f48586406025d6efb522a0729da090"
