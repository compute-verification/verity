"""The reference prover under refresh: every credited chunk, registered and appended, re-keyed on the rota with its
origin, planned only under a key that joined fewer than L windows before and after its probation, and opened with its
origin's position, salt and path; and with refresh off, the prover's bytes are as before."""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
from poub_fakes import Entropy
from test_appends import N, honest_session, pages, salts, store

from verity.commitments import Commitment, verify_opening
from verity.commitments import Opening as MerkleOpening
from verity.protocols.resource_accounting.memory.poub.prover import audit_prover
from verity.protocols.resource_accounting.memory.poub.prover.rota import REKEY_MAX, Rota
from verity.protocols.resource_accounting.memory.poub.verifier import audit, codec, receipts

LIFETIME, WINDOWS, PASSES = 2, 7, 2


@dataclasses.dataclass(frozen=True)
class RefreshOpening(audit.Opening):
    """An opening with a re-key's origin position, which the verifier's opening carries under refresh."""

    origin_position: int | None = None


class Verifier:
    """The verifier's refresh rules, applied to what it receives: a re-key's origin is the rota's at the window it
    joins and its root is its declaration; a plan of window t holds a registered block only while t < L and an
    appended one only under a key with joined ≤ t < joined + L, its payload born at least L windows before; and an
    opening decodes under its block's key and coin to a payload whose leaf at its origin's position is under the
    origin's root."""

    def __init__(self, store: audit_prover.Store, rota: Rota) -> None:
        self.reg, self.rota, self.n = store.registration, rota, store.registration.blocks
        self.coins = Entropy(b"append coins")
        self.appends: list[dict] = []
        self.released: set[int] = set()
        self.extent = self.n

    def append(self, root: bytes, count: int, origin, window: int) -> tuple[int, bytes]:
        first, rota, life = self.extent, self.rota, self.rota.lifetime
        a = {"first": first, "count": count, "root": root, "origin": origin, "joined": window,
             "coin": self.coins(codec.COIN_BYTES), "ordinal": None}
        if origin is None:
            a["born"] = window
            a["ordinal"] = sum(x["origin"] is None for x in self.appends)
        else:
            assert count <= REKEY_MAX and root == audit_prover.rekey_root(self.reg.digest, first, count, origin)
            if origin[0] == "registered":
                c0, cn = rota.class_runs(window % life)
                assert window >= 1 and c0 <= origin[1] and origin[1] + count <= c0 + cn
                a["born"] = None
            else:
                k = self.appends[origin[1]]
                assert k["origin"] is None and origin[2] + count <= k["count"]
                assert window > k["joined"]
                assert (window - rota.first(k["joined"], rota.append_class(k["ordinal"]))) % life == 0
                a["born"] = k["joined"]
        self.appends.append(a)
        self.extent += count
        return first, a["coin"]

    def _of(self, j: int) -> dict:
        return next(a for a in self.appends if a["first"] <= j < a["first"] + a["count"])

    def release(self, indices) -> None:
        assert not self.released & set(indices)
        self.released |= set(indices)

    def receive(self, t: int, plan: receipts.Plan) -> None:
        life = self.rota.lifetime
        for j in plan:
            assert j not in self.released, (t, j)
            if j < self.n:
                assert t < life, (t, j)
                continue
            a = self._of(j)
            assert a["joined"] <= t < a["joined"] + life, (t, j, a["joined"])
            assert a["born"] is None or a["born"] + life <= t, (t, j, a["born"])

    def check(self, j: int, o: RefreshOpening) -> None:
        k = self.reg.key
        if j < self.n:
            payload, at = codec.decode(o.block, j, k), j
            assert o.origin_position is None
        else:
            a = self._of(j)
            payload = codec.decode(o.block, j, k, o.digest, a["coin"])
            origin = a["origin"]
            if origin is None:
                at = j
                assert o.origin_position is None
            else:
                start = origin[1] if origin[0] == "registered" else self.appends[origin[1]]["first"] + origin[2]
                at = start + j - a["first"]
                assert o.origin_position == at
        assert payload is not None and (j < self.n or codec.digest(payload) == o.digest)
        if at < self.n:
            domain, root = audit.payload_domain(self.reg.node, self.n), Commitment(self.reg.payload_root, self.n)
        else:
            a = self._of(at)
            domain = audit.append_domain(self.reg.digest, a["first"], a["count"])
            root = Commitment(a["root"], a["count"])
        leaf = MerkleOpening(at, receipts.hidden(payload, o.payload_salt), o.payload_path)
        assert verify_opening(domain, root, leaf, audit.PAYLOAD_SCHEMA, receipts.LIMITS)


def session(freed: dict[int, list[int]] | None = None):
    """A refresh session over the test store: W windows of L = 2, each opened by its due re-keys, a fresh append of
    pages in windows 0 and 1, and `PASSES` passes a window over the refresh plan; `freed` names the chunks freed at
    the start of a window.  Every receipt of every pass is opened and checked."""
    s, rota = store(), Rota(N, LIFETIME, WINDOWS)
    refresh, prover, v = audit_prover.Refresh(s, rota), audit_prover.Prover(s, None, entropy=Entropy(b"salts")), None
    v = Verifier(s, rota)
    plans, sent = [], []

    def send(a: audit_prover.Append, t: int) -> None:
        first, coin = v.append(a.root, a.count, a.origin, t)
        assert first == a.first
        if a.origin is not None:
            s.encode(first, coin)
        sent.append((a.origin, a.first, a.count))
        if old := refresh.appended(a, t):
            v.release(old)

    for t in range(WINDOWS):
        for p in (freed or {}).get(t, []):
            v.release([refresh.freed(p)])
        for origin, count in refresh.due(t):
            send(s.rekey(origin, count), t)
        if t < 2:
            count = 3 if t == 0 else 2
            send(s.append(pages(s.extent, count), salts(b"page salts %d" % s.extent, count)), t)
        for i in range(PASSES):
            plan = refresh.plan(t)
            v.receive(t, plan)
            ps = prover.pass_under(t, i, hashlib.sha512(b"seed %d %d" % (t, i)).digest(), plan)
            for q in range(plan.length):
                v.check(plan.block(q), prover.open(ps, q))
            plans.append((t, plan))
    return s, rota, refresh, plans, sent


@pytest.fixture
def refresh_opening(monkeypatch):
    monkeypatch.setattr(audit, "Opening", RefreshOpening)


def test_a_refresh_session_rekeys_every_chunk_on_the_rota_and_plans_it_only_under_a_live_key(refresh_opening):
    """Registered class c (blocks 4c … 4c + 3 at N = 8, L = 2) is re-keyed in the windows ≡ c mod 2 from 1; the pages
    of window 0 (append 0, fresh ordinal 0) in windows 2, 4, 6 and those of window 1 (append 2, after window 1's
    re-key; fresh ordinal 1) in 3, 5.  A page is first planned at its born + L, under its first re-key, and never
    under its own append's key."""
    s, rota, refresh, plans, sent = session()
    rekeys = [(o, c) for o, _, c in sent if o is not None]
    assert rekeys == [(("registered", 4), 4),
                      (("registered", 0), 4), (("append", 0, 0), 3), (("registered", 4), 4), (("append", 2, 0), 2),
                      (("registered", 0), 4), (("append", 0, 0), 3), (("registered", 4), 4), (("append", 2, 0), 2),
                      (("registered", 0), 4), (("append", 0, 0), 3)]
    by_window = {t: plan for t, plan in plans}
    assert by_window[0].runs == ((0, N),)
    pages0, pages1 = range(N, N + 3), range(N + 7, N + 9)
    for t, plan in plans:
        origins = {p if (p := s.origin_position(j)) is not None else j for j in plan}
        assert set(range(N)) <= origins, t
        assert (set(pages0) <= origins) == (t >= 2) and (set(pages1) <= origins) == (t >= 3), t
        assert not set(pages0) & set(plan) and not set(pages1) & set(plan)
    assert all(s.blocks[j] is None for j in [*pages0, *pages1])


def test_a_chunk_freed_is_not_rekeyed_and_a_freed_run_cuts_its_piece(refresh_opening):
    """Registered blocks 5 and 6 freed before window 3 leave class 1's re-key at 3 as the runs 4 and 7; a page freed
    before window 2 leaves its append's re-key as the other two pages."""
    s, _, _, plans, sent = session({2: [N + 1], 3: [5, 6]})
    rekeys = [(o, c) for o, _, c in sent if o is not None]
    assert rekeys[2:7] == [(("append", 0, 0), 1), (("append", 0, 2), 1), (("registered", 4), 1),
                           (("registered", 7), 1), (("append", 2, 0), 2)]
    for t, plan in plans:
        origins = {p if (p := s.origin_position(j)) is not None else j for j in plan}
        assert N + 1 not in origins and (t < 3 or not {5, 6} & origins)


def test_a_rekey_rejects_an_origin_outside_the_store_or_of_a_rekey(refresh_opening):
    s, _, _, _, _ = session()
    with pytest.raises(ValueError, match="never of a re-key"):
        s.rekey(("append", 4, 0), 1)
    with pytest.raises(ValueError, match="never of a re-key"):
        s.rekey(("registered", N - 1), 2)
    with pytest.raises(ValueError, match="1 … 512"):
        s.rekey(("registered", 0), 0)


def test_the_rules_refuse_a_wrong_origin_another_coin_and_a_page_under_its_own_key(refresh_opening):
    """The rules this file holds the prover to are not vacuous: a re-key's opening naming another origin position, or
    whose block is encoded under another coin, fails, and a plan holding a page under its own append's key fails."""
    s, rota = store(), Rota(N, LIFETIME, 4)
    v, prover = Verifier(s, rota), audit_prover.Prover(s, None, entropy=Entropy(b"salts"))
    a = s.rekey(("registered", 4), 4)
    first, coin = v.append(a.root, a.count, a.origin, 1)
    s.encode(first, coin)
    plan = receipts.Plan(((first, 4),), s.extent)
    ps = prover.pass_under(1, 0, bytes(64), plan)
    o = prover.open(ps, 1)
    v.check(first + 1, o)
    with pytest.raises(AssertionError):
        v.check(first + 1, dataclasses.replace(o, origin_position=6))
    other = codec.encode(s.payload(first + 1), first + 1, s.registration.key, bound=True, coin=bytes(32))
    with pytest.raises(AssertionError):
        v.check(first + 1, dataclasses.replace(o, block=other))
    page = s.append(pages(s.extent, 1), salts(b"page", 1))
    v.append(page.root, 1, None, 1)
    with pytest.raises(AssertionError):
        v.receive(2, receipts.Plan(((page.first, 1),), s.extent))


def test_with_refresh_off_the_reference_prover_s_bytes_are_as_before():
    """`test_appends.honest_session`, which never re-keys under refresh, opened whole: its pass roots and openings
    hash to the digest they had before refresh existed, and no opening names an origin."""
    r = honest_session()
    h = hashlib.sha512()
    for (t, i), ps in sorted(r.passes.items()):
        h.update(ps.root)
        for q in range(ps.plan.length):
            o = r.prover.open(ps, q)
            assert not hasattr(o, "origin_position") or o.origin_position is None
            h.update(repr(dataclasses.astuple(o)).encode())
    assert all(a.origin is None and a.root == a.tree.commitment.root for a in r.store.appends)
    assert h.hexdigest()[:32] == OFF_DIGEST


OFF_DIGEST = "44f48586406025d6efb522a0729da090"
