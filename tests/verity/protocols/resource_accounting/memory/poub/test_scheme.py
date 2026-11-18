"""PoUB's two declarations as resource schemes (`verity.protocols.resource_accounting.memory.poub.scheme`)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from verity import hooks as H
from verity.protocols.compose import schemes
from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.memory.poub import scheme as poub
from verity.protocols.resource_accounting.memory.poub.prover import binder
from verity.protocols.resource_accounting.memory.poub.verifier import audit, p2_parties


def test_each_is_registered_under_its_one_name() -> None:
    assert schemes.get("bandwidth") is poub.BANDWIDTH and schemes.get("poub") is poub.POUB


def test_bandwidth_is_the_p2_verifiers_component() -> None:
    assert (p2_parties.COMPONENT, p2_parties.PROTOCOL) == (poub.BANDWIDTH.name, poub.BANDWIDTH.protocol)
    [timed] = [o for o in poub.BANDWIDTH.log if o.within_ns is not None]
    assert timed.within_ns == p2_parties.SESSION["deadline_ns"] and timed.direction == "after"
    assert timed.assumption in poub.BANDWIDTH.assumptions
    assert poub.BANDWIDTH.hooks == () and dict(poub.BANDWIDTH.settings) == {}


def test_poub_is_the_remote_sessions_and_the_binders_name() -> None:
    assert poub.POUB.protocol == audit.PROTOCOL_ID
    assert binder.NAME == poub.POUB.name and poub.POUB.hooks == ("declared", "memory", "steps")
    assert set(poub.POUB.processes) == {poub.SESSION_PROCESS} == {"poub-session"}
    assert S.PROCESSES[poub.SESSION_PROCESS] is poub.POUB.processes[poub.SESSION_PROCESS]
    assert poub.prover("binder") is binder.Binder and set(poub.PROVER) == {"binder", poub.SESSION_PROCESS}
    assert poub.POUB.unit == poub.BANDWIDTH.unit == "bits"


def test_the_binder_subscribes_under_the_declaration() -> None:
    class Exchange:
        def __init__(self) -> None:
            self.requests, self.subscribed = [], []

        def request(self, name, request) -> None:
            self.requests.append((name, request))

        def subscribe(self, name, hook, fn):
            self.subscribed.append((name, hook))
            return lambda: None

    ex = Exchange()
    b = poub.POUB.attach(ex, {"store": None, "binder": {"passes": False}})
    assert isinstance(b, binder.Binder) and not b.passes
    assert ex.requests == [("poub", poub.REUSE_OLDEST_FREED)]
    assert ex.subscribed == [("poub", "memory"), ("poub", "steps")]
    ex = Exchange()
    binder.Binder(None, fifo=False).install(ex)
    assert ex.requests == []


def test_the_sessions_requests_refuse_a_run_it_cannot_follow() -> None:
    assert poub.POUB.requests({}) == (poub.REUSE_OLDEST_FREED,) and poub.BANDWIDTH.requests({}) == ()
    filed = poub.POUB.requests({"fifo": False, "session": True, "ranges": 8})
    assert filed == (*poub.SESSION, H.FixRanges(8))
    ok = {"world": 1, "prefix_cache": False, "async_steps": False}
    named = [("poub", r) for r in filed]
    assert H.unmet(named, ok) == []
    for key, bad in (("world", 2), ("prefix_cache", True), ("async_steps", True), ("async_steps", None)):
        [(who, request)] = H.unmet(named, {**ok, key: bad})
        assert who == "poub" and request.property == key


def test_no_block_is_declared_and_the_requirement_says_where_the_opening_unit_is() -> None:
    for scheme in (poub.BANDWIDTH, poub.POUB):
        assert scheme.blocks == () and "BlockOpeningCheck_v2" in scheme.requires["one-design"]
        assert S.credited([scheme], [("Dither_v1", [(0, 4)])]) == frozenset()


def test_each_says_its_term_as_the_sketch_bounds_error_at_its_layout_and_its_two_sides() -> None:
    """Each term is the Lean's `δSketch g b Q P N εrank` at its receipts' layout, evaluated exactly at the four
    parameters a caller gives; the sides are the credited bits over the tier's reads, the factor (1 − γ) cleared."""
    zero = {name: 0 for name in poub.CHARGES}
    lean = "ResourceAccounting.Guarantees.Remote.δSketch"
    assert {n: (t.lean, t.parameters) for n, t in poub.DELTA_SCHEME.items()} == {
        "bandwidth": (f"{lean} 96 257", ("Q", "P", "N", "rank_tail")),
        "poub": (f"{lean} 216 513", ("Q", "P", "N", "rank_tail"))}
    for scheme in (poub.BANDWIDTH, poub.POUB):
        assert scheme.delta_scheme is poub.DELTA_SCHEME[scheme.name]
        assert (scheme.over, scheme.under) == (poub.OVER, poub.UNDER) and "n·P·N·b" in scheme.over
    at = {"Q": 2**80, "P": 6, "N": 2**20, "rank_tail": Fraction(1, 2**100)}
    q = 2**80 + 6
    for name, (g, b) in {"bandwidth": (96, 257), "poub": (216, 513)}.items():
        assert poub.DELTA_SCHEME[name].value(**at) == (
            Fraction(2**80, 2**g) + q * (Fraction(1, 2**100) + Fraction(2**20, 2**512)) + Fraction(q * q, 2**b))
    assert all(c.bounded for c in poub.POUB.charges)
    got = S.bound(poub.POUB, Fraction(1, 100), 0, zero, at, epsilon=Fraction(0))
    assert got.delta_scheme == poub.DELTA_SCHEME["poub"].value(**at) and 0 < got.delta_scheme < Fraction(1, 2**19)
    with pytest.raises(ValueError, match=r"\['rank_tail'\] were not given"):
        S.bound(poub.POUB, Fraction(1, 100), 0, zero, {k: v for k, v in at.items() if k != "rank_tail"}, epsilon=Fraction(0))
    assert not any(c.bounded for c in poub.BANDWIDTH.charges)
    with pytest.raises(ValueError, match="nothing bounds yet"):
        S.bound(poub.BANDWIDTH, Fraction(1, 100), 0, zero)


def test_refresh_is_poubs_one_setting_off_by_default() -> None:
    assert poub.settings({}) == {"refresh": False} == poub.settings({"refresh": False})
    with pytest.raises(NotImplementedError, match=poub.REFRESH_GOAL):
        poub.settings({"refresh": True})
    with pytest.raises(ValueError, match="has no setting"):
        poub.settings({"lifetime": 2})
    with pytest.raises(ValueError, match="is a bool"):
        poub.settings({"refresh": 1})
