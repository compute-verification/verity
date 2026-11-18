"""PoUB's two declarations as resource schemes (`verity.protocols.resource_accounting.memory.poub.scheme`)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from verity import hooks as H
from verity.protocols.compose import schemes
from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.memory.poub import scheme as poub
from verity.protocols.resource_accounting.memory.poub.prover import binder
from verity.protocols.resource_accounting.memory.poub.verifier import audit, p2_parties, period


def test_each_is_registered_under_its_one_name() -> None:
    assert schemes.get("bandwidth") is poub.BANDWIDTH and schemes.get("poub") is poub.POUB


def test_bandwidth_is_the_p2_verifiers_component() -> None:
    assert (p2_parties.COMPONENT, p2_parties.PROTOCOL) == (poub.BANDWIDTH.name, poub.BANDWIDTH.protocol)
    [timed] = [o for o in poub.BANDWIDTH.log if o.within_ns is not None]
    assert timed.within_ns == p2_parties.SESSION["deadline_ns"] and timed.direction == "after"
    assert timed.assumption in poub.BANDWIDTH.assumptions
    assert poub.BANDWIDTH.hooks == () and dict(poub.BANDWIDTH.settings) == {}


def test_poub_is_the_remote_sessions_and_the_binders_name() -> None:
    assert poub.POUB.protocol == audit.PROTOCOL_ID == period.PROTOCOL
    assert binder.NAME == poub.POUB.name and poub.POUB.hooks == ("memory", "steps")
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
        assert isinstance(scheme.blocks, S.Lacks) and "BlockOpeningCheck_v2" in scheme.blocks.why
        assert "BlockOpeningCheck_v2" in scheme.requires["one-design"]
        assert S.credited([scheme], [("Dither_v1", [(0, 4)])]) == frozenset()


def test_neither_term_is_said_and_each_names_its_lean_constant() -> None:
    zero = {name: 0 for name in poub.CHARGES}
    assert sorted(poub.DELTA_SCHEME) == ["bandwidth", "poub"]
    assert isinstance(poub.POUB.delta_scheme, S.Lacks) and all(c.bounded for c in poub.POUB.charges)
    with pytest.raises(ValueError, match="delta_scheme .* is not said"):
        S.bound(poub.POUB, Fraction(1, 100), 0, zero)
    assert not any(c.bounded for c in poub.BANDWIDTH.charges)
    with pytest.raises(ValueError, match="nothing bounds yet"):
        S.bound(poub.BANDWIDTH, Fraction(1, 100), 0, zero)


def test_refresh_is_poubs_one_setting_off_by_default() -> None:
    assert S.settings(poub.POUB, {}) == {"refresh": False} == S.settings(poub.POUB, {"refresh": False})
    with pytest.raises(ValueError, match=f"poub: refresh waits on {poub.REFRESH_GOAL}"):
        S.settings(poub.POUB, {"refresh": True})
    with pytest.raises(ValueError, match="has no setting"):
        S.settings(poub.POUB, {"lifetime": 2})
    with pytest.raises(ValueError, match="is a bool"):
        S.settings(poub.POUB, {"refresh": 1})


def test_neither_says_its_term_and_says_why() -> None:
    for s in (poub.POUB, poub.BANDWIDTH):
        assert isinstance(s.delta_scheme, S.Lacks) and "δs" in s.delta_scheme.why
    with pytest.raises(LookupError, match="bandwidth attaches through no hook"):
        poub.BANDWIDTH.attach(None, {})
    with pytest.raises(LookupError, match="poub declares no record: no combined record"):
        S.declared(poub.POUB, "record")
