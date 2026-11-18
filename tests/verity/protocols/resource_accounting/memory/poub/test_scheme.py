"""PoUB's two declarations as resource schemes (`verity.protocols.resource_accounting.memory.poub.scheme`)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from verity import hooks as H
from verity.protocols.compose import schemes
from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.memory.poub import scheme as poub
from verity.protocols.resource_accounting.memory.poub.prover import binder
from verity.protocols.resource_accounting.memory.poub.prover.node_store import registry
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
    assert binder.NAME == poub.POUB.name and poub.POUB.hooks == ("declared", "memory", "steps")
    assert set(poub.POUB.processes) == {poub.SESSION_PROCESS} == {"poub-session"}
    assert S.PROCESSES[poub.SESSION_PROCESS] is poub.POUB.processes[poub.SESSION_PROCESS]
    assert poub.prover("binder") is binder.Binder and poub.prover("schedule") is binder.Every
    assert poub.prover("store") is registry.NodeMemory and poub.prover("tiles") is audit.NoiseTiles
    assert set(poub.PROVER) == {"binder", "store", "schedule", "tiles", "weights", "region", poub.SESSION_PROCESS}
    assert poub.POUB.unit == poub.BANDWIDTH.unit == "bits"


class Exchange:
    def __init__(self) -> None:
        self.requests, self.subscribed = [], []

    def request(self, name, request) -> None:
        self.requests.append((name, request))

    def subscribe(self, name, hook, fn):
        self.subscribed.append((name, hook))
        return lambda: None


def ctx(given=None, **launcher):
    return {"settings": S.settings(poub.POUB, given or {}), **launcher}


def test_the_binder_subscribes_under_the_declaration() -> None:
    ex = Exchange()
    b = poub.POUB.attach(ex, ctx({"passes": False}))
    assert isinstance(b, binder.Binder) and not b.passes
    assert ex.requests == [("poub", poub.REUSE_OLDEST_FREED)]
    assert ex.subscribed == [("poub", "memory"), ("poub", "steps")]
    ex = Exchange()
    binder.Binder(None, fifo=False).install(ex)
    assert ex.requests == []


def test_the_binder_holds_what_the_settings_say_over_a_store_read_back_from_it() -> None:
    """The attach makes the store when the launcher hands none, and it is read back as the binder's ``store``; a
    launcher's store is the one the binder holds; each setting reaches the binder, ``weight_blocks`` 0 as no limit."""
    b = poub.POUB.attach(Exchange(), ctx())
    assert isinstance(b.store, registry.NodeMemory)
    assert (b.passes, b.kv, b.weight_blocks, b.messages, b.tiles, b.reader) == (True, True, None, None, None, None)
    assert b.weights is binder.model_weights
    store = registry.NodeMemory()
    b = poub.POUB.attach(Exchange(), ctx({"passes": False, "kv": False, "weight_blocks": 5}, store=store))
    assert b.store is store and (b.passes, b.kv, b.weight_blocks) == (False, False, 5)


def test_a_window_is_the_fixed_schedule_with_the_launchers_challenge() -> None:
    reader = lambda segments: bytes(64)  # noqa: E731
    b = poub.POUB.attach(Exchange(), ctx({"window": 2}, challenge=lambda t: bytes([t]) * 32, reader=reader))
    assert isinstance(b.messages, binder.Every) and b.messages.passes == 2 and isinstance(b.tiles, audit.NoiseTiles)
    assert b.messages() == [bytes(32)] and b.messages() == [] and b.messages() == [b"\x01" * 32]
    assert b.reader is reader and isinstance(b.receipt, binder.PoubReceipt)
    with pytest.raises(ValueError, match="poub: window 2 with no challenge"):
        poub.POUB.attach(Exchange(), ctx({"window": 2}))
    with pytest.raises(ValueError, match="poub: window 0 with a challenge"):
        poub.POUB.attach(Exchange(), ctx(challenge=lambda t: bytes(32)))


def test_a_launchers_regions_follow_the_registered_values_and_a_wrong_shape_is_refused_by_name() -> None:
    codec = object()
    b = poub.POUB.attach(Exchange(), ctx(regions=lambda values: [("w.store", "T", "enc:w@0", codec)]))
    got = b.weights([("w", "t")])
    assert got == [binder.WeightRegion("w", "t", "param:w"), binder.WeightRegion("w.store", "T", "enc:w@0", codec=codec)]
    for wrong in (("w.store", "T", "enc:w@0"), ["w.store", "T", "enc:w@0", codec], (1, "T", "enc:w@0", codec),
                  ("w.store", "T", None, codec), binder.WeightRegion("w", "t", "param:w")):
        b = poub.POUB.attach(Exchange(), ctx(regions=lambda values, wrong=wrong: [("ok", "T", "s", codec), wrong]))
        with pytest.raises(ValueError, match=r"poub: regions' entry 1 is .*\(name, tensor, source, codec\)"):
            b.weights([])


@pytest.mark.parametrize("name, value", [("binder", {"passes": False}), ("passes", False), ("kv", False),
                                         ("weight_blocks", 5), ("window", 2), ("refresh", False)])
def test_a_setting_handed_beside_the_settings_is_refused_by_name(name, value) -> None:
    with pytest.raises(ValueError, match=rf"poub: \['{name}'\] handed beside the settings"):
        poub.POUB.attach(Exchange(), ctx(**{name: value}))


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


def test_each_says_its_term_as_the_sketch_bounds_error_at_its_layout_and_its_two_sides() -> None:
    """Each term is the Lean's `δSketch g b Q P N εrank` at its receipts' layout, evaluated exactly at the four
    parameters a caller gives; the sides are the route form's, naturals in bits with no factor."""
    zero = {name: 0 for name in poub.CHARGES}
    lean = "ResourceAccounting.Guarantees.Remote.δSketch"
    assert {n: (t.lean, t.parameters) for n, t in poub.DELTA_SCHEME.items()} == {
        "bandwidth": (f"{lean} 96 257", ("Q", "P", "N", "rank_tail")),
        "poub": (f"{lean} 216 513", ("Q", "P", "N", "rank_tail"))}
    for scheme in (poub.BANDWIDTH, poub.POUB):
        assert scheme.delta_scheme is poub.DELTA_SCHEME[scheme.name]
        assert "Remote.Run.committed" in scheme.over and "Remote.Run.paid" in scheme.under
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


def test_poubs_settings_are_refresh_and_what_its_binder_holds_each_refused_by_name() -> None:
    defaults = {"refresh": False, "passes": True, "kv": True, "weight_blocks": 0, "window": 0}
    assert S.settings(poub.POUB, {}) == defaults == S.settings(poub.POUB, dict(defaults)) == dict(poub.SETTINGS)
    assert S.settings(poub.POUB, {"passes": False, "kv": False, "weight_blocks": 5}) == {
        **defaults, "passes": False, "kv": False, "weight_blocks": 5}
    assert S.settings(poub.POUB, {"window": 2})["window"] == 2
    with pytest.raises(ValueError, match=f"poub: refresh waits on {poub.REFRESH_GOAL}"):
        S.settings(poub.POUB, {"refresh": True})
    with pytest.raises(ValueError, match="has no setting"):
        S.settings(poub.POUB, {"lifetime": 2})
    for name, wrong, kind in (("refresh", 1, "bool"), ("passes", 0, "bool"), ("kv", "no", "bool"),
                              ("weight_blocks", True, "int"), ("weight_blocks", None, "int"), ("window", 2.0, "int")):
        with pytest.raises(ValueError, match=f"poub.{name} is a {kind}"):
            S.settings(poub.POUB, {name: wrong})
    for name in ("weight_blocks", "window"):
        with pytest.raises(ValueError, match=f"poub: {name} is a count, 0 or more, not -1"):
            S.settings(poub.POUB, {name: -1})
    with pytest.raises(ValueError, match="poub: window 2 with passes off"):
        S.settings(poub.POUB, {"window": 2, "passes": False})


def test_the_remote_session_keeps_a_weight_derived_copy_and_the_bandwidth_audit_keeps_none() -> None:
    assert S.declared(poub.POUB, "keeps_weight_copy") is True
    assert S.declared(poub.BANDWIDTH, "keeps_weight_copy") is False


def test_what_each_lacks_says_why() -> None:
    with pytest.raises(LookupError, match="bandwidth attaches through no hook"):
        poub.BANDWIDTH.attach(None, {})
    with pytest.raises(LookupError, match="poub declares no record: no combined record"):
        S.declared(poub.POUB, "record")
