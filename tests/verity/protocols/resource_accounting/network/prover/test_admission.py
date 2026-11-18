"""The network attester's one entry on a served node (`admission.Admitter`), on a fake exchange: what it builds from its
configuration, and what it files and subscribes there."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from verity import hooks as H
from verity.protocols.resource_accounting.network.prover import admission as NA

TICK = {"bucket_us": 100_000, "sat_us": 30_000, "free_us": 40_000}
RUN = "ab"


class Exchange:
    """The adapter's exchange as a scheme sees it: ``request(name, request)``, and ``subscribe(name, hook, fn)``
    returning the removal."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, Any]] = []
        self.subscribers: list[tuple[str, str, Callable[[Any], None]]] = []

    def request(self, name: str, request: Any) -> None:
        self.requests.append((name, request))

    def subscribe(self, name: str, hook: str, fn: Callable[[Any], None]) -> Callable[[], None]:
        entry = (name, hook, fn)
        self.subscribers.append(entry)
        return lambda: self.subscribers.remove(entry)


def tick(**cfg: Any) -> NA.Admitter:
    q: list[Any] = []
    return NA.Admitter({"order": "tick", "hold_ms": 600_000, **TICK, **cfg}, submit=q.append, abort=q.append,
                       clock=lambda: 0.0)


def test_tick_builds_the_gate_and_the_frontend_and_files_both_under_the_name():
    a = tick()
    assert type(a.gate) is NA.TickGate and a.gate.tick == NA.Tick(**TICK) and a.gate.hold_s == 600.0
    assert all(callable(getattr(a.gate, m)) for m in ("add", "pending", "ready", "open"))       # a `hooks.Gate`
    assert isinstance(a.front, NA.TickAdmit) and a.front.take == a.gate.take
    ex = Exchange()
    assert a.install(ex, "n") is a
    (n0, gate), (n1, route) = ex.requests
    assert (n0, n1) == ("n", "n") and gate == H.StartOnBoundary(a.gate)
    assert route == H.ServeRoute(NA.WATERMARK_PATH, a.front.serve_watermark)
    assert ex.subscribers == [("n", "steps", a.gate.on)]


def test_a_lost_release_in_the_engine_is_refused_through_the_exchange():
    a, ex = tick(), Exchange()
    a.install(ex, "n")
    adm = SimpleNamespace(engine_id="e0", release=a.front.key({"x-verity-release": "0", "x-verity-release-run": RUN}))
    a.front.submit(adm)
    assert a.front.watermark({"x-verity-release-run": RUN}).feed(b"0 2 0\n")
    assert ex.requests[2:] == [("n", H.RefuseInput("e0", NA.LOST.format(n=0, run=RUN)))]


def test_a_rule_other_than_the_registered_one_is_a_rule_gate():
    assert type(tick(token_ns=0, catch_up="quiet").gate) is NA.TickGate
    g = tick(token_ns=500, catch_up="marks").gate
    assert type(g) is NA.RuleGate and (g.token_ns, g.catch_up) == (500, "marks")
    with pytest.raises(ValueError, match="only a benchmark arm that names it measures it"):
        tick(catch_up="open")


def test_without_submit_the_gate_alone_reads_its_records_from_pull():
    records: list[dict[str, Any]] = []
    pull = lambda: []                                                   # noqa: E731
    a = NA.Admitter({**TICK, "hold_ms": 1000}, pull=pull, log=records.append, clock=lambda: 0.0)
    assert a.front is None and a.gate.pull is pull and a.gate.log == records.append and a.gate.hold_s == 1.0
    ex = Exchange()
    a.install(ex, "n")
    assert ex.requests == [("n", H.StartOnBoundary(a.gate))] and ex.subscribers == [("n", "steps", a.gate.on)]


def test_release_order_makes_the_frontend_alone_and_files_nothing():
    q: list[Any] = []
    a = NA.Admitter({"order": "release-order", "hold_ms": 250}, submit=q.append, abort=q.append)
    assert a.gate is None and type(a.front) is NA.ReleaseOrder and a.front.hold_s == 0.25
    ex = Exchange()
    a.install(ex, "n")
    assert ex.requests == [] and ex.subscribers == []


def test_what_a_configuration_lacks_is_refused():
    with pytest.raises(ValueError, match=r"order one of \['release-order', 'tick'\], not 'fifo'"):
        NA.Admitter({"order": "fifo"})
    with pytest.raises(ValueError, match="frontend forwards each request through submit and abort"):
        NA.Admitter({"order": "release-order"})
    with pytest.raises(ValueError, match="frontend forwards each request through submit and abort"):
        NA.Admitter(TICK, submit=print)
    with pytest.raises(ValueError, match="tick bucket_us: a positive integer of microseconds"):
        NA.Admitter({"sat_us": 1, "free_us": 1})


def test_uninstall_removes_the_subscription_and_an_admission_installs_once():
    a, ex = tick(), Exchange()
    a.install(ex, "n")
    with pytest.raises(RuntimeError, match="is installed"):
        a.install(ex, "n")
    a.uninstall()
    assert ex.subscribers == [] and len(ex.requests) == 2


def test_the_name_is_the_network_attesters_declaration():
    from verity.protocols.resource_accounting.network.scheme import NETWORK_ATTESTER
    ex = Exchange()
    tick().install(ex)
    assert {n for n, _ in ex.requests} == {n for n, _, _ in ex.subscribers} == {NETWORK_ATTESTER.name}
