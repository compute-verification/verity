"""The network attester's declaration as a resource scheme (`verity.protocols.resource_accounting.network.scheme`)."""

from __future__ import annotations

import pytest

from verity import hooks as H
from verity.circuits.defs import REGISTRY, CompositeDefinition
from verity.protocols.compose import schemes
from verity.protocols.interface.charge import charge_chain, window_charge
from verity.protocols.resource_accounting.network import scheme as NS
from verity.protocols.resource_accounting.network.prover.response_tie import ResponseTieStop


def test_registered_and_found_by_name() -> None:
    assert schemes.get("network-attester") is NS.NETWORK_ATTESTER
    assert NS.NETWORK_ATTESTER.unit == "bits"


def test_hooks_are_a_served_runs() -> None:
    assert set(NS.NETWORK_ATTESTER.hooks) <= set(H.HOOKS)


def test_every_block_is_a_registered_definition() -> None:
    ids = [b.definition for b in NS.NETWORK_ATTESTER.blocks]
    assert len(set(ids)) == len(ids)
    for d, tid in zip(NS.DEFINITIONS, ids, strict=True):
        assert isinstance(d, CompositeDefinition) and REGISTRY.defs[tid] is d


def test_its_own_term_is_sha512s_collision_bound_at_the_phases_calls() -> None:
    from fractions import Fraction

    from verity.protocols.resource_accounting import scheme as S

    names = {c.name: 0 for c in NS.NETWORK_ATTESTER.charges}
    got = S.bound(NS.NETWORK_ATTESTER, Fraction(1, 100), 0, names, {"q": 2**40}, epsilon=Fraction(0))
    assert got.term == "Claims.ecrBound" and got.delta_scheme == Fraction(2**40, 2**256)
    with pytest.raises(ValueError, match="were not given"):
        S.bound(NS.NETWORK_ATTESTER, Fraction(1, 100), 0, names, {}, epsilon=Fraction(0))
    assert NS.NETWORK_ATTESTER.over.startswith("bits: ") and NS.NETWORK_ATTESTER.under == "bits: zero; the attester credits nothing"


def test_cost_is_output_bits() -> None:
    cost = NS.NETWORK_ATTESTER.blocks[NS.DEFINITIONS.index(ResponseTieStop)].cost
    assert cost({"O": 4}) == 176


def test_the_charge_circuits_are_the_interfaces() -> None:
    ids = {b.definition for b in NS.NETWORK_ATTESTER.blocks}
    assert ids.isdisjoint({charge_chain.ChargeLink.id, window_charge.ChargeCount.id, window_charge.ChargeLink.id})
    assert [c.name for c in NS.NETWORK_ATTESTER.charges][0] == "advice"


def test_ports_are_its_blocks() -> None:
    ids = {b.definition for b in NS.NETWORK_ATTESTER.blocks}
    assert {(p.direction, p.definition) for p in NS.NETWORK_ATTESTER.ports} == {
        ("input", "IngressScanStart_v1"), ("output", "EgressBucket_v1")}
    assert {p.definition for p in NS.NETWORK_ATTESTER.ports} <= ids


class _Exchange:
    def __init__(self) -> None:
        self.filed: list[tuple[str, object]] = []

    def request(self, name: str, request: object) -> None:
        self.filed.append((name, request))

    def subscribe(self, name: str, hook: str, fn: object):
        self.filed.append((name, hook))
        return lambda: None


def test_requests_are_the_id_relay_unless_admission_is_configured() -> None:
    ex = _Exchange()
    NS.NETWORK_ATTESTER.attach(ex, {})
    assert ex.filed == [("network-attester", NS.START_ID_RELAY)]
    assert NS.NETWORK_ATTESTER.requests({}) == (NS.START_ID_RELAY,) and NS.NETWORK_ATTESTER.requests({"settings": {"order": "release-order"}}) == ()
    assert set(NS.NETWORK_ATTESTER.processes) == {"id-relay"}


def test_settings_are_the_admissions_with_their_defaults_and_refuse_what_admission_would() -> None:
    assert NS.settings({}) == NS.SETTINGS and NS.settings({"order": "release-order"})["hold_ms"] == 5000
    tick = {"order": "tick", "sat_us": 30_000, "free_us": 40_000}
    assert NS.settings(tick)["bucket_us"] == 100_000
    for bad in ({"feed": "f"}, {"hold_ms": 1.5}, {"order": "fifo"}, {"hold_ms": 0}, {"order": "tick"}):
        with pytest.raises(ValueError):
            NS.settings(bad)


class _Hooks(_Exchange):
    def __init__(self) -> None:
        super().__init__()
        self.fns: list = []

    def subscribe(self, name: str, hook: str, fn):
        self.fns.append(fn)
        return super().subscribe(name, hook, fn)

    def emit(self, ev: object) -> None:
        for fn in self.fns:
            fn(ev)


def test_the_front_feeds_the_engine_in_another_process(tmp_path) -> None:
    from verity.protocols.resource_accounting.network.prover import admission as A
    settings = {"bucket_us": 1000, "sat_us": 100, "free_us": 10}
    front, eng = _Hooks(), _Hooks()
    NS.attach(front, {"process": H.FRONT, "feed": str(tmp_path), "settings": settings})
    admitter = NS.attach(eng, {"process": H.ENGINE, "feed": str(tmp_path), "settings": settings})
    assert any(isinstance(r, H.ServeRoute) and r.path == A.WATERMARK_PATH for _, r in front.filed)
    rel = {A.RELEASE_HEADER: "0", A.RUN_HEADER: "r", A.BUCKET_HEADER: "3"}
    front.emit(H.InputArrived("a", 0, 0, rel, "POST /v1/completions HTTP/1.1"))
    front.emit(H.InputArrived("b", 0, 0, {**rel, A.RELEASE_HEADER: "1"}, "POST /v1/completions HTTP/1.1"))
    front.emit(H.InputArrived("c", 0, 0, rel, "POST /v1/completions HTTP/1.1"))
    front.emit(H.InputArrived("d", 0, 0, {}, "POST /v1/completions HTTP/1.1"))
    front.emit(H.OutputsComplete("b", 0))
    front.emit(H.OutputsComplete("a", 7))
    assert [r for _, r in front.filed if isinstance(r, H.RefuseInput)] == [
        H.RefuseInput("c", f"{A.RELEASE_HEADER}: 0 came twice in run 'r'")]
    assert A.FeedReader(A.FrontFeed(str(tmp_path)).path)() == [
        {"key": "a", "run": "r", "n": 0, "bucket": 3}, {"key": "b", "run": "r", "n": 1, "bucket": 3},
        {"run": "r", "n": 1, "passed": True}]
    assert isinstance(admitter, A.Admitter) and admitter.gate is not None and admitter.front is None


@pytest.mark.parametrize("lanes,outputs,every", [(4, 64, 16), (2, 8, 4)])
def test_the_scan_states_port_lays_its_row_out_as_the_scan_does(lanes, outputs, every) -> None:
    from verity.protocols.resource_accounting import scheme as S
    from verity.protocols.resource_accounting.network.prover import ingress_scan as IS

    shape = {"B": 512, "r": 4, "lanes": lanes, "C": 47, "IDB": 17, "EVERY": every, "WITHIN": 12, "D": 12 + every - 1,
             "KO": outputs, "KI": 1024}
    cls = IS.Class(**shape)
    got = S.layout(NS.NETWORK_ATTESTER, "scan-state", shape)
    assert got.bits == IS.state_bits(cls) and dict(got.fields) == IS.layout(cls)
    assert dict(got.counts) == {"K0": cls.K0, "K": cls.K}
