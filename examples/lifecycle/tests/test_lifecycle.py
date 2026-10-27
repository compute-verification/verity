"""The lifecycle example's two composites over one monitoring period: honest, each accepts; each way the developer deviates
(`lifecycle.CHEATS`, at least one a phase) is refused by name or rejected by the component whose check it breaks, except
what the remote composite can't see, which goes uncredited."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import Any

import lifecycle as L
import numpy as np
import onsite
import pytest
import remote

from verity.core.protocols.sampled_proofs.parties import IDS

MODS = {"onsite": onsite, "remote": remote}

REFUSED = {"hardware": ("duplicate", "hardware registration"), "builder": ("builder", "registration"),
           "placement": ("placement", "operation"), "advice-cap": ("advice-cap", "declaration"),
           "recommit": ("recommit", "declaration")}
"""Deviations the verifier refuses, ending the run: the refusal's name and the phase whose check it breaks."""
REJECTED = {"wrong-value": (("onsite", "remote"), 2, "sampled-proofs"), "unreported": (("onsite",), 2, "certifier"),
            "io-mismatch": (("onsite",), 2, "certifier"), "lost-theta": (("onsite",), 1, "pous"),
            "off-node": (("onsite",), 1, "bandwidth")}
"""Deviations a component rejects: the composites they reach, the first segment they touch, and the component."""


@cache
def outcome(protocol: str, cheat: str | None = None) -> dict[str, Any]:
    return MODS[protocol].run(cheat)


def _accepts(o: dict[str, Any], protocol: str) -> None:
    product, ids = L.products()[protocol]
    assert o["status"] == "completed", o.get("refused")
    assert o["period"]["verdict"] == "accept", o["period"]["reason"]
    for line in o["lines"]:
        assert line["composite"]["product"] == product and line["composite"]["verdict"] == "accept"
        assert set(line["answers"]) == set(ids) and all(a["verdict"] == "accept" for a in line["answers"].values())
        assert line["real"] < line["units"] and len(line["draw"]["units"]) == line["draw"]["k"]
    assert o["period"]["head"] == o["lines"][-1]["record"]
    assert 0 < o["period"]["credited_ands"] < o["period"]["appended_ands"]
    assert o["period"]["advice_bits_charged"] == sum(line["advice_bits"] for line in o["lines"])


def test_onsite_period() -> None:
    _accepts(outcome("onsite"), "onsite")


def test_remote_period() -> None:
    _accepts(outcome("remote"), "remote")


def test_every_cheat_is_a_case() -> None:
    assert set(L.CHEATS) == set(REFUSED) | set(REJECTED) | {"fork"}
    assert {p for _, p in REFUSED.values()} | {"audit", "accounting"} == {
        "hardware registration", "registration", "operation", "declaration", "audit", "accounting"}


@pytest.mark.parametrize("protocol", MODS)
@pytest.mark.parametrize("cheat", sorted(REFUSED))
def test_refused(protocol: str, cheat: str) -> None:
    o = outcome(protocol, cheat)
    assert o["status"] == "refused" and o["refused"]["name"] == REFUSED[cheat][0], o.get("refused")
    assert "period" not in o


def test_hardware_registration_grows() -> None:
    o = outcome("onsite")
    first, later = o["hardware"]
    assert (first["after"], later["after"]) == (0, 1) and first["layout"] != later["layout"]
    assert o["lines"][0]["layout"] == first["layout"] and o["lines"][1]["layout"] == later["layout"]
    assert "gpu2" not in o["lines"][0]["placement"] and "gpu2" in o["lines"][1]["placement"]


def _layout(**kw: Any) -> L.Layout:
    return replace(L.Layout("onsite"), **kw)


@pytest.mark.parametrize(("registration", "name"), [
    (replace(L.LATER, site="onsite"), None),
    (_layout(devices=(L.Device("gpu0", "sim-int8-a", "node0"),)), "duplicate"),
    (_layout(devices=(L.Device("gpu9", "no-such-sku", "node0"),)), "data-sheet"),
    (_layout(stores=(L.Store("nvme9", "node9", 1),)), "node"),
    (_layout(links=(L.Link("l9", ("node0", "node9"), 1, 1),)), "link"),
    (_layout(nodes=("node1",), links=(L.Link("uplink1", ("node1", L.OUTSIDE), 1, 1),)), "uplink-certifier"),
    (_layout(nodes=("node1",), links=(L.Link("uplink1", ("node1", L.OUTSIDE), 1, 1),),
             auditors=(L.AuditorDevice("cert9", "certifier", "uplink1"),)), "auditor-device"),
    (replace(L.LATER, site="remote"), "site"),
])
def test_check_registration(registration: L.Layout, name: str | None) -> None:
    """Each check `check_registration` makes, on a later registration of the onsite site."""
    auditor = (onsite.CERT, onsite.CHAL)
    layout = L.check_registration(None, onsite.LAYOUT, auditor)
    if name is None:
        assert L.check_registration(layout, registration, auditor).devices[-1].name == "gpu2"
        return
    with pytest.raises(L.Refusal) as e:
        L.check_registration(layout, registration, auditor)
    assert e.value.name == name


@pytest.mark.parametrize(("protocol", "cheat"), [(p, c) for c, (ps, _, _) in REJECTED.items() for p in ps])
def test_rejected(protocol: str, cheat: str) -> None:
    _, k, component = REJECTED[cheat]
    o = outcome(protocol, cheat)
    assert o["status"] == "completed" and o["period"]["verdict"] == "reject"
    line = o["lines"][k - 1]
    assert line["answers"][IDS[component]]["verdict"] == "reject"
    assert line["composite"]["verdict"] == "reject" and IDS[component] in line["composite"]["reason"]
    assert o["period"]["credited_ands"] == 0


@pytest.mark.parametrize("cheat", ["unreported", "io-mismatch"])
def test_onsite_io_cheats_pass_sampled_proofs(cheat: str) -> None:
    """The committed values are consistent; only the certifier's view of the system's I/O catches the lie."""
    line = outcome("onsite", cheat)["lines"][1]
    assert line["answers"][IDS["sampled-proofs"]]["verdict"] == "accept"


@pytest.mark.parametrize("protocol", MODS)
def test_fork_rejected_in_accounting(protocol: str) -> None:
    o = outcome(protocol, "fork")
    assert all(line["composite"]["verdict"] == "accept" for line in o["lines"])
    assert o["period"]["verdict"] == "reject" and "not the previous record" in o["period"]["reason"]


def test_remote_unreported_is_uncredited() -> None:
    """Remote has no view of the system's I/O: work left out of the advice is accepted and credited less."""
    o, honest = outcome("remote", "unreported"), outcome("remote")
    assert o["period"]["verdict"] == "accept"
    assert o["period"]["credited_ands"] < honest["period"]["credited_ands"]


def test_circuit_is_the_model() -> None:
    rng = np.random.default_rng(5)
    cases = [(L.theta_of(5), [int(v) for v in rng.integers(-128, 128, L.D)]) for _ in range(6)]
    cases += [(L.Theta((127,) * L.D ** 2, (127,) * L.D ** 2), [127] * L.D),
              (L.Theta((127,) * L.D ** 2, (-128,) * L.D ** 2), [127] * L.D),
              (L.Theta((-128,) * L.D ** 2, (127,) * L.D ** 2), [127] * L.D)]
    state = (1 << L.STATE) - 3
    for theta, x in cases:
        assert L.evaluate_unit(theta, state, x) == L.forward(theta, state, x)
        state = L.forward(theta, state, x)[1]


def test_builder_is_prefix_extending() -> None:
    b = L.Builder(plan=L.PLAN)
    a1, a2 = b.encode([0, 0, 1]), b.encode([0, 1, 1])
    c1 = b.build(a1)
    assert b.build(a1 + a2)[:1] == c1 and b.grow(c1, a1, a2) == b.build(a1 + a2)[1]
    assert [p.device for p in c1[0].units] == ["gpu0", "gpu1", "gpu0"]
    assert [p.device for p in b.grow(c1, a1, a2).units] == ["gpu0", "gpu1", "gpu2"]
    for circuit, advice, a_k, name in ((c1, a1, "0" * (b.cap + 1), "advice-cap"), (c1, a1, a2[:-1], "advice"),
                                       (c1, a1, b.encode([1]), "advice"), (c1, a1, b.encode([0, 0, 0]), "advice"),
                                       (b.build(a2), a1, a2, "prefix")):
        with pytest.raises(L.Refusal) as e:
            b.grow(circuit, advice, a_k)
        assert e.value.name == name


def test_builder_descriptor_round_trips() -> None:
    b = L.Builder(plan=L.PLAN)
    assert L.Builder.of(b.descriptor()) == b
    for bad in ({**b.descriptor(), "format": "another/builder/v1"}, {**b.descriptor(), "plan": []},
                {**b.descriptor(), "plan": [[2, ["gpu0"]]]}, {"cap": "x"}):
        with pytest.raises(L.Refusal, match="builder"):
            L.Builder.of(bad)
