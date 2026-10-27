"""The lifecycle example's remote composite over one monitoring period: honest, it accepts; each way the developer deviates
(`lifecycle.CHEATS`, at least one a phase) is refused by name or rejected by the component whose check it breaks, except
work left out of the declared advice, which the remote composite can't see and doesn't credit."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import Any

import lifecycle as L
import numpy as np
import pytest
import remote

from verity.core.protocols.sampled_proofs.parties import IDS

REFUSED = {"hardware": ("duplicate", "hardware registration"), "builder": ("builder", "registration"),
           "placement": ("placement", "operation"), "advice-cap": ("advice-cap", "declaration"),
           "recommit": ("recommit", "declaration")}
"""Deviations the verifier refuses, ending the run: the refusal's name and the phase whose check it breaks."""
REJECTED = {"wrong-value": (2, "sampled-proofs")}
"""Deviations a component rejects: the first declaration they touch, and the component."""


@cache
def outcome(cheat: str | None = None) -> dict[str, Any]:
    return remote.run(cheat)


def test_remote_period() -> None:
    o = outcome()
    product, ids = L.products()["remote"]
    assert o["status"] == "completed", o.get("refused")
    assert o["period"]["verdict"] == "accept", o["period"]["reason"]
    for line in o["lines"]:
        assert line["composite"]["product"] == product and line["composite"]["verdict"] == "accept"
        assert set(line["answers"]) == set(ids) and all(a["verdict"] == "accept" for a in line["answers"].values())
        assert line["real"] < line["units"] and len(line["draw"]["units"]) == line["draw"]["k"]
    assert o["period"]["head"] == o["lines"][-1]["record"]
    assert 0 < o["period"]["credited_ands"] < o["period"]["appended_ands"]
    assert o["period"]["advice_bits_charged"] == sum(line["advice_bits"] for line in o["lines"])


def test_every_cheat_is_a_case() -> None:
    assert set(L.CHEATS) == set(REFUSED) | set(REJECTED) | {"unreported", "fork"}
    assert {p for _, p in REFUSED.values()} | {"audit", "accounting"} == {
        "hardware registration", "registration", "operation", "declaration", "audit", "accounting"}


@pytest.mark.parametrize("cheat", sorted(REFUSED))
def test_refused(cheat: str) -> None:
    o = outcome(cheat)
    assert o["status"] == "refused" and o["refused"]["name"] == REFUSED[cheat][0], o.get("refused")
    assert "period" not in o


def test_hardware_registration_grows() -> None:
    o = outcome()
    first, later = o["hardware"]
    assert (first["after"], later["after"]) == (0, 1) and first["layout"] != later["layout"]
    assert o["lines"][0]["layout"] == first["layout"] and o["lines"][1]["layout"] == later["layout"]
    assert "gpu2" not in o["lines"][0]["placement"] and "gpu2" in o["lines"][1]["placement"]


def _layout(**kw: Any) -> L.Layout:
    return replace(L.Layout("remote"), **kw)


UPLINK = L.Link("uplink1", ("node1", L.OUTSIDE), 1, 1)


@pytest.mark.parametrize(("layout", "registration", "name"), [
    (remote.LAYOUT, replace(L.LATER, site="remote"), None),
    (remote.LAYOUT, _layout(devices=(L.Device("gpu0", "sim-int8-a", "node0"),)), "duplicate"),
    (remote.LAYOUT, _layout(devices=(L.Device("gpu9", "no-such-sku", "node0"),)), "data-sheet"),
    (remote.LAYOUT, _layout(stores=(L.Store("nvme9", "node9", 1),)), "node"),
    (remote.LAYOUT, _layout(links=(L.Link("l9", ("node0", "node9"), 1, 1),)), "link"),
    (remote.LAYOUT, _layout(nodes=("node1",), links=(UPLINK,), auditors=(L.AuditorDevice("cert9", "certifier", "uplink1"),)),
     "auditor-device"),
    (remote.LAYOUT, replace(L.LATER, site="onsite"), "site"),
    (None, L.Layout("onsite", ("node1",), links=(UPLINK,)), "uplink-certifier"),
])
def test_check_registration(layout: L.Layout | None, registration: L.Layout, name: str | None) -> None:
    """Each check `check_registration` makes: on a later registration of the remote site, and an onsite site's first
    registration with an uplink no network certifier sits on."""
    so_far = layout and L.check_registration(None, layout)
    if name is None:
        assert L.check_registration(so_far, registration).devices[-1].name == "gpu2"
        return
    with pytest.raises(L.Refusal) as e:
        L.check_registration(so_far, registration)
    assert e.value.name == name


@pytest.mark.parametrize("cheat", sorted(REJECTED))
def test_rejected(cheat: str) -> None:
    k, component = REJECTED[cheat]
    o = outcome(cheat)
    assert o["status"] == "completed" and o["period"]["verdict"] == "reject"
    line = o["lines"][k - 1]
    assert line["answers"][IDS[component]]["verdict"] == "reject"
    assert line["composite"]["verdict"] == "reject" and IDS[component] in line["composite"]["reason"]
    assert o["period"]["credited_ands"] == 0


def test_fork_rejected_in_accounting() -> None:
    o = outcome("fork")
    assert all(line["composite"]["verdict"] == "accept" for line in o["lines"])
    assert o["period"]["verdict"] == "reject" and "not the previous record" in o["period"]["reason"]


def test_remote_unreported_is_uncredited() -> None:
    """Remote has no view of the system's I/O: work left out of the advice is accepted and credited less."""
    o, honest = outcome("unreported"), outcome()
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
