"""The lifecycle example's two composites over one monitoring period: honest, each accepts; each way the developer deviates
(`lifecycle.CHEATS` and `ONSITE_CHEATS`, at least one a phase) is refused by name or rejected by the component whose check it breaks, except
what the remote composite can't see, which goes uncredited."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from functools import cache
from typing import Any

import lifecycle as L
import numpy as np
import onsite
import pytest
import remote
from lifecycle import IDS

MODS = {"onsite": onsite, "remote": remote}

REFUSED = {"hardware": ("duplicate-id", "hardware registration"), "builder": ("builder", "registration"),
           "placement": ("placement-off-layout", "operation"), "advice-cap": ("advice-cap", "declaration"),
           "recommit": ("recommit", "declaration")}
"""Deviations the verifier refuses, ending the run: the refusal's name and the phase whose check it breaks."""
REJECTED = {"wrong-value": (("onsite", "remote"), 2, "sampled-proofs"), "unreported": (("onsite",), 2, "certifier"),
            "io-mismatch": (("onsite",), 2, "certifier"), "lost-theta": (("onsite",), 1, "pous"),
            "off-node": (("onsite",), 1, "bandwidth"), "foreign": (("onsite",), 1, "wipe")}
"""Deviations a component rejects: the composites they reach, the first declaration they touch, and the component."""


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
    assert set(L.CHEATS + L.ONSITE_CHEATS) == set(REFUSED) | set(REJECTED) | {"fork"}
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
    assert (first["after"], later["after"]) == (0, 1) and later["previous"] == first["head"] != later["head"]
    assert o["lines"][0]["layout"] == first["head"] and o["lines"][1]["layout"] == later["head"]
    assert "gpu2" not in o["lines"][0]["placement"] and "gpu2" in o["lines"][1]["placement"]


AUDITOR = {"onsite": onsite.AUDITOR, "remote": {}}
A = L.CARD_A.digest()
UPLINK9 = (L.Link("u9", ("node0", "cert9"), 1, 1), L.Link("u9-out", ("cert9", L.OUTSIDE), 1, 0))
ENTRIES = {
    "accepted": (None, L.LATER),
    "stale": ("previous", L.LATER),
    "gpu0-again": ("duplicate-id", L.Registration("", locs=(L.Loc("gpu0", "device", "node0", card=A),))),
    "no-parent": ("unknown-ref", L.Registration("", locs=(L.Loc("hbm9", "memory", "gpu9", tier="hbm", bytes=1),))),
    "device-at-site": ("kind", L.Registration("", locs=(L.Loc("gpu9", "device", "site0", card=A),))),
    "loop-link": ("kind", L.Registration("", links=(L.Link("l9", ("node0", "node0"), 1, 1),))),
    "not-the-sheet": ("card-digest", L.Registration("", cards=(replace(L.CARD_A, tc=1 << 20),))),
    "unregistered-card": ("card-digest", L.Registration("", locs=(L.Loc("gpu9", "device", "node0", card="00"),))),
    "over-hbm": ("capacity", L.Registration("", locs=(L.Loc("gpu9", "device", "node0", card=A),
                                                      L.Loc("hbm9", "memory", "gpu9", tier="hbm", bytes=(64 << 30) + 1)))),
    "empty-memory": ("capacity", L.Registration("", locs=(L.Loc("dram9", "memory", "node0", tier="dram"),))),
    "foreign-certifier": ("owner", L.Registration("", locs=(L.Loc("cert9", "certifier", "site0", "auditor", key="00"),),
                                                  links=UPLINK9)),
    "auditor-host": ("owner", L.Registration("", locs=(L.Loc("node9", "host", "site0", "auditor"),))),
    "bare-uplink": ("uncertified", L.Registration("", links=(L.Link("u9", ("node0", L.OUTSIDE), 1, 1),))),
    "unreached-host": ("challenger-reach", L.Registration("", locs=(L.Loc("node1", "host", "site0"),
                                                                    L.Loc("dram1", "memory", "node1", tier="dram", bytes=1)),
                                                          links=(L.Link("n01", ("node0", "node1"), 1, 1),))),
}
"""A later entry for each check `check_registration` makes, with the refusal it draws."""
REMOTE_ACCEPTS = {"bare-uplink", "unreached-host"}
"""The entries a remote site accepts: it has no certifier and no memory challenger to hold its links to."""


def _first(protocol: str) -> L.Layout:
    g = L.Layout.genesis(protocol)
    return L.check_registration(g, replace(MODS[protocol].LAYOUT, previous=g.head), AUDITOR[protocol])


@pytest.mark.parametrize("protocol", MODS)
@pytest.mark.parametrize("case", ENTRIES)
def test_check_registration(protocol: str, case: str) -> None:
    """Each check `check_registration` makes, on a later entry of each composite's site's log."""
    layout = _first(protocol)
    name, entry = ENTRIES[case]
    entry = replace(entry, previous=L.Layout.genesis(protocol).head if case == "stale" else layout.head)
    if name is None or (protocol == "remote" and case in REMOTE_ACCEPTS):
        assert L.check_registration(layout, entry, AUDITOR[protocol]).head == entry.digest()
        return
    with pytest.raises(L.Refusal) as e:
        L.check_registration(layout, entry, AUDITOR[protocol])
    assert e.value.name == name, e.value


def test_projections() -> None:
    """Each subprotocol's accounting is the layout's: the later entry's device adds its card's rate to PoUW's compute and
    its memory to PoUS's; onsite every boundary link is certified, remote none is."""
    on, rem = _first("onsite"), _first("remote")
    grown = on.add(replace(L.LATER, previous=on.head))
    a, b, r = on.accounting(1), grown.accounting(1), rem.accounting(1)
    assert b["pouw"]["tc"] - a["pouw"]["tc"] == L.CARD_B.tc and b["pous"]["hbm"] - a["pous"]["hbm"] == 128 << 30
    assert a["certifier"]["certified"] == a["certifier"]["boundary"] == ["uplink0"] and a["certifier"]["uncertified_bits"] == 0
    assert r["certifier"]["certified"] == [] and r["certifier"]["uncertified_bits"] == r["bandwidth"]["boundary_bits"]
    assert on.device("gpu0") == {"tc": 64, "bw": 3200, "chip": 1 << 25, "mem": 64 << 30}


@pytest.mark.parametrize(("p", "name"), [
    ({"units": ["gpu0", "gpu1"], "theta": "hbm0"}, None),
    ({"layout": "00", "units": ["gpu0"], "theta": "hbm0"}, "placement-off-layout"),
    ({"units": ["gpu9"], "theta": "hbm0"}, "placement-off-layout"),
    ({"units": ["cert0"], "theta": "hbm0"}, "placement-auditor"),
    ({"units": ["hbm0"], "theta": "hbm0"}, "placement-kind"),
    ({"units": ["gpu0"], "theta": "gpu1"}, "placement-kind"),
    ({"units": ["gpu0"], "theta": "tiny"}, "placement-capacity"),
])
def test_check_placement(p: dict[str, Any], name: str | None) -> None:
    on = _first("onsite")
    layout = L.check_registration(on, L.Registration(on.head, locs=(
        L.Loc("tiny", "memory", "node0", tier="dram", bytes=8),)), onsite.AUDITOR)
    p = {"layout": layout.head, **p}
    if name is None:
        L.check_placement(layout, p)
        return
    with pytest.raises(L.Refusal) as e:
        L.check_placement(layout, p)
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


def test_inference_only_policy() -> None:
    """Each declaration block divides into inference blocks, one per request, within the policy's budgets; a Γ below one
    request's gates, or an α below its fresh input per gate, makes the policy and so each composite reject."""
    line = outcome("remote")["lines"][0]
    blk = L.Builder(plan=L.PLAN).build(line["advice"])[0]
    d = L.Declaration(1, line["advice"], {"circuit": blk.program})
    a = L.inference_only(d, blk)
    assert a["verdict"] == "accept" and a["commitment"] == d.digest
    assert a["profile"]["blocks"] == [[19620, 48]] * blk.n
    for gamma, alpha, over in ((19619, L.ALPHA, "has more than Γ gates"),
                               (L.GAMMA, Fraction(47, 19620), "reads more than α bits of fresh input per gate")):
        out = L.policy(blk.n, gamma, (alpha.numerator, alpha.denominator))
        assert out["failure"] == [0, "size" if gamma < L.GAMMA else "nci"], out
    wrong = L.inference_only(L.Declaration(1, line["advice"], {"circuit": "00" * 64}), blk)
    assert wrong["verdict"] == "reject" and wrong["reason"] == "the record does not name the declared circuit"
    other = {**a, "commitment": L.Declaration(1, line["advice"], {"circuit": blk.program, "prev": "00"}).digest}
    assert other["commitment"] != a["commitment"]
    for protocol in MODS:
        _, ids = L.products()[protocol]
        assert IDS["inference-only"] in ids
        rejected = L.compose(protocol, {i: (a if i != IDS["inference-only"] else wrong) for i in ids})
        assert rejected["verdict"] == "reject" and IDS["inference-only"] in rejected["reason"]
        tied = L.compose(protocol, {i: (a if i != IDS["inference-only"] else other) for i in ids})
        assert tied["verdict"] == "reject" and tied["reason"].startswith(f"record: {IDS['inference-only']} committed ")


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
