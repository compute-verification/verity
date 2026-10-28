"""K, the network certifiers' leak bound, as rho reads it (`rho.leak`): per link and per datacenter from
network-accounting's `leak_bounds`, at the declared class's own rate, and the serving block's certified channel at the
lesser of the certified ingress and its share of the datacenter bound."""

import dataclasses
import math

import pytest
from rho import envelopes, hardware, lean, leak
from rho.params import Registry
from rho.scenario import evaluate_named

REG = Registry.load()


def w63(**fields) -> envelopes.Onsite:
    r = envelopes.regime(REG, "onsite-w63")
    return dataclasses.replace(r, certifier=dataclasses.replace(r.certifier, **fields)) if fields else r


def ctx(fleet: str = "world-h200") -> envelopes.Context:
    f = hardware.fleet(REG, fleet)
    return envelopes.Context(f, hardware.location(REG, f.location), envelopes.policy(REG, "v4-pro-session"),
                             hardware.horizon(REG, "year"), hardware.units(REG))


def test_the_link_is_the_declared_class_at_its_own_rate():
    """W63's class (B 512, r 4, 4 lanes, C 47, IDB 17, D 27, KO 64) certifies 4 lanes × ⌈32 / 28⌉ requests a 3.2 s
    link-window, 2.5 a second: the rate network-accounting's tests take."""
    link = leak.link(w63().certifier)
    assert (link.scan.B, link.scan.r, link.scan.lanes, link.scan.C, link.scan.IDB, link.scan.D, link.scan.KO) == (
        512, 4, 4, 47, 17, 27, 64)
    assert (link.rate, link.T, link.tau) == (2.5, 32, 0.1)


@pytest.mark.parametrize("delta", [math.nan, 0.01])
def test_the_datacenter_sum_is_leak_bounds_datacenter(delta):
    """`bounds` takes the sum over identical links as `links` times one link at δ / links, what `leak_bounds.datacenter`
    gives on the list of them, and the per-link bound is `per_link`'s."""
    c = w63(leak_delta=delta).certifier
    lb, link, s = leak.calculator(), leak.link(c), 3600.0
    d = None if math.isnan(delta) else delta
    k = leak.bounds(c, 3, s)
    assert k.total == lb.datacenter([link] * 3, s, c.leak, d).bits
    assert k.per_link == lb.per_link(link.scan, link.rate, s, c.leak, d, T=link.T, tau=link.tau).bits
    assert k.datacenter == k.total and k.statement == lb.STATEMENTS["datacenter"]


def test_the_datacenters_own_bound_counts_where_it_is_tighter():
    sum_ = leak.bounds(w63().certifier, 10, 86400.0).total
    assert leak.bounds(w63(leak_datacenter=sum_ / 4).certifier, 10, 86400.0).datacenter == sum_ / 4
    assert leak.bounds(w63(leak_datacenter=sum_ * 4).certifier, 10, 86400.0).datacenter == sum_


def test_the_serving_block_reads_the_lesser_of_the_ingress_and_ks_share():
    """At W63 a location's share of the datacenter's K, 45.0 kbit/s in the registered reading's per-period form, is
    below the declared ingress, 91.1 kbit/s, so it sets the certified channel and the serving cut cites K's statement
    and the datacenter's composition. A datacenter's own bound below the sum lowers it further; without `leak` the
    ingress stands."""
    c, r, w = ctx(), w63(), 86400.0
    share = r.certifier.leak_share(c, w)
    assert share / w == pytest.approx(44_975, rel=1e-4) and r.certifier.prompts(c) == pytest.approx(91_075)
    side = c.location.hosts * (r.certifier.timing + r.certifier.side) * w
    assert r.leak_binds(c) and r.budget(c).inputs == pytest.approx(r.state(c) + share + side)
    assert {"NetTiming.Guarantees.PeriodLeakAccepted", "draft:datacenter-leak"} <= set(r.cites(c))
    assert lean.status("NetTiming.Guarantees.PeriodLeakAccepted") == "proved"
    own = w63(leak_datacenter=r.certifier.leak_bounds(c).total / 10)
    assert own.budget(c).inputs == pytest.approx(r.state(c) + share / 10 + side)
    off = w63(leak="")
    assert not off.leak_binds(c) and "draft:datacenter-leak" not in off.cites(c)
    assert off.budget(c).inputs == pytest.approx(r.state(c) + r.certifier.prompts(c) * w + side)


def test_k_only_tightens_the_bound():
    on = evaluate_named(REG, "onsite-w63")
    off = evaluate_named(REG.override({"regime.onsite-w63.leak": ""}), "onsite-w63")
    assert on.lo >= off.lo


def test_the_pooled_form_adds_its_delta():
    c = ctx()
    assert w63().certifier.delta(c) == (0.0, 0.0)
    assert w63(leak_delta=1e-6).certifier.delta(c) == (1e-6, 0.0)
    assert set(w63(leak_delta=1e-6).certifier.leak_cites()) == {"draft:horizon-leak-accepted", "draft:datacenter-leak"}


def test_k_needs_a_declared_class_and_a_reading():
    with pytest.raises(ValueError, match="declared"):
        envelopes.Certifier(leak="registered")
    with pytest.raises(ValueError, match="leak is one of"):
        dataclasses.replace(w63().certifier, leak="everything")
