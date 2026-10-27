import math

from rho import operating
from rho.params import Registry
from rho.scenario import build, evaluate_named

REG = Registry.load()


def test_the_reference_numbers_match_the_spec_arithmetic():
    # one 17 GB SparseLoCo sync takes about 4.3 years through one server's 1 kb/s (the side channel alone; the
    # certifier's timing channel adds its own); the pretraining run takes about 35 s on the whole fleet unencumbered at
    # the training MFU, and the dark pool takes that over its share (0.1%)
    sync = REG["exfil.sparseloco-sync.bits"]
    alone = evaluate_named(REG.override({"regime.onsite.timing": 0.0}), "reference-onsite-h200")
    assert math.isclose(operating.exfil(alone.scenario, sync)["one location"] / (365 * 86400), 4.31, rel_tol=0.01)
    r = evaluate_named(REG, "reference-onsite-h200")
    both = REG["regime.onsite.timing"] + REG["regime.onsite.side"]
    assert math.isclose(operating.exfil(r.scenario, sync)["one location"], sync / both)
    work = operating.work(r, REG["work.v4-pretrain.flops"])
    assert 30 < work["unencumbered"] < 40
    assert math.isclose(work["dark pool"], work["unencumbered"] / r.scenario.context.fleet.eps_dark)
    assert math.isclose(work["lower bound"], r.scenario.seconds(r.lo))
    assert math.isclose(work["best attack"], r.scenario.seconds(r.attack))
    assert work["unencumbered"] <= work["lower bound"] <= work["best attack"] * (1 + 1e-9) <= work["dark pool"] * 1.001


def test_remote_exfiltration_is_unbounded():
    r = evaluate_named(REG, "reference-remote-h200")
    assert operating.exfil(r.scenario, 1e12) == {"one location": 0.0, "every location": 0.0}


def test_a_rack_exfiltrates_through_every_host():
    h200 = evaluate_named(REG, "reference-onsite-h200").scenario
    rack = evaluate_named(REG, "reference-onsite-nvl72").scenario
    assert math.isclose(operating.exfil(h200, 1e9)["one location"] / operating.exfil(rack, 1e9)["one location"],
                        rack.context.location.hosts / h200.context.location.hosts)


def test_every_reference_has_an_operating_point():
    r = evaluate_named(REG, "reference-onsite-nvl72")
    names = {p.name for p in operating.points(REG, r)}
    assert names == set(REG.names("work")) | set(REG.names("exfil"))


def test_faulty_outputs_carry_every_reference_before_their_first_detection():
    # from tiers' sampled-proof defaults at delta = 1e-6: the channel carries 1.06e17 bits before its first detection, and catches the weights with chance 8.33e-4, one
    # SparseLoCo sync 1.77e-5 and 1,000 syncs 1.76%
    row = operating.faulty(REG, build(REG, "reference-remote-h200-sampled"))
    assert math.isclose(row.q, 2.0833e-15, rel_tol=1e-4)
    assert math.isclose(row.capacity, 1.0610e17, rel_tol=1e-3)
    assert set(row.detect) == set(REG.names("exfil"))
    assert all(REG[f"exfil.{n}.bits"] < row.capacity for n in row.detect)
    for name, p in (("weights-fp4", 8.330e-4), ("sparseloco-sync", 1.7708e-5), ("sparseloco-1000", 0.017552)):
        assert math.isclose(row.detect[name], p, rel_tol=1e-3)


def test_faulty_outputs_capacity_is_where_detection_reaches_one_minus_delta():
    for identities in (1, 3):
        f = operating.Faulty(2.0833e-15, 1e-6, 16, 1, identities)
        assert math.isclose(1 - f.detect(f.capacity / identities), f.delta, rel_tol=1e-6)
        assert f.detect(1e8) < f.detect(1e12) < f.detect(1e16) < 1


def test_faulty_outputs_are_the_same_in_either_regime_and_undetected_unmonitored():
    onsite = operating.faulty(REG, build(REG, "reference-onsite-nvl72"))
    assert operating.faulty(REG, build(REG, "reference-remote-h200")) == onsite
    # every unit proved: the first faulty step is drawn, so the channel carries J ln(1/δ) (1 + floor) steps' bits
    assert onsite.q == 1 and math.isclose(onsite.capacity, math.log(1e6) * 2 * 16)
    unmonitored = operating.faulty(REG, build(REG, "unmonitored-h200-kimi"))
    assert unmonitored.capacity == math.inf and set(unmonitored.detect.values()) == {0.0}


def test_human_units():
    assert operating.human(30) == "30 s" and operating.human(7200) == "2 h" and operating.human(math.inf) == "never"
