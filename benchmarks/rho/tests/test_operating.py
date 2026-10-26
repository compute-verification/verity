import math

from rho import operating
from rho.params import Registry
from rho.scenario import evaluate_named

REG = Registry.load()


def test_the_reference_numbers_match_the_spec_arithmetic():
    # top's arithmetic on Daniel's spec: one 17 GB SparseLoCo sync takes about 4.3 years through one server's 1 kb/s;
    # the pretraining run takes about 35 s on the whole fleet unencumbered, about 58 min at a 1% dark pool
    r = evaluate_named(REG, "reference-onsite-h200")
    exfil = operating.exfil(r.scenario, REG["exfil.sparseloco-sync.bits"])
    assert math.isclose(exfil["one location"] / (365 * 86400), 4.31, rel_tol=0.01)
    work = operating.work(r, REG["work.v4-pretrain.flops"])
    assert 30 < work["unencumbered"] < 40
    assert math.isclose(work["dark pool"], work["unencumbered"] / 0.01)
    assert work["unencumbered"] <= work["guaranteed"] <= work["best attack"] * (1 + 1e-9) <= work["dark pool"] * 1.001


def test_remote_exfiltration_is_unbounded():
    r = evaluate_named(REG, "reference-remote-h200")
    assert operating.exfil(r.scenario, 1e12) == {"one location": 0.0, "every location": 0.0}


def test_every_reference_has_an_operating_point():
    r = evaluate_named(REG, "reference-onsite-nvl72")
    names = {p.name for p in operating.points(REG, r)}
    assert names == set(REG.names("work")) | set(REG.names("exfil"))


def test_human_units():
    assert operating.human(30) == "30 s" and operating.human(7200) == "2 h" and operating.human(math.inf) == "never"
