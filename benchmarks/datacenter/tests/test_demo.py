"""The rack demo: an honest night is sound and accounts every feature; the adversary's five attempts are each refused."""
from __future__ import annotations

from datacenter.demo import night
from datacenter.observe import Deviation


def test_an_honest_night_on_a_rack_is_sound_and_located():
    acc = night()
    assert acc.sound
    assert acc.located == {"kv": ["n1.gpu0"], "wipe:n0.dpu": ["n0.dpu"], "wipe:n0.bmc": ["n0.bmc"]}
    assert acc.offramp_bits == {"telemetry": 1024} and acc.advice_bits == 3
    n0, n1 = acc.blocks
    assert n0.credited > 0 and n1.credited > 0 and n1.reads > 0
    assert n0.state < n1.state            # n0's DPU and BMC were wiped


def test_each_of_the_adversarys_attempts_is_refused_for_its_own_reason():
    acc = night(Deviation(extra={(("n0.dpu", "n0.cert"), 1): 2}, tampered={"image"}, moved={"kv": "n1.dram"},
                          failed_wipes={"n0.bmc"}, down={(("n1.dpu", "n1.cert"), 2)}))
    reasons = " | ".join(r.why for r in acc.refused)
    for why in ("no record", "occupancy differs", "registered digest", "past its deadline", "answer was wrong",
                "after the window"):
        assert why in reasons, why
