"""The rack demo: an honest night is sound and accounts every feature; each of the adversary's attempts is refused for its
own reason, and the refused windows are charged at line rate."""
from __future__ import annotations

from datacenter import Deviation, Port
from datacenter.demo import night


def test_an_honest_night_on_a_rack_is_sound_and_located():
    acc = night()
    assert acc.sound
    assert acc.located == {"kv": ["n1.gpu0"], "labels@n0.dpu": ["n0.dpu"], "labels@n0.bmc": ["n0.bmc"]}
    assert acc.input_bits == {"x": 32, "temperature": 16} and acc.advice_bits == 3
    n0, n1 = acc.blocks
    assert n0.credited > 0 and n1.credited > 0 and n1.reads > 0
    assert n0.carried < n1.carried            # n0's DPU and BMC were wiped


def test_each_of_the_adversarys_attempts_is_refused_for_its_own_reason():
    acc = night(Deviation(extra={(Port("n0.cert", "egress"), 5): 2}, moved={"kv": "n1.dram"}, failed_wipes={"n0.bmc"},
                          down={(Port("n1.cert", "ingress"), 2)}), image=0xBADC0DE)
    reasons = " | ".join(r.why for r in acc.refused)
    for why in ("no record", "occupancy differs", "not a committed crossing", "registered digest", "past its deadline",
                "spot check", "after the window"):
        assert why in reasons, why
    assert acc.refused_bits > 0 and acc.covert_bits > night().covert_bits
