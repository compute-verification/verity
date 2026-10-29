"""The rack demo: an honest night is sound and accounts every feature; each of the adversary's attempts is refused for its
own reason, the refused windows are charged at line rate, and a KV store read from host DRAM certifies no local reads."""
from __future__ import annotations

import pytest

from datacenter import Deviation, Port
from datacenter.demo import night
from datacenter.models import words


def test_an_honest_night_on_a_rack_is_sound_and_located():
    acc = night()
    assert acc.sound
    assert acc.located == {"kv": ["n1.gpu0"], "labels@n0.dpu": ["n0.dpu"], "labels@n0.bmc": ["n0.bmc"]}
    assert acc.input_bits == {"x": 512, "temperature": 16} and acc.advice_bits == 3
    n0, n1 = acc.blocks
    assert n0.credited == n1.credited == 530752 + 16586 and n1.reads > 0   # a linear's and a noised product's ANDs each
    assert n0.macs == n1.macs == {"mac:bf16": 1024 + 32}
    # n0's DPU and BMC state no write ceiling, so their wipes fill what the links into them carry in 0.2 ms
    assert n1.carried - n0.carried == pytest.approx((400e9 + 400e9 + 1e9 + 1e9) * 2e-4)


def test_under_salt_custody_each_nodes_noised_product_is_pinned_to_it():
    from datacenter import SALT_CUSTODY
    n0, n1 = night(assume=(SALT_CUSTODY,)).blocks
    assert (n0.custody, n1.custody) == (3 + 4 * 7, 3 + 4 * 7 + 16)    # advice, egress timing, n1's telemetry
    assert n0.local_at_least == n1.local_at_least == 14907 and n0.local_at_least < n0.credited   # the salt's cone


def test_each_of_the_adversarys_attempts_is_refused_for_its_own_reason():
    acc = night(Deviation(extra={(Port("n0.cert", "egress"), 5): 2}, moved={"kv": "n1.cpu0"}, failed_wipes={"n0.bmc"},
                          down={(Port("n1.cert", "ingress"), 2)}, swapped={"image": words("other image", 32)}))
    reasons = " | ".join(r.why for r in acc.refused)
    for why in ("no record", "occupancy differs", "not a committed crossing", "differ from the registration",
                "past its deadline", "spot check"):
        assert why in reasons, why
    honest = night()
    assert acc.refused_bits > 0 and acc.covert_bits > honest.covert_bits
    assert acc.blocks[1].reads_local_at_least == 0 < honest.blocks[1].reads_local_at_least   # KV read from host DRAM
