"""The rack demo: an honest night is sound and accounts every feature, each node's compute at PoUW's proved bound; each of
the adversary's attempts is refused for its own reason, the refused windows are charged at line rate, and a KV store read from host DRAM certifies no local reads."""
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


def test_under_salt_custody_each_nodes_noised_product_is_pinned_to_it():
    from datacenter import SALT_CUSTODY
    n0, n1 = night(assume=(SALT_CUSTODY,)).blocks
    assert (n0.custody, n1.custody) == (3 + 4 * 7, 3 + 4 * 7 + 16)    # advice, egress timing, n1's telemetry
    assert n0.local_at_least == n1.local_at_least == 6 * 32 * 32 and n0.local_at_least < n0.credited


def test_each_nodes_compute_is_pouws_proved_bound_placed_by_attribution():
    from datacenter import FLEET_CLOSED
    from datacenter.demo import POUW_TICKS

    def compute(acc):
        return [(c.unit, c.work, c.grade, c.basis) for c in acc.credits if c.resource.startswith("compute")]

    assert compute(night(assume=(FLEET_CLOSED,))) == [("n0", POUW_TICKS, "fleet-closed", "proved"),
                                                      ("n1", POUW_TICKS, "fleet-closed", "proved")]
    honest = night()
    assert compute(honest) == [("rack", POUW_TICKS, None, "proved")] * 2       # no premise places it on a block
    assert [b.busy for b in honest.blocks] == [POUW_TICKS / 16] * 2


def test_each_of_the_adversarys_attempts_is_refused_for_its_own_reason():
    acc = night(Deviation(extra={(Port("n0.cert", "egress"), 5): 2}, moved={"kv": "n1.cpu0"}, failed_wipes={"n0.bmc"},
                          down={(Port("n1.cert", "ingress"), 2)}), image=0xBADC0DE)
    reasons = " | ".join(r.why for r in acc.refused)
    for why in ("no record", "occupancy differs", "not a committed crossing", "registered digest", "past its deadline",
                "spot check"):
        assert why in reasons, why
    honest = night()
    assert acc.refused_bits > 0 and acc.covert_bits > honest.covert_bits
    assert acc.blocks[1].reads_local_at_least == 0 < honest.blocks[1].reads_local_at_least   # KV read from host DRAM
