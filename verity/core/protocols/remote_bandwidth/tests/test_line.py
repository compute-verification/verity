"""The memory line held over time (`line`) in exact rationals: a stretch's span, L, rebuilds, fill and both readings
as `RemoteHeld.lean`'s forms charge them, each form's rate and burst from the node's, the worst stretch the first on a
tie, every refusal, and the JSON form the Lean reads, W the pooled bound of the record's windows."""

from __future__ import annotations

from fractions import Fraction

import pytest

from verity.core.protocols.remote_bandwidth import ledger, line, pooled

N, B = 1000, 4096
NONE_WRONG = dict(windows=[(1, 1)], delta=2.0 ** -40)
"""A record whose one window was opened whole: no receipt can be wrong."""


def groups(stretch: int = 1, *, weights: int = 600, host: int = 0, n: int = 12) -> list[line.Group]:
    """Groups of two passes over N blocks, each pass reading `weights` of them in HBM and `host` in the host tier, a
    group 10 ns apart but group 5, `stretch` times that, each group's last root at the next coin."""
    out, t = [], 0
    for g in range(n):
        span = 10 * (stretch if g == 5 else 1)
        out.append(line.Group(t, t + span, 2, N, 2 * weights, 2 * (N - host), 2 * host))
        t += span
    return out


def stretches(gs: list[line.Group], rate: Fraction = Fraction(0), burst: Fraction = Fraction(0),
              wrong: object = 0, chip_bytes: int = 0) -> tuple[line.Stretch, ...]:
    return line.stretches(gs, per=4, blocks=N, block_bytes=B, chip_bytes=chip_bytes, wrong=wrong,
                          charge=line.Charge(rate, burst))


def test_a_stretch_charges_its_rebuilds_at_two_l_and_its_wrong_receipts_at_their_groups_rates():
    s = stretches(groups(9), Fraction(1, 1000), wrong=50)[1]
    assert (s.lo, s.hi, s.span_ns, s.passes, s.l_max_ns, s.l_sum_ns) == (4, 8, 120, 8, 90, 120)
    assert s.rebuild_blocks == Fraction(3, 10)
    assert s.fill == 50 * 45
    assert s.time.lost_blocks == (2 * 90 * Fraction(3, 10) + 50 * 45) / 120
    assert s.snapshot.lost_blocks == (50 + Fraction(3, 10)) / 8
    assert s.time.weights == (600 - s.time.lost_blocks) * B and s.time.filler == (400 - s.time.lost_blocks) * B


def test_a_burst_is_charged_at_two_l_held_over_time_and_once_over_the_passes():
    plain, burst = stretches(groups(9)), stretches(groups(9), burst=Fraction(7))
    assert burst[1].time.lost_blocks - plain[1].time.lost_blocks == Fraction(2 * 90 * 7, 120)
    assert burst[1].snapshot.lost_blocks - plain[1].snapshot.lost_blocks == Fraction(7, 8)


def test_the_chips_are_charged_in_whole_blocks_rounded_up():
    """`ChipCeiling` counts whole blocks, so S is S_chip/b rounded up: one byte over a block costs a second block."""
    assert stretches(groups(1), chip_bytes=B + 1) == stretches(groups(1), chip_bytes=2 * B)
    assert stretches(groups(1), chip_bytes=B + 1)[0].time.lost_blocks == 2 + stretches(groups(1))[0].time.lost_blocks


def test_the_wrong_receipts_fill_the_groups_highest_rate_first_at_most_n_blocks_a_pass():
    # stretch 1's groups 4..8 pay 5, 45, 5, 5 and 45 (L/n in group 8) block-ns a wrong receipt, 2N receipts each
    assert stretches(groups(9), wrong=4000)[1].fill == 4000 * 45
    assert stretches(groups(9), wrong=4001)[1].fill == 4000 * 45 + 5
    assert stretches(groups(9), wrong=10 ** 9)[1].fill == 2 * N * 2 * 45 + 3 * 2 * N * 5


def test_each_form_charges_the_nodes_rates_and_bursts_in_blocks_a_ns():
    rates = line.Rates(Fraction(3 * 10 ** 9), 3, 4.4e9, 5e10, B, 2 * B, 6 * B)
    c = line.charges(rates, B)
    assert c[line.TIME_HELD] == line.Charge(Fraction(1) + Fraction(44, 10 * B), Fraction(3))
    assert c[line.LINK_CHARGED] == line.Charge(Fraction(1) + Fraction(544, 10 * B), Fraction(9))
    assert line.FORMS == ledger.FORMS[1:]


def test_the_line_is_each_readings_worst_stretch_the_first_on_a_tie_against_the_node():
    gs = groups(1)
    whole = line.line(gs, per=4, blocks=N, block_bytes=B, chip_bytes=0, node_bytes=10 * N * B, **NONE_WRONG,
                      rates=line.Rates(0, 1))
    assert whole.wrong == 0
    for f in line.FORMS:
        assert [x.time.weights for x in whole.form(f).stretches] == [600 * B, 600 * B]
        assert whole.form(f).worst(line.TIME).stretch == 0
        assert whole.share(f) == Fraction(6, 100) and whole.share(f, line.SNAPSHOT, "filler") == Fraction(4, 100)
    burst = line.line(groups(9), per=4, blocks=N, block_bytes=B, chip_bytes=B, node_bytes=10 * N * B,
                      windows=[(100, 10)], delta=2.0 ** -40, rates=line.Rates(10 ** 6, 1))
    assert burst.form(line.TIME_HELD).worst(line.TIME).stretch == 1
    assert burst.wrong == pooled.bound([(100, 10)], Fraction(1, 2 ** 40)).wrong == 90


def test_the_json_form_is_the_lines_and_refuses_what_isnt_its_inputs():
    gs = groups(9, host=300)
    args = dict(per=4, blocks=N, block_bytes=B, chip_bytes=3 * B, node_bytes=10 * N * B, windows=[(4000, 7), (900, 7)],
                delta=1e-9, rates=line.Rates(1.5e11, 7, 4.4e9, 5e10, 2.5, 3, 6291456))
    j = line.inputs_json(gs, **args)
    assert j["windows"] == [[4000, 7], [900, 7]] and j["rates"]["squarings_per_s"] == "150000000000"
    assert line.number(j["delta"]) == Fraction(1e-9) != Fraction("1e-9")
    assert line.run_json(j) == line.line(gs, **args).to_json()
    assert line.run_json(j)["wrong"] == line.text(pooled.bound(args["windows"], 1e-9).wrong)
    assert line.number("-3/6") == Fraction(-1, 2) and line.number(0.1) == Fraction(1, 10)
    for bad in ({**j, "per": -1}, {**j, "delta": 0.5}, {**j, "delta": "1/0"}, {**j, "delta": "1.5"},
                {**j, "windows": [[1]]}, {**j, "windows": [[2, True]]}, {**j, "windows": [[-1, 1]]},
                {**j, "groups": [{**j["groups"][0], "least": True}]}, {k: v for k, v in j.items() if k != "rates"},
                {k: v for k, v in j.items() if k != "windows"}):
        with pytest.raises(line.Malformed):
            line.run_json(bad)


@pytest.mark.parametrize("change,why", [
    (dict(block_bytes=0), "a block is one or more bytes"),
    (dict(blocks=0), "the store is one or more blocks"),
    (dict(node_bytes=0), "the node is one or more bytes"),
    (dict(rates=line.Rates(1, 0)), "a block's encode is one or more squarings"),
    (dict(rates=line.Rates(1, 1, link_burst_bytes=-1)), "link_burst_bytes is nonnegative"),
    (dict(windows=[]), "a session has one or more windows"),
    (dict(delta=1), "δ is above 0 and below 1"),
    (dict(delta=0.0), "δ is above 0 and below 1"),
    (dict(windows=[(5, 2), (3, 0)]), "window 1 draws no receipt"),
    (dict(windows=[(2, 3)], block_bytes=0), "window 0 draws 3 receipts of its 2"),
    (dict(per=0), "a stretch is one or more groups"),
    (dict(per=1), "stretch 0's span is empty: group 1's coin left no later than group 0's end"),
    (dict(per=7), "the session's 12 groups at 7 a stretch leave no stretch before the closing one"),
    (dict(groups=[]), "the session's 0 groups at 4 a stretch leave no stretch before the closing one"),
    (dict(groups=groups()[:1]), "the session's 1 groups at 4 a stretch leave no stretch before the closing one"),
])
def test_the_line_refuses_inputs_it_has_no_line_of(change, why):
    args = dict(groups=groups(), per=4, blocks=N, block_bytes=B, chip_bytes=0, node_bytes=10 * N * B, **NONE_WRONG,
                rates=line.Rates(1, 1)) | change
    with pytest.raises(line.Refused) as e:
        line.line(**args)
    assert str(e.value) == why
    assert line.run_json(line.inputs_json(**args)) == {"refused": why}


@pytest.mark.parametrize("g,change,why", [
    (3, dict(passes=0, hbm=0, weights=0), "group 3 has no pass"),
    (3, dict(last_ns=29), "group 3's last root came before its coin"),
    (3, dict(released_ns=15, last_ns=40), "group 3's coin left before group 2's"),
    (2, dict(least=990), "group 2's least pass read 990 of the 1000 registered blocks: the paced line needs every pass "
                         "to read every registered block, since a block a pass leaves out can be absent across it with "
                         "neither a wrong read nor a rebuild"),
    (2, dict(hbm=1999), "group 2's 2 passes read 1999 registered blocks in all, not 2000"),
    (2, dict(weights=2001, hbm=1999, host=1), "group 2's passes read more weights than HBM, which holds them"),
])
def test_the_line_refuses_a_group_it_cant_read(g, change, why):
    gs = groups()
    x = gs[g]
    gs[g] = line.Group(**{k: change.get(k, getattr(x, k)) for k in line.GROUP})
    with pytest.raises(line.Refused) as e:
        stretches(gs)
    assert str(e.value) == why
