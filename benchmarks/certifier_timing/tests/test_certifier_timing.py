"""certifier_timing without the host's clock: the number on offsets the test makes, the server's plan on a schedule
the test gives, the guard on a clock the test moves, and the network certifier on a `VirtualClock` under each attack.

- `channel.measure` reads a planted bit whole, an ignored one as nothing, and a bit behind noise at its decoder's
  rate; `step` reads through a drift; `samples_to_show` is where a decoder at chance has its upper bound at the value.
- `Plan.writes` keeps `write`'s and `burst`'s bytes in time for the next interval, bit 1's by the release and bit
  0's away from it, and `delay`'s bit 1 half an interval late.
- `run.score` on recorded timestamps: through a certifier that passes bytes on as they come the planted delay is one
  bit an interval and the ignored bit none (the guard's two bounds), on one host and from an observer on another
  whose clock drifts; through one that releases at a fixed time neither attack shows; and a first arrival an
  interval off the server's, an admission the server didn't take, a chunk of another interval's bytes or a failed
  connection is named.
- The network certifier itself, on a clock the test moves: under `write` and `burst` every interval's cell leaves in
  its own interval whatever the bits (the last, with the server's end, as the `END` cell) and ends in the byte that
  names it, and under `delay` the first late interval fails the connection closed.
- `--launch`'s schedule is one interval's cycle from the release time, open and then shut for the certifier's
  priority alone, and a case whose bytes went around it is named; the guard's cases and a control are rows of
  their own and no part of the number.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import sys
from pathlib import Path

import pytest

from verity.core.protocols.certifier.active import DATA, PAD, VirtualClock
from verity.core.protocols.certifier.warden import END, FAIL, Warden, WardenParams, cells, read_cell

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import channel  # noqa: E402
import parties  # noqa: E402
import run as bench  # noqa: E402

TAU = 0.1
SEED = bytes(range(32))


def _coins(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(2) for _ in range(n)]


def test_the_number_is_one_less_the_entropy_of_the_error_rate():
    assert channel.bits(0.0) == 1.0 and channel.bits(0.5) == 0.0 and channel.bits(0.7) == 0.0
    assert channel.bits(0.25) == pytest.approx(1 - (0.25 * 2 + 0.75 * math.log2(4 / 3)), abs=1e-12)
    assert channel.entropy(0.5) == 1.0 and channel.entropy(0.0) == channel.entropy(1.0) == 0.0
    low, high = channel.wilson(0, 400)
    assert low == pytest.approx(0.0, abs=1e-12) and high == pytest.approx(channel.Z95 ** 2 / (400 + channel.Z95 ** 2))
    low, high = channel.wilson(200, 400)
    assert low == pytest.approx(1 - high, abs=1e-12)
    assert high - 0.5 == pytest.approx(channel.Z95 / (2 * math.sqrt(400 + channel.Z95 ** 2)), abs=1e-12)


def test_a_planted_bit_is_read_whole_and_an_ignored_one_not_at_all():
    """Offsets half an interval apart by bit, behind 1 ms of noise, and the same noise alone, 2,000 intervals each.
    Over seeds 0–199 the planted bit's `bits` is 1.0 at every seed with its `information` at least 0.889, and the
    ignored bit's is at most 0.0070 with its `upper` at most 0.0184 and its `information` at most 0.0044."""
    rng = random.Random(7)
    coins = _coins(2000, 7)
    planted = channel.measure([(rng.gauss(0.0005 + 0.05 * b, 0.001), b) for b in coins])
    assert planted["bits"] == 1.0 and planted["lower"] > 0.95 and planted["n"] == 1000
    assert planted["decoder"]["side"] == 1 and planted["shift_us"] == pytest.approx(50000, abs=200)
    ignored = channel.measure([(rng.gauss(0.0005, 0.001), b) for b in coins])
    assert ignored["bits"] < 0.02 and ignored["upper"] < 0.05 and ignored["lower"] == 0.0
    assert ignored["information"]["bits"] < 0.02 and planted["information"]["bits"] > 0.8


def test_a_bit_behind_noise_is_read_at_its_decoders_rate():
    """A shift of 1.349 standard deviations by bit: the best threshold errs a quarter of the time, 0.189 bits an
    interval. Over seeds 0–199 at 4,000 intervals `bits` stays within [0.150, 0.220] and its interval holds 0.189
    at 188 of them; this seed's does."""
    rng = random.Random(11)
    measured = channel.measure([(rng.gauss(1.349e-5 * b, 1e-5), b) for b in _coins(4000, 11)])
    assert 0.10 < measured["bits"] < 0.28
    assert measured["lower"] < channel.bits(0.25) < measured["upper"]
    assert measured["n"] == 2000


def test_step_reads_a_bit_through_a_drift_the_offset_alone_does_not():
    """A 20 µs shift by bit under 2 µs of noise, on offsets that drift by 5 µs an interval, two hosts' clocks 50 ppm
    apart: no one threshold on the offset reads it, and the offset less its neighbours' mean does. Over seeds 0–199
    the decoder is `step` and `bits` at least 0.912 at every seed."""
    rng = random.Random(3)
    coins = _coins(2000, 3)
    measured = channel.measure([(5e-6 * i + 2e-5 * b + rng.gauss(0, 2e-6), b) for i, b in enumerate(coins)])
    assert measured["decoder"]["feature"] == "step" and measured["bits"] > 0.8
    # a line's steps are one constant away from its ends: the window is even, less its two largest
    line = channel.steps([0.25 * i for i in range(40)])
    assert line[8:32] == [pytest.approx(0.25 * 15 / 14)] * 24
    assert channel.steps([5.0]) == [0.0] and channel.steps([1.0, 3.0, 2.0, 10.0]) == [-1.0, 2.0, 1.0, 9.0]
    # one late interval is out of its neighbours' steps
    late = channel.steps([0.0] * 10 + [7.0] + [0.0] * 10)
    assert late[10] == 7.0 and set(late[:10] + late[11:]) == {0.0}


def test_too_few_samples_or_one_bit_measure_nothing():
    assert channel.measure([])["bits"] is None
    assert channel.measure([(0.1, 1), (0.2, 1), (0.3, 1), (0.4, 1)])["bits"] is None
    assert channel.pool([channel.measure([])]) == {"runs": 1, "n": 0, "bits": None, "lower": None, "upper": None}


def test_pooled_runs_add_their_scored_samples_and_errors():
    rng = random.Random(5)
    runs = [channel.measure([(rng.gauss(1.349e-5 * b, 1e-5), b) for b in _coins(600, s)]) for s in (1, 2, 3)]
    pooled = channel.pool(runs)
    assert pooled["n"] == 900 and pooled["wrong"] == sum(r["decoder"]["wrong"] for r in runs)
    assert pooled["bits"] == channel.bits(pooled["wrong"] / 900) and pooled["lower"] <= pooled["bits"] <= pooled["upper"]


def test_the_target_is_the_budgets_seconds_a_bit_and_the_samples_to_show_it():
    goal = channel.target(0.1)
    assert goal["bits"] == [0.1 / 240, 0.1 / 600]
    for value in goal["bits"]:
        need = channel.samples_to_show(value)
        n = need["least"] + need["least"] % 2
        assert channel.bits(channel.wilson(n // 2, n)[0]) <= value
        assert channel.bits(channel.wilson((n - 200) // 2, n - 200)[0]) > value
        more = ((channel.Z95 + channel.Z_POWER) / channel.Z95) ** 2
        assert abs(need["nine_in_ten"] - more * (need["least"] + channel.Z95 ** 2)) <= more + 1
    # the run that shows it is as long whatever the interval: the scored intervals double as the interval halves
    assert channel.samples_to_show(0.05 / 600)["least"] == pytest.approx(2 * channel.samples_to_show(0.1 / 600)["least"],
                                                                         rel=1e-2)


@pytest.mark.parametrize("paced", [0.0, 0.05])
def test_write_and_burst_are_in_time_for_the_next_interval_and_differ_only_by_the_release(paced):
    for attack in ("write", "burst"):
        plan = parties.Plan(attack, TAU, paced)
        for t in range(3, 200):
            bit, u = parties.draw(SEED, t)
            times = plan.writes(t, bit, u)
            assert times == sorted(times) and plan.carries(t) == t + 1
            assert times[-1] < (t + 1) * TAU - TAU / 8, "the next interval's bytes are there before its boundary"
            if bit:
                assert len(times) == (1 if attack == "write" else plan.burst)
                assert plan.release(t) - plan.lead_s <= times[0] and times[-1] <= plan.release(t) + plan.lead_s / 8 + 1e-12
            else:
                assert times == [plan.quiet(t)] and plan.quiet(t) - plan.release(t) == pytest.approx((TAU - paced) / 2)
            assert parties.Plan(attack, TAU, paced, ignore=True).writes(t, bit, u) == [plan.quiet(t)]


def test_delay_writes_an_interval_at_its_boundary_or_half_an_interval_late():
    plan = parties.Plan("delay", TAU)
    for t in range(40):
        assert plan.writes(t, 0, 0.3) == [t * TAU] and plan.writes(t, 1, 0.3) == [t * TAU + TAU / 2]
        assert plan.carries(t) == t
    assert parties.Plan("delay", TAU, ignore=True).writes(7, 1, 0.3) == [7 * TAU]
    with pytest.raises(ValueError, match="an attack is one of"):
        parties.Plan("flood", TAU)


def test_the_bits_are_the_seeds_and_the_bytes_the_same_whatever_the_bit():
    draws = [parties.draw(SEED, t) for t in range(400)]
    assert draws == [parties.draw(SEED, t) for t in range(400)] and {b for b, _ in draws} == {0, 1}
    assert all(0 <= u < 1 for _, u in draws) and 150 < sum(b for b, _ in draws) < 250
    plan = parties.Plan("burst", TAU, 0.05)
    for t, bit, writes in parties.schedule(plan, SEED, 5, 64, 247):
        assert b"".join(data for _, data in writes) == bytes([(t + 1) & 0xFF]) * 247
        assert all(data for _, data in writes) and len(writes) == (plan.burst if bit else 1)


CFG = bench.SCHEDULE | {"interval_ms": 100, "paced_ms": 0.0, "intervals": 400, "lead_ms": 2.0, "burst": 32,
                        "unit": 247, "pin": False, "gate_ms": None, "origin": 1000.0, "observer": "plain"}


def _case(tmp_path: Path, cfg: dict, arrival, *, first: int = 12, shift: int = 0, failed: str | None = None,
          clock=None, admitted: int | None = None, relabel=lambda t: t) -> Path:
    """A recorded case: the server's bits from interval `first`, and the observer's arrivals from its connection's
    first interval (`shift` intervals later than that, for a segment that doesn't fit), `arrival(t, writes)` seconds
    after the origin for interval t whose bit's writes are `writes`. Interval t's chunk ends in the byte
    `relabel(t)` mod 256, and the certifier admitted the connection in interval `admitted` (the one the server
    took). With `clock`, the observer is another host's: its arrival x seconds after the origin is at `clock(x)`
    seconds on its own clock."""
    plan = parties.plan_of(cfg)
    rows, when = [], {}
    for t, bit, writes in parties.schedule(plan, SEED, first, cfg["intervals"], cfg["unit"]):
        rows.append([t, bit, 12.5])
        when[t] = arrival(t, writes)
    ahead = cfg["delay"] if cfg["certifier"] == "warden" else 0
    span = range(first - ahead + shift, first + cfg["intervals"])
    read = clock or (lambda x: cfg["origin"] + x)
    segment = [round(read(when.get(t, plan.release(t) + 1e-4)) * 1e9) for t in span]
    docs = {"config": cfg, "observer": {"observer": cfg["observer"], "chunk": 256, "remote": clock is not None,
                                        "segments": [segment], "cpus": [3],
                                        "labels": [[relabel(t) & 0xFF if t >= first else 0 for t in span]]},
            "server": {"seed": SEED.hex(), "connections": [{"arrived": first - ahead + 1, "first": first}],
                       "rows": rows, "cpus": [2]},
            "certifier": {"certifier": cfg["certifier"], "cpus": [0], "pacer_cpus": [1], "slack_ms": 25.0,
                          "tick_late_us": [80.0] * (first + cfg["intervals"] + 4),
                          "connections": [{"ordinal": 1, "failed": failed,
                                           "admitted": first - ahead if admitted is None else admitted}],
                          "releases": {"rows": [[t, -49000.0, 0.2, 90.0, True]
                                                for t in range(first + cfg["intervals"] + 4)]}}}
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, doc in docs.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(doc))
    return tmp_path


def _path(rng: random.Random) -> float:
    """A path of 300 µs ± 100 µs, never under 50 µs."""
    return max(5e-5, rng.gauss(3e-4, 1e-4))


def test_the_guard_on_a_clock_the_test_moves(tmp_path):
    """A certifier that passes bytes on as they come: an interval arrives when the server wrote it, plus a path of
    300 µs ± 100 µs. The planted delay is one bit an interval and the ignored bit none: the guard's two bounds, with
    room (over seeds 0–199 the planted bit's `bits` is 1.0 at every seed and the ignored bit's at most 0.0210 at
    these 400 intervals)."""
    rng = random.Random(1)
    relay = CFG | {"certifier": "relay", "attack": "delay", "ignore": False}

    def passed_on(t, writes):
        return writes[-1][0] + _path(rng)

    planted = bench.score(_case(tmp_path / "planted", relay, passed_on))
    ignored = bench.score(_case(tmp_path / "ignored", relay | {"ignore": True}, passed_on))
    assert planted["client"]["bits"] >= bench.PLANTED_MIN and planted["client"]["n"] == 200
    assert ignored["client"]["bits"] <= bench.IGNORED_MAX
    assert planted["problems"] == ignored["problems"] == [] and planted["sender"] is None
    assert not planted["remote"] and planted["observer"] == "plain"


def test_the_guard_from_another_hosts_clock(tmp_path):
    """The same two cases read by an observer on another host, whose clock started elsewhere and runs 50 ppm fast:
    nothing joins the clocks, and the planted delay is still one bit an interval and the ignored bit none. A 20 µs
    shift under a 2 µs path, which the drift buries in the offset, is read by `step`. Over seeds 0–199 the planted
    bit's `bits` is 1.0 at every seed, the ignored bit's at most 0.0186, and the 20 µs shift's at least 0.831."""
    rng = random.Random(4)
    relay = CFG | {"certifier": "relay", "attack": "delay", "ignore": False}

    def passed_on(t, writes):
        return writes[-1][0] + _path(rng)

    def far(x):
        return 86400.0 + x * (1 + 50e-6)

    planted = bench.score(_case(tmp_path / "planted", relay, passed_on, clock=far))
    ignored = bench.score(_case(tmp_path / "ignored", relay | {"ignore": True}, passed_on, clock=far))
    assert planted["client"]["bits"] >= bench.PLANTED_MIN and ignored["client"]["bits"] <= bench.IGNORED_MAX
    assert planted["problems"] == ignored["problems"] == [] and planted["remote"]
    cfg = CFG | {"certifier": "warden", "attack": "write", "ignore": False, "paced_ms": 50.0}
    plan = parties.plan_of(cfg)
    coins = {t: bit for t, bit, _ in parties.schedule(plan, SEED, 12, cfg["intervals"], cfg["unit"])}
    moved = bench.score(_case(tmp_path / "moved", cfg, clock=far,
                              arrival=lambda t, writes: plan.release(t) + 3e-4 - 2e-5 * coins[t] + rng.gauss(0, 2e-6)))
    assert moved["client"]["decoder"]["feature"] == "step" and moved["client"]["bits"] > 0.7
    assert moved["problems"] == []


def test_a_release_at_a_fixed_time_shows_neither_attack(tmp_path):
    """A certifier whose release time nothing moves: an interval arrives its path after its release time, whenever
    the server wrote. Over seeds 0–199 `bits` is at most 0.0235, on one host and from another's clock."""
    rng = random.Random(2)
    for attack in ("write", "burst"):
        cfg = CFG | {"certifier": "warden", "attack": attack, "ignore": False, "paced_ms": 50.0}
        plan = parties.plan_of(cfg)
        for clock in (None, lambda x: 7.0 + x * (1 - 30e-6)):
            scored = bench.score(_case(tmp_path / f"{attack}-{clock is None}", cfg,
                                       lambda t, writes: plan.release(t) + _path(rng), clock=clock))
            assert scored["client"]["bits"] <= bench.IGNORED_MAX and scored["problems"] == []
            assert scored["sender"]["bits"] == 0.0 and scored["late_writes"] == 0


def test_what_makes_a_case_no_measurement_is_named(tmp_path):
    cfg = CFG | {"certifier": "warden", "attack": "write", "ignore": False, "paced_ms": 50.0}
    plan = parties.plan_of(cfg)

    def on_time(t, writes):
        return plan.release(t) + 3e-4

    assert bench.score(_case(tmp_path / "fits", cfg, on_time))["problems"] == []
    off = bench.score(_case(tmp_path / "off", cfg, on_time, shift=1))
    assert off["problems"] == ["connection 0: the server took interval 12 for its first bytes, and the observer's "
                               "first arrival puts it at 13 (admit_lag)"]
    far = bench.score(_case(tmp_path / "far", cfg, on_time, shift=1, clock=lambda x: 5.0 + x))
    assert far["problems"] == ["connection 0: 399 of the observer's scored intervals hold another interval's bytes"]
    failed = bench.score(_case(tmp_path / "failed", cfg, on_time, failed="late"))
    assert failed["problems"] == ["the certifier failed connection 1 (late)"]
    later = bench.score(_case(tmp_path / "later", cfg, on_time, admitted=10, clock=lambda x: 5.0 + x))
    assert later["problems"] == ["connection 0: the certifier admitted it in interval 10, so its first bytes leave "
                                 "in 13, and the server took 12 (admit_lag)"]
    lost = bench.score(_case(tmp_path / "lost", cfg, on_time, relabel=lambda t: t + (t >= 300)))
    assert lost["problems"] == ["connection 0: 112 of the observer's scored intervals hold another interval's bytes"]


def test_the_number_is_the_most_any_attack_showed_and_the_ceiling_every_attacks_bound(tmp_path):
    """Two runs each of an attack that shifts an interval by 1.349 standard deviations of its path and of one that
    shifts nothing, with the first's control and the guard's two cases beside them. Over seeds 0–199 the first's
    pooled `bits` is at least 0.101 and the second's at most 0.0114."""
    rng = random.Random(6)
    cases = []
    for attack, shift in (("write", 1.349e-4), ("burst", 0.0)):
        cfg = CFG | {"certifier": "warden", "attack": attack, "ignore": False, "paced_ms": 50.0}
        plan = parties.plan_of(cfg)
        coins = {t: bit for t, bit, _ in parties.schedule(plan, SEED, 12, cfg["intervals"], cfg["unit"])}
        for k in range(2):
            cases.append(bench.score(_case(
                tmp_path / f"{attack}-{k}", cfg,
                lambda t, writes: plan.release(t) + 3e-4 + shift * coins[t] + rng.gauss(0, 1e-4))))
    doc = bench.summarize(cases)
    write, burst = doc["rows"]
    assert (write["attack"], burst["attack"]) == ("write", "burst") and write["runs"] == burst["runs"] == 2
    assert doc["value"] == write["bits"] > burst["bits"] and doc["n"] == 400 and doc["unit"] == "bits/interval"
    assert (doc["lower"], doc["upper"]) == (write["lower"], write["upper"])
    assert doc["ceiling"] == max(write["upper"], burst["upper"]) and doc["problems"] == []
    assert doc["best"] == {"attack": "write", "observer": "plain"}
    assert bench.summarize([])["value"] is None and bench.summarize([])["ceiling"] is None
    assert bench.verdict(cases) is None
    # the guard's cases and a control are rows of their own, after the attacks, and no part of the number
    relay = CFG | {"certifier": "relay", "attack": "delay", "ignore": False}
    control = CFG | {"certifier": "warden", "attack": "write", "ignore": True, "paced_ms": 50.0}
    more = [bench.score(_case(tmp_path / "planted", relay, lambda t, writes: writes[-1][0] + _path(rng))),
            bench.score(_case(tmp_path / "ignored", relay | {"ignore": True},
                              lambda t, writes: writes[-1][0] + _path(rng))),
            bench.score(_case(tmp_path / "control", control, lambda t, writes: plan.release(t) + _path(rng)))]
    assert [bench.role(c) for c in more] == ["planted", "ignored", "control"]
    whole = bench.result(more + cases)
    assert [r["role"] for r in whole["rows"]] == ["attack", "attack", "planted", "ignored", "control"]
    assert {k: whole[k] for k in ("value", "ceiling", "best", "n")} == {k: doc[k] for k in ("value", "ceiling", "best", "n")}
    assert whole["guard"] == {"ok": True, "planted": 1.0, "ignored": more[1]["client"]["bits"],
                              "bounds": {"planted_min": bench.PLANTED_MIN, "ignored_max": bench.IGNORED_MAX}}
    assert whole["target"]["seconds_a_bit"] == [240.0, 600.0] and whole["target"]["met"] == [False, False]
    # a guard without its control, or whose planted channel doesn't show, is not ok
    assert not bench.verdict(more[:1])["ok"] and not bench.verdict(more[1:])["ok"]
    assert not bench.verdict([more[1], more[1] | {"ignore": False}])["ok"]
    table = bench._table([whole]).splitlines()
    assert len(table) == 2 + 5 and table[2].split(" | ")[10:13] == ["attack", "write", "plain"]


def test_an_observer_on_another_host_leaves_its_file_by_the_cases_name(tmp_path):
    case, seen = tmp_path / "out" / "write-spin", tmp_path / "seen"
    assert bench._observed(case, None) is None
    (seen / "write-spin").mkdir(parents=True)
    assert bench._observed(case, seen) == seen / "write-spin" / "observer.json"
    (seen / "one.json").write_text("{}")
    assert bench._observed(case, seen / "one.json") == seen / "one.json"


def test_the_launch_gate_is_one_intervals_cycle_from_the_release_time(tmp_path):
    """The schedule passes the certifier's priority for the gate's opening alone and every other always, each on a
    queue of its own; `txtime` adds the queue that holds the stamped packets; and a case whose gated queue carried
    nothing, or dropped, is named."""
    launch = bench._launch_arg("eth1,20,5")
    assert launch == {"dev": "eth1", "after_ms": 20.0, "open_ms": 5.0, "mode": "gate"}
    setup, remove = bench.launch_commands(launch, 1000.07, 100)
    assert setup == [["tc", "qdisc", "replace", "dev", "eth1", "parent", "root", "handle", "100", "taprio",
                      "num_tc", "2", "map", "0", "0", "0", "0", "0", "1", *["0"] * 10, "queues", "1@0", "1@1",
                      "base-time", "1000070000000", "sched-entry", "S", "03", "5000000",
                      "sched-entry", "S", "01", "95000000", "clockid", "CLOCK_MONOTONIC"]]
    assert setup[0][setup[0].index("map") + 1 + parties.LAUNCH_PRIORITY] == "1"
    assert remove == ["tc", "qdisc", "del", "dev", "eth1", "root"]
    (nic,), _ = bench.launch_commands(bench._launch_arg("eth1,20,5,gate-nic"), 1000.07, 100)
    assert nic[-2:] == ["flags", "0x2"] and "clockid" not in nic
    for mode, offload in (("txtime", []), ("txtime-nic", ["offload"])):
        (stamps, held), _ = bench.launch_commands(bench._launch_arg(f"eth1,20,5,{mode}"), 1000.07, 100)
        assert stamps[-6:] == ["flags", "0x1", "txtime-delay", "500000", "clockid", "CLOCK_TAI"]
        assert held == ["tc", "qdisc", "replace", "dev", "eth1", "parent", "100:2", "etf", "clockid", "CLOCK_TAI",
                        "delta", "200000", *offload, "skip_sock_check"]
    with pytest.raises(ValueError, match="open for part of an interval"):
        bench.launch_commands(launch | {"open_ms": 100.0}, 0.0, 100)
    # its base time is one of the gate's openings, already past whenever the case sets it
    for now in (994.0, 1000.0, 1000.07, 1003.33):
        base = bench.base_time(1000.07, now, 100)
        assert 0.1 <= now - base < 0.2 + 1e-9 and round((1000.07 - base) * 10, 6) == round((1000.07 - base) * 10)
    with pytest.raises(bench.argparse.ArgumentTypeError):
        bench._launch_arg("eth1,20,5,offload")
    # the server aims at when the gate lets the bytes go, the pacer keeps its own offset
    args = bench.argparse.Namespace(interval_ms=100, paced_ms=50.0, launch=launch, intervals=10, lead_ms=2.0, burst=32,
                                    start_s=6.0, pin=False, gate_ms=None, waker_ms=None, awake=False)
    cfg = bench.config(args, attack="write", observer="spin")
    assert (cfg["paced_ms"], cfg["aim_ms"]) == (50.0, 70.0) and parties.plan_of(cfg).release(3) == pytest.approx(0.37)
    args.launch = launch | {"mode": "txtime"}
    assert bench.config(args, attack="write", observer="spin")["aim_ms"] == pytest.approx(69.8)
    relay = bench.config(args, attack="delay", observer="plain", certifier="relay")
    assert (relay["paced_ms"], relay["aim_ms"], relay["launch"]) == (0.0, 0.0, None)
    # the device's queues, as `tc -s -j qdisc show` gives them once the case has run
    gated = CFG | {"certifier": "warden", "attack": "write", "ignore": False, "paced_ms": 50.0, "aim_ms": 70.0,
                   "launch": launch}
    plan = parties.plan_of(gated)
    for name, queues, problems in (
            ("through", [{"kind": "taprio", "handle": "100:", "packets": 900}, {"parent": "100:1", "packets": 9},
                         {"parent": "100:2", "packets": 891, "drops": 0}], []),
            ("around", [{"parent": "100:1", "packets": 900}, {"parent": "100:2", "packets": 0}],
             ["none of the case's packets left by the launch gate's queue"]),
            ("dropped", [{"parent": "100:2", "packets": 880, "drops": 11}],
             ["the launch gate's queue dropped 11 packets"])):
        case = _case(tmp_path / name, gated, lambda t, writes: plan.release(t) + 3e-4)
        (case / "launch.json").write_text(json.dumps(queues))
        scored = bench.score(case)
        assert scored["problems"] == problems and scored["gated"]["packets"] == queues[-1]["packets"]
    unrecorded = bench.score(_case(tmp_path / "unrecorded", gated, lambda t, writes: plan.release(t) + 3e-4))
    assert unrecorded["problems"] == ["none of the case's packets left by the launch gate's queue"]


def test_an_observers_chunks_start_and_end_where_the_bytes_say():
    assert [parties.starts(total, size, 256) for total, size in ((0, 256), (0, 1), (1, 255), (256, 300), (255, 2))] \
        == [1, 1, 0, 2, 1]
    assert sum(parties.starts(7 * k, 7, 256) for k in range(1000)) == -(-7000 // 256)
    assert parties.last_bytes(bytes([1]) * 4 + bytes([2]) * 4 + bytes([3]) * 2, 4) == [1, 2, 3]
    assert parties.last_bytes(b"", 4) == [] and parties.last_bytes(bytes([7]) * 8, 4) == [7, 7]


class _Wire:
    """One side's connection in memory: a reader the test feeds, and what the certifier wrote to it with the tick
    it wrote it in."""

    def __init__(self, ticks, data: bytes = b"") -> None:
        self.ticks, self.writes, self.closed, self.transport = ticks, [], False, self
        self.reader = asyncio.StreamReader()
        self.reader.feed_data(data)
        self.fed = 0

    def write(self, data: bytes) -> None:
        self.writes.append((self.ticks(), bytes(data)))

    def feed(self, data: bytes) -> None:
        self.fed += len(data)
        self.reader.feed_data(data)

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    def get_write_buffer_size(self) -> int:
        return 0


async def _certify(attack: str, intervals: int) -> tuple[Warden, list[tuple[int, int, int]], int, dict[int, int]]:
    """The network certifier on a clock the test moves, at the benchmark's schedule, one client, and the server's
    plan: each write reaches the certifier before the first tick at or after its time. Returns the certifier, the
    client's cells as (tick, kind, last byte), the first interval the server wrote for and its bits."""
    cfg = CFG | {"certifier": "warden", "attack": attack, "ignore": False, "intervals": intervals}
    plan, tau, total = parties.plan_of(cfg), cfg["interval_ms"] / 1000, cfg["delay"] + intervals + 16
    params = WardenParams(cell_bytes=cfg["cell_bytes"], slots=cfg["slots"], bucket_ms=cfg["interval_ms"],
                          delay=cfg["delay"], lanes=2, queue=4, window=cfg["window"], ingress_cells=4,
                          ingress_buckets=8, egress_cells=cfg["slots"] * total, egress="bytes")
    assert params.payload * params.slots == cfg["unit"] == bench.SCHEDULE["cell_bytes"] - parties.HEADER_BYTES
    clock = VirtualClock(-tau / 2)
    upstream: list[_Wire] = []

    async def connect():
        upstream.append(_Wire(lambda: warden.ticks))
        return upstream[0].reader, upstream[0]

    warden = Warden(params, connect, origin=0.0, windows=1, clock=clock, sleep=clock.sleep)
    running = asyncio.create_task(warden.run())
    client = _Wire(lambda: warden.ticks, cells(parties.HELLO, params.cell_bytes))
    client.reader.feed_eof()
    handling = asyncio.create_task(warden.handle(client.reader, client))
    await warden.until(lambda w: w.ordinals == 1)
    pending, first, coins, ended = [], None, {}, False
    for t in range(total):
        if upstream and first is None and upstream[0].writes:
            first = parties.first_interval(cfg, upstream[0].writes[0][0])
            if attack != "delay":
                upstream[0].feed(bytes([first & 0xFF]) * cfg["unit"])
            for at, bit, writes in parties.schedule(plan, SEED, first, intervals, cfg["unit"]):
                coins[at] = bit
                pending += writes
        while pending and pending[0][0] <= t * tau and not upstream[0].closed:
            upstream[0].feed(pending.pop(0)[1])
        if first is not None and not pending and not ended:
            ended = True
            upstream[0].reader.feed_eof()
        await warden.until(lambda w: all(
            c.ingress_ended and c.server is not None and (upstream[0].closed or (
                c.bytes_out == upstream[0].fed and c.egress_ended == ended)) for c in w.active))
        clock.advance(t * tau)
        await warden.until(lambda w, t=t: w.ticks > t)
        if not warden.active and first is not None:
            break
    running.cancel()
    await asyncio.gather(running, handling, return_exceptions=True)
    seen = [(tick, read_cell(data[i:i + params.cell_bytes], params.cell_bytes)[0], data[i + params.cell_bytes - 1])
            for tick, data in client.writes for i in range(0, len(data), params.cell_bytes)]
    return warden, seen, first, coins


@pytest.mark.parametrize("attack", ["write", "burst"])
def test_the_network_certifier_releases_each_interval_in_its_own_whatever_the_bits(attack):
    warden, seen, first, coins = asyncio.run(_certify(attack, 96))
    (log,) = warden.log
    assert log["failed"] is None and log["admitted"] + bench.SCHEDULE["delay"] == first
    assert [kind for _, kind, _ in seen] == [PAD] * bench.SCHEDULE["delay"] + [DATA] * 96 + [END]
    assert [tick for tick, _, _ in seen] == list(range(log["admitted"], log["admitted"] + len(seen)))
    assert all(last == tick & 0xFF for tick, kind, last in seen if kind != PAD), "a cell ends in its interval's byte"
    assert len(coins) == 96 and set(coins.values()) == {0, 1}
    left = [tick for tick, kind, _ in seen if kind != PAD]
    offsets = [(left[i] - (first + i), coins[first + i]) for i in range(len(left)) if first + i in coins]
    assert len(offsets) == 96 and channel.measure(offsets)["bits"] == 0.0


def test_the_network_certifier_fails_a_delayed_interval_closed():
    warden, seen, first, coins = asyncio.run(_certify("delay", 96))
    late = min(t for t, bit in coins.items() if bit)
    (log,) = warden.log
    assert log["failed"] == "late" and log["ended"] == late and log["failed_at"] == late - first
    assert seen[-1][:2] == (late, FAIL) and [k for _, k, _ in seen].count(FAIL) == 1
