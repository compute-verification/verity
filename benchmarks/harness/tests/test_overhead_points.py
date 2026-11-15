"""`verity bench overhead` on CPU with stand-in timings: the operating points' rule and sweep, γ and its interval, and busy
SM-time from an Nsight Systems export."""

from __future__ import annotations

import sqlite3

import pytest
from overhead_standins import H200, LINEARS, SCHEMA, brute_force, roof_ms
from overhead_standins import stamped as stamped
from verity_bench import kit, nsys
from verity_bench import overhead as O

# ---- the points ---------------------------------------------------------------------------------------------------------------


def test_the_sweep_finds_the_rules_points_over_every_m_it_could_sample(tmp_path):
    """C and L from the powers of two and bisection on multiples of 16 are the rule's over every multiple of 16 from 16 to
    32,768: the sweep samples a few dozen m's where the grid has 2,048, and finds the same points."""

    times: dict[int, float] = {}
    while todo := O.next_ms(times):
        for m in todo:
            times[m] = sum(min(roof_ms(f, c, m, n, k) for f in O.FP8 for c in ("cublaslt", "cutlass-vllm"))
                           for n, k in LINEARS.values())
    got, want = O.final(times), brute_force()
    assert (got.c, got.lat) == (want.c, want.lat) and got.c % O.GRID == 0 and got.lat % O.GRID == 0
    assert 16 < got.lat < got.c < O.M_MAX and len(times) < 60
    flops = {m: O.layer_flop(O.KERNEL_MODEL, m) / t for m, t in times.items()}
    assert all(flops[m] >= O.C_SHARE * got.best_flop_per_ms for m in times if m >= got.c)
    assert all(times[m] <= (1 + O.L_SLACK) * times[1] for m in times if m <= got.lat)


def test_c_needs_every_m_above_it_and_l_every_m_below_it():
    """A dip at m = 4,096 below C's share keeps C above it, though 2,048 passes; a spike at m = 32 past L's slack keeps L
    below it, though 64 is within it. The rule refuses a sweep without m = 1, and `final` an unfinished one."""

    flop, best = O.layer_flop(O.KERNEL_MODEL, 1), 1000 * O.layer_flop(O.KERNEL_MODEL, 1)
    share = {2048: 0.95, 4096: 0.5, 8192: 0.97, 32768: 1.0}
    times = {1: 1.0, 16: 1.02, 32: 1.5, 64: 1.05} | {m: m * flop / (s * best) for m, s in share.items()}
    p = O.rule(times)
    assert p.best_flop_per_ms == pytest.approx(best)
    assert p.c == 8192 and p.c_bracket == (4096, 8192)
    assert p.lat == 16 and p.l_bracket == (16, 32)
    del times[4096]
    assert O.rule(times).c == 2048
    with pytest.raises(ValueError, match="sample m = 1"):
        O.rule({16: 1.0})
    with pytest.raises(ValueError, match="isn't finished"):
        O.final(times)


def test_sweep_times_takes_each_linears_fastest_kernel_of_every_scaling_and_refuses_a_gap():
    rows = [{"label": O.label(16, n, k), "m": 16, "n": n, "k": k,
             "summary": {f"baseline/{O.label(16, n, k)}/fp8-e4m3/a": {"graph": {"median": 2.0}},
                         f"baseline/{O.label(16, n, k)}/fp8-blk128/b": {"graph": {"median": 1.0}},
                         f"twin/{O.label(16, n, k)}/fp8-blk128/b": {"graph": {"median": 0.1}}}}
            for n, k in LINEARS.values()]
    times, chosen = O.sweep_times([{"shapes": rows}])
    assert times == {16: 4.0} and {v["kernel"] for v in chosen[16].values()} == {"fp8-blk128/b"}
    rows[2] = {**rows[2], "summary": {}, "skipped": "operands need 9 bytes; 1 free"}
    with pytest.raises(ValueError, match="gate_up at m = 16 .*operands need"):
        O.sweep_times([{"shapes": rows}])


# ---- γ -------------------------------------------------------------------------------------------------------------------------


def test_the_interval_is_the_medians_95_percent_order_statistics():
    assert kit.paired([1.5] * 5, [1.0] * 5)["ci95"] is None
    g = kit.paired([1.5 + 0.001 * i for i in range(20)], [1.0] * 20)
    assert g["gamma"] == pytest.approx(0.5095) and g["ci95"] == pytest.approx([0.505, 0.514]) and g["n"] == 20
    assert O.of_record_why(g) is None
    assert "fewer than 20" in O.of_record_why(kit.paired([1.5] * 19, [1.0] * 19))
    assert "wider than ±5%" in O.of_record_why(kit.paired([1.0 + 0.1 * i for i in range(20)], [1.0] * 20))
    with pytest.raises(ValueError, match="one for one"):
        kit.paired([1.0], [1.0, 2.0])


def test_a_calls_cost_is_its_traced_busy_sm_time_and_the_hosts_enqueue_at_the_cards_weight():
    assert O.traced_cost({"burst": 4}, {"sm_ns": 8e9, "occupied_sm_ns": 1e10}) == pytest.approx(
        {"sm_s": 2.0, "occupied_sm_s": 2.5, "sm_ns": 8e9, "occupied_sm_ns": 1e10})
    assert O.traced_cost({"burst": 4}, {"sm_ns": None}) is None and O.traced_cost({"burst": 4}, None) is None
    assert O.host_cost({"burst": 4, "cpu_s": 2.0}, H200) == pytest.approx(0.48 * 2 / 4)


def test_periods_scale_gamma_to_the_datacenter_and_the_world():
    p = O.periods(0.25, 100_000, 600, 9.36e30, 1.9789e15, 250)
    assert p["share"] == pytest.approx(0.2)
    assert p["year"]["datacenter_gpu_h"] == pytest.approx(100_000 * 8760 * 0.2)
    assert p["year"]["world_flop"] == pytest.approx(0.25 * 9.36e30)
    assert p["week"]["world_gates"] == pytest.approx(0.25 * 9.36e30 * 168 / 8760 * 250)
    assert p["window"]["datacenter_gates"] == pytest.approx(100_000 * 600 * 0.2 * 1.9789e15 * 250)


# ---- busy SM-time ----------------------------------------------------------------------------------------------------------------


def test_busy_sm_time_integrates_sms_active_over_each_ranges_kernels_and_checks_it_against_their_occupancy(tmp_path):
    """Two ranges on one thread: a graph launch of two kernels (one of 32 blocks, on 32 SMs), and a JIT library's
    driver launch; a third range on another thread launched nothing. A range's text may sit in StringIds. SMs Active is
    integrated over the kernels' span between samples, and a range reading busier than its kernels occupy is flagged."""

    db_path = tmp_path / "prof.sqlite"
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    db.execute("INSERT INTO StringIds VALUES (5, 'item 1')")
    db.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)",
                   [(0, 10_000, 1, "item 0", None), (20_000, 30_000, 1, None, 5), (0, 30_000, 2, "item 2", None),
                    (0, 5, 1, "other", None)])
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (100, 200, 1, 11)")
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (20100, 20200, 1, 12)")
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, 1, 1, 128, 1, 1, 32, 0, 0)",
                   [(1000, 5000, 11, H200.sms * 16), (5000, 6000, 11, 32), (21000, 29000, 12, H200.sms * 16),
                    (15000, 16000, 13, 1)])
    # SMs Active: full over the first kernel, 32 of 132 SMs over the second, then 100% sampled every 2 µs over the third
    # but reading 110% of the SMs (a metric that over-reads)
    samples = [(1000, 0.0), (5000, 100.0), (6000, 100 * 32 / H200.sms), (21000, 0.0)]
    samples += [(t, 110.0) for t in range(23000, 29001, 2000)]
    db.executemany("INSERT INTO GPU_METRICS VALUES (?, 3, ?)", samples)
    db.commit()
    db.close()

    rep = nsys.report(db_path, H200, "item ")
    got = rep["busy"]
    assert rep["ranges"] == 3 and rep["stray"] == 1
    assert set(got) == {"item 0", "item 1", "item 2"}
    assert got["item 0"]["kernels"] == 2 and got["item 0"]["span_ns"] == 5000
    assert got["item 0"]["sm_ns"] == pytest.approx(H200.sms * 4000 + 32 * 1000)
    assert got["item 0"]["occupied_sm_ns"] == pytest.approx(H200.sms * 4000 + 32 * 1000)
    assert got["item 0"]["ratio"] == pytest.approx(1)
    assert got["item 1"]["sm_ns"] == pytest.approx(H200.sms * 8000 * 1.1)
    assert got["item 2"] == {"sm_ns": None, "occupied_sm_ns": None, "ratio": None, "kernels": 0}
    sqlite3.connect(tmp_path / "prime.sqlite").close()
    assert nsys.report(tmp_path / "prime.sqlite", H200, "item ") == {"busy": {}, "ranges": 0, "stray": 0}
    assert O.cross_check_why({"a": got["item 0"]["ratio"]}) is None
    assert "1 entries" in O.cross_check_why({"a": got["item 0"]["ratio"], "b": got["item 1"]["ratio"]})
    assert O.cross_check_why({"tail": 0.7, "a": 1.02}) is None and O.cross_check_why({"none": None})
    assert "exceeds their kernels' occupied SM-time by over 5% (over: 1.07)" in O.cross_check_why({"over": 1.07, "a": 1.02})


def test_a_kernel_occupies_at_most_one_sm_a_block():
    """The scheduler spreads a grid of fewer blocks than the card's SMs over one SM a block, however many an SM could hold:
    24 small blocks read 24 SMs Active, not the one SM they would fit in."""

    assert nsys.occupied_ns(nsys.Kernel(0, 10, 1000), H200) == 10 * H200.sms
    assert nsys.occupied_ns(nsys.Kernel(0, 10, 24), H200) == 10 * 24
