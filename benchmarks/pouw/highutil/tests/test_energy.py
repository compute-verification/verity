from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

import pytest

import energy

H200 = {"sku": "h200-sxm-141gb", "clock_mhz": 1980, "tensor": 4096, "sms": 132}
FL = energy.FLOORS["h200-sxm-141gb"]
THETA = {"tensor": 4096 * 132 * 1830e6, "boost": 4096 * 132 * 1980e6}
T0 = datetime(2026, 10, 27, 12, 0, 0, tzinfo=UTC)


def workload(served: bool = True) -> dict:
    return {"m": 1024, "layers": [["a", 256, 512]], "lp": {"a": 128}, "L": 512, "served": served,
            "step_macs": 1024 * 256 * 512}


def summary(served: bool = True) -> dict:
    return {"card": H200, "workload": workload(served)}


def period(name: str, session_s: int, wall_s: float, steps: int, cheat: str = "none") -> dict:
    t = T0 + timedelta(seconds=session_s)
    return {"period": name, "arm": "proto", "cheat": cheat, "wall_s": wall_s, "steps": steps, "session_open_s": 0.0,
            "accepted": cheat == "none", "registration": {"row": 1024},
            "coins": {"session": f"s{t:%Y%m%dT%H%M%S}Z-abcd"}}


def trace(watts) -> str:
    """nvidia-smi's CSV, a sample every half second; `watts(t)` gives each sample's power at t seconds from T0."""
    head = "timestamp, clocks.current.sm [MHz], clocks.current.graphics [MHz], power.draw [W]"
    rows = [f"{T0 + timedelta(seconds=i / 2):%Y/%m/%d %H:%M:%S.%f}, 1830 MHz, 1830 MHz, {watts(i / 2):.6f} W"
            for i in range(-10, 120)]
    return "\n".join([head, *rows])


def floors_w(steps: int, wall: float, served: bool = True) -> float:
    m = energy.terms.macs_per_step(workload(served))
    comps = energy.compressions_per_step(workload(served), 1024)
    return (FL["idle_w"][0] + steps * (m["declared"] * FL["mac_j"]["lt"][0] + m["peels"] * FL["mac_j"]["rand"][0]
                                       + comps * FL["compression_j"][0]) / wall)


def test_the_compression_count_is_the_shape_table_s_value_roots_and_coins():
    """art:9f8d21ae's table: #5465 served (L 4096 / L' 512, row 16384) hashes 7.70 + 0.54 M a T MAC in its blocks, 5d28's
    filler form (row 8192) 4.70 + 1.33; the operand roots (0.61, 1.17) are built in prep."""
    eight_b = {"m": 16384, "layers": [["qkv", 6144, 4096], ["o", 4096, 4096], ["up", 28672, 4096], ["down", 4096, 14336]],
               "lp": dict.fromkeys(("qkv", "o", "up", "down"), 512), "L": 4096, "step_macs": 3573412790272}
    served = energy.compressions_per_step({**eight_b, "served": True}, 16384) / eight_b["step_macs"] * 1e6
    filler = energy.compressions_per_step({**eight_b, "served": False}, 8192) / eight_b["step_macs"] * 1e6
    assert round(served, 2) == 8.24 and round(filler, 2) == 6.03


def test_power_at_the_floors_plus_x_watts_is_x_over_e_hidden_theta():
    steps, wall, x = 200_000, 20.0, 41.5
    p = floors_w(steps, wall) + x
    samples = energy.terms.smi_samples(trace(lambda t: p if 1 <= t <= 19 else 5.0))
    out = energy.energy(summary(), [period("p0", 0, wall, steps)], samples)
    e_hidden = FL["mac_j"]["u2"][0]
    for name, theta in THETA.items():
        r = out["run"]["readings"][name]
        assert math.isclose(r["U"]["share"], x / (e_hidden * theta), rel_tol=1e-6)
        assert math.isclose(r["watts"]["unexplained"], x, rel_tol=1e-6)
        assert math.isclose(r["points_a_watt_of_meter_error"], 100 / (e_hidden * theta))
        cert = steps * out["macs_per_step"]["certified"] / (theta * wall)
        assert math.isclose(r["certified_share"], cert) and math.isclose(r["tensor_time_cap"]["share"], 1 - cert)
        assert math.isclose(sum(r["U_points"].values()) + r["U"]["share"] * 100, p / (e_hidden * theta) * 100, rel_tol=1e-6)
    assert out["energy_source"] == energy.SENSOR and "not a premise" in out["energy_source"]
    assert out["periods"][0]["samples"] == 37 and math.isclose(out["periods"][0]["mean_w"], p, rel_tol=1e-6)


def test_u_is_never_below_zero_nor_past_the_tensor_time_the_certified_macs_leave():
    steps, wall = 200_000, 20.0
    low = energy.energy(summary(), [period("p0", 0, wall, steps)],
                        energy.terms.smi_samples(trace(lambda t: floors_w(steps, wall) - 30)))
    r = low["run"]["readings"]["tensor"]
    assert r["U"]["share"] == 0 and r["watts"]["unexplained"] < 0
    high = energy.energy(summary(), [period("p0", 0, wall, steps)], energy.terms.smi_samples(trace(lambda t: 5000.0)))
    r = high["run"]["readings"]["tensor"]
    assert math.isclose(r["U"]["share"], r["tensor_time_cap"]["share"]) and r["U_energy_uncapped"]["share"] > r["U"]["share"]


def test_the_trace_s_mean_is_the_trapezoid_over_the_trimmed_window():
    steps, wall = 200_000, 20.0
    out = energy.energy(summary(), [period("p0", 0, wall, steps)], energy.terms.smi_samples(trace(lambda t: 100 + 10 * t)))
    assert math.isclose(out["periods"][0]["mean_w"], 200.0) and out["periods"][0]["samples"] == 37
    assert math.isclose(out["periods"][0]["energy_j"], 200.0 * wall)


def test_the_run_sums_the_honest_periods_and_leaves_the_cheat_out():
    steps, wall, x = 100_000, 20.0, 20.0
    base = floors_w(steps, wall)
    periods = [period("p0", 0, wall, steps), period("neg", 22, 10.0, steps, cheat="skip:2@3"), period("p1", 34, wall, steps)]
    samples = energy.terms.smi_samples(trace(lambda t: base + (x if t < 22 else 3 * x)))
    out = energy.energy(summary(), periods, samples)
    assert [p["period"] for p in out["periods"]] == ["p0", "p1"]
    run = out["run"]
    assert run["steps"] == 2 * steps and math.isclose(run["wall_s"], 2 * wall)
    assert math.isclose(run["readings"]["tensor"]["watts"]["unexplained"], 2 * x, rel_tol=1e-6)


def test_each_share_states_its_gap_to_three_percent():
    assert energy.vs3(0.026)["vs_3pct"] == "0.40 under" and energy.vs3(0.039)["vs_3pct"] == "0.90 over"
    assert math.isclose(energy.vs3(0.806)["gap_points"], 77.6)


def test_the_declared_floor_s_class_moves_u_by_its_energy():
    steps, wall = 200_000, 20.0
    p = floors_w(steps, wall) + 10
    out = energy.energy(summary(), [period("p0", 0, wall, steps)], energy.terms.smi_samples(trace(lambda t: p)))
    r = out["run"]["readings"]["tensor"]
    by = r["U_by_declared_class"]
    assert math.isclose(by["lt"]["share"], r["U"]["share"])
    decl = steps * out["macs_per_step"]["declared"]
    gap = decl * (FL["mac_j"]["lt"][0] - FL["mac_j"]["rand"][0]) / (FL["mac_j"]["u2"][0] * THETA["tensor"] * wall)
    assert math.isclose(by["rand"]["share"] - by["lt"]["share"], gap)
    assert math.isclose(r["points_a_percent_of_declared_floor"],
                        decl * FL["mac_j"]["lt"][0] / (FL["mac_j"]["u2"][0] * THETA["tensor"] * wall))


def test_an_unknown_card_or_class_is_refused():
    with pytest.raises(SystemExit, match="no energy floors"):
        energy.energy({"card": {**H200, "sku": "rtx-pro-6000"}, "workload": workload()}, [], [])
    with pytest.raises(SystemExit, match="no MAC class"):
        energy.energy(summary(), [period("p0", 0, 20.0, 1)], [], declared="cutlass")


def test_the_cli_prints_one_json_object(tmp_path, capsys):
    steps, wall = 200_000, 20.0
    (tmp_path / "s.json").write_text(json.dumps(summary()))
    (tmp_path / "p.jsonl").write_text(json.dumps(period("p0", 0, wall, steps)) + "\n")
    (tmp_path / "t.csv").write_text(trace(lambda t: 600.0))
    assert energy.main([str(tmp_path / x) for x in ("s.json", "p.jsonl", "t.csv")] + ["--hash-m", "8.24"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hash"]["source"] == "--hash-m 8.24" and math.isclose(out["hash"]["M_per_TMAC"], 8.24, rel_tol=1e-3)
    assert out["run"]["readings"]["tensor"]["U"]["vs_3pct"].endswith(("over", "under"))
