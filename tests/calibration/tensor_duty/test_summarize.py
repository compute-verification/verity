"""summarize.py's verdict, the falsification test of its card's `clock.tensor`: the card is the record whose `driver`
holds the GPU the run found, its bound is the card's tensor rate, a point refutes the card only by its MACs
over the host's wall clock, and a point the host stamps' resolution can't place on either side of the bound leaves the
verdict undecided."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import summarize as S
from verity_bench import kit

from verity.device import load_card, rate

CARD_FILES = kit.cards_directory()
BOUND = 132 * 4096 * 1830e6
H200, H100 = "h200-sxm-141gb", "h100-sxm-80gb"
#: the names nvidia-smi gives an H200, an H100 and a card no data sheet holds
NAME = {H200: "NVIDIA H200", H100: "NVIDIA H100 80GB HBM3", None: "NVIDIA A100-SXM4-40GB"}
SECS = 1.2


def point(of_bound: float, *, host_s: float = SECS, event_s: float = SECS, gt_s: float = SECS, **at) -> dict:
    """A duty.cu line whose MACs over ``host_s`` of the host's wall clock are ``of_bound`` of the card's rate."""
    return {"clock": "1980", "arm": "wg_s8", "class": "zero", "fill": "idle", "burst": 65536, "duty": 100,
            "host_t0": 1000.0, "host_t1": 1000.0 + host_s, "event_s": event_s, "gt_s": gt_s,
            "macs": of_bound * BOUND * host_s, "clock64_mhz": 1830.0, "tensor_mhz": 1830.0, "fill_mhz": 0.0,
            "duty_achieved": 1.0, **at}


def run(tmp_path: Path, *pts: dict, name: str = "out", sku: str = H200) -> tuple[int, dict]:
    """summarize.py's exit code and JSON verdict over one out/ directory holding ``pts``."""
    out = tmp_path / name
    out.mkdir()
    (out / "points.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pts))
    (out / "found.json").write_text(json.dumps({"name": NAME[sku]}))
    rc = S.main([str(out), "--json", str(tmp_path / f"{name}.json")])
    return rc, json.loads((tmp_path / f"{name}.json").read_text())


def test_the_bound_is_the_cards_tensor_rate_and_follows_its_clock() -> None:
    card = load_card(H200, directory=CARD_FILES)
    assert S.readings(card) == (BOUND, 132 * 4096 * 1980e6, 132 * 4096 * 1e6)
    card = copy.deepcopy(card)
    card["clock"]["tensor"]["value"] = 1900
    assert S.readings(card)[0] == 132 * 4096 * 1900e6


def test_a_card_whose_tensor_clock_is_no_ceiling_is_bound_by_its_sm_reading(tmp_path: Path) -> None:
    """The H100 marks `clock.tensor` a vendor's figure, so its bound is the tensor rate at `clock.sm`: a point at 1.0225 of the
    1830 MHz rate passes, and one over the 1980 MHz rate refutes."""
    card = load_card(H100, directory=CARD_FILES)
    sm = 132 * 4096 * 1980e6
    assert card["clock"]["tensor"]["mark"] != "ceiling" and S.ceiling_reading(card) == "sm"
    assert S.readings(card) == (sm, sm, 132 * 4096 * 1e6)
    rc, v = run(tmp_path, point(0.5), point(1.0225), name="under", sku=H100)
    assert (rc, v["verdict"], v["refuting"], v["bound_reading"]) == (0, "PASS", 0, "sm")
    assert v["top"]["of_bound"] == pytest.approx(1.0225 * BOUND / sm)
    rc, v = run(tmp_path, point(1.0225), point(1.01 * sm / BOUND), name="over", sku=H100)
    assert (rc, v["verdict"], v["refuting"]) == (1, "REFUTED", 1)


def test_points_under_the_card_pass(tmp_path: Path) -> None:
    rc, v = run(tmp_path, point(0.5), point(0.999, duty=99, fill="alu"))
    assert (rc, v["verdict"], v["refuting"], v["undecided"]) == (0, "PASS", 0, 0)
    assert v["top"]["of_bound"] == pytest.approx(0.999)


def test_a_point_over_the_card_refutes_it(tmp_path: Path) -> None:
    rc, v = run(tmp_path, point(0.5), point(1.01))
    assert (rc, v["verdict"], v["refuting"]) == (1, "REFUTED", 1)


def test_a_point_within_a_stamp_of_the_bound_is_undecided(tmp_path: Path) -> None:
    """At 1.2 s a millisecond stamp moves the rate by about 0.08%: 1.0005 of the bound may be under it, and 0.9995
    over it."""
    for of_bound in (1.0005, 0.9995):
        rc, v = run(tmp_path, point(of_bound), name=str(of_bound))
        assert (rc, v["verdict"], v["refuting"], v["undecided"]) == (1, "UNDECIDED", 0, 1)


def test_the_gpus_own_timers_do_not_judge(tmp_path: Path) -> None:
    """A point the GPU's timers put over the bound passes on the host's wall clock, and one they put under it
    doesn't."""
    fast = point(0.99, event_s=SECS * 0.98, gt_s=SECS * 0.98)
    assert fast["macs"] / fast["event_s"] > BOUND
    assert run(tmp_path, fast, name="fast")[1]["verdict"] == "PASS"
    slow = point(1.01, event_s=SECS * 1.02, gt_s=SECS * 1.02)
    assert slow["macs"] / slow["event_s"] < BOUND
    assert run(tmp_path, slow, name="slow")[1]["verdict"] == "REFUTED"


def test_nvidia_smi_lines_are_read_and_the_rest_counted(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "smi.csv").write_text("1970/01/01 00:16:40.500, 1830, 1980, 690.5, 700.00, 61, 0x0000000000000000\n"
                                 "Unable to determine the device handle for GPU0: Unknown Error\n")
    (out / "points.jsonl").write_text(json.dumps(point(0.5)) + "\n")
    (out / "found.json").write_text(json.dumps({"name": NAME[H200]}))
    assert S.main([str(out), "--json", str(tmp_path / "v.json")]) == 0
    v = json.loads((tmp_path / "v.json").read_text())
    assert (v["verdict"], v["smi_unread"]) == ("PASS", 1)
    smi, unread = S.samples(out)
    assert [s[1:] for s in smi] == [(1830.0, 690.5, 0)] and len(unread) == 1


def test_the_fills_table_gives_a_fill_and_bursts_highest_rate_and_mean_clocks() -> None:
    """A fill and burst's row is over its points at duties 90 to 99: the highest rate among them and the mean of each
    clock. A point at duty 100 is not in it."""
    pts = [point(0.5, duty=95, clock64_mhz=1800.0, tensor_mhz=1810.0, fill_mhz=1700.0, warm=False),
           point(0.25, duty=90, clock64_mhz=1820.0, tensor_mhz=1830.0, fill_mhz=1720.0, warm=False),
           point(0.75, warm=False)]
    _, lines, _ = S.report(pts, load_card(H200, directory=CARD_FILES))
    assert lines[lines.index("| --- | ---: | ---: | ---: | ---: | ---: |") + 1:] == [
        f"| idle | 65536 | {0.5 * BOUND * 1e-12:.2f} | 1810.0 | 1820.0 | 1710.0 |"]
    assert lines[0].startswith(f"**PASS**: of 3 points, 0 exceed the card's tensor rate, 132 x 4096 x clock.tensor = "
                               f"{BOUND:.4e} MACs a second")


def test_no_points_is_no_verdict(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    assert S.main([str(tmp_path / "out")]) == 2


HIGH = 132 * 4096 * 1980e6
SKIPPED = """run.sh: line 77: sudo: command not found
pl rc 127
run.sh: line 85: sudo: command not found
clock 1980 rc 127
clocks.current.sm [MHz], clocks.applications.graphics [MHz], power.limit [W], enforced.power.limit [W]
345 MHz, [Requested functionality has been deprecated], 700.00 W, 700.00 W
run.sh: line 84: sudo: command not found
clock default rc 127
"""
RAN = """Power limit for GPU 00000000:8D:00.0 was set to 700.00 W from 700.00 W.
All done.
pl rc 0
GPU clocks set to "(gpuClkMin 1980, gpuClkMax 1980)" for GPU 00000000:8D:00.0
All done.
clock 1980 rc 0
clocks.current.sm [MHz], clocks.applications.graphics [MHz], power.limit [W], enforced.power.limit [W]
345 MHz, 1980 MHz, 700.00 W, 700.00 W
All done.
clock default rc 0
"""


def recorded(tmp_path: Path, lock: str | None, name: str = NAME[H200]) -> Path:
    """An out/ holding an unlocked H200 run's fastest point (default clock, wg_s8 on zero operands), the nvidia-smi
    samples inside its host interval, ``lock`` as its lock.txt, and ``name`` as the GPU its found.json names."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    (out / "found.json").write_text(json.dumps({"name": name}))
    (out / "points.jsonl").write_text(json.dumps(
        {"clock": "default", "arm": "wg_s8", "class": "zero", "fill": "alu", "burst": 2097152, "duty": 100,
         "host_t0": 1794894758.174, "host_t1": 1794894759.374, "event_s": 1.200591, "gt_s": 1.200583,
         "macs": 1178092000000000.0, "clock64_mhz": 1815.0, "tensor_mhz": 1815.0, "fill_mhz": 0.0,
         "duty_achieved": 1.0}) + "\n")
    (out / "smi.csv").write_text("".join(
        f"2026/11/17 05:52:{s}, 1980, [Requested functionality has been deprecated], {w}, 700.00, 42, "
        "0x0000000000000000\n"
        for s, w in (("38.069", 272.52), ("38.270", 272.46), ("38.470", 272.47), ("38.670", 272.49),
                     ("38.871", 272.51), ("39.071", 272.55), ("39.272", 272.61), ("39.472", 265.80))))
    if lock is not None:
        (out / "lock.txt").write_text(lock)
    return out


def summary(out: Path, card: dict | None = None) -> dict:
    smi, _ = S.samples(out)
    pts = S.load_points(out)
    for r in pts:
        S.join(r, smi)
    return S.top_share(pts, card or S.sheet(out), lock=S.lock_result(out))


def test_the_metric_is_the_fastest_points_share_of_the_higher_ceiling_and_carries_both_readings(
        tmp_path: Path) -> None:
    t = summary(recorded(tmp_path, SKIPPED))
    rate = 1178092000000000.0 / 1.2
    assert t["reading"] == "sm" and t["share"] == pytest.approx(rate / HIGH) == pytest.approx(0.91706, abs=5e-6)
    assert t["share_lo"] < t["share"] < t["share_hi"]
    assert t["rate"] == {"value": pytest.approx(rate), "unit": "MACs a second"}
    assert t["readings"] == [
        {"reading": "tensor", "clock": {"value": 1830, "unit": "MHz"},
         "ceiling": {"value": BOUND, "unit": "MACs a second"}, "share": pytest.approx(0.99223, abs=5e-6)},
        {"reading": "sm", "clock": {"value": 1980, "unit": "MHz"},
         "ceiling": {"value": HIGH, "unit": "MACs a second"}, "share": pytest.approx(0.91706, abs=5e-6)}]
    assert t["held"] == {"smi_mean": {"value": 1980.0, "unit": "MHz"}, "smi_max": {"value": 1980.0, "unit": "MHz"},
                         "smi_samples": 6, "clock64": {"value": 1815.0, "unit": "MHz"},
                         "tensor_phase": {"value": 1815.0, "unit": "MHz"}}
    assert t["lock"] == [{"clock": {"value": 1980, "unit": "MHz"}, "ran": False, "rc": 127}]
    assert (t["sheet"], t["clock"], t["arm"], t["class"]) == (H200, "default", "wg_s8", "zero")
    json.dumps(t, allow_nan=False)


def test_the_headline_follows_the_higher_of_the_cards_readings(tmp_path: Path) -> None:
    card = copy.deepcopy(load_card(H200, directory=CARD_FILES))
    card["clock"]["tensor"]["value"] = 2100
    t = summary(recorded(tmp_path, None), card)
    assert t["reading"] == "tensor" and t["share"] == pytest.approx(1178092000000000.0 / 1.2 / (132 * 4096 * 2100e6))
    assert t["lock"] is None


def test_the_lock_is_read_from_lock_txt(tmp_path: Path) -> None:
    (tmp_path / "lock.txt").write_text(RAN)
    assert S.lock_result(tmp_path) == [{"clock": {"value": 1980, "unit": "MHz"}, "ran": True, "rc": 0}]
    (tmp_path / "lock.txt").write_text("no lock: clock and power limit as found\nclock default rc 0\n")
    assert S.lock_result(tmp_path) == []


def test_the_registered_unit_is_the_one_the_point_is_written_in() -> None:
    import tomllib

    metrics = tomllib.loads((Path(S.__file__).parent / "metrics.toml").read_text())["metric"]
    assert [(m["id"], m["unit"]) for m in metrics] == [(S.METRIC, S.UNIT)]


def test_a_build_lease_or_point_exiting_nonzero_fails_the_gate_and_a_skipped_clock_does_not(tmp_path: Path) -> None:
    assert S.gate_failures(tmp_path) == [f"no {tmp_path / 'rc.txt'}"]
    (tmp_path / "rc.txt").write_text("build rc 0\nclock 1980 rc 1\nclock 1980 skipped: not locked\n"
                                     "default wg_s8 zero idle 65536 rc 0\nlease rc 0\n")
    assert S.gate_failures(tmp_path) == []
    (tmp_path / "rc.txt").write_text("build rc 0\ndefault wg_s8 zero idle 65536 rc 139\nlease rc 2\n")
    assert S.gate_failures(tmp_path) == ["default wg_s8 zero idle 65536 rc 139", "lease rc 2"]


def test_the_card_is_the_record_whose_driver_holds_the_gpu_the_run_found(tmp_path: Path) -> None:
    for sku in (H200, H100):
        (tmp_path / "found.json").write_text(json.dumps({"name": NAME[sku]}))
        assert S.sheet(tmp_path) == load_card(sku, directory=CARD_FILES)
    (tmp_path / "found.json").write_text(json.dumps({"name": NAME[None]}))
    with pytest.raises(SystemExit, match=f"0 GPU card files under .* {NAME[None]!r}, the name the device reports"):
        S.sheet(tmp_path)
    (tmp_path / "found.json").unlink()
    with pytest.raises(SystemExit, match="0 GPU card files under .* '', the name the device reports"):
        S.sheet(tmp_path)


@pytest.fixture
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host whose harness-kit stamp names no card, whatever GPUs the test's host has."""
    from verity_bench import kit

    monkeypatch.delenv("GPU_LEASE_UUID", raising=False)
    monkeypatch.setattr(kit.shutil, "which", lambda name: None)


def written(tmp_path: Path, sku: str | None) -> tuple[Path, int, dict]:
    """The out/ of a run whose gates held on the card of ``sku``'s name, summarize.py --record's exit code on it, and
    the one measurement it wrote."""
    out = recorded(tmp_path / str(sku), SKIPPED, NAME[sku])
    (out / "rc.txt").write_text("build rc 0\ndefault wg_s8 zero alu 2097152 rc 0\nlease rc 0\n")
    rc = S.main([str(out), "--record"])
    doc = json.loads((out / "result.json").read_text())
    assert doc["validation"]["status"] == "passed"
    [m] = doc["measurements"]
    return out, rc, m


def test_the_point_names_its_sheet_and_card(tmp_path: Path, no_gpu: None) -> None:
    """Each card's point is `top_share` on its own sheet, at the readings it states, and names the card: the H200's
    member is ``h200`` and the H100's ``h100``."""
    out, rc, h200 = written(tmp_path, H200)
    t = summary(out)
    assert rc == 0 and h200["member"] == "h200" and h200["value"] == t.pop("share")
    assert {k: h200[k] for k in t} == json.loads(json.dumps(t)) and h200["sheet"] == H200
    _, rc, h100 = written(tmp_path, H100)
    card = load_card(H100, directory=CARD_FILES)
    assert rc == 0 and (h100["member"], h100["sheet"]) == ("h100", H100)
    assert [(r["reading"], r["clock"]["value"], r["ceiling"]["value"]) for r in h100["readings"]] == [
        (r, card["clock"][r]["value"], pytest.approx(float(rate(card, mhz=card["clock"][r]["value"]))))
        for r in ("tensor", "sm")]


def test_a_card_no_data_sheet_holds_writes_no_value_and_the_refusal_is_why(tmp_path: Path, no_gpu: None) -> None:
    _, rc, m = written(tmp_path, None)
    assert rc == 2 and m["value"] is None and "member" not in m
    assert m["why"].startswith("0 GPU card files under ") and repr(NAME[None]) in m["why"]
