"""The harness kit's four calls on CPU: the lease's argv, the host stamp with no GPU and over stand-in nvidia-smi answers,
the paired ratio and its interval against values worked by hand, and the writer's result, which research's own check takes,
with what it refuses. No test starts a program: the stamp's queries go through a stand-in."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import pytest

from verity_bench import kit

SHA = "c" * 40
RUN = "run-1"
CARD = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
GPU0, GPU1 = "GPU-11111111-2222-3333-4444-555555555555", "GPU-66666666-7777-8888-9999-000000000000"
HEADER = ("+-----------------------------------------------------------------------------------------+\n"
          "| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |\n")


def _row(uuid: str, sm: str = "2100", mem: str = "9501", name: str = CARD, driver: str = "580.95.05") -> str:
    return f"{name}, {uuid}, {driver}, {sm}, {mem}\n"


def _host(monkeypatch: pytest.MonkeyPatch, answers: dict[str, tuple[int, str]] | None, sha: str | None = SHA) -> list[list[str]]:
    """A host whose nvidia-smi answers `answers` (by `query` or `header`; None: a host without one) and whose git answers
    nothing; the queries asked, in order."""

    asked: list[list[str]] = []

    def run(argv: list[str]) -> tuple[int, str]:
        asked.append(list(argv))
        if argv[0] == "git":
            return 128, "fatal: not a git repository"
        assert argv[0] == "nvidia-smi" and answers is not None, argv
        return answers["query" if any(a.startswith("--query-gpu=") for a in argv) else "header"]

    monkeypatch.setattr(kit.shutil, "which", lambda name: None if answers is None else f"/usr/bin/{name}")
    monkeypatch.setattr(kit, "_run", run)
    monkeypatch.delenv("GPU_LEASE_UUID", raising=False)
    monkeypatch.setenv("RESEARCH_RUN_ID", RUN)
    if sha is None:
        monkeypatch.delenv("RESEARCH_SOURCE_SHA", raising=False)
    else:
        monkeypatch.setenv("RESEARCH_SOURCE_SHA", sha)
    return asked


@pytest.fixture
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    return _host(monkeypatch, None)


# ---- the lease ---------------------------------------------------------------------------------------------------------------


def test_the_lease_is_the_gpu_lease_wait_for_the_jobs_gpus_under_its_bound():
    assert kit.lease(1, 60) == ["gpu-lease", "1", "--wait", "--max-min", "60", "--"]
    assert kit.lease(8, 90.2) == ["gpu-lease", "8", "--wait", "--max-min", "91", "--"]
    for gpus in (0, -1, 1.5, True, "1"):
        with pytest.raises(ValueError, match="one GPU or more"):
            kit.lease(gpus, 60)
    for minutes in (0, -5, math.inf, math.nan, None, "60"):
        with pytest.raises(ValueError, match="its bound in minutes"):
            kit.lease(1, minutes)


# ---- the host stamp ----------------------------------------------------------------------------------------------------------


def test_the_host_stamp_with_no_gpu_says_none(no_gpu):
    assert kit.host_stamp() == {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": platform.node(),
                                "commit": SHA, "why": "no nvidia-smi on PATH"}
    assert no_gpu == []


def test_the_host_stamp_says_none_and_why_when_nvidia_smi_finds_no_gpu(monkeypatch):
    for answer, why in (((9, "NVIDIA-SMI has failed because it couldn't communicate\nwith the NVIDIA driver.\n"),
                         "nvidia-smi exited 9: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."),
                        ((127, "TimeoutExpired: 60 s"), "nvidia-smi exited 127: TimeoutExpired: 60 s"),
                        ((0, "\n"), "nvidia-smi lists no GPU"),
                        ((0, "No devices were found\n"), f"nvidia-smi answered 'No devices were found', not {', '.join(kit.QUERY)}")):
        asked = _host(monkeypatch, {"query": answer})
        stamp = kit.host_stamp()
        assert (stamp["card"], stamp["gpus"], stamp["driver"], stamp["cuda"], stamp["why"]) == ("none", [], None, None, why)
        assert len(asked) == 1


def test_the_host_stamp_takes_its_commit_from_the_run_else_the_checkout(monkeypatch):
    asked = _host(monkeypatch, None, sha=None)
    assert kit.host_stamp()["commit"] is None and [a[0] for a in asked] == ["git"] and asked[0][-2:] == ["rev-parse", "HEAD"]
    monkeypatch.setattr(kit, "_run", lambda argv: (0, SHA + "\n"))
    assert kit.host_stamp()["commit"] == SHA


def test_the_host_stamp_names_the_card_each_gpus_uuid_and_clocks_the_driver_and_cuda(monkeypatch):
    asked = _host(monkeypatch, {"query": (0, _row(GPU0) + _row(GPU1, sm="1950", mem="[N/A]")), "header": (0, HEADER)})
    assert kit.host_stamp() == {
        "card": CARD, "gpus": [{"uuid": GPU0, "sm_mhz": 2100, "mem_mhz": 9501}, {"uuid": GPU1, "sm_mhz": 1950, "mem_mhz": None}],
        "driver": "580.95.05", "cuda": "13.0", "node": platform.node(), "commit": SHA}
    assert asked == [["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem", "--format=csv,noheader,nounits"],
                     ["nvidia-smi"]]


def test_the_host_stamp_is_of_the_leases_gpus(monkeypatch):
    asked = _host(monkeypatch, {"query": (0, _row(GPU1)), "header": (1, "")})
    monkeypatch.setenv("GPU_LEASE_UUID", GPU1)
    stamp = kit.host_stamp()
    assert asked[0][:3] == ["nvidia-smi", "-i", GPU1] and [g["uuid"] for g in stamp["gpus"]] == [GPU1]
    assert stamp["cuda"] is None and "why" not in stamp
    _host(monkeypatch, {"query": (0, _row(GPU0) + _row(GPU1, name="NVIDIA H200", driver="570.1")), "header": (0, "no version")})
    stamp = kit.host_stamp()
    assert (stamp["card"], stamp["driver"], stamp["cuda"]) == (f"{CARD}, NVIDIA H200", "580.95.05, 570.1", None)


# ---- the paired ratio --------------------------------------------------------------------------------------------------------


def test_the_paired_ratio_is_the_median_of_the_reps_ratios_rep_by_rep():
    g = kit.paired([3.0, 1.0, 4.0], [2.0, 1.0, 2.0])
    assert g == {"gamma": 0.5, "ratio": 1.5, "n": 3, "ratios": [1.0, 1.5, 2.0], "ci95": None, "half_width": None}
    assert kit.paired([2.0, 1.0], [1.0, 2.0])["ratios"] == [0.5, 2.0]
    assert kit.paired([2.0, 1.0], [1.0, 2.0])["ratio"] == 1.25
    assert kit.paired([0.0], [1.0])["gamma"] == -1.0


def test_the_interval_is_the_medians_order_statistics_at_the_confidence():
    arm, stock = [1.5 + 0.001 * i for i in range(20)], [1.0] * 20
    g = kit.paired(arm[::-1], stock)
    # 20 reps at 95%: P(Binomial(20, 1/2) < 6) = 0.0207 and P(< 7) = 0.0577, so the 6th and the 15th ratios
    assert g["ratio"] == pytest.approx(1.5095) and g["gamma"] == pytest.approx(0.5095) and g["n"] == 20
    assert g["ci95"] == pytest.approx([0.505, 0.514]) and g["half_width"] == pytest.approx(0.0045 / 1.5095)
    # 6 reps: P(< 1) = 1/64, so the 1st and the 6th; 5 reps: P(< 1) = 1/32, above 0.025, so no interval
    six = kit.paired([1.1, 1.6, 1.2, 1.5, 1.3, 1.4], [1.0] * 6)
    assert six["ci95"] == pytest.approx([0.1, 0.6]) and six["half_width"] == pytest.approx(0.25 / 1.35)
    for n in (1, 2, 5):
        few = kit.paired([1.5] * n, [1.0] * n)
        assert few["ci95"] is None and few["half_width"] is None and few["n"] == n
    # at 99%: P(< 4) = 0.0013 and P(< 5) = 0.0059, so the 4th and the 17th; at 80%: P(< 7) = 0.0577 and P(< 8) = 0.1316, so
    # the 7th and the 14th
    wide, narrow = kit.paired(arm, stock, confidence=0.99), kit.paired(arm, stock, 0.8)
    assert wide["ci99"] == pytest.approx([0.503, 0.516]) and "ci95" not in wide
    assert narrow["ci80"] == pytest.approx([0.506, 0.513]) and narrow["ratio"] == g["ratio"]
    # 5 reps bound the median at 50%: P(< 2) = 0.1875 and P(< 3) = 0.5, so the 2nd and the 4th
    assert kit.paired([1.5, 1.1, 1.4, 1.2, 1.3], [1.0] * 5, 0.5)["ci50"] == pytest.approx([0.2, 0.4])


def test_an_arm_whose_reps_cost_nothing_has_its_interval_and_no_half_width():
    """A median ratio of 0 has an interval like any other, and no half-width as a share of it."""

    free = kit.paired([0.0] * 6, [1.0] * 6)
    assert free == {"gamma": -1.0, "ratio": 0.0, "n": 6, "ratios": [0.0] * 6, "ci95": [-1.0, -1.0], "half_width": None}
    # most reps free and one not: the median ratio is still 0, and the interval reaches the one that cost
    mostly = kit.paired([0.0, 0.0, 3.0, 0.0, 0.0, 0.0], [1.0, 1.0, 2.0, 1.0, 1.0, 1.0])
    assert (mostly["ratio"], mostly["ci95"], mostly["half_width"]) == (0.0, [-1.0, 0.5], None)
    # one free rep among six leaves the median above 0, and its half-width a share of it
    one = kit.paired([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0] * 6)
    assert (one["ratio"], one["ci95"], one["half_width"]) == (1.0, [-1.0, 0.0], 1.0)


def test_paired_refuses_reps_that_dont_pair_a_stock_rep_of_no_cost_and_a_confidence_out_of_range():
    for arm, stock in (([1.0], [1.0, 2.0]), ([], [])):
        with pytest.raises(ValueError, match="one for one"):
            kit.paired(arm, stock)
    for zero in (0.0, -1.0):
        with pytest.raises(ValueError, match="every stock rep above zero"):
            kit.paired([1.0, 1.0], [1.0, zero])
    for confidence in (0, 1, 95, -0.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            kit.paired([1.0], [1.0], confidence)


# ---- the writer --------------------------------------------------------------------------------------------------------------


def _result(out: Path) -> dict:
    return json.loads((out / kit.RESULT).read_text())


def test_the_writer_appends_measurements_in_the_form_a_reader_of_numbers_takes(tmp_path, no_gpu):
    from research import result as research_result

    out = tmp_path / "run" / "out"
    w = kit.writer(out)
    stamp = kit.host_stamp()
    first = w.measurement("served_tok_s", 4120.5, "tok/s")
    assert first == {"name": "served_tok_s", "value": 4120.5, "unit": "tok/s", "commit": SHA, "run": RUN, "ci95": None,
                     "target": None, "verdict": None, "inputs": [], "host": stamp}
    assert _result(out) == {"schema": kit.RESULT_SCHEMA, "run_id": RUN, "measurements": [first]}
    second = kit.writer(out).measurement("serving_cost", 0.031, "ratio", interval=(0.029, 0.034), target=0.05,
                                         verdict="within target", inputs=("served_tok_s", "stock_tok_s"))
    assert (second["ci95"], second["target"], second["verdict"]) == ([0.029, 0.034], 0.05, "within target")
    assert second["inputs"] == ["served_tok_s", "stock_tok_s"] and second["host"]["card"] == "none"
    w.measurement("reps", 20, "", inputs="served_tok_s")
    w.measurement("stock_tok_s", None, "tok/s", target=4000, verdict="cannot tell")
    doc = _result(out)
    assert [(m["name"], m["value"]) for m in doc["measurements"]] == [
        ("served_tok_s", 4120.5), ("serving_cost", 0.031), ("reps", 20), ("stock_tok_s", None)]
    assert doc["measurements"][2]["inputs"] == ["served_tok_s"]
    assert all(m.keys() >= {"value", "unit", "commit", "run"} and m["host"] == stamp for m in doc["measurements"])
    check = research_result.check_doc(out, doc, RUN)
    assert check.valid and check.errors == [] and list(out.iterdir()) == [out / kit.RESULT]


def test_the_writer_appends_to_the_result_a_runs_validation_is_in(tmp_path, no_gpu):
    envelope = {"schema": kit.RESULT_SCHEMA, "run_id": RUN, "workload_fingerprint": {"door": "run"},
                "validation": {"status": "passed", "checker": "a benchmark", "detail": "", "evidence": []},
                "measurement_files": []}
    (tmp_path / kit.RESULT).write_text(json.dumps(envelope))
    kit.writer(tmp_path).measurement("gamma", 0.5, "ratio")
    doc = _result(tmp_path)
    assert {k: doc[k] for k in envelope} == envelope and [m["name"] for m in doc["measurements"]] == ["gamma"]


def test_the_writer_refuses_a_verdict_of_record_and_a_seal(tmp_path, no_gpu):
    w = kit.writer(tmp_path)
    w.measurement("kept", 1.0, "s", target=2.0, verdict="within target")
    before = (tmp_path / kit.RESULT).read_bytes()
    for verdict in ("PASS", "FAIL", "accept", "Accepted", "rejected", "passed: within target", "the verifier accepts",
                    "sealed", "seal", "passing", "Failing", "rejection", "the proof's acceptance", "a failure", "accepting",
                    "rejecting: over target", "sealing"):
        with pytest.raises(ValueError, match="reads as a verdict of record or a seal .* nothing under benchmarks/ writes"):
            w.measurement("gate", 1.0, "s", target=2.0, verdict=verdict)
    with pytest.raises(ValueError, match="'within target' has no target"):
        w.measurement("gate", 1.0, "s", verdict="within target")
    for verdict in (True, 1, "", "  "):
        with pytest.raises(ValueError, match="says in words where its value stands against its target"):
            w.measurement("gate", 1.0, "s", target=2.0, verdict=verdict)
    for value in (True, "accept", "PASS", "sha512:" + "0" * 128, b"\x00", {"verdict": "accept"}, math.nan, math.inf):
        with pytest.raises(ValueError, match="value is a finite number.*a verdict of record or a seal is never a measurement"):
            w.measurement("gate", value, "verdict")
    assert (tmp_path / kit.RESULT).read_bytes() == before and list(tmp_path.iterdir()) == [tmp_path / kit.RESULT]
    # a target check that only holds one of those words inside another is no verdict of record
    for verdict in ("within target", "over target", "meets", "misses", "cannot tell", "surpasses target", "acceptable"):
        w.measurement(f"gate {verdict}", 1.0, "s", target=2.0, verdict=verdict)


def test_the_writer_refuses_what_isnt_a_measurement_of_this_run(tmp_path, no_gpu, monkeypatch):
    w = kit.writer(tmp_path)
    w.measurement("a", 1, "s")
    with pytest.raises(ValueError, match="already holds a measurement named a"):
        kit.writer(tmp_path).measurement("a", 2, "s")
    for kw, why in (({"name": ""}, "has a name"), ({"name": 3}, "has a name"), ({"unit": None}, "unit is a string"),
                    ({"inputs": ["a", 3]}, "inputs are the names"), ({"inputs": [""]}, "inputs are the names"),
                    ({"interval": (1.0,)}, r"interval is \(low, high\)"), ({"interval": (2.0, 1.0)}, r"interval is \(low, high\)"),
                    ({"interval": 0.5}, r"interval is \(low, high\)"),
                    ({"interval": (0.0, math.inf)}, "interval is a finite number"), ({"target": "low"}, "target is a finite number")):
        with pytest.raises(ValueError, match=why):
            w.measurement(**{"name": "b", "value": 1.0, "unit": "s"} | kw)
    assert [m["name"] for m in _result(tmp_path)["measurements"]] == ["a"]
    monkeypatch.setenv("RESEARCH_RUN_ID", "run-2")
    with pytest.raises(ValueError, match="holds run run-1's measurements, and this run is run-2"):
        kit.writer(tmp_path).measurement("b", 1, "s")
    for i, text in enumerate(('{"schema": "verity-bench/overhead/v1"}', "[]", "not json")):
        other = tmp_path / str(i)
        other.mkdir()
        (other / kit.RESULT).write_text(text)
        with pytest.raises(ValueError, match=f"isn't (a {kit.RESULT_SCHEMA} result|JSON)"):
            kit.writer(other).measurement("b", 1, "s")
        assert (other / kit.RESULT).read_text() == text


def test_a_measurement_with_no_value_says_why_and_only_it_holds_a_why(tmp_path, no_gpu):
    w = kit.writer(tmp_path)
    absent = w.measurement("capacity", None, "MAC", why="the record holds no registered device")
    assert (absent["value"], absent["why"]) == (None, "the record holds no registered device")
    assert "why" not in w.measurement("verified", 5, "MAC") and "why" not in w.measurement("free", None, "MAC")
    for value, why in ((1, "a value beside it"), (None, ""), (None, "  "), (None, 3)):
        with pytest.raises(ValueError, match="why says in words why it has no value, beside a value of None"):
            w.measurement("gate", value, "MAC", why=why)
    assert [m.get("why") for m in _result(tmp_path)["measurements"]] == ["the record holds no registered device", None, None]


def test_overhead_paired_is_the_kits_by_overheads_name_until_serving_cost_names_the_kit():
    from verity_bench import overhead

    arm, stock = [2.0, 2.1, 1.9] * 7, [1.0] * 21
    assert overhead.paired(arm, stock) == kit.paired(arm, stock)
    assert overhead.paired.__module__ == "verity_bench.overhead"
