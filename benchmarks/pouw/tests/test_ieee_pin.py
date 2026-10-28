"""`harness/ieee_pin.py`, the SASS gate's numeric pin: a pinned build's run passes only when every check ran all its inputs,
none differs from its reference, the float checks saw subnormal inputs and outputs, and it ran on the leased GPU (an sm_90
card runs the checks of kernels' units.cu besides); a control
passes only when it differs somewhere. Where a CUDA toolkit is present, `ieee_pin.cu` built with it must pin exactly that
toolkit's entry in `sass_pins.json`, with every pinned build passing the gate and every control failing it."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "harness"))

import ieee_pin as IP  # noqa: E402
import sass_gate as SG  # noqa: E402

UUID = "GPU-00112233-4455-6677-8899-aabbccddeeff"


def good_run(sm: int = 120) -> dict:
    lines = [{"device": "NVIDIA RTX PRO 6000 Blackwell" if sm == 120 else "NVIDIA H200", "uuid": UUID, "sm": sm,
              "sms": 188 if sm == 120 else 132}]
    for name, n in {**IP.CASES, **(IP.UNITS_CASES if sm == 90 else {})}.items():
        skipped = 0 if name in IP.FLOAT else 3
        lines.append({"check": name, "cases": n - skipped, "skipped": skipped,
                      "subnormal_in": 7 if name in IP.FLOAT else 0,
                      "subnormal_out": 5 if name in IP.FLOAT and name not in IP.SQRT else 0, "bad": 0, "first": [0] * 4})
    return {"exit": 0, "stderr": "", "lines": lines, "lease_uuid": UUID.upper()}


def with_check(run: dict, name: str, **kw) -> dict:
    return {**run, "lines": [{**ln, **kw} if ln.get("check") == name else ln for ln in run["lines"]]}


def test_an_exact_run_on_the_leased_gpu_passes():
    assert IP.judge_run(good_run(), control=False) == []


@pytest.mark.parametrize("run,why", [
    (with_check(good_run(), "div", bad=1, first=[1, 2, 3, 4]), "div: 1 differ from the reference, first [1, 2, 3, 4]"),
    (with_check(good_run(), "u64", cases=1), "u64: 1 cases and 3 skipped, not 1073741824 inputs"),
    (with_check(good_run(), "rcp", subnormal_out=0), "rcp: 7 subnormal inputs, 0 subnormal outputs"),
    (with_check(good_run(), "host_sqrt", subnormal_in=0), "host_sqrt: 0 subnormal inputs"),
    (with_check(good_run(), "cdiv3", skipped=1, cases=IP.CASES["cdiv3"] - 1), "cdiv3: 7 subnormal inputs, 5 subnormal outputs, 1 skipped"),
    ({**good_run(), "lines": [ln for ln in good_run()["lines"] if ln.get("check") != "u16_call"]}, "no u16_call check"),
    ({**good_run(), "lines": good_run()["lines"] + [{"check": "rsqrt", "cases": 1, "skipped": 0, "subnormal_in": 0,
                                                   "subnormal_out": 0, "bad": 0, "first": [0] * 4}]}, "unknown checks"),
    ({**good_run(), "lease_uuid": "GPU-ffffffff-4455-6677-8899-aabbccddeeff"}, "not the leased"),
    ({**good_run(), "lines": [{**good_run()["lines"][0], "sm": 90}] + good_run()["lines"][1:]}, "ran on sm_90"),
    ({**good_run(), "exit": 2, "stderr": "CUDA error"}, "exit 2: CUDA error"),
    ({"exit": None, "lines": []}, "no device line"),
])
def test_a_run_that_differs_skips_or_sees_no_subnormals_fails(run, why):
    assert any(why in p for p in IP.judge_run(run, control=False)), IP.judge_run(run, control=False)


def test_a_control_must_differ_somewhere():
    assert "exact everywhere" in IP.judge_run(good_run(), control=True)[0]
    assert IP.judge_run(with_check(good_run(), "host_div", bad=9), control=True) == []


def test_every_shipped_toolkit_names_a_run_of_these_builds():
    """A repin that changes a body drops its toolkit's run (`sass_gate.update_pins`), and a toolkit without one exempts
    nothing; a change to the pinned builds needs a new run too."""
    for name, tk in SG.load_pins()["toolkits"].items():
        want = [" ".join(["nvcc", *IP.nvcc_args(IP.arch_of(name)), *flags]) for flags in IP.PINNED]
        assert tk["numeric"] and re.fullmatch(r"r\d{8}-\d{6}-[0-9a-f]{4}", tk["numeric"]["run"]), name
        assert tk["numeric"]["builds"] == want, name


def test_a_toolkit_names_its_target_and_a_run_has_one():
    assert (IP.arch_of("13.0.1"), IP.arch_of("13.0.1-sm_90a")) == ("sm_120a", "sm_90a") and IP.nvcc_args("sm_120a") == IP.NVCC_ARGS
    assert "arch=compute_90a,code=sm_90a" in IP.nvcc_args("sm_90a")
    with pytest.raises(IP.PinFailed, match="no target sm_100a"):
        IP.arch_of("13.0.1-sm_100a")
    pins = {"toolkits": {"13.0.1": {}, "13.0.1-sm_90a": {}}}
    with pytest.raises(IP.PinFailed, match="more than one target"):
        IP.homes(["13.0.1=/a", "13.0.1-sm_90a=/b"], pins)
    assert IP.homes(["13.0.1-sm_90a=/b"], pins) == {"13.0.1-sm_90a": "/b"}
    assert IP.judge_run(good_run(90), control=False, sm=90) == []
    assert any("not sm_90" in p for p in IP.judge_run(good_run(), control=False, sm=90))


def test_only_an_sm90_card_runs_the_units_checks():
    """units.cu's bodies are sm_90a's, so an H200's run must hold all five checks to them, subnormals included, and an
    RTX PRO 6000's run has none of them."""
    h200 = good_run(90)
    assert IP.UNITS_CASES and set(IP.UNITS_CASES) <= IP.FLOAT and not set(IP.UNITS_CASES) & set(IP.CASES)
    for name in IP.UNITS_CASES:
        dropped = {**h200, "lines": [ln for ln in h200["lines"] if ln.get("check") != name]}
        assert f"no {name} check" in IP.judge_run(dropped, control=False, sm=90)
        assert any(p.startswith(f"{name}: 1 differ") for p in IP.judge_run(with_check(h200, name, bad=1), control=False, sm=90))
    assert any("subnormal outputs" in p for p in IP.judge_run(with_check(h200, "units_soft_one", subnormal_out=0),
                                                              control=False, sm=90))
    assert IP.judge_run(with_check(h200, "units_sqrt", subnormal_out=0), control=False, sm=90) == []
    sm120 = {**good_run(), "lines": good_run()["lines"] + [ln for ln in h200["lines"] if ln.get("check") == "units_rms_rs"]}
    assert any("unknown checks ['units_rms_rs']" in p for p in IP.judge_run(sm120, control=False))


@pytest.mark.slow
def test_ieee_pin_cu_pins_the_entry_of_the_toolkit_that_builds_it(tmp_path):
    """The CPU stage with the toolkit the gate finds (CUDA_HOME, POUW_CUDA_BIN or /usr/local/cuda): its entry in
    sass_pins.json is what ieee_pin.cu's pinned builds pin, those builds pass the gate covering every sequence and pinned
    body, and the controls fail it. It catches an edit to ieee_pin.cu, the pins or the gate after which the numeric run
    would test other sequences than the pinned ones, which no listing can show. Skipped without a pinned toolkit."""
    try:
        tools = SG.find_tools()
    except SG.SassGateFailed as ex:
        pytest.skip(str(ex))
    home = Path(tools["cuobjdump"]["path"]).parent.parent
    if not (home / "bin" / "nvcc").exists():
        pytest.skip(f"no nvcc in {home / 'bin'}")
    version = subprocess.run([str(home / "bin" / "nvcc"), "--version"], capture_output=True, text=True, check=True).stdout
    pins = SG.load_pins()
    tk = next((n for n, t in pins["toolkits"].items() if t["version"] in version and t["branch"] in version), None)
    if tk is None:
        pytest.skip(f"no pinned toolkit is {version.strip().splitlines()[-1]}")
    for t in pins["toolkits"].values():
        t["numeric"] = {"pending": "test"}
    pending = tmp_path / "pins.json"
    pending.write_text(json.dumps(pins))
    rec = IP.build_toolkit(tk, str(home), tmp_path, SG.PINS, pending)
    assert rec["problems"] == []
    assert [(b["control"], b["gate"]["ok"]) for b in rec["builds"]] == [(False, True)] * 2 + [(True, False)] * 3
