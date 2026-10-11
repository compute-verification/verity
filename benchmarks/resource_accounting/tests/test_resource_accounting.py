"""The matmul chain's plan, its certified columns and its renderer (`benchmarks/resource_accounting`), on a CPU."""

from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


def _load(name: str):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


S = _load("sweep")
C = _load("certified")
R = _load("render")
CHAIN = _load("chain")


def test_every_chain_streams_its_weights_from_device_memory():
    """A length below the shape's floor is raised to it: L weights of the smallest format the arms read are at least
    twice the L2."""
    l2 = 128 << 20
    assert S.min_length(4096, 4096, l2, S.ARMS) == 16
    assert S.min_length(4096, 4096, l2, ("native-bf16",)) == 8
    pts = S.points({"o": (4096, 4096)}, (1, 32), (4, 32), l2, S.ARMS)
    assert [p.key for p in pts] == ["o-n4096-k4096-b1-L16", "o-n4096-k4096-b32-L16", "o-n4096-k4096-b1-L32", "o-n4096-k4096-b32-L32"]
    assert all(p.length * 4096 * 4096 >= 2 * l2 for p in pts)
    assert pts[1].macs == 32 * 4096 * 4096 * 16


def test_shapes_parse_by_name_and_refuse_what_they_dont_name():
    assert S.parse_shapes("llama") == S.LLAMA31_8B
    assert S.parse_shapes("o,8192x1024") == {"o": (4096, 4096), "n8192k1024": (8192, 1024)}
    with pytest.raises(ValueError, match="not a shape name"):
        S.parse_shapes("attn")


def test_gamma_is_the_pinned_twins_larger_price_at_the_cube_only():
    """γ comes from the price twins at both FP32 prices, the larger published, on the path the draw's statement cites;
    outside the pinned cubes it is missing, never filled in."""
    P = C.price_twins()
    g = C.gamma(8192, 8192)
    assert (g["statement"], g["path"], g["pins"], g["draws"]) == ("pearl-c-sm120/v1", "cast-8.72", ["pinned"], 27_713)
    twins = [t for t in P.load() if t["statement"] == g["statement"] and t["path"] == g["path"] and t["shape"] == 8192
             and t["reading"] == "forming-credited" and t["kind"] in ("unit", "value") and t.get("cap") == "1/1000"]
    assert Fraction(g["published"]) == max(Fraction(t["gamma"]) for t in twins)
    assert {Fraction(t["fadd"]) for t in twins} == set(P.FADD_PRICES)
    llama = C.gamma(4096, 14336)
    assert llama["published"] is None and "a pinned cube for (n, k) = (4096, 14336)" in llama["missing"][0]


def test_w_cert_and_the_residual_split():
    """W_cert = (1 − γ)(1 − ε_s)·ΣW_ref over the credited cells, and the residual is its two parts' sum."""
    rows, seconds, rate = 32 * 32, 0.01, 4e14
    w = C.compute(8192, 8192, rows, seconds, rate)
    wref = C.wref_cell(8192, 8192) * 8192 * rows
    assert w["wref"] == float(wref)
    assert w["w_cert"] == float((1 - Fraction(w["gamma"])) * (1 - C.EPS_S) * wref)
    assert w["capacity"] == rate * seconds
    assert w["residual"] == pytest.approx(w["honest_overhead"] + w["soundness_slack"])
    assert 1.08 < float(wref) / (8192 * 8192 * rows) < 1.1
    off = C.compute(4096, 4096, rows, seconds, rate)
    assert off["w_cert"] is None and off["residual"] is None and "pinned cube" in off["unproved"]
    assert off["honest_overhead"] == pytest.approx(rate * seconds - off["wref"])


def _arm(median: float, chains: int = 4, **extra) -> dict:
    return {"chains": chains, "seconds": [median] * 3, "median_s": median, "roots": [], **extra}


def _doc() -> dict:
    """A recorded run's shape, with made-up times: a cube point with every arm and a Llama point with Pearl-C refused."""
    pc = {"mode": "whole", "screened_rows": 4 * 32 * 32, "failing_rows": 0, "unsure_rows": 4, "roots_repeat": True}
    cube = {"key": "cube8192-n8192-k8192-b32-L32", "shape": "cube8192", "n": 8192, "k": 8192, "batch": 32, "length": 32,
            "macs": 32 * 8192 * 8192 * 32, "gates": [], "refused": {}, "clocks": [{"sm_mhz": 2400}],
            "arms": {"native-bf16": _arm(0.0040), "native-bf16-gemm": _arm(0.0038), "native-fp8": _arm(0.0020),
                     "native-fp8-gemm": _arm(0.0019), "pearl-c": _arm(0.0030, **pc), "pearl-c-nohash": _arm(0.0025, mode="whole")}}
    o = {"key": "o-n4096-k4096-b1-L32", "shape": "o", "n": 4096, "k": 4096, "batch": 1, "length": 32, "macs": 4096 * 4096 * 32,
         "gates": [], "refused": {"pearl-c": "a gate"}, "arms": {"native-fp8": _arm(0.0004), "native-fp8-gemm": _arm(0.0003)}}
    return {"schema": R.SCHEMA, "run": "r-test", "status": "complete", "identity": {"name": "test", "memory_bytes": 96 << 30},
            "read_rate": {"bytes_per_s": 1.6e12, "kernel": "test"}, "points": [cube, o, {"key": "x", "skipped": "budget"}]}


def test_the_table_derives_every_column_from_the_run_and_the_code():
    rows = R.rows(_doc())
    by = {(r["key"], r.get("arm")): r for r in rows}
    pc = by[("cube8192-n8192-k8192-b32-L32", "pearl-c")]
    assert pc["overhead"] == pytest.approx(0.0030 / 0.0020 - 1)
    assert pc["useful_tflops"] == pytest.approx(2 * 32 * 8192 * 8192 * 32 / 0.0030 / 1e12)
    assert pc["credited_share"] == pytest.approx(1 - 1 / (32 * 32))
    assert pc["w_cert"] is not None and pc["q_unproved"] == C.UNPROVED_Q and pc["s_unproved"] == C.NO_POUS
    assert pc["compute_residual"] == pytest.approx(pc["compute_honest_overhead"] + pc["compute_soundness_slack"])
    assert by[("cube8192-n8192-k8192-b32-L32", "native-fp8")]["overhead"] is None
    assert by[("cube8192-n8192-k8192-b32-L32", "native-fp8")]["gemm_tflops"] == pytest.approx(2 * 32 * 8192 ** 2 * 32 / 0.0019 / 1e12)
    assert by[("cube8192-n8192-k8192-b32-L32", "pearl-c-nohash")].get("w_cert") is None
    assert ("x", None) in by
    md = R.markdown(_doc(), {"pass": "ACCEPT x"}, "fixture")
    assert "| cube8192 B=32 L=32 | pearl-c | whole |" in md and "unproved" in md and "failed or skipped" in md


def test_render_reads_a_chain_json_and_refuses_another_schema(tmp_path):
    p = tmp_path / "chain.json"
    p.write_text(json.dumps(_doc()))
    (tmp_path / "verify").mkdir()
    (tmp_path / "verify" / "pass.txt").write_text("noise\nACCEPT run r-test sha256:00\n")
    doc, verdicts, result = R.load(str(p))
    assert doc["run"] == "r-test" and verdicts == {"pass": "ACCEPT run r-test sha256:00"} and result is None
    (tmp_path / "pous" / "bench").mkdir(parents=True)
    (tmp_path / "pous" / "bench" / "bench-result.json").write_text(json.dumps(_pous_result()))
    assert R.load(str(p))[2] == _pous_result()
    p.write_text(json.dumps({**_doc(), "schema": "other/v0"}))
    with pytest.raises(SystemExit, match="not ra-matmul-chain/v1"):
        R.load(str(p))


def _pous_result(certified: bool = True) -> dict:
    """The harness's bench-result.json, cut to what the renderer reads: P2's deployed point, made-up answer times."""
    sp = {"blocks": 1 << 19, "block_bits": 16448, "k": 105, "limit_ns": 854_000, "state_bound_bits": (1 << 19) * 16448 * 18 // 19,
          "certificate": {"theorem": "Pous.SecurityProofs.P2SlackFamilyUncond", "delta": "1/100", "eps_log2": -128},
          "gaps": [], "certified": certified}
    audit = {"audits": 20, "audits_accepted": 19, "answers": 2100, "late": 1, "wrong": 0,
             "elapsed_s": {"n": 2100, "min": 1e-4, "p50": 2e-4, "p99": 4e-4, "p99.9": 6e-4, "max": 9e-4},
             "margin_s": {"min": -4.6e-5}, "calibration_here": {"rtt_s": {"p50": 5e-5, "p99.9": 9e-5}}}
    return {"passed": False, "gates": {"timed_audit": {"passed": False}, "value_control": {"passed": True}},
            "metrics": {"audit": audit, "deadline": {"limit_ns": 854_000, "limit_fixed_before_load": True}},
            "profile": {"certified": certified, "gaps": [] if certified else ["operating point: k is not the frozen value"],
                        "experimental": "P2 remains experimental", "storage_profile": sp}}


def test_the_pous_arm_reports_its_audit_s_cert_and_the_chain_beside_it():
    """S_cert is the storage profile's ⌊ρ·|C|⌋ in bytes only when it is certified, and the capacity residual splits into
    M − |C| and |C| − S_cert; without a profile it is unproved with the honest audit's count; the chain's slowdown is its
    time beside the audit over its time alone."""
    st = C.storage(_pous_result())
    stored = (1 << 19) * 16448 / 8
    assert st["stored"] == stored and st["s_cert"] == (1 << 19) * 16448 * 18 // 19 / 8
    assert st["theorem"] == "Pous.SecurityProofs.P2SlackFamilyUncond" and st["unproved"] is None
    assert "operating point" in C.storage(_pous_result(certified=False))["unproved"]
    assert C.storage({**_pous_result(), "profile": None})["unproved"] == "no profile: 19 of 20 honest audits accepted"
    assert C.storage(None)["unproved"] == C.NO_POUS
    doc = _doc()
    doc["beside"] = {"window_start_ns": 0, "window_end_ns": 3_000_000_000, "window_closed_by": "audit", "harness_rc": 1,
                     "left_cpus": [2, 3], "points": {"o-n4096-k4096-b1-L32": {"native-fp8": {
                         "items": [[1, 0.0005], [2, 0.0007]], "median_s": 0.0006, "alone_median_s": 0.0004, "slowdown": 0.5}}}}
    p = R.pous(doc, _pous_result())
    cap, memory = p["capacity"], 96 << 30
    assert cap["residual"] == pytest.approx(memory - st["s_cert"])
    assert cap["honest_overhead"] == pytest.approx(memory - stored) and cap["soundness_slack"] == pytest.approx(stored - st["s_cert"])
    assert (p["accepted"], p["audits"], p["late"], p["window_s"], p["failed_gates"]) == (19, 20, 1, 3.0, ["timed_audit"])
    assert p["beside"] == [{"key": "o-n4096-k4096-b1-L32", "arm": "native-fp8", "items": 2, "alone_ms": 0.4, "beside_ms": 0.6,
                            "slowdown": 0.5}]
    md = R.markdown(doc, {}, "fixture", _pous_result())
    assert "19 of 20 audits accepted" in md and "limit 854 µs" in md and "S_cert 1.021 GB" in md and "| +50.0% | 2 |" in md
    assert "gates failed: timed_audit" in md and R.pous(_doc(), None) is None
    doc["beside"] = {**doc["beside"], "idle": True, "points": {}}
    assert "Honest audit beside the idle chain (the control): 19 of 20" in R.markdown(doc, {}, "fixture", _pous_result())


def test_the_plan_needs_no_gpu(capsys):
    assert CHAIN.main(["--plan", "--shapes", "o,gate_up", "--batches", "1,2048"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["points"] == ["o-n4096-k4096-b1-L32", "o-n4096-k4096-b2048-L32", "gate_up-n28672-k4096-b1-L32",
                              "gate_up-n28672-k4096-b2048-L32"]
    assert plan["weight_bytes"]["gate_up-L32"] == 7 * 28672 * 4096 * 32
