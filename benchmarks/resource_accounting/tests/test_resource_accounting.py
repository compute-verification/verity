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


def test_gamma_is_the_pinned_twins_larger_price_at_each_linear_and_the_cube():
    """γ comes from the price twins at both FP32 prices, the larger published, on the path the draw's statement cites, at a
    cube or a Llama linear's (n, k); the layout's is the max over its linears; a shape no twin states is missing, never
    filled in."""
    P = C.price_twins()
    g = C.gamma(8192, 8192)
    assert (g["statement"], g["path"], g["pins"], g["draws"]) == ("pearl-c-sm120/v1", "cast-8.72", ["pinned"], 27_713)
    twins = [t for t in P.load() if t["statement"] == g["statement"] and t["path"] == g["path"] and t["shape"] == 8192
             and t["reading"] == "forming-credited" and t["kind"] in ("unit", "value") and t.get("cap") == "1/1000"]
    assert Fraction(g["published"]) == max(Fraction(t["gamma"]) for t in twins)
    assert {Fraction(t["fadd"]) for t in twins} == set(P.FADD_PRICES)
    linears = {name: C.gamma(n, k) for name, (n, k) in S.LLAMA31_8B.items()}
    assert all(g["pins"] == ["pinned"] and g["published"] is not None for g in linears.values())
    layout = C.gamma(None, None, C.LAYOUT)
    assert layout["pins"] == ["pinned"] and Fraction(layout["published"]) == max(Fraction(g["published"]) for g in linears.values())
    assert f"{float(Fraction(layout['published'])):.5%}" == "0.38912%"
    off = C.gamma(8192, 1024)
    assert off["published"] is None and "at (n, k) = (8192, 1024)" in off["missing"][0]


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
    assert C.compute(4096, 4096, rows, seconds, rate)["w_cert"] is not None
    off = C.compute(8192, 1024, rows, seconds, rate)
    assert off["w_cert"] is None and off["residual"] is None and off["unproved"].startswith("no γ (a forming-credited twin")
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
    return {"schema": R.SCHEMA, "run": "r-test", "status": "complete",
            "identity": {"name": "test", "memory_bytes": 96 << 30, "l2_bytes": 128 << 20, "sms": 188, "regs_per_sm": 65_536},
            "read_rate": {"bytes_per_s": 1.6e12, "kernel": "test"}, "points": [cube, o, {"key": "x", "skipped": "budget"}]}


def test_the_table_derives_every_column_from_the_run_and_the_code():
    rows = R.rows(_doc())
    by = {(r["key"], r.get("arm")): r for r in rows}
    pc = by[("cube8192-n8192-k8192-b32-L32", "pearl-c")]
    assert pc["overhead"] == pytest.approx(0.0030 / 0.0020 - 1)
    assert pc["useful_tflops"] == pytest.approx(2 * 32 * 8192 * 8192 * 32 / 0.0030 / 1e12)
    assert pc["credited_share"] == pytest.approx(1 - 1 / (32 * 32))
    assert pc["w_cert"] is not None and pc["q_cert"] == 0 and pc["s_unproved"] == C.NO_POUS
    assert pc["compute_residual"] == pytest.approx(pc["compute_honest_overhead"] + pc["compute_soundness_slack"])
    assert by[("cube8192-n8192-k8192-b32-L32", "native-fp8")]["overhead"] is None
    assert by[("cube8192-n8192-k8192-b32-L32", "native-fp8")]["gemm_tflops"] == pytest.approx(2 * 32 * 8192 ** 2 * 32 / 0.0019 / 1e12)
    assert by[("cube8192-n8192-k8192-b32-L32", "pearl-c-nohash")].get("w_cert") is None
    assert ("x", None) in by
    md, doc_no_regs = R.markdown(_doc(), {"pass": "ACCEPT x"}, "fixture"), _doc()
    assert "| cube8192 B=32 L=32 | pearl-c | whole |" in md and "failed or skipped" in md
    assert f"| compute {C.RESIDUAL} (honest + slack) | Q_cert | S_cert |" in md and "| 0 (no windows) | none |" in md
    assert ("Q_cert (red-team grant at 5761861fd, Bandwidth.SecurityProofs.AuditRO): an accepted ro-blake3/v1 window at "
            "s = 1,849 and T = 1.2 ms, no RTT (the verifier in the prover's process)") in md
    assert ("S_chip 239,992,832 bytes = l2 + sms * (regs_per_sm * 4 + 128 KiB + 160 KiB) + 1 MiB, the bandwidth harness's "
            "count with the constant and instruction caches") in md
    del doc_no_regs["identity"]["regs_per_sm"]
    assert "S_chip – (this run recorded no registers a SM, so no Q_cert)" in R.markdown(doc_no_regs, {}, "fixture")
    assert "T = 1,354,057 ns, the RTT at P2's cap (354,000 ns) inside the rebuild floor of 1,708,058 ns" in md
    assert ("not a per-run covert bound: a per-run compute residual holds only conditional on BudgetAbort for the deployed "
            "cost model (Daniel's decision 9; #1821's CovertResidualPerRun, and #1827") in md
    assert "Note: first fetches are charged to the bandwidth account, not compute (Daniel's ruling)" in md
    assert ("larger FP32 price at the point's (n, k) (#1809): cube8192 (8192, 8192) 0.36949%; llama31_8b's layout, the max "
            "over its linears, 0.38912%.") in md
    assert "sidecar" not in md
    assert ("Every arm ran beside a sidecar, an idle CUDA context in a second process held through the run (ready in 2.0 s)"
            in R.markdown({**_doc(), "sidecar": {"pid": 7, "ready_s": 2.04, "rc": 0}}, {}, "fixture"))


def test_an_arm_divides_by_the_fastest_native_of_its_precision_the_point_measured():
    """The divisor is the fastest plain GEMM of the arm's precision (the PoUW panel's rule): a tuned arm that beats torch's
    is Pearl-C's divisor and F, torch's own arm is shown over it, and a tuned arm torch beats divides by torch's."""
    assert S.divisor("pearl-c", {"native-fp8": 2.0, "native-fp8-tuned": 1.5, "native-bf16-tuned": 1.0}) == "native-fp8-tuned"
    assert S.divisor("native-bf16-tuned", {"native-bf16": 1.0, "native-bf16-tuned": 1.2}) == "native-bf16"
    assert S.divisor("pearl-c", {"native-bf16": 1.0}) is None
    assert S.fp8(("native-bf16", "native-fp8-tuned")) and not S.fp8(("native-bf16-tuned", "pearl-c"))
    assert S.weight_bytes(16, 16, 2, ("native-fp8-tuned",)) == 2 * 16 * 16 * 2 + 16 * 16 * 2
    doc = _doc()
    cube = doc["points"][0]
    cube["arms"] |= {"native-fp8-tuned": _arm(0.0016), "native-fp8-tuned-gemm": _arm(0.0015)}
    t = {"pick": {"name": "lt13_fp8-e4m3_0_2_algo66_tile20", "library": "lt13", "ms": 1.5}, "gated": 40, "tried": 52,
         "fastest_ineligible": {"name": "cutlass3x_fp8_128x128", "library": "cutlass", "ms": 1.4, "ineligible": "no scale"}}
    cube["tuned"] = {"native-fp8-tuned": t}
    doc["points"][1]["tuned"] = {"native-fp8-tuned": {"error": "Traceback ...\nRuntimeError: no eligible candidate"}}
    doc["tuner"] = {"lt_candidates": 32, "cutlass_kernels": 30, "cutlass_sched": True,
                    "versions": {"cublaslt_libs": [{"prefix": "lt", "version": 120902}, {"prefix": "lt13", "version": 130101}]}}
    by = {(r["key"], r["arm"]): r for r in R.rows(doc) if "arm" in r}
    key = "cube8192-n8192-k8192-b32-L32"
    assert by[(key, "pearl-c")]["divisor"] == "native-fp8-tuned"
    assert by[(key, "pearl-c")]["overhead"] == pytest.approx(0.0030 / 0.0016 - 1)
    assert by[(key, "native-fp8")]["overhead"] == pytest.approx(0.0020 / 0.0016 - 1)
    assert by[(key, "native-fp8-tuned")]["overhead"] is None
    assert by[(key, "native-fp8-tuned")]["bytes_per_s"] == by[(key, "native-fp8")]["bytes_per_s"] * 0.0020 / 0.0016
    plain = {(r["key"], r["arm"]): r for r in R.rows(_doc()) if "arm" in r}
    assert by[(key, "pearl-c")]["w_cert"] == plain[(key, "pearl-c")]["w_cert"]
    assert by[(key, "pearl-c")]["compute_residual"] > plain[(key, "pearl-c")]["compute_residual"]
    md = R.markdown(doc, {}, "fixture")
    assert "| +87.5% ÷ native-fp8-tuned |" in md and "| cube8192 B=32 L=32 | native-fp8-tuned | graph | 1.600 | the divisor |" in md
    assert "(32 a library: lt 120902, lt13 130101) and every CUTLASS kernel of the family (30 in the registry, in each tile" in md
    assert ("| cube8192 B=32 L=32 | native-fp8-tuned | lt13_fp8-e4m3_0_2_algo66_tile20 | lt13 | 1.500 | 40 / 52 | "
            "cutlass3x_fp8_128x128 1.400 ms (cutlass: no scale) |") in md
    assert "| o B=1 L=32 | native-fp8-tuned | refused: RuntimeError: no eligible candidate | | | | |" in md
    doc["tuner"] = {"error": "Traceback\nOSError: libcublasLt.so.12: cannot open shared object file"}
    assert "Tuned arms: no tuner (OSError: libcublasLt.so.12: cannot open shared object file)." in R.markdown(doc, {}, "fixture")


def test_the_tune_takes_the_fastest_eligible_and_times_each_library_s_best_beside_it():
    """The finalists are the fastest eligible screened candidates and each library's fastest eligible one; the kind-1
    registry kernels, which read no scale, are not the FP8 chain's."""
    T = _load("tuned")

    class Cand:
        def __init__(self, source, cutlass=None, **config):
            self.source, self.cutlass, self.config = source, cutlass, config

    assert T.ineligible("fp8-e4m3", Cand("cutlass", cutlass=3)) and T.ineligible("fp8-e4m3", Cand("verity", cutlass=9))
    assert T.ineligible("fp8-e4m3", Cand("cublaslt", library="lt")) is None and T.ineligible("bf16", Cand("cutlass", 3)) is None
    assert T.library(Cand("cublaslt", library="lt13")) == "lt13" and T.library(Cand("cutlass", 3)) == "cutlass"
    timed = [{"name": "a", "library": "lt", "ms": 1.0, "ineligible": None},
             {"name": "b", "library": "lt", "ms": 0.9, "ineligible": None},
             {"name": "c", "library": "lt", "ms": 1.1, "ineligible": None},
             {"name": "d", "library": "lt", "ms": 1.2, "ineligible": None},
             {"name": "e", "library": "lt13", "ms": 1.5, "ineligible": None},
             {"name": "f", "library": "cutlass", "ms": 0.5, "ineligible": "no scale"},
             {"name": "g", "library": "cutlass", "ms": None, "ineligible": None, "error": "prepare"}]
    assert [t["name"] for t in T.finalists(timed)] == ["b", "a", "c", "e"]
    assert T.finalists(timed[5:]) == []


def test_the_tuner_gates_refuses_a_known_bad_variant_accepted_and_keeps_only_the_pick_s_plan(monkeypatch):
    """`Tuner.pick` over a fake harness: the gate's failures are dropped, both known-bad variants are run and must be
    rejected, the FP8 pick is the fastest eligible finalist with the fastest scale-less kernel recorded beside it, and every
    cuBLASLt plan but the pick's is released, all of them when a known-bad variant passes."""
    from types import SimpleNamespace as NS
    T = _load("tuned")

    class Cand:
        def __init__(self, name, source="cublaslt", cutlass=None, plan=None, library="lt"):
            self.name, self.source, self.cutlass, self.sf_bytes = name, source, cutlass, (0, 0)
            self.config = {"library": library} if source == "cublaslt" else {}
            self.lt = None if plan is None else (NS(value=plan), None)

    lts = {"lt": [Cand("lt_a", plan=1), Cand("lt_b", plan=1), Cand("lt_bad", plan=1)], "lt13": [Cand("lt13_a", plan=2, library="lt13")]}
    cuts = [Cand("cutlass_x", "cutlass", cutlass=0)]
    released, ran, accept_bad = [], [], []

    class Call:
        def __init__(self, dev, cand, shape, ops, ws):
            self.cand = cand

        def free(self):
            pass

    def gate(dev, fam, shape, call, seed, corrupt=False):
        ran.append((call.cand.name, corrupt))
        return {"ok": bool(accept_bad) or (not corrupt and call.cand.name not in ("lt_bad", "half"))}

    B = NS(FAMILIES={"bf16": NS(name="bf16"), "fp8-e4m3": NS(name="fp8-e4m3")}, LT_CANDIDATES=32, BaselineCall=Call, gate=gate,
           cutlass_candidates=lambda dev, fam, shape, kernels, sched: (cuts, [{"name": "no"}]),
           allocate=lambda *a: NS(ops=lambda fam, s: "gate set", free=lambda dev: None),
           lt_candidates=lambda dev, fam, shape, ops, want, refused, lt: list(lts[lt]), hash_seed=hash,
           Operands=lambda *a: a, half_k_call=lambda dev, fam, shape, ops, ws: Call(dev, Cand("half", plan=3), shape, ops, ws),
           release_plans=lambda dev, cands: released.extend(c.name for c in cands))
    A = NS(Shape=lambda label, m, k, n: NS(label=label, m=m, k=k, n=n), Buffer=lambda ptr, nbytes: ptr)
    monkeypatch.setattr(T, "harness", lambda: (A, B, None))
    ms = {"lt_a": 1.0, "lt_b": 0.8, "lt13_a": 1.2, "cutlass_x": 0.5}
    monkeypatch.setattr(T.Tuner, "item", staticmethod(lambda g, reps: ms[g] / 1e3))
    monkeypatch.setattr(T.Tuner, "graph", lambda self, calls: calls[0].cand.name)

    class Tensor:
        def __init__(self, *shape):
            self.shape = shape

        def data_ptr(self):
            return id(self)

        def numel(self):
            return 1

        def element_size(self):
            return 1

    def tuner():
        t = object.__new__(T.Tuner)
        t.A, t.B, t.dev, t.kernels, t.sched, t.lt_ws, t.want, t.seed, t.picks = (
            A, B, NS(lts=["lt", "lt13"], sync=lambda: None), [], True, None, 32, 1, {})
        return t

    W, x0, ys = [Tensor(64, 32) for _ in range(4)], Tensor(8, 32), [Tensor(8, 64) for _ in range(4)]
    tu = tuner()
    pick = tu.pick("native-fp8-tuned", W, [Tensor()] * 4, Tensor(), x0, ys)
    rec = pick.record
    assert pick.cand.name == "lt_b" and rec["pick"] == {"name": "lt_b", "source": "cublaslt", "library": "lt", "config": {"library": "lt"},
                                                        "ms": 0.8}
    assert (rec["tried"], rec["gated"], rec["failed_count"], rec["cutlass_unimplementable"]) == (5, 4, 1, 1)
    assert rec["controls"] == {"flipped": {"candidate": "lt_a", "rejected": True}, "half_k": {"candidate": "half", "rejected": True}}
    assert ("lt_a", True) in ran and ("half", False) in ran
    assert set(rec["finals"]) == {"lt_b", "lt_a", "lt13_a"} and len(rec["finals"]["lt_b"]["ms"]) == T.FINAL_ROUNDS
    assert rec["fastest_ineligible"]["name"] == "cutlass_x" and rec["fastest_ineligible"]["ms"] == 0.5
    assert released == ["half", "lt13_a"]
    assert tu.pick("native-fp8-tuned", W, [Tensor()] * 4, Tensor(), x0, ys) is pick
    released.clear()
    assert tuner().pick("native-bf16-tuned", W, None, Tensor(), x0, ys).cand.name == "cutlass_x"
    assert released == ["half", "lt_a", "lt_b", "lt_bad", "lt13_a"]
    released.clear()
    accept_bad.append(True)
    with pytest.raises(RuntimeError, match="known-bad variants"):
        tuner().pick("native-bf16-tuned", W, None, Tensor(), x0, ys)
    assert released == ["half", "lt_a", "lt_b", "lt_bad", "lt13_a"]


def test_render_reads_a_chain_json_and_refuses_another_schema(tmp_path):
    p = tmp_path / "chain.json"
    p.write_text(json.dumps(_doc()))
    (tmp_path / "verify").mkdir()
    (tmp_path / "verify" / "pass.txt").write_text("noise\nACCEPT run r-test sha256:00\n")
    doc, verdicts, result, rounds = R.load(str(p))
    assert doc["run"] == "r-test" and verdicts == {"pass": "ACCEPT run r-test sha256:00"} and result is None and rounds == {}
    (tmp_path / "pous" / "bench" / "inproc").mkdir(parents=True)
    (tmp_path / "pous" / "bench" / "bench-result.json").write_text(json.dumps(_pous_result()))
    (tmp_path / "pous" / "bench" / "inproc" / "bench-result.json").write_text(json.dumps(_pous_result(certified=False)))
    p.write_text(json.dumps({**_doc(), "rounds": {"order": ["separate", "inproc"]}}))
    assert R.load(str(p))[2:] == (_pous_result(), {"inproc": _pous_result(certified=False)})
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
    bare = C.storage({**_pous_result(), "profile": None, "blocks": 1 << 19, "segments": 8, "scheme": "p2-16448/v4"})
    assert bare["unproved"] == "no profile: 19 of 20 honest audits accepted" and bare["stored"] == stored
    assert C.storage(None)["unproved"] == C.NO_POUS
    doc = _doc()
    doc["beside"] = {"window_start_ns": 0, "window_end_ns": 3_000_000_000, "window_closed_by": "audit", "harness_rc": 1,
                     "left_cpus": [2, 3], "points": {"o-n4096-k4096-b1-L32": {"native-fp8": {
                         "items": [[1, 0.0005], [2, 0.0007]], "median_s": 0.0006, "alone_median_s": 0.0004, "slowdown": 0.5,
                         "after": [0.0005] * 10, "after_median_s": 0.0005}}}}
    p = R.pous(doc, _pous_result())
    cap, memory = p["capacity"], 96 << 30
    assert cap["residual"] == pytest.approx(memory - st["s_cert"])
    assert cap["honest_overhead"] == pytest.approx(memory - stored) and cap["soundness_slack"] == pytest.approx(stored - st["s_cert"])
    assert (p["accepted"], p["audits"], p["late"], p["window_s"], p["failed_gates"]) == (19, 20, 1, 3.0, ["timed_audit"])
    assert p["beside"] == [{"key": "o-n4096-k4096-b1-L32", "arm": "native-fp8", "items": 2, "alone_ms": 0.4, "beside_ms": 0.6,
                            "slowdown": 0.5, "after_ms": 0.5, "slowdown_after": pytest.approx(0.2)}]
    md = R.markdown(doc, {}, "fixture", _pous_result())
    assert "19 of 20 audits accepted" in md and "limit 854 µs" in md and "S_cert 1.021 GB" in md
    assert ("S_cert's basis: Pous.SecurityProofs.P2SlackFamilyUncond; P2 is experimental, and the deployed ring's theorem "
            "(EndToEndRing, #1790) is held.") in md
    assert "The deployed store is 2^19 blocks, |C| = 1.078 GB of the card's 103 GB (M); lane ra-pous-scale" in md
    assert f"Capacity {C.RESIDUAL}: M − S_cert = " in md
    assert "| 0.400 | 0.600 | 0.500 | +50.0% | +20.0% | 2 |" in md
    assert "gates failed: timed_audit" in md and R.pous(_doc(), None) is None
    doc["beside"] = {**doc["beside"], "idle": True, "points": {}}
    assert "Honest audit beside the idle chain (the control): 19 of 20" in R.markdown(doc, {}, "fixture", _pous_result())


def test_the_plan_needs_no_gpu(capsys):
    assert CHAIN.main(["--plan", "--shapes", "o,gate_up", "--batches", "1,2048"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["points"] == ["o-n4096-k4096-b1-L32", "o-n4096-k4096-b2048-L32", "gate_up-n28672-k4096-b1-L32",
                              "gate_up-n28672-k4096-b2048-L32"]
    assert plan["weight_bytes"]["gate_up-L32"] == 7 * 28672 * 4096 * 32
    assert CHAIN.main(["--plan", "--rounds", "joint-v1,separate", "--tagger", "t.so", "--pause-every", "none,10",
                       "--host-bench", "bench.py", "--audits", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["args"]["pause_every"] == ["none", "10"]
    with pytest.raises(SystemExit):
        CHAIN.main(["--plan", "--rounds", "joint-v1", "--tagger", "t.so", "--pause-every", "10-noprep",
                    "--host-bench", "bench.py"])
    assert "a window cadence is none or <ms>, not '10-noprep'" in capsys.readouterr().err
    assert CHAIN.main(["--plan", "--sidecar"]) == 0 and json.loads(capsys.readouterr().out)["args"]["sidecar"] is True
    with pytest.raises(SystemExit):
        CHAIN.main(["--plan", "--sidecar", "--probe"])
    assert "not with --probe" in capsys.readouterr().err


def test_the_hosted_harness_s_argv_has_its_out_replaced():
    argv = ["--impl", "p2-v3-mc-gpu", "--out", "/o/bench", "--verifier-cpu=2"]
    assert CHAIN.flag(argv, "--out") == "/o/bench" and CHAIN.flag(argv, "--verifier-cpu") == "2"
    assert CHAIN.flag(argv, "--responder-cpu") is None
    assert CHAIN.with_flag(argv, "--out", "/o/bench/inproc") == ["--impl", "p2-v3-mc-gpu", "--out", "/o/bench/inproc",
                                                                 "--verifier-cpu=2"]
    assert CHAIN.with_flag(argv, "--verifier-cpu", "5")[-1] == "--verifier-cpu=5"
    assert CHAIN.with_flag(["--impl", "x"], "--out", "d") == ["--impl", "x", "--out", "d"]


def test_q_cert_is_the_granted_rate_and_t_holds_the_rtt_at_p2_s_cap_under_the_floor():
    """An accepted window at the red team's grant certifies PROTOCOL.md §6's 741,353,308 bytes of a 2^19-block store at
    the RTX PRO 6000's S_chip, error 2^-40.42; an uncertified tag has no Q_cert; a chain without windows certifies none;
    and the T a remote verifier would grant holds the RTT at P2's cap one ns under the rebuild floor."""
    from verity.core.protocols.bandwidth import audit
    from verity.core.protocols.pous.schemes.p2 import RTT_ALLOWANCE_CAP_NS
    p = C.window_params("ro-blake3/v1")
    assert (p.samples, p.rtt_allowance_ns, p.limit_ns) == (audit.SAMPLES, RTT_ALLOWANCE_CAP_NS, audit.REGEN_FLOOR_NS - 1)
    assert p.deadline_ns == audit.REGEN_FLOOR_NS - RTT_ALLOWANCE_CAP_NS - 1 == 1_354_057
    g = C.granted_params("ro-blake3/v1")
    assert (g.samples, g.deadline_ns, g.rtt_allowance_ns, g.limit_ns) == (1_849, 1_200_000, 0, 1_200_000)
    chip = C.chip({"l2_bytes": 128 << 20, "sms": 188, "regs_per_sm": 65_536})
    assert chip["s_chip_bytes"] == 239_992_832 and "160 KiB" in chip["formula"]
    q = C.q_cert("ro-blake3/v1", 1 << 19, 2056, chip["s_chip_bytes"])
    assert (q["q_cert"], q["side_bytes"], f"{q['error_log2']:.2f}", q["basis"]) == (741_353_308, 75_618_462, "-40.42",
                                                                                    "red-team grant at 5761861fd")
    with pytest.raises(ValueError, match="uncertified fast check"):
        C.q_cert("gf31-poly/v1", 1 << 19, 2056, chip["s_chip_bytes"])
    bw = C.bandwidth(1e12, 0.001, 5e8)
    assert (bw["q_cert"], bw["basis"], bw["theorem"]) == (0.0, C.Q_GRANT, "Bandwidth.SecurityProofs.AuditRO")
    assert (bw["residual"], bw["honest_overhead"], bw["soundness_slack"]) == (1e9, 5e8, 5e8)


J = _load("joint")


def _ticks(step: int):
    t = [0]

    def clock() -> int:
        t[0] += step
        return t[0]
    return clock


DEVICE = bytes(range(16))


@pytest.fixture(scope="module")
def small_store():
    """A real P2 store of 16 blocks (the bandwidth audit's tests' geometry) set up under the bandwidth audit's salt for
    `DEVICE`, its `vk`, and what the verifier checks the salt against."""
    from types import SimpleNamespace

    from verity.core.protocols.bandwidth import audit
    from verity.core.protocols.pous import protocol, schemes
    from verity.core.protocols.pous.schemes.p2 import P2Params
    sch = schemes.P2(P2Params(128, 173, 8, 2))
    w = bytes(range(256))[: sch.params.w_bytes]
    com, source = protocol.commit(sch, w), b"setup source"
    salt = audit.draw_salt(source, com, DEVICE)
    encoded = protocol.encode(sch, w, salt)
    vk = protocol.verifier_key(sch, com, salt, mode=protocol.PUBLIC, encoded=encoded)
    return SimpleNamespace(store=audit.Store.from_segments(sch.params, salt, encoded), vk=vk, commitment=com, source=source)


class _Commitment:
    def __init__(self, c) -> None:
        self.root, self._c = c.root, c

    def open(self, j: int, tree) -> object:
        return self._c.open(j)


class _Tagger:
    """`gpu.Tagger` on the CPU over one store (`audit.tag_pass`): a buffer is a number, a prefix pass zeroes the tags it
    skips, and once `fail_after` seeded passes have run, the next fails.  `calls` logs the buffers' uses: a pass under
    the throwaway seed as ``warm``."""

    def __init__(self, store, fail_after: int | None = None) -> None:
        self.store, self.calls, self.buffers, self.left = store, [], 0, fail_after

    def alloc(self, nbytes: int) -> int:
        self.buffers += 1
        return self.buffers

    def free(self, ptr: int) -> None:
        self.calls.append(("free", ptr))

    def memset(self, ptr: int, value: int, nbytes: int) -> None:
        self.calls.append(("memset", ptr, value))

    def read_scrub(self, ptr: int, nbytes: int) -> None:
        self.calls.append(("scrub", ptr))

    def workspace_bytes(self, tag: str, blocks: int) -> int:
        return 64 * blocks

    def last_pass_ms(self) -> tuple[float, float]:
        return 0.4, 0.5

    def tag_pass(self, tag: str, store: int, blocks: int, seed: bytes, workspace: int, stream: int = 0) -> _Commitment:
        return self.tag_pass_prefix(tag, store, blocks, blocks, seed, workspace)

    def tag_pass_prefix(self, tag: str, store: int, blocks: int, tagged: int, seed: bytes, workspace: int,
                        stream: int = 0) -> _Commitment:
        from verity.core.protocols.bandwidth import audit
        if seed == J.THROWAWAY:
            self.calls.append(("warm", tag, workspace))
        else:
            self.calls.append(("pass", tag, workspace, None if tagged == blocks else tagged))
            if self.left is not None:
                self.left -= 1
                if self.left < 0:
                    raise RuntimeError("the tagger failed")
        c = audit.tag_pass(self.store, tag, seed)
        if tagged < blocks:
            c = audit.TagCommitment(self.store, tag, seed, [*c.ys[:tagged], *(bytes(len(y)) for y in c.ys[tagged:])])
        return _Commitment(c)


def _device(setup, fail_after: int | None = None):
    """The store's one prover (`joint.Device` over `gpu.Prover`) on `_Tagger`, its raw read made up."""
    store = setup.store
    tree = J.store_key(store.params, store.salt, b"".join(store.blocks))[1]
    d = J.Device(J.load_gpu(), _Tagger(store, fail_after), 0, store.params.blocks, tree)
    d.raw_read = lambda reps: {"kernel": "none", "bytes": len(store.blocks) * store.params.block_bytes,
                               "device_ms": [0.25] * reps}
    return d


def _challenger(setup, state_dir: Path, step: int = 1_000):
    return J.Challenger.open(state_dir, setup.vk, DEVICE, setup.commitment, setup.source, clock=_ticks(step))


def _records(state_dir: Path) -> list[str]:
    (f,) = state_dir.iterdir()
    return [line.split()[0] for line in f.read_text().splitlines()[1:]]


def test_proof_only_reads_time_every_tag_and_reject_the_skip_control(small_store, tmp_path, monkeypatch):
    """(d): `vk` over the bytes of E is the store's; every tag's passes run from a scrubbed L2, and its windows go through
    `gpu.Prover` (an untimed pass, the scrub, σ, the pass, the root) to a verifier at the granted point, sharing the store's
    one state: the honest ones are accepted and on time at T, and the skip control, over a poisoned workspace, is
    rejected."""
    key, tree = J.store_key(small_store.store.params, small_store.store.salt, b"".join(small_store.store.blocks))
    assert key.root == small_store.vk.root and tree.root == small_store.vk.root
    monkeypatch.setattr(J, "SKIP", 0.25)
    prover = _device(small_store)
    d = J.alone(prover, _challenger(small_store, tmp_path), ["ro-blake3/v1", "ro-sha512/v1"], passes=3, windows=2)
    assert d["raw_read"]["device_ms"] == [0.25] * 3
    blake = d["tags"]["ro-blake3/v1"]
    assert blake["certified"] and (blake["deadline_ns"], blake["limit_ns"], blake["T_ns"]) == (1_200_000, 1_200_000,
                                                                                               1_354_057)
    assert [p["host_ns"] for p in blake["passes"]] == [1_000] * 3 and blake["passes"][0]["device_ms"] == 0.5
    assert [(w["accepted"], w["on_time_at_T"], w["elapsed_ns"]) for w in blake["windows"]] == [(True, True, 1_000)] * 2
    assert blake["skip"]["accepted"] is False and blake["skip"]["failure"] == "tag-mismatch" and blake["skip"]["tagged"] == 12
    t = J.JOINT_TAG
    window = [("warm", t, 2), ("scrub", 1), ("pass", t, 2, None)]
    one = prover.tagger.calls[:prover.tagger.calls.index(("free", 2)) + 1]
    assert one == [("memset", 1, 0), ("warm", t, 2), *[("scrub", 1), ("pass", t, 2, None)] * 3, *window * 2,
                   ("warm", t, 2), ("memset", 2, 0xA5), ("scrub", 1), ("pass", t, 2, 12), ("free", 2)]
    assert not prover.prover.workspaces
    assert _records(tmp_path) == ["open", "close"] * 6


def test_joint_windows_commit_while_the_audit_runs_and_are_judged_after(small_store, tmp_path):
    """(e): windows open at random gaps until the honest audit ends, each in a workspace of its own, and are judged once
    closed; one past the granted 1.2 ms is late (no samples drawn) though on time at T, one past T is neither; a window
    that fails stops the thread and is recorded."""
    gaps: list[float] = []
    prover = _device(small_store)
    j = J.Joint(prover, _challenger(small_store, tmp_path / "e"), lambda: len(j.pending) >= 3, slots=5, gap_s=2.0,
                wait=lambda s: gaps.append(s) or False)
    j.start()
    out = j.close()
    assert len(out["windows"]) == 3 and all(w["accepted"] and w["on_time_at_T"] for w in out["windows"])
    assert out["error"] is None and not out["out_of_slots"] and len(gaps) == 4 and all(1.0 <= g <= 3.0 for g in gaps)
    assert [c[2] for c in prover.tagger.calls if c[0] == "pass"] == [2, 3, 4]
    assert sorted(c[1] for c in prover.tagger.calls if c[0] == "free") == [2, 3, 4, 5, 6]
    for step, on_time in ((1_300_000, True), (1_400_000, False)):
        late = J.Joint(_device(small_store), _challenger(small_store, tmp_path / str(step), step), lambda: False, slots=2,
                       wait=lambda s: False)
        late.run()
        w = late.close()
        assert w["out_of_slots"] and [(x["failure"], x["on_time_at_T"]) for x in w["windows"]] == [("late", on_time)] * 2
    broken = J.Joint(_device(small_store, fail_after=1), _challenger(small_store, tmp_path / "broken"), lambda: False,
                     slots=3, wait=lambda s: False)
    broken.run()
    w = broken.close()
    assert len(w["windows"]) == 1 and "the tagger failed" in w["error"]
    assert _records(tmp_path / "broken") == ["open", "close"] * 2
    spans = [{"start_ns": 100, "elapsed_ns": 50}, {"start_ns": 1_000, "elapsed_ns": None}]
    o = J.overlap([[0, 2e-8], [120, 1e-9], [150, 1e-9], [-200, 2e-8]], 10, spans)
    assert (o["with"], o["without"], o["median_with_s"], o["median_without_s"]) == (2, 2, pytest.approx(1.05e-8),
                                                                                     pytest.approx(1.05e-8))


def test_the_hosted_rounds_report_the_responder_s_isolation_and_the_joint_round_s_arms(small_store, tmp_path, monkeypatch):
    monkeypatch.setattr(J, "SKIP", 0.25)
    d = J.alone(_device(small_store), _challenger(small_store, tmp_path / "d"), ["gf31-poly/v1", "ro-blake3/v1"],
                passes=2, windows=1)
    d.update(store={"blocks": 1 << 19, "block_bytes": 2056, "bytes": (1 << 19) * 2056, "root": "00"}, device="GPU-test")
    j = J.Joint(_device(small_store), _challenger(small_store, tmp_path / "e"), lambda: len(j.pending) >= 2, slots=4,
                wait=lambda s: False)
    j.run()
    w = j.close()
    key, arm = "o-n4096-k4096-b1-L32", "native-fp8"
    start = w["windows"][0]["start_ns"]
    items = [[start, 0.0005], [start - 10**9, 0.0004]]
    pts = {key: {arm: {"items": items, "chains": 4, "median_s": 0.00045, "alone_median_s": 0.0004, "slowdown": 0.125,
                       "after": [0.0004] * 10, "after_median_s": 0.0004}}}
    doc = _doc()
    doc["rounds"] = {"bench": "benchmarks/pous/bench.py", "argv": [], "order": ["separate", "inproc", "joint"],
                     "separate": {"round": "separate", "responder": "its own process", "harness_rc": 1, "points": pts},
                     "inproc": {"round": "inproc", "error": "Traceback\nRuntimeError: no device"},
                     "joint": {"round": "joint", "responder": "a thread of the chain's process", "harness_rc": 1,
                               "points": pts, "alone": d, "windows": w,
                               "overlap": {key: {arm: J.overlap(items, 4, w["windows"])}}}}
    results = {"separate": _pous_result(certified=False), "joint": _pous_result()}
    rounds = {r["round"]: r for r in R.hosted(doc, results)}
    assert list(rounds) == ["separate", "inproc", "joint"] and rounds["inproc"]["audits"] is None
    assert rounds["joint"]["proof_reads"]["raw_gb_per_s"] == pytest.approx((1 << 19) * 2056 / 0.25 / 1e6)
    e = rounds["joint"]["joint"]
    assert (e["windows"], e["on_time_at_T"], e["accepted"], e["alone_device_ms_p50"]) == (2, 2, 2, 0.5)
    assert e["overlap"] == [{"key": key, "arm": arm, "with": 1, "without": 1, "median_with_ms": pytest.approx(0.5),
                             "median_without_ms": pytest.approx(0.4)}]
    md = R.markdown(doc, {}, "fixture", None, results)
    assert ("| separate | its own process | 19 of 20 | 2100 | 1 | 0 | 200 / 600 / 900 | -46 | 854 | B=1 L=32 native-fp8 +12.5% "
            "| unproved | exit 1, gates failed: timed_audit |") in md
    assert "| inproc | – | failed: RuntimeError: no device |" in md and "| 1.021 | exit 1" in md
    assert "S_cert (round joint) 1.021 GB" in md and "The deployed store is 2^19 blocks" in md
    assert "| gf31-poly/v1 | no | 0.500 / 0.500 | 0.400 |" in md and "| ro-blake3/v1 | yes |" in md
    assert "| 1 / 1 / 1 | 0.001 | – | rejected (tag-mismatch) |" in md and "| 1 / 1 / 1 | 0.001 | 0.741 GB | rejected" in md
    assert "= 4312 GB/s" in md and "under the bandwidth audit's salt for GPU-test" in md
    assert ("committed within T = 1,354,057 ns (the RTT at P2's cap inside the rebuild floor), and accepted when it "
            "committed within the granted 1,200,000 ns with its openings. ro-blake3/v1's accepted window: Q_cert "
            "741,353,308 bytes (68.8% of E) within 1,200,000 ns, error 2^-40.42 (Bandwidth.SecurityProofs.AuditRO, "
            "red-team grant at 5761861fd, S_chip 239,992,832 bytes).") in md
    assert ("2 of 2 on time at T (1,354 µs), 2 accepted within the granted 1,200 µs, together Q_cert 1.483 GB, failures "
            "none") in md
    assert f"| {key} | {arm} | 1 | 0.500 | 1 | 0.400 |" in md


def test_a_window_cadence_is_none_or_some_ms():
    assert [(e.ms, e.label) for e in J.EVERY] == [(None, "none"), (10, "10ms"), (100, "100ms"), (1000, "1000ms")]
    for bad in ("0", "10ms", "10-noprep", "-5", ""):
        with pytest.raises(ValueError, match="a window cadence is none or <ms>"):
            J.Every.parse(bad)


def test_the_joint_rounds_refuse_a_store_under_pous_s_own_salt(tmp_path):
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="under PoUS's own salt, which the bandwidth verifier refuses"):
        J.store_setup("libbwtag.so", None, SimpleNamespace(params=None, device=None), "GPU-x", tmp_path)
    with pytest.raises(ValueError, match="no registered P2 params"):
        J.store_setup("libbwtag.so", None, SimpleNamespace(device=DEVICE), "GPU-x", tmp_path)


class _Chain:
    """`joint.Chain` on the CPU: each mark the next integer, a quarter of a ms of device time apart."""

    def __init__(self) -> None:
        self.calls, self.n = [], 0

    def launch(self) -> None:
        self.calls.append("launch")

    def mark(self) -> int:
        self.n += 1
        return self.n

    def wait(self, m: int) -> None:
        self.calls.append(("wait", m))

    def drain(self) -> None:
        self.calls.append("drain")

    def gap(self, a: int, b: int) -> float:
        return 0.25 * (b - a)


def _pause(setup, state_dir, chains, every, prover=None, seconds=0.05):
    return J.Pause(prover or _device(setup), _challenger(setup, state_dir, 1_000_000), chains,
                   [J.Every.parse(x) for x in every], seconds=seconds, judge_each=1, cover_ms=2.0)


def test_the_pause_hook_drains_prepares_and_times_each_window(small_store, tmp_path, monkeypatch):
    """(f): a segment without windows only issues the chain, at most `lookahead` chains in flight; with a cadence, each
    window opens while the chain runs, drains it at a chain boundary and commits through `gpu.Prover` (an untimed pass,
    the scrub, σ, the pass into a workspace of its own when it is to be judged, the root); on the root the chain resumes
    with `burst` chains queued, on a verifier's clock held at the root, so the window's elapsed time is σ to the root
    alone, which the judged windows' verdicts agree with; the skip control is rejected."""
    monkeypatch.setattr(J, "SKIP", 0.25)
    key, arm, chain = "o-n4096-k4096-b1-L32", "native-fp8", _Chain()
    p = _pause(small_store, tmp_path, {(key, arm): chain}, ("none", "10", "20"))
    calls, t = p.prover.tagger.calls, J.JOINT_TAG
    assert calls == [("memset", 1, 0), ("warm", t, 2), *[("scrub", 1), ("warm", t, 2)] * J.ALONE_START]
    assert p.chain_ms == {(key, arm): 0.0625} and p.burst == {(key, arm): 32}
    segs = [p.next() for _ in range(3)]
    out = p.close()
    passes = out["passes"]
    assert len(passes["start"]) == len(passes["end"]) == J.ALONE_START
    assert [(s["key"], s["arm"], len(s["passes"])) for s in passes["segments"]] == [(key, arm, J.ALONE_SEGMENT)]
    first = passes["start"][0]
    assert first["t_ns"] - out["anchor"]["clock_ns"] == 1_000_000 and set(out["anchor"]) == {"clock_ns", "unix_ns"}
    assert {k: v for k, v in first.items() if k != "t_ns"} == {"host_ns": 1_000_000, "device_tags_ms": 0.4, "device_ms": 0.5}
    assert [s["every"] for s in segs] == ["none", "10ms", "20ms"] and out["error"] is None
    assert segs[0]["windows"] == [] and segs[0]["chains"] == 49 and segs[0]["wall_ns"] == 51_000_000
    assert chain.calls[:10] == ["launch"] * 5 + ["drain", "launch", "launch", ("wait", 3), "launch"]
    ten, twenty = segs[1]["windows"], segs[2]["windows"]
    assert len(ten) == 2 and len(twenty) == 1 and out["windows_started"] == 4
    assert [w["slot"] for w in ten] == [1, 0] and twenty[0]["slot"] == 2
    w = ten[0]
    assert (w["open_ns"], w["drain_ns"], w["prep_ns"], w["seed_ns"], w["pass_ns"], w["resume_ns"], w["commit_ns"],
            w["pause_ns"], w["bubble_ms"]) == (1_000_000,) * 7 + (4_000_000, 0.25)
    assert all(x["elapsed_ns"] == 1_000_000 and x["on_time_at_T"] and x["drawn"] for x in ten + twenty)
    assert segs[1]["chains"] == 3 * 9 + 2 * 32 and segs[2]["chains"] == 2 * 19 + 32
    windows = [i for i, c in enumerate(calls) if c[0] == "pass"]
    assert [calls[i][2:] for i in windows] == [(3, None), (2, None), (4, None), (2, 12)]
    assert all(calls[i - 2:i] == [("warm", t, calls[i][2]), ("scrub", 1)] for i in windows[:3])
    assert calls[windows[3] - 3:windows[3]] == [("warm", t, 2), ("memset", 2, 0xA5), ("scrub", 1)]
    assert [(j["window"], j["accepted"], j["elapsed_agrees"]) for j in out["judged"]] == [(0, True, True), (2, True, True)]
    assert out["skip"]["accepted"] is False and out["skip"]["failure"] == "tag-mismatch" and out["skip"]["tagged"] == 12
    assert sorted(c[1] for c in calls if c[0] == "free") == [2, 3, 4] and not p.prover.prover.workspaces
    assert _records(tmp_path) == ["open", "close"] * 4
    assert (out["deadline_ns"], out["limit_ns"], out["T_ns"], out["cover_ms"]) == (1_200_000, 1_200_000, 1_354_057, 2.0)

    store = {"blocks": 1 << 19, "block_bytes": 2056, "bytes": (1 << 19) * 2056}
    r = {"pause": out, "pause_setup": {"store": store, "device": "GPU-test"}, "macs": {key: 4096 * 4096 * 32}}
    f = R.pause(r, 239_992_832)
    rows = {x["every"]: x for x in f["rows"]}
    none = 49 / 0.051
    assert rows["none"]["chains_per_s"] == pytest.approx(none) and rows["none"]["loss"] is None
    ten_row = rows["10ms"]
    rate = segs[1]["chains"] / (segs[1]["wall_ns"] / 1e9)
    assert ten_row["chains_per_s"] == pytest.approx(rate) and ten_row["loss"] == pytest.approx(1 - rate / none)
    assert ten_row["tflops_lost"] == pytest.approx(2 * 4096 * 4096 * 32 * (none - rate) / 1e12)
    assert ten_row["on_time_at_T"] == ten_row["within_limit"] == ten_row["windows"] == 2
    assert ten_row["q_cert_bytes_per_s"] == pytest.approx(2 * 741_353_308 / (segs[1]["wall_ns"] / 1e9))
    assert ten_row["model_loss"] == pytest.approx(2 / (segs[1]["wall_ns"] / 1e9) * 0.25 / 1e3)
    assert (f["judged"], f["judged_accepted"], f["elapsed_agrees"]) == (2, 2, True)
    doc = _doc()
    doc["rounds"] = {"order": ["joint-v1"], "joint-v1": {"round": "joint-v1", "responder": "a thread", "harness_rc": 0,
                                                         "points": {}, **r}}
    md = R.markdown(doc, {}, "fixture", None, {"joint-v1": _pous_result()})
    assert "(f) Joint v1, round joint-v1: the bandwidth audit's honest overhead." in md
    assert "then resumed the chain with 2 ms of chains queued before the verifier drew its samples" in md
    assert ("2 of 2 judged accepted (their elapsed time the timing record's: True); the skip control (25% untagged) "
            "rejected (tag-mismatch). An accepted window: Q_cert 741,353,308 bytes (68.8% of E) within 1,200,000 ns") in md
    assert f"| B=1 L=32 | native-fp8 | 10ms | {rate:.1f} | " in md and "| 2 / 2 | 2 / 2 | 1.000 / 1.000 |" in md
    assert "on time at T is within T = 1,354,057 ns (the RTT at P2's cap inside the rebuild floor)" in md
    assert "(tag kernel / whole pass): 10ms: 0.400 / 0.500; 20ms: 0.400 / 0.500." in md
    assert (f"at its start ({J.ALONE_START}) 0.400 / 0.500; after each segment without windows ({J.ALONE_SEGMENT}) 0.400 / "
            f"0.500; at its end ({J.ALONE_START}) 0.400 / 0.500.") in md


def test_a_failed_window_stops_the_windows_and_the_segments_turn_one_later_each_round(small_store, tmp_path):
    chains = {("a", "native-fp8"): _Chain(), ("b", "native-fp8"): _Chain()}
    p = _pause(small_store, tmp_path, chains, ("none", "10"), prover=_device(small_store, fail_after=2))
    segs = [p.next() for _ in range(8)]
    out = p.close()
    assert [(s["key"], s["every"]) for s in segs] == [("a", "none"), ("a", "10ms"), ("b", "none"), ("b", "10ms"),
                                                      ("a", "10ms"), ("b", "none"), ("b", "10ms"), ("a", "none")]
    assert "the tagger failed" in out["error"] and out["skip"] is None
    assert [w["window"] for w in segs[1]["windows"]] == [0, 1] and [w["window"] for w in segs[3]["windows"]] == [None]
    assert all(not s["windows"] for s in segs[4:])
    assert [j["window"] for j in out["judged"]] == [0] and out["judged"][0]["accepted"]
    assert [x["windows"] for x in R.pause_rows(out, {}, None)] == [0, 2, 0, 0]


def test_the_probe_summarizes_each_condition_and_renders_its_table():
    P = _load("probe")
    items = [[0.0012, 0.0011, 0.0001, None, 3, 2_000_000, 30], [0.0014, 0.0013, 0.0003, None, 4, None, 30]]
    s = P.summarize(items)
    assert s == {"n": 2, "host_ms": pytest.approx(1.3), "device_ms": pytest.approx(1.2), "launch_ms": pytest.approx(0.2),
                 "cpu_mhz": 2000.0}
    held = [[0.002, 0.0011, 0.0001, 0.004, 3, None, 30], [0.002, 0.0011, 0.0002, 0.005, 3, None, 30]]
    assert P.summarize(held)["queued"] is False and P.summarize(held[:1])["queued"] is True
    assert P.cpulist("0-2,5,7-8\n") == {0, 1, 2, 5, 7, 8} and P.node_of(5, {0: [0, 1], 1: [5]}) == 1
    doc = _doc()
    doc["probe"] = {"point": "o-n4096-k4096-b1-L32", "numa": {"gpu_node": 1},
                    "phases": {"alone": {"clocks_before": {"clocks.sm": "2092"},
                                         "summary": {"native-fp8/default": {**s, "nodes": [1]},
                                                     "native-fp8/default+held": {**P.summarize(held[:1]), "nodes": [1]}}},
                               "near": {"skipped": "no CPU of this mask on a node at the GPU's"}}}
    md = R.markdown(doc, {}, "fixture")
    assert "Probe at o-n4096-k4096-b1-L32 (`probe.py`)" in md and "(the GPU's node is 1)" in md and "alone 2092, near –" in md
    assert "| alone | native-fp8/default | 1.3000 | 1.2000 | 0.2000 | 2000 | 1 |" in md
    assert "| alone | native-fp8/default+held (queued) | 2.0000 | 1.1000 | 0.1000 | – | 1 |" in md
    assert "Probe near: no CPU of this mask on a node at the GPU's" in md


def test_the_course_finds_the_onset_and_the_offset_of_a_co_tenant_s_effect():
    P = _load("probe")
    slow, fast = 1.318e-3, 1.209e-3
    times = [(-2.0, slow), (-1.0, slow), (-0.5, slow), (1.0, slow), (4.0, fast), (6.0, fast), (10.9, fast), (11.2, fast),
             (12.5, fast), (13.1, slow), (13.5, fast), (14.0, slow), (14.6, slow)]
    series = [[t, "native-bf16", d] for t, d in times] + [[t, "native-fp8", 0.928e-3] for t in (-1.0, 5.0, 12.0)]
    marks = {"ready": 4.8, "close": 10.8, "exited": 11.0}
    s = P.course_summary(series, marks, after_s=4)
    bf = s["native-bf16"]
    assert (bf["before_ms"], bf["ready_ms"], bf["closing_ms"]) == (pytest.approx(1.318), pytest.approx(1.209),
                                                                   pytest.approx(1.209))
    assert bf["starting_ms"] == pytest.approx((1.318 + 1.209) / 2) and bf["effect"] == pytest.approx(0.109 / 1.318)
    assert bf["after_ms"] == [pytest.approx(1.209), pytest.approx(1.209), pytest.approx((1.318 + 1.209) / 2),
                              pytest.approx(1.318)]
    assert bf["onset_s"] == 4.0 and bf["offset_s"] == pytest.approx(3.0)
    fp = s["native-fp8"]
    assert fp["effect"] == pytest.approx(0.0) and fp["onset_s"] is None and fp["offset_s"] is None
    never = P.course_summary([[t, "native-bf16", fast] for t in (11.5, 12.0)] + [[-1.0, "native-bf16", slow]] +
                             [[5.0, "native-bf16", fast]], marks, after_s=2)
    assert never["native-bf16"]["offset_s"] is None
    doc = _doc()
    clocks = {"clocks.sm": "2092", "clocks.mem": "12481"}
    doc["probe"] = {"point": "o-n4096-k4096-b1-L32", "phases": {},
                    "course": {"context": {"count": 1, "marks": marks, "summary": s, "clocks_before": clocks,
                                           "clocks_after": clocks},
                               "spinners": {"count": 4, "marks": {}, "summary": {}, "error": "1 of 4 never got ready in 60 s"}}}
    md = R.markdown(doc, {}, "fixture")
    assert "context 2092/12481 to 2092/12481; spinners –/– to –/–." in md
    assert ("| context | native-bf16 | 4.80 | 1.318 | 1.263 | 1.209 | +8.3% | 4.00 | 1.209, 1.209, 1.263, 1.318 | 3.00 |"
            in md)
    assert "| context | native-fp8 | 4.80 | 0.928 | – | 0.928 | +0.0% | – | –, 0.928, –, – | – |" in md
    assert "Course spinners ×4: 1 of 4 never got ready in 60 s" in md


def test_the_decay_probe_summarizes_each_burst_before_and_after():
    P = _load("probe")
    pt = lambda tags, at=None: {**({} if at is None else {"at_ms": at}), "tags_ms": tags, "raw_ms": 0.72, "l2_ms": 0.1}
    s = P.decay_summary([{"kind": "gemm-o-fp8", "seconds": 0.01, "pre": [pt(0.84), pt(0.86), pt(0.83)],
                          "post": [pt(1.2, 0), pt(0.9, 1000)]}])
    assert s == [{"kind": "gemm-o-fp8", "seconds": 0.01, "before": {"tags_ms": 0.84, "raw_ms": 0.72, "l2_ms": 0.1},
                  "after": [{"at_ms": 0, "tags_ms": 1.2, "raw_ms": 0.72, "l2_ms": 0.1},
                            {"at_ms": 1000, "tags_ms": 0.9, "raw_ms": 0.72, "l2_ms": 0.1}]}]
    with pytest.raises(SystemExit):
        P.main(["decay", "--out", "x.json"])


def test_the_rounds_chain_times_carry_the_device_s_and_the_launch_s():
    turns = object.__new__(CHAIN.Turns)
    turns.plans = {"k": {"arms": {"native-fp8": {"chains": 4, "median_s": 0.002, "device_median_s": 0.0019}}}}
    pts = turns.summary({"k": {"native-fp8": [[0, 0.002, 0.0019, 0.0001], [1, 0.004, 0.0039, 0.0003]]}},
                        {("k", "native-fp8"): [(0.001, 0.0009, 0.0002)] * 3})
    r = pts["k"]["native-fp8"]
    assert (r["median_s"], r["device_median_s"], r["launch_median_s"]) == (pytest.approx(0.003), pytest.approx(0.0029),
                                                                           pytest.approx(0.0002))
    assert (r["after"], r["after_device_median_s"], r["after_launch_median_s"]) == ([0.001] * 3, 0.0009, 0.0002)
    row = R.beside_rows(pts)[0]
    assert (row["device_ms"], row["launch_ms"], row["after_launch_ms"]) == (pytest.approx(2.9), pytest.approx(0.2),
                                                                            pytest.approx(0.2))
