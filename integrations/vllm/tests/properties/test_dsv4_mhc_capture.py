"""The DeepSeek-V4-Pro mHC capture without a GPU: the TF32 probe tells every step model, B reading and squared-sum contraction
apart and names each alone, its words are the Definitions', each comparison job takes the twin's words and finds a flipped one,
the reference pieces are the Definitions at the job's statics, the spy, the source scan, the bindings, the verdict and `main` end
to end (the GPU run is `tests/properties/dsv4_mhc_capture_gpu.py --out ...` on an H200)."""
from __future__ import annotations

import copy
import hashlib
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.properties import dsv4_mhc_capture_gpu as X
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity_vllm.program.kernels import mhc_twin as W
from verity_vllm.program.registry import mhc as R
from verity_vllm.properties.record import REGISTRY, digest_of

SMALL = {"HC": 2, "H": 64, "TH": 2, "TH_HEAD": 2, "TN": 2, "REPEAT": 2, "EPS": 1e-6, "HC_EPS": 1e-6}
SMALL_NS = {"fused": 2, "tf32": 2, "broadcast": 1, "head": None}
OTHER_STEP = "g4+4_w25_f139"


def _total(results: list[dict]) -> dict:
    assert not [r for r in results if "error" in r], results
    return {k: sum(r[k] for r in results) for k in X.TOTALS}


# ------------------------------------------------------------------------------------------------------------------ the probe
@pytest.fixture(scope="module")
def launches() -> list[dict]:
    return X.probe_launches()


def _models(*names: str):
    return [m for m in X.step_models() if X.model_name(m) in names]


def _got(launches, tables, step: str = X.CANDIDATE_NAME, rounding: str = "rne", sqr: str = "lo") -> list[dict]:
    """The words a GPU whose step is ``step``, whose B reading is ``rounding`` and whose squared sum is ``sqr`` would write."""
    return [{"yp": tables["rounding"][rounding][i] if L["family"] == "round" else tables["step"][step][i],
             "rp": tables["sqrsum"][sqr][i]} for i, L in enumerate(launches)]


def test_the_probe_launches_are_deepgemms_shapes(launches) -> None:
    assert [L["family"] for L in launches] == list(X.PROBE_FAMILIES)
    for L in launches:
        (M, K), (N, Kb) = L["a"].shape, L["b"].shape
        assert (M, N, K) == (X.PROBE_M, X.PROBE_N, Kb) and N <= 32 and N % 8 == 0 and K % 64 == 0 and 1 <= L["NS"] <= K // 64
        assert L["a"].dtype == np.uint16 and L["b"].dtype == np.uint32


def test_the_adder_families_read_b_exactly(launches) -> None:
    """Every launch but `round` has B words exact in TF32, so no reading of B changes its words: they pin the adder alone."""
    for L in launches:
        if L["family"] != "round":
            assert not (L["b"] & np.uint32(0x1FFF)).any(), L["family"]
            for r in W.B_ROUNDINGS:
                assert np.array_equal(W.round_tf32(L["b"], r), L["b"]), (L["family"], r)


def test_the_round_family_reads_b_alone(launches) -> None:
    """One-hot A rows: each word is a B word as the step reads it, the same under every adder of the grid."""
    i = X.PROBE_FAMILIES.index("round")
    L = launches[i]
    ys = [W.prenorm_tf32(L["a"], L["b"], L["NS"], m)[0] for m in X.step_models()[::7]]
    assert all(np.array_equal(y, ys[0]) for y in ys)
    for r in W.B_ROUNDINGS:
        y = W.prenorm_tf32(L["a"], L["b"], L["NS"], W.CANDIDATE, r)[0]
        k = np.arange(X.PROBE_M) % L["a"].shape[1]
        assert np.array_equal(y[:, 0, :], W.round_tf32(L["b"][:, k].T, r) & ~np.uint32(0x1FFF)), r


@pytest.mark.slow
def test_the_probe_names_each_hypothesis_alone(launches) -> None:
    """Every pair of the 50 step models, the three B readings and the three contractions writes different words somewhere, so a
    GPU that is any one of them is named alone; the candidate's words are GO and another's name it."""
    tables = X.probe_tables(launches)

    def digest(arrays) -> str:
        return hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays if a is not None)).hexdigest()

    for axis in ("step", "rounding", "sqrsum"):
        assert len({digest(v) for v in tables[axis].values()}) == len(tables[axis]), axis
    assert len(tables["step"]) == 50 and X.CANDIDATE_NAME in tables["step"]
    fit = X.probe_fit(launches, tables, _got(launches, tables))
    assert fit["ok"] and fit["nearest"][0] == (X.CANDIDATE_NAME, 0) and fit["words"] > 0 and not any(fit["candidate_by_family"].values())
    for step in (OTHER_STEP, "g2+2+2+2_w28_f132", "g8_w26_f139"):
        fit = X.probe_fit(launches, tables, _got(launches, tables, step=step))
        assert fit["step"] == [step] and not fit["ok"]


def test_the_probe_fit_names_each_axis(launches) -> None:
    tables = X.probe_tables(launches, _models(X.CANDIDATE_NAME, OTHER_STEP))
    fit = X.probe_fit(launches, tables, _got(launches, tables))
    assert fit["ok"] and (fit["step"], fit["rounding"], fit["sqrsum"]) == ([X.CANDIDATE_NAME], ["rne"], ["lo"])
    for kw, axis, name in (({"rounding": "rna"}, "rounding", "rna"), ({"rounding": "trunc"}, "rounding", "trunc"),
                           ({"sqr": "hi"}, "sqrsum", "hi"), ({"sqr": "mul"}, "sqrsum", "mul"), ({"step": OTHER_STEP}, "step", OTHER_STEP)):
        fit = X.probe_fit(launches, tables, _got(launches, tables, **kw))
        assert fit[axis] == [name] and not fit["ok"], kw
    got = _got(launches, tables)
    got[X.PROBE_FAMILIES.index("floor")]["yp"] = got[X.PROBE_FAMILIES.index("floor")]["yp"].copy()
    got[X.PROBE_FAMILIES.index("floor")]["yp"][0, 0, 0] ^= 1
    fit = X.probe_fit(launches, tables, got)
    assert fit["step"] == [] and not fit["ok"]
    assert fit["candidate_by_family"] == {f: int(f == "floor") for f in X.PROBE_FAMILIES if f != "round"}


def test_the_floor_rows_show_the_adders_grid() -> None:
    """The floor family's built rows: 2^-151 + 2^t and 2^-149 less that, into a zero accumulator, sum to 2^-149 under a step
    whose grid keeps 2^t (floor - width + 1 <= t) and to +0 under any other."""
    L = X.probe_launch(np.random.default_rng(3), "floor")
    for m in X.step_models():
        if len(m.groups) == 8:
            continue
        yp = W.prenorm_tf32(L["a"][:1], L["b"], L["NS"], m)[0][0]
        t = -160 + (np.arange(X.PROBE_N)[None, :] + np.arange(L["NS"])[:, None]) % 6
        assert np.array_equal(yp & np.uint32(0x7FFFFFFF), np.where(m.zero_exponent - m.width + 1 <= t, 1, 0)), X.model_name(m)


@pytest.mark.parametrize("family,rows", [("floor", (0, 40)), ("sqr", (3,)), ("special", (0, 5)), ("round", (2,)), ("cancel", (1,))])
def test_the_probe_words_are_the_definitions(launches, family: str, rows: tuple) -> None:
    """The candidate's words on the probe are `MhcPrenormTf32_v1`'s at each launch's K and NS."""
    L = launches[X.PROBE_FAMILIES.index(family)]
    yp, rp = W.prenorm_tf32(L["a"], L["b"], L["NS"])
    d = bind(R.MhcPrenormTf32, K=L["a"].shape[1], NS=L["NS"], OUT=X.PROBE_N)
    for m in rows:
        y = np.array(evaluate(d, L["a"][m].tolist(), L["b"].reshape(-1).tolist()), dtype=np.uint32)
        assert np.array_equal(y, np.concatenate([yp[m].reshape(-1), rp[m]])), (family, m)


def test_the_failed_h200_capture_is_pinned() -> None:
    """The small committed cut of r20261013-115825-580e pins the before counts and why no raw-word replay is possible."""
    path = Path(__file__).with_name("fixtures") / "dsv4_mhc_capture_20261013.json"
    rec = json.loads(path.read_text())
    assert rec["run"] == "r20261013-115825-580e" and rec["head"] == "c07829364"
    assert rec["raw_arrays_preserved"] is False
    assert rec["operand_source"].startswith("seed-regenerated from the capture's deterministic")
    assert rec["probe"] == {"step": ["g8_w26_f133"], "rounding": ["rne"], "sqrsum": [],
                            "rounding_mismatch": {"trunc": 862, "rna": 109, "rne": 0},
                            "sqrsum_mismatch": {"lo": 8, "hi": 35, "mul": 18}}
    assert rec["totals"] == {"words": 20980103, "mismatch": 943797, "ref_words": 965808, "ref_mismatch": 23903}
    assert all(v > 0 for v in rec["case_mismatches"].values())


def test_the_recorded_tf32_cases_replay_with_rne() -> None:
    """Every failed case's recorded first yp word replays from its fixture operand slice through the fixed twin and Definition."""
    path = Path(__file__).with_name("fixtures") / "dsv4_mhc_capture_20261013.json"
    rec = json.loads(path.read_text())
    seen = {}
    for key, sample in rec["samples"].items():
        aa = np.asarray(sample["a"], np.uint16).reshape(1, sample["K"])
        bb = np.asarray(sample["b"], np.uint32).reshape(1, sample["K"])
        old = int(W.tf32_split(aa, bb, b_round="trunc")[0][0, 0])
        fixed = int(W.tf32_split(aa, bb)[0][0, 0])
        definition = bind(R.MhcPrenormTf32Split, K=sample["K"], OUT=1)
        word = int(evaluate(definition, aa[0].tolist(), bb.reshape(-1).tolist())[0])
        assert (old, fixed, word) == (sample["old"], sample["kernel"], sample["kernel"]), key
        seen[key] = fixed
    assert seen.keys() == rec["case_mismatches"].keys()


# ---------------------------------------------------------------------------------------------------------------- the layers
def _case(tmp_path, phase: str, T: int, flip: tuple[str, tuple] | None = None, seed: int = 5):
    """One case at small statics as `phase_layers` saves it, the kernel's words the twin's (``flip``: one bit of one word off)."""
    S, NS = SMALL, SMALL_NS[phase]
    wts = X.weights(seed, hidden=S["H"], hc=S["HC"])
    wts["fnb"] = X.f32_words(wts["fn"].view(np.float32).reshape(-1, S["HC"], S["H"]).sum(axis=1, dtype=np.float32))
    tk = X.tokens(seed + T, T, hidden=S["H"], hc=S["HC"])
    tk = {"x": tk["x"]} if phase == "broadcast" else {"residual": tk["residual"]} if phase == "head" else tk
    got = X.twin_outputs(phase, {**wts, **tk}, NS, S)
    if flip is not None:
        got[flip[0]] = got[flip[0]].copy()
        got[flip[0]][flip[1]] ^= 1
    paths = {**X._save(tmp_path / "weights", **wts), **X._save(tmp_path / phase / f"T{T}", **tk, **got)}
    return X.layer_jobs(phase, T, NS, paths, S), got


@pytest.mark.parametrize("phase", X.PHASES)
def test_the_twin_outputs_are_the_kernels_shapes(tmp_path, phase: str) -> None:
    T, S = 3, SMALL
    _, got = _case(tmp_path, phase, T)
    hc, h, ns, out = S["HC"], S["H"], SMALL_NS[phase], S["HC"] * (2 + S["HC"])
    want = {"res": (T, hc, h), "yp": (T, ns, out), "rp": (T, ns), "out": (T, h), "post2": (T, hc), "comb2": (T, hc, hc)}
    keys = {"fused": list(want), "tf32": list(want), "broadcast": ["yp", "rp", "out", "post2", "comb2"], "head": ["out"]}[phase]
    assert sorted(got) == sorted(keys) and all(got[k].shape == want[k] for k in keys)
    assert all(got[k].dtype == (np.uint16 if k in ("res", "out") else np.uint32) for k in keys)


@pytest.mark.parametrize("phase,flip,piece", [("fused", ("yp", (2, 1, 3)), "split"), ("tf32", ("res", (0, 1, 5)), "post"),
                                              ("broadcast", ("rp", (2, 0)), "prenorm"), ("head", ("out", (0, 9)), "head")])
def test_the_layer_and_reference_jobs_take_the_kernels_words(tmp_path, phase: str, flip, piece: str) -> None:
    """At small statics each case's jobs find no mismatch on the twin's words, the reference pieces (the Definitions through the
    evaluator) on its first and last token among them; one flipped word is one mismatch, and the piece that states it sees it."""
    jobs, got = _case(tmp_path / "clean", phase, 3)
    refs = [j for j in jobs if j["kind"] == "ref"]
    assert [j["kind"] for j in jobs].count("layer") == 1 and {j["token"] for j in refs} == {0, 2}
    assert len(refs) == 2 * len(X.reference_pieces(phase, SMALL_NS[phase]))
    res = [X.compare(j) for j in jobs]
    t = _total(res)
    assert t["words"] == sum(v.size for v in got.values()) and t["mismatch"] == 0
    assert t["reference"] == len(refs) and t["ref_words"] > 0 and t["ref_mismatch"] == 0
    jobs, _ = _case(tmp_path / "flip", phase, 3, flip=flip)
    res = [X.compare(j) for j in jobs]
    t = _total(res)
    assert t["mismatch"] == 1 and res[0]["first_mismatch"]["output"] == flip[0]
    assert res[0]["first_mismatch"]["index"] == list(flip[1])
    hit = [r for j, r in zip(jobs, res) if j["kind"] == "ref" and r["ref_mismatch"]]
    assert hit and all(r["first_mismatch"]["piece"] == piece for r in hit[:1])


def test_the_expf_job_takes_the_kernels_words(tmp_path) -> None:
    rng = np.random.default_rng(8)
    x = np.concatenate([X.f32_words(rng.uniform(-110, 92, 300)), np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000],
                                                                         np.uint32)])
    got = W.expf(x)
    paths = X._save(tmp_path / "mufu", x=x, got=got)
    jobs = [{"phase": "mufu", "case": "expf", "kind": "expf", "paths": paths, "range": list(r)} for r in X._chunks(x.size, 128)]
    assert _total([X.compare(j) for j in jobs]) == {"words": x.size, "mismatch": 0, "reference": 0, "ref_words": 0,
                                                    "ref_mismatch": 0}
    got = got.copy()
    got[200] ^= 1
    paths = X._save(tmp_path / "mufu2", x=x, got=got)
    res = [X.compare({**j, "paths": paths}) for j in jobs]
    assert _total(res)["mismatch"] == 1 and res[1]["first_mismatch"]["index"] == [72]


def test_a_comparison_error_is_recorded(tmp_path) -> None:
    jobs, _ = _case(tmp_path, "head", 1)
    r = X.compare({**jobs[1], "piece": "nope"})
    assert set(r) == {"phase", "case", "error"} and "no reference piece" in r["error"]


def test_the_spy_keeps_the_partials_and_restores_the_kernel() -> None:
    calls = []

    def kernel(yp, rp, *rest, **kw):
        calls.append((rest, kw))
        yp.zero_()

    mod = types.SimpleNamespace(k=kernel)
    yp = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    rp = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    with X.Spy(mod, "k") as spy:
        mod.k(yp, rp, 7, flag=True)
    assert mod.k is kernel and calls == [((7,), {"flag": True})] and not yp.any()
    p = X._partials(spy)
    assert p["yp"].shape == (3, 2, 4) and p["rp"].shape == (3, 2)
    assert p["yp"][1, 1, 2] == np.float32(1 * 4 + 12 + 2).view(np.uint32) and p["rp"][2, 1] == np.float32(5).view(np.uint32)


def test_the_source_scan_wants_each_kernel_and_no_fast_math(tmp_path, monkeypatch) -> None:
    import sys

    class Kernel:
        def __init__(self, text):
            self.text = text

        def get_kernel_source(self):
            return self.text

    body = "float v = expf(x) * rsqrtf(y); v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1)); out = __float2bfloat16_rn(v);"
    tk = types.ModuleType("tilelang_kernels")
    for name in X.KERNELS:
        setattr(tk, name, types.SimpleNamespace(_kernel_cache={(1,): Kernel(body + name)}))
    parts = ["vllm", "vllm.model_executor", "vllm.model_executor.kernels", "vllm.model_executor.kernels.mhc"]
    for p in parts:
        monkeypatch.setitem(sys.modules, p, types.ModuleType(p))
    monkeypatch.setattr(sys.modules[parts[-1]], "tilelang_kernels", tk, raising=False)
    monkeypatch.setitem(sys.modules, parts[-1] + ".tilelang_kernels", tk)
    got = X.sources(tmp_path)
    assert got["ok"] and len(list((tmp_path / "cuda").glob("*.cu"))) == len(X.KERNELS)
    assert got["kernels"]["hc_head_fuse_tilelang"][0]["counts"] == {"expf": 1, "rsqrtf": 1, "fmaxf": 1, "__shfl_xor_sync": 1,
                                                                    "__float2bfloat16_rn": 1}
    tk.mhc_post_tilelang._kernel_cache[(2,)] = Kernel(body + "__expf(v)")
    assert X.sources(tmp_path)["ok"] is False
    del tk.mhc_post_tilelang._kernel_cache[(2,)]
    tk.hc_head_fuse_tilelang._kernel_cache.clear()
    assert X.sources(tmp_path)["ok"] is False


def test_the_programs_bind_at_dsv4s_statics() -> None:
    got = X.bind_check()
    assert got["ok"], {k: v for k, v in got["bindings"].items() if v is not True}
    for name in ("MhcFusedPostPre_v1{4,7168,4,256,64}", "MhcPostPreTf32_v1{4,7168,112,64}", "MhcPreBroadcast_v1{4,7168,28,64}",
                 "HcHead_v1{4,7168,128}", "HopperTf32WgmmaDot8_v1"):
        assert name in got["bindings"], name


def test_the_statics_are_dsv4s() -> None:
    """DeepSeek-V4-Pro's config (hidden 7168, hc_mult 4, 20 Sinkhorn iterations, hc_eps and rms_norm_eps 1e-6) and the split
    counts each served token count takes."""
    assert (X.H, X.HC, X.REPEAT, X.HC_EPS, X.EPS, X.OUT) == (7168, 4, 20, 1e-6, 1e-6, 24)
    assert max(X.FUSED_T) == 16 and min(X.TF32_T) == 17
    assert [W.tf32_num_split(T, X.HC * X.H, X.SMS) for T in X.TF32_T] == [112, 112, 66, 66, 44]
    assert [W.tf32_num_split(T, X.H, X.SMS) for T in X.BROADCAST_T] == [28, 28, 28, 28]


# ----------------------------------------------------------------------------------------------------------------- verdict
TOT = {"words": 10, "mismatch": 0, "reference": 1, "ref_words": 5, "ref_mismatch": 0}
PLAN = {"fused": [1], "tf32": [17], "broadcast": [1], "head": [1]}
CASES = [("mufu", "expf"), ("fused", "T1"), ("tf32", "T17"), ("broadcast", "T1"), ("head", "T1")]


def _doc() -> dict:
    comps = [{"phase": p, "case": c, **TOT, **({"reference": 0, "ref_words": 0} if p == "mufu" else {})} for p, c in CASES]
    return {"schema": X.SCHEMA, "plan": copy.deepcopy(PLAN), "gate": {"ok": True}, "errors": {}, "stack": {"ok": True},
            "mufu": {"ok": True}, "probe": {"ok": True}, "sources": {"ok": True}, "bind": {"ok": True},
            "launches": [{"phase": p, "case": c, "checks": {"deterministic": True}} for p, c in CASES[1:]],
            "comparisons": comps, "totals": {k: sum(r[k] for r in comps) for k in X.TOTALS}}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("errors", "layers"), "boom"), (("stack", "ok"), False),
                                        (("mufu", "ok"), False), (("probe", "ok"), None), (("sources", "ok"), False),
                                        (("bind", "ok"), False), (("totals", "mismatch"), 1), (("totals", "ref_mismatch"), 1),
                                        (("totals", "reference"), 0), (("totals", "words"), 0),
                                        (("launches", 0, "checks", "deterministic"), False), (("launches", 3, "checks"), {}),
                                        (("comparisons", 1, "mismatch"), 1), (("comparisons", 2, "reference"), 0),
                                        (("comparisons", 0, "ref_mismatch"), 1), (("comparisons", 0, "words"), 0),
                                        (("errors", "compare/head/T1"), "boom")])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = _doc()
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    assert not X._verdict(doc)


def test_each_case_has_its_own_verdict() -> None:
    assert list(X.cases(_doc())) == [f"{p}/{c}" for p, c in CASES] and all(c["ok"] for c in X.cases(_doc()).values())
    doc = _doc()
    doc["launches"] = [x for x in doc["launches"] if x["phase"] != "head"]
    doc["comparisons"] = [x for x in doc["comparisons"] if x["phase"] != "mufu"]
    assert {k for k, c in X.cases(doc).items() if not c["ok"]} == {"head/T1", "mufu/expf"}
    doc = _doc()
    doc["errors"]["compare/tf32/T17"] = "boom"
    doc["comparisons"][3]["ref_mismatch"] = 1
    assert {k for k, c in X.cases(doc).items() if not c["ok"]} == {"tf32/T17", "broadcast/T1"}


def test_the_tf32_line_reads_the_probe_and_the_tf32_cases() -> None:
    assert X.tf32_go(_doc())
    doc = _doc()
    doc["comparisons"][1]["mismatch"] = 1
    assert X.tf32_go(doc) and not X._verdict(doc)
    for k, v in (("probe", {"ok": False}), ("comparisons", None)):
        doc = _doc()
        if v is None:
            doc["comparisons"][3]["mismatch"] = 1
        else:
            doc[k] = v
        assert not X.tf32_go(doc), k


@pytest.mark.slow
def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys) -> None:
    """`main --quick` with the GPU's phases stood in for (small-statics cases, one word flipped in the broadcast's layer input)
    and vLLM's config module stubbed: the jobs through the spawned worker pool, the per-case verdicts, the sealed record, the
    TF32 and capture lines and the exit status, so a slip in the orchestration shows here and not in the H200's first hour."""
    import contextlib
    import sys

    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vc)
    for k in ("VLLM_USE_DEEP_GEMM", "VLLM_ENABLE_V1_MULTIPROCESSING", "TRITON_CACHE_DIR", "DG_JIT_CACHE_DIR", "TILELANG_CACHE_DIR"):
        monkeypatch.setenv(k, "")
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "sources", lambda outdir: {"kernels": {}, "ok": True})
    monkeypatch.setattr(X, "bind_check", lambda: {"bindings": {}, "ok": True})
    x = X.f32_words(np.linspace(-20, 20, 64, dtype=np.float32))
    expf = X._save(tmp_path / "mufu", x=x, got=W.expf(x))
    jobs = [{"phase": "mufu", "case": "expf", "kind": "expf", "paths": expf, "range": [0, 64]}]
    for phase, flip in (("fused", None), ("broadcast", ("out", (0, 3))), ("head", None)):
        jobs += _case(tmp_path / phase, phase, 1, flip=flip)[0]

    def gpu(work, outdir, plan, doc):
        doc["stack"], doc["mufu"] = {"ok": True}, {"ok": True}
        doc["probe"] = {"ok": True, "step": [X.CANDIDATE_NAME], "rounding": ["rne"], "sqrsum": ["lo"],
                        "nearest": [[X.CANDIDATE_NAME, 0]]}
        doc["launches"] += [{"phase": p, "case": "T1", "checks": {"deterministic": True}} for p in ("fused", "broadcast", "head")]
        return jobs

    monkeypatch.setattr(X, "gpu", gpu)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--quick"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"] and doc["tf32_go"] is False
    assert {c: v["ok"] for c, v in doc["cases"].items()} == {
        "mufu/expf": True, "fused/T1": True, "fused/T16": False, "tf32/T17": False, "tf32/T65": False, "broadcast/T1": False,
        "head/T1": True, "head/T16": False}
    assert doc["totals"]["mismatch"] == 1 and doc["totals"]["ref_mismatch"] >= 1 and doc["totals"]["reference"] > 0
    assert doc["env"]["VLLM_USE_DEEP_GEMM"] == "1" and doc["env"]["TILELANG_CACHE_DIR"].endswith("tilelang")
    printed = capsys.readouterr().out
    assert f"DSV4-MHC TF32 NO-GO step=['{X.CANDIDATE_NAME}'] rounding=['rne'] sqrsum=['lo']" in printed
    assert f"DSV4-MHC-CAPTURE NO-GO {doc['digest']}" in printed
    assert "no-go cases=['fused/T16', 'tf32/T17', 'tf32/T65', 'broadcast/T1', 'head/T16']" in printed


def test_the_gate_failure_writes_a_sealed_record(tmp_path, monkeypatch, capsys) -> None:
    for k in ("VLLM_USE_DEEP_GEMM", "VLLM_ENABLE_V1_MULTIPROCESSING", "TRITON_CACHE_DIR", "DG_JIT_CACHE_DIR", "TILELANG_CACHE_DIR"):
        monkeypatch.setenv(k, "")
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": False, "cc": "8.9"})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    with pytest.raises(SystemExit) as e:
        X.main(["--out", str(tmp_path / "out"), "--scratch", str(tmp_path), "--quick"])
    doc = json.loads((tmp_path / "out" / X.RECORD).read_text())
    assert e.value.code == 3 and doc["ok"] is False and doc["digest"] == digest_of(doc)
    assert "DSV4-MHC GATE-FAIL" in capsys.readouterr().out


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["dsv4_mhc_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
