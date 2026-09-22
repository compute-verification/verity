"""CPU tests for W11's independent model of vLLM's batch-invariant Triton ``_rms_norm_kernel``
(``verity_vllm_numerics.rms_triton_model`` / ``rms_relation``) on the saved fixtures
``fixtures/W11R-40f0cebeb670-20260907T1802Z``.

No GPU: every expected output is a BF16 word the production kernel wrote on the RTX 4090 (natural
Qwen2.5-1.5B serving inputs of ``layers.0.input_layernorm``; §11 stress inputs; one-bit mutations).
What is pinned down:

* the MUFU.SQRT / MUFU.RCP tables reconstruct from the shipped deltas to the hashed hardware dumps and
  the table-driven ``sqrt.approx.ftz`` / ``div.full`` models behave as measured on special values;
* every natural call (prefill and decode rows) and every stress family reproduce the production words
  bit-for-bit -- including inf / NaN / subnormal operands, eps = 0, eps < 0 and other n_cols;
* one-bit flips of an x / w word and eps changes move the model's output exactly where production's moved;
* the dataflow alternatives that are arithmetically distinct from the compiled kernel mismatch production
  (the two that are provably equivalent for BF16 inputs do not);
* the coverage gate: ``independent_rmsnorm`` predicts only inside the measured family and raises
  ``RmsUnresolved`` (``unresolved: ...``) elsewhere.

Needs g++ (or $CXX) for ``cpp/rms_triton_model.cpp``; skips where it cannot be built.
Run: ``cd vllm-poc && python -m pytest tests/test_rmsnorm_triton_model.py -q``
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FIX = ROOT / "fixtures" / "W11R-40f0cebeb670-20260907T1802Z"

from verity_vllm_numerics import rms_triton_model as RM  # noqa: E402
from verity_vllm_numerics import rms_relation as RR  # noqa: E402


@pytest.fixture(autouse=True)
def _rms_tables_env(monkeypatch):
    # Per-test, not import-time (see test_rmsnorm_cuda_model.py, which needs the W11C tables).
    monkeypatch.setenv("VERITY_RMS_TABLES", str(FIX / "tables"))

pytestmark = pytest.mark.skipif(not FIX.exists(), reason="fixture directory missing")


def _have_cpp() -> bool:
    try:
        RM.lib()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_cpp = pytest.mark.skipif(not _have_cpp(), reason="cpp/rms_triton_model.cpp could not be built (no g++)")

SQRT_RAW_SHA256 = "4e017678de306e0be8bed66193c2ebf09e9c60a9600cad56cd05bb7b6c3f266f"  # hw dump, 2^24 x u32 (rms_probe table)


@pytest.fixture(scope="module")
def tables() -> RM.RmsTables:
    return RM.RmsTables.load(FIX / "tables")


@pytest.fixture(scope="module")
def natural() -> dict:
    return json.loads((FIX / "natural" / "index.json").read_text())


@pytest.fixture(scope="module")
def stress() -> dict:
    return json.loads((FIX / "stress" / "index.json").read_text())


def _w(p: Path) -> np.ndarray:
    return np.load(p).view(np.uint16)


def _call(fix: Path, idx: dict, c: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _w(fix / c["x_file"]).reshape(c["rows"], c["n_cols"])
    out = _w(fix / c["out_file"]).reshape(c["rows"], c["n_cols"])
    w = _w(fix / idx["weights"][c["weight"]]["file"])
    return x, w, out


# ---- tables -----------------------------------------------------------------------------------------

def test_sqrt_table_reconstructs_to_hardware_dump(tables):
    assert tables.sqrt.shape == (2 * RM.TABLE_N,) and tables.sqrt.dtype == np.uint32
    assert hashlib.sha256(tables.sqrt.tobytes()).hexdigest() == SQRT_RAW_SHA256
    assert tables.sqrt[0] == 0x3F800000  # sqrt(1) = 1
    assert tables.sqrt[RM.TABLE_N] == 0x3FB504F3  # sqrt(2)
    d = np.diff(tables.sqrt.astype(np.int64))
    assert d.min() == -1 and d.max() == 1
    # the hardware approximation is NOT monotone: 4 one-ulp dips at segment boundaries of the sqrt(1.m) table
    assert np.nonzero(d < 0)[0].tolist() == [2097151, 3014655, 3145727, 6029311]
    # ... and within one ulp of the correctly rounded sqrt everywhere (2,602,502 entries 1 ulp low, 227,301 1 ulp high)
    m = np.arange(RM.TABLE_N, dtype=np.float64) / RM.TABLE_N
    cr = np.concatenate([np.sqrt(1 + m), np.sqrt(2 * (1 + m))]).astype(np.float32).view(np.uint32).astype(np.int64)
    err = tables.sqrt.astype(np.int64) - cr
    assert err.min() == -1 and err.max() == 1 and int((err == 0).sum()) == 13947413


@needs_cpp
def test_sqrt_and_div_full_special_values(tables):
    x = np.array([4.0, 2.0, 1e-6, 0.0, -0.0, np.inf, -1.0, 1e-40, 0.25, 2.0 ** -126, 2.0 ** 100], np.float32)
    y = RM.mufu_sqrt(x, tables)
    assert y[0] == 2.0 and y[3] == 0.0 and np.signbit(y[4]) and y[5] == np.inf and np.isnan(y[6])
    assert y[7] == 0.0  # ftz: subnormal input flushed
    assert y[8] == 0.5 and y[9] == 2.0 ** -63 and y[10] == 2.0 ** 50
    assert y[1].view(np.uint32) == 0x3FB504F3
    assert y[2].view(np.uint32) == np.float32(0.001).view(np.uint32) - 1  # sqrt(1e-6): the hardware is one ulp below 1e-3
    r = RM.div_full(np.array([1.0, 1.0, 1.0, 1.0, 5.0, 1.0], np.float32), np.array([1536.0, 0.0, np.inf, 5e-39, 2.0, 1e-39], np.float32), tables)
    assert abs(r[0] - 1 / 1536) <= 2 * np.spacing(np.float32(1 / 1536))
    assert r[1] == np.inf and r[2] == 0.0 and r[4] == 2.5
    assert np.isfinite(r[3]) and abs(r[3] - 2e38) < 1e34  # subnormal divisor: the 2^24 range scaling keeps 1/5e-39 finite
    assert r[5] == np.inf  # 1/1e-39 = 1e39 > FLT_MAX


# ---- natural inputs -----------------------------------------------------------------------------------

@needs_cpp
def test_natural_calls_bit_exact(tables, natural):
    """Every natural call of the fixture (3,146 rows x 1536 = 4,832,256 words): 0 mismatching words."""
    words = 0
    for c in natural["calls"]:
        x, w, hw = _call(FIX / "natural", natural, c)
        md = RM.rms_norm_rows(x, w, c["eps"], tables)
        assert (md == hw).all(), f"natural seq {c['seq']} ({c['phase']}): {(md != hw).sum()} mismatching words"
        words += hw.size
    assert words == 4832256
    assert {c["n_cols"] for c in natural["calls"]} == {1536} and {c["eps"] for c in natural["calls"]} == {1e-6}


def test_natural_fixture_identity(natural):
    assert natural["kernel_metadata"]["num_warps"] == 4 and natural["kernel_metadata"]["name"] == "_rms_norm_kernel"
    assert natural["profiler_kernels"]["_rms_norm_kernel"] == natural["n_calls"] == 300
    assert natural["kernel_files"]["_rms_norm_kernel.cubin"]["sha256"]
    assert sum(c["rows"] for c in natural["calls"]) == 3146


# ---- stress / mutations -------------------------------------------------------------------------------

@needs_cpp
def test_stress_families_bit_exact(tables, stress):
    fams = {}
    for c in stress["calls"]:
        x, w, hw = _call(FIX / "stress", stress, c)
        md = RM.rms_norm_rows(x, w, c["eps"], tables)
        assert (md == hw).all(), f"stress {c['family']}/{c['case']}: {(md != hw).sum()} mismatching words"
        fams[c["family"]] = fams.get(c["family"], 0) + hw.size
        assert c["new_specialisations"] == 0  # same compiled kernel as the natural run
    assert set(fams) >= {"zeros", "absorption", "rounding_boundaries", "subnormals", "extremes", "near_uniform", "random_scale", "n_cols", "beyond_natural_range"}
    assert sum(fams.values()) == 3094656


@needs_cpp
def test_non_finite_and_zero_eps_semantics(tables, stress):
    """inf / NaN operands, BF16 max (squares overflow), eps = 0 with fp32-subnormal squares: the model follows the hardware."""
    by_case = {c["case"]: c for c in stress["calls"]}
    for case in ("inf_in_x", "nan_in_x", "nan_in_w", "bf16_max_row", "fp32_subnormal_squares_eps0", "bf16_subnormal_rows_eps0", "negative_eps", "eps_inf"):
        c = by_case[case]
        x, w, hw = _call(FIX / "stress", stress, c)
        md = RM.rms_norm_rows(x, w, c["eps"], tables)
        assert (md == hw).all(), case
    x, w, hw = _call(FIX / "stress", stress, by_case["nan_in_x"])
    assert (hw == 0x7FFF).all()  # NaN in the sum -> every output the canonical BF16 NaN
    x, w, hw = _call(FIX / "stress", stress, by_case["fp32_subnormal_squares_eps0"])
    assert (hw[np.nonzero(x)] & 0x7FFF).min() == 0x7F80  # sqrt.approx.ftz flushed the subnormal mean -> rcp(0) = inf -> inf words


@needs_cpp
def test_mutations_tracked(tables, stress):
    muts = json.loads((FIX / "stress" / "mutations.json").read_text())
    calls = {c["seq"]: c for c in stress["calls"]}
    n = 0
    for m in muts["mutations"]:
        base = calls[m["base_seq"]]
        x, w, hw_base = _call(FIX / "stress", stress, base)
        x, w = x.copy(), w.copy()
        eps = base["eps"]
        if m["kind"] == "x_bit_flip":
            x[m["row"], m["col"]] ^= np.uint16(1 << m["bit"])
        elif m["kind"] == "w_bit_flip":
            w[m["col"]] ^= np.uint16(1 << m["bit"])
        else:
            eps = m["eps"]
        hw_mut = _w(FIX / "stress" / m["out_file"]).reshape(hw_base.shape)
        md_mut = RM.rms_norm_rows(x, w, eps, tables)
        md_base = RM.rms_norm_rows(*_call(FIX / "stress", stress, base)[:2], base["eps"], tables)
        assert m["production_output_words_changed"] > 0
        assert (md_mut == hw_mut).all(), m
        assert (md_mut != md_base).sum() == (hw_mut != hw_base).sum() > 0
        n += 1
    assert n == 36


@needs_cpp
def test_dataflow_alternatives_mismatch(tables, stress, natural):
    """The distinct dataflow choices each break exactness somewhere; the two BF16-equivalent ones cannot."""
    by_case = {c["case"]: c for c in stress["calls"]}
    sample = [by_case[k] for k in ("within_1ulp_of_tie", "exact_bf16_ties_pre_cast", "scale_2^-60", "constant_rows", "wide_dynamic_range_2^-12..2^12")]
    nat = [c for c in natural["calls"] if c["phase"] == "prefill"][:1]
    mism = {}
    for name, v in RM.VARIANTS.items():
        n = 0
        for c in sample:
            x, w, hw = _call(FIX / "stress", stress, c)
            n += int((RM.rms_norm_rows(x, w, c["eps"], tables, variant=v) != hw).sum())
        for c in nat:
            x, w, hw = _call(FIX / "natural", natural, c)
            n += int((RM.rms_norm_rows(x, w, c["eps"], tables, variant=v) != hw).sum())
        mism[name] = n
    assert mism["model"] == 0
    for name in ("no_eps_fusion", "sequential_sum", "ieee_sqrt", "ieee_rcp_div", "fp64_sum", "out_xw_first"):
        assert mism[name] > 0, name
    # x1^2 first vs x0^2 first and fma vs separate rounding of x^2 are identities for BF16 inputs (x^2 is exact in fp32)
    assert mism["thread_order_x0_first"] == 0 and mism["unfused_squares"] == 0


# ---- relation / coverage gate ----------------------------------------------------------------------------

def _node(op="verity.bi_rms_norm", eps=1e-6, **extra):
    return {"op_type": "rmsnorm", "extra": {"runtime_op": op, "consts": {"eps": eps, **extra}}}


@needs_cpp
def test_relation_predicts_natural_row(tables, natural):
    c = natural["calls"][0]
    x, w, hw = _call(FIX / "natural", natural, c)
    out = RR.independent_rmsnorm(_node(), {"x": x[7], "w": w})
    assert out == hw[7].astype("<u2").tobytes()
    cov = RR.coverage(_node(), {"x": x[7], "w": w})
    assert cov["covered"] and cov["n_cols"] == 1536


@needs_cpp
def test_gate_unresolved_outside_measured_family(tables, natural):
    c = natural["calls"][0]
    x, w, _ = _call(FIX / "natural", natural, c)
    with pytest.raises(RR.RmsUnresolved, match="unresolved: runtime_op"):
        RR.independent_rmsnorm(_node(op="_C.fused_add_rms_norm"), {"x": x[0], "w": w})
    with pytest.raises(RR.RmsUnresolved, match="not divisible by 16"):
        RR.independent_rmsnorm(_node(), {"x": x[0, :1000], "w": w[:1000]})
    with pytest.raises(RR.RmsUnresolved, match="not in the measured family"):
        RR.independent_rmsnorm(_node(), {"x": x[0, :1280], "w": w[:1280]})
    with pytest.raises(RR.RmsUnresolved, match="sm_89"):
        RR.independent_rmsnorm(_node(arch="sm_80"), {"x": x[0], "w": w})
    with pytest.raises(RR.RmsRelationError, match="without const 'eps'"):
        RR.independent_rmsnorm({"op_type": "rmsnorm", "extra": {"runtime_op": "verity.bi_rms_norm"}}, {"x": x[0], "w": w})
    with pytest.raises(RR.RmsRelationError, match="words but w has"):
        RR.independent_rmsnorm(_node(), {"x": x[0], "w": w[:-16]})
