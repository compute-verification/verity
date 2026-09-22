"""CPU tests for W11's independent model of vLLM's CUDA ``vllm::rms_norm_kernel<bf16, VEC_SIZE=8>``
(``torch.ops._C.rms_norm``; ``verity_vllm_numerics.rms_triton_model`` with ``kernel="cuda"`` /
``rms_relation.independent_rmsnorm_cuda``) on the saved fixtures
``fixtures/W11C-40f0cebeb670-20260907T1900Z`` (outputs) + ``W11R-40f0cebeb670-20260907T1802Z`` (inputs).

No GPU: every expected output is a BF16 word the production kernel wrote on the RTX 4090 -- the Triton
fixture's natural Qwen2.5-1.5B rows (``layers.0.input_layernorm``) replayed through ``_C.rms_norm``, the
same §11 stress inputs, the same 36 one-bit mutations.  What is pinned down:

* the MUFU.RSQ table reconstructs from the shipped deltas to the hashed hardware dump; the bare instruction
  and the kernel's ``rsqrtf`` sequence (subnormal range scaling) behave as measured on special values;
* every natural call and every stress family reproduce the production words bit-for-bit;
* one-bit flips of an x / w word and eps changes move the model's output exactly where production's moved;
* the dataflow alternatives (Triton-style butterfly, sequential sum, IEEE rsqrt, div.full mean, fp64 sum,
  other output association, reversed warp aggregates) each mismatch production somewhere;
* the coverage gate: ``independent_rmsnorm_cuda`` predicts only for ``runtime_op == "_C.rms_norm"`` inside the
  measured family and raises ``RmsUnresolved`` (``unresolved: ...``) elsewhere.

Needs g++ (or $CXX) for ``cpp/rms_triton_model.cpp``; skips where it cannot be built.
Run: ``cd vllm-poc && python -m pytest tests/test_rmsnorm_cuda_model.py -q``
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

FIX = ROOT / "fixtures" / "W11C-40f0cebeb670-20260907T1900Z"
REF = ROOT / "fixtures" / "W11R-40f0cebeb670-20260907T1802Z"

from verity_vllm_numerics import rms_triton_model as RM  # noqa: E402
from verity_vllm_numerics import rms_relation as RR  # noqa: E402


@pytest.fixture(autouse=True)
def _rms_tables_env(monkeypatch):
    # Per-test, not import-time: the Triton test module points the same variable at the W11R tables
    # (no MUFU.RSQ), and collection order must not decide which tables the relation loads.
    monkeypatch.setenv("VERITY_RMS_TABLES", str(FIX / "tables"))

pytestmark = pytest.mark.skipif(not (FIX.exists() and REF.exists()), reason="fixture directories missing")


def _have_cpp() -> bool:
    try:
        RM.lib()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_cpp = pytest.mark.skipif(not _have_cpp(), reason="cpp/rms_triton_model.cpp could not be built (no g++)")

RSQ_RAW_SHA256 = "218fc04d271773269367dedf4cff9ff719081b1218c3d67d248d8d6126040246"  # hw dump, 2^24 x u32 (rms_probe rsq_table)


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
    x = _w(fix / c["x_file"]).reshape(c["rows"], c["n_cols"])  # x_file points into the W11R fixture
    out = _w(fix / c["out_file"]).reshape(c["rows"], c["n_cols"])
    w = _w(fix / idx["weights"][c["weight"]]["file"])
    return x, w, out


def _model(x, w, eps, tables, **kw):
    return RM.rms_norm_rows(x, w, eps, tables, kernel="cuda", **kw)


# ---- tables -----------------------------------------------------------------------------------------

def test_rsq_table_reconstructs_to_hardware_dump(tables):
    assert tables.rsq is not None and tables.rsq.shape == (2 * RM.TABLE_N,) and tables.rsq.dtype == np.uint32
    assert hashlib.sha256(tables.rsq.tobytes()).hexdigest() == RSQ_RAW_SHA256
    assert tables.rsq[0] == 0x3F800000  # rsqrt(1) = 1
    assert tables.rsq[RM.TABLE_N] == 0x3F3504F2  # rsqrt(2): one ulp below the correctly rounded 0x3F3504F3
    assert tables.rsq[-1] == 0x3F000000  # rsqrt(4 - 2^-22) rounds up to 0.5
    d = np.diff(tables.rsq.astype(np.int64))
    assert d.max() == 0 and d.min() == -3  # monotone non-increasing; 2 three-ulp and 21 two-ulp steps
    assert int((d == -3).sum()) == 2 and int((d == -2).sum()) == 21
    # within 2 ulp of the correctly rounded rsqrt: 8 entries 2 low, 2,344,391 1 low, 13,143,086 exact, 1,289,731 1 high
    m = np.arange(RM.TABLE_N, dtype=np.float64) / RM.TABLE_N
    cr = np.concatenate([1 / np.sqrt(1 + m), 1 / np.sqrt(2 * (1 + m))]).astype(np.float32).view(np.uint32).astype(np.int64)
    err = tables.rsq.astype(np.int64) - cr
    assert err.min() == -2 and err.max() == 1
    assert int((err == 0).sum()) == 13143086 and int((err == -2).sum()) == 8
    # the W11R tables are the same objects (sqrt / rcp): W11C/tables is a superset
    ref = RM.RmsTables.load(REF / "tables")
    assert (ref.sqrt == tables.sqrt).all() and (ref.rcp == tables.rcp).all()


@needs_cpp
def test_rsq_special_values(tables):
    x = np.array([4.0, 2.0, 1e-6, 0.0, -0.0, np.inf, -1.0, 1e-40, 0.25, 2.0 ** -126, 2.0 ** 100, np.nan], np.float32)
    y = RM.mufu_rsq(x, tables)  # the bare MUFU.RSQ (ftz)
    assert y[0] == 0.5 and y[3] == np.inf and y[4] == -np.inf and y[5] == 0.0 and np.isnan(y[6]) and np.isnan(y[11])
    assert y[7] == np.inf  # ftz: the subnormal input is flushed to 0
    assert y[8] == 2.0 and y[9] == 2.0 ** 63 and y[10] == 2.0 ** -50
    assert y[1].view(np.uint32) == 0x3F3504F2
    assert y[2] == np.float32(1000.0)
    z = RM.rsqrt_f32(x, tables)  # rsqrtf as compiled: range scaling keeps subnormals
    assert (z[:7].view(np.uint32) == y[:7].view(np.uint32)).all() and (z[8:].view(np.uint32) == y[8:].view(np.uint32)).all()
    exact = 1 / np.sqrt(np.float64(np.float32(1e-40)))  # the subnormal input itself is only ~1e-40 to 6 digits
    assert np.isfinite(z[7]) and abs(z[7] - exact) <= 4 * np.spacing(np.float32(exact))  # 2^24 scaling before, 2^12 after
    tiny = np.array([2.0 ** -140, 2.0 ** -149, 5e-39], np.float32)
    zt = RM.rsqrt_f32(tiny, tables)
    assert abs(zt[0] - 2.0 ** 70) <= 2 * np.spacing(np.float32(2.0 ** 70)) and np.isfinite(zt[1]) and np.isfinite(zt[2])


# ---- natural inputs -----------------------------------------------------------------------------------

@needs_cpp
def test_natural_calls_bit_exact(tables, natural):
    """Every natural call (3,146 rows x 1536 = 4,832,256 words) through _C.rms_norm: 0 mismatching words."""
    words = 0
    for c in natural["calls"]:
        x, w, hw = _call(FIX / "natural", natural, c)
        md = _model(x, w, c["eps"], tables)
        assert (md == hw).all(), f"natural seq {c['seq']} ({c['phase']}): {(md != hw).sum()} mismatching words"
        words += hw.size
    assert words == 4832256
    assert {c["n_cols"] for c in natural["calls"]} == {1536} and {c["eps"] for c in natural["calls"]} == {1e-6}


def test_natural_fixture_identity(natural, stress):
    assert natural["kind"] == "natural-inputs-replayed" and natural["kernel"] == "cuda" and natural["n_calls"] == 300
    assert natural["entry_point"].startswith("torch.ops._C.rms_norm(")
    assert sum(c["rows"] for c in natural["calls"]) == 3146
    assert stress["entry_point"].endswith("vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>")
    # the inputs are the W11R files: every x_file resolves into the sibling fixture and hashes as listed there
    sums = dict(line.split("  ", 1)[::-1] for line in (REF / "natural" / "SHA256SUMS").read_text().splitlines())
    c = natural["calls"][0]
    p = (FIX / "natural" / c["x_file"]).resolve()
    assert p.parent.parent.parent == REF.resolve()
    assert hashlib.sha256(p.read_bytes()).hexdigest() == sums[str(p.relative_to(REF.resolve() / "natural"))]


def test_cuda_and_triton_kernels_differ_on_the_same_inputs(natural):
    """Same rows, two production kernels: the outputs are NOT identical (so the two models are not interchangeable)."""
    tri = json.loads((REF / "natural" / "index.json").read_text())
    tri_calls = {c["seq"]: c for c in tri["calls"]}
    diff = 0
    for c in natural["calls"][:40]:
        hw_cuda = _w(FIX / "natural" / c["out_file"])
        hw_tri = _w(REF / "natural" / tri_calls[c["natural_seq"]]["out_file"])
        diff += int((hw_cuda != hw_tri).sum())
    assert diff > 0


# ---- stress / mutations -------------------------------------------------------------------------------

@needs_cpp
def test_stress_families_bit_exact(tables, stress):
    fams = {}
    for c in stress["calls"]:
        x, w, hw = _call(FIX / "stress", stress, c)
        md = _model(x, w, c["eps"], tables)
        assert (md == hw).all(), f"stress {c['family']}/{c['case']}: {(md != hw).sum()} mismatching words"
        fams[c["family"]] = fams.get(c["family"], 0) + hw.size
    assert set(fams) >= {"zeros", "absorption", "rounding_boundaries", "subnormals", "extremes", "near_uniform", "random_scale", "n_cols", "beyond_natural_range"}
    assert sum(fams.values()) == 3094656
    # the n_cols family exercises partial warps (16 -> 2 threads, 576 -> 72 threads) and 512-thread blocks (4096)
    assert {c["n_cols"] for c in stress["calls"]} == set(RM.MEASURED_N_COLS)


@needs_cpp
def test_non_finite_and_zero_eps_semantics(tables, stress):
    """inf / NaN operands, BF16 max, eps = 0 with fp32-subnormal squares, eps < 0 / inf: the model follows the hardware."""
    by_case = {c["case"]: c for c in stress["calls"]}
    for case in ("inf_in_x", "nan_in_x", "nan_in_w", "bf16_max_row", "fp32_subnormal_squares_eps0", "bf16_subnormal_rows_eps0", "negative_eps", "eps_inf"):
        c = by_case[case]
        x, w, hw = _call(FIX / "stress", stress, c)
        md = _model(x, w, c["eps"], tables)
        assert (md == hw).all(), case
    x, w, hw = _call(FIX / "stress", stress, by_case["nan_in_x"])
    assert (hw == 0x7FFF).all()  # NaN in the sum -> every output the canonical BF16 NaN
    # unlike the Triton kernel (sqrt.approx.ftz -> inf words), rsqrt.approx.f32's range scaling keeps a subnormal mean finite
    x, w, hw = _call(FIX / "stress", stress, by_case["fp32_subnormal_squares_eps0"])
    tri = json.loads((REF / "stress" / "index.json").read_text())
    tri_hw = _w(REF / "stress" / {c["case"]: c for c in tri["calls"]}["fp32_subnormal_squares_eps0"]["out_file"]).reshape(hw.shape)
    assert (tri_hw[np.nonzero(x)] & 0x7FFF).min() == 0x7F80 and ((hw[np.nonzero(x)] & 0x7FFF) < 0x7F80).all()


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
        md_mut = _model(x, w, eps, tables)
        md_base = _model(*_call(FIX / "stress", stress, base)[:2], base["eps"], tables)
        assert m["production_output_words_changed"] > 0
        assert (md_mut == hw_mut).all(), m
        assert (md_mut != md_base).sum() == (hw_mut != hw_base).sum() > 0
        n += 1
    assert n == 36


@needs_cpp
def test_dataflow_alternatives_mismatch(tables, stress, natural):
    """Every alternative dataflow breaks exactness somewhere on a small sample; only the compiled one is exact."""
    by_case = {c["case"]: c for c in stress["calls"]}
    sample = [by_case[k] for k in ("within_1ulp_of_tie", "exact_bf16_ties_pre_cast", "scale_2^-60", "constant_rows", "wide_dynamic_range_2^-12..2^12")]
    sample += [c for c in stress["calls"] if c["family"] == "beyond_natural_range"][:2]
    nat = [c for c in natural["calls"] if c["phase"] == "prefill"][:1]
    mism = {}
    for name, v in RM.CUDA_VARIANTS.items():
        n = 0
        for c in sample:
            x, w, hw = _call(FIX / "stress", stress, c)
            n += int((_model(x, w, c["eps"], tables, variant=v) != hw).sum())
        for c in nat:
            x, w, hw = _call(FIX / "natural", natural, c)
            n += int((_model(x, w, c["eps"], tables, variant=v) != hw).sum())
        mism[name] = n
    assert mism["model"] == 0
    for name in ("butterfly_warp_tree", "sequential_sum", "ieee_rsqrt", "div_full_mean", "fp64_sum", "out_xw_first", "warp_aggregates_reversed"):
        assert mism[name] > 0, name


@needs_cpp
def test_triton_model_is_not_the_cuda_kernel(tables, natural):
    """The Triton model applied to the CUDA kernel's rows mismatches (the two kernels round differently)."""
    c = [c for c in natural["calls"] if c["phase"] == "prefill"][0]
    x, w, hw = _call(FIX / "natural", natural, c)
    assert int((RM.rms_norm_rows(x, w, c["eps"], tables, kernel="triton") != hw).sum()) > 0


# ---- relation / coverage gate ----------------------------------------------------------------------------

def _node(op="_C.rms_norm", eps=1e-6, **extra):
    return {"op_type": "rmsnorm", "extra": {"runtime_op": op, "consts": {"eps": eps, **extra}}}


@needs_cpp
def test_relation_predicts_natural_row(tables, natural):
    c = natural["calls"][0]
    x, w, hw = _call(FIX / "natural", natural, c)
    out = RR.independent_rmsnorm_cuda(_node(), {"x": x[7], "w": w})
    assert out == hw[7].astype("<u2").tobytes()
    cov = RR.coverage(_node(), {"x": x[7], "w": w}, kernel="cuda")
    assert cov["covered"] and cov["n_cols"] == 1536 and cov["kernel"] == "cuda"
    assert "rmsnorm.cuda.bf16.v1" in RR.RELATION_IDS and RR.RELATIONS["rmsnorm.cuda.bf16.v1"] is RR.independent_rmsnorm_cuda


@needs_cpp
def test_gate_unresolved_outside_measured_family(tables, natural):
    c = natural["calls"][0]
    x, w, _ = _call(FIX / "natural", natural, c)
    with pytest.raises(RR.RmsUnresolved, match="unresolved: runtime_op"):
        RR.independent_rmsnorm_cuda(_node(op="verity.bi_rms_norm"), {"x": x[0], "w": w})
    with pytest.raises(RR.RmsUnresolved, match="unresolved: runtime_op"):
        RR.independent_rmsnorm(_node(op="_C.rms_norm"), {"x": x[0], "w": w})  # and the Triton relation declines _C.rms_norm
    with pytest.raises(RR.RmsUnresolved, match="not divisible by 8"):
        RR.independent_rmsnorm_cuda(_node(), {"x": x[0, :1500], "w": w[:1500]})
    with pytest.raises(RR.RmsUnresolved, match="not in the measured family"):
        RR.independent_rmsnorm_cuda(_node(), {"x": x[0, :1528], "w": w[:1528]})
    with pytest.raises(RR.RmsUnresolved, match="sm_89"):
        RR.independent_rmsnorm_cuda(_node(arch="sm_80"), {"x": x[0], "w": w})
    with pytest.raises(RR.RmsRelationError, match="without const 'eps'"):
        RR.independent_rmsnorm_cuda({"op_type": "rmsnorm", "extra": {"runtime_op": "_C.rms_norm"}}, {"x": x[0], "w": w})
