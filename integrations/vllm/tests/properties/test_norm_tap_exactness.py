"""Norm-tap exactness without a GPU: the sealed record's digest and verdict, the cases, the PTX arithmetic comparison and the source diff
(the GPU run is `tests/properties/norm_tap_exactness_gpu.py --so ... --out ...` on a pod)."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from tests.properties import norm_tap_exactness_gpu as G
from verity_vllm.properties import norm_tap_exactness as X
from verity_vllm.properties.record import REGISTRY, digest_of

TAP_LINES = ("    scale_ptr,\n", "    tl.store(scale_ptr + row_idx + tl.arange(0, 1), inv_rms)\n")
PTX = """
.visible .entry _rms_norm_kernel(
\t.param .u64 _rms_norm_kernel_param_0,
\t.param .u32 _rms_norm_kernel_param_3
)
{
\tld.param.u64 \t%rd1, [_rms_norm_kernel_param_0];
\tcvt.rn.f32.s32 \t%f1, %r1;
$L__BB0_1:
\tfma.rn.f32 \t%f2, %f3, %f3, %f4;
\tshfl.sync.bfly.b32\t%r5, %r4, 16, 31, -1;
\tadd.f32 \t%f5, %f2, %f6;
\t@%p1 sqrt.approx.ftz.f32 \t%f7, %f5;
\tmul.wide.s32 \t%rd4, %r2, 2;
{PAD}\tcvt.rn.bf16.f32 \t%rs1, %f8;
\t@%p2 st.global.b16 \t[%rd5], %rs1;
\tret;
}
"""


def _ptx(pad: str = "") -> str:
    return PTX.replace("{PAD}", pad)


def _cases():
    cuda = {"name": "cuda_n2048_r7", "kernel": "cuda", "out_eq_installed": True, "out_eq_untapped": True, "unwritten": 0, "scale_ne_twin": 0,
            "scale_ne_ir": 0, "ir_rows": 3}
    tri = {"name": "triton_n2048_r7", "kernel": "triton", "out_eq_installed": True, "direct_eq_vllm": True, "unwritten": 0, "scale_ne_twin": 0,
           "scale_ne_ir": 0, "ir_rows": 3, "ptx": {"arith_equal": True, "extra_global_stores": 1}}
    for c in (cuda, tri):
        c["ok"] = X.case_ok(c)
    return [cuda, tri]


def _record(**fields):
    src = {"outputs_equal": True, "outside_step_acquired": 0, "names_ok": True, "scales_equal": True, "restored": True, "errors": 0}
    src["ok"] = X.source_ok(src)
    tri = {"installed_sha256": "a" * 64, "pinned_sha256": "a" * 64, "diff": {"ok": True}}
    return X.seal({"schema": X.SCHEMA, "cases": _cases(), "triton": tri, "source": src, **fields})


def test_registered():
    assert REGISTRY["norm_tap_exactness"].schema == X.SCHEMA


def test_sealed_record_verifies_and_every_edit_is_named():
    doc = _record()
    assert doc["ok"] is True and X.verify_record(doc) == []
    edited = copy.deepcopy(doc)
    edited["cases"][0]["scale_ne_ir"] = 1
    stale = ["case cuda_n2048_r7: ok=True but its fields say False", "ok=True but the entries say False"]
    assert X.verify_record(edited) == ["digest does not match the record", *stale]
    assert X.verify_record({**edited, "digest": digest_of(edited)}) == stale
    assert X.seal(edited)["ok"] is False


@pytest.mark.parametrize("edit", [
    lambda d: d["cases"].pop(),                                             # one kernel missing
    lambda d: d["cases"][1]["ptx"].update(arith_equal=False),               # the copy's arithmetic changed
    lambda d: d["cases"][1]["ptx"].update(extra_global_stores=2),           # it stores more than the scale
    lambda d: d["cases"][0].update(out_eq_untapped=False),                  # the tap-off build differs
    lambda d: d["cases"][0].update(unwritten=1),                            # a row's scale not written
    lambda d: d["cases"][0].update(ir_rows=0),                              # nothing checked against the circuit
    lambda d: d["triton"].update(installed_sha256="b" * 64),                # the installed kernel is not the pinned one
    lambda d: d["triton"]["diff"].update(ok=False),                         # the copy changes more than the tap lines
    lambda d: d["source"].update(outside_step_acquired=1),                  # the source acquired outside a committed step
    lambda d: d["source"].update(restored=False),                           # a patched site left behind
    lambda d: d.pop("source"),
])
def test_every_clause_fails_the_record(edit):
    doc = copy.deepcopy(_record())
    edit(doc)
    for e in doc.get("cases", []):
        e["ok"] = X.case_ok(e)
    if doc.get("source"):
        doc["source"]["ok"] = X.source_ok(doc["source"])
    assert X.seal(doc)["ok"] is False


def test_verify_cli(tmp_path):
    path = tmp_path / X.RECORD
    path.write_text(json.dumps(_record()))
    assert G.main(["--verify", str(path)]) == 0
    doc = _record()
    doc["cases"][0]["unwritten"] = 3
    path.write_text(json.dumps(doc))
    assert G.main(["--verify", str(path)]) == 1


def test_cases_cover_both_kernels_every_width_the_specials_and_no_weight():
    cs = X.cases()
    assert {c["kernel"] for c in cs} == set(X.KERNELS) and len({c["name"] for c in cs}) == len(cs)
    for k in X.KERNELS:
        assert {c["N"] for c in cs if c["kernel"] == k and not c["special"]} == {n for n, _e in X.WIDTHS}
        assert any(c["special"] for c in cs if c["kernel"] == k) and any(not c["weight"] for c in cs if c["kernel"] == k)
    assert any(c["residual_special"] for c in cs if c["kernel"] == "cuda")
    assert all(c["rows"] == len(X.SPECIAL) for c in cs if c["special"])
    assert {c["name"] for c in X.cases(quick=True)} == set(X.QUICK) and set(X.QUICK) <= {c["name"] for c in cs}


def test_ptx_arithmetic_ignores_the_tap_store_and_names_a_change():
    base = _ptx()
    tapped = _ptx("\tmad.lo.s64 \t%rd9, %rd2, 4, %rd8;\n\t@%p3 st.global.b32 \t[%rd9], %f7;\n")
    assert X.arith_ops(base) == ["cvt.rn.f32.s32", "fma.rn.f32", "shfl.sync.bfly.b32", "add.f32", "sqrt.approx.ftz.f32", "cvt.rn.bf16.f32"]
    r = X.ptx_check(base, tapped)
    assert r["arith_equal"] and r["extra_global_stores"] == 1 and r["first_arith_difference"] is None
    changed = X.ptx_check(base, tapped.replace("fma.rn.f32", "fma.rz.f32"))
    assert not changed["arith_equal"] and changed["first_arith_difference"] == {"at": 1, "installed": "fma.rn.f32", "tapped": "fma.rz.f32"}


PINNED = '''@triton.jit
def k(
    input_ptr,
    output_ptr,
    n_cols,
):
    inv_rms = 1.0 / rms
    tl.store(output_ptr, inv_rms)
'''


def test_source_diff_accepts_only_the_tap_lines():
    tapped = PINNED.replace("    output_ptr,\n", "    output_ptr,\n" + TAP_LINES[0]).replace("    inv_rms = 1.0 / rms\n", "    inv_rms = 1.0 / rms\n" + TAP_LINES[1])
    assert X.source_diff(PINNED, tapped, TAP_LINES)["ok"]
    assert not X.source_diff(PINNED, tapped.replace("1.0 / rms", "1.0 / rms + 0.0"), TAP_LINES)["ok"]
    assert not X.source_diff(PINNED, tapped + "    tl.store(output_ptr, rms)\n", TAP_LINES)["ok"]
    assert not X.source_diff(PINNED, PINNED, TAP_LINES)["ok"]


@pytest.mark.parametrize("kernel", X.KERNELS)
def test_ir_scale_is_the_twins_scale(kernel):
    """The scale gate read off the evaluation trace (`ir_scale`) is the twin's `s` / `inv` on random rows (N = 64): the gate lookup is right."""
    from verity.ml.definitions.numerics import derived_rows as D
    try:
        D._rms_tables()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"MUFU tables unavailable: {e}")
    rng = np.random.default_rng(64)
    N, eps = 64, 1e-6
    bf16 = lambda s: ((rng.standard_normal(s) * 2).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)  # noqa: E731
    x, res, w = bf16((2, N)), bf16((2, N)), bf16(N)
    twin = X.twin_scales(kernel, x, res if kernel == "cuda" else None, w, eps)
    for r in range(2):
        assert X.ir_scale(kernel, N, eps, x[r], res[r] if kernel == "cuda" else None, w) == int(twin[r])
