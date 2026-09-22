"""CPU tests for W4c's compiled/eager divergence diagnosis (CONTRACT §16.6) against the committed
reproducer bundle ``vllm-poc/bundles/divergence-B0-784c37e1`` and the layer-0 fixtures under
``fixtures/B0-divergence-20260907T1604Z``. numpy only, ~1 MB of data, no GPU.

Pins down:
* the first divergent canonical value (layer 0 input_layernorm, position 194, coordinate 458,
  eager 0xbc55 vs compiled 0xbc54) straight from the captured words;
* that the single-rounding row model with the IEEE inverse-RMS scalar reproduces the compiled
  kernel on all 512 rows and the eager kernel on 511 rows, the 512th needing s0 + 1 ulp
  (the divergence is the per-row scalar, not the epilogue);
* the compiled fused_add_rms_norm normalises the UNROUNDED fp32 residual sum while the eager
  kernel normalises the bf16-rounded sum (each model fits its own kernel, none fits the other);
* silu_and_mul: one rounding (compiled) vs two roundings (eager), 786432/786432 elements each.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verity_vllm_numerics import inductor_models as M  # noqa: E402

BUNDLE = ROOT / "bundles" / "divergence-B0-784c37e1"
FIXTURES = ROOT / "fixtures" / "B0-divergence-20260907T1604Z"
EPS = 1e-5

pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="reproducer bundle not present")


def _bundle():
    ids = np.load(BUNDLE / "input_ids.npy")
    table = np.load(BUNDLE / "embed_rows.npy")
    return table[ids], np.load(BUNDLE / "norm_weight.npy"), np.load(BUNDLE / "eager_out.npy"), np.load(BUNDLE / "compiled_out.npy")


def test_bundle_integrity():
    sums = dict(line.split()[::-1] for line in (BUNDLE / "sha256sums.txt").read_text().splitlines() if line.strip())
    import hashlib

    for name, digest in sums.items():
        assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == digest, name
    meta = json.loads((BUNDLE / "meta.json").read_text())
    assert meta["request_id"].startswith("784c37e1")
    assert meta["first_divergence"]["first_diff"]["position"] == 194


def test_first_divergent_value():
    _, _, eager, compiled = _bundle()
    fd = M.first_diff(eager, compiled)
    assert fd == {
        "n_diff": 1,
        "position": 194,
        "coordinate": 458,
        "expected_bits": "0xbc55",
        "actual_bits": "0xbc54",
        "expected_value": -0.01300048828125,
        "actual_value": -0.012939453125,
    }


def test_rms_norm_scalar_is_the_divergence():
    x, w, eager, compiled = _bundle()
    xf = M.bf16_to_f32(x)
    k_compiled = M.fit_ulp_offsets(compiled, xf, w, EPS)
    k_eager = M.fit_ulp_offsets(eager, xf, w, EPS)
    assert np.all(k_compiled == 0), "Inductor's libdevice.rsqrt is the IEEE scalar on every row"
    assert int(np.sum(k_eager == 0)) == 511 and k_eager[194] == 1, "vLLM's 1/sqrt is 1 ulp high on row 194"
    assert np.array_equal(M.rms_norm_compiled(x, w, EPS), compiled)
    assert np.array_equal(M.rms_norm_compiled(x, w, EPS, k=k_eager), eager)


@pytest.mark.skipif(not FIXTURES.exists(), reason="fixtures not present")
def test_fused_add_rms_norm_rounding_placement():
    d = FIXTURES / "layers.0.post_attention_layernorm_fused_add_rms_norm"
    w = np.load(d / "weight.npy")
    for mode, model_fits, model_fails in (("compiled", "unrounded", "rounded"), ("eager", "rounded", "unrounded")):
        x = np.load(d / mode / "in_x.npy")
        r = np.load(d / mode / "in_residual.npy")
        out = np.load(d / mode / "out_normed.npy")
        z_unrounded = (M.bf16_to_f32(x) + M.bf16_to_f32(r)).astype(np.float32)
        z_rounded = M.bf16_to_f32(M.f32_to_bf16(z_unrounded))
        fits = {"unrounded": M.fit_ulp_offsets(out, z_unrounded, w, EPS), "rounded": M.fit_ulp_offsets(out, z_rounded, w, EPS)}
        assert np.all(fits[model_fits] != 99) and np.all(np.abs(fits[model_fits]) <= 1), (mode, model_fits)
        assert np.all(fits[model_fails] == 99), (mode, model_fails)
    # the eager kernel's residual output is bf16(x + residual)
    x = np.load(d / "eager" / "in_x.npy")
    r = np.load(d / "eager" / "in_residual.npy")
    assert np.array_equal(M.fused_add_rms_norm_eager(x, r, w, EPS)[1], np.load(d / "eager" / "out_residual_out.npy"))


@pytest.mark.skipif(not FIXTURES.exists(), reason="fixtures not present")
def test_two_output_norm_keeps_residual_in_fp32():
    d = FIXTURES / "layers.1.input_layernorm_fused_add_rms_norm_two_outputs"
    w = np.load(d / "weight.npy")
    down = np.load(d / "compiled" / "in_down_proj_out.npy")
    o = np.load(d / "compiled" / "in_o_proj_out.npy")
    r = np.load(d / "compiled" / "in_residual_in.npy")
    normed, resid = M.fused_add_rms_norm_two_outputs_compiled(down, o, r, w, EPS)
    assert np.array_equal(resid, np.load(d / "compiled" / "out_residual_out.npy"))
    z = (M.bf16_to_f32(down) + (M.bf16_to_f32(o) + M.bf16_to_f32(r)).astype(np.float32)).astype(np.float32)
    k = M.fit_ulp_offsets(np.load(d / "compiled" / "out_normed.npy"), z, w, EPS)
    assert np.all(np.abs(k) <= 1)
    # and NOT the eager chain bf16(down + bf16(o + r))
    z_eager = (M.bf16_to_f32(down) + M.bf16_to_f32(M.f32_to_bf16((M.bf16_to_f32(o) + M.bf16_to_f32(r)).astype(np.float32)))).astype(np.float32)
    assert not np.array_equal(M.f32_to_bf16(z_eager), resid)


@pytest.mark.skipif(not FIXTURES.exists(), reason="fixtures not present")
def test_silu_mul_single_vs_double_rounding():
    d = FIXTURES / "layers.0.mlp.act_fn_silu_and_mul"
    for mode, model in (("compiled", M.silu_mul_compiled), ("eager", M.silu_mul_eager)):
        gu = np.load(d / mode / "in_gate_up.npy")
        out = np.load(d / mode / "out_out.npy")
        assert np.array_equal(model(gu), out), mode
    gu = np.load(d / "compiled" / "in_gate_up.npy")
    assert not np.array_equal(M.silu_mul_eager(gu), np.load(d / "compiled" / "out_out.npy"))
