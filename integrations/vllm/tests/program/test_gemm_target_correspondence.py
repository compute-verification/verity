"""The GEMM correspondence driver's CPU half: the linear shapes it derives from HF configs, its sampling, and `check` on captures
made by the registered step itself (exact), with one flipped word, and under the wrong target's step (refused)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity_vllm.config import ROOT
from verity_vllm.pipeline import cli
from verity.ml.definitions.numerics import cpu_model as CM
from verity_vllm.program import gemm_target_correspondence as G

CONFIGS = ROOT / "verity_vllm" / "engine" / "profiles" / "hf_configs"


def _config(name: str) -> dict:
    return json.loads((CONFIGS / f"{name}.config.json").read_text())


def test_linear_shapes_of_dense_moe_and_sharded_configs():
    assert G.linear_shapes(_config("B0")) == {"qkv_proj": (576, 960), "o_proj": (576, 576), "gate_up_proj": (576, 3072),
                                             "down_proj": (1536, 576), "lm_head": (576, 49152)}
    assert G.linear_shapes(_config("LLAMA32_1B")) == {"qkv_proj": (2048, 3072), "o_proj": (2048, 2048), "gate_up_proj": (2048, 16384),
                                                     "down_proj": (8192, 2048), "lm_head": (2048, 128256)}
    assert G.linear_shapes(_config("GEMMA2_2B"))["qkv_proj"] == (2304, 4096) and G.linear_shapes(_config("GEMMA2_2B"))["o_proj"] == (2048, 2304)
    moe = G.linear_shapes(_config("QWEN3_30B_A3B"))
    assert moe == {"qkv_proj": (2048, 5120), "o_proj": (4096, 2048), "router": (2048, 128), "lm_head": (2048, 151936)}
    assert G.linear_shapes(_config("QWEN3_30B_A3B"), tp=2) == {"qkv_proj": (2048, 2560), "o_proj": (2048, 2048), "router": (2048, 128),
                                                              "lm_head": (2048, 75968)}
    assert G.linear_shapes(_config("OLMOE"), tp=2) == {"qkv_proj": (2048, 3072), "o_proj": (1024, 2048), "router": (2048, 64),
                                                      "lm_head": (2048, 25152)}
    assert G.linear_shapes(_config("PYTHIA160M")) == {"qkv_proj": (768, 2304), "o_proj": (768, 768), "up_proj": (768, 3072),
                                                     "down_proj": (3072, 768), "lm_head": (768, 50304)}
    with pytest.raises(ValueError, match="tp=3"):
        G.linear_shapes(_config("LLAMA32_1B"), tp=3)


def test_sample_indices_keep_both_tails():
    assert list(G.sample_indices(7, 128, 32)) == list(range(7))
    s = G.sample_indices(151936, 4096, 256)
    assert len(s) <= 4096 and np.all(np.diff(s) > 0) and s[0] == 0 and s[-1] == 151935
    assert set(range(256)) <= set(s) and set(range(151936 - 256, 151936)) <= set(s)
    assert G.derived_seed(13, "a", 1) == G.derived_seed(13, "a", 1) != G.derived_seed(13, "a", 2)


def _bf16(rng, shape, scale=1.0) -> np.ndarray:
    f = (rng.standard_normal(shape) * scale).astype(np.float32)
    return (f.view(np.uint32) >> 16).astype(np.uint16)


def _bits(rng, shape, lo, hi) -> np.ndarray:
    s = rng.integers(0, 2, shape, dtype=np.uint16)
    e = rng.integers(lo, hi + 1, shape).astype(np.uint16)
    m = rng.integers(0, 128, shape, dtype=np.uint16)
    return (s << 15) | (e << 7) | m


def _capture(tmp_path, cc, *, bias_mode=None, flip=False, M=6, K=64, N=40, x=None, w=None):
    """A capture directory whose GPU words are the Hopper k16 chain's (cpu_model), as a correct sm_120 device would write them."""
    rng = np.random.default_rng(7)
    if x is None:
        x, w = _bf16(rng, (M, K)), _bf16(rng, (N, K), 0.02)
    (M, K), N = x.shape, w.shape[0]
    b = _bf16(rng, (N,), 0.02) if bias_mode is not None else None
    _, hopper = G.step_model("HopperBF16WgmmaDot16")
    gpu, _ = CM.gemm_model(hopper, x, w, b, CM.GemmModelSpec(hopper.name, bias_mode=bias_mode or 0), want_f32=False)
    if flip:
        gpu[2, 3] ^= 1
    np.savez(tmp_path / "w_0.npz", w=w, cols=np.arange(N), **({"b": b} if b is not None else {}))
    np.savez(tmp_path / "case_0.npz", x=x, gpu=gpu, rows=np.arange(M))
    case = {"i": 0, "config": "SYNTH", "tp": 1, "shape": "o_proj", "K": K, "N": N, "family": "randn", "M": M, "w_file": "w_0.npz",
            "bias": b is not None, "kernels": ["synthetic"], "gpu_deterministic_rerun": True, "rows_evaluated": M, "cols_evaluated": N,
            "coordinates": M * N}
    (tmp_path / "cases.json").write_text(json.dumps({"schema": G.SCHEMA, "target": {"compute_capability": list(cc), "tuned_gemm_arch_family": None},
                                                     "configs": ["SYNTH"], "tp": "1", "ms": str(M), "seed": 7, "cases": [case]}))
    return G.CheckOptions(case_dir=str(tmp_path), out=str(tmp_path / "report.json"), crosscheck=64)


def test_check_is_exact_on_the_registered_step_and_cross_checks_it(tmp_path):
    assert G.check(_capture(tmp_path, (12, 0))) == 0
    r = json.loads((tmp_path / "report.json").read_text())
    assert r["all_exact"] and r["crosscheck_all_exact"] and r["crosscheck_attempted"] == 64
    assert r["gemm_target"]["key"] == "blackwell_consumer" and r["definition"]["step"]["groups"] == [16]
    assert r["definition"]["kind"] == "Gemm_v2{K,N,DOT=HopperBF16WgmmaDot16_v1}" and r["kernels"][0]["name"] == "synthetic"


def test_check_refuses_one_flipped_word(tmp_path):
    assert G.check(_capture(tmp_path, (12, 0), flip=True)) == 1
    r = json.loads((tmp_path / "report.json").read_text())
    assert not r["all_exact"] and r["results"][0]["first_mismatch"]["mismatching_coordinates"] == 1
    assert (r["results"][0]["first_mismatch"]["row"], r["results"][0]["first_mismatch"]["col"]) == (2, 3)


def test_hopper_words_are_not_the_ampere_chain(tmp_path):
    """The Hopper step's words checked as an A100 capture (cc 8.0, the Ampere step) are refused.  Rounding to bf16 hides nearly every
    difference between the two chains; these wide-exponent operands keep one coordinate apart."""
    rng = np.random.default_rng(44)
    x, w = _bits(rng, (8, 64), 100, 150), _bits(rng, (32, 64), 100, 150)
    assert G.check(_capture(tmp_path, (12, 0), x=x, w=w)) == 0
    assert G.check(_capture(tmp_path, (8, 0), x=x, w=w)) == 1
    r = json.loads((tmp_path / "report.json").read_text())
    assert r["gemm_target"]["key"] == "ampere" and r["results"][0]["first_mismatch"]["mismatching_coordinates"] == 1


def test_a_bias_case_reports_the_epilogue_that_holds(tmp_path):
    assert G.check(_capture(tmp_path, (12, 0), bias_mode=G.BIAS_EPILOGUES["fp32_before_rounding"])) == 0
    r = json.loads((tmp_path / "report.json").read_text())
    assert r["results"][0]["epilogue"] == "fp32_before_rounding" and r["bias_epilogues"]["fp32_before_rounding"] == 1


def test_the_cli_parses_both_phases_and_requires_a_seed():
    opts = cli.parse("gemm-correspondence", ["check", "--dir", "d", "--out", "o"])
    assert isinstance(opts, G.CheckOptions) and opts.crosscheck == 256
    cap = cli.parse("gemm-correspondence", ["capture", "--configs", "a.json", "--seed", "13", "--dir", "d"])
    assert cap.cmd == "capture" and cap.tp == "1" and cap.max_cols == 4096
    with pytest.raises(SystemExit):
        cli.parse("gemm-correspondence", ["capture", "--configs", "a.json", "--dir", "d"])
