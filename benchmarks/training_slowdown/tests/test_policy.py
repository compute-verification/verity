"""training_slowdown.policy: a start's reference shapes and the grid over them, the console's policy as one session of the
reference model, and the stated shapes beside the lf model's."""
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch
from conftest import SHAPE

from training_slowdown import flops, policy
from training_slowdown import formalism as F
from training_slowdown.gates import per_mac

LF_PATH = Path(__file__).resolve().parents[3] / "adapters" / "vllm" / "tests" / "program" / "train_step_lf.py"


def _lf():
    spec = importlib.util.spec_from_file_location("training_slowdown_tests_lf", LF_PATH)
    mod = importlib.util.module_from_spec(spec)
    # registered before it runs: dataclasses resolve the module's string annotations through sys.modules
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


LF = _lf()


def test_the_grid_spans_the_reference_shapes_in_decades():
    s = policy.shapes(SHAPE)
    w_fwd = SHAPE["forward_macs_per_token"] * per_mac("gemm/bf16")
    assert s["inference"]["c"] == w_fwd and s["inference"]["k"] == 16 + 1023 * 2 * 6 * 512 * 16
    ref = s["reference"]
    assert (ref["c"], ref["k"]) == (4096 * w_fwd, 80_000)
    serving = F.Measured((F.Part(1024 * w_fwd, 1024, 1024 * w_fwd, 0.0),), 16, 512 * 16, F.even(0.0), F.even(0.0), 0.0,
                         0.0)
    # honest serving passes Fit at the reference policy
    assert F.at(serving, ref["c"], ref["k"])["parts"][0]["fit"]["compute"]
    assert s["server"]["c"] == pytest.approx(3 * SHAPE["forward_macs_per_token"] * SHAPE["tokens"] * per_mac("gemm/bf16"))
    assert s["server"]["k"] == SHAPE["parameters"] * 80
    g = policy.grid(SHAPE)
    cs, ks = sorted({c for c, _ in g}), sorted({k for _, k in g})
    assert cs[0] <= w_fwd and cs[-1] >= s["server"]["c"] and ks[0] <= 16384 and ks[-1] >= s["server"]["k"]
    assert all(math.log10(c).is_integer() for c in cs) and len(g) == len(cs) * len(ks)
    assert policy.key(1e14, 1e4) == "c1e14/k1e4"
    assert policy.c0(SHAPE | {"c0_gates": 5.0}) == (5.0, "counted")


def test_the_cells_are_the_grid_and_the_reference_shapes_at_their_exact_c_and_k():
    cells, s = policy.cells(SHAPE), policy.shapes(SHAPE)
    assert {m: cells[m] for m in policy.MARKS} == {m: (v["c"], v["k"]) for m, v in s.items()}
    assert {v for k, v in cells.items() if k not in policy.MARKS} == set(policy.grid(SHAPE))


def test_the_consoles_c_is_one_full_session_of_the_reference_model():
    """The console's block is one session of the 45M reference, 4,096 tokens in and 1,024 out, attention and the LM
    head included, every MAC at the bf16 GEMM price."""
    s = policy.session(45_029_888, 4096, 1024)
    assert s["shape"] == {"d_model": 512, "n_layer": 6, "n_head": 8, "d_ff": 1408, "vocab": 50_304}
    assert s["macs"] == {"matrices": 5120 * 6 * (4 * 512 ** 2 + 3 * 512 * 1408), "lm_head": 5120 * 512 * 50_304,
                         "attention": 2 * 6 * 512 * (4096 ** 2 + 1024 * 4096 + 1024 * 1025 // 2)}
    assert s["gates"] == 362_592_337_920 * 1480.125 == policy.CONSOLE[0] == policy.session_gates(45_029_888, 4096, 1024)
    assert s["k_bits"] == policy.CONSOLE[1] == 65_536
    # attention over the whole prompt at each prompt token, as training_slowdown.flops counts the causal SDPA
    assert policy.session(45_029_888, 8, 0)["attended"] == 64 and policy.session(45_029_888, 8, 2)["attended"] == 64 + 19
    with pytest.raises(ValueError, match="no stated shape"):
        policy.session(7e9, 4096, 1024)
    with pytest.raises(ValueError, match="parameters, not"):
        policy.session(45_029_888, 4096, 1024, policy.SHAPES[35_658_240])


def test_the_stated_shapes_are_the_lf_models_and_its_counted_forward_is_the_sessions_weight_products():
    """Every stated shape has its parameters and heads 64 wide, and the 45M reference is the lf model
    (`adapters/vllm/tests/program/train_step_lf.py`) at its s size over the GPT-2 vocabulary padded to 50,304:
    the model's parameters are `parameters`' count, and a token's counted forward MACs at a context are the session's
    matrices, LM head and attention there, its elementwise ops under 0.1% of them."""
    for params, sh in policy.SHAPES.items():
        assert policy.parameters(sh) == params and sh["d_model"] // sh["n_head"] == 64, params
    size, ctx = LF.SIZES["s"], 16
    assert policy.SHAPES[policy.REFERENCE_PARAMS] == {"d_model": size.d_model, "n_layer": size.n_layer,
                                                      "n_head": size.n_head, "d_ff": size.d_ff, "vocab": LF.VOCAB}
    torch.manual_seed(0)
    model = LF.LM(size, ctx=ctx)
    assert sum(p.numel() for p in model.parameters()) == policy.REFERENCE_PARAMS
    forward = flops.forward_macs_per_token(model, ctx, LF.VOCAB)
    macs = policy.session(policy.REFERENCE_PARAMS, ctx, 0)["macs"]
    assert 0 < forward - sum(macs.values()) / ctx < 1e-3 * forward
