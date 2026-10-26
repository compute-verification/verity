"""The Definitions whose program values are inputs (eps, the attention scale) compute what their constant-holding twins compute,
and hold none of those values as constants."""

from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.constants import program_data
from verity.ml.circuits.defs import bind
from verity.ml.circuits.program import Program
from verity.ml.definitions.registry import b1, hopper
from verity.ml.definitions.registry import fa2_check_inf as F2
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry import value_inputs as VI


def _bf16(rng, n: int, scale: float = 1.0) -> list[int]:
    return [int(P.bits_of(np.float32(v))) >> 16 for v in rng.standard_normal(n).astype(np.float32) * scale]


def _flat(x) -> list[int]:
    return [int(v) for v in np.asarray(x).reshape(-1)]


def _held(spec) -> set[str]:
    return {cid for held in program_data(Program(spec).root).values() for _g, cid in held}


def _const_id(bits: int) -> str:
    return f"Const32[0x{bits:08x}]_v1"


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_rmsnorm_triton_eps_input(eps):
    rng = np.random.default_rng(1)
    N, eps_bits = 64, P.bits_of(np.float32(eps))
    old, new = bind(b1.RMSNormTriton, N=N, EPS=eps), bind(VI.RMSNormTritonV3, N=N)
    assert new.id == "RMSNormTriton_v3{N=64}"
    for _ in range(3):
        x, w = _bf16(rng, N), _bf16(rng, N)
        assert _flat(evaluate(new, x, w, eps_bits)) == _flat(evaluate(old, x, w))
    assert _const_id(eps_bits) in _held(old)
    assert _const_id(eps_bits) not in _held(new)


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_rmsnorm_fused_cuda_eps_input(eps):
    rng = np.random.default_rng(2)
    N, eps_bits = 64, P.bits_of(np.float32(eps))
    old, new = bind(b1.RMSNormFusedCuda, N=N, EPS=eps), bind(VI.RMSNormFusedCudaV4, N=N)
    assert new.id == "RMSNormFusedCuda_v4{N=64}"
    for _ in range(3):
        x, res, w = _bf16(rng, N), _bf16(rng, N), _bf16(rng, N)
        assert _flat(evaluate(new, x, res, w, eps_bits)) == _flat(evaluate(old, x, res, w))
    assert _const_id(eps_bits) in _held(old)
    assert _const_id(eps_bits) not in _held(new)


def test_attention_scale_input():
    """Two key blocks (T = BN + 4): the second block rescales, so every use of the scale is exercised."""
    rng = np.random.default_rng(3)
    T, NH, KVH, D, BN = 20, 2, 1, 16, 16
    scale = b1.scale_log2_bits(D)
    old = bind(b1.AttentionV3, T=T, NH=NH, KVH=KVH, D=D, BN=BN)
    new = bind(VI.AttentionV10, T=T, NH=NH, KVH=KVH, D=D, BN=BN)
    for _ in range(2):
        q, kc, vc = _bf16(rng, NH * D, 0.5), _bf16(rng, T * KVH * D, 0.5), _bf16(rng, T * KVH * D)
        assert _flat(evaluate(new, q, kc, vc, scale)) == _flat(evaluate(old, q, kc, vc))
    assert _const_id(scale) in _held(old)
    assert _const_id(scale) not in _held(new)


@pytest.mark.parametrize("masked_from", [0, 1])
def test_fa2_check_inf_attention_scale_input(masked_from):
    """The FA2 per-iteration Check_inf chain (`Attention_v5`, the RTX PRO 6000's) over two key blocks, block 0 guarded or not."""
    rng = np.random.default_rng(4)
    T, NH, KVH, D, BN = 20, 2, 1, 16, 16
    scale = b1.scale_log2_bits(D)
    st = dict(T=T, NH=NH, KVH=KVH, D=D, BN=BN, DOT=hopper.HopperBF16WgmmaDot16, INV=P.Fa2InvSum, MASKED_FROM=masked_from)
    old, new = bind(F2.AttentionV5, **st), bind(F2.AttentionV11, **st)
    assert new.id.startswith("Attention_v11{")
    for _ in range(2):
        q, kc, vc = _bf16(rng, NH * D, 0.5), _bf16(rng, T * KVH * D, 0.5), _bf16(rng, T * KVH * D)
        assert _flat(evaluate(new, q, kc, vc, scale)) == _flat(evaluate(old, q, kc, vc))
    assert _const_id(scale) in _held(old)
    assert _const_id(scale) not in _held(new)


@pytest.mark.slow
def test_derive_registered_values_evaluates_like_the_constants_program():
    """A tiny greedy serve derived with `registered_values`: the norms' eps and the attention scale are the fields of the registered
    `values` parameter, held as no constant, and the Program computes what the constants derivation's flat Program computes."""
    import torch

    from verity_vllm.program.frontend import derive_examples as ex
    from verity_vllm.program.frontend.derive import derive_report
    from verity.ml.definitions.numerics.serve3_reference import materialize

    C = {**b1.B0_CONFIG, "H": 64, "NH": 4, "KVH": 2, "D": 16, "I": 128, "QKV": 128, "GU": 256, "NL": 1, "V": 32, "MAX_POS": 64}
    with torch.device("meta"):
        m = ex.GreedyServe(C, 1)
    inputs = (torch.empty(3, dtype=torch.int32, device="meta"),)
    const = derive_report(m, inputs, mode="profile", profile="b1-eager", config=C, name="Serve")
    reg = derive_report(m, inputs, mode="profile", profile="b1-eager", config=C, name="Serve", registered_values=True)
    assert const.ok and reg.ok, reg.unsupported
    eps, scale = P.bits_of(np.float32(C["EPS"])), b1.scale_log2_bits(C["D"])
    fields = reg.provenance["supplied"]["registration"]["fields"]
    assert [(f["name"], f["width"], int(f["bits"], 16)) for f in fields] == [("eps", 32, eps), ("scale_log2_d16", 32, scale)]
    prog, twin = reg.program, Program(const.flat_definition)
    assert [n for n, _ in prog.fn.body.params] == ["prompt", "values", "weights"]
    assert prog.bindings == ("run", "registration", "run")
    assert {n.fn.base_id for n in reg.flat_definition.body.nodes} >= {"RMSNormTriton_v3", "RMSNormFusedCuda_v4", "Attention_v10"}
    assert not {_const_id(eps), _const_id(scale)} & _held(prog.fn)
    rng = np.random.default_rng(5)
    prompt = [int(v) for v in rng.integers(0, C["V"], 3)]
    w = _bf16(rng, twin.input_coll(1).type.leaves, 0.05)
    a, b = materialize(twin, [prompt, w]), materialize(prog, [prompt, [eps, scale], w])
    assert [b[g] for g in prog.circuit.root_outputs()] == [a[g] for g in twin.circuit.root_outputs()]


def test_target_profile_registered_values_is_declared_true_or_absent():
    from verity_vllm.program.frontend.target_profile import TargetProfile

    assert "registered_values" not in TargetProfile().to_json()
    assert TargetProfile(registered_values=True).to_json()["registered_values"] is True
    with pytest.raises(ValueError, match="None"):
        TargetProfile(registered_values=False)
    both = TargetProfile(registered_values=True, ir="boolean").to_json()
    assert both["registered_values"] is True and both["ir"] == "boolean"


@pytest.mark.slow
def test_derive_registered_values_on_bits_reads_the_values_through_boolean_definitions():
    """The tiny registered-values serve on bits: every Call has a Boolean version (the norms with eps an input, `Attention_v14`),
    the lift is σ of the word function, and the `values` parameter is the fields' bits."""
    import torch

    from verity.ml.circuits import boolean as BL
    from verity.ml.definitions import boolean as BB
    from verity_vllm.program.frontend import derive_examples as ex
    from verity_vllm.program.frontend.derive import derive_report

    C = {**b1.B0_CONFIG, "H": 64, "NH": 4, "KVH": 2, "D": 16, "I": 128, "QKV": 128, "GU": 256, "NL": 1, "V": 32, "MAX_POS": 64}
    with torch.device("meta"):
        m = ex.GreedyServe(C, 1)
    inputs = (torch.empty(3, dtype=torch.int32, device="meta"),)
    reg = derive_report(m, inputs, mode="profile", profile="b1-eager", config=C, name="Serve", registered_values=True)
    assert reg.ok, reg.unsupported
    assert BB.dry_run(reg.program)["gaps"] == []
    bits = BB.lift(reg.program.fn)
    assert BB.sigma_check(reg.program.fn, bits)["ok"]
    called = {n.fn.base_id for n in bits.body.nodes}
    assert {"RMSNormTriton_v4", "RMSNormFusedCuda_v5", "Attention_v14"} <= called
    assert all(BL.is_boolean(n.fn) for n in bits.body.nodes if n.fn.base_id in {"RMSNormTriton_v4", "RMSNormFusedCuda_v5",
                                                                                 "Attention_v14"})
    (_, values), = [p for p in bits.signature()[0] if p[0] == "values"]
    assert values == BL.bit_type(dict(reg.program.fn.signature()[0])["values"])
