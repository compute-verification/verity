"""The remaining chains with their program values as inputs: `Attention_v12` / `v13` (the scale; the target-keyed chain without FA2's
per-iteration Check_inf, and FA3's), `AttentionSoftcap_v4` / `v5` (the softcap's two words) and the elementwise scalars
(`Bf16DivScalar_v3`, `Bf16MulScalar_v3`, `AddScalarF32_v3`, `AddScalarBf16_v3`), and the reference and aten norms' epsilon
(`RMSNormRef_v2`, `LayerNormRef_v2`, `LayerNormAten_v2`). Each computes what its constant-holding twin computes and holds none of its
values as a constant."""

from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.codec import encode_program, program_digest
from verity.ml.circuits.constants import program_data
from verity.ml.circuits.defs import bind
from verity.ml.circuits.program import Program
from verity_vllm.program.registry import b1, dense as DN, fa2_softcap as F2S, fa3_check_inf as F3, hopper
from verity_vllm.program.registry import prims as P, ref_prims as RP
from verity_vllm.program.registry import value_inputs as VI
from verity_vllm.program.registry import layer_norm as LN

HOP = dict(DOT=hopper.HopperBF16WgmmaDot16)


def _bf16(rng, n: int, scale: float = 1.0) -> list[int]:
    return [int(P.bits_of(np.float32(v))) >> 16 for v in rng.standard_normal(n).astype(np.float32) * scale]


def _flat(x) -> list[int]:
    return [int(v) for v in np.asarray(x).reshape(-1)]


def _held(spec) -> set[str]:
    return {cid for held in program_data(Program(spec).root).values() for _g, cid in held}


def _const_id(bits: int) -> str:
    return f"Const32[0x{bits:08x}]_v1"


def _twins(old, new, values: tuple[int, ...], T: int, NH: int, KVH: int, D: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(2):
        q, kc, vc = _bf16(rng, NH * D, 0.5), _bf16(rng, T * KVH * D, 0.5), _bf16(rng, T * KVH * D)
        assert _flat(evaluate(new, q, kc, vc, *values)) == _flat(evaluate(old, q, kc, vc))
    held_old, held_new = _held(old), _held(new)
    for bits in values:
        assert _const_id(bits) in held_old and _const_id(bits) not in held_new


def test_attention_v12_scale_input():
    """`Attention_v2` (a target's k16 step, FA2 without per-iteration Check_inf) over two key blocks (T = BN + 4)."""
    T, NH, KVH, D, BN = 20, 2, 1, 16, 16
    st = dict(T=T, NH=NH, KVH=KVH, D=D, BN=BN, INV=P.Fa2InvSum, **HOP)
    old, new = bind(b1.AttentionV2, **st), bind(VI.AttentionV12, **st)
    assert new.id.startswith("Attention_v12{")
    _twins(old, new, (b1._scale_log2_bits(D),), T, NH, KVH, D, 11)


@pytest.mark.parametrize("masked_from", [0, 1])
def test_attention_v13_scale_input(masked_from):
    """`Attention_v4` (FA3's per-iteration Check_inf) over two key blocks, block 0 guarded or not."""
    T, NH, KVH, D, BN = 20, 2, 1, 16, 16
    st = dict(T=T, NH=NH, KVH=KVH, D=D, BN=BN, INV=hopper.Fa3InvSum, MASKED_FROM=masked_from, **HOP)
    old, new = bind(F3.AttentionV4, **st), bind(VI.AttentionV13, **st)
    assert new.id.startswith("Attention_v13{")
    _twins(old, new, (b1._scale_log2_bits(D),), T, NH, KVH, D, 12 + masked_from)


@pytest.mark.parametrize("cap", [30.0, 50.0])
def test_attention_softcap_v4_inputs(cap):
    """`AttentionSoftcap_v1` (the accepted target) over two key blocks; `CAP` is no static of the twin."""
    T, NH, KVH, D, BN = 20, 2, 1, 16, 16
    old = bind(DN.AttentionSoftcap, T=T, NH=NH, KVH=KVH, D=D, BN=BN, CAP=cap)
    new = bind(VI.AttentionSoftcapV4, T=T, NH=NH, KVH=KVH, D=D, BN=BN)
    assert new.id == "AttentionSoftcap_v4{T=20,NH=2,KVH=1,D=16,BN=16}"
    _twins(old, new, (DN.softcap_mul_bits(D, cap), DN.softcap_scale_log2_bits(cap)), T, NH, KVH, D, 14)


@pytest.mark.parametrize("masked_from", [0, 1])
def test_attention_softcap_v5_inputs(masked_from):
    """`AttentionSoftcap_v2` (FA2's per-iteration Check_inf) over two key blocks, block 0 guarded or not."""
    T, NH, KVH, D, BN, cap = 20, 2, 1, 16, 16, 50.0
    st = dict(T=T, NH=NH, KVH=KVH, D=D, BN=BN, INV=P.Fa2InvSum, MASKED_FROM=masked_from, **HOP)
    old, new = bind(F2S.AttentionSoftcapV2, CAP=cap, **st), bind(VI.AttentionSoftcapV5, **st)
    assert new.id.startswith("AttentionSoftcap_v5{") and "CAP" not in new.id
    _twins(old, new, (DN.softcap_mul_bits(D, cap), DN.softcap_scale_log2_bits(cap)), T, NH, KVH, D, 16 + masked_from)


def _all_bf16() -> list[int]:
    return list(range(1 << 16))


@pytest.mark.parametrize("c", [30.0, 0.125])
def test_bf16_scalar_chain_inputs_over_every_word(c):
    """The logits softcap's division and multiplication on all 65,536 bf16 words."""
    N = 1 << 16
    x = _all_bf16()
    inv, cb = DN.reciprocal_bits(c), P.bits_of(np.float32(c))
    div_old, div_new = bind(DN.Bf16DivScalar, N=N, C=c), bind(VI.Bf16DivScalarV3, N=N)
    mul_old, mul_new = bind(DN.Bf16MulScalar, N=N, C=c), bind(VI.Bf16MulScalarV3, N=N)
    assert _flat(evaluate(div_new, x, inv)) == _flat(evaluate(div_old, x))
    assert _flat(evaluate(mul_new, x, cb)) == _flat(evaluate(mul_old, x))
    small_div, small_mul = bind(VI.Bf16DivScalarV3, N=4), bind(VI.Bf16MulScalarV3, N=4)
    assert _const_id(inv) in _held(bind(DN.Bf16DivScalar, N=4, C=c)) and not _held(small_div)
    assert _const_id(cb) in _held(bind(DN.Bf16MulScalar, N=4, C=c)) and not _held(small_mul)


@pytest.mark.parametrize("c", [1e-6, 1.0])
def test_add_scalar_inputs(c):
    rng = np.random.default_rng(18)
    N, cb = 64, P.bits_of(np.float32(c))
    x = [int(P.bits_of(v)) for v in rng.standard_normal(N).astype(np.float32)]
    f_old, f_new = bind(DN.AddScalarF32, N=N, C=c), bind(VI.AddScalarF32V3, N=N)
    w_old, w_new = bind(DN.AddScalarBf16, N=N, C=c), bind(VI.AddScalarBf16V3, N=N)
    w = _bf16(rng, N)
    assert _flat(evaluate(f_new, x, cb)) == _flat(evaluate(f_old, x))
    assert _flat(evaluate(w_new, w, cb)) == _flat(evaluate(w_old, w))
    assert _const_id(cb) in _held(f_old) and _const_id(cb) in _held(w_old)
    assert not _held(f_new) and not _held(w_new)


def _rows(rng, n: int, DT: str) -> list[int]:
    return _bf16(rng, n) if DT == "bf16" else [int(P.bits_of(v)) for v in rng.standard_normal(n).astype(np.float32)]


@pytest.mark.parametrize("DT", ["bf16", "f32"])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_reference_norm_eps_inputs(DT, eps):
    rng = np.random.default_rng(19)
    N, eb = 32, P.bits_of(np.float32(eps))
    rms_old, rms_new = bind(RP.RMSNormRef, N=N, EPS=eps, DT=DT), bind(VI.RMSNormRefV2, N=N, DT=DT)
    ln_old, ln_new = bind(RP.LayerNormRef, N=N, EPS=eps, DT=DT), bind(VI.LayerNormRefV2, N=N, DT=DT)
    assert rms_new.id == f"RMSNormRef_v2{{N=32,DT='{DT}'}}" and ln_new.id == f"LayerNormRef_v2{{N=32,DT='{DT}'}}"
    for _ in range(3):
        x, g, b = (_rows(rng, N, DT) for _ in range(3))
        assert _flat(evaluate(rms_new, x, g, eb)) == _flat(evaluate(rms_old, x, g))
        assert _flat(evaluate(ln_new, x, g, b, eb)) == _flat(evaluate(ln_old, x, g, b))
    assert _const_id(eb) in _held(rms_old) and _const_id(eb) in _held(ln_old)
    assert _const_id(eb) not in _held(rms_new) and _const_id(eb) not in _held(ln_new)


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_layer_norm_aten_eps_input(eps):
    rng = np.random.default_rng(20)
    N, eb = 16, P.bits_of(np.float32(eps))
    old, new = bind(LN.LayerNormAten, N=N, EPS=eps), bind(LN.LayerNormAtenV2, N=N)
    assert new.id == "LayerNormAten_v2{N=16}"
    for _ in range(3):
        x, w, b = _bf16(rng, N, 4.0), _bf16(rng, N), _bf16(rng, N)
        assert _flat(evaluate(new, x, w, b, eb)) == _flat(evaluate(old, x, w, b))
    assert _const_id(eb) in _held(old) and _const_id(eb) not in _held(new)


def test_the_constant_twins_keep_their_digests():
    """The constant bodies now take their values optionally; the Programs they build are the ones they built before."""
    st = dict(INV=P.Fa2InvSum, **HOP)
    got = {
        "Attention_v2": bind(b1.AttentionV2, T=17, NH=2, KVH=1, D=16, BN=16, **st),
        "Attention_v4": bind(F3.AttentionV4, T=33, NH=2, KVH=1, D=16, BN=16, **{**st, "INV": hopper.Fa3InvSum}, MASKED_FROM=1),
        "AttentionSoftcap_v1": bind(DN.AttentionSoftcap, T=17, NH=2, KVH=1, D=16, BN=16, CAP=50.0),
        "AttentionSoftcap_v2": bind(F2S.AttentionSoftcapV2, T=33, NH=2, KVH=1, D=16, BN=16, CAP=50.0, MASKED_FROM=1, **st),
        "Bf16DivScalar_v1": bind(DN.Bf16DivScalar, N=8, C=30.0),
        "Bf16MulScalar_v1": bind(DN.Bf16MulScalar, N=8, C=30.0),
        "AddScalarF32_v1": bind(DN.AddScalarF32, N=8, C=1e-6),
        "AddScalarBf16_v1": bind(DN.AddScalarBf16, N=8, C=1.0),
        "RMSNormRef_v1": bind(RP.RMSNormRef, N=8, EPS=1e-6, DT="bf16"),
        "LayerNormRef_v1": bind(RP.LayerNormRef, N=8, EPS=1e-5, DT="bf16"),
        "LayerNormAten_v1": bind(LN.LayerNormAten, N=8, EPS=1e-5),
    }
    assert {k: program_digest(encode_program(Program(v)))[:16] for k, v in got.items()} == {
        "Attention_v2": "62fa3bb28d03e5d8", "Attention_v4": "95413195ae00c692", "AttentionSoftcap_v1": "eb63193194c87cf0",
        "AttentionSoftcap_v2": "3e69667726a0a607", "Bf16DivScalar_v1": "25bace9a57773ab6", "Bf16MulScalar_v1": "efe2e181523aafd8",
        "AddScalarF32_v1": "5d68003bd74e3840", "AddScalarBf16_v1": "22017d4f8b23c248",
        "RMSNormRef_v1": "c1f320c64fef5c6b", "LayerNormRef_v1": "7f97dc6cdf51458d", "LayerNormAten_v1": "93931c7a49661fb0"}
