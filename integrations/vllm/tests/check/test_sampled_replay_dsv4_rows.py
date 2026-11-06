"""DeepSeek-V4-Pro B1's row evaluators (`kernels.dsv4_rows`) as the sampled replay calls them.  The replay hands a row kernel the
statics `spec_statics` parses from the Program's spec text (`{"f64":...}` floats, no statics at all for `GatherI32x{V}_v1`), not
the Python values the kernel self-check binds, so each family is checked here on those spellings, at small statics, against its
Definition's word reference, every port in the Definition's return order."""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.evaluation.batch import type_dtype, type_shape
from verity.ml.circuits.evaluation.reference import signature_of, statics_of
from verity.ml.definitions.numerics import dsv4_rows, rows
from verity.ml.definitions.numerics.dsa_norm_rows import block_rmsnorm_rows
from verity.ml.definitions.numerics.kernel_registry import Declined
from verity.ml.definitions.registry import (dsa_norm, dsv4_compressor, dsv4_decode_indices, dsv4_indexer, dsv4_moe, dsv4_qnorm_insert,
                                          flashmla, flashmla_prefill, mhc)
from verity.ml.definitions.registry.hopper import HopperBF16WgmmaDot16
from verity_vllm.query.manifest import format as RM

#: the 19 families the per-rank no-evaluator rule failed DeepSeek-V4-Pro B1's deferred replay on
DSV4_B1_FAMILIES = ("SiluMulClamp_v1", "TritonBlockRMSNorm_v1", "MhcPostPreTf32_v1", "FlashMlaGatherRow_v1", "Dsv4QNormRopeKvInsert_v1",
                    "MoeRouterSqrtSoftplus_v1", "FlashMlaPrefill_v1", "GatherI32x129280_v1", "MhcFusedPostPre_v1",
                    "Dsv4MlaCompressC4_v1", "FlashMlaDecode_v1", "MoeRouterHash_v1", "MhcPreBroadcast_v1", "MhcResidualBroadcast_v1",
                    "MhcPost_v1", "HcHead_v1", "Dsv4MlaCompressC128_v1", "Dsv4InvRopeQuantHead_v1", "Dsv4CompressorScore_v1")
#: the served Programs' spellings of the last two
HEAD_SPEC, SCORE_SPEC = "Dsv4InvRopeQuantHead_v1{CHUNKS=4,QG=128,ROPE=64}", "Dsv4CompressorScore_v1{W=1024}"

EPS = '{"f64":"0x1.0c6f7a0b5ed8dp-20"}'
TWO5 = f'{{"f64":"{float.hex(2.5)}"}}'
TEN = f'{{"f64":"{float.hex(10.0)}"}}'
FUSE = dict(REPEAT=2, EPS=1e-6, HC_EPS=1e-6)
FUSE_TEXT = f"REPEAT=2,EPS={EPS},HC_EPS={EPS}"
ATTN = dict(DOT=HopperBF16WgmmaDot16, SCALE=32 ** -0.5)
ATTN_TEXT = f'DOT={{"fn":"{HopperBF16WgmmaDot16.id}"}},SCALE={{"f64":"{float.hex(32 ** -0.5)}"}}'
DECODE = dict(NH=2, DN=16, DR=16, QT=16, BN=16, **ATTN)
DECODE_TEXT = f"NH=2,DN=16,DR=16,QT=16,BN=16,{ATTN_TEXT}"

#: (spec text as a Program spells it, the Definition bound at the same statics)
CASES = [
    ("MhcPost_v1{HC=4,N=8}", bind(mhc.MhcPost, HC=4, N=8)),
    (f"MhcFusedPostPre_v1{{HC=2,H=32,NS=2,TH=4,TN=1,{FUSE_TEXT}}}", bind(mhc.MhcFusedPostPre, HC=2, H=32, NS=2, TH=4, TN=1, **FUSE)),
    (f"MhcPostPreTf32_v1{{HC=2,H=64,NS=2,TN=2,{FUSE_TEXT}}}", bind(mhc.MhcPostPreTf32, HC=2, H=64, NS=2, TN=2, **FUSE)),
    (f"MhcPreBroadcast_v1{{HC=2,H=64,NS=1,TN=2,{FUSE_TEXT}}}", bind(mhc.MhcPreBroadcast, HC=2, H=64, NS=1, TN=2, **FUSE)),
    ("MhcResidualBroadcast_v1{HC=4,H=8}", bind(mhc.MhcResidualBroadcast, HC=4, H=8)),
    (f"HcHead_v1{{HC=2,H=16,TH=2,EPS={EPS},HC_EPS=1e-06}}", bind(mhc.HcHead, HC=2, H=16, TH=2, EPS=1e-6, HC_EPS=1e-6)),
    (f"Dsv4MlaCompressC4_v1{{ROWS=8,D=32,QB=16,EPS={EPS}}}", bind(dsv4_compressor.Dsv4MlaCompressC4, ROWS=8, D=32, QB=16, EPS=1e-6)),
    (f"Dsv4MlaCompressC4_v1{{ROWS=4,D=32,QB=16,EPS={EPS}}}", bind(dsv4_compressor.Dsv4MlaCompressC4, ROWS=4, D=32, QB=16, EPS=1e-6)),
    (f"Dsv4MlaCompressC128_v1{{W=2,R=4,D=32,QB=16,EPS={EPS}}}",
     bind(dsv4_compressor.Dsv4MlaCompressC128, W=2, R=4, D=32, QB=16, EPS=1e-6)),
    ("GatherI32x129280_v1", dsv4_decode_indices.gather_i32(129280)),
    (f"FlashMlaDecode_v1{{SPLITS=[0,1,2],NW=16,{DECODE_TEXT}}}", bind(flashmla.FlashMlaDecode, SPLITS=(0, 1, 2), NW=16, **DECODE)),
    (f"FlashMlaDecode_v1{{SPLITS=[0,2],NW=32,{DECODE_TEXT}}}", bind(flashmla.FlashMlaDecode, SPLITS=(0, 2), NW=32, **DECODE)),
    (f"FlashMlaPrefill_v1{{NH=2,NP=1,D=32,BN=16,{ATTN_TEXT}}}", bind(flashmla_prefill.FlashMlaPrefill, NH=2, NP=1, D=32, BN=16, **ATTN)),
    ("FlashMlaGatherRow_v1{DN=32,DR=16,QT=16}", bind(flashmla_prefill.FlashMlaGatherRow, DN=32, DR=16, QT=16)),
    (f"Dsv4QNormRopeKvInsert_v1{{NH=2,NHP=4,DN=48,DR=16,QB=16,EPS={EPS}}}",
     bind(dsv4_qnorm_insert.Dsv4QNormRopeKvInsert, NH=2, NHP=4, DN=48, DR=16, QB=16, EPS=1e-6)),
    (f"TritonBlockRMSNorm_v1{{N=136,BLOCK=256,WARPS=2,EPS={EPS}}}", bind(dsa_norm.TritonBlockRMSNorm, N=136, BLOCK=256, WARPS=2, EPS=1e-6)),
    (f"TritonBlockRMSNorm_v1{{N=1536,BLOCK=2048,WARPS=8,EPS={EPS}}}",
     bind(dsa_norm.TritonBlockRMSNorm, N=1536, BLOCK=2048, WARPS=8, EPS=1e-6)),
    (f"MoeRouterSqrtSoftplus_v1{{E=16,SCALE={TWO5}}}", bind(dsv4_moe.MoeRouterSqrtSoftplus, E=16, SCALE=2.5)),
    (f"MoeRouterHash_v1{{E=16,SCALE={TWO5}}}", bind(dsv4_moe.MoeRouterHash, E=16, SCALE=2.5)),
    (f"SiluMulClamp_v1{{I=9,LIMIT={TEN}}}", bind(dsv4_moe.SiluMulClamp, I=9, LIMIT=10.0)),
    (SCORE_SPEC, bind(dsv4_indexer.Dsv4CompressorScore, W=1024)),
    ("Dsv4InvRopeQuantHead_v1{CHUNKS=2,QG=8,ROPE=4}", bind(dsv4_indexer.Dsv4InvRopeQuantHead, CHUNKS=2, QG=8, ROPE=4)),
    (HEAD_SPEC, bind(dsv4_indexer.Dsv4InvRopeQuantHead, CHUNKS=4, QG=128, ROPE=64)),
]

#: the families whose operands are sampled with their twin's sampler (indices, lengths, e4m3 / e8m0 bytes and edge words a uniform draw
#: rarely makes)
DRAWN = ("FlashMlaDecode_v1", "FlashMlaPrefill_v1", "FlashMlaGatherRow_v1", "Dsv4QNormRopeKvInsert_v1", "MoeRouterSqrtSoftplus_v1",
         "MoeRouterHash_v1", "SiluMulClamp_v1", "Dsv4CompressorScore_v1", "Dsv4InvRopeQuantHead_v1")


def _words(rng: np.random.Generator, pt) -> np.ndarray:
    shape, w = type_shape(pt), set(pt.leaf_widths()).pop()
    if w == 16:
        return ((rng.standard_normal(shape)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return rng.standard_normal(shape).astype(np.float32).view(np.uint32)


def _reference(d, params, args) -> np.ndarray:
    return np.asarray(evaluate_batch(d, *(np.asarray(a).astype(type_dtype(p)).reshape(1, *type_shape(p)) for a, p in zip(args, params)),
                                     kernel="reference")).reshape(-1).astype(np.uint64)


def _row(family: str, statics: dict, args) -> np.ndarray:
    got = rows.ROWS[family].fn(statics, args)
    return np.concatenate([np.asarray(got[m]).reshape(-1).astype(np.uint64) for m in rows.ROWS[family].returns])


def test_every_family_the_dsv4_b1_rule_named_has_a_row_evaluator():
    missing = [f for f in DSV4_B1_FAMILIES if f not in rows.ROWS or f not in rows.EVALUATORS]
    assert not missing, f"no replay row evaluator for {missing}"
    assert set(dsv4_rows.ROWS) <= set(rows.ROWS) and set(dsv4_rows.PROVENANCE) <= set(rows.EVALUATOR_PROVENANCE)


@pytest.mark.parametrize("spec,d", CASES, ids=[s.split("{")[0] + f"-{i}" for i, (s, _) in enumerate(CASES)])
def test_the_row_kernel_on_spec_text_statics_is_the_definition(spec, d):
    family, statics = RM.spec_family(spec), RM.spec_statics(spec)
    params, _ = signature_of(d)
    rng = np.random.default_rng(len(spec))
    sample = dsv4_rows.SAMPLES.get(family) if family in DRAWN else None
    for k in range(3):
        args = [a[0] for a in sample()(rng, 1, params, **statics_of(d))] if sample else [_words(rng, p) for p in params]
        if family.startswith("GatherI32x"):
            args[0] = np.asarray([(5, -1, 129280)[k]], dtype=np.int64).astype(np.uint32)
        got, want = _row(family, statics, args), _reference(d, params, args)
        assert got.shape == want.shape and np.array_equal(got, want), spec


@pytest.mark.parametrize("token", [np.int32(-2), np.int32(129279), np.uint32(0x80000000)])
def test_the_gather_reads_a_committed_i32_token_word_as_signed(token):
    """A committed token word may arrive as int32 or as its u32 bits; either reads as the Definition's signed i32."""
    d = dsv4_decode_indices.gather_i32(129280)
    col = np.arange(129280, dtype=np.uint32) * np.uint32(3)
    got = rows.ROWS["GatherI32x129280_v1"].fn({}, [np.asarray([token]), col])["0"]
    want = _reference(d, signature_of(d)[0], [np.asarray([token]).view(np.uint32), col])
    assert got.astype(np.uint64).tolist() == want.tolist()


@pytest.mark.parametrize("family", ["FlashMlaDecode_v1", "FlashMlaPrefill_v1"])
def test_a_flashmla_row_at_another_dot_step_is_declined(family):
    spec = next(s for s, _ in CASES if RM.spec_family(s) == family).replace(HopperBF16WgmmaDot16.id, "HopperBF16WgmmaDot8_v1")
    with pytest.raises(Declined, match="no twin for DOT"):
        rows.ROWS[family].fn(RM.spec_statics(spec), [])


def test_the_block_norm_row_is_the_definitions_reduction_tree():
    """A served q row (1536 of a 2048 block on 8 warps) rarely shows its reduction tree through the bf16 output, so the rows here are
    ones of a seeded draw whose output differs between 8, 4 and 2 warps: the row kernel is the Definition on each, and the other trees
    are not (which keeps the rows discriminating)."""
    N, BLOCK, R, pick = 1536, 2048, 400, [166, 287, 347]
    rng = np.random.default_rng(11)
    x = ((rng.standard_normal((R, N)) * rng.choice([0.02, 1, 40], (R, 1))).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    w = (rng.standard_normal(N).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    d = bind(dsa_norm.TritonBlockRMSNorm, N=N, BLOCK=BLOCK, WARPS=8, EPS=1e-6)
    want = np.asarray(evaluate_batch(d, x[pick], np.broadcast_to(w, (len(pick), N)).copy(), kernel="reference")).astype(np.uint16)
    statics = RM.spec_statics(f"TritonBlockRMSNorm_v1{{N={N},BLOCK={BLOCK},WARPS=8,EPS={EPS}}}")
    assert [rows.ROWS["TritonBlockRMSNorm_v1"].fn(statics, [x[i], w])["0"].tolist() for i in pick] == want.tolist()
    for warps in (4, 2):
        assert not np.array_equal(block_rmsnorm_rows(x[pick], w, 1e-6, BLOCK, warps)["out"].astype(np.uint16), want), warps


def test_a_hash_id_outside_the_experts_reads_logit_zero():
    """`MoeRouterPick_v1`'s select chain falls through to logits[0] for an id outside [0, E); the row kernel reads the same."""
    d = bind(dsv4_moe.MoeRouterHash, E=16, SCALE=2.5)
    logits = (np.random.default_rng(4).standard_normal(16) * 3).astype(np.float32).view(np.uint32)
    ids = np.asarray([-1, 16, 17, -(1 << 31), (1 << 31) - 1, 3], dtype=np.int64).astype(np.uint32)
    got = _row("MoeRouterHash_v1", RM.spec_statics(f"MoeRouterHash_v1{{E=16,SCALE={TWO5}}}"), [logits, ids])
    assert got.tolist() == _reference(d, signature_of(d)[0], [logits, ids]).tolist()


def test_the_clamp_row_on_non_finite_and_limit_words_is_the_definition():
    """NaN and infinite gates and ups (which the self-check declines as non-finite bf16) and the limit's neighbours."""
    d = bind(dsv4_moe.SiluMulClamp, I=9, LIMIT=10.0)
    lim = dsv4_moe.limit_bits(10.0) >> 16
    g = np.asarray([0x7FC0, 0xFFC0, 0x7F80, 0xFF80, lim + 1, lim | 0x8000, 0x7F81, 0x3F80, 0xC2C8], dtype=np.uint16)
    gu = np.concatenate([g, g[::-1]])
    got = _row("SiluMulClamp_v1", RM.spec_statics(f"SiluMulClamp_v1{{I=9,LIMIT={TEN}}}"), [gu])
    assert got.tolist() == _reference(d, signature_of(d)[0], [gu]).tolist()


def test_the_head_row_on_non_finite_words_is_the_definition():
    """A served head with +inf in a plain group, NaN and -inf among its rotary words and a NaN sin (which the self-check declines
    as non-finite bf16): the infinite groups' scales are +inf, and the row kernel is the Definition on every byte and scale."""
    d = bind(dsv4_indexer.Dsv4InvRopeQuantHead, CHUNKS=4, QG=128, ROPE=64)
    rng = np.random.default_rng(12)
    x = ((rng.standard_normal(512) * 3).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    x[5], x[450], x[500] = 0x7F80, 0x7FC0, 0xFF80
    ang = rng.uniform(0, 2 * np.pi, 32)
    cs = np.concatenate([np.cos(ang), np.sin(ang)]).astype(np.float32).view(np.uint32)
    cs[40] = 0x7FC00000
    got = _row("Dsv4InvRopeQuantHead_v1", RM.spec_statics(HEAD_SPEC), [x, cs])
    assert got.tolist() == _reference(d, signature_of(d)[0], [x, cs]).tolist()
    assert got[512] == got[515] == 0x7F800000 and got[513] != 0x7F800000
