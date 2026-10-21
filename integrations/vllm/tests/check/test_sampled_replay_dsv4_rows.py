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
from verity.ml.circuits.evaluation.reference import signature_of
from verity_vllm.program.kernels import dsv4_rows, rows
from verity_vllm.program.registry import dsv4_compressor, dsv4_decode_indices, mhc
from verity_vllm.query.manifest import format as RM

#: the families the per-rank no-evaluator rule failed DeepSeek-V4-Pro B1's deferred replay r20261021-114900-5d55 on that this
#: module evaluates
DSV4_B1_FAMILIES = ("MhcPostPreTf32_v1", "GatherI32x129280_v1", "MhcFusedPostPre_v1", "Dsv4MlaCompressC4_v1", "MhcPreBroadcast_v1",
                    "MhcResidualBroadcast_v1", "MhcPost_v1", "HcHead_v1", "Dsv4MlaCompressC128_v1")

EPS = '{"f64":"0x1.0c6f7a0b5ed8dp-20"}'
FUSE = dict(REPEAT=2, EPS=1e-6, HC_EPS=1e-6)
FUSE_TEXT = f"REPEAT=2,EPS={EPS},HC_EPS={EPS}"

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
]


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
    for k in range(3):
        args = [_words(rng, p) for p in params]
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
