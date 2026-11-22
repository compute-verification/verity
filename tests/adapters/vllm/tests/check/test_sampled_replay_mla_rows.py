"""FlashMLA's and the masked-block RMSNorm's row evaluators (`numerics.mla_rows`) as the sampled replay calls them.  The replay hands
a row kernel the statics `spec_statics` parses from the Program's spec text (`{"f64":...}` floats), not the Python values the kernel
self-check binds, so each family is checked here on those spellings, at small statics, against its Definition's word reference, every
port in the Definition's return order."""
from __future__ import annotations

import numpy as np
import pytest
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.evaluation.batch import type_dtype, type_shape
from verity.circuits.evaluation.reference import signature_of, statics_of

from verity_vllm.definitions.numerics import mla_rows, rows
from verity_vllm.definitions.numerics.dsa_norm_rows import block_rmsnorm_rows
from verity_vllm.definitions.numerics.kernel_registry import Declined
from verity_vllm.definitions.registry import dsa_norm, flashmla, flashmla_prefill
from verity_vllm.definitions.registry.hopper import HopperBF16WgmmaDot16
from verity_vllm.query.manifest import format as RM

EPS = '{"f64":"0x1.0c6f7a0b5ed8dp-20"}'
ATTN = dict(DOT=HopperBF16WgmmaDot16, SCALE=32 ** -0.5)
ATTN_TEXT = f'DOT={{"fn":"{HopperBF16WgmmaDot16.id}"}},SCALE={{"f64":"{float.hex(32 ** -0.5)}"}}'
DECODE = dict(NH=2, DN=16, DR=16, QT=16, BN=16, **ATTN)
DECODE_TEXT = f"NH=2,DN=16,DR=16,QT=16,BN=16,{ATTN_TEXT}"

#: (spec text as a Program spells it, the Definition bound at the same statics)
CASES = [
    (f"FlashMlaDecode_v1{{SPLITS=[0,1,2],NW=16,{DECODE_TEXT}}}", bind(flashmla.FlashMlaDecode, SPLITS=(0, 1, 2), NW=16, **DECODE)),
    (f"FlashMlaDecode_v1{{SPLITS=[0,2],NW=32,{DECODE_TEXT}}}", bind(flashmla.FlashMlaDecode, SPLITS=(0, 2), NW=32, **DECODE)),
    (f"FlashMlaPrefill_v1{{NH=2,NP=1,D=32,BN=16,{ATTN_TEXT}}}", bind(flashmla_prefill.FlashMlaPrefill, NH=2, NP=1, D=32, BN=16, **ATTN)),
    ("FlashMlaGatherRow_v1{DN=32,DR=16,QT=16}", bind(flashmla_prefill.FlashMlaGatherRow, DN=32, DR=16, QT=16)),
    (f"TritonBlockRMSNorm_v1{{N=136,BLOCK=256,WARPS=2,EPS={EPS}}}", bind(dsa_norm.TritonBlockRMSNorm, N=136, BLOCK=256, WARPS=2, EPS=1e-6)),
    (f"TritonBlockRMSNorm_v1{{N=1536,BLOCK=2048,WARPS=8,EPS={EPS}}}",
     bind(dsa_norm.TritonBlockRMSNorm, N=1536, BLOCK=2048, WARPS=8, EPS=1e-6)),
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


def test_every_row_kernel_here_is_registered():
    assert set(mla_rows.ROWS) <= set(rows.ROWS) and set(mla_rows.PROVENANCE) <= set(rows.EVALUATOR_PROVENANCE)


@pytest.mark.parametrize("spec,d", CASES, ids=[s.split("{")[0] + f"-{i}" for i, (s, _) in enumerate(CASES)])
def test_the_row_kernel_on_spec_text_statics_is_the_definition(spec, d):
    family, statics = RM.spec_family(spec), RM.spec_statics(spec)
    params, _ = signature_of(d)
    rng = np.random.default_rng(len(spec))
    sample = mla_rows.SAMPLES.get(family)
    for _ in range(3):
        args = [a[0] for a in sample()(rng, 1, params, **statics_of(d))] if sample else [_words(rng, p) for p in params]
        got, want = _row(family, statics, args), _reference(d, params, args)
        assert got.shape == want.shape and np.array_equal(got, want), spec


@pytest.mark.parametrize("family", ["FlashMlaDecode_v1", "FlashMlaPrefill_v1"])
def test_a_flashmla_row_at_another_dot_step_is_declined(family):
    spec = next(s for s, _ in CASES if RM.spec_family(s) == family).replace(HopperBF16WgmmaDot16.id, "HopperBF16WgmmaDot8_v1")
    with pytest.raises(Declined, match="no twin for DOT"):
        rows.ROWS[family].fn(RM.spec_statics(spec), [])


def test_the_block_norm_row_is_the_definitions_reduction_tree():
    """A served q row (1536 of a 2048 block on 8 warps) rarely shows its reduction tree through the bf16 output, so the rows here are
    ones of a seeded sample whose output differs between 8, 4 and 2 warps: the row kernel is the Definition on each, and the other trees
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
