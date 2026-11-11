"""The served templates whose word bodies read a MUFU table are served as their Boolean ``_v2`` versions, which compute the table
in gates: FlashInfer's FP8 linear, a coordinate and its quantisation part (RCP), the MoE router's row and slots form (EX2 and RCP)
and DSA's indexer query (EX2). Inline, a word body's reads put its net past the GPU prover's 32-bit offsets (``GPU_NNZ_MAX``,
`verity/ml/flock/live/src/circuit.rs`: a K = 6144 coordinate's at 9.1e9 nonzeros, the router's at 4.8e10); in lookup slots its
last tail stage returns the outputs, a form ``verify --zk`` refuses (`Flock.KsScope`, ``ProvedScope.scopeOkH``). A Boolean
template is the word template's layout on the same rows, its reads inline, and its staged statement is one flat net that
``--zk`` takes (`one_stage.proofs.ks_form_why`)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.spot.one_stage import proofs as PF
from verity.ml.circuits.codec import descriptor_id
from verity.ml.definitions import boolean as BB
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW

ROUTER = 'E=256,TOPK=8,SCALE={"f64":"0x1.4000000000000p+1"},KERNEL="block"'
#: GLM-5.3's served word templates whose bodies read a MUFU table, at their served statics
SERVED = ("FlashInferFp8Linear_v1{K=6144,N=1}", "FlashInferFp8Linear_v1{K=2048,N=1}", "FlashInferFp8Quant1x128_v1{K=6144}",
          "FlashInferFp8Quant1x128_v1{K=2048}", f"MoeRouterSigmoidBias_v1{{{ROUTER}}}",
          f"MoeRouterSigmoidBiasSlots_v1{{{ROUTER}}}", "DsaIndexerQ_v1{H=32,D=128,ROPE=64}")


def _boolean(word: str):
    wt = SRW.template(word)
    return wt, SRW.template(descriptor_id(BB.boolean_version(wt.fn)))


@pytest.mark.parametrize("word", SERVED)
def test_the_boolean_version_is_the_word_template_on_its_rows_with_its_reads_inline(word):
    wt, bt = _boolean(word)
    assert bt.id != wt.id and bt.word == wt.id and bt.layout == wt.layout
    assert CR.unit_rows(bt) == CR.unit_rows(wt)
    assert CR.statement(bt) == {"reads": "inline"}


def _leaves(rng: np.random.Generator, ty) -> np.ndarray:
    """One row of a parameter of type ``ty`` as u16 words: bf16 of N(0, 1), E4M3 codes other than NaN, or f32 powers of two."""
    w = set(ty.leaf_widths())
    if w == {16}:
        return (rng.standard_normal((1, ty.leaves)).astype("<f4").view("<u4") >> 16).astype("<u2")
    if w == {8}:
        c = rng.integers(0, 256, (1, ty.leaves), dtype=np.uint8)
        c[(c & 0x7F) == 0x7F] ^= 1
        return c.view("<u2")
    assert w == {32}, w
    return np.ldexp(np.ones((1, ty.leaves), np.float32), rng.integers(-10, -3, (1, ty.leaves))).astype("<f4").view("<u2")


def _stage(tp, wt, out: Path, reads: str) -> Path:
    """One input of template ``tp`` on word template ``wt``'s rows, staged as the served composer stages a unit, its MUFU reads
    laid out as ``reads``."""
    from verity_flock import class_statement as CS

    rng = np.random.default_rng(0)
    vals = {p: _leaves(rng, ty) for p, ty in wt.fn.signature()[0]}
    in_rows, out_rows = CR.unit_rows(wt)
    rows = {r: np.concatenate([vals[p] for p in ps], axis=1) for r, _s, _b, ps in in_rows}
    out.mkdir()
    st = CS.UnitStager(tp.fn, "unit/serving-rows", in_rows, out_rows, circuit=out / "circuit.txt", reads=reads)
    st.stage(rows, out, hashlib.sha256(b"verity/tests/serving-rows-gates").digest())
    return out


@pytest.mark.slow
def test_a_boolean_version_stages_one_flat_net_zk_takes_where_the_word_template_in_slots_is_refused(tmp_path):
    """FlashInfer's coordinate and quantisation part at K = 128: their Boolean versions' staged statements pass `ks_form_why`, and
    the word coordinate with its reads in lookup slots is refused for its form. `ks_form_why` reads the staged circuit's head (its
    ``out_net`` and input groups), which only a staging writes, so the fast test can't check it; about a minute."""
    for part, word in (("coordinate", "FlashInferFp8Linear_v1{K=128,N=1}"), ("quant", "FlashInferFp8Quant1x128_v1{K=128}")):
        wt, bt = _boolean(word)
        d = _stage(bt, wt, tmp_path / part, "inline")
        assert PF.ks_form_why(d / "circuit.txt", d / "pub.bin") is None, bt.id
    wt = SRW.template("FlashInferFp8Linear_v1{K=128,N=1}")
    d = _stage(wt, wt, tmp_path / "word-slots", "slots")
    why = PF.ks_form_why(d / "circuit.txt", d / "pub.bin")
    assert why is not None and "scopeOkH" in why, why
