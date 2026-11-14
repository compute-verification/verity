"""verity_vllm.program.predict.dsa over an fp8_ds_mla KV cache (φ's engine `kv_cache_dtype`, GLM-5.3 at `--kv-cache-dtype fp8` under
vLLM's FLASHMLA_SPARSE): each token's latent row is one DsaLatentRowDsMla, which writes its cache row; the launch's sparse rows take
FlashMlaV32Decode over INDEX_TOPK rows each at the planner's splits for one request of that many query rows; a dense-MHA prefill
reads its own rows' bf16 latent and its context through Fp8DsMlaUpconvertRow.

GLM-5.3's checkpoint is cut to its first 4 layers (three full indexers, then one that shares the third's top-k), its INDEX_TOPK
lowered to 64, one block of FlashMLA's, and vLLM's decode threshold at the rank's 8 query heads lowered to 2."""

from __future__ import annotations

import re
import pytest
from verity.definitions.numerics import flashmla_twin as FT
from verity.definitions.registry.flashmla_dsv32 import FLASHMLA_GLM53

from tests.program.predict.test_dsa_dense_prefill import _calls, _families, _topk
from tests.program.predict.test_node import _first_layers
from tests.program.predict.test_predict import GLM53, _row_phi
from verity_vllm.program.frontend.rules import vllm_dsa
from verity_vllm.program.predict.build import build_program
from verity_vllm.program.predict.common import Unexpressible
from verity_vllm.program.predict.phi import Phi

TOPK, NH, LAYERS, INDEXERS = 64, 8, 4, 3


def _fp8(phi: Phi) -> Phi:
    return Phi.from_json({**phi.to_json(), "engine": {**phi.engine, "kv_cache_dtype": "fp8_ds_mla"}})


@pytest.fixture
def bf16(tmp_path, monkeypatch) -> Phi:
    monkeypatch.setitem(vllm_dsa.DECODE_THRESHOLD, NH, 2)
    return _topk(_first_layers(_row_phi(GLM53, "GLM53", tmp_path), LAYERS), TOPK)


@pytest.fixture
def phi(bf16) -> Phi:
    return _fp8(bf16)


def _splits(phi: Phi, tokens: int) -> tuple[int, ...]:
    metas, ns = FT.sched_meta([TOPK], topk=TOPK, parts=FT.num_sm_parts(64, s_q=tokens, num_sms=int(phi.target_profile().num_sms)))
    return FT.request_splits(metas, ns, 0, TOPK // 64)


def test_a_decode_over_an_fp8_cache_is_flashmlas_decode_over_its_cache_rows(phi, bf16):
    """A 3-row prompt prefilled alone (dense MHA, its own rows' bf16 latent), then a decode token at position 3 beside another
    request's: every token's latent row is one DsaLatentRowDsMla, no DsaRMSNorm or RoPE of kv_c and k_pe, and the decode's
    attention is FlashMlaV32Decode at the planner's splits for the step's 2 sparse rows, no SparseMlaSplit.  The default cache's
    Program of the same schedule is SparseMlaSplit's, as before."""
    batch = [[[3, 0]], [[1, 3], [1, 7]]]
    r = build_program(phi, "request", LP=3, T=1, batch=batch, with_desc=False)
    fam = _families(r)
    assert (fam["DsaLatentRowDsMla_v1"], fam["DsaLatentRow_v1"], fam["SparseMlaSplit_v1"]) == (4 * LAYERS, 0, 0)
    assert not [n for n in _calls(r, "DsaRMSNorm_v1") if n.fn.bindings["N"] == 512]
    assert fam["Attention_v4"] == 3 * NH * LAYERS and fam["Fp8DsMlaUpconvertRow_v1"] == 0
    dec = _calls(r, "FlashMlaV32Decode_v1")
    assert len(dec) == LAYERS
    assert all(n.fn.bindings == {**FLASHMLA_GLM53, "SPLITS": _splits(phi, 2)} for n in dec)
    default = _families(build_program(bf16, "request", LP=3, T=1, batch=batch, with_desc=False))
    assert (default["SparseMlaSplit_v1"], default["DsaLatentRowDsMla_v1"], default["FlashMlaV32Decode_v1"]) == (LAYERS, 0, 0)


def test_the_splits_are_the_planners_for_the_launchs_sparse_rows(phi):
    """One decode token alone and beside two others: the planner cuts the call of s_q tokens over max(132 / s_q, 1) partitions,
    so the token's SPLITS move with the launch, not with its own keys."""
    alone = build_program(phi, "request", LP=3, T=1, batch=[[[3, 0]], [[1, 3]]], with_desc=False)
    three = build_program(phi, "request", LP=3, T=1, batch=[[[3, 0]], [[1, 3], [1, 9], [1, 20]]], with_desc=False)
    assert {n.fn.bindings["SPLITS"] for n in _calls(alone, "FlashMlaV32Decode_v1")} == {_splits(phi, 1)}
    assert {n.fn.bindings["SPLITS"] for n in _calls(three, "FlashMlaV32Decode_v1")} == {_splits(phi, 3)}


def test_a_chunk_with_context_upconverts_its_context_rows(phi):
    """A 6-row prompt in two chunks of 3: the second chunk's context partial reads its 3 history rows through
    Fp8DsMlaUpconvertRow, each layer's, then kv_b_proj over them as before."""
    r = build_program(phi, "request", LP=6, T=0, batch=[[[3, 0]], [[3, 3]]], chunks=(3, 3), with_desc=False)
    fam = _families(r)
    assert (fam["Fp8DsMlaUpconvertRow_v1"], fam["DsaLatentRowDsMla_v1"], fam["MergeAttnStates_v1"]) == (3 * LAYERS, 6 * LAYERS, 3 * LAYERS)
    assert {(n.fn.bindings["DN"], n.fn.bindings["DR"]) for n in _calls(r, "Fp8DsMlaUpconvertRow_v1")} == {(512, 64)}
    kv_b = [n for n in _calls(r, "FlashInferFp8Linear_v1") if n.fn.bindings["N"] == NH * (192 + 256)]
    assert len(kv_b) == (3 + 3 + 3) * LAYERS


def test_what_the_fp8_cache_does_not_model_yet_is_refused(phi):
    """A history read as Inputs (a prefix-cache hit), a row past INDEX_TOPK keys, and an INDEX_TOPK that is not whole 64-row
    blocks."""
    with pytest.raises(Unexpressible, match=re.escape("a history of 2 positions over an fp8_ds_mla cache read as Inputs")):
        build_program(phi, "request", LP=4, T=0, C=2, with_desc=False)
    with pytest.raises(Unexpressible, match=re.escape(f"a row of {TOPK + 1} keys, past INDEX_TOPK ({TOPK}), over an fp8_ds_mla cache")):
        build_program(phi, "request", LP=TOPK, T=1, batch=[[[TOPK, 0]], [[1, TOPK]]], with_desc=False)
    with pytest.raises(Unexpressible, match=re.escape("an fp8_ds_mla cache at INDEX_TOPK 16 (FlashMLA reads whole 64-row blocks)")):
        build_program(_topk(phi, 16), "step", M=1, pos0=0, with_desc=False)


def test_the_cache_rows_feed_the_decode(phi):
    """A decode at position 3 reads 4 cache rows: layer 0's nope, scale and rope operands are its 4 DsaLatentRowDsMla Calls'
    (the prompt's 3, then the decode's own), row 0's again to INDEX_TOPK rows, and its index operand 4 words of 0 then -1."""
    r = build_program(phi, "request", LP=3, T=1, batch=[[[3, 0]], [[1, 3]]], with_desc=False)
    nodes = r.program.fn.body.nodes
    rows = [i for i, n in enumerate(nodes) if n.fn.id.startswith("DsaLatentRowDsMla_v1{")]
    dec = next(n for n in nodes if n.fn.id.startswith("FlashMlaV32Decode_v1{"))
    _, nope, scale, rope, idx = (a.refs for a in dec.args)

    def rows_of(refs, width):
        return [refs[j * width][:2] for j in range(TOPK)]

    layer0 = [("n", k) for k in (*rows[:3], rows[3 * LAYERS])]
    for refs, width in ((nope, 512), (scale, 4), (rope, 64)):
        assert rows_of(refs, width) == layer0 + layer0[:1] * (TOPK - 4)
    words = [nodes[idx[j][1]].fn.id for j in range(TOPK)]
    assert words == ["Const32[0x00000000]_v1"] * 4 + ["Const32[0xffffffff]_v1"] * (TOPK - 4)


def test_the_counting_mirrors_refuse_it(phi):
    """`step_needs` and `template_counts` count the default cache's Calls, so over an fp8_ds_mla cache they refuse."""
    from verity_vllm.program.predict.template_counts import TemplateCounts
    with pytest.raises(Unexpressible, match=re.escape("the needs of a step over an fp8_ds_mla cache")):
        TemplateCounts(phi)
