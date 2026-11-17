"""GLM-5.3 at TP8 over an FP8 KV cache (`--kv-cache-dtype fp8`, vLLM's `fp8_ds_mla` layout under FLASHMLA_SPARSE), folded from
RECORDED events.

`fixtures/glm53_tp8_fp8_ds_mla.rank0.jsonl.gz` is rank 0's whole observer log of a tp-capture of the row
`glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager` at `kv_cache_dtype=fp8`, two tokens, on the 8x H200 node (vLLM
d9105ea80, eager): the 256-row prefill and one decode row at position 256.  Its source is named by the fixture's store record.

* every layer's four FP8-cache kernels fold on both steps: `_fused_norm_rope_kernel` writing the token's 656-byte cache row is
  `DsaLatentRowDsMla_v1` a token beside the q norm and the index-K row, `concat_mla_q` is a copy, the 128-column request-index
  conversion is staging, and FlashMLA's `sparse_decode_fwd` is `FlashMlaV32Decode_v1` a row over the request's cache rows; the
  indexer's invisible logits allocation and its top-k then place as on the bf16 cache, and nothing of either step is refused or left
  unread;
* the decode step's fold is the predicted step of the row's rank 0 over an `fp8_ds_mla` cache (`aot.predict`, M 1 at position 256),
  Call for Call in order, but for what a step Program can't hold: it reads no history (`build.build_program` takes no prefix on a
  step), so each indexer layer's logits read its own key alone where the served row reads its history's 257;
* on the bf16 cache the patterns form exactly what they formed before the FP8-cache patterns were added (a digest of each
  resolution of the bf16 launches `test_fold_glm53_dsa_launches` builds).
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import collections
import dataclasses
import difflib
import hashlib
import json
from pathlib import Path

import pytest
from verity.circuits.codec import descriptor_id, encode_program

from tests.observe import test_fold_glm53_dsa_launches as B
from verity_vllm.check.match import program_compare as PC
from verity_vllm.check.match.instance_sequence import RootView
from verity_vllm.config import WORKLOADS, RowEngine, workload_file
from verity_vllm.engine.profiles.dense_generic import HF_CONFIG_DIR
from verity_vllm.observe.fold.fold import fold
from verity_vllm.observe.fold.patterns import dsa as D
from verity_vllm.observe.fold.patterns import dsa_flashmla as F
from verity_vllm.observe.fold.resolve_log import load_profile
from verity_vllm.observe.fold.resolver import Compute, Group, spec_label
from verity_vllm.observe.log import read_log
from verity_vllm.program.predict import aot
from verity_vllm.program.predict import score as S
from verity_vllm.program.predict.phi import Phi, _refuse_unstated

FIXTURE = Path(__file__).parent / "fixtures" / "glm53_tp8_fp8_ds_mla.rank0.jsonl.gz"
ROW = "glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager"
LAYERS, INDEXERS, PROMPT = 78, 21, 256
FP8_KINDS = ("_fused_norm_rope_kernel", "concat_mla_q", "ConvertReqIndexToGlobalIndexKernel", "sparse_decode_fwd")


@pytest.fixture(scope="module")
def folded():
    """The fold, shared by this module's tests and released after them: a worker that went on holding it would carry its GBs into
    every later test it runs."""
    return fold(read_log(str(FIXTURE)), load_profile("derived_GLM53_tp8")[0], strict=False, with_program=False)


@pytest.mark.slow
def test_both_steps_fold_every_fp8_cache_launch_and_nothing_is_refused_or_unread(folded):
    """Minutes (the fold of all 78 layers of a 256-row prefill and a decode row): the only test that holds the FP8-cache patterns to
    launches vLLM served, whose arguments, cache views and FlashMLA split plan no synthetic launch reproduces."""
    r = folded
    assert r.report.unsupported_ops == {}, r.report.unsupported_examples[:4]
    assert r.errors == [], r.errors[:4]
    assert [s.get("unsupported") for s in r.steps] == [0, 0]
    by = r.report.by_pattern
    assert {k: by.get(k) for k in ("dsa.fused_norm_rope", "dsa.concat_mla_q", "dsa.index_share_copy", "dsa.flashmla_sparse_decode")} == {
        "dsa.fused_norm_rope": 2 * LAYERS, "dsa.concat_mla_q": 2 * LAYERS, "dsa.index_share_copy": 2 * LAYERS,
        "dsa.flashmla_sparse_decode": 2 * LAYERS}
    rows = collections.Counter()
    for label, n in r.report.by_spec.items():
        rows[label.split("{")[0]] += n
    tokens = PROMPT + 1
    assert rows["DsaLatentRowDsMla_v1"] == LAYERS * tokens and rows["FlashMlaV32Decode_v1"] == LAYERS * tokens
    assert rows["DsaIndexerKRow_v1"] == rows["DsaMqaLogits_v3"] == rows["DsaTopKShortRow_v1"] == INDEXERS * tokens
    assert r.reference_eligible


def _predicted_decode() -> list[str]:
    """Rank 0's predicted step over an fp8_ds_mla cache at the capture's decode (M 1 at position 256), T-normalised specs in order."""
    w = json.loads(workload_file(WORKLOADS / f"{ROW}.json").read_text())
    hf = json.loads((HF_CONFIG_DIR / "GLM53.config.json").read_text())
    layout = _refuse_unstated({"kv_cache_dtype": "fp8"}, hf.get("model_type"), (9, 0))
    base = S.phi_of({"row": ROW, "workload": w}, config=str(HF_CONFIG_DIR / "GLM53.config.json"))
    phi = Phi.from_json({**base.to_json(), "tp": {**base.tp, "rank": 0}, "engine": {**base.engine, "kv_cache_dtype": layout}})
    assert RowEngine.of(w).revision and layout == "fp8_ds_mla"
    view = RootView("predicted", encode_program(aot.predict(phi, "step", M=1, pos0=PROMPT).program), {}, tok_node=0, compact=True)
    return [PC._norm_T(view.inst[i]["spec"]) for s in view.steps for i in s]


@pytest.mark.slow
def test_the_decode_step_s_fold_is_the_predicted_step_call_for_call(folded):
    """Minutes (the fold, then the rank's step prediction over 78 layers): the only test that holds the served FP8-cache step to
    the Program the predictor writes ahead of it. A step Program reads no history, so on each indexer layer its logits and their
    length read one key where the served row reads its history's PROMPT + 1; nothing else differs."""
    r = folded
    served = [PC._norm_T(descriptor_id(r.body.nodes[i["node"]].fn)) for i in r.instances if int(i["step"]) == 1]
    predicted = _predicted_decode()
    sm = difflib.SequenceMatcher(None, predicted, served, autojunk=False)
    apart = [(predicted[i1:i2], served[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
    keys = PROMPT + 1
    history = (["DsaMqaLogits_v3{N=1,H=32,D=128}", "Const32[0x00000001]_v1"],
               [f"DsaMqaLogits_v3{{N={keys},H=32,D=128}}", f"Const32[0x{keys:08x}]_v1"])
    hp, hs = collections.Counter(predicted), collections.Counter(served)
    assert len(predicted) == len(served) and apart == [history] * INDEXERS, {
        "unmatched_predicted": dict(hp - hs), "unmatched_served": dict(hs - hp), "first": apart[:6]}


def _canon(r) -> object:
    if isinstance(r, Group):
        return ("Group", r.consumed, r.hint, r.pattern, tuple(_canon(p) for p in r.parts))
    if isinstance(r, Compute):
        return tuple((f.name, spec_label(r.spec) if f.name == "spec" else repr(getattr(r, f.name))) for f in dataclasses.fields(r))
    return repr(r)


#: sha256 of the bf16 launches' resolutions below and the layer state they leave, as the patterns formed them before the FP8 cache's
BF16_DIGEST = "72169a2ef982ffe8fa2152c9548e93235afad8df0d9d2213831f2f43e25a8cf9"


def test_the_bf16_cache_s_launches_fold_as_before():
    ctx = B._ctx(B.TWO, slots=[5, 70])
    ctx.header = dataclasses.replace(ctx.header, profile={"gpu": {"multi_processor_count": 132}})
    made = [D.DsaNormRope().match(B._norm(), ctx), D.DsaQ().match(B._q(), ctx), D.DsaIndexShare().match(B._share(), ctx),
            D.DsaSparseMla().match(B._mla(**B.MLA_DEFAULT, **B._halves())[0], ctx), D.DsaNormRope().match(B._norm(indexer=False), ctx)]
    others = [p for p in D.dsa_patterns() if type(p).__module__ in (D.__name__, F.__name__)
              and not isinstance(p, (D.DsaNormRope, D.DsaIndexShare, D.DsaSparseMla))]
    made += [r for p in others for node in (B._norm(), B._share(), B._mla(**B.MLA_DEFAULT, **B._halves())[0]) if (r := p.match(node, ctx))]
    state = {k: repr(v) for k, v in sorted(D._state(ctx)["modules"][B.LAYER].items())}
    blob = repr(([_canon(r) for r in made], state, D._state(ctx)["topk"])).encode()
    assert hashlib.sha256(blob).hexdigest() == BF16_DIGEST
