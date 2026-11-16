"""verity_vllm.program.predict.mtp: GLM-5.3's MTP draft forward over a target cut to its first 2 layers (a dense layer and a MoE
one, both with a full indexer; `first_k_dense_replace` 1, so the draft layer, index 2, is MoE as layer 78 is served), against
vLLM d9105ea80's launches in order, each segment with the lines it is read from (`mtp`'s module docstring).  Paths are under
`vllm/`; `nvidia/` is `models/deepseek_v32/nvidia/` and `speculator.py` is `v1/worker/gpu/spec_decode/autoregressive/speculator.py`."""

from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import re
from bisect import bisect_right
from itertools import accumulate

import pytest
from verity.circuits.defs import Builder, bind
from verity.circuits.refs import Affine, Coll
from verity.circuits.types import Array, RecordType, Value
from verity_vllm.definitions.registry import mtp_draft as MD
from verity_vllm.definitions.registry.prims import BF16
from verity_vllm.program.predict import mtp as MTP
from verity_vllm.program.predict.common import Unexpressible, _arr, _Peers
from verity_vllm.program.predict.dsa import Net as TargetNet
from verity_vllm.program.predict.phi import STATED_SPECULATIVE, Phi

from tests.program.predict.test_predict import GLM53, _row_phi

I32, I64 = Value(32), Value(64)
LAYERS, H, R = 2, 6144, 64


def _cut(phi: Phi, speculative: dict | None = STATED_SPECULATIVE) -> Phi:
    hf = phi.hf_config
    hf.update(num_hidden_layers=LAYERS, indexer_types=["full"] * LAYERS, mlp_layer_types=["dense", "sparse"], first_k_dense_replace=1)
    engine = {**phi.engine, **({"speculative": dict(speculative)} if speculative else {})}
    return Phi.from_json({**phi.to_json(), "model": {**phi.model, "hf_config": hf}, "engine": engine})


@pytest.fixture
def phi(tmp_path) -> Phi:
    return _cut(_row_phi(GLM53, "GLM53", tmp_path))


class Run:
    """The target's forward over a prompt of `rows` at position 0, then draft step 0 over the same rows keeping `kept` of them,
    each row past them a `Leftover` of Inputs, all in one body."""

    def __init__(self, phi: Phi, rows: int, kept: int, steps: int = 0) -> None:
        net = self.net = MTP.Net(phi, rows, rows)
        world, left = net.world, rows - kept
        shapes = [*net.peer_shapes(rows)[:-1], *net.draft_peer_shapes(rows), *net.draft_peer_shapes(1) * steps]
        params = [("prompt", Array(rows, I32)), ("sampled", Array(1, I32)), ("left_ids", Array(left, I32)),
                  ("left_pos", Array(left, I64)), ("left_rope", Array(left, Array(R, BF16))),
                  *[(f"peers_{k}", _arr(s, 16)) for k, s in enumerate(s for s in shapes for _ in range(world - 1))],
                  ("weights", RecordType(*net.leaves()))]
        self.B = B = Builder(tuple(params), Value(1))
        inputs = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)]
        self.W = W = inputs[-1]
        target = net.model(B, W, _Peers(inputs[5:-1], world, net.rank))
        self.ids = [Coll(I32, inputs[0].refs.slice(i, i + 1)) for i in range(rows)]
        self.sampled = inputs[1]
        self.left = [MTP.Leftover(Coll(I32, inputs[2].refs.slice(j, j + 1)), Coll(I64, inputs[3].refs.slice(j, j + 1)),
                                  Coll(Array(R, BF16), inputs[4].refs.slice(j * R, (j + 1) * R))) for j in range(left)]
        self.hidden = target.forward(self.ids, 0)
        self.mark = len(B.nodes)
        self.draft = net.draft(target, W)
        self.s0 = self.draft.step0(self.ids, self.sampled, self.hidden, 0, kept, self.left)
        self.nodes = B.nodes[self.mark:]

    def ids_of(self, nodes) -> list[str]:
        return [n.fn.id for n in nodes]

    def weights_read(self, nodes) -> set[str]:
        """The weight leaves the nodes' arguments start in."""
        leaves = self.net.leaves()
        ends, iw = list(accumulate(t.leaves for _, t in leaves)), len(self.B.params) - 1
        return {leaves[bisect_right(ends, r[2])][0] for n in nodes for a in n.args for r in (a.refs[0],) if r[:2] == ("p", iw)}


def _family(i: str) -> str:
    """A call's Definition family: its id before its statics (`{...}`) or a const's word (`Const64[0x...]_v1`)."""
    return re.split(r"[{\[]", i, maxsplit=1)[0]


def _layer_reference(phi: Phi, rows: int) -> list[str]:
    """`dsa._Model`'s calls for an indexer MoE layer (target layer 1) over `rows` rows at position 0, after its input norm: its
    attention, the post-attention fused all-reduce norm, the MoE block and the fused all-reduce norm after it."""
    net = MTP.Net(phi, rows, rows)
    shapes = [s for s in [(rows, H)] * 2 for _ in range(net.world - 1)]
    params = [("x", Array(rows, Array(H, BF16))), *[(f"peers_{k}", _arr(s, 16)) for k, s in enumerate(shapes)],
              ("weights", RecordType(*net.leaves()))]
    B = Builder(tuple(params), Value(1))
    inputs = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)]
    m = net.model(B, inputs[-1], _Peers(inputs[1:-1], net.world, net.rank))
    x = [inputs[0].refs.slice(i * H, (i + 1) * H) for i in range(rows)]
    y, res = m.reduce_norm(m.attention(1, x, 0), x, m.w[(1, "ln2")])
    m.reduce_norm(m.moe(1, y), res, m.w["norm"])
    return [n.fn.id for n in B.nodes]


def _eh_cut_ids() -> list[str]:
    (zero, _), (sumsq, _), (apply, n) = MD.eh_cut(bind(MD.MtpEhNorm, H=H, BLOCK=8192, WARPS=4, EPS=1e-5))
    return [zero.id, *([sumsq.id] + [apply.id] * (n // 2)) * 2]


def test_draft_step0_emits_the_pins_launches_in_order(phi):
    """Draft step 0 over a 3-row prompt, every row kept: each segment of its calls in order, then nothing else."""
    run = Run(phi, 3, 3)
    ids, target = run.ids_of(run.nodes), run.ids_of(run.B.nodes[:run.mark])
    n, at = len(ids), 0

    def take(k: int) -> list[str]:
        nonlocal at
        seg, at = ids[at:at + k], at + k
        return seg

    # the prepare kernel (speculator.py:714-730): wires; each kept row's position, the eh-norm's I64 operand
    assert [_family(i) for i in take(3)] == ["Const64"] * 3
    # the embedding shard and its all-reduce (nvidia/mtp.py:215-222): the target's own first launches
    assert take(4) == target[:4] and [_family(i) for i in target[:3]] == ["EmbeddingShard_v1"] * 3
    # _fused_eh_norm_kernel (nvidia/mtp.py:115-123) per row as MtpEhNorm_v1's cut: the select, then each half's scale and 96 applies
    assert take(3 * 195) == _eh_cut_ids() * 3
    # eh_proj (nvidia/mtp.py:132-134; nvidia/glm52_low_latency_gemm.py:143-148): cuBLAS's bf16 linear from 2H to H, each row
    eh = take(3)
    assert {_family(i) for i in eh} <= {"Gemm_v2", "CublasSplitKGemm_v1"} and len(set(eh)) == 1 and "K=12288" in eh[0] and "N=6144" in eh[0]
    # the decoder layer's input norm on eh_proj's rows (nvidia/model.py:125-131): layer 0's kind
    assert take(3) == [target[4]] * 3
    # layer 2 with its indexer on (nvidia/mtp.py:135-137; nvidia/model.py:146-160): dsa's indexer MoE layer, its fused norms
    # the post-attention one and shared_head.norm's (nvidia/mtp.py:154-156)
    layer = _layer_reference(phi, 3)
    assert take(len(layer)) == layer and any(_family(i) == "DsaTopKShortRow_v1" for i in layer)
    # at the sampled row (speculator.py:467-482; nvidia/mtp.py:233-243): the head GEMM, the logits all-gather and the greedy select
    head = take(3)
    assert _family(head[0]) in {"Gemm_v2", "CublasSplitKGemm_v1"} and _family(head[1]) == "AllGather_v1"
    assert _family(head[2]).startswith("TokenSelect_")
    # compact_topk_indices (nvidia/mtp.py:195-202): wires
    assert at == n and run.s0.topk is None


def test_the_draft_layer_reads_its_own_weights_and_the_targets_embedding_and_head(phi):
    """Every weight the draft step reads is layer 2's (its own and its decoder layer's), the embedding, the head or the RoPE table."""
    run = Run(phi, 3, 3)
    read = run.weights_read(run.nodes)
    shared = {"model.embed_tokens.weight", "lm_head.weight"}
    assert shared <= read and all(n.startswith("model.layers.2.") or n in shared or "cos_sin" in n for n in read)
    assert {f"model.layers.2.{s}" for s in MTP.OWN} <= read
    assert any(".indexer." in n for n in read) and any(".mlp.experts." in n for n in read)


def test_rows_past_kept_are_leftover_inputs(phi):
    """A 6-row step keeping 3: rows 0 and 1 embed the target's ids 1 and 2, row 2 the sampled token, rows 3..5 their Leftover ids;
    their positions are the Leftover Inputs, their RoPE rows the Leftover rows; the sample is at row 2."""
    run = Run(phi, 6, 3)
    emb = [n for n in run.nodes if _family(n.fn.id) == "EmbeddingShard_v1"]
    want = [run.ids[1].refs[0], run.ids[2].refs[0], run.sampled.refs[0], *(o.id.refs[0] for o in run.left)]
    assert [n.args[0].refs[0] for n in emb] == want
    zero = [n for n in run.nodes if _family(n.fn.id) == "MtpEhZero_v1"]
    assert [n.args[0].refs[0][:2] for n in zero[3:]] == [("p", 3)] * 3 and all(n.args[0].refs[0][0] == "n" for n in zero[:3])
    ropes = [n for n in run.nodes if _family(n.fn.id) == "RoPEHeadGptJ_v2"]
    assert {n.args[1].refs[0][:2] for n in ropes[3:6]} == {("p", 4)}
    sel = next(n for n in run.nodes if _family(n.fn.id).startswith("TokenSelect_"))
    assert sel is run.nodes[-1] and run.s0.hidden is not None


def _fp8(phi: Phi) -> Phi:
    return Phi.from_json({**phi.to_json(), "engine": {**phi.engine, "kv_cache_dtype": "fp8_ds_mla"}})


def _attention_spans(monkeypatch) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    attention = MTP.Draft.attention

    def spanned(self, *a, **kw):
        start = len(self.B.nodes)
        out = attention(self, *a, **kw)
        spans.append((start, len(self.B.nodes)))
        return out

    monkeypatch.setattr(MTP.Draft, "attention", spanned)
    return spans


F, LR, BG, Q = "FlashInferFp8Linear_v1", "DsaLatentRowDsMla_v1", "BatchedGemm_v1", "RoPEHeadGptJ_v2"
NORM, K32, MLA = "DsaRMSNorm_v1", "Const32", "FlashMlaV32Decode_v1"
INDEXER = {"DsaIndexerKRow_v1", "DsaIndexerQ_v1", "DsaMqaLogits_v3", "DsaTopKShortRow_v1"}
NH = 8

STEP0_ATTENTION = ([F] * 3 + ["CublasSplitKGemm_v1"] * 3 + [NORM] * 3 + [LR] * 3 + ["DsaIndexerKRow_v1"] * 3 + [F] * 3 + [BG] * 3 + [F] * 3
                   + [Q] * (3 * NH) + ["DsaIndexerQ_v1"] * 3 + ["DsaMqaLogits_v3", K32, "DsaTopKShortRow_v1"] * 3
                   + [K32, K32] + [MLA] * 3 + [BG] * 3 + [F] * 3)
STEP1_ATTENTION = [F, NORM, LR, F, BG, *[Q] * NH, K32, K32, MLA, BG, F]


def test_a_drafted_step_reads_step_0s_selection_with_its_indexer_off(phi, monkeypatch):
    from verity_vllm.definitions.numerics import flashmla_twin as FT

    spans = _attention_spans(monkeypatch)
    run = Run(_fp8(phi), 3, 3, steps=1)
    mark = len(run.B.nodes)
    s1 = run.draft.decode(run.s0, 1, 3)
    (a0, b0), (a1, b1) = spans
    nodes = run.B.nodes
    assert [_family(n.fn.id) for n in nodes[a0:b0]] == STEP0_ATTENTION
    assert [n.fn.bindings["N"] for n in nodes[a0:b0] if _family(n.fn.id) == "DsaMqaLogits_v3"] == [1, 2, 3]
    assert [_family(n.fn.id) for n in nodes[a1:b1]] == STEP1_ATTENTION
    assert not INDEXER & {_family(n.fn.id) for n in nodes[mark:]}
    assert run.s0.keys == s1.keys == 3 and s1.topk is None and _family(nodes[-1].fn.id).startswith("TokenSelect_")
    assert s1.token.refs[0][:2] == ("n", len(nodes) - 1)
    TOPK = int(run.net.C["INDEX_TOPK"])
    mla = nodes[b1 - 3]
    meta, ns = FT.sched_meta([TOPK], topk=TOPK, parts=FT.num_sm_parts(64, s_q=1, num_sms=int(phi.target_profile().num_sms)))
    assert mla.fn.bindings["SPLITS"] == FT.request_splits(meta, ns, 0, TOPK // 64)
    _, nope, _, _, idx = (a.refs for a in mla.args)
    latent = [i for i in range(a0, b1) if _family(nodes[i].fn.id) == LR]
    assert [nope[j * 512][:2] for j in range(3)] == [("n", i) for i in latent[:3]]
    assert ("n", latent[3]) not in {r[:2] for r in nope}
    valid, pad = (nodes[b1 - 5 + k] for k in (0, 1))
    assert len(idx) == 32 * TOPK and valid.fn.id != pad.fn.id
    assert [idx[32 * j][:2] for j in range(TOPK)] == [("n", b1 - 5)] * 3 + [("n", b1 - 4)] * (TOPK - 3)
    logs = run.draft.logs[run.net.NL]
    assert (len(logs["latent"]), len(logs["ik"])) == (4, 3)


def test_a_drafted_step_over_the_default_cache_reads_step_0s_rows(phi, monkeypatch):
    spans = _attention_spans(monkeypatch)
    run = Run(phi, 3, 3, steps=1)
    mark = len(run.B.nodes)
    run.draft.decode(run.s0, 1, 3)
    a1, b1 = spans[1]
    split = [n for n in run.B.nodes[a1:b1] if _family(n.fn.id) == "SparseMlaSplit_v1"]
    assert len(split) == 1 and split[0].fn.bindings["T"] == 3
    assert not INDEXER & {_family(n.fn.id) for n in run.B.nodes[mark:]}


def test_a_drafted_step_reusing_a_row_past_index_topk_is_refused(phi):
    run = Run(_fp8(phi), 3, 3, steps=1)
    TOPK = int(run.net.C["INDEX_TOPK"])
    with pytest.raises(Unexpressible, match=re.escape(f"a row reusing the selection of a row of {TOPK + 1} keys, past INDEX_TOPK "
                                                      f"({TOPK})")):
        run.draft.decode(run.s0._replace(keys=TOPK + 1), 1, 3)


def test_the_weights_record_is_the_targets_then_the_draft_layers(phi, tmp_path):
    """The target's record is a prefix; the draft layer's own four leaves come first, then its decoder layer's, every name layer
    2's.  A φ that drafts nothing is refused."""
    net, target = MTP.Net(phi, 1, 1), TargetNet(phi, 1, 1)
    leaves, base = net.leaves(), target.leaves()
    assert leaves[:len(base)] == base
    own = leaves[len(base):]
    assert [n for n, _ in own[:4]] == [f"model.layers.2.{s}" for s in MTP.OWN] and all(n.startswith("model.layers.2.") for n, _ in own)
    assert own[2][1] == Array(H, Array(2 * H, BF16))
    layer1 = [n.removeprefix("model.layers.1.") for n, _ in base if n.startswith("model.layers.1.")]
    assert [n.removeprefix("model.layers.2.") for n, _ in own[4:]] == layer1
    with pytest.raises(Unexpressible, match="MTP draft under engine speculative None"):
        MTP.Net(_cut(_row_phi(GLM53, "GLM53", tmp_path / "plain"), None), 1, 1)


def test_the_draft_layer_map_is_the_records_draft_leaves(phi):
    """`draft_layer`'s names are exactly the record's draft-layer leaves, in record order; its configuration is one layer longer
    with layer 2 an indexer layer; a decoder layer's path is a path of that configuration's record."""
    net, base = MTP.Net(phi, 1, 1), TargetNet(phi, 1, 1).leaves()
    layer = MTP.draft_layer(phi.hf_config, net.C)
    assert list(layer.paths) == [n for n, _ in net.leaves()[len(base):]]
    assert layer.config["NL"] == LAYERS + 1 and LAYERS in layer.config["INDEXER_LAYERS"]
    assert layer.paths["model.layers.2.eh_proj.weight"] == ("eh_proj",)
    assert {p[0] for p in layer.paths.values() if p is not None and len(p) > 1} == {"layers", "indexers"}
