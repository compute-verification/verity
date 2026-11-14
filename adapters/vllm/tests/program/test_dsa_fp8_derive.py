"""The Build's DeepSeek sparse attention over an `fp8_ds_mla` KV cache (GLM-5.3 at `--kv-cache-dtype fp8` on the H200, vLLM's
FLASHMLA_SPARSE), on `test_dsa_derive`'s synthetic layer at a latent `DsaLatentRowDsMla_v1` takes:

* `dsa_leaves.dsa_leaves` exports `fused_norm_rope` as `vllm_iface.dsa_norm_rope_ds_mla`, whose outputs carry the cache row's
  NoPE bytes and scales, and `_sparse_indexer_and_attn` as `vllm_iface.dsa_flashmla_attn` over them;
* `rules/vllm_dsa_fp8.py` derives per token one `DsaLatentRowDsMla_v1` and per row `FlashMlaV32Decode_v1` over TOPK cache rows,
  the history's first n and then row 0's under index -1, at the planner's splits for the call's rows;
* a prefill vLLM serves by dense MHA derives as over the default cache (`Attention_v4` over the call's own bf16 rows), its cache
  rows opening the history the decode after it reads;
* each precondition refuses by name.

The fixture's FlashMLA geometry is one 16-row block at a tile of 128 NoPE values; its norms and absorbs are `test_dsa_derive`'s
stand-ins."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import numpy as np
import pytest
import torch
import torch.nn as nn
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.definitions.registry import b1, cublas_bf16, dsa, dsa_ds_mla
from verity.definitions.registry.boolean_rope import RoPEHeadGptJV2
from verity.definitions.registry.flashmla_dsv32 import FlashMlaV32Decode

from tests.program import test_dsa_derive as T
from verity_vllm.program.frontend.dsa_leaves import dsa_leaves
from verity_vllm.program.frontend.rules import vllm_bindings as VB
from verity_vllm.program.frontend.rules import vllm_dsa, vllm_dsa_dense, vllm_dsa_fp8
from verity_vllm.program.frontend.rules.vocab import PROFILES

NH, P, R, NQ, H, D, ROPE, EPS, SCALE, DOT, NAMES = T.NH, T.P, T.R, T.NQ, T.H, T.D, T.ROPE, T.EPS, T.SCALE, T.DOT, T.NAMES
L, TILE, TOPK, BN = 256, vllm_dsa_fp8.DS_MLA_TILE, 16, 16
DS_MLA, IMPL = vllm_dsa_fp8.DS_MLA, vllm_dsa_fp8.FLASHMLA_SPARSE_IMPL
STEPS = T.STEPS
LATENT = bind(dsa_ds_mla.DsaLatentRowDsMla, KVL=L, R=R, TILE=TILE, EPS=EPS)


class _KvBProj(nn.Module):
    """`kv_b_proj` as the latent's first `T.LD` columns, the dense layer's k_nope and v per head."""

    def forward(self, x):
        return x[:, :T.LD], None


class _Layer(T._Layer):
    """`test_dsa_derive`'s layer writing an `fp8_ds_mla` cache, at latent L."""

    def __init__(self, name: str, indexer: bool, seed: int, topk: int = TOPK, dense: bool = False) -> None:
        super().__init__(name, indexer, seed, topk=topk, dense=dense)
        rng = np.random.default_rng(seed + 100)
        self.kv_lora_rank, self.kv_cache_dtype = L, DS_MLA
        self.kv_a_layernorm = T._Norm(L, rng)
        self.W_UK_T = nn.Parameter(T._bf16(rng, NH, P, L, sd=0.25), requires_grad=False)
        self.W_UV = nn.Parameter(T._bf16(rng, NH, L, self.v_head_dim, sd=0.25), requires_grad=False)
        if dense:
            self.kv_b_proj = _KvBProj()


def _model(order=(True, False), topk: int = TOPK, steps=STEPS, dense: bool = False) -> T._Model:
    return T._Model([_Layer(NAMES[i], indexer, seed=20 + i, topk=topk, dense=dense) for i, indexer in enumerate(order)], steps)


def _observation(**over) -> dict:
    return T._observation(**{"impl": IMPL, "backend": "FLASHMLA_SPARSE", "kv_cache_dtype": DS_MLA, "layer_kv_cache_dtype": DS_MLA,
                             "kv_lora_rank": L, "num_splits": None, **over})


@pytest.fixture
def served(monkeypatch):
    """The H200 serve as the Build observes it: the target, FlashMLA's sparse implementation over each layer's `fp8_ds_mla` cache,
    and its fp8 decode at the fixture's widths."""
    saved = dict(VB._OBSERVED)
    monkeypatch.setitem(PROFILES, T.STANDIN.name, T.STANDIN)
    monkeypatch.setitem(vllm_dsa_fp8.FLASHMLA_FP8_GEOMETRY, (R, L), {"QT": TILE, "BN": BN, "DOT": DOT})
    VB.observe_target_profile(T.H200)
    VB.observe_attention_impls({n: _observation() for n in NAMES})
    yield
    VB._OBSERVED.clear()
    VB._OBSERVED.update(saved)


@pytest.fixture
def dense(served, monkeypatch):
    """The served layers with FlashAttnPrefillBackend and a decode threshold of 1, as `test_dsa_derive`'s dense fixture."""
    monkeypatch.setitem(vllm_dsa.DECODE_THRESHOLD, NH, 1)
    monkeypatch.setattr(vllm_dsa_dense, "_dense_splits", lambda tp, T, NH, D: 1)
    VB.observe_attention_impls({n: _observation(prefill_backend=T.PREFILL, prefill_fa_version=3, prefill_requires_v_padding=False,
                                                prefill_scale=SCALE) for n in NAMES})


def _decode(tokens: int):
    return bind(FlashMlaV32Decode, NH=NH, SPLITS=vllm_dsa_fp8.decode_splits(tokens, TOPK, int(T.H200.num_sms), BN), DN=L, DR=R,
                QT=TILE, BN=BN, DOT=DOT, SCALE=SCALE)


def _reference(model: T._Model, inputs) -> list[int]:
    """The module's outputs as the Definitions compose them: per token the q norm and `DsaLatentRowDsMla_v1`, whose NoPE bytes,
    scales and RoPE words are its cache row; per sparse row `FlashMlaV32Decode_v1` over the layer's first n cache rows at index 0..
    and row 0's under -1 to TOPK rows, then `W_UV`; on the dense route `test_dsa_derive`'s `Attention_v4` over the call's own bf16
    rows, k_nope and v each head's halves of the latent's first `T.LD` words."""
    qn, rope = bind(b1.RMSNormTriton, N=NQ, EPS=EPS), bind(RoPEHeadGptJV2, D=R)
    hist: dict[str, list[tuple]] = {n: [] for n in NAMES}
    out, k = [], 0
    for p0, M in model.steps:
        for layer in model.layers:
            vd = layer.v_head_dim
            up = bind(cublas_bf16.BatchedGemmWord, B=NH, K=P, N=L, DOT=DOT)
            down = bind(cublas_bf16.BatchedGemmWord, B=NH, K=L, N=vd, DOT=DOT)
            q_c, kv_c, k_pe, q = inputs[k:k + 4]
            k += layer.n_inputs
            cs, qv = layer.rotary_emb.cos_sin_cache, q.view(M, NH, P + R)
            wuk, wuv = T._words(layer.W_UK_T.permute(0, 2, 1)), T._words(layer.W_UV.permute(0, 2, 1))
            h = hist[layer.layer_name]
            dense = T._dense(layer, len(h), M)
            rows, qcs, ks, vs = [], [], [], []
            for i in range(M):
                p = p0 + i
                qcs += list(evaluate(qn, T._words(q_c[i]), T._words(layer.q_a_layernorm.weight)))
                o = list(evaluate(LATENT, T._words(kv_c[i]) + T._words(k_pe[i]), T._words(layer.kv_a_layernorm.weight), T._words(cs[p])))
                kv, kr = o[:L], o[L:L + R]
                h.append((o[L + R:2 * L + R], o[2 * L + R:], kr))
                qr = [list(evaluate(rope, T._words(head), T._words(cs[p]))) for head in qv[i, :, P:]]
                if dense:
                    heads = [kv[j * (P + vd):(j + 1) * (P + vd)] for j in range(NH)]
                    ks += [w for hd in heads for w in hd[:P] + kr]
                    vs += [w for hd in heads for w in hd[P:]]
                    qrow = [w for j in range(NH) for w in T._words(qv[i, j, :P]) + qr[j]]
                    rows += list(evaluate(T._attention_v4(i + 1, i, M), qrow, ks, vs))
                    continue
                ql = list(evaluate(up, T._words(qv[i, :, :P]), wuk))
                n = len(h)
                sel = h + h[:1] * (TOPK - n)
                a = list(evaluate(_decode(M), [w for j in range(NH) for w in ql[j * L:(j + 1) * L] + qr[j]],
                                  *([w for r in sel for w in r[c]] for c in range(3)), [0] * n + [0xFFFFFFFF] * (TOPK - n)))
                rows += list(evaluate(down, a, wuv))
            out += rows + qcs
    return [int(w) for w in out]


def test_the_shim_exports_the_cache_rows_and_flashmlas_decode():
    model = _model()
    with dsa_leaves(T.SITE, model):
        ep = torch.export.export(model, T._inputs(model), strict=False)
    ops = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
    per_step = len(STEPS) * len(NAMES)
    assert ops.count("vllm_iface.dsa_norm_rope_ds_mla.default") == per_step == ops.count("vllm_iface.dsa_flashmla_attn.default")
    assert not [o for o in ops if o in ("vllm_iface.dsa_norm_rope.default", "vllm_iface.dsa_indexer_attn.default")]
    (row,) = [n for n in ep.graph.nodes if str(n.target) == "vllm_iface.dsa_norm_rope_ds_mla.default"][:1]
    nope, scale = row.meta["val"][3:]
    assert (tuple(nope.shape), nope.dtype, tuple(scale.shape), scale.dtype) == ((2, L), torch.float8_e4m3fn, (2, L // TILE), torch.float32)


def test_a_layer_whose_cache_is_not_the_one_its_rows_were_written_in_refuses():
    """`fused_norm_rope` called for a bf16 row (`mla_kv_cache_dtype` "auto") on a layer whose attention reads `fp8_ds_mla`."""
    model = _model()
    with dsa_leaves(T.SITE, model):
        patched = T.SITE.fused_norm_rope
        T.SITE.fused_norm_rope = lambda *a, **k: patched(*a, **{**k, "mla_kv_cache_dtype": "auto"})
        try:
            with pytest.raises(NotImplementedError, match="is not the one its cache rows were written in"):
                torch.export.export(model, T._inputs(model), strict=False)
        finally:
            T.SITE.fused_norm_rope = patched
    assert T.SITE.fused_norm_rope is T._unpatched


def test_an_indexer_and_a_share_layer_derive_over_their_cache_rows(served):
    """A two-row call then a decode: per token the q norm, one `DsaLatentRowDsMla_v1` and on the indexer layer its index-K slot;
    per row the logits and short-row top-k as over the default cache; then two index words and per row `FlashMlaV32Decode_v1` at
    the planner's splits for the call's rows.  The circuit's outputs are the Definitions' composition."""
    model = _model()
    xs = T._inputs(model)
    rep = T._derive(model, xs)
    assert rep.program is not None, rep.unsupported
    norm_q, qr = bind(b1.RMSNormTriton, N=NQ, EPS=EPS).id, bind(RoPEHeadGptJV2, D=R).id
    up = bind(cublas_bf16.BatchedGemmWord, B=NH, K=P, N=L, DOT=DOT).id
    down = bind(cublas_bf16.BatchedGemmWord, B=NH, K=L, N=T.V, DOT=DOT).id
    krow, iq = bind(dsa.DsaIndexerKRow, D=D, ROPE=ROPE, EPS=EPS).id, bind(dsa.DsaIndexerQ, H=H, D=D, ROPE=ROPE).id
    sel = bind(dsa.DsaTopKShortRow, TOPK=TOPK).id
    want = []
    for p0, M in STEPS:
        for indexer in (True, False):
            want += [norm_q] * M + [LATENT.id] * M + ([krow] * M if indexer else []) + [up] * M + [qr] * (M * NH)
            want += [iq] * M if indexer else []
            for p in range(p0, p0 + M) if indexer else ():
                want += [bind(dsa.DsaMqaLogitsV3, N=p + 1, H=H, D=D).id, f"Const32[0x{p + 1:08x}]_v1", sel]
            want += ["Const32[0x00000000]_v1", "Const32[0xffffffff]_v1"] + [_decode(M).id] * M + [down] * M
    assert T._ids(rep) == want
    assert T._run(rep, model, xs) == _reference(model, xs)
    apps = [a["preconditions_checked"] for a in rep.provenance["applications"] if a["rule"] == "DsaFlashMlaAttn[vllm-dsa]_v1"]
    assert [(a["layer_name"], a["indexer"], a["top_k_from"], a["history_rows_after"], a["launch_sparse_rows"], a["splits"])
            for a in apps] == [(NAMES[0], True, NAMES[0], 2, 2, [0, 1]), (NAMES[1], False, NAMES[0], 2, 2, [0, 1]),
                               (NAMES[0], True, NAMES[0], 3, 1, [0, 1]), (NAMES[1], False, NAMES[0], 3, 1, [0, 1])]


def test_the_cache_rows_feed_the_decode(served):
    """The decode at position 2 reads 3 cache rows: layer 0's NoPE, scale and RoPE operands are its 3 `DsaLatentRowDsMla_v1` Calls'
    (the first call's 2, then its own), row 0's again to TOPK rows, and its index operand 3 words of 0 then -1."""
    model = _model()
    rep = T._derive(model, T._inputs(model))
    nodes = rep.program.fn.body.nodes
    rows = [i for i, n in enumerate(nodes) if n.fn.id == LATENT.id]
    dec = [n for n in nodes if n.fn.id.startswith("FlashMlaV32Decode_v1{")][2 * len(NAMES)]
    _, nope, scale, rope, idx = (a.refs for a in dec.args)
    layer0 = [("n", k) for k in (rows[0], rows[1], rows[2 * len(NAMES)])]
    for refs, width in ((nope, L), (scale, L // TILE), (rope, R)):
        assert [refs[j * width][:2] for j in range(TOPK)] == layer0 + layer0[:1] * (TOPK - 3)
    assert [nodes[idx[j][1]].fn.id for j in range(TOPK)] == ["Const32[0x00000000]_v1"] * 3 + ["Const32[0xffffffff]_v1"] * (TOPK - 3)


def test_the_splits_are_the_planners_for_the_launchs_sparse_rows():
    """The planner cuts a call of s_q rows at 64 padded heads over max(132 / s_q, 1) partitions of GLM-5.3's 32 blocks: a
    partition's payload is ceil((32 + 5) / partitions) blocks, the 5 its fixed overhead, so 8 rows' 16 partitions take 3 each."""
    assert vllm_dsa_fp8.decode_splits(1, 2048, 132, 64) == tuple(range(33))
    assert vllm_dsa_fp8.decode_splits(8, 2048, 132, 64) == (*range(0, 32, 3), 32)
    assert vllm_dsa_fp8.decode_splits(132, 2048, 132, 64) == vllm_dsa_fp8.decode_splits(200, 2048, 132, 64) == (0, 32)


def test_a_dense_prefill_opens_the_history_the_decode_reads(dense):
    """The two-row prefill by dense MHA (`Attention_v4` per row and head over the call's own bf16 rows, no logits), then the decode
    by `FlashMlaV32Decode_v1` over the prefill's cache rows and its own."""
    model = _model(dense=True)
    xs = T._inputs(model)
    rep = T._derive(model, xs)
    assert rep.program is not None, rep.unsupported
    ids = T._ids(rep)
    assert ids.count(LATENT.id) == 3 * len(NAMES) and ids.count(_decode(1).id) == len(NAMES)
    assert [i for i in ids if i.startswith("Attention_v4{")] == [T._attention_v4(i + 1, i, 2, heads=1).id for i in range(2)
                                                                 for _ in range(NH)] * len(NAMES)
    assert not [i for i in ids if i.startswith(("SparseMla", "DsaLatentRow_v1"))]
    assert T._run(rep, model, xs) == _reference(model, xs)
    apps = [a["preconditions_checked"] for a in rep.provenance["applications"] if a["rule"] == "DsaDenseMhaDsMla[vllm-dsa]_v1"]
    assert [(a["layer_name"], a["history_rows_after"]) for a in apps] == [(n, 2) for n in NAMES]


@pytest.mark.parametrize("case,needle", [
    ("impl", "FlashMlaV32Decode_v1 states vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseImpl"),
    ("cache", "FlashMlaV32Decode_v1 reads fp8_ds_mla rows"),
    ("heads", "FlashMLA's mixed batch serves fewer than 32"),
    ("cp", "runs under context parallelism"),
    ("geometry", "FlashMLA's sparse fp8 decode at rope 16, latent 256 is not registered"),
    ("blocks", "TOPK 4 is not whole blocks of FlashMLA's 16 rows"),
    ("past-topk", "a row of 17 keys, past TOPK (16), over an fp8_ds_mla cache"),
])
def test_what_the_rules_do_not_state_refuses_by_name(case, needle, served, monkeypatch):
    model = _model(topk=4) if case == "blocks" else _model()
    over = {"impl": {"impl": T.IMPL}, "cache": {"kv_cache_dtype": "fp8"}, "heads": {"q_heads_per_rank": 32},
            "cp": {"dcp_world_size": 2}}.get(case)
    if over:
        VB.observe_attention_impls({n: _observation(**over) for n in NAMES})
    if case == "geometry":
        monkeypatch.delitem(vllm_dsa_fp8.FLASHMLA_FP8_GEOMETRY, (R, L))
    if case == "past-topk":
        monkeypatch.setattr(T, "MAXPOS", 32)
        model = _model(steps=((0, 16), (16, 1)))
    hint = T._hint(T._derive(model, T._inputs(model)))
    assert needle in hint and "DsaFlashMlaAttn[vllm-dsa]_v1" in hint, hint


def test_a_row_whose_definition_does_not_take_its_widths_refuses_by_name(served, monkeypatch):
    """A tile the row is not whole tiles of: `DsaLatentRowDsMla_v1`'s own statics check, by the norm's rule."""
    monkeypatch.setattr(vllm_dsa_fp8, "DS_MLA_TILE", 96)
    model = _model()
    hint = T._hint(T._derive(model, T._inputs(model)))
    assert "a tile is a whole warp's elements" in hint and "DsaNormRopeDsMla[vllm-dsa]_v1" in hint, hint
