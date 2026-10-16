"""A GLM-5.3 tensor-parallel rank's Match on CPU in seconds: a two-layer capture log in the served shape (rank 0 of 8, cc 9.0, the
fused norm/RoPE Triton launch and the CUDA norms carrying no module, FlashInfer's fused all-reduce norm, the MLA absorb `bmm` over the
strided W_UV view at 131,072 runs a read) folded with the GLM53 profile and persisted in both stored forms, then loaded and compared as
`tp-fold-match` does:

* the Match's weight map has no overlapping fields and names every weight read by its checkpoint Root, each layer apart;
* `compare_steps` in progressions form (`--operands progressions`) reaches the runs form's verdict, equal and on a changed W_UV window,
  without expanding the strided weight reads (hash, canon, load or GM-01's trie).
"""
from __future__ import annotations

import collections
import itertools
import json
from pathlib import Path

import pytest

from tests.observe.test_fold_glm53_defaults import BMM_OUT_SCHEMA, CUDA, FAR_SCHEMA, H200, view
from verity_vllm.check.match import global_match_fast as GF
from verity_vllm.check.match import program_compare as CMP
from verity_vllm.engine.profiles import generic as G
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepEnd, StepStart, StorageAlloc, TritonLaunch, ViewDesc
from verity_vllm.observe.fold.fold import fold
from verity_vllm.observe.fold.patterns import dsa as DSA
from verity_vllm.pipeline.match import INSTANCES_FORM_ENV, persist_program
from verity_vllm.program import compact as K

CFG = json.loads((Path(__file__).parent / "fixtures" / "glm_moe_dsa" / "config.json").read_text())
LAYERS, PROMPT = (0, 1), [5, 7]
M = len(PROMPT)
#: the absorb reads 64 latent rows of W_UV: 8 x 256 x 64 = 131,072 unit runs, past compact.HASH_RLE_RUNS
KUV = 64
UV_RUNS = 8 * 256 * KUV
SIZE = {"bfloat16": 2, "float32": 4, "int32": 4, "int64": 8, "uint8": 1}
RMS_SCHEMA = "_C::rms_norm(Tensor($0! -> ) result, Tensor input, Tensor? weight, float epsilon) -> ()"
ADD_SCHEMA = "_C::fused_add_rms_norm(Tensor($0! -> ) input, Tensor($1! -> ) residual, Tensor? weight, float epsilon) -> ()"
NORM_ROPE_PINS = {"Q_DIM": 2048, "Q_BLOCK_SIZE": 2048, "KV_DIM": 512, "KPE_HALF_ROT_DIM": 32, "MLA_CACHE_BLOCK_SIZE": 64,
                  "MLA_CACHE_FP8": False, "MLA_CACHE_DS_MLA": False, "MLA_NUM_TILES": 1, "MLA_TILE_DIM": 1, "TOPK": 2048,
                  "TOPK_BLOCK_SIZE": 1024, "INDEX_ROPE_INTERLEAVE": True, "HAS_INDEXER": True, "INDEX_K_DIM": 128,
                  "INDEX_K_BLOCK_SIZE": 128, "INDEX_K_HALF_ROT_DIM": 32}


@pytest.fixture(scope="module")
def profile():
    return G.build_profile("GLM53", 8, hf_config=CFG, repo="zai-org/GLM-5.3-FP8", revision="aca966e4")


def _log(profile, crossed: bool = False) -> list:
    """One prefill step of rank 0: embedding, then per layer the input norm, the fused norm/RoPE launch (q_a / kv_a norms, k_pe, index-K)
    and the W_UV absorb, then the post-attention norm; the residual norms alternate between FlashInfer's fused all-reduce (module = the
    decoder layer) and the moduleless CUDA fused add.  `crossed`: layer 1's absorb reads layer 0's W_UV."""
    C = profile.config
    H, VS, EPS = C["H"], C["V"], C["EPS"]
    ids_ = itertools.count(1)
    ev = [RunHeader("verity-capture/log/v1", "2026-10-16T00:00:00Z", "glm53-rank-rehearsal",
                    {"gpu": {"name": "NVIDIA H200", "compute_capability": [9, 0], "multi_processor_count": 132,
                             "tuned_gemm_arch_family": "hopper"}, "backends": {}, "target": H200.to_json()},
                    {"repo": "zai-org/GLM-5.3-FP8", "config": {}},
                    {"requests": [{"request_id": "r0", "prompt_token_ids": PROMPT, "max_tokens": 1}], "sampling": {"temperature": 0.0}},
                    notes={"tp2": {"rank": 0, "world_size": 8}})]

    def alloc(shape, dtype="bfloat16", origin="op", nbytes=None):
        v = view(next(ids_), shape, dtype)
        ev.append(StorageAlloc(v.alloc, 0x100000 * v.alloc, nbytes or v.numel * SIZE[dtype], CUDA, 0, origin))
        return v

    def root(name, shape, dtype="bfloat16"):
        v = alloc(shape, dtype, "param")
        ev.append(Root(v, "weight", name))
        return v

    embed = root("model.embed_tokens.weight", (VS, H))
    cs = root("model.layers.0.self_attn.rotary_emb.cos_sin_cache", (C["MAX_POS"], 64))
    W = {}
    for i in LAYERS:
        p = f"model.layers.{i}"
        W[i] = {"ln1": root(f"{p}.input_layernorm.weight", (H,)), "ln2": root(f"{p}.post_attention_layernorm.weight", (H,)),
                "q_a": root(f"{p}.self_attn.q_a_layernorm.weight", (2048,)), "kv_a": root(f"{p}.self_attn.kv_a_layernorm.weight", (512,)),
                "k_w": root(f"{p}.self_attn.indexer.k_norm.weight", (128,), "float32"),
                "k_b": root(f"{p}.self_attn.indexer.k_norm.bias", (128,), "float32")}
        kv_b = alloc((512, 8 * 448), origin="param")
        W[i]["w_uv"] = view(kv_b.alloc, (8, 512, 256), strides=(448, 3584, 1), offset=192)
        ev += [Root(view(kv_b.alloc, (8, 192, 512), strides=(448, 1, 3584)), "weight", f"{p}.self_attn.W_UK_T"),
               Root(W[i]["w_uv"], "weight", f"{p}.self_attn.W_UV")]
    norm = root("model.norm.weight", (H,))
    ids, pos, topk = alloc((M,), "int32"), alloc((M,), "int64"), alloc((16, 2048), "int32")
    caches = {}
    for i in LAYERS:
        ic = alloc((2, 64, 132), "uint8")
        caches[i] = (alloc((2, 64, 576)), ic, view(ic.alloc, (2, 64, 33), "float32"))
    ev.append(StepStart(0, [("r0", t) for t in range(M)], PROMPT, list(range(M)), {"r0": M}, {"r0": 0}, {"r0": [0]}, 64, list(range(M)),
                        [("r0", M - 1)], [M - 1], 1, M, M, ids, pos, "prefill"))
    ev.append(Root(ids, "prompt_token", "prompt[r0]", PROMPT))
    eid = itertools.count(100)

    def dispatch(op, schema, args, kwargs=None, module=""):
        k = next(eid)
        ev.extend([DispatchEnter(k, None, 0, "forward", op, schema, list(args), kwargs or {}, 7, module), DispatchExit(k, None)])

    def residual_norm(site, x, res, w, i):
        if site % 2:
            h = alloc((M, H))
            dispatch("vllm.flashinfer_trtllm_fused_allreduce_norm.default", FAR_SCHEMA, [x, res, w, EPS, 8, True, True, 42, 1, h],
                     module=f"model.layers.{i}")
            return h, x
        dispatch("_C.fused_add_rms_norm.default", ADD_SCHEMA, [x, res, w, EPS])
        return x, res

    def attention(i, x):
        lat, ikc, isc = caches[i]
        col = lambda lo, w: ViewDesc(x.alloc, x.offset + lo, (M, w), (H, 1), "bfloat16", CUDA)  # noqa: E731
        qo = alloc((M, 2048))
        args = {"pos_ptr": pos, "q_c_ptr": col(0, 2048), "q_c_out_ptr": qo, "q_rms_norm_w_ptr": W[i]["q_a"], "q_rms_eps": 1e-6,
                "kv_ptr": col(2048, 512), "kv_rms_norm_w_ptr": W[i]["kv_a"], "kv_rms_eps": 1e-6, "kpe_ptr": col(2560, 64),
                "kpe_rope_cos_sin_cache_ptr": cs, "mla_cache_ptr": lat, "index_k_ptr": col(2624, 128),
                "index_k_layer_norm_w_ptr": W[i]["k_w"], "index_k_layer_norm_bias_ptr": W[i]["k_b"], "index_k_layer_norm_eps": 1e-6,
                "index_k_rope_cos_sin_cache_ptr": cs, "indexer_cache_ptr": ikc, "indexer_cache_scale_ptr": isc, "topk_indices_ptr": topk}
        ev.append(TritonLaunch(next(eid), None, 0, "forward", f"vllm.v1.attention.ops.glm.{DSA.NORM_KERNEL}", DSA.NORM_SRC_SHA,
                               {**args, **NORM_ROPE_PINS}, NORM_ROPE_PINS, [M, 1, 1], 4, 3, 7, None))
        w = W[0 if crossed else i]["w_uv"]
        dispatch("aten.bmm.out", BMM_OUT_SCHEMA, [ViewDesc(qo.alloc, 0, (8, M, KUV), (64, 2048, 1), "bfloat16", CUDA),
                                                  ViewDesc(w.alloc, w.offset, (8, KUV, 256), w.strides, "bfloat16", CUDA)],
                 kwargs={"out": ViewDesc(alloc((M, 8 * 256)).alloc, 0, (8, M, 256), (256, 2048, 1), "bfloat16", CUDA)})

    emb, x = alloc((M, H)), alloc((M, H))
    dispatch("_C.vocab_parallel_embedding.default", "_C.vocab_parallel_embedding(...)", [emb, ids, embed, 0, VS, 0, VS, VS],
             module="model.embed_tokens")
    dispatch("_C.rms_norm.default", RMS_SCHEMA, [x, emb, W[0]["ln1"], EPS])
    res, site = emb, 1
    for i in LAYERS:
        if i:
            x, res = residual_norm(site, x, res, W[i]["ln1"], i)
            site += 1
        attention(i, x)
        x, res = residual_norm(site, x, res, W[i]["ln2"], i)
        site += 1
    dispatch("_C.rms_norm.default", RMS_SCHEMA, [alloc((M, H)), x, norm, EPS])
    ev.append(StepEnd(0))
    return ev


@pytest.fixture(scope="module")
def folds(profile, tmp_path_factory):
    """{(log, stored form): fold dir}, for the served log ("served") and the one whose layer 1 reads layer 0's W_UV ("crossed")."""
    out = {}
    for which in ("served", "crossed"):
        r = fold(_log(profile, which == "crossed"), profile, strict=False, with_program=True)
        assert not r.report.unsupported, r.report.unsupported
        for form in ("runs", "progressions"):
            d = tmp_path_factory.mktemp(f"{which}-{form}")
            with pytest.MonkeyPatch.context() as mp:
                mp.setenv(INSTANCES_FORM_ENV, form)
                persist_program(r, d, profile_name=profile.id, log_path="log.jsonl.gz")
            out[which, form] = str(d)
    return out


@pytest.fixture(scope="module")
def loaded(folds):
    """Each fold dir loaded as tp-fold-match's compare leg loads it: compact for the progressions form, runs for the record form."""
    return {k: CMP.load_oracle(d, f"{k[0]}-{k[1]}", compact=k[1] == "progressions") for k, d in folds.items()}


def _root_names(profile) -> set[str]:
    return {e.name for e in _log(profile) if isinstance(e, Root) and e.role == "weight"}


@pytest.mark.parametrize("form", ["runs", "progressions"])
def test_a_moduleless_rank_fold_maps_each_layers_weights_apart(loaded, profile, form):
    """[04f0af376] Without Root names the moduleless launches hinted every layer's q_a / kv_a / index-K norm and the CUDA norms alike:
    the weight map unioned them into ranges spanning the model, overlapping (GM-01 C5) and sending every weight read per run."""
    P = loaded["served", form]
    assert P.oracle_wmap and not P.oracle_wmap_overlaps, P.oracle_wmap_overlaps[:4]
    names = [nm for _lo, _hi, nm in P.oracle_wmap]
    assert len(names) == len(set(names))
    for i in LAYERS:
        p = f"model.layers.{i}"
        assert {f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight", f"{p}.self_attn.q_a_layernorm.weight",
                f"{p}.self_attn.kv_a_layernorm.weight", f"{p}.self_attn.indexer.k_norm.weight", f"{p}.self_attn.indexer.k_norm.bias",
                f"{p}.self_attn.W_UV"} <= set(names), names
    assert set(names) <= _root_names(profile), sorted(set(names) - _root_names(profile))


def _verdict(res: dict) -> tuple:
    t = res["totals"]
    return (t.get("steps_spec_equal"), t.get("operands_equal"), t.get("operands_different"), res["canonical_sha256"]["equal"],
            sorted(collections.Counter(k for s in res["steps"] for k in (s.get("wiring") or {}).get("different_by_spec_arg", {})).items()))


def test_the_progressions_compare_reaches_the_runs_verdict(loaded):
    """`--operands progressions` (both sides compact, stored as progressions) against the record form: the same verdict on the served
    fold against itself and against the fold whose layer 1 reads layer 0's W_UV (each of its token rows' W_UV operand, and only it,
    differs)."""
    same = {f: CMP.compare_steps(loaded["served", f], loaded["served", f]) for f in ("runs", "progressions")}
    crossed = {f: CMP.compare_steps(loaded["served", f], loaded["crossed", f]) for f in ("runs", "progressions")}
    assert _verdict(same["runs"]) == _verdict(same["progressions"])
    assert same["runs"]["totals"]["operands_different"] == 0 and same["runs"]["canonical_sha256"]["equal"]
    assert _verdict(crossed["runs"]) == _verdict(crossed["progressions"])
    assert crossed["runs"]["totals"]["operands_different"] == M and not crossed["runs"]["canonical_sha256"]["equal"]
    (key,) = [k for k, _n in _verdict(crossed["progressions"])[-1]]
    assert key.startswith("BatchedGemm_v1{B=8,K=64,N=256,") and key.endswith("#arg1"), key


def test_the_progressions_compare_never_expands_a_strided_weight_read(loaded, monkeypatch):
    """[8c01ad3dc, 04f0af376] The compact compare hashes and canons a W_UV read as its progressions: neither `_canonical_hash` nor the
    weight map's per-run fallback (an overlapping map) expands its 131,072 runs (the record-form leg held ~1e6 run tuples a read)."""
    P = loaded["served", "progressions"]
    expanded, expand_into = [0], K._expand_into

    def counting(out, head, base, delta, c):
        expanded[0] += c
        return expand_into(out, head, base, delta, c)

    monkeypatch.setattr(K, "_expand_into", counting)
    res = CMP.compare_steps(P, P)
    assert res["canonical_sha256"]["equal"] and res["totals"]["operands_different"] == 0
    assert expanded[0] < UV_RUNS, expanded[0]


def test_the_compact_load_splices_each_aliased_weight_read(folds, monkeypatch):
    """[799e16bad] Each token row's W_UV read after the first is an alias of the first: the compact load splices it in one list copy,
    so `_Acc.progression` runs once per progression of each layer's first read, not again for every alias (51 % of GM-01's fold load)."""
    calls, progression = [0], K._Acc.progression

    def counting(self, *a):
        calls[0] += 1
        return progression(self, *a)

    monkeypatch.setattr(K._Acc, "progression", counting)
    P = CMP.load_oracle(folds["served", "progressions"], "served", compact=True)
    uv = [i for i, x in enumerate(P.inst) if x["spec"].startswith("BatchedGemm")]
    assert len(uv) == len(LAYERS) * M and all(K.n_runs(P.inst[i]["args"][1]) == UV_RUNS for i in uv)
    first_reads = sum(len(P.inst[i]["args"][1]) for i in uv[::M])
    assert calls[0] <= first_reads, (calls[0], first_reads)


def test_the_dag_hashes_take_one_trie_step_per_weight_progression(loaded):
    """[8dec947cf] GM-01's fast DAG hashes walk a compact W_UV read in one trie step per progression (~2048), not per run (131,072 a
    read, ~5e9 steps a rank), and partition the instances as the runs-form load does."""
    try:
        GF.reset()
        h = GF._dag_hashes_intern(loaded["served", "progressions"])
        trie = len(GF._trie)
        GF.reset()
        r = GF._dag_hashes_intern(loaded["served", "runs"])
    finally:
        GF.reset()
    assert trie < UV_RUNS, trie

    def partition(hs):
        by = collections.defaultdict(list)
        for i, v in hs.items():
            by[v].append(i)
        return sorted(by.values())

    assert partition(h) == partition(r) and len(h) == len(r)
