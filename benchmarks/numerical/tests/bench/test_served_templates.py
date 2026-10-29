"""The templates of every served workload beyond #101 (``bench.templates``): FA3 attention, greedy token select, embedding row, the
MoE router (with and without renormalisation), expert coordinates (plain and weighted), MoE sum, the FP8 block-scaled GEMM, bias-add,
the bf16 scalar multiply, the served SiLU·mul (SiluMul_v2) and RMSNorms (the width-8 CUDA kernels), and the registered compiler's
dead-free RMSNormTriton_v5 and AttentionHead_v18.  Each evaluation path equals its subcircuit's own Definition, and every
census-bound subcircuit id of every workload is the spine's."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity_numerical.bench import census
from verity_numerical.bench import generate as G
from verity_numerical.bench import input_sets as IS
from verity_numerical.bench import templates as TM

pytest.importorskip("verity_vllm", reason="these templates evaluate through the vLLM integration's Definitions")


def _draws(sub: TM.Subcircuit, n: int, seed: int = 0) -> list[TM.Draw]:
    key = G.key_for(sub, seed)
    return [sub.draw(key, j) for j in range(n)]


def _eval(sub: TM.Subcircuit, draws: list[TM.Draw]) -> dict[str, list[np.ndarray]]:
    return sub.evaluate({p.name: [d.inputs[p.name] for d in draws] for p in sub.inputs}, [d.instance_statics for d in draws])


def _bound(mix: dict) -> str | None:
    """The census subcircuit a workload mix entry binds: its template at exactly its parameters."""
    return next((s.id for s in census.SUBCIRCUITS.values()
                 if s.template == mix["subcircuit_template"] and vars(s.parameters) == (mix["subcircuit_parameters"] or {})), None)


def test_every_bound_subcircuit_id_is_the_spines():
    for w in census.raw("workloads"):
        for m in w["template_mix"]:
            sid = _bound(m)
            sub = TM.parse_subcircuit_id(sid)
            assert (sub.id, sub.display) == (sid, census.SUBCIRCUITS[sid].display)


def test_fa3_head_equals_its_definition():
    from verity.ml.circuits.evaluation import evaluate_batch

    sub = TM.subcircuit("attention-head", D=16, BN=16, T="17:40", semantics="sm90.fa3.bf16")
    assert sub.relation["definition"] == "Attention_v2" and sub.set_name == "attention-head-hopper-bf16-wgmma-fa3invsum_v1-d16-bn16"
    for d in _draws(sub, 4):
        ref = evaluate_batch(sub.definition(**d.instance_statics), *[d.inputs[p.name][None, :] for p in sub.inputs], kernel="reference")
        assert np.array_equal(ref.reshape(-1), _eval(sub, [d])["out"][0].astype(ref.dtype))
    assert TM.identify(sub.relation).id == sub.id == TM.parse_subcircuit_id(sub.id).id


@pytest.mark.parametrize("weighted", [False, True])
def test_expert_coordinate_equals_its_definition(weighted):
    """The slab path (MoeExpertGemm(W)_v1{N=1}) equals MoeExpertCoordinate(W)_v1's gate reference on the same slab."""
    from verity.ml.circuits.evaluation import evaluate_batch

    sub = TM.subcircuit("moe-expert-coordinate-weighted" if weighted else "moe-expert-coordinate", E=3, K=32)
    draws = _draws(sub, 6)
    got = _eval(sub, draws)["y"]
    for d, y in zip(draws, got):
        slab = np.zeros((1, 3, 32), np.uint16)
        slab[0, int(d.inputs["e"][0])] = d.inputs["w"]
        args = [d.inputs["x"][None], d.inputs["e"][None], *([d.inputs["g"][None]] if weighted else []), slab]
        assert int(evaluate_batch(sub.definition(), *args, kernel="reference").reshape(-1)[0]) == int(y[0])
    if not weighted:
        gemm = TM.subcircuit("gemm-coordinate", K=32)
        want = gemm.evaluate({"x": [d.inputs["x"] for d in draws], "w": [d.inputs["w"] for d in draws]})["y"]
        assert [int(a[0]) for a in got] == [int(b[0]) for b in want]            # GemmCoordinate on the routed expert's row


def test_embedding_equals_its_definition():
    from verity.ml.circuits.evaluation import evaluate_batch

    sub = TM.subcircuit("embedding-row", V=40, H=6)
    draws = _draws(sub, 8)
    out = _eval(sub, draws)["out"]
    for d, o in zip(draws, out):
        table = np.zeros((1, 40, 6), np.uint16)
        table[0, int(d.inputs["tok"][0])] = d.inputs["row"]
        assert np.array_equal(o, d.inputs["row"])
        assert np.array_equal(evaluate_batch(sub.target(), d.inputs["tok"][None], table).reshape(-1), o)
        assert np.array_equal(evaluate_batch(sub.target(), d.inputs["tok"][None], table, kernel="reference").reshape(-1), o)


def test_silu_mul_v2_equals_its_definition():
    """The served SiLU·mul's evaluation path (`SiluMul_v2`'s registered row kernel) equals the Definition's gate reference."""
    from verity.ml.circuits.evaluation import evaluate_batch

    sub = TM.subcircuit("silu-mul-v2", I=32)
    assert sub.relation["definition"] == "SiluMul_v2" and sub.set_name == "silumul-v2-i32"
    draws = _draws(sub, 8)
    ref = evaluate_batch(sub.definition(), np.stack([d.inputs["in0"] for d in draws]), kernel="reference")
    assert np.array_equal(np.asarray(ref).reshape(8, 32), np.stack(_eval(sub, draws)["out.out"]).astype(np.asarray(ref).dtype))
    assert TM.identify(sub.relation).id == sub.id == TM.parse_subcircuit_id(sub.id).id


@pytest.mark.parametrize("template,family", [("rmsnorm-fused-cuda-vec8", "RMSNormFusedCudaVector8_v1"),
                                             ("rmsnorm-cuda-vec8", "RMSNormCudaVector8_v1")])
def test_vec8_rmsnorm_equals_its_definition(template, family):
    """The served RMSNorms' evaluation paths (vLLM's width-8 CUDA kernels) equal the Definitions' gate reference, at a launch
    whose threads hold two vectors each; BLOCK is in the subcircuit id."""
    from verity.ml.circuits.evaluation import evaluate_batch

    sub = TM.subcircuit(template, N=64, EPS=1e-5, BLOCK=4)
    assert sub.relation["definition"] == family and sub.id == f"{template}/n64-b4-eps1e-05/bf16"
    draws = _draws(sub, 8)
    ref = np.asarray(evaluate_batch(sub.definition(), *[np.stack([d.inputs[p.name] for d in draws]) for p in sub.inputs],
                                    kernel="reference")).reshape(8, -1)
    got = _eval(sub, draws)
    assert np.array_equal(ref, np.concatenate([np.stack(got[p.name]) for p in sub.outputs], axis=1).astype(ref.dtype))
    assert TM.identify(sub.relation).id == sub.id == TM.parse_subcircuit_id(sub.id).id


def test_rmsnorm_triton_v5_equals_its_definition():
    """The dead-free RMSNormTriton_v5's evaluation path (its registered row kernel) equals its Definition's gate reference."""
    from verity.ml.circuits.evaluation import evaluate_batch

    sub = TM.subcircuit("rmsnorm-triton-v5", N=128, EPS=1e-6)
    assert sub.relation["definition"] == "RMSNormTriton_v5" and sub.id == "rmsnorm-triton-v5/n128-eps1e-06/bf16"
    assert sub.set_name.startswith("rmsnormtriton-v5-n128-")
    draws = _draws(sub, 4)
    ref = evaluate_batch(sub.definition(), *[np.stack([d.inputs[p.name] for d in draws]) for p in sub.inputs], kernel="reference")
    assert np.array_equal(np.asarray(ref).reshape(4, 128), np.stack(_eval(sub, draws)["out.out"]).astype(np.asarray(ref).dtype))
    assert TM.identify(sub.relation).id == sub.id == TM.parse_subcircuit_id(sub.id).id


def test_v18_head_equals_its_definition_at_the_decode_launch():
    """AttentionHead_v18's evaluation path (Attention_v18{NH=1, KVH=1}'s row kernel) equals the head's gate reference, at the
    MASKED_FROM FA2's decode-only launch guards (its last block of keys, 0-based: ceil(T/BN) - 1)."""
    from verity.ml.circuits.evaluation import evaluate_batch

    sub = TM.subcircuit("attention-head-v18", D=16, BN=16, T="17:40")
    assert sub.relation["definition"] == "Attention_v18" and sub.set_name == "attention-head-v18-decode-d16-bn16"
    for d in _draws(sub, 4):
        T = d.instance_statics["T"]
        defn = sub.definition(**d.instance_statics)
        assert 17 <= T <= 40 and defn.bindings["MASKED_FROM"] == -(-T // 16) - 1
        ref = evaluate_batch(defn, *[d.inputs[p.name][None, :] for p in sub.inputs], kernel="reference")
        assert np.array_equal(ref.reshape(-1), _eval(sub, [d])["out"][0].astype(ref.dtype))
    assert TM.identify(sub.relation).id == sub.id == TM.parse_subcircuit_id(sub.id).id
    with pytest.raises(KeyError):
        TM.identify(dict(sub.relation, masked_from="causal"))


def test_greedy_token_select_is_the_first_argmax():
    sub = TM.subcircuit("token-select-greedy", V=64)
    draws = _draws(sub, 16)
    for d, t in zip(draws, _eval(sub, draws)["out.sampled_token_ids"]):
        v = (d.inputs["in0"].astype(np.uint32) << 16).view(np.float32)
        assert int(t[0]) == int(np.argmax(v))


@pytest.mark.parametrize("norm", [False, True])
def test_router_picks_distinct_experts_with_positive_weights(norm):
    sub = TM.subcircuit("moe-router-topk-norm" if norm else "moe-router-topk", E=16, TOPK=4, VPT=8)
    out = _eval(sub, _draws(sub, 16))
    for w, ids in zip(out["out.router_w"], out["out.router_ids"]):
        wf = np.asarray(w, np.uint32).view(np.float32)
        assert len(set(ids.tolist())) == 4 and all(0 <= i < 16 for i in ids) and (wf > 0).all()
        if norm:
            assert abs(float(wf.sum()) - 1.0) < 1e-5


def test_fp8_block_coordinate_reads_its_scales():
    sub = TM.subcircuit("scaled-mm-fp8-block-coordinate", K=256, G=128)
    draws = _draws(sub, 8)
    y = [int(a[0]) for a in _eval(sub, draws)["y"]]
    assert len(set(y)) > 4 and all((v & 0x7F80) != 0x7F80 for v in y)
    doubled = [TM.Draw(dict(d.inputs, sw=(d.inputs["sw"] + np.uint32(1 << 23)).astype(np.uint32)), {}, {}) for d in draws]
    y2 = [int(a[0]) for a in _eval(sub, doubled)["y"]]
    f = lambda w: float((np.uint32(w) << 16).view(np.float32))  # noqa: E731
    assert all(abs(f(b) - 2 * f(a)) <= abs(f(a)) * 2 ** -6 for a, b in zip(y, y2))    # sw x 2 doubles y (up to bf16 rounding)
    assert not ((draws[0].inputs["xq"] & 0x7F) == 0x7F).any()
    assert all(np.array_equal(draws[i].inputs["xq"], draws[0].inputs["xq"]) for i in range(8))


def test_scale_words_are_pinned():
    r = np.array([0, 0x7FFFFF, 0x00800000, 0xFFFFFFFF], dtype=np.uint32)
    got = TM.scale_words(r)
    assert got.tolist() == [107 << 23, (107 << 23) | 0x7FFFFF, 108 << 23, ((107 + (0x1FF % 19)) << 23) | 0x7FFFFF]
    f = got.view(np.float32)
    assert (f > 0).all() and (f < 1).all()


def test_workloads_suite(tmp_path):
    suite = G.workload_suite((39, 57, 60, 67, 73, 74))
    sids = {TM.subcircuit(t, **p).id for t, p, _n in suite}
    want = {_bound(m) for w in census.raw("workloads") if (w.get("source") or {}).get("row") in (39, 57, 60, 67, 73, 74)
            for m in w["template_mix"]}
    assert sids == want and len(suite) == len(sids) == 45
    fa3 = next(p for t, p, _n in suite if p.get("semantics") == "sm90.fa3.bf16")
    rng = census.SUBCIRCUITS["attention-head/d128-bn128/sm90-fa3-bf16"].variable_ranges.T
    assert fa3["T"] == f"{rng.min}:{rng.max}"


def test_a_served_set_round_trips(tmp_path):
    s = IS.InputSet.open(G.generate(TM.parse_subcircuit_id("moe-router-topk/e64-topk8-vpt8/bf16"), 4, 1, tmp_path)["path"])
    assert IS.verify(s.path)["ok"] and s.subcircuit.id == "moe-router-topk/e64-topk8-vpt8/bf16"
    assert json.loads((s.path / "manifest.json").read_text())["relation"]["definition"] == "MoeRouterTopK_v1"
