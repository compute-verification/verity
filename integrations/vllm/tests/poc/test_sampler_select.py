"""CPU-only tests for W3's sampler.select relation, workload generator and stopping relations.

Run: python -m pytest tests/poc/test_sampler_select.py -q   (numpy only; no torch/vllm)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verity_vllm_sampler import select as S  # noqa: E402
from verity_vllm_sampler import stopping as ST  # noqa: E402
from verity_vllm_sampler import workload as W  # noqa: E402

# Values produced by the installed Triton 3.7.1 kernel on the RTX 4090 pod
# (fixtures/results/w3-sampler/validate_rng.json): tl.randint(seed, pos) with int64 pos.
GPU_GUMBEL_SEEDS = {
    (1, 31): 3177633084,
    (2, 127): 1499188156,
    (3, 2047): 1538132466,
    (1, 0): 3823634032,
    (12345678901234, 5): 200727609,
    (-7, 9): 688642594,
    (2**62 + 3, 100000): 310714282,
}


@pytest.mark.parametrize("seed,pos", list(GPU_GUMBEL_SEEDS))
def test_philox_matches_triton_gpu_values(seed, pos):
    assert S.gumbel_seed_for(seed, pos) == GPU_GUMBEL_SEEDS[(seed, pos)]


def test_uniform_mapping_range_and_clamp():
    u = S.tl_rand32(3177633084, np.arange(4096, dtype=np.int32))
    assert u.dtype == np.float32
    assert np.all(u >= 0.0) and np.all(u < 1.0)
    # the clamp only lifts exact zeros/tiny values, never lowers
    assert np.all(np.maximum(u, S.TL_RAND_MIN) >= u)
    # scale constant is the fp32 just below 2^-31
    assert S.RAND_SCALE.view(np.uint32) == 0x2FFFFFFF


def test_noise_is_pure_function_of_seed_pos_token():
    a = S.gumbel_noise(1, 31, 1000)
    b = S.gumbel_noise(1, 31, 1000)
    c = S.gumbel_noise(1, 32, 1000)
    d = S.gumbel_noise(2, 31, 1000)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c) and not np.array_equal(a, d)
    # prefix property: token v's noise does not depend on vocab size
    assert np.array_equal(S.gumbel_noise(1, 31, 500), a[:500])


def test_greedy_first_index_tie_rule():
    x = np.array([1.0, 5.0, 5.0, 2.0], dtype=np.float32)
    r = S.greedy_select(x)
    assert r.predicted == 1 and r.details["ties_at_max"] == 2 and r.margin == 0.0


def test_bf16_bytes_roundtrip():
    x = np.array([1.5, -2.25, 1e-3, 3e4], dtype=np.float32)
    bf = (x.view(np.uint32) >> 16).astype(np.uint16)  # truncation to bf16 (exact for these)
    y = S.logits_from_bytes(bf.tobytes(), "bfloat16")
    assert y.dtype == np.float32
    assert np.array_equal(y, ((bf.astype(np.uint32) << 16).view(np.float32)))


def test_effective_params_normalisation():
    assert S.effective_params(0.0, 50, 0.9, 100) == (0.0, 100, 1.0)  # greedy wipes top-k/p
    assert S.effective_params(1e-6, 50, 0.9, 100) == (0.0, 100, 1.0)
    assert S.effective_params(1.0, 0, 1.0, 100) == (1.0, 100, 1.0)  # top_k 0 -> disabled
    assert S.effective_params(1.0, 500, 0.9, 100) == (1.0, 100, 0.9)


def test_nucleus_masks_semantics():
    x = np.log(np.array([0.5, 0.3, 0.1, 0.06, 0.04], dtype=np.float32))
    keep, amb = S.nucleus_masks(x, top_k=5, top_p=0.85)
    # mass strictly above: [0, .5, .8, .9, .96] -> kept iff < 0.85
    assert keep.tolist() == [True, True, True, False, False]
    keep_k, _ = S.nucleus_masks(x, top_k=2, top_p=1.0)
    assert keep_k.tolist() == [True, True, False, False, False]
    # ties at the k-th value: torch sort path keeps all of them (ambiguous); the Triton kernel keeps
    # exactly k, surplus dropped in vocabulary order
    xt = np.array([3.0, 2.0, 2.0, 1.0], dtype=np.float32)
    keep_t, amb_t = S.nucleus_masks(xt, top_k=2, top_p=1.0, topk_ties="keep_all")
    assert keep_t.tolist() == [True, True, True, False] and amb_t[1] and amb_t[2]
    keep_e, amb_e = S.nucleus_masks(xt, top_k=2, top_p=1.0, topk_ties="exact_k")
    assert keep_e.tolist() == [True, True, False, False] and amb_e[1] and amb_e[2] and not amb_e[0]
    with pytest.raises(ValueError):
        S.nucleus_masks(xt, top_k=2, top_p=1.0, topk_ties="whatever")


def test_topk_tie_rule_moves_the_nucleus_boundary_and_is_pinned_or_abstained():
    """The bug found on the nightly rows: ties at the k-th logit change the softmax renormalisation,
    so exact_k (Triton) and keep_all (torch) keep different nucleus sets far from any rounding
    tolerance, and can select different tokens under the same noise."""
    V = 4096
    x = np.full(V, -30.0, dtype=np.float32)
    x[:6] = np.array([10.0, 9.5, 9.0, 8.5, 8.25, 8.0], dtype=np.float32)
    x[6:16] = 7.0  # 10-way tie exactly at the k-th value for k = 10 (4 survive under exact_k)
    keep_e, _ = S.nucleus_masks(x, top_k=10, top_p=1.0, topk_ties="exact_k")
    keep_a, _ = S.nucleus_masks(x, top_k=10, top_p=1.0, topk_ties="keep_all")
    assert keep_e.sum() == 10 and keep_a.sum() == 16
    assert keep_e[6:10].all() and not keep_e[10:16].any()  # vocabulary order: lowest ids kept
    # with a nucleus cut, the larger renormalisation set of keep_all keeps more tokens
    keep_e, _ = S.nucleus_masks(x, top_k=10, top_p=0.93, topk_ties="exact_k")
    keep_a, _ = S.nucleus_masks(x, top_k=10, top_p=0.93, topk_ties="keep_all")
    assert keep_a.sum() > keep_e.sum()
    # find a seed where the two rules select different tokens; the unpinned checker must abstain there
    found = False
    for seed in range(1, 400):
        a = S.independent_select(x, 1.0, 10, 0.93, seed, 5, topk_ties="exact_k")
        b = S.independent_select(x, 1.0, 10, 0.93, seed, 5, topk_ties="keep_all")
        u = S.independent_select(x, 1.0, 10, 0.93, seed, 5, topk_ties=None)
        assert u.predicted == a.predicted
        if a.predicted != b.predicted:
            found = True
            assert not u.certified and "not pinned" in (u.reason or "")
            assert u.details["keep_all_prediction"] == b.predicted
    assert found


def test_independent_select_greedy_and_stochastic_paths():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(2048) * 2).astype(np.float32)
    x[17] += 12.0
    g = S.independent_select(x, 0.0, 0, 1.0, None, 40)
    assert g.predicted == 17 and g.certified
    s1 = S.independent_select(x, 1.0, 50, 0.9, 1, 40)
    s1b = S.independent_select(x, 1.0, 50, 0.9, 1, 40)
    assert s1.predicted == s1b.predicted and s1.cls == "independent"
    # N09 flavour: wrong seed/pos are different functions of the logits (flat distribution)
    flat = (rng.standard_normal(2048) * 3).astype(np.float32)
    preds = {S.independent_select(flat, 1.0, 50, 0.9, seed, 40).predicted for seed in range(1, 30)}
    assert len(preds) > 1
    preds_pos = {S.independent_select(flat, 1.0, 0, 1.0, 1, pos).predicted for pos in range(30, 60)}
    assert len(preds_pos) > 1
    # a request without a seed is not reproducible and says so
    r = S.independent_select(x, 1.0, 50, 0.9, None, 40)
    assert r.predicted == -1 and not r.certified and "no seed" in r.reason


def test_certification_flags_close_races():
    # two tokens with the same logit and identical noise -> zero margin -> not certified
    x = np.full(64, -30.0, dtype=np.float32)
    x[3] = 10.0
    x[9] = 10.0
    r = S.independent_select(x, 1.0, 0, 1.0, 1, 5)
    if r.margin is not None and r.margin < 1e-6:
        assert not r.certified
    else:
        assert r.certified  # comfortably separated by noise -> certified


class _ToyTokenizer:
    """Whitespace tokenizer with a stable vocabulary; enough for the generator's contract."""

    eos_token = "<eos>"

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def encode(self, text, add_special_tokens=False):
        out = []
        for w in text.split(" "):
            if w == "":
                continue
            out.append(self.vocab.setdefault(w, len(self.vocab) + 5))
        return out

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


def test_workload_exact_lengths_and_determinism():
    tok = _ToyTokenizer()
    wl1 = W.build_workload(tok, model="toy", seed=0)
    wl2 = W.build_workload(_ToyTokenizer(), model="toy", seed=0)
    assert [r.prompt_len for r in wl1.requests] == [32, 32, 32, 32, 128, 128, 128, 128, 512, 512, 512, 512, 2048, 2048, 2048, 2048]
    assert [r.max_tokens for r in wl1.requests] == [16, 16, 64, 64] * 4
    assert all(len(r.prompt_token_ids) == r.prompt_len for r in wl1.requests)
    assert [r.request_id for r in wl1.requests] == [r.request_id for r in wl2.requests]
    assert len({r.request_id for r in wl1.requests}) == 16
    sreqs = wl1.expand("topk_topp", 1)
    assert all(s.sampling == {"temperature": 1.0, "top_p": 0.9, "top_k": 50, "max_tokens": s.max_tokens, "seed": 1} for s in sreqs)
    assert len({s.request_id for s in wl1.expand_all()}) == 16 * 7
    g = wl1.expand("greedy", 3)
    assert all(s.seed is None and "seed" not in s.sampling for s in g)
    j = wl1.to_json()
    assert W.Workload.from_json(j).requests[0] == wl1.requests[0]


def _rec(ids, max_tokens, fr, sr=None, seed=None, extra=None, presence=True):
    rec = {"request_id": "r", "index": 0, "prompt_len": 10, "max_tokens": max_tokens, "output_token_ids": ids,
           "finish_reason": fr, "stop_reason": sr, "sampling": {"temperature": 0.0, "top_k": 0, "top_p": 1.0},
           "seed": seed, "sampling_extra": extra or {}}
    if presence:
        rec["output_presence"] = list(ids) + [None] * (max_tokens - len(ids))
    return rec


EOS = 151643


def test_stopping_relations():
    ok, f = ST.check_finish_reason(_rec([1, 2, 3, 4], 4, "length"), EOS)
    assert ok, f
    ok, f = ST.check_finish_reason(_rec([1, 2, EOS], 4, "stop"), EOS)
    assert ok, f
    # EOS exactly at max_tokens is "stop" in vLLM 0.28.0 (check_stop tests EOS before length)
    ok, f = ST.check_finish_reason(_rec([1, 2, 3, EOS], 4, "stop"), EOS)
    assert ok, f
    ok, f = ST.check_finish_reason(_rec([1, 2, 3, EOS], 4, "length"), EOS)
    assert not ok
    # stop token id -> stop_reason is the id
    ok, f = ST.check_finish_reason(_rec([1, 198], 4, "stop", 198, extra={"stop_token_ids": [198]}), EOS)
    assert ok, f
    # a wrong claim: "length" while EOS came earlier
    ok, f = ST.check_finish_reason(_rec([EOS, 2, 3, 4], 4, "length"), EOS)
    assert not ok
    # abort: partial, consistent
    r = _rec([1, 2], 8, "abort")
    r["client_finish_reason"] = "abort"
    ok, f = ST.check_finish_reason(r, EOS)
    assert ok, f
    ok, f = ST.check_output_presence(_rec([1, 2, 3], 5, "abort"))
    assert ok
    bad = _rec([1, 2, 3], 5, "length")
    bad["output_presence"][4] = 7  # a non-blank slot after finish (N14 flavour)
    assert not ST.check_output_presence(bad)[0]
    x = np.zeros(EOS + 10, dtype=np.float32)
    x[EOS] = 5.0
    assert ST.check_eos_selection(_rec([EOS], 4, "stop"), x, EOS, 10)[0]
