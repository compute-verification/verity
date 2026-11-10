"""The cross-rank collective-output pass: picks are rank-agnostic and seeded; the definition is bit-exact per kind;
a received output that is not the definition over the committed partials fails BY NAME; a missing member is a named FAIL, never a pass."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.check.match import xrank_collectives as XC  # noqa: E402

SITES = ["model.layers.0.self_attn.o_proj", "model.layers.0.mlp.down_proj", "model.embed_tokens"]


def _layouts(rank: int, steps: int = 6, prefill_rows: int = 40, hidden: int = 8):
    lay = {}
    for s in range(steps):
        rows = prefill_rows if s == 0 else 3
        ms = []
        for site in SITES:
            ms.append((f"{site}/part_rank{rank}", (rows, hidden)))
            ms.append((f"{site}/0", (rows, hidden)))
        ms.append((f"lm_head/shard_rank{rank}", (rows, hidden // 2)))
        ms.append(("lm_head/0", (rows, hidden)))
        lay[s] = ms
    return lay


def test_picks_are_identical_across_ranks_and_stratified_by_site_kind_phase():
    seed = b"\x01" * 32
    p0 = XC.pick_strata(_layouts(0), 0, seed, per_stratum=2, max_rows=4)
    p1 = XC.pick_strata(_layouts(1), 1, seed, per_stratum=2, max_rows=4)
    k0 = [(p["site"], p["step"], p["kind"], tuple(p["rows"])) for p in p0["picks"]]
    k1 = [(p["site"], p["step"], p["kind"], tuple(p["rows"])) for p in p1["picks"]]
    assert k0 == k1 and k0
    assert p0["picks"][0]["member"].endswith("_rank0") and p1["picks"][0]["member"].endswith("_rank1")
    # the received output may be committed as <site>/out (MoE), <site>/0 (dense tuple output) or the BARE site (single-tensor module output:
    # VocabParallelEmbedding, LogitsProcessor -- measured b1_r16_tp12pre pair 0: 5 picks 'received not committed' before the bare name was a candidate)
    emb = next(p for p in p0["picks"] if p["site"] == "model.embed_tokens")
    assert emb["recv_candidates"] == ["model.embed_tokens/out", "model.embed_tokens/0", "model.embed_tokens"]
    strata = p0["strata"]
    # every site x kind has a prefill stratum (1 step -> 1 pick) and a decode stratum (5 steps -> 2 picks)
    for site in SITES:
        assert strata[f"{site}|all_reduce|prefill"]["picked"] == [0]
        assert len(strata[f"{site}|all_reduce|decode"]["picked"]) == 2
    assert strata["lm_head|all_gather|prefill"]["family"] == "AllGather2_v1"
    assert strata["model.embed_tokens|all_reduce|decode"]["family"] == "AllReduce2_v1" and XC.is_embedding_site("model.embed_tokens")
    assert strata["model.layers.0.mlp.down_proj|all_reduce|decode"]["family"] == "AllReduce2_v1"
    # prefill picks take a row subset; decode picks take every row
    pre = next(p for p in p0["picks"] if p["step"] == 0 and p["kind"] == "all_reduce")
    assert len(pre["rows"]) == 4 and pre["rows_total"] == 40 and all(0 <= r < 40 for r in pre["rows"])
    dec = next(p for p in p0["picks"] if p["step"] != 0 and p["kind"] == "all_reduce")
    assert dec["rows"] == [0, 1, 2]
    # a different seed draws different decode steps (with overwhelming probability over 3 sites x 5 steps)
    q0 = XC.pick_strata(_layouts(0), 0, b"\x02" * 32, per_stratum=2, max_rows=4)
    assert [p["step"] for p in q0["picks"]] != [p["step"] for p in p0["picks"]] or [p["rows"] for p in q0["picks"]] != [p["rows"] for p in p0["picks"]]


def _dumps(flip: str | None = None, drop_recv_rank: int | None = None, gather_wrong: bool = False, emb_bad_rank: int | None = None, emb_missing: bool = False):
    torch.manual_seed(0)
    picks0, picks1 = [], []
    for i, site in enumerate(SITES + ["lm_head"]):
        kind = "all_gather" if site == "lm_head" else "all_reduce"
        a, b = torch.randn(3, 8).to(torch.bfloat16), torch.randn(3, 8).to(torch.bfloat16)
        if kind == "all_reduce":
            real_b = b.clone()
            if flip == site:
                real_b[0, 0] = real_b[0, 0] * 2 + 1024.0        # what rank 1 fed the collective (not what it committed)
            recv = (a.float() + real_b.float()).to(torch.bfloat16)
        else:
            recv = torch.cat([a, b], dim=-1)
            if gather_wrong:
                recv = torch.cat([b, a], dim=-1)                # the ranks' shards concatenated in the wrong order
        base = {"step": 2, "site": site, "kind": kind, "family": XC.family_of(site, kind), "stratum": f"{site}|{kind}|decode", "rows": [0, 1, 2]}
        emb0 = emb1 = None
        if XC.is_embedding_site(site) and not emb_missing:
            emb0 = {"ok": emb_bad_rank != 0, "n_diff": (3 if emb_bad_rank == 0 else 0), "vocab_range": [0, 64128], "masked_rows": 1, "why": None}
            emb1 = {"ok": emb_bad_rank != 1, "n_diff": (3 if emb_bad_rank == 1 else 0), "vocab_range": [64128, 128256], "masked_rows": 2, "why": None}
        picks0.append(dict(base, part=a, recv=recv.clone(), recv_member=f"{site}/0", why=None, emb=emb0))
        picks1.append(dict(base, part=b, recv=(None if drop_recv_rank == 1 else recv.clone()), recv_member=f"{site}/0", why=("received output not committed" if drop_recv_rank == 1 else None), emb=emb1))
    return {0: {"rank": 0, "picks": picks0}, 1: {"rank": 1, "picks": picks1}}


def test_definition_holds_on_the_honest_record_for_every_family():
    r = XC.check_picks(_dumps(), world=2)
    # 4 received-output picks (3 all_reduce incl. the embedding site + 1 all_gather) + the EmbeddingShard_v1 partial leg at the embedding site
    assert r["result"] is True and r["picks"] == 5 and r["equal"] == 5 and not r["mismatches"] and not r["missing"]
    assert set(r["by_family"]) == {"AllReduce2_v1", "EmbeddingShard_v1", "AllGather2_v1"}
    assert r["by_family"]["AllGather2_v1"]["equal"] == 1 and r["by_family"]["AllReduce2_v1"]["equal"] == 3 and r["by_family"]["EmbeddingShard_v1"]["equal"] == 1


def test_a_partial_that_is_not_the_masked_gather_from_the_ranks_shard_fails_the_embedding_leg_by_rank():
    r = XC.check_picks(_dumps(emb_bad_rank=1), world=2)
    assert r["result"] is False and r["mismatch_n"] == 1 and r["by_family"]["EmbeddingShard_v1"]["mismatch"] == 1 and r["by_family"]["AllReduce2_v1"]["equal"] == 3
    m = r["mismatches"][0]
    assert m["kind"] == "embedding_shard" and m["site"] == "model.embed_tokens" and m["received_equals_definition_per_rank"] == {"0": True, "1": False}
    r2 = XC.check_picks(_dumps(emb_missing=True), world=2)
    assert r2["result"] is False and r2["missing_n"] == 1 and "embedding-shard leg not evaluable" in r2["missing"][0]["why"]


def test_a_collective_input_that_is_not_the_committed_partial_fails_by_site():
    r = XC.check_picks(_dumps(flip="model.layers.0.mlp.down_proj"), world=2)
    assert r["result"] is False and r["mismatch_n"] == 1 and r["equal"] == 4
    m = r["mismatches"][0]
    assert m["site"] == "model.layers.0.mlp.down_proj" and m["kind"] == "all_reduce" and m["family"] == "AllReduce2_v1"
    assert m["received_equals_definition_per_rank"] == {"0": False, "1": False} and m["received_equal_across_ranks"] is True
    assert m["n_diff_per_rank"] == {"0": 1, "1": 1}
    assert "down_proj" in r["why"]


def test_a_wrong_gather_order_fails_the_all_gather_pick():
    r = XC.check_picks(_dumps(gather_wrong=True), world=2)
    assert r["result"] is False and r["mismatches"][0]["site"] == "lm_head" and r["mismatches"][0]["kind"] == "all_gather"


def test_a_received_output_missing_on_one_rank_is_a_named_fail_not_a_pass():
    r = XC.check_picks(_dumps(drop_recv_rank=1), world=2)
    assert r["result"] is False and r["missing_n"] == 4 and r["equal"] == 1      # the embedding-shard leg (per rank, no received member needed) still holds
    assert "rank 1" in r["missing"][0]["why"] and "received" in r["missing"][0]["why"]


def _chain(parts, order):
    acc = parts[order[0]]
    for r in order[1:]:
        acc = (acc.float() + parts[r].float()).to(torch.bfloat16)
    return acc


def _dumps_world(parts, recv):
    site = "model.layers.0.mlp.down_proj"
    world = len(parts)
    base = {"step": 2, "site": site, "kind": "all_reduce", "family": XC.family_of(site, "all_reduce", world), "stratum": f"{site}|all_reduce|decode",
            "rows": list(range(16))}
    return {r: {"rank": r, "picks": [dict(base, part=parts[r], recv=recv.clone(), recv_member=f"{site}/0", why=None, emb=None)]} for r in range(world)}


def test_world_4_all_reduce_is_the_descending_rank_chain():
    g = torch.Generator().manual_seed(4)
    parts = [(torch.randn(16, 64, generator=g) * 4.0 ** r).to(torch.bfloat16) for r in range(4)]
    desc, asc = _chain(parts, [3, 2, 1, 0]), _chain(parts, [0, 1, 2, 3])
    assert not torch.equal(desc, asc)
    r = XC.check_picks(_dumps_world(parts, desc), world=4)
    assert r["result"] is True and r["equal"] == 1 and set(r["by_family"]) == {"AllReduce_v2"}
    r = XC.check_picks(_dumps_world(parts, asc), world=4)
    assert r["result"] is False and r["mismatch_n"] == 1 and r["mismatches"][0]["family"] == "AllReduce_v2"


def test_one_rank_dump_is_a_fail():
    d = _dumps()
    r = XC.check_picks({0: d[0]}, world=2)
    assert r["result"] is False and "1/2 rank dumps" in r["why"]


def test_fold_over_pairs_is_true_only_when_every_pair_is_true():
    ok, bad = XC.check_picks(_dumps(), world=2), XC.check_picks(_dumps(flip="model.embed_tokens"), world=2)
    assert XC.fold_pairs([ok, ok])["result"] is True
    f = XC.fold_pairs([ok, bad])
    assert f["result"] is False and f["picks"] == 10 and f["mismatch_n"] == 1 and "embed_tokens" in f["why"]
    assert f["by_family"]["AllReduce2_v1"] == {"picks": 6, "equal": 5, "mismatch": 1, "missing": 0}
    assert f["by_family"]["EmbeddingShard_v1"] == {"picks": 2, "equal": 2, "mismatch": 0, "missing": 0}
    assert XC.fold_pairs([])["result"] is None


def test_flip_site_alters_the_collective_input_but_not_the_committed_partial():
    """The TP-12 negative: TPPartialSource binds x (commits it) and then hands the collective a different tensor."""
    from verity_vllm.acquire.sources import partial_source as TPS

    class _C:  # a committer stub: records what was bound
        _cur = object()

        def __init__(self):
            self.bound = []
            self.errors = []

        def _acquire(self, member, x, job):
            self.bound.append((member, x.clone()))

    src = TPS.TPPartialSource.__new__(TPS.TPPartialSource)
    src.c, src.rank, src.world, src.label_rank, src.omit_site, src.reference = _C(), 1, 2, 1, None, False
    src.flip_site, src.flipped, src.stats = "layers.0.mlp.down_proj", 0, {"unattributed": 0, "launches": 0, "bytes": 0, "acquire_s": 0.0, "shapes": {}, "kinds": {}}
    src.flip_mode = "gross"
    src.cur = ("model.layers.0.mlp.down_proj", "RowParallelLinear")
    src.ref, src.ref_in, src._cur_in, src.ref_reexec_steps = {}, {}, None, 0
    seen = {}

    def orig(x):
        seen["x"] = x.clone()
        return x * 2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.ones(2, 4, dtype=torch.bfloat16, device=dev)
    if dev == "cpu":
        # _bind only binds CUDA tensors; on CPU we exercise the flip branch through the wrapper with a device-typed stub
        class _X(torch.Tensor):
            pass
        pytest.skip("flip branch needs a CUDA tensor (exercised on the pod by neg_flip_partial)")
    w = src._wrap("all_reduce", orig)
    w(x)
    assert src.flipped == 1
    assert torch.equal(src.c.bound[0][1], torch.ones(2, 4, dtype=torch.bfloat16, device=dev))      # committed = genuine
    exp0 = float((torch.ones((), dtype=torch.bfloat16, device=dev) * 2 + 1024.0))                   # bf16(1*2+1024) = 1024.0 (spacing 8 at 2^10)
    assert float(seen["x"].view(-1)[0]) == exp0 != 1.0 and float(seen["x"].view(-1)[1]) == 1.0      # the collective saw a different tensor
    # `:ulp` mode: element 0 moves by exactly one bf16 ulp (1.0 -> 1.0078125), nothing else; the committed partial stays genuine
    src.flip_mode, src.c.bound = "ulp", []
    w(torch.ones(2, 4, dtype=torch.bfloat16, device=dev))
    assert src.flipped == 2
    assert torch.equal(src.c.bound[0][1], torch.ones(2, 4, dtype=torch.bfloat16, device=dev))
    assert float(seen["x"].view(-1)[0]) == 1.0078125 and float(seen["x"].view(-1)[1]) == 1.0


def test_flip_partial_spec_parses_the_ulp_mode():
    from verity_vllm.acquire.sources import partial_source as TPS
    for spec, site, mode in (("layers.3.mlp.down_proj", "layers.3.mlp.down_proj", "gross"), ("layers.3.mlp.down_proj:ulp", "layers.3.mlp.down_proj", "ulp"), (None, None, "gross")):
        assert TPS.TPPartialSource.parse_flip(spec) == (site, mode)


def test_coverage_of_unreplayable_sites_is_complete_only_when_every_program_site_has_a_stratum():
    """The strata cover the sites the replay could not evaluate; a site with a received output but no stratum is a named FAIL."""
    r = XC.check_picks(_dumps(), world=2)
    assert r["result"] is True
    meta = [{"rank": 0, "unreplayable_sites": {"AllReduce2_v1": SITES, "AllGather2_v1": ["lm_head"], "EmbeddingShard_v1": ["model.embed_tokens"]}},
            {"rank": 1, "unreplayable_sites": {"AllReduce2_v1": SITES, "AllGather2_v1": ["lm_head"], "EmbeddingShard_v1": ["model.embed_tokens"]}}]
    cov = XC.coverage_of_unreplayable(r, meta)
    assert cov["complete"] is True and r["result"] is True and cov["families"]["AllReduce2_v1"]["missing"] == []
    # the Program names a prefixed path for the same module: covered by suffix
    r2 = XC.check_picks(_dumps(), world=2)
    cov2 = XC.coverage_of_unreplayable(r2, [{"rank": 0, "unreplayable_sites": {"AllReduce2_v1": ["inner.model." + s for s in SITES]}}, {"rank": 1, "unreplayable_sites": {"AllReduce2_v1": []}}])
    assert cov2["complete"] is True
    # a site the pass never stratified (no committed partial for it) -> missing by name, result False
    r3 = XC.check_picks(_dumps(), world=2)
    cov3 = XC.coverage_of_unreplayable(r3, [{"rank": 0, "unreplayable_sites": {"AllReduce2_v1": SITES + ["model.layers.7.mlp.down_proj"]}}, {"rank": 1, "unreplayable_sites": {"AllReduce2_v1": SITES}}])
    assert cov3["complete"] is False and cov3["families"]["AllReduce2_v1"]["missing"] == ["model.layers.7.mlp.down_proj"]
    assert r3["result"] is False and "model.layers.7.mlp.down_proj" in r3["why"]
    # a rank that could not name its sites -> not complete (never silently complete)
    r4 = XC.check_picks(_dumps(), world=2)
    cov4 = XC.coverage_of_unreplayable(r4, [{"rank": 0, "unreplayable_sites": {"error": "boom"}}, {"rank": 1, "unreplayable_sites": {"AllReduce2_v1": SITES}}])
    assert cov4["complete"] is False and r4["result"] is False and "boom" in r4["why"]
    f = XC.fold_pairs([r, r3])
    assert f["coverage_of_unreplayable"]["complete"] is False and f["result"] is False
