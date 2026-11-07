"""`tp_state` on stand-in workers holding CPU tensors shaped like GLM-5.3's caches (a bf16 MLA latent layer, a uint8
indexer-key layer, both block-major over one pool)."""
from __future__ import annotations

import random
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

import tp_state as S  # noqa: E402

BLOCKS, BLOCK, LATENT, IDX = 12, 4, 6, 5


class Layer:
    def __init__(self, kv_cache):
        self.kv_cache = kv_cache


def worker(seed: int, rank: int = 0, *, shared: bool = True):
    """A stand-in worker: two attention layers, an indexer cache, one layer whose cache isn't over the pool, and one
    without a cache; with ``shared``, a second name for the first layer's tensor (as KV sharing binds it)."""
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(BLOCKS, BLOCK, LATENT, generator=g).to(torch.bfloat16)
    latent10 = torch.randn(BLOCKS, BLOCK, LATENT, generator=g).to(torch.bfloat16)
    idx = torch.randint(0, 256, (BLOCKS, BLOCK, IDX), generator=g, dtype=torch.uint8)
    ctx = {"model.layers.10.self_attn.attn": Layer(latent10), "model.layers.2.self_attn.attn": Layer(latent),
           "model.layers.2.self_attn.indexer.k_cache": Layer([idx]), "model.layers.2.mlp": NS(),
           "model.layers.3.self_attn.attn": Layer(torch.zeros(BLOCKS + 1, 2))}
    if shared:
        ctx["model.layers.2.self_attn.attn_alias"] = Layer(latent)
    runner = NS(kv_cache_config=NS(num_blocks=BLOCKS), compilation_config=NS(static_forward_context=ctx))
    return NS(model_runner=runner, rank=rank, local_rank=rank)


def tensors(w):
    return [t for _, t in S.kv_layers(S.runner_of(w))]


def test_kv_layers_are_the_pool_layers_once_each_in_layer_order():
    names = [n for n, _ in S.kv_layers(S.runner_of(worker(0)))]
    assert names == ["model.layers.2.self_attn.attn", "model.layers.2.self_attn.indexer.k_cache",
                     "model.layers.10.self_attn.attn"]
    spec = S.layer_spec(S.kv_layers(S.runner_of(worker(0))))
    assert [(s["dtype"], s["block"], s["block_bytes"]) for s in spec] == [
        ("bfloat16", [BLOCK, LATENT], BLOCK * LATENT * 2), ("uint8", [BLOCK, IDX], BLOCK * IDX),
        ("bfloat16", [BLOCK, LATENT], BLOCK * LATENT * 2)]


def test_a_dump_loads_back_at_the_same_ids_and_every_other_block_is_nan(tmp_path):
    served, fresh = worker(1), worker(2)
    ids = [0, 3, 4, 9]
    wrote = S.dump(served, str(tmp_path), ids, ranks=[0])
    assert wrote["bytes"] == len(ids) * sum(s["block_bytes"] for s in S.layer_spec(S.kv_layers(S.runner_of(served))))
    assert S.dump(served, str(tmp_path), ids, ranks=[1]) is None
    S.load(fresh, str(tmp_path / "kv.rank0"))
    others = [b for b in range(BLOCKS) if b not in ids]
    for a, b in zip(tensors(served), tensors(fresh)):
        assert torch.equal(a[ids].view(torch.uint8), b[ids].view(torch.uint8))
        assert (b[others].view(torch.uint8) == S.FILL).all()
        if b.dtype == torch.bfloat16:
            assert torch.isnan(b[others].float()).all()


def test_each_rank_loads_its_own_dump_if_there_is_one_else_rank_zeros(tmp_path):
    S.dump(worker(1, rank=0), str(tmp_path), [0, 1], ranks=[0])
    S.dump(worker(3, rank=2), str(tmp_path), [0, 1], ranks=[2])
    one, two = worker(5, rank=1), worker(5, rank=2)
    S.load_rank(one, str(tmp_path)), S.load_rank(two, str(tmp_path))
    assert torch.equal(tensors(one)[0][:2], tensors(worker(1))[0][:2])
    assert torch.equal(tensors(two)[0][:2], tensors(worker(3))[0][:2])


def test_a_subset_loads_and_a_block_the_dump_lacks_is_refused(tmp_path):
    served, fresh = worker(1), worker(2)
    S.dump(served, str(tmp_path), [0, 3, 4], ranks=[0])
    S.load(fresh, str(tmp_path / "kv.rank0"), ids=[0, 4], fill=False)
    before = worker(2)
    for a, b, c in zip(tensors(served), tensors(fresh), tensors(before)):
        assert torch.equal(a[[0, 4]].view(torch.uint8), b[[0, 4]].view(torch.uint8))
        assert torch.equal(b[3].view(torch.uint8), c[3].view(torch.uint8))
    with pytest.raises(ValueError, match=r"blocks \[7\] aren't in the dump"):
        S.load(fresh, str(tmp_path / "kv.rank0"), ids=[0, 7])


def test_a_dump_from_another_engine_is_refused_naming_the_layer(tmp_path):
    S.dump(worker(1), str(tmp_path), [0, 1], ranks=[0])
    other = worker(2, shared=False)
    ctx = S.runner_of(other).compilation_config.static_forward_context
    ctx["model.layers.2.self_attn.indexer.k_cache"] = Layer(torch.zeros(BLOCKS, BLOCK, IDX + 1, dtype=torch.uint8))
    with pytest.raises(ValueError, match=r"first differ at layer 1 .*indexer"):
        S.load(other, str(tmp_path / "kv.rank0"))


def test_equal_ranks_digest_equal_and_the_first_differing_block_is_found(tmp_path):
    ids = [0, 2, 5, 7]
    a = S.digest(worker(1, rank=0), ids, str(tmp_path))
    b = S.digest(worker(1, rank=1), ids, str(tmp_path))
    assert a["layer_digests"] == b["layer_digests"] and len(a["layer_digests"]) == 3
    w = worker(1, rank=2)
    tensors(w)[1][5, 0, 0] ^= 1
    c = S.digest(w, ids, str(tmp_path))
    assert c["layer_digests"][0] == a["layer_digests"][0] and c["layer_digests"][1] != a["layer_digests"][1]
    got = S.first_difference((tmp_path / "digests.rank0.bin").read_bytes(), (tmp_path / "digests.rank2.bin").read_bytes(),
                             a["layers"], ids)
    assert got == {"layer": 1, "name": "model.layers.2.self_attn.indexer.k_cache", "block": 5, "index": 2,
                   "blocks_differing": 1}
    assert S.first_difference(b"x" * 16 * 12, b"x" * 16 * 12, a["layers"], ids) is None
    with pytest.raises(ValueError, match="expected 192"):
        S.first_difference(b"x", b"x", a["layers"], ids)
    tensors(w)[2][0, 0, 0] += 1
    tensors(w)[0][5, 0, 0] += 1
    S.digest(w, ids, str(tmp_path))
    assert S.differing_blocks((tmp_path / "digests.rank0.bin").read_bytes(), (tmp_path / "digests.rank2.bin").read_bytes(),
                              a["layers"], ids) == [0, 5]
    assert S.differing_blocks(b"x" * 16 * 12, b"x" * 16 * 12, a["layers"], ids) == []
    with pytest.raises(ValueError, match="expected 192"):
        S.differing_blocks(b"x", b"x", a["layers"], ids)


def test_rng_state_puts_back_every_generator(monkeypatch):
    w = worker(0)
    np.random.seed(7)
    random.seed(7)
    torch.manual_seed(7)
    st = S.rng_state(w)
    want = (np.random.randint(0, 1 << 30, 4).tolist(), random.random(), torch.rand(3).tolist())
    np.random.seed(99)
    random.seed(99)
    torch.manual_seed(99)
    got = S.set_rng(w, st)
    assert (np.random.randint(0, 1 << 30, 4).tolist(), random.random(), torch.rand(3).tolist()) == want
    assert got["rng_digest"] == S.rng_digest({k: v for k, v in st.items() if k != "seeds"})
    assert st["runner"] == "v1" and "seeds" not in st
    assert not hasattr(S.runner_of(w), "_verity_seed_hook")


class V2:
    """A stand-in for Model Runner V2's request slots and sampling states, its ``add_request`` as vLLM d9105ea80 has it
    (a seedless request draws its seed from numpy's global generator)."""

    def __init__(self, n: int = 8):
        self.req_states = NS(req_id_to_index={}, index_to_req_id={}, free=list(range(n)))
        sampling = NS(seeds=NS(np=np.zeros(n, dtype=np.int64)), seeds_set=np.zeros(n, dtype=bool))

        def add_request(req_idx, sampling_params):
            seed = sampling_params.seed
            sampling.seeds_set[req_idx] = seed is not None
            if seed is None:
                seed = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max)
            sampling.seeds.np[req_idx] = seed

        sampling.add_request = add_request
        self.sampler = NS(sampling_states=sampling)

    def add(self, rid: str, seed: int | None = None) -> int:
        i = self.req_states.free.pop()
        self.req_states.req_id_to_index[rid], self.req_states.index_to_req_id[i] = i, rid
        self.sampler.sampling_states.add_request(i, NS(seed=seed))
        return i

    def seed_of(self, rid: str) -> int:
        return int(self.sampler.sampling_states.seeds.np[self.req_states.req_id_to_index[rid]])


def test_restored_requests_keep_their_served_seeds_and_new_ones_draw_as_served():
    np.random.seed(0)
    served = V2()
    served.add("a"), served.add("b", seed=5)
    st = S.rng_state(NS(model_runner=served))
    assert st["runner"] == "v2" and st["seeds"] == {"a": [served.seed_of("a"), False], "b": [5, True]}
    served.add("c")
    np.random.seed(123)
    fresh = V2()
    S.set_rng(NS(model_runner=fresh), st, seeds=True)
    hook = fresh._verity_seed_hook
    fresh.add("c"), fresh.add("a"), fresh.add("b", seed=5)
    assert {r: fresh.seed_of(r) for r in "abc"} == {r: served.seed_of(r) for r in "abc"}
    assert not hook.pending and fresh.sampler.sampling_states.add_request is hook.orig


def test_the_seed_hook_refuses_the_v1_runner():
    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=1"):
        S.SeedHook(S.runner_of(worker(0)), {"a": 1})
