"""The weights of record of a block-FP8 MoE's experts on a TP rank: e4m3 `w13_weight` / `w2_weight` sliced like any expert stack, and
`w13_weight_scale_inv` / `w2_weight_scale_inv` on the rank's refined block grid (`check.fp8_moe_scales`).

The record holds the refined f32 tensor vLLM d9105ea80 serves after loading, so its digest is compared with the served parameter's like
any other field.  Three oracles for that tensor, on a toy checkpoint shaped so that TP8 refines (I = 1536, G = 128: I/8 = 192, g = 64)
and TP16 refines twice as far (g = 32):

  * the index map, by explicit loops: `w13 gate (up): ws[j][k] = ws128[(r·I_r/g + j) // f][k // f]`, `w2: ws[n][k] = ws128[n // f][(r·I_r/g
    + k) // f]`, f = G/g;
  * the loader restated in torch: `create_weights`' scale shapes, `weight_loader`'s `repeat_interleave` of a BLOCK scale, `_load_w13`'s
    `narrow(0, loaded_per_rank·r)` into the gate / up half, `_load_w2`'s `narrow(1, ...)`;
  * vLLM's own `refine_fp8_moe_block_shape`, `RoutedExperts._load_w13` / `_load_w2`, where vLLM imports.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

from verity_vllm.check import weights_of_record as WR
from tests.check.test_weights_of_record import live_digests, write_safetensors

G, I, H, E = 128, 1536, 256, 2
CONTAINER = "model.layers.0.mlp.experts"
SERVED = {n: f"{CONTAINER}.routed_experts.{n}" for n in ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv")}
PARTS = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
WORLDS = (1, 2, 4, 8, 16)
OF_RECORD = {"program_digests_of_record": {"e" * 128}, "checkpoint_of_record": {"repo": "test/fp8-moe", "revision": "abc"}}


def _ty(shape, bits):
    t = f"Value<{bits}>"
    for n in reversed(shape):
        t = f"Array<{n},{t}>"
    return t


def _block(world: int) -> int:
    from verity_vllm.program.model import fp8_moe_block
    return fp8_moe_block(G, I // world, H)


def _fields(world: int) -> list[tuple[str, list[int], int]]:
    I_r = I // world
    g = _block(world)
    return [("model." + SERVED["w13_weight"], [E, 2 * I_r, H], 8), ("model." + SERVED["w13_weight_scale_inv"], [E, 2 * I_r // g, H // g], 32),
            ("model." + SERVED["w2_weight"], [E, H, I_r], 8), ("model." + SERVED["w2_weight_scale_inv"], [E, H // g, I_r // g], 32)]


def _program(world: int, rank: int) -> dict:
    struct_ = "{" + ", ".join(f"{n}: {_ty(s, b)}" for n, s, b in _fields(world)) + "}"
    tp = {"tp": {"world": world, "rank": rank}} if world > 1 else {}
    return {"instances": {"rows": [], "params": [["weights", struct_]], "digest": "e" * 128}, "result": {"program_digest": "e" * 128, **tp},
            "dir": "synthetic"}


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """Per-expert e4m3 weights (random bytes) and f32 block scales whose every word is distinct, in the HF layout of Qwen3-MoE-FP8."""
    rng = np.random.default_rng(5)
    t: dict[str, tuple[str, np.ndarray]] = {}
    for e in range(E):
        for part, (n, k) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            name = f"{CONTAINER}.{e}.{part}"
            t[name + ".weight"] = ("F8_E4M3", rng.integers(0, 256, size=(n, k), dtype=np.uint8))
            base = 1.0 + 1000.0 * (3 * e + ("gate_proj", "up_proj", "down_proj").index(part))
            t[name + ".weight_scale_inv"] = ("F32", (base + np.arange(n // G * k // G, dtype=np.float32)).reshape(n // G, k // G) / 7)
    d = tmp_path_factory.mktemp("fp8moe") / "snap"
    d.mkdir()
    write_safetensors(str(d / "model.safetensors"), t)
    cfg = {"hidden_size": H, "moe_intermediate_size": I, "num_experts": E, "tie_word_embeddings": False,
           "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [G, G]}}
    (d / "config.json").write_text(json.dumps(cfg))
    man = d.parent / "checkpoints.json"
    man.write_text(json.dumps({"checkpoints": [{"repo": "test/fp8-moe", "revision": "abc", "local_path": str(d),
                                                "files": {"model.safetensors": {"sha256": WR.sha256_file(str(d / "model.safetensors"))}}}]}))
    return {"dir": d, "man": man, "t": {n: a for n, (_, a) in t.items()}}


def _ck(m) -> WR.CheckpointIndex:
    return WR.CheckpointIndex(str(m["dir"]), checkpoints_manifest=str(m["man"]))


def _composed(m, world: int, rank: int) -> dict[str, np.ndarray]:
    ck = _ck(m)
    return {n: WR.compose(SERVED[n], s, b, ck, world=world, rank=rank)[0] for n, (_, s, b) in zip(SERVED, _fields(world))}


def _src(m, e: int, shard_id: str, suffix: str) -> np.ndarray:
    return m["t"][f"{CONTAINER}.{e}.{PARTS[shard_id]}.{suffix}"]


# ------------------------------------------------------------------------------------------------------------------ oracle 1: index map

def _by_index_map(m, world: int, rank: int) -> dict[str, np.ndarray]:
    """The served tensors at (world, rank) word by word: weights sliced, scales through the index map."""
    I_r, g = I // world, _block(world)
    f, off = G // g, rank * I_r // g
    w13_ws = np.empty((E, 2 * I_r // g, H // g), dtype=np.float32)
    w2_ws = np.empty((E, H // g, I_r // g), dtype=np.float32)
    for e in range(E):
        for h, sid in enumerate(("w1", "w3")):
            src = _src(m, e, sid, "weight_scale_inv")
            for j in range(I_r // g):
                for k in range(H // g):
                    w13_ws[e, h * I_r // g + j, k] = src[(off + j) // f, k // f]
        src = _src(m, e, "w2", "weight_scale_inv")
        for n in range(H // g):
            for k in range(I_r // g):
                w2_ws[e, n, k] = src[n // f, (off + k) // f]
    return {"w13_weight": np.stack([np.concatenate([_src(m, e, s, "weight")[rank * I_r:(rank + 1) * I_r] for s in ("w1", "w3")]) for e in range(E)]),
            "w13_weight_scale_inv": w13_ws.view(np.uint32),
            "w2_weight": np.stack([_src(m, e, "w2", "weight")[:, rank * I_r:(rank + 1) * I_r] for e in range(E)]),
            "w2_weight_scale_inv": w2_ws.view(np.uint32)}


@pytest.mark.parametrize("world", WORLDS)
def test_compose_is_the_index_map_at_every_rank(ckpt, world):
    for rank in range(world):
        got, want = _composed(ckpt, world, rank), _by_index_map(ckpt, world, rank)
        for n in SERVED:
            assert got[n].shape == want[n].shape and np.array_equal(got[n], want[n]), (world, rank, n)


def test_tp8_scale_shapes_and_the_record_names_the_refinement(ckpt):
    rec = WR.weights_of_record(_program(8, 3), _ck(ckpt), prefix="model.")
    assert rec["complete"] is True and rec["unresolved"] == [] and rec["tp"] == {"world": 8, "rank": 3}
    by = {f["field"][len("model."):]: f for f in rec["fields"]}
    assert by[SERVED["w13_weight_scale_inv"]]["shape"] == [E, 6, 4] and by[SERVED["w2_weight_scale_inv"]]["shape"] == [E, 4, 3]
    for n in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
        assert by[SERVED[n]]["refine"].startswith("block 128 -> 64: each scale repeated 2x along both dims"), by[SERVED[n]]
    assert "refine" not in by[SERVED["w13_weight"]] and "refine" not in by[SERVED["w2_weight"]]
    assert "refine" not in {f["field"]: f for f in WR.weights_of_record(_program(4, 1), _ck(ckpt), prefix="model.")["fields"]}[
        "model." + SERVED["w13_weight_scale_inv"]]


@pytest.mark.parametrize("world", [8, 16])
def test_the_rank_record_passes_its_served_rank_and_fails_every_other_by_name(ckpt, world):
    served = {r: {SERVED[n]: a for n, a in _by_index_map(ckpt, world, r).items()} for r in range(world)}
    for rank in range(world):
        rec = WR.weights_of_record(_program(world, rank), _ck(ckpt), prefix="model.")
        ok = WR.check(rec, live_digests(served[rank]), **OF_RECORD)
        assert ok["result"] is True and ok["compared"] == 4 and ok["mismatched"] == [], ok
        bad = WR.check(rec, live_digests(served[(rank + 1) % world]), **OF_RECORD)
        assert bad["result"] is False and sorted(x["served_name"] for x in bad["mismatched"]) == sorted(SERVED.values())


def test_a_scale_grid_that_is_not_vllms_fails_by_name(ckpt):
    """The checkpoint words, in another arrangement on the same refined grid (tiled instead of repeated; transposed blocks), fail the
    scale fields alone."""
    rec = WR.weights_of_record(_program(8, 5), _ck(ckpt), prefix="model.")
    honest = {SERVED[n]: a for n, a in _by_index_map(ckpt, 8, 5).items()}
    I_r = I // 8

    def tiled(sid, e):
        return np.tile(_src(ckpt, e, sid, "weight_scale_inv"), (2, 2))
    w13 = np.stack([np.concatenate([tiled(s, e)[5 * I_r // 64:6 * I_r // 64] for s in ("w1", "w3")]) for e in range(E)]).view(np.uint32)
    w2 = np.stack([tiled("w2", e)[:, 5 * I_r // 64:6 * I_r // 64] for e in range(E)]).view(np.uint32)
    for field, arr in ((SERVED["w13_weight_scale_inv"], w13), (SERVED["w2_weight_scale_inv"], w2)):
        assert arr.shape == honest[field].shape and not np.array_equal(arr, honest[field])
        res = WR.check(rec, live_digests(dict(honest, **{field: arr})), **OF_RECORD)
        assert res["result"] is False and [x["served_name"] for x in res["mismatched"]] == [field], res


# ------------------------------------------------------------------------------------------------------- oracles 2 and 3: the loader

def _loaded(m, world: int, rank: int, *, refine: Callable, load_w13: Callable, load_w2: Callable) -> dict[str, np.ndarray]:
    """The four parameters at (world, rank) as vLLM d9105ea80 creates and fills them: `Fp8MoEMethod.__init__` / `create_weights` size
    the scales on `refine`'s block, `RoutedExperts.weight_loader` repeats a BLOCK scale `G // g` times along N and K, then `_load_w13`
    (gate as `w1`, up as `w3`, shard dim 0) or `_load_w2` (shard dim 1) cuts the rank's slice into one expert's parameter."""
    torch = pytest.importorskip("torch")
    I_r = I // world
    shape = refine(SimpleNamespace(intermediate_size_per_partition=I_r, tp_size=world, hidden_dim=H), [G, G])
    if shape is None and I_r % G:
        raise ValueError(f"I_r = {I_r}: vLLM refines no grid and create_weights raises")
    g = shape[0] if shape else G
    f = G // g
    params = {"w13_weight": torch.empty(E, 2 * I_r, H, dtype=torch.uint8), "w2_weight": torch.empty(E, H, I_r, dtype=torch.uint8),
              "w13_weight_scale_inv": torch.ones(E, 2 * -(-I_r // g), -(-H // g), dtype=torch.float32),
              "w2_weight_scale_inv": torch.ones(E, -(-H // g), -(-I_r // g), dtype=torch.float32)}
    for e in range(E):
        for suffix in ("weight", "weight_scale_inv"):
            for sid in ("w1", "w3", "w2"):
                lw = torch.from_numpy(np.ascontiguousarray(_src(m, e, sid, suffix)))
                if suffix == "weight_scale_inv" and f > 1:
                    lw = lw.repeat_interleave(f, dim=-2).repeat_interleave(f, dim=-1)
                if sid == "w2":
                    load_w2(params[f"w2_{suffix}"].data[e], 1, lw, rank)
                else:
                    load_w13(params[f"w13_{suffix}"].data[e], 0, sid, lw, rank)
    return {n: (p.numpy().view(np.uint32) if p.dtype == torch.float32 else p.numpy()) for n, p in params.items()}


def _restated_refine(config, weight_block_size):
    """`refine_fp8_moe_block_shape`, fused_moe/oracle/fp8.py:272."""
    import math
    block_n, block_k = weight_block_size
    ispp = config.intermediate_size_per_partition
    if ispp % block_n == 0 and (config.tp_size == 1 or ispp % block_k == 0):
        return None
    g = math.gcd(block_n, block_k, ispp, config.hidden_dim)
    return None if g < 32 else [g, g]


def _restated_loaders(world: int):
    """`RoutedExperts._load_w13` / `_load_w2` (fused_moe/routed_experts.py:478, 529) for act-and-mul experts, no padding."""
    def load_w13(expert_data, shard_dim, shard_id, loaded_weight, tp_rank):
        shard_size = expert_data.shape[shard_dim] // 2
        per = loaded_weight.shape[shard_dim] // world
        loaded_weight = loaded_weight.narrow(shard_dim, per * tp_rank, per)
        expert_data.narrow(shard_dim, 0 if shard_id == "w1" else shard_size, shard_size).copy_(loaded_weight)

    def load_w2(expert_data, shard_dim, loaded_weight, tp_rank):
        per = loaded_weight.shape[shard_dim] // world
        expert_data.copy_(loaded_weight.narrow(shard_dim, per * tp_rank, per))
    return load_w13, load_w2


@pytest.mark.parametrize("world", WORLDS)
def test_compose_is_the_restated_loader_at_every_rank(ckpt, world):
    for rank in range(world):
        load_w13, load_w2 = _restated_loaders(world)
        want = _loaded(ckpt, world, rank, refine=_restated_refine, load_w13=load_w13, load_w2=load_w2)
        got = _composed(ckpt, world, rank)
        for n in SERVED:
            assert got[n].shape == want[n].shape and np.array_equal(got[n], want[n]), (world, rank, n)


def _vllm_loaders(routed_experts_cls, world: int):
    """vLLM's own `_load_w13` / `_load_w2`, called on a stub owner that carries what they read (the TP size, act-and-mul, and the
    class's static padding helpers)."""
    owner = SimpleNamespace(moe_config=SimpleNamespace(is_act_and_mul=True, moe_parallel_config=SimpleNamespace(tp_size=world)),
                            _get_hidden_dim=routed_experts_cls._get_hidden_dim,
                            _narrow_expert_data_for_padding=routed_experts_cls._narrow_expert_data_for_padding)

    def load_w13(expert_data, shard_dim, shard_id, loaded_weight, tp_rank):
        routed_experts_cls._load_w13(owner, expert_data, shard_dim, shard_id, loaded_weight, tp_rank)

    def load_w2(expert_data, shard_dim, loaded_weight, tp_rank):
        routed_experts_cls._load_w2(owner, expert_data, shard_dim, loaded_weight, tp_rank)
    return load_w13, load_w2


@pytest.mark.parametrize("world", WORLDS)
def test_compose_is_vllms_loader_at_every_rank(ckpt, world):
    oracle = pytest.importorskip("vllm.model_executor.layers.fused_moe.oracle.fp8")
    routed = pytest.importorskip("vllm.model_executor.layers.fused_moe.routed_experts")
    for rank in range(world):
        load_w13, load_w2 = _vllm_loaders(routed.RoutedExperts, world)
        want = _loaded(ckpt, world, rank, refine=oracle.refine_fp8_moe_block_shape, load_w13=load_w13, load_w2=load_w2)
        got = _composed(ckpt, world, rank)
        for n in SERVED:
            assert got[n].shape == want[n].shape and np.array_equal(got[n], want[n]), (world, rank, n)


# ---------------------------------------------------------------------------------------------------------------------- refusals

def test_scales_without_their_facts_are_refused_by_name(ckpt, tmp_path):
    ck = _ck(ckpt)
    name, shape = SERVED["w13_weight_scale_inv"], _fields(8)[1][1]
    with pytest.raises(ValueError, match="refuses the checkpoint"):                  # I/64 = 24: gcd(128, 24, 256) = 8 < 32
        WR.compose(name, [E, 2 * 24 // 8, H // 8], 32, ck, world=64, rank=0)
    with pytest.raises(ValueError, match="width of the world 5"):
        WR.compose(name, shape, 32, ck, world=5, rank=0)
    ck.config = {k: v for k, v in ck.config.items() if k != "quantization_config"}
    with pytest.raises(KeyError, match="no square weight_block_size"):
        WR.compose(name, shape, 32, ck, world=8, rank=0)
    ck.config = dict(_ck(ckpt).config, quantization_config={"weight_block_size": [256, 256]})
    with pytest.raises(ValueError, match=r"scale shape \[12, 2\] is not the \[6, 1\] blocks"):
        WR.compose(name, shape, 32, ck, world=8, rank=0)
    rec = WR.weights_of_record(_program(8, 0), _ck(ckpt), prefix="model.")
    assert rec["complete"] is True
    bad = _program(8, 0)
    bad["instances"]["params"][0][1] = bad["instances"]["params"][0][1].replace(_ty([E, 6, 4], 32), _ty([E, 12, 2], 32))
    rec = WR.weights_of_record(bad, _ck(ckpt), prefix="model.")
    assert rec["complete"] is False and [u["field"] for u in rec["unresolved"]] == ["model." + SERVED["w13_weight_scale_inv"]]
