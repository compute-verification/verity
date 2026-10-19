"""DeepSeek-V4's checkpoint names its tensors in its own scheme (`layers.N...`, a `.scale` beside each quantized weight) and stores
its MXFP4 experts as packed e2m1 bytes with e8m0 scales; the index names each tensor as vLLM's loader renames it, and the composer
builds a rank's fused expert stacks from those bytes as the loader writes them.  CPU, a synthetic checkpoint (real safetensors)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity_vllm.check import weights_of_record as WR
from verity_vllm.check.checkpoint_index import renamed, renames
from tests.check.test_weights_of_record import live_digests, write_safetensors
from tests.check.test_weights_of_record_routed_experts import _program

H, I, E, WORLD, RANK = 64, 64, 3, 2, 1          # hidden, expert width, experts; TP world and the rank composed
CONTAINER = "model.layers.0.ffn.experts"
EX = f"{CONTAINER}.routed_experts"


@pytest.fixture
def dsv4(tmp_path):
    rng = np.random.default_rng(5)

    def u8(*shape):
        return rng.integers(0, 256, shape, dtype=np.uint8)

    t = {"embed.weight": ("BF16", u8(8, H).view(np.uint16)), "head.weight": ("BF16", u8(8, H).view(np.uint16)),
         "layers.0.attn.wo_b.weight": ("F8_E4M3", u8(H, H)), "layers.0.attn.wo_b.scale": ("F8_E8M0", u8(1, 1)),
         "layers.0.ffn.gate.bias": ("F32", rng.random(E, dtype=np.float32)),
         "layers.0.ffn.shared_experts.w2.weight": ("F8_E4M3", u8(H, I)), "mtp.0.emb.weight": ("BF16", u8(8, H).view(np.uint16))}
    for e in range(E):
        for p in ("w1", "w3"):
            t[f"layers.0.ffn.experts.{e}.{p}.weight"] = ("I8", u8(I, H // 2).view(np.int8))
            t[f"layers.0.ffn.experts.{e}.{p}.scale"] = ("F8_E8M0", u8(I, H // 32))
        t[f"layers.0.ffn.experts.{e}.w2.weight"] = ("I8", u8(H, I // 2).view(np.int8))
        t[f"layers.0.ffn.experts.{e}.w2.scale"] = ("F8_E8M0", u8(H, I // 32))
    d = tmp_path / "snap"
    d.mkdir()
    write_safetensors(str(d / "model.safetensors"), t)
    json.dump({"model_type": "deepseek_v4", "expert_dtype": "fp4", "hidden_size": H}, open(d / "config.json", "w"))
    man = tmp_path / "checkpoints.json"
    json.dump({"checkpoints": [{"repo": "test/dsv4", "revision": "abc", "local_path": str(d),
                                "files": {"model.safetensors": {"sha256": WR.sha256_file(str(d / "model.safetensors"))}}}]}, open(man, "w"))
    return {"t": t, "ck": WR.CheckpointIndex(str(d), checkpoints_manifest=str(man))}


def test_the_index_names_each_tensor_as_vllms_loader_renames_it(dsv4):
    ck = dsv4["ck"]
    assert sorted(n for n in ck.tensors if not n.startswith(f"{CONTAINER}.")) == [
        "lm_head.weight", "model.embed_tokens.weight", "model.layers.0.attn.wo_b.weight", "model.layers.0.attn.wo_b.weight_scale_inv",
        "model.layers.0.ffn.gate.e_score_correction_bias", "model.layers.0.ffn.shared_experts.down_proj.weight"], "the draft head is not loaded"
    assert ck.dtype("model.layers.0.ffn.experts.2.w3.weight_scale") == "F8_E8M0" and ck.has("model.layers.0.ffn.experts.0.w2.weight")
    fp8 = renames({"model_type": "deepseek_v4", "expert_dtype": "fp8"})
    assert renamed("layers.4.ffn.experts.7.w1.scale", fp8) == "model.layers.4.ffn.experts.7.w1.weight_scale_inv"
    assert renames({"model_type": "deepseek_v3"}) is None


def test_a_ranks_mxfp4_expert_stacks_are_the_checkpoints_bytes_as_the_loader_writes_them(dsv4):
    t, ck = dsv4["t"], dsv4["ck"]

    def raw(name):
        return t[name][1].view(np.uint8)

    rows, cols = slice(RANK * I // 2, (RANK + 1) * I // 2), slice(RANK * I // 2 // 2, (RANK + 1) * I // 2 // 2)
    scols = slice(RANK * I // 32 // 2, (RANK + 1) * I // 32 // 2)
    want = {f"{EX}.w13_weight": np.stack([np.concatenate([raw(f"layers.0.ffn.experts.{e}.{p}.weight")[rows] for p in ("w1", "w3")]) for e in range(E)]),
            f"{EX}.w13_weight_scale": np.stack([np.concatenate([raw(f"layers.0.ffn.experts.{e}.{p}.scale")[rows] for p in ("w1", "w3")])
                                                for e in range(E)]),
            f"{EX}.w2_weight": np.stack([raw(f"layers.0.ffn.experts.{e}.w2.weight")[:, cols] for e in range(E)]),
            f"{EX}.w2_weight_scale": np.stack([raw(f"layers.0.ffn.experts.{e}.w2.scale")[:, scols] for e in range(E)])}
    for name, w in want.items():
        got, note = WR.compose(name, list(w.shape), 8, ck, world=WORLD, rank=RANK)
        assert got.dtype == np.uint8 and np.array_equal(got, w), name
        assert note["shard"] == f"rank {RANK}/{WORLD} ({'per-part dim 0' if '.w13_' in name else 'dim 1'})"
    prog = _program([(f"model.{n}", list(w.shape), 8) for n, w in want.items()])
    prog["result"]["tp"] = {"world": WORLD, "rank": RANK}
    rec = WR.weights_of_record(prog, ck, prefix="model.")
    assert rec["complete"] is True and rec["unresolved"] == [], rec["unresolved"]
    ok = WR.check(rec, live_digests(want), program_digests_of_record={"e" * 128}, checkpoint_of_record={"repo": "test/dsv4", "revision": "abc"})
    assert ok["result"] is True and ok["compared"] == len(want), ok
