"""MAN-09 (rev-cb M-0610 / coord M-0614 (1)): the weights of record are derived from the PINNED CHECKPOINT BYTES, never from the live
engine; a served model that differs from the checkpoint in ONE element fails the Commit by name."""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct

import numpy as np
import pytest

from verity_vllm.check import weights_of_record as WR
from verity_vllm.pipeline import cli

H, I, HQ, HKV, D, V = 8, 12, 2, 1, 4, 20          # hidden, intermediate, heads, kv heads, head dim, vocab


def _bf16(rng, *shape) -> np.ndarray:
    return (rng.random(shape, dtype=np.float32).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def write_safetensors(path: str, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """A real safetensors shard: 8-byte header length, JSON header, raw little-endian data."""
    hdr, blobs, off = {"__metadata__": {"format": "pt"}}, [], 0
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


@pytest.fixture
def checkpoint(tmp_path):
    rng = np.random.default_rng(7)
    t = {
        "model.embed_tokens.weight": ("BF16", _bf16(rng, V, H)),
        "model.layers.0.self_attn.q_proj.weight": ("BF16", _bf16(rng, HQ * D, H)),
        "model.layers.0.self_attn.k_proj.weight": ("BF16", _bf16(rng, HKV * D, H)),
        "model.layers.0.self_attn.v_proj.weight": ("BF16", _bf16(rng, HKV * D, H)),
        "model.layers.0.self_attn.q_proj.bias": ("BF16", _bf16(rng, HQ * D)),
        "model.layers.0.self_attn.k_proj.bias": ("BF16", _bf16(rng, HKV * D)),
        "model.layers.0.self_attn.v_proj.bias": ("BF16", _bf16(rng, HKV * D)),
        "model.layers.0.self_attn.o_proj.weight": ("BF16", _bf16(rng, H, HQ * D)),
        "model.layers.0.mlp.gate_proj.weight": ("BF16", _bf16(rng, I, H)),
        "model.layers.0.mlp.up_proj.weight": ("BF16", _bf16(rng, I, H)),
        "model.layers.0.mlp.down_proj.weight": ("BF16", _bf16(rng, H, I)),
        "model.layers.0.input_layernorm.weight": ("F32", rng.random(H, dtype=np.float32)),   # stored f32, served bf16 (cast rule)
        "model.norm.weight": ("BF16", _bf16(rng, H)),
        # a 2-expert MoE layer: per-expert checkpoint tensors, fused w13 / w2 in the served model
        "model.layers.1.mlp.gate.weight": ("BF16", _bf16(rng, 2, H)),
        "model.layers.1.mlp.experts.0.gate_proj.weight": ("BF16", _bf16(rng, I, H)),
        "model.layers.1.mlp.experts.0.up_proj.weight": ("BF16", _bf16(rng, I, H)),
        "model.layers.1.mlp.experts.0.down_proj.weight": ("BF16", _bf16(rng, H, I)),
        "model.layers.1.mlp.experts.1.gate_proj.weight": ("BF16", _bf16(rng, I, H)),
        "model.layers.1.mlp.experts.1.up_proj.weight": ("BF16", _bf16(rng, I, H)),
        "model.layers.1.mlp.experts.1.down_proj.weight": ("BF16", _bf16(rng, H, I)),
    }
    d = tmp_path / "snap"
    d.mkdir()
    write_safetensors(str(d / "model.safetensors"), t)
    json.dump({"tie_word_embeddings": True, "num_attention_heads": HQ, "num_key_value_heads": HKV, "hidden_size": H}, open(d / "config.json", "w"))
    sha = WR.sha256_file(str(d / "model.safetensors"))
    man = tmp_path / "checkpoints.json"
    json.dump({"checkpoints": [{"repo": "test/tiny", "revision": "abc", "local_path": str(d), "files": {"model.safetensors": {"sha256": sha}}}]}, open(man, "w"))
    return d, man, t


PREFIX = "model."          # export wrapper: Program `model.model.layers...` -> served `model.layers...`


def _program(tp=None):
    fields = [
        ("model.model.embed_tokens.weight", [V, H], 16),
        ("model.model.layers.0.self_attn.qkv_proj.weight", [(HQ + 2 * HKV) * D, H], 16),
        ("model.model.layers.0.self_attn.qkv_proj.bias", [(HQ + 2 * HKV) * D], 16),
        ("model.model.layers.0.self_attn.o_proj.weight", [H, HQ * D], 16),
        ("model.model.layers.0.mlp.gate_up_proj.weight", [2 * I, H], 16),
        ("model.model.layers.0.mlp.down_proj.weight", [H, I], 16),
        ("model.model.layers.0.input_layernorm.weight", [H], 16),
        ("model.model.layers.0.self_attn.attn._k_scale", [], 32),
        ("model.model.layers.1.mlp.gate.weight", [2, H], 16),
        ("model.model.layers.1.mlp.experts.w13_weight", [2, 2 * I, H], 16),
        ("model.model.layers.1.mlp.experts.w2_weight", [2, H, I], 16),
        ("model.model.norm.weight", [H], 16),
        ("model.lm_head.weight", [V, H], 16),
    ]
    if tp:
        w = tp["world"]
        fields = [(n, ([s[0] // w] + s[1:]) if n.endswith(("qkv_proj.weight", "qkv_proj.bias", "gate_up_proj.weight")) else
                      ([s[0], s[1] // w] if n.endswith(("o_proj.weight", "down_proj.weight")) else
                       ([s[0], s[1] // w, s[2]] if n.endswith("w13_weight") else ([s[0], s[1], s[2] // w] if n.endswith("w2_weight") else s))), b)
                  for n, s, b in fields]

    def ty(shape, bits):
        t = f"Value<{bits}>"
        for n in reversed(shape):
            t = f"Array<{n},{t}>"
        return t
    struct_ = "{" + ", ".join(f"{n}: {ty(s, b)}" for n, s, b in fields) + "}"
    return {"instances": {"rows": [], "params": [["weights", struct_]], "digest": "d" * 128},
            "result": {"program_digest": "d" * 128, **({"tp": tp} if tp else {})}, "dir": "synthetic"}, fields


def _served(t, fields, tp=None):
    """The served model's parameters as the implementation composes them (what register_weights / live_param_digests see)."""
    def A(n):
        arr = t[n][1]
        return WR.to_served(arr, t[n][0], 16)[0]
    w, r = (tp["world"], tp["rank"]) if tp else (1, 0)
    sd0 = (lambda a: a[r * (a.shape[0] // w):(r + 1) * (a.shape[0] // w)]) if w > 1 else (lambda a: a)
    sd1 = (lambda a: a[:, r * (a.shape[1] // w):(r + 1) * (a.shape[1] // w)]) if w > 1 else (lambda a: a)
    L = "model.layers.0."
    served = {
        "model.embed_tokens.weight": A("model.embed_tokens.weight"),
        L + "self_attn.qkv_proj.weight": np.concatenate([sd0(A(L + f"self_attn.{p}_proj.weight")) for p in "qkv"]),
        L + "self_attn.qkv_proj.bias": np.concatenate([sd0(A(L + f"self_attn.{p}_proj.bias")) for p in "qkv"]),
        L + "self_attn.o_proj.weight": sd1(A(L + "self_attn.o_proj.weight")),
        L + "mlp.gate_up_proj.weight": np.concatenate([sd0(A(L + "mlp.gate_proj.weight")), sd0(A(L + "mlp.up_proj.weight"))]),
        L + "mlp.down_proj.weight": sd1(A(L + "mlp.down_proj.weight")),
        L + "input_layernorm.weight": A(L + "input_layernorm.weight"),
        L + "self_attn.attn._k_scale": np.ones((), dtype=np.float32).view(np.uint32),
        "model.layers.1.mlp.gate.weight": A("model.layers.1.mlp.gate.weight"),
        "model.layers.1.mlp.experts.w13_weight": np.stack([np.concatenate([sd0(A(f"model.layers.1.mlp.experts.{e}.gate_proj.weight")),
                                                                            sd0(A(f"model.layers.1.mlp.experts.{e}.up_proj.weight"))]) for e in range(2)]),
        "model.layers.1.mlp.experts.w2_weight": np.stack([sd1(A(f"model.layers.1.mlp.experts.{e}.down_proj.weight")) for e in range(2)]),
        "model.norm.weight": A("model.norm.weight"),
    }
    # tied lm_head: the same tensor object as embed_tokens -> named_parameters() lists it once
    return served


def live_digests(served: dict[str, np.ndarray]) -> dict:
    return {"params": {n: {"digest": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(), "dtype": "bfloat16", "shape": list(a.shape)}
                       for n, a in served.items()}, "n": len(served)}


def test_record_is_derived_from_the_pinned_shards_and_names_every_field(checkpoint):
    d, man, t = checkpoint
    prog, fields = _program()
    ck = WR.CheckpointIndex(str(d), checkpoints_manifest=str(man))
    rec = WR.weights_of_record(prog, ck, prefix=PREFIX)
    assert rec["complete"] is True and rec["unresolved"] == []
    assert rec["checkpoint"]["all_shards_pinned"] is True
    assert rec["checkpoint"]["shards"]["model.safetensors"]["pinned"] is True
    assert rec["n_fields"] == len(fields) and rec["n_pinned"] == len(fields)               # [MAN-09c] the kv scale is pinned as the engine constant
    assert rec["unpinned"] == [] and rec["engine_assumptions"] == {"kv_cache_dtype": "auto"}
    by = {f["field"]: f for f in rec["fields"]}
    assert by["model.model.layers.0.self_attn.attn._k_scale"]["rule"].startswith("engine constant: kv-cache scale == f32 1.0")
    assert by["model.model.layers.0.self_attn.attn._k_scale"]["digest"] == hashlib.sha256(np.float32(1.0).tobytes()).hexdigest()
    assert by["model.model.layers.0.self_attn.qkv_proj.weight"]["rule"].startswith("fused self_attn.qkv_proj")
    assert by["model.model.layers.0.mlp.gate_up_proj.weight"]["sources"] == ["model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight"]
    assert by["model.model.layers.1.mlp.experts.w13_weight"]["rule"].startswith("fused experts w13")
    assert by["model.lm_head.weight"]["rule"].startswith("tied lm_head")
    assert by["model.model.layers.0.input_layernorm.weight"]["cast"] == "F32->bf16 (RNE)"
    # the record is a function of the checkpoint bytes only: rebuilt, same root
    rec2 = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)
    assert rec2["root_of_record"] == rec["root_of_record"]


def test_honest_served_model_passes_and_one_flipped_element_fails_by_name(checkpoint):
    d, man, t = checkpoint
    prog, fields = _program()
    rec = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)
    served = _served(t, fields)
    ok = WR.check(rec, live_digests(served))
    assert ok["result"] is True and ok["mismatched"] == [] and ok["missing"] == [], ok
    assert ok["compared"] == len(fields)                    # every pinned field compared (tied lm_head via embed_tokens; kv scale as 1.0)
    # rev-cb flipweight: ONE element of a served weight differs from the checkpoint (before register_weights)
    bad = dict(served)
    w = bad["model.layers.0.mlp.down_proj.weight"].copy()
    w[3, 5] ^= 0x0040
    bad["model.layers.0.mlp.down_proj.weight"] = w
    res = WR.check(rec, live_digests(bad))
    assert res["result"] is False
    assert [m["served_name"] for m in res["mismatched"]] == ["model.layers.0.mlp.down_proj.weight"]
    assert "differ from the checkpoint of record" in res["why"] and "down_proj" in res["why"]
    # a served model that lacks a Program weight field is a named FAIL too
    short = dict(served)
    del short["model.layers.1.mlp.experts.w2_weight"]
    res2 = WR.check(rec, live_digests(short))
    assert res2["result"] is False and res2["missing"] == ["model.model.layers.1.mlp.experts.w2_weight"]
    # [MAN-09c] a served kv scale that is not the engine constant 1.0 (a quantized cache's scale, or a tampered one) fails by name
    sc = dict(served)
    sc["model.layers.0.self_attn.attn._k_scale"] = np.float32(0.5).reshape(()).view(np.uint32)
    res3 = WR.check(rec, live_digests(sc))
    assert res3["result"] is False and [m["served_name"] for m in res3["mismatched"]] == ["model.layers.0.self_attn.attn._k_scale"]
    # the record's engine assumption (kv_cache_dtype auto) is not the engine's -> refused by name, the scales are not re-assumed
    res4 = WR.check(rec, live_digests(served), kv_cache_dtype="fp8_e4m3")
    assert res4["result"] is False and "record assumes kv_cache_dtype=auto" in res4["why"] and "fp8_e4m3" in res4["why"]
    assert WR.check(rec, live_digests(served), kv_cache_dtype="auto")["result"] is True
    # a record built for a quantized kv cache leaves the scales unpinned BY NAME (no checkpoint scale, no 1.0 assumption)
    recq = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX, kv_cache_dtype="fp8_e4m3")
    assert recq["unpinned"] == ["model.model.layers.0.self_attn.attn._k_scale"] and "kv_cache_dtype=fp8_e4m3" in {f["field"]: f for f in recq["fields"]}["model.model.layers.0.self_attn.attn._k_scale"]["why"]


def test_rotary_table_is_pinned_by_the_frozen_analytic_bound(checkpoint, tmp_path):
    """[MAN-09c, rev-cb W3] the served cos_sin_cache is compared against analytic.check_cos_sin (root_policy gate I9): the table vLLM
    builds from config.json passes; a table from another theta / a corrupted row fails by name; a declared shape that is not
    [MAX_POS, ROT] of config.json is unresolved (fail closed); a rope_scaling type without a table stays unpinned by name."""
    from verity_vllm.program import model as analytic
    d, man, t = checkpoint
    cfg = json.load(open(d / "config.json"))
    cfg.update({"model_type": "llama", "num_hidden_layers": 1, "intermediate_size": 12, "vocab_size": 20, "rope_theta": 10000.0,
                "max_position_embeddings": 64, "head_dim": 4, "rms_norm_eps": 1e-5, "torch_dtype": "bfloat16"})
    json.dump(cfg, open(d / "config.json", "w"))
    C, _ = analytic.config_of(cfg)
    MAXP, ROT = int(C["MAX_POS"]), analytic.rotary_dim(C)
    prog, fields = _program()
    field = "model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"
    prog["instances"]["params"] = [["weights", "{" + f"{field}: Array<{MAXP},Array<{ROT},Value<16>>>, model.model.norm.weight: Array<8,Value<16>>" + "}"]]
    rec = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)
    assert rec["complete"] is True and rec["unpinned"] == [] and rec["analytic_fields"] == [field] and rec["n_pinned"] == 2
    ent = {f["field"]: f for f in rec["fields"]}[field]
    assert ent["pinned"] == "analytic" and ent["digest"] is None and ent["analytic"]["expected_shape"] == [MAXP, ROT]
    table = analytic.cos_sin_table(C)                         # what vLLM's RotaryEmbedding builds (numpy libm here)
    norm = WR.to_served(t["model.norm.weight"][1], t["model.norm.weight"][0], 16)[0]
    live = {"params": {"model.layers.0.self_attn.rotary_emb.cos_sin_cache": {"digest": "x", "dtype": "bfloat16", "shape": [MAXP, ROT]},
                       "model.norm.weight": {"digest": hashlib.sha256(np.ascontiguousarray(norm).tobytes()).hexdigest(), "dtype": "bfloat16", "shape": [8]}},
            "raw": {"model.layers.0.self_attn.rotary_emb.cos_sin_cache": table}}
    ok = WR.check(rec, live)
    assert ok["result"] is True and ok["analytic"][0]["status"] == "pass" and ok["analytic"][0]["n_outside_bound"] == 0, ok["analytic"]
    assert ok["compared"] == 2
    # another theta: the table is not the analytic function of config.json -> FAIL named
    other = analytic.cos_sin_table(dict(C, ROPE_THETA=500000.0))
    bad = WR.check(rec, dict(live, raw={"model.layers.0.self_attn.rotary_emb.cos_sin_cache": other}))
    assert bad["result"] is False and bad["analytic"][0]["status"] == "fail" and "outside the frozen analytic bound" in bad["why"] and "cos_sin_cache" in bad["why"]
    # one corrupted word (a flipped exponent bit at position 7) -> FAIL named with the offending position
    corrupt = table.copy(); corrupt[7, 1] ^= 0x4000
    bad2 = WR.check(rec, dict(live, raw={"model.layers.0.self_attn.rotary_emb.cos_sin_cache": corrupt}))
    assert bad2["result"] is False and bad2["analytic"][0]["n_outside_bound"] >= 1 and bad2["analytic"][0]["first_offending"][0]["pos"] == 7
    # served words not captured -> not a pass
    assert WR.check(rec, dict(live, raw={}))["result"] is False
    # a declared table shape that is not config.json's [MAX_POS, ROT]: unresolved, complete False (fail closed)
    prog2 = dict(prog); prog2["instances"] = dict(prog["instances"], params=[["weights", "{" + f"{field}: Array<{MAXP * 2},Array<{ROT},Value<16>>>" + "}"]])
    rec2 = WR.weights_of_record(prog2, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)
    assert rec2["complete"] is False and rec2["unresolved"][0]["field"] == field and "!= analytic [MAX_POS, ROT]" in rec2["unresolved"][0]["why"]
    # a rope_scaling type without an analytic table: unpinned BY NAME (not assumed, not a silent pass)
    json.dump(dict(cfg, rope_scaling={"rope_type": "longrope", "factor": 2.0}), open(d / "config.json", "w"))
    rec3 = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)
    assert rec3["unpinned"] == [field] and "longrope" in {f["field"]: f for f in rec3["fields"]}[field]["why"]


def test_tp_rank_shards_are_the_per_part_slices(checkpoint):
    d, man, t = checkpoint
    for rank in (0, 1):
        tp = {"world": 2, "rank": rank}
        prog, fields = _program(tp=tp)
        rec = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)
        assert rec["complete"] is True, rec["unresolved"]
        assert rec["tp"] == tp
        served = _served(t, fields, tp=tp)
        res = WR.check(rec, live_digests(served))
        assert res["result"] is True, res
        # a rank-0 record does not pass rank 1's shards
        other = _served(t, fields, tp={"world": 2, "rank": 1 - rank})
        assert WR.check(rec, live_digests(other))["result"] is False


def test_unpinned_shard_and_unresolved_field_are_named(checkpoint, tmp_path):
    d, man, t = checkpoint
    # the manifest pins a different sha: the shard is NOT the checkpoint of record
    bad_man = tmp_path / "bad.json"
    json.dump({"checkpoints": [{"repo": "test/tiny", "local_path": str(d), "files": {"model.safetensors": {"sha256": "0" * 64}}}]}, open(bad_man, "w"))
    ck = WR.CheckpointIndex(str(d), checkpoints_manifest=str(bad_man))
    assert ck.pin["shards"]["model.safetensors"]["pinned"] is False
    prog, fields = _program()
    rec = WR.weights_of_record(prog, ck, prefix=PREFIX)
    assert rec["checkpoint"]["all_shards_pinned"] is False and rec["of_record"] is False and rec["complete"] is True
    # [rev-cb M-0655 W1a] the served model equals the on-disk shards, but the shards are NOT the pinned ones: FAIL named (shard, sha, pin)
    res = WR.check(rec, live_digests(_served(t, fields)))
    assert res["result"] is False and res["shards_ok"] is False
    assert "not the pinned ones (W1)" in res["why"] and "model.safetensors sha256" in res["why"] and "!= manifests/checkpoints.json pin 0000000000000000" in res["why"]
    # [W1b] a checkpoint dir with NO checkpoints.json entry (a hard-link copy outside the manifest): pinned None -> FAIL named
    rec_none = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), checkpoints_manifest=str(tmp_path / "absent.json")), prefix=PREFIX)
    assert rec_none["checkpoint"]["shards"]["model.safetensors"]["pinned"] is None and rec_none["of_record"] is False
    res = WR.check(rec_none, live_digests(_served(t, fields)))
    assert res["result"] is False and "no checkpoints.json entry" in res["why"]
    # [W1c] --no-verify-shards leaves no pin on record: never a PASS
    rec_nv = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), verify_shards=False), prefix=PREFIX)
    assert rec_nv["of_record"] is False
    assert WR.check(rec_nv, live_digests(_served(t, fields)))["result"] is False
    # a Program field with no checkpoint tensor and no composition rule: unresolved, named, complete False -> check False
    prog["instances"]["params"] = [["weights", "{model.model.layers.0.mlp.mystery.weight: Array<8,Value<16>>, model.model.norm.weight: Array<8,Value<16>>}"]]
    rec2 = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), verify_shards=False), prefix=PREFIX)
    assert rec2["complete"] is False and rec2["unresolved"][0]["field"] == "model.model.layers.0.mlp.mystery.weight"
    res = WR.check(rec2, live_digests(_served(t, fields)))
    assert res["result"] is False and "incomplete" in res["why"] and "mystery" in res["why"]


def test_fp8_kv_scale_loader_placeholders_are_pinned_as_the_sentinel_unless_the_checkpoint_ships_them(checkpoint, tmp_path):
    """[R16 fp8 F-fp8-PIN1] A quantized checkpoint's attention layers carry vLLM's `attn.q_scale/k_scale/v_scale/prob_scale` loader
    placeholders (Fp8KVCacheMethod.create_weights: KVCacheScaleParameter = f32 -1.0; never written and read by no kernel with the kv
    cache unquantized) beside the `_k_scale` engine constants.  Qwen3-4B-FP8 B1 on cp-h100 (07:57Z): 144 such fields -> 'no checkpoint
    tensor and no composition rule' -> weights of record incomplete -> Commit FAIL by name, although every one of the 434 checkpoint-
    backed fields (e4m3 weights, f32 weight_scale_inv, norms, embeddings) compared equal.  MAN-09c rule: pinned EXACTLY as the sentinel
    under kv_cache_dtype auto; a served value other than -1.0 FAILS by name; a checkpoint that DOES ship a kv scale of that name is
    pinned 1:1 from the checkpoint (the `not ck.has(live)` guard); another kv_cache_dtype leaves them unpinned by name."""
    d, man, t = checkpoint
    prog, fields = _program()
    L = "model.model.layers.0.self_attn.attn."
    extra = [(L + s, [], 32) for s in ("q_scale", "k_scale", "v_scale", "prob_scale")]

    def ty(shape, bits):
        tt = f"Value<{bits}>"
        for n in reversed(shape):
            tt = f"Array<{n},{tt}>"
        return tt
    prog["instances"]["params"] = [["weights", "{" + ", ".join(f"{n}: {ty(s, b)}" for n, s, b in fields + extra) + "}"]]
    ck = WR.CheckpointIndex(str(d), checkpoints_manifest=str(man))
    rec = WR.weights_of_record(prog, ck, prefix=PREFIX)
    assert rec["complete"] is True and rec["unresolved"] == [], rec["unresolved"]
    assert rec["n_pinned"] == len(fields) + len(extra) and rec["unpinned"] == []     # the placeholders are pinned as the -1.0 sentinel
    by = {f["field"]: f for f in rec["fields"]}
    assert by[L + "k_scale"]["pinned"] is True and "Fp8KVCacheMethod loader placeholder == f32 -1.0" in by[L + "k_scale"]["rule"]
    assert by[L + "k_scale"]["digest"] == hashlib.sha256(np.float32(-1.0).tobytes()).hexdigest()
    # the served model carries the -1.0 sentinels: PASS, every placeholder compared
    served = _served(t, fields)
    for s in ("q_scale", "k_scale", "v_scale", "prob_scale"):
        served["model.layers.0.self_attn.attn." + s] = np.full((), -1.0, dtype=np.float32).view(np.uint32)
    ok = WR.check(rec, live_digests(served))
    assert ok["result"] is True and ok["compared"] == len(fields) + len(extra), ok
    # [F-fp8-PIN2] the measured Programs declare the placeholders Value<16>: vLLM builds `torch.tensor(-1.0)` at the MODEL dtype (bf16), so
    # the served word is bf16 0xBF80 -- pinned as -1.0 in the declared width; a bf16 word other than the sentinel fails by name
    prog16 = json.loads(json.dumps(prog))
    prog16["instances"]["params"] = [["weights", "{" + ", ".join(f"{n}: {ty(s, b)}" for n, s, b in fields) + ", "
                                     + ", ".join(f"{n}: Value<16>" for n, _s, _b in extra) + "}"]]
    rec16 = WR.weights_of_record(prog16, ck, prefix=PREFIX)
    assert rec16["complete"] is True and rec16["unpinned"] == [] and rec16["n_pinned"] == len(fields) + len(extra)
    by16 = {f["field"]: f for f in rec16["fields"]}
    assert by16[L + "v_scale"]["digest"] == hashlib.sha256(np.array([0xBF80], dtype=np.uint16).tobytes()).hexdigest()
    assert "bf16 -1.0 sentinel in the declared width" in by16[L + "v_scale"]["rule"]
    served16 = _served(t, fields)
    for s in ("q_scale", "k_scale", "v_scale", "prob_scale"):
        served16["model.layers.0.self_attn.attn." + s] = np.array([0xBF80], dtype=np.uint16).reshape(())
    assert WR.check(rec16, live_digests(served16))["result"] is True
    served16["model.layers.0.self_attn.attn.q_scale"] = np.array([0x3F80], dtype=np.uint16).reshape(())      # bf16 +1.0: written by someone
    r16 = WR.check(rec16, live_digests(served16))
    assert r16["result"] is False and [m["served_name"] for m in r16["mismatched"]] == ["model.layers.0.self_attn.attn.q_scale"]
    # a declared width with no rule (Value<8>) stays unpinned BY NAME
    prog8 = json.loads(json.dumps(prog))
    prog8["instances"]["params"] = [["weights", "{" + ", ".join(f"{n}: {ty(s, b)}" for n, s, b in fields) + ", " + L + "k_scale: Value<8>}"]]
    rec8 = WR.weights_of_record(prog8, ck, prefix=PREFIX)
    assert rec8["complete"] is True and rec8["unpinned"] == [L + "k_scale"] and "one bf16/f32 word" in {f["field"]: f for f in rec8["fields"]}[L + "k_scale"]["why"]
    # a placeholder the loader wrote from somewhere else (1.0 where the checkpoint has no scale) fails BY NAME
    sc = dict(served)
    sc["model.layers.0.self_attn.attn.v_scale"] = np.float32(1.0).reshape(()).view(np.uint32)
    res = WR.check(rec, live_digests(sc))
    assert res["result"] is False and [m["served_name"] for m in res["mismatched"]] == ["model.layers.0.self_attn.attn.v_scale"]
    # a flipped checkpoint-backed field still fails by name beside the placeholders
    bad = dict(served)
    w = bad["model.layers.0.self_attn.o_proj.weight"].copy()
    w[1, 2] ^= 0x0001
    bad["model.layers.0.self_attn.o_proj.weight"] = w
    assert WR.check(rec, live_digests(bad))["result"] is False
    # another kv_cache_dtype: the placeholders are live scales without a rule -> unpinned BY NAME, never assumed
    recq = WR.weights_of_record(prog, ck, prefix=PREFIX, kv_cache_dtype="fp8_e4m3")
    assert set(recq["unpinned"]) == {L + "_k_scale"} | {n for n, _s, _b in extra}
    assert "kv_cache_dtype=fp8_e4m3" in {f["field"]: f for f in recq["fields"]}[L + "k_scale"]["why"]
    # the guard: a checkpoint that SHIPS `k_scale` (a kv_cache_scheme checkpoint) pins it 1:1 as an f32 word, not as the sentinel
    t2 = dict(t)
    t2["model.layers.0.self_attn.attn.k_scale"] = ("F32", np.array(0.0625, dtype=np.float32))
    d2 = tmp_path / "snap2"
    d2.mkdir()
    write_safetensors(str(d2 / "model.safetensors"), t2)
    json.dump(json.load(open(d / "config.json")), open(d2 / "config.json", "w"))
    man2 = tmp_path / "checkpoints2.json"
    json.dump({"checkpoints": [{"repo": "test/tiny2", "revision": "abc", "local_path": str(d2),
                                "files": {"model.safetensors": {"sha256": WR.sha256_file(str(d2 / "model.safetensors"))}}}]}, open(man2, "w"))
    rec2 = WR.weights_of_record(prog, WR.CheckpointIndex(str(d2), checkpoints_manifest=str(man2)), prefix=PREFIX)
    assert rec2["complete"] is True and rec2["n_pinned"] == len(fields) + len(extra)
    by2 = {f["field"]: f for f in rec2["fields"]}
    assert by2[L + "k_scale"]["pinned"] is True and by2[L + "k_scale"]["rule"] == "1:1"
    served2 = dict(served)
    served2["model.layers.0.self_attn.attn.k_scale"] = np.array(0.0625, dtype=np.float32).view(np.uint32)
    assert WR.check(rec2, live_digests(served2))["result"] is True
    served2["model.layers.0.self_attn.attn.k_scale"] = np.array(-1.0, dtype=np.float32).view(np.uint32)   # the sentinel where the checkpoint has a scale
    res2 = WR.check(rec2, live_digests(served2))
    assert res2["result"] is False and [m["served_name"] for m in res2["mismatched"]] == ["model.layers.0.self_attn.attn.k_scale"]


def test_f_r19_int_11_a_loader_placeholder_the_engine_deleted_after_loading_is_named_not_missing(checkpoint, tmp_path):
    """[R19 F-r19-int-11] vLLM d9105ea80's `BaseKVCacheMethod.process_weights_after_loading` ends with `del layer.k_scale / v_scale /
    q_scale / prob_scale` (kv_cache.py:106-109, 209-212): the loader placeholders the record pins as the -1.0 sentinel do not exist on
    the served model.  #74 Qwen3-4B-FP8 B8 on vyv-h100 (12:10Z): 579/579 compared equal, 144 placeholders 'not found among the served
    parameters', 3/3 pairs FAIL with every other check green.  A deleted parameter cannot reach a served value: an absent placeholder is
    reported in `placeholders_deleted` by name and the pin passes -- iff the record assumes kv_cache_dtype auto AND the `_x_scale` the
    kernels read was compared.  A placeholder that IS served is still compared exactly; any other absent field is still `missing`."""
    d, man, t = checkpoint
    prog, fields = _program()
    L = "model.model.layers.0.self_attn.attn."
    extra = [(L + s, [], 32) for s in ("q_scale", "k_scale", "v_scale", "prob_scale")]

    def ty(shape, bits):
        tt = f"Value<{bits}>"
        for n in reversed(shape):
            tt = f"Array<{n},{tt}>"
        return tt
    prog["instances"]["params"] = [["weights", "{" + ", ".join(f"{n}: {ty(s, b)}" for n, s, b in fields + extra) + "}"]]
    ck = WR.CheckpointIndex(str(d), checkpoints_manifest=str(man))
    rec = WR.weights_of_record(prog, ck, prefix=PREFIX)
    assert rec["complete"] is True and rec["n_pinned"] == len(fields) + len(extra)
    # the served model after process_weights_after_loading: `_k_scale` present (compared), the four placeholders deleted
    served = _served(t, fields)
    ok = WR.check(rec, live_digests(served), kv_cache_dtype="auto")
    assert ok["result"] is True, ok["why"]
    assert ok["missing"] == [] and ok["compared"] == len(fields)
    assert sorted(ok["placeholders_deleted"]) == sorted(n for n, _s, _b in extra) and "process_weights_after_loading deletes them" in ok["placeholders_deleted_note"]
    # one placeholder still served (an older engine) is compared exactly: the sentinel passes, another word fails by name
    part = dict(served)
    part["model.layers.0.self_attn.attn.k_scale"] = np.full((), -1.0, dtype=np.float32).view(np.uint32)
    r = WR.check(rec, live_digests(part), kv_cache_dtype="auto")
    assert r["result"] is True and r["compared"] == len(fields) + 1 and len(r["placeholders_deleted"]) == 3
    part["model.layers.0.self_attn.attn.k_scale"] = np.float32(1.0).reshape(()).view(np.uint32)
    r = WR.check(rec, live_digests(part), kv_cache_dtype="auto")
    assert r["result"] is False and [m["served_name"] for m in r["mismatched"]] == ["model.layers.0.self_attn.attn.k_scale"]
    # the rule needs the scale the kernels read: without `_k_scale` among the served parameters the placeholders stay MISSING by name
    no_scale = {k: v for k, v in served.items() if not k.endswith("attn._k_scale")}
    r = WR.check(rec, live_digests(no_scale), kv_cache_dtype="auto")
    assert r["result"] is False and r["placeholders_deleted"] == [] and len(r["missing"]) == len(extra) + 1 and "not found among the served parameters" in r["why"]
    # any other absent field is still missing
    gone = {k: v for k, v in served.items() if k != "model.layers.0.self_attn.o_proj.weight"}
    r = WR.check(rec, live_digests(gone), kv_cache_dtype="auto")
    assert r["result"] is False and r["missing"] == ["model.model.layers.0.self_attn.o_proj.weight"] and len(r["placeholders_deleted"]) == 4
    # another kv_cache_dtype on the record leaves the placeholders unpinned by name -- nothing to delete-classify (they are never compared)
    recq = WR.weights_of_record(prog, ck, prefix=PREFIX, kv_cache_dtype="fp8_e4m3")
    rq = WR.check(recq, live_digests(served), kv_cache_dtype="fp8_e4m3")
    assert rq["placeholders_deleted"] == [] and set(rq["unpinned"]) >= {n for n, _s, _b in extra}


def _quantized_prog_and_ck(t, tmp_path, extra_tensors: dict, sub: str, *, placeholder_bits: int = 32):
    d0 = tmp_path / "snap"                                                          # the fixture's checkpoint dir (config.json: tied lm_head)
    prog, fields = _program()
    L = "model.model.layers.0.self_attn.attn."
    extra = [(L + s, [], placeholder_bits) for s in ("q_scale", "k_scale", "v_scale", "prob_scale")]

    def ty(shape, bits):
        tt = f"Value<{bits}>"
        for n in reversed(shape):
            tt = f"Array<{n},{tt}>"
        return tt
    prog["instances"]["params"] = [["weights", "{" + ", ".join(f"{n}: {ty(s, b)}" for n, s, b in fields + extra) + "}"]]
    t2 = dict(t)
    t2.update(extra_tensors)
    d2 = tmp_path / sub
    d2.mkdir()
    write_safetensors(str(d2 / "model.safetensors"), t2)
    (d2 / "config.json").write_text((d0 / "config.json").read_text())
    man2 = tmp_path / f"{sub}.json"
    json.dump({"checkpoints": [{"repo": f"test/{sub}", "revision": "abc", "local_path": str(d2),
                                "files": {"model.safetensors": {"sha256": WR.sha256_file(str(d2 / "model.safetensors"))}}}]}, open(man2, "w"))
    return prog, fields, L, WR.CheckpointIndex(str(d2), checkpoints_manifest=str(man2))


def test_f_r16_21_a_kv_scale_the_checkpoint_ships_under_a_remapped_name_is_pinned_from_its_source_never_a_placeholder(checkpoint, tmp_path):
    """[rev M-0489 (2) / M-0516: the F-r16-21 residual, the remap-aware compose] kv_cache_scheme checkpoints ship kv scales as `<layer>.k_scale` / `<layer>.v_scale`
    (or modelopt `<layer>.k_proj.output_scale`, legacy `<layer>.kv_scale`) and vLLM's maybe_remap_kv_scale_name renames them to
    `<layer>.attn.k_scale` at load -- a name-exact `ck.has(served)` guard cannot see them and would class a checkpoint-loaded,
    load-bearing scale as an engine constant.  Now the placeholder rule is reached only when NO remap source exists: a source is pinned
    1:1 from the checkpoint (rule names the remap), two sources for one scale are refused, and the sentinel served where the checkpoint
    has a scale FAILS by name."""
    d, man, t = checkpoint
    f32 = lambda x: ("F32", np.array(x, dtype=np.float32))  # noqa: E731
    # (1) kv_cache_scheme names: self_attn.k_scale / v_scale -> pinned from the source; q / prob have no source -> placeholders by name
    prog, fields, L, ck = _quantized_prog_and_ck(t, tmp_path, {"model.layers.0.self_attn.k_scale": f32(0.0625), "model.layers.0.self_attn.v_scale": f32(0.125)}, "kvs")
    rec = WR.weights_of_record(prog, ck, prefix=PREFIX)
    # under auto the sourceless q/prob placeholders are PINNED to the -1.0 sentinel (fp8 621a31fe8), the shipped k/v pinned from their source
    assert rec["complete"] is True and rec["unpinned"] == [] and rec["n_pinned"] == len(fields) + 4
    by = {f["field"]: f for f in rec["fields"]}
    assert by[L + "q_scale"]["sources"][0].startswith("engine:Fp8KVCacheMethod") and by[L + "prob_scale"]["pinned"] is True
    assert by[L + "k_scale"]["pinned"] is True and by[L + "k_scale"]["sources"] == ["model.layers.0.self_attn.k_scale"]
    assert by[L + "k_scale"]["rule"].startswith("kv scale (remapped from checkpoint model.layers.0.self_attn.k_scale -> model.layers.0.self_attn.attn.k_scale")
    assert by[L + "k_scale"]["digest"] == hashlib.sha256(np.float32(0.0625).tobytes()).hexdigest()
    assert by[L + "v_scale"]["digest"] == hashlib.sha256(np.float32(0.125).tobytes()).hexdigest()
    served = _served(t, fields)
    for s, v in (("q_scale", -1.0), ("k_scale", 0.0625), ("v_scale", 0.125), ("prob_scale", -1.0)):
        served["model.layers.0.self_attn.attn." + s] = np.array(v, dtype=np.float32).view(np.uint32)
    assert WR.check(rec, live_digests(served))["result"] is True
    bad = dict(served)
    bad["model.layers.0.self_attn.attn.k_scale"] = np.array(-1.0, dtype=np.float32).view(np.uint32)      # the sentinel where the checkpoint ships a scale
    res = WR.check(rec, live_digests(bad))
    assert res["result"] is False and [m["served_name"] for m in res["mismatched"]] == ["model.layers.0.self_attn.attn.k_scale"]
    # the same rule under a quantized kv cache: the shipped scales are pinned from their source (the remap-aware compose rev M-0516 names
    # as the remedy for the half-cell), the sourceless placeholders unpinned BY NAME (fp8 621a31fe8; never assumed)
    recq = WR.weights_of_record(prog, ck, prefix=PREFIX, kv_cache_dtype="fp8_e4m3")
    assert recq["complete"] is True and set(recq["unpinned"]) == {L + "q_scale", L + "prob_scale", L + "_k_scale"}   # _k_scale: MAN-09c's own non-auto rule
    byq = {f["field"]: f for f in recq["fields"]}
    assert byq[L + "k_scale"]["pinned"] is True and byq[L + "v_scale"]["pinned"] is True and "kv_cache_dtype=fp8_e4m3" in byq[L + "q_scale"]["why"]
    # (2) modelopt: k_proj.output_scale; legacy fused kv_scale feeds BOTH k and v
    prog2, fields2, L2, ck2 = _quantized_prog_and_ck(t, tmp_path, {"model.layers.0.self_attn.k_proj.output_scale": f32(0.5), "model.layers.0.self_attn.v_proj.output_scale": f32(0.25)}, "modelopt")
    by2 = {f["field"]: f for f in WR.weights_of_record(prog2, ck2, prefix=PREFIX)["fields"]}
    assert by2[L2 + "k_scale"]["sources"] == ["model.layers.0.self_attn.k_proj.output_scale"] and by2[L2 + "v_scale"]["sources"] == ["model.layers.0.self_attn.v_proj.output_scale"]
    prog3, fields3, L3, ck3 = _quantized_prog_and_ck(t, tmp_path, {"model.layers.0.self_attn.kv_scale": f32(0.75)}, "legacy")
    by3 = {f["field"]: f for f in WR.weights_of_record(prog3, ck3, prefix=PREFIX)["fields"]}
    assert by3[L3 + "k_scale"]["sources"] == by3[L3 + "v_scale"]["sources"] == ["model.layers.0.self_attn.kv_scale"]
    assert by3[L3 + "k_scale"]["digest"] == by3[L3 + "v_scale"]["digest"] == hashlib.sha256(np.float32(0.75).tobytes()).hexdigest()
    # (3) two sources for one scale -> unresolved by name (never pick one); a non-scalar source -> unresolved
    prog4, fields4, L4, ck4 = _quantized_prog_and_ck(t, tmp_path, {"model.layers.0.self_attn.k_scale": f32(0.0625), "model.layers.0.self_attn.kv_scale": f32(0.75)}, "ambig")
    rec4 = WR.weights_of_record(prog4, ck4, prefix=PREFIX)
    assert rec4["complete"] is False and any(u["field"] == L4 + "k_scale" and "ambiguous" in u["why"] for u in rec4["unresolved"])
    prog5, fields5, L5, ck5 = _quantized_prog_and_ck(t, tmp_path, {"model.layers.0.self_attn.k_scale": ("F32", np.array([0.1, 0.2], dtype=np.float32))}, "vec")
    rec5 = WR.weights_of_record(prog5, ck5, prefix=PREFIX)
    assert rec5["complete"] is False and any(u["field"] == L5 + "k_scale" and "2 elements" in u["why"] for u in rec5["unresolved"])
    # (4) [F-fp8-PIN2 on the remap branch] a bf16 model declares the placeholders Value<16> (KVCacheScaleParameter is torch.tensor(-1.0) at the
    # model dtype); the loader copies the checkpoint's f32 scale INTO that bf16 Parameter -> the served word is the scale cast RNE to bf16, and
    # the record composes it the same way (never refuses an honest shipped scale for its width; never pins the f32 word a bf16 model cannot serve)
    prog6, fields6, L6, ck6 = _quantized_prog_and_ck(t, tmp_path, {"model.layers.0.self_attn.k_scale": f32(0.1), "model.layers.0.self_attn.v_scale": f32(0.125)}, "bf16m", placeholder_bits=16)
    rec6 = WR.weights_of_record(prog6, ck6, prefix=PREFIX)
    assert rec6["complete"] is True and rec6["unpinned"] == []
    by6 = {f["field"]: f for f in rec6["fields"]}
    bf16_01 = np.array([0.1], dtype=np.float32).view(np.uint32)
    bf16_01 = (((bf16_01 + 0x7FFF + ((bf16_01 >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)                   # RNE f32 -> bf16 (0x3DCD)
    assert by6[L6 + "k_scale"]["digest"] == hashlib.sha256(bf16_01.tobytes()).hexdigest() and by6[L6 + "k_scale"]["cast"] == "F32->bf16 (RNE)"
    assert by6[L6 + "v_scale"]["digest"] == hashlib.sha256(np.array([0x3E00], dtype=np.uint16).tobytes()).hexdigest()   # 0.125 exact in bf16
    assert by6[L6 + "q_scale"]["digest"] == hashlib.sha256(np.array([0xBF80], dtype=np.uint16).tobytes()).hexdigest()   # the bf16 sentinel (PIN2)
    assert "Value<16>" in by6[L6 + "k_scale"]["rule"]
    served6 = {"model.layers.0.self_attn.attn.k_scale": bf16_01.reshape(()), "model.layers.0.self_attn.attn.v_scale": np.array(0x3E00, dtype=np.uint16),
               "model.layers.0.self_attn.attn.q_scale": np.array(0xBF80, dtype=np.uint16), "model.layers.0.self_attn.attn.prob_scale": np.array(0xBF80, dtype=np.uint16)}
    served6.update({k: v for k, v in _served(t, fields6).items() if k not in served6})
    assert WR.check(rec6, live_digests(served6))["result"] is True
    bad6 = dict(served6)
    bad6["model.layers.0.self_attn.attn.k_scale"] = np.array(0xBF80, dtype=np.uint16)                            # the sentinel where the checkpoint ships a scale
    res6 = WR.check(rec6, live_digests(bad6))
    assert res6["result"] is False and [m["served_name"] for m in res6["mismatched"]] == ["model.layers.0.self_attn.attn.k_scale"]
    # the source table itself
    assert WR.kv_scale_remap_sources("model.layers.3.self_attn.attn.v_scale") == ["model.layers.3.self_attn.v_scale", "model.layers.3.self_attn.v_proj.v_scale",
                                                                                   "model.layers.3.self_attn.v_proj.output_scale", "model.layers.3.self_attn.kv_scale"]
    assert WR.kv_scale_remap_sources("model.layers.3.self_attn.attn.prob_scale")[-1] == "model.layers.3.self_attn.prob_proj.output_scale"
    assert WR.kv_scale_remap_sources("model.layers.3.self_attn.attn._k_scale") == []


def test_w2_record_is_bound_to_the_program_and_checkpoint_of_record(checkpoint):
    """[rev-cb M-0655 W2] a record built for another Program (or with edited digests) or another checkpoint entry is refused by name;
    Program digests are compared whole: a prefix of the digest of record, on either side, is refused by name."""
    d, man, t = checkpoint
    prog, fields = _program()
    rec = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)
    live = live_digests(_served(t, fields))
    ok = WR.check(rec, live, program_digests_of_record={"d" * 128}, checkpoint_of_record={"repo": "test/tiny", "revision": "abc"})
    assert ok["result"] is True and ok["program_of_record"] is True and ok["checkpoint_entry_of_record"] is True
    for short in (WR.check(rec, live, program_digests_of_record={"d" * 16}), WR.check(dict(rec, program_digest="d" * 16), live, program_digests_of_record={"d" * 128})):
        assert short["result"] is False and short["program_of_record"] is False and "not full sha512" in short["why"] and "d" * 16 in short["why"]
    bad = WR.check(rec, live, program_digests_of_record={"e" * 128})
    assert bad["result"] is False and bad["program_of_record"] is False and "!= Program of record" in bad["why"]
    edited = dict(rec, program_digest="e" * 128)
    assert WR.check(edited, live, program_digests_of_record={"d" * 128})["result"] is False
    other = WR.check(rec, live, checkpoint_of_record={"repo": "other/model", "revision": "abc"})
    assert other["result"] is False and "!= the workload's model other/model" in other["why"]
    rev = WR.check(rec, live, checkpoint_of_record={"repo": "test/tiny", "revision": "ffffffffff"})
    assert rev["result"] is False and "revision" in rev["why"]


def test_gemma_normalizer_is_composed_from_config_and_compared_like_a_weight(checkpoint):
    """dense (MAN-09 preflight on the real Gemma-2-2B Program, 316 fields): `model.model.normalizer` -- vLLM's non-persistent 0-d bf16
    buffer torch.tensor(hidden_size ** 0.5) -- has no checkpoint tensor and was the ONE unresolved field (complete False -> every Gemma
    Commit would FAIL 'weights of record' closed).  Rule: composed from config.json's hidden_size by the frozen analytic rule
    (root_policy ANALYTIC_TABLES['normalizer'] / analytic.normalizer_word: sqrt(H) rounded once to bf16), pinned and compared."""
    from verity_vllm.program.model import normalizer_word
    assert normalizer_word(2304) == 0x4240                              # Gemma-2-2B: sqrt(2304) = 48.0 exactly, bf16 0x4240
    d, man, t = checkpoint
    prog, fields = _program()
    prog["instances"]["params"] = [["weights", "{model.model.embed_tokens.weight: Array<%d,Array<%d,Value<16>>>, model.model.normalizer: Value<16>, "
                                    "model.model.norm.weight: Array<%d,Value<16>>}" % (V, H, H)]]
    rec = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)
    assert rec["complete"] is True and rec["unresolved"] == [] and rec["unpinned"] == []
    by = {f["field"]: f for f in rec["fields"]}
    nf = by["model.model.normalizer"]
    assert nf["pinned"] is True and nf["served_name"] == "model.normalizer" and nf["sources"] == ["config.json:hidden_size"]
    assert nf["rule"].startswith("config-derived: bf16(sqrt(hidden_size=%d))" % H)
    word = np.array(normalizer_word(H), dtype=np.uint16)
    assert nf["digest"] == hashlib.sha256(word.tobytes()).hexdigest()
    # the served buffer (named_buffers(): model.normalizer, 0-d bf16) is compared field by field like a weight ...
    served = {"model.embed_tokens.weight": WR.to_served(t["model.embed_tokens.weight"][1], "BF16", 16)[0],
              "model.normalizer": word, "model.norm.weight": WR.to_served(t["model.norm.weight"][1], "BF16", 16)[0]}
    ok = WR.check(rec, live_digests(served))
    assert ok["result"] is True and ok["compared"] == 3, ok
    # ... and a served normalizer that is not bf16(sqrt(H)) -- one bit off -- fails BY NAME (the value is pinned, not merely listed)
    bad = dict(served, **{"model.normalizer": np.array(normalizer_word(H) ^ 0x0001, dtype=np.uint16)})
    res = WR.check(rec, live_digests(bad))
    assert res["result"] is False and [m["served_name"] for m in res["mismatched"]] == ["model.normalizer"]
    # a normalizer declared at another width has no rule: unresolved, named, complete False (fail closed)
    prog["instances"]["params"] = [["weights", "{model.model.normalizer: Value<32>, model.model.norm.weight: Array<%d,Value<16>>}" % H]]
    rec2 = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man), verify_shards=False), prefix=PREFIX)
    assert rec2["complete"] is False and rec2["unresolved"][0]["field"] == "model.model.normalizer" and "one bf16 word" in rec2["unresolved"][0]["why"]


def test_f32_to_bf16_cast_is_round_to_nearest_even():
    x = np.array([1.0, 1.00390625, 1.01171875, -2.5, 0.0, 3.0e38], dtype=np.float32)   # 1+2^-8 ties to even (down), 1+3*2^-8 ties up
    got = WR._f32_to_bf16_bits(x)
    exp = np.array([0x3F80, 0x3F80, 0x3F82, 0xC020, 0x0000, 0x7F62], dtype=np.uint16)     # 3.0e38 = 0x7F61B1E6: low half > 0x8000 rounds up
    assert got.tolist() == exp.tolist()


def test_verdict_requires_the_pin_on_a_row_record_and_fails_by_name():
    """commit_verdict: a served weight that differs from the checkpoint of record is a named FAIL at every B; on a row-procedure record
    (binding block) an absent / not-evaluated pin is 'weights unpinned' (fail closed); legacy records without a binding block are untouched."""
    from verity_vllm.check.verdict import commit_verdict
    bnd = {"context": {"ok": True}, "coverage": {"ok": True}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": 2}}
    base = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64},
            "value_correspondence": {"method": "sampled-exact-replay(complete) + boundary-linkage", "result": True},
            "sampled_replay": {"result": True, "grade": "complete"}, "boundary_linkage": {"result": True, "pairs": 6, "linked": 6}}
    ok_wp = {"result": True, "compared": 98, "mismatched": [], "missing": []}
    ok_, why_ = commit_verdict({**base, "binding": bnd, "weights_pin": ok_wp}, True, min_openings=64)
    assert ok_ is True and why_.startswith("PASS"), why_                        # [rev M-0243] at B>=2 without form (B) the PASS names the attribution gap
    bad = {"result": False, "compared": 98, "mismatched": [{"served_name": "model.layers.5.mlp.down_proj.weight"}],
           "why": "1 served weight(s) differ from the checkpoint of record: model.layers.5.mlp.down_proj.weight"}
    for b in ({"binding": bnd}, {}):                      # named at every B, with or without the binding block
        ok, why = commit_verdict({**base, **b, "weights_pin": bad}, True, min_openings=64)
        assert not ok and why.startswith("weights of record:") and "model.layers.5.mlp.down_proj.weight" in why
    ok, why = commit_verdict({**base, "binding": bnd}, True, min_openings=64)
    assert not ok and why.startswith("weights unpinned (MAN-09)") and "not run" in why
    ok, why = commit_verdict({**base, "binding": bnd, "weights_pin": {"result": None, "why": "weights of record not derived: FileNotFoundError(...)"}}, True, min_openings=64)
    assert not ok and why.startswith("weights unpinned (MAN-09)") and "not derived" in why
    # a legacy record (no binding block) without a pin keeps the legacy rule
    assert commit_verdict(base, True, min_openings=64) == (True, "PASS")


def test_verdict_reads_a_components_only_replay_by_its_method_rel_m0621():
    """[rel M-0621] compiled S3 writes its form-(A) replay under value_correspondence.components and leaves the top level empty: the O1 gate
    reads the components where the top level is absent -- only a component that says it is a sampled exact replay counts; another method
    (an oracle compare, a root compare) never stands in for the replay of record."""
    from verity_vllm.check.verdict import commit_verdict
    bnd = {"context": {"ok": True}, "coverage": {"ok": True}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": 1}}
    base = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64}, "binding": bnd,
            "weights_pin": {"result": True, "compared": 10}}
    comps_ok = {"sampled_replay": {"result": True, "grade": "complete", "method": "sampled-exact-replay(kernel launches)"},
                "boundary_linkage": {"result": True, "pairs": 4, "linked": 4}}
    assert commit_verdict({**base, "value_correspondence": {"method": "compiled", "result": True, "components": comps_ok}}, True, min_openings=64) == (True, "PASS")
    comps_bad = dict(comps_ok, sampled_replay={"result": False, "grade": "mismatch", "method": "sampled-exact-replay(kernel launches)", "why": "launch 7 output != committed"})
    ok, why = commit_verdict({**base, "value_correspondence": {"method": "compiled", "result": False, "components": comps_bad}}, True, min_openings=64)
    assert not ok and "launch 7" in why                   # named (the composite's own why names the failing component's cause, R1)
    ok, why = commit_verdict({**base, "value_correspondence": {"method": "compiled", "result": None, "components": comps_bad}}, True, min_openings=64)
    assert not ok and why.startswith("C2 sampled replay:") and "launch 7" in why
    comps_other = dict(comps_ok, sampled_replay={"result": True, "method": "oracle-compare"})          # not a replay: does not count
    ok, why = commit_verdict({**base, "value_correspondence": {"method": "compiled", "result": True, "components": comps_other}}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established")


def test_cli_writes_the_record_beside_the_program(checkpoint, tmp_path):
    d, man, t = checkpoint
    prog, fields = _program()
    import gzip
    pd = tmp_path / "build_request"
    pd.mkdir()
    with gzip.open(pd / "instances.json.gz", "wt") as f:
        json.dump(prog["instances"], f)
    json.dump(prog["result"], open(pd / "result.json", "w"))
    rc = cli.main(["weights-of-record", "--program-dir", str(pd), "--checkpoint-dir", str(d), "--checkpoints", str(man)])
    assert rc == 0
    rec = json.load(open(pd / "weights_of_record.json"))
    assert rec["complete"] is True and rec["program_digest"] == "d" * 128 and rec["wrapper_prefix"] == PREFIX


def _write_program(d, prog, digest, LP, T, extra_field=None):
    import gzip
    d.mkdir(parents=True)
    inst = json.loads(json.dumps(prog["instances"]))
    inst["digest"] = digest
    if extra_field:
        name, ty = extra_field
        inst["params"] = [[n, (s[:-1] + f", {name}: {ty}}}") if n == "weights" else s] for n, s in inst["params"]]
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump(inst, f)
    json.dump({"program_digest": digest, "ok": True, "LP": LP, "T": T}, open(d / "result.json", "w"))


def test_of_record_set_is_stamped_from_every_component_and_checked_by_set_equality_rev_cb_m0733(checkpoint, tmp_path):
    """[rev-cb M-0733 / hopper M-0723] --programs-root stamps the WHOLE of-record set (every component Program's digest + the workload
    digest + the manifest's), after checking each component's weights struct equals the record's; check() accepts iff the run's
    of-record set equals the stamp -- a record of another row (a missing or extra component), an unsound stamp (a component of another
    struct), or an un-stamped record built for a non-component all FAIL by name; the un-stamped single-digest rule still holds."""
    d, man, t = checkpoint
    prog, fields = _program()
    A, B, W = "a" * 128, "b" * 128, "f" * 128
    root = tmp_path / "row"
    _write_program(root / "build_request_LP4_T2", prog, A, 4, 2)
    _write_program(root / "build_request_LP9_T1", prog, B, 9, 1)
    _write_program(root / "build_request", prog, "d" * 128, 9, 2)             # the default wrapper shape: NOT a workload component
    (root / "build_workload").mkdir()
    json.dump({"workload_digest": W}, open(root / "build_workload" / "workload_program.json", "w"))
    json.dump({"program_digest": W, "component_digests": [A, B], "components": {"r0": {"program_digest": A, "LP": 4, "T": 2}, "r1": {"program_digest": B, "LP": 9, "T": 1}}},
              open(root / "manifest.json", "w"))
    # the CLI: derive from the first component, stamp the set, write beside the root
    out = tmp_path / "wor.json"
    rc = cli.main(["weights-of-record", "--programs-root", str(root), "--checkpoint-dir", str(d), "--checkpoints", str(man), "--out", str(out)])
    rec = json.load(open(out))
    assert rc == 0 and rec["of_record"] is True and rec["program_digest"] == A and rec["program_dir"].endswith("build_request_LP4_T2")
    ors = rec["of_record_set"]
    # the default-shape build_request is struct-checked but NOT of record (not a workload component): its digest never joins the set
    assert ors["program_digests"] == sorted({A, B, W}) and ors["workload_digest"] == W and ors["why"] is None
    assert [(c["dir"], c["of_record"], c["struct_equal"]) for c in ors["components"]] == \
        [("build_request_LP4_T2", True, True), ("build_request_LP9_T1", True, True), ("build_request", False, True)]
    live = live_digests(_served(t, fields))
    run_set = {W, A, B}                                    # commit_delta's: --program-digest (workload) + the manifest's component digests
    ok = WR.check(rec, live, program_digests_of_record=run_set)
    assert ok["result"] is True and ok["program_of_record"] is True and ok["of_record_set"] == sorted(run_set)
    # prefix-truncated run digests name no Program: refused by name, never matched to the stamp's full ones
    short = WR.check(rec, live, program_digests_of_record={x[:16] for x in run_set})
    assert short["result"] is False and short["program_of_record"] is False and "not full sha512" in short["why"]
    # the run's set differs: a component missing from the run, or a run digest the record never saw -> FAIL by name
    bad = WR.check(rec, live, program_digests_of_record={W, A})
    assert bad["result"] is False and "record's of-record set != the run's" in bad["why"] and B[:16] in bad["why"]
    bad = WR.check(rec, live, program_digests_of_record=run_set | {"e" * 128})
    assert bad["result"] is False and ("e" * 16) in bad["why"]
    # B1 shape of a row: only build_request under the root -> it IS of record (the manifest's program digest)
    r1 = tmp_path / "row_b1"
    _write_program(r1 / "build_request", prog, "d" * 128, 9, 2)
    json.dump({"program_digest": "d" * 128}, open(r1 / "manifest.json", "w"))
    rec1 = WR.stamp_of_record_set(WR.build_record(str(r1 / "build_request"), str(d), checkpoints_manifest=str(man)), str(r1))
    assert rec1["of_record"] is True and rec1["of_record_set"]["program_digests"] == ["d" * 128]
    assert WR.check(rec1, live, program_digests_of_record={"d" * 128})["result"] is True
    # an unsound stamp: a component Program of ANOTHER weights struct under the root -> record not of record, check FAILs by name
    _write_program(root / "build_request_LP2_T7", prog, "c" * 128, 2, 7, extra_field=("model.model.layers.2.mlp.down_proj.weight", "Array<8,Array<4,Value<16>>>"))
    rec2 = WR.stamp_of_record_set(WR.build_record(str(root / "build_request_LP4_T2"), str(d), checkpoints_manifest=str(man)), str(root))
    assert rec2["of_record"] is False and "build_request_LP2_T7" in rec2["of_record_why"] and "weights struct differs" in rec2["of_record_why"]
    assert "c" * 128 not in rec2["of_record_set"]["program_digests"]
    bad = WR.check(rec2, live, program_digests_of_record=run_set)
    assert bad["result"] is False and "of-record stamp is not sound" in bad["why"]
    # un-stamped record (the --program-dir path) keeps the single-digest rule: built for the default shape, refused when it is not a component
    rec3 = WR.build_record(str(root / "build_request"), str(d), checkpoints_manifest=str(man))
    assert "of_record_set" not in rec3
    bad = WR.check(rec3, live, program_digests_of_record={W, A, B})
    assert bad["result"] is False and bad["why"].startswith("record built for Program dddddddddddddddd != Program of record")
    assert WR.check(rec3, live, program_digests_of_record={W, A, B, "d" * 128})["result"] is True
    # a manifest carrying a truncated Program digest: the stamp is not of record, by name
    r2 = tmp_path / "row_short"
    _write_program(r2 / "build_request", prog, "d" * 128, 9, 2)
    json.dump({"program_digest": "d" * 16}, open(r2 / "manifest.json", "w"))
    rec4 = WR.stamp_of_record_set(WR.build_record(str(r2 / "build_request"), str(d), checkpoints_manifest=str(man)), str(r2))
    assert rec4["of_record"] is False and "not full sha512" in rec4["of_record_why"] and "d" * 16 in rec4["of_record_why"]
    # a root without any component Program is refused by the CLI
    assert cli.main(["weights-of-record", "--programs-root", str(tmp_path / "empty"), "--checkpoint-dir", str(d), "--checkpoints", str(man), "--out", str(tmp_path / "x.json")]) == 2


def test_cli_record_carries_its_derivation_provenance_and_names_what_it_supersedes(checkpoint, tmp_path, monkeypatch):
    """[COORD R16 M-0038 (2)] a record the CLI writes carries `derived` = {tool, code (git HEAD or null), at, sources (Program files, checkpoint
    dir, checkpoints.json), argv, supersedes}; check() ignores it (digests are the evidence).  argv is the command's own flags, as the
    process `verity-vllm weights-of-record ...` sees them."""
    from verity_vllm.pipeline import row_records as R

    d, man, t = checkpoint
    prog, fields = _program()
    pdir = tmp_path / "build_request_LP9_T2"
    _write_program(pdir, prog, "d" * 128, 9, 2)
    stale = pdir / "weights_of_record.stale-20260920T042250Z.json"
    stale.write_text(json.dumps({"schema": WR.SCHEMA_V1, "program_digest": "d" * 128, "complete": True, "unpinned": []}))
    out = pdir / "weights_of_record.json"
    argv = ["--program-dir", str(pdir), "--checkpoint-dir", str(d), "--checkpoints", str(man), "--out", str(out), f"--supersedes={stale}"]
    monkeypatch.setattr("sys.argv", ["verity-vllm", "weights-of-record", *argv])
    assert cli.main() == 0
    rec = json.load(open(out))
    dv = rec["derived"]
    assert dv["tool"] == "verity_vllm.check.weights_of_record" and dv["supersedes"] == str(stale) and dv["argv"] == argv
    assert dv["sources"]["program_dir"] == str(pdir) and dv["sources"]["program_files"] == ["instances.json.gz", "result.json"]
    assert dv["sources"]["checkpoint_dir"] == str(d) and dv["sources"]["checkpoints_manifest"] == str(man)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", dv["at"]) and (dv["code"] is None or re.fullmatch(r"[0-9a-f]{40}", dv["code"]))
    assert WR.check(rec, live_digests(_served(t, fields)), program_digests_of_record={"d" * 128})["result"] is True
    # the row's reuse rule accepts the freshly derived record (current schema, this Program)
    (pdir / "result.json").write_text(json.dumps({"ok": True, "program_digest": "d" * 128, "LP": 9, "T": 2}))
    assert R.wor_stale(str(out), str(pdir / "result.json"), WR.SCHEMA) == ""


def test_provenance_code_is_the_shipped_commit_on_a_pod_tree_and_git_head_in_a_git_tree(tmp_path):
    """[COORD M-0038 (2)] derived.code names the code that ran the derivation.  The release tree on a pod has NO .git (ship.sh writes
    EXPORT.json with the shipped commit): the int-1 wave-1 records carried code null (exec r16_0413).  Now: EXPORT.json's commit first
    (code_source 'EXPORT.json', the dirty-patch digest named when present), git HEAD in a git tree, null only when neither exists."""
    root = os.path.abspath(os.path.join(os.path.dirname(WR.__file__), "..", ".."))
    exp = os.path.join(root, "EXPORT.json")
    if os.path.exists(exp):
        pytest.skip("this tree carries an EXPORT.json already (a shipped pod tree)")
    dv = WR.derivation_provenance(str(tmp_path), str(tmp_path), None, ["x"])
    if dv["code"] is not None:                                         # a git tree (the laptop): HEAD, source git
        assert re.fullmatch(r"[0-9a-f]{40}", dv["code"]) and dv["code_source"] == "git"
    try:
        with open(exp, "w") as f:
            json.dump({"commit": "a" * 40, "patch_sha256": "deadbeefcafe0000"}, f)
        dv2 = WR.derivation_provenance(str(tmp_path), str(tmp_path), None, ["x"])
        assert dv2["code"] == "a" * 40 and dv2["code_source"] == "EXPORT.json (dirty patch deadbeefcafe)"
        with open(exp, "w") as f:
            json.dump({"commit": "b" * 40}, f)
        dv3 = WR.derivation_provenance(str(tmp_path), str(tmp_path), None, ["x"])
        assert dv3["code"] == "b" * 40 and dv3["code_source"] == "EXPORT.json"
    finally:
        os.unlink(exp)
