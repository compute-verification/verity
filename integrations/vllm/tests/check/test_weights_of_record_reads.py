"""The weights of record read only what a rank's record needs, and record exactly what the whole-tensor reads recorded.

`compose` reads a dim-0 shard (column-parallel fields, the fused experts' w13) as the rank's row range of each checkpoint tensor, a
positioned read of a contiguous byte sub-range; the TP driver hashes the checkpoint's shards once and hands every rank the pin; the
record reads the Program's header and its rows up to the first attention row.  GOLDEN holds the sha256 of each canonical record as the
whole-tensor implementation wrote it on the checkpoint below: BF16 and block-FP8 experts (the refined scale grid at
world 4), F32 parts cast to bf16, fused / 1:1 / tied / pre-fused (Phi-3) / replicated-kv-head fields, and a fused field whose part is
not divisible by the world (its named refusal is part of the record).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os

import numpy as np
import pytest

from verity_vllm.check import weights_of_record as WR
from verity_vllm.check import checkpoint_index as CI
from tests.check.test_weights_of_record import write_safetensors

H, I, HQ, HKV, D, V, E, G = 128, 128, 4, 2, 16, 16, 2, 64
WORLDS = ((1, 0), (2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3))

GOLDEN = {(1, 0): "f4518f5eda3988dd4aa3b171af0ce98af5dfa1251128015f06ffacdf386619b2", (2, 0): "36bd25a3de1c951cf97321842d36c757bc11a53a601117f0761c50988de41a78",
          (2, 1): "20526cdda1e6f7cfe8b3c075fadb4f21f71e99b58326658ead584bd7b6f60dc3", (4, 0): "d419a07f7155d5d32234dec22eec16ea20d6de75adfbec2907377bfa9520ce7c",
          (4, 1): "6beaec17c12b5921c77e49d3f296a4575f9edbb39a499a99ba85db4fc0165933", (4, 2): "ff03f4b80aec61f2f490aa5e38545b30837a3e5bcb997f5519947f4931f6bc68",
          (4, 3): "329657392c5d525b46be58ac7901f2785cacbce28b5d1529e12cbcea475223e3"}
GOLDEN_TRUNCATED = {(1, 0): "2ec259864224a851c9e37fa50579ebab9e181c0e670964f55bacc56a7ef67c2b", (2, 0): "237bf208e5dff6116bbffb07611477dc38b976b663990a11a2505e78ba0b27de",
                    (2, 1): "4d65663c05b0e4781e42200ee834a99415011a4191bd4396cf01f1dd2eb406bf", (4, 0): "f2e30f4cfc800ac30f1bd4a40811c1489e4c793da1944b21827d8ddab9b041b7",
                    (4, 1): "d45bb7be4b0a606b9531197781a15af48337ec979bc66768e139910a078a4376", (4, 2): "5f1a0e07040c72ba873562690ee7274d6188fc9611331f0c089fc46d428e6c72",
                    (4, 3): "f11be2d5058d8c979532415920118a841b4e10e59266a925fcf239b481bbe3e9"}
GOLDEN_BUILT = {(1, 0): "0cdcc81bcd857236d794826b8177127335b6fd193773682391a3265ac548a786", (2, 0): "c2ac89cbc7610ca1abb9a503f02270615a5b10a47f45a4e73fbb738957238740",
                (2, 1): "7b5e214c9497d3e4219e0268b606db5312e87568dca35e8dd76bb4bbeca002d3", (4, 0): "c80bca9dc4ee00d2c4846cff630a69b817fbc6fec2aced97d215569a4c20aec8",
                (4, 1): "6a6f5d3855a75b0dd7eaa0252a2afb30568de084d7ba643149e7ae969caac38a", (4, 2): "389d2e862c60792218e234143b4f07cc3b349e1d61264d0c769edaed5fd3ef64",
                (4, 3): "887977fb3928bf7bfae71e0af832ec0623d2f9227f53b2f2c2b8537cf2d98b73"}


def _bf16(rng, *shape) -> np.ndarray:
    return (rng.random(shape, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _e4m3(rng, *shape) -> np.ndarray:
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _f32(rng, *shape) -> np.ndarray:
    return rng.random(shape, dtype=np.float32)


def tensors() -> dict[str, tuple[str, np.ndarray]]:
    rng = np.random.default_rng(11)
    L0, L1, L2 = "model.layers.0.", "model.layers.1.", "model.layers.2."
    t = {"model.embed_tokens.weight": ("BF16", _bf16(rng, V, H)), "lm_head.weight": ("BF16", _bf16(rng, V, H)),
         "model.norm.weight": ("F32", _f32(rng, H))}
    for p, n in (("q", HQ), ("k", HKV), ("v", HKV)):
        t[L0 + f"self_attn.{p}_proj.weight"] = ("BF16", _bf16(rng, n * D, H))
    t[L0 + "self_attn.o_proj.weight"] = ("BF16", _bf16(rng, H, HQ * D))
    t[L0 + "mlp.gate_proj.weight"] = ("F32", _f32(rng, I, H))
    t[L0 + "mlp.up_proj.weight"] = ("F32", _f32(rng, I, H))
    t[L0 + "mlp.down_proj.weight"] = ("BF16", _bf16(rng, H, I))
    t[L0 + "input_layernorm.weight"] = ("F32", _f32(rng, H))
    t[L0 + "self_attn.q_a_proj.weight"] = ("BF16", _bf16(rng, 32, H))
    t[L0 + "self_attn.kv_a_proj_with_mqa.weight"] = ("BF16", _bf16(rng, 16, H))
    t[L0 + "self_attn.q_b_proj.weight"] = ("F8_E4M3", _e4m3(rng, 2 * G, H))
    t[L0 + "self_attn.q_b_proj.weight_scale_inv"] = ("F32", _f32(rng, 2, H // G))
    t[L0 + "indexer.wk.weight"] = ("BF16", _bf16(rng, 3, H))
    t[L0 + "indexer.weights_proj.weight"] = ("BF16", _bf16(rng, 5, H))
    t[L1 + "mlp.gate.weight"] = ("BF16", _bf16(rng, E, H))
    t[L1 + "mlp.shared_experts.gate_proj.weight"] = ("BF16", _bf16(rng, I, H))
    t[L1 + "mlp.shared_experts.up_proj.weight"] = ("BF16", _bf16(rng, I, H))
    for e in range(E):
        for p in ("gate_proj", "up_proj"):
            t[L1 + f"mlp.experts.{e}.{p}.weight"] = ("BF16", _bf16(rng, I, H))
        t[L1 + f"mlp.experts.{e}.down_proj.weight"] = ("BF16", _bf16(rng, H, I))
        for p in ("gate_proj", "up_proj"):
            t[L2 + f"mlp.experts.{e}.{p}.weight"] = ("F8_E4M3", _e4m3(rng, I, H))
            t[L2 + f"mlp.experts.{e}.{p}.weight_scale_inv"] = ("F32", _f32(rng, I // G, H // G))
        t[L2 + f"mlp.experts.{e}.down_proj.weight"] = ("F8_E4M3", _e4m3(rng, H, I))
        t[L2 + f"mlp.experts.{e}.down_proj.weight_scale_inv"] = ("F32", _f32(rng, H // G, I // G))
    t[L2 + "mlp.gate_up_proj.weight"] = ("BF16", _bf16(rng, 2 * I, H))          # stored pre-fused (Phi-3): split gate|up per rank
    return t


def write_checkpoint(d, *, truncate: int = 0) -> None:
    """Two shards (the pin is per shard), config.json with the block-FP8 quantization and the attention geometry.  The second shard
    ends with `lm_head.weight` (a dim-0 shard); `truncate` cuts that many bytes off its end."""
    t = tensors()
    names = [n for n in sorted(t) if n != "lm_head.weight"]
    os.makedirs(d, exist_ok=True)
    write_safetensors(os.path.join(d, "model-00001-of-00002.safetensors"), {n: t[n] for n in names[: len(names) // 2]})
    second = os.path.join(d, "model-00002-of-00002.safetensors")
    write_safetensors(second, {n: t[n] for n in names[len(names) // 2:] + ["lm_head.weight"]})
    if truncate:
        os.truncate(second, os.path.getsize(second) - truncate)
    json.dump({"tie_word_embeddings": False, "num_attention_heads": HQ, "num_key_value_heads": HKV, "hidden_size": H, "head_dim": D,
               "quantization_config": {"weight_block_size": [G, G]}}, open(os.path.join(d, "config.json"), "w"))


@pytest.fixture
def ckpt(tmp_path):
    d = str(tmp_path / "snap")
    write_checkpoint(d)
    man = tmp_path / "checkpoints.json"
    files = {fn: {"sha256": WR.sha256_file(os.path.join(d, fn))} for fn in sorted(os.listdir(d)) if fn.endswith(".safetensors")}
    json.dump({"checkpoints": [{"repo": "test/reads", "revision": "abc", "local_path": d, "files": files}]}, open(man, "w"))
    return d, str(man)


def fields(world: int) -> list[tuple[str, list[int], int]]:
    from verity_vllm.program.model import fp8_moe_block
    w = world
    g = fp8_moe_block(G, I // w, H)
    hkv_r = HKV // w if HKV >= w else 1
    M = "model.model."
    L0, L1, L2 = M + "layers.0.", M + "layers.1.", M + "layers.2."
    return [(M + "embed_tokens.weight", [V // w, H], 16), ("model.lm_head.weight", [V // w, H], 16), (M + "norm.weight", [H], 16),
            (L0 + "self_attn.qkv_proj.weight", [(HQ // w + 2 * hkv_r) * D, H], 16), (L0 + "self_attn.o_proj.weight", [H, HQ * D // w], 16),
            (L0 + "mlp.gate_up_proj.weight", [2 * I // w, H], 16), (L0 + "mlp.down_proj.weight", [H, I // w], 16),
            (L0 + "input_layernorm.weight", [H], 16), (L0 + "self_attn.fused_qkv_a_proj.weight", [48, H], 16),
            (L0 + "self_attn.q_b_proj.weight", [2 * G // w, H], 8), (L0 + "self_attn.q_b_proj.weight_scale_inv", [2 // w if w <= 2 else 1, H // G], 32),
            (L0 + "indexer.wk_weights_proj.weight", [8 // w, H], 16), (L0 + "self_attn.attn._k_scale", [], 32),
            (L1 + "mlp.gate.weight", [E, H], 16), (L1 + "mlp.shared_experts.gate_up_proj.weight", [2 * I // w, H], 16),
            (L1 + "mlp.experts.w13_weight", [E, 2 * I // w, H], 16), (L1 + "mlp.experts.w2_weight", [E, H, I // w], 16),
            (L2 + "mlp.experts.w13_weight", [E, 2 * I // w, H], 8), (L2 + "mlp.experts.w13_weight_scale_inv", [E, 2 * I // (w * g), H // g], 32),
            (L2 + "mlp.experts.w2_weight", [E, H, I // w], 8), (L2 + "mlp.experts.w2_weight_scale_inv", [E, H // g, I // (w * g)], 32),
            (L2 + "mlp.experts.routed_experts.w13_weight", [E, 2 * I // w, H], 8), (L2 + "mlp.gate_up_proj.weight", [2 * I // w, H], 16)]


def _ty(shape, bits) -> str:
    t = f"Value<{bits}>"
    for n in reversed(shape):
        t = f"Array<{n},{t}>"
    return t


def weights_param(world: int) -> list[str]:
    return ["weights", "{" + ", ".join(f"{n}: {_ty(s, b)}" for n, s, b in fields(world)) + "}"]


def program(world: int, rank: int) -> dict:
    tp = {"world": world, "rank": rank} if world > 1 else None
    return {"instances": {"rows": [], "params": [weights_param(world)], "digest": "d" * 128},
            "result": {"program_digest": "d" * 128, **({"tp": tp} if tp else {})}, "dir": "synthetic"}


def write_program_dir(root, world: int, rank: int) -> str:
    """A rank Program dir as the Build writes it: v2 rows (a stored parameter progression), an attention row whose module names the
    export wrapper against result.json's observed implementation paths."""
    tp = {"world": world, "rank": rank} if world > 1 else None
    rows = [{"i": 0, "spec": "Embedding_v1{V=16,H=128}", "form": "call", "name": "model.model.embed_tokens/embedding[0]",
             "args": [[["p", 1, 0, 1]], [["P", 0, 0, 4, 0, 8, 8, 1000]]]},
            {"i": 1, "spec": "Attention_v3{T=1,NH=2,KVH=1,D=16,BN=16}", "form": "call", "name": "model.model.layers.0.self_attn.attn/attn[t=0]",
             "args": [[["n", 0, 0, 4]], [["N", 0, 0, 4, 0, 0, 0, 2]]]}]
    d = os.path.join(str(root), f"build_request_w{world}r{rank}")
    os.makedirs(d, exist_ok=True)
    with gzip.open(os.path.join(d, "instances.json.gz"), "wt") as f:
        json.dump({"schema": "cb-a/instance-sequence/v2", "digest": "e" * 128, "params": [weights_param(world), ["prompt", "Array<4,Value<32>>"]],
                   "rows": rows}, f)
    json.dump({"program_digest": "e" * 128, "attention_impls_observed": {"model.layers.0.self_attn.attn": "FLASH_ATTN"}, **({"tp": tp} if tp else {})},
              open(os.path.join(d, "result.json"), "w"))
    return d


def canonical(rec: dict) -> str:
    """sha256 of the record minus what names the deriving host: its paths and wall time."""
    r = json.loads(json.dumps(rec, default=str))
    r.pop("wall_s", None)
    r.pop("program_dir", None)
    r.pop("derived", None)
    ck = r.get("checkpoint") or {}
    ck.pop("dir", None)
    ck.pop("manifest", None)
    return hashlib.sha256(json.dumps(r, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def records(d: str, man: str, **kw) -> dict[tuple[int, int], str]:
    out = {}
    for world, rank in WORLDS:
        ck = WR.CheckpointIndex(d, checkpoints_manifest=man, **kw)
        out[(world, rank)] = canonical(WR.weights_of_record(program(world, rank), ck, prefix="model."))
    return out


def test_every_rank_record_is_the_whole_read_record(ckpt):
    d, man = ckpt
    assert records(d, man) == GOLDEN


def test_a_truncated_shard_is_refused_with_the_whole_read_words_on_every_rank(tmp_path):
    """The last 1500 bytes of lm_head.weight are gone: ranks 0 and 1 of world 4 could read their rows, but the whole tensor is not on
    disk, so every rank's field is unresolved with the whole read's words (the record never depends on which rows a rank reads)."""
    d = str(tmp_path / "snap")
    write_checkpoint(d, truncate=1500)
    assert records(d, str(tmp_path / "absent.json")) == GOLDEN_TRUNCATED


def test_a_rank_given_the_drivers_pin_records_what_hashing_every_shard_itself_records(ckpt, tmp_path):
    d, man = ckpt
    pin = CI.shard_pin(d, checkpoints_manifest=man, model=None, revision=None)
    assert records(d, man, verify_shards=False, pin=pin) == GOLDEN
    for w, r in WORLDS:
        assert canonical(WR.build_record(write_program_dir(tmp_path, w, r), d, checkpoints_manifest=man, shard_pin=pin)) == GOLDEN_BUILT[(w, r)]
    # a substituted shard is still a named failure under the shared pin: the driver hashed the file that is on disk
    with open(os.path.join(d, "model-00001-of-00002.safetensors"), "r+b") as f:
        f.seek(-1, os.SEEK_END)
        b = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([b[0] ^ 1]))
    bad = CI.shard_pin(d, checkpoints_manifest=man, model=None, revision=None, workers=1)
    rec = WR.weights_of_record(program(1, 0), WR.CheckpointIndex(d, checkpoints_manifest=man, pin=bad), prefix="model.")
    assert rec["checkpoint"]["all_shards_pinned"] is False and rec["checkpoint"]["shards"]["model-00001-of-00002.safetensors"]["pinned"] is False
    assert WR.check(rec, {"params": {}})["why"].startswith("checkpoint shards are not the pinned ones (W1)")


def test_a_pin_of_other_shards_or_another_manifest_is_refused(ckpt, tmp_path):
    d, man = ckpt
    pin = CI.shard_pin(d, checkpoints_manifest=man, model=None, revision=None)
    other = dict(pin, shards={k: v for k, v in pin["shards"].items() if "00002" not in k})
    with pytest.raises(ValueError, match="pin covers shards"):
        WR.CheckpointIndex(d, checkpoints_manifest=man, pin=other)
    with pytest.raises(ValueError, match="pin was taken against manifest"):
        WR.CheckpointIndex(d, checkpoints_manifest=str(tmp_path / "other.json"), pin=pin)


def test_parallel_shard_hashing_is_the_serial_pin(ckpt):
    d, man = ckpt
    serial = WR.CheckpointIndex(d, checkpoints_manifest=man, workers=1).pin
    assert CI.shard_pin(d, checkpoints_manifest=man, model=None, revision=None, workers=4) == serial
    assert list(serial["shards"]) == sorted(serial["shards"])


def test_built_record_reads_the_program_header_and_is_the_full_load_record(ckpt, tmp_path):
    d, man = ckpt
    for w, r in WORLDS:
        assert canonical(WR.build_record(write_program_dir(tmp_path, w, r), d, checkpoints_manifest=man)) == GOLDEN_BUILT[(w, r)]
    assert WR.record_shard_pin(d, checkpoints_manifest=man) == WR.CheckpointIndex(d, checkpoints_manifest=man).pin
    # a driver that can't read the checkpoint hands the ranks no pin: each takes it itself, and its own failure is the named one
    assert WR.record_shard_pin(str(tmp_path / "absent"), checkpoints_manifest=man) is None
    with pytest.raises(FileNotFoundError):
        WR.build_record(write_program_dir(tmp_path, 1, 0), str(tmp_path / "absent"), checkpoints_manifest=man, shard_pin=None)


def test_program_head_is_the_loaded_program_with_its_rows_as_stored(tmp_path):
    from verity_vllm.query.manifest import format as MF
    pdir = write_program_dir(tmp_path, 2, 1)
    full, head = MF.load_program(pdir), WR.load_program_head(pdir)
    assert {k: v for k, v in head["instances"].items() if k != "rows"} == {k: v for k, v in full["instances"].items() if k != "rows"}
    assert (head["result"], head["artifact"], head["dir"]) == (full["result"], full["artifact"], full["dir"])
    with gzip.open(os.path.join(pdir, "instances.json.gz"), "rt") as f:
        stored = json.load(f)["rows"]
    assert list(head["instances"]["rows"]) == stored == list(head["instances"]["rows"])
    assert MF.wrapper_prefix_of(head["instances"]["rows"], ["model.layers.0.self_attn.attn"]) == "model."
    # a document whose rows precede its params is read whole
    with gzip.open(os.path.join(pdir, "instances.json.gz"), "wt") as f:
        json.dump({"rows": stored, "params": full["instances"]["params"], "digest": full["instances"]["digest"]}, f)
    head = WR.load_program_head(pdir)
    assert head["instances"]["params"] == full["instances"]["params"] and head["instances"]["rows"] == stored


def test_row_range_reads_are_the_slices_of_the_whole_tensor(ckpt):
    d, man = ckpt
    ck = WR.CheckpointIndex(d, verify_shards=False)
    for name in ("lm_head.weight", "model.layers.0.self_attn.q_b_proj.weight", "model.layers.0.mlp.gate_proj.weight", "model.norm.weight"):
        whole = ck.array(name)
        n = whole.shape[0]
        for lo, hi in ((0, n), (0, 1), (n // 2, n), (3, 3), (1, n - 1)):
            assert np.array_equal(ck.array(name, rows=(lo, hi)), whole[lo:hi])
    with pytest.raises(ValueError, match="rows"):
        ck.array("lm_head.weight", rows=(2, 1))
