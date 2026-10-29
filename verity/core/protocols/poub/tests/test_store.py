"""`store`: the registered store a pinned BF16 checkpoint alone fixes, and the verifier's binding of a registration to it.
On a small checkpoint: the arena is vLLM's 2-D weights in its module order with q/k/v and gate/up fused, each from a block
boundary; its payload root, recomputed from the weights' digests in worker processes, is `audit.payload_tree`'s; a hello
whose store is that store under the verifier's key is held and bound, and any other (another key, another checkpoint,
another layout, a store short of the arenas) is refused with why; a pin whose files aren't the checkpoint's is refused
before anything reads them; and core's `models.json` is read as its roles' pins. The sessions that run the binding end to
end are `benchmarks/poub_session/tests/test_store_layout.py`'s."""
from __future__ import annotations

import itertools
import json
import math
import random
import re
import struct
from pathlib import Path

import pytest

from verity.core.protocols.poub import audit, store
from verity.ml.circuits import codec as circuits_codec

ROLE = "synthetic-bf16"
NODE = b"test-node"
KEY = bytes(range(32))
B = store.BLOCK
TINY = {"layers": 1, "hidden": 16, "heads": 2, "kv_heads": 1, "intermediate": 32, "vocab": 64}
"""A checkpoint of 6 weights blocks an arena."""
MODELS_JSON = Path(__file__).resolve().parents[2] / "pouw" / "lean" / "check" / "models.json"


def checkpoint(directory: Path, *, layers: int = 2, hidden: int = 48, heads: int = 4, kv_heads: int = 2,
               intermediate: int = 80, vocab: int = 128, tied: bool = False, seed: int = 0,
               architecture: str = "Qwen2ForCausalLM", shards: int = 2) -> dict:
    """A small BF16 checkpoint in `directory`: config.json and a Qwen2's tensors, its norms and q/k/v biases among them,
    over `shards` safetensors files; its pin written beside it as pin.json: the pin."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    hd = hidden // heads
    shapes = {"model.embed_tokens.weight": (vocab, hidden), "model.norm.weight": (hidden,)}
    for i in range(layers):
        p = f"model.layers.{i}."
        shapes |= {p + "self_attn.q_proj.weight": (heads * hd, hidden), p + "self_attn.q_proj.bias": (heads * hd,),
                   p + "self_attn.k_proj.weight": (kv_heads * hd, hidden), p + "self_attn.k_proj.bias": (kv_heads * hd,),
                   p + "self_attn.v_proj.weight": (kv_heads * hd, hidden), p + "self_attn.v_proj.bias": (kv_heads * hd,),
                   p + "self_attn.o_proj.weight": (hidden, hidden), p + "mlp.gate_proj.weight": (intermediate, hidden),
                   p + "mlp.up_proj.weight": (intermediate, hidden), p + "mlp.down_proj.weight": (hidden, intermediate),
                   p + "input_layernorm.weight": (hidden,), p + "post_attention_layernorm.weight": (hidden,)}
    if not tied:
        shapes["lm_head.weight"] = (vocab, hidden)
    names = sorted(shapes)
    for s in range(shards):
        mine = names[s::shards]
        raws = {t: rng.randbytes(2 * math.prod(shapes[t])) for t in mine}
        header, at = {"__metadata__": {"format": "pt"}}, 0
        for t in mine:
            header[t] = {"dtype": "BF16", "shape": list(shapes[t]), "data_offsets": [at, at + len(raws[t])]}
            at += len(raws[t])
        h = json.dumps(header).encode()
        h += b" " * (-len(h) % 8)
        (directory / f"model-{s + 1:05d}-of-{shards:05d}.safetensors").write_bytes(
            struct.pack("<Q", len(h)) + h + b"".join(raws[t] for t in mine))
    (directory / "config.json").write_text(json.dumps({
        "architectures": [architecture], "num_hidden_layers": layers, "vocab_size": vocab, "tie_word_embeddings": tied}))
    files = {p.name: store.file_sha512(str(p)) for p in sorted(directory.iterdir())}
    pin = {"name": ROLE, "checkpoint": files, "checkpoint_sha512": store.checkpoint_sha512(files)}
    (directory / "pin.json").write_text(json.dumps(pin))
    return pin


@pytest.fixture
def ckpt(tmp_path) -> Path:
    checkpoint(tmp_path / "m", **TINY)
    return tmp_path / "m"


def raw(directory: Path, name: str) -> bytes:
    path, meta, base = store.tensors(directory, [p.name for p in directory.iterdir()])[name]
    lo, hi = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + lo)
        return f.read(hi - lo)


def binding(ckpt: Path, engines: int = 2, arena_blocks: int = 7, **kw) -> store.Binding:
    return store.Binding(ROLE, ckpt, ckpt / "pin.json", engines, arena_blocks, log=lambda m: None, **kw)


def root_of(s: store.Store, key: bytes = KEY) -> bytes:
    """`audit.payload_tree`'s root over the store's payloads, salted under `key` as the prover salts them."""
    salts = [store.store_salt(key, store.PAYLOAD_LABEL, j) for j in range(s.blocks)]
    return audit.payload_tree(NODE, s.payloads(), salts).commitment.root


def test_the_local_canonical_json_is_ml_circuits_codecs_on_the_shapes_pins_use():
    """This protocol may not import `verity.ml.circuits`, so `store.canonical_json` is its own: the same bytes as
    `codec.canonical_json` on a pin's checkpoint ({file: SHA-512 hex}), a pin, a models.json entry and core's
    `models.json` itself, whose every role's `checkpoint_sha512` it reproduces."""
    files = {"model-00002-of-00002.safetensors": "ab" * 64, "config.json": "cd" * 64, "tokenizer.json": "0f" * 64}
    pin = {"name": "QWEN25_32B_BF16", "checkpoint": files, "checkpoint_sha512": "12" * 64}
    entry = {"checkpoint_sha512": "34" * 64, "files": files, "recipe": "bf16/v1", "repo": "Qwen/Qwen2.5-32B",
             "revision": "1818d35814b8319459f4bd55ed1ac8709630f003"}
    models = json.loads(MODELS_JSON.read_text())
    for doc in (files, pin, entry, {"ROLE": entry}, models, {"é": [1, -2, None, True]}):
        assert store.canonical_json(doc) == circuits_codec.canonical_json(doc)
    assert models
    for role, e in models.items():
        assert store.checkpoint_sha512(e["files"]) == e["checkpoint_sha512"], role


@pytest.mark.parametrize("tied", [False, True])
def test_the_arena_is_vllms_weights_in_module_order_and_its_root_the_payload_trees(tmp_path, tied):
    """vLLM's module order at one GPU an engine: the embedding, each layer's fused qkv (q's, k's, then v's rows), o,
    fused gate_up and down, then the LM head unless it is tied; each from the block boundary after the one before,
    norms and biases left out, zeros between. The root `expected_root` builds from the weights' digests, in subtrees
    of any size and in worker processes, is `audit.payload_tree`'s over the store's payloads under the key."""
    pin = checkpoint(tmp_path / "m", tied=tied)
    d = tmp_path / "m"
    a = store.arena(d, pin["checkpoint"])
    names = ["model.embed_tokens.weight"] + [f"model.layers.{i}.{m}.weight" for i in range(2) for m in
                                             ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj",
                                              "mlp.down_proj")] + ([] if tied else ["lm_head.weight"])
    assert [e.name for e in a.entries] == names
    sizes = [e.nbytes for e in a.entries]
    assert [e.offset for e in a.entries] == list(itertools.accumulate((-(-n // B) * B for n in sizes[:-1]), initial=0))
    assert a.weights_blocks * B == a.entries[-1].offset + -(-sizes[-1] // B) * B
    qkv = a.entries[1]
    assert a.read(qkv.offset, qkv.nbytes) == b"".join(
        raw(d, f"model.layers.0.self_attn.{x}_proj.weight") for x in "qkv")
    assert a.read(qkv.offset + qkv.nbytes, -qkv.nbytes % B) == bytes(-qkv.nbytes % B)
    assert a.read(a.entries[3].offset, a.entries[3].nbytes) == b"".join(
        raw(d, f"model.layers.0.mlp.{x}_proj.weight") for x in ("gate", "up"))
    s = store.Store(a, 2, a.weights_blocks + 3, 5)
    want = root_of(s)
    store.weight_digests(a, tmp_path / "digests", 2)
    for workers, log2 in ((1, 16), (1, 2), (1, 0), (2, 2)):
        assert store.expected_root(s, NODE, KEY, tmp_path / "digests", workers, log2) == want
    assert store.expected_root(s, NODE, bytes(32), tmp_path / "digests") != want
    W, A = a.weights_blocks, s.arena_blocks
    assert s.regions() == [{"kind": "weights", "tier": "hbm", "first": 0, "blocks": W},
                           {"kind": "filler", "tier": "hbm", "first": W, "blocks": 3},
                           {"kind": "weights", "tier": "hbm", "first": A, "blocks": W},
                           {"kind": "filler", "tier": "hbm", "first": A + W, "blocks": 3},
                           {"kind": "filler", "tier": "host", "first": 2 * A, "blocks": 5}]


def test_a_checkpoint_the_layout_doesnt_cover_is_refused(ckpt, tmp_path):
    """A vocabulary vLLM would pad, an architecture whose module order the layout doesn't state, and arenas smaller
    than the weights have no store the checkpoint alone fixes."""
    pin = checkpoint(tmp_path / "pad", vocab=100)
    with pytest.raises(store.Refused, match="vocab_size 100 is not a multiple of 64"):
        store.arena(tmp_path / "pad", pin["checkpoint"])
    pin = checkpoint(tmp_path / "moe", architecture="MixtralForCausalLM")
    with pytest.raises(store.Refused, match="whose vLLM module order the layout states"):
        store.arena(tmp_path / "moe", pin["checkpoint"])
    a = store.arena_of(ckpt)
    with pytest.raises(store.Refused, match=f"the weights take {a.weights_blocks} blocks, more than an arena's"):
        store.Store(a, 1, a.weights_blocks - 1)


def test_a_hello_whose_store_is_the_pinned_models_is_bound_and_any_other_refused_with_why(ckpt, tmp_path):
    """`hold` against a hello's store, payload root and layout: the root of the right store salted under another key,
    another checkpoint's arenas of the same shape, a layout of other arenas, and a store short of the arenas are each
    refused with why and leave the binding unbound; the pinned model's store under the verifier's key is bound, with
    and without a layout, and its regions are the layout's."""
    b = binding(ckpt, workers=2)
    s = store.Store(b.arena, 2, 7, 2)
    assert b.arena.weights_blocks == 6 and s.blocks == 16
    layout = [(r["kind"], r["tier"], r["first"], r["blocks"]) for r in s.regions()]
    checkpoint(tmp_path / "other", seed=1, **TINY)
    other = store.Store(store.arena_of(tmp_path / "other"), 2, 7, 2)
    wider = [(r["kind"], r["tier"], r["first"], r["blocks"]) for r in store.Store(b.arena, 2, 8).regions()]
    cases = [
        (16, root_of(s, bytes(32)), layout,
         re.escape("is not the pinned model synthetic-bf16's store (2 arenas of 7 blocks, 6 of weights each, then 2 "
                   "host filler blocks) under this verifier's payload key")),
        (16, root_of(other), layout, "is not the pinned model synthetic-bf16's store"),
        (16, root_of(s), wider, "the hello's layout .* is not the pinned model synthetic-bf16's"),
        (12, root_of(s), None, "the hello's store is 12 blocks, fewer than the pinned model's 2 arenas of 7"),
    ]
    for blocks, root, lay, why in cases:
        with pytest.raises(store.Refused, match=why) as e:
            b.hold(NODE, blocks, root, KEY, lay)
        assert b.record()["bound"] is False, why
    with pytest.raises(store.Refused) as e:
        b.hold(NODE, 16, root_of(s, bytes(32)), KEY, layout)
    assert str(e.value).endswith("register the store `poub.store` lays out, salted with the key the "
                                 "verifier sent")
    assert b.hold(NODE, 16, root_of(s), KEY, layout) == tuple(layout)
    rec = b.record()
    assert rec["bound"] is True and rec["name"] == ROLE and rec["checkpoint_sha512"] == b.pin["checkpoint_sha512"]
    assert rec["expected_root"] == rec["payload_root"] == root_of(s).hex() and rec["payload_key"] == KEY.hex()
    assert (rec["host_blocks"], rec["blocks"], rec["weights_blocks"]) == (2, 16, 6)
    assert b.hold(NODE, 16, root_of(s), KEY, None) == tuple(layout)


def test_a_pin_whose_files_arent_the_checkpoints_is_refused_before_anything_reads_them(ckpt, tmp_path):
    """The binding holds every pinned file to its SHA-512 first: a pin naming another hash for a file, a file changed
    since its pin, a checkpoint_sha512 that isn't its files', another role's pin and a symlinked file are each refused
    with why; a registry's lines of the role (`models.jsonl`) are read as its pin."""
    pin = json.loads((ckpt / "pin.json").read_text())
    shard = sorted(f for f in pin["checkpoint"] if f.endswith(".safetensors"))[0]

    def pinned(files: dict, **over) -> Path:
        path = tmp_path / f"pin-{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps({"name": ROLE, "checkpoint": files,
                                    "checkpoint_sha512": store.checkpoint_sha512(files)} | over))
        return path

    def refused(pin_path: Path, role: str = ROLE) -> str:
        with pytest.raises(store.Refused) as e:
            store.Binding(role, ckpt, pin_path, 2, 7, log=lambda m: None)
        return str(e.value)

    other = {**pin["checkpoint"], shard: "ab" * 64}
    assert refused(pinned(other)) == (f"{ckpt / shard}: SHA-512 {pin['checkpoint'][shard][:16]}..., not the pin's "
                                      f"{'ab' * 8}... of {ROLE}")
    assert "is not the SHA-512 of its checkpoint's canonical JSON" in refused(
        pinned(pin["checkpoint"], checkpoint_sha512="cd" * 64))
    assert refused(ckpt / "pin.json", "llama31-8b") == f"the pin {ckpt / 'pin.json'} is {ROLE}'s, not --model llama31-8b's"
    registry = tmp_path / "models.jsonl"
    registry.write_text("".join(json.dumps(x) + "\n" for x in (
        {"model": "other", "checkpoint": other, "checkpoint_sha512": store.checkpoint_sha512(other)},
        {"model": ROLE, "checkpoint": pin["checkpoint"], "checkpoint_sha512": pin["checkpoint_sha512"]})))
    assert store.Binding(ROLE, ckpt, registry, 2, 7, log=lambda m: None).pin == pin
    with (ckpt / shard).open("r+b") as f:
        f.seek(-1, 2)
        last = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([last[0] ^ 1]))
    assert refused(ckpt / "pin.json").startswith(f"{ckpt / shard}: SHA-512 ")
    (ckpt / shard).rename(tmp_path / shard)
    (ckpt / shard).symlink_to(tmp_path / shard)
    assert "not a regular file (no symlink)" in refused(ckpt / "pin.json")


@pytest.fixture
def models_json(ckpt, tmp_path) -> Path:
    """Core's pin in its shape (`verity/core/protocols/pouw/lean/check/models.json`, {ROLE: {checkpoint_sha512, files,
    recipe, repo, revision}}), pinning the synthetic checkpoint as SYNTHETIC_BF16 beside another role."""
    pin = json.loads((ckpt / "pin.json").read_text())
    other = {**pin["checkpoint"], "config.json": "ab" * 64}
    doc = {role: {"checkpoint_sha512": store.checkpoint_sha512(files), "files": files, "recipe": "bf16/v1",
                  "repo": f"example/{role}", "revision": "0" * 40}
           for role, files in (("SYNTHETIC_BF16", pin["checkpoint"]), ("OTHER_BF16", other))}
    path = tmp_path / "models.json"
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return path


def test_cores_models_json_is_read_as_its_roles_pin_held_to_the_same_checks(ckpt, models_json, tmp_path):
    """`read_pin` reads core's `models.json` as its entry of the role, named the role, its `files` the checkpoint, so
    the verifier binds to core's pin itself (the auditor host's `--model`); a role the file lacks is refused with the
    roles it has, and an entry is held to what a pin is."""
    pin = json.loads((ckpt / "pin.json").read_text())
    want = {"name": "SYNTHETIC_BF16", "checkpoint": pin["checkpoint"], "checkpoint_sha512": pin["checkpoint_sha512"]}
    assert store.read_pin(models_json, "SYNTHETIC_BF16") == want
    assert store.Binding("SYNTHETIC_BF16", ckpt, models_json, 2, 7, log=lambda m: None).pin == want
    for role, named in (("LLAMA31_8B_INT8", '"LLAMA31_8B_INT8"'), (None, "null")):
        with pytest.raises(store.Refused) as e:
            store.read_pin(models_json, role)
        assert str(e.value) == f'the pin {models_json} pins no role {named}: its roles are ["OTHER_BF16","SYNTHETIC_BF16"]'
    with pytest.raises(store.Refused, match=re.escape(f"{ckpt / 'config.json'}: SHA-512 ")):
        store.Binding("OTHER_BF16", ckpt, models_json, 2, 7, log=lambda m: None)

    doc = json.loads(models_json.read_text())

    def refused(entry: dict) -> str:
        path = tmp_path / f"models-{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps({**doc, "SYNTHETIC_BF16": entry}))
        with pytest.raises(store.Refused) as e:
            store.read_pin(path, "SYNTHETIC_BF16")
        return str(e.value)

    mine = doc["SYNTHETIC_BF16"]
    assert "is not the SHA-512 of its checkpoint's canonical JSON" in refused({**mine, "checkpoint_sha512": "cd" * 64})
    shape = "is not {ROLE: {checkpoint_sha512, files: {file: SHA-512 hex}, ...}} with a file name (no directory) a file"
    for entry in ({k: v for k, v in mine.items() if k != "files"}, {**mine, "files": {}},
                  {**mine, "files": {**mine["files"], "../x": "ab" * 64}},
                  {**mine, "files": {**mine["files"], "config.json": "AB" * 64}}):
        assert shape in refused(entry)
    no_config = {f: h for f, h in mine["files"].items() if f != "config.json"}
    assert "pins no config.json or no .safetensors file" in refused(
        {**mine, "files": no_config, "checkpoint_sha512": store.checkpoint_sha512(no_config)})
