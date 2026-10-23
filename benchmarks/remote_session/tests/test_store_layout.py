"""`store_layout`: the store a pinned BF16 checkpoint alone fixes, which the verifier binds a registration to
(`verifier.py --model`). On a small synthetic checkpoint: the arena is the engine's packing of vLLM's 2-D weights in its
module order with q/k/v and gate/up fused, and its payload root, recomputed from the weights' digests in worker
processes, is `audit.payload_tree`'s; a prover that registers that store under the verifier's payload key is accepted
and bound, and one whose root is any other (its own salts, another checkpoint, another shape) is refused with the
reason before the registry draws anything; a pin whose files aren't the checkpoint's is refused before anything reads
them; and the paced line reports weights held and filler held apart."""
from __future__ import annotations

import json
import os
import random
import re
import socket
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

from verity.core.protocols.remote_bandwidth import audit  # noqa: E402

import engine  # noqa: E402
import prover  # noqa: E402
import store_layout as sl  # noqa: E402
import verifier  # noqa: E402
import wire  # noqa: E402
from test_session import P512, run  # noqa: E402

ROLE = "synthetic-bf16"
NODE = b"test-node"
B = sl.BLOCK
TINY = {"layers": 1, "hidden": 16, "heads": 2, "kv_heads": 1, "intermediate": 32, "vocab": 64}
"""A checkpoint of 6 weights blocks an arena, so two arenas of 7 and 2 host blocks make `test_session`'s 16-block store
(the prover's encoding is most of a session's time)."""


@pytest.fixture
def ckpt(tmp_path) -> Path:
    sl.synthetic(tmp_path / "m", name=ROLE, **TINY)
    return tmp_path / "m"


def raw(directory: Path, name: str) -> bytes:
    path, meta, base = sl.tensors(directory, os.listdir(directory))[name]
    lo, hi = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + lo)
        return f.read(hi - lo)


def binding(ckpt: Path, engines: int = 2, arena_blocks: int = 7, **kw) -> sl.Binding:
    return sl.Binding(ROLE, ckpt, ckpt / "pin.json", engines, arena_blocks, log=lambda m: None, **kw)


@pytest.mark.parametrize("tied", [False, True])
def test_the_arena_is_the_engines_packing_of_vllms_weights_and_its_root_the_payload_trees(tmp_path, tied):
    """vLLM's module order at one GPU an engine: the embedding, each layer's fused qkv (q's, k's, then v's rows), o,
    fused gate_up and down, then the LM head unless it is tied; each from a block boundary as `engine.pack` places
    them, norms and biases left out, zeros between. The root `expected_root` builds from the weights' digests, in
    subtrees of any size and in worker processes, is `audit.payload_tree`'s over the store's payloads under the key."""
    pin = sl.synthetic(tmp_path / "m", name=ROLE, tied=tied)
    d = tmp_path / "m"
    a = sl.arena(d, pin["checkpoint"])
    names = ["model.embed_tokens.weight"] + [f"model.layers.{i}.{m}.weight" for i in range(2) for m in
                                             ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj",
                                              "mlp.down_proj")] + ([] if tied else ["lm_head.weight"])
    assert [e.name for e in a.entries] == names
    sizes = [e.nbytes for e in a.entries]
    assert [e.offset for e in a.entries] == engine.pack(sizes, a.weights_blocks * B)
    assert a.weights_blocks * B == a.entries[-1].offset + -(-sizes[-1] // B) * B
    qkv = a.entries[1]
    assert a.read(qkv.offset, qkv.nbytes) == b"".join(
        raw(d, f"model.layers.0.self_attn.{x}_proj.weight") for x in "qkv")
    assert a.read(qkv.offset + qkv.nbytes, -qkv.nbytes % B) == bytes(-qkv.nbytes % B)
    assert a.read(a.entries[3].offset, a.entries[3].nbytes) == b"".join(
        raw(d, f"model.layers.0.mlp.{x}_proj.weight") for x in ("gate", "up"))
    store = sl.Store(a, 2, a.weights_blocks + 3, 5)
    key = bytes(range(32))
    salts = [sl.store_salt(key, sl.PAYLOAD_LABEL, j) for j in range(store.blocks)]
    want = audit.payload_tree(NODE, store.payloads(), salts).commitment.root
    sl.weight_digests(a, tmp_path / "digests", 2)
    for workers, log2 in ((1, 16), (1, 2), (1, 0), (2, 2)):
        assert sl.expected_root(store, NODE, key, tmp_path / "digests", workers, log2) == want
    assert sl.expected_root(store, NODE, bytes(32), tmp_path / "digests") != want
    W, A = a.weights_blocks, store.arena_blocks
    assert store.regions() == [{"kind": "weights", "tier": "hbm", "first": 0, "blocks": W},
                               {"kind": "filler", "tier": "hbm", "first": W, "blocks": 3},
                               {"kind": "weights", "tier": "hbm", "first": A, "blocks": W},
                               {"kind": "filler", "tier": "hbm", "first": A + W, "blocks": 3},
                               {"kind": "filler", "tier": "host", "first": 2 * A, "blocks": 5}]


def test_a_checkpoint_the_layout_doesnt_cover_is_refused(ckpt, tmp_path):
    """A vocabulary vLLM would pad, an architecture whose module order the layout doesn't state, and arenas smaller
    than the weights have no store the checkpoint alone fixes."""
    pin = sl.synthetic(tmp_path / "pad", vocab=100)
    with pytest.raises(sl.Refused, match="vocab_size 100 is not a multiple of 64"):
        sl.arena(tmp_path / "pad", pin["checkpoint"])
    pin = sl.synthetic(tmp_path / "moe", architecture="MixtralForCausalLM")
    with pytest.raises(sl.Refused, match="whose vLLM module order the layout states"):
        sl.arena(tmp_path / "moe", pin["checkpoint"])
    a = sl.arena_of(ckpt)
    with pytest.raises(sl.Refused, match=f"the weights take {a.weights_blocks} blocks, more than an arena's"):
        sl.Store(a, 1, a.weights_blocks - 1)


def test_a_store_registered_as_the_layout_lays_it_out_is_accepted_and_bound(ckpt):
    """The prover's store is the pinned checkpoint's (two arenas of weights and zero filler, then a host tier), salted
    under the payload key the verifier sent: the verifier recomputes the same root, binds the registration, takes its
    regions from the layout, and the honest session is accepted."""
    b = binding(ckpt, workers=2)
    store = sl.Store(b.arena, 2, 7, 2)
    got = {}
    (r,), (p,) = run("honest", store=store, binding=b, verdict="python",
                     then=lambda v: got.update(reg=v.registration, regions=v.regions, key=v.payload_key))
    assert r["accepted"] and p["accepted"]
    rec = b.record()
    assert rec["bound"] is True and rec["name"] == ROLE and rec["checkpoint_sha512"] == b.pin["checkpoint_sha512"]
    assert rec["expected_root"] == rec["payload_root"] == got["reg"].payload_root.hex()
    assert rec["payload_key"] == got["key"].hex() and rec["host_blocks"] == 2 and rec["blocks"] == 16
    assert got["regions"] == tuple((x["kind"], x["tier"], x["first"], x["blocks"]) for x in store.regions())


def greet(backend, b: sl.Binding | None, **kw) -> tuple[verifier.Verifier, ValueError | None, list[str]]:
    """A prover over `backend` greeting a verifier that binds to `b`: the verifier, its refusal, and the `bye` the prover
    read."""
    x, y = socket.socketpair()
    p = prover.ProverSide(wire.Conn(y), backend, NODE, prover.parse_plan("honest"), seed=11, log=lambda m: None, **kw)
    heard = []

    def prove() -> None:
        try:
            p.register(codec=P512)
        except wire.Closed as e:
            heard.append(str(e))

    th = threading.Thread(target=prove, daemon=True)
    th.start()
    v = verifier.Verifier(wire.Conn(x), source=os.urandom(32), windows=3, window_ns=20, samples=6, delta=2.0 ** -40,
                          jitter_ns=10 ** 9, log=lambda m: None, verdict="python", binding=b)
    refused = None
    try:
        v.register()
    except ValueError as e:
        refused = e
    th.join()
    v.stop()
    return v, refused, heard


def test_a_store_whose_root_isnt_the_pinned_models_is_refused_with_why_before_the_registry_draws(ckpt, tmp_path):
    """Each of: the right store salted with the prover's own key (`--seed-payload-key`), another checkpoint's arenas
    of the same shape, a store of other arenas, and a store short of the arenas. The verifier answers `bye` with the
    reason and the registry has drawn nothing for the node."""
    b = binding(ckpt)
    store = sl.Store(b.arena, 2, 7, 2)
    assert b.arena.weights_blocks == 6
    sl.synthetic(tmp_path / "other", name=ROLE, seed=1, **TINY)
    cases = [
        (prover.CpuBackend(16, random.Random(7), store=store), {"seed_payload_key": True},
         re.escape("is not the pinned model synthetic-bf16's store (2 arenas of 7 blocks, 6 of weights each, then 2 "
                   "host filler blocks) under this verifier's payload key")),
        (prover.CpuBackend(16, random.Random(7), store=sl.Store(sl.arena_of(tmp_path / "other"), 2, 7, 2)), {},
         "is not the pinned model synthetic-bf16's store"),
        (prover.CpuBackend(16, random.Random(7), store=sl.Store(b.arena, 2, 8)), {},
         "the hello's layout .* is not the pinned model synthetic-bf16's"),
        (prover.CpuBackend(12, random.Random(7)), {}, "the hello's store is 12 blocks, fewer than the pinned model's 2 "
                                                      "arenas of 7"),
    ]
    for backend, kw, why in cases:
        v, refused, heard = greet(backend, b, **kw)
        assert isinstance(refused, sl.Refused), why
        assert re.search(why, str(refused)), (why, str(refused))
        assert heard == [f"refused: {refused}"] and v.registry.drawn(NODE) == 0
        assert b.record()["bound"] is False
    assert "register the store `store_layout` lays out, salted with the key the verifier sent" in str(
        greet(cases[0][0], b, seed_payload_key=True)[1])
    v, refused, heard = greet(prover.CpuBackend(16, random.Random(7), store=store), b)
    assert refused is None and heard == [] and v.registry.drawn(NODE) == 1 and b.record()["bound"] is True


def test_a_pin_whose_files_arent_the_checkpoints_is_refused_before_anything_reads_them(ckpt, tmp_path):
    """The binding holds every pinned file to its SHA-512 first: a pin naming another hash for a file, a file changed
    since its pin, a checkpoint_sha512 that isn't its files', another role's pin and a symlinked file are each refused
    with why; a registry's lines of the role (`models.jsonl`) are read as its pin. `verifier.py --model` refuses the
    same before it listens, and keeps why in verifier.json."""
    pin = json.loads((ckpt / "pin.json").read_text())
    shard = sorted(f for f in pin["checkpoint"] if f.endswith(".safetensors"))[0]

    def pinned(files: dict, **over) -> Path:
        path = tmp_path / f"pin-{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps({"name": ROLE, "checkpoint": files, "checkpoint_sha512": sl.checkpoint_sha512(files)}
                                   | over))
        return path

    def refused(pin_path: Path, role: str = ROLE) -> str:
        with pytest.raises(sl.Refused) as e:
            sl.Binding(role, ckpt, pin_path, 2, 7, log=lambda m: None)
        return str(e.value)

    other = {**pin["checkpoint"], shard: "ab" * 64}
    assert refused(pinned(other)) == (f"{ckpt / shard}: SHA-512 {pin['checkpoint'][shard][:16]}..., not the pin's "
                                      f"{'ab' * 8}... of {ROLE}")
    assert "is not the SHA-512 of its checkpoint's canonical JSON" in refused(
        pinned(pin["checkpoint"], checkpoint_sha512="cd" * 64))
    assert refused(ckpt / "pin.json", "llama31-8b") == f"the pin {ckpt / 'pin.json'} is {ROLE}'s, not --model llama31-8b's"
    registry = tmp_path / "models.jsonl"
    registry.write_text("".join(json.dumps(x) + "\n" for x in (
        {"model": "other", "checkpoint": other, "checkpoint_sha512": sl.checkpoint_sha512(other)},
        {"model": ROLE, "checkpoint": pin["checkpoint"], "checkpoint_sha512": pin["checkpoint_sha512"]})))
    assert sl.Binding(ROLE, ckpt, registry, 2, 7, log=lambda m: None).pin == pin
    with (ckpt / shard).open("r+b") as f:
        f.seek(-1, 2)
        last = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([last[0] ^ 1]))
    assert refused(ckpt / "pin.json").startswith(f"{ckpt / shard}: SHA-512 ")
    (ckpt / shard).rename(tmp_path / shard)
    (ckpt / shard).symlink_to(tmp_path / shard)
    assert "not a regular file (no symlink)" in refused(ckpt / "pin.json")
    (ckpt / shard).unlink()
    (tmp_path / shard).rename(ckpt / shard)
    with (ckpt / shard).open("r+b") as f:
        f.seek(-1, 2)
        f.write(last)
    out = tmp_path / "out"
    sys_argv = sys.argv
    try:
        sys.argv = ["verifier.py", "--out", str(out), "--verdict", "python", "--model", ROLE, "--checkpoint", str(ckpt),
                    "--pin", str(pinned(other)), "--arena-blocks", "7", "--layout-workers", "1"]
        assert verifier.main() == 3
    finally:
        sys.argv = sys_argv
    doc = json.loads((out / "verifier.json").read_text())
    assert "not the pin's" in doc["refused"] and doc["binding"]["bound"] is False


@pytest.fixture
def models_json(ckpt, tmp_path) -> Path:
    """Core's pin in its shape (`verity/core/protocols/pouw/lean/check/models.json`, {ROLE: {checkpoint_sha512, files,
    recipe, repo, revision}}), pinning the synthetic checkpoint as SYNTHETIC_BF16 beside another role."""
    pin = json.loads((ckpt / "pin.json").read_text())
    other = {**pin["checkpoint"], "config.json": "ab" * 64}
    doc = {role: {"checkpoint_sha512": sl.checkpoint_sha512(files), "files": files, "recipe": "bf16/v1",
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
    assert sl.read_pin(models_json, "SYNTHETIC_BF16") == want
    assert sl.Binding("SYNTHETIC_BF16", ckpt, models_json, 2, 7, log=lambda m: None).pin == want
    for role in ("LLAMA31_8B_INT8", None):
        with pytest.raises(sl.Refused) as e:
            sl.read_pin(models_json, role)
        assert str(e.value) == f"the pin {models_json} pins no role {role!r}: its roles are ['OTHER_BF16', 'SYNTHETIC_BF16']"
    with pytest.raises(sl.Refused, match=re.escape(f"{ckpt / 'config.json'}: SHA-512 ")):
        sl.Binding("OTHER_BF16", ckpt, models_json, 2, 7, log=lambda m: None)

    doc = json.loads(models_json.read_text())

    def refused(entry: dict) -> str:
        path = tmp_path / f"models-{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps({**doc, "SYNTHETIC_BF16": entry}))
        with pytest.raises(sl.Refused) as e:
            sl.read_pin(path, "SYNTHETIC_BF16")
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
        {**mine, "files": no_config, "checkpoint_sha512": sl.checkpoint_sha512(no_config)})


def test_the_paced_line_reports_weights_held_and_filler_held_apart():
    """A stretch's credit is each kind held less its lost blocks: weights, and filler, which is the rest of HBM and the
    host tier, each in both forms; filler never counts toward the weights line."""
    N, b = 1000, 4096
    held = {"weights": 600 * b, "hbm": 700 * b, "host": 300 * b}
    mem = {"windows": [{"held": held, "average": {"held": held}} for _ in range(12)]}
    seen = [{"released_ns": 10 * g, "passes": 2, "arrivals_ns": [10 * g + 5, 10 * g + 10]} for g in range(12)]

    def line(wrong: float) -> dict:
        return verifier.paced_line(mem, seen, groups=4, wrong=wrong, blocks=N, rebuild_per_ns=0.0, chip_bytes=0,
                                   block_bytes=b, node_bytes=10 * N * b)

    whole = line(0.0)
    for form in ("time", "snapshot"):
        assert whole[form]["weights_bytes"] == 600 * b and whole[form]["filler_bytes"] == 400 * b
        assert whole[form]["share"] == 0.06 and whole[form]["filler_share"] == 0.04
    lossy = line(80.0)
    lost = lossy["snapshot"]["lost_blocks"] * b
    assert lost > 0 and lossy["snapshot"]["weights_bytes"] == pytest.approx(600 * b - lost)
    assert lossy["snapshot"]["filler_bytes"] == pytest.approx(400 * b - lost)
    assert lossy["stretches"][0][verifier.TIME_HELD]["credited"]["filler"] == pytest.approx(
        400 * b - lossy["stretches"][0][verifier.TIME_HELD]["lost_blocks"] * b)
