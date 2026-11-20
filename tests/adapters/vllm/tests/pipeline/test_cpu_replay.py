"""The CPU replay of a deferred Commit (`pipeline.cpu_replay`): the checkpoint's tensors opened against the committed weights root (a
flipped checkpoint byte fails by name), a bundle that is not the Commit record's refused before anything is replayed, and a decided
record read back from its slim keep as it awaited its bundle."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from verity.commitments import salt_file as SF

from verity_vllm.check import weights_of_record as WR
from verity_vllm.commit import step_rows
from verity_vllm.pipeline import cpu_replay as CPU
from verity_vllm.pipeline import replay_bundle as RB

GEO = bytes(range(64))
SALTS: dict[tuple[str, int], bytes] = {}
RNG = np.random.default_rng(7)
PARTS = {"model.layers.0.self_attn.q_proj.weight": RNG.integers(0, 1 << 16, (4, 8), dtype=np.uint16),
         "model.layers.0.self_attn.k_proj.weight": RNG.integers(0, 1 << 16, (2, 8), dtype=np.uint16),
         "model.layers.0.self_attn.v_proj.weight": RNG.integers(0, 1 << 16, (2, 8), dtype=np.uint16),
         "model.norm.weight": RNG.integers(0, 1 << 16, (8,), dtype=np.uint16)}
QKV = "model.layers.0.self_attn.qkv_proj.weight"
COS_SIN: np.ndarray = np.arange(8, dtype=np.float32)


def _shard(path, tensors):
    hdr, off, blobs = {}, 0, []
    for name, a in tensors.items():
        b = a.tobytes()
        hdr[name] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _served():
    qkv = np.concatenate([PARTS[f"model.layers.0.self_attn.{p}_proj.weight"] for p in "qkv"])
    return {QKV: ("bfloat16", qkv), "model.norm.weight": ("bfloat16", PARTS["model.norm.weight"]),
            "model.rotary_emb.cos_sin_cache": ("float32", COS_SIN), "model.half.weight": ("float16", np.zeros(4, np.float16)),
            "model.unknown.weight": ("bfloat16", np.zeros(4, np.uint16))}


def tensor_salts(name: str, nbytes: int) -> bytes:
    """A registered tensor's row salts, one fresh draw per name and size for the whole module."""
    return SALTS.setdefault((name, nbytes), step_rows.fresh_tensor_salts(nbytes))


def weights_tree(tensors: dict[str, tuple[str, np.ndarray]]) -> dict:
    """The registered tensors' weights tree as `register_weights` keeps it, under `tensor_salts`."""
    names, roots = [], []
    for i, (name, (dt, a)) in enumerate(tensors.items()):
        raw = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
        entry = [name, dt, list(a.shape), int(raw.size)]
        roots.append(step_rows.tensor_root(GEO, i, entry, raw, tensor_salts(name, int(raw.size))))
        names.append(entry)
    return step_rows.weights_tree(GEO, names, roots)


def write_salts(bundle: Path, tree: dict) -> None:
    """The registration's salt file beside the bundle's weights tree, as `replay_bundle.write_store_files` writes it."""
    SF.write(CPU.salts_path(bundle), {n: tensor_salts(n, int(nb)) for n, _dt, _s, nb in tree["names"] if int(nb)})


@pytest.fixture
def world(tmp_path):
    ckd, bundle = tmp_path / "ck", tmp_path / "bundle"
    ckd.mkdir()
    (ckd / "config.json").write_text("{}")
    _shard(ckd / "model.safetensors", PARTS)
    (bundle / "weights" / "extra").mkdir(parents=True)
    tree, extra = weights_tree(_served()), {}
    for i, (name, (_dt, a)) in enumerate(_served().items()):
        if name == "model.rotary_emb.cos_sin_cache":
            (bundle / "weights" / "extra" / f"{i}.bin").write_bytes(np.ascontiguousarray(a).tobytes())
            extra[name] = i
    tree["extra"] = extra
    write_salts(bundle, tree)
    return {"ck": ckd, "bundle": bundle, "tree": tree, "root": tree["root"], "other_root": "00" * (len(tree["root"]) // 2)}


def _open(w):
    return CPU.open_weights(w["bundle"], w["tree"], w["root"], WR.CheckpointIndex(str(w["ck"]), verify_shards=False), threads=2)


def test_every_servable_tensor_opens_against_the_committed_root_and_is_served_as_its_words(world):
    words, wc = _open(world)
    assert wc["equal"] is True and wc["why"] is None, wc
    assert wc["servable"] == 4 and wc["opened"] == 3 and wc["opened_by_source"] == {"checkpoint": 2, "bundle": 1}
    assert [u["name"] for u in wc["unopened"]] == ["model.unknown.weight"] and wc["mismatched_n"] == 0
    consumed: set[str] = set()
    rd = CPU.provider(words, "model.", consumed)
    assert np.array_equal(rd("model." + QKV, 0, 64), _served()[QKV][1].reshape(-1)) and rd("model." + QKV, 0, 65) is None
    assert np.array_equal(rd("model.rotary_emb.cos_sin_cache", 0, 8), COS_SIN.view(np.uint32))
    assert consumed == {QKV, "model.rotary_emb.cos_sin_cache"}
    assert rd("model.unknown.weight", 0, 1) is None and rd("model.half.weight", 0, 1) is None


def test_a_flipped_checkpoint_byte_fails_the_opening_by_the_tensors_name(world):
    shard = world["ck"] / "model.safetensors"
    b = bytearray(shard.read_bytes())
    b[-(PARTS["model.norm.weight"].nbytes + 3)] ^= 1       # inside v_proj, a part of qkv_proj
    shard.write_bytes(bytes(b))
    words, wc = _open(world)
    assert wc["equal"] is False and [m["name"] for m in wc["mismatched"]] == [QKV]
    assert QKV in wc["why"] and "committed weights root" in wc["why"] and QKV not in words


def test_a_changed_bundle_constant_fails_by_name(world):
    p = world["bundle"] / "weights" / "extra" / "2.bin"
    p.write_bytes(bytes([p.read_bytes()[0] ^ 1]) + p.read_bytes()[1:])
    _, wc = _open(world)
    assert wc["equal"] is False and wc["mismatched"] == [{"name": "model.rotary_emb.cos_sin_cache", "source": "bundle",
                                                          "why": "tensor root != the committed tensor root"}]


def test_a_tree_that_is_not_the_committed_root_opens_nothing_of_record(world):
    _, wc = CPU.open_weights(world["bundle"], world["tree"], world["other_root"], WR.CheckpointIndex(str(world["ck"]), verify_shards=False))
    assert wc["equal"] is False and "is not the committed weights root" in wc["why"]
    world["tree"]["tensor_roots"][1] = world["other_root"].replace("0", "1")
    _, wc = _open(world)
    assert wc["equal"] is False and "does not fold to its root" in wc["why"]


def test_only_a_rows_sha512_tree_under_a_rows_sha512_root_of_record_is_read(world):
    """A tree that names no scheme (one recorded before trees named theirs: vllm-v1's), or another one, is refused by name before
    anything opens, and so is a rows-sha512 tree under a 32-byte (SHA-256) root of record; `OnRead` refuses the same."""
    ck = WR.CheckpointIndex(str(world["ck"]), verify_shards=False)
    assert (CPU.tree_scheme(world["tree"]), CPU.root_scheme(world["root"]), CPU.root_scheme("ab" * 32)) == (step_rows.NAME, step_rows.NAME, None)
    for tree, why in (({k: v for k, v in world["tree"].items() if k not in ("scheme", "key")}, "the bundle's weights tree names no scheme"),
                      (world["tree"] | {"scheme": "vllm-v1"}, "the bundle's weights tree is vllm-v1's")):
        words, wc = CPU.open_weights(world["bundle"], tree, world["root"], ck)
        assert (words, wc["equal"], wc["opened"], wc["why"]) == ({}, False, 0, f"{why}: only a {step_rows.NAME} tree is read")
        with pytest.raises(ValueError, match=why):
            CPU.OnRead(world["bundle"], tree, world["root"], None)
    _words, wc = CPU.open_weights(world["bundle"], world["tree"], "ab" * 32, ck)
    assert wc["equal"] is False and wc["why"].startswith(f"the committed weights root is 32 bytes, not a {step_rows.NAME} root")
    with pytest.raises(ValueError, match="the committed weights root is 32 bytes"):
        CPU.OnRead(world["bundle"], world["tree"], "ab" * 32, None)


def test_a_rows_sha512_tree_names_its_scheme_and_a_sha512_root_and_its_tensor_roots_are_the_rows_trees():
    """Under rows-sha512 a tensor's root is the root of its rows' frame-v3-sha512 tree (hiding leaves over its zero-padded ROW_BYTES
    rows, row i under the salt file's 192 bytes i, under the tensor's domain), the tree holds no salt, and the weights root is 64
    bytes; other salts give other roots."""
    from verity.commitments.merkle import tree_levels

    a = RNG.integers(0, 256, step_rows.ROW_BYTES + 100, dtype=np.uint8)
    entry = ["model.x.weight", "uint8", [a.size], int(a.size)]
    dom = step_rows.weights_domain(GEO, 0, entry)
    rows = [a[:step_rows.ROW_BYTES].tobytes(), a[step_rows.ROW_BYTES:].tobytes() + bytes(step_rows.ROW_BYTES - 100)]
    ys = tensor_salts("model.x.weight", a.size)
    hs = [step_rows.hiding_leaf(r, ys[192 * i:192 * (i + 1)], 8 * step_rows.ROW_BYTES)[1] for i, r in enumerate(rows)]
    want = tree_levels(dom, [step_rows.leaf(dom, i, h) for i, h in enumerate(hs)])[-1][0]
    assert dom.count == 2 and len(ys) == 2 * step_rows.SALT_BYTES and step_rows.tensor_root(GEO, 0, entry, a, ys) == want
    tree = weights_tree({"model.x.weight": ("uint8", a)})
    assert (tree["scheme"], len(bytes.fromhex(tree["root"])), "key" in tree) == (step_rows.NAME, step_rows.ROOT_BYTES, False)
    assert CPU.folded_root(tree) == tree["root"] and tree["tensor_roots"] == [want.hex()]
    assert step_rows.tensor_root(GEO, 0, entry, a, bytes(len(ys))) != want
    assert step_rows.tensor_root(GEO, 1, entry, a, ys) != want, "each tensor has its own domain"
    with pytest.raises(ValueError, match="2 rows take 384 bytes of salts"):
        step_rows.tensor_root(GEO, 0, entry, a, ys[:192])


def test_on_read_opens_a_tensor_on_its_first_read_as_open_weights_does_and_refuses_one_by_name(world):
    words, _wc = _open(world)
    shard = world["ck"] / "model.safetensors"
    b = bytearray(shard.read_bytes())
    b[-(PARTS["model.norm.weight"].nbytes + 3)] ^= 1       # inside v_proj, a part of qkv_proj
    shard.write_bytes(bytes(b))
    lazy = CPU.OnRead(world["bundle"], world["tree"], world["root"], WR.CheckpointIndex(str(world["ck"]), verify_shards=False))
    assert set(lazy) == {n for n, *_ in world["tree"]["names"] if n != "model.half.weight"} and not lazy.words
    rd = CPU.provider(lazy, "model.", set())
    assert np.array_equal(rd("model.norm.weight", 0, 8), words["model.norm.weight"]) and set(lazy.words) == {"model.norm.weight"}
    with pytest.raises(LookupError, match=f"{QKV} \\(checkpoint\\) does not open .*tensor root != the committed tensor root"):
        rd("model." + QKV, 0, 1)
    with pytest.raises(ValueError, match="is not the committed weights root"):
        CPU.OnRead(world["bundle"], world["tree"], world["other_root"], None)
    world["tree"]["tensor_roots"][1] = world["other_root"].replace("0", "1")
    with pytest.raises(ValueError, match="does not fold to its root"):
        CPU.OnRead(world["bundle"], world["tree"], world["root"], None)


def _safetensors(path, tensors):
    """A safetensors file of `tensors` (name -> (safetensors dtype, array))."""
    hdr, off, blobs = {}, 0, []
    for name, (dt, a) in tensors.items():
        b = np.ascontiguousarray(a).tobytes()
        hdr[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_an_int32_table_opens_as_the_checkpoints_i64_words_in_range(tmp_path):
    """DeepSeek-V4's hash-routed layers register `ffn.gate.tid2eid` as int32 [V, 6], stored I64 in the checkpoint: it opens as the
    checkpoint's values in i32 words (the replay's `GatherI32x{V}` rows read it), and a value outside int32 leaves it unopened."""
    name = "model.layers.0.ffn.gate.tid2eid"
    table = RNG.integers(0, 384, (16, 6), dtype=np.int64)

    def opened(ck_table):
        ckd, bundle = tmp_path / "ck", tmp_path / "bundle"
        ckd.mkdir(exist_ok=True)
        (ckd / "config.json").write_text("{}")
        _safetensors(ckd / "model.safetensors", {name: ("I64", ck_table)})
        tree = weights_tree({name: ("int32", table.astype(np.int32))}) | {"extra": {}}
        write_salts(bundle, tree)
        return CPU.open_weights(bundle, tree, tree["root"], WR.CheckpointIndex(str(ckd), verify_shards=False), threads=1)

    words, wc = opened(table)
    assert wc["equal"] is True and wc["servable"] == wc["opened"] == 1, wc
    got = CPU.provider(words, "model.", set())("model." + name, 0, 96)
    assert got.dtype == np.uint32 and np.array_equal(got.view(np.int32), table.reshape(-1))
    wide = table.copy()
    wide[3, 2] = 1 << 40
    words, wc = opened(wide)
    assert name not in words and [u["name"] for u in wc["unopened"]] == [name] and "outside int32" in wc["unopened"][0]["why"]


# ------------------------------------------------------------------------------------------------ the bundle is the record's

def _deferred(tmp_path, steps=None, tree=None):
    """A deferred record and its sealed bundle; ``tree``: the bundle's weights tree (its ``extra`` the cos/sin cache)."""
    pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    from verity_vllm.commit import store_dump as SD

    com, rc = commit_steps(steps or {0: [("model.embed_tokens", bytes(range(256)) * 2)], 1: [("runner.sampler", b"\x01\x02\x03\x04")]})
    p = RB.partial(tmp_path, 1)
    SD.dump(com, rc, p / "store")
    (p / "weights" / "extra").mkdir(parents=True)
    tree = tree or weights_tree({"model.rotary_emb.cos_sin_cache": ("float32", COS_SIN)}) | {"extra": {"model.rotary_emb.cos_sin_cache": 0}}
    (p / "weights" / "tree.json").write_text(json.dumps(tree))
    write_salts(p, tree)
    for i in tree["extra"].values():
        (p / "weights" / "extra" / f"{i}.bin").write_bytes(COS_SIN.tobytes())
    (tmp_path / "binding_map_p1.json").write_text(json.dumps({"entries": []}))
    sr, lk = RB.deferred_records(RB.path(tmp_path, 1))
    row = {"pair": 1, "arm": "instrumented", "commit": {"run_root": rc.run_root.hex(), "commit_pass": None, "sampled_replay": sr, "boundary_linkage": lk}}
    m = RB.seal(tmp_path, 1, {"pair": 1, "run_root": rc.run_root.hex()}, tmp_path / "binding_map_p1.json")
    (tmp_path / "runs.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "verdict.json").write_text(json.dumps({"replay_deferred": {"bundles": [{"pair": 1, "manifest_sha256": m["sha256"]}]}}))
    return tmp_path


def test_the_bundle_of_record_opens_with_its_run(tmp_path):
    out = _deferred(tmp_path)
    rows, row, _verdict, _store, run = CPU.of_record(RB.open_bundle(RB.path(out, 1)), out)
    assert row["pair"] == 1 and run.run_root.hex() == row["commit"]["run_root"] and len(rows) == 1


@pytest.mark.parametrize("tamper,why", [
    (lambda o: (o / "verdict.json").write_text(json.dumps({"replay_deferred": {"bundles": [{"pair": 1, "manifest_sha256": "00"}]}})), "names no bundle"),
    (lambda o: (o / "runs.jsonl").write_text((o / "runs.jsonl").read_text().replace('"commit_pass": null', '"commit_pass": true')), "awaiting"),
    (lambda o: (o / "runs.jsonl").write_text(json.dumps({**json.loads((o / "runs.jsonl").read_text()), "commit": {
        **json.loads((o / "runs.jsonl").read_text())["commit"], "run_root": "cd" * 32}}) + "\n"), "run root of record"),
    (lambda o: (o / "binding_map_p1.json").write_text(json.dumps({"entries": [1]})), "binding map"),
])
def test_a_bundle_that_is_not_the_records_is_refused(tmp_path, tamper, why):
    out = _deferred(tmp_path)
    tamper(out)
    with pytest.raises(CPU.Refused, match=why):
        CPU.of_record(RB.open_bundle(RB.path(out, 1)), out)


def test_the_replay_takes_the_commits_workers_unless_told_otherwise():
    ctx = {"args": {"replay_workers": None, "programs_root": "x", "program_dir": "y"}}
    o = CPU.Options(bundle="b", row_dir="/row", checkpoint_dir="c")
    assert CPU.replay_args(o, ctx).replay_workers is None
    assert CPU.replay_args(o, {"args": {**ctx["args"], "replay_workers": 12}}).replay_workers == 12
    a = CPU.replay_args(CPU.Options(bundle="b", row_dir="/row", checkpoint_dir="c", replay_workers=4), ctx)
    assert a.replay_workers == 4 and a.program_dir == "/row/build_request" and ctx["args"]["replay_workers"] is None


STEPS = {0: [("model.embed_tokens", bytes(range(256)) * 20), ("model.layers.0.self_attn.qkv_proj", b"\x07" * 900)],
         1: [("model.embed_tokens", bytes(reversed(range(256))) * 3), ("runner.sampler", b"\x01\x02\x03\x04")],
         2: [("model.layers.0.mlp.down_proj", bytes(i % 251 for i in range(3000)))]}
PICKS = [(0, "model.layers.0.self_attn.qkv_proj", 10, 20), (1, "runner.sampler", 0, 4)]


def _kept(tmp_path, tree=None):
    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.commit import store_dump as SD

    out = _deferred(tmp_path, STEPS, tree)
    b = RB.open_bundle(RB.path(out, 1))
    _rows, _row, _verdict, store, run = CPU.of_record(b, out)
    with SD.touched(store) as t:
        want = {(s, n): OC.OpenedReader(store, run)(s, n, lo, hi) for s, n, lo, hi in PICKS}
    c = {"commit_pass": True, "sampled_replay": {"sample": {"picked": 2}, "programs": {"r0": {"dir": "/b", "program_digest": "ef" * 32}}}}
    plan = {"plan": True, "picked": 2, "picks": [["r0", s, 5 + s, n] for s, n, _lo, _hi in PICKS]}
    return out, b, store, run, t, want, c, plan


