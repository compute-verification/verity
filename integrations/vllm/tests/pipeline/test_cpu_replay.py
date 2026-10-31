"""The CPU replay of a deferred Commit (`pipeline.cpu_replay`): the checkpoint's tensors opened against the committed weights root (a
flipped checkpoint byte fails by name), a bundle that is not the Commit record's refused before anything is replayed, and a decided
record read back from its slim keep as it awaited its bundle."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verity_vllm.check import weights_of_record as WR
from verity_vllm.commit import scheme, step_rows
from verity_vllm.pipeline import cpu_replay as CPU
from verity_vllm.pipeline import replay_bundle as RB

CHUNK = 16
GEO = bytes(range(32))
WK = bytes(range(64, 128))
RNG = np.random.default_rng(7)
PARTS = {"model.layers.0.self_attn.q_proj.weight": RNG.integers(0, 1 << 16, (4, 8), dtype=np.uint16),
         "model.layers.0.self_attn.k_proj.weight": RNG.integers(0, 1 << 16, (2, 8), dtype=np.uint16),
         "model.layers.0.self_attn.v_proj.weight": RNG.integers(0, 1 << 16, (2, 8), dtype=np.uint16),
         "model.norm.weight": RNG.integers(0, 1 << 16, (8,), dtype=np.uint16)}
QKV = "model.layers.0.self_attn.qkv_proj.weight"
COS_SIN = np.arange(8, dtype=np.float32)


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


def weights_tree(tensors: dict[str, tuple[str, np.ndarray]], commit_scheme: str = step_rows.NAME) -> dict:
    """The registered tensors' weights tree as `register_weights` keeps it, under ``commit_scheme``."""
    names, roots = [], []
    for i, (name, (dt, a)) in enumerate(tensors.items()):
        raw = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
        entry = [name, dt, list(a.shape), int(raw.size)]
        roots.append(step_rows.tensor_root(GEO, i, entry, raw, WK) if commit_scheme == step_rows.NAME else CPU.tensor_root(raw, CHUNK))
        names.append(entry)
    if commit_scheme == step_rows.NAME:
        return step_rows.weights_tree(GEO, names, scheme.names_digest(names), roots, WK)
    return {"names": names, "tensor_roots": [r.hex() for r in roots], "root": scheme.weights_root(GEO, scheme.names_digest(names), roots).hex(),
            "chunk": CHUNK, "geo_digest": GEO.hex()}


@pytest.fixture(params=[step_rows.NAME, scheme.NAME])
def world(tmp_path, request):
    ckd, bundle = tmp_path / "ck", tmp_path / "bundle"
    ckd.mkdir()
    (ckd / "config.json").write_text("{}")
    _shard(ckd / "model.safetensors", PARTS)
    (bundle / "weights" / "extra").mkdir(parents=True)
    tree, extra = weights_tree(_served(), request.param), {}
    for i, (name, (_dt, a)) in enumerate(_served().items()):
        if name == "model.rotary_emb.cos_sin_cache":
            (bundle / "weights" / "extra" / f"{i}.bin").write_bytes(np.ascontiguousarray(a).tobytes())
            extra[name] = i
    tree["extra"] = extra
    return {"ck": ckd, "bundle": bundle, "tree": tree, "root": tree["root"], "other_root": "00" * (len(tree["root"]) // 2)}


def _open(w):
    return CPU.open_weights(w["bundle"], w["tree"], w["root"], WR.CheckpointIndex(str(w["ck"]), verify_shards=False), threads=2)


def test_every_servable_tensor_opens_against_the_committed_root_and_is_served_as_its_words(world):
    words, wc = _open(world)
    assert wc["equal"] is True and wc["why"] is None, wc
    assert wc["servable"] == 4 and wc["opened"] == 3 and wc["opened_by_source"] == {"checkpoint": 2, "bundle": 1}
    assert [u["name"] for u in wc["unopened"]] == ["model.unknown.weight"] and wc["mismatched_n"] == 0
    consumed = set()
    rd = CPU.provider(words, "model.", consumed)
    assert np.array_equal(rd("model." + QKV, 0, 64), _served()[QKV][1].reshape(-1)) and rd("model." + QKV, 0, 65) is None
    assert np.array_equal(rd("model.rotary_emb.cos_sin_cache", 0, 8), COS_SIN.view(np.uint32))
    assert consumed == {QKV, "model.rotary_emb.cos_sin_cache"}
    assert rd("model.unknown.weight", 0, 1) is None and rd("model.half.weight", 0, 1) is None


def test_the_weights_attest_counts_the_consumed_fields_and_attests_only_when_every_one_opened(world):
    words, wc = _open(world)
    c = {"weights_pin": {"result": True, "root_of_record": "ab" * 32}}
    b = {"manifest_sha256": "cd" * 32}
    a = CPU.weights_attest(c, wc, {QKV, "model.rotary_emb.cos_sin_cache"}, words, world["tree"]["extra"], b)
    assert a == {"pin": {"equal": True, "digest": "ab" * 32}, "root_openings": "2/2", "sources": {"checkpoint": 1, "bundle": 1},
                 "attests": CPU.ATTESTS, "replay": f"cpu, from bundle {'cd' * 32}"}
    a = CPU.weights_attest(c, wc, {QKV, "model.unknown.weight"}, words, world["tree"]["extra"], b)
    assert a["root_openings"] == "1/2" and a["attests"] is None, "a consumed field that did not open attests nothing"
    assert CPU.weights_attest(c, dict(wc, equal=False), {QKV}, words, world["tree"]["extra"], b)["attests"] is None


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


def test_a_tree_is_read_only_under_the_scheme_its_root_of_record_was_registered_under(world):
    """A vllm-v1 (SHA-256) weights tree is read only for a registration recorded under vllm-v1: under a SHA-512 root of record it is
    refused by name before anything opens, as a rows-sha512 tree is under a 32-byte root, and `OnRead` refuses the same."""
    have = CPU.tree_scheme(world["tree"])
    other = scheme.NAME if have == step_rows.NAME else step_rows.NAME
    root = "ab" * (step_rows.ROOT_BYTES if other == step_rows.NAME else 32)
    assert CPU.root_scheme(world["root"]) == have and CPU.root_scheme(root) == other
    words, wc = CPU.open_weights(world["bundle"], world["tree"], root, WR.CheckpointIndex(str(world["ck"]), verify_shards=False))
    assert wc["equal"] is False and wc["why"] == (f"the bundle's weights tree is {have}'s, and the committed weights root is {other}'s: "
                                                  f"a {have} tree is read only for a registration recorded under {have}")
    with pytest.raises(ValueError, match=f"weights tree is {have}'s, and the committed weights root is {other}'s"):
        CPU.OnRead(world["bundle"], world["tree"], root, None)


def test_a_rows_sha512_tree_names_its_scheme_and_a_sha512_root_and_its_tensor_roots_are_the_rows_trees():
    """Under rows-sha512 a tensor's root is the root of its rows' frame-v3-sha512 tree (`step_rows.levels` over its zero-padded
    ROW_BYTES rows, under the tensor's domain and salt key), and the weights root is 64 bytes; another key gives other roots."""
    a = RNG.integers(0, 256, step_rows.ROW_BYTES + 100, dtype=np.uint8)
    entry = ["model.x.weight", "uint8", [a.size], int(a.size)]
    dom = step_rows.weights_domain(GEO, 0, entry)
    rows = [a[:step_rows.ROW_BYTES].tobytes(), a[step_rows.ROW_BYTES:].tobytes() + bytes(step_rows.ROW_BYTES - 100)]
    lv = step_rows.levels(dom, rows, step_rows.tensor_key(WK, 0), 8 * step_rows.ROW_BYTES)
    assert dom.count == 2 and step_rows.tensor_root(GEO, 0, entry, a, WK) == lv[-1][0]
    tree = weights_tree({"model.x.weight": ("uint8", a)})
    assert (tree["scheme"], len(bytes.fromhex(tree["root"])), tree["key"]) == (step_rows.NAME, step_rows.ROOT_BYTES, WK.hex())
    assert CPU.folded_root(tree) == tree["root"] and tree["tensor_roots"] == [lv[-1][0].hex()]
    assert step_rows.tensor_root(GEO, 0, entry, a, bytes(64)) != lv[-1][0]
    assert step_rows.tensor_root(GEO, 1, entry, a, WK) != lv[-1][0], "each tensor has its own domain and salt key"


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


def test_an_int32_table_opens_as_the_checkpoints_i64_words_in_range(tmp_path):
    """DeepSeek-V4's hash-routed layers register `ffn.gate.tid2eid` as int32 [V, 6], stored I64 in the checkpoint: it opens as the
    checkpoint's values in i32 words (the replay's `GatherI32x{V}` rows read it), and a value outside int32 leaves it unopened."""
    from tests.pipeline.test_tp_replay_deferred import _safetensors

    name = "model.layers.0.ffn.gate.tid2eid"
    table = RNG.integers(0, 384, (16, 6), dtype=np.int64)

    def opened(ck_table):
        ckd, bundle = tmp_path / "ck", tmp_path / "bundle"
        ckd.mkdir(exist_ok=True)
        (ckd / "config.json").write_text("{}")
        _safetensors(ckd / "model.safetensors", {name: ("I64", ck_table)})
        raw = table.astype(np.int32).reshape(-1).view(np.uint8)
        names, roots = [[name, "int32", [16, 6], int(raw.size)]], [CPU.tensor_root(raw, CHUNK)]
        root = scheme.weights_root(GEO, scheme.names_digest(names), roots).hex()
        tree = {"names": names, "tensor_roots": [r.hex() for r in roots], "root": root, "chunk": CHUNK, "geo_digest": GEO.hex(), "extra": {}}
        return CPU.open_weights(bundle, tree, root, WR.CheckpointIndex(str(ckd), verify_shards=False), threads=1)

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
    tree = tree or {"names": [], "extra": {"model.rotary_emb.cos_sin_cache": 0}}
    (p / "weights" / "tree.json").write_text(json.dumps(tree))
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


def test_a_workers_baseline_is_the_word_replays_unless_measured():
    """The baseline `driver.fit_pool` adds to each pick's stated peak; the Boolean replay states its picks' and keeps the baseline."""
    from verity_vllm.pipeline import boolean_replay as BR
    assert CPU.worker_peak(BR.Options(bundle="b", row_dir="r", checkpoint_dir="c", out="o")) == 250 << 20
    assert CPU.worker_peak(CPU.Options(bundle="b", row_dir="r", checkpoint_dir="c", worker_peak_gb=1.5)) == int(1.5 * (1 << 30))


STEPS = {0: [("model.embed_tokens", bytes(range(256)) * 5), ("model.layers.0.self_attn.qkv_proj", b"\x07" * 900)],
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


def test_the_slim_keep_reopens_every_pick_against_the_run_root_and_nothing_else(tmp_path):
    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.commit import opened as OV
    from verity_vllm.commit import store_dump as SD

    out, b, store, run, t, want, c, plan = _kept(tmp_path)
    doc = CPU.keep(out, b, store, run, t, plan, c, opened=set())
    k = CPU.slim_path(out, 1)
    assert json.loads((k / CPU.SLIM_DOC).read_text()) == doc and not k.with_name(k.name + ".partial").exists()
    assert (doc["schema"], doc["pair"], doc["run_root"], doc["picked"], doc["picks"], doc["commit_pass"]) == (
        CPU.SLIM_SCHEMA, 1, run.run_root.hex(), 2, plan["picks"], True)
    assert doc["binding_map_sha256"] == b["manifest"]["files"]["binding_map.json"]["sha256"] and doc["bundle_manifest_sha256"] == b["manifest_sha256"]
    assert doc["programs"] == c["sampled_replay"]["programs"]
    on_disk = {x.relative_to(k).as_posix() for x in k.rglob("*") if x.is_file()} - {CPU.SLIM_DOC}
    assert set(doc["files"]) == on_disk and {"binding_map.json", "context.json", "weights/tree.json", "weights/extra/0.bin", "store/store.json"} <= on_disk
    assert all(RB._sha256(k / r) == f["sha256"] and (k / r).stat().st_size == f["bytes"] for r, f in doc["files"].items())
    bundled = sum(x.stat().st_size for x in (Path(b["path"]) / "store").rglob("*.bin"))
    assert sum(f["bytes"] for r, f in doc["files"].items() if r.endswith(".bin") and r.startswith("store/")) < bundled
    slim, srun = SD.load(k / "store")
    assert slim._run_root(srun) == run.run_root, "the slim store folds to the run root of record"
    rd = OC.OpenedReader(slim, srun)
    assert all(rd(s, n, lo, hi) == want[(s, n)] is not None for s, n, lo, hi in PICKS), "every pick re-opens against the run root"
    assert rd(2, "model.layers.0.mlp.down_proj", 0, 100) is None and rd(0, "model.embed_tokens", 0, 64) is None, "and nothing else"
    raw = k / "store" / "raw" / "s1.bin"
    raw.write_bytes(bytes(x ^ 1 for x in raw.read_bytes()))
    with pytest.raises(OV.OpeningNotVerified):
        OC.OpenedReader(*SD.load(k / "store"))(1, "runner.sampler", 0, 4)


def test_the_slim_keep_is_written_only_for_the_replay_s_own_draw(tmp_path):
    out, b, store, run, t, _want, c, plan = _kept(tmp_path)
    for bad in (dict(plan, picked=1), {k: v for k, v in plan.items() if k != "plan"}, dict(plan, picks=plan["picks"][:1])):
        assert CPU.keep(out, b, store, run, t, bad, c, opened=set()) is None
    assert CPU.keep(out, b, store, run, t, plan, {"sampled_replay": {"grade": "not-evaluated"}}, opened=set()) is None
    assert not list(out.glob("replay_slim_p*"))


def test_a_rows_sha512_keep_carries_only_the_salt_keys_of_the_tensors_its_replay_opened_never_the_registrations_key(tmp_path):
    """``Kw`` opens every weights leaf (each tensor's salt key derives from it), and a keep is read by later replays: no file of a
    rows-sha512 keep holds ``Kw`` or an unopened tensor's ``tensor_key(Kw, t)``, only those of the tensors the replay opened, with the
    bundle's bytes of those alone.  A replay from the keep opens the same tensors and reports the rest unopened, not mismatched."""
    tree = weights_tree(_served()) | {"extra": {"model.rotary_emb.cos_sin_cache": 2}}
    out, b, store, run, t, _want, c, plan = _kept(tmp_path, tree)
    names = [n for n, *_r in tree["names"]]
    assert WK.hex().encode() in (Path(b["path"]) / "weights" / "tree.json").read_bytes(), "the private bundle holds Kw"
    for opened in ({QKV, "model.rotary_emb.cos_sin_cache"}, {QKV}):
        CPU.keep(out, b, store, run, t, plan, c, opened=opened)
        k = CPU.slim_path(out, 1)
        held = [i for i, n in enumerate(names) if n in opened]
        secret = [WK, *(step_rows.tensor_key(WK, i) for i in range(len(names)) if i not in held)]
        for f in (x for x in k.rglob("*") if x.is_file()):
            data = f.read_bytes()
            assert not [s for s in secret if s in data or s.hex().encode() in data], f"{f.relative_to(k)} holds Kw or an unopened tensor's key"
        kt = json.loads((k / "weights" / "tree.json").read_text())
        assert "key" not in kt and kt["tensor_keys"] == {str(i): step_rows.tensor_key(WK, i).hex() for i in held}
        assert (kt["root"], kt["tensor_roots"], kt["names"]) == (tree["root"], tree["tensor_roots"], tree["names"])
        assert kt["extra"] == {n: i for n, i in tree["extra"].items() if n in opened}
        assert sorted(x.name for x in (k / "weights" / "extra").glob("*")) == [f"{i}.bin" for i in sorted(kt["extra"].values())]
    ckd = tmp_path / "ck"
    ckd.mkdir()
    (ckd / "config.json").write_text("{}")
    _shard(ckd / "model.safetensors", PARTS)
    CPU.keep(out, b, store, run, t, plan, c, opened={QKV, "model.rotary_emb.cos_sin_cache"})
    k = CPU.slim_path(out, 1)
    words, wc = CPU.open_weights(k, json.loads((k / "weights" / "tree.json").read_text()), tree["root"],
                                 WR.CheckpointIndex(str(ckd), verify_shards=False), threads=2)
    assert wc["equal"] is True and set(words) == {QKV, "model.rotary_emb.cos_sin_cache"} and wc["mismatched_n"] == 0, wc
    assert {u["name"]: u["why"] for u in wc["unopened"]} == {n: "the slim keep carries no salt key for it" for n in ("model.norm.weight", "model.unknown.weight")}
    with pytest.raises(KeyError, match="carries no salt key"):
        CPU.tensor_root_of(json.loads((k / "weights" / "tree.json").read_text()), 1, PARTS["model.norm.weight"].view(np.uint8))


def _decided(tmp_path):
    """A record whose deferred replay was decided and left its keep, and its instrumented row as it awaited the replay."""
    out, b, store, run, t, _want, c, plan = _kept(tmp_path)
    CPU.keep(out, b, store, run, t, plan, c, opened=set())
    rows, row, verdict, _s, _r = CPU.of_record(b, out)
    rc = row["commit"]
    rc.update(commit_fail_reason=RB.PENDING, value_correspondence={"method": "offline-suite", "result": None, "why": None,
                                                                   "components": {"weights_pin": {"result": True}}})
    awaiting = json.loads(json.dumps(row))
    rc.update(commit_pass=True, commit_fail_reason="PASS", sampled_replay={"result": True, "grade": "complete"},
              boundary_linkage={"result": True, "linked": 2}, opened_values={"method": "opened", "replay_linkage": {"n": 2}},
              weights_attest={"root_openings": "2/2"})
    vc = rc["value_correspondence"]
    vc.update(method="sampled-exact-replay(complete) + boundary-linkage + offline-suite", result=True)
    vc["components"].update(sampled_replay={"result": True}, boundary_linkage={"result": True}, negatives={"suite": "s"})
    CPU.record(out, rows, verdict, b, 1.0)
    return out, awaiting


def test_a_decided_record_reads_back_from_its_keep_as_it_awaited_its_bundle(tmp_path):
    out, awaiting = _decided(tmp_path)
    assert not RB.awaits_replay(json.loads((out / "runs.jsonl").read_text())["commit"])
    k, b = CPU.open_source(CPU.slim_path(out, 1)), CPU.open_source(RB.path(out, 1))
    assert "keep" in k and "keep" not in b and k["manifest_sha256"] == b["manifest_sha256"] and k["context"] == b["context"]
    assert k["manifest"]["files"]["binding_map.json"] == b["manifest"]["files"]["binding_map.json"]
    _rows, row, _verdict, store, run = CPU.of_record(k, out)
    assert row == awaiting and RB.awaits_replay(row["commit"])
    assert store._run_root(run) == run.run_root and run.run_root.hex() == row["commit"]["run_root"]


@pytest.mark.parametrize("tamper,why", [
    (lambda k: (k / "context.json").write_text("{}"), "sha256 is not slim.json's"),
    (lambda k: (k / "weights" / "stray.bin").write_bytes(b"\x00"), "extra"),
    (lambda k: (k / "binding_map.json").unlink(), "missing"),
    (lambda k: (k / CPU.SLIM_DOC).write_text(json.dumps({**json.loads((k / CPU.SLIM_DOC).read_text()), "dump_sha256": "00"})), "dump sha256"),
])
def test_a_keep_that_is_not_its_slim_json_is_refused(tmp_path, tamper, why):
    out, _ = _decided(tmp_path)
    tamper(CPU.slim_path(out, 1))
    with pytest.raises(ValueError, match=why):
        CPU.open_keep(CPU.slim_path(out, 1))


def test_a_record_not_decided_by_the_keep_s_bundle_is_refused(tmp_path):
    out, _ = _decided(tmp_path)
    v = json.loads((out / "verdict.json").read_text())
    v["replay_deferred"]["replayed"][0]["manifest_sha256"] = "00" * 32
    (out / "verdict.json").write_text(json.dumps(v))
    with pytest.raises(CPU.Refused, match="not decided by a replay of the bundle"):
        CPU.of_record(CPU.open_keep(CPU.slim_path(out, 1)), out)


def test_a_keep_replays_only_under_the_draw_it_was_kept_for(tmp_path, monkeypatch):
    from verity_vllm.pipeline import c2_replay as C2

    out, _ = _decided(tmp_path)
    slim = CPU.open_keep(CPU.slim_path(out, 1))["keep"]
    site = C2.ReplaySources(args=SimpleNamespace(replay_draw="uniform"), out=out, pair=1, com=None, rc=None, row={}, reqs=[], sampling={}, tokens={},
                            manifest=None, stop_of_record={}, ids_full={}, versions={}, components={}, vc_result=None, weights_of=None,
                            weights_check=None, tl=None)
    drawn = {"picks": [tuple(p) for p in slim["picks"]]}
    monkeypatch.setattr(C2, "plan", lambda s: {"replay": drawn})
    CPU.kept_draw(site, slim)
    drawn["picks"] = drawn["picks"][::-1]
    with pytest.raises(CPU.Refused, match=r"shares 2 of the keep's 2 picks \(2 drawn\); the record draws by 'uniform'"):
        CPU.kept_draw(site, slim)
    drawn["picks"] = [["r0", 9, 9, "model.embed_tokens"], *slim["picks"][:1]]
    with pytest.raises(CPU.Refused, match=r"shares 1 of the keep's 2 picks"):
        CPU.kept_draw(site, slim)

    def lacks(s):
        raise KeyError("tree node 1 of step 0 level 0 is not in the slim dumped store")

    monkeypatch.setattr(C2, "plan", lacks)
    with pytest.raises(CPU.Refused, match="reads what the keep does not hold .*tree node 1 of step 0"):
        CPU.kept_draw(site, slim)


def test_the_replay_s_engine_of_record_is_the_commit_engine_s(tmp_path):
    art = tmp_path / "sampled_replay_p1.json"
    cpu = {"VLLM_BATCH_INVARIANT": None, "compute_capability": "unavailable: no CUDA device"}
    art.write_text(json.dumps({"sampled_replay": {"engine_of_record": cpu}, "boundary_linkage": {}}))
    row = {"commit": {"sampled_replay": {"engine_of_record": cpu}}}
    gpu = {"VLLM_BATCH_INVARIANT": "1", "compute_capability": [12, 0]}
    CPU.engine_of_record(row, {"engine_of_record": gpu}, art)
    for sr in (row["commit"]["sampled_replay"], json.loads(art.read_text())["sampled_replay"]):
        assert sr["engine_of_record"] == gpu and sr["engine_of_replay"] == cpu


def test_the_record_is_rewritten_whole_with_the_decided_run(tmp_path):
    out = _deferred(tmp_path)
    b = RB.open_bundle(RB.path(out, 1))
    rows, row, verdict, _s, _r = CPU.of_record(b, out)
    (out / "summary.json").write_text(json.dumps({"commit_pass": None}))
    row["commit"].update(commit_pass=False, commit_fail_reason="C2 FAIL", sampled_replay={"result": False, "grade": "mismatch"})
    CPU.record(out, rows, verdict, b, 1.5)
    v = json.loads((out / "verdict.json").read_text())
    assert v["commit_pass"] is False and v["first_fail_reason"] == "C2 FAIL" and v["replay_deferred"]["pending_pairs"] == []
    assert v["replay_deferred"]["replayed"][0]["manifest_sha256"] == b["manifest_sha256"]
    assert json.loads((out / "summary.json").read_text())["commit_pass"] is False
    assert json.loads((out / "runs.jsonl").read_text())["commit"]["commit_fail_reason"] == "C2 FAIL"
    assert not list(out.glob("*.tmp"))
