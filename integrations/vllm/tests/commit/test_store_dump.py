"""The committed store on disk (`commit.store_dump`): a dumped rows-sha512 store (`step_rows`) opens and verifies like the committer it
was dumped from; its full dump keeps the steps' salt keys, its slim dump only the salts of the leaves its reads opened."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tests.check.opened import commit_steps, retained_block
from verity_vllm.check import oracle_compare as OC
from verity_vllm.commit import opened as OV
from verity_vllm.commit import step_rows as R
from verity_vllm.commit import store_dump as SD
from verity_vllm.commit.committer.native_collect import _HostRanges, _HostWindows

K = R.ROW_BYTES // 256                   # every size and offset below is so many sixteenths of a row
STEPS = {0: [("model.embed_tokens", bytes(range(256)) * 5 * K), ("model.layers.0.self_attn.qkv_proj", b"\x07" * 900 * K)],
         1: [("model.embed_tokens", bytes(reversed(range(256))) * 3 * K), ("runner.sampler", b"\x01\x02\x03\x04")],
         2: [("model.layers.0.mlp.down_proj", bytes(i % 251 for i in range(3000 * K)))]}


def _reads(reader):
    return {(s, m.name): reader(s, m.name, 0, m.nbytes) for s, ms in reader.com._layouts.items() for m in ms}


def test_a_dumped_store_reads_every_member_the_committer_reads(tmp_path):
    com, rc = commit_steps(STEPS)
    SD.dump(com, rc, tmp_path)
    store, run = SD.load(tmp_path)
    assert run.run_root == rc.run_root and [s.root for s in run.steps] == [s.root for s in rc.steps]
    want, got = _reads(OC.OpenedReader(com, rc)), _reads(OC.OpenedReader(store, run))
    assert got == want and all(v for v in got.values())
    assert OC.OpenedReader(store, run)(0, "model.layers.0.self_attn.qkv_proj", 10, 20) == b"\x07" * 10


def test_the_dump_hashes_every_step_file_as_it_writes_it(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(SD, "FLUSH", 100)                            # a flush between the chunks of every step
    com, rc = commit_steps(STEPS)
    got = {}
    SD.dump(com, rc, tmp_path, threads=3, digests=got)
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.bin"))
    assert sorted(got) == files and len(files) == 2 * len(STEPS) + 1
    assert all(got[f] == hashlib.sha256((tmp_path / f).read_bytes()).hexdigest() for f in files)
    assert _reads(OC.OpenedReader(*SD.load(tmp_path))) == _reads(OC.OpenedReader(com, rc))


def test_a_flipped_dumped_byte_does_not_verify(tmp_path):
    com, rc = commit_steps(STEPS)
    SD.dump(com, rc, tmp_path)
    raw = tmp_path / "raw" / "s0.bin"
    b = bytearray(raw.read_bytes())
    b[1300 * K] ^= 1                                                 # inside qkv_proj (offset 1280 K)
    raw.write_bytes(bytes(b))
    store, run = SD.load(tmp_path)
    rd = OC.OpenedReader(store, run)
    assert rd(0, "model.embed_tokens", 0, 64) == bytes(range(64))
    with pytest.raises(OV.OpeningNotVerified, match="qkv_proj"):
        rd(0, "model.layers.0.self_attn.qkv_proj", 0, 100)


def test_a_dumped_root_that_is_not_the_run_root_verifies_nothing(tmp_path):
    com, rc = commit_steps(STEPS)
    SD.dump(com, rc, tmp_path)
    store, run = SD.load(tmp_path)
    run.run_root = bytes(len(run.run_root))
    with pytest.raises(OV.OpeningNotVerified):
        OC.OpenedReader(store, run)(1, "runner.sampler", 0, 4)


def test_sparse_host_retainers_dump_what_they_hold(tmp_path):
    com, rc = commit_steps(STEPS)
    keep0, used0 = com._gpu_blocks[0]
    C = com.chunk
    W = 512 * K
    wins = [(k, keep0[k * W:(k + 1) * W], min(W, int(keep0.numel()) - k * W)) for k in range(-(-int(keep0.numel()) // W)) if k != 2]
    com._gpu_blocks[0] = (_HostWindows(wins, used0, W, C), used0)
    keep2, used2 = com._gpu_blocks[2]
    com._gpu_blocks[2] = (_HostRanges([(0, 1024 * K, keep2[:1024 * K]), (2048 * K, int(keep2.numel()), keep2[2048 * K:])], used2, chunk=C), used2)
    SD.dump(com, rc, tmp_path)
    store, run = SD.load(tmp_path)
    want, got = _reads(OC.OpenedReader(com, rc)), _reads(OC.OpenedReader(store, run))
    assert got == want
    assert got[(0, "model.embed_tokens")] is None                    # window 2 (bytes 1024 K..1536 K) not retained: not a read, never zeros
    assert OC.OpenedReader(store, run)(2, "model.layers.0.mlp.down_proj", 0, 1000 * K) == bytes(i % 251 for i in range(1000 * K))
    assert OC.OpenedReader(store, run)(2, "model.layers.0.mlp.down_proj", 1000 * K, 1100 * K) is None


def test_the_dumped_block_is_the_retained_block(tmp_path):
    com, rc = commit_steps(STEPS)
    SD.dump(com, rc, tmp_path)
    assert (tmp_path / "raw" / "s1.bin").read_bytes() == bytes(retained_block(com, 1).numpy())


def test_a_slim_dump_serves_the_reads_it_recorded_and_refuses_the_rest(tmp_path):
    com, rc = commit_steps(STEPS)
    with SD.touched(com) as t:
        want = OC.OpenedReader(com, rc)(0, "model.layers.0.self_attn.qkv_proj", 10, 20)
    assert t.ranges[0] and t.nodes[0] and not t.ranges.get(2)
    SD.dump(com, rc, tmp_path / "slim", touched=t)
    SD.dump(com, rc, tmp_path / "full")
    assert sum(p.stat().st_size for p in (tmp_path / "slim").rglob("*.bin")) < sum(p.stat().st_size for p in (tmp_path / "full").rglob("*.bin"))
    store, run = SD.load(tmp_path / "slim")
    assert store._run_root(run) == rc.run_root
    rd = OC.OpenedReader(store, run)
    assert rd(0, "model.layers.0.self_attn.qkv_proj", 10, 20) == want == b"\x07" * 10
    assert rd(2, "model.layers.0.mlp.down_proj", 0, 100) is None, "a member the recorded reads never opened is not in the slim store"


def test_a_dumped_store_dumps_again_slim(tmp_path):
    com, rc = commit_steps(STEPS)
    SD.dump(com, rc, tmp_path / "full")
    store, run = SD.load(tmp_path / "full")
    with SD.touched(store) as t:
        want = OC.OpenedReader(store, run)(1, "runner.sampler", 0, 4)
    SD.dump(store, run, tmp_path / "slim", touched=t)
    assert OC.OpenedReader(*SD.load(tmp_path / "slim"))(1, "runner.sampler", 0, 4) == want == b"\x01\x02\x03\x04"


def test_touching_a_read_records_what_the_verified_read_opens():
    com, rc = commit_steps(STEPS)
    reads = [(0, "model.layers.0.self_attn.qkv_proj", 10, 20), (1, "runner.sampler", 0, 4), (2, "model.layers.0.mlp.down_proj", 900 * K, 2500 * K)]
    with SD.touched(com) as verified:
        rd = OC.OpenedReader(com, rc)
        for r in reads:
            rd(*r)
    with SD.touched(com) as touched:
        SD.touch(com, reads + [(0, "no.such.member", 0, 4)])
    assert touched.ranges == verified.ranges and touched.nodes == verified.nodes and touched.ranges[2]


def test_a_rows_store_dumps_its_nodes_and_keys_in_full(tmp_path):
    com, rc = commit_steps(STEPS)
    doc = SD.dump(com, rc, tmp_path)
    assert doc["commit_scheme"] == R.NAME and len(rc.run_root) == R.ROOT_BYTES
    assert (tmp_path / "levels" / "s2.bin").stat().st_size == R.ROOT_BYTES * sum(doc["steps"][2]["levels"])
    assert (tmp_path / "keys.bin").read_bytes() == b"".join(com.row_keys[s] for s in sorted(STEPS))
    assert SD.nbytes(com) == sum(p.stat().st_size for p in tmp_path.rglob("*.bin")) - (tmp_path / "keys.bin").stat().st_size
    store, run = SD.load(tmp_path)
    assert store.rows and store.row_keys == com.row_keys
    ro = store.open_range(2, 3, 6)
    assert ro.salts == b"".join(R.salt(com.row_keys[2], i) for i in range(3, 6)) and store.verify_range(run, ro)


def test_a_slim_rows_store_keeps_its_leaves_salts_and_no_key(tmp_path):
    com, rc = commit_steps(STEPS)
    with SD.touched(com) as t:
        want = OC.OpenedReader(com, rc)(2, "model.layers.0.mlp.down_proj", 300 * K, 700 * K)
    assert t.leaves(2) == [(1, 3)] and t.nbytes() == 2 * R.ROW_BYTES + R.ROOT_BYTES * len(t.nodes[2]) + 2 * R.SALT_BYTES
    doc = SD.dump(com, rc, tmp_path / "slim", touched=t)
    assert not (tmp_path / "slim" / "keys.bin").exists() and doc["steps"][2]["salted"] == [[1, 3]]
    assert (tmp_path / "slim" / "salts" / "s2.bin").read_bytes() == R.salt(com.row_keys[2], 1) + R.salt(com.row_keys[2], 2)
    assert (tmp_path / "slim" / "salts" / "s0.bin").read_bytes() == b""
    store, run = SD.load(tmp_path / "slim")
    assert not store.row_keys
    assert OC.OpenedReader(store, run)(2, "model.layers.0.mlp.down_proj", 300 * K, 700 * K) == want
    with pytest.raises(KeyError, match="salt of leaf 3 of step 2"):
        store._gpu_salt(2, 3)
    with pytest.raises(ValueError, match="without their salt keys"):
        SD.dump(store, run, tmp_path / "again")
    with SD.touched(store) as t2:
        OC.OpenedReader(store, run)(2, "model.layers.0.mlp.down_proj", 260 * K, 760 * K)
    SD.dump(store, run, tmp_path / "slimmer", touched=t2)
    assert OC.OpenedReader(*SD.load(tmp_path / "slimmer"))(2, "model.layers.0.mlp.down_proj", 300 * K, 700 * K) == want
    salts = tmp_path / "slim" / "salts" / "s2.bin"
    b = bytearray(salts.read_bytes())
    b[0] ^= 1
    salts.write_bytes(bytes(b))
    store, run = SD.load(tmp_path / "slim")
    with pytest.raises(OV.OpeningNotVerified, match="down_proj"):
        OC.OpenedReader(store, run)(2, "model.layers.0.mlp.down_proj", 300 * K, 700 * K)


def test_a_store_of_another_commit_scheme_is_refused(tmp_path):
    com, rc = commit_steps(STEPS)
    com.commit_scheme = "rows-sha256"
    with pytest.raises(ValueError, match="rows-sha256 steps: not dumped"):
        SD.dump(com, rc, tmp_path)
