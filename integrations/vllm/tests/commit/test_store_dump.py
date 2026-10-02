"""The committed store on disk (`commit.store_dump`): a dumped GPU-tree store opens and verifies like the committer it was dumped from."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tests.check.opened import commit_steps, retained_block
from verity_vllm.check import oracle_compare as OC
from verity_vllm.commit import opened as OV
from verity_vllm.commit import store_dump as SD
from verity_vllm.commit.committer.native_collect import (
    _HostRanges,
    _HostWindows,
)

STEPS = {0: [("model.embed_tokens", bytes(range(256)) * 5), ("model.layers.0.self_attn.qkv_proj", b"\x07" * 900)],
         1: [("model.embed_tokens", bytes(reversed(range(256))) * 3), ("runner.sampler", b"\x01\x02\x03\x04")],
         2: [("model.layers.0.mlp.down_proj", bytes(i % 251 for i in range(3000)))]}


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
    assert sorted(got) == files and len(files) == 2 * len(STEPS)
    assert all(got[f] == hashlib.sha256((tmp_path / f).read_bytes()).hexdigest() for f in files)
    assert _reads(OC.OpenedReader(*SD.load(tmp_path))) == _reads(OC.OpenedReader(com, rc))


def test_a_flipped_dumped_byte_does_not_verify(tmp_path):
    com, rc = commit_steps(STEPS)
    SD.dump(com, rc, tmp_path)
    raw = tmp_path / "raw" / "s0.bin"
    b = bytearray(raw.read_bytes())
    b[1300] ^= 1                                                     # inside qkv_proj (offset 1280)
    raw.write_bytes(bytes(b))
    store, run = SD.load(tmp_path)
    rd = OC.OpenedReader(store, run)
    assert rd(0, "model.embed_tokens", 0, 64) == (bytes(range(256)) * 5)[:64]
    with pytest.raises(OV.OpeningNotVerified, match="qkv_proj"):
        rd(0, "model.layers.0.self_attn.qkv_proj", 0, 100)


def test_a_dumped_root_that_is_not_the_run_root_verifies_nothing(tmp_path):
    com, rc = commit_steps(STEPS)
    SD.dump(com, rc, tmp_path)
    store, run = SD.load(tmp_path)
    run.run_root = bytes(32)
    with pytest.raises(OV.OpeningNotVerified):
        OC.OpenedReader(store, run)(1, "runner.sampler", 0, 4)


def test_sparse_host_retainers_dump_what_they_hold(tmp_path):
    com, rc = commit_steps(STEPS)
    keep0, used0 = com._gpu_blocks[0]
    W = 512
    wins = [(k, keep0[k * W:(k + 1) * W], min(W, int(keep0.numel()) - k * W)) for k in range(-(-int(keep0.numel()) // W)) if k != 2]
    com._gpu_blocks[0] = (_HostWindows(wins, used0, W), used0)
    keep2, used2 = com._gpu_blocks[2]
    com._gpu_blocks[2] = (_HostRanges([(0, 1024, keep2[:1024]), (2048, int(keep2.numel()), keep2[2048:])], used2), used2)
    SD.dump(com, rc, tmp_path)
    store, run = SD.load(tmp_path)
    want, got = _reads(OC.OpenedReader(com, rc)), _reads(OC.OpenedReader(store, run))
    assert got == want
    assert got[(0, "model.embed_tokens")] is None                    # window 2 (bytes 1024..1536) not retained: not a read, never zeros
    assert OC.OpenedReader(store, run)(2, "model.layers.0.mlp.down_proj", 0, 1000) == bytes(i % 251 for i in range(1000))
    assert OC.OpenedReader(store, run)(2, "model.layers.0.mlp.down_proj", 1000, 1100) is None


def test_a_store_with_padding_steps_is_refused(tmp_path):
    com, rc = commit_steps(STEPS)
    com._padding = {3: object()}
    with pytest.raises(ValueError, match="padding steps"):
        SD.dump(com, rc, tmp_path)


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
    reads = [(0, "model.layers.0.self_attn.qkv_proj", 10, 20), (1, "runner.sampler", 0, 4), (2, "model.layers.0.mlp.down_proj", 900, 2500)]
    with SD.touched(com) as verified:
        rd = OC.OpenedReader(com, rc)
        for r in reads:
            rd(*r)
    with SD.touched(com) as touched:
        SD.touch(com, reads + [(0, "no.such.member", 0, 4)])
    assert touched.ranges == verified.ranges and touched.nodes == verified.nodes and touched.ranges[2]
