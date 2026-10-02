"""The honest-latency soak's pieces (`verity_vllm.pipeline.pous_soak`): the cell cross and its schedule, the Poisson draws and
reads, the vLLM state join, and the native responder, both verifier loops, the ring and the checker end to end over loopback
against host memory (the library built with `SOAK_FAKE_CUDA`)."""
from __future__ import annotations

import ctypes
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from verity_vllm.pipeline import pous_soak as S


def test_cells_are_the_whole_cross_once():
    cs = S.cells()
    assert len(cs) == len(S.LOADS) * len(S.RATES) * len(S.SIZES_KIB) * len(S.CONFIGS) == 72
    assert [c["id"] for c in cs] == list(range(72))
    assert len({(c["load"], c["rate"], c["kib"], c["config"]) for c in cs}) == 72


def test_schedule_is_passes_of_permutations_fixed_by_the_seed():
    a, b = S.draw_schedule(b"x" * 32), S.draw_schedule(b"x" * 32)
    assert a == b and len(a) == S.PASSES
    assert all(sorted(p) == list(range(72)) for p in a)
    assert a != S.draw_schedule(b"y" * 32)


def test_poisson_times_and_reads():
    rng = np.random.default_rng(1)
    t = S.poisson_times(rng, 1000, 10.0, 5_000)
    assert np.all(np.diff(t) > 0) and t[0] >= 5_000 and t[-1] < 5_000 + 10e9
    assert 9_500 < len(t) < 10_500
    blk = S.block_starts(rng, 10_000, 100, 144 * 1024)
    assert blk.min() >= 0 and blk.max() <= 97
    assert list(S.contained(7, 144 * 1024)) == [7, 8] and list(S.contained(7, 384 * 1024)) == list(range(7, 13))


def test_join_steps_reads_the_step_holding_each_time():
    #             cpu_t0 cpu_t1 gpu_t0 gpu_t1 tokens reqs new running
    steps = np.array([[0, 0, 100, 200, 8192, 4, 4, 4],
                      [0, 0, 300, 400, 8, 8, 0, 8]], np.int64)
    state, reqs, running = S.join_steps(np.array([50, 100, 150, 250, 300, 399, 400]), steps)
    assert state.tolist() == [0, S.STATE_PREFILL, S.STATE_PREFILL, 0, S.STATE_DECODE, S.STATE_DECODE, 0]
    assert reqs.tolist() == [0, 4, 4, 0, 8, 8, 0] and running.tolist() == [0, 4, 4, 0, 8, 8, 0]


def test_read_steps_round_trips_the_step_log(tmp_path):
    p = tmp_path / "steps.bin"
    p.write_bytes(S.STEP.pack(1, 2, 3, 4, 5, 6, 7, 8) + S.STEP.pack(9, 10, 11, 12, 13, 14, 15, 16) + b"\x01\x02")
    assert S.read_steps(p).tolist() == [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("no g++")
    lib = S.load_native(S.build_native(tmp_path_factory.mktemp("native"), None, fake=True))
    from verity_pous.verifier import leaf_hash

    blocks = 16
    data = np.random.default_rng(7).integers(0, 256, blocks * S.BLOCK, dtype=np.uint8)
    params = SimpleNamespace(blocks_per_segment=8, blocks=blocks, block_bytes=S.BLOCK)
    salt = hashlib.sha256(b"soak test salt").digest()[:24]
    vk = SimpleNamespace(params=params, salt=salt,
                         leaves=tuple(leaf_hash(params, salt, i, data[i * S.BLOCK:(i + 1) * S.BLOCK].tobytes()) for i in range(blocks)))
    resp = S.Responder(lib, data.ctypes.data, data.nbytes, 0, -1)
    yield SimpleNamespace(lib=lib, data=data, vk=vk, resp=resp, blocks=blocks)
    resp.stop()


@pytest.mark.parametrize("tuned", [True, False])
@pytest.mark.parametrize("kib", [64, 144, 384])
def test_loops_answer_every_read_and_the_checker_accepts_them(fake, tuned, kib):
    ring = S.Ring(nslots=32)
    n, nbytes = 20, kib * 1024
    rng = np.random.default_rng(kib)
    t_sched = np.zeros(n, np.int64)
    blk = S.block_starts(rng, n, fake.blocks, nbytes)
    rec = np.zeros((n, len(S.REC)), np.int64)
    c = S.Client(fake.lib, fake.resp.port, tuned)
    assert c.config[15] == S.MAGIC and int(c.config[10]) & 1 == int(tuned)
    if tuned:
        done, status = S.run_tuned(fake.lib, c, ring, t_sched, blk, nbytes, rec, 2**62, 5)
    else:
        done, status = S.run_baseline(c, ring, t_sched, blk, nbytes, rec, 2**62, 5, [False])
    ping = c.exchange(S.OP_PING, 0)
    c.close()
    assert (done, status) == (n, 0)
    col = S.COL
    assert np.all(rec[:, col["magic"]] == S.MAGIC) and np.all(rec[:, col["status"]] == 0)
    assert np.all(rec[:, col["nbytes"]] == nbytes) and np.all(rec[:, col["srv_nbytes"]] == nbytes)
    assert rec[:, col["blk"]].tolist() == blk.tolist()
    assert np.all(rec[:, col["t_recv"]] >= rec[:, col["t_send0"]])
    assert np.all(rec[:, col["srv_utx"]] >= rec[:, col["srv_urx"]])
    assert rec[:, col["srv_seq"]].tolist() == list(range(1, n + 1))
    assert np.all(rec[:, col["srv_krx"]] > 0), "the responder's kernel RX timestamps"
    assert int(ping[6]) > 0 and int(ping[7]) == int(ping[14]), "the last answer's kernel TX timestamp, by its key"
    if tuned:
        assert np.all(rec[:, col["c_krx_last"]] > 0) and np.all(rec[:, col["c_ktx"]] > 0)
        assert np.all(rec[:, col["srv_flags"]] & 1)
    assert int(ring.ctl[S.Ring.HEAD]) == n
    checker = S.Checker.__new__(S.Checker)
    checker.__init__(fake.vk)
    checked = [S.check_slot(ring, checker, seq) for seq in range(n)]
    assert all(why is None for _, why in checked)
    assert sum(k for k, _ in checked) == n * (nbytes // S.BLOCK)


def test_the_checker_catches_a_wrong_block(fake):
    ring = S.Ring(nslots=4)
    c = S.Client(fake.lib, fake.resp.port, True)
    rec = np.zeros((1, len(S.REC)), np.int64)
    fake.data[3 * S.BLOCK + 100] ^= 1
    try:
        done, status = S.run_tuned(fake.lib, c, ring, np.zeros(1, np.int64), np.array([2], np.int64), 2 * S.BLOCK, rec, 2**62, 0)
    finally:
        fake.data[3 * S.BLOCK + 100] ^= 1
        c.close()
    assert (done, status) == (1, 0)
    n, why = S.check_slot(ring, S.Checker(fake.vk), 0)
    assert n == 1 and why is not None and "block 3" in why


def test_out_of_range_reads_are_refused(fake):
    c = S.Client(fake.lib, fake.resp.port, False)
    ring = S.Ring(nslots=4)
    rec = np.zeros((1, len(S.REC)), np.int64)
    done, status = S.run_baseline(c, ring, np.zeros(1, np.int64), np.array([fake.blocks - 1], np.int64), 2 * S.BLOCK, rec, 2**62, 0,
                                  [False])
    c.close()
    assert (done, status) == (1, 0) and rec[0, S.COL["status"]] == 1
    assert S.check_slot(ring, S.Checker(fake.vk), 0)[1] is not None


def test_write_chunk_round_trips(tmp_path):
    soak = SimpleNamespace(job=tmp_path, runs=[{"cell": 3}], recs=[np.arange(2 * len(S.REC), dtype=np.int64).reshape(2, -1)],
                           rec_meta=[np.array([[3, 0, 0], [3, 0, 0]], np.int64)])
    (tmp_path / "steps.bin").write_bytes(S.STEP.pack(0, 0, 0, 10**12, 8, 8, 0, 8))
    doc = S.write_chunk(tmp_path / "chunk.npz", soak, {"wrong": 0})
    with np.load(tmp_path / "chunk.npz") as z:
        assert z["rec"].shape == (2, len(S.REC)) and z["rec_columns"].tolist() == list(S.REC)
        assert z["rec_vllm_state"].tolist() == [S.STATE_DECODE, S.STATE_DECODE]
        assert json.loads(str(z["meta_json"]))["summary"] == doc["summary"]
    assert doc["summary"]["3"]["answers"] == 2


def test_record_layout_matches_the_library(fake):
    assert fake.lib.soak_record_words() == len(S.REC) and ctypes.sizeof(ctypes.c_int64) * S.HEADER_WORDS == S.HEADER_BYTES
    assert Path(S.SOURCE).exists()
