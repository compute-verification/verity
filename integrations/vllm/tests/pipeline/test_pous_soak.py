"""The honest-latency soak's pieces (`verity_vllm.pipeline.pous_soak`): the cell cross and its schedule, the Poisson draws and
reads, the vLLM state join, and the native responder, both verifier loops, the ring and the checker end to end over loopback
against host memory (the library built with `SOAK_FAKE_CUDA`)."""
from __future__ import annotations

import ctypes
import hashlib
import json
import shutil
import socket
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


def test_a_chosen_cross_and_its_schedule():
    cs = S.cells(("decode", "prefill", "mixed"), S.ALL_CONFIGS)
    assert len(cs) == 3 * 3 * 3 * 3 == 81 and [c["id"] for c in cs] == list(range(81))
    assert {c["load"] for c in cs} == {"decode", "prefill", "mixed"} and {c["config"] for c in cs} == set(S.ALL_CONFIGS)
    assert S.cells() == S.cells(S.LOADS, S.CONFIGS), "the soak's own cross keeps its ids"
    passes = S.draw_schedule(b"z" * 32, 2, 0, len(cs))
    assert len(passes) == 2 and all(sorted(c for c, ts in p) == list(range(81)) and all(ts for _, ts in p) for p in passes)


def test_schedule_is_passes_of_permutations_fixed_by_the_seed():
    a, b = S.draw_schedule(b"x" * 32), S.draw_schedule(b"x" * 32)
    assert a == b and len(a) == S.PASSES
    for p in a:
        assert sorted(c for c, ts in p if ts) == list(range(72))
        twins = [c for c, ts in p if not ts]
        assert len(twins) == len(set(twins)) == S.TWINS_PER_PASS
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
    path = S.build_native(tmp_path_factory.mktemp("native"), None, fake=True)
    lib = S.load_native(path)
    from verity_pous.verifier import leaf_hash

    blocks = 16
    data = np.random.default_rng(7).integers(0, 256, blocks * S.BLOCK, dtype=np.uint8)
    params = SimpleNamespace(blocks_per_segment=8, blocks=blocks, block_bytes=S.BLOCK)
    salt = hashlib.sha256(b"soak test salt").digest()[:24]
    vk = SimpleNamespace(params=params, salt=salt,
                         leaves=tuple(leaf_hash(params, salt, i, data[i * S.BLOCK:(i + 1) * S.BLOCK].tobytes()) for i in range(blocks)))
    resp = S.Responder(lib, data.ctypes.data, data.nbytes, 0, -1)
    yield SimpleNamespace(lib=lib, path=path, data=data, vk=vk, resp=resp, blocks=blocks)
    resp.stop()


def _run(fake, tuned, ts, t_sched, blk, ops, nbytes, ring, zerocopy=False):
    rec = np.zeros((len(t_sched), len(S.REC)), np.int64)
    c = S.Client(fake.lib, fake.resp.port, tuned, ts, zerocopy)
    assert c.config[15] == S.MAGIC and int(c.config[10]) & 1 == int(tuned) and bool(int(c.config[10]) & 32) == (not ts)
    if tuned:
        done, status = S.run_tuned(fake.lib, c, ring, t_sched, blk, ops, nbytes, rec, 2**62, 5)
    else:
        done, status = S.run_baseline(c, ring, t_sched, blk, ops, nbytes, rec, 2**62, 5, [False])
    ping = c.exchange(S.OP_PING, 0)
    c.close()
    final = {"srv_ktx_newest": int(ping[6]), "srv_ktx_key": int(ping[7]), "srv_prev_send_calls": int(ping[8]),
             "srv_prev_send_end": int(ping[9]), "srv_prev_last_key": int(ping[14])}
    return done, status, rec, ping, final


@pytest.mark.parametrize("tuned", [True, False])
@pytest.mark.parametrize("kib", [64, 144, 384])
def test_loops_answer_every_read_and_ping_and_the_checker_accepts_them(fake, tuned, kib):
    ring = S.Ring(nslots=32)
    n, nbytes = 20, kib * 1024
    rng = np.random.default_rng(kib)
    t_sched = np.zeros(n, np.int64)
    blk = S.block_starts(rng, n, fake.blocks, nbytes)
    ops = np.full(n, S.OP_READ, np.int64)
    ops[[3, 11]] = S.OP_PING
    done, status, rec, ping, final = _run(fake, tuned, True, t_sched, blk, ops, nbytes, ring)
    assert (done, status) == (n, 0)
    col = S.COL
    assert np.all(rec[:, col["magic"]] == S.MAGIC) and np.all(rec[:, col["status"]] == 0)
    assert np.all(rec[:, col["nbytes"]] == nbytes) and np.all(rec[:, col["srv_nbytes"]] == nbytes)
    assert rec[:, col["op"]].tolist() == ops.tolist() and rec[:, col["srv_op"]].tolist() == ops.tolist()
    assert rec[:, col["blk"]].tolist() == blk.tolist()
    assert np.all(rec[:, col["t_recv"]] >= rec[:, col["t_send0"]])
    assert np.all(rec[:, col["srv_utx"]] >= rec[:, col["srv_urx"]])
    reads = ops == S.OP_READ
    assert np.all(rec[reads, col["srv_s2"]] >= rec[reads, col["srv_s1"]]) and np.all(rec[reads, col["srv_s1"]] > 0)
    assert np.all(rec[~reads, col["srv_s1"]] == 0), "a ping does no device work"
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
    assert sum(k for k, _ in checked) == (n - 2) * (nbytes // S.BLOCK)
    runs = [{"cell": 0, "ts": True, "final_ping": final}]
    ktx, s4 = S.response_tx(rec, np.zeros(n, np.int64), runs)
    assert np.all(ktx > 0) and np.all(s4 > 0), "every answer's server TX timestamp and send end, from a later header"
    assert np.all(ktx >= rec[:, col["srv_utx"]]) and np.all(ktx <= rec[:, col["t_recv"]] + 1000)
    v = S.validate(rec, np.zeros(n, np.int64), runs, np.zeros((0, 8), np.int64), {}, (n - 2) * (nbytes // S.BLOCK), 60.0)
    assert v["blocks"]["equal"] and v["pings"] == 2 and v["status_nonzero"] == {}
    assert v["absent"]["K_rx_req"] == v["absent"]["K_tx_resp"] == v["absent"]["S4"] == v["absent"]["S1_S2_reads"] == 0
    assert v["gpu_span_violations"] == 0
    # The kernel may transmit a response's tail after sendmsg returns (pushed on the receiver's ACK): K_tx_resp > S4 is counted,
    # not an error.
    bad = {k: x for k, x in v["order_violations"].items() if x and k != "..<=S4"}
    assert not bad, bad


@pytest.mark.parametrize("kib", [64, 384])
def test_zero_copy_sends_answer_every_read_after_each_completion(fake, kib):
    ring = S.Ring(nslots=32)
    n, nbytes = 20, kib * 1024
    blk = S.block_starts(np.random.default_rng(kib + 1), n, fake.blocks, nbytes)
    ops = np.full(n, S.OP_READ, np.int64)
    ops[5] = S.OP_PING
    errors = fake.resp.stat()["errors"]
    done, status, rec, ping, _ = _run(fake, True, True, np.zeros(n, np.int64), blk, ops, nbytes, ring, zerocopy=True)
    assert (done, status) == (n, 0)
    col = S.COL
    assert np.all(rec[:, col["srv_flags"]] & 128) and np.all(rec[:, col["status"]] == 0)
    # each response waited for the one before it to complete, so by the next header every earlier one has
    assert rec[1:, col["srv_zc_completions"]].min() >= 1 and int(ping[25]) >= 1
    assert np.all(np.diff(rec[:, col["srv_zc_completions"]]) >= 0) and np.all(rec[:, col["srv_zc_wait_ns"]] >= 0)
    assert int(ping[10]) & 256, "loopback delivers a zero-copy send by copying it, and says so"
    checker = S.Checker(fake.vk)
    checked = [S.check_slot(ring, checker, seq) for seq in range(n)]
    assert all(why is None for _, why in checked) and sum(k for k, _ in checked) == (n - 1) * (nbytes // S.BLOCK)
    after = fake.resp.stat()
    assert after["errors"] - errors == (0 if after["mlocked"] else 1), "no error but a refused mlockall (a small memlock limit)"
    assert rec[:, col["srv_zc_wait_ns"]].max() < 10**9, "no completion timed out"


@pytest.mark.parametrize("tuned", [True, False])
def test_a_timestamping_off_twin_carries_no_kernel_stamps(fake, tuned):
    ring = S.Ring(nslots=8)
    n = 6
    done, status, rec, _, final = _run(fake, tuned, False, np.zeros(n, np.int64), np.zeros(n, np.int64),
                                       np.full(n, S.OP_READ, np.int64), S.BLOCK, ring)
    assert (done, status) == (n, 0)
    col = S.COL
    assert np.all(rec[:, col["srv_krx"]] == 0) and np.all(rec[:, col["c_ktx"]] == 0) and np.all(rec[:, col["c_krx_last"]] == 0)
    assert np.all(rec[:, col["srv_flags"]] & 32) and not np.any(rec[:, col["srv_flags"]] & 8)
    v = S.validate(rec, np.zeros(n, np.int64), [{"cell": 0, "ts": False, "final_ping": final}], np.zeros((0, 8), np.int64), {},
                   n, 60.0)
    assert v["absent"]["timestamping_on"] == 0 and v["absent"]["K_rx_req"] == 0


def test_validate_counts_an_order_violation_and_a_clock_step():
    rec = np.zeros((2, len(S.REC)), np.int64)
    col = S.COL
    for i, base in enumerate((10_000_000, 20_000_000)):
        rec[i, [col["t_sched"], col["t_send0"], col["srv_krx"], col["srv_urx"], col["srv_utx"], col["t_recv"]]] = \
            [base, base + 10, base + 20, base + 30, base + 40, base + 100]
        rec[i, col["op"]] = S.OP_PING
    rec[1, col["srv_urx"]] = rec[1, col["srv_krx"]] - 5000
    tri = np.array([[0, 10**9, 10], [10**8, 10**9 + 10**8, 10**8 + 10], [2 * 10**8, 10**9 + 2 * 10**8 + 50_000, 2 * 10**8 + 10]])
    v = S.validate(rec, np.zeros(2, np.int64), [{"cell": 0, "ts": True}], np.zeros((0, 8), np.int64), {"clock_triple": tri}, 0, 60.0)
    assert v["order_violations"]["..<=S0"] == 1 and v["order_violations"]["..<=S3"] == 0
    assert v["clock_offset"]["steps_over_1us"] == 1 and v["clock_offset"]["max_step_ns"] == 50_000


def test_the_checker_catches_a_wrong_block(fake):
    ring = S.Ring(nslots=4)
    c = S.Client(fake.lib, fake.resp.port, True)
    rec = np.zeros((1, len(S.REC)), np.int64)
    fake.data[3 * S.BLOCK + 100] ^= 1
    try:
        done, status = S.run_tuned(fake.lib, c, ring, np.zeros(1, np.int64), np.array([2], np.int64), np.full(1, S.OP_READ),
                                   2 * S.BLOCK, rec, 2**62, 0)
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
    done, status = S.run_baseline(c, ring, np.zeros(1, np.int64), np.array([fake.blocks - 1], np.int64), np.full(1, S.OP_READ),
                                  2 * S.BLOCK, rec, 2**62, 0, [False])
    c.close()
    assert (done, status) == (1, 0) and rec[0, S.COL["status"]] == 1
    assert S.check_slot(ring, S.Checker(fake.vk), 0)[1] is not None


def test_write_chunk_round_trips(tmp_path):
    soak = SimpleNamespace(job=tmp_path, runs=[{"cell": 3}], recs=[np.arange(2 * len(S.REC), dtype=np.int64).reshape(2, -1)],
                           rec_meta=[np.array([[3, 0, 0], [3, 0, 0]], np.int64)])
    (tmp_path / "steps.bin").write_bytes(S.STEP.pack(0, 0, 0, 10**12, 8, 8, 0, 8))
    doc = S.write_chunk(tmp_path / "chunk.npz", soak, {"wrong": 0, "checked_blocks": 0})
    with np.load(tmp_path / "chunk.npz") as z:
        assert z["rec"].shape == (2, len(S.REC)) and z["rec_columns"].tolist() == list(S.REC)
        assert z["rec_vllm_state"].tolist() == [S.STATE_DECODE, S.STATE_DECODE]
        assert json.loads(str(z["meta_json"]))["summary"] == doc["summary"]
    assert doc["summary"]["3"]["answers"] == 2 and doc["validation"]["answers"] == 2


def test_write_chunk_carries_the_second_engine_per_run(tmp_path):
    soak = SimpleNamespace(job=tmp_path, runs=[{"cell": 1, "start_ns": 0, "end_ns": 10**9}, {"cell": 2, "start_ns": 10**9,
                                                                                            "end_ns": 2 * 10**9}],
                           recs=[], rec_meta=[], table=S.cells(("decode",), S.ALL_CONFIGS))
    (tmp_path / "engine2.bin").write_bytes(S.ENGINE2.pack(10, 20, 10**9) + S.ENGINE2.pack(30, 40, 10**9))
    doc = S.write_chunk(tmp_path / "chunk.npz", soak, {"wrong": 0, "checked_blocks": 0})
    assert doc["cells"] == soak.table and len(doc["cells"]) == 27
    e2 = doc["validation"]["second_engine"]
    assert e2["runs_without_steps"] == [1] and doc["runs"][0]["engine2"] == {"steps": 2, "gb_per_s": 2.0}
    with np.load(tmp_path / "chunk.npz") as z:
        assert z["engine2_steps"].tolist() == [[10, 20, 10**9], [30, 40, 10**9]]


def test_the_second_engine_reads_every_weight_each_step(tmp_path):
    torch = pytest.importorskip("torch")
    shapes = ((8, 16), (16, 8))
    e = S.SecondEngine(tmp_path / "engine2.bin", 3, 4, "cpu", shapes)
    assert e.bytes == 2 * 3 * (8 * 16 + 16 * 8) == sum(w.numel() * w.element_size() for w in e.w)
    info = e.warm(2)
    assert info["layers"] == 3 and info["bytes_per_step"] == e.bytes and not info["graph"]
    want = e.x[16] @ e.w[-1]
    for _ in range(3):
        e.step()
    assert torch.equal(e.y[8], want)
    e.close()
    steps = S.read_engine2(tmp_path / "engine2.bin")
    assert steps.shape == (3, 3) and np.all(steps[:, 2] == e.bytes) and np.all(steps[:, 1] >= steps[:, 0])
    assert S.SecondEngine.__init__.__defaults__[-1] == S.LLAMA8B_LAYER
    assert 2 * 32 * sum(i * o for i, o in S.LLAMA8B_LAYER) == 13_958_643_712, "Llama-3.1-8B's 32 decoder layers in BF16"


def test_record_layout_matches_the_library(fake):
    assert fake.lib.soak_record_words() == len(S.REC) and ctypes.sizeof(ctypes.c_int64) * S.HEADER_WORDS == S.HEADER_BYTES
    assert Path(S.SOURCE).exists()


def test_cpu_list_reads_ranges_and_singles():
    assert S.cpu_list("96-98,116,120-121") == (96, 97, 98, 116, 120, 121)
    assert S.cpu_list("") == () and S.cpu_list("5,3-4,5") == (3, 4, 5)


def test_dcgm_rows_reads_dmon_lines_and_marks_na():
    lines = [(1, "#Entity   DRAMA  PCITX  PCIRX  GPUTL  MCUTL"), (2, "ID"), (3, "GPU 7   0.912  1000  2000  98  40"),
             (4, "GPU 7   N/A  1000  2000  98  40"), (5, "Error: something")]
    rows = S.dcgm_rows(lines)
    assert rows.shape == (2, 1 + len(S.DCGM_FIELDS)) and rows[0].tolist() == [3, 0.912, 1000, 2000, 98, 40]
    assert np.isnan(rows[1, 1]) and S.dcgm_rows([]).shape == (0, 1 + len(S.DCGM_FIELDS))


def _serve_loop(ctl, acked, stop):
    """The serve loop's side of `Control`: each load taken is applied and acknowledged."""
    while True:
        with ctl.cv:
            ctl.cv.wait_for(lambda: ctl.want is not None or stop.is_set() or ctl.stopped.is_set())
        if stop.is_set() or ctl.stopped.is_set():
            return
        w = ctl.take()
        if w is not None:
            acked.append(w)
            ctl.ack(w[1], {"mode": w[0], "seq": w[1], "t_ns": 1})


def test_control_round_trip_ready_clock_load_rotate_and_stop(tmp_path):
    import threading

    (tmp_path / "steps.bin").write_bytes(S.STEP.pack(*range(8)) + S.STEP.pack(*range(8, 16)) + b"\x00\x01")
    ctl = S.Control("127.0.0.1", 0, "127.0.0.1", {"root": "ab", "port": 1}, tmp_path, None)
    acked, stop = [], threading.Event()
    loop = threading.Thread(target=_serve_loop, args=(ctl, acked, stop), daemon=True)
    loop.start()
    c = S.ControlClient("127.0.0.1", ctl.port)
    try:
        assert c.call("ready")[0] == {"root": "ab", "port": 1}
        clk = c.clock(20)
        assert clk["lo_ns"] <= clk["offset_at_rtt_min_ns"] <= clk["hi_ns"] and clk["lo_ns"] <= 0 <= clk["hi_ns"], "one clock"
        assert c.call("load", mode="serving", seq=1)[0]["mode"] == "serving" and acked == [("serving", 1)]
        _, blobs = c.call("rotate")
        (tmp_path / "got.bin").write_bytes(blobs["steps"])
        assert len(blobs["steps"]) == 2 * S.STEP.size, "whole steps only"
        assert S.read_steps(tmp_path / "got.bin").tolist() == [list(range(8)), list(range(8, 16))]
        with open(tmp_path / "steps.bin", "ab") as f:
            f.write(b"\x02" * (S.STEP.size - 2) + S.STEP.pack(*range(16, 24)))
        assert len(c.call("rotate")[1]["steps"]) == 2 * S.STEP.size, "the next rotate starts where the last ended"
        with pytest.raises(OSError, match="unknown op"):
            c.call("nope")
        c.close()
        assert ctl.stopped.is_set(), "set before the reply to stop"
    finally:
        stop.set()
        with ctl.cv:
            ctl.cv.notify_all()
        loop.join()
        ctl.close()


def test_control_closes_a_connection_from_another_address(tmp_path):
    ctl = S.Control("127.0.0.1", 0, "10.9.9.9", {}, tmp_path, None)
    try:
        c = S.ControlClient("127.0.0.1", ctl.port)
        with pytest.raises((ConnectionError, OSError)):
            c.call("ready")
        c.close(stop=False)
        assert not ctl.stopped.is_set()
    finally:
        ctl.close()


def test_a_remote_server_clock_is_bracketed_and_checked_within_each_host():
    rec = np.zeros((3, len(S.REC)), np.int64)
    col, theta = S.COL, 7_000_000
    for i in range(3):
        b = 10**12 + i * 10**6
        rec[i, [col[k] for k in ("t_sched", "t_send0", "c_ktx", "c_krx_first", "c_krx_last", "t_recv")]] = \
            [b, b + 1000, b + 2000, b + 40_000, b + 41_000, b + 42_000]
        rec[i, [col[k] for k in ("srv_krx", "srv_urx", "srv_utx")]] = [b + theta + 12_000, b + theta + 14_000, b + theta + 20_000]
        rec[i, col["op"]] = S.OP_PING
    run_idx = np.zeros(3, np.int64)
    lo, hi = S.offset_brackets(rec, run_idx)[0]
    assert lo <= theta <= hi and (lo, hi) == (theta - 20_000, theta + 10_000)
    runs = [{"cell": 0, "ts": True}]
    v = S.validate(rec, run_idx, runs, np.zeros((0, 8), np.int64), {}, 0, 60.0, remote=True)
    assert not any(v["order_violations"].values()) and v["offset"]["empty"] == 0 and v["offset"]["width_ns_p50"] == 30_000
    assert "K_tx_resp<=V1" not in v["order_violations"]
    local = S.validate(rec, run_idx, runs, np.zeros((0, 8), np.int64), {}, 0, 60.0)
    assert local["order_violations"]["..<=K_rx_req"] == 0, "server later than the verifier: no chain step goes backwards"
    rec[1, col["srv_krx"]] = rec[1, col["c_ktx"]] + theta - 30_000      # an answer received before it was sent, given the rest
    assert S.validate(rec, run_idx, runs, np.zeros((0, 8), np.int64), {}, 0, 60.0, remote=True)["offset"]["empty"] == 1


def test_one_host_brackets_hold_zero_over_a_real_connection(fake):
    ring = S.Ring(nslots=32)
    n = 20
    c = S.Client(fake.lib, fake.resp.port, True, True, False, host="127.0.0.1")
    rec = np.zeros((n, len(S.REC)), np.int64)
    done, status = S.run_tuned(fake.lib, c, ring, np.zeros(n, np.int64), np.zeros(n, np.int64), np.full(n, S.OP_PING), S.BLOCK,
                               rec, 2**62, 0)
    info = c.tcp_info()
    c.close()
    assert (done, status) == (n, 0) and info["peer"] == ["127.0.0.1", fake.resp.port]
    lo, hi = S.offset_brackets(rec, np.zeros(n, np.int64))[0]
    assert lo <= 1000 and hi >= -1000, (lo, hi)


def test_serving_keeps_its_requests_in_flight_and_pause_drains_them(monkeypatch):
    import sys
    import types

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", vllm)

    class Engine:  # the engine's request ids carry a suffix; its outputs name a request by ours
        def __init__(self):
            self.reqs, self.aborted = {}, []

        def add_request(self, rid, prompt, params):
            self.reqs[f"{rid}-x"] = (rid, params.max_tokens)
            return f"{rid}-x"

        def step(self):
            done = [k for k in sorted(self.reqs)[:2]]
            return [SimpleNamespace(request_id=self.reqs.pop(k)[0], finished=True, outputs=[SimpleNamespace(token_ids=[1, 2, 3])])
                    for k in done]

        def abort_request(self, ids):
            self.aborted += ids
            self.reqs = {k: v for k, v in self.reqs.items() if v[0] not in ids}

        def has_unfinished_requests(self):
            return bool(self.reqs)

    eng = Engine()
    s = S.Serving(SimpleNamespace(llm_engine=eng), np.random.default_rng(0), 5, prompt=(4, 8), output=(2, 3))
    for _ in range(3):
        s.step()
        assert len(s.live) == 3 and len(eng.reqs) == 3, "finished requests leave; the next step refills to 5"
    s.pause()
    assert not s.live and not eng.reqs and len(eng.aborted) == 3 and all(not r.endswith("-x") for r in eng.aborted)
    assert s.stats() == {"added": 9, "finished": 6, "aborted": 3, "finished_tokens": 18}


@pytest.mark.slow
def test_verify_drives_a_server_on_another_host_through_its_control_port(fake, tmp_path):
    """`verify --server` end to end on the CPU: the real verifier process against a `Control` and a responder over host memory
    holding one segment of band-chain/d12/v1, nine cells (about 25 s).  Catches what the pieces' tests can't: the remote path's
    wiring (ready over the port, loads acknowledged, rotate per chunk, offset brackets in the chunk, the server stopped at the
    end) in the process `check` would never otherwise run before a GPU job does."""
    import os
    import subprocess
    import sys
    import threading

    from verity_pous.verifier import VerifierKey, leaf_hash
    from verity_vllm.pipeline import pous_run as R

    scheme = "band-chain/d12/v1"
    params = R.geometry(scheme).with_segments(1)
    data = np.random.default_rng(11).integers(0, 256, params.blocks * S.BLOCK, dtype=np.uint8)
    salt = hashlib.sha256(b"remote soak salt").digest()[:24]
    leaves = [leaf_hash(params, salt, i, data[i * S.BLOCK:(i + 1) * S.BLOCK].tobytes()) for i in range(params.blocks)]
    root = VerifierKey.from_leaves(params, salt, leaves).root.hex()
    setup = tmp_path / "setup"
    setup.mkdir()
    (setup / "vk.json").write_text(json.dumps({"salt": salt.hex(), "root": root, "params": params.manifest(), "scheme": scheme}))
    (setup / "vk-leaves.bin").write_bytes(b"".join(leaves))
    resp = S.Responder(fake.lib, data.ctypes.data, data.nbytes, 0, -1)
    ready = {"port": resp.port, "host": "127.0.0.1", "root": root, "blocks": params.blocks, "block_bytes": S.BLOCK,
             "total": data.nbytes}
    (tmp_path / "server").mkdir()
    ctl = S.Control("127.0.0.1", 0, "127.0.0.1", ready, tmp_path / "server", None, lambda: {"responder": resp.stat()})
    acked, stop = [], threading.Event()
    loop = threading.Thread(target=_serve_loop, args=(ctl, acked, stop), daemon=True)
    loop.start()
    job, state = tmp_path / "job", tmp_path / "state"
    job.mkdir()
    state.mkdir()
    cpu = str(min(os.sched_getaffinity(0)))
    try:
        p = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.pous_soak", "verify", "--server", f"127.0.0.1:{ctl.port}",
                            "--setup", str(setup), "--job", str(job), "--state", str(state), "--native", str(fake.path),
                            "--until", "4000000000", "--loads", "idle", "--configs", "tuned", "--passes", "1", "--twins", "0",
                            "--cell-s", "0.3", "--cpu", cpu, "--checker-cpu", cpu, "--sampler-cpu", cpu, "--sample-cpus", cpu],
                           env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}, capture_output=True, text=True, check=False)
        assert p.returncode == 0, p.stderr[-2000:]
        assert ctl.stopped.is_set(), "a finished cross stops the server"
    finally:
        stop.set()
        with ctl.cv:
            ctl.cv.notify_all()
        loop.join()
        ctl.close()
        resp.stop()
    assert acked == [("idle", 1)], "one load, acknowledged by the serve loop"
    assert json.loads((job / "ready.json").read_text())["root"] == root
    (line,) = [json.loads(x) for x in (state / "chunks.log").read_text().splitlines()]
    v = line["validation"]
    assert line["wrong"] == 0 and line["runs"] == 9 and line["answers"] > 0 and v["blocks"]["equal"]
    bad = {k: x for k, x in v["order_violations"].items() if x and k != "..<=S4"}   # the kernel's tail after sendmsg: counted
    assert not bad, bad
    assert v["offset"]["runs"] == 9 and v["offset"]["empty"] == 0
    with np.load(job / line["chunk"]) as z:
        meta = json.loads(str(z["meta_json"]))
    assert meta["remote"]["server"] == f"127.0.0.1:{ctl.port}" and "control_clock_close" in meta
    assert len(meta["control_clock_open"]["rtt_ns"]) == 200 and min(meta["control_clock_open"]["rtt_ns"]) > 0
    net = v["network"]
    assert net["answers"] > 0 and net["negative"] == 0 and {c["config"] for c in net["cells"].values()} == {"tuned"}
    r = meta["server_status"]["responder"]
    assert r["answered"] + r["pings"] >= line["answers"], "the server's own counts, at the chunk's close"
    assert meta["control_clock_open"]["lo_ns"] <= 0 <= meta["control_clock_open"]["hi_ns"], "one host: offset 0 is bracketed"


@pytest.mark.slow
def test_a_stub_server_answers_a_remote_verifier_and_reports_the_network_term(fake, tmp_path):
    """`stub` and `verify --server` as two processes, nine cells (about 30 s): the verifier starts first and outlasts a
    listener that accepts and closes (a tunnel's, before the server is up), the stub writes the vk the verifier reads only once
    connected, takes the idle load, ends when the verifier stops it, and each chunk carries the transport, its probe at open
    and close, and the per-cell network term.  Catches the stub's wiring and the tunnel's early accept, which only the
    cross-node run would otherwise exercise."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    cpu = str(min(os.sched_getaffinity(0)))
    setup, job, state, srv = (tmp_path / n for n in ("setup", "job", "state", "srv"))
    for d in (job, state):
        d.mkdir()
    # The verifier starts first, against a listener that accepts and closes, as an ssh tunnel's does before the server is up.
    early = socket.create_server(("127.0.0.1", 0))
    port = early.getsockname()[1]
    verify = subprocess.Popen([sys.executable, "-m", "verity_vllm.pipeline.pous_soak", "verify", "--server", f"127.0.0.1:{port}",
                               "--transport", "loopback (test)", "--transport-probe", "echo probed",
                               "--setup", str(setup), "--job", str(job), "--state", str(state), "--native", str(fake.path),
                               "--until", "4000000000", "--loads", "idle", "--configs", "tuned", "--passes", "1", "--twins", "0",
                               "--cell-s", "0.3", "--cpu", cpu, "--checker-cpu", cpu, "--sampler-cpu", cpu, "--sample-cpus", cpu],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stub = None
    try:
        conn, _ = early.accept()
        conn.close()
        early.close()
        stub = subprocess.Popen([sys.executable, "-m", "verity_vllm.pipeline.pous_soak", "stub", "--job", str(srv), "--setup-out",
                                 str(setup), "--native", str(fake.path), "--segments", "1", "--control-port", str(port)],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _, verr = verify.communicate()
        assert verify.returncode == 0, verr[-2000:]
        _, err = stub.communicate()
    finally:
        for p in (verify, stub):
            if p is not None and p.poll() is None:
                p.kill()
                p.communicate()
    assert stub.returncode == 0, err[-2000:]
    served = json.loads((srv / "serve.json").read_text())
    assert served["idle_exit"] is None and served["responder"]["answered"] > 0
    (line,) = [json.loads(x) for x in (state / "chunks.log").read_text().splitlines()]
    v = line["validation"]
    assert line["wrong"] == 0 and line["runs"] == 9 and v["blocks"]["equal"] and v["offset"]["empty"] == 0
    with np.load(job / line["chunk"]) as z:
        meta = json.loads(str(z["meta_json"]))
    assert meta["transport"] == "loopback (test)" and meta["ready"]["stub"]
    assert meta["transport_probe_open"]["stdout"] == "probed\n" == meta["transport_probe_close"]["stdout"]
    net = v["network"]
    assert net["negative"] == 0 and len(net["cells"]) == 9 and all("reads" in c for c in net["cells"].values())
