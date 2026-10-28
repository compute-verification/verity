"""--retain device (`commit.committer.device_pool`): a window's committed values wait in a bounded device pool for its draw-gated
release, which keeps exactly the drawn leaves, byte-equal to what --retain host keeps of them, and refuses any other leaf by name.
The pool is a CPU tensor here; the windowed collector puts the same blocks on the GPU."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tests.check.opened import commit_steps
from verity_vllm.commit import opened as OV
from verity_vllm.commit import step_rows as R
from verity_vllm.commit import store_dump as SD
from verity_vllm.commit.committer import device_pool as DP
from verity_vllm.commit.committer.host_retention import _HostRanges
from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES


def steps(k: int) -> dict:
    return {0: [("model.embed_tokens", bytes(range(256)) * 5 * k), ("model.layers.0.self_attn.qkv_proj", b"\x07" * 900 * k)],
            1: [("model.embed_tokens", bytes(reversed(range(256))) * 3 * k), ("runner.sampler", b"\x01\x02\x03\x04")],
            2: [("model.layers.0.mlp.down_proj", bytes(i % 251 for i in range(3000 * k)))]}


@pytest.fixture(params=[None, R.NAME], ids=["vllm-v1", R.NAME])
def cs(request):
    return request.param


def committed(cs, values=None):
    """Steps committed as --retain host keeps them (their blocks, ``host``), then moved into a device pool as --retain device holds them:
    ``values`` (`commit_steps`' form), else `steps`'."""
    k = (R.ROW_BYTES if cs else GPU_CHUNK_BYTES) // 256
    dev, rc = commit_steps(values or steps(k), commit_scheme=cs)
    host = {s: bytes(blk.numpy()) for s, (blk, _u) in dev._gpu_blocks.items()}
    dev.device_pool = DP.DevicePool(sum(len(b) for b in host.values()) + 4096 * len(host), device="cpu")
    for s, (blk, used) in list(dev._gpu_blocks.items()):
        v, _w = dev.device_pool.take(int(blk.numel()), DP.pool_key(dev))
        v.copy_(blk)
        held = _HostRanges([(0, int(blk.numel()), v)], used, chunk=dev.chunk)
        held.on_device = True
        dev._gpu_blocks[s] = (held, used)
    return host, rc, dev


def leaf_bytes(com, s, i):
    return bytes(com.open_range(s, i, i + 1).value)


def host_leaf(host, C, s, i):
    return host[s][i * C:(i + 1) * C]


def test_the_release_keeps_exactly_the_drawn_leaves_byte_equal_to_host(cs, tmp_path):
    host, rc, dev = committed(cs)
    C = dev.chunk
    drawn = {0: [1, 2], 2: [0, 5]}
    assert leaf_bytes(dev, 0, 3) == host_leaf(host, C, 0, 3)
    with pytest.raises(ValueError, match="held on the device until their window's draw-gated release"):
        SD.dump(dev, rc, tmp_path / "early")
    rec = DP.release(dev, drawn)
    assert rec["drawn_leaves"] == 4 and rec["kept_bytes"] == 4 * C == sum(b.retained_bytes for b, _u in dev._gpu_blocks.values())
    assert rec["dropped_bytes"] == sum(len(b) for b in host.values()) - 4 * C
    assert rec["device_freed_bytes"] > 0 and dev.device_pool.held == 0
    assert dev._gpu_blocks[1][0].retained_bytes == 0
    SD.dump(dev, rc, tmp_path / "w")
    store, run = SD.load(tmp_path / "w")
    assert [s.root for s in run.steps] == [s.root for s in rc.steps]
    for com in (dev, store):
        for s, leaves in drawn.items():
            for i in leaves:
                assert leaf_bytes(com, s, i) == host_leaf(host, C, s, i)
    assert sum((tmp_path / "w" / "raw" / f"s{s}.bin").stat().st_size for s in range(3)) == 4 * C


def test_a_dropped_leaf_is_refused_by_name_never_replayed(cs, tmp_path):
    host, rc, dev = committed(cs)
    DP.release(dev, {0: [1]})
    SD.dump(dev, rc, tmp_path)
    for com in (dev, SD.load(tmp_path)[0]):
        assert leaf_bytes(com, 0, 1) == host_leaf(host, dev.chunk, 0, 1)
        with pytest.raises(RuntimeError, match="draw-gated release"):
            com.open_range(0, 0, 1)
        with pytest.raises(RuntimeError, match="draw-gated release"):
            com.open_range(1, 0, 1)
    replay, _ = commit_steps(steps((R.ROW_BYTES if cs else GPU_CHUNK_BYTES) // 256), commit_scheme=cs)
    with pytest.raises(RuntimeError, match="draw-gated release"):
        dev.open([(0, 3)], values_from=replay)
    with pytest.raises(KeyError, match="draw-gated release"):
        dev._gpu_blocks[0][0][0:dev.chunk]


def test_a_released_window_dumps_slim_to_its_drawn_leaves(cs, tmp_path):
    C = R.ROW_BYTES if cs else GPU_CHUNK_BYTES
    host, rc, dev = committed(cs, {0: [("model.embed_tokens", (bytes(range(251)) * (200 * C // 251 + 1))[:200 * C])]})
    runs = [(3, 6), (100, 101), (199, 200)]
    DP.release(dev, {0: [i for a, b in runs for i in range(a, b)]})
    doc = SD.dump(dev, rc, tmp_path / "w")
    st = doc["steps"][0]
    assert st["drawn_only"] and st["ranges"] == [[a * C, b * C] for a, b in runs]
    assert 0 < len(st["nodes"]) < sum(st["levels"]) // 3
    assert (tmp_path / "w" / "levels" / "s0.bin").stat().st_size == SD._node(dev) * len(st["nodes"])
    assert not (tmp_path / "w" / "keys.bin").exists() and (st.get("salted") == [list(r) for r in runs]) == bool(cs)
    store, run = SD.load(tmp_path / "w")
    assert [s.root for s in run.steps] == [s.root for s in rc.steps]
    rd = OV.OpenedReader(store, run)
    for a, b in runs:
        for lo in range(a, b):
            for hi in range(lo + 1, b + 1):
                assert rd(0, "model.embed_tokens", lo * C, hi * C) == host[0][lo * C:hi * C]
    assert rd(0, "model.embed_tokens", 3 * C + 7, 6 * C - 3) == host[0][3 * C + 7:6 * C - 3]
    with pytest.raises(RuntimeError, match="draw-gated release"):
        store.open_range(0, 6, 7)
    again = SD.dump(store, run, tmp_path / "again")
    assert again["steps"][0]["nodes"] == st["nodes"] and again["steps"][0]["ranges"] == st["ranges"]


def test_a_release_names_only_committed_leaves_and_happens_once(cs):
    _host, _rc, dev = committed(cs)
    with pytest.raises(KeyError, match="steps \\[7\\]"):
        DP.release(dev, {7: [0]})
    with pytest.raises(IndexError, match="leaves, not"):
        DP.release(dev, {0: [10 ** 6]})
    DP.release(dev, {})
    with pytest.raises(RuntimeError, match="released already"):
        DP.release(dev, {0: [0]})


class _Seen(threading.Condition):
    """A condition that says when a take has started waiting on it."""

    def __init__(self):
        super().__init__()
        self.waiting = threading.Event()

    def wait(self, timeout=None):
        self.waiting.set()
        return super().wait(timeout)


def test_a_full_pool_waits_for_an_earlier_windows_release():
    pool = DP.DevicePool(4096, device="cpu")
    pool._cv = _Seen()
    pool.take(3072, "w0")
    got = {}
    t = threading.Thread(target=lambda: got.update(v=pool.take(2048, "w1")))
    t.start()
    pool._cv.waiting.wait()
    assert "v" not in got
    assert pool.free("w0") == 3072
    t.join()
    assert got["v"][0].numel() == 2048 and pool.waits == 1 and pool.held_by("w1") == 2048


def test_the_open_windows_own_bytes_are_refused_not_waited_on():
    pool = DP.DevicePool(4096, device="cpu")
    pool.take(3072, "w0")
    with pytest.raises(DP.DevicePoolFull, match="raise --retain-device-gb"):
        pool.take(2048, "w0")
    pool.take(1024, "w1")
    pool._cv = _Seen()
    got = {}
    t = threading.Thread(target=lambda: got.update(e=pytest.raises(RuntimeError, pool.take, 2048, "w2")))
    t.start()
    pool._cv.waiting.wait()
    pool.fail("the run failed")
    t.join()
    assert "the run failed" in str(got["e"].value)


def test_freed_pieces_coalesce_first_fit():
    pool = DP.DevicePool(8 * 512, device="cpu")
    for k in "abcd":
        pool.take(2 * 512, k)
    pool.free("b")
    pool.free("c")
    v, _w = pool.take(4 * 512, "e")
    assert v.data_ptr() == pool.slab[2 * 512:].data_ptr()


def opts(retention=(), **kw):
    r = {"retain": "device", "retain_device_gb": 32.0, "retain_exclude": "", **dict(retention)}
    return SimpleNamespace(**{"retention": SimpleNamespace(**r), "commit": True, "scheme": R.NAME, "window_gb": 32.0, **kw})


def test_serve_openai_refuses_retain_device_without_the_rows_commit_or_room_for_a_window():
    from verity_vllm.pipeline import serve_seal as SS

    assert SS.retention_refusal(opts()) is None
    assert "needs the Commit under" in SS.retention_refusal(opts(commit=False))
    assert "needs the Commit under" in SS.retention_refusal(opts(scheme=SS.V1))
    assert "is under --window-gb" in SS.retention_refusal(opts(retention={"retain_device_gb": 8.0}))
    assert SS.retention_refusal(opts(retention={"retain": "host", "retain_device_gb": 1.0})) is None
