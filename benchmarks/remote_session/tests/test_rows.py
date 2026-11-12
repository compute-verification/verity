"""`rows.py --ctas-at`: a row count runs at the receipt CTA counts its word names, each with the arms it names, in
the word's order, and at `--sketch-ctas` alone where no word names it; a word for a row count that doesn't run, or one
whose counts aren't numbers or whose arms aren't `rows.ARMS`', ends the run saying what a word is. `rows.reference`, the whole-store check's receipts, is the reference
receipter's at both row counts, at the int8 extremes and at any first block."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rows
from rows import rb


def test_a_row_count_takes_its_words_counts_in_order_and_the_default_elsewhere():
    assert rows.ctas_at(["64=96,128,264"], [16, 64, 16, 64], 128) == {16: [(128, ())],
                                                                      64: [(96, ()), (128, ()), (264, ())]}
    assert rows.ctas_at([], [64, 16], 0) == {64: [(0, ())], 16: [(0, ())]}
    assert rows.ctas_at(["64=132", "16=128"], [64, 16], 64) == {64: [(132, ())], 16: [(128, ())]}


def test_a_count_takes_the_arms_it_names():
    assert rows.ctas_at(["64=264,264+p0,264+after,264", "16=128+p0+after"], [64, 16], 128) == {
        64: [(264, ()), (264, ("p0",)), (264, ("after",)), (264, ())], 16: [(128, ("p0", "after"))]}


@pytest.mark.parametrize("word", ["48=128", "64", "64=", "64=128,", "64=a", "x=128", "64=-1", "64=128+", "64=128+x",
                                  "64=+p0", "64=128+p0+"])
def test_a_word_that_names_no_running_row_count_or_no_counts_ends_the_run(word):
    with pytest.raises(SystemExit, match="ROWS=N"):
        rows.ctas_at([word], [64], 128)


@pytest.mark.parametrize("k", rb.ROWS)
def test_the_whole_store_reference_is_the_reference_receipters(k):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(k)
    seed = rb.seed(rng.bytes(rb.CHALLENGE_BYTES), 0, bytes(rb.DIGEST_BYTES))
    blocks = [rng.bytes(rb.BLOCK_BYTES) for _ in range(5)]
    blocks += [bytes([b]) * rb.BLOCK_BYTES for b in (0x80, 0x7F, 0xFF, 0xA5)]
    # every byte the int8 extreme whose product with row 0's sign is positive: the largest σ a row can hold
    row0 = np.unpackbits(np.frombuffer(rb.sign_bits(seed, k)[0], np.uint8), bitorder="little")
    blocks.append(np.where(row0 == 1, 0x80, 0x7F).astype(np.uint8).tobytes())
    j0 = (1 << 40) + 3
    got = rows.reference(torch, seed, k)(torch.frombuffer(bytearray(b"".join(blocks)), dtype=torch.uint8), j0)
    r = rb.receipter(rb.SKETCH_SHA512, seed, k)
    assert got == b"".join(r(j0 + i, b) for i, b in enumerate(blocks))


def test_a_row_keeps_each_slow_pass_with_every_gpus_part():
    observed = [0, 10, 20, 35, 45, 55, 66]   # ns: the fourth interval is 1.5 medians, the last exactly 1.1
    trace = [{"launch": [at, 0, 0, 0, 0, 0, at, at, at, at, 0, 0, 0, 0, 0], "wait": [at, at], "gpu_ms": 1.0 + i,
              "part_ms": [0.5 + i], "host_ms": []} for i, at in enumerate(observed)]
    report = rows.device.stall_report(observed, trace, None, (0, 0), [])
    assert rows.slow(report) == {"over_median": 1.1, "passes": [3], "each": [
        {"pass": 3, "interval_ms": 15 / 1e6, "gpu_ms": 4.0, "part_ms": [3.5]}]}
    # every pass in order, the slow one among them at its own index: the first has no interval
    parts = {"gpu_ms": [t["gpu_ms"] for t in trace], "part_ms": [t["part_ms"] for t in trace]}
    kept = rows.every(report, parts)
    assert kept["interval_ms"] == [None, 10 / 1e6, 10 / 1e6, 15 / 1e6, 10 / 1e6, 10 / 1e6, 11 / 1e6]
    assert kept["gpu_ms"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert kept["part_ms"] == [[0.5], [1.5], [2.5], [3.5], [4.5], [5.5], [6.5]]
    assert kept["part_ms"][3] == rows.slow(report)["each"][0]["part_ms"]
    # the host's side of every pass rides with them: a launch's stamps from its own start, None where it has none
    trace[2]["launch"] = None
    host = rows.device.host_side(trace, [1_000_000 * x for x in observed])
    assert host["launch_ms"] == [[0.0] * 4, [0.0] * 4, None, [0.0] * 4, [0.0] * 4, [0.0] * 4, [0.0] * 4]
    assert host["launch_at_ms"][2] is None and host["launch_at_ms"][0] == 0.0
    assert host["root_at_ms"] == [0.0, 10.0, 20.0, 35.0, 45.0, 55.0, 66.0]
    kept = rows.every(report, {**parts, **host, "offload_ms": [0.25] * 7})
    assert kept["launch_ms"] == host["launch_ms"] and kept["root_at_ms"][3] == 35.0 and kept["offload_ms"] == [0.25] * 7
    assert set(rows.every(report, parts)) == {"interval_ms", "gpu_ms", "part_ms"}


def test_a_backend_builds_its_store_with_the_stores_own_shape(monkeypatch, tmp_path):
    """The backend's shape also holds its side stream's arms, which no store takes: its own construction line, run
    here over stand-ins, hands the store arguments the store's real signature binds."""
    import inspect
    from types import SimpleNamespace

    device = rows.device
    real, built = inspect.signature(device.DeviceStore.__init__), {}

    def store(*args, **kwargs):
        built.update(real.bind(None, *args, **kwargs).arguments)
        return SimpleNamespace()

    lib = lambda path, torch: SimpleNamespace(so=SimpleNamespace(rs_enable_peers=lambda: 0))  # noqa: E731
    monkeypatch.setattr(device, "Lib", lib)
    monkeypatch.setattr(device, "Kv", lambda path, torch: SimpleNamespace())
    monkeypatch.setattr(device, "Gpu", lambda torch, index: SimpleNamespace(index=index))
    monkeypatch.setattr(device, "DeviceStore", store)
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1, synchronize=lambda: None))
    so = tmp_path / "session.so"
    so.write_bytes(b"")
    backend = device.GpuBackend(torch, str(so), str(so), 1 << 20, seed=7, gpus=1, sketch_ctas=132)
    assert backend.shape["side_priority"] == -1 and backend.shape["side_after"] is False
    assert built["sketch_ctas"] == 132 and built["blocks"] == 1 << 20
    assert device.store_shape(backend.shape) == {"chunk": 1 << 18, "kept_from": 6, "sketch_ctas": 132,
                                                 "payload_kept_from": 6}
    with pytest.raises(TypeError, match="side_priority"):
        store(None, None, None, None, 1 << 20, b"", 7, **backend.shape)


def test_the_16_row_own_tile_gate_is_off_unless_asked(monkeypatch):
    """`--own-tiles-16` reaches the backend, and without it the backend's gate runs its pass with no receipt grid at
    64 rows only."""
    import argparse
    import inspect
    from types import SimpleNamespace

    device = rows.device
    assert inspect.signature(device.GpuBackend.__init__).parameters["own_tiles_16"].default is False
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    device.add_arguments(ap)
    seen = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(device, "GpuBackend", lambda *args, **kwargs: seen.append(kwargs))
    for extra, want in (([], False), (["--own-tiles-16"], True)):
        device.from_arguments(ap.parse_args(["--lib", "session.so", "--kv", "kv.so", *extra]))
        assert seen[-1]["own_tiles_16"] is want
    script = (Path(__file__).resolve().parents[1] / "rows.sh").read_text()
    assert script.count("${OWN_TILES_16:+--own-tiles-16}") == 2   # the rows step's gate and the in-place gate
