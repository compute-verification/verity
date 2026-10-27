"""`mockdc.py seal`, the cost of a sealed pair link, on an injected clock: the Python path seals and opens every frame
through the certifier's two halves and reports bytes over the clock's seconds; the C fast path seals the Python's units,
byte for byte, where a C compiler and libcrypto are."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def m():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "path", list(sys.path))
        spec = importlib.util.spec_from_file_location("network_sim_mockdc", HERE / "mockdc.py")
        mod = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, spec.name, mod)
        spec.loader.exec_module(mod)
        yield mod


def test_the_python_path_opens_every_frame_and_reports_bytes_over_its_cpu_seconds(m):
    """Over 2·CHUNK + 1 frames, three chunks, on a clock that ticks one second a reading: each half takes one second a
    chunk, every unit opens to its frame (or the bench stops), and the rates are the frames' bytes over those seconds,
    with the cores one 400 Gb/s uplink takes at them."""
    ticks = itertools.count()
    frames = 2 * m.CHUNK + 1
    r = m.seal_python(64, frames, cpu=lambda: float(next(ticks)))
    assert (r["frame_bytes"], r["frames"], r["seal_cpu_s"], r["open_cpu_s"]) == (64, frames, 3.0, 3.0)
    assert r["seal_cpu_Bps"] == r["open_cpu_Bps"] == frames * 64 / 3
    assert r["seal_cpu_cores_at_line_rate"] == m.LINE_RATE_BPS / 8 / (frames * 64 / 3)


def test_the_c_fast_path_seals_the_pythons_units(m, tmp_path, monkeypatch):
    """`seal_fast.c` built against the host's libcrypto seals the Python's units, byte for byte, at both frame sizes and
    one that ends mid-block, and opens every one; a fast path whose units differ stops the bench."""
    exe = m.build_fast(tmp_path)
    if not isinstance(exe, Path):
        pytest.skip(exe)
    for b in (*m.SEAL_FRAME_BYTES, 100):
        r = m.fast_path(exe, b, 256)
        assert r["frames"] == 256 and r["seal_cpu_s"] >= 0 and r["open_cpu_s"] >= 0
    monkeypatch.setattr(m, "fast_vector", lambda b: "0" * 128)
    with pytest.raises(SystemExit, match="aren't the Python's"):
        m.fast_path(exe, 512, 64)
