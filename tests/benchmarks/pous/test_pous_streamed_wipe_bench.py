"""`benchmarks/pous/streamed_wipe/bench.py`'s session step at a tiny size on the CPU, the challenger's clock injected:
under `--verdict python` no Lean judges the honest session, so it is recorded undecided (`wipe_verdict.NO_JUDGE`), never
accepted, and the Python's accept is only in its own side."""
from __future__ import annotations

import functools
import importlib.util
import socket
from pathlib import Path

import pytest

pytest.importorskip("verity.protocols.resource_accounting.memory.pous.verifier")

BENCH = Path(__file__).resolve().parents[3] / "benchmarks" / "pous" / "streamed_wipe" / "bench.py"
POUS_TESTS = Path(__file__).resolve().parents[2] / "verity" / "protocols" / "resource_accounting" / "memory" / "pous" / "verifier"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_under_python_an_honest_session_is_undecided_and_never_recorded_accepted(monkeypatch):
    bench = _load("streamed_wipe_bench", BENCH)
    clock = _load("pous_fakes", POUS_TESTS / "pous_fakes.py").Clock(10 ** 9, 1_000)
    monkeypatch.setattr(bench, "SESSION_LABELS", 1 << 10)
    monkeypatch.setattr(bench.sw, "DEPLOYED_K", 64)
    monkeypatch.setattr(bench.WC, "WipeMemoryChallenger", functools.partial(bench.WC.WipeMemoryChallenger, clock=clock))
    got = bench.session(_free_port(), "python")
    assert got["ended"] == bench.sw.COMPLETE and got["verdicts"] == {"python": {"accepted": True, "reasons": []}}
    assert (got["accepted"], got["outcome"], got["reasons"]) == (False, "undecided", [bench._wipe_verdict().NO_JUDGE])
