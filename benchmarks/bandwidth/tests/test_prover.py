"""The honest prover prepares before every window: an untimed pass under a throwaway seed, then the read that leaves L2
clean, and only then `σ`.  A fake tagger records the harness's calls; nothing runs on a GPU."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import gpu  # noqa: E402
from verity.core.protocols.bandwidth import audit, tags  # noqa: E402

THROWAWAY = bytes(tags.SEED_BYTES)
WORKSPACE, SCRUB = 2, 3


class _Commitment:
    root = bytes(64)

    def open(self, j, store_tree):
        return None


class _Tagger:
    """`gpu.Tagger`'s calls, in `log`: a pass records whether it ran under the throwaway seed."""

    def __init__(self, log: list) -> None:
        self.log, self.next = log, 0

    def alloc(self, nbytes: int) -> int:
        self.next += 1
        return self.next

    def free(self, ptr: int) -> None:
        pass

    def fill_test_store(self, store: int, blocks: int) -> None:
        pass

    def workspace_bytes(self, tag: str, blocks: int) -> int:
        return 64

    def last_pass_ms(self) -> tuple[float, float]:
        return 0.0, 0.0

    def memset(self, ptr: int, value: int, nbytes: int) -> None:
        self.log.append(("memset", ptr))

    def read_scrub(self, ptr: int, nbytes: int) -> None:
        self.log.append(("scrub", ptr, nbytes))

    def tag_pass(self, tag, store, blocks, seed, workspace, stream=0):
        self.log.append(("pass", seed == THROWAWAY))
        return _Commitment()

    def tag_pass_prefix(self, tag, store, blocks, tagged, seed, workspace, stream=0):
        self.log.append(("prefix", seed == THROWAWAY))
        return _Commitment()


class _Window:
    def __init__(self, log: list) -> None:
        self.log = log

    def start(self) -> bytes:
        self.log.append(("start",))
        return bytes(range(32))

    def commit(self, root: bytes) -> tuple[int, ...]:
        self.log.append(("commit",))
        return (0,)


@pytest.mark.parametrize("tagged", [None, 30])
def test_the_prover_prepares_before_it_asks_for_the_seed(tagged) -> None:
    log: list = []
    prover = gpu.Prover(_Tagger(log), "ro-blake3/v1", 1, 37, WORKSPACE, SCRUB)
    _, drawn, _ = prover.commit(_Window(log), tagged)
    poison = [("memset", WORKSPACE)] if tagged else []
    timed = ("prefix" if tagged else "pass", False)
    assert drawn == (0,)
    assert log == [("pass", True), *poison, ("scrub", SCRUB, gpu.SCRUB_BYTES), ("start",), timed, ("commit",)]


def test_every_window_of_the_harness_is_prepared(monkeypatch, tmp_path) -> None:
    """Honest windows, skip controls and the openings check alike: each `σ` follows a throwaway pass and the read, with
    nothing between them but a skip control's poisoned workspace, and comes before the timed pass."""
    log: list = []
    monkeypatch.setattr(gpu, "Tagger", lambda lib: _Tagger(log))
    start = audit.Window.start

    def logged(self):
        log.append(("start",))
        return start(self)

    monkeypatch.setattr(audit.Window, "start", logged)
    leaves = tmp_path / "leaves.bin"
    leaves.write_bytes(bytes(range(64)) * 37)
    a = SimpleNamespace(lib="fake", leaves=leaves, blocks=37, samples=4, deadline_ns=1_200_000, windows=3, skip=[0.1],
                        tags=sorted(gpu.TAG_IDS), source=b"s".hex())
    out = gpu._windows(a)
    starts = [i for i, e in enumerate(log) if e == ("start",)]
    assert len(starts) == len(a.tags) * (a.windows + len(a.skip) + 1)
    assert len(out["results"]) == len(a.tags) * (a.windows + len(a.skip)) and len(out["openings_check"]) == 3
    for i in starts:
        before = log[i - 3:i] if log[i - 2][0] == "memset" else log[i - 2:i]
        assert before[0] == ("pass", True) and before[-1][0] == "scrub" and before[-1][2] == gpu.SCRUB_BYTES
        assert log[i + 1] in (("pass", False), ("prefix", False))
