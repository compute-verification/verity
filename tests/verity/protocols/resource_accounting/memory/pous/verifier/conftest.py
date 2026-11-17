"""Every memory attester in this suite runs on an injected clock (AGENTS.md "Tests"). `tests/test_no_wall_clock.py`
scans for clock reads in test source, and cannot see an `audit.UnsignedRounds`, `ContinuousMemoryAttester` or
`memory_attester.MemoryAttester` left on its default `time.perf_counter_ns`, or a `wipe_challenger.WipeMemoryAttester`
left on its default `streamed_wipe.raw_clock_ns`; such a test passes or fails with the load on `check`'s slot. Here each
one's default clock, whichever it is, is replaced by one that fails the test that reads it, every time. The test signer's
leaf states and session logs go under the suite's basetemp (`pous_test_signer.DIR`)."""

from __future__ import annotations

import inspect
import shutil

import pous_test_signer
import pytest

from verity.protocols.resource_accounting.memory.pous.attester import memory_attester
from verity.protocols.resource_accounting.memory.pous.verifier import audit
from verity.protocols.resource_accounting.memory.pous.attester import wipe_challenger
from verity.protocols.resource_accounting.memory.pous.attester import wipe_stream
from verity.protocols.resource_accounting.memory.pous.attester.continuous import ContinuousMemoryAttester


def _wall_clock() -> int:
    raise AssertionError("a memory attester read its default wall clock: pass clock=, e.g. pous_fakes.Clock(step=1_000)")


@pytest.fixture(autouse=True)
def _no_default_clock(monkeypatch):
    for cls in (audit.UnsignedRounds, ContinuousMemoryAttester, memory_attester.MemoryAttester,
                wipe_challenger.WipeMemoryAttester):
        init = cls.__init__
        if inspect.signature(init).parameters["clock"].kind is inspect.Parameter.KEYWORD_ONLY:
            monkeypatch.setattr(init, "__kwdefaults__", {**init.__kwdefaults__, "clock": _wall_clock})
            continue
        defaulted = [p.name for p in inspect.signature(init).parameters.values()
                     if p.kind is p.POSITIONAL_OR_KEYWORD and p.default is not p.empty]
        defaults = list(init.__defaults__)
        defaults[defaulted.index("clock")] = _wall_clock
        monkeypatch.setattr(init, "__defaults__", tuple(defaults))


@pytest.fixture(autouse=True)
def _nothing_committed(monkeypatch):
    """No configuration is committed unless a test hands in its own set (`memory_attester.committed`): the committed
    hashes are the `verity` command's file, which only the table's tests under `tests/tools/verity/cli/` read."""
    monkeypatch.setattr(memory_attester, "committed", lambda: frozenset())


@pytest.fixture(scope="session")
def wipe_streamer(tmp_path_factory) -> wipe_stream.Streamer:
    """`wipe_stream.c` built once for the suite, on two threads."""
    if shutil.which("cc") is None:
        pytest.skip("no C compiler (cc) on this host to build wipe_stream.c")
    return wipe_stream.Streamer.build(tmp_path_factory.mktemp("wipe-stream"), threads=2)


@pytest.fixture(autouse=True, scope="session")
def _test_signer_dir(tmp_path_factory):
    """`pous_test_signer`'s files under the suite's basetemp, never `$TMPDIR`."""
    pous_test_signer.DIR = tmp_path_factory.mktemp("test-signer")
    yield
    pous_test_signer.DIR = None
