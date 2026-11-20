"""A windowed record run whose learning step outgrows the transient host budget fails by name and never crashes the serving thread, on
the real collector, window ring and kernels.

GLM-5.3 TP8 eager, rows-sha512: the worker abandoned step 0 past the 4 GiB budget, erasing the step's
windows, and the serving thread then segfaulted in `Collector::ensure_windows` on them.  Here the budget is 16 KiB and each step ~56 KiB
over a ring of four 8 KiB windows, so the worker abandons step 0 at its third window while the serving thread is still staging it: every
later acquisition is refused, every step is a placeholder, the run has no root and names the failure, and its verdict leads with that
name.  The next run on the same collector and ring (warm-up, then record) commits the reference roots: the abandon handed each slot back
once and left no failure behind.
"""
from __future__ import annotations

import os
import sys
import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("NATIVE_COLLECT_BUILD"),
                                reason="needs a GPU and the JIT-built native_collect extension (NATIVE_COLLECT_BUILD)")

from tests.commit import test_windowed_rows_sha512 as RS  # noqa: E402


def run_failing(com, sched):
    """RS.commit_run without its no-error assertion: the steps, then finalize."""
    com.reset()
    com.install()
    for kv in sched:
        so = RS._Sched(kv)
        com.runner.tensors = RS.step_tensors(kv)
        com.runner.input_batch = SimpleNamespace(req_ids=so.req_ids, num_reqs=len(kv), num_scheduled_tokens=None)
        com.runner.execute_model(so)
        com._end_step()
    rc = com.finalize()
    com.uninstall()
    return rc


def test_a_slot_taken_while_the_worker_abandons_goes_back_once(monkeypatch):
    """The interleaving that broke the recovery above, forced: the serving thread, out of slots, takes the slot the worker just released
    and waits for the GIL, which the worker keeps until it has abandoned the step.  abandon_step hands that slot back (its last occupant
    was the step's window 0), so the serving thread refuses without handing it back again: the ring holds each of its four slots once."""
    com = RS.committer(monkeypatch, retain_levels=False)
    nc, side = com._nc, torch.cuda.Stream()
    t = torch.arange(40 * 1024 // 4, dtype=torch.int32, device="cuda")          # five 8 KiB windows over a ring of four
    seen: dict[str, tuple] = {}

    def worker() -> None:
        seen["window"] = nc.wait_window(0, 0, side.cuda_stream)                  # marked when the serving thread runs out of slots
        nc.release_slot(int(seen["window"][2]), side.cuda_stream)
        while nc.free_slots():                                                   # the serving thread took it; the GIL stays here
            pass
        nc.abandon_step(0, "forced abandon", side.cuda_stream)

    interval = sys.getswitchinterval()
    sys.setswitchinterval(60.0)                                                  # no forced GIL hand-off between release and abandon
    try:
        com.reset()
        nc.begin_step(1, 0, 0, True, 0)
        th = threading.Thread(target=worker)
        th.start()
        with pytest.raises(RuntimeError, match="the collector failed earlier in this run: forced abandon"):
            nc.stage(com._intern("t"), t)
        th.join()
        assert seen["window"][0] and nc.free_slots() == 4, (seen, nc.free_slots())
        nc.flush()
        nc.end_step()
        nc.end_run()
        assert nc.failure == "" and nc.free_slots() == 4
    finally:
        sys.setswitchinterval(interval)
        com.close()
