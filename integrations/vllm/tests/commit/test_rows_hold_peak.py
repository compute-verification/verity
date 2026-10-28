"""A ``rows-sha512`` learning step whose h is many times `rows_hold_bytes` peaks on the device within ``BOUND``, twice the hold plus one
window's scratch, counted from the step's start (`torch.cuda.max_memory_allocated` past what was allocated then: the step's tensors,
the ring and the scratch window), and commits the roots the default hold commits with the same salt key.

The bound, for windows of r rows (r = W / ROW_BYTES = 4096 here, so j = 12) and h, leaves and nodes of 64 bytes a row:

- staging: the windows' h up to the hold, and `rows_h`'s salt scratch (192 B a row) for the window it hashes: hold + 192 r;
- framing the held h once the step's length is known: the h not yet framed and the leaves framed from them, at most the hold and
  one window's pads; with no levels kept, a fold's concatenation of the held leaves (at most the hold, and then the levels above it,
  smaller), beside the new window's leaves (64 r): at most 2 hold + 128 r;
- re-hashing the rest from host through the scratch window: the last framed window's h, kept until the step's tree (64 r), the held
  leaves, and the window's leaves (64 r) with `rows_leaves`' salt scratch in its split form (192 r, sm_90 and sm_100; sm_120 fuses
  it) or a fold: at most hold + 320 r or 2 hold + 128 r;
- fixed: the row mask table (8 KiB, on the committer's first step), each window's or fold's top nodes, which keep their last level
  launch's buffer (one 512-byte block at j = 12), and the upper tree over the tops at the step's end: under 64 KiB at 32 windows.

So BOUND = 2 HOLD + 320 r + 64 KiB.  The step runs in the GPU's leaf form and in the split form, which reaches the hold + 320 r term
on any GPU.  The default hold (1 GiB) does not bind on this step's 8 MiB of h, so the reference holds all of it, and with no levels
kept all its leaves and their concatenation: its peak passes the bound, as the step's did before the hold.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("NATIVE_COLLECT_BUILD"),
                                reason="needs a GPU and the JIT-built native_collect extension (NATIVE_COLLECT_BUILD)")

from tests.commit import test_windowed_rows_sha512 as RS  # noqa: E402
from verity_vllm.commit import step_rows as R  # noqa: E402
from verity_vllm.commit.committer import native_collect as NC  # noqa: E402

WINDOW_MB = 16
ROWS = (WINDOW_MB << 20) // R.ROW_BYTES                         # r: rows a window
HOLD = 2 * 64 * ROWS                                            # two windows' h
WINDOW_SCRATCH = (192 + 64 + 64) * ROWS + (64 << 10)
BOUND = 2 * HOLD + WINDOW_SCRATCH
KV = (40, 77)
PLANE = 16_500_929                                              # bf16 a token row of each of 8 planes: 31.5 windows, odd-sized


def step_tensors() -> list[tuple[str, torch.Tensor]]:
    out = []
    for i in range(8):
        v = (torch.arange(len(KV) * PLANE, dtype=torch.int32, device="cuda") * 7 + i * 31) % 251 - 125
        out.append((f"model.layers.{i}.mlp.down_proj", v.to(torch.bfloat16).view(len(KV), PLANE)))
    return out


def learn(mp, ts: list[tuple[str, torch.Tensor]], retain_levels: bool, *, hold: int | None = None,
          keys: dict[int, bytes] | None = None) -> SimpleNamespace:
    """One learning step on a fresh committer (with `keys`, a replay that takes them) -> its commitment, peak and salt keys."""
    com = RS.committer(mp, retain_levels=retain_levels)
    try:
        if hold is not None:
            com.rows_hold_bytes = hold
        com.reset()
        com.replay_row_keys = dict(keys) if keys else None
        com.install()
        so = RS._Sched(KV)
        com.runner.tensors = ts
        com.runner.input_batch = SimpleNamespace(req_ids=so.req_ids, num_reqs=len(KV), num_scheduled_tokens=None)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        com.runner.execute_model(so)
        com._end_step()
        rc = com.finalize()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        com.uninstall()
        assert com.errors == [], com.errors[:3]
        return SimpleNamespace(rc=rc, peak=peak, keys=dict(com.row_keys))
    finally:
        com.close()


@pytest.mark.parametrize("form", ["", "split"], ids=["gpu-form", "split-form"])
@pytest.mark.parametrize("retain_levels", [False, True], ids=["no-levels", "levels"])
def test_a_learning_step_past_the_hold_peaks_within_twice_the_hold(monkeypatch, retain_levels, form):
    monkeypatch.setattr(RS, "WINDOW_MB", str(WINDOW_MB))
    if form:
        monkeypatch.setenv("VERITY_ROWS_FORM", form)
    ts = step_tensors()
    small = learn(monkeypatch, ts, retain_levels, hold=HOLD)
    assert small.peak <= BOUND, f"a learning step past rows_hold_bytes={HOLD} peaked at {small.peak} bytes, over the bound {BOUND}"
    full = learn(monkeypatch, ts, retain_levels, keys=small.keys)
    assert full.keys == small.keys
    assert [s.root for s in small.rc.steps] == [s.root for s in full.rc.steps] and small.rc.run_root == full.rc.run_root
    w, wf = small.rc.extra["windows"], full.rc.extra["windows"]
    assert w["rows_hold_bytes"] == HOLD and wf["rows_hold_bytes"] == NC.ROWS_HOLD_BYTES
    assert w["hfirst_steps"] == wf["hfirst_steps"] == 1
    assert w["reupload_bytes"] > 0 and wf["reupload_bytes"] == 0, "the windows past the hold re-hash from host"
    assert (w["leaf_folds"] > 0) is not retain_levels and wf["leaf_folds"] == 0
    assert full.peak > BOUND, f"the default hold's peak ({full.peak} bytes) is within the bound {BOUND}: the step does not test it"
