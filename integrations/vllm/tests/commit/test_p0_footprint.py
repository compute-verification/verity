"""Unit tests for the collector-footprint mechanism pieces that run without a GPU:

* `_DeviceArena.take()` backpressure: the ring never grows while a fitting block is merely busy; the taker waits for `release()`;
  growth only when no block of the ring can ever fit (or VERITY_ARENA_GROW=1).
* `_HostWindows`: the raw bytes of a step retained as pinned host windows quack like the contiguous `keep` tensor for openings
  (slices that span two windows, the tail, `numel()`, `is_cuda`).
* the padded-length bookkeeping the windowed worker relies on (leaves per window sum to the step's leaves).
* the collector commits rows-sha512 on the windowed worker only, and refuses every other scheme and path.

The GPU end-to-end (roots of a windowed run == roots of the single-block run for the same manifest; B=2 green) is the pod acceptance
run, re-run with VERITY_E2E_WORKLOAD/VERITY_E2E_CASE (see `test_e2e_windowed_roots`).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.commit.committer import native_host as nh  # noqa: E402


class _FakeBlk:
    def __init__(self, n: int) -> None:
        self._n = n

    def numel(self) -> int:
        return self._n


def _arena(sizes: list[int]) -> nh._DeviceArena:
    a = nh._DeviceArena(sizes[0], n_blocks=0, extra_blocks=sizes)
    def _alloc(nbytes: int):                                         # no CUDA on the test host
        a.all.append(_FakeBlk(nbytes))
        a.allocs += 1
        return a.all[-1]

    a._alloc = _alloc                                                # type: ignore[method-assign]
    a.prealloc()
    return a


def test_take_best_fit_and_no_growth_while_busy(monkeypatch):
    monkeypatch.setattr(nh._DeviceArena, "WAIT_TIMEOUT_S", float("inf"))   # the wait below ends by the release, never by the clock
    a = _arena([64, 64, 1024])
    d1, d2 = a.take(10), a.take(10)
    assert d1.numel() == 64 and d2.numel() == 64                     # decode-sized steps never consume the prefill block
    p = a.take(700)
    assert p.numel() == 1024
    assert a.allocs == 3                                             # nothing grew
    got: list = []
    waiting = threading.Event()

    class _Cv(threading.Condition):
        def wait(self, *args, **kwargs):
            waiting.set()
            return super().wait(*args, **kwargs)

    a._cv = _Cv(a._lock)

    def taker():
        got.append(a.take(10))                                       # every block busy -> must WAIT for a release, not allocate

    t = threading.Thread(target=taker)
    t.start()
    waiting.wait()                                                   # the taker is in the arena's wait; only release() below ends it
    assert not got and t.is_alive() and a.allocs == 3
    a.release(d1)
    t.join()
    assert got and got[0] is d1 and a.waits == 1 and a.wait_s > 0


def test_take_grows_only_when_nothing_could_ever_fit():
    a = _arena([64, 64])
    b = a.take(1000)                                                 # larger than every block of the ring: growth (the arena-sizing refusal is upstream)
    assert b.numel() == 1000 and a.allocs == 3


def test_take_timeout_is_an_error(monkeypatch):
    a = _arena([64])
    a.take(1)
    monkeypatch.setattr(nh._DeviceArena, "WAIT_TIMEOUT_S", 0.0)
    with pytest.raises(RuntimeError, match="waited"):
        a.take(1)


def test_host_windows_slices_like_a_tensor():
    from verity_vllm.commit.committer.native_collect import _HostWindows

    W = 512
    raw = torch.arange(0, 1200, dtype=torch.int32).to(torch.uint8)         # 1200 used bytes -> padded 1280 (256-multiple)
    padded = -(-raw.numel() // 256) * 256
    full = torch.zeros(padded, dtype=torch.uint8)
    full[: raw.numel()] = raw
    wins = [(i // W, full[i:i + W].clone(), min(W, padded - i)) for i in range(0, padded, W)]
    hw = _HostWindows(wins, raw.numel(), W)
    assert hw.numel() == padded and hw.is_cuda is False and hw.cpu() is hw and hw.retained_bytes == padded
    for lo in (0, 255, 256, 500, 511, 512, 700, 1000, 1024, padded - 256):
        assert torch.equal(hw[lo:lo + 256], full[lo:lo + 256]), lo     # including slices spanning a window boundary
    assert hw[padded:padded + 256].numel() == 0
    # retain=sample: a SPARSE set of windows (only those holding challenged leaves); a slice into a missing window is an error, never zeros
    sparse = _HostWindows([w for w in wins if w[0] != 1], raw.numel(), W)
    assert sparse.numel() == padded and sparse.retained_bytes == padded - W
    assert torch.equal(sparse[0:256], full[0:256]) and torch.equal(sparse[1024:1280], full[1024:1280])
    with pytest.raises(KeyError):
        sparse[512:768]
    with pytest.raises(KeyError):
        sparse[500:756]                                                    # straddles into the missing window


@pytest.mark.slow
def test_window_leaf_count_bookkeeping():
    # what `_commit_step_windowed` asserts: per-window padded chunks sum to the step's padded chunks for any W (multiple of 256)
    C = 256
    for used in (1, 255, 256, 257, 100_000, 7_609 << 20):
        for W in (C, 4 * C, 256 << 20):
            padded = -(-used // C) * C
            n_w = -(-padded // W)
            per = [min(W, padded - k * W) for k in range(n_w)]
            assert all(p % C == 0 for p in per) and sum(p // C for p in per) == padded // C


def test_the_collector_commits_rows_sha512_on_the_windowed_worker_only(monkeypatch):
    """The collector's one scheme is rows-sha512, on the windowed worker (`--committer native_collect_v2b`, `--window-mb` > 0):
    another scheme, or a collector off the windowed path, is refused before the extension is built, and the Commit names its
    committers when it is asked for a retired one."""
    from verity_vllm.commit.committer import native_collect as NC
    from verity_vllm.pipeline import commit as CD

    monkeypatch.delenv("VERITY_WINDOW_MB", raising=False)
    with pytest.raises(ValueError, match="unknown commit scheme 'vllm-v1'"):
        NC.NativeCollectCommitter(None, commit_scheme="vllm-v1")
    with pytest.raises(ValueError, match="rows-sha512 is implemented on the windowed worker"):
        NC.NativeCollectCommitter(None)
    with pytest.raises(ValueError, match="retain_levels=False is implemented on the windowed path only"):
        NC.NativeCollectCommitter(None, retain_levels=False)
    args = argparse.Namespace(chunk=256, hash_threads=1, arena_mb=None)
    for kind in ("native_host", "native_host_direct", "native_collect", "native_collect_v2"):
        with pytest.raises(ValueError, match="the Commit's committers are native_collect_v2b"):
            CD.make_committer(kind, None, args, "run")


@pytest.mark.skipif(not (os.environ.get("VERITY_E2E_WORKLOAD") and os.environ.get("VERITY_E2E_CASE")), reason="GPU pod end-to-end only (needs vLLM + a workload)")
def test_e2e_windowed_roots(tmp_path: Path):
    """Same manifest, single-block path vs windowed path (256 MiB windows, host retention): identical run roots, both fail-closed PASS."""
    wl, case = os.environ["VERITY_E2E_WORKLOAD"], os.environ["VERITY_E2E_CASE"]
    base = [sys.executable, "-m", "verity_vllm.pipeline.cli", "commit", "--case", case, "--workload", wl, "--committer", "native_collect_v2b",
            "--openings", "64", "--pairs", "1", "--warmup", "1", "--max-num-seqs", os.environ.get("VERITY_E2E_B", "1"),
            "--engine-arg", "gpu_memory_utilization=" + os.environ.get("VERITY_E2E_UTIL", "0.35"), "--retain", "host"]
    if os.environ.get("VERITY_E2E_HIDDEN_SO"):
        base += ["--hidden-so", os.environ["VERITY_E2E_HIDDEN_SO"]]
    roots = []
    win = ["--window-mb", os.environ.get("VERITY_E2E_WINDOW_MB", "256"), "--window-slots", "8"]
    arms = [("block", ["--window-mb", "0"]), ("win", win + ["--no-bounded-staging"])]
    if os.environ.get("VERITY_E2E_BOUNDED", "1") != "0":
        # the BOUNDED staging arms: same roots as the retained implementation -- (a) retain=host with the hidden family's
        # interior chunks excluded from host (they open by replay), (b) retain=none (everything by replay); both with openings after release
        fam = os.environ.get("VERITY_E2E_EXCLUDE", "fa3_hidden_m1")
        arms += [("win-bounded-x", win + ["--bounded-staging", "--retain-exclude", fam, "--open-after-release", "--learn-budget-mb", os.environ.get("VERITY_E2E_LEARN_MB", "4096")]),
                 ("win-bounded-none", win + ["--bounded-staging", "--open-after-release", "--learn-budget-mb", os.environ.get("VERITY_E2E_LEARN_MB", "4096")])]
    for name, extra in arms:
        cmd = list(base)
        if name == "win-bounded-none":
            cmd[cmd.index("--retain") + 1] = "none"
        p = subprocess.run(cmd + extra + ["--out", str(tmp_path / name)], capture_output=True, text=True)
        vp = tmp_path / name / "verdict.json"
        assert vp.exists(), f"{name}: exit {p.returncode}, no verdict\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
        v = json.loads(vp.read_text())
        # a MECHANISM fixture has no Build (no Programs), so C2 of record cannot run: the ONLY verdict failure this test accepts is
        # the verdict naming exactly that ('sampled replay None -- not run'; commit_delta exits 3 on it); every other check (bindings,
        # openings, bounded staging, replay openings, admission) must hold -- the workload's Commit of record runs the full verdict with Programs.
        _c2_only = str(v.get("first_fail_reason") or "").startswith("C2 of record not established: sampled replay None -- not run")
        assert v["commit_pass"] is True or _c2_only, f"{name}: exit {p.returncode}: {v.get('first_fail_reason') or v.get('commit_fail_reason')}"
        assert p.returncode == 0 or (p.returncode == 3 and _c2_only), f"{name}: exit {p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
        rows = [json.loads(l) for l in (tmp_path / name / "runs.jsonl").read_text().splitlines()]
        rr = {r["commit"]["run_root"] for r in rows if r["arm"] != "control"}
        assert len(rr) == 1
        roots.append(rr.pop())
        if name.startswith("win-bounded"):
            for r in rows:
                if r["arm"] == "control":
                    continue
                bs = r["commit"]["extra"]["bounded_staging"]
                assert bs["enabled"] and not bs["learned_only_steps"] and bs["plan_mismatches"] == 0, bs
                oar = r["commit"]["openings_after_release"]
                assert oar["verified"] == oar["n"] and oar["replay_root_equal"] is True, oar
                assert (oar["source"] == "replay") if name == "win-bounded-none" else (oar["source"] in ("host+replay", "replay")), oar["source"]
                if name == "win-bounded-x":
                    assert bs["excluded_bytes"] > 0 and fam in (oar.get("replay_families") or {}), (bs["excluded_bytes"], oar.get("replay_families"))
    assert len(set(roots)) == 1, f"the run root differs across arms {dict(zip([a for a, _ in arms], roots))} (layout must be unchanged under chunk-leaf-v1)"


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("NATIVE_COLLECT_BUILD"), reason="needs a GPU and the JIT-built native_collect extension")
@pytest.mark.parametrize("n_leaves,j", [(1, 4), (5, 4), (16, 4), (17, 4), (48, 4), (49, 4), (1000, 4), (3, 6), (64, 6), (65, 6), (200, 3)])
def test_tree_window_equals_whole_tree(n_leaves: int, j: int):
    """Per-window subtrees (window = 2^j leaves, `StepHasher.tree_window`) + the ordinary tree over the window tops == the whole-step tree
    (`StepHasher.tree`): same root AND the same nodes at every level (the levels are what openings read); `tree_window_top` gives the
    same window tops.  Covers: a single collapsed
    window (n < 2^j, height < j), exact multiples, a partial last window that collapses below the window level (the lift chain), and
    steps with many windows."""
    from verity_vllm.commit.committer.native_collect import ext

    H = ext(os.environ["NATIVE_COLLECT_BUILD"]).StepHasher()
    side = torch.cuda.Stream()
    g = torch.Generator(device="cuda").manual_seed(n_leaves * 131 + j)
    leaves = torch.randint(-(2**31), 2**31 - 1, (n_leaves, 8), dtype=torch.int32, device="cuda", generator=g)
    ref_levels, ref_root, _ = H.tree(leaves.clone(), side.cuda_stream, True, torch.empty(0, dtype=torch.uint8))
    # windowed: heights / levels exactly as the worker computes them
    height, m = 0, n_leaves
    while m > 1:
        m = (m + 1) // 2
        height += 1
    n_levels = min(j, height) + 1
    sizes = [-(-n_leaves // (1 << L)) for L in range(n_levels)]
    views = [torch.empty((nL, 8), dtype=torch.int32, pin_memory=True) for nL in sizes]
    tops = []
    per_win = 1 << j
    for k in range(-(-n_leaves // per_win)):
        lv_k = leaves[k * per_win:(k + 1) * per_win].clone()
        n_k = int(lv_k.shape[0])
        dsts = [views[L][(k << (j - L)):(k << (j - L)) + -(-n_k // (1 << L))] for L in range(n_levels)]
        tops.append(H.tree_window(lv_k, n_levels, side.cuda_stream, dsts))
    assert len(tops) == sizes[-1]
    tops_t = torch.cat(tops) if len(tops) > 1 else tops[0]
    upper, root, _ = H.tree(tops_t, side.cuda_stream, True, torch.empty(0, dtype=torch.uint8))
    levels = list(views[:n_levels - 1]) + list(upper)
    assert root == ref_root
    # retain_levels=False: the same window tops on the device, with no host copy and no host sync -- ordered on `side` only (the worker
    # reads them there), so the windows' leaves outlive those kernels and the tops are read on that stream
    wins = [leaves[k * per_win:(k + 1) * per_win].clone() for k in range(len(tops))]
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        dev_tops = torch.cat([H.tree_window_top(w, n_levels, side.cuda_stream) for w in wins])
        assert torch.equal(dev_tops, tops_t)
        assert H.tree(dev_tops, side.cuda_stream, False, torch.empty(0, dtype=torch.uint8))[1] == ref_root
    assert len(levels) == len(ref_levels)
    for L, (a, b) in enumerate(zip(levels, ref_levels)):
        assert a.shape == b.shape, L
        assert torch.equal(a.cpu(), b.cpu()), f"level {L} differs"
