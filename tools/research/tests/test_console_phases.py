"""The console's held-idle phases: the Commit leases it reads from circuits' row timelines on node 1."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

_spec = importlib.util.spec_from_file_location("verity_console", Path(__file__).parents[1] / "console" / "verity_console.py")
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)

T0 = datetime(2026, 10, 2, 23, tzinfo=timezone.utc).timestamp()


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def row(root: Path, name: str, events: list[tuple], started: float | None = None, released: dict | None = None) -> Path:
    d = root / "cov-x" / name
    d.mkdir(parents=True)
    (d / "timeline.jsonl").write_text("".join(json.dumps({"stage": s, "ev": ev, "t": t, "pid": 7}) + "\n" for s, ev, t in events))
    if started is not None:
        (d / "gpu_lease.started").write_text(iso(started) + "\n")
    if released is not None:
        (d / "gpu_lease.released").write_text(json.dumps(released))
    return d / "timeline.jsonl"


def tp_commit(w0: float, w1: float | None) -> list[tuple]:
    return [("stage.commit", "start", w0), ("commit.tp_commit", "start", w0 + 3),
            *([("commit.tp_commit", "end", w1), ("stage.commit", "end", w1)] if w1 is not None else [])]


def test_a_tp_commit_counts_its_lease_from_the_row_on_each_rank_it_released(tmp_path):
    # gm357: Commit 23:43:02 to 00:00:40, lease 23:43:29 to 23:45:54 on 2 GPUs, and nothing in the timeline but commit.tp_commit
    w0, a, b, w1 = T0 + 43 * 60 + 2, T0 + 43 * 60 + 29, T0 + 45 * 60 + 54, T0 + 3640
    p = row(tmp_path, "phi4-14b__bf16__rtxpro6000__tp2__b8", tp_commit(w0, w1), a, {"t": b, "ranks": [{"rank": 0}, {"rank": 1}]})
    leases = vc.commit_leases([p], now=w1 + 60)
    assert leases == [(a, b, [], 2)]
    assert vc.phases_by_hour(leases) == {"2026-10-02T23:00Z": {
        "leases": 1, **{s: 0.0 for s in (*vc.LEASE_PHASES, "work")}, "other": round(145 * 2 / 3600, 3)}}


def test_a_commit_with_lease_spans_is_one_gpu_and_never_read_from_the_row(tmp_path):
    p = row(tmp_path, "qwen3-8b__bf16__rtxpro6000__tp1__b8", [
        ("stage.commit", "start", T0), ("gpu_lease.wait", "start", T0 + 1), ("gpu_lease.wait", "end", T0 + 10),
        ("work.pair0", "start", T0 + 20), ("work.pair0", "end", T0 + 100), ("gpu_lease.release", "start", T0 + 110),
        ("gpu_lease.release", "end", T0 + 130), ("stage.commit", "end", T0 + 200)], T0 + 10, {"t": T0 + 130, "ranks": [{}]})
    leases = vc.commit_leases([p], now=T0 + 300)
    assert leases == [(T0 + 10, T0 + 130, [("work", T0 + 20, T0 + 100)], 1)]
    hour = vc.phases_by_hour(leases)["2026-10-02T23:00Z"]
    assert (hour["work"], hour["other"]) == (round(80 / 3600, 3), round(40 / 3600, 3))


def test_a_tp_lease_still_open_runs_to_now_on_the_rows_tp_gpus_and_an_earlier_attempts_files_do_not_count(tmp_path):
    name, now = "phi4-14b__bf16__rtxpro6000__tp2__b8", T0 + 1000
    assert vc.commit_leases([row(tmp_path / "open", name, tp_commit(T0, None), T0 + 30)], now) == [(T0 + 30, now, [], 2)]
    earlier = {"t": T0 - 500, "ranks": [{}, {}]}
    assert vc.commit_leases([row(tmp_path / "rerun", name, tp_commit(T0, None), T0 + 30, earlier)], now) == [(T0 + 30, now, [], 2)]
    assert vc.commit_leases([row(tmp_path / "stale", name, tp_commit(T0, None), T0 - 600, earlier)], now) == []
    assert vc.commit_leases([row(tmp_path / "ended", name, tp_commit(T0, T0 + 400), T0 + 30)], now) == [(T0 + 30, T0 + 400, [], 2)]
