"""The recorder's trace writer on CPU: the open-loop schedule, chunk stamping on an injected clock, the conversion to the
calibration tool's format with its exclusions, and the merge. No GPU, no server, no clock."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from verity_network_warden import calibration

HERE = Path(__file__).resolve().parents[1]


def _record():
    spec = importlib.util.spec_from_file_location("network_traces_record", HERE / "record.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rec = _record()
WINDOW_MS = rec.WINDOW_BUCKETS * rec.BUCKET_MS


def request(rid: str, send_ms: float, chunks: list, *, prompt_len: int = 100, ok: bool = True) -> dict:
    return {"rid": rid, "send_ms": send_ms, "chunks": chunks, "prompt_len": prompt_len, "request_bytes": 9000,
            "ok": ok, "max_tokens": sum(n for _, n in chunks)}


def samples(windows: int, **at: dict) -> list[dict]:
    """A sample every SAMPLE_S from t0 through the tail, quiet unless `at` overrides the sample at that t_ms."""
    step = int(1000 * rec.SAMPLE_S)
    out = []
    for t in range(step, windows * WINDOW_MS + rec.TAIL_MS + step, step):
        s = {"t_ms": float(t), "dt_s": rec.SAMPLE_S, "hb_max_ms": 1.0, "loop_max_ms": 2.0, "foreign_pids": [],
             "others_busy": 0.05, "load": [1.0, 1.0, 1.0], "own_cores": 2.0, "sm_mhz": 2400}
        out.append(s | at.get(f"t{t}", {}))
    return out


def test_the_schedule_is_open_loop_seeded_and_never_shares_a_prefix() -> None:
    wl = [{"index": 0, "offset_s": 0.0, "prompt": [7, 8, 9], "max_tokens": 4},
          {"index": 1, "offset_s": 0.5, "prompt": [7, 8, 9, 10], "max_tokens": 2}]
    a = rec.schedule([("b4", wl, 2.0)], -2.0, 6.0, seed=0)
    assert a == rec.schedule([("b4", wl, 2.0)], -2.0, 6.0, seed=0)
    assert a != rec.schedule([("b4", wl, 2.0)], -2.0, 6.0, seed=1)
    assert [q["instance"] for q in a if q["index"] == 0] == [0, 1, 2, 3]
    for q in a:
        k = q["instance"]
        base = q["at_s"] - wl[q["index"]]["offset_s"]
        assert -2.0 + 2 * k <= base < -2.0 + 2 * k + 1.0
    assert [q["at_s"] for q in a] == sorted(q["at_s"] for q in a)
    assert len({tuple(q["prompt"]) for q in a}) == len(a)
    assert all(q["prompt"][1:] == wl[q["index"]]["prompt"][1:] for q in a)


def test_a_stream_stamps_each_chunk_with_its_tokens_on_the_injected_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([10.25, 10.5, 10.75, 11.0])
    monkeypatch.setattr(rec, "clock", lambda: next(times))
    s = rec.Stream(t0=10.0)
    for line in [b": keep-alive\n",
                 b'data: {"choices": [{"token_ids": [5], "prompt_token_ids": [1, 2]}]}\n', b"\n",
                 b'data: {"choices": [{"token_ids": [6, 7]}]}\n',
                 b'data: {"choices": [{"token_ids": [], "finish_reason": "length"}]}\n',
                 b"data: [DONE]\n"]:
        s.line(line)
    assert s.chunks == [[250.0, 1], [500.0, 2]] and s.done


def test_a_chunk_becomes_frames_in_canonical_order_and_ingress_frames() -> None:
    records = [request("a", 50.0, [[120.0, 1], [130.5, 2]], prompt_len=1000),
               request("b", 99.9, [[125.0, 1], [210.0, 1]]),
               request("c", WINDOW_MS + 50.0, [[WINDOW_MS + 210.0, 1], [WINDOW_MS + 300.0, 1]])]
    configs, summary = rec.build_trace(records, samples(2), name="t", windows=2)
    assert summary["segments"] == [(0, 2)] and summary["windows_kept"] == 2 and not summary["frozen"]
    (cfg,) = configs
    frames = [(f["session"], f["index"], f["anchor"], f["ready"]) for f in cfg["egress"][0]["frames"]]
    assert frames == [("a", 0, 0, 2), ("b", 0, 0, 2), ("a", 1, 0, 2), ("a", 2, 0, 2), ("b", 1, 0, 3),
                      ("c", 0, 600, 603), ("c", 1, 600, 604)]
    assert cfg["ingress"][0]["arrival_ms"] == [50.0, 50.0, 99.9, WINDOW_MS + 50.0]
    (body,), _ = rec.build_trace(records, samples(2), name="t", windows=2, ingress="body")
    assert body["ingress"][0]["arrival_ms"] == [50.0] * 3 + [99.9] * 3 + [WINDOW_MS + 50.0] * 3
    report = calibration.calibrate(rec.trace_doc(configs))
    assert (report["egress"][0]["honest_windows"], report["egress"][0]["honest_misses"]) == (2, 0)


def test_contamination_and_failures_drop_windows_and_split_the_chunk() -> None:
    records = [request("a", 50.0, [[120.0, 1]], ok=False), request("b", WINDOW_MS + 50.0, [[WINDOW_MS + 150.0, 1]]),
               request("c", 2 * WINDOW_MS + 50.0, [[2 * WINDOW_MS + 150.0, 1]]),
               request("d", 3 * WINDOW_MS + 50.0, [[3 * WINDOW_MS + 150.0, 1]])]
    quiet_gpu = {f"t{2 * WINDOW_MS + 2000}": {"foreign_pids": [4242]}}
    configs, summary = rec.build_trace(records, samples(4, **quiet_gpu), name="t", windows=4)
    assert summary["windows_dropped"] == {"0": "request failed", "2": "foreign GPU process"}
    assert summary["segments"] == [(1, 2), (3, 4)] and summary["foreign_pids"] == [4242]
    assert [c["name"] for c in configs] == ["t/w1-2", "t/w3-4"]
    assert [f["anchor"] for c in configs for f in c["egress"][0]["frames"]] == [0, 0]
    busy = {f"t{WINDOW_MS}": {"others_busy": 0.9}}
    _, summary = rec.build_trace(records[1:], samples(4, **busy), name="t", windows=4)
    assert summary["windows_dropped"] == {"0": "NUMA node busy", "1": "NUMA node busy"}


@pytest.mark.parametrize("stall", [{"hb_max_ms": 1500.0}, {"loop_max_ms": 1500.0}, {"dt_s": 5.0}])
def test_a_stall_or_a_gap_in_the_samples_voids_the_chunk(stall: dict) -> None:
    records = [request("a", 50.0, [[120.0, 1]])]
    configs, summary = rec.build_trace(records, samples(1, t30000=stall), name="t", windows=1)
    assert configs == [] and summary["frozen"]


def test_a_stopped_chunk_keeps_the_whole_windows_before_the_stop() -> None:
    records = [request("a", 50.0, [[120.0, 1]]), request("b", WINDOW_MS + 50.0, [[WINDOW_MS + 150.0, 1]], ok=False)]
    configs, summary = rec.build_trace(records, samples(3), name="t", windows=3, stopped_ms=WINDOW_MS + 30_000.0)
    assert summary["windows_planned"] == 1 and summary["segments"] == [(0, 1)] and summary["requests"] == 1


def test_lines_cut_by_a_kill_are_dropped_and_merge_checks_the_grid(tmp_path: Path) -> None:
    log = rec.JsonLines(tmp_path / "requests.jsonl")
    log({"rid": "a"})
    log.close()
    with open(tmp_path / "requests.jsonl", "a") as fh:
        fh.write('{"rid": "b", "chu')
    assert rec.read_lines(tmp_path / "requests.jsonl") == [{"rid": "a"}]
    one = rec.trace_doc([{"name": "x", "egress": [], "ingress": []}])
    (tmp_path / "trace.json").write_text(json.dumps(one))
    (tmp_path / "other.json").write_text(json.dumps(one | {"bucket_ms": 50}))
    assert rec.merge([tmp_path, tmp_path / "trace.json"])["configurations"] == one["configurations"] * 2
    with pytest.raises(ValueError, match="grid"):
        rec.merge([tmp_path / "other.json"])
