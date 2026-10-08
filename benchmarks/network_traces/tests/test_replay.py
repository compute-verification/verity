"""The end-to-end replay on a tiny synthetic trace calibrated by the library: an honest run is accepted on every
link-window, raw and committed, with one sync per window shared by a configuration's egress links, and a session made
late past every sync is a `mismatch` in its own window and in the window its frames spill into."""

from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest
from verity.core.protocols.certifier import calibration

HERE = Path(__file__).resolve().parents[1]
T = 60  # buckets per window


def _replay():
    spec = importlib.util.spec_from_file_location("network_traces_replay", HERE / "replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rep = _replay()


def trace() -> dict:
    """Three windows of T buckets. Each window has three sessions of four frames, one frame per bucket, whose first
    frame is ready 1 to 3 buckets after its anchor; each request is one ingress frame, 30 ms after its anchor."""
    frames, arrival_ms = [], []
    for w in range(3):
        for j, (offset, latency) in enumerate(((0, 1), (10, 3), (20, 2))):
            anchor = w * T + offset
            frames += [{"session": f"s{w}.{j}", "index": i, "anchor": anchor, "ready": anchor + latency + i}
                       for i in range(4)]
            arrival_ms.append(anchor * 100 + 30.0)
    return {"bucket_ms": 100, "window_buckets": T, "frame_bytes": 4096, "queue_capacity": 1 << 16,
            "configurations": [{"name": "synthetic", "labels": {"run": "r0", "windows": [0, 3]},
                                "egress": [{"link": "responses", "frames": canonical(frames)}],
                                "ingress": [{"link": "requests", "arrival_ms": arrival_ms}]}]}


def canonical(frames: list[dict]) -> list[dict]:
    """The recorder's canonical order: by ready bucket, then anchor."""
    return sorted(frames, key=lambda f: (f["ready"], f["anchor"], f["session"], f["index"]))


def run(tmp_path: Path, doc: dict, report: dict, *flags: str) -> tuple[dict, list[dict]]:
    (tmp_path / "trace.json").write_text(json.dumps(doc))
    (tmp_path / "cal.json").write_text(json.dumps(report))
    out = tmp_path / "out"
    assert rep.main([str(tmp_path / "trace.json"), "--calibration", str(tmp_path / "cal.json"), "--out", str(out),
                     *flags]) == 0
    rows = [json.loads(line) for line in (out / "verdicts.jsonl").read_text().splitlines()]
    return json.loads((out / "summary.json").read_text()), rows


def test_an_honest_run_is_accepted_raw_and_committed(tmp_path: Path) -> None:
    doc = trace()
    report = calibration.calibrate(doc)
    assert report["clock_sync"]["honest_miss_rate"] == 0
    summary, rows = run(tmp_path, doc, report)
    assert [(r["direction"], r["window"], r["raw"], r["committed"]) for r in rows] == \
        [("egress", w, None, None) for w in range(3)] + [("ingress", w, None, None) for w in range(3)]
    e, i = summary["egress"], summary["ingress"]
    assert e["accepted"] == i["accepted"] == {"raw": 3, "committed": 3, "certified": 3}
    assert summary["committed_agrees"] and not summary["violations"] and summary["runs_accepted"]["raw"] == 1
    assert e["frames"] == 36 and e["unreleased_frames"] == 0 and e["calibrator_misses"] == 0
    # each window's δ is its slowest first frame, 3 buckets: the other sessions wait 1 or 2 buckets for it
    assert {r["sync"][0] for r in rows if r["direction"] == "egress"} == {3}
    assert e["latency_ms"]["shaping"]["max"] == 200 and e["latency_ms"]["queueing"]["max"] == 0
    assert e["latency_ms"]["first_frame"]["max"] == 300
    r = report["egress"][0]["allocation"]
    assert e["padding_share"] == 1 - 36 / (3 * T * r)
    assert i["frames"] == 9 and i["delivery_delay_ms"]["max"] == 70.0


def test_the_egress_links_of_a_configuration_share_one_sync_per_window(tmp_path: Path) -> None:
    """One clock per spatial unit: a second link whose first frames are 5 buckets late raises every window's δ from 3
    to 5 on both links, and both are accepted, raw and committed."""
    doc = trace()
    cfg = doc["configurations"][0]
    slow = [f | {"ready": f["anchor"] + 5 + f["index"]} for f in cfg["egress"][0]["frames"]]
    cfg["egress"].append({"link": "responses-b", "frames": canonical(slow)})
    report = calibration.calibrate(doc)
    summary, rows = run(tmp_path, doc, report)
    egress = [r for r in rows if r["direction"] == "egress"]
    assert sorted((r["link"], r["window"]) for r in egress) == [(link, w) for link in ("responses", "responses-b")
                                                                for w in range(3)]
    assert all(r["raw"] is None and r["committed"] is None for r in rows)
    assert {(r["window"], r["sync"][0], r["sync"][1]) for r in egress} == {(w, 5, egress[w]["sync"][1]) for w in range(3)}
    e = summary["egress"]
    assert e["accepted"] == {"raw": 6, "committed": 6, "certified": 6} and e["calibrator_misses"] == 0
    assert summary["committed_agrees"] and summary["runs_accepted"] == {"raw": 1, "committed": 1, "certified": 1}


def test_a_session_late_past_every_sync_is_a_mismatch(tmp_path: Path) -> None:
    """Window 1's second session (anchor 70) is 45 buckets late, ready at 118-121, past δ's largest value (15). No sync
    meets window 1, so the honest prover declares the most lenient one, δ = 15, and the session is logically due
    behind the window's third session, at bucket 98. The shaper hands its frames when ready, the last two in window 2.
    Both windows are mismatches, raw and committed: window 1 is the miss, from its bucket 38 (98), and window 2
    carries it from its first bucket."""
    honest = trace()
    report = calibration.calibrate(honest)
    assert max(report["clock_sync"]["deltas"]) == 15
    doc = copy.deepcopy(honest)
    link = doc["configurations"][0]["egress"][0]
    for f in link["frames"]:
        if f["session"] == "s1.1":
            f["ready"] += 45
    link["frames"] = canonical(link["frames"])
    summary, rows = run(tmp_path, doc, report)
    egress = [r for r in rows if r["direction"] == "egress"]
    assert [(r["raw"], r["committed"], r.get("cause")) for r in egress] == \
        [(None, None, None), ("mismatch", "mismatch", "miss"), ("mismatch", "mismatch", "carried")]
    assert egress[1]["detail"] == {"own_late_frames": 4, "carried_late_frames": 0, "first_bucket": 38}
    assert egress[2]["detail"] == {"own_late_frames": 0, "carried_late_frames": 2, "first_bucket": 0}
    assert all(r["raw"] is None and r["committed"] is None for r in rows if r["direction"] == "ingress")
    e = summary["egress"]
    assert e["accepted"] == {"raw": 1, "committed": 1, "certified": 1}
    assert e["violations_by_cause"] == {"miss": 1, "carried": 1}
    assert e["calibrator_misses"] == 1 and e["rejected_without_own_late_frame"] == 1
    assert summary["committed_agrees"] and summary["runs_accepted"] == {"raw": 0, "committed": 0, "certified": 0}


def test_the_certified_audit_gives_every_link_window_the_raw_audits_verdict(tmp_path: Path) -> None:
    """The difftest on traces: under each prover the replay offers, on the honest trace, the trace with a session late
    past every sync, and the honest trace with two requests per ingress bucket through a queue of one frame, the audit
    of network certificates gives every link-window the raw and the committed audits' verdict, acceptances and
    rejections alike."""
    honest = trace()
    report = calibration.calibrate(honest)
    late = copy.deepcopy(honest)
    link = late["configurations"][0]["egress"][0]
    for f in link["frames"]:
        if f["session"] == "s1.1":
            f["ready"] += 45
    link["frames"] = canonical(link["frames"])
    tight, tight_report = copy.deepcopy(honest) | {"queue_capacity": 1}, copy.deepcopy(report)
    tight_report["grid"]["queue_capacity"] = 1
    ingress = tight["configurations"][0]["ingress"][0]
    ingress["arrival_ms"] = sorted(ingress["arrival_ms"] + [t + 10 for t in ingress["arrival_ms"]])
    verdicts: Counter = Counter()
    for name, doc, rep_ in (("honest", honest, report), ("late", late, report), ("tight", tight, tight_report)):
        for syncs in ("hindsight", "envelope", "cover-2-1"):
            (tmp_path / name / syncs).mkdir(parents=True)
            summary, rows = run(tmp_path / name / syncs, doc, rep_, "--syncs", syncs)
            assert summary["certified_agrees"] and summary["committed_agrees"] and not summary["disagreements"]
            assert all(r["raw"] == r["committed"] == r["certified"] for r in rows)
            verdicts.update(r["certified"] for r in rows)
    assert {None, "mismatch", "status:overflow"} <= set(verdicts), verdicts


def test_an_online_prover_declares_from_what_it_has_seen_and_carries_a_runs_history(tmp_path: Path) -> None:
    """Under `--syncs cover-2-1` an honest run is accepted: the first window has no history and declares the
    calibration's envelope (δ = 3), and each later one δ = 4, a bucket past what the earlier windows needed. A second
    configuration of the same run starts from the first one's history unless `--fresh`."""
    doc = trace()
    report = calibration.calibrate(doc)
    second = copy.deepcopy(doc["configurations"][0]) | {"name": "synthetic-2"}
    second["labels"] = {"run": "r0", "windows": [4, 7]}
    doc["configurations"].append(second)
    for flags, first in (((), 4), (("--fresh",), 3)):
        summary, rows = run(tmp_path, doc, report, "--syncs", "cover-2-1", *flags)
        assert summary["sync_policy"] == "cover-2-1" and summary["fresh"] == bool(flags)
        assert summary["runs_accepted"] == {"raw": 2, "committed": 2, "certified": 2}
        assert summary["egress"]["calibrator_misses"] is None
        egress = [r for r in rows if r["direction"] == "egress"]
        assert [r["sync"][0] for r in egress] == [3, 4, 4, first, 4, 4]
        assert egress[0]["choices"] == 1 and egress[1]["choices"] > 1
        assert (egress[3]["choices"] == 1) == bool(flags)
    with pytest.raises(SystemExit):
        rep.main([str(tmp_path / "trace.json"), "--calibration", str(tmp_path / "cal.json"), "--out",
                  str(tmp_path / "bad"), "--syncs", "cover-2"])
