"""Pod 85's release side (`pod/rec_inner.py release-parts`, `release-report`): V*'s statements named as 85 stages them, the
release windows its ``RELEASE_WINDOWS_S`` gives them, and a recursive session read back from the ZK server's record against
that schedule and the Lean session loop's ``FIREWALL-TIMES`` lines."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from verity_flock import firewall as FW

REC_INNER = Path(__file__).resolve().parents[4] / "verity/ml/flock/pod/rec_inner.py"
UNITS = ["rec-open/l32-h11", "rec-open/l16-h10", "rec-open/l16-h8", "rec-residuals/m29-k26-c6-p0",
         "rec-residuals/m29-k26-c6-p1", "rec-cap/n128-l1", "rec-cap/n64-l2", "rec-cap/n128"]
PARTS = ["L0", "L1", "L2", "alg-p0", "alg-p1", "cap-L1", "cap-L2", "cap"]


def _rec_inner():
    spec = importlib.util.spec_from_file_location("rec_inner", REC_INNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _rec_inner()


def test_parts_are_85s_directory_names_in_v_stars_order():
    assert R.release_parts(UNITS) == PARTS
    assert R.release_parts(["rec-open/l8-h3", "rec-link/x"]) == ["L0", "rec-link/x"]
    with pytest.raises(ValueError, match="same part twice"):
        R.release_parts(["rec-cap/n64", "rec-cap/n128"])


def test_windows_are_per_class_with_the_lead_on_the_first():
    ws = R.release_windows_ms(PARTS, "lead=100,L0=30,L=10,alg=5,cap=2")
    assert ws == [130_000, 10_000, 10_000, 5_000, 5_000, 2_000, 2_000, 2_000]
    assert R.release_windows_ms(PARTS, "L=10,alg=5,cap=2")[0] == 10_000
    with pytest.raises(ValueError, match="no window for cap"):
        R.release_windows_ms(PARTS, "L=10,alg=5")
    with pytest.raises(ValueError, match="positive"):
        R.release_windows_ms(PARTS, "L=10,alg=0,cap=2")


def _session(tmp_path, n_released: int, stop: str | None = None):
    fw = FW.Firewall(tmp_path / "fw")
    ws = R.release_windows_ms(PARTS, "L=10,alg=5,cap=2")
    sid = fw.open("circuit/rep0", PARTS, budget=1, release={"start_ms": 1_000, "windows_ms": ws})
    fw.finish(sid)
    times = []
    for j, p in enumerate(PARTS[:n_released]):
        at = 1_000 + sum(ws[:j + 1])
        fw.release(sid, p)
        times.append({"session": sid, "part": p, "ready_ms": at - 700, "release_ms": at, "outcome": "go"})
    if stop is not None:
        fw.stop(sid, stop, "late")
        times.append({"session": sid, "part": stop, "ready_ms": 99_000, "release_ms": None, "outcome": "stop"})
    return tmp_path / "fw", sid, times


def test_a_session_released_part_by_part_at_its_times_is_ok(tmp_path):
    state, sid, times = _session(tmp_path, len(PARTS))
    rep = R.release_report(state, sid, times)
    assert rep["ok"] and rep["released"] == PARTS and rep["inner_finished"] and not rep["stops"]
    assert [x["slack_ms"] for x in rep["parts"]] == [700] * len(PARTS)
    assert rep["parts"][0] == {"part": "L0", "window_ms": 10_000, "release_ms": 11_000, "ready_ms": 10_300,
                               "decided": "go", "record": ["release"], "slack_ms": 700, "loop_release_ms": 11_000}
    assert rep["schedule_ms"] == 3 * 10_000 + 2 * 5_000 + 3 * 2_000


def test_a_stop_a_missing_part_or_a_loop_time_off_schedule_is_not_ok(tmp_path):
    state, sid, times = _session(tmp_path / "a", 3, stop="alg-p0")
    rep = R.release_report(state, sid, times)
    assert not rep["ok"] and rep["stops"][0]["part"] == "alg-p0" and rep["parts"][3]["decided"] == "stop"
    state, sid, times = _session(tmp_path / "b", len(PARTS) - 1)
    assert not R.release_report(state, sid, times)["ok"]
    state, sid, times = _session(tmp_path / "c", len(PARTS))
    times[2] = {**times[2], "release_ms": times[2]["release_ms"] + 1}
    assert not R.release_report(state, sid, times)["ok"]


def test_the_cli_reads_firewall_times_lines_and_exits_on_the_verdict(tmp_path, capsys):
    state, sid, times = _session(tmp_path, len(PARTS))
    err = tmp_path / "prove.err"
    err.write_text("noise\n" + "".join(f"FIREWALL-TIMES {json.dumps(t)}\n" for t in times))
    assert R.main(["release-report", "--state", str(state), "--session", sid, "--times", str(err)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert R.main(["release-report", "--state", str(state), "--session", sid]) == 1
    st = tmp_path / "rec-statements.txt"
    st.write_text("RECVSTAR " + json.dumps({"ok": True, "statements": [[u, 1] for u in UNITS]}) + "\n")
    capsys.readouterr()
    assert R.main(["release-parts", "--statements", str(st), "--windows-s", "L=10,alg=5,cap=2"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["parts"] == PARTS and got["units"] == UNITS and len(got["windows_ms"]) == len(PARTS)
