"""`cluster.usage`: per node, owner, kind and UTC hour, the GPU-seconds held, busy, idle, unmeasured and useful, from per-minute
readings and pool leases in the console's shapes (node 1's DCGM minutes and n1_lease.py spans, node 2's sampler minutes, 3 Oct
2026); the held-idle lines derived from them; the record; and `cluster usage hourly | wasters`."""
from __future__ import annotations

import argparse
import json

import pytest

from cluster import cli, usage
from cluster.ledger import epoch

H0 = epoch("2026-10-03T08:00:00Z")
LANES = {"gpu-pool-circuits-a": "circuits", "Kueue vllm-epoch-run": "circuits", "fill bc-15ada664": "memory-accounting"}


def attribute(holder: str) -> tuple[str, str]:
    kind = "gpu-lease pool" if holder.startswith("gpu-pool-") else holder.split(" ")[0] if " bc-" in holder else holder
    return LANES.get(holder, holder), kind


def by(rows: list[dict]) -> dict[tuple, dict]:
    return {(r["hour"], r["owner"], r["kind"]): r for r in rows}


def secs(r: dict) -> tuple:
    return tuple(round(r[k], 3) for k in ("held_s", "busy_s", "idle_s", "unmeasured_s", "useful_s"))


def test_a_lease_holds_its_gpu_for_its_span_busy_by_the_gpus_reading_and_the_rest_of_its_edge_minutes_counts_by_the_reading():
    pool = "gpu-pool-circuits-a"
    minutes = [(H0, 0, 1.0, 0, 0, 0, pool), (H0 + 60, 0, 0, 1.0, 0, 0, pool), (H0 + 120, 0, 0, 1.0, 0, 0, pool)]
    leases = [usage.Lease(0, H0 + 30, H0 + 150, "circuits", "gpu-lease"), usage.Lease(1, H0, H0 + 60, "proofs", "gpu-lease")]
    rows = by(usage.hourly("vy-nebius-1", minutes, leases, attribute=attribute, pooled=lambda h: h.startswith("gpu-pool-")))
    h = "2026-10-03T08:00Z"
    # 30 s of the busy minute, the idle minute, 30 s of the last: the lease's; the edges' other halves: the pool's, unleased
    assert secs(rows[(h, "circuits", "gpu-lease")]) == (120, 30, 90, 0, 30)
    assert secs(rows[(h, "circuits", usage.POOL_UNLEASED)]) == (60, 30, 30, 0, 0)
    # GPU 1's lease has no reading: held, unmeasured, never idle
    assert secs(rows[(h, "proofs", "gpu-lease")]) == (60, 0, 0, 60, 0)
    assert set(rows) == {(h, "circuits", "gpu-lease"), (h, "circuits", usage.POOL_UNLEASED), (h, "proofs", "gpu-lease")}


def test_leases_that_overlap_on_a_gpu_share_its_minute_and_never_hold_it_longer():
    # the two pools can't hold one GPU at once, but nothing upstream checks their spans are disjoint (red team M)
    minutes = [(H0, 0, 1.0, 0, 0, 0, "gpu-pool-circuits-a")]
    leases = [usage.Lease(0, H0, H0 + 60, "circuits", "gpu-lease"), usage.Lease(0, H0 + 30, H0 + 60, "proofs", "gpu-lease")]
    rows = by(usage.hourly("vy-nebius-1", minutes, leases, attribute=attribute, pooled=lambda h: h.startswith("gpu-pool-")))
    h = "2026-10-03T08:00Z"
    assert secs(rows[(h, "circuits", "gpu-lease")]) == (40, 40, 0, 0, 40)
    assert secs(rows[(h, "proofs", "gpu-lease")]) == (20, 20, 0, 0, 20)
    assert set(rows) == {(h, "circuits", "gpu-lease"), (h, "proofs", "gpu-lease")}


def test_without_leases_a_pool_pod_is_an_ordinary_holder_and_time_nobody_holds_has_no_owner():
    minutes = [(H0, 0, 0, 1.0, 0, 0, "gpu-pool-circuits-a"), (H0, 1, 0, 0, 1.0, 0, None), (H0, 2, 0, 0, 0, 1.0, None)]
    rows = by(usage.hourly("vy-nebius-1", minutes, attribute=attribute, pooled=lambda h: True))
    h = "2026-10-03T08:00Z"
    assert secs(rows[(h, "circuits", "gpu-lease pool")]) == (60, 0, 60, 0, 0)
    assert secs(rows[(h, None, None)]) == (0, 60, 60, 0, 0)


def test_a_timed_minute_is_held_and_unmeasured_and_a_part_held_minute_splits_by_its_samples():
    # node 2: circuits-tp8's timed lease (no NVML, all 6 samples unmeasured); a fill lease busy 2 of 6 samples, free for the rest
    minutes = [(H0, 0, 0, 0, 0, 0, "circuits-tp8", 1.0), (H0, 1, 2 / 6, 0, 0, 4 / 6, "fill bc-15ada664", 0.0)]
    rows = by(usage.hourly("vy-nebius-2", minutes, attribute=attribute))
    h = "2026-10-03T08:00Z"
    assert secs(rows[(h, "circuits-tp8", "circuits-tp8")]) == (60, 0, 0, 60, 0)
    assert secs(rows[(h, "memory-accounting", "fill")]) == (20, 20, 0, 0, 20)
    assert secs(rows[(h, None, None)]) == (0, 0, 40, 0, 0)


def test_busy_time_on_a_gpu_only_memory_or_a_process_holds_is_never_useful():
    minutes = [(H0, 0, 1.0, 0, 0, 0, usage.NO_POD), (H0, 1, 1.0, 0, 0, 0, usage.NO_LEASE), (H0, 2, 1.0, 0, 0, 0, "Kueue vllm-epoch-run")]
    rows = by(usage.hourly("vy-nebius-1", minutes, attribute=attribute))
    h = "2026-10-03T08:00Z"
    assert [rows[(h, o, o)]["useful_s"] for o in (usage.NO_POD, usage.NO_LEASE)] == [0, 0]
    assert rows[(h, "circuits", "Kueue vllm-epoch-run")]["useful_s"] == 60


def test_rows_split_at_the_hour_and_come_held_first_by_owner_then_the_free_time():
    minutes = [(H0 - 60, 0, 1.0, 0, 0, 0, "Kueue vllm-epoch-run"), (H0, 0, 0, 1.0, 0, 0, "Kueue vllm-epoch-run"), (H0, 1, 0, 0, 0, 1.0, None)]
    leases = [usage.Lease(2, H0 - 30, H0 + 30, "proofs", "gpu-lease")]
    rows = usage.hourly("vy-nebius-1", minutes, leases, attribute=attribute)
    assert [(r["hour"], r["owner"], r["kind"], r["held_s"]) for r in rows] == [
        ("2026-10-03T07:00Z", "circuits", "Kueue vllm-epoch-run", 60), ("2026-10-03T07:00Z", "proofs", "gpu-lease", 30),
        ("2026-10-03T08:00Z", "circuits", "Kueue vllm-epoch-run", 60), ("2026-10-03T08:00Z", "proofs", "gpu-lease", 30),
        ("2026-10-03T08:00Z", None, None, 0)]
    assert all(r["schema"] == usage.ROW and r["node"] == "vy-nebius-1" for r in rows)
    # 07Z was read for one minute, 08Z for one (two GPUs); GPU 2's lease reads nothing
    assert {(r["hour"], r["minutes_read"]) for r in rows} == {("2026-10-03T07:00Z", 1), ("2026-10-03T08:00Z", 1)}


def row(h: str, node: str, owner: str | None, kind: str | None, busy: float, idle: float, um: float = 0, useful: float | None = None,
        read: int = 60):
    held = owner is not None
    return {"schema": usage.ROW, "hour": h, "node": node, "owner": owner, "kind": kind, "held_s": busy + idle + um if held else 0,
            "busy_s": busy, "idle_s": idle, "unmeasured_s": um, "useful_s": (busy if held else 0) if useful is None else useful,
            "minutes_read": read}


NAMES = {"vy-nebius-1": "node 1", "vy-nebius-2": "node 2"}


def test_held_idle_lines_keep_the_files_format_lanes_targets_and_the_change_from_the_lanes_hour_before():
    rows = [row("2026-10-03T07:00Z", "vy-nebius-1", "circuits", "gpu-lease", 1800, 1800),
            row("2026-10-03T08:00Z", "vy-nebius-1", "circuits", "gpu-lease", 2700, 900),
            row("2026-10-03T08:00Z", "vy-nebius-1", "circuits", usage.POOL_UNLEASED, 0, 720),
            row("2026-10-03T08:00Z", "vy-nebius-1", None, None, 3600, 3600),
            row("2026-10-03T10:00Z", "vy-nebius-1", "circuits", "gpu-lease", 0, 3600),
            row("2026-10-03T10:00Z", "vy-nebius-2", "circuits-tp8", "gpu-lease", 0, 0, 28800)]
    lines = usage.held_idle(rows, now=epoch("2026-10-03T10:30:00Z"), targets={("node 1", "circuits"): 40.0}, names=NAMES)
    assert list(lines[0]) == ["hour", "node", "lane", "complete", "held_gpu_h", "held_idle_gpu_h", "held_idle_pct",
                              "unmeasured_gpu_h", "target_pct", "change_pts"]
    got = {(d["hour"][11:13], d["node"], d["lane"]): (d["complete"], d["held_gpu_h"], d["held_idle_gpu_h"], d["held_idle_pct"],
                                                      d["unmeasured_gpu_h"], d["target_pct"], d["change_pts"]) for d in lines}
    assert got == {
        ("07", "node 1", usage.ALL): (True, 1.0, 0.5, 50.0, 0.0, None, None),
        ("07", "node 1", "circuits"): (True, 1.0, 0.5, 50.0, 0.0, 40.0, None),
        ("08", "node 1", usage.ALL): (True, 1.2, 0.45, 37.5, 0.0, None, -12.5),
        ("08", "node 1", "circuits"): (True, 1.0, 0.25, 25.0, 0.0, 40.0, -25.0),
        ("08", "node 1", usage.POOL_UNLEASED): (True, 0.2, 0.2, 100.0, 0.0, None, None),
        # 09Z has no line, so 10Z has no change; a lane wholly unmeasured has no share
        ("10", "node 1", usage.ALL): (False, 1.0, 1.0, 100.0, 0.0, None, None),
        ("10", "node 1", "circuits"): (False, 1.0, 1.0, 100.0, 0.0, 40.0, None),
        ("10", "node 2", usage.ALL): (False, 0.0, 0.0, None, 8.0, None, None),
        ("10", "node 2", "circuits-tp8"): (False, 0.0, 0.0, None, 8.0, None, None)}
    assert [(d["hour"], d["node"], d["lane"]) for d in lines] == sorted((d["hour"], d["node"], d["lane"]) for d in lines)


def test_record_writes_each_complete_hour_the_window_covered_whole_once_per_key(tmp_path):
    p = tmp_path / "usage-hourly.jsonl"
    now, since = epoch("2026-10-03T10:30:00Z"), epoch("2026-10-03T07:20:00Z")
    rows = [row(h, n, "circuits", "gpu-lease", 60, 60) for h in ("2026-10-03T07:00Z", "2026-10-03T08:00Z", "2026-10-03T09:00Z",
                                                                  "2026-10-03T10:00Z") for n in ("vy-nebius-1", "vy-nebius-2")]
    assert usage.record(rows, p, ("hour", "node"), now, since) == 4      # 07Z began before the window, 10Z hasn't ended
    assert {(d["hour"], d["node"]) for d in map(json.loads, p.read_text().splitlines())} == {
        (h, n) for h in ("2026-10-03T08:00Z", "2026-10-03T09:00Z") for n in ("vy-nebius-1", "vy-nebius-2")}
    late = [row("2026-10-03T09:00Z", "vy-nebius-1", "proofs", "gpu-lease", 1, 1), row("2026-10-03T10:00Z", "vy-nebius-1", "x", "y", 1, 1)]
    assert usage.record(late, p, ("hour", "node"), epoch("2026-10-03T11:02:00Z"), since) == 0   # 10Z's last scrape may be out
    assert usage.record(late, p, ("hour", "node"), epoch("2026-10-03T11:05:00Z"), since) == 1   # 09Z was written whole; 10Z now is
    lines = [{**row("2026-10-03T09:00Z", "vy-nebius-1", "circuits", "gpu-lease", 1, 1), "lane": "circuits", "complete": True}]
    q = tmp_path / "held-idle-hourly.jsonl"
    assert usage.record(lines, q, ("hour", "node", "lane"), now, since) == 1
    assert usage.record(lines, q, ("hour", "node", "lane"), now, since) == 0
    assert "complete" not in json.loads(q.read_text())


def test_record_reads_only_the_files_tail_for_the_keys_it_has(tmp_path, monkeypatch):
    p = tmp_path / "r.jsonl"
    old = row("2026-10-03T08:00Z", "vy-nebius-1", "a", "b", 1, 1)
    p.write_text("".join(json.dumps(old) + "\n" for _ in range(50)))
    monkeypatch.setattr(usage, "TAIL_BYTES", len(json.dumps(old)) * 3)    # cuts into a line: its first is partial, and skipped
    assert usage.record([old], p, ("hour", "node"), epoch("2026-10-03T10:00:00Z")) == 0


def test_node_hours_and_wasters():
    rows = [row("2026-10-03T08:00Z", "vy-nebius-1", "circuits", "gpu-lease", 2700, 900),
            row("2026-10-03T08:00Z", "vy-nebius-1", "proofs", "gpu-lease", 3000, 600, 600),
            row("2026-10-03T08:00Z", "vy-nebius-1", "circuits", usage.NO_POD, 360, 0, useful=0),
            row("2026-10-03T08:00Z", "vy-nebius-1", None, None, 0, 7200),
            row("2026-10-03T09:00Z", "vy-nebius-1", "circuits", "gpu-lease", 0, 1800)]
    hours = usage.node_hours(rows)
    assert [h["hour"] for h in hours] == ["2026-10-03T09:00Z", "2026-10-03T08:00Z"]
    h = hours[1]
    assert (h["gpu_s"], h["held_s"], h["busy_s"], h["idle_s"], h["unmeasured_s"], h["useful_s"], h["held_idle_s"]) == (
        15360, 8160, 6060, 8700, 600, 5700, 1500)
    assert (h["busy_pct"], h["useful_pct"]) == (round(100 * 6060 / 14760, 1), round(100 * 5700 / 6060, 1))
    assert [(x["owner"], x["kind"]) for x in h["holdings"]] == [("circuits", "gpu-lease"), ("proofs", "gpu-lease"),
                                                               ("circuits", usage.NO_POD)]
    w = usage.wasters(rows, top=2)
    assert [(x["owner"], x["kind"], x["idle_s"], x["idle_pct"]) for x in w] == [
        ("circuits", "gpu-lease", 2700, round(100 * 2700 / 5400, 1)), ("proofs", "gpu-lease", 600, round(100 * 600 / 3600, 1))]


def record_file(tmp_path, rows) -> str:
    p = tmp_path / "usage-hourly.jsonl"
    p.write_text("not json\n" + "".join(json.dumps(r) + "\n" for r in rows) + '{"schema": "other"}\n')
    return str(p)


def run(capsys, *argv) -> tuple[int, str]:
    rc = cli.main(["usage", *argv])
    return rc, capsys.readouterr().out


def test_hourly_reports_the_last_h_hours_per_node_and_exits_1_when_a_node_is_stale(tmp_path, capsys, monkeypatch):
    rows = [row(f"2026-10-03T{h:02d}:00Z", n, "circuits", "gpu-lease", 1800, 1800) for h in range(5, 10)
            for n in ("vy-nebius-1", "vy-nebius-2")]
    f = record_file(tmp_path, rows)
    monkeypatch.setattr(usage, "clock", lambda: epoch("2026-10-03T10:20:00Z"))
    rc, out = run(capsys, "hourly", "--hours", "3", "--file", f, "--json")
    rep = json.loads(out)
    assert rc == 0 and rep["schema"] == usage.HOURLY and rep["findings"] == []
    assert [(r["hour"][11:13], r["node"]) for r in rep["rows"]] == [
        (h, n) for h in ("09", "08", "07") for n in ("vy-nebius-2", "vy-nebius-1")]
    assert rep["rows"][0]["busy_pct"] == 50.0 and rep["targets"] == {"busy_pct": 95.0, "useful_pct": 90.0}
    rc, out = run(capsys, "hourly", "--node", "nebius-2", "--hours", "3", "--file", f)
    assert rc == 0 and "vy-nebius-1" not in out and "useful today:" in out
    monkeypatch.setattr(usage, "clock", lambda: epoch("2026-10-03T12:30:00Z"))   # 09Z ended 2.5 h ago
    rc, out = run(capsys, "hourly", "--file", f, "--json")
    findings = json.loads(out)["findings"]
    assert rc == 1 and [x.split(":")[0] for x in findings] == ["vy-nebius-1", "vy-nebius-1", "vy-nebius-2", "vy-nebius-2"]
    assert "newest recorded hour is 2026-10-03T09:00Z" in findings[0]
    assert findings[1].startswith("vy-nebius-1: 17 of the last 24 hours ended over 2 h ago and aren't recorded, from 2026-10-02T12:00Z")


def test_an_hour_read_for_under_50_minutes_has_no_shares_and_a_part_read_or_missing_hour_is_a_finding(tmp_path, capsys,
                                                                                                    monkeypatch):
    # red team N: node 2's 8 GPUs read for 15 minutes of 08Z are 100 % busy of what was read, not of the hour
    monkeypatch.setattr(usage, "clock", lambda: epoch("2026-10-03T11:20:00Z"))   # 07Z and 08Z ended over 2 h ago; 09Z not yet
    f = record_file(tmp_path, [row("2026-10-03T07:00Z", "vy-nebius-2", "memory-accounting", "fill", 8 * 3600, 0),
                               row("2026-10-03T08:00Z", "vy-nebius-2", "memory-accounting", "fill", 8 * 15 * 60, 0, read=15),
                               row("2026-10-03T10:00Z", "vy-nebius-2", "memory-accounting", "fill", 8 * 3600, 0),
                               *(row(f"2026-10-03T{h}:00Z", "vy-nebius-1", "proofs", "gpu-lease", 8 * 3600, 0) for h in ("08", "09", "10"))])
    rc, out = run(capsys, "hourly", "--hours", "4", "--file", f, "--json")
    rep = json.loads(out)
    n2 = next(r for r in rep["rows"] if r["node"] == "vy-nebius-2" and r["hour"] == "2026-10-03T08:00Z")
    assert (n2["gpu_s"], n2["minutes_read"], n2["busy_pct"], n2["useful_pct"]) == (7200.0, 15, None, None)
    assert rc == 1 and rep["findings"] == [
        "vy-nebius-1: 1 of the last 4 hours ended over 2 h ago and aren't recorded, from 2026-10-03T07:00Z",
        "vy-nebius-2 2026-10-03T08:00Z: 15 of its 60 minutes read, so it has no busy or useful share"]
    rc, out = run(capsys, "hourly", "--hours", "4", "--file", f)
    assert rc == 1 and "MIN READ" in out and any(ln.startswith("2026-10-03T08:00Z  vy-nebius-2  15 ") for ln in out.splitlines())


def day_of_free_time(day: str = "2026-10-03") -> list[dict]:
    return [row(f"{day}T{h:02d}:00Z", n, None, None, 0, 8 * 3600) for h in range(24) for n in ("vy-nebius-1", "vy-nebius-2")]


def test_wasters_ranks_one_days_holdings_and_finds_a_day_with_no_hours(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(usage, "clock", lambda: epoch("2026-10-04T00:10:00Z"))
    f = record_file(tmp_path, [*day_of_free_time(), row("2026-10-03T08:00Z", "vy-nebius-2", "memory-accounting", "fill", 100, 900),
                               row("2026-10-03T09:00Z", "vy-nebius-1", "proofs", "gpu-lease", 100, 300),
                               row("2026-10-02T23:00Z", "vy-nebius-1", "circuits", "gpu-lease", 0, 9000)])
    rc, out = run(capsys, "wasters", "--day", "2026-10-03", "--top", "1", "--file", f, "--json")
    rep = json.loads(out)
    assert rc == 0 and rep["schema"] == usage.WASTERS and rep["hours_covered"] == {"vy-nebius-1": 24, "vy-nebius-2": 24}
    assert [(w["node"], w["owner"], w["idle_s"]) for w in rep["wasters"]] == [("vy-nebius-2", "memory-accounting", 900)]
    rc, out = run(capsys, "wasters", "--day", "2026-10-01", "--file", f)
    assert rc == 1 and "finding: no hours of 2026-10-01" in out


def test_wasters_say_the_hours_they_cover_and_a_day_short_of_the_hours_that_ended_is_a_finding(tmp_path, capsys, monkeypatch):
    # red team P: a record that stops at 01Z ranked one hour as the day's wasters, exit 0
    monkeypatch.setattr(usage, "clock", lambda: epoch("2026-10-04T00:10:00Z"))     # 22Z and 23Z ended under 2 h ago
    f = record_file(tmp_path, [row("2026-10-03T00:00Z", "vy-nebius-1", "proofs", "gpu-lease", 100, 300)])
    rc, out = run(capsys, "wasters", "--day", "2026-10-03", "--file", f, "--json")
    rep = json.loads(out)
    assert rc == 1 and rep["hours_covered"] == {"vy-nebius-1": 1, "vy-nebius-2": 0}
    assert rep["findings"] == [
        "vy-nebius-1: 21 of 2026-10-03's hours ended over 2 h ago and aren't recorded, from 2026-10-03T01:00Z: the wasters "
        "cover 1 of its 24 hours",
        "vy-nebius-2: 22 of 2026-10-03's hours ended over 2 h ago and aren't recorded, from 2026-10-03T00:00Z: the wasters "
        "cover 0 of its 24 hours"]
    f = record_file(tmp_path, [r for r in day_of_free_time() if r["hour"] < "2026-10-03T22"])
    rc, out = run(capsys, "wasters", "--day", "2026-10-03", "--file", f)
    assert rc == 0 and "hours covered: vy-nebius-1 22 of 24, vy-nebius-2 22 of 24" in out


def test_bad_arguments_exit_2_and_an_unreadable_record_exits_1(tmp_path, capsys):
    for argv in (["wasters", "--day", "2026-10-32"], ["hourly", "--hours", "0"], ["wasters"], ["hourly", "--node", "n3"]):
        with pytest.raises(SystemExit) as e:
            cli.main(["usage", *argv])
        assert e.value.code == 2
    capsys.readouterr()
    rc, out = run(capsys, "hourly", "--file", str(tmp_path / "missing.jsonl"), "--json")
    assert rc == 1 and json.loads(out)["findings"][0].startswith("could not read")


def test_without_file_it_reads_vy_nebius_1s_record_with_one_tail_or_grep(capsys, monkeypatch):
    monkeypatch.setattr(usage, "clock", lambda: epoch("2026-10-03T10:20:00Z"))
    r = row("2026-10-03T09:00Z", "vy-nebius-1", "circuits", "gpu-lease", 1, 1)
    sent = []

    def runner(cmd: str) -> str:
        sent.append(cmd)
        return "tail of a line\n" + "".join(json.dumps(x) + "\n" for x in [*day_of_free_time(), r])

    a = argparse.Namespace(ucmd="hourly", node="nebius-1", hours=2, json=True, file=None)
    assert usage.main(a, runner=runner) == 0
    a = argparse.Namespace(ucmd="wasters", day="2026-10-03", top=3, json=True, file=None)
    assert usage.main(a, runner=runner) == 0
    assert sent == [f"tail -c {5 * 256 * 1024} {usage.RECORD}", f"grep -F '\"hour\": \"2026-10-03T' {usage.RECORD} || test $? -eq 1"]
    assert '"hour": "2026-10-03T' in json.dumps(r)        # the grep's pattern is the record's own format
    assert json.loads(capsys.readouterr().out.split("\n}\n")[0] + "\n}")["source"] == f"vy-nebius-1:{usage.RECORD}"
