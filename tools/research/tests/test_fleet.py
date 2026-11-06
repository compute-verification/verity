"""pods/nebius/fleet.py (INF-60): the fleet's usage record and daily compute report, the H200s included, and the quota watch."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from research.pods.nebius import fleet as F


def test_a_probe_reads_gpus_load_cores_and_memory_and_a_cpu_vm_has_no_gpus():
    got = F.parse_probe("gpu=0,37,0,\ncores=128\nload=28.86\ndeadline=1792468800\nmem=1185,1574\n")
    assert got == {"gpu_util": [0, 37, 0], "cores": 128, "load": 28.86, "mem_used_gib": 1185, "mem_total_gib": 1574,
                   "deadline": 1792468800}
    got = F.parse_probe("gpu=\ncores=192\nload=3.5\ndeadline=\nmem=68,755\n")
    assert got["gpu_util"] == [] and got["deadline"] is None, "no /etc/research/deadline: none recorded"


def test_the_report_prices_hours_seen_running_and_counts_idle_hours_and_unread_samples(tmp_path, monkeypatch):
    """vy-nebius-1's GPUs sat ~0% busy for a week unseen (18 Oct): the report puts each VM's dollars beside its GPU busy share and
    idle hours, most expensive first; a sample that couldn't be read is never idle."""
    monkeypatch.setattr(F, "FLEET_DIR", tmp_path)
    t0 = 1_800_000_000

    def row(i, name, rate, read=True, gpu=(), load=0.0, cores=128):
        return {"t": F.utc(t0 + i * F.INTERVAL_S), "name": name, "preset": "p", "rate_usd_per_hour": rate, "read": read,
                **({"gpu_util": list(gpu), "load": load, "cores": cores, "mem_used_gib": 100, "mem_total_gib": 1000} if read else {})}

    rows = [row(i, "h200", 43.2, gpu=[0] * 8, load=1.0) for i in range(12)]                                  # 1 h, all idle
    rows += [row(i, "cpu", 6.34, gpu=[], load=64.0) for i in range(12)]                                      # 1 h, busy CPU
    rows += [row(i, "n2", 14.4, gpu=[50] + [0] * 7, load=2.0) for i in range(6)] + [row(6, "n2", 14.4, read=False)]
    rows += [row(0, "new", None, gpu=[], load=0.0)]
    (tmp_path / f"samples-{F.utc(t0)[:10].replace('-', '')}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = F.report(F.load_samples(t0, t0 + 3600), 1)
    by = {r["name"]: r for r in rep}
    assert [r["name"] for r in rep][:3] == ["h200", "n2", "cpu"]
    assert by["h200"]["usd"] == 43.2 and by["h200"]["idle_hours"] == 1.0 and by["h200"]["gpu_busy_pct"] == 0.0
    assert by["cpu"]["idle_hours"] == 0.0 and by["cpu"]["gpus"] == 0 and by["cpu"]["load_per_core"] == 0.5
    assert by["n2"]["gpu_busy_pct"] == 12.5 and by["n2"]["unread_samples"] == 1 and by["n2"]["hours"] == 0.6
    assert by["new"]["usd"] is None
    text = F.render(rep, 1, t0 + 3600)
    assert text.startswith("Compute, the 1 h to") and "$43 of it on idle hours" in text and "$? (no rate registered)" in text


def test_a_memory_bound_job_on_few_cores_is_not_idle(tmp_path, monkeypatch):
    """vy-glm53-cpu-2, 25 Oct: circuits' GLM folds ran back to back on 2-4 of 192 cores holding 150-700 GB of 755 GB, and the
    daily post called it idle 22.3 h of 24. Memory in use at IDLE_MEM_SHARE or more counts as use."""
    monkeypatch.setattr(F, "FLEET_DIR", tmp_path)
    t0 = 1_800_000_000

    def row(i, name, used):
        return {"t": F.utc(t0 + i * F.INTERVAL_S), "name": name, "preset": "p", "rate_usd_per_hour": 6.34, "read": True,
                "gpu_util": [], "load": 3.0, "cores": 192, "mem_used_gib": used, "mem_total_gib": 755}

    rows = [row(i, "glm53-cpu-2", 292) for i in range(12)] + [row(i, "spare", 20) for i in range(12)]
    (tmp_path / f"samples-{F.utc(t0)[:10].replace('-', '')}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    by = {r["name"]: r for r in F.report(F.load_samples(t0, t0 + 3600), 1)}
    assert by["glm53-cpu-2"]["idle_hours"] == 0.0 and by["spare"]["idle_hours"] == 1.0


def test_the_daily_post_rings_each_owning_lane_with_only_its_own_idle_machines(tmp_path, monkeypatch):
    """The lane whose budget line pays for a node stops it (the lane contract, §3c): besides the whole
    report, each lane that owns an idle machine gets one doorbell with only its machines' idle hours and idle $. A machine with no
    `lane`, or no registry entry, is infra's; a lane whose machines were never idle gets nothing."""
    monkeypatch.setattr(F, "FLEET_DIR", tmp_path / "fleet")
    monkeypatch.setattr(F, "MACHINES_D", tmp_path / "machines.d")
    F.MACHINES_D.mkdir()
    for name, lane in (("vy-glm53-eu", "circuits"), ("vy-glm53-cpu-2", "circuits"), ("vy-nebius-cpu-3", "proofs"),
                       ("vy-dc-h200", "kernels"), ("vy-nebius-1", None)):
        (F.MACHINES_D / f"{name}.toml").write_text('provider = "ssh"\n' + (f'lane = "{lane}"\n' if lane else ""))
    t0 = 1_800_000_000
    now = t0 + 86400

    def row(i, name, rate, gpu, load):
        return {"t": F.utc(t0 + i * F.INTERVAL_S), "name": name, "preset": "p", "rate_usd_per_hour": rate, "read": True,
                "gpu_util": gpu, "load": load, "cores": 128, "mem_used_gib": 1, "mem_total_gib": 10}

    rows = [row(i, "vy-glm53-eu", 43.2, [0] * 8, 0.5) for i in range(24)]                   # 2 h idle
    rows += [row(i, "vy-glm53-cpu-2", 6.34, [], 1.0) for i in range(12)]                     # 1 h idle
    rows += [row(i, "vy-nebius-cpu-3", 4.23, [], 100.0) for i in range(12)]                  # busy: proofs gets nothing
    rows += [row(i, "vy-dc-h200", None, [0], 0.0) for i in range(6)]                         # 0.5 h idle, no rate
    rows += [row(i, "vy-nebius-1", 14.4, [0] * 8, 0.0) for i in range(12)]                   # no lane: infra
    rows += [row(i, "vy-unregistered", 1.0, [], 0.0) for i in range(12)]                     # not registered: infra
    F.FLEET_DIR.mkdir()
    (F.FLEET_DIR / f"samples-{F.utc(t0)[:10].replace('-', '')}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    sent = []
    monkeypatch.setattr(F, "post", lambda args, text: sent.append((args, text)))
    F.daily(now)
    assert [a for a, _ in sent[:2]] == [["--thread", F.STATUS_THREAD, "--kind", "fyi"], ["--to", "@top", "--kind", "fyi"]]
    rings = {a[1]: text for a, text in sent[2:]}
    assert all(a[0] == ("--thread" if a[1] == F.STATUS_THREAD else "--to") and a[2:] == ["--kind", "fyi"] for a, _ in sent[2:])
    assert sorted(rings) == sorted(["@circuits", "@kernels", F.STATUS_THREAD]), "posts go out as infra, so its own ring is its thread"
    assert rings["@circuits"].startswith("Your machines sat idle 3.0 h ($93) in the 24 h to")
    assert "vy-glm53-eu 2.0 h of 2.0 h ($86)" in rings["@circuits"] and "vy-glm53-cpu-2 1.0 h of 1.0 h ($6)" in rings["@circuits"]
    assert "vy-nebius-1" not in rings["@circuits"] and "vy-dc-h200" not in rings["@circuits"]
    assert "plus machines with no rate registered" in rings["@kernels"] and "vy-dc-h200 0.5 h of 0.5 h ($?)" in rings["@kernels"]
    assert "vy-nebius-1 1.0 h" in rings[F.STATUS_THREAD] and "vy-unregistered 1.0 h" in rings[F.STATUS_THREAD]
    assert not any("@" in text for _, text in sent[2:]), "a body naming a handle past its start is refused by msg send"
    assert sent[0][1].startswith("Compute, the 24 h to") and "Idlest: vy-glm53-eu 2.0 h ($86), vy-nebius-1 1.0 h ($14)" in sent[0][1]
    assert all(len(text.split()) <= 60 for _, text in sent), "research msg refuses a post in #agent-coordination past 120 words"


def test_a_daily_post_that_fails_is_alerted_on_and_a_long_list_stays_short(tmp_path, monkeypatch):
    """research msg refuses a post over twice the 60 words one may have, and refused the whole 397-word report
    and post() sent its exit to DEVNULL. A post names at most POST_NAMES machines, and a post that fails rings #agent-alerts."""
    monkeypatch.setattr(F, "FLEET_DIR", tmp_path / "fleet")
    monkeypatch.setattr(F, "MACHINES_D", tmp_path / "machines.d")
    t0 = 1_800_000_000
    rows = [{"t": F.utc(t0 + i * F.INTERVAL_S), "name": f"vy-nebius-cpu-{n}", "preset": "p", "rate_usd_per_hour": 6.34,
             "read": True, "gpu_util": [], "load": 0.0, "cores": 192, "mem_used_gib": 1, "mem_total_gib": 755}
            for n in range(20) for i in range(12)]
    F.FLEET_DIR.mkdir()
    (F.FLEET_DIR / f"samples-{F.utc(t0)[:10].replace('-', '')}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    sent = []

    def send(args, text):
        sent.append((args, text))
        return "exit 1: research msg: 397 words" if text.startswith("Compute,") and args[0] == "--thread" else None
    assert F.daily(t0 + 86400, send) == ["--thread 1791351683.780959: exit 1: research msg: 397 words"]
    assert all(len(text.split()) <= 60 for args, text in sent if args[0] != "--channel")
    assert "and 16 more" in sent[0][1] and "and 16 more" in sent[2][1]
    alert = sent[-1]
    assert alert[0] == ["--channel", "agent-alerts", "--kind", "alert"] and "failed 1 of 3 times" in alert[1]


def test_a_sample_copies_each_vms_deadline_into_its_registry_entry_once(tmp_path, monkeypatch):
    """A stopped VM's /etc/research/deadline can't be read, so the control pod's wake reads the copy the sampler keeps in the
    registry; it is rewritten (and synced) only when the VM's file changed."""
    monkeypatch.setattr(F, "MACHINES_D", tmp_path / "machines.d")
    F.MACHINES_D.mkdir()
    (F.MACHINES_D / "vy-dc-h200.toml").write_text('provider = "ssh"\nhost = "203.0.113.7"\ndeadline = "2026-10-19T16:00:00Z"\n')
    (F.MACHINES_D / "vy-dc-b200.toml").write_text('provider = "ssh"\nhost = "203.0.113.8"\n')
    synced = []
    monkeypatch.setattr(F, "sync_registry", synced.append)
    t = int(F.datetime(2026, 10, 20, 4, tzinfo=F.timezone.utc).timestamp())
    rows = [{"name": "vy-dc-h200", "deadline": t}, {"name": "vy-dc-b200", "deadline": t}, {"name": "vy-x", "deadline": t},
            {"name": "vy-dc-b200", "deadline": None}]
    assert F.record_deadlines(rows, F.registry()) == ["vy-dc-h200", "vy-dc-b200"]
    assert synced == ["machines.d: vy-dc-h200, vy-dc-b200's deadline (fleet.py)"]
    assert F.registry()["vy-dc-h200"] == {"provider": "ssh", "host": "203.0.113.7", "deadline": "2026-10-20T04:00:00Z"}
    assert F.registry()["vy-dc-b200"]["deadline"] == "2026-10-20T04:00:00Z"
    assert F.record_deadlines(rows, F.registry()) == [] and len(synced) == 1


def test_the_card_alerts_at_80_percent_of_its_daily_limit_only_when_one_is_set(monkeypatch):
    rows = [{"rate_usd_per_hour": 120.0}] * 100                                  # 100 samples of 5 min at $120/h: $1,000
    monkeypatch.delenv("FLEET_CARD_DAILY_USD", raising=False)
    assert F.card_finding(rows) is None
    monkeypatch.setenv("FLEET_CARD_DAILY_USD", "1200")
    assert F.card_finding(rows) == {"spent": 1000.0, "cap": 1200.0}
    monkeypatch.setenv("FLEET_CARD_DAILY_USD", "2000")
    assert F.card_finding(rows) is None


def test_a_quota_at_80_percent_of_its_limit_is_a_finding(monkeypatch):
    items = [{"metadata": {"name": "compute.instance.non-gpu.vcpu"}, "spec": {"limit": "600", "region": "uk-south2"},
              "status": {"usage": "576"}},
             {"metadata": {"name": "vpc.ipv4-address.public.count"}, "spec": {"limit": "6", "region": "eu-north1"},
              "status": {"usage": "3"}},
             {"metadata": {"name": "compute.disk.count"}, "spec": {"limit": "0", "region": "eu-north2"}, "status": {}}]
    monkeypatch.setattr(F, "nb", lambda *a, **k: {"items": items})
    assert F.quota_findings() == [{"quota": "compute.instance.non-gpu.vcpu", "region": "uk-south2", "usage": 576, "limit": 600,
                                   "share": 0.96}]


def test_a_restarted_loop_doesnt_post_a_quota_alert_it_already_posted_that_day(tmp_path, monkeypatch):
    """Each supervisor restart re-posted every quota alert (four at 02:06, 02:19 and 02:23Z on 20 Oct): what was told is kept per UTC
    day in $FLEET_DIR, and a day's file older than two days goes."""
    monkeypatch.setattr(F, "FLEET_DIR", tmp_path)
    monkeypatch.setattr(F, "sample", lambda now: None)
    monkeypatch.delenv("FLEET_CARD_DAILY_USD", raising=False)
    monkeypatch.setattr(F, "quota_findings", lambda: [{"quota": "q", "region": r, "usage": 9, "limit": 10, "share": 0.9}
                                                      for r in ("uk-south2", "us-central1")])
    t0 = 1_800_000_000                                                            # 2027-01-15T08:00Z, not the report's hour
    old = tmp_path / f"quota-told-{F.utc(t0 - 3 * 86400)[:10]}.json"
    old.write_text('["q@uk-south2"]\n')
    posts: list[str] = []

    def run(now: float) -> None:
        assert F.loop(clock=lambda: now, sleep=lambda s: None, send=lambda args, text: posts.append(text), passes=1) == 0

    run(t0)
    assert len(posts) == 2 and "q in uk-south2: 9 of 10 (90%)" in posts[0] and not old.exists()
    assert json.loads((tmp_path / f"quota-told-{F.utc(t0)[:10]}.json").read_text()) == ["q@uk-south2", "q@us-central1"]
    run(t0 + 600)
    assert len(posts) == 2, "a restart the same UTC day posts nothing again"
    run(t0 + 86400)
    assert len(posts) == 4, "the next day posts each quota once more"


def test_a_running_vm_unread_ten_minutes_is_alerted_once_with_its_serial_tail_and_kept_off_auto_until_read(tmp_path, monkeypatch):
    """Nodes sat in a soft lockup, RUNNING in Nebius and answering no ssh, for hours before anyone looked."""
    monkeypatch.setattr(F, "FLEET_DIR", tmp_path / "fleet")
    monkeypatch.setattr(F, "MACHINES_D", tmp_path / "machines.d")
    F.MACHINES_D.mkdir()
    (F.MACHINES_D / "vy-node-x.toml").write_text('provider = "ssh"\nhost = "203.0.113.9"\nproject = "verity"\n')
    synced, sent, tails = [], [], []
    monkeypatch.setattr(F, "sync_registry", synced.append)

    def tail(instance_id):
        tails.append(instance_id)
        return ["[ 9.1] ok", "[8811.2] watchdog: BUG: soft lockup - CPU#77 stuck for 23s! [lean:4182]"]
    row = {"name": "vy-node-x", "id": "computeinstance-x", "read": False}
    t0 = 1_800_000_000.0
    for dt in (-1200, -900, -600):
        assert F.watch_unreachable([row], t0 + dt, lambda a, t: sent.append(t), tail) == [], "never read: not this key's to judge"
    F.watch_unreachable([{**row, "read": True}], t0 - 300, lambda a, t: sent.append(t), tail)
    for dt in (0, 300):
        assert F.watch_unreachable([row], t0 + dt, lambda a, t: sent.append(t), tail) == []
    assert F.watch_unreachable([row], t0 + 600, lambda a, t: sent.append(t), tail) == ["vy-node-x"]
    assert F.watch_unreachable([row], t0 + 900, lambda a, t: sent.append(t), tail) == [], "alerted once"
    (text,) = sent
    assert "soft lockup - CPU#77" in text and "since " + F.utc(t0) in text and "marked `unreachable_since`" in text and tails == ["computeinstance-x"]
    assert F.registry()["vy-node-x"]["unreachable_since"] == F.utc(t0)
    F.watch_unreachable([{**row, "read": True}], t0 + 1200, lambda a, t: sent.append(t), tail)
    assert "unreachable_since" not in F.registry()["vy-node-x"] and len(synced) == 2


def _sample(t, name, rate, platform="cpu-d3", preset="192vcpu-768gb", read=True, gpu=(), load=0.0, cores=192, deadline=None):
    return {"t": F.utc(t), "name": name, "platform": platform, "preset": preset, "rate_usd_per_hour": rate, "read": read,
            **({"gpu_util": list(gpu), "load": load, "cores": cores, "deadline": deadline} if read else {})}


def test_the_online_panel_names_each_running_vm_with_its_class_lane_rate_and_busy_share():
    """The console's Online tab: one row per VM the last pass saw, a GPU machine busy by its GPUs, a CPU one by load over cores
    (at most 1), an unread one with no share, and a VM with no rate left out and named in the note."""
    t = 1_800_000_000
    rows = [_sample(t, "h200", 5.4, platform="gpu-h200-sxm", preset="1gpu-16vcpu-200gb", gpu=[40], deadline=1_800_036_000),
            _sample(t, "cpu", 6.34, load=384.0), _sample(t, "gone", 6.34, read=False), _sample(t, "new", None)]
    reg = {"h200": {"lane": "network-accounting"}, "cpu": {"deadline": "2026-11-14T02:00:00Z"}}
    p = F.online_panel(rows, reg, t)
    assert p["id"] == "compute/online" and p["kind"] == "table"
    by = {r[0]: dict(zip(p["columns"], r)) for r in p["rows"]}
    assert by["h200"] == {"machine": "h200", "class": "H200", "hardware": "1× H200", "provider": "Nebius",
                               "lane": "network-accounting", "holder": None, "usd_per_hour": 5.4, "busy": 0.4, "busy_of": "gpu",
                               "stops_at": F.utc(1_800_036_000), "sampled_at": F.utc(t)}
    assert by["cpu"]["busy"] == 1.0 and by["cpu"]["busy_of"] == "cores" and by["cpu"]["lane"] == "infra"
    assert by["cpu"]["hardware"] == "192 CPUs" and by["cpu"]["stops_at"] == "2026-11-14T02:00:00Z"
    assert by["gone"]["busy"] is None and "new" not in by and "No rate registered, so left out: new." in p["note"]
    assert F.hardware({"platform": "gpu-rtx6000-a", "preset": "8gpu-192vcpu-1744gb"}) == "8× RTX PRO 6000"
    assert F.machine_class("gpu-rtx6000") == F.machine_class("gpu-rtx6000-a") == "RTX PRO 6000"


def test_hourly_and_daily_panels_price_each_sample_and_weight_busy_by_cost(tmp_path, monkeypatch):
    """compute/hourly and compute/daily: a sample costs its 5 minutes at its rate; busy dollars are each read sample's cost times
    its share, an unread sample at its slot's mean share; `minutes` counts what was sampled. A Pacific day spans two UTC days."""
    t0 = int(datetime(2026, 11, 5, 7, 0, tzinfo=UTC).timestamp())          # 11 PM Pacific, the day before
    samples = [_sample(t0 + i * F.INTERVAL_S, "cpu", 6.0, load=96.0) for i in range(11)]     # half busy
    samples.append(_sample(t0 + 11 * F.INTERVAL_S, "cpu", 6.0, read=False))
    samples += [_sample(t0 + 3600 + i * F.INTERVAL_S, "cpu", 6.0, load=0.0) for i in range(12)]  # midnight PST: a new day
    samples.append(_sample(t0, "new", None))
    hourly, daily = F.bucket_panels(samples, {"cpu": {"lane": "training"}}, t0 + 7200)
    rows = [dict(zip(hourly["columns"], r)) for r in hourly["rows"]]
    assert rows == [{"hour": "2026-11-05T07:00:00Z", "machine": "cpu", "class": "CPU", "lane": "training", "usd": 6.0,
                     "busy_usd": 3.0, "minutes": 60},
                    {"hour": "2026-11-05T08:00:00Z", "machine": "cpu", "class": "CPU", "lane": "training", "usd": 6.0,
                     "busy_usd": 0.0, "minutes": 60}]
    days = {r[0]: dict(zip(daily["columns"], r)) for r in daily["rows"]}
    assert days["2026-11-04"]["usd"] == 6.0 and days["2026-11-05"]["usd"] == 6.0 and days["2026-11-04"]["busy_usd"] == 3.0


def test_a_panel_over_the_sites_body_limit_drops_its_oldest_slots_and_says_so(monkeypatch):
    monkeypatch.setattr(F, "PANEL_MAX_BYTES", 1200)
    rows = [[f"2026-11-0{d}T00:00:00Z", f"m{i}", "CPU", "infra", 1.0, 0.5, 60] for d in range(1, 6) for i in range(5)]
    p = F.fitted(F._panel("compute/hourly", "t", ["hour", "machine", "class", "lane", "usd", "busy_usd", "minutes"], rows, 0, "n"))
    assert len(json.dumps(p).encode()) <= 1200 and p["rows"] and p["rows"][-1][0] == "2026-11-05T00:00:00Z"
    assert p["note"].startswith(f"From {p['rows'][0][0]} only") and min(r[0] for r in p["rows"]) > "2026-11-01"


def test_publish_sends_the_three_panels_with_the_key_and_nothing_without_one(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "FLEET_DIR", tmp_path)
    monkeypatch.setattr(F, "MACHINES_D", tmp_path / "machines.d")
    monkeypatch.setattr(F, "PANELS_KEY", tmp_path / "panels.key")
    t = 1_800_000_000
    rows = [_sample(t, "cpu", 6.34, load=10.0)]
    (tmp_path / f"samples-{F.utc(t)[:10].replace('-', '')}.jsonl").write_text(json.dumps(rows[0]) + "\n")
    sent = []
    assert F.publish(rows, t, put=lambda p, k: sent.append(p)) == [f"no panels key at {tmp_path / 'panels.key'}"] and not sent
    (tmp_path / "panels.key").write_text("k\n")
    assert F.publish(rows, t, put=lambda p, k: sent.append((p["id"], len(p["rows"]), k)) or None) == []
    assert sent == [("compute/online", 1, "k"), ("compute/hourly", 1, "k"), ("compute/daily", 1, "k")]
    assert F.publish(rows, t, put=lambda p, k: f"{p['id']}: HTTP 413") == [
        "compute/online: HTTP 413", "compute/hourly: HTTP 413", "compute/daily: HTTP 413"]


def test_each_days_samples_are_put_in_the_store_once_the_day_is_over(tmp_path, monkeypatch):
    """The usage record's second copy, off the control pod: the last UTC day's samples file goes in the store once, its art id
    kept so a restarted loop doesn't put it again; a failed put is retried and says why."""
    monkeypatch.setattr(F, "FLEET_DIR", tmp_path)
    now = int(datetime(2026, 11, 6, 0, 30, tzinfo=UTC).timestamp())
    assert F.archive(now, put=lambda p: (_ for _ in ()).throw(AssertionError("no file, no put"))) is None
    (tmp_path / "samples-20261105.jsonl").write_text("{}\n")
    assert F.archive(now, put=lambda p: (None, "exit 1: no remote")) == "samples-20261105.jsonl: exit 1: no remote"
    put = []
    assert F.archive(now, put=lambda p: put.append(p.name) or ("art:ab", "")) is None and put == ["samples-20261105.jsonl"]
    assert json.loads((tmp_path / "archived.json").read_text()) == {"20261105": "art:ab"}
    assert F.archive(now + 3600, put=lambda p: put.append(p.name) or ("art:cd", "")) is None and len(put) == 1


def test_a_panel_goes_only_to_an_https_site(monkeypatch):
    """The panels key is sent only over https: with $FLEET_PANELS_URL unset or not https, put_panel sends nothing and says why."""
    panel = {"id": "compute/online", "rows": []}
    monkeypatch.setattr(F.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("nothing is sent")))
    for url in (None, "http://site.invalid", "file:///etc/passwd"):
        monkeypatch.setattr(F, "PANELS_URL", url)
        assert F.put_panel(panel, "k") == f"compute/online: $FLEET_PANELS_URL is {url!r}, not an https URL: nothing sent"


def test_an_unreadable_archive_state_is_the_passes_error_and_nothing_is_put(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "FLEET_DIR", tmp_path)
    now = int(datetime(2026, 11, 6, 0, 30, tzinfo=UTC).timestamp())
    (tmp_path / "samples-20261105.jsonl").write_text("{}\n")
    (tmp_path / "archived.json").write_text("{not json")
    why = F.archive(now, put=lambda p: (_ for _ in ()).throw(AssertionError("an unreadable state puts nothing")))
    assert why.startswith("archived.json unreadable (JSONDecodeError") and why.endswith("nothing archived until it is fixed")


def test_fleet_runs_by_path_with_no_research_importable(tmp_path):
    """cloud_watch.toml runs `python3.12 .../fleet.py instances` with no PYTHONPATH, where `research` isn't installed: a top-level
    `from research...` import failed there with ModuleNotFoundError. -S keeps the venv's installed `research` out of the run."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    p = subprocess.run([sys.executable, "-S", str(Path(F.__file__)), "--help"], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert p.returncode == 0 and "instances" in p.stdout, p.stderr[-500:]
