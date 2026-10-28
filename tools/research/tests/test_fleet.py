"""pods/nebius/fleet.py (INF-60): the fleet's usage record and daily compute report, the H200s included, and the quota watch."""
from __future__ import annotations

import json

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
    assert all(a[0] == "--to" and a[2:] == ["--kind", "fyi"] for a, _ in sent[2:])
    assert sorted(rings) == ["@circuits", "@infra", "@kernels"]
    assert rings["@circuits"].startswith("Your machines sat idle 3.0 h ($93) in the 24 h to")
    assert "- vy-glm53-eu: idle 2.0 h of 2.0 h running, $86" in rings["@circuits"] and "vy-glm53-cpu-2: idle 1.0 h" in rings["@circuits"]
    assert "vy-nebius-1" not in rings["@circuits"] and "vy-dc-h200" not in rings["@circuits"]
    assert "plus machines with no rate registered" in rings["@kernels"] and "$? (no rate registered)" in rings["@kernels"]
    assert "vy-nebius-1: idle 1.0 h" in rings["@infra"] and "vy-unregistered: idle 1.0 h" in rings["@infra"]
    assert not any("@" in text for _, text in sent[2:]), "a body naming a handle past its start is refused by msg send"


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
