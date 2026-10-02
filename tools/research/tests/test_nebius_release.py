"""Node 1's Commit pacer (sky/release.py) holds deployments-gpu itself, because Kueue admits a leased Commit (no GPU request) before
a tick can deactivate it: on Oct 1 four such Commits projected 2.1 TB against a 1.2 TB cap. The hold stays while the admitted
Commits' projection plus the next Commit's would pass the cap, or the disk is at DISK_PAUSE_PCT; with nothing admitted it lets
one through, so a big Commit can't starve. It releases only a hold it set, never in the quiet hour, and never while the disk
guard holds the queue."""
from __future__ import annotations

import calendar
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from research.pods import registry

SKY = Path(registry.__file__).resolve().parent / "nebius" / "sky"
NOW = calendar.timegm((2026, 10, 2, 4, 0, 0))
QUIET = calendar.timegm((2026, 10, 2, 12, 45, 0))
B8, B32 = "phi3-mini__bf16__rtxpro6000__tp1__b8__i1024__o128__g", "phi3-mini__bf16__rtxpro6000__tp1__b32__i1024__o128__g"  # 120, 480 GB


@pytest.fixture
def pacer(tmp_path, monkeypatch):
    monkeypatch.setenv("VY_PACER_DIR", str(tmp_path / "pacer"))
    spec = importlib.util.spec_from_file_location("release", SKY / "release.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "release", mod)
    spec.loader.exec_module(mod)
    mod.ROOT.mkdir(parents=True)
    mod.cluster = {"policy": "None", "patches": [], "guard": set(), "wls": {}}

    def k(*args, stdin=None):
        if args[:3] == ("get", "clusterqueue", mod.GPU_QUEUE):
            return "" if mod.cluster["policy"] == "None" else mod.cluster["policy"]
        if args[:3] == ("patch", "clusterqueue", mod.GPU_QUEUE):
            mod.cluster["policy"] = json.loads(args[-1])["spec"]["stopPolicy"]
            mod.cluster["patches"].append(mod.cluster["policy"])
            return ""
        if args[:2] == ("patch", "workload"):
            mod.cluster["wls"][args[2]]["spec"]["active"] = json.loads(args[-1])["spec"]["active"]
            return ""
        if args[:2] == ("get", "jobs"):
            return json.dumps({"items": [{"metadata": {"name": f"job-{n}", "annotations": {"verity.dev/item": f"vllm-epoch-run/{n}"}}}
                                         for n in mod.cluster["wls"]]})
        if args[:2] == ("get", "workloads"):
            return json.dumps({"items": list(mod.cluster["wls"].values())})
        raise AssertionError(args)

    monkeypatch.setattr(mod, "k", k)
    monkeypatch.setattr(mod, "guard_holds", lambda: mod.cluster["guard"])
    monkeypatch.setattr(mod, "written_gb", lambda w: 0.0)
    mod.disk = {"bundles": 0.0, "pct": 50.0, "room": 1200}
    monkeypatch.setattr(mod, "bundles_gb", lambda: mod.disk["bundles"])
    monkeypatch.setattr(mod, "disk_pct", lambda: mod.disk["pct"])
    monkeypatch.setattr(mod, "room_cap", lambda gb: mod.disk["room"])
    return mod


def commit(mod, name: str, row: str, admitted: bool = False, active: bool = True) -> None:
    """A circuits Commit workload in deployments-gpu."""
    mod.cluster["wls"][name] = {
        "metadata": {"name": name, "ownerReferences": [{"kind": "Job", "name": f"job-{name}"}]},
        "spec": {"queueName": "deployments-gpu", "active": active,
                 "podSets": [{"template": {"spec": {"containers": [{"env": [{"name": "ROW", "value": row}]}]}}}]},
        "status": {"admission": {"clusterQueue": "deployments-gpu"}} if admitted else {}}


def test_it_holds_while_the_admitted_projection_passes_the_cap_and_lets_its_own_hold_go_under_it(pacer):
    commit(pacer, "cov-a", B8, admitted=True)
    pacer.disk["bundles"] = 1100                     # 1100 + 120 > 1200
    pacer.tick(NOW)
    assert pacer.cluster["patches"] == ["Hold"] and pacer.GATE.exists()
    pacer.tick(NOW)
    assert pacer.cluster["patches"] == ["Hold"]
    pacer.disk["bundles"] = 900                      # 900 + 120, and room for another 120
    pacer.tick(NOW)
    assert pacer.cluster["patches"] == ["Hold", "None"] and not pacer.GATE.exists()


def test_it_opens_only_when_the_next_waiting_commit_fits_too(pacer):
    commit(pacer, "cov-a", B8, admitted=True)
    commit(pacer, "cov-b", B32, active=False)
    pacer.disk["bundles"] = 800                      # 920 is under the cap, 920 + 480 isn't
    pacer.tick(NOW)
    assert pacer.cluster["policy"] == "Hold" and pacer.cluster["wls"]["cov-b"]["spec"]["active"] is False
    pacer.disk["bundles"] = 500                      # 620 + 480 fits: cov-b is released, and the gate opens for it
    pacer.tick(NOW)
    assert pacer.cluster["policy"] == "None" and pacer.cluster["wls"]["cov-b"]["spec"]["active"] is True


def test_with_nothing_in_flight_a_big_commit_is_let_through(pacer):
    commit(pacer, "cov-b", B32, active=False)
    pacer.cluster["policy"] = "Hold"
    pacer.GATE.write_text("held\n")
    pacer.disk["bundles"] = 1100                     # 1100 + 480 is over the cap, but nothing is in flight
    pacer.tick(NOW)
    assert pacer.cluster["wls"]["cov-b"]["spec"]["active"] is True and pacer.cluster["policy"] == "None"


def test_a_pending_lease_pool_holder_opens_the_gate_so_admitted_commits_get_their_gpu(pacer):
    commit(pacer, "cov-a", B8, admitted=True)
    pacer.disk["bundles"] = 1100
    pacer.tick(NOW)
    assert pacer.cluster["policy"] == "Hold"
    pacer.cluster["wls"]["gpu-pool-circuits-1"] = {
        "metadata": {"name": "gpu-pool-circuits-1", "ownerReferences": [{"kind": "Job", "name": "gpu-pool-circuits-1"}]},
        "spec": {"queueName": "deployments-gpu", "active": True, "podSets": [{"template": {"spec": {"containers": [{}]}}}]},
        "status": {}}
    pacer.tick(NOW)
    assert pacer.cluster["policy"] == "None"
    pacer.cluster["wls"]["gpu-pool-circuits-1"]["status"] = {"admission": {"clusterQueue": "deployments-gpu"}}
    pacer.tick(NOW)
    assert pacer.cluster["patches"] == ["Hold", "None", "Hold"]


def test_at_80_percent_disk_it_holds_even_under_the_cap(pacer):
    pacer.disk.update(bundles=10, pct=80.0)
    pacer.tick(NOW)
    assert pacer.cluster["policy"] == "Hold" and pacer.GATE.exists()
    pacer.disk["pct"] = 79.0
    pacer.tick(NOW)
    assert pacer.cluster["policy"] == "None"


def test_it_never_releases_a_hold_it_did_not_set(pacer):
    pacer.cluster["policy"] = "Hold"
    pacer.tick(NOW)
    assert pacer.cluster["patches"] == [] and pacer.cluster["policy"] == "Hold"


def test_it_leaves_its_hold_to_the_disk_guard_and_the_quiet_hour(pacer):
    commit(pacer, "cov-a", B8, admitted=True)
    pacer.disk["bundles"] = 1100
    pacer.tick(NOW)
    pacer.disk["bundles"] = 100
    pacer.cluster["guard"] = {pacer.GPU_QUEUE}
    pacer.tick(NOW)
    assert pacer.cluster["patches"] == ["Hold"] and not pacer.GATE.exists()
    pacer.cluster["guard"] = set()
    pacer.GATE.write_text("held\n")
    pacer.tick(QUIET)
    assert pacer.cluster["patches"] == ["Hold"] and pacer.cluster["policy"] == "Hold"


def test_it_holds_nothing_in_the_quiet_hour_and_never_downgrades_a_stronger_hold(pacer):
    commit(pacer, "cov-a", B8, admitted=True)
    pacer.disk["bundles"] = 1100
    pacer.tick(QUIET)
    assert pacer.cluster["patches"] == []
    pacer.cluster["policy"] = "HoldAndDrain"
    pacer.tick(NOW)
    assert pacer.cluster["patches"] == [] and not pacer.GATE.exists()
