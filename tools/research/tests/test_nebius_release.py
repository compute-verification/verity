"""Node 1's Commit pacer (sky/release.py) holds the deployments-gpu LocalQueue itself, because Kueue admits a leased Commit (no GPU
request) before a tick can deactivate it: on Oct 1 four such Commits projected 2.1 TB against a 1.2 TB cap. The hold stays while
the admitted Commits' projection plus the next Commit's would pass the cap, or the disk is at DISK_PAUSE_PCT; with nothing
admitted it lets one through, so a big Commit can't starve. It releases only a hold it set, and it never touches the
ClusterQueue (the quiet hour's and the disk guard's) or the lease pools' holders, which n1_lease.py submits to their own
LocalQueue: holding them once left two admitted Commits waiting for a GPU (Oct 1, 9:09-9:32 PM PDT)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from research.pods import registry

NEBIUS = Path(registry.__file__).resolve().parent / "nebius"
B8, B32 = "phi3-mini__bf16__rtxpro6000__tp1__b8__i1024__o128__g", "phi3-mini__bf16__rtxpro6000__tp1__b32__i1024__o128__g"  # 120, 480 GB


def load(name: str, path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def pacer(tmp_path, monkeypatch):
    monkeypatch.setenv("VY_PACER_DIR", str(tmp_path / "pacer"))
    mod = load("release", NEBIUS / "sky" / "release.py", monkeypatch)
    mod.ROOT.mkdir(parents=True)
    mod.cluster = {"policy": "None", "patches": [], "wls": {}, "jobs": {}}

    def k(*args, stdin=None):
        if args[:5] == ("get", "localqueue", mod.GPU_QUEUE, "-n", "default"):
            return "" if mod.cluster["policy"] == "None" else mod.cluster["policy"]
        if args[:5] == ("patch", "localqueue", mod.GPU_QUEUE, "-n", "default"):
            mod.cluster["policy"] = json.loads(args[-1])["spec"]["stopPolicy"]
            mod.cluster["patches"].append(mod.cluster["policy"])
            return ""
        if args[:2] == ("patch", "workload"):
            mod.cluster["wls"][args[2]]["spec"]["active"] = json.loads(args[-1])["spec"]["active"]
            return ""
        if args[:2] == ("get", "jobs"):
            return json.dumps({"items": list(mod.cluster["jobs"].values())})
        if args[:2] == ("get", "workloads"):
            return json.dumps({"items": list(mod.cluster["wls"].values())})
        raise AssertionError(f"the pacer touched something other than its LocalQueue and Commits: {args}")

    monkeypatch.setattr(mod, "k", k)
    mod.written = {}
    monkeypatch.setattr(mod, "written_gb", lambda w: mod.written.get(w["row"], 0.0))
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
    mod.cluster["jobs"][f"job-{name}"] = {"metadata": {"name": f"job-{name}", "annotations": {"verity.dev/item": f"vllm-epoch-run/{name}"}},
                                          "status": {}}


def test_it_holds_while_the_admitted_projection_passes_the_cap_and_lets_its_own_hold_go_under_it(pacer):
    commit(pacer, "cov-a", B8, admitted=True)
    pacer.disk["bundles"] = 1100                     # 1100 + 120 > 1200
    pacer.tick()
    assert pacer.cluster["patches"] == ["Hold"] and pacer.GATE.exists()
    pacer.tick()
    assert pacer.cluster["patches"] == ["Hold"]
    pacer.disk["bundles"] = 900                      # 900 + 120, and room for another 120
    pacer.tick()
    assert pacer.cluster["patches"] == ["Hold", "None"] and not pacer.GATE.exists()


def test_it_opens_only_when_the_next_waiting_commit_fits_too(pacer):
    commit(pacer, "cov-a", B8, admitted=True)
    commit(pacer, "cov-b", B32, active=False)
    pacer.disk["bundles"] = 800                      # 920 is under the cap, 920 + 480 isn't
    pacer.tick()
    assert pacer.cluster["policy"] == "Hold" and pacer.cluster["wls"]["cov-b"]["spec"]["active"] is False
    pacer.disk["bundles"] = 500                      # 620 + 480 fits: cov-b is released, and the gate opens for it
    pacer.tick()
    assert pacer.cluster["policy"] == "None" and pacer.cluster["wls"]["cov-b"]["spec"]["active"] is True


def test_with_nothing_in_flight_a_big_commit_is_let_through(pacer):
    commit(pacer, "cov-b", B32, active=False)
    pacer.cluster["policy"] = "Hold"
    pacer.GATE.write_text("held\n")
    pacer.disk["bundles"] = 1100                     # 1100 + 480 is over the cap, but nothing is in flight
    pacer.tick()
    assert pacer.cluster["wls"]["cov-b"]["spec"]["active"] is True and pacer.cluster["policy"] == "None"


def test_at_80_percent_disk_it_holds_even_under_the_cap(pacer):
    pacer.disk.update(bundles=10, pct=80.0)
    pacer.tick()
    assert pacer.cluster["policy"] == "Hold" and pacer.GATE.exists()
    pacer.disk["pct"] = 79.0
    pacer.tick()
    assert pacer.cluster["policy"] == "None"


def test_it_never_releases_a_hold_it_did_not_set_nor_a_hold_and_drain(pacer):
    pacer.cluster["policy"] = "Hold"
    pacer.tick()
    assert pacer.cluster["patches"] == [] and pacer.cluster["policy"] == "Hold"
    commit(pacer, "cov-a", B8, admitted=True)
    pacer.disk["bundles"] = 1100
    pacer.cluster["policy"] = "HoldAndDrain"
    pacer.tick()
    assert pacer.cluster["patches"] == [] and not pacer.GATE.exists()


def test_a_succeeded_commits_bundle_teaches_its_model(pacer):
    commit(pacer, "cov-a", B32, admitted=True)
    pacer.written[B32] = 60.0
    pacer.tick()
    pacer.written[B32] = 96.0                        # its bundle grows to 96 GB for 32 x 1152
    pacer.tick()
    assert pacer.estimate_gb(B8) == 120.0            # in flight: nothing learned yet
    del pacer.cluster["wls"]["cov-a"]
    pacer.cluster["jobs"]["job-cov-a"]["status"] = {"succeeded": 1}
    pacer.tick()
    assert pacer.estimate_gb(B8) == pytest.approx(96.0 / 4 * 1.25)
    assert json.loads(pacer.SIZES.read_text()) == {"phi3-mini": pytest.approx(96.0 / (32 * 1152))}


def test_a_failed_commits_partial_bundle_teaches_nothing(pacer):
    commit(pacer, "cov-a", B32, admitted=True)
    pacer.written[B32] = 5.0
    pacer.tick()
    del pacer.cluster["wls"]["cov-a"]
    pacer.cluster["jobs"]["job-cov-a"]["status"] = {"failed": 1}
    pacer.tick()
    assert pacer.estimate_gb(B8) == 120.0 and not pacer.SIZES.exists()


def test_the_lease_pool_holders_go_to_their_own_local_queue_on_the_same_cluster_queue(monkeypatch):
    monkeypatch.setenv("VY_POOL_QUEUE", "deployments-gpu")
    monkeypatch.setenv("VY_POOL_LOCAL_QUEUE", "deployments-gpu-pool")
    lease = load("n1_lease", NEBIUS / "n1_lease.py", monkeypatch)
    assert lease.holder_job(0.0)["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "deployments-gpu-pool"
    assert lease.QUEUE == "deployments-gpu"
    monkeypatch.delenv("VY_POOL_LOCAL_QUEUE")
    assert load("n1_lease", NEBIUS / "n1_lease.py", monkeypatch).holder_job(0.0)["metadata"]["labels"][
        "kueue.x-k8s.io/queue-name"] == "deployments-gpu"
    kueue = (NEBIUS / "sky" / "kueue.yaml").read_text()
    assert "metadata: {name: deployments-gpu-pool, namespace: default}" in kueue
