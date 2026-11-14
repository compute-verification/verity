"""Node 1's Commit pacer (sky/release.py) holds the deployments-gpu LocalQueue itself, because Kueue admits a leased Commit (no GPU
request) before a tick can deactivate it, and a few such Commits project far past the limit. The hold stays while
the admitted Commits' projection plus the next Commit's would pass the limit, or the disk is at DISK_PAUSE_PCT; with nothing
admitted it lets one through, so a big Commit can't starve. It releases only a hold it set, and it never touches the
ClusterQueue (the quiet hour's and the disk guard's) or the lease pools' holders, which n1_lease.py submits to their own
LocalQueue: holding them would leave admitted Commits waiting for a GPU."""
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
    monkeypatch.setattr(mod, "room_limit", lambda gb: mod.disk["room"])
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


def test_it_holds_while_the_admitted_projection_passes_the_limit_and_lets_its_own_hold_go_under_it(pacer):
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
    pacer.disk["bundles"] = 800                      # 920 is under the limit, 920 + 480 isn't
    pacer.tick()
    assert pacer.cluster["policy"] == "Hold" and pacer.cluster["wls"]["cov-b"]["spec"]["active"] is False
    pacer.disk["bundles"] = 500                      # 620 + 480 fits: cov-b is released, and the gate opens for it
    pacer.tick()
    assert pacer.cluster["policy"] == "None" and pacer.cluster["wls"]["cov-b"]["spec"]["active"] is True


def test_with_nothing_in_flight_a_big_commit_is_let_through(pacer):
    commit(pacer, "cov-b", B32, active=False)
    pacer.cluster["policy"] = "Hold"
    pacer.GATE.write_text("held\n")
    pacer.disk["bundles"] = 1100                     # 1100 + 480 is over the limit, but nothing is in flight
    pacer.tick()
    assert pacer.cluster["wls"]["cov-b"]["spec"]["active"] is True and pacer.cluster["policy"] == "None"


def test_at_80_percent_disk_it_holds_even_under_the_limit(pacer):
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
    assert pacer.SHARED.read_text() == pacer.SIZES.read_text()      # mirrored where the pack pods read it (commit_pack.py)


def test_a_failed_commits_partial_bundle_teaches_nothing(pacer):
    commit(pacer, "cov-a", B32, admitted=True)
    pacer.written[B32] = 5.0
    pacer.tick()
    del pacer.cluster["wls"]["cov-a"]
    pacer.cluster["jobs"]["job-cov-a"]["status"] = {"failed": 1}
    pacer.tick()
    assert pacer.estimate_gb(B8) == 120.0 and not pacer.SIZES.exists()


YI = "yi15-6b__bf16__rtxpro6000__tp1__b8__i1024__o128__{}"   # no size: the formula guesses it as Phi-3-mini, 120 GB at b8


def active(pacer) -> list[str]:
    return sorted(n for n, w in pacer.cluster["wls"].items() if w["spec"]["active"])


def test_six_commits_of_a_model_with_no_size_go_one_at_a_time_until_the_first_has_taught_it(pacer):
    """qwen25-3b and yi15-6b b32 Commits, guessed up to 6x low by the formula, would fill node 1's disk in minutes."""
    for n in range(6):
        commit(pacer, f"cov-u{n}", YI.format(n), active=False)
    commit(pacer, "cov-p", B8, active=False)                 # a model the pacer has a size for isn't held back by them
    pacer.tick()
    assert active(pacer) == ["cov-p", "cov-u0"]              # all seven would fit the 1200 GB limit at the formula's 120 GB
    for n in ("cov-u0", "cov-p"):
        pacer.cluster["wls"][n]["status"] = {"admission": {"clusterQueue": "deployments-gpu"}}
    pacer.written[YI.format(0)] = 30.0
    pacer.tick()
    assert active(pacer) == ["cov-p", "cov-u0"] and pacer.cluster["policy"] == "Hold"    # the next arrival may be a yi15-6b too
    del pacer.cluster["wls"]["cov-u0"]
    pacer.cluster["jobs"]["job-cov-u0"]["status"] = {"succeeded": 1}
    pacer.tick()                                             # its 30 GB for 8 x 1152 taught it: 37.5 GB each from now on
    assert active(pacer) == ["cov-p", "cov-u1", "cov-u2", "cov-u3", "cov-u4", "cov-u5"] and pacer.cluster["policy"] == "None"


def test_a_commit_of_a_model_with_no_size_admitted_on_arrival_holds_the_queue_for_the_next_one(pacer):
    """Commits that arrive while the gate is open are each admitted by Kueue the moment they arrive."""
    commit(pacer, "cov-u0", YI.format(0), admitted=True)
    pacer.tick()
    assert pacer.cluster["policy"] == "Hold"
    commit(pacer, "cov-u1", YI.format(1))                    # arrives under the hold: active, not admitted
    pacer.tick()
    assert active(pacer) == ["cov-u0"]
    del pacer.cluster["wls"]["cov-u0"]
    pacer.cluster["jobs"]["job-cov-u0"]["status"] = {"failed": 1}   # a failed one teaches nothing: cov-u1 is the next to go
    pacer.tick()
    assert active(pacer) == ["cov-u1"] and pacer.cluster["policy"] == "None"


def test_a_tp_commit_writes_no_bundle_so_its_model_needs_no_size_and_holds_nothing(pacer):
    """A TP2 qwen3-14b Commit must not hold the queue as a model with no size while node 1's GPUs are free: a TP Commit replays
    in its own process and writes no replay bundle, so it never teaches one either."""
    tp2 = "qwen3-14b__bf16__rtxpro6000__tp2__b8__i1024__o128__g"
    commit(pacer, "cov-t", tp2, admitted=True)
    commit(pacer, "cov-u", YI.format(0), active=False)
    pacer.tick()
    assert pacer.estimate_gb(tp2) == 0.0 and pacer.cluster["policy"] == "None" and active(pacer) == ["cov-t", "cov-u"]
    assert not pacer.sized(tp2.replace("__tp2__", "__tp1__")), "its TP1 rows write bundles, and still teach the size first"


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


def test_the_latch_under_its_older_name_still_latches_and_is_renamed(pacer):
    pacer.OLD_LATCH.write_text("2026-11-01T00:00:00Z disk 78.2%\n")
    pacer.disk["bundles"] = 140                      # 140 + 120 is over the dropped 150 GB, under the 1200 GB room
    commit(pacer, "cov-a", B8, admitted=True)
    pacer.tick()
    assert not pacer.OLD_LATCH.exists() and pacer.LIMIT_LATCH.read_text() == "2026-11-01T00:00:00Z disk 78.2%\n"
    assert pacer.LIMIT_LATCH.name == "limit-150" and pacer.cluster["policy"] == "Hold", "latched: the limit is LIMIT_LOW_GB"
