"""Node 1's Commit pacer (sky/release.py) holds deployments-gpu itself while its bundle projection is at the cap or the disk at
DISK_PAUSE_PCT, because Kueue admits a leased Commit (no GPU request) before a tick can deactivate it: on Oct 1 four such
Commits projected 2.1 TB against a 1.2 TB cap. It releases only a hold it set, never in the quiet hour, and never while the
disk guard holds the queue."""
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


@pytest.fixture
def pacer(tmp_path, monkeypatch):
    monkeypatch.setenv("VY_PACER_DIR", str(tmp_path / "pacer"))
    spec = importlib.util.spec_from_file_location("release", SKY / "release.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "release", mod)
    spec.loader.exec_module(mod)
    mod.ROOT.mkdir(parents=True)
    mod.cluster = {"policy": "None", "patches": [], "guard": set()}

    def k(*args, stdin=None):
        if args[:3] == ("get", "clusterqueue", mod.GPU_QUEUE):
            return "" if mod.cluster["policy"] == "None" else mod.cluster["policy"]
        if args[:3] == ("patch", "clusterqueue", mod.GPU_QUEUE):
            mod.cluster["policy"] = json.loads(args[-1])["spec"]["stopPolicy"]
            mod.cluster["patches"].append(mod.cluster["policy"])
            return ""
        if args[:2] in (("get", "jobs"), ("get", "workloads")):
            return json.dumps({"items": []})
        raise AssertionError(args)

    monkeypatch.setattr(mod, "k", k)
    monkeypatch.setattr(mod, "guard_holds", lambda: mod.cluster["guard"])
    monkeypatch.setattr(mod, "written_gb", lambda w: 0.0)
    mod.disk = {"bundles": 0.0, "pct": 50.0, "room": 1200}
    monkeypatch.setattr(mod, "bundles_gb", lambda: mod.disk["bundles"])
    monkeypatch.setattr(mod, "disk_pct", lambda: mod.disk["pct"])
    monkeypatch.setattr(mod, "room_cap", lambda gb: mod.disk["room"])
    return mod


def test_at_the_cap_the_pacer_holds_the_gpu_queue_and_lets_its_own_hold_go_under_it(pacer):
    pacer.disk["bundles"] = 1300
    pacer.tick(NOW)
    assert pacer.cluster["patches"] == ["Hold"] and pacer.GATE.exists()
    pacer.tick(NOW)
    assert pacer.cluster["patches"] == ["Hold"]
    pacer.disk["bundles"] = 100
    pacer.tick(NOW)
    assert pacer.cluster["patches"] == ["Hold", "None"] and not pacer.GATE.exists()


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
    pacer.disk["bundles"] = 1300
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
    pacer.disk["bundles"] = 1300
    pacer.tick(QUIET)
    assert pacer.cluster["patches"] == []
    pacer.cluster["policy"] = "HoldAndDrain"
    pacer.tick(NOW)
    assert pacer.cluster["patches"] == [] and not pacer.GATE.exists()
