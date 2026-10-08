"""The pod registry (pods/registry.py): one file per machine, merged with machines.toml, refused when the two disagree; `register`
checks the pod against the account before writing; G1 (create, register, `--on` resolves it without a hand edit) and G2 (an unknown
pod id is refused).  No network: the RunPod lookups are stubbed."""
from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

from research import notes_sync, remote
from research.pods import part, registry, runpod

POD = {"id": "abc123def45678", "name": "vy-lane-a-veritor-campaign", "desiredStatus": "RUNNING"}
OTHER = {"id": "zzz999yyy88877", "name": "vy-lane-b-veritor-campaign", "desiredStatus": "RUNNING"}


def pods(*known: dict):
    by = {p["id"]: p for p in known}

    def get(pod_id: str) -> dict:
        if pod_id not in by:
            raise runpod.PodError(f"GET /pods/{pod_id}: HTTP 404", code=404, pod_id=pod_id)
        return by[pod_id]
    return get


@pytest.fixture
def reg(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "notes" / "machines.d"
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setenv("RESEARCH_MACHINES_D", str(d))
    monkeypatch.setenv("RESEARCH_NOTES", str(d.parent))
    return d


def test_register_and_run_on_use_one_directory_with_or_without_machines_toml(tmp_path):
    mt = tmp_path / "machines.toml"
    mt.write_text('[machines.vy-x]\nprovider = "runpod"\npod_id = "abc123def45678"\nproject = "verity"\n')
    notes_d = tmp_path / "notes" / "machines.d"
    notes_d.mkdir(parents=True)
    env = {"RESEARCH_NOTES": str(tmp_path / "notes")}
    for toml in (mt, tmp_path / "absent.toml"):
        assert registry.registry_dir(toml, env) == registry.write_dir(toml, env) == notes_d
    r = {**env, "RESEARCH_MACHINES_D": str(tmp_path / "r")}
    assert registry.registry_dir(mt, r) == registry.write_dir(mt, r) == tmp_path / "r"
    off = {**env, "RESEARCH_MACHINES_D": "none"}
    assert registry.registry_dir(mt, off) is None
    with pytest.raises(remote.Refused, match="registry is off"):
        registry.write_dir(mt, off)


def test_a_pod_registered_beside_machines_toml_is_one_run_on_resolves(tmp_path, monkeypatch):
    mt = tmp_path / "machines.toml"
    mt.write_text('[machines.vy-x]\nprovider = "runpod"\npod_id = "abc123def45678"\nproject = "verity"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(mt))
    monkeypatch.setenv("RESEARCH_NOTES", str(tmp_path / "notes"))
    monkeypatch.delenv("RESEARCH_MACHINES_D", raising=False)
    (tmp_path / "notes").mkdir()
    p, _ = registry.register("vy-lane-a", POD["id"], "verity", get_pod=pods(POD))
    assert p == tmp_path / "notes" / "machines.d" / "vy-lane-a.toml"
    assert remote.resolve_machine("vy-lane-a", "verity")["pod_id"] == POD["id"]
    assert remote.resolve_machine("vy-x", "verity")["pod_id"] == "abc123def45678"


def test_register_writes_one_file_and_run_on_resolves_it(reg):
    p, what = registry.register("vy-lane-a", POD["id"], "verity", guard="45", get_pod=pods(POD))
    assert what == "registered" and p == reg / "vy-lane-a.toml"
    doc = tomllib.loads(p.read_text())
    assert doc["pod_id"] == POD["id"] and doc["project"] == "verity" and doc["guard"] == 45 and doc["registered_at"]
    m = remote.resolve_machine("vy-lane-a", "verity")
    assert m["pod_id"] == POD["id"] and m["root"] == "/workspace/research" and remote.guard_idle_min(m) == 45.0
    with pytest.raises(remote.Refused, match="belongs to project"):
        remote.resolve_machine("vy-lane-a", "veritor")
    assert registry.register("vy-lane-a", POD["id"], "verity", guard="45", get_pod=pods(POD))[1] == "unchanged"
    registry.register("vy-lane-b", OTHER["id"], "verity", get_pod=pods(OTHER))
    assert sorted(x.name for x in reg.glob("*.toml")) == ["vy-lane-a.toml", "vy-lane-b.toml"]   # two pods, two files, no merge


def test_g2_an_unknown_pod_id_is_refused_and_nothing_is_written(reg):
    with pytest.raises(remote.Refused, match="not in the RunPod account"):
        registry.register("vy-lane-a", "nosuchpod00000", "verity", get_pod=pods(POD))

    def down(_):
        raise OSError("no network")
    with pytest.raises(remote.Refused, match="cannot look pod"):
        registry.register("vy-lane-a", POD["id"], "verity", get_pod=down)
    assert not reg.exists()
    with pytest.raises(remote.Refused, match="not in .*only listed machines"):
        remote.resolve_machine("nosuchpod00000", "verity")


def test_register_refuses_a_foreign_name_a_bad_name_and_a_silent_rebind(reg, tmp_path):
    with pytest.raises(remote.Refused, match="named 'vy-lane-a-veritor-campaign', not 'vy-other'"):
        registry.register("vy-other", POD["id"], "verity", get_pod=pods(POD))
    assert registry.register("vy-other", POD["id"], "verity", any_name=True, get_pod=pods(POD))[1] == "registered"
    with pytest.raises(remote.Refused, match="not a machine name"):
        registry.register("../x", POD["id"], "verity", any_name=True, get_pod=pods(POD))
    with pytest.raises(remote.Refused, match="--replace rebinds"):
        registry.register("vy-other", OTHER["id"], "verity", any_name=True, get_pod=pods(OTHER))
    assert registry.register("vy-other", OTHER["id"], "verity", any_name=True, replace=True, get_pod=pods(OTHER))[1] == "updated"
    (tmp_path / "machines.toml").write_text(f'[machines.vy-lane-b]\nprovider = "runpod"\npod_id = "{POD["id"]}"\nproject = "verity"\n')
    with pytest.raises(remote.Refused, match="machines.toml binds vy-lane-b to pod abc123"):
        registry.register("vy-lane-b", OTHER["id"], "verity", get_pod=pods(OTHER))


def test_a_name_the_two_sources_bind_to_different_pods_is_refused_not_guessed(reg, tmp_path):
    (tmp_path / "machines.toml").write_text(
        f'[machines.vy-lane-a]\nprovider = "runpod"\npod_id = "{OTHER["id"]}"\nproject = "verity"\n'
        f'[machines.vy-old]\nprovider = "runpod"\npod_id = "{OTHER["id"]}"\nproject = "verity"\n')
    registry.write_entry(reg, "vy-lane-a", {"provider": "runpod", "pod_id": POD["id"], "project": "verity"})
    registry.write_entry(reg, "vy-old", {"provider": "runpod", "pod_id": OTHER["id"], "project": "verity"})   # agrees: fine
    (reg / "vy-broken.toml").write_text("pod_id = \n")
    (reg / "vy-noproj.toml").write_text(f'provider = "runpod"\npod_id = "{POD["id"]}"\n')
    ms = remote.load_machines()
    assert set(ms) == {"vy-old"}
    with pytest.raises(remote.Refused, match="machines.toml binds it to pod zzz.*to pod abc.*refused"):
        remote.resolve_machine("vy-lane-a", "verity")
    with pytest.raises(remote.Refused, match="does not parse"):
        remote.resolve_machine("vy-broken", "verity")
    with pytest.raises(remote.Refused, match="no project"):
        remote.resolve_machine("vy-noproj", "verity")
    assert remote.load_machines(tmp_path / "machines.toml").keys() == {"vy-lane-a", "vy-old"}    # an explicit file stays that file
    listing = {r["name"]: r for r in registry.rows(tmp_path / "machines.toml")}
    assert listing["vy-lane-a"]["source"] == "refused" and listing["vy-old"]["source"] == "both"


def test_families_come_from_machines_toml_then_the_registry(reg, tmp_path):
    (tmp_path / "machines.toml").write_text("[families.rtx4090]\nvram_mb = 24564\n")
    reg.mkdir(parents=True)
    (reg / "_families.toml").write_text("[families.rtx4090]\nvram_mb = 1\n[families.h100]\nvram_mb = 81559\n")
    assert part.load_families() == {"rtx4090": {"vram_mb": 24564}, "h100": {"vram_mb": 81559}}
    assert remote.load_machines() == {}                     # `_families.toml` is not a machine


def test_import_seeds_the_registry_from_machines_toml(reg, tmp_path):
    mt = tmp_path / "machines.toml"
    mt.write_text(f'[machines.vy-lane-a]\nprovider = "runpod"\npod_id = "{POD["id"]}"\nproject = "verity"\nguard = 90\n'
                  f'[machines.vy-dead]\nprovider = "runpod"\npod_id = "{OTHER["id"]}"\nproject = "verity"\n')
    plan = {r["name"]: r for r in registry.plan_import(mt, reg, running={POD["id"]})}
    assert plan["vy-lane-a"]["action"] == "write" and plan["vy-dead"]["why"] == "pod not running"
    assert {r["name"] for r in registry.plan_import(mt, reg, running=None) if r["action"] == "write"} == {"vy-lane-a", "vy-dead"}


def test_g1_create_register_then_run_on_needs_no_hand_edit(reg, monkeypatch, capsys):
    monkeypatch.setattr(runpod, "_request", lambda method, path, body=None: [])      # create_once's live-pod lookup: none of the name
    monkeypatch.setattr(runpod, "create", lambda ns: {"id": POD["id"]})
    monkeypatch.setattr(runpod, "wait_ssh", lambda pod_id: ("203.0.113.7", 40123))
    monkeypatch.setattr(runpod, "get", pods(POD))
    monkeypatch.setattr(part, "check_pod", lambda pod_id: {"pod_id": pod_id, "sku": "x", "count": 1, "vram_mb": 1, "reference_vram_mb": 1,
                                                           "cpu": "c", "load": 0, "hostname": "h", "verdict": "reference", "reference": True})
    from research.pods import budgets, lease

    monkeypatch.setattr(budgets, "check_create", lambda name, hours: (None, "vy-lane-"))
    monkeypatch.setattr(lease, "arm", lambda ssh, seconds, **kw: {"expires_utc": "t", "lease_pid": 1, "lease_alive": True, "guard_pid": 2,
                                                                  "guard_alive": True})
    assert part.main(["create", "--name", "vy-lane-a", "--max-hours", "2", "--register", "--project", "verity", "--guard", "true"]) == 0
    assert "REGISTERED vy-lane-a -> pod abc123def45678" in capsys.readouterr().out
    m = remote.resolve_machine("vy-lane-a", "verity")
    assert m["pod_id"] == POD["id"] and remote.guard_idle_min(m) == 90.0
    assert part.main(["registry"]) == 0 and "vy-lane-a" in capsys.readouterr().out


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def test_registrations_on_two_machines_both_reach_the_notes_remote(tmp_path, monkeypatch):
    """Without being asked: vy-nebius-cpu-1 was registered at 15:20Z on 7 Oct and invisible to every other agent's `run --on`
    until a notes sync at 19:25Z."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "-q", str(origin), str(seed)], check=True)
    for k, v in (("user.email", "t@example.com"), ("user.name", "t")):
        _git(seed, "config", k, v)
    (seed / "README.md").write_text("notes\n")
    _git(seed, "add", "README.md")
    _git(seed, "commit", "-qm", "seed")
    _git(seed, "push", "-q", "origin", "HEAD:main")
    clones = []
    for n in ("a", "b"):
        c = tmp_path / f"notes-{n}"
        subprocess.run(["git", "clone", "-q", str(origin), str(c)], check=True)
        for k, v in (("user.email", f"{n}@example.com"), ("user.name", n)):
            _git(c, "config", k, v)
        clones.append(c)
    monkeypatch.setattr(runpod, "get", pods(POD, OTHER))
    monkeypatch.delenv(notes_sync.SYNC_ENV, raising=False)
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "absent.toml"))
    monkeypatch.delenv("RESEARCH_MACHINES_D")
    for c, (name, pod) in zip(clones, (("vy-lane-a", POD), ("vy-lane-b", OTHER))):
        monkeypatch.setenv("RESEARCH_NOTES", str(c))
        assert registry.main(["register", name, "--pod-id", pod["id"], "--project", "verity"]) == 0
    check = tmp_path / "check"
    subprocess.run(["git", "clone", "-q", str(origin), str(check)], check=True)
    assert sorted(p.name for p in (check / "machines.d").glob("*.toml")) == ["vy-lane-a.toml", "vy-lane-b.toml"]
    monkeypatch.setenv("RESEARCH_NOTES", str(check))
    assert remote.resolve_machine("vy-lane-b", "verity")["pod_id"] == OTHER["id"]
    head = _git(origin, "rev-parse", "main")
    monkeypatch.setenv(notes_sync.SYNC_ENV, "0")
    monkeypatch.setenv("RESEARCH_NOTES", str(clones[1]))
    assert registry.main(["register", "vy-lane-c", "--pod-id", OTHER["id"], "--project", "verity", "--any-name"]) == 0
    assert (clones[1] / "machines.d" / "vy-lane-c.toml").exists() and _git(origin, "rev-parse", "main") == head, "0 leaves it local"


def test_an_override_directory_never_hides_a_machine_the_notes_clone_lists(tmp_path, monkeypatch):
    """The lander's VM, 8 Oct 03:43Z: RESEARCH_MACHINES_D=/tmp/machines.d, copied by hand on 30 Sep, refused vy-nebius-cpu-1,
    which had been in the notes' machines.d since 7 Oct 17:16Z."""
    notes_d, override = tmp_path / "notes" / "machines.d", tmp_path / "copy"
    for d, names in ((notes_d, ("vy-new", "vy-both")), (override, ("vy-both",))):
        d.mkdir(parents=True)
        for n in names:
            host = "10.0.0.2" if d == override else "10.0.0.1"
            (d / f"{n}.toml").write_text(f'provider = "ssh"\nhost = "{host}"\nproject = "verity"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setenv("RESEARCH_NOTES", str(tmp_path / "notes"))
    monkeypatch.setenv("RESEARCH_MACHINES_D", str(override))
    assert remote.resolve_machine("vy-new", "verity")["host"] == "10.0.0.1"
    assert remote.resolve_machine("vy-both", "verity")["host"] == "10.0.0.2"          # the override still wins on its own names
    with pytest.raises(remote.Refused, match="is not in"):
        remote.resolve_machine("vy-absent", "verity")
    monkeypatch.setenv("RESEARCH_MACHINES_D", "none")
    with pytest.raises(remote.Refused, match="is not in"):
        remote.resolve_machine("vy-new", "verity")                                     # off means machines.toml alone
