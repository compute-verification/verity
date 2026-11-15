"""The control pod's watchers: the supervisor (infra/control/watchers.py) restarts a loop that exits, holds one whose need
is missing, finishes an extra.d job that exits 0, and writes a heartbeat; the VM stops (infra/control/vm_stop.py) keep h200-idle's
idle rule and the deadline and lease backstops."""
import calendar
import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CONTROL = ROOT / "infra" / "control"


def module(name):
    spec = importlib.util.spec_from_file_location(name, CONTROL / f"{name}.py")
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)     # dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


W = module("watchers")
V = module("vm_stop")
T0 = float(calendar.timegm((2026, 10, 19, 20, 0, 0)))


class Proc:
    pids = iter(range(1000, 10**6))

    def __init__(self):
        self.pid, self.rc = next(self.pids), None

    def poll(self):
        return self.rc


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


def supervisor(tmp_path, loops, beats=None):
    clock, started = Clock(), []

    def spawn(lp, logf):
        started.append(lp.name)
        return Proc()

    cfg = {"dir": str(tmp_path), "src": str(tmp_path / "src"), "host": "pod", "extra": str(tmp_path / "extra.d")}
    publish = (lambda body, t: beats.append(json.loads(body))) if beats is not None else None
    return W.Supervisor(cfg, loops, clock=clock, spawn=spawn, publish=publish), clock, started


def test_the_checked_in_config_loads_and_each_credential_comes_from_a_file_on_the_pod():
    cfg, loops = W.load(CONTROL / "watchers.toml")
    assert {"vm-stop", "fleet", "monitors-relay"} <= {lp.name for lp in loops}
    for lp in loops:
        assert all(f.startswith(("/root/.config/verity/", "/data/")) for f in lp.env_files), lp.name
        assert all(v.startswith("/") for k, v in lp.env.items() if k.endswith(("TOKEN", "KEY", "_B64"))), lp.name
        assert all("{" not in a for a in lp.argv), lp.name
    assert cfg["dir"] == "/data/watchers" and lp.env["PYTHONPATH"].startswith("/data/watchers/src/")


def test_the_labels_pack_loop_republishes_hourly_from_a_store_on_the_pods_own_disk():
    """The labels pack is as fresh as this loop keeps it: it runs the deployed tree's `research data labels-pack`, repeating
    hourly, with the store's credential alone and a store off the NFS volume, where a catalog is too slow."""
    from research.store.cli import COMMANDS

    _, loops = W.load(CONTROL / "watchers.toml")
    (lp,) = [lp for lp in loops if lp.name == "labels-pack"]
    assert lp.argv == ["python3.12", "-s", "-m", "research", "data", "labels-pack", "--every", "3600"]
    assert lp.argv[5] in COMMANDS, "the loop's command is one `research data` has"
    assert lp.env["PYTHONPATH"].split(":")[0] == "/data/watchers/src/tools/research/src", "`-m research` is the deployed tree's"
    assert lp.env_files == ["/root/.config/verity/r2.env"] and not lp.needs
    store = lp.env["RESEARCH_STORE"]
    assert store.startswith("/root/") and store not in {o.env.get("RESEARCH_STORE") for o in loops if o is not lp}, "a store of its own"


def test_a_loop_that_exits_starts_again_after_restart_s_and_every_loop_up_marks_all_up(tmp_path):
    beats = []
    sup, clock, started = supervisor(tmp_path, [W.Loop("a", ["a"]), W.Loop("b", ["b"])], beats)
    sup.step()
    assert started == ["a", "b"] and sup.all_up_t is None
    clock.t += W.UP_S
    sup.step()
    assert sup.all_up_t == clock.t
    sup.loops["a"].proc.rc = 2
    clock.t += 1
    sup.step()
    a = sup.loops["a"]
    assert (a.state, a.last_exit, a.restarts) == ("restarting", 2, 1) and sup.all_up_t == clock.t - 1
    clock.t += W.RESTART_S - 1
    sup.step()
    assert started == ["a", "b"]
    clock.t += 1
    sup.step()
    assert started == ["a", "b", "a"] and a.state == "running"
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["schema"] == W.SCHEMA and status["host"] == "pod" and status["loops"]["a"]["restarts"] == 1
    assert beats and beats[-1]["t"] == status["t"]


def test_a_loop_whose_credential_file_is_missing_waits_and_starts_once_it_is_placed(tmp_path):
    key = tmp_path / "key.env"
    sup, clock, started = supervisor(tmp_path, [W.Loop("needy", ["x"], env_files=[str(key)])])
    sup.step()
    assert started == [] and sup.loops["needy"].state == "waiting" and sup.all_up_t == T0
    assert json.loads((tmp_path / "status.json").read_text())["loops"]["needy"]["waiting_for"] == [str(key)]
    key.write_text("export A='b'\n# c\nnot a pair\n")
    clock.t += W.TICK_S
    sup.step()
    assert started == ["needy"]
    assert W.env_file(key) == {"A": "b"}


def test_a_watcher_imports_only_the_deployed_tree(tmp_path):
    """A hand edit that points agent-watch-relay's PYTHONPATH at another checkout (/data/infra/fixtures-export-repo), so
    the loop ran code no deploy named."""
    cfg = tmp_path / "w.toml"
    head = f'dir = "{tmp_path}"\nsrc = "{tmp_path}/src"\n[env]\nPYTHONPATH = "{{src}}/tools/research/src"\n'
    cfg.write_text(head + '[[loop]]\nname = "a"\nargv = ["true"]\nenv = { PYTHONPATH = "/data/infra/fixtures-export-repo/tools/research/src:{src}/x" }\n')
    with pytest.raises(ValueError, match="loop a: PYTHONPATH /data/infra/fixtures-export-repo/tools/research/src is outside"):
        W.load(cfg)
    key = tmp_path / "k.env"
    key.write_text("A=b\nPYTHONPATH=/elsewhere\n")
    cfg.write_text(head + f'[[loop]]\nname = "a"\nargv = ["true"]\nenv_files = ["{key}"]\n')
    _, (lp,) = W.load(cfg)
    assert W.loop_env(lp)["PYTHONPATH"] == f"{tmp_path}/src/tools/research/src" and W.loop_env(lp)["A"] == "b"


def test_exec_runs_a_command_in_one_loops_environment_and_refuses_before_its_credential_is_placed(tmp_path, capfd):
    key = tmp_path / "key.env"
    cfg = tmp_path / "w.toml"
    cfg.write_text(f'dir = "{tmp_path}"\nsrc = "{tmp_path}/src"\n[env]\nPATH = "/usr/bin:/bin"\n'
                   f'[[loop]]\nname = "a"\nargv = ["true"]\nenv_files = ["{key}"]\nenv = {{ WHERE = "{{dir}}/x" }}\n')
    assert W.main(["exec", "--config", str(cfg), "a", "--", "sh", "-c", "echo $WHERE"]) == 2
    key.write_text("A=b\n")
    assert W.main(["exec", "--config", str(cfg), "a", "--", "sh", "-c", 'echo "$WHERE $A $(pwd)"']) == 0
    assert W.main(["exec", "--config", str(cfg), "nope"]) == 2
    out = capfd.readouterr().out.splitlines()
    assert out[0] == f"a has no {key} yet" and out[1] == f"{tmp_path}/x b {tmp_path}" and out[2].startswith("no loop 'nope'")


def config(tmp_path, name, *loops):
    """A watchers config at tmp_path/name holding `loops`, each the TOML lines of one [[loop]] after its name."""
    path = tmp_path / name
    path.write_text(f'dir = "{tmp_path}"\nsrc = "{tmp_path}/src"\n[env]\nPATH = "/usr/bin:/bin"\n'
                    + "".join(f'[[loop]]\nname = "{n}"\n{body}\n' for n, body in loops))
    return path


def test_check_refuses_a_config_that_lacks_a_loop_the_live_one_runs_and_says_how_to_keep_it(tmp_path, capfd):
    """A loop added to the pod's config by hand and never to the master: copying the master over it stops the loop at the
    supervisor's next start, with nothing said."""
    kept, added = ("kept", 'argv = ["true"]'), ("added-on-the-pod", 'argv = ["sh", "/data/elsewhere/loop.sh"]')
    live, new = config(tmp_path, "live.toml", kept, added), config(tmp_path, "new.toml", kept)
    assert W.main(["check", "--config", str(new), "--replaces", str(live)]) == 1
    out = capfd.readouterr().out
    assert f"replacing {live}:\n    added-on-the-pod: dropped\n        live  sh /data/elsewhere/loop.sh\n" in out
    assert out.splitlines()[-1] == (
        f"refused: {new} lacks added-on-the-pod, which {live} runs: copied over it, the supervisor's next start stops that loop. "
        "To keep a loop, add it to infra/control/watchers.toml in a PR and deploy that commit; to stop one on purpose, pass "
        "--allow-drop added-on-the-pod")
    assert W.main(["check", "--config", str(new), "--replaces", str(live), "--allow-drop", "added-on-the-pod"]) == 0
    out = capfd.readouterr().out
    assert "    added-on-the-pod: dropped (--allow-drop)\n" in out and "refused" not in out
    assert W.main(["check", "--config", str(live), "--replaces", str(live), "--allow-drop", "kept", "--allow-drop", "gone"]) == 0
    assert capfd.readouterr().out.endswith(f"replacing {live}:\n    --allow-drop gone: the live config has no such loop\n"
                                           "    --allow-drop kept: the config keeps that loop\n")
    assert W.main(["check", "--config", str(new), "--replaces", str(tmp_path / "none.toml")]) == 2
    assert f"{tmp_path}/none.toml can't be read as a watchers config (FileNotFoundError" in capfd.readouterr().out
    with pytest.raises(SystemExit) as e:
        W.main(["check", "--config", str(new), "--allow-drop", "kept"])
    assert e.value.code == 2 and "--allow-drop goes with --replaces" in capfd.readouterr().err


def test_check_prints_a_loop_that_is_new_or_differs_from_the_live_one_without_failing_and_no_environment_value(tmp_path, capfd):
    same = ("same", 'argv = ["true"]\nenv_files = ["/k.env"]\nneeds = ["/seed"]\nenv = { A = "1" }')
    live = config(tmp_path, "live.toml", same, ("moved", 'argv = ["true", "--on", "n2"]\nenv_files = ["/a.env"]\n'
                                                         'env = { STORE = "/live-value", GONE = "x" }'))
    new = config(tmp_path, "new.toml", same, ("moved", 'argv = ["true", "--on", "cpu1"]\nenv_files = ["/a.env", "/b.env"]\n'
                                                       'needs = ["/seed"]\nenv = { STORE = "/new-value", MORE = "y" }'),
                 ("brand-new", 'argv = ["true", "new"]'))
    assert W.main(["check", "--config", str(new), "--replaces", str(live)]) == 0
    out = capfd.readouterr().out
    assert out.endswith(f"replacing {live}:\n"
                        "    moved: argv differs\n        live  true --on n2\n        new   true --on cpu1\n"
                        "    moved: env_files differs\n        live  /a.env\n        new   /a.env /b.env\n"
                        "    moved: needs differs\n        live  (none)\n        new   /seed\n"
                        "    moved: env differs in GONE, MORE, STORE\n"
                        "    brand-new: new\n        new   true new\n")
    assert "-value" not in out, "an environment's values are never printed"
    assert W.main(["check", "--config", str(new), "--replaces", str(new)]) == 0
    assert capfd.readouterr().out.endswith(f"replacing {new}:\n    the same loops, none of them changed\n")
    other = tmp_path / "other.toml"
    other.write_text(new.read_text().replace(f'dir = "{tmp_path}"', f'dir = "{tmp_path}"\nlock = "/elsewhere.lock"'))
    assert W.main(["check", "--config", str(new), "--replaces", str(other)]) == 0
    assert capfd.readouterr().out.endswith(f"replacing {other}:\n    settings differ: lock\n")


def test_the_relay_runs_every_relay_side_monitor_but_the_one_that_reads_the_node_side_monitors():
    """The relay runs only what its `--only` lists, so a relay-side monitor added to research.monitors and not to the loop
    never runs. `node-heartbeat` alone stays out: it reads the node-side monitors' last pass on the `--on` host, and no
    machine runs them."""
    from research import monitors

    _, loops = W.load(CONTROL / "watchers.toml")
    (lp,) = [lp for lp in loops if lp.name == "monitors-relay"]
    first = lp.argv.index("--only") + 1
    only = lp.argv[first:next(i for i in range(first, len(lp.argv)) if lp.argv[i].startswith("--"))]
    relay_side = {m.id for m in monitors.MONITORS if m.where == "relay"}
    assert relay_side - set(only) == {"node-heartbeat"} and set(only) <= relay_side, \
        "infra/control/watchers.toml's monitors-relay runs only what its --only lists: list each relay-side monitor there"
    assert len(only) == len(set(only))


def test_an_extra_job_that_exits_0_is_done_and_one_that_fails_starts_again(tmp_path):
    extra = tmp_path / "extra.d"
    extra.mkdir()
    for name in ("ok", "bad", "off"):
        (extra / f"{name}.sh").write_text("#!/bin/sh\n")
        (extra / f"{name}.sh").chmod(0o755 if name != "off" else 0o644)
    sup, clock, started = supervisor(tmp_path, [])
    sup.step()
    assert sorted(started) == ["extra-bad", "extra-ok"]
    sup.loops["extra-ok"].proc.rc, sup.loops["extra-bad"].proc.rc = 0, 1
    clock.t += 1
    sup.step()
    assert sup.loops["extra-ok"].state == "done" and (extra / "ok.done").is_file()
    assert sup.loops["extra-bad"].state == "restarting"
    sup2, _, started2 = supervisor(tmp_path, [])
    sup2.step()
    assert started2 == ["extra-bad"]


def test_the_heartbeat_upsert_quotes_the_body_and_creates_the_table_once():
    p = W.Publisher("postgres://x", "store_private.control_heartbeat", "vy-control-verity")
    s = p.script(b'{"a": "it\'s"}', T0)
    assert s.startswith("CREATE TABLE IF NOT EXISTS store_private.control_heartbeat")
    assert "'{\"a\": \"it''s\"}'" in s and "'2026-10-19T20:00:00Z'" in s and "ON CONFLICT (host)" in s
    p.created = True
    assert not p.script(b"{}", T0).startswith("CREATE")


def vm(rules, deadline=None):
    return {"name": "v", "rules": rules, "id": "i", "host": "h", "user": "research", "deadline": deadline,
            "idle_s": V.IDLE_S, "lease_idle_s": V.LEASE_IDLE_S}


def probe(gpu=0, leases=0, runs=0, lease=None):
    return {"gpu": gpu, "runs": runs, "leases": leases, "lease": lease}


def test_the_probe_line_parses_and_an_unread_one_is_none():
    assert V.parse_probe("gpu=7 runs=1 leases=8 lease=1792500000") == probe(7, 8, 1, 1792500000.0)
    assert V.parse_probe("gpu=0 runs=0 leases=0 lease=") == probe()
    assert V.parse_probe("ssh: connect refused") is None


def test_an_idle_h200_is_stopped_after_idle_s_and_one_restarted_counts_as_busy_from_its_first_reading():
    st, v = {}, vm(["idle", "deadline"])
    assert V.judge(v, st, "RUNNING", probe(), T0)[0] is None
    assert V.judge(v, st, "RUNNING", probe(), T0 + V.IDLE_S - 300)[0] is None
    why, line = V.judge(v, st, "RUNNING", probe(), T0 + V.IDLE_S)
    assert why.startswith("idle 120 min") and "mean2h=0" in line
    assert V.judge(v, st, "STOPPED", None, T0 + V.IDLE_S + 300)[0] is None and "busy" not in st
    assert V.judge(v, st, "RUNNING", probe(), T0 + 3 * V.IDLE_S)[0] is None


def test_leases_held_over_idle_gpus_keep_a_vm_only_for_lease_idle_s_and_a_busy_mean_keeps_it():
    st, v = {}, vm(["idle"])
    V.judge(v, st, "RUNNING", probe(gpu=50, leases=8), T0)
    for k in range(1, 47):
        assert V.judge(v, st, "RUNNING", probe(gpu=1, leases=8), T0 + 300 * k)[0] is None, k
    assert st["busy"] == T0 + V.LEASE_IDLE_S - 300
    assert V.judge(v, st, "RUNNING", probe(gpu=1, leases=8), T0 + 300 * 47)[0].startswith("idle 120 min")
    st2 = {}
    for k in range(60):
        assert V.judge(v, st2, "RUNNING", probe(gpu=10 if k % 2 else 0), T0 + 300 * k)[0] is None


def test_the_deadline_and_lease_backstops_stop_only_a_running_vm():
    dl = T0 + 3600
    v = vm(["deadline"], dl)
    assert V.judge(v, {}, "RUNNING", None, dl - V.DEADLINE_LEAD_S - 1)[0] is None
    assert V.judge(v, {}, "RUNNING", None, dl - V.DEADLINE_LEAD_S)[0] == "deadline 2026-10-19T21:00:00Z"
    assert V.judge(v, {}, "STOPPED", None, dl + 1)[0] is None
    v = vm(["lease"])
    assert V.judge(v, {}, "RUNNING", probe(lease=T0), T0 + V.LEASE_GRACE_S)[0] is None
    assert V.judge(v, {}, "RUNNING", probe(lease=T0), T0 + V.LEASE_GRACE_S + 1)[0] == "lease 2026-10-19T20:00:00Z passed"


def test_an_entry_overrides_the_registry_and_the_checked_in_config_names_known_rules(tmp_path):
    (tmp_path / "a.toml").write_text('host = "1.2.3.4"\nnebius_instance_id = "i-a"\ndeadline = "2026-10-23T21:00:00Z"\n')
    got = V.vms({"vm": [{"name": "a", "rules": ["idle"]}, {"name": "a", "rules": ["deadline"], "id": "i-b",
                                                           "deadline": "2026-10-20T04:00:00Z"}]}, tmp_path)
    assert [(g["id"], g["host"], V.iso(g["deadline"])) for g in got] == [("i-a", "1.2.3.4", "2026-10-23T21:00:00Z"),
                                                                       ("i-b", "1.2.3.4", "2026-10-20T04:00:00Z")]
    for e in tomllib.loads((CONTROL / "vm_stop.toml").read_text())["vm"]:
        assert set(e["rules"]) <= {"idle", "deadline", "lease"} and e["rules"], e["name"]
