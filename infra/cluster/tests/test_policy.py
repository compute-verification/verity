"""`cluster policy`: the description's policy keys and their owners, and `diff` against node outputs in the shapes the live
nodes return (3 Oct 2026): drift, undeclared and unknown keys, the exit codes, and that the commands only read, and read only
the lease pools' own variables of a process's environment."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from cluster import cli, ledger, policy
from cluster.ledger import epoch

NEBIUS = Path(__file__).parents[1] / "descriptions" / "nebius.toml"
N1, N2 = "vy-nebius-1", "vy-nebius-2"
T = epoch("2026-10-03T19:40:30Z")
QUIET, DISK = policy.HOLDERS["vy-quiet-hour"], policy.HOLDERS["vy-disk-guard"]

LEASE = """session n1-lease
loop n1-lease 812
VY_POOL_MAX=8
VY_POOL_BORROW=4
VY_POOL_BORROW_UNTIL=2026-10-04T10:30:00Z
session n1-lease-circuits
loop n1-lease-circuits 913
VY_POOL_NAME=circuits
GPU_LEASE_DIR=/run/gpu-lease-circuits
VY_POOL_QUEUE=deployments-gpu
VY_POOL_LOCAL_QUEUE=deployments-gpu-pool
VY_POOL_PRIORITY=circuits-gpu
VY_POOL_MAX=6
VY_POOL_IDLE_S=30
"""
FILES = {policy.LEAN_SLOTS: "# NAME CAP\ncheck 2\naudit 2\n",
         policy.CHECK_SLOTS: "c 8-31\na 32-63 only=trains\nb 64-95\nd 128-159  # since 2 Oct\n",
         "/etc/vy/direct-gpus": "none\n", "/etc/vy/direct-cpus": "# the non-benchmark CPUs\n0-95,128-159\n",
         policy.USER_CPUSET: "# delegate cpuset\n[Service]\nDelegate=cpu cpuset memory pids\n",
         policy.DISPATCH_ENV: "VY_PROVER_CPUS=160-191\n"}
STATUS = ("2026-10-03T19:39:59Z free GPUs 7/8, kept free -, waiters -, timed False, window waiting False, backfill False, "
          "queued 0 (gpu 0, cpu 0), done 1033, failed 30, held for an unknown exit 0")
N2_FILES = {f"{policy.FILL}/windows": "# start UTC, minutes\n2026-10-03T18:25Z 10 # merge checks off\n",
            f"{policy.FILL}/cpu-sets": "bc-e6a46970-b6ef-5738-af64-182b143075d4 0-47 4 40  # verifies\n",
            policy.CHECK_SLOTS: "d 0-47 windows=/workspace/pouw/fill/windows lead=30 only=trains\ne 48-79 windows=/workspace/pouw/fill/windows lead=30\n",
            "/etc/vy/direct-gpus": None, "/etc/vy/direct-cpus": None,
            policy.USER_CPUSET: "[Service]\nDelegate=cpu cpuset memory pids\n"}


def dump(files: dict[str, str | None]) -> str:
    return "".join(f"{policy.ABSENT_MARK}{p}\n" if t is None else f"{policy.FILE_MARK}{p}\n{t}\n" for p, t in files.items())


def cq(name: str, stop: str | None = None, **res: dict[str, str]) -> dict:
    """A ClusterQueue as `kubectl get clusterqueues -o json` lists it, one flavor holding each resource."""
    rs = [{"name": policy.KUEUE_RESOURCES[r], **{policy.KUEUE_FIELDS[f]: q for f, q in fields.items()}} for r, fields in res.items()]
    return {"metadata": {"name": name}, "spec": {**({"stopPolicy": stop} if stop else {}), "resourceGroups": [
        {"coveredResources": [r["name"] for r in rs], "flavors": [{"name": "default", "resources": rs}]}]}}


def queues(**changes: str) -> dict[str, dict[str, dict[str, str]]]:
    """The description's ClusterQueues as kubectl spells them (memory in bytes where the description says Gi), with
    `changes` (`provers__gpu__nominal="6"`) applied."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for k in policy.load(NEBIUS).values():
        p = k.key.split(".")
        if p[:2] == [N1, "kueue"] and len(p) == 5:
            v = str(k.value)
            out.setdefault(p[2], {}).setdefault(p[3], {})[p[4]] = str(int(v[:-2]) * 2**30) if v.endswith("Gi") else v
    for path, v in changes.items():
        q, r, f = path.split("__")
        out.setdefault(q, {}).setdefault(r, {})[f] = v
    return out


def kueue(qs: dict | None = None, stops: dict[str, str] | None = None) -> str:
    qs = queues() if qs is None else qs
    return json.dumps({"items": [cq(q, (stops or {}).get(q), **res) for q, res in qs.items()]})


def outputs(**over) -> dict[str, str | Exception]:
    out = {"lease": LEASE, "kueue": kueue(), "holds": dump({QUIET: None, DISK: None}), "files": dump(FILES),
           "overrides": dump({policy.OVERRIDES: None}), "n2": f"status {STATUS}\n" + dump(N2_FILES)}
    out.update(over)
    return out


def fake(outs: dict[str, str | Exception], ran: list | None = None):
    by_cmd = {policy.COMMANDS[name]: out for name, out in outs.items()}

    def for_node(node: str):
        def run(cmd: str) -> str:
            if ran is not None:
                ran.append((node, cmd))
            out = by_cmd[cmd]
            if isinstance(out, Exception):
                raise out
            return out
        return run
    return for_node


@pytest.fixture(autouse=True)
def _clock(monkeypatch):
    monkeypatch.setattr(policy, "clock", lambda: T)


def diff(capsys, which: str = "all", description: Path = NEBIUS, ran: list | None = None, **over) -> tuple[int, dict]:
    rc = policy.main_diff(description, which, True, runner=fake(outputs(**over), ran))
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "cluster/policy-diff/v1" and out["t"] == policy.clock()
    return rc, {r["key"]: r for rep in out["nodes"] for r in rep["keys"]}


def described(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "d.toml"
    p.write_text(NEBIUS.read_text().split("\n[policy.")[0] + "\n" + text)
    return p


def test_the_nebius_policy_loads_with_an_owner_and_a_source_per_key():
    keys = policy.load(NEBIUS)
    assert keys[f"{N1}.lease-pool.provers.max"].value == 8 and keys[f"{N1}.lease-pool.provers.borrow"].value == 4
    assert keys[f"{N1}.kueue.provers.memory.nominal"].value == "384Gi"
    assert keys[f"{N1}.pools.checks"].value == "8-95,128-159" and keys[f"{N1}.pools.provers"].value == "160-191"
    assert {k.owner for k in keys.values()} == {"infra"}
    assert all(policy.source(k) for k in keys)
    assert policy.source(f"{N1}.lease-pool.provers.max") == \
        "tmux n1-lease: n1_lease.py loop's VY_POOL_MAX, or its n1-lease.limits under VY_POOL_POLICY"
    assert policy.source(f"{N1}.lease-pool.provers.idle-s") == "tmux n1-lease: n1_lease.py loop's VY_POOL_IDLE_S"
    assert policy.source(f"{N2}.fill.cpu-sets.x") == "/workspace/pouw/fill/cpu-sets"


def test_an_owner_covers_the_keys_below_it_unless_a_table_names_its_own(tmp_path):
    keys = policy.load(described(tmp_path, '[policy.vy-nebius-1]\nowner = "infra"\ndirect-gpus = "none"\n'
                                           'lean-slots = { owner = "proofs", audit = 2 }\n'
                                           '[policy.vy-nebius-1.pools]\nowner = "research"\n'))
    assert {k: v.owner for k, v in keys.items()} == {
        f"{N1}.direct-gpus": "infra", f"{N1}.lean-slots.audit": "proofs", f"{N1}.pools.checks": "research",
        f"{N1}.pools.provers": "research"}


N1_TABLE = '[policy.vy-nebius-1]\nowner = "infra"\n'


@pytest.mark.parametrize("text, why", [
    ('[policy.vy-nebius-1]\ndirect-gpus = "none"\n', f"{N1}.direct-gpus: no owner"),
    ('[policy.vy-nebius-9]\nowner = "infra"\ndirect-gpus = "none"\n', "unknown node 'vy-nebius-9'"),
    ("[policy]\nvy-nebius-1 = 3\n", "policy.vy-nebius-1 is not a table"),
    (N1_TABLE + 'fill = { cpu-sets = { x = "0-3 1" } }\n', "no reader for 'fill' on vy-nebius-1"),
    (N1_TABLE + 'direct-cpus = ["0-3"]\n', "a policy value is a string or a number"),
    (N1_TABLE + 'check-slots = { d = "0-47x" }\n', "bad CPU list item"),
    (N1_TABLE + 'lease-pool = { provers = { borrow-until = "2026-10-04T10:30Z" } }\n',
     "n1_lease.py reads only %Y-%m-%dT%H:%M:%SZ (else it never stops borrowing)"),
    (N1_TABLE + "lease-pool = { builders = { max = 2 } }\n", "a lease pool key is lease-pool.POOL.FIELD"),
    (N1_TABLE + "lease-pool = { provers = { max = inf } }\n", f"{N1}.lease-pool.provers.max: 'inf' is not a whole number"),
    (N1_TABLE + "lease-pool = { provers = { max = -1 } }\n", "'-1' is not a whole number"),
    (N1_TABLE + "lease-pool = { provers = { max = 2.5 } }\n", "'2.5' is not a whole number"),
    (N1_TABLE + "lease-pool = { provers = { borrow = 9 } }\n", f"{N1}.lease-pool.provers.borrow: 9 is more GPUs than {N1} has (8)"),
    (N1_TABLE + "kueue = { provers = { gpu = { quota = 3 } } }\n", "a Kueue key is kueue.QUEUE.stop-policy"),
    (N1_TABLE + 'kueue = { provers = { memory = { nominal = "lots" } } }\n', "kueue.provers.memory.nominal"),
    (N1_TABLE + "kueue = { provers = { gpu = { nominal = 3 } } }\n",
     f"{N1}.kueue.provers.gpu.nominal = 3, but workstream provers says 2"),
    (N1_TABLE + 'pools = { checks = "8-95" }\n', "is the node's own `pools`"),
    (N1_TABLE + 'user-cpuset = { allowed = "0-3" }\n', "the cpuset drop-in's only key is user-cpuset.delegate"),
    (N1_TABLE + '[policy.vy-nebius-2.fill.window]\nowner = "infra"\n"2026-10-04T06:30:00Z" = 30\n',
     "write the start as 2026-10-04T06:30Z"),
    (N1_TABLE + '[policy.vy-nebius-2.fill.window]\nowner = "infra"\n"tomorrow" = 30\n', "is not a UTC start"),
    (N1_TABLE + '[policy.vy-nebius-2.fill.window]\nowner = "infra"\n"2026-10-04T06:30Z" = 0\n', "is not a number of minutes"),
])
def test_a_description_is_refused_naming_each_problem(tmp_path, capsys, text, why):
    p = described(tmp_path, text)
    with pytest.raises(policy.PolicyError, match=re.escape(why)):
        policy.load(p)
    assert policy.main_show(p, "all", False) == 2 and policy.main_diff(p, "all", False, runner=fake({})) == 2
    assert why in capsys.readouterr().out


def test_values_compare_in_one_spelling_per_surface():
    c = policy.canon
    assert c(f"{N1}.kueue.provers.memory.nominal", "384Gi") == c(f"{N1}.kueue.provers.memory.nominal", "412316860416")
    assert c(f"{N1}.kueue.provers.cpu.nominal", "500m") == c(f"{N1}.kueue.provers.cpu.nominal", "0.5")
    assert c(f"{N1}.kueue.provers.cpu.nominal", "64") != c(f"{N1}.kueue.provers.cpu.nominal", "64k")
    assert c(f"{N1}.pools.checks", "128-159,8-95") == c(f"{N1}.direct-cpus", "8-95,128-159") == "8-95,128-159"
    assert c(f"{N2}.check-slots.d", "0-47 lead=30 windows=/w") == c(f"{N2}.check-slots.d", " 0-15,16-47  windows=/w lead=30")
    assert c(f"{N2}.fill.window.2026-10-04T06:30Z", "190.0") == c(f"{N2}.fill.window.2026-10-04T06:30Z", 190) == "190"
    assert c(f"{N2}.fill.cpu-sets.x", "47,0-46 4 40") == "0-47 4 40"
    assert c(f"{N1}.user-cpuset.delegate", "cpu  cpuset\tmemory pids") == "cpu cpuset memory pids"
    with pytest.raises(ValueError, match="not a whole number"):
        c(f"{N1}.lean-slots.audit", "two")


def test_live_as_declared_is_exit_0(capsys):
    rc, rows = diff(capsys)
    assert rc == 0 and {r["status"] for r in rows.values()} == {"ok"}
    assert set(rows) == set(policy.load(NEBIUS))
    assert rows[f"{N1}.kueue.provers.memory.nominal"]["live"] == "412316860416"


def test_drift_absent_and_undeclared_keys_are_exit_1(capsys):
    lease = LEASE.replace("VY_POOL_MAX=8\n", "VY_POOL_MAX=6\nVY_POOL_IDLE_S=60\n").replace("VY_POOL_IDLE_S=30\n", "")
    files = {**FILES, policy.LEAN_SLOTS: "check 2\naudit 3\nbench 1\n"}
    rc, rows = diff(capsys, "nebius-1", lease=lease, files=dump(files),
                    kueue=kueue(queues(provers__gpu__nominal="6", **{"deployments-gpu__gpu__nominal": "2"})))
    assert rc == 1
    found = {k: (r["status"], r["declared"], r["live"], r["why"]) for k, r in rows.items() if r["status"] != "ok"}
    assert found == {
        f"{N1}.lease-pool.provers.max": ("drift", 8, "6", "live differs"),
        f"{N1}.lease-pool.provers.idle-s": ("undeclared", None, "60", "live, and not in the description"),
        f"{N1}.lease-pool.circuits.idle-s": ("drift", 30, None, "absent live"),
        f"{N1}.lean-slots.audit": ("drift", 2, "3", "live differs"),
        f"{N1}.lean-slots.bench": ("undeclared", None, "1", "live, and not in the description"),
        f"{N1}.kueue.provers.gpu.nominal": ("drift", 2, "6", "live differs"),
        f"{N1}.kueue.deployments-gpu.gpu.nominal": ("drift", 6, "2", "live differs"),
    }
    assert rows[f"{N1}.lean-slots.bench"]["owner"] == "infra"


def test_text_names_each_key_that_differs_with_its_owner_and_source(capsys):
    rc = policy.main_diff(NEBIUS, "nebius-1", False, runner=fake(outputs(kueue=kueue(queues(provers__gpu__nominal="6")))))
    out = capsys.readouterr().out.splitlines()
    assert rc == 1 and out[0] == f"{N1}: 49 as declared, 1 drift, 0 undeclared, 0 unknown"
    assert out[1].split() == ["drift", f"{N1}.kueue.provers.gpu.nominal", "declared", "2", "live", "6", "(live", "differs;",
                              "owner", "infra;", "ClusterQueue", "provers", "nvidia.com/gpu", "nominalQuota)"]


def test_a_live_value_that_doesnt_parse_is_drift(capsys):
    rc, rows = diff(capsys, "nebius-1", files=dump({**FILES, policy.LEAN_SLOTS: "check 2\naudit lots\n"}))
    assert rc == 1 and rows[f"{N1}.lean-slots.audit"]["status"] == "drift"
    assert "the live value doesn't parse" in rows[f"{N1}.lean-slots.audit"]["why"]


def test_a_hold_by_a_declared_writer_is_as_declared_and_any_other_is_drift(capsys):
    held = kueue(stops={"provers": "Hold", "backfill": "HoldAndDrain"})
    rc, rows = diff(capsys, "nebius-1", kueue=held, holds=dump({QUIET: "provers\ndeployments-gpu\n", DISK: None}))
    assert rc == 1
    assert (rows[f"{N1}.kueue.provers.stop-policy"]["status"], rows[f"{N1}.kueue.provers.stop-policy"]["why"]) == \
        ("ok", "held by vy-quiet-hour")
    assert (rows[f"{N1}.kueue.backfill.stop-policy"]["status"], rows[f"{N1}.kueue.backfill.stop-policy"]["why"]) == \
        ("drift", "held by no declared writer")
    rc, rows = diff(capsys, "nebius-1", kueue=held, holds=OSError("Permission denied"))
    assert rc == 3 and rows[f"{N1}.kueue.provers.stop-policy"]["status"] == "unknown"
    assert "the holders' state files can't be read" in rows[f"{N1}.kueue.provers.stop-policy"]["why"]


@pytest.mark.parametrize("lease, why", [
    (LEASE.replace("session n1-lease\n", ""), "no tmux session n1-lease"),
    (LEASE.replace("loop n1-lease 812\n", "unreadable n1-lease 812\n"), "the environment of tmux session n1-lease's"),
    (LEASE.replace("loop n1-lease 812\n", ""), "no n1_lease.py loop in tmux session n1-lease"),
    (LEASE.replace("session n1-lease-circuits\n", "loop n1-lease 814\nVY_POOL_MAX=6\nsession n1-lease-circuits\n"),
     "2 n1_lease.py loops in tmux session n1-lease, with different settings"),
    (RuntimeError("ssh: connect to host 81.85.2.165 port 22: timed out"), "the lease pools' tmux panes couldn't be read"),
])
def test_a_lease_pool_that_cant_be_read_is_unknown_never_no_drift(capsys, lease, why):
    rc, rows = diff(capsys, "nebius-1", lease=lease)
    assert rc == 3
    max_ = rows[f"{N1}.lease-pool.provers.max"]
    assert max_["status"] == "unknown" and why in max_["why"] and max_["live"] is None


def test_two_loops_with_one_setting_are_one_pool(capsys):
    twice = LEASE.replace("session n1-lease-circuits\n",
                          "loop n1-lease 814\nVY_POOL_MAX=8\nVY_POOL_BORROW=4\nVY_POOL_BORROW_UNTIL=2026-10-04T10:30:00Z\n"
                          "session n1-lease-circuits\n")
    assert diff(capsys, "nebius-1", lease=twice)[0] == 0


def _under_policy(limits: str) -> str:
    return LEASE.replace("VY_POOL_BORROW_UNTIL=2026-10-04T10:30:00Z\n", "VY_POOL_BORROW_UNTIL=2026-10-04T10:30:00Z\n"
                         "VY_POOL_POLICY=cd /workspace/verity-guest/cluster-src && python3 -m cluster policy show --json\n"
                         f"{limits}\n").replace("VY_POOL_MAX=8\n", "VY_POOL_MAX=6\n", 1)


@pytest.mark.parametrize("limits, rc, status, why", [
    ("limits max=8 borrow=4 borrow-until=2026-10-04T10:30:00Z from=policy", 0, "ok", None),
    ("limits max=6 borrow=4 borrow-until=2026-10-04T10:30:00Z from=environment", 1, "drift",
     "n1_lease.py can't read $VY_POOL_POLICY, so its environment's values are in force"),
    ("limits max=7 borrow=4 borrow-until=2026-10-04T10:30:00Z from=last-policy", 1, "drift",
     "n1_lease.py can't read $VY_POOL_POLICY, so the values it last read there are in force"),
    ("limits max=8 borrow=4 borrow-until=2026-10-04T10:30:00Z from=environment", 1, "drift",    # as declared, and still drift
     "n1_lease.py can't read $VY_POOL_POLICY, so its environment's values are in force"),
    ("limits", 3, "unknown", "n1_lease.py runs under $VY_POOL_POLICY and its n1-lease.limits can't be read"),
    ("limits max=8 borrow=4", 3, "unknown",
     "n1_lease.py runs under $VY_POOL_POLICY and its n1-lease.limits lacks borrow-until, from"),
])
def test_a_loop_under_the_policy_is_read_from_its_limits_file_not_its_environment(capsys, limits, rc, status, why):
    got, rows = diff(capsys, "nebius-1", lease=_under_policy(limits))
    assert got == rc and not any("VY_POOL" in k for k in rows)
    assert (rows[f"{N1}.lease-pool.provers.max"]["status"], rows[f"{N1}.lease-pool.provers.max"].get("why")) == (status, why)
    assert rows[f"{N1}.lease-pool.circuits.max"]["status"] == "ok"
    got, rows = diff(capsys, "nebius-1", lease=_under_policy("limits max=8 borrow=4 borrow-until= from=policy"))
    assert got == 1 and rows[f"{N1}.lease-pool.provers.borrow-until"]["why"] == "absent live"


def _two_flavors() -> str:
    d = json.loads(kueue())
    next(q for q in d["items"] if q["metadata"]["name"] == "provers")["spec"]["resourceGroups"].append(
        {"flavors": [{"name": "spare", "resources": [{"name": "nvidia.com/gpu", "nominalQuota": "1"}]}]})
    return json.dumps(d)


@pytest.mark.parametrize("over, prefix", [
    ({"kueue": RuntimeError("sudo: a password is required")}, f"{N1}.kueue"),
    ({"files": RuntimeError("cat: /etc/vy/direct-cpus: Permission denied")}, f"{N1}.lean-slots"),
    ({"kueue": "<html>not json"}, N1),
    ({"kueue": _two_flavors()}, f"{N1}.kueue.provers.gpu"),
])
def test_an_unread_command_makes_its_keys_unknown_and_exit_3(capsys, over, prefix):
    rc, rows = diff(capsys, "nebius-1", **over)
    assert rc == 3
    mine = [r for k, r in rows.items() if k == prefix or k.startswith(prefix + ".")]
    assert mine and {r["status"] for r in mine} == {"unknown"}


@pytest.mark.parametrize("kueue, error", [
    ({"items": [{"metadata": {"name": "provers"}, "spec": "x"}]}, "AttributeError"),
    ({"items": [{"metadata": {"name": "provers"}, "spec": {"resourceGroups": [{"flavors": [{"resources": [7]}]}]}}]},
     "TypeError"),
])
def test_live_output_of_an_unexpected_shape_makes_the_node_unknown_and_exit_3_never_drift(capsys, kueue, error):
    rc, rows = diff(capsys, "nebius-1", kueue=json.dumps(kueue))
    assert rc == 3 and rows and {r["status"] for r in rows.values()} == {"unknown"}
    assert all(r["why"].startswith(f"its live policy couldn't be parsed: {error}") for r in rows.values())


def test_node_2_refusing_to_be_read_runs_one_command_and_every_key_is_unknown(capsys):
    ran: list = []
    refused = ("refused a timed window is on or waiting: 2026-10-03T19:20:01Z free GPUs 0/8, kept free -, waiters -, "
               "timed True, window waiting False\n")
    rc, rows = diff(capsys, "nebius-2", ran=ran, n2=refused)
    assert rc == 3 and ran == [(N2, policy.COMMANDS["n2"])]
    assert set(rows) == {k for k in policy.load(NEBIUS) if k.startswith(N2 + ".")}
    assert {r["status"] for r in rows.values()} == {"unknown"}
    assert "node 2 refused to be read, and nothing else there was: a timed window is on or waiting: 2026-10-03T19:20:01Z" \
        in rows[f"{N2}.user-cpuset.delegate"]["why"]


@pytest.mark.parametrize("first", [f"timed {STATUS}", STATUS, ""])
def test_node_2_output_that_is_neither_a_status_nor_a_refusal_is_unknown(capsys, first):
    rc, rows = diff(capsys, "nebius-2", n2=first + "\n" + dump(N2_FILES))
    assert rc == 3 and all(r["status"] == "unknown" and "neither a status nor a refusal" in r["why"] for r in rows.values())


def test_node_2_windows_that_ended_are_left_out_and_a_new_booking_is_undeclared(tmp_path, capsys):
    p = described(tmp_path, N1_TABLE + '[policy.vy-nebius-2]\nowner = "infra"\n[policy.vy-nebius-2.fill.window]\n'
                                       '"2026-10-03T18:25Z" = 10\n"2026-10-04T06:30Z" = 190\n')
    windows = ("2026-10-03T18:25Z 10\n2026-10-04T06:30:00Z 190.0 # circuits\n2026-10-04T10:30 30 # served\n"
               "garbage 5 # fill_runner.py skips it\n")
    rc, rows = diff(capsys, "nebius-2", description=p, n2=f"status {STATUS}\n" + dump({f"{policy.FILL}/windows": windows}))
    assert rc == 1
    assert {k: r["status"] for k, r in rows.items() if ".fill." in k} == {
        f"{N2}.fill.window.2026-10-04T06:30Z": "ok", f"{N2}.fill.window.2026-10-04T10:30Z": "undeclared"}
    shown = {r["key"]: r.get("ended", False) for r in policy.show(policy.load(p), [N2], T)}
    assert shown == {f"{N2}.fill.window.2026-10-03T18:25Z": True, f"{N2}.fill.window.2026-10-04T06:30Z": False}


def test_show_prints_each_key_with_its_owner_and_source(capsys):
    assert policy.main_show(NEBIUS, "nebius-2", True, log=None) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "cluster/policy/v1" and {r["key"].split(".")[0] for r in out["keys"]} == {N2}
    assert out["overrides"] is None
    assert {"key": f"{N2}.user-cpuset.delegate", "value": "cpu cpuset memory pids", "owner": "infra",
            "source": f"{policy.USER_CPUSET} Delegate="} in out["keys"]
    assert policy.main_show(NEBIUS, "all", False, log=None) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["KEY", "VALUE", "OWNER", "SOURCE"] and len(lines) == 1 + len(policy.load(NEBIUS))


def test_show_key_prints_one_key_or_those_under_it_and_refuses_a_key_naming_none(capsys):
    assert policy.main_show(NEBIUS, "all", True, f"{N1}.lease-pool.provers", log=None) == 0
    rows = json.loads(capsys.readouterr().out)["keys"]
    assert {r["key"]: r["value"] for r in rows} == {f"{N1}.lease-pool.provers.max": 8, f"{N1}.lease-pool.provers.borrow": 4,
                                                    f"{N1}.lease-pool.provers.borrow-until": "2026-10-04T10:30:00Z"}
    assert policy.main_show(NEBIUS, "all", True, f"{N1}.lease-pool.prover", log=None) == 2
    assert "no key vy-nebius-1.lease-pool.prover, nor any under it" in capsys.readouterr().out


def test_the_cli_reads_the_default_description_and_refuses_a_bad_one(capsys, tmp_path):
    assert cli.main(["policy", "show", "--node", "nebius-1", "--json", "--log", str(tmp_path / "none.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out)["description"] == str(NEBIUS)
    bad = described(tmp_path, '[policy.vy-nebius-1]\ndirect-gpus = "none"\n')
    assert cli.main(["policy", "diff", "--description", str(bad)]) == 2
    with pytest.raises(SystemExit) as e:
        cli.main(["policy", "diff", "--node", "nebius-3"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        cli.main(["policy", "override", f"{N1}.lease-pool.provers.max", "6", "--until", "2026-10-04T10:30:00Z"])
    assert e.value.code == 2


GPU = f"{N1}.kueue.provers.gpu.nominal"
UNTIL = "2026-10-04T10:30:00Z"


def override(capsys, log: Path, key: str = GPU, value: str = "6", until: str = UNTIL, by: str = "infra",
             reason: str = "the 15:07Z shift", apply: bool = True) -> tuple[int, str]:
    rc = cli.main(["policy", "override", key, value, "--until", until, "--by", by, "--reason", reason, "--log", str(log),
                   *(["--apply"] if apply else [])])
    return rc, capsys.readouterr().out


def test_an_override_is_a_dry_run_until_applied_and_show_merges_it_with_who_why_and_until(capsys, tmp_path):
    log = tmp_path / "policy" / "overrides.jsonl"
    rc, out = override(capsys, log, apply=False)
    assert rc == 0 and not log.exists()
    assert out.startswith(f"would override {GPU}: 2 -> 6 until {UNTIL}, by infra: the 15:07Z shift (in {log}; --apply")
    rc, out = override(capsys, log)
    assert rc == 0 and out.startswith(f"overrode {GPU}: 2 -> 6 until {UNTIL}")
    [r] = ledger.read(log)
    assert ledger.verify([r]) == [] and (r["kind"], r["t"], r["key"], r["value"], r["until"], r["by"]) == \
        ("override", T, GPU, 6, UNTIL, "infra")
    assert policy.main_show(NEBIUS, "nebius-1", True, GPU, log) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["overrides"] == str(log) and out["keys"] == [
        {"key": GPU, "value": 6, "owner": "infra", "source": "ClusterQueue provers nvidia.com/gpu nominalQuota",
         "override": {"by": "infra", "reason": "the 15:07Z shift", "until": UNTIL, "was": 2}}]
    assert policy.main_show(NEBIUS, "nebius-1", False, GPU, log) == 0
    assert f"(override by infra until {UNTIL}, was 2: the 15:07Z shift)" in capsys.readouterr().out


@pytest.mark.parametrize("args, why", [
    ({"until": "2026-10-04T10:30Z"}, "an absolute UTC time, %Y-%m-%dT%H:%M:%SZ"),
    ({"until": "2026-10-03T19:40:30Z"}, "--until 2026-10-03T19:40:30Z has passed"),
    ({"by": " "}, "--by: the handle who decided it"),
    ({"reason": ""}, "--reason: why"),
    ({"key": f"{N1}.pools.provers", "value": "128-191"}, "change them there"),
    ({"key": f"{N1}.lease-pool.builders.max"}, "a lease pool key is lease-pool.POOL.FIELD"),
    ({"key": f"{N2}.kueue.provers.gpu.nominal"}, "no reader for 'kueue' on vy-nebius-2"),
    ({"key": f"{N1}.lease-pool.provers.borrow-until", "value": "tomorrow"}, "n1_lease.py reads only"),
    ({"key": f"{N1}.lease-pool.provers.max", "value": "-1"}, "'-1' is not a whole number"),
    ({"key": f"{N1}.lease-pool.provers.borrow", "value": "-4"}, "'-4' is not a whole number"),
    ({"key": f"{N1}.lease-pool.provers.max", "value": "1.5"}, "'1.5' is not a whole number"),
    ({"key": f"{N1}.lease-pool.provers.max", "value": "inf"}, "'inf' is not a whole number"),
    ({"key": f"{N1}.lease-pool.provers.max", "value": "8x"}, "'8x' is not a whole number"),
    ({"key": f"{N1}.lease-pool.provers.max", "value": "9"}, "9 is more GPUs than vy-nebius-1 has (8)"),
    ({"key": f"{N1}.check-slots.d", "value": "128-15x"}, "bad CPU list item"),
])
def test_an_override_is_refused_naming_each_problem(capsys, tmp_path, args, why):
    log = tmp_path / "o.jsonl"
    rc, out = override(capsys, log, **args)
    assert rc == 2 and why in out and not log.exists()


def test_an_override_may_add_a_key_and_a_later_one_replaces_it_until_its_own_until(capsys, tmp_path):
    log = tmp_path / "o.jsonl"
    window = f"{N2}.fill.window.2026-10-04T06:30Z"
    assert override(capsys, log, key=window, value="190", until="2026-10-04T09:40:00Z", reason="circuits' 235B")[0] == 0
    assert override(capsys, log, key=window, value="150", until="2026-10-03T20:00:00Z", reason="shortened")[0] == 0
    keys = policy.load(NEBIUS)
    eff = policy.effective(keys, policy.overrides(log), T)
    assert (eff[window].value, eff[window].owner, eff[window].override["was"]) == (150, "infra", None)
    assert window not in policy.effective(keys, policy.overrides(log), epoch("2026-10-03T20:00:00Z"))
    assert policy.effective(keys, policy.overrides(log), T, N1) == keys


def test_a_log_that_doesnt_verify_is_refused_never_read_as_no_override(capsys, tmp_path):
    log = tmp_path / "o.jsonl"
    assert override(capsys, log)[0] == 0
    log.write_text(log.read_text().replace('"value":6', '"value":8'))
    assert policy.main_show(NEBIUS, "all", False, log=log) == 3
    assert "hash does not match its content" in capsys.readouterr().out
    rc, out = override(capsys, log)
    assert rc == 3 and "the log doesn't verify" in out
    rc, rows = diff(capsys, "nebius-1", overrides=dump({policy.OVERRIDES: log.read_text()}))
    assert rc == 3 and rows[GPU]["status"] == "unknown" and "its override log doesn't verify" in rows[GPU]["why"]
    rc, rows = diff(capsys, "nebius-1", overrides=RuntimeError("cat: Permission denied"))
    assert rc == 3 and {r["status"] for r in rows.values()} == {"unknown"}


def test_diff_compares_live_with_the_override_in_force_on_each_node(capsys, tmp_path, monkeypatch):
    n1, n2 = tmp_path / "n1.jsonl", tmp_path / "n2.jsonl"
    assert override(capsys, n1)[0] == 0
    assert override(capsys, n1, key=f"{N1}.kueue.deployments-gpu.gpu.nominal", value="2")[0] == 0
    assert override(capsys, n1, key=f"{N1}.lean-slots.audit", value="3", until="2026-10-03T19:40:31Z")[0] == 0
    assert override(capsys, n2, key=f"{N2}.fill.window.2026-10-04T06:30Z", value="190", until="2026-10-04T09:40:00Z")[0] == 0
    assert override(capsys, n2, key=f"{N1}.lean-slots.check", value="1")[0] == 0   # node 1's key in node 2's log: no effect
    monkeypatch.setattr(policy, "clock", lambda: T + 10)                          # lean-slots.audit's has ended
    shifted = kueue(queues(provers__gpu__nominal="6", **{"deployments-gpu__gpu__nominal": "2"}))
    n2_out = f"status {STATUS}\n" + dump({**N2_FILES, policy.OVERRIDES: n2.read_text()})
    rc, rows = diff(capsys, kueue=shifted, overrides=dump({policy.OVERRIDES: n1.read_text()}), n2=n2_out)
    window = f"{N2}.fill.window.2026-10-04T06:30Z"
    assert rc == 1
    assert {k: r["status"] for k, r in rows.items() if r["status"] != "ok" or "override" in r} == {
        GPU: "ok", f"{N1}.kueue.deployments-gpu.gpu.nominal": "ok", window: "drift"}
    assert rows[GPU]["declared"] == 6 and rows[GPU]["override"]["was"] == 2
    assert rows[f"{N1}.lean-slots.audit"]["declared"] == 2 and rows[f"{N1}.lean-slots.check"]["declared"] == 2
    assert rows[window]["why"] == "absent live"
    policy.main_diff(NEBIUS, "nebius-1", False, runner=fake(outputs(kueue=shifted, overrides=dump({policy.OVERRIDES: n1.read_text()}))))
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"{N1}: 50 as declared, 0 drift, 0 undeclared, 0 unknown; 2 under an override"
    assert f"  ok         {GPU}  declared 6  live 6  (override by infra until {UNTIL}, was 2: the 15:07Z shift; owner infra;" \
        in "\n".join(out)


@pytest.mark.parametrize("name", sorted(policy.COMMANDS))
def test_every_command_only_reads(name):
    cmd = policy.COMMANDS[name]
    assert ">" not in cmd.replace("2>/dev/null", "")
    words = set(re.findall(r"[a-z][a-z-]*", cmd))
    assert not words & {"rm", "mv", "cp", "dd", "tee", "touch", "mkdir", "chmod", "chown", "sed", "kill", "pkill", "systemctl",
                        "apply", "patch", "edit", "delete", "label", "annotate", "replace", "scale", "send-keys", "nvidia-smi",
                        "du", "find", "install"}
    if "kubectl" in cmd:
        assert cmd.split("kubectl", 1)[1].split()[0] == "get"


def test_only_the_lease_pools_variables_match():
    env_re = re.compile(policy._ENV_RE)
    assert all(env_re.match(f"{v}=x") for v in policy.ENV.values())
    for name in ("HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "VY_POOLX", "XVY_POOL_MAX", "GPU_LEASE_DIRS", "SLACK_BOT_TOKEN"):
        assert not env_re.match(f"{name}=x")


def _sh(cmd: str, root: Path, path: str = "") -> str:
    """Run one of `COMMANDS` with its absolute paths under `root`."""
    for top in ("/workspace/", "/etc/", "/var/", "/proc/"):
        cmd = cmd.replace(top, f"{root}{top}")
    env = {**os.environ, "PATH": f"{path}:{os.environ['PATH']}" if path else os.environ["PATH"]}
    return subprocess.run(["sh", "-c", cmd], capture_output=True, text=True, check=True, env=env).stdout.replace(str(root), "")


def _put(root: Path, files: dict[str, str]) -> None:
    for p, text in files.items():
        f = root / p.lstrip("/")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)


def test_the_lease_command_reads_only_the_loop_process_and_its_pool_variables(tmp_path):
    bin_ = tmp_path / "bin"
    _put(tmp_path, {"bin/tmux": "#!/bin/sh\nprintf 'n1-lease 100\\nn1-other 300\\nn1-lease-circuits 200\\n'\n",
                    "bin/ps": '#!/bin/sh\n[ "$2" = 100 ] && echo " 101"\nexit 0\n',
                    "/proc/100/cmdline": "bash\0-c\0cd /workspace && python3 tools/n1_lease.py loop\0",
                    "/proc/100/environ": "VY_POOL_MAX=99\0",
                    "/proc/101/cmdline": "/usr/bin/python3\0/workspace/infra/n1_lease.py\0loop\0",
                    "/proc/101/environ": "VY_POOL_MAX=8\0HF_TOKEN=dummy\0GPU_LEASE_DIR=/run/gpu-lease\0VY_POOL_BORROW=4\0"
                                         "MOTD=dummy\nVY_POOL_IDLE_S=1\nGPU_LEASE_DIR=/x\n\0VY_POOL_NAME=a\nb\0",
                    "/proc/200/cmdline": "python3\0n1_lease.py\0loop\0",
                    "/proc/300/cmdline": "python3\0n1_lease.py\0loop\0", "/proc/300/environ": "VY_POOL_MAX=1\0"})
    (tmp_path / "proc/200/environ").mkdir()
    for f in bin_.iterdir():
        f.chmod(0o755)
    out = _sh(policy.COMMANDS["lease"], tmp_path, str(bin_))
    assert out == ("session n1-lease\nloop n1-lease 101\nVY_POOL_MAX=8\nGPU_LEASE_DIR=/run/gpu-lease\nVY_POOL_BORROW=4\n"
                   "VY_POOL_NAME=a b\nsession n1-lease-circuits\nunreadable n1-lease-circuits 200\n")


def test_the_lease_command_reads_the_limits_file_of_a_loop_under_the_policy(tmp_path):
    bin_ = tmp_path / "bin"
    _put(tmp_path, {"bin/tmux": "#!/bin/sh\necho 'n1-lease 100'\n", "bin/ps": '#!/bin/sh\n[ "$2" = 100 ] && echo " 101"\nexit 0\n',
                    "/proc/101/cmdline": "python3\0n1_lease.py\0loop\0",
                    "/proc/101/environ": f"VY_POOL_MAX=6\0GPU_LEASE_DIR={tmp_path}/gl\0VY_POOL_POLICY=python3 -m cluster\0"})
    for f in bin_.iterdir():
        f.chmod(0o755)
    head = "session n1-lease\nloop n1-lease 101\nVY_POOL_MAX=6\nGPU_LEASE_DIR=/gl\nVY_POOL_POLICY=python3 -m cluster\n"
    assert _sh(policy.COMMANDS["lease"], tmp_path, str(bin_)) == head + "limits\n"
    _put(tmp_path, {"gl/n1-lease.limits": "max=8 borrow=4 borrow-until= from=policy\n"})
    out = _sh(policy.COMMANDS["lease"], tmp_path, str(bin_))
    assert out == head + "limits max=8 borrow=4 borrow-until= from=policy\n"
    lv = policy.Live({}, {}, {})
    policy._read_lease(N1, out, lv)
    assert {k: v for k, v in lv.values.items() if ".provers." in k} == {
        f"{N1}.lease-pool.provers.max": "8", f"{N1}.lease-pool.provers.borrow": "4",
        f"{N1}.lease-pool.provers.dir": "/gl"}
    _put(tmp_path, {"/proc/101/environ": f"VY_POOL_MAX=6\0GPU_LEASE_DIR={tmp_path}/gl\0VY_POOL_POLICY=\0"})
    out = _sh(policy.COMMANDS["lease"], tmp_path, str(bin_))
    assert out == "session n1-lease\nloop n1-lease 101\nVY_POOL_MAX=6\nGPU_LEASE_DIR=/gl\nVY_POOL_POLICY=\n"
    lv = policy.Live({}, {}, {})
    policy._read_lease(N1, out, lv)
    assert {k: v for k, v in lv.values.items() if ".provers." in k} == {      # empty is off, to n1_lease.py as here
        f"{N1}.lease-pool.provers.max": "6", f"{N1}.lease-pool.provers.dir": "/gl"}


def test_the_files_commands_mark_absent_files_and_read_one_line_of_the_dispatch_env(tmp_path):
    _put(tmp_path, {policy.LEAN_SLOTS: "check 2\naudit 2", policy.DISPATCH_ENV: "HF_TOKEN=dummy\nVY_PROVER_CPUS=160-191\n"})
    files = policy._files(_sh(policy.COMMANDS["files"], tmp_path))
    assert files == {policy.LEAN_SLOTS: "check 2\naudit 2\n", policy.CHECK_SLOTS: None, "/etc/vy/direct-gpus": None,
                     "/etc/vy/direct-cpus": None, policy.USER_CPUSET: None, policy.DISPATCH_ENV: "VY_PROVER_CPUS=160-191\n"}
    _put(tmp_path, {policy.DISPATCH_ENV: "HF_TOKEN=dummy\n"})
    assert policy._files(_sh(policy.COMMANDS["files"], tmp_path))[policy.DISPATCH_ENV] == ""
    _put(tmp_path, {QUIET: "provers\n"})
    assert policy._files(_sh(policy.COMMANDS["holds"], tmp_path)) == {DISK: None, QUIET: "provers\n\n"}


def _n2(tmp_path: Path, status: str, windows: str | None = None) -> str:
    """Run the node 2 command with node 2's clock at T: a `date` on PATH that answers `date +%s` and passes the rest on."""
    _put(tmp_path, {f"{policy.FILL}/status.txt": status + "\n  running x\n", f"{policy.FILL}/cpu-sets": "0 0-7\n",
                    "bin/date": f'#!/bin/sh\n[ "$*" = +%s ] && {{ echo {T}; exit 0; }}\nexec {shutil.which("date")} "$@"\n'})
    (tmp_path / "bin/date").chmod(0o755)
    w = tmp_path / policy.FILL.lstrip("/") / "windows"
    if windows is None:
        w.unlink(missing_ok=True)
    else:
        w.write_text(windows)
    return _sh(policy.COMMANDS["n2"], tmp_path, str(tmp_path / "bin"))


@pytest.mark.parametrize("status, windows, why", [
    (STATUS.replace("timed False", "timed True"), None, "a timed window is on or waiting"),
    (STATUS.replace("window waiting False", "window waiting True"), "2026-10-03T19:41Z 20\n", "a timed window is on or waiting"),
    (STATUS.replace("19:39:59Z", "19:39:29Z"), None, "status.txt is 61 s old: is the fill runner running?"),
    (STATUS.replace("19:39:59Z", "17:00:00Z"), "2026-10-03T19:30Z 30\n", "status.txt is 9630 s old"),
    (STATUS.replace("19:39:59Z", "19:40:31Z"), None, "status.txt is stamped 1 s ahead of node 2's clock"),
    (STATUS.split(" ", 1)[1], None, "status.txt does not say when it was written"),
    (STATUS, "2026-10-03T19:40Z 20\n", "window 2026-10-03T19:40Z (20 min) is on or starts within 60 s"),
    (STATUS, "# start UTC, minutes\n2026-10-03T19:41Z 10 # in 30 s\n", "window 2026-10-03T19:41Z (10 min) is on"),
    (STATUS, "2026-10-03T19:15Z # 30 minutes when unsaid\n", "window 2026-10-03T19:15Z (30 min) is on"),
    (STATUS, "2026-10-03T19:00Z 40.6", "window 2026-10-03T19:00Z (40.6 min) is on"),
    (STATUS, "garbage 5\n2026-10-04T06:30Z 190\n", "windows line 'garbage 5' does not parse"),
    (STATUS, "2026-10-03T19:00Z 1e3\n", "windows line '2026-10-03T19:00Z 1e3' does not parse"),
])
def test_node_2_s_gate_refuses_before_reading_anything_else_unless_quiet_fresh_and_clear_of_windows(tmp_path, status, windows,
                                                                                                    why):
    out = _n2(tmp_path, status, windows)
    assert out.startswith(f"refused {why}") and out.endswith(f": {status}\n") and out.count("\n") == 1


@pytest.mark.parametrize("windows", [
    None, "", "# start UTC, minutes\n\n2026-10-03T18:25Z 10 # merge checks off\n2026-10-03T19:42Z 10\n",
    "2026-10-03T19:00Z 40.4", "2026-10-03T19:41:31Z 1 # starts in 61 s\n"])
def test_node_2_s_gate_reads_the_files_while_quiet_fresh_and_clear_of_windows(tmp_path, windows):
    out = _n2(tmp_path, STATUS, windows)
    files = policy._files(out)
    assert out.startswith(f"status {STATUS}\n") and files[f"{policy.FILL}/cpu-sets"] == "0 0-7\n\n"
    assert files[f"{policy.FILL}/windows"] == (None if windows is None else windows + "\n")


def test_node_2_s_gate_fails_when_status_txt_cant_be_read(tmp_path):
    _n2(tmp_path, STATUS)
    (tmp_path / policy.FILL.lstrip("/") / "status.txt").unlink()
    with pytest.raises(subprocess.CalledProcessError) as e:
        _sh(policy.COMMANDS["n2"], tmp_path, str(tmp_path / "bin"))
    assert e.value.returncode == 1 and e.value.stdout == ""
