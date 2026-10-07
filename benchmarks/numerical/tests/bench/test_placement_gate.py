"""The proof service's deployment gate (``bench.placement.gate``): the firewall, network certifiers, memory challengers and
challenger each where the architecture puts them, over injected probe records and ``/proc`` readings."""
from __future__ import annotations

import io
import json
import os

import pytest

from verity_numerical.bench import placement as PL

NODE1 = {"product_uuid": "u-n1", "boot_id": "b-n1", "hypervisor": True, "sys_vendor": "QEMU", "public_ip": "203.0.113.1",
         "cpu_model": "AMD EPYC 9654"}
NODE2 = {"product_uuid": "u-n2", "boot_id": "b-n2", "hypervisor": True, "sys_vendor": "QEMU", "public_ip": "203.0.113.2",
         "cpu_model": "AMD EPYC 9654"}
FIREWALL = {"product_uuid": "u-fw", "boot_id": "b-fw", "hypervisor": True, "sys_vendor": "QEMU", "public_ip": "203.0.113.9",
            "cpu_model": "AMD EPYC 9654"}
CHALLENGER = {"product_uuid": "u-ch", "boot_id": "b-ch", "hypervisor": False, "sys_vendor": "Dell Inc.", "public_ip": "198.51.100.7",
              "cpu_model": "Intel Xeon 6538N"}
#: The memory challenger inside node 2's guest: its kernel's boot id; the DMI uuid is root-only, so a user's probe lacks it.
INSIDE_N2 = {"boot_id": "b-n2", "hypervisor": True, "sys_vendor": "QEMU", "public_ip": "203.0.113.2", "cpu_model": "AMD EPYC 9654"}

SCOPE = "/pous-quiet.slice/run-u7.scope"
CORES = frozenset(range(124, 128))
OWN = "4000"


def _task(pid, name, cpus, cgroup, kthread=False):
    return {"pid": pid, "name": name, "cpus": cpus, "cgroup": cgroup, "kthread": kthread}


#: Node 2 with vy-pous-quiet's scope on 124-127: the role's process and a thread of it pinned to one core, a child scope of it,
#: the rest of the host on 0-123, provers.slice on 128-191, and per-CPU kernel threads on every core.
QUIET = {OWN: _task(OWN, "python3", "124-127", SCOPE), "4001": _task(OWN, "python3", "126", SCOPE),
         "4100": _task("4100", "responder", "125", SCOPE + "/responder"),
         "1": _task("1", "systemd", "0-123", "/init.scope"), "800": _task("800", "sshd", "0-123", "/system.slice/ssh.service"),
         "900": _task("900", "vllm", "128-191", "/provers.slice/run-u3.scope"),
         "25": _task("25", "ksoftirqd/125", "125", "/", kthread=True), "2": _task("2", "kthreadd", "0-191", "/", kthread=True)}

GUARDED = [("worker", "vy-nebius-2", NODE2)]
ABSENT = ("firewall", "challenger")


def _mc(**kw):
    args = {"role": "memory-challenger", "me": INSIDE_N2, "peers": GUARDED, "absent": ABSENT, "cores": CORES, "tasks": QUIET,
            "own": OWN, **kw}
    return PL.gate(**args)


def _problems(v):
    assert v["passed"] is False and v["problems"], v
    return "\n".join(v["problems"])


def test_the_rule_table_names_every_role_and_only_known_peers():
    assert set(PL.GATE_RULES) == {"firewall", "certifier", "memory-challenger", "challenger"}
    for role, rule in PL.GATE_RULES.items():
        assert set(rule) == {"same", "separate", "vm", "cores"}
        assert set(rule["same"]) | set(rule["separate"]) <= set(PL.PEER_ROLES) and not set(rule["same"]) & set(rule["separate"])
    assert PL.GATE_RULES["certifier"] == PL.GATE_RULES["memory-challenger"]


def test_the_memory_challenger_inside_node_2_on_its_quiet_cores_passes():
    v = _mc()
    assert v["passed"] is True and v["problems"] == [] and v["schema"] == PL.GATE_SCHEMA
    assert v["peers"] == [{"name": "vy-nebius-2", "role": "worker", "relation": "same machine", "problems": []}]
    assert v["absent"] == ["challenger", "firewall"]
    assert v["cores"] == {"declared": "124-127", "own": "124-127", "cgroup": SCOPE, "tasks": len(QUIET), "kernel_threads": 2,
                          "intruders": {}}
    assert PL.gate("certifier", INSIDE_N2, GUARDED, ABSENT, CORES, QUIET, OWN)["passed"] is True


@pytest.mark.parametrize("me,why", [
    ({**INSIDE_N2, "boot_id": "b-n1"}, "its boot_id b-n1 is not the node's b-n2"),                  # on node 1
    ({**INSIDE_N2, "boot_id": "b-host"}, "its boot_id b-host is not the node's b-n2"),              # on the guest's host
    ({**INSIDE_N2, "product_uuid": "u-n1"}, "its product_uuid u-n1 is not the node's u-n2"),
    ({"public_ip": "203.0.113.2", "cpu_model": "AMD EPYC 9654"}, "none of machine_id, product_uuid, boot_id is recorded on both"),
])
def test_a_memory_challenger_outside_the_node_it_guards_is_refused(me, why):
    assert f"vy-nebius-2 (worker): {why}" in _problems(_mc(me=me))


@pytest.mark.parametrize("kw,why", [
    ({"peers": []}, "a memory-challenger guards exactly one worker (--peer worker=FILE): 0 named"),
    ({"peers": GUARDED + [("worker", "vy-nebius-1", NODE1)]}, "guards exactly one worker (--peer worker=FILE): 2 named"),
    ({"peers": [], "absent": ABSENT + ("worker",)}, "--absent worker: a memory-challenger's rule needs its worker"),
    ({"absent": ("firewall",)}, "no challenger named: name every one (--peer challenger[:NAME]=FILE), or --absent challenger"),
    ({"peers": GUARDED + [("firewall", "fw", FIREWALL)]}, "firewall is both named and declared absent"),
])
def test_every_peer_role_is_named_or_declared_absent(kw, why):
    assert why in _problems(_mc(**kw))


def test_a_memory_challenger_sharing_the_firewalls_or_the_challengers_machine_is_refused():
    v = _mc(peers=GUARDED + [("firewall", "fw", {**FIREWALL, "boot_id": "b-n2"})], absent=("challenger",))
    assert "fw (firewall): they share boot_id b-n2 (one machine)" in _problems(v)
    v = _mc(peers=GUARDED + [("challenger", "ch", CHALLENGER)], absent=("firewall",))
    assert v["passed"] is True and [p["relation"] for p in v["peers"]] == ["same machine", "separate machine"]


@pytest.mark.parametrize("tasks,why", [
    ({**QUIET, OWN: {**QUIET[OWN], "cpus": "0-191"}}, "this process may run on CPUs 0-191, not exactly the declared 124-127"),
    ({**QUIET, OWN: {**QUIET[OWN], "cpus": "124-126"}}, "may run on CPUs 124-126, not exactly the declared 124-127"),
    ({**QUIET, "1": {**QUIET["1"], "cpus": "0-191"}}, "1 process(es) outside cgroup /pous-quiet.slice/run-u7.scope may run on the "
                                                     "dedicated CPUs 124-127: systemd (pid 1, /init.scope) on 124-127"),
    ({**QUIET, "901": _task("900", "vllm", "120-127", "/provers.slice/run-u3.scope")}, "vllm (pid 900, /provers.slice/run-u3.scope) "
                                                                                      "on 124-127"),
    ({**QUIET, "5000": _task("5000", "stress", "127", "/pous-quiet.slice/run-u8.scope")}, "stress (pid 5000, "
                                                                                         "/pous-quiet.slice/run-u8.scope) on 127"),
    ({**QUIET, "5001": _task("5001", "sibling", "0-191", SCOPE + "-other")}, "sibling (pid 5001"),       # a prefix, not a child
    ({**QUIET, "5002": _task("5002", "odd", None, "/user.slice")}, "odd (pid 5002, /user.slice) on 124-127"),   # unreadable affinity
    ({**QUIET, "5003": _task("5003", "odd", "x-y", "/user.slice")}, "odd (pid 5003, /user.slice) on 124-127"),
    ({**QUIET, OWN: {**QUIET[OWN], "cgroup": "/"}}, "this process (task 4000) is not in a cgroup of its own: '/'"),
    ({k: t for k, t in QUIET.items() if k != OWN}, "this process (task 4000) is not in a cgroup of its own: None"),
])
def test_the_dedicated_cores_are_the_roles_alone(tasks, why):
    v = _mc(tasks=tasks)
    assert any(p.startswith("cores: ") and why in p for p in v["problems"]), v["problems"]


def test_the_cores_refusal_names_the_first_intruders_and_counts_the_rest():
    many = {**QUIET, **{str(6000 + i): _task(str(6000 + i), f"job{i}", "0-191", "/user.slice") for i in range(7)}}
    v = _mc(tasks=many)
    assert "7 process(es) outside cgroup" in _problems(v) and "job4 (pid 6004" in _problems(v) and "and 2 more" in _problems(v)
    assert "job5" not in _problems(v) and set(v["cores"]["intruders"]) == {str(6000 + i) for i in range(7)}
    assert v["cores"]["intruders"]["6000"] == {"name": "job0", "cgroup": "/user.slice", "cpus": "124-127"}
    assert "cores: no dedicated cores declared (--cores)" in _problems(_mc(cores=None))


def test_the_firewall_is_a_vm_of_its_own_separate_from_every_worker_and_the_challenger():
    peers = [("worker", "vy-nebius-1", NODE1), ("worker", "vy-nebius-2", NODE2), ("challenger", "ch", CHALLENGER)]
    fw_tasks = {OWN: _task(OWN, "firewall", "2-3", "/firewall.slice/fw.service"), "1": _task("1", "systemd", "0-1", "/init.scope")}
    v = PL.gate("firewall", FIREWALL, peers, (), frozenset({2, 3}), fw_tasks, OWN)
    assert v["passed"] is True, v["problems"]
    assert [(p["name"], p["relation"]) for p in v["peers"]] == [("vy-nebius-1", "separate machine"), ("vy-nebius-2", "separate machine"),
                                                                 ("ch", "separate machine")]
    on_n2 = PL.gate("firewall", {**FIREWALL, "boot_id": "b-n2"}, peers, (), frozenset({2, 3}), fw_tasks, OWN)
    assert "vy-nebius-2 (worker): they share boot_id b-n2 (one machine)" in _problems(on_n2)
    metal = PL.gate("firewall", {**FIREWALL, "hypervisor": False, "sys_vendor": "Supermicro"}, peers, (), frozenset({2, 3}), fw_tasks, OWN)
    assert "a firewall runs in a VM guest of its own: this machine's hypervisor flag is False and its DMI sys_vendor 'Supermicro'" \
        in _problems(metal)
    assert PL.gate("firewall", {**FIREWALL, "hypervisor": None}, peers, (), frozenset({2, 3}), fw_tasks, OWN)["passed"] is True  # QEMU
    shared = PL.gate("firewall", FIREWALL, peers, (), frozenset({1, 2, 3}), fw_tasks, OWN)
    assert "cores: this process may run on CPUs 2-3, not exactly the declared 1-3" in _problems(shared)
    assert "systemd (pid 1, /init.scope) on 1" in _problems(shared)
    alone = PL.gate("firewall", FIREWALL, peers[:2], ("challenger",), frozenset({2, 3}), fw_tasks, OWN)
    assert alone["passed"] is True and alone["absent"] == ["challenger"]
    assert "no worker named" in _problems(PL.gate("firewall", FIREWALL, peers[2:], (), frozenset({2, 3}), fw_tasks, OWN))


def test_the_challenger_is_separate_from_the_firewall_and_every_node_and_needs_no_cores():
    peers = [("firewall", "fw", FIREWALL), ("worker", "vy-nebius-1", NODE1), ("worker", "vy-nebius-2", NODE2)]
    v = PL.gate("challenger", CHALLENGER, peers)
    assert v["passed"] is True and v["cores"] is None
    v = PL.gate("challenger", {**CHALLENGER, "product_uuid": "u-fw"}, peers)
    assert "fw (firewall): they share product_uuid u-fw (one machine)" in _problems(v)
    v = PL.gate("challenger", {**CHALLENGER, "public_ip": "203.0.113.1"}, peers)       # one public IP, no NAT exception for VMs
    assert "vy-nebius-1 (worker): they share public_ip 203.0.113.1" in _problems(v)
    v = PL.gate("challenger", CHALLENGER, peers + [("challenger", "ch2", CHALLENGER)])
    assert "ch2 (challenger): a challenger's rule names no challenger" in _problems(v)


RTT = {"method": PL.RTT_METHOD, "median_ms": 0.42, "min_ms": 0.38, "n": 30, "failed": 0}


@pytest.mark.parametrize("link,why", [
    ({"peer": "203.0.113.9:22", "peer_ip": "203.0.113.9", "rtt": {**RTT, "median_ms": 0.03}}, "no TCP-connect round trip of at least 0.1 ms"),
    ({"peer": "203.0.113.9:22", "peer_ip": "203.0.113.9", "rtt": {**RTT, "n": 29, "failed": 1}}, "29 connected, 1 failed"),
    ({"peer": "203.0.113.9:22", "peer_ip": "203.0.113.9"}, "median None ms"),                                     # not measured
    ({"peer": "172.24.0.3:22", "peer_ip": "172.24.0.3", "rtt": RTT}, "it reaches the peer at 172.24.0.3, a host bridge"),
    ({"peer": "203.0.113.7:22", "peer_ip": "203.0.113.7", "rtt": RTT}, "at its own public IP 203.0.113.7"),
    ({"peer": "10.0.5.9:22", "peer_ip": "10.0.5.9", "rtt": RTT}, "a private address, and the machine ids are not known to differ"),
])
def test_a_measured_link_to_a_separate_peer_is_a_routed_one_above_the_rtt_floor(link, why):
    peers = [("firewall", "fw", FIREWALL), ("worker", "vy-nebius-2", NODE2)]
    me = {**CHALLENGER, "public_ip": "203.0.113.7"}
    ok = {"peer": "203.0.113.9:22", "peer_ip": "203.0.113.9", "source_ip": "198.51.100.7", "rtt": RTT}
    assert PL.gate("challenger", me, peers, links={"fw": ok})["passed"] is True
    assert "fw (firewall): " in _problems(v := PL.gate("challenger", me, peers, links={"fw": link})) and why in _problems(v)
    assert "--link nobody: no peer of that name is checked separate" in _problems(PL.gate("challenger", me, peers, links={"nobody": ok}))


def test_cpu_lists_round_trip():
    assert PL.cpu_list("0-3,8, 10-11") == {0, 1, 2, 3, 8, 10, 11} and PL.cpu_text(PL.cpu_list("0-3,8,10-11")) == "0-3,8,10-11"
    assert PL.cpu_list("") == frozenset() and PL.cpu_list(None) == frozenset() and PL.cpu_text(frozenset()) == ""
    with pytest.raises(ValueError):
        PL.cpu_list("a-b")


def _proc(root, tasks: dict, cgroups: dict | None = None):
    """A ``/proc`` tree under ``root``: ``tasks`` is ``{(pid, tid): status text}``, ``cgroups`` the ``cgroup`` file per pair."""
    for (pid, tid), status in tasks.items():
        d = root / pid / "task" / tid
        d.mkdir(parents=True)
        (d / "status").write_text(status)
        if (cgroups or {}).get((pid, tid)) is not None:
            (d / "cgroup").write_text(cgroups[(pid, tid)])
    (root / "self").mkdir(exist_ok=True)
    return str(root)


def _status(name, ppid, cpus, kthread=None):
    lines = [f"Name:\t{name}", f"PPid:\t{ppid}", *([f"Kthread:\t{kthread}"] if kthread is not None else []),
             f"Cpus_allowed_list:\t{cpus}"]
    return "\n".join(lines) + "\n"


def test_read_tasks_reads_each_threads_affinity_and_cgroup(tmp_path):
    root = _proc(tmp_path / "proc", {
        ("1", "1"): _status("systemd", 0, "0-123"),
        ("2", "2"): _status("kthreadd", 0, "0-191"),
        ("25", "25"): _status("ksoftirqd/125", 2, "125"),
        ("30", "30"): _status("kworker", 9, "0-191", kthread=1),
        ("4000", "4000"): _status("python3", 1, "124-127"),
        ("4000", "4001"): _status("python3", 1, "126"),
        ("77", "77"): _status("gone", 1, "0-191"),                     # exited between its status and its cgroup
        ("88", "88"): _status("v1", 1, "0-7"),
    }, {("1", "1"): "0::/init.scope\n", ("2", "2"): "0::/\n", ("25", "25"): "0::/\n", ("30", "30"): "0::/\n",
        ("4000", "4000"): f"0::{SCOPE}\n", ("4000", "4001"): f"0::{SCOPE}\n",
        ("88", "88"): "12:memory:/docker/abc\n5:cpuset:/docker/abc\n1:name=systemd:/docker/abc\n"})
    t = PL.read_tasks(root)
    assert set(t) == {"1", "2", "25", "30", "4000", "4001", "88"}
    assert t["4001"] == {"pid": "4000", "name": "python3", "cpus": "126", "cgroup": SCOPE, "kthread": False}
    assert t["1"]["cgroup"] == "/init.scope" and not t["1"]["kthread"]
    assert t["2"]["kthread"] and t["25"]["kthread"] and t["30"]["kthread"]
    assert t["88"]["cgroup"] == "/docker/abc"
    probs, ev = PL.dedicated_cores(CORES, t, "4000")
    assert probs == [] and ev["kernel_threads"] == 3 and ev["intruders"] == {} and ev["tasks"] == 7


def _cli(monkeypatch, tmp_path, capsys, *argv, stdin=None, tasks=None):
    own = str(os.getpid())
    tasks = tasks if tasks is not None else {(own, own): (_status("python3", 1, "124-127"), f"0::{SCOPE}\n"),
                                            ("1", "1"): (_status("systemd", 0, "0-123"), "0::/init.scope\n")}
    root = tmp_path / "proc"
    monkeypatch.setattr(PL, "PROC", _proc(root, {k: s for k, (s, _) in tasks.items()}, {k: c for k, (_, c) in tasks.items()}))
    monkeypatch.setattr(PL, "probe", lambda *a, **k: pytest.fail("the gate probed although --self was given"))
    if stdin is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    rc = PL.main(["gate", *argv])
    out = capsys.readouterr()
    return rc, json.loads(out.out) if out.out else None, out.err


def test_the_gate_command_exits_0_on_a_pass_and_1_with_the_reasons_on_a_refusal(monkeypatch, tmp_path, capsys):
    (tmp_path / "me.json").write_text(json.dumps(INSIDE_N2))
    (tmp_path / "n2.json").write_text(json.dumps(NODE2))
    (tmp_path / "n1.json").write_text(json.dumps(NODE1))
    out = tmp_path / "placement-gate.json"
    argv = ["--role", "memory-challenger", "--self", str(tmp_path / "me.json"), "--peer", f"worker:vy-nebius-2={tmp_path / 'n2.json'}",
            "--absent", "firewall", "--absent", "challenger", "--cores", "124-127", "--out", str(out)]
    rc, v, err = _cli(monkeypatch, tmp_path / "a", capsys, *argv)
    assert rc == 0 and v["passed"] is True and err == "" and json.loads(out.read_text()) == v
    assert v["self"] == INSIDE_N2 and v["peers"][0]["name"] == "vy-nebius-2" and v["cores"]["cgroup"] == SCOPE
    argv[5] = f"worker:vy-nebius-1={tmp_path / 'n1.json'}"
    rc, v, err = _cli(monkeypatch, tmp_path / "b", capsys, *argv)
    assert rc == 1 and v["passed"] is False and json.loads(out.read_text()) == v
    assert "placement gate: memory-challenger: vy-nebius-1 (worker): its boot_id b-n2 is not the node's b-n1" in err
    rc, v, err = _cli(monkeypatch, tmp_path / "c", capsys, *argv[:2], "--self", "-", *argv[4:5], f"worker={tmp_path / 'n2.json'}",
                      *argv[6:], stdin=json.dumps(INSIDE_N2))
    assert rc == 0 and v["peers"][0]["name"] == "worker"


@pytest.mark.parametrize("argv,why", [
    (["--peer", "worker"], "--peer worker: ROLE[:NAME]=FILE"),
    (["--peer", "auditor=x.json"], "ROLE one of worker, firewall, challenger"),
    (["--peer", "worker=/nonexistent/n2.json"], "/nonexistent/n2.json: FileNotFoundError"),
    (["--cores", "a-b"], "--cores a-b: a CPU list"),
    (["--link", "fw"], "--link fw: NAME=HOST:PORT"),
])
def test_a_malformed_gate_command_is_a_usage_error(monkeypatch, tmp_path, capsys, argv, why):
    (tmp_path / "me.json").write_text(json.dumps(INSIDE_N2))
    with pytest.raises(SystemExit) as e:
        _cli(monkeypatch, tmp_path, capsys, "--role", "memory-challenger", "--self", str(tmp_path / "me.json"), *argv)
    assert e.value.code == 2 and why in capsys.readouterr().err


def test_two_peers_of_one_name_are_refused(monkeypatch, tmp_path, capsys):
    (tmp_path / "n.json").write_text(json.dumps(NODE2))
    with pytest.raises(SystemExit):
        _cli(monkeypatch, tmp_path, capsys, "--role", "challenger", "--self", str(tmp_path / "n.json"),
             "--peer", f"worker={tmp_path / 'n.json'}", "--peer", f"worker={tmp_path / 'n.json'}")
    assert "each NAME once" in capsys.readouterr().err
