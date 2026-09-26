"""Prover and verifier on separate machines (``bench.placement``, ``bench.cell``): machine identity, not container hostnames, and a
routed link, not a host-private bridge (red-team-flock, coordinator 2026-09-26 12:00Z)."""
from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
from pathlib import Path

import pytest

from verity_numerical.bench import cell as C
from verity_numerical.bench import placement as PL

from test_cell import BF16, FLOCK, SEPARATE, _cell, _doc, _real_runs  # noqa: E402

REMOTE_PROBE = PL.remote_probe


@pytest.fixture(autouse=True)
def _no_ssh(monkeypatch):
    """No test reaches a pod: a plan that would probe over ssh sees it fail unless the test supplies the probes."""
    monkeypatch.setattr(PL, "remote_probe", lambda m, peer=None: (_ for _ in ()).throw(RuntimeError("no ssh in tests")))


@pytest.fixture
def listener():
    """A local TCP server that completes handshakes (the kernel's backlog; nothing is accepted): an RTT target."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(128)
    yield f"127.0.0.1:{s.getsockname()[1]}"
    s.close()

#: The five L40S cells' pair: two containers of one US-TX-4 host (one machine id, CPU and public IP; a 172.24.0.x bridge).
CORESIDENT = {"prover": {"pod_id": "pod-l40s-a", "machine_id": "m-tx4", "public_ip": "192.0.2.44", "cpu_model": "AMD EPYC 9354"},
              "verifier": {"pod_id": "pod-l40s-b", "machine_id": "m-tx4", "public_ip": "192.0.2.44", "cpu_model": "AMD EPYC 9354"},
              "link": {"peer": "172.24.0.3:7400", "peer_ip": "172.24.0.3"}}


def _with(**changes) -> dict:
    p = copy.deepcopy(SEPARATE)
    for path, v in changes.items():
        side, key = path.split("__")
        p[side][key] = v
    return p


@pytest.mark.parametrize("placement,why", [
    (CORESIDENT, "share machine_id m-tx4"),
    (_with(verifier__machine_id=None, verifier__public_ip="203.0.113.10"), "share public_ip 203.0.113.10"),
    (_with(prover__boot_id="k1", verifier__boot_id="k1"), "share boot_id k1"),
    (_with(prover__product_uuid="U", verifier__product_uuid="U"), "share product_uuid U"),
    ({**_with(prover__machine_id=None, verifier__machine_id=None), "link": {"peer": "172.24.0.3:7400", "peer_ip": "172.24.0.3"}},
     "host bridge"),
    ({"prover": {"hostname": "a"}, "verifier": {"hostname": "b"}, "link": {}}, "machine identity not recorded"),
    ({**SEPARATE, "link": {"peer": "203.0.113.10:7400", "peer_ip": "203.0.113.10"}}, "its own public IP"),
])
def test_a_plan_on_one_machine_is_refused(placement, why):
    with pytest.raises(C.NotSupported, match=why):
        _cell(placement=placement)


def test_separate_machines_pass():
    assert PL.separation(SEPARATE["prover"], SEPARATE["verifier"], SEPARATE["link"]) == []
    ip_cpu = {"prover": {"public_ip": "203.0.113.10", "cpu_model": "Xeon"}, "verifier": {"public_ip": "198.51.100.20", "cpu_model": "Xeon"}}
    assert PL.separation(ip_cpu["prover"], ip_cpu["verifier"], {"peer_ip": "198.51.100.20"}) == []      # "at least CPU model plus public IP"
    uuids = {"product_uuid": "A"}, {"product_uuid": "B"}
    assert PL.separation(*uuids) == []
    assert PL.is_host_private("172.24.0.3") and PL.is_host_private("127.0.0.1") and PL.is_host_private("100.64.1.1")
    assert not PL.is_host_private("8.8.8.10") and not PL.is_host_private("198.51.100.20")


def test_without_the_api_a_plan_is_refused(monkeypatch):
    kw = {k: v for k, v in FLOCK.items() if k != "placement"}
    monkeypatch.setattr(PL, "from_runpod", lambda m: (_ for _ in ()).throw(RuntimeError("no RUNPOD_API_KEY")))
    with pytest.raises(C.NotSupported, match="machine identity is unavailable"):
        C.plan("C-interactive", *BF16, "vy-flock-h100", **kw)
    monkeypatch.setattr(PL, "from_runpod", lambda m: CORESIDENT["prover"] if m == "vy-flock-h100" else CORESIDENT["verifier"])
    with pytest.raises(C.NotSupported, match="share machine_id"):
        C.plan("C-interactive", *BF16, "vy-flock-h100", **kw)
    monkeypatch.setattr(PL, "from_runpod", lambda m: SEPARATE["prover"] if m == "vy-flock-h100" else SEPARATE["verifier"])
    cell = C.plan("C-interactive", *BF16, "vy-flock-h100", **kw)
    assert cell.placement["prover"]["machine_id"] == "m-h100" and cell.placement["link"]["peer_ip"] == "10.0.0.2"


def test_the_probe_runs_before_the_job_and_records_the_machine(tmp_path, listener):
    """The wrapper each job runs: placement.json in the run dir (identity, bare-metal facts, the link and its TCP-connect RTT), then
    the job itself with its own arguments."""
    env = dict(os.environ, RESEARCH_RUN_DIR=str(tmp_path), RUNPOD_POD_ID="pod-x", RUNPOD_PUBLIC_IP="203.0.113.7")
    r = subprocess.run(["bash", "-c", C.PROBE_WRAP, "cell-job", "127.0.0.1:9", listener, "echo", "job", "ran"], cwd=C.LW.REPO, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "job ran", r.stderr
    rec = json.loads((tmp_path / "placement.json").read_text())
    assert (rec["pod_id"], rec["public_ip"]) == ("pod-x", "203.0.113.7") and rec["boot_id"] and rec["link"]["peer_ip"] == "127.0.0.1"
    assert rec["hypervisor"] in (True, False, None) and "sys_vendor" in rec and "product_name" in rec
    assert rec["link"]["rtt"]["n"] == PL.RTT_CONNECTS and rec["link"]["rtt"]["method"] == PL.RTT_METHOD
    ident = PL.from_job(tmp_path)
    assert ident["pod_id"] == "pod-x" and ident["link"]["peer_ip"] == "127.0.0.1"
    r = subprocess.run(["bash", "-c", C.PROBE_WRAP, "cell-job", "", "", "true"], cwd=C.LW.REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "link" not in json.loads((tmp_path / "placement.json").read_text())   # the verifier's job


def test_registration_records_both_machines_and_refuses_a_shared_one(tmp_path):
    p, v = _real_runs(tmp_path)
    (p / "placement.json").write_text(json.dumps({"pod_id": "pod-p", "cpu_model": "Xeon 8470", "public_ip": "203.0.113.10",
                                                  "link": {"peer": "10.0.0.2:7400", "peer_ip": "10.0.0.2"}}))
    (v / "placement.json").write_text(json.dumps({"pod_id": "pod-v", "cpu_model": "EPYC 7502", "public_ip": "198.51.100.20"}))
    res = C.register(_cell(), str(p), str(v), tmp_path / "work", dry_run=True)
    assert res["problems"] == [] and "refused" not in res
    pl = json.loads(Path(res["meta"]).read_text())["cell"]["placement"]
    assert (pl["prover"]["machine_id"], pl["verifier"]["machine_id"]) == ("m-h100", "m-a40")
    assert (pl["prover"]["cpu_model"], pl["verifier"]["cpu_model"]) == ("Xeon 8470", "EPYC 7502")
    assert pl["prover"]["boot_id"] != pl["verifier"]["boot_id"] and pl["plan"] == SEPARATE
    (v / "placement.json").write_text(json.dumps({"pod_id": "pod-v", "public_ip": "203.0.113.10"}))   # the verifier's pod: the prover's IP
    res = C.register(_cell(), str(p), str(v), tmp_path / "work2", dry_run=True)
    assert res["refused"] and any("share public_ip" in x for x in res["problems"])
    (v / "placement.json").write_text(json.dumps({"pod_id": "pod-elsewhere", "public_ip": "198.51.100.20"}))
    assert any("ran on pod pod-elsewhere, the cell planned pod-v" in x for x in C.placement_problems(_cell(), p, v))


def test_the_real_cells_still_pass_on_documents():
    assert C.check(_cell(), _doc("6d1295ed")) == []


# --- a datacenter's shared NAT IP (coordinator 2026-09-26 22:00Z; red-team-flock-3's conditions S1-S5, 22:05Z) ------------------

IP = "91.199.227.82"
#: An L40S prover and an H100 verifier in EU-NL-1 as the RunPod API shows them: two machine ids behind one egress NAT, the
#: verifier reached by its global-network name.
NAT = {"prover": {"pod_id": "pod-p", "machine_id": "fpuyltj3q41d", "public_ip": IP},
       "verifier": {"pod_id": "pod-v", "machine_id": "go1zlmhqhfmq", "public_ip": IP},
       "link": {"peer": "pod-v.runpod.internal:7400", "peer_ip": "10.65.0.12"}}
RTT = {"method": PL.RTT_METHOD, "median_ms": 0.31, "min_ms": 0.27, "n": 30, "failed": 0, "target": "pod-v.runpod.internal:22",
       "target_ip": "10.65.0.12"}
#: What each pod's probe records (the plan's over ssh, and each run's placement.json): bare metal, and the prover's link and RTT.
PROBES = {"vy-flock-h100": {"pod_id": "pod-p", "public_ip": IP, "product_uuid": "uuid-p", "boot_id": "boot-p", "cpu_model": "AMD EPYC 9354",
                            "hypervisor": False, "sys_vendor": "Supermicro", "product_name": "AS -4125GS-TNRT",
                            "link": {**NAT["link"], "source_ip": "10.65.0.9", "rtt": RTT}},
          "vy-flock-verifier": {"pod_id": "pod-v", "public_ip": IP, "product_uuid": "uuid-v", "boot_id": "boot-v",
                                "cpu_model": "Intel Xeon 8470", "hypervisor": False, "sys_vendor": "Supermicro",
                                "product_name": "SYS-821GE-TNHR"}}


def _pair(**over):
    """The probed pair as ``assess`` takes it, with ``side__key`` overrides (sides prover, verifier, link, rtt; None drops the key)."""
    p = PL.merge(NAT["prover"], {k: x for k, x in PROBES["vy-flock-h100"].items() if k != "link"})
    v = PL.merge(NAT["verifier"], PROBES["vy-flock-verifier"])
    link = copy.deepcopy(PROBES["vy-flock-h100"]["link"])
    sides = {"prover": p, "verifier": v, "link": link, "rtt": link["rtt"]}
    for path, x in over.items():
        side, key = path.split("__")
        if x is None:
            sides[side].pop(key, None)
        else:
            sides[side][key] = x
    return p, v, link


@pytest.fixture
def probes(monkeypatch):
    """Both pods' ssh probes as PROBES records them; the calls made."""
    calls = []

    def probe(machine, peer=None):
        calls.append((machine, peer))
        return copy.deepcopy(PROBES[machine])

    monkeypatch.setattr(PL, "remote_probe", probe)
    return calls


def _nat_cell():
    return _cell(placement=copy.deepcopy(NAT), verifier_addr=NAT["link"]["peer"])


def _nat_runs(tmp_path, prover=None, verifier=None, mutate=None):
    """The real runs with each pod's placement.json as its probe recorded it; ``prover``/``verifier`` override keys (None drops)."""
    p, v = _real_runs(tmp_path, mutate)
    for run, machine, over in ((p, "vy-flock-h100", prover), (v, "vy-flock-verifier", verifier)):
        rec = {**copy.deepcopy(PROBES[machine]), **(over or {})}
        (run / "placement.json").write_text(json.dumps({k: x for k, x in rec.items() if x is not None}))
    return p, v


def test_a_shared_nat_ip_plan_probes_both_pods_and_records_the_evidence(probes):
    cell = _nat_cell()
    assert probes == [("vy-flock-h100", "pod-v.runpod.internal:7400"), ("vy-flock-verifier", None)]
    pl = cell.placement
    assert pl["probed"] is True and pl["shared_public_ip"] is True
    sep = pl["separation"]
    assert (sep["machine_id"], sep["product_uuid"], sep["boot_id"]) == (["fpuyltj3q41d", "go1zlmhqhfmq"], ["uuid-p", "uuid-v"],
                                                                        ["boot-p", "boot-v"])
    assert sep["bare_metal"]["verifier"] == {"hypervisor": False, "sys_vendor": "Supermicro", "product_name": "SYS-821GE-TNHR",
                                             "cpu_model": "Intel Xeon 8470"}
    assert sep["link"] == {"name": "pod-v.runpod.internal", "address": "10.65.0.12"} and sep["public_ip"] == IP
    assert (sep["rtt"]["method"], sep["rtt"]["median_ms"], sep["rtt"]["target_ip"], sep["rtt_floor_ms"]) == (PL.RTT_METHOD, 0.31,
                                                                                                             "10.65.0.12", 0.1)
    argv = cell.jobs[1]["argv"]
    assert argv[argv.index("--") + 5:][:2] == ["pod-v.runpod.internal:7400", "pod-v.runpod.internal:22"]   # the prover's probe
    assert C.Cell(**json.loads(cell.to_json())) == cell


def test_a_shared_nat_ip_that_cannot_be_probed_is_refused():
    with pytest.raises(C.NotSupported, match="could not be probed"):
        _nat_cell()


def test_a_shared_nat_ip_registers_on_the_runs_own_records(tmp_path, probes):
    p, v = _nat_runs(tmp_path)
    res = C.register(_nat_cell(), str(p), str(v), tmp_path / "work", dry_run=True)
    assert res["problems"] == [] and "refused" not in res
    pl = json.loads(Path(res["meta"]).read_text())["cell"]["placement"]
    assert pl["shared_public_ip"] is True and pl["plan"]["probed"] is True
    sep = pl["separation"]
    assert sep["boot_id"] == ["boot-p", "boot-v"] and sep["bare_metal"]["prover"]["hypervisor"] is False
    assert sep["rtt"]["median_ms"] == 0.31 and sep["link"]["name"] == "pod-v.runpod.internal"


@pytest.mark.parametrize("prover,verifier,why", [
    ({}, {"boot_id": "boot-p"}, "share boot_id boot-p"),                                   # two containers of one kernel
    ({}, {"product_uuid": "uuid-p"}, "share product_uuid uuid-p"),
    ({}, {"hypervisor": True}, "the verifier pod is a virtual machine"),                    # S1
    ({"sys_vendor": "QEMU"}, {}, "the prover pod is a virtual machine"),
    ({}, {"sys_vendor": None}, "the verifier run did not record sys_vendor"),              # S2: the plan's probe does not stand in
    ({"hypervisor": None}, {}, "the prover run did not record hypervisor"),
    ({}, {"boot_id": None}, "the verifier run's boot_id"),                                  # job.json's boot id, not the plan's
    ({"pod_id": None}, {}, "the prover run did not record pod_id"),
    ({"link": NAT["link"]}, {}, "the prover run did not record link.rtt"),
    ({}, {"product_uuid": "uuid-w"}, "product_uuid uuid-w is not the uuid-v the plan probed on pod pod-v"),
    ({"link": {**PROBES["vy-flock-h100"]["link"], "rtt": {**RTT, "median_ms": 0.03}}}, {}, "no TCP-connect RTT of at least 0.1 ms"),
    ({"link": {"peer": "pod-v.runpod.internal:7400", "peer_ip": "172.24.0.3", "rtt": {**RTT, "target_ip": "172.24.0.3"}}}, {},
     "host bridge"),                                                                        # S3
])
def test_a_shared_nat_ip_without_the_proof_is_refused_at_registration(tmp_path, probes, prover, verifier, why):
    p, v = _nat_runs(tmp_path, prover, verifier)
    res = C.register(_nat_cell(), str(p), str(v), tmp_path / "work", dry_run=True)
    assert res.get("refused") and any(why in x for x in res["problems"]), res["problems"]


@pytest.mark.parametrize("over,why", [
    ({"prover__hypervisor": True}, "the prover pod is a virtual machine"),
    ({"verifier__sys_vendor": "QEMU"}, "the verifier pod is a virtual machine"),
    ({"verifier__sys_vendor": "Microsoft Corporation"}, "the verifier pod is a virtual machine"),      # Hyper-V
    ({"prover__sys_vendor": "Amazon EC2"}, "the prover pod is a virtual machine"),
    ({"prover__hypervisor": None}, "the prover pod's hypervisor flag and DMI sys_vendor are not recorded"),
    ({"verifier__sys_vendor": None}, "the verifier pod's hypervisor flag and DMI sys_vendor are not recorded"),
])
def test_s1_a_shared_ip_needs_both_pods_bare_metal(over, why):
    a = PL.assess(*_pair(**over))
    assert a["shared_public_ip"] and any(why in x for x in a["problems"]), a["problems"]
    assert PL.assess(*_pair())["problems"] == []
    assert PL.is_hypervisor_vendor("VMware, Inc.") and PL.is_hypervisor_vendor("Xen") and not PL.is_hypervisor_vendor("Supermicro")
    assert not PL.is_hypervisor_vendor(None) and not PL.is_hypervisor_vendor("")


def test_s1_a_plan_whose_probe_shows_a_vm_is_refused(monkeypatch):
    vm = {**PROBES["vy-flock-verifier"], "hypervisor": True, "sys_vendor": "QEMU"}
    monkeypatch.setattr(PL, "remote_probe", lambda m, peer=None: copy.deepcopy(vm if m == "vy-flock-verifier" else PROBES[m]))
    with pytest.raises(C.NotSupported, match="the verifier pod is a virtual machine"):
        _nat_cell()


def test_s1_the_probe_reads_the_hypervisor_flag(monkeypatch):
    cpuinfo = {"guest": "flags\t\t: fpu vme hypervisor lahf_lm\n", "metal": "flags\t\t: fpu vme lahf_lm\n", "none": "processor : 0\n"}
    for kind, want in (("guest", True), ("metal", False), ("none", None)):
        monkeypatch.setattr(PL, "_read", lambda path, text=cpuinfo[kind]: text if path == "/proc/cpuinfo" else None)
        assert PL._hypervisor_flag() is want


@pytest.mark.parametrize("over,why", [
    ({"link__peer": "pod-w.runpod.internal:7400"}, "the link must be RunPod's global network"),      # another pod's name
    ({"link__peer": "10.65.0.12:7400"}, "the link must be RunPod's global network"),                 # an address, not the name
    ({"link__peer_ip": "100.64.0.3", "rtt__target_ip": "100.64.0.3"}, "resolving into 10.0.0.0/8"),
    ({"link__peer_ip": "172.24.0.3", "rtt__target_ip": "172.24.0.3"}, "host bridge"),
    ({"link__peer_ip": "127.0.0.1", "rtt__target_ip": "127.0.0.1"}, "host bridge"),
    ({"link__peer": f"{IP}:7400", "link__peer_ip": IP, "rtt__target_ip": IP}, f"its own public IP {IP}"),   # dialling the shared IP
])
def test_s3_a_shared_ip_links_only_by_the_verifiers_global_name(over, why):
    a = PL.assess(*_pair(**over))
    assert any(why in x for x in a["problems"]), a["problems"]


@pytest.mark.parametrize("over", [
    {"rtt__median_ms": 0.03},                                  # the co-residence round trip
    {"link__rtt": None},                                       # not measured
    {"rtt__median_ms": None, "rtt__n": 0, "rtt__failed": 30},  # every connect failed
    {"rtt__method": "hello-ping: the session's Hello/Ping round trip"},
    {"rtt__target_ip": IP, "rtt__median_ms": 4.2},             # timed via the public IP, not the link's route
])
def test_s4_a_shared_ip_needs_a_tcp_connect_rtt_on_the_links_route(over):
    a = PL.assess(*_pair(**over))
    assert any("no TCP-connect RTT of at least 0.1 ms measured on the link's own route" in x for x in a["problems"]), a["problems"]


def test_s4_the_sessions_rtt_does_not_stand_in_for_the_tcp_connects(tmp_path, probes):
    slow_session = lambda d: [m.update(value=5.0) for m in d["measurements"] if m["name"] == "net.rtt_ms"]
    fast = {"link": {**PROBES["vy-flock-h100"]["link"], "rtt": {**RTT, "median_ms": 0.03, "min_ms": 0.02}}}
    p, v = _nat_runs(tmp_path, fast, mutate=slow_session)
    res = C.register(_nat_cell(), str(p), str(v), tmp_path / "work", dry_run=True)
    assert res.get("refused") and any("no TCP-connect RTT of at least 0.1 ms" in x for x in res["problems"]), res["problems"]


def test_s4_tcp_rtt_times_kernel_connects_to_the_target(listener):
    r = PL.tcp_rtt(listener, n=10)
    assert (r["target"], r["target_ip"], r["n"], r["failed"], r["method"]) == (listener, "127.0.0.1", 10, 0, PL.RTT_METHOD)
    assert 0 < r["min_ms"] <= r["median_ms"]
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    closed = f"127.0.0.1:{s.getsockname()[1]}"
    s.close()
    r = PL.tcp_rtt(closed, n=3)
    assert (r["n"], r["failed"], r["median_ms"]) == (0, 3, None)
    p, v, link = _pair()
    assert PL.separation(p, v, {**link, "rtt": r})                                       # no connect, no exception


def test_s4_the_rtt_target_is_the_verifier_pods_sshd_on_its_global_name(monkeypatch):
    """Not the session port (the verifiers count every connection on it as a session): sshd on the same name, so the same route."""
    assert PL.rtt_target("pod-v.runpod.internal:7400") == "pod-v.runpod.internal:22"
    assert PL.rtt_target("10.0.0.2:7400") is None and PL.rtt_target(None) is None
    seen = {}

    def run(argv, **kw):
        seen.update(argv=argv, input=kw["input"])
        return subprocess.CompletedProcess(argv, 0, stdout="motd\n" + json.dumps(PROBES["vy-flock-h100"]), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert REMOTE_PROBE("vy-flock-h100", "pod-v.runpod.internal:7400") == PROBES["vy-flock-h100"]
    assert seen["argv"] == ["research", "pods", "ssh", "vy-flock-h100", "--", "python3", "-", "probe", "--peer",
                            "pod-v.runpod.internal:7400", "--rtt", "pod-v.runpod.internal:22"]
    assert seen["input"] == Path(PL.__file__).read_text()


# --- the DMI uuid on bare metal (coordinator 22:40Z: RunPod containers cannot read it) ----------------------------------------

NO_UUID = {m: {k: x for k, x in rec.items() if k != "product_uuid"} for m, rec in PROBES.items()}


def test_uuid_bare_metal_with_the_uuid_unreadable_is_accepted(tmp_path, monkeypatch):
    p, v, link = _pair(prover__product_uuid=None, verifier__product_uuid=None)
    a = PL.assess(p, v, link)
    assert a["problems"] == [] and a["evidence"]["product_uuid_read"] == [False, False]
    monkeypatch.setattr(PL, "remote_probe", lambda m, peer=None: copy.deepcopy(NO_UUID[m]))
    cell = _nat_cell()
    assert cell.placement["separation"]["product_uuid"] == [None, None]
    q, w = _nat_runs(tmp_path, {"product_uuid": None}, {"product_uuid": None})
    res = C.register(cell, str(q), str(w), tmp_path / "work", dry_run=True)
    assert res["problems"] == [] and "refused" not in res
    sep = json.loads(Path(res["meta"]).read_text())["cell"]["placement"]["separation"]
    assert sep["product_uuid_read"] == [False, False] and sep["boot_id"] == ["boot-p", "boot-v"]
    assert PL.assess(*_pair(verifier__product_uuid=None))["evidence"]["product_uuid_read"] == [True, False]   # one side read


def test_uuid_bare_metal_with_the_uuid_readable_and_equal_is_refused(tmp_path, probes):
    a = PL.assess(*_pair(verifier__product_uuid="uuid-p"))
    assert any("share product_uuid uuid-p" in x for x in a["problems"]) and a["evidence"]["product_uuid_read"] == [True, True]
    q, w = _nat_runs(tmp_path, verifier={"product_uuid": "uuid-p"})
    res = C.register(_nat_cell(), str(q), str(w), tmp_path / "work", dry_run=True)
    assert res.get("refused") and any("share product_uuid uuid-p" in x for x in res["problems"]), res["problems"]


@pytest.mark.parametrize("uuids", [{}, {"prover__product_uuid": None, "verifier__product_uuid": None}])
@pytest.mark.parametrize("vm", [{"verifier__hypervisor": True}, {"prover__sys_vendor": "QEMU"}, {"verifier__hypervisor": None}])
def test_uuid_a_vm_is_refused_whatever_the_uuid(vm, uuids):
    a = PL.assess(*_pair(**vm, **uuids))
    assert any("virtual machine" in x or "are not recorded" in x for x in a["problems"]), a["problems"]
    if uuids:                                                  # off bare metal an unreadable uuid is missing, as before
        assert any("missing: product_uuid" in x for x in a["problems"]), a["problems"]


def test_uuid_the_co_resident_pair_is_refused_without_a_uuid():
    """Two bare-metal containers of one host with no uuid read: their one kernel's boot id, the bridge and the 0.03 ms RTT refuse."""
    metal = {"boot_id": "k", "hypervisor": False, "sys_vendor": "Supermicro"}
    bridge = {**CORESIDENT["link"], "rtt": {**RTT, "median_ms": 0.03, "target_ip": "172.24.0.3"}}
    a = PL.assess({**CORESIDENT["prover"], **metal}, {**CORESIDENT["verifier"], **metal}, bridge)
    for why in ("share machine_id m-tx4", "share boot_id k", "host bridge", "no TCP-connect RTT of at least 0.1 ms"):
        assert any(why in x for x in a["problems"]), (why, a["problems"])
    a = PL.assess({**CORESIDENT["prover"], **metal, "machine_id": "m-a"}, {**CORESIDENT["verifier"], **metal, "machine_id": "m-b"},
                  {**PROBES["vy-flock-h100"]["link"]})
    assert a["problems"] and any("share boot_id k" in x for x in a["problems"])   # distinct machine ids alone never suffice


def test_the_co_resident_pair_is_still_refused():
    """Same machine id, CPU and public IP, a docker-bridge link at 0.03 ms (the five L40S cells): refused at plan and on any record."""
    with pytest.raises(C.NotSupported, match="share machine_id m-tx4"):
        _cell(placement=CORESIDENT)
    bridge = {**CORESIDENT["link"], "rtt": {**RTT, "median_ms": 0.03, "target_ip": "172.24.0.3"}}
    a = PL.assess(CORESIDENT["prover"], CORESIDENT["verifier"], bridge)
    assert a["shared_public_ip"] is True
    for why in ("share machine_id m-tx4", "host bridge", "missing: product_uuid, boot_id", "hypervisor flag and DMI sys_vendor are not recorded",
                "global network", "no TCP-connect RTT of at least 0.1 ms"):
        assert any(why in x for x in a["problems"]), (why, a["problems"])
    metal = {"product_uuid": "u", "boot_id": "k", "hypervisor": False, "sys_vendor": "Supermicro"}   # every probe fact, one kernel
    a = PL.assess({**CORESIDENT["prover"], **metal, "machine_id": "m-a"}, {**CORESIDENT["verifier"], **metal, "machine_id": "m-b"}, bridge)
    assert any("share boot_id k" in x for x in a["problems"]) and any("host bridge" in x for x in a["problems"])
    assert PL.is_host_bridge("172.24.0.3") and PL.is_host_bridge("127.0.0.1") and not PL.is_host_bridge("10.65.0.12")
