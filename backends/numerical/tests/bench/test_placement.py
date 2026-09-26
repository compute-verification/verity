"""Prover and verifier on separate machines (``bench.placement``, ``bench.cell``): machine identity, not container hostnames, and a
routed link, not a host-private bridge (red-team-flock, coordinator 2026-09-26 12:00Z)."""
from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from verity_numerical.bench import cell as C
from verity_numerical.bench import placement as PL

from test_cell import BF16, FLOCK, SEPARATE, _cell, _doc, _real_runs  # noqa: E402

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
     "host-private address"),
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


def test_the_probe_runs_before_the_job_and_records_the_machine(tmp_path):
    """The wrapper each job runs: placement.json in the run dir, then the job itself with its own arguments."""
    env = dict(os.environ, RESEARCH_RUN_DIR=str(tmp_path), RUNPOD_POD_ID="pod-x", RUNPOD_PUBLIC_IP="203.0.113.7")
    r = subprocess.run(["bash", "-c", C.PROBE_WRAP, "cell-job", "127.0.0.1:9", "echo", "job", "ran"], cwd=C.LW.REPO, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "job ran", r.stderr
    rec = json.loads((tmp_path / "placement.json").read_text())
    assert (rec["pod_id"], rec["public_ip"]) == ("pod-x", "203.0.113.7") and rec["boot_id"] and rec["link"]["peer_ip"] == "127.0.0.1"
    ident = PL.from_job(tmp_path)
    assert ident["pod_id"] == "pod-x" and ident["link"]["peer_ip"] == "127.0.0.1"


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
