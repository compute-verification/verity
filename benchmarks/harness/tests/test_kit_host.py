"""The harness kit's lease argv and its host stamp, with no GPU and over stand-in nvidia-smi answers. No test starts a program:
the stamp's queries go through a stand-in."""

from __future__ import annotations

import math
import platform

import pytest
from kit_standins import CARD, GPU0, GPU1, HEADER, SHA, _host, _row
from kit_standins import no_gpu as no_gpu
from verity_bench import kit

# ---- the lease ---------------------------------------------------------------------------------------------------------------


def test_the_lease_is_the_gpu_lease_wait_for_the_jobs_gpus_under_its_bound():
    assert kit.lease(1, 60) == ["gpu-lease", "1", "--wait", "--max-min", "60", "--"]
    assert kit.lease(8, 90.2) == ["gpu-lease", "8", "--wait", "--max-min", "91", "--"]
    for gpus in (0, -1, 1.5, True, "1"):
        with pytest.raises(ValueError, match="one GPU or more"):
            kit.lease(gpus, 60)
    for minutes in (0, -5, math.inf, math.nan, None, "60"):
        with pytest.raises(ValueError, match="its bound in minutes"):
            kit.lease(1, minutes)


# ---- the host stamp ----------------------------------------------------------------------------------------------------------


def test_the_host_stamp_with_no_gpu_says_none(no_gpu):
    assert kit.host_stamp() == {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": platform.node(),
                                "commit": SHA, "why": "no nvidia-smi on PATH"}
    assert no_gpu == []


def test_the_host_stamp_says_none_and_why_when_nvidia_smi_finds_no_gpu(monkeypatch):
    for answer, why in (((9, "NVIDIA-SMI has failed because it couldn't communicate\nwith the NVIDIA driver.\n"),
                         "nvidia-smi exited 9: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."),
                        ((127, "TimeoutExpired: 60 s"), "nvidia-smi exited 127: TimeoutExpired: 60 s"),
                        ((0, "\n"), "nvidia-smi lists no GPU"),
                        ((0, "No devices were found\n"), f"nvidia-smi answered 'No devices were found', not {', '.join(kit.QUERY)}")):
        asked = _host(monkeypatch, {"query": answer})
        stamp = kit.host_stamp()
        assert (stamp["card"], stamp["gpus"], stamp["driver"], stamp["cuda"], stamp["why"]) == ("none", [], None, None, why)
        assert len(asked) == 1


def test_the_host_stamp_takes_its_commit_from_the_run_else_the_checkout(monkeypatch):
    asked = _host(monkeypatch, None, sha=None)
    assert kit.host_stamp()["commit"] is None and [a[0] for a in asked] == ["git"] and asked[0][-2:] == ["rev-parse", "HEAD"]
    monkeypatch.setattr(kit, "_run", lambda argv: (0, SHA + "\n"))
    assert kit.host_stamp()["commit"] == SHA


def test_the_host_stamp_names_the_card_each_gpus_uuid_and_clocks_the_driver_and_cuda(monkeypatch):
    asked = _host(monkeypatch, {"query": (0, _row(GPU0) + _row(GPU1, sm="1950", mem="[N/A]")), "header": (0, HEADER)})
    assert kit.host_stamp() == {
        "card": CARD, "gpus": [{"uuid": GPU0, "sm_mhz": 2100, "mem_mhz": 9501}, {"uuid": GPU1, "sm_mhz": 1950, "mem_mhz": None}],
        "driver": "580.95.05", "cuda": "13.0", "node": platform.node(), "commit": SHA}
    assert asked == [["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem", "--format=csv,noheader,nounits"],
                     ["nvidia-smi"]]


def test_the_host_stamp_is_of_the_leases_gpus(monkeypatch):
    asked = _host(monkeypatch, {"query": (0, _row(GPU1)), "header": (1, "")})
    monkeypatch.setenv("GPU_LEASE_UUID", GPU1)
    stamp = kit.host_stamp()
    assert asked[0][:3] == ["nvidia-smi", "-i", GPU1] and [g["uuid"] for g in stamp["gpus"]] == [GPU1]
    assert stamp["cuda"] is None and "why" not in stamp
    _host(monkeypatch, {"query": (0, _row(GPU0) + _row(GPU1, name="NVIDIA H200", driver="570.1")), "header": (0, "no version")})
    stamp = kit.host_stamp()
    assert (stamp["card"], stamp["driver"], stamp["cuda"]) == (f"{CARD}, NVIDIA H200", "580.95.05, 570.1", None)
