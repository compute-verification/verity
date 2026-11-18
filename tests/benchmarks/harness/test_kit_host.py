"""The harness kit's lease argv and its host stamp, with no GPU and over stand-in nvidia-smi answers. No test starts a program:
the stamp's queries go through a stand-in."""

from __future__ import annotations

import math
import platform
import re

import pytest
from kit_standins import CARD, GPU0, GPU1, HEADER, SHA, _host, _row
from kit_standins import no_gpu as no_gpu
from verity_bench import kit
from verity.device import load_card, skus
from verity_config import gpus as registry

CARD_FILES = kit.cards_directory()

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


# ---- the card's short name ---------------------------------------------------------------------------------------------------


def test_a_card_short_is_refused_by_name_unless_one_records_driver_holds_it_exactly(tmp_path):
    assert kit.card_short("NVIDIA H200") == "h200"
    for name in ("NVIDIA A100-SXM4-80GB", "NVIDIA H100", "nvidia h200", ""):
        with pytest.raises(SystemExit, match=f"^0 GPU card files under {re.escape(str(CARD_FILES))} have {re.escape(repr(name))}.*give it a `short`"):
            kit.card_short(name)
    h200 = (CARD_FILES / "h200-sxm-141gb.toml").read_text()
    (tmp_path / "h200-sxm-141gb.toml").write_text(h200)
    (tmp_path / "h200-twin.toml").write_text(h200.replace('sku = "h200-sxm-141gb"', 'sku = "h200-twin"'))
    with pytest.raises(SystemExit, match="^2 GPU card files .* add { value = 'NVIDIA H200', source = ... }"):
        kit.card_short("NVIDIA H200", directory=tmp_path)
    assert kit.wrong_card("spec", "h200", "NVIDIA H200", directory=tmp_path) is not None


def test_the_kit_finds_a_card_by_its_driver_name_only_through_the_devices_lookup(tmp_path, monkeypatch):
    """`verity.device.gpu_card_reporting` is the one lookup (TOP-441, 18 Nov): the kit keeps no matcher of its own, a
    record without a `short` is refused naming it, and a name no single record holds is no card of a spec's."""

    import inspect

    import verity.device

    source = inspect.getsource(kit)
    assert "_holding" not in source and "_card_named" not in source and source.count("gpu_card_reporting(") == 1
    asked = []
    lookup = verity.device.gpu_card_reporting
    monkeypatch.setattr(verity.device, "gpu_card_reporting", lambda name, **k: asked.append(name) or lookup(name, **k))
    assert [kit.card_record("NVIDIA H200")["sku"], kit.card_short("NVIDIA H200"), kit.card_arch("NVIDIA H200")] == [
        "h200-sxm-141gb", "h200", "sm_90a"]
    assert kit.is_card("NVIDIA H200", "h200") and asked == ["NVIDIA H200"] * 4
    h200 = (CARD_FILES / "h200-sxm-141gb.toml").read_text()
    (tmp_path / "h200-sxm-141gb.toml").write_text(re.sub(r'(?m)^short = .*\n', "", h200))
    with pytest.raises(SystemExit, match="^card record 'h200-sxm-141gb' under .* has 'NVIDIA H200' in `driver` and no `short`"):
        kit.card_short("NVIDIA H200", directory=tmp_path)
    assert not kit.is_card("NVIDIA H200", "h200", directory=tmp_path)
    (tmp_path / "h200-sxm-141gb.toml").write_text(h200)
    (tmp_path / "h200-twin.toml").write_text(h200.replace('sku = "h200-sxm-141gb"', 'sku = "h200-twin"'))
    assert not kit.is_card("NVIDIA H200", "h200", directory=tmp_path)


def test_a_members_card_part_is_one_gpu_card_files_short(tmp_path):
    assert kit.card_named_short("h200")["sku"] == "h200-sxm-141gb"
    with pytest.raises(SystemExit, match=r"^0 GPU card files under .* have `short` 'NVIDIA H200': one of \[.*'h200'"):
        kit.card_named_short("NVIDIA H200")
    h200 = (CARD_FILES / "h200-sxm-141gb.toml").read_text()
    for sku in ("h200-sxm-141gb", "h200-twin"):
        (tmp_path / f"{sku}.toml").write_text(h200.replace('sku = "h200-sxm-141gb"', f'sku = "{sku}"'))
    with pytest.raises(SystemExit, match="^2 GPU card files"):
        kit.card_named_short("h200", directory=tmp_path)


@pytest.mark.parametrize("card, short", [("NVIDIA H200", "h200"), ("NVIDIA H200, NVIDIA H200", "h200"),
                                         ("NVIDIA H100 80GB HBM3", "h100"), ("NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                                                             "rtx-pro-6000"), ("NVIDIA B200", "b200")])
def test_a_card_is_the_workloads_when_card_short_resolves_each_name_to_its_short(card, short):
    assert kit.is_card(card, short) and kit.wrong_card("spec", short, card) is None


@pytest.mark.parametrize("card", ["NVIDIA H100 NVL", "NVIDIA H100 PCIe", "NVIDIA H200, NVIDIA B200", "none", ""])
def test_any_other_card_is_refused_naming_the_workloads(card):
    assert not kit.is_card(card, "h200")
    assert kit.wrong_card("steady", "h200", card) == (
        f"the card is a {card}, not the h200 of spec 'steady' (NVIDIA H200): run it on a node whose card is a NVIDIA H200, "
        "or use the spec for this card")


def test_each_gpu_a_spec_may_name_is_one_card_records_short():
    shorts = [kit.card_short(d["value"]) for sku in skus("gpu", directory=CARD_FILES)
              for d in load_card(sku, directory=CARD_FILES).get("driver", ())]
    assert set(registry()) <= set(shorts), sorted(set(registry()) - set(shorts))


def test_a_gpu_target_reads_its_names_from_the_card_file_and_refuses_what_has_none(tmp_path):
    assert kit.gpu_target("h200-sxm-141gb", directory=CARD_FILES) == {"device_name": "NVIDIA H200", "compute_capability": [9, 0],
                                                                      "num_sms": 132}
    assert kit.gpu_target("b200-sxm-180gb", directory=CARD_FILES)["compute_capability"] == [10, 0]
    with pytest.raises(KeyError, match="no card 'a100-sxm4-80gb'"):
        kit.gpu_target("a100-sxm4-80gb", directory=CARD_FILES)
    with pytest.raises(SystemExit, match="^card 'xeon-6776p-2s' .* is no compile target"):
        kit.gpu_target("xeon-6776p-2s", directory=CARD_FILES)
    h200 = (CARD_FILES / "h200-sxm-141gb.toml").read_text()
    (tmp_path / "h200-sxm-141gb.toml").write_text(h200.replace('arch = "sm_90a"', 'arch = "GH100"'))
    with pytest.raises(SystemExit, match="arch 'GH100'"):
        kit.gpu_target("h200-sxm-141gb", directory=tmp_path)
