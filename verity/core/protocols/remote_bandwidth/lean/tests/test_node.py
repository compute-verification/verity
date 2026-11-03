"""`line.node_run_json` held to its Lean (`RemoteAudit.Line.ofSite`, served by `remote-audit-difftest`'s `node`): the
line's inputs read off the Site's `node h`, refusals as outputs so that both sides' reasons are compared word for word.
A node of one SKU and of two, with its disks and links stated or not and their bursts stated or not, and figures drawn
at random; each refusal: a log the Site's check refuses, an id that is no host, a host with no device, an SKU with no
card figures, a width a device states no squarings at, and a negative statement."""

from __future__ import annotations

import dataclasses
import shutil
from fractions import Fraction

import pytest
from hypothesis import strategies as st

import ra_cases as M
import verity_difftest as D
from verity.core import site
from verity.core.protocols.remote_bandwidth import line

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

A = site.Sheet(site.DeviceFigures("gpu-a", 1, 1 << 20), bw=1 << 14, chip=1 << 30, mem=96 << 30)
B = site.Sheet(site.DeviceFigures("gpu-b", 1, 1 << 21), bw=1 << 15, chip=1 << 31, mem=141 << 30)
HOST_CARD = {"chip_bytes": 3_105_652_736, "link_window_bytes": 6_291_456}


def port(inputs: st.SearchStrategy | None = None) -> D.Port:
    return D.Port(M.EXE, "node", line.node_run_json, st.nothing() if inputs is None else inputs, package=M.PACKAGE,
                  of_record="lean", rejects=M.REFUSALS)


server = D.server_fixture(port())


def log(*skus: str) -> site.Registration:
    """A remote site's log of one entry: host node0 holding a device of each SKU with its device memory, node0's DRAM
    and disk and its uplink; and a second host, node1, with DRAM alone."""
    sheets = {k.sku: k for k in (A, B)}
    used = {sku: sheets[sku] for sku in dict.fromkeys(skus)}
    devices = [x for i, sku in enumerate(skus) for x in (
        site.Loc(f"gpu{i}", "device", "node0", sheet=used[sku].digest()),
        site.Loc(f"hbm{i}", "memory", f"gpu{i}", tier="hbm", bytes=used[sku].mem))]
    r = site.Registration("remote")
    return r.append(r.entry(tuple(used.values()), (
        site.Loc("site0", "site"), site.Loc("node0", "host", "site0"), *devices,
        site.Loc("dram0", "memory", "node0", tier="dram", bytes=1 << 40),
        site.Loc("disk0", "memory", "node0", tier="disk", bytes=1 << 44),
        site.Loc("node1", "host", "site0"), site.Loc("dram1", "memory", "node1", tier="dram", bytes=1 << 40)),
        (site.Link("up0", ("node0", site.OUTSIDE), 100, 1000),)))


GPUS = [{"sku": "gpu-a", "chip_bytes": 263_163_904, "squarings_per_s": {"p1024": "25033300000", "p512": "80387100000"}},
        {"sku": "gpu-b", "chip_bytes": 51_118_080, "squarings_per_s": {"p1024": "1000000/3"}}]


def inputs(reg: site.Registration, host: str = "node0", gpus=GPUS, host_card=HOST_CARD, **stated) -> dict:
    return {"log": reg.to_json(), "auditor": site.Auditor((A, B)).to_json(), "host": host, "gpus": gpus,
            "host_card": host_card,
            "stated": {"field": "p1024", "encode_squarings": 3, "disk_bytes_per_s": 0, "link_bytes_per_s": 0,
                       "disk_burst_bytes": None, "link_burst_bytes": None, **stated}}


def output(v: dict) -> tuple[str, str]:
    return "output", D.canonical(line.node_run_json(v))


def test_eight_devices_of_one_sku_are_their_cards_summed(server) -> None:
    """Eight devices of one SKU: their squarings at the run's width and their on-chip bytes summed with the host's, the
    node's memory its HBM and DRAM (its disk is no node memory), a rebuild's burst S_chip, and with nothing stated of
    the disks and links, no disk or link term."""
    v = inputs(log(*["gpu-a"] * 8))
    chip = 8 * 263_163_904 + HOST_CARD["chip_bytes"]
    want = line.Rates(Fraction(8 * 25_033_300_000), 3, 0, 0, Fraction(chip), 0, 0)
    assert line.node_run_json(v) == {"chip_bytes": chip, "node_bytes": 8 * (96 << 30) + (1 << 40),
                                     "rates": want.to_json()}
    assert D.replay(port(), v, server) == output(v)


@pytest.mark.parametrize("stated", [
    {"disk_bytes_per_s": "4400000000"},
    {"link_bytes_per_s": "50000000000"},
    {"disk_bytes_per_s": "4400000000", "link_bytes_per_s": "5/2", "disk_burst_bytes": "7", "link_burst_bytes": "0"},
    {"field": "p1024", "encode_squarings": 1, "disk_burst_bytes": "1/3"},
], ids=["disks", "links", "bursts", "encode"])
def test_two_skus_with_the_runs_statements_are_the_leans(server, stated) -> None:
    """Devices of two SKUs, each read at its own card's figures, with the disks' and links' ceilings and bursts
    stated or left to their defaults: one second of the disks, and one host link window where the links are
    charged."""
    v = inputs(log("gpu-a", "gpu-b", "gpu-a"), **stated)
    assert "refused" not in line.node_run_json(v)
    assert D.replay(port(), v, server) == output(v)


def test_the_links_burst_is_the_host_window_only_where_the_links_are_charged() -> None:
    charged = line.node_run_json(inputs(log("gpu-a"), link_bytes_per_s="1"))
    assert charged["rates"]["link_burst_bytes"] == str(HOST_CARD["link_window_bytes"])
    assert line.node_run_json(inputs(log("gpu-a")))["rates"]["link_burst_bytes"] == "0"


def broken() -> site.Registration:
    reg = log("gpu-a")
    return dataclasses.replace(reg, entries=(dataclasses.replace(reg.entries[0], previous="00" * 64),))


@pytest.mark.parametrize("v, why", [
    (inputs(broken()), "the Site's log is refused at entry 0: previous"),
    (inputs(log("gpu-a"), host="gpu0"), "gpu0 is no host the Site registers"),
    (inputs(log("gpu-a"), host="node9"), "node9 is no host the Site registers"),
    (inputs(log("gpu-a"), host="node1"), "host node1 holds no device the Site registers"),
    (inputs(log("gpu-a", "gpu-b"), gpus=GPUS[:1]), "no card figures for gpu-b"),
    (inputs(log("gpu-a", "gpu-b"), field="p512"),
     "the node states no squaring ceiling at p512, the record's codec's width"),
    (inputs(log("gpu-a"), disk_bytes_per_s="-1"), "disk_bytes_per_s is nonnegative"),
    (inputs(log("gpu-a"), link_burst_bytes="-1/2"), "link_burst_bytes is nonnegative"),
], ids=["log", "device", "unregistered", "no-device", "figures", "width", "disk", "link-burst"])
def test_each_refusal_is_the_leans_word_for_word(server, v, why) -> None:
    assert line.node_run_json(v) == {"refused": why}
    assert D.replay(port(), v, server) == ("output", D.canonical({"refused": why}))


def rational() -> st.SearchStrategy:
    return st.one_of(st.integers(0, 10 ** 12).map(str), st.fractions(0, 10 ** 12, max_denominator=10 ** 4).map(line.text))


@st.composite
def nodes(draw) -> dict:
    skus = draw(st.lists(st.sampled_from(("gpu-a", "gpu-b")), min_size=1, max_size=8))
    gpus = [{"sku": sku, "chip_bytes": draw(st.integers(0, 1 << 40)),
             "squarings_per_s": {"p1024": draw(rational()), **draw(st.dictionaries(st.just("p512"), rational()))}}
            for sku in ("gpu-a", "gpu-b")]
    optional = st.one_of(st.none(), rational())
    return inputs(log(*skus), host=draw(st.sampled_from(("node0", "node0", "node1", "gpu0"))), gpus=gpus,
                  host_card={"chip_bytes": draw(st.integers(0, 1 << 40)),
                             "link_window_bytes": draw(st.integers(0, 1 << 30))},
                  field=draw(st.sampled_from(("p1024", "p512"))), encode_squarings=draw(st.integers(0, 64)),
                  disk_bytes_per_s=draw(st.one_of(rational(), st.just("1/0"))), link_bytes_per_s=draw(st.sampled_from(("0", "1/7", "50000000000"))),
                  disk_burst_bytes=draw(optional), link_burst_bytes=draw(optional))


def test_node(server) -> None:
    """Nodes of one to eight devices at random figures and statements, on a host or not, with a malformed rate now and
    then, which both sides reject."""
    assert D.run(port(nodes()), examples=150, server=server).outputs
