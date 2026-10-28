"""A node's figures: each a sum over its devices' cards and its host's, exact whatever their order, and a node refused
while any figure it reads is a placeholder."""

from __future__ import annotations

import math

import pytest

from verity.hardware import load
from verity.hardware.node import VY_NEBIUS, Node, device_of, host_of, node


def test_a_vy_nebius_node_is_its_eight_cards_and_their_host() -> None:
    """R measured and at spec, S_chip, the SHA-512 rate and the squaring ceiling are the sums over the eight cards and
    the host, and the write ceilings are the bandwidth ceilings of one card and of host DRAM."""
    n = node(*VY_NEBIUS)
    card = load("rtx-pro-6000-server")
    gpu, host = (card["memory"], card["store"], card["hash"], card["squaring"]), card["host"]
    assert len(n.devices) == 8 and n.host.sku == "rtx-pro-6000-server"
    assert n.read_bytes_per_s == 8 * gpu[0]["read"]["value"] + host["dram_read"]["value"] == 12_918.8e9
    assert n.spec_bytes_per_s == 8 * gpu[0]["bandwidth"]["value"] + host["dram_bandwidth"]["value"]
    assert n.device_read_bytes_per_s == 8 * 1.539e12
    assert n.chip_bytes == 8 * gpu[1]["chip"]["value"] + host["chip"]["value"] == 3_105_652_736
    assert n.sha512_per_s == 8 * gpu[2]["sha512"]["value"] == 56e9
    assert dict(n.squarings_per_s) == {w: 8 * e["value"] for w, e in gpu[3].items()}
    assert (n.device_write_bytes_per_s, n.host_write_bytes_per_s) == (1.597568e12, 819.2e9)
    assert n.host.link_window_bytes == 6_291_456


def test_a_mixed_node_sums_exactly_whatever_the_order() -> None:
    """The sums are exact (`math.fsum`), so a node's figures don't depend on the order its devices are listed in; a
    width is in the squaring ceiling only where every device states it, and one device's write ceiling is the
    fastest's."""
    a, b = device_of("rtx-pro-6000-server"), host_of("rtx-pro-6000-server")
    slow = type(a)("slow", 1e-3, 3.0, 1, 0.1, {"p1024": 0.5})
    one, two = Node((a, slow, a), b), Node((slow, a, a), b)
    assert one.read_bytes_per_s == two.read_bytes_per_s == math.fsum([2 * a.read_bytes_per_s, 1e-3,
                                                                      b.read_bytes_per_s])
    assert dict(one.squarings_per_s) == {"p1024": math.fsum([2 * a.squarings_per_s["p1024"], 0.5])}
    assert one.device_write_bytes_per_s == a.spec_bytes_per_s
    with pytest.raises(ValueError, match="one or more devices"):
        Node((), b)


@pytest.mark.parametrize(("devices", "host", "refusal"), [
    (["h200-sxm-141gb"] * 8, "h200-sxm-141gb", "h200-sxm-141gb: memory.read is a placeholder"),
    (["rtx-pro-6000-server"], "h200-sxm-141gb", "h200-sxm-141gb: host.dram_read is a placeholder"),
])
def test_a_placeholder_refuses_the_node(devices: list[str], host: str, refusal: str) -> None:
    """The H200 nodes' reads, chip storage, SHA-512 and squaring rates are not measured, so no node of them is read
    until their card files carry measurements."""
    with pytest.raises(ValueError, match=refusal):
        node(devices, host)
