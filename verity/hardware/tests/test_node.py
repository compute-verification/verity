"""A node's figures: each a sum over its devices' GPU cards and its host card, exact whatever their order, and a node
refused while it reads a placeholder its caller doesn't name."""

from __future__ import annotations

import math

import pytest

from verity.hardware import load
from verity.hardware.node import VY_NEBIUS, NodeFigures, device_of, host_of, node


def test_a_vy_nebius_node_is_its_eight_cards_and_their_host() -> None:
    """R measured and at spec, S_chip, the SHA-512 rate and the squaring ceiling are the sums over the eight cards and
    the host, the write ceilings are one card's bandwidth and host DRAM's write, and the node carries the host DRAM
    placeholders it was read with."""
    n = node(*VY_NEBIUS)
    card, host = load("rtx-pro-6000-server"), load("xeon-6776p-2s")
    gpu = (card["memory"], card["store"], card["hash"], card["squaring"])
    assert len(n.devices) == 8 and n.host.sku == "xeon-6776p-2s"
    assert n.read_bytes_per_s == 8 * gpu[0]["read"]["value"] + host["memory"]["read"]["value"] == 12_918.8e9
    assert n.spec_bytes_per_s == 8 * gpu[0]["bandwidth"]["value"] + host["memory"]["bandwidth"]["value"]
    assert n.device_read_bytes_per_s == 8 * 1.539e12
    assert n.chip_bytes == 8 * gpu[1]["chip"]["value"] + host["store"]["chip"]["value"] == 3_105_652_736
    assert n.sha512_per_s == 8 * gpu[2]["sha512"]["value"] == 56e9
    assert dict(n.squarings_per_s) == {w: 8 * e["value"] for w, e in gpu[3].items()}
    assert (n.device_write_bytes_per_s, n.host_write_bytes_per_s) == (1.597568e12, 819.2e9)
    assert n.host.link_window_bytes == 6_291_456
    assert n.placeholders == ("xeon-6776p-2s memory.bandwidth", "xeon-6776p-2s memory.write")


def test_a_mixed_node_sums_exactly_whatever_the_order() -> None:
    """The sums are exact (`math.fsum`), so a node's figures don't depend on the order its devices are listed in; a
    width is in the squaring ceiling only where every device states it, and one device's write ceiling is the
    fastest's."""
    a, b = device_of("rtx-pro-6000-server"), host_of("xeon-6776p-2s", VY_NEBIUS[2])
    slow = type(a)("slow", 1e-3, 3.0, 1, 0.1, {"p1024": 0.5})
    one, two = NodeFigures((a, slow, a), b), NodeFigures((slow, a, a), b)
    assert one.read_bytes_per_s == two.read_bytes_per_s == math.fsum([2 * a.read_bytes_per_s, 1e-3,
                                                                      b.read_bytes_per_s])
    assert dict(one.squarings_per_s) == {"p1024": math.fsum([2 * a.squarings_per_s["p1024"], 0.5])}
    assert one.device_write_bytes_per_s == a.spec_bytes_per_s
    with pytest.raises(ValueError, match="one or more devices"):
        NodeFigures((), b)


@pytest.mark.parametrize(("devices", "host", "placeholders", "refusal"), [
    (["h200-sxm-141gb"] * 8, "xeon-platinum-8468-2s", (), "h200-sxm-141gb memory.read is a placeholder"),
    (["b200-sxm-180gb"], "xeon-6776p-2s", VY_NEBIUS[2], "b200-sxm-180gb memory.read is a placeholder"),
    (["rtx-pro-6000-server"], "xeon-platinum-8468-2s", (), "xeon-platinum-8468-2s memory.read is a placeholder"),
    (["rtx-pro-6000-server"], "xeon-6776p-2s", (), "xeon-6776p-2s memory.bandwidth is a placeholder"),
    (["rtx-pro-6000-server"], "xeon-6776p-2s", (*VY_NEBIUS[2], "xeon-6776p-2s memory.read"),
     r"\['xeon-6776p-2s memory.read'\] name no placeholder the node reads"),
    (["xeon-6776p-2s"], "xeon-6776p-2s", VY_NEBIUS[2], "xeon-6776p-2s is a host card"),
    (["rtx-pro-6000-server"], "rtx-pro-6000-server", (), "rtx-pro-6000-server is a gpu card"),
])
def test_a_placeholder_the_caller_doesnt_name_refuses_the_node(devices: list[str], host: str,
                                                              placeholders: tuple[str, ...], refusal: str) -> None:
    """The H200's and the B200's reads, chip storage, SHA-512 and squaring rates are not measured, so no node of them
    is read until their card files carry measurements; a placeholder is read only where the caller names it, and a
    name that is no placeholder the node reads is refused, as is a card of the wrong kind."""
    with pytest.raises(ValueError, match=refusal):
        node(devices, host, placeholders)
