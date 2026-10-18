import pytest


def device_graph(blocks: int = 2) -> dict:
    """A device graph in `graph.Hardware.to_dict()`'s schema: per block a GPU and its HBM, a NIC whose uplink goes to the
    block's certifier, the certifiers and one outside host on a switch. Latencies: uplink 1, certifier to switch 2, host
    to switch 4, inside a block pending."""
    leaves, links = {}, {}

    def leaf(x, kind, block=None):
        leaves[x] = {"id": x, "kind": kind, "block": block}

    def link(e, a, b, beta, ell):
        links[e] = {"id": e, "ends": [a, b], "kind": "fabric", "beta": {"value": beta}, "ell": {"value": ell}}

    leaf("sw", "switch")
    for i in range(blocks):
        b = f"b{i}"
        for x, kind in (("gpu0", "gpu"), ("hbm0", "hbm"), ("nic0", "nic")):
            leaf(f"{b}.{x}", kind, b)
        leaf(f"{b}.cert", "certifier")
        link(f"{b}.hbm0.bus", f"{b}.gpu0", f"{b}.hbm0", 100, None)
        link(f"{b}.nic0.pcie", f"{b}.gpu0", f"{b}.nic0", 50, None)
        link(f"{b}.uplink", f"{b}.nic0", f"{b}.cert", 40, 1)
        link(f"{b}.cert.fabric", f"{b}.cert", "sw", 40, 2)
    leaf("ext0", "host")
    link("ext0.fabric", "ext0", "sw", 40, 4)
    return {"leaves": leaves, "links": links, "units": {}, "cert": [f"b{i}.cert" for i in range(blocks)],
            "bucket_ticks": 10}


@pytest.fixture
def graph():
    return device_graph
