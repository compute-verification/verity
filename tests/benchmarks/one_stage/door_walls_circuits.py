"""The resource-accounting circuits `door_walls.py` puts through the door, built by their protocols' own provers."""
from __future__ import annotations

from typing import Any


def pous_setup() -> Any:
    """PoUS's setup C at its test's point (`window.Setup(64, 189, 16, 4, 4)`), an `IF.Circuit`."""
    from verity.protocols.resource_accounting.memory.pous.prover import window as WIN
    return WIN.Setup(64, 189, 16, 4, 4).circuit()


def network_scan() -> tuple[Any, dict]:
    """`scan_zk.scan_root` over the scan's own templates (start, ingress, egress) at its test class and T = 4 buckets,
    and its template query."""
    from verity.protocols.resource_accounting.network.prover import scan_zk as sz
    cls, t = sz.TEST.cls, 4
    defs = sz._defs(cls)
    counts = sz.counts_of(sz.members_of(sz.TEMPLATES, t))
    root = sz.scan_root([(defs[k], counts[k], sz.FIXED[k]) for k in sz.TEMPLATES], t // cls.EVERY, cls.EVERY)
    return root, sz.PA.template_instances_query([defs[k] for k in sz.TEMPLATES])
