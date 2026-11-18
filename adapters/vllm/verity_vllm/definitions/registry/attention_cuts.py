"""The `Attention` versions cut by key block, and a cut row's units: what the window circuit lowers a row's heads into
(`program.predict.window_lowering`) and what the census counts a row as (`pipeline.serving_templates.KEY_BLOCKS`)."""
from __future__ import annotations

from typing import Any

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind

from verity_vllm.definitions.registry.dead_free import AttentionV19
from verity_vllm.definitions.registry.fa3_check_inf import AttentionV20

#: the `Attention` versions cut by key block, each by the version of its function whose heads' steps are Calls: FA2's v18
#: by v19, FA3's v4 by v20
CUTS = {18: AttentionV19, 19: AttentionV19, 4: AttentionV20, 20: AttentionV20}


def heads(fn: Any, family: str, count: int) -> Any:
    """The one ``family`` Definition ``fn``'s body is ``count`` heads of, batched; ValueError where its body is anything else."""
    nodes = fn.body.nodes
    head = nodes[0].fn
    if head.name != family or any(m.form != "batch" or descriptor_id(m.fn) != descriptor_id(head) for m in nodes) \
            or sum(m.n for m in nodes) != count:
        raise ValueError(f"{fn.id}: its body is not {count} heads of one {family} Definition")
    return head


def units(fn: Any) -> list[tuple[Any, int]]:
    """Each Definition a head of the attention binding ``fn`` Calls at its version's cut (`CUTS`), in the order the head
    first Calls it, and how many of it the row's NH heads Call: a head's first visited key block, the blocks below it and
    its end. KeyError for a version with no cut, ValueError where a head's step is no Call."""
    NH = int(fn.bindings["NH"])
    head = heads(bind(CUTS[fn.version], **fn.bindings), "AttentionHead", NH)
    out: dict[str, list] = {}
    for n in head.body.nodes:
        if n.form != "call":
            raise ValueError(f"{head.id}: a step of its body is a {n.form}, not a Call")
        out.setdefault(descriptor_id(n.fn), [n.fn, 0])[1] += NH
    return [(d, k) for d, k in out.values()]


__all__ = ["CUTS", "heads", "units"]
