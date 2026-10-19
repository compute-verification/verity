"""The workload Program's launch units: a MoE align launch that routes several requests' tokens, computed once.

`moe_align_block_size` sorts every pair of its launch, so at a launch shared by several requests each request's blocks and the
launch's block count read the other requests' routing.  A request Program built under its launch batch takes them as `schedule`
words (`CollectiveBus.launch_schedule`, its record's form `linked`: its pairs' blocks, then the block count).  The workload Program
computes each such launch once, `MoeAlignRanks{E,P,BM}` over every pair of the launch, a key per pair and the pair indices, then
`MoeAlignBlocks{P,BM}`, and wires its outputs into each member request's `schedule` argument at the member's slots, the way a request
Program feeds each sampled token to its next step.  The unit's router ids and keys are root inputs: `launch[e<e>.<j>].ids`, which the
link records bind to each member's router ids at its slots, and `launch[e<e>.<j>].key`, the GPU's sorted position of each pair (any
key gives a schedule the kernel can produce).

The units are planned from the declaration alone: at engine step e the members are the requests scheduled there in admission order
(`arrivals.admission_order`: by arrival, decodes before the step's prefills, the order vLLM's runner lays the launch out in), and a
member's slot is the pairs of the members before it.  The plan refuses a launch whose members disagree on it (tokens, experts, moe
block, top-k), whose members' pairs are not their declared rows, or whose pairs do not fill it: no two requests claim a slot, and the
counts sum.
"""
from __future__ import annotations

from verity.ml.circuits.refs import Coll, array_of, coll_concat
from verity.ml.circuits.types import Array, Value

#: the request Program parameter the linked words are part of (`CollectiveBus`'s `schedule` input)
SCHEDULE = "schedule"
I32 = Value(32)


def launch_units(requests) -> tuple[dict, ...]:
    """The shared align launches of `requests` (`workload.WorkloadRequest`, manifest order), by engine step e and the j-th linked
    align call of that step: {"e", "j", "E", "BM", "P", "members": ((request, call k, word offset, slot, pairs), ...)}.  See the module
    docstring for the rule and its refusals."""
    from verity_vllm.engine.arrivals import admission_order
    from verity_vllm.pipeline.launch_context import rows_at
    decl = [{"i": i, "arrive_step": r.arrive_step, "prompt_len": r.prompt_len, "max_tokens": r.cap} for i, r in enumerate(requests)]
    by_step: dict[int, dict[int, list[dict]]] = {}
    for q in admission_order(decl):
        r = requests[q["i"]]
        for rec in r.component.schedule:
            if rec.get("form") == "linked":
                by_step.setdefault(r.arrive_step + rec["step"], {}).setdefault(q["i"], []).append(rec)
    units = []
    for e in sorted(by_step):
        members = by_step[e]
        if len({len(v) for v in members.values()}) != 1:
            raise ValueError(f"engine step {e}: the requests run {sorted({len(v) for v in members.values()})} linked align calls; one "
                             f"launch per call needs every request in it")
        for j in range(len(next(iter(members.values())))):
            calls = [(i, members[i][j]) for i in members]
            launch = {(rec["launch_tokens"], rec["E"], rec["BM"], rec["shape"][1]) for _, rec in calls}
            if len(launch) != 1:
                raise ValueError(f"engine step {e}, linked align call {j}: the requests disagree on the launch (tokens, E, BM, top-k): "
                                 f"{sorted(launch)}")
            L, E, BM, TOPK = launch.pop()
            slot, rows = 0, []
            for i, rec in calls:
                own = rows_at(decl[i], e)
                if rec["pairs"] != own * TOPK:
                    raise ValueError(f"engine step {e}: request {requests[i].request_id}'s linked align call {rec['k']} routes "
                                     f"{rec['pairs']} pairs, not its declared {own} rows x {TOPK}")
                rows.append((i, rec["k"], rec["offset"], slot, rec["pairs"]))
                slot += rec["pairs"]
            if slot != L * TOPK:
                raise ValueError(f"engine step {e}, linked align call {j}: the requests' {slot} pairs do not fill the launch's {L} tokens x "
                                 f"{TOPK}: a request of the launch is missing, or two claim one slot")
            units.append({"e": e, "j": j, "E": E, "BM": BM, "P": L * TOPK, "members": tuple(rows)})
    return tuple(units)


def linked_segments(units) -> dict[int, list[tuple[int, int, int, int]]]:
    """Per request: (word offset, unit index, slot, pairs) of each of its linked calls, in word order."""
    out: dict[int, list] = {}
    for u_i, u in enumerate(units):
        for i, _k, off, slot, pairs in u["members"]:
            out.setdefault(i, []).append((off, u_i, slot, pairs))
    return {i: sorted(v) for i, v in out.items()}


def key_words(words: int, segments) -> int:
    """The words of a `schedule` parameter of `words` that are keys, the rest being its linked calls' blocks and block counts."""
    return words - sum(pairs + 1 for _off, _u, _slot, pairs in segments)


def launch_params(units) -> list[tuple[str, Array]]:
    """The units' root inputs: each launch's router ids and keys, one i32 per pair."""
    return [(f"launch[e{u['e']}.{u['j']}].{name}", Array(u["P"], I32)) for u in units for name in ("ids", "key")]


def unit_blocks(B, u: dict, ids: Coll, key: Coll) -> tuple[Coll, Coll]:
    """The unit's body nodes: the pair indices, `MoeAlignRanks{E,P,BM}` and `MoeAlignBlocks{P,BM}`; (blk, par)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.prims import const
    from verity_vllm.program.registry import marlin_moe as MM
    P, h = u["P"], f"launch[e{u['e']}.{u['j']}]"
    ix = array_of([B.call(const(32, p), name=f"{h}/i32[{p}]") for p in range(P)])
    rank, start = B.call(bind(MM.MoeAlignRanks, E=u["E"], P=P, BM=u["BM"]), ids, key, ix, name=f"{h}/rank")
    blk, par = B.call(bind(MM.MoeAlignBlocks, P=P, BM=u["BM"]), ids, rank, start, name=h)
    return blk, par


def schedule_arg(words: int, keys: Coll | None, segments, blocks: list[tuple[Coll, Coll]]) -> Coll:
    """A member request's `schedule` argument of `words`: its key words from `keys` (its root input) in order, and at each linked
    call its unit's blocks at its slots, then the unit's block count."""
    parts, cursor, taken = [], 0, 0
    for off, u_i, slot, pairs in segments:
        if off > cursor:
            parts.append(keys[taken:taken + off - cursor])
            taken += off - cursor
        blk, par = blocks[u_i]
        parts += [blk[slot:slot + pairs], par]
        cursor = off + pairs + 1
    if words > cursor:
        parts.append(keys[taken:taken + words - cursor])
    return coll_concat(Array(words, I32), parts)


__all__ = ["SCHEDULE", "key_words", "launch_params", "launch_units", "linked_segments", "schedule_arg", "unit_blocks"]
