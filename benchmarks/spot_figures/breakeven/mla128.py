"""The 128k workload's sparse-MLA blocks a token over the census request's, by block type, for a `workload` card whose
sparse-MLA rows are cut into head blocks, or cut further by a `partition` card into each head block's scores unit and
values units.

A computed token at position t of its request, counted from 1, reads T = min(t, `TOPK`) keys at each (layer, rank)
row. Serving cuts the row at the split count its launch gives it (`sparse_mla_split.launch_num_splits`: a request's
prefill is one launch and each decode step a launch of one token, the census request's own shape) as
`sparse_mla_split.split_chain` does, which `units` counts: per head, each split's first visited key block at its key
count and its later ones at `BN` keys; at one split the head's end; above it each split's end in its column blocks
and, per head, the combine of each column block. A row of so few keys that the card holds it whole stays whole. A
key block's scores unit and each of its values units are the key block's unit, and a unit cut into column blocks
(`COLUMNS`) is `DV` over a block's columns of them, so each takes its unit's factor.

`volume` counts each block type's units a computed token over the serve plan's requests (`requests`: a long
request's prefix hit is not computed) and over the card's own request, and gives their ratio: the factor on the
card's blocks a token. The rule is held to the card: over the card's request it gives the card's count of every
sparse-MLA block type at one whole number of (layer, rank) rows, or the card is refused. A combine over n splits
has no type on a card that holds the census request's combine alone, and counts as n / m of the card's m-split
combine: its AND gates are linear in the splits.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
#: the serve plan's requests, a line each: `prompt` and `output` tokens, and on a long request that repeats an earlier
#: one's prefix the whole blocks of `PLAN_BLOCK` tokens the two share (`benchmarks/serve_tax/prefix_plan.py`)
PLAN = ROOT / "benchmarks/serve_tax/prefix_plan.jsonl"
PLAN_BLOCK = 64
#: the keys a row reads at most (the indexer's top-k width) and the H200's SMs, which set a launch's split counts
TOPK, SMS = 2048, 132
FAMILY = "SparseMla"
#: a unit by its block's prefix and the statics that tell one type of the kind from another: a key block whole, or its
#: scores unit and its values units (the partition card `head-scores-values`), each the key block's unit
KEY_BLOCK = (r"NVIS=(\d+),FIRST=(true|false)", lambda m: ("first" if m[2] == "true" else "later", int(m[1])))
KINDS = (("SparseMlaHeadBlock_v1{", *KEY_BLOCK),
         ("SparseMlaHeadScores_v1{", *KEY_BLOCK),
         ("SparseMlaHeadValues_v1{", *KEY_BLOCK),
         ("SparseMlaHeadEnd_v1{", "", lambda m: ("end", 0)),
         ("SparseMlaSplitEndCols_v1{", "", lambda m: ("cols", 0)),
         ("SparseMlaSplitEnd_v1{", "", lambda m: ("cols", 0)),
         ("SparseMlaCombine_v1{", r"NSPLIT=(\d+)", lambda m: ("combine", int(m[1]))),
         ("SparseMlaSplit_v1{", r"T=(\d+),NS=1,", lambda m: ("whole", int(m[1]))))
#: the kinds whose block is one of a unit's column blocks, and the static that gives the block's columns: a unit is
#: the row's `DV` columns over those of its blocks (`SparseMlaSplitEndCols_v1` names each column block a type, one a unit)
COLUMNS = (("SparseMlaHeadValues_v1{", "G"), ("SparseMlaHeadEnd_v1{", "DV"), ("SparseMlaSplitEnd_v1{", "DV"))


def glm53() -> dict:
    """GLM-5.3's sparse-MLA statics: its heads `NH`, a key block's keys `BN` and a row's output columns `DV`."""
    from verity.definitions.registry.sparse_mla import SPARSE_MLA_GLM53
    return SPARSE_MLA_GLM53


def requests(path: Path = PLAN) -> list[tuple[int, int, int]]:
    """The serve plan's requests as (prompt tokens, output tokens, the prompt's tokens read from the prefix cache)."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [(r["prompt"], r["output"], r.get("blocks", 0) * PLAN_BLOCK) for r in rows]


@lru_cache(maxsize=None)
def decode_splits(T: int) -> int:
    from verity.definitions.registry.sparse_mla_split import launch_num_splits
    return launch_num_splits([T], SMS, TOPK)[0]


def units(T: int, NS: int, heads: int, whole: frozenset = frozenset()) -> Counter:
    """The units of one row of T keys at NS splits, {(kind, parameter): count}: `first` and `later` key blocks by
    their keys, a head's `end`, a split's end in each column block (`cols`, counted once for all of them), the
    `combine` of a column block by its splits, or the `whole` row where T is in `whole` at one split."""
    from verity.definitions.registry.sparse_mla_split import combine_cols, split_blocks

    if NS == 1 and T in whole:
        return Counter({("whole", T): 1})
    S = glm53()
    BN, splits, out = S["BN"], split_blocks(T, NS, S["BN"]), Counter()
    for _b0, b1 in splits:
        out[("first", min(T, b1 * BN) - (b1 - 1) * BN)] += heads
    if later := heads * sum(b1 - b0 - 1 for b0, b1 in splits):
        out[("later", BN)] += later
    if NS == 1:
        out[("end", 0)] += heads
    else:
        out[("cols", 0)] += heads * len(splits)
        out[("combine", len(splits))] += heads * (S["DV"] // combine_cols(S["DV"]))
    return out


def rows_of(p: int, o: int, h: int = 0) -> Counter:
    """One request's rows a (layer, rank), {(keys, splits): count}: its prefill, positions h + 1 to p, one launch, and
    each of its o decode steps a launch of one token."""
    from verity.definitions.registry.sparse_mla_split import MAX_SPLIT_BATCH, launch_num_splits

    out = Counter()
    if p - h > MAX_SPLIT_BATCH:
        # nothing splits in a launch this large: every row at one split, counted by its keys
        for t in range(h + 1, min(p, TOPK) + 1):
            out[(t, 1)] += 1
        if p > max(h, TOPK):
            out[(TOPK, 1)] += p - max(h, TOPK)
    elif p > h:
        valid = [min(t, TOPK) for t in range(h + 1, p + 1)]
        out.update(zip(valid, launch_num_splits(valid, SMS, TOPK), strict=True))
    for t in range(p + 1, p + o + 1):
        T = min(t, TOPK)
        out[(T, decode_splits(T))] += 1
    return out


def count(reqs: list[tuple[int, int, int]], heads: int, whole: frozenset) -> tuple[Counter, int]:
    """The units of every row of `reqs` a (layer, rank), and their computed tokens."""
    rows, tokens = Counter(), 0
    for p, o, h in reqs:
        rows.update(rows_of(p, o, h))
        tokens += p - h + o
    out = Counter()
    for (T, NS), n in rows.items():
        for u, k in units(T, NS, heads, whole).items():
            out[u] += n * k
    return out, tokens


def unit_of(block: str) -> tuple | None:
    """The unit a sparse-MLA block type is, None for a block of another family."""
    if not block.startswith(FAMILY):
        return None
    for prefix, statics, unit in KINDS:
        if block.startswith(prefix) and (m := re.search(statics, block)):
            return unit(m)
    raise ValueError(f"`{block}`: a sparse-MLA block type the 128k lever has no unit for (mla128.KINDS)")


def columns(block: str) -> int:
    """How many blocks of `block`'s type make one of its unit: the row's columns over the block's for a kind cut into
    column blocks (`COLUMNS`), else one."""
    for prefix, static in COLUMNS:
        if block.startswith(prefix):
            return glm53()["DV"] // int(re.search(rf"[{{,]{static}=(\d+)", block)[1])
    return 1


def volume(card: dict, reqs: list[tuple[int, int, int]] | None = None, heads: int | None = None) -> dict[str, float]:
    """{block: factor} for every sparse-MLA block of the `workload` card's value `card`: the block's units a computed
    token over `reqs` (the serve plan's when not given) with `heads` heads a row (GLM-5.3's when not given), over its
    units a token of the card's own request."""
    nh = glm53()["NH"]
    blocks = {x["block"]: (u, x["per_token"]) for x in card["strata"] if (u := unit_of(x["block"])) is not None}
    whole = frozenset(u[1] for u, _n in blocks.values() if u[0] == "whole")
    own = [(card["request"]["prefill"], card["tokens"] - card["request"]["prefill"], 0)]
    census, tokens = count(own, nh, whole)
    if tokens != card["tokens"]:
        raise ValueError(f"the card's request computes {tokens} tokens, not its {card['tokens']}")
    held = {u for u, _n in blocks.values()}
    if lost := sorted(u for u in census if u not in held):
        raise ValueError(f"the card's own request has sparse-MLA units the card holds no block type for: {lost}; the "
                         "card is of another cut than `sparse_mla_split.split_chain`'s, which this lever counts")
    rows = {b: n * tokens / (census[u] * columns(b)) if census[u] else 0.0 for b, (u, n) in blocks.items()}
    most = Counter(round(r) for r in rows.values()).most_common(1)[0][0]
    if bad := sorted(b for b, r in rows.items() if abs(r - most) > 1e-6 * max(most, 1)):
        raise ValueError(f"the card's sparse-MLA counts are not its own request's at {nh} heads a row, cut as "
                         f"`sparse_mla_split.split_chain` cuts it: {len(bad)} of {len(rows)} block types differ, the "
                         f"first `{bad[0]}`. The lever counts the 128k workload by that rule, so it would misstate "
                         "this card's blocks")
    plan, computed = count(requests() if reqs is None else reqs, nh if heads is None else heads, whole)
    combines = {u[1]: b for b, (u, _n) in blocks.items() if u[0] == "combine"}
    target: Counter = Counter()
    for u, n in plan.items():
        if u[0] == "combine" and u[1] not in combines:
            if not combines:
                raise ValueError(f"the 128k workload combines {u[1]} splits and the card holds no combine")
            m = max(combines)
            target[("combine", m)] += n * u[1] / m
        elif u in held:
            target[u] += n
        else:
            raise ValueError(f"the 128k workload has a sparse-MLA unit the card holds no block type for: {u}")
    return {b: (target[u] / computed) / (census[u] / tokens) for b, (u, _n) in blocks.items()}
