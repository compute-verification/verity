"""A tensor-parallel rank's vocabulary range as a required value: the three committed interior values `Q_word_v1{16,32}` derives for
`EmbeddingShard_v1{VS,H,START}` (vLLM's `VocabParallelEmbedding` at tp_size > 1, one Call per token row).  Opt-in (`manifest build
--vocab-range`, the row's VOCAB_TAP): with it off nothing here runs, and no manifest, Program, root or verdict changes.

Every output word of the shard reads the shifted id and both range bits, so the partition commits, per token row and in the Call's
canonical gate order (`layout`): `local = tok - START` (I32Add, wrapping, 32 bits), `ge = START <= tok` and `le = tok <= START + VS - 1`
(I32Le, signed, 1 bit each).  `slots` reads them off the partition and refuses any other set.  They are interior to the Call:
`kernel_rows` adds one `vocab_range` identity per (step, embedding site), ranging over the site's shard Calls at that step, and the
vocabulary-range source (`acquire.sources.vocab_range_source`) acquires `<embedding module>/vocab_range`, int32 [rows, 3] =
(local, ge, le), computed from the rank's token ids exactly as `values` does.  No kernel changes: the values are integer functions of the
committed token id and the rank's static range.  The Match record holds no value for it.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from functools import lru_cache
from typing import Any, Callable, Mapping

FAMILY = "vocab_range"
MEMBER = "vocab_range"
SHARDS = frozenset({"EmbeddingShard_v1"})
#: (name, primitive) of each committed value, in the Call's canonical gate order
LAYOUT = (("local", "I32Add_v1"), ("ge", "I32Le_v1"), ("le", "I32Le_v1"))
WORDS = len(LAYOUT)

#: `Q_word_v1` committed interior words per Call that the source acquires (`word.check_calls`)
TAP_WORDS: dict[str, int] = {fam: WORDS for fam in SHARDS}

POLICY_TEXT = ("vocabulary range: per token row of EmbeddingShard_v1 the rank's committed range values -- tok - START (i32), START <= tok, "
               "tok <= START + VS - 1 (1 bit each) -- `<embedding module>/vocab_range` (family vocab_range, int32 [rows, 3])")


def _s32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & 0x80000000 else x


def values(tok: int, start: int, vs: int) -> tuple[int, int, int]:
    """(local, ge, le) of one token id, as signed int32 / 0-1: the circuit's I32Add (wrapping) and I32Le (signed) on the id's 32-bit word."""
    t = _s32(tok)
    return _s32(t - start), int(_s32(start) <= t), int(t <= _s32(start + vs - 1))


def shard_fn(VS: int, H: int, START: int) -> Any:
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import b1_tp2

    return bind(b1_tp2.EmbeddingShard, VS=VS, H=H, START=START)


@lru_cache(maxsize=None)
def slots(VS: int, H: int, START: int) -> tuple[int, ...]:
    """The shard Call's committed interior gates under `Q_word_v1{16,32}` (offsets in its canonical layout, in gate order): word i of a
    row is the value of gate `slots(...)[i]`.  Refuses a partition whose cut fails, that recomputes a gate, or that commits other values."""
    from verity.ml.definitions import word as W

    G = W.Graph(shard_fn(VS, H, START))
    R = W.units(G, 16, 32)
    if not R["cut"].ok or G.recomputed:
        raise ValueError(f"vocab range: the partition of EmbeddingShard_v1{{VS={VS},H={H},START={START}}} fails {R['cut'].codes} "
                         f"with {len(G.recomputed)} recomputed gates")
    owner, kinds = R["owner"], R["kinds"]
    crossing = set(G.src[owner[G.src] != owner[G.dst]].tolist())
    committed = sorted(g for g in crossing if kinds[owner[g]] == "committed")
    got = [G.prim_names[int(G.prim[g])] for g in committed]
    if got != [p for _n, p in LAYOUT]:
        raise ValueError(f"vocab range: EmbeddingShard_v1{{VS={VS},H={H},START={START}}} commits {got}, the source acquires "
                         f"{[p for _n, p in LAYOUT]}")
    return tuple(int(G.keep[g]) for g in committed)


def kernel_rows(pop: Any, add: Callable[..., None], impl: Callable[[str], str], aliases: Mapping[str, tuple[str, int]]) -> dict[str, Any]:
    """Add one `vocab_range` identity per (step, embedding site): its range is the site's shard Calls at that step, `WORDS` int32 each.
    `add` is `request_manifest`'s row writer."""
    groups: dict[tuple[int, str], dict[str, Any]] = {}
    for c in pop.P.calls:
        if c.family not in SHARDS:
            continue
        st = c.statics
        key = (pop.step_of(c.id), pop.program_path(c.id))
        g = groups.setdefault(key, {"VS": int(st["VS"]), "START": int(st["START"]), "rows": 0})
        if (g["VS"], g["START"]) != (int(st["VS"]), int(st["START"])):
            raise ValueError(f"vocab range: module {impl(key[1])} step {key[0]} runs two shards ({g}, {st})")
        g["rows"] += 1
    rows: Counter = Counter()
    for (step, mod), g in sorted(groups.items()):
        canon, k = aliases.get(mod, (mod, 0))
        add(FAMILY, step, k, impl(canon), MEMBER, 0, g["rows"] * WORDS, rows=g["rows"], width=WORDS, dtype="i32", spec="EmbeddingShard_v1",
            start=g["START"], shard_rows=g["VS"])
        rows[impl(canon)] += g["rows"]
    return {"words": WORDS * sum(rows.values()), "rows_by_site": dict(sorted(rows.items()))}


def header(q: dict[str, Any], stats: Mapping[str, Any] | None) -> dict[str, Any]:
    """The manifest's query header under the vocabulary-range policy (`q` itself when the policy is off, `stats` None)."""
    if stats is None:
        return q
    text = f"{q['query_text']}; {POLICY_TEXT}"
    out = dict(q, query_text=text, query_id_digest=hashlib.sha256(text.encode()).hexdigest(),
               required_families=[*q.get("required_families", []), FAMILY])
    out["vocab_range"] = {"family": FAMILY, "member": MEMBER, "layout": [n for n, _p in LAYOUT], **stats}
    return out


class Policy:
    """The vocabulary-range policy as `required.request_manifest(taps=(POLICY,))` applies it."""

    name = FAMILY
    tap_words = TAP_WORDS
    kernel_rows = staticmethod(kernel_rows)
    header = staticmethod(header)


POLICY = Policy()


def policy_of(manifest: Mapping[str, Any] | None) -> bool:
    q = (manifest or {}).get("query")
    return isinstance(q, dict) and isinstance(q.get("vocab_range"), dict)


def manifest_sites(manifest: Mapping[str, Any] | None) -> dict[str, dict[str, int]]:
    """op_path -> {"start", "shard_rows"} of every `vocab_range` identity of a manifest; a site named with two ranges is refused."""
    out: dict[str, dict[str, int]] = {}
    for r in (manifest or {}).get("identities") or []:
        if r.get("family") != FAMILY:
            continue
        g = {"start": int(r["start"]), "shard_rows": int(r["shard_rows"])}
        if out.setdefault(str(r["op_path"]), g) != g:
            raise ValueError(f"vocab range: {r['op_path']} is named with two ranges ({out[str(r['op_path'])]}, {g})")
    return out


__all__ = ["FAMILY", "MEMBER", "SHARDS", "LAYOUT", "WORDS", "TAP_WORDS", "POLICY_TEXT", "POLICY", "Policy", "values", "shard_fn", "slots", "kernel_rows",
           "header", "policy_of", "manifest_sites"]
