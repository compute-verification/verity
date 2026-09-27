"""The guarded max as a committed value: one f32 per (query row, head, key block after the row's first) of every attention Call, in word
3 of the attention hidden stream's ROW entry.  Opt-in (`manifest build --guarded-max`, the row's GUARDED_MAX_TAP): with it off nothing
here runs, and no manifest, Program, root or verdict changes.

In each key block after a row's first, the FA2 / FA3 online softmax reads `GuardNegInfZero(m)` (the running max, -inf read as 0) in two
places, the block's rescale factor and `max * scale`, so the no-recompute partition commits it.  The guarded-max tap builds
(`ops/pod_fa2_tap.sh` FA2_TAP_ROW_GUARD=1, `ops/pod_fa3_tap.sh` FA3_TAP_ROW_GUARD=1; `verity_tap_row3() == 1`) write it into ROW word 3
(`commit.hidden_stream.ROW_GUARD_FIELD`), which the default builds never write.  The stream's identities, layout and size stay the
record's; only that word's bytes differ, so the policy lives in the query header, and the Commit refuses a tap build whose word 3 does not
match it (`acquire.sources.hidden_source.make_hidden_source`).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable, Mapping

#: `verity_tap_row3()` of the tap build the policy needs (0 = ROW word 3 never written)
TAP_ROW3 = 1

POLICY_TEXT = ("guarded max: in every attention Call, GuardNegInfZero(m) of each key block after a query row's first is committed as word 3 "
               "of that (row, head, block)'s ROW entry in the attention hidden stream (the guarded-max tap build)")


def attention_words(P: Any) -> dict[str, int]:
    """The guarded maxes the query commits, by attention Definition family: NH x (ceil(T / BN) - 1) per Call, one per head and key block
    after the query row's first (`BN` = 64 where the Definition has no such static)."""
    from verity_vllm.query.required import FLASH_ATTENTION_LAUNCHES

    out: dict[str, int] = {}
    for c in P.calls:
        if c.family not in FLASH_ATTENTION_LAUNCHES:
            continue
        st = c.statics
        if "T" not in st or "NH" not in st:
            raise ValueError(f"guarded max: {c.definition} has no (T, NH) statics; tensor-parallel attention is not covered")
        out[c.family] = out.get(c.family, 0) + int(st["NH"]) * (-(-int(st["T"]) // int(st.get("BN", 64))) - 1)
    return dict(sorted(out.items()))


def header(q: dict[str, Any], words: Mapping[str, int]) -> dict[str, Any]:
    """The manifest's query header under the guarded-max policy; `words` is `attention_words`, summed over the request Programs."""
    text = f"{q['query_text']}; {POLICY_TEXT}"
    return dict(q, query_text=text, query_id_digest=hashlib.sha256(text.encode()).hexdigest(),
                guarded_max={"stream_family": q.get("attention_hidden_family"), "row_field": 3, "tap_row3": TAP_ROW3,
                             "words": sum(words.values()), "words_by_family": dict(sorted(words.items()))})


def merged(q: dict[str, Any], parts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """A global manifest's query header `q` (its first component's) with the guarded maxes of every component header in `parts` summed."""
    words: Counter = Counter()
    for p in parts:
        words.update(p["guarded_max"]["words_by_family"])
    return dict(q, guarded_max=dict(q["guarded_max"], words=sum(words.values()), words_by_family=dict(sorted(words.items()))))


def policy_of(manifest: Mapping[str, Any] | None) -> bool:
    """Whether a manifest was built under the guarded-max policy (its query header says so)."""
    q = (manifest or {}).get("query")
    return isinstance(q, dict) and isinstance(q.get("guarded_max"), dict)


__all__ = ["TAP_ROW3", "POLICY_TEXT", "attention_words", "header", "merged", "policy_of"]
